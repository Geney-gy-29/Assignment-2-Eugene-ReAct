"""CLI driver: runs a chosen method over a sampled dataset and writes
structured JSONL results to results/raw/.

Usage:
  python -m react_repro.run --domain hotpotqa --method react --limit 10 \\
      --data data/hotpotqa_dev_sample.jsonl --out results/raw/hotpotqa_react.jsonl

[A3-IMPROVEMENT] Adds --trigger (back-off ablation arm), --workers (parallel
questions), --max_tokens, and per-record token/cost accounting.
"""

import argparse
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from react_repro import llm
from react_repro.backoff import SC_TRIGGERS, TRIGGERS, fired_signals
from react_repro.envs.wiki_env import WikiEnv
from react_repro.metrics import em, fever_acc
from react_repro.strategies import (
    FEVER_ACT_INSTRUCTION,
    act,
    cot_single,
    cot_sc,
    cotsc_to_react,
    react_strategy,
    react_to_cotsc,
    standard,
)

PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "prompts")


def _load_prompt(name: str) -> str:
    with open(os.path.join(PROMPTS_DIR, name), "r", encoding="utf-8") as f:
        return f.read()


def _load_data(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def run_one(
    domain: str,
    method: str,
    item: dict,
    n_sc: int,
    max_steps: int,
    temperature: float,
    trigger_name: str = "paper",
    diagnose: bool = False,
) -> dict:
    question = item["question"] if domain == "hotpotqa" else item["claim"]
    gold = item["answer"] if domain == "hotpotqa" else item["label"]
    query_label = "Question" if domain == "hotpotqa" else "Claim"
    # FEVER's official exemplars (fever.json) bake their own instruction
    # line in; HotpotQA's exemplars need the separate INSTRUCTION preamble.
    react_instruction = "" if domain == "fever" else None  # None -> agent.py/strategies.py default
    act_instruction = FEVER_ACT_INSTRUCTION if domain == "fever" else None

    react_ex = _load_prompt(f"{domain}_react_examples.txt")
    cot_ex = _load_prompt(f"{domain}_cot_examples.txt")
    act_ex = _load_prompt(f"{domain}_act_examples.txt")

    backoff_triggered = False
    used_method = method
    diagnostics = None

    if method == "standard":
        result = standard(question, temperature=temperature, domain=domain)
    elif method == "cot":
        result = cot_single(question, cot_ex, temperature=temperature, query_label=query_label)
    elif method == "cot_sc":
        result = cot_sc(question, cot_ex, n=n_sc, temperature=0.7, query_label=query_label)
    elif method == "act":
        env = WikiEnv()
        kwargs = {"query_label": query_label}
        if act_instruction is not None:
            kwargs["instruction"] = act_instruction
        result = act(question, act_ex, env, max_steps=max_steps, temperature=temperature, **kwargs)
    elif method == "react":
        env = WikiEnv()
        kwargs = {"query_label": query_label}
        if react_instruction is not None:
            kwargs["instruction"] = react_instruction
        result = react_strategy(question, react_ex, env, max_steps=max_steps, temperature=temperature, **kwargs)
        # [A3-IMPROVEMENT] On the plain-ReAct arm, evaluate every signal
        # independently against the oracle correctness label. This is what
        # gives the report per-signal precision/recall without spending a
        # back-off on each one.
        if diagnose:
            diagnostics = fired_signals(result, question=question, query_label=query_label)
    elif method == "react_cotsc":
        env = WikiEnv()
        kwargs = {"query_label": query_label}
        if react_instruction is not None:
            kwargs["instruction"] = react_instruction
        result = react_to_cotsc(
            question, react_ex, cot_ex, env, max_steps=max_steps, n=n_sc,
            temperature=temperature, trigger=TRIGGERS[trigger_name], **kwargs
        )
        used_method = result["method"]
        backoff_triggered = result["backoff_triggered"]
    elif method == "cotsc_react":
        env = WikiEnv()
        kwargs = {"query_label": query_label}
        if react_instruction is not None:
            kwargs["instruction"] = react_instruction
        result = cotsc_to_react(
            question, react_ex, cot_ex, env, max_steps=max_steps, n=n_sc,
            temperature=temperature,
            sc_trigger=SC_TRIGGERS.get(trigger_name, SC_TRIGGERS["paper"]), **kwargs
        )
        used_method = result["method"]
        backoff_triggered = result["backoff_triggered"]
    else:
        raise ValueError(f"Unknown method: {method}")

    prediction = result["answer"]
    correct = em(prediction, gold) if domain == "hotpotqa" else fever_acc(prediction, gold)

    record = {
        "task_id": item.get("id", item.get("claim")),
        "domain": domain,
        "method": method,
        "trigger": trigger_name,
        "used_method": used_method,
        "backoff_triggered": backoff_triggered,
        "question": question,
        "gold": gold,
        "prediction": prediction,
        "correct": correct,
        "n_steps": result.get("n_steps"),
        "n_calls": result.get("n_calls"),
        "n_informative": sum(
            1 for o in result.get("observations", []) if o and not o.strip().lower().startswith(
                ("could not find", "no more results", "invalid action:")
            )
        ),
        "react_answer": result.get("react_answer"),
        "trajectory_text": result.get("trajectory", ""),
    }
    if diagnostics is not None:
        record["signals"] = diagnostics
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", required=True, choices=["hotpotqa", "fever"])
    parser.add_argument(
        "--method",
        required=True,
        choices=["standard", "cot", "cot_sc", "act", "react", "react_cotsc", "cotsc_react"],
    )
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=None, help="Cap number of examples processed")
    parser.add_argument("--n_sc", type=int, default=21, help="Number of CoT-SC samples")
    parser.add_argument("--max_steps", type=int, default=7)
    parser.add_argument("--temperature", type=float, default=0.0)
    # [A3-IMPROVEMENT] ablation arm selector; "paper" reproduces section 3.3.
    parser.add_argument("--trigger", default="paper", choices=sorted(TRIGGERS.keys()))
    parser.add_argument("--workers", type=int, default=1, help="Parallel questions")
    parser.add_argument("--max_tokens", type=int, default=llm.DEFAULT_MAX_TOKENS)
    parser.add_argument(
        "--diagnose", action="store_true",
        help="On --method react, evaluate all back-off signals per question",
    )
    args = parser.parse_args()

    llm.DEFAULT_MAX_TOKENS = args.max_tokens

    data = _load_data(args.data)
    if args.limit:
        data = data[: args.limit]

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    lock = threading.Lock()
    progress = {"done": 0, "correct": 0}

    def process(idx_item):
        idx, item = idx_item
        before = llm.METER.snapshot()
        start = time.time()
        try:
            record = run_one(
                args.domain, args.method, item, args.n_sc, args.max_steps,
                args.temperature, trigger_name=args.trigger, diagnose=args.diagnose,
            )
        except Exception as e:
            record = {
                "task_id": item.get("id", item.get("claim")),
                "domain": args.domain,
                "method": args.method,
                "trigger": args.trigger,
                "error": f"{type(e).__name__}: {e}",
                "correct": 0,
            }
        record["elapsed_sec"] = round(time.time() - start, 2)
        record["index"] = idx
        record.update(llm.METER.delta(before))
        with lock:
            progress["done"] += 1
            progress["correct"] += record.get("correct", 0)
            d, c = progress["done"], progress["correct"]
            print(f"[{d}/{len(data)}] correct={record.get('correct')} "
                  f"running_acc={c/d:.3f} ({record['elapsed_sec']:.1f}s)", flush=True)
        return record

    if args.workers > 1:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            records = list(pool.map(process, enumerate(data)))
    else:
        records = [process(x) for x in enumerate(data)]

    # Written in input order regardless of completion order, so arms stay
    # paired item-by-item for the McNemar test.
    records.sort(key=lambda r: r["index"])
    with open(args.out, "w", encoding="utf-8") as out_f:
        for record in records:
            out_f.write(json.dumps(record) + "\n")

    n_correct = sum(r.get("correct", 0) for r in records)
    n_errors = sum(1 for r in records if "error" in r)
    total_cost = sum(r.get("cost_usd", 0) for r in records)
    print(f"Done. {n_correct}/{len(data)} = {n_correct/len(data):.3f} | "
          f"errors={n_errors} | cost=${total_cost:.4f}")


if __name__ == "__main__":
    main()
