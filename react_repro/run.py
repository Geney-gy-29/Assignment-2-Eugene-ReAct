"""CLI driver: runs a chosen method over a sampled dataset and writes
structured JSONL results to results/raw/.

Usage:
  python -m react_repro.run --domain hotpotqa --method react --limit 10 \\
      --data data/hotpotqa_dev_sample.jsonl --out results/raw/hotpotqa_react.jsonl
"""

import argparse
import json
import os
import time

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


def run_one(domain: str, method: str, item: dict, n_sc: int, max_steps: int, temperature: float) -> dict:
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
    elif method == "react_cotsc":
        env = WikiEnv()
        kwargs = {"query_label": query_label}
        if react_instruction is not None:
            kwargs["instruction"] = react_instruction
        result = react_to_cotsc(question, react_ex, cot_ex, env, max_steps=max_steps, n=n_sc, temperature=temperature, **kwargs)
        used_method = result["method"]
        backoff_triggered = result["backoff_triggered"]
    elif method == "cotsc_react":
        env = WikiEnv()
        kwargs = {"query_label": query_label}
        if react_instruction is not None:
            kwargs["instruction"] = react_instruction
        result = cotsc_to_react(question, react_ex, cot_ex, env, max_steps=max_steps, n=n_sc, temperature=temperature, **kwargs)
        used_method = result["method"]
        backoff_triggered = result["backoff_triggered"]
    else:
        raise ValueError(f"Unknown method: {method}")

    prediction = result["answer"]
    correct = em(prediction, gold) if domain == "hotpotqa" else fever_acc(prediction, gold)

    return {
        "task_id": item.get("id", item.get("claim")),
        "domain": domain,
        "method": method,
        "used_method": used_method,
        "backoff_triggered": backoff_triggered,
        "question": question,
        "gold": gold,
        "prediction": prediction,
        "correct": correct,
        "n_steps": result.get("n_steps"),
        "n_calls": result.get("n_calls"),
        "trajectory_text": result.get("trajectory", ""),
    }


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
    args = parser.parse_args()

    data = _load_data(args.data)
    if args.limit:
        data = data[: args.limit]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    n_correct = 0
    with open(args.out, "w", encoding="utf-8") as out_f:
        for i, item in enumerate(data):
            start = time.time()
            try:
                record = run_one(args.domain, args.method, item, args.n_sc, args.max_steps, args.temperature)
            except Exception as e:
                record = {
                    "task_id": item.get("id", item.get("claim")),
                    "domain": args.domain,
                    "method": args.method,
                    "error": str(e),
                    "correct": 0,
                }
            elapsed = time.time() - start
            record["elapsed_sec"] = round(elapsed, 2)
            n_correct += record.get("correct", 0)
            out_f.write(json.dumps(record) + "\n")
            out_f.flush()
            print(f"[{i+1}/{len(data)}] correct={record.get('correct')} running_acc={n_correct/(i+1):.3f} ({elapsed:.1f}s)")

    print(f"Done. {n_correct}/{len(data)} = {n_correct/len(data):.3f}")


if __name__ == "__main__":
    main()
