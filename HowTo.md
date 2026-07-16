# HowTo — ReAct Paper Reproduction

This document explains how the deliverables in this repository were built and how to rerun the experiment.

---

## 1. AI models and tools used

| Model / Tool | Role |
|---|---|
| **Claude (Sonnet 5, via Claude Code CLI)** | Wrote all code (`react_repro/`, `analysis/`), debugged live issues, authored `AI_LOG.md`, `report/Baseline_Reproduction_Report.md`/`.pdf`, and this file. |
| **z-ai/glm-5.2** (via OpenRouter) | The LLM backbone under evaluation — the model whose Standard/CoT/Act/ReAct/CoT-SC/hybrid trajectories are measured against the paper's PaLM-540B numbers. |

No other LLMs were used. All paper-reference numbers (Table 1, Yao et al. 2023) were retrieved via web search of the source paper, not estimated.

---

## 2. Main workflow and prompting strategy

The build was driven interactively, one milestone at a time, gated before scaling:

1. **Source exemplars verbatim, don't hand-transcribe.** Prompts like *"pull the few-shot exemplars byte-exact from the paper's official repo (github.com/ysymyth/ReAct) instead of retyping them from a PDF"* — avoids transcription drift.
2. **Smoke-test the LLM client before building on top of it.** Before writing any agent logic, a live smoke test against `z-ai/glm-5.2` surfaced two silent-failure gotchas (hidden reasoning tokens eating `max_tokens`; server-side `n>1` sampling silently ignored) — fixed in `react_repro/llm.py` first, since every downstream method depends on it.
3. **Build once, parametrize for reuse.** HotpotQA's `WikiEnv`/`agent.py`/`strategies.py` were built first and gated at n=10; FEVER reused the same scaffolding via domain-aware `instruction`/`query_label` parameters rather than a parallel implementation.
4. **Gate at n=10 before scaling.** Each domain was run across all 7 methods (Standard, CoT, Act, ReAct, CoT-SC, ReAct→CoT-SC, CoT-SC→ReAct) at n=10 with zero errors as an explicit checkpoint before committing, per the plan's time-boxed schedule. n=100 scale-up is deferred (see limitations in the report).
5. **Verify by inspecting output, not by trusting a clean exit.** Several bugs (Wikipedia 403, nav-chrome leaking into scraped text, a method-key mismatch that silently dropped 3/7 methods from the charts) produced *no error* — caught only by reading the actual output (a scraped page, a generated CSV) rather than trusting "the script ran."
6. **Log every AI-assisted fix as it happens.** `AI_LOG.md` is append-only, updated immediately after each issue/fix, not reconstructed after the fact.

---

## 3. Environment and data setup

```powershell
cd C:\Users\User\Assignment-2-Eugene-ReAct
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
# copy .env.example to .env and fill in OPENROUTER_API_KEY and OPENROUTER_MODEL=z-ai/glm-5.2
```

Datasets (not hand-picked — sampled with the paper's own methodology):
- `data/hotpotqa_dev_sample.jsonl` — HotpotQA distractor dev split (HF Hub), `random.Random(233).shuffle(range(7405))`, first 100 taken.
- `data/fever_dev_sample.jsonl` — FEVER `paper_dev.jsonl` (fever.ai), same seed-233 shuffle methodology, first 100 taken.

Both are already committed to the repo — no manual download needed to rerun the gate.

---

## 4. Commands for running the experiment

```powershell
# single method/domain
python -m react_repro.run --domain hotpotqa --method react --data data/hotpotqa_dev_sample.jsonl --out results/raw/hotpotqa_react.jsonl --limit 10
python -m react_repro.run --domain fever --method cot_sc --n_sc 21 --data data/fever_dev_sample.jsonl --out results/raw/fever_cotsc.jsonl --limit 10

# --method is one of: standard, cot, cot_sc, act, react, react_cotsc, cotsc_react

# aggregate results -> summary CSV + comparison charts
python analysis/make_charts.py

# render the markdown report to PDF
python analysis/md_to_pdf.py

# unit tests
pytest
```

---

## 5. Expected outputs

- `results/raw/*.jsonl` — one record per example: `task_id, domain, method, used_method, backoff_triggered, question, gold, prediction, correct, n_steps, n_calls, trajectory_text`.
- `results/summary/accuracy_summary.csv` — one row per (domain, method): accuracy %, backoff trigger rate %.
- `analysis/figures/*.png` — 4 charts: HotpotQA paper-vs-repro, FEVER paper-vs-repro, cross-domain comparison, hybrid backoff rates.
- `report/Baseline_Reproduction_Report.md` / `.pdf` — the written analysis, embedding the above charts.

At the current n=10 gate, expect roughly: HotpotQA reproduction accuracy above the paper's PaLM-540B numbers across all 7 methods; FEVER reproduction accuracy below the paper across all 7 methods (see `report/Baseline_Reproduction_Report.md` §1 for the full table and discussion).

---

## 6. Dataset, model, or parameter changes vs. the paper

| Paper | This reproduction | Why |
|---|---|---|
| PaLM-540B | `z-ai/glm-5.2` (via OpenRouter) | PaLM-540B is not publicly accessible; GLM-5.2 is a comparably-scaled instruction-following model reachable via API. |
| HotpotQA + FEVER + ALFWorld + WebShop | HotpotQA + FEVER only | ALFWorld cut (no native Windows TextWorld support, would require a WSL2 fallback); WebShop cut (requires a local Flask server + 1.18M product index). Time/token-budget decision, documented in `AI_LOG.md`. |
| n=500-7405/condition | n=10 (gate) → n=100 (planned scale-up) | Reduced scale to fit the assignment deadline; deterministic prefix of the paper's own shuffle order, not an arbitrary resample. |
| CoT-SC n=21 samples | n=21 (unchanged) | Matched paper's self-consistency sample count directly. |

---

## 7. The bundled Agent Skill (`skills/`)

The reusable process behind this reproduction is packaged as an Agent Skill in [`skills/llm-paper-reproduction/`](skills/llm-paper-reproduction/SKILL.md) — see that file for the step-by-step, reapplicable to other ReAct-style paper reproductions.
