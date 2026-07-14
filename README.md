# ReAct Reproduction — Assignment 2

Reduced-scale reproduction of ["ReAct: Synergizing Reasoning and Acting in Language Models"](https://arxiv.org/abs/2210.03629) (Yao et al., ICLR 2023), built on the design in Assignment 1's blueprint report.

## Scope

- **Domains**: HotpotQA (multi-hop QA), FEVER (fact verification). ALFWorld and WebShop are out of scope (ALFWorld cut to conserve time/token budget for the deadline; WebShop requires a local Flask server + 1.18M product index).
- **Methods**: Standard, Chain-of-Thought (CoT), CoT Self-Consistency (CoT-SC), Act, ReAct, and two hybrid back-off strategies (ReAct→CoT-SC, CoT-SC→ReAct).
- **Scale**: n=50-100 examples per condition for HotpotQA/FEVER (paper used 500-7405). This is an explicitly reduced-scale reproduction, not a full-scale one — see `report/Baseline_Reproduction_Report.md` for the methodology note.

## AI tools used

| Tool | Role |
|---|---|
| Claude Code | Wrote `react_repro/`, `analysis/`, and this scaffolding |
| GLM-5.2 (via OpenRouter) | The LLM backbone being evaluated — i.e. the model whose ReAct/CoT/Act trajectories are measured |

See [AI_LOG.md](AI_LOG.md) for a detailed, timestamped log of AI-assisted coding and debugging steps.

## Setup

```powershell
cd C:\Users\User\Assignment-2-Eugene-ReAct
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
# copy .env.example to .env and fill in OPENROUTER_API_KEY
```

## Running

```powershell
python -m react_repro.run --domain hotpotqa --method react --data data/hotpotqa_dev_sample.jsonl --out results/raw/hotpotqa_react.jsonl --limit 50
python -m react_repro.run --domain hotpotqa --method cot_sc --n_sc 21 --data data/hotpotqa_dev_sample.jsonl --out results/raw/hotpotqa_cotsc.jsonl --limit 50
python -m react_repro.run --domain fever --method react_cotsc --data data/fever_dev_sample.jsonl --out results/raw/fever_reactcotsc.jsonl --limit 50
```

`--method` is one of `standard`, `cot`, `cot_sc`, `act`, `react`, `react_cotsc`, `cotsc_react`.

## Analysis

```powershell
python analysis/make_charts.py
```

Regenerates all comparison figures in `analysis/figures/` from `results/summary/*.csv`.

## Repo structure

```
react_repro/
├── llm.py          # OpenRouter client
├── prompts/        # few-shot exemplars, transcribed verbatim from the official ReAct repo
├── envs/           # WikiEnv (HotpotQA/FEVER)
├── agent.py        # the react() loop, action parsing
├── strategies.py   # standard/cot/cot_sc/act/react/back-off strategies
├── metrics.py       # em(), fever_acc()
└── run.py          # CLI entry point
```
