# Submission — ReAct Paper Reproduction (Eugene)

## GitHub repository

**https://github.com/Geney-gy-29/Assignment-2-Eugene-ReAct**

Clone with:

```powershell
git clone https://github.com/Geney-gy-29/Assignment-2-Eugene-ReAct.git
```

## Deliverables

| Deliverable | Location |
|---|---|
| Reproduction report (Markdown) | [report/Baseline_Reproduction_Report.md](report/Baseline_Reproduction_Report.md) |
| Reproduction report (PDF) | [report/Baseline_Reproduction_Report.pdf](report/Baseline_Reproduction_Report.pdf) |
| ACM-format report | [report/acm/](report/acm/) |
| Presentation deck (source) | [slides/slides.md](slides/slides.md) |
| Presentation deck (PDF / PPTX) | [slides/build/slides.pdf](slides/build/slides.pdf) · [slides/build/slides.pptx](slides/build/slides.pptx) |
| How it was built / how to rerun | [Assignment2-Eugene/HowTo.md](Assignment2-Eugene/HowTo.md) |
| AI-assisted coding log | [AI_LOG.md](AI_LOG.md) |
| Reusable Agent Skill | [Assignment2-Eugene/skills/llm-paper-reproduction/SKILL.md](Assignment2-Eugene/skills/llm-paper-reproduction/SKILL.md) |
| Packaged submission bundle | [Assignment2-Eugene.zip](Assignment2-Eugene.zip) |

## Code and results

| What | Location |
|---|---|
| Agent, environment, strategies, LLM client | [react_repro/](react_repro/) |
| Aggregation and chart scripts | [analysis/](analysis/) |
| Dataset samples (HotpotQA, FEVER) | [data/](data/) |
| Raw trajectories and summary CSV | [results/](results/) |
| Figures used in the report and deck | [analysis/figures/](analysis/figures/) · [slides/assets/](slides/assets/) |

## Summary

Reduced-scale reproduction of ["ReAct: Synergizing Reasoning and Acting in Language Models"](https://arxiv.org/abs/2210.03629) (Yao et al., ICLR 2023) on HotpotQA and FEVER, across 7 methods (Standard, CoT, CoT-SC, Act, ReAct, ReAct→CoT-SC, CoT-SC→ReAct), using `z-ai/glm-5.2` via OpenRouter in place of the paper's PaLM-540B. Assignment 3 adds a Confidence-Gated Adaptive Back-off extension to the paper's §3.3 hybrid.

See [README.md](README.md) for scope and setup, and [Assignment2-Eugene/HowTo.md](Assignment2-Eugene/HowTo.md) for the full rerun instructions.
