---
name: llm-paper-reproduction
description: >-
  Reproduce an LLM-agent paper (e.g. ReAct-style prompting/tool-use papers) at
  reduced scale: build the environment/prompts/strategies, gate correctness at
  a small n before scaling, aggregate results against the paper's own published
  tables, and write a comparison report. Use when a user asks to "reproduce a
  paper", "build a baseline reproduction", "compare against Table X of the
  paper", or run a small-scale experimental replication of a published LLM
  method. Triggers: "reproduce this paper", "baseline reproduction",
  "gate at n=10 then scale up", "compare our results to the paper's table".
license: For coursework/personal use.
---

# LLM Paper Reproduction

## What this skill does

Turns a published LLM-agent paper into a runnable, reduced-scale reproduction:
source exemplars/data exactly as the paper did, implement the method(s) behind
a small CLI, gate correctness at low n before spending on a full run, then
aggregate results into a report that cites the paper's own numbers directly
(never estimated).

## When to use

- The deliverable is **working code that reproduces experiments from a paper**,
  not just a summary or blueprint of it (for that, see a paper-deconstruction
  skill instead).
- The paper defines multiple **methods/strategies** to compare (e.g. Standard /
  CoT / Act / ReAct) across one or more **domains/datasets**.
- Full-scale reproduction is infeasible (cost, time, access to the original
  model) — an explicitly reduced-scale reproduction is acceptable and should be
  labeled as such throughout.

## Core principles

1. **Gate before you scale.** Run every method/domain combination at a small n
   (e.g. 10) end-to-end with zero errors before committing to a larger run.
   Catches integration bugs (parsing, API quirks, domain mismatches) cheaply.
2. **Match the paper's own sampling methodology**, not an arbitrary resample —
   e.g. if the paper shuffles the full dev set with a fixed seed and takes a
   prefix, do exactly that; a reduced-scale subset should be a deterministic
   prefix of the paper's own order, not a fresh random draw.
3. **Never trust a clean exit as correctness.** Several real bugs in this class
   of pipeline produce no exception — an LLM API silently truncating output, a
   scraper returning nav-chrome instead of article text, a dict-key mismatch
   that silently filters out 3 of 7 methods from a chart. Verify by reading
   actual output (a generated file, a rendered figure), not by the absence of
   a stack trace.
4. **Transcribe reference numbers, never estimate them.** Any "paper says X%"
   value going into a chart or table must be copied from the paper's own
   results table (cite table/figure number), retrieved via reading the paper
   or a web search of it — not guessed from memory.
5. **Build once per domain, parametrize for reuse.** If the paper reuses one
   environment/agent loop across multiple domains (e.g. a Wikipedia-search
   environment for both QA and fact-verification), implement it once with
   domain-aware parameters (instruction text, query label, metric function)
   rather than duplicating the pipeline per domain.
6. **Log AI-assisted issues as you hit them**, not reconstructed afterward —
   an append-only `AI_LOG.md` with Prompt/Issue/Fix/Verification per entry.

## Workflow

### 1. Source exemplars and data exactly, not from memory
Pull few-shot prompts/exemplars byte-exact from the paper's official code
repo if one exists, rather than hand-transcribing from the PDF. Pull the
actual dataset split the paper used (not a lookalike), and if reproducing at
reduced scale, use the paper's own sampling procedure (fixed-seed shuffle +
prefix) so the subset is traceable to the paper's methodology.

### 2. Smoke-test the model client before building on it
Before writing any agent logic, run a live smoke test of basic generation
against the target model/provider. Check for silent-failure classes
specific to the provider (e.g. reasoning-model providers spending the token
budget on hidden CoT before emitting content; a provider silently capping
`n` in sampling requests to 1). Fix these in the client first — every
downstream method depends on it, so debugging it late means re-running
everything.

### 3. Build the core pipeline for one domain, gate at low n
Implement environment(s), the agent loop, and each method/strategy. Run all
of them at a small n (e.g. 10) with zero errors as an explicit gate before
committing further work. Manually inspect at least one full trajectory for
qualitative sanity (does the reasoning/tool-use actually make sense, even
when the final answer is wrong).

### 4. Extend to additional domains via parametrization
Add new domains by threading new parameters through the existing pipeline
(instruction text, label format, metric function) rather than forking it.
Gate each new domain at low n the same way.

### 5. Aggregate results and build comparison charts — `scripts/make_charts.py`
Aggregate raw per-example result files (e.g. JSONL) into a summary table
(accuracy per domain/method, any auxiliary rates like backoff-trigger
frequency). Build grouped bar charts: paper's reference values vs. the
reproduction's values, per method, per domain. **Double check dict/column
keys used to join reproduction data against paper reference data match
exactly** — a silent mismatch (e.g. `cot_sc` vs `cotsc`) will drop methods
from the chart with no error.

### 6. Write the report
Cover: paper + reproduction target; original vs. reproduced experimental
settings; AI tools/prompts used; a table/figure comparing results to the
paper; whether the paper's conclusion is approximately supported; limitations
and likely reasons for any differences; approximate token usage/cost/AI-time/
runtime; a link to the GitHub repo. Render to PDF — `scripts/md_to_pdf.py`
converts a Markdown report (with relative image links) to a self-contained
PDF via `markdown` + `xhtml2pdf`, resolving image paths to absolute
filesystem paths (xhtml2pdf on Windows cannot resolve `file://` URIs).

### 7. Verify
Re-open every generated figure and the final PDF/report to confirm labels,
legends, and values are correct before committing — script exit code 0 is
not sufficient evidence.

## Files in this skill

- `scripts/make_charts.py` — template for aggregating `results/raw/*.jsonl`
  into a summary CSV and paper-vs-reproduction bar charts. Edit `METHOD_ORDER`,
  `METHOD_LABELS`, and the `PAPER` reference-value dict per paper/table.
- `scripts/md_to_pdf.py` — converts a Markdown report (with relative image
  links) to PDF via `markdown` + `xhtml2pdf`, with Windows-safe absolute image
  path resolution.

## Dependencies

Python 3 with `pandas`, `matplotlib`, `markdown`, `xhtml2pdf`. An LLM API
client (e.g. OpenRouter) for the model under test. No GPU required.
