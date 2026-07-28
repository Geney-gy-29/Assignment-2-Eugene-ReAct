"""[A3-IMPROVEMENT] Generalized back-off triggers for the ReAct/CoT-SC hybrid.

Entry point modified: Yao et al. (ICLR 2023) section 3.3.

The paper's ReAct -> CoT-SC hybrid backs off on exactly one condition: ReAct
exhausted `max_steps` without emitting a `Finish[...]` action. That trigger is
a proxy for "the agent got stuck", and it only detects the *stuck* failure
mode. It is structurally blind to the opposite failure: the agent stopping
*early* with unwarranted confidence.

Evidence from the Assignment-2 n=10 reproduction (results/raw/*_n10.jsonl):

  - FEVER: 8/10 ReAct episodes terminated at step 2 -- one Search, then an
    immediate Finish. The paper's step-exhaustion trigger fired on NONE of
    them. Gold NOT ENOUGH INFO was predicted REFUTES 6/6 times: the model
    confidently rules on claims for which it retrieved no bearing evidence.
    FEVER ReAct scored 30% against the paper's reported 60.9%.
  - HotpotQA: every *failing* ReAct episode instead burned all 7 steps (one
    reissued Search[Hardley Flood] verbatim after already loading that page),
    so there the paper's trigger does fire -- confirming it tracks step count,
    not evidence quality.

This module decomposes the trigger into three signals over a completed ReAct
trajectory, so back-off can fire on premature confidence as well as on
exhaustion:

  S1 `exhausted`      -- the paper's original condition (control).
  S2 `thin_evidence`  -- fewer than TAU informative retrievals before Finish.
  S3 `unsupported`    -- an LLM verifier judges the answer not entailed by the
                         retrieved observations.

Full CGA (Confidence-Gated Adaptive back-off) = S1 or S2 or S3.

Each signal is a callable `(result: dict) -> bool` over the dict returned by
`agent.react()`, registered in TRIGGERS so run.py can select an arm by name.
"""

from collections import Counter
from math import log

from react_repro.llm import generate

# Minimum number of informative retrievals required before a Finish is
# treated as evidence-grounded. 2 mirrors HotpotQA's multi-hop structure
# (the paper's own tasks require composing at least two facts) and is the
# value ablated in the report.
TAU_EVIDENCE = 2

# Observation prefixes/values that carry no evidence. WikiEnv emits the first
# two verbatim (see envs/wiki_env.py::_search_live and ::lookup).
_UNINFORMATIVE_PREFIXES = (
    "could not find",
    "no more results",
    "invalid action:",
)


def is_informative(observation: str) -> bool:
    """True if an observation actually returned page content.

    A failed search ("Could not find X. Similar: [...]") or an exhausted
    lookup ("No more results.") tells the agent that its query missed -- it is
    a retrieval failure, not evidence. Counting those as evidence is precisely
    what lets a 1-search-then-Finish trajectory look well-grounded.
    """
    if not observation:
        return False
    text = observation.strip().lower()
    if not text:
        return False
    return not text.startswith(_UNINFORMATIVE_PREFIXES)


def n_informative(result: dict) -> int:
    return sum(1 for o in result.get("observations", []) if is_informative(o))


# --------------------------------------------------------------------------
# Signals
# --------------------------------------------------------------------------


def s1_exhausted(result: dict, **kwargs) -> bool:
    """The paper's original section 3.3 trigger: ReAct ran out of steps
    without committing to an answer."""
    # Prefer the explicit flag from agent.react(); fall back to the empty
    # answer string for any caller that predates it.
    if "exhausted" in result:
        return bool(result["exhausted"])
    return not result.get("answer")


def s2_thin_evidence(result: dict, tau: int = TAU_EVIDENCE, **kwargs) -> bool:
    """Fired when ReAct committed to an answer on fewer than `tau` informative
    retrievals -- the premature-confidence mode the paper's trigger misses."""
    if s1_exhausted(result):
        return False  # exhaustion is S1's job; keep the signals disjoint
    return n_informative(result) < tau


def s3_unsupported(result: dict, question: str = "", query_label: str = "Question", **kwargs) -> bool:
    """Fired when a verifier judges the committed answer not entailed by the
    evidence ReAct actually retrieved.

    Costs exactly one extra LLM call per question -- cheap relative to the
    21-call CoT-SC it gates. Deliberately shown ONLY the retrieved
    observations, not the model's own Thought steps, so it checks grounding
    rather than re-running the same reasoning that produced the answer.
    """
    if s1_exhausted(result):
        return False  # no answer to verify
    evidence = [o for o in result.get("observations", []) if is_informative(o)]
    if not evidence:
        return True  # committed with zero retrieved evidence
    evidence_text = "\n".join(f"- {o}" for o in evidence)
    prompt = (
        "You are checking whether a proposed answer is supported by retrieved evidence.\n"
        "Consider ONLY the evidence below. Do not use outside knowledge.\n\n"
        f"{query_label}: {question}\n"
        f"Proposed answer: {result.get('answer', '')}\n\n"
        "Retrieved evidence:\n"
        f"{evidence_text}\n\n"
        "Does the evidence above sufficiently support the proposed answer? "
        "Reply with exactly one word: SUFFICIENT or INSUFFICIENT.\n"
        "Verdict:"
    )
    verdict = generate(prompt, stop=["\n"], temperature=0.0, max_tokens=16)[0]
    return "INSUFFICIENT" in verdict.strip().upper()


# --------------------------------------------------------------------------
# Composite triggers (the ablation arms)
# --------------------------------------------------------------------------


def _any(*signals):
    def trigger(result: dict, **kwargs) -> bool:
        return any(sig(result, **kwargs) for sig in signals)

    return trigger


def _s2_tau(tau: int):
    def sig(result: dict, **kwargs) -> bool:
        kwargs.pop("tau", None)
        return s2_thin_evidence(result, tau=tau, **kwargs)

    return sig


TRIGGERS = {
    # A1: control -- the paper's section 3.3 condition, unchanged.
    "paper": s1_exhausted,
    # A2: + evidence thinness (no extra LLM call)
    "s1s2": _any(s1_exhausted, s2_thin_evidence),
    # A3: + verifier only
    "s1s3": _any(s1_exhausted, s3_unsupported),
    # A4: full Confidence-Gated Adaptive back-off
    "cga": _any(s1_exhausted, s2_thin_evidence, s3_unsupported),
    # A5: verifier alone -- is S1 still carrying weight, or subsumed?
    "s3only": s3_unsupported,
    # A6: CGA with tau=1 (back off only on ZERO informative evidence).
    # Added after the n=10 gate: at tau=2, S2 fired on 9/10 FEVER episodes,
    # because FEVER claims are frequently single-hop and genuinely resolvable
    # from one page. A trigger that fires on ~everything degenerates into
    # plain CoT-SC and wins nothing. tau=1 tests whether the useful part of
    # S2 is specifically the zero-evidence case.
    "cga_tau1": _any(s1_exhausted, _s2_tau(1), s3_unsupported),
}


def fired_signals(result: dict, question: str = "", query_label: str = "Question") -> dict:
    """Evaluate every signal independently, for the trigger-diagnostics
    analysis (fire rate, and precision/recall against the oracle 'ReAct was
    actually wrong' label). Kept separate from TRIGGERS so the diagnostic pass
    doesn't change what an arm actually did."""
    return {
        "s1_exhausted": s1_exhausted(result),
        "s2_thin_evidence": s2_thin_evidence(result),
        "s3_unsupported": s3_unsupported(result, question=question, query_label=query_label),
        "n_informative": n_informative(result),
    }


# --------------------------------------------------------------------------
# Reverse direction: CoT-SC -> ReAct
# --------------------------------------------------------------------------


def vote_entropy(answers: list[str]) -> float:
    """Normalized Shannon entropy of the CoT-SC vote distribution, in [0, 1].

    The paper gates CoT-SC -> ReAct on `majority_count < n/2`, which reads only
    the top bin. That cannot distinguish a clean two-way split (2 competing
    hypotheses) from genuine scatter across many distinct answers, yet both
    fall below n/2. Entropy uses the whole distribution.

    Normalized by log(n_samples), NOT log(n_distinct): the latter rescales
    every uniform vote to exactly 1.0, so a 2-way even split and a 10-way
    scatter would score identically -- destroying the discrimination this
    signal exists to provide.
    """
    non_empty = [a for a in answers if a]
    if not non_empty:
        return 1.0
    counts = Counter(non_empty)
    total = len(non_empty)
    if len(counts) <= 1 or total <= 1:
        return 0.0
    h = -sum((c / total) * log(c / total) for c in counts.values())
    return h / log(total)


# Entropy above which the CoT-SC vote is treated as too scattered to trust.
TAU_ENTROPY = 0.5


def sc_lowconf_paper(sc_result: dict, n: int) -> bool:
    """The paper's condition: majority answer occurs in fewer than n/2 samples."""
    return sc_result.get("majority_count", 0) < n / 2


def sc_lowconf_cga(sc_result: dict, n: int) -> bool:
    """Generalized: back off on a below-majority vote OR a high-entropy vote.

    Also treats a vote where many samples produced no parseable answer as
    low-confidence, which the bare majority threshold ignores.
    """
    if sc_lowconf_paper(sc_result, n):
        return True
    return vote_entropy(sc_result.get("all_answers", [])) > TAU_ENTROPY


SC_TRIGGERS = {
    "paper": sc_lowconf_paper,
    "cga": sc_lowconf_cga,
}
