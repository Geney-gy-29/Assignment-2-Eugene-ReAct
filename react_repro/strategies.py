"""Baseline and reproduction strategies for HotpotQA/FEVER: Standard, CoT,
CoT-SC (majority vote over n samples), Act, ReAct, and the two hybrid
back-off strategies described in the paper (Yao et al., ICLR 2023, section
3.3):
  - ReAct -> CoT-SC: if ReAct exhausts max_steps without a Finish action,
    back off to CoT-SC.
  - CoT-SC -> ReAct: if the CoT-SC majority answer occurs in fewer than
    n/2 of the n samples, back off to ReAct.
"""

from collections import Counter

from react_repro.agent import INSTRUCTION, parse_action, react
from react_repro.llm import generate

# The official ReAct repo's prompts_naive.json ships webact_simple6 exemplars
# but no matching instruction preamble (only the ReAct one is published in
# hotpotqa.ipynb). This mirrors that instruction with the Thought step
# removed, since Act needs the same action-space description to parse
# correctly — documented as an assumption in AI_LOG.md.
ACT_INSTRUCTION = """Solve a question answering task with actions. Action can be three types:
(1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.
(2) Lookup[keyword], which returns the next sentence containing keyword in the current passage.
(3) Finish[answer], which returns the answer and finishes the task.
Here are some examples.
"""


def standard(question: str, temperature: float = 0.0) -> dict:
    """Direct question -> answer, no reasoning or actions."""
    prompt = f"Answer the following question with a short answer.\nQuestion: {question}\nAnswer:"
    text = generate(prompt, stop=["\n"], temperature=temperature)[0]
    return {"answer": text.strip(), "n_calls": 1, "trajectory": prompt + text}


def cot(question: str, exemplars: str, temperature: float = 0.0) -> tuple[str, str]:
    """One CoT completion: Thought + Answer. Returns (answer, full_completion_text)."""
    prompt = exemplars + f"Question: {question}\nThought:"
    text = generate(prompt, stop=["\nQuestion:"], temperature=temperature)[0]
    answer = ""
    if "\nAnswer:" in text:
        answer = text.split("\nAnswer:", 1)[1].strip().split("\n")[0]
    return answer, prompt + text


def cot_single(question: str, exemplars: str, temperature: float = 0.0) -> dict:
    answer, trajectory = cot(question, exemplars, temperature)
    return {"answer": answer, "n_calls": 1, "trajectory": trajectory}


def cot_sc(question: str, exemplars: str, n: int = 21, temperature: float = 0.7) -> dict:
    """CoT with self-consistency: sample n reasoning chains, majority-vote
    the final answers."""
    prompt = exemplars + f"Question: {question}\nThought:"
    completions = generate(prompt, stop=["\nQuestion:"], n=n, temperature=temperature)
    answers = []
    for text in completions:
        answer = ""
        if "\nAnswer:" in text:
            answer = text.split("\nAnswer:", 1)[1].strip().split("\n")[0]
        answers.append(answer)
    non_empty = [a for a in answers if a]
    counts = Counter(non_empty)
    if counts:
        majority_answer, majority_count = counts.most_common(1)[0]
    else:
        majority_answer, majority_count = "", 0
    return {
        "answer": majority_answer,
        "n_calls": n,
        "majority_count": majority_count,
        "n_samples": n,
        "all_answers": answers,
        "trajectory": prompt + f"\n[{n} samples, majority={majority_count}/{n}]",
    }


def act(question: str, exemplars: str, env, max_steps: int = 7, temperature: float = 0.0) -> dict:
    """Act-only: Action/Observation loop with no Thought steps, using the
    webact_simple6 exemplars prefixed with ACT_INSTRUCTION."""
    prompt = ACT_INSTRUCTION + exemplars + f"Question: {question}\n"
    n_calls = 0
    answer = None
    i = 1
    for i in range(1, max_steps + 1):
        n_calls += 1
        action_text = generate(prompt, stop=[f"\nObservation {i}:"], temperature=temperature)[0]
        action_text = action_text.strip()
        if action_text.startswith(f"Action {i}:"):
            action_text = action_text[len(f"Action {i}:"):].strip()
        else:
            action_text = action_text.split("\n")[0].strip()

        try:
            act_obj = parse_action(action_text)
        except ValueError:
            obs = f"Invalid action: {action_text}"
            prompt += f"Action {i}: {action_text}\nObservation {i}: {obs}\n"
            continue

        if act_obj.kind == "finish":
            answer = act_obj.arg
            prompt += f"Action {i}: {action_text}\n"
            break

        obs = env.step(act_obj.kind, act_obj.arg)
        obs = obs.replace("\n", "")
        prompt += f"Action {i}: {action_text}\nObservation {i}: {obs}\n"

    return {
        "answer": answer if answer is not None else "",
        "n_steps": i,
        "n_calls": n_calls,
        "trajectory": prompt,
    }


def react_strategy(question: str, exemplars: str, env, max_steps: int = 7, temperature: float = 0.0) -> dict:
    return react(question, exemplars, env, max_steps=max_steps, temperature=temperature)


def react_to_cotsc(
    question: str,
    react_exemplars: str,
    cot_exemplars: str,
    env,
    max_steps: int = 7,
    n: int = 21,
    temperature: float = 0.0,
) -> dict:
    """ReAct -> CoT-SC: if ReAct exhausts max_steps without a Finish action
    (i.e. no confident answer), back off to CoT-SC."""
    result = react(question, react_exemplars, env, max_steps=max_steps, temperature=temperature)
    if result["answer"]:
        result["backoff_triggered"] = False
        result["method"] = "react"
        return result
    sc_result = cot_sc(question, cot_exemplars, n=n, temperature=0.7)
    sc_result["backoff_triggered"] = True
    sc_result["method"] = "cot_sc"
    sc_result["n_calls"] += result["n_calls"]
    return sc_result


def cotsc_to_react(
    question: str,
    react_exemplars: str,
    cot_exemplars: str,
    env,
    max_steps: int = 7,
    n: int = 21,
    temperature: float = 0.0,
) -> dict:
    """CoT-SC -> ReAct: if the CoT-SC majority answer occurs in fewer than
    n/2 samples (low internal-knowledge confidence), back off to ReAct."""
    sc_result = cot_sc(question, cot_exemplars, n=n, temperature=0.7)
    if sc_result["majority_count"] >= n / 2:
        sc_result["backoff_triggered"] = False
        sc_result["method"] = "cot_sc"
        return sc_result
    react_result = react(question, react_exemplars, env, max_steps=max_steps, temperature=temperature)
    react_result["backoff_triggered"] = True
    react_result["method"] = "react"
    react_result["n_calls"] += sc_result["n_calls"]
    return react_result
