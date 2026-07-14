"""Core ReAct agent loop for HotpotQA/FEVER (Wikipedia-backed QA), following
the official implementation's stop-sequence and retry logic
(github.com/ysymyth/ReAct, hotpotqa.ipynb::webthink)."""

from dataclasses import dataclass

from react_repro.llm import generate

INSTRUCTION = """Solve a question answering task with interleaving Thought, Action, Observation steps. Thought can reason about the current situation, and Action can be three types:
(1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.
(2) Lookup[keyword], which returns the next sentence containing keyword in the current passage.
(3) Finish[answer], which returns the answer and finishes the task.
Here are some examples.
"""


@dataclass
class Action:
    kind: str  # "search" | "lookup" | "finish"
    arg: str


def parse_action(action_text: str) -> Action:
    """Parse a `Kind[argument]` action string (case-insensitive kind)."""
    action_text = action_text.strip()
    open_idx = action_text.find("[")
    if not action_text.endswith("]") or open_idx == -1:
        raise ValueError(f"Malformed action: {action_text!r}")
    kind = action_text[:open_idx].strip().lower()
    arg = action_text[open_idx + 1 : -1]
    if kind not in ("search", "lookup", "finish"):
        raise ValueError(f"Unknown action kind: {kind!r} in {action_text!r}")
    return Action(kind=kind, arg=arg)


def react(
    question: str,
    exemplars: str,
    env,
    max_steps: int = 7,
    temperature: float = 0.0,
    instruction: str = INSTRUCTION,
    query_label: str = "Question",
) -> dict:
    """Run the ReAct loop for one HotpotQA/FEVER question against `env`
    (a WikiEnv instance). Returns dict with keys: answer, n_steps, n_calls,
    n_badcalls, trajectory.

    `instruction`/`query_label` default to the HotpotQA convention (a
    separate action-space preamble, "Question:" label). FEVER's official
    exemplars (fever.json) bake their own instruction line in, so callers
    pass instruction="" and query_label="Claim" for that domain."""
    prompt = instruction + exemplars + f"\n{query_label}: {question}\n"
    n_calls = 0
    n_badcalls = 0
    answer = None

    for i in range(1, max_steps + 1):
        n_calls += 1
        thought_action = generate(
            prompt + f"Thought {i}:",
            stop=[f"\nObservation {i}:"],
            temperature=temperature,
        )[0]
        try:
            thought, action_text = thought_action.strip().split(f"\nAction {i}: ", 1)
        except ValueError:
            n_badcalls += 1
            n_calls += 1
            thought = thought_action.strip().split("\n")[0]
            action_text = generate(
                prompt + f"Thought {i}: {thought}\nAction {i}:",
                stop=["\n"],
                temperature=temperature,
            )[0].strip()

        try:
            action = parse_action(action_text)
        except ValueError:
            obs = f"Invalid action: {action_text}"
            prompt += f"Thought {i}: {thought}\nAction {i}: {action_text}\nObservation {i}: {obs}\n"
            continue

        if action.kind == "finish":
            answer = action.arg
            step_str = f"Thought {i}: {thought}\nAction {i}: {action_text}\n"
            prompt += step_str
            break

        obs = env.step(action.kind, action.arg)
        obs = obs.replace("\n", "")
        step_str = f"Thought {i}: {thought}\nAction {i}: {action_text}\nObservation {i}: {obs}\n"
        prompt += step_str

    return {
        "answer": answer if answer is not None else "",
        "n_steps": i,
        "n_calls": n_calls,
        "n_badcalls": n_badcalls,
        "trajectory": prompt,
    }
