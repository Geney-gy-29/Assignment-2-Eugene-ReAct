import os
import threading
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from openai import APIError, APIConnectionError, RateLimitError

load_dotenv()

_client = None

# [A3-IMPROVEMENT] Default raised 256 -> 512. At 256 the model silently
# truncated completions before emitting "\nAnswer:", producing empty
# predictions in the Assignment-2 n=10 FEVER runs (fever_cot/cotsc/standard).
# Empty strings are then dropped from the CoT-SC vote, so truncation quietly
# shrank the effective sample size. Applied to baseline AND improved arms so
# they stay comparable.
DEFAULT_MAX_TOKENS = 512

# OpenRouter USD per token for z-ai/glm-5.2 (verified live against
# https://openrouter.ai/api/v1/models on 2026-07-28). Override via env if the
# model is changed.
_PRICE_IN = float(os.environ.get("OPENROUTER_PRICE_IN", "0.0000006692"))
_PRICE_OUT = float(os.environ.get("OPENROUTER_PRICE_OUT", "0.0000021032"))


class TokenMeter:
    """[A3-IMPROVEMENT] Process-wide token/cost accounting.

    Assignment 2 discarded `response.usage` entirely, so there was no data for
    the accuracy-vs-computational-cost trade-off analysis. This accumulates
    usage across every completion; run.py snapshots it per question via
    `delta()` so each result record carries its own cost.

    Thread-safe: generate() fans out concurrent calls for CoT-SC sampling, and
    run.py fans out concurrent questions.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.n_calls = 0

    def add(self, prompt_tokens: int, completion_tokens: int) -> None:
        with self._lock:
            self.prompt_tokens += prompt_tokens
            self.completion_tokens += completion_tokens
            self.n_calls += 1

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "n_calls": self.n_calls,
            }

    def delta(self, before: dict) -> dict:
        """Usage accrued since `before` (a prior snapshot), plus USD cost."""
        now = self.snapshot()
        d = {k: now[k] - before[k] for k in now}
        d["cost_usd"] = round(
            d["prompt_tokens"] * _PRICE_IN + d["completion_tokens"] * _PRICE_OUT, 8
        )
        return d


METER = TokenMeter()


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.environ["OPENROUTER_API_KEY"]
        _client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
    return _client


@retry(
    retry=retry_if_exception_type((RateLimitError, APIConnectionError, APIError)),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    stop=stop_after_attempt(5),
)
def _generate_one(
    prompt: str,
    stop: list[str] | None,
    temperature: float,
    max_tokens: int,
) -> str:
    """Issue a single completion request. The provider caps n=1 server-side
    regardless of the requested n, so CoT-SC sampling issues one request per
    sample (see generate())."""
    client = _get_client()
    model = os.environ.get("OPENROUTER_MODEL", "z-ai/glm-5.2")
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        stop=stop,
        temperature=temperature,
        n=1,
        max_tokens=max_tokens,
        # z-ai/glm-5.2 is a reasoning model whose hidden reasoning phase eats
        # max_tokens and can leave content=null with no error. Must stay off.
        extra_body={"reasoning": {"enabled": False}},
    )
    # [A3-IMPROVEMENT] Record usage for the cost analysis.
    usage = getattr(response, "usage", None)
    if usage is not None:
        METER.add(
            getattr(usage, "prompt_tokens", 0) or 0,
            getattr(usage, "completion_tokens", 0) or 0,
        )
    text = response.choices[0].message.content or ""
    if stop:
        text = _truncate_at_stop(text, stop)
    return text


def generate(
    prompt: str,
    stop: list[str] | None = None,
    temperature: float = 0.0,
    n: int = 1,
    max_tokens: int | None = None,
) -> list[str]:
    """Call the configured OpenRouter model, returning n completion strings."""
    if max_tokens is None:
        max_tokens = DEFAULT_MAX_TOKENS
    if n == 1:
        return [_generate_one(prompt, stop, temperature, max_tokens)]
    # The provider silently caps server-side n to 1, so fan out n independent
    # requests instead of trusting len(response.choices). Must stay.
    with ThreadPoolExecutor(max_workers=n) as pool:
        futures = [pool.submit(_generate_one, prompt, stop, temperature, max_tokens) for _ in range(n)]
        return [f.result() for f in futures]


def _truncate_at_stop(text: str, stop: list[str]) -> str:
    """Client-side fallback truncation, in case the provider doesn't honor `stop` server-side."""
    earliest = len(text)
    for s in stop:
        idx = text.find(s)
        if idx != -1:
            earliest = min(earliest, idx)
    return text[:earliest]
