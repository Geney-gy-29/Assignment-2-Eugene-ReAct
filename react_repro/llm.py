import os
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from openai import APIError, APIConnectionError, RateLimitError

load_dotenv()

_client = None


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
        extra_body={"reasoning": {"enabled": False}},
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
    max_tokens: int = 256,
) -> list[str]:
    """Call the configured OpenRouter model, returning n completion strings."""
    if n == 1:
        return [_generate_one(prompt, stop, temperature, max_tokens)]
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
