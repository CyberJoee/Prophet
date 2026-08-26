"""
Shared LLM client.
Uses Groq (free tier) by default.

MODEL HISTORY: llama-3.3-70b-versatile was Groq's default model here, but
Groq deprecated it (announced 2026-06-17). Every call to this function
started 404ing on deprecation day and silently fell back to the mock
decision path in both strategy_agent.py and research_agent.py — meaning
NO real LLM decision was made for weeks; every trade came from the
templated mock path, and the journal agent's per-trade lessons also
stopped generating (same hardcoded model string, no fallback there).

Fix: model is now read from GROQ_MODEL (set it in Railway's variables so
future Groq deprecations are a one-line env change, not a silent multi-week
outage). Defaults to openai/gpt-oss-120b, Groq's official recommended
replacement for llama-3.3-70b-versatile as of this writing. Check
https://console.groq.com/docs/models if this 404s again — Groq deprecates
models with a few weeks' notice, and this default WILL go stale eventually.
"""
import os
import json
from dotenv import load_dotenv

load_dotenv()

DEFAULT_MODEL = "openai/gpt-oss-120b"

# Raised from 1500 on 2026-08-26. The research agent asks for up to 5
# opportunities each carrying a thesis and a risk note, and gpt-oss-120b is a
# reasoning model whose internal tokens count against this budget. At 1500 the
# JSON was truncated mid-string and the whole trading day was aborted.
DEFAULT_MAX_TOKENS = 3000


class LLMTruncated(RuntimeError):
    """The model hit its token budget before finishing the JSON."""


class LLMBadJSON(RuntimeError):
    """The model returned something that is not parseable JSON."""


def _strip_fences(raw: str) -> str:
    """Remove markdown code fences the model sometimes wraps JSON in."""
    raw = raw.strip()
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1] if len(parts) > 1 else raw
        if raw.startswith("json"):
            raw = raw[4:]
    return raw.strip()


def _looks_truncated(raw: str, err: Exception = None) -> bool:
    """
    Did this response get cut off mid-JSON?

    finish_reason is the reliable signal, but it is not always populated, so
    the shape of the parse error is used as a fallback. A truncated document
    fails with 'Unterminated string' or by running out of input — distinct
    from a model that simply emitted prose instead of JSON, which should NOT
    be retried at a larger budget because more tokens will not fix it.
    """
    if not raw:
        return False
    msg = str(err or "").lower()
    if "unterminated string" in msg or "expecting" in msg and "eof" in msg:
        return True
    # Balanced-delimiter check: a complete object ends closed.
    if raw.lstrip().startswith(("{", "[")) and not raw.rstrip().endswith(("}", "]")):
        return True
    return False


def call_llm(system_prompt: str, user_prompt: str,
             max_tokens: int = None, _allow_retry: bool = True) -> dict:
    """
    Call the LLM and return a parsed JSON dict.

    Raises on failure — callers are responsible for their own handling, but
    should ALERT (not silently substitute) since a failure here means every
    downstream decision this run would be templated rather than reasoned.

    Two distinct failure modes, deliberately given distinct exceptions because
    they need different responses:

      LLMTruncated  — ran out of output budget. Retryable, and retried once
                      automatically at double the budget.
      LLMBadJSON    — returned non-JSON. More tokens will not help; this is a
                      prompt or model problem.

    The previous version raised a bare json.JSONDecodeError for both, which
    surfaced as "Unterminated string starting at line 39" and sent the reader
    looking at GROQ_MODEL and the API key — neither of which was the problem.
    An accurate error is as important as a loud one.
    """
    from groq import Groq

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or api_key.startswith("your_"):
        raise ValueError("GROQ_API_KEY not set in environment variables")

    if max_tokens is None:
        max_tokens = int(os.getenv("LLM_MAX_TOKENS", DEFAULT_MAX_TOKENS))

    model = os.getenv("GROQ_MODEL", DEFAULT_MODEL)
    client = Groq(api_key=api_key)

    response = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=0.3,   # low temp = more consistent structured output
    )

    choice = response.choices[0]
    raw = (choice.message.content or "").strip()
    finish = getattr(choice, "finish_reason", None)

    def _retry(why: str):
        print(f"  [llm] {why} at max_tokens={max_tokens} — "
              f"retrying once at {max_tokens * 2}")
        return call_llm(system_prompt, user_prompt,
                        max_tokens=max_tokens * 2, _allow_retry=False)

    if finish == "length":
        if _allow_retry:
            return _retry("response hit the token budget")
        raise LLMTruncated(
            f"model stopped at max_tokens={max_tokens} (finish_reason=length) "
            f"even after retry; got {len(raw)} chars. The prompt or the "
            f"requested output is too large — raise LLM_MAX_TOKENS or ask for "
            f"fewer opportunities.")

    cleaned = _strip_fences(raw)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        if _looks_truncated(cleaned, e):
            if _allow_retry:
                return _retry("response looked truncated")
            raise LLMTruncated(
                f"response truncated mid-JSON even after retry at "
                f"max_tokens={max_tokens}: {e}. Ends with: "
                f"...{cleaned[-120:]!r}") from e
        raise LLMBadJSON(
            f"model returned unparseable JSON ({e}). More tokens will not "
            f"help. Starts with: {cleaned[:200]!r}") from e


def is_llm_available() -> bool:
    """Check if a valid Groq API key is configured."""
    key = os.getenv("GROQ_API_KEY", "")
    return bool(key) and not key.startswith("your_")
