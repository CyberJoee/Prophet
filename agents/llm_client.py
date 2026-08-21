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


def call_llm(system_prompt: str, user_prompt: str, max_tokens: int = 1500) -> dict:
    """
    Call the LLM and return a parsed JSON dict.
    Handles markdown fence stripping and JSON parsing.
    Raises on failure — callers are responsible for their own mock fallback,
    but should also ALERT (not just silently substitute) since a 404 here
    means every downstream decision this run is templated, not reasoned.
    """
    from groq import Groq

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or api_key.startswith("your_"):
        raise ValueError("GROQ_API_KEY not set in environment variables")

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

    raw = response.choices[0].message.content.strip()

    # Strip accidental markdown fences
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1] if len(parts) > 1 else raw
        if raw.startswith("json"):
            raw = raw[4:]

    return json.loads(raw.strip())


def is_llm_available() -> bool:
    """Check if a valid Groq API key is configured."""
    key = os.getenv("GROQ_API_KEY", "")
    return bool(key) and not key.startswith("your_")
