"""
Shared LLM client.
Uses Groq (free tier) by default — llama-3.3-70b-versatile.
Set GROQ_API_KEY in Railway variables.
"""
import os
import json
from dotenv import load_dotenv

load_dotenv()


def call_llm(system_prompt: str, user_prompt: str, max_tokens: int = 1500) -> dict:
    """
    Call the LLM and return a parsed JSON dict.
    Handles markdown fence stripping and JSON parsing.
    """
    from groq import Groq

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or api_key.startswith("your_"):
        raise ValueError("GROQ_API_KEY not set in environment variables")

    client = Groq(api_key=api_key)

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
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
