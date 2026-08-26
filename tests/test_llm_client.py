"""
Tests for agents/llm_client.py response handling.

Written after 2026-08-26, when the morning pipeline aborted with

    [ALERT] LLM CALL FAILED: Unterminated string starting at line 39 column 20
    Morning pipeline aborted — fix GROQ_MODEL / API key

GROQ_MODEL and the API key were both fine. The model had run out of output
budget mid-JSON. The failure was loud but misdiagnosed, and it cost a full
trading day that a retry would have saved.

These tests drive the real call_llm() against a stubbed Groq client, so the
retry and the error classification are exercised without network access.

    python tests/test_llm_client.py
"""
import json
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["GROQ_API_KEY"] = "test-key-not-real"

failures = []


def check(label, cond, detail=""):
    print(("  PASS  " if cond else "  FAIL  ") + label +
          (f"   [{detail}]" if detail and not cond else ""))
    if not cond:
        failures.append(label)


# ── Stub the groq SDK ───────────────────────────────────────────────────────
class _Msg:
    def __init__(self, content):
        self.content = content


class _Choice:
    def __init__(self, content, finish_reason):
        self.message = _Msg(content)
        self.finish_reason = finish_reason


class _Resp:
    def __init__(self, content, finish_reason):
        self.choices = [_Choice(content, finish_reason)]


class _Completions:
    """
    Call log lives on _FakeGroq, not here: call_llm() constructs a fresh
    Groq() on every attempt including the retry, so per-instance state would
    reset between attempts and the script would replay from the start.
    """
    def create(self, **kw):
        _FakeGroq.calls.append(kw)
        i = min(len(_FakeGroq.calls) - 1, len(_FakeGroq.script) - 1)
        content, finish = _FakeGroq.script[i]
        return _Resp(content, finish)


class _FakeGroq:
    script = []
    calls = []

    def __init__(self, api_key=None):
        self.chat = types.SimpleNamespace(completions=_Completions())


groq_mod = types.ModuleType("groq")
groq_mod.Groq = _FakeGroq
sys.modules["groq"] = groq_mod

import agents.llm_client as L


def run(script):
    _FakeGroq.script = script
    _FakeGroq.calls = []
    return L.call_llm("sys", "user")


GOOD = json.dumps({"market_mood": "mixed", "opportunities": []})

# The literal shape of the 2026-08-26 failure: valid JSON cut off inside a
# string value.
TRUNCATED = '{"market_mood": "mixed", "opportunities": [{"symbol": "NVDA", "thesis": "momentum contin'

# ── 1. Happy path ───────────────────────────────────────────────────────────
print("\n=== 1. Normal response ===")
out = run([(GOOD, "stop")])
check("parses JSON", out["market_mood"] == "mixed", str(out))
check("only one API call", len(_FakeGroq.calls) == 1, str(len(_FakeGroq.calls)))

print("\n=== 2. Markdown fences are stripped ===")
out = run([("```json\n" + GOOD + "\n```", "stop")])
check("fenced JSON parses", out["market_mood"] == "mixed", str(out))

# ── 3. finish_reason=length triggers a retry at double budget ───────────────
print("\n=== 3. Token-budget truncation retries ===")
out = run([(TRUNCATED, "length"), (GOOD, "stop")])
check("retry produced a parsed result", out["market_mood"] == "mixed", str(out))
calls = _FakeGroq.calls
check("exactly two calls", len(calls) == 2, str(len(calls)))
check("second call doubled max_tokens",
      calls[1]["max_tokens"] == calls[0]["max_tokens"] * 2,
      f"{calls[0]['max_tokens']} -> {calls[1]['max_tokens']}")

# ── 4. Truncation without finish_reason is still detected ───────────────────
print("\n=== 4. Truncation detected from JSON shape alone ===")
out = run([(TRUNCATED, None), (GOOD, "stop")])
check("retried on unterminated JSON", out["market_mood"] == "mixed", str(out))
check("two calls", len(_FakeGroq.calls) == 2, str(len(_FakeGroq.calls)))

# ── 5. Persistent truncation raises LLMTruncated, not a JSON error ──────────
print("\n=== 5. Still truncated after retry ===")
try:
    run([(TRUNCATED, "length"), (TRUNCATED, "length")])
    check("raises", False, "no exception")
except L.LLMTruncated as e:
    check("raises LLMTruncated", True)
    check("message names the token budget", "max_tokens" in str(e), str(e)[:160])
    check("message suggests LLM_MAX_TOKENS", "LLM_MAX_TOKENS" in str(e), str(e)[:160])
except Exception as e:
    check("raises LLMTruncated", False, f"{type(e).__name__}: {e}")
check("did not retry more than once", len(_FakeGroq.calls) == 2,
      str(len(_FakeGroq.calls)))

# ── 6. Non-JSON prose is NOT retried — more tokens cannot fix it ────────────
print("\n=== 6. Unparseable prose is a different failure ===")
try:
    run([("I'm sorry, I can't help with that request.", "stop")])
    check("raises", False, "no exception")
except L.LLMBadJSON as e:
    check("raises LLMBadJSON", True)
    check("says more tokens will not help", "not" in str(e).lower(), str(e)[:160])
except Exception as e:
    check("raises LLMBadJSON", False, f"{type(e).__name__}: {e}")
check("no wasteful retry on prose", len(_FakeGroq.calls) == 1,
      str(len(_FakeGroq.calls)))

# ── 7. The two failures are distinguishable by type ─────────────────────────
print("\n=== 7. Distinct exception types ===")
check("LLMTruncated is not LLMBadJSON",
      not issubclass(L.LLMTruncated, L.LLMBadJSON))
check("both are RuntimeError",
      issubclass(L.LLMTruncated, RuntimeError) and issubclass(L.LLMBadJSON, RuntimeError))

# ── 8. Budget is configurable and larger than the value that failed ─────────
print("\n=== 8. Token budget ===")
check("default raised above the 1500 that failed", L.DEFAULT_MAX_TOKENS > 1500,
      str(L.DEFAULT_MAX_TOKENS))
os.environ["LLM_MAX_TOKENS"] = "4321"
run([(GOOD, "stop")])
check("LLM_MAX_TOKENS honoured", _FakeGroq.calls[0]["max_tokens"] == 4321,
      str(_FakeGroq.calls[0]["max_tokens"]))
del os.environ["LLM_MAX_TOKENS"]

# ── 9. Truncation heuristic ─────────────────────────────────────────────────
print("\n=== 9. _looks_truncated ===")
check("open object is truncated", L._looks_truncated('{"a": "b'))
check("closed object is not", not L._looks_truncated('{"a": "b"}'))
check("prose is not truncated", not L._looks_truncated("Sorry, I cannot."))
check("empty is not truncated", not L._looks_truncated(""))

print("\n" + ("ALL PASS" if not failures else f"{len(failures)} FAILURE(S): {failures}"))
sys.exit(1 if failures else 0)
