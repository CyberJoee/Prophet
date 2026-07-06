"""
Options Flow Collector — positioning signals from option chains (yfinance).

What most retail bots never look at: where the options money is leaning.
Per symbol, per morning:

  cp_volume_ratio    call volume / put volume (nearest 2 expiries)
                     > 2.0 = heavy call bias, < 0.5 = heavy put bias
  unusual_contracts  count of contracts where volume > 3x open interest —
                     fresh positioning that did NOT exist yesterday
  unusual_call_bias  share of that unusual volume on the call side (0-1)
  atm_iv             at-the-money implied vol, nearest expiry —
                     the market's own estimate of imminent movement
  total_opt_volume   raw options volume (baseline builds in DB over time)

All lookups fail OPEN per-symbol: a yfinance hiccup yields no signal for
that symbol, never an error that blocks the pipeline.
"""
from typing import Optional

UNUSUAL_VOL_OI_MULT = 3.0
MIN_CONTRACT_VOLUME = 100      # ignore illiquid noise contracts


def collect_options_flow(symbol: str) -> Optional[dict]:
    """Best-effort options positioning snapshot. None on any failure."""
    try:
        import yfinance as yf
        t = yf.Ticker(symbol)
        expiries = t.options
        if not expiries:
            return None

        spot = None
        try:
            spot = float(t.fast_info["last_price"])
        except Exception:
            pass

        call_vol = put_vol = 0
        unusual = unusual_calls = 0
        atm_iv = None

        for expiry in expiries[:2]:              # nearest two expiries
            try:
                chain = t.option_chain(expiry)
            except Exception:
                continue

            for df, is_call in ((chain.calls, True), (chain.puts, False)):
                if df is None or df.empty:
                    continue
                vol = df["volume"].fillna(0)
                oi = df["openInterest"].fillna(0)
                side_vol = int(vol.sum())
                if is_call:
                    call_vol += side_vol
                else:
                    put_vol += side_vol

                mask = (vol > MIN_CONTRACT_VOLUME) & (vol > UNUSUAL_VOL_OI_MULT * oi)
                n_unusual = int(mask.sum())
                unusual += n_unusual
                if is_call:
                    unusual_calls += n_unusual

            # ATM IV from nearest expiry's calls
            if atm_iv is None and spot and chain.calls is not None and not chain.calls.empty:
                calls = chain.calls.dropna(subset=["impliedVolatility"])
                if not calls.empty:
                    idx = (calls["strike"] - spot).abs().idxmin()
                    atm_iv = round(float(calls.loc[idx, "impliedVolatility"]), 4)

        total = call_vol + put_vol
        if total < 500:                          # too thin to mean anything
            return None

        return {
            "cp_volume_ratio": round(call_vol / put_vol, 3) if put_vol else None,
            "call_volume": call_vol,
            "put_volume": put_vol,
            "total_opt_volume": total,
            "unusual_contracts": unusual,
            "unusual_call_bias": round(unusual_calls / unusual, 3) if unusual else None,
            "atm_iv": atm_iv,
        }
    except Exception as e:
        print(f"  [options_flow] {symbol}: failed open ({e})")
        return None


def describe(symbol: str, m: dict, baseline: Optional[dict] = None) -> str:
    """One-line human summary for the LLM briefing."""
    bits = []
    cp = m.get("cp_volume_ratio")
    if cp is not None:
        if cp >= 2.0:
            bits.append(f"heavy CALL bias (C/P {cp})")
        elif cp <= 0.5:
            bits.append(f"heavy PUT bias (C/P {cp})")
        else:
            bits.append(f"C/P ratio {cp}")
    if m.get("unusual_contracts"):
        side = ""
        ucb = m.get("unusual_call_bias")
        if ucb is not None:
            side = " mostly calls" if ucb > 0.7 else (" mostly puts" if ucb < 0.3 else "")
        bits.append(f"{m['unusual_contracts']} unusual-volume contracts{side}")
    if m.get("atm_iv") is not None:
        iv_note = f"ATM IV {m['atm_iv']:.0%}"
        if baseline and baseline.get("atm_iv"):
            chg = m["atm_iv"] / baseline["atm_iv"] - 1
            if abs(chg) > 0.15:
                iv_note += f" ({chg:+.0%} vs recent avg — market expects movement)"
        bits.append(iv_note)
    if baseline and baseline.get("total_opt_volume") and m.get("total_opt_volume"):
        rel = m["total_opt_volume"] / baseline["total_opt_volume"]
        if rel > 2.0:
            bits.append(f"options volume {rel:.1f}x its recent average")
    return f"{symbol}: " + "; ".join(bits) if bits else ""
