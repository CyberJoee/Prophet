"""
Geometry sweep — does ANY stop/target/time-exit combination have an edge?

THE MOTIVATION
--------------
Live and backtested both agree on the exit distribution: roughly 65% of
trades drift to the EOD close, ~27% stop out, and only ~7% ever reach
target. That is a geometry mismatch, not bad luck — a 1.0x ATR target is
rarely achievable in the hours between a 9:45 entry and the close, while a
0.5x ATR stop is close enough to get hit by noise.

This module tests whether a different (stop, target, time-exit) triple does
better, using the real execution path: BacktestEngineV2 driving the actual
execution/sizing.py, with spread and slippage charged.

THE DISCIPLINE THAT MAKES THE ANSWER MEAN ANYTHING
--------------------------------------------------
Searching a grid guarantees a winner. With 30 combinations on pure noise,
the best one will look good — that is what "best of 30" means, not evidence.
Three safeguards, all mandatory:

1. FIT / VALIDATE SPLIT. Sessions are split chronologically. Combinations are
   ranked on the FIT window only. The VALIDATE window is scored but never
   ranked on, and never used to choose anything.

2. THE FULL GRID IS REPORTED ON BOTH WINDOWS, with the fit-vs-validate rank
   correlation printed. Near zero means fit ranking carried no information and
   any winner is noise.

   BUT A POSITIVE RANK CORRELATION IS NOT EVIDENCE OF EDGE. Measured on
   structureless synthetic data this sweep still returns rho = +0.61, because
   ranking is dominated by a deterministic structural effect: costs are
   charged per side in dollars, so in R terms cost drag scales inversely with
   stop width. Wider stops therefore rank higher in EVERY window, edge or no
   edge. Rho is a necessary condition, never a sufficient one — the
   out-of-sample lower bound below is what actually decides.

3. SYNTHETIC NO-EDGE CONTROL. run_sweep() can be pointed at structureless
   random-walk data, where the correct answer is "no combination has an
   edge." If the sweep reports a winner there, the sweep itself is broken.
   This is the same bar engine_v2 had to clear before it was trusted.

Nothing here decides anything. It reports, with the uncertainty attached.
Promoting a geometry into execution/sizing.py is a human decision, and one
that should not be made on a validate-window expectancy whose standard error
covers zero.
"""
import math
import statistics
import time
from dataclasses import dataclass
from typing import Optional

from backtesting.engine_v2 import BacktestEngineV2, BacktestConfig


# Deliberately modest. Every extra point multiplies the multiple-comparisons
# problem; a 200-cell grid on a few hundred trades is self-deception.
DEFAULT_STOPS   = (0.5, 0.75, 1.0, 1.5)
DEFAULT_TARGETS = (0.5, 0.75, 1.0, 1.5)
DEFAULT_TIME_EXITS = (None, 60, 120)     # None = hold to EOD


@dataclass
class CellResult:
    stop_mult: float
    target_mult: float
    time_exit: Optional[int]
    fit: dict
    validate: dict

    @property
    def label(self) -> str:
        te = "eod" if self.time_exit is None else f"{self.time_exit}m"
        return f"stop{self.stop_mult:g}/tgt{self.target_mult:g}/{te}"


def _split_sessions(bars_by_symbol: dict, fit_frac: float = 0.6):
    """
    Chronological split. Returns (fit_bars, validate_bars, fit_days, val_days).

    Chronological, never random: a random split would leak future sessions
    into the fit window, which is the exact failure the honest backtest was
    built to avoid.
    """
    days = sorted({b["timestamp"].date()
                   for bars in bars_by_symbol.values() for b in bars})
    if len(days) < 40:
        raise ValueError(
            f"only {len(days)} sessions available — need at least 40 to split "
            "into a fit and validate window with enough trades in each")
    cut = days[int(len(days) * fit_frac)]

    fit, val = {}, {}
    for sym, bars in bars_by_symbol.items():
        fit[sym] = [b for b in bars if b["timestamp"].date() < cut]
        val[sym] = [b for b in bars if b["timestamp"].date() >= cut]
    return fit, val, [d for d in days if d < cut], [d for d in days if d >= cut]


def _run_cell(bars, stop_mult, target_mult, time_exit, base: BacktestConfig) -> dict:
    cfg = BacktestConfig(
        spread_bps=base.spread_bps,
        slippage_bps=base.slippage_bps,
        starting_equity=base.starting_equity,
        use_regime_gate=base.use_regime_gate,
        setups=base.setups,
        stop_mult=stop_mult,
        target_mult=target_mult,
        time_exit_minutes=time_exit,
    )
    return BacktestEngineV2(bars, cfg).run()["overall"]


def _spearman(xs: list[float], ys: list[float]) -> Optional[float]:
    """Rank correlation. Returns None when it would be meaningless."""
    n = len(xs)
    if n < 3:
        return None

    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):          # average ties
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = rank(xs), rank(ys)
    mx, my = statistics.mean(rx), statistics.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return round(num / den, 3) if den else None


def run_sweep(bars_by_symbol: dict,
              stops=DEFAULT_STOPS,
              targets=DEFAULT_TARGETS,
              time_exits=DEFAULT_TIME_EXITS,
              fit_frac: float = 0.6,
              min_trades: int = 30,
              base: BacktestConfig = None,
              progress: bool = True) -> dict:
    """
    Run the grid over fit and validate windows.

    min_trades is the floor below which a cell's expectancy is not reported as
    meaningful. Cells thinner than this are still run and shown, but are
    excluded from ranking and from any winner.
    """
    base = base or BacktestConfig()
    fit_bars, val_bars, fit_days, val_days = _split_sessions(bars_by_symbol, fit_frac)

    combos = [(sm, tm, te) for sm in stops for tm in targets for te in time_exits]
    total = len(combos)
    if progress:
        print(f"Sweeping {total} cells "
              f"({len(fit_days)} fit / {len(val_days)} validate sessions)...",
              flush=True)

    cells: list[CellResult] = []
    t_start = time.time()
    for i, (sm, tm, te) in enumerate(combos, start=1):
        cell = CellResult(
            stop_mult=sm, target_mult=tm, time_exit=te,
            fit=_run_cell(fit_bars, sm, tm, te, base),
            validate=_run_cell(val_bars, sm, tm, te, base),
        )
        cells.append(cell)
        if progress:
            elapsed = time.time() - t_start
            eta = elapsed / i * (total - i)
            fx = cell.fit.get("expectancy_r")
            vx = cell.validate.get("expectancy_r")
            print(f"  [{i:3d}/{total}] {cell.label:22s} "
                  f"fit={fx:+.3f}R " if fx is not None else
                  f"  [{i:3d}/{total}] {cell.label:22s} fit=  n/a  ", end="")
            print(f"val={vx:+.3f}R " if vx is not None else "val=  n/a  ", end="")
            print(f"| {elapsed:5.0f}s elapsed, ~{eta:.0f}s left", flush=True)

    eligible = [c for c in cells
                if (c.fit.get("trades") or 0) >= min_trades
                and (c.validate.get("trades") or 0) >= min_trades]

    ranked = sorted(eligible, key=lambda c: c.fit["expectancy_r"], reverse=True)
    winner = ranked[0] if ranked else None

    rho = None
    if len(eligible) >= 3:
        rho = _spearman([c.fit["expectancy_r"] for c in eligible],
                        [c.validate["expectancy_r"] for c in eligible])

    verdict, survived = _judge(winner, rho, eligible)

    return {
        "cells": cells,
        "eligible": eligible,
        "ranked": ranked,
        "winner": winner,
        "survived": survived,
        "rank_correlation": rho,
        "verdict": verdict,
        "fit_sessions": len(fit_days),
        "validate_sessions": len(val_days),
        "fit_range": (str(fit_days[0]), str(fit_days[-1])) if fit_days else None,
        "validate_range": (str(val_days[0]), str(val_days[-1])) if val_days else None,
        "min_trades": min_trades,
        "grid_size": len(cells),
    }


def _judge(winner, rho, eligible) -> tuple:
    """
    Decide what, if anything, this sweep is entitled to claim.

    Returns (verdict_text, survived_bool). The bar for `survived` is
    deliberately awkward to clear: the fit-best cell must ALSO be positive out
    of sample by more than one standard error. Anything less is noise wearing
    a rosette.
    """
    if not eligible:
        return ("NO VERDICT — no grid cell produced enough trades in both "
                "windows. Widen the date range or lower min_trades (and trust "
                "the result correspondingly less).", False)

    v = winner.validate
    exp, se = v["expectancy_r"], v["stderr_r"] or 0.0
    lower = exp - se

    if lower > 0:
        base = (f"SURVIVED — {winner.label} is the fit-window best AND is "
                f"positive out of sample: {exp:+.3f}R +/- {se:.3f} over "
                f"{v['trades']} validate trades (lower bound {lower:+.3f}R).")
        ok = True
    elif exp > 0:
        base = (f"INCONCLUSIVE — {winner.label} is positive out of sample "
                f"({exp:+.3f}R) but its standard error ({se:.3f}) covers zero. "
                f"Not evidence of an edge; at best a reason to collect more data.")
        ok = False
    else:
        base = (f"FAILED — the fit-window best ({winner.label}) is negative out "
                f"of sample: {exp:+.3f}R over {v['trades']} trades. The fit "
                f"ranking did not generalise.")
        ok = False

    if rho is None:
        base += " Rank correlation unavailable (too few eligible cells)."
    elif rho < 0.2:
        base += (f" Fit-vs-validate rank correlation is {rho:+.2f} — near zero, "
                 "meaning fit performance carried almost no information about "
                 "out-of-sample performance. Treat ANY winner here as noise, "
                 "including a 'surviving' one.")
        ok = False
    else:
        base += (f" Fit-vs-validate rank correlation {rho:+.2f}. Note this is "
                 "NOT independent evidence of edge: cost drag scales inversely "
                 "with stop width, so wider stops rank higher in both windows "
                 "even on structureless data (the synthetic control returns "
                 "rho = +0.61). Judge on the out-of-sample lower bound, not rho.")

    fit_exps = [c.fit["expectancy_r"] for c in eligible]
    if fit_exps and max(fit_exps) < 0:
        base += (f" Every eligible cell is negative in the fit window "
                 f"(best {max(fit_exps):+.3f}R) — consistent with no edge "
                 "anywhere in the grid, only varying amounts of cost drag.")
    return base, ok


def format_report(res: dict) -> str:
    lines = []
    lines.append("GEOMETRY SWEEP")
    lines.append("=" * 78)
    lines.append(f"  grid: {res['grid_size']} cells | min trades per window: {res['min_trades']}")
    lines.append(f"  fit:      {res['fit_sessions']:3d} sessions  {res['fit_range']}")
    lines.append(f"  validate: {res['validate_sessions']:3d} sessions  {res['validate_range']}")
    lines.append("")

    hdr = (f"  {'geometry':22s} {'fitN':>5s} {'fit exp':>9s} "
           f"{'valN':>5s} {'val exp':>9s} {'val +/-':>8s} {'tgt%':>6s} {'eod%':>6s}")
    lines.append(hdr)
    lines.append("  " + "-" * (len(hdr) - 2))

    order = res["ranked"] + [c for c in res["cells"] if c not in res["ranked"]]
    for c in order:
        f, v = c.fit, c.validate
        fn, vn = f.get("trades") or 0, v.get("trades") or 0
        if not vn:
            lines.append(f"  {c.label:22s} {fn:5d} {'-':>9s} {vn:5d} "
                         f"{'-':>9s} {'-':>8s} {'-':>6s} {'-':>6s}")
            continue
        ex = v["exits"]
        tgt = ex.get("target_hit", 0) / vn * 100
        eod = (ex.get("eod_close", 0) + ex.get("time_exit", 0)) / vn * 100
        thin = "" if c in res["ranked"] else "  (thin — not ranked)"
        lines.append(
            f"  {c.label:22s} {fn:5d} {f['expectancy_r']:+9.3f} {vn:5d} "
            f"{v['expectancy_r']:+9.3f} {v['stderr_r']:8.3f} "
            f"{tgt:5.1f}% {eod:5.1f}%{thin}")

    lines.append("")
    lines.append("VERDICT")
    lines.append("-" * 78)
    for chunk in _wrap(res["verdict"], 76):
        lines.append("  " + chunk)
    lines.append("")
    lines.append(f"  survived: {res['survived']}")
    if not res["survived"]:
        lines.append("  -> Do NOT promote any geometry into execution/sizing.py "
                     "on this result.")
    return "\n".join(lines)


def _wrap(text: str, width: int) -> list[str]:
    words, out, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            out.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        out.append(cur)
    return out
