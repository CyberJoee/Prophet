## What changed and why

<!-- The why matters more than the what. A future session reads this before
     the diff. -->

## Does this change trading behaviour?

<!-- Pick one and say which. -->

- [ ] **No** — docs, tests, tooling, or logging only
- [ ] **Yes** — it changes what gets traded, how it is sized, or when

If yes: what is the expected effect, and what would tell you it went wrong?

## How was it verified?

<!-- "It looks right" is not verification. This project has a long history of
     plausible-looking wrong data: phantom fills, mock trades recorded as live,
     templated journal entries, duplicate P&L rows. -->

- [ ] `python scripts/run_tests.py` passes
- [ ] New behaviour has a test that FAILS without the change
- [ ] Ran against real data via a preview script (`check_expectancy.py`,
      `eval_signals.py`, `run_geometry_sweep.py`, `dedupe_trades.py --dry-run`)
- [ ] Not applicable, because:

## Deploy timing

<!-- Merging to main auto-deploys BOTH services and restarts the scheduler. -->

- [ ] Market is closed, or this cannot affect a live position
- [ ] This needs to land during market hours, because:

## Anything measured?

<!-- If this claims an improvement, put the number here. If it claims a null
     result, say what effect size the test could actually have detected —
     "not significant" and "not there" are different claims. -->
