"""
Run Prophet's self-contained test suite.

    python scripts/run_tests.py           # the suite CI runs
    python scripts/run_tests.py --all     # also the Postgres-dependent suite
    python scripts/run_tests.py -v        # show output from passing tests too

Each test file is a standalone script that exits non-zero on failure, so this
just runs them as subprocesses and aggregates. One command locally and in CI
means "it passed on my machine" and "it passed in CI" mean the same thing.

WHY test_suite.py IS EXCLUDED BY DEFAULT
----------------------------------------
The original suite drops and recreates tables against a real Postgres via
DATABASE_URL. That cannot run in CI, and running it locally against a
DATABASE_URL pointing at production would destroy the trade history. It is
opt-in through --all, and you should be certain DATABASE_URL is disposable
before using that flag.

Everything else is deliberately self-contained: in-memory SQLite with
@compiles shims for the Postgres-only column types, and stubbed alpaca/groq
modules. No network, no database, no API keys.
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TESTS = REPO / "tests"

# Requires a real Postgres and is destructive to whatever DATABASE_URL points at.
NEEDS_POSTGRES = {"test_suite.py"}


def discover(include_all: bool):
    files = sorted(TESTS.glob("test_*.py"))
    if include_all:
        return files
    return [f for f in files if f.name not in NEEDS_POSTGRES]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true",
                    help="also run tests needing a real Postgres (DESTRUCTIVE "
                         "to the database DATABASE_URL points at)")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="print output from passing tests as well")
    args = ap.parse_args()

    files = discover(args.all)
    if not files:
        print("No test files found.")
        return 1

    if args.all:
        print("WARNING: --all includes tests that DROP AND RECREATE tables on "
              f"DATABASE_URL={os.getenv('DATABASE_URL', '(unset)')[:40]!r}\n")

    skipped = sorted(NEEDS_POSTGRES) if not args.all else []
    passed, failed = [], []
    t0 = time.time()

    for f in files:
        started = time.time()
        proc = subprocess.run([sys.executable, str(f)], cwd=str(REPO),
                              capture_output=True, text=True)
        took = time.time() - started
        ok = proc.returncode == 0
        (passed if ok else failed).append(f.name)
        print(f"  {'PASS' if ok else 'FAIL'}  {f.name:<32} {took:5.1f}s")
        if not ok or args.verbose:
            out = (proc.stdout or "") + (proc.stderr or "")
            for line in out.rstrip().splitlines():
                print(f"        {line}")

    print()
    print(f"{len(passed)} passed, {len(failed)} failed, "
          f"{len(skipped)} skipped in {time.time() - t0:.1f}s")
    if skipped:
        print(f"skipped (needs Postgres): {', '.join(skipped)}")
    if failed:
        print(f"FAILED: {', '.join(failed)}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
