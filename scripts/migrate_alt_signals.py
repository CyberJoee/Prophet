"""
Migration: create the alt_signals table.

Run once, BEFORE deploying the alt-signals code (or right after — the
collectors fail open if the table is missing, but you'd lose that day's
snapshots):

    python scripts/migrate_alt_signals.py

Safe to re-run — create_all skips existing tables.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv()


def migrate():
    from db.connection import engine
    from db.models import Base, AltSignal
    Base.metadata.create_all(engine, tables=[AltSignal.__table__])
    print("✓ alt_signals table ready")


if __name__ == "__main__":
    migrate()
