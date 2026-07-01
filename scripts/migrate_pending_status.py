"""
Migration: add PENDING_FILL to the tradestatus enum in Postgres.

Run ONCE on Railway (or locally against DATABASE_URL) BEFORE deploying
the patched code:

    python scripts/migrate_pending_status.py

Safe to re-run — uses IF NOT EXISTS.
Note: ALTER TYPE ... ADD VALUE must run outside a transaction block,
hence the autocommit connection.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv

load_dotenv()


def migrate():
    from sqlalchemy import create_engine, text

    url = os.getenv("DATABASE_URL")
    if not url:
        print("DATABASE_URL not set — aborting")
        sys.exit(1)
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql://", 1)

    engine = create_engine(url, isolation_level="AUTOCOMMIT")

    with engine.connect() as conn:
        # SQLAlchemy native enums store the member NAME (e.g. 'PENDING_FILL')
        conn.execute(text(
            "ALTER TYPE tradestatus ADD VALUE IF NOT EXISTS 'PENDING_FILL' BEFORE 'OPEN'"
        ))
        print("✓ tradestatus enum now includes PENDING_FILL")

        # Verify
        rows = conn.execute(text(
            "SELECT enumlabel FROM pg_enum "
            "JOIN pg_type ON pg_enum.enumtypid = pg_type.oid "
            "WHERE pg_type.typname = 'tradestatus' ORDER BY enumsortorder"
        )).fetchall()
        print(f"  current values: {[r[0] for r in rows]}")

        # Clean up zero-vector embeddings written by the old broken embed path.
        # Zero vectors are all 'identical' under cosine distance, so they would
        # poison any future semantic search. NULL is the honest state.
        result = conn.execute(text(
            "UPDATE trade_journals SET embedding = NULL "
            "WHERE embedding IS NOT NULL "
            "AND embedding <#> embedding = 0"   # inner product of zero vector with itself
        ))
        print(f"✓ nulled {result.rowcount} zero-vector embeddings")


if __name__ == "__main__":
    migrate()
