"""One-time database initialization. Run once after creating the prophet DB."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db.connection import engine, Base, test_connection
from db.models import *
from db.operations import seed_default_watchlist
from db.connection import SessionLocal

print("Connecting to database...")
test_connection()

print("Creating tables...")
Base.metadata.create_all(bind=engine)
print("✓ All tables created")

print("Seeding watchlist...")
db = SessionLocal()
seed_default_watchlist(db)
wl_count = db.query(WatchlistItem).count()
db.close()
print(f"✓ Watchlist seeded: {wl_count} symbols")

print("\n✓ Database ready. Run tests with: python3 -m pytest tests/test_suite.py -v")
