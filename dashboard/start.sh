#!/bin/bash
# Start FastAPI — no build step needed, frontend is pure HTML/JS
set -e
cd /app
uvicorn dashboard.api:app --host 0.0.0.0 --port ${PORT:-8000}
