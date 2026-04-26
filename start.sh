#!/usr/bin/env bash
set -e

echo "Running database migrations..."
python -m alembic upgrade head

echo "Starting API + worker..."
python -m uvicorn api.main:app --host 0.0.0.0 --port $PORT