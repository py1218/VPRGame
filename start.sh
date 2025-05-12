#!/usr/bin/env bash
set -e
cd /workspace
exec uvicorn app:app --host 0.0.0.0 --port ${PORT:-8080}
