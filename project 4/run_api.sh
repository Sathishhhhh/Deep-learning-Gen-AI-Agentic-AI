#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
python -m uvicorn crypto_backend:app --reload --port 8014
