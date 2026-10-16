#!/usr/bin/env bash
set -euo pipefail
REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONDONTWRITEBYTECODE=1
exec "${PYTHON_BIN:-python3}" "$REPO_DIR/pipeline/run_complete.py" "$@"
