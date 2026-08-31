#!/usr/bin/env sh
set -eu

PROJECT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
VENV_PYTHON="$PROJECT_DIR/.venv/bin/python"

if [ ! -x "$VENV_PYTHON" ]; then
    echo "FuncDroid is not set up yet. Run ./setup.sh first." >&2
    exit 1
fi

if [ ! -f "$PROJECT_DIR/funcdroid/.env" ]; then
    cp "$PROJECT_DIR/funcdroid/.env.example" "$PROJECT_DIR/funcdroid/.env"
    echo "Created funcdroid/.env. Fill API_KEY, BASE_URL, and MODEL, then run ./run.sh again." >&2
    exit 1
fi

cd "$PROJECT_DIR/funcdroid"
exec "$VENV_PYTHON" main.py "$@"
