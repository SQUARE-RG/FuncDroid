#!/usr/bin/env sh
set -eu

PROJECT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
VENV_DIR="$PROJECT_DIR/.venv"

is_supported_python() {
    "$1" -c 'import sys; raise SystemExit(0 if (3, 11) <= sys.version_info[:2] <= (3, 13) else 1)' \
        >/dev/null 2>&1
}

find_python() {
    if [ -n "${PYTHON_BIN:-}" ]; then
        if command -v "$PYTHON_BIN" >/dev/null 2>&1 && is_supported_python "$PYTHON_BIN"; then
            command -v "$PYTHON_BIN"
            return 0
        fi
        echo "PYTHON_BIN must point to Python 3.11, 3.12, or 3.13." >&2
        return 1
    fi

    for candidate in python3.13 python3.12 python3.11 python3; do
        if command -v "$candidate" >/dev/null 2>&1 && is_supported_python "$candidate"; then
            command -v "$candidate"
            return 0
        fi
    done
    return 1
}

cd "$PROJECT_DIR"
if [ -x "$VENV_DIR/bin/python" ] && is_supported_python "$VENV_DIR/bin/python"; then
    echo "[setup] Reusing $($VENV_DIR/bin/python --version) from .venv."
else
    if [ -n "${PYTHON_BIN:-}" ]; then
        PYTHON_PATH=$(find_python)
    else
        PYTHON_PATH=$(find_python || true)
    fi

    if [ -n "$PYTHON_PATH" ]; then
        echo "[setup] Using $($PYTHON_PATH --version) at $PYTHON_PATH"
        "$PYTHON_PATH" -m venv "$VENV_DIR"
    elif command -v uv >/dev/null 2>&1; then
        echo "[setup] Python 3.11-3.13 was not found; installing Python 3.13 with uv."
        uv python install 3.13
        uv venv --python 3.13 --seed "$VENV_DIR"
    else
        echo "[setup] Python 3.11, 3.12, or 3.13 is required." >&2
        echo "Install Python from https://www.python.org/downloads/ and run ./setup.sh again." >&2
        exit 1
    fi
fi

VENV_PYTHON="$VENV_DIR/bin/python"
"$VENV_PYTHON" -m pip install --upgrade pip
"$VENV_PYTHON" -m pip install -r "$PROJECT_DIR/requirements.txt"

if [ ! -f "$PROJECT_DIR/funcdroid/.env" ]; then
    cp "$PROJECT_DIR/funcdroid/.env.example" "$PROJECT_DIR/funcdroid/.env"
    echo "[setup] Created funcdroid/.env from the example."
else
    echo "[setup] Kept the existing funcdroid/.env."
fi

echo ""
echo "Setup complete."
echo "1. Fill API_KEY, BASE_URL, and MODEL in funcdroid/.env."
echo "2. Enable USB debugging, connect a device, and find its ID with adb devices."
echo "3. Run: ./run.sh --device-id <device-id>"
