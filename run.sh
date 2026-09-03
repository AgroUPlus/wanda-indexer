#!/usr/bin/env bash
# Wanda indexer launcher for Linux / macOS / WSL.
# All arguments are passed straight through to main.py.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

PYTHON="${PYTHON:-}"
if [[ -z "$PYTHON" ]]; then
  for candidate in python3 python; do
    if command -v "$candidate" >/dev/null 2>&1; then PYTHON="$candidate"; break; fi
  done
fi
if [[ -z "$PYTHON" ]]; then
  echo "[ERROR] No Python interpreter found. Install Python 3.9 or newer." >&2
  exit 1
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "[ERROR] ffmpeg is required and is not on PATH." >&2
  echo "        Debian/Ubuntu: sudo apt install ffmpeg" >&2
  echo "        macOS:         brew install ffmpeg" >&2
  exit 1
fi

exec "$PYTHON" main.py "$@"
