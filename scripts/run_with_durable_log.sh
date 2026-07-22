#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 3 ]]; then
  echo "usage: $0 OUTPUT MAX_BYTES COMMAND [ARG ...]" >&2
  exit 2
fi

OUTPUT="$1"
MAX_BYTES="$2"
shift 2

case "$MAX_BYTES" in
  ''|*[!0-9]*) echo "MAX_BYTES must be a non-negative integer" >&2; exit 2 ;;
esac
PARENT="$(dirname "$OUTPUT")"
mkdir -p "$PARENT"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
: "${PYTHON:?PYTHON must name the trusted absolute Python interpreter}"
if [[ "$PYTHON" != /* || ! -x "$PYTHON" ]]; then
  echo "PYTHON must be an executable absolute path" >&2
  exit 2
fi

exec "$PYTHON" -I -B "$ROOT/scripts/reject_nonfinite_log.py" \
  --max-bytes "$MAX_BYTES" --capture "$OUTPUT" -- "$@"
