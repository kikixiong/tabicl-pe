#!/usr/bin/env bash
set -euo pipefail

OVERLAY="${1:-}"
[[ -n "$OVERLAY" && "$OVERLAY" == /* ]] || {
  echo "usage: $0 /absolute/path/to/h100-gate-overlay.json" >&2
  exit 2
}
: "${RUN_POLICY:?RUN_POLICY must be fresh}"
[[ "$RUN_POLICY" == "fresh" ]] || { echo "H100 validation submission is fresh-only" >&2; exit 2; }
: "${TABICL_EXACT_ROOT:?TABICL_EXACT_ROOT is required}"
ROOT="$(cd "$TABICL_EXACT_ROOT" && pwd -P)"
: "${PYTHON:?PYTHON must be an absolute trusted interpreter}"
[[ "$PYTHON" == /* && -x "$PYTHON" ]] || { echo "invalid PYTHON" >&2; exit 2; }

unset PYTHONHOME PYTHONSTARTUP PYTHONINSPECT PYTHONUSERBASE
export PYTHONPATH="$ROOT/src"
export PYTHONNOUSERSITE=1
exec "$PYTHON" -I -B "$ROOT/scripts/submit_h100_identity_validation.py" \
  --overlay "$OVERLAY" --exact-root "$ROOT"
