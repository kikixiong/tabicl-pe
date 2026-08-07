#!/usr/bin/env bash
set -euo pipefail

OVERLAY="${1:-}"
[[ -n "$OVERLAY" && "$OVERLAY" == /* ]] || {
  echo "usage: $0 /absolute/path/to/formal-overlay.json" >&2
  exit 2
}
: "${RUN_POLICY:?RUN_POLICY must be fresh}"
[[ "$RUN_POLICY" == "fresh" ]] || {
  echo "formal submission is fresh-only" >&2
  exit 2
}
: "${TABICL_EXACT_ROOT:?TABICL_EXACT_ROOT is required}"
ROOT="$(cd "$TABICL_EXACT_ROOT" && pwd -P)"
: "${PYTHON:?PYTHON must be an absolute trusted interpreter}"
[[ "$PYTHON" == /* && -x "$PYTHON" ]] || {
  echo "PYTHON must be an absolute executable" >&2
  exit 2
}

# The controller and every submitted job receive a closed import environment.
# Python -I ignores PYTHONPATH; exact-T bootstrap code inserts the attested src
# directory explicitly.  PYTHONPATH remains an attestable value only.
unset PYTHONHOME PYTHONSTARTUP PYTHONINSPECT PYTHONUSERBASE
export PYTHONPATH="$ROOT/src"
export PYTHONNOUSERSITE=1

ARGS=(
  --submit-overlay "$OVERLAY"
  --exact-root "$ROOT"
)
if [[ -n "${FORMAL_FAULT_LEDGER_STAGE:-}" ]]; then
  ARGS+=(--fault-ledger-stage "$FORMAL_FAULT_LEDGER_STAGE")
fi
if [[ -n "${FORMAL_FAULT_JOURNAL_SEAL_STAGE:-}" ]]; then
  ARGS+=(--fault-journal-seal-stage "$FORMAL_FAULT_JOURNAL_SEAL_STAGE")
fi
if [[ -n "${FORMAL_FAULT_COMMIT_STAGE:-}" ]]; then
  ARGS+=(--fault-commit-stage "$FORMAL_FAULT_COMMIT_STAGE")
fi
if [[ -n "${FORMAL_TEST_SIGNAL_AFTER_COMMIT:-}" ]]; then
  ARGS+=(--test-signal-after-commit)
fi
exec "$PYTHON" -I -B "$ROOT/scripts/verify_formal_overlay.py" "${ARGS[@]}"
