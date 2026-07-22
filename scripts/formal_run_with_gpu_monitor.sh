#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 5 ]]; then
  echo "usage: $0 OUTPUT.csv OUTPUT.jsonl MAX_BYTES -- COMMAND [ARG ...]" >&2
  exit 2
fi
CSV_PATH="$1"
JSONL_PATH="$2"
MAX_BYTES="$3"
shift 3
[[ "${1:-}" == "--" ]] || { echo "literal -- delimiter is required" >&2; exit 2; }
shift
[[ "$#" -gt 0 ]] || { echo "training command is required" >&2; exit 2; }
: "${PYTHON:?PYTHON must name the trusted absolute Python interpreter}"
: "${FORMAL_GPU_EXPECTED_COUNT:?FORMAL_GPU_EXPECTED_COUNT must be 1 or 2}"
[[ "$PYTHON" == /* && -x "$PYTHON" ]] || { echo "invalid PYTHON" >&2; exit 2; }
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
PARENT="$(dirname "$JSONL_PATH")"
mkdir -p "$PARENT" "$(dirname "$CSV_PATH")"
CONTROL_DIR="$(mktemp -d "$PARENT/.gpu-control.XXXXXX")"
READY="$CONTROL_DIR/ready"
STOP="$CONTROL_DIR/stop"
ACTIVE_START="$CONTROL_DIR/active-start"
ACTIVE_END="$CONTROL_DIR/active-end"
cleanup() {
  rm -f "$READY" "$STOP" "$ACTIVE_START" "$ACTIVE_END"
  rmdir "$CONTROL_DIR" 2>/dev/null || true
  if [[ -n "${MONITOR_PID:-}" ]]; then
    kill "$MONITOR_PID" 2>/dev/null || true
    wait "$MONITOR_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

"$PYTHON" -I -B "$ROOT/scripts/summarize_formal_gpu_usage.py" \
  "$CSV_PATH" --record-jsonl "$JSONL_PATH" --ready-path "$READY" \
  --stop-path "$STOP" --active-start-path "$ACTIVE_START" \
  --active-end-path "$ACTIVE_END" --max-bytes "$MAX_BYTES" \
  --record-expected-gpus "$FORMAL_GPU_EXPECTED_COUNT" &
MONITOR_PID=$!
for _ in {1..100}; do
  [[ -e "$READY" ]] && break
  kill -0 "$MONITOR_PID" 2>/dev/null || { wait "$MONITOR_PID"; exit 1; }
  sleep 0.1
done
[[ -e "$READY" ]] || { echo "GPU recorder did not become ready" >&2; exit 1; }

set +e
FORMAL_GPU_ACTIVE_START_SIGNAL="$ACTIVE_START" \
FORMAL_GPU_ACTIVE_END_SIGNAL="$ACTIVE_END" "$@"
COMMAND_STATUS=$?
set -e
: >"$STOP"
set +e
wait "$MONITOR_PID"
MONITOR_STATUS=$?
set -e
MONITOR_PID=""
rm -f "$READY" "$STOP" "$ACTIVE_START" "$ACTIVE_END"
rmdir "$CONTROL_DIR"
trap - EXIT INT TERM
(( MONITOR_STATUS == 0 )) || exit "$MONITOR_STATUS"
exit "$COMMAND_STATUS"
