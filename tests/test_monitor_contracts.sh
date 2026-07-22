#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
SCRIPT="$ROOT/scripts/monitor_formal_identity.py"
PYTHON="${PYTHON:-python3}"
TMP="$(mktemp -d "${TMPDIR:-/tmp}/tabicl-monitor-contract.XXXXXX")"
TMP="$(cd "$TMP" && pwd -P)"
trap 'status=$?; if [[ $status -ne 0 && -f "$TMP/stderr" ]]; then cat "$TMP/stderr" >&2; fi; rm -rf "$TMP"; exit $status' EXIT

mkdir -p "$TMP/bin" "$TMP/work" "$TMP/artifacts" "$TMP/state" "$TMP/events"
CALLS="$TMP/scheduler.calls"
for command in squeue sacct; do
  cat >"$TMP/bin/$command" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' "$(basename "$0")" >>"$MONITOR_CALLS"
if [[ "$(basename "$0")" == "squeue" ]]; then
  printf '%s\n' 'job-a|PENDING'
else
  printf '%s\n' 'job-a|PENDING|0:0'
fi
SH
  chmod +x "$TMP/bin/$command"
done

printf '%s\n' 'step=1 loss=0.5' >"$TMP/artifacts/train.log"
RAW_SCRIPT="$ROOT/scripts/train_v2_clf_identity_stage1.sh"
RAW_BEFORE="$(git -C "$ROOT" hash-object "$RAW_SCRIPT")"
LOG_BEFORE="$(git -C "$ROOT" hash-object "$TMP/artifacts/train.log")"

cat >"$TMP/monitor.json" <<JSON
{
  "schema_version": 1,
  "disk_path": "$TMP/artifacts",
  "max_event_ledger_bytes": 1048576,
  "jobs": [{
    "job_id": "job-a",
    "arm": "rope",
    "stage": "stage1",
    "log_path": "$TMP/artifacts/train.log",
    "checkpoint_path": "$TMP/artifacts/step-500000.ckpt",
    "validation_report_path": "$TMP/artifacts/validation.json",
    "gpu_samples_path": null,
    "utilization_required": false,
    "stall_seconds": 60,
    "checkpoint_stale_seconds": 120,
    "max_checkpoint_bytes": 1048576,
    "expected_gpu_uuids": [],
    "expected_gpu_count": 0
  }]
}
JSON

(
  cd "$TMP/work"
  PATH="$TMP/bin:$PATH" MONITOR_CALLS="$CALLS" "$SCRIPT" \
    --manifest "$TMP/monitor.json" \
    --state-dir "$TMP/state" \
    --event-ledger-dir "$TMP/events" \
    --once >"$TMP/stdout.jsonl" 2>"$TMP/stderr"
)

test ! -s "$TMP/stderr"
test -f "$TMP/state/state.json"
test "$(git -C "$ROOT" hash-object "$TMP/artifacts/train.log")" = "$LOG_BEFORE"
test "$(git -C "$ROOT" hash-object "$RAW_SCRIPT")" = "$RAW_BEFORE"
test "$(sort "$CALLS" | tr '\n' ' ')" = "sacct squeue "

"$PYTHON" - "$TMP/stdout.jsonl" "$TMP/events/events.jsonl" <<'PY'
import json
from pathlib import Path
import sys

stdout_path, ledger_path = map(Path, sys.argv[1:])
lines = stdout_path.read_text().splitlines()
for line in lines:
    event = json.loads(line)
    assert event["category"] in {"anomaly", "stage_transition", "checkpoint_issue"}
if lines:
    assert ledger_path.read_text().splitlines() == lines
else:
    assert not ledger_path.exists()
PY

READY="$TMP/lock.ready"
"$PYTHON" - "$TMP/state/monitor.lock" "$READY" <<'PY' &
import fcntl
from pathlib import Path
import sys
import time

with open(sys.argv[1], "a+b") as handle:
    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
    Path(sys.argv[2]).touch()
    time.sleep(10)
PY
LOCK_HOLDER=$!
for _ in {1..100}; do
  [[ -f "$READY" ]] && break
  sleep 0.01
done
test -f "$READY"
CALL_COUNT_BEFORE="$(wc -l <"$CALLS" | tr -d ' ')"
PATH="$TMP/bin:$PATH" MONITOR_CALLS="$CALLS" "$SCRIPT" \
  --manifest "$TMP/monitor.json" \
  --state-dir "$TMP/state" \
  --event-ledger-dir "$TMP/events" \
  --once >"$TMP/locked.stdout" 2>"$TMP/locked.stderr"
test ! -s "$TMP/locked.stdout"
test ! -s "$TMP/locked.stderr"
test "$(wc -l <"$CALLS" | tr -d ' ')" = "$CALL_COUNT_BEFORE"
kill "$LOCK_HOLDER"
wait "$LOCK_HOLDER" 2>/dev/null || true

if grep -Eq 's(bat(ch)?|control|cancel)' "$SCRIPT"; then
  echo "monitor contains a scheduler-mutating command" >&2
  exit 1
fi

echo "monitor contract tests passed"
