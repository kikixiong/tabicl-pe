#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
SCRIPT="$ROOT/scripts/monitor_formal_identity.py"
PYTHON="${PYTHON:-python3}"
TMP="$(mktemp -d "${TMPDIR:-/tmp}/tabicl-monitor-contract.XXXXXX")"
TMP="$(cd "$TMP" && pwd -P)"
WRITABLE_TMP=""
for CANDIDATE in /dev/shm /tmp "$ROOT/.."; do
  if [[ -d "$CANDIDATE" && -w "$CANDIDATE" && -x "$CANDIDATE" \
    && "$(stat -c %d "$CANDIDATE")" != "$(stat -c %d "$TMP")" ]]; then
    WRITABLE_TMP="$(mktemp -d "$CANDIDATE/tabicl-monitor-writable.XXXXXX")"
    WRITABLE_TMP="$(cd "$WRITABLE_TMP" && pwd -P)"
    break
  fi
done
[[ -n "$WRITABLE_TMP" ]] || { echo "monitor test requires a second writable filesystem" >&2; exit 1; }
trap 'status=$?; if [[ $status -ne 0 && -f "$TMP/stderr" ]]; then cat "$TMP/stderr" >&2; fi; rm -rf "$TMP" "$WRITABLE_TMP"; exit $status' EXIT

mkdir -p "$TMP/bin" "$TMP/work" "$TMP/artifacts" "$WRITABLE_TMP/state" "$WRITABLE_TMP/events"
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
SQUEUE_SHA="$($PYTHON - "$TMP/bin/squeue" <<'PY'
import hashlib
from pathlib import Path
import sys
print(hashlib.sha256(Path(sys.argv[1]).read_bytes()).hexdigest())
PY
)"
SACCT_SHA="$($PYTHON - "$TMP/bin/sacct" <<'PY'
import hashlib
from pathlib import Path
import sys
print(hashlib.sha256(Path(sys.argv[1]).read_bytes()).hexdigest())
PY
)"

printf '%s\n' 'step=1 loss=0.5' >"$TMP/artifacts/train.log"
RAW_SCRIPT="$ROOT/scripts/train_v2_clf_identity_stage1.sh"
RAW_BEFORE="$(git -C "$ROOT" hash-object "$RAW_SCRIPT")"
LOG_BEFORE="$(git -C "$ROOT" hash-object "$TMP/artifacts/train.log")"

cat >"$TMP/monitor.json" <<JSON
{
  "schema_version": 2,
  "disk_path": "$TMP/artifacts",
  "max_event_ledger_bytes": 1048576,
  "squeue_path": "$TMP/bin/squeue",
  "squeue_sha256": "$SQUEUE_SHA",
  "sacct_path": "$TMP/bin/sacct",
  "sacct_sha256": "$SACCT_SHA",
  "jobs": [{
    "job_id": "job-a",
    "arm": "rope",
    "stage": "stage1",
    "log_path": "$TMP/artifacts/train.log",
    "live_log_path": null,
    "checkpoint_dir": "$TMP/artifacts",
    "validation_report_path": null,
    "validation_report_required": false,
    "formal_validation": null,
    "gpu_samples_path": null,
    "gpu_sample_format": "none",
    "step_offset": 0,
    "utilization_required": false,
    "stall_seconds": 60,
    "checkpoint_stale_seconds": 120,
    "max_checkpoint_bytes": 1048576,
    "expected_gpu_uuids": [],
    "expected_gpu_count": 0
  }]
}
JSON

mkdir -p "$TMP/bad-state" "$TMP/bad-events"
if PATH="$TMP/bin:$PATH" MONITOR_CALLS="$CALLS" "$SCRIPT" \
  --manifest "$TMP/monitor.json" \
  --state-dir "$TMP/bad-state" \
  --event-ledger-dir "$TMP/bad-events" \
  --once >"$TMP/bad.stdout" 2>"$TMP/bad.stderr"; then
  echo "monitor accepted writable state on the monitored filesystem" >&2
  exit 1
fi
grep -F "different filesystem" "$TMP/bad.stderr" >/dev/null
test ! -e "$TMP/bad-state/state.json"
test ! -e "$TMP/bad-events/events.jsonl"
test ! -e "$CALLS"

(
  cd "$TMP/work"
  PATH="$TMP/bin:$PATH" MONITOR_CALLS="$CALLS" "$SCRIPT" \
    --manifest "$TMP/monitor.json" \
    --state-dir "$WRITABLE_TMP/state" \
    --event-ledger-dir "$WRITABLE_TMP/events" \
    --once >"$TMP/stdout.jsonl" 2>"$TMP/stderr"
)

test ! -s "$TMP/stderr"
test -f "$WRITABLE_TMP/state/state.json"
test "$(git -C "$ROOT" hash-object "$TMP/artifacts/train.log")" = "$LOG_BEFORE"
test "$(git -C "$ROOT" hash-object "$RAW_SCRIPT")" = "$RAW_BEFORE"
test "$(sort "$CALLS" | tr '\n' ' ')" = "sacct squeue "

"$PYTHON" - "$TMP/stdout.jsonl" "$WRITABLE_TMP/events/events.jsonl" <<'PY'
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

# Schema v2 observes the bounded live inode before durable publication and
# discovers the numeric latest checkpoint from a no-follow stage directory.
mkdir -p "$TMP/artifacts/pilot-stage" "$WRITABLE_TMP/pilot-state" "$WRITABLE_TMP/pilot-events"
printf '%s\n' 'step=12 loss=0.4' >"$TMP/artifacts/pilot.log.live"
"$PYTHON" - "$TMP/artifacts/pilot-stage" <<'PY'
from pathlib import Path
import sys
import zipfile

root = Path(sys.argv[1])
for step in (9, 10):
    with zipfile.ZipFile(root / f"step-{step}.ckpt", "w") as archive:
        archive.writestr("checkpoint/data.pkl", f"step-{step}".encode())
PY
cat >"$TMP/pilot-monitor.json" <<JSON
{
  "schema_version": 2,
  "disk_path": "$TMP/artifacts",
  "max_event_ledger_bytes": 1048576,
  "squeue_path": "$TMP/bin/squeue",
  "squeue_sha256": "$SQUEUE_SHA",
  "sacct_path": "$TMP/bin/sacct",
  "sacct_sha256": "$SACCT_SHA",
  "jobs": [{
    "job_id": "job-a",
    "arm": "none",
    "stage": "stage1",
    "log_path": "$TMP/artifacts/pilot.log",
    "live_log_path": "$TMP/artifacts/pilot.log.live",
    "checkpoint_dir": "$TMP/artifacts/pilot-stage",
    "validation_report_path": null,
    "validation_report_required": false,
    "gpu_samples_path": null,
    "gpu_sample_format": "none",
    "step_offset": 0,
    "utilization_required": false,
    "stall_seconds": 60,
    "checkpoint_stale_seconds": 120,
    "max_checkpoint_bytes": 1048576,
    "expected_gpu_uuids": [],
    "expected_gpu_count": 0,
    "formal_validation": null
  }]
}
JSON
PATH="$TMP/bin:$PATH" MONITOR_CALLS="$CALLS" "$SCRIPT" \
  --manifest "$TMP/pilot-monitor.json" \
  --state-dir "$WRITABLE_TMP/pilot-state" \
  --event-ledger-dir "$WRITABLE_TMP/pilot-events" \
  --once >"$TMP/pilot-live.stdout" 2>"$TMP/pilot-live.stderr"
test ! -s "$TMP/pilot-live.stderr"
"$PYTHON" - "$TMP/pilot-live.stdout" "$WRITABLE_TMP/pilot-events/events.jsonl" <<'PY'
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
"$PYTHON" - "$WRITABLE_TMP/pilot-state/state.json" <<'PY'
import json
from pathlib import Path
import sys

state = json.loads(Path(sys.argv[1]).read_text())
job = state["jobs"]["job-a"]
assert job["last_step"] == 12
assert job["checkpoint_step"] == 10
assert job["checkpoint_selected_path"].endswith("/step-10.ckpt")
assert "validation_report_identity" not in job
PY
ln "$TMP/artifacts/pilot.log.live" "$TMP/artifacts/pilot.log"
rm "$TMP/artifacts/pilot.log.live"
PATH="$TMP/bin:$PATH" MONITOR_CALLS="$CALLS" "$SCRIPT" \
  --manifest "$TMP/pilot-monitor.json" \
  --state-dir "$WRITABLE_TMP/pilot-state" \
  --event-ledger-dir "$WRITABLE_TMP/pilot-events" \
  --once >"$TMP/pilot-final.stdout" 2>"$TMP/pilot-final.stderr"
test ! -s "$TMP/pilot-final.stdout"
test ! -s "$TMP/pilot-final.stderr"

READY="$TMP/lock.ready"
"$PYTHON" - "$WRITABLE_TMP/state/monitor.lock" "$READY" <<'PY' &
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
  --state-dir "$WRITABLE_TMP/state" \
  --event-ledger-dir "$WRITABLE_TMP/events" \
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
