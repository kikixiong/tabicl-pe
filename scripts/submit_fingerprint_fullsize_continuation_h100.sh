#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
required=(
  SOURCE_ROOT CONTINUATION_ARTIFACT_ROOT PYTHON ORIGIN_SOURCE_SHA
  ROPE_PARENT_CHECKPOINT ROPE_PARENT_SHA256 ROPE_ORIGIN_COMPLETION
  ROPE_ORIGIN_COMPLETION_SHA256 FINGERPRINT_PARENT_CHECKPOINT
  FINGERPRINT_PARENT_SHA256 FINGERPRINT_ORIGIN_COMPLETION
  FINGERPRINT_ORIGIN_COMPLETION_SHA256 RUN_KIND
)
for name in "${required[@]}"; do
  [[ -n "${!name:-}" ]] || { echo "missing $name" >&2; exit 2; }
done
case "$RUN_KIND" in smoke|full) ;; *) echo "RUN_KIND must be smoke or full" >&2; exit 2 ;; esac
[[ "$SOURCE_ROOT" == /* && "$CONTINUATION_ARTIFACT_ROOT" == /* ]] || {
  echo "source and artifact roots must be absolute" >&2
  exit 2
}
[[ "$PYTHON" == /* && -x "$PYTHON" ]] || { echo "invalid PYTHON" >&2; exit 2; }
[[ "$ORIGIN_SOURCE_SHA" =~ ^[0-9a-f]{40}$ ]] || { echo "invalid origin source SHA" >&2; exit 2; }
[[ "$(realpath "$ROOT")" == "$(realpath "$SOURCE_ROOT")" ]] || {
  echo "submit controller must execute from SOURCE_ROOT" >&2
  exit 1
}
[[ ! -e "$CONTINUATION_ARTIFACT_ROOT" ]] || {
  echo "artifact root must be fresh: $CONTINUATION_ARTIFACT_ROOT" >&2
  exit 1
}

EXPECTED_SOURCE_SHA="$(git -C "$SOURCE_ROOT" rev-parse HEAD)"
[[ "$EXPECTED_SOURCE_SHA" =~ ^[0-9a-f]{40}$ ]] || { echo "invalid source HEAD" >&2; exit 1; }
[[ -z "$(git -C "$SOURCE_ROOT" status --porcelain --untracked-files=all)" ]] || {
  echo "SOURCE_ROOT must be clean" >&2
  exit 1
}
if git -C "$SOURCE_ROOT" symbolic-ref -q HEAD >/dev/null; then
  echo "SOURCE_ROOT must be a detached checkout" >&2
  exit 1
fi
REMOTE_SHA="$(timeout 30 git -C "$SOURCE_ROOT" ls-remote --refs origin \
  refs/heads/codex/fingerprint-setbind-pilot | awk 'NR == 1 {print $1}')"
[[ "$REMOTE_SHA" == "$EXPECTED_SOURCE_SHA" ]] || {
  echo "exact source commit is not advertised by the public pilot branch" >&2
  exit 1
}

if [[ "$RUN_KIND" == full ]]; then
  QOS=medium
  TIME_LIMIT=1-00:00:00
  TARGETS=(20000 35000 50000)
  # 22 planned output/staged-parent slots at the externally enforced 300 MB
  # ceiling, plus the immutable 20 GiB reserve and 25% margin.
  REQUIRED_FREE_BYTES=29724836480
else
  QOS=short
  TIME_LIMIT=00:30:00
  TARGETS=(5001)
  REQUIRED_FREE_BYTES=22974836480
fi

AVAILABLE_BYTES="$($PYTHON - "$CONTINUATION_ARTIFACT_ROOT" <<'PY'
import os
import sys

path = sys.argv[1]
probe = os.path.dirname(path)
while not os.path.exists(probe):
    parent = os.path.dirname(probe)
    if parent == probe:
        raise SystemExit("no existing artifact ancestor")
    probe = parent
value = os.statvfs(probe)
print(value.f_bavail * value.f_frsize)
PY
)"
[[ "$AVAILABLE_BYTES" =~ ^[0-9]+$ ]] || { echo "invalid free-space result" >&2; exit 1; }
(( AVAILABLE_BYTES >= REQUIRED_FREE_BYTES )) || {
  echo "insufficient capacity: $AVAILABLE_BYTES < $REQUIRED_FREE_BYTES" >&2
  exit 1
}

umask 077
mkdir "$CONTINUATION_ARTIFACT_ROOT"
mkdir "$CONTINUATION_ARTIFACT_ROOT/logs"
mkdir "$CONTINUATION_ARTIFACT_ROOT/parents"
mkdir "$CONTINUATION_ARTIFACT_ROOT/work"

EXPECTED_ENVIRONMENT_SHA256="$(PYTHONPATH="$SOURCE_ROOT/src" PYTHONNOUSERSITE=1 \
  "$PYTHON" -B "$SOURCE_ROOT/scripts/run_fingerprint_fullsize_continuation.py" \
  prepare-environment --output "$CONTINUATION_ARTIFACT_ROOT/environment.json")"
[[ "$EXPECTED_ENVIRONMENT_SHA256" =~ ^[0-9a-f]{64}$ ]] || {
  echo "invalid continuation environment fingerprint" >&2
  exit 1
}

for ARM in rope fingerprint; do
  if [[ "$ARM" == rope ]]; then
    CHECKPOINT="$ROPE_PARENT_CHECKPOINT"
    CHECKPOINT_SHA="$ROPE_PARENT_SHA256"
    ORIGIN_COMPLETION="$ROPE_ORIGIN_COMPLETION"
    ORIGIN_COMPLETION_SHA="$ROPE_ORIGIN_COMPLETION_SHA256"
  else
    CHECKPOINT="$FINGERPRINT_PARENT_CHECKPOINT"
    CHECKPOINT_SHA="$FINGERPRINT_PARENT_SHA256"
    ORIGIN_COMPLETION="$FINGERPRINT_ORIGIN_COMPLETION"
    ORIGIN_COMPLETION_SHA="$FINGERPRINT_ORIGIN_COMPLETION_SHA256"
  fi
  PYTHONPATH="$SOURCE_ROOT/src" PYTHONNOUSERSITE=1 "$PYTHON" -B \
    "$SOURCE_ROOT/scripts/run_fingerprint_fullsize_continuation.py" prepare-parent \
    --arm "$ARM" \
    --checkpoint "$CHECKPOINT" \
    --checkpoint-sha256 "$CHECKPOINT_SHA" \
    --origin-completion "$ORIGIN_COMPLETION" \
    --origin-completion-sha256 "$ORIGIN_COMPLETION_SHA" \
    --origin-source-commit "$ORIGIN_SOURCE_SHA" \
    --environment-manifest "$CONTINUATION_ARTIFACT_ROOT/environment.json" \
    --output "$CONTINUATION_ARTIFACT_ROOT/parents/$ARM-step-005000.json"
done

declare -a JOB_IDS=()
declare -a PREVIOUS_PAIR=()
declare -a JOB_RECORDS=()
released=0
rollback() {
  status=$?
  trap - EXIT HUP INT TERM
  if [[ "$released" -eq 0 && "${#JOB_IDS[@]}" -gt 0 ]]; then
    for ((index=${#JOB_IDS[@]}-1; index>=0; index--)); do
      scancel "${JOB_IDS[$index]}" 2>/dev/null || true
    done
  fi
  exit "$status"
}
trap rollback EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

FROM_STEP=5000
for TO_STEP in "${TARGETS[@]}"; do
  declare -a CURRENT_PAIR=()
  for ARM in rope fingerprint; do
    if [[ "$FROM_STEP" -eq 5000 ]]; then
      PARENT_MANIFEST="$CONTINUATION_ARTIFACT_ROOT/parents/$ARM-step-005000.json"
    else
      printf -v PARENT_FROM_LABEL '%06d' "$PREVIOUS_FROM_STEP"
      printf -v PARENT_TO_LABEL '%06d' "$FROM_STEP"
      PARENT_MANIFEST="$CONTINUATION_ARTIFACT_ROOT/arms/$ARM/seed-42/segments/${PARENT_FROM_LABEL}-${PARENT_TO_LABEL}/resource/completion.json"
    fi
    printf -v FROM_LABEL '%06d' "$FROM_STEP"
    printf -v TO_LABEL '%06d' "$TO_STEP"
    DEPENDENCY_ARGS=()
    if [[ "${#PREVIOUS_PAIR[@]}" -eq 2 ]]; then
      DEPENDENCY_ARGS=(--dependency="afterok:${PREVIOUS_PAIR[0]}:${PREVIOUS_PAIR[1]}" --kill-on-invalid-dep=yes)
    fi
    JOB_ID="$({ sbatch --parsable --hold \
      --job-name="tabicl-${ARM}-${FROM_LABEL}-${TO_LABEL}" \
      --qos="$QOS" --time="$TIME_LIMIT" \
      "${DEPENDENCY_ARGS[@]}" \
      --chdir="$CONTINUATION_ARTIFACT_ROOT/work" \
      --output="$CONTINUATION_ARTIFACT_ROOT/logs/${ARM}-${FROM_LABEL}-${TO_LABEL}-%j.out" \
      --error="$CONTINUATION_ARTIFACT_ROOT/logs/${ARM}-${FROM_LABEL}-${TO_LABEL}-%j.err" \
      --export="ALL,ARM=$ARM,SOURCE_ROOT=$SOURCE_ROOT,EXPECTED_SOURCE_SHA=$EXPECTED_SOURCE_SHA,CONTINUATION_ARTIFACT_ROOT=$CONTINUATION_ARTIFACT_ROOT,PYTHON=$PYTHON,FROM_STEP=$FROM_STEP,TO_STEP=$TO_STEP,PARENT_MANIFEST=$PARENT_MANIFEST,ENVIRONMENT_MANIFEST=$CONTINUATION_ARTIFACT_ROOT/environment.json" \
      "$SOURCE_ROOT/scripts/slurm_fingerprint_fullsize_continuation_h100.sh"; } | cut -d';' -f1)"
    [[ "$JOB_ID" =~ ^[0-9]+$ ]] || { echo "invalid sbatch result: $JOB_ID" >&2; exit 1; }
    JOB_IDS+=("$JOB_ID")
    CURRENT_PAIR+=("$JOB_ID")
    JOB_RECORDS+=("$ARM:$FROM_STEP:$TO_STEP:$JOB_ID")
  done
  PREVIOUS_PAIR=("${CURRENT_PAIR[@]}")
  PREVIOUS_FROM_STEP="$FROM_STEP"
  FROM_STEP="$TO_STEP"
done

# Recheck capacity while every job is held.
HELD_AVAILABLE_BYTES="$($PYTHON -c 'import os,sys; v=os.statvfs(sys.argv[1]); print(v.f_bavail*v.f_frsize)' "$CONTINUATION_ARTIFACT_ROOT")"
(( HELD_AVAILABLE_BYTES >= REQUIRED_FREE_BYTES )) || {
  echo "held capacity recheck failed" >&2
  exit 1
}
HELD_REMOTE_SHA="$(timeout 30 git -C "$SOURCE_ROOT" ls-remote --refs origin \
  refs/heads/codex/fingerprint-setbind-pilot | awk 'NR == 1 {print $1}')"
[[ "$HELD_REMOTE_SHA" == "$EXPECTED_SOURCE_SHA" ]] || {
  echo "public source ref changed while jobs were held" >&2
  exit 1
}

export EXPECTED_SOURCE_SHA ORIGIN_SOURCE_SHA EXPECTED_ENVIRONMENT_SHA256 RUN_KIND QOS TIME_LIMIT
export AVAILABLE_BYTES HELD_AVAILABLE_BYTES REQUIRED_FREE_BYTES HELD_REMOTE_SHA
JOB_RECORDS_JOINED="$(IFS=,; echo "${JOB_RECORDS[*]}")"
export JOB_RECORDS_JOINED CONTINUATION_ARTIFACT_ROOT
publish_transaction_record() {
  local phase="$1"
  "$PYTHON" - "$phase" <<'PY'
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

root = Path(os.environ["CONTINUATION_ARTIFACT_ROOT"])
phase = sys.argv[1]
if phase not in {"held", "released"}:
    raise SystemExit("invalid transaction publication phase")
jobs = []
for item in os.environ["JOB_RECORDS_JOINED"].split(","):
    arm, from_step, to_step, job_id = item.split(":")
    jobs.append(
        {
            "arm": arm,
            "from_step": int(from_step),
            "to_step": int(to_step),
            "job_id": int(job_id),
        }
    )
parents = {}
for arm in ("rope", "fingerprint"):
    path = root / "parents" / f"{arm}-step-005000.json"
    value = json.loads(path.read_bytes())
    parents[arm] = {"path": str(path), "sha256": value["sha256"]}
environment = json.loads((root / "environment.json").read_bytes())
payload = {
    "schema_version": 1,
    "study": "tabiclv2-fullsize-rope-fingerprint-continuation-v1",
    "formal_eligible": False,
    "source_commit": os.environ["EXPECTED_SOURCE_SHA"],
    "origin_source_commit": os.environ["ORIGIN_SOURCE_SHA"],
    "environment_sha256": os.environ["EXPECTED_ENVIRONMENT_SHA256"],
    "seed": 42,
    "run_kind": os.environ["RUN_KIND"],
    "scheduler_horizon_steps": 500000,
    "qos": os.environ["QOS"],
    "time_limit": os.environ["TIME_LIMIT"],
    "capacity": {
        "available_bytes": int(os.environ["AVAILABLE_BYTES"]),
        "held_available_bytes": int(os.environ["HELD_AVAILABLE_BYTES"]),
        "required_free_bytes": int(os.environ["REQUIRED_FREE_BYTES"]),
    },
    "held_public_ref_sha": os.environ["HELD_REMOTE_SHA"],
    "environment_manifest_sha256": environment["sha256"],
    "parent_manifests": parents,
    "jobs": jobs,
}
if phase == "held":
    payload["jobs_held_at_publication"] = True
    kind = "fingerprint_fullsize_continuation_held_plan"
    output = root / "held-plan.json"
else:
    held_path = root / "held-plan.json"
    held_raw = held_path.read_bytes()
    held = json.loads(held_raw)
    payload["held_plan_file_sha256"] = hashlib.sha256(held_raw).hexdigest()
    payload["held_plan_manifest_sha256"] = held["sha256"]
    payload["all_jobs_released"] = True
    kind = "fingerprint_fullsize_continuation_release_receipt"
    output = root / "submission.json"
body = {"schema_version": 1, "kind": kind, "payload": payload}
canonical_body = json.dumps(
    body, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
).encode("utf-8")
record = {**body, "sha256": hashlib.sha256(canonical_body).hexdigest()}
encoded = json.dumps(
    record, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
).encode("utf-8") + b"\n"
fd, name = tempfile.mkstemp(prefix=f".{output.name}.tmp-", dir=root)
try:
    os.fchmod(fd, 0o600)
    with os.fdopen(fd, "wb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.link(name, output)
    directory_fd = os.open(root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
finally:
    if os.path.exists(name):
        os.unlink(name)
PY
}

# The durable held plan closes the SIGKILL/host-failure discovery window: every
# submitted ID is recoverable before the first job is released.
publish_transaction_record held

for JOB_ID in "${JOB_IDS[@]}"; do
  scontrol release "$JOB_ID"
done

publish_transaction_record released

released=1
trap - EXIT HUP INT TERM
printf '%s\n' "${JOB_IDS[@]}"
