#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
: "${SOURCE_ROOT:?SOURCE_ROOT must be a clean detached checkout of the pushed commit}"
: "${PILOT_ARTIFACT_ROOT:?PILOT_ARTIFACT_ROOT must be a fresh absolute external directory}"
: "${PYTHON:?PYTHON must name the validated absolute interpreter}"
MAX_STEPS="${MAX_STEPS:-5000}"
case "$MAX_STEPS" in 1|5000) ;; *) echo "MAX_STEPS must be 1 or 5000" >&2; exit 2 ;; esac
[[ "$SOURCE_ROOT" == /* ]] || { echo "SOURCE_ROOT must be absolute" >&2; exit 2; }
[[ "$PILOT_ARTIFACT_ROOT" == /* ]] || { echo "PILOT_ARTIFACT_ROOT must be absolute" >&2; exit 2; }
[[ "$PYTHON" == /* && -x "$PYTHON" ]] || { echo "invalid PYTHON" >&2; exit 2; }
[[ ! -e "$PILOT_ARTIFACT_ROOT" ]] || { echo "PILOT_ARTIFACT_ROOT must be fresh" >&2; exit 1; }

EXPECTED_SOURCE_SHA="$(git -C "$SOURCE_ROOT" rev-parse HEAD)"
[[ "$EXPECTED_SOURCE_SHA" =~ ^[0-9a-f]{40}$ ]] || { echo "invalid source SHA" >&2; exit 1; }
[[ -z "$(git -C "$SOURCE_ROOT" status --porcelain --untracked-files=all)" ]] || {
  echo "SOURCE_ROOT must be clean" >&2
  exit 1
}
REMOTE_SHA="$(git -C "$SOURCE_ROOT" ls-remote --refs origin refs/heads/codex/fingerprint-setbind-pilot | awk 'NR == 1 {print $1}')"
[[ "$REMOTE_SHA" == "$EXPECTED_SOURCE_SHA" ]] || {
  echo "exact source commit is not advertised by the public pilot branch" >&2
  exit 1
}

if [[ "$MAX_STEPS" -eq 1 ]]; then
  QOS=short
  TIME_LIMIT=00:30:00
else
  QOS=medium
  TIME_LIMIT=08:00:00
fi

umask 077
mkdir "$PILOT_ARTIFACT_ROOT"
mkdir "$PILOT_ARTIFACT_ROOT/logs"

declare -a JOB_IDS=()
released=0
rollback() {
  status=$?
  if [[ "$released" -eq 0 && "${#JOB_IDS[@]}" -gt 0 ]]; then
    for ((index=${#JOB_IDS[@]}-1; index>=0; index--)); do
      scancel "${JOB_IDS[$index]}" 2>/dev/null || true
    done
  fi
  exit "$status"
}
trap rollback ERR INT TERM HUP

for ARM in rope fingerprint; do
  JOB_ID="$({ sbatch --parsable --hold \
    --job-name="tabicl-full-${ARM}-${MAX_STEPS}" \
    --qos="$QOS" --time="$TIME_LIMIT" \
    --output="$PILOT_ARTIFACT_ROOT/logs/${ARM}-%j.out" \
    --error="$PILOT_ARTIFACT_ROOT/logs/${ARM}-%j.err" \
    --export="ALL,ARM=$ARM,SOURCE_ROOT=$SOURCE_ROOT,EXPECTED_SOURCE_SHA=$EXPECTED_SOURCE_SHA,PILOT_ARTIFACT_ROOT=$PILOT_ARTIFACT_ROOT,PYTHON=$PYTHON,MAX_STEPS=$MAX_STEPS,SEED=42" \
    "$ROOT/scripts/slurm_fingerprint_fullsize_h100_pilot.sh"; } | cut -d';' -f1)"
  [[ "$JOB_ID" =~ ^[0-9]+$ ]] || { echo "sbatch returned invalid job id: $JOB_ID" >&2; exit 1; }
  JOB_IDS+=("$JOB_ID")
done

for JOB_ID in "${JOB_IDS[@]}"; do
  scontrol release "$JOB_ID"
done
released=1
trap - ERR INT TERM HUP

export EXPECTED_SOURCE_SHA MAX_STEPS QOS TIME_LIMIT PILOT_ARTIFACT_ROOT
export ROPE_JOB_ID="${JOB_IDS[0]}"
export FINGERPRINT_JOB_ID="${JOB_IDS[1]}"
"$PYTHON" - <<'PY'
import json
import os
from pathlib import Path

record = {
    "schema_version": 1,
    "study": "tabiclv2-fullsize-rope-fingerprint-pilot-v1",
    "formal_evidence": False,
    "fresh_from_scratch": True,
    "source_commit": os.environ["EXPECTED_SOURCE_SHA"],
    "seed": 42,
    "max_steps": int(os.environ["MAX_STEPS"]),
    "qos": os.environ["QOS"],
    "time_limit": os.environ["TIME_LIMIT"],
    "jobs": {
        "rope": os.environ["ROPE_JOB_ID"],
        "fingerprint": os.environ["FINGERPRINT_JOB_ID"],
    },
}
path = Path(os.environ["PILOT_ARTIFACT_ROOT"]) / "submission.json"
tmp = path.with_suffix(".json.tmp")
tmp.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
tmp.replace(path)
PY
printf '%s\n' "${JOB_IDS[@]}"
