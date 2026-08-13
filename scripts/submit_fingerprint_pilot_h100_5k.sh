#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
: "${SOURCE_ROOT:?SOURCE_ROOT must be a clean detached checkout of the pushed commit}"
: "${PILOT_ARTIFACT_ROOT:?PILOT_ARTIFACT_ROOT must be an absolute external artifact directory}"
: "${PYTHON:?PYTHON must name the validated absolute interpreter}"
[[ "$SOURCE_ROOT" == /* ]] || { echo "SOURCE_ROOT must be absolute" >&2; exit 2; }
[[ "$PILOT_ARTIFACT_ROOT" == /* ]] || { echo "PILOT_ARTIFACT_ROOT must be absolute" >&2; exit 2; }
[[ "$PYTHON" == /* && -x "$PYTHON" ]] || { echo "invalid PYTHON" >&2; exit 2; }

EXPECTED_SOURCE_SHA="$(git -C "$SOURCE_ROOT" rev-parse HEAD)"
[[ "$EXPECTED_SOURCE_SHA" =~ ^[0-9a-f]{40}$ ]] || { echo "invalid source SHA" >&2; exit 1; }
[[ -z "$(git -C "$SOURCE_ROOT" status --porcelain --untracked-files=all)" ]] || {
  echo "SOURCE_ROOT must be clean" >&2
  exit 1
}
git -C "$SOURCE_ROOT" merge-base --is-ancestor "$EXPECTED_SOURCE_SHA" origin/codex/fingerprint-setbind-pilot || {
  echo "source commit is not present on the public pilot branch" >&2
  exit 1
}

LOG_ROOT="$PILOT_ARTIFACT_ROOT/logs"
mkdir -p "$LOG_ROOT"
JOB_ID="$({ sbatch --parsable \
  --output="$LOG_ROOT/%A_%a.out" \
  --error="$LOG_ROOT/%A_%a.err" \
  --export="ALL,SOURCE_ROOT=$SOURCE_ROOT,EXPECTED_SOURCE_SHA=$EXPECTED_SOURCE_SHA,PILOT_ARTIFACT_ROOT=$PILOT_ARTIFACT_ROOT,PYTHON=$PYTHON" \
  "$ROOT/scripts/slurm_fingerprint_pilot_h100_5k_array.sh"; } | cut -d';' -f1)"
[[ "$JOB_ID" =~ ^[0-9]+$ ]] || { echo "sbatch returned invalid job id: $JOB_ID" >&2; exit 1; }
printf '%s\n' "$JOB_ID"
