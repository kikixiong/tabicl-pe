#!/usr/bin/env bash
# Submit only the audited legacy-pilot continuation edges.  Dry-run is default.
set -euo pipefail

MODE=dry-run
CONTINUATION_ID=seed42-rope479k-none-s2-6200-v1
RECEIPT=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) MODE=dry-run; shift ;;
    --submit) MODE=submit; shift ;;
    --continuation-id) CONTINUATION_ID="${2:?missing continuation ID}"; shift 2 ;;
    --receipt) RECEIPT="${2:?missing receipt path}"; shift 2 ;;
    *) echo "usage: $0 [--dry-run|--submit] [--continuation-id ID] [--receipt PATH]" >&2; exit 2 ;;
  esac
done
[[ "$CONTINUATION_ID" =~ ^[a-z0-9][a-z0-9._-]{0,95}$ ]] || {
  echo "unsafe continuation ID" >&2; exit 2;
}

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
PYTHON="$ROOT/.venv/bin/python"
VALIDATOR=("$PYTHON" "$ROOT/scripts/validate_pilot_checkpoint.py")
SOURCE_SHA="$(git -C "$ROOT" rev-parse HEAD)"
CKPT_ROOT="$ROOT/artifacts/tabiclv2-clf-identity"
RECEIPT="${RECEIPT:-$ROOT/artifacts/pilot-continuation/$CONTINUATION_ID/submission-receipt.json}"
WRAPPER="$ROOT/scripts/slurm_h100_pilot_continuation.sh"

[[ -x "$PYTHON" ]] || { echo "pilot Python environment is unavailable" >&2; exit 1; }
[[ -f "$WRAPPER" ]] || { echo "pilot continuation wrapper is missing" >&2; exit 1; }
[[ ! -e "$RECEIPT" && ! -L "$RECEIPT" ]] || {
  echo "submission receipt already exists: $RECEIPT" >&2; exit 1;
}

available_kib="$(df -Pk "$ROOT" | awk 'NR==2 {print $4}')"
reserve_kib=$((20 * 1024 * 1024))
[[ "$available_kib" =~ ^[0-9]+$ && "$available_kib" -ge "$reserve_kib" ]] || {
  echo "less than 20 GiB is available; refusing submission" >&2; exit 1;
}

"${VALIDATOR[@]}" \
  --checkpoint "$CKPT_ROOT/rope/seed-42/stage1/step-479000.ckpt" \
  --mode rope --step 479000 \
  --expected-sha256 7f36e035e586ddbd2d33fa0e0cb7f649456946569dbaeb9335a4e7907654e117 \
  --require-latest-in-dir >/dev/null
"${VALIDATOR[@]}" \
  --checkpoint "$CKPT_ROOT/none/seed-42/stage1/step-500000.ckpt" \
  --mode none --step 500000 \
  --expected-sha256 14efa93349cb7b6f7d458508012a084526d970aeb00749bbb1cebbbe38aaec87 >/dev/null
"${VALIDATOR[@]}" \
  --checkpoint "$CKPT_ROOT/none/seed-42/stage2/step-6200.ckpt" \
  --mode none --step 6200 \
  --expected-sha256 34564fe378d57d89b10e800b01ab63185170df25ad9210175f162c31749a16be \
  --require-latest-in-dir >/dev/null

ROPE_STAGE2_DIR="$CKPT_ROOT/rope/seed-42/stage2"
if [[ -d "$ROPE_STAGE2_DIR" ]] && find "$ROPE_STAGE2_DIR" -maxdepth 1 -type f -name 'step-*.ckpt' -print -quit | grep -q .; then
  echo "RoPE Stage 2 child namespace is not fresh" >&2
  exit 1
fi

COMMON=(
  --parsable --partition=h100 --qos=long --nodes=1 --gres=gpu:1
  --cpus-per-task=64 --mem=128G
)
export_value() {
  local action="$1"
  printf 'ALL,PILOT_ROOT=%s,PILOT_ACTION=%s,PILOT_CONTINUATION_ID=%s,PILOT_SOURCE_SHA=%s' \
    "$ROOT" "$action" "$CONTINUATION_ID" "$SOURCE_SHA"
}
render() {
  printf '%q ' "$@"
  printf '\n'
}

if [[ "$MODE" == dry-run ]]; then
  echo "Dry run only; no Slurm jobs or artifacts will be created."
  render sbatch "${COMMON[@]}" \
    --job-name=tabicl-pilot-rope-s1-500k \
    --output="$ROOT/artifacts/logs/pilot-${CONTINUATION_ID}-rope-s1-%j.out" \
    --error="$ROOT/artifacts/logs/pilot-${CONTINUATION_ID}-rope-s1-%j.err" \
    --export="$(export_value rope-stage1-479k-to-500k)" "$WRAPPER"
  render sbatch "${COMMON[@]}" --dependency='afterok:<rope-stage1-job>' \
    --job-name=tabicl-pilot-rope-s2-40k \
    --output="$ROOT/artifacts/logs/pilot-${CONTINUATION_ID}-rope-s2-%j.out" \
    --error="$ROOT/artifacts/logs/pilot-${CONTINUATION_ID}-rope-s2-%j.err" \
    --export="$(export_value rope-stage2-after-500k)" "$WRAPPER"
  render sbatch "${COMMON[@]}" \
    --job-name=tabicl-pilot-none-s2-40k \
    --output="$ROOT/artifacts/logs/pilot-${CONTINUATION_ID}-none-s2-%j.out" \
    --error="$ROOT/artifacts/logs/pilot-${CONTINUATION_ID}-none-s2-%j.err" \
    --export="$(export_value none-stage2-6200-to-40k)" "$WRAPPER"
  echo "Receipt would be written to: $RECEIPT"
  exit 0
fi

[[ -z "$(git -C "$ROOT" status --porcelain=v1 --untracked-files=all)" ]] || {
  echo "actual submission requires a clean committed checkout" >&2; exit 1;
}
if squeue -u "$USER" -h -o '%j' | grep -Eq '^tabicl-pilot-(rope-s1-500k|rope-s2-40k|none-s2-40k)$'; then
  echo "a targeted pilot-continuation job name already exists for this user" >&2
  exit 1
fi

mkdir -p "$ROOT/artifacts/logs" "$(dirname "$RECEIPT")"
created_jobs=()
transaction_complete=false
rollback() {
  local status=$?
  trap - ERR INT TERM
  if [[ "$transaction_complete" != true && ${#created_jobs[@]} -gt 0 ]]; then
    echo "submission failed; cancelling only jobs created by this invocation: ${created_jobs[*]}" >&2
    scancel "${created_jobs[@]}" || true
  fi
  exit "$status"
}
trap rollback ERR INT TERM

submit_one() {
  local raw
  raw="$(sbatch "$@")"
  raw="${raw%%;*}"
  [[ "$raw" =~ ^[1-9][0-9]*$ ]] || {
    echo "sbatch returned an invalid job ID: $raw" >&2
    return 1
  }
  created_jobs+=("$raw")
  LAST_JOB_ID="$raw"
}

LAST_JOB_ID=""
submit_one "${COMMON[@]}" \
  --job-name=tabicl-pilot-rope-s1-500k \
  --output="$ROOT/artifacts/logs/pilot-${CONTINUATION_ID}-rope-s1-%j.out" \
  --error="$ROOT/artifacts/logs/pilot-${CONTINUATION_ID}-rope-s1-%j.err" \
  --export="$(export_value rope-stage1-479k-to-500k)" "$WRAPPER"
rope_stage1="$LAST_JOB_ID"
submit_one "${COMMON[@]}" --dependency="afterok:$rope_stage1" \
  --job-name=tabicl-pilot-rope-s2-40k \
  --output="$ROOT/artifacts/logs/pilot-${CONTINUATION_ID}-rope-s2-%j.out" \
  --error="$ROOT/artifacts/logs/pilot-${CONTINUATION_ID}-rope-s2-%j.err" \
  --export="$(export_value rope-stage2-after-500k)" "$WRAPPER"
rope_stage2="$LAST_JOB_ID"
submit_one "${COMMON[@]}" \
  --job-name=tabicl-pilot-none-s2-40k \
  --output="$ROOT/artifacts/logs/pilot-${CONTINUATION_ID}-none-s2-%j.out" \
  --error="$ROOT/artifacts/logs/pilot-${CONTINUATION_ID}-none-s2-%j.err" \
  --export="$(export_value none-stage2-6200-to-40k)" "$WRAPPER"
none_stage2="$LAST_JOB_ID"

"$PYTHON" "$ROOT/scripts/write_pilot_continuation_receipt.py" \
  --output "$RECEIPT" --continuation-id "$CONTINUATION_ID" \
  --source-commit "$SOURCE_SHA" --repository-root "$ROOT" \
  --rope-stage1-job "$rope_stage1" --rope-stage2-job "$rope_stage2" \
  --none-stage2-job "$none_stage2" >/dev/null
transaction_complete=true
trap - ERR INT TERM
printf 'rope_stage1=%s rope_stage2=%s none_stage2=%s receipt=%s\n' \
  "$rope_stage1" "$rope_stage2" "$none_stage2" "$RECEIPT"
