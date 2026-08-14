#!/usr/bin/env bash
#SBATCH --job-name=pe-fingerprint-talent
#SBATCH --partition=normal
#SBATCH --qos=medium
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00

set -euo pipefail

required=(
  PE_ANALYSIS_ROOT PE_MODEL_ROOT PE_TALENT_ROOT PE_ROPE_CHECKPOINT
  PE_ROPE_SHA256 PE_FINGERPRINT_CHECKPOINT PE_FINGERPRINT_SHA256
  PE_RELEASED_CHECKPOINT PE_OUTPUT_DIR PE_PYTHON PE_COMPARISON_STEP
  PE_EXPECTED_MODEL_SHA PE_EXPECTED_ANALYSIS_SHA
)
for name in "${required[@]}"; do
  [[ -n "${!name:-}" ]] || { echo "missing $name" >&2; exit 2; }
done

lineage_args=()
if [[ -n "${PE_SNAPSHOT_RECEIPT:-}" ]]; then
  for name in PE_SUBMISSION_RECEIPT PE_ROPE_LAUNCH_RECEIPT \
    PE_ROPE_COMPLETION_RECEIPT PE_FINGERPRINT_LAUNCH_RECEIPT \
    PE_FINGERPRINT_COMPLETION_RECEIPT; do
    [[ -z "${!name:-}" ]] || { echo "$name conflicts with PE_SNAPSHOT_RECEIPT" >&2; exit 2; }
  done
  lineage_args=(--snapshot-receipt "$PE_SNAPSHOT_RECEIPT")
else
  for name in PE_SUBMISSION_RECEIPT PE_ROPE_LAUNCH_RECEIPT \
    PE_ROPE_COMPLETION_RECEIPT PE_FINGERPRINT_LAUNCH_RECEIPT \
    PE_FINGERPRINT_COMPLETION_RECEIPT; do
    [[ -n "${!name:-}" ]] || { echo "missing $name" >&2; exit 2; }
  done
  lineage_args=(
    --submission-receipt "$PE_SUBMISSION_RECEIPT"
    --rope-launch-receipt "$PE_ROPE_LAUNCH_RECEIPT"
    --rope-completion-receipt "$PE_ROPE_COMPLETION_RECEIPT"
    --fingerprint-launch-receipt "$PE_FINGERPRINT_LAUNCH_RECEIPT"
    --fingerprint-completion-receipt "$PE_FINGERPRINT_COMPLETION_RECEIPT"
  )
fi

dataset_args=()
while (( $# > 0 )); do
  if [[ "$1" != "--dataset" ]]; then
    echo "unsupported argument: $1 (only repeated '--dataset NAME' pairs are allowed)" >&2
    exit 2
  fi
  if (( $# < 2 )) || [[ -z "$2" ]] || [[ "$2" == -* ]]; then
    echo "--dataset requires a non-empty NAME that does not begin with '-'" >&2
    exit 2
  fi
  dataset_args+=("--dataset" "$2")
  shift 2
done

export PYTHONHASHSEED=0
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$PE_MODEL_ROOT/src:$PE_ANALYSIS_ROOT/analysis/pe_mechanism/src"

exec "$PE_PYTHON" -B \
  "$PE_ANALYSIS_ROOT/analysis/pe_mechanism/scripts/run_fingerprint_talent_exploratory.py" \
  --analysis-root "$PE_ANALYSIS_ROOT" \
  --model-root "$PE_MODEL_ROOT" \
  --talent-root "$PE_TALENT_ROOT" \
  --rope-checkpoint "$PE_ROPE_CHECKPOINT" \
  --rope-sha256 "$PE_ROPE_SHA256" \
  --fingerprint-checkpoint "$PE_FINGERPRINT_CHECKPOINT" \
  --fingerprint-sha256 "$PE_FINGERPRINT_SHA256" \
  --released-checkpoint "$PE_RELEASED_CHECKPOINT" \
  --comparison-step "$PE_COMPARISON_STEP" \
  --expected-model-sha "$PE_EXPECTED_MODEL_SHA" \
  --expected-analysis-sha "$PE_EXPECTED_ANALYSIS_SHA" \
  "${lineage_args[@]}" \
  --output-dir "$PE_OUTPUT_DIR" \
  "${dataset_args[@]}"
