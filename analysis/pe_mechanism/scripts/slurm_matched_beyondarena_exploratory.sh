#!/usr/bin/env bash
#SBATCH --job-name=pe-matched-beyond
#SBATCH --partition=normal
#SBATCH --qos=short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=03:00:00

set -euo pipefail

required=(
  PE_ANALYSIS_ROOT PE_MODEL_ROOT PE_TABARENA_ROOT PE_BEYONDARENA_CACHE
  PE_MATCHED_MANIFEST PE_OUTPUT_DIR PE_PYTHON PE_EXPECTED_ANALYSIS_SHA
  PE_EXPECTED_MODEL_SHA PE_EXPECTED_TABARENA_SHA
)
for name in "${required[@]}"; do
  [[ -n "${!name:-}" ]] || { echo "missing $name" >&2; exit 2; }
done

dataset_args=()
while (( $# > 0 )); do
  if [[ "$1" != "--dataset-name" ]] || (( $# < 2 )) || [[ -z "$2" ]] \
    || [[ "$2" == -* ]]; then
    echo "only repeated --dataset-name NAME pairs are supported" >&2
    exit 2
  fi
  dataset_args+=("--dataset-name" "$2")
  shift 2
done

released_args=()
if [[ -n "${PE_RELEASED_CHECKPOINT:-}" ]]; then
  released_args=(--released-checkpoint "$PE_RELEASED_CHECKPOINT")
fi

export PYTHONHASHSEED=0
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PE_ANALYSIS_ROOT/analysis/pe_mechanism/src"
exec "$PE_PYTHON" -B \
  "$PE_ANALYSIS_ROOT/analysis/pe_mechanism/scripts/run_matched_beyondarena_exploratory.py" \
  --analysis-root "$PE_ANALYSIS_ROOT" \
  --model-root "$PE_MODEL_ROOT" \
  --tabarena-root "$PE_TABARENA_ROOT" \
  --cache-root "$PE_BEYONDARENA_CACHE" \
  --snapshot-manifest "$PE_MATCHED_MANIFEST" \
  --expected-analysis-sha "$PE_EXPECTED_ANALYSIS_SHA" \
  --expected-model-sha "$PE_EXPECTED_MODEL_SHA" \
  --expected-tabarena-sha "$PE_EXPECTED_TABARENA_SHA" \
  --output-dir "$PE_OUTPUT_DIR" \
  "${released_args[@]}" \
  "${dataset_args[@]}"
