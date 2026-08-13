#!/usr/bin/env bash
#SBATCH --job-name=pe-fingerprint-tabarena
#SBATCH --partition=normal
#SBATCH --qos=short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=03:00:00

set -euo pipefail

required=(
  PE_ANALYSIS_ROOT PE_MODEL_ROOT PE_TABARENA_ROOT PE_OPENML_CACHE
  PE_ROPE_CHECKPOINT PE_ROPE_SHA256 PE_FINGERPRINT_CHECKPOINT
  PE_FINGERPRINT_SHA256 PE_RELEASED_CHECKPOINT PE_OUTPUT_DIR PE_PYTHON
)
for name in "${required[@]}"; do
  [[ -n "${!name:-}" ]] || { echo "missing $name" >&2; exit 2; }
done

export PYTHONHASHSEED=0
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PE_ANALYSIS_ROOT/analysis/pe_mechanism/src"
exec "$PE_PYTHON" \
  "$PE_ANALYSIS_ROOT/analysis/pe_mechanism/scripts/run_fingerprint_tabarena_exploratory.py" \
  --analysis-root "$PE_ANALYSIS_ROOT" \
  --model-root "$PE_MODEL_ROOT" \
  --tabarena-root "$PE_TABARENA_ROOT" \
  --openml-cache "$PE_OPENML_CACHE" \
  --rope-checkpoint "$PE_ROPE_CHECKPOINT" --rope-sha256 "$PE_ROPE_SHA256" \
  --fingerprint-checkpoint "$PE_FINGERPRINT_CHECKPOINT" \
  --fingerprint-sha256 "$PE_FINGERPRINT_SHA256" \
  --released-checkpoint "$PE_RELEASED_CHECKPOINT" \
  --output-dir "$PE_OUTPUT_DIR"
