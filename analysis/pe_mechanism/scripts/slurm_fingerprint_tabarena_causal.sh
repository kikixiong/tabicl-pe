#!/usr/bin/env bash
#SBATCH --job-name=pe-fingerprint-causal
#SBATCH --partition=normal
#SBATCH --qos=short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=03:00:00

set -euo pipefail

required=(
  PE_ANALYSIS_ROOT PE_MODEL_ROOT PE_TABARENA_ROOT PE_OPENML_CACHE
  PE_FINGERPRINT_CHECKPOINT PE_FINGERPRINT_SHA256 PE_OUTPUT_DIR PE_PYTHON
  PE_EXPECTED_ANALYSIS_SHA PE_EXPECTED_MODEL_SHA PE_EXPECTED_TABARENA_SHA
)
for name in "${required[@]}"; do
  [[ -n "${!name:-}" ]] || { echo "missing $name" >&2; exit 2; }
done

scratch_parent="${TMPDIR:?TMPDIR must name a pre-existing private scratch directory}"
[[ -d "$scratch_parent" && ! -L "$scratch_parent" ]] || {
  echo "TMPDIR must be a real directory" >&2
  exit 2
}
scratch_parent="$(cd -- "$scratch_parent" && pwd -P)"
job_token="${SLURM_JOB_ID:-manual-$$}"
[[ "$job_token" =~ ^[A-Za-z0-9._-]+$ ]] || {
  echo "unsafe Slurm job identifier" >&2
  exit 2
}
job_tmp="$scratch_parent/fingerprint-causal-$job_token"
[[ "$job_tmp" == "$scratch_parent"/fingerprint-causal-* ]] || exit 2
[[ ! -e "$job_tmp" ]] || { echo "job scratch already exists: $job_tmp" >&2; exit 2; }
mkdir -m 700 -- "$job_tmp"
cleanup() {
  if [[ -n "${job_tmp:-}" && "$job_tmp" == "$scratch_parent"/fingerprint-causal-* ]]; then
    rm -rf -- "$job_tmp"
  fi
}
trap cleanup EXIT HUP INT TERM

export TMPDIR="$job_tmp"
export TMP="$job_tmp"
export TEMP="$job_tmp"
export PYTHONHASHSEED=0
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PE_ANALYSIS_ROOT/analysis/pe_mechanism/src"

command=(
  "$PE_PYTHON"
  "$PE_ANALYSIS_ROOT/analysis/pe_mechanism/scripts/run_fingerprint_tabarena_causal.py"
  --analysis-root "$PE_ANALYSIS_ROOT"
  --model-root "$PE_MODEL_ROOT"
  --tabarena-root "$PE_TABARENA_ROOT"
  --openml-cache "$PE_OPENML_CACHE"
  --fingerprint-checkpoint "$PE_FINGERPRINT_CHECKPOINT"
  --fingerprint-sha256 "$PE_FINGERPRINT_SHA256"
  --expected-analysis-sha "$PE_EXPECTED_ANALYSIS_SHA"
  --expected-model-sha "$PE_EXPECTED_MODEL_SHA"
  --expected-tabarena-sha "$PE_EXPECTED_TABARENA_SHA"
  --output-dir "$PE_OUTPUT_DIR"
)
if [[ -n "${PE_DATASET:-}" ]]; then
  command+=(--dataset "$PE_DATASET")
fi
"${command[@]}"
