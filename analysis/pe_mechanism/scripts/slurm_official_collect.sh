#!/usr/bin/env bash
#SBATCH --job-name=pe-official-collect
#SBATCH --partition=h100
#SBATCH --qos=short
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

set -euo pipefail

require_absolute_path() {
    local name=$1
    local value=${!name-}
    if [[ -z "$value" || "$value" != /* ]]; then
        printf 'error: %s must be a non-empty absolute path\n' "$name" >&2
        exit 2
    fi
}

for name in \
    PE_ANALYSIS_ROOT \
    PE_MODEL_ROOT \
    PE_CONFIG \
    PE_OUTPUT_DIR \
    PE_PYTHON
do
    require_absolute_path "$name"
done

if [[ ! -d "$PE_ANALYSIS_ROOT/src/pe_mechanism" ]]; then
    printf 'error: PE_ANALYSIS_ROOT must contain src/pe_mechanism\n' >&2
    exit 2
fi
if [[ ! -d "$PE_MODEL_ROOT/src/tabicl" ]]; then
    printf 'error: PE_MODEL_ROOT must contain src/tabicl\n' >&2
    exit 2
fi
if [[ ! -f "$PE_CONFIG" ]]; then
    printf 'error: PE_CONFIG must name a regular file\n' >&2
    exit 2
fi
if [[ ! -f "$PE_PYTHON" || ! -x "$PE_PYTHON" ]]; then
    printf 'error: PE_PYTHON must name an executable regular file\n' >&2
    exit 2
fi

export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
# The bound TabICL revision groups normalization views through a Python set.
# Fixing the interpreter hash seed makes that official view order reproducible
# across separate condition jobs; train-repr still verifies the actual schedule.
export PYTHONHASHSEED=0
export PYTHONPATH="$PE_ANALYSIS_ROOT/src:$PE_MODEL_ROOT/src"

exec "$PE_PYTHON" -B -m pe_mechanism official-collect \
    --config "$PE_CONFIG" \
    --output-dir "$PE_OUTPUT_DIR"
