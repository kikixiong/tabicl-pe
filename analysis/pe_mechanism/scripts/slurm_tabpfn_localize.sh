#!/usr/bin/env bash
#SBATCH --job-name=pe-tabpfn-v26-localize
#SBATCH --partition=normal
#SBATCH --qos=short
#SBATCH --time=03:00:00
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
    PE_TABPFN_ROOT \
    PE_CONFIG \
    PE_OUTPUT_DIR \
    PE_RUNTIME_ROOT \
    PE_PYTHON
do
    require_absolute_path "$name"
done

if [[ ! -d "$PE_ANALYSIS_ROOT/src/pe_mechanism" ]]; then
    printf 'error: PE_ANALYSIS_ROOT must contain src/pe_mechanism\n' >&2
    exit 2
fi
if [[ ! -d "$PE_TABPFN_ROOT/src/tabpfn" ]]; then
    printf 'error: PE_TABPFN_ROOT must contain src/tabpfn\n' >&2
    exit 2
fi
if [[ ! -f "$PE_CONFIG" ]]; then
    printf 'error: PE_CONFIG must name a regular file\n' >&2
    exit 2
fi
if [[ -e "$PE_OUTPUT_DIR" ]]; then
    printf 'error: PE_OUTPUT_DIR must not exist before a fresh run\n' >&2
    exit 2
fi
if [[ ! -d "$PE_RUNTIME_ROOT" || -L "$PE_RUNTIME_ROOT" ]]; then
    printf 'error: PE_RUNTIME_ROOT must name an existing real directory\n' >&2
    exit 2
fi
if [[ ! -f "$PE_PYTHON" || ! -x "$PE_PYTHON" ]]; then
    printf 'error: PE_PYTHON must name an executable regular file\n' >&2
    exit 2
fi

analysis_git_root=$(git -C "$PE_ANALYSIS_ROOT" rev-parse --show-toplevel)
tabpfn_git_root=$(git -C "$PE_TABPFN_ROOT" rev-parse --show-toplevel)
working_dir=$(pwd -P)
config_path=$(readlink -m -- "$PE_CONFIG")
output_path=$(readlink -m -- "$PE_OUTPUT_DIR")
runtime_path=$(readlink -f -- "$PE_RUNTIME_ROOT")

reject_source_path() {
    local label=$1
    local value=$2
    local source_root
    for source_root in "$analysis_git_root" "$tabpfn_git_root"; do
        case "$value/" in
            "$source_root/"*)
                printf 'error: %s must be outside every verified source checkout\n' \
                    "$label" >&2
                exit 2
                ;;
        esac
    done
}

reject_source_path "Slurm working directory" "$working_dir"
reject_source_path "PE_CONFIG" "$config_path"
reject_source_path "PE_OUTPUT_DIR" "$output_path"
reject_source_path "PE_RUNTIME_ROOT" "$runtime_path"

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    for descriptor in 1 2; do
        stream_target=$(readlink -f "/proc/$$/fd/$descriptor" 2>/dev/null || true)
        if [[ -n "$stream_target" ]]; then
            reject_source_path "Slurm stdout/stderr" "$stream_target"
        fi
    done
fi

mkdir -p \
    "$runtime_path/home" \
    "$runtime_path/cache/huggingface" \
    "$runtime_path/cache/xdg" \
    "$runtime_path/tmp"

# Acquisition credentials must never cross into the scheduled inference run.
unset TABPFN_TOKEN HF_TOKEN HUGGING_FACE_HUB_TOKEN
export HOME="$runtime_path/home"
export HF_HOME="$runtime_path/cache/huggingface"
export XDG_CACHE_HOME="$runtime_path/cache/xdg"
export TMPDIR="$runtime_path/tmp"
export HF_HUB_OFFLINE=1
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONHASHSEED=0
export PYTHONPATH="$PE_ANALYSIS_ROOT/src:$PE_TABPFN_ROOT/src"

exec "$PE_PYTHON" -B -m pe_mechanism tabpfn-localize \
    --config "$PE_CONFIG" \
    --output-dir "$PE_OUTPUT_DIR"
