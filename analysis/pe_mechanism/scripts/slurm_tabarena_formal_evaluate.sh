#!/usr/bin/env bash
#SBATCH --job-name=pe-tabarena-formal
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

for name in PE_ANALYSIS_ROOT PE_CONFIG PE_OUTPUT_DIR PE_RUNTIME_ROOT PE_PYTHON; do
    require_absolute_path "$name"
done

if [[ ! -d "$PE_ANALYSIS_ROOT/src/pe_mechanism" ]]; then
    printf 'error: PE_ANALYSIS_ROOT must contain src/pe_mechanism\n' >&2
    exit 2
fi
if [[ ! -f "$PE_CONFIG" || -L "$PE_CONFIG" ]]; then
    printf 'error: PE_CONFIG must name a regular non-symlink file\n' >&2
    exit 2
fi
if [[ -e "$PE_OUTPUT_DIR" ]]; then
    printf 'error: PE_OUTPUT_DIR must not exist before a fresh run\n' >&2
    exit 2
fi
if [[ -e "$PE_RUNTIME_ROOT" || -L "$PE_RUNTIME_ROOT" ]]; then
    printf 'error: PE_RUNTIME_ROOT must not exist before a fresh run\n' >&2
    exit 2
fi
if [[ ! -f "$PE_PYTHON" || ! -x "$PE_PYTHON" ]]; then
    printf 'error: PE_PYTHON must name an executable regular file\n' >&2
    exit 2
fi

analysis_git_root=$(git -C "$PE_ANALYSIS_ROOT" rev-parse --show-toplevel)
working_dir=$(pwd -P)
config_path=$(readlink -f -- "$PE_CONFIG")
output_path=$(readlink -m -- "$PE_OUTPUT_DIR")
runtime_path=$(readlink -m -- "$PE_RUNTIME_ROOT")

reject_source_path() {
    local label=$1
    local value=$2
    case "$value/" in
        "$analysis_git_root/"*)
            printf 'error: %s must be outside the analysis checkout\n' "$label" >&2
            exit 2
            ;;
    esac
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

gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader | sed -n '1p')
if [[ "$gpu_name" != "NVIDIA A10" ]]; then
    printf 'error: formal TabArena evaluation requires NVIDIA A10, got %s\n' \
        "$gpu_name" >&2
    exit 2
fi

mkdir -p \
    "$runtime_path/home" \
    "$runtime_path/cache/huggingface" \
    "$runtime_path/cache/xdg" \
    "$runtime_path/tmp"

unset TABPFN_TOKEN HF_TOKEN HUGGING_FACE_HUB_TOKEN WANDB_API_KEY
export HOME="$runtime_path/home"
export HF_HOME="$runtime_path/cache/huggingface"
export XDG_CACHE_HOME="$runtime_path/cache/xdg"
export TMPDIR="$runtime_path/tmp"
export HF_HUB_OFFLINE=1
export WANDB_MODE=disabled
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONHASHSEED=0
export PYTHONPATH="$PE_ANALYSIS_ROOT/src"

exec "$PE_PYTHON" -B -m pe_mechanism tabarena-formal-evaluate \
    --config "$PE_CONFIG" \
    --output-dir "$PE_OUTPUT_DIR"
