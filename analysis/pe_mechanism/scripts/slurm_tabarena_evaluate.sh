#!/usr/bin/env bash
#SBATCH --job-name=pe-tabarena
#SBATCH --partition=normal
#SBATCH --qos=short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00

set -euo pipefail

: "${PE_ANALYSIS_ROOT:?set PE_ANALYSIS_ROOT to the mechanism package root}"
: "${PE_CONFIG:?set PE_CONFIG to the private JSON config}"
: "${PE_OUTPUT_DIR:?set PE_OUTPUT_DIR outside every source checkout}"
: "${PE_PYTHON:?set PE_PYTHON to the prepared TabArena interpreter}"

for value in "$PE_ANALYSIS_ROOT" "$PE_CONFIG" "$PE_OUTPUT_DIR" "$PE_PYTHON"; do
  case "$value" in
    /*) ;;
    *) echo "all PE_* paths must be absolute" >&2; exit 2 ;;
  esac
done

analysis_git_root=$(git -C "$PE_ANALYSIS_ROOT" rev-parse --show-toplevel)
working_dir=$(pwd -P)
case "$working_dir/" in
  "$analysis_git_root/"*)
    echo "Slurm working directory must be outside the verified source checkout" >&2
    exit 2
    ;;
esac
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  for descriptor in 1 2; do
    stream_target=$(readlink -f "/proc/$$/fd/$descriptor" 2>/dev/null || true)
    case "$stream_target" in
      "$analysis_git_root"|"$analysis_git_root/"*)
        echo "Slurm stdout/stderr must be routed outside the verified source checkout" >&2
        exit 2
        ;;
    esac
  done
fi

export PYTHONHASHSEED=0
export PYTHONPATH="$PE_ANALYSIS_ROOT/src"
exec "$PE_PYTHON" -m pe_mechanism tabarena-evaluate \
  --config "$PE_CONFIG" \
  --output-dir "$PE_OUTPUT_DIR"
