#!/usr/bin/env bash
#SBATCH --job-name=pe-talent-pair
#SBATCH --partition=h100
#SBATCH --qos=medium
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00

set -euo pipefail

(( $# == 0 )) || { printf 'error: positional arguments are not supported\n' >&2; exit 2; }

required=(
  PE_TALENT_ANALYSIS_ROOT PE_TALENT_MODEL_ROOT PE_TALENT_DATA_ROOT
  PE_TALENT_RUN_CONFIG PE_TALENT_SHARD_PLAN PE_TALENT_OUTPUT_ROOT
  PE_TALENT_PYTHON PE_TALENT_EXPECTED_ANALYSIS_SHA
  PE_TALENT_EXPECTED_MODEL_SHA PE_TALENT_SCRATCH_ROOT
  PE_TALENT_GPU_CSV_ROOT PE_TALENT_EXPECTED_RUN_CONFIG_SHA256
  PE_TALENT_EXPECTED_SHARD_PLAN_SHA256
)
for name in "${required[@]}"; do
  [[ -n "${!name:-}" ]] || { printf 'error: missing %s\n' "$name" >&2; exit 2; }
done
for name in \
  PE_TALENT_ANALYSIS_ROOT PE_TALENT_MODEL_ROOT PE_TALENT_DATA_ROOT \
  PE_TALENT_RUN_CONFIG PE_TALENT_SHARD_PLAN PE_TALENT_OUTPUT_ROOT \
  PE_TALENT_PYTHON PE_TALENT_SCRATCH_ROOT PE_TALENT_GPU_CSV_ROOT; do
  [[ "${!name}" == /* ]] || {
    printf 'error: %s must be absolute\n' "$name" >&2
    exit 2
  }
done

canary_args=()
if [[ "${PE_TALENT_CANARY:-0}" == "1" ]]; then
  [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]] || {
    printf 'error: canary must not be an array task\n' >&2
    exit 2
  }
  shard_id=canary
  execution_scope=canary_largest_two_plus_madeline
  canary_args=(--canary)
else
  [[ "${PE_TALENT_CANARY:-0}" == "0" ]] || {
    printf 'error: PE_TALENT_CANARY must be 0 or 1\n' >&2
    exit 2
  }
  shard_value=${PE_TALENT_SHARD_ID:-${SLURM_ARRAY_TASK_ID:-}}
  [[ "$shard_value" =~ ^[0-9]+$ ]] || {
    printf 'error: PE_TALENT_SHARD_ID or SLURM_ARRAY_TASK_ID must be numeric\n' >&2
    exit 2
  }
  (( 10#$shard_value <= 999 )) || {
    printf 'error: shard ID exceeds 999\n' >&2
    exit 2
  }
  printf -v shard_id '%03d' "$((10#$shard_value))"
  execution_scope=full_planned_shard
fi
[[ "$PE_TALENT_EXPECTED_ANALYSIS_SHA" =~ ^[0-9a-f]{40}$ ]] || exit 2
[[ "$PE_TALENT_EXPECTED_MODEL_SHA" =~ ^[0-9a-f]{40}$ ]] || exit 2
[[ "$PE_TALENT_EXPECTED_RUN_CONFIG_SHA256" =~ ^[0-9a-f]{64}$ ]] || exit 2
[[ "$PE_TALENT_EXPECTED_SHARD_PLAN_SHA256" =~ ^[0-9a-f]{64}$ ]] || exit 2

for directory in \
  "$PE_TALENT_ANALYSIS_ROOT" "$PE_TALENT_MODEL_ROOT" "$PE_TALENT_DATA_ROOT" \
  "$PE_TALENT_OUTPUT_ROOT" "$PE_TALENT_SCRATCH_ROOT" \
  "$PE_TALENT_GPU_CSV_ROOT"; do
  [[ -d "$directory" && ! -L "$directory" ]] || {
    printf 'error: directory must exist and not be a symlink: %s\n' "$directory" >&2
    exit 2
  }
done
for file in "$PE_TALENT_RUN_CONFIG" "$PE_TALENT_SHARD_PLAN"; do
  [[ -f "$file" && ! -L "$file" ]] || {
    printf 'error: input must be a regular non-symlink file: %s\n' "$file" >&2
    exit 2
  }
done
[[ -x "$PE_TALENT_PYTHON" && ! -L "$PE_TALENT_PYTHON" ]] || {
  printf 'error: PE_TALENT_PYTHON must be a real executable\n' >&2
  exit 2
}

gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader | sed -n '1p')
[[ "$gpu_name" == *H100* ]] || {
  printf 'error: TALENT full-suite evaluation requires H100, got %s\n' "$gpu_name" >&2
  exit 2
}

scratch=$(mktemp -d \
  "$PE_TALENT_SCRATCH_ROOT/talent-pair-${SLURM_JOB_ID:-manual}-${shard_id}.XXXXXX")
monitor_pid=''
cleanup() {
  if [[ -n "$monitor_pid" ]]; then
    kill "$monitor_pid" 2>/dev/null || true
    wait "$monitor_pid" 2>/dev/null || true
  fi
  if [[ -n "${gpu_csv_partial:-}" ]]; then
    rm -f -- "$gpu_csv_partial"
  fi
  rm -rf -- "$scratch"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

export SLURM_TMPDIR="$scratch"
export TMPDIR="$scratch"
export TMP="$scratch"
export TEMP="$scratch"
export HF_HOME="$scratch/huggingface"
export XDG_CACHE_HOME="$scratch/xdg-cache"
mkdir -p "$HF_HOME" "$XDG_CACHE_HOME"
export PYTHONHASHSEED=0
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=disabled
export PYTHONPATH="$PE_TALENT_MODEL_ROOT/src:$PE_TALENT_ANALYSIS_ROOT/analysis/pe_mechanism/src"

gpu_csv="$PE_TALENT_GPU_CSV_ROOT/shard-${shard_id}.csv"
gpu_csv_partial="$gpu_csv.partial"
[[ ! -e "$gpu_csv" && ! -L "$gpu_csv" && ! -e "$gpu_csv_partial" && ! -L "$gpu_csv_partial" ]] || {
  printf 'error: GPU monitor CSV already exists: %s\n' "$gpu_csv" >&2
  exit 2
}
(
  printf 'unix_seconds,run_config_sha256,analysis_sha,model_sha,shard_id,execution_scope,index,name,utilization_gpu_percent,memory_used_mib,memory_total_mib,power_draw_watts\n'
  while true; do
    timestamp=$(date +%s)
    nvidia-smi \
      --query-gpu=index,name,utilization.gpu,memory.used,memory.total,power.draw \
      --format=csv,noheader,nounits | \
      sed "s/^/${timestamp},${PE_TALENT_EXPECTED_RUN_CONFIG_SHA256},${PE_TALENT_EXPECTED_ANALYSIS_SHA},${PE_TALENT_EXPECTED_MODEL_SHA},${shard_id},${execution_scope},/"
    sleep 30
  done
) >"$gpu_csv_partial" &
monitor_pid=$!

"$PE_TALENT_PYTHON" -B \
  "$PE_TALENT_ANALYSIS_ROOT/analysis/pe_mechanism/scripts/run_talent_paired_full_suite.py" \
  --analysis-root "$PE_TALENT_ANALYSIS_ROOT" \
  --model-root "$PE_TALENT_MODEL_ROOT" \
  --talent-root "$PE_TALENT_DATA_ROOT" \
  --run-config "$PE_TALENT_RUN_CONFIG" \
  --shard-plan "$PE_TALENT_SHARD_PLAN" \
  --shard-id "$shard_id" \
  --expected-analysis-sha "$PE_TALENT_EXPECTED_ANALYSIS_SHA" \
  --expected-model-sha "$PE_TALENT_EXPECTED_MODEL_SHA" \
  --expected-run-config-sha256 "$PE_TALENT_EXPECTED_RUN_CONFIG_SHA256" \
  --expected-shard-plan-sha256 "$PE_TALENT_EXPECTED_SHARD_PLAN_SHA256" \
  --scratch-root "$scratch" \
  "${canary_args[@]}" \
  --output-dir "$PE_TALENT_OUTPUT_ROOT/shard-$shard_id"

kill -0 "$monitor_pid" 2>/dev/null || {
  printf 'error: GPU monitor exited before evaluator completion\n' >&2
  exit 1
}
kill "$monitor_pid"
wait "$monitor_pid" 2>/dev/null || true
monitor_pid=''
(( $(wc -l <"$gpu_csv_partial") >= 3 )) || {
  printf 'error: GPU monitor produced no sample\n' >&2
  exit 1
}
mv -- "$gpu_csv_partial" "$gpu_csv"
