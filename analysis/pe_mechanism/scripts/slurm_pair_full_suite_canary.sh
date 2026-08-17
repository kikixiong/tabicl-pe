#!/usr/bin/env bash
#SBATCH --job-name=pe-pair-canary
#SBATCH --partition=h100
#SBATCH --qos=medium
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00

set -euo pipefail

required_paths=(
  PE_PAIR_ANALYSIS_ROOT PE_PAIR_MODEL_ROOT PE_PAIR_TABARENA_ROOT
  PE_PAIR_CACHE_ROOT PE_PAIR_MANIFEST PE_PAIR_ROSTER PE_PAIR_SHARD_PLAN
  PE_PAIR_RUN_ROOT PE_PAIR_PYTHON
)
for name in "${required_paths[@]}"; do
  value=${!name:-}
  [[ -n "$value" && "$value" == /* ]] || {
    printf 'error: %s must be a non-empty absolute value\n' "$name" >&2
    exit 2
  }
done
for name in PE_PAIR_EXPECTED_ANALYSIS_SHA PE_PAIR_EXPECTED_TABARENA_SHA; do
  value=${!name:-}
  [[ "$value" =~ ^[0-9a-f]{40}$|^[0-9a-f]{64}$ ]] || {
    printf 'error: %s must be a full lowercase Git object ID\n' "$name" >&2
    exit 2
  }
done
for name in \
  PE_PAIR_EXPECTED_MANIFEST_SHA256 PE_PAIR_EXPECTED_ROSTER_SHA256 \
  PE_PAIR_EXPECTED_SHARD_PLAN_SHA256; do
  value=${!name:-}
  [[ "$value" =~ ^[0-9a-f]{64}$ ]] || {
    printf 'error: %s must be a lowercase SHA-256 digest\n' "$name" >&2
    exit 2
  }
done
reject_symlink_components() {
  local path=$1 label=$2 current=/ component
  local -a components=()
  IFS=/ read -r -a components <<< "${path#/}"
  for component in "${components[@]}"; do
    [[ -n "$component" ]] || continue
    current="${current%/}/$component"
    [[ ! -L "$current" ]] || {
      printf 'error: %s traverses a symlink: %s\n' "$label" "$current" >&2
      exit 2
    }
  done
}
ensure_real_directory() {
  local path=$1 label=$2
  reject_symlink_components "$path" "$label"
  mkdir -p -- "$path"
  reject_symlink_components "$path" "$label"
  [[ -d "$path" && ! -L "$path" ]] || {
    printf 'error: %s must be a real directory\n' "$label" >&2
    exit 2
  }
}
verify_clean_detached_checkout() {
  local root=$1 expected=$2 label=$3 head symbolic_status status
  head=$(git -C "$root" rev-parse --verify HEAD) || {
    printf 'error: cannot resolve %s checkout HEAD\n' "$label" >&2
    exit 2
  }
  [[ "$head" == "$expected" ]] || {
    printf 'error: %s checkout HEAD mismatch\n' "$label" >&2
    exit 2
  }
  if git -C "$root" symbolic-ref -q HEAD >/dev/null 2>&1; then
    printf 'error: %s checkout must be detached\n' "$label" >&2
    exit 2
  else
    symbolic_status=$?
    [[ "$symbolic_status" == 1 ]] || {
      printf 'error: cannot verify detached %s checkout\n' "$label" >&2
      exit 2
    }
  fi
  status=$(git -C "$root" status --porcelain=v1 --untracked-files=all) || {
    printf 'error: cannot inspect %s checkout status\n' "$label" >&2
    exit 2
  }
  [[ -z "$status" ]] || {
    printf 'error: %s checkout must be clean\n' "$label" >&2
    exit 2
  }
}
verify_sha256() {
  local path=$1 expected=$2 label=$3 output actual
  output=$(sha256sum -- "$path") || {
    printf 'error: cannot hash %s\n' "$label" >&2
    exit 2
  }
  actual=${output%% *}
  [[ "$actual" =~ ^[0-9a-f]{64}$ && "$actual" == "$expected" ]] || {
    printf 'error: %s SHA-256 mismatch\n' "$label" >&2
    exit 2
  }
}
for name in "${required_paths[@]}"; do
  reject_symlink_components "${!name}" "$name"
done
for file in "$PE_PAIR_MANIFEST" "$PE_PAIR_ROSTER" "$PE_PAIR_SHARD_PLAN"; do
  [[ -f "$file" && ! -L "$file" ]] || {
    printf 'error: input manifest must be a real file: %s\n' "$file" >&2
    exit 2
  }
done
[[ -x "$PE_PAIR_PYTHON" && ! -L "$PE_PAIR_PYTHON" ]] || {
  printf 'error: PE_PAIR_PYTHON must be a real executable\n' >&2
  exit 2
}

analysis_root=$(readlink -f -- "$PE_PAIR_ANALYSIS_ROOT")
run_root=$(readlink -m -- "$PE_PAIR_RUN_ROOT")
case "$run_root/" in
  "$analysis_root/"*)
    printf 'error: generated results must be outside the analysis checkout\n' >&2
    exit 2
    ;;
esac
verify_clean_detached_checkout \
  "$analysis_root" "$PE_PAIR_EXPECTED_ANALYSIS_SHA" analysis
verify_sha256 \
  "$PE_PAIR_MANIFEST" "$PE_PAIR_EXPECTED_MANIFEST_SHA256" pair-manifest
verify_sha256 "$PE_PAIR_ROSTER" "$PE_PAIR_EXPECTED_ROSTER_SHA256" roster
verify_sha256 \
  "$PE_PAIR_SHARD_PLAN" "$PE_PAIR_EXPECTED_SHARD_PLAN_SHA256" shard-plan
gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader | sed -n '1p')
[[ "$gpu_name" == *H100* ]] || {
  printf 'error: BeyondArena canary requires H100, got %s\n' "$gpu_name" >&2
  exit 2
}

runtime_root="$run_root/.slurm-runtime/${SLURM_JOB_ID:-manual}-canary"
ensure_real_directory "$runtime_root/home" runtime-home
ensure_real_directory "$runtime_root/cache/huggingface" runtime-huggingface
ensure_real_directory "$runtime_root/cache/xdg" runtime-xdg
ensure_real_directory "$runtime_root/tmp" runtime-tmp
export HOME="$runtime_root/home"
export HF_HOME="$runtime_root/cache/huggingface"
export XDG_CACHE_HOME="$runtime_root/cache/xdg"
export TMPDIR="$runtime_root/tmp"
export WANDB_MODE=disabled
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONHASHSEED=0
export PYTHONPATH="$analysis_root/analysis/pe_mechanism/src"

monitor_root="$run_root/gpu-monitor"
ensure_real_directory "$monitor_root" gpu-monitor
monitor_csv="$monitor_root/canary-${SLURM_JOB_ID:-manual}.csv"
monitor_helper="$analysis_root/analysis/pe_mechanism/scripts/safe_gpu_monitor.sh"
reject_symlink_components "$monitor_helper" monitor-helper
[[ -x "$monitor_helper" && -f "$monitor_helper" && ! -L "$monitor_helper" ]] || {
  printf 'error: GPU monitor helper must be a real executable\n' >&2
  exit 2
}
# safe_gpu_monitor.sh owns one no-follow descriptor and samples with "sleep 30".
coproc PAIR_GPU_MONITOR {
  "$monitor_helper" "$PE_PAIR_PYTHON" "$monitor_csv"
}
monitor_pid=$PAIR_GPU_MONITOR_PID
monitor_ready_fd=${PAIR_GPU_MONITOR[0]}
monitor_input_fd=${PAIR_GPU_MONITOR[1]}
exec {monitor_input_fd}>&-
monitor_ready=''
if ! IFS= read -r -t 30 monitor_ready <&"$monitor_ready_fd" \
  || [[ "$monitor_ready" != READY ]]; then
  wait "$monitor_pid" 2>/dev/null || true
  printf 'error: GPU monitor failed secure initialization\n' >&2
  exit 2
fi
exec {monitor_ready_fd}<&-
cleanup_monitor() {
  kill "$monitor_pid" 2>/dev/null || true
  wait "$monitor_pid" 2>/dev/null || true
}
trap cleanup_monitor EXIT INT TERM

"$PE_PAIR_PYTHON" -B \
  "$analysis_root/analysis/pe_mechanism/scripts/run_pair_full_suite_shard.py" \
  --analysis-root "$analysis_root" \
  --model-root "$PE_PAIR_MODEL_ROOT" \
  --tabarena-root "$PE_PAIR_TABARENA_ROOT" \
  --cache-root "$PE_PAIR_CACHE_ROOT" \
  --pair-manifest "$PE_PAIR_MANIFEST" \
  --roster "$PE_PAIR_ROSTER" \
  --shard-plan "$PE_PAIR_SHARD_PLAN" \
  --run-root "$run_root" \
  --expected-analysis-sha "$PE_PAIR_EXPECTED_ANALYSIS_SHA" \
  --expected-tabarena-sha "$PE_PAIR_EXPECTED_TABARENA_SHA" \
  --expected-suite beyondarena \
  --phase canary

kill -0 "$monitor_pid" 2>/dev/null || {
  printf 'error: GPU monitor exited before evaluator completion\n' >&2
  exit 1
}
