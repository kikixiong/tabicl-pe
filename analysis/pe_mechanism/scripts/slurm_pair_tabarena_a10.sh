#!/usr/bin/env bash
#SBATCH --job-name=pe-pair-tabarena
#SBATCH --partition=normal
#SBATCH --qos=medium
#SBATCH --array=0-7%2
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00

set -euo pipefail

required=(
  PE_PAIR_ANALYSIS_ROOT PE_PAIR_MODEL_ROOT PE_PAIR_TABARENA_ROOT
  PE_PAIR_CACHE_ROOT PE_PAIR_MANIFEST PE_PAIR_ROSTER PE_PAIR_SHARD_PLAN
  PE_PAIR_RUN_ROOT PE_PAIR_PYTHON PE_PAIR_PYTHON_CONTRACT PE_PAIR_EXPECTED_ANALYSIS_SHA
  PE_PAIR_EXPECTED_TABARENA_SHA PE_PAIR_EXPECTED_MANIFEST_SHA256
  PE_PAIR_EXPECTED_ROSTER_SHA256 PE_PAIR_EXPECTED_SHARD_PLAN_SHA256
  PE_PAIR_EXPECTED_PYTHON_CONTRACT_DOCUMENT_SHA256
  PE_PAIR_EXPECTED_PYTHON_CONTRACT_FILE_SHA256
  PE_PAIR_PHASE
)
for name in "${required[@]}"; do
  value=${!name:-}
  [[ -n "$value" ]] || { printf 'error: missing %s\n' "$name" >&2; exit 2; }
done
for name in PE_PAIR_EXPECTED_ANALYSIS_SHA PE_PAIR_EXPECTED_TABARENA_SHA; do
  value=${!name}
  [[ "$value" =~ ^[0-9a-f]{40}$|^[0-9a-f]{64}$ ]] || {
    printf 'error: %s must be a full lowercase Git object ID\n' "$name" >&2
    exit 2
  }
done
for name in \
  PE_PAIR_EXPECTED_MANIFEST_SHA256 PE_PAIR_EXPECTED_ROSTER_SHA256 \
  PE_PAIR_EXPECTED_SHARD_PLAN_SHA256 \
  PE_PAIR_EXPECTED_PYTHON_CONTRACT_DOCUMENT_SHA256 \
  PE_PAIR_EXPECTED_PYTHON_CONTRACT_FILE_SHA256; do
  value=${!name}
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
reject_parent_symlink_components() {
  local path=$1 label=$2
  reject_symlink_components "$(dirname -- "$path")" "$label parent"
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
for name in \
  PE_PAIR_ANALYSIS_ROOT PE_PAIR_MODEL_ROOT PE_PAIR_TABARENA_ROOT \
  PE_PAIR_CACHE_ROOT PE_PAIR_MANIFEST PE_PAIR_ROSTER PE_PAIR_SHARD_PLAN \
  PE_PAIR_RUN_ROOT PE_PAIR_PYTHON_CONTRACT; do
  reject_symlink_components "${!name}" "$name"
done
reject_parent_symlink_components "$PE_PAIR_PYTHON" PE_PAIR_PYTHON
for name in \
  PE_PAIR_ANALYSIS_ROOT PE_PAIR_MODEL_ROOT PE_PAIR_TABARENA_ROOT \
  PE_PAIR_CACHE_ROOT PE_PAIR_MANIFEST PE_PAIR_ROSTER PE_PAIR_SHARD_PLAN \
  PE_PAIR_RUN_ROOT PE_PAIR_PYTHON PE_PAIR_PYTHON_CONTRACT; do
  value=${!name}
  [[ "$value" == /* ]] || {
    printf 'error: %s must be absolute\n' "$name" >&2
    exit 2
  }
done
[[ "${SLURM_ARRAY_TASK_ID:-}" =~ ^[0-9]+$ ]] || {
  printf 'error: A10 wrapper must run as a Slurm array task\n' >&2
  exit 2
}
case "$PE_PAIR_PHASE" in
  canary)
    [[ "${SLURM_ARRAY_TASK_COUNT:-}" == 1 && "$SLURM_ARRAY_TASK_ID" == 0 ]] || {
      printf 'error: TabArena canary requires sbatch --array=0-0\n' >&2
      exit 2
    }
    phase_args=(--phase canary)
    ;;
  full)
    [[ "${SLURM_ARRAY_TASK_COUNT:-}" == 8 ]] || {
      printf 'error: TabArena full run requires exactly eight array tasks\n' >&2
      exit 2
    }
    phase_args=(--phase full --shard-index "$SLURM_ARRAY_TASK_ID")
    ;;
  *)
    printf 'error: PE_PAIR_PHASE must be canary or full\n' >&2
    exit 2
    ;;
esac
for file in \
  "$PE_PAIR_MANIFEST" "$PE_PAIR_ROSTER" "$PE_PAIR_SHARD_PLAN" \
  "$PE_PAIR_PYTHON_CONTRACT"; do
  [[ -f "$file" && ! -L "$file" ]] || {
    printf 'error: input manifest must be a real file: %s\n' "$file" >&2
    exit 2
  }
done
[[ -x "$PE_PAIR_PYTHON" && -L "$PE_PAIR_PYTHON" ]] || {
  printf 'error: PE_PAIR_PYTHON must be a venv symlink entry\n' >&2
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
python_verifier="$analysis_root/analysis/pe_mechanism/scripts/verify_python_environment.py"
reject_symlink_components "$python_verifier" python-environment-verifier
[[ -f "$python_verifier" && ! -L "$python_verifier" ]] || {
  printf 'error: Python environment verifier must be a real file\n' >&2
  exit 2
}
run_bound_python() {
  "$PE_PAIR_PYTHON" -I -B "$python_verifier" \
    --entry "$PE_PAIR_PYTHON" \
    --contract "$PE_PAIR_PYTHON_CONTRACT" \
    --expected-document-sha256 "$PE_PAIR_EXPECTED_PYTHON_CONTRACT_DOCUMENT_SHA256" \
    --expected-file-sha256 "$PE_PAIR_EXPECTED_PYTHON_CONTRACT_FILE_SHA256" \
    -- "$@"
}
gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader | sed -n '1p')
[[ "$gpu_name" == "NVIDIA A10" ]] || {
  printf 'error: TabArena wrapper requires NVIDIA A10, got %s\n' "$gpu_name" >&2
  exit 2
}

runtime_root="$run_root/.slurm-runtime/${SLURM_ARRAY_JOB_ID:-manual}-${PE_PAIR_PHASE}-${SLURM_ARRAY_TASK_ID}"
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
monitor_csv="$monitor_root/${PE_PAIR_PHASE}-${SLURM_ARRAY_JOB_ID:-manual}-${SLURM_ARRAY_TASK_ID}.csv"
# The coprocess shell is replaced by the already verified Python monitor process.
coproc PAIR_GPU_MONITOR {
  exec "$PE_PAIR_PYTHON" -I -B "$python_verifier" \
    --entry "$PE_PAIR_PYTHON" \
    --contract "$PE_PAIR_PYTHON_CONTRACT" \
    --expected-document-sha256 "$PE_PAIR_EXPECTED_PYTHON_CONTRACT_DOCUMENT_SHA256" \
    --expected-file-sha256 "$PE_PAIR_EXPECTED_PYTHON_CONTRACT_FILE_SHA256" \
    --gpu-monitor "$monitor_csv"
}
monitor_pid=$PAIR_GPU_MONITOR_PID
monitor_ready_fd=${PAIR_GPU_MONITOR[0]}
monitor_input_fd=${PAIR_GPU_MONITOR[1]}
monitor_finalized=0

close_monitor_input_fd() {
  if [[ -n "${monitor_input_fd:-}" ]]; then
    exec {monitor_input_fd}>&- || true
    monitor_input_fd=''
  fi
}
close_monitor_ready_fd() {
  if [[ -n "${monitor_ready_fd:-}" ]]; then
    exec {monitor_ready_fd}<&- || true
    monitor_ready_fd=''
  fi
}
cleanup_monitor_resources() {
  close_monitor_input_fd
  if (( ${monitor_finalized:-0} == 0 )) && [[ -n "${monitor_pid:-}" ]]; then
    wait "$monitor_pid" 2>/dev/null || true
    monitor_pid=''
  fi
  close_monitor_ready_fd
}
cleanup_monitor() {
  local original_status=$?
  trap - EXIT INT TERM
  cleanup_monitor_resources
  exit "$original_status"
}
cleanup_monitor_signal() {
  local signal_number=$1
  trap - EXIT INT TERM
  cleanup_monitor_resources
  exit "$((128 + signal_number))"
}
finalize_monitor() {
  local monitor_complete='' monitor_wait_status=0
  printf 'STOP\n' >&"$monitor_input_fd" || return 1
  exec {monitor_input_fd}>&-
  monitor_input_fd=''
  IFS= read -r -t 45 monitor_complete <&"$monitor_ready_fd" || return 1
  [[ "$monitor_complete" == COMPLETE ]] || return 1
  exec {monitor_ready_fd}<&-
  monitor_ready_fd=''
  if wait "$monitor_pid"; then
    monitor_wait_status=0
  else
    monitor_wait_status=$?
  fi
  monitor_pid=''
  (( monitor_wait_status == 0 )) || return 1
  monitor_finalized=1
}
trap cleanup_monitor EXIT
trap 'cleanup_monitor_signal 2' INT
trap 'cleanup_monitor_signal 15' TERM

monitor_ready=''
if ! IFS= read -r -t 30 monitor_ready <&"$monitor_ready_fd" \
  || [[ "$monitor_ready" != READY ]]; then
  printf 'error: GPU monitor failed secure initialization\n' >&2
  exit 2
fi

run_bound_python -B \
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
  --expected-python-contract-document-sha256 "$PE_PAIR_EXPECTED_PYTHON_CONTRACT_DOCUMENT_SHA256" \
  --expected-python-contract-file-sha256 "$PE_PAIR_EXPECTED_PYTHON_CONTRACT_FILE_SHA256" \
  --expected-suite tabarena-v0.1 \
  "${phase_args[@]}"

if ! finalize_monitor; then
  printf 'error: GPU monitor did not finalize successfully\n' >&2
  exit 1
fi
trap - EXIT INT TERM
