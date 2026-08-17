#!/usr/bin/env bash
#SBATCH --job-name=pe-pair-aggregate
#SBATCH --partition=normal
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00

set -euo pipefail

required=(
  PE_PAIR_ANALYSIS_ROOT PE_PAIR_MANIFEST PE_PAIR_ROSTER PE_PAIR_SHARD_PLAN
  PE_PAIR_RUN_ROOT PE_PAIR_PYTHON PE_PAIR_PYTHON_CONTRACT
)
for name in "${required[@]}"; do
  value=${!name:-}
  [[ -n "$value" && "$value" == /* ]] || {
    printf 'error: %s must be a non-empty absolute value\n' "$name" >&2
    exit 2
  }
done
[[ "${PE_PAIR_EXPECTED_ANALYSIS_SHA:-}" =~ ^[0-9a-f]{40}$|^[0-9a-f]{64}$ ]] || {
  printf 'error: PE_PAIR_EXPECTED_ANALYSIS_SHA must be a full lowercase Git object ID\n' >&2
  exit 2
}
for name in \
  PE_PAIR_EXPECTED_MANIFEST_SHA256 PE_PAIR_EXPECTED_ROSTER_SHA256 \
  PE_PAIR_EXPECTED_SHARD_PLAN_SHA256 \
  PE_PAIR_EXPECTED_PYTHON_CONTRACT_DOCUMENT_SHA256 \
  PE_PAIR_EXPECTED_PYTHON_CONTRACT_FILE_SHA256; do
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
reject_parent_symlink_components() {
  local path=$1 label=$2
  reject_symlink_components "$(dirname -- "$path")" "$label parent"
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
for name in "${required[@]}"; do
  [[ "$name" == PE_PAIR_PYTHON ]] && continue
  reject_symlink_components "${!name}" "$name"
done
reject_parent_symlink_components "$PE_PAIR_PYTHON" PE_PAIR_PYTHON
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
run_root=$(readlink -f -- "$PE_PAIR_RUN_ROOT")
case "$run_root/" in
  "$analysis_root/"*)
    printf 'error: aggregate input must be outside the analysis checkout\n' >&2
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
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONHASHSEED=0
export PYTHONPATH="$analysis_root/analysis/pe_mechanism/src"

run_bound_python -B \
  "$analysis_root/analysis/pe_mechanism/scripts/aggregate_pair_full_suite.py" \
  --pair-manifest "$PE_PAIR_MANIFEST" \
  --roster "$PE_PAIR_ROSTER" \
  --shard-plan "$PE_PAIR_SHARD_PLAN" \
  --run-root "$run_root" \
  --expected-python-contract-document-sha256 "$PE_PAIR_EXPECTED_PYTHON_CONTRACT_DOCUMENT_SHA256" \
  --expected-python-contract-file-sha256 "$PE_PAIR_EXPECTED_PYTHON_CONTRACT_FILE_SHA256"
