#!/usr/bin/env bash
#SBATCH --job-name=pe-talent-aggregate
#SBATCH --partition=normal
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00

set -euo pipefail

(( $# == 0 )) || { printf 'error: positional arguments are not supported\n' >&2; exit 2; }
required=(
  PE_TALENT_ANALYSIS_ROOT PE_TALENT_RUN_CONFIG PE_TALENT_SHARD_PLAN
  PE_TALENT_OUTPUT_ROOT PE_TALENT_PYTHON PE_TALENT_EXPECTED_ANALYSIS_SHA
  PE_TALENT_EXPECTED_RUN_CONFIG_SHA256 PE_TALENT_EXPECTED_SHARD_PLAN_SHA256
  PE_TALENT_PYTHON_CONTRACT
  PE_TALENT_EXPECTED_PYTHON_CONTRACT_DOCUMENT_SHA256
  PE_TALENT_EXPECTED_PYTHON_CONTRACT_FILE_SHA256
)
for name in "${required[@]}"; do
  [[ -n "${!name:-}" ]] || { printf 'error: missing %s\n' "$name" >&2; exit 2; }
done
for name in \
  PE_TALENT_ANALYSIS_ROOT PE_TALENT_RUN_CONFIG PE_TALENT_SHARD_PLAN \
  PE_TALENT_OUTPUT_ROOT PE_TALENT_PYTHON PE_TALENT_PYTHON_CONTRACT; do
  [[ "${!name}" == /* ]] || { printf 'error: %s must be absolute\n' "$name" >&2; exit 2; }
done
[[ "$PE_TALENT_EXPECTED_ANALYSIS_SHA" =~ ^[0-9a-f]{40}$ ]] || exit 2
[[ "$PE_TALENT_EXPECTED_RUN_CONFIG_SHA256" =~ ^[0-9a-f]{64}$ ]] || exit 2
[[ "$PE_TALENT_EXPECTED_SHARD_PLAN_SHA256" =~ ^[0-9a-f]{64}$ ]] || exit 2
[[ "$PE_TALENT_EXPECTED_PYTHON_CONTRACT_DOCUMENT_SHA256" =~ ^[0-9a-f]{64}$ ]] || exit 2
[[ -d "$PE_TALENT_ANALYSIS_ROOT" && ! -L "$PE_TALENT_ANALYSIS_ROOT" ]] || exit 2
[[ -d "$PE_TALENT_OUTPUT_ROOT" && ! -L "$PE_TALENT_OUTPUT_ROOT" ]] || exit 2
[[ -f "$PE_TALENT_RUN_CONFIG" && ! -L "$PE_TALENT_RUN_CONFIG" ]] || exit 2
[[ -f "$PE_TALENT_SHARD_PLAN" && ! -L "$PE_TALENT_SHARD_PLAN" ]] || exit 2
[[ "$PE_TALENT_EXPECTED_PYTHON_CONTRACT_FILE_SHA256" =~ ^[0-9a-f]{64}$ ]] || exit 2
[[ -f "$PE_TALENT_PYTHON_CONTRACT" && ! -L "$PE_TALENT_PYTHON_CONTRACT" ]] || exit 2
current=$(dirname -- "$PE_TALENT_PYTHON")
while true; do
  [[ -d "$current" && ! -L "$current" ]] || exit 2
  [[ "$current" == / ]] && break
  current=$(dirname -- "$current")
done
[[ -x "$PE_TALENT_PYTHON" && -L "$PE_TALENT_PYTHON" ]] || exit 2
python_verifier="$PE_TALENT_ANALYSIS_ROOT/analysis/pe_mechanism/scripts/verify_python_environment.py"
[[ -f "$python_verifier" && ! -L "$python_verifier" ]] || exit 2
run_bound_python() {
  "$PE_TALENT_PYTHON" -I -B "$python_verifier" \
    --entry "$PE_TALENT_PYTHON" \
    --contract "$PE_TALENT_PYTHON_CONTRACT" \
    --expected-document-sha256 "$PE_TALENT_EXPECTED_PYTHON_CONTRACT_DOCUMENT_SHA256" \
    --expected-file-sha256 "$PE_TALENT_EXPECTED_PYTHON_CONTRACT_FILE_SHA256" \
    -- "$@"
}

export PYTHONHASHSEED=0
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PE_TALENT_ANALYSIS_ROOT/analysis/pe_mechanism/src"

shard_args=()
monitor_args=(--gpu-csv "$PE_TALENT_OUTPUT_ROOT/monitor/shard-canary.csv")
for index in $(seq 0 7); do
  printf -v shard_id '%03d' "$index"
  shard_args+=(--shard-output "$PE_TALENT_OUTPUT_ROOT/shards/shard-$shard_id")
  monitor_args+=(--gpu-csv "$PE_TALENT_OUTPUT_ROOT/monitor/shard-$shard_id.csv")
done

run_bound_python -B \
  "$PE_TALENT_ANALYSIS_ROOT/analysis/pe_mechanism/scripts/aggregate_talent_paired_full_suite.py" \
  --analysis-root "$PE_TALENT_ANALYSIS_ROOT" \
  --expected-analysis-sha "$PE_TALENT_EXPECTED_ANALYSIS_SHA" \
  --expected-run-config-sha256 "$PE_TALENT_EXPECTED_RUN_CONFIG_SHA256" \
  --expected-shard-plan-sha256 "$PE_TALENT_EXPECTED_SHARD_PLAN_SHA256" \
  --expected-python-contract-document-sha256 "$PE_TALENT_EXPECTED_PYTHON_CONTRACT_DOCUMENT_SHA256" \
  --expected-python-contract-file-sha256 "$PE_TALENT_EXPECTED_PYTHON_CONTRACT_FILE_SHA256" \
  --shard-plan "$PE_TALENT_SHARD_PLAN" \
  --run-config "$PE_TALENT_RUN_CONFIG" \
  --canary-output "$PE_TALENT_OUTPUT_ROOT/canary/shard-canary" \
  --submission-receipt "$PE_TALENT_OUTPUT_ROOT/submission-receipt.json" \
  --release-receipt "$PE_TALENT_OUTPUT_ROOT/release-receipt.json" \
  "${shard_args[@]}" \
  "${monitor_args[@]}" \
  --output-dir "$PE_TALENT_OUTPUT_ROOT/aggregate"
