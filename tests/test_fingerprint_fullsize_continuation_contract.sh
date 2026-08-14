#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
RUNNER="$ROOT/scripts/run_fingerprint_fullsize_continuation.py"
SLURM="$ROOT/scripts/slurm_fingerprint_fullsize_continuation_h100.sh"
SUBMIT="$ROOT/scripts/submit_fingerprint_fullsize_continuation_h100.sh"

for script in "$SLURM" "$SUBMIT"; do
  bash -n "$script"
done

contains() {
  local path="$1"
  local expected="$2"
  grep -F -- "$expected" "$path" >/dev/null || {
    echo "missing contract text in $path: $expected" >&2
    exit 1
  }
}

contains "$RUNNER" 'SCHEDULER_HORIZON_STEPS = 500_000'
contains "$RUNNER" 'trainer.config.max_steps = args.stop_after_step'
contains "$RUNNER" 'if not getattr(trainer, "_loaded_full_resume", False)'
contains "$RUNNER" '"--only_load_model",'
contains "$RUNNER" '"False",'
contains "$RUNNER" 'CHECKPOINT_CEILING_BYTES = 300_000_000'
contains "$SLURM" '#SBATCH --partition=h100'
contains "$SLURM" '#SBATCH --cpus-per-task=64'
contains "$SLURM" '#SBATCH --mem=128G'
contains "$SLURM" '#SBATCH --gres=gpu:1'
contains "$SLURM" '[[ "${GPU_NAMES[0]}" == *"NVIDIA H100"* ]]'
contains "$SLURM" 'MONITOR_INTERVAL=1'
if grep -F 'ls-remote' "$SLURM" >/dev/null; then
  echo "compute job must not depend on live GitHub availability" >&2
  exit 1
fi
contains "$SUBMIT" 'TARGETS=(20000 35000 50000)'
contains "$SUBMIT" 'QOS=medium'
contains "$SUBMIT" 'TIME_LIMIT=1-00:00:00'
contains "$SUBMIT" 'sbatch --parsable --hold'
contains "$SUBMIT" 'trap rollback EXIT'
contains "$SUBMIT" '--dependency="afterok:${PREVIOUS_PAIR[0]}:${PREVIOUS_PAIR[1]}"'
contains "$SUBMIT" '--kill-on-invalid-dep=yes'
contains "$SUBMIT" 'scontrol release "$JOB_ID"'
contains "$SUBMIT" 'scancel "${JOB_IDS[$index]}"'
contains "$SUBMIT" 'public source ref changed while jobs were held'
contains "$SUBMIT" 'publish_transaction_record held'
contains "$SUBMIT" 'held-plan.json'
contains "$SUBMIT" 'os.fsync(directory_fd)'

HELD_LINE="$(grep -nF 'publish_transaction_record held' "$SUBMIT" | cut -d: -f1)"
RELEASE_LINE="$(grep -nF 'scontrol release "$JOB_ID"' "$SUBMIT" | cut -d: -f1)"
(( HELD_LINE < RELEASE_LINE )) || {
  echo "durable held plan must be published before the first release" >&2
  exit 1
}

echo "full-size continuation launcher contract: PASS"
