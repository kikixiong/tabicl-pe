#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
fail() { echo "maxseq smoke contract failed: $*" >&2; exit 1; }
contains() { grep -F -- "$2" "$1" >/dev/null || fail "$(basename "$1") lacks $2"; }

for SCRIPT in \
  "$ROOT/scripts/run_h100_identity_maxseq_smoke.sh" \
  "$ROOT/scripts/run_slurm_h100_identity_case.sh" \
  "$ROOT/scripts/slurm_h100_identity_maxseq_smoke.sh" \
  "$ROOT/scripts/slurm_h100_identity_nccl_smoke.sh" \
  "$ROOT/scripts/submit_h100_identity_validation.sh" \
  "$ROOT/scripts/formal_run_with_gpu_monitor.sh"; do
  [[ -x "$SCRIPT" ]] || fail "$SCRIPT is not executable"
  bash -n "$SCRIPT"
done

contains "$ROOT/scripts/slurm_h100_identity_maxseq_smoke.sh" '#SBATCH --qos=short'
contains "$ROOT/scripts/slurm_h100_identity_maxseq_smoke.sh" '#SBATCH --gres=gpu:1'
contains "$ROOT/scripts/slurm_h100_identity_nccl_smoke.sh" '#SBATCH --gres=gpu:2'
contains "$ROOT/scripts/slurm_h100_identity_maxseq_smoke.sh" 'nccl_2gpu must use the dedicated two-GPU wrapper'
contains "$ROOT/scripts/slurm_h100_identity_nccl_smoke.sh" 'reserved for nccl_2gpu'
contains "$ROOT/scripts/run_slurm_h100_identity_case.sh" 'checkout --quiet --detach'
contains "$ROOT/scripts/run_slurm_h100_identity_case.sh" 'hermetic_git() {'
contains "$ROOT/scripts/run_slurm_h100_identity_case.sh" 'GIT_ALLOW_PROTOCOL=https'
contains "$ROOT/scripts/run_slurm_h100_identity_case.sh" 'hermetic_git -C "$CHECKOUT_PARENT" clone --quiet --no-hardlinks --no-checkout --'
contains "$ROOT/scripts/run_slurm_h100_identity_case.sh" 'remote get-url origin'
contains "$ROOT/scripts/run_slurm_h100_identity_case.sh" '--query-token "$GPU_TOKEN"'
contains "$ROOT/scripts/run_slurm_h100_identity_case.sh" '--query-fields uuid,name,driver_version'
contains "$ROOT/scripts/exec_digest_bound_nvidia_smi.py" 'QUERY_TIMEOUT_SECONDS = 15'
contains "$ROOT/scripts/run_slurm_h100_identity_case.sh" 'trap '\''status=$?; trap - EXIT HUP INT TERM; cleanup; exit "$status"'\'' EXIT'
contains "$ROOT/scripts/run_slurm_h100_identity_case.sh" "trap 'exit 129' HUP"
contains "$ROOT/scripts/run_slurm_h100_identity_case.sh" 'FORMAL_H100_VERIFY_FILESYSTEM_FD'
contains "$ROOT/scripts/run_slurm_h100_identity_case.sh" 'FORMAL_H100_VERIFY_ENVIRONMENT_FD'
contains "$ROOT/scripts/run_slurm_h100_identity_case.sh" '--expected-sha256 "$FORMAL_EXPECTED_ENVIRONMENT_SHA256"'
contains "$ROOT/scripts/slurm_h100_identity_maxseq_smoke.sh" 'exec "$PYTHON" -I -B - 1'
contains "$ROOT/scripts/slurm_h100_identity_nccl_smoke.sh" 'exec "$PYTHON" -I -B - 2'
for SCRIPT in "$ROOT/scripts/slurm_h100_identity_maxseq_smoke.sh" "$ROOT/scripts/slurm_h100_identity_nccl_smoke.sh"; do
  contains "$SCRIPT" '"FORMAL_TRUSTED_RUNNER_FD": ("run_slurm_h100_identity_case.sh", True)'
  contains "$SCRIPT" 'runner_fd = trusted_fds["FORMAL_TRUSTED_RUNNER_FD"]'
  contains "$SCRIPT" '"FORMAL_H100_NVIDIA_LAUNCHER_FD": ("exec_digest_bound_nvidia_smi.py", False)'
  contains "$SCRIPT" '"FORMAL_H100_VERIFY_GIT_FD": ("verify_git_repository.py", False)'
  contains "$SCRIPT" '"FORMAL_H100_VERIFY_FILESYSTEM_FD": ("verify_filesystem_isolation.py", False)'
  contains "$SCRIPT" '"FORMAL_H100_VERIFY_ENVIRONMENT_FD": ("verify_formal_environment.py", False)'
  contains "$SCRIPT" 'trusted H100 helper differs from committed source'
done
contains "$ROOT/scripts/run_slurm_h100_identity_case.sh" '"/proc/self/fd/$FORMAL_H100_NVIDIA_LAUNCHER_FD"'
contains "$ROOT/scripts/run_slurm_h100_identity_case.sh" '"/proc/self/fd/$FORMAL_H100_VERIFY_GIT_FD"'
contains "$ROOT/scripts/run_slurm_h100_identity_case.sh" '"/proc/self/fd/$FORMAL_H100_VERIFY_FILESYSTEM_FD"'
contains "$ROOT/scripts/run_slurm_h100_identity_case.sh" '"/proc/self/fd/$FORMAL_H100_VERIFY_ENVIRONMENT_FD"'
if grep -F '"$GIT" -C' "$ROOT/scripts/slurm_h100_identity_maxseq_smoke.sh" "$ROOT/scripts/slurm_h100_identity_nccl_smoke.sh" >/dev/null; then
  fail "static Slurm wrapper executes mutable GIT before shared digest binding"
fi
if grep -F 'dirname "${BASH_SOURCE[0]}"' "$ROOT/scripts/slurm_h100_identity_maxseq_smoke.sh" >/dev/null; then
  fail "one-GPU spool wrapper resolves an untrusted sibling directory"
fi
if grep -F 'dirname "${BASH_SOURCE[0]}"' "$ROOT/scripts/slurm_h100_identity_nccl_smoke.sh" >/dev/null; then
  fail "two-GPU spool wrapper resolves an untrusted sibling directory"
fi
contains "$ROOT/scripts/run_h100_identity_maxseq_smoke.sh" 'FORMAL_VISIBLE_GPU_UUIDS'
contains "$ROOT/scripts/run_h100_identity_maxseq_smoke.sh" 'FORMAL_EXPECTED_ENVIRONMENT_SHA256'
contains "$ROOT/scripts/run_h100_identity_maxseq_smoke.sh" '--nproc_per_node=2'
contains "$ROOT/scripts/run_h100_identity_maxseq_smoke.sh" 'main NVIDIA-SMI is not its reacquired verified descriptor'
contains "$ROOT/scripts/run_h100_identity_maxseq_smoke.sh" 'DIGEST_BOUND_COMPUTE=('
contains "$ROOT/scripts/run_h100_identity_maxseq_smoke.sh" '"${DIGEST_BOUND_COMPUTE[@]}"'
if grep -F 'MAXSEQ_REPEAT_STEPS' "$ROOT/scripts/run_h100_identity_maxseq_smoke.sh" >/dev/null; then
  fail "hostile repeat-step override is exposed"
fi
contains "$ROOT/scripts/run_h100_identity_maxseq_smoke.sh" '--utilization-required'
contains "$ROOT/scripts/run_h100_identity_maxseq_smoke.sh" 'functional case created unbound GPU monitor artifacts'
contains "$ROOT/scripts/run_h100_identity_maxseq_smoke.sh" '--checkpoint-ceiling-bytes'
contains "$ROOT/scripts/run_h100_identity_maxseq_smoke.sh" '--runtime-evidence-output'
contains "$ROOT/scripts/run_h100_identity_maxseq_smoke.sh" 'os.mkdir(case_id, mode=0o700, dir_fd=directory_fd)'
contains "$ROOT/scripts/run_h100_identity_maxseq_smoke.sh" 'os.O_NOFOLLOW'
contains "$ROOT/scripts/run_h100_identity_maxseq_smoke.sh" 'os.path.normpath(root) != root'
if grep -F 'mkdir -- "$CASE_ROOT"' "$ROOT/scripts/run_h100_identity_maxseq_smoke.sh" >/dev/null; then
  fail "case root creation follows a path instead of a validated parent dirfd"
fi
if grep -F 'mkdir -p "$CASE_ROOT"' "$ROOT/scripts/run_h100_identity_maxseq_smoke.sh" >/dev/null; then
  fail "case root creation is not atomic fresh-only mkdir"
fi
if grep -F '[[ ! -e "$CASE_ROOT"' "$ROOT/scripts/run_h100_identity_maxseq_smoke.sh" >/dev/null; then
  fail "case root uses a check-then-create race"
fi
contains "$ROOT/scripts/formal_run_with_gpu_monitor.sh" 'summarize_formal_gpu_usage.py'
contains "$ROOT/scripts/formal_run_with_gpu_monitor.sh" 'FORMAL_GPU_ACTIVE_START_SIGNAL='
contains "$ROOT/scripts/formal_run_with_gpu_monitor.sh" 'FORMAL_GPU_ACTIVE_END_SIGNAL='
contains "$ROOT/scripts/summarize_formal_gpu_usage.py" 'interval_seconds != 1.0'

MATRIX="$($PYTHON -I -B "$ROOT/scripts/run_h100_identity_validation.py" --print-matrix)"
COUNT="$($PYTHON -I -B -c 'import json,sys; print(len(json.load(sys.stdin)))' <<<"$MATRIX")"
[[ "$COUNT" -eq 12 ]] || fail "matrix count is $COUNT, expected 12"

for CASE_ID in \
  stage1_rope_one_step stage1_temporary_one_step stage1_none_one_step \
  stage2_rope_maxseq10240 stage2_temporary_maxseq10240 stage2_none_maxseq10240 \
  stage3_rope_maxseq60000_recompute stage3_temporary_maxseq60000_recompute \
  stage3_none_maxseq60000_recompute temporary_cuda_rng_resume nccl_2gpu \
  prior_dataloader_resume; do
  "$PYTHON" -I -B "$ROOT/scripts/run_h100_identity_validation.py" \
    --dry-run-case "$CASE_ID" --root "$ROOT" >/dev/null
done

if "$ROOT/scripts/run_h100_identity_maxseq_smoke.sh" invalid-case >/dev/null 2>&1; then
  fail "invalid validation case was accepted"
fi

echo "maxseq smoke contracts passed"
