#!/usr/bin/env bash
set -euo pipefail

EXPECTED_GPUS="${1:-}"
case "$EXPECTED_GPUS" in 1|2) ;; *) echo "expected GPU count must be 1 or 2" >&2; exit 2 ;; esac
[[ "${PATH:-}" == "/usr/bin:/bin" ]] || {
  echo "validation batch PATH must be exactly /usr/bin:/bin" >&2
  exit 2
}
: "${FORMAL_EXPECTED_GPUS:?}"
[[ "$FORMAL_EXPECTED_GPUS" == "$EXPECTED_GPUS" ]] || {
  echo "wrapper GPU count differs from the submitted resource contract" >&2
  exit 2
}
[[ "${FORMAL_REQUESTED_GPUS:-}" == "$EXPECTED_GPUS" ]] || {
  echo "requested GPU resource differs from wrapper" >&2; exit 2;
}
: "${VALIDATION_CASE_ID:?}"
case "$VALIDATION_CASE_ID" in
  nccl_2gpu) CASE_GPUS=2 ;;
  stage1_rope_one_step|stage1_temporary_one_step|stage1_none_one_step|\
  stage2_rope_maxseq10240|stage2_temporary_maxseq10240|stage2_none_maxseq10240|\
  stage3_rope_maxseq60000_recompute|stage3_temporary_maxseq60000_recompute|stage3_none_maxseq60000_recompute|\
  temporary_cuda_rng_resume|prior_dataloader_resume) CASE_GPUS=1 ;;
  *) echo "unknown validation case" >&2; exit 2 ;;
esac
[[ "$CASE_GPUS" == "$EXPECTED_GPUS" ]] || {
  echo "validation case was submitted through the wrong GPU wrapper" >&2
  exit 2
}

: "${PYTHON:?PYTHON must name the trusted absolute interpreter}"
: "${NVIDIA_SMI:?NVIDIA_SMI must name the trusted absolute executable}"
: "${FORMAL_NVIDIA_SMI_SHA256:?FORMAL_NVIDIA_SMI_SHA256 is required}"
for FD_NAME in FORMAL_H100_NVIDIA_LAUNCHER_FD FORMAL_H100_VERIFY_GIT_FD \
  FORMAL_H100_VERIFY_FILESYSTEM_FD FORMAL_H100_VERIFY_ENVIRONMENT_FD; do
  [[ "${!FD_NAME:-}" =~ ^[0-9]+$ ]] || {
    echo "$FD_NAME lacks its trusted static descriptor binding" >&2
    exit 2
  }
done
if [[ -z "${FORMAL_NVIDIA_SMI_FD:-}" ]]; then
  [[ "$PYTHON" == /* && -x "$PYTHON" && "$NVIDIA_SMI" == /* ]] || {
    echo "invalid digest-bound NVIDIA-SMI bootstrap" >&2; exit 2;
  }
  exec "$PYTHON" -I -B \
    "/proc/self/fd/$FORMAL_H100_NVIDIA_LAUNCHER_FD" \
    --nvidia-smi "$NVIDIA_SMI" \
    --expected-sha256 "$FORMAL_NVIDIA_SMI_SHA256" -- \
    "${BASH_SOURCE[0]}" "$EXPECTED_GPUS"
fi
[[ "$NVIDIA_SMI" == "/proc/self/fd/$FORMAL_NVIDIA_SMI_FD" ]] || {
  echo "NVIDIA_SMI is not the verified open descriptor" >&2; exit 2;
}
[[ "${FORMAL_NVIDIA_SMI_FD_OWNER_PID:-}" == "$BASHPID" ]] || {
  echo "NVIDIA_SMI descriptor owner is not the validation wrapper" >&2; exit 2;
}
: "${GIT:?GIT must name the trusted absolute executable}"
: "${FORMAL_GIT_SHA256:?FORMAL_GIT_SHA256 is required}"
if [[ -z "${FORMAL_GIT_FD:-}" ]]; then
  [[ "$GIT" == /* && -x "$GIT" ]] || {
    echo "invalid digest-bound Git bootstrap" >&2; exit 2;
  }
  exec "$PYTHON" -I -B \
    "$FORMAL_SUBMISSION_EXACT_ROOT/scripts/exec_digest_bound_git.py" \
    --git "$GIT" --expected-sha256 "$FORMAL_GIT_SHA256" -- \
    "${BASH_SOURCE[0]}" "$EXPECTED_GPUS"
fi
[[ "${FORMAL_GIT_COMMAND:-}" == "/proc/self/fd/$FORMAL_GIT_FD" ]] || {
  echo "Git command is not the verified open descriptor" >&2; exit 2;
}
[[ "${FORMAL_GIT_FD_OWNER_PID:-}" == "$BASHPID" ]] || {
  echo "Git descriptor owner is not the validation wrapper" >&2; exit 2;
}

for NAME in CANDIDATE_REPOSITORY CANDIDATE_REPOSITORY_REF \
  FORMAL_SOURCE_COMMIT_SHA FORMAL_SOURCE_TREE_SHA \
  FORMAL_SOURCE_MANIFEST FORMAL_SOURCE_SHA256 FORMAL_EXPECTED_ENVIRONMENT_SHA256 \
  FORMAL_GIT_SHA256 FORMAL_REPOSITORY_IDENTITY_SHA256 \
  FORMAL_REPOSITORY_QUERY_SHA256 \
  VALIDATION_ARTIFACT_IDENTITY_SHA256 H100_CASE_WORK_ROOT H100_VALIDATION_ROOT \
  CHECKPOINT_CEILING_BYTES FORMAL_RUN_LOG_CEILING_BYTES \
  FORMAL_GPU_MONITOR_CEILING_BYTES FORMAL_ATTESTATION_CEILING_BYTES PYTHON GIT \
  NVIDIA_SMI FORMAL_NVIDIA_SMI_SHA256 FORMAL_NVIDIA_SMI_FD \
  FORMAL_NVIDIA_SMI_FD_OWNER_PID \
  FORMAL_GIT_FD FORMAL_GIT_FD_OWNER_PID FORMAL_GIT_COMMAND \
  RUN_POLICY FORMAL_SUBMISSION_EXACT_ROOT SLURM_JOB_ID \
  CUDA_VISIBLE_DEVICES FORMAL_REQUESTED_RESOURCE_SHA256 \
  FORMAL_REQUESTED_PARTITION FORMAL_REQUESTED_QOS FORMAL_REQUESTED_TIME_LIMIT \
  FORMAL_REQUESTED_NODES FORMAL_REQUESTED_GPUS FORMAL_REQUESTED_CPUS \
  FORMAL_REQUESTED_MEMORY_MB SLURM_JOB_PARTITION SLURM_CPUS_PER_TASK \
  SLURM_MEM_PER_NODE SLURM_CLUSTER_NAME; do
  [[ -n "${!NAME:-}" ]] || { echo "$NAME is required" >&2; exit 2; }
done
[[ "$RUN_POLICY" == "fresh" ]] || { echo "H100 validation cases are fresh-only" >&2; exit 2; }
[[ "$GIT" == /* && -x "$GIT" ]] || { echo "invalid GIT" >&2; exit 2; }
readonly FORMAL_GIT_FD FORMAL_GIT_FD_OWNER_PID FORMAL_GIT_COMMAND
hermetic_git() {
  env -i PATH=/usr/bin:/bin LC_ALL=C LANG=C \
    GIT_CONFIG_NOSYSTEM=1 GIT_CONFIG_GLOBAL=/dev/null GIT_CONFIG_COUNT=0 \
    GIT_CEILING_DIRECTORIES=/ GIT_TERMINAL_PROMPT=0 \
    GIT_PROTOCOL_FROM_USER=0 GIT_ALLOW_PROTOCOL=https \
    "$FORMAL_GIT_COMMAND" \
      -c core.fsmonitor=false \
      -c core.hooksPath=/dev/null \
      -c core.filemode=true "$@"
}
[[ "$(hermetic_git -C "$FORMAL_SUBMISSION_EXACT_ROOT" rev-parse HEAD)" == "$FORMAL_SOURCE_COMMIT_SHA" ]] || {
  echo "exact submission root commit mismatch" >&2; exit 2;
}
[[ "$(hermetic_git -C "$FORMAL_SUBMISSION_EXACT_ROOT" rev-parse 'HEAD^{tree}')" == "$FORMAL_SOURCE_TREE_SHA" ]] || {
  echo "exact submission root tree mismatch" >&2; exit 2;
}
[[ -z "$(hermetic_git -C "$FORMAL_SUBMISSION_EXACT_ROOT" status --porcelain=v1 --untracked-files=all)" ]] || {
  echo "exact submission root is dirty" >&2; exit 2;
}
if hermetic_git -C "$FORMAL_SUBMISSION_EXACT_ROOT" symbolic-ref -q HEAD >/dev/null; then
  echo "exact submission root must be detached" >&2
  exit 2
fi
[[ "$PYTHON" == /* && -x "$PYTHON" ]] || { echo "invalid PYTHON" >&2; exit 2; }
[[ "$NVIDIA_SMI" == /* && -x "$NVIDIA_SMI" ]] || { echo "invalid NVIDIA_SMI" >&2; exit 2; }
export PYTHONNOUSERSITE=1

[[ "${FORMAL_TRUSTED_RUNNER_FD:-}" =~ ^[0-9]+$ ]] || {
  echo "validation runner lacks its trusted static descriptor binding" >&2
  exit 2
}
[[ "${BASH_SOURCE[0]}" == "/proc/self/fd/$FORMAL_TRUSTED_RUNNER_FD" ]] || {
  echo "validation runner is not the trusted static open descriptor" >&2
  exit 2
}
[[ "$CANDIDATE_REPOSITORY" == "https://github.com/kikixiong/tabicl-pe.git" ]] || {
  echo "validation candidate repository is not canonical" >&2
  exit 2
}
[[ "$CANDIDATE_REPOSITORY_REF" == "refs/heads/codex/position-identity-v1" ]] || {
  echo "validation candidate ref is not canonical" >&2
  exit 2
}
"$PYTHON" -I -B \
  "/proc/self/fd/$FORMAL_H100_VERIFY_GIT_FD" \
  --git "$GIT" \
  --git-sha256 "$FORMAL_GIT_SHA256" \
  --expected-commit-sha "$FORMAL_SOURCE_COMMIT_SHA" \
  --expected-identity-sha256 "$FORMAL_REPOSITORY_IDENTITY_SHA256" \
  --expected-query-sha256 "$FORMAL_REPOSITORY_QUERY_SHA256" >/dev/null

[[ "$SLURM_JOB_ID" =~ ^[1-9][0-9]{0,19}$ ]] || { echo "invalid SLURM_JOB_ID" >&2; exit 2; }
IFS=',' read -r -a GPU_TOKENS <<<"$CUDA_VISIBLE_DEVICES"
CUDA_VISIBLE_CANONICAL="$(IFS=,; printf '%s' "${GPU_TOKENS[*]}")"
[[ "$CUDA_VISIBLE_CANONICAL" == "$CUDA_VISIBLE_DEVICES" ]] || {
  echo "CUDA_VISIBLE_DEVICES is not a canonical comma-separated token list" >&2
  exit 1
}
[[ "${#GPU_TOKENS[@]}" -eq "$EXPECTED_GPUS" ]] || {
  echo "CUDA_VISIBLE_DEVICES count differs from requested GPUs" >&2
  exit 1
}
GPU_UUIDS=()
declare -A SEEN_GPU_UUIDS=()
readonly NVIDIA_QUERY_MAX_KIB=16
for GPU_TOKEN in "${GPU_TOKENS[@]}"; do
  [[ "$GPU_TOKEN" =~ ^([0-9]+|GPU-[A-Za-z0-9._-]+|MIG-[A-Za-z0-9._-]+)$ ]] || {
    echo "invalid CUDA_VISIBLE_DEVICES token" >&2; exit 1;
  }
  TOKEN_OUTPUT_PATH="$(/usr/bin/mktemp \
    "$H100_CASE_WORK_ROOT/.nvidia-query.${SLURM_JOB_ID}.${GPU_TOKEN}.XXXXXX")"
  if ! (
    ulimit -f "$NVIDIA_QUERY_MAX_KIB"
    "$PYTHON" -I -B \
      "/proc/self/fd/$FORMAL_H100_NVIDIA_LAUNCHER_FD" \
      --retained-fd-owner-pid "$FORMAL_NVIDIA_SMI_FD_OWNER_PID" \
      --retained-fd "$FORMAL_NVIDIA_SMI_FD" \
      --expected-sha256 "$FORMAL_NVIDIA_SMI_SHA256" \
      --query-token "$GPU_TOKEN" \
      --query-fields uuid,name,driver_version \
      >"$TOKEN_OUTPUT_PATH" 2>&1
  ); then
    rm -f -- "$TOKEN_OUTPUT_PATH"
    echo "CUDA-visible nvidia-smi query failed" >&2
    exit 1
  fi
  [[ -f "$TOKEN_OUTPUT_PATH" && ! -L "$TOKEN_OUTPUT_PATH" ]] || {
    rm -f -- "$TOKEN_OUTPUT_PATH"
    echo "CUDA-visible nvidia-smi query output is not regular" >&2
    exit 1
  }
  TOKEN_OUTPUT_SIZE="$(/usr/bin/stat -c %s -- "$TOKEN_OUTPUT_PATH")"
  [[ "$TOKEN_OUTPUT_SIZE" =~ ^[0-9]+$ && "$TOKEN_OUTPUT_SIZE" -le $((NVIDIA_QUERY_MAX_KIB * 1024)) ]] || {
    rm -f -- "$TOKEN_OUTPUT_PATH"
    echo "CUDA-visible nvidia-smi query output exceeds its ceiling" >&2
    exit 1
  }
  mapfile -t TOKEN_ROWS <"$TOKEN_OUTPUT_PATH"
  rm -f -- "$TOKEN_OUTPUT_PATH"
  [[ "${#TOKEN_ROWS[@]}" -eq 1 ]] || {
    echo "CUDA-visible token did not resolve to exactly one GPU" >&2; exit 1;
  }
  [[ "${TOKEN_ROWS[0]//[^,]/}" == ",," ]] || {
    echo "CUDA-visible nvidia-smi row does not have exactly three fields" >&2
    exit 1
  }
  IFS=',' read -r GPU_UUID GPU_NAME GPU_DRIVER <<<"${TOKEN_ROWS[0]}"
  GPU_UUID="${GPU_UUID//[[:space:]]/}"
  GPU_NAME="${GPU_NAME#${GPU_NAME%%[![:space:]]*}}"
  GPU_NAME="${GPU_NAME%${GPU_NAME##*[![:space:]]}}"
  GPU_DRIVER="${GPU_DRIVER#${GPU_DRIVER%%[![:space:]]*}}"
  GPU_DRIVER="${GPU_DRIVER%${GPU_DRIVER##*[![:space:]]}}"
  [[ -n "$GPU_UUID" && "$GPU_NAME" == *H100* && -n "$GPU_DRIVER" ]] || {
    echo "CUDA-visible allocation contains a non-H100 or malformed GPU row" >&2
    exit 1
  }
  [[ -z "${SEEN_GPU_UUIDS[$GPU_UUID]+x}" ]] || {
    echo "CUDA-visible GPU UUIDs are not unique" >&2; exit 1;
  }
  SEEN_GPU_UUIDS["$GPU_UUID"]=1
  GPU_UUIDS+=("$GPU_UUID")
done
export FORMAL_VISIBLE_GPU_TOKENS="$CUDA_VISIBLE_DEVICES"
export FORMAL_VISIBLE_GPU_UUIDS="$(IFS=,; echo "${GPU_UUIDS[*]}")"

"$PYTHON" -I -B \
  "/proc/self/fd/$FORMAL_H100_VERIFY_FILESYSTEM_FD" \
  --work-root "$H100_CASE_WORK_ROOT" \
  --artifact-root "$H100_VALIDATION_ROOT" \
  --work-label "H100 case work root" \
  --artifact-label "H100 artifact filesystem" >/dev/null
"$PYTHON" -I -B \
  "/proc/self/fd/$FORMAL_H100_VERIFY_ENVIRONMENT_FD" \
  --exact-root "$FORMAL_SUBMISSION_EXACT_ROOT" \
  --expected-sha256 "$FORMAL_EXPECTED_ENVIRONMENT_SHA256" \
  --expected-gpus "$EXPECTED_GPUS" >/dev/null

CHECKOUT_PARENT="$(mktemp -d "$H100_CASE_WORK_ROOT/${VALIDATION_CASE_ID}.XXXXXX")"
CHECKOUT="$CHECKOUT_PARENT/candidate"
cleanup() { rm -rf -- "$CHECKOUT_PARENT"; }
trap 'status=$?; trap - EXIT HUP INT TERM; cleanup; exit "$status"' EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM
RUNTIME_HOME="$CHECKOUT_PARENT/home"
mkdir -m 700 -- "$RUNTIME_HOME"
[[ -d "$RUNTIME_HOME" && ! -L "$RUNTIME_HOME" ]] || {
  echo "node-local H100 runtime HOME is not a physical directory" >&2
  exit 2
}
export HOME="$RUNTIME_HOME" USER=tabicl LOGNAME=tabicl
export XDG_CACHE_HOME="$RUNTIME_HOME/.cache"
export XDG_CONFIG_HOME="$RUNTIME_HOME/.config"
export XDG_DATA_HOME="$RUNTIME_HOME/.local/share"
readonly RUNTIME_HOME
hermetic_git -C "$CHECKOUT_PARENT" clone --quiet --no-hardlinks --no-checkout -- \
  "$CANDIDATE_REPOSITORY" candidate
[[ "$(hermetic_git -C "$CHECKOUT" remote get-url origin)" == "$CANDIDATE_REPOSITORY" ]]
[[ "$(hermetic_git -C "$CHECKOUT" rev-parse refs/remotes/origin/codex/position-identity-v1)" == "$FORMAL_SOURCE_COMMIT_SHA" ]]
hermetic_git -C "$CHECKOUT" checkout --quiet --detach "$FORMAL_SOURCE_COMMIT_SHA"
[[ "$(hermetic_git -C "$CHECKOUT" rev-parse HEAD)" == "$FORMAL_SOURCE_COMMIT_SHA" ]]
[[ "$(hermetic_git -C "$CHECKOUT" rev-parse 'HEAD^{tree}')" == "$FORMAL_SOURCE_TREE_SHA" ]]
[[ -z "$(hermetic_git -C "$CHECKOUT" status --porcelain=v1 --untracked-files=all)" ]]
if hermetic_git -C "$CHECKOUT" symbolic-ref -q HEAD >/dev/null; then
  echo "validation checkout must be detached" >&2
  exit 1
fi

export TABICL_EXACT_ROOT="$CHECKOUT"
export PYTHONPATH="$CHECKOUT/src"
"$CHECKOUT/scripts/run_h100_identity_maxseq_smoke.sh" "$VALIDATION_CASE_ID"
