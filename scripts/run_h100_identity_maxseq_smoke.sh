#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "--rank-worker" ]]; then
  shift
  CASE_ID="${1:?case ID required}"
  ROOT="${TABICL_EXACT_ROOT:?}"
  : "${PYTHON:?}"
  : "${FORMAL_NVIDIA_SMI_FD:?}"
  : "${FORMAL_NVIDIA_SMI_FD_OWNER_PID:?}"
  : "${FORMAL_NVIDIA_SMI_SHA256:?}"
  if [[ "$FORMAL_NVIDIA_SMI_FD_OWNER_PID" != "$BASHPID" ]]; then
    [[ "$FORMAL_NVIDIA_SMI_FD_OWNER_PID" =~ ^[1-9][0-9]*$ ]] || {
      echo "invalid retained NVIDIA-SMI owner PID" >&2; exit 2;
    }
    [[ "$FORMAL_NVIDIA_SMI_FD" =~ ^[0-9]+$ ]] || {
      echo "invalid retained NVIDIA-SMI descriptor" >&2; exit 2;
    }
    exec "$PYTHON" -I -B "$ROOT/scripts/exec_digest_bound_nvidia_smi.py" \
      --retained-fd-owner-pid "$FORMAL_NVIDIA_SMI_FD_OWNER_PID" \
      --retained-fd "$FORMAL_NVIDIA_SMI_FD" \
      --expected-sha256 "$FORMAL_NVIDIA_SMI_SHA256" -- \
      "${BASH_SOURCE[0]}" --rank-worker "$CASE_ID"
  fi
  [[ "$NVIDIA_SMI" == "/proc/self/fd/$FORMAL_NVIDIA_SMI_FD" && -x "$NVIDIA_SMI" ]] || {
    echo "rank NVIDIA-SMI is not its reacquired verified descriptor" >&2; exit 2;
  }
  RANK_ID="${LOCAL_RANK:?}"
  exec "$PYTHON" -I -B "$ROOT/scripts/run_exact_tabicl.py" \
    --archive-root "$ROOT" --source-manifest "$FORMAL_SOURCE_MANIFEST" \
    --expected-manifest-sha256 "$FORMAL_SOURCE_SHA256" \
    --expected-commit-sha "$FORMAL_SOURCE_COMMIT_SHA" \
    --expected-tree-sha "$FORMAL_SOURCE_TREE_SHA" \
    --action h100-validation \
    --source-attestation-output "$CASE_ROOT/source-attestation.json" \
    --source-attestation-max-bytes "$FORMAL_ATTESTATION_CEILING_BYTES" \
    --runtime-evidence-output "$CASE_ROOT/runtime-evidence.json" \
    --completion-receipt-output "$CASE_ROOT/action-completion.json" \
    --case-id "$CASE_ID" --artifact-identity-sha256 "$VALIDATION_ARTIFACT_IDENTITY_SHA256" -- \
    --output-dir "$CASE_ROOT/compute" \
    --checkpoint-ceiling-bytes "$CHECKPOINT_CEILING_BYTES"
fi

CASE_ID="${1:-}"
case "$CASE_ID" in
  stage1_rope_one_step|stage1_temporary_one_step|stage1_none_one_step)
    UTILIZATION_REQUIRED=false; EXPECTED_GPUS=1 ;;
  stage2_rope_maxseq10240|stage2_temporary_maxseq10240|stage2_none_maxseq10240|\
  stage3_rope_maxseq60000_recompute|stage3_temporary_maxseq60000_recompute|stage3_none_maxseq60000_recompute)
    UTILIZATION_REQUIRED=true; EXPECTED_GPUS=1 ;;
  temporary_cuda_rng_resume|prior_dataloader_resume)
    UTILIZATION_REQUIRED=false; EXPECTED_GPUS=1 ;;
  nccl_2gpu)
    UTILIZATION_REQUIRED=false; EXPECTED_GPUS=2 ;;
  *) echo "unknown H100 validation case: $CASE_ID" >&2; exit 2 ;;
esac

ROOT="${TABICL_EXACT_ROOT:?}"; ROOT="$(cd "$ROOT" && pwd -P)"
: "${PYTHON:?}"; [[ "$PYTHON" == /* && -x "$PYTHON" ]] || { echo "invalid PYTHON" >&2; exit 2; }
: "${NVIDIA_SMI:?}"
: "${FORMAL_NVIDIA_SMI_FD:?}"
: "${FORMAL_NVIDIA_SMI_FD_OWNER_PID:?}"
: "${FORMAL_NVIDIA_SMI_SHA256:?}"
if [[ "$FORMAL_NVIDIA_SMI_FD_OWNER_PID" != "$BASHPID" ]]; then
  [[ "$FORMAL_NVIDIA_SMI_FD_OWNER_PID" =~ ^[1-9][0-9]*$ ]] || {
    echo "invalid retained NVIDIA-SMI owner PID" >&2; exit 2;
  }
  [[ "$FORMAL_NVIDIA_SMI_FD" =~ ^[0-9]+$ ]] || {
    echo "invalid retained NVIDIA-SMI descriptor" >&2; exit 2;
  }
  exec "$PYTHON" -I -B "$ROOT/scripts/exec_digest_bound_nvidia_smi.py" \
    --retained-fd-owner-pid "$FORMAL_NVIDIA_SMI_FD_OWNER_PID" \
    --retained-fd "$FORMAL_NVIDIA_SMI_FD" \
    --expected-sha256 "$FORMAL_NVIDIA_SMI_SHA256" -- \
    "${BASH_SOURCE[0]}" "$CASE_ID"
fi
[[ "$NVIDIA_SMI" == "/proc/self/fd/$FORMAL_NVIDIA_SMI_FD" && -x "$NVIDIA_SMI" ]] || {
  echo "main NVIDIA-SMI is not its reacquired verified descriptor" >&2; exit 2;
}
export PYTHONPATH="$ROOT/src" PYTHONNOUSERSITE=1
for NAME in FORMAL_SOURCE_MANIFEST FORMAL_SOURCE_SHA256 FORMAL_SOURCE_COMMIT_SHA \
  FORMAL_SOURCE_TREE_SHA VALIDATION_ARTIFACT_IDENTITY_SHA256 \
  FORMAL_EXPECTED_ENVIRONMENT_SHA256 \
  H100_VALIDATION_ROOT FORMAL_RUN_LOG_CEILING_BYTES FORMAL_GPU_MONITOR_CEILING_BYTES \
  FORMAL_ATTESTATION_CEILING_BYTES; do
  [[ -n "${!NAME:-}" ]] || { echo "$NAME is required" >&2; exit 2; }
done
: "${CHECKPOINT_CEILING_BYTES:?CHECKPOINT_CEILING_BYTES is required}"

CASE_ROOT="$H100_VALIDATION_ROOT/$CASE_ID"
export CASE_ROOT TABICL_EXACT_ROOT="$ROOT" TABICL_ATTESTED_CASE_ID="$CASE_ID"
export TABICL_ATTESTED_ARTIFACT_SHA256="$VALIDATION_ARTIFACT_IDENTITY_SHA256"
export FORMAL_GPU_EXPECTED_COUNT="$EXPECTED_GPUS"
if ! "$PYTHON" -I -B - "$H100_VALIDATION_ROOT" "$CASE_ID" <<'PY'
import os
from pathlib import PurePosixPath
import sys


root = sys.argv[1]
case_id = sys.argv[2]
if (
    not root.startswith("/")
    or root.startswith("//")
    or os.path.normpath(root) != root
):
    raise SystemExit("H100_VALIDATION_ROOT must be a normalized absolute path")
if "/" in case_id or case_id in {"", ".", ".."}:
    raise SystemExit("validation case ID is not a safe directory name")
if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_DIRECTORY"):
    raise SystemExit("no-follow directory traversal is unavailable")

flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
directory_fd = os.open("/", flags)
try:
    for component in PurePosixPath(root).parts[1:]:
        next_fd = os.open(component, flags, dir_fd=directory_fd)
        os.close(directory_fd)
        directory_fd = next_fd
    os.mkdir(case_id, mode=0o700, dir_fd=directory_fd)
    case_fd = os.open(case_id, flags, dir_fd=directory_fd)
    os.close(case_fd)
    os.fsync(directory_fd)
except OSError as error:
    raise SystemExit(
        f"physical validation-root traversal or fresh case creation failed: {error}"
    ) from error
finally:
    os.close(directory_fd)
PY
then
  echo "validation case is fresh-only and its precreated physical parent must exist" >&2
  exit 1
fi
CSV_PATH="$CASE_ROOT/gpu.csv"
JSONL_PATH="$CASE_ROOT/gpu.jsonl"
COMPUTE_LOG="$CASE_ROOT/compute.log"
DIGEST_BOUND_COMPUTE=(
  "$PYTHON" -I -B "$ROOT/scripts/exec_digest_bound_nvidia_smi.py"
  --retained-fd-owner-pid "$FORMAL_NVIDIA_SMI_FD_OWNER_PID"
  --retained-fd "$FORMAL_NVIDIA_SMI_FD"
  --expected-sha256 "$FORMAL_NVIDIA_SMI_SHA256" --
)

if [[ "$CASE_ID" == "nccl_2gpu" ]]; then
  COMPUTE=(
    "${DIGEST_BOUND_COMPUTE[@]}"
    "$PYTHON" -I -B -m torch.distributed.run --standalone --nproc_per_node=2
    --no-python "$ROOT/scripts/run_h100_identity_maxseq_smoke.sh"
    --rank-worker "$CASE_ID"
  )
else
  COMPUTE=(
    "${DIGEST_BOUND_COMPUTE[@]}"
    "$PYTHON" -I -B "$ROOT/scripts/run_exact_tabicl.py"
    --archive-root "$ROOT" --source-manifest "$FORMAL_SOURCE_MANIFEST"
    --expected-manifest-sha256 "$FORMAL_SOURCE_SHA256"
    --expected-commit-sha "$FORMAL_SOURCE_COMMIT_SHA"
    --expected-tree-sha "$FORMAL_SOURCE_TREE_SHA"
    --action h100-validation
    --source-attestation-output "$CASE_ROOT/source-attestation.json"
    --source-attestation-max-bytes "$FORMAL_ATTESTATION_CEILING_BYTES"
    --runtime-evidence-output "$CASE_ROOT/runtime-evidence.json"
    --completion-receipt-output "$CASE_ROOT/action-completion.json"
    --case-id "$CASE_ID" --artifact-identity-sha256 "$VALIDATION_ARTIFACT_IDENTITY_SHA256" --
    --output-dir "$CASE_ROOT/compute"
    --checkpoint-ceiling-bytes "$CHECKPOINT_CEILING_BYTES"
  )
fi

if [[ "$UTILIZATION_REQUIRED" == true ]]; then
  PYTHON="$PYTHON" "$ROOT/scripts/formal_run_with_gpu_monitor.sh" \
    "$CSV_PATH" "$JSONL_PATH" "$FORMAL_GPU_MONITOR_CEILING_BYTES" -- \
    env PYTHON="$PYTHON" "$ROOT/scripts/run_with_durable_log.sh" \
      "$COMPUTE_LOG" "$FORMAL_RUN_LOG_CEILING_BYTES" "${COMPUTE[@]}"
else
  [[ ! -e "$CSV_PATH" && ! -L "$CSV_PATH" && ! -e "$JSONL_PATH" && ! -L "$JSONL_PATH" ]] || {
    echo "functional case must not have GPU monitor artifacts" >&2; exit 1;
  }
  PYTHON="$PYTHON" "$ROOT/scripts/run_with_durable_log.sh" \
    "$COMPUTE_LOG" "$FORMAL_RUN_LOG_CEILING_BYTES" "${COMPUTE[@]}"
  [[ ! -e "$CSV_PATH" && ! -L "$CSV_PATH" && ! -e "$JSONL_PATH" && ! -L "$JSONL_PATH" ]] || {
    echo "functional case created unbound GPU monitor artifacts" >&2; exit 1;
  }
fi

"$PYTHON" -I -B "$ROOT/scripts/reject_nonfinite_log.py" \
  "$COMPUTE_LOG" --max-bytes "$FORMAL_RUN_LOG_CEILING_BYTES"

# The scheduler transaction precommits the runtime-environment digest for each
# GPU-count cohort.  Check it inside every independently cloned case rather
# than waiting for the twelve-case assembler to discover drift after all jobs
# have consumed their allocations.
"$PYTHON" -I -B - \
  "$CASE_ROOT/runtime-evidence.json" \
  "$FORMAL_EXPECTED_ENVIRONMENT_SHA256" "$EXPECTED_GPUS" "$CASE_ID" <<'PY'
import hashlib
import json
from pathlib import Path
import re
import sys


def reject(message):
    raise ValueError(message)


def pairs(items):
    value = {}
    for key, item in items:
        if key in value:
            reject(f"duplicate runtime-evidence key: {key}")
        value[key] = item
    return value


path = Path(sys.argv[1])
expected = sys.argv[2]
world_size = int(sys.argv[3])
case_id = sys.argv[4]
if re.fullmatch(r"[0-9a-f]{64}", expected) is None:
    reject("expected environment digest is malformed")
raw = path.read_bytes()
try:
    value = json.loads(
        raw.decode("utf-8"),
        object_pairs_hook=pairs,
        parse_constant=lambda token: reject(f"invalid constant: {token}"),
    )
except (UnicodeDecodeError, json.JSONDecodeError) as error:
    raise ValueError("runtime evidence is invalid JSON") from error
canonical = lambda item: json.dumps(
    item,
    sort_keys=True,
    separators=(",", ":"),
    ensure_ascii=False,
    allow_nan=False,
).encode("utf-8")
if raw != canonical(value) + b"\n":
    reject("runtime evidence is not canonical newline-terminated JSON")
if set(value) != {"schema_version", "kind", "payload", "sha256"}:
    reject("runtime evidence envelope keys mismatch")
body = {key: value[key] for key in ("schema_version", "kind", "payload")}
if value["sha256"] != hashlib.sha256(canonical(body)).hexdigest():
    reject("runtime evidence self-hash mismatch")
payload = value["payload"]
environment = payload.get("environment")
if payload.get("case_id") != case_id or payload.get("world_size") != world_size:
    reject("runtime evidence case/world-size mismatch")
if not isinstance(environment, dict) or environment.get("sha256") != expected:
    reject("runtime environment differs from the precommitted digest")
if environment.get("payload", {}).get("visible_cuda_device_count") != world_size:
    reject("runtime environment visible-device count mismatch")
PY

if [[ "$UTILIZATION_REQUIRED" == true ]]; then
  IFS=',' read -r -a GPU_UUIDS <<<"${FORMAL_VISIBLE_GPU_UUIDS:?}"
  [[ "${#GPU_UUIDS[@]}" -eq "$EXPECTED_GPUS" ]] || { echo "unexpected visible GPU count" >&2; exit 1; }
  SUMMARY_ARGS=()
  for GPU_UUID in "${GPU_UUIDS[@]}"; do SUMMARY_ARGS+=(--expected-gpu-uuid "$GPU_UUID"); done
  PYTHON="$PYTHON" "$ROOT/scripts/run_with_durable_log.sh" \
    "$CASE_ROOT/gpu-summary.json" "$FORMAL_RUN_LOG_CEILING_BYTES" \
    "$PYTHON" -I -B "$ROOT/scripts/summarize_formal_gpu_usage.py" "$CSV_PATH" \
      --utilization-required --expected-gpus "$EXPECTED_GPUS" "${SUMMARY_ARGS[@]}"
fi
