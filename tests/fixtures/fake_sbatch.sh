#!/bin/bash
set -u

STATE_DIR="${BASH_SOURCE[0]%/*}"
cd "$STATE_DIR" || exit 97
COUNT=0
if [[ -f sbatch_count ]]; then read -r COUNT < sbatch_count; fi
COUNT=$((COUNT + 1))
printf '%s\n' "$COUNT" > sbatch_count
printf 'sbatch' >> calls.log
printf ' <%s>' "$@" >> calls.log
printf '\n' >> calls.log
ARGS=("$@")
SCRIPT_PATH="${ARGS[${#ARGS[@]}-1]}"
[[ "$SCRIPT_PATH" =~ ^/proc/self/fd/[0-9]+$ && -r "$SCRIPT_PATH" ]] || exit 82

HELD=0
ONE_GPU=0
EXPORT_VALUE=""
JOB_NAME=""
STDOUT_PATH=""
STDERR_PATH=""
CHDIR_PATH=""
for ARG in "$@"; do
  [[ "$ARG" == "--hold" ]] && HELD=1
  [[ "$ARG" == "--gres=gpu:1" ]] && ONE_GPU=1
  case "$ARG" in
    --export=*) EXPORT_VALUE="${ARG#--export=}" ;;
    --job-name=*) JOB_NAME="${ARG#--job-name=}" ;;
    --output=*) STDOUT_PATH="${ARG#--output=}" ;;
    --error=*) STDERR_PATH="${ARG#--error=}" ;;
    --chdir=*) CHDIR_PATH="${ARG#--chdir=}" ;;
  esac
  [[ "$ARG" != *"gpu:2"* ]] || exit 91
  [[ "$ARG" != "--export=ALL"* ]] || exit 92
done
[[ "$HELD" -eq 1 && "$ONE_GPU" -eq 1 && -n "$EXPORT_VALUE" ]] || exit 93
[[ -n "$JOB_NAME" && -n "$STDOUT_PATH" && -n "$STDERR_PATH" && -d "$CHDIR_PATH" ]] || exit 87
[[ "$STDOUT_PATH" == "$CHDIR_PATH"/scheduler-logs/* ]] || exit 86
[[ "$STDERR_PATH" == "$CHDIR_PATH"/scheduler-logs/* ]] || exit 85
[[ "$EXPORT_VALUE" != ALL,* && "$EXPORT_VALUE" != *,ALL,* ]] || exit 94
[[ "$EXPORT_VALUE" == *"PYTHONNOUSERSITE=1"* ]] || exit 95
[[ "$EXPORT_VALUE" == *"PYTHONPATH="*"/src"* ]] || exit 96
[[ "$EXPORT_VALUE" == *"NVIDIA_SMI=/usr/bin/true"* ]] || exit 90
[[ "$EXPORT_VALUE" == *"FORMAL_TRANSACTION_ID="* ]] || exit 84
[[ "$EXPORT_VALUE" == *"FORMAL_EXPECTED_JOB_NAME=$JOB_NAME"* ]] || exit 83
[[ "$EXPORT_VALUE" != *"PYTHONHOME="* && "$EXPORT_VALUE" != *"poison"* ]] || exit 98

ARTIFACT_ROOT=""
GIT_PATH=""
IFS=',' read -r -a EXPORTS <<< "$EXPORT_VALUE"
for ITEM in "${EXPORTS[@]}"; do
  case "$ITEM" in
    FORMAL_ARTIFACT_ROOT=*) ARTIFACT_ROOT="${ITEM#FORMAL_ARTIFACT_ROOT=}" ;;
    GIT=*) GIT_PATH="${ITEM#GIT=}" ;;
  esac
done
[[ "$GIT_PATH" == /* && -x "$GIT_PATH" ]] || exit 88
[[ -n "$ARTIFACT_ROOT" && "$ARTIFACT_ROOT" == "$CHDIR_PATH" ]] || exit 99
[[ -f "${ARTIFACT_ROOT%/*}/capacity.called" ]] || exit 89
: > "$STDOUT_PATH"
: > "$STDERR_PATH"
printf '%s\n' "$ARTIFACT_ROOT/transaction-ledger.json" > ledger_path
printf '%s\n' "$ARTIFACT_ROOT/submission-receipt.json" > receipt_path
if [[ -f precreate_receipt_at ]]; then
  read -r PRECREATE_AT < precreate_receipt_at
  if [[ "$COUNT" -eq "$PRECREATE_AT" ]]; then
    printf '%s\n' '{"invalid":true}' > "$ARTIFACT_ROOT/submission-receipt.json"
  fi
fi

if [[ -f fail_sbatch_at ]]; then
  read -r FAIL_AT < fail_sbatch_at
  [[ "$COUNT" -ne "$FAIL_AT" ]] || exit 41
fi

JOB_ID="$((1000 + COUNT))"
CLUSTER=""
if [[ -f cluster_suffix_at ]]; then
  read -r CLUSTER_SUFFIX_AT < cluster_suffix_at
  [[ "$COUNT" -ne "$CLUSTER_SUFFIX_AT" ]] || CLUSTER="cluster-a"
fi
printf '%s|%s|PENDING|JobHeldUser|%s\n' "$JOB_ID" "$JOB_NAME" "$CLUSTER" >> jobs.tsv

if [[ -f signal_hup_sbatch_at ]]; then
  read -r SIGNAL_HUP_AT < signal_hup_sbatch_at
  if [[ "$COUNT" -eq "$SIGNAL_HUP_AT" ]]; then
    kill -HUP "$PPID"
    /bin/sleep 60
  fi
fi
if [[ -f sleep_sbatch_at ]]; then
  read -r SLEEP_AT < sleep_sbatch_at
  [[ "$COUNT" -ne "$SLEEP_AT" ]] || /bin/sleep 60
fi

if [[ -f response_loss_at ]]; then
  read -r RESPONSE_LOSS_AT < response_loss_at
  [[ "$COUNT" -ne "$RESPONSE_LOSS_AT" ]] || exit 41
fi
if [[ -f empty_at ]]; then
  read -r EMPTY_AT < empty_at
  [[ "$COUNT" -ne "$EMPTY_AT" ]] || exit 0
fi
if [[ -f malformed_at ]]; then
  read -r MALFORMED_AT < malformed_at
  if [[ "$COUNT" -eq "$MALFORMED_AT" ]]; then printf 'not-a-job-id\n'; exit 0; fi
fi
if [[ -f duplicate_at ]]; then
  read -r DUPLICATE_AT < duplicate_at
  if [[ "$COUNT" -eq "$DUPLICATE_AT" ]]; then printf '1001\n'; exit 0; fi
fi
if [[ -n "$CLUSTER" ]]; then
  printf '%s;%s\n' "$JOB_ID" "$CLUSTER"
else
  printf '%s\n' "$JOB_ID"
fi
