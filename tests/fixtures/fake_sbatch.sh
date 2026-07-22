#!/bin/bash
set -u

STATE_DIR="${BASH_SOURCE[0]%/*}"
cd "$STATE_DIR" || exit 97
STATE_DIR="$PWD"
COUNT=0
if [[ -f sbatch_count ]]; then read -r COUNT < sbatch_count; fi
COUNT=$((COUNT + 1))
printf '%s\n' "$COUNT" > sbatch_count
printf 'sbatch' >> calls.log
printf ' <%s>' "$@" >> calls.log
printf '\n' >> calls.log

HELD=0
ONE_GPU=0
EXPORT_VALUE=""
for ARG in "$@"; do
  [[ "$ARG" == "--hold" ]] && HELD=1
  [[ "$ARG" == "--gres=gpu:1" ]] && ONE_GPU=1
  case "$ARG" in
    --export=*) EXPORT_VALUE="${ARG#--export=}" ;;
  esac
  [[ "$ARG" != *"gpu:2"* ]] || exit 91
  [[ "$ARG" != "--export=ALL"* ]] || exit 92
done
[[ "$HELD" -eq 1 && "$ONE_GPU" -eq 1 && -n "$EXPORT_VALUE" ]] || exit 93
[[ "$EXPORT_VALUE" != ALL,* && "$EXPORT_VALUE" != *,ALL,* ]] || exit 94
[[ "$EXPORT_VALUE" == *"PYTHONNOUSERSITE=1"* ]] || exit 95
[[ "$EXPORT_VALUE" == *"PYTHONPATH="*"/src"* ]] || exit 96
[[ "$EXPORT_VALUE" == *"GIT=/usr/bin/git"* ]] || exit 88
[[ "$EXPORT_VALUE" == *"NVIDIA_SMI=/usr/bin/true"* ]] || exit 90
[[ "$EXPORT_VALUE" != *"PYTHONHOME="* && "$EXPORT_VALUE" != *"poison"* ]] || exit 98

ARTIFACT_ROOT=""
IFS=',' read -r -a EXPORTS <<< "$EXPORT_VALUE"
for ITEM in "${EXPORTS[@]}"; do
  case "$ITEM" in
    FORMAL_ARTIFACT_ROOT=*) ARTIFACT_ROOT="${ITEM#FORMAL_ARTIFACT_ROOT=}" ;;
  esac
done
[[ -n "$ARTIFACT_ROOT" ]] || exit 99
[[ -f "${ARTIFACT_ROOT%/*}/capacity.called" ]] || exit 89
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
if [[ -f cluster_suffix_at ]]; then
  read -r CLUSTER_SUFFIX_AT < cluster_suffix_at
  if [[ "$COUNT" -eq "$CLUSTER_SUFFIX_AT" ]]; then
    printf '%s;%s\n' "$((1000 + COUNT))" "cluster-a"
    exit 0
  fi
fi
printf '%s\n' "$((1000 + COUNT))"
