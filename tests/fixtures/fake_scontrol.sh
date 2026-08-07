#!/bin/bash
set -u

STATE_DIR="${BASH_SOURCE[0]%/*}"
cd "$STATE_DIR" || exit 97
COUNT=0
if [[ -f release_count ]]; then read -r COUNT < release_count; fi
COUNT=$((COUNT + 1))
printf '%s\n' "$COUNT" > release_count
printf 'scontrol' >> calls.log
printf ' <%s>' "$@" >> calls.log
printf '\n' >> calls.log

CLUSTER=""
if [[ "${1:-}" == --clusters=* ]]; then
  CLUSTER="${1#--clusters=}"
  shift
fi
[[ "${1:-}" == release && "${2:-}" =~ ^[1-9][0-9]*$ && "$#" -eq 2 ]] || exit 91
JOB_ID="$2"
read -r LEDGER < ledger_path
read -r RECEIPT < receipt_path
[[ -s "$LEDGER" && -s "$RECEIPT" ]] || exit 92
if [[ -f fail_release_at ]]; then
  read -r FAIL_AT < fail_release_at
  [[ "$COUNT" -ne "$FAIL_AT" ]] || exit 42
fi
awk -F'|' -v OFS='|' -v id="$JOB_ID" -v cluster="$CLUSTER" '
  $1 == id && $5 == cluster {$3="PENDING"; $4="Resources"; found=1}
  {print}
  END {if (!found) exit 44}
' jobs.tsv > jobs.tsv.next || exit $?
mv jobs.tsv.next jobs.tsv
if [[ -f delete_scancel_after_release_at ]]; then
  read -r DELETE_AT < delete_scancel_after_release_at
  if [[ "$COUNT" -eq "$DELETE_AT" && -x scancel ]]; then
    mv scancel scancel.missing
  fi
fi
