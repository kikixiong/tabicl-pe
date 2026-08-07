#!/bin/bash
set -u

STATE_DIR="${BASH_SOURCE[0]%/*}"
cd "$STATE_DIR" || exit 97
COUNT=0
if [[ -f squeue_count ]]; then read -r COUNT < squeue_count; fi
COUNT=$((COUNT + 1))
printf '%s\n' "$COUNT" > squeue_count
printf 'squeue' >> calls.log
printf ' <%s>' "$@" >> calls.log
printf '\n' >> calls.log
if [[ -f fail_squeue_at ]]; then
  read -r FAIL_AT < fail_squeue_at
  [[ "$COUNT" -ne "$FAIL_AT" ]] || exit 45
fi
if [[ -f squeue_banner ]]; then
  while IFS= read -r LINE; do printf '%s\n' "$LINE"; done < squeue_banner
fi
CLUSTER=""
NAME=""
JOB_ID=""
for ARG in "$@"; do
  case "$ARG" in
    --clusters=*) CLUSTER="${ARG#--clusters=}" ;;
    --name=*) NAME="${ARG#--name=}" ;;
    --jobs=*) JOB_ID="${ARG#--jobs=}" ;;
  esac
done
[[ -n "$NAME" || "$JOB_ID" =~ ^[1-9][0-9]*$ ]] || exit 91
if [[ -f hide_squeue_until ]]; then
  read -r HIDE_UNTIL < hide_squeue_until
  [[ "$COUNT" -gt "$HIDE_UNTIL" ]] || exit 0
fi
[[ -f jobs.tsv ]] || exit 0
if [[ -n "$JOB_ID" ]]; then
  awk -F'|' -v id="$JOB_ID" -v cluster="$CLUSTER" '
    $1 == id && $5 == cluster {printf "%s|%s|%s\n", $1, $3, $4}
  ' jobs.tsv
  exit 0
fi
if [[ -f malformed_exact_name_until ]]; then
  read -r MALFORMED_UNTIL < malformed_exact_name_until
  if [[ "$COUNT" -le "$MALFORMED_UNTIL" ]]; then
    printf '1001|%s|PENDING\n' "$NAME"
    exit 0
  fi
fi
awk -F'|' -v name="$NAME" -v cluster="$CLUSTER" '
  $2 == name && $5 == cluster {printf "%s|%s|%s|%s\n", $1, $2, $3, $4}
' jobs.tsv
