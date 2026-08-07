#!/bin/bash
set -u

STATE_DIR="${BASH_SOURCE[0]%/*}"
cd "$STATE_DIR" || exit 97
COUNT=0
if [[ -f sacct_count ]]; then read -r COUNT < sacct_count; fi
COUNT=$((COUNT + 1))
printf '%s\n' "$COUNT" > sacct_count
printf 'sacct' >> calls.log
printf ' <%s>' "$@" >> calls.log
printf '\n' >> calls.log
if [[ -f fail_sacct_at ]]; then
  read -r FAIL_AT < fail_sacct_at
  [[ "$COUNT" -ne "$FAIL_AT" ]] || exit 46
fi
JOB_ID=""
for ARG in "$@"; do
  case "$ARG" in
    --jobs=*) JOB_ID="${ARG#--jobs=}" ;;
  esac
done
[[ "$JOB_ID" =~ ^[1-9][0-9]*$ ]] || exit 91
if [[ -f hide_sacct_until ]]; then
  read -r HIDE_UNTIL < hide_sacct_until
  [[ "$COUNT" -gt "$HIDE_UNTIL" ]] || exit 0
fi
[[ -f accounting.tsv ]] || exit 0
awk -F'|' -v id="$JOB_ID" '
  $1 == id {printf "%s|%s|\n", $1, $2}
' accounting.tsv
