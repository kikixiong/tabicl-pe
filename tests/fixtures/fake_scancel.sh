#!/bin/bash
set -u

STATE_DIR="${BASH_SOURCE[0]%/*}"
cd "$STATE_DIR" || exit 97
printf 'scancel' >> calls.log
printf ' <%s>' "$@" >> calls.log
printf '\n' >> calls.log
CLUSTER=""
if [[ "${1:-}" == --clusters=* ]]; then
  CLUSTER="${1#--clusters=}"
  shift
fi
[[ "$#" -eq 1 && "${1:-}" =~ ^[1-9][0-9]*$ ]] || exit 91
if [[ -f signal_parent_on_cancel ]]; then
  kill -TERM "$PPID"
fi
if [[ -f fail_cancel_ids ]]; then
  while IFS= read -r FAILED; do
    [[ "$1" != "$FAILED" ]] || exit 43
  done < fail_cancel_ids
fi
if [[ -f sleep_cancel_ids ]]; then
  while IFS= read -r SLEEPING; do
    [[ "$1" != "$SLEEPING" ]] || /bin/sleep 60
  done < sleep_cancel_ids
fi
awk -F'|' -v id="$1" -v cluster="$CLUSTER" '
  $1 == id && $5 == cluster {printf "%s|CANCELLED\n", $1; found=1}
  END {if (!found) exit 44}
' jobs.tsv >> accounting.tsv || exit $?
awk -F'|' -v id="$1" -v cluster="$CLUSTER" '
  !($1 == id && $5 == cluster) {print}
  ($1 == id && $5 == cluster) {found=1}
  END {if (!found) exit 44}
' jobs.tsv > jobs.tsv.next || exit $?
mv jobs.tsv.next jobs.tsv
