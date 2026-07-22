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
[[ "${1:-}" == release && "${2:-}" =~ ^[1-9][0-9]*$ && "$#" -eq 2 ]] || exit 91
read -r LEDGER < ledger_path
read -r RECEIPT < receipt_path
[[ -s "$LEDGER" && -s "$RECEIPT" ]] || exit 92
if [[ -f fail_release_at ]]; then
  read -r FAIL_AT < fail_release_at
  [[ "$COUNT" -ne "$FAIL_AT" ]] || exit 42
fi
