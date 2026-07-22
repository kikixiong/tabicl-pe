#!/bin/bash
set -u

STATE_DIR="${BASH_SOURCE[0]%/*}"
cd "$STATE_DIR" || exit 97
printf 'scancel' >> calls.log
printf ' <%s>' "$@" >> calls.log
printf '\n' >> calls.log
[[ "$#" -eq 1 && "${1:-}" =~ ^[1-9][0-9]*$ ]] || exit 91
if [[ -f fail_cancel_ids ]]; then
  while IFS= read -r FAILED; do
    [[ "$1" != "$FAILED" ]] || exit 43
  done < fail_cancel_ids
fi
