#!/usr/bin/env bash
#SBATCH --job-name=tabicl-pilot-monitor
#SBATCH --partition=normal
#SBATCH --qos=long
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G

set -euo pipefail
: "${PILOT_ROOT:?PILOT_ROOT is required}"
: "${PILOT_RECEIPT:?PILOT_RECEIPT is required}"

ROOT="$(cd "$PILOT_ROOT" && pwd -P)"
[[ "$(git -C "$ROOT" rev-parse --show-toplevel)" == "$ROOT" ]] || {
  echo "PILOT_ROOT is not the expected Git root" >&2; exit 2;
}
exec "$ROOT/.venv/bin/python" "$ROOT/scripts/monitor_pilot_continuation.py" \
  --receipt "$PILOT_RECEIPT" --interval-seconds 1800
