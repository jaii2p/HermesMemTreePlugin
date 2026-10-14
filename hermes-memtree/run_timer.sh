#!/usr/bin/env bash
# The Tree's timer: the watchdog waking the model up to check on it.
#
# Hosted in place, like run_server.sh: pins PYTHONPATH and MEMTREE_HOME to its own
# data dir, so wakes are recorded under ./var/memtree/{timer.json,timer.log.jsonl}.
#
#   ./run_timer.sh --arm --every 10m      # arm it (and exit)
#   ./run_timer.sh --run                  # stay up, print each wake
#   ./run_timer.sh --check                # is a wake waiting?
#   ./run_timer.sh --status
#   ./run_timer.sh --stop
#
# A wake never writes to the Tree: it brings the memory block and asks whether you
# want to write anything down. Nothing is stored unless the model chooses to.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$HERE${PYTHONPATH:+:$PYTHONPATH}"
export MEMTREE_HOME="${MEMTREE_HOME:-$HERE/var}"
mkdir -p "$MEMTREE_HOME/memtree"
exec python3 -m memtree_mcp.timer "$@"
