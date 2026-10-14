#!/usr/bin/env bash
# memtree MCP server launcher.
#
# Hosted in place: this script lives beside the package, so it pins PYTHONPATH to
# its own directory and MEMTREE_HOME to its own data dir (./var). The server then
# keeps its tree files under ./var/memtree/ and never touches a Hermes profile.
#
#   ./run_server.sh                 # stdio server (what Claude clients spawn)
#   MEMTREE_PEER=alice ./run_server.sh
#
# Files it owns (created on first write):
#   var/memtree/tree.json             the whole Tree (atomic snapshot)
#   var/memtree/tree.journal.jsonl    turns written since the last snapshot
#   var/memtree/mute.json             watchdog mute state
#   var/memtree.json                  optional per-turn settings (budget_chars, ...)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$HERE${PYTHONPATH:+:$PYTHONPATH}"
export MEMTREE_HOME="${MEMTREE_HOME:-$HERE/var}"
mkdir -p "$MEMTREE_HOME/memtree"
exec python3 -m memtree_mcp "$@"
