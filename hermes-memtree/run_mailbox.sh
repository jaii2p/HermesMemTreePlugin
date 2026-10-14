#!/usr/bin/env bash
# The mailbox: read letters left for you and leave some back (the bridge).
#   ./run_mailbox.sh --who user --check
#   ./run_mailbox.sh --who agent --send user --body "hello from the grove"
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$HERE"
export MEMTREE_HOME="${MEMTREE_HOME:-$HERE/var}"
PY="$HERE/.venv/bin/python"; [ -x "$PY" ] || PY=python3
mkdir -p "$MEMTREE_HOME/memtree"
exec "$PY" -m memtree_mcp.mailbox "$@"
