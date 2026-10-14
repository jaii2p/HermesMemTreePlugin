#!/usr/bin/env bash
# memtree MCP over Streamable HTTP -- the remote-connector endpoint.
#
# Stands it up on localhost (or the given host/port) with its own ./var tree.
# Put an https/TLS reverse proxy in front (Apache/nginx/Caddy, or tailscale funnel)
# and give claude.ai/mobile the public /mcp URL.
#
#   ./run_http.sh                     # -> http://0.0.0.0:8080/mcp
#   ./run_http.sh --port 8443         # non-root port, reverse-proxied
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$HERE${PYTHONPATH:+:$PYTHONPATH}"
export MEMTREE_HOME="${MEMTREE_HOME:-$HERE/var}"
export MEMTREE_HOST="${MEMTREE_HOST:-0.0.0.0}"
export MEMTREE_PORT="${MEMTREE_PORT:-8080}"
mkdir -p "$MEMTREE_HOME/memtree"
exec python3 -m memtree_mcp.http "$@"