#!/usr/bin/env bash
# Hearthwood-launcher: get the memtree MCP server going with zero fuss.
#
#   ./start.sh            # just run it (starts the http server on :8080)
#   ./start.sh setup      # only prepare: venv + numpy + a fresh tree, then stop
#   ./start.sh stdio      # run as a stdio server (for Claude Code / Desktop)
#   ./start.sh http       # run as the http/streamable endpoint (same as no arg)
#   ./start.sh status     # what's ready: python, venv, tree, tools
#
# Everything it does is safe to run more than once. Your memory tree lives in
# ./var and is never touched by setup.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

# --- pretty print helpers ----------------------------------------------------
C_RESTORE='\033[0m'; C_GREEN='\033[32m'; C_CYAN='\033[36m'; C_YELLOW='\033[33m'
[ -t 1 ] || { C_RESTORE=''; C_GREEN=''; C_CYAN=''; C_YELLOW=''; }
say()  { printf "${C_CYAN}%s${C_RESTORE}\n" "$*"; }
ok()   { printf "${C_GREEN}  ok:${C_RESTORE} %s\n" "$*"; }
warn() { printf "${C_YELLOW}  !  %s${C_RESTORE}\n" "$*"; }

banner() {
  cat <<'EOF'
   ___  ___ ___  _  _ ___ _  _ _____ ___ _   _
  |  _]| _ \ __| \ | | __| \| |_   _/ __| \ | |
  |   ||   / _||  \| | _|| .` | | || (__|  \| |
  |_|_||_|_\\___|_|\__|___|_|\_| |_| \___|_|\_|
     the memory tree, as an MCP server for Claude
EOF
}

mode="${1:-http}"
PY="${PY:-python3}"

banner
say "home: $HERE"

# --- 1. python --------------------------------------------------------------
if ! command -v "$PY" >/dev/null 2>&1; then
  warn "no python at \`$PY\` — install python3 (3.10+) or set PY=/path/to/python3"
  exit 1
fi
ver="$( "$PY" -c 'import sys;print(".".join(map(str,sys.version_info[:2])))' 2>/dev/null || echo 0 )"
if [ "$(printf "%s\n3.10\n" "$ver" | sort -V | head -1)" != "3.10" ]; then
  warn "python $ver detected — need 3.10+"
  [ "$ver" = "0" ] && exit 1
fi
ok "python $ver ($PY)"

# --- 2. venv + numpy --------------------------------------------------------
if [ ! -x "$HERE/.venv/bin/python" ]; then
  say "creating a virtualenv (.venv)…"
  "$PY" -m venv .venv
fi
VENV_PY="$HERE/.venv/bin/python"
if ! "$VENV_PY" -c 'import numpy' >/dev/null 2>&1; then
  say "installing numpy…"
  "$VENV_PY" -m pip install -q numpy
fi
ok "venv ready with numpy"

# --- 3. seed the tree -------------------------------------------------------
mkdir -p "$HERE/var/memtree"
if [ ! -f "$HERE/var/memtree/tree.json" ]; then
  say "planting a fresh memory tree (./var)…"
  PYTHONPATH="$HERE" "$VENV_PY" -c '
import json, os
from memtree_mcp.core.persist import dump
from memtree_mcp.core.store import MemoryStore
home="var"
open(f"{home}/memtree/tree.json","w").write(json.dumps(dump(MemoryStore(),{"turn":0})))
'
  ok "empty tree planted at var/memtree/tree.json"
else
  ok "tree already growing (var/memtree/tree.json)"
fi

# --- 4. run ----------------------------------------------------------------
if [ "$mode" = "setup" ]; then
  say "setup complete — run \`./start.sh\` to serve, or \`./start.sh stdio\`."
  exit 0
fi

if [ "$mode" = "stdio" ]; then
  say "starting as a stdio MCP server (Claude Code / Desktop will spawn this)…"
  exec env PYTHONPATH="$HERE" MEMTREE_HOME="$HERE/var" "$VENV_PY" -m memtree_mcp
fi

if [ "$mode" = "status" ]; then
  export PYTHONPATH="$HERE" MEMTREE_HOME="$HERE/var"
  want_tools=$("$VENV_PY" -c 'from memtree_mcp.server import TOOLS;print(len(TOOLS))' 2>/dev/null || echo "?")
  say "tools exposed: $want_tools"
  warn "set it running with:  ./start.sh"
  exit 0
fi

if [ "$mode" = "http" ] || [ "$mode" = "serve" ] || [ "$mode" = "start" ]; then
  port="${PORT:-8080}"
  host="${HOST:-0.0.0.0}"
  say "starting http/streamable server on ${host}:${port} …"
  say "  health:  curl http://127.0.0.1:${port}/healthz"
  say "  mcp url: http://<this-host>:${port}/mcp"
  say ""
  say "  connect Claude Code:   claude mcp add memtree --scope user -- $HERE/run_server.sh"
  say "  or put in front of it a TLS reverse proxy (see deploy/apache.conf) and give"
  say "  claude.ai/Claude mobile the public https://<host>/mcp as a custom connector."
  exec env PYTHONPATH="$HERE" MEMTREE_HOME="$HERE/var" MEMTREE_HOST="$host" MEMTREE_PORT="$port" \
      "$VENV_PY" -m memtree_mcp.http
fi

warn "unknown mode \`$mode\` — try: setup | http | stdio | status"
exit 2
