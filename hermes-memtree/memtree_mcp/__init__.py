"""memtree-mcp: the Tree memory substrate, served to Claude over MCP.

The Tree is a living memory: nodes are keys, whispers are edges, and every turn
the watchdog hands the model one fixed-size memory block. This package serves
that substrate two ways:

  * as an MCP server (`memtree_mcp.server`) a Claude connector can attach to;
  * as a first-turn context injector (`memtree_mcp.bridge` + `hooks/`), so the
    memory block is handed to the model BEFORE it answers, exactly like the
    Hermes memory-context block.

Layout mirrors the reverse-engineered Hermes plugin:
    memtree_mcp/core/   vendored forest (store, watchdog, persist, runtime)
    substrate.py        load / prefetch / recall / remember / mute / sync
    server.py           stdio MCP server (tools, prompts, resources)
    bridge.py           wrap + inject the block as leading context
"""
from .substrate import DEFAULTS, Substrate, default_home
from .bridge import wrap_context
from .timer import Timer, parse_every

__all__ = ["Substrate", "DEFAULTS", "default_home", "wrap_context", "Timer", "parse_every"]
__version__ = "0.1.0"
