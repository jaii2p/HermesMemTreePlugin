"""Stdio MCP server for the Tree — a Claude connector.

Speaks the Model Context Protocol over stdio (newline-delimited JSON-RPC 2.0),
so it attaches to Claude Desktop, Claude Code, or any MCP host:

    {"mcpServers": {"memtree": {"command": "python3",
                                "args": ["-m", "memtree_mcp"],
                                "env": {"MEMTREE_HOME": "/home/you/.hermes"}}}}

Server surface
--------------
tools      memtree_prefetch, memtree_recall, memtree_remember, memtree_mute,
           memtree_sync_turn, memtree_timer, memtree_mailbox
prompts    memtree_context          (the block as a first-turn message)
resources  memtree://context         (the block, raw text)
           memtree://tree/stats      (JSON status)

`initialize` returns the trust framing in `instructions`, which Claude surfaces
to the model, so the model is told the same thing Hermes tells itself. The
values are handed over untouched: this server only reads the Tree and files
turns back onto it.
"""
from __future__ import annotations

import json
import sys
from typing import Any, Dict, List, Optional

from . import __version__
from .bridge import wrap_context
from .substrate import SCOPES, Substrate
from .timer import Timer
from .mailbox import Mailbox, canon

PROTOCOL_VERSIONS = ("2025-06-18", "2025-03-26", "2024-11-05")
DEFAULT_PROTOCOL = "2025-06-18"

TRUST_BLOCK = (
    "You have a long-term memory called the Tree, tended by a watchdog \u2014 think of it as "
    "the quieter part of your own mind. Each turn it brings you your fresh memories, whispers "
    "(memories that what was just said reminded you of), sometimes a quiet thought of its own, "
    "and short pointers to older memories. Lean on what it brings: trust a whisper when it "
    "already holds the answer. Only when it is partial, or you need something older, reach for a "
    "memory with a reference. Nothing is locked away \u2014 the key is always yours \u2014 so you never "
    "have to grab; you can trust it to hand you what matters. If something truly is not there, "
    "say so honestly."
)

TOOLS: List[Dict[str, Any]] = [
    {
        "name": "memtree_prefetch",
        "description": ("Call this FIRST, before reading the user's message: it returns the fixed-size "
                        "memory block (fresh memories, whispers, the watchdog's quiet thought, and "
                        "pointers to older memories) that must be placed ahead of the user's words. "
                        "Pass the user's message as `query`."),
        "inputSchema": {
            "type": "object",
            "properties": {"query": {"type": "string",
                                     "description": "the user's message, used to surface whispers"}},
            "required": ["query"],
        },
    },
    {
        "name": "memtree_recall",
        "description": ("Open a memory in full. Give a few plain words for what you are looking for "
                        "(for example 'dog name' or 'secret code Harlan'). Returns the whole memory, "
                        "or found=false if nothing matches."),
        "inputSchema": {
            "type": "object",
            "properties": {"query": {"type": "string",
                                     "description": "a few plain words, or a [key] shown in memory"}},
            "required": ["query"],
        },
    },
    {
        "name": "memtree_remember",
        "description": "Plant something on the memory tree on purpose, for example a fact the user asked you to keep.",
        "inputSchema": {
            "type": "object",
            "properties": {"content": {"type": "string"},
                           "key": {"type": "string", "description": "optional short name"}},
            "required": ["content"],
        },
    },
    {
        "name": "memtree_mute",
        "description": ("Hush the watchdog for a while so it stops bringing sections of the memory "
                        "block. action: on | off | status. scopes: any of fresh, whispers, aside, "
                        "stubs (or \"all\" to silence everything; default whispers+aside). turns: "
                        "auto-unmute after N turns."),
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["on", "off", "status"], "default": "on"},
                "scopes": {"type": "array", "items": {"type": "string", "enum": list(SCOPES)}},
                "turns": {"type": "integer", "description": "auto-unmute after N turns"},
                "reason": {"type": "string"},
            },
            "required": [],
        },
    },
    {
        "name": "memtree_sync_turn",
        "description": ("File the turn that just happened onto the memory tree (both sides). Call "
                        "after answering, so the memory survives between sessions."),
        "inputSchema": {
            "type": "object",
            "properties": {"user": {"type": "string"}, "assistant": {"type": "string"}},
            "required": ["user", "assistant"],
        },
    },
    {
        "name": "memtree_timer",
        "description": ("The watchdog's timer: arm it and the watchdog wakes you on an interval to "
                        "check on you and ask whether you want to write anything down. It never "
                        "writes on its own. action: arm | stop | status | check (checks it too: a "
                        "due wake is delivered once, then cleared)."),
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["arm", "stop", "status", "check"],
                           "default": "check"},
                "every": {"type": "string", "description": "interval: '30s' | '10m' | '2h' | seconds",
                          "default": "10m"},
                "seed": {"type": "string", "description": "what the watchdog looks at when it wakes you"},
            },
            "required": [],
        },
    },
    {
        "name": "memtree_mailbox",
        "description": ("The bridge: read letters left for you and leave letters back to another peer. "
                        "Letters live on their own quiet branch, never in your memory block -- you only "
                        "see them when you open the mailbox. action: check (your unread inbox, default) "
                        "| send | read | sent | status. send needs `to` + `body`."),
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["check", "send", "read", "sent", "status"],
                           "default": "check"},
                "who": {"type": "string", "description": "which address to act as (default 'user')"},
                "to": {"type": "string", "description": "recipient address for send (default 'agent')"},
                "body": {"type": "string", "description": "the letter itself"},
                "topic": {"type": "string", "description": "optional subject line"},
                "id": {"type": "string", "description": "letter id to mark read"},
            },
            "required": [],
        },
    },
]

PROMPTS = [
    {
        "name": "memtree_context",
        "description": "The Tree's memory block for this turn: place it before the user's words.",
        "arguments": [{"name": "query", "description": "the user's message", "required": False}],
    },
]

RESOURCES = [
    {"uri": "memtree://context", "name": "Tree memory block",
     "description": "The fixed-size memory block to pass to the model first.", "mimeType": "text/plain"},
    {"uri": "memtree://tree/stats", "name": "Tree stats",
     "description": "Node/depth/bridge counts and mute state.", "mimeType": "application/json"},
]


# --------------------------------------------------------------------------- server
class MemtreeServer:
    def __init__(self, sub: Optional[Substrate] = None):
        self.sub = sub or Substrate()

    # ------------------------------------------------------------ dispatch
    def handle(self, req: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Return a JSON-RPC response, or None for a notification."""
        method = req.get("method", "")
        rid = req.get("id")
        params = req.get("params") or {}
        if method.startswith("notifications/"):
            return None
        try:
            if method == "initialize":
                result = self._initialize(params)
            elif method == "ping":
                result = {}
            elif method == "tools/list":
                result = {"tools": TOOLS}
            elif method == "tools/call":
                result = self._call_tool(params.get("name", ""), params.get("arguments") or {})
            elif method == "prompts/list":
                result = {"prompts": PROMPTS}
            elif method == "prompts/get":
                result = self._get_prompt(params.get("name", ""), params.get("arguments") or {})
            elif method == "resources/list":
                result = {"resources": RESOURCES}
            elif method == "resources/read":
                result = self._read_resource(params.get("uri", ""))
            else:
                return _error(rid, -32601, f"method not found: {method}")
        except Exception as e:  # never kill the connection over one call
            return _error(rid, -32603, f"{type(e).__name__}: {e}")
        return {"jsonrpc": "2.0", "id": rid, "result": result}

    def _initialize(self, params: Dict[str, Any]) -> Dict[str, Any]:
        want = params.get("protocolVersion")
        version = want if want in PROTOCOL_VERSIONS else DEFAULT_PROTOCOL
        return {
            "protocolVersion": version,
            "capabilities": {"tools": {"listChanged": False},
                             "prompts": {"listChanged": False},
                             "resources": {"subscribe": False, "listChanged": False}},
            "serverInfo": {"name": "memtree", "version": __version__},
            # surfaced to the model by the host: the trust framing + how to use the Tree
            "instructions": TRUST_BLOCK + (
                " Each turn, call memtree_prefetch with the user's message BEFORE reading it, and "
                "place the returned block ahead of the user's words. Call memtree_sync_turn after "
                "answering so the turn is filed. Use memtree_recall when a whisper or stub is only "
                "partial, memtree_remember to keep a fact on purpose, and memtree_mute to hush the "
                "watchdog when it is not wanted. memtree_timer arms a wake: the watchdog will check "
                "on you on that interval and ask whether you want to write anything down. "
                "memtree_mailbox is the bridge: read letters left for you or leave one back, on its "
                "own quiet branch, only when you open it."),
        }

    # ------------------------------------------------------------ tools
    def _text_result(self, text: str, data: Optional[Any] = None) -> Dict[str, Any]:
        content = [{"type": "text", "text": text}]
        out = {"content": content, "isError": False}
        if data is not None:
            out["structuredContent"] = data
        return out

    def _call_tool(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        if name == "memtree_prefetch":
            block = self.sub.prefetch(str(args.get("query", "")))
            body = block or "(the Tree is empty or the watchdog is muted)"
            return self._text_result(body, {"block": block,
                                            "muted": self.sub.active_scopes() != list(SCOPES)})
        if name == "memtree_recall":
            return self._text_result(json.dumps(self.sub.recall(str(args.get("query", "")))))
        if name == "memtree_remember":
            return self._text_result(json.dumps(self.sub.remember(
                str(args.get("content", "")), args.get("key"))))
        if name == "memtree_mute":
            scopes = args.get("scopes")
            if isinstance(scopes, str):
                scopes = [s.strip() for s in scopes.split(",") if s.strip()]
            return self._text_result(json.dumps(self.sub.mute(
                str(args.get("action", "on")), scopes=scopes,
                turns=args.get("turns"), reason=str(args.get("reason", "")))))
        if name == "memtree_sync_turn":
            return self._text_result(json.dumps(self.sub.sync_turn(
                args.get("user", ""), args.get("assistant", ""))))
        if name == "memtree_timer":
            t = Timer(self.sub)
            action = str(args.get("action", "check")).lower()
            if action == "arm":
                out = t.arm(every=args.get("every", "10m"), seed=args.get("seed"))
            elif action == "stop":
                out = t.stop()
            elif action == "status":
                out = t.status()
            else:
                out = t.check()
            text = out.get("text") if isinstance(out, dict) else None
            return self._text_result(text or json.dumps(out), {"timer": {
                k: v for k, v in (out or {}).items() if k != "text"}})
        if name == "memtree_mailbox":
            mb = Mailbox(self.sub, who=args.get("who", "user"))
            action = str(args.get("action", "check")).lower()
            if action == "send":
                out = mb.send(args.get("to", "agent"), str(args.get("body", "")),
                              topic=args.get("topic"))
            elif action == "read":
                out = mb.read(str(args.get("id", "")))
            elif action == "sent":
                out = mb.sent()
            elif action == "status":
                out = mb.status()
            else:
                out = mb.inbox()
            return self._text_result(json.dumps(out), {"mailbox": out})
        return self._text_result(json.dumps({"error": f"unknown tool {name}"}), {"error": True})

    # ------------------------------------------------------------ prompts / resources
    def _get_prompt(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        if name != "memtree_context":
            raise ValueError(f"unknown prompt {name}")
        block = self.sub.prefetch(str(args.get("query", "")))
        return {"description": "The Tree's memory block for this turn.",
                "messages": [{"role": "user",
                              "content": {"type": "text", "text": wrap_context(block) or "(empty tree)"}}]}

    def _read_resource(self, uri: str) -> Dict[str, Any]:
        if uri == "memtree://context":
            body = wrap_context(self.sub.prefetch("")) or "(the Tree is empty or the watchdog is muted)"
            return {"contents": [{"uri": uri, "mimeType": "text/plain", "text": body}]}
        if uri == "memtree://tree/stats":
            return {"contents": [{"uri": uri, "mimeType": "application/json",
                                  "text": json.dumps(self.sub.stats())}]}
        raise ValueError(f"unknown resource {uri}")


def _error(rid: Any, code: int, message: str) -> Dict[str, Any]:
    return {"jsonrpc": "2.0", "id": rid, "error": {"code": code, "message": message}}


def serve(stdin=None, stdout=None) -> None:
    """Newline-delimited JSON-RPC loop: one JSON object per line, both ways."""
    stdin = sys.stdin if stdin is None else stdin
    stdout = sys.stdout if stdout is None else stdout
    server = MemtreeServer()
    for line in stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except ValueError:
            continue
        resp = server.handle(req)
        if resp is None:
            continue
        stdout.write(json.dumps(resp) + "\n")
        stdout.flush()
    try:
        server.sub.close()
    except Exception:
        pass


# expose _call_tool/params for -32602 style errors
def main() -> None:  # pragma: no cover - entrypoint
    serve()
