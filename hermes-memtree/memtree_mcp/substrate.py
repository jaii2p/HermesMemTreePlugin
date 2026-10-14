"""The substrate: the Tree loaded from disk, plus the watchdog's deliveries.

This is the whole memtree surface a host needs, with no MCP in sight:

    sub = Substrate()
    sub.prefetch("what did we decide about the gate?")   -> the fixed-size block
    sub.recall("dog name")                               -> the whole memory
    sub.remember("the port is 5824", key="api port")     -> plant it
    sub.mute("on", scopes=["whispers", "aside"])         -> hush the watchdog
    sub.sync_turn(user, assistant)                       -> file the turn

The block is what gets passed to the model FIRST, before the user's words
(bridge.inject_first / the Claude Code UserPromptSubmit hook). It is bounded:
`budget_chars` however large the tree grows.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .core.persist import load_file, save_file
from .core.runtime import (
    WHISPER_HEADER,
    _clip,
    file_user_turn,
    whisper_query,
)
from .core.store import LogGate, MemoryStore
from .core.watchdog import Watchdog

ASK_HINT = "open any of these in full with the memtree_recall tool"

# All sections the watchdog can bring. `mute` silences a subset of them.
SCOPES = ("fresh", "whispers", "aside", "stubs")

DEFAULTS = {
    "budget_chars": 2400,     # hard cap on the memory block handed to the model
    "entry_chars": 300,       # fresh-memory entries are clipped to this
    "whispers": 3,            # associations surfaced per turn
    "stubs": 6,               # one-line pointers to older memories
    "deliver_chars": 1500,    # cap on one recalled memory
    "tail_words": 400,        # how much of a long message counts as "the present"
    "span_words": 120,        # size of the spans long messages are copied in
    "sweep_budget": 8,        # watchdog checks per turn
    "snapshot_every": 25,     # full tree write cadence; other turns append a journal line
}


def default_home() -> Path:
    """Where the Tree lives.

    Order: $MEMTREE_HOME, then $HERMES_HOME, then the server's own data dir that
    ships beside the package (`mcp/var`), then ~/.hermes. The launcher pins
    MEMTREE_HOME to `mcp/var`, so a hosted server keeps its own tree files and
    never writes into a Hermes profile.
    """
    for var in ("MEMTREE_HOME", "HERMES_HOME"):
        v = os.environ.get(var)
        if v:
            return Path(v).expanduser()
    packaged = Path(__file__).resolve().parent.parent / "var"
    if packaged.exists():
        return packaged
    return Path.home() / ".hermes"


def _digest(*parts: str) -> str:
    return hashlib.sha1("\x1f".join(parts).encode("utf-8", "replace")).hexdigest()[:12]


class Substrate:
    """One peer's Tree + watchdog + mute state. Thread-safe, no network."""

    def __init__(self, home: Optional[str] = None, peer: str = "user",
                 cfg: Optional[Dict[str, Any]] = None, read_only: bool = False):
        self.home = Path(home).expanduser() if home else default_home()
        self.peer = peer or "user"
        self.dir = self.home / "memtree"
        self.path = str(self.dir / "tree.json")
        self.journal_path = str(self.dir / "tree.journal.jsonl")
        self.mute_path = self.dir / "mute.json"
        self.cfg = dict(DEFAULTS)
        cfg_file = self.home / "memtree.json"
        if cfg_file.exists():
            try:
                self.cfg.update(json.loads(cfg_file.read_text()))
            except (OSError, ValueError):
                pass
        if cfg:
            self.cfg.update(cfg)
        self.read_only = read_only
        self._lock = threading.RLock()
        self._since_snap = 0
        self.store: MemoryStore = MemoryStore()
        self.mute_state: Dict[str, Any] = {}
        self.loaded_meta: Dict[str, Any] = {}
        self.load_error: Optional[str] = None
        self.load()

    # --------------------------------------------------------------- lifecycle
    def load(self) -> None:
        with self._lock:
            try:
                self.store, self.loaded_meta = load_file(self.path)
            except (OSError, ValueError, KeyError, TypeError) as e:
                # never clobber an unreadable tree: keep it aside and start clean
                self.load_error = str(e)
                backup = f"{self.path}.unreadable-{int(time.time())}"
                try:
                    os.replace(self.path, backup)
                except OSError:
                    pass
                self.store, self.loaded_meta = MemoryStore(), {}
            self.turn = int(self.loaded_meta.get("turn", 0))
            self.gate = LogGate(self.store)
            self.dog = Watchdog(self.store)
            self._replay_journal()
            self._load_mute()

    def _replay_journal(self) -> int:
        if not os.path.exists(self.journal_path):
            return 0
        n = 0
        try:
            with open(self.journal_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        self._apply(json.loads(line))
                        n += 1
                    except (ValueError, KeyError):
                        continue  # torn last line from a crash mid-write
        except OSError:
            return 0
        return n

    def _apply(self, rec: Dict[str, Any]) -> None:
        t = rec.get("t")
        if t == "turn":
            self.turn = max(self.turn, int(rec.get("n", 0)))
            if str(rec.get("u", "")).strip():
                file_user_turn(self.gate, self.peer, self.turn, rec["u"],
                               int(self.cfg["tail_words"]), int(self.cfg["span_words"]))
            if str(rec.get("a", "")).strip():
                self.gate.ingest(self.peer, f"turn{self.turn}:assistant", rec["a"])
            self.dog.sweep(self.peer, budget=int(self.cfg["sweep_budget"]))
        elif t in ("remember", "msg", "builtin"):
            self.gate.ingest(self.peer, rec["k"], rec["c"])

    # ------------------------------------------------------------------- mute
    def _load_mute(self) -> None:
        self.mute_state = {}
        if self.mute_path.exists():
            try:
                self.mute_state = json.loads(self.mute_path.read_text()) or {}
            except (OSError, ValueError):
                self.mute_state = {}

    def _save_mute(self) -> None:
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            tmp = f"{self.mute_path}.tmp"
            Path(tmp).write_text(json.dumps(self.mute_state, indent=2))
            os.replace(tmp, self.mute_path)
        except OSError:
            pass

    def active_scopes(self) -> List[str]:
        """Which sections the watchdog is allowed to bring this turn."""
        m = self.mute_state
        until = m.get("until_turn")
        if until is not None and self.turn >= int(until):
            return list(SCOPES)  # the mute expired by turn count
        if m.get("muted") and not m.get("scopes"):
            return []  # master mute: everything
        muted = set(m.get("scopes") or [])
        return [s for s in SCOPES if s not in muted]

    def mute(self, action: str = "on", scopes: Optional[List[str]] = None,
             turns: Optional[int] = None, reason: str = "") -> Dict[str, Any]:
        """Hush the watchdog. action: on | off | status.

        scopes: any of fresh, whispers, aside, stubs, or "all" (default:
        whispers+aside). turns: auto-unmute after N turns of sync_turn.
        """
        with self._lock:
            a = (action or "on").lower()
            if a == "status":
                silenced = set(self.mute_state.get("scopes") or [])
                master = bool(self.mute_state.get("muted") and not silenced)
                return {"muted": bool(master or silenced), "all": master,
                        "scopes": sorted(silenced),
                        "until_turn": self.mute_state.get("until_turn"),
                        "reason": self.mute_state.get("reason", ""),
                        "active": self.active_scopes()}
            if a in ("off", "unmute", "none"):
                self.mute_state = {}
                self._save_mute()
                return {"ok": True, "muted": False, "active": self.active_scopes()}
            if a not in ("on", "mute", "hide"):
                return {"ok": False, "error": f"unknown action {action!r}"}
            requested = scopes or ["whispers", "aside"]
            if isinstance(requested, str):
                requested = [s.strip() for s in requested.split(",") if s.strip()]
            master = any(s in ("all", "everything", "*") for s in requested)
            chosen = [s for s in requested if s in SCOPES]
            unknown = [s for s in requested if s not in SCOPES and s not in ("all", "everything", "*")]
            if not chosen and not master and not unknown:
                chosen = ["whispers", "aside"]
            until = self.turn + int(turns) if turns else None
            pre_existing = set(self.mute_state.get("scopes") or [])
            self.mute_state = {
                "muted": bool(master),
                "scopes": [] if master else sorted(pre_existing | set(chosen)),
                "until_turn": until,
                "reason": reason,
            }
            self._save_mute()
            return {"ok": True, "muted": True, "all": master,
                    "scopes": self.mute_state["scopes"],
                    "until_turn": until, "unknown": unknown, "active": self.active_scopes()}

    # --------------------------------------------------------------- the block
    def _render(self, whispers, aside, scopes) -> str:
        dog, peer = self.dog, self.peer
        entry_chars = int(self.cfg["entry_chars"])
        lines: List[str] = []
        if "fresh" in scopes:
            fresh = [f"[{n.key}] {_clip(n.value, entry_chars)}" for n in dog.context_view(peer)]
            lines.append("Fresh memory:")
            lines += fresh or ["(empty)"]
        if whispers and "whispers" in scopes:
            lines.append(WHISPER_HEADER)
            lines += [f"[{w.key}] {w.summary}" for w in whispers]
        if aside and "aside" in scopes:
            lines.append(f"Watchdog (a quiet thought): {aside}")
        stubs = dog.boundary_stubs(peer) + dog.proactive(peer)
        seen, uniq = {w.key for w in whispers}, []
        for s in stubs:
            if s.key not in seen:
                seen.add(s.key)
                uniq.append(s)
        if uniq and "stubs" in scopes:
            lines.append(f"Also remembered ({ASK_HINT}):")
            lines += [f"[{s.key}] {s.summary}" for s in uniq[: int(self.cfg["stubs"])]]
        return "\n".join(lines)

    def prefetch(self, query: str = "") -> str:
        """The fixed-size memory block, mute-aware and bounded to budget_chars.

        This is what gets passed to the model FIRST, ahead of the user turn.
        """
        with self._lock:
            scopes = self.active_scopes()
            if not scopes:
                return ""
            whispers = []
            if "whispers" in scopes:
                whispers = self.dog.whispers(self.peer, whisper_query(query),
                                             limit=int(self.cfg["whispers"]))
            aside = self.dog.aside(self.peer, whispers) if "aside" in scopes else None
            if "whispers" not in scopes:
                aside = self.dog.aside(self.peer, []) if "aside" in scopes else None
            block = self._render(whispers, aside, scopes)
        if block.strip() in ("", "Fresh memory:\n(empty)"):
            return ""
        return _clip(block, int(self.cfg["budget_chars"]))

    # --------------------------------------------------------------- the key
    def recall(self, query: str) -> Dict[str, Any]:
        """Manual recollection: a few plain words in, the whole memory out."""
        query = str(query or "").strip()
        if not query:
            return {"found": False, "error": "empty query"}
        with self._lock:
            d = self.dog.reference(self.peer, query,
                                   budget_chars=int(self.cfg["deliver_chars"]), expand=True)
            if not d.found:
                return {"found": False, "query": query}
            memory = d.value if d.value is not None else (d.stub.summary if d.stub else None)
            if memory is None:  # already in fresh memory: deliver it anyway, it was asked for
                node = self.store.lookup(self.peer, d.provenance.get("key", query))
                memory = node[0].value if node else None
            return {"found": True, "key": d.provenance.get("key"), "memory": memory,
                    "confidence": d.confidence, "matched_by": d.provenance.get("matched_by")}

    # --------------------------------------------------------------- the write
    def remember(self, content: str, key: Optional[str] = None) -> Dict[str, Any]:
        content = str(content or "").strip()
        if not content:
            return {"ok": False, "error": "empty content"}
        if self.read_only:
            return {"ok": False, "error": "memory is read-only in this context"}
        k = str(key or f"note:{_digest(content)}")
        with self._lock:
            status = self.gate.ingest(self.peer, k, content)
            self._record({"t": "remember", "k": k, "c": content})
        return {"ok": True, "key": k, "status": status}

    def sync_turn(self, user_content: Any, assistant_content: Any) -> Dict[str, Any]:
        """File both sides of the turn onto the tree."""
        if self.read_only:
            return {"ok": False, "error": "read-only"}
        user, assistant = _text(user_content), _text(assistant_content)
        with self._lock:
            n = self.turn + 1
            self._record({"t": "turn", "n": n, "u": user, "a": assistant})
        return {"ok": True, "turn": self.turn, "filed": bool(user or assistant)}

    # --------------------------------------------------------------- internals
    def _record(self, rec: Dict[str, Any]) -> None:
        self._apply(rec)
        self._journal(rec)
        self._since_snap += 1
        if self._since_snap >= int(self.cfg["snapshot_every"]):
            self.save()

    def _journal(self, rec: Dict[str, Any]) -> None:
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            with open(self.journal_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except OSError:
            pass

    def save(self) -> None:
        if self.read_only:
            return
        save_file(self.store, self.path, {"turn": self.turn})
        try:
            if os.path.exists(self.journal_path):
                os.remove(self.journal_path)
        except OSError:
            pass
        self._since_snap = 0

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            b = self.store.branches.get(self.peer)
            nodes = len(b) if b else 0
            depths = len(b.levels) if b else 0
            return {"peer": self.peer, "turn": self.turn, "nodes": nodes,
                    "depths": depths, "bridges": self.store.bridges.n_edges(),
                    "home": str(self.home), "tree": self.path,
                    "read_only": self.read_only, "mute": self.active_scopes(),
                    "load_error": self.load_error}

    def close(self) -> None:
        with self._lock:
            self.dog.sweep(self.peer, budget=int(self.cfg["sweep_budget"]) * 4)
            self.save()


def _text(content: Any) -> str:
    """Content can be a string or a list of OpenAI-style parts."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    return str(content)
