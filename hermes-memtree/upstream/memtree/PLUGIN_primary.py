"""memtree: the Tree as a Hermes Agent memory provider.

The main model is remote and attends quadratically over whatever it is sent.
So the Tree works at the boundary, as a substrate between the conversation and
the main model:

  * every turn, the main model receives one fixed-size memory block, never a
    growing history of memories: fresh memory (the center of the tree),
    whispers (memories that what was just said stirs up, wherever they are),
    and a few stubs. Its cost per turn stays constant however large the tree
    grows: linear, sparse passage into the main model;
  * the full memory is reached with the memtree_recall tool, the key: plain
    words in, the whole memory out, found through the word index and the
    tree in roughly logarithmic time;
  * after each turn, the conversation goes through the logarithmic gate onto
    the tree (dedup, history, word index). The people's words are never
    changed; long messages are copied onto the tree in whole spans;
  * memories are faint living things: they fade when unused, strengthen when
    recalled or met again, drift toward the center or sink, and are never
    deleted. The watchdog keeps the tree clean (dedup, cleanup, sweeps);
  * the tree is saved to $HERMES_HOME/memtree/tree.json after every turn, so
    memories survive between sessions.

Install: copy this directory to $HERMES_HOME/plugins/memtree/ and set
memory.provider: memtree in config.yaml (see README.md).
"""
import hashlib
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

try:  # inside Hermes
    from agent.memory_provider import MemoryProvider, spawn_context_thread
except ImportError:  # standalone use and tests
    class MemoryProvider:  # minimal stand-in with the same hook names
        pass

    def spawn_context_thread(target, name=None):
        return threading.Thread(target=target, name=name, daemon=True)

from .core.persist import load_file, save_file
from .core.runtime import _clip, file_user_turn, render_memory, whisper_query
from .core.store import LogGate, MemoryStore
from .core.watchdog import Watchdog

logger = logging.getLogger(__name__)

ASK_HINT = "open any of these in full with the memtree_recall tool"
DEFAULTS = {
    "budget_chars": 2400,     # hard cap on the memory block sent to the main model each turn
    "entry_chars": 300,       # fresh-memory entries are clipped to this; full text is a recall away
    "whispers": 3,            # associations surfaced per turn
    "stubs": 6,               # one-line pointers to older memories
    "deliver_chars": 1500,    # cap on one recalled memory
    "tail_words": 400,        # how much of a long message counts as "the present"
    "span_words": 120,        # size of the spans long messages are copied onto the tree in
    "sweep_budget": 8,        # watchdog checks per turn
    "snapshot_every": 25,     # write the full tree every N turns; each turn only appends a tiny journal line
}


def _digest(*parts: str) -> str:
    return hashlib.sha1("\x1f".join(parts).encode("utf-8", "replace")).hexdigest()[:12]


def _text(content) -> str:
    """Hermes content can be a string or a list of OpenAI-style parts."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    return str(content)


class MemtreeProvider(MemoryProvider):
    def __init__(self):
        self._lock = threading.RLock()
        self._sync_thread = None
        self.store: Optional[MemoryStore] = None
        self.cfg = dict(DEFAULTS)

    # ------------------------------------------------------------ identity / setup
    @property
    def name(self) -> str:
        return "memtree"

    def is_available(self) -> bool:
        return True  # local, no credentials, no network

    def get_config_schema(self) -> List[Dict[str, Any]]:
        return []  # nothing required; optional settings live in $HERMES_HOME/memtree.json

    def save_config(self, values: Dict[str, Any], hermes_home: str) -> None:
        path = Path(hermes_home) / "memtree.json"
        path.write_text(json.dumps({**DEFAULTS, **(values or {})}, indent=2))

    def initialize(self, session_id: str, **kwargs) -> None:
        home = kwargs.get("hermes_home") or os.environ.get("HERMES_HOME") or str(Path.home() / ".hermes")
        self.home = Path(home)
        cfg_path = self.home / "memtree.json"
        if cfg_path.exists():
            try:
                self.cfg.update(json.loads(cfg_path.read_text()))
            except (OSError, ValueError) as e:
                logger.warning("memtree: ignoring unreadable %s: %s", cfg_path, e)
        self.dir = self.home / "memtree"
        self.path = str(self.dir / "tree.json")
        self.journal_path = str(self.dir / "tree.journal.jsonl")
        self._since_snap = 0
        self.session_id = session_id
        self.peer = str(kwargs.get("user_id") or kwargs.get("user_name") or "user")
        # cron jobs and subagents read memory but never write it
        self.write_enabled = kwargs.get("agent_context", "primary") == "primary"
        with self._lock:
            try:
                self.store, meta = load_file(self.path)
            except (OSError, ValueError, KeyError, TypeError) as e:
                # keep the unreadable file instead of overwriting it on the next save
                backup = f"{self.path}.unreadable-{int(time.time())}"
                try:
                    os.replace(self.path, backup)
                except OSError:
                    backup = "(could not move it)"
                logger.warning("memtree: could not load %s (%s); kept it as %s and started a fresh tree",
                               self.path, e, backup)
                self.store, meta = MemoryStore(), {}
            self.turn = int(meta.get("turn", 0))
            self.gate = LogGate(self.store)
            self.dog = Watchdog(self.store)
            replayed = self._replay_journal()  # recover turns written since the last snapshot
            if replayed:
                logger.info("memtree: replayed %d journaled records after the last snapshot", replayed)
                if self.write_enabled:
                    self._save_locked()  # fold them in and clear the journal

    def system_prompt_block(self) -> str:
        return ("You have a long-term memory called the Tree. Each turn you are shown your fresh "
                "memories, whispers (memories that what was just said reminded you of) and short "
                "pointers to older ones. Trust a whisper when it already holds the answer. When it is "
                "only partial, or you need something older, call memtree_recall with a few plain "
                "words for what you are looking for. If nothing is found, say so honestly.")

    # --------------------------------------------------------------- the substrate
    def prefetch(self, query: str, *, session_id: str = "") -> str:
        """The fixed-size memory block handed to the main model before each call."""
        if self.store is None:
            return ""
        with self._lock:
            whispers = self.dog.whispers(self.peer, whisper_query(_text(query)),
                                         limit=int(self.cfg["whispers"]))
            aside = self.dog.aside(self.peer, whispers)
            block = render_memory(self.dog, self.peer, stub_limit=int(self.cfg["stubs"]),
                                  entry_chars=int(self.cfg["entry_chars"]), whispers=whispers,
                                  ask_hint=ASK_HINT, aside=aside)
        if block.strip() == "Fresh memory:\n(empty)":
            return ""
        return _clip(block, int(self.cfg["budget_chars"]))

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        return None  # prefetch is local and fast; nothing to pre-warm

    def sync_turn(self, user_content, assistant_content, *, session_id: str = "", messages=None) -> None:
        """After each turn: the gate files both sides onto the tree. Non-blocking."""
        if not self.write_enabled or self.store is None:
            return
        user, assistant = _text(user_content), _text(assistant_content)

        def _sync():
            try:
                with self._lock:
                    self._record({"t": "turn", "n": self.turn + 1, "u": user, "a": assistant})
            except Exception as e:  # never break the agent over memory
                logger.warning("memtree: sync failed: %s", e)

        if self._sync_thread is not None and self._sync_thread.is_alive():
            self._sync_thread.join(timeout=5.0)
        self._sync_thread = spawn_context_thread(_sync, name="memtree-sync")
        self._sync_thread.start()

    # ------------------------------------------------------------------ the key
    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [
            {"name": "memtree_recall",
             "description": ("Open a memory in full. Give a few plain words for what you are looking "
                             "for (for example 'dog name' or 'secret code Harlan'). Returns the whole "
                             "memory, or found=false if nothing matches."),
             "parameters": {"type": "object",
                            "properties": {"query": {"type": "string",
                                                     "description": "a few plain words, or a [key] shown in memory"}},
                            "required": ["query"]}},
            {"name": "memtree_remember",
             "description": "Plant something on the memory tree on purpose, for example a fact the user asked you to keep.",
             "parameters": {"type": "object",
                            "properties": {"content": {"type": "string"},
                                           "key": {"type": "string", "description": "optional short name"}},
                            "required": ["content"]}},
        ]

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        args = args or {}
        if self.store is None:
            return json.dumps({"error": "memtree is not initialized"})
        if tool_name == "memtree_recall":
            query = str(args.get("query", "")).strip()
            if not query:
                return json.dumps({"found": False, "error": "empty query"})
            with self._lock:
                d = self.dog.reference(self.peer, query, budget_chars=int(self.cfg["deliver_chars"]),
                                       expand=True)
            if not d.found:
                return json.dumps({"found": False, "query": query})
            memory = d.value if d.value is not None else (d.stub.summary if d.stub else None)
            if memory is None:  # already in fresh memory: deliver it anyway, it was asked for
                node = self.store.lookup(self.peer, d.provenance.get("key", query))
                memory = node[0].value if node else None
            return json.dumps({"found": True, "key": d.provenance.get("key"), "memory": memory,
                               "confidence": d.confidence, "matched_by": d.provenance.get("matched_by")})
        if tool_name == "memtree_remember":
            content = str(args.get("content", "")).strip()
            if not content:
                return json.dumps({"ok": False, "error": "empty content"})
            if not self.write_enabled:
                return json.dumps({"ok": False, "error": "memory is read-only in this context"})
            key = str(args.get("key") or f"note:{_digest(content)}")
            with self._lock:
                status = self.gate.ingest(self.peer, key, content)
                self._journal({"t": "remember", "k": key, "c": content})
                self._since_snap += 1
                if self._since_snap >= int(self.cfg["snapshot_every"]):
                    self._save_locked()
            return json.dumps({"ok": True, "key": key, "status": status})
        return json.dumps({"error": f"unknown tool {tool_name}"})

    # ------------------------------------------------------------- other hooks
    def on_pre_compress(self, messages) -> str:
        """Before Hermes compresses the context, make sure every turn is on the
        tree. Idempotent: messages are keyed by content."""
        if not self.write_enabled or self.store is None:
            return ""
        n = 0
        with self._lock:
            for m in messages or []:
                role = m.get("role") if isinstance(m, dict) else None
                if role not in ("user", "assistant"):
                    continue
                text = _text(m.get("content"))
                if text.strip():
                    key = f"msg:{_digest(role, text)}"
                    if self.gate.ingest(self.peer, key, text) == "new":
                        self._journal({"t": "msg", "k": key, "c": text})
                        n += 1
            self._save_locked()  # compression is rare and a good checkpoint
        return f"memtree: the full conversation is kept on the memory tree ({n} new messages archived)."

    def on_memory_write(self, action, target, content, metadata=None) -> None:
        """Mirror Hermes's built-in memory. Removals are not mirrored:
        memories on the tree fade, they are never deleted."""
        if not self.write_enabled or self.store is None or action not in ("add", "replace"):
            return
        text = _text(content)
        if text.strip():
            key = f"builtin:{target}:{_digest(text)}"
            with self._lock:
                self.gate.ingest(self.peer, key, text)
                self._journal({"t": "builtin", "k": key, "c": text})
                self._since_snap += 1
                if self._since_snap >= int(self.cfg["snapshot_every"]):
                    self._save_locked()

    def on_session_end(self, messages) -> None:
        self._flush()

    def shutdown(self) -> None:
        self._flush()

    # ------------------------------------------------------------------ helpers
    def _flush(self):
        if self._sync_thread is not None and self._sync_thread.is_alive():
            self._sync_thread.join(timeout=10.0)
        if self.store is not None and self.write_enabled:
            with self._lock:
                self.dog.sweep(self.peer, budget=int(self.cfg["sweep_budget"]) * 4)
                self._save_locked()

    def _apply(self, rec):
        """Perform one write record on the tree. Used live AND on replay, so a
        reloaded tree is identical to the one that crashed."""
        t = rec.get("t")
        if t == "turn":
            self.turn = max(self.turn, rec["n"])
            if rec.get("u", "").strip():
                file_user_turn(self.gate, self.peer, rec["n"], rec["u"],
                               int(self.cfg["tail_words"]), int(self.cfg["span_words"]))
            if rec.get("a", "").strip():
                self.gate.ingest(self.peer, f"turn{rec['n']}:assistant", rec["a"])
            self.dog.sweep(self.peer, budget=int(self.cfg["sweep_budget"]))
        elif t in ("remember", "msg", "builtin"):
            self.gate.ingest(self.peer, rec["k"], rec["c"])

    def _journal(self, rec):
        """Append one record. O(size of this record) — never O(tree)."""
        try:
            os.makedirs(self.dir, exist_ok=True)
            with open(self.journal_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except OSError as e:
            logger.warning("memtree: could not journal: %s", e)

    def _record(self, rec):
        """Live write: apply, journal, and snapshot only on the cadence."""
        self._apply(rec)
        self._journal(rec)
        self._since_snap += 1
        if self._since_snap >= int(self.cfg["snapshot_every"]):
            self._save_locked()

    def _replay_journal(self):
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
                        continue  # skip a torn last line from a crash mid-write
        except OSError:
            return 0
        return n

    def _save_locked(self):
        """Write the full snapshot (O(tree)) and clear the journal. Amortized
        O(tree / snapshot_every) per turn; the per-turn hot path stays O(turn)."""
        save_file(self.store, self.path, {"turn": self.turn})
        try:
            if os.path.exists(self.journal_path):
                os.remove(self.journal_path)
        except OSError:
            pass
        self._since_snap = 0


def register(ctx) -> None:
    ctx.register_memory_provider(MemtreeProvider())
