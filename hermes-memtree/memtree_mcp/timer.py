"""The timer: the watchdog waking the model up to check on it.

The watchdog does not only answer — it can notice that time has passed and come
and ask. Arm a timer and, every interval, the watchdog builds a wake block (the
same fixed-size memory block, plus one line of its own) and leaves it pending
until the model looks. It never writes anything down itself: the whole point is
to wake the model and ASK whether it wants to.

    t = Timer(sub)                 # sub: a Substrate (or the server's)
    t.arm(every="10m")             # or every=600
    t.tick()                       # due? -> leaves a wake pending
    t.check()                      # -> {"wake": True, "block": ..., "ask": ...}

Files (under <home>/memtree/):
    timer.json          arm state: interval, last wake, next due, pending flag
    timer.log.jsonl     every wake, appended (one line per wake)

The CLI runs the same thing in the foreground so a shell, a service, or a cron
entry can drive it:

    python3 -m memtree_mcp.timer --every 10m --run      # loop, print wakes
    python3 -m memtree_mcp.timer --check                # is a wake waiting?
    python3 -m memtree_mcp.timer --status
    python3 -m memtree_mcp.timer --stop
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

from .substrate import Substrate

DEFAULT_EVERY = 600.0  # 10 minutes
DEFAULT_SEED = "what is on your mind right now"
ASK = "the watchdog looked in on you. is there anything you want to write down?"
WAKE_HEADER = "Watchdog (a timer, checking on you):"
DEFAULT_STATE = "timer.json"
DEFAULT_LOG = "timer.log.jsonl"


def parse_every(value) -> float:
    """'30s' | '10m' | '2h' | 600 -> seconds."""
    if value is None:
        return DEFAULT_EVERY
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip().lower()
    try:
        if s.endswith("ms"):
            return float(s[:-2]) / 1000.0
        if s.endswith("s"):
            return float(s[:-1])
        if s.endswith("m"):
            return float(s[:-1]) * 60.0
        if s.endswith("h"):
            return float(s[:-1]) * 3600.0
        return float(s)
    except ValueError as e:
        raise ValueError(f"cannot read an interval from {value!r}") from e


def human(seconds: float) -> str:
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m{seconds % 60:02d}s"
    return f"{seconds // 3600}h{(seconds % 3600) // 60:02d}m"


class Timer:
    """Arm a wake interval for one peer, on one Substrate."""

    def __init__(self, sub: Substrate, peer: Optional[str] = None,
                 state_file: str = DEFAULT_STATE, log_file: str = DEFAULT_LOG):
        self.sub = sub if isinstance(sub, Substrate) else sub.sub
        self.peer = peer or getattr(sub, "peer", "user")
        self.state_path = Path(self.sub.dir) / state_file
        self.log_path = Path(self.sub.dir) / log_file
        self.state: Dict[str, Any] = self._load()

    # ------------------------------------------------------------------ state
    def _load(self) -> Dict[str, Any]:
        if self.state_path.exists():
            try:
                st = json.loads(self.state_path.read_text()) or {}
                if st.get("peer") == self.peer:
                    return st
            except (OSError, ValueError):
                pass
        return {"peer": self.peer, "armed": False, "every": DEFAULT_EVERY,
                "last": None, "next_due": None, "pending": False, "pending_at": None,
                "seed": DEFAULT_SEED, "wakes": 0}

    def _save(self) -> None:
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = f"{self.state_path}.tmp"
            Path(tmp).write_text(json.dumps(self.state, indent=2))
            os.replace(tmp, self.state_path)
        except OSError:
            pass

    def _log(self, rec: Dict[str, Any]) -> None:
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except OSError:
            pass

    # ------------------------------------------------------------------ arm/stop
    def arm(self, every=DEFAULT_EVERY, seed: Optional[str] = None,
            now: Optional[float] = None) -> Dict[str, Any]:
        now = time.time() if now is None else now
        self.state.update({"armed": True, "every": parse_every(every),
                           "seed": seed or self.state.get("seed") or DEFAULT_SEED,
                           "next_due": now + parse_every(every)})
        self._save()
        return self.status(now=now)

    def stop(self, now: Optional[float] = None) -> Dict[str, Any]:
        self.state.update({"armed": False, "next_due": None, "pending": False, "pending_at": None})
        self._save()
        return self.status(now=now)

    def status(self, now: Optional[float] = None) -> Dict[str, Any]:
        now = time.time() if now is None else now
        due = self.state.get("next_due")
        return {"armed": bool(self.state.get("armed")), "every_s": self.state.get("every"),
                "every": human(self.state.get("every") or DEFAULT_EVERY),
                "pending": bool(self.state.get("pending")),
                "pending_at": self.state.get("pending_at"),
                "last_wake": self.state.get("last"),
                "next_due": due,
                "in_s": None if not due else max(0.0, float(due) - now),
                "wakes": self.state.get("wakes", 0),
                "peer": self.peer,
                "state_file": str(self.state_path),
                "log_file": str(self.log_path)}

    # ------------------------------------------------------------------ firing
    def _wake_text(self, extra_whispers: int = 0) -> str:
        seed = str(self.state.get("seed") or DEFAULT_SEED)
        block = self.sub.prefetch(seed)
        lines = [WAKE_HEADER, ASK]
        if block:
            lines += ["", block]
        return "\n".join(lines)

    def tick(self, now: Optional[float] = None) -> Dict[str, Any]:
        """If the interval has elapsed, leave a wake pending. Nothing is written
        to the Tree: the watchdog only brings the block and asks."""
        now = time.time() if now is None else now
        if not self.state.get("armed"):
            return {"wake": False, "armed": False}
        due = self.state.get("next_due") or now
        if now < float(due):
            return {"wake": False, "next_due": due, "in_s": float(due) - now}
        if self.state.get("pending"):
            return {"wake": False, "already_pending": True, "pending_at": self.state.get("pending_at")}
        text = self._wake_text()
        self.state.update({"last": now, "next_due": now + float(self.state["every"]),
                           "pending": True, "pending_at": now,
                           "wakes": int(self.state.get("wakes", 0)) + 1})
        self._save()
        self._log({"t": "wake", "at": now, "peer": self.peer,
                   "every": self.state["every"], "text": text})
        return {"wake": True, "at": now, "text": text,
                "next_due": self.state["next_due"]}

    def check(self, now: Optional[float] = None) -> Dict[str, Any]:
        """Deliver a pending wake once, then clear it (so a wake is never missed
        and never repeated)."""
        now = time.time() if now is None else now
        self.tick(now)
        if not self.state.get("pending"):
            return {"wake": False, **{k: self.state.get(k) for k in ("next_due", "last", "armed")}}
        text = self._wake_text()
        self.state.update({"pending": False, "pending_at": None})
        self._save()
        self._log({"t": "check", "at": now, "peer": self.peer})
        return {"wake": True, "text": text, "ask": ASK,
                "at": self.state.get("last"), "next_due": self.state.get("next_due")}


# ------------------------------------------------------------------------ CLI
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="The Tree's timer: the watchdog waking you up.")
    ap.add_argument("--home", default=None, help="MEMTREE_HOME (default: $MEMTREE_HOME/var)")
    ap.add_argument("--peer", default="user")
    ap.add_argument("--every", default=DEFAULT_EVERY, help="30s | 10m | 2h | seconds")
    ap.add_argument("--seed", default=None, help="what the watchdog looks at when it wakes you")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--arm", action="store_true", help="arm (or re-arm) the timer")
    g.add_argument("--stop", action="store_true")
    g.add_argument("--status", action="store_true")
    g.add_argument("--check", action="store_true", help="is a wake waiting? deliver it")
    g.add_argument("--run", action="store_true", help="stay up and fire the timer")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    args = ap.parse_args(argv)

    sub = Substrate(home=args.home, peer=args.peer)
    t = Timer(sub, peer=args.peer)

    def show(obj):
        print(json.dumps(obj, indent=2) if args.json else _pretty(obj))
        return 0

    if args.status:
        return show(t.status())
    if args.stop:
        return show(t.stop())
    if args.check:
        out = t.check()
        if out.get("wake") and not args.json:
            print(out["text"])
            return 0
        return show(out)
    if args.run:
        every = parse_every(args.every)
        t.arm(every=every, seed=args.seed)
        print(f"timer armed every {human(every)} (Ctrl-C to stop)", flush=True)
        try:
            while True:
                out = t.tick()
                if out.get("wake"):
                    print("\n" + out["text"], flush=True)
                time.sleep(min(5.0, max(1.0, every / 20.0)))
        except KeyboardInterrupt:
            t.stop()
            print("\ntimer stopped", flush=True)
            return 0
    # default action: arm it
    return show(t.arm(every=args.every, seed=args.seed))


def _pretty(obj: Dict[str, Any]) -> str:
    if "armed" in obj:
        when = "—" if obj.get("next_due") is None else f"in {human(obj['in_s'])}"
        return (f"timer: {'armed' if obj['armed'] else 'off'} every {obj.get('every')} "
                f"(next {when}, wakes {obj.get('wakes', 0)}, pending {obj.get('pending')})")
    if obj.get("wake"):
        return obj.get("text", "")
    return json.dumps(obj)


if __name__ == "__main__":
    raise SystemExit(main())
