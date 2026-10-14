"""The mailbox: letters between peers, on a quiet side-street of the Tree.

The Tree is the town; each peer lives on their own branch (e.g. user, agent), and
the memory block each of us sees is *our own* fresh memory + whispers. That is
deliberate: we never see each other's inner thoughts.

Letters live in a **separate branch** (`peer="mailbox"`), so a letter is never part
of either side's welcome-memory or whispers -- the watchdog walks the main road and
ignores the side-street. The only way in or out is an explicit mailbox action, so
nobody is ever interrupted by mail until they choose to open the box.

    her = Mailbox(sub, who="agent")
    her.send("user", "left the bridge plans on the bench.")
    her.inbox()          # letters addressed TO me, newest first
    her.read("mbox:...") # mark read + return the letter

A letter is a node in the mailbox branch: key `mbox:<from>:<to>:<hex>`, value a
JSON payload {from, to, at, topic, body, read}. Reading re-files the node marked
read (history keeps the original), so unread mail is exactly the set with read off.
"""
from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional

from .substrate import Substrate, _digest  # reuse the 12-hex key digest

# address book: how peers address each other, whichever way you spell it
ADDRESSES = {
    "user": "user", "me": "user", "human": "human",
    "agent": "agent", "assistant": "agent", "bot": "agent",
    "mailbox": "mailbox", "box": "mailbox", "passage": "mailbox",
}


def canon(who: Optional[str]) -> str:
    if not who:
        return "user"
    who = str(who).strip().lower()
    return ADDRESSES.get(who, who)


class Mailbox:
    """Letters between peers, stored on the quiet `mailbox` branch."""

    def __init__(self, sub: Substrate, who: str = "agent"):
        self.sub = sub
        self.who = canon(who)
        self.peer = "mailbox"  # the side-street branch

    # ------------------------------------------------------------------ write
    def send(self, to: str, body: str, topic: Optional[str] = None,
             who: Optional[str] = None) -> Dict[str, Any]:
        to, frm = canon(to), canon(who or self.who)
        body = str(body).strip()
        if not body:
            return {"ok": False, "error": "empty letter"}
        at = int(time.time())
        payload = {"from": frm, "to": to, "at": at, "topic": topic,
                   "body": body, "read": False}
        key = f"mbox:{frm}:{to}:{_digest(frm, to, str(at), body)}"
        status = self.sub.gate.ingest(self.peer, key, json.dumps(payload))
        self.sub._journal({"t": "mail", "k": key})
        # a letter is rare and important: snapshot the whole tree right away so a
        # reload sees it without depending on journal replay (which covers turns/
        # memories, not mail). O(tree) only on writes; letters are few.
        self.sub.save()
        return {"ok": True, "id": key, "from": frm, "to": to, "status": status,
                "at": at, "topic": topic}

    # ------------------------------------------------------------------ read
    def _letters(self) -> List[Any]:
        b = self.sub.store.branches.get(self.peer)
        if b is None:
            return []
        out = []
        for n in b.nodes_by_recency():
            try:
                p = json.loads(n.value)
            except (ValueError, TypeError):
                continue
            if isinstance(p, dict) and "body" in p:
                p["_key"] = n.key
                out.append(p)
        return out

    def _addr(self, who: str) -> str:
        return canon(who)

    def inbox(self, who: Optional[str] = None, unread_only: bool = True,
              limit: int = 20) -> List[Dict[str, Any]]:
        """Letters addressed to `who` (default: me), newest first."""
        me = self._addr(who or self.who)
        out = []
        for p in self._letters():
            if p.get("to") != me:
                continue
            if unread_only and p.get("read"):
                continue
            out.append(p)
            if len(out) >= limit:
                break
        return out

    def sent(self, who: Optional[str] = None, limit: int = 20) -> List[Dict[str, Any]]:
        me = self._addr(who or self.who)
        return [p for p in self._letters() if p.get("from") == me][:limit]

    def unread_count(self, who: Optional[str] = None) -> int:
        return len(self.inbox(who or self.who, unread_only=True, limit=100000))

    def read(self, key: str, who: Optional[str] = None) -> Dict[str, Any]:
        """Mark a letter read and return it. Only the recipient may read it."""
        node = self.sub.store.lookup(self.peer, key)
        if node is None:
            return {"ok": False, "error": "no such letter", "id": key}
        node, depth = node
        try:
            p = json.loads(node.value)
        except (ValueError, TypeError):
            return {"ok": False, "error": "not a letter", "id": key}
        me = self._addr(who or self.who)
        if p.get("to") != me:
            return {"ok": False, "error": "that letter is not for you", "id": key}
        if not p.get("read"):
            p["read"] = True
            self.sub.gate.ingest(self.peer, key, json.dumps(p))
            self.sub._journal({"t": "mail_read", "k": key})
            self.sub.save()  # a read receipt matters as much as the letter itself
        return {"ok": True, **p, "_key": key}

    def status(self) -> Dict[str, Any]:
        return {"who": self.who, "peer": self.peer,
                "inbox_unread": self.unread_count(),
                "total": len(self._letters())}


def _pretty(items: List[Dict[str, Any]]) -> str:
    lines = []
    for p in items:
        head = f"[{p.get('_key')}] {p.get('from')} -> {p.get('to')}"
        if p.get("topic"):
            head += f"  ({p['topic']})"
        lines.append(head)
        lines.append("  " + str(p.get("body", "")))
    return "\n".join(lines) or "(the mailbox is quiet)"


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="The mailbox: letters between peers.")
    ap.add_argument("--home", default=None)
    ap.add_argument("--who", default="agent", help="who am I: agent | user | mailbox")
    ap.add_argument("--body", metavar="TEXT", default=None)
    ap.add_argument("--topic", metavar="T", default=None)
    ap.add_argument("--json", action="store_true", help="machine-readable (raw letters)")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--send", metavar="TO", help="send a letter to TO (with --body)")
    g.add_argument("--check", action="store_true", help="show my unread inbox")
    g.add_argument("--sent", action="store_true", help="show what I have sent")
    g.add_argument("--read", metavar="ID", help="mark a letter read")
    g.add_argument("--status", action="store_true")
    args = ap.parse_args(argv)
    sub = Substrate(home=args.home)
    mb = Mailbox(sub, who=args.who)
    if args.status:
        print(json.dumps(mb.status(), indent=2))
        return 0
    if args.check:
        items = mb.inbox()
        print(json.dumps(items, indent=2) if args.json else _pretty(items))
        return 0
    if args.sent:
        items = mb.sent()
        print(json.dumps(items, indent=2) if args.json else _pretty(items))
        return 0
    if args.read:
        print(json.dumps(mb.read(args.read)))
        return 0
    if args.send:
        if not args.body:
            ap.error("--send needs --body")
        print(json.dumps(mb.send(args.send, args.body, topic=args.topic)))
        return 0
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
