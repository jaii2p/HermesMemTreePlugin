#!/usr/bin/env python3
"""Claude Code UserPromptSubmit hook: pass the memory block to the model FIRST.

Claude Code runs this before the model sees the prompt. It reads the hook event
on stdin and writes a JSON object whose `additionalContext` is placed at the head
of the turn -- i.e. the Tree's block arrives ahead of the user's words, exactly
like the Hermes memory-context block.

Wire it up in .claude/settings.json (see settings.snippet.json):

    {
      "hooks": {
        "UserPromptSubmit": [
          {"hooks": [{"type": "command",
                      "command": "python3 /path/to/memtree_mcp/hooks/prompt_submit.py"}]}
        ]
      }
    }

Env: MEMTREE_HOME (defaults to $HERMES_HOME, then ~/.hermes). MEMTREE_PEER
selects the branch. Never writes to the tree.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from memtree_mcp import Substrate, Timer, wrap_context  # noqa: E402


def main() -> int:
    raw = sys.stdin.read()
    try:
        event = json.loads(raw) if raw.strip() else {}
    except ValueError:
        event = {}
    prompt = str(event.get("prompt") or event.get("user_prompt") or "")
    home = event.get("memtree_home") or os.environ.get("MEMTREE_HOME") or os.environ.get("HERMES_HOME")
    peer = os.environ.get("MEMTREE_PEER", "user")
    wake = ""
    block = ""
    try:
        sub = Substrate(home=home, peer=peer)
        block = sub.prefetch(prompt)
        # a due timer wake rides in with the block: the watchdog checks on you first
        out = Timer(sub, peer=peer).check()
        if out.get("wake"):
            wake = out.get("text", "")
    except Exception:
        pass  # memory must never break the turn
    combined = "\n\n".join(part for part in (wake, block) if part)
    wrapped = wrap_context(combined)
    if not wrapped:
        print(json.dumps({}))
        return 0
    print(json.dumps({"hookSpecificOutput": {
        "hookEventName": "UserPromptSubmit",
        "additionalContext": wrapped,
    }}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
