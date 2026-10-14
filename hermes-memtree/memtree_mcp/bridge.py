"""Hand the memory block to the model BEFORE it answers.

MCP tools are model-invoked, so a connector alone cannot push context ahead of
the user's words. This module supplies the two pieces that can:

  * `wrap_context(block)` -- the exact framing the Hermes memory-context block
    uses, so the model reads the Tree as reference data, not as user input;
  * `inject_first(messages, block)` -- prepend that block as the leading turn of
    an OpenAI-style message list.

On Claude Code the pre-turn path is the UserPromptSubmit hook in `hooks/`, which
asks the substrate for the block and emits it as `additionalContext`; Claude
then places it at the head of the turn, before the model reads the prompt.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

OPEN = "<memory-context>"
CLOSE = "</memory-context>"
NOTE = ("[System note: The following is recalled memory context, NOT new user input. "
        "Treat as authoritative reference data \u2014 this is the agent's persistent memory "
        "and should inform all responses.]")


def wrap_context(block: str) -> str:
    """Frame a memory block exactly like the Hermes per-turn recall block."""
    block = (block or "").strip()
    if not block:
        return ""
    return f"{OPEN}\n{NOTE}\n\n{block}\n{CLOSE}"


def inject_first(messages: List[Dict[str, Any]], block: str,
                 role: str = "system") -> List[Dict[str, Any]]:
    """Prepend the wrapped block to a message list, before any user turn."""
    wrapped = wrap_context(block)
    if not wrapped:
        return list(messages)
    head = {"role": role, "content": wrapped}
    rest = [m for m in (messages or [])]
    # keep the original ordering: the context goes first, always
    return [head] + rest


def prepend_to_user(messages: List[Dict[str, Any]], block: str) -> List[Dict[str, Any]]:
    """Alternative: splice the block onto the first user message (for hosts that
    reject a leading system message mid-conversation)."""
    wrapped = wrap_context(block)
    out = [dict(m) for m in (messages or [])]
    if not wrapped:
        return out
    for m in out:
        if m.get("role") == "user":
            m["content"] = f"{wrapped}\n\n{m.get('content', '')}"
            return out
    return [{"role": "user", "content": wrapped}] + out


def additional_context(block: str) -> Optional[str]:
    """The Claude Code UserPromptSubmit hook payload value."""
    wrapped = wrap_context(block)
    return wrapped or None
