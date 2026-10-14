"""Substrate tests. Run: python3 tests/test_substrate.py  (from mcp/)

Uses a copy of a real Tree as the fixture, so the block/recall paths are
exercised on real nodes rather than a toy tree.
"""
import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memtree_mcp import Substrate, wrap_context  # noqa: E402

LIVE = os.environ.get("MEMTREE_LIVE_TREE",
                      os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "demo_tree.json"))


def fixture_home() -> str:
    home = tempfile.mkdtemp(prefix="memtree-mcp-")
    os.makedirs(os.path.join(home, "memtree"))
    if os.path.exists(LIVE):
        shutil.copy(LIVE, os.path.join(home, "memtree", "tree.json"))
    return home


def test_prefetch_on_real_tree():
    home = fixture_home()
    sub = Substrate(home=home)
    assert sub.stats()["nodes"] > 0, "fixture tree did not load"
    block = sub.prefetch("what did we decide about the gate?")
    assert "Fresh memory:" in block
    assert len(block) <= sub.cfg["budget_chars"] + 4, "block exceeded budget_chars"
    print("  prefetch chars:", len(block), "nodes:", sub.stats()["nodes"])


def test_recall_by_key_exact():
    home = fixture_home()
    sub = Substrate(home=home)
    keys = [n.key for n in sub.dog.context_view("user")]
    assert keys, "no center nodes"
    r = sub.recall(keys[0])
    assert r["found"] is True and r["confidence"] == 1.0 and r["matched_by"] == "key"
    assert r["memory"], "no memory text returned"
    print("  recall:", r["key"], "->", str(r["memory"])[:60].replace("\n", " "))


def test_mute_hides_scopes_then_unmutes():
    home = fixture_home()
    sub = Substrate(home=home)
    m = sub.mute("on", scopes=["whispers", "aside"])
    assert m["muted"] and "whispers" not in m["active"] and "fresh" in m["active"]
    st = sub.mute("status")
    assert st["scopes"] == ["aside", "whispers"]
    off = sub.mute("off")
    assert off["muted"] is False and sub.active_scopes() == ["fresh", "whispers", "aside", "stubs"]


def test_mute_all_returns_empty_block():
    home = fixture_home()
    sub = Substrate(home=home)
    r = sub.mute("on", scopes=["all"])
    assert r["all"] is True and r["active"] == []
    assert sub.prefetch("anything") == ""
    assert sub.mute("status")["all"] is True
    print("  master mute -> empty block: ok")


def test_mute_all_four_scopes_returns_empty_block():
    home = fixture_home()
    sub = Substrate(home=home)
    sub.mute("on", scopes=["fresh", "whispers", "aside", "stubs"])
    assert sub.prefetch("anything") == ""
    print("  all four scopes silenced -> empty block: ok")


def test_mute_expires_by_turns():
    home = fixture_home()
    sub = Substrate(home=home)
    sub.mute("on", scopes=["whispers"], turns=1)
    assert "whispers" not in sub.active_scopes()
    sub.turn += 1
    assert "whispers" in sub.active_scopes()
    print("  mute auto-expired at turn", sub.turn)


def test_remember_and_sync_persist():
    home = fixture_home()
    sub = Substrate(home=home)
    before = sub.stats()["turn"]
    r = sub.remember("the fixture port is 5824", key="fixture port")
    assert r["ok"] and r["key"] == "fixture port"
    s = sub.sync_turn("do you remember the port?", "yes, 5824.")
    assert s["ok"] and s["turn"] == before + 1
    sub.close()
    fresh = Substrate(home=home)
    assert fresh.recall("fixture port")["found"] is True
    assert fresh.turn == before + 1, "turn counter did not survive the reload"
    print("  remember+sync persisted; turn", fresh.turn)


def test_read_only_blocks_writes():
    home = fixture_home()
    sub = Substrate(home=home, read_only=True)
    assert sub.remember("nope")["ok"] is False
    assert sub.sync_turn("a", "b")["ok"] is False


def test_wrap_context_framing():
    w = wrap_context("Fresh memory:\n[k] v")
    assert w.startswith("<memory-context>") and w.endswith("</memory-context>")
    assert "NOT new user input" in w
    assert wrap_context("") == ""


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print("PASS", fn.__name__)
        except Exception as e:
            failed += 1
            print("FAIL", fn.__name__, "->", type(e).__name__, e)
    print(f"\n{len(fns) - failed} passed, {failed} failed")
    raise SystemExit(1 if failed else 0)
