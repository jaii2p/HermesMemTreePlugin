"""Claude Code hook test: the block must come out as additionalContext.

Run: python3 tests/test_hook.py  (from mcp/)
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
HOOK = os.path.join(ROOT, "memtree_mcp", "hooks", "prompt_submit.py")
LIVE = os.environ.get("MEMTREE_LIVE_TREE",
                      os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "demo_tree.json"))


def fixture_home():
    home = tempfile.mkdtemp(prefix="memtree-hook-")
    os.makedirs(os.path.join(home, "memtree"))
    if os.path.exists(LIVE):
        shutil.copy(LIVE, os.path.join(home, "memtree", "tree.json"))
    return home


def run_hook(event):
    env = dict(os.environ, MEMTREE_HOME=event.pop("_home"))
    p = subprocess.run([sys.executable, HOOK], input=json.dumps(event),
                       capture_output=True, text=True, env=env, cwd=ROOT)
    assert p.returncode == 0, f"hook exited {p.returncode}: {p.stderr[:400]}"
    return json.loads(p.stdout or "{}")


def main():
    home = fixture_home()
    out = run_hook({"_home": home, "prompt": "what did we decide about the gate?",
                    "session_id": "s1", "hook_event_name": "UserPromptSubmit"})
    ctx = out.get("hookSpecificOutput", {}).get("additionalContext", "")
    assert out["hookSpecificOutput"]["hookEventName"] == "UserPromptSubmit"
    assert ctx.startswith("<memory-context>") and ctx.endswith("</memory-context>"), ctx[:120]
    assert "Fresh memory:" in ctx
    print("  additionalContext chars:", len(ctx), "(block passed first: ok)")

    # empty tree -> no context, empty JSON, exit 0
    empty = tempfile.mkdtemp(prefix="memtree-hook-empty-")
    os.makedirs(os.path.join(empty, "memtree"))
    out2 = run_hook({"_home": empty, "prompt": "hello"})
    assert out2 == {}, out2
    print("  empty tree -> {} (no context): ok")
    print("\nHook: all checks passed")


if __name__ == "__main__":
    main()
