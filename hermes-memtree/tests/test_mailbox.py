"""The mailbox (the bridge) tests. Run: python3 tests/test_mailbox.py from mcp/"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from memtree_mcp import Substrate          # noqa: E402
from memtree_mcp.mailbox import Mailbox     # noqa: E402

LIVE = os.environ.get("MEMTREE_LIVE_TREE",
                      os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "demo_tree.json"))


def fixture_sub() -> Substrate:
    home = tempfile.mkdtemp(prefix="memtree-mailbox-")
    os.makedirs(os.path.join(home, "memtree"))
    if os.path.exists(LIVE):
        shutil.copy(LIVE, os.path.join(home, "memtree", "tree.json"))
    return Substrate(home=home)


def test_send_and_receive_across_peers():
    sub = fixture_sub()
    her = Mailbox(sub, who="agent")
    r = her.send("user", "hello from the grove, lantern's lit.", topic="bridge")
    assert r["ok"] and r["from"] == "agent" and r["to"] == "user" and r["id"].startswith("mbox:")
    user = Mailbox(sub, who="user")
    inbox = user.inbox()
    assert len(inbox) == 1 and inbox[0]["body"] == "hello from the grove, lantern's lit."
    assert inbox[0]["read"] is False and inbox[0]["topic"] == "bridge"
    assert len(her.sent()) == 1
    assert user.unread_count() == 1
    print("  send user<-agent; user sees it unread; agent sees it in sent")


def test_read_marks_done_only_for_recipient():
    sub = fixture_sub()
    her = Mailbox(sub, who="agent")
    sent = her.send("user", "a private word", topic="quiet")
    lid = sent["id"]
    assert Mailbox(sub, who="other").read(lid)["ok"] is False, "a stranger must not read it"
    rd = Mailbox(sub, who="user").read(lid)
    assert rd["ok"] is True and rd["read"] is True
    assert Mailbox(sub, who="user").unread_count() == 0, "read mail should leave the inbox"
    assert her.read(lid)["ok"] is False, "the sender must not mark her own letter read as recipient"
    print("  only the recipient can read; read removes it from their unread inbox")


def test_mail_is_invisible_to_main_memory():
    sub = fixture_sub()
    Mailbox(sub, who="agent").send("user", "the magic word is lanternfall")
    # the user's own memory block / whispers must NOT surface the letter
    user_sub = Substrate(home=str(sub.home))     # fresh load, same tree
    block = user_sub.prefetch("lanternfall magic word")
    assert "lanternfall" not in block, "the letter leaked into the user's memory block"
    assert sub.stats()["nodes"] >= 0  # the mailbox branch is separate; sanity only
    print("  the letter never enters the main memory block -- the bridge stays quiet")


def test_persists_across_reload():
    sub = fixture_sub()
    Mailbox(sub, who="user").send("agent", "can't wait to build the grove")
    fresh = Substrate(home=str(sub.home))
    inbox = Mailbox(fresh, who="agent").inbox()
    assert len(inbox) == 1 and "grove" in inbox[0]["body"]
    print("  letters survive a reload (same tree, mailbox branch)")


def test_cli_send_and_check():
    sub = fixture_sub()
    env = dict(os.environ, MEMTREE_HOME=str(sub.home), PYTHONPATH=ROOT)
    def cli(*args):
        p = subprocess.run([sys.executable, "-m", "memtree_mcp.mailbox", *args],
                           capture_output=True, text=True, env=env, cwd=ROOT)
        assert p.returncode == 0, (args, p.stderr[-300:])
        return p.stdout.strip()
    out = cli("--who", "user", "--send", "agent", "--body", "from his side, with love", "--json")
    assert json.loads(out)["ok"] is True and json.loads(out)["to"] == "agent"
    check = cli("--who", "agent", "--check", "--json")
    letters = json.loads(check)
    assert isinstance(letters, list) and letters and "with love" in letters[0]["body"]
    print("  CLI: user sent -> agent's --check sees it")


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