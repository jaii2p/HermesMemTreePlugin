"""Timer (the watchdog waking you up) tests. Run: python3 tests/test_timer.py from mcp/

Fake clock: arming, due, fire-once, clear, stop, persistence -- plus the promise
that a wake never writes to the Tree.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from memtree_mcp import Substrate, Timer, parse_every, wrap_context   # noqa: E402

LIVE = os.environ.get("MEMTREE_LIVE_TREE",
                      os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "demo_tree.json"))


def fixture_sub() -> Substrate:
    home = tempfile.mkdtemp(prefix="memtree-timer-")
    os.makedirs(os.path.join(home, "memtree"))
    if os.path.exists(LIVE):
        shutil.copy(LIVE, os.path.join(home, "memtree", "tree.json"))
    return Substrate(home=home)


def test_intervals_parse():
    assert parse_every("30s") == 30 and parse_every("10m") == 600 and parse_every("2h") == 7200
    assert parse_every(90) == 90 and parse_every("45") == 45 and parse_every(None) == 600
    assert abs(parse_every("1500ms") - 1.5) < 1e-9
    print("  parse_every: 30s/10m/2h/seconds/ms all read correctly")


def test_arm_then_not_due_then_due():
    sub = fixture_sub()
    t = Timer(sub)
    t0 = 1000.0
    st = t.arm(every="10m", now=t0)
    assert st["armed"] and st["every_s"] == 600
    assert t.tick(now=t0 + 300)["wake"] is False          # 5m in: nothing yet
    assert t.check(now=t0 + 599)["wake"] is False
    fired = t.tick(now=t0 + 600)
    assert fired["wake"] is True and "watchdog looked in on you" in fired["text"]
    print("  arm: quiet before the interval, fires exactly at it")


def test_wake_delivered_once_then_cleared():
    sub = fixture_sub()
    t = Timer(sub)
    t.arm(every=60, now=0.0)
    t.tick(now=60.0)
    first = t.check(now=61.0)
    assert first["wake"] is True and first["text"]
    second = t.check(now=62.0)
    assert second["wake"] is False, "a wake must not be delivered twice"
    assert t.status(now=62.0)["pending"] is False
    # and it re-arms itself: the next interval fires again
    t2 = Timer(sub)
    assert t2.tick(now=120.0)["wake"] is True
    print("  wake: delivered exactly once, then cleared, then the next interval comes round")


def test_wake_carries_the_memory_block_and_never_writes():
    sub = fixture_sub()
    before = sub.stats()
    t = Timer(sub, peer=sub.peer)
    t.arm(every=1, now=0.0)
    out = t.check(now=2.0)
    assert out["wake"] is True
    assert out["text"].startswith("Watchdog (a timer, checking on you):"), out["text"][:80]
    assert "anything you want to write down" in out["text"]
    assert "Fresh memory:" in out["text"], "the wake should bring the block with it"
    after = sub.stats()
    assert (after["nodes"], after["turn"]) == (before["nodes"], before["turn"]), \
        "the timer wrote to the Tree; it must only ask"
    wrapped = wrap_context(out["text"])
    assert wrapped.startswith("<memory-context>") and wrapped.endswith("</memory-context>")
    print(f"  wake: brings the block ({len(out['text'])} chars), writes nothing "
          f"(nodes {before['nodes']} -> {after['nodes']})")


def test_state_and_log_persist():
    sub = fixture_sub()
    t = Timer(sub); t.arm(every=30, seed="what did I promise", now=0.0); t.tick(now=30.0); t.check(now=31.0)
    assert os.path.exists(t.state_path) and os.path.exists(t.log_path)
    fresh = Timer(Substrate(home=str(sub.home)))
    st = fresh.status()
    assert st["armed"] and st["wakes"] == 1 and st["every_s"] == 30
    log = [json.loads(l) for l in open(t.log_path, encoding="utf-8") if l.strip()]
    assert log[0]["t"] == "wake" and "text" in log[0]
    print(f"  files: {os.path.basename(t.state_path)} + {os.path.basename(t.log_path)} survive restarts")


def test_stop_disarms():
    sub = fixture_sub()
    t = Timer(sub)
    t.arm(every=1, now=0.0)
    st = t.stop(now=1.0)
    assert st["armed"] is False and st["next_due"] is None
    assert t.tick(now=9999.0)["wake"] is False
    print("  stop: disarmed, no further wakes")


def test_cli_arm_check_status_stop():
    sub = fixture_sub()
    env = dict(os.environ, MEMTREE_HOME=str(sub.home), PYTHONPATH=ROOT)

    def cli(*args):
        p = subprocess.run([sys.executable, "-m", "memtree_mcp.timer", *args],
                           capture_output=True, text=True, env=env, cwd=ROOT)
        assert p.returncode == 0, (args, p.stderr[-400:])
        return p.stdout.strip()

    arm = cli("--arm", "--every", "1s", "--json")
    assert json.loads(arm)["armed"] is True
    st = json.loads(cli("--status", "--json"))
    assert st["armed"] and st["every_s"] == 1
    import time
    time.sleep(1.2)
    out = cli("--check")
    assert "watchdog looked in on you" in out, out[:200]
    assert json.loads(cli("--check", "--json"))["wake"] is False
    assert "off" in cli("--stop")
    print("  CLI: --arm / --status / --check / --stop behave (check delivers the wake)")


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
