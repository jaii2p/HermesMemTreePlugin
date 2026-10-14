"""End-to-end MCP test: spawn the real server and speak JSON-RPC to it.

Run: python3 tests/test_server.py  (from mcp/)
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LIVE = os.environ.get("MEMTREE_LIVE_TREE",
                      os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "demo_tree.json"))


class Server:
    def __init__(self, home):
        env = dict(os.environ, MEMTREE_HOME=home, PYTHONPATH=ROOT)
        self.p = subprocess.Popen([sys.executable, "-m", "memtree_mcp"],
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE, text=True, env=env, cwd=ROOT)
        self._id = 0

    def call(self, method, params=None):
        self._id += 1
        req = {"jsonrpc": "2.0", "id": self._id, "method": method, "params": params or {}}
        self.p.stdin.write(json.dumps(req) + "\n")
        self.p.stdin.flush()
        line = self.p.stdout.readline()
        assert line, f"no response for {method}; stderr={self.p.stderr.read()[:400]}"
        return json.loads(line)

    def notify(self, method, params=None):
        self.p.stdin.write(json.dumps({"jsonrpc": "2.0", "method": method,
                                       "params": params or {}}) + "\n")
        self.p.stdin.flush()

    def close(self):
        try:
            self.p.stdin.close()
        except Exception:
            pass
        self.p.wait(timeout=10)


def fixture_home():
    home = tempfile.mkdtemp(prefix="memtree-mcp-e2e-")
    os.makedirs(os.path.join(home, "memtree"))
    if os.path.exists(LIVE):
        shutil.copy(LIVE, os.path.join(home, "memtree", "tree.json"))
    return home


def main():
    home = fixture_home()
    s = Server(home)
    try:
        r = s.call("initialize", {"protocolVersion": "2025-06-18",
                                  "capabilities": {}, "clientInfo": {"name": "t", "version": "0"}})
        assert r["result"]["serverInfo"]["name"] == "memtree", r
        assert "long-term memory called the Tree" in r["result"]["instructions"]
        assert r["result"]["protocolVersion"] in ("2025-06-18", "2025-03-26", "2024-11-05")
        s.notify("notifications/initialized")

        r = s.call("tools/list")
        names = [t["name"] for t in r["result"]["tools"]]
        assert names == ["memtree_prefetch", "memtree_recall", "memtree_remember",
                         "memtree_mute", "memtree_sync_turn", "memtree_timer",
                         "memtree_mailbox"], names
        print("  tools:", names)

        r = s.call("tools/call", {"name": "memtree_prefetch", "arguments": {"query": "gate"}})
        block = r["result"]["content"][0]["text"]
        assert "Fresh memory:" in block, block[:200]
        print("  prefetch block chars:", len(block))

        r = s.call("tools/call", {"name": "memtree_recall", "arguments": {"query": "dog name"}})
        rec = json.loads(r["result"]["content"][0]["text"])
        assert rec["found"] is True, rec
        print("  recall 'dog name' ->", rec["key"], "|", str(rec["memory"])[:40])

        r = s.call("tools/call", {"name": "memtree_mute",
                                  "arguments": {"action": "on", "scopes": ["whispers", "aside"]}})
        assert json.loads(r["result"]["content"][0]["text"])["muted"] is True
        r = s.call("tools/call", {"name": "memtree_prefetch", "arguments": {"query": "gate"}})
        muted_block = r["result"]["content"][0]["text"]
        assert "This reminds me of" not in muted_block, "mute did not silence whispers"
        print("  mute silenced whispers; block chars:", len(muted_block))
        r = s.call("tools/call", {"name": "memtree_mute", "arguments": {"action": "off"}})
        assert json.loads(r["result"]["content"][0]["text"])["muted"] is False

        r = s.call("tools/call", {"name": "memtree_remember",
                                  "arguments": {"content": "e2e marker", "key": "e2e:marker"}})
        assert json.loads(r["result"]["content"][0]["text"])["ok"] is True
        r = s.call("tools/call", {"name": "memtree_sync_turn",
                                  "arguments": {"user": "hi", "assistant": "hello"}})
        assert json.loads(r["result"]["content"][0]["text"])["ok"] is True

        r = s.call("prompts/list")
        assert r["result"]["prompts"][0]["name"] == "memtree_context"
        r = s.call("prompts/get", {"name": "memtree_context", "arguments": {"query": "gate"}})
        msg = r["result"]["messages"][0]["content"]["text"]
        assert msg.startswith("<memory-context>"), msg[:80]
        print("  prompt memtree_context wraps block: ok")

        r = s.call("resources/list")
        uris = [x["uri"] for x in r["result"]["resources"]]
        assert "memtree://context" in uris and "memtree://tree/stats" in uris
        r = s.call("resources/read", {"uri": "memtree://tree/stats"})
        st = json.loads(r["result"]["contents"][0]["text"])
        assert st["nodes"] > 0
        print("  stats:", {k: st[k] for k in ("nodes", "depths", "bridges", "turn")})

        r = s.call("ping")
        assert r["result"] == {}

        # the timer tool: arm, wait out a one-second interval, get woken
        import time as _t
        r = s.call("tools/call", {"name": "memtree_timer",
                                  "arguments": {"action": "arm", "every": "1s"}})
        assert json.loads(r["result"]["content"][0]["text"])["armed"] is True
        _t.sleep(1.2)
        r = s.call("tools/call", {"name": "memtree_timer", "arguments": {"action": "check"}})
        wake = r["result"]["content"][0]["text"]
        assert "watchdog looked in on you" in wake, wake[:200]
        r = s.call("tools/call", {"name": "memtree_timer", "arguments": {"action": "status"}})
        assert json.loads(r["result"]["content"][0]["text"])["armed"] is True
        r = s.call("tools/call", {"name": "memtree_timer", "arguments": {"action": "stop"}})
        assert json.loads(r["result"]["content"][0]["text"])["armed"] is False
        print("  timer tool: arm -> wake -> status -> stop")
        print("\nE2E MCP: all checks passed")
    finally:
        s.close()


if __name__ == "__main__":
    main()
