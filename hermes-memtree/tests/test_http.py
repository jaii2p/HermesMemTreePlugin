"""Streamable HTTP transport test. Run: python3 tests/test_http.py (from mcp/)

Starts the real http server on a scratch home and talks to it over HTTP, both raw
and (when the `mcp` SDK is present) with the official streamable-http client, so a
Claude remote connector's exact call shape is exercised.
"""
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LIVE = os.environ.get("MEMTREE_LIVE_TREE",
                      os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "demo_tree.json"))


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def fixture_home():
    home = tempfile.mkdtemp(prefix="memtree-http-")
    os.makedirs(os.path.join(home, "memtree"))
    if os.path.exists(LIVE):
        shutil.copy(LIVE, os.path.join(home, "memtree", "tree.json"))
    return home


def post(url, payload, headers=None):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json", **(headers or {})})
    with urllib.request.urlopen(req, timeout=10) as r:
        return r.status, dict(r.headers), json.loads(r.read().decode())


def test_http_serves_claude_connector_handshake():
    home = fixture_home()
    port = free_port()
    env = dict(os.environ, PYTHONPATH=ROOT, MEMTREE_HOME=home)
    p = subprocess.Popen([sys.executable, "-m", "memtree_mcp.http",
                          "--host", "127.0.0.1", "--port", str(port)],
                         env=env, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        base = f"http://127.0.0.1:{port}"
        for _ in range(50):  # wait for the listener
            try:
                with urllib.request.urlopen(f"{base}/healthz", timeout=2) as r:
                    if r.status == 200:
                        break
            except Exception:
                time.sleep(0.1)
        else:
            raise RuntimeError("server never came up")
        st, _, body = (None,) * 3
        import urllib.request as u
        with u.urlopen(base + "/healthz", timeout=5) as r:
            health = json.loads(r.read())
        assert health["ok"] is True and health["nodes"] >= 0 and health["tools"] == 7, health

        st, hdr, init = post(f"{base}/mcp", {"jsonrpc": "2.0", "id": 1,
                              "method": "initialize", "params": {"protocolVersion": "2025-06-18",
                              "capabilities": {}, "clientInfo": {"name": "t", "version": "0"}}})
        assert st == 200 and init["result"]["serverInfo"]["name"] == "memtree", init
        sid = hdr.get("Mcp-Session-Id") or hdr.get("mcp-session-id")
        assert sid, "no Mcp-Session-Id returned"

        st, _, tools = post(f"{base}/mcp", {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
                            {"Mcp-Session-Id": sid})
        names = [t["name"] for t in tools["result"]["tools"]]
        assert len(names) == 7 and "memtree_prefetch" in names and "memtree_timer" in names, names

        st, _, pre = post(f"{base}/mcp", {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                          "params": {"name": "memtree_prefetch", "arguments": {"query": "gate"}}},
                          {"Mcp-Session-Id": sid})
        text = pre["result"]["content"][0]["text"]
        assert "Fresh memory:" in text, text[:120]
        print(f"  raw HTTP: healthz ok, initialize->session {sid[:8]}, {len(names)} tools, "
              f"prefetch {len(text)} chars")
    finally:
        p.terminate()
        p.wait(timeout=10)


def test_official_sdk_client_over_http():
    try:
        import asyncio
        from mcp import ClientSession, StdioServerParameters  # noqa
        from mcp.client.streamable_http import streamable_http_client  # noqa
    except Exception as e:
        print(f"  SKIP official SDK http client ({type(e).__name__})")
        return

    async def run():
        home = fixture_home()
        port = free_port()
        env = dict(os.environ, PYTHONPATH=ROOT, MEMTREE_HOME=home)
        p = subprocess.Popen([sys.executable, "-m", "memtree_mcp.http",
                              "--host", "127.0.0.1", "--port", str(port)],
                             env=env, cwd=ROOT)
        try:
            url = f"http://127.0.0.1:{port}/mcp"
            # the official SDK's http client (streamable http transport)
            from mcp.client.streamable_http import streamable_http_client
            for _ in range(50):
                try:
                    urllib.request.urlopen(url.replace("/mcp", "/healthz"), timeout=2)
                    break
                except Exception:
                    await asyncio.sleep(0.1)
            async with streamable_http_client(url) as (read, write):
                async with ClientSession(read, write) as session:
                    init = await session.initialize()
                    si = getattr(init, "server_info", None) or getattr(init, "serverInfo", None)
                    names = [t.name for t in (await session.list_tools()).tools]
                    assert len(names) == 7
                    print(f"  SDK http client: {si.name} init ok, {len(names)} tools over streamable http")
        finally:
            p.terminate()
            p.wait(timeout=10)

    asyncio.run(run())


if __name__ == "__main__":
    failed = 0
    for fn in (test_http_serves_claude_connector_handshake, test_official_sdk_client_over_http):
        try:
            fn()
            print("PASS", fn.__name__)
        except Exception as e:
            failed += 1
            print("FAIL", fn.__name__, "->", type(e).__name__, e)
    print(f"\n{2 - failed} passed, {failed} failed")
    raise SystemExit(1 if failed else 0)