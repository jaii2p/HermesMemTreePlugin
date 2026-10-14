"""Connector proof: drive the server with the official `mcp` SDK client.

This is the same handshake a Claude connector performs. Skipped (not failed) when
the `mcp` package is absent.

Run: python3 tests/test_sdk_client.py  (from mcp/)
"""
import asyncio
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LIVE = os.environ.get("MEMTREE_LIVE_TREE",
                      os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "demo_tree.json"))


def fixture_home():
    home = tempfile.mkdtemp(prefix="memtree-sdk-")
    os.makedirs(os.path.join(home, "memtree"))
    if os.path.exists(LIVE):
        shutil.copy(LIVE, os.path.join(home, "memtree", "tree.json"))
    return home


async def run():
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    home = fixture_home()
    params = StdioServerParameters(
        command=sys.executable, args=["-m", "memtree_mcp"],
        env=dict(os.environ, MEMTREE_HOME=home, PYTHONPATH=ROOT), cwd=ROOT)
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            init = await session.initialize()
            si = getattr(init, "server_info", None) or getattr(init, "serverInfo", None)
            print("  server:", si.name, si.version, "| protocol:", init.protocol_version
                  if hasattr(init, "protocol_version") else init.protocolVersion)
            tools = await session.list_tools()
            names = [t.name for t in tools.tools]
            print("  tools:", names)
            r = await session.call_tool("memtree_prefetch", {"query": "what did we decide?"})
            block = r.content[0].text
            assert "Fresh memory:" in block, block[:200]
            print("  prefetch via SDK client:", len(block), "chars")
            r = await session.call_tool("memtree_recall", {"query": "dog name"})
            print("  recall via SDK client:", r.content[0].text[:90])
            prompts = await session.list_prompts()
            print("  prompts:", [p.name for p in prompts.prompts])
            res = await session.list_resources()
            print("  resources:", [str(x.uri) for x in res.resources])
            stats = await session.read_resource("memtree://tree/stats")
            print("  stats:", stats.contents[0].text[:120])


if __name__ == "__main__":
    try:
        import mcp  # noqa: F401
    except ImportError:
        print("SKIP: `mcp` SDK not installed")
        raise SystemExit(0)
    asyncio.run(run())
    print("\nSDK client: connector handshake OK")
