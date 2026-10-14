"""Streamable HTTP transport for the memtree MCP server (stdlib only).

Lets the same `MemtreeServer` talk to Claude over HTTP(S) -- the shape a remote
custom connector (claude.ai / mobile / Cowork) needs -- behind Apache/nginx/Caddy
as a TLS reverse proxy, or directly.

Endpoints
    GET  /healthz      200 "ok" + one-line JSON status
    POST /mcp          JSON-RPC in, JSON-RPC out (one object per line)
    GET  /mcp          optional SSE stream (Accept: text/event-stream) with the
                       Mcp-Session-Id handshake; responses still come on POST
    OPTIONS /mcp       CORS preflight

Sessions: each client gets a `MemtreeServer` keyed by a Mcp-Session-Id header.
The first request (initialize) returns `Mcp-Session-Id`; later requests echo it.
Thread-safe: each handler holds its own server + the shared Substrate's lock.

Run it:
    PYTHONPATH=$PWD MEMTREE_HOME=$PWD/var python3 -m memtree_mcp.http --port 8080
    ./run_http.sh --port 8080

Apache example in deploy/apache.conf.
"""
from __future__ import annotations

import argparse
import json
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict, Optional

from .server import MemtreeServer
from .substrate import Substrate

SESSION_HEADER = "Mcp-Session-Id"
CORS = {"Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type, Mcp-Session-Id, Accept, Authorization"}


class Handler(BaseHTTPRequestHandler):
    session_servers: Dict[str, MemtreeServer] = {}
    sessions_lock = threading.Lock()
    shared_sub: Optional[Substrate] = None

    # ---------------------------------------------------------------- helpers
    def _json(self, code: int, obj) -> None:
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        for k, v in CORS.items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _text(self, code: int, text: str, ctype: str = "text/plain") -> None:
        body = text.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for k, v in CORS.items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _server_for(self, sid: Optional[str]) -> MemtreeServer:
        if not sid:
            sid = uuid.uuid4().hex
        with Handler.sessions_lock:
            srv = Handler.session_servers.get(sid)
            if srv is None:
                srv = MemtreeServer(Handler.shared_sub)
                Handler.session_servers[sid] = srv
        return srv, sid

    # ---------------------------------------------------------------- routing
    def do_OPTIONS(self):
        self.send_response(204)
        for k, v in CORS.items():
            self.send_header(k, v)
        self.end_headers()

    def do_DELETE(self):
        """MCP Streamable HTTP: DELETE /mcp ends the session (RFC-ish close)."""
        sid = self.headers.get(SESSION_HEADER)
        if sid:
            with Handler.sessions_lock:
                Handler.session_servers.pop(sid, None)
        self.send_response(204)
        for k, v in CORS.items():
            self.send_header(k, v)
        self.end_headers()

    def do_GET(self):
        if self.path in ("/healthz", "/health", "/"):
            st = Handler.shared_sub.stats() if Handler.shared_sub else {}
            self._json(200, {"ok": True, "service": "memtree-mcp",
                             "home": st.get("home"), "nodes": st.get("nodes"),
                             "peer": st.get("peer"), "turn": st.get("turn"),
                             "tools": 7})
            return
        if self.path.startswith("/mcp") and "text/event-stream" in (self.headers.get("Accept") or ""):
            # open an SSE stream so the client can receive server->client events;
            # responses still come back on POST. keep the connection warm.
            sid = self.headers.get(SESSION_HEADER) or uuid.uuid4().hex
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header(SESSION_HEADER, sid)
            for k, v in CORS.items():
                self.send_header(k, v)
            self.end_headers()
            try:
                while True:
                    self.wfile.write(b": keepalive\n\n")
                    self.wfile.flush()
                    import time
                    time.sleep(15)
            except (BrokenPipeError, ConnectionResetError):
                pass
            return
        self._json(404, {"error": "not found"})

    def do_POST(self):
        if not self.path.startswith("/mcp"):
            self._json(404, {"error": "not found"})
            return
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b""
        srv, sid = self._server_for(self.headers.get(SESSION_HEADER))
        responses = []
        for line in raw.decode("utf-8", "replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                req = json.loads(line)
            except ValueError:
                continue
            resp = srv.handle(req)
            if resp is not None:
                responses.append(resp)
        if not responses:
            responses = [{"jsonrpc": "2.0", "id": None, "error": {"code": -32600, "message": "no request"}}]
        body = ("\n".join(json.dumps(r) for r in responses) + "\n").encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header(SESSION_HEADER, sid)
        for k, v in CORS.items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):  # quiet
        pass


def serve(host: str = "127.0.0.1", port: int = 8080, home: Optional[str] = None,
          unix: Optional[str] = None) -> None:
    Handler.shared_sub = Substrate(home=home)
    Handler.session_servers = {}
    httpd = ThreadingHTTPServer((host, port), Handler)
    print(f"memtree-mcp http (streamable) on http://{host}:{port}/mcp  "
          f"(home {Handler.shared_sub.home}, tools: elementary 6, presses Ctrl-C to stop)",
          flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped", flush=True)
    finally:
        try:
            Handler.shared_sub.close()
        except Exception:
            pass


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="memtree MCP over Streamable HTTP")
    ap.add_argument("--host", default=os_env("MEMTREE_HOST", "0.0.0.0"))
    ap.add_argument("--port", type=int, default=int(os_env("MEMTREE_PORT", "8080")))
    ap.add_argument("--home", default=None)
    args = ap.parse_args(argv)
    serve(args.host, args.port, args.home)
    return 0


def os_env(k, d):
    import os
    return os.environ.get(k, d)


if __name__ == "__main__":
    raise SystemExit(main())