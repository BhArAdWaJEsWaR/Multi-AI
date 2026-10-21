"""Small browser frontend for the local multi-agent assistant.

Run from the project root:
    python3 frontend_server.py
Then open http://127.0.0.1:8000
"""

from __future__ import annotations

import json
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from core.orchestrator import Orchestrator


ROOT = Path(__file__).resolve().parent
FRONTEND_DIR = ROOT / "frontend"
HOST = "127.0.0.1"
PORT = 8000


class FrontendHandler(BaseHTTPRequestHandler):
    orchestrator = Orchestrator(use_validator=True)

    def log_message(self, fmt: str, *args: object) -> None:
        print(f"[frontend] {self.address_string()} - {fmt % args}", flush=True)

    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
        except BrokenPipeError:
            # The browser may cancel a request when the user submits another
            # message or closes the page. The model request itself completed.
            return

    def _send_file(self, path: Path) -> None:
        if not path.is_file() or FRONTEND_DIR not in path.parents:
            self._send_json(404, {"error": "Not found"})
            return
        body = path.read_bytes()
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        route = urlparse(self.path).path
        if route == "/":
            self._send_file(FRONTEND_DIR / "index.html")
        elif route.startswith("/static/"):
            self._send_file(FRONTEND_DIR / route.removeprefix("/static/"))
        elif route == "/api/health":
            self._send_json(200, {"ok": True})
        else:
            self._send_json(404, {"error": "Not found"})

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/chat":
            self._send_json(404, {"error": "Not found"})
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 1_000_000:
                raise ValueError("Request body must be between 1 byte and 1 MB.")
            payload = json.loads(self.rfile.read(length))
            query = str(payload.get("message", "")).strip()
            if not query:
                raise ValueError("Message cannot be empty.")

            result = self.orchestrator.handle(query, verbose=False)
            self._send_json(200, {
                "response": result["response"],
                "task_type": result["task_type"],
                "difficulty": result["difficulty"],
                "tier": result["tier"],
                "escalated": result["escalated"],
            })
        except (ValueError, json.JSONDecodeError) as exc:
            self._send_json(400, {"error": str(exc)})
        except Exception as exc:
            print(f"[frontend] request failed: {exc}", flush=True)
            self._send_json(500, {"error": "The assistant could not process this request."})


def main() -> None:
    server = ThreadingHTTPServer((HOST, PORT), FrontendHandler)
    print(f"Frontend ready at http://{HOST}:{PORT}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nFrontend stopped.", flush=True)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
