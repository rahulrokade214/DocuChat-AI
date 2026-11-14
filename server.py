"""Run:  python server.py   then open http://localhost:8000"""
import json
import os
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from rag.llm import answer, load_env
from rag.store import Store

load_env()
PORT = int(os.getenv("PORT", "8000"))
MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
store = Store()
PAGE = Path(__file__).with_name("index.html").read_bytes()


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, payload, ctype="application/json"):
        data = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _body(self):
        n = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(n) or b"{}")

    def do_GET(self):
        if self.path == "/":
            self._send(200, PAGE, "text/html; charset=utf-8")
        elif self.path == "/api/status":
            self._send(200, {"sources": store.sources(), "has_env_key": bool(os.getenv("GROQ_API_KEY")),
                             "model": MODEL})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        try:
            data = self._body()
            if self.path == "/api/upload":
                n = store.add_file(data["name"], data["data"])
                self._send(200, {"chunks": n, "sources": store.sources()})
            elif self.path == "/api/clear":
                store.clear()
                self._send(200, {"sources": []})
            elif self.path == "/api/ask":
                key = (data.get("api_key") or os.getenv("GROQ_API_KEY", "")).strip()
                if not key:
                    raise ValueError("Add your Groq API key (in .env or the sidebar box).")
                q = data["question"]
                hits = store.search(q)
                text = answer(key, MODEL, q, hits, data.get("history", []))
                self._send(200, {"answer": text, "sources": hits})
            else:
                self._send(404, {"error": "not found"})
        except Exception as e:  # show readable errors in the UI
            self._send(400, {"error": str(e)})

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    url = f"http://localhost:{PORT}"
    print(f"Groq RAG Lite running at {url}  (Ctrl+C to stop)")
    try:
        webbrowser.open(url)
    except Exception:
        pass
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
