"""PathSnap evaluator UI server (Laptop 2). Standard library only, no pip install needed.

    python ui/serve.py --api http://<laptop-1-ip>:8000 --port 8501
    then open http://localhost:8501

It serves the static page in this folder and tells it where the API is (via /config.js).
"""
import argparse
import functools
import json
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent


class Handler(SimpleHTTPRequestHandler):
    api_url = "http://localhost:8000"

    def do_GET(self):
        if self.path.split("?")[0] == "/config.js":
            body = f"window.PATHSNAP_API = {json.dumps(self.api_url)};\n".encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/javascript")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        super().do_GET()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://localhost:8000", help="base URL of the PathSnap API on Laptop 1")
    ap.add_argument("--port", type=int, default=8501)
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()
    Handler.api_url = a.api.rstrip("/")
    server = ThreadingHTTPServer(("0.0.0.0", a.port), functools.partial(Handler, directory=str(HERE)))
    url = f"http://localhost:{a.port}"
    print(f"PathSnap UI on {url}  (API: {Handler.api_url})  Ctrl+C to stop")
    if not a.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
