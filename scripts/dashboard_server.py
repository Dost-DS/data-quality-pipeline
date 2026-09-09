"""Serve the dashboard against the live experiment database.

Each page load re-runs the queries, so the dashboard reflects the current
contents of the database rather than a snapshot.
"""
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dashboard_queries import collect

TEMPLATE = "dashboard/template.html"
PORT = int(os.environ.get("DASHBOARD_PORT", 8050))


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path not in ("/", "/index.html"):
            self.send_error(404)
            return
        try:
            data = collect()
            html = open(TEMPLATE).read()
            html = html.replace("__DATA__", json.dumps(data, separators=(",", ":")))
            html = html.replace("__LIVE__", "true")
            body = html.encode()
        except Exception as e:
            self.send_error(500, f"database unavailable: {e}")
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    print(f"dashboard on http://localhost:{PORT}   (ctrl-c to stop)")
    HTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
