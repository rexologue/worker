from http.server import BaseHTTPRequestHandler, HTTPServer
from datetime import datetime
import json

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/time':
            now = datetime.now().isoformat()
            payload = json.dumps({'time': now}).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        # Suppress default console logging
        pass

def main():
    server = HTTPServer(('127.0.0.1', 5000), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()

if __name__ == '__main__':
    main()
