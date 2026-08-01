"""HTTP-сервер живого просмотра: страница, поток дельт, экспорт, управление.

Всё на стандартной библиотеке, поэтому просмотр поднимается одной строкой из
любого скрипта рисования и не тянет за собой веб-фреймворк.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

UI_FILE = Path(__file__).with_name("ui.html")


def serve(view, tries: int = 20) -> ThreadingHTTPServer:
    """Поднять сервер в отдельном потоке и вернуть его.

    Занятый порт не повод падать: рядом почти всегда есть свободный, а
    занимает его обычно прошлый запуск того же скрипта.
    """
    for offset in range(tries):
        try:
            server = ThreadingHTTPServer(("127.0.0.1", view.port + offset), _handler(view))
        except OSError:
            continue
        view.port += offset
        server.daemon_threads = True
        threading.Thread(target=server.serve_forever, daemon=True).start()
        return server
    raise OSError(f"свободный порт не нашёлся: {view.port}…{view.port + tries - 1}")


def _handler(view):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):  # тишина в консоли скрипта
            pass

        # --- ответы ---

        def _send(self, body: bytes, ctype: str, download: str = "") -> None:
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            if download:
                self.send_header("Content-Disposition", f'attachment; filename="{download}"')
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802
            path, _, query = self.path.partition("?")
            p = dict(kv.split("=", 1) for kv in query.split("&") if "=" in kv)

            if path == "/":
                self._send(UI_FILE.read_bytes(), "text/html; charset=utf-8")
            elif path == "/stream":
                self._stream()
            elif path == "/frame.png":
                png = view.frame_png()
                if png is None:
                    self.send_error(404)
                else:
                    self._send(png, "image/png", download=p.get("download", ""))
            elif path == "/step.png":
                png = view.step_png(int(p.get("i", -1)))
                if png is None:
                    self.send_error(404)
                else:
                    self._send(png, "image/png")
            elif path == "/timelapse.gif":
                gif = view.timelapse_gif(scale=int(p.get("scale", 4)), fps=int(p.get("fps", 6)))
                self._send(gif, "image/gif", download="timelapse.gif")
            elif path == "/control":
                if "hold" in p:
                    view.hold(p["hold"] == "1")
                if "pace" in p:
                    view.pace = float(p["pace"])
                self._send(
                    json.dumps({"pace": view.pace}).encode(),
                    "application/json",
                )
            else:
                self.send_error(404)

        # --- поток дельт ---

        def _stream(self):
            sub = view.subscribe()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            try:
                while view._running:
                    if not sub.q:
                        sub.ev.wait(timeout=15.0)
                        sub.ev.clear()
                        if not sub.q:
                            self.wfile.write(b": keep-alive\n\n")
                            self.wfile.flush()
                            continue
                    while sub.q:
                        self.wfile.write(sub.q.popleft().encode())
                    self.wfile.flush()
                    if sub.lost:
                        # зритель отстал, очередь чистили: досылаем всё целиком
                        sub.lost = False
                        view._send_full(sub)
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass
            finally:
                view.unsubscribe(sub)

    return Handler
