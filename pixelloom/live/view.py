"""Живой просмотр: пиксели уезжают в браузер в тот момент, когда их пишут.

Как это работает. Холст зовёт наблюдателя после каждой записи. Наблюдатель
сравнивает холст с прошлым снимком, режет разницу на порции и отдаёт их
подписчикам по одной. Браузер кладёт порции в свой холст, поэтому рисунок
проступает ровно в том порядке, в каком его ведёт код.

Картинка целиком по сети не гоняется: в сообщении едет либо прямоугольник
изменившейся области, либо разрозненные пары «смещение, цвет» — смотря что
короче. Полный кадр отправляется один раз, когда зритель только подключился.
"""

from __future__ import annotations

import base64
import io
import struct
import sys
import threading
import time
from collections import deque
from contextlib import contextmanager

import numpy as np
from PIL import Image

from ..canvas import Canvas
from ..render import to_image

# 255 в дельте означает прозрачный пиксель: палитры длиннее 255 цветов
# индексированная графика всё равно не переживёт
CLEAR = 255

# сколько порций тратим на один шаг, каким бы большим он ни был
MAX_CHUNKS = 120
# меньше этого порцию не дробим: сообщение на каждый пиксель забивает канал,
# а глаз всё равно не различает такие подробности на ходу
MIN_CHUNK = 24
# пауза между порциями при pace = 1
CHUNK_DELAY = 0.03
# и потолок на весь шаг: заливка в полкартины не должна тянуться полминуты
STEP_BUDGET = 1.2


class _Sub:
    """Очередь сообщений одного зрителя."""

    __slots__ = ("q", "ev", "lost")

    def __init__(self):
        self.q = deque(maxlen=2048)
        self.ev = threading.Event()
        self.lost = False

    def push(self, msg: str) -> None:
        if len(self.q) == self.q.maxlen:
            # зритель не успевает; чистим очередь и просим полный кадр
            self.q.clear()
            self.lost = True
        self.q.append(msg)
        self.ev.set()


class Step:
    __slots__ = ("name", "t", "pixels", "png")

    def __init__(self, name: str, t: float):
        self.name = name
        self.t = t
        self.pixels = 0
        self.png: bytes | None = None


class LiveView:
    """Один холст, много зрителей, поток пикселей в реальном времени."""

    def __init__(
        self,
        port: int = 8765,
        pace: float = 1.0,
        reveal: str = "rows",
        keep: int = 240,
        title: str = "pixelloom",
        chunk: int = MIN_CHUNK,
    ):
        self.port = port
        self.reveal = reveal
        self.keep = keep
        self.chunk = max(1, chunk)

        self._lock = threading.Lock()
        self._subs: list[_Sub] = []
        self._canvas: Canvas | None = None
        self._prev: np.ndarray | None = None
        self._title = title
        self._version = 0
        self._pace = pace
        self._gate = threading.Event()
        self._gate.set()  # снят с паузы

        self._step_name: str | None = None
        self._steps: list[Step] = []
        self._current: Step | None = None
        self._drawn = 0
        self._rate = 0.0
        self._rate_mark = time.time()
        self._rate_count = 0
        self._box = (0, 0, 0, 0)
        self._color = -1

        self._queue: list[dict] = []
        self._anims: list[dict] = []
        self._server = None
        self._running = False
        self._started_at = time.time()

    # ── жизненный цикл ────────────────────────────────────────────────

    def start(self, open_browser: bool = False) -> "LiveView":
        from .server import serve

        self._running = True
        self._server = serve(self)
        if open_browser:
            import subprocess

            subprocess.Popen(
                ["xdg-open", self.url],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        return self

    def stop(self) -> None:
        self._running = False
        if self._server:
            self._server.shutdown()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}/"

    def __enter__(self) -> "LiveView":
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    # ── подключение холста ────────────────────────────────────────────

    def attach(self, canvas: Canvas, title: str = "") -> None:
        """Следить за холстом. Любая запись пикселей уедет в браузер сама."""
        with self._lock:
            self._canvas = canvas
            if title:
                self._title = title
            self._prev = np.full_like(np.asarray(canvas.data), -1)
            self._version += 1
        canvas.watch(self._on_change)
        self._broadcast_meta()
        self._send_full()

    def detach(self) -> None:
        if self._canvas is not None:
            self._canvas.watch(None)

    # ── разметка работы ───────────────────────────────────────────────

    @contextmanager
    def step(self, name: str):
        """Назвать шаг вручную: `with view.step("глаза"): ...`"""
        prev = self._step_name
        self._step_name = name
        try:
            yield self
        finally:
            self._close_step()
            self._step_name = prev

    def task(self, name: str, total: int = 0) -> None:
        """Отметить, какая работа пошла: имя видно в очереди слева."""
        with self._lock:
            for q in self._queue:
                if q["name"] == name:
                    q["state"] = "now"
                elif q["state"] == "now":
                    q["state"] = "done"
            if not any(q["name"] == name for q in self._queue):
                self._queue.append({"name": name, "state": "now"})
            self._steps.clear()
            self._current = None
            self._drawn = 0
            self._started_at = time.time()
        self._broadcast_meta()

    def plan(self, names: list[str]) -> None:
        """Заранее показать очередь работ."""
        with self._lock:
            self._queue = [{"name": n, "state": "wait"} for n in names]
        self._broadcast_meta()

    def animation(self, frames: list[Canvas], name: str = "анимация", fps: int = 10) -> None:
        """Отдать готовую анимацию в плеер справа."""
        png = []
        for f in frames:
            buf = io.BytesIO()
            to_image(f).save(buf, format="PNG")
            png.append(base64.b64encode(buf.getvalue()).decode())
        with self._lock:
            self._anims = [a for a in self._anims if a["name"] != name]
            self._anims.append({"name": name, "fps": fps, "frames": png})
        self._broadcast_meta()

    def wait_for_viewer(self, timeout: float = 20.0) -> bool:
        """Подождать зрителя. Без этого начало работы уходит в пустоту."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self._lock:
                if self._subs:
                    return True
            time.sleep(0.1)
        return False

    # ── темп ──────────────────────────────────────────────────────────

    @property
    def pace(self) -> float:
        return self._pace

    @pace.setter
    def pace(self, value: float) -> None:
        self._pace = max(0.0, float(value))

    def hold(self, on: bool) -> None:
        """Пауза: рисующий поток останавливается на следующей же записи."""
        if on:
            self._gate.clear()
        else:
            self._gate.set()

    # ── наблюдение ────────────────────────────────────────────────────

    def _on_change(self, canvas: Canvas) -> None:
        if self._prev is None or not self._running:
            return
        self._gate.wait()

        cur = np.asarray(canvas.data)
        diff = cur != self._prev
        n = int(diff.sum())
        if n == 0:
            return

        name = self._step_name or self._caller_name()
        if self._current is None or self._current.name != name:
            self._close_step()
            self._open_step(name)

        ys, xs = np.where(diff)
        vals = cur[ys, xs]
        order = self._order(ys, xs, n)
        chunk = max(self.chunk, -(-n // MAX_CHUNKS))
        self._color = int(vals[order[0]])

        # мелкий шаг успевает показать себя целиком, крупный укладывается в срок
        parts = -(-n // chunk)
        delay = min(CHUNK_DELAY, STEP_BUDGET / parts)

        for start in range(0, n, chunk):
            self._gate.wait()
            sel = order[start : start + chunk]
            self._emit(ys[sel], xs[sel], vals[sel], name)
            if self._pace > 0:
                time.sleep(delay * self._pace)

        np.copyto(self._prev, cur)
        if self._current is not None:
            self._current.pixels += n

    def _order(self, ys: np.ndarray, xs: np.ndarray, n: int) -> np.ndarray:
        """Порядок, в котором изменения проступают в браузере."""
        if self.reveal == "random":
            rng = np.random.default_rng(n)
            return rng.permutation(n)
        if self.reveal == "radial":
            cy, cx = ys.mean(), xs.mean()
            return np.argsort((ys - cy) ** 2 + (xs - cx) ** 2)
        return np.arange(n)  # rows: numpy уже отдал координаты построчно

    def _emit(self, ys: np.ndarray, xs: np.ndarray, vals: np.ndarray, step: str) -> None:
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        area = (x1 - x0) * (y1 - y0)
        canvas = self._canvas
        assert canvas is not None

        if area <= 2 * len(xs):
            # плотный кусок: дешевле отдать прямоугольник целиком
            canvas.mute(True)
            block = np.asarray(canvas.data)[y0:y1, x0:x1]
            canvas.mute(False)
            body = np.where(block < 0, CLEAR, block).astype(np.uint8).tobytes()
            shape = "rect"
        else:
            w = canvas.width
            packed = bytearray()
            for y, x, v in zip(ys.tolist(), xs.tolist(), vals.tolist()):
                packed += struct.pack("<IB", y * w + x, CLEAR if v < 0 else v)
            body = bytes(packed)
            shape = "sparse"
        box = [x0, y0, x1 - x0, y1 - y0]

        self._version += 1
        self._drawn += len(xs)
        self._rate_count += len(xs)
        now = time.time()
        if now - self._rate_mark >= 0.15:
            self._rate = self._rate_count / (now - self._rate_mark)
            self._rate_mark = now
            self._rate_count = 0
        self._box = tuple(box)

        self._broadcast(
            "delta",
            {
                "v": self._version,
                "shape": shape,
                "box": box,
                "data": base64.b64encode(body).decode(),
                "step": step,
                "color": self._color,
                "drawn": self._drawn,
                "rate": round(self._rate),
                "t": round(now - self._started_at, 2),
            },
        )

    def _caller_name(self) -> str:
        """Имя шага из стека вызовов: первый кадр за пределами библиотеки."""
        f = sys._getframe(1)
        for _ in range(14):
            if f is None:
                break
            mod = f.f_globals.get("__name__", "")
            if not mod.startswith("pixelloom"):
                name = f.f_code.co_name
                return "рисование" if name in ("<module>", "main") else name
            f = f.f_back
        return "рисование"

    # ── шаги и снимки ─────────────────────────────────────────────────

    def _open_step(self, name: str) -> None:
        with self._lock:
            self._current = Step(name, round(time.time() - self._started_at, 2))
            self._steps.append(self._current)
            if len(self._steps) > self.keep:
                del self._steps[0]
        self._broadcast("steps", {"steps": self._steps_json()})

    def _close_step(self) -> None:
        if self._current is None or self._canvas is None:
            return
        self._canvas.mute(True)
        buf = io.BytesIO()
        to_image(self._canvas).save(buf, format="PNG")
        self._canvas.mute(False)
        self._current.png = buf.getvalue()
        self._current = None
        self._broadcast("steps", {"steps": self._steps_json()})

    def _steps_json(self) -> list[dict]:
        return [
            {"i": i, "name": s.name, "t": s.t, "px": s.pixels, "shot": s.png is not None}
            for i, s in enumerate(self._steps)
        ]

    # ── связь ─────────────────────────────────────────────────────────

    def subscribe(self) -> _Sub:
        sub = _Sub()
        with self._lock:
            self._subs.append(sub)
        self._send_meta_to(sub)
        self._send_full(sub)
        return sub

    def unsubscribe(self, sub: _Sub) -> None:
        with self._lock:
            if sub in self._subs:
                self._subs.remove(sub)

    def _broadcast(self, kind: str, payload: dict) -> None:
        import json

        msg = f"data: {json.dumps({'kind': kind, **payload})}\n\n"
        with self._lock:
            subs = list(self._subs)
        for s in subs:
            s.push(msg)

    def _broadcast_meta(self) -> None:
        with self._lock:
            subs = list(self._subs)
        for s in subs:
            self._send_meta_to(s)

    def _meta(self) -> dict:
        c = self._canvas
        pal = []
        if c is not None:
            pal = [
                {"i": i, "hex": "#{:02X}{:02X}{:02X}".format(*col), "label": c.palette.labels[i]}
                for i, col in enumerate(c.palette.colors)
            ]
        return {
            "kind": "meta",
            "title": self._title,
            "width": c.width if c else 0,
            "height": c.height if c else 0,
            "palette": pal,
            "queue": list(self._queue),
            "anims": list(self._anims),
            "steps": self._steps_json(),
            "pace": self._pace,
        }

    def _send_meta_to(self, sub: _Sub) -> None:
        import json

        sub.push(f"data: {json.dumps(self._meta())}\n\n")

    def _send_full(self, sub: _Sub | None = None) -> None:
        """Полный кадр одним прямоугольником: так зритель догоняет работу,
        которая шла до его подключения."""
        c = self._canvas
        if c is None:
            return
        import json

        c.mute(True)
        block = np.asarray(c.data)
        body = np.where(block < 0, CLEAR, block).astype(np.uint8).tobytes()
        c.mute(False)
        msg = "data: " + json.dumps(
            {
                "kind": "delta",
                "shape": "rect",
                "v": self._version,
                "box": [0, 0, c.width, c.height],
                "data": base64.b64encode(body).decode(),
                "step": self._current.name if self._current else "",
                "color": self._color,
                "drawn": self._drawn,
                "rate": 0,
                "t": round(time.time() - self._started_at, 2),
                "full": True,
            }
        ) + "\n\n"
        if sub is not None:
            sub.push(msg)
        else:
            with self._lock:
                subs = list(self._subs)
            for s in subs:
                s.push(msg)

    # ── экспорт ───────────────────────────────────────────────────────

    def frame_png(self) -> bytes | None:
        c = self._canvas
        if c is None:
            return None
        c.mute(True)
        buf = io.BytesIO()
        to_image(c).save(buf, format="PNG")
        c.mute(False)
        return buf.getvalue()

    def step_png(self, index: int) -> bytes | None:
        with self._lock:
            if 0 <= index < len(self._steps):
                return self._steps[index].png
        return None

    def timelapse_gif(self, scale: int = 4, fps: int = 6) -> bytes:
        """Лента шагов одним файлом: показать другим, как шла работа."""
        with self._lock:
            shots = [s.png for s in self._steps if s.png]
        if not shots:
            cur = self.frame_png()
            shots = [cur] if cur else []
        imgs = []
        for png in shots:
            img = Image.open(io.BytesIO(png)).convert("RGBA")
            img = img.resize((img.width * scale, img.height * scale), Image.NEAREST)
            flat = Image.new("RGBA", img.size, (11, 10, 15, 255))
            flat.alpha_composite(img)
            imgs.append(flat.convert("RGB"))
        if not imgs:
            return b""
        # общая палитра на всю ленту: покадровое квантование разводит цвета
        base = imgs[0].quantize(colors=128, method=Image.MEDIANCUT)
        imgs = [base] + [im.quantize(palette=base, dither=Image.NONE) for im in imgs[1:]]
        buf = io.BytesIO()
        imgs[0].save(
            buf,
            format="GIF",
            save_all=True,
            append_images=imgs[1:],
            duration=int(1000 / fps),
            loop=0,
            disposal=2,
        )
        return buf.getvalue()
