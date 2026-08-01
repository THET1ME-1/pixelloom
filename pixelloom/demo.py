"""Демонстрация: росток в горшке, которого видно, пока он рисуется.

    pixelloom                # открыть браузер и рисовать
    pixelloom --fast         # без пауз, сразу результат
    pixelloom --port 9000    # другой порт

Шаги названы вручную через `view.step`. Без этого имя шага взялось бы из
имени функции, которая пишет пиксели.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np

from .canvas import TRANSPARENT, Canvas
from .live import LiveView
from .palette import Palette
from . import draw
from .render import save_gif, save_png

N = 48

PAL = Palette.from_hex(
    "sprout",
    [
        "#22101a", "#3a1f2b", "#fff3f5", "#ff9fb8",
        "#2f7a45", "#63c46f", "#a8e6a0",
        "#3d2a1c", "#6b4a2f",
        "#8a4425", "#c86a45", "#e79a6c",
    ],
    labels=[
        "ink", "shadow", "white", "blush",
        "leaf_d", "leaf", "leaf_l",
        "dirt_d", "dirt",
        "pot_d", "pot", "pot_l",
    ],
)
I = {name: i for i, name in enumerate(PAL.labels)}


def trapezoid(c: Canvas, y0: int, y1: int, w0: int, w1: int, cx: int, index: int) -> None:
    """Горшок: книзу уже, чем сверху."""
    for y in range(y0, y1 + 1):
        t = (y - y0) / max(1, y1 - y0)
        w = round(w0 + (w1 - w0) * t)
        c.data[y, max(0, cx - w) : min(c.width, cx + w + 1)] = index


def face(c: Canvas, cx: int, y: int, spread: int, blink: bool = False) -> None:
    """Глаза, блики, румянец и рот одним махом."""
    if blink:
        c.data[y + 1, cx - spread - 1 : cx - spread + 1] = I["ink"]
        c.data[y + 1, cx + spread : cx + spread + 2] = I["ink"]
    else:
        c.data[y : y + 3, cx - spread - 1 : cx - spread + 1] = I["ink"]
        c.data[y : y + 3, cx + spread : cx + spread + 2] = I["ink"]
        c.set(cx - spread - 1, y, I["white"])
        c.set(cx + spread, y, I["white"])
    c.data[y + 3, cx - spread - 4 : cx - spread - 2] = I["blush"]
    c.data[y + 3, cx + spread + 3 : cx + spread + 5] = I["blush"]
    c.set(cx - 1, y + 4, I["ink"])
    c.set(cx, y + 5, I["ink"])
    c.set(cx + 1, y + 4, I["ink"])


def sprout(c: Canvas, view: LiveView | None = None, phase: float = 0.0, blink: bool = False) -> None:
    """Собрать персонажа. Порядок шагов тот же, каким его вёл бы художник."""
    cx = c.width // 2
    lean = round(math.sin(phase * math.tau) * 1.4)  # покачивание листьев

    def step(name: str):
        # без живого просмотра контекст не нужен, но код должен работать и так
        return view.step(name) if view else _quiet()

    with step("горшок"):
        trapezoid(c, 28, 40, 9, 7, cx, I["pot"])
    with step("горшок: объём"):
        yy, xx = np.mgrid[0 : c.height, 0 : c.width]
        pot = c.data == I["pot"]
        c.data[pot & ((xx < cx - 3) | (yy > 37))] = I["pot_d"]
        c.data[pot & (xx > cx + 4) & (yy < 36)] = I["pot_l"]
    with step("земля"):
        c.data[27:30, cx - 9 : cx + 10] = I["dirt"]
        c.data[29:30, cx - 9 : cx + 10] = I["dirt_d"]
    with step("стебель"):
        c.data[16:29, cx - 1 : cx + 2] = I["leaf"]
    with step("листья"):
        draw.ellipse(c, cx - 7 + lean, 15, 6, 4, I["leaf"])
        draw.ellipse(c, cx + 7 + lean, 12, 6, 4, I["leaf"])
    with step("листья: тень"):
        leaf = c.data == I["leaf"]
        yy, xx = np.mgrid[0 : c.height, 0 : c.width]
        c.data[leaf & (yy > 15) & (xx < cx - 2)] = I["leaf_d"]
        c.data[leaf & (yy > 12) & (xx > cx + 2)] = I["leaf_d"]
        c.data[16:29, cx + 1 : cx + 2] = I["leaf_d"]
    with step("листья: блик"):
        leaf = c.data == I["leaf"]
        yy, xx = np.mgrid[0 : c.height, 0 : c.width]
        c.data[leaf & (yy < 14) & (xx < cx - 5)] = I["leaf_l"]
        c.data[leaf & (yy < 11) & (xx > cx + 5)] = I["leaf_l"]
    with step("лицо"):
        face(c, cx, 32, 4, blink=blink)
    with step("контур"):
        draw.outline(c, I["ink"])
    with step("тень"):
        # тень ложится последней и только по прозрачному, иначе контур
        # обводит её как отдельный предмет
        shade = Canvas(c.width, c.height, c.palette)
        draw.ellipse(shade, cx, 42, 11, 2, I["shadow"])
        free = np.asarray(c.data) == TRANSPARENT
        c.data[free & (np.asarray(shade.data) != TRANSPARENT)] = I["shadow"]


class _quiet:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="рисовать без пауз")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-open", action="store_true", help="не открывать браузер")
    ap.add_argument("--out", default="out", help="куда сложить картинки")
    args = ap.parse_args()

    view = LiveView(port=args.port, pace=0.0 if args.fast else 1.0, title="Росток")
    view.start(open_browser=not args.no_open)
    print(f"живой просмотр: {view.url}")
    view.plan(["Росток", "Росток: моргает"])

    if not args.no_open:
        view.wait_for_viewer(timeout=20)
        time.sleep(0.4)  # дать браузеру дорисовать страницу

    # 1. Персонаж на глазах у зрителя
    view.task("Росток")
    c = Canvas(N, N, PAL)
    view.attach(c, "Росток")
    sprout(c, view)

    # 2. Кадры анимации: считаем молча и отдаём в плеер справа
    frames = []
    for i in range(12):
        f = Canvas(N, N, PAL)
        sprout(f, None, phase=i / 12, blink=i in (6, 7))
        frames.append(f)
    view.animation(frames, name="покачивание", fps=10)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    save_png(c, out / "sprout.png", scale=8)
    save_gif(frames, out / "sprout.gif", fps=10, scale=6, transparent=True)
    print(f"картинка: {out/'sprout.png'}   анимация: {out/'sprout.gif'}")

    if args.no_open:
        return
    print("окно живёт, пока скрипт не остановят: Ctrl+C")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        view.stop()


if __name__ == "__main__":
    main()
