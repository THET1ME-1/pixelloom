#!/usr/bin/env python3
"""Планета: свет, рельеф из шума и дизеринг по шкале палитры.

    python3 examples/planet.py                # рисовать с показом в браузере
    python3 examples/planet.py --size 128     # крупнее
    python3 examples/planet.py --spin 24      # ещё и кадры вращения

Пример показывает вторую половину библиотеки: непрерывное поле яркости
раскладывается по шкале палитры, а дизеринг разбивает границы полос.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from pixelloom import Canvas, draw, noise, palettes, save_gif, save_png
from pixelloom.live import LiveView


def build(c: Canvas, view: LiveView | None, spin: float = 0.0) -> None:
    """Собрать планету. Каждый шаг подписан, поэтому в ленте видно порядок."""
    n = c.width
    pal = c.palette
    yy, xx = np.mgrid[0:n, 0:n]
    cx = cy = (n - 1) / 2
    r = n * 0.42

    # сфера: нормаль к поверхности и освещение с левого верха
    dx, dy = (xx - cx) / r, (yy - cy) / r
    d2 = dx * dx + dy * dy
    disc = d2 <= 1.0
    dz = np.sqrt(np.clip(1 - d2, 0, 1))
    light = np.clip(dx * -0.55 + dy * -0.55 + dz * 0.62, 0, 1)

    # долгота и широта: по ним шум едет вместе с поверхностью при вращении
    lon = np.arctan2(dx, np.where(dz == 0, 1e-6, dz)) + spin * np.pi * 2
    lat = np.arcsin(np.clip(dy, -1, 1))
    field = noise.fbm((n, n), cells=3, octaves=5, seed=7)
    shifted = np.roll(field, int(spin * n) % n, axis=1)
    relief = shifted * 0.55 + (np.cos(lon * 2) * 0.12 + 0.5) * 0.45

    step = view.step if view else _quiet
    ground = pal.ramp("ground")

    with step("диск"):
        c.data[disc] = ground[2]
    with step("свет"):
        draw.shade_from_field(c, light, ground, mask=disc, dither=0.0)
    with step("рельеф"):
        mix = np.clip(light * 0.72 + relief * 0.42, 0, 1)
        draw.shade_from_field(c, mix, ground, mask=disc, dither=0.9, noise=0.35, seed=3)
    with step("полюса светлее"):
        caps = disc & (np.abs(lat) > 1.02)
        if caps.any():
            bright = np.clip(light + 0.25, 0, 1)
            draw.shade_from_field(c, bright, ground[-2:], mask=caps, dither=0.6)
    with step("край"):
        rim = disc & (d2 > 0.86)
        c.data[rim & (light < 0.22)] = ground[0]


class _quiet:
    def __init__(self, *a):
        pass

    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=96)
    ap.add_argument("--spin", type=int, default=0, help="сколько кадров вращения посчитать")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--no-open", action="store_true")
    ap.add_argument("--out", default="out")
    args = ap.parse_args()

    view = LiveView(port=args.port, pace=0.0 if args.fast else 1.0,
                    reveal="radial", title="Планета")
    view.start(open_browser=not args.no_open)
    print(f"живой просмотр: {view.url}")
    if not args.no_open:
        view.wait_for_viewer(timeout=20)
        time.sleep(0.4)

    c = Canvas(args.size, args.size, palettes.MARS)
    view.attach(c, "Планета")
    build(c, view)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    save_png(c, out / "planet.png", scale=4)
    print("картинка:", out / "planet.png")

    if args.spin:
        frames = []
        for i in range(args.spin):
            f = Canvas(args.size, args.size, palettes.MARS)
            build(f, None, spin=i / args.spin)
            frames.append(f)
        view.animation(frames, name="вращение", fps=12)
        save_gif(frames, out / "planet.gif", fps=12, scale=3)
        print("вращение:", out / "planet.gif")

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
