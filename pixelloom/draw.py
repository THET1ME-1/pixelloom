"""Примитивы рисования по пикселям и дизеринг."""

from __future__ import annotations

import numpy as np

from .canvas import TRANSPARENT, Canvas

# Матрица Байера 4x4: порог упорядоченного дизеринга, значения 0..15
BAYER4 = np.array(
    [
        [0, 8, 2, 10],
        [12, 4, 14, 6],
        [3, 11, 1, 9],
        [15, 7, 13, 5],
    ],
    dtype=np.float64,
)

BAYER8 = np.array(
    [
        [0, 32, 8, 40, 2, 34, 10, 42],
        [48, 16, 56, 24, 50, 18, 58, 26],
        [12, 44, 4, 36, 14, 46, 6, 38],
        [60, 28, 52, 20, 62, 30, 54, 22],
        [3, 35, 11, 43, 1, 33, 9, 41],
        [51, 19, 59, 27, 49, 17, 57, 25],
        [15, 47, 7, 39, 13, 45, 5, 37],
        [63, 31, 55, 23, 61, 29, 53, 21],
    ],
    dtype=np.float64,
)


def line(canvas: Canvas, x0: int, y0: int, x1: int, y1: int, index: int) -> None:
    """Отрезок по Брезенхэму, толщиной ровно один пиксель."""
    dx, dy = abs(x1 - x0), -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    while True:
        canvas.set(x0, y0, index)
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy


def rect(canvas: Canvas, x0: int, y0: int, x1: int, y1: int, index: int, fill: bool = False) -> None:
    if fill:
        xa, xb = sorted((x0, x1))
        ya, yb = sorted((y0, y1))
        xa, ya = max(0, xa), max(0, ya)
        xb, yb = min(canvas.width - 1, xb), min(canvas.height - 1, yb)
        if xa <= xb and ya <= yb:
            canvas.data[ya : yb + 1, xa : xb + 1] = index
        return
    line(canvas, x0, y0, x1, y0, index)
    line(canvas, x1, y0, x1, y1, index)
    line(canvas, x1, y1, x0, y1, index)
    line(canvas, x0, y1, x0, y0, index)


def ellipse(
    canvas: Canvas, cx: float, cy: float, rx: float, ry: float, index: int, fill: bool = True
) -> None:
    yy, xx = np.mgrid[0 : canvas.height, 0 : canvas.width]
    d = ((xx - cx) / max(rx, 0.001)) ** 2 + ((yy - cy) / max(ry, 0.001)) ** 2
    if fill:
        canvas.data[d <= 1.0] = index
    else:
        canvas.data[(d <= 1.0) & (d > 0.82)] = index


def circle(canvas: Canvas, cx: float, cy: float, r: float, index: int, fill: bool = True) -> None:
    ellipse(canvas, cx, cy, r, r, index, fill)


def flood_fill(canvas: Canvas, x: int, y: int, index: int) -> None:
    """Заливка области одного цвета по четырём соседям."""
    target = canvas.get(x, y)
    if target == index or not canvas.inside(x, y):
        return
    stack = [(x, y)]
    while stack:
        px, py = stack.pop()
        if not canvas.inside(px, py) or canvas.data[py, px] != target:
            continue
        canvas.data[py, px] = index
        stack.extend([(px + 1, py), (px - 1, py), (px, py + 1), (px, py - 1)])


def shade_from_field(
    canvas: Canvas,
    field: np.ndarray,
    ramp: list[int],
    mask: np.ndarray | None = None,
    dither: float = 0.0,
    matrix: np.ndarray = BAYER4,
    noise: float = 0.0,
    seed: int = 0,
    noise_field: np.ndarray | None = None,
) -> None:
    """Разложить непрерывное поле яркости 0..1 по шкале палитры.

    dither задаёт долю ступени, которую размывает матрица Байера. Ноль даёт
    жёсткие полосы, 1.0 полностью растворяет границу между соседними
    цветами шкалы в шахматный узор. noise подмешивает случайное дрожание:
    без него на больших пологих участках проступает решётка матрицы.
    """
    n = len(ramp)
    v = np.clip(field, 0.0, 1.0) * (n - 1)
    if dither > 0:
        m = matrix / matrix.size
        th = np.tile(
            m,
            (canvas.height // m.shape[0] + 1, canvas.width // m.shape[1] + 1),
        )[: canvas.height, : canvas.width]
        v = v + (th - 0.5) * dither
    if noise > 0:
        if noise_field is None:
            rng = np.random.default_rng(seed)
            jitter = rng.random((canvas.height, canvas.width)) - 0.5
        else:
            # дрожание, снятое с поверхности: при вращении оно едет вместе с
            # планетой, а не кипит на месте, как экранный шум
            jitter = noise_field - 0.5
        v = v + jitter * noise
    idx = np.clip(np.rint(v), 0, n - 1).astype(int)
    lut = np.array(ramp, dtype=np.int16)
    out = lut[idx]
    if mask is None:
        canvas.data[:, :] = out
    else:
        canvas.data[mask] = out[mask]


def outline(canvas: Canvas, index: int, diagonal: bool = False) -> None:
    """Обвести непрозрачный силуэт снаружи."""
    solid = canvas.data != TRANSPARENT
    grown = np.zeros_like(solid)
    shifts = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if diagonal:
        shifts += [(-1, -1), (1, -1), (-1, 1), (1, 1)]
    for dx, dy in shifts:
        grown |= np.roll(np.roll(solid, dy, axis=0), dx, axis=1)
    canvas.data[grown & ~solid] = index
