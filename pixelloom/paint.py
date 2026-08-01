"""Живое рисование: слои ложатся на холст постепенно, а не мгновенно.

Numpy считает кадр за доли секунды, и зритель видит только результат. Эти
утилиты растягивают наложение во времени, поэтому в браузере картинка
проявляется так, как её вёл бы художник: сначала силуэт, потом свет, потом
детали.
"""

from __future__ import annotations

import time

import numpy as np

from .canvas import Canvas


def reveal_rows(
    canvas: Canvas,
    target: np.ndarray,
    mask: np.ndarray | None = None,
    step: int = 2,
    delay: float = 0.010,
) -> None:
    """Проявить слой сверху вниз полосами по step строк."""
    h = canvas.height
    for y in range(0, h, step):
        y1 = min(h, y + step)
        band = target[y:y1]
        if mask is None:
            canvas.data[y:y1] = band
        else:
            m = mask[y:y1]
            canvas.data[y:y1][m] = band[m]
        if delay:
            time.sleep(delay)


def reveal_wipe(
    canvas: Canvas,
    target: np.ndarray,
    mask: np.ndarray | None = None,
    step: int = 2,
    delay: float = 0.010,
) -> None:
    """Проявить слой слева направо: полезно, когда сверху вниз уже приелось."""
    w = canvas.width
    for x in range(0, w, step):
        x1 = min(w, x + step)
        band = target[:, x:x1]
        if mask is None:
            canvas.data[:, x:x1] = band
        else:
            m = mask[:, x:x1]
            canvas.data[:, x:x1][m] = band[m]
        if delay:
            time.sleep(delay)


def reveal_radial(
    canvas: Canvas,
    target: np.ndarray,
    mask: np.ndarray | None = None,
    rings: int = 28,
    delay: float = 0.016,
) -> None:
    """Проявить слой кольцами от центра: под шар подходит лучше полос."""
    cy, cx = (canvas.height - 1) / 2.0, (canvas.width - 1) / 2.0
    yy, xx = np.mgrid[0 : canvas.height, 0 : canvas.width]
    d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    dmax = d.max()
    for i in range(1, rings + 1):
        ring = d <= dmax * i / rings
        sel = ring if mask is None else (ring & mask)
        canvas.data[sel] = target[sel]
        if delay:
            time.sleep(delay)


def dissolve(
    canvas: Canvas,
    target: np.ndarray,
    mask: np.ndarray | None = None,
    passes: int = 14,
    delay: float = 0.018,
    seed: int = 0,
) -> None:
    """Проявить слой случайными пикселями. Годится для звёздного неба
    и для замены одного покрытия другим без резкого щелчка."""
    rng = np.random.default_rng(seed)
    order = rng.random(canvas.data.shape)
    for i in range(1, passes + 1):
        sel = order <= i / passes
        if mask is not None:
            sel &= mask
        canvas.data[sel] = target[sel]
        if delay:
            time.sleep(delay)
