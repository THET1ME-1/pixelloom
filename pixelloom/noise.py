"""Шум для фактуры поверхности. Детерминирован по seed."""

from __future__ import annotations

import numpy as np


def _smootherstep(t: np.ndarray) -> np.ndarray:
    return t * t * t * (t * (t * 6 - 15) + 10)


def value_noise(shape: tuple[int, int], cells: int, seed: int) -> np.ndarray:
    """Гладкий шум 0..1 на решётке cells x cells."""
    h, w = shape
    rng = np.random.default_rng(seed)
    grid = rng.random((cells + 1, cells + 1))
    grid[-1, :] = grid[0, :]
    grid[:, -1] = grid[:, 0]

    gy = np.linspace(0, cells, h, endpoint=False)
    gx = np.linspace(0, cells, w, endpoint=False)
    y0 = np.floor(gy).astype(int)
    x0 = np.floor(gx).astype(int)
    fy = _smootherstep(gy - y0)[:, None]
    fx = _smootherstep(gx - x0)[None, :]

    y1 = np.minimum(y0 + 1, cells)
    x1 = np.minimum(x0 + 1, cells)
    v00 = grid[np.ix_(y0, x0)]
    v01 = grid[np.ix_(y0, x1)]
    v10 = grid[np.ix_(y1, x0)]
    v11 = grid[np.ix_(y1, x1)]
    top = v00 * (1 - fx) + v01 * fx
    bot = v10 * (1 - fx) + v11 * fx
    return top * (1 - fy) + bot * fy


def fbm(
    shape: tuple[int, int],
    cells: int = 4,
    octaves: int = 5,
    persistence: float = 0.5,
    lacunarity: float = 2.0,
    seed: int = 0,
) -> np.ndarray:
    """Сумма октав шума. Даёт рельеф с крупными пятнами и мелкой крошкой."""
    total = np.zeros(shape)
    amp, freq, norm = 1.0, float(cells), 0.0
    for o in range(octaves):
        total += value_noise(shape, max(1, int(round(freq))), seed + o * 101) * amp
        norm += amp
        amp *= persistence
        freq *= lacunarity
    return total / norm


def ridged(shape: tuple[int, int], cells: int = 4, octaves: int = 5, seed: int = 0) -> np.ndarray:
    """Хребтовой вариант: острые гребни вместо мягких холмов."""
    n = fbm(shape, cells, octaves, seed=seed)
    return 1.0 - np.abs(n * 2 - 1)
