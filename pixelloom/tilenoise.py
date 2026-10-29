"""Шум для тайлов и блоков: период в ребро клетки, без шва на стыке соседей.

Родом из meadow-blocks. Два урока, которые здесь зашиты:

- Шум с сидом блока на кромке не совпадает с шумом соседа — по полю травы
  идёт сетка швов. seamless_noise сводит шум варианта к общему шуму материала
  у краёв квадрата: середины блоков разные, кромки одинаковые (T12).
- У value noise ноль наклона в узлах решётки, и линии уровня (блики воды,
  контуры пятен) ложатся вдоль кромок блока ромбической сеткой. Две октавы со
  сдвигом узлов на дробную долю ячейки эту сетку разбивают (T14).
"""
from __future__ import annotations

import numpy as np

EDGE = 16.0


def periodic_noise(a, b, cells: int, seed: int, edge: float = EDGE) -> np.ndarray:
    """Гладкий шум 0..1 по решётке cells×cells на квадрате 0..edge, свёрнутый по краям."""
    rng = np.random.default_rng(seed)
    lattice = rng.random((cells, cells))
    a = np.asarray(a, dtype=float) / edge * cells
    b = np.asarray(b, dtype=float) / edge * cells
    a0 = np.floor(a).astype(int)
    b0 = np.floor(b).astype(int)
    fa = a - a0
    fb = b - b0
    fa = fa * fa * (3 - 2 * fa)
    fb = fb * fb * (3 - 2 * fb)
    i0, i1 = a0 % cells, (a0 + 1) % cells
    j0, j1 = b0 % cells, (b0 + 1) % cells
    top = lattice[i0, j0] * (1 - fa) + lattice[i1, j0] * fa
    bottom = lattice[i0, j1] * (1 - fa) + lattice[i1, j1] * fa
    return top * (1 - fb) + bottom * fb


def seamless_noise(a, b, cells: int, shared_seed: int, variant_seed: int,
                   edge: float = EDGE) -> np.ndarray:
    """Шум варианта, который к краям квадрата сходится к общему для материала."""
    shared = periodic_noise(a, b, cells, shared_seed, edge)
    own = periodic_noise(a, b, cells, variant_seed, edge)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    window = np.sqrt(np.clip(np.sin(np.pi * a / edge) * np.sin(np.pi * b / edge), 0, 1))
    return shared * (1 - window) + own * window


def two_octave(a, b, seed: int, edge: float = EDGE) -> np.ndarray:
    """Две октавы со сдвигом узлов: линии уровня не ложатся сеткой по кромкам."""
    return (0.6 * periodic_noise(np.asarray(a) + 0.7, np.asarray(b) + 3.3, 4, seed, edge)
            + 0.4 * periodic_noise(np.asarray(a) + 5.1, np.asarray(b) + 1.1, 8, seed + 1, edge))


def noise_1d(a, cells: int, seed: int, edge: float = EDGE) -> np.ndarray:
    """Шум вдоль ребра с тем же периодом."""
    return periodic_noise(a, np.zeros_like(np.asarray(a, dtype=float)), cells, seed, edge)


def wrapped_distance(a, b, point_a, point_b, edge: float = EDGE):
    """Разница координат до точки с заворотом через край квадрата."""
    da = (np.asarray(a) - point_a + edge / 2) % edge - edge / 2
    db = (np.asarray(b) - point_b + edge / 2) % edge - edge / 2
    return da, db
