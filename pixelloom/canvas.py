"""Холст: сетка индексов палитры, которая умеет рассказывать о себе.

Каждая ячейка хранит индекс палитры, -1 означает прозрачность. Numpy внутри,
чтобы кисти по всему полю считались мгновенно.

Отличие от обычного массива одно: холст замечает запись пикселей и зовёт
наблюдателя. На этом держится живой просмотр — ни один пресет не знает, что
за ним смотрят, он просто рисует.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from .palette import Palette

TRANSPARENT = -1


def _root(array: np.ndarray) -> np.ndarray:
    """Владелец памяти: срез ведёт к тому же буферу, вычисление — к своему."""
    while getattr(array, "base", None) is not None:
        array = array.base
    return array


class _Watched(np.ndarray):
    """Массив, который сообщает холсту о каждой записи.

    Уведомляют только сам буфер холста и его срезы. Результаты вычислений
    (сравнения, np.roll, копии) numpy тоже отдаёт этим классом, но они пишут
    в свою память, поэтому ссылку на холст не наследуют: признак — общий
    буфер, а не тип.
    """

    _canvas = None

    def __array_finalize__(self, obj):
        if obj is None:
            return
        owner = getattr(obj, "_canvas", None)
        buf = getattr(owner, "_data", None) if owner is not None else None
        self._canvas = owner if buf is not None and _root(self) is _root(buf) else None

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        c = self._canvas
        if c is not None:
            c._touch()


class Canvas:
    """Прямоугольник пикселей."""

    def __init__(self, width: int, height: int, palette: Palette):
        self.width = width
        self.height = height
        self.palette = palette
        self._watcher: Callable[["Canvas"], None] | None = None
        self._muted = False
        self.data = np.full((height, width), TRANSPARENT, dtype=np.int16)

    # --- наблюдение ---

    @property
    def data(self) -> np.ndarray:
        return self._data

    @data.setter
    def data(self, array: np.ndarray) -> None:
        arr = np.asarray(array, dtype=np.int16).view(_Watched)
        arr._canvas = self
        self._data = arr
        self._touch()

    def watch(self, watcher: Callable[["Canvas"], None] | None) -> None:
        """Назначить наблюдателя. Он получает холст после каждой записи."""
        self._watcher = watcher

    def mute(self, on: bool = True) -> None:
        """Временно отключить уведомления: нужно наблюдателю, когда он сам
        читает холст, и подготовительным расчётам, которые показывать нечего."""
        self._muted = on

    def _touch(self) -> None:
        w = self._watcher
        if w is not None and not self._muted:
            w(self)

    # --- доступ ---

    def get(self, x: int, y: int) -> int:
        if not self.inside(x, y):
            return TRANSPARENT
        return int(self._data[y, x])

    def set(self, x: int, y: int, index: int) -> None:
        if self.inside(x, y):
            self._data[y, x] = index

    def inside(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def clear(self, index: int = TRANSPARENT) -> None:
        self._data[:, :] = index

    def copy(self) -> "Canvas":
        c = Canvas(self.width, self.height, self.palette)
        c.data = np.asarray(self._data).copy()
        return c

    # --- преобразование ---

    def to_rgba(self) -> np.ndarray:
        """Массив (h, w, 4) uint8 для сохранения картинкой."""
        lut = np.zeros((len(self.palette) + 1, 4), dtype=np.uint8)
        for i, (r, g, b) in enumerate(self.palette.colors):
            lut[i] = (r, g, b, 255)
        lut[len(self.palette)] = (0, 0, 0, 0)  # слот прозрачности
        idx = np.where(self._data < 0, len(self.palette), self._data)
        return lut[idx]

    def paste(self, other: "Canvas", x: int = 0, y: int = 0) -> None:
        """Наложить другой холст, прозрачные пиксели пропустить."""
        h, w = other.data.shape
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(self.width, x + w), min(self.height, y + h)
        if x0 >= x1 or y0 >= y1:
            return
        src = np.asarray(other.data)[y0 - y : y1 - y, x0 - x : x1 - x]
        dst = np.asarray(self._data)[y0:y1, x0:x1]
        mask = src != TRANSPARENT
        dst[mask] = src[mask]
        self._touch()

    def flip_h(self) -> "Canvas":
        c = self.copy()
        c.data = np.fliplr(np.asarray(self._data)).copy()
        return c

    def bbox(self) -> tuple[int, int, int, int] | None:
        """Границы непрозрачной части: (x0, y0, x1, y1), правый край не входит."""
        ys, xs = np.where(np.asarray(self._data) != TRANSPARENT)
        if len(xs) == 0:
            return None
        return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
