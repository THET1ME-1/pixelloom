"""Мост к GridFab: спрайт как текст туда и обратно.

GridFab держит работу в двух текстовых файлах — `grid.txt` с сеткой алиасов и
`palette.txt` с цветами. Формат ценен тем, что рисунок можно править руками в
редакторе и читать построчно кодом: правка человека видна как изменение
символов, а не как «стало иначе».

    canvas = load("sprite/")          # текст → холст
    save(canvas, "sprite/", names)    # холст → текст
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .canvas import TRANSPARENT, Canvas
from .palette import Palette, hex_to_rgb, rgb_to_hex

__all__ = ["load", "save", "load_palette"]


def load_palette(path: str | Path, name: str = "gridfab") -> tuple[Palette, dict[str, int]]:
    """Прочитать palette.txt. Возвращает палитру и карту алиас → индекс."""
    hexes: list[str] = []
    labels: list[str] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        alias, value = (part.strip() for part in line.split("=", 1))
        labels.append(alias)
        hexes.append(value)
    palette = Palette.from_hex(name, hexes, labels=labels)
    return palette, {alias: i for i, alias in enumerate(labels)}


def load(directory: str | Path, name: str = "gridfab") -> Canvas:
    """Прочитать спрайт целиком: grid.txt плюс palette.txt."""
    directory = Path(directory)
    palette, alias = load_palette(directory / "palette.txt", name)
    rows = [
        line.split()
        for line in (directory / "grid.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    height = len(rows)
    width = max(len(r) for r in rows)
    canvas = Canvas(width, height, palette)
    data = np.full((height, width), TRANSPARENT, dtype=np.int16)
    for y, row in enumerate(rows):
        for x, cell in enumerate(row):
            if cell == "." or cell == "..":
                continue
            if cell.startswith("#"):            # цвет прямо в сетке
                rgb = hex_to_rgb(cell)
                if rgb not in palette.colors:
                    palette.colors.append(rgb)
                    palette.labels.append(f"c{len(palette.colors)}")
                data[y, x] = palette.colors.index(rgb)
                continue
            idx = alias.get(cell)
            if idx is not None:
                data[y, x] = idx
    canvas.data = data
    return canvas


def save(canvas: Canvas, directory: str | Path, header: str = "") -> Path:
    """Записать холст обратно в текст, чтобы правку можно было доделать руками."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    lines = ["# Palette: ALIAS=#RRGGBB"]
    if header:
        lines += ["# " + h for h in header.splitlines()]
    used = sorted({int(v) for v in np.unique(np.asarray(canvas.data)) if v >= 0})
    alias_of: dict[int, str] = {}
    for i in used:
        label = canvas.palette.labels[i] if i < len(canvas.palette.labels) else f"c{i}"
        alias = "".join(ch for ch in label if ch.isalnum())[:2].upper() or f"C{i}"
        base, n = alias, 1
        while alias in alias_of.values():       # алиасы в GridFab уникальны
            n += 1
            alias = f"{base[0]}{n}"
        alias_of[i] = alias
        lines.append(f"{alias}={rgb_to_hex(canvas.palette.colors[i])}")
    (directory / "palette.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    grid = []
    for row in np.asarray(canvas.data):
        grid.append(" ".join(f"{alias_of.get(int(v), '.'):>2}" if v >= 0 else " ."
                             for v in row))
    (directory / "grid.txt").write_text("\n".join(grid) + "\n", encoding="utf-8")
    return directory
