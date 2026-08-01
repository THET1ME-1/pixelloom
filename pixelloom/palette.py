"""Палитры: индексированный цвет, ramp-шкалы, поиск ближайшего."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


def hex_to_rgb(value: str) -> tuple[int, int, int]:
    v = value.lstrip("#")
    if len(v) == 3:
        v = "".join(c * 2 for c in v)
    return int(v[0:2], 16), int(v[2:4], 16), int(v[4:6], 16)


def rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    return "#{:02X}{:02X}{:02X}".format(*rgb)


@dataclass
class Palette:
    """Именованный набор цветов. Индекс -1 всегда означает прозрачность."""

    name: str
    colors: list[tuple[int, int, int]] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)
    ramps: dict[str, list[int]] = field(default_factory=dict)

    @classmethod
    def from_hex(
        cls,
        name: str,
        hexes: list[str],
        labels: list[str] | None = None,
        ramps: dict[str, list[int]] | None = None,
    ) -> "Palette":
        colors = [hex_to_rgb(h) for h in hexes]
        return cls(
            name=name,
            colors=colors,
            labels=labels or [f"c{i}" for i in range(len(colors))],
            ramps=ramps or {},
        )

    def __len__(self) -> int:
        return len(self.colors)

    def rgb(self, index: int) -> tuple[int, int, int]:
        return self.colors[index]

    def index_of(self, label: str) -> int:
        return self.labels.index(label)

    def ramp(self, name: str) -> list[int]:
        """Шкала светлоты: индексы от самого тёмного к самому светлому."""
        return self.ramps[name]

    def nearest(self, rgb: tuple[int, int, int], among: list[int] | None = None) -> int:
        """Ближайший индекс палитры. Расстояние считаем во взвешенном RGB,
        так глаз воспринимает разницу точнее, чем в сыром евклиде."""
        pool = among if among is not None else range(len(self.colors))
        r, g, b = rgb
        best, best_d = 0, float("inf")
        for i in pool:
            cr, cg, cb = self.colors[i]
            rmean = (cr + r) / 2
            dr, dg, db = cr - r, cg - g, cb - b
            d = (2 + rmean / 256) * dr * dr + 4 * dg * dg + (2 + (255 - rmean) / 256) * db * db
            if d < best_d:
                best, best_d = i, d
        return best

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "colors": [rgb_to_hex(c) for c in self.colors],
            "labels": self.labels,
            "ramps": self.ramps,
        }

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2))
        return path

    @classmethod
    def load(cls, path: str | Path) -> "Palette":
        data = json.loads(Path(path).read_text())
        return cls.from_hex(data["name"], data["colors"], data.get("labels"), data.get("ramps"))
