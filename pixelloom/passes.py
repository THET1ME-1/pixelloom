"""Конвейер рисования: палитра, силуэт, заливки, тени, акценты.

Порядок не украшение, а защита от переделок. Работа идёт проходами, и каждый
следующий не начинается, пока предыдущий не сдан:

0. **Палитра.** Все цвета работы объявляются заранее, шкалами по материалам.
   Цвет, придуманный на ходу, ломает единство: половина спрайта живёт в одной
   гамме, половина в другой.
1. **Силуэт.** Два цвета, фон и объект. Пока фигура не узнаётся чёрным пятном,
   дальше идти нельзя — детали силуэт не спасут, а спрячут.
2. **Крупные заливки.** Волосы, кожа, одежда одним тоном каждая. Здесь видно,
   верно ли поделены массы.
3. **Тени.** Один-два тона, свет строго с одной стороны. Тень с обеих сторон
   сразу убивает объём.
4. **Контур и акценты.** Выборочный контур по теневой стороне и одна яркая
   точка на всю работу.

`Artwork` ведёт этот порядок, а `gate` на каждом шаге мерит результат
критиком: сорванный силуэт виден сразу, а не через сорок правок.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import critique, shading
from .canvas import TRANSPARENT, Canvas
from .palette import Palette

__all__ = ["Artwork", "PassError"]


class PassError(RuntimeError):
    """Проход не сдан: работа не пускается дальше."""


@dataclass
class Artwork:
    """Работа, которая рисуется по проходам.

    Держит палитру, маски материалов и отчёт по каждому шагу, поэтому в конце
    видно, где именно просела работа.
    """

    width: int
    height: int
    name: str = "работа"
    strict: bool = False               # ронять на несданном проходе или ворчать
    palette: Palette = field(init=False)
    canvas: Canvas = field(init=False)
    ramps: dict[str, list[int]] = field(default_factory=dict, init=False)
    masks: dict[str, np.ndarray] = field(default_factory=dict, init=False)
    log: list[str] = field(default_factory=list, init=False)
    _stage: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.palette = Palette.from_hex(self.name, ["#000000"], labels=["ink"])
        self.canvas = Canvas(self.width, self.height, self.palette)

    # ── 0. палитра ────────────────────────────────────────────────────────

    def palette_from(self, materials: dict[str, str], steps: int = 5,
                     ink: str = "#1a1420") -> "Artwork":
        """Объявить материалы: имя → базовый цвет. Шкалы строятся сразу.

        Ограничение по числу цветов здесь и живёт: пять ступеней на материал
        и общий тёмный тон на контур. Больше — и работа расползается по цвету.
        """
        self.palette.colors[0] = tuple(int(ink.lstrip("#")[i : i + 2], 16) for i in (0, 2, 4))
        for mat, base in materials.items():
            self.ramps[mat] = shading.build_ramp(self.palette, base, mat, steps=steps)
        self.log.append(f"палитра: {len(materials)} материалов, {len(self.palette)} цветов")
        self._stage = 1
        return self

    # ── 1. силуэт ─────────────────────────────────────────────────────────

    def silhouette(self, mask: np.ndarray, min_fill: float = 0.10,
                   max_fill: float = 0.72) -> "Artwork":
        """Заливка силуэта одним тоном и проверка читаемости.

        Три вещи ловятся здесь и больше нигде: фигура не занимает кадр целиком
        и не теряется в нём, не упирается в края, не рассыпается на куски.
        """
        self._require(1, "силуэт")
        self.canvas.data[:, :] = TRANSPARENT
        self.canvas.data[mask] = 0
        self.masks["silhouette"] = mask.copy()

        fill = mask.sum() / (self.width * self.height)
        border = np.zeros_like(mask)
        border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True
        bleed = (mask & border).sum() / max(1, border.sum())
        parts = self._islands(mask)

        problems = []
        if fill < min_fill:
            problems.append(f"фигура мелкая: {fill:.0%} кадра")
        if fill > max_fill:
            problems.append(f"фигура распирает кадр: {fill:.0%}")
        if bleed > 0.18:
            problems.append(f"силуэт упирается в край: {bleed:.0%} рамки")
        if parts > 3:
            problems.append(f"силуэт рассыпан на {parts} частей")
        self._gate("силуэт", problems)
        self.log.append(f"силуэт: {fill:.0%} кадра, кусков {parts}")
        self._stage = 2
        return self

    # ── 2. крупные заливки ────────────────────────────────────────────────

    def blocks(self, blocks: dict[str, np.ndarray], tone: int = 2) -> "Artwork":
        """Разложить силуэт по материалам одним тоном на каждый.

        [tone] — какая ступень шкалы берётся за базу. Тени и света придут
        следующим проходом, сейчас важны только массы.
        """
        self._require(2, "заливки")
        covered = np.zeros((self.height, self.width), dtype=bool)
        for mat, mask in blocks.items():
            if mat not in self.ramps:
                raise PassError(f"материал «{mat}» не объявлен в палитре")
            ramp = self.ramps[mat]
            self.canvas.data[mask] = ramp[min(tone, len(ramp) - 1)]
            self.masks[mat] = mask.copy()
            covered |= mask

        silhouette = self.masks.get("silhouette")
        problems = []
        if silhouette is not None:
            missed = int((silhouette & ~covered).sum())
            if missed > silhouette.sum() * 0.04:
                problems.append(f"не закрашено {missed} пикселей силуэта")
        shares = {m: float(mask.sum()) for m, mask in blocks.items()}
        total = sum(shares.values()) or 1.0
        biggest = max(shares.values()) / total
        if biggest > 0.82 and len(blocks) > 1:
            problems.append("один материал занимает почти всё: массы не поделены")
        self._gate("заливки", problems)
        self.log.append("заливки: " + ", ".join(
            f"{m} {v / total:.0%}" for m, v in sorted(shares.items(), key=lambda kv: -kv[1])))
        self._stage = 3
        return self

    # ── 3. тени ───────────────────────────────────────────────────────────

    def shade(self, light: tuple[float, float, float] = (-0.6, -0.7, 0.45),
              per_material: dict[str, dict] | None = None,
              default: dict | None = None) -> "Artwork":
        """Свет с одной стороны на каждый материал по его шкале."""
        self._require(3, "тени")
        per_material = per_material or {}
        base = dict(ambient=0.28, bounce=0.26, dither=0.14, roundness=0.8)
        base.update(default or {})
        for mat, mask in self.masks.items():
            if mat == "silhouette" or mat not in self.ramps:
                continue
            kw = dict(base)
            kw.update(per_material.get(mat, {}))
            shading.shade_mask(self.canvas, mask, self.ramps[mat], light=light, **kw)

        report = self._measure()
        problems = []
        if report.contrast < 0.35:
            problems.append(f"объём не вылеплен: контраст {report.contrast:.2f}")
        if report.dither_noise > 0.14:
            problems.append(f"дизеринг шумит: {report.dither_noise:.0%} площади")
        self._gate("тени", problems)
        self.log.append(f"тени: контраст {report.contrast:.2f}, "
                        f"полосы {report.banding:.0%}")
        self._stage = 4
        return self

    # ── 4. контур и акценты ───────────────────────────────────────────────

    def outline(self, ink: int = 0, light: tuple[float, float] = (-1, -1)) -> "Artwork":
        """Выборочный контур: тёмный со стороны тени, пропуск со стороны света."""
        self._require(4, "контур")
        mask = self.masks.get("silhouette")
        if mask is None:
            raise PassError("контур без силуэта")
        inner = mask.copy()
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            inner &= np.roll(np.roll(mask, dy, axis=0), dx, axis=1)
        edge = mask & ~inner
        lx, ly = light
        shaded = np.zeros_like(mask)
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            if (dx == -int(np.sign(lx))) or (dy == -int(np.sign(ly))):
                continue                      # со стороны света контур не кладём
            shaded |= edge & ~np.roll(np.roll(mask, dy, axis=0), dx, axis=1)
        self.canvas.data[shaded] = ink
        self.log.append(f"контур: {int(shaded.sum())} пикселей по теневой стороне")
        return self

    def accent(self, mask: np.ndarray, index: int) -> "Artwork":
        """Акцент: самая яркая точка работы. Одна на весь кадр."""
        self._require(4, "акценты")
        self.canvas.data[mask] = index
        self.log.append(f"акцент: {int(mask.sum())} пикселей")
        return self

    # ── итог ──────────────────────────────────────────────────────────────

    def finish(self, clean: bool = True) -> Canvas:
        if clean:
            fixed = shading.clean_stray(self.canvas, rounds=2)
            if fixed:
                self.log.append(f"чистка: убрано {fixed} одиночных пикселей")
        report = self._measure()
        self.log.append(report.table(self.name))
        return self.canvas

    def report(self) -> critique.Report:
        return self._measure()

    def story(self) -> str:
        return "\n".join(self.log)

    # ── внутреннее ────────────────────────────────────────────────────────

    def _require(self, stage: int, what: str) -> None:
        if self._stage < stage:
            raise PassError(
                f"«{what}» раньше срока: сперва "
                f"{['палитра', 'силуэт', 'заливки', 'тени'][self._stage]}"
            )

    def _gate(self, step: str, problems: list[str]) -> None:
        if not problems:
            return
        message = f"проход «{step}» не сдан: " + "; ".join(problems)
        if self.strict:
            raise PassError(message)
        self.log.append("! " + message)

    def _measure(self) -> critique.Report:
        rgba = self.canvas.to_rgba()
        return critique.measure(rgba[..., :3], rgba[..., 3])

    def _islands(self, mask: np.ndarray) -> int:
        """Сколько несвязных кусков в силуэте."""
        seen = np.zeros_like(mask)
        count = 0
        ys, xs = np.nonzero(mask)
        for y0, x0 in zip(ys.tolist(), xs.tolist()):
            if seen[y0, x0]:
                continue
            count += 1
            stack = [(y0, x0)]
            seen[y0, x0] = True
            while stack:
                y, x = stack.pop()
                for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    ny, nx = y + dy, x + dx
                    if (0 <= ny < mask.shape[0] and 0 <= nx < mask.shape[1]
                            and mask[ny, nx] and not seen[ny, nx]):
                        seen[ny, nx] = True
                        stack.append((ny, nx))
        return count
