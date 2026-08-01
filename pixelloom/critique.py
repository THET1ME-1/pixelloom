"""Критик: измеряет качество пиксельной работы вместо оценки на глаз.

Каждая метрика отвечает на претензию, которую иначе приходится ловить
взглядом и по десять раз переделывать:

* **stray** — одиночные пиксели без соседа своего цвета. Их избыток и есть то
  самое ощущение «грязного» спрайта.
* **banding** — длинные ровные полосы одного тона. Градиент, разложенный
  ступенями без разрыва, читается как полосатый флаг.
* **dither_noise** — доля шахматного чередования. Дизеринг хорош на стыке
  ступеней и превращается в шум, когда покрывает всю площадь.
* **tones** — сколько цветов реально работает и не свалена ли половина
  площади в один тон (признак плоской заливки).
* **contrast** — разброс яркости. Низкий разброс означает, что объём не
  вылеплен, а сцена сливается.
* **focus** — есть ли выраженный центр внимания: зона, где контраст заметно
  выше среднего по кадру.
* **edge_bleed** — упирается ли рисунок в границы кадра.
* **readability** — что остаётся от работы при уменьшении вдвое. Спрайт,
  который перестаёт читаться, в интерфейсе не живёт.

`detect_scale` находит, во сколько раз картинка увеличена, чтобы мерить
чужие работы в их родном разрешении, а не по растянутым блокам.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["Report", "Check", "measure", "detect_scale", "downscale", "compare",
           "inspect_volume"]


@dataclass
class Report:
    """Замеры одной работы. Значения в долях, если не сказано иначе."""

    size: tuple[int, int]
    colors: int
    stray: float
    banding: float
    dither_noise: float
    top_tone: float
    contrast: float
    focus: float
    edge_bleed: float
    readability: float
    notes: list[str] = field(default_factory=list)

    def table(self, name: str = "") -> str:
        head = f"{name}  {self.size[0]}×{self.size[1]}, цветов {self.colors}"
        rows = [
            f"  одиночные пиксели   {self.stray * 100:5.1f}%",
            f"  полосы              {self.banding * 100:5.1f}%",
            f"  шум дизеринга       {self.dither_noise * 100:5.1f}%",
            f"  главный тон         {self.top_tone * 100:5.1f}%",
            f"  контраст            {self.contrast:5.2f}",
            f"  центр внимания      {self.focus:5.2f}",
            f"  выход за край       {self.edge_bleed * 100:5.1f}%",
            f"  читается вдвое      {self.readability * 100:5.1f}%",
        ]
        if self.notes:
            rows.append("  замечания: " + "; ".join(self.notes))
        return "\n".join([head, *rows])


def _luma(rgb: np.ndarray) -> np.ndarray:
    return (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]) / 255.0


def detect_scale(rgb: np.ndarray, limit: int = 16) -> int:
    """Во сколько раз картинка увеличена: ищем самый крупный шаг сетки, при
    котором блоки остаются одноцветными."""
    h, w = rgb.shape[:2]
    best = 1
    for k in range(2, limit + 1):
        if h % k or w % k:
            continue
        blocks = rgb[: h // k * k, : w // k * k].reshape(h // k, k, w // k, k, -1)
        same = (blocks == blocks[:, :1, :, :1, :]).all()
        if same:
            best = k
    return best


def downscale(rgb: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return rgb
    h, w = rgb.shape[:2]
    return rgb[: h // factor * factor : factor, : w // factor * factor : factor]


def measure(rgb: np.ndarray, alpha: np.ndarray | None = None) -> Report:
    """Замерить работу. [rgb] — массив (h, w, 3) uint8 в родном разрешении."""
    h, w = rgb.shape[:2]
    flat = rgb.reshape(-1, 3)
    solid = np.ones((h, w), dtype=bool) if alpha is None else alpha > 0

    uniq, counts = np.unique(flat[solid.reshape(-1)], axis=0, return_counts=True)
    colors = len(uniq)
    top_tone = float(counts.max() / max(1, counts.sum()))

    rgb32 = rgb.astype(np.int32)
    key = rgb32[..., 0] * 65536 + rgb32[..., 1] * 256 + rgb32[..., 2]
    same = np.zeros((h, w), dtype=np.int8)
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        same += (np.roll(np.roll(key, dy, axis=0), dx, axis=1) == key).astype(np.int8)
    stray = float((solid & (same == 0)).sum() / max(1, solid.sum()))

    # полосы: длина максимального одноцветного отрезка в строке
    runs = np.zeros(h, dtype=np.float32)
    for y in range(h):
        row = key[y]
        edges = np.flatnonzero(np.diff(row)) + 1
        lengths = np.diff(np.concatenate(([0], edges, [w])))
        runs[y] = lengths.max() / w if len(lengths) else 1.0
    banding = float(np.mean(runs > 0.55))

    # шахматка: пиксель отличается от соседей по горизонтали, но равен через один
    left = np.roll(key, 1, axis=1)
    right = np.roll(key, -1, axis=1)
    two = np.roll(key, 2, axis=1)
    checker = (key != left) & (key != right) & (key == two)
    dither_noise = float((checker & solid).sum() / max(1, solid.sum()))

    lum = _luma(rgb)
    vals = lum[solid]
    contrast = float(vals.max() - vals.min()) if vals.size else 0.0

    # центр внимания: максимум локального контраста против среднего
    pad = np.pad(lum, 3, mode="edge")
    local = np.zeros_like(lum)
    for dy in (-3, 0, 3):
        for dx in (-3, 0, 3):
            local = np.maximum(local, np.abs(pad[3 + dy : 3 + dy + h, 3 + dx : 3 + dx + w] - lum))
    focus = float(local.max() / max(1e-6, local.mean() + 1e-6)) / 10.0

    border = np.zeros((h, w), dtype=bool)
    border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True
    edge_bleed = float((solid & border).sum() / max(1, border.sum()))

    small = rgb[::2, ::2]
    big = np.repeat(np.repeat(small, 2, axis=0), 2, axis=1)[:h, :w]
    readability = float((np.abs(big.astype(int) - rgb.astype(int)).max(axis=2) < 24).mean())

    notes: list[str] = []
    if stray > 0.03:
        notes.append("много одиночных пикселей, нужна чистка")
    if banding > 0.35:
        notes.append("полосит: градиент идёт ровными лентами")
    if dither_noise > 0.12:
        notes.append("дизеринг покрывает лишнее, съедает чистоту заливок")
    if top_tone > 0.45:
        notes.append("почти половина площади в одном тоне — плоско")
    if contrast < 0.45:
        notes.append("низкий контраст, объём не вылеплен")
    if focus < 0.55:
        notes.append("нет выраженного центра внимания")
    if edge_bleed > 0.25:
        notes.append("рисунок упирается в края кадра")
    if readability < 0.55:
        notes.append("при уменьшении вдвое разваливается")

    return Report(
        size=(w, h), colors=colors, stray=stray, banding=banding,
        dither_noise=dither_noise, top_tone=top_tone, contrast=contrast,
        focus=focus, edge_bleed=edge_bleed, readability=readability, notes=notes,
    )


def compare(reports: dict[str, Report]) -> str:
    """Свести замеры в таблицу: свои работы против эталонов."""
    cols = ["цвета", "одиноч", "полосы", "дизер", "глав.тон", "контраст", "фокус", "мелко"]
    width = max(len(n) for n in reports) + 2
    out = [" " * width + "".join(f"{c:>9}" for c in cols)]
    for name, r in reports.items():
        out.append(
            f"{name:<{width}}"
            f"{r.colors:>9}"
            f"{r.stray * 100:>8.1f}%"
            f"{r.banding * 100:>8.1f}%"
            f"{r.dither_noise * 100:>8.1f}%"
            f"{r.top_tone * 100:>8.1f}%"
            f"{r.contrast:>9.2f}"
            f"{r.focus:>9.2f}"
            f"{r.readability * 100:>8.1f}%"
        )
    return "\n".join(out)


# ── проверка объёма ──────────────────────────────────────────────────────

@dataclass
class Check:
    """Один пункт чеклиста: что проверяли, сдано ли, чем измерено."""

    name: str
    passed: bool
    detail: str

    def __str__(self) -> str:
        return f"[{'✓' if self.passed else '×'}] {self.name}: {self.detail}"


def _edge_distance(mask: np.ndarray) -> np.ndarray:
    dist = np.zeros(mask.shape, dtype=np.float32)
    cur = mask.copy()
    for step in range(1, 64):
        if not cur.any():
            break
        dist[cur] = step
        nxt = cur.copy()
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nxt &= np.roll(np.roll(cur, dy, axis=0), dx, axis=1)
        nxt[0, :] = nxt[-1, :] = False
        nxt[:, 0] = nxt[:, -1] = False
        cur = nxt
    return dist


def inspect_volume(
    rgb: np.ndarray,
    alpha: np.ndarray | None = None,
    light: tuple[float, float] = (-0.6, -0.8),
) -> list[Check]:
    """Чеклист объёма: шесть вопросов, на которые смотрят глазами и врут.

    Проверяется ровно то, без чего форма читается как аппликация: единое
    направление света, терминатор внутри формы, рефлекс на теневом крае,
    окклюзия в стыках, узнаваемый силуэт и отсутствие затемнения по кругу.
    """
    solid = np.ones(rgb.shape[:2], dtype=bool) if alpha is None else alpha > 0
    lum = _luma(rgb)
    checks: list[Check] = []
    if not solid.any():
        return [Check("силуэт", False, "пусто")]

    dist = _edge_distance(solid)
    inside = lum[solid]
    lo, hi = np.percentile(inside, 12), np.percentile(inside, 88)

    # 1. одно ли направление света
    ys, xs = np.nonzero(solid & (lum >= hi))
    ds, dxs = np.nonzero(solid & (lum <= lo))
    if len(xs) and len(dxs):
        vx = float(dxs.mean() - xs.mean())
        vy = float(ds.mean() - ys.mean())
        norm = max(1e-6, (vx * vx + vy * vy) ** 0.5)
        lx, ly = light
        lnorm = max(1e-6, (lx * lx + ly * ly) ** 0.5)
        dot = (vx / norm) * (-lx / lnorm) + (vy / norm) * (-ly / lnorm)
        checks.append(Check("свет с одной стороны", dot > 0.45,
                            f"тень уходит от света на {dot:+.2f} (нужно > 0.45)"))
    else:
        checks.append(Check("свет с одной стороны", False, "тонов слишком мало"))

    # 2. самая тёмная точка: терминатор или кромка
    darkest = solid & (lum <= lo)
    mean_depth = float(dist[darkest].mean()) if darkest.any() else 0.0
    checks.append(Check("тёмное на терминаторе", mean_depth > 1.6,
                        f"тёмные пиксели в среднем на {mean_depth:.1f} px от края"))

    # 3. рефлекс на теневом крае
    lx, ly = light
    step_x, step_y = -int(np.sign(lx)) or 1, -int(np.sign(ly)) or 1
    shadow_edge = solid & (dist <= 1.5) & (lum <= np.percentile(inside, 45))
    innerward = np.roll(np.roll(lum, -step_y * 2, axis=0), -step_x * 2, axis=1)
    brighter = shadow_edge & (lum > innerward + 0.02)
    share = float(brighter.sum() / max(1, shadow_edge.sum()))
    checks.append(Check("рефлекс на теневом крае", share > 0.12,
                        f"{share:.0%} теневой кромки светлее внутренней зоны"))

    # 4. окклюзия в стыках
    darkest_tone = solid & (lum <= np.percentile(inside, 4))
    deep = darkest_tone & (dist >= 2)
    checks.append(Check("окклюзия в стыках", deep.sum() >= max(4, solid.sum() * 0.004),
                        f"{int(deep.sum())} пикселей самой тёмной линии внутри формы"))

    # 5. читается ли силуэт
    fill = float(solid.sum() / solid.size)
    holes = int((~solid & (_edge_distance(~solid) > 2)).sum())
    checks.append(Check("силуэт узнаётся", 0.08 < fill < 0.75,
                        f"{fill:.0%} кадра, внешних пустот {holes}"))

    # 6. нет ли затемнения по всем краям сразу
    edge = solid & (dist <= 1.5)
    core = solid & (dist > 2.5)
    if edge.any() and core.any():
        drop = float(lum[core].mean() - lum[edge].mean())
        checks.append(Check("нет затемнения по кругу", drop < 0.16,
                            f"кромка темнее середины на {drop:+.2f} (порог 0.16)"))
    return checks
