"""Объём как арифметика: форма заявляет нормали, тон считается сам.

Художественное суждение «здесь потемнее» здесь не участвует вовсе. Сцена
объявляет один вектор света, каждая часть сообщает, куда повёрнута её
поверхность, и освещённость выходит скалярным произведением. Ошибиться в
том, где тень, становится нечем.

Три вещи, без которых форма остаётся аппликацией, считаются тут же:

* **терминатор** — самый тёмный тон лежит на границе света и тени, ближе к
  середине формы, а не на кромке силуэта;
* **рефлекс** — на дальнем от света крае идёт полоса светлее тени: это свет,
  отражённый окружением. Один-два пикселя, и предмет отрывается от фона;
* **окклюзия** — там, где формы соприкасаются, ложится линия темнее всей
  шкалы. Она даёт глубины больше, чем всё остальное вместе.

Примитивы возвращают маску и поле нормалей, поэтому сфера остаётся сферой
независимо от того, что об этом думает тот, кто её поставил.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .canvas import Canvas

__all__ = [
    "Body", "sphere", "cylinder", "cone", "shell", "plane",
    "light_field", "render", "occlude", "Scene", "strand", "hair_mass",
]


@dataclass
class Body:
    """Часть формы: где она и куда повёрнута её поверхность."""

    mask: np.ndarray
    nx: np.ndarray
    ny: np.ndarray
    nz: np.ndarray
    material: str = ""
    depth: float = 0.0          # кто выше: больший перекрывает меньший

    def shifted(self, dx: int = 0, dy: int = 0) -> "Body":
        roll = lambda a: np.roll(np.roll(a, dy, axis=0), dx, axis=1)  # noqa: E731
        return Body(roll(self.mask), roll(self.nx), roll(self.ny), roll(self.nz),
                    self.material, self.depth)


def _grid(shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]]
    return yy.astype(np.float32), xx.astype(np.float32)


def _normalize(nx, ny, nz):
    length = np.sqrt(nx * nx + ny * ny + nz * nz)
    length[length == 0] = 1.0
    return nx / length, ny / length, nz / length


def sphere(shape, cx: float, cy: float, r: float, squash: float = 1.0,
           material: str = "", depth: float = 0.0) -> Body:
    """Шар. Нормаль в каждой точке смотрит из центра наружу."""
    yy, xx = _grid(shape)
    dx = (xx - cx) / r
    dy = (yy - cy) / (r * squash)
    d2 = dx * dx + dy * dy
    mask = d2 <= 1.0
    dz = np.sqrt(np.clip(1 - d2, 0, 1))
    nx, ny, nz = _normalize(dx * mask, dy * mask, dz * mask)
    return Body(mask, nx, ny, nz, material, depth)


def cylinder(shape, cx: float, cy: float, half_w: float, half_h: float,
             axis: str = "y", material: str = "", depth: float = 0.0) -> Body:
    """Труба. Вдоль оси нормаль не меняется, поперёк — как у половины круга."""
    yy, xx = _grid(shape)
    inside = (np.abs(xx - cx) <= half_w) & (np.abs(yy - cy) <= half_h)
    if axis == "y":
        t = np.clip((xx - cx) / max(half_w, 1e-6), -1, 1)
        nx, ny = t, np.zeros_like(t)
    else:
        t = np.clip((yy - cy) / max(half_h, 1e-6), -1, 1)
        nx, ny = np.zeros_like(t), t
    nz = np.sqrt(np.clip(1 - t * t, 0, 1))
    nx, ny, nz = _normalize(nx * inside, ny * inside, nz * inside)
    return Body(inside, nx, ny, nz, material, depth)


def cone(shape, cx: float, y0: float, y1: float, w0: float, w1: float,
         material: str = "", depth: float = 0.0) -> Body:
    """Юбка, ствол, конус. Поперёк круглая, вдоль расширяется."""
    yy, xx = _grid(shape)
    k = np.clip((yy - y0) / max(1.0, y1 - y0), 0, 1)
    half = w0 + (w1 - w0) * k
    mask = (np.abs(xx - cx) <= half) & (yy >= y0) & (yy <= y1)
    t = np.clip((xx - cx) / np.maximum(half, 1e-6), -1, 1)
    nx = t
    ny = np.full_like(t, -(w1 - w0) / max(1.0, y1 - y0) * 0.5)
    nz = np.sqrt(np.clip(1 - t * t, 0, 1))
    nx, ny, nz = _normalize(nx * mask, ny * mask, nz * mask)
    return Body(mask, nx, ny, nz, material, depth)


def shell(base: Body, grow: int = 2, keep: str = "outer", material: str = "",
          depth: float = 0.0) -> Body:
    """Оболочка поверх формы: волосы на черепе, шлем, кожура.

    Нормали берутся у основы — оболочка повторяет её кривизну, поэтому
    волосы освещаются как голова, а не как отдельный предмет.
    """
    grown = base.mask.copy()
    for _ in range(max(0, grow)):
        acc = grown.copy()
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            acc |= np.roll(np.roll(grown, dy, axis=0), dx, axis=1)
        grown = acc
    mask = grown & ~base.mask if keep == "outer" else grown
    return Body(mask, base.nx, base.ny, base.nz, material or base.material, depth)


def plane(mask: np.ndarray, normal: tuple[float, float, float] = (0, 0, 1),
          material: str = "", depth: float = 0.0) -> Body:
    """Плоскость: грань, ткань, вывеска. Нормаль одна на всю площадь."""
    nx = np.full(mask.shape, float(normal[0]), dtype=np.float32) * mask
    ny = np.full(mask.shape, float(normal[1]), dtype=np.float32) * mask
    nz = np.full(mask.shape, float(normal[2]), dtype=np.float32) * mask
    nx, ny, nz = _normalize(nx, ny, nz)
    return Body(mask.copy(), nx, ny, nz, material, depth)


# ── свет ─────────────────────────────────────────────────────────────────

def light_field(
    body: Body,
    light: tuple[float, float, float],
    ambient: float = 0.18,
    reflex: float = 0.55,
    reflex_width: float = 0.45,
) -> np.ndarray:
    """Освещённость части: диффузный свет плюс рефлекс на теневом крае.

    Терминатор получается сам собой: там, где нормаль перпендикулярна лучу,
    диффузная часть падает до нуля, а рефлекс ещё не начался. Это и есть
    самая тёмная полоса, и лежит она внутри формы, а не по кромке.
    """
    lx, ly, lz = light
    norm = max(1e-6, (lx * lx + ly * ly + lz * lz) ** 0.5)
    lx, ly, lz = lx / norm, ly / norm, lz / norm

    ndotl = body.nx * lx + body.ny * ly + body.nz * lz
    diffuse = np.clip(ndotl, 0, 1)

    # рефлекс: работает там, где поверхность отвёрнута от света и уже завалена
    # к краю силуэта (nz мал). Без него теневая сторона выглядит вырезанной.
    facing_away = np.clip(-ndotl, 0, 1)
    near_edge = np.clip(1 - np.abs(body.nz) / max(reflex_width, 1e-6), 0, 1)
    bounce = reflex * facing_away * near_edge

    return np.where(body.mask, np.clip(ambient + (1 - ambient) * diffuse + bounce, 0, 1), 0)


def render(
    canvas: Canvas,
    bodies: list[Body],
    ramps: dict[str, list[int]],
    light: tuple[float, float, float] = (-0.6, -0.8, 0.4),
    ambient: float = 0.18,
    reflex: float = 0.55,
    dither: float = 0.12,
) -> dict[str, np.ndarray]:
    """Разложить части по холсту. Возвращает поля освещённости по материалам.

    Части кладутся по глубине: что выше, то и перекрывает. Тон берётся
    индексом в шкале материала — конкретных цветов эта функция не знает.
    """
    fields: dict[str, np.ndarray] = {}
    tile = np.tile(
        np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]],
                 dtype=np.float32) / 16.0,
        (canvas.height // 4 + 1, canvas.width // 4 + 1),
    )[: canvas.height, : canvas.width]

    for body in sorted(bodies, key=lambda b: b.depth):
        ramp = ramps.get(body.material)
        if not ramp or not body.mask.any():
            continue
        field = light_field(body, light, ambient=ambient, reflex=reflex)
        value = field * (len(ramp) - 1)
        if dither > 0:
            frac = value - np.floor(value)
            band = (frac < dither) | (frac > 1 - dither)
            value = np.where(band, value + (tile - 0.5) * dither * 2.0, value)
        idx = np.clip(np.rint(value), 0, len(ramp) - 1).astype(int)
        canvas.data[body.mask] = np.array(ramp, dtype=np.int16)[idx][body.mask]
        fields[body.material] = np.maximum(fields.get(body.material, 0), field)
    return fields


def occlude(canvas: Canvas, upper: Body, lower: Body, index: int,
            width: int = 1, drop: tuple[int, int] = (0, 1)) -> int:
    """Линия смыкания: под чёлкой, под подбородком, в стыке руки и корпуса.

    [drop] — куда падает тень от верхней части. По умолчанию вниз: тень от
    чёлки ложится на лоб, а не обводит его кольцом. Обвод по всему контуру
    читается как оправа очков, это уже проверено на голове.
    """
    dx, dy = int(np.sign(drop[0])), int(np.sign(drop[1]))
    edge = upper.mask.copy()
    for step in range(1, max(1, width) + 1):
        edge = edge | np.roll(np.roll(upper.mask, dy * step, axis=0), dx * step, axis=1)
    line = edge & lower.mask & ~upper.mask
    canvas.data[line] = index
    return int(line.sum())


@dataclass
class Scene:
    """Сцена: один свет на всё, части и материалы.

    Свет объявляется здесь и больше нигде: части не решают, откуда он падает,
    а разъезд направлений — самая заметная ошибка на глаз.
    """

    canvas: Canvas
    light: tuple[float, float, float] = (-0.6, -0.8, 0.4)
    ambient: float = 0.18
    reflex: float = 0.55
    bodies: list[Body] = field(default_factory=list)

    def add(self, body: Body) -> Body:
        self.bodies.append(body)
        return body

    def draw(self, ramps: dict[str, list[int]], dither: float = 0.12) -> dict:
        return render(self.canvas, self.bodies, ramps, light=self.light,
                      ambient=self.ambient, reflex=self.reflex, dither=dither)


# ── сложные формы ────────────────────────────────────────────────────────

def strand(shape, points: list[tuple[float, float]], width: float = 3.0,
           taper: float = 0.45, material: str = "", depth: float = 0.0) -> Body:
    """Прядь по кривой: жгут, локон, лиана, складка ткани.

    Одна гладкая шапка читается как полоска и есть та самая простота, из-за
    которой причёска выглядит куском картона. Настоящая форма набирается
    прядями: каждая идёт своей дугой, к концу утончается, а перекрытия между
    ними и дают объём.

    Нормаль поперёк жгута считается как у цилиндра, поэтому по каждой пряди
    проходит свой блик, а не общий на всю массу.
    """
    yy, xx = _grid(shape)
    best = np.full(shape, 1e9, dtype=np.float32)
    perp_x = np.zeros(shape, dtype=np.float32)
    perp_y = np.zeros(shape, dtype=np.float32)
    radius = np.zeros(shape, dtype=np.float32)

    # ломаная сглаживается по трём соседним точкам: без этого на стыках
    # сегментов остаются углы, и прядь выглядит собранной из палок
    dense: list[tuple[float, float]] = []
    for i in range(len(points) - 1):
        (x0, y0), (x1, y1) = points[i], points[i + 1]
        steps = max(2, int(abs(x1 - x0) + abs(y1 - y0)))
        for s in range(steps):
            t = s / steps
            dense.append((x0 + (x1 - x0) * t, y0 + (y1 - y0) * t))
    dense.append(points[-1])

    total = max(1, len(dense) - 1)
    for i in range(total):
        (x0, y0), (x1, y1) = dense[i], dense[i + 1]
        t = i / total
        r = width * (1 - taper * t)               # к концу прядь тоньше
        dx, dy = x1 - x0, y1 - y0
        seg = max(1e-6, (dx * dx + dy * dy) ** 0.5)
        ux, uy = dx / seg, dy / seg
        # расстояние до отрезка
        px, py = xx - x0, yy - y0
        proj = np.clip(px * ux + py * uy, 0, seg)
        cx_, cy_ = x0 + ux * proj, y0 + uy * proj
        dist = np.sqrt((xx - cx_) ** 2 + (yy - cy_) ** 2)
        closer = dist < best
        best = np.where(closer, dist, best)
        perp_x = np.where(closer, -uy, perp_x)    # перпендикуляр к оси пряди
        perp_y = np.where(closer, ux, perp_y)
        radius = np.where(closer, r, radius)

    mask = best <= radius
    t = np.clip(np.where(mask, best / np.maximum(radius, 1e-6), 0), 0, 1)
    side = np.sign((xx - shape[1] / 2) * perp_x + (yy - shape[0] / 2) * perp_y)
    side[side == 0] = 1
    nx = perp_x * t * side
    ny = perp_y * t * side
    nz = np.sqrt(np.clip(1 - t * t, 0, 1))
    nx, ny, nz = _normalize(nx * mask, ny * mask, nz * mask)
    return Body(mask, nx, ny, nz, material, depth)


def hair_mass(shape, cx: float, cy: float, r: float, count: int = 9,
              spread: float = 1.15, length: float = 1.5, width: float = 3.4,
              jitter: float = 0.22, material: str = "hair",
              depth: float = 1.0, seed: int = 0,
              keep_clear: np.ndarray | None = None) -> list[Body]:
    """Причёска из прядей, а не шапка одним куском.

    Пряди расходятся от макушки, каждая со своей длиной и наклоном, поэтому
    нижний край выходит рваным сам собой. Соседние ложатся с разной глубиной,
    и между ними получаются настоящие перекрытия.

    [keep_clear] — куда прядям хода нет. Без этой маски они сходятся к центру
    и закрывают лицо занавесом: объём есть, а персонажа нет.
    """
    rng = np.random.default_rng(seed)
    out: list[Body] = []
    for i in range(count):
        k = i / max(1, count - 1)
        ang = (-spread / 2 + spread * k) * np.pi
        wobble = float(rng.uniform(-jitter, jitter))
        tip = float(rng.uniform(0.75, 1.25)) * length
        pts = [
            (cx + np.sin(ang) * r * 0.25, cy - r * 0.75),
            (cx + np.sin(ang + wobble) * r * 0.85, cy - r * 0.15),
            (cx + np.sin(ang + wobble * 1.6) * r * 1.05, cy + r * 0.55 * tip),
            (cx + np.sin(ang + wobble * 2.1) * r * 1.0, cy + r * 0.95 * tip),
        ]
        body = strand(shape, pts, width=width * float(rng.uniform(0.8, 1.2)),
                      taper=0.5, material=material, depth=depth + i * 0.01)
        if keep_clear is not None:
            body = Body(body.mask & ~keep_clear, body.nx, body.ny, body.nz,
                        body.material, body.depth)
        if body.mask.any():
            out.append(body)
    return out
