"""Изометрические спрайты из тел, посчитанные лучами.

Рисовать каждый предмет кистью — работа без конца. Вместо этого собираем его
из простых тел (цилиндр, сфера, коробка, конус, тор) и считаем картинку
честно: для каждого пикселя пускаем луч, находим ближайшее тело, берём
нормаль, освещаем и квантуем в палитру. Тогда цилиндр получает цилиндрический
свет, купол — сферический, а новый предмет появляется из рецепта, а не из
пикселей.

    from pixelloom import palettes
    from pixelloom.isoparts import Material, Part, render_parts

    parts = [Part("box", pos=(0, 0, 0.4), size=(1.2, 1.2, 0.8),
                  material=Material(ramp="grey"))]
    canvas = render_parts(parts, palettes.BASE16, 64, 64, px_per_m=24)

Шкала берётся из палитры по имени, и палитра обязана её знать. Готовые наборы
pixelloom дают `grey`, `warm`, `green`; имя по умолчанию (`hull`) рассчитано на
свою палитру, где есть шкала корпуса.

Проекция классическая изометрическая: луч идёт вдоль (1, 1, 1), поэтому
экранные координаты связаны с мировыми как
    sx = (x - y) * k,  sy = (x + y) * k / 2 - z * k
Мир меряется в метрах, k задаёт, сколько пикселей приходится на метр.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .canvas import TRANSPARENT, Canvas
from .draw import BAYER4, shade_from_field
from .palette import Palette

# Свет: сверху, слева и от зрителя. Тот же, что на планете, чтобы постройки
# и грунт освещались согласованно.
LIGHT = np.array([-0.45, -0.30, 0.84])
LIGHT /= np.linalg.norm(LIGHT)

# Камера стоит сверху-слева-спереди и смотрит вниз. Направление взгляда
# обязано иметь отрицательную вертикаль: с (1,1,1) мы разглядывали сцену
# снизу, и срезанные купола показывали изнанку.
VIEW = np.array([-1.0, -1.0, -1.0]) / np.sqrt(3.0)


@dataclass
class Material:
    """Как красить поверхность."""

    ramp: str = "hull"
    ambient: float = 0.34
    gloss: float = 0.0  # доля зеркального блика
    emissive: bool = False  # светится сам, темноту не боится


@dataclass
class Part:
    """Одно тело в локальных координатах модуля, метры.

    Луч переводится в систему тела, поэтому наклон и вытягивание достаются
    бесплатно: коробка с поворотом становится скошенной опорой, сфера с
    масштабом — надувным модулем, конус — соплом или мачтой.
    """

    kind: str  # sphere | cylinder | box | cone | torus
    pos: tuple[float, float, float]
    size: tuple[float, float, float]
    material: Material = field(default_factory=Material)
    axis: str = "x"  # ось цилиндра, конуса, тора
    cut_below: float | None = None  # обрезать снизу: купол вместо шара
    yaw: float = 0.0  # поворот вокруг вертикали, градусы
    pitch: float = 0.0  # наклон, градусы
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    detail: str = ""  # узор поверхности: panels, ribs, ports, grid, stripe
    detail_scale: float = 1.0


def _rot_matrix(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    """Поворот вокруг вертикали, затем наклон."""
    ya, pa = np.deg2rad(yaw_deg), np.deg2rad(pitch_deg)
    cy, sy = np.cos(ya), np.sin(ya)
    cp, sp = np.cos(pa), np.sin(pa)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    return rz @ ry


def _ray_cone(o, d, half_h, r_bottom, r_top):
    """Усечённый конус вдоль оси Z в локальных координатах."""
    dr = (r_top - r_bottom) / (2 * half_h)
    m = r_bottom + dr * (o[..., 2] + half_h)
    a = d[0] ** 2 + d[1] ** 2 - (dr * d[2]) ** 2
    b = 2 * (o[..., 0] * d[0] + o[..., 1] * d[1] - m * dr * d[2])
    c = o[..., 0] ** 2 + o[..., 1] ** 2 - m * m
    disc = b * b - 4 * a * c
    hit = (disc >= 0) & (np.abs(a) > 1e-9)
    sq = np.sqrt(np.where(hit, disc, 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        t0 = (-b - sq) / (2 * a)
        t1 = (-b + sq) / (2 * a)

    def ok(t):
        z = o[..., 2] + t * d[2]
        # радиус в точке пересечения обязан быть неотрицательным: иначе луч
        # попал на зеркальное продолжение конуса за вершиной, и по кадру
        # растягивается ложная полоса
        r_at = r_bottom + dr * (z + half_h)
        return (t > 1e-4) & (np.abs(z) <= half_h) & (r_at >= 0)

    # без проверки дискриминанта корни существуют даже там, где пересечения
    # нет: sqrt от нуля даёт двойной ложный корень и полосу через весь кадр
    t = np.where(hit & ok(t0), t0, np.where(hit & ok(t1), t1, np.inf))
    t = np.where(np.abs(a) > 1e-9, t, np.inf)

    # крышки
    if abs(d[2]) > 1e-9:
        for sign, rad in ((-1.0, r_bottom), (1.0, r_top)):
            tc = (sign * half_h - o[..., 2]) / d[2]
            px = o[..., 0] + tc * d[0]
            py = o[..., 1] + tc * d[1]
            good = (tc > 1e-4) & (px * px + py * py <= rad * rad) & (tc < t)
            t = np.where(good, tc, t)
    return np.where(np.isfinite(t), t, np.inf)


def _cone_normal(p, half_h, r_bottom, r_top):
    dr = (r_top - r_bottom) / (2 * half_h)
    on_cap = np.abs(np.abs(p[..., 2]) - half_h) < 1e-3
    radial = np.stack([p[..., 0], p[..., 1], np.zeros_like(p[..., 0])], axis=-1)
    rn = np.linalg.norm(radial, axis=-1, keepdims=True)
    radial = radial / np.maximum(rn, 1e-9)
    n = radial.copy()
    n[..., 2] = -dr
    n = n / np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-9)
    cap = np.zeros_like(p)
    cap[..., 2] = np.sign(p[..., 2])
    return np.where(on_cap[..., None], cap, n)


def _ray_torus(o, d, big_r, small_r):
    """Тор в плоскости XY.

    Аналитика даёт уравнение четвёртой степени, поэтому идём маршем. Но
    начинаем не от камеры, а от входа в габаритную сферу: иначе шаги тратятся
    на пустоту, тело не находится вовсе и по кадру ползут ложные попадания.
    """
    bound = big_r + small_r * 1.05
    t_enter = _ray_sphere(o, d, np.zeros(3), bound)
    reach = 2.2 * bound
    dn = float(np.linalg.norm(d)) if np.ndim(d) == 1 else 1.0

    t = np.where(np.isfinite(t_enter), t_enter, 0.0)
    hit_t = np.full(t.shape, np.inf)
    alive = np.isfinite(t_enter)
    travelled = np.zeros_like(t)

    for _ in range(90):
        if not alive.any():
            break
        p = o + t[..., None] * d
        q = np.sqrt(p[..., 0] ** 2 + p[..., 1] ** 2) - big_r
        dist = np.sqrt(q * q + p[..., 2] ** 2) - small_r
        landed = alive & (dist < 0.012)
        hit_t = np.where(landed, t, hit_t)
        alive = alive & ~landed & (travelled < reach)
        step = np.maximum(dist * 0.8, small_r * 0.06) / max(dn, 1e-6)
        t = np.where(alive, t + step, t)
        travelled = travelled + np.where(alive, step, 0.0)
    return hit_t


def _torus_normal(p, big_r):
    rad = np.sqrt(p[..., 0] ** 2 + p[..., 1] ** 2)
    cx = p[..., 0] / np.maximum(rad, 1e-9) * big_r
    cy = p[..., 1] / np.maximum(rad, 1e-9) * big_r
    n = np.stack([p[..., 0] - cx, p[..., 1] - cy, p[..., 2]], axis=-1)
    return n / np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-9)


def surface_detail(part, p_local, normal, scale=1.0):
    """Узор на поверхности: возвращает поправку к яркости.

    Швы панелей, рёбра жёсткости, ряды люков и решётки рисуются не кистью, а
    по координатам самой поверхности. Поэтому они ложатся по форме и не едут
    при смене размеров.
    """
    kind = part.detail
    if not kind:
        return 0.0
    s = max(part.detail_scale, 1e-3) * scale
    x, y, z = p_local[..., 0], p_local[..., 1], p_local[..., 2]
    ang = np.arctan2(y, x)

    if kind == "panels":
        # прямоугольная нарезка обшивки
        u = np.floor(z / (0.9 * s)) + np.floor(ang / (np.pi / 5 * s))
        seam = (np.abs(((z / (0.9 * s)) % 1.0) - 0.5) > 0.47) | (
            np.abs(((ang / (np.pi / 4 * s)) % 1.0) - 0.5) > 0.47)
        return np.where(seam, -0.07, 0.015 * ((u % 2) * 2 - 1))
    if kind == "ribs":
        band = np.abs((z / (0.55 * s) % 1.0) - 0.5)
        return np.where(band > 0.36, 0.13, -0.05)
    if kind == "ports":
        row = np.abs((z / (0.8 * s) % 1.0) - 0.5)
        col = np.abs((ang / (np.pi / 6 * s) % 1.0) - 0.5)
        return np.where((row < 0.18) & (col < 0.2), -0.30, 0.0)
    if kind == "grid":
        gx = np.abs((x / (0.5 * s) % 1.0) - 0.5)
        gy = np.abs((y / (0.5 * s) % 1.0) - 0.5)
        return np.where((gx > 0.38) | (gy > 0.38), -0.18, 0.05)
    if kind == "stripe":
        band = (z / (1.1 * s)) % 1.0
        return np.where(band < 0.16, 0.22, 0.0)
    if kind == "hatch":
        d2 = x * x + y * y
        return np.where(np.abs(np.sqrt(d2) - 0.55 * s) < 0.07 * s, -0.25, 0.0)
    return 0.0


def _ray_sphere(o, d, c, r):
    """Пересечение пучка лучей со сферой.

    Формула полная, с коэффициентом при квадрате направления: после
    масштабирования тела вектор направления перестаёт быть единичным, и
    сокращённый вариант врёт — сплюснутая сфера заливала весь кадр.
    """
    oc = o - c
    a = float(np.dot(d, d)) if np.ndim(d) == 1 else np.einsum("...i,...i->...", d, d)
    b = 2.0 * np.einsum("...i,i->...", oc, d) if np.ndim(d) == 1 else 2.0 * np.einsum("...i,...i->...", oc, d)
    cc = np.einsum("...i,...i->...", oc, oc) - r * r
    disc = b * b - 4 * a * cc
    hit = disc >= 0
    sq = np.sqrt(np.where(hit, disc, 0.0))
    t0 = (-b - sq) / (2 * a)
    t1 = (-b + sq) / (2 * a)
    t = np.where(t0 > 1e-4, t0, t1)
    return np.where(hit & (t > 1e-4), t, np.inf)


def _ray_box(o, d, lo, hi):
    """Пересечение с параллелепипедом по методу плит."""
    inv = 1.0 / np.where(np.abs(d) < 1e-9, 1e-9, d)
    t_lo = (lo - o) * inv
    t_hi = (hi - o) * inv
    t_near = np.max(np.minimum(t_lo, t_hi), axis=-1)
    t_far = np.min(np.maximum(t_lo, t_hi), axis=-1)
    ok = (t_far >= t_near) & (t_far > 1e-4)
    t = np.where(t_near > 1e-4, t_near, t_far)
    return np.where(ok, t, np.inf)


def _box_normal(p, lo, hi):
    """Нормаль коробки: та грань, к которой точка ближе всего."""
    c = (lo + hi) / 2
    d = (hi - lo) / 2
    rel = (p - c) / np.maximum(d, 1e-6)
    a = np.abs(rel)
    m = a.max(axis=-1, keepdims=True)
    n = np.where(a >= m - 1e-3, np.sign(rel), 0.0)
    norm = np.linalg.norm(n, axis=-1, keepdims=True)
    return n / np.maximum(norm, 1e-9)


def _ray_cylinder(o, d, c, radius, half, axis):
    """Цилиндр с крышками вдоль одной из осей."""
    ax = {"x": 0, "y": 1, "z": 2}[axis]
    other = [i for i in range(3) if i != ax]

    oc = o - c
    a = d[other[0]] ** 2 + d[other[1]] ** 2
    b = 2 * (oc[..., other[0]] * d[other[0]] + oc[..., other[1]] * d[other[1]])
    cc = oc[..., other[0]] ** 2 + oc[..., other[1]] ** 2 - radius**2
    disc = b * b - 4 * a * cc
    hit = disc >= 0
    sq = np.sqrt(np.where(hit, disc, 0.0))
    t0 = (-b - sq) / (2 * a)
    t1 = (-b + sq) / (2 * a)

    def valid(t):
        z = oc[..., ax] + t * d[ax]
        return (t > 1e-4) & (np.abs(z) <= half)

    t_side = np.where(valid(t0), t0, np.where(valid(t1), t1, np.inf))

    # крышки
    t_cap = np.full(t_side.shape, np.inf)
    if abs(d[ax]) > 1e-9:
        for sign in (-1.0, 1.0):
            tc = (sign * half - oc[..., ax]) / d[ax]
            px = oc[..., other[0]] + tc * d[other[0]]
            py = oc[..., other[1]] + tc * d[other[1]]
            good = (tc > 1e-4) & (px * px + py * py <= radius * radius)
            t_cap = np.where(good & (tc < t_cap), tc, t_cap)

    return np.where(hit | np.isfinite(t_cap), np.minimum(t_side, t_cap), np.inf)


def _cylinder_normal(p, c, radius, half, axis):
    ax = {"x": 0, "y": 1, "z": 2}[axis]
    other = [i for i in range(3) if i != ax]
    rel = p - c
    on_cap = np.abs(np.abs(rel[..., ax]) - half) < 1e-3
    n = np.zeros_like(rel)
    n[..., other[0]] = rel[..., other[0]]
    n[..., other[1]] = rel[..., other[1]]
    norm = np.linalg.norm(n, axis=-1, keepdims=True)
    n = n / np.maximum(norm, 1e-9)
    cap_n = np.zeros_like(rel)
    cap_n[..., ax] = np.sign(rel[..., ax])
    return np.where(on_cap[..., None], cap_n, n)


def _axis_matrix(axis: str) -> np.ndarray:
    """Канонические тела вытянуты вдоль Z; axis доворачивает их."""
    if axis == "z":
        return np.eye(3)
    if axis == "x":
        return _rot_matrix(0.0, 90.0)
    return _rot_matrix(90.0, 90.0)


# Ось имеет смысл только для вытянутых тел. Коробку и сферу разворачивать
# по умолчанию нельзя: у Part ось стоит "x", и без этой проверки все коробки
# ложились набок.
AXIAL_KINDS = ("cylinder", "cone", "torus")


def _prepare(part: Part):
    """Матрица поворота и масштаб тела."""
    axis = _axis_matrix(part.axis) if part.kind in AXIAL_KINDS else np.eye(3)
    rot = _rot_matrix(part.yaw, part.pitch) @ axis
    scale = np.array(part.scale, dtype=float)
    return rot, scale


def _hit_local(part: Part, ol, dl):
    """Пересечение луча с телом в его собственных координатах."""
    s = np.array(part.size, dtype=float)
    if part.kind == "sphere":
        return _ray_sphere(ol, dl, np.zeros(3), s[0])
    if part.kind == "box":
        return _ray_box(ol, dl, -s / 2, s / 2)
    if part.kind == "cylinder":
        return _ray_cylinder(ol, dl, np.zeros(3), s[0], s[1], "z")
    if part.kind == "cone":
        return _ray_cone(ol, dl, s[1], s[0], s[2])
    if part.kind == "torus":
        return _ray_torus(ol, dl, s[0], s[1])
    raise ValueError(f"неизвестное тело: {part.kind}")


def _normal_local(part: Part, pl):
    s = np.array(part.size, dtype=float)
    if part.kind == "sphere":
        return pl / np.maximum(np.linalg.norm(pl, axis=-1, keepdims=True), 1e-9)
    if part.kind == "box":
        return _box_normal(pl, -s / 2, s / 2)
    if part.kind == "cylinder":
        return _cylinder_normal(pl, np.zeros(3), s[0], s[1], "z")
    if part.kind == "cone":
        return _cone_normal(pl, s[1], s[0], s[2])
    return _torus_normal(pl, s[0])


def render_parts(
    parts: list[Part],
    palette: Palette,
    width: int,
    height: int,
    px_per_m: float = 8.0,
    origin: tuple[float, float] = (0.5, 0.85),
    dither: float = 0.35,
    grain: float = 0.10,
    shadows: bool = True,
    seed: int = 1,
) -> Canvas:
    """Собрать спрайт из тел.

    Луч переводится в систему каждого тела, поэтому наклон, поворот и
    вытягивание работают для всех примитивов одинаково.
    """
    canvas = Canvas(width, height, palette)
    canvas.data[:, :] = TRANSPARENT

    yy, xx = np.mgrid[0:height, 0:width]
    sx = (xx - origin[0] * width).astype(float)
    sy = (yy - origin[1] * height).astype(float)

    k = px_per_m
    u = sx / k
    v = sy / k
    wx = (u + 2 * v) / 2.0
    wy = (2 * v - u) / 2.0
    back = 80.0
    o = np.stack([wx + back, wy + back, np.full_like(wx, back)], axis=-1)
    d = VIEW

    best_t = np.full((height, width), np.inf)
    best_idx = np.full((height, width), -1, dtype=int)

    prepared = []
    for i, part in enumerate(parts):
        c = np.array(part.pos, dtype=float)
        rot, scale = _prepare(part)
        ol = np.einsum("ij,...j->...i", rot.T, o - c) / scale
        dl = (rot.T @ d) / scale
        t = _hit_local(part, ol, dl)

        if part.cut_below is not None:
            p_world = o + np.where(np.isfinite(t), t, 0.0)[..., None] * d
            t = np.where(p_world[..., 2] >= part.cut_below, t, np.inf)

        prepared.append((part, c, rot, scale, ol, dl))
        closer = t < best_t
        best_t = np.where(closer, t, best_t)
        best_idx = np.where(closer, i, best_idx)

    solid = np.isfinite(best_t)
    if not solid.any():
        return canvas

    t_safe = np.where(solid, best_t, 0.0)
    point = o + t_safe[..., None] * d

    for i, (part, c, rot, scale, ol, dl) in enumerate(prepared):
        mask = solid & (best_idx == i)
        if not mask.any():
            continue

        pl = ol + t_safe[..., None] * dl
        nl = _normal_local(part, pl)
        n = np.einsum("ij,...j->...i", rot, nl / scale)
        n = n / np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-9)

        lambert = np.clip((n * LIGHT).sum(axis=-1), 0, 1)
        mat = part.material
        value = mat.ambient + (1.0 - mat.ambient) * lambert

        if mat.gloss > 0:
            half = LIGHT + VIEW
            half /= np.linalg.norm(half)
            spec = np.clip((n * half).sum(axis=-1), 0, 1) ** 24
            value = np.clip(value + spec * mat.gloss, 0, 1)

        if part.detail:
            value = value + surface_detail(part, pl, nl)

        if shadows and not mat.emissive:
            lit = np.ones_like(value)
            start = point + n * 0.03
            for other, oc2, rot2, sc2, _, _ in prepared:
                if other is part:
                    continue
                ol2 = np.einsum("ij,...j->...i", rot2.T, start - oc2) / sc2
                dl2 = (rot2.T @ LIGHT) / sc2
                ts = _hit_local(other, ol2, dl2)
                lit = np.where(np.isfinite(ts), 0.58, lit)
            value = value * lit

        if mat.emissive:
            value = np.full_like(value, 0.92)

        tmp = Canvas(width, height, palette)
        shade_from_field(
            tmp, np.clip(value, 0, 1), palette.ramp(mat.ramp),
            dither=dither, matrix=BAYER4, noise=grain, seed=seed + i * 7,
        )
        canvas.data[mask] = tmp.data[mask]

    return canvas


def ground_shadow(canvas: Canvas, parts: list[Part], px_per_m: float = 8.0,
                  origin: tuple[float, float] = (0.5, 0.85), index: int | None = None) -> None:
    """Положить под модуль тень на землю: проекция тел на плоскость z=0."""
    palette = canvas.palette
    idx = index if index is not None else palette.index_of("shadow")
    h, w = canvas.data.shape
    yy, xx = np.mgrid[0:h, 0:w]
    sx = (xx - origin[0] * w) / px_per_m
    sy = (yy - origin[1] * h) / px_per_m
    wx = (sx + 2 * sy) / 2.0
    wy = (2 * sy - sx) / 2.0

    shade = np.zeros((h, w), bool)
    for part in parts:
        c = np.array(part.pos, dtype=float)
        s = np.array(part.size, dtype=float)
        # тень падает по направлению света, смещение тем больше, чем выше тело
        off = -LIGHT[:2] / max(LIGHT[2], 0.2) * c[2]
        dx = wx - (c[0] + off[0])
        dy = wy - (c[1] + off[1])
        sc = np.array(part.scale, dtype=float)
        if part.kind == "sphere":
            shade |= (dx / sc[0]) ** 2 + (dy / sc[1]) ** 2 <= s[0] ** 2
        elif part.kind == "box":
            shade |= (np.abs(dx) <= s[0] / 2 * sc[0]) & (np.abs(dy) <= s[1] / 2 * sc[1])
        elif part.kind == "torus":
            shade |= dx**2 + dy**2 <= (s[0] + s[1]) ** 2
        else:
            r, hl = s[0], s[1]
            if part.axis == "x":
                shade |= (np.abs(dx) <= hl) & (np.abs(dy) <= r)
            elif part.axis == "y":
                shade |= (np.abs(dx) <= r) & (np.abs(dy) <= hl)
            else:
                shade |= dx**2 + dy**2 <= r**2
    canvas.data[shade & (canvas.data == TRANSPARENT)] = idx


def screen_bounds(parts: list[Part], px_per_m: float) -> tuple[int, int, float, float]:
    """Подобрать размер спрайта под габариты тел.

    Углы описывающей коробки поворачиваются вместе с телом, а не заменяются
    шаром по диагонали: иначе у фигуры с наклонёнными конечностями кадр
    раздувается вдвое и вокруг остаётся пустота.
    """
    xs, ys = [], []
    for part in parts:
        c = np.array(part.pos, dtype=float)
        s = np.array(part.size, dtype=float)
        sc = np.array(part.scale, dtype=float)
        if part.kind == "sphere":
            half = np.array([s[0], s[0], s[0]])
        elif part.kind == "box":
            half = s / 2
        elif part.kind == "torus":
            half = np.array([s[0] + s[1], s[0] + s[1], s[1]])
        elif part.kind == "cone":
            r = max(s[0], s[2])
            half = np.array([r, r, s[1]])
        else:
            half = np.array([s[0], s[0], s[1]])
        half = half * sc

        rot, _ = _prepare(part)
        for dx in (-1, 1):
            for dy in (-1, 1):
                for dz in (-1, 1):
                    local = half * np.array([dx, dy, dz])
                    p = c + rot @ local
                    xs.append((p[0] - p[1]) * px_per_m)
                    ys.append((p[0] + p[1]) * px_per_m / 2 - p[2] * px_per_m)
    pad = 3
    w = int(round(max(xs) - min(xs))) + pad * 2
    h = int(round(max(ys) - min(ys))) + pad * 2
    ox = (-min(xs) + pad) / w
    oy = (-min(ys) + pad) / h
    return w, h, ox, oy


def render_module(parts: list[Part], palette: Palette, px_per_m: float = 7.0,
                  shadow: bool = True, **kwargs) -> Canvas:
    """Отрисовать модуль, сам подобрав кадр."""
    w, h, ox, oy = screen_bounds(parts, px_per_m)
    if shadow:
        h += int(px_per_m * 1.2)
    canvas = render_parts(parts, palette, w, h, px_per_m=px_per_m, origin=(ox, oy), **kwargs)
    if shadow:
        ground_shadow(canvas, parts, px_per_m=px_per_m, origin=(ox, oy))
    return canvas
