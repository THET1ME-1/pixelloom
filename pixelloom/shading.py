"""Свет как система: объём, кромки и шкалы тонов из плоской маски силуэта.

Плоская заливка с тенью справа даёт пластик. Настоящий объём держится на
четырёх вещах, и все четыре считаются отсюда:

* **форма** — расстояние до края маски задаёт псевдовысоту, из неё берутся
  нормали, а из нормалей ламбертово освещение;
* **отражённый свет** — снизу и с теневой стороны кромка подсвечена тем, что
  отразилось от земли, иначе тень выглядит дырой;
* **контровой свет** — тонкая яркая полоса по краю со стороны источника,
  она отделяет предмет от фона;
* **температура** — тень не просто темнее, она холоднее и насыщеннее, свет
  теплее и бледнее. Ровное затемнение читается как грязь.

Шкала тонов строится из одного цвета (`ramp_from_base`), а квантование
раскладывает поле яркости по ней, размывая дизерингом только узкую полосу на
границе ступеней — сплошной дизеринг по всей площади и есть тот шум, из-за
которого картинка выглядит нечистой.
"""

from __future__ import annotations

import colorsys

import numpy as np

from .canvas import TRANSPARENT, Canvas
from .palette import Palette, hex_to_rgb, rgb_to_hex

__all__ = [
    "edge_distance",
    "height_field",
    "normals",
    "lambert",
    "rim_light",
    "bounce_light",
    "shade_mask",
    "ramp_from_base",
    "build_ramp",
    "clean_stray",
]


# ── форма ────────────────────────────────────────────────────────────────

def _erode(mask: np.ndarray, diagonal: bool = False) -> np.ndarray:
    """Сжать маску на пиксель. [diagonal] добавляет угловых соседей."""
    out = mask.copy()
    shifts = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if diagonal:
        shifts += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    for dy, dx in shifts:
        out &= np.roll(np.roll(mask, dy, axis=0), dx, axis=1)
    out[0, :] = out[-1, :] = False
    out[:, 0] = out[:, -1] = False
    return out


def _blur(field: np.ndarray, rounds: int = 2) -> np.ndarray:
    """Мягкое усреднение 3×3: снимает ступеньки изолиний."""
    out = field.astype(np.float32)
    for _ in range(max(0, rounds)):
        acc = out.copy()
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            acc = acc + np.roll(np.roll(out, dy, axis=0), dx, axis=1)
        out = acc / 5.0
    return out


def edge_distance(mask: np.ndarray, limit: int = 64) -> np.ndarray:
    """Сколько пикселей до ближайшего края маски. Ноль вне маски.

    Считается послойным сжатием: для спрайтов это дешевле и точнее на глаз,
    чем евклидово расстояние, потому что слои совпадают с пиксельной сеткой.
    """
    dist = np.zeros(mask.shape, dtype=np.float32)
    cur = mask.copy()
    for step in range(1, limit + 1):
        if not cur.any():
            break
        dist[cur] = step
        # чередуем прямую и диагональную эрозию: чистая четырёхсвязная даёт
        # ромбовидные изолинии, и на круглом теле видно диагональный шов
        cur = _erode(cur, diagonal=step % 2 == 1)
    return dist


def height_field(mask: np.ndarray, roundness: float = 1.0) -> np.ndarray:
    """Псевдовысота В ПИКСЕЛЯХ: у края ноль, к середине купол.

    Высота обязана быть в тех же единицах, что и ширина. Нормированная в доли
    единицы карта даёт наклон около нуля, нормали смотрят прямо на зрителя, и
    ламберт возвращает ровную заливку — ровно так объём и терялся.

    [roundness] правит профиль: 1.0 — сфера (наклон у края крутой), 0.0 —
    конус с прямыми боками. Металл и ткань любят промежуточные значения.
    """
    d = edge_distance(mask)
    if not d.any():
        return d
    d = np.where(mask, _blur(d, rounds=2), 0.0)
    dmax = float(d.max())
    t = d / dmax
    sphere = np.sqrt(np.clip(1 - (1 - t) ** 2, 0, 1))
    return dmax * (roundness * sphere + (1 - roundness) * t)


def normals(height: np.ndarray, relief: float = 1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Нормали поверхности из карты высот. [relief] — насколько выпукло."""
    gy, gx = np.gradient(height * relief)
    nx, ny, nz = -gx, -gy, np.ones_like(height)
    length = np.sqrt(nx * nx + ny * ny + nz * nz)
    length[length == 0] = 1.0
    return nx / length, ny / length, nz / length


# ── свет ─────────────────────────────────────────────────────────────────

def lambert(
    n: tuple[np.ndarray, np.ndarray, np.ndarray],
    direction: tuple[float, float, float] = (-0.55, -0.7, 0.45),
) -> np.ndarray:
    """Диффузное освещение. Направление задаётся в экранных осях: x вправо,
    y вниз, z на зрителя."""
    lx, ly, lz = direction
    norm = max(1e-6, (lx * lx + ly * ly + lz * lz) ** 0.5)
    lx, ly, lz = lx / norm, ly / norm, lz / norm
    nx, ny, nz = n
    return np.clip(nx * lx + ny * ly + nz * lz, 0.0, 1.0)


def rim_light(
    mask: np.ndarray,
    direction: tuple[float, float] = (-1, -1),
    width: int = 1,
) -> np.ndarray:
    """Контровая полоса по краю со стороны источника.

    Без неё предмет липнет к фону: именно эта полоса отделяет силуэт, а не
    чёрная обводка.
    """
    dx, dy = int(np.sign(direction[0])), int(np.sign(direction[1]))
    inner = mask.copy()
    for _ in range(max(1, width)):
        inner = _erode(inner)
    shifted = np.roll(np.roll(mask, -dy, axis=0), -dx, axis=1)
    return mask & ~shifted | (mask & ~inner & np.roll(np.roll(mask, dy, axis=0), dx, axis=1) & ~shifted)


def bounce_light(mask: np.ndarray, height: float = 0.35) -> np.ndarray:
    """Отражённый снизу свет: доля от 0 до 1 по нижней части силуэта.

    Земля возвращает часть света обратно, поэтому низ предмета никогда не
    бывает самым тёмным местом. Пропустить это — и предмет проваливается.
    """
    ys, _ = np.nonzero(mask)
    if len(ys) == 0:
        return np.zeros(mask.shape, dtype=np.float32)
    top, bottom = ys.min(), ys.max()
    span = max(1.0, (bottom - top) * height)
    rows = np.arange(mask.shape[0], dtype=np.float32)[:, None]
    fade = np.clip((rows - (bottom - span)) / span, 0.0, 1.0)
    return np.where(mask, fade, 0.0).astype(np.float32)


# ── цвет ─────────────────────────────────────────────────────────────────

def ramp_from_base(
    base: str,
    steps: int = 5,
    shade_shift: float = -0.055,
    light_shift: float = 0.03,
    saturate_shade: float = 1.35,
    desaturate_light: float = 0.55,
) -> list[str]:
    """Шкала из одного цвета: от глубокой тени к свету.

    Тень уходит по тону в холод и набирает насыщенность, свет уходит в тепло
    и бледнеет. Простое умножение яркости даёт серую грязь — глаз читает
    объём именно по сдвигу тона.
    """
    r, g, b = (v / 255 for v in hex_to_rgb(base))
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    out: list[str] = []
    for i in range(steps):
        t = i / max(1, steps - 1)            # 0 тень, 1 свет
        k = (t - 0.5) * 2                    # -1 … +1
        hh = (h + (shade_shift if k < 0 else light_shift) * abs(k)) % 1.0
        ss = s * (1 + (saturate_shade - 1) * -k if k < 0 else 1 - (1 - desaturate_light) * k)
        # яркость идёт по кривой: тени расходятся сильнее светов, иначе
        # ступени сливаются и объём читается как грязное пятно
        vv = v * (0.30 + 0.70 * t ** 0.8) if k < 0 else v + (1 - v) * (k ** 0.9) * 0.85
        rr, gg, bb = colorsys.hsv_to_rgb(hh, float(np.clip(ss, 0, 1)), float(np.clip(vv, 0, 1)))
        out.append(rgb_to_hex((round(rr * 255), round(gg * 255), round(bb * 255))))
    return out


def build_ramp(palette: Palette, base: str, name: str, steps: int = 5) -> list[int]:
    """Добавить шкалу в палитру и вернуть индексы от тени к свету."""
    hexes = ramp_from_base(base, steps=steps)
    indices: list[int] = []
    for i, hx in enumerate(hexes):
        rgb = hex_to_rgb(hx)
        if rgb in palette.colors:
            indices.append(palette.colors.index(rgb))
            continue
        palette.colors.append(rgb)
        palette.labels.append(f"{name}{i}")
        indices.append(len(palette.colors) - 1)
    palette.ramps[name] = indices
    return indices


# ── рендер ───────────────────────────────────────────────────────────────

# Матрица Байера 4×4 для полосы перехода между ступенями
_BAYER = np.array(
    [[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]],
    dtype=np.float32,
) / 16.0


def shade_mask(
    canvas: Canvas,
    mask: np.ndarray,
    ramp: list[int],
    light: tuple[float, float, float] = (-0.55, -0.7, 0.45),
    roundness: float = 1.0,
    relief: float = 1.0,
    ambient: float = 0.16,
    bounce: float = 0.30,
    rim: float = 0.0,
    dither: float = 0.16,
    contrast: float = 1.35,
    outline: int | None = None,
) -> np.ndarray:
    """Залить маску объёмом по шкале и вернуть поле освещённости.

    Порядок тот же, каким работает художник: сначала общий свет по форме,
    потом отражённый снизу, потом контровая кромка, и только затем разбивка
    по ступеням шкалы с дизерингом в полосе перехода.
    """
    if not mask.any():
        return np.zeros(mask.shape, dtype=np.float32)

    h = height_field(mask, roundness=roundness)
    lit = lambert(normals(h, relief=relief), light) ** (1 / max(0.2, contrast))
    field = ambient + (1 - ambient) * lit
    field = field + bounce * bounce_light(mask)
    if rim:
        field = np.where(rim_light(mask, (light[0], light[1])), field + rim, field)
    field = np.clip(field, 0.0, 1.0)

    n = len(ramp)
    value = field * (n - 1)
    if dither > 0:
        tile = np.tile(
            _BAYER,
            (mask.shape[0] // 4 + 1, mask.shape[1] // 4 + 1),
        )[: mask.shape[0], : mask.shape[1]]
        frac = value - np.floor(value)
        # дизеринг живёт только у самой границы ступени: в середине он
        # превращается в шум и съедает чистоту заливки
        band = (frac < dither) | (frac > 1 - dither)
        value = np.where(band, value + (tile - 0.5) * dither * 2.0, value)

    idx = np.clip(np.rint(value), 0, n - 1).astype(int)
    lut = np.array(ramp, dtype=np.int16)
    canvas.data[mask] = lut[idx][mask]

    if outline is not None:
        # тёмная кромка по контуру силуэта, но только с теневой стороны:
        # со стороны света её съедает контровая полоса
        inner = _erode(mask)
        edge = mask & ~inner
        shadow_side = field < (ambient + (1 - ambient) * 0.45)
        canvas.data[edge & shadow_side] = outline
    return field


def clean_stray(canvas: Canvas, rounds: int = 1) -> int:
    """Убрать одиночные пиксели: те, у кого нет ни одного соседа своего цвета.

    Такие точки художник вычищает в конце руками — они и создают ощущение
    «шумного» спрайта, даже когда формы построены верно.
    """
    fixed = 0
    for _ in range(max(1, rounds)):
        data = np.asarray(canvas.data)
        same = np.zeros(data.shape, dtype=np.int8)
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            same += (np.roll(np.roll(data, dy, axis=0), dx, axis=1) == data).astype(np.int8)
        lonely = (data != TRANSPARENT) & (same == 0)
        if not lonely.any():
            break
        ys, xs = np.nonzero(lonely)
        for y, x in zip(ys.tolist(), xs.tolist()):
            neighbours = [
                int(data[y + dy, x + dx])
                for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1))
                if 0 <= y + dy < data.shape[0] and 0 <= x + dx < data.shape[1]
                and data[y + dy, x + dx] != TRANSPARENT
            ]
            if not neighbours:
                continue
            canvas.data[y, x] = max(set(neighbours), key=neighbours.count)
            fixed += 1
    return fixed
