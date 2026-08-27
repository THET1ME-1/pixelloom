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

# Насколько далеко отодвигается начало луча, чтобы вся сцена оказалась перед
# камерой. Метры; сцена крупнее этого не бывает.
BACK = 80.0


@dataclass(frozen=True)
class Camera:
    """Как мир ложится на экран.

    Проекция была зашита в четырёх местах рендера, и сменить её значило
    переписать их все. Теперь рендер спрашивает камеру: куда падает точка и
    откуда пустить луч. Изометрия, фасад и вид сверху отличаются только этими
    двумя ответами.

    `ground` есть не у всякой камеры: тень на пол кладут те, кто этот пол
    видит. У фасада плоскость z=0 вырождается в линию, и тень там — забота
    сцены, а не рендера.
    """

    name: str
    view: np.ndarray
    ground: bool = True

    def to_screen(self, points: np.ndarray, px_per_m: float):
        """Мир → экран, в пикселях от начала координат кадра."""
        p = np.asarray(points, dtype=float)
        x, y, z = p[..., 0], p[..., 1], p[..., 2]
        if self.name == "iso":
            return (x - y) * px_per_m, (x + y) * px_per_m / 2 - z * px_per_m
        if self.name == "front":
            return x * px_per_m, -z * px_per_m
        if self.name == "top":
            return x * px_per_m, y * px_per_m
        raise KeyError(self.name)

    def to_world(self, sx: np.ndarray, sy: np.ndarray):
        """Экран → начало луча. Единицы экрана уже переведены в метры."""
        if self.name == "iso":
            wx = (sx + 2 * sy) / 2.0
            wy = (2 * sy - sx) / 2.0
            return np.stack([wx + BACK, wy + BACK,
                             np.full_like(wx, BACK)], axis=-1)
        if self.name == "front":
            return np.stack([sx, np.full_like(sx, -BACK), -sy], axis=-1)
        if self.name == "top":
            return np.stack([sx, sy, np.full_like(sx, BACK)], axis=-1)
        raise KeyError(self.name)

    def plane(self, sx: np.ndarray, sy: np.ndarray):
        """Точка на плоскости z=0 под экранным пикселем — для теней."""
        if not self.ground:
            raise ValueError(f"камера {self.name} не видит плоскости земли")
        if self.name == "iso":
            return (sx + 2 * sy) / 2.0, (2 * sy - sx) / 2.0
        return sx, sy          # top


CAMERAS: dict[str, Camera] = {
    "iso": Camera("iso", VIEW),
    # Фасад: X вправо, Z вверх, взгляд вдоль +Y вглубь сцены. Ближе к зрителю
    # то, у чего y меньше.
    "front": Camera("front", np.array([0.0, 1.0, 0.0]), ground=False),
    # Строго сверху: X вправо, Y вниз по экрану, взгляд вниз.
    "top": Camera("top", np.array([0.0, 0.0, -1.0])),
}


def camera_of(camera) -> Camera:
    """Камера по имени или готовым объектом."""
    if isinstance(camera, Camera):
        return camera
    if camera not in CAMERAS:
        raise KeyError(f"неизвестная камера {camera!r}; "
                       f"есть {', '.join(sorted(CAMERAS))}")
    return CAMERAS[camera]


@dataclass
class Material:
    """Как красить поверхность."""

    ramp: str = "hull"
    ambient: float = 0.34
    gloss: float = 0.0  # доля зеркального блика
    emissive: bool = False  # светится сам, темноту не боится
    # Фактура самого материала. Стекло, кирпич или штукатурка выглядят
    # одинаково везде, где встречаются, и расставлять узор у каждого тела —
    # работа без конца: в одном доме полторы сотни окон. Узор тела, если он
    # задан, старше материала.
    detail: str = ""
    detail_scale: float = 1.0
    # Сдвиг узора в метрах. Нужен там, где одно и то же тело повторяется
    # много раз: секции ряда, вагоны, панели. Без сдвига у всех копий узор
    # совпадает пиксель в пиксель, и стена читается обоями.
    detail_offset: tuple[float, float] = (0.0, 0.0)


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
    # Тень от тела к телу считается перебором каждого против всех, и это
    # квадрат по числу тел. У накладок, лежащих в плоскости грани — наличников,
    # поясов, переплётов, — собственная тень всё равно не видна, а в переборе
    # они дают основной вес: у дома в полторы сотни тел рендер уходит с секунд
    # на минуты. Такие тела помечаются здесь и из перебора выпадают.
    casts_shadow: bool = True


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


def _hash01(*args):
    """Повторяемый шум из целых номеров: 0…1, одинаковый между кадрами."""
    acc = 0.0
    for i, a in enumerate(args):
        acc = acc + a * (12.9898 + i * 27.317)
    h = np.sin(acc) * 43758.5453
    return h - np.floor(h)


def surface_detail(part, p_local, normal, scale=1.0, kind=None):
    """Узор на поверхности: возвращает поправку к яркости.

    Швы панелей, рёбра жёсткости, ряды люков и решётки рисуются не кистью, а
    по координатам самой поверхности. Поэтому они ложатся по форме и не едут
    при смене размеров.

    Узоры складываются: `detail="brick+grime"` кладёт кладку и поверх неё
    затемнение к цоколю. Так материал получает и рисунок, и возраст, не заводя
    отдельного тела и не плодя новых имён под каждое сочетание.
    """
    kind = kind if kind is not None else part.detail
    if not kind:
        return 0.0
    off = getattr(part.material, "detail_offset", (0.0, 0.0))
    if "+" in kind:
        total = 0.0
        for one in kind.split("+"):
            total = total + _detail_one(part, one.strip(), p_local, scale, off,
                                        normal)
        return total
    return _detail_one(part, kind, p_local, scale, off, normal)


def _detail_one(part, kind, p_local, scale=1.0, offset=(0.0, 0.0),
                normal=None):
    s = max(part.detail_scale, 1e-3) * scale
    # Сдвиг узора: одно и то же тело, повторённое в ряду, обязано выглядеть
    # по-разному, иначе десять секций складываются в обои с шагом в секцию.
    x = p_local[..., 0] + offset[0]
    y = p_local[..., 1]
    z = p_local[..., 2] + offset[1]
    # Угловая координата годится телу вращения, но не коробке. У цилиндра шаг
    # по углу и есть шаг по поверхности; у коробки размером в четыре метра
    # угол через всю грань меняется на четверть оборота, и «кирпич» шириной
    # 25 см растягивается на полстены. Кладка на соседнем доме выходила из
    # блоков по сорок пикселей.
    #
    # Поэтому у коробки узор идёт по ЛИНЕЙНОЙ координате вдоль видимой грани:
    # та из осей, что не смотрит в камеру, и есть направление ряда. Масштаб
    # приводится к тому же шагу, что у угловой ветки (π/7 ≈ 0.45 м), чтобы
    # старые числа detail_scale не поехали.
    if getattr(part, "kind", "") == "box":
        u_lin = np.where(np.abs(x) > np.abs(y), y, x)
        ang = u_lin * (np.pi / 7.0) / 0.45
    else:
        ang = np.arctan2(y, x)
    # Горизонтальная грань коробки узором стены не покрывается. Выбор оси по
    # `|x| > |y|` меняется по диагоналям, и на верхней грани кладка сходилась
    # крестом из четырёх секторов — самый заметный артефакт на крышах.
    flat_top = None
    if normal is not None and getattr(part, "kind", "") == "box":
        flat_top = np.abs(np.asarray(normal)[..., 2]) > 0.7

    def wall(value):
        """Настенный узор гасится на горизонтальной грани коробки."""
        if flat_top is None:
            return value
        return np.where(flat_top, 0.0, value)

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
    if kind == "brick":
        # Кладка. Ровная сетка даёт пластик: настоящую стену держит разнотон
        # отдельных кирпичей, а не швы — на девяти пикселях в метре шов и так
        # занимает полпикселя. Ряды со смещением через один, тон кирпича берётся
        # из его номера, поэтому картинка повторяема между кадрами дня и ночи.
        row_h = 0.30 * s
        brick_w = 0.62 * s
        row = np.floor(z / row_h)
        u = np.where(np.abs(x) > np.abs(y), y, x)      # ось вдоль стены
        offset = (row % 2) * brick_w * 0.5
        col = np.floor((u + offset) / brick_w)
        seam_z = np.abs((z / row_h % 1.0) - 0.5) > 0.42
        seam_u = np.abs((((u + offset) / brick_w) % 1.0) - 0.5) > 0.44
        h = np.sin(row * 12.9898 + col * 78.233) * 43758.5453
        tone = (h - np.floor(h)) - 0.5                  # −0.5…0.5, повторяемо
        # Амплитуда считается от ширины ступени палитры, а не «на глаз». При
        # четырёх ступенях ступень занимает четверть диапазона, и поправка в
        # 0,06 переключает тон только там, где значение и так сидит у границы:
        # на теневой грани кладка видна, а на освещённой выгорает в заливку.
        # Отсюда швы в пятую часть диапазона и редкий тёмный кирпич.
        # Шаг между соседними тонами шкалы — треть диапазона, поэтому разнотон
        # кирпича считается в тех же долях: при 0,06 стена меняла тон только у
        # границы ступени, то есть на теневой грани, а на освещённой оставалась
        # ровной заливкой. Пёстрым это не выглядит: ступеней всего четыре, и
        # разброс попадает в соседнюю, а не через одну.
        dark_one = _hash01(row, col, 4.0) > 0.88
        return wall(np.where(seam_z, -0.26,
                             np.where(seam_u, -0.17,
                                      np.where(dark_one, -0.42,
                                               tone * 0.52))))
    if kind == "clapboard":
        # Обшивка доской: тень под каждой доской, а не симметричное ребро.
        # Симметричная полоса читается гофрой, у доски свет ложится сверху и
        # уходит в тень у нижней кромки.
        t = (z / (0.42 * s)) % 1.0
        u = np.where(np.abs(x) > np.abs(y), y, x)
        # Разнотон доски считается от ШАГА СТУПЕНИ палитры. При 0,09 на
        # семиступенчатой шкале (шаг 0,167) две трети площади доски получали
        # поправку вчетверо меньше шага — стена оставалась ровной заливкой
        # при честно посчитанной фактуре. Доска не бывает одного тона с
        # соседней: партия пилится из разных брёвен.
        board = (_hash01(np.floor(z / (0.42 * s)),
                         np.floor(u / (2.4 * s))) - 0.5) * 0.40
        # продольная волокнистость внутри доски
        grain = (_hash01(np.floor(z / (0.42 * s)),
                         np.floor(u / (0.18 * s)), 7.0) - 0.5) * 0.10
        return wall(np.where(t < 0.16, -0.26,
                             np.where(t < 0.32, 0.16, board + grain)))
    if kind == "shingles":
        # Черепица: ряды со сдвигом и тенью под кромкой.
        row_h = 0.34 * s
        row = np.floor(z / row_h)
        u = np.where(np.abs(x) > np.abs(y), y, x)
        col = np.floor((u + (row % 2) * 0.5 * 0.7 * s) / (0.7 * s))
        t = (z / row_h) % 1.0
        h = np.sin(row * 27.13 + col * 91.7) * 21374.13
        tone = (h - np.floor(h)) - 0.5
        return wall(np.where(t < 0.20, -0.19, 0.04 + tone * 0.12))
    if kind == "stucco":
        # Штукатурка. Держится не рисунком, а разнотоном: ровная заливка на
        # большой плоскости — главный источник мёртвой картинки. Крап мелкий,
        # клетками по две трети метра, плюс широкая пологая волна, чтобы стена
        # не выглядела равномерно засыпанной точками.
        # Клетка крапа — 22 см, то есть семь пикселей на игровом масштабе.
        # При 46 см зерно выходило пятнами в полметра: стена читалась не
        # штукатуркой, а разводами сырости.
        cell = 0.22 * s
        # По вертикали клетка КРУПНЕЕ: изометрия сжимает высоту вдвое, и
        # зерно, равное по осям, выходит на экране вытянутыми штрихами —
        # стена читается не штукатуркой, а дождём по стеклу.
        gx = np.floor(x / cell)
        gy = np.floor(y / cell)
        gz = np.floor(z / (cell * 2.0))
        grit = _hash01(gx, gy, gz) - 0.5
        wave = np.sin(x * 0.7 + z * 0.45) * np.cos(y * 0.6 - z * 0.3)
        # Крап считается от ШАГА СТУПЕНИ, а не от абсолютной величины. На
        # шкале из трёх ступеней шаг равен трети диапазона, и поправка в
        # 0,085 не переключала тон вовсе: стена оставалась ровной заливкой
        # при честно посчитанной фактуре. На шкале из семи ступеней шаг
        # 0,14 — крап обязан быть с ним сопоставим.
        # Волна держится МАЛОЙ: на шкале из семи ступеней она переключает
        # тон полосами, и штукатурка читается досками, уложенными наискось.
        # Работает крап — клеточное зерно, оно и есть фактура штукатурки.
        return grit * 0.17 + wave * 0.015
    if kind == "concrete":
        # Сборный бетон: панели со швами, у каждой свой тон, по низу шва —
        # подтёк. Без разнотона панелей шов читается сеткой, наклеенной на
        # ровное поле.
        pw, ph = 1.55 * s, 1.20 * s
        u = np.where(np.abs(x) > np.abs(y), y, x)
        col = np.floor(u / pw)
        row = np.floor(z / ph)
        tone = (_hash01(col, row) - 0.5) * 0.13
        seam_u = np.abs((u / pw % 1.0) - 0.5) > 0.470
        seam_z = np.abs((z / ph % 1.0) - 0.5) > 0.455
        drip = (z / ph % 1.0 < 0.16) & (_hash01(col, row, 3.0) > 0.62)
        grit = (_hash01(np.floor(u / (0.3 * s)), np.floor(z / (0.3 * s))) - 0.5)
        base = tone + grit * 0.03
        return np.where(seam_u | seam_z, -0.22,
                        np.where(drip, base - 0.09, base))
    if kind == "roofing":
        # Рулонная кровля: полосы внахлёст, тёмная кромка нахлёста, редкие
        # заплаты посветлее и общий крап. Плоская кровля в изометрии занимает
        # до половины силуэта — без фактуры это самая мёртвая часть кадра.
        band = 0.95 * s
        u = x + y * 0.35
        row = np.floor(u / band)
        t = (u / band) % 1.0
        patch = _hash01(row, np.floor((x - y) / (1.7 * s)))
        grit = (_hash01(np.floor(x / (0.34 * s)), np.floor(y / (0.34 * s))) - 0.5)
        out = grit * 0.10 + (_hash01(row, 7.0) - 0.5) * 0.09
        out = np.where(t < 0.13, out - 0.16, out)
        return np.where(patch > 0.86, out + 0.13, out)
    if kind == "seam":
        # Фальцевая кровля: частые стоячие швы вдоль ската и лёгкая разница
        # тона между картинами.
        step = 0.52 * s
        u = x - y
        t = np.abs((u / step % 1.0) - 0.5)
        tone = (_hash01(np.floor(u / step), 1.0) - 0.5) * 0.05
        return np.where(t > 0.42, 0.14, np.where(t > 0.36, -0.10, tone))
    if kind == "stone":
        # Бутовая кладка: блоки разного размера, глубокий шов, сильный
        # разнотон. Ряды сдвигаются на случайную долю, иначе выходит кирпич.
        row_h = 0.46 * s
        row = np.floor(z / row_h)
        u = np.where(np.abs(x) > np.abs(y), y, x)
        width = (0.62 + _hash01(row, 5.0) * 0.5) * s
        shift = _hash01(row, 9.0) * width
        col = np.floor((u + shift) / width)
        seam_z = np.abs((z / row_h % 1.0) - 0.5) > 0.40
        seam_u = np.abs((((u + shift) / width) % 1.0) - 0.5) > 0.42
        tone = (_hash01(row, col) - 0.5) * 0.16
        return wall(np.where(seam_z | seam_u, -0.20, tone))
    if kind == "metal":
        # Профнастил: частые рёбра с бликом на гребне и тенью в ложбине.
        step = 0.22 * s
        t = (z / step) % 1.0
        return np.where(t < 0.22, 0.11, np.where(t < 0.5, -0.07, 0.0))
    if kind == "whitewash":
        # Побелка. Её кладут кистью снизу вверх, слой за слоем, и стена
        # никогда не бывает ровной: видны широкие мазки, места, где кисть
        # прошла дважды, и просвечивающая кладка под тонким слоем.
        #
        # Узор считается по ширине и высоте, а не по глубине: анфас глубина
        # у всей стены одна, и любой замешанный на ней рисунок вырождается в
        # горизонтальные полосы. На этом я уже обжёгся с булыжником.
        u = x
        stroke = np.sin(z * 3.1 / s + np.sin(u * 1.9 / s) * 2.2)
        broad = np.sin(u * 0.62 / s + 1.1) * np.cos(z * 0.44 / s + 0.4)
        cell = 0.30 * s
        grit = _hash01(np.floor(u / cell), np.floor(z / cell)) - 0.5
        # Пятна затирки: крупные, редкие, чуть светлее фона.
        blot = _hash01(np.floor(u / (1.3 * s)), np.floor(z / (1.1 * s)))
        # Проступающая кладка там, где слой тоньше.
        block = ((np.floor(z / (0.62 * s)) * 7 + np.floor(u / (1.1 * s))) % 5) == 0
        thin = (broad > 0.45) & block
        # Крап держат ниже половины ступени палитры: выше — и стена читается
        # не побелкой, а камуфляжной раскраской. Это ровно то, что случилось
        # на первом заходе с усиленной фактурой.
        return (stroke * 0.035 + broad * 0.06 + grit * 0.055
                + (blot - 0.5) * 0.045 - thin * 0.07)
    if kind == "rubble":
        # Бутовая кладка: камни разного размера, между ними раствор. Ряды не
        # держат линию, поэтому решётка смещается от ряда к ряду, а высота
        # ряда гуляет — от ровных горизонталей кладка выглядит кирпичной.
        u = x
        band = 0.40 * s
        row = np.floor(z / band + np.sin(u * 0.7 / s) * 0.18)
        shift = _hash01(row, 3.0) * 0.9
        wide = 0.42 * s + _hash01(row, 8.0) * 0.28 * s
        col = np.floor(u / wide + shift)
        tone = (_hash01(col, row) - 0.5) * 0.17
        du = np.abs(((u / wide + shift) % 1.0) - 0.5)
        dz = np.abs(((z / band + np.sin(u * 0.7 / s) * 0.18) % 1.0) - 0.5)
        mortar = (du > 0.44) | (dz > 0.43)
        # Скол на камне: светлая грань там, где откололся кусок.
        chip = _hash01(col + 4.0, row + 6.0) > 0.88
        return np.where(mortar, -0.15, tone + chip * 0.10)
    if kind == "cobble":
        # Булыжник: округлые камни со своим тоном и светлой маковкой, между
        # ними тёмный шов. Клетка считается по ширине и высоте — так узор
        # работает и на мостовой, видимой с торца, и на подпорной стенке.
        cell = 0.30 * s
        gx = np.floor(x / cell)
        gz = np.floor(z / cell)
        shift = _hash01(gz, 2.0) * 0.5
        gx = np.floor(x / cell + shift)
        jx = (_hash01(gx, gz) - 0.5) * 0.35
        jz = (_hash01(gx + 3.0, gz + 7.0) - 0.5) * 0.35
        dx = ((x / cell + shift) % 1.0) - 0.5 + jx
        dz = ((z / cell) % 1.0) - 0.5 + jz
        r = np.sqrt(dx * dx + dz * dz * 1.4)
        tone = (_hash01(gx + 5.0, gz + 9.0) - 0.5) * 0.13
        return np.where(r > 0.40, -0.14, tone + np.clip(0.28 - r, 0, 1) * 0.26)
    if kind == "chipped":
        # Крашеное дерево, отслужившее своё: краска сходит пятнами, сильнее
        # внизу — там её бьёт водой и песком, — и вдоль волокон доски.
        half = max(float(np.array(part.size, dtype=float)[2]) * 0.5, 1e-3)
        t = np.clip((z + half) / (2 * half), 0.0, 1.0)
        grain = np.sin(z * 11.0 / s
                       + _hash01(np.floor(x / (0.08 * s)), 4.0) * 6.0)
        cell = 0.13 * s
        spots = _hash01(np.floor(x / cell), np.floor(z / cell))
        wear = spots > (0.86 + t * 0.10)
        edge = _hash01(np.floor(x / (0.05 * s)), np.floor(z / (0.24 * s)))
        return grain * 0.035 + wear * 0.16 + (edge - 0.5) * 0.03
    if kind == "terracotta":
        # Обожжённая глина: пористая, с редкими выщербинами и следом от
        # гончарного круга — кольцами поперёк формы.
        # Амплитуды приведены к шагу ступени: при 0,045 и 0,075 кольца и
        # поры не переключали тон вовсе, и горшок выходил ровным пятном.
        rings = np.sin(z * 7.0 / s) * 0.5 + 0.5
        pore = _hash01(np.floor(x / (0.10 * s)), np.floor(z / (0.10 * s)))
        chip = _hash01(np.floor(x / (0.22 * s)), np.floor(z / (0.22 * s)), 3.0)
        return ((rings - 0.5) * 0.13 + (pore - 0.5) * 0.16
                + np.where(chip > 0.93, -0.20, 0.0))
    if kind == "grime":
        # Возраст: потемнение к низу тела и потёки. Высота берётся у самого
        # тела, поэтому один и тот же узор работает и на цоколе, и на башне.
        half = max(float(np.array(part.size, dtype=float)[2]) * 0.5, 1e-3)
        t = np.clip((z + half) / (2 * half), 0.0, 1.0)
        low = np.clip(1.0 - t * 3.2, 0.0, 1.0) ** 2
        u = np.where(np.abs(x) > np.abs(y), y, x)
        streak = _hash01(np.floor(u / (0.42 * s)), 11.0)
        drip = np.clip(1.0 - t * 1.9, 0.0, 1.0) * (streak > 0.74)
        # Грязь держится слабой: она складывается с фактурой материала, и
        # вместе они уводят всю стену на ступень вниз — кладка снова ровнеет.
        return -low * 0.07 - drip * 0.035
    if kind == "glazing":
        # Остекление. Плоское стекло — самая мёртвая поверхность в кадре: у
        # него нет ни рельефа, ни собственного тона, только заливка. Работают
        # три вещи, и все три обязаны быть слабее шага палитры, иначе окно
        # перестаёт читаться окном:
        #   отражение неба — верх светлее низа;
        #   косой блик — узкая полоса под тем же углом, что изометрия;
        #   расстекловка — тонкие тёмные линии, деление на створки.
        # Плюс на каждое четвёртое окно опускаются жалюзи, и это единственное,
        # что делает ряд одинаковых проёмов жилым домом, а не сеткой.
        size = np.array(part.size, dtype=float)
        half_z = max(size[2] * 0.5, 1e-3)
        t = np.clip((z + half_z) / (2 * half_z), 0.0, 1.0)
        u = np.where(np.abs(x) > np.abs(y), y, x)
        half_u = max(min(size[0], size[1]) * 0.0 + max(size[0], size[1]) * 0.5,
                     1e-3)

        sky = (t - 0.45) * 0.20                       # отражение неба
        diag = (u * 0.85 + z * 1.0) / max(0.9 * s, 1e-3)
        band = np.abs((diag % 1.0) - 0.5)
        streak = np.where(band > 0.44, 0.16, np.where(band > 0.38, 0.07, 0.0))

        # расстекловка: одна вертикальная и одна горизонтальная линия
        vert = np.abs(u) < 0.055 * max(half_u, 0.5)
        horz = np.abs(t - 0.62) < 0.035
        mullion = np.where(vert | horz, -0.22, 0.0)

        # у каждого окна свой характер: жалюзи, штора или чистое стекло
        pos = np.array(part.pos, dtype=float)
        pick = _hash01(np.floor(pos[0] * 3.1), np.floor(pos[2] * 2.7),
                       np.floor(pos[1] * 1.9))
        blinds = np.where((t > 0.30) & (((z / (0.19 * s)) % 1.0) < 0.45),
                          -0.13, 0.05)
        curtain = np.where(t < 0.52, 0.11, -0.04)
        pattern = blinds if pick > 0.74 else (curtain if pick > 0.52 else 0.0)

        return sky + streak + mullion + pattern
    if kind == "fade":
        # Выцветание к верху: слабее грязи и всегда в плюс.
        half = max(float(np.array(part.size, dtype=float)[2]) * 0.5, 1e-3)
        t = np.clip((z + half) / (2 * half), 0.0, 1.0)
        return np.clip(t - 0.55, 0.0, 1.0) * 0.09
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


def _part_half(part: Part) -> np.ndarray:
    """Полугабарит тела в его локальных осях."""
    s = np.array(part.size, dtype=float)
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
    return half * np.array(part.scale, dtype=float)


def _screen_window(part: Part, px_per_m: float, origin, width: int, height: int,
                   camera: Camera = CAMERAS["iso"]):
    """Окно кадра, в котором тело вообще может оказаться.

    Без него каждое тело трассируется по всей картинке, хотя занимает её малую
    часть: у дома в полторы сотни тел это уводит рендер на минуту. Окно берётся
    по восьми углам повёрнутого габарита с запасом в два пикселя.
    """
    c = np.array(part.pos, dtype=float)
    half = _part_half(part)
    rot, _ = _prepare(part)
    corners = np.array([c + rot @ (half * np.array([dx, dy, dz]))
                        for dx in (-1, 1) for dy in (-1, 1) for dz in (-1, 1)])
    px, py = camera.to_screen(corners, px_per_m)
    xs = px + origin[0] * width
    ys = py + origin[1] * height
    pad = 2
    x0 = max(0, int(np.floor(min(xs))) - pad)
    x1 = min(width, int(np.ceil(max(xs))) + pad)
    y0 = max(0, int(np.floor(min(ys))) - pad)
    y1 = min(height, int(np.ceil(max(ys))) + pad)
    if x0 >= x1 or y0 >= y1:
        return None
    return slice(y0, y1), slice(x0, x1)


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
    light: tuple[float, float, float] | None = None,
    camera: "str | Camera" = "iso",
) -> Canvas:
    """Собрать спрайт из тел.

    Луч переводится в систему каждого тела, поэтому наклон, поворот и
    вытягивание работают для всех примитивов одинаково.

    `light` задаёт свой источник вместо общего. Он нужен там, где предмет
    обязан читаться сам по себе: у постройки свет сцены освещает верхние
    грани и этого хватает, а у фигуры, стоящей лицом к зрителю, лицо при том
    же свете уходит в ровную заливку рассеянным — объём пропадает целиком.
    """
    cam = camera_of(camera)
    light_v = LIGHT if light is None else np.array(light, dtype=float)
    light_v = light_v / np.linalg.norm(light_v)
    canvas = Canvas(width, height, palette)
    canvas.data[:, :] = TRANSPARENT

    yy, xx = np.mgrid[0:height, 0:width]
    sx = (xx - origin[0] * width).astype(float)
    sy = (yy - origin[1] * height).astype(float)

    k = px_per_m
    o = cam.to_world(sx / k, sy / k)
    d = cam.view

    best_t = np.full((height, width), np.inf)
    best_idx = np.full((height, width), -1, dtype=int)

    prepared = []
    for i, part in enumerate(parts):
        c = np.array(part.pos, dtype=float)
        rot, scale = _prepare(part)
        win = _screen_window(part, px_per_m, origin, width, height, cam)
        if win is None:
            prepared.append(None)
            continue
        ys, xs_ = win
        o_w = o[ys, xs_]
        ol = np.einsum("ij,...j->...i", rot.T, o_w - c) / scale
        dl = (rot.T @ d) / scale
        t = _hit_local(part, ol, dl)

        if part.cut_below is not None:
            p_world = o_w + np.where(np.isfinite(t), t, 0.0)[..., None] * d
            t = np.where(p_world[..., 2] >= part.cut_below, t, np.inf)

        prepared.append((part, c, rot, scale, win))
        sub = best_t[ys, xs_]
        closer = t < sub
        best_t[ys, xs_] = np.where(closer, t, sub)
        best_idx[ys, xs_] = np.where(closer, i, best_idx[ys, xs_])

    solid = np.isfinite(best_t)
    if not solid.any():
        return canvas

    t_safe = np.where(solid, best_t, 0.0)
    point = o + t_safe[..., None] * d

    for i, entry in enumerate(prepared):
        if entry is None:
            continue
        part, c, rot, scale, (ys, xs_) = entry
        mask_w = solid[ys, xs_] & (best_idx[ys, xs_] == i)
        if not mask_w.any():
            continue

        o_w = o[ys, xs_]
        t_w = t_safe[ys, xs_]
        ol = np.einsum("ij,...j->...i", rot.T, o_w - c) / scale
        dl = (rot.T @ d) / scale
        pl = ol + t_w[..., None] * dl
        nl = _normal_local(part, pl)
        n = np.einsum("ij,...j->...i", rot, nl / scale)
        n = n / np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-9)

        lambert = np.clip((n * light_v).sum(axis=-1), 0, 1)
        mat = part.material
        value = mat.ambient + (1.0 - mat.ambient) * lambert

        if mat.gloss > 0:
            half = light_v + VIEW
            half /= np.linalg.norm(half)
            spec = np.clip((n * half).sum(axis=-1), 0, 1) ** 24
            value = np.clip(value + spec * mat.gloss, 0, 1)

        kind = part.detail or mat.detail
        if kind:
            value = value + surface_detail(part, pl, nl, kind=kind)

        if shadows and not mat.emissive:
            lit = np.ones_like(value)
            start = o_w + t_w[..., None] * d + n * 0.03
            for other_entry in prepared:
                if other_entry is None:
                    continue
                other, oc2, rot2, sc2, _ = other_entry
                if other is part or not other.casts_shadow:
                    continue
                ol2 = np.einsum("ij,...j->...i", rot2.T, start - oc2) / sc2
                dl2 = (rot2.T @ light_v) / sc2
                ts = _hit_local(other, ol2, dl2)
                lit = np.where(np.isfinite(ts), 0.58, lit)
            value = value * lit

        if mat.emissive:
            lit_value = np.full_like(value, 0.92)
            if kind:
                lit_value = lit_value + surface_detail(part, pl, nl, kind=kind) * 0.7
            value = lit_value

        tmp = Canvas(value.shape[1], value.shape[0], palette)
        shade_from_field(
            tmp, np.clip(value, 0, 1), palette.ramp(mat.ramp),
            dither=dither, matrix=BAYER4, noise=grain, seed=seed + i * 7,
        )
        sub = canvas.data[ys, xs_]
        canvas.data[ys, xs_] = np.where(mask_w, tmp.data, sub)

    return canvas


def ground_shadow(canvas: Canvas, parts: list[Part], px_per_m: float = 8.0,
                  origin: tuple[float, float] = (0.5, 0.85),
                  index: int | None = None,
                  camera: "str | Camera" = "iso") -> None:
    """Положить под модуль тень на землю: проекция тел на плоскость z=0.

    Камера обязана эту плоскость видеть. У фасада она вырождается в линию, и
    тень там рисует сцена — полосой под домом, а не проекцией тел.
    """
    cam = camera_of(camera)
    palette = canvas.palette
    idx = index if index is not None else palette.index_of("shadow")
    h, w = canvas.data.shape
    yy, xx = np.mgrid[0:h, 0:w]
    sx = (xx - origin[0] * w) / px_per_m
    sy = (yy - origin[1] * h) / px_per_m
    wx, wy = cam.plane(sx, sy)

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


def screen_bounds(parts: list[Part], px_per_m: float,
                  camera: "str | Camera" = "iso") -> tuple[int, int, float, float]:
    """Подобрать размер спрайта под габариты тел.

    Углы описывающей коробки поворачиваются вместе с телом, а не заменяются
    шаром по диагонали: иначе у фигуры с наклонёнными конечностями кадр
    раздувается вдвое и вокруг остаётся пустота.
    """
    cam = camera_of(camera)
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
        corners = np.array([c + rot @ (half * np.array([dx, dy, dz]))
                            for dx in (-1, 1) for dy in (-1, 1)
                            for dz in (-1, 1)])
        px, py = cam.to_screen(corners, px_per_m)
        xs.extend(px.tolist())
        ys.extend(py.tolist())
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
