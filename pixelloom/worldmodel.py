"""Предметы в мировых координатах для изометрии 2:1: ящик, цилиндр, бревно, шар, стержень.

Родом из meadow-blocks (паки блоков 32×32). Точка мира (u, v, h) ложится на экран
как у блока: x = u − v + origin_x, y = (u + v)/2 − h + origin_y. Луч зрения идёт
вдоль (1, 1, 1), поэтому в пикселе остаётся точка с наибольшим u + v + h.

Свет — сдвиг ступени шкалы по нормали (C12), откалиброванный под грани блока:
верх на своей ступени, грань +v на одну темнее, грань +u на две. Предмет и тайл
так стоят в одном мире.

Рендер держит грабли, на которые наступили при рисовании 55 предметов:
- кайма по силуэту самой тёмной ступенью, но тонкая вытянутая линия (≤ 2 px
  поперёк) каймы не получает — иначе жердь целиком уходит в чёрное (B09);
- одиночки чистятся дважды: в своём теле до каймы и по всем соседям после;
  тела-детали (рамка ящика) объявляются protect(), их чистка не трогает (D18);
- шар строится сеткой по широте и долготе, а не случайными точками (D17);
- clip и world_u/v/h дают резать предмет больше клетки на куски уже ПОСЛЕ
  рендера: куски, рендеренные порознь, получают шов каймы на стыке (G33).

Палитра — любой объект с .index(ramp, step), .length[ramp], .colors; готовая —
RampPalette.
"""
from __future__ import annotations

import numpy as np

SAMPLE = 0.25
SHADOW_RGBA = (20, 18, 34, 96)
VIEW = np.array([1.0, 1.0, 1.0]) / np.sqrt(3)
# свет сверху-слева в мире (u, v, h): даёт верх 0, грань +v −1, грань +u −2
DEFAULT_LIGHT = np.array([-0.2, 0.35, 0.9]) / np.linalg.norm([-0.2, 0.35, 0.9])


def light_shift(normal, light=DEFAULT_LIGHT) -> np.ndarray:
    """Сдвиг ступени по нормали: верх 0, грань +v −1, грань +u −2, как у блока."""
    flat, side = light[2], light[1]
    return (np.asarray(normal) @ light - flat) / (flat - side)


def hex_to_rgb(code: str) -> tuple[int, int, int]:
    code = code.lstrip("#")
    return tuple(int(code[i:i + 2], 16) for i in (0, 2, 4))


class RampPalette:
    """Плоский список цветов из шкал {имя: [hex от тёмного к светлому]}."""

    def __init__(self, ramps: dict[str, list[str]]):
        self.colors: list[tuple[int, int, int]] = []
        self.start: dict[str, int] = {}
        self.length: dict[str, int] = {}
        for name, codes in ramps.items():
            self.start[name] = len(self.colors)
            self.length[name] = len(codes)
            self.colors.extend(hex_to_rgb(c) for c in codes)

    def index(self, ramp: str, step: int) -> int:
        """Ступень зажимается в пределы шкалы: сдвиги света не вылетают за край."""
        step = max(0, min(self.length[ramp] - 1, int(step)))
        return self.start[ramp] + step


class Model:
    def __init__(self, width: int = 32, height: int = 40, palette=None, light=DEFAULT_LIGHT):
        if palette is None:
            raise ValueError("Model нужна палитра шкал: RampPalette или совместимая")
        self.palette = palette
        self.light = np.asarray(light, dtype=float)
        self.width, self.height = width, height
        self.depth = np.full((height, width), -np.inf)
        self.ramp = np.full((height, width), "", dtype=object)
        self.step = np.zeros((height, width))
        self.part = np.full((height, width), -1)
        self.fixed = np.zeros((height, width), dtype=bool)
        self.shadow = np.zeros((height, width), dtype=bool)
        self.parts = 0
        self.protected = set()
        # окно (u0, u1, v0, v1): точки вне него в кадр не попадают
        self.clip = None
        self.casts = True
        # начало координат на экране: по умолчанию центр клетки, ромб клетки внизу кадра
        self.origin_x = width / 2
        self.origin_y = height - 16
        # точка мира, оставшаяся в пикселе: по ней большой кадр режется на клетки
        self.world_u = np.zeros((height, width))
        self.world_v = np.zeros((height, width))
        self.world_h = np.zeros((height, width))

    def shift(self, normal) -> np.ndarray:
        return light_shift(normal, self.light)

    def protect(self, part: int) -> int:
        """Тело-деталь: последняя чистка его не трогает."""
        self.protected.add(part)
        return part

    def new_part(self) -> int:
        self.parts += 1
        return self.parts

    # ------------------------------------------------------------ основа

    def screen(self, u, v, h):
        x = np.floor(np.asarray(u) - v + self.origin_x).astype(int)
        y = np.floor((np.asarray(u) + v) / 2 - h + self.origin_y).astype(int)
        return x, y

    def splat(self, u, v, h, ramp, step, part=None, cast=True, fixed=False):
        """Точки мира в пиксели; в пикселе остаётся ближайшая к зрителю (u + v + h)."""
        part = self.new_part() if part is None else part
        u, v, h, step = (np.asarray(c, dtype=float).ravel()
                         for c in np.broadcast_arrays(u, v, h, np.asarray(step, dtype=float)))
        if self.clip is not None:
            u0, u1, v0, v1 = self.clip
            keep = (u >= u0) & (u < u1) & (v >= v0) & (v < v1)
            u, v, h, step = u[keep], v[keep], h[keep], step[keep]
        cast = cast and self.casts
        x, y = self.screen(u, v, h)
        depth = u + v + h
        inside = (x >= 0) & (x < self.width) & (y >= 0) & (y < self.height)
        for xi, yi, d, s, pu, pv, ph in zip(x[inside], y[inside], depth[inside], step[inside],
                                            u[inside], v[inside], h[inside]):
            if d > self.depth[yi, xi]:
                self.depth[yi, xi] = d
                self.ramp[yi, xi] = ramp
                self.step[yi, xi] = s
                self.part[yi, xi] = part
                self.fixed[yi, xi] = fixed
                self.world_u[yi, xi] = pu
                self.world_v[yi, xi] = pv
                self.world_h[yi, xi] = ph
        if cast:
            # тень: точку сносит по земле против света на высоту h (K05)
            gu = u - h * self.light[0] / self.light[2]
            gv = v - h * self.light[1] / self.light[2]
            sx, sy = self.screen(gu, gv, np.zeros_like(gu))
            ok = (sx >= 0) & (sx < self.width) & (sy >= 0) & (sy < self.height)
            self.shadow[sy[ok], sx[ok]] = True
        return part

    # ------------------------------------------------------------ тела

    def box(self, u0, u1, v0, v1, h0, h1, ramp, base, texture=None, part=None, cast=True):
        """Коробка: видны верх, грань +v (левая) и грань +u (правая).

        texture(face, a, b) → сдвиг ступени; face — top, left, right; a, b — координаты
        по грани: у верха (u, v), у левой (u, h), у правой (v, h).
        """
        part = self.new_part() if part is None else part
        tex = texture or (lambda face, a, b: 0)
        us = np.arange(u0 + SAMPLE / 2, u1, SAMPLE)
        vs = np.arange(v0 + SAMPLE / 2, v1, SAMPLE)
        hs = np.arange(h0 + SAMPLE / 2, h1, SAMPLE)
        uu, vv = np.meshgrid(us, vs, indexing="ij")
        self.splat(uu, vv, np.full_like(uu, h1), ramp, base + tex("top", uu, vv), part, cast)
        uu, hh = np.meshgrid(us, hs, indexing="ij")
        self.splat(uu, np.full_like(uu, v1), hh, ramp, base - 1 + tex("left", uu, hh), part, cast)
        vv, hh = np.meshgrid(vs, hs, indexing="ij")
        self.splat(np.full_like(vv, u1), vv, hh, ramp, base - 2 + tex("right", vv, hh), part, cast)
        return part

    def cylinder(self, cu, cv, radius, h0, h1, ramp, base, texture=None, part=None, top=True,
                 top_ramp=None, top_base=None, top_texture=None, cast=True):
        """Стоячий цилиндр: бок по нормали, крышка сверху. texture(angle, h) → сдвиг."""
        part = self.new_part() if part is None else part
        tex = texture or (lambda angle, h: 0)
        # видимая половина: нормаль (cos, sin) смотрит на зрителя, к +u+v — центр дуги π/4
        angles = np.arange(np.pi * 0.25 - np.pi / 2, np.pi * 0.25 + np.pi / 2 + 0.01,
                           SAMPLE / max(radius, 1))
        hs = np.arange(h0 + SAMPLE / 2, h1, SAMPLE)
        aa, hh = np.meshgrid(angles, hs, indexing="ij")
        normal = np.stack([np.cos(aa), np.sin(aa), np.zeros_like(aa)], axis=-1)
        step = base + self.shift(normal) + tex(aa, hh)
        self.splat(cu + radius * np.cos(aa), cv + radius * np.sin(aa), hh, ramp, step, part, cast)
        if top:
            self.disk(cu, cv, radius, h1, top_ramp or ramp, base if top_base is None else top_base,
                      top_texture, part, cast)
        return part

    def disk(self, cu, cv, radius, h, ramp, base, texture=None, part=None, cast=True,
             inner: float = 0.0):
        """Горизонтальный круг: крышка бочки, столешница; inner > 0 — кольцо обода."""
        part = self.new_part() if part is None else part
        tex = texture or (lambda u, v: 0)
        grid = np.arange(-radius + SAMPLE / 2, radius, SAMPLE)
        uu, vv = np.meshgrid(grid, grid, indexing="ij")
        ring = uu * uu + vv * vv
        inside = (ring <= radius * radius) & (ring >= inner * inner)
        uu, vv = uu[inside], vv[inside]
        self.splat(cu + uu, cv + vv, np.full_like(uu, h), ramp, base + tex(uu, vv), part, cast)
        return part

    def log(self, axis, start, end, cv, ch, radius, ramp, base, texture=None, part=None,
            cap_ramp=None, cap_base=None, cap_texture=None, cast=True, cap_mask=None):
        """Лежачий цилиндр вдоль u или v: бревно, лежачая бочка, колесо.

        cap_mask(a, b) → какие точки торца рисовать: у колеса между спицами пусто.
        """
        part = self.new_part() if part is None else part
        tex = texture or (lambda angle, t: 0)
        along = np.arange(start + SAMPLE / 2, end, SAMPLE)
        angles = np.linspace(0, 2 * np.pi, int(2 * np.pi * radius / SAMPLE) + 8, endpoint=False)
        tt, aa = np.meshgrid(along, angles, indexing="ij")
        across, up = np.cos(aa), np.sin(aa)
        if axis == "u":
            u, v, normal = tt, cv + radius * across, np.stack([0 * aa, across, up], axis=-1)
        else:
            u, v, normal = cv + radius * across, tt, np.stack([across, 0 * aa, up], axis=-1)
        h = ch + radius * up
        facing = normal @ VIEW > -0.05
        step = base + self.shift(normal) + tex(aa, tt)
        self.splat(u[facing], v[facing], h[facing], ramp, step[facing], part, cast)
        # торец, что смотрит на зрителя (+u или +v)
        grid = np.arange(-radius + SAMPLE / 2, radius, SAMPLE)
        aa2, bb2 = np.meshgrid(grid, grid, indexing="ij")
        inside = aa2 * aa2 + bb2 * bb2 <= radius * radius
        if cap_mask is not None:
            inside &= cap_mask(aa2, bb2)
        cap_tex = cap_texture or (lambda a, b: 0)
        cap_step = (base if cap_base is None else cap_base) - (2 if axis == "u" else 1) \
            + cap_tex(aa2[inside], bb2[inside])
        if axis == "u":
            cu, cvv = np.full(inside.sum(), end), cv + aa2[inside]
        else:
            cu, cvv = cv + aa2[inside], np.full(inside.sum(), end)
        self.splat(cu, cvv, ch + bb2[inside], cap_ramp or ramp, cap_step, part, cast)
        return part

    def sphere(self, cu, cv, ch, radius, ramp, base, squash=1.0, texture=None, part=None,
               cast=True):
        """Шар: тыква, капуста, ком сена. squash — сплющенность по высоте."""
        part = self.new_part() if part is None else part
        tex = texture or (lambda n: 0)
        # ровная сетка по широте и долготе: случайные точки давали дыры и рябь одиночек (D17)
        steps = max(12, int(np.pi * radius / (SAMPLE * 0.7)))
        lat = np.linspace(-np.pi / 2, np.pi / 2, steps)
        lon = np.linspace(0, 2 * np.pi, 2 * steps, endpoint=False)
        la, lo = np.meshgrid(lat, lon, indexing="ij")
        normal = np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)],
                          axis=-1).reshape(-1, 3)
        normal = normal[normal @ VIEW > -0.1]
        u = cu + radius * normal[:, 0]
        v = cv + radius * normal[:, 1]
        h = ch + radius * squash * normal[:, 2]
        shaded = normal.copy()
        shaded[:, 2] /= max(squash, 0.2)
        shaded /= np.linalg.norm(shaded, axis=1, keepdims=True)
        step = base + self.shift(shaded) + tex(normal)
        self.splat(u, v, h, ramp, step, part, cast)
        return part

    def rod(self, start, end, radius, ramp, base, part=None, cast=True):
        """Стержень от точки до точки мира: перекладина, древко, верёвка.

        Радиус 0,8 и больше даёт линию в два пикселя: в один пиксель линия по
        диагонали — цепочка одиночек (D04).
        """
        part = self.new_part() if part is None else part
        start, end = np.asarray(start, float), np.asarray(end, float)
        length = np.linalg.norm(end - start)
        count = max(2, int(length / SAMPLE))
        t = np.linspace(0, 1, count)
        points = start[None] + (end - start)[None] * t[:, None]
        offsets = [(0, 0, 0)]
        if radius > 0.5:
            offsets = [(dx, dy, dz) for dx in (-radius / 2, radius / 2)
                       for dy in (-radius / 2, radius / 2) for dz in (-radius / 2, radius / 2)]
        for dx, dy, dz in offsets:
            self.splat(points[:, 0] + dx, points[:, 1] + dy, points[:, 2] + dz, ramp, base,
                       part, cast)
        return part

    def paint(self, u, v, h, ramp, step):
        """Деталь поверх тела без своей глубины: глазок, шляпка гвоздя, ягода."""
        x, y = self.screen(np.asarray([u]), np.asarray([v]), np.asarray([h]))
        x, y = int(x[0]), int(y[0])
        if 0 <= x < self.width and 0 <= y < self.height:
            self.ramp[y, x] = ramp
            self.step[y, x] = step
            self.fixed[y, x] = True

    # ------------------------------------------------------------ рендер

    def neighbours(self, y, x):
        for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if 0 <= ny < self.height and 0 <= nx < self.width:
                yield ny, nx

    def thin_at(self, filled, y, x, limit: int = 2) -> bool:
        """Тонкая вытянутая линия: поперёк ≤ limit пикселей, вдоль длиннее."""
        def run(dy, dx):
            length, ny, nx = 0, y + dy, x + dx
            while 0 <= ny < self.height and 0 <= nx < self.width and filled[ny, nx] \
                    and length <= limit:
                length += 1
                ny, nx = ny + dy, nx + dx
            return length
        vertical = 1 + run(-1, 0) + run(1, 0)
        horizontal = 1 + run(0, -1) + run(0, 1)
        # короткий выступ в пиксель получает кайму, иначе кончик бруса торчит светлой точкой
        return (vertical <= limit and horizontal > limit) or \
            (horizontal <= limit and vertical > limit)

    def render(self, outline: bool = True) -> np.ndarray:
        palette = self.palette
        image = np.zeros((self.height, self.width, 4), dtype=np.uint8)
        filled = self.ramp != ""
        image[self.shadow & ~filled] = SHADOW_RGBA
        steps = np.zeros((self.height, self.width), dtype=int)
        for y, x in zip(*np.nonzero(filled)):
            top = palette.length[self.ramp[y, x]] - 1
            steps[y, x] = int(np.clip(np.round(self.step[y, x]), 0, top))

        # одиночка среди соседей своей шкалы и своего тела берёт их ступень
        source = steps.copy()
        for y, x in zip(*np.nonzero(filled & ~self.fixed)):
            around = [source[ny, nx] for ny, nx in self.neighbours(y, x)
                      if filled[ny, nx] and self.ramp[ny, nx] == self.ramp[y, x]
                      and self.part[ny, nx] == self.part[y, x]]
            if len(around) >= 3 and source[y, x] not in around:
                values, counts = np.unique(around, return_counts=True)
                steps[y, x] = values[np.argmax(counts)]

        # стык тел: пиксель заднего тела над передним темнее на ступень (V04)
        source = steps.copy()
        for y, x in zip(*np.nonzero(filled)):
            if y + 1 < self.height and filled[y + 1, x] and self.part[y + 1, x] != self.part[y, x] \
                    and self.depth[y + 1, x] > self.depth[y, x] + 1.5:
                steps[y, x] = max(0, source[y, x] - 1)

        if outline:
            # кромка силуэта по 8 соседям: по 4 на скате лесенкой она рвётся на одиночки
            final = steps.copy()
            for y, x in zip(*np.nonzero(filled)):
                border = y in (0, self.height - 1) or x in (0, self.width - 1)
                ring = [(y + dy, x + dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx]
                if not (border or any(not filled[ny, nx] for ny, nx in ring)):
                    continue
                if border or not self.thin_at(filled, y, x):
                    final[y, x] = 0
            steps = final

        # последняя чистка по всем соседям; защищённые тела-детали не трогает
        source = steps.copy()
        guarded = np.isin(self.part, list(self.protected))
        for y, x in zip(*np.nonzero(filled & ~self.fixed & ~guarded)):
            around = [(self.ramp[ny, nx], source[ny, nx]) for ny, nx in self.neighbours(y, x)
                      if filled[ny, nx]]
            here = (self.ramp[y, x], source[y, x])
            if len(around) >= 3 and here not in around:
                keys = [f"{r}:{s}" for r, s in around]
                values, counts = np.unique(keys, return_counts=True)
                ramp, step = values[np.argmax(counts)].split(":")
                self.ramp[y, x] = ramp
                steps[y, x] = int(step)
        colors = np.array(palette.colors, dtype=np.uint8)
        for y, x in zip(*np.nonzero(filled)):
            image[y, x, :3] = colors[palette.index(self.ramp[y, x], steps[y, x])]
            image[y, x, 3] = 255
        return image


def slice_by_cells(model: Model, image: np.ndarray, axis: str, cells: int, cell_height: int,
                   edge: float = 16.0) -> list[np.ndarray]:
    """Режет готовый кадр предмета на куски по клеткам вдоль оси u или v.

    Предмет больше клетки рендерится ЦЕЛИКОМ (кайма по настоящему силуэту), и лишь
    потом пиксель отдаётся клетке, в которой лежит его точка мира. Кадр клетки 0 —
    в левом верхнем углу (ось u) или в правом (ось v), как ставит origin_x.
    """
    along = model.world_u if axis == "u" else model.world_v
    owner = np.clip(np.floor(along / edge).astype(int), 0, cells - 1)
    extra = cells - 1
    pieces = []
    for cell in range(cells):
        x0 = 16 * cell if axis == "u" else 16 * (extra - cell)
        y0 = 8 * cell
        piece = image[y0:y0 + cell_height, x0:x0 + 32].copy()
        mine = owner[y0:y0 + cell_height, x0:x0 + 32] == cell
        piece[~mine] = 0
        pieces.append(piece)
    return pieces
