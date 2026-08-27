"""Чистильщик: превращает сырую сетку в пиксель-арт.

Форму даёт что угодно — рендер сцены, примитивы, обвод эскиза. На выходе
всегда одно и то же: одиночные пиксели, рваные ступеньки, полосы равной
ширины, кромка без обводки. Художник вычищает это руками, здесь то же самое
делают операторы с числами. Каждый оператор отвечает за одно правило графа
(D01, D03, D04, C04) и проверяется отдельным тестом.

Работает по индексам палитры: -1 — прозрачность.
"""
from __future__ import annotations

import numpy as np

NEIGH4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _shift(a: np.ndarray, dy: int, dx: int, fill=-2) -> np.ndarray:
    out = np.full_like(a, fill)
    h, w = a.shape
    ys = slice(max(0, dy), h + min(0, dy))
    xs = slice(max(0, dx), w + min(0, dx))
    ys_s = slice(max(0, -dy), h + min(0, -dy))
    xs_s = slice(max(0, -dx), w + min(0, -dx))
    out[ys, xs] = a[ys_s, xs_s]
    return out


def stray_mask(idx: np.ndarray, pal=None) -> np.ndarray:
    """Пиксели без соседа своего цвета по четырём сторонам (D04).

    Одинокий пиксель — мусор НЕ всегда: ровно так выглядит клетка ручного
    антиалиасинга, и она обязана быть одинокой. Поэтому при известной палитре
    из мусора исключаются пиксели, чей тон лежит между тонами соседей: они
    сглаживают ступеньку, а не шумят.
    """
    same = np.zeros(idx.shape, bool)
    for dy, dx in NEIGH4:
        same |= _shift(idx, dy, dx) == idx
    lonely = ~same & (idx >= 0)
    if pal is None or not lonely.any():
        return lonely
    lum = np.asarray(pal.colors, float) @ [0.299, 0.587, 0.114]
    ys, xs = np.nonzero(lonely)
    h, w = idx.shape
    for y, x in zip(ys, xs):
        vals = [int(idx[y + dy, x + dx]) for dy, dx in NEIGH4
                if 0 <= y + dy < h and 0 <= x + dx < w and idx[y + dy, x + dx] >= 0]
        if len(vals) < 2:
            continue
        here = lum[int(idx[y, x])]
        lo, hi = min(lum[v] for v in vals), max(lum[v] for v in vals)
        if lo - 1 <= here <= hi + 1 and hi - lo > 4:
            lonely[y, x] = False          # это антиалиасинг, а не мусор
    return lonely


def despeckle(idx: np.ndarray) -> np.ndarray:
    """Убрать одиночные пиксели, заменив цветом большинства соседей."""
    out = idx.copy()
    lonely = stray_mask(idx)
    if not lonely.any():
        return out
    ys, xs = np.nonzero(lonely)
    h, w = idx.shape
    for y, x in zip(ys, xs):
        vals = [idx[y + dy, x + dx] for dy, dx in NEIGH4
                if 0 <= y + dy < h and 0 <= x + dx < w]
        vals = [v for v in vals if v != idx[y, x]]
        if vals:
            uniq, cnt = np.unique(vals, return_counts=True)
            out[y, x] = uniq[cnt.argmax()]
    return out


def _row_runs(mask: np.ndarray) -> list[tuple[int, int, int]]:
    """Для каждой строки — начало и конец занятого участка маски."""
    runs = []
    for y, row in enumerate(mask):
        xs = np.nonzero(row)[0]
        if xs.size:
            runs.append((y, int(xs.min()), int(xs.max())))
    return runs


def jaggies(idx: np.ndarray, min_rows: int = 6) -> list[tuple[int, int]]:
    """Сбои ритма ступенчатой кромки (D03).

    Ступеньки должны расти ровно: 2,2,2 или 3,3,3. Одна выпавшая ступенька
    рвёт линию — глаз цепляется именно за неё.
    """
    out = []
    for value in np.unique(idx):
        if value < 0:
            continue
        runs = _row_runs(idx == value)
        if len(runs) < min_rows:
            continue
        for edge in (1, 2):                      # левая и правая кромки
            xs = [r[edge] for r in runs]
            steps, start = [], 0
            for i in range(1, len(xs)):
                if xs[i] != xs[i - 1]:
                    steps.append((i - start, runs[start][0]))
                    start = i
            steps.append((len(xs) - start, runs[start][0]))
            if len(steps) < 3:
                continue
            lens = [s[0] for s in steps]
            typical = int(np.median(lens))
            for (ln, y), i in zip(steps, range(len(steps))):
                if 0 < i < len(steps) - 1 and ln != typical and abs(ln - typical) <= 2:
                    out.append((y, xs[min(i, len(xs) - 1)]))
    return out


def banding(idx: np.ndarray) -> list[tuple[int, int]]:
    """Три и более полосы подряд одинаковой ширины (D01)."""
    out = []
    h, w = idx.shape
    for x in range(0, w, max(1, w // 24)):
        col = idx[:, x]
        runs, start = [], 0
        for y in range(1, h):
            if col[y] != col[y - 1]:
                runs.append((y - start, start))
                start = y
        runs.append((h - start, start))
        run = 1
        for i in range(1, len(runs)):
            if runs[i][0] == runs[i - 1][0] and runs[i][0] > 1:
                run += 1
                if run >= 3:
                    out.append((runs[i][1], x))
            else:
                run = 1
    return out


def lint(idx: np.ndarray, pal=None) -> list[dict]:
    """Нарушения с координатами: чистильщику нужно знать, ГДЕ править."""
    issues = []
    ys, xs = np.nonzero(stray_mask(idx, pal))
    for y, x in zip(ys, xs):
        issues.append({"rule": "D04", "at": (int(y), int(x)),
                       "text": "одиночный пиксель"})
    for y, x in jaggies(idx):
        issues.append({"rule": "D03", "at": (int(y), int(x)),
                       "text": "сбой ритма ступенек"})
    for y, x in banding(idx):
        issues.append({"rule": "D01", "at": (int(y), int(x)),
                       "text": "полосы одинаковой ширины"})
    return issues


def fix_jaggies(idx: np.ndarray) -> np.ndarray:
    """Подтянуть выпавшую ступеньку к общему ритму."""
    out = idx.copy()
    for value in np.unique(idx):
        if value < 0:
            continue
        mask = idx == value
        runs = _row_runs(mask)
        if len(runs) < 4:
            continue
        for edge, direction in ((1, +1), (2, -1)):
            xs = [r[edge] for r in runs]
            steps, start = [], 0
            for i in range(1, len(xs)):
                if xs[i] != xs[i - 1]:
                    steps.append((start, i - start))
                    start = i
            steps.append((start, len(xs) - start))
            if len(steps) < 3:
                continue
            typical = int(np.median([ln for _, ln in steps]))
            if typical < 1:
                continue
            for k, (first, ln) in enumerate(steps):
                if k == 0 or k == len(steps) - 1 or ln >= typical:
                    continue
                # короткая ступенька: дотягиваем её строки до соседней кромки
                for row in range(first, first + ln):
                    y = runs[row][0]
                    x = xs[row]
                    nx = xs[max(0, first - 1)]
                    if edge == 1:
                        lo, hi = sorted((x, nx))
                        out[y, lo:hi] = value
                    else:
                        lo, hi = sorted((x, nx))
                        out[y, lo + 1:hi + 1] = value
    return out


def _mid_index(pal, a: int, b: int) -> int:
    """Ближайший индекс палитры к среднему цвету двух тонов.

    Промежуточный тон не выдумывается, а ищется в палитре: пиксель-арт живёт
    на ограниченном наборе цветов, и антиалиасинг не имеет права его
    расширять (C07).
    """
    colours = np.asarray(pal.colors, float)
    target = (colours[a] + colours[b]) / 2
    d = ((colours - target) ** 2).sum(1)
    d[[a, b]] = np.inf
    return int(d.argmin())


def antialias(idx: np.ndarray, pal, min_gap: float = 25.0,
              max_gap: float = 120.0) -> np.ndarray:
    """Ручной антиалиасинг: промежуточный тон только в УГЛАХ ступенек.

    По прямой кромке антиалиасинг не ставят — он её просто размывает (D05).
    Угол — это место, где ступенька меняет направление: у пикселя два соседа
    другого цвета, стоящие под прямым углом друг к другу.
    """
    out = idx.copy()
    h, w = idx.shape
    up, down = _shift(idx, 1, 0), _shift(idx, -1, 0)
    left, right = _shift(idx, 0, 1), _shift(idx, 0, -1)
    for dy_pair, dx_pair in (((up, down), (left, right)),):
        for vert in dy_pair:
            for horz in dx_pair:
                corner = (vert != idx) & (horz != idx) & (vert == horz) & (idx >= 0) & (vert >= 0)
                if not corner.any():
                    continue
                ys, xs = np.nonzero(corner)
                lum = np.asarray(pal.colors, float) @ [0.299, 0.587, 0.114]
                for y, x in zip(ys, xs):
                    a, b = int(idx[y, x]), int(vert[y, x])
                    if a == b:
                        continue
                    gap = abs(lum[a] - lum[b])
                    # Сглаживают близкие ступени, а не пропасть между
                    # материалами: тон «между небом и бетоном» — это не
                    # антиалиасинг, а грязь, и линтер справедливо зовёт её
                    # мусором. Порог снизу отсекает бессмысленные правки.
                    if gap < min_gap or gap > max_gap:
                        continue
                    mid = _mid_index(pal, a, b)
                    if mid in (a, b):
                        continue
                    if not (min(lum[a], lum[b]) < lum[mid] < max(lum[a], lum[b])):
                        continue
                    out[y, x] = mid
    return out


def selective_outline(idx: np.ndarray, pal, light=(-0.7, -0.7), tone: int | None = None):
    """Контур только с теневой стороны (C04).

    Обводка по всему кругу гасит объём: со стороны света кромку держит тон
    темнее заливки, а чёрная линия там лишняя.
    """
    out = idx.copy()
    body = idx >= 0
    ly, lx = light
    norm = float(np.hypot(ly, lx)) or 1.0
    # Теневой край — тот, что остаётся открытым, когда тело двигают В СТОРОНУ
    # света. Со знаком наоборот контур ложится по освещённой кромке и гасит
    # объём ровно так, как это делает обводка по кругу.
    dy, dx = int(round(ly / norm)), int(round(lx / norm))
    if dy == 0 and dx == 0:
        return out
    shadow_side = body & ~_shift(body, dy, dx).astype(bool)
    if tone is None:
        tone = 0
    out[shadow_side] = tone
    return out


def debanding(idx: np.ndarray) -> np.ndarray:
    """Разорвать ровные полосы дизерингом по их границе (D01, D02)."""
    out = idx.copy()
    h, w = idx.shape
    checker = ((np.arange(h)[:, None] + np.arange(w)[None, :]) % 2) == 0
    for y, x in banding(idx):
        for yy in (y, y - 1):
            if 0 <= yy < h - 1:
                row_above = out[yy]
                row_below = out[min(yy + 1, h - 1)]
                mix = np.where(checker[yy], row_below, row_above)
                out[yy] = mix
    return out


def tidy(idx: np.ndarray, pal, light=(-0.7, -0.7), outline: bool = False,
         straighten: bool = False) -> np.ndarray:
    """Полный проход: мусор → (ступеньки) → антиалиасинг → (контур).

    Выравнивание ступенек по умолчанию выключено: на сложной сцене оно пока
    приносит больше мусора, чем чинит. Долг записан, включается флагом.
    """
    out = despeckle(idx)
    if straighten:
        out = fix_jaggies(out)
    out = antialias(out, pal)
    if outline:
        out = selective_outline(out, pal, light)
    return out.astype(idx.dtype)
