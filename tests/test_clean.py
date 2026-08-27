"""Чистильщик пиксель-арта: операторы, каждый со своей проверкой.

Смысл модуля: рендер и любая другая «форма» дают сырую сетку — одиночные
пиксели, рваные ступеньки, полосы одинаковой ширины. Художник вычищает это
руками; здесь то же самое, но правилами с числами.
"""
import numpy as np
import pytest

from pixelloom import clean
from pixelloom.palette import Palette


@pytest.fixture
def pal():
    return Palette.from_hex(
        "test",
        ["#101018", "#303048", "#505078", "#7878a8", "#a0a0d0", "#d0d0f0"],
        ["c0", "c1", "c2", "c3", "c4", "c5"],
    )


def test_despeckle_ubiraet_odinochnyy_piksel():
    a = np.zeros((7, 7), int)
    a[3, 3] = 5                      # один чужой пиксель посреди поля
    out = clean.despeckle(a)
    assert out[3, 3] == 0
    assert (out == 5).sum() == 0


def test_despeckle_ne_trogaet_kluster():
    a = np.zeros((7, 7), int)
    a[2:5, 2:5] = 5                  # кластер 3×3 — это форма, а не мусор
    out = clean.despeckle(a)
    assert (out == 5).sum() == 9


def test_despeckle_beryot_tsvet_bolshinstva_sosedey():
    a = np.zeros((5, 5), int)
    a[:, 3:] = 2
    a[2, 1] = 4
    out = clean.despeckle(a)
    assert out[2, 1] == 0


def test_lint_nahodit_koordinaty_musora():
    a = np.zeros((6, 6), int)
    a[1, 1] = 3
    a[4, 4] = 3
    issues = clean.lint(a)
    stray = [i for i in issues if i["rule"] == "D04"]
    assert {tuple(i["at"]) for i in stray} == {(1, 1), (4, 4)}


def test_fix_jaggies_vyravnivaet_stupenku():
    """Ступеньки 2,2,1,2 — сбой ритма: средняя выпадает и рвёт линию."""
    a = np.zeros((10, 12), int)
    runs = [2, 2, 1, 2, 2]
    x = 0
    for row, n in enumerate(runs):
        a[row + 2, x:x + n] = 1
        x += n
    before = clean.lint(a)
    out = clean.fix_jaggies(a)
    after = clean.lint(out)
    n_before = len([i for i in before if i["rule"] == "D03"])
    n_after = len([i for i in after if i["rule"] == "D03"])
    assert n_after < n_before


def test_antialias_stavit_promezhutochnyy_ton(pal):
    """У ступеньки в углу появляется тон между двумя соседними."""
    a = np.full((12, 12), 2)
    for row in range(12):
        a[row, : 2 + row // 2] = 4
    out = clean.antialias(a, pal)
    mid = set(np.unique(out)) - {2, 4}
    assert mid, "промежуточных тонов не появилось"
    assert all(2 < m < 4 for m in mid), f"тон вне пары соседей: {mid}"


def test_antialias_ne_sglazhivaet_propast(pal):
    """Между самым тёмным и самым светлым тоном антиалиасинг не ставят.

    Одна клетка «между небом и бетоном» — это не сглаживание, а грязь: она
    остаётся одиноким пикселем и читается мусором (D04). Такой переход
    лечится лишней ступенью шкалы, а не антиалиасингом.
    """
    a = np.zeros((12, 12), int)
    for row in range(12):
        a[row, : 2 + row // 2] = 5
    out = clean.antialias(a, pal)
    assert set(np.unique(out)) == {0, 5}


def test_antialias_ne_trogaet_pryamuyu_kromku(pal):
    a = np.zeros((10, 10), int)
    a[:, :5] = 5
    out = clean.antialias(a, pal)
    assert set(np.unique(out)) == {0, 5}


def test_selective_outline_tolko_s_tenevoy_storony(pal):
    ys, xs = np.mgrid[0:16, 0:16]
    disc = ((ys - 8) ** 2 + (xs - 8) ** 2) < 30
    a = np.full((16, 16), -1)
    a[disc] = 4
    out = clean.selective_outline(a, pal, light=(-0.7, -0.7))
    dark = out == 0
    ys_d, xs_d = np.nonzero(dark)
    assert len(ys_d) > 0, "контур не поставлен"
    assert ys_d.mean() > 8 and xs_d.mean() > 8, "контур лёг со стороны света"


def test_debanding_razryvaet_rovnye_polosy():
    a = np.repeat(np.arange(4)[:, None], 16, axis=1)
    a = np.repeat(a, 3, axis=0)             # четыре полосы ровно по 3 строки
    before = len([i for i in clean.lint(a) if i["rule"] == "D01"])
    out = clean.debanding(a)
    after = len([i for i in clean.lint(out) if i["rule"] == "D01"])
    assert before > 0 and after < before


def test_konveyer_ne_menyaet_razmer(pal):
    a = np.random.default_rng(3).integers(0, 6, (24, 24))
    out = clean.tidy(a, pal, light=(-0.7, -0.7))
    assert out.shape == a.shape
    assert out.dtype == a.dtype
