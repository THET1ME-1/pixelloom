"""Изометрические тела: что рейтрейсер вообще рисует и куда попадает.

Пиксель в пиксель картинку тут не проверить: она зависит от палитры и света.
Зато проверяемо главное — тело видно, оно стоит там, где заказано, и разные
примитивы дают разные силуэты.
"""

import numpy as np
import pytest

from pixelloom import isoparts, palettes
from pixelloom.canvas import TRANSPARENT
from pixelloom.isoparts import Material, Part, ground_shadow, render_parts

GREY = Material(ramp="grey")


def box(size=(1.0, 1.0, 1.0), **kw):
    return Part("box", pos=(0.0, 0.0, size[2] / 2), size=size, material=GREY, **kw)


def drawn(canvas):
    return int((canvas.data != TRANSPARENT).sum())


def test_модуль_доступен_из_пакета():
    assert isoparts.render_parts is render_parts
    for name in ("Material", "Part", "render_parts", "ground_shadow"):
        assert hasattr(isoparts, name), name


def test_тело_видно():
    canvas = render_parts([box()], palettes.BASE16, 48, 48, px_per_m=16)
    assert drawn(canvas) > 100


def test_пустой_список_даёт_пустой_холст():
    canvas = render_parts([], palettes.BASE16, 32, 32)
    assert drawn(canvas) == 0


def test_больше_пикселей_на_метр_даёт_крупнее():
    small = render_parts([box()], palettes.BASE16, 64, 64, px_per_m=8)
    big = render_parts([box()], palettes.BASE16, 64, 64, px_per_m=16)
    assert drawn(big) > drawn(small) * 2


def test_привязка_двигает_тело_по_холсту():
    left = render_parts([box()], palettes.BASE16, 64, 64, px_per_m=12,
                        origin=(0.25, 0.5))
    right = render_parts([box()], palettes.BASE16, 64, 64, px_per_m=12,
                         origin=(0.75, 0.5))
    cx = lambda c: np.nonzero(c.data != TRANSPARENT)[1].mean()  # noqa: E731
    assert cx(left) < cx(right) - 10


@pytest.mark.parametrize("kind", ["sphere", "cylinder", "box", "cone", "torus"])
def test_каждый_примитив_рисуется(kind):
    part = Part(kind, pos=(0.0, 0.0, 0.5), size=(0.5, 0.5, 0.5), material=GREY)
    canvas = render_parts([part], palettes.BASE16, 48, 48, px_per_m=16)
    assert drawn(canvas) > 20, f"{kind} не нарисовался"


def test_срез_снизу_убирает_нижнюю_половину():
    """Порог `cut_below` меряется в мире, не от центра тела.

    Шар стоит на земле: центр на 0.5, мировые высоты от 0 до 1. Срез по 0.5
    оставляет купол, срез по 0.0 не отрезает ничего.
    """
    whole = Part("sphere", pos=(0.0, 0.0, 0.5), size=(0.5, 0.5, 0.5), material=GREY)
    dome = Part("sphere", pos=(0.0, 0.0, 0.5), size=(0.5, 0.5, 0.5), material=GREY,
                cut_below=0.5)
    full = render_parts([whole], palettes.BASE16, 48, 48, px_per_m=16)
    half = render_parts([dome], palettes.BASE16, 48, 48, px_per_m=16)
    assert drawn(half) < drawn(full)


def test_тень_ложится_и_не_трогает_прозрачность_тела():
    canvas = render_parts([box()], palettes.BASE16, 64, 64, px_per_m=12)
    before = drawn(canvas)
    ground_shadow(canvas, [box()], px_per_m=12, origin=(0.5, 0.85),
                  index=palettes.BASE16.index_of("ink"))
    assert drawn(canvas) > before
