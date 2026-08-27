"""Камера как сменная деталь: изометрия, фасад, вид сверху.

Проекция раньше была зашита в четырёх местах, и поменять её значило
переписать рендер. Теперь она живёт в объекте: рендер спрашивает камеру, куда
падает точка и откуда идёт луч, а какая именно камера стоит — дело вызова.
"""

import numpy as np
import pytest

from pixelloom.canvas import TRANSPARENT
from pixelloom.isoparts import (CAMERAS, Camera, Material, Part, camera_of,
                                render_parts, screen_bounds)
from pixelloom.palette import Palette

PAL = Palette.from_hex("test",
                       ["#141414", "#3C3C46", "#8C4038", "#C8643C", "#F0F0F0"],
                       ["shadow", "hull0", "hull1", "hull2", "hull3"],
                       {"hull": [1, 2, 3, 4]})


def silhouette(canvas):
    """Габарит непрозрачной части: ширина, высота."""
    solid = canvas.data != TRANSPARENT
    if not solid.any():
        return 0, 0
    ys, xs = np.nonzero(solid)
    return xs.max() - xs.min() + 1, ys.max() - ys.min() + 1


def test_registry_knows_three_cameras():
    assert set(CAMERAS) >= {"iso", "front", "top"}


def test_camera_of_accepts_name_and_object():
    cam = camera_of("front")
    assert isinstance(cam, Camera)
    assert camera_of(cam) is cam


def test_unknown_camera_is_a_clear_error():
    with pytest.raises(KeyError):
        camera_of("side")


def test_iso_projection_matches_the_old_formula():
    """Старые вызовы обязаны рисовать ровно то же, что рисовали."""
    cam = camera_of("iso")
    p = np.array([[2.0, 3.0, 1.5]])
    sx, sy = cam.to_screen(p, px_per_m=8.0)
    assert sx[0] == pytest.approx((2.0 - 3.0) * 8.0)
    assert sy[0] == pytest.approx((2.0 + 3.0) * 8.0 / 2 - 1.5 * 8.0)


def test_front_projection_ignores_depth():
    """У фасада глубина не двигает точку по экрану — в этом весь смысл."""
    cam = camera_of("front")
    near = cam.to_screen(np.array([[2.0, -4.0, 1.5]]), px_per_m=8.0)
    far = cam.to_screen(np.array([[2.0, 9.0, 1.5]]), px_per_m=8.0)
    assert near[0][0] == pytest.approx(far[0][0])
    assert near[1][0] == pytest.approx(far[1][0])
    assert near[0][0] == pytest.approx(2.0 * 8.0)
    assert near[1][0] == pytest.approx(-1.5 * 8.0)


def test_top_projection_ignores_height():
    cam = camera_of("top")
    low = cam.to_screen(np.array([[2.0, 3.0, 0.0]]), px_per_m=8.0)
    high = cam.to_screen(np.array([[2.0, 3.0, 5.0]]), px_per_m=8.0)
    assert low[0][0] == pytest.approx(high[0][0])
    assert low[1][0] == pytest.approx(high[1][0])
    assert low[1][0] == pytest.approx(3.0 * 8.0)


def test_front_camera_draws_a_box_as_a_rectangle():
    """Куб 2×1×3 анфас: ширина по X, высота по Z, глубина не видна."""
    part = Part("box", (0, 0, 0), (2.0, 1.0, 3.0))
    canvas = render_parts([part], PAL, 120, 120, px_per_m=10.0,
                          origin=(0.5, 0.5), camera="front", shadows=False)
    w, h = silhouette(canvas)
    assert w == pytest.approx(20, abs=2)
    assert h == pytest.approx(30, abs=2)


def test_top_camera_draws_a_box_by_its_footprint():
    part = Part("box", (0, 0, 0), (2.0, 4.0, 1.0))
    canvas = render_parts([part], PAL, 140, 140, px_per_m=10.0,
                          origin=(0.5, 0.5), camera="top", shadows=False)
    w, h = silhouette(canvas)
    assert w == pytest.approx(20, abs=2)
    assert h == pytest.approx(40, abs=2)


def test_front_camera_draws_a_sphere_as_a_circle():
    part = Part("sphere", (0, 0, 0), (1.5, 1.5, 1.5))
    canvas = render_parts([part], PAL, 120, 120, px_per_m=10.0,
                          origin=(0.5, 0.5), camera="front", shadows=False)
    w, h = silhouette(canvas)
    assert w == pytest.approx(h, abs=2)
    assert w == pytest.approx(30, abs=2)


def test_front_camera_hides_what_stands_behind():
    """Ближнее тело закрывает дальнее: у фасада это единственный признак
    глубины, и если он не работает, картинка разваливается.

    Два куба для проверки не годятся: обе грани смотрят в камеру одинаково и
    красятся одним индексом. Ближним ставим шар — у него в середине своя
    нормаль, и подмена сразу видна по цвету.
    """
    mat = Material(ramp="hull")
    back = Part("box", (0, 4.0, 0), (3.0, 1.0, 3.0), material=mat)
    near = Part("sphere", (0, -4.0, 0), (1.0, 1.0, 1.0), material=mat)

    both = render_parts([back, near], PAL, 120, 120, px_per_m=10.0,
                        origin=(0.5, 0.5), camera="front", shadows=False)
    only_back = render_parts([back], PAL, 120, 120, px_per_m=10.0,
                             origin=(0.5, 0.5), camera="front", shadows=False)
    only_near = render_parts([near], PAL, 120, 120, px_per_m=10.0,
                             origin=(0.5, 0.5), camera="front", shadows=False)

    assert both.data[60, 60] == only_near.data[60, 60]
    assert both.data[60, 60] != only_back.data[60, 60]
    # а по краям, куда шар не достаёт, остаётся стена
    assert both.data[60, 100] == only_back.data[60, 100]


def test_screen_bounds_follows_the_camera():
    """Кадр под фасад считается по фасаду, а не по изометрии."""
    part = Part("box", (0, 0, 0), (2.0, 6.0, 1.0))
    w_iso, h_iso, _, _ = screen_bounds([part], px_per_m=10.0)
    w_front, h_front, _, _ = screen_bounds([part], px_per_m=10.0,
                                           camera="front")
    assert w_front < w_iso           # глубина шесть метров не тянет кадр вширь
    assert h_front < h_iso


def test_default_camera_is_iso():
    part = Part("box", (0, 0, 0), (2.0, 2.0, 2.0))
    a = render_parts([part], PAL, 100, 100, px_per_m=8.0, shadows=False)
    b = render_parts([part], PAL, 100, 100, px_per_m=8.0, camera="iso",
                     shadows=False)
    assert np.array_equal(a.data, b.data)
