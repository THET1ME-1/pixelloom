"""Модель предметов в мировых координатах и шум без шва: грабли meadow-blocks числами."""
import numpy as np

from pixelloom.tilenoise import periodic_noise, seamless_noise
from pixelloom.worldmodel import Model, RampPalette, light_shift, slice_by_cells

PALETTE = RampPalette({
    "wood": ["#2a1a17", "#45291f", "#673d27", "#8a5532", "#a96f3f", "#c48d52", "#dbad6e"],
})


def model(width=32, height=40):
    return Model(width, height, PALETTE)


def test_light_matches_block_faces():
    # верх на своей ступени, грань +v на одну темнее, грань +u на две — как у блока
    assert abs(light_shift([0, 0, 1])) < 1e-9
    assert abs(light_shift([0, 1, 0]) + 1) < 1e-9
    assert abs(light_shift([1, 0, 0]) + 2) < 1e-9


def test_box_faces_step_down_from_top():
    m = model()
    m.box(4, 12, 4, 12, 0, 8, "wood", 5)
    image = m.render(outline=False)
    filled = image[..., 3] == 255
    light = image[..., :3].sum(axis=-1)
    # грани находятся по точке мира в пикселе: верх h = 8, левая v = 12, правая u = 12
    top = light[filled & np.isclose(m.world_h, 8)].mean()
    left = light[filled & np.isclose(m.world_v, 12) & ~np.isclose(m.world_h, 8)].mean()
    right = light[filled & np.isclose(m.world_u, 12) & ~np.isclose(m.world_h, 8)].mean()
    assert top > left > right


def test_thin_rail_keeps_its_colour():
    # жердь 2 единицы высотой: кайма по краю, но середина светлая, а не чёрная (B09)
    m = model()
    m.box(0, 16, 7.6, 8.4, 4, 6, "wood", 5)
    image = m.render()
    filled = image[..., 3] == 255
    darkest = np.array(PALETTE.colors[PALETTE.index("wood", 0)])
    dark = (image[..., :3] == darkest).all(axis=-1) & filled
    assert dark.sum() < filled.sum() * 0.75


def test_sphere_has_no_holes():
    # шар сеткой по широте и долготе: внутри силуэта нет пустых пикселей (D17)
    m = model()
    m.sphere(8, 8, 5, 4, "wood", 4)
    filled = m.render(outline=False)[..., 3] == 255
    rows = np.nonzero(filled.any(axis=1))[0]
    for y in rows:
        xs = np.nonzero(filled[y])[0]
        assert filled[y, xs.min():xs.max() + 1].all()


def test_slice_by_cells_keeps_every_pixel():
    # предмет на две клетки: куски вместе дают весь кадр, ни пикселя не теряется (G33)
    # кадр клетки 40 в высоту; большой кадр выше на 8 за каждую лишнюю клетку
    m = Model(48, 48, PALETTE)
    m.origin_x = 16
    m.origin_y = 40 - 16
    m.box(0, 32, 6, 10, 0, 4, "wood", 4)
    image = m.render()
    pieces = slice_by_cells(m, image, "u", 2, 40)
    total = sum(int((p[..., 3] == 255).sum()) for p in pieces)
    assert total == int((image[..., 3] == 255).sum())


def test_periodic_noise_wraps_on_edge():
    a = np.linspace(0, 16, 33)
    assert np.allclose(periodic_noise(np.zeros(1), a, 4, 3), periodic_noise(np.full(1, 16.0), a, 4, 3))


def test_seamless_noise_matches_neighbours_on_edge():
    # два разных варианта сходятся на кромке к общему шуму: шва между блоками нет (T12)
    b = np.linspace(0, 16, 33)
    edge_one = seamless_noise(np.zeros_like(b), b, 4, 11, 101)
    edge_two = seamless_noise(np.zeros_like(b), b, 4, 11, 202)
    assert np.allclose(edge_one, edge_two)
    middle_one = seamless_noise(np.full(5, 8.0), np.linspace(6, 10, 5), 4, 11, 101)
    middle_two = seamless_noise(np.full(5, 8.0), np.linspace(6, 10, 5), 4, 11, 202)
    assert not np.allclose(middle_one, middle_two)
