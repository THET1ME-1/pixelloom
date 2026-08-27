"""Эскиз символами → спрайт: форму задаёт человек, тон считает библиотека."""
import numpy as np
import pytest

from pixelloom import sketch
from pixelloom.canvas import TRANSPARENT


SAMPLE = """# проба
. . . .
. A A .
. A B .
. . . .
"""


@pytest.fixture
def grid_file(tmp_path):
    p = tmp_path / "s.txt"
    p.write_text(SAMPLE.replace(" ", ""), encoding="utf-8")
    return p


def test_zagruzka_daet_setku_bez_kommentariev(grid_file):
    grid = sketch.load_grid(grid_file)
    assert grid.shape == (4, 4)
    assert grid[1, 1] == "A" and grid[2, 2] == "B"
    assert grid[0, 0] == "."


def test_pustye_kletki_ostayutsya_prozrachnymi(grid_file):
    zones = {"A": sketch.Zone("#8a5a2b"), "B": sketch.Zone("#404050")}
    canvas = sketch.render(grid_file, zones)
    assert canvas.data[0, 0] == TRANSPARENT
    assert canvas.data[1, 1] != TRANSPARENT


def test_zona_poluchaet_svoyu_shkalu(grid_file):
    zones = {"A": sketch.Zone("#8a5a2b", steps=5), "B": sketch.Zone("#404050", steps=4)}
    canvas = sketch.render(grid_file, zones)
    assert set(canvas.palette.ramps) == {"A", "B"}
    assert len(canvas.palette.ramps["A"]) == 5


def test_neizvestnyy_simvol_zameten(grid_file):
    with pytest.raises(KeyError) as e:
        sketch.render(grid_file, {"A": sketch.Zone("#8a5a2b")})
    assert "B" in str(e.value)


def test_lint_setki_schitaet_proportsii(tmp_path):
    """Проверять форму надо ДО раскраски: тут это ещё дёшево исправить."""
    p = tmp_path / "fig.txt"
    p.write_text("\n".join([
        "..HH..",
        "..HH..",
        ".BBBB.",
        ".BBBB.",
        "..L.L.",
        "..L.L.",
    ]), encoding="utf-8")
    grid = sketch.load_grid(p)
    stats = sketch.measure(grid, body=("H", "B", "L"))
    assert stats["height"] == 6
    assert stats["width"] == 4
    assert 0 < stats["support"] <= 1
