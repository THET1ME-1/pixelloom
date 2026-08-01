"""Холст должен замечать записи пикселей и молчать обо всём остальном."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pixelloom import Canvas, TRANSPARENT, draw, palettes  # noqa: E402


def fresh() -> tuple[Canvas, list]:
    c = Canvas(16, 16, palettes.BASE16)
    hits: list[int] = []
    c.watch(lambda _: hits.append(1))
    return c, hits


def test_set_notifies():
    c, hits = fresh()
    c.set(3, 4, 2)
    assert hits, "запись одного пикселя должна уведомить наблюдателя"


def test_slice_notifies():
    c, hits = fresh()
    c.data[2:5, 2:5] = 7
    assert hits


def test_view_of_slice_notifies():
    """Так пишет paint.reveal_rows: сначала срез, потом запись по маске."""
    c, hits = fresh()
    band = c.data[0:4]
    mask = np.zeros((4, 16), dtype=bool)
    mask[1, 1] = True
    band[mask] = 5
    assert hits


def test_boolean_mask_notifies():
    c, hits = fresh()
    c.data[np.asarray(c.data) == TRANSPARENT] = 1
    assert hits


def test_derived_arrays_are_silent():
    """Сравнения и сдвиги заводят свою память и уведомлять не должны."""
    c, hits = fresh()
    solid = c.data != TRANSPARENT
    rolled = np.roll(solid, 1, axis=0)
    hits.clear()
    solid[0, 0] = True
    rolled[0, 0] = True
    assert not hits, "производные массивы холст не трогают"


def test_copy_is_independent():
    c, hits = fresh()
    twin = c.copy()
    hits.clear()
    twin.data[0, 0] = 3
    assert not hits, "копия не должна дёргать наблюдателя исходного холста"
    assert c.get(0, 0) == TRANSPARENT


def test_mute_stops_notifications():
    c, hits = fresh()
    c.mute(True)
    c.set(1, 1, 4)
    assert not hits
    c.mute(False)
    c.set(1, 2, 4)
    assert hits


def test_primitives_notify():
    c, hits = fresh()
    draw.ellipse(c, 8, 8, 4, 4, 3)
    assert hits
    hits.clear()
    draw.outline(c, 1)
    assert hits


def test_bbox_and_rgba():
    c, _ = fresh()
    assert c.bbox() is None
    c.set(5, 6, 2)
    assert c.bbox() == (5, 6, 6, 7)
    rgba = c.to_rgba()
    assert rgba.shape == (16, 16, 4)
    assert rgba[6, 5, 3] == 255 and rgba[0, 0, 3] == 0
