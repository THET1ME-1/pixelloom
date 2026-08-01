"""Дельты живого просмотра должны собираться обратно в тот же рисунок."""

from __future__ import annotations

import base64
import json
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pixelloom import Canvas, TRANSPARENT, draw, palettes  # noqa: E402
from pixelloom.live import LiveView  # noqa: E402
from pixelloom.live.view import CLEAR  # noqa: E402


def collect(view, sub) -> list[dict]:
    """Дельты зрителя. Служебные сообщения о шагах в разбор не идут."""
    out = []
    while sub.q:
        msg = json.loads(sub.q.popleft()[len("data: ") :])
        if msg.get("kind") == "delta":
            out.append(msg)
    return out


def replay(messages: list[dict], w: int, h: int) -> np.ndarray:
    """Собрать картинку зрителя из потока дельт, как это делает браузер."""
    screen = np.full((h, w), CLEAR, dtype=np.uint8)
    for m in messages:
        if m.get("kind") != "delta":
            continue
        raw = base64.b64decode(m["data"])
        x0, y0, bw, bh = m["box"]
        if m["shape"] == "rect":
            block = np.frombuffer(raw, dtype=np.uint8).reshape(bh, bw)
            screen[y0 : y0 + bh, x0 : x0 + bw] = block
        else:
            for o in range(0, len(raw), 5):
                at, v = struct.unpack_from("<IB", raw, o)
                screen[at // w, at % w] = v
    return screen


def make_view(canvas: Canvas):
    view = LiveView(pace=0.0)
    view._running = True          # без сети: сервер в тесте не нужен
    view.attach(canvas, "тест")
    sub = view.subscribe()
    sub.q.clear()                 # выбрасываем мету и стартовый полный кадр
    return view, sub


def expected(canvas: Canvas) -> np.ndarray:
    data = np.asarray(canvas.data)
    return np.where(data < 0, CLEAR, data).astype(np.uint8)


def test_dense_change_travels_as_rect():
    c = Canvas(32, 32, palettes.BASE16)
    view, sub = make_view(c)
    draw.ellipse(c, 16, 16, 8, 8, 5)
    msgs = collect(view, sub)
    assert msgs, "изменение холста должно уехать зрителю"
    assert any(m["shape"] == "rect" for m in msgs), "плотный кусок едет прямоугольником"
    assert np.array_equal(replay(msgs, 32, 32), expected(c))


def test_sparse_change_travels_as_pairs():
    c = Canvas(32, 32, palettes.BASE16)
    view, sub = make_view(c)
    scatter = np.zeros((32, 32), dtype=bool)
    for i in range(6):
        scatter[i * 5, i * 5] = True      # редкая диагональ по всему полю
    c.data[scatter] = 3
    msgs = collect(view, sub)
    assert any(m["shape"] == "sparse" for m in msgs), "редкие пиксели едут парами"
    assert np.array_equal(replay(msgs, 32, 32), expected(c))


def test_full_picture_matches_after_many_steps():
    c = Canvas(48, 48, palettes.BASE16)
    view, sub = make_view(c)
    draw.rect(c, 8, 30, 40, 44, 7, fill=True)
    draw.ellipse(c, 24, 20, 10, 8, 14)
    draw.outline(c, 0)
    c.data[np.asarray(c.data) == 7] = 8
    msgs = collect(view, sub)
    assert np.array_equal(replay(msgs, 48, 48), expected(c))


def test_step_names_come_from_context_and_stack():
    c = Canvas(16, 16, palettes.BASE16)
    view, sub = make_view(c)
    with view.step("глаза"):
        c.set(4, 4, 1)
    msgs = collect(view, sub)
    assert any(m.get("step") == "глаза" for m in msgs if m["kind"] == "delta")

    sub.q.clear()
    _draw_helper(c)
    msgs = collect(view, sub)
    named = [m["step"] for m in msgs if m["kind"] == "delta"]
    assert "_draw_helper" in named, "без явного шага имя берётся из функции"


def _draw_helper(c: Canvas) -> None:
    c.set(9, 9, 2)


def test_transparent_pixels_survive_the_trip():
    c = Canvas(16, 16, palettes.BASE16)
    draw.rect(c, 2, 2, 12, 12, 6, fill=True)
    view, sub = make_view(c)
    c.data[4:8, 4:8] = TRANSPARENT
    msgs = collect(view, sub)
    screen = replay(msgs, 16, 16)
    assert (screen[4:8, 4:8] == CLEAR).all(), "стёртое должно стать прозрачным и у зрителя"


def test_steps_are_recorded_with_snapshots():
    c = Canvas(16, 16, palettes.BASE16)
    view, _ = make_view(c)
    with view.step("первый"):
        c.set(1, 1, 1)
    with view.step("второй"):
        c.set(2, 2, 2)
    names = [s.name for s in view._steps]
    assert names == ["первый", "второй"]
    assert view.step_png(0), "у закрытого шага есть снимок для ленты и таймлапса"
