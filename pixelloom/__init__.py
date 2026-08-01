"""pixelloom — рисование пиксель-арта кодом с живым просмотром в браузере.

    from pixelloom import Canvas, palettes
    from pixelloom.live import LiveView

    view = LiveView().start(open_browser=True)
    c = Canvas(48, 48, palettes.BASE16)
    view.attach(c, "первый рисунок")
    draw.ellipse(c, 24, 28, 10, 9, palettes.BASE16.index_of("leaf"))

Холст сообщает о каждой записи пикселей, живой просмотр отдаёт разницу в
браузер порциями. Поэтому в окне видно, как рисунок собирается, а не только
чем всё кончилось.
"""

from .canvas import TRANSPARENT, Canvas
from .palette import Palette, hex_to_rgb, rgb_to_hex
from . import draw, isoparts, noise, paint, palettes, render
from .render import (
    save_contact_sheet,
    save_gif,
    save_palette_strip,
    save_png,
    save_preview,
    save_sheet,
    stack,
    to_image,
)

__version__ = "0.2.0"

__all__ = [
    "Canvas",
    "Palette",
    "TRANSPARENT",
    "draw",
    "hex_to_rgb",
    "isoparts",
    "noise",
    "paint",
    "palettes",
    "render",
    "rgb_to_hex",
    "save_contact_sheet",
    "save_gif",
    "save_palette_strip",
    "save_png",
    "save_preview",
    "save_sheet",
    "stack",
    "to_image",
    "__version__",
]
