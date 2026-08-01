"""Вывод холста в файлы: превью, спрайт-лист, GIF, контактный лист.

Превью существует ради обратной связи: масштаб делает пиксель различимым,
шахматка показывает прозрачность, сетка помогает считать координаты.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .canvas import Canvas

CHECKER_A = (34, 34, 40)
CHECKER_B = (26, 26, 31)
GRID_COLOR = (255, 255, 255, 38)
AXIS_COLOR = (255, 255, 255, 96)

# Встроенный шрифт Pillow не знает кириллицы и рисует подписи квадратиками
FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
]


def _font(size: int = 12):
    for path in FONT_CANDIDATES:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return ImageFont.load_default()


def to_image(canvas: Canvas) -> Image.Image:
    return Image.fromarray(canvas.to_rgba(), mode="RGBA")


def _checkerboard(width: int, height: int, cell: int) -> Image.Image:
    bg = Image.new("RGBA", (width, height), CHECKER_A + (255,))
    d = ImageDraw.Draw(bg)
    for y in range(0, height, cell):
        for x in range(0, width, cell):
            if (x // cell + y // cell) % 2:
                d.rectangle([x, y, x + cell - 1, y + cell - 1], fill=CHECKER_B + (255,))
    return bg


def save_png(canvas: Canvas, path: str | Path, scale: int = 1) -> Path:
    """Чистая картинка без украшений: то, что пойдёт в игру."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    img = to_image(canvas)
    if scale > 1:
        img = img.resize((canvas.width * scale, canvas.height * scale), Image.NEAREST)
    img.save(path)
    return path


def save_preview(
    canvas: Canvas,
    path: str | Path,
    scale: int = 8,
    grid: bool = True,
    grid_step: int = 8,
    checker: bool = True,
) -> Path:
    """Увеличенное превью с шахматкой и сеткой. Смотреть глазами, не грузить в игру."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    w, h = canvas.width * scale, canvas.height * scale
    art = to_image(canvas).resize((w, h), Image.NEAREST)
    base = _checkerboard(w, h, max(4, scale)) if checker else Image.new("RGBA", (w, h), (0, 0, 0, 255))
    base.alpha_composite(art)

    if grid and scale >= 4:
        layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        for x in range(0, canvas.width + 1, grid_step):
            color = AXIS_COLOR if x % (grid_step * 4) == 0 else GRID_COLOR
            d.line([(x * scale, 0), (x * scale, h)], fill=color, width=1)
        for y in range(0, canvas.height + 1, grid_step):
            color = AXIS_COLOR if y % (grid_step * 4) == 0 else GRID_COLOR
            d.line([(0, y * scale), (w, y * scale)], fill=color, width=1)
        base.alpha_composite(layer)

    base.convert("RGB").save(path)
    return path


def save_sheet(
    frames: list[Canvas],
    path: str | Path,
    columns: int = 0,
    scale: int = 1,
    padding: int = 0,
) -> Path:
    """Спрайт-лист: кадры в ряд для движка."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = columns or len(frames)
    rows = (len(frames) + cols - 1) // cols
    fw, fh = frames[0].width, frames[0].height
    cw, ch = fw + padding * 2, fh + padding * 2
    sheet = Image.new("RGBA", (cw * cols * scale, ch * rows * scale), (0, 0, 0, 0))
    for i, f in enumerate(frames):
        img = to_image(f)
        if scale > 1:
            img = img.resize((fw * scale, fh * scale), Image.NEAREST)
        x = ((i % cols) * cw + padding) * scale
        y = ((i // cols) * ch + padding) * scale
        sheet.alpha_composite(img, (x, y))
    sheet.save(path)
    return path


def save_gif(
    frames: list[Canvas],
    path: str | Path,
    fps: int = 8,
    scale: int = 4,
    background: tuple[int, int, int] = (10, 8, 16),
    transparent: bool = False,
) -> Path:
    """GIF из индексов холста с одной палитрой на всю анимацию.

    Квантовать каждый кадр отдельно нельзя: у кадров получаются разные
    локальные палитры, и при склейке цвета разъезжаются в кислоту. Наша
    графика уже индексированная, поэтому пишем индексы как есть.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    palette = frames[0].palette
    n = len(palette)
    table = []
    for r, g, b in palette.colors:
        table += [r, g, b]
    table += list(background)  # индекс n отводим под прозрачные пиксели
    table += [0, 0, 0] * (256 - n - 1)

    imgs = []
    for f in frames:
        idx = np.where(f.data < 0, n, f.data).astype(np.uint8)
        img = Image.fromarray(idx, mode="P")
        img.putpalette(table)
        if scale > 1:
            img = img.resize((f.width * scale, f.height * scale), Image.NEAREST)
        imgs.append(img)

    extra = {}
    if transparent:
        # прозрачным объявляем служебный индекс, а кадры чистим перед
        # отрисовкой следующего, иначе планета оставляет за собой шлейф
        extra = {"transparency": n, "disposal": 2}
    else:
        extra = {"disposal": 1}

    imgs[0].save(
        path,
        save_all=True,
        append_images=imgs[1:],
        duration=int(1000 / fps),
        loop=0,
        optimize=False,
        **extra,
    )
    return path


def save_contact_sheet(
    items: list[tuple[str, Canvas]],
    path: str | Path,
    scale: int = 4,
    columns: int = 4,
    label_height: int = 16,
    gap: int = 8,
) -> Path:
    """Все ассеты рядом с подписями. Один взгляд на весь набор."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = min(columns, len(items))
    rows = (len(items) + cols - 1) // cols
    cw = max(c.width for _, c in items) * scale
    chh = max(c.height for _, c in items) * scale
    cell_w, cell_h = cw + gap, chh + gap + label_height
    sheet = Image.new("RGB", (cell_w * cols + gap, cell_h * rows + gap), (18, 16, 22))
    d = ImageDraw.Draw(sheet)
    for i, (name, c) in enumerate(items):
        img = to_image(c).resize((c.width * scale, c.height * scale), Image.NEAREST)
        x = gap + (i % cols) * cell_w
        y = gap + (i // cols) * cell_h
        tile = _checkerboard(img.width, img.height, max(4, scale))
        tile.alpha_composite(img)
        sheet.paste(tile.convert("RGB"), (x + (cw - img.width) // 2, y))
        d.text((x + 2, y + chh + 3), name, fill=(198, 194, 210), font=_font(11))
    sheet.save(path)
    return path


def save_palette_strip(palette, path: str | Path, swatch: int = 48) -> Path:
    """Полоска палитры с номерами: чтобы сверять индексы глазами."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(palette)
    img = Image.new("RGB", (swatch * n, swatch + 16), (18, 16, 22))
    d = ImageDraw.Draw(img)
    for i, rgb in enumerate(palette.colors):
        d.rectangle([i * swatch, 0, (i + 1) * swatch - 1, swatch - 1], fill=rgb)
        lum = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
        d.text((i * swatch + 4, swatch - 16), str(i), fill=(0, 0, 0) if lum > 120 else (255, 255, 255), font=_font(11))
        d.text((i * swatch + 4, swatch + 2), palette.labels[i][:9], fill=(198, 194, 210), font=_font(10))
    img.save(path)
    return path


def stack(*canvases: Canvas) -> Canvas:
    """Сложить слои сверху вниз в один холст."""
    base = canvases[0].copy()
    for c in canvases[1:]:
        base.paste(c)
    return base


def as_array(canvas: Canvas) -> np.ndarray:
    return canvas.data.copy()
