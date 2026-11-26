from functools import lru_cache
from math import ceil

import numpy as np
from PIL import Image, ImageDraw, ImageFont


@lru_cache(maxsize=24)
def label_font(width, height):
    size = max(8, round(14 * min(width, height) / 480))
    font = ImageFont.load_default(size=size)
    font.getmask2 = lru_cache(maxsize=64)(font.getmask2)
    return font


def label_layout(title, label, font, width, height=None):
    size = getattr(font, "size", 10)
    padding = max(4, round(6 * size / 14))
    gap = max(2, round(4 * size / 14))
    line_height = sum(font.getmetrics())
    available = max(1, width - 2 * padding)

    def wrap(text):
        if font.getlength(text) <= available:
            return [text]
        lines, line = ([], "")
        for word in text.split():
            candidate = f"{line} {word}" if line else word
            if line and font.getlength(candidate) > available:
                lines.append(line)
                line = word
            else:
                line = candidate
        if line:
            lines.append(line)
        return lines

    title_lines = wrap(title)
    lines = title_lines + (wrap(label) if label is not None else [])
    panel_width = min(width, ceil(max((font.getlength(line) for line in lines))) + 2 * padding)
    panel_height = 2 * padding + len(lines) * line_height + (len(lines) - 1) * gap
    if height is not None:
        panel_height = min(height, panel_height)
    return dict(
        width=panel_width,
        height=panel_height,
        padding=padding,
        gap=gap,
        line_height=line_height,
        lines=lines,
        title_lines=len(title_lines),
    )


class RGBCanvas:
    def __init__(self, rgb):
        self.rgb = np.asarray(rgb)
        if self.rgb.ndim != 3 or self.rgb.shape[-1] != 3:
            self.rgb = np.asarray(Image.fromarray(self.rgb).convert("RGB"))
        if not self.rgb.flags.writeable:
            self.rgb = self.rgb.copy()
        self.height, self.width = self.rgb.shape[:2]

    def crop(self, box):
        left, top, right, bottom = box
        return Image.fromarray(self.rgb[top:bottom, left:right])

    def paste(self, patch, position):
        left, top = position
        self.rgb[top : top + patch.height, left : left + patch.width] = np.asarray(patch)


class PillowCanvas:
    def __init__(self, rgb):
        self.image = Image.fromarray(rgb)
        if self.image.mode != "RGB":
            self.image = self.image.convert("RGB")
        self.width, self.height = self.image.size

    @property
    def rgb(self):
        return np.asarray(self.image)

    def crop(self, box):
        return self.image.crop(box)

    def paste(self, patch, position):
        self.image.paste(patch, position)


def draw_skeleton_roi(canvas, uv, valid, edges, color, joint_colors=None):
    if isinstance(canvas, PillowCanvas):
        _paint_skeleton(
            ImageDraw.Draw(canvas.image),
            uv,
            valid,
            edges,
            color,
            canvas.width,
            canvas.height,
            joint_colors,
        )
        return
    points = uv[valid]
    if not len(points):
        return
    margin = (
        max(4, round(3 * min(canvas.width, canvas.height) / 480) + 1)
        if joint_colors is not None
        else 4
    )
    left = max(0, int(np.floor(points[:, 0].min())) - margin)
    top = max(0, int(np.floor(points[:, 1].min())) - margin)
    right = min(canvas.width, int(np.ceil(points[:, 0].max())) + margin + 1)
    bottom = min(canvas.height, int(np.ceil(points[:, 1].max())) + margin + 1)
    if right <= left or bottom <= top:
        return
    mask = Image.new("1" if joint_colors is None else "L", (canvas.width, canvas.height))
    painter = ImageDraw.Draw(mask)
    _paint_skeleton(
        painter,
        uv,
        valid,
        edges,
        1,
        canvas.width,
        canvas.height,
        None if joint_colors is None else list(range(2, 23)),
    )
    occupied = np.asarray(mask.crop((left, top, right, bottom)))
    if joint_colors is None:
        canvas.rgb[top:bottom, left:right][occupied] = color
    else:
        palette = np.vstack([color, color, joint_colors]).astype(np.uint8)
        canvas.rgb[top:bottom, left:right][occupied != 0] = palette[occupied[occupied != 0]]


def draw_labels_roi(canvas, title, label, font):
    if isinstance(canvas, PillowCanvas):
        _paint_labels(ImageDraw.Draw(canvas.image), title, label, font, canvas.width, canvas.height)
        return
    layout = label_layout(title, label, font, canvas.width, canvas.height)
    patch = canvas.crop((0, 0, layout["width"], layout["height"]))
    painter = ImageDraw.Draw(patch)
    _paint_labels(painter, title, label, font, canvas.width, canvas.height)
    canvas.paste(patch, (0, 0))


def _paint_skeleton(painter, uv, valid, edges, fill, width, height, joint_fills=None):
    radius = max(3, round(3 * min(width, height) / 480)) if joint_fills is not None else 2
    for a, b in edges:
        if valid[a] and valid[b]:
            painter.line([tuple(uv[a]), tuple(uv[b])], fill=fill, width=2)
    for index, ((x, y), keep) in enumerate(zip(uv, valid, strict=True)):
        if keep and 0 <= x < width and (0 <= y < height):
            value = fill if joint_fills is None else joint_fills[index]
            if isinstance(value, np.ndarray):
                value = tuple(map(int, value))
            painter.ellipse((x - radius, y - radius, x + radius, y + radius), fill=value)


def _paint_labels(painter, title, label, font, width, height=None):
    layout = label_layout(title, label, font, width, height)
    painter.rectangle((0, 0, layout["width"] - 1, layout["height"] - 1), fill=(12, 15, 20))
    for index, line in enumerate(layout["lines"]):
        y = layout["padding"] + index * (layout["line_height"] + layout["gap"])
        painter.text((layout["padding"], y), line, fill="white", font=font)
