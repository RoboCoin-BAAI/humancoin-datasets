import numpy as np
from PIL import Image, ImageDraw


def draw_mesh_roi(canvas, uv, vertex_valid, depths, triangles, color):
    polygons = uv[triangles]
    valid = vertex_valid[triangles].all(-1)
    valid &= np.isfinite(polygons).all((-1, -2))
    valid &= np.abs(polygons).max((-1, -2)) <= 10000
    order = np.argsort(depths)[::-1]
    polygons = polygons[order[valid[order]]]
    if not len(polygons):
        return canvas
    polygons = np.trunc(polygons).astype(np.int64)
    left = max(0, int(np.floor(polygons[..., 0].min())) - 2)
    top = max(0, int(np.floor(polygons[..., 1].min())) - 2)
    right = min(canvas.width, int(np.ceil(polygons[..., 0].max())) + 3)
    bottom = min(canvas.height, int(np.ceil(polygons[..., 1].max())) + 3)
    if right <= left or bottom <= top:
        return canvas
    box = (left, top, right, bottom)
    layer = Image.new("RGBA", (right - left, bottom - top))
    painter = ImageDraw.Draw(layer)
    polygons = (polygons - [left, top]).reshape(-1, 6).tolist()
    fill, outline = ((*color, 90), (*color, 130))
    native = painter.draw
    fill_ink, outline_ink = (native.draw_ink(fill), native.draw_ink(outline))
    for points in polygons:
        native.draw_polygon(points, fill_ink, 1)
        native.draw_polygon(points, outline_ink, 0, 1)
    patch = Image.alpha_composite(canvas.crop(box).convert("RGBA"), layer).convert("RGB")
    canvas.paste(patch, (left, top))
    return canvas
