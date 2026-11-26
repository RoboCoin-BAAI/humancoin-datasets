import numpy as np

from .ffmpeg_video import resize_plan as original_resize_plan


def resize_plan(source_size, mode="long_edge_pad", *, long_edge=832, width=672, height=480):
    plan = original_resize_plan(source_size, mode, long_edge=long_edge, width=width, height=height)
    if mode != "long_edge_pad" or plan["size"] != [832, 448]:
        return plan
    old_px, old_py = plan["padding"][:2]
    plan["size"] = [832, 480]
    rw, rh = plan["resized_content_size"]
    px, py = ((832 - rw) // 2, (480 - rh) // 2)
    plan["padding"] = [px, py, 832 - rw - px, 480 - rh - py]
    plan["content_rect"] = [px, py, px + rw, py + rh]
    matrix = np.asarray(plan["source_to_canvas"], dtype=np.float64)
    matrix[0, 2] += px - old_px
    matrix[1, 2] += py - old_py
    plan["source_to_canvas"] = matrix.tolist()
    plan["canvas_to_source_matrix"] = np.linalg.inv(matrix).tolist()
    sx, sy = (matrix[0, 0], matrix[1, 1])
    plan["crop_affine"] = [-matrix[0, 2] / sx, -matrix[1, 2] / sy, sx, sy]
    filters = plan["ffmpeg_filter"].split(",")
    filters = [
        f"pad=832:480:{px}:{py}:color=black" if item.startswith("pad=") else item
        for item in filters
    ]
    plan["ffmpeg_filter"] = ",".join(filters)
    plan["expected_latent_grid"] = [30, 52]
    plan["canvas_padding_policy"] = "832x448_to_832x480_border_only"
    return plan
