import numpy as np
from PIL import Image, ImageDraw

COLORS = ((30, 235, 140), (255, 80, 175))
METRIC = "heatmap_one_minus_normalized_entropy_v1"
HEATMAP_FIELDS = ("joint_heatmaps", "heatmap_frame_indices", "window_starts", "window_counts")


def heatmap_concentration(heatmaps):
    values = np.asarray(heatmaps, dtype=np.float32)
    if values.shape[-1] * values.shape[-2] <= 1:
        return np.full(values.shape[:-2], np.nan, dtype=np.float32)
    good = np.isfinite(values).all((-1, -2)) & (values >= 0).all((-1, -2))
    mass = values.sum((-1, -2), dtype=np.float64)
    p = values / np.maximum(mass[..., None, None], 1e-30)
    entropy = -(p * np.log(np.maximum(p, 1e-30))).sum((-1, -2))
    score = np.clip(1 - entropy / np.log(values.shape[-1] * values.shape[-2]), 0, 1)
    return np.where(good & (mass > 0), score, np.nan).astype(np.float32)


def joint_concentration(result):
    if "overlay_joint_concentration" in result:
        return result["overlay_joint_concentration"]
    if not all((key in result for key in HEATMAP_FIELDS)):
        return None
    n = len(result["frame_indices"])
    sums = np.zeros((n, 2, 21), np.float64)
    weights = np.zeros_like(sums)
    maps = result["joint_heatmaps"]
    if maps.ndim != 6 or maps.shape[2:4] != (2, 21):
        raise ValueError("Expected window/native-knot/hand/joint/H/W heatmaps")
    for ordinal, (start, count) in enumerate(
        zip(result["window_starts"], result["window_counts"], strict=True)
    ):
        start, count = (int(start), int(count))
        if start < 0 or count < 1 or start + count > n:
            raise ValueError("Heatmap window outside prediction frames")
        offset = start % 4
        native = 1 + (offset + count - 1 + 3) // 4
        knots = np.asarray(result["heatmap_frame_indices"][ordinal, :native])
        if len(knots) != native or not np.array_equal(knots, np.arange(native) * 4 - offset):
            raise ValueError("Heatmap knots do not match the saved latent window coordinates")
        scores = heatmap_concentration(maps[ordinal, :native])
        times = np.arange(count)
        triangular = np.minimum(times + 1, count - times)
        for hand in (0, 1):
            for joint in range(21):
                values = np.interp(times, knots, scores[:, hand, joint])
                valid = np.isfinite(values)
                weight = triangular * valid
                sums[start : start + count, hand, joint] += np.nan_to_num(values) * weight
                weights[start : start + count, hand, joint] += weight
    output = np.full(sums.shape, np.nan, np.float32)
    np.divide(sums, weights, out=output, where=weights > 0)
    return output


def concentration_colors(scores):
    score = np.clip(np.nan_to_num(scores), 0, 1)
    red = np.interp(score, [0, 0.5, 1], [240, 250, 40])
    green = np.interp(score, [0, 0.5, 1], [60, 210, 225])
    blue = np.interp(score, [0, 0.5, 1], [60, 50, 110])
    rgb = np.rint(np.stack((red, green, blue), -1)).astype(np.uint8)
    return np.where(np.isfinite(scores)[..., None], rgb, 145).astype(np.uint8)


def indicator_label(result):
    return (
        "Bars: L/R on-screen | joints: heatmap concentration (red low / yellow mid / green high)"
        if "overlay_joint_concentration" in result
        else "Bars: L/R on-screen | joint concentration unavailable"
    )


def prepare_indicators(result, metadata):
    scores = joint_concentration(result)
    if scores is not None:
        result = dict(result, overlay_joint_concentration=scores)
    metadata["overlay_indicators"] = dict(
        hand_bar_field="existence_prob",
        hand_order=["left", "right"],
        bar_range=[0, 1],
        joint_color_metric=METRIC if scores is not None else None,
        joint_color_range=[0, 1],
        joint_color_stops=["red", "yellow", "green"],
        missing_color="gray",
        calibrated_joint_probability=False,
        interpolation="native_window_knots_then_triangular_overlap_fusion",
        prediction_modified=False,
    )
    return result


def bar_panels(probabilities, width, height, font):
    pad = max(2, round(font.size * 0.25))
    numeric = width >= 160
    pw = min(width // 4, max(12, int(np.ceil(font.getlength("100%"))) + 2 * pad))
    ph = max(16, min(height, round(height * 0.65)))
    top = (height - ph) // 2
    panels = []
    for hand, probability in enumerate(probabilities):
        image = Image.new("RGBA", (pw, ph), (12, 15, 20, 175))
        draw = ImageDraw.Draw(image)
        line = sum(font.getmetrics())
        label = "L" if hand == 0 else "R"
        draw.text(((pw - font.getlength(label)) / 2, 1), label, fill=COLORS[hand], font=font)
        y0 = min(ph - 2, line + pad)
        y1 = max(y0, ph - (line + pad if numeric else pad) - 1)
        bw = max(3, round(min(width, height) * 8 / 480))
        x0 = max(1, (pw - bw) // 2)
        x1 = min(pw - 2, x0 + bw - 1)
        draw.rectangle((x0, y0, x1, y1), fill=(65, 70, 80, 255))
        valid = np.isfinite(probability)
        value = float(np.clip(probability, 0, 1)) if valid else 0
        filled = round((y1 - y0 + 1) * value)
        if filled:
            draw.rectangle((x0, y1 - filled + 1, x1, y1), fill=(*COLORS[hand], 255))
        if numeric:
            text = f"{value:.0%}" if valid else "--"
            draw.text(
                ((pw - font.getlength(text)) / 2, ph - line - 1), text, fill="white", font=font
            )
        panels.append(np.asarray(image))
    return (np.stack(panels), top)


def draw_bars(canvas, probabilities, font):
    panels, top = bar_panels(probabilities, canvas.width, canvas.height, font)
    for hand, pixels in enumerate(panels):
        image = Image.fromarray(pixels)
        left = 0 if hand == 0 else canvas.width - image.width
        background = canvas.crop((left, top, left + image.width, top + image.height)).convert(
            "RGBA"
        )
        canvas.paste(Image.alpha_composite(background, image).convert("RGB"), (left, top))
