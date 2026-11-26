import triton
import triton.language as tl


@triton.jit
def _rgb_yuv(r, g, b, KR: tl.constexpr, KB: tl.constexpr, FULL: tl.constexpr):
    kg = 1.0 - KR - KB
    y = KR * r + kg * g + KB * b
    u = (b - y) / (2.0 * (1.0 - KB))
    v = (r - y) / (2.0 * (1.0 - KR))
    if FULL:
        return (y, u + 128.0, v + 128.0)
    return (y * (219.0 / 255.0) + 16.0, u * (224.0 / 255.0) + 128.0, v * (224.0 / 255.0) + 128.0)


@triton.jit(do_not_specialize=["RX", "RY", "RW", "RH", "FACES", "TW", "TH", "BW", "BH", "BY"])
def _composite(
    Y,
    UV,
    Rast,
    Altitude,
    Skeleton,
    Text,
    Palette,
    Bars,
    W: tl.constexpr,
    H: tl.constexpr,
    YS: tl.constexpr,
    US: tl.constexpr,
    RX,
    RY,
    RW,
    RH,
    FACES,
    TW,
    TH,
    BW,
    BH,
    BY,
    COLORED: tl.constexpr,
    HUD: tl.constexpr,
    MESH: tl.constexpr,
    KR: tl.constexpr,
    KB: tl.constexpr,
    FULL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    cell = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    batch = tl.program_id(1)
    keep = cell < W * H // 4
    x = cell % (W // 2) * 2
    y = cell // (W // 2) * 2
    base = batch * US + y // 2 * W + x
    orig_u = tl.load(UV + base, keep, other=128).to(tl.float32)
    orig_v = tl.load(UV + base + 1, keep, other=128).to(tl.float32)
    sum_u = tl.full((BLOCK,), 0, tl.float32)
    sum_v = tl.full((BLOCK,), 0, tl.float32)
    for corner in tl.static_range(4):
        px = x + corner % 2
        py = y + corner // 2
        iy = batch * YS + py * W + px
        yy = tl.load(Y + iy, keep, other=16).to(tl.float32)
        uu, vv = (orig_u, orig_v)
        inside = keep & (px >= RX) & (px < RX + RW) & (py >= RY) & (py < RY + RH)
        roi = (py - RY) * RW + px - RX
        for hand in tl.static_range(2):
            layer = (batch * 2 + hand) * RH * RW + roi
            alpha = tl.full((BLOCK,), 0, tl.float32)
            if MESH:
                rid = tl.load(Rast + layer * 4 + 3, inside, other=0).to(tl.int32) - 1
                b0 = tl.load(Rast + layer * 4, inside, other=0)
                b1 = tl.load(Rast + layer * 4 + 1, inside, other=0)
                ap = ((batch * 2 + hand) * FACES + rid) * 3
                a0 = tl.load(Altitude + ap, inside & (rid >= 0), other=0)
                a1 = tl.load(Altitude + ap + 1, inside & (rid >= 0), other=0)
                a2 = tl.load(Altitude + ap + 2, inside & (rid >= 0), other=0)
                edge = tl.minimum(tl.minimum(b0 * a0, b1 * a1), (1.0 - b0 - b1) * a2)
                alpha = tl.where(
                    inside & (rid >= 0), tl.where(edge <= 0.75, 130.0 / 255.0, 90.0 / 255.0), 0.0
                )
            skel = tl.load(Skeleton + layer, inside, other=0)
            alpha = tl.where(skel != 0, 1.0, alpha)
            if COLORED:
                cp = ((batch * 2 + hand) * 23 + tl.minimum(skel, 22)) * 3
                r = tl.load(Palette + cp, keep, other=0).to(tl.float32)
                g = tl.load(Palette + cp + 1, keep, other=0).to(tl.float32)
                b = tl.load(Palette + cp + 2, keep, other=0).to(tl.float32)
                cy, cu, cv = _rgb_yuv(r, g, b, KR, KB, FULL)
            elif hand == 0:
                cy, cu, cv = _rgb_yuv(30.0, 235.0, 140.0, KR, KB, FULL)
            else:
                cy, cu, cv = _rgb_yuv(255.0, 80.0, 175.0, KR, KB, FULL)
            yy = yy + alpha * (cy - yy)
            uu = uu + alpha * (cu - uu)
            vv = vv + alpha * (cv - vv)
        text_valid = keep & (px < TW) & (py < TH)
        off = ((batch * TH + py) * TW + px) * 4
        r = tl.load(Text + off, text_valid, other=0).to(tl.float32)
        g = tl.load(Text + off + 1, text_valid, other=0).to(tl.float32)
        b = tl.load(Text + off + 2, text_valid, other=0).to(tl.float32)
        alpha = tl.load(Text + off + 3, text_valid, other=0).to(tl.float32) / 255.0
        cy, cu, cv = _rgb_yuv(r, g, b, KR, KB, FULL)
        yy = yy + alpha * (cy - yy)
        uu = uu + alpha * (cu - uu)
        vv = vv + alpha * (cv - vv)
        if HUD:
            side = tl.where(px < BW, 0, 1)
            bx = tl.where(side == 0, px, px - (W - BW))
            by = py - BY
            valid_bar = keep & (bx >= 0) & (bx < BW) & (by >= 0) & (by < BH)
            off = (((batch * 2 + side) * BH + by) * BW + bx) * 4
            r = tl.load(Bars + off, valid_bar, other=0).to(tl.float32)
            g = tl.load(Bars + off + 1, valid_bar, other=0).to(tl.float32)
            b = tl.load(Bars + off + 2, valid_bar, other=0).to(tl.float32)
            alpha = tl.load(Bars + off + 3, valid_bar, other=0).to(tl.float32) / 255.0
            cy, cu, cv = _rgb_yuv(r, g, b, KR, KB, FULL)
            yy = yy + alpha * (cy - yy)
            uu = uu + alpha * (cu - uu)
            vv = vv + alpha * (cv - vv)
        tl.store(Y + iy, tl.minimum(tl.maximum(tl.floor(yy + 0.5), 0), 255).to(tl.uint8), keep)
        sum_u += uu
        sum_v += vv
    tl.store(
        UV + base, tl.minimum(tl.maximum(tl.floor(sum_u * 0.25 + 0.5), 0), 255).to(tl.uint8), keep
    )
    tl.store(
        UV + base + 1,
        tl.minimum(tl.maximum(tl.floor(sum_v * 0.25 + 0.5), 0), 255).to(tl.uint8),
        keep,
    )


def composite_nv12(
    y,
    uv,
    rast,
    altitudes,
    skeleton,
    text,
    box,
    palette=None,
    bars=None,
    bar_top=0,
    *,
    matrix=6,
    full=False,
):
    batch, height, width = y.shape
    left, top, rw, rh = box
    kr, kb = (0.2126, 0.0722) if matrix == 1 else (0.299, 0.114)
    _composite[triton.cdiv(width * height // 4, 256), batch](
        y,
        uv,
        rast if rast is not None else y,
        altitudes if altitudes is not None else y,
        skeleton,
        text,
        palette if palette is not None else y,
        bars if bars is not None else y,
        width,
        height,
        y.stride(0),
        uv.stride(0),
        left,
        top,
        rw,
        rh,
        altitudes.shape[1] if altitudes is not None else 0,
        text.shape[2],
        text.shape[1],
        bars.shape[3] if bars is not None else 0,
        bars.shape[2] if bars is not None else 0,
        bar_top,
        palette is not None,
        bars is not None,
        rast is not None,
        kr,
        kb,
        full,
        256,
        num_warps=4,
    )
