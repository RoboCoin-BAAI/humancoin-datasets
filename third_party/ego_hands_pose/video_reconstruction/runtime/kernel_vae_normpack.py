from types import MethodType

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from .kernel_vae_weight_layout import LayoutFP8Conv, _layout_valid_fp8


@triton.jit
def _normpack(
    X,
    HISTORY,
    GAMMA,
    PACKED,
    CACHE,
    SCALE,
    B: tl.constexpr,
    C: tl.constexpr,
    T: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    HC: tl.constexpr,
    CT: tl.constexpr,
    PC: tl.constexpr,
    PT: tl.constexpr,
    PH: tl.constexpr,
    PW: tl.constexpr,
    LT: tl.constexpr,
    LH: tl.constexpr,
    LW: tl.constexpr,
    XS0: tl.constexpr,
    XS1: tl.constexpr,
    XS2: tl.constexpr,
    XS3: tl.constexpr,
    XS4: tl.constexpr,
    HS0: tl.constexpr,
    HS1: tl.constexpr,
    HS2: tl.constexpr,
    HS3: tl.constexpr,
    HS4: tl.constexpr,
    NORM_SCALE: tl.constexpr,
    ROWS: tl.constexpr,
    RC: tl.constexpr,
):
    i = tl.program_id(0) * ROWS + tl.arange(0, ROWS)[:, None]
    c = tl.arange(0, RC)[None, :]
    col = i % PW - LW
    row = i // PW % PH - LH
    time = i // (PW * PH) % PT - LT
    batch = i // (PW * PH * PT)
    valid = (i < B * PT * PH * PW) & (row >= 0) & (row < H) & (col >= 0) & (col < W)
    current = valid & (time >= 0) & (time < T)
    x = tl.load(
        X + batch * XS0 + c * XS1 + time * XS2 + row * XS3 + col * XS4, current & (c < C), other=0.0
    ).to(tl.float32)
    gamma = tl.load(GAMMA + c, c < C, other=0.0).to(tl.float32)
    square = x * x
    denominator = tl.maximum(tl.sqrt_rn(tl.sum(square, 1)[:, None]), 1e-12)
    normalized = x / denominator
    scaled = normalized * NORM_SCALE
    weighted = scaled * gamma
    biased = weighted + 0.0
    activated = (biased / (1.0 + libdevice.exp(-biased))).to(tl.float16)
    if HC > 0:
        past = tl.load(
            HISTORY + batch * HS0 + c * HS1 + (time + HC) * HS2 + row * HS3 + col * HS4,
            valid & (time < 0) & (time >= -HC) & (c < C),
            other=0.0,
        )
        activated = tl.where(time < 0, past, activated)
    value = tl.maximum(tl.minimum(activated.to(tl.float32) / tl.load(SCALE), 448.0), -448.0)
    tl.store(PACKED + i * PC + c, value, (i < B * PT * PH * PW) & (c < PC))
    cache_offset = (((batch * CT + time - (T - CT)) * H + row) * W + col) * C + c
    tl.store(CACHE + cache_offset, activated, current & (time >= T - CT) & (c < C))


def prepare_normpack(norm, conv, x, history=None, *, rows=8, warps=4):
    if not isinstance(conv, LayoutFP8Conv) or x.dtype != torch.float16:
        raise TypeError("Normpack requires the default NK FP8 convolution and FP16 activation")
    norm = norm.norm
    if not norm.channel_first or isinstance(norm.bias, torch.Tensor):
        raise ValueError("Only channel-first, scalar-zero-bias Wan RMS normalization is measured")
    if norm.bias != 0.0 or conv.implementation != "native" or conv.layout != "channels_last":
        raise ValueError("Unsupported normalization/packing recipe")
    b, c, t, h, w = x.shape
    lw, rw, lh, rh, lt, rt = conv.causal_padding
    pc = triton.cdiv(c, conv.alignment) * conv.alignment
    pt, ph, pw = (t + lt + rt, h + lh + rh, w + lw + rw)
    ct = min(t, 2)
    packed = torch.empty(
        (b, pc, pt, ph, pw),
        device=x.device,
        dtype=torch.float8_e4m3fn,
        memory_format=torch.channels_last_3d,
    )
    cache = torch.empty(
        (b, c, ct, h, w), device=x.device, dtype=x.dtype, memory_format=torch.channels_last_3d
    )
    hc = history.shape[2] if history is not None and lt > 0 else 0
    conv.last_normpack_kernel = _normpack[triton.cdiv(b * pt * ph * pw, rows),](
        x,
        history if hc else x,
        norm.gamma,
        packed,
        cache,
        conv.activation_scale,
        b,
        c,
        t,
        h,
        w,
        hc,
        ct,
        pc,
        pt,
        ph,
        pw,
        lt,
        lh,
        lw,
        *x.stride(),
        *(history.stride() if hc else (0,) * 5),
        norm.scale,
        rows,
        triton.next_power_of_2(max(c, pc)),
        num_warps=warps,
        enable_fp_fusion=True,
    )
    return (packed, cache)


def convolve_prepacked(conv, packed, *, residual=None, output_dtype=torch.float16):
    b, ci, t, h, w = packed.shape
    kt, kh, kw = conv.kernel_size
    st, sh, sw = conv.stride
    dt, dh, dw = conv.dilation
    ot, oh, ow = (
        (t - dt * (kt - 1) - 1) // st + 1,
        (h - dh * (kh - 1) - 1) // sh + 1,
        (w - dw * (kw - 1) - 1) // sw + 1,
    )
    if residual is not None and (
        residual.shape != (b, conv.out_channels, ot, oh, ow) or residual.dtype != output_dtype
    ):
        raise ValueError("Residual shape/dtype mismatch")
    y = torch.empty((b, ot, oh, ow, conv.out_channels), device=packed.device, dtype=output_dtype)
    bm, bn, bk = conv.block
    conv.last_kernel = _layout_valid_fp8[
        triton.cdiv(b * ot * oh * ow, bm), triton.cdiv(conv.out_channels, bn)
    ](
        packed,
        conv.fp8_weight,
        y,
        conv.bias if conv.bias is not None else y,
        conv.activation_scale,
        conv.weight_scale,
        residual if residual is not None else y,
        *(residual.stride() if residual is not None else (0,) * 5),
        b,
        t,
        h,
        w,
        ci,
        conv.out_channels,
        ot,
        oh,
        ow,
        kt,
        kh,
        kw,
        st,
        sh,
        sw,
        dt,
        dh,
        dw,
        conv.bias is not None,
        bm,
        bn,
        bk,
        {"direct": 0, "incremental": 2}[conv.mode],
        residual is not None,
        conv.group_m,
        num_warps=conv.num_warps,
        num_stages=conv.num_stages,
        **{"maxnreg": conv.maxnreg} if conv.maxnreg is not None else {},
    )
    return y.permute(0, 4, 1, 2, 3)


def residual_normpack(self, x, feat_cache=None, feat_idx=None):
    feat_idx = [0] if feat_idx is None else feat_idx
    shortcut = self.conv_shortcut(x)
    for name in ("conv1", "conv2"):
        conv = getattr(self, name)
        norm = getattr(self, "norm1" if name == "conv1" else "norm2")
        last = name == "conv2"
        idx = feat_idx[0]
        past = feat_cache[idx] if feat_cache is not None else None
        measured_shape = (
            x.dtype == torch.float16
            and x.shape[0] == 1
            and (x.shape[1] == 160)
            and (x.shape[2] in (1, 8))
            and (x.shape[3:] == (240, 240))
            and (x.stride(1) == 1)
        )
        if name in self._normpack_choices and measured_shape:
            packed, history = prepare_normpack(norm, conv, x, past, **self._normpack_choices[name])
            if history.shape[2] < 2 and past is not None:
                history = torch.cat((past[:, :, -1:], history), 2)
            x = convolve_prepacked(conv, packed, residual=shortcut if last else None)
        else:
            x = self.nonlinearity(norm(x))
            if last:
                x = self.dropout(x)
            history = x[:, :, -2:]
            if history.shape[2] < 2 and past is not None:
                history = torch.cat((past[:, :, -1:], history), 2)
            x = conv(x, past, residual=shortcut) if last else conv(x, past)
        if feat_cache is not None:
            feat_cache[idx] = history
            feat_idx[0] += 1
    return x


def install_normpack(vae, choices):
    measured = {f"encoder.down_blocks.0.resnets.{r}.conv{c}" for r in (0, 1) for c in (1, 2)}
    for name, choice in choices.items():
        if name not in measured or choice not in ({"rows": 4, "warps": 4}, {"rows": 8, "warps": 8}):
            raise ValueError(f"Unmeasured normpack layer/configuration: {name}")
    for name, choice in choices.items():
        parent, attribute = name.rsplit(".", 1)
        block = vae.get_submodule(parent)
        if block.training or block.dropout.p != 0:
            raise ValueError("Normpack requires inference and zero dropout")
        conv = getattr(block, attribute)
        if not isinstance(conv, LayoutFP8Conv) or not isinstance(block.conv2, LayoutFP8Conv):
            raise ValueError("Normpack requires an existing NK residual block")
        if not hasattr(block, "_normpack_choices"):
            block._normpack_choices = {}
            block.forward = MethodType(residual_normpack, block)
        block._normpack_choices[attribute] = choice
    vae.clear_cache()
    return list(choices)
