from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _pack(
    X,
    HISTORY,
    Y,
    SCALE,
    C: tl.constexpr,
    T: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    HC: tl.constexpr,
    XS0: tl.constexpr,
    XS1: tl.constexpr,
    XS2: tl.constexpr,
    XS3: tl.constexpr,
    XS4: tl.constexpr,
    CS0: tl.constexpr,
    CS1: tl.constexpr,
    CS2: tl.constexpr,
    CS3: tl.constexpr,
    CS4: tl.constexpr,
    LEFT_T: tl.constexpr,
    LEFT_H: tl.constexpr,
    LEFT_W: tl.constexpr,
    PT: tl.constexpr,
    PH: tl.constexpr,
    PW: tl.constexpr,
    PC: tl.constexpr,
    KT: tl.constexpr,
    ST: tl.constexpr,
    DT: tl.constexpr,
    LOWER: tl.constexpr,
    CHANNEL_LAST: tl.constexpr,
    QUANTIZE: tl.constexpr,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    if CHANNEL_LAST:
        ci = index % PC
        spatial = index // PC
        wi = spatial % PW
        hi = spatial // PW % PH
        bt = spatial // (PW * PH)
    else:
        wi = index % PW
        hi = index // PW % PH
        if LOWER:
            ci = index // (PW * PH) % PC
            bt = index // (PW * PH * PC)
        else:
            ti = index // (PW * PH) % PT
            ci = index // (PW * PH * PT) % PC
            bi = index // (PW * PH * PT * PC)
            bt = bi * PT + ti
    bi = bt // PT
    ti = bt % PT
    if LOWER:
        ch = ci // KT
        tap = ci % KT
        time = ti * ST + tap * DT - LEFT_T
        real = ch < C
    else:
        ch = ci
        time = ti - LEFT_T
        real = ch < C
    row = hi - LEFT_H
    col = wi - LEFT_W
    valid = (index < N) & real & (row >= 0) & (row < H) & (col >= 0) & (col < W)
    value = tl.load(
        X + bi * XS0 + ch * XS1 + time * XS2 + row * XS3 + col * XS4,
        valid & (time >= 0) & (time < T),
        other=0,
    )
    if HC > 0:
        past = tl.load(
            HISTORY + bi * CS0 + ch * CS1 + (time + HC) * CS2 + row * CS3 + col * CS4,
            valid & (time < 0) & (time >= -HC),
            other=0,
        )
        value = tl.where(time < 0, past, value)
    if QUANTIZE:
        value = tl.maximum(tl.minimum(value.to(tl.float32) / tl.load(SCALE), 448.0), -448.0)
    tl.store(Y + index, value, index < N)


def fused_pack(conv, x, cache=None, *, implementation=None, layout=None, quant_scale=None):
    implementation = implementation or conv.implementation
    layout = layout or conv.layout
    if conv.ndim != 3:
        raise ValueError("fused_pack expects a 3D convolution")
    b, c, t, h, w = x.shape
    causal = conv.causal_padding or (0, 0, 0, 0, 0, 0)
    lw, rw, lh, rh, lt, rt = causal
    lower = implementation == "conv2d"
    kt = conv.kernel_size[0]
    dt, st = (conv.dilation[0], conv.stride[0])
    if lower:
        lt += conv.padding[0]
        rt += conv.padding[0]
        pt = (t + lt + rt - dt * (kt - 1) - 1) // st + 1
        pc = triton.cdiv(c * kt, conv.alignment) * conv.alignment
        shape = (b * pt, pc, h + lh + rh, w + lw + rw)
    else:
        pt = t + lt + rt
        pc = triton.cdiv(c, conv.alignment) * conv.alignment
        shape = (b, pc, pt, h + lh + rh, w + lw + rw)
    cl = layout == "channels_last"
    out = torch.empty(
        shape,
        device=x.device,
        dtype=getattr(torch, conv.compute_dtype),
        memory_format=(torch.channels_last if lower else torch.channels_last_3d)
        if cl
        else torch.contiguous_format,
    )
    history = (
        cache if cache is not None and conv.causal_padding is not None and (causal[4] > 0) else None
    )
    strides = history.stride() if history is not None else (0,) * 5
    _pack[triton.cdiv(out.numel(), 512),](
        x,
        history if history is not None else x,
        out,
        quant_scale if quant_scale is not None else x,
        c,
        t,
        h,
        w,
        history.shape[2] if history is not None else 0,
        *x.stride(),
        *strides,
        lt,
        lh,
        lw,
        pt,
        shape[-2],
        shape[-1],
        pc,
        kt,
        st,
        dt,
        lower,
        cl,
        quant_scale is not None,
        out.numel(),
        512,
    )
    return (out, pt)


@triton.jit
def _implicit_fp8(
    X,
    W,
    Y,
    BIAS,
    SCALE_X,
    SCALE_W,
    B: tl.constexpr,
    T: tl.constexpr,
    H: tl.constexpr,
    WIDTH: tl.constexpr,
    CI: tl.constexpr,
    CO: tl.constexpr,
    OT: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    KT: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    ST: tl.constexpr,
    SH: tl.constexpr,
    SW: tl.constexpr,
    PT: tl.constexpr,
    PH: tl.constexpr,
    PW: tl.constexpr,
    DT: tl.constexpr,
    DH: tl.constexpr,
    DW: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    k0 = tl.arange(0, BK)
    total = KT * KH * KW * CI
    ow = m % OW
    oh = m // OW % OH
    ot = m // (OW * OH) % OT
    batch = m // (OW * OH * OT)
    acc = tl.full((BM, BN), 0, tl.float32)
    for start in range(tl.cdiv(total, BK)):
        k = start * BK + k0
        c = k % CI
        kw = k // CI % KW
        kh = k // (CI * KW) % KH
        kt = k // (CI * KW * KH)
        it = ot[:, None] * ST + kt[None, :] * DT - PT
        ih = oh[:, None] * SH + kh[None, :] * DH - PH
        iw = ow[:, None] * SW + kw[None, :] * DW - PW
        mask = (
            (m[:, None] < B * OT * OH * OW)
            & (k[None, :] < total)
            & (it >= 0)
            & (it < T)
            & (ih >= 0)
            & (ih < H)
            & (iw >= 0)
            & (iw < WIDTH)
        )
        a = tl.load(
            X + (((batch[:, None] * T + it) * H + ih) * WIDTH + iw) * CI + c[None, :],
            mask,
            other=0.0,
        )
        b = tl.load(
            W + k[:, None] * CO + n[None, :], (k[:, None] < total) & (n[None, :] < CO), other=0.0
        )
        acc = tl.dot(a, b, acc)
    sx = tl.load(SCALE_X)
    sw = tl.load(SCALE_W + n, n < CO, other=0)
    value = acc * sx * sw[None, :]
    if HAS_BIAS:
        value += tl.load(BIAS + n, n < CO, other=0)[None, :]
    tl.store(
        Y + m[:, None] * CO + n[None, :], value, (m[:, None] < B * OT * OH * OW) & (n[None, :] < CO)
    )


@triton.jit
def _rope_qk(
    Q,
    K,
    F,
    QO,
    KO,
    N: tl.constexpr,
    S: tl.constexpr,
    HEADS: tl.constexpr,
    D: tl.constexpr,
    Q0: tl.constexpr,
    Q1: tl.constexpr,
    Q2: tl.constexpr,
    Q3: tl.constexpr,
    K0: tl.constexpr,
    K1: tl.constexpr,
    K2: tl.constexpr,
    K3: tl.constexpr,
    FP64: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    pair = i % (D // 2)
    head = i // (D // 2) % HEADS
    pos = i // (D // 2 * HEADS) % S
    batch = i // (D // 2 * HEADS * S)
    offsetq = batch * Q0 + pos * Q1 + head * Q2 + 2 * pair * Q3
    offsetk = batch * K0 + pos * K1 + head * K2 + 2 * pair * K3
    qr = tl.load(Q + offsetq, i < N, other=0).to(tl.float32)
    qi = tl.load(Q + offsetq + Q3, i < N, other=0).to(tl.float32)
    kr = tl.load(K + offsetk, i < N, other=0).to(tl.float32)
    ki = tl.load(K + offsetk + K3, i < N, other=0).to(tl.float32)
    co = tl.load(F + pos * D + 2 * pair, i < N, other=0)
    si = tl.load(F + pos * D + 2 * pair + 1, i < N, other=0)
    if FP64:
        qr = qr.to(tl.float64)
        qi = qi.to(tl.float64)
        kr = kr.to(tl.float64)
        ki = ki.to(tl.float64)
    else:
        co = co.to(tl.float32)
        si = si.to(tl.float32)
    qreal = (qr * co - qi * si).to(tl.float32)
    qimag = (qr * si + qi * co).to(tl.float32)
    kreal = (kr * co - ki * si).to(tl.float32)
    kimag = (kr * si + ki * co).to(tl.float32)
    tl.store(QO + 2 * i, qreal, i < N)
    tl.store(QO + 2 * i + 1, qimag, i < N)
    tl.store(KO + 2 * i, kreal, i < N)
    tl.store(KO + 2 * i + 1, kimag, i < N)


def fused_rope(q, k, phase, *, fp64=False, output_dtype=torch.bfloat16):
    if not phase.is_complex() or phase.shape != (q.shape[1], 1, q.shape[-1] // 2):
        raise ValueError("fused RoPE requires the actual homogeneous-grid complex phase")
    if q.shape != k.shape or q.shape[-1] % 2:
        raise ValueError("invalid Q/K shape")
    qo = torch.empty(q.shape, device=q.device, dtype=output_dtype)
    ko = torch.empty_like(qo)
    n = q.numel() // 2
    real = torch.view_as_real(phase)
    _rope_qk[triton.cdiv(n, 256),](
        q,
        k,
        real,
        qo,
        ko,
        n,
        *q.shape[1:],
        *q.stride(),
        *k.stride(),
        fp64,
        256,
        enable_fp_fusion=False,
        num_warps=4,
    )
    return (qo, ko)
