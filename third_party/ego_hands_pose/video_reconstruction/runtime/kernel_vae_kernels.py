from __future__ import annotations

from types import MethodType

import torch
import triton
import triton.language as tl
from torch import nn

from .fp8_kernels import fused_pack
from .fp8_modules import ImplicitFP8Conv


@triton.jit
def _valid_fp8(
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
    DT: tl.constexpr,
    DH: tl.constexpr,
    DW: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    MODE: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pm, pn = (tl.program_id(0), tl.program_id(1))
    if GROUP_M > 0:
        count_m = tl.cdiv(B * OT * OH * OW, BM)
        count_n = tl.cdiv(CO, BN)
        pid = pm + pn * count_m
        group = pid // (GROUP_M * count_n)
        first = group * GROUP_M
        size = tl.minimum(count_m - first, GROUP_M)
        pm = first + pid % (GROUP_M * count_n) % size
        pn = pid % (GROUP_M * count_n) // size
    m = pm * BM + tl.arange(0, BM)
    n = pn * BN + tl.arange(0, BN)
    k0 = tl.arange(0, BK)
    total = KT * KH * KW * CI
    ow = m % OW
    oh = m // OW % OH
    ot = m // (OW * OH) % OT
    batch = m // (OW * OH * OT)
    base = (((batch * T + ot * ST) * H + oh * SH) * WIDTH + ow * SW) * CI
    acc = tl.full((BM, BN), 0, tl.float32)
    cb = tl.full((), 0, tl.int32)
    wbase = tl.full((), 0, tl.int32)
    hbase = tl.full((), 0, tl.int32)
    tbase = tl.full((), 0, tl.int32)
    for step in range(tl.cdiv(total, BK)):
        k = step * BK + k0
        if MODE == 2:
            c = cb + k0
            if CI % BK == 0:
                cw, ch, ct = (wbase, hbase, tbase)
            else:
                wrap = c >= CI
                c = c - tl.where(wrap, CI, 0)
                cw = wbase + wrap.to(tl.int32)
                ch = hbase + (cw >= KW).to(tl.int32)
                cw = cw - tl.where(cw >= KW, KW, 0)
                ct = tbase + (ch >= KH).to(tl.int32)
                ch = ch - tl.where(ch >= KH, KH, 0)
            offsets = (ct * DT * H * WIDTH + ch * DH * WIDTH + cw * DW) * CI + c
        else:
            c = k % CI
            kw = k // CI % KW
            kh = k // (CI * KW) % KH
            kt = k // (CI * KW * KH)
            offsets = (kt * DT * H * WIDTH + kh * DH * WIDTH + kw * DW) * CI + c
        a = tl.load(
            X + base[:, None] + offsets[None, :],
            (m[:, None] < B * OT * OH * OW) & (k[None, :] < total),
            other=0.0,
        )
        w = tl.load(
            W + k[:, None] * CO + n[None, :], (k[:, None] < total) & (n[None, :] < CO), other=0.0
        )
        acc = tl.dot(a, w, acc)
        if MODE == 2:
            cb += BK
            wrap_c = cb >= CI
            cb -= tl.where(wrap_c, CI, 0)
            wbase += wrap_c.to(tl.int32)
            wrap_w = wbase >= KW
            wbase -= tl.where(wrap_w, KW, 0)
            hbase += wrap_w.to(tl.int32)
            wrap_h = hbase >= KH
            hbase -= tl.where(wrap_h, KH, 0)
            tbase += wrap_h.to(tl.int32)
    value = acc * tl.load(SCALE_X) * tl.load(SCALE_W + n, n < CO, other=0)[None, :]
    if HAS_BIAS:
        value += tl.load(BIAS + n, n < CO, other=0)[None, :]
    tl.store(
        Y + m[:, None] * CO + n[None, :], value, (m[:, None] < B * OT * OH * OW) & (n[None, :] < CO)
    )


class IteratorFP8Conv(ImplicitFP8Conv):
    def __init__(
        self,
        base,
        *,
        mode="direct",
        block=None,
        warps=None,
        stages=None,
        group_m=0,
        maxnreg=None,
    ):
        nn.Module.__init__(self)
        if not isinstance(base, ImplicitFP8Conv):
            raise TypeError("Iterator requires an existing calibrated FP8 convolution")
        if any(base.padding):
            raise ValueError(
                "Optimized iterator only accepts valid convolutions after causal packing"
            )
        self.__dict__.update(
            {
                k: v
                for k, v in base.__dict__.items()
                if k not in ("_modules", "_parameters", "_buffers")
            }
        )
        self._buffers = dict(base._buffers)
        self._modules, self._parameters = ({}, {})
        self.mode = mode
        self.block = tuple(block or base.block)
        self.num_warps = warps or base.num_warps
        self.num_stages = stages or base.num_stages
        self.group_m = group_m
        self.maxnreg = maxnreg
        if maxnreg is not None and (maxnreg <= 0 or maxnreg % 8):
            raise ValueError("maxnreg must be a positive multiple of eight")
        if mode not in ("direct", "incremental"):
            raise ValueError(mode)
        if mode == "incremental" and self.block[2] > self.fp8_weight.shape[3]:
            raise ValueError("Incremental iterator requires BK <= padded input channels")

    def forward(self, x, cache_x=None):
        packed, _ = fused_pack(self, x, cache_x, quant_scale=self.activation_scale)
        b, ci, t, h, w = packed.shape
        kt, kh, kw = self.kernel_size
        st, sh, sw = self.stride
        dt, dh, dw = self.dilation
        ot = (t - dt * (kt - 1) - 1) // st + 1
        oh = (h - dh * (kh - 1) - 1) // sh + 1
        ow = (w - dw * (kw - 1) - 1) // sw + 1
        output = torch.empty((b, ot, oh, ow, self.out_channels), device=x.device, dtype=x.dtype)
        bm, bn, bk = self.block
        kernel = _valid_fp8
        self.last_kernel = kernel[
            triton.cdiv(b * ot * oh * ow, bm), triton.cdiv(self.out_channels, bn)
        ](
            packed,
            self.fp8_weight,
            output,
            self.bias if self.bias is not None else output,
            self.activation_scale,
            self.weight_scale,
            b,
            t,
            h,
            w,
            ci,
            self.out_channels,
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
            self.bias is not None,
            bm,
            bn,
            bk,
            {"direct": 0, "incremental": 2}[self.mode],
            self.group_m,
            num_warps=self.num_warps,
            num_stages=self.num_stages,
            **{"maxnreg": self.maxnreg} if self.maxnreg is not None else {},
        )
        return output.permute(0, 4, 1, 2, 3)


def downblock_without_clone(self, x, feat_cache=None, feat_idx=None):
    feat_idx = [0] if feat_idx is None else feat_idx
    shortcut_input = x
    for resnet in self.resnets:
        x = resnet(x, feat_cache=feat_cache, feat_idx=feat_idx)
    if self.downsampler is not None:
        x = self.downsampler(x, feat_cache=feat_cache, feat_idx=feat_idx)
    return x + self.avg_shortcut(shortcut_input)


def remove_downblock_clones(vae):
    from diffusers.models.autoencoders.autoencoder_kl_wan import WanResidualDownBlock

    changed = []
    for name, child in vae.named_modules():
        if isinstance(child, WanResidualDownBlock):
            child.forward = MethodType(downblock_without_clone, child)
            changed.append(name)
    return changed


def install_iterators(vae, choices):
    changed = []
    for name, choice in choices.items():
        if choice.get("mode") == "original":
            continue
        child = vae.get_submodule(name)
        if name in ("encoder.conv_in", "encoder.conv_out", "quant_conv"):
            raise ValueError("Protected layer cannot use the FP8 iterator")
        vae.set_submodule(name, IteratorFP8Conv(child, **choice))
        changed.append(name)
    return changed


@triton.jit
def _avg_shortcut(
    X,
    Y,
    T: tl.constexpr,
    CO: tl.constexpr,
    OT: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    FT: tl.constexpr,
    FS: tl.constexpr,
    G: tl.constexpr,
    PAD: tl.constexpr,
    XS0: tl.constexpr,
    XS1: tl.constexpr,
    XS2: tl.constexpr,
    XS3: tl.constexpr,
    XS4: tl.constexpr,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    co = i % CO
    row = i // CO
    ow = row % OW
    oh = row // OW % OH
    ot = row // (OW * OH) % OT
    batch = row // (OW * OH * OT)
    acc = tl.full((BLOCK,), 0, tl.float32)
    for g in tl.static_range(G):
        grouped = co * G + g
        c = grouped // (FT * FS * FS)
        ft = grouped // (FS * FS) % FT
        fh = grouped // FS % FS
        fw = grouped % FS
        t = ot * FT + ft - PAD
        value = tl.load(
            X + batch * XS0 + c * XS1 + t * XS2 + (oh * FS + fh) * XS3 + (ow * FS + fw) * XS4,
            (i < N) & (t >= 0) & (t < T),
            other=0.0,
        ).to(tl.float32)
        acc = acc + value
    tl.store(Y + i, acc / G, i < N)


class FusedAvgShortcut(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.factor_t = base.factor_t
        self.factor_s = base.factor_s
        self.group_size = base.group_size
        self.out_channels = base.out_channels

    def forward(self, x):
        b, c, t, h, w = x.shape
        ft, fs = (self.factor_t, self.factor_s)
        if (ft, fs) == (1, 1) and c == self.out_channels:
            return x
        pad = -t % ft
        ot = (t + pad) // ft
        oh, ow = (h // fs, w // fs)
        output = torch.empty((b, ot, oh, ow, self.out_channels), device=x.device, dtype=x.dtype)
        _avg_shortcut[triton.cdiv(output.numel(), 256),](
            x,
            output,
            t,
            self.out_channels,
            ot,
            oh,
            ow,
            ft,
            fs,
            self.group_size,
            pad,
            *x.stride(),
            output.numel(),
            256,
        )
        return output.permute(0, 4, 1, 2, 3)


def fuse_avg_shortcuts(vae):
    from diffusers.models.autoencoders.autoencoder_kl_wan import WanResidualDownBlock

    changed = []
    for name, child in vae.named_modules():
        if isinstance(child, WanResidualDownBlock):
            child.avg_shortcut = FusedAvgShortcut(child.avg_shortcut)
            changed.append(name + ".avg_shortcut")
    return changed
