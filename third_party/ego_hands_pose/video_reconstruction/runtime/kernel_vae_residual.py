from __future__ import annotations

from types import MethodType

import torch
import triton
import triton.language as tl

from .fp8_kernels import fused_pack
from .kernel_vae_kernels import IteratorFP8Conv


@triton.jit
def _residual_valid_fp8(
    X,
    W,
    Y,
    BIAS,
    SCALE_X,
    SCALE_W,
    RESIDUAL,
    RS0: tl.constexpr,
    RS1: tl.constexpr,
    RS2: tl.constexpr,
    RS3: tl.constexpr,
    RS4: tl.constexpr,
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
    residual_ptrs = (
        RESIDUAL
        + batch[:, None] * RS0
        + n[None, :] * RS1
        + ot[:, None] * RS2
        + oh[:, None] * RS3
        + ow[:, None] * RS4
    )
    residual = tl.load(residual_ptrs, (m[:, None] < B * OT * OH * OW) & (n[None, :] < CO), other=0)
    value = value.to(Y.dtype.element_ty).to(tl.float32) + residual.to(tl.float32)
    tl.store(
        Y + m[:, None] * CO + n[None, :], value, (m[:, None] < B * OT * OH * OW) & (n[None, :] < CO)
    )


class ResidualFP8Conv(IteratorFP8Conv):
    def forward(self, x, cache_x=None, *, residual=None):
        if residual is None:
            return super().forward(x, cache_x)
        packed, _ = fused_pack(self, x, cache_x, quant_scale=self.activation_scale)
        b, ci, t, h, w = packed.shape
        kt, kh, kw = self.kernel_size
        st, sh, sw = self.stride
        dt, dh, dw = self.dilation
        ot = (t - dt * (kt - 1) - 1) // st + 1
        oh = (h - dh * (kh - 1) - 1) // sh + 1
        ow = (w - dw * (kw - 1) - 1) // sw + 1
        if residual.shape != (b, self.out_channels, ot, oh, ow) or residual.dtype != x.dtype:
            raise ValueError("Residual shape/dtype does not match convolution output")
        output = torch.empty((b, ot, oh, ow, self.out_channels), device=x.device, dtype=x.dtype)
        bm, bn, bk = self.block
        kernel = _residual_valid_fp8
        self.last_kernel = kernel[
            triton.cdiv(b * ot * oh * ow, bm), triton.cdiv(self.out_channels, bn)
        ](
            packed,
            self.fp8_weight,
            output,
            self.bias if self.bias is not None else output,
            self.activation_scale,
            self.weight_scale,
            residual,
            *residual.stride(),
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


def residual_with_fused_epilogue(self, x, feat_cache=None, feat_idx=None):
    feat_idx = [0] if feat_idx is None else feat_idx
    h = self.conv_shortcut(x)
    for norm, conv in ((self.norm1, self.conv1), (self.norm2, self.conv2)):
        x = self.nonlinearity(norm(x))
        last = conv is self.conv2
        if last:
            x = self.dropout(x)
        if feat_cache is None:
            x = conv(x, residual=h) if last else conv(x)
        else:
            idx = feat_idx[0]
            history = x[:, :, -2:]
            if history.shape[2] < 2 and feat_cache[idx] is not None:
                history = torch.cat((feat_cache[idx][:, :, -1:], history), 2)
            x = conv(x, feat_cache[idx], residual=h) if last else conv(x, feat_cache[idx])
            feat_cache[idx] = history
            feat_idx[0] += 1
    return x


def fuse_residual_epilogues(vae):
    from diffusers.models.autoencoders.autoencoder_kl_wan import WanResidualBlock

    names = []
    for name, child in vae.named_modules():
        if isinstance(child, WanResidualBlock) and isinstance(child.conv2, IteratorFP8Conv):
            base = child.conv2
            child.conv2 = ResidualFP8Conv(
                base,
                mode=base.mode,
                block=base.block,
                warps=base.num_warps,
                stages=base.num_stages,
                group_m=base.group_m,
                maxnreg=base.maxnreg,
            )
            child.forward = MethodType(residual_with_fused_epilogue, child)
            names.append(name)
    return names
