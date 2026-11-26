from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .acceleration import EquivalentConv
from .fp8_kernels import _implicit_fp8, fused_pack


class PackedConv(EquivalentConv):
    def __init__(self, base, implementation=None, layout=None):
        nn.Module.__init__(self)
        self.__dict__.update(
            {
                k: v
                for k, v in base.__dict__.items()
                if k not in ("_modules", "_parameters", "_buffers")
            }
        )
        self._modules = {}
        self._parameters = {}
        self._buffers = dict(base._buffers)
        self.implementation = implementation or base.implementation
        self.layout = layout or base.layout
        self._prepared = {}
        self._compiled_call_impl = None
        dtype = getattr(torch, self.compute_dtype)
        weight = self.prepared_weight(dtype)
        self.packed_bias = None if self.bias is None else self.bias.to(dtype)
        if self.packed_bias is not None and weight.shape[0] != self.out_channels:
            self.packed_bias = F.pad(self.packed_bias, (0, weight.shape[0] - self.out_channels))

    def forward(self, x, cache_x=None):
        if self.ndim != 3:
            return super().forward(x, cache_x)
        no_pad = self.causal_padding is None or not any(self.causal_padding)
        format = (
            torch.channels_last_3d if self.layout == "channels_last" else torch.contiguous_format
        )
        if (
            self.implementation == "native"
            and no_pad
            and (cache_x is None)
            and (x.shape[1] % self.alignment == 0)
            and (x.dtype == getattr(torch, self.compute_dtype))
            and x.is_contiguous(memory_format=format)
        ):
            packed, ot = (x, x.shape[2])
        else:
            packed, ot = fused_pack(self, x, cache_x)
        weight = self.prepared_weight(packed.dtype)
        bias = self.packed_bias
        with torch.autocast(x.device.type, enabled=False):
            if self.implementation == "conv2d":
                y = F.conv2d(
                    packed,
                    weight,
                    bias,
                    self.stride[1:],
                    self.padding[1:],
                    self.dilation[1:],
                    self.groups,
                )
                y = y[:, : self.out_channels].to(x.dtype)
                return y.reshape(x.shape[0], ot, self.out_channels, *y.shape[-2:]).permute(
                    0, 2, 1, 3, 4
                )
            y = F.conv3d(
                packed, weight, bias, self.stride, self.padding, self.dilation, self.groups
            )
        return y[:, : self.out_channels].to(x.dtype)


def install_packing(vae, choices=None):
    changed = []
    for name, child in list(vae.named_modules()):
        if isinstance(child, EquivalentConv) and child.ndim == 3:
            choice = (choices or {}).get(name, {})
            if choice.get("implementation") == "original":
                continue
            vae.set_submodule(
                name, PackedConv(child, choice.get("implementation"), choice.get("layout"))
            )
            changed.append(name)
    return changed


class ImplicitFP8Conv(EquivalentConv):
    def __init__(
        self, base, block=(64, 64, 64), *, activation_scale, alignment=16, num_warps=4, num_stages=3
    ):
        if base.ndim != 3 or base.groups != 1:
            raise ValueError("Implicit FP8 requires 3D convolution with groups=1")
        nn.Module.__init__(self)
        for key in (
            "ndim",
            "kernel_size",
            "stride",
            "padding",
            "dilation",
            "groups",
            "in_channels",
            "out_channels",
            "causal_padding",
        ):
            setattr(self, key, getattr(base, key))
        self.implementation = "native"
        self.layout = "channels_last"
        if alignment < 16 or alignment % 16:
            raise ValueError("FP8 channel alignment must be a positive multiple of 16")
        self.alignment = alignment
        self.compute_dtype = "float8_e4m3fn"
        self.block = block
        self.num_warps, self.num_stages = (num_warps, num_stages)
        if not math.isfinite(float(activation_scale)) or not float(activation_scale) > 0:
            raise ValueError("A positive offline activation scale is required")
        self.register_buffer(
            "activation_scale", torch.tensor(float(activation_scale), device=base.weight.device)
        )
        weight = base.weight.float()
        weight = F.pad(weight, (0, 0, 0, 0, 0, 0, 0, -self.in_channels % self.alignment))
        scales = weight.abs().flatten(1).amax(1).clamp_min(1e-12) / 448
        quantized = (weight / scales[:, None, None, None, None]).clamp(-448, 448)
        self.register_buffer(
            "fp8_weight", quantized.to(torch.float8_e4m3fn).permute(2, 3, 4, 1, 0).contiguous()
        )
        self.register_buffer("weight_scale", scales)
        self.register_buffer("bias", None if base.bias is None else base.bias.detach().float())
        self.maxnreg = None

    def forward(self, x, cache_x=None):
        import triton

        scale = self.activation_scale
        packed, _ = fused_pack(self, x, cache_x, quant_scale=scale)
        b, ci, t, h, w = packed.shape
        kt, kh, kw = self.kernel_size
        st, sh, sw = self.stride
        pt, ph, pw = self.padding
        dt, dh, dw = self.dilation
        ot = (t + 2 * pt - dt * (kt - 1) - 1) // st + 1
        oh = (h + 2 * ph - dh * (kh - 1) - 1) // sh + 1
        ow = (w + 2 * pw - dw * (kw - 1) - 1) // sw + 1
        output = torch.empty((b, ot, oh, ow, self.out_channels), device=x.device, dtype=x.dtype)
        bm, bn, bk = self.block
        self.last_kernel = _implicit_fp8[
            triton.cdiv(b * ot * oh * ow, bm), triton.cdiv(self.out_channels, bn)
        ](
            packed,
            self.fp8_weight,
            output,
            self.bias if self.bias is not None else output,
            scale,
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
            pt,
            ph,
            pw,
            dt,
            dh,
            dw,
            self.bias is not None,
            bm,
            bn,
            bk,
            num_warps=self.num_warps,
            num_stages=self.num_stages,
            **{"maxnreg": self.maxnreg} if self.maxnreg is not None else {},
        )
        return output.permute(0, 4, 1, 2, 3)


def install_fp8_vae(vae, choices, scales):
    changed = []
    for name, choice in choices.items():
        if choice.get("implementation") != "implicit_fp8":
            continue
        child = vae.get_submodule(name)
        if not isinstance(child, EquivalentConv):
            raise TypeError(f"FP8 VAE expects a validated floating conv: {name}")
        layer = scales[name]
        vae.set_submodule(
            name,
            ImplicitFP8Conv(
                child,
                tuple(choice["block"]),
                activation_scale=layer["scale"],
                alignment=choice.get("alignment", 16),
                num_warps=choice.get("num_warps", 4),
                num_stages=choice.get("num_stages", 3),
            ),
        )
        changed.append(name)
    if not changed:
        raise ValueError("FP8 VAE choices contain no executed FP8 convolutions")
    return changed


def populate_rope_phases(net, grids, temporal_lengths=range(1, 23)):
    backbone = net.backbone
    freqs = backbone.wan.freqs.to(next(backbone.parameters()).device)
    channels = freqs.shape[-1]
    temporal, vertical, horizontal = freqs.split(
        [channels - 2 * (channels // 3), channels // 3, channels // 3], 1
    )
    phases = getattr(backbone, "_aggressive_rope_phase_cache", {})
    for h, w in grids:
        for frames in temporal_lengths:
            if (frames, h, w) in phases:
                continue
            phases[frames, h, w] = torch.cat(
                (
                    temporal[:frames].view(frames, 1, 1, -1).expand(frames, h, w, -1),
                    vertical[:h].view(1, h, 1, -1).expand(frames, h, w, -1),
                    horizontal[:w].view(1, 1, w, -1).expand(frames, h, w, -1),
                ),
                dim=-1,
            ).reshape(frames * h * w, 1, -1)
    backbone._aggressive_rope_phase_cache = phases


def install_rope(net, *, grids=((15, 15),), temporal_lengths=(21, 22)):
    from runtime.vendor import transformer as wan

    from .fp8_kernels import fused_rope

    populate_rope_phases(net, grids, temporal_lengths)

    def apply(q, k, freqs):
        return fused_rope(q, k, freqs, fp64=False)

    wan.rope_apply_qk = apply
    return dict(
        operator="fused_qk_rope",
        rotation_precision="float32",
        attention_output_dtype="bfloat16",
        position_encoding_and_grid_unchanged=True,
        norm_fused=False,
    )
