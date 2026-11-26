from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

PROFILES = ("bf16", "fp8")


class FP8Linear(nn.Module):
    def __init__(self, linear):
        super().__init__()
        weight = linear.weight.detach().float()
        self.register_buffer("weight_scale", weight.abs().amax().clamp_min(1e-12) / 448)
        self.register_buffer(
            "weight", (weight / self.weight_scale).clamp(-448, 448).to(torch.float8_e4m3fn)
        )
        self.register_buffer(
            "bias", None if linear.bias is None else linear.bias.detach().to(torch.bfloat16)
        )
        self.in_features, self.out_features = (linear.in_features, linear.out_features)

    def forward(self, x):
        shape = x.shape
        flat = x.reshape(-1, shape[-1]).contiguous()
        scale = flat.float().abs().amax().clamp_min(1e-12) / 448
        quantized = (flat.float() / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
        rows = quantized.shape[0]
        quantized = F.pad(quantized, (0, 0, 0, -rows % 16))
        with torch.autocast(x.device.type, enabled=False):
            output = torch._scaled_mm(
                quantized,
                self.weight.t(),
                scale_a=scale,
                scale_b=self.weight_scale,
                out_dtype=torch.float16,
            )
        output = output[:rows].to(torch.bfloat16)
        if self.bias is not None:
            output = output + self.bias
        return output.reshape(*shape[:-1], self.out_features)


class EquivalentConv(nn.Module):
    def __init__(
        self,
        conv,
        implementation="native",
        alignment=1,
        layout="contiguous",
        compute_dtype="bfloat16",
    ):
        super().__init__()
        self.register_buffer("weight", conv.weight.detach())
        self.register_buffer("bias", None if conv.bias is None else conv.bias.detach())
        self.ndim = conv.weight.ndim - 2
        self.kernel_size, self.stride = (conv.kernel_size, conv.stride)
        self.padding, self.dilation, self.groups = (conv.padding, conv.dilation, conv.groups)
        self.in_channels, self.out_channels = (conv.in_channels, conv.out_channels)
        self.causal_padding = getattr(conv, "_padding", None)
        self.implementation, self.alignment, self.layout = (implementation, alignment, layout)
        self.compute_dtype = compute_dtype
        self._prepared = {}

    def prepare_input(self, x, cache_x=None):
        if self.causal_padding is not None:
            pad = list(self.causal_padding)
            if cache_x is not None and pad[4] > 0:
                x = torch.cat((cache_x.to(x.device), x), 2)
                pad[4] -= cache_x.shape[2]
            x = F.pad(x, pad)
        return x

    def prepared_weight(self, dtype):
        key = (self.implementation, self.alignment, self.layout, dtype)
        if key not in self._prepared:
            w = self.weight.to(dtype)
            if self.implementation == "conv2d" and self.ndim == 3:
                w = w.flatten(1, 2)
            if self.groups != 1 and self.alignment != 1:
                raise ValueError("Grouped padding requires group-specific lowering")
            ci, co = (w.shape[1], w.shape[0])
            w = F.pad(w, (0, 0) * (w.ndim - 2) + (0, -ci % self.alignment, 0, -co % self.alignment))
            if self.layout == "channels_last":
                w = w.contiguous(
                    memory_format=torch.channels_last if w.ndim == 4 else torch.channels_last_3d
                )
            self._prepared[key] = w
        return self._prepared[key]

    def spatial(self, x):
        temporal_padding = self.padding[0]
        if temporal_padding:
            x = F.pad(x, (0, 0, 0, 0, temporal_padding, temporal_padding))
        b, c, t, h, w = x.shape
        kt, kh, kw = self.kernel_size
        dt, dh, dw = self.dilation
        st, sh, sw = self.stride
        ot = (t - dt * (kt - 1) - 1) // st + 1
        taps = [x[:, :, tap * dt : tap * dt + ot * st : st] for tap in range(kt)]
        stack = torch.stack(taps, 2).permute(0, 3, 1, 2, 4, 5).reshape(b * ot, c * kt, h, w)
        output = self.apply_conv(stack, (sh, sw), self.padding[1:], (dh, dw))
        return output.reshape(b, ot, self.out_channels, output.shape[-2], output.shape[-1]).permute(
            0, 2, 1, 3, 4
        )

    def apply_conv(self, x, stride, padding, dilation):
        dtype = getattr(torch, self.compute_dtype) if x.device.type == "cuda" else x.dtype
        original_dtype = x.dtype
        x = x.to(dtype)
        weight = self.prepared_weight(dtype)
        ci = x.shape[1]
        if -ci % self.alignment:
            x = F.pad(x, (0, 0) * (x.ndim - 2) + (0, -ci % self.alignment))
        if self.layout == "channels_last":
            x = x.contiguous(
                memory_format=torch.channels_last if x.ndim == 4 else torch.channels_last_3d
            )
        bias = self.bias
        if bias is not None:
            bias = F.pad(bias.to(dtype), (0, weight.shape[0] - self.out_channels))
        conv = F.conv2d if x.ndim == 4 else F.conv3d
        with torch.autocast(x.device.type, enabled=False):
            y = conv(x, weight, bias, stride, padding, dilation, self.groups)[
                :, : self.out_channels
            ]
        return y.to(original_dtype)

    def forward(self, x, cache_x=None):
        x = self.prepare_input(x, cache_x)
        if self.implementation == "conv2d" and self.ndim == 3:
            return self.spatial(x)
        return self.apply_conv(x, self.stride, self.padding, self.dilation)


def enable_compilation(net, vae):
    setattr(
        torch._dynamo.config,
        "recompile_limit"
        if hasattr(torch._dynamo.config, "recompile_limit")
        else "cache_size_limit",
        128,
    )
    setattr(
        torch._dynamo.config,
        "accumulated_recompile_limit"
        if hasattr(torch._dynamo.config, "accumulated_recompile_limit")
        else "accumulated_cache_size_limit",
        512,
    )
    net.backbone.compile(dynamic=False, fullgraph=False)
    from diffusers.models.autoencoders.autoencoder_kl_wan import WanRMS_norm

    compiled = []
    for name, child in vae.named_modules():
        if isinstance(child, (WanRMS_norm, nn.SiLU)):
            child.compile(dynamic=False, fullgraph=False)
            compiled.append(name)
    return compiled


def prepare_profile(net, vae, profile, *, choices=None, compile_backbone=True):
    if profile not in PROFILES:
        raise ValueError("Unsupported precision")
    torch.backends.cudnn.benchmark = torch.cuda.get_device_capability() != (9, 0)
    evidence = dict(profile=profile, fp8_layers=[])
    if profile == "fp8":
        for name, child in list(net.backbone.named_modules()):
            if (
                isinstance(child, nn.Linear)
                and name.startswith("wan.blocks.")
                and (
                    ".ffn." in name
                    or any((name.endswith("." + key) for key in ("q", "k", "v", "o")))
                )
            ):
                net.backbone.set_submodule(name, FP8Linear(child))
                evidence["fp8_layers"].append(name)
        if not evidence["fp8_layers"]:
            raise RuntimeError("No FP8 layers converted")
    for name, child in list(vae.named_modules()):
        if isinstance(child, (nn.Conv2d, nn.Conv3d)):
            choice = (choices or {}).get(name, {})
            vae.set_submodule(
                name,
                EquivalentConv(
                    child,
                    **{
                        key: choice[key]
                        for key in ("implementation", "alignment", "layout", "compute_dtype")
                        if key in choice
                    },
                ),
            )
    if compile_backbone:
        enable_compilation(net, vae)
    return evidence
