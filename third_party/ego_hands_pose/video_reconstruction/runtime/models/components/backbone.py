"""Executed feature layers and frozen conditioning."""

import torch
from torch import nn

from runtime.vendor.transformer import FeatureTransformer, rope_params, sinusoidal_embedding_1d


class FeatureBackbone(nn.Module):
    def __init__(self, config, context):
        super().__init__()
        previous = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.bfloat16)
            with torch.device("meta"):
                self.wan = FeatureTransformer(**config)
                self.wan.patch_embedding.to(dtype=torch.float32)
        finally:
            torch.set_default_dtype(previous)
        d = config["dim"] // config["num_heads"]
        self.wan.freqs = torch.cat(
            (
                rope_params(1024, d - 4 * (d // 6)),
                rope_params(1024, 2 * (d // 6)),
                rope_params(1024, 2 * (d // 6)),
            ),
            dim=1,
        )
        self._complex_grid_cache = {}
        self.register_buffer("empty_context", context, persistent=False)
        self.register_buffer("conditioning_e", torch.empty(0), persistent=False)
        self.register_buffer("conditioning_context", torch.empty(0), persistent=False)

    @property
    def output_dim(self):
        return self.wan.dim

    @torch.no_grad()
    def prepare_conditioning(self):
        parameter = next(self.wan.time_embedding.parameters())
        timestep = torch.zeros(1, device=parameter.device, dtype=torch.long)
        with torch.autocast(parameter.device.type, enabled=False):
            embedding = torch.func.functional_call(
                self.wan.time_embedding,
                {name: value.float() for name, value in self.wan.time_embedding.named_parameters()},
                (sinusoidal_embedding_1d(self.wan.freq_dim, timestep).float(),),
            )
            self.conditioning_e = torch.func.functional_call(
                self.wan.time_projection,
                {
                    name: value.float()
                    for name, value in self.wan.time_projection.named_parameters()
                },
                (embedding,),
            ).unflatten(1, (6, self.wan.dim))
        context = self.empty_context.to(device=parameter.device, dtype=parameter.dtype)[
            :, : self.wan.text_len
        ]
        if context.shape[1] < self.wan.text_len:
            context = torch.nn.functional.pad(
                context, (0, 0, 0, self.wan.text_len - context.shape[1])
            )
        self.conditioning_context = self.wan.text_embedding(context)

    def forward(self, latent):
        batch = latent.shape[0]
        dtype = torch.bfloat16
        latent = latent.to(dtype)
        packed = torch.cat((latent, latent.new_zeros(batch, 100, *latent.shape[2:])), dim=1)
        patch = self.wan.patch_embedding
        if torch.is_autocast_enabled(packed.device.type) or patch.weight.dtype == packed.dtype:
            feature = patch(packed)
        else:
            feature = torch.nn.functional.conv3d(
                packed,
                patch.weight.to(dtype),
                patch.bias.to(dtype),
                patch.stride,
                patch.padding,
                patch.dilation,
                patch.groups,
            )
        _, _, frames, height, width = feature.shape
        x = feature.flatten(2).transpose(1, 2)
        grid = (frames, height, width)
        key = (*grid, x.device)
        compiling = torch.compiler.is_compiling()
        if hasattr(self, "_aggressive_rope_phase_cache"):
            phase = self._aggressive_rope_phase_cache[grid]
        elif compiling or key not in self._complex_grid_cache:
            if len(self._complex_grid_cache) >= 8:
                self._complex_grid_cache.clear()
            channels = self.wan.freqs.shape[-1]
            ft, fh, fw = self.wan.freqs.split(
                (channels - 2 * (channels // 3), channels // 3, channels // 3), dim=1
            )
            phase = torch.cat(
                (
                    ft[:frames].view(frames, 1, 1, -1).expand(frames, height, width, -1),
                    fh[:height].view(1, height, 1, -1).expand(frames, height, width, -1),
                    fw[:width].view(1, 1, width, -1).expand(frames, height, width, -1),
                ),
                dim=-1,
            ).reshape(frames * height * width, 1, -1)
            if not compiling:
                self._complex_grid_cache[key] = phase
        else:
            phase = self._complex_grid_cache[key]
        kwargs = dict(
            e=self.conditioning_e.expand(batch, -1, -1),
            freqs=phase,
            context=self.conditioning_context.expand(batch, -1, -1),
            dtype=dtype,
        )
        for block in self.wan.blocks:
            x = block(x, **kwargs)
        return x.reshape(batch, frames, height, width, self.wan.dim)
