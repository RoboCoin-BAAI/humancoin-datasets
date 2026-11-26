import hashlib
import inspect
from types import MethodType

from . import kernel_vae_normpack as original


class Canvas832Normpack:
    def __init__(self, vae):
        source = inspect.getsource(original.residual_normpack)
        selector = "x.shape[3:] == (240, 240)"
        if source.count(selector) != 1:
            raise ValueError("Original normpack selector changed")
        derived = source.replace(selector, "x.shape[3:] == (240, 416)")
        namespace = dict(original.__dict__)
        exec(compile(derived, "<kernel_832_normpack>", "exec"), namespace)
        self.forward = namespace["residual_normpack"]
        self.blocks = []
        for index in (0, 1):
            block = vae.get_submodule(f"encoder.down_blocks.0.resnets.{index}")
            if (
                getattr(block, "_normpack_choices", None)
                != {"conv1": {"rows": 4, "warps": 4}, "conv2": {"rows": 4, "warps": 4}}
                or block.training
                or block.dropout.p != 0
            ):
                raise ValueError("832 scope requires the validated inference normpack")
            self.blocks.append((block, block.forward, block._normpack_choices))
        self.choices = {name: {"rows": 4, "warps": 4} for name in ("conv1", "conv2")}
        self.evidence = dict(
            original_source_sha256=hashlib.sha256(source.encode()).hexdigest(),
            derived_source_sha256=hashlib.sha256(derived.encode()).hexdigest(),
            canvas=[832, 480],
            spatial=[240, 416],
            times=[1, 8],
            rows=4,
            warps=4,
            reduction_quantization_cache_and_convolution_unchanged=True,
        )

    def configure(self, enabled):
        for block, saved, choices in self.blocks:
            block.forward = MethodType(self.forward, block) if enabled else saved
            block._normpack_choices = self.choices if enabled else choices
