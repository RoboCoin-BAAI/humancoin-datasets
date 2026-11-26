import hashlib
import inspect
from types import MethodType

from . import kernel_vae_normpack as original


class WideNormpack:
    def __init__(self, vae):
        source = inspect.getsource(original.residual_normpack)
        selector = "x.shape[3:] == (240, 240)"
        if source.count(selector) != 1:
            raise ValueError("The validated early normpack selector changed")
        derived = source.replace(selector, "x.shape[3:] == (240, 336)")
        namespace = dict(original.__dict__)
        exec(compile(derived, "<kernel_wide_normpack>", "exec"), namespace)
        self.forward = namespace["residual_normpack"]
        self.blocks = []
        for r in (0, 1):
            block = vae.get_submodule(f"encoder.down_blocks.0.resnets.{r}")
            if not hasattr(block, "_normpack_choices") or block._normpack_choices != {
                "conv1": {"rows": 4, "warps": 4},
                "conv2": {"rows": 4, "warps": 4},
            }:
                raise ValueError("Wide scope requires the frozen four-layer ROW4/WARP4 recipe")
            if block.training or block.dropout.p != 0:
                raise ValueError("Wide normpack requires inference and zero dropout")
            self.blocks.append((block, block.forward))
        self.evidence = dict(
            original_source_sha256=hashlib.sha256(source.encode()).hexdigest(),
            derived_source_sha256=hashlib.sha256(derived.encode()).hexdigest(),
            original_selector=selector,
            derived_selector="x.shape[3:] == (240, 336)",
            reduction_quantization_cache_and_convolution_unchanged=True,
            canvas=[672, 480],
            batch=1,
            channels=160,
            times=[1, 8],
            rows=4,
            warps=4,
            accepted=False,
        )

    def configure(self, enabled):
        for block, saved in self.blocks:
            block.forward = MethodType(self.forward, block) if enabled else saved
