"""Validated data movement optimizations for SM80 BF16 inference."""

from types import MethodType

import torch

from .acceleration import EquivalentConv


def validated_device(device):
    return (
        torch.cuda.get_device_capability(device) == (8, 0)
        and str(torch.__version__) == "2.13.0a0+8145d630e8.nv26.06"
    )


def supported(device, canvas, compile_enabled):
    return (
        compile_enabled
        and validated_device(device)
        and tuple(canvas) in {(480, 480), (672, 480), (832, 480), (832, 608)}
    )


def _stable_convolution(self, x, cache_x=None):
    benchmark = torch.backends.cudnn.benchmark
    torch.backends.cudnn.benchmark = False
    try:
        return self._convolution_forward(x, cache_x)
    finally:
        torch.backends.cudnn.benchmark = benchmark


def _stable_network(self, batch):
    benchmark = torch.backends.cudnn.benchmark
    torch.backends.cudnn.benchmark = False
    try:
        return self._network_forward(batch)
    finally:
        torch.backends.cudnn.benchmark = benchmark


def stabilize_convolutions(net, vae):
    net._network_forward = net.forward
    net.forward = MethodType(_stable_network, net)
    names = []
    for name, child in vae.named_modules():
        if isinstance(child, EquivalentConv) and (
            child.ndim == 2 or tuple(child.kernel_size[-2:]) == (1, 1)
        ):
            child._convolution_forward = child.forward
            child.forward = MethodType(_stable_convolution, child)
            names.append(name)
    return names


def prepare(net, vae):
    from .fp8_modules import install_packing
    from .fp8_runtime import cache_views, decode_without_cpu_read
    from .kernel_vae_kernels import remove_downblock_clones

    evidence = dict(
        cache_views=cache_views(vae),
        downblock_views=remove_downblock_clones(vae),
        packed_convolutions=install_packing(vae),
    )
    net.decoder.compile(dynamic=False, fullgraph=False)
    evidence["compiled_readout"] = True
    net._deferred_camera_checks = []
    net._decode = MethodType(decode_without_cpu_read, net)
    return evidence
