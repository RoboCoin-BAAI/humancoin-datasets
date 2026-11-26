import os
from dataclasses import fields, is_dataclass, replace
from functools import wraps

import torch


def configure_geometry_precision():
    os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "0"
    torch.set_float32_matmul_precision("highest")


def fp32_geometry(function):
    configure_geometry_precision()

    @wraps(function)
    def wrapped(*args, **kwargs):
        devices = []

        def convert(value):
            if isinstance(value, torch.Tensor):
                devices.append(value.device.type)
                return value.float() if value.dtype in (torch.float16, torch.bfloat16) else value
            if is_dataclass(value) and (not isinstance(value, type)):
                return replace(
                    value, **{f.name: convert(getattr(value, f.name)) for f in fields(value)}
                )
            if isinstance(value, dict):
                return {k: convert(v) for k, v in value.items()}
            return value

        args = tuple((convert(value) for value in args))
        kwargs = {key: convert(value) for key, value in kwargs.items()}
        with torch.autocast(device_type=devices[0] if devices else "cpu", enabled=False):
            return function(*args, **kwargs)

    return wrapped
