import ctypes
import logging
from functools import lru_cache
from math import prod


class _Driver:
    def __init__(self):
        self.lib = ctypes.CDLL("libcuda.so.1")
        self.log = logging.getLogger(__name__)
        signatures = {
            "cuCtxGetDevice": [ctypes.POINTER(ctypes.c_int)],
            "cuDevicePrimaryCtxRetain": [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int],
            "cuDevicePrimaryCtxRelease_v2": [ctypes.c_int],
            "cuCtxPushCurrent_v2": [ctypes.c_void_p],
            "cuCtxPopCurrent_v2": [ctypes.POINTER(ctypes.c_void_p)],
            "cuMemAlloc_v2": [ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t],
            "cuMemFree_v2": [ctypes.c_uint64],
            "cuGetErrorName": [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)],
        }
        for name, args in signatures.items():
            function = getattr(self.lib, name)
            function.argtypes, function.restype = (args, ctypes.c_int)

    def check(self, status, operation):
        if status:
            name = ctypes.c_char_p()
            self.lib.cuGetErrorName(status, ctypes.byref(name))
            detail = name.value.decode() if name.value else str(status)
            raise RuntimeError(f"NVENC buffer {operation} failed: {detail}")


@lru_cache(maxsize=1)
def _driver():
    return _Driver()


class _Allocation:
    def __init__(self, shape, driver):
        self.driver = driver
        self.pointer = ctypes.c_uint64()
        self.context = ctypes.c_void_p()
        self.device = ctypes.c_int()
        driver.check(driver.lib.cuCtxGetDevice(ctypes.byref(self.device)), "get device")
        driver.check(
            driver.lib.cuDevicePrimaryCtxRetain(ctypes.byref(self.context), self.device),
            "retain context",
        )
        driver.check(driver.lib.cuCtxPushCurrent_v2(self.context), "push context")
        try:
            driver.check(
                driver.lib.cuMemAlloc_v2(ctypes.byref(self.pointer), prod(shape)), "allocate"
            )
        finally:
            previous = ctypes.c_void_p()
            driver.check(driver.lib.cuCtxPopCurrent_v2(ctypes.byref(previous)), "pop context")
        self.__cuda_array_interface__ = {
            "shape": shape,
            "typestr": "|u1",
            "data": (self.pointer.value, False),
            "version": 3,
        }

    def __del__(self):
        driver = self.driver
        try:
            if self.pointer.value:
                driver.check(driver.lib.cuCtxPushCurrent_v2(self.context), "push for free")
                try:
                    driver.check(driver.lib.cuMemFree_v2(self.pointer), "free")
                    self.pointer.value = 0
                finally:
                    previous = ctypes.c_void_p()
                    driver.check(
                        driver.lib.cuCtxPopCurrent_v2(ctypes.byref(previous)), "pop after free"
                    )
        except Exception as error:
            driver.log.warning("NVENC buffer cleanup failed: %s", error)
        finally:
            if self.context.value:
                status = driver.lib.cuDevicePrimaryCtxRelease_v2(self.device)
                if status:
                    driver.log.warning("NVENC context release failed: CUDA status %s", status)
                self.context.value = None


def nvenc_buffer(batch, height, width, device):
    import torch

    if min(batch, height, width) <= 0 or height % 2 or width % 2:
        raise ValueError("NVENC buffers require a positive batch and even dimensions")
    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError("NVENC buffer requires a CUDA device")
    with torch.cuda.device(device):
        owner = _Allocation((batch, height * 3 // 2, width), _driver())
        return torch.as_tensor(owner, dtype=torch.uint8, device=device)
