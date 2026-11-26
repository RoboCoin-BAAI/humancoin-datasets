import torch
import triton
import triton.language as tl
from torch._inductor.runtime import triton_helpers
from triton.language.extra import libdevice


@triton.jit
def _gemm_amax(
    X,
    W,
    Y,
    SX,
    SW,
    BIAS,
    PARTIAL,
    M: tl.constexpr,
    REAL_M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    cm: tl.constexpr = triton.cdiv(M, BM)
    cn: tl.constexpr = triton.cdiv(N, BN)
    pid = tl.program_id(0)
    group = pid // (GROUP_M * cn)
    first = group * GROUP_M
    size = tl.minimum(cm - first, GROUP_M)
    im = first + pid % (GROUP_M * cn) % size
    inn = pid % (GROUP_M * cn) // size
    m = im * BM + tl.arange(0, BM)
    n = inn * BN + tl.arange(0, BN)
    kk = tl.arange(0, 128)
    acc = tl.full((BM, BN), 0, tl.float32)
    for tile in range(triton.cdiv(K, 128)):
        k = tile * 128 + kk
        a = tl.load(X + m[:, None] * K + k[None, :], (m[:, None] < M) & (k[None, :] < K), other=0.0)
        b = tl.load(W + n[None, :] * K + k[:, None], (n[None, :] < N) & (k[:, None] < K), other=0.0)
        partial = tl.dot(a, b, max_num_imprecise_acc=0)
        acc = tl.inline_asm_elementwise(
            "add.f32 $0, $1, $2;",
            constraints="=f,f,f",
            args=[acc, partial],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
    value = acc * (tl.load(SX) * tl.load(SW))
    rounded = value.to(tl.float16)
    tl.store(Y + m[:, None] * N + n[None, :], rounded, (m[:, None] < M) & (n[None, :] < N))
    x = rounded.to(tl.float32) + tl.load(BIAS + n[None, :], n[None, :] < N, other=0).to(tl.float32)
    half = x * 0.5
    square = x * x
    cube = square * x
    weighted = cube * 0.044715
    argument = (x + weighted) * 0.7978845608028654
    activated = half * (libdevice.tanh(argument) + 1.0)
    masked = tl.where((m[:, None] < REAL_M) & (n[None, :] < N), tl.abs(activated), float("-inf"))
    maximum = triton_helpers.max2(triton_helpers.max2(masked, 1), 0)
    tl.store(PARTIAL + pid, maximum)


def invoke(x, weight, sx, sw, bias, real_rows, *, block=(64, 128), warps=4, stages=3, out=None):
    m, k = x.shape
    n = weight.shape[0]
    bm, bn = block
    count = triton.cdiv(m, bm) * triton.cdiv(n, bn)
    output = torch.empty((m, n), dtype=torch.float16, device=x.device) if out is None else out
    if output.shape != (m, n) or output.dtype != torch.float16 or output.stride() != (n, 1):
        raise ValueError("Exact GEMM requires native contiguous FP16 output storage")
    partial = torch.empty(count, dtype=torch.float32, device=x.device)
    kernel = _gemm_amax[count,](
        x,
        weight,
        output,
        sx,
        sw,
        bias,
        partial,
        m,
        real_rows,
        n,
        k,
        bm,
        bn,
        8,
        num_warps=warps,
        num_stages=stages,
    )
    return (output, partial, kernel)


@triton.jit
def _compress_partial(P, OUT, COUNT: tl.constexpr, TARGET: tl.constexpr, R: tl.constexpr):
    i = tl.program_id(0)
    offsets = i * R + tl.arange(0, R)
    value = tl.load(P + offsets, offsets < COUNT, other=float("-inf"))
    maximum = triton_helpers.max2(value, 0)
    tl.store(OUT + i, maximum, i < TARGET)


def compress_partial(partial, original_output):
    count, target = (partial.numel(), original_output.numel())
    group = triton.next_power_of_2(triton.cdiv(count, target))
    _compress_partial[target,](partial, original_output, count, target, group, num_warps=4)
