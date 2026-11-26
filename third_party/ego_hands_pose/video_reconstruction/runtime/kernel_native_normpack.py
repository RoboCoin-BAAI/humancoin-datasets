import ast
import contextvars
import hashlib
from pathlib import Path
from types import MethodType

import torch
import triton
import triton.language as tl

from .fp8_modules import ImplicitFP8Conv
from .kernel_vae_residual import ResidualFP8Conv

SPEC = contextvars.ContextVar("native_normpack_spec", default=None)


@triton.jit
def _past(
    HISTORY,
    PACKED,
    SCALE,
    C: tl.constexpr,
    HC: tl.constexpr,
    LT: tl.constexpr,
    T: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    PH: tl.constexpr,
    PW: tl.constexpr,
    PC: tl.constexpr,
    LH: tl.constexpr,
    LW: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    S2: tl.constexpr,
    S3: tl.constexpr,
    S4: tl.constexpr,
    NUM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    c = i % PC
    col = i // PC % PW - LW
    row = i // (PC * PW) % PH - LH
    time = i // (PC * PW * PH) % LT - LT
    batch = i // (PC * PW * PH * LT)
    valid = (i < NUM) & (c < C) & (col >= 0) & (col < W) & (row >= 0) & (row < H) & (time >= -HC)
    x = tl.load(
        HISTORY + batch * S0 + c * S1 + (time + HC) * S2 + row * S3 + col * S4, valid, other=0
    ).to(tl.float32)
    value = tl.maximum(tl.minimum(x / tl.load(SCALE), 448.0), -448.0)
    out = (((batch * (T + LT) + time + LT) * PH + row + LH) * PW + col + LW) * PC + c
    tl.store(PACKED + out, value, i < NUM)


class NativeNormpack:
    def __init__(self, vae):
        from diffusers.models.autoencoders.autoencoder_kl_wan import WanResidualBlock
        from torch._inductor.codecache import PyCodeCache

        from . import fp8_modules, kernel_vae_kernels, kernel_vae_residual, kernel_vae_weight_layout

        self.kernels = []
        self.blocks = []
        self.pack_functions = []
        self.derived = {}
        self.evidence = []
        self.executed = {}
        if vae.training:
            raise ValueError("Native normpack is restricted to eager inference")
        for module in PyCodeCache.modules:
            path = Path(getattr(module, "__file__", ""))
            if not path.is_file():
                continue
            for statement in ast.parse(path.read_text()).body:
                if not isinstance(statement, ast.Assign) or not isinstance(
                    statement.value, ast.Call
                ):
                    continue
                call = statement.value
                if ast.unparse(call.func) != "async_compile.triton" or len(call.args) < 2:
                    continue
                name, source = (call.args[0].value, call.args[1].value)
                if "linalg_vector_norm" not in name or "silu" not in name:
                    continue
                kernel = getattr(module, statement.targets[0].id)
                fn = next((n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)))
                names = [a.arg for a in fn.args.args]
                stores = [
                    n
                    for n in ast.walk(fn)
                    if isinstance(n, ast.Call) and ast.unparse(n.func) == "tl.store"
                ]
                if len(stores) != 1 or not isinstance(stores[0].args[0], ast.BinOp):
                    continue
                out = stores[0].args[0].left
                if not isinstance(out, ast.Name) or out.id not in names:
                    continue
                self.kernels.append((kernel, kernel.run, source, names.index(out.id), name))
        self.block_names = {}
        for name, child in vae.named_modules():
            if (
                isinstance(child, WanResidualBlock)
                and (not hasattr(child, "_normpack_choices"))
                and all(
                    (isinstance(getattr(child, c), ImplicitFP8Conv) for c in ["conv1", "conv2"])
                )
            ):
                if child.training or child.dropout.p != 0:
                    raise ValueError("Native normpack requires inference and zero dropout")
                if any(
                    (n._forward_hooks or n._forward_pre_hooks for n in [child.norm1, child.norm2])
                ):
                    raise ValueError(
                        "Native normpack requires unobserved intermediate norm tensors"
                    )
                self.blocks.append((child, child.forward))
                self.block_names[id(child)] = name
        for module in [
            fp8_modules,
            kernel_vae_kernels,
            kernel_vae_residual,
            kernel_vae_weight_layout,
        ]:
            if hasattr(module, "fused_pack"):
                self.pack_functions.append((module, module.fused_pack))
        if not self.kernels or not self.blocks:
            raise ValueError("No original compiled VAE norm kernels/blocks found")

    def _compile(self, source, name, spec):
        from torch._inductor.async_compile import AsyncCompile

        tree = ast.parse(source)
        fn = next((n for n in tree.body if isinstance(n, ast.FunctionDef)))
        old_name = fn.name
        constexpr = next((i for i, a in enumerate(fn.args.args) if a.annotation is not None))
        added = [ast.arg(arg="PACKED"), ast.arg(arg="SCALE")]
        added.append(ast.arg(arg="CACHE"))
        fn.args.args[constexpr:constexpr] = added
        for decorator in fn.decorator_list:
            if not isinstance(decorator, ast.Call):
                continue
            for kw in decorator.keywords:
                if kw.arg != "triton_meta":
                    continue
                if not isinstance(kw.value, ast.Dict):
                    raise ValueError("Unrecognized original Triton metadata")
                signature_index = next(
                    (i for i, k in enumerate(kw.value.keys) if ast.literal_eval(k) == "signature")
                )
                old_signature = ast.literal_eval(kw.value.values[signature_index])
                ordered = {}
                for arg in fn.args.args:
                    ordered[arg.arg] = (
                        "*fp8e4nv"
                        if arg.arg == "PACKED"
                        else "*fp32"
                        if arg.arg == "SCALE"
                        else "*fp16"
                        if arg.arg == "CACHE"
                        else old_signature[arg.arg]
                    )
                kw.value.values[signature_index] = ast.parse(repr(ordered), mode="eval").body
        stores = [
            n
            for n in ast.walk(fn)
            if isinstance(n, ast.Expr)
            and isinstance(n.value, ast.Call)
            and (ast.unparse(n.value.func) == "tl.store")
        ]
        if len(stores) != 1:
            raise ValueError("Expected one original normalized activation store")
        store = stores[0].value
        ptr = store.args[0]
        if (
            not isinstance(ptr, ast.BinOp)
            or not isinstance(ptr.left, ast.Name)
            or (not ptr.left.id.startswith("out_ptr"))
        ):
            raise ValueError("Unrecognized native norm output indexing")
        b, c, t, h, w = spec["shape"]
        lw, rw, lh, rh, lt, rt = spec["conv"].causal_padding
        if rt:
            raise ValueError("Future padding is outside this causal encoder")
        pc = triton.cdiv(c, spec["conv"].alignment) * spec["conv"].alignment
        ph, pw = (h + lh + rh, w + lw + rw)
        index, value, mask = (
            ast.unparse(ptr.right),
            ast.unparse(store.args[1]),
            ast.unparse(store.args[2]),
        )
        if spec["channel_last"]:
            coordinates = f"native_row = native_flat // {c}\nnative_c = native_flat % {c}\n"
        else:
            coordinates = f"native_row = native_flat // {c * t * h * w} * {t * h * w} + native_flat % {t * h * w}\nnative_c = native_flat // {t * h * w} % {c}\n"
        ct = min(t, 2)
        if spec["channel_last"]:
            cache_index = f"((native_row // {t * h * w} * {ct} + native_time - {t - ct}) * {h * w} + native_row % {h * w}) * {c} + native_c"
        else:
            cache_index = f"((native_row // {t * h * w} * {c} + native_c) * {ct} + native_time - {t - ct}) * {h * w} + native_row % {h * w}"
        cache_store = (
            f"tl.store(CACHE + ({cache_index}), native_y, ({mask}) & (native_time >= {t - ct}))"
        )
        code = f"native_flat = {index}\n{coordinates}\nnative_time = native_row // {h * w} % {t}\nnative_y = ({value}).to(tl.float16)\n{cache_store}\nnative_packed = (((native_row // {t * h * w} * {t + lt} + native_time + {lt}) * {ph} + native_row // {w} % {h} + {lh}) * {pw} + native_row % {w} + {lw}) * {pc} + native_c\nnative_q = tl.maximum(tl.minimum(native_y.to(tl.float32) / tl.load(SCALE), 448.0), -448.0)\ntl.store(PACKED + native_packed, native_q, {mask})\n"
        target = stores[0]

        class Insert(ast.NodeTransformer):
            def visit_Expr(self, node):
                return ast.parse(code).body if node is target else node

        Insert().visit(fn)
        ast.fix_missing_locations(tree)
        body = ast.unparse(tree).replace(old_name, name)
        compiler = AsyncCompile()
        namespace = {name: compiler.triton(name, body, device_str="cuda")}
        compiler.wait(namespace)
        self.evidence.append(
            dict(
                name=name,
                scope=spec["shape"],
                padding=spec["conv"].causal_padding,
                original_source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                derived_source_sha256=hashlib.sha256(body.encode()).hexdigest(),
            )
        )
        return namespace[name]

    def configure(self, enabled):
        owner = self
        for kernel, original, source, out_index, name in self.kernels:
            if not enabled:
                kernel.run = original
                continue

            def run(
                *args, original=original, source=source, out_index=out_index, name=name, **kwargs
            ):
                spec = SPEC.get()
                if spec is None:
                    return original(*args, **kwargs)
                out = args[out_index]
                if out.dtype != torch.float16 or out.numel() != spec["input"].numel():
                    raise ValueError("Native norm output is outside the FP16 domain")
                if out.ndim != 5 or not (out.stride(1) == 1 or out.is_contiguous()):
                    raise ValueError(f"Unexpected native norm layout: {out.shape}, {out.stride()}")
                spec["channel_last"] = out.stride(1) == 1
                key = (
                    hashlib.sha256(source.encode()).hexdigest(),
                    tuple(spec["shape"]),
                    tuple(spec["conv"].causal_padding),
                    spec["conv"].alignment,
                    spec["channel_last"],
                )
                b, c, t, h, w = spec["shape"]
                spec["cache"] = torch.empty(
                    (b, c, min(t, 2), h, w),
                    device=out.device,
                    dtype=torch.float16,
                    memory_format=torch.channels_last_3d
                    if spec["channel_last"]
                    else torch.contiguous_format,
                )
                if key not in owner.derived:
                    suffix = hashlib.sha256(repr(key).encode()).hexdigest()[:12]
                    owner.derived[key] = owner._compile(
                        source, name + "_native_pack_" + suffix, spec
                    )
                spec["out_ptr"] = out.data_ptr()
                owner.executed[name] = owner.executed.get(name, 0) + 1
                extra = [spec["packed"], spec["conv"].activation_scale]
                extra.append(spec["cache"])
                return owner.derived[key].run(*args, *extra, **kwargs)

            kernel.run = run
        for module, original in self.pack_functions:
            if not enabled:
                module.fused_pack = original
                continue

            def pack(conv, x, *args, original=original, **kwargs):
                spec = SPEC.get()
                if spec is not None and spec["conv"] is conv:
                    if x.data_ptr() != spec.get("out_ptr"):
                        raise ValueError("Consumer does not read the prepared native norm output")
                    return (spec["packed"], spec["packed"].shape[2])
                return original(conv, x, *args, **kwargs)

            module.fused_pack = pack
        for block, original in self.blocks:
            block._native_normpack_name = self.block_names[id(block)]
            block.forward = MethodType(self._block, block) if enabled else original

    @staticmethod
    def _block(self, x, feat_cache=None, feat_idx=None):
        feat_idx = [0] if feat_idx is None else feat_idx
        shortcut = self.conv_shortcut(x)
        for norm, conv in [(self.norm1, self.conv1), (self.norm2, self.conv2)]:
            b, c, t, h, w = x.shape
            lw, rw, lh, rh, lt, rt = conv.causal_padding
            pc = triton.cdiv(c, conv.alignment) * conv.alignment
            packed = torch.empty(
                (b, pc, t + lt + rt, h + lh + rh, w + lw + rw),
                device=x.device,
                dtype=torch.float8_e4m3fn,
                memory_format=torch.channels_last_3d,
            )
            packed.zero_()
            index = feat_idx[0]
            past = feat_cache[index] if feat_cache is not None else None
            if lt and past is not None:
                num = b * lt * (h + lh + rh) * (w + lw + rw) * pc
                _past[triton.cdiv(num, 512),](
                    past,
                    packed,
                    conv.activation_scale,
                    c,
                    past.shape[2],
                    lt,
                    t,
                    h,
                    w,
                    h + lh + rh,
                    w + lw + rw,
                    pc,
                    lh,
                    lw,
                    *past.stride(),
                    num,
                    512,
                )
            spec = dict(conv=conv, input=x, shape=list(x.shape), packed=packed)
            token = SPEC.set(spec)
            try:
                x = self.nonlinearity(norm(x))
                if "out_ptr" not in spec:
                    raise RuntimeError(
                        f"Original norm kernel was not captured: {self._native_normpack_name}, conv2={conv is self.conv2}, input={spec['shape']}, strides={spec['input'].stride()}; no silent fallback"
                    )
                last = conv is self.conv2
                if last:
                    x = self.dropout(x)
                history = spec["cache"]
                if history.shape[2] < 2 and past is not None:
                    history = torch.cat((past[:, :, -1:], history), 2)
                if last and isinstance(conv, ResidualFP8Conv):
                    x = conv(x, past, residual=shortcut)
                else:
                    x = conv(x, past)
                    if last:
                        x = x + shortcut
                if feat_cache is not None:
                    feat_cache[index] = history
                    feat_idx[0] += 1
            finally:
                SPEC.reset(token)
        return x
