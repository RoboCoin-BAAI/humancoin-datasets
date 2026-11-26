from __future__ import annotations

import ast
import copy
import hashlib
from pathlib import Path


def _name(node):
    if not isinstance(node, ast.Name):
        raise ValueError(f"Unsupported generated pointer: {ast.dump(node)}")
    return node.id


def _call_name(node):
    return ast.unparse(node.func) if isinstance(node, ast.Call) else ""


def _single_store_source(source, old_name, new_name):
    tree = ast.parse(source)
    kernel = next((x for x in tree.body if isinstance(x, ast.FunctionDef)))
    stores = [
        x for x in kernel.body if isinstance(x, ast.Expr) and _call_name(x.value) == "tl.store"
    ]
    if len(stores) != 3:
        raise ValueError("Expected three pointwise stores")
    if not all((f"out_ptr{i}" in ast.unparse(x.value.args[0]) for i, x in enumerate(stores))):
        raise ValueError("Unrecognized pack store layout")
    kernel.body = [x for x in kernel.body if x not in stores[1:]]
    kernel.name = new_name
    lines = source.splitlines(keepends=True)
    for store in reversed(stores[1:]):
        del lines[store.lineno - 1 : store.end_lineno]
    return "".join(lines).replace(old_name, new_name)


class CompiledQKV:
    def __init__(self, *, gemm_amax=False):
        self.patches = []
        self.evidence = []
        self.active = False
        self.gemm_amax = gemm_amax

    def _gemm(
        self, x, weight_transposed, sx, sw, bias, real_rows, *, out_dtype, use_fast_accum, out
    ):
        import torch

        from .kernel_gemm_amax import invoke

        if out_dtype != torch.float16 or use_fast_accum:
            raise ValueError("Exact FFN-up epilogue requires native FP16 and slow accumulation")
        _, partial, _ = invoke(
            x,
            weight_transposed.t(),
            sx,
            sw,
            bias,
            real_rows,
            block=(64, 128),
            warps=8,
            stages=3,
            out=out,
        )
        return partial

    def discover(self):
        from torch._inductor.async_compile import AsyncCompile
        from torch._inductor.codecache import PyCodeCache

        known = {id(fn) for fn, _, _ in self.patches}
        for module in list(PyCodeCache.modules):
            call = getattr(module, "call", None)
            fn = getattr(call, "__func__", call)
            if not callable(fn) or not hasattr(fn, "__code__") or id(fn) in known:
                continue
            path = Path(getattr(module, "__file__", ""))
            if not path.is_file():
                continue
            source = path.read_text()
            if "extern_kernels._scaled_mm(" not in source:
                continue
            tree = ast.parse(source)
            declarations = {}
            for statement in tree.body:
                if not isinstance(statement, ast.Assign) or not isinstance(
                    statement.value, ast.Call
                ):
                    continue
                value = statement.value
                if _call_name(value) == "async_compile.triton" and len(value.args) >= 2:
                    declarations[_name(statement.targets[0])] = value.args[1].value
            kernels = {
                name: body
                for name, body in declarations.items()
                if "out_ptr2" in body
                and "tmp29.to(tl.float8" not in body
                and ("tl.float8e4nv" in body)
                and ("def " in body)
                and ("in_ptr3" in body)
                and ("R0_BLOCK" not in body)
            }
            kernels = {
                name: body
                for name, body in kernels.items()
                if body.count("tl.store(") == 3 and "tmp1 / tmp7" in body
            }
            if not kernels and (not self.gemm_amax):
                continue
            function = next(
                (x for x in ast.walk(tree) if isinstance(x, ast.FunctionDef) and x.name == "call")
            )
            replacement = copy.deepcopy(function)
            assignments = {}
            for node in ast.walk(replacement):
                if (
                    isinstance(node, ast.Assign)
                    and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name)
                ):
                    assignments[node.targets[0].id] = node
            new_globals, groups, insert_before, insert_after, skip_lines = ({}, [], {}, {}, set())
            for node in ast.walk(replacement):
                if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                    continue
                if node.func.attr != "run" or not isinstance(node.func.value, ast.Name):
                    continue
                name = node.func.value.id
                if name not in kernels:
                    continue
                if len(node.args) != 8:
                    raise ValueError("Unrecognized triple pack call")
                real = [_name(x) for x in node.args[4:7]]
                roots = []
                for ptr in real:
                    value = assignments[ptr].value
                    roots.append(
                        _name(value.args[0]) if _call_name(value) == "reinterpret_tensor" else ptr
                    )

                def layout(ptr):
                    value = assignments[ptr].value
                    if isinstance(value, ast.Name):
                        return layout(value.id)
                    name = _call_name(value)
                    if name == "empty_strided_cuda":
                        return tuple((ast.unparse(x) for x in value.args[:3]))
                    if name == "reinterpret_tensor":
                        shape, stride = [ast.unparse(x) for x in value.args[1:3]]
                        return (shape, stride, layout(_name(value.args[0]))[2])
                    raise ValueError(f"Unsupported FP8 allocation lineage: {ast.unparse(value)}")

                specs = [layout(x) for x in real]
                if len(set(specs)) != 1:
                    raise ValueError("Q/K/V real layouts differ")
                if specs[0][2] != "torch.float8_e4m3fn":
                    raise ValueError("Expected native E4M3 allocations")
                shared = f"exact_qkv_shared_{node.lineno}"
                consumers = [
                    x
                    for x in ast.walk(replacement)
                    if isinstance(x, ast.Call)
                    and _call_name(x) == "extern_kernels._scaled_mm"
                    and isinstance(x.args[0], ast.Name)
                    and (x.args[0].id in roots)
                ]
                if len(consumers) != 3 or {x.args[0].id for x in consumers} != set(roots):
                    raise ValueError("Expected exactly one Q/K/V GEMM read per pack")
                insert_before[node.lineno] = ast.Assign(
                    targets=[ast.Name(id=shared, ctx=ast.Store())],
                    value=ast.Name(id=roots[0], ctx=ast.Load()),
                )
                insert_after[max((x.lineno for x in consumers))] = ast.Delete(
                    targets=[ast.Name(id=shared, ctx=ast.Del())]
                )
                for consumer in consumers:
                    consumer.args[0] = ast.Name(id=shared, ctx=ast.Load())
                scale_evidence = None
                scalar_names = [_name(x) for x in node.args[1:4]]
                producers = []
                for scalar in scalar_names:
                    found = [
                        x
                        for x in ast.walk(replacement)
                        if isinstance(x, ast.Call)
                        and isinstance(x.func, ast.Attribute)
                        and (x.func.attr == "run")
                        and (len(x.args) == 5)
                        and isinstance(x.args[1], ast.Name)
                        and (x.args[1].id == scalar)
                    ]
                    if len(found) != 1:
                        raise ValueError("Unrecognized QKV scalar producer")
                    producers.append(found[0])
                if (
                    len({_call_name(x) for x in producers}) != 1
                    or len(
                        {ast.unparse(ast.Tuple(elts=x.args[3:], ctx=ast.Load())) for x in producers}
                    )
                    != 1
                ):
                    raise ValueError("Scalar reductions differ")
                partials = [_name(x.args[0]) for x in producers]
                fused = [
                    x
                    for x in ast.walk(replacement)
                    if isinstance(x, ast.Call)
                    and isinstance(x.func, ast.Attribute)
                    and (x.func.attr == "run")
                    and isinstance(x.func.value, ast.Name)
                    and all(
                        (
                            any((isinstance(a, ast.Name) and a.id == p for a in x.args))
                            for p in partials
                        )
                    )
                ]
                if len(fused) != 1:
                    raise ValueError("Partial amax tensors lack a common producer")
                producer = fused[0]
                body = ast.parse(declarations[producer.func.value.id])
                kernel_fn = next((x for x in body.body if isinstance(x, ast.FunctionDef)))
                names = [x.arg for x in kernel_fn.args.args]
                symbols = [
                    names[
                        next(
                            (
                                i
                                for i, a in enumerate(producer.args)
                                if isinstance(a, ast.Name) and a.id == p
                            )
                        )
                    ]
                    for p in partials
                ]
                writes = []
                for symbol in symbols:
                    matches = [
                        x
                        for x in ast.walk(kernel_fn)
                        if isinstance(x, ast.Call)
                        and _call_name(x) == "tl.store"
                        and any(
                            (
                                isinstance(a, ast.Name) and a.id == symbol
                                for a in ast.walk(x.args[0])
                            )
                        )
                    ]
                    if len(matches) != 1:
                        raise ValueError("Ambiguous partial amax writes")
                    writes.append(tuple((ast.unparse(a) for a in matches[0].args[1:])))
                if len(set(writes)) != 1:
                    raise ValueError("Partial amax values/reduction boundaries differ")
                source_scale = _name(producers[0].args[2])
                last_mm = max((x.lineno for x in consumers))
                for assignment in assignments.values():
                    if producers[0].lineno < assignment.lineno < last_mm and any(
                        (
                            isinstance(x, ast.Name)
                            and isinstance(x.ctx, ast.Load)
                            and (x.id == source_scale)
                            for x in ast.walk(assignment.value)
                        )
                    ):
                        raise ValueError("First scale storage is reused before the last GEMM")
                scale_shared = f"exact_qkv_scale_{node.lineno}"
                insert_after[producers[0].lineno] = ast.Assign(
                    targets=[ast.Name(id=scale_shared, ctx=ast.Store())],
                    value=ast.Name(id=source_scale, ctx=ast.Load()),
                )
                for consumer in consumers:
                    consumer.args[2] = ast.Name(id=scale_shared, ctx=ast.Load())
                insert_after[last_mm].targets.append(ast.Name(id=scale_shared, ctx=ast.Del()))
                skip_lines.update((x.lineno for x in producers[1:]))
                scale_evidence = dict(
                    removed_launches=2,
                    partials=partials,
                    identical_store=writes[0],
                    original_reduction=producer.func.value.id,
                    scalar_reduction=_call_name(producers[0]),
                )
                new_name = name + "_exact_one_store"
                if new_name not in new_globals:
                    body = _single_store_source(kernels[name], name, new_name)
                    compiler = AsyncCompile()
                    new_globals[new_name] = compiler.triton(new_name, body, device_str="cuda")
                    compiler.wait(new_globals)
                node.func.value.id = new_name
                groups.append(
                    dict(
                        kernel=name,
                        shared_root=roots[0],
                        redirected_reads=roots[1:],
                        real_outputs=real,
                        explicit_lifetime=shared,
                        original_memory_plan_preserved=True,
                        scale_dedup=scale_evidence,
                    )
                )
            gemm_groups, replacements = ([], {})
            if self.gemm_amax:
                from .kernel_gemm_amax import compress_partial

                for statement in ast.walk(replacement):
                    if not isinstance(statement, ast.Expr) or not isinstance(
                        statement.value, ast.Call
                    ):
                        continue
                    node = statement.value
                    if _call_name(node) != "extern_kernels._scaled_mm":
                        continue
                    weight = node.args[1]
                    if (
                        _call_name(weight) != "reinterpret_tensor"
                        or ast.unparse(weight.args[1]) != "(3072, 14336)"
                    ):
                        continue
                    outputs = [x.value for x in node.keywords if x.arg == "out"]
                    if len(outputs) != 1:
                        raise ValueError("Unrecognized FFN-up out storage")
                    out_name = _name(outputs[0])
                    reductions = [
                        x
                        for x in ast.walk(replacement)
                        if isinstance(x, ast.Call)
                        and isinstance(x.func, ast.Attribute)
                        and isinstance(x.func.value, ast.Name)
                        and x.func.value.id.startswith(
                            "triton_red_fused__to_copy_abs_add_amax_gelu_"
                        )
                        and isinstance(x.args[0], ast.Name)
                        and (x.args[0].id == out_name)
                    ]
                    quantizers = [
                        x
                        for x in ast.walk(replacement)
                        if isinstance(x, ast.Call)
                        and isinstance(x.func, ast.Attribute)
                        and isinstance(x.func.value, ast.Name)
                        and x.func.value.id.startswith(
                            "triton_poi_fused__to_copy_add_clamp_clamp_min_div_gelu_"
                        )
                        and isinstance(x.args[0], ast.Name)
                        and (x.args[0].id == out_name)
                    ]
                    if len(reductions) != 1 or len(quantizers) != 1:
                        raise ValueError(
                            "FFN-up must feed exactly one original GELU amax/quantizer"
                        )
                    reduction, quantizer = (reductions[0], quantizers[0])
                    if _name(reduction.args[1]) != _name(quantizer.args[1]):
                        raise ValueError("GELU bias differs")
                    numel = ast.literal_eval(quantizer.args[-1])
                    if not isinstance(numel, int) or numel % 14336:
                        raise ValueError("Unrecognized real GELU row count")
                    partial = f"exact_gemm_partial_{node.lineno}"
                    call = ast.Call(
                        func=ast.Name(id="_exact_gemm_gelu_amax", ctx=ast.Load()),
                        args=node.args
                        + [copy.deepcopy(reduction.args[1]), ast.Constant(numel // 14336)],
                        keywords=node.keywords,
                    )
                    replacements[node.lineno] = ast.Assign(
                        targets=[ast.Name(id=partial, ctx=ast.Store())], value=call
                    )
                    replacements[reduction.lineno] = ast.Expr(
                        value=ast.Call(
                            func=ast.Name(id="_exact_gemm_compress_amax", ctx=ast.Load()),
                            args=[
                                ast.Name(id=partial, ctx=ast.Load()),
                                copy.deepcopy(reduction.args[2]),
                            ],
                            keywords=[],
                        )
                    )
                    insert_after[reduction.lineno] = ast.Delete(
                        targets=[ast.Name(id=partial, ctx=ast.Del())]
                    )
                    gemm_groups.append(
                        dict(
                            output=out_name,
                            real_rows=numel // 14336,
                            original_reduction=reduction.func.value.id,
                            block=[64, 128],
                            warps=8,
                            stages=3,
                            accumulation="FP32 promotion every K128, native scale product",
                            original_FP16_boundary=True,
                        )
                    )
                new_globals.update(
                    _exact_gemm_gelu_amax=self._gemm, _exact_gemm_compress_amax=compress_partial
                )
            if not groups and (not gemm_groups):
                continue

            class Lifetimes(ast.NodeTransformer):
                def visit_Expr(self, statement):
                    before = (
                        [insert_before[statement.lineno]]
                        if statement.lineno in insert_before
                        else []
                    )
                    after = (
                        [insert_after[statement.lineno]] if statement.lineno in insert_after else []
                    )
                    body = replacements.get(statement.lineno, statement)
                    return before + ([] if statement.lineno in skip_lines else [body]) + after

            replacement = Lifetimes().visit(replacement)
            ast.fix_missing_locations(replacement)
            replacement.decorator_list = []
            namespace = dict(module.__dict__)
            namespace.update(new_globals)
            compiled = compile(
                ast.Module(body=[replacement], type_ignores=[]),
                str(path) + ":exact_qkv_in_memory",
                "exec",
            )
            exec(compiled, namespace)
            candidate = namespace["call"]
            if fn.__code__.co_freevars or candidate.__code__.co_freevars:
                raise ValueError("Cannot replace a closure")
            module.__dict__.update(new_globals)
            original = fn.__code__
            self.patches.append((fn, original, candidate.__code__))
            self.evidence.append(
                dict(
                    path=str(path),
                    sha256=hashlib.sha256(source.encode()).hexdigest(),
                    pack_groups=groups,
                    count=len(groups),
                    gemm_groups=gemm_groups,
                    candidate_code_sha256=hashlib.sha256(
                        ast.unparse(replacement).encode()
                    ).hexdigest(),
                )
            )
            if self.active:
                fn.__code__ = candidate.__code__
        return self.evidence

    def configure(self, enabled):
        self.active = enabled
        for fn, original, candidate in self.patches:
            fn.__code__ = candidate if enabled else original
