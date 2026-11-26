import ast
import hashlib

from .kernel_native_normpack import NativeNormpack


class FrozenNativeNormpack(NativeNormpack):
    def _compile(self, source, name, spec):
        candidates = [kernel for kernel, _, body, *_ in self.kernels if body == source]
        if not candidates or len(candidates[0].launchers) != 1:
            raise RuntimeError("Original norm launch must be warmed before freezing its reduction")
        original = candidates[0]
        config = original.launchers[0].config
        params = dict(
            num_warps=config.num_warps, num_stages=config.num_stages, num_ctas=config.num_ctas
        )
        if config.maxnreg is not None:
            params["maxnreg"] = config.maxnreg
        launch = (
            "triton.Config("
            + repr(config.kwargs)
            + ", "
            + ", ".join((f"{key}={value!r}" for key, value in params.items()))
            + ")"
        )
        tree = ast.parse(source)
        function = next((node for node in tree.body if isinstance(node, ast.FunctionDef)))
        changed = False
        for decorator in function.decorator_list:
            if isinstance(decorator, ast.Call) and ast.unparse(decorator.func) in {
                "triton_heuristics.persistent_reduction",
                "triton_heuristics.reduction",
                "triton_heuristics.pointwise",
            }:
                decorator.func = ast.parse("triton_heuristics.cached_autotune", mode="eval").body
                decorator.keywords = [
                    kw
                    for kw in decorator.keywords
                    if kw.arg in ("size_hints", "filename", "triton_meta", "inductor_meta")
                ]
                decorator.keywords.extend(
                    [
                        ast.keyword(
                            arg="configs",
                            value=ast.List(
                                elts=[ast.parse(launch, mode="eval").body], ctx=ast.Load()
                            ),
                        ),
                        ast.keyword(
                            arg="heuristic_type",
                            value=ast.parse(
                                "triton_heuristics.HeuristicType." + original.heuristic_type.name,
                                mode="eval",
                            ).body,
                        ),
                    ]
                )
                changed = True
        if not changed:
            raise ValueError("Unsupported norm decorator; no numerical fallback")
        ast.fix_missing_locations(tree)
        result = super()._compile(ast.unparse(tree), name + "_frozen_launch", spec)
        self.evidence[-1].update(
            original_source_sha256=hashlib.sha256(source.encode()).hexdigest(),
            frozen_launch=dict(kwargs=dict(config.kwargs), **params),
            independent_autotuning=False,
        )
        return result
