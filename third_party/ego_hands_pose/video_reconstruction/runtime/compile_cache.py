"""Per-device compiler caches; cache files are generated on the target host."""

import hashlib
import json
import os
import time
from pathlib import Path

import torch

from .runtime import ROOT, digest

CANVASES = ((480, 480), (672, 480), (832, 480), (832, 640))


class PersistentCompileCache:
    def __init__(self, config, device):
        import triton

        prop = torch.cuda.get_device_properties(device)
        manifest = json.loads((ROOT / "assets/manifest.json").read_text())
        self.identity = dict(
            torch=str(torch.__version__),
            cuda=torch.version.cuda,
            cudnn=torch.backends.cudnn.version(),
            triton=triton.__version__,
            gpu=prop.name,
            capability=[prop.major, prop.minor],
            precision=config["acceleration_profile"],
            files=manifest["sha256"],
        )
        self.key = hashlib.sha256(json.dumps(self.identity, sort_keys=True).encode()).hexdigest()
        cache_base = os.environ.get("HAND_POSE_COMPILE_CACHE")
        self.root = (Path(cache_base) if cache_base else ROOT / ".cache") / self.key
        self.root.mkdir(parents=True, exist_ok=True)
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(self.root / "inductor")
        os.environ["TRITON_CACHE_DIR"] = str(self.root / "triton")
        # Older Triton reads TRITON_CACHE_DIR directly.
        if hasattr(triton, "knobs"):
            triton.knobs.cache.dir = os.environ["TRITON_CACHE_DIR"]
        torch._inductor.config.fx_graph_cache = True
        if hasattr(torch._dynamo.config, "caching_precompile"):
            torch._dynamo.config.caching_precompile = False
        self._saved = None

    def load(self):
        path = self.root / "bundle.json"
        if not path.exists():
            return dict(status="miss", key=self.key)
        manifest = json.loads(path.read_text())
        if (
            manifest["identity"] != self.identity
            or digest(self.root / "bundle.bin") != manifest["sha256"]
        ):
            raise ValueError("Compiler cache checksum or identity mismatch")
        if torch.compiler.load_cache_artifacts((self.root / "bundle.bin").read_bytes()) is None:
            raise RuntimeError("Compiler cache could not be restored")
        return dict(status="loaded", key=self.key)

    def save_if_changed(self):
        started = time.perf_counter()
        from torch._dynamo.utils import counters

        signature = repr(dict(counters["inductor"]))
        if signature == self._saved:
            return dict(status="unchanged", seconds=0.0)
        result = torch.compiler.save_cache_artifacts()
        if result is None:
            return dict(status="empty", seconds=time.perf_counter() - started)
        blob, _ = result
        temporary = self.root / "bundle.pending"
        temporary.write_bytes(blob)
        temporary.replace(self.root / "bundle.bin")
        manifest = dict(identity=self.identity, sha256=hashlib.sha256(blob).hexdigest())
        temporary = self.root / "manifest.pending"
        temporary.write_text(json.dumps(manifest))
        temporary.replace(self.root / "bundle.json")
        self._saved = signature
        return dict(status="saved", seconds=time.perf_counter() - started)


def persist_request_cache(predictor):
    cache = getattr(predictor.vae, "_persistent_compile_cache", None)
    if cache is None:
        return dict(status="disabled", seconds=0.0)
    try:
        return cache.save_if_changed()
    except (OSError, RuntimeError) as error:
        return dict(status="failed", seconds=0.0, reason=str(error))
