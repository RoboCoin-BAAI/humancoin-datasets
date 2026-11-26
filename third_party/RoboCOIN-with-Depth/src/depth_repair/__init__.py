"""深度修复（精简：仅 DA3）。"""

from src.depth_repair.base import BaseDepthRepair
from src.depth_repair.da3 import DA3DepthRepair


def create_depth_repair(strategy: str = "da3", **kwargs) -> BaseDepthRepair:
    registry = {
        "da3": DA3DepthRepair,
    }
    if strategy not in registry:
        raise ValueError(
            f"Unknown strategy: {strategy!r}. Available: {list(registry.keys())}"
        )
    return registry[strategy](**kwargs)
