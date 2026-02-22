"""Model registry for benchmarks."""
from dataclasses import dataclass, field
from typing import Callable, Dict, List


@dataclass
class BenchmarkModel:
    name: str
    num_params: int
    data_generator: Callable
    description: str = ""
    tags: List[str] = field(default_factory=list)


MODEL_REGISTRY: Dict[str, BenchmarkModel] = {}


def register(name: str, num_params: int, data_generator: Callable, description: str = "", tags: list = None):
    MODEL_REGISTRY[name] = BenchmarkModel(
        name=name, num_params=num_params,
        data_generator=data_generator, description=description,
        tags=tags or [],
    )


def get_models(tags: list = None) -> Dict[str, BenchmarkModel]:
    if tags is None:
        return MODEL_REGISTRY
    return {k: v for k, v in MODEL_REGISTRY.items() if any(t in v.tags for t in tags)}
