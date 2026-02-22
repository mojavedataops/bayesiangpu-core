"""Metrics collection for benchmarks."""
import time
import tracemalloc
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict
import numpy as np


@dataclass
class TimingResult:
    wall_seconds: float
    peak_memory_mb: float


@contextmanager
def measure_performance():
    """Context manager to measure wall time and peak memory."""
    tracemalloc.start()
    start = time.perf_counter()
    result = TimingResult(0.0, 0.0)
    try:
        yield result
    finally:
        result.wall_seconds = time.perf_counter() - start
        _, peak = tracemalloc.get_traced_memory()
        result.peak_memory_mb = peak / 1024 / 1024
        tracemalloc.stop()


def compute_ess(samples: Dict[str, np.ndarray], num_chains: int) -> Dict[str, float]:
    """Compute effective sample size using ArviZ."""
    try:
        import arviz as az
        # Reshape samples to (chains, draws) format
        ess_values = {}
        for name, vals in samples.items():
            draws_per_chain = len(vals) // num_chains
            if draws_per_chain == 0:
                ess_values[name] = 0.0
                continue
            chains = np.array(vals[:draws_per_chain * num_chains]).reshape(num_chains, draws_per_chain)
            data = az.convert_to_dataset({name: chains[np.newaxis, ...]})
            ess_val = az.ess(data)
            ess_values[name] = float(ess_val[name].values)
        return ess_values
    except ImportError:
        return {}


def min_ess(ess_dict: Dict[str, float]) -> float:
    """Get minimum ESS across all parameters."""
    if not ess_dict:
        return 0.0
    return min(ess_dict.values())
