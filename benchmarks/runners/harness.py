"""Benchmark orchestration harness."""
import signal
import sys
from typing import List, Optional
from ..adapters.base import BenchmarkResult, FrameworkAdapter
from ..models.registry import MODEL_REGISTRY, BenchmarkModel


class TimeoutError(Exception):
    pass


def _timeout_handler(signum, frame):
    raise TimeoutError("Benchmark timed out")


def run_single(adapter: FrameworkAdapter, model: BenchmarkModel,
               num_samples: int = 1000, num_warmup: int = 1000,
               num_chains: int = 4, seed: int = 42,
               timeout: int = 600) -> BenchmarkResult:
    """Run a single benchmark with timeout."""
    if not adapter.supports_model(model.name):
        return BenchmarkResult(
            framework=adapter.name, model_name=model.name,
            wall_time_seconds=0, peak_memory_mb=0,
            ess_per_second=0, min_ess=0,
            error="Model not supported",
        )

    # Set timeout (Unix only)
    old_handler = None
    if sys.platform != "win32":
        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(timeout)

    try:
        data = model.data_generator()
        result = adapter.run(model.name, data, num_samples, num_warmup, num_chains, seed)
        return result
    except TimeoutError:
        return BenchmarkResult(
            framework=adapter.name, model_name=model.name,
            wall_time_seconds=timeout, peak_memory_mb=0,
            ess_per_second=0, min_ess=0,
            error=f"Timed out after {timeout}s",
        )
    except Exception as e:
        return BenchmarkResult(
            framework=adapter.name, model_name=model.name,
            wall_time_seconds=0, peak_memory_mb=0,
            ess_per_second=0, min_ess=0,
            error=str(e),
        )
    finally:
        if old_handler is not None:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)


def run_benchmarks(
    adapters: List[FrameworkAdapter],
    model_names: Optional[List[str]] = None,
    num_repeats: int = 3,
    num_samples: int = 1000,
    num_warmup: int = 1000,
    num_chains: int = 4,
    seed: int = 42,
    timeout: int = 600,
) -> List[BenchmarkResult]:
    """Run full benchmark suite."""
    if model_names is None:
        model_names = list(MODEL_REGISTRY.keys())

    results = []
    for model_name in model_names:
        model = MODEL_REGISTRY[model_name]
        for adapter in adapters:
            print(f"  {adapter.name} x {model_name}...", end=" ", flush=True)
            repeat_results = []
            for r in range(num_repeats):
                result = run_single(
                    adapter, model,
                    num_samples=num_samples, num_warmup=num_warmup,
                    num_chains=num_chains, seed=seed + r,
                    timeout=timeout,
                )
                repeat_results.append(result)

            # Take median by wall time
            successful = [r for r in repeat_results if r.succeeded]
            if successful:
                successful.sort(key=lambda r: r.wall_time_seconds)
                median = successful[len(successful) // 2]
                results.append(median)
                print(f"{median.wall_time_seconds:.2f}s (ESS/s: {median.ess_per_second:.1f})")
            else:
                results.append(repeat_results[0])  # Report first error
                print(f"FAILED: {repeat_results[0].error}")

    return results
