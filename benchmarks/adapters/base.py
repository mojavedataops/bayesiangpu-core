"""Base adapter interface for benchmark frameworks."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Optional
import numpy as np


@dataclass
class BenchmarkResult:
    framework: str
    model_name: str
    wall_time_seconds: float
    peak_memory_mb: float
    ess_per_second: float
    min_ess: float
    samples: Optional[Dict[str, np.ndarray]] = field(default=None, repr=False)
    error: Optional[str] = None

    @property
    def succeeded(self) -> bool:
        return self.error is None


class FrameworkAdapter(ABC):
    """Base class for benchmark framework adapters."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Framework name for display."""
        pass

    @abstractmethod
    def run(self, model_name: str, data: dict,
            num_samples: int = 1000, num_warmup: int = 1000,
            num_chains: int = 4, seed: int = 42) -> BenchmarkResult:
        """Run inference and return results."""
        pass

    @abstractmethod
    def supports_model(self, model_name: str) -> bool:
        """Check if this framework supports the given model."""
        pass
