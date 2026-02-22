"""brms (R) benchmark adapter."""
import json
import os
import subprocess
import tempfile
import numpy as np
from pathlib import Path
from .base import FrameworkAdapter, BenchmarkResult
from ..runners.metrics import measure_performance

R_SCRIPTS_DIR = Path(__file__).parent.parent / "r_scripts"

SUPPORTED_MODELS = [
    "beta_binomial", "normal_mean", "linear_regression",
    "logistic_regression", "hierarchical_intercepts", "eight_schools",
    "wide_regression", "deep_hierarchy",
]


class BRMSAdapter(FrameworkAdapter):
    @property
    def name(self) -> str:
        return "brms"

    def supports_model(self, model_name: str) -> bool:
        return model_name in SUPPORTED_MODELS

    def run(self, model_name: str, data: dict,
            num_samples: int = 1000, num_warmup: int = 1000,
            num_chains: int = 4, seed: int = 42) -> BenchmarkResult:

        r_script = R_SCRIPTS_DIR / f"{model_name}.R"
        if not r_script.exists():
            return BenchmarkResult(self.name, model_name, 0, 0, 0, 0,
                                   error=f"R script not found: {r_script}")

        # Write data to temp JSON file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            # Convert numpy arrays to lists for JSON
            json_data = {}
            for k, v in data.items():
                if isinstance(v, np.ndarray):
                    json_data[k] = v.tolist()
                else:
                    json_data[k] = v
            json.dump(json_data, f)
            data_path = f.name

        try:
            with measure_performance() as perf:
                result = subprocess.run(
                    ["Rscript", str(r_script), data_path,
                     str(num_samples), str(num_warmup), str(num_chains), str(seed)],
                    capture_output=True, text=True, timeout=600,
                )

            if result.returncode != 0:
                return BenchmarkResult(
                    self.name, model_name, perf.wall_seconds, perf.peak_memory_mb,
                    0, 0, error=f"R error: {result.stderr[:200]}",
                )

            # Parse JSON output from R script
            try:
                output = json.loads(result.stdout)
                min_e = output.get("min_ess", 0.0)
                ess_s = min_e / perf.wall_seconds if perf.wall_seconds > 0 else 0.0
                return BenchmarkResult(
                    framework=self.name, model_name=model_name,
                    wall_time_seconds=perf.wall_seconds,
                    peak_memory_mb=perf.peak_memory_mb,
                    ess_per_second=ess_s, min_ess=min_e,
                )
            except json.JSONDecodeError:
                return BenchmarkResult(
                    self.name, model_name, perf.wall_seconds, perf.peak_memory_mb,
                    0, 0, error="Failed to parse R output",
                )
        finally:
            os.unlink(data_path)
