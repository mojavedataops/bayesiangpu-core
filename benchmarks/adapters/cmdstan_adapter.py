"""CmdStan benchmark adapter."""
import numpy as np
from pathlib import Path
from .base import FrameworkAdapter, BenchmarkResult
from ..runners.metrics import measure_performance, compute_ess, min_ess

STAN_DIR = Path(__file__).parent.parent / "stan_models"

SUPPORTED_MODELS = [
    "beta_binomial", "normal_mean", "linear_regression",
    "logistic_regression", "hierarchical_intercepts", "eight_schools",
    "wide_regression", "deep_hierarchy",
]


class CmdStanAdapter(FrameworkAdapter):
    @property
    def name(self) -> str:
        return "CmdStan"

    def supports_model(self, model_name: str) -> bool:
        return model_name in SUPPORTED_MODELS

    def run(self, model_name: str, data: dict,
            num_samples: int = 1000, num_warmup: int = 1000,
            num_chains: int = 4, seed: int = 42) -> BenchmarkResult:
        from cmdstanpy import CmdStanModel

        stan_file = STAN_DIR / f"{model_name}.stan"
        if not stan_file.exists():
            return BenchmarkResult(self.name, model_name, 0, 0, 0, 0,
                                   error=f"Stan file not found: {stan_file}")

        # Compile (excluded from timing)
        stan_model = CmdStanModel(stan_file=str(stan_file))

        # Prepare data
        stan_data = getattr(self, f"_data_{model_name}", lambda d: d)(data)

        with measure_performance() as perf:
            fit = stan_model.sample(
                data=stan_data, chains=num_chains,
                iter_sampling=num_samples, iter_warmup=num_warmup,
                seed=seed, show_progress=False,
            )

        # Extract samples
        samples = {}
        for name in fit.column_names:
            if name not in ("lp__", "accept_stat__", "stepsize__", "treedepth__",
                            "n_leapfrog__", "divergent__", "energy__"):
                samples[name] = fit.stan_variable(name).flatten().tolist()

        ess_vals = compute_ess(samples, num_chains) if samples else {}
        min_e = min_ess(ess_vals)
        ess_s = min_e / perf.wall_seconds if perf.wall_seconds > 0 else 0.0

        return BenchmarkResult(
            framework=self.name, model_name=model_name,
            wall_time_seconds=perf.wall_seconds,
            peak_memory_mb=perf.peak_memory_mb,
            ess_per_second=ess_s, min_ess=min_e,
        )

    def _data_beta_binomial(self, data):
        return {"N": data["n_trials"], "y": data["y"]}

    def _data_normal_mean(self, data):
        return {"N": len(data["y"]), "y": data["y"]}

    def _data_linear_regression(self, data):
        X = data["X"]
        if isinstance(X, np.ndarray):
            X = X.tolist()
        return {"N": data["n"], "P": data["p"], "X": X, "y": data["y"]}

    def _data_logistic_regression(self, data):
        return self._data_linear_regression(data)

    def _data_hierarchical_intercepts(self, data):
        return {
            "N": len(data["y"]), "J": data["n_groups"],
            "group": [g + 1 for g in data["group_ids"]],  # Stan 1-indexed
            "y": data["y"],
        }

    def _data_eight_schools(self, data):
        return {"J": data["J"], "y": data["y"], "sigma": data["sigma"]}

    def _data_wide_regression(self, data):
        return self._data_linear_regression(data)

    def _data_deep_hierarchy(self, data):
        return {
            "N": len(data["y"]),
            "J": data["n_groups"],
            "K": data["n_subgroups"],
            "group": [g + 1 for g in data["group_ids"]],
            "subgroup": [s + 1 for s in data["subgroup_ids"]],
            "y": data["y"],
        }
