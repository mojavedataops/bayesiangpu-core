"""PyMC benchmark adapter."""
import numpy as np
from .base import FrameworkAdapter, BenchmarkResult
from ..runners.metrics import measure_performance, compute_ess, min_ess

SUPPORTED_MODELS = [
    "beta_binomial", "normal_mean", "linear_regression",
    "logistic_regression", "hierarchical_intercepts", "eight_schools",
    "wide_regression", "deep_hierarchy",
]


class PyMCAdapter(FrameworkAdapter):
    @property
    def name(self) -> str:
        return "PyMC"

    def supports_model(self, model_name: str) -> bool:
        return model_name in SUPPORTED_MODELS

    def run(self, model_name: str, data: dict,
            num_samples: int = 1000, num_warmup: int = 1000,
            num_chains: int = 4, seed: int = 42) -> BenchmarkResult:
        import pymc as pm
        import arviz as az

        builder = getattr(self, f"_build_{model_name}", None)
        if builder is None:
            return BenchmarkResult(self.name, model_name, 0, 0, 0, 0, error="No builder")

        pymc_model = builder(pm, data)

        with measure_performance() as perf:
            with pymc_model:
                trace = pm.sample(
                    draws=num_samples, tune=num_warmup,
                    chains=num_chains, random_seed=seed,
                    progressbar=False,
                )

        # Compute ESS via ArviZ
        ess_data = az.ess(trace)
        ess_values = {}
        for var in ess_data.data_vars:
            vals = ess_data[var].values
            if vals.ndim == 0:
                ess_values[var] = float(vals)
            else:
                for i, v in enumerate(vals.flat):
                    ess_values[f"{var}[{i}]"] = float(v)

        min_e = min(ess_values.values()) if ess_values else 0.0
        ess_s = min_e / perf.wall_seconds if perf.wall_seconds > 0 else 0.0

        return BenchmarkResult(
            framework=self.name, model_name=model_name,
            wall_time_seconds=perf.wall_seconds,
            peak_memory_mb=perf.peak_memory_mb,
            ess_per_second=ess_s, min_ess=min_e,
        )

    def _build_beta_binomial(self, pm, data):
        with pm.Model() as model:
            theta = pm.Beta("theta", alpha=1, beta=1)
            pm.Binomial("y", n=data["n_trials"], p=theta, observed=data["y"])
        return model

    def _build_normal_mean(self, pm, data):
        with pm.Model() as model:
            mu = pm.Normal("mu", mu=0, sigma=10)
            sigma = pm.HalfNormal("sigma", sigma=5)
            pm.Normal("y", mu=mu, sigma=sigma, observed=data["y"])
        return model

    def _build_linear_regression(self, pm, data):
        X = np.array(data["X"]) if not isinstance(data["X"], np.ndarray) else data["X"]
        with pm.Model() as model:
            beta = pm.Normal("beta", mu=0, sigma=1, shape=data["p"])
            sigma = pm.HalfNormal("sigma", sigma=2)
            mu = pm.math.dot(X, beta)
            pm.Normal("y", mu=mu, sigma=sigma, observed=data["y"])
        return model

    def _build_logistic_regression(self, pm, data):
        X = np.array(data["X"]) if not isinstance(data["X"], np.ndarray) else data["X"]
        with pm.Model() as model:
            beta = pm.Normal("beta", mu=0, sigma=1, shape=data["p"])
            p = pm.math.sigmoid(pm.math.dot(X, beta))
            pm.Bernoulli("y", p=p, observed=data["y"])
        return model

    def _build_hierarchical_intercepts(self, pm, data):
        with pm.Model() as model:
            mu = pm.Normal("mu", mu=0, sigma=10)
            tau = pm.HalfNormal("tau", sigma=5)
            theta = pm.Normal("theta", mu=mu, sigma=tau, shape=data["n_groups"])
            sigma = pm.HalfNormal("sigma", sigma=5)
            pm.Normal("y", mu=theta[data["group_ids"]], sigma=sigma, observed=data["y"])
        return model

    def _build_eight_schools(self, pm, data):
        with pm.Model() as model:
            mu = pm.Normal("mu", mu=0, sigma=5)
            tau = pm.HalfCauchy("tau", beta=5)
            theta = pm.Normal("theta", mu=mu, sigma=tau, shape=data["J"])
            pm.Normal("y", mu=theta, sigma=data["sigma"], observed=data["y"])
        return model

    def _build_wide_regression(self, pm, data):
        return self._build_linear_regression(pm, data)

    def _build_deep_hierarchy(self, pm, data):
        n_groups = data["n_groups"]
        n_sub = data["n_subgroups"]
        with pm.Model() as model:
            mu = pm.Normal("mu", mu=0, sigma=10)
            tau_group = pm.HalfNormal("tau_group", sigma=5)
            group_means = pm.Normal("group_means", mu=mu, sigma=tau_group, shape=n_groups)
            tau_sub = pm.HalfNormal("tau_sub", sigma=5)
            sub_means = pm.Normal("sub_means",
                                  mu=group_means[np.repeat(np.arange(n_groups), n_sub)],
                                  sigma=tau_sub, shape=n_groups * n_sub)
            sigma = pm.HalfNormal("sigma", sigma=5)
            pm.Normal("y", mu=sub_means[data["subgroup_ids"]], sigma=sigma, observed=data["y"])
        return model
