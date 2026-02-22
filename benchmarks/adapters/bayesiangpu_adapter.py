"""BayesianGPU benchmark adapter."""
import numpy as np
from .base import FrameworkAdapter, BenchmarkResult
from ..runners.metrics import measure_performance, compute_ess, min_ess

SUPPORTED_MODELS = [
    "beta_binomial", "normal_mean", "linear_regression",
    "logistic_regression", "hierarchical_intercepts", "eight_schools",
    "wide_regression", "deep_hierarchy",
]


class BayesianGPUAdapter(FrameworkAdapter):
    @property
    def name(self) -> str:
        try:
            import bayesiangpu as bg
            backend = bg.backend_name()
        except Exception:
            backend = "unknown"
        return f"BayesianGPU ({backend})"

    def supports_model(self, model_name: str) -> bool:
        return model_name in SUPPORTED_MODELS

    def run(self, model_name: str, data: dict,
            num_samples: int = 1000, num_warmup: int = 1000,
            num_chains: int = 4, seed: int = 42) -> BenchmarkResult:
        import bayesiangpu as bg

        builder = getattr(self, f"_build_{model_name}", None)
        if builder is None:
            return BenchmarkResult(self.name, model_name, 0, 0, 0, 0, error="No builder")

        model = builder(bg, data)

        with measure_performance() as perf:
            result = bg.sample(model, num_samples=num_samples,
                               num_warmup=num_warmup, num_chains=num_chains,
                               seed=seed)

        # Extract full sample chains for ESS computation
        samples = {}
        for name in result.param_names:
            samples[name] = np.array(result.get_samples(name))

        ess_vals = compute_ess(samples, num_chains) if samples else {}
        min_e = min_ess(ess_vals) if ess_vals else 0.0
        ess_s = min_e / perf.wall_seconds if perf.wall_seconds > 0 else 0.0

        return BenchmarkResult(
            framework=self.name, model_name=model_name,
            wall_time_seconds=perf.wall_seconds,
            peak_memory_mb=perf.peak_memory_mb,
            ess_per_second=ess_s, min_ess=min_e,
        )

    def _build_beta_binomial(self, bg, data):
        model = bg.Model()
        model.param("theta", bg.Beta(1, 1))
        model.observe(bg.Binomial(data["n_trials"], "theta"), [float(data["y"])])
        return model

    def _build_normal_mean(self, bg, data):
        model = bg.Model()
        model.param("mu", bg.Normal(0, 10))
        model.param("sigma", bg.HalfNormal(5))
        model.observe(bg.Normal("mu", "sigma"), data["y"])
        return model

    def _build_linear_regression(self, bg, data):
        model = bg.Model()
        p = data["p"]
        model.param("beta", bg.Normal(0, 1), size=p)
        model.param("sigma", bg.HalfNormal(2))
        model.observe(bg.Normal(bg.LinearPredictor(data["X"], "beta"), "sigma"), data["y"])
        return model

    def _build_logistic_regression(self, bg, data):
        model = bg.Model()
        p = data["p"]
        model.param("beta", bg.Normal(0, 1), size=p)
        model.observe(bg.Bernoulli(bg.LinearPredictor(data["X"], "beta")), data["y"])
        return model

    def _build_hierarchical_intercepts(self, bg, data):
        model = bg.Model()
        model.param("mu", bg.Normal(0, 10))
        model.param("tau", bg.HalfNormal(5))
        model.param("theta", bg.Normal("mu", "tau"), size=data["n_groups"])
        model.param("sigma", bg.HalfNormal(5))
        y = data["y"]
        group_ids = data["group_ids"]
        model.observe(bg.Normal("theta", "sigma"), y,
                      known={"theta_idx": [float(g) for g in group_ids]})
        return model

    def _build_eight_schools(self, bg, data):
        model = bg.Model()
        model.param("mu", bg.Normal(0, 5))
        model.param("tau", bg.HalfCauchy(5))
        model.param("theta", bg.Normal("mu", "tau"), size=data["J"])
        model.observe(bg.Normal("theta", "sigma"), data["y"],
                      known={"sigma": [float(s) for s in data["sigma"]]})
        return model

    def _build_wide_regression(self, bg, data):
        return self._build_linear_regression(bg, data)  # Same structure

    def _build_deep_hierarchy(self, bg, data):
        model = bg.Model()
        model.param("mu", bg.Normal(0, 10))
        model.param("tau_group", bg.HalfNormal(5))
        model.param("group_means", bg.Normal("mu", "tau_group"), size=data["n_groups"])
        model.param("tau_sub", bg.HalfNormal(5))
        n_total_sub = data["n_groups"] * data["n_subgroups"]
        model.param("sub_means", bg.Normal(0, 5), size=n_total_sub)
        model.param("sigma", bg.HalfNormal(5))
        model.observe(bg.Normal("sub_means", "sigma"), data["y"])
        return model
