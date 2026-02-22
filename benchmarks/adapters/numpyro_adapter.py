"""NumPyro benchmark adapter."""
import numpy as np
from .base import FrameworkAdapter, BenchmarkResult
from ..runners.metrics import measure_performance, compute_ess, min_ess

SUPPORTED_MODELS = [
    "beta_binomial", "normal_mean", "linear_regression",
    "logistic_regression", "hierarchical_intercepts", "eight_schools",
    "wide_regression", "deep_hierarchy",
]


class NumPyroAdapter(FrameworkAdapter):
    @property
    def name(self) -> str:
        return "NumPyro"

    def supports_model(self, model_name: str) -> bool:
        return model_name in SUPPORTED_MODELS

    def run(self, model_name: str, data: dict,
            num_samples: int = 1000, num_warmup: int = 1000,
            num_chains: int = 4, seed: int = 42) -> BenchmarkResult:
        import jax
        import jax.numpy as jnp
        import numpyro
        import numpyro.distributions as dist
        from numpyro.infer import MCMC, NUTS

        builder = getattr(self, f"_build_{model_name}", None)
        if builder is None:
            return BenchmarkResult(self.name, model_name, 0, 0, 0, 0, error="No builder")

        model_fn = builder(numpyro, dist, jnp, data)

        # JIT warmup (excluded from timing)
        kernel = NUTS(model_fn)
        mcmc = MCMC(kernel, num_warmup=10, num_samples=10, num_chains=1)
        mcmc.run(jax.random.PRNGKey(0))
        mcmc._last_state = None  # Reset

        # Actual benchmark
        with measure_performance() as perf:
            kernel = NUTS(model_fn)
            mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples,
                        num_chains=num_chains, progress_bar=False)
            mcmc.run(jax.random.PRNGKey(seed))

        samples = {k: np.array(v) for k, v in mcmc.get_samples().items()}

        # Flatten multi-dim params for ESS computation
        flat_samples = {}
        for k, v in samples.items():
            if v.ndim == 1:
                flat_samples[k] = v.tolist()
            else:
                for i in range(v.shape[1]):
                    flat_samples[f"{k}[{i}]"] = v[:, i].tolist()

        ess_vals = compute_ess(flat_samples, num_chains) if flat_samples else {}
        min_e = min_ess(ess_vals)
        ess_s = min_e / perf.wall_seconds if perf.wall_seconds > 0 else 0.0

        return BenchmarkResult(
            framework=self.name, model_name=model_name,
            wall_time_seconds=perf.wall_seconds,
            peak_memory_mb=perf.peak_memory_mb,
            ess_per_second=ess_s, min_ess=min_e,
        )

    def _build_beta_binomial(self, numpyro, dist, jnp, data):
        def model():
            theta = numpyro.sample("theta", dist.Beta(1, 1))
            numpyro.sample("y", dist.Binomial(total_count=data["n_trials"], probs=theta),
                           obs=jnp.array(data["y"]))
        return model

    def _build_normal_mean(self, numpyro, dist, jnp, data):
        y = jnp.array(data["y"])
        def model():
            mu = numpyro.sample("mu", dist.Normal(0, 10))
            sigma = numpyro.sample("sigma", dist.HalfNormal(5))
            numpyro.sample("y", dist.Normal(mu, sigma), obs=y)
        return model

    def _build_linear_regression(self, numpyro, dist, jnp, data):
        X = jnp.array(data["X"])
        y = jnp.array(data["y"])
        p = data["p"]
        def model():
            beta = numpyro.sample("beta", dist.Normal(0, 1).expand([p]))
            sigma = numpyro.sample("sigma", dist.HalfNormal(2))
            mu = jnp.dot(X, beta)
            numpyro.sample("y", dist.Normal(mu, sigma), obs=y)
        return model

    def _build_logistic_regression(self, numpyro, dist, jnp, data):
        X = jnp.array(data["X"])
        y = jnp.array(data["y"])
        p = data["p"]
        def model():
            beta = numpyro.sample("beta", dist.Normal(0, 1).expand([p]))
            logits = jnp.dot(X, beta)
            numpyro.sample("y", dist.Bernoulli(logits=logits), obs=y)
        return model

    def _build_hierarchical_intercepts(self, numpyro, dist, jnp, data):
        y = jnp.array(data["y"])
        group_ids = jnp.array(data["group_ids"])
        def model():
            mu = numpyro.sample("mu", dist.Normal(0, 10))
            tau = numpyro.sample("tau", dist.HalfNormal(5))
            theta = numpyro.sample("theta", dist.Normal(mu, tau).expand([data["n_groups"]]))
            sigma = numpyro.sample("sigma", dist.HalfNormal(5))
            numpyro.sample("y", dist.Normal(theta[group_ids], sigma), obs=y)
        return model

    def _build_eight_schools(self, numpyro, dist, jnp, data):
        y = jnp.array(data["y"], dtype=jnp.float32)
        sigma = jnp.array(data["sigma"], dtype=jnp.float32)
        def model():
            mu = numpyro.sample("mu", dist.Normal(0, 5))
            tau = numpyro.sample("tau", dist.HalfCauchy(5))
            theta = numpyro.sample("theta", dist.Normal(mu, tau).expand([data["J"]]))
            numpyro.sample("y", dist.Normal(theta, sigma), obs=y)
        return model

    def _build_wide_regression(self, numpyro, dist, jnp, data):
        return self._build_linear_regression(numpyro, dist, jnp, data)

    def _build_deep_hierarchy(self, numpyro, dist, jnp, data):
        y = jnp.array(data["y"])
        subgroup_ids = jnp.array(data["subgroup_ids"])
        n_groups = data["n_groups"]
        n_sub = data["n_subgroups"]
        def model():
            mu = numpyro.sample("mu", dist.Normal(0, 10))
            tau_group = numpyro.sample("tau_group", dist.HalfNormal(5))
            group_means = numpyro.sample("group_means",
                                         dist.Normal(mu, tau_group).expand([n_groups]))
            tau_sub = numpyro.sample("tau_sub", dist.HalfNormal(5))
            sub_means = numpyro.sample("sub_means", dist.Normal(
                jnp.repeat(group_means, n_sub), tau_sub))
            sigma = numpyro.sample("sigma", dist.HalfNormal(5))
            numpyro.sample("y", dist.Normal(sub_means[subgroup_ids], sigma), obs=y)
        return model
