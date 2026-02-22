"""Synthetic data generators for benchmark models."""
import numpy as np


def beta_binomial_data(seed=42):
    """Beta-Binomial: 1 parameter."""
    rng = np.random.default_rng(seed)
    theta_true = 0.65
    n_trials = 100
    y = rng.binomial(n_trials, theta_true)
    return {"n_trials": n_trials, "y": int(y), "theta_true": theta_true}


def normal_mean_data(n=100, seed=42):
    """Normal mean estimation: 2 parameters."""
    rng = np.random.default_rng(seed)
    mu_true, sigma_true = 3.5, 1.2
    y = rng.normal(mu_true, sigma_true, size=n)
    return {"y": y.tolist(), "mu_true": mu_true, "sigma_true": sigma_true, "n": n}


def linear_regression_data(n=200, p=10, seed=42):
    """Linear regression: p+1 parameters."""
    rng = np.random.default_rng(seed)
    beta_true = np.zeros(p)
    beta_true[:3] = [1.0, -0.5, 0.3]  # sparse
    sigma_true = 0.5
    X = rng.standard_normal((n, p))
    y = X @ beta_true + rng.normal(0, sigma_true, size=n)
    return {"X": X, "y": y.tolist(), "beta_true": beta_true, "sigma_true": sigma_true, "n": n, "p": p}


def logistic_regression_data(n=500, p=50, seed=42):
    """Logistic regression: p parameters."""
    rng = np.random.default_rng(seed)
    beta_true = np.zeros(p)
    beta_true[:5] = [0.8, -0.6, 0.4, -0.3, 0.2]
    X = rng.standard_normal((n, p))
    logits = X @ beta_true
    probs = 1 / (1 + np.exp(-logits))
    y = rng.binomial(1, probs)
    return {"X": X, "y": y.tolist(), "beta_true": beta_true, "n": n, "p": p}


def hierarchical_intercepts_data(n_groups=20, n_per_group=50, seed=42):
    """Hierarchical intercepts: ~n_groups + 2 parameters."""
    rng = np.random.default_rng(seed)
    mu_global = 5.0
    tau = 2.0
    sigma = 1.0
    group_means = rng.normal(mu_global, tau, size=n_groups)
    y = []
    group_ids = []
    for g in range(n_groups):
        y_g = rng.normal(group_means[g], sigma, size=n_per_group)
        y.extend(y_g.tolist())
        group_ids.extend([g] * n_per_group)
    return {
        "y": y, "group_ids": group_ids,
        "n_groups": n_groups, "n_per_group": n_per_group,
        "mu_true": mu_global, "tau_true": tau, "sigma_true": sigma,
        "group_means_true": group_means.tolist(),
    }


def eight_schools_data():
    """Classic Eight Schools: 10 parameters."""
    return {
        "y": [28, 8, -3, 7, -1, 1, 18, 12],
        "sigma": [15, 10, 16, 11, 9, 11, 10, 18],
        "J": 8,
    }


def wide_regression_data(n=500, p=1000, seed=42):
    """Wide regression: p+1 parameters."""
    rng = np.random.default_rng(seed)
    beta_true = np.zeros(p)
    beta_true[:5] = [1.0, -0.5, 0.3, -0.2, 0.1]
    sigma_true = 1.0
    X = rng.standard_normal((n, p))
    y = X @ beta_true + rng.normal(0, sigma_true, size=n)
    return {"X": X, "y": y.tolist(), "beta_true": beta_true, "sigma_true": sigma_true, "n": n, "p": p}


def deep_hierarchy_data(n_groups=50, n_subgroups=10, n_per=10, seed=42):
    """Deep (3-level) hierarchy: ~n_groups*n_subgroups + n_groups + 4 parameters."""
    rng = np.random.default_rng(seed)
    mu_global = 0.0
    tau_group = 2.0
    tau_sub = 1.0
    sigma = 0.5
    group_means = rng.normal(mu_global, tau_group, size=n_groups)
    y, group_ids, subgroup_ids = [], [], []
    subgroup_means = []
    for g in range(n_groups):
        sub_means = rng.normal(group_means[g], tau_sub, size=n_subgroups)
        subgroup_means.extend(sub_means.tolist())
        for s in range(n_subgroups):
            obs = rng.normal(sub_means[s], sigma, size=n_per)
            y.extend(obs.tolist())
            group_ids.extend([g] * n_per)
            subgroup_ids.extend([g * n_subgroups + s] * n_per)
    return {
        "y": y, "group_ids": group_ids, "subgroup_ids": subgroup_ids,
        "n_groups": n_groups, "n_subgroups": n_subgroups, "n_per": n_per,
        "mu_true": mu_global, "tau_group_true": tau_group,
        "tau_sub_true": tau_sub, "sigma_true": sigma,
    }
