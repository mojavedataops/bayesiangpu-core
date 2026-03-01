"""GPU-scale benchmark models (1M-10M observations)."""
from functools import partial
from .registry import register
from ..data.generators import (
    normal_mean_data, linear_regression_data,
    gamma_regression_data, beta_regression_data,
)

register("big_normal_mean", num_params=2,
         data_generator=partial(normal_mean_data, n=1_000_000),
         description="Normal mean estimation with 1M observations",
         tags=["gpu", "large-n"])

register("big_linear_regression", num_params=11,
         data_generator=partial(linear_regression_data, n=1_000_000, p=10),
         description="Linear regression (10 predictors) with 1M observations",
         tags=["gpu", "large-n", "regression"])

register("huge_normal_mean", num_params=2,
         data_generator=partial(normal_mean_data, n=10_000_000),
         description="Normal mean estimation with 10M observations",
         tags=["gpu", "large-n"])

register("big_gamma_regression", num_params=3,
         data_generator=partial(gamma_regression_data, n=1_000_000),
         description="Gamma regression with 1M observations",
         tags=["gpu", "large-n"])

register("big_beta_regression", num_params=3,
         data_generator=partial(beta_regression_data, n=1_000_000),
         description="Beta regression with 1M observations",
         tags=["gpu", "large-n"])
