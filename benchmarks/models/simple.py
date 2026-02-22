"""Simple benchmark models (1-11 parameters)."""
from .registry import register
from ..data.generators import beta_binomial_data, normal_mean_data, linear_regression_data

register("beta_binomial", num_params=1, data_generator=beta_binomial_data,
         description="Beta-Binomial conjugate model (1 parameter)", tags=["simple"])

register("normal_mean", num_params=2, data_generator=normal_mean_data,
         description="Normal mean estimation with unknown variance (2 parameters)", tags=["simple"])

register("linear_regression", num_params=11, data_generator=linear_regression_data,
         description="Linear regression with 10 predictors (11 parameters)", tags=["simple", "regression"])
