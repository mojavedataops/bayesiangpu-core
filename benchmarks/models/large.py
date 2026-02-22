"""Large benchmark models (500+ parameters)."""
from .registry import register
from ..data.generators import wide_regression_data, deep_hierarchy_data

register("wide_regression", num_params=1001, data_generator=wide_regression_data,
         description="Wide regression with 1000 predictors", tags=["large", "regression"])

register("deep_hierarchy", num_params=504, data_generator=deep_hierarchy_data,
         description="3-level hierarchical model (~504 parameters)", tags=["large", "hierarchical"])
