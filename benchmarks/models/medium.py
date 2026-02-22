"""Medium benchmark models (10-100 parameters)."""
from .registry import register
from ..data.generators import (logistic_regression_data, hierarchical_intercepts_data, eight_schools_data)

register("logistic_regression", num_params=50, data_generator=logistic_regression_data,
         description="Logistic regression with 50 predictors", tags=["medium", "regression"])

register("hierarchical_intercepts", num_params=22, data_generator=hierarchical_intercepts_data,
         description="Hierarchical intercepts model (~22 parameters)", tags=["medium", "hierarchical"])

register("eight_schools", num_params=10, data_generator=eight_schools_data,
         description="Classic Eight Schools (non-centered parameterization)", tags=["medium", "hierarchical"])
