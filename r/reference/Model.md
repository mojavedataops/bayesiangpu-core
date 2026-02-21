# Bayesian Model Builder

An R6 class for building Bayesian models using a fluent API.

## Methods

- `initialize()`:

  Create a new empty model

- `param(name, distribution, size = 1L)`:

  Add a parameter with a prior distribution

- `observe(distribution, data, known = NULL)`:

  Set the likelihood (observed data) for the model

- `param_names()`:

  Get the list of parameter names

- `num_params()`:

  Get the number of parameters

- `has_likelihood()`:

  Check if the model has a likelihood

- `to_json()`:

  Convert model to JSON string (for internal use)

- [`print()`](https://rdrr.io/r/base/print.html):

  Print the model

## Examples

``` r
if (FALSE) { # \dontrun{
model <- Model()
model$param("mu", Normal(0, 10))
model$param("sigma", HalfNormal(1))
model$observe(Normal("mu", "sigma"), c(2.3, 2.1, 2.5))

# Or using pipe syntax
model <- Model() |>
  param("theta", Beta(1, 1)) |>
  observe(Binomial(100, "theta"), 65)
} # }
```
