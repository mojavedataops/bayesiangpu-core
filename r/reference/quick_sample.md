# Quick Sampling

Run NUTS sampling with fewer iterations (good for testing).

## Usage

``` r
quick_sample(model, seed = 42L)
```

## Arguments

- model:

  A BayesianModel object

- seed:

  Random seed for reproducibility

## Value

An InferenceResult object

## Examples

``` r
if (FALSE) { # \dontrun{
model <- Model() |>
  param("theta", Beta(1, 1)) |>
  observe(Binomial(100, "theta"), 65)

result <- quick_sample(model)
} # }
```
