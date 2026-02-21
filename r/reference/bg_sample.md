# Run NUTS Sampling

Run the No-U-Turn Sampler (NUTS) on a Bayesian model.

## Usage

``` r
bg_sample(
  model,
  num_samples = 1000L,
  num_warmup = 1000L,
  num_chains = 4L,
  target_accept = 0.8,
  seed = 42L
)
```

## Arguments

- model:

  A BayesianModel object

- num_samples:

  Number of samples to draw (after warmup)

- num_warmup:

  Number of warmup iterations

- num_chains:

  Number of parallel chains

- target_accept:

  Target acceptance probability (0-1)

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

result <- bg_sample(model, num_samples = 1000, num_chains = 4)
summary(result)
} # }
```
