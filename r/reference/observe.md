# Set observed data on a model (pipe-friendly)

Set observed data on a model (pipe-friendly)

## Usage

``` r
observe(model, distribution, data, known = NULL)
```

## Arguments

- model:

  A BayesianModel object

- distribution:

  Likelihood distribution

- data:

  Observed data points

- known:

  Named list of per-observation known data

## Value

The model (invisibly, for piping)
