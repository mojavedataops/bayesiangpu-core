# Add a parameter to a model (pipe-friendly)

Add a parameter to a model (pipe-friendly)

## Usage

``` r
param(model, name, distribution, size = 1L)
```

## Arguments

- model:

  A BayesianModel object

- name:

  Parameter name

- distribution:

  Prior distribution

- size:

  Number of elements for vector parameters (defaults to 1)

## Value

The model (invisibly, for piping)
