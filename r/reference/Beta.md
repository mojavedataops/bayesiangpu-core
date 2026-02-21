# Beta Distribution

Create a Beta distribution. Defined on (0, 1), useful for probability
parameters.

## Usage

``` r
Beta(alpha, beta)
```

## Arguments

- alpha:

  Shape parameter alpha (concentration1)

- beta:

  Shape parameter beta (concentration0)

## Value

A distribution specification

## Examples

``` r
if (FALSE) { # \dontrun{
Beta(1, 1)   # Uniform on [0, 1]
Beta(2, 5)   # Skewed toward 0
} # }
```
