# Truncated Normal Distribution

Create a Truncated Normal distribution. A Normal distribution
constrained to the interval \[low, high\].

## Usage

``` r
TruncatedNormal(loc, scale, low, high)
```

## Arguments

- loc:

  Location parameter (mean of underlying normal)

- scale:

  Scale parameter (std dev of underlying normal, must be positive)

- low:

  Lower bound

- high:

  Upper bound

## Value

A distribution specification

## Examples

``` r
if (FALSE) { # \dontrun{
TruncatedNormal(0, 1, -2, 2)  # Normal(0,1) truncated to [-2, 2]
} # }
```
