# Student's t Distribution

Create a Student's t distribution. For robust inference with potential
outliers.

## Usage

``` r
StudentT(df, loc = 0, scale = 1)
```

## Arguments

- df:

  Degrees of freedom

- loc:

  Location parameter (default: 0)

- scale:

  Scale parameter (default: 1)

## Value

A distribution specification

## Examples

``` r
if (FALSE) { # \dontrun{
StudentT(3)         # t with 3 df
StudentT(5, 0, 2)   # t with 5 df, loc=0, scale=2
} # }
```
