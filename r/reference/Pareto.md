# Pareto Distribution

Create a Pareto distribution. Used for modeling heavy-tailed phenomena
such as wealth distribution and city sizes.

## Usage

``` r
Pareto(alpha, x_m)
```

## Arguments

- alpha:

  Shape parameter (alpha \> 0)

- x_m:

  Scale parameter / minimum value (x_m \> 0)

## Value

A bg_distribution object

## Examples

``` r
if (FALSE) { # \dontrun{
Pareto(2, 1)  # alpha=2, x_m=1
} # }
```
