# Beta-Binomial Distribution

Create a Beta-Binomial distribution. A compound distribution where the
success probability follows a Beta distribution. Useful for
overdispersed binomial data.

## Usage

``` r
BetaBinomial(n, alpha, beta)
```

## Arguments

- n:

  Number of trials (positive integer)

- alpha:

  First shape parameter of the Beta prior (alpha \> 0)

- beta:

  Second shape parameter of the Beta prior (beta \> 0)

## Value

A bg_distribution object

## Examples

``` r
if (FALSE) { # \dontrun{
BetaBinomial(10, 2, 5)  # n=10, alpha=2, beta=5
} # }
```
