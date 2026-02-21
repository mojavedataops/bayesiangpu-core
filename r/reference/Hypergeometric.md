# Hypergeometric Distribution

Create a Hypergeometric distribution. Models the number of successes in
draws without replacement from a finite population.

## Usage

``` r
Hypergeometric(big_n, big_k, n)
```

## Arguments

- big_n:

  Population size

- big_k:

  Number of success states in population

- n:

  Number of draws

## Value

A bg_distribution object

## Examples

``` r
if (FALSE) { # \dontrun{
Hypergeometric(50, 25, 10)  # N=50, K=25, n=10
} # }
```
