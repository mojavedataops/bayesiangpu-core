# Dirichlet Distribution

Create a Dirichlet distribution. Prior over probability simplexes
(vectors that sum to 1). Essential for topic modeling (LDA) and
categorical data modeling.

## Usage

``` r
Dirichlet(alpha)
```

## Arguments

- alpha:

  Concentration vector (numeric vector of positive values)

## Value

A distribution specification

## Examples

``` r
if (FALSE) { # \dontrun{
Dirichlet(c(1, 1, 1))       # Uniform over 3-simplex
Dirichlet(c(10, 10, 10))    # Concentrated at center
Dirichlet(c(0.1, 0.1, 0.1)) # Concentrated at corners
} # }
```
