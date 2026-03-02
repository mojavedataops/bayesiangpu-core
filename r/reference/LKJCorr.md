# LKJ Correlation Distribution

Create an LKJ prior for correlation matrices, parameterized by
concentration parameter eta. The LKJ distribution (Lewandowski,
Kurowicka, and Joe, 2009) is the standard prior for correlation matrices
in Bayesian hierarchical models.

## Usage

``` r
LKJCorr(dim, eta = 1)
```

## Arguments

- dim:

  Integer. Dimension of the correlation matrix (\>= 2).

- eta:

  Numeric. Concentration parameter (\> 0). eta = 1: uniform over valid
  correlation matrices. eta \> 1: concentrates toward identity (less
  correlation). eta \< 1: concentrates toward extreme correlations.

## Value

A distribution specification

## Examples

``` r
if (FALSE) { # \dontrun{
LKJCorr(3, 2.0)  # 3x3 correlation matrix, slightly favoring identity
LKJCorr(4, 1.0)  # 4x4, uniform over correlation matrices
} # }
```
