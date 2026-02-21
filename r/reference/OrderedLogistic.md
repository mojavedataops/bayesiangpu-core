# Ordered Logistic Distribution

Create an Ordered Logistic (proportional odds) distribution. Models
ordinal response data with ordered categories.

## Usage

``` r
OrderedLogistic(eta, cutpoints)
```

## Arguments

- eta:

  Linear predictor

- cutpoints:

  Ordered vector of cutpoints

## Value

A bg_distribution object

## Examples

``` r
if (FALSE) { # \dontrun{
OrderedLogistic(0, c(-1, 0, 1))  # 4 categories with cutpoints at -1, 0, 1
} # }
```
