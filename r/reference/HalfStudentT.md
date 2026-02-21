# Half Student's t Distribution

Create a Half Student's t distribution (positive values only). A robust
alternative to HalfNormal for scale parameters, with heavier tails
controlled by the degrees of freedom.

## Usage

``` r
HalfStudentT(df, scale = 1)
```

## Arguments

- df:

  Degrees of freedom (df \> 0)

- scale:

  Scale parameter (default: 1, must be positive)

## Value

A bg_distribution object

## Examples

``` r
if (FALSE) { # \dontrun{
HalfStudentT(3)      # 3 df, scale=1
HalfStudentT(5, 2)   # 5 df, scale=2
} # }
```
