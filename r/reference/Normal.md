# Normal Distribution

Create a Normal (Gaussian) distribution specification.

## Usage

``` r
Normal(loc, scale)
```

## Arguments

- loc:

  Mean (can be a number or parameter name as string)

- scale:

  Standard deviation (can be a number or parameter name)

## Value

A distribution specification (JSON string internally)

## Examples

``` r
if (FALSE) { # \dontrun{
Normal(0, 1)           # Standard normal
Normal("mu", "sigma")  # Parameters from model
} # }
```
