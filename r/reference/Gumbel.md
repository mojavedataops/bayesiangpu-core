# Gumbel Distribution

Create a Gumbel distribution. Used for modeling extreme values (maximum
or minimum of samples).

## Usage

``` r
Gumbel(loc = 0, scale = 1)
```

## Arguments

- loc:

  Location parameter (default: 0)

- scale:

  Scale parameter (default: 1, must be positive)

## Value

A bg_distribution object

## Examples

``` r
if (FALSE) { # \dontrun{
Gumbel()       # Standard Gumbel
Gumbel(0, 2)   # Wider Gumbel
} # }
```
