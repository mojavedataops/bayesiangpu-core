# Inference Result

An R6 class containing MCMC sampling results and diagnostics.

## Fields

- `samples`:

  Named list of sample vectors for each parameter

- `chains`:

  Named list of chain samples (list of vectors)

- `rhat`:

  Named list of R-hat values

- `ess`:

  Named list of effective sample size values

- `divergences`:

  Number of divergent transitions

- `step_size`:

  Final adapted step size

- `num_samples`:

  Number of samples per chain

- `num_warmup`:

  Number of warmup iterations

- `num_chains`:

  Number of chains

- `param_names`:

  Names of parameters

## Methods

- `initialize(result_list)`:

  Create a new InferenceResult from Rust output

- `summarize(param_name)`:

  Get summary statistics for a parameter

- [`summary()`](https://rdrr.io/r/base/summary.html):

  Get summary for all parameters as a data frame

- `format_summary()`:

  Format summary as a string

- [`warnings()`](https://rdrr.io/r/base/warnings.html):

  Check for sampling warnings

- `is_converged()`:

  Check if sampling converged

- `as_tibble()`:

  Convert to a tibble

- [`print()`](https://rdrr.io/r/base/print.html):

  Print the result
