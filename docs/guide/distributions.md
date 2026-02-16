# Distributions

BayesianGPU supports 10 probability distributions for priors and likelihoods.

## Continuous Distributions

### Normal

Gaussian distribution for real-valued parameters.

```javascript
import { Normal } from 'bayesiangpu';

Normal(loc, scale)  // mean, std deviation

// Examples
Normal(0, 1)        // Standard normal
Normal(100, 15)     // IQ distribution
Normal('mu', 'sigma')  // Reference other parameters
```

### HalfNormal

Positive half of a normal distribution. Useful for scale parameters.

```javascript
import { HalfNormal } from 'bayesiangpu';

HalfNormal(scale)

// Example: prior for standard deviation
HalfNormal(10)
```

### Beta

Distribution on [0, 1]. Perfect for probability parameters.

```javascript
import { Beta } from 'bayesiangpu';

Beta(alpha, beta)

// Examples
Beta(1, 1)   // Uniform on [0,1]
Beta(2, 2)   // Symmetric, mode at 0.5
Beta(2, 8)   // Skewed toward 0
Beta(8, 2)   // Skewed toward 1
```

### Gamma

Positive-valued distribution. Common for rate/scale parameters.

```javascript
import { Gamma } from 'bayesiangpu';

Gamma(shape, rate)

// Examples
Gamma(1, 1)    // Exponential
Gamma(2, 0.5)  // Mean = 4, variance = 8
```

### Exponential

Special case of Gamma. Useful for waiting times.

```javascript
import { Exponential } from 'bayesiangpu';

Exponential(rate)

// Example: prior for rate parameter
Exponential(1)  // Mean = 1
```

### Uniform

Flat distribution over an interval.

```javascript
import { Uniform } from 'bayesiangpu';

Uniform(low, high)

// Example
Uniform(0, 10)  // Uniform on [0, 10]
```

### LogNormal

Positive distribution with log-normal shape.

```javascript
import { LogNormal } from 'bayesiangpu';

LogNormal(loc, scale)  // Parameters of the log

// Example
LogNormal(0, 1)  // Median = 1
```

### StudentT

Heavy-tailed distribution. Robust to outliers.

```javascript
import { StudentT } from 'bayesiangpu';

StudentT(df, loc, scale)

// Examples
StudentT(3, 0, 1)   // t-distribution with 3 df
StudentT(30, 0, 1)  // Approximately normal
```

### Cauchy

Very heavy tails. Useful for weakly informative priors.

```javascript
import { Cauchy } from 'bayesiangpu';

Cauchy(loc, scale)

// Example: weakly informative prior
Cauchy(0, 2.5)
```

## Discrete Distributions

### Bernoulli

Binary outcomes (0 or 1).

```javascript
import { Bernoulli } from 'bayesiangpu';

Bernoulli(p)

// Example as likelihood
.observe(Bernoulli('theta'), [1, 0, 1, 1, 0])
```

### Binomial

Count of successes in n trials.

```javascript
import { Binomial } from 'bayesiangpu';

Binomial(n, p)

// Example
.observe(Binomial(100, 'theta'), [65])  // 65 successes in 100 trials
```

### Poisson

Count data with no upper bound.

```javascript
import { Poisson } from 'bayesiangpu';

Poisson(rate)

// Example: count model
.observe(Poisson('lambda'), [3, 4, 5, 3, 4])
```

## Parameter References

Any parameter can reference another by name:

```javascript
const model = new Model()
  .param('mu', Normal(0, 10))
  .param('sigma', HalfNormal(1))
  .observe(Normal('mu', 'sigma'), data)  // References mu and sigma
  .build();
```
