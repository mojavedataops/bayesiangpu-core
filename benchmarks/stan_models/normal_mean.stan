data {
  int<lower=0> N;
  array[N] real y;
}
parameters {
  real mu;
  real<lower=0> sigma;
}
model {
  mu ~ normal(0, 10);
  sigma ~ normal(0, 5);
  y ~ normal(mu, sigma);
}
