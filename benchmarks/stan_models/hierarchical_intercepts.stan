data {
  int<lower=0> N;
  int<lower=0> J;
  array[N] int<lower=1, upper=J> group;
  array[N] real y;
}
parameters {
  real mu;
  real<lower=0> tau;
  array[J] real theta;
  real<lower=0> sigma;
}
model {
  mu ~ normal(0, 10);
  tau ~ normal(0, 5);
  theta ~ normal(mu, tau);
  sigma ~ normal(0, 5);
  for (n in 1:N)
    y[n] ~ normal(theta[group[n]], sigma);
}
