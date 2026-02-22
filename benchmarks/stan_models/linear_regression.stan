data {
  int<lower=0> N;
  int<lower=0> P;
  matrix[N, P] X;
  array[N] real y;
}
parameters {
  vector[P] beta;
  real<lower=0> sigma;
}
model {
  beta ~ normal(0, 1);
  sigma ~ normal(0, 2);
  y ~ normal(X * beta, sigma);
}
