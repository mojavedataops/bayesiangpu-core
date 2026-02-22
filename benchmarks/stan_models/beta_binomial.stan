data {
  int<lower=0> N;
  int<lower=0, upper=N> y;
}
parameters {
  real<lower=0, upper=1> theta;
}
model {
  theta ~ beta(1, 1);
  y ~ binomial(N, theta);
}
