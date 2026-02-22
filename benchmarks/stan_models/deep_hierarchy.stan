data {
  int<lower=0> N;
  int<lower=0> J;
  int<lower=0> K;
  array[N] int<lower=1, upper=J> group;
  array[N] int<lower=1, upper=J * K> subgroup;
  array[N] real y;
}
parameters {
  real mu;
  real<lower=0> tau_group;
  array[J] real group_means;
  real<lower=0> tau_sub;
  array[J * K] real sub_means;
  real<lower=0> sigma;
}
model {
  mu ~ normal(0, 10);
  tau_group ~ normal(0, 5);
  group_means ~ normal(mu, tau_group);
  tau_sub ~ normal(0, 5);
  for (jk in 1:(J * K))
    sub_means[jk] ~ normal(group_means[((jk - 1) / K) + 1], tau_sub);
  sigma ~ normal(0, 5);
  for (n in 1:N)
    y[n] ~ normal(sub_means[subgroup[n]], sigma);
}
