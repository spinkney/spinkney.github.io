data {
  int N;                   // Number of observation
  int P;                   // Number of predictors
  real<lower=0, upper=1> q;
  vector[N] y;             // Response variable sorted
  matrix[N, P] x;
}
parameters {
  vector[P] beta;
  real<lower=0> sigma;
  vector<lower=0>[N] w;
}
transformed parameters {
  real<lower=0> tau = pow(sigma, 2);
}
model {
  vector[N] me = (1 - 2 * q) / (q * (1 - q)) * w;
  vector[N] pe = 2 * w / (q * (1 - q) * tau);
  vector[N] pe2 = sqrt(pe);
  // Priors
  beta  ~ normal(0, 4);
  sigma ~ exponential(1);

  // Data Augmentation
  w ~ exponential(tau);

  // The model
  y ~ normal(me + x * beta, pe2);
}
