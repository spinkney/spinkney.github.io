data {
  int N;                   // Number of observation
  int P;                   // Number of predictors
  real<lower=0, upper=1> q;
  vector[N] y;             // Response variable sorted
  matrix[N, P] x;
}
transformed data {
  real theta = (1 - 2 * q) / (q * (1 - q));
  real tau = sqrt(2 / (q * (1 - q)));
}
parameters {
  vector[P] beta;
  vector<lower=0>[N] z;
  real<lower=0> sigma;
}
model {
  beta ~ normal(0, 2);
  sigma ~ exponential(1);
  
  // Data Augmentation
  z ~ exponential(1);
  y ~ normal(x * beta + sigma * theta * z, tau * sqrt(z) * sigma);
}
