functions{
real q_loss(real q, vector u){
  return 0.5 * sum(abs(u) + (2 * q - 1) * u);
}

real ald_lpdf(vector y, real q, real sigma, vector q_est){
  return log(q) + log1m(q) - log(sigma) - q_loss(q, y - q_est) / sigma;
}
}
data {
  int N;                 // Number of observation
  int P;                 // Number of predictors
  real q;
  vector[N] y;           // Response variable sorted
  matrix[N, P] x;
}
parameters {
  vector[P] beta;
  real<lower=0> sigma;
}
model {
  beta ~ normal(0, 4);
  sigma ~ exponential(1);
  y ~ ald(q, sigma, x * beta);
}
