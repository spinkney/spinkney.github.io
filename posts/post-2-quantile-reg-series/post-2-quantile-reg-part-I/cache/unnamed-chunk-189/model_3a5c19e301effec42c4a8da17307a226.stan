functions{
real q_loss(real q, vector u){
  return 0.5 * sum(abs(u) + (2 * q - 1) * u);
}

real ald_lpdf(vector y, real q, real sigma, vector q_est){
  int N = num_elements(y);
 
  return N * (log(q) + log1m(q) - log(sigma)) - q_loss(q, y - q_est) / sigma;
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
  beta[1] ~ student_t( 3, 5, 2.5);
  sigma ~ exponential(1);
  y ~ ald(q, 1, x * beta);
}
generated quantities {
 matrix[P, P] L_hat = cholesky_decompose(add_diag(tcrossprod(diag_pre_multiply(rep_vector(sqrt(sigma), P), x' )), 1e-6));
 vector[P] asymtotic_beta = beta + L_hat \ rep_vector(1, P);
}
