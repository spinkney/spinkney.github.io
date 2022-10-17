functions {
 real inv_gaussian_lpdf (vector y, vector mu, real lambda) {
    if (lambda <= 0) 
      reject("lambda must be greater than 0 found lambda = ", lambda);
    
    return sum(0.5 * (log(lambda) - log(2 * pi()) - 3 *log(y)) - lambda * ((y - mu)^2 ./ (2 * mu^2 .* y)));
  }
}

data {
  int N;                   // Number of observation
  int P;                   // Number of predictors
  real<lower=0, upper=1> q;
  vector[N] y;             // Response variable sorted
  matrix[N, P] x;
}
parameters {
  vector[P] beta_raw;
  real<lower=0> sigma_inv;
  vector<lower=0>[N] v_inv;
}
transformed parameters {
  vector[P] beta;
  real xi = 1 - 2 * q;
  
  {
    vector[N] v = 1 ./ v_inv;
    vector[N] V = 0.5 * v_inv * sigma_inv ;
    matrix[P, P] varcov = chol2inv( cholesky_decompose(add_diag( crossprod(diag_pre_multiply(sqrt(V), x)), 1e-6 )));
    vector[P] betam = varcov * transpose(x) * diag_matrix(V) * (y - xi * v);
    beta = betam + cholesky_decompose(varcov) * beta_raw;
  }
}
model {
  real zeta = q * (1 - q);
  vector[N] mu = 1 ./ abs(y - x * beta);
  vector[N] Mu = x * beta + xi ./ v_inv;
  
  v_inv ~ inv_gaussian(mu, 0.5 * sigma_inv);
  sigma_inv ~ inv_gamma(3.0 / 2 * N, 1 / (dot_product(0.25 * v_inv,  (y - Mu)^2) + zeta / sum(v_inv)));
  
  // Priors
  beta_raw  ~ std_normal();
  
  // Data Augmentation
}
