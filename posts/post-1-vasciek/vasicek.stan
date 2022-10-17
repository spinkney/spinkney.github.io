functions {
  real vasicek_quantile_lpdf(vector y, real theta, vector mu, real tau) {
    if (tau <= 0 || tau >= 1) 
      reject("tau must be between 0 and 1 found tau = ", tau);
    
    if (theta <= 0 || theta >= 1) 
      reject("theta must be between 0 and 1 found theta = ", theta);
    
    int N = num_elements(y);
    real lpdf = 0.5 * N * (log1m(theta) - log(theta));
    vector[N] qnorm_mu = inv_Phi(mu);
    real qnorm_tau = inv_Phi(tau);
    vector[N] qnorm_y = inv_Phi(y);
    vector[N] qnorm_alpha = -sqrt(theta) * qnorm_tau + qnorm_mu * sqrt(1 - theta);
    
    return lpdf + 0.5 * dot_self(qnorm_y) - 0.5 * sum( (sqrt(1 - theta) * qnorm_y - qnorm_alpha)^2 / theta);
  }
}
data {
  int<lower=0> N;
  vector[N] y;
  vector[N] x;
  real<lower=0, upper=1> tau;
} 
parameters {
  real alpha;
  real beta;
  real sigma;
} 
model {
  alpha ~ normal(0, 4);
  beta ~ normal(0, 4);
  sigma ~ normal(0, 2);
  y ~ vasicek_quantile(inv_logit(sigma), inv_logit(alpha + beta * x), tau);
}
