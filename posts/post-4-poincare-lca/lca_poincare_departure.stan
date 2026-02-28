// lca_poincare_departure.stan
// LCA with Geometric Surface + Regularized Horseshoe Departures
// =============================================================
//
// logit_theta[k,j] = geometric_logit[k,j] + sigma_delta[k,j]
//
// geometric_logit[k,j] = alpha[j] + lam[j] * r[k] * cos(angle[k] - phi[j])
// with fixed phi[j] on [0, pi/2] as in the base model.
//
// Regularized horseshoe prior for departures:
//   sigma_delta[k,j] = z_delta[k,j] * hs_tau * hs_lambda_tilde[k,j]
//   hs_lambda_tilde = sqrt(c2 * lambda^2 / (c2 + hs_tau^2 * lambda^2))
//   z_delta ~ normal(0,1), lambda ~ half-Student-t, hs_tau ~ half-normal,
//   c2 = slab_scale^2 * hs_c2_aux, hs_c2_aux ~ inv_gamma(nu/2, nu/2)
//
// This strongly shrinks most departures while preserving large effects via
// local scales and a finite slab.

data {
  int<lower=1> P;
  int<lower=2> K;
  int<lower=2> J;
  array[P, J] int<lower=0, upper=1> pattern;
  array[P] int<lower=1> count;
}

transformed data {
  vector[J] phi;
  for (j in 1:J)
    phi[j] = (pi() / 2) * (j - 1.0) / (J - 1.0);

  matrix[P, J] Y;
  for (p in 1:P)
    for (j in 1:J)
      Y[p, j] = pattern[p, j];

  row_vector[J] rv = rep_row_vector(1.0, J);

  // Regularized horseshoe hyperparameters.
  // Using mildly lighter tails than half-Cauchy improves mixing
  // in small-N, higher-K settings while preserving sparse behavior.
  real hs_slab_df = 7.0;
  real hs_slab_scale = 0.50;
  real hs_global_scale = 0.10;
}

parameters {
  ordered[K] logit_r;
  vector[K] angle_raw;
  vector[J] alpha;
  vector<lower=0>[J] lam;

  matrix[K, J] z_delta;
  matrix<lower=0>[K, J] hs_lambda;
  real<lower=0> hs_tau;
  real<lower=0> hs_c2_aux;

  simplex[K] pi_w;
}

transformed parameters {
  vector<lower=0, upper=1>[K] radii;
  vector<lower=0>[K] angles;
  matrix[K, J] geometric_logit;
  matrix[K, J] raw_sigma_delta;
  matrix[K, J] sigma_delta;
  matrix[K, J] logit_theta;
  real hs_c2 = square(hs_slab_scale) * hs_c2_aux;
  real hs_tau2 = square(hs_tau);

  for (k in 1:K) {
    radii[k] = inv_logit(logit_r[k]);
    angles[k] = (pi() / 2) * inv_logit(angle_raw[k]);
  }

  for (k in 1:K)
    for (j in 1:J) {
      real lambda2 = square(hs_lambda[k, j]);
      real lambda_tilde = sqrt(hs_c2 * lambda2 / (hs_c2 + hs_tau2 * lambda2 + 1e-12));
      raw_sigma_delta[k, j] = z_delta[k, j] * hs_tau * lambda_tilde;
      geometric_logit[k, j] = alpha[j] + lam[j] * radii[k] * cos(angles[k] - phi[j]);
    }

  // Item-wise centering removes location confounding with alpha[j].
  for (j in 1:J) {
    real delta_mean_j = mean(raw_sigma_delta[, j]);
    for (k in 1:K) {
      sigma_delta[k, j] = raw_sigma_delta[k, j] - delta_mean_j;
      logit_theta[k, j] = geometric_logit[k, j] + sigma_delta[k, j];
    }
  }
}

model {
  // Match geometric priors from lca_poincare.stan.
  // The horseshoe block below is the only additional regularization.
  logit_r ~ normal(0, 2);
  angle_raw ~ normal(0, 2);
  alpha ~ normal(0, 3);
  lam ~ normal(0, 3);  // half-normal via <lower=0>

  to_vector(z_delta) ~ std_normal();
  to_vector(hs_lambda) ~ student_t(5, 0, 0.75);
  hs_tau ~ normal(0, hs_global_scale);
  hs_c2_aux ~ inv_gamma(0.5 * hs_slab_df, 0.5 * hs_slab_df);

  pi_w ~ dirichlet(rep_vector(2.0, K));

  {
    row_vector[K] sum_log1p = rv * log1p_exp(logit_theta');
    vector[K] c = -sum_log1p';
    vector[K] log_nu = log(pi_w);
    matrix[P, K] S = Y * logit_theta';

    for (p in 1:P)
      target += count[p] * log_sum_exp(log_nu + (S[p]' + c));
  }
}

generated quantities {
  vector[P] log_lik;
  matrix[K, J] theta;
  matrix[K, J] geometric_theta;
  matrix[K, J] delta;
  real min_profile_dist;
  matrix[K, K] hyp_dist;
  real min_hyp_dist;

  for (k in 1:K)
    for (j in 1:J) {
      theta[k, j] = inv_logit(logit_theta[k, j]);
      geometric_theta[k, j] = inv_logit(geometric_logit[k, j]);
      delta[k, j] = sigma_delta[k, j];
    }

  {
    row_vector[K] sum_log1p = rv * log1p_exp(logit_theta');
    vector[K] c = -sum_log1p';
    vector[K] log_nu = log(pi_w);
    matrix[P, K] S = Y * logit_theta';

    for (p in 1:P)
      log_lik[p] = log_sum_exp(log_nu + (S[p]' + c));
  }

  min_profile_dist = positive_infinity();
  for (a in 1:K)
    for (b in (a + 1):K) {
      real d = 0;
      for (j in 1:J)
        d += square(theta[a, j] - theta[b, j]);
      if (sqrt(d) < min_profile_dist)
        min_profile_dist = sqrt(d);
    }

  for (a in 1:K) {
    hyp_dist[a, a] = 0;
    for (b in (a + 1):K) {
      real xa = radii[a] * cos(angles[a]);
      real ya = radii[a] * sin(angles[a]);
      real xb = radii[b] * cos(angles[b]);
      real yb = radii[b] * sin(angles[b]);
      real num = sqrt(square(xa - xb) + square(ya - yb));
      real re = 1 - (xa * xb + ya * yb);
      real im = -(xa * yb - ya * xb);
      real den = sqrt(square(re) + square(im));
      hyp_dist[a, b] = 2 * atanh(fmin(num / den, 0.999));
      hyp_dist[b, a] = hyp_dist[a, b];
    }
  }

  min_hyp_dist = positive_infinity();
  for (a in 1:K)
    for (b in (a + 1):K)
      if (hyp_dist[a, b] < min_hyp_dist)
        min_hyp_dist = hyp_dist[a, b];
}
