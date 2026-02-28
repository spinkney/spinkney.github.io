// lca_poincare.stan
// LCA with Restricted Cosine Poincaré Disk + BLAS matmul likelihood
// =================================================================
//
// theta[k,j] = inv_logit(alpha[j] + lam[j] * r[k] * cos(angle[k] - phi[j]))
//
// Items spread evenly at phi[j] = (π/2)(j-1)/(J-1) in [0, π/2].
// Classes identified by ordered radii.
//
// Likelihood uses BLAS matmul: S = Y * η' avoids inner loop over items.

data {
  int<lower=1> P;                              // unique response patterns
  int<lower=2> K;                              // latent classes
  int<lower=2> J;                              // items (≥2 for phi spacing)
  array[P, J] int<lower=0, upper=1> pattern;  // unique binary patterns
  array[P] int<lower=1> count;                 // count per pattern
}

transformed data {
  int N = sum(count);
  vector[J] phi;
  for (j in 1:J)
    phi[j] = (pi() / 2) * (j - 1.0) / (J - 1.0);

  // Real matrix version of pattern for BLAS matmul
  matrix[P, J] Y;
  for (p in 1:P)
    for (j in 1:J)
      Y[p, j] = pattern[p, j];

  row_vector[J] rv = rep_row_vector(1.0, J);
}

parameters {
  ordered[K] logit_r;           // ordered logit-radii → identifiability
  vector[K] angle_raw;          // mapped to (0, pi/2) via scaled logistic
  vector[J] alpha;              // item baselines
  vector<lower=0>[J] lam;      // item discriminations
  simplex[K] pi_w;             // mixing weights
}

transformed parameters {
  vector<lower=0, upper=1>[K] radii;
  vector<lower=0>[K] angles;
  matrix[K, J] logit_theta;    // η[k,j] = alpha[j] + lam[j] * r[k] * cos(a[k] - phi[j])
  // matrix[K, J] theta;

  for (k in 1:K) {
    radii[k] = inv_logit(logit_r[k]);
    angles[k] = (pi() / 2) * inv_logit(angle_raw[k]);
  }

  for (k in 1:K)
    for (j in 1:J) {
      logit_theta[k, j] = alpha[j] + lam[j] * radii[k] * cos(angles[k] - phi[j]);
     // theta[k, j] = inv_logit(logit_theta[k, j]);
    }
}

model {
  // Priors
  logit_r ~ normal(0, 2);
  angle_raw ~ normal(0, 2);
  alpha ~ normal(0, 3);
  lam ~ normal(0, 3);           // half-normal via <lower=0>
  pi_w ~ dirichlet(rep_vector(2.0, K));

  // --- fast marginalized mixture likelihood ---
  {
    // For each class k:
    //   Σ_j log p(y_j | η_j,k)
    // = (y^T η_:,k) - Σ_j log(1+exp η_j,k)
    row_vector[K] sum_log1p = rv * log1p_exp(logit_theta'); // length K
    vector[K] c = -sum_log1p';
    vector[K] log_nu = log(pi_w);

    // S[p,k] = Σ_j Y[p,j] * η[k,j]  (BLAS matmul)
    matrix[P, K] S = Y * logit_theta';

    for (p in 1:P)
      target += count[p] * log_sum_exp(log_nu + (S[p]' + c));
  }

  // --- sufficient-statistics likelihood (loop version) ---
  // {
  //   matrix[K, J] lt = log(theta);
  //   matrix[K, J] lt1m = log1m(theta);
  //
  //   for (p in 1:P) {
  //     vector[K] lps;
  //     for (k in 1:K) {
  //       lps[k] = log(pi_w[k]);
  //       for (j in 1:J)
  //         lps[k] += pattern[p, j] == 1 ? lt[k, j] : lt1m[k, j];
  //     }
  //     target += count[p] * log_sum_exp(lps);
  //   }
  // }
}

generated quantities {
  vector[P] log_lik;
  matrix[K, J] theta_free;     // relaxed (unconstrained) item probs
  real min_profile_dist;
  matrix[K, K] hyp_dist;
  real min_hyp_dist;

  // Per-pattern log-lik (BLAS version) + responsibilities for theta_free
  {
    row_vector[K] sum_log1p = rv * log1p_exp(logit_theta');
    vector[K] c = -sum_log1p';
    vector[K] log_nu = log(pi_w);
    matrix[P, K] S = Y * logit_theta';

    // log-lik
    for (p in 1:P)
      log_lik[p] = log_sum_exp(log_nu + (S[p]' + c));

    // Relaxed theta via iterated EM (seeded from geometric model)
    // Initial E-step uses geometric logit_theta; then iterate M→E→M...
    {
      // Seed theta_free from geometric model
      matrix[K, J] eta = logit_theta;

      for (em_iter in 1:200) {
        // E-step: responsibilities from current eta
        matrix[P, K] W;
        {
          row_vector[K] slp = rv * log1p_exp(eta');
          vector[K] cc = -slp';
          matrix[P, K] SS = Y * eta';

          for (p in 1:P) {
            vector[K] lp = log_nu + SS[p]' + cc;
            real lse = log_sum_exp(lp);
            for (k in 1:K)
              W[p, k] = count[p] * exp(lp[k] - lse);
          }
        }

        // M-step: theta_free = weighted sample means
        matrix[K, J] numer = W' * Y;
        for (k in 1:K) {
          real denom = sum(W[, k]);
          for (j in 1:J) {
            theta_free[k, j] = numer[k, j] / denom;
            // Update eta for next E-step (logit scale, clamp to avoid ±inf)
            eta[k, j] = log(fmax(theta_free[k, j], 1e-8))
                       - log(fmax(1 - theta_free[k, j], 1e-8));
          }
        }
      }
    }
  }

  // Min pairwise profile distance
  min_profile_dist = positive_infinity();
  for (a in 1:K)
    for (b in (a + 1):K) {
      real d = 0;
      for (j in 1:J)
        d += square(inv_logit(logit_theta[a, j]) - inv_logit(logit_theta[b, j]));
      if (sqrt(d) < min_profile_dist)
        min_profile_dist = sqrt(d);
    }

  // Pairwise hyperbolic distances on Poincaré disk
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
