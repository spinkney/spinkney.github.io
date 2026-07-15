functions {
  int is_zero(int row_idx, int col_idx, array[,] int zeros, int zero_idx) {
    return zero_idx <= size(zeros)
      && zeros[zero_idx, 1] == row_idx
      && zeros[zero_idx, 2] == col_idx;
  }

  matrix cholesky_corr_constrain_ldl_jacobian(int K, vector raw,
                                              array[,] int zeros) {
    // Omega = L * diag(D) * L' with unit lower-triangular L;
    // the Cholesky factor is diag_post_multiply(L, sqrt(D))
    matrix[K, K] L = diag_matrix(rep_vector(1, K));
    vector[K] D;
    vector[K] log_D;
    int raw_idx = 1;
    int zero_idx = 1;
    D[1] = 1;
    log_D[1] = 0;

    for (i in 2:K) {
      // log remaining radius^2 of row i: log(1 - sum_{k<j} L[i,k]^2 * D[k])
      real log_r = 0;
      for (j in 1:(i - 1)) {
        real b1 = dot_product(D[1:(j - 1)]' .* L[j, 1:(j - 1)],
                              L[i, 1:(j - 1)]);
        // Omega[i, j] = b1 + L[i, j] * D[j] and the correlation C satisfies
        // |C - b1| <= sqrt(D[j] * r), intersected with (-1, 1)
        real x;
        if (is_zero(i, j, zeros, zero_idx)) {
          x = -b1;  // Omega[i, j] = b1 + Lij * Dj = 0
          zero_idx += 1;
        } else {
          real b2 = exp(0.5 * (log_r + log_D[j]));
          real low = max({-b2, -1 - b1});
          real up = min({b2, 1 - b1});
          x = lower_upper_bound_jacobian(raw[raw_idx], low, up);
          raw_idx += 1;
        }
        L[i, j] = x / D[j];
        // free entries: change of variables x -> L_chol[i, j] = x / sqrt(D[j])
        // zero entries: Carpenter's correction -log(L_chol[j, j])
        jacobian += -0.5 * log_D[j];
        log_r = log_diff_exp(log_r, log_D[j] + 2 * log(abs(L[i, j])));
      }
      log_D[i] = log_r;
      D[i] = exp(log_r);
    }
    return diag_post_multiply(L, sqrt(D));
  }
}
data {
  int<lower=2> K;                             // dimension of correlation matrix
  real<lower=0> eta;                          // concentration in LKJ Cholesky
  int<lower=0, upper=choose(K, 2)> N_zero;    // # structural zero correlations
  array[N_zero, 2] int zeros;                 // lower triangular, row major order
}
parameters {
  vector[choose(K, 2) - N_zero] raw;
}
transformed parameters {
  matrix[K, K] L_Omega = cholesky_corr_constrain_ldl_jacobian(K, raw, zeros);
}
model {
  L_Omega ~ lkj_corr_cholesky(eta);
}
generated quantities {
  matrix[K, K] Omega = multiply_lower_tri_self_transpose(L_Omega);
}
