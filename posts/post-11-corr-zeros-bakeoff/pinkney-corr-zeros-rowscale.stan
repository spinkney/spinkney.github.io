functions {
  int is_zero(int row_idx, int col_idx, array[,] int zeros, int zero_idx) {
    return zero_idx <= size(zeros)
      && zeros[zero_idx, 1] == row_idx
      && zeros[zero_idx, 2] == col_idx;
  }

  matrix cholesky_corr_constrain_rowscale_jacobian(int K, vector raw,
                                                   array[,] int zeros) {
    // Samples L[i, j] directly (bounds pre-divided by L[j, j]) and keeps the
    // remaining radius in the untouched tail of the row: each entry of
    // L[i, j:i] holds the current radius until column j is sampled, then the
    // tail is scaled down. L[i, i] ends up as the diagonal automatically.
    matrix[K, K] L = rep_matrix(0, K, K);
    int raw_idx = 1;
    int zero_idx = 1;
    L[1, 1] = 1;

    for (i in 2:K) {
      L[i, 1:i] = rep_row_vector(1, i);
      for (j in 1:(i - 1)) {
        real l_ij_old = L[i, j];  // remaining radius of row i
        real b1 = dot_product(L[j, 1:(j - 1)], L[i, 1:(j - 1)]);
        if (is_zero(i, j, zeros, zero_idx)) {
          L[i, j] = -b1 / L[j, j];  // Omega[i, j] = b1 + Lij * Ljj = 0
          jacobian += -log(L[j, j]);  // Carpenter's correction
          zero_idx += 1;
        } else {
          // |L[i, j]| <= radius and Omega[i, j] = b1 + Lij * Ljj in (-1, 1);
          // sampling L directly, the 1 / L[j, j] factor is inside the
          // bound width, so no separate jacobian term
          real low = max({-l_ij_old, (-1 - b1) / L[j, j]});
          real up = min({l_ij_old, (1 - b1) / L[j, j]});
          L[i, j] = lower_upper_bound_jacobian(raw[raw_idx], low, up);
          raw_idx += 1;
        }
        L[i, (j + 1):i] *= sqrt(1 - (L[i, j] / l_ij_old)^2);
      }
    }
    return L;
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
  matrix[K, K] L_Omega = cholesky_corr_constrain_rowscale_jacobian(K, raw, zeros);
}
model {
  L_Omega ~ lkj_corr_cholesky(eta);
}
generated quantities {
  matrix[K, K] Omega = multiply_lower_tri_self_transpose(L_Omega);
}
