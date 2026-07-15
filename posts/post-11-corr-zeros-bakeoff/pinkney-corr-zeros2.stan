functions {
  int is_zero(int row_idx, int col_idx, array[,] int zeros, int zero_idx) {
    return zero_idx <= size(zeros)
      && zeros[zero_idx, 1] == row_idx
      && zeros[zero_idx, 2] == col_idx;
  }

  matrix cholesky_corr_constrain_outer_jacobian(int K, vector raw,
                                                array[,] int zeros) {
    matrix[K, K] L = rep_matrix(0, K, K);
    int raw_idx = 1;
    int zero_idx = 1;
    L[1, 1] = 1;

    for (i in 2:K) {
      // remaining radius of row i: sqrt(1 - sum of squares so far)
      real l_ij_old = 1;
      for (j in 1:(i - 1)) {
        real l_ij_old_x_l_jj = l_ij_old * L[j, j];
        real b1 = dot_product(L[j, 1:(j - 1)], L[i, 1:(j - 1)]);
        // how to derive the bounds
        // we know that the correlation value C is bound by
        // b1 - Ljj * Lij_old <= C <= b1 + Ljj * Lij_old
        // Now we want our bounds to be enforced too so
        // max(lb, b1 - Ljj * Lij_old) <= C <= min(ub, b1 + Ljj * Lij_old)
        // We have Lij_new = (C - b1) / Ljj
        real x;
        if (is_zero(i, j, zeros, zero_idx)) {
          x = -b1;  // Omega[i, j] = b1 + Lij * Ljj = 0
          zero_idx += 1;
        } else {
          real low = max({-l_ij_old_x_l_jj, -1 - b1});
          real up = min({l_ij_old_x_l_jj, 1 - b1});
          x = lower_upper_bound_jacobian(raw[raw_idx], low, up);
          raw_idx += 1;
        }
        L[i, j] = x / L[j, j];
        // free entries: change of variables x -> L[i, j]
        // zero entries: Carpenter's correction for the LKJ over-adjustment
        jacobian += -log(L[j, j]);
        l_ij_old *= sqrt(1 - (L[i, j] / l_ij_old)^2);
      }
      L[i, i] = l_ij_old;
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
  matrix[K, K] L_Omega = cholesky_corr_constrain_outer_jacobian(K, raw, zeros);
}
model {
  L_Omega ~ lkj_corr_cholesky(eta);
}
generated quantities {
  matrix[K, K] Omega = multiply_lower_tri_self_transpose(L_Omega);
}
