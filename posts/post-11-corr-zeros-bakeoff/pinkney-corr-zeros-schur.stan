functions {
  int is_zero(int row_idx, int col_idx, array[,] int zeros, int zero_idx) {
    return zero_idx <= size(zeros)
      && zeros[zero_idx, 1] == row_idx
      && zeros[zero_idx, 2] == col_idx;
  }

  matrix cholesky_corr_constrain_schur_jacobian(int K, vector raw,
                                                array[,] int zeros) {
    // Canonical partial correlation construction: each free entry is the
    // remaining radius times tanh(raw), so |L[i, j]| < radius automatically
    // and no correlation bounds are needed. The untouched tail of each row
    // holds the current radius (row-scaling trick).
    matrix[K, K] L = rep_matrix(0, K, K);
    int raw_idx = 1;
    int zero_idx = 1;
    L[1, 1] = 1;

    for (i in 2:K) {
      L[i, 1:i] = rep_row_vector(1, i);
      for (j in 1:(i - 1)) {
        real r = L[i, j];  // remaining radius of row i
        if (is_zero(i, j, zeros, zero_idx)) {
          real b1 = dot_product(L[j, 1:(j - 1)], L[i, 1:(j - 1)]);
          L[i, j] = -b1 / L[j, j];  // Omega[i, j] = b1 + Lij * Ljj = 0
          jacobian += -log(L[j, j]);  // Carpenter's correction
          zero_idx += 1;
          L[i, (j + 1):i] *= sqrt(1 - (L[i, j] / r)^2);
        } else {
          real x = raw[raw_idx];
          real cosh_x = cosh(x);
          // tanh jacobian and x -> L[i, j] = r * tanh(x);
          // the radius update sqrt(1 - tanh^2) is 1 / cosh
          jacobian += -2 * log(cosh_x) + log(r);
          L[i, j] = r * tanh(x);
          raw_idx += 1;
          L[i, (j + 1):i] /= cosh_x;
        }
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
  matrix[K, K] L_Omega = cholesky_corr_constrain_schur_jacobian(K, raw, zeros);
}
model {
  L_Omega ~ lkj_corr_cholesky(eta);
}
generated quantities {
  matrix[K, K] Omega = multiply_lower_tri_self_transpose(L_Omega);
}
