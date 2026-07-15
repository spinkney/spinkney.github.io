functions {
  // traversal here is tree-by-tree (column major), so scan instead of the
  // merge-join used by the row-major models
  int is_zero_scan(int row_idx, int col_idx, array[,] int zeros) {
    for (n in 1:size(zeros)) {
      if (zeros[n, 1] == row_idx && zeros[n, 2] == col_idx) {
        return 1;
      }
    }
    return 0;
  }

  matrix corr_constrain_cvine_jacobian(int K, vector raw,
                                       array[,] int zeros, real eta) {
    matrix[K, K] C = identity_matrix(K);
    matrix[K, K] P = identity_matrix(K);
    // LKJ in C-vine form: partial corr in tree i ~ Beta(a - i/2 adjustments);
    // constant base a with the -0.5 * i * log1m(p^2) correction below
    real a = eta + 0.5 * (K - 1);
    int raw_idx = 1;

    // tree 1: marginal correlations with variable 1
    for (k in 2:K) {
      if (is_zero_scan(k, 1, zeros)) {
        P[1, k] = 0;  // prior terms at p = 0 are constants
      } else {
        P[1, k] = lower_upper_bound_jacobian(raw[raw_idx], -1, 1);
        raw_idx += 1;
        jacobian += beta_lpdf(0.5 * (P[1, k] + 1) | a, a)
                    - 0.5 * log1m(P[1, k]^2);
      }
      C[1, k] = P[1, k];
      C[k, 1] = P[1, k];
    }

    for (i in 2:(K - 1)) {
      for (j in (i + 1):K) {
        vector[i - 1] b1;
        vector[i - 1] b2;
        int m = i;
        for (k in 1:(i - 1)) {
          m -= 1;
          b1[k] = P[m, i] * P[m, j];
          b2[k] = sqrt((1 - P[m, i]^2) * (1 - P[m, j]^2));
        }
        // unwinding the partial correlation recursion is affine:
        // C[i, j] = A + B * P[i, j]
        real A = 0;
        real B = 1;
        for (k in 1:(i - 1)) {
          A = b1[k] + A * b2[k];
          B *= b2[k];
        }
        real p_ij;
        if (is_zero_scan(j, i, zeros)) {
          p_ij = -A / B;         // C[i, j] = 0
          jacobian += -log(B);   // delta in Omega coords -> partial coords
        } else {
          p_ij = lower_upper_bound_jacobian(raw[raw_idx], -1, 1);
          raw_idx += 1;
        }
        // LKJ prior on the tree-i partial (evaluated at p* for zeros)
        jacobian += beta_lpdf(0.5 * (p_ij + 1) | a, a)
                    - 0.5 * i * log1m(p_ij^2);
        P[i, j] = p_ij;
        C[i, j] = A + B * p_ij;
        C[j, i] = C[i, j];
      }
    }
    return C;
  }
}
data {
  int<lower=2> K;                             // dimension of correlation matrix
  real<lower=0> eta;                          // concentration in LKJ
  int<lower=0, upper=choose(K, 2)> N_zero;    // # structural zero correlations
  array[N_zero, 2] int zeros;                 // lower triangular index pairs
}
parameters {
  vector[choose(K, 2) - N_zero] raw;
}
transformed parameters {
  // prior is included via the beta terms in the constraint, so the
  // model block stays empty
  matrix[K, K] Omega = corr_constrain_cvine_jacobian(K, raw, zeros, eta);
}
model {
}
