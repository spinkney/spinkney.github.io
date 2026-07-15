functions {
  matrix cholesky_corr_constrain_unitvec2_jacobian(int K, vector raw,
                                                   array[,] int zeros) {
    // Same exact-constraint geometry as the unitvec version: row i lives on
    // the unit sphere in the orthogonal complement of its zero-partner rows,
    // so every raw value maps to a valid constrained matrix. The chart on
    // the subsphere is the canonical-partial-correlation (tanh) stick
    // construction instead of the gnomonic map, to lighten the tails.
    matrix[K, K] L = rep_matrix(0, K, K);
    int raw_idx = 1;
    int zero_idx = 1;
    L[1, 1] = 1;

    for (i in 2:K) {
      int n_z = 0;
      while (zero_idx + n_z <= size(zeros)
             && zeros[zero_idx + n_z, 1] == i) {
        n_z += 1;
      }
      int d = i - 1 - n_z;

      // subspace coordinates via tanh sticks: w_k = tanh(x_k) * radius,
      // radius update sqrt(1 - tanh^2) = 1 / cosh, so the whole chart
      // jacobian collapses to accumulated powers of log(cosh)
      vector[d] w;
      real r = 1;
      for (k in 1:d) {
        real x = raw[raw_idx];
        real cosh_x = cosh(x);
        w[k] = r * tanh(x);
        jacobian += -(d - k + 2) * log(cosh_x);
        r /= cosh_x;
        raw_idx += 1;
      }

      if (n_z == 0) {
        if (d > 0) {
          L[i, 1:d] = w';
        }
        L[i, i] = r;
      } else {
        // columns are the zero-partner rows restricted to 1:(i - 1);
        // -0.5 * log det(Gram) from the joint delta over this row's zeros
        matrix[i - 1, n_z] A;
        for (n in 1:n_z) {
          A[, n] = L[zeros[zero_idx + n - 1, 2], 1:(i - 1)]';
        }
        // full qr: the complement basis lives in Q's last d columns and
        // diagonal(R) supplies the Gram term from one decomposition
        matrix[i - 1, i - 1] Q;
        matrix[i - 1, n_z] R;
        (Q, R) = qr(A);
        jacobian += -sum(log(abs(diagonal(R))));
        if (d > 0) {
          L[i, 1:(i - 1)] = (Q[, (n_z + 1):(i - 1)] * w)';
        }
        L[i, i] = r;
        zero_idx += n_z;
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
  matrix[K, K] L_Omega = cholesky_corr_constrain_unitvec2_jacobian(K, raw, zeros);
}
model {
  L_Omega ~ lkj_corr_cholesky(eta);
}
generated quantities {
  matrix[K, K] Omega = multiply_lower_tri_self_transpose(L_Omega);
}
