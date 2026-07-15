functions {
  matrix cholesky_corr_constrain_unitvec_jacobian(int K, vector raw,
                                                  array[,] int zeros) {
    // Row i of a Cholesky correlation factor is a unit vector in R^i with
    // positive last coordinate. Omega[i, j] = 0 means row i is orthogonal
    // to row j, so row i lives on the unit sphere in the orthogonal
    // complement of its zero-partner rows: the constraint is satisfied
    // exactly by construction and no proposal can ever be infeasible.
    //
    // Chart per row: unit vector u = (C * y, 1)' / s with s = sqrt(1 + y'y),
    // C an orthonormal basis of the feasible subspace. Jacobian onto the
    // subsphere is -(d + 2) * log(s); the joint delta over the row's zeros
    // contributes -0.5 * log det(Gram) = -sum(log|diag(R)|) from the QR.
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

      if (n_z == 0) {
        vector[d] y = raw[raw_idx:(raw_idx + d - 1)];
        real s = sqrt(1 + dot_self(y));
        L[i, 1:(i - 1)] = y' / s;
        L[i, i] = 1 / s;
        jacobian += -(d + 2) * log(s);
        raw_idx += d;
      } else {
        // columns are the zero-partner rows restricted to 1:(i - 1)
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
          vector[d] y = raw[raw_idx:(raw_idx + d - 1)];
          real s = sqrt(1 + dot_self(y));
          L[i, 1:(i - 1)] = (Q[, (n_z + 1):(i - 1)] * y)' / s;
          L[i, i] = 1 / s;
          jacobian += -(d + 2) * log(s);
          raw_idx += d;
        } else {
          L[i, i] = 1;  // row fully constrained: e_i
        }
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
  matrix[K, K] L_Omega = cholesky_corr_constrain_unitvec_jacobian(K, raw, zeros);
}
model {
  L_Omega ~ lkj_corr_cholesky(eta);
}
generated quantities {
  matrix[K, K] Omega = multiply_lower_tri_self_transpose(L_Omega);
}
