functions {
  matrix cholesky_corr_constrain_tri_jacobian(int K, vector raw,
                                              array[,] int zeros) {
    // Exact-constraint subspace geometry but with a canonical
    // triangular basis instead of Householder QR: basis vector k is a unit
    // at free column k with the row's forced columns filled in by
    // back-substitution through the zero constraints, then orthonormalized
    // by the Cholesky factor of its small Gram matrix. The basis is
    // anchored to matrix entries and varies smoothly with earlier rows.
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

      // subsphere coordinates via tanh sticks (cosh-form jacobian)
      vector[d] z;
      real r = 1;
      for (k in 1:d) {
        real x = raw[raw_idx];
        real cosh_x = cosh(x);
        z[k] = r * tanh(x);
        jacobian += -(d - k + 2) * log(cosh_x);
        r /= cosh_x;
        raw_idx += 1;
      }

      if (n_z == 0) {
        if (d > 0) {
          L[i, 1:d] = z';
        }
        L[i, i] = r;
      } else {
        array[n_z] int zc;
        for (m in 1:n_z) {
          zc[m] = zeros[zero_idx + m - 1, 2];
        }

        // the joint-delta term -0.5 * log det(Gram of zero-partner rows)
        // factors: splitting the partner matrix A by forced/free coords,
        // A_Z is triangular with the partners' diagonals, and by Sylvester
        // det(A'A) = prod(L[j,j])^2 * det(T'T)
        for (m in 1:n_z) {
          jacobian += -log(L[zc[m], zc[m]]);
        }

        if (d > 0) {
          array[d] int free_cols;
          {
            int fc = 0;
            int zp = 1;
            for (c in 1:(i - 1)) {
              if (zp <= n_z && zc[zp] == c) {
                zp += 1;
              } else {
                fc += 1;
                free_cols[fc] = c;
              }
            }
          }
          // triangular basis: unit at each free column, forced columns
          // filled by back-substitution so every column is orthogonal to
          // the zero-partner rows
          matrix[i - 1, d] T = rep_matrix(0, i - 1, d);
          for (k in 1:d) {
            int col = free_cols[k];
            T[col, k] = 1;
            for (m in 1:n_z) {
              int j = zc[m];
              if (j > col) {
                T[j, k] = -dot_product(L[j, 1:(j - 1)], T[1:(j - 1), k])
                          / L[j, j];
              }
            }
          }
          // T's free-column rows are the identity, so T'T = I + W'W with
          // W the forced-column rows only
          matrix[n_z, d] W;
          for (m in 1:n_z) {
            W[m, ] = T[zc[m], ];
          }
          // orthonormalize: v = T * S'^{-1} * z has ||v|| = ||z||;
          // diagonal(S) doubles as the det(T'T) half of the delta term
          matrix[d, d] S = cholesky_decompose(add_diag(crossprod(W), 1.0));
          jacobian += -sum(log(diagonal(S)));
          vector[d] t = mdivide_right_tri_low(z', S)';
          L[i, 1:(i - 1)] = (T * t)';
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
  matrix[K, K] L_Omega = cholesky_corr_constrain_tri_jacobian(K, raw, zeros);
}
model {
  L_Omega ~ lkj_corr_cholesky(eta);
}
generated quantities {
  matrix[K, K] Omega = multiply_lower_tri_self_transpose(L_Omega);
}
