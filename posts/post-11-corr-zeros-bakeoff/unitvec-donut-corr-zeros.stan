functions {
  matrix cholesky_corr_constrain_donut_jacobian(int K, vector raw,
                                                array[,] int zeros,
                                                real m_radius) {
    // Exact-constraint subspace geometry with the triangular basis, but the
    // subsphere is parameterized by normalizing a free vector ("donut"):
    // one extra parameter per row, u = y / ||y||, with -0.5 * y'y keeping
    // the radius Gaussian and independent of the direction. The diagonal
    // is |u_last| (the two mirror hemispheres fold onto the same L), and
    // the measure conversion dz = |u_last| dsigma contributes log|u_last|
    // on top of the LKJ weight supplied by the model block.
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

      vector[d + 1] y = raw[raw_idx:(raw_idx + d)];
      real s = norm2(y);
      vector[d + 1] u = y / s;
      // radial prior centered at m_radius repels the norm from the
      // origin singularity of the normalization (Axen, "A better unit
      // vector", Stan Discourse post 30); m_radius = 0 recovers the
      // implicit chi-distributed radius. The direction marginal, and
      // hence the Omega posterior, is unchanged.
      jacobian += -0.5 * square(s - m_radius) + log(abs(u[d + 1]));
      L[i, i] = abs(u[d + 1]);
      raw_idx += d + 1;

      if (n_z == 0) {
        if (d > 0) {
          L[i, 1:d] = u[1:d]';
        }
      } else {
        array[n_z] int zc;
        for (m in 1:n_z) {
          zc[m] = zeros[zero_idx + m - 1, 2];
        }

        // -0.5 * log det(Gram of zero-partner rows) from the joint delta
        matrix[i - 1, n_z] A;
        for (m in 1:n_z) {
          A[, m] = L[zc[m], 1:(i - 1)]';
        }
        jacobian += -sum(log(diagonal(cholesky_decompose(crossprod(A)))));

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
          // orthonormalize: v = T * S'^{-1} * u[1:d] has ||v|| = ||u[1:d]||
          matrix[d, d] S = cholesky_decompose(crossprod(T));
          vector[d] t = mdivide_right_tri_low(u[1:d]', S)';
          L[i, 1:(i - 1)] = (T * t)';
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
  real<lower=0> m_radius;                     // radial prior center; 0 = chi radius
}
parameters {
  // one extra parameter per row: dim d + 1 per row sums to
  // choose(K, 2) - N_zero + (K - 1)
  vector[choose(K, 2) - N_zero + K - 1] raw;
}
transformed parameters {
  matrix[K, K] L_Omega = cholesky_corr_constrain_donut_jacobian(K, raw, zeros,
                                                                m_radius);
}
model {
  L_Omega ~ lkj_corr_cholesky(eta);
}
generated quantities {
  matrix[K, K] Omega = multiply_lower_tri_self_transpose(L_Omega);
}
