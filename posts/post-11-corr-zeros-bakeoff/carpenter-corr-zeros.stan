functions {
  int is_zero(int row_idx, int col_idx, array[,] int zeros, int zero_idx) {
    return zero_idx <= size(zeros)
      && zeros[zero_idx, 1] == row_idx
      && zeros[zero_idx, 2] == col_idx;
  }
  
  matrix cholesky_corr_zeros_jacobian(int D,
                                      vector raw,
                                      array[,] int zeros) {
    int raw_idx = 1;
    int zero_idx = 1;
    matrix[D, D] L = rep_matrix(0, D, D);
    for (i in 1:D) {
      real stick = 1;
      for (j in 1:(i - 1)) {
        if (is_zero(i, j, zeros, zero_idx)) {
          real b = dot_product(L[j, 1:(j - 1)], L[i, 1:(j - 1)]);
          L[i, j] = -b / L[j, j];  // implies Omega[i, j] == 0
          zero_idx += 1;
        } else {
	  real sqrt_stick = sqrt(stick);
          L[i, j] = lower_upper_bound_jacobian(raw[raw_idx], -sqrt_stick, sqrt_stick);  // inside stick
          raw_idx += 1;
          jacobian += 0;
        }
	stick -= L[i, j]^2;
      }
      L[i, i] = sqrt(stick);
    }
    return L;
  }

  real lkj_cholesky_corr_zeros_lpdf(matrix L,
                                    real nu,
                                    array[,] int zeros) {
    real lp = lkj_corr_cholesky_lpdf(L | nu);  // over-adjusts
    int N_zero = size(zeros);
    for (n in 1:N_zero) {                      
      int col_idx = zeros[n, 2];
      lp -= log(L[col_idx, col_idx]);          // correct over-adjustment
    }
    return lp;
  }
}
data {
  int<lower=2> D;                             // dimension of correlation matrix
  real<lower=0> eta;                          // concentration in LKJ Cholesky
  int<lower=0, upper=choose(D, 2)> N_zero;    // # structural zero correlations
  array[N_zero, 2] int zeros;                 // lower triangular, row major order
}
parameters {
  vector[choose(D, 2) - N_zero] raw;           // raw parameters
}
transformed parameters {
  matrix[D, D] L_Omega = cholesky_corr_zeros_jacobian(D, raw, zeros);
}
model {
  L_Omega ~ lkj_cholesky_corr_zeros(eta, zeros);
}
generated quantities {
  matrix[D, D] Omega = multiply_lower_tri_self_transpose(L_Omega);
}
