functions {
  int is_zero(int row_idx, int col_idx, array[,] int zeros, int zero_idx) {
    return zero_idx <= size(zeros)
      && zeros[zero_idx, 1] == row_idx
      && zeros[zero_idx, 2] == col_idx;
  }

  int is_zero_scan(int row_idx, int col_idx, array[,] int zeros) {
    if (size(zeros) > 0) {
      for (n in 1:size(zeros)) {
        if (zeros[n, 1] == row_idx && zeros[n, 2] == col_idx) {
          return 1;
        }
      }
    }
    return 0;
  }

  real unit_circle_root_product(real a, real b) {
    real q = (1 - a) * (1 + a) * (1 - b) * (1 + b);
    return sqrt(fmax(0.0, q));
  }

  vector corr3_project_first(real y_lo_in, real y_hi_in,
                             real r_lo_in, real r_hi_in) {
    real y_lo = fmax(-1.0, fmin(1.0, y_lo_in));
    real y_hi = fmax(-1.0, fmin(1.0, y_hi_in));
    real r_lo = fmax(-1.0, fmin(1.0, r_lo_in));
    real r_hi = fmax(-1.0, fmin(1.0, r_hi_in));
    real x_lo;
    real x_hi;
    vector[2] out;

    if (y_lo > y_hi || r_lo > r_hi) {
      reject("corr3_project_first received a reversed interval: y=[",
             y_lo, ",", y_hi, "], r=[", r_lo, ",", r_hi, "].");
    }

    if (fmax(y_lo, r_lo) <= fmin(y_hi, r_hi)) {
      x_hi = 1.0;
    } else if (y_lo > r_hi) {
      x_hi = y_lo * r_hi + unit_circle_root_product(y_lo, r_hi);
    } else {
      x_hi = y_hi * r_lo + unit_circle_root_product(y_hi, r_lo);
    }

    if (fmax(y_lo, -r_hi) <= fmin(y_hi, -r_lo)) {
      x_lo = -1.0;
    } else if (y_lo > -r_lo) {
      x_lo = y_lo * r_lo - unit_circle_root_product(y_lo, r_lo);
    } else {
      x_lo = y_hi * r_hi - unit_circle_root_product(y_hi, r_hi);
    }

    out[1] = fmax(-1.0, fmin(1.0, x_lo));
    out[2] = fmax(-1.0, fmin(1.0, x_hi));
    return out;
  }

  matrix cholesky_corr_zeros_lookahead_jacobian(
      int K, vector raw, array[,] int zeros, int propagation_passes) {
    int raw_idx = 1;
    int zero_idx = 1;
    matrix[K, K] L = rep_matrix(0, K, K);

    for (i in 1:K) {
      real stick = 1;
      vector[K] corr_lo = rep_vector(-1.0, K);
      vector[K] corr_hi = rep_vector(1.0, K);

      if (i > 1) {
        for (j in 1:(i - 1)) {
          if (is_zero_scan(i, j, zeros)) {
            corr_lo[j] = 0.0;
            corr_hi[j] = 0.0;
          }
        }
      }

      for (j in 1:(i - 1)) {
        int current_is_zero = is_zero(i, j, zeros, zero_idx);

        if (propagation_passes == 0) {
          // Keep this branch operation-for-operation equivalent to carpenter.
          if (current_is_zero) {
            real b = dot_product(L[j, 1:(j - 1)],
                                 L[i, 1:(j - 1)]);
            L[i, j] = -b / L[j, j];
            zero_idx += 1;
          } else {
            real sqrt_stick = sqrt(stick);
            L[i, j] = lower_upper_bound_jacobian(
                raw[raw_idx], -sqrt_stick, sqrt_stick);
            raw_idx += 1;
            jacobian += 0;
          }
        } else {
          real r_row = sqrt(stick);
          real L_jj = L[j, j];
          real b1 = dot_product(L[j, 1:(j - 1)],
                                L[i, 1:(j - 1)]);
          real direct_lo = fmax(-1.0, b1 - r_row * L_jj);
          real direct_hi = fmin(1.0, b1 + r_row * L_jj);

          corr_lo[j] = fmax(corr_lo[j], direct_lo);
          corr_hi[j] = fmin(corr_hi[j], direct_hi);

          if (corr_lo[j] > corr_hi[j] + 1e-12) {
            reject("Current ordinary-correlation interval is empty at pair (",
                   i, ",", j, "): [", corr_lo[j], ",", corr_hi[j], "].");
          }

          if (i > 2) {
            for (pass in 1:propagation_passes) {
              vector[K] old_lo = corr_lo;
              vector[K] old_hi = corr_hi;
              vector[K] next_lo = old_lo;
              vector[K] next_hi = old_hi;

              for (a in 1:(i - 2)) {
                for (b in (a + 1):(i - 1)) {
                  real corr_ab = dot_product(L[a, 1:a], L[b, 1:a]);
                  vector[2] projected_a = corr3_project_first(
                      old_lo[b], old_hi[b], corr_ab, corr_ab);
                  vector[2] projected_b = corr3_project_first(
                      old_lo[a], old_hi[a], corr_ab, corr_ab);

                  next_lo[a] = fmax(next_lo[a], projected_a[1]);
                  next_hi[a] = fmin(next_hi[a], projected_a[2]);
                  next_lo[b] = fmax(next_lo[b], projected_b[1]);
                  next_hi[b] = fmin(next_hi[b], projected_b[2]);
                }
              }

              for (a in 1:(i - 1)) {
                if (next_lo[a] > next_hi[a] + 1e-12) {
                  reject("3 x 3 propagation emptied row-correlation interval ",
                         "at pair (", i, ",", a, ") on pass ", pass,
                         ": [", next_lo[a], ",", next_hi[a], "].");
                }
                if (next_lo[a] > next_hi[a]) {
                  real midpoint = 0.5 * (next_lo[a] + next_hi[a]);
                  next_lo[a] = midpoint;
                  next_hi[a] = midpoint;
                }
              }

              corr_lo = next_lo;
              corr_hi = next_hi;
            }
          }

          if (current_is_zero) {
            if (0.0 < corr_lo[j] - 1e-10
                || 0.0 > corr_hi[j] + 1e-10) {
              reject("Structural zero is infeasible after look-ahead at pair (",
                     i, ",", j, "). Interval=[", corr_lo[j], ",",
                     corr_hi[j], "].");
            }
            L[i, j] = -b1 / L_jj;
            zero_idx += 1;
          } else {
            real low = fmax(-r_row, (corr_lo[j] - b1) / L_jj);
            real up = fmin(r_row, (corr_hi[j] - b1) / L_jj);

            if (!(up - low > 1e-12)) {
              reject("Free direct-L interval is empty at pair (", i, ",", j,
                     "). Interval=[", low, ",", up, "].");
            }
            if (raw_idx > num_elements(raw)) {
              reject("Too few raw coordinates; missing pair (", i, ",", j,
                     ").");
            }
            L[i, j] = lower_upper_bound_jacobian(raw[raw_idx], low, up);
            raw_idx += 1;
          }
        }

        {
          real corr_ij = current_is_zero
                         ? 0.0 : dot_product(L[j, 1:j], L[i, 1:j]);
          corr_lo[j] = corr_ij;
          corr_hi[j] = corr_ij;
        }
        stick -= L[i, j]^2;
      }
      L[i, i] = sqrt(stick);
    }

    if (raw_idx != num_elements(raw) + 1) {
      reject("Received ", num_elements(raw), " raw coordinates but consumed ",
             raw_idx - 1, ".");
    }
    return L;
  }

  real lkj_cholesky_corr_zeros_lpdf(matrix L, real nu,
                                    array[,] int zeros) {
    real lp = lkj_corr_cholesky_lpdf(L | nu);
    int N_zero = size(zeros);
    for (n in 1:N_zero) {
      int col_idx = zeros[n, 2];
      lp -= log(L[col_idx, col_idx]);
    }
    return lp;
  }
}

data {
  int<lower=2> K;
  real<lower=0> eta;
  int<lower=0, upper=choose(K, 2)> N_zero;
  array[N_zero, 2] int zeros;
  int<lower=0, upper=8> propagation_passes;
}

parameters {
  vector[choose(K, 2) - N_zero] raw;
}

transformed parameters {
  matrix[K, K] L_Omega = cholesky_corr_zeros_lookahead_jacobian(
      K, raw, zeros, propagation_passes);
}

model {
  L_Omega ~ lkj_cholesky_corr_zeros(eta, zeros);
}

generated quantities {
  matrix[K, K] Omega = multiply_lower_tri_self_transpose(L_Omega);
}
