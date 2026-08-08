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

  matrix cholesky_corr_constrain_outer_lookahead_jacobian(
      int K, vector raw, array[,] int zeros, int propagation_passes) {
    matrix[K, K] L = rep_matrix(0, K, K);
    int raw_idx = 1;
    int zero_idx = 1;
    L[1, 1] = 1;

    for (i in 2:K) {
      real l_ij_old = 1;
      vector[K] corr_lo = rep_vector(-1.0, K);
      vector[K] corr_hi = rep_vector(1.0, K);

      for (j in 1:(i - 1)) {
        if (is_zero_scan(i, j, zeros)) {
          corr_lo[j] = 0.0;
          corr_hi[j] = 0.0;
        }
      }

      for (j in 1:(i - 1)) {
        int current_is_zero = is_zero(i, j, zeros, zero_idx);
        real l_ij_old_x_l_jj = l_ij_old * L[j, j];
        real b1 = dot_product(L[j, 1:(j - 1)],
                              L[i, 1:(j - 1)]);
        real x;

        if (propagation_passes == 0) {
          // Keep this branch operation-for-operation equivalent to pinkney.
          if (current_is_zero) {
            x = -b1;
            zero_idx += 1;
          } else {
            real low = max({-l_ij_old_x_l_jj, -1 - b1});
            real up = min({l_ij_old_x_l_jj, 1 - b1});
            x = lower_upper_bound_jacobian(raw[raw_idx], low, up);
            raw_idx += 1;
          }
        } else {
          real direct_lo = fmax(-1.0, b1 - l_ij_old_x_l_jj);
          real direct_hi = fmin(1.0, b1 + l_ij_old_x_l_jj);

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
            x = -b1;
            zero_idx += 1;
          } else {
            // x = Omega[i,j] - b1 = L[i,j] * L[j,j]. The exact
            // x -> L[i,j] scale remains the -log(L[j,j]) below.
            real low = fmax(-l_ij_old_x_l_jj, corr_lo[j] - b1);
            real up = fmin(l_ij_old_x_l_jj, corr_hi[j] - b1);

            if (!(up - low > 1e-12)) {
              reject("Free correlation-scale interval is empty at pair (",
                     i, ",", j, "). Interval=[", low, ",", up, "].");
            }
            if (raw_idx > num_elements(raw)) {
              reject("Too few raw coordinates; missing pair (", i, ",", j,
                     ").");
            }
            x = lower_upper_bound_jacobian(raw[raw_idx], low, up);
            raw_idx += 1;
          }
        }

        L[i, j] = x / L[j, j];
        jacobian += -log(L[j, j]);

        {
          real corr_ij = b1 + x;
          corr_lo[j] = corr_ij;
          corr_hi[j] = corr_ij;
        }
        l_ij_old *= sqrt(1 - (L[i, j] / l_ij_old)^2);
      }
      L[i, i] = l_ij_old;
    }

    if (raw_idx != num_elements(raw) + 1) {
      reject("Received ", num_elements(raw), " raw coordinates but consumed ",
             raw_idx - 1, ".");
    }
    return L;
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
  matrix[K, K] L_Omega = cholesky_corr_constrain_outer_lookahead_jacobian(
      K, raw, zeros, propagation_passes);
}

model {
  L_Omega ~ lkj_corr_cholesky(eta);
}

generated quantities {
  matrix[K, K] Omega = multiply_lower_tri_self_transpose(L_Omega);
}
