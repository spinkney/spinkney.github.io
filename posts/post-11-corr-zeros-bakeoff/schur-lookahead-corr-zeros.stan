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

  // sqrt((1-a^2)(1-b^2)), evaluated stably on [-1,1]^2.
  real unit_circle_root_product(real a, real b) {
    real q = (1 - a) * (1 + a) * (1 - b) * (1 + b);
    return sqrt(fmax(0.0, q));
  }

  /*
    Exact projection of the 3 x 3 correlation elliptope. Let x be the
    correlation to project, with y in [y_lo, y_hi] and r in [r_lo, r_hi].
    The result is [minimum feasible x, maximum feasible x].
  */
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

    // Maximum x.
    if (fmax(y_lo, r_lo) <= fmin(y_hi, r_hi)) {
      x_hi = 1.0;
    } else if (y_lo > r_hi) {
      x_hi = y_lo * r_hi + unit_circle_root_product(y_lo, r_hi);
    } else {
      x_hi = y_hi * r_lo + unit_circle_root_product(y_hi, r_lo);
    }

    // Minimum x.
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

  matrix cholesky_corr_constrain_schur_lookahead_jacobian(
      int K, vector raw, array[,] int zeros, int propagation_passes) {
    matrix[K, K] L = rep_matrix(0, K, K);
    int raw_idx = 1;
    int zero_idx = 1;
    L[1, 1] = 1;

    for (i in 2:K) {
      // These boxes contain ordinary correlations Omega[i,j], not L entries.
      vector[K] corr_lo = rep_vector(-1.0, K);
      vector[K] corr_hi = rep_vector(1.0, K);

      for (j in 1:(i - 1)) {
        if (is_zero_scan(i, j, zeros)) {
          corr_lo[j] = 0.0;
          corr_hi[j] = 0.0;
        }
      }

      // The untouched tail stores the current remaining row radius.
      L[i, 1:i] = rep_row_vector(1, i);

      for (j in 1:(i - 1)) {
        int current_is_zero = is_zero(i, j, zeros, zero_idx);
        real r_row = L[i, j];
        real L_jj = L[j, j];
        real b1 = dot_product(L[j, 1:(j - 1)],
                              L[i, 1:(j - 1)]);
        real direct_lo = fmax(-1.0, b1 - r_row * L_jj);
        real direct_hi = fmin(1.0, b1 + r_row * L_jj);

        // Intersect the current correlation with its exact Cholesky-stick
        // range. Future correlations retain conservative boxes until current.
        corr_lo[j] = fmax(corr_lo[j], direct_lo);
        corr_hi[j] = fmin(corr_hi[j], direct_hi);

        if (corr_lo[j] > corr_hi[j] + 1e-12) {
          reject("Current ordinary-correlation interval is empty at pair (",
                 i, ",", j, "): [", corr_lo[j], ",", corr_hi[j], "].");
        }

        /*
          Synchronous propagation across all pairs in this row. Omega[a,b]
          is fixed because rows a and b are already complete. Each pass reads
          only old_lo/old_hi; sampled row cells are point intervals.
        */
        if (propagation_passes > 0 && i > 2) {
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
          jacobian += -log(L_jj);  // ordinary-zero delta correction
          zero_idx += 1;
          L[i, (j + 1):i] *= sqrt(1 - square(L[i, j] / r_row));
        } else {
          if (raw_idx > num_elements(raw)) {
            reject("Too few raw coordinates; missing pair (", i, ",", j,
                   ").");
          }

          if (propagation_passes == 0) {
            // Preserve the original canonical Schur chart exactly.
            real x = raw[raw_idx];
            real cosh_x = cosh(x);
            jacobian += log(r_row) - 2 * log(cosh_x);
            L[i, j] = r_row * tanh(x);
            L[i, (j + 1):i] /= cosh_x;
          } else {
            /*
              Translate the propagated ordinary-correlation box to the
              normalized direct entry u=L[i,j]/r_row. The map

                q = inv_logit(2 raw),  u = u_lo + (u_hi-u_lo) q

              equals tanh(raw) when [u_lo,u_hi]=[-1,1]. Thus it is the
              bounded extension of the original chart. Its raw-to-L
              Jacobian is r_row * (u_hi-u_lo) * 2 q (1-q).
            */
            real low_L = fmax(-r_row, (corr_lo[j] - b1) / L_jj);
            real up_L = fmin(r_row, (corr_hi[j] - b1) / L_jj);
            real u_lo = low_L / r_row;
            real u_hi = up_L / r_row;
            real width_u = u_hi - u_lo;
            real z = 2.0 * raw[raw_idx];
            real q;

            if (!(width_u > 1e-12)) {
              reject("Free normalized-L interval is empty at pair (", i,
                     ",", j, "). Interval=[", u_lo, ",", u_hi, "].");
            }

            q = inv_logit(z);
            if (!(q > 0.0 && q < 1.0)) {
              reject("Scaled logistic reached an endpoint at raw[", raw_idx,
                     "]=", raw[raw_idx], ".");
            }

            L[i, j] = r_row * (u_lo + width_u * q);
            jacobian += log(r_row) + log(width_u) + log(2.0)
                        + log_inv_logit(z) + log1m_inv_logit(z);
            L[i, (j + 1):i] *= sqrt(1 - square(L[i, j] / r_row));
          }
          raw_idx += 1;
        }

        // Sampled ordinary correlations become point intervals. Preserve an
        // exact zero in the propagation state rather than its roundoff image.
        {
          real corr_ij = current_is_zero
                         ? 0.0 : b1 + L[i, j] * L_jj;
          corr_lo[j] = corr_ij;
          corr_hi[j] = corr_ij;
        }
      }
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
  array[N_zero, 2] int zeros;  // lower triangular, row-major order
  int<lower=0, upper=8> propagation_passes;
}

parameters {
  vector[choose(K, 2) - N_zero] raw;
}

transformed parameters {
  matrix[K, K] L_Omega =
      cholesky_corr_constrain_schur_lookahead_jacobian(
          K, raw, zeros, propagation_passes);
}

model {
  L_Omega ~ lkj_corr_cholesky(eta);
}

generated quantities {
  matrix[K, K] Omega = multiply_lower_tri_self_transpose(L_Omega);
}
