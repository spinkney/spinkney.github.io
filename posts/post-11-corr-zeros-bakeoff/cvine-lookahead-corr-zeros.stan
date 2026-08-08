/*
  Look-ahead C-vine parameterization of a correlation matrix with exact structural
  zeros. This is the equality-specialized interface to the look-ahead C-vine
  transform in https://gist.github.com/spinkney/7d994fb03079c9edb8a0e83a40d2faad.

  Each structural zero is an exact ordinary-correlation equality and consumes
  no element of `raw`. All other ordinary correlations have bounds (-1, 1).
  The transform accumulates the reduced raw-to-free-correlation Jacobian, and
  the model adds the LKJ determinant kernel with respect to those free cells.
*/

functions {
  // sqrt((1-a^2)(1-b^2)), evaluated stably on [-1,1]^2.
  real unit_circle_root_product(real a, real b) {
    real q = (1 - a) * (1 + a) * (1 - b) * (1 + b);
    return sqrt(fmax(0.0, q));
  }

  /*
    Exact projection of the 3 x 3 correlation elliptope. Let x be the
    correlation to project, with y in [y_lo, y_hi] and r in [r_lo, r_hi].
    The result is [minimum feasible x, maximum feasible x]. Degenerate point
    intervals are retained for exact zero constraints.
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

  /*
    Map unconstrained free coordinates to a box-constrained correlation
    matrix while solving exact ordinary-correlation equalities. The complete
    reduced raw -> free ordinary-correlation Jacobian is accumulated here.
  */
  tuple(matrix, real) cvine_bound_equality_jacobian(
      vector raw,
      matrix corr_lower,
      matrix corr_upper,
      real interval_tolerance,
      real validation_tolerance) {
    int K = rows(corr_lower);
    int M_free = num_elements(raw);
    int pos_free = 1;
    array[K, K] int pair_is_free = rep_array(0, K, K);
    matrix[K, K] lo_work = corr_lower;
    matrix[K, K] hi_work = corr_upper;
    matrix[K, K] P_work = identity_matrix(K);
    matrix[K, K] C_work = identity_matrix(K);
    real logJ = 0.0;
    real logdet = 0.0;

    if (cols(corr_lower) != K
        || rows(corr_upper) != K
        || cols(corr_upper) != K) {
      reject("cvine_bound_equality_transform requires equally sized square ",
             "bound matrices.");
    }

    // Classify coordinates and prepare the initial conditional bounds.
    for (i in 1:K) {
      lo_work[i, i] = 1.0;
      hi_work[i, i] = 1.0;

      if (i > 1) {
        for (j in 1:(i - 1)) {
          real width = corr_upper[i, j] - corr_lower[i, j];
          real lo_ij = corr_lower[i, j];
          real hi_ij = corr_upper[i, j];

          if (width > 0.0) {
            pair_is_free[i, j] = 1;
            pair_is_free[j, i] = 1;
            lo_ij = fmax(-1.0, corr_lower[i, j]);
            hi_ij = fmin(1.0, corr_upper[i, j]);
          } else if (width == 0.0) {
            // Preserve a point equality exactly.
            lo_ij = corr_lower[i, j];
            hi_ij = corr_lower[i, j];
          } else {
            reject("cvine_bound_equality_transform received a reversed ",
                   "interval at pair (", i, ",", j, "): [",
                   corr_lower[i, j], ",", corr_upper[i, j], "].");
          }

          lo_work[i, j] = lo_ij;
          lo_work[j, i] = lo_ij;
          hi_work[i, j] = hi_ij;
          hi_work[j, i] = hi_ij;
        }
      }
    }

    // Sequential C-vine construction.
    for (i in 1:(K - 1)) {
      for (j in (i + 1):K) {
        int is_free = pair_is_free[i, j];
        real lo_direct = lo_work[i, j];
        real hi_direct = hi_work[i, j];
        real lo = fmax(-1.0, lo_direct);
        real hi = fmin(1.0, hi_direct);
        real x;
        real log_scale = 0.0;

        // d C[i,j] / d P[i,j], holding earlier C-vine rows fixed.
        if (i > 1) {
          for (m in 1:(i - 1)) {
            log_scale += 0.5 * (
                log1m(square(P_work[m, i]))
                + log1m(square(P_work[m, j])));
          }
        }

        if (is_free == 1) {
          real width;

          // Intersect the direct interval with every 3 x 3 projection.
          if (j < K) {
            for (k in (j + 1):K) {
              vector[2] projected = corr3_project_first(
                  lo_work[i, k], hi_work[i, k],
                  lo_work[j, k], hi_work[j, k]);
              lo = fmax(lo, projected[1]);
              hi = fmin(hi, projected[2]);
            }
          }

          width = hi - lo;
          if (!(width > interval_tolerance)) {
            reject("Empty free C-vine interval at partial pair (", i, ",", j,
                   "). Interval after look-ahead=[", lo, ",", hi, "].");
          }
          if (pos_free > M_free) {
            reject("cvine_bound_equality_transform received too few raw ",
                   "coordinates; missing the coordinate for pair (", i, ",",
                   j, ").");
          }

          {
            real q = inv_logit(raw[pos_free]);

            if (!(q > 0.0 && q < 1.0)) {
              reject("inv_logit(raw[", pos_free,
                     "]) numerically reached an endpoint; raw=",
                     raw[pos_free], ".");
            }

            x = lo + width * q;
            logJ += log(width)
                    + log_inv_logit(raw[pos_free])
                    + log1m_inv_logit(raw[pos_free])
                    + log_scale;
          }

          pos_free += 1;
        } else {
          real fixed_x = 0.5 * (lo_direct + hi_direct);

          if (abs(hi_direct - lo_direct) > validation_tolerance) {
            reject("Internal equality interval acquired nonzero width at pair (",
                   i, ",", j, "): [", lo_direct, ",", hi_direct, "].");
          }
          if (!(fixed_x > -1.0 && fixed_x < 1.0)) {
            reject("Exact equality implies a non-interior partial correlation ",
                   "at pair (", i, ",", j, "): ", fixed_x, ".");
          }

          // Validate rather than alter the exact point during look-ahead.
          if (j < K) {
            for (k in (j + 1):K) {
              vector[2] projected = corr3_project_first(
                  lo_work[i, k], hi_work[i, k],
                  lo_work[j, k], hi_work[j, k]);

              if (fixed_x < projected[1] - validation_tolerance
                  || fixed_x > projected[2] + validation_tolerance) {
                reject("Exact equality partial at pair (", i, ",", j,
                       ") is incompatible with future pair ", k,
                       ". Fixed partial=", fixed_x,
                       ", projected interval=[", projected[1], ",",
                       projected[2], "].");
              }
            }
          }

          x = fixed_x;
        }

        P_work[i, j] = x;
        P_work[j, i] = x;
        logdet += log1m(square(x));

        // Update future intervals in the current C-vine row.
        if (j < K) {
          for (k in (j + 1):K) {
            vector[2] future_projected = corr3_project_first(
                x, x, lo_work[j, k], hi_work[j, k]);

            if (pair_is_free[i, k] == 1) {
              real next_lo = fmax(lo_work[i, k], future_projected[1]);
              real next_hi = fmin(hi_work[i, k], future_projected[2]);

              if (!(next_hi - next_lo > interval_tolerance)) {
                reject("Future free C-vine interval became empty after partial (",
                       i, ",", j, "). Future pair=", i, ",", k,
                       ", interval=[", next_lo, ",", next_hi, "].");
              }

              lo_work[i, k] = next_lo;
              lo_work[k, i] = next_lo;
              hi_work[i, k] = next_hi;
              hi_work[k, i] = next_hi;
            } else {
              real fixed_future = 0.5 * (
                  lo_work[i, k] + hi_work[i, k]);

              if (abs(hi_work[i, k] - lo_work[i, k])
                  > validation_tolerance) {
                reject("Internal future equality interval acquired width at ",
                       "pair (", i, ",", k, ").");
              }
              if (fixed_future < future_projected[1] - validation_tolerance
                  || fixed_future > future_projected[2]
                                      + validation_tolerance) {
                reject("Sampling partial (", i, ",", j,
                       ") made exact future equality pair (", i, ",", k,
                       ") infeasible. Fixed future partial=", fixed_future,
                       ", projected interval=[", future_projected[1], ",",
                       future_projected[2], "].");
              }

              lo_work[i, k] = fixed_future;
              lo_work[k, i] = fixed_future;
              hi_work[i, k] = fixed_future;
              hi_work[k, i] = fixed_future;
            }
          }
        }
      }

      // Propagate remaining intervals to the next C-vine level.
      if (i < K - 1) {
        for (j in (i + 1):(K - 1)) {
          for (k in (j + 1):K) {
            real pij = P_work[i, j];
            real pik = P_work[i, k];
            real denom = sqrt((1 - square(pij)) * (1 - square(pik)));

            if (!(denom > 0.0)) {
              reject("Zero conditional-correlation denominator at level ", i,
                     " for pair (", j, ",", k, ").");
            }

            if (pair_is_free[j, k] == 1) {
              real next_lo = fmax(
                  -1.0, (lo_work[j, k] - pij * pik) / denom);
              real next_hi = fmin(
                  1.0, (hi_work[j, k] - pij * pik) / denom);

              if (!(next_hi - next_lo > interval_tolerance)) {
                reject("Conditional free box became empty after C-vine row ",
                       i, " for pair (", j, ",", k, "). Interval=[",
                       next_lo, ",", next_hi, "].");
              }

              lo_work[j, k] = next_lo;
              lo_work[k, j] = next_lo;
              hi_work[j, k] = next_hi;
              hi_work[k, j] = next_hi;
            } else {
              real fixed_current = 0.5 * (
                  lo_work[j, k] + hi_work[j, k]);
              real fixed_next;

              if (abs(hi_work[j, k] - lo_work[j, k])
                  > validation_tolerance) {
                reject("Internal equality interval acquired width before ",
                       "conditioning at pair (", j, ",", k, ").");
              }

              fixed_next = (fixed_current - pij * pik) / denom;
              if (!(fixed_next > -1.0 && fixed_next < 1.0)) {
                reject("Exact ordinary equality implies a non-interior ",
                       "conditional correlation after level ", i,
                       " for pair (", j, ",", k, "): ", fixed_next, ".");
              }

              lo_work[j, k] = fixed_next;
              lo_work[k, j] = fixed_next;
              hi_work[j, k] = fixed_next;
              hi_work[k, j] = fixed_next;
            }
          }
        }
      }
    }

    if (pos_free != M_free + 1) {
      reject("cvine_bound_equality_transform received ", M_free,
             " raw coordinates but consumed ", pos_free - 1, ".");
    }

    // Convert C-vine partial correlations to ordinary correlations.
    for (i in 1:(K - 1)) {
      for (j in (i + 1):K) {
        real value = P_work[i, j];

        if (i > 1) {
          int m = i;
          for (step in 1:(i - 1)) {
            m -= 1;
            value = P_work[m, i] * P_work[m, j]
                    + value
                      * sqrt((1 - square(P_work[m, i]))
                             * (1 - square(P_work[m, j])));
          }
        }

        C_work[i, j] = value;
        C_work[j, i] = value;
      }
    }

    jacobian += logJ;
    return (C_work, logdet);
  }
}

data {
  int<lower=2> K;                             // dimension of correlation matrix
  real<lower=0> eta;                          // LKJ concentration
  int<lower=0, upper=choose(K, 2)> N_zero;    // # structural zero correlations
  array[N_zero, 2] int zeros;                 // lower triangular index pairs
}

transformed data {
  matrix[K, K] corr_lower = rep_matrix(-1.0, K, K);
  matrix[K, K] corr_upper = rep_matrix(1.0, K, K);

  for (i in 1:K) {
    corr_lower[i, i] = 1.0;
    corr_upper[i, i] = 1.0;
  }

  if (N_zero > 0) {
    for (n in 1:N_zero) {
      int i = zeros[n, 1];
      int j = zeros[n, 2];

      if (i < 2 || i > K || j < 1 || j >= i) {
        reject("zeros[", n, "] must be a lower-triangular index pair; got (",
               i, ",", j, ").");
      }
      if (n > 1) {
        for (m in 1:(n - 1)) {
          if (zeros[m, 1] == i && zeros[m, 2] == j) {
            reject("Duplicate structural-zero pair (", i, ",", j, ").");
          }
        }
      }

      corr_lower[i, j] = 0.0;
      corr_lower[j, i] = 0.0;
      corr_upper[i, j] = 0.0;
      corr_upper[j, i] = 0.0;
    }
  }
}

parameters {
  // Exact zeros are omitted from the parameter vector.
  vector[choose(K, 2) - N_zero] raw;
}

transformed parameters {
  corr_matrix[K] Omega;
  real log_det_Omega;

  {
    tuple(matrix[K, K], real) transform = cvine_bound_equality_jacobian(
        raw, corr_lower, corr_upper, 1e-12, 1e-10);
    Omega = transform.1;
    log_det_Omega = transform.2;
  }
}

model {
  // LKJ density with respect to the free ordinary-correlation coordinates.
  target += (eta - 1.0) * log_det_Omega;
}
