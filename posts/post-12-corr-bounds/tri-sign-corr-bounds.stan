/*
  Smooth triangular transform for prescribed correlation signs.

  Every off-diagonal cell is either positive, with bounds [0, 1], or
  negative, with bounds [-1, 0].  In Cholesky row i, let B be the leading
  Cholesky block and c the ordinary correlations with earlier variables.
  The transform creates a vector x in the requested orthant, solves

      w = B^{-1} x,

  and applies the modified stereographic hemisphere map

      z = sqrt(w'w + 2) / (w'w + 1) * w,
      L[i,i] = 1 / (w'w + 1).

  Because c = B z is a positive scalar multiple of x, every requested sign
  is exact and every raw value produces a positive-definite correlation
  matrix.  The B and B^{-1} determinants cancel in the raw-to-c Jacobian.
*/

functions {
  tuple(matrix, real) corr_cholesky_sign_jacobian(
      int K,
      vector raw,
      matrix corr_lower,
      matrix corr_upper) {
    matrix[K, K] L = rep_matrix(0.0, K, K);
    int raw_pos = 1;
    real log_det = 0.0;
    real input_scale = 2.0 * K;

    L[1, 1] = 1.0;

    for (i in 2:K) {
      int d = i - 1;
      vector[d] x;
      matrix[d, d] previous_cholesky = L[1:d, 1:d];

      for (j in 1:d) {
        real raw_ij = raw[raw_pos + j - 1];
        real magnitude = log1p_exp(raw_ij) / input_scale;
        real sign_ij = corr_lower[i, j] == 0.0 ? 1.0 : -1.0;

        x[j] = sign_ij * magnitude;
        jacobian += log_inv_logit(raw_ij) - log(input_scale);
      }

      {
        vector[d] w = mdivide_left_tri_low(previous_cholesky, x);
        real t = dot_self(w);
        real scale = sqrt(t + 2.0) / (t + 1.0);

        L[i, 1:d] = (scale * w)';
        L[i, i] = 1.0 / (t + 1.0);
        log_det += 2.0 * log(L[i, i]);

        // Exact determinant of w -> scale(t) * w.  The constant log(2)
        // is retained even though it could be dropped from the target.
        jacobian += log(2.0)
                    + 0.5 * (d - 2.0) * log(t + 2.0)
                    - (d + 1.0) * log1p(t);
      }

      raw_pos += d;
    }

    if (raw_pos != num_elements(raw) + 1) {
      reject("Sign transform consumed ", raw_pos - 1,
             " raw values but received ", num_elements(raw), ".");
    }

    return (L, log_det);
  }
}

data {
  int<lower=2> K;
  real<lower=0> eta;
  matrix[K, K] corr_lower;
  matrix[K, K] corr_upper;
  real<lower=0> validation_tolerance;
}

transformed data {
  if (!(eta > 0.0)) {
    reject("The LKJ shape eta must be strictly positive; received ", eta,
           ".");
  }

  for (i in 1:K) {
    if (abs(corr_lower[i, i] - 1.0) > validation_tolerance
        || abs(corr_upper[i, i] - 1.0) > validation_tolerance) {
      reject("Both bound matrices must equal one on diagonal ", i, ".");
    }

    if (i > 1) {
      for (j in 1:(i - 1)) {
        real lo = corr_lower[i, j];
        real hi = corr_upper[i, j];
        int positive_sign = lo == 0.0 && hi == 1.0;
        int negative_sign = lo == -1.0 && hi == 0.0;

        if (abs(lo - corr_lower[j, i]) > validation_tolerance
            || abs(hi - corr_upper[j, i]) > validation_tolerance) {
          reject("Bounds are not symmetric at pair (", i, ",", j, ").");
        }
        if (!(positive_sign || negative_sign)) {
          reject("Pair (", i, ",", j,
                 ") must request [0,1] or [-1,0]; received [",
                 lo, ",", hi, "].");
        }
      }
    }
  }
}

parameters {
  vector[choose(K, 2)] raw;
}

transformed parameters {
  matrix[K, K] L_Omega;
  corr_matrix[K] Omega;
  real log_det_Omega;

  {
    tuple(matrix[K, K], real) result = corr_cholesky_sign_jacobian(
        K, raw, corr_lower, corr_upper);
    L_Omega = result.1;
    log_det_Omega = result.2;
    Omega = multiply_lower_tri_self_transpose(L_Omega);
  }
}

model {
  target += (eta - 1.0) * log_det_Omega;
}
