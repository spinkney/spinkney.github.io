/*
  Direct ordinary-correlation transform for a box certified to lie entirely
  inside the positive-definite correlation cone.

  This model deliberately performs no feasibility look-ahead.  It is valid
  only when the supplied box has been certified externally (for example with
  a spectral-norm/Weyl bound around a positive-definite anchor).
*/

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

        if (abs(lo - corr_lower[j, i]) > validation_tolerance
            || abs(hi - corr_upper[j, i]) > validation_tolerance) {
          reject("Bounds are not symmetric at pair (", i, ",", j, ").");
        }
        if (lo < -1.0 || hi > 1.0 || !(lo < hi)) {
          reject("Pair (", i, ",", j,
                 ") must have positive-width bounds inside [-1,1]; got [",
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
  corr_matrix[K] Omega;
  real log_det_Omega;

  {
    matrix[K, K] work = identity_matrix(K);
    int pos = 1;

    for (i in 2:K) {
      for (j in 1:(i - 1)) {
        real value = lower_upper_bound_jacobian(
            raw[pos], corr_lower[i, j], corr_upper[i, j]);
        work[i, j] = value;
        work[j, i] = value;
        pos += 1;
      }
    }

    Omega = work;
    log_det_Omega = log_determinant_spd(Omega);
  }
}

model {
  target += (eta - 1.0) * log_det_Omega;
}
