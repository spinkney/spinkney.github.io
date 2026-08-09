/*
  Globally feasible transform for arbitrary elementwise correlation bounds
  and exact fixed ordinary correlations.

  The feasible set is the intersection of the positive-definite correlation
  cone and an ordinary-correlation box.  Given a strictly interior anchor,
  every ray from that anchor remains feasible until it first reaches either a
  box face or the positive-definite boundary.  Cells whose lower and upper
  bounds are exactly equal are held fixed and omitted from the direction
  vector.  An auxiliary isotropic normal vector supplies a uniform direction
  on the sphere of the remaining free coordinates; its radius is independent
  of the resulting correlation matrix.

  The target is det(Omega)^(eta - 1) with respect to the remaining free
  ordinary off-diagonal correlations.
*/

functions {
  tuple(matrix, real, real, real, real) corr_global_ray_jacobian(
      vector direction_raw,
      real radius_raw,
      matrix corr_lower,
      matrix corr_upper,
      matrix anchor,
      matrix anchor_cholesky) {
    int K = rows(anchor);
    int D = num_elements(direction_raw);
    real direction_norm = sqrt(dot_self(direction_raw));
    vector[D] direction;
    matrix[K, K] direction_matrix = rep_matrix(0.0, K, K);
    real max_radius_box = positive_infinity();
    real max_radius_pd = positive_infinity();
    real max_radius;
    real radius_fraction = inv_logit(radius_raw);
    real radius;
    matrix[K, K] Omega;
    real log_det;
    int pos = 1;

    if (!(direction_norm > 0.0)) {
      reject("The auxiliary direction vector has zero norm.");
    }
    if (!(radius_fraction > 0.0 && radius_fraction < 1.0)) {
      reject("The radial inverse logit reached an endpoint; raw value=",
             radius_raw, ".");
    }

    direction = direction_raw / direction_norm;

    for (i in 2:K) {
      for (j in 1:(i - 1)) {
        if (corr_lower[i, j] != corr_upper[i, j]) {
          real component = direction[pos];
          real distance_to_face;

          direction_matrix[i, j] = component;
          direction_matrix[j, i] = component;

          if (component > 0.0) {
            distance_to_face = (corr_upper[i, j] - anchor[i, j])
                               / component;
            max_radius_box = fmin(max_radius_box, distance_to_face);
          } else if (component < 0.0) {
            distance_to_face = (corr_lower[i, j] - anchor[i, j])
                               / component;
            max_radius_box = fmin(max_radius_box, distance_to_face);
          }

          pos += 1;
        }
      }
    }

    if (pos != D + 1) {
      reject("Global-ray transform consumed ", pos - 1,
             " free direction values but received ", D, ".");
    }

    // C(t) = C0 + t*S is positive definite exactly while
    // I + t * L0^{-1} S L0^{-T} is positive definite.
    {
      matrix[K, K] left_solve = mdivide_left_tri_low(
          anchor_cholesky, direction_matrix);
      matrix[K, K] standardized_direction = mdivide_left_tri_low(
          anchor_cholesky, left_solve')';
      vector[K] eigenvalues = eigenvalues_sym(
          0.5 * (standardized_direction + standardized_direction'));
      real lambda_min = eigenvalues[1];

      if (lambda_min < 0.0) {
        max_radius_pd = -1.0 / lambda_min;
      }
      max_radius = fmin(max_radius_box, max_radius_pd);
    }

    if (!(max_radius > 0.0) || is_inf(max_radius)) {
      reject("The supplied anchor has no finite positive ray in this ",
             "direction; max_radius=", max_radius, ".");
    }

    radius = radius_fraction * max_radius;
    Omega = anchor + radius * direction_matrix;

    {
      matrix[K, K] L = cholesky_decompose(Omega);
      log_det = 2.0 * sum(log(diagonal(L)));
    }

    // In D dimensions, dc = radius^(D-1) d(radius) d(surface).
    // Normalizing an isotropic standard-normal auxiliary vector supplies
    // uniform surface measure, so only the ray and logistic terms remain.
    jacobian += (D - 1.0) * log(radius)
                + log(max_radius)
                + log_inv_logit(radius_raw)
                + log1m_inv_logit(radius_raw);

    return (Omega, log_det, max_radius, max_radius_box, max_radius_pd);
  }
}

data {
  // With only one free cell, at any K, the unit direction is the
  // discontinuous sign of one scalar; use a direct bounded logistic for
  // that one-dimensional special case.
  int<lower=3> K;
  real<lower=0> eta;
  int<lower=0, upper=choose(K, 2)> N_fixed;
  matrix[K, K] corr_lower;
  matrix[K, K] corr_upper;
  corr_matrix[K] anchor;
  real<lower=0> validation_tolerance;
}

transformed data {
  matrix[K, K] anchor_cholesky = cholesky_decompose(anchor);
  int fixed_count = 0;
  int D_free = choose(K, 2) - N_fixed;

  if (!(eta > 0.0)) {
    reject("The LKJ shape eta must be strictly positive; received ", eta,
           ".");
  }
  if (D_free < 2) {
    reject("The normalized-direction global-ray transform requires at least ",
           "two free correlations; received ", D_free, ".");
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
        if (lo < -1.0 || hi > 1.0 || lo > hi) {
          reject("Pair (", i, ",", j,
                 ") must have ordered bounds inside [-1,1]; got [",
                 lo, ",", hi, "].");
        }

        if (lo == hi) {
          fixed_count += 1;
          if (anchor[i, j] != lo) {
            reject("Anchor must equal the exact fixed value at pair (", i,
                   ",", j, "): anchor=", anchor[i, j],
                   ", fixed value=", lo, ".");
          }
        } else {
          if (!(anchor[i, j] > lo + validation_tolerance
                && anchor[i, j] < hi - validation_tolerance)) {
            reject("Anchor is not strictly inside the bounds at pair (", i,
                   ",", j, "): anchor=", anchor[i, j], ", bounds=[",
                   lo, ",", hi, "].");
          }
        }
      }
    }
  }

  if (fixed_count != N_fixed) {
    reject("N_fixed declares ", N_fixed, " exact cells, but the lower and ",
           "upper bound matrices contain ", fixed_count,
           " equal off-diagonal pairs.");
  }
}

parameters {
  vector[choose(K, 2) - N_fixed] direction_raw;
  real radius_raw;
}

transformed parameters {
  corr_matrix[K] Omega;
  real log_det_Omega;
  real max_radius;
  real max_radius_box;
  real max_radius_pd;
  real pd_boundary_active;

  {
    tuple(matrix[K, K], real, real, real, real) result
        = corr_global_ray_jacobian(
            direction_raw, radius_raw,
            corr_lower, corr_upper, anchor, anchor_cholesky);
    Omega = result.1;
    log_det_Omega = result.2;
    max_radius = result.3;
    max_radius_box = result.4;
    max_radius_pd = result.5;
    pd_boundary_active = max_radius_pd < max_radius_box ? 1.0 : 0.0;
  }
}

model {
  // The auxiliary radius integrates out independently; isotropy makes the
  // normalized direction uniform on the sphere.
  direction_raw ~ std_normal();
  target += (eta - 1.0) * log_det_Omega;
}
