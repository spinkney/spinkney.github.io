functions {
  /**
   * Squashes unconstrained partial autocorrelation (PACF) parameters to [-1, 1].
   * Uses tanh transformation with scaling to maintain numerical stability.
   *
   * @param psi_raw A vector of unconstrained PACF parameters
   * @return A vector of squashed PACF parameters in [-1, 1]
   */
  vector squash_pacf(vector psi_raw) {
    int D = rows(psi_raw);
    if (D <= 1) {
      return tanh(psi_raw);
    }
    return tanh(psi_raw / sqrt(D - 1.0));
  }

  /**
   * Converts PACF coefficients to ACF coefficients using the Durbin-Levinson
   * algorithm. This is essential for constructing valid Toeplitz correlation
   * matrices from PACF parameters.
   *
   * @param psi_raw A vector of D unconstrained PACF parameters
   * @return A vector of D+1 ACF coefficients [rho_0, rho_1, ..., rho_D],
   *         where rho_0 = 1
   */
  vector pacf_to_acf(vector psi_raw) {
    int D = rows(psi_raw);
    vector[D + 1] rho;
    rho[1] = 1.0;
    if (D == 0) {
      return rho;
    }
    vector[D] psi = squash_pacf(psi_raw);
    vector[D] phi;
    for (k in 1:D) {
      real psi_k = psi[k];
      real phi_k_k = psi_k;
      if (k > 1) {
        vector[k - 1] phi_prev = segment(phi, 1, k - 1);
        for (j in 1:(k - 1)) {
          phi[j] = phi_prev[j] - psi_k * phi_prev[k - j];
        }
      }
      phi[k] = phi_k_k;
      rho[k + 1] = dot_product(segment(phi, 1, k),
                               reverse(segment(rho, 1, k)));
    }
    return rho;
  }

  /**
   * Builds a compound symmetric block-Toeplitz correlation matrix.
   *
   * The correlation matrix has the form:
   *   R = s * I_N (x) T_A + (1-s) * 1_N 1_N' (x) T_B
   *
   * where T_A and T_B are Toeplitz correlation matrices parametrized by
   * their PACF coefficients, I_N is the NxN identity matrix, and 1_N is
   * a vector of ones.
   *
   * @param psiA_raw Unconstrained PACF parameters for within-block correlations
   * @param psiB_raw Unconstrained PACF parameters for between-block correlations
   * @param s Mixing parameter in (0, 1)
   * @param M_blocksize Dimension of each block
   * @param N_blocks Number of blocks
   * @return The M_total x M_total correlation matrix where M_total = M_blocksize * N_blocks
   */
  matrix build_cs_block_toeplitz_corr(vector psiA_raw,
                                      vector psiB_raw,
                                      real s,
                                      int M_blocksize,
                                      int N_blocks) {
    int M_total = M_blocksize * N_blocks;
    matrix[M_total, M_total] R;

    vector[M_blocksize] rho_A = pacf_to_acf(psiA_raw);
    vector[M_blocksize] rho_B = pacf_to_acf(psiB_raw);

    vector[M_blocksize] rho_on_vec = s * rho_A + (1.0 - s) * rho_B;
    vector[M_blocksize] rho_off_vec = (1.0 - s) * rho_B - (s / (N_blocks - 1.0)) * rho_A;

    matrix[M_blocksize, M_blocksize] rho_on;
    matrix[M_blocksize, M_blocksize] rho_off;
    for (i in 1:M_blocksize) {
      for (j in i:M_blocksize) {
        real val_on = rho_on_vec[j - i + 1];
        rho_on[i, j] = val_on;
        rho_on[j, i] = val_on;

        real val_off = rho_off_vec[j - i + 1];
        rho_off[i, j] = val_off;
        rho_off[j, i] = val_off;
      }
    }

    for (i in 1:N_blocks) {
      int r1 = (i - 1) * M_blocksize + 1;
      int r2 = i * M_blocksize;

      R[r1:r2, r1:r2] = rho_on;

      for (j in (i + 1):N_blocks) {
        int c1 = (j - 1) * M_blocksize + 1;
        int c2 = j * M_blocksize;
        R[r1:r2, c1:c2] = rho_off;
        R[c1:c2, r1:r2] = rho_off;
      }
    }

    return R;
  }

  /**
   * Computes the log-determinant of a Toeplitz covariance matrix
   * using the innovations algorithm. Optimized for the case where
   * no quadratic form is needed.
   *
   * @param c The autocovariance vector (first row of the covariance matrix).
   * @return The log determinant.
   */
  real innovations_log_det_only(vector c) {
    int n = rows(c);
    real EPS = 1e-12;

    if (n == 0) {
      return 0.0;
    }
    if (c[1] <= EPS) {
      reject("Variance c[1] must be positive.");
    }

    array[n] int rev_idx;
    for (i in 1:n) {
      rev_idx[i] = n - i + 1;
    }

    real log_det = 0.0;
    vector[n - 1] phi_vec = rep_vector(0.0, n - 1);
    real v_prev;

    real v_curr = c[1];
    log_det += log(v_curr);
    v_prev = v_curr;

    for (k in 2:n) {
      int ar_order = k - 1;

      real s = 0.0;
      if (ar_order > 1) {
        array[ar_order - 1] int sub_rev_idx = rev_idx[(n - (ar_order - 1) + 1):n];
        s = dot_product(segment(phi_vec, 1, ar_order - 1),
                        segment(c, 2, ar_order - 1)[sub_rev_idx]);
      }
      real reflection_coeff = fmax(-1.0 + EPS, fmin(1.0 - EPS, (c[k] - s) / v_prev));

      if (ar_order > 1) {
        vector[ar_order - 1] phi_temp = segment(phi_vec, 1, ar_order - 1);
        array[ar_order - 1] int sub_rev_idx = rev_idx[(n - (ar_order - 1) + 1):n];
        phi_vec[1:(ar_order - 1)] = phi_temp - reflection_coeff * phi_temp[sub_rev_idx];
      }
      phi_vec[ar_order] = reflection_coeff;

      v_curr = v_prev * (1.0 - square(reflection_coeff));

      if (v_curr <= 0) {
        v_curr = EPS;
      }

      log_det += log(v_curr);
      v_prev = v_curr;
    }

    return log_det;
  }

  /**
   * Computes the quadratic form z' T^{-1} z for a Toeplitz covariance
   * matrix T using the innovations algorithm.
   *
   * @param z The data vector of length n.
   * @param c The autocovariance vector (first row of the covariance matrix).
   * @return The quadratic form z' T^{-1} z.
   */
  real innovations_qf(vector z, vector c) {
    int n = rows(z);
    real EPS = 1e-12;

    if (rows(c) != n) {
      reject("z and c must have the same dimension.");
    }
    if (n == 0) {
      return 0.0;
    }
    if (c[1] <= EPS) {
      reject("Variance c[1] must be positive.");
    }

    array[n] int rev_idx;
    for (i in 1:n) {
      rev_idx[i] = n - i + 1;
    }

    real quad_form = 0.0;
    vector[n - 1] phi_vec = rep_vector(0.0, n - 1);
    real v_prev;

    real v_curr = c[1];
    quad_form += square(z[1]) / v_curr;
    v_prev = v_curr;

    for (k in 2:n) {
      int ar_order = k - 1;
      real s = 0.0;
      if (ar_order > 1) {
        array[ar_order - 1] int sub_rev_idx = rev_idx[(n - (ar_order - 1) + 1):n];
        s = dot_product(segment(phi_vec, 1, ar_order - 1),
                        segment(c, 2, ar_order - 1)[sub_rev_idx]);
      }
      real reflection_coeff = fmax(-1.0 + EPS, fmin(1.0 - EPS, (c[k] - s) / v_prev));

      if (ar_order > 1) {
        vector[ar_order - 1] phi_temp = segment(phi_vec, 1, ar_order - 1);
        array[ar_order - 1] int sub_rev_idx = rev_idx[(n - (ar_order - 1) + 1):n];
        phi_vec[1:(ar_order - 1)] = phi_temp - reflection_coeff * phi_temp[sub_rev_idx];
      }
      phi_vec[ar_order] = reflection_coeff;

      array[ar_order] int sub_rev_idx = rev_idx[(n - ar_order + 1):n];
      real pred = dot_product(segment(phi_vec, 1, ar_order),
                              segment(z, 1, ar_order)[sub_rev_idx]);
      real error = z[k] - pred;

      v_curr = v_prev * (1.0 - square(reflection_coeff));
      if (v_curr <= 0) {
        v_curr = EPS;
      }
      quad_form += square(error) / v_curr;

      v_prev = v_curr;
    }
    return quad_form;
  }

  /**
   * Vectorized log-determinant of Toeplitz correlation matrices
   * parametrized by PACF coefficients.
   *
   * @param psi_raw Array of unconstrained PACF parameter vectors
   * @return Vector of log-determinants
   */
  vector log_det_toeplitz_corr_array(array[] vector psi_raw) {
    int N = size(psi_raw);
    int D = rows(psi_raw[1]);
    if (D == 0) return rep_vector(0, N);
    int M = D + 1;
    real de = sqrt(D - 1.0);
    vector[N] ld = (M - 1) * log1m(square(tanh(to_vector(psi_raw[:, 1]) / de)));
    for (k in 2:D) {
      ld += (M - k) * log1m(square(tanh(to_vector(psi_raw[:, k]) / de)));
    }
    return ld;
  }

  /**
   * Log determinant of Jacobian for PACF to covariance transformation.
   *
   * @param psi_raw Unconstrained PACF parameters
   * @return Log determinant of Jacobian
   */
  real log_det_block_jac(vector psi_raw) {
    int D = rows(psi_raw);
    if (D == 0) return 0;
    vector[D] psi = squash_pacf(psi_raw);
    real ld = (D > 1) ? -0.5 * D * log(D - 1) : 0;
    for (k in 1:D)
      ld += (D - k + 1) * log1m(square(psi[k]));
    return ld;
  }

  /**
   * Log determinant of Jacobian for compound symmetric transformation.
   *
   * @param psiA_raw PACF parameters for within-block correlations
   * @param psiB_raw PACF parameters for between-block correlations
   * @param s Mixing parameter in (0, 1)
   * @param N_blocks Number of blocks
   * @return Log determinant of Jacobian
   */
  real log_det_jac_cs(vector psiA_raw,
                      vector psiB_raw,
                      real s,
                      int N_blocks) {
    int D = rows(psiA_raw);
    real ld = log_det_block_jac(psiA_raw) + log_det_block_jac(psiB_raw);
    ld += D * (log(s) + log1m(s));
    ld += (D + 1) * (log(N_blocks) - log(N_blocks - 1));
    return ld;
  }

  /**
   * Computes the log-determinant of a compound symmetric block-Toeplitz
   * correlation matrix.
   *
   * @param psiA_raw PACF parameters for within-block correlations
   * @param psiB_raw PACF parameters for between-block correlations
   * @param s Mixing parameter in (0, 1)
   * @param M Dimension of each block
   * @param N_blocks Number of blocks
   * @return Log-determinant of the full correlation matrix
   */
  real log_det_corr_cs(vector psiA_raw,
                       vector psiB_raw,
                       real s,
                       int M,
                       int N_blocks) {
    vector[2] log_dets = log_det_toeplitz_corr_array({psiA_raw, psiB_raw});
    real log_sigmaA2 = log(s) + log(N_blocks) - log(N_blocks - 1);
    real log_sigmaB2 = log1m(s) + log(N_blocks);
    return (N_blocks - 1) * (M * log_sigmaA2 + log_dets[1])
         +                  (M * log_sigmaB2 + log_dets[2]);
  }

  /**
   * Computes the log-PDF for an LKJ prior on a Toeplitz correlation matrix.
   *
   * @param psi_raw A vector of D unconstrained PACF parameters.
   * @param eta The shape parameter of the LKJ distribution (eta > 0).
   * @return The log-probability density, log p(T(psi_raw) | eta).
   */
  real lkj_toeplitz_lpdf(vector psi_raw, real eta) {
    int D = rows(psi_raw);
    if (D == 0) {
      return 0.0;
    }
    if (eta <= 0) {
      reject("LKJ shape parameter eta must be positive, but was ", eta);
    }

    vector[D] psi = squash_pacf(psi_raw);

    // Both log|T| and log|J| have weight (D-k+1) per PACF coefficient,
    // so the combined density simplifies to eta * (D-k+1) * log(1 - psi_k^2).
    // log|T| = sum_k (M-k) log(1-psi_k^2) where M = D+1, so M-k = D-k+1
    // log|J| = sum_k (D-k+1) log(1-psi_k^2) - 0.5*D*log(D-1)
    real lp = (D > 1) ? -0.5 * D * log(D - 1.0) : 0;
    for (k in 1:D) {
      lp += eta * (D - k + 1) * log1m(square(psi[k]));
    }

    return lp;
  }

  /**
   * Places independent LKJ priors on the two component Toeplitz correlation
   * matrices (T_A and T_B) of the compound-symmetric block-Toeplitz structure.
   *
   * @param psiA_raw PACF parameters for the within-block correlation matrix T_A.
   * @param psiB_raw PACF parameters for the between-block correlation matrix T_B.
   * @param eta_A LKJ shape parameter for T_A.
   * @param eta_B LKJ shape parameter for T_B.
   * @return The total log-probability density from both priors.
   */
  real lkj_corr_cs_lpdf(vector psiA_raw,
                        vector psiB_raw,
                        real eta_A,
                        real eta_B) {
    real lp = lkj_toeplitz_lpdf(psiA_raw | eta_A);
    lp += lkj_toeplitz_lpdf(psiB_raw | eta_B);
    return lp;
  }

  /**
   * LKJ prior for the full compound symmetric block-Toeplitz correlation matrix.
   *
   * @param psiA_raw PACF parameters for within-block correlations
   * @param psiB_raw PACF parameters for between-block correlations
   * @param s Mixing parameter in (0, 1)
   * @param eta LKJ shape parameter
   * @param M Dimension of each block
   * @param N_blocks Number of blocks
   * @return Log probability density
   */
  real lkj_corr_compound_cs_lpdf(vector psiA_raw,
                                 vector psiB_raw,
                                 real s,
                                 real eta,
                                 int M,
                                 int N_blocks) {
    int P = M * N_blocks;
    real lp = (eta - 1) * log_det_corr_cs(psiA_raw, psiB_raw, s,
                                          M, N_blocks);
    lp += log_det_jac_cs(psiA_raw, psiB_raw, s, N_blocks);
    return lp;
  }

  /**
   * Log-probability density for N i.i.d. observations from a multivariate normal
   * with a compound-symmetric block-Toeplitz covariance structure.
   *
   * Uses the innovations algorithm to avoid forming the full P x P matrix.
   *
   * @param y A matrix of observations (N_obs x P). Each row is one observation.
   * @param mu A vector of means (P x 1).
   * @param psiA_raw Unconstrained PACF parameters for T_A (M-1 x 1).
   * @param psiB_raw Unconstrained PACF parameters for T_B (M-1 x 1).
   * @param s Mixing parameter in (0, 1).
   * @param M Dimension of the inner Toeplitz blocks.
   * @param N_blocks Number of blocks. P must equal M * N_blocks.
   * @return The total log-probability density, log p(y | mu, Sigma).
   */
  real mvn_cs_block_toeplitz_dll_lpdf(matrix y,
                                  vector mu,
                                  vector psiA_raw,
                                  vector psiB_raw,
                                  real s,
                                  int M,
                                  int N_blocks) {
    int N_obs = rows(y);
    int P = cols(y);

    vector[M] rho_A = pacf_to_acf(psiA_raw);
    vector[M] rho_B = pacf_to_acf(psiB_raw);
    real sigA = s * N_blocks / (N_blocks - 1.0);
    real sigB = (1.0 - s) * N_blocks;

    vector[M] a = sigA * rho_A;
    vector[M] b = sigB * rho_B;

    real log_det_a = innovations_log_det_only(a);
    real log_det_b = innovations_log_det_only(b);
    real total_log_det_constant = N_obs * ((N_blocks - 1) * log_det_a + log_det_b);

    real total_quad_form = 0.0;
    real sqrtN_blocks = sqrt(N_blocks);

    for (i in 1:N_obs) {
      vector[P] z = y[i]' - mu;

      vector[M] y_bar_sum = rep_vector(0.0, M);
      for (n in 1:N_blocks) {
        y_bar_sum += segment(z, (n - 1) * M + 1, M);
      }
      vector[M] y_bar = y_bar_sum / N_blocks;

      total_quad_form += innovations_qf(sqrtN_blocks * y_bar, b);

      for (n in 1:N_blocks) {
        vector[M] z_tilde = segment(z, (n - 1) * M + 1, M) - y_bar;
        total_quad_form += innovations_qf(z_tilde, a);
      }
    }

    return -0.5 * (N_obs * P * log(2 * pi()) + total_log_det_constant + total_quad_form);
  }
}

data {
  int<lower=1> M_blocksize;
  int<lower=1> N_blocks;
  real eta_lkj_a;
  real eta_lkj_b;
  int N;
  int K;
  matrix[N, K] y;
  vector[K] mu;
  int<lower=0, upper=1> prior_on_full_corr;
}

transformed data {
  int D = M_blocksize - 1;
  int M_total = N_blocks * M_blocksize;
}

parameters {
  vector[D] psi_A_raw;
  vector[D] psi_B_raw;
  real<lower=0, upper=1> s;
}

model {
  if (prior_on_full_corr == 1) {
    target += lkj_corr_compound_cs_lpdf(psi_A_raw | psi_B_raw, s,
                                        max([eta_lkj_a, eta_lkj_b]), M_blocksize, N_blocks);
  } else {
    target += lkj_corr_cs_lpdf(psi_A_raw | psi_B_raw, eta_lkj_a, eta_lkj_b);
  }

  target += mvn_cs_block_toeplitz_dll_lpdf(y | mu,
                        psi_A_raw, psi_B_raw,
                        s, M_blocksize, N_blocks);
}

generated quantities {
  matrix[M_total, M_total] R = build_cs_block_toeplitz_corr(psi_A_raw, psi_B_raw, s, M_blocksize, N_blocks);
}
