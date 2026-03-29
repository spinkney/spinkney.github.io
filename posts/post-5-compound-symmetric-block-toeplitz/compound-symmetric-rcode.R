library(cmdstanr)
library(mvtnorm)

model <- cmdstan_model("compound_symmetric_block_toeplitz.stan", force_recompile = T)

#####  
##### Functions to simulate
#####

squash_pacf <- function(psi_raw) {
  D <- length(psi_raw)
  if (D <= 1) {
    return(tanh(psi_raw))
  }
  return(tanh(psi_raw / sqrt(D - 1)))
}

# Function to convert PACF to ACF using Durbin-Levinson
pacf_to_acf <- function(psi_raw) {
  D <- length(psi_raw)
  rho <- numeric(D + 1)
  rho[1] <- 1.0
  
  if (D == 0) {
    return(rho)
  }
  
  psi <- squash_pacf(psi_raw)
  phi <- numeric(D)
  
  for (k in 1:D) {
    psi_k <- psi[k]
    phi_k_k <- psi_k
    
    if (k > 1) {
      phi_prev <- phi[1:(k-1)]
      for (j in 1:(k-1)) {
        phi[j] <- phi_prev[j] - psi_k * phi_prev[k - j]
      }
    }
    
    phi[k] <- phi_k_k
    rho[k + 1] <- sum(phi[1:k] * rev(rho[1:k]))
  }
  
  return(rho)
}

build_toeplitz <- function(r) {
  n <- length(r)
  outer(seq_len(n), seq_len(n), function(i, j) r[abs(i - j) + 1])
}

build_full_R <- function(psiA, psiB, s, M, N) {
  
  rhoA <- pacf_to_acf(psiA)
  rhoB <- pacf_to_acf(psiB)
  
  sigmaA <- s * N / (N - 1)
  sigmaB <- (1 - s) * N
  a <- sigmaA * rhoA
  b <- sigmaB * rhoB
  
  r_on  <- ((N - 1) * a + b) / N
  r_off <- (b - a) / N
  
  T_on  <- build_toeplitz(r_on)
  T_off <- build_toeplitz(r_off)
  
  P <- M * N
  R <- matrix(0, P, P)
  for (i in 0:(N - 1))
    for (j in 0:(N - 1)) {
      idx_i <- (i * M + 1):((i + 1) * M)
      idx_j <- (j * M + 1):((j + 1) * M)
      R[idx_i, idx_j] <- if (i == j) T_on else T_off
    }
  R
}

#####  
##### Simulation
#####

# Set random seed for reproducibility
set.seed(42)

# Define simulation parameters
M_blocksize <- 5    # Size of each Toeplitz block
N_blocks <- 2      # Number of blocks
D <- M_blocksize - 1  # Order of AR process
M_total <- N_blocks * M_blocksize  # Total dimension
N <- 20           # Number of observations
K <- M_total       # Dimension of each observation
eta_lkj_a <- 4
eta_lkj_b <- 2

# True parameter values for simulation
true_s <- 0.7  # Mixing parameter

# Generate true PACF parameters (unconstrained)
# These will be transformed to [-1, 1] via tanh
true_psi_A_raw <- rnorm(D, mean = 0, sd = 1.5)  # Within-block correlations
true_psi_B_raw <- rnorm(D, mean = 0, sd = 1)  # Between-block correlations

# Build the true correlation matrix
R_true <- build_full_R(true_psi_A_raw, true_psi_B_raw, true_s, 
                       M_blocksize, N_blocks)

# Generate multivariate normal data
y_row <- rmvnorm(N, sigma = R_true)   # row-major order

# Create Stan data list
stan_data <- list(
  M_blocksize = M_blocksize,
  N_blocks = N_blocks,
  eta_lkj_a = eta_lkj_a,
  eta_lkj_b = eta_lkj_b,
  N = N,
  K = K,
  y = y_row,
  mu = rep(0, K),
  prior_on_full_corr = 0
)

fit <- model$sample(data = stan_data,
                    iter_warmup = 500, 
                    iter_sampling = 500,
                    # init = 0.5,
                    parallel_chains = 4)
true_params <- list(
  psi_A_raw = true_psi_A_raw,
  psi_B_raw = true_psi_B_raw,
  s = true_s,
  R = R_true
)

# just look at the first 10 x 10 block of the matrix for inspection
true_params$R[1:10, 1:10]
matrix(fit$summary("R")$mean, K, K)[1:10, 1:10]