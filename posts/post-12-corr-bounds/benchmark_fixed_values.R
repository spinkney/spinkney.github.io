## Validate the reduced global-ray transform with exact nonzero correlations.
## Run from this directory. Result CSVs are written alongside the post.

library(cmdstanr)

iter_warmup <- as.integer(Sys.getenv("FIXED_WARMUP", "750"))
iter_sampling <- as.integer(Sys.getenv("FIXED_SAMPLING", "1500"))
chains <- as.integer(Sys.getenv("FIXED_CHAINS", "4"))
parallel_chains <- min(
  chains,
  as.integer(Sys.getenv("FIXED_PARALLEL", "4"))
)
reference_proposals <- as.integer(
  Sys.getenv("FIXED_REFERENCE_PROPOSALS", "500000")
)

K <- 4L
eta <- 2
D <- choose(K, 2)
N_fixed <- 2L
D_free <- D - N_fixed
validation_tolerance <- 1e-10

pair_index <- do.call(rbind, lapply(2:K, function(i) {
  cbind(i = i, j = seq_len(i - 1L))
}))
pair_names <- sprintf("Omega[%d,%d]", pair_index[, "i"], pair_index[, "j"])
free_index <- matrix(c(
  2L, 1L,
  3L, 2L,
  4L, 1L,
  4L, 3L
), ncol = 2L, byrow = TRUE, dimnames = list(NULL, c("i", "j")))
free_names <- sprintf("Omega[%d,%d]", free_index[, "i"], free_index[, "j"])
fixed_index <- matrix(c(
  3L, 1L,
  4L, 2L
), ncol = 2L, byrow = TRUE, dimnames = list(NULL, c("i", "j")))
fixed_names <- sprintf(
  "Omega[%d,%d]", fixed_index[, "i"], fixed_index[, "j"]
)
fixed_values <- c(0.55, -0.35)

corr_lower <- corr_upper <- diag(K)

corr_lower[2, 1] <- corr_lower[1, 2] <- -0.70
corr_upper[2, 1] <- corr_upper[1, 2] <- 0.40
corr_lower[3, 2] <- corr_lower[2, 3] <- -0.50
corr_upper[3, 2] <- corr_upper[2, 3] <- 0.70
corr_lower[4, 1] <- corr_lower[1, 4] <- -0.60
corr_upper[4, 1] <- corr_upper[1, 4] <- 0.60
corr_lower[4, 3] <- corr_lower[3, 4] <- -0.40
corr_upper[4, 3] <- corr_upper[3, 4] <- 0.70

for (n in seq_len(N_fixed)) {
  i <- fixed_index[n, "i"]
  j <- fixed_index[n, "j"]
  corr_lower[i, j] <- corr_lower[j, i] <- fixed_values[n]
  corr_upper[i, j] <- corr_upper[j, i] <- fixed_values[n]
}

anchor <- diag(K)
for (n in seq_len(N_fixed)) {
  i <- fixed_index[n, "i"]
  j <- fixed_index[n, "j"]
  anchor[i, j] <- anchor[j, i] <- fixed_values[n]
}
anchor_min_eigenvalue <- min(
  eigen(anchor, symmetric = TRUE, only.values = TRUE)$values
)
stopifnot(anchor_min_eigenvalue > 0)

model <- cmdstan_model("global-ray-corr-bounds.stan", quiet = TRUE)
stan_data <- list(
  K = K,
  eta = eta,
  N_fixed = N_fixed,
  corr_lower = unname(corr_lower),
  corr_upper = unname(corr_upper),
  anchor = unname(anchor),
  validation_tolerance = validation_tolerance
)

set.seed(1210)
fit <- model$sample(
  data = stan_data,
  seed = 1210,
  chains = chains,
  parallel_chains = parallel_chains,
  iter_warmup = iter_warmup,
  iter_sampling = iter_sampling,
  adapt_delta = 0.90,
  max_treedepth = 12,
  refresh = 0,
  show_messages = FALSE,
  show_exceptions = FALSE,
  init = function() {
    list(direction_raw = rnorm(D_free), radius_raw = 0)
  },
  output_dir = tempdir()
)
stopifnot(all(fit$return_codes() == 0))

draws_free <- fit$draws(free_names, format = "matrix")
draws_fixed <- fit$draws(fixed_names, format = "matrix")
draws_all <- fit$draws(pair_names, format = "matrix")
free_summary <- fit$summary(free_names)
parameter_summary <- fit$summary(c("direction_raw", "radius_raw"))
all_summary <- fit$summary(c("direction_raw", "radius_raw", free_names))
sampler_summary <- fit$diagnostic_summary(quiet = TRUE)

free_lower <- vapply(seq_len(nrow(free_index)), function(n) {
  corr_lower[free_index[n, "i"], free_index[n, "j"]]
}, numeric(1))
free_upper <- vapply(seq_len(nrow(free_index)), function(n) {
  corr_upper[free_index[n, "i"], free_index[n, "j"]]
}, numeric(1))

lower_violation <- sweep(
  draws_free, 2, free_lower, FUN = function(x, bound) bound - x
)
upper_violation <- sweep(
  draws_free, 2, free_upper, FUN = function(x, bound) x - bound
)
max_bound_violation <- max(0, lower_violation, upper_violation)
max_fixed_error <- max(abs(sweep(draws_fixed, 2, fixed_values, FUN = "-")))

min_eigenvalue <- Inf
for (draw_id in seq_len(nrow(draws_all))) {
  Omega <- diag(K)
  for (n in seq_len(nrow(pair_index))) {
    i <- pair_index[n, "i"]
    j <- pair_index[n, "j"]
    Omega[i, j] <- Omega[j, i] <- draws_all[draw_id, n]
  }
  min_eigenvalue <- min(
    min_eigenvalue,
    min(eigen(Omega, symmetric = TRUE, only.values = TRUE)$values)
  )
}

# Enumerate all corners to establish that the affine box is not itself a
# subset of the elliptope; the PD ray limit is genuinely necessary.
corner_choices <- as.matrix(expand.grid(rep(list(c(0L, 1L)), D_free)))
corner_minimum_eigenvalues <- numeric(nrow(corner_choices))
for (corner_id in seq_len(nrow(corner_choices))) {
  Omega <- anchor
  for (n in seq_len(D_free)) {
    i <- free_index[n, "i"]
    j <- free_index[n, "j"]
    value <- if (corner_choices[corner_id, n] == 0L) {
      free_lower[n]
    } else {
      free_upper[n]
    }
    Omega[i, j] <- Omega[j, i] <- value
  }
  corner_minimum_eigenvalues[corner_id] <- min(
    eigen(Omega, symmetric = TRUE, only.values = TRUE)$values
  )
}
non_pd_corners <- sum(corner_minimum_eigenvalues <= 0)

# Independent reference: draw uniformly from the four-dimensional box.  For
# eta=2, accepting an SPD proposal with probability det(Omega) produces IID
# draws from the same fixed-coordinate LKJ slice because det(Omega) <= 1.
set.seed(1211)
reference <- cbind(
  runif(reference_proposals, free_lower[1], free_upper[1]),
  runif(reference_proposals, free_lower[2], free_upper[2]),
  runif(reference_proposals, free_lower[3], free_upper[3]),
  runif(reference_proposals, free_lower[4], free_upper[4])
)
colnames(reference) <- free_names

a <- reference[, 1]
c <- reference[, 2]
b <- reference[, 3]
d <- reference[, 4]
e <- fixed_values[1]
f <- fixed_values[2]

leading_minor_3 <- 1 + 2 * a * e * c - a^2 - e^2 - c^2
determinant_4 <- 1 - a^2 - b^2 - c^2 - d^2 - e^2 - f^2 +
  2 * (a * e * c + a * b * f + e * b * d + c * f * d) +
  a^2 * d^2 + b^2 * c^2 + e^2 * f^2 -
  2 * (a * e * f * d + a * b * c * d + e * b * c * f)

# Verify the vectorized determinant formula at representative proposals.
formula_check <- unique(round(seq(
  1, reference_proposals, length.out = min(100L, reference_proposals)
)))
for (proposal_id in formula_check) {
  Omega <- anchor
  for (n in seq_len(D_free)) {
    i <- free_index[n, "i"]
    j <- free_index[n, "j"]
    Omega[i, j] <- Omega[j, i] <- reference[proposal_id, n]
  }
  stopifnot(abs(det(Omega) - determinant_4[proposal_id]) < 1e-10)
}

is_spd <- leading_minor_3 > 0 & determinant_4 > 0
accepted <- is_spd &
  runif(reference_proposals) < determinant_4^(eta - 1)
reference <- reference[accepted, , drop = FALSE]
reference_accepted <- nrow(reference)
stopifnot(reference_accepted >= 10000L)

reference_summary <- data.frame(
  variable = free_names,
  reference_mean = colMeans(reference),
  reference_sd = apply(reference, 2, sd),
  reference_q05 = apply(reference, 2, quantile, probs = 0.05),
  reference_q50 = apply(reference, 2, quantile, probs = 0.50),
  reference_q95 = apply(reference, 2, quantile, probs = 0.95),
  reference_mcse_mean = apply(reference, 2, sd) /
    sqrt(reference_accepted)
)

comparison <- merge(
  data.frame(
    variable = free_names,
    stan_mean = colMeans(draws_free),
    stan_sd = apply(draws_free, 2, sd),
    stan_q05 = apply(draws_free, 2, quantile, probs = 0.05),
    stan_q50 = apply(draws_free, 2, quantile, probs = 0.50),
    stan_q95 = apply(draws_free, 2, quantile, probs = 0.95),
    stan_mcse_mean = free_summary$sd / sqrt(free_summary$ess_bulk)
  ),
  reference_summary,
  by = "variable",
  sort = FALSE
)
comparison$mean_difference <- comparison$stan_mean - comparison$reference_mean
comparison$standardized_mean_difference <- comparison$mean_difference /
  sqrt(comparison$stan_mcse_mean^2 + comparison$reference_mcse_mean^2)
comparison$max_quantile_difference <- pmax(
  abs(comparison$stan_q05 - comparison$reference_q05),
  abs(comparison$stan_q50 - comparison$reference_q50),
  abs(comparison$stan_q95 - comparison$reference_q95)
)

pd_boundary_fraction <- mean(
  fit$draws("pd_boundary_active", format = "matrix")[, 1]
)
n_gradient <- sum(fit$sampler_diagnostics()[, , "n_leapfrog__"])
diagnostics <- data.frame(
  K = K,
  eta = eta,
  N_fixed = N_fixed,
  D_free = D_free,
  retained_draws = nrow(draws_free),
  divergences = sum(sampler_summary$num_divergent),
  treedepth_hits = sum(sampler_summary$num_max_treedepth),
  min_ebfmi = min(sampler_summary$ebfmi),
  max_rhat = max(all_summary$rhat),
  min_bulk_ess = min(all_summary$ess_bulk),
  min_parameter_bulk_ess = min(parameter_summary$ess_bulk),
  ess_per_1000_gradients = 1000 * min(free_summary$ess_bulk) / n_gradient,
  max_bound_violation = max_bound_violation,
  max_fixed_error = max_fixed_error,
  min_eigenvalue = min_eigenvalue,
  anchor_min_eigenvalue = anchor_min_eigenvalue,
  non_pd_corners = non_pd_corners,
  total_corners = nrow(corner_choices),
  pd_boundary_fraction = pd_boundary_fraction,
  reference_proposals = reference_proposals,
  reference_accepted = reference_accepted,
  reference_acceptance = reference_accepted / reference_proposals,
  max_abs_mean_difference = max(abs(comparison$mean_difference)),
  max_abs_standardized_mean_difference = max(
    abs(comparison$standardized_mean_difference)
  ),
  max_quantile_difference = max(comparison$max_quantile_difference),
  seconds = fit$time()$total
)

stopifnot(
  max_bound_violation <= 1e-12,
  max_fixed_error <= 1e-12,
  min_eigenvalue > 0,
  non_pd_corners > 0,
  all(is.finite(all_summary$rhat)),
  max(all_summary$rhat) < 1.05,
  diagnostics$divergences == 0,
  diagnostics$treedepth_hits == 0
)

write.csv(comparison, "fixed-values-summary.csv", row.names = FALSE)
write.csv(diagnostics, "fixed-values-diagnostics.csv", row.names = FALSE)
write.csv(data.frame(
  K = K,
  eta = eta,
  N_fixed = N_fixed,
  D_free = D_free,
  chains = chains,
  parallel_chains = parallel_chains,
  iter_warmup = iter_warmup,
  iter_sampling = iter_sampling,
  adapt_delta = 0.90,
  max_treedepth = 12,
  reference_proposals = reference_proposals,
  cmdstan_version = as.character(cmdstan_version()),
  run_time_utc = format(Sys.time(), tz = "UTC", usetz = TRUE)
), "fixed-values-run-settings.csv", row.names = FALSE)
write.csv(data.frame(
  file = c("benchmark_fixed_values.R", "global-ray-corr-bounds.stan"),
  md5 = unname(tools::md5sum(c(
    "benchmark_fixed_values.R", "global-ray-corr-bounds.stan"
  )))
), "fixed-values-source-hashes.csv", row.names = FALSE)

cat(
  "wrote fixed-values-summary.csv, fixed-values-diagnostics.csv, ",
  "fixed-values-run-settings.csv, and fixed-values-source-hashes.csv\n",
  sep = ""
)
