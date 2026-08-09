## Reproducible experiments for the sign and arbitrary-box transforms.
## Run from this directory. Result CSVs are written alongside the post.

library(cmdstanr)

iter_warmup <- as.integer(Sys.getenv("BOUNDS_WARMUP", "750"))
iter_sampling <- as.integer(Sys.getenv("BOUNDS_SAMPLING", "1500"))
chains <- as.integer(Sys.getenv("BOUNDS_CHAINS", "4"))
parallel_chains <- min(chains, as.integer(Sys.getenv("BOUNDS_PARALLEL", "4")))

K <- 5L
eta <- 2
D <- choose(K, 2)
validation_tolerance <- 1e-10

pair_index <- do.call(rbind, lapply(2:K, function(i) {
  cbind(i = i, j = seq_len(i - 1L))
}))
pair_names <- sprintf("Omega[%d,%d]", pair_index[, "i"], pair_index[, "j"])

equicorrelation <- function(K, rho) {
  out <- matrix(rho, K, K)
  diag(out) <- 1
  out
}

sign_bounds <- function(sign_matrix) {
  lower <- upper <- diag(1, K)
  for (n in seq_len(nrow(pair_index))) {
    i <- pair_index[n, "i"]
    j <- pair_index[n, "j"]
    if (sign_matrix[i, j] > 0) {
      lower[i, j] <- lower[j, i] <- 0
      upper[i, j] <- upper[j, i] <- 1
    } else {
      lower[i, j] <- lower[j, i] <- -1
      upper[i, j] <- upper[j, i] <- 0
    }
  }
  list(lower = lower, upper = upper)
}

all_positive_signs <- matrix(1, K, K)
diag(all_positive_signs) <- 0

all_negative_signs <- matrix(-1, K, K)
diag(all_negative_signs) <- 0

set.seed(1203)
mixed_signs <- matrix(0, K, K)
mixed_values <- sample(c(-1, 1), D, replace = TRUE)
mixed_signs[lower.tri(mixed_signs)] <- mixed_values
mixed_signs <- mixed_signs + t(mixed_signs)

positive_bounds <- sign_bounds(all_positive_signs)
negative_bounds <- sign_bounds(all_negative_signs)
mixed_bounds <- sign_bounds(mixed_signs)

# Strict interior anchors for the three sign-constrained elliptopes.
positive_anchor <- equicorrelation(K, 0.20)
negative_anchor <- equicorrelation(K, -0.10)
mixed_anchor <- diag(K) + 0.08 * mixed_signs
stopifnot(min(eigen(mixed_anchor, symmetric = TRUE, only.values = TRUE)$values) > 0)

# A mixed-sign anchor followed by asymmetric random widths.  The widths are
# rescaled so every matrix in the resulting box is positive definite:
# ||E||_2 <= ||E||_infinity < lambda_min(anchor).
set.seed(1204)
anchor_direction <- matrix(0, K, K)
anchor_direction[lower.tri(anchor_direction)] <- runif(D, -1, 1)
anchor_direction <- anchor_direction + t(anchor_direction)
spectral_radius <- max(abs(eigen(
  anchor_direction, symmetric = TRUE, only.values = TRUE
)$values))
heterogeneous_anchor <- diag(K) + 0.45 * anchor_direction / spectral_radius
anchor_min_eigenvalue <- min(eigen(
  heterogeneous_anchor, symmetric = TRUE, only.values = TRUE
)$values)

width_lower <- width_upper <- matrix(0, K, K)
width_lower[lower.tri(width_lower)] <- runif(D, 0.03, 0.18)
width_upper[lower.tri(width_upper)] <- runif(D, 0.03, 0.18)
width_lower <- width_lower + t(width_lower)
width_upper <- width_upper + t(width_upper)
max_perturbation <- pmax(width_lower, width_upper)
diag(max_perturbation) <- 0
raw_row_radius <- max(rowSums(max_perturbation))
width_scale <- 0.60 * anchor_min_eigenvalue / raw_row_radius
width_lower <- width_scale * width_lower
width_upper <- width_scale * width_upper

heterogeneous_lower <- heterogeneous_anchor - width_lower
heterogeneous_upper <- heterogeneous_anchor + width_upper
diag(heterogeneous_lower) <- 1
diag(heterogeneous_upper) <- 1
certified_row_radius <- max(rowSums(pmax(width_lower, width_upper)))
certified_eigenvalue_floor <- anchor_min_eigenvalue - certified_row_radius
stopifnot(certified_eigenvalue_floor > 0)

# A genuinely wide heterogeneous box around the same anchor.  It remains
# nonempty because the anchor is strict, but some of its 2^D corners are not
# positive definite, so the global ray must sometimes stop at the PD boundary.
set.seed(1205)
wide_lower_width <- wide_upper_width <- matrix(0, K, K)
wide_lower_width[lower.tri(wide_lower_width)] <- runif(D, 0.25, 0.65)
wide_upper_width[lower.tri(wide_upper_width)] <- runif(D, 0.25, 0.65)
wide_lower_width <- wide_lower_width + t(wide_lower_width)
wide_upper_width <- wide_upper_width + t(wide_upper_width)
wide_heterogeneous_lower <- pmax(
  heterogeneous_anchor - wide_lower_width, -0.95
)
wide_heterogeneous_upper <- pmin(
  heterogeneous_anchor + wide_upper_width, 0.95
)
diag(wide_heterogeneous_lower) <- 1
diag(wide_heterogeneous_upper) <- 1

corner_choices <- as.matrix(expand.grid(rep(list(c(0L, 1L)), D)))
minimum_corner_eigenvalue <- Inf
for (corner_id in seq_len(nrow(corner_choices))) {
  corner <- diag(K)
  for (n in seq_len(nrow(pair_index))) {
    i <- pair_index[n, "i"]
    j <- pair_index[n, "j"]
    value <- if (corner_choices[corner_id, n] == 0L) {
      wide_heterogeneous_lower[i, j]
    } else {
      wide_heterogeneous_upper[i, j]
    }
    corner[i, j] <- corner[j, i] <- value
  }
  minimum_corner_eigenvalue <- min(
    minimum_corner_eigenvalue,
    min(eigen(corner, symmetric = TRUE, only.values = TRUE)$values)
  )
}
stopifnot(minimum_corner_eigenvalue < 0)

scenarios <- list(
  all_positive = list(
    lower = positive_bounds$lower,
    upper = positive_bounds$upper,
    anchor = positive_anchor,
    certificate = NA_real_,
    minimum_corner_eigenvalue = NA_real_
  ),
  all_negative = list(
    lower = negative_bounds$lower,
    upper = negative_bounds$upper,
    anchor = negative_anchor,
    certificate = NA_real_,
    minimum_corner_eigenvalue = NA_real_
  ),
  mixed_signs = list(
    lower = mixed_bounds$lower,
    upper = mixed_bounds$upper,
    anchor = mixed_anchor,
    certificate = NA_real_,
    minimum_corner_eigenvalue = NA_real_
  ),
  heterogeneous_box = list(
    lower = heterogeneous_lower,
    upper = heterogeneous_upper,
    anchor = heterogeneous_anchor,
    certificate = certified_eigenvalue_floor,
    minimum_corner_eigenvalue = NA_real_
  ),
  wide_heterogeneous_box = list(
    lower = wide_heterogeneous_lower,
    upper = wide_heterogeneous_upper,
    anchor = heterogeneous_anchor,
    certificate = NA_real_,
    minimum_corner_eigenvalue = minimum_corner_eigenvalue
  )
)

model_files <- c(
  tri_sign = "tri-sign-corr-bounds.stan",
  global_ray = "global-ray-corr-bounds.stan",
  direct_box = "direct-certified-box-corr.stan"
)
models <- lapply(model_files, cmdstan_model, quiet = TRUE)

write.csv(data.frame(
  K = K,
  eta = eta,
  chains = chains,
  parallel_chains = parallel_chains,
  iter_warmup = iter_warmup,
  iter_sampling = iter_sampling,
  adapt_delta = 0.90,
  max_treedepth = 12,
  cmdstan_version = as.character(cmdstan_version()),
  run_time_utc = format(Sys.time(), tz = "UTC", usetz = TRUE)
), "bounds-run-settings.csv", row.names = FALSE)

source_files <- c("benchmark_bounds.R", unname(model_files))
write.csv(data.frame(
  file = source_files,
  md5 = unname(tools::md5sum(source_files))
), "bounds-source-hashes.csv", row.names = FALSE)

experiments <- data.frame(
  scenario = c(
    "all_positive", "all_positive",
    "all_negative", "all_negative",
    "mixed_signs", "mixed_signs",
    "heterogeneous_box", "heterogeneous_box",
    "wide_heterogeneous_box"
  ),
  model = c(
    "tri_sign", "global_ray",
    "tri_sign", "global_ray",
    "tri_sign", "global_ray",
    "direct_box", "global_ray",
    "global_ray"
  )
)

diagnostic_rows <- list()
cell_rows <- list()
selected_rows <- list()

for (experiment_id in seq_len(nrow(experiments))) {
  scenario_name <- experiments$scenario[experiment_id]
  model_name <- experiments$model[experiment_id]
  scenario <- scenarios[[scenario_name]]

  stan_data <- list(
    K = K,
    eta = eta,
    corr_lower = unname(scenario$lower),
    corr_upper = unname(scenario$upper),
    validation_tolerance = validation_tolerance
  )
  if (model_name == "global_ray") {
    stan_data$N_fixed <- 0L
    stan_data$anchor <- unname(scenario$anchor)
  }

  init <- switch(
    model_name,
    tri_sign = 0,
    direct_box = 0,
    global_ray = function() {
      list(direction_raw = rnorm(D), radius_raw = 0)
    }
  )

  message("sampling ", scenario_name, " / ", model_name)
  fit <- models[[model_name]]$sample(
    data = stan_data,
    seed = 9100 + experiment_id,
    chains = chains,
    parallel_chains = parallel_chains,
    iter_warmup = iter_warmup,
    iter_sampling = iter_sampling,
    adapt_delta = 0.90,
    max_treedepth = 12,
    refresh = 0,
    show_messages = FALSE,
    show_exceptions = FALSE,
    init = init,
    output_dir = tempdir()
  )
  stopifnot(all(fit$return_codes() == 0))

  diagnostics <- fit$diagnostic_summary(quiet = TRUE)
  draws <- fit$draws(pair_names, format = "matrix")
  fit_summary <- fit$summary(pair_names)
  parameter_variables <- switch(
    model_name,
    tri_sign = "raw",
    direct_box = "raw",
    global_ray = c("direction_raw", "radius_raw")
  )
  parameter_summary <- fit$summary(parameter_variables)
  # lower.tri uses column-major order; match the explicit row-major variables.
  lower_values <- vapply(seq_len(nrow(pair_index)), function(n) {
    scenario$lower[pair_index[n, "i"], pair_index[n, "j"]]
  }, numeric(1))
  upper_values <- vapply(seq_len(nrow(pair_index)), function(n) {
    scenario$upper[pair_index[n, "i"], pair_index[n, "j"]]
  }, numeric(1))

  lower_violation <- sweep(draws, 2, lower_values, FUN = function(x, b) b - x)
  upper_violation <- sweep(draws, 2, upper_values, FUN = function(x, b) x - b)
  max_bound_violation <- max(0, lower_violation, upper_violation)

  min_eigenvalue <- Inf
  for (draw_id in seq_len(nrow(draws))) {
    Omega <- diag(K)
    for (n in seq_len(nrow(pair_index))) {
      i <- pair_index[n, "i"]
      j <- pair_index[n, "j"]
      Omega[i, j] <- Omega[j, i] <- draws[draw_id, n]
    }
    min_eigenvalue <- min(
      min_eigenvalue,
      min(eigen(Omega, symmetric = TRUE, only.values = TRUE)$values)
    )
  }

  stopifnot(
    is.finite(max_bound_violation),
    max_bound_violation <= 1e-12,
    is.finite(min_eigenvalue),
    min_eigenvalue > 0,
    all(is.finite(parameter_summary$rhat)),
    max(parameter_summary$rhat) < 1.05
  )
  if (is.finite(scenario$certificate)) {
    stopifnot(min_eigenvalue >= scenario$certificate - 1e-10)
  }

  pd_boundary_fraction <- if (model_name == "global_ray") {
    mean(fit$draws("pd_boundary_active", format = "matrix")[, 1])
  } else {
    NA_real_
  }
  radius_rhat <- if (model_name == "global_ray") {
    fit$summary("radius_raw")$rhat
  } else {
    NA_real_
  }

  n_gradient <- sum(fit$sampler_diagnostics()[, , "n_leapfrog__"])
  diagnostic_rows[[length(diagnostic_rows) + 1L]] <- data.frame(
    scenario = scenario_name,
    model = model_name,
    divergences = sum(diagnostics$num_divergent),
    treedepth_hits = sum(diagnostics$num_max_treedepth),
    min_ebfmi = min(diagnostics$ebfmi),
    min_bulk_ess = min(fit_summary$ess_bulk),
    min_tail_ess = min(fit_summary$ess_tail),
    max_rhat = max(fit_summary$rhat),
    max_parameter_rhat = max(parameter_summary$rhat),
    min_parameter_bulk_ess = min(parameter_summary$ess_bulk),
    radius_rhat = radius_rhat,
    ess_per_1000_gradients = 1000 * min(fit_summary$ess_bulk) / n_gradient,
    seconds = fit$time()$total,
    max_bound_violation = max_bound_violation,
    min_eigenvalue = min_eigenvalue,
    certified_eigenvalue_floor = scenario$certificate,
    minimum_corner_eigenvalue = scenario$minimum_corner_eigenvalue,
    pd_boundary_fraction = pd_boundary_fraction
  )

  cell_rows[[length(cell_rows) + 1L]] <- data.frame(
    scenario = scenario_name,
    model = model_name,
    variable = pair_names,
    row = pair_index[, "i"],
    col = pair_index[, "j"],
    lower = lower_values,
    upper = upper_values,
    mean = colMeans(draws),
    sd = apply(draws, 2, sd),
    q05 = apply(draws, 2, quantile, probs = 0.05),
    q50 = apply(draws, 2, quantile, probs = 0.50),
    q95 = apply(draws, 2, quantile, probs = 0.95),
    mcse_mean = fit_summary$sd / sqrt(fit_summary$ess_bulk)
  )

  selected_draw_ids <- unique(round(seq(
    1, nrow(draws),
    length.out = min(1500L, nrow(draws))
  )))
  selected_rows[[length(selected_rows) + 1L]] <- data.frame(
    scenario = scenario_name,
    model = model_name,
    draw = selected_draw_ids,
    value = unname(draws[selected_draw_ids, 1])
  )

  write.csv(
    do.call(rbind, diagnostic_rows),
    "bounds-diagnostics.csv", row.names = FALSE
  )
  write.csv(
    do.call(rbind, cell_rows),
    "bounds-cell-summary.csv", row.names = FALSE
  )
  write.csv(
    do.call(rbind, selected_rows),
    "bounds-selected-draws.csv", row.names = FALSE
  )
}

bound_rows <- lapply(names(scenarios), function(scenario_name) {
  scenario <- scenarios[[scenario_name]]
  data.frame(
    scenario = scenario_name,
    row = pair_index[, "i"],
    col = pair_index[, "j"],
    lower = vapply(seq_len(nrow(pair_index)), function(n) {
      scenario$lower[pair_index[n, "i"], pair_index[n, "j"]]
    }, numeric(1)),
    anchor = vapply(seq_len(nrow(pair_index)), function(n) {
      scenario$anchor[pair_index[n, "i"], pair_index[n, "j"]]
    }, numeric(1)),
    upper = vapply(seq_len(nrow(pair_index)), function(n) {
      scenario$upper[pair_index[n, "i"], pair_index[n, "j"]]
    }, numeric(1)),
    certified_eigenvalue_floor = scenario$certificate,
    minimum_corner_eigenvalue = scenario$minimum_corner_eigenvalue
  )
})
write.csv(do.call(rbind, bound_rows), "bounds-scenarios.csv", row.names = FALSE)

cat("wrote bounds-diagnostics.csv, bounds-cell-summary.csv, ",
    "bounds-selected-draws.csv, bounds-scenarios.csv, and run metadata\n",
    sep = "")
