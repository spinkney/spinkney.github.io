## Compare one all-pairs 3 x 3 propagation pass with the same transform off.
## Run from this directory. Writes forced-lookahead-results.csv by default.

library(cmdstanr)

out_csv <- Sys.getenv("FORCED_LOOKAHEAD_OUT", "forced-lookahead-results.csv")
iter_warmup <- as.integer(Sys.getenv("FORCED_LOOKAHEAD_WARMUP", "1000"))
iter_sampling <- as.integer(Sys.getenv("FORCED_LOOKAHEAD_SAMPLING", "2000"))
pattern_ids <- seq_len(as.integer(Sys.getenv("FORCED_LOOKAHEAD_PATTERNS", "5")))

K <- 7L
eta <- 4

model_files <- c(
  carpenter = "carpenter-lookahead-corr-zeros.stan",
  pinkney = "pinkney-lookahead-corr-zeros.stan",
  ldl2 = "ldl2-lookahead-corr-zeros.stan",
  ldl3 = "ldl3-lookahead-corr-zeros.stan",
  rowscale = "rowscale-lookahead-corr-zeros.stan",
  schur = "schur-lookahead-corr-zeros.stan"
)
models <- lapply(model_files, cmdstan_model)

make_pattern <- function(K, seed) {
  set.seed(seed)
  zeros <- matrix(0L, K, K)
  zeros[lower.tri(zeros)] <- rbinom(choose(K, 2), 1, 0.5)
  index <- which(zeros == 1L, arr.ind = TRUE)
  index <- index[order(index[, 1], index[, 2]), , drop = FALSE]
  list(matrix = zeros, index = index, n = nrow(index))
}

rows <- list()
for (pattern_id in pattern_ids) {
  pattern <- make_pattern(K, pattern_id)
  lower <- lower.tri(pattern$matrix)
  free <- pattern$matrix[lower] == 0

  for (model_name in names(models)) {
    for (passes in 0:1) {
      result <- tryCatch({
        fit <- models[[model_name]]$sample(
          data = list(
            K = K,
            eta = eta,
            N_zero = pattern$n,
            zeros = pattern$index,
            propagation_passes = passes
          ),
          seed = 1,
          chains = 4,
          parallel_chains = 4,
          iter_warmup = iter_warmup,
          iter_sampling = iter_sampling,
          refresh = 0,
          show_messages = FALSE,
          show_exceptions = FALSE,
          output_dir = tempdir()
        )
        if (!all(fit$return_codes() == 0)) {
          stop("one or more chains failed")
        }

        diagnostics <- fit$diagnostic_summary(quiet = TRUE)
        min_ess <- min(matrix(fit$summary("Omega")$ess_bulk, K, K)[lower][free])
        n_grad <- sum(fit$sampler_diagnostics()[, , "n_leapfrog__"])
        zero_names <- sprintf("Omega[%d,%d]",
                              pattern$index[, 1], pattern$index[, 2])
        max_abs_zero_mean <- if (length(zero_names)) {
          max(abs(fit$summary(zero_names)$mean))
        } else {
          0
        }

        data.frame(
          pattern = pattern_id,
          model = model_name,
          propagation_passes = passes,
          N_zero = pattern$n,
          divergences = sum(diagnostics$num_divergent),
          min_ess = min_ess,
          ess_per_1k_grad = 1000 * min_ess / n_grad,
          seconds = fit$time()$total,
          max_abs_zero_mean = max_abs_zero_mean,
          failed = FALSE,
          error = NA_character_
        )
      }, error = function(e) {
        data.frame(
          pattern = pattern_id,
          model = model_name,
          propagation_passes = passes,
          N_zero = pattern$n,
          divergences = NA_integer_,
          min_ess = NA_real_,
          ess_per_1k_grad = NA_real_,
          seconds = NA_real_,
          max_abs_zero_mean = NA_real_,
          failed = TRUE,
          error = conditionMessage(e)
        )
      })

      rows[[length(rows) + 1]] <- result
      output <- do.call(rbind, rows)
      write.csv(output, out_csv, row.names = FALSE)
      cat(sprintf(
        "pattern=%d %-9s pass=%d %s\n",
        pattern_id,
        model_name,
        passes,
        if (result$failed) paste("FAILED:", result$error)
        else sprintf("div=%d min_ess=%.0f eff=%.1f sec=%.2f",
                     result$divergences, result$min_ess,
                     result$ess_per_1k_grad, result$seconds)
      ))
    }
  }
}

cat("wrote", out_csv, "\n")
