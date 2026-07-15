## Expanded benchmark over K and eta for the blog post.
## Writes sweep-results.csv into the blog post folder.

library(cmdstanr)

out_csv <- Sys.getenv("SWEEP_OUT", "sweep-results.csv")

sweep_models <- c(
  carpenter = "carpenter-corr-zeros.stan",
  pinkney   = "pinkney-corr-zeros2.stan",
  ldl2      = "pinkney-corr-zeros-ldl2.stan",
  ldl3      = "pinkney-corr-zeros-ldl3.stan",
  cvine     = "cvine-corr-zeros.stan",
  schur     = "pinkney-corr-zeros-schur.stan",
  qr2       = "qr2-corr-zeros.stan",
  tri       = "unitvec-tri-corr-zeros.stan"
)
mods <- lapply(sweep_models, cmdstan_model)

Ks <- c(5, 7, 9, 15, 25)
etas <- c(1, 2, 3, 4)

make_pattern <- function(K, seed) {
  set.seed(seed)
  zeros <- matrix(0, K, K)
  zeros[lower.tri(zeros)] <- rbinom(choose(K, 2), 1, 0.5)
  zi <- which(zeros == 1, arr.ind = TRUE)
  zi <- zi[order(zi[, 1], zi[, 2]), , drop = FALSE]
  list(zeros = zeros, index = zi, n = sum(zeros))
}

rows <- list()
for (K in Ks) {
  pat <- make_pattern(K, seed = K)
  low <- lower.tri(pat$zeros)
  free <- pat$zeros[low] == 0
  for (eta in etas) {
    for (m in names(mods)) {
      res <- tryCatch({
        fit <- mods[[m]]$sample(
          data = list(D = K, K = K, eta = eta, N_zero = pat$n, zeros = pat$index),
          seed = 1, parallel_chains = 4,
          iter_warmup = 1000, iter_sampling = 2000,
          refresh = 0, show_messages = FALSE, show_exceptions = FALSE
        )
        div <- sum(fit$diagnostic_summary(quiet = TRUE)$num_divergent)
        ess <- min(matrix(fit$summary("Omega")$ess_bulk, K, K)[low][free])
        ngrad <- sum(fit$sampler_diagnostics()[, , "n_leapfrog__"])
        data.frame(K = K, eta = eta, model = m, N_zero = pat$n,
                   divergences = div, min_ess = ess,
                   ess_per_1k_grad = 1000 * ess / ngrad,
                   seconds = fit$time()$total, failed = FALSE)
      }, error = function(e) {
        data.frame(K = K, eta = eta, model = m, N_zero = pat$n,
                   divergences = NA, min_ess = NA, ess_per_1k_grad = NA,
                   seconds = NA, failed = TRUE)
      })
      rows[[length(rows) + 1]] <- res
      cat(sprintf("done K=%d eta=%d %-9s %s\n", K, eta, m,
                  if (res$failed) "FAILED"
                  else sprintf("div=%d ess=%.0f", res$divergences, res$min_ess)))
    }
  }
}
out <- do.call(rbind, rows)
write.csv(out, out_csv, row.names = FALSE)
cat("wrote", out_csv, "\n")
