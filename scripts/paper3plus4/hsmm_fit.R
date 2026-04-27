#!/usr/bin/env Rscript
# WS-P3-10: Fit a hidden semi-Markov model (HSMM) to the longitudinal NSD-ISS
# stage trajectories and compare its log-likelihood / AIC against the existing
# continuous-time Markov chain (CTMC) baseline reported in §III Markov.
#
# The CTMC assumes per-state sojourn times follow an Exponential(rate) distribution
# (memoryless). The HSMM relaxes this to allow non-exponential sojourn distributions
# (we use Gamma, which contains Exponential as a special case at shape=1).
#
# Input:  data/06_longitudinal_staging/longitudinal_nsd_iss.csv (1,900 patients,
#         16,699 visits, NSD-ISS stages 0/1/2B/3/4/5/6)
# Output: outputs/paper3plus4/hsmm/hsmm_results.json
#
# Methodology: each patient contributes a sequence of NSD-ISS stages observed at
# scheduled visits. We fit a parametric HSMM via the EM-style fitting in mhsmm
# 0.4.21, with a Gamma sojourn-time distribution per state and a generic
# multinomial transition matrix (estimated jointly with sojourn parameters).
#
# Comparison to CTMC: we report the HSMM log-likelihood, AIC, per-state
# Gamma sojourn shape parameter (k=1 means Exponential matches CTMC; k>1 means
# coxlike sojourn; k<1 means heavier tails than exponential), and per-state
# mean/median sojourn times. The CTMC's Q-matrix (from outputs/paper3_markov/
# markov_results.json) provides the per-state mean sojourn times under the
# Exponential assumption; the HSMM-vs-CTMC comparison shows whether the
# Exponential assumption is empirically supported.
#
# Run:
#   Rscript scripts/paper3plus4/hsmm_fit.R

suppressPackageStartupMessages({
  library(mhsmm)
  library(jsonlite)
})

set.seed(42)

PROJECT_ROOT <- Sys.getenv("PROJECT_ROOT", normalizePath("."))
INPUT_CSV <- file.path(PROJECT_ROOT, "data/06_longitudinal_staging/longitudinal_nsd_iss.csv")
OUTPUT_DIR <- file.path(PROJECT_ROOT, "outputs/paper3plus4/hsmm")
dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)

cat("Reading", INPUT_CSV, "...\n")
d <- read.csv(INPUT_CSV, stringsAsFactors = FALSE)
cat("  ", nrow(d), "rows,", length(unique(d$PATNO)), "unique patients\n")

# Stage encoding: 0,1,2B,3,4,5,6 -> integer states 1..7 (mhsmm uses 1-indexed)
stage_levels <- c("0", "1", "2B", "3", "4", "5", "6")
n_states <- length(stage_levels)
d$state <- match(d$nsd_stage, stage_levels)
d <- d[!is.na(d$state), ]
d <- d[order(d$PATNO, d$months_from_baseline), ]
cat("  ", nrow(d), "rows after dropping NA stages\n")

# Build sequences as a list (one element per patient)
# mhsmm hsmmfit expects an hsmm.data object with $x (concat sequence) and
# $N (vector of per-patient sequence lengths)
patnos <- unique(d$PATNO)
sequences <- vector("list", length(patnos))
seq_lens <- integer(length(patnos))
for (i in seq_along(patnos)) {
  rows <- d[d$PATNO == patnos[i], ]
  sequences[[i]] <- rows$state
  seq_lens[i] <- nrow(rows)
}

# Drop singleton sequences (HSMM needs at least 2 obs)
keep <- seq_lens >= 2
sequences <- sequences[keep]
seq_lens <- seq_lens[keep]
cat("  ", length(sequences), "patient sequences with length >= 2 for HSMM\n")

x_concat <- unlist(sequences)
hsmm_data <- list(x = x_concat, N = seq_lens)
class(hsmm_data) <- "hsmm.data"

# Fit HSMM with Gamma sojourn distribution
# Initialise with uniform transition matrix (off-diagonal) and per-state
# Gamma(shape=2, scale=2) sojourn (mean ~ 4 visits)
init_init <- rep(1 / n_states, n_states)
trans_init <- matrix(1 / (n_states - 1), n_states, n_states)
diag(trans_init) <- 0  # mhsmm does NOT allow self-transitions in HSMM (sojourn handles dwell)

# Emissions: deterministic identity (stage observed = stage state). We use a
# multinomial emission with confusion-matrix init = identity (high-confidence
# observation; HSMM contracts to plain semi-Markov on observed states).
emis_p <- matrix(0.05 / (n_states - 1), n_states, n_states)
diag(emis_p) <- 0.95
B0 <- list(p = emis_p)
class(B0) <- "list"

# Sojourn: per-state Gamma(shape, scale) with shape>=1 init
sojourn_init <- list(
  shape = rep(2, n_states),
  scale = rep(2, n_states),
  type = "gamma"
)

# Multinomial state-emission density (must be defined BEFORE hsmmspec()):
# observed stage = true state with high probability via the confusion matrix p.
dmstate <- function(x, j, model) {
  p <- model$parms.emission$p
  return(p[j, x])
}
rmstate <- function(j, model) {
  p <- model$parms.emission$p
  return(sample(seq_len(ncol(p)), 1, prob = p[j, ]))
}
mstep_mstate <- function(x, wt) {
  # Re-estimate confusion matrix from posterior weights wt (T x J) over true
  # states given observed stages x
  J <- ncol(wt)
  p <- matrix(0, J, J)
  for (j in seq_len(J)) {
    for (k in seq_len(J)) {
      p[j, k] <- sum(wt[x == k, j]) + 1e-6  # Laplace smoothing
    }
    p[j, ] <- p[j, ] / sum(p[j, ])
  }
  list(p = p)
}

start_model <- hsmmspec(
  init = init_init,
  transition = trans_init,
  parms.emission = list(p = emis_p),
  sojourn = sojourn_init,
  dens.emission = dmstate
)

cat("\nFitting HSMM (Gamma sojourn) ...\n")
t_start <- Sys.time()
fit <- tryCatch({
  hsmmfit(
    hsmm_data,
    start_model,
    mstep = mstep_mstate,
    M = max(seq_lens),
    maxit = 50
  )
}, error = function(e) {
  cat("HSMM fit error:", conditionMessage(e), "\n")
  NULL
})
elapsed <- as.numeric(difftime(Sys.time(), t_start, units = "secs"))
cat("  fit elapsed:", round(elapsed, 1), "s\n")

if (is.null(fit)) {
  res <- list(
    status = "FIT_FAILED",
    elapsed_sec = elapsed,
    n_sequences = length(sequences),
    n_observations = sum(seq_lens),
    n_states = n_states
  )
  write_json(res, file.path(OUTPUT_DIR, "hsmm_results.json"), pretty = TRUE, auto_unbox = TRUE)
  cat("Wrote", file.path(OUTPUT_DIR, "hsmm_results.json"), "(FIT_FAILED)\n")
  quit(status = 1)
}

# Extract fit summary
loglik_history <- fit$loglik
final_loglik <- loglik_history[length(loglik_history)]
n_params <- (n_states - 1) +                            # initial state probs
            (n_states * (n_states - 1)) +               # transition matrix (off-diagonal)
            (2 * n_states) +                            # gamma sojourn shape + scale
            (n_states * n_states)                       # emission confusion matrix
aic <- -2 * final_loglik + 2 * n_params

# Per-state Gamma sojourn parameters
sojourn_shape <- fit$model$sojourn$shape
sojourn_scale <- fit$model$sojourn$scale
sojourn_mean <- sojourn_shape * sojourn_scale  # gamma mean

# Compare to CTMC baseline (read from existing markov_results.json if present)
ctmc_path <- file.path(PROJECT_ROOT, "outputs/paper3_markov/markov_results.json")
ctmc_loglik <- NA
ctmc_sojourn <- NA
if (file.exists(ctmc_path)) {
  ctmc <- fromJSON(ctmc_path)
  if (!is.null(ctmc$loglik)) ctmc_loglik <- ctmc$loglik
  if (!is.null(ctmc$sojourn_times)) ctmc_sojourn <- ctmc$sojourn_times
}

res <- list(
  status = "FIT_SUCCESS",
  elapsed_sec = round(elapsed, 1),
  n_sequences = length(sequences),
  n_observations = sum(seq_lens),
  n_states = n_states,
  stage_levels = stage_levels,
  hsmm = list(
    loglik = final_loglik,
    n_params = n_params,
    aic = aic,
    iters = length(loglik_history),
    loglik_history = as.numeric(loglik_history),
    sojourn_shape = as.numeric(sojourn_shape),
    sojourn_scale = as.numeric(sojourn_scale),
    sojourn_mean_visits = as.numeric(sojourn_mean),
    transition = as.matrix(fit$model$transition),
    emission_p = as.matrix(fit$model$parms.emission$p)
  ),
  ctmc_baseline = list(
    loglik = ctmc_loglik,
    sojourn_times = ctmc_sojourn,
    note = "CTMC mean sojourn is in YEARS; HSMM sojourn_mean is in VISITS (multiply by avg visit interval ~12 months for years)."
  )
)

write_json(res, file.path(OUTPUT_DIR, "hsmm_results.json"), pretty = TRUE, auto_unbox = TRUE)
cat("\n=== HSMM Fit Summary ===\n")
cat("  log-lik:", round(final_loglik, 1), "\n")
cat("  AIC:    ", round(aic, 1), "\n")
cat("  iters:  ", length(loglik_history), "\n")
for (i in seq_len(n_states)) {
  cat(sprintf("  Stage %s: gamma shape=%.2f scale=%.2f mean=%.2f visits\n",
              stage_levels[i], sojourn_shape[i], sojourn_scale[i], sojourn_mean[i]))
}
cat("Wrote", file.path(OUTPUT_DIR, "hsmm_results.json"), "\n")
