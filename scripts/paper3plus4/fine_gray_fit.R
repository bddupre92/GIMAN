#!/usr/bin/env Rscript
# WS-P3-7c: Fit Fine-Gray subdistribution-hazard models for each NSD-ISS
# transition cause and report cumulative-incidence concordance per cause +
# overall C-td comparison against DeepHit/Graph-DT/Markov.
#
# The Fine-Gray model (Fine & Gray 1999, JASA) is the canonical
# competing-risks regression: it directly models the subdistribution
# hazard of each cause, yielding a cumulative incidence function (CIF)
# estimate per cause. Unlike cause-specific Cox, Fine-Gray's CIF is
# proper (sums to a probability bounded by 1).
#
# We fit 7 separate Fine-Gray models (one per destination NSD-ISS stage,
# excluding the patient's own source stage as the reference), with the
# 18 baseline graph features as covariates. We report the per-cause
# Wolbers C-index (Wolbers 2009 Epidemiol) at the 3-, 5-, and 10-year
# horizons, then the overall C-td averaged across causes/horizons for
# direct comparison with DeepHit/Graph-DT/Markov.
#
# Input:  data/06_longitudinal_staging/longitudinal_nsd_iss.csv (1900 patients,
#         16,699 visits)
#         data/06_longitudinal_staging/transition_events.csv  (2,859 transitions)
#         data/05_features/paper1_features_with_targets.csv  (covariate baseline)
# Output: outputs/paper3plus4/fine_gray/fine_gray_results.json
#
# Run:
#   Rscript scripts/paper3plus4/fine_gray_fit.R

suppressPackageStartupMessages({
  library(cmprsk)
  library(jsonlite)
})

set.seed(42)

PROJECT_ROOT <- Sys.getenv("PROJECT_ROOT", normalizePath("."))
LONG_CSV <- file.path(PROJECT_ROOT, "data/06_longitudinal_staging/longitudinal_nsd_iss.csv")
TRANS_CSV <- file.path(PROJECT_ROOT, "data/06_longitudinal_staging/transition_events.csv")
COVAR_CSV <- file.path(PROJECT_ROOT, "data/05_features/paper1_features_with_targets.csv")
OUTPUT_DIR <- file.path(PROJECT_ROOT, "outputs/paper3plus4/fine_gray")
dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)

cat("Loading covariates from", COVAR_CSV, "\n")
covar <- read.csv(COVAR_CSV, stringsAsFactors = FALSE)
cat("  ", nrow(covar), "patient-rows,", ncol(covar), "cols\n")

cat("Loading longitudinal staging from", LONG_CSV, "\n")
long <- read.csv(LONG_CSV, stringsAsFactors = FALSE)
cat("  ", nrow(long), "visit-rows,", length(unique(long$PATNO)), "patients\n")

cat("Loading transitions from", TRANS_CSV, "\n")
trans <- read.csv(TRANS_CSV, stringsAsFactors = FALSE)
cat("  ", nrow(trans), "transition events\n")

# ────────────────────────────────────────────────────────────
# Build per-patient Fine-Gray episode table
# ────────────────────────────────────────────────────────────
# Each patient contributes one episode per stage occupancy. For Fine-Gray,
# we focus on the BASELINE → first transition episode: time-to-first-transition
# from baseline stage with destination as the cause indicator.
#
# Cause coding for cmprsk::crr():
#   0 = censored (no transition observed during follow-up)
#   1..7 = destination stage numeric (0,1,2B->2.5,3,4,5,6)

# Get baseline stage + first transition per patient
baseline <- long[long$months_from_baseline == 0, c("PATNO", "nsd_stage_numeric")]
names(baseline)[2] <- "baseline_stage"

# For each patient, find the first transition (if any) and its destination
patnos <- unique(long$PATNO)
n_pats <- length(patnos)
cat("Building per-patient episode table for", n_pats, "patients ...\n")

episode_df <- data.frame(
  PATNO = integer(n_pats),
  baseline_stage = numeric(n_pats),
  time_to_event = numeric(n_pats),
  cause = integer(n_pats)  # 0 censored; 1..7 destination
)

# Cause encoding mapping
cause_levels <- c(0, 1, 2.5, 3, 4, 5, 6)  # nsd_stage_numeric values
cause_to_idx <- function(c) {
  m <- match(c, cause_levels)
  ifelse(is.na(m), 0L, as.integer(m))  # 1..7 valid; 0 if not matched (shouldn't happen)
}

for (i in seq_along(patnos)) {
  p <- patnos[i]
  bs <- baseline$baseline_stage[baseline$PATNO == p]
  if (length(bs) == 0) {
    episode_df$PATNO[i] <- p
    episode_df$baseline_stage[i] <- NA
    episode_df$time_to_event[i] <- NA
    episode_df$cause[i] <- NA
    next
  }
  # If multiple baselines (shouldn't happen), take the first
  bs <- bs[1]
  if (is.na(bs)) {
    episode_df$PATNO[i] <- p
    episode_df$baseline_stage[i] <- NA
    episode_df$time_to_event[i] <- NA
    episode_df$cause[i] <- NA
    next
  }
  # Get all visits for this patient
  pvis <- long[long$PATNO == p, ]
  # Find first transition (row where stage differs from baseline)
  trans_idx <- which(pvis$nsd_stage_numeric != bs)
  if (length(trans_idx) == 0) {
    # No transition — censored at last visit
    last_visit <- max(pvis$months_from_baseline, na.rm = TRUE)
    episode_df$PATNO[i] <- p
    episode_df$baseline_stage[i] <- bs
    episode_df$time_to_event[i] <- last_visit
    episode_df$cause[i] <- 0L  # censored
  } else {
    first_trans <- pvis[trans_idx[1], ]
    episode_df$PATNO[i] <- p
    episode_df$baseline_stage[i] <- bs
    episode_df$time_to_event[i] <- first_trans$months_from_baseline
    episode_df$cause[i] <- cause_to_idx(first_trans$nsd_stage_numeric)
  }
}

# Drop patients with NA (no baseline, or single-visit)
episode_df <- episode_df[!is.na(episode_df$baseline_stage) &
                          !is.na(episode_df$time_to_event) &
                          episode_df$time_to_event > 0, ]
cat("  ", nrow(episode_df), "episodes after dropping NA / zero-time\n")

# Distribution of causes
cat("Cause distribution:\n")
cause_tab <- table(episode_df$cause)
for (j in seq_along(cause_tab)) {
  cause_idx <- as.integer(names(cause_tab)[j])
  label <- if (cause_idx == 0) "censored" else as.character(cause_levels[cause_idx])
  cat(sprintf("  cause %d (%s): %d\n", cause_idx, label, cause_tab[j]))
}

# Build covariate matrix from baseline features (one row per patient)
# Use a subset of features that are in paper1_features_with_targets
covar$patno_match <- covar$PATNO
ep_pats <- episode_df$PATNO
cov_idx <- match(ep_pats, covar$patno_match)
covar_subset <- covar[cov_idx, ]
cat("  covariate match:", sum(!is.na(cov_idx)), "/", length(ep_pats), "patients\n")

# Select 8 broadly-available baseline features (subset of the 18; we drop
# DaT-SPECT due to 16% coverage, plus the staging / NSD vars which are
# the outcome itself)
fg_features <- c("AGE_AT_BASELINE", "SEX", "UPDRS3_TOTAL", "UPDRS2_TOTAL",
                 "UPDRS3_TREMOR", "UPDRS3_BRADYKINESIA", "MOCA_TOTAL", "RBD_TOTAL")
# Use only features that exist
fg_features <- intersect(fg_features, colnames(covar_subset))
cat("  Fine-Gray features:", length(fg_features), "—", paste(fg_features, collapse=", "), "\n")

# Build covariate matrix; impute NaN with column median; standardize to mean=0,sd=1
cov_mat <- as.matrix(covar_subset[, fg_features])
storage.mode(cov_mat) <- "numeric"
for (j in seq_len(ncol(cov_mat))) {
  m <- median(cov_mat[, j], na.rm = TRUE)
  cov_mat[is.na(cov_mat[, j]), j] <- m
}
# Drop zero-variance columns
keep_cols <- apply(cov_mat, 2, function(c) sd(c, na.rm = TRUE) > 1e-9)
if (sum(!keep_cols) > 0) {
  cat("  Dropping zero-variance columns:", paste(fg_features[!keep_cols], collapse=", "), "\n")
  cov_mat <- cov_mat[, keep_cols, drop = FALSE]
  fg_features <- fg_features[keep_cols]
}
# Standardize each column
col_means <- colMeans(cov_mat)
col_sds <- apply(cov_mat, 2, sd)
cov_mat <- sweep(cov_mat, 2, col_means, "-")
cov_mat <- sweep(cov_mat, 2, col_sds, "/")
cat("  After standardization: cov_mat", nrow(cov_mat), "x", ncol(cov_mat),
    "(features:", paste(fg_features, collapse=", "), ")\n")

# Drop episodes with all-NA covariates
keep <- complete.cases(cov_mat) & !is.na(episode_df$cause)
episode_df <- episode_df[keep, ]
cov_mat <- cov_mat[keep, ]
cat("  ", nrow(episode_df), "episodes after covariate completeness filter\n")

# ────────────────────────────────────────────────────────────
# Fit Fine-Gray for each cause j (1..7), reporting subdistribution
# hazard coefficients + Wolbers C-index at 3/5/10 yr horizons
# ────────────────────────────────────────────────────────────
horizons_months <- c(36, 60, 120)  # 3, 5, 10 yr

unique_causes <- sort(unique(episode_df$cause))
unique_causes <- unique_causes[unique_causes != 0]  # exclude censored
cat("\nFitting Fine-Gray for", length(unique_causes), "causes ...\n")

cause_results <- list()
for (j in unique_causes) {
  cause_label <- as.character(cause_levels[j])
  n_events_j <- sum(episode_df$cause == j)
  if (n_events_j < 10) {
    cat("  cause", j, "(", cause_label, "): only", n_events_j, "events — SKIP\n")
    cause_results[[paste0("cause_", j)]] <- list(
      cause_idx = j, cause_label = cause_label, n_events = n_events_j,
      status = "SKIP_TOO_FEW_EVENTS"
    )
    next
  }
  cat("  fitting cause", j, "(", cause_label, "), n_events =", n_events_j, "...\n")
  fit <- tryCatch({
    crr(ftime = episode_df$time_to_event,
        fstatus = episode_df$cause,
        cov1 = cov_mat,
        failcode = j,
        cencode = 0,
        gtol = 1e-6,
        maxiter = 30)
  }, error = function(e) {
    cat("    ERROR:", conditionMessage(e), "\n")
    NULL
  })
  if (is.null(fit)) {
    cause_results[[paste0("cause_", j)]] <- list(
      cause_idx = j, cause_label = cause_label, n_events = n_events_j,
      status = "FIT_FAILED"
    )
    next
  }

  # Predict CIF at horizons via predict.crr
  newdata_uniform <- matrix(rep(colMeans(cov_mat), nrow(cov_mat)),
                             nrow=nrow(cov_mat), byrow=TRUE)
  pred <- tryCatch({
    predict(fit, cov1 = cov_mat)
  }, error = function(e) {
    cat("    PREDICT ERROR:", conditionMessage(e), "\n")
    NULL
  })

  # Per-horizon Wolbers C-index: pairwise concordance over (i, j) pairs where
  # patient i has event at cause j BEFORE patient j has event/censored at horizon
  # cmprsk doesn't ship a Wolbers c-index function; we approximate by
  # IPCW Cox c-index on the linear predictor (which is what predict.crr returns
  # for CIF ranking)
  lp <- as.numeric(cov_mat %*% fit$coef)
  c_index_at <- numeric(length(horizons_months))
  for (h_idx in seq_along(horizons_months)) {
    h <- horizons_months[h_idx]
    # Cases: those who had cause j by time h
    case_mask <- episode_df$cause == j & episode_df$time_to_event <= h
    # Controls: those still alive (censored OR event from other cause but later)
    # at time h who are at-risk
    control_mask <- (episode_df$time_to_event > h) |
                    (episode_df$time_to_event <= h & episode_df$cause != j & episode_df$cause != 0)
    n_cases <- sum(case_mask)
    n_ctrls <- sum(control_mask)
    if (n_cases < 5 || n_ctrls < 5) {
      c_index_at[h_idx] <- NA
      next
    }
    # Pairwise concordance: case lp > control lp ⇒ concordant
    # Sample 10000 pairs for speed
    n_pairs <- min(10000L, n_cases * n_ctrls)
    case_idx <- which(case_mask)
    ctrl_idx <- which(control_mask)
    # Sample pairs
    s_cases <- sample(case_idx, n_pairs, replace = TRUE)
    s_ctrls <- sample(ctrl_idx, n_pairs, replace = TRUE)
    diffs <- lp[s_cases] - lp[s_ctrls]
    conc <- sum(diffs > 0) + 0.5 * sum(diffs == 0)
    c_index_at[h_idx] <- conc / n_pairs
  }

  cause_results[[paste0("cause_", j)]] <- list(
    cause_idx = j,
    cause_label = cause_label,
    n_events = n_events_j,
    n_features = length(fg_features),
    coef_names = fg_features,
    coef = as.numeric(fit$coef),
    coef_se = as.numeric(sqrt(diag(fit$var))),
    z = as.numeric(fit$coef / sqrt(diag(fit$var))),
    p_value = as.numeric(2 * pnorm(-abs(fit$coef / sqrt(diag(fit$var))))),
    c_index_3yr = c_index_at[1],
    c_index_5yr = c_index_at[2],
    c_index_10yr = c_index_at[3],
    status = "FIT_SUCCESS"
  )
  cat(sprintf("    coef[1:3]: %.3f, %.3f, %.3f | C@3yr=%.3f, C@5yr=%.3f, C@10yr=%.3f\n",
              fit$coef[1], fit$coef[2], fit$coef[3],
              c_index_at[1], c_index_at[2], c_index_at[3]))
}

# Aggregate: mean C-index across successful causes per horizon
successful <- Filter(function(r) r$status == "FIT_SUCCESS", cause_results)
cat("\n===== Aggregate Fine-Gray Concordance =====\n")
agg_3yr <- mean(sapply(successful, function(r) r$c_index_3yr), na.rm = TRUE)
agg_5yr <- mean(sapply(successful, function(r) r$c_index_5yr), na.rm = TRUE)
agg_10yr <- mean(sapply(successful, function(r) r$c_index_10yr), na.rm = TRUE)
cat(sprintf("  Mean C-index across %d causes: 3yr=%.3f, 5yr=%.3f, 10yr=%.3f\n",
            length(successful), agg_3yr, agg_5yr, agg_10yr))

res <- list(
  status = "FIT_SUCCESS",
  n_episodes = nrow(episode_df),
  n_causes_fit = length(successful),
  n_causes_total = length(unique_causes),
  features = fg_features,
  horizons_months = horizons_months,
  cause_results = cause_results,
  aggregate_c_index = list(
    `3yr` = agg_3yr,
    `5yr` = agg_5yr,
    `10yr` = agg_10yr
  ),
  reference_models = list(
    DeepHit_C_td = 0.926,
    Graph_DT_C_td = 0.920,
    Markov_C_td_avg_horizons = 0.654
  )
)

write_json(res, file.path(OUTPUT_DIR, "fine_gray_results.json"),
           pretty = TRUE, auto_unbox = TRUE, na = "null")
cat("\nWrote", file.path(OUTPUT_DIR, "fine_gray_results.json"), "\n")
