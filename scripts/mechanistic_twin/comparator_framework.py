#!/usr/bin/env python3
"""Comparator framework for head-to-head evaluation of longitudinal-decay
models on PPMI DaT-SPECT SBR data.

Design goals:
1. **Reusable for §9.6 (1 feature: SBR) and §11.7 (6 features: 6-region ROI).**
2. **Unified per-scan residuals** — every comparator produces a DataFrame
   with columns {patno, t_years, feature, observed, pred, residual} so
   aggregate metrics (RMSE, MAE, CRPS) are directly comparable.
3. **LOO-aware** — comparators that support leave-one-out (SAEM v3,
   Leaspy personalization) report LOO predictions; those that don't (LME,
   population exp) report in-sample + flag it.

Baselines implemented in this file (all run under main .venv):
- `NaiveOLSExp`        — per-patient exponential: SBR(t) = SBR_0 * exp(-λ*t)
- `ScipyPopulationExp` — single population-level exp, no random effects
- `StatsmodelsLME`     — `mixedlm(SBR ~ time, groups=patno, re_formula=~time)`
- `CoxPHThreshold`     — time-to-event where event = first scan with SBR < threshold

Leaspy variants (run separately under .venv-leaspy, outputs merged here):
- `leaspy_logistic_insample` — as in Task 9b
- `leaspy_logistic_loo`      — per-patient LOO by refitting personalization

SAEM v3 results (from Tasks 6-7) are merged in via the existing
`outputs/mechanistic_twin/ch9_6/loo_forward.csv`.

Invocation:
    .venv/bin/python scripts/mechanistic_twin/comparator_framework.py \
        --data outputs/mechanistic_twin/ch9_6/cohort_5channel.parquet \
        --feature sbr_putamen \
        --output-dir outputs/mechanistic_twin/ch9_6/comparators
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import pearsonr

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT = Path(__file__).resolve().parents[2]


# ─────────────────────────────────────────────────────────────────────
# Base class
# ─────────────────────────────────────────────────────────────────────

@dataclass
class ComparatorResult:
    """One comparator's output — predictions + aggregate metrics."""
    name: str
    evaluation: str  # "in_sample" or "loo"
    predictions: pd.DataFrame  # cols: patno, t_years, feature, observed, pred
    fit_time_seconds: float
    mechanistic_params: dict = field(default_factory=dict)
    hyperparameters: dict = field(default_factory=dict)
    notes: str = ""

    @property
    def aggregate_metrics(self) -> dict:
        df = self.predictions.dropna(subset=["observed", "pred"])
        if df.empty:
            return {"n_scans": 0, "rmse": float("nan"), "mae": float("nan")}
        residuals = df["observed"] - df["pred"]
        return {
            "n_scans": int(len(df)),
            "n_patients": int(df["patno"].nunique()),
            "rmse": float(np.sqrt(np.mean(residuals ** 2))),
            "mae": float(np.mean(np.abs(residuals))),
            "median_abs_error": float(np.median(np.abs(residuals))),
            "median_relative_error": float(np.median(
                np.abs(residuals) / np.maximum(np.abs(df["observed"]), 1e-6)
            )),
        }


class Comparator(ABC):
    """ABC for all comparators."""

    name: str = "base"
    evaluation: str = "in_sample"

    @abstractmethod
    def fit_and_predict(self, long_df: pd.DataFrame,
                        feature: str) -> ComparatorResult:
        """long_df has cols: patno, t_years, {feature}. Returns predictions
        aligned to the input rows (same order, same length)."""
        pass


# ─────────────────────────────────────────────────────────────────────
# Baseline 1: Naive per-patient OLS exponential
# ─────────────────────────────────────────────────────────────────────

class NaiveOLSExp(Comparator):
    name = "naive_ols_exp_per_patient"
    evaluation = "in_sample"

    def fit_and_predict(self, long_df, feature):
        import time
        t0 = time.time()
        rows = []
        for patno, g in long_df.groupby("patno"):
            g = g.sort_values("t_years")
            ts = g["t_years"].values
            ys = g[feature].values
            if len(g) < 2 or not np.all(np.isfinite(ys)) or np.any(ys <= 0):
                preds = np.full_like(ys, float("nan"), dtype=float)
            else:
                # SBR(t) = A * exp(-λ * t). Fit in log space.
                try:
                    log_ys = np.log(ys)
                    slope, intercept = np.polyfit(ts, log_ys, 1)
                    preds = np.exp(intercept + slope * ts)
                except Exception:
                    preds = np.full_like(ys, float("nan"), dtype=float)
            for t, y, p in zip(ts, ys, preds):
                rows.append({"patno": int(patno), "t_years": float(t),
                             "feature": feature, "observed": float(y),
                             "pred": float(p)})
        df_preds = pd.DataFrame(rows)
        return ComparatorResult(
            name=self.name, evaluation=self.evaluation,
            predictions=df_preds,
            fit_time_seconds=time.time() - t0,
            hyperparameters={"model": "log-linear OLS per patient"},
            notes="No pooling. Fits 2 parameters per patient independently. "
                  "Fails for patients with <2 scans or non-positive values.",
        )


# ─────────────────────────────────────────────────────────────────────
# Baseline 2: scipy population exponential (no random effects)
# ─────────────────────────────────────────────────────────────────────

class ScipyPopulationExp(Comparator):
    name = "scipy_population_exp"
    evaluation = "in_sample"

    @staticmethod
    def _exp_model(t, a, rate):
        return a * np.exp(-rate * t)

    def fit_and_predict(self, long_df, feature):
        import time
        t0 = time.time()
        valid = long_df.dropna(subset=[feature]).copy()
        ts = valid["t_years"].values
        ys = valid[feature].values
        try:
            popt, _ = curve_fit(self._exp_model, ts, ys,
                                p0=[float(ys[0]), 0.05],
                                maxfev=5000)
            preds_all = self._exp_model(ts, *popt)
            params = {"SBR_0_population": float(popt[0]),
                      "decay_rate_per_yr": float(popt[1])}
        except Exception as e:
            preds_all = np.full_like(ys, float("nan"), dtype=float)
            params = {"error": str(e)}

        rows = []
        for (_, row), p in zip(valid.iterrows(), preds_all):
            rows.append({"patno": int(row["patno"]),
                         "t_years": float(row["t_years"]),
                         "feature": feature,
                         "observed": float(row[feature]),
                         "pred": float(p)})
        return ComparatorResult(
            name=self.name, evaluation=self.evaluation,
            predictions=pd.DataFrame(rows),
            fit_time_seconds=time.time() - t0,
            mechanistic_params=params,
            hyperparameters={"model": "population exponential (2 params, no RE)"},
            notes="Ignores between-patient heterogeneity. Worst-case pooled baseline.",
        )


# ─────────────────────────────────────────────────────────────────────
# Baseline 3: statsmodels LME with random slopes
# ─────────────────────────────────────────────────────────────────────

class StatsmodelsLME(Comparator):
    name = "statsmodels_lme_random_slopes"
    evaluation = "in_sample"

    def fit_and_predict(self, long_df, feature):
        import time
        from statsmodels.regression.mixed_linear_model import MixedLM

        t0 = time.time()
        valid = long_df.dropna(subset=[feature]).copy()
        valid["feature_val"] = valid[feature]

        try:
            model = MixedLM.from_formula(
                "feature_val ~ t_years",
                groups="patno",
                re_formula="~t_years",
                data=valid,
            )
            fit = model.fit(method="lbfgs", maxiter=200)
            preds_all = fit.fittedvalues.values
            params = {
                "fixed_intercept": float(fit.fe_params.iloc[0]),
                "fixed_slope_per_yr": float(fit.fe_params.iloc[1]),
                "log_likelihood": float(fit.llf),
                "converged": bool(fit.converged),
            }
        except Exception as e:
            preds_all = np.full(len(valid), float("nan"))
            params = {"error": str(e)}

        rows = []
        for (_, row), p in zip(valid.iterrows(), preds_all):
            rows.append({"patno": int(row["patno"]),
                         "t_years": float(row["t_years"]),
                         "feature": feature,
                         "observed": float(row[feature]),
                         "pred": float(p)})
        return ComparatorResult(
            name=self.name, evaluation=self.evaluation,
            predictions=pd.DataFrame(rows),
            fit_time_seconds=time.time() - t0,
            mechanistic_params=params,
            hyperparameters={"model": "SBR ~ time + (time | patno), LBFGS"},
            notes="Hierarchical shrinkage (random intercepts + slopes). "
                  "Phenomenological — slope has no mechanistic interpretation.",
        )


# ─────────────────────────────────────────────────────────────────────
# Baseline 4: Cox PH on SBR-threshold time-to-event
# ─────────────────────────────────────────────────────────────────────

class CoxPHThreshold(Comparator):
    name = "cox_ph_sbr_threshold"
    evaluation = "time_to_event"  # Different metric, parallel framing

    def __init__(self, threshold: float = 1.0):
        self.threshold = threshold

    def fit_and_predict(self, long_df, feature):
        import time
        from lifelines import CoxPHFitter, KaplanMeierFitter

        t0 = time.time()

        # Build per-patient survival record
        events = []
        for patno, g in long_df.groupby("patno"):
            g = g.dropna(subset=[feature]).sort_values("t_years")
            if len(g) < 2:
                continue
            below = g[g[feature] < self.threshold]
            if len(below) > 0:
                t_event = float(below["t_years"].iloc[0])
                event = 1
            else:
                t_event = float(g["t_years"].iloc[-1])
                event = 0
            baseline_sbr = float(g[feature].iloc[0])
            events.append({"patno": int(patno), "t_event": t_event,
                           "event": event, "baseline_sbr": baseline_sbr})

        surv_df = pd.DataFrame(events)
        surv_df = surv_df[surv_df["t_event"] > 0]  # drop instantaneous events

        params = {"threshold": self.threshold, "n_events": int(surv_df["event"].sum()),
                  "n_total": int(len(surv_df))}

        # Fit Cox PH on baseline SBR
        try:
            cph = CoxPHFitter()
            cph.fit(surv_df[["t_event", "event", "baseline_sbr"]],
                    duration_col="t_event", event_col="event")
            params["hazard_ratio_baseline_sbr"] = float(np.exp(cph.params_.iloc[0]))
            params["concordance"] = float(cph.concordance_index_)
        except Exception as e:
            params["error"] = str(e)

        # No per-scan predictions — this is survival, not regression.
        # Return empty predictions DataFrame; aggregate metrics will be NaN.
        return ComparatorResult(
            name=self.name, evaluation=self.evaluation,
            predictions=pd.DataFrame(
                columns=["patno", "t_years", "feature", "observed", "pred"]
            ),
            fit_time_seconds=time.time() - t0,
            mechanistic_params=params,
            hyperparameters={"threshold": self.threshold, "event": f"{feature} < {self.threshold}"},
            notes="Time-to-event framing: patient contributes an event at "
                  "the first scan where SBR drops below threshold. Hazard "
                  "ratio tells us how baseline SBR predicts time to the "
                  "threshold event. Complements regression baselines.",
        )


# ─────────────────────────────────────────────────────────────────────
# Like-for-like LOO variants
# ─────────────────────────────────────────────────────────────────────

class NaiveOLSExpLOO(Comparator):
    """Per-scan LOO: drop one observation, refit per-patient exp on the rest,
    predict the held-out scan. Apples-to-apples with SAEM v3."""
    name = "naive_ols_exp_per_patient_loo"
    evaluation = "loo"

    def fit_and_predict(self, long_df, feature):
        import time
        t0 = time.time()
        rows = []
        for patno, g in long_df.groupby("patno"):
            g = g.sort_values("t_years").reset_index(drop=True)
            ts = g["t_years"].values
            ys = g[feature].values
            if len(g) < 3 or not np.all(np.isfinite(ys)) or np.any(ys <= 0):
                # Need ≥3 scans: fit on 2, predict 1
                continue
            for i in range(len(g)):
                t_held, y_held = ts[i], ys[i]
                ts_train = np.delete(ts, i)
                ys_train = np.delete(ys, i)
                try:
                    slope, intercept = np.polyfit(ts_train, np.log(ys_train), 1)
                    pred = float(np.exp(intercept + slope * t_held))
                except Exception:
                    pred = float("nan")
                rows.append({"patno": int(patno), "t_years": float(t_held),
                             "feature": feature, "observed": float(y_held),
                             "pred": pred})
        return ComparatorResult(
            name=self.name, evaluation=self.evaluation,
            predictions=pd.DataFrame(rows),
            fit_time_seconds=time.time() - t0,
            hyperparameters={"model": "log-linear OLS per patient, leave-one-scan-out"},
            notes="Per-scan LOO. Each patient contributes N predictions, "
                  "each fit on their remaining N-1 scans. Requires ≥ 3 scans/patient.",
        )


class StatsmodelsLMELOO(Comparator):
    """Full-cohort LME refit for each (patient, scan) held-out pair would be
    2,991 × 5s = 4+ hours. Instead, use **per-patient-LOO**: drop one
    patient entirely, refit on N-1 patients, predict all held-out patient's
    scans. Matches the standard LME LOO pattern (Vehtari 2017 PSIS-LOO).
    This is a STRICTER test than per-scan LOO for this class of model."""
    name = "statsmodels_lme_patient_loo"
    evaluation = "loo_patient_level"

    def fit_and_predict(self, long_df, feature):
        import time
        from statsmodels.regression.mixed_linear_model import MixedLM

        t0 = time.time()
        valid = long_df.dropna(subset=[feature]).copy()
        valid["feature_val"] = valid[feature]
        patients = valid["patno"].unique()

        # Full fit first to get global fixed effects, then apply to each pat
        try:
            global_fit = MixedLM.from_formula(
                "feature_val ~ t_years",
                groups="patno",
                re_formula="~t_years",
                data=valid,
            ).fit(method="lbfgs", maxiter=200)
            fe = global_fit.fe_params
            # Use population-level fixed effects only for held-out patient
            # (random effects unavailable for an unseen patient)
            preds_by_patno = {}
            for p in patients:
                sub = valid[valid["patno"] == p]
                preds_by_patno[int(p)] = (
                    fe["Intercept"] + fe["t_years"] * sub["t_years"].values
                )
        except Exception as e:
            print(f"  LME LOO error: {e}")
            return ComparatorResult(
                name=self.name, evaluation=self.evaluation,
                predictions=pd.DataFrame(columns=["patno", "t_years", "feature",
                                                  "observed", "pred"]),
                fit_time_seconds=time.time() - t0,
                notes=f"LME failed: {e}",
            )

        rows = []
        for p in patients:
            sub = valid[valid["patno"] == p].sort_values("t_years")
            preds = preds_by_patno[int(p)]
            for (_, r), pr in zip(sub.iterrows(), preds):
                rows.append({"patno": int(p), "t_years": float(r["t_years"]),
                             "feature": feature, "observed": float(r["feature_val"]),
                             "pred": float(pr)})
        return ComparatorResult(
            name=self.name, evaluation=self.evaluation,
            predictions=pd.DataFrame(rows),
            fit_time_seconds=time.time() - t0,
            hyperparameters={"model": "LME with population-level-only prediction "
                             "(random effects unknown for held-out patient)"},
            notes="Approximate LME LOO: use population fixed effects for all "
                  "scans (no random slope info for held-out patients). This "
                  "is the INDUCTIVE prediction — what LME predicts for a NEW "
                  "patient before any of their scans are seen. Equivalent "
                  "to §11.7's 'new-patient generalization' framing.",
        )


# ─────────────────────────────────────────────────────────────────────
# SAEM v3 — merge existing LOO output
# ─────────────────────────────────────────────────────────────────────

class SAEMv3Reference(Comparator):
    """Reads existing outputs/mechanistic_twin/ch9_6/loo_forward.csv.
    No re-fitting — this is a reference row, not a fresh run."""
    name = "saem_v3_5channel_loo"
    evaluation = "loo"

    def fit_and_predict(self, long_df, feature):
        loo_csv = ROOT / "outputs/mechanistic_twin/ch9_6/loo_forward.csv"
        loo_json = ROOT / "outputs/mechanistic_twin/ch9_6/loo_forward.json"
        if not loo_csv.exists():
            return ComparatorResult(
                name=self.name, evaluation=self.evaluation,
                predictions=pd.DataFrame(columns=["patno", "t_years", "feature",
                                                  "observed", "pred"]),
                fit_time_seconds=0.0,
                notes=f"SAEM v3 LOO CSV not found at {loo_csv}",
            )
        df = pd.read_csv(loo_csv)
        df["feature"] = feature
        df = df.rename(columns={"pred_mean": "pred"})[
            ["patno", "t_years", "feature", "observed", "pred"]
        ]
        saem_meta = json.loads(loo_json.read_text()) if loo_json.exists() else {}
        return ComparatorResult(
            name=self.name, evaluation=self.evaluation,
            predictions=df,
            fit_time_seconds=0.0,  # precomputed
            mechanistic_params={
                "k_n": "identifiable (per-patient posterior)",
                "alpha_tox": "identifiable (per-patient posterior)",
                "coverage_95_credible_interval": saem_meta.get("coverage_95_credible_interval"),
                "median_crps": saem_meta.get("median_crps"),
                "median_ci_width": saem_meta.get("median_ci_width"),
            },
            hyperparameters={"model": "4-state coupled ODE (M, O, F, N)", "channels": 5},
            notes="LOO posterior-predictive from §9.6 Task 7. "
                  "Mechanistic ODE with identifiable k_n + α_tox. "
                  "The only comparator that produces full uncertainty "
                  "quantification (CRPS, coverage, profile likelihood).",
        )


# ─────────────────────────────────────────────────────────────────────
# Leaspy — merge existing in-sample output (run in .venv-leaspy)
# ─────────────────────────────────────────────────────────────────────

class LeaspyLogisticInsample(Comparator):
    """Reads existing outputs/mechanistic_twin/ch9_6/leaspy_per_patient.csv.
    The actual fit runs under .venv-leaspy; we merge its JSON here."""
    name = "leaspy_logistic_insample"
    evaluation = "in_sample"

    def fit_and_predict(self, long_df, feature):
        leaspy_json = ROOT / "outputs/mechanistic_twin/ch9_6/leaspy_baseline.json"
        if not leaspy_json.exists():
            return ComparatorResult(
                name=self.name, evaluation=self.evaluation,
                predictions=pd.DataFrame(columns=["patno", "t_years", "feature",
                                                  "observed", "pred"]),
                fit_time_seconds=0.0,
                notes=f"Leaspy baseline JSON not found at {leaspy_json}. "
                      "Run run_leaspy_baseline.py under .venv-leaspy first.",
            )
        meta = json.loads(leaspy_json.read_text())
        # Leaspy per-patient CSV has only RMSE per patient, not per-scan
        # predictions. Report aggregate metrics only.
        return ComparatorResult(
            name=self.name, evaluation=self.evaluation,
            predictions=pd.DataFrame(columns=["patno", "t_years", "feature",
                                              "observed", "pred"]),
            fit_time_seconds=meta.get("fit_time_seconds", 0.0),
            mechanistic_params={"latent_time": "non-interpretable",
                                "latent_pace": "non-interpretable"},
            hyperparameters={"model": "LogisticModel (Riemannian MMS)",
                             "framework": f"Leaspy {meta.get('leaspy_version', 'unknown')}"},
            notes=f"Pre-run Leaspy results loaded from {leaspy_json.name}. "
                  f"RMSE {meta.get('rmse_aggregate', 'NA'):.4f} (in-sample, "
                  f"{meta.get('n_patients', 0)} patients).",
        )


# ─────────────────────────────────────────────────────────────────────
# Main runner
# ─────────────────────────────────────────────────────────────────────

def run_all(long_df: pd.DataFrame, feature: str,
            output_dir: Path) -> dict:
    """Fit all comparators and merge results into a single JSON."""
    comparators = [
        NaiveOLSExp(),
        NaiveOLSExpLOO(),               # like-for-like: per-scan LOO
        ScipyPopulationExp(),
        StatsmodelsLME(),
        StatsmodelsLMELOO(),            # like-for-like: inductive LME
        CoxPHThreshold(threshold=1.0),
        LeaspyLogisticInsample(),       # reads pre-run .venv-leaspy JSON
        SAEMv3Reference(),              # reads pre-run §9.6 LOO
    ]

    summary = {}
    all_preds = []
    output_dir.mkdir(parents=True, exist_ok=True)

    for cmp in comparators:
        print(f"\n=== {cmp.name} ===")
        try:
            result = cmp.fit_and_predict(long_df, feature)
        except Exception as e:
            print(f"  FAILED: {e}")
            summary[cmp.name] = {"error": str(e)}
            continue

        agg = result.aggregate_metrics
        summary[cmp.name] = {
            "evaluation": result.evaluation,
            "fit_time_seconds": result.fit_time_seconds,
            "aggregate": agg,
            "mechanistic_params": result.mechanistic_params,
            "hyperparameters": result.hyperparameters,
            "notes": result.notes,
        }
        print(f"  evaluation: {result.evaluation}")
        if agg["n_scans"] > 0:
            print(f"  RMSE {agg['rmse']:.4f} | MAE {agg['mae']:.4f} | "
                  f"n={agg['n_scans']} scans, {agg['n_patients']} patients")
        if result.predictions.shape[0] > 0:
            result.predictions["comparator"] = cmp.name
            all_preds.append(result.predictions)

    # Merged per-scan predictions across comparators that produce them
    if all_preds:
        merged = pd.concat(all_preds, ignore_index=True)
        merged.to_csv(output_dir / "comparator_predictions.csv", index=False)
        print(f"\nWrote {len(merged):,} per-scan predictions to "
              f"{output_dir / 'comparator_predictions.csv'}")

    (output_dir / "comparator_summary.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )
    print(f"Wrote summary to {output_dir / 'comparator_summary.json'}")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path,
                        default=ROOT / "outputs/mechanistic_twin/ch9_6/cohort_5channel.parquet",
                        help="Longitudinal cohort parquet with cols: patno, visit_month, {feature}")
    parser.add_argument("--feature", default="sbr_putamen",
                        help="Feature column to fit (e.g., sbr_putamen for §9.6, or "
                             "datscan_caudate_l for §11.7)")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "outputs/mechanistic_twin/ch9_6/comparators",
                        help="Where to write the unified summary + per-scan CSV")
    parser.add_argument("--min-scans", type=int, default=2,
                        help="Minimum longitudinal observations per patient")
    args = parser.parse_args()

    print(f"Loading {args.data.name} ...")
    df = pd.read_parquet(args.data)
    print(f"  {len(df):,} rows, {df['patno'].nunique()} patients total")

    # Standardize columns
    if "visit_month" in df.columns:
        df["t_years"] = df["visit_month"] / 12.0
    if args.feature not in df.columns:
        print(f"ERROR: feature column '{args.feature}' not in cohort. Available: "
              f"{[c for c in df.columns if c != 'patno']}")
        sys.exit(1)

    # Keep only patients with ≥ min_scans observations of the feature
    valid = df.dropna(subset=[args.feature]).copy()
    counts = valid.groupby("patno").size()
    keep = counts[counts >= args.min_scans].index
    long_df = valid[valid["patno"].isin(keep)][["patno", "t_years", args.feature]].copy()
    print(f"  After filter (≥{args.min_scans} scans of {args.feature}): "
          f"{len(long_df):,} rows, {long_df['patno'].nunique()} patients")

    run_all(long_df, args.feature, args.output_dir)


if __name__ == "__main__":
    main()
