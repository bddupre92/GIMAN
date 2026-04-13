"""Configuration: paths to data, checkpoints, and constants."""

from __future__ import annotations

from pathlib import Path

# ── Project Root ─────────────────────────────────────────────
# clinical_dashboard/ sits inside the project root
DASHBOARD_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = DASHBOARD_DIR.parent

# ── Data Paths ───────────────────────────────────────────────
PAPER1_FEATURES_PATH = PROJECT_ROOT / "data" / "05_features" / "paper1_features_with_targets.csv"
LONGITUDINAL_FEATURES_PATH = PROJECT_ROOT / "data" / "07_paper3_features" / "longitudinal_features.csv"
TRANSITION_EVENTS_PATH = PROJECT_ROOT / "data" / "06_longitudinal_staging" / "transition_events.csv"

# ── Model Checkpoints ────────────────────────────────────────
CATBOOST_CHECKPOINT = PROJECT_ROOT / "outputs" / "paper6" / "pipeline_results" / "catboost_nsd_positive.cbm"
DEEPHIT_CHECKPOINT = PROJECT_ROOT / "outputs" / "paper3_checkpoints" / "deephit" / "fold0_deephit.pt"
GRAPHDT_CHECKPOINT = PROJECT_ROOT / "outputs" / "paper3_checkpoints" / "graph_dt" / "fold0_graph_dt.pt"

# ── Results / Metadata ───────────────────────────────────────
CONFORMAL_AGGREGATE_PATH = PROJECT_ROOT / "outputs" / "paper4" / "conformal" / "aggregate_summary.json"
MARKOV_RESULTS_PATH = PROJECT_ROOT / "outputs" / "paper3_markov" / "markov_results.json"
PRECOMPUTED_PIPELINE_DIR = PROJECT_ROOT / "outputs" / "paper6" / "pipeline_results"

# ── Synthetic Data ───────────────────────────────────────────
SYNTHETIC_WEARABLE_DIR = DASHBOARD_DIR / "synthetic_data" / "wearable" / "samples"

# ── Annotations ──────────────────────────────────────────────
ANNOTATIONS_DIR = DASHBOARD_DIR / "annotations"

# ── CatBoost 12-Feature Clinical Model ───────────────────────
CATBOOST_12_FEATURES = [
    "AGE_AT_BASELINE",
    "SEX",
    "UPDRS1_TOTAL",
    "UPDRS2_TOTAL",
    "UPDRS3_TREMOR",
    "UPDRS3_RIGIDITY",
    "UPDRS3_BRADYKINESIA",
    "UPDRS3_AXIAL",
    "UPDRS4_TOTAL",
    "MOCA_TOTAL",
    "ESS_TOTAL",
    "RBD_TOTAL",
]

# Mapping from Paper 3 longitudinal feature names to Paper 1 names
LONGITUDINAL_TO_PAPER1 = {
    "age_at_baseline": "AGE_AT_BASELINE",
    "sex": "SEX",
    "updrs1_total": "UPDRS1_TOTAL",
    "updrs2_total": "UPDRS2_TOTAL",
    "moca_total": "MOCA_TOTAL",
    "ess_total": "ESS_TOTAL",
    "rbd_total": "RBD_TOTAL",
}

# ── NSD-ISS Stage Constants ──────────────────────────────────
STAGE_LABELS = {0: "0", 1: "1", 2: "2B", 3: "3", 4: "4", 5: "5", 6: "6"}
NSD_POSITIVE_LABELS = {0: "1", 1: "2B", 2: "3", 3: "4"}
STAGE_TO_IDX = {"0": 0, "1": 1, "2B": 2, "3": 3, "4": 4, "5": 5, "6": 6}

# Time bins for survival models (months)
TIME_BIN_ENDS = [3, 6, 12, 18, 24, 36, 48, 60, 84, 120, 180]
N_CAUSES = 7
N_TIME_BINS = 11

# Stage colors for consistent UI rendering
STAGE_COLORS = {
    "0": "#2ecc71",   # Green
    "1": "#3498db",   # Blue
    "2B": "#f1c40f",  # Yellow
    "3": "#e67e22",   # Orange
    "4": "#e74c3c",   # Red
    "5": "#9b59b6",   # Purple
    "6": "#7f8c8d",   # Gray
}

# ── Feature Ranges for What-If Sliders ───────────────────────
FEATURE_RANGES = {
    "AGE_AT_BASELINE": {"min": 30, "max": 90, "step": 1, "label": "Age at Baseline"},
    "SEX": {"min": 0, "max": 1, "step": 1, "label": "Sex (0=Male, 1=Female)"},
    "UPDRS1_TOTAL": {"min": 0, "max": 52, "step": 1, "label": "UPDRS-I (Non-Motor)"},
    "UPDRS2_TOTAL": {"min": 0, "max": 52, "step": 1, "label": "UPDRS-II (Motor ADL)"},
    "UPDRS3_TREMOR": {"min": 0, "max": 40, "step": 1, "label": "UPDRS-III Tremor"},
    "UPDRS3_RIGIDITY": {"min": 0, "max": 20, "step": 1, "label": "UPDRS-III Rigidity"},
    "UPDRS3_BRADYKINESIA": {"min": 0, "max": 36, "step": 1, "label": "UPDRS-III Bradykinesia"},
    "UPDRS3_AXIAL": {"min": 0, "max": 20, "step": 1, "label": "UPDRS-III Axial"},
    "UPDRS4_TOTAL": {"min": 0, "max": 24, "step": 1, "label": "UPDRS-IV (Complications)"},
    "MOCA_TOTAL": {"min": 0, "max": 30, "step": 1, "label": "MoCA (Cognitive)"},
    "ESS_TOTAL": {"min": 0, "max": 24, "step": 1, "label": "Epworth Sleepiness"},
    "RBD_TOTAL": {"min": 0, "max": 13, "step": 1, "label": "RBD Screening"},
}

# Default conformal band width (from Paper 4, 95% CL)
DEFAULT_CONFORMAL_BAND_WIDTH = 0.037
