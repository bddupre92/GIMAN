"""Flexible NSD-ISS cohort/training experiment runner for Paper 1.

Generalises the PD-only retraining script into a single CLI that can express:
  (1) full-PPMI baseline training (current Paper 1)
  (2) PD+Prodromal-only training (our current mitigation)
  (3) sample-weighted training (e.g. HC weighted 0.1)
  (4) Stage-A of hierarchical (HC vs PD+Prodromal as its own binary task)
  (5) Stage-B of hierarchical (S+ vs S- conditional on PD+Prodromal)

Literature motivation:
  - Espay 2025 DOI:10.1002/mds.30269 (refutation of SAA-only NSD-ISS staging)
  - Simuni 2025 DOI:10.1002/mds.30272 (reply)
  - Bentivoglio 2026 DOI:10.1002/mds.70197 (clinical/imaging characterisation of
    SAA-negative PD --- motivates our "SAA- PD" training subclass)
  - Russo 2025 / 2025 BioFIND DOI:10.1038/s41531-025-00992-3 (BioFIND NSD-ISS)
  - "Reconsidering NSD-ISS clinical foundations" 2025 DOI:10.1002/mds.70061

Usage examples
--------------
# (1) Paper 1 original baseline
python scripts/paper1/run_paper1_cohort_experiment.py \\
    --name baseline --cohorts all --target binary --model catboost

# (2) PD+Prodromal only (current mitigation)
python scripts/paper1/run_paper1_cohort_experiment.py \\
    --name pd_only --cohorts pd,prodromal --target binary --model catboost

# (3) Sample-weighted (HC and SWEDD downweighted to 0.1)
python scripts/paper1/run_paper1_cohort_experiment.py \\
    --name weighted --cohorts all --target binary --model catboost \\
    --cohort-weights 'Healthy Control:0.1,SWEDD:0.1'

# (4) Stage-A hierarchical: HC vs PD+Prodromal (PD-detection head)
python scripts/paper1/run_paper1_cohort_experiment.py \\
    --name stage_a --cohorts all --target hc_vs_pd --model catboost

# (5) Stage-B hierarchical: S+ vs S- conditional on PD+Prodromal
python scripts/paper1/run_paper1_cohort_experiment.py \\
    --name stage_b --cohorts pd,prodromal --target binary --model catboost

Outputs per run land in outputs/paper1_cohort/<name>/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data"
OUT_ROOT = REPO / "outputs" / "paper1_cohort"

SEED = 42

COMMON_FEATURES = [
    "AGE_AT_BASELINE", "SEX",
    "UPDRS1_TOTAL", "UPDRS2_TOTAL",
    "UPDRS3_TREMOR", "UPDRS3_RIGIDITY", "UPDRS3_BRADYKINESIA", "UPDRS3_AXIAL",
    "UPDRS4_TOTAL",
    "MOCA_TOTAL", "ESS_TOTAL", "RBD_TOTAL",
]

COHORT_ALIASES = {
    "pd": "Parkinson's Disease",
    "hc": "Healthy Control",
    "prodromal": "Prodromal",
    "swedd": "SWEDD",
}


def parse_cohorts(spec: str) -> list[str] | None:
    if spec == "all":
        return None
    wanted = [c.strip().lower() for c in spec.split(",") if c.strip()]
    mapped = []
    for w in wanted:
        if w in COHORT_ALIASES:
            mapped.append(COHORT_ALIASES[w])
        else:
            mapped.append(w)  # already a full cohort label
    return mapped


def parse_weights(spec: str | None) -> dict[str, float]:
    if not spec:
        return {}
    out = {}
    for pair in spec.split(","):
        if not pair.strip():
            continue
        cohort, w = pair.rsplit(":", 1)
        out[cohort.strip()] = float(w)
    return out


def load_ppmi() -> pd.DataFrame:
    feat = pd.read_csv(DATA / "05_features" / "paper1_features_with_targets.csv")
    ps = pd.read_csv(DATA / "00_raw/GIMAN/ppmi_data_csv/Participant_Status_30Sep2025.csv")
    ps_min = ps[["PATNO", "COHORT_DEFINITION"]].drop_duplicates("PATNO")
    return feat.merge(ps_min, on="PATNO", how="left")


def load_biofind_balanced() -> pd.DataFrame:
    """Load BioFIND features + balanced NSD-ISS staging.

    Russo 2025 staging captures n=103 NSD+ patients; BioFIND features file
    contains n=118 PD patients. The 15 extra patients are SAA-negative PD,
    which per Bentivoglio 2026 (DOI:10.1002/mds.70197) are a well-defined
    real clinical subgroup and the correct NSD-negative reference class for
    a within-PD binary task. We therefore build a balanced BioFIND test set
    as: target_binary = 1 for the 103 Russo-staged NSD+ PD and 0 for the
    15 unstaged SAA-negative PD.
    """
    feat = pd.read_csv(DATA / "05_features" / "biofind_features.csv")
    stg = pd.read_csv(DATA / "04_staging" / "biofind_nsd_iss_staging.csv")
    staged_ids = set(stg["participant_id"].tolist())
    staged = feat.merge(
        stg[["participant_id", "target_binary"]], on="participant_id", how="inner"
    )
    unstaged = feat[~feat["participant_id"].isin(staged_ids)].copy()
    unstaged["target_binary"] = 0
    balanced = pd.concat([staged, unstaged], ignore_index=True)
    balanced["target_binary"] = balanced["target_binary"].astype(int)
    balanced["PATNO"] = (
        balanced["participant_id"].astype(str).str.replace("BF-", "", regex=False).astype(int)
    )
    return balanced


def prepare_xy(
    df: pd.DataFrame,
    target: str,
    cohorts: list[str] | None,
    features: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """Return (X, y, sample_weight_placeholder, filtered_df)."""
    if cohorts is not None and "COHORT_DEFINITION" in df.columns:
        df = df[df["COHORT_DEFINITION"].isin(cohorts)].copy()

    if target == "binary":
        if "target_binary" not in df.columns:
            raise ValueError("target=binary requires 'target_binary' column")
        df = df.dropna(subset=["target_binary"]).copy()
        y = df["target_binary"].astype(int).values
    elif target == "hc_vs_pd":
        if "COHORT_DEFINITION" not in df.columns:
            raise ValueError("target=hc_vs_pd requires 'COHORT_DEFINITION' column")
        df["_y_hc_vs_pd"] = df["COHORT_DEFINITION"].map({
            "Parkinson's Disease": 1, "Prodromal": 1,
            "Healthy Control": 0, "SWEDD": 0,
        })
        df = df.dropna(subset=["_y_hc_vs_pd"]).copy()
        y = df["_y_hc_vs_pd"].astype(int).values
    else:
        raise ValueError(f"Unknown target: {target}")

    present_features = [f for f in features if f in df.columns]
    for c in present_features:
        df[c] = df[c].fillna(df[c].median())
    X = df[present_features].values
    sw = np.ones(len(df))
    return X, y, sw, df


def apply_cohort_weights(
    sw: np.ndarray, df: pd.DataFrame, weights: dict[str, float]
) -> np.ndarray:
    if not weights:
        return sw
    sw_out = sw.copy()
    cohorts = df["COHORT_DEFINITION"].values if "COHORT_DEFINITION" in df.columns else None
    if cohorts is None:
        return sw_out
    for cohort, w in weights.items():
        mask = cohorts == cohort
        sw_out[mask] = w
    return sw_out


def make_model(name: str):
    if name == "catboost":
        return CatBoostClassifier(
            iterations=1000, depth=6,
            auto_class_weights="Balanced",
            random_seed=SEED, verbose=False,
        )
    if name == "logreg":
        return LogisticRegression(max_iter=5000, class_weight="balanced", random_state=SEED)
    raise ValueError(f"Unknown model: {name}")


def cv_evaluate(X, y, sw, model_name: str, scale: bool = False) -> dict:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    per_fold = []
    for fi, (tr, te) in enumerate(skf.split(X, y)):
        clf = make_model(model_name)
        if scale:
            scaler = StandardScaler()
            Xtr = scaler.fit_transform(X[tr])
            Xte = scaler.transform(X[te])
        else:
            Xtr, Xte = X[tr], X[te]
        if model_name == "catboost":
            clf.fit(Xtr, y[tr], sample_weight=sw[tr])
        else:
            clf.fit(Xtr, y[tr], sample_weight=sw[tr])
        y_pred = (
            clf.predict(Xte).ravel().astype(int) if model_name == "catboost"
            else clf.predict(Xte).astype(int)
        )
        y_prob = clf.predict_proba(Xte)[:, 1]
        per_fold.append({
            "fold": fi,
            "n_test": int(len(te)),
            "bal_acc": float(balanced_accuracy_score(y[te], y_pred)),
            "auc": float(roc_auc_score(y[te], y_prob))
            if len(np.unique(y[te])) > 1 else float("nan"),
        })
    ba = np.array([f["bal_acc"] for f in per_fold])
    auc = np.array([f["auc"] for f in per_fold])
    return {
        "per_fold": per_fold,
        "bal_acc_mean": float(np.nanmean(ba)),
        "bal_acc_sd": float(np.nanstd(ba, ddof=1)),
        "auc_mean": float(np.nanmean(auc)),
        "auc_sd": float(np.nanstd(auc, ddof=1)),
    }


def train_full_and_apply(
    X_tr, y_tr, sw_tr, X_ext, y_ext, model_name: str, scale: bool = False
) -> dict:
    clf = make_model(model_name)
    if scale:
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_ext = scaler.transform(X_ext)
    clf.fit(X_tr, y_tr, sample_weight=sw_tr)
    y_pred = (
        clf.predict(X_ext).ravel().astype(int) if model_name == "catboost"
        else clf.predict(X_ext).astype(int)
    )
    y_prob = clf.predict_proba(X_ext)[:, 1]
    both_classes = len(np.unique(y_ext)) > 1
    return {
        "n_test": int(len(y_ext)),
        "n_pos": int((y_ext == 1).sum()),
        "n_neg": int((y_ext == 0).sum()),
        "bal_acc": float(balanced_accuracy_score(y_ext, y_pred)),
        "auc": float(roc_auc_score(y_ext, y_prob)) if both_classes else float("nan"),
        "y_pred": y_pred.tolist(),
        "y_prob": y_prob.tolist(),
        "y_true": y_ext.tolist(),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True, help="run name (output subdir)")
    ap.add_argument("--cohorts", default="all",
                    help="comma-separated: pd,prodromal,hc,swedd or 'all'")
    ap.add_argument("--target", default="binary",
                    choices=["binary", "hc_vs_pd"],
                    help="binary=target_binary NSD+/NSD-; hc_vs_pd=Stage-A hierarchical head")
    ap.add_argument("--model", default="catboost",
                    choices=["catboost", "logreg"])
    ap.add_argument("--cohort-weights", default=None,
                    help="comma-separated 'Cohort Label:weight' pairs, e.g. 'Healthy Control:0.1,SWEDD:0.1'")
    ap.add_argument("--external", default="biofind_balanced",
                    choices=["none", "biofind_russo", "biofind_balanced"])
    ap.add_argument("--features", default="common",
                    choices=["common"],
                    help="feature set (only 'common' supported here; 12 shared features)")
    args = ap.parse_args()

    out_dir = OUT_ROOT / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"=== {args.name} ===")
    print(f"  cohorts     : {args.cohorts}")
    print(f"  target      : {args.target}")
    print(f"  model       : {args.model}")
    print(f"  weights     : {args.cohort_weights or '(uniform)'}")
    print(f"  external    : {args.external}")

    ppmi = load_ppmi()
    cohorts_filter = parse_cohorts(args.cohorts)
    weights = parse_weights(args.cohort_weights)
    features = COMMON_FEATURES

    X, y, sw, df = prepare_xy(ppmi, args.target, cohorts_filter, features)
    sw = apply_cohort_weights(sw, df, weights)
    scale = args.model == "logreg"

    print(f"\n[train] n={len(X)}, pos={(y==1).sum()}, neg={(y==0).sum()}, "
          f"weighted_n={sw.sum():.1f}")
    cv = cv_evaluate(X, y, sw, args.model, scale=scale)
    print(f"  5-fold CV: Bal.Acc = {cv['bal_acc_mean']:.3f} ± {cv['bal_acc_sd']:.3f}, "
          f"AUC = {cv['auc_mean']:.3f} ± {cv['auc_sd']:.3f}")

    ext_result = None
    if args.external != "none" and args.target == "binary":
        if args.external == "biofind_balanced":
            bio = load_biofind_balanced()
            # Impute features missing in BioFIND with training-set medians
            for c in features:
                if c in bio.columns:
                    bio[c] = bio[c].fillna(bio[c].median())
                else:
                    bio[c] = df[c].median()
            X_bio = bio[features].values
            y_bio = bio["target_binary"].values
            label = "BioFIND (balanced, n=118)"
        else:  # biofind_russo
            bio = load_biofind_balanced()
            bio = bio[bio["target_binary"] == 1].copy()  # just the 103
            for c in features:
                if c in bio.columns:
                    bio[c] = bio[c].fillna(bio[c].median())
                else:
                    bio[c] = df[c].median()
            X_bio = bio[features].values
            y_bio = bio["target_binary"].values
            label = "BioFIND (Russo-only, n=103, all pos)"

        ext_result = train_full_and_apply(X, y, sw, X_bio, y_bio, args.model, scale=scale)
        print(f"\n[external] {label}")
        print(f"  n={ext_result['n_test']}, pos={ext_result['n_pos']}, neg={ext_result['n_neg']}")
        print(f"  Bal.Acc = {ext_result['bal_acc']:.3f}, AUC = {ext_result['auc']:.3f}")

    # Save
    summary = {
        "name": args.name,
        "seed": SEED,
        "cohorts_arg": args.cohorts,
        "cohorts_resolved": sorted(df["COHORT_DEFINITION"].dropna().unique().tolist())
            if "COHORT_DEFINITION" in df.columns else None,
        "target": args.target,
        "model": args.model,
        "cohort_weights": weights,
        "features": features,
        "n_train": int(len(X)),
        "n_pos": int((y == 1).sum()),
        "n_neg": int((y == 0).sum()),
        "cv": cv,
        "external_name": args.external,
        "external_result": ext_result,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[wrote] {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
