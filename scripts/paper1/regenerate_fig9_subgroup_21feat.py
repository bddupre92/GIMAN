"""Regenerate Fig 9 panel (b) — per-genotype subgroup AUC on 21-feat primary.

Uses corrected carrier flags post-commit 672b439:
  LRRK2+ (any)        = 175
  GBA+ only           = 104  (GBA+ AND NOT LRRK2+)
  APOE+ only          = 375  (APOE+ AND NOT LRRK2+ AND NOT GBA+)
  Non-carrier         = 1547

Full 21-feat Path 3 CatBoost 5-fold stratified CV with 1,000-sample
patient-level bootstrap 95% CIs. Output: fig9_shap_subgroup.pdf/png
(panel (b) regenerated; panel (a) SHAP importance unchanged since it's on
22-feat reference, retained for transparency).

Output: outputs/mechanistic_twin/paper1_submission/ieee-jbhi/figures/fig9b_subgroup_21feat.{pdf,png}
         Also updates subgroup JSON at outputs/paper1_shap_subgroup/subgroup_binary_21feat.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path("/Users/blair.dupre/Projects/CSCI-FALL-2025")
sys.path.insert(0, str(ROOT))

from scripts.paper1.run_fold_local_imputation import (  # noqa: E402
    STAGING_COLS, HIGH_MISS_COLS, FEATURES_PATH,
)

OUT_FIG_DIR = ROOT / "outputs/mechanistic_twin/paper1_submission/ieee-jbhi/figures"
OUT_JSON = ROOT / "outputs/paper1_shap_subgroup/subgroup_binary_21feat.json"

N_FOLDS = 5
CV_SEED = 42
BOOT_N = 1000
PATH3_EXCLUDE = {"CAUDATE_PUTAMEN_RATIO"}

# Okabe-Ito
C_REF = "#0072B2"  # blue non-carrier
C_LRRK2 = "#E69F00"  # orange
C_GBA = "#CC79A7"  # pink
C_APOE = "#009E73"  # green


def main() -> None:
    df = pd.read_csv(FEATURES_PATH)
    feat_cols = [c for c in df.columns
                  if c not in STAGING_COLS and c not in HIGH_MISS_COLS
                  and c not in PATH3_EXCLUDE]
    mask = df["target_binary"] >= 0
    sub = df[mask].copy().reset_index(drop=True)
    X = sub[feat_cols].to_numpy(dtype=float)
    y = sub["target_binary"].astype(int).to_numpy()

    lrrk2 = sub["LRRK2_CARRIER"].fillna(0).astype(bool).values
    gba = sub["GBA_CARRIER"].fillna(0).astype(bool).values
    apoe = sub["APOE_E4_CARRIER"].fillna(0).astype(bool).values

    strata = {
        "LRRK2+": lrrk2,
        "GBA+ only": gba & ~lrrk2,
        "APOE-ε4+ only": apoe & ~lrrk2 & ~gba,
        "Non-carrier": ~lrrk2 & ~gba & ~apoe,
    }
    for s, m in strata.items():
        print(f"{s}: n={m.sum()}")

    # OOF
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=CV_SEED)
    oof = np.zeros((len(y), 2))
    for tr, te in skf.split(X, y):
        imp = SimpleImputer(strategy="median")
        X_tr = imp.fit_transform(X[tr]); X_te = imp.transform(X[te])
        sc = StandardScaler()
        X_tr = sc.fit_transform(X_tr); X_te = sc.transform(X_te)
        clf = CatBoostClassifier(iterations=500, depth=6, learning_rate=0.05,
                                   random_seed=42, verbose=False,
                                   auto_class_weights="Balanced")
        clf.fit(X_tr, y[tr])
        oof[te] = clf.predict_proba(X_te)

    rng = np.random.default_rng(CV_SEED)
    results = {}
    for s, m in strata.items():
        y_s = y[m]
        p_s = oof[m, 1]
        if len(np.unique(y_s)) < 2 or len(y_s) < 10:
            results[s] = {"n": int(m.sum()), "auc": None,
                            "ci_low": None, "ci_high": None}
            continue
        auc = roc_auc_score(y_s, p_s)
        boots = []
        for _ in range(BOOT_N):
            idx = rng.integers(0, len(y_s), len(y_s))
            try:
                boots.append(roc_auc_score(y_s[idx], p_s[idx]))
            except ValueError:
                continue
        results[s] = {
            "n": int(m.sum()),
            "auc": float(auc),
            "ci_low": float(np.percentile(boots, 2.5)),
            "ci_high": float(np.percentile(boots, 97.5)),
        }
        print(f"  {s:20s} n={m.sum():4d}  AUC={auc:.4f} [{results[s]['ci_low']:.3f},{results[s]['ci_high']:.3f}]")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps({"workstream": "q_r2_subgroup_carrier_21feat",
                                     "target": "binary",
                                     "feature_spec": "21-feat Path3 strict circularity",
                                     "per_group_auc": results}, indent=2))
    print(f"Wrote {OUT_JSON}")

    # Forest plot
    strat_names = list(results.keys())
    aucs = [results[s]["auc"] for s in strat_names]
    lo = [results[s]["ci_low"] for s in strat_names]
    hi = [results[s]["ci_high"] for s in strat_names]
    colors = [C_LRRK2, C_GBA, C_APOE, C_REF]

    fig, ax = plt.subplots(figsize=(5.8, 3.5), dpi=300)
    y_pos = np.arange(len(strat_names))[::-1]
    ax.errorbar(aucs, y_pos, xerr=[[a - l for a, l in zip(aucs, lo)],
                                      [h - a for a, h in zip(aucs, hi)]],
                 fmt="o", capsize=4, markersize=9,
                 ecolor="#333333", markerfacecolor=colors[0],
                 markeredgecolor="black", markeredgewidth=0.6, linewidth=1.2)

    # Set different marker colors per stratum
    for i, (yp, c, a) in enumerate(zip(y_pos, colors, aucs)):
        ax.plot(a, yp, "o", markersize=11, markerfacecolor=c,
                 markeredgecolor="black", markeredgewidth=0.6)
    # Non-carrier reference shaded band
    ref_lo = results["Non-carrier"]["ci_low"]
    ref_hi = results["Non-carrier"]["ci_high"]
    ax.axvspan(ref_lo, ref_hi, alpha=0.15, color=C_REF, zorder=0,
                label="Non-carrier 95% CI")

    # Labels with n
    yticklabels = [f"{s}\n(n={results[s]['n']})" for s in strat_names]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(yticklabels, fontsize=9)
    ax.set_xlabel("Binary NSD+ AUC [95% bootstrap CI]", fontsize=10)
    ax.set_xlim(0.82, 1.0)
    ax.axvline(0.90, color="grey", linestyle=":", linewidth=0.5, alpha=0.5)
    ax.grid(axis="x", linestyle="-", linewidth=0.4, alpha=0.25)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title("Per-genotype subgroup AUC (21-feature primary)",
                  fontsize=10, pad=8)

    fig.tight_layout()
    for suffix in ("pdf", "png"):
        out = OUT_FIG_DIR / f"fig9b_subgroup_21feat.{suffix}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
