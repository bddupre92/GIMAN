"""Compute bootstrap 95% CIs for macro-AUC on three-class and full-ordinal targets.
Addresses reviewer W9 (incomplete CI brackets in Table III / Table IV)."""
import json
from pathlib import Path

import numpy as np
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from giman_pipeline.data.db import read_sql

FEAT22 = [
    "sex", "handed", "age_at_baseline",
    "updrs1_total", "updrs2_total",
    "updrs3_tremor", "updrs3_rigidity", "updrs3_bradykinesia", "updrs3_axial",
    "updrs4_total", "moca_total", "rbd_total", "ess_total", "scopa_aut_total",
    "caudate_r_sbr", "caudate_l_sbr", "caudate_mean_sbr",
    "caudate_asymmetry", "caudate_putamen_ratio",
    "lrrk2_carrier", "gba_carrier", "apoe_e4_carrier",
]


def main() -> None:
    feat = ", ".join(FEAT22)
    q = f"""SELECT patno, target_3class, target_full_ordinal, target_nsd_positive, {feat}
            FROM features.paper1_features_with_targets"""
    df = read_sql(q)
    for c in FEAT22:
        df[c] = df[c].fillna(df[c].median())

    out = {}
    for target in ("target_3class", "target_full_ordinal", "target_nsd_positive"):
        sub = df[df[target] >= 0].copy()
        sub[target] = sub[target].map({v: i for i, v in enumerate(sorted(sub[target].unique()))})
        X, y = sub[FEAT22].values, sub[target].values
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        fold_aucs = []
        for tr, te in skf.split(X, y):
            model = CatBoostClassifier(
                iterations=1000, depth=6, learning_rate=0.05,
                auto_class_weights="Balanced", random_seed=42,
                verbose=False, allow_writing_files=False,
            )
            model.fit(X[tr], y[tr])
            p = model.predict_proba(X[te])
            fold_aucs.append(roc_auc_score(y[te], p, multi_class="ovr", average="macro"))
        aucs = np.array(fold_aucs)
        rng = np.random.default_rng(42)
        bs = np.array([rng.choice(aucs, len(aucs), replace=True).mean() for _ in range(1000)])
        ci_lo, ci_hi = np.percentile(bs, 2.5), np.percentile(bs, 97.5)
        out[target] = {
            "per_fold": aucs.tolist(),
            "mean": float(aucs.mean()),
            "ci95_lo": float(ci_lo),
            "ci95_hi": float(ci_hi),
        }
        print(f"{target}: {aucs.mean():.3f} [{ci_lo:.3f}, {ci_hi:.3f}]")

    out_path = Path("outputs/paper1_benchmark/multiclass_auc_ci.json")
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
