"""Forest plot for Analysis E site-LOSO results."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path("outputs/paper1_site_loso")


def main() -> None:
    per_fold = json.loads((OUT_DIR / "site_loso_per_fold.json").read_text())
    summary = json.loads((OUT_DIR / "site_loso_summary.json").read_text())
    per_fold = [f for f in per_fold if not f["skipped"]]
    per_fold.sort(key=lambda f: f["auc"])

    fig, ax = plt.subplots(figsize=(8, 6))
    y = np.arange(len(per_fold))
    aucs = [f["auc"] for f in per_fold]
    labels = [f"{f['fold']} (n={f['n_test']}, +{f['n_pos']}/-{f['n_neg']})" for f in per_fold]

    # CIs where available
    for i, f in enumerate(per_fold):
        if f["ci95_lo"] is not None:
            ax.plot([f["ci95_lo"], f["ci95_hi"]], [i, i], color="#4a69bd", lw=2, zorder=2)
    ax.scatter(aucs, y, s=60, color="#1e3799", zorder=3)

    # Pre-registered thresholds
    ax.axvline(0.90, color="#e55039", ls="--", lw=1, alpha=0.7, label="min-AUC threshold (0.90)")
    ax.axvline(summary["mean_auc"], color="#38ada9", ls=":", lw=1.5, alpha=0.9,
               label=f"pooled mean ({summary['mean_auc']:.3f})")
    ax.axvline(0.979, color="#78e08f", ls=":", lw=1.5, alpha=0.7,
               label="Paper 1 headline (0.979)")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("AUC (binary NSD-ISS)")
    ax.set_xlim(0.60, 1.02)
    verdict_color = "#e55039" if summary["verdict"] == "FAIL" else "#38ada9"
    ax.set_title(
        f"Analysis E site-LOSO (n=647, {summary['n_folds_valid']} folds) — verdict: {summary['verdict']}",
        color=verdict_color, fontsize=11,
    )
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    for ext in ("pdf", "png"):
        out = OUT_DIR / f"loso_forest_plot.{ext}"
        plt.savefig(out, dpi=300, bbox_inches="tight")
        print(f"Wrote {out}")
    plt.close()


if __name__ == "__main__":
    main()
