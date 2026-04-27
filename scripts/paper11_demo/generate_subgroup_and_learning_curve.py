"""Generate fig6 subgroup MAE + fig7 learning curve for Paper 11 npj-pd submission.

Sources (all Postgres-verified at runtime):
  - Per-patient test errors: mechanistic.paper11_sciml_results (baseline_v3 + learning-curve)
  - Demographics: data/07_paper3_features/longitudinal_features.csv (sex, lrrk2, gba, age)
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT_DIR = Path(
    "/Users/blair.dupre/Projects/CSCI-FALL-2025/outputs/mechanistic_twin/"
    "paper11_submission/npj-pd/figures"
)
OUT_DIR.mkdir(exist_ok=True, parents=True)

CONN = "postgresql://blair.dupre@localhost:5432/giman_research"

# Okabe-Ito palette
OI = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "purple": "#CC79A7",
    "grey": "#555555",
}

# ----- Load test-set per-patient errors: hybrid + pure_mech_fair -----
q = """
SELECT patno, model, abs_err
FROM mechanistic.paper11_sciml_results
WHERE config_id = 'baseline_v3' AND split = 'test'
  AND model IN ('hybrid', 'pure_mech_fair')
"""
err = pd.read_sql(q, CONN).pivot(index="patno", columns="model", values="abs_err").reset_index()
err.columns.name = None
print(f"Test patients with paired errors: {len(err)}")

# ----- Load demographics -----
df = pd.read_csv(
    "/Users/blair.dupre/Projects/CSCI-FALL-2025/data/07_paper3_features/"
    "longitudinal_features.csv"
)
# Baseline per patient
base = (
    df.sort_values(["PATNO", "months_from_baseline"]).groupby("PATNO").first().reset_index()
)
base = base[["PATNO", "age_at_visit", "sex", "lrrk2_carrier", "gba_carrier"]]
base = base.rename(columns={"PATNO": "patno"})
merged = err.merge(base, on="patno", how="left")
print(f"Merged rows: {len(merged)}  missing sex: {merged['sex'].isna().sum()}")

# Age bin
def age_bin(a):
    if pd.isna(a): return "NA"
    if a < 60: return "<60"
    if a < 70: return "60-70"
    return ">=70"

merged["age_group"] = merged["age_at_visit"].apply(age_bin)
merged["sex_label"] = merged["sex"].map({0: "Male", 1: "Female"}).fillna("NA")
merged["lrrk2_label"] = merged["lrrk2_carrier"].map({0: "LRRK2-", 1: "LRRK2+"}).fillna("NA")
merged["gba_label"] = merged["gba_carrier"].map({0: "GBA-", 1: "GBA+"}).fillna("NA")


def paired_bootstrap_delta(a: np.ndarray, b: np.ndarray, n=1000, seed=42):
    """Return (point estimate of mean(a - b), lo, hi) via percentile bootstrap."""
    if len(a) < 5:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    diffs = a - b
    boots = []
    for _ in range(n):
        idx = rng.integers(0, len(diffs), len(diffs))
        boots.append(diffs[idx].mean())
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(diffs.mean()), float(lo), float(hi)


# ----- Compute per-subgroup MAE + paired delta -----
subgroups = {
    "Sex": ("sex_label", ["Male", "Female"]),
    "Age": ("age_group", ["<60", "60-70", ">=70"]),
    "LRRK2": ("lrrk2_label", ["LRRK2-", "LRRK2+"]),
    "GBA": ("gba_label", ["GBA-", "GBA+"]),
}

summary = []
for panel, (col, levels) in subgroups.items():
    for lvl in levels:
        sub = merged[merged[col] == lvl]
        n = len(sub)
        if n == 0:
            continue
        h = sub["hybrid"].to_numpy()
        m = sub["pure_mech_fair"].to_numpy()
        mae_h = float(h.mean())
        mae_m = float(m.mean())
        d, lo, hi = paired_bootstrap_delta(h, m)
        summary.append({
            "panel": panel, "level": lvl, "n": int(n),
            "mae_hybrid": mae_h, "mae_puremech": mae_m,
            "delta": d, "delta_lo": lo, "delta_hi": hi,
        })

sdf = pd.DataFrame(summary)
print("\nSubgroup summary:")
print(sdf.to_string(index=False, float_format=lambda x: f"{x:.4f}" if isinstance(x, float) else str(x)))
sdf.to_csv(OUT_DIR / "fig6_subgroup_data.csv", index=False)

# ----- fig6: 2x2 subgroup MAE panel -----
fig, axes = plt.subplots(2, 2, figsize=(10, 7.5))
for ax, (panel, (col, levels)) in zip(axes.flat, subgroups.items()):
    rows = sdf[sdf["panel"] == panel]
    x = np.arange(len(rows))
    width = 0.35
    labels = [f"{r['level']}\n(n={r['n']})" for _, r in rows.iterrows()]
    ax.bar(x - width/2, rows["mae_hybrid"], width, color=OI["blue"],
           label="Hybrid (physics-informed)")
    ax.bar(x + width/2, rows["mae_puremech"], width, color=OI["grey"],
           label="Pure-mechanistic (fair)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Test MAE (SBR units)")
    ax.set_title(panel, fontsize=11, fontweight="bold")
    ax.axhline(0.2052, color="#333", lw=0.6, ls=":", alpha=0.5)
    ax.set_ylim(0, 0.30)
    # Annotate small-n caveat on single-level panels
    for _, r in rows.iterrows():
        if r["n"] < 5:
            ax.text(
                list(rows["level"]).index(r["level"]),
                max(r["mae_hybrid"], r["mae_puremech"]) + 0.015,
                f"N={int(r['n'])}", ha="center", fontsize=8, color="#AA0000"
            )

# Shared legend
axes[0,0].legend(loc="upper right", fontsize=8, frameon=False)
fig.suptitle(
    "Subgroup-stratified test MAE: hybrid vs. pure-mechanistic (fair) baseline",
    fontsize=12, fontweight="bold", y=0.99,
)
plt.tight_layout()
plt.savefig(OUT_DIR / "fig6_subgroup_mae.pdf", bbox_inches="tight")
plt.savefig(OUT_DIR / "fig6_subgroup_mae.png", bbox_inches="tight", dpi=300)
plt.close()
print(f"Saved: {OUT_DIR / 'fig6_subgroup_mae.pdf'}")

# ----- Learning curve: pull n=100 / n=200 / n=299 hybrid MAE -----
lc_q = """
SELECT config_id, test_mae, n_train
FROM mechanistic.paper11_sciml_summary
WHERE model = 'hybrid' AND use_gru = false
  AND config_id IN (
    'grid_lp0.1_lm0.1_n100',
    'grid_lp0.1_lm0.1_n200',
    'grid_lp0.1_lm0.1_mlp'
  )
ORDER BY n_train
"""
lc = pd.read_sql(lc_q, CONN)
print("\nLearning curve:")
print(lc.to_string(index=False))

pm_fair_mae = 0.2052

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(lc["n_train"], lc["test_mae"], "o-", color=OI["blue"], lw=2, markersize=8,
        label=f"Hybrid (physics-informed, λ_phys=λ_mono=0.1)")
ax.axhline(pm_fair_mae, color=OI["grey"], ls="--", lw=1.2,
           label=f"Pure-mechanistic (fair): {pm_fair_mae:.3f}")
for _, r in lc.iterrows():
    ax.annotate(
        f"{r['test_mae']:.3f}",
        (r["n_train"], r["test_mae"]),
        xytext=(0, 10), textcoords="offset points",
        ha="center", fontsize=9,
    )
ax.set_xlabel("Training-set size (n patients)")
ax.set_ylabel("Test MAE (SBR units)")
ax.set_title("Learning curve: hybrid-UDE test MAE vs. training-set size",
             fontsize=11, fontweight="bold")
ax.set_ylim(0.12, 0.22)
ax.set_xticks([100, 200, 299])
ax.legend(loc="upper right", fontsize=9, frameon=False)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR / "fig7_learning_curve.pdf", bbox_inches="tight")
plt.savefig(OUT_DIR / "fig7_learning_curve.png", bbox_inches="tight", dpi=300)
plt.close()
print(f"Saved: {OUT_DIR / 'fig7_learning_curve.pdf'}")

# Save summary JSON for audit
audit = {
    "subgroup_summary": summary,
    "learning_curve": lc.to_dict(orient="records"),
    "pure_mech_fair_mae": pm_fair_mae,
}
with open(OUT_DIR / "fig6_fig7_audit.json", "w") as f:
    json.dump(audit, f, indent=2, default=float)
print(f"Saved audit: {OUT_DIR / 'fig6_fig7_audit.json'}")
