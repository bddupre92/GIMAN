#!/usr/bin/env python3
"""
Step 7: Comprehensive Benchmark Suite for Paper 3.

Consolidates results from all three stage transition models:
    1. Multi-State Markov (interpretable baseline)
    2. Dynamic-DeepHit (temporal deep learning)
    3. Graph-Informed Digital Twin (graph + temporal)

Produces:
    - Model comparison table (C-td, IBS, Brier@horizons)
    - Per-transition C-td breakdown
    - Paired statistical tests across folds
    - Cohort characteristics summary
    - Sojourn time comparison (Markov vs Simuni 2025)

Usage:
    python scripts/paper3/run_benchmark.py

Outputs:
    outputs/paper3_benchmark/
        benchmark_summary.json      — All metrics in one file
        model_comparison.csv        — Table I: Model comparison
        per_transition_ctd.csv      — Table II: Per-transition C-td
        brier_at_horizons.csv       — Table III: Calibration
        statistical_tests.csv       — Paired tests between models
        sojourn_comparison.csv      — Markov vs Simuni (2025)
        cohort_summary.json         — Data characteristics
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "paper3_benchmark"

# ── Load Results ──────────────────────────────────────────────────────

def load_results():
    """Load results from all three models."""
    results = {}

    # Markov
    markov_path = PROJECT_ROOT / "outputs" / "paper3_markov" / "markov_results.json"
    if markov_path.exists():
        with open(markov_path) as f:
            results["markov"] = json.load(f)
        print(f"  Markov: loaded ({results['markov']['n_patients']} patients)")
    else:
        print("  WARNING: Markov results not found")

    # DeepHit
    deephit_path = PROJECT_ROOT / "outputs" / "paper3_deephit" / "deephit_results.json"
    if deephit_path.exists():
        with open(deephit_path) as f:
            results["deephit"] = json.load(f)
        print(f"  DeepHit: loaded (C-td={results['deephit']['c_td']:.4f})")
    else:
        print("  WARNING: DeepHit results not found")

    # Graph-DT
    gdt_path = PROJECT_ROOT / "outputs" / "paper3_graph_dt" / "graph_dt_results.json"
    if gdt_path.exists():
        with open(gdt_path) as f:
            results["graph_dt"] = json.load(f)
        print(f"  Graph-DT: loaded (C-td={results['graph_dt']['c_td']:.4f})")
    else:
        print("  WARNING: Graph-DT results not found")

    return results


# ── Model Comparison Table ────────────────────────────────────────────

def build_comparison_table(results):
    """Table I: Overall model comparison."""
    rows = []

    # Markov — no C-td/IBS (not a survival model, but has transition probs)
    if "markov" in results:
        m = results["markov"]
        rows.append({
            "Model": "Multi-State Markov",
            "C-td": "—",
            "C-td (95% CI)": "—",
            "IBS": "—",
            "IBS (95% CI)": "—",
            "Brier 1yr": "—",
            "Brier 5yr": "—",
            "Brier 10yr": "—",
            "Parameters": "Q matrix (20 intensities)",
            "Training Time": "< 1 min",
        })

    for name, key in [("Dynamic-DeepHit", "deephit"), ("Graph Digital Twin", "graph_dt")]:
        if key not in results:
            continue
        r = results[key]
        folds = r["c_td_per_fold"]
        ibs_folds = r["ibs_per_fold"]
        bh = r["brier_at_horizons"]

        # Bootstrap-style CI from folds (mean ± 1.96 * SE)
        se_ctd = np.std(folds) / np.sqrt(len(folds))
        ci_ctd = f"[{np.mean(folds) - 1.96*se_ctd:.3f}–{np.mean(folds) + 1.96*se_ctd:.3f}]"
        se_ibs = np.std(ibs_folds) / np.sqrt(len(ibs_folds))
        ci_ibs = f"[{np.mean(ibs_folds) - 1.96*se_ibs:.4f}–{np.mean(ibs_folds) + 1.96*se_ibs:.4f}]"

        hp = r.get("hyperparams", {})
        n_params = _estimate_params(hp, key)

        rows.append({
            "Model": name,
            "C-td": f"{r['c_td']:.4f} ± {r['c_td_std']:.4f}",
            "C-td (95% CI)": ci_ctd,
            "IBS": f"{r['ibs']:.4f} ± {r['ibs_std']:.4f}",
            "IBS (95% CI)": ci_ibs,
            "Brier 1yr": f"{bh.get('1yr', float('nan')):.4f}",
            "Brier 5yr": f"{bh.get('5yr', float('nan')):.4f}",
            "Brier 10yr": f"{bh.get('10yr', float('nan')):.4f}",
            "Parameters": n_params,
            "Training Time": _format_time(r),
        })

    return pd.DataFrame(rows)


def _estimate_params(hp, model_type):
    """Rough parameter count estimate."""
    hd = hp.get("hidden_dim", 128)
    inp = hp.get("input_dim", 22)
    if model_type == "deephit":
        # GRU + output head
        gru = 3 * hd * (inp + hd + 1) * hp.get("n_gru_layers", 2)
        head = hd * 128 + 128 * 64 + 64 * 78
        return f"~{(gru + head) // 1000}K"
    elif model_type == "graph_dt":
        gru = 3 * hd * (inp + hd + 1) * hp.get("n_gru_layers", 2)
        # node_encoder + GAT + gate + attn + head
        ne = 18 * 64 + 64 * hd
        gat = hd * hd * hp.get("gat_layers", 2) * hp.get("gat_heads", 4)
        gate = hd * 2 * hd
        attn = hd * (hd // 2) + (hd // 2)
        head = hd * 128 + 128 * 64 + 64 * 78
        return f"~{(gru + ne + gat + gate + attn + head) // 1000}K"
    return "—"


def _format_time(r):
    """Format total training time from hyperparams."""
    # Estimate from results if not stored
    return "—"


# ── Per-Transition C-td Table ─────────────────────────────────────────

def build_transition_table(results):
    """Table II: Per-transition discriminative performance."""
    transitions = ["→0", "→1", "→2B", "→3", "→4", "→5", "→6"]

    rows = []
    for trans in transitions:
        row = {"Transition": trans}
        for name, key in [("DeepHit", "deephit"), ("Graph-DT", "graph_dt")]:
            if key in results and "per_transition_ctd" in results[key]:
                ptc = results[key]["per_transition_ctd"]
                val = ptc.get(trans, ptc.get(f"→{trans[1:]}", None))
                row[name] = f"{val:.4f}" if val is not None else "—"
            else:
                row[name] = "—"
        # Delta
        if row.get("DeepHit", "—") != "—" and row.get("Graph-DT", "—") != "—":
            dh = float(row["DeepHit"])
            gd = float(row["Graph-DT"])
            delta = gd - dh
            row["Δ (Graph-DT − DeepHit)"] = f"{delta:+.4f}"
        else:
            row["Δ (Graph-DT − DeepHit)"] = "—"
        rows.append(row)

    return pd.DataFrame(rows)


# ── Brier Scores at Horizons ─────────────────────────────────────────

def build_brier_table(results):
    """Table III: Calibration at clinical horizons."""
    horizons = ["1yr", "2yr", "5yr", "10yr"]
    rows = []
    for h in horizons:
        row = {"Horizon": h}
        for name, key in [("DeepHit", "deephit"), ("Graph-DT", "graph_dt")]:
            if key in results:
                bh = results[key].get("brier_at_horizons", {})
                val = bh.get(h)
                row[name] = f"{val:.4f}" if val is not None else "—"
            else:
                row[name] = "—"
        rows.append(row)
    return pd.DataFrame(rows)


# ── Statistical Tests ─────────────────────────────────────────────────

def paired_tests(results):
    """Paired statistical comparisons across folds."""
    tests = []

    if "deephit" in results and "graph_dt" in results:
        dh = results["deephit"]["c_td_per_fold"]
        gd = results["graph_dt"]["c_td_per_fold"]

        if len(dh) == len(gd):
            # Paired t-test
            t_stat, p_val = stats.ttest_rel(dh, gd)
            tests.append({
                "Comparison": "DeepHit vs Graph-DT (C-td)",
                "Test": "Paired t-test",
                "Statistic": f"{t_stat:.4f}",
                "p-value": f"{p_val:.4f}",
                "Significant (α=0.05)": "Yes" if p_val < 0.05 else "No",
                "Mean Diff": f"{np.mean(dh) - np.mean(gd):+.4f}",
                "Interpretation": _interpret_ctd_test(dh, gd, p_val),
            })

            # Wilcoxon signed-rank (non-parametric)
            try:
                w_stat, w_p = stats.wilcoxon(dh, gd)
                tests.append({
                    "Comparison": "DeepHit vs Graph-DT (C-td)",
                    "Test": "Wilcoxon signed-rank",
                    "Statistic": f"{w_stat:.4f}",
                    "p-value": f"{w_p:.4f}",
                    "Significant (α=0.05)": "Yes" if w_p < 0.05 else "No",
                    "Mean Diff": f"{np.mean(dh) - np.mean(gd):+.4f}",
                    "Interpretation": _interpret_ctd_test(dh, gd, w_p),
                })
            except ValueError:
                pass  # All differences are zero

            # IBS comparison
            dh_ibs = results["deephit"]["ibs_per_fold"]
            gd_ibs = results["graph_dt"]["ibs_per_fold"]
            t_ibs, p_ibs = stats.ttest_rel(dh_ibs, gd_ibs)
            tests.append({
                "Comparison": "DeepHit vs Graph-DT (IBS)",
                "Test": "Paired t-test",
                "Statistic": f"{t_ibs:.4f}",
                "p-value": f"{p_ibs:.4f}",
                "Significant (α=0.05)": "Yes" if p_ibs < 0.05 else "No",
                "Mean Diff": f"{np.mean(dh_ibs) - np.mean(gd_ibs):+.4f}",
                "Interpretation": "Lower IBS is better (better calibration)" if np.mean(dh_ibs) < np.mean(gd_ibs) else "Models have comparable calibration",
            })

    return pd.DataFrame(tests) if tests else pd.DataFrame()


def _interpret_ctd_test(dh_folds, gd_folds, p_val):
    diff = np.mean(dh_folds) - np.mean(gd_folds)
    if p_val >= 0.05:
        return f"No significant difference (Δ={diff:+.4f}, p={p_val:.3f})"
    elif diff > 0:
        return f"DeepHit significantly better (Δ={diff:+.4f}, p={p_val:.3f})"
    else:
        return f"Graph-DT significantly better (Δ={diff:+.4f}, p={p_val:.3f})"


# ── Sojourn Time Comparison ──────────────────────────────────────────

def build_sojourn_comparison(results):
    """Compare Markov sojourn times with Simuni et al. (2025) KM estimates."""
    # Simuni 2025 reported median transition times (KM)
    simuni_2025 = {
        "2B→3": {"median_years": 1.19, "ci_lower": 1.1, "ci_upper": 2.0},
        "3→4": {"median_years": 4.98, "ci_lower": 4.1, "ci_upper": 5.4},
        "4→5": {"median_years": 9.83, "ci_lower": 7.0, "ci_upper": None},
    }

    rows = []
    if "markov" in results:
        m = results["markov"]
        sojourn = m.get("sojourn_times", {})
        boot = m.get("bootstrap_ci", {}).get("sojourn_ci", {})

        for stage, soj_years in sojourn.items():
            row = {
                "Stage": stage,
                "Markov Sojourn (years)": f"{soj_years:.2f}",
            }
            if stage in boot:
                ci = boot[stage]
                row["Markov 95% CI"] = f"[{ci['ci_lower']:.2f}–{ci['ci_upper']:.2f}]"
            else:
                row["Markov 95% CI"] = "—"

            # Match with Simuni for relevant transitions
            if stage == "2B":
                s = simuni_2025["2B→3"]
                row["Simuni 2025 (KM median)"] = f"{s['median_years']:.2f}"
                row["Simuni 95% CI"] = f"[{s['ci_lower']:.1f}–{s['ci_upper']:.1f}]"
            elif stage == "3":
                s = simuni_2025["3→4"]
                row["Simuni 2025 (KM median)"] = f"{s['median_years']:.2f}"
                row["Simuni 95% CI"] = f"[{s['ci_lower']:.1f}–{s['ci_upper']:.1f}]"
            elif stage == "4":
                s = simuni_2025["4→5"]
                row["Simuni 2025 (KM median)"] = f"{s['median_years']:.2f}"
                ci_upper = f"{s['ci_upper']:.1f}" if s["ci_upper"] else "NA"
                row["Simuni 95% CI"] = f"[{s['ci_lower']:.1f}–{ci_upper}]"
            else:
                row["Simuni 2025 (KM median)"] = "—"
                row["Simuni 95% CI"] = "—"

            rows.append(row)

    return pd.DataFrame(rows)


# ── Cohort Summary ───────────────────────────────────────────────────

def build_cohort_summary(results):
    """Data characteristics for the Methods section."""
    summary = {}

    # Load transition events for cohort stats
    trans_path = PROJECT_ROOT / "data" / "06_longitudinal_staging" / "transition_events.csv"
    if trans_path.exists():
        trans = pd.read_csv(trans_path)
        summary["n_transitions"] = len(trans)
        summary["forward_transitions"] = int((trans["direction"] == "forward").sum()) if "direction" in trans.columns else "—"
        summary["backward_transitions"] = int((trans["direction"] == "backward").sum()) if "direction" in trans.columns else "—"

    # From model results
    for key in ["deephit", "graph_dt"]:
        if key in results:
            r = results[key]
            summary["n_episodes"] = r.get("n_episodes", "—")
            summary["n_events"] = r.get("n_events", "—")
            summary["n_censored"] = r.get("n_censored", "—")
            break

    if "markov" in results:
        m = results["markov"]
        summary["n_patients"] = m.get("n_patients", "—")
        summary["n_observations"] = m.get("n_observations", "—")
        summary["markov_transitions"] = m.get("n_transitions", "—")

    if "graph_dt" in results:
        gs = results["graph_dt"].get("graph_stats", {})
        summary["graph_nodes"] = gs.get("n_nodes", "—")
        summary["graph_edges"] = gs.get("n_edges", "—")
        summary["graph_avg_degree"] = gs.get("avg_degree", "—")
        summary["graph_k_neighbors"] = gs.get("k_neighbors", "—")
        summary["graph_baseline_features"] = gs.get("n_baseline_features", "—")

    # Load longitudinal features for more stats
    feat_path = PROJECT_ROOT / "data" / "07_paper3_features" / "longitudinal_features.csv"
    if feat_path.exists():
        feat = pd.read_csv(feat_path, nrows=5)
        summary["n_features"] = len(feat.columns)

    # Load cohort summary if exists
    cohort_path = PROJECT_ROOT / "data" / "06_longitudinal_staging" / "cohort_summary.json"
    if cohort_path.exists():
        with open(cohort_path) as f:
            cs = json.load(f)
        summary["cohort"] = cs

    return summary


# ── Main ──────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("PAPER 3: BENCHMARK SUITE")
    print("=" * 70)

    print("\nLoading results...")
    results = load_results()

    if not results:
        print("ERROR: No results found. Run models first.")
        return

    # 1. Model comparison table
    print("\n── Table I: Model Comparison ──")
    comp_table = build_comparison_table(results)
    print(comp_table.to_string(index=False))
    comp_table.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)

    # 2. Per-transition C-td
    print("\n── Table II: Per-Transition C-td ──")
    trans_table = build_transition_table(results)
    print(trans_table.to_string(index=False))
    trans_table.to_csv(OUTPUT_DIR / "per_transition_ctd.csv", index=False)

    # 3. Brier scores at horizons
    print("\n── Table III: Brier Scores at Clinical Horizons ──")
    brier_table = build_brier_table(results)
    print(brier_table.to_string(index=False))
    brier_table.to_csv(OUTPUT_DIR / "brier_at_horizons.csv", index=False)

    # 4. Statistical tests
    print("\n── Statistical Tests ──")
    test_table = paired_tests(results)
    if not test_table.empty:
        print(test_table.to_string(index=False))
        test_table.to_csv(OUTPUT_DIR / "statistical_tests.csv", index=False)
    else:
        print("  No paired tests available (need both DeepHit and Graph-DT)")

    # 5. Sojourn comparison
    print("\n── Sojourn Times: Markov vs Simuni (2025) ──")
    soj_table = build_sojourn_comparison(results)
    if not soj_table.empty:
        print(soj_table.to_string(index=False))
        soj_table.to_csv(OUTPUT_DIR / "sojourn_comparison.csv", index=False)

    # 6. Cohort summary
    print("\n── Cohort Summary ──")
    cohort = build_cohort_summary(results)
    for k, v in cohort.items():
        if k != "cohort":
            print(f"  {k}: {v}")

    # 7. Save everything
    benchmark_summary = {
        "models": {},
        "cohort": cohort,
    }
    for name, key in [("markov", "markov"), ("deephit", "deephit"), ("graph_dt", "graph_dt")]:
        if key in results:
            r = results[key]
            benchmark_summary["models"][name] = {
                "c_td": r.get("c_td"),
                "c_td_std": r.get("c_td_std"),
                "c_td_per_fold": r.get("c_td_per_fold"),
                "ibs": r.get("ibs"),
                "ibs_std": r.get("ibs_std"),
                "ibs_per_fold": r.get("ibs_per_fold"),
                "brier_at_horizons": r.get("brier_at_horizons"),
                "per_transition_ctd": r.get("per_transition_ctd"),
            }

    # Add statistical test results
    if not test_table.empty:
        benchmark_summary["statistical_tests"] = test_table.to_dict(orient="records")

    with open(OUTPUT_DIR / "benchmark_summary.json", "w") as f:
        json.dump(benchmark_summary, f, indent=2, default=str)

    print(f"\n  Results saved to {OUTPUT_DIR}")

    # Print key findings
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)

    if "deephit" in results and "graph_dt" in results:
        dh = results["deephit"]
        gd = results["graph_dt"]

        print(f"\n  Dynamic-DeepHit:    C-td = {dh['c_td']:.4f} ± {dh['c_td_std']:.4f}")
        print(f"  Graph Digital Twin: C-td = {gd['c_td']:.4f} ± {gd['c_td_std']:.4f}")

        # Fold-by-fold comparison
        dh_folds = dh["c_td_per_fold"]
        gd_folds = gd["c_td_per_fold"]
        n_gd_wins = sum(1 for d, g in zip(dh_folds, gd_folds) if g > d)
        print(f"\n  Graph-DT wins {n_gd_wins}/{len(dh_folds)} folds")

        _, p_val = stats.ttest_rel(dh_folds, gd_folds)
        if p_val >= 0.05:
            print(f"  Paired t-test: p = {p_val:.3f} (NOT significant)")
            print("  → Models have comparable discriminative performance")
        else:
            winner = "DeepHit" if np.mean(dh_folds) > np.mean(gd_folds) else "Graph-DT"
            print(f"  Paired t-test: p = {p_val:.3f} ({winner} significantly better)")

        # Per-transition insights
        print("\n  Per-transition insights:")
        dh_ptc = dh.get("per_transition_ctd", {})
        gd_ptc = gd.get("per_transition_ctd", {})
        for trans in sorted(set(list(dh_ptc.keys()) + list(gd_ptc.keys()))):
            dh_v = dh_ptc.get(trans)
            gd_v = gd_ptc.get(trans)
            if dh_v is not None and gd_v is not None:
                delta = gd_v - dh_v
                winner = "Graph-DT" if delta > 0 else "DeepHit"
                print(f"    {trans}: DeepHit={dh_v:.3f}, Graph-DT={gd_v:.3f} "
                      f"(Δ={delta:+.3f}, {winner})")

    if "markov" in results:
        m = results["markov"]
        print(f"\n  Markov model: {m['n_transitions']} transitions across "
              f"{m['n_patients']} patients")
        print(f"  Sojourn times (years): "
              f"2B={m['sojourn_times']['2B']:.2f}, "
              f"3={m['sojourn_times']['3']:.2f}, "
              f"4={m['sojourn_times']['4']:.2f}")
        print(f"  vs Simuni 2025: 2B→3: 1.19yr, 3→4: 4.98yr, 4→5: 9.83yr")

    print("\nDone!")


if __name__ == "__main__":
    main()
