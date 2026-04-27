"""Tests for §9.6 publication figures (Task 8)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIGS = ROOT / "outputs/mechanistic_twin/ch9_6/figures"


def test_all_six_figures_exist():
    stems = [
        "fig_9_6_1_channel_map",
        "fig_9_6_2_fim_ablation",
        "fig_9_6_3_loo_coverage",
        "fig_9_6_4_nfl_holdout",
        "fig_9_6_5_posterior_compare",
        "fig_9_6_6_progressor_strata",
    ]
    for stem in stems:
        for ext in ["png", "pdf"]:
            p = FIGS / f"{stem}.{ext}"
            assert p.exists(), f"missing figure: {p}"


def test_figures_non_empty():
    for f in FIGS.glob("fig_9_6_*.png"):
        assert f.stat().st_size > 10_000, f"{f.name} < 10 KB — likely empty"
