"""Tests for ch9_6_extract_gfap.py — verify mechanistic.ch9_6_gfap_longitudinal.

Source: ppmi_raw.current_biospecimen_analysis_results (Project 152, Batria-Utermann)
Values stored as log2(ng/mL) in npx column for compatibility with Olink NPX convention.
371 patients, 1,604 CSF GFAP measurements across 11 clinical events.
"""
from sqlalchemy import create_engine, text

ENGINE = create_engine("postgresql+psycopg2://blair.dupre@localhost:5432/giman_research")


def test_gfap_table_exists():
    with ENGINE.connect() as conn:
        x = conn.execute(text(
            "SELECT to_regclass('mechanistic.ch9_6_gfap_longitudinal')::text"
        )).scalar()
    assert x == "mechanistic.ch9_6_gfap_longitudinal"


def test_gfap_has_at_least_150_patients():
    """Project 152 has 371 patients with CSF GFAP — expect most survive QC."""
    with ENGINE.connect() as conn:
        n = conn.execute(text(
            "SELECT COUNT(DISTINCT patno) FROM mechanistic.ch9_6_gfap_longitudinal"
        )).scalar()
    assert n >= 150, f"only {n} patients with GFAP — expected >= 150"


def test_gfap_npx_values_in_range():
    """Values stored as log2(ng/mL). GFAP in PPMI CSF is 0.337–33.99 ng/mL,
    so log2 range is approx [-1.6, 5.1]. Assert within [-10, 20] with mean in [0, 5]."""
    with ENGINE.connect() as conn:
        row = conn.execute(text(
            "SELECT MIN(npx), MAX(npx), AVG(npx) FROM mechanistic.ch9_6_gfap_longitudinal"
        )).fetchone()
    assert -10 < row[0] < row[1] < 20, f"NPX range [{row[0]:.3f}, {row[1]:.3f}] unexpected"
    assert 0 < row[2] < 5, f"mean NPX {row[2]:.3f} out of expected range [0, 5]"


def test_gfap_has_longitudinal_structure():
    """Need multiple visits per patient. Project 152 covers BL+V02+V04+V06+V08+V10."""
    with ENGINE.connect() as conn:
        n_multi = conn.execute(text(
            "SELECT COUNT(*) FROM (SELECT patno, COUNT(*) as c "
            "FROM mechanistic.ch9_6_gfap_longitudinal GROUP BY patno HAVING COUNT(*) >= 2) s"
        )).scalar()
    assert n_multi >= 80, f"only {n_multi} patients with >=2 visits — may be cross-sectional"
