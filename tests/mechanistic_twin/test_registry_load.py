"""Gate test: DATA_LITERATURE_REGISTRY.md → mechanistic.* SQL tables."""
import pytest
from sqlalchemy import create_engine, text

ENGINE = create_engine("postgresql+psycopg2://blair.dupre@localhost:5432/giman_research")


def test_data_registry_table_exists():
    with ENGINE.connect() as conn:
        row = conn.execute(text(
            "SELECT to_regclass('mechanistic.data_registry')::text"
        )).scalar()
    assert row == "mechanistic.data_registry", "data_registry table not created"


def test_data_registry_has_gamma_entry():
    """GAMMA = 0.7 (Lee 2019) must be findable — it's a Phase 1 canonical param."""
    with ENGINE.connect() as conn:
        row = conn.execute(text(
            "SELECT parameter, value, source FROM mechanistic.data_registry "
            "WHERE parameter = 'GAMMA'"
        )).fetchone()
    assert row is not None
    assert float(row.value) == pytest.approx(0.7, rel=1e-6)
    assert "Lee 2019" in row.source


def test_empirical_findings_has_mollenhauer_csf_asyn():
    """The CSF α-syn vs SAA TTT ρ=-0.011 finding must be queryable."""
    with ENGINE.connect() as conn:
        row = conn.execute(text(
            "SELECT finding, metric_value FROM mechanistic.empirical_findings "
            "WHERE finding ILIKE '%CSF total α-syn vs SAA TTT%'"
        )).fetchone()
    assert row is not None
    assert abs(float(row.metric_value) - (-0.011)) < 0.005


def test_literature_anchors_has_rutledge2024():
    with ENGINE.connect() as conn:
        n = conn.execute(text(
            "SELECT COUNT(*) FROM mechanistic.literature_anchors "
            "WHERE source ILIKE '%Rutledge 2024%'"
        )).scalar()
    assert n >= 1
