"""Phase B B0 — claim-lineage SQLite schema tests."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB = PROJECT_ROOT / "outputs/defense_prep/e2e_audit/claim_lineage.sqlite3"

EXPECTED_TABLES = {
    "chapter",
    "claim",
    "citation",
    "citation_use",
    "numerical_claim",
    "code_artifact",
    "code_artifact_link",
    "test_link",
    "data_source",
    "data_source_link",
    "mempalace_link",
    "reviewer_flag",
}

EXPECTED_VIEWS = {
    "v_chapter_scorecard",
    "v_critical_flags",
    "v_ezproxy_queue",
}


@pytest.fixture(scope="module")
def conn():
    if not DB.exists():
        pytest.skip(f"DB not bootstrapped: {DB}")
    c = sqlite3.connect(DB)
    c.row_factory = sqlite3.Row
    yield c
    c.close()


def test_db_exists():
    assert DB.exists(), f"Bootstrap script not run: {DB}"


def test_all_expected_tables(conn):
    tables = {
        r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }
    missing = EXPECTED_TABLES - tables
    assert not missing, f"Missing tables: {missing}"


def test_all_expected_views(conn):
    views = {
        r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='view'")
    }
    missing = EXPECTED_VIEWS - views
    assert not missing, f"Missing views: {missing}"


def test_table_count_matches_plan(conn):
    """Plan specifies 12 user tables. sqlite_sequence is auto-created by
    SQLite for any table with AUTOINCREMENT and does not count."""
    tables = list(
        conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
    )
    assert len(tables) == 12, f"Plan says 12 user tables; got {len(tables)}"


def test_foreign_key_claim_chapter(conn):
    """Inserting a claim with non-existent chapter_id should fail when FK on.
    Non-destructive: uses a chapter_id guaranteed not to exist."""
    conn.execute("PRAGMA foreign_keys = ON")
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO claim (chapter_id, claim_type, claim_text) VALUES (9999, 'numerical', 'bad claim')"
        )


def test_views_return_something(conn):
    # Views should at least be queryable without error
    conn.execute("SELECT * FROM v_chapter_scorecard LIMIT 1").fetchall()
    conn.execute("SELECT * FROM v_critical_flags LIMIT 1").fetchall()
    conn.execute("SELECT * FROM v_ezproxy_queue LIMIT 1").fetchall()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
