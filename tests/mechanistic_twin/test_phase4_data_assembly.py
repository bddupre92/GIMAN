"""Unit tests for Phase 4 data assembly: LEDD + UPDRS-III + Part IV + posteriors.

Tests cover:
1. parse_mmyyyy() — MM/YYYY date string → datetime
2. compute_visit_ledd() — per-visit LEDD summation with date windowing
3. filter_off_state() — OFF-state UPDRS-III filtering logic

Run: conda run -n base python -m pytest tests/mechanistic_twin/test_phase4_data_assembly.py -v
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = REPO_ROOT / "scripts" / "mechanistic_twin"
sys.path.insert(0, str(SCRIPT_DIR))

# Import the functions under test — the script uses importable module-level functions
import phase4_assemble_ledd_updrs as p4  # noqa: E402


# =====================================================================
# parse_mmyyyy
# =====================================================================

class TestParseMmyyyy:
    """Tests for the MM/YYYY date parser."""

    def test_valid_date(self):
        assert p4.parse_mmyyyy("02/2023") == datetime(2023, 2, 1)

    def test_december(self):
        assert p4.parse_mmyyyy("12/2020") == datetime(2020, 12, 1)

    def test_january(self):
        assert p4.parse_mmyyyy("01/2011") == datetime(2011, 1, 1)

    def test_nan_returns_none(self):
        assert p4.parse_mmyyyy(np.nan) is None

    def test_empty_string_returns_none(self):
        assert p4.parse_mmyyyy("") is None

    def test_none_returns_none(self):
        assert p4.parse_mmyyyy(None) is None

    def test_whitespace_returns_none(self):
        assert p4.parse_mmyyyy("  ") is None


# =====================================================================
# compute_visit_ledd
# =====================================================================

class TestComputeVisitLedd:
    """Tests for per-visit LEDD computation."""

    def _make_ledd_df(self, rows: list[dict]) -> pd.DataFrame:
        """Helper to build a LEDD DataFrame from simple dicts."""
        return pd.DataFrame(rows)

    def test_single_med_active(self):
        """One medication active at the visit date."""
        ledd_df = self._make_ledd_df([
            {"PATNO": "3001", "LEDD_numeric": 700.0,
             "start_dt": datetime(2020, 7, 1), "stop_dt": None},
        ])
        visit_date = datetime(2021, 1, 1)
        result = p4.compute_visit_ledd("3001", visit_date, ledd_df)
        assert result == 700.0

    def test_multiple_meds_active(self):
        """Two medications active — should sum."""
        ledd_df = self._make_ledd_df([
            {"PATNO": "3001", "LEDD_numeric": 700.0,
             "start_dt": datetime(2020, 7, 1), "stop_dt": None},
            {"PATNO": "3001", "LEDD_numeric": 200.0,
             "start_dt": datetime(2020, 7, 1), "stop_dt": None},
        ])
        visit_date = datetime(2021, 1, 1)
        result = p4.compute_visit_ledd("3001", visit_date, ledd_df)
        assert result == 900.0

    def test_before_start_date(self):
        """Visit before medication start — LEDD should be 0."""
        ledd_df = self._make_ledd_df([
            {"PATNO": "3001", "LEDD_numeric": 700.0,
             "start_dt": datetime(2022, 1, 1), "stop_dt": None},
        ])
        visit_date = datetime(2021, 1, 1)
        result = p4.compute_visit_ledd("3001", visit_date, ledd_df)
        assert result == 0.0

    def test_after_stop_date(self):
        """Visit after medication stop — LEDD should be 0."""
        ledd_df = self._make_ledd_df([
            {"PATNO": "3001", "LEDD_numeric": 700.0,
             "start_dt": datetime(2020, 1, 1), "stop_dt": datetime(2020, 6, 1)},
        ])
        visit_date = datetime(2021, 1, 1)
        result = p4.compute_visit_ledd("3001", visit_date, ledd_df)
        assert result == 0.0

    def test_on_start_date(self):
        """Visit exactly on start date — medication IS active."""
        ledd_df = self._make_ledd_df([
            {"PATNO": "3001", "LEDD_numeric": 500.0,
             "start_dt": datetime(2021, 1, 1), "stop_dt": None},
        ])
        visit_date = datetime(2021, 1, 1)
        result = p4.compute_visit_ledd("3001", visit_date, ledd_df)
        assert result == 500.0

    def test_on_stop_date(self):
        """Visit on stop date — medication is NOT active (stop_dt > visit)."""
        ledd_df = self._make_ledd_df([
            {"PATNO": "3001", "LEDD_numeric": 500.0,
             "start_dt": datetime(2020, 1, 1), "stop_dt": datetime(2021, 1, 1)},
        ])
        # stop_dt = visit_date → NOT active (we use strict: stop_dt > visit_date)
        visit_date = datetime(2021, 1, 1)
        result = p4.compute_visit_ledd("3001", visit_date, ledd_df)
        assert result == 0.0

    def test_no_meds_for_patient(self):
        """Patient has no medication records at all."""
        ledd_df = self._make_ledd_df([
            {"PATNO": "9999", "LEDD_numeric": 700.0,
             "start_dt": datetime(2020, 1, 1), "stop_dt": None},
        ])
        visit_date = datetime(2021, 1, 1)
        result = p4.compute_visit_ledd("3001", visit_date, ledd_df)
        assert result == 0.0

    def test_mixed_active_inactive(self):
        """Two meds, one active and one stopped — should only sum the active one."""
        ledd_df = self._make_ledd_df([
            {"PATNO": "3001", "LEDD_numeric": 700.0,
             "start_dt": datetime(2020, 1, 1), "stop_dt": None},
            {"PATNO": "3001", "LEDD_numeric": 200.0,
             "start_dt": datetime(2019, 1, 1), "stop_dt": datetime(2019, 12, 1)},
        ])
        visit_date = datetime(2021, 1, 1)
        result = p4.compute_visit_ledd("3001", visit_date, ledd_df)
        assert result == 700.0


# =====================================================================
# filter_off_state
# =====================================================================

class TestFilterOffState:
    """Tests for OFF-state filtering of UPDRS-III."""

    def test_off_state_only(self):
        """Should keep only PDSTATE=='OFF' rows."""
        df = pd.DataFrame({
            "PATNO": [1, 1, 1, 1],
            "PDSTATE": ["OFF", "ON", "OFF", np.nan],
            "PDMEDYN": [0, 1, 0, 0],
            "NP3TOT": [10, 20, 15, 5],
        })
        result = p4.filter_off_state(df)
        assert len(result) == 2
        assert list(result["NP3TOT"]) == [10, 15]

    def test_fallback_pdmedyn(self):
        """If no OFF rows, fallback to PDMEDYN==0."""
        df = pd.DataFrame({
            "PATNO": [1, 1, 1],
            "PDSTATE": [np.nan, np.nan, np.nan],
            "PDMEDYN": [0, 1, 0],
            "NP3TOT": [10, 20, 15],
        })
        result = p4.filter_off_state(df)
        assert len(result) == 2
        assert list(result["NP3TOT"]) == [10, 15]

    def test_no_off_no_pdmedyn0(self):
        """If no OFF rows and no PDMEDYN==0 rows, return all (unmedicated patients)."""
        df = pd.DataFrame({
            "PATNO": [1, 1],
            "PDSTATE": [np.nan, np.nan],
            "PDMEDYN": [np.nan, np.nan],
            "NP3TOT": [10, 15],
        })
        result = p4.filter_off_state(df)
        # Should return all rows as a fallback for unmedicated patients
        assert len(result) == 2
