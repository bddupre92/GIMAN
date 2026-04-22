#!/usr/bin/env python3
"""Load Paper 11 SciML full-cohort results into Postgres (``mechanistic`` schema).

Reads outputs of ``scripts/paper11_demo/hybrid_sciml_full_cohort.py`` which
writes one subdir per config under ``outputs/paper11_demo/full_cohort/<config_id>/``
with:
    - ``summary.json``          — run metadata + per-model metrics + bootstrap deltas
    - ``per_patient_results.csv`` — one row per (model, patno, split)

Creates two tables:
    - ``mechanistic.paper11_sciml_summary``  — one row per (config_id, model)
    - ``mechanistic.paper11_sciml_results``  — one row per (config_id, model, patno, split)

Idempotency: by default, re-running on the same config deletes the prior
rows for those ``config_id``s and inserts fresh ones (this is safe — PK is
``(config_id, model)`` on summary and ``(config_id, model, patno, split)`` on
results, so partial loads are clean).

``--force-reload`` TRUNCATEs the two new tables only (never touches other
``mechanistic.*`` tables).

``--dry-run`` prints the plan without hitting the database.

``--skip-registry-update`` skips the CLAUDE.md schema-table edit
(useful during development when you re-run the loader many times).

Usage:
    python scripts/load_paper11_to_pg.py                        # load everything
    python scripts/load_paper11_to_pg.py --glob 'baseline'      # one config
    python scripts/load_paper11_to_pg.py --glob 'grid_*'        # grid sweep only
    python scripts/load_paper11_to_pg.py --force-reload         # truncate + reload
    python scripts/load_paper11_to_pg.py --dry-run              # no writes
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sqlalchemy import text

from giman_pipeline.data.db import get_engine

PROJECT_ROOT = Path("/Users/blair.dupre/Projects/CSCI-FALL-2025")
COHORT_DIR = PROJECT_ROOT / "outputs/paper11_demo/full_cohort"
CLAUDE_MD = PROJECT_ROOT / "CLAUDE.md"

# Per-patient CSV uses "pure_mech" for the fair-mechanistic baseline;
# summary.json uses "pure_mech_fair". Normalize on load so the join key
# matches the summary-table model column.
PER_PATIENT_MODEL_RENAME = {"pure_mech": "pure_mech_fair"}

SUMMARY_MODELS = ("pure_mech_fair", "pure_mech_anchor_last", "pure_nn", "hybrid")

DDL_SUMMARY = """
CREATE TABLE IF NOT EXISTS mechanistic.paper11_sciml_summary (
    config_id                         TEXT             NOT NULL,
    run_timestamp                     TIMESTAMPTZ,
    git_sha                           TEXT,
    command_line                      TEXT,
    model                             TEXT             NOT NULL,
    lambda_physics                    DOUBLE PRECISION,
    lambda_monotone                   DOUBLE PRECISION,
    use_gru                           BOOLEAN,
    gru_state_aware                   BOOLEAN,
    gru_hidden                        INTEGER,
    gru_dropout                       DOUBLE PRECISION,
    solver_method                     TEXT,
    solver_atol                       DOUBLE PRECISION,
    solver_rtol                       DOUBLE PRECISION,
    solver_step_size                  DOUBLE PRECISION,
    fold_index                        INTEGER,
    n_folds                           INTEGER,
    seed                              INTEGER,
    epochs_max                        INTEGER,
    patience                          INTEGER,
    bootstrap_resamples               INTEGER,
    n_train                           INTEGER,
    n_val                             INTEGER,
    n_test                            INTEGER,
    n_features                        INTEGER,
    test_mae                          DOUBLE PRECISION,
    test_rmse                         DOUBLE PRECISION,
    test_median_abs_err               DOUBLE PRECISION,
    best_val_mae                      DOUBLE PRECISION,
    best_epoch                        INTEGER,
    epochs_trained                    INTEGER,
    learned_k_age_per_yr              DOUBLE PRECISION,
    delta_vs_puremech_fair_point      DOUBLE PRECISION,
    delta_vs_puremech_fair_ci_lo      DOUBLE PRECISION,
    delta_vs_puremech_fair_ci_hi      DOUBLE PRECISION,
    delta_vs_purenn_point             DOUBLE PRECISION,
    delta_vs_purenn_ci_lo             DOUBLE PRECISION,
    delta_vs_purenn_ci_hi             DOUBLE PRECISION,
    PRIMARY KEY (config_id, model)
);
"""

DDL_RESULTS = """
CREATE TABLE IF NOT EXISTS mechanistic.paper11_sciml_results (
    config_id          TEXT             NOT NULL,
    model              TEXT             NOT NULL,
    patno              INTEGER          NOT NULL,
    split              TEXT             NOT NULL,
    s_obs_last         DOUBLE PRECISION,
    s_pred_last        DOUBLE PRECISION,
    abs_err            DOUBLE PRECISION,
    t_horizon_yrs      DOUBLE PRECISION,
    horizon_bin        TEXT,
    t_baseline_yrs     DOUBLE PRECISION,
    n_observed_visits  INTEGER,
    PRIMARY KEY (config_id, model, patno, split)
);
"""

DDL_RESULTS_INDEX = """
CREATE INDEX IF NOT EXISTS idx_paper11_sciml_results_model_split
    ON mechanistic.paper11_sciml_results (model, split);
"""


def _nan_to_none(v):
    """Convert NaN / pandas NA to Python None for Postgres NULL."""
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    return v


def _parse_timestamp(ts: str | None) -> datetime | None:
    if not ts:
        return None
    # summary.json uses ISO 8601 with '+00:00' suffix. Python 3.11+ fromisoformat
    # handles this natively; 3.10 rejects the offset suffix and would silently
    # return None (storing NULL for run_timestamp). The main .venv is 3.12 but
    # .venv-leaspy is 3.10.14, so keep a compat fallback.
    try:
        return datetime.fromisoformat(ts)
    except (TypeError, ValueError):
        pass
    try:
        for suffix in ("+00:00", "Z"):
            if ts.endswith(suffix):
                return datetime.fromisoformat(ts[: -len(suffix)]).replace(
                    tzinfo=timezone.utc
                )
        # Last resort: strip trailing timezone-ish characters (ISO 8601 shape).
        return datetime.fromisoformat(ts[:19])
    except (TypeError, ValueError):
        return None


def _model_metrics_row(
    summary: dict,
    model: str,
    deltas: dict,
) -> dict | None:
    """Build one summary-table row for a given model.

    Returns None if the model was not trained (i.e. summary has no sub-dict).
    ``pure_mech_fair`` and ``pure_mech_anchor_last`` are always present since
    they are closed-form evaluations. ``pure_nn`` / ``hybrid`` only when
    ``--models`` included them.
    """
    model_data = summary.get(model)
    # Producer unconditionally emits {} for non-trained models; skip those too.
    if not isinstance(model_data, dict) or not model_data:
        return None

    row = {
        "config_id": summary["config_id"],
        "run_timestamp": _parse_timestamp(summary.get("timestamp_utc")),
        "git_sha": summary.get("git_sha"),
        "command_line": summary.get("command_line"),
        "model": model,
        "lambda_physics": summary.get("lambda_physics"),
        "lambda_monotone": summary.get("lambda_monotone"),
        "use_gru": summary.get("use_gru"),
        "gru_state_aware": summary.get("gru_state_aware"),
        "gru_hidden": summary.get("gru_hidden"),
        "gru_dropout": summary.get("gru_dropout"),
        "solver_method": summary.get("solver_method"),
        "solver_atol": summary.get("solver_atol"),
        "solver_rtol": summary.get("solver_rtol"),
        "solver_step_size": summary.get("solver_step_size"),
        "fold_index": summary.get("fold_index"),
        "n_folds": summary.get("n_folds"),
        "seed": summary.get("seed"),
        "epochs_max": summary.get("epochs_max"),
        "patience": summary.get("patience"),
        "bootstrap_resamples": summary.get("bootstrap_resamples"),
        "n_train": summary.get("n_train"),
        "n_val": summary.get("n_val"),
        "n_test": summary.get("n_test"),
        "n_features": summary.get("n_features"),
        "test_mae": _nan_to_none(model_data.get("test_mae")),
        "test_rmse": _nan_to_none(model_data.get("test_rmse")),
        "test_median_abs_err": _nan_to_none(model_data.get("test_median_abs_err")),
        "best_val_mae": _nan_to_none(model_data.get("best_val_mae")),
        "best_epoch": _nan_to_none(model_data.get("best_epoch")),
        "epochs_trained": _nan_to_none(model_data.get("epochs_trained")),
        "learned_k_age_per_yr": _nan_to_none(model_data.get("learned_k_age_per_yr")),
        "delta_vs_puremech_fair_point": None,
        "delta_vs_puremech_fair_ci_lo": None,
        "delta_vs_puremech_fair_ci_hi": None,
        "delta_vs_purenn_point": None,
        "delta_vs_purenn_ci_lo": None,
        "delta_vs_purenn_ci_hi": None,
    }

    # Deltas attach to the hybrid row only.
    if model == "hybrid":
        d_fair = (deltas or {}).get("hybrid_minus_puremech_fair") or {}
        row["delta_vs_puremech_fair_point"] = _nan_to_none(d_fair.get("point"))
        row["delta_vs_puremech_fair_ci_lo"] = _nan_to_none(d_fair.get("ci_lo"))
        row["delta_vs_puremech_fair_ci_hi"] = _nan_to_none(d_fair.get("ci_hi"))

        d_nn = (deltas or {}).get("hybrid_minus_purenn") or {}
        row["delta_vs_purenn_point"] = _nan_to_none(d_nn.get("point"))
        row["delta_vs_purenn_ci_lo"] = _nan_to_none(d_nn.get("ci_lo"))
        row["delta_vs_purenn_ci_hi"] = _nan_to_none(d_nn.get("ci_hi"))

    return row


def _build_summary_rows(summary: dict) -> list[dict]:
    deltas = summary.get("deltas") or {}
    rows = []
    for model in SUMMARY_MODELS:
        row = _model_metrics_row(summary, model, deltas)
        if row is not None:
            rows.append(row)
    return rows


def _build_results_df(per_patient_csv: Path, config_id: str) -> pd.DataFrame:
    df = pd.read_csv(per_patient_csv)
    expected_cols = {
        "patno", "split", "model",
        "s_obs_last", "s_pred_last", "abs_err",
        "t_horizon_yrs", "horizon_bin", "t_baseline_yrs", "n_observed_visits",
    }
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"per_patient_results.csv at {per_patient_csv} missing columns: {missing}"
        )

    # Normalize CSV "pure_mech" → summary's "pure_mech_fair" so the PKs join cleanly.
    df["model"] = df["model"].replace(PER_PATIENT_MODEL_RENAME)
    df.insert(0, "config_id", config_id)

    # Re-order to match DDL.
    df = df[[
        "config_id", "model", "patno", "split",
        "s_obs_last", "s_pred_last", "abs_err",
        "t_horizon_yrs", "horizon_bin", "t_baseline_yrs", "n_observed_visits",
    ]]
    # Coerce ints.
    df["patno"] = df["patno"].astype(int)
    df["n_observed_visits"] = df["n_observed_visits"].astype(int)
    return df


def _discover_configs(glob_pattern: str) -> list[tuple[str, Path, Path]]:
    """Return list of (config_id, summary_json_path, per_patient_csv_path).

    Skips subdirs missing either file, with a warning.
    """
    results: list[tuple[str, Path, Path]] = []
    warnings: list[str] = []
    for subdir in sorted(COHORT_DIR.glob(glob_pattern)):
        if not subdir.is_dir():
            continue
        config_id = subdir.name
        summary_json = subdir / "summary.json"
        per_patient = subdir / "per_patient_results.csv"
        if not summary_json.exists():
            warnings.append(f"  SKIP {config_id}: missing summary.json")
            continue
        if not per_patient.exists():
            warnings.append(f"  SKIP {config_id}: missing per_patient_results.csv")
            continue
        results.append((config_id, summary_json, per_patient))
    if warnings:
        print("Warnings during discovery:")
        for w in warnings:
            print(w)
    return results


def _ensure_schemas(engine, dry_run: bool) -> None:
    if dry_run:
        print("[dry-run] Would run DDL:")
        for ddl in (DDL_SUMMARY, DDL_RESULTS, DDL_RESULTS_INDEX):
            print(ddl.strip())
        return
    with engine.begin() as conn:
        conn.execute(text(DDL_SUMMARY))
        conn.execute(text(DDL_RESULTS))
        conn.execute(text(DDL_RESULTS_INDEX))


def _truncate_tables(engine, dry_run: bool) -> None:
    stmts = [
        "TRUNCATE TABLE mechanistic.paper11_sciml_summary",
        "TRUNCATE TABLE mechanistic.paper11_sciml_results",
    ]
    if dry_run:
        print("[dry-run] Would TRUNCATE:")
        for s in stmts:
            print(f"  {s};")
        return
    with engine.begin() as conn:
        for s in stmts:
            conn.execute(text(s))


def _preview_delete_config_rows(config_ids: list[str]) -> None:
    """Dry-run preview of the per-config DELETE."""
    if not config_ids:
        return
    print(f"[dry-run] Would DELETE rows for {len(config_ids)} config_ids from "
          "mechanistic.paper11_sciml_summary + paper11_sciml_results (one "
          "transaction per config, DELETE + INSERT summary + INSERT results "
          "atomic):")
    for cid in config_ids:
        print(f"  {cid}")


def _delete_config_rows_on_conn(conn, config_id: str) -> None:
    """DELETE rows for a single config_id using an open Connection.

    Caller owns the transaction (see ``_upsert_config_atomically``).
    """
    conn.execute(
        text("DELETE FROM mechanistic.paper11_sciml_summary WHERE config_id = :id"),
        {"id": config_id},
    )
    conn.execute(
        text("DELETE FROM mechanistic.paper11_sciml_results WHERE config_id = :id"),
        {"id": config_id},
    )


def _preview_insert_summary(rows: list[dict]) -> int:
    if not rows:
        return 0
    df = pd.DataFrame(rows)
    print(f"[dry-run] Would INSERT {len(df)} rows into "
          f"mechanistic.paper11_sciml_summary; columns={list(df.columns)}")
    # Show a compact preview.
    preview = df[["config_id", "model", "test_mae", "test_rmse",
                  "delta_vs_puremech_fair_point"]].copy()
    print(preview.to_string(index=False))
    return len(df)


def _preview_insert_results(df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    print(f"[dry-run] Would INSERT {len(df)} rows into "
          f"mechanistic.paper11_sciml_results; columns={list(df.columns)}")
    print("  split × model counts:")
    counts = df.groupby(["split", "model"]).size().unstack(fill_value=0)
    print(counts.to_string())
    return len(df)


def _upsert_config_atomically(
    engine,
    config_id: str,
    summary_rows: list[dict],
    results_df: pd.DataFrame,
    skip_delete: bool,
) -> tuple[int, int]:
    """Run DELETE + INSERT summary + INSERT results in ONE transaction.

    If any step raises, the whole transaction rolls back — prevents the
    silent data-gap bug where a DELETE committed but the INSERT failed.

    ``skip_delete=True`` is used after ``--force-reload`` TRUNCATE; the
    tables are already empty so the per-config DELETE is unnecessary. The
    INSERTs still run inside the same transaction for consistency.

    pandas ``DataFrame.to_sql`` accepts a live ``Connection`` via the ``con``
    kwarg and reuses the caller's transaction (does NOT start its own).
    """
    summary_df = pd.DataFrame(summary_rows) if summary_rows else None
    with engine.begin() as conn:
        if not skip_delete:
            _delete_config_rows_on_conn(conn, config_id)
        if summary_df is not None and not summary_df.empty:
            summary_df.to_sql(
                "paper11_sciml_summary",
                con=conn,
                schema="mechanistic",
                if_exists="append",
                index=False,
                method="multi",
                chunksize=500,
            )
        if not results_df.empty:
            results_df.to_sql(
                "paper11_sciml_results",
                con=conn,
                schema="mechanistic",
                if_exists="append",
                index=False,
                method="multi",
                chunksize=1000,
            )
    n_sum = 0 if summary_df is None else len(summary_df)
    n_res = len(results_df)
    return n_sum, n_res


def _update_claude_md(dry_run: bool) -> None:
    """Increment the mechanistic row table count 27→29 and verification
    one-liner 185→187 tables. Also update the verification date marker."""
    text_content = CLAUDE_MD.read_text()
    original = text_content

    # Row: | `mechanistic` | 27 | Phase 1–5 outputs ...
    # Update to: | `mechanistic` | 29 | Phase 1–5 outputs ..., Paper 11 SciML ...
    mech_row_old = (
        "| `mechanistic` | 27 | Phase 1–5 outputs (posteriors, LOO, counterfactuals, "
        "Phase 4 assembled data, Phase 5 Blocks 4/5, ch9.6 GFAP longitudinal, "
        "Paper 12 Phase 1 v6 smoke results `paper12_w4_smoke_results` + Q2 gate "
        "verdict `paper12_q2_gate_verdict`) |"
    )
    mech_row_new = (
        "| `mechanistic` | 29 | Phase 1–5 outputs (posteriors, LOO, counterfactuals, "
        "Phase 4 assembled data, Phase 5 Blocks 4/5, ch9.6 GFAP longitudinal, "
        "Paper 12 Phase 1 v6 smoke results `paper12_w4_smoke_results` + Q2 gate "
        "verdict `paper12_q2_gate_verdict`, Paper 11 SciML (summary + per-patient)) |"
    )
    if mech_row_old in text_content:
        text_content = text_content.replace(mech_row_old, mech_row_new)
    else:
        print("WARNING: mechanistic row not found verbatim in CLAUDE.md — skipping row edit.")

    # Size/tables/schemas line: "**Size: 718 MB · 185 tables across 14 schemas** (verified 2026-04-20)."
    size_line_re = re.compile(
        r"\*\*Size: 718 MB · 185 tables across 14 schemas\*\* \(verified 2026-04-20\)\."
    )
    size_line_new = (
        "**Size: 718 MB · 187 tables across 14 schemas** "
        "(verified 2026-04-20 (post-paper11-sciml-load))."
    )
    if size_line_re.search(text_content):
        text_content = size_line_re.sub(size_line_new, text_content)
    else:
        print("WARNING: size/tables line not found verbatim in CLAUDE.md — skipping size edit.")

    if text_content == original:
        print("CLAUDE.md unchanged — nothing to write.")
        return

    if dry_run:
        # Diff-style preview of changes.
        print("[dry-run] CLAUDE.md would be updated:")
        if mech_row_old in original and mech_row_new in text_content:
            print("  - mechanistic schema row: 27 → 29 tables; appended Paper 11 SciML")
        if size_line_re.search(original) and size_line_new in text_content:
            print("  - verification line: 185 → 187 tables, appended (post-paper11-sciml-load)")
        return

    CLAUDE_MD.write_text(text_content)
    print(f"Updated {CLAUDE_MD} (mechanistic row 27→29, 185→187 tables).")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--glob", default="*",
                        help="Glob pattern relative to outputs/paper11_demo/full_cohort/ "
                             "for config subdirs to load (default: all).")
    parser.add_argument("--force-reload", action="store_true",
                        help="TRUNCATE both paper11_* tables before load. Does NOT "
                             "touch other mechanistic.* tables.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be loaded without writing.")
    parser.add_argument("--skip-registry-update", action="store_true",
                        help="Skip the CLAUDE.md schema-table row update.")
    args = parser.parse_args()

    if not COHORT_DIR.exists():
        raise SystemExit(f"Cohort output dir not found: {COHORT_DIR}")

    configs = _discover_configs(args.glob)
    if not configs:
        raise SystemExit(f"No configs matched glob {args.glob!r} under {COHORT_DIR}")

    print(f"Discovered {len(configs)} config(s) under {COHORT_DIR}:")
    for cid, _, _ in configs:
        print(f"  {cid}")

    engine = get_engine()

    # DDL first.
    _ensure_schemas(engine, args.dry_run)
    if not args.dry_run:
        print("Tables: mechanistic.paper11_sciml_summary + paper11_sciml_results ensured.")

    # Force-reload path: TRUNCATE once, then the per-config loop can skip DELETE.
    # Default path: per-config DELETE is part of each atomic transaction.
    skip_delete_per_config = False
    if args.force_reload:
        print("--force-reload: TRUNCATing paper11_sciml_summary + paper11_sciml_results "
              "(other mechanistic.* tables untouched).")
        _truncate_tables(engine, args.dry_run)
        skip_delete_per_config = True
    else:
        if args.dry_run:
            _preview_delete_config_rows([cid for cid, _, _ in configs])

    # Load each config atomically (DELETE + INSERT summary + INSERT results
    # in ONE engine.begin() per config — prevents silent data gaps on failure).
    total_summary = 0
    total_results = 0
    per_config_counts = []
    for cid, summary_json, per_patient_csv in configs:
        with open(summary_json) as f:
            summary = json.load(f)
        # Ensure the JSON's config_id matches the directory name.
        if summary.get("config_id") != cid:
            print(f"  WARNING: {cid}: summary.json config_id="
                  f"{summary.get('config_id')!r} disagrees with dir name.")

        summary_rows = _build_summary_rows(summary)
        results_df = _build_results_df(per_patient_csv, cid)

        if args.dry_run:
            n_sum = _preview_insert_summary(summary_rows)
            n_res = _preview_insert_results(results_df)
        else:
            n_sum, n_res = _upsert_config_atomically(
                engine,
                cid,
                summary_rows,
                results_df,
                skip_delete=skip_delete_per_config,
            )
        total_summary += n_sum
        total_results += n_res
        per_config_counts.append((cid, n_sum, n_res))
        print(f"  {cid}: {n_sum} summary rows, {n_res} per-patient rows")

    # Registry update.
    if not args.skip_registry_update:
        _update_claude_md(args.dry_run)
    else:
        print("--skip-registry-update: CLAUDE.md not touched.")

    # Final summary.
    print()
    print("=" * 60)
    print(f"Configs processed: {len(configs)}")
    print(f"paper11_sciml_summary rows inserted: {total_summary}")
    print(f"paper11_sciml_results rows inserted: {total_results}")
    if args.dry_run:
        print("(dry-run — nothing was actually written)")
    print("Done.")


if __name__ == "__main__":
    main()
