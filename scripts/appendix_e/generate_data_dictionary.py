"""Generate Appendix E.1 data dictionary from live giman_research PostgreSQL.

Reads information_schema for all 10 dissertation schemas, counts rows per table,
samples top-10 non-null column stats, emits Markdown.

Usage:
    .venv/bin/python scripts/appendix_e/generate_data_dictionary.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import text

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from giman_pipeline.data.db import get_engine  # noqa: E402

OUTPUT = Path("outputs/dissertation/appendix_e/E1a_postgres_data_dictionary.md")

SCHEMAS = [
    ("ppmi_raw", "PPMI clinical/imaging raw tables (AMP-PD v4 BigQuery + LONI IDA)"),
    ("biofind_raw", "BioFIND external validation cohort (Russo 2025 replication source)"),
    ("pdbp_raw", "PDBP external prediction cohort (includes April 2026 LONI expansion)"),
    ("hbs_raw", "HBS external prediction cohort (low-data deployment test)"),
    ("staging", "NSD-ISS staging results per cohort"),
    ("features", "Assembled ML feature matrices for Papers 1-3 + cross-cohort"),
    ("longitudinal", "Paper 3 longitudinal NSD-ISS staging + transition events"),
    ("paper3", "Paper 3 per-visit longitudinal feature vectors"),
    ("ledd", "Levodopa-equivalent daily dose + concomitant PD medication"),
    ("mechanistic", "Phase 1-4 mechanistic twin outputs (posteriors, LOO, counterfactuals)"),
]


def list_tables(engine, schema: str) -> pd.DataFrame:
    return pd.read_sql(
        text(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = :s AND table_type = 'BASE TABLE' "
            "ORDER BY table_name"
        ),
        engine,
        params={"s": schema},
    )


def describe_columns(engine, schema: str, table: str) -> pd.DataFrame:
    return pd.read_sql(
        text(
            """
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_schema = :s AND table_name = :t
            ORDER BY ordinal_position
            """
        ),
        engine,
        params={"s": schema, "t": table},
    )


def row_count(engine, schema: str, table: str) -> int:
    return pd.read_sql(
        text(f'SELECT count(*) AS n FROM "{schema}"."{table}"'),
        engine,
    )["n"].iloc[0]


def render_table(engine, schema: str, table: str) -> str:
    cols = describe_columns(engine, schema, table)
    n_rows = row_count(engine, schema, table)
    lines = [f"### `{schema}.{table}` — {n_rows:,} rows, {len(cols)} columns\n"]
    lines.append("| Column | Type | Nullable |")
    lines.append("|---|---|---|")
    for _, r in cols.iterrows():
        lines.append(f"| `{r.column_name}` | {r.data_type} | {r.is_nullable} |")
    return "\n".join(lines) + "\n\n"


def main() -> None:
    engine = get_engine()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    total_tables = 0
    total_rows = 0

    with OUTPUT.open("w") as fh:
        fh.write("# Appendix E.1a — PostgreSQL Data Dictionary\n\n")
        fh.write(
            "Auto-generated from the live `giman_research` PostgreSQL 17 database "
            "(connection: `postgresql+psycopg2://blair.dupre@localhost:5432/giman_research`). "
            "Regenerate via `scripts/appendix_e/generate_data_dictionary.py`.\n\n"
        )
        fh.write("## Schema overview\n\n")
        fh.write("| Schema | Tables | Description |\n")
        fh.write("|---|---|---|\n")

        schema_tables: dict[str, pd.DataFrame] = {}
        for schema, description in SCHEMAS:
            tables = list_tables(engine, schema)
            schema_tables[schema] = tables
            fh.write(f"| `{schema}` | {len(tables)} | {description} |\n")

        fh.write("\n---\n\n")

        for schema, description in SCHEMAS:
            tables = schema_tables[schema]
            fh.write(f"## Schema `{schema}` ({len(tables)} tables)\n\n")
            fh.write(f"_{description}_\n\n")
            for table_name in tables["table_name"]:
                fh.write(render_table(engine, schema, table_name))
                total_tables += 1
                total_rows += row_count(engine, schema, table_name)

        fh.write("---\n\n")
        fh.write(f"**Totals:** {total_tables} tables, {total_rows:,} rows.\n")

    print(f"Wrote {OUTPUT}")
    print(f"  {total_tables} tables, {total_rows:,} total rows across {len(SCHEMAS)} schemas")


if __name__ == "__main__":
    main()
