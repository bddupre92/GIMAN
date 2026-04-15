"""Parse outputs/mechanistic_twin/phase2/DATA_LITERATURE_REGISTRY.md pipe tables
into three structured tables in mechanistic.* schema.

Sections parsed:
- Parameter tables (columns: Parameter, Value, Unit, ...) → mechanistic.data_registry
- Literature anchor tables (columns include Literature Anchor / Reference /
  Supporting paper / Assumption) → mechanistic.literature_anchors
- Empirical findings tables (columns include Finding / Claim / Observation) → mechanistic.empirical_findings
"""
from __future__ import annotations
import re
from pathlib import Path
import pandas as pd
from sqlalchemy import create_engine, text

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "outputs/mechanistic_twin/phase2/DATA_LITERATURE_REGISTRY.md"
ENGINE = create_engine("postgresql+psycopg2://blair.dupre@localhost:5432/giman_research")


def parse_markdown_tables(md_text: str) -> list[pd.DataFrame]:
    """Return each pipe-table in the file as a DataFrame, preserving order."""
    tables: list[pd.DataFrame] = []
    lines = md_text.splitlines()
    i = 0
    while i < len(lines):
        if lines[i].strip().startswith("|") and "|" in lines[i]:
            header = [c.strip() for c in lines[i].strip().strip("|").split("|")]
            if i + 1 < len(lines) and re.match(r"^\s*\|[\s|:-]+\|", lines[i + 1]):
                rows = []
                j = i + 2
                while j < len(lines) and lines[j].strip().startswith("|"):
                    cells = [c.strip() for c in lines[j].strip().strip("|").split("|")]
                    if len(cells) == len(header):
                        rows.append(cells)
                    j += 1
                if rows:
                    tables.append(pd.DataFrame(rows, columns=header))
                i = j
                continue
        i += 1
    return tables


def classify_and_normalize(tables: list[pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Bucket tables by their header signature into 3 output DataFrames."""
    registry_rows: list[dict] = []
    anchor_rows: list[dict] = []
    finding_rows: list[dict] = []

    for t in tables:
        cols_lower = {c.lower() for c in t.columns}
        cols_map = {c.lower(): c for c in t.columns}  # lower → original

        # ── data_registry: must have "parameter" + "value" + some unit-like column ──
        if "parameter" in cols_lower and "value" in cols_lower and (
            "unit" in cols_lower or "units" in cols_lower
        ):
            unit_col = cols_map.get("unit") or cols_map.get("units")
            for _, r in t.iterrows():
                registry_rows.append({
                    "parameter": r.get(cols_map.get("parameter", "Parameter")),
                    "value": r.get(cols_map.get("value", "Value")),
                    "unit": r.get(unit_col) if unit_col else None,
                    "description": next(
                        (r[cols_map[k]] for k in cols_lower
                         if k in {"description", "biological meaning"}), None
                    ),
                    "source": next(
                        (r[cols_map[k]] for k in cols_lower
                         if k in {"source", "literature source", "literature anchor",
                                  "anchoring evidence"}), None
                    ),
                    "rationale": next(
                        (r[cols_map[k]] for k in cols_lower
                         if k == "rationale"), None
                    ),
                    "file_refs": next(
                        (r[cols_map[k]] for k in cols_lower
                         if k in {"file refs", "file references", "codebase location"}), None
                    ),
                    "status": next(
                        (r[cols_map[k]] for k in cols_lower
                         if k == "status"), None
                    ),
                })

        # ── empirical_findings: must have "finding" or "claim" or "observation" ──
        elif "finding" in cols_lower or "claim" in cols_lower or "observation" in cols_lower:
            finding_key = (
                cols_map.get("finding") or cols_map.get("claim") or cols_map.get("observation")
            )
            for _, r in t.iterrows():
                finding_rows.append({
                    "finding": r.get(finding_key),
                    "metric_value": next(
                        (r[cols_map[k]] for k in cols_lower
                         if k in {"value", "metric"}), None
                    ),
                    "context": next(
                        (r[cols_map[k]] for k in cols_lower
                         if k in {"context", "validation method", "key finding"}), None
                    ),
                    "interpretation": next(
                        (r[cols_map[k]] for k in cols_lower
                         if k in {"interpretation", "implication", "how it validates our assumption",
                                  "how it validates"}), None
                    ),
                    "source_run": next(
                        (r[cols_map[k]] for k in cols_lower
                         if k in {"source", "source run", "step/block",
                                  "supporting literature"}), None
                    ),
                })

        # ── literature_anchors: "literature anchor", "reference", "supporting paper",
        #    "assumption", or "cite key" columns ──
        elif (
            "literature anchor" in cols_lower
            or "reference" in cols_lower
            or "supporting paper" in cols_lower
            or "assumption" in cols_lower
            or "cite key" in cols_lower
        ):
            # Determine which column holds the primary anchor label
            anchor_key = (
                cols_map.get("literature anchor")
                or cols_map.get("reference")
                or cols_map.get("supporting paper")
                or cols_map.get("assumption")
                or cols_map.get("cite key")
            )
            for _, r in t.iterrows():
                anchor_rows.append({
                    "anchor": r.get(anchor_key),
                    "year": next(
                        (r[cols_map[k]] for k in cols_lower if k == "year"), None
                    ),
                    "reason": next(
                        (r[cols_map[k]] for k in cols_lower
                         if k in {"reason", "finding", "key finding",
                                  "role", "how it validates our assumption"}), None
                    ),
                    "source": next(
                        (r[cols_map[k]] for k in cols_lower
                         if k in {"source", "authors", "supporting paper"}), None
                    ),
                    "decision_trail": next(
                        (r[cols_map[k]] for k in cols_lower
                         if k in {"decision trail", "decision", "status",
                                  "why chosen", "journal"}), None
                    ),
                })

    def _num(v):
        if not isinstance(v, str):
            return v
        # Normalize unicode minus to ASCII minus before parsing
        v_norm = v.replace("\u2212", "-").replace("−", "-")
        m = re.search(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", v_norm)
        return float(m.group(0)) if m else None

    registry = pd.DataFrame(registry_rows)
    if not registry.empty and "value" in registry.columns:
        registry["value_num"] = registry["value"].apply(_num)

    findings = pd.DataFrame(finding_rows)
    if not findings.empty and "metric_value" in findings.columns:
        findings["metric_value"] = findings["metric_value"].apply(_num)

    anchors = pd.DataFrame(anchor_rows)

    return {
        "data_registry": registry,
        "literature_anchors": anchors,
        "empirical_findings": findings,
    }


def main() -> None:
    md = REGISTRY.read_text(encoding="utf-8")
    tables = parse_markdown_tables(md)
    print(f"Parsed {len(tables)} markdown tables from {REGISTRY.name}")

    dfs = classify_and_normalize(tables)

    with ENGINE.begin() as conn:
        conn.execute(text("CREATE SCHEMA IF NOT EXISTS mechanistic"))

    for name, df in dfs.items():
        if df.empty:
            print(f"  {name}: 0 rows — skipping")
            continue
        df.to_sql(name, ENGINE, schema="mechanistic", if_exists="replace",
                  index=False, method="multi", chunksize=200)
        print(f"  mechanistic.{name}: {len(df):,} rows × {len(df.columns)} cols")


if __name__ == "__main__":
    main()
