"""Build features.paper1_site_assignments from PPMI LONI IDA root XML metadata.

Idempotent. Extracts one (patno, site_key, earliest_scan_date, scanner_mfg,
scanner_model, field_strength) row per patient from the 2,445 root-level
PPMI_*.xml files at data/PPMI_metadata/. When a patient has multiple T1
acquisitions across different sites, the earliest scan's site wins (ties
broken lexicographically on scan ID). Loads to Postgres for Analysis E.

Usage:
    .venv/bin/python scripts/paper1/create_sql_paper1_site_assignments.py
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd
from sqlalchemy import text

from giman_pipeline.data.db import get_engine

XML_ROOT = Path("data/PPMI_metadata")


def parse_one(xml_path: Path) -> dict | None:
    try:
        tree = ET.parse(xml_path)
    except ET.ParseError:
        return None
    proj = tree.getroot().find("project")
    if proj is None:
        return None
    subj = proj.find("subject")
    if subj is None:
        return None
    patno = subj.findtext("subjectIdentifier")
    site = proj.findtext("siteKey")
    if not (patno and site):
        return None
    study = subj.find("study")
    date = None
    if study is not None:
        series = study.find("series")
        if series is not None:
            date = series.findtext("dateAcquired")
    mfg = model = field = None
    for pt in proj.iter("protocol"):
        term = pt.attrib.get("term")
        if term == "Manufacturer":
            mfg = pt.text
        elif term == "Mfg Model":
            model = pt.text
        elif term == "Field Strength":
            field = pt.text
    return {
        "patno": int(patno),
        "site_key": site,
        "scan_date": date,
        "scanner_mfg": mfg,
        "scanner_model": model,
        "field_strength": float(field) if field else None,
        "scan_id": xml_path.name,
    }


def main() -> None:
    files = sorted(XML_ROOT.glob("PPMI_*.xml"))
    print(f"Parsing {len(files)} root XMLs")
    rows = [r for p in files if (r := parse_one(p))]
    print(f"Parsed {len(rows)} rows from XMLs")

    df = pd.DataFrame(rows)
    df["scan_date"] = pd.to_datetime(df["scan_date"], errors="coerce")
    df = df.sort_values(["patno", "scan_date", "scan_id"])
    per_patient = df.drop_duplicates(subset="patno", keep="first").reset_index(drop=True)
    per_patient = per_patient.rename(columns={"scan_date": "earliest_scan_date"})
    per_patient = per_patient[
        [
            "patno",
            "site_key",
            "earliest_scan_date",
            "scanner_mfg",
            "scanner_model",
            "field_strength",
        ]
    ]
    print(f"Unique patients with site info: {len(per_patient)}")
    print(f"Unique sites: {per_patient['site_key'].nunique()}")

    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS features.paper1_site_assignments"))
    per_patient.to_sql(
        "paper1_site_assignments",
        engine,
        schema="features",
        if_exists="replace",
        index=False,
    )
    with engine.begin() as conn:
        conn.execute(
            text(
                "ALTER TABLE features.paper1_site_assignments "
                "ADD PRIMARY KEY (patno)"
            )
        )
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS paper1_site_assignments_site_idx "
                "ON features.paper1_site_assignments (site_key)"
            )
        )
        n = conn.execute(
            text("SELECT COUNT(*) FROM features.paper1_site_assignments")
        ).scalar_one()
    print(f"Wrote features.paper1_site_assignments ({n} rows)")


if __name__ == "__main__":
    main()
