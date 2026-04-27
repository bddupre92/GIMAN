"""Paper 1 R2 / W5 — Full-cohort site recovery for PPMI (target: 2,201 patients).

This is a PLACEHOLDER LOADER that will populate `features.paper1_site_assignments_full`
once additional LONI IDA XML metadata is downloaded. As of 2026-04-23 session, ONLY
T1-MRI XMLs are available locally (`data/PPMI_metadata/`, 2,445 files, 657 patients,
of whom 647 are in the Paper-1 cohort). The remaining 1,554 patients require
DaT-SPECT / SRRL / DTI / SWI / FLAIR XMLs that are not in the local archive.

STATUS: FAILED for full-cohort recovery (no additional XMLs accessible from local
filesystem). This script documents:

  (a) Exactly what the gap is (1,554 / 2,201 patients missing site IDs),
  (b) Why PPMI's harmonized public CSVs don't fill it (CNO column is suppressed
      for DUA compliance — the data dictionary lists CNO as "Site Number" in 30+
      page definitions but no release CSV exposes it),
  (c) How a future download would plug in to populate features.paper1_site_assignments_full.

Runs as-is to populate the table with whatever XMLs are available. Re-invoke
after downloading more XMLs to incrementally widen coverage.

Usage:
    .venv/bin/python scripts/load_paper1_site_assignments_full.py

Author: Blair Dupre (UND BME)
"""

from __future__ import annotations

import logging
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd
from sqlalchemy import text

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from giman_pipeline.data.db import get_engine  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("w5_site_full")


# Directories we scan for PPMI LONI IDA root XML metadata. Add dirs here as
# more LONI IDA XML downloads land. All paths must be folders of PPMI_*.xml
# files at the top level (NOT nested inside subject / modality subdirectories).
XML_DIRS: list[Path] = [
    ROOT / "data" / "PPMI_metadata",                      # T1-MRI, 2,445 files, 657 pts
    ROOT / "data" / "PPMI_metadata_datspect",             # Future: DaT-SPECT XMLs
    ROOT / "data" / "PPMI_metadata_srrl",                 # Future: SRRL XMLs
    ROOT / "data" / "PPMI_metadata_dti",                  # Future: DTI XMLs
    ROOT / "data" / "PPMI_metadata_swi_flair",            # Future: SWI/FLAIR XMLs
]


def parse_one(xml_path: Path) -> dict | None:
    try:
        tree = ET.parse(xml_path)
    except ET.ParseError:
        return None
    root = tree.getroot()
    proj = root.find("project")
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
    modality = None
    if study is not None:
        series = study.find("series")
        if series is not None:
            date = series.findtext("dateAcquired")
            modality = series.findtext("modality")
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
        "modality": modality,
        "scanner_mfg": mfg,
        "scanner_model": model,
        "field_strength": float(field) if field else None,
        "scan_id": xml_path.name,
        "xml_source_dir": str(xml_path.parent.relative_to(ROOT)),
    }


def main() -> None:
    # Collect rows from every available XML dir
    all_rows: list[dict] = []
    for xd in XML_DIRS:
        if not xd.exists():
            log.info("  Skipping missing dir: %s", xd.relative_to(ROOT))
            continue
        files = sorted(xd.glob("PPMI_*.xml"))
        log.info("  Scanning %s — %d PPMI_*.xml files", xd.relative_to(ROOT), len(files))
        rows = [r for p in files if (r := parse_one(p))]
        log.info("    Parsed %d rows from this dir", len(rows))
        all_rows.extend(rows)

    if not all_rows:
        log.error("No XMLs parsed from any dir in XML_DIRS. Nothing to load.")
        return

    df = pd.DataFrame(all_rows)
    log.info("Total parsed rows: %d (across %d XML dirs)", len(df), df["xml_source_dir"].nunique())
    log.info("  Unique patients: %d", df["patno"].nunique())
    log.info("  Unique sites: %d", df["site_key"].nunique())
    log.info("  Modality breakdown: %s", df["modality"].value_counts().to_dict())

    # De-duplicate: earliest scan per patient wins (ties broken on scan_id)
    df["scan_date"] = pd.to_datetime(df["scan_date"], errors="coerce")
    df = df.sort_values(["patno", "scan_date", "scan_id"])
    per_patient = df.drop_duplicates(subset="patno", keep="first").reset_index(drop=True)
    per_patient = per_patient.rename(columns={"scan_date": "earliest_scan_date"})

    engine = get_engine()
    with engine.connect() as c:
        r = c.execute(
            text("SELECT patno FROM features.paper1_features_with_targets")
        ).fetchall()
    paper1_patnos = {int(row[0]) for row in r}
    log.info("Paper1 cohort size: %d patients", len(paper1_patnos))
    log.info(
        "Paper1 coverage gained: %d / %d (%.1f%%)",
        int(per_patient["patno"].isin(paper1_patnos).sum()),
        len(paper1_patnos),
        100.0 * int(per_patient["patno"].isin(paper1_patnos).sum()) / len(paper1_patnos),
    )

    # Filter to Paper-1 cohort (consistent with paper1_site_assignments)
    per_patient_p1 = per_patient[per_patient["patno"].isin(paper1_patnos)].reset_index(drop=True)
    log.info(
        "Writing features.paper1_site_assignments_full: %d rows (Paper-1 patients only)",
        len(per_patient_p1),
    )

    out = per_patient_p1[
        [
            "patno", "site_key", "earliest_scan_date", "scanner_mfg",
            "scanner_model", "field_strength", "modality", "xml_source_dir",
        ]
    ]

    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS features.paper1_site_assignments_full"))
    out.to_sql(
        "paper1_site_assignments_full",
        engine,
        schema="features",
        if_exists="replace",
        index=False,
    )
    with engine.begin() as conn:
        conn.execute(
            text(
                "ALTER TABLE features.paper1_site_assignments_full "
                "ADD PRIMARY KEY (patno)"
            )
        )
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS paper1_site_assignments_full_site_idx "
                "ON features.paper1_site_assignments_full (site_key)"
            )
        )
        n = conn.execute(
            text("SELECT COUNT(*) FROM features.paper1_site_assignments_full")
        ).scalar_one()
    log.info("Wrote features.paper1_site_assignments_full (%d rows)", n)
    log.info("  Delta vs features.paper1_site_assignments (657 rows): +%d rows", n - 657)


if __name__ == "__main__":
    main()
