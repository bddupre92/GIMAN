"""Create/refresh features.paper1_features_extended_33 Postgres table.

Paper 1's canonical 33-feature schema. Idempotent: drops and recreates
the table and its index on every run.

Schema (49 columns total):
  - 16 metadata/target columns from features.paper1_features_with_targets
  - 22 original features (same source table)
  - 11 extended features from features.paper2_gimin_cohort (baseline visit per PATNO):
    * 6 cortical thickness (Fischl 2012): entorhinal L/R, cingulate L/R, precentral L/R
    * 4 CSF biomarkers (Mollenhauer 2017): alpha-synuclein, total tau, abeta42, pTau181
    * 1 polygenic risk score (Nalls 2019): grs_total

Literature grounding per feature:
  - Simuni 2024 (NSD-ISS) — staging anchor features
  - Goetz 2008 (MDS-UPDRS) — motor scales
  - Marek 2018 (PPMI) — cohort battery
  - Fischl 2012 (FreeSurfer) — cortical thickness
  - Mollenhauer 2017 (PPMI CSF) — CSF biomarkers
  - Nalls 2019 (GRS) — polygenic risk score

Outputs:
  - Postgres table: features.paper1_features_extended_33
  - JSON reports: outputs/paper1_sql/null_rate_report.json
                 outputs/paper1_sql/metadata.json

Author: Blair Dupre
Date: 2026-04-22
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import text

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from giman_pipeline.data.db import get_engine, read_sql

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


OUTPUT_DIR = ROOT / "outputs" / "paper1_sql"

CREATE_SQL = """
DROP TABLE IF EXISTS features.paper1_features_extended_33;

CREATE TABLE features.paper1_features_extended_33 AS
WITH baseline AS (
  SELECT DISTINCT ON ("PATNO") *
  FROM features.paper2_gimin_cohort
  ORDER BY "PATNO", "AGE_AT_VISIT" ASC
)
SELECT
  p1.*,
  b."ENTORHINAL_L_CTH"   AS entorhinal_l_cth,
  b."ENTORHINAL_R_CTH"   AS entorhinal_r_cth,
  b."CINGULATE_L_CTH"    AS cingulate_l_cth,
  b."CINGULATE_R_CTH"    AS cingulate_r_cth,
  b."PRECENTRAL_L_CTH"   AS precentral_l_cth,
  b."PRECENTRAL_R_CTH"   AS precentral_r_cth,
  b."ALPHA_SYNUCLEIN"    AS csf_alpha_synuclein,
  b."TOTAL_TAU"          AS csf_total_tau,
  b."ABETA42"            AS csf_abeta42,
  b."PTAU181"            AS csf_ptau181,
  b."GENETIC_RISK_SCORE" AS grs_total
FROM features.paper1_features_with_targets p1
INNER JOIN baseline b ON p1.patno = b."PATNO";

CREATE INDEX idx_paper1_ext33_patno ON features.paper1_features_extended_33 (patno);
"""

FEATURE_METADATA = [
    # 22 original features
    ("sex", "Demographics", "Biological sex (0=F, 1=M)", "marek2018ppmi"),
    ("handed", "Demographics", "Handedness (R/L/A)", "marek2018ppmi"),
    ("age_at_baseline", "Demographics", "Age at enrollment (years)", "marek2018ppmi"),
    ("updrs1_total", "Motor (UPDRS)", "MDS-UPDRS Part I total", "goetz2008"),
    ("updrs2_total", "Motor (UPDRS)", "MDS-UPDRS Part II total", "goetz2008"),
    ("updrs3_tremor", "Motor (UPDRS)", "UPDRS-III tremor subscore", "goetz2008"),
    ("updrs3_rigidity", "Motor (UPDRS)", "UPDRS-III rigidity subscore", "goetz2008"),
    ("updrs3_bradykinesia", "Motor (UPDRS)", "UPDRS-III bradykinesia subscore", "goetz2008"),
    ("updrs3_axial", "Motor (UPDRS)", "UPDRS-III axial subscore", "goetz2008"),
    ("updrs4_total", "Motor (UPDRS)", "MDS-UPDRS Part IV total (high missingness)", "goetz2008"),
    ("moca_total", "Cognitive", "MoCA total (high missingness)", "marek2018ppmi"),
    ("rbd_total", "Sleep", "REM sleep behavior disorder total", "marek2018ppmi"),
    ("ess_total", "Sleep", "Epworth Sleepiness Scale total", "marek2018ppmi"),
    ("scopa_aut_total", "Autonomic", "SCOPA-AUT total score", "marek2018ppmi"),
    ("caudate_l_sbr", "DaT Imaging", "Left caudate specific binding ratio", "simuni2024"),
    ("caudate_r_sbr", "DaT Imaging", "Right caudate SBR", "simuni2024"),
    ("caudate_mean_sbr", "DaT Imaging", "Mean caudate SBR", "simuni2024"),
    ("caudate_asymmetry", "DaT Imaging", "Caudate L/R asymmetry", "simuni2024"),
    ("caudate_putamen_ratio", "DaT Imaging", "Caudate/putamen SBR ratio", "simuni2024"),
    ("lrrk2_carrier", "Genetics", "LRRK2 mutation carrier status", "marek2018ppmi"),
    ("gba_carrier", "Genetics", "GBA mutation carrier status", "marek2018ppmi"),
    ("apoe_e4_carrier", "Genetics", "APOE e4 allele carrier status", "marek2018ppmi"),
    # 11 extended features
    ("entorhinal_l_cth", "Cortical Thickness", "Left entorhinal cortical thickness (FreeSurfer)", "fischl2012"),
    ("entorhinal_r_cth", "Cortical Thickness", "Right entorhinal CTH", "fischl2012"),
    ("cingulate_l_cth", "Cortical Thickness", "Left cingulate CTH", "fischl2012"),
    ("cingulate_r_cth", "Cortical Thickness", "Right cingulate CTH", "fischl2012"),
    ("precentral_l_cth", "Cortical Thickness", "Left precentral CTH", "fischl2012"),
    ("precentral_r_cth", "Cortical Thickness", "Right precentral CTH", "fischl2012"),
    ("csf_alpha_synuclein", "CSF Biomarkers", "CSF alpha-synuclein", "mollenhauer2017"),
    ("csf_total_tau", "CSF Biomarkers", "CSF total tau", "mollenhauer2017"),
    ("csf_abeta42", "CSF Biomarkers", "CSF amyloid-beta 42", "mollenhauer2017"),
    ("csf_ptau181", "CSF Biomarkers", "CSF phospho-tau 181", "mollenhauer2017"),
    ("grs_total", "Polygenic Risk", "Parkinson's polygenic risk score (Nalls 2019)", "nalls2019"),
]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    engine = get_engine()

    # Execute CREATE TABLE
    logger.info("Creating features.paper1_features_extended_33 from existing features tables...")
    with engine.begin() as conn:
        for statement in CREATE_SQL.strip().split(";"):
            stmt = statement.strip()
            if stmt:
                conn.execute(text(stmt))

    # Verify row and column counts
    n_rows = read_sql("SELECT COUNT(*) AS n FROM features.paper1_features_extended_33").iloc[0]["n"]
    n_cols = read_sql(
        "SELECT COUNT(*) AS n FROM information_schema.columns "
        "WHERE table_schema='features' AND table_name='paper1_features_extended_33'"
    ).iloc[0]["n"]
    logger.info(f"Table created: {n_rows} rows × {n_cols} columns")
    assert n_rows == 2201, f"Expected 2,201 rows; got {n_rows}"
    assert n_cols == 49, f"Expected 49 columns; got {n_cols}"

    # Null-rate report
    logger.info("Computing null-rate per feature...")
    df = read_sql("SELECT * FROM features.paper1_features_extended_33")
    null_rates = {}
    for col, domain, desc, cite in FEATURE_METADATA:
        if col in df.columns:
            non_null_pct = round(100.0 * df[col].notna().sum() / len(df), 2)
            null_rates[col] = {
                "domain": domain,
                "description": desc,
                "literature_key": cite,
                "non_null_pct": non_null_pct,
            }

    null_report_path = OUTPUT_DIR / "null_rate_report.json"
    null_report_path.write_text(json.dumps({
        "created_at": "2026-04-22",
        "source_table": "features.paper1_features_extended_33",
        "n_rows": int(n_rows),
        "n_cols": int(n_cols),
        "feature_count": len(FEATURE_METADATA),
        "per_feature": null_rates,
    }, indent=2))
    logger.info(f"Null-rate report -> {null_report_path}")

    # Feature metadata JSON
    metadata_path = OUTPUT_DIR / "metadata.json"
    metadata_path.write_text(json.dumps({
        "table": "features.paper1_features_extended_33",
        "created_at": "2026-04-22",
        "assembled_from": [
            "features.paper1_features_with_targets (22 features + 16 metadata)",
            "features.paper2_gimin_cohort (baseline visit per PATNO, 11 extension features)",
        ],
        "join_type": "INNER JOIN on PATNO",
        "cohort_size": int(n_rows),
        "feature_count": len(FEATURE_METADATA),
        "literature_references": {
            "simuni2024": "Simuni et al. 2024 Lancet Neurol — NSD-ISS staging framework",
            "goetz2008": "Goetz et al. 2008 Mov Disord — MDS-UPDRS",
            "marek2018ppmi": "Marek et al. 2018 Ann Clin Transl Neurol — PPMI cohort",
            "fischl2012": "Fischl 2012 NeuroImage — FreeSurfer",
            "mollenhauer2017": "Mollenhauer et al. 2017 Neurology — PPMI CSF biomarkers",
            "nalls2019": "Nalls et al. 2019 Lancet Neurol — Parkinson's GRS",
        },
        "features": [
            {"name": col, "domain": domain, "description": desc, "literature_key": cite}
            for col, domain, desc, cite in FEATURE_METADATA
        ],
    }, indent=2))
    logger.info(f"Feature metadata -> {metadata_path}")

    # Print coverage summary
    print("\n" + "=" * 70)
    print("PAPER 1 EXTENDED 33-FEATURE SCHEMA — Coverage Summary")
    print("=" * 70)
    for domain in [
        "Demographics", "Motor (UPDRS)", "Cognitive", "Sleep", "Autonomic",
        "DaT Imaging", "Genetics", "Cortical Thickness", "CSF Biomarkers", "Polygenic Risk"
    ]:
        cols = [m[0] for m in FEATURE_METADATA if m[1] == domain]
        if cols:
            coverage = [null_rates[c]["non_null_pct"] for c in cols if c in null_rates]
            avg = sum(coverage) / len(coverage) if coverage else 0
            print(f"  {domain:25s} — {len(cols)} features, avg coverage {avg:.1f}%")
    print("=" * 70)


if __name__ == "__main__":
    main()
