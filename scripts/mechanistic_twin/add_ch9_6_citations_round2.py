#!/usr/bin/env python3
"""Add Round-2 §9.6 citations to audit.citation (post-Task-8.5 audit additions).

These 13 citations are referenced in the §9.6 prose and supporting tables
(matched-cohort ablation narrative, CRPS framing, Comparison to Prior Work,
and reviewer-preempt limitations section). They supplement the Round-1 adds
(Liu 2023, Bäckström 2020, Sampedro 2020, Bartl 2021) from Task 1.

All DOIs verified via the Task 8.5 literature review (agents with PubMed MCP
+ OpenAlex + web fetch). zotero_verified=0 initially; manual Zotero RT8B9N2J
sync will update this to 1 after BibTeX re-export.
"""
from __future__ import annotations

import pandas as pd
from sqlalchemy import create_engine, text

ENGINE = create_engine("postgresql+psycopg2://blair.dupre@localhost:5432/giman_research")

ROWS = [
    # Prior-work comparators (Task 8.5c)
    {"cite_key": "gupta2025cpt", "author": "Gupta, S.",
     "year": 2025, "title": "Biomarker-directed clinical endpoint model for early Parkinson's disease",
     "journal": "Clinical Pharmacology & Therapeutics",
     "doi": "10.1002/cpt.3593", "pmid": "40077911"},
    {"cite_key": "koval2021admap", "author": "Koval, I.",
     "year": 2021, "title": "AD Course Map charts Alzheimer's disease progression",
     "journal": "Scientific Reports",
     "doi": "10.1038/s41598-021-87434-1", "pmid": ""},
    {"cite_key": "veronneauveilleux2021", "author": "Véronneau-Veilleux, F.",
     "year": 2021, "title": "An integrative dynamical model of dopamine function in Parkinson's disease",
     "journal": "Journal of Pharmacokinetics and Pharmacodynamics",
     "doi": "10.1007/s10928-020-09723-y", "pmid": ""},
    {"cite_key": "iljina2016pnas", "author": "Iljina, M.",
     "year": 2016, "title": "Kinetic model of the aggregation of alpha-synuclein provides insights into prion-like spreading",
     "journal": "Proceedings of the National Academy of Sciences",
     "doi": "10.1073/pnas.1524128113", "pmid": "26884195"},
    {"cite_key": "hahnel2024npjpd", "author": "Hähnel, T.",
     "year": 2024, "title": "Progression subtypes in Parkinson's disease identified via joint modelling",
     "journal": "npj Parkinson's Disease",
     "doi": "10.1038/s41531-024-00712-3", "pmid": ""},
    {"cite_key": "chen2024jneurol", "author": "Chen, Y.",
     "year": 2024, "title": "Milestone-based identification of progression subtypes in Parkinson's disease",
     "journal": "Journal of Neurology",
     "doi": "10.1007/s00415-024-12645-1", "pmid": ""},

    # Calibration / identifiability methodology (Task 8.5b CRPS + Task 2/7 framing)
    {"cite_key": "gneiting2007jasa", "author": "Gneiting, T.",
     "year": 2007, "title": "Strictly proper scoring rules, prediction, and estimation",
     "journal": "Journal of the American Statistical Association",
     "doi": "10.1198/016214506000001437", "pmid": ""},
    {"cite_key": "savickarlsson2009aaps", "author": "Savic, R. M.",
     "year": 2009, "title": "Importance of shrinkage in empirical Bayes estimates for diagnostics",
     "journal": "AAPS Journal",
     "doi": "10.1208/s12248-009-9133-0", "pmid": ""},

    # DaT-SPECT test-retest anchors (Task 7 LOO framing)
    {"cite_key": "seibyl1997jnm", "author": "Seibyl, J. P.",
     "year": 1997, "title": "Test/retest reproducibility of [I-123]beta-CIT SPECT dopamine transporter imaging in Parkinson's disease",
     "journal": "Journal of Nuclear Medicine",
     "doi": "", "pmid": "9293807"},
    {"cite_key": "buchert2020ejnmmi", "author": "Buchert, R.",
     "year": 2020, "title": "Reproducibility of DAT-SPECT in PPMI",
     "journal": "EJNMMI Physics",
     "doi": "10.1186/s40658-020-00304-z", "pmid": ""},
    {"cite_key": "kerstens2020ejnmmi", "author": "Kerstens, V. S.",
     "year": 2020, "title": "Reliability of dopamine transporter PET with [18F]FE-PE2I in Parkinson's disease",
     "journal": "EJNMMI Research",
     "doi": "10.1186/s13550-020-00629-x", "pmid": ""},

    # NSD-ISS framing (Task 9 chapter-position anchor)
    {"cite_key": "feuerstein2026prd", "author": "Feuerstein, J. S.",
     "year": 2026, "title": "Putamen SBR tracks NSD-ISS phenoconversion in isolated RBD",
     "journal": "Parkinsonism & Related Disorders",
     "doi": "10.1016/j.parkreldis.2026.108266", "pmid": "41780487"},

    # Liu 2023 DOI backfill (Task 1 entry had empty DOI)
    {"cite_key": "liu2023gfap_v2", "author": "Liu, Y.",
     "year": 2023, "title": "CSF GFAP predicts longitudinal CSF alpha-synuclein and cognitive decline in de novo Parkinson's disease",
     "journal": "Journal of Neuroinflammation",
     "doi": "10.1186/s12974-023-02812-y", "pmid": "37475029"},
]


def main() -> None:
    df = pd.DataFrame(ROWS)
    df["zotero_key"] = ""
    df["zotero_verified"] = 0
    df["ezproxy_verified"] = 0
    df["last_checked"] = "2026-04-16"

    with ENGINE.begin() as conn:
        for _, r in df.iterrows():
            conn.execute(
                text(
                    "DELETE FROM audit.citation WHERE cite_key = :cite_key"
                ),
                {"cite_key": r["cite_key"]},
            )
            conn.execute(
                text(
                    "INSERT INTO audit.citation "
                    "(cite_key, author, year, title, journal, doi, pmid, "
                    "zotero_key, zotero_verified, ezproxy_verified, last_checked) "
                    "VALUES (:cite_key, :author, :year, :title, :journal, :doi, "
                    ":pmid, :zotero_key, :zotero_verified, :ezproxy_verified, "
                    ":last_checked)"
                ),
                r.to_dict(),
            )
    print(f"Inserted/updated {len(df)} round-2 citations in audit.citation")

    with ENGINE.connect() as conn:
        n = conn.execute(
            text("SELECT COUNT(*) FROM audit.citation WHERE cite_key = ANY(:keys)"),
            {"keys": list(df["cite_key"])},
        ).scalar()
        assert n == len(df), f"expected {len(df)}, got {n}"
    print(f"All {len(df)} citations confirmed present.")


if __name__ == "__main__":
    main()
