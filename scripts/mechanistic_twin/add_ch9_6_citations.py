"""Add the four new §9.6 citations to audit.citation.

Sources identified 2026-04-15:
- Liu 2023 J Neuroinflammation — CSF GFAP predicts longitudinal cognition + CSF α-syn
- Bäckström 2020 Neurology — NfL predicts PD survival + UPDRS-III rate
- Sampedro 2020 Parkinsonism Relat Disord — Serum NfL reflects cortical, NOT striatal DAT
- Bartl 2021 PLoS ONE — NfL/sTREM2/YKL40/GFAP panel in PPMI PD
"""
from __future__ import annotations
import pandas as pd
from sqlalchemy import create_engine, text

ENGINE = create_engine("postgresql+psycopg2://blair.dupre@localhost:5432/giman_research")

ROWS = [
    {"cite_key": "liu2023gfap", "author": "Liu, Y.", "year": 2023,
     "title": "CSF GFAP predicts cognitive decline and longitudinal alpha-synuclein in de novo Parkinson's disease",
     "journal": "Journal of Neuroinflammation", "doi": "", "pmid": "",
     "zotero_key": "", "zotero_verified": 0, "ezproxy_verified": 0,
     "last_checked": "2026-04-15"},
    {"cite_key": "backstrom2020nfl", "author": "Bäckström, D.", "year": 2020,
     "title": "NfL as a biomarker for neurodegeneration and survival in Parkinson disease",
     "journal": "Neurology", "doi": "", "pmid": "",
     "zotero_key": "", "zotero_verified": 0, "ezproxy_verified": 0,
     "last_checked": "2026-04-15"},
    {"cite_key": "sampedro2020nfl", "author": "Sampedro, F.", "year": 2020,
     "title": "Serum neurofilament light chain reflects cortical neurodegeneration, not striatal DAT",
     "journal": "Parkinsonism & Related Disorders", "doi": "", "pmid": "",
     "zotero_key": "", "zotero_verified": 0, "ezproxy_verified": 0,
     "last_checked": "2026-04-15"},
    {"cite_key": "bartl2021ppmi", "author": "Bartl, M.", "year": 2021,
     "title": "NfL/sTREM2/YKL40/GFAP panel in PPMI PD — TREM2 does not discriminate",
     "journal": "PLoS ONE", "doi": "", "pmid": "",
     "zotero_key": "", "zotero_verified": 0, "ezproxy_verified": 0,
     "last_checked": "2026-04-15"},
]


def main() -> None:
    df = pd.DataFrame(ROWS)
    with ENGINE.begin() as conn:
        for _, r in df.iterrows():
            # audit.citation has no UNIQUE constraint (loaded via to_sql), so use
            # DELETE-then-INSERT for idempotency.
            conn.execute(
                text("DELETE FROM audit.citation WHERE cite_key = :cite_key"),
                {"cite_key": r["cite_key"]},
            )
            conn.execute(
                text(
                    "INSERT INTO audit.citation (cite_key, author, year, title, journal, "
                    "doi, pmid, zotero_key, zotero_verified, ezproxy_verified, last_checked) "
                    "VALUES (:cite_key, :author, :year, :title, :journal, :doi, :pmid, "
                    ":zotero_key, :zotero_verified, :ezproxy_verified, :last_checked)"
                ),
                r.to_dict(),
            )
    print(f"Inserted/updated {len(df)} citations in audit.citation")

    with ENGINE.connect() as conn:
        for key in df["cite_key"]:
            n = conn.execute(
                text("SELECT COUNT(*) FROM audit.citation WHERE cite_key = :k"),
                {"k": key},
            ).scalar()
            assert n == 1, f"missing {key}"
    print("All 4 citations confirmed present.")


if __name__ == "__main__":
    main()
