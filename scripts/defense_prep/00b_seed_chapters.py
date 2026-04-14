#!/usr/bin/env python3
"""Phase B B0.2 — Seed chapter table with 15 dissertation chapters + Appendix D.

Populates the chapter metadata that all audit downstream queries join against.
All chapters start with defensibility_score = 'pending' until B1-B15 audit
runs set them to green/yellow/red.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB = PROJECT_ROOT / "outputs/defense_prep/e2e_audit/claim_lineage.sqlite3"

CHAPTERS = [
    # (chapter_id, number, title, tex_path, paper_label)
    (1, 1, "Introduction", "outputs/dissertation/chapters/ch01_introduction.tex", None),
    (2, 2, "Systematic Literature Review", "outputs/dissertation/chapters/ch02_systematic_review.tex", None),
    (3, 3, "Paper 1: NSD-ISS Stage Classification", "outputs/dissertation/chapters/ch03_paper1.tex", "Paper 1"),
    (4, 4, "Paper 2: GIMIN Imputation", "outputs/dissertation/chapters/ch04_paper2.tex", "Paper 2"),
    (5, 5, "Paper 3: Graph-Informed Digital Twin (Transition Timing)", "outputs/dissertation/chapters/ch05_paper3.tex", "Paper 3"),
    (6, 6, "Paper 4: Conformalized Survival Analysis", "outputs/dissertation/chapters/ch06_paper4.tex", "Paper 4"),
    (7, 7, "Paper 5: Temporal Validation", "outputs/dissertation/chapters/ch07_paper5.tex", "Paper 5"),
    (8, 8, "Paper 6: Unified Clinical Decision Support", "outputs/dissertation/chapters/ch08_paper6.tex", "Paper 6"),
    (9, 9, "Paper 7: Per-Patient Bayesian Calibration of Coupled ODE", "outputs/dissertation/chapters/ch09_paper7.tex", "Paper 7"),
    (10, 10, "Paper 8a: Spatial Propagation Identifiability", "outputs/dissertation/chapters/ch10_paper8a.tex", "Paper 8a"),
    (11, 11, "Paper 8b: Regional DaT-SPECT Decline Rates", "outputs/dissertation/chapters/ch11_paper8b.tex", "Paper 8b"),
    (12, 12, "Paper 9: Three-Pathway PK-PD Analysis", "outputs/dissertation/chapters/ch12_paper9.tex", "Paper 9"),
    (13, 13, "Paper 10: Bidirectional-Ready Mechanistic Twin + NASEM Audit", "outputs/dissertation/chapters/ch13_paper10.tex", "Paper 10"),
    (14, 14, "Discussion (Unified, Papers 1-10)", "outputs/dissertation/chapters/ch14_discussion.tex", None),
    (15, 15, "Conclusion + Future Work Catalog", "outputs/dissertation/chapters/ch15_conclusion.tex", None),
    # Appendix D: mechanistic twin mathematical reference (non-numbered chapter)
    (99, 99, "Appendix D: Mechanistic Digital Twin Mathematical Reference", "outputs/dissertation/chapters/mechtwin_review.tex", "Appendix D"),
]


def main() -> None:
    if not DB.exists():
        raise SystemExit("DB not bootstrapped. Run 00_bootstrap_claim_lineage_db.py first.")
    conn = sqlite3.connect(DB)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executemany(
        "INSERT OR REPLACE INTO chapter (chapter_id, chapter_number, title, tex_path, paper_label, defensibility_score) "
        "VALUES (?, ?, ?, ?, ?, 'pending')",
        CHAPTERS,
    )
    conn.commit()
    n = conn.execute("SELECT COUNT(*) FROM chapter").fetchone()[0]
    conn.close()
    print(f"[B0.2] Seeded {n} chapters (15 numbered + Appendix D).")
    for row in CHAPTERS:
        print(f"  Ch {row[1]:>2}: {row[2]}")


if __name__ == "__main__":
    main()
