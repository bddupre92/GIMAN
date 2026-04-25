"""Paper 1 R3-Q7 — Explicit NSD-ISS staging-assignment flowchart with counts per decision path.

Reviewer 3 asks: "For SAA-missing but D+ cases, please provide an explicit flow
chart or table with counts at each decision step to fully document how labels
were derived and how often each rule path was invoked."

This script enumerates every (S anchor × D anchor × clinical-signs × functional-
impairment) combination present in the 2,201 PPMI staging-results table and
counts how each combination mapped to a final NSD-ISS stage. The output
documents:

  1. Anchor availability by S/D/both/neither
  2. The specific rule path each cohort-stratum took
  3. SAA-missing + D-positive handling (Reviewer 3's specific concern)
  4. Final stage distribution as a sanity check (must sum to 2,201)

Sources:
  - PostgreSQL `staging.nsd_iss_staging_results` (2,201 rows; columns documented in
    `src/giman_pipeline/staging/nsd_iss.py::stage_cohort`).

Decision-path semantics follow `compute_nsd_iss_stage()` in
`src/giman_pipeline/staging/nsd_iss.py`:

  - `not s_positive` is `True` when `s_positive is None` (Known Issue:
    `nsd_iss.py:285`). So in the algorithm, S-missing patients are treated as
    S-negative when the genetic-risk branch is evaluated.
  - A patient enters Stages 1+ iff `s_positive is True OR d_positive is True`.
    SAA-missing + D+ patients enter Stages 1+ via the D anchor alone.
  - Functional impairment (mild/moderate/severe/complete) drives Stage 3/4/5/6.
  - Clinical signs (no impairment) drive Stage 2B.
  - No clinical signs + biological anchor drive Stage 1.
  - "unclassified" is reserved for patients with both anchors missing AND no
    genetic-risk fallback.

Output:
  - outputs/paper1_r2_responses/q_r3_q7_staging_flow.json
  - outputs/paper1_r2_responses/q_r3_q7_staging_flow_table.md
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from giman_pipeline.data.db import read_table

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s", datefmt="%H:%M:%S"
)
log = logging.getLogger("r3_q7_staging_flow")

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "outputs" / "paper1_r2_responses"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

JSON_OUT = OUTPUT_DIR / "q_r3_q7_staging_flow.json"
MD_OUT = OUTPUT_DIR / "q_r3_q7_staging_flow_table.md"


def s_label(v: object) -> str:
    if v is True:
        return "S+"
    if v is False:
        return "S-"
    return "S?"


def d_label(v: object) -> str:
    if v is True:
        return "D+"
    if v is False:
        return "D-"
    return "D?"


def main() -> None:
    log.info("Loading staging.nsd_iss_staging_results from PostgreSQL...")
    df = read_table("staging", "nsd_iss_staging_results")
    log.info("  Loaded %d rows × %d cols", len(df), df.shape[1])

    n_total = len(df)
    assert n_total == 2201, f"Expected 2,201 PPMI rows, got {n_total}"

    # ------------------------------------------------------------------
    # 1. Anchor availability summary
    # ------------------------------------------------------------------
    saa_avail = df["s_positive"].notna().sum()
    dat_avail = df["d_positive"].notna().sum()
    both_avail = (df["s_positive"].notna() & df["d_positive"].notna()).sum()
    neither_avail = (df["s_positive"].isna() & df["d_positive"].isna()).sum()
    saa_only = (df["s_positive"].notna() & df["d_positive"].isna()).sum()
    dat_only = (df["s_positive"].isna() & df["d_positive"].notna()).sum()

    anchor_availability = {
        "saa_available_count": int(saa_avail),
        "saa_available_pct": round(100 * saa_avail / n_total, 2),
        "dat_available_count": int(dat_avail),
        "dat_available_pct": round(100 * dat_avail / n_total, 2),
        "both_anchors_count": int(both_avail),
        "both_anchors_pct": round(100 * both_avail / n_total, 2),
        "saa_only_count": int(saa_only),
        "dat_only_count": int(dat_only),
        "neither_anchor_count": int(neither_avail),
        "neither_anchor_pct": round(100 * neither_avail / n_total, 2),
    }
    log.info(
        "Anchor coverage: SAA %d (%.1f%%), DaT %d (%.1f%%), both %d, neither %d",
        saa_avail,
        anchor_availability["saa_available_pct"],
        dat_avail,
        anchor_availability["dat_available_pct"],
        both_avail,
        neither_avail,
    )

    # ------------------------------------------------------------------
    # 2. Decision-path enumeration
    # ------------------------------------------------------------------
    # Group by every dimension that affects stage assignment.
    grouped = (
        df.groupby(
            [
                "s_positive",
                "d_positive",
                "has_clinical_signs",
                "has_functional_impairment",
                "functional_impairment_level",
                "nsd_iss_stage",
            ],
            dropna=False,
        )
        .size()
        .reset_index(name="n")
        .sort_values(["s_positive", "d_positive", "n"], ascending=[True, True, False], na_position="first")
    )

    # Build readable decision-path entries grouped by (S, D, clinical, impairment).
    # For each (S, D, clinical, impairment_level) combination, summarise how many
    # patients ended up in each final stage.
    path_rows: list[dict] = []
    path_id = 0
    grouping_cols = [
        "s_positive",
        "d_positive",
        "has_clinical_signs",
        "has_functional_impairment",
        "functional_impairment_level",
    ]
    for keys, sub in grouped.groupby(grouping_cols, dropna=False):
        path_id += 1
        s_pos, d_pos, has_clin, has_imp, imp_level = keys
        stage_dist = (
            sub.groupby("nsd_iss_stage")["n"].sum().astype(int).to_dict()
        )
        n_path = int(sub["n"].sum())
        path_rows.append(
            {
                "path_id": f"P{path_id:02d}",
                "saa_status": s_label(s_pos),
                "dat_status": d_label(d_pos),
                "has_clinical_signs": bool(has_clin) if has_clin is not None else None,
                "has_functional_impairment": bool(has_imp)
                if has_imp is not None
                else None,
                "functional_impairment_level": str(imp_level)
                if imp_level is not None and not (isinstance(imp_level, float) and pd.isna(imp_level))
                else "none",
                "n_patients": n_path,
                "assigned_stages": {str(k): int(v) for k, v in stage_dist.items()},
                "description": _describe_path(s_pos, d_pos, has_clin, has_imp, imp_level, stage_dist),
            }
        )

    # Sort paths so the largest cohorts appear first within each (S, D) cell.
    path_rows.sort(key=lambda r: (r["saa_status"], r["dat_status"], -r["n_patients"]))
    # Reassign path IDs after sort for stable ordering.
    for i, row in enumerate(path_rows, start=1):
        row["path_id"] = f"P{i:02d}"

    # Verify counts sum
    n_sum = sum(r["n_patients"] for r in path_rows)
    assert n_sum == n_total, f"Path counts sum to {n_sum}, expected {n_total}"
    log.info("Enumerated %d decision paths covering %d patients (sum check OK)", len(path_rows), n_sum)

    # ------------------------------------------------------------------
    # 3. SAA-missing + D-positive specific breakdown (Reviewer 3's concern)
    # ------------------------------------------------------------------
    saa_miss_d_pos = df[df["s_positive"].isna() & (df["d_positive"] == True)]
    n_smdp = len(saa_miss_d_pos)
    smdp_stage_dist = (
        saa_miss_d_pos["nsd_iss_stage"].value_counts().sort_index().astype(int).to_dict()
    )
    smdp_breakdown = {
        "description": (
            "SAA-missing + DaT-positive cohort. The staging algorithm enters Stages 1+ "
            "when EITHER anchor is positive (`s_positive is True OR d_positive is True` "
            "in `compute_nsd_iss_stage`), so the missing S anchor does not preclude "
            "biological staging when D is positive. The final stage within {1, 2B, 3, 4} "
            "is then determined by clinical signs and functional impairment per "
            "Simuni 2024."
        ),
        "n_patients": int(n_smdp),
        "rule_applied": (
            "Treated as biologically positive via D anchor alone; S-missing assumed "
            "non-S+ (the algorithm does NOT impute S+; missing S is allowed to coexist "
            "with D+ for stage assignment)."
        ),
        "stage_distribution": {str(k): int(v) for k, v in smdp_stage_dist.items()},
    }
    log.info(
        "SAA-missing + D+: %d patients → stages %s",
        n_smdp,
        smdp_breakdown["stage_distribution"],
    )

    # Also note SAA-missing + D-negative (these collapse to Stage 0)
    saa_miss_d_neg = df[df["s_positive"].isna() & (df["d_positive"] == False)]
    saa_miss_d_neg_dist = (
        saa_miss_d_neg["nsd_iss_stage"].value_counts().sort_index().astype(int).to_dict()
    )
    saa_miss_d_miss = df[df["s_positive"].isna() & df["d_positive"].isna()]
    saa_miss_d_miss_dist = (
        saa_miss_d_miss["nsd_iss_stage"].value_counts().sort_index().astype(int).to_dict()
    )

    saa_missing_summary = {
        "saa_missing_d_positive": smdp_breakdown,
        "saa_missing_d_negative": {
            "n_patients": int(len(saa_miss_d_neg)),
            "rule_applied": (
                "Both S- (assumed via missing) and D-: no biological anchor → Stage 0 "
                "(via the `has_biological is False` branch in compute_nsd_iss_stage)."
            ),
            "stage_distribution": {str(k): int(v) for k, v in saa_miss_d_neg_dist.items()},
        },
        "saa_missing_d_missing": {
            "n_patients": int(len(saa_miss_d_miss)),
            "rule_applied": (
                "Both anchors missing AND no genetic-risk fallback → 'unclassified'. "
                "These are the 4 patients excluded from all 5-class targets."
            ),
            "stage_distribution": {str(k): int(v) for k, v in saa_miss_d_miss_dist.items()},
        },
    }

    # ------------------------------------------------------------------
    # 4. Final stage distribution sanity check
    # ------------------------------------------------------------------
    stage_dist = df["nsd_iss_stage"].value_counts().sort_index().astype(int).to_dict()
    log.info("Final stage distribution: %s", stage_dist)

    # ------------------------------------------------------------------
    # 5. Write JSON
    # ------------------------------------------------------------------
    out = {
        "workstream": "r3_q7_staging_flow",
        "description": (
            "NSD-ISS staging-assignment flow chart for the 2,201 PPMI cohort. "
            "Documents how each (S, D, clinical, impairment) combination was mapped "
            "to a final NSD-ISS stage and addresses Reviewer 3's specific concern "
            "about SAA-missing + DaT-positive handling."
        ),
        "n_patients_total": int(n_total),
        "data_source": "postgresql://staging.nsd_iss_staging_results",
        "algorithm_reference": "src/giman_pipeline/staging/nsd_iss.py::compute_nsd_iss_stage",
        "literature_anchor": "Simuni et al., Lancet Neurology (2024)",
        "anchor_availability": anchor_availability,
        "decision_paths": path_rows,
        "saa_missing_handling": saa_missing_summary,
        "final_stage_distribution": {str(k): int(v) for k, v in stage_dist.items()},
        "sum_check_path_counts": int(n_sum),
        "sum_check_passes": bool(n_sum == n_total),
    }
    JSON_OUT.write_text(json.dumps(out, indent=2))
    log.info("Wrote %s", JSON_OUT)

    # ------------------------------------------------------------------
    # 6. Write markdown decision-path table
    # ------------------------------------------------------------------
    md_lines: list[str] = []
    md_lines.append("# Supplementary Table S-R3Q7 — NSD-ISS Staging Decision Paths\n")
    md_lines.append(
        f"**Cohort:** PPMI N = {n_total}. **Algorithm:** Simuni 2024 (operationalised in "
        "`src/giman_pipeline/staging/nsd_iss.py`).\n"
    )
    md_lines.append("\n## Anchor Availability\n")
    md_lines.append(
        "| Quantity | N | % |\n|---|---:|---:|\n"
        f"| SAA available | {saa_avail} | {anchor_availability['saa_available_pct']} |\n"
        f"| DaT-SPECT available | {dat_avail} | {anchor_availability['dat_available_pct']} |\n"
        f"| Both anchors | {both_avail} | {anchor_availability['both_anchors_pct']} |\n"
        f"| SAA only | {saa_only} | {round(100*saa_only/n_total,2)} |\n"
        f"| DaT only | {dat_only} | {round(100*dat_only/n_total,2)} |\n"
        f"| Neither anchor | {neither_avail} | {anchor_availability['neither_anchor_pct']} |\n"
    )

    md_lines.append("\n## Decision Paths (one row per unique anchor × clinical × impairment combination)\n")
    md_lines.append(
        "| Path | S | D | Clinical signs | Impairment level | N | Final stages (count) |\n"
        "|---|---|---|---|---|---:|---|\n"
    )
    for r in path_rows:
        clinic = (
            "yes"
            if r["has_clinical_signs"] is True
            else "no"
            if r["has_clinical_signs"] is False
            else "?"
        )
        imp = r["functional_impairment_level"] or "none"
        stage_str = ", ".join(f"{k}: {v}" for k, v in sorted(r["assigned_stages"].items()))
        md_lines.append(
            f"| {r['path_id']} | {r['saa_status']} | {r['dat_status']} | {clinic} | {imp} | "
            f"{r['n_patients']} | {stage_str} |\n"
        )

    md_lines.append(
        f"\n**Sum check:** path counts sum to {n_sum} = N total ({n_total}).\n"
    )

    md_lines.append("\n## Reviewer 3 specific concern: SAA-missing + DaT-positive\n")
    md_lines.append(
        f"\n**N = {n_smdp}** patients have a missing SAA result but a positive DaT-SPECT "
        "scan. The Simuni 2024 algorithm, as operationalised in this pipeline, enters "
        "Stages 1+ when **either** anchor is positive — the missing S anchor does not "
        "preclude biological staging when D is positive. Within this stratum, the final "
        "stage is then determined by clinical signs (UPDRS-III ≥ 10 or PRIMDIAG = idiopathic "
        "PD) and functional impairment (Hoehn & Yahr threshold). The resulting stage "
        "distribution for this stratum is:\n\n"
    )
    md_lines.append("| Stage | N |\n|---|---:|\n")
    for k, v in sorted(smdp_breakdown["stage_distribution"].items()):
        md_lines.append(f"| {k} | {v} |\n")

    md_lines.append(
        "\nFor completeness, the related strata are: SAA-missing + D-negative "
        f"(N = {len(saa_miss_d_neg)}, all assigned Stage 0 via the `has_biological is "
        f"False` branch) and SAA-missing + DaT-missing (N = {len(saa_miss_d_miss)}, "
        "assigned `unclassified` and excluded from 5-class targets).\n"
    )

    md_lines.append("\n## Final stage distribution (sanity check)\n")
    md_lines.append("| Stage | N | % |\n|---|---:|---:|\n")
    for k, v in sorted(stage_dist.items()):
        md_lines.append(f"| {k} | {v} | {round(100*v/n_total, 2)} |\n")

    MD_OUT.write_text("".join(md_lines))
    log.info("Wrote %s", MD_OUT)

    print("\nSummary:")
    print(f"  Decision paths enumerated: {len(path_rows)}")
    print(f"  SAA-missing + D+ patients: {n_smdp}")
    print(f"  SAA-missing + D+ stage distribution: {smdp_breakdown['stage_distribution']}")
    print(f"  Sum check: {n_sum} == {n_total} ({'OK' if n_sum == n_total else 'FAIL'})")


def _describe_path(
    s_pos: object,
    d_pos: object,
    has_clin: object,
    has_imp: object,
    imp_level: object,
    stage_dist: dict,
) -> str:
    """Generate a short human-readable description of a decision path."""
    parts = [s_label(s_pos), d_label(d_pos)]
    if has_clin is True:
        parts.append("clinical+")
    elif has_clin is False:
        parts.append("clinical-")
    if has_imp is True:
        parts.append(f"impairment={imp_level}")
    elif has_imp is False:
        parts.append("no-impairment")
    primary_stage = max(stage_dist.items(), key=lambda kv: kv[1])[0] if stage_dist else "?"
    parts.append(f"→ Stage {primary_stage}")
    return " | ".join(parts)


if __name__ == "__main__":
    main()
