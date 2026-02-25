"""Compute NSD-ISS stages for BioFIND using Russo et al. (2025) methodology.

Replicates the staging from:
  Russo MJ et al., "Validation of the Neuronal alpha-Synuclein Disease
  Integrated Staging System (NSD-ISS) in the BioFIND cohort"
  npj Parkinson's Disease, 2025.

Reference code: github.com/dr-russo/nsd-iss_biofind

Key differences from PPMI staging:
  - BioFIND has NO DaT-SPECT: all diagnosed PD patients assumed D+
  - BioFIND has NO UPSIT: hyposmia criterion excluded
  - Stages 1A/1B collapsed to 1, 2A/2B collapsed to 2
  - Only S+ patients receive NSD-ISS staging

Variables needed (7 total):
  NP1COG   = MDS-UPDRS Part I Q1.1 (cognitive impairment, 0-4)
  MCATOT   = MoCA total score (0-30)
  P1TOT    = MDS-UPDRS Part I total EXCLUDING NP1COG
  P2TOT    = MDS-UPDRS Part II total (motor ADL)
  P3TOT    = MDS-UPDRS Part III total (motor exam)
  PDMEDYN  = On PD medication (binary)
  RBD_STATUS = RBDSQ >= 6 (binary)

Author: GIMAN Research Team
Date: February 2026
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
BIOFIND_DIR = ROOT / "data" / "00_raw" / "BioFind"
OUTPUT_DIR = ROOT / "data" / "04_staging"


# ── NSD-ISS Staging Functions (from Russo et al. 2025) ──────────────────

def compute_nsd_iss_stage(row: pd.Series) -> int:
    """Assign NSD-ISS stage based on Russo et al. (2025) thresholds.

    Returns stage 1-6, or 0 if unclassifiable.
    """
    np1cog = row.get("NP1COG", np.nan)
    mcatot = row.get("MCATOT", np.nan)
    p1tot = row.get("P1TOT", np.nan)
    p2tot = row.get("P2TOT", np.nan)
    p3tot = row.get("P3TOT", np.nan)
    pdmedyn = row.get("PDMEDYN", np.nan)
    rbd_status = row.get("RBD_STATUS", np.nan)

    # Handle missing values
    if pd.isna(np1cog) or pd.isna(p3tot):
        return 0  # Unclassifiable

    np1cog = int(np1cog)
    p3tot = float(p3tot)
    mcatot = float(mcatot) if not pd.isna(mcatot) else np.nan
    p1tot = float(p1tot) if not pd.isna(p1tot) else np.nan
    p2tot = float(p2tot) if not pd.isna(p2tot) else np.nan
    pdmedyn = int(pdmedyn) if not pd.isna(pdmedyn) else 0
    rbd_status = int(rbd_status) if not pd.isna(rbd_status) else 0

    # Stage 6: Severe
    if (np1cog == 4 and not pd.isna(mcatot) and mcatot <= 24):
        return 6
    if (not pd.isna(p2tot) and p2tot >= 40):
        return 6
    if (not pd.isna(p1tot) and p1tot >= 37):
        return 6

    # Stage 5: Severe functional impairment
    if (np1cog == 3 and not pd.isna(mcatot) and mcatot <= 24):
        return 5
    if (np1cog == 4 and not pd.isna(mcatot) and mcatot >= 25):
        return 5
    if (not pd.isna(p2tot) and 27 <= p2tot <= 39):
        return 5
    if (not pd.isna(p1tot) and 25 <= p1tot <= 36):
        return 5

    # Stage 4: Moderate functional impairment
    if (np1cog == 2 and not pd.isna(mcatot) and mcatot <= 24):
        return 4
    if (np1cog == 3 and not pd.isna(mcatot) and mcatot >= 25):
        return 4
    if (not pd.isna(p2tot) and 14 <= p2tot <= 26):
        return 4
    if (not pd.isna(p1tot) and 13 <= p1tot <= 24):
        return 4
    if (not pd.isna(p1tot) and p1tot >= 13 and pdmedyn == 1 and rbd_status == 1):
        return 4

    # Stage 3: Mild functional impairment
    if (np1cog == 1 and not pd.isna(mcatot) and mcatot <= 24):
        return 3
    if (np1cog == 2 and not pd.isna(mcatot) and mcatot >= 25):
        return 3
    if (not pd.isna(p2tot) and 3 <= p2tot <= 13 and (p3tot >= 5 or pdmedyn == 1)):
        return 3

    # Stage 2: Clinical motor signs, no functional impairment
    if (np1cog == 1 and not pd.isna(mcatot) and mcatot >= 25):
        return 2
    if (p3tot >= 5 or pdmedyn == 1):
        return 2
    if rbd_status == 1:
        return 2

    # Stage 1: No clinical signs
    if (np1cog == 0 and p3tot < 5 and pdmedyn == 0 and rbd_status == 0):
        return 1

    return 0  # Unclassifiable


def compute_domain_stages(row: pd.Series) -> dict:
    """Compute per-domain stages (cognitive, motor, non-motor)."""
    np1cog = row.get("NP1COG", np.nan)
    mcatot = row.get("MCATOT", np.nan)
    p1tot = row.get("P1TOT", np.nan)
    p2tot = row.get("P2TOT", np.nan)
    p3tot = row.get("P3TOT", np.nan)
    pdmedyn = row.get("PDMEDYN", np.nan)
    rbd_status = row.get("RBD_STATUS", np.nan)

    # Cognitive domain
    cog_stage = np.nan
    if not pd.isna(np1cog):
        np1cog = int(np1cog)
        mc = float(mcatot) if not pd.isna(mcatot) else np.nan
        if np1cog == 0:
            cog_stage = 1
        elif np1cog == 1 and not pd.isna(mc) and mc >= 25:
            cog_stage = 2
        elif (np1cog == 1 and not pd.isna(mc) and mc <= 24) or \
             (np1cog == 2 and not pd.isna(mc) and mc >= 25):
            cog_stage = 3
        elif (np1cog == 2 and not pd.isna(mc) and mc <= 24) or \
             (np1cog == 3 and not pd.isna(mc) and mc >= 25):
            cog_stage = 4
        elif (np1cog == 3 and not pd.isna(mc) and mc <= 24) or \
             (np1cog == 4 and not pd.isna(mc) and mc >= 25):
            cog_stage = 5
        elif np1cog == 4 and not pd.isna(mc) and mc <= 24:
            cog_stage = 6

    # Motor domain
    motor_stage = np.nan
    if not pd.isna(p2tot):
        p2 = float(p2tot)
        p3 = float(p3tot) if not pd.isna(p3tot) else 0
        med = int(pdmedyn) if not pd.isna(pdmedyn) else 0
        if p2 >= 40:
            motor_stage = 6
        elif 27 <= p2 <= 39:
            motor_stage = 5
        elif 14 <= p2 <= 26:
            motor_stage = 4
        elif 3 <= p2 <= 13 and (p3 >= 5 or med == 1):
            motor_stage = 3
        elif p3 >= 5 or med == 1:
            motor_stage = 2

    # Non-motor domain
    nm_stage = np.nan
    if not pd.isna(p1tot):
        p1 = float(p1tot)
        rbd = int(rbd_status) if not pd.isna(rbd_status) else 0
        if p1 >= 37:
            nm_stage = 6
        elif 25 <= p1 <= 36:
            nm_stage = 5
        elif 13 <= p1 <= 24:
            nm_stage = 4
        elif rbd == 1:
            nm_stage = 2

    return {
        "cognitive_stage": cog_stage,
        "motor_stage": motor_stage,
        "nonmotor_stage": nm_stage,
    }


# ── Data Loading ─────────────────────────────────────────────────────────

def load_biofind_staging_variables() -> pd.DataFrame:
    """Load and merge all BioFIND variables needed for NSD-ISS staging."""

    # 1. Case-control (filter to PD)
    cc = pd.read_csv(BIOFIND_DIR / "amp_pd_case_control.csv", low_memory=False)
    cc_bl = cc.drop_duplicates(subset=["participant_id"], keep="first")
    pd_ids = set(cc_bl.loc[cc_bl["case_control_other_at_baseline"].str.lower() == "case", "participant_id"])
    logger.info(f"PD patients: {len(pd_ids)}")

    # 2. SAA consensus
    saa = pd.read_csv(BIOFIND_DIR / "biofind_saa_consensus.csv")
    saa["participant_id"] = "BF-" + saa["PATNO"].astype(str)
    saa_pd = saa[saa["participant_id"].isin(pd_ids)]
    logger.info(f"SAA data for PD: {len(saa_pd)} patients")
    saa_pos = saa_pd[saa_pd["SAA_RESULT"] == True]
    logger.info(f"S+ PD patients: {len(saa_pos)}")

    # Start with S+ patients
    df = saa_pos[["participant_id", "SAA_RESULT", "SAA_SCORE"]].copy()

    # 3. UPDRS Part I → NP1COG and P1TOT
    u1 = pd.read_csv(BIOFIND_DIR / "MDS_UPDRS_Part_I.csv", low_memory=False)
    u1_pd = u1[u1["participant_id"].isin(pd_ids)]
    # Get baseline visit (M0_5 preferred, fallback M0)
    u1_bl = u1_pd[u1_pd["visit_name"] == "M0_5"]
    if len(u1_bl) == 0:
        u1_bl = u1_pd[u1_pd["visit_name"] == "M0"]
    u1_bl = u1_bl.drop_duplicates(subset=["participant_id"], keep="first")

    staging = df.merge(u1_bl[["participant_id"]], on="participant_id", how="inner")

    # NP1COG = code_upd2101_cognitive_impairment (0-4 numeric)
    u1_bl_data = u1_bl.set_index("participant_id")
    staging["NP1COG"] = staging["participant_id"].map(
        u1_bl_data["code_upd2101_cognitive_impairment"]
    ).astype(float)

    # P1TOT = Part I summary score MINUS NP1COG
    staging["P1TOT_RAW"] = staging["participant_id"].map(
        u1_bl_data["mds_updrs_part_i_summary_score"]
    ).astype(float)
    staging["P1TOT"] = staging["P1TOT_RAW"] - staging["NP1COG"]

    # 4. UPDRS Part II → P2TOT
    u2 = pd.read_csv(BIOFIND_DIR / "MDS_UPDRS_Part_II.csv", low_memory=False)
    u2_pd = u2[u2["participant_id"].isin(pd_ids)]
    u2_bl = u2_pd[u2_pd["visit_name"] == "M0_5"]
    if len(u2_bl) == 0:
        u2_bl = u2_pd[u2_pd["visit_name"] == "M0"]
    u2_bl = u2_bl.drop_duplicates(subset=["participant_id"], keep="first")
    u2_bl_data = u2_bl.set_index("participant_id")
    staging["P2TOT"] = staging["participant_id"].map(
        u2_bl_data["mds_updrs_part_ii_summary_score"]
    ).astype(float)

    # 5. UPDRS Part III → P3TOT
    u3 = pd.read_csv(BIOFIND_DIR / "MDS_UPDRS_Part_III.csv", low_memory=False)
    u3_pd = u3[u3["participant_id"].isin(pd_ids)]
    u3_bl = u3_pd[u3_pd["visit_name"] == "M0_5"]
    if len(u3_bl) == 0:
        u3_bl = u3_pd[u3_pd["visit_name"] == "M0"]
    u3_bl = u3_bl.drop_duplicates(subset=["participant_id"], keep="first")
    u3_bl_data = u3_bl.set_index("participant_id")
    staging["P3TOT"] = staging["participant_id"].map(
        u3_bl_data["mds_updrs_part_iii_summary_score"]
    ).astype(float)

    # 6. MoCA → MCATOT
    moca = pd.read_csv(BIOFIND_DIR / "MOCA.csv", low_memory=False)
    moca_pd = moca[moca["participant_id"].isin(pd_ids)]
    moca_bl = moca_pd[moca_pd["visit_name"] == "M0_5"]
    if len(moca_bl) == 0:
        moca_bl = moca_pd[moca_pd["visit_name"] == "M0"]
    moca_bl = moca_bl.drop_duplicates(subset=["participant_id"], keep="first")
    moca_bl_data = moca_bl.set_index("participant_id")
    staging["MCATOT"] = staging["participant_id"].map(
        moca_bl_data["moca_total_score"]
    ).astype(float)

    # 7. PD Medication → PDMEDYN
    # Prefer LONI IDA file (Use_of_PD_Medication_*) which has PDMEDYN directly
    loni_med_files = list(BIOFIND_DIR.glob("Use_of_PD_Medication_*.csv"))
    if loni_med_files:
        pdmed_loni = pd.read_csv(loni_med_files[0], low_memory=False)
        # LONI IDA uses PATNO (numeric), convert to participant_id
        pdmed_loni["participant_id"] = "BF-" + pdmed_loni["PATNO"].astype(str)
        # Filter to baseline (BL event)
        pdmed_bl = pdmed_loni[pdmed_loni["EVENT_ID"] == "BL"]
        if len(pdmed_bl) == 0:
            pdmed_bl = pdmed_loni.drop_duplicates(subset=["participant_id"], keep="first")
        else:
            pdmed_bl = pdmed_bl.drop_duplicates(subset=["participant_id"], keep="first")
        pdmed_bl_data = pdmed_bl.set_index("participant_id")
        staging["PDMEDYN"] = staging["participant_id"].map(
            pdmed_bl_data["PDMEDYN"]
        ).astype(float)
        logger.info(f"  PD Medication from LONI IDA: {staging['PDMEDYN'].notna().sum()}/{len(staging)}")
    else:
        # Fallback: AMP-PD BigQuery PD_Medical_History
        pdmed = pd.read_csv(BIOFIND_DIR / "PD_Medical_History.csv", low_memory=False)
        pdmed_pd = pdmed[pdmed["participant_id"].isin(pd_ids)]
        pdmed_bl = pdmed_pd.drop_duplicates(subset=["participant_id"], keep="first")
        pdmed_bl_data = pdmed_bl.set_index("participant_id")
        if "use_of_pd_medication" in pdmed_bl.columns:
            med_map = pdmed_bl_data["use_of_pd_medication"]
            staging["PDMEDYN"] = staging["participant_id"].map(med_map)
            staging["PDMEDYN"] = staging["PDMEDYN"].map(
                lambda x: 1 if str(x).lower() in ("yes", "true", "1", "1.0") else 0
                if not pd.isna(x) else np.nan
            )
        else:
            staging["PDMEDYN"] = np.nan

    # 8. RBD → RBD_STATUS (RBDSQ >= 6)
    rbd = pd.read_csv(BIOFIND_DIR / "REM_Sleep_Stiasny_Kolster.csv", low_memory=False)
    rbd_pd = rbd[rbd["participant_id"].isin(pd_ids)]
    rbd_bl = rbd_pd[rbd_pd["visit_name"] == "M0_5"]
    if len(rbd_bl) == 0:
        rbd_bl = rbd_pd[rbd_pd["visit_name"] == "M0"]
    rbd_bl = rbd_bl.drop_duplicates(subset=["participant_id"], keep="first")
    rbd_bl_data = rbd_bl.set_index("participant_id")
    staging["RBDSQ_TOTAL"] = staging["participant_id"].map(
        rbd_bl_data["rbd_summary_score"]
    ).astype(float)
    staging["RBD_STATUS"] = (staging["RBDSQ_TOTAL"] >= 6).astype(int)

    logger.info(f"Staging variables assembled for {len(staging)} S+ PD patients")
    logger.info(f"Variable coverage:")
    for col in ["NP1COG", "MCATOT", "P1TOT", "P2TOT", "P3TOT", "PDMEDYN", "RBD_STATUS"]:
        n = staging[col].notna().sum()
        logger.info(f"  {col}: {n}/{len(staging)} ({n/len(staging)*100:.1f}%)")

    return staging


# ── Main Pipeline ────────────────────────────────────────────────────────

def stage_biofind():
    """Compute NSD-ISS stages for BioFIND S+ PD patients."""
    staging = load_biofind_staging_variables()

    # Apply staging function
    staging["nsd_iss_stage"] = staging.apply(compute_nsd_iss_stage, axis=1)

    # Domain stages
    domain_stages = staging.apply(compute_domain_stages, axis=1, result_type="expand")
    staging = pd.concat([staging, domain_stages], axis=1)

    # Summary
    print(f"\n{'='*60}")
    print(f"BioFIND NSD-ISS Staging Results (Russo et al. 2025 method)")
    print(f"{'='*60}")
    print(f"S+ PD patients staged: {len(staging)}")
    print(f"\nStage Distribution:")
    for stage in sorted(staging["nsd_iss_stage"].unique()):
        n = (staging["nsd_iss_stage"] == stage).sum()
        pct = n / len(staging) * 100
        print(f"  Stage {stage}: {n:4d} ({pct:5.1f}%)")

    # Compare to Russo et al. published distribution
    print(f"\nComparison to Russo et al. (2025) published results (N=104):")
    russo_dist = {1: 0, 2: 9, 3: 58, 4: 35, 5: 2, 6: 0}
    print(f"{'Stage':>8s} {'Ours':>8s} {'Russo':>8s} {'Published %':>12s}")
    print("-" * 40)
    for stage in [1, 2, 3, 4, 5, 6]:
        ours = (staging["nsd_iss_stage"] == stage).sum()
        theirs = russo_dist.get(stage, 0)
        pct = theirs / 104 * 100
        print(f"  {stage:>5d} {ours:>8d} {theirs:>8d} {pct:>11.1f}%")

    # Domain stage distribution
    print(f"\nDomain-Specific Stages:")
    for domain in ["cognitive_stage", "motor_stage", "nonmotor_stage"]:
        print(f"\n  {domain}:")
        vals = staging[domain].dropna()
        for s in sorted(vals.unique()):
            n = (vals == s).sum()
            print(f"    Stage {int(s)}: {n}")

    # Map stages to our ML target encoding
    # Binary: NSD+ (stages 1+) vs NSD- (stage 0)
    # Since all BioFIND S+ patients are NSD+ by definition, binary = all 1
    # More useful: map to our 4 target formulations
    staging["target_binary"] = 1  # All S+ → NSD+

    # Three-class: Early(0-1), Mild(2B), Impaired(3-4)
    stage_to_3class = {0: np.nan, 1: 0, 2: 0, 3: 1, 4: 2, 5: 2, 6: 2}
    staging["target_3class"] = staging["nsd_iss_stage"].map(stage_to_3class)

    # Full ordinal: map Russo stages to our PPMI stages
    # Russo: 1,2,3,4,5,6  → PPMI: 1,2B,3,4 (collapse 5→4, 6→4)
    stage_to_ppmi = {0: np.nan, 1: 0, 2: 1, 3: 2, 4: 3, 5: 3, 6: 3}
    staging["target_nsd_positive"] = staging["nsd_iss_stage"].map(stage_to_ppmi)

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "biofind_nsd_iss_staging.csv"
    staging.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")
    print(f"{'='*60}")

    return staging


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    stage_biofind()
