#!/usr/bin/env python3
"""Phase 5 Task 1: Populate PosteriorStore HDF5 from existing Phase 2 chain parquets.

The existing Phase 2 IS v5 chains at outputs/mechanistic_twin/data/posteriors/
chains_is_v5{,_waveb}/ contain resampled equal-weight posteriors for 1,065
patients (304 Wave A + 761 Wave B), 5,000 samples each, with columns
[k_n, alpha_tox, T_tox].

This script loads each patient's chain parquet and saves it to HDF5 via
PosteriorStore.save() as version v1, source tagged by cohort.

Enables bidirectional updating: update_posterior() in updater.py can reweight
these samples via SIR when new observations arrive.

Run:
    .venv/bin/python scripts/mechanistic_twin/phase5_persist_full_posteriors.py

Output:
    outputs/mechanistic_twin/paper10_mech_vs_giman/phase2_posteriors_full_samples.h5

Author: Blair Dupre
Date: 2026-04-13
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from giman_pipeline.mechanistic_twin_v2.posterior_store import (
    PatientPosterior,
    PosteriorStore,
)
from scripts.mechanistic_twin._reproducibility import capture_provenance

CHAINS_WAVE_A = PROJECT_ROOT / "outputs/mechanistic_twin/data/posteriors/chains_is_v5"
CHAINS_WAVE_B = PROJECT_ROOT / "outputs/mechanistic_twin/data/posteriors/chains_is_v5_waveb"
SUMMARY_CSV = PROJECT_ROOT / "outputs/mechanistic_twin/data/posteriors/phase2_combined_1065.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs/mechanistic_twin/paper10_mech_vs_giman"
OUTPUT_H5 = OUTPUT_DIR / "phase2_posteriors_full_samples.h5"

PARAM_NAMES = ["k_n", "alpha_tox", "T_tox"]


def load_chain(path: Path) -> tuple[np.ndarray, int]:
    """Load a chain parquet, return (samples, n_samples)."""
    df = pd.read_parquet(path)
    samples = df[PARAM_NAMES].to_numpy()
    return samples, len(samples)


def load_patient_to_store(
    store: PosteriorStore,
    patno: int,
    chain_path: Path,
    source: str,
) -> dict:
    """Load one patient's chain and save to PosteriorStore as v1."""
    samples, n = load_chain(chain_path)
    weights = np.ones(n) / n  # equal weights (IS already resampled)
    ess = float(n)  # ESS = n for uniform weights

    post = PatientPosterior(
        patno=patno,
        version=1,
        samples=samples,
        weights=weights,
        ess=ess,
        log_marg_lik=0.0,  # Not stored in chain parquets; placeholder
        param_names=PARAM_NAMES,
        source=source,
    )
    store.save(post)
    return {
        "patno": patno,
        "n_samples": n,
        "k_n_median": float(np.median(samples[:, 0])),
        "T_tox_median": float(np.median(samples[:, 2])),
        "source": source,
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    prov = capture_provenance(
        script_path=Path(__file__),
        repo_root=PROJECT_ROOT,
        input_files=[SUMMARY_CSV],  # Chain directories not hashed individually
    )

    # Remove existing store to ensure clean rebuild
    if OUTPUT_H5.exists():
        print(f"Removing existing store: {OUTPUT_H5}")
        OUTPUT_H5.unlink()

    store = PosteriorStore(OUTPUT_H5)

    print(f"Loading Wave A chains from {CHAINS_WAVE_A}...")
    wave_a_files = sorted(CHAINS_WAVE_A.glob("PATNO_*.parquet"))
    print(f"  Found {len(wave_a_files)} Wave A patient chains")

    print(f"Loading Wave B chains from {CHAINS_WAVE_B}...")
    wave_b_files = sorted(CHAINS_WAVE_B.glob("PATNO_*.parquet"))
    print(f"  Found {len(wave_b_files)} Wave B patient chains")

    t0 = time.time()
    records = []
    for chain_file in wave_a_files:
        patno = int(chain_file.stem.replace("PATNO_", ""))
        rec = load_patient_to_store(store, patno, chain_file, source="phase2_is_v5_wavea")
        records.append(rec)

    for chain_file in wave_b_files:
        patno = int(chain_file.stem.replace("PATNO_", ""))
        rec = load_patient_to_store(store, patno, chain_file, source="phase2_is_v5_waveb")
        records.append(rec)

    elapsed = time.time() - t0
    print(f"\nLoaded {len(records)} patients in {elapsed:.1f}s")

    # Verify: compare loaded medians to published summary
    print("\nVerifying against published summary...")
    summary_df = pd.read_csv(SUMMARY_CSV)
    summary_df["PATNO"] = summary_df["PATNO"].astype(int)

    verification_ok = True
    max_k_n_diff = 0.0
    max_t_tox_diff = 0.0
    n_verified = 0
    for rec in records[:50]:  # Spot-check first 50
        pub = summary_df[summary_df["PATNO"] == rec["patno"]]
        if pub.empty:
            continue
        pub_row = pub.iloc[0]
        if pd.notna(pub_row.get("k_n_median")):
            k_n_diff = abs(rec["k_n_median"] - pub_row["k_n_median"]) / max(abs(pub_row["k_n_median"]), 1e-10)
            max_k_n_diff = max(max_k_n_diff, k_n_diff)
            if k_n_diff > 0.10:  # >10% discrepancy
                print(f"  WARN: PATNO {rec['patno']} k_n_median: chain={rec['k_n_median']:.3e}, pub={pub_row['k_n_median']:.3e}")
                verification_ok = False
        if pd.notna(pub_row.get("T_tox_median")):
            t_tox_diff = abs(rec["T_tox_median"] - pub_row["T_tox_median"]) / max(abs(pub_row["T_tox_median"]), 1e-10)
            max_t_tox_diff = max(max_t_tox_diff, t_tox_diff)
        n_verified += 1

    print(f"  Verified {n_verified} patients")
    print(f"  Max k_n rel diff: {max_k_n_diff:.4f}")
    print(f"  Max T_tox rel diff: {max_t_tox_diff:.4f}")
    print(f"  Verification: {'PASS' if verification_ok else 'FAIL'}")

    # Save summary JSON
    summary = {
        "n_patients_wave_a": len(wave_a_files),
        "n_patients_wave_b": len(wave_b_files),
        "n_total": len(records),
        "n_samples_per_patient": 5000,
        "param_names": PARAM_NAMES,
        "store_path": str(OUTPUT_H5.relative_to(PROJECT_ROOT)),
        "store_size_mb": OUTPUT_H5.stat().st_size / 1e6,
        "elapsed_seconds": elapsed,
        "verification": {
            "n_verified": n_verified,
            "max_k_n_rel_diff": max_k_n_diff,
            "max_T_tox_rel_diff": max_t_tox_diff,
            "passed": verification_ok,
        },
        "_provenance": prov,
    }
    summary_path = OUTPUT_DIR / "phase2_posteriors_full_samples_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nStore: {OUTPUT_H5} ({summary['store_size_mb']:.1f} MB)")
    print(f"Summary: {summary_path}")
    print(f"Patients in store: {len(store)}")


if __name__ == "__main__":
    main()
