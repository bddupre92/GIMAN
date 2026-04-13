"""
Phase 8.2 Week 1: Clinical Biomarkers Extraction (Simple Version)

Extract UPSIT, RBD, SCOPA-AUT, ESS from PPMI data.
"""

from pathlib import Path
import numpy as np
import pandas as pd


def main():
    print("=" * 70)
    print("PHASE 8.2 WEEK 1: CLINICAL BIOMARKERS EXTRACTION")
    print("=" * 70)

    base_dir = Path(__file__).resolve().parents[4]
    data_dir = base_dir / "data"
    output_dir = data_dir / "03_prodromal" / "enhanced"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load prodromal cohort
    prodromal_file = data_dir / "prodromal_cohort" / "prodromal_survival_data.csv"
    print(f"\nLoading prodromal cohort: {prodromal_file}")
    prodromal_df = pd.read_csv(prodromal_file)
    print(f"✓ Loaded {len(prodromal_df)} prodromal patients")

    # Initialize merged dataframe
    merged_df = prodromal_df[["PATNO"]].copy()

    # 1. UPSIT
    print("\n" + "-" * 70)
    print("1. UPSIT (Smell Test)")
    print("-" * 70)
    upsit_file = data_dir / "00_raw" / "GIMAN" / "ppmi_data_csv" / "University_of_Pennsylvania_Smell_Identification_Test__UPSIT__30Sep2025.csv"
    if upsit_file.exists():
        print(f"Loading: {upsit_file}")
        upsit_df = pd.read_csv(upsit_file)
        # Look for UPSIT total column
        upsit_cols = [c for c in upsit_df.columns if 'TOTAL' in c.upper() or 'UPSIT' in c.upper()]
        if upsit_cols and 'PATNO' in upsit_df.columns:
            upsit_bl = upsit_df[upsit_df.get('EVENT_ID', 'BL') == 'BL'][['PATNO', upsit_cols[0]]].drop_duplicates('PATNO')
            upsit_bl.columns = ['PATNO', 'UPSIT_SCORE']
            merged_df = merged_df.merge(upsit_bl, on='PATNO', how='left')
            print(f"✓ UPSIT: {merged_df['UPSIT_SCORE'].notna().sum()}/{len(merged_df)} ({100*merged_df['UPSIT_SCORE'].notna().mean():.1f}%)")
        else:
            merged_df['UPSIT_SCORE'] = np.nan
            print("⚠ No UPSIT total column found")
    else:
        merged_df['UPSIT_SCORE'] = np.nan
        print(f"⚠ File not found")

    # 2. RBD
    print("\n" + "-" * 70)
    print("2. RBD (Sleep Disorder)")
    print("-" * 70)
    rbd_file = data_dir / "00_raw" / "GIMAN" / "ppmi_data_csv" / "REM_Sleep_Behavior_Disorder_Questionnaire_18Sep2025.csv"
    if rbd_file.exists():
        print(f"Loading: {rbd_file}")
        rbd_df = pd.read_csv(rbd_file)
        # Sum RBD items or use total
        rbd_cols = [c for c in rbd_df.columns if c.startswith('RBD') and c[3:].isdigit()]
        if rbd_cols and 'PATNO' in rbd_df.columns:
            rbd_df['RBD_SCORE'] = rbd_df[rbd_cols].apply(pd.to_numeric, errors='coerce').sum(axis=1)
            rbd_bl = rbd_df[rbd_df.get('EVENT_ID', 'BL') == 'BL'][['PATNO', 'RBD_SCORE']].drop_duplicates('PATNO')
            merged_df = merged_df.merge(rbd_bl, on='PATNO', how='left')
            print(f"✓ RBD: {merged_df['RBD_SCORE'].notna().sum()}/{len(merged_df)} ({100*merged_df['RBD_SCORE'].notna().mean():.1f}%)")
        else:
            merged_df['RBD_SCORE'] = np.nan
            print("⚠ No RBD items found")
    else:
        merged_df['RBD_SCORE'] = np.nan
        print(f"⚠ File not found")

    # 3. SCOPA-AUT
    print("\n" + "-" * 70)
    print("3. SCOPA-AUT (Autonomic Dysfunction)")
    print("-" * 70)
    scopa_file = data_dir / "00_raw" / "GIMAN" / "ppmi_data_csv" / "SCOPA-AUT_18Sep2025.csv"
    if scopa_file.exists():
        print(f"Loading: {scopa_file}")
        scopa_df = pd.read_csv(scopa_file)
        # Sum SCAU items or use total
        scopa_cols = [c for c in scopa_df.columns if c.startswith('SCAU') and any(x in c for x in ['1', '2', '3', '4', '5', '6', '7', '8', '9'])]
        if scopa_cols and 'PATNO' in scopa_df.columns:
            # Convert to numeric, coercing errors
            for col in scopa_cols:
                scopa_df[col] = pd.to_numeric(scopa_df[col], errors='coerce')
            scopa_df['SCOPA_AUT_SCORE'] = scopa_df[scopa_cols].sum(axis=1)
            scopa_bl = scopa_df[scopa_df.get('EVENT_ID', 'BL') == 'BL'][['PATNO', 'SCOPA_AUT_SCORE']].drop_duplicates('PATNO')
            merged_df = merged_df.merge(scopa_bl, on='PATNO', how='left')
            print(f"✓ SCOPA-AUT: {merged_df['SCOPA_AUT_SCORE'].notna().sum()}/{len(merged_df)} ({100*merged_df['SCOPA_AUT_SCORE'].notna().mean():.1f}%)")
        else:
            merged_df['SCOPA_AUT_SCORE'] = np.nan
            print("⚠ No SCOPA-AUT items found")
    else:
        merged_df['SCOPA_AUT_SCORE'] = np.nan
        print(f"⚠ File not found")

    # 4. ESS
    print("\n" + "-" * 70)
    print("4. ESS (Sleepiness Scale)")
    print("-" * 70)
    ess_file = data_dir / "00_raw" / "GIMAN" / "ppmi_data_csv" / "Epworth_Sleepiness_Scale_18Sep2025.csv"
    if ess_file.exists():
        print(f"Loading: {ess_file}")
        ess_df = pd.read_csv(ess_file)
        # Sum ESS items or use total
        ess_cols = [c for c in ess_df.columns if c.startswith('ESS') and c[3:].isdigit()]
        if ess_cols and 'PATNO' in ess_df.columns:
            ess_df['ESS_SCORE'] = ess_df[ess_cols].apply(pd.to_numeric, errors='coerce').sum(axis=1)
            ess_bl = ess_df[ess_df.get('EVENT_ID', 'BL') == 'BL'][['PATNO', 'ESS_SCORE']].drop_duplicates('PATNO')
            merged_df = merged_df.merge(ess_bl, on='PATNO', how='left')
            print(f"✓ ESS: {merged_df['ESS_SCORE'].notna().sum()}/{len(merged_df)} ({100*merged_df['ESS_SCORE'].notna().mean():.1f}%)")
        else:
            merged_df['ESS_SCORE'] = np.nan
            print("⚠ No ESS items found")
    else:
        merged_df['ESS_SCORE'] = np.nan
        print(f"⚠ File not found")

    # Save
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)
    output_file = output_dir / "clinical_biomarkers.csv"
    merged_df.to_csv(output_file, index=False)
    print(f"✓ Saved: {output_file}")
    print(f"  Shape: {merged_df.shape}")

    # Coverage stats
    feature_cols = ['UPSIT_SCORE', 'RBD_SCORE', 'SCOPA_AUT_SCORE', 'ESS_SCORE']
    coverage = {col: 100 * merged_df[col].notna().mean() for col in feature_cols}
    avg_coverage = np.mean(list(coverage.values()))

    print(f"\n✓ Average coverage: {avg_coverage:.1f}%")
    for col, cov in coverage.items():
        print(f"  {col}: {cov:.1f}%")

    # Metadata
    import json
    metadata = {
        "extraction_date": "2025-10-12",
        "n_patients": len(merged_df),
        "n_features": 4,
        "features": feature_cols,
        "coverage": coverage,
        "average_coverage": avg_coverage
    }
    metadata_file = output_dir / "clinical_biomarkers_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Saved metadata: {metadata_file}")

    print("\n" + "=" * 70)
    print("CLINICAL BIOMARKERS EXTRACTION COMPLETE")
    print("=" * 70)
    print("\nNext: scripts/phase8_2/extract_cortical_thickness.py")


if __name__ == "__main__":
    main()
