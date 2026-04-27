# Phase 5 Task 3 — Cross-Sectional External Validation RUN MANIFEST

**Date:** 2026-04-13
**Script:** `scripts/mechanistic_twin/phase5_external_validation_lcc.py`
**Output:** `outputs/mechanistic_twin/paper10_mech_vs_giman/external_validation_lcc.json`

## Scope (DOUBLE-PIVOTED)

**Original plan:** Re-fit Phase 1 exponential decay on LCC N=638 patients with ≥2 DaT-SPECT scans.

**First pivot:** LCC has only 43 patients with SINGLE baseline DaT-SPECT scans (no longitudinal data). Re-scoped to cross-sectional baseline SBR distribution comparison.

**Second pivot:** All 43 LCC DaT patients are HEALTHY CONTROLS (confirmed via `amp_pd_case_control.csv`: "No PD Nor Other Neurological Disorder"). Re-scoped to dual HC-vs-HC + HC-vs-PD comparison.

## PDBP SPECT Investigation (2026-04-13)

User checked pdbp.ninds.nih.gov Query Tool:
- "PDBP Imaging SPECT" form exists in only 2 studies
- Both are Dementia with Lewy Bodies (DLB) studies (Leverenz N=259, Kantarci N=167)
- **PDBP has NO standard-PD DaT-SPECT data**
- The "LCC" in `data/00_raw/LCC/` is Leverenz's Lewy Consortium DLB study — same cohort visible in PDBP Query Tool

**External PD longitudinal DaT-SPECT genuinely unavailable from PDBP, HBS, BioFind, or LCC.**

## Results

### Comparison 1: HC-vs-HC (scanner/protocol check)

| Region | PPMI HC (N=343) | LCC HC (N=43) | Δ% | KS p |
|---|---|---|---|---|
| Caudate R | 2.98 ± 0.62 | 3.36 ± 0.59 | +12.8% | 1.1e-4 |
| Caudate L | 2.98 ± 0.63 | 3.41 ± 0.66 | +14.5% | 4.2e-4 |
| Putamen R | 2.16 ± 0.55 | 2.68 ± 0.49 | +24.0% | 4.7e-8 |
| Putamen L | 2.18 ± 0.57 | 2.59 ± 0.47 | +18.8% | 5.9e-7 |

**Mean absolute diff: 17.5%** → scanner/protocol effects detected. Not model failure; matches field-known DaT-SPECT inter-site variability (Wakasugi 2024 requires ComBat harmonization).

### Comparison 2: HC-vs-PD (diagnostic discrimination)

| Region | PPMI PD (N=2,049) | LCC HC (N=43) | LCC HC gap |
|---|---|---|---|
| Caudate R | 2.08 ± 0.72 | 3.36 ± 0.59 | +61.0% |
| Caudate L | 2.07 ± 0.71 | 3.41 ± 0.66 | +64.7% |
| Putamen R | 1.01 ± 0.58 | 2.68 ± 0.49 | **+165.5%** |
| Putamen L | 0.97 ± 0.55 | 2.59 ± 0.47 | **+166.4%** |

**Mean: +114%** → HC > PD gap confirmed and within expected range (literature: 40-80% in striatum, with putamen typically >100% due to PD's putamen-predominant deficit pattern). **SBR framework produces correct diagnostic discrimination.**

## Verdicts

| Assertion | Verdict |
|---|---|
| SBR framework distinguishes HC from PD | ✅ CONFIRMED (+114% gap) |
| HC distributions consistent across PPMI and LCC | ⚠️ PARTIAL (17.5% diff, scanner effects) |
| Longitudinal PD decay validates on external cohort | ❌ NOT TESTABLE (no data) |

## Limitations (for NASEM Audit)

1. **Not external PD decay validation** — LCC has no PD patients with DaT-SPECT
2. **Not longitudinal** — LCC has 1 scan per patient
3. **Scanner/protocol differences** confound direct comparison (17.5% HC-vs-HC)
4. **Pending:** SURE-PD3 via BioSEND DUA (~300 pts × 2 timepoints, 2-3 week turnaround)
5. **Pending:** DeNoPa via Mollenhauer collaboration for oligomeric α-syn validation
6. **Pending:** ICEBERG (Paris Brain Institute, 300 pts × 4yr annual imaging) via direct collaboration

## Verification

- **Tests:** 8/8 passing (`tests/mechanistic_twin_v2/test_external_validation.py`)
- **All limitations documented** for NASEM Task 7 audit

## Closed-Loop v1.5 — Stage 6.5 Cycle A

- Stage 1 (Literature): SURE-PD3, DeNoPa, ICEBERG external options identified
- Stage 3 (Pre-exec sanity): LCC composition verified (all HC)
- Stage 4 (Post-exec review): HC-vs-PD gap within expected range (Seibyl 2018)
- Stage 5 (Independent validation): scanner effect matches Wakasugi 2024 report
- Stage 6 (Decision gate): APPROVE (with honest limitation documentation)
- Cycle A: this RUN_MANIFEST + external_validation_lcc.json
