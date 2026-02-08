# Resolving the PATNO Overlap Problem & Cohort-First Model Building

**Date:** 2026-02-08
**Context:** Zero PATNO overlap between SAA labels (920 patients) and training features (1,871 patients)

---

## THE PROBLEM IN ONE PICTURE

```
CURRENT PIPELINE (BROKEN):

   Phase 8.2 Features              SAA Labels
   ┌─────────────────────┐         ┌──────────────────┐
   │ unified_longitudinal │         │ saa_raw_labels   │
   │ _early_pd.csv        │         │ .csv             │
   │                      │         │                  │
   │ 1,871 unique PATNOs  │         │ 920 unique       │
   │ ├─ 381 Prodromal     │         │ PATNOs           │
   │ │  (Sep 18 pull)     │         │ (Sep 30 pull)    │
   │ └─ 1,490 Early PD    │         │                  │
   │    (SYNTHETIC features│         │ Source:          │
   │     mean-imputed)     │         │ Biospecimen CSV  │
   └──────────┬───────────┘         └────────┬─────────┘
              │                              │
              └──────────┬───────────────────┘
                         │
                    INNER JOIN ON PATNO
                         │
                    ┌────┴────┐
                    │ 0 rows  │  ← ZERO OVERLAP
                    └─────────┘
```

**Root causes:**

1. **Different source populations.** Features come from `Participant_Status_18Sep2025.csv` filtered to `ENROLL_CAT = 'Prodromal'` + artificially injected early PD. SAA labels come from `Current_Biospecimen_Analysis_Results_30Sep2025.csv` which spans ALL PPMI cohorts.

2. **Different data pulls.** Features use Sep 18, 2025 files. SAA uses Sep 30, 2025 biospecimen file. PPMI may reorganize PATNOs between releases.

3. **Cohort filtering mismatch.** The prodromal cohort extraction (extract_prodromal_cohort.py) filters to specific enrollment categories (`Prodromal`, `GENPD`, `GENUN`, `GENPS`). The biospecimen file contains patients across ALL categories. But the intersection is zero — meaning the 381 prodromal PATNOs don't appear in the biospecimen data, and the 920 biospecimen PATNOs don't appear in the prodromal enrollment filter.

4. **Synthetic data contamination.** 1,490 of the 1,871 training PATNOs are "early PD" patients added with **mean prodromal features** — not real data. These PATNOs almost certainly don't have CSF biospecimen samples.

---

## THE FIX: COHORT-FIRST DESIGN

The fundamental mistake was building features first, then trying to attach labels. **You must start from the labels and build outward.**

### Step 1: Start From What You Can Label

```
CORRECT PIPELINE:

   Step 1: Label Universe           Step 2: Feature Universe
   ┌────────────────────┐          ┌──────────────────────────┐
   │ ALL available       │          │ ALL available features   │
   │ label sources       │          │ from Feb 7/8 raw pull    │
   │                     │          │                          │
   │ A. Biospecimen SAA   │          │ Demographics (7,489)     │
   │    (920 PATNOs)     │          │ UPDRS-I (29K visits)     │
   │                     │          │ UPDRS-III (34K visits)   │
   │ B. Participant      │          │ MoCA (many)              │
   │    Status changes   │          │ DAT-SPECT SBR (1,459)   │
   │    (conversions)    │          │ Cortical thick (1,716)   │
   │                     │          │ Genetics (4,294-6,265)   │
   │ C. UPDRS motor      │          │ RBD (297)                │
   │    subtype          │          │ UPSIT (300+)             │
   │    (from items)     │          │ SCOPA-AUT (300+)         │
   └─────────┬──────────┘          └────────────┬─────────────┘
             │                                  │
             └──────────────┬───────────────────┘
                            │
                    LEFT JOIN: Labels → Features
                    (keep ALL labeled patients)
                            │
                    ┌───────┴──────────┐
                    │ Labeled cohort   │
                    │ with available   │
                    │ features per     │
                    │ patient          │
                    └───────┬──────────┘
                            │
                    Modality availability audit
                    (per patient: which features exist?)
                            │
                    ┌───────┴──────────┐
                    │ Final training   │
                    │ cohort with      │
                    │ feature + label  │
                    │ + modality mask  │
                    └──────────────────┘
```

### Step 2: Build Three Labeled Cohorts From Scratch

Each task gets its own cohort, built label-first from your Feb 7/8 raw CSVs:

---

## TASK A: SAA Classification

### Label Source
```
File: Current_Biospecimen_Analysis_Results (latest from Google Drive)
Filter: TESTNAME == 'Amprion Clinical Lab aSyn SAA, Semi Quantitative'
        OR TESTNAME == 'CSF Alpha-synuclein'
Dedup: One label per PATNO (baseline visit preferred, else earliest)
Expected: 920 patients (CSF α-syn) or 26 (Amprion gold standard)
```

**Critical decision: Which SAA definition?**

| Option | N patients | Quality | Recommendation |
|--------|-----------|---------|----------------|
| Amprion SAA (binary: Detected/Not Detected) | ~26 | Gold standard | Too few for ML |
| CSF α-synuclein > 80th percentile | ~920 | Proxy, arbitrary threshold | Not clinically validated |
| **CSF α-synuclein > published cutoff** | ~920 | **Use 1,000 pg/mL** | **Best: literature-based cutoff** |
| PPMI biological definition flag | Variable | If available in data | Check `Participant_Status` for biological definition column |

**Recommendation:** Check if your Feb 7/8 PPMI pull includes the **PPMI Biological Definition** column in `Participant_Status`. PPMI recently adopted a biological classification. If available, this is the cleanest label. If not, use CSF α-synuclein with a literature-derived cutoff (not percentile-based).

### Feature Assembly (For SAA-Labeled PATNOs Only)
```python
# Pseudocode for cohort-first assembly
saa_patnos = set(saa_labels['PATNO'].unique())  # 920 PATNOs

# For each feature source, extract ONLY these PATNOs
demographics = load_csv('Demographics*.csv').query('PATNO in @saa_patnos')
updrs_i = load_csv('MDS-UPDRS_Part_I*.csv').query('PATNO in @saa_patnos')
genetics = load_csv('iu_genetic_consensus*.csv').query('PATNO in @saa_patnos')
dat_spect = load_csv('Xing_Core_Lab*.csv').query('PATNO in @saa_patnos')
moca = load_csv('MoCA*.csv').query('PATNO in @saa_patnos')
upsit = load_csv('UPSIT*.csv').query('PATNO in @saa_patnos')
rbd = load_csv('RBD*.csv').query('PATNO in @saa_patnos')
scopa = load_csv('SCOPA-AUT*.csv').query('PATNO in @saa_patnos')

# Report modality availability for SAA cohort
for modality_name, modality_df in modalities.items():
    overlap = len(set(modality_df['PATNO']) & saa_patnos)
    print(f"{modality_name}: {overlap}/{len(saa_patnos)} = {overlap/len(saa_patnos):.0%}")
```

### Expected Overlap (Realistic Estimates)
```
SAA labels:       920 PATNOs (from biospecimen)
Demographics:     ~850-900 overlap (demographics covers nearly everyone)
UPDRS-I:          ~600-800 overlap (clinical visits common)
UPDRS-III:        ~600-800 overlap
Genetics:         ~400-600 overlap (not all genotyped)
DAT-SPECT:        ~200-400 overlap (imaging subset)
MoCA:             ~500-700 overlap
UPSIT:            ~200-400 overlap
RBD:              ~100-200 overlap
SCOPA-AUT:        ~200-400 overlap
```

**You won't know the real numbers until you run this on your actual Feb 7/8 data.** This is Step 1.

---

## TASK B: Prodromal Conversion

### Label Source
```
File: Participant_Status (latest from Google Drive)
Definition:
  - Baseline: enrolled as Prodromal/Genetic-at-risk/SWEDD
  - Label: later reclassified to "Parkinson's Disease" in subsequent visit
  - Time: months from enrollment to reclassification (or last visit if censored)
Expected: ~150-300 prodromal PATNOs, ~10-25% conversion rate
```

### Feature Assembly
Same approach — start from labeled prodromal PATNOs, then pull features.

---

## TASK C: Motor Subtype

### Label Source
```
File: MDS-UPDRS_Part_III (latest from Google Drive)
Filter: Patients with PD diagnosis AND baseline UPDRS-III completed
Definition: Tremor-dominant (TD) vs PIGD from UPDRS-III items
  tremor_items = [NP3PTRMR, NP3PTRML, NP3KTRMR, NP3KTRML, ...]
  pigd_items = [NP3GAIT, NP3FRZGT, NP3PSTBL, NP3RISNG, NP3POSTR]
  ratio = sum(tremor) / (sum(pigd) + 0.001)
  TD if ratio >= 1.15, PIGD if ratio <= 0.90, Indeterminate otherwise
Expected: ~800-1500 PD patients with complete baseline UPDRS-III
```

### Feature Assembly (Separate From Label Items)
- Label uses UPDRS-III individual items → these are BANNED from features
- Features: demographics, genetics, DAT-SPECT, MoCA, UPSIT, RBD, SCOPA-AUT, CSF

---

## IMPLEMENTATION: THE COHORT-FIRST PIPELINE

### New Script: `scripts/build_labeled_cohorts.py`

```python
"""
Cohort-First Data Assembly Pipeline

The critical principle: LABELS FIRST, then attach features.
Never build features and try to attach labels after.

Usage:
    python scripts/build_labeled_cohorts.py \
        --raw-dir /path/to/google/drive/ppmi_data_csv \
        --output-dir data/05_cohort_first \
        --tasks saa,conversion,subtype
"""

class CohortFirstBuilder:
    """
    Builds ML-ready datasets by:
    1. Extracting labels from authoritative sources
    2. Identifying labeled PATNOs
    3. Pulling ALL available features for labeled PATNOs only
    4. Reporting per-modality overlap (not filtering by it)
    5. Generating modality-availability masks
    """

    def __init__(self, raw_dir: Path):
        self.raw_dir = raw_dir
        self.raw_files = self._discover_raw_files()

    def _discover_raw_files(self) -> dict:
        """Find latest version of each PPMI CSV (handles multiple dates)."""
        # Pattern: filename_DDMMMYYYY.csv
        # Always use the LATEST file per logical dataset
        ...

    def build_saa_cohort(self) -> CohortResult:
        """Build SAA classification cohort (label-first)."""
        # 1. Extract SAA labels
        labels = self._extract_saa_labels()
        labeled_patnos = set(labels['PATNO'])

        # 2. Pull features for labeled PATNOs
        features = self._assemble_features(labeled_patnos)

        # 3. Merge labels + features
        cohort = labels.merge(features, on='PATNO', how='left')

        # 4. Generate modality mask
        modality_mask = self._compute_modality_mask(cohort)

        # 5. Audit and report
        self._audit_cohort(cohort, modality_mask, task='saa')

        return CohortResult(cohort, modality_mask, labels)

    def _assemble_features(self, target_patnos: set) -> pd.DataFrame:
        """
        Pull ALL available features for a given set of PATNOs.
        Use baseline visit. Report what's available, don't filter.
        """
        features = pd.DataFrame({'PATNO': list(target_patnos)})

        for modality_name, loader_fn in self.modality_loaders.items():
            modality_df = loader_fn(self.raw_dir)
            # Filter to target PATNOs and baseline visit
            modality_baseline = modality_df[
                (modality_df['PATNO'].isin(target_patnos)) &
                (modality_df['EVENT_ID'] == 'BL')
            ]
            overlap = modality_baseline['PATNO'].nunique()
            total = len(target_patnos)
            print(f"  {modality_name}: {overlap}/{total} ({100*overlap/total:.1f}%)")

            features = features.merge(
                modality_baseline, on='PATNO', how='left'  # LEFT join, keep all
            )

        return features

    def _compute_modality_mask(self, cohort: pd.DataFrame) -> pd.DataFrame:
        """
        For each patient, which modalities have data?
        Returns binary mask DataFrame.
        """
        mask = pd.DataFrame({'PATNO': cohort['PATNO']})
        for modality, cols in self.modality_columns.items():
            available_cols = [c for c in cols if c in cohort.columns]
            if available_cols:
                mask[f'has_{modality}'] = cohort[available_cols].notna().any(axis=1).astype(int)
            else:
                mask[f'has_{modality}'] = 0
        return mask
```

### New Script: `scripts/audit_cohort_overlap.py`

Run this FIRST, before any modeling, to understand your actual data landscape:

```python
"""
Cohort Overlap Audit

Answers: "For each prediction task, how many labeled patients
have data in each modality?"

Run this on your Feb 7/8 raw pull BEFORE building any model.
"""

def audit_all_overlaps(raw_dir: Path) -> pd.DataFrame:
    """
    For each (task × modality) pair, compute:
    - N labeled patients
    - N with modality data
    - Overlap %

    Output: overlap_matrix.csv with tasks as rows, modalities as columns
    """
    tasks = {
        'saa': extract_saa_patnos(raw_dir),
        'conversion': extract_conversion_patnos(raw_dir),
        'subtype': extract_subtype_patnos(raw_dir),
    }

    modalities = {
        'demographics': load_demographics_patnos(raw_dir),
        'updrs_i': load_updrs_i_patnos(raw_dir),
        'updrs_iii': load_updrs_iii_patnos(raw_dir),
        'genetics': load_genetics_patnos(raw_dir),
        'dat_spect': load_dat_spect_patnos(raw_dir),
        'moca': load_moca_patnos(raw_dir),
        'upsit': load_upsit_patnos(raw_dir),
        'rbd': load_rbd_patnos(raw_dir),
        'scopa_aut': load_scopa_patnos(raw_dir),
        'cortical_thickness': load_fs7_patnos(raw_dir),
        'csf_biomarkers': load_csf_patnos(raw_dir),
    }

    results = []
    for task_name, task_patnos in tasks.items():
        row = {'task': task_name, 'n_labeled': len(task_patnos)}
        for mod_name, mod_patnos in modalities.items():
            overlap = len(task_patnos & mod_patnos)
            row[f'{mod_name}_n'] = overlap
            row[f'{mod_name}_pct'] = 100 * overlap / len(task_patnos) if task_patnos else 0
        results.append(row)

    return pd.DataFrame(results)
```

---

## DECISION TREE: WHAT TO DO BASED ON OVERLAP RESULTS

After running the overlap audit on your Feb 7/8 data:

```
IF saa_overlap >= 200 with >= 3 modalities:
    → Proceed with Task A (SAA classification)
    → Use modality attention masking for missing modalities
    → Target: >90% AUC

ELIF saa_overlap >= 50 but < 200:
    → Proceed with Task A but use LOOCV
    → Simpler model (XGBoost, not GNN)
    → Target: >85% AUC

ELIF saa_overlap < 50:
    → SAA task is not viable with current data
    → Pivot to Task C (motor subtype) which uses UPDRS-III items as labels
      and has ~1000+ PD patients available
    → Or pivot to Task B (conversion) if prodromal cohort is sufficient

IF conversion_overlap >= 100 with conversion rate >= 10%:
    → Proceed with Task B
    → Need >= 15 events for Cox modeling

IF subtype_overlap >= 300:
    → Proceed with Task C (largest, most reliable cohort)
    → This is your backup plan — always achievable
```

---

## WHAT YOU NEED TO DO RIGHT NOW

### Action 1: Run the Overlap Audit (30 minutes)

Point the audit script at your Feb 7/8 Google Drive raw pull. You need ONE number per cell:

| | Demographics | UPDRS-I | UPDRS-III | Genetics | DAT-SPECT | MoCA | UPSIT | RBD | SCOPA | Cortical | CSF |
|---|---|---|---|---|---|---|---|---|---|---|---|
| SAA (N=?) | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? |
| Conversion (N=?) | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? |
| Subtype (N=?) | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? |

**This table determines everything.** Until you have it, no modeling decision is valid.

### Action 2: Check for PPMI Biological Definition

PPMI recently added a **biological classification** column that may be in your Feb 7/8 data. Check:
```
Participant_Status*.csv → look for columns like:
  BIOSPCMN_DEFINITION, BIOLOGICAL_STATUS, BIO_CLASSIFICATION,
  SAA_STATUS, SYNUCLEIN_POSITIVE
```

If this exists, it's a much cleaner label source than the 80th percentile hack.

### Action 3: Verify PATNO Format Consistency

A common overlap-killer:
```python
# Check if PATNO is integer in one file and string in another
print(type(saa_labels['PATNO'].iloc[0]))   # e.g., int64
print(type(features['PATNO'].iloc[0]))     # e.g., str '3001'
# If types differ, merge returns 0 rows
```

Also check for leading zeros, whitespace, or prefix differences:
```python
print(saa_labels['PATNO'].head())     # e.g., 3001, 3002, 3003
print(features['PATNO'].head())       # e.g., 100232, 100677 ← DIFFERENT ID SPACE?
```

If the PATNO ranges don't even overlap numerically, these are fundamentally different patient populations (e.g., PPMI original vs. PPMI3 expansion).

### Action 4: Build Cohort-First (After Audit)

Based on overlap results, build the cohort using the label-first approach described above.

---

## WHY THE OLD APPROACH FAILED

The Phase 8.2 pipeline made three architectural mistakes:

1. **Feature-first design:** Built a 1,871-patient feature matrix, THEN tried to attach SAA labels. Should have been labels-first.

2. **Synthetic data injection:** Added 1,490 "early PD" patients with mean-imputed features. These patients have no real biospecimen data, guaranteeing zero SAA overlap.

3. **Different data pulls:** Features from Sep 18, SAA from Sep 30. Even a 12-day difference can change PATNO assignments in PPMI.

The fix is not to patch the merge — it's to **rebuild from scratch using the label-first principle** on a single consistent data pull (your Feb 7/8 download).
