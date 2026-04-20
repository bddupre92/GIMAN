# Paper 12 — Next Steps After Compact (2026-04-20 session end)

**Session ended:** 2026-04-20, user paused to download Wang 2025 CNODE PPMI T1 MRI data from LONI IDA (Advanced Image Search).

**Branch:** `feat/paper12-phys-gimin` · **Latest commit:** `427285a` · **Tests:** 94/94 passing · **Worktree:** `~/.config/superpowers/worktrees/CSCI-FALL-2025/feat-paper12-phys-gimin`

---

## Resume command (one-liner)

```
"Continue Paper 12 Phase 2 from NEXT_STEPS_POST_COMPACT.md — pick up at the LONI IDA download status check."
```

---

## What's done (23 commits — full Phase 1 + W5)

- Phase 1 (W2–W4): 79/79 tests, full infrastructure, partitioned CONTINUE verdict (commits through `99f1e6c`)
- **Phase 2 W5 firmly closed**: de Rooij 2025 vendored (MIT), PPMI adapter, juliacall equivalence bridge verifying Python regularizers match Julia to **< 1e-10** (commits `3de536c`, `59bc828`, `c51a304`, `427285a`)
- License corrected across 10 scoping/plan docs (CC-BY → MIT for de Rooij)

---

## Where we paused — LONI IDA download for W6

Wang 2025 CNODE (arXiv 2511.04789) fidelity-gate needs **N=161 PD patients** with longitudinal T1 MRI (111 × 2 visits + 50 × 3 visits).

**Local inventory from pre-compact check:**

| Metric | Have | Wang needs | Gap |
|---|---|---|---|
| PD DICOM dirs | 200 | — | ✓ |
| PD with ≥1 T1 MRI visit | 56 | — | — |
| **PD with ≥2 T1 visits** | **14** | 111 | **−97** |
| **PD with ≥3 T1 visits** | **7** | 50 | **−43** |

Root cause: most local PD DICOM dirs have DaTScan only, not T1.

### LONI IDA Advanced Image Search filters (user was setting up)

- **PROJECT/PHASE:** PPMI ✓
- **SUBJECT → Research Group:** PD only
- **STUDY/VISIT → PPMI timepoints:** Baseline + Month 12 + Month 24
- **IMAGE → Modality:** MRI ✓
- **IMAGING PROTOCOL → Acquisition Type:** 3D ✓
- **IMAGING PROTOCOL → Weighting:** T1 ✓ (do NOT include T2)
- **Image Types:** Original ✓ + Post-processed ✓ (raw DICOM is primary)
- Manufacturer / Mfg Model / Field Strength / Acquisition Plane — leave blank (Wang used multi-vendor, all 3T)

**User was struggling to kick off the LONI download.** If still blocked:
- Check LONI IDA session state / re-authenticate
- Try the "Simple Image Search" mode instead of Advanced
- Alternatively use AMP-PD BigQuery to request the imaging collection via the data-request form

**Expected download:** ~15–30 GB raw DICOMs across ~100 patients × 2–3 scans.

---

## When downloads complete — W6 execution checklist

### Step 1. Ingest new DICOMs into local tree

Place in `data/00_raw/GIMAN/PPMI_dcm/{PATNO}/...` matching the existing layout. If LONI provides a zip, extract preserving directory structure.

Verify:
```bash
/Users/blair.dupre/Projects/CSCI-FALL-2025/.venv/bin/python <<'EOF'
from pathlib import Path
from collections import Counter

dcm_dir = Path("/Users/blair.dupre/Projects/CSCI-FALL-2025/data/00_raw/GIMAN/PPMI_dcm")
pd_patnos_with_t1 = 0
visit_counts = Counter()
for pdir in dcm_dir.iterdir():
    if not (pdir.is_dir() and pdir.name.isdigit()):
        continue
    t1_visits = set()
    for proto_dir in pdir.iterdir():
        if proto_dir.is_dir() and any(k in proto_dir.name.upper() for k in ["MPRAGE", "T1", "3D"]):
            for date_dir in proto_dir.iterdir():
                if date_dir.is_dir():
                    t1_visits.add(date_dir.name[:10])
    if t1_visits:
        pd_patnos_with_t1 += 1
        visit_counts[len(t1_visits)] += 1

print(f"Patients with ≥1 T1 visit: {pd_patnos_with_t1}")
print(f"Distribution: {dict(visit_counts)}")
print(f"≥2 visits: {sum(c for n, c in visit_counts.items() if n >= 2)} (target 111)")
print(f"≥3 visits: {sum(c for n, c in visit_counts.items() if n >= 3)} (target 50)")
EOF
```

### Step 2. Convert DICOM → NIfTI

Use `dcm2niix` (already on disk per earlier `t1_expansion_nifti/` pipeline):
```bash
dcm2niix -o data/01_processed/GIMAN/t1_expansion_nifti/PATNO_{PATNO}/ \
         data/00_raw/GIMAN/PPMI_dcm/{PATNO}/
```

### Step 3. Run FreeSurfer recon-all on each T1 scan

Heavy compute (~1 hr/scan on A5000 CUDA, longer on MPS). Extract:
- 68 subcortical volumes (Desikan-Killiany)
- Vertex-wise medial thickness

**Consider using PC/A5000 for this step** per `DUAL_MACHINE_SETUP.md`. CPU-only MPS will be very slow for FreeSurfer.

### Step 4. Dispatch W6 subagent (clean-room CNODE)

Prompt template saved inline below. Sonnet model. Scope:

1. Read Wang 2025 arXiv 2511.04789 §II.B–II.D carefully.
2. Clean-room `paper12_phys_gimin/baselines/wang_2025_cnode_ppmi/cnode.py` + `train.py` from paper equations.
3. Run on the FreeSurfer-extracted features (68 subcortical + vertex-wise thickness).
4. Fidelity gate: 5-fold CV RMSE ∈ [0.145, 0.177], R² ∈ [0.743, 0.909] (from `clean_room_verification_protocol.md` §4 row 2).
5. Build adapter at `src/phys_gimin/baseline_adapters/cnode_adapter.py` for PPMI 33-feature schema (wraps CNODE for phys-GIMIN benchmark pipeline).
6. Add 5 adapter tests.
7. Commit `feat(paper12-w6): Wang 2025 CNODE PPMI clean-room + adapter + fidelity gate`.

Full prompt template in `paper12_phys_gimin/docs/W6_SUBAGENT_PROMPT.md` (to be created at W6 start).

### Step 5. SQL updates after W6

Load the fidelity gate result into `mechanistic.paper12_competitor_fidelity`:
```sql
-- Schema to create in same commit as the data load
CREATE TABLE mechanistic.paper12_competitor_fidelity (
  id SERIAL PRIMARY KEY,
  competitor TEXT NOT NULL,           -- 'wang_2025_cnode' | 'de_rooij_2025' | 'demirkaya_2021' | 'zou_2025' | 'lagcnn_2024'
  benchmark TEXT NOT NULL,            -- 'ppmi_5fold_cv' | 'glucose_minimal_model' | 'retinal_snr22_56' | etc
  metric_name TEXT NOT NULL,
  published_value DOUBLE PRECISION,
  reproduced_value DOUBLE PRECISION,
  gap_percent DOUBLE PRECISION,
  within_10pct_gate BOOLEAN,
  commit_sha TEXT,
  notes TEXT,
  evaluated_at TIMESTAMP DEFAULT now()
);
```

This closes the per-competitor fidelity tracking loop for W5–W8.

**Same-commit rule:** bump `mechanistic` schema count in both worktree and main-repo CLAUDE.md from 27 → 28.

---

## W7 + W8 (after W6)

Sequenced per `Docs/superpowers/plans/2026-04-20-paper12-phase2-competitor-baselines.md`:

- **W7a:** Demirkaya 2021 CKF clean-room (retinal SNR 22.56, MAPE gate [3.19, 3.89])
- **W7b:** Zou 2025 MNODE-HGS clean-room (T1DEXI, RMSE gate [31.1, 37.9]) — parallel with W7a
- **W8:** LagCNN clean-room (Weather 12.5% mask, MSE gate [0.025, 0.031]) as DL imputation baseline

W7/W8 don't need additional data — the competitor original-dataset benchmarks use public data or synthetic.

---

## Pre-compact status

- **Git:** branch `feat/paper12-phys-gimin` is 23 commits ahead of main. Nothing uncommitted.
- **SQL:** 718 MB · 185 tables · 14 schemas · 2 paper12 tables in `mechanistic`. Matches CLAUDE.md registry.
- **Tests:** 94/94 passing.
- **Compute options available:** Mac MPS (primary), PC RTX A5000 (per DUAL_MACHINE_SETUP.md), Colab Pro + UND HPC (future).
- **Main-repo CLAUDE.md:** has uncommitted Phase 1 summary + registry drift on `feat/ch9-6-multichannel`. Stage in whatever commit happens next on that branch.

## Things NOT to forget

1. When downloads land, **subsample to N=161 PD** if we overshoot (Wang's exact cohort), OR report a larger cohort in the manuscript as "extended Wang reproduction."
2. **FreeSurfer version matters** for fidelity — Wang didn't specify. Use **v7.4.x** (most current, widely adopted) and document in `CLEAN_ROOM_NOTES.md`.
3. **juliacall bridge** is fully working — reusable pattern if any other competitor (unlikely) also has Julia code.
4. **Q2 gate amendment** (effect-size override) is pre-registered — Phase 2 results feeding into the gate use the amended rubric by default.
5. Check `memory/paper12_phys_gimin_status.md` when resuming — always auto-loaded at session start.
