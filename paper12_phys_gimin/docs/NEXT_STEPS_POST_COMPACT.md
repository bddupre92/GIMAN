# Paper 12 — Next Steps After Compact (2026-04-20 session end)

**Session ended:** 2026-04-20, user paused to download Wang 2025 CNODE PPMI T1 MRI data from LONI IDA (Advanced Image Search).

**2026-04-20 update — MRI DOWNLOADS COMPLETE.** User returned with 861 patient DICOM directories across `PPMI_MRI_1` (204) + `PPMI MRI_2` (657). Cross-referenced with PPMI `participant_status.cohort_definition = 'PD'`: **761 PD patients with ≥1 T1 visit, 327 with ≥2 visits, 206 with ≥3 visits** — 3–4× Wang's N=161 target. W6 data gate is UNBLOCKED.

**Strategy agreed (2026-04-20):** Dual-path execution — (a) subsample to Wang's exact N=161 (111×2-visit + 50×3-visit) for unambiguous fidelity-gate pass/fail, then (b) retrain on full PD pool (N=327 ≥2-visit) for extended Wang reproduction + adapter + phys-GIMIN benchmark.

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

## W6 execution checklist (data gate CLEARED 2026-04-20)

### Data locations (merged inventory, post-download)

- **New download batch 1:** `data/00_raw/PPMI_MRI_1/` (204 patients, SAG_3D_T1_FSPGR + MPRAGE)
- **New download batch 2:** `data/00_raw/PPMI MRI_2/` (657 patients, SAG_3D_MPRAGE + variants, many with 2+ timepoints already)
- **Existing:** `data/00_raw/GIMAN/PPMI_dcm/` (200 PD patients, mostly DaTScan-only, some T1)
- **Merged PD+T1 longitudinal cohort:** 761 / 327 (≥2 visits) / 206 (≥3 visits)

### Drive archive plan (retention policy)

**Hard rule: never run FreeSurfer recon-all directly into a Drive-synced path.** Drive File Stream's file-locking + millions-of-small-files pattern that FreeSurfer creates will tank recon-all or silently corrupt intermediate state. Keep FreeSurfer on local disk, migrate finished outputs to Drive after extraction.

**Target Drive folder:** `~/My Drive (dupre.blair92@gmail.com)/PPMIData_FreeSurfer/`

| Stage | Local (working) | Drive (archive) | Action |
|---|---|---|---|
| Raw DICOMs | `data/00_raw/PPMI_MRI_{1,2}/` | `PPMIData_FreeSurfer/dicom_archive/{PATNO}.tar.gz` | After FreeSurfer validates, tarball per-patient → Drive → delete local |
| dcm2niix NIfTI | `data/01_processed/GIMAN/t1_expansion_nifti/PATNO_{PATNO}/` | (optional) `PPMIData_FreeSurfer/nifti_archive/{PATNO}_{VISIT}.tar.gz` | Keep local (small, cheap reproducibility starting point) |
| FreeSurfer full | `data/02_freesurfer/{PATNO}_{VISIT}/` (~1 GB/scan) | — (too big) | Delete local `/mri`, `/surf`, `/label`, `/scripts` after extraction |
| FreeSurfer `/stats` + `aparc+aseg.mgz` | `data/02_freesurfer/{PATNO}_{VISIT}/stats/` | `PPMIData_FreeSurfer/freesurfer_stats/{PATNO}_{VISIT}/` | Copy to Drive + keep local |
| SQL features | `mechanistic.paper12_wang_features` | — | Authoritative; backup via `pg_dump` |

**Peak vs working set:** ~700 GB during recon-all (local scratch) → ~15 GB post-cleanup (local) + ~30 GB compressed (Drive).

**Pre-flight checks required before W6 Step 4 (FreeSurfer):**
1. Confirm Drive quota ≥ 50 GB free on `dupre.blair92@gmail.com` account (compressed DICOMs + `/stats` alone ≈ 30 GB)
2. Confirm the Drive folder is auto-syncing (create test file, verify appears in web UI)
3. Decide: is `dupre.blair92@gmail.com` the right long-term home vs. UND account (for reviewer access at peer review)?

### Step 1. Ingest new DICOMs into canonical layout

Merge `PPMI_MRI_1/` + `PPMI MRI_2/` directories into `data/00_raw/GIMAN/PPMI_dcm/{PATNO}/` (preserve protocol/date subdirs). Use `rsync -av --ignore-existing` to avoid overwriting existing DaTScan directories for the same PATNO.

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

**Must use PC/A5000 for this step** per `DUAL_MACHINE_SETUP.md`. CPU-only MPS is ~10× slower for FreeSurfer.

**CRITICAL: run on local disk, NOT on `~/My Drive/PPMIData_FreeSurfer/`.** Drive File Stream's file-locking + millions-of-small-files pattern that FreeSurfer creates will tank recon-all speed or silently corrupt intermediate state. Output to `data/02_freesurfer/{PATNO}_{VISIT}/` locally; migrate after extraction per Step 9.

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

**Same-commit rule:** bump `mechanistic` schema count in both worktree and main-repo CLAUDE.md from 27 → 28 (competitor_fidelity) or 29 (add `paper12_wang_features` table too). SQL registry hook will block the commit if count is stale.

### Step 9. Drive archive + local cleanup (after SQL load + validation)

**Pre-archive validation (non-negotiable):** spot-check 10 random patients — re-extract FreeSurfer `aseg.stats` + `aparc.stats` row values into Python, compare to `mechanistic.paper12_wang_features` rows. Equality to 6 decimal places required before any local deletion.

Per-patient migration loop (sketch):

```bash
DRIVE_ROOT="$HOME/My Drive (dupre.blair92@gmail.com)/PPMIData_FreeSurfer"
mkdir -p "$DRIVE_ROOT/dicom_archive" "$DRIVE_ROOT/freesurfer_stats"

for PATNO in $(cat data/02_freesurfer/validated_patnos.txt); do
  # 1. DICOM tarball → Drive → delete local
  tar -czf "$DRIVE_ROOT/dicom_archive/${PATNO}.tar.gz" \
    -C data/00_raw/GIMAN/PPMI_dcm "$PATNO"
  rm -rf "data/00_raw/GIMAN/PPMI_dcm/$PATNO"

  # 2. FreeSurfer /stats + aparc+aseg.mgz → Drive
  for VISIT_DIR in data/02_freesurfer/${PATNO}_*; do
    VISIT=$(basename "$VISIT_DIR")
    mkdir -p "$DRIVE_ROOT/freesurfer_stats/$VISIT"
    cp -r "$VISIT_DIR/stats" "$DRIVE_ROOT/freesurfer_stats/$VISIT/"
    cp "$VISIT_DIR/mri/aparc+aseg.mgz" "$DRIVE_ROOT/freesurfer_stats/$VISIT/"
    # 3. Delete bulky local FreeSurfer dirs
    rm -rf "$VISIT_DIR/mri" "$VISIT_DIR/surf" "$VISIT_DIR/label" "$VISIT_DIR/scripts"
  done
done

du -sh "$DRIVE_ROOT"        # expect ~30 GB
du -sh data/02_freesurfer/  # expect <5 GB after cleanup
```

**Do NOT delete `PPMI_MRI_1/` or `PPMI MRI_2/` raw directories until every PATNO has confirmed Drive-side tarball + SQL row.** Archive is append-only; reviewers may ask for raw DICOM access during peer review. Drive sync is cheaper than re-downloading from LONI IDA later.

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
