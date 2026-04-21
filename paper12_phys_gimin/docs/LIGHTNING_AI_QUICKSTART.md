# Lightning AI Quickstart — W6 FastSurfer Batch

End-to-end steps to run FastSurfer on Wang N=161 (~401 T1 scans) via Lightning AI A100 Studio. Budget: ~$7-10 single A100, ~$7-10 dual A100 (same total cost, half wall-clock).

## 1. Prepare your FreeSurfer license

FastSurfer depends on FreeSurfer for surface recon. License is free but requires registration:

1. Visit https://surfer.nmr.mgh.harvard.edu/registration.html
2. Fill out the form (research use; no institutional email required)
3. You'll receive an email with `license.txt` content
4. Save it locally as `license.txt`

## 2. Package NIfTI data locally (Mac)

After `scripts/w6_dcm2nii_fullcohort.py` finishes, produce an uploadable tarball:

```bash
cd /Users/blair.dupre/Projects/CSCI-FALL-2025/data/01_processed/GIMAN/t1_expansion_nifti

# Filter to Wang N=161 scans only (from manifest)
.venv/bin/python <<'EOF'
import csv, shutil
from pathlib import Path

src = Path("/Users/blair.dupre/Projects/CSCI-FALL-2025/data/01_processed/GIMAN/t1_expansion_nifti")
stage = Path("/tmp/nifti_wang_n161")
stage.mkdir(exist_ok=True)

manifest = "/Users/blair.dupre/.config/superpowers/worktrees/CSCI-FALL-2025/feat-paper12-phys-gimin/paper12_phys_gimin/data/wang_n161_manifest.csv"
copied = 0
with open(manifest) as f:
    for row in csv.DictReader(f):
        patno, date, proto = row["patno"], row["visit_date"], row["protocol"]
        subj_dir = src / f"PATNO_{patno}" / f"{date}_{proto}"
        for nii in subj_dir.glob("*.nii.gz"):
            dst = stage / nii.name
            shutil.copy(nii, dst)
            copied += 1

print(f"Staged {copied} NIfTI files")
EOF

tar -czf /tmp/nifti_wang_n161.tar.gz -C /tmp nifti_wang_n161
ls -lh /tmp/nifti_wang_n161.tar.gz  # expect ~5-10 GB
```

## 3. Launch Lightning AI Studio

1. Go to https://lightning.ai/studios
2. Click **New Studio** → **Machine Learning** template
3. Click the **compute selector** (top-right GPU icon)
4. Pick **A100 40GB** (or **2x A100** if available in your tier)
5. Studio boots in ~30-60 seconds

## 4. Upload data

**Option A — Lightning Files panel (browser):**
- Click the Files icon in the studio sidebar
- Upload `nifti_wang_n161.tar.gz` (~5-10 GB) and `license.txt`
- Upload `lightning_fastsurfer_setup.sh` (from `scripts/`)

**Option B — scp from terminal (faster):**
```bash
# Lightning provides SSH credentials in the Studio settings panel
scp /tmp/nifti_wang_n161.tar.gz user@studio.lightning.ai:/teamspace/studios/this_studio/
scp license.txt user@studio.lightning.ai:/teamspace/studios/this_studio/
scp lightning_fastsurfer_setup.sh user@studio.lightning.ai:/teamspace/studios/this_studio/
```

## 5. Run FastSurfer

Inside Lightning Studio terminal:

```bash
cd /teamspace/studios/this_studio
chmod +x lightning_fastsurfer_setup.sh
bash lightning_fastsurfer_setup.sh 2>&1 | tee fastsurfer_run.log
```

**Expected timing:**
- Single A100: ~5-7 hrs for Wang N=161 (401 scans)
- Dual A100: ~2.5-3.5 hrs for Wang N=161

**Monitor progress** (Lightning dashboard or `tail -f fastsurfer_run.log`):

```
[OK] PATNO100001_2020-10-07_SAG_3D_MPRAGE
[OK] PATNO100001_2022-11-29_SAG_3D_MPRAGE
...
```

## 6. Download results

After completion:
```bash
# Inside Lightning studio
ls -lh fastsurfer_stats.tar.gz  # expect ~2-3 GB

# From your Mac
scp user@studio.lightning.ai:/teamspace/studios/this_studio/fastsurfer_stats.tar.gz \
    /Users/blair.dupre/Projects/CSCI-FALL-2025/data/02_freesurfer/
```

## 7. Pause/stop the studio (stop billing)

- Click the compute selector → **Stop** (keeps the studio files for later; no GPU billing)
- Or **Delete** studio entirely if done

## 8. Extract features locally

Back on Mac:
```bash
cd /Users/blair.dupre/Projects/CSCI-FALL-2025/data/02_freesurfer
tar -xzf fastsurfer_stats.tar.gz
# Then run scripts/paper12_extract_freesurfer_features.py (to be written) → SQL
```

## Cost tracking

Lightning AI shows running total in the Studio settings panel. Stop immediately if you approach the $20 cap.

## Troubleshooting

- **"FreeSurfer license not found"**: Confirm `license.txt` is in `/teamspace/studios/this_studio/`
- **OOM on A100 40GB**: Unlikely (FastSurfer peak ~15 GB), but if it happens, add `--seg_only` first and skip surface recon
- **Session timeout**: Lightning Studios can run continuously if GPU is active. If you pause and resume, the script is idempotent (skips already-processed subjects)
