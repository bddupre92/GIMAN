# Lightning AI Quickstart — W6 FastSurfer Batch (SDK-driven)

End-to-end FastSurfer batch on Wang N=161 (401 T1 scans) via Lightning AI `lightning-sdk`. One Python command on your Mac drives everything: CPU setup → A100 swap → batch run → results download → CPU teardown → Studio stop.

**Budget:** ~$7-10 (single A100) or ~$7-10 (dual A100, same total $, half wall-clock).

---

## 1. One-time local setup (5 minutes)

```bash
# Install SDK
pip install lightning-sdk

# Get auth keys
# - Visit https://lightning.ai/me/keys (Account → Keys → Programmatic Login)
# - Copy LIGHTNING_USER_ID and LIGHTNING_API_KEY, export them:
export LIGHTNING_USER_ID="u_..."
export LIGHTNING_API_KEY="..."

# (Persist across shells by adding to ~/.zshrc)
```

## 2. Get FreeSurfer license (2 minutes, one-time)

FastSurfer uses FreeSurfer for surface recon; license is free but needs registration.

1. Visit https://surfer.nmr.mgh.harvard.edu/registration.html
2. Fill form (research use, no institutional email needed)
3. Receive email with license content
4. Save as `paper12_phys_gimin/data/license.txt`:

```bash
# Example contents — replace with your actual license:
cat > paper12_phys_gimin/data/license.txt <<'EOF'
your.email@und.edu
12345
*1a2b3c4d5e6f
 FSxyz123abc
EOF
```

## 3. Pre-flight check — local files

```bash
ls -lh /tmp/nifti_wang_n161.tar.gz          # 4.9 GB (W6 Step 3 output)
ls -lh paper12_phys_gimin/data/license.txt  # required after Step 2
ls -lh scripts/lightning_fastsurfer_setup.sh
ls -lh scripts/lightning_driver.py
```

## 4. Run the driver — one command

**Default (dual A100, ~2.5-3.5 hrs, ~$7):**

```bash
python scripts/lightning_driver.py --dual-gpu
```

**Single A100 fallback (if dual unavailable in your tier):**

```bash
python scripts/lightning_driver.py --single-gpu
```

**Install-only dry-run (no GPU, free):**

```bash
python scripts/lightning_driver.py --install-only
```

### What it does

| Phase | Machine | Time | Cost |
|---|---|---|---|
| Attach/create Studio `paper12-fastsurfer` | — | 5 s | $0 |
| Start on CPU-4 | CPU free tier | 30-60 s | $0 |
| Upload tarball (4.9 GB) + license + setup.sh | CPU free tier | 5-10 min | $0 |
| Run install phase (FreeSurfer + FastSurfer) | CPU free tier | 3-5 min | $0 |
| `switch_machine(A100_40GB_X_2)` | — | ~30 s | $0 |
| Run FastSurfer batch detached | **2× A100 40GB** | ~2.5-3.5 hrs | ~$7 |
| Poll progress every 3 min (logs streamed to Mac) | — | — | — |
| Download results tarball (~2-3 GB) | 2× A100 | 2-3 min | — |
| `switch_machine(CPU_X_4)` + `stop()` | CPU | 15 s | $0 |

## 5. Safety features

- **Hard timeout:** 8 hours on GPU phase. If the batch hangs, driver force-kills the process and tears down the Studio.
- **Emergency teardown:** On any unhandled exception, the driver attempts to switch back to CPU and stop the Studio before exiting.
- **Idempotent:** Re-running `lightning_driver.py` resumes — skips already-uploaded files and already-processed subjects.
- **Manual kill switch:** If anything looks wrong, go to https://lightning.ai/studios and click Stop on `paper12-fastsurfer`. Billing halts within 30 s.

## 6. Monitoring from terminal

The driver streams progress every 3 minutes. You'll see blocks like:

```
[poll #5] t=15.0m · tail:
[OK] PATNO100001_2020-10-07_SAG_3D_MPRAGE
[OK] PATNO100001_2022-11-29_SAG_3D_MPRAGE
[OK] PATNO100017_2024-03-15_SAG_3D_MPRAGE
...
```

Expected throughput: dual A100 ≈ ~2 scans/min (both GPUs busy) · single A100 ≈ ~1 scan/min.

## 7. Results

Driver downloads to:

```
paper12_phys_gimin/data/fastsurfer_stats.tar.gz   (~2-3 GB)
```

Extract locally:

```bash
mkdir -p data/02_freesurfer
tar -xzf paper12_phys_gimin/data/fastsurfer_stats.tar.gz -C data/02_freesurfer/
```

Then proceed to W6 Step 5 (feature extraction + CNODE fidelity gate).

## 8. Troubleshooting

- **"LIGHTNING_USER_ID not set"** → re-export the env vars, restart terminal if needed.
- **"license.txt not found"** → see Step 2; path is `paper12_phys_gimin/data/license.txt`.
- **"Machine.A100_40GB_X_2 not available"** → fallback with `--single-gpu`.
- **Long poll with empty log** → FastSurfer spends ~1 min on bias-field correction per scan before producing visible progress. If first poll after `Now running on Machine.A100_40GB_X_2` shows nothing for 5+ minutes, cancel with Ctrl-C; driver teardown will fire automatically.
- **Cost approaching $20** → Ctrl-C the driver (emergency teardown triggers) OR manually stop Studio at https://lightning.ai/studios.

## 9. Cost tracking

Lightning AI billing shows running total in Account → Usage. Rough burn rates:
- CPU-4: $0/hr (free tier, one per account)
- A100 40GB (1×): ~$1.40/hr
- A100 40GB (2×): ~$2.80/hr
- Stopped Studio: $0/hr compute (~pennies/day for Drive storage)

Dual A100 × 3 hrs = ~$8.40. Well inside $20 cap.
