# Paper 12 W6 — Resume Guide (2026-04-21 evening)

**Session closed:** 2026-04-21, multi-path FastSurfer compute exploration — GPU wasteful, Mac too slow, pivoting to user's Threadripper+A5000 desktop.

**Branch:** `feat/paper12-phys-gimin` · **Worktree:** `~/.config/superpowers/worktrees/CSCI-FALL-2025/feat-paper12-phys-gimin` · **Latest:** `7eeb4b9`

---

## Resume command (one-liner)

```
"Continue Paper 12 W6 from paper12_phys_gimin/docs/NEXT_STEPS_POST_COMPACT.md — CNODE clean-room done, FastSurfer compute pivoting to Threadripper+A5000 desktop via WSL2+Docker. Write scripts/windows_wsl2_fastsurfer.sh and walk user through setup."
```

---

## Where we are (2026-04-21)

### ✅ Done
- Wang N=161 manifest + full 327-PD cohort manifest (committed)
- 952 T1 scans converted to NIfTI via dcm2niix (committed)
- `/tmp/nifti_wang_n161.tar.gz` (4.9 GB) ready for upload
- License.txt at `paper12_phys_gimin/data/license.txt`
- **W6 Step 5: Wang CNODE clean-room implementation + 99/99 tests** (commit `9b7dd3e`)
- Pipeline **end-to-end validated** via Mac Docker 3-scan smoke test (all 3 produced `stats/aseg+DKT.stats`)

### 🚫 Dead paths (don't retry these)
- **Lightning A100 GPU** — FastSurfer full pipeline is 97% CPU-bound. GPU idle 97% of time. Wasteful for the workload.
- **Mac Docker (Rosetta2 emulation)** — ~50 min/scan. 401 × 50 min = ~150 hrs. Too slow.
- **Mac native MPS** — PyTorch MPS backend missing `add_alpha_strided_cast_half_float` op. FastSurfer crashes. Known PyTorch limitation.
- **Lightning CPU_X_96** — requires Team plan at $150/mo. Not worth for one-off.

### 🎯 Best path forward: USER'S Threadripper WRX90 + RTX A5000 desktop

Spec: AMD Threadripper Pro WRX90 (up to 96 cores depending on SKU), RTX A5000 (24 GB GDDR6, 8192 CUDA), 256 GB RAM, 4 TB NVMe, Windows 11 Pro.

**Setup path: Windows 11 → WSL2 → Docker Desktop → FastSurfer Docker image with GPU passthrough.**

Expected wall-clock by CPU SKU:
| SKU | Cores | Parallel scans (4 threads each) | 401-scan full pipeline |
|-----|-------|--------------------------------|------------------------|
| 7965WX | 24 | 6 | ~17 hrs |
| 7975WX | 32 | 8 | ~13 hrs |
| 7985WX | 64 | 16 | ~6.5 hrs |
| 7995WX | 96 | 24 | **~4 hrs** |

Cost: $0 compute.

---

## Immediate next action — on Windows Threadripper

### Prerequisites (user confirms)
1. Windows 11 Pro (confirmed from spec)
2. NVIDIA A5000 driver up to date
3. `git` installed (or download repo ZIP)
4. ~50 GB free disk for Docker image + outputs

### IMPORTANT: Why WSL2 + Docker — FreeSurfer is Linux-only

FreeSurfer doesn't run natively on Windows. The pipeline is:
- Windows 11 host → WSL2 Ubuntu 22.04 (real Linux kernel) → Docker Desktop (Linux containers via WSL2 backend) → `deepmi/fastsurfer` container (Ubuntu 22 + FreeSurfer 7.4.1)

No emulation layer (unlike Mac where x86 Docker runs under Rosetta). Docker on Windows runs containers natively on x86_64 hardware — expect ~2-3 min/scan vs Mac's 23-50 min.

### Step 1 — Install WSL2 + Docker Desktop (one-time, ~30 min)

In Windows PowerShell (Admin):
```powershell
wsl --install -d Ubuntu-22.04
```
Reboot. Then inside Ubuntu WSL2 terminal:
```bash
# Verify GPU visible
nvidia-smi
```
Should show RTX A5000. If not, update NVIDIA driver from https://www.nvidia.com/Download/index.aspx.

Download Docker Desktop for Windows: https://www.docker.com/products/docker-desktop/
- Install → Settings → Resources → WSL Integration → **Enable for Ubuntu-22.04**
- Settings → Resources → **Enable GPU acceleration** (NVIDIA checkbox)

Verify Docker sees GPU:
```bash
docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi
```
Should print A5000 details.

### Step 2 — Clone repo on Windows (WSL2 side)

```bash
cd ~
git clone https://github.com/bddupre92/PD_PHD.git pd_phd
cd pd_phd
# Switch to the paper12 worktree branch
git fetch origin feat/paper12-phys-gimin
git checkout feat/paper12-phys-gimin
```

### Step 3 — Copy the NIfTI tarball + license to WSL2

From Mac (sync to Google Drive or direct SCP):
- `/tmp/nifti_wang_n161.tar.gz` (4.9 GB)
- `paper12_phys_gimin/data/license.txt`

To WSL2 Ubuntu:
- Download from Drive, or SCP over the LAN, or USB drive transfer
- Target locations:
  - `/home/<user>/fastsurfer_work/nifti_wang_n161.tar.gz`
  - `/home/<user>/fastsurfer_work/license.txt`

### Step 4 — Run the Windows/WSL2 FastSurfer batch

**Script to be created next session:** `scripts/windows_wsl2_fastsurfer.sh`

Design:
- Auto-detect CPU count (`nproc`)
- `N_PARALLEL = n_cpu / 4` (4 threads per FastSurfer scan)
- Use `docker run --gpus all deepmi/fastsurfer:latest` with full pipeline (no `--seg_only`)
- Each container: `--device cuda` for DL seg, CPU for surface recon
- GPU shared across parallel containers (DL seg is brief ~20 sec, low VRAM ~3 GB per scan)
- `xargs -P$N_PARALLEL` to launch concurrent containers
- Idempotent resume (`stats/aseg+DKT.stats` + `stats/lh.aparc.DKTatlas.mapped.stats` check)
- Output: per-subject dir under `/output/<subj_id>/`

### Step 5 — Monitor + collect results

```bash
# Live tail
tail -f ~/fastsurfer_work/batch.log

# Progress check
find ~/fastsurfer_work/fastsurfer_out -name 'aseg+DKT.stats' | wc -l   # X / 401
```

When done, tarball results:
```bash
cd ~/fastsurfer_work/fastsurfer_out
tar -czf ~/fastsurfer_full_stats.tar.gz */stats/ */mri/aparc*.mgz
```

Then SCP to Mac for SQL load (W6 Step 8).

---

## Fallback paths if Windows desktop unavailable

1. **Google Colab Pro** (user has subscription): run `scripts/paper12_fastsurfer_colab.ipynb`. ~10 hrs on A100 if available, ~20 hrs on T4. Zero new cost. File manually uploaded via Drive.
2. **AWS c5.24xlarge** (96 vCPU): $4/hr × 4 hrs = ~$16. Clean SSH-based flow. Adapt `scripts/lightning_cpu_parallel_fastsurfer.sh` with minor path changes.
3. **UND HPC Talon** (ticket already submitted): 72 cores, free, but 1-5 day approval wait.

---

## Cross-device chat continuity (new Windows session)

**Claude Code conversations DO NOT sync across devices.** But the context persists via committed files:

1. **Memory files** — `~/.claude/projects/-Users-blair-dupre-Projects-CSCI-FALL-2025/memory/` (Mac-side). On Windows, Claude Code auto-loads from Windows-equivalent `.claude/projects/<project-hash>/memory/` — you'd need to manually recreate key memory files OR copy.

2. **What actually persists across devices (and is the "right" way)**:
   - `paper12_phys_gimin/docs/NEXT_STEPS_POST_COMPACT.md` (this file — committed)
   - `paper12_phys_gimin/docs/PHASE_1_SUMMARY.md`
   - `CLAUDE.md` (main project — committed)
   - Everything in the git repo

**Workflow on Windows:**
1. Clone the repo
2. Open Claude Code in the repo directory
3. Give resume command: *"Read paper12_phys_gimin/docs/NEXT_STEPS_POST_COMPACT.md and main CLAUDE.md; pick up from the Windows/WSL2 setup step."*
4. Claude Code reads these files automatically (via CLAUDE.md auto-load) and has full context.

**VS Code extension "Settings Sync" does NOT sync Claude Code chat history** — only VS Code settings/extensions. Chat is per-device.

---

## Key artifacts to reference on Windows machine

- Scripts folder: `scripts/`
  - `lightning_cpu_parallel_fastsurfer.sh` — adaptable template (Linux CPU+Docker path)
  - `local_fastsurfer_batch.py` — Mac Docker version (don't use on Windows; reference logic only)
  - `paper12_fastsurfer_colab.ipynb` — Colab fallback
  - `windows_wsl2_fastsurfer.sh` — **TO BE WRITTEN** next session
- Manifests: `paper12_phys_gimin/data/wang_n161_manifest.csv`, `pd_full_cohort_manifest.csv`
- Wang CNODE clean-room: `paper12_phys_gimin/baselines/wang_2025_cnode_ppmi/` (commit `9b7dd3e`)
- Adapter: `paper12_phys_gimin/src/phys_gimin/baseline_adapters/cnode_adapter.py`
- Tests: `paper12_phys_gimin/tests/test_wang_cnode.py` (5 tests, 99/99 suite)

---

## Current Mac state at session close

- Mac Docker smoke test (PID 17079) may still be running — produced 3/3 valid `aseg+DKT.stats`
- Safe to `kill $(pgrep -f local_fastsurfer_batch)` if you want to free Mac resources
- `caffeinate` may still be backgrounded — `pkill caffeinate` to stop
- `~/.lightning/credentials.json` set with correct bddupre92 UUID
- Lightning Studio `paper12-fastsurfer` is Stopped (billing halted)
