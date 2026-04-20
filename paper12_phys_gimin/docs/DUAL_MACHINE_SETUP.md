# phys-GIMIN — Dual-Machine (+ HPC + Colab) Setup Guide

This package is designed to run on four environments:

| Environment | Device | Compute speed (relative) | Setup complexity |
|---|---|---|---|
| MacBook (Apple Silicon) | MPS | 1.0x | trivial — nothing to install |
| PC (RTX A5000, 24GB) | CUDA | ~3x | medium — CUDA + data sync |
| Colab Pro (A100/V100) | CUDA | ~3-5x | easy — notebook-based; data via Drive |
| UND HPC cluster | CUDA (varies) | ~3x | medium — SSH + scheduler |

**Key design principle:** **Mac is the authoritative dev environment** (code, DB, paper writing). Other machines are compute workers. Code syncs via `git`; data syncs via `rsync` or Google Drive.

---

## Before you start — required env var

Every machine that is NOT the original Mac MUST set:

```bash
export CSCI_FALL_2025_ROOT=/absolute/path/to/CSCI-FALL-2025
```

The `phys_gimin.utils.paths.get_project_root()` helper uses this env var to locate the main project. Without it, the fallback path only works on the original Mac (`/Users/blair.dupre/Projects/CSCI-FALL-2025`).

---

## Environment 1 — MacBook (Apple Silicon, MPS)

**Nothing to install.** Already configured. Just:

```bash
cd ~/.config/superpowers/worktrees/CSCI-FALL-2025/feat-paper12-phys-gimin
/Users/blair.dupre/Projects/CSCI-FALL-2025/.venv/bin/python \
  paper12_phys_gimin/scripts/phase1/run_smoke_benchmark.py \
  --config paper12_phys_gimin/configs/phys_gimin_lit.yaml \
  --mask-fractions 0.10 0.25 0.50 0.75 \
  --n-seeds 3 \
  --n-epochs 100 \
  --device mps \
  --no-mock-data \
  --output-dir paper12_phys_gimin/outputs/runs/smoke_mps_$(date +%Y%m%d_%H%M%S)
```

Expected wall time: 5-10 hours for 12 runs.

**MPS caveats:**

- Non-deterministic across runs — byte-identical checkpoints NOT guaranteed (unlike CPU).
- Test `test_deterministic_under_seed` may fail if run with `device=mps`; use `device=cpu` for determinism tests.
- Some PyG ops may fall back to CPU silently — not a correctness issue but can bottleneck.

---

## Environment 2 — PC (RTX A5000, non-Ada)

### One-time setup (2-3 hours)

**Assumes Windows with WSL2 Ubuntu or native Ubuntu Linux on the PC.**

#### 2.1 Install CUDA toolkit matching PyTorch 2.8.0

RTX A5000 (Ampere generation) supports CUDA 11.8 and 12.x. Pick CUDA 12.1 for the best PyTorch 2.8.0 wheel match:

```bash
# On WSL2 Ubuntu or native Linux:
wget https://developer.download.nvidia.com/compute/cuda/12.1.1/local_installers/cuda_12.1.1_530.30.02_linux.run
sudo sh cuda_12.1.1_530.30.02_linux.run
```

Verify: `nvidia-smi` shows the A5000 at driver >=530.

#### 2.2 Install PostgreSQL 17 + restore the DB

```bash
sudo apt install postgresql-17 postgresql-contrib
sudo -u postgres createuser -s $(whoami)
createdb giman_research

# rsync the DB dump from Mac (one-time)
rsync -avz <mac>:~/Projects/CSCI-FALL-2025/db_dump/schema_and_data.sql ./

# Restore
psql giman_research < schema_and_data.sql   # ~5 min for 702MB
```

Verify: `psql giman_research -c "SELECT COUNT(*) FROM features.paper2_gimin_cohort"` returns 35687.

#### 2.3 Clone the repo + install Python env

```bash
git clone https://github.com/bddupre92/PD_PHD.git CSCI-FALL-2025
cd CSCI-FALL-2025
git checkout feat/paper12-phys-gimin

# Create venv with same Python version as Mac
python3.10 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# Install CUDA PyTorch (NOT the default CPU wheel)
pip install torch==2.8.0 torchvision --index-url https://download.pytorch.org/whl/cu121

# Install the rest of the main project deps
pip install -e .
pip install -e GIMImpN_imputation

# Install phys-GIMIN as editable
pip install -e paper12_phys_gimin --no-deps
```

#### 2.4 rsync data files (one-time, ~30 min depending on network)

```bash
# From the PC:
rsync -avz --progress <mac>:~/Projects/CSCI-FALL-2025/GIMImpN_imputation/outputs/  ./GIMImpN_imputation/outputs/
rsync -avz --progress <mac>:~/Projects/CSCI-FALL-2025/data/  ./data/
rsync -avz --progress <mac>:~/Projects/CSCI-FALL-2025/outputs/mechanistic_twin/paper10_mech_vs_giman/  ./outputs/mechanistic_twin/paper10_mech_vs_giman/
```

(Replace `<mac>` with your Mac's hostname / Tailscale name / LAN IP.)

#### 2.5 Set the env var + verify

```bash
echo "export CSCI_FALL_2025_ROOT=$PWD" >> ~/.bashrc
source ~/.bashrc

# Smoke test
python -c "from phys_gimin.utils.paths import get_project_root, get_device; \
  print('root:', get_project_root()); print('device:', get_device())"
# Expected: root: /path/to/CSCI-FALL-2025, device: cuda
```

### Running on A5000

```bash
python paper12_phys_gimin/scripts/phase1/run_smoke_benchmark.py \
  --config paper12_phys_gimin/configs/phys_gimin_lit.yaml \
  --mask-fractions 0.10 0.25 0.50 0.75 \
  --n-seeds 3 \
  --n-epochs 100 \
  --device cuda \
  --no-mock-data \
  --output-dir paper12_phys_gimin/outputs/runs/smoke_a5000_$(date +%Y%m%d_%H%M%S)
```

Expected wall time: **~3 hours for 12 runs on A5000** (vs. 5-10h on MPS).

### Ongoing sync

After finishing a run on PC, rsync results BACK to Mac:

```bash
rsync -avz paper12_phys_gimin/outputs/runs/smoke_a5000_<ts>/  <mac>:/path/to/CSCI-FALL-2025/paper12_phys_gimin/outputs/runs/
```

Then on Mac: `git add` any code changes, `git push`, and the PC can `git pull` on next run.

---

## Environment 3 — Colab Pro (A100 / V100)

### Setup in a Colab notebook

```python
# Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# Clone repo
!git clone https://github.com/bddupre92/PD_PHD.git /content/CSCI-FALL-2025
%cd /content/CSCI-FALL-2025
!git checkout feat/paper12-phys-gimin

# Install deps — Colab already has CUDA PyTorch; just install the package
!pip install -e .
!pip install -e GIMImpN_imputation
!pip install -e paper12_phys_gimin --no-deps
!pip install 'pydantic>=2.0'

# Data — expect users to sync data into /content/drive/MyDrive/CSCI_FALL_2025_data/
# or rsync from Mac once via Colab's ssh feature
import os
os.environ["CSCI_FALL_2025_ROOT"] = "/content/CSCI-FALL-2025"
```

### Run

```python
!python paper12_phys_gimin/scripts/phase1/run_smoke_benchmark.py \
  --config paper12_phys_gimin/configs/phys_gimin_lit.yaml \
  --mask-fractions 0.10 0.25 0.50 0.75 \
  --n-seeds 3 \
  --n-epochs 100 \
  --device cuda \
  --no-mock-data \
  --output-dir paper12_phys_gimin/outputs/runs/smoke_colab_$(date +%Y%m%d_%H%M%S)
```

**Colab caveat:** session timeouts at ~12h (Pro) or ~24h (Pro+). For Tier 2 (100-seed, ~50h), chunk into 3-4 sessions with checkpoint resume.

---

## Environment 4 — UND HPC cluster

### Request access

Email `ndus-it@und.edu` for a Research Computing account. Specify: A100 or A6000 GPU, ~100 GPU-hours for Paper 12 Phase 1, then again for Phase 4 (Tier 2 population study, ~50 hours).

### SLURM job script template

Save as `~/scripts/paper12_smoke.slurm`:

```bash
#!/bin/bash
#SBATCH --job-name=paper12_smoke
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

module load cuda/12.1
module load python/3.10

export CSCI_FALL_2025_ROOT=/scratch/$USER/CSCI-FALL-2025
cd $CSCI_FALL_2025_ROOT

source .venv/bin/activate

python paper12_phys_gimin/scripts/phase1/run_smoke_benchmark.py \
  --config paper12_phys_gimin/configs/phys_gimin_lit.yaml \
  --mask-fractions 0.10 0.25 0.50 0.75 \
  --n-seeds 3 \
  --n-epochs 100 \
  --device cuda \
  --no-mock-data \
  --output-dir $CSCI_FALL_2025_ROOT/paper12_phys_gimin/outputs/runs/smoke_und_$(date +%Y%m%d_%H%M%S)
```

Submit: `sbatch ~/scripts/paper12_smoke.slurm`.

Monitor: `squeue -u $USER` and tail the output files.

---

## Decision matrix — which machine when?

| Task | Recommended machine |
|---|---|
| Code writing, debugging, manuscript | Mac (always) |
| Smoke benchmark (12 runs, W4) | Any — Mac overnight works |
| Tier 1 main horse race (168 runs, W5-W12) | A5000 or Colab Pro |
| Tier 2 population study (300 runs, W13) | UND HPC or spot-A100 on Lambda/Runpod |
| Tier 3 identifiability (60 runs, W13) | A5000 or Colab Pro |
| Q2 gate script, figure generation | Mac (lightweight) |

---

## Troubleshooting

### `torch.use_deterministic_algorithms(True)` errors on CUDA

Expected on RTX A5000 for some PyG ops. The trainer sets `warn_only=True` so runs don't crash — they just log warnings. For the W13 population study we accept approximate determinism; bit-identical reproduction requires CPU.

### `ImportError: No module named 'giman_pipeline'`

Run `pip install -e /path/to/CSCI-FALL-2025 --no-deps` (the main project is editable-installed).

### `FileNotFoundError: CSCI_FALL_2025_ROOT ... is not an existing directory`

The env var is set wrong. Use `echo $CSCI_FALL_2025_ROOT` and `ls $CSCI_FALL_2025_ROOT` to diagnose.

### Postgres DB is outdated on a secondary machine

Re-run the DB restore step (2.2 above). The authoritative DB lives on the Mac; others are snapshots.
