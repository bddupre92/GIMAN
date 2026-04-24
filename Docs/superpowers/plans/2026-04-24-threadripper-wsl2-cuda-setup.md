# Threadripper + WSL2 CUDA Compute Environment — Setup & Handoff

> **For the receiving Claude instance:** this document is a self-contained plan
> to bring up a Windows 11 + WSL2 Ubuntu workstation as a CUDA compute node for
> the GIMAN dissertation project. It assumes you have not seen this codebase
> before. Work through the phases in order; each has validation criteria before
> moving to the next. After Phase 9 the workstation is ready to execute
> compute-heavy Paper 3+4 workstreams — see §After-Setup.

## Context

**Repository:** `CSCI-FALL-2025`. Mac-side active branch: `feat/ch9-6-multichannel`. Threadripper-side compute lives on a dedicated branch `feat/paper3plus4-cuda-reruns` (cut from `feat/ch9-6-multichannel`) so Threadripper CUDA output doesn't collide with Mac text edits.
**Primary user:** Blair Dupre, Department of Biomedical Engineering, University of North Dakota.
**Target hardware (confirmed):**

- **Chassis:** Fractal Define 7 XL + SYS-WS-PRO-MAX-WRX90 Threadripper WRX90 workstation
- **RAM:** 256 GB DDR5 ECC 5600 MT/s 8-channel (8 × 32 GB UDIMMs)
- **GPU:** 3 × NVIDIA RTX A5000, 24 GB GDDR6 each, 8,192 CUDA cores / 256 Tensor cores / 64 RT cores per card — **72 GB total VRAM across 3 independent CUDA devices** (check for NVLink bridges; line items hint at possible pairing hardware)
- **Storage (primary / compute):** 4 TB WD SN850X PCIe Gen4 NVMe
- **Storage (cold archive):** 3 × 12 TB Seagate Ironwolf Pro 7200 rpm HDDs = 36 TB raw (likely RAID5 or JBOD)
- **PSU + cooling:** 1200 W Platinum, 360 mm AIO liquid CPU cooler, full fan config
- **OS:** Microsoft Windows 11 Pro (3 licenses — presumably 1 active + 2 spare)
- **Warranty:** 3-year parts/labor + lifetime technical support

**Companion machine:** macOS laptop (Apple M-series). The Mac is the original development host; this setup migrates + mirrors the compute environment to the Threadripper. State moves both directions but the Threadripper becomes the authoritative host for heavy CUDA runs and for the PostgreSQL database.

### What the hardware enables

- **Parallel CUDA workstreams (3 devices):** the Paper 3+4 critical path drops from ~14 days serial to ~5-6 days parallel. For example, `CUDA_VISIBLE_DEVICES=0 python ws_p3_16_pdbp.py` in one tmux + `CUDA_VISIBLE_DEVICES=1 python ws_p3_1_inductive.py` in another + `CUDA_VISIBLE_DEVICES=2 python ws_p3_4_survtrace.py` in a third, all concurrent.
- **Memory headroom:** 192 GB allocatable to WSL2 (75% of 256 GB) covers any pandas DataFrame + multi-process DataLoader combination we'll hit. Leaves 64 GB for Windows + GPU driver overhead.
- **Fast compute filesystem:** 4 TB NVMe gives ~6-7 GB/s sequential reads — eliminates I/O as a training bottleneck.
- **Cold archive on 36 TB spinning disks:** raw PPMI/BioFIND/PDBP/HBS data + historical snapshots of the Google Drive mirror can live there indefinitely without eating the NVMe.

**What this setup enables:**

- Execute Paper 3+4 (npj Digital Medicine) revision workstreams that require CUDA
  (WS-P3-1 inductive graph retrain, WS-P3-4 SurvTRACE+SurvLatent-ODE+CRISP-NAM baselines,
  WS-P3-5 GraphMAE, WS-P3-13 5-seed variance, WS-P3-16 PDBP external validation).
- Paper 11 SciML UDE residual experiments (from `Docs/NEXT_STEPS_2026-04-22.md`).
- Any future CUDA-native workstream at up-to-24-GB VRAM budgets.

**What this setup explicitly does NOT do:**

- Replace the Mac for prose writing, figure generation, or MPS-suitable compute.
- Provide a shared-database active-active setup; PostgreSQL is single-authoritative (hosted on WSL2).
- Automate sync of raw medical data via the cloud — raw data stays on-disk on each machine; only small artifacts flow through git + Google Drive.

## Key reference documents

Read these in order after finishing setup:

1. **`CLAUDE.md`** (repo root) — project overview, conventions, PostgreSQL schema registry. The "Local Research Database" section at line 17 is the source of truth for what Postgres should look like. The current correct state is **741 MB, 193 tables across 14 schemas** (verified 2026-04-24).

2. **`Docs/superpowers/plans/2026-04-23-paper3plus4-reviewer-response-execution.md`** — the Paper 3+4 reviewer-response execution plan, 19 workstreams in 4 phases (S/R/V/W). Most remaining work lives here.

3. **`Docs/NEXT_STEPS_2026-04-23.md`** — current session resume anchor with full commit arc and priority list.

4. **`Docs/superpowers/plans/2026-04-23-paper1-R2-reviewer-response.md`** — Paper 1 round-2 plan (if any cross-paper work is needed).

5. **`src/giman_pipeline/data/db.py`** — the Postgres helper (`get_engine()`, `read_sql()`, `read_table()`). Verify it connects after Phase 7.

6. **`~/.claude/projects/<project-dir>/memory/MEMORY.md`** — persisted conversation memory. Read this after Phase 8 to inherit project context.

## Phase 0: Pre-flight inventory (already decided + residual inputs)

The user has confirmed the following strategic decisions:

- ✅ **Postgres authority:** single-authoritative PostgreSQL hosted on WSL2. Mac tunnels in via Tailscale.
- ✅ **Branching:** `feat/paper3plus4-cuda-reruns` branch cut from `feat/ch9-6-multichannel` on Threadripper. Clean separation from Mac text edits.
- ✅ **Tailscale location:** inside WSL2 (not Windows) — so `ssh threadripper` from Mac lands directly in the Linux shell.
- ✅ **WSL RAM cap:** 192 GB (of 256 GB total system RAM).
- ✅ **Hardware:** 3 × A5000 GPUs, 4 TB NVMe primary, 36 TB HDD cold archive.

Still needs to be captured once Threadripper is booted:

- [ ] **Exact Threadripper CPU SKU** — `lscpu` inside WSL (tells us core count for WSL `processors=` cap).
- [ ] **Windows version** — run `winver` in PowerShell. Target ≥ 22H2.
- [ ] **Current NVIDIA Windows driver version** — `nvidia-smi` in Windows PowerShell. Target ≥ 535 for CUDA 12.1+ support. Update via GeForce Experience or manual download if older.
- [ ] **Existing WSL2 installation status** — `wsl --version` and `wsl -l -v` in PowerShell. Need WSL 2.x with Ubuntu 22.04 or 24.04 LTS.
- [ ] **NVLink bridges present between A5000 cards?** — check with `nvidia-smi topo -m` inside WSL (shows NVLink in the inter-GPU matrix). If present, pairs give combined 48 GB VRAM pool; if absent, each GPU is independent.
- [ ] **HDD mount plan:** RAID5 across the 3 × 12 TB disks for ~24 TB redundant capacity, or JBOD for 36 TB? RAID5 recommended (one-drive-failure tolerance with only 33% capacity overhead).
- [ ] **NVMe partition plan:** whole 4 TB for WSL's ext4 home, or carve a Windows data partition first? Recommend: leave 200-300 GB for Windows, dedicate the remaining ~3.7 TB to WSL2 via `wsl --install --location`.
- [ ] **Repo access method** — HTTPS with token, or SSH with existing key? Clone URL?

## Phase 1: Windows-side prep (15 min, one reboot)

### 1.1 Update WSL core and the Windows side

```powershell
# PowerShell as Administrator
wsl --update
wsl --shutdown
```

### 1.2 Install / update NVIDIA driver on Windows

Go to <https://www.nvidia.com/Download/index.aspx>, select RTX A5000 + Windows 11, install. **Do NOT install a Linux NVIDIA driver inside WSL.** CUDA in WSL2 uses the Windows driver via PCI passthrough (`/usr/lib/wsl/lib/libcuda.so.1`).

### 1.3 Create `.wslconfig` with resource caps (tuned for 256 GB / Threadripper)

```powershell
# %USERPROFILE%\.wslconfig  (create if absent)
# Sized for 256 GB RAM, multi-GPU Threadripper
Set-Content -Path "$env:USERPROFILE\.wslconfig" -Value @"
[wsl2]
memory=192GB
processors=0
vmIdleTimeout=-1
swap=32GB
localhostForwarding=true
nestedVirtualization=true
"@
```

`processors=0` lets WSL2 use all available logical cores (adjust down to leave Windows headroom if the CPU SKU has <32 threads — for Threadripper Pro 7965WX / 7985WX / 7995WX this is fine as-is). `vmIdleTimeout=-1` is the critical setting — without it WSL2 shuts down ~60 s after the last process exits, which kills tmux/postgres/tailscale. `nestedVirtualization=true` enables Docker-in-WSL if needed later.

### 1.4 Reboot (or `wsl --shutdown` then re-enter)

## Phase 2: WSL2 init + systemd (20 min)

### 2.1 Enter Ubuntu WSL, verify version

```bash
lsb_release -a    # want Ubuntu 22.04 or 24.04
uname -r          # WSL kernel, should be ≥ 5.15
```

### 2.2 Enable systemd

```bash
sudo tee /etc/wsl.conf > /dev/null <<'EOF'
[boot]
systemd=true

[network]
generateResolvConf=true

[interop]
enabled=true
appendWindowsPath=false
EOF
```

### 2.3 Restart WSL from Windows

```powershell
wsl --shutdown
wsl
```

### 2.4 Verify systemd active

```bash
systemctl --version
ps -p 1 -o comm=   # should print: systemd
```

**Validation criterion:** `ps -p 1 -o comm=` returns `systemd`.

## Phase 3: Multi-GPU verification inside WSL2 (10 min)

```bash
# Should list all 3 A5000 cards
nvidia-smi

# Topology: confirms NVLink presence (or absence)
nvidia-smi topo -m
# Legend: NV# = NVLink at gen #; PIX / PHB = PCIe hops; SYS = NUMA traversal
# A5000 NVLink bridge (if present) shows as "NV1" between paired GPUs.

# Sanity — WSL CUDA shim
ls -l /usr/lib/wsl/lib/libcuda.so.1

# Multi-GPU PyTorch check (run after Phase 5 venv exists, but the CUDA runtime itself
# should be visible immediately):
python3 -c "
import subprocess
r = subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.total,driver_version',
                    '--format=csv,noheader'], capture_output=True, text=True)
print(r.stdout.strip())
# Expected three lines, one per GPU index 0, 1, 2:
#   0, NVIDIA RTX A5000, 24564 MiB, 550.xx
#   1, NVIDIA RTX A5000, 24564 MiB, 550.xx
#   2, NVIDIA RTX A5000, 24564 MiB, 550.xx
"
```

**Validation criteria:**

- `nvidia-smi` lists all 3 A5000 cards without error
- `nvidia-smi topo -m` reveals whether NVLink is wired between any pair (look for `NV1` / `NV2` entries; `PIX` / `PHB` mean no NVLink, just PCIe)
- Driver version ≥ 535 on all 3 devices

If `nvidia-smi` fails or shows fewer than 3 cards, the problem is the **Windows driver**, not anything in WSL — update the driver via nvidia.com and retry. If one card shows `ERR` or missing PCIe, physical reseat may be required (rare on first install).

### Per-workstream GPU pinning pattern

Set `CUDA_VISIBLE_DEVICES` to dedicate one GPU per workstream so runs don't contend. Convention for Paper 3+4 critical path:

| Workstream | GPU | Rationale |
|---|---|---|
| WS-P3-16 PDBP external validation | `cuda:0` | Longest runtime (5-6 d), gets the first GPU |
| WS-P3-1 Inductive graph retrain | `cuda:1` | 4 d GAT training |
| WS-P3-4 SurvTRACE/SurvLatent-ODE/CRISP-NAM | `cuda:2` | 4-5 d, three vendored architectures run sequentially on one GPU |
| WS-P3-5 GraphMAE | `cuda:0` (after P3-16 done) | Uses NVLink pair if available for larger batch |
| WS-P3-13 5-seed variance | `cuda:1` (after P3-1 done) | Retrains DeepHit + Graph-DT 25× |

Launch pattern:

```bash
tmux new -s ws_p3_16
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/paper3plus4/run_pdbp_external_validation.py \
  2>&1 | tee logs/ws_p3_16_$(date +%Y%m%d_%H%M).log
```

## Phase 4: Base packages + SSH + Tailscale (20 min)

### 4.1 Base packages

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y \
  build-essential git curl wget htop tmux rsync unzip jq \
  openssh-server \
  postgresql-17 postgresql-client-17 \
  rclone
```

### 4.2 Enable SSH server

```bash
sudo systemctl enable --now ssh
sudo ufw allow 22/tcp || true
```

### 4.3 Install Tailscale inside WSL

```bash
curl -fsSL https://tailscale.com/install.sh | sh
sudo systemctl enable --now tailscaled
sudo tailscale up --ssh --accept-routes
```

Copy the Tailscale hostname it prints; it looks like `<something>.tail-scale.ts.net` or similar. Give it to the user.

### 4.4 On the Mac, add SSH config

```
# On the Mac, append to ~/.ssh/config
Host threadripper
  HostName <tailscale-hostname>.tail-scale.ts.net
  User <wsl-linux-username>
  ServerAliveInterval 30
  ServerAliveCountMax 3
```

### 4.5 Install SSH key from Mac

```bash
# On Mac
ssh-keygen -t ed25519 -f ~/.ssh/threadripper_ed25519
ssh-copy-id -i ~/.ssh/threadripper_ed25519 threadripper
# Then ssh threadripper should land in WSL2 Linux shell without password
```

**Validation criterion:** `ssh threadripper 'uname -a'` from the Mac returns the WSL kernel version.

## Phase 5: Dev environment (30 min)

### 5.1 Python + uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc   # or reopen shell
uv --version       # ≥ 0.3
```

### 5.2 Clone the repo

```bash
mkdir -p ~/Projects
cd ~/Projects
# Use HTTPS with a GitHub token, or SSH if key is registered
git clone https://github.com/bddupre92/PD_PHD.git CSCI-FALL-2025
cd CSCI-FALL-2025
git checkout feat/ch9-6-multichannel
git pull --rebase
```

### 5.3 Create virtualenv with CUDA PyTorch

```bash
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install -e ".[dev]" 2>/dev/null || uv pip install -e .
# CUDA-enabled torch (pick CUDA 12.1 wheels; A5000 supports ≤ 12.x)
uv pip install torch==2.8.0 torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 5.4 Verify CUDA pipeline

```bash
python -c "
import torch
assert torch.cuda.is_available(), 'CUDA not available inside venv'
print('CUDA:', torch.cuda.is_available())
print('Device:', torch.cuda.get_device_name(0))
print('CUDA version (torch):', torch.version.cuda)
x = torch.randn(2048, 2048, device='cuda')
y = x @ x.T
print('matmul sum:', y.sum().item())
"
```

**Validation criterion:** all three lines print, matmul runs in < 2 seconds, no `RuntimeError`.

## Phase 6: Google Drive sync via rclone (15 min + transfer time)

```bash
rclone config   # interactive: n (new remote) → name it "gdrive" → google drive → OAuth in browser (Windows side) → accept
# Test:
rclone lsd gdrive:
```

User plan: the Mac pushes raw data to a shared Google Drive folder; Threadripper pulls via rclone.

```bash
# Example syncs — adapt to actual gdrive folder structure
mkdir -p ~/data
rclone sync gdrive:"CSCI FALL 2025/data/00_raw" ~/data/00_raw \
  --fast-list --transfers 8 --checkers 16 --progress
rclone sync gdrive:"CSCI FALL 2025/data/05_features" ~/data/05_features \
  --fast-list --transfers 8 --checkers 16 --progress
```

Symlink the repo's `data/` directory to this path if you want the code to pick it up transparently:

```bash
cd ~/Projects/CSCI-FALL-2025
# Move any existing empty ./data out of the way first
test -d data && test -z "$(ls data)" && rmdir data
ln -s ~/data data
ls -la data   # should show symlink
```

Alternative: initial one-shot pull directly from Mac over Tailscale (faster for 15 GB first load):

```bash
# Run on Mac
rsync -avP --exclude='.venv' --exclude='*.pyc' \
  ~/Projects/CSCI-FALL-2025/data/ \
  threadripper:/home/<user>/data/
```

**Validation criterion:** `ls ~/data/05_features/paper1_features_with_targets.csv`
and `wc -l` returns 2,202 (2,201 rows + header).

## Phase 7: PostgreSQL migration (30 min)

WSL2 hosts the authoritative Postgres. Mac tunnels in when needed.

### 7.1 Create user + DB on WSL2

```bash
sudo systemctl enable --now postgresql
sudo -u postgres createuser --createdb --superuser $USER
createdb giman_research
```

### 7.2 Dump on Mac, restore in WSL2

```bash
# On Mac
pg_dump -Fc giman_research > ~/giman_research_$(date +%Y%m%d).dump
rsync -avP ~/giman_research_*.dump threadripper:~/
```

```bash
# On WSL2
pg_restore -d giman_research ~/giman_research_*.dump
```

### 7.3 Verify schema count matches CLAUDE.md

```bash
psql giman_research -Atc "
SELECT pg_size_pretty(pg_database_size('giman_research')) || ' · ' ||
  (SELECT COUNT(*) FROM pg_tables WHERE schemaname NOT IN ('pg_catalog','information_schema')) || ' tables · ' ||
  (SELECT COUNT(DISTINCT schemaname) FROM pg_tables WHERE schemaname NOT IN ('pg_catalog','information_schema','public')) || ' user schemas'
"
# Expected (as of 2026-04-24): 741 MB · 193 tables · 14 user schemas
```

### 7.4 Verify `giman_pipeline` can read

```bash
cd ~/Projects/CSCI-FALL-2025
.venv/bin/python -c "
from giman_pipeline.data.db import read_sql, read_table
df = read_table('features', 'paper1_features_with_targets')
print('paper1_features rows:', len(df))
# Expected: 2201
"
```

**Validation criterion:** 741 MB × 193 tables × 14 schemas; paper1_features_with_targets has 2,201 rows.

### 7.5 (Optional) Configure Mac→WSL tunnel for Mac-side queries

```bash
# On Mac, add to ~/.ssh/config
# Host threadripper
#   ...existing lines...
#   LocalForward 55432 localhost:5432
```

Then on Mac: `psql -h localhost -p 55432 giman_research` queries the WSL-hosted DB.

## Phase 8: Claude Code + Mempalace state port (30 min)

### 8.1 Install Claude Code on WSL2

```bash
# Per the latest Anthropic instructions — check claude.ai for the current installer
# Usually:
curl -fsSL https://claude.ai/install.sh | bash
claude --version
```

### 8.2 Port Claude Code settings + skills from Mac

```bash
# On Mac
rsync -avP ~/.claude/settings.json threadripper:~/.claude/
rsync -avP ~/.claude/skills/ threadripper:~/.claude/skills/
rsync -avP ~/.claude/plugins/ threadripper:~/.claude/plugins/

# Chat history for this project (note path name translation Mac → Linux):
rsync -avP ~/.claude/projects/-Users-blair-dupre-Projects-CSCI-FALL-2025/ \
  threadripper:~/.claude/projects/-home-<user>-Projects-CSCI-FALL-2025/
# Replace <user> with the WSL2 Linux username
```

### 8.3 Port mempalace

```bash
# On Mac
tar czf mempalace.tgz -C ~ Projects/.mempalace .mempalace
rsync -avP mempalace.tgz threadripper:~/

# On WSL2
tar xzf ~/mempalace.tgz -C ~/
# Verify:
~/.local/bin/mempalace status
# Expected: drawer + wing counts matching Mac's palace
```

### 8.4 Expected path-mismatch warning

Memory drawers contain absolute Mac paths like `/Users/blair.dupre/…`. These are
**annotations, not live links** — treat them as historical provenance. Do not
auto-rewrite. New mempalace entries written on Threadripper will use
`/home/<user>/…` paths; mixed is fine.

### 8.5 Read inherited project memory

```bash
cat ~/.claude/projects/-home-<user>-Projects-CSCI-FALL-2025/memory/MEMORY.md
```

That index points at project-specific memory files. Read the most relevant ones
before starting work. Particularly:

- `paper1_r1_r2_arc.md` — Paper 1 rounds 1+2 summary.
- `session_2026_04_18_robustness_workstreams.md` — cross-paper W1-W4 robustness status.
- `paper_arc_p10_p13_strategic.md` — Paper 10-13 strategic plan.

**Validation criterion:** `mempalace status` shows ≥ 10k drawers, `ls ~/.claude/projects/` shows the project directory, `MEMORY.md` read successfully.

## Phase 9: End-to-end validation — WS-P3-6 Markov (30-60 min)

WS-P3-6 is a pure-CPU Paper 3+4 workstream that's also the unblocker for
WS-P3-S3 (Table II Markov predictive metrics row in the main paper). Running
it as the first real workstream validates the full pipeline (data → Postgres →
code → artifact) before we spend 5-6 days on CUDA-heavy work.

```bash
cd ~/Projects/CSCI-FALL-2025
tmux new -s p3_6_markov

# Inside tmux:
.venv/bin/python scripts/paper3/run_multistate_model.py 2>&1 | tee /tmp/p3_6.log

# Ctrl-B D to detach. Reattach with: tmux attach -t p3_6_markov
```

**Validation criterion:** script produces `outputs/paper3_markov/markov_results.json`
with Q matrix + sojourn times. Compare against the existing
`outputs/paper3_markov/markov_results.json` on the Mac — numbers should match
within CV-noise tolerance (typically < 0.001 per entry).

Once this passes, the workstation is ready for CUDA-heavy work.

## After-Setup — parallel workstream launch (3 GPUs)

Open `Docs/superpowers/plans/2026-04-23-paper3plus4-reviewer-response-execution.md`
for each workstream's pre-registered decision rules, output paths, and
acceptance criteria. Launch the critical-path trio in parallel across the 3 A5000s:

### Day 0-6 (parallel on 3 GPUs)

| GPU | tmux session | Workstream | Effort | Script |
|---|---|---|---|---|
| `cuda:0` | `ws_p3_16` | **WS-P3-16 PDBP external validation** | 5-6 d | `scripts/paper3plus4/run_pdbp_external_validation.py` |
| `cuda:1` | `ws_p3_1` | WS-P3-1 Inductive graph retrain | 4 d | `scripts/paper3plus4/run_inductive_graph_retrain.py` |
| `cuda:2` | `ws_p3_4` | WS-P3-4 SurvTRACE + SurvLatent-ODE + CRISP-NAM baselines | 4-5 d | 3 scripts under `scripts/paper3plus4/` |

**Launch command (repeat for each with appropriate CUDA_VISIBLE_DEVICES):**

```bash
tmux new -s ws_p3_16
cd ~/Projects/CSCI-FALL-2025
mkdir -p logs
CUDA_VISIBLE_DEVICES=0 .venv/bin/python \
  scripts/paper3plus4/run_pdbp_external_validation.py 2>&1 \
  | tee logs/ws_p3_16_$(date +%Y%m%d_%H%M).log
# Ctrl-B D to detach. `tmux attach -t ws_p3_16` to reattach.
# `tmux ls` to see all active sessions.
```

### Day 6-10 (second wave after first wave completes)

| GPU | Workstream | Effort |
|---|---|---|
| `cuda:0` (freed from P3-16) | WS-P3-5 GraphMAE pre-training | 3-4 d |
| `cuda:1` (freed from P3-1) | WS-P3-13 5-seed variance (DeepHit + Graph-DT + new baselines × 5 seeds) | 2 d |
| `cuda:2` (freed from P3-4) | WS-P3-7c Fine-Gray + dynamic landmarking + jmstate (MPS-suitable but can use CUDA) | 3-4 d |

### Day 10-14: MPS-suitable / CPU workstreams in parallel with whatever is still running

| Runs well on | Workstream | Effort |
|---|---|---|
| Mac MPS | WS-P3-2 Subject-clustered bootstrap | 1-2 d |
| CPU inside WSL | WS-P3-6 Markov predictive metrics (unblocks Table II S-3) | 0.5 d |
| CPU inside WSL | WS-P3-8 Brier decomp + DCA + reliability diagrams | 1-2 d |
| CPU inside WSL | WS-P3-9 5-dim ablation grid (60 configs) | 2-3 d |
| CPU inside WSL | WS-P3-10 HSMM misclassification HMM | 2-3 d |
| Mac | WS-P3-15 Faithfulness metrics | 1-2 d |
| Mac prose | WS-P3-17 Imputation strategy disclosure | 0.5 d |

**Total critical path with parallelism:** ~14 days wall-clock vs. ~40 days serial.

### Monitoring pattern

From **any** machine (Mac, phone, laptop on the road) via Tailscale:

```bash
ssh threadripper 'tmux ls'                # see all sessions
ssh threadripper 'tail -n 50 ~/Projects/CSCI-FALL-2025/logs/ws_p3_16_*.log'
ssh threadripper 'nvidia-smi'              # GPU load per device
```

VS Code Remote-SSH gives a richer experience — the tree + terminal + log tails all stream natively. TensorBoard (if used) auto-forwards from WSL:6006 to Windows:6006 to the Mac browser via SSH port forward:

```bash
# From Mac
ssh -L 6006:localhost:6006 threadripper &
# Then open http://localhost:6006 on the Mac browser
```

## Ongoing sync strategy — both directions

| Artifact | Direction | Mechanism | Frequency |
|---|---|---|---|
| Source code + scripts | Mac ↔ WSL2 | git push/pull via GitHub | per commit |
| Small JSON results (< 10 MB) | Mac ↔ WSL2 | committed to git under `outputs/paper*_r2_responses/` etc. | per experiment |
| Large model checkpoints (> 100 MB) | WSL2 → Mac | git LFS or manual rsync over Tailscale | per major run |
| Raw medical data (PPMI/BioFIND/PDBP/HBS) | Mac → gdrive → WSL2 | rclone sync | initial + occasional |
| PostgreSQL state | WSL2 authoritative | pg_dump + restore, or Mac SSH tunnel to WSL2 | WSL2 is always the source of truth after migration |
| Mempalace | Mac ↔ WSL2 | accept divergent; manually merge if both sides add a lot | rarely |
| Claude Code chat history | Mac ↔ WSL2 | per-project rsync; keep both as local history | end of each cross-machine session |

Rule of thumb: **git is the backbone for everything < 100 MB.** Everything
bigger flows through rsync/rclone and is NOT committed.

## Known gotchas — WSL2 specific

| Gotcha | Fix |
|---|---|
| WSL auto-suspends, killing tmux sessions | `.wslconfig` → `vmIdleTimeout=-1` + at least one persistent systemd service running (e.g., `tailscaled`, `postgresql`) |
| `/mnt/c/...` I/O is glacial | Keep everything compute-heavy under `~/` (ext4). Never run training loops off `/mnt/c`. |
| `nvidia-smi` missing inside WSL | Update the **Windows** NVIDIA driver, not anything in Linux. `/usr/lib/wsl/lib/libcuda.so.1` comes from the Windows driver. |
| WSL2 eats system RAM | `.wslconfig` → `memory=XGB` (cap to ~50-75% of total). |
| Postgres not reachable after WSL restart | `sudo systemctl enable postgresql` ensures it restarts with WSL. |
| SSH from Mac keeps disconnecting | Both Mac and WSL should use Tailscale ≥ 1.60. Mac `~/.ssh/config` `ServerAliveInterval 30`. |
| Git line endings (CRLF vs LF) | On WSL: `git config --global core.autocrlf input`. Never edit files from Windows-side tools that rewrite line endings. |
| GPU VRAM exhausted by a prior Python process after crash | `nvidia-smi` → find stuck PID → `kill -9`. If WSL VRAM state gets corrupt: `wsl --shutdown` + restart. |
| TensorBoard port collision with Windows apps | Forward a non-default port: `tensorboard --port=6007 --bind_all` then access from Windows browser at `localhost:6007`. |

## Rollback / recovery

If the WSL environment gets into a bad state (corrupt venv, broken apt, whatever):

```powershell
# From Windows PowerShell — exports the distro first
wsl --export Ubuntu C:\wsl-backups\ubuntu-backup-$(Get-Date -Format 'yyyyMMdd').tar
# Then if recovery needed:
wsl --unregister Ubuntu
wsl --import Ubuntu C:\WSL\Ubuntu C:\wsl-backups\ubuntu-backup-*.tar --version 2
```

Keep at least one weekly backup after the initial setup is validated.

## For the Claude instance receiving this doc

Your job on first run:

1. Greet the user, confirm you're on Threadripper (Linux uname check).
2. Ask Blair for the Phase 0 pre-flight inputs (RAM, driver version, WSL distro).
3. Walk through Phase 1-9 sequentially. **Do not skip validation criteria.**
4. After Phase 9 passes, read `MEMORY.md` + the 3 most relevant memory files.
5. Ask Blair which workstream to start first (recommend WS-P3-16).
6. Set up the first `tmux` session and kick off compute.

Before each `sudo apt install`, `wsl --shutdown`, or `pg_restore`, state what
you're about to do and get user confirmation. These are system-level changes
that the Windows/WSL2 owner must authorize.

**Do NOT:**

- Install a Linux NVIDIA driver inside WSL (breaks GPU passthrough).
- Move data to `/mnt/c/` for training (catastrophic I/O perf).
- Commit large model checkpoints or raw medical data to git (use rsync/rclone).
- Assume Mac paths work — the project dir is `/home/<user>/Projects/CSCI-FALL-2025`, not `/Users/blair.dupre/…`.
- Run `scripts/vault_sync.py` or `scripts/defense_prep/*.py` on Threadripper
  until you've validated the full Postgres + audit.claim + mempalace chain —
  those scripts assume Mac-specific paths in several places.

## Contact

Blair Dupre · <blair.dupre@und.edu> · Dept. of Biomedical Engineering, UND.

Session-origin Claude: running on macOS (Apple M-series) at
`/Users/blair.dupre/Projects/CSCI-FALL-2025` on branch `feat/ch9-6-multichannel`
as of commit `68b912c` (2026-04-24).
