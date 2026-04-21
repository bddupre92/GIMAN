# Paper 12 W6 — Resume Guide (2026-04-21 late evening)

**Session closed:** 2026-04-21, FastSurfer batch LAUNCHED on Lightning academic account `blairdupre`, 64-core CPU tier.

**Branch:** `feat/paper12-phys-gimin` · **Worktree:** `~/.config/superpowers/worktrees/CSCI-FALL-2025/feat-paper12-phys-gimin` · **Latest pushed:** `b285a12`

---

## Resume command

```
"Continue Paper 12 W6 — FastSurfer batch running on Lightning Studio paper12-fastsurfer (blairdupre/default-teamspace, 64-core CPU). Read paper12_phys_gimin/docs/NEXT_STEPS_POST_COMPACT.md and check /tmp/fastsurfer.log status on the Studio via SDK."
```

---

## Current live state

| Item | Value |
|---|---|
| Lightning account | `blairdupre` (academic, UND) |
| Teamspace | `default-teamspace` |
| Studio | `paper12-fastsurfer` |
| Studio ID | `01kps6rns0a9wsbnccrnkkzy20` |
| Compute | **CPU_X_64** (upgrade in progress, ~5 min) |
| Expected runtime | ~6.5 hrs for 401 scans |
| Parallelism | 16 concurrent FastSurfer processes × 4 threads each |
| Pipeline | FULL (DL seg + surface recon → aseg+DKT.stats + thickness) |

## Files in Studio root (`/teamspace/studios/this_studio/`)

- ✅ `nifti_wang_n161.tar.gz` (5.0 GB)
- ✅ `license.txt` (102 B)
- ✅ `lightning_cpu_parallel_fastsurfer.sh` (4.8 KB)

## Credentials state (Mac)

- `~/.lightning/credentials.json` — set to `blairdupre` UUID (a844d035-c12...)
- `~/.lightning/lightning_rsa` — regenerated for new account, SCP/SSH working
- `paper12_phys_gimin/data/.env` — LIGHTNING_USER_ID/API_KEY/USERNAME/TEAMSPACE populated

## Auto-sleep concern

Lightning Studio defaults to "auto sleep after 10 mins of inactivity." FastSurfer batch should keep the Studio active (16 parallel CPU workers), but to be safe, keep-alive heartbeat runs alongside the batch (writes to `/tmp/keepalive.log` every 5 min).

---

## Launch sequence (if not already done)

Once 64-core CPU is ready:

```bash
/Users/blair.dupre/Projects/CSCI-FALL-2025/.venv/bin/lightning studio ssh --name paper12-fastsurfer --teamspace blairdupre/default-teamspace
```

Inside Studio:

```bash
cd /teamspace/studios/this_studio

# Step 1: Install FreeSurfer + FastSurfer + deps (~5 min)
bash lightning_cpu_parallel_fastsurfer.sh install

# Step 2: Start keep-alive heartbeat
nohup bash -c 'while true; do date >> /tmp/keepalive.log; sleep 300; done' > /dev/null 2>&1 &
echo $! > /tmp/keepalive.pid

# Step 3: Launch FastSurfer batch detached
nohup bash lightning_cpu_parallel_fastsurfer.sh process > /tmp/fastsurfer.log 2>&1 &
echo $! > /tmp/fastsurfer.pid

# Step 4: Verify both alive
sleep 30 && ps -p $(cat /tmp/fastsurfer.pid) && echo "--- log tail ---" && tail -15 /tmp/fastsurfer.log
```

Exit SSH. Batch runs ~6.5 hrs.

---

## Monitoring (from Mac after reopening Claude)

Query Studio from local Mac:

```bash
set -a; source /Users/blair.dupre/.config/superpowers/worktrees/CSCI-FALL-2025/feat-paper12-phys-gimin/paper12_phys_gimin/data/.env; set +a

/Users/blair.dupre/Projects/CSCI-FALL-2025/.venv/bin/python -c "
import os, json
c = json.load(open(os.path.expanduser('~/.lightning/credentials.json')))
os.environ['LIGHTNING_USER_ID'] = c['user_id']; os.environ['LIGHTNING_API_KEY'] = c['api_key']
from lightning_sdk import Studio
s = Studio('paper12-fastsurfer', teamspace='default-teamspace', user='blairdupre', create_ok=False)
out = s.run('''
echo \"OK count:\"; grep -c \"\\[OK\\]\" /tmp/fastsurfer.log
echo \"START count:\"; grep -c \"\\[START\\]\" /tmp/fastsurfer.log
echo \"keep-alive alive:\"; ps -p \$(cat /tmp/keepalive.pid) >/dev/null && echo YES || echo NO
echo \"--- last 8 lines ---\"; tail -8 /tmp/fastsurfer.log
''')
print(out)
"
```

## When batch completes

Inside Studio (or via SDK run):

```bash
# Package results
cd /teamspace/studios/fastsurfer_out
find . \( -path '*/stats/*' -o -name 'aparc*.mgz' \) -type f | tar -czf /teamspace/studios/this_studio/fastsurfer_full_stats.tar.gz -T -
du -sh /teamspace/studios/this_studio/fastsurfer_full_stats.tar.gz
```

Download tarball back to Mac:

```bash
# Same scp pattern as earlier uploads
scp -i ~/.lightning/lightning_rsa \
  s_01kps6rns0a9wsbnccrnkkzy20@ssh.lightning.ai:/teamspace/studios/this_studio/fastsurfer_full_stats.tar.gz \
  /Users/blair.dupre/Projects/CSCI-FALL-2025/data/02_fastsurfer_out.tar.gz
```

Extract + load to PG (next session Step 8).

## Stop Studio after completion (halt billing)

```bash
/Users/blair.dupre/Projects/CSCI-FALL-2025/.venv/bin/python -c "
import os, json
c = json.load(open(os.path.expanduser('~/.lightning/credentials.json')))
os.environ['LIGHTNING_USER_ID'] = c['user_id']; os.environ['LIGHTNING_API_KEY'] = c['api_key']
from lightning_sdk import Studio
s = Studio('paper12-fastsurfer', teamspace='default-teamspace', user='blairdupre', create_ok=False)
s.stop(); print('Stopped:', s.status)
"
```

---

## What's still pending (post-batch)

- **W6 Step 6:** Fidelity gate eval on real FastSurfer features (volumes + thickness) against Wang 2025 [RMSE 0.145-0.177, R² 0.743-0.909]
- **W6 Step 8:** SQL load (`mechanistic.paper12_competitor_fidelity` + `paper12_wang_features`) + schema count bump
- **W6 Step 9:** Drive archive + local cleanup
- **W7a:** Demirkaya 2021 CKF clean-room
- **W7b:** Zou 2025 MNODE-HGS clean-room
- **W8:** LagCNN clean-room

---

## Cross-device chat continuity reminder

Claude Code chat history is LOCAL per device. What persists across Mac/Windows/etc:

- ✅ This doc + `CLAUDE.md` + all committed code (pulled via `git pull`)
- ✅ Lightning credentials in `~/.lightning/credentials.json` (machine-local, must re-setup on each device)
- ❌ Chat transcripts (don't sync)
- ❌ Memory files at `~/.claude/projects/.../memory/` (Mac-local)

**On a new device:** clone repo, paste the resume command above, Claude reads CLAUDE.md + this file automatically → full context.

## Fallback plan if Lightning batch fails overnight

1. **UND HPC Talon** ticket submitted earlier — if access lands, same script pattern, 72 cores, free
2. **User's Threadripper WRX90 + RTX A5000** desktop — WSL2+Docker, script template ready in `scripts/lightning_cpu_parallel_fastsurfer.sh` (small adaptations for Windows)
3. **AWS c5.24xlarge** — $16, ~4 hrs, SSH-based, clean cloud flow
