"""Paper 12 W6 Step 4b — Lightning AI SDK driver for FreeSurfer concordance anchor.

Orchestrates vanilla FreeSurfer recon-all on 20 random Wang T1 scans via
Lightning AI Studio on CPU_X_32. Gives us in-cohort FastSurfer-vs-FreeSurfer
concordance (Dice + cortical-thickness correlation) for the Methods section.

Runs SEQUENTIALLY after lightning_driver.py completes (free plan = 1 active
Studio at a time). Uses a separate Studio name so the two jobs don't collide.

Prerequisites (local Mac):
  pip install lightning-sdk
  export LIGHTNING_USER_ID=...
  export LIGHTNING_API_KEY=...

Required local files:
  /tmp/nifti_freesurfer_anchor.tar.gz         (20 scans · 307 MB)
  paper12_phys_gimin/data/license.txt         (FreeSurfer license)
  scripts/lightning_freesurfer_anchor.sh

Timing:
  CPU_X_32, 4 parallel scans with -openmp 8, ~4 hrs per scan, 20 scans = ~20 hrs

Cost (pay-as-you-go on free plan, after credits):
  CPU_X_32 ≈ $1-1.50/hr × ~20 hrs ≈ $20-30 (subtract ~$15 monthly credits ≈ $5-15)

Usage:
  python scripts/lightning_anchor_driver.py
  python scripts/lightning_anchor_driver.py --install-only   # debug
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

def _load_dotenv() -> None:
    """Load paper12_phys_gimin/data/.env into os.environ if present (before any SDK import)."""
    env_path = Path(__file__).resolve().parents[1] / "paper12_phys_gimin/data/.env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip("'\"")
        if k and v and k not in os.environ:
            os.environ[k] = v


_load_dotenv()

try:
    from lightning_sdk import Machine, Studio
except ImportError:
    sys.stderr.write("ERROR: lightning-sdk not installed. Run: pip install lightning-sdk\n")
    sys.exit(1)

STUDIO_NAME = "paper12-fs-anchor"
WORKTREE = Path(__file__).resolve().parents[1]

LOCAL_NIFTI_TAR = Path("/tmp/nifti_freesurfer_anchor.tar.gz")
LOCAL_LICENSE = WORKTREE / "paper12_phys_gimin/data/license.txt"
LOCAL_SETUP_SH = WORKTREE / "scripts/lightning_freesurfer_anchor.sh"
LOCAL_RESULTS = WORKTREE / "paper12_phys_gimin/data/freesurfer_anchor_stats.tar.gz"

# 30 hr hard cap accommodates slow scans + buffer, ~$45 worst case
CPU_HARD_TIMEOUT_S = 30 * 3600


def check_auth() -> None:
    env_set = bool(os.environ.get("LIGHTNING_USER_ID") and os.environ.get("LIGHTNING_API_KEY"))
    creds_file = Path.home() / ".lightning" / "credentials"
    if not env_set and not creds_file.exists():
        sys.stderr.write(
            "ERROR: no Lightning credentials. Run `lightning login` in Terminal, "
            "or export LIGHTNING_USER_ID + LIGHTNING_API_KEY.\n"
        )
        sys.exit(1)
    print(f"[auth] Using {'env vars' if env_set else 'cached credentials'}")


def check_local_files(install_only: bool) -> None:
    required = [LOCAL_SETUP_SH]
    if not install_only:
        required += [LOCAL_NIFTI_TAR, LOCAL_LICENSE]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        sys.stderr.write("ERROR: missing local files:\n" + "\n".join(f"  - {m}" for m in missing) + "\n")
        sys.exit(1)


def upload_if_missing(studio: Studio, local: Path, remote_name: str) -> None:
    ls = studio.run(f"ls -la {remote_name} 2>/dev/null || echo MISSING")
    if "MISSING" in ls:
        print(f"[upload] {local.name} ({local.stat().st_size / 1e6:.1f} MB) -> {remote_name}")
        studio.upload_file(str(local), remote_name)
    else:
        print(f"[upload] {remote_name} already present — skip")


def tail_log(studio: Studio, log_path: str, last_n: int = 15) -> str:
    return studio.run(f"tail -n {last_n} {log_path} 2>/dev/null || echo '(no log yet)'")


def wait_for_recon_all(studio: Studio, pid_file: str = "/tmp/fs_anchor.pid") -> bool:
    start = time.time()
    iteration = 0
    while True:
        elapsed = time.time() - start
        if elapsed > CPU_HARD_TIMEOUT_S:
            print(f"[ABORT] Timeout at {elapsed/3600:.1f} hrs (cap = {CPU_HARD_TIMEOUT_S/3600:.0f}).")
            studio.run(f"kill $(cat {pid_file}) 2>/dev/null || true")
            return False

        status = studio.run(f"kill -0 $(cat {pid_file}) 2>/dev/null && echo RUNNING || echo DONE")
        if "DONE" in status:
            done = studio.run("ls -la freesurfer_anchor_stats.tar.gz 2>/dev/null || echo MISSING")
            if "MISSING" not in done:
                print(f"[anchor] Completed at {elapsed/3600:.2f} hrs · results packaged")
                return True
            print("[anchor] Process exited but no tarball. Last log:")
            print(tail_log(studio, "/tmp/fs_anchor.log", 40))
            return False

        iteration += 1
        print(f"[poll #{iteration}] t={elapsed/3600:.2f}h · tail:")
        print(tail_log(studio, "/tmp/fs_anchor.log", 6))
        # FreeSurfer recon-all is slow — poll every 15 min not 3
        time.sleep(900)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--install-only", action="store_true", help="Install FreeSurfer only (debug)")
    ap.add_argument("--teamspace", default=None)
    args = ap.parse_args()

    check_auth()
    check_local_files(args.install_only)

    cpu_machine = Machine.CPU_X_32

    teamspace = args.teamspace or os.environ.get("LIGHTNING_TEAMSPACE")
    username = os.environ.get("LIGHTNING_USERNAME")
    print(f"[anchor] Attaching Studio '{STUDIO_NAME}' (teamspace={teamspace or 'auto'}, user={username or 'auto'})")
    studio_kwargs: dict = {"name": STUDIO_NAME, "create_ok": True}
    if teamspace:
        studio_kwargs["teamspace"] = teamspace
    if username:
        studio_kwargs["user"] = username
    studio = Studio(**studio_kwargs)

    try:
        print(f"[anchor] Starting on {cpu_machine}...")
        studio.start(machine=cpu_machine)
        print(f"[anchor] Studio running on {studio.machine}")

        upload_if_missing(studio, LOCAL_SETUP_SH, "lightning_freesurfer_anchor.sh")
        studio.run("chmod +x lightning_freesurfer_anchor.sh")

        if not args.install_only:
            upload_if_missing(studio, LOCAL_NIFTI_TAR, "nifti_freesurfer_anchor.tar.gz")
            upload_if_missing(studio, LOCAL_LICENSE, "license.txt")

        print("[anchor] Running install phase (FreeSurfer 7.4.1 + system deps)...")
        install_result = studio.run("bash lightning_freesurfer_anchor.sh install 2>&1 | tee install.log")
        print(install_result[-1500:])

        if args.install_only:
            print("[anchor] --install-only specified. Stopping Studio.")
            studio.stop()
            return 0

        print("[anchor] Kicking off FreeSurfer batch (detached, ~20 hrs)...")
        studio.run(
            "nohup bash lightning_freesurfer_anchor.sh process "
            "> /tmp/fs_anchor.log 2>&1 & echo $! > /tmp/fs_anchor.pid"
        )
        time.sleep(10)

        success = wait_for_recon_all(studio)

        if success:
            print("[anchor] Downloading results...")
            LOCAL_RESULTS.parent.mkdir(parents=True, exist_ok=True)
            studio.download_file("freesurfer_anchor_stats.tar.gz", str(LOCAL_RESULTS))
            size_mb = LOCAL_RESULTS.stat().st_size / 1e6
            print(f"[anchor] Saved {LOCAL_RESULTS} ({size_mb:.1f} MB)")

        print("[anchor] Stopping Studio (billing halts)...")
        studio.stop()
        return 0 if success else 1

    except Exception as e:
        print(f"[anchor] FATAL: {e}")
        try:
            print("[anchor] Emergency teardown...")
            studio.stop()
        except Exception as cleanup_err:
            print(f"[anchor] Cleanup also failed: {cleanup_err}. Stop manually at lightning.ai/studios.")
        return 2


if __name__ == "__main__":
    sys.exit(main())
