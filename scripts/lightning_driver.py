"""Paper 12 W6 Step 4 — Lightning AI SDK driver for FastSurfer batch.

Orchestrates from your Mac:
  1. Attach/create Studio on free CPU tier
  2. Upload NIfTI tarball + license + setup script
  3. Run install phase on CPU (FreeSurfer + FastSurfer install)
  4. switch_machine to A100 40GB (or 2× A100 if requested)
  5. Run process phase detached, poll progress
  6. Download results tarball
  7. switch_machine back to CPU · stop Studio

Prerequisites (local Mac):
  pip install lightning-sdk
  export LIGHTNING_USER_ID=<from Lightning Account Settings → Keys>
  export LIGHTNING_API_KEY=<from Lightning Account Settings → Keys>

Required local files:
  /tmp/nifti_wang_n161.tar.gz         (from W6 Step 3)
  paper12_phys_gimin/data/license.txt (you register at FreeSurfer site)
  scripts/lightning_fastsurfer_setup.sh

Safety:
  Hard time cap on GPU phase (default 8 hrs).
  Studio switches back to CPU + stops automatically on success or failure.

Usage:
  python scripts/lightning_driver.py --dual-gpu      # default cost-optimal
  python scripts/lightning_driver.py --single-gpu
  python scripts/lightning_driver.py --install-only  # debug: no GPU switch
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
    sys.stderr.write(
        "ERROR: lightning-sdk not installed. Run:\n"
        "  pip install lightning-sdk\n"
    )
    sys.exit(1)

STUDIO_NAME = "paper12-fastsurfer"
WORKTREE = Path(__file__).resolve().parents[1]
REPO_ROOT = Path("/Users/blair.dupre/Projects/CSCI-FALL-2025")

LOCAL_NIFTI_TAR = Path("/tmp/nifti_wang_n161.tar.gz")
LOCAL_LICENSE = WORKTREE / "paper12_phys_gimin/data/license.txt"
LOCAL_SETUP_SH = WORKTREE / "scripts/lightning_fastsurfer_setup.sh"
LOCAL_RESULTS = WORKTREE / "paper12_phys_gimin/data/fastsurfer_stats.tar.gz"

GPU_HARD_TIMEOUT_S = 8 * 3600  # $20 cap → 8 hrs @ $2/hr dual A100 = $16


def check_auth() -> None:
    env_set = bool(os.environ.get("LIGHTNING_USER_ID") and os.environ.get("LIGHTNING_API_KEY"))
    creds_file = Path.home() / ".lightning" / "credentials"
    if not env_set and not creds_file.exists():
        sys.stderr.write(
            "ERROR: no Lightning credentials found. Either:\n"
            "  1. Run `lightning login` in a Terminal (saves to ~/.lightning/credentials)\n"
            "  2. Export LIGHTNING_USER_ID and LIGHTNING_API_KEY in your shell\n"
            "Get keys at https://lightning.ai/me/keys\n"
        )
        sys.exit(1)
    print(f"[auth] Using {'env vars' if env_set else 'cached credentials at ~/.lightning/credentials'}")


def check_local_files(install_only: bool) -> None:
    required = [LOCAL_SETUP_SH]
    if not install_only:
        required += [LOCAL_NIFTI_TAR, LOCAL_LICENSE]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        sys.stderr.write("ERROR: missing local files:\n" + "\n".join(f"  - {m}" for m in missing) + "\n")
        if str(LOCAL_LICENSE) in missing:
            sys.stderr.write(
                "\nFreeSurfer license: register at "
                "https://surfer.nmr.mgh.harvard.edu/registration.html and save the email's license.txt to:\n"
                f"  {LOCAL_LICENSE}\n"
            )
        sys.exit(1)


def upload_if_missing(studio: Studio, local: Path, remote_name: str) -> None:
    ls = studio.run(f"ls -la {remote_name} 2>/dev/null || echo MISSING")
    if "MISSING" in ls:
        print(f"[upload] {local.name} -> {remote_name}")
        studio.upload_file(str(local), remote_name)
    else:
        print(f"[upload] {remote_name} already present — skip")


def tail_log(studio: Studio, log_path: str, last_n: int = 15) -> str:
    return studio.run(f"tail -n {last_n} {log_path} 2>/dev/null || echo '(no log yet)'")


def wait_for_process_phase(studio: Studio, pid_file: str = "/tmp/fastsurfer.pid") -> bool:
    start = time.time()
    iteration = 0
    while True:
        elapsed = time.time() - start
        if elapsed > GPU_HARD_TIMEOUT_S:
            print(f"[ABORT] GPU timeout hit at {elapsed/3600:.1f} hrs (cap = {GPU_HARD_TIMEOUT_S/3600:.0f}). Killing + teardown.")
            studio.run(f"kill $(cat {pid_file}) 2>/dev/null || true")
            return False

        status = studio.run(f"kill -0 $(cat {pid_file}) 2>/dev/null && echo RUNNING || echo DONE")
        if "DONE" in status:
            done = studio.run("ls -la /teamspace/studios/this_studio/fastsurfer_stats.tar.gz 2>/dev/null || echo MISSING")
            if "MISSING" not in done:
                print(f"[process] Completed at {elapsed/60:.1f} min · results packaged")
                return True
            print(f"[process] Process exited but no tarball. Inspecting last log:")
            print(tail_log(studio, "/tmp/fastsurfer.log", 40))
            return False

        iteration += 1
        print(f"[poll #{iteration}] t={elapsed/60:.1f}m · tail:")
        print(tail_log(studio, "/tmp/fastsurfer.log", 5))
        time.sleep(180)  # 3 min poll


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dual-gpu", action="store_true", help="Use 2× A100 40GB (halves wall-clock at same total $)")
    ap.add_argument("--single-gpu", action="store_true", help="Use 1× A100 40GB (default)")
    ap.add_argument("--install-only", action="store_true", help="Install phase only (debug, no GPU switch)")
    ap.add_argument("--teamspace", default=None, help="Override teamspace (default: user's default)")
    args = ap.parse_args()

    check_auth()
    check_local_files(args.install_only)

    gpu_machine = Machine.A100_40GB_X_2 if args.dual_gpu else Machine.A100_40GB
    cpu_machine = Machine.CPU_X_4

    print(f"[driver] Attaching Studio '{STUDIO_NAME}' (teamspace={args.teamspace or 'default'})")
    studio_kwargs = {"name": STUDIO_NAME, "create_ok": True}
    if args.teamspace:
        studio_kwargs["teamspace"] = args.teamspace
    studio = Studio(**studio_kwargs)

    try:
        # ---- Boot on CPU ----
        print(f"[driver] Starting on {cpu_machine} (free tier)...")
        studio.start(machine=cpu_machine)
        print(f"[driver] Studio running on {studio.machine}")

        # ---- Upload files ----
        upload_if_missing(studio, LOCAL_SETUP_SH, "lightning_fastsurfer_setup.sh")
        studio.run("chmod +x lightning_fastsurfer_setup.sh")

        if not args.install_only:
            upload_if_missing(studio, LOCAL_NIFTI_TAR, "nifti_wang_n161.tar.gz")
            upload_if_missing(studio, LOCAL_LICENSE, "license.txt")

        # ---- Install phase on CPU ----
        print("[driver] Running install phase on CPU...")
        install_result = studio.run("bash lightning_fastsurfer_setup.sh install 2>&1 | tee install.log")
        print(install_result[-2000:])

        if args.install_only:
            print("[driver] --install-only specified. Stopping Studio.")
            studio.stop()
            return 0

        # ---- Switch to GPU ----
        print(f"[driver] Switching compute to {gpu_machine}...")
        studio.switch_machine(gpu_machine)
        print(f"[driver] Now running on {studio.machine}")

        # ---- Launch process phase detached ----
        print("[driver] Kicking off FastSurfer batch (detached)...")
        studio.run(
            "nohup bash lightning_fastsurfer_setup.sh process "
            "> /tmp/fastsurfer.log 2>&1 & echo $! > /tmp/fastsurfer.pid"
        )
        time.sleep(5)

        # ---- Poll ----
        success = wait_for_process_phase(studio)

        # ---- Download results ----
        if success:
            print(f"[driver] Downloading results tarball...")
            LOCAL_RESULTS.parent.mkdir(parents=True, exist_ok=True)
            studio.download_file("fastsurfer_stats.tar.gz", str(LOCAL_RESULTS))
            size_mb = LOCAL_RESULTS.stat().st_size / 1e6
            print(f"[driver] Saved {LOCAL_RESULTS} ({size_mb:.1f} MB)")

        # ---- Teardown ----
        print(f"[driver] Switching back to CPU (free tier)...")
        studio.switch_machine(cpu_machine)
        print(f"[driver] Stopping Studio (Drive storage preserved)...")
        studio.stop()
        return 0 if success else 1

    except Exception as e:
        print(f"[driver] FATAL: {e}")
        try:
            print("[driver] Attempting emergency teardown...")
            studio.switch_machine(cpu_machine)
            studio.stop()
        except Exception as cleanup_err:
            print(f"[driver] Cleanup also failed: {cleanup_err}. Go to Lightning dashboard and stop manually.")
        return 2


if __name__ == "__main__":
    sys.exit(main())
