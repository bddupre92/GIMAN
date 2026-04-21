"""Paper 12 W6 Step 4 (local) — FastSurfer seg_only batch on Mac via Docker.

Runs FastSurfer's DL segmentation (`--seg_only`) on all Wang N=161 cohort
NIfTIs. Uses `deepmi/fastsurfer` Docker image. No GPU — Mac CPU with 8-16
threads per scan. ~3-5 min/scan → ~20-30 hrs for 401 scans. Free (electricity).

Output per subject: data/02_fastsurfer_out/<subj_id>/stats/aseg.stats (68 volumes).

Idempotent: skips subjects whose aseg.stats already exists. Safe to Ctrl-C and
resume — just re-run.

Usage:
  python scripts/local_fastsurfer_batch.py               # run full Wang N=161
  python scripts/local_fastsurfer_batch.py --limit 3     # process 3 scans (smoke test)
  python scripts/local_fastsurfer_batch.py --threads 12  # override OpenMP threads (default: 8)
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path

DOCKER = "/Applications/Docker.app/Contents/Resources/bin/docker"
DOCKER_BIN_DIR = "/Applications/Docker.app/Contents/Resources/bin"
IMAGE = "deepmi/fastsurfer:latest"


def _docker_env() -> dict:
    """Build env with Docker bin in PATH so docker-credential-desktop is findable."""
    import os as _os
    env = _os.environ.copy()
    env["PATH"] = f"{DOCKER_BIN_DIR}:{env.get('PATH', '')}"
    return env

WORKTREE = Path(__file__).resolve().parents[1]
REPO_ROOT = Path("/Users/blair.dupre/Projects/CSCI-FALL-2025")

MANIFEST = WORKTREE / "paper12_phys_gimin/data/wang_n161_manifest.csv"
T1_NIFTI_ROOT = REPO_ROOT / "data/01_processed/GIMAN/t1_expansion_nifti"
OUTPUT_DIR = REPO_ROOT / "data/02_fastsurfer_out"
LICENSE_FILE = WORKTREE / "paper12_phys_gimin/data/license.txt"
LOG_FILE = WORKTREE / "paper12_phys_gimin/data/local_fastsurfer_batch.log"


def check_prereqs() -> None:
    for p in [MANIFEST, T1_NIFTI_ROOT, LICENSE_FILE]:
        if not p.exists():
            sys.stderr.write(f"ERROR: missing {p}\n")
            sys.exit(1)
    env = _docker_env()
    # Docker daemon check
    r = subprocess.run([DOCKER, "info"], capture_output=True, text=True, env=env)
    if r.returncode != 0:
        sys.stderr.write("ERROR: Docker daemon not running. Open Docker Desktop.\n")
        sys.exit(1)
    # Image pull (idempotent)
    r = subprocess.run([DOCKER, "image", "inspect", IMAGE], capture_output=True, text=True, env=env)
    if r.returncode != 0:
        print(f"[prereq] Pulling {IMAGE} (~5 GB, one-time)...")
        subprocess.run([DOCKER, "pull", IMAGE], check=True, env=env)
    else:
        print(f"[prereq] Image {IMAGE} already pulled")


def load_scans() -> list[tuple[str, Path]]:
    """Return list of (subject_id, nifti_path) for Wang N=161 cohort."""
    scans: list[tuple[str, Path]] = []
    with MANIFEST.open() as f:
        for row in csv.DictReader(f):
            patno = row["patno"]
            date = row["visit_date"]
            proto = row["protocol"]
            subj_dir = T1_NIFTI_ROOT / f"PATNO_{patno}" / f"{date}_{proto}"
            niis = list(subj_dir.glob("*.nii.gz"))
            if not niis:
                print(f"WARN: no nifti for PATNO{patno} {date} {proto}")
                continue
            subj_id = f"PATNO{patno}_{date}_{proto}"
            scans.append((subj_id, niis[0]))
    return scans


def is_complete(subj_id: str) -> bool:
    """FastSurfer --seg_only writes stats/aseg+DKT.stats (68 subcortical volumes,
    Desikan-Killiany atlas). NOT stats/aseg.stats — that's FreeSurfer's classic
    output which only appears after full surface recon."""
    return (OUTPUT_DIR / subj_id / "stats" / "aseg+DKT.stats").exists()


def run_one(subj_id: str, nifti: Path, threads: int) -> tuple[str, float]:
    """Run FastSurfer seg_only on one scan. Returns (status, elapsed_sec)."""
    t0 = time.time()
    if is_complete(subj_id):
        return ("SKIP", 0.0)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cmd = [
        DOCKER, "run", "--rm",
        "-v", f"{nifti.parent}:/input",
        "-v", f"{OUTPUT_DIR}:/output",
        "-v", f"{LICENSE_FILE}:/fs_license.txt",
        "--user", f"{__import__('os').getuid()}:{__import__('os').getgid()}",
        IMAGE,
        "--t1", f"/input/{nifti.name}",
        "--sid", subj_id,
        "--sd", "/output",
        "--seg_only",
        "--threads", str(threads),
        "--fs_license", "/fs_license.txt",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800, env=_docker_env())
    elapsed = time.time() - t0
    if r.returncode != 0 or not is_complete(subj_id):
        # Persist failure log
        fail_log = OUTPUT_DIR / subj_id / "_fail.log"
        fail_log.parent.mkdir(parents=True, exist_ok=True)
        fail_log.write_text(f"RC={r.returncode}\n\n=== stdout ===\n{r.stdout}\n\n=== stderr ===\n{r.stderr}")
        return ("FAIL", elapsed)
    return ("OK", elapsed)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="process only N scans (smoke test)")
    ap.add_argument("--threads", type=int, default=8, help="OpenMP threads per scan (default 8)")
    args = ap.parse_args()

    check_prereqs()
    scans = load_scans()
    if args.limit:
        scans = scans[: args.limit]
    print(f"[batch] {len(scans)} scans queued · threads={args.threads} · output={OUTPUT_DIR}")

    from collections import Counter
    counts: Counter = Counter()
    total_elapsed = 0.0

    with LOG_FILE.open("a") as logf:
        logf.write(f"\n=== batch start {time.strftime('%Y-%m-%d %H:%M:%S')} (n={len(scans)}) ===\n")
        for i, (subj_id, nifti) in enumerate(scans, 1):
            status, elapsed = run_one(subj_id, nifti, args.threads)
            total_elapsed += elapsed
            counts[status] += 1
            done = counts["OK"] + counts["SKIP"]
            avg_min = (total_elapsed / 60) / max(1, counts["OK"]) if counts["OK"] else 0
            eta_hrs = avg_min * (len(scans) - done) / 60 if avg_min else 0
            msg = f"[{i}/{len(scans)}] {status} {subj_id} ({elapsed:.0f}s)  done={done} OK={counts['OK']} FAIL={counts['FAIL']} avg_min={avg_min:.1f} ETA_hrs={eta_hrs:.1f}"
            print(msg)
            logf.write(msg + "\n")
            logf.flush()

    print(f"\n[batch] DONE  OK={counts['OK']}  SKIP={counts['SKIP']}  FAIL={counts['FAIL']}")
    return 0 if counts["FAIL"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
