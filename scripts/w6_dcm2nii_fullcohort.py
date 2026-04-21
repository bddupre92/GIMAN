"""Paper 12 W6 Step 3 — parallel dcm2niix conversion for full PD cohort."""
from __future__ import annotations

import csv
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

MANIFEST = Path(__file__).resolve().parents[1] / "paper12_phys_gimin/data/pd_full_cohort_manifest.csv"
NIFTI_ROOT = Path("/Users/blair.dupre/Projects/CSCI-FALL-2025/data/01_processed/GIMAN/t1_expansion_nifti")
LOG_OUT = Path(__file__).resolve().parents[1] / "paper12_phys_gimin/data/dcm2nii_log.csv"
N_WORKERS = 12


@dataclass(frozen=True)
class ScanJob:
    patno: int
    visit_date: str
    protocol: str
    dicom_path: str

    @property
    def out_dir(self) -> Path:
        return NIFTI_ROOT / f"PATNO_{self.patno}" / f"{self.visit_date}_{self.protocol}"

    @property
    def filename_base(self) -> str:
        return f"PATNO{self.patno}_{self.visit_date}_{self.protocol}"


def convert(job: ScanJob) -> dict:
    job.out_dir.mkdir(parents=True, exist_ok=True)
    existing = list(job.out_dir.glob("*.nii.gz"))
    if existing:
        return {"patno": job.patno, "date": job.visit_date, "status": "skipped", "output": str(existing[0])}
    cmd = [
        "dcm2niix",
        "-z", "y",
        "-f", job.filename_base,
        "-o", str(job.out_dir),
        "-b", "y",
        job.dicom_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        if result.returncode != 0:
            return {"patno": job.patno, "date": job.visit_date, "status": "failed", "output": result.stderr[:200]}
        produced = list(job.out_dir.glob("*.nii.gz"))
        if not produced:
            return {"patno": job.patno, "date": job.visit_date, "status": "no_output", "output": result.stdout[-200:]}
        return {"patno": job.patno, "date": job.visit_date, "status": "ok", "output": str(produced[0])}
    except subprocess.TimeoutExpired:
        return {"patno": job.patno, "date": job.visit_date, "status": "timeout", "output": ""}


def main() -> None:
    NIFTI_ROOT.mkdir(parents=True, exist_ok=True)
    jobs: list[ScanJob] = []
    with MANIFEST.open() as f:
        for row in csv.DictReader(f):
            jobs.append(ScanJob(
                patno=int(row["patno"]),
                visit_date=row["visit_date"],
                protocol=row["protocol"],
                dicom_path=row["dicom_path"],
            ))
    print(f"[dcm2nii] jobs={len(jobs)} workers={N_WORKERS}")

    results: list[dict] = []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futures = {ex.submit(convert, j): j for j in jobs}
        for i, fut in enumerate(as_completed(futures), 1):
            r = fut.result()
            results.append(r)
            if i % 25 == 0 or i == len(jobs):
                from collections import Counter
                status_counts = Counter(x["status"] for x in results)
                print(f"[dcm2nii] {i}/{len(jobs)}  {dict(status_counts)}")

    LOG_OUT.parent.mkdir(parents=True, exist_ok=True)
    with LOG_OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["patno", "date", "status", "output"])
        w.writeheader()
        w.writerows(results)

    from collections import Counter
    final = Counter(r["status"] for r in results)
    print(f"[dcm2nii] DONE  {dict(final)}  log={LOG_OUT}")


if __name__ == "__main__":
    main()
