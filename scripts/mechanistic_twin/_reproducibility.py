"""Shared reproducibility-header helper for Phase 2 mechanistic-twin Python scripts.

Contract (locked 2026-04-09, closed-loop methodology v1.0 reproducibility rule):

Every durable-artifact-producing Python script under `scripts/mechanistic_twin/`
MUST, at the beginning of its `main()`, call `capture_provenance(...)` to
record:

  (a) Git SHA of the repository + dirty flag
  (b) SHA-256 hash of the script file itself (self-hash)
  (c) SHA-256 hash + row count of every input data file it reads
  (d) Python version, NumPy/Pandas/PyArrow versions, platform string
  (e) Full CLI argv
  (f) UTC timestamp

and MUST:

  (1) Embed the returned provenance dict inside the JSON summary it writes
      (under top-level key `_provenance`)
  (2) Write a standalone `<step>_RUN_MANIFEST.md` companion file that mirrors
      the provenance in human-readable form, co-located with the primary
      JSON output. This file is the canonical "what exactly produced these
      artifacts" receipt and is referenced in
      `outputs/mechanistic_twin/phase2/REPRODUCIBILITY_MANIFEST.md`.

The helper intentionally avoids any third-party dependencies beyond the
standard library + NumPy/Pandas/PyArrow which are already required by any
downstream Python script in this tree.

Test: `tests/mechanistic_twin/test_reproducibility_helper.py`
"""
from __future__ import annotations

import hashlib
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import numpy as _np
    _NUMPY_VERSION = _np.__version__
except Exception:  # pragma: no cover - numpy is always present in .venv
    _NUMPY_VERSION = "not-installed"
try:
    import pandas as _pd
    _PANDAS_VERSION = _pd.__version__
except Exception:  # pragma: no cover
    _PANDAS_VERSION = "not-installed"
try:
    import pyarrow as _pa
    _PYARROW_VERSION = _pa.__version__
except Exception:  # pragma: no cover
    _PYARROW_VERSION = "not-installed"


def _sha256_file(path: Path, chunk_size: int = 2**20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_info(repo_root: Path) -> dict[str, Any]:
    """Return {sha, dirty, branch} or {error: ...} if not a git repo."""
    try:
        sha = subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        branch = subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "--abbrev-ref", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        # Dirty flag: does this specific script + any of its declared input
        # files have uncommitted modifications? (cheap approximation: global
        # repo dirty state. Safe side — we always err toward "dirty".)
        status = subprocess.check_output(
            ["git", "-C", str(repo_root), "status", "--porcelain"],
            stderr=subprocess.DEVNULL,
        ).decode()
        dirty = len(status.strip()) > 0
        return {"sha": sha, "branch": branch, "dirty": dirty}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def capture_provenance(
    script_path: Path,
    repo_root: Path,
    input_files: list[Path],
    cli_args: list[str] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Capture a complete provenance record.

    Parameters
    ----------
    script_path
        Absolute path to the script calling this helper (use ``Path(__file__)``).
    repo_root
        Absolute path to the git repo root (use ``Path(__file__).resolve().parents[N]``).
    input_files
        List of absolute paths to every input data file the script reads.
        Each file is hashed + its row count recorded if it is a tabular
        (parquet/csv) file.
    cli_args
        Optional — if None, uses ``sys.argv``.
    extra
        Optional additional dict to merge into the provenance record under
        the ``extra`` key (e.g. random seed, N_prior).

    Returns
    -------
    A provenance dict suitable for JSON serialization.
    """
    provenance: dict[str, Any] = {
        "datetime_utc": datetime.now(timezone.utc).isoformat(),
        "python": {
            "version": sys.version.split()[0],
            "executable": sys.executable,
            "platform": platform.platform(),
        },
        "packages": {
            "numpy":   _NUMPY_VERSION,
            "pandas":  _PANDAS_VERSION,
            "pyarrow": _PYARROW_VERSION,
        },
        "git": _git_info(repo_root),
        "script": {
            "path": str(script_path.relative_to(repo_root)) if script_path.is_absolute() else str(script_path),
            "sha256": _sha256_file(script_path) if script_path.exists() else None,
            "size_bytes": script_path.stat().st_size if script_path.exists() else None,
        },
        "cli_args": list(cli_args if cli_args is not None else sys.argv),
        "input_files": [],
    }

    for p in input_files:
        if not p.exists():
            provenance["input_files"].append({
                "path": str(p.relative_to(repo_root)) if p.is_absolute() else str(p),
                "status": "MISSING",
            })
            continue
        row_count: int | None = None
        try:
            suffix = p.suffix.lower()
            if suffix == ".parquet":
                import pyarrow.parquet as pq
                row_count = pq.ParquetFile(str(p)).metadata.num_rows
            elif suffix == ".csv":
                # Count lines minus header
                with p.open("rb") as f:
                    row_count = sum(1 for _ in f) - 1
        except Exception as exc:
            row_count = None
            provenance.setdefault("row_count_errors", []).append(
                f"{p.name}: {type(exc).__name__}: {exc}"
            )
        provenance["input_files"].append({
            "path": str(p.relative_to(repo_root)) if p.is_absolute() else str(p),
            "sha256": _sha256_file(p),
            "size_bytes": p.stat().st_size,
            "row_count": row_count,
        })

    if extra:
        provenance["extra"] = extra

    return provenance


def write_run_manifest(
    manifest_path: Path,
    step_name: str,
    provenance: dict[str, Any],
    gate_results: dict[str, Any] | None = None,
    summary_metrics: dict[str, Any] | None = None,
) -> None:
    """Write a human-readable RUN_MANIFEST.md companion file.

    Called at the end of a script's main(), after all outputs are written.
    """
    lines: list[str] = []
    lines.append(f"# {step_name} — Run Manifest")
    lines.append("")
    lines.append("**Purpose:** canonical provenance receipt for this specific run. "
                 "Every scientific claim traced to this run's outputs must cite this "
                 "manifest row.")
    lines.append("")
    lines.append(f"**Run timestamp (UTC):** {provenance['datetime_utc']}")
    lines.append("")
    lines.append("## Environment")
    lines.append("")
    lines.append("| Component | Value |")
    lines.append("|---|---|")
    lines.append(f"| Python         | {provenance['python']['version']} |")
    lines.append(f"| Platform       | {provenance['python']['platform']} |")
    lines.append(f"| NumPy          | {provenance['packages']['numpy']} |")
    lines.append(f"| Pandas         | {provenance['packages']['pandas']} |")
    lines.append(f"| PyArrow        | {provenance['packages']['pyarrow']} |")
    git = provenance["git"]
    if "sha" in git:
        dirty_flag = " (DIRTY)" if git.get("dirty") else ""
        lines.append(f"| Git SHA        | `{git['sha']}`{dirty_flag} |")
        lines.append(f"| Git branch     | `{git.get('branch', '?')}` |")
    else:
        lines.append(f"| Git            | {git.get('error', 'unknown')} |")
    lines.append("")
    lines.append("## Script self-hash")
    lines.append("")
    s = provenance["script"]
    lines.append(f"- **Path:** `{s['path']}`")
    lines.append(f"- **SHA-256:** `{s['sha256']}`")
    lines.append(f"- **Size:** {s['size_bytes']} bytes")
    lines.append("")
    lines.append("## CLI invocation")
    lines.append("")
    lines.append("```")
    lines.append(" ".join(provenance["cli_args"]))
    lines.append("```")
    lines.append("")
    lines.append("## Input files")
    lines.append("")
    lines.append("| Path | SHA-256 (first 16) | Rows | Size (bytes) |")
    lines.append("|---|---|---|---|")
    for f in provenance["input_files"]:
        if f.get("status") == "MISSING":
            lines.append(f"| `{f['path']}` | **MISSING** | — | — |")
        else:
            h16 = f["sha256"][:16] if f.get("sha256") else "?"
            rc = f.get("row_count", "-")
            lines.append(f"| `{f['path']}` | `{h16}` | {rc} | {f.get('size_bytes', '-')} |")
    lines.append("")

    if provenance.get("extra"):
        lines.append("## Run-specific parameters")
        lines.append("")
        lines.append("```json")
        import json as _json
        lines.append(_json.dumps(provenance["extra"], indent=2))
        lines.append("```")
        lines.append("")

    if summary_metrics:
        lines.append("## Summary metrics (published)")
        lines.append("")
        for k, v in summary_metrics.items():
            lines.append(f"- **{k}:** {v}")
        lines.append("")

    if gate_results:
        lines.append("## Gate results")
        lines.append("")
        for k, v in gate_results.items():
            status = "PASS" if v else "FAIL"
            lines.append(f"- **{k}:** {status} ({v})")
        lines.append("")

    lines.append("## Reproducing this run")
    lines.append("")
    lines.append("```bash")
    lines.append(f"cd \"${{REPO_ROOT}}\"")
    lines.append(f"git checkout {git.get('sha', 'HEAD')}")
    lines.append(f".venv/bin/python {s['path']}")
    lines.append("```")
    lines.append("")
    lines.append("If the rerun produces different output hashes, one of the following has "
                 "drifted: (a) input data vintages, (b) Python package versions, (c) "
                 "script source, (d) RNG seed. Each of these is recorded above — diff "
                 "the new run's manifest against this one to identify the drift source.")
    lines.append("")

    manifest_path.write_text("\n".join(lines))
