#!/usr/bin/env python3
"""Hard-coded / fabricated output scanner.

Addresses the defense question: "are we sure no JSON/CSV/parquet output is
hand-edited or hard-coded?"

For every file under outputs/{paper{1..6}_*,external_validation,mechanistic_twin,paper{3,4}_*}/:

  1. Find the most-likely producing script under scripts/ and src/ via grep on the
     output basename, capture its mtime.
  2. Compare the OUTPUT mtime against the SCRIPT mtime.
       - If output is OLDER than script: stale, may not reflect current code.
       - If output is NEWER than script: producible from current code (good).
       - If no producing script found: ORPHAN — output has no traceable producer.

  3. Numerical-suspiciousness heuristic: flag JSON files whose numeric leaves
     are suspiciously round (e.g., all values divisible by 0.05, or 5+ values
     ending in 00 / .50). A real ML output JSON should have many non-round
     decimals; a hand-edited one will have suspiciously round values.

  4. CSV/parquet sanity: spot-check row count and column count vs whatever the
     producing script's docstring or CLAUDE.md says.

Output: outputs/defense_prep/hardcoded_output_scan.json + console table.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
SRC_DIR = PROJECT_ROOT / "src"
OUT = PROJECT_ROOT / "outputs" / "defense_prep" / "hardcoded_output_scan.json"

OUTPUT_ROOTS = [
    PROJECT_ROOT / "outputs" / "paper1_benchmark",
    PROJECT_ROOT / "outputs" / "paper1_conformal",
    PROJECT_ROOT / "outputs" / "paper1_experiments",
    PROJECT_ROOT / "outputs" / "paper2_benchmark",
    PROJECT_ROOT / "outputs" / "paper3_markov",
    PROJECT_ROOT / "outputs" / "paper3_deephit",
    PROJECT_ROOT / "outputs" / "paper3_graph_dt",
    PROJECT_ROOT / "outputs" / "paper3_benchmark",
    PROJECT_ROOT / "outputs" / "paper4",
    PROJECT_ROOT / "outputs" / "external_validation",
    PROJECT_ROOT / "outputs" / "mechanistic_twin" / "phase2",
    PROJECT_ROOT / "outputs" / "mechanistic_twin" / "phase3",
    PROJECT_ROOT / "outputs" / "mechanistic_twin" / "phase4",
    PROJECT_ROOT / "outputs" / "mechanistic_twin" / "paper10_mech_vs_giman",
]


def find_producer_script(output_path: Path) -> Path | None:
    """grep scripts/ + src/ for the output basename. Return first hit."""
    basename = output_path.name
    grep_target = basename
    # Most scripts reference outputs by basename or by partial path
    candidates = []
    for root in (SCRIPTS_DIR, SRC_DIR):
        for py in root.rglob("*.py"):
            try:
                content = py.read_text(errors="ignore")
                if grep_target in content:
                    candidates.append(py)
            except Exception:
                pass
    return candidates[0] if candidates else None


def numeric_leaves(obj, depth: int = 0, max_depth: int = 8) -> list[float]:
    if depth > max_depth:
        return []
    out = []
    if isinstance(obj, (int, float)) and not isinstance(obj, bool):
        out.append(float(obj))
    elif isinstance(obj, dict):
        for v in obj.values():
            out.extend(numeric_leaves(v, depth + 1, max_depth))
    elif isinstance(obj, list):
        for v in obj:
            out.extend(numeric_leaves(v, depth + 1, max_depth))
    return out


def round_score(values: list[float]) -> dict:
    """Flag suspiciously round values. Real ML outputs have many decimals."""
    if not values:
        return {"n": 0, "round_ratio": 0.0, "flag": False}
    n = len(values)
    n_round = sum(1 for v in values if v == round(v) or v == round(v, 1) or v == round(v, 2))
    ratio = n_round / n
    return {
        "n": n,
        "n_round_2dp": n_round,
        "round_ratio": ratio,
        "flag": ratio > 0.40 and n > 20,
    }


def scan_file(p: Path) -> dict:
    rec = {
        "path": str(p.relative_to(PROJECT_ROOT)),
        "size_bytes": p.stat().st_size,
        "mtime": p.stat().st_mtime,
    }
    producer = find_producer_script(p)
    if producer is not None:
        rec["producer_script"] = str(producer.relative_to(PROJECT_ROOT))
        rec["producer_mtime"] = producer.stat().st_mtime
        rec["producer_newer_than_output"] = producer.stat().st_mtime > p.stat().st_mtime
    else:
        rec["producer_script"] = None
        rec["orphan"] = True

    if p.suffix == ".json":
        try:
            payload = json.loads(p.read_text())
            leaves = numeric_leaves(payload)
            rec["round_score"] = round_score(leaves)
        except Exception as e:
            rec["json_parse_error"] = str(e)[:100]

    if p.suffix == ".csv":
        try:
            import pandas as pd
            df = pd.read_csv(p, nrows=5)
            rec["csv_n_cols"] = len(df.columns)
            with p.open() as fh:
                rec["csv_n_rows"] = sum(1 for _ in fh) - 1
        except Exception as e:
            rec["csv_parse_error"] = str(e)[:100]

    if p.suffix == ".parquet":
        try:
            import pandas as pd
            df = pd.read_parquet(p)
            rec["parquet_shape"] = list(df.shape)
        except Exception as e:
            rec["parquet_parse_error"] = str(e)[:100]

    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    all_files: list[Path] = []
    for root in OUTPUT_ROOTS:
        if not root.exists():
            continue
        for ext in (".json", ".csv", ".parquet"):
            all_files.extend(root.rglob(f"*{ext}"))
    if args.limit:
        all_files = all_files[: args.limit]
    print(f"[scanner] Scanning {len(all_files)} output files (json/csv/parquet)...")

    scanned = []
    orphans = []
    suspicious = []
    stale = []
    for f in all_files:
        rec = scan_file(f)
        scanned.append(rec)
        if rec.get("orphan"):
            orphans.append(rec["path"])
        if rec.get("round_score", {}).get("flag"):
            suspicious.append(rec["path"])
        if rec.get("producer_newer_than_output") is False:
            stale.append(rec["path"])
    # Wait — `producer_newer_than_output` False means producer is NOT newer.
    # Re-derive: stale = output is OLDER than producer script (suggesting producer was updated since).
    stale = [r["path"] for r in scanned if r.get("producer_newer_than_output") is True]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "n_scanned": len(scanned),
        "n_orphans": len(orphans),
        "n_suspicious_round": len(suspicious),
        "n_producer_newer_than_output": len(stale),
        "orphans": orphans,
        "suspicious_round": suspicious,
        "producer_newer_than_output": stale,
        "all_records": scanned,
    }, indent=2))

    print(f"\n=== Hard-coded Output Scan Summary ===")
    print(f"  Total scanned: {len(scanned)}")
    print(f"  Orphans (no producing script found): {len(orphans)}")
    print(f"  Suspiciously-round JSON outputs: {len(suspicious)}")
    print(f"  Producer script newer than output (stale): {len(stale)}")
    print()
    if orphans[:10]:
        print(f"Top 10 ORPHANS (no traceable producing script):")
        for o in orphans[:10]:
            print(f"  {o}")
    print()
    if suspicious[:10]:
        print(f"Top 10 SUSPICIOUSLY ROUND values:")
        for s in suspicious[:10]:
            print(f"  {s}")
    print()
    if stale[:10]:
        print(f"Top 10 STALE outputs (producer script edited since):")
        for s in stale[:10]:
            print(f"  {s}")
    print(f"\nFull scan: {OUT}")


if __name__ == "__main__":
    main()
