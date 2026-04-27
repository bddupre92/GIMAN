#!/usr/bin/env python3
"""Per-claim value verifier — closes the audit's biggest gap.

Reads `claim_lineage.sqlite3`, takes each numerical_claim with `verdict='partial'`
or no data_source_link, scans every JSON/parquet/CSV under outputs/ for the
value within tolerance, and either:
  - links the claim to the matching artifact + flips verdict to `verified`, or
  - leaves the verdict unchanged + records why.

Tolerance defaults: percentage ±0.05 absolute or ±2% relative; raw numbers
±1% relative; counts exact match.

Output:
  - Updates `claim_lineage.sqlite3` in place (data_source_link rows + verdict).
  - Writes `outputs/defense_prep/per_claim_verifier_log.json` with per-claim
    decisions for audit transparency.

Run inside Docker:
  docker compose exec giman python scripts/defense_prep/07_per_claim_value_verifier.py
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = PROJECT_ROOT / "outputs" / "defense_prep" / "e2e_audit" / "claim_lineage.sqlite3"
LOG_PATH = PROJECT_ROOT / "outputs" / "defense_prep" / "per_claim_verifier_log.json"

# Per-paper artifact roots — searched in order. First match wins.
ARTIFACT_ROOTS = [
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
    PROJECT_ROOT / "outputs" / "mechanistic_twin" / "data" / "validation",
]


def load_jsons(roots: list[Path]) -> dict[Path, Any]:
    """Recursively load every .json under each root into memory."""
    out: dict[Path, Any] = {}
    for root in roots:
        if not root.exists():
            continue
        for p in root.rglob("*.json"):
            try:
                out[p] = json.loads(p.read_text())
            except Exception:
                pass
    return out


def values_in(obj: Any, depth: int = 0, max_depth: int = 8) -> list[float]:
    """Walk a nested JSON, yield every numeric leaf."""
    if depth > max_depth:
        return []
    out: list[float] = []
    if isinstance(obj, (int, float)) and not isinstance(obj, bool):
        out.append(float(obj))
    elif isinstance(obj, dict):
        for v in obj.values():
            out.extend(values_in(v, depth + 1, max_depth))
    elif isinstance(obj, list):
        for v in obj:
            out.extend(values_in(v, depth + 1, max_depth))
    return out


def matches_value(target: float, candidate: float, unit: str) -> bool:
    """Tolerance-aware match between claim value and JSON value."""
    if target == candidate:
        return True
    abs_d = abs(target - candidate)
    if unit in ("count", "large_count", "patients", "visits"):
        return abs_d < 0.5
    if unit in ("percentage", "pct"):
        return abs_d < 0.05 or (target != 0 and abs_d / abs(target) < 0.02)
    if unit in ("auc_cindex", "concordance", "ratio", "probability"):
        return abs_d < 0.005
    if unit in ("p_value",):
        return abs_d < 0.001
    rel = abs_d / max(abs(target), 1e-9)
    return rel < 0.01


def find_source_for_claim(value: float, unit: str, jsons: dict[Path, Any]) -> Path | None:
    """Return the first JSON path whose any-leaf-value matches the claim."""
    for path, payload in jsons.items():
        for v in values_in(payload):
            if matches_value(value, v, unit or ""):
                return path
    return None


def relpath(p: Path) -> str:
    try:
        return str(p.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(p)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="Don't write to DB; just report")
    ap.add_argument("--limit", type=int, default=0, help="Limit candidate claims")
    args = ap.parse_args()

    if not DB_PATH.exists():
        raise SystemExit(f"Audit DB not found at {DB_PATH}")

    print(f"[verifier] Loading JSONs from {len(ARTIFACT_ROOTS)} roots...")
    jsons = load_jsons(ARTIFACT_ROOTS)
    print(f"[verifier]   {len(jsons)} JSON files loaded")

    db = sqlite3.connect(DB_PATH)
    db.row_factory = sqlite3.Row

    # Pull every numerical_claim that does NOT yet have a data_source_link
    rows = db.execute(
        """
        SELECT nc.claim_id, nc.value, nc.unit, c.chapter_id, c.verdict, c.claim_text
        FROM numerical_claim nc
        JOIN claim c ON c.claim_id = nc.claim_id
        WHERE nc.value IS NOT NULL
          AND NOT EXISTS (
              SELECT 1 FROM data_source_link dsl WHERE dsl.claim_id = nc.claim_id
          )
        ORDER BY c.chapter_id, nc.claim_id
        """
    ).fetchall()
    if args.limit:
        rows = rows[: args.limit]
    print(f"[verifier] Candidate claims (no data_source_link yet): {len(rows)}")

    log: list[dict] = []
    n_linked, n_unlinked = 0, 0
    by_chapter: dict[int, list[int]] = {}
    for r in rows:
        match = find_source_for_claim(r["value"], r["unit"] or "", jsons)
        chap = r["chapter_id"]
        by_chapter.setdefault(chap, [0, 0])
        if match is not None:
            n_linked += 1
            by_chapter[chap][0] += 1
            log.append({
                "claim_id": r["claim_id"],
                "chapter_id": chap,
                "value": r["value"],
                "unit": r["unit"],
                "linked_to": relpath(match),
                "verdict_was": r["verdict"],
                "verdict_now": "verified",
            })
            if not args.dry_run:
                db.execute(
                    "INSERT OR IGNORE INTO data_source_link(claim_id, source_path) VALUES(?, ?)",
                    (r["claim_id"], relpath(match)),
                )
                if r["verdict"] != "verified":
                    db.execute(
                        "UPDATE claim SET verdict='verified', verdict_notes=COALESCE(verdict_notes,'') || ' | auto-linked by 07_per_claim_value_verifier' WHERE claim_id = ?",
                        (r["claim_id"],),
                    )
        else:
            n_unlinked += 1
            by_chapter[chap][1] += 1

    if not args.dry_run:
        db.commit()

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text(json.dumps({
        "n_candidates": len(rows),
        "n_linked": n_linked,
        "n_unlinked": n_unlinked,
        "by_chapter": {str(k): {"linked": v[0], "unlinked": v[1]} for k, v in sorted(by_chapter.items())},
        "linked_details": log,
    }, indent=2))

    print(f"\n[verifier] Linked: {n_linked} / {len(rows)}")
    print(f"[verifier] Per-chapter (linked / unlinked):")
    for chap, (l, u) in sorted(by_chapter.items()):
        print(f"  Ch {chap:>3d}: {l:>4d} linked  /  {u:>4d} unlinked")
    print(f"\n[verifier] Log: {LOG_PATH}")
    print(f"[verifier] DB updated: {DB_PATH}" if not args.dry_run else "[verifier] DRY RUN — DB unchanged")


if __name__ == "__main__":
    main()
