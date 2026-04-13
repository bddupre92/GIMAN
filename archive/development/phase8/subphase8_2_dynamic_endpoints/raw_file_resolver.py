"""Utilities for resolving latest raw PPMI input files with strict validation."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

MONTH_MAP = {
    "JAN": 1,
    "FEB": 2,
    "MAR": 3,
    "APR": 4,
    "MAY": 5,
    "JUN": 6,
    "JUL": 7,
    "AUG": 8,
    "SEP": 9,
    "OCT": 10,
    "NOV": 11,
    "DEC": 12,
}
DATE_RE = re.compile(r"(\d{2})([A-Za-z]{3})(\d{4})")


@dataclass(frozen=True)
class ResolvedRawFile:
    """Resolved file with provenance metadata."""

    modality_id: str
    path: str
    pattern: str
    root: str
    resolution_mode: str
    size_bytes: int
    sha256: str
    modified_utc: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def default_raw_roots(project_root: Path) -> list[Path]:
    """Ordered resolver roots: newest ad-hoc downloads first, legacy fallback last."""
    return [
        project_root / "data" / "00_raw",
        project_root / "data" / "00_raw" / "download-2",
        project_root / "data" / "00_raw" / "GIMAN" / "ppmi_data_csv",
    ]


def _filename_date_rank(path: Path) -> tuple[int, int, int]:
    """Extract DDMonYYYY date from filename for ranking."""
    m = DATE_RE.search(path.name)
    if not m:
        return (0, 0, 0)
    day = int(m.group(1))
    month = MONTH_MAP.get(m.group(2).upper(), 0)
    year = int(m.group(3))
    return (year, month, day)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


class RawFileResolver:
    """Resolve latest available files by modality with strict gates."""

    def __init__(self, roots: Iterable[Path]) -> None:
        self.roots = [Path(r) for r in roots]

    def resolve_latest(
        self,
        modality_id: str,
        patterns: list[str],
        *,
        required: bool = True,
        allow_empty: bool = False,
        required_columns: list[str] | None = None,
    ) -> ResolvedRawFile | None:
        candidates: list[tuple[Path, str, int]] = []
        for root_idx, root in enumerate(self.roots):
            if not root.exists():
                continue
            for pattern in patterns:
                for p in root.rglob(pattern):
                    if p.is_file():
                        candidates.append((p, pattern, root_idx))

        if not candidates:
            if required:
                raise FileNotFoundError(
                    f"No file found for modality '{modality_id}'. Patterns={patterns}, roots={self.roots}"
                )
            return None

        # Prefer latest filename date, then latest mtime.
        candidates.sort(
            key=lambda t: (_filename_date_rank(t[0]), t[0].stat().st_mtime),
            reverse=True,
        )
        path, pattern, root_idx = candidates[0]
        size_bytes = int(path.stat().st_size)
        if size_bytes == 0 and not allow_empty:
            raise ValueError(
                f"Resolved file for modality '{modality_id}' is empty: {path}"
            )

        if required_columns:
            try:
                header = pd.read_csv(path, nrows=0).columns.tolist()
            except Exception as exc:
                raise ValueError(f"Failed to read CSV header: {path}") from exc
            missing = [c for c in required_columns if c not in header]
            if missing:
                raise ValueError(
                    f"Resolved file for modality '{modality_id}' missing required columns {missing}: {path}"
                )

        mode = "primary" if root_idx == 0 else f"fallback_{root_idx}"
        modified_utc = datetime.fromtimestamp(
            path.stat().st_mtime, tz=timezone.utc
        ).isoformat()
        return ResolvedRawFile(
            modality_id=modality_id,
            path=str(path),
            pattern=pattern,
            root=str(self.roots[root_idx]),
            resolution_mode=mode,
            size_bytes=size_bytes,
            sha256=_sha256_file(path),
            modified_utc=modified_utc,
        )
