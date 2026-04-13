"""Phase 3 Step 1c — Build 4×4 striatal connectivity matrix.

Constructs a literature-grounded connectivity matrix for the 4-region
propagation model (caudate L, caudate R, putamen L, putamen R).

Three variants are produced for sensitivity analysis:
  1. Anatomy-grounded (default): asymmetric weights from known neuroanatomy
  2. Equal weights: all non-zero connections = 1.0
  3. Putamen-dominant: stronger putamen↔putamen than caudate↔caudate

Output: outputs/mechanistic_twin/data/connectivity_4region.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "mechanistic_twin"))
from _reproducibility import capture_provenance, write_run_manifest

OUTPUT_DIR = REPO_ROOT / "outputs" / "mechanistic_twin" / "data"
OUTPUT_JSON = OUTPUT_DIR / "connectivity_4region.json"
MANIFEST_PATH = REPO_ROOT / "outputs" / "mechanistic_twin" / "phase2" / "phase3_step1c_RUN_MANIFEST.md"

# Region ordering (must be consistent across all Phase 3 scripts)
REGIONS = ["caudate_L", "caudate_R", "putamen_L", "putamen_R"]


def build_anatomy_grounded() -> np.ndarray:
    """Anatomy-grounded 4×4 connectivity matrix.

    Known anatomy (Lehéricy 2004, Parent & Hazrati 1995):
    - Ipsilateral caudate↔putamen: STRONG (striatal interneurons, ~0.5)
    - Contralateral caudate↔caudate: MODERATE (anterior commissure, ~0.2)
    - Contralateral putamen↔putamen: MODERATE (anterior commissure, ~0.2)
    - Contralateral caudate↔putamen: WEAK (indirect via thalamus, ~0.05)
    """
    A = np.array([
        [0.0,  0.20, 0.50, 0.05],  # caudate_L → others
        [0.20, 0.0,  0.05, 0.50],  # caudate_R → others
        [0.50, 0.05, 0.0,  0.20],  # putamen_L → others
        [0.05, 0.50, 0.20, 0.0 ],  # putamen_R → others
    ])
    return A


def build_equal_weights() -> np.ndarray:
    """Equal-weight connectivity (null connectivity model)."""
    A = np.ones((4, 4)) - np.eye(4)
    return A


def build_putamen_dominant() -> np.ndarray:
    """Putamen-dominant: stronger putamen bilateral connection.

    Rationale: putamen is the primary target of nigrostriatal projections
    and receives denser dopaminergic innervation.
    """
    A = np.array([
        [0.0,  0.15, 0.50, 0.05],
        [0.15, 0.0,  0.05, 0.50],
        [0.50, 0.05, 0.0,  0.35],  # putamen bilateral stronger
        [0.05, 0.50, 0.35, 0.0 ],
    ])
    return A


def row_normalize(A: np.ndarray) -> np.ndarray:
    """Row-normalize so each row sums to 1."""
    row_sums = A.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return A / row_sums


def main() -> int:
    prov = capture_provenance(
        script_path=Path(__file__),
        repo_root=REPO_ROOT,
        input_files=[],
        extra={"regions": REGIONS, "n_variants": 3},
    )

    print("=" * 72)
    print("Phase 3 Step 1c — 4×4 Striatal Connectivity Matrix")
    print("=" * 72)

    variants = {}
    for name, builder in [
        ("anatomy_grounded", build_anatomy_grounded),
        ("equal_weights", build_equal_weights),
        ("putamen_dominant", build_putamen_dominant),
    ]:
        A_raw = builder()
        A_norm = row_normalize(A_raw)

        # Verify symmetry
        assert np.allclose(A_raw, A_raw.T), f"{name}: matrix is NOT symmetric"

        # Compute graph Laplacian (for NDM: dL/dt = -k_clear*L + k_spread * A * L)
        # The Laplacian L = D - A where D = diag(row_sums(A))
        # But for our NDM we use A directly, not the Laplacian
        D = np.diag(A_norm.sum(axis=1))
        laplacian = D - A_norm

        eigenvalues = np.linalg.eigvalsh(laplacian)

        print(f"\n  Variant: {name}")
        print(f"    Raw matrix:\n{A_raw}")
        print(f"    Row-normalized:\n{np.round(A_norm, 4)}")
        print(f"    Laplacian eigenvalues: {np.round(eigenvalues, 4)}")
        print(f"    Fiedler value (algebraic connectivity): {eigenvalues[1]:.4f}")

        variants[name] = {
            "raw": A_raw.tolist(),
            "normalized": A_norm.tolist(),
            "laplacian_eigenvalues": eigenvalues.tolist(),
            "fiedler_value": float(eigenvalues[1]),
        }

    output = {
        "regions": REGIONS,
        "region_indices": {r: i for i, r in enumerate(REGIONS)},
        "description": "4×4 striatal connectivity for Phase 3 NDM. 3 variants for sensitivity.",
        "default_variant": "anatomy_grounded",
        "literature_sources": {
            "anatomy_grounded": "Lehéricy 2004, Parent & Hazrati 1995 (known basal ganglia circuitry)",
            "equal_weights": "Null model (no anatomical weighting)",
            "putamen_dominant": "Kish 1988 (putamen receives denser nigrostriatal projection)",
        },
        "sensitivity_note": "Model comparison conclusions should be INVARIANT to variant choice. "
                           "If not, connectivity is a confound and must be addressed.",
        "variants": variants,
        "_provenance": prov,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nWrote {OUTPUT_JSON}")

    import hashlib
    out_sha = hashlib.sha256(OUTPUT_JSON.read_bytes()).hexdigest()[:16]

    write_run_manifest(
        manifest_path=MANIFEST_PATH,
        step_name="Phase 3 Step 1c — Connectivity Matrix",
        provenance=prov,
        gate_results={"n_variants": "3", "all_symmetric": "PASS"},
        summary_metrics={"connectivity_4region.json SHA": out_sha},
    )
    print(f"Wrote {MANIFEST_PATH}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
