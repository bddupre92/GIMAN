"""CLI wrapper for the smoke benchmark — multi-seed × multi-fraction × phys-GIMIN-lit + Mean.

Usage:
    python paper12_phys_gimin/scripts/phase1/run_smoke_benchmark.py \\
        --config paper12_phys_gimin/configs/phys_gimin_lit.yaml \\
        --mask-fractions 0.10 0.25 0.50 0.75 \\
        --n-seeds 3 \\
        --n-epochs 100 \\
        --output-dir outputs/paper12_phys_gimin/runs/smoke_w4_$(date +%Y%m%d_%H%M%S)

For real PPMI benchmark runs, use --no-mock-data (default is --mock-data for testing).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from phys_gimin.smoke_benchmark import run_multi_seed, SmokeRunResult


def main():
    parser = argparse.ArgumentParser(description="Phase 1 smoke benchmark")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mask-fractions", type=float, nargs="+",
                        default=[0.10, 0.25, 0.50, 0.75])
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--base-seed", type=int, default=1001)
    parser.add_argument("--n-epochs", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mock-data", dest="mock_data", action="store_true", default=True)
    parser.add_argument("--no-mock-data", dest="mock_data", action="store_false")
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "mps", "cuda"],
        default="auto",
        help="Training device (auto-detects CUDA -> MPS -> CPU by default).",
    )
    args = parser.parse_args()

    seeds = [args.base_seed + i for i in range(args.n_seeds)]
    all_results: list[SmokeRunResult] = []
    for frac in args.mask_fractions:
        # phys-GIMIN-lit across seeds
        phys_results = run_multi_seed(
            method="phys_gimin_lit", seeds=seeds, mask_fraction=frac,
            n_epochs=args.n_epochs, output_dir=args.output_dir / f"frac_{frac}",
            mock_data=args.mock_data, device=args.device,
        )
        # Mean baseline (single seed — deterministic)
        mean_results = run_multi_seed(
            method="mean", seeds=[args.base_seed], mask_fraction=frac,
            n_epochs=1, output_dir=args.output_dir / f"frac_{frac}",
            mock_data=args.mock_data, device=args.device,
        )
        all_results.extend(phys_results)
        all_results.extend(mean_results)

    # Aggregate summary
    summary = {
        "config": str(args.config),
        "seeds": seeds,
        "mask_fractions": args.mask_fractions,
        "n_epochs": args.n_epochs,
        "device": args.device,
        "total_runs": len(all_results),
        "completed_runs": sum(1 for r in all_results if r.status == "completed"),
        "per_run": [{"method": r.method, "seed": r.seed,
                     "mask_fraction": r.mask_fraction, "rmse": r.final_rmse,
                     "status": r.status} for r in all_results],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "smoke_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Wrote {args.output_dir / 'smoke_summary.json'}")


if __name__ == "__main__":
    main()
