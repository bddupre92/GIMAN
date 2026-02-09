#!/usr/bin/env python3
"""Step 3: Build the patient similarity graph.

Constructs a k-nearest-neighbor patient similarity graph from the
extracted multimodal feature matrix. The graph is used by the GIMIN
GNN for message-passing during imputation.

Similarity is computed using cosine similarity over shared observed
features, with a minimum overlap threshold to avoid spurious edges.

Prerequisites:
    - outputs/ppmi_full_cohort.parquet (from Step 1 or 2)
    - outputs/missingness_mask.parquet

Outputs:
    outputs/patient_graph.pt           -- PyTorch Geometric Data object
    outputs/patient_graph_metadata.json -- graph statistics

Usage:
    python scripts/03_build_graph.py [--config configs/default.yaml]
    python scripts/03_build_graph.py --k-neighbors 20 --min-overlap 5
"""

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build patient similarity graph for GIMIN."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--input-features",
        type=str,
        default=None,
        help="Path to feature parquet. Default: outputs/ppmi_full_cohort.parquet",
    )
    parser.add_argument(
        "--input-mask",
        type=str,
        default=None,
        help="Path to missingness mask parquet.",
    )
    parser.add_argument(
        "--k-neighbors",
        type=int,
        default=None,
        help="Number of nearest neighbors (overrides config).",
    )
    parser.add_argument(
        "--min-overlap",
        type=int,
        default=None,
        help="Minimum shared observed features for an edge.",
    )
    parser.add_argument(
        "--similarity-metric",
        type=str,
        default=None,
        choices=["cosine", "euclidean", "correlation"],
        help="Similarity metric for kNN.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger("gimin.graph")

    import pandas as pd

    from gimin.config import GIMINConfig

    # Load config
    if args.config:
        config = GIMINConfig.from_yaml(args.config)
    else:
        config = GIMINConfig()

    # Apply CLI overrides
    k = args.k_neighbors or config.graph.k_neighbors
    min_overlap = args.min_overlap or config.graph.min_overlap
    metric = args.similarity_metric or config.graph.similarity_metric

    output_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    feat_path = (
        Path(args.input_features)
        if args.input_features
        else output_dir / "ppmi_full_cohort.parquet"
    )
    mask_path = (
        Path(args.input_mask)
        if args.input_mask
        else output_dir / "missingness_mask.parquet"
    )

    print("=" * 70)
    print("GIMIN: Patient Similarity Graph Construction (Step 3)")
    print("=" * 70)

    # Load data
    logger.info("Loading features from: %s", feat_path)
    features_df = pd.read_parquet(feat_path)
    logger.info("Loading mask from: %s", mask_path)
    mask_df = pd.read_parquet(mask_path)

    logger.info(
        "  %d patients, %d features", features_df.shape[0], features_df.shape[1]
    )
    logger.info("  k=%d, min_overlap=%d, metric=%s", k, min_overlap, metric)

    # Build graph
    import torch

    from gimin.graph import build_patient_graph

    graph_result = build_patient_graph(
        features_df=features_df,
        mask_df=mask_df,
        k_neighbors=k,
        min_overlap=min_overlap,
        similarity_metric=metric,
    )

    # Save the graph tensors (exclude large similarity_matrix)
    save_dict = {
        "edge_index": graph_result["edge_index"],
        "edge_weight": graph_result["edge_weight"],
        "overlap_frac": graph_result["overlap_frac"],
        "eligible_indices": graph_result.get("eligible_indices"),
    }
    graph_path = output_dir / "patient_graph.pt"
    torch.save(save_dict, graph_path)
    logger.info("Saved graph to: %s", graph_path)

    num_nodes = graph_result["num_nodes"]
    num_edges = graph_result["num_edges"]
    metadata = {
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "total_patients": graph_result.get("total_patients", num_nodes),
        "k_neighbors": k,
        "min_overlap": min_overlap,
        "similarity_metric": metric,
        "avg_degree": float(num_edges / max(num_nodes, 1)),
        "num_isolated": graph_result.get("num_isolated", 0),
    }

    meta_path = output_dir / "patient_graph_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n{'=' * 70}")
    print("Graph Construction Complete")
    print(f"{'=' * 70}")
    print(f"  Nodes (patients): {metadata.get('num_nodes', 'N/A')}")
    print(f"  Edges:            {metadata.get('num_edges', 'N/A')}")
    print(f"  k-neighbors:      {k}")
    print(f"  Min overlap:      {min_overlap}")
    print(f"  Metric:           {metric}")
    print(f"  Output:           {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
