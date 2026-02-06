"""Phase 9 quick neuro-fuzzy verification with strict label contracts."""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score, roc_auc_score

# Add project root to sys.path
project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root))

# Import GIMAN components
sys.path.append(
    str(project_root / "archive/development/phase8/subphase8_2_dynamic_endpoints")
)
from train_final_giman_survival import GIMANSurvivalGAT

from archive.development.phase9.neuro_fuzzy import NeuroFuzzyGIMAN

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def parse_args() -> argparse.Namespace:
    """Parse CLI args."""
    parser = argparse.ArgumentParser(
        description="Phase 9 quick neuro-fuzzy verification"
    )
    parser.add_argument(
        "--train-data-path",
        type=Path,
        default=project_root / "data/03_prodromal/final_pyg_data/train_data.pt",
    )
    parser.add_argument(
        "--test-data-path",
        type=Path,
        default=project_root / "data/03_prodromal/final_pyg_data/test_data.pt",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "outputs/phase9_neuro_fuzzy",
    )
    parser.add_argument("--classification-label-key", type=str, default="saa_label")
    parser.add_argument("--survival-event-key", type=str, default="event")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-rules", type=int, default=16)
    parser.add_argument("--bootstrap-iters", type=int, default=500)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    """Set deterministic seed state."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def require_attr(data: object, key: str, purpose: str) -> torch.Tensor:
    """Load a required tensor attr."""
    if not hasattr(data, key):
        raise ValueError(f"Missing required key '{key}' for {purpose}.")
    value = getattr(data, key)
    if not torch.is_tensor(value):
        raise TypeError(f"Key '{key}' for {purpose} must be a tensor.")
    return value


def validate_label_contract(args: argparse.Namespace) -> None:
    """Disallow proxy-label classification."""
    forbidden_proxy_keys = {"event", "event_observed", "phenoconverted"}
    if args.classification_label_key in forbidden_proxy_keys:
        raise ValueError(
            "classification_label_key points to survival endpoint/proxy. Use explicit SAA labels."
        )
    if args.classification_label_key == args.survival_event_key:
        raise ValueError(
            "classification_label_key must differ from survival_event_key."
        )


def safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute AUC with guard."""
    if len(np.unique(y_true)) < 2:
        return 0.5
    return float(roc_auc_score(y_true, y_score))


def bootstrap_auc_ci(
    y_true: np.ndarray, y_score: np.ndarray, seed: int, n_bootstrap: int
) -> tuple[float, float]:
    """Bootstrap 95% CI for AUC."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    vals: list[float] = []

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        y_b = y_true[idx]
        if len(np.unique(y_b)) < 2:
            continue
        vals.append(float(roc_auc_score(y_b, y_score[idx])))

    if not vals:
        return (0.5, 0.5)

    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def train_neuro_fuzzy(args: argparse.Namespace) -> None:
    """Quick neuro-fuzzy verification on pre-split train/test data."""
    logging.info(
        "🚀 Starting Phase 9: Neuro-Fuzzy Enhancement (SOTA-hardened quick run)"
    )
    validate_label_contract(args)
    set_seed(args.seed)

    if not args.train_data_path.exists() or not args.test_data_path.exists():
        raise FileNotFoundError(
            "Expected pre-split datasets missing. Run prepare_final_pyg_data.py first."
        )

    train_data = torch.load(args.train_data_path, weights_only=False)
    test_data = torch.load(args.test_data_path, weights_only=False)

    _ = require_attr(
        train_data, args.classification_label_key, "train classification labels"
    )
    _ = require_attr(
        test_data, args.classification_label_key, "test classification labels"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_data = train_data.to(device)
    test_data = test_data.to(device)
    train_targets = getattr(train_data, args.classification_label_key).long()
    test_targets_t = getattr(test_data, args.classification_label_key).long()

    in_features = train_data.x.shape[1]
    gat_encoder = GIMANSurvivalGAT(in_features=in_features, hidden_dim=128)
    model = NeuroFuzzyGIMAN(gat_encoder, num_classes=2, num_rules=args.num_rules).to(
        device
    )

    optimizer = optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    criterion = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()

        logits, weights = model(train_data)
        loss = criterion(logits, train_targets)

        entropy = -torch.sum(weights * torch.log(weights + 1e-6), dim=1).mean()
        loss = loss + 0.01 * entropy

        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            logging.info(
                "Epoch %d/%d, Loss: %.4f", epoch + 1, args.epochs, float(loss.item())
            )

    model.eval()
    with torch.no_grad():
        logits_test, _ = model(test_data)
        probs = F.softmax(logits_test, dim=1)[:, 1].detach().cpu().numpy()
        y_test = test_targets_t.detach().cpu().numpy()

    auc = safe_auc(y_test, probs)
    acc = float(accuracy_score(y_test, probs > 0.5))
    auc_ci_low, auc_ci_high = bootstrap_auc_ci(
        y_true=y_test,
        y_score=probs,
        seed=args.seed,
        n_bootstrap=args.bootstrap_iters,
    )

    logging.info("✅ Phase 9 verification complete")
    logging.info("SAA AUC: %.4f (95%% CI [%.4f, %.4f])", auc, auc_ci_low, auc_ci_high)
    logging.info("Accuracy: %.4f", acc)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = {
        "seed": args.seed,
        "classification_label_key": args.classification_label_key,
        "saa_auc": auc,
        "saa_auc_ci_95": [auc_ci_low, auc_ci_high],
        "accuracy": acc,
        "epochs": args.epochs,
    }
    (args.output_dir / "quick_neuro_fuzzy_results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    train_neuro_fuzzy(parse_args())
