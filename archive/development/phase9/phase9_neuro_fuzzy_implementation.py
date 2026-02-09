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
    parser.add_argument("--metadata-path", type=Path, default=None)
    parser.add_argument(
        "--feature-blacklist",
        type=str,
        default="",
        help="Comma-separated feature names to zero during training/eval.",
    )
    parser.add_argument(
        "--feature-dropout-rate",
        type=float,
        default=0.0,
        help="Random per-feature dropout probability applied each epoch.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-rules", type=int, default=16)
    parser.add_argument("--bootstrap-iters", type=int, default=500)
    parser.add_argument(
        "--classification-loss",
        type=str,
        default="cross_entropy",
        choices=["cross_entropy", "focal"],
    )
    parser.add_argument("--focal-gamma", type=float, default=2.0)
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


class FocalLoss(nn.Module):
    """Focal loss for imbalanced binary classification logits."""

    def __init__(self, class_weights: torch.Tensor, gamma: float = 2.0):
        super().__init__()
        self.class_weights = class_weights
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(
            logits, targets, reduction="none", weight=self.class_weights
        )
        pt = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        return loss.mean()


def resolve_feature_blacklist_indices(
    metadata_path: Path | None,
    blacklist_csv: str,
) -> list[int]:
    """Resolve blacklist feature names to indices from metadata."""
    if not blacklist_csv.strip() or metadata_path is None:
        return []
    if not metadata_path.exists():
        logging.warning(
            "Metadata path not found: %s; skipping feature blacklist", metadata_path
        )
        return []
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    feature_names = payload.get("feature_names", [])
    if not feature_names:
        return []

    requested = [x.strip() for x in blacklist_csv.split(",") if x.strip()]
    idxs: list[int] = []
    missing: list[str] = []
    for name in requested:
        if name in feature_names:
            idxs.append(int(feature_names.index(name)))
        else:
            missing.append(name)
    if missing:
        logging.warning("Requested blacklist features not found: %s", missing)
    return sorted(set(idxs))


def apply_feature_regularization(
    x: torch.Tensor,
    feature_dropout_rate: float,
    blacklist_indices: list[int],
) -> torch.Tensor:
    """Apply deterministic blacklist zeroing + stochastic feature dropout."""
    x_reg = x.clone()
    if blacklist_indices:
        x_reg[:, blacklist_indices] = 0.0
    if feature_dropout_rate > 0:
        drop = torch.rand((x_reg.shape[1],), device=x_reg.device) < feature_dropout_rate
        x_reg[:, drop] = 0.0
    return x_reg


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
    pos_count = int((train_targets == 1).sum().item())
    neg_count = int((train_targets == 0).sum().item())
    if pos_count == 0 or neg_count == 0:
        raise ValueError("Training set has one class for classification_label_key")

    in_features = train_data.x.shape[1]
    blacklist_indices = resolve_feature_blacklist_indices(
        metadata_path=args.metadata_path,
        blacklist_csv=args.feature_blacklist,
    )
    if blacklist_indices:
        logging.info("Applying feature blacklist indices: %s", blacklist_indices)
    if args.feature_dropout_rate > 0:
        logging.info("Applying feature dropout rate: %.3f", args.feature_dropout_rate)

    gat_encoder = GIMANSurvivalGAT(in_features=in_features, hidden_dim=128)
    model = NeuroFuzzyGIMAN(gat_encoder, num_classes=2, num_rules=args.num_rules).to(
        device
    )

    optimizer = optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    class_weights = torch.tensor([1.0, neg_count / pos_count], device=device)
    criterion: nn.Module = nn.CrossEntropyLoss(weight=class_weights)
    if args.classification_loss == "focal":
        criterion = FocalLoss(class_weights=class_weights, gamma=args.focal_gamma)

    best_auc = 0.0

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()

        train_batch = train_data.clone()
        train_batch.x = apply_feature_regularization(
            train_batch.x,
            feature_dropout_rate=args.feature_dropout_rate,
            blacklist_indices=blacklist_indices,
        )
        logits, weights = model(train_batch)
        loss = criterion(logits, train_targets)

        entropy = -torch.sum(weights * torch.log(weights + 1e-6), dim=1).mean()
        loss = loss + 0.01 * entropy

        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                eval_batch = test_data.clone()
                eval_batch.x = apply_feature_regularization(
                    eval_batch.x,
                    feature_dropout_rate=0.0,
                    blacklist_indices=blacklist_indices,
                )
                logits_eval, _ = model(eval_batch)
                probs_eval = F.softmax(logits_eval, dim=1)[:, 1].detach().cpu().numpy()
                y_eval = test_targets_t.detach().cpu().numpy()
                eval_auc = safe_auc(y_eval, probs_eval)
                best_auc = max(best_auc, eval_auc)
            logging.info(
                "Epoch %d/%d, Loss: %.4f, Eval AUC: %.4f",
                epoch + 1,
                args.epochs,
                float(loss.item()),
                eval_auc,
            )

    model.eval()
    with torch.no_grad():
        test_batch = test_data.clone()
        test_batch.x = apply_feature_regularization(
            test_batch.x,
            feature_dropout_rate=0.0,
            blacklist_indices=blacklist_indices,
        )
        logits_test, _ = model(test_batch)
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
        "metadata_path": str(args.metadata_path) if args.metadata_path else None,
        "feature_blacklist": [
            x.strip() for x in args.feature_blacklist.split(",") if x.strip()
        ],
        "feature_blacklist_indices": blacklist_indices,
        "feature_dropout_rate": args.feature_dropout_rate,
        "saa_auc": auc,
        "saa_auc_ci_95": [auc_ci_low, auc_ci_high],
        "best_saa_auc_seen": best_auc,
        "accuracy": acc,
        "classification_loss": args.classification_loss,
        "focal_gamma": args.focal_gamma,
        "epochs": args.epochs,
    }
    (args.output_dir / "quick_neuro_fuzzy_results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    train_neuro_fuzzy(parse_args())
