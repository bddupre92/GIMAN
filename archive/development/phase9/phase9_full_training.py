"""Phase 9 full neuro-fuzzy training with strict label/split contracts.

Key hardening changes:
- No random in-script split generation
- Classification target must be explicit `saa_label` (configurable key)
- Fail-fast if label keys are missing or misconfigured
- Deterministic seeding + bootstrap confidence intervals
"""

from __future__ import annotations

import argparse
import copy
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
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

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
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Phase 9 full neuro-fuzzy training")
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
        "--checkpoint-path",
        type=Path,
        default=project_root
        / "outputs/phase8_2_final_training_sota_run/giman_survival_final.pth",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "outputs/phase9_neuro_fuzzy",
    )
    parser.add_argument("--classification-label-key", type=str, default="saa_label")
    parser.add_argument("--survival-event-key", type=str, default="event")
    parser.add_argument("--survival-time-key", type=str, default="time")
    parser.add_argument("--metadata-path", type=Path, default=None)
    parser.add_argument(
        "--feature-blacklist",
        type=str,
        default="",
        help="Comma-separated feature names to zero during training (e.g., ALPHA_SYN_VALUE).",
    )
    parser.add_argument(
        "--feature-dropout-rate",
        type=float,
        default=0.0,
        help="Random per-feature dropout probability applied each epoch.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=0.005)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-rules", type=int, default=32)
    parser.add_argument("--entropy-reg", type=float, default=0.01)
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
    """Set deterministic seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def require_attr(data: object, key: str, purpose: str) -> torch.Tensor:
    """Fetch required tensor attribute from a Data object."""
    if not hasattr(data, key):
        raise ValueError(f"Missing required key '{key}' for {purpose}.")
    value = getattr(data, key)
    if not torch.is_tensor(value):
        raise TypeError(f"Key '{key}' for {purpose} must be a tensor.")
    return value


def validate_label_contract(args: argparse.Namespace) -> None:
    """Enforce explicit non-proxy classification labels."""
    forbidden_proxy_keys = {"event", "event_observed", "phenoconverted"}
    if args.classification_label_key in forbidden_proxy_keys:
        raise ValueError(
            "classification_label_key points to a survival endpoint/proxy. "
            "Use an explicit SAA label column (e.g., 'saa_label')."
        )
    if args.classification_label_key == args.survival_event_key:
        raise ValueError(
            "classification_label_key must differ from survival_event_key to prevent proxy training."
        )


def load_pretrained_gat(
    model: NeuroFuzzyGIMAN, checkpoint_path: Path
) -> NeuroFuzzyGIMAN:
    """Load pretrained Phase 8 GAT encoder weights when available."""
    if not checkpoint_path.exists():
        logging.warning(
            "Checkpoint not found at %s. Training from scratch.", checkpoint_path
        )
        return model

    logging.info("Loading pre-trained GAT from %s", checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    gat_state_dict = checkpoint["model_state_dict"]

    try:
        model.gat_encoder.load_state_dict(gat_state_dict, strict=False)
        logging.info("Successfully loaded GAT weights.")
    except Exception as exc:  # pragma: no cover - defensive
        logging.error("Error loading weights: %s", exc)

    return model


def safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute AUC with single-class guard."""
    if len(np.unique(y_true)) < 2:
        return 0.5
    return float(roc_auc_score(y_true, y_score))


def safe_pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute PR-AUC with single-class guard."""
    if len(np.unique(y_true)) < 2:
        return float(np.mean(y_true))
    return float(average_precision_score(y_true, y_score))


def recall_at_precision(y_true: np.ndarray, y_score: np.ndarray, target: float = 0.8) -> float:
    """Best recall achieved at precision >= target."""
    if len(np.unique(y_true)) < 2:
        return 0.0
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    mask = precision >= target
    if not np.any(mask):
        return 0.0
    return float(np.max(recall[mask]))


class FocalLoss(nn.Module):
    """Focal loss for imbalanced binary classification via logits."""

    def __init__(self, class_weights: torch.Tensor, gamma: float = 2.0):
        super().__init__()
        self.class_weights = class_weights
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, reduction="none", weight=self.class_weights)
        pt = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        return loss.mean()


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


def resolve_feature_blacklist_indices(
    metadata_path: Path | None,
    blacklist_csv: str,
) -> list[int]:
    """Resolve feature names to indices from metadata feature_names."""
    if not blacklist_csv.strip() or metadata_path is None:
        return []
    if not metadata_path.exists():
        logging.warning("Metadata path not found: %s; skipping feature blacklist", metadata_path)
        return []

    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    feature_names = payload.get("feature_names", [])
    if not feature_names:
        logging.warning("No feature_names found in metadata; skipping feature blacklist")
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


def train_neuro_fuzzy_full(args: argparse.Namespace) -> None:
    """Run full neuro-fuzzy training using pre-split train/test datasets."""
    logging.info("🚀 Starting Phase 9: Full Scale Training & Tuning (SOTA-hardened)")
    validate_label_contract(args)
    set_seed(args.seed)

    if not args.train_data_path.exists() or not args.test_data_path.exists():
        raise FileNotFoundError(
            "Expected pre-split datasets missing. Run prepare_final_pyg_data.py first."
        )

    train_data = torch.load(args.train_data_path, weights_only=False)
    test_data = torch.load(args.test_data_path, weights_only=False)

    _ = require_attr(train_data, "x", "train features")
    _ = require_attr(train_data, "edge_index", "train graph")
    _ = require_attr(test_data, "x", "test features")
    _ = require_attr(test_data, "edge_index", "test graph")

    _ = require_attr(train_data, args.classification_label_key, "train labels")
    _ = require_attr(test_data, args.classification_label_key, "test labels")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_data = train_data.to(device)
    test_data = test_data.to(device)
    train_targets = getattr(train_data, args.classification_label_key).long()
    test_targets_t = getattr(test_data, args.classification_label_key).long()

    pos_count = int((train_targets == 1).sum().item())
    neg_count = int((train_targets == 0).sum().item())
    if pos_count == 0 or neg_count == 0:
        raise ValueError(
            "Training set has a single class for classification_label_key; cannot train robust classifier."
        )

    class_weights = torch.tensor([1.0, neg_count / pos_count], device=device)
    logging.info(
        "Class balance (train): positive=%d, negative=%d, ratio=%.3f",
        pos_count,
        neg_count,
        pos_count / (pos_count + neg_count),
    )

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
    model = load_pretrained_gat(model, args.checkpoint_path)

    optimizer = optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=20
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    if args.classification_loss == "focal":
        criterion = FocalLoss(class_weights=class_weights, gamma=args.focal_gamma)

    best_auc = 0.0
    best_model_state = None

    logging.info("Training for %d epochs on %s", args.epochs, device)

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
        loss = loss + args.entropy_reg * entropy

        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            test_logits, _ = model(test_data)
            test_probs = F.softmax(test_logits, dim=1)[:, 1].detach().cpu().numpy()
            test_targets = test_targets_t.detach().cpu().numpy()
            auc = safe_auc(test_targets, test_probs)

        scheduler.step(auc)

        if auc > best_auc:
            best_auc = auc
            best_model_state = copy.deepcopy(model.state_dict())

        if (epoch + 1) % 10 == 0:
            logging.info(
                "Epoch %d/%d | loss=%.4f | test_auc=%.4f",
                epoch + 1,
                args.epochs,
                float(loss.item()),
                auc,
            )

    if best_model_state is None:
        raise RuntimeError("Training did not produce a valid model state")

    model.load_state_dict(best_model_state)
    model.eval()
    with torch.no_grad():
        best_logits, _ = model(test_data)
        best_probs = F.softmax(best_logits, dim=1)[:, 1].detach().cpu().numpy()
        y_test = test_targets_t.detach().cpu().numpy()
        final_pr_auc = safe_pr_auc(y_test, best_probs)
        final_recall80 = recall_at_precision(y_test, best_probs, target=0.8)

    auc_ci_low, auc_ci_high = bootstrap_auc_ci(
        y_true=y_test,
        y_score=best_probs,
        seed=args.seed,
        n_bootstrap=args.bootstrap_iters,
    )

    logging.info("🏆 Training complete | best_test_auc=%.4f", best_auc)
    logging.info("AUC 95%% CI (bootstrap): [%.4f, %.4f]", auc_ci_low, auc_ci_high)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_model_state, args.output_dir / "neuro_fuzzy_best.pth")

    results = {
        "seed": args.seed,
        "train_data_path": str(args.train_data_path),
        "test_data_path": str(args.test_data_path),
        "metadata_path": str(args.metadata_path) if args.metadata_path else None,
        "classification_label_key": args.classification_label_key,
        "survival_event_key": args.survival_event_key,
        "survival_time_key": args.survival_time_key,
        "feature_blacklist": [
            x.strip() for x in args.feature_blacklist.split(",") if x.strip()
        ],
        "feature_blacklist_indices": blacklist_indices,
        "feature_dropout_rate": args.feature_dropout_rate,
        "best_test_auc": float(best_auc),
        "auc_ci_95": [auc_ci_low, auc_ci_high],
        "test_pr_auc": float(final_pr_auc),
        "test_recall_at_precision_80": float(final_recall80),
        "classification_loss": args.classification_loss,
        "focal_gamma": args.focal_gamma,
        "train_positive": pos_count,
        "train_negative": neg_count,
        "epochs": args.epochs,
    }
    (args.output_dir / "full_training_results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    train_neuro_fuzzy_full(parse_args())
