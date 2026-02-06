"""Phase 9 multi-task (survival + SAA) training with strict contracts."""

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
from sklearn.metrics import roc_auc_score

# Add project root to sys.path
project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root))

sys.path.append(
    str(project_root / "archive/development/phase8/subphase8_2_dynamic_endpoints")
)
from train_final_giman_survival import (
    GIMANSurvivalGAT,
    concordance_index,
    cox_partial_likelihood_loss,
)

from archive.development.phase9.neuro_fuzzy import MultiTaskNeuroFuzzyGIMAN

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def parse_args() -> argparse.Namespace:
    """Parse CLI args."""
    parser = argparse.ArgumentParser(
        description="Phase 9 multi-task neuro-fuzzy training"
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.005)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-rules", type=int, default=32)
    parser.add_argument("--classification-loss-weight", type=float, default=1.0)
    parser.add_argument("--bootstrap-iters", type=int, default=500)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    """Set deterministic random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def require_attr(data: object, key: str, purpose: str) -> torch.Tensor:
    """Fetch required tensor attribute from Data."""
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
            "classification_label_key points to survival endpoint/proxy. Use explicit SAA labels."
        )
    if args.classification_label_key == args.survival_event_key:
        raise ValueError(
            "classification_label_key must differ from survival_event_key."
        )


def load_pretrained_gat(
    model: MultiTaskNeuroFuzzyGIMAN, checkpoint_path: Path
) -> MultiTaskNeuroFuzzyGIMAN:
    """Load pretrained GAT weights if present."""
    if not checkpoint_path.exists():
        logging.warning(
            "Checkpoint not found at %s. Training from scratch.", checkpoint_path
        )
        return model

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    gat_state_dict = checkpoint["model_state_dict"]
    try:
        model.gat_encoder.load_state_dict(gat_state_dict, strict=False)
        logging.info("Loaded pretrained GAT weights")
    except Exception as exc:  # pragma: no cover - defensive
        logging.warning(
            "Could not load pretrained GAT weights: %s. Training from scratch.", exc
        )
    return model


def safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute AUC safely for potential single-class sets."""
    if len(np.unique(y_true)) < 2:
        return 0.5
    return float(roc_auc_score(y_true, y_score))


def safe_c_index(risk: np.ndarray, time: np.ndarray, event: np.ndarray) -> float:
    """Compute c-index with defensive fallback."""
    if len(risk) < 2:
        return 0.5
    return float(concordance_index(risk, time, event))


def bootstrap_ci(
    metric_fn,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    seed: int,
    n_bootstrap: int,
    *extra_arrays: np.ndarray,
) -> tuple[float, float]:
    """Generic bootstrap CI helper."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    vals: list[float] = []

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        boot_true = y_true[idx]
        if len(np.unique(boot_true)) < 2:
            continue
        boot_pred = y_pred[idx]
        boot_extra = [arr[idx] for arr in extra_arrays]
        vals.append(float(metric_fn(boot_pred, boot_true, *boot_extra)))

    if not vals:
        return (0.5, 0.5)
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def _auc_metric(pred: np.ndarray, true: np.ndarray, *_unused) -> float:
    return safe_auc(true, pred)


def _cindex_metric(risk: np.ndarray, event: np.ndarray, time: np.ndarray) -> float:
    return safe_c_index(risk, time, event)


def train_multitask(args: argparse.Namespace) -> None:
    """Train multi-task neuro-fuzzy model on canonical pre-split datasets."""
    logging.info(
        "🚀 Starting Phase 9: Multi-Task Learning Verification (SOTA-hardened)"
    )
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

    _ = require_attr(
        train_data, args.classification_label_key, "train classification labels"
    )
    _ = require_attr(
        test_data, args.classification_label_key, "test classification labels"
    )
    _ = require_attr(train_data, args.survival_event_key, "train survival event")
    _ = require_attr(train_data, args.survival_time_key, "train survival time")
    _ = require_attr(test_data, args.survival_event_key, "test survival event")
    _ = require_attr(test_data, args.survival_time_key, "test survival time")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_data = train_data.to(device)
    test_data = test_data.to(device)
    train_class_targets = getattr(train_data, args.classification_label_key).long()
    test_class_targets_t = getattr(test_data, args.classification_label_key).long()
    train_event = getattr(train_data, args.survival_event_key).long()
    train_time = getattr(train_data, args.survival_time_key).float()
    test_event_t = getattr(test_data, args.survival_event_key).long()
    test_time_t = getattr(test_data, args.survival_time_key).float()
    pos_count = int((train_class_targets == 1).sum().item())
    neg_count = int((train_class_targets == 0).sum().item())
    if pos_count == 0 or neg_count == 0:
        raise ValueError("Training set has one class for classification_label_key")

    in_features = train_data.x.shape[1]
    gat_encoder = GIMANSurvivalGAT(in_features=in_features, hidden_dim=128)
    model = MultiTaskNeuroFuzzyGIMAN(
        gat_encoder, num_classes=2, num_rules=args.num_rules
    ).to(device)
    model = load_pretrained_gat(model, args.checkpoint_path)

    optimizer = optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    class_weights = torch.tensor([1.0, neg_count / pos_count], device=device)
    classification_criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_auc = 0.0
    best_c_index = 0.0

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()

        logits_train, risk_train, _ = model(train_data)

        surv_loss = cox_partial_likelihood_loss(risk_train, train_time, train_event)
        class_loss = classification_criterion(logits_train, train_class_targets)
        loss = surv_loss + args.classification_loss_weight * class_loss

        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                logits_test, risk_test, _ = model(test_data)

                probs = F.softmax(logits_test, dim=1)[:, 1].detach().cpu().numpy()
                y_cls = test_class_targets_t.detach().cpu().numpy()
                auc = safe_auc(y_cls, probs)

                risk_np = risk_test.detach().cpu().numpy()
                time_np = test_time_t.detach().cpu().numpy()
                event_np = test_event_t.detach().cpu().numpy()
                c_index = safe_c_index(risk_np, time_np, event_np)

                best_auc = max(best_auc, auc)
                best_c_index = max(best_c_index, c_index)

            logging.info(
                "Epoch %d/%d | total=%.4f | surv=%.4f | class=%.4f | test_auc=%.4f | test_cindex=%.4f",
                epoch + 1,
                args.epochs,
                float(loss.item()),
                float(surv_loss.item()),
                float(class_loss.item()),
                auc,
                c_index,
            )

    model.eval()
    with torch.no_grad():
        logits_test, risk_test, _ = model(test_data)
        probs = F.softmax(logits_test, dim=1)[:, 1].detach().cpu().numpy()
        y_cls = test_class_targets_t.detach().cpu().numpy()
        risk_np = risk_test.detach().cpu().numpy()
        time_np = test_time_t.detach().cpu().numpy()
        event_np = test_event_t.detach().cpu().numpy()

    auc_ci_low, auc_ci_high = bootstrap_ci(
        _auc_metric,
        y_true=y_cls,
        y_pred=probs,
        seed=args.seed,
        n_bootstrap=args.bootstrap_iters,
    )
    c_ci_low, c_ci_high = bootstrap_ci(
        _cindex_metric,
        event_np,
        risk_np,
        args.seed,
        args.bootstrap_iters,
        time_np,
    )

    final_auc = safe_auc(y_cls, probs)
    final_c_index = safe_c_index(risk_np, time_np, event_np)

    logging.info("✅ Multi-task verification complete")
    logging.info(
        "Final SAA AUC: %.4f (95%% CI [%.4f, %.4f])", final_auc, auc_ci_low, auc_ci_high
    )
    logging.info(
        "Final C-index: %.4f (95%% CI [%.4f, %.4f])",
        final_c_index,
        c_ci_low,
        c_ci_high,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = {
        "seed": args.seed,
        "classification_label_key": args.classification_label_key,
        "survival_event_key": args.survival_event_key,
        "survival_time_key": args.survival_time_key,
        "final_saa_auc": final_auc,
        "final_saa_auc_ci_95": [auc_ci_low, auc_ci_high],
        "final_c_index": final_c_index,
        "final_c_index_ci_95": [c_ci_low, c_ci_high],
        "best_saa_auc_seen": best_auc,
        "best_c_index_seen": best_c_index,
        "epochs": args.epochs,
    }
    (args.output_dir / "multitask_training_results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    train_multitask(parse_args())
