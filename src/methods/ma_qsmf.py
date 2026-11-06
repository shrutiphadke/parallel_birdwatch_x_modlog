"""Evaluate the 2 x 2 MA-QSMF ablation.

``cn_inductive`` fixes rho without alignment; ``qsmf`` learns non-negative rho;
``ma_mf`` adds alignment with fixed rho; ``ma_qsmf`` combines both. TRAIN fits,
VAL selects lambda_y and the threshold, and TEST supplies metrics.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import logging
import math
from pathlib import Path
import random
import sys
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
import torch
from torch import nn
from torch.nn import functional as F

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.layout import REDDIT  # noqa: E402
from src.utils.splits import discover_splits, load_vote_partitions  # noqa: E402
from src.utils.threshold_search import per_class_f1_over_thresholds  # noqa: E402

log = logging.getLogger(__name__)

VARIANTS = ("cn_inductive", "qsmf", "ma_mf", "ma_qsmf")
ALIGNED_VARIANTS = {"ma_mf", "ma_qsmf"}
LEARNED_RHO_VARIANTS = {"qsmf", "ma_qsmf"}


@dataclass(frozen=True)
class ModelConfig:
    """Hyperparameters shared by every split and ablation."""

    learning_rate: float = 0.03
    rho_learning_rate_scale: float = 0.1
    base_epochs: int = 300
    aligned_epochs: int = 150
    patience: int = 35
    tolerance: float = 1e-7
    lambda_user_bias: float = 0.05
    lambda_quality: float = 0.05
    lambda_user_view: float = 0.05
    lambda_item_view: float = 0.05
    lambda_rho: float = 0.05
    lambda_classifier: float = 0.01
    seed: int = 10
    device: str = "cpu"


@dataclass
class EncodedTrain:
    """TRAIN tensors and deterministic identifier mappings."""

    frame: pd.DataFrame
    users: list[str]
    items: list[str]
    user_to_idx: dict[str, int]
    item_to_idx: dict[str, int]
    user_idx: torch.Tensor
    item_idx: torch.Tensor
    votes: torch.Tensor
    label_item_idx: torch.Tensor
    labels01: torch.Tensor
    item_labels: pd.DataFrame


class MAQSMF(nn.Module):
    """One-dimensional quality and viewpoint matrix-factorization model."""

    def __init__(self, n_users: int, n_items: int, learn_rho: bool, seed: int) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.learn_rho = learn_rho
        self.mu = nn.Parameter(torch.zeros(()))
        self.user_bias = nn.Parameter(torch.zeros(n_users))
        self.user_view = nn.Parameter(torch.randn(n_users, generator=generator) * 0.02)
        self.item_quality = nn.Parameter(torch.zeros(n_items))
        self.item_view = nn.Parameter(torch.randn(n_items, generator=generator) * 0.02)
        if learn_rho:
            # softplus(inverse_softplus(1)) = 1, up to floating-point error.
            initial_raw = math.log(math.expm1(1.0))
            self.rho_raw = nn.Parameter(torch.full((n_users,), initial_raw))
        else:
            self.register_buffer("fixed_rho", torch.ones(n_users))
        self.classifier_weight = nn.Parameter(torch.ones(()))
        self.classifier_bias = nn.Parameter(torch.zeros(()))

    def rho(self) -> torch.Tensor:
        """Return non-negative user quality sensitivities."""
        if self.learn_rho:
            return F.softplus(self.rho_raw)
        return self.fixed_rho

    def reconstruct(self, user_idx: torch.Tensor, item_idx: torch.Tensor) -> torch.Tensor:
        """Reconstruct individual votes."""
        return (
            self.mu
            + self.user_bias[user_idx]
            + self.rho()[user_idx] * self.item_quality[item_idx]
            + self.user_view[user_idx] * self.item_view[item_idx]
        )

    def item_logits(self, item_idx: torch.Tensor) -> torch.Tensor:
        """Predict moderator labels from the quality dimension."""
        return self.classifier_weight * self.item_quality[item_idx] + self.classifier_bias


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, default=_json_default, allow_nan=False)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _validate_partition(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    required = {"item_id", "username", "vote", "label"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{name} is missing columns: {sorted(missing)}")

    result = frame.loc[:, ["item_id", "username", "vote", "label"]].copy()
    if result.isna().any().any():
        counts = result.isna().sum()
        bad = {key: int(value) for key, value in counts.items() if value}
        raise ValueError(f"{name} contains missing required values: {bad}")
    result["item_id"] = result["item_id"].astype(str)
    result["username"] = result["username"].astype(str)
    result["vote"] = pd.to_numeric(result["vote"], errors="raise").astype(float)
    result["label"] = pd.to_numeric(result["label"], errors="raise").astype(int)
    if not set(result["vote"].unique()).issubset({-1.0, 1.0}):
        raise ValueError(f"{name}.vote must contain only -1 and +1")
    if not set(result["label"].unique()).issubset({-1, 1}):
        raise ValueError(f"{name}.label must contain only -1 and +1")
    per_item_labels = result.groupby("item_id", sort=False)["label"].nunique()
    if (per_item_labels != 1).any():
        examples = per_item_labels[per_item_labels != 1].index[:5].tolist()
        raise ValueError(f"{name} has inconsistent labels for items: {examples}")
    return result.reset_index(drop=True)


def encode_train(train: pd.DataFrame, device: torch.device) -> EncodedTrain:
    """Validate and encode TRAIN without inspecting any other partition."""
    frame = _validate_partition(train, "TRAIN")
    users = sorted(frame["username"].unique().tolist())
    items = sorted(frame["item_id"].unique().tolist())
    user_to_idx = {value: idx for idx, value in enumerate(users)}
    item_to_idx = {value: idx for idx, value in enumerate(items)}
    user_idx_np = frame["username"].map(user_to_idx).to_numpy(dtype=np.int64, copy=True)
    item_idx_np = frame["item_id"].map(item_to_idx).to_numpy(dtype=np.int64, copy=True)

    item_labels = (
        frame.drop_duplicates("item_id")[["item_id", "label"]]
        .sort_values("item_id")
        .reset_index(drop=True)
    )
    label_item_idx_np = item_labels["item_id"].map(item_to_idx).to_numpy(
        dtype=np.int64, copy=True
    )
    labels01_np = (item_labels["label"].to_numpy(dtype=np.int64, copy=True) == 1).astype(
        np.float32
    )
    return EncodedTrain(
        frame=frame,
        users=users,
        items=items,
        user_to_idx=user_to_idx,
        item_to_idx=item_to_idx,
        user_idx=torch.as_tensor(user_idx_np, dtype=torch.long, device=device),
        item_idx=torch.as_tensor(item_idx_np, dtype=torch.long, device=device),
        votes=torch.as_tensor(
            frame["vote"].to_numpy(dtype=np.float32), dtype=torch.float32, device=device
        ),
        label_item_idx=torch.as_tensor(label_item_idx_np, dtype=torch.long, device=device),
        labels01=torch.as_tensor(labels01_np, dtype=torch.float32, device=device),
        item_labels=item_labels,
    )


def _regularization(model: MAQSMF, config: ModelConfig, aligned: bool) -> torch.Tensor:
    penalty = (
        config.lambda_user_bias * model.user_bias.square().mean()
        + config.lambda_quality * model.item_quality.square().mean()
        + config.lambda_user_view * model.user_view.square().mean()
        + config.lambda_item_view * model.item_view.square().mean()
    )
    if model.learn_rho:
        penalty = penalty + config.lambda_rho * (model.rho() - 1.0).square().mean()
    if aligned:
        penalty = penalty + config.lambda_classifier * model.classifier_weight.square()
    return 0.5 * penalty


def _optimizer(model: MAQSMF, config: ModelConfig, aligned: bool) -> torch.optim.Optimizer:
    slow: list[nn.Parameter] = []
    regular: list[nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not aligned and name in {"classifier_weight", "classifier_bias"}:
            continue
        if name == "rho_raw":
            slow.append(parameter)
        else:
            regular.append(parameter)
    groups: list[dict[str, Any]] = [{"params": regular, "lr": config.learning_rate}]
    if slow:
        groups.append(
            {
                "params": slow,
                "lr": config.learning_rate * config.rho_learning_rate_scale,
            }
        )
    return torch.optim.Adam(groups)


def fit_model(
    model: MAQSMF,
    data: EncodedTrain,
    config: ModelConfig,
    *,
    aligned: bool,
    lambda_y: float,
    epochs: int,
) -> dict[str, Any]:
    """Fit on TRAIN losses only, with early stopping on TRAIN objective."""
    if aligned and lambda_y <= 0:
        raise ValueError("Aligned fitting requires lambda_y > 0")
    optimizer = _optimizer(model, config, aligned)
    best_loss = math.inf
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    last_parts: dict[str, float] = {}

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        reconstructed = model.reconstruct(data.user_idx, data.item_idx)
        vote_loss = 0.5 * F.mse_loss(reconstructed, data.votes)
        reg_loss = _regularization(model, config, aligned)
        label_loss = torch.zeros((), device=data.votes.device)
        if aligned:
            label_loss = F.binary_cross_entropy_with_logits(
                model.item_logits(data.label_item_idx), data.labels01
            )
        loss = vote_loss + reg_loss + lambda_y * label_loss
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite objective at epoch {epoch}")
        loss.backward()
        optimizer.step()

        current = float(loss.detach().cpu())
        last_parts = {
            "objective": current,
            "vote_loss": float(vote_loss.detach().cpu()),
            "regularization_loss": float(reg_loss.detach().cpu()),
            "label_bce": float(label_loss.detach().cpu()),
        }
        improvement = best_loss - current
        required = config.tolerance * max(1.0, abs(best_loss)) if math.isfinite(best_loss) else 0
        if improvement > required:
            best_loss = current
            # Keep the early-stopping snapshot on the active device.  Copying
            # every improving state to CPU forces a full GPU synchronization and
            # PCIe transfer on most epochs for large user/item tables.
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= config.patience:
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a finite model")
    model.load_state_dict(best_state)
    model.eval()
    last_parts.update({"best_objective": best_loss, "epochs_run": epoch, "early_stopped": epoch < epochs})
    return last_parts


def _fit_quality_classifier(model: MAQSMF, data: EncodedTrain) -> tuple[float, float]:
    """Fit the post-hoc TRAIN-only q-to-label map used by unaligned variants."""
    quality = model.item_quality.detach().cpu().numpy()[
        data.label_item_idx.detach().cpu().numpy()
    ].reshape(-1, 1)
    labels01 = data.labels01.detach().cpu().numpy().astype(int)
    if np.unique(labels01).size < 2 or float(np.std(quality)) < 1e-12:
        prevalence = float(np.clip(labels01.mean(), 1e-6, 1 - 1e-6))
        weight, bias = 0.0, math.log(prevalence / (1.0 - prevalence))
    else:
        classifier = LogisticRegression(C=1.0, solver="lbfgs", random_state=0)
        classifier.fit(quality, labels01)
        weight = float(classifier.coef_[0, 0])
        bias = float(classifier.intercept_[0])
    with torch.no_grad():
        model.classifier_weight.fill_(weight)
        model.classifier_bias.fill_(bias)
    return weight, bias


def _clone_model(source: MAQSMF, data: EncodedTrain, config: ModelConfig) -> MAQSMF:
    clone = MAQSMF(len(data.users), len(data.items), source.learn_rho, config.seed)
    clone.load_state_dict(source.state_dict())
    return clone.to(torch.device(config.device))


def _training_rmse(model: MAQSMF, data: EncodedTrain) -> float:
    with torch.no_grad():
        residual = model.reconstruct(data.user_idx, data.item_idx) - data.votes
        return float(torch.sqrt(residual.square().mean()).cpu())


def infer_new_items(
    frame: pd.DataFrame,
    model: MAQSMF,
    train: EncodedTrain,
    config: ModelConfig,
    partition_name: str,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    """Infer q,z for unseen items using votes only and frozen TRAIN parameters."""
    part = _validate_partition(frame, partition_name)
    overlap = set(part["item_id"]).intersection(train.item_to_idx)
    if overlap:
        examples = sorted(overlap)[:5]
        raise ValueError(
            f"{partition_name} contains {len(overlap)} TRAIN item IDs; examples: {examples}"
        )

    with torch.no_grad():
        mu = float(model.mu.detach().cpu())
        user_bias = model.user_bias.detach().cpu().numpy()
        rho = model.rho().detach().cpu().numpy()
        user_view = model.user_view.detach().cpu().numpy()
        classifier_weight = float(model.classifier_weight.detach().cpu())
        classifier_bias = float(model.classifier_bias.detach().cpu())

    # Convert mean-regularized training penalties into the equivalent per-item
    # ridge terms used in each new item's two-dimensional conditional problem.
    n_votes = max(len(train.frame), 1)
    n_items = max(len(train.items), 1)
    ridge_quality = config.lambda_quality * n_votes / n_items
    ridge_view = config.lambda_item_view * n_votes / n_items

    # Every unseen item is an independent two-parameter ridge problem.  Form all
    # normal equations with grouped bincount reductions and solve the closed-form
    # 2x2 systems in one vectorized pass instead of one Python groupby/solve per
    # item.  This is algebraically identical and is critical for large folds.
    item_names = sorted(part["item_id"].unique().tolist())
    item_lookup = {item_id: index for index, item_id in enumerate(item_names)}
    item_indices = part["item_id"].map(item_lookup).to_numpy(dtype=np.int64)
    user_indices = part["username"].map(train.user_to_idx).fillna(-1).to_numpy(dtype=np.int64)
    known = user_indices >= 0
    safe = np.maximum(user_indices, 0)
    biases = np.where(known, user_bias[safe], 0.0)
    sensitivities = np.where(known, rho[safe], 1.0)
    viewpoints = np.where(known, user_view[safe], 0.0)
    votes = part["vote"].to_numpy(dtype=float)
    response = votes - mu - biases
    n_items = len(item_names)

    def grouped(values: np.ndarray) -> np.ndarray:
        return np.bincount(item_indices, weights=values, minlength=n_items)

    normal_qq = grouped(sensitivities * sensitivities) + ridge_quality
    normal_qz = grouped(sensitivities * viewpoints)
    normal_zz = grouped(viewpoints * viewpoints) + ridge_view
    rhs_q = grouped(sensitivities * response)
    rhs_z = grouped(viewpoints * response)
    determinant = normal_qq * normal_zz - normal_qz * normal_qz
    stable = np.abs(determinant) > np.finfo(float).eps
    quality = np.divide(
        rhs_q * normal_zz - rhs_z * normal_qz,
        determinant,
        out=np.zeros(n_items, dtype=float),
        where=stable,
    )
    item_view = np.divide(
        normal_qq * rhs_z - normal_qz * rhs_q,
        determinant,
        out=np.zeros(n_items, dtype=float),
        where=stable,
    )
    # Positive ridge terms should make every system nonsingular; retain a
    # defensive least-squares fallback for pathological floating-point inputs.
    for index in np.flatnonzero(~stable):
        normal = np.asarray(
            [
                [normal_qq[index], normal_qz[index]],
                [normal_qz[index], normal_zz[index]],
            ]
        )
        quality[index], item_view[index] = np.linalg.lstsq(
            normal, np.asarray([rhs_q[index], rhs_z[index]]), rcond=None
        )[0]

    reconstructed = (
        mu
        + biases
        + sensitivities * quality[item_indices]
        + viewpoints * item_view[item_indices]
    )
    squared_errors = np.square(votes - reconstructed)
    logits = classifier_weight * quality + classifier_bias
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))
    counts = np.bincount(item_indices, minlength=n_items).astype(np.int64)
    known_counts = np.bincount(
        item_indices, weights=known.astype(np.int8), minlength=n_items
    ).astype(np.int64)
    label_lookup = part.drop_duplicates("item_id").set_index("item_id")["label"]
    result = pd.DataFrame(
        {
            "item_id": item_names,
            "label": [int(label_lookup[item_id]) for item_id in item_names],
            "quality": quality,
            "viewpoint": item_view,
            "moderator_logit": logits,
            "score": probabilities,
            "n_votes": counts,
            "n_known_rater_votes": known_counts,
            "known_rater_fraction": np.divide(
                known_counts,
                counts,
                out=np.zeros(n_items, dtype=float),
                where=counts > 0,
            ),
        }
    )
    known_votes_total = int(known.sum())
    diagnostics: dict[str, float | int] = {
        "n_items": int(len(result)),
        "n_votes": int(len(part)),
        "n_known_rater_votes": known_votes_total,
        "known_rater_vote_fraction": float(known_votes_total / len(part)) if len(part) else math.nan,
        "n_items_without_known_rater": int((result["n_known_rater_votes"] == 0).sum()),
        "vote_reconstruction_rmse": float(np.sqrt(np.mean(squared_errors)))
        if len(squared_errors)
        else math.nan,
        "ridge_quality": float(ridge_quality),
        "ridge_viewpoint": float(ridge_view),
    }
    return result, diagnostics


def calibrate_threshold(items: pd.DataFrame) -> tuple[float, pd.DataFrame]:
    """Select a probability threshold on VAL macro F1 only."""
    y_true = items["label"].to_numpy(dtype=int)
    scores = items["score"].to_numpy(dtype=float)
    grid = np.unique(np.concatenate([np.linspace(0.0, 1.0, 401), scores]))
    f1_approve, f1_remove = per_class_f1_over_thresholds(y_true, scores, grid)
    calibration = pd.DataFrame(
        {
            "threshold": grid.astype(float),
            "macro_f1": np.mean(np.vstack([f1_approve, f1_remove]), axis=0),
            "f1_approve": f1_approve,
            "f1_remove": f1_remove,
        }
    )
    calibration["distance_from_half"] = (calibration["threshold"] - 0.5).abs()
    best = calibration.sort_values(
        ["macro_f1", "f1_remove", "distance_from_half", "threshold"],
        ascending=[False, False, True, True],
    ).iloc[0]
    return float(best["threshold"]), calibration.drop(columns="distance_from_half")


def evaluate_items(items: pd.DataFrame, threshold: float) -> tuple[dict[str, Any], pd.DataFrame]:
    """Apply one frozen threshold and compute item-level metrics."""
    scored = items.copy()
    scored["prediction"] = np.where(scored["score"] >= threshold, 1, -1)
    y_true = scored["label"].to_numpy(dtype=int)
    y_pred = scored["prediction"].to_numpy(dtype=int)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[1, -1], average=None, zero_division=0
    )
    try:
        auc = float(roc_auc_score((y_true == 1).astype(int), scored["score"]))
    except ValueError:
        auc = math.nan
    metrics: dict[str, Any] = {
        "threshold": float(threshold),
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),
        "precision_approve": float(precision[0]),
        "recall_approve": float(recall[0]),
        "f1_approve": float(f1[0]),
        "precision_remove": float(precision[1]),
        "recall_remove": float(recall[1]),
        "f1_remove": float(f1[1]),
        # Aliases preserve compatibility with the existing benchmark reports.
        "precision_pos": float(precision[0]),
        "recall_pos": float(recall[0]),
        "f1_pos": float(f1[0]),
        "precision_neg": float(precision[1]),
        "recall_neg": float(recall[1]),
        "f1_neg": float(f1[1]),
        "roc_auc": auc,
        "n_items": int(len(scored)),
        "n_approve": int(support[0]),
        "n_remove": int(support[1]),
    }
    return metrics, scored


def _parameter_diagnostics(model: MAQSMF, data: EncodedTrain) -> dict[str, Any]:
    with torch.no_grad():
        rho = model.rho().detach().cpu().numpy()
        quality = model.item_quality.detach().cpu().numpy()
        item_view = model.item_view.detach().cpu().numpy()
        labels = data.item_labels.set_index("item_id").loc[data.items, "label"].to_numpy()
        classifier_weight = float(model.classifier_weight.detach().cpu())
        classifier_bias = float(model.classifier_bias.detach().cpu())
    return {
        "mu": float(model.mu.detach().cpu()),
        "classifier_weight": classifier_weight,
        "classifier_bias": classifier_bias,
        "rho_mean": float(np.mean(rho)),
        "rho_std": float(np.std(rho)),
        "rho_min": float(np.min(rho)),
        "rho_q05": float(np.quantile(rho, 0.05)),
        "rho_median": float(np.median(rho)),
        "rho_q95": float(np.quantile(rho, 0.95)),
        "rho_max": float(np.max(rho)),
        "rho_fraction_below_0_1": float(np.mean(rho < 0.1)),
        "quality_mean_approve": float(np.mean(quality[labels == 1])),
        "quality_mean_remove": float(np.mean(quality[labels == -1])),
        "quality_std": float(np.std(quality)),
        "viewpoint_std": float(np.std(item_view)),
        "train_vote_reconstruction_rmse": _training_rmse(model, data),
        "n_train_users": len(data.users),
        "n_train_items": len(data.items),
        "n_train_votes": len(data.frame),
    }


def _parameter_frames(model: MAQSMF, data: EncodedTrain) -> tuple[pd.DataFrame, pd.DataFrame]:
    with torch.no_grad():
        users = pd.DataFrame(
            {
                "username": data.users,
                "user_bias": model.user_bias.detach().cpu().numpy(),
                "rho": model.rho().detach().cpu().numpy(),
                "user_viewpoint": model.user_view.detach().cpu().numpy(),
            }
        )
        items = pd.DataFrame(
            {
                "item_id": data.items,
                "quality": model.item_quality.detach().cpu().numpy(),
                "viewpoint": model.item_view.detach().cpu().numpy(),
            }
        ).merge(data.item_labels, on="item_id", how="left", validate="one_to_one")
    return users, items


def _save_variant(
    output_dir: Path,
    model: MAQSMF,
    data: EncodedTrain,
    val_scored: pd.DataFrame,
    test_scored: pd.DataFrame,
    calibration: pd.DataFrame,
    metrics: dict[str, Any],
    candidates: pd.DataFrame | None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    users, train_items = _parameter_frames(model, data)
    users.to_parquet(output_dir / "user_params.parquet", index=False)
    train_items.to_parquet(output_dir / "train_item_params.parquet", index=False)
    val_scored.to_parquet(output_dir / "val_predictions.parquet", index=False)
    test_scored.to_parquet(output_dir / "test_predictions.parquet", index=False)
    calibration.to_parquet(output_dir / "val_threshold_calibration.parquet", index=False)
    if candidates is not None:
        candidates.to_csv(output_dir / "val_lambda_candidates.csv", index=False)
    _write_json(output_dir / "metrics.json", metrics)


def _evaluate_frozen_model(
    model: MAQSMF,
    train: EncodedTrain,
    val: pd.DataFrame,
    test: pd.DataFrame,
    config: ModelConfig,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    val_items, val_inference = infer_new_items(val, model, train, config, "VAL")
    threshold, calibration = calibrate_threshold(val_items)
    val_metrics, val_scored = evaluate_items(val_items, threshold)
    # This is the only point at which TEST is inferred/evaluated for a selected model.
    test_items, test_inference = infer_new_items(test, model, train, config, "TEST")
    test_metrics, test_scored = evaluate_items(test_items, threshold)
    metrics = {
        "threshold_selected_on": "VAL",
        "val": {**val_metrics, "inference": val_inference},
        "test": {**test_metrics, "inference": test_inference},
    }
    return metrics, val_scored, test_scored, calibration


def _select_aligned_model(
    base_model: MAQSMF,
    train: EncodedTrain,
    val: pd.DataFrame,
    config: ModelConfig,
    lambda_y_grid: Sequence[float],
) -> tuple[MAQSMF, float, pd.DataFrame, dict[str, Any]]:
    """Choose lambda_y using VAL; never evaluate candidate models on TEST."""
    records: list[dict[str, Any]] = []
    candidate_models: dict[float, MAQSMF] = {}
    candidate_training: dict[float, dict[str, Any]] = {}
    for lambda_y in lambda_y_grid:
        candidate = _clone_model(base_model, train, config)
        training = fit_model(
            candidate,
            train,
            config,
            aligned=True,
            lambda_y=float(lambda_y),
            epochs=config.aligned_epochs,
        )
        val_items, inference = infer_new_items(val, candidate, train, config, "VAL")
        threshold, _ = calibrate_threshold(val_items)
        val_metrics, _ = evaluate_items(val_items, threshold)
        records.append(
            {
                "lambda_y": float(lambda_y),
                "threshold": threshold,
                "val_macro_f1": val_metrics["macro_f1"],
                "val_f1_approve": val_metrics["f1_approve"],
                "val_f1_remove": val_metrics["f1_remove"],
                "val_roc_auc": val_metrics["roc_auc"],
                "val_vote_reconstruction_rmse": inference["vote_reconstruction_rmse"],
                "train_vote_reconstruction_rmse": _training_rmse(candidate, train),
                "epochs_run": training["epochs_run"],
                "best_objective": training["best_objective"],
            }
        )
        candidate_models[float(lambda_y)] = candidate
        candidate_training[float(lambda_y)] = training
    table = pd.DataFrame(records).sort_values("lambda_y").reset_index(drop=True)
    best = table.sort_values(
        ["val_macro_f1", "val_f1_remove", "lambda_y"], ascending=[False, False, True]
    ).iloc[0]
    selected_lambda = float(best["lambda_y"])
    return (
        candidate_models[selected_lambda],
        selected_lambda,
        table,
        candidate_training[selected_lambda],
    )


def evaluate_split(
    split_name: str,
    train_frame: pd.DataFrame,
    val_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    output_root: Path,
    config: ModelConfig,
    variants: Sequence[str],
    lambda_y_grid: Sequence[float],
) -> dict[str, Any]:
    """Run requested ablations on one prepared split."""
    log.info("[%s] encoding TRAIN", split_name)
    device = torch.device(config.device)
    train = encode_train(train_frame, device)
    val = _validate_partition(val_frame, "VAL")
    test = _validate_partition(test_frame, "TEST")
    overlap = set(train.frame["item_id"]) & (set(val["item_id"]) | set(test["item_id"]))
    if overlap:
        raise ValueError(f"{split_name}: item leakage across partitions ({len(overlap)} IDs)")

    requested = set(variants)
    need_fixed = bool(requested & {"cn_inductive", "ma_mf"})
    need_learned = bool(requested & {"qsmf", "ma_qsmf"})
    bases: dict[bool, tuple[MAQSMF, dict[str, Any]]] = {}
    for learn_rho in (False, True):
        if (learn_rho and not need_learned) or (not learn_rho and not need_fixed):
            continue
        label = "learned rho" if learn_rho else "fixed rho"
        log.info("[%s] fitting vote-only base (%s)", split_name, label)
        _seed_everything(config.seed)
        model = MAQSMF(len(train.users), len(train.items), learn_rho, config.seed).to(device)
        with torch.no_grad():
            model.mu.fill_(float(train.votes.mean().cpu()))
        training = fit_model(
            model, train, config, aligned=False, lambda_y=0.0, epochs=config.base_epochs
        )
        _fit_quality_classifier(model, train)
        bases[learn_rho] = (model, training)

    split_summary: dict[str, Any] = {}
    for variant in variants:
        learn_rho = variant in LEARNED_RHO_VARIANTS
        base_model, base_training = bases[learn_rho]
        candidates: pd.DataFrame | None = None
        if variant in ALIGNED_VARIANTS:
            log.info("[%s/%s] selecting lambda_y on VAL", split_name, variant)
            model, selected_lambda, candidates, training = _select_aligned_model(
                base_model, train, val, config, lambda_y_grid
            )
        else:
            log.info("[%s/%s] evaluating vote-only ablation", split_name, variant)
            model = base_model
            selected_lambda = 0.0
            training = base_training

        metrics, val_scored, test_scored, calibration = _evaluate_frozen_model(
            model, train, val, test, config
        )
        metrics.update(
            {
                "method": variant,
                "split": split_name,
                "learn_rho": learn_rho,
                "moderator_aligned": variant in ALIGNED_VARIANTS,
                "selected_lambda_y": selected_lambda,
                "selection_rule": "VAL macro_f1; tie: VAL f1_remove; tie: smaller lambda_y",
                "test_used_for_selection": False,
                "training": training,
                "parameters": _parameter_diagnostics(model, train),
                "config": asdict(config),
            }
        )
        variant_dir = output_root / split_name / "ma_qsmf" / variant
        _save_variant(
            variant_dir,
            model,
            train,
            val_scored,
            test_scored,
            calibration,
            metrics,
            candidates,
        )
        split_summary[variant] = metrics
        log.info(
            "[%s/%s] TEST macro-F1=%.4f F1-remove=%.4f AUC=%.4f",
            split_name,
            variant,
            metrics["test"]["macro_f1"],
            metrics["test"]["f1_remove"],
            metrics["test"]["roc_auc"],
        )
    return split_summary


def _selected_splits(
    available: dict[str, Path], requested: Iterable[str] | None
) -> dict[str, Path]:
    if not requested:
        return available
    names = list(requested)
    missing = sorted(set(names).difference(available))
    if missing:
        raise ValueError(f"Unknown splits {missing}; available: {sorted(available)}")
    return {name: available[name] for name in names}


def evaluate_all_splits(
    votes_dir: Path,
    output_dir: Path,
    config: ModelConfig,
    variants: Sequence[str],
    lambda_y_grid: Sequence[float],
    split_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Discover and evaluate all canonical benchmark splits by default."""
    invalid = sorted(set(variants).difference(VARIANTS))
    if invalid:
        raise ValueError(f"Unknown variants: {invalid}")
    if not variants:
        raise ValueError("At least one variant is required")
    if any(value <= 0 for value in lambda_y_grid):
        raise ValueError("Every lambda_y candidate must be positive")
    available = discover_splits(votes_dir)
    selected = _selected_splits(available, split_names)
    if not selected:
        raise FileNotFoundError(f"No prepared splits found below {votes_dir}")

    summary: dict[str, Any] = {
        "protocol": {
            "split_count": len(selected),
            "splits": list(selected),
            "variants": list(variants),
            "lambda_y_grid": [float(value) for value in lambda_y_grid],
            "fit_partitions": "TRAIN only",
            "selection_partition": "VAL only",
            "final_evaluation_partition": "TEST only",
            "new_item_inference": "q,z from item votes with frozen TRAIN parameters",
            "cold_start_user_prior": {"user_bias": 0.0, "rho": 1.0, "user_viewpoint": 0.0},
            "config": asdict(config),
        },
        "results": {},
    }
    log.info("Running %d splits x %d ablations", len(selected), len(variants))
    for split_name, split_dir in selected.items():
        train, val, test = load_vote_partitions(split_dir)
        summary["results"][split_name] = evaluate_split(
            split_name,
            train,
            val,
            test,
            output_dir,
            config,
            variants,
            lambda_y_grid,
        )
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inductive MA-QSMF and its 2x2 ablation on prepared Reddit splits"
    )
    parser.add_argument("--votes-dir", type=Path, default=REDDIT.splits)
    parser.add_argument("--output-dir", type=Path, default=REDDIT.results)
    parser.add_argument("--splits", nargs="+", help="Optional subset; default: all discovered splits")
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--lambda-y-grid", nargs="+", type=float, default=[0.01, 0.1, 1.0, 10.0])
    parser.add_argument("--learning-rate", type=float, default=ModelConfig.learning_rate)
    parser.add_argument(
        "--rho-learning-rate-scale", type=float, default=ModelConfig.rho_learning_rate_scale
    )
    parser.add_argument("--base-epochs", type=int, default=ModelConfig.base_epochs)
    parser.add_argument("--aligned-epochs", type=int, default=ModelConfig.aligned_epochs)
    parser.add_argument("--patience", type=int, default=ModelConfig.patience)
    parser.add_argument("--tolerance", type=float, default=ModelConfig.tolerance)
    parser.add_argument("--lambda-user-bias", type=float, default=ModelConfig.lambda_user_bias)
    parser.add_argument("--lambda-quality", type=float, default=ModelConfig.lambda_quality)
    parser.add_argument("--lambda-user-view", type=float, default=ModelConfig.lambda_user_view)
    parser.add_argument("--lambda-item-view", type=float, default=ModelConfig.lambda_item_view)
    parser.add_argument("--lambda-rho", type=float, default=ModelConfig.lambda_rho)
    parser.add_argument(
        "--lambda-classifier", type=float, default=ModelConfig.lambda_classifier
    )
    parser.add_argument("--seed", type=int, default=ModelConfig.seed)
    parser.add_argument("--device", default=ModelConfig.device, help="cpu, cuda, or cuda:N")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    args = _parse_args()
    config = ModelConfig(
        learning_rate=args.learning_rate,
        rho_learning_rate_scale=args.rho_learning_rate_scale,
        base_epochs=args.base_epochs,
        aligned_epochs=args.aligned_epochs,
        patience=args.patience,
        tolerance=args.tolerance,
        lambda_user_bias=args.lambda_user_bias,
        lambda_quality=args.lambda_quality,
        lambda_user_view=args.lambda_user_view,
        lambda_item_view=args.lambda_item_view,
        lambda_rho=args.lambda_rho,
        lambda_classifier=args.lambda_classifier,
        seed=args.seed,
        device=args.device,
    )
    evaluate_all_splits(
        votes_dir=args.votes_dir,
        output_dir=args.output_dir,
        config=config,
        variants=args.variants,
        lambda_y_grid=args.lambda_y_grid,
        split_names=args.splits,
    )


if __name__ == "__main__":
    main()
