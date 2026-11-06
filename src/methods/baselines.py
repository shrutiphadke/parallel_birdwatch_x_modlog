from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.layout import REDDIT, load_item_scores  # noqa: E402
from src.utils.splits import discover_splits, load_split_data  # noqa: E402
from src.utils.threshold_search import (  # noqa: E402
    best_threshold,
    macro_prf_over_thresholds,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

THRESHOLD_GRID = np.linspace(-200, 200, 800)
ALPHA_GRID = np.linspace(1.0, 5.0, 20)
MIN_SUB_VAL_ITEMS = 10
N_FOLDS = 5
REDDIT_SCORES_PATH = REDDIT.default_inputs().item_scores
CANONICAL_BASELINES = (
    "BL1_net_vote",
    "BL2_net_vote_alpha",
    "BL3_net_vote_alpha_subreddit",
    "BL4_reddit_score",
    "BL5_subreddit",
)


def _timestamp_date(value):
    if isinstance(value, pd.Timestamp):
        return value.date()
    return pd.to_datetime(value, unit="s", utc=True).date()


# Calibration sample size when validation is empty.
MIN_CAL_FALLBACK_ITEMS = 50


# Data loading
def load_all_votes(step1_dir: Path) -> pd.DataFrame:
    """Load all votes from its configured source."""
    splits_dir = step1_dir / "intersection"
    dfs = []
    for name in ("train_votes", "val_votes", "test_votes"):
        df = pd.read_parquet(splits_dir / f"{name}.parquet")
        dfs.append(df)
    all_votes = pd.concat(dfs, ignore_index=True)
    log.info(
        "Loaded all votes: %d votes | %d items | %d users",
        len(all_votes),
        all_votes["item_id"].nunique(),
        all_votes["username"].nunique(),
    )
    return all_votes


# Score: positive votes minus alpha times negative votes.
def compute_net_scores(
    votes_df: pd.DataFrame,
    alpha: float = 1.0,
) -> pd.DataFrame:
    """Compute net scores from the supplied data."""
    # Vectorized counts: a per-group Python lambda dominated runtime (~80%).
    agg = (
        votes_df.assign(
            n_up=(votes_df["vote"] == 1).astype("int64"),
            n_down=(votes_df["vote"] == -1).astype("int64"),
        )
        .groupby("item_id")
        .agg(community=("community", "first"), n_up=("n_up", "sum"), n_down=("n_down", "sum"))
        .reset_index()
    )
    agg["n_votes"] = agg["n_up"] + agg["n_down"]
    agg["score"] = agg["n_up"] - alpha * agg["n_down"]
    return agg


# Labels
def get_item_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Return item labels for the supplied input."""
    return df.drop_duplicates("item_id")[["item_id", "label"]].copy()


# Chronological folds
def make_chrono_folds(all_votes: pd.DataFrame, n_folds: int = N_FOLDS) -> List[pd.DataFrame]:
    """Create chrono folds from the supplied data."""
    item_times = (
        all_votes.groupby("item_id")["timestamp"]
        .min()
        .sort_values()
        .reset_index()
        .rename(columns={"timestamp": "first_vote"})
    )
    n_items = len(item_times)
    fold_size = n_items // n_folds
    folds: List[pd.DataFrame] = []
    for k in range(n_folds):
        start = k * fold_size
        end = (k + 1) * fold_size if k < n_folds - 1 else n_items
        fold_items = item_times.iloc[start:end]["item_id"].values
        fold_votes = all_votes[all_votes["item_id"].isin(fold_items)].copy()
        folds.append(fold_votes)
        log.info(
            "Fold %d: %d items | %d votes | %s to %s",
            k,
            len(fold_items),
            len(fold_votes),
            _timestamp_date(fold_votes["timestamp"].min())
            if "timestamp" in fold_votes.columns
            else "?",
            _timestamp_date(fold_votes["timestamp"].max())
            if "timestamp" in fold_votes.columns
            else "?",
        )
    return folds


# Threshold search
def _best_threshold(
    y_true: np.ndarray, scores: np.ndarray, grid: np.ndarray, default: float
) -> Tuple[float, float]:
    """First grid threshold with the strictly best macro-F1 (loop semantics)."""
    return best_threshold(y_true, scores, grid, default=default)


def calibrate_global_threshold(
    scores_df: pd.DataFrame,
    val_labels: pd.DataFrame,
    grid: np.ndarray = THRESHOLD_GRID,
) -> Tuple[float, pd.DataFrame]:
    """Calibrate global threshold on validation data."""
    merged = val_labels.merge(scores_df[["item_id", "score"]], on="item_id", how="inner")
    merged = merged.dropna(subset=["score"])
    y_true = merged["label"].values
    scores = merged["score"].values
    precision, recall, f1 = macro_prf_over_thresholds(y_true, scores, grid)
    cal_df = pd.DataFrame(
        {
            "threshold": np.asarray(grid, dtype=float),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    )
    best = cal_df.loc[cal_df["f1"].idxmax()]
    best_thr = float(best["threshold"])
    return best_thr, cal_df


# Alpha selection
def select_alpha(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    alpha_grid: np.ndarray = ALPHA_GRID,
    thr_grid: np.ndarray = THRESHOLD_GRID,
) -> float:
    """Select alpha using validation data."""
    train_labels = get_item_labels(train_df)
    val_labels = get_item_labels(val_df)
    best_alpha_f1, best_alpha = -1.0, 1.0

    for alpha in alpha_grid:
        train_scores = compute_net_scores(train_df, alpha=alpha)
        merged_tr = train_labels.merge(
            train_scores[["item_id", "score"]], on="item_id", how="left"
        )
        merged_tr["score"] = merged_tr["score"].fillna(0.0)
        y_tr, s_tr = merged_tr["label"].values, merged_tr["score"].values

        best_thr_tr, _ = _best_threshold(y_tr, s_tr, thr_grid, default=0.0)

        val_scores = compute_net_scores(
            pd.concat([train_df, val_df], ignore_index=True), alpha=alpha
        )
        merged_val = val_labels.merge(val_scores[["item_id", "score"]], on="item_id", how="left")
        merged_val["score"] = merged_val["score"].fillna(0.0)
        y_pred_val = np.where(merged_val["score"].values >= best_thr_tr, 1, -1)
        _, _, f_val, _ = precision_recall_fscore_support(
            merged_val["label"].values,
            y_pred_val,
            labels=[1, -1],
            average="macro",
            zero_division=0,
        )
        if f_val > best_alpha_f1:
            best_alpha_f1, best_alpha = f_val, alpha

    return best_alpha


# Community thresholds
def calibrate_per_subreddit(
    scores_df: pd.DataFrame,
    val_labels_full: pd.DataFrame,
    global_threshold: float,
    grid: np.ndarray = THRESHOLD_GRID,
    min_items: int = MIN_SUB_VAL_ITEMS,
) -> Dict[str, float]:
    """Calibrate per subreddit on validation data."""
    merged = val_labels_full.merge(scores_df[["item_id", "score"]], on="item_id", how="inner")
    merged = merged.dropna(subset=["score"])

    sub_thresholds: Dict[str, float] = {}
    for sub, group in merged.groupby("community"):
        n_classes = group["label"].nunique()
        class_counts = group["label"].value_counts()
        min_class_count = int(class_counts.min()) if len(class_counts) > 0 else 0
        if len(group) < min_items or n_classes < 2 or min_class_count < max(3, min_items // 3):
            sub_thresholds[sub] = global_threshold
            continue
        y_true = group["label"].values
        scores = group["score"].values
        sub_thresholds[sub], _ = _best_threshold(y_true, scores, grid, default=global_threshold)
    return sub_thresholds


# Fold evaluation
def evaluate_fold(
    scores_df: pd.DataFrame,
    test_df: pd.DataFrame,
    predictions: np.ndarray,
    auc_observed_only: bool = False,
    omit_missing_scores: bool = False,
) -> Dict:
    """Evaluate fold and return its metrics."""
    test_labels = get_item_labels(test_df)
    items_test = test_labels.merge(scores_df, on="item_id", how="left")
    n_test_items = int(len(items_test))
    n_missing_score = int(items_test["score"].isna().sum())
    observed_score = items_test["score"].notna()
    score_coverage = float(observed_score.mean()) if n_test_items else 0.0
    if omit_missing_scores:
        items_test = items_test.loc[observed_score].copy()
    else:
        items_test["score"] = items_test["score"].fillna(0.0)
    if len(predictions) != len(items_test):
        raise ValueError(
            f"Prediction count {len(predictions)} does not match evaluated items {len(items_test)}"
        )
    items_test["prediction"] = predictions
    y_true = items_test["label"].values
    y_pred = items_test["prediction"].values
    scores = items_test["score"].values

    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[1, -1], average="macro", zero_division=0
    )
    p1, r1, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[1], average="binary", zero_division=0
    )
    p_1, r_1, f_1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[-1], average="binary", pos_label=-1, zero_division=0
    )
    auc_mask = np.ones(len(items_test), dtype=bool)
    if auc_observed_only and not omit_missing_scores:
        auc_mask = observed_score.to_numpy()
    try:
        auc = float(roc_auc_score((y_true[auc_mask] == 1).astype(int), scores[auc_mask]))
    except ValueError:
        auc = float("nan")

    diagnostic_items = items_test.loc[auc_mask] if auc_observed_only else items_test
    mean_s_pos = float(diagnostic_items.loc[diagnostic_items["label"] == 1, "score"].mean())
    mean_s_neg = float(diagnostic_items.loc[diagnostic_items["label"] == -1, "score"].mean())

    native_true = y_true[auc_mask]
    native_pred = y_pred[auc_mask]
    native_macro_f1 = (
        float(
            f1_score(
                native_true,
                native_pred,
                average="macro",
                zero_division=0,
            )
        )
        if len(native_true)
        else float("nan")
    )
    native_f1_pos = (
        float(f1_score(native_true, native_pred, pos_label=1, zero_division=0))
        if len(native_true)
        else float("nan")
    )
    native_f1_neg = (
        float(f1_score(native_true, native_pred, pos_label=-1, zero_division=0))
        if len(native_true)
        else float("nan")
    )

    try:
        prec_curve, rec_curve, thr_curve = precision_recall_curve(
            (y_true[auc_mask] == 1).astype(int), scores[auc_mask]
        )
        pr_precision = prec_curve.tolist()
        pr_recall = rec_curve.tolist()
        pr_thresholds = thr_curve.tolist()
    except Exception:
        pr_precision = pr_recall = pr_thresholds = []

    return {
        "macro_precision": float(p),
        "macro_recall": float(r),
        "macro_f1": float(f),
        "precision_pos": float(p1),
        "recall_pos": float(r1),
        "f1_pos": float(f1),
        "precision_neg": float(p_1),
        "recall_neg": float(r_1),
        "f1_neg": float(f_1),
        "roc_auc": auc,
        "polarity_correct": mean_s_pos > mean_s_neg,
        "n_items": int(len(items_test)),
        "n_test_items": n_test_items,
        "n_items_missing_score": n_missing_score,
        "n_native_items": int(auc_mask.sum()),
        "n_fallback_items": 0,
        "score_coverage": score_coverage,
        "final_decision_coverage": (
            float(len(items_test) / n_test_items) if n_test_items else 0.0
        ),
        "native_macro_f1": native_macro_f1,
        "native_f1_pos": native_f1_pos,
        "native_f1_neg": native_f1_neg,
        "auc_observed_only": bool(auc_observed_only),
        "n_auc_items": int(auc_mask.sum()),
        "auc_population": (
            "items_with_observed_external_score" if auc_observed_only else "all_test_items"
        ),
        "prediction_population": (
            "items_with_observed_external_score" if omit_missing_scores else "all_test_items"
        ),
        "mean_score_approve": mean_s_pos,
        "mean_score_remove": mean_s_neg,
        "pr_curve_precision": pr_precision,
        "pr_curve_recall": pr_recall,
        "pr_curve_thresholds": pr_thresholds,
    }


# Fold aggregation
def aggregate_fold_metrics(fold_results: List[Dict], baseline_name: str, alpha: float) -> Dict:
    """Aggregate fold metrics across groups."""
    scalar_keys = [
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "precision_pos",
        "recall_pos",
        "f1_pos",
        "precision_neg",
        "recall_neg",
        "f1_neg",
        "roc_auc",
        "score_coverage",
        "final_decision_coverage",
        "native_macro_f1",
    ]
    agg = {"baseline": baseline_name, "alpha": alpha, "n_folds": len(fold_results)}
    for key in scalar_keys:
        vals = [r[key] for r in fold_results if not np.isnan(r[key])]
        agg[key] = float(np.mean(vals))
        agg[f"{key}_std"] = float(np.std(vals))
        agg[f"{key}_per_fold"] = vals
    agg["polarity_correct_per_fold"] = [r["polarity_correct"] for r in fold_results]
    agg["pr_curves_per_fold"] = [
        {
            "precision": r.get("pr_curve_precision", []),
            "recall": r.get("pr_curve_recall", []),
            "thresholds": r.get("pr_curve_thresholds", []),
        }
        for r in fold_results
    ]
    log.info(
        "%s | macro F1 %.4f +/- %.4f | AUC %.4f +/- %.4f",
        baseline_name,
        agg["macro_f1"],
        agg["macro_f1_std"],
        agg["roc_auc"],
        agg["roc_auc_std"],
    )
    return agg


# Output
def save_outputs(
    output_dir: Path,
    metrics: Dict,
    fold_details: List[Dict],
    baseline_name: str,
) -> None:
    """Write outputs to disk."""
    d = output_dir / baseline_name
    d.mkdir(parents=True, exist_ok=True)
    # Store PR curves separately from scalar fold metrics.
    fold_rows = [
        {k: v for k, v in row.items() if not k.startswith("pr_curve")} for row in fold_details
    ]
    with open(d / "metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=2)
    pd.DataFrame(fold_rows).to_parquet(d / "fold_details.parquet", index=False)
    log.info("Baseline %s saved: %s", baseline_name, d)


# K-fold runner
def run_baseline_kfold(
    folds: List[pd.DataFrame],
    all_votes: pd.DataFrame,
    baseline_name: str,
    use_alpha: bool = False,
    use_subreddit: bool = False,
) -> Tuple[Dict, List[Dict]]:
    """Run the baseline kfold workflow."""
    fold_results = []
    fold_details = []
    chosen_alphas = []

    for k, val_fold in enumerate(folds):
        # All folds except the current fold form training data.
        train_fold = pd.concat([folds[i] for i in range(len(folds)) if i != k], ignore_index=True)
        train_labels = get_item_labels(train_fold)
        train_labels_full = train_fold.drop_duplicates("item_id")[
            ["item_id", "label", "community"]
        ].copy()
        if use_alpha:
            # Select alpha on a chronological holdout within training.
            item_times_inner = (
                train_fold.groupby("item_id")["timestamp"].min().sort_values().reset_index()
            )
            n_inner = len(item_times_inner)
            cut = int(n_inner * 0.8)
            inner_train_items = item_times_inner.iloc[:cut]["item_id"].values
            inner_val_items = item_times_inner.iloc[cut:]["item_id"].values
            inner_train = train_fold[train_fold["item_id"].isin(inner_train_items)]
            inner_val = train_fold[train_fold["item_id"].isin(inner_val_items)]
            alpha = select_alpha(inner_train, inner_val)
        else:
            alpha = 1.0

        chosen_alphas.append(alpha)
        scores_all = compute_net_scores(all_votes, alpha=alpha)
        threshold, _ = calibrate_global_threshold(scores_all, train_labels)

        if use_subreddit:
            sub_thr = calibrate_per_subreddit(scores_all, train_labels_full, threshold)
            test_items = (
                get_item_labels(val_fold)
                .merge(val_fold.drop_duplicates("item_id")[["item_id", "community"]], on="item_id")
                .merge(scores_all[["item_id", "score"]], on="item_id", how="left")
            )
            test_items["score"] = test_items["score"].fillna(0.0)
            thr_per_item = test_items["community"].map(sub_thr).fillna(threshold)
            preds = np.where(test_items["score"].values >= thr_per_item.values, 1, -1)
        else:
            test_items = get_item_labels(val_fold).merge(
                scores_all[["item_id", "score"]], on="item_id", how="left"
            )
            test_items["score"] = test_items["score"].fillna(0.0)
            preds = np.where(test_items["score"].values >= threshold, 1, -1)

        fold_metric = evaluate_fold(scores_all, val_fold, preds)
        fold_metric["fold"] = k
        fold_metric["alpha"] = alpha
        fold_metric["threshold"] = threshold
        fold_results.append(fold_metric)
        fold_details.append(fold_metric)

        log.info(
            "  Fold %d | macro F1 %.4f | AUC %.4f | alpha=%.2f | thr=%.4f",
            k,
            fold_metric["macro_f1"],
            fold_metric["roc_auc"],
            alpha,
            threshold,
        )

    mean_alpha = float(np.mean(chosen_alphas))
    metrics = aggregate_fold_metrics(fold_results, baseline_name, alpha=mean_alpha)
    metrics["alpha_per_fold"] = chosen_alphas
    return metrics, fold_details


# Reddit scores
def load_external_scores(path: Path) -> pd.DataFrame:
    """Load external scores from its configured source."""
    df = load_item_scores(path)
    log.info(
        "External scores loaded: %d items | %d with valid score",
        len(df),
        df["score"].notna().sum(),
    )
    return df


def external_score_auc(
    scores_df: pd.DataFrame,
    calibration_labels: pd.DataFrame,
) -> float:
    """Measure the raw external-score AUC on the calibration labels."""
    merged = (
        calibration_labels[["item_id", "label"]]
        .merge(scores_df[["item_id", "score"]], on="item_id", how="left")
        .dropna(subset=["score"])
    )
    if not merged.empty and merged["label"].nunique() == 2:
        return float(roc_auc_score((merged["label"] == 1).astype(int), merged["score"]))
    return float("nan")


def run_bl4_kfold(
    folds: List[pd.DataFrame],
    external_scores: pd.DataFrame,
    baseline_name: str,
) -> Tuple[Dict, List[Dict]]:
    """Run the bl4 kfold workflow."""
    fold_results = []
    fold_details = []

    # Signed log compression limits outliers.
    scores_df = external_scores[["item_id"]].copy()
    raw = pd.to_numeric(external_scores["score"], errors="coerce")
    scores_df["score"] = np.sign(raw) * np.log1p(np.abs(raw))

    for k, val_fold in enumerate(folds):
        train_fold = pd.concat([folds[i] for i in range(len(folds)) if i != k], ignore_index=True)
        train_labels = get_item_labels(train_fold)

        validation_raw_auc = external_score_auc(scores_df, train_labels)

        # Calibrate on the training fold.
        threshold, _ = calibrate_global_threshold(scores_df, train_labels)

        test_items = get_item_labels(val_fold).merge(
            scores_df[["item_id", "score"]], on="item_id", how="left"
        )
        test_items = test_items.dropna(subset=["score"])
        preds = np.where(test_items["score"].values >= threshold, 1, -1)

        fold_metric = evaluate_fold(
            scores_df,
            val_fold,
            preds,
            auc_observed_only=True,
            omit_missing_scores=True,
        )
        fold_metric["fold"] = k
        fold_metric["alpha"] = float("nan")
        fold_metric["threshold"] = threshold
        fold_metric["validation_roc_auc_raw"] = validation_raw_auc
        fold_results.append(fold_metric)
        fold_details.append(fold_metric)

        log.info(
            "  BL4 Fold %d | macro F1 %.4f | AUC %.4f | thr=%.4f | f1_neg=%.4f",
            k,
            fold_metric["macro_f1"],
            fold_metric["roc_auc"],
            threshold,
            fold_metric["f1_neg"],
        )

    metrics = aggregate_fold_metrics(fold_results, baseline_name, alpha=float("nan"))
    return metrics, fold_details


def run_bl5_kfold(
    folds: List[pd.DataFrame],
    external_scores: pd.DataFrame,
    baseline_name: str,
    min_sub_items: int = MIN_SUB_VAL_ITEMS,
) -> Tuple[Dict, List[Dict]]:
    """Run the bl5 kfold workflow."""
    fold_results = []
    fold_details = []

    # Use the same fixed log compression as BL4.
    scores_df = external_scores[["item_id"]].copy()
    raw = pd.to_numeric(external_scores["score"], errors="coerce")
    scores_df["score"] = np.sign(raw) * np.log1p(np.abs(raw))

    for k, val_fold in enumerate(folds):
        train_fold = pd.concat([folds[i] for i in range(len(folds)) if i != k], ignore_index=True)
        train_labels_full = train_fold.drop_duplicates("item_id")[
            ["item_id", "label", "community"]
        ].copy()
        train_labels = train_labels_full[["item_id", "label"]].copy()

        validation_raw_auc = external_score_auc(scores_df, train_labels)

        # Global threshold.
        threshold_global, _ = calibrate_global_threshold(
            scores_df, train_labels, grid=THRESHOLD_GRID
        )

        # Community thresholds.
        sub_thr = calibrate_per_subreddit(
            scores_df,
            train_labels_full,
            threshold_global,
            grid=THRESHOLD_GRID,
            min_items=min_sub_items,
        )

        # Apply the matching community threshold.
        test_items = (
            get_item_labels(val_fold)
            .merge(
                val_fold.drop_duplicates("item_id")[["item_id", "community"]],
                on="item_id",
                how="left",
            )
            .merge(scores_df[["item_id", "score"]], on="item_id", how="left")
        )
        test_items = test_items.dropna(subset=["score"])

        thr_per_item = test_items["community"].map(sub_thr).fillna(threshold_global)
        preds = np.where(test_items["score"].values >= thr_per_item.values, 1, -1)

        fold_metric = evaluate_fold(
            scores_df,
            val_fold,
            preds,
            auc_observed_only=True,
            omit_missing_scores=True,
        )
        fold_metric["fold"] = k
        fold_metric["alpha"] = float("nan")
        fold_metric["threshold"] = threshold_global
        fold_metric["validation_roc_auc_raw"] = validation_raw_auc
        fold_results.append(fold_metric)
        fold_details.append(fold_metric)

        log.info(
            "  BL5 Fold %d | macro F1 %.4f | AUC %.4f | global_thr=%.4f | community thresholds=%d",
            k,
            fold_metric["macro_f1"],
            fold_metric["roc_auc"],
            threshold_global,
            len([v for v in sub_thr.values() if v != threshold_global]),
        )

    metrics = aggregate_fold_metrics(fold_results, baseline_name, alpha=float("nan"))
    return metrics, fold_details


# Legacy K-fold entry point
def run_baselines(
    step1_dir: Path,
    output_dir: Path,
    n_folds: int = N_FOLDS,
) -> Dict[str, Dict]:
    """Run the baselines workflow."""

    all_votes = load_all_votes(step1_dir)
    folds = make_chrono_folds(all_votes, n_folds)
    results = {}

    external_scores = load_external_scores(REDDIT_SCORES_PATH)
    metrics_bl4, details_bl4 = run_bl4_kfold(folds, external_scores, "BL4_reddit_score")
    save_outputs(output_dir, metrics_bl4, details_bl4, "BL4_reddit_score")
    results["BL4_reddit_score"] = metrics_bl4

    metrics_bl5, details_bl5 = run_bl5_kfold(folds, external_scores, "BL5_subreddit")
    save_outputs(output_dir, metrics_bl5, details_bl5, "BL5_subreddit")
    results["BL5_subreddit"] = metrics_bl5

    return results


# Canonical benchmark: TRAIN selects alpha, VAL calibrates thresholds, and
# TEST supplies final metrics. Scores use only the current split's votes.


def _fallback_val_from_train(
    split_label: str,
    baseline_name: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
) -> Optional[pd.DataFrame]:
    """Create val from train when validation data is unavailable."""
    if not val_df.empty and val_df["item_id"].nunique() > 0:
        return val_df

    log.warning(
        "[%s/%s] empty validation set; calibrating on a TRAIN sample",
        split_label,
        baseline_name,
    )
    train_items = train_df.drop_duplicates("item_id")["item_id"].values
    if len(train_items) == 0:
        log.warning("[%s/%s] skipped: no training items", split_label, baseline_name)
        return None
    rng = np.random.RandomState(10)
    sample_size = min(len(train_items), max(MIN_CAL_FALLBACK_ITEMS, int(0.2 * len(train_items))))
    sample_ids = rng.choice(train_items, size=sample_size, replace=False)
    return train_df[train_df["item_id"].isin(sample_ids)]


def run_single_split_baseline(
    split_label: str,
    all_df: pd.DataFrame,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    baseline_name: str,
    use_alpha: bool = False,
    use_subreddit: bool = False,
) -> Tuple[Optional[Dict], pd.DataFrame]:
    """Run the single split baseline workflow."""
    if test_df.empty or test_df["item_id"].nunique() == 0:
        log.warning("[%s/%s] skipped: empty test set", split_label, baseline_name)
        return None, pd.DataFrame()

    val_for_cal = _fallback_val_from_train(split_label, baseline_name, train_df, val_df)
    if val_for_cal is None:
        return None, pd.DataFrame()

    alpha = select_alpha(train_df, val_for_cal) if use_alpha else 1.0

    scores_all = compute_net_scores(all_df, alpha=alpha)
    val_labels = get_item_labels(val_for_cal)
    threshold, _ = calibrate_global_threshold(scores_all, val_labels)

    test_items = (
        get_item_labels(test_df)
        .merge(
            test_df.drop_duplicates("item_id")[["item_id", "community"]], on="item_id", how="left"
        )
        .merge(scores_all[["item_id", "score"]], on="item_id", how="left")
    )
    test_items["score"] = test_items["score"].fillna(0.0)

    if use_subreddit:
        val_labels_full = val_for_cal.drop_duplicates("item_id")[
            ["item_id", "label", "community"]
        ].copy()
        sub_thr = calibrate_per_subreddit(scores_all, val_labels_full, threshold)
        thr_per_item = test_items["community"].map(sub_thr).fillna(threshold)
    else:
        thr_per_item = pd.Series(threshold, index=test_items.index)

    preds = np.where(test_items["score"].values >= thr_per_item.values, 1, -1)

    metric = evaluate_fold(scores_all, test_df, preds)
    metric["split"] = split_label
    metric["baseline"] = baseline_name
    metric["alpha"] = alpha
    metric["threshold"] = threshold
    metric["n_train_items"] = int(train_df["item_id"].nunique())
    metric["n_val_items"] = int(val_df["item_id"].nunique())

    log.info(
        "  [%s/%s] macro_f1=%.4f  roc_auc=%.4f  alpha=%.2f  thr=%.4f",
        split_label,
        baseline_name,
        metric["macro_f1"],
        metric["roc_auc"],
        alpha,
        threshold,
    )

    item_scores = test_items.copy()
    item_scores["prediction"] = preds
    return metric, item_scores


def run_single_split_external(
    split_label: str,
    external_scores: pd.DataFrame,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    baseline_name: str,
    use_subreddit: bool = False,
) -> Tuple[Optional[Dict], pd.DataFrame]:
    """Run the single split external workflow."""
    if test_df.empty or test_df["item_id"].nunique() == 0:
        log.warning("[%s/%s] skipped: empty test set", split_label, baseline_name)
        return None, pd.DataFrame()

    val_for_cal = _fallback_val_from_train(split_label, baseline_name, train_df, val_df)
    if val_for_cal is None:
        return None, pd.DataFrame()

    val_labels = get_item_labels(val_for_cal)
    validation_raw_auc = external_score_auc(external_scores, val_labels)
    threshold, _ = calibrate_global_threshold(external_scores, val_labels)

    test_items = (
        get_item_labels(test_df)
        .merge(
            test_df.drop_duplicates("item_id")[["item_id", "community"]], on="item_id", how="left"
        )
        .merge(external_scores[["item_id", "score"]], on="item_id", how="left")
    )

    missing_external = test_items["score"].isna()
    test_items = test_items.loc[~missing_external].copy()

    if use_subreddit:
        val_labels_full = val_for_cal.drop_duplicates("item_id")[
            ["item_id", "label", "community"]
        ].copy()
        sub_thr = calibrate_per_subreddit(external_scores, val_labels_full, threshold)
        thr_per_item = test_items["community"].map(sub_thr).fillna(threshold)
    else:
        thr_per_item = pd.Series(threshold, index=test_items.index)

    preds = np.where(test_items["score"].values >= thr_per_item.values, 1, -1)
    test_items["decision_source"] = "archived_reddit_score"

    metric = evaluate_fold(
        external_scores,
        test_df,
        preds,
        auc_observed_only=True,
        omit_missing_scores=True,
    )
    metric["split"] = split_label
    metric["baseline"] = baseline_name
    metric["alpha"] = float("nan")
    metric["threshold"] = threshold
    metric["validation_roc_auc_raw"] = validation_raw_auc
    metric["n_train_items"] = int(train_df["item_id"].nunique())
    metric["n_val_items"] = int(val_df["item_id"].nunique())
    metric["prediction_score_policy"] = "observed_external_score_only"
    if metric.get("n_items_missing_score", 0) > 0:
        log.warning(
            "[%s/%s] score coverage=%.2f%% | evaluated=%d | omitted=%d",
            split_label,
            baseline_name,
            100.0 * metric["score_coverage"],
            metric["n_items"],
            metric["n_items_missing_score"],
        )

    log.info(
        "  [%s/%s] macro_f1=%.4f  roc_auc=%.4f  thr=%.4f",
        split_label,
        baseline_name,
        metric["macro_f1"],
        metric["roc_auc"],
        threshold,
    )

    item_scores = test_items.copy()
    item_scores["prediction"] = preds
    return metric, item_scores


def _save_split_baseline_outputs(
    split_out_dir: Path,
    baseline_name: str,
    metric: Dict,
    item_scores: pd.DataFrame,
) -> None:
    """Write split baseline outputs to disk."""
    d = split_out_dir / baseline_name
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "metrics.json", "w") as fh:
        json.dump(metric, fh, indent=2)
    item_scores.to_parquet(d / "item_scores.parquet", index=False)


def evaluate_splits(
    votes_dir: Path,
    output_dir: Path,
    external_scores_path: Path = REDDIT_SCORES_PATH,
    only_external: bool = False,
    baseline_names: Optional[List[str]] = None,
) -> Dict[str, Dict[str, Dict]]:
    """Evaluate splits and return its metrics."""
    splits = discover_splits(votes_dir)
    if not splits:
        log.warning(
            "No prepared splits found under %s",
            votes_dir,
        )
        return {}

    selected = set(baseline_names or CANONICAL_BASELINES)
    unknown = sorted(selected.difference(CANONICAL_BASELINES))
    if unknown:
        raise ValueError(f"Unknown baselines: {unknown}")
    if only_external:
        selected.intersection_update({"BL4_reddit_score", "BL5_subreddit"})

    external_scores_compressed = None
    needs_external_scores = bool(selected.intersection({"BL4_reddit_score", "BL5_subreddit"}))
    if needs_external_scores and external_scores_path.exists():
        ext = load_external_scores(external_scores_path)
        # Preserve missing external scores through the fixed log transform.
        # ``run_single_split_external`` detects these NaNs and applies the
        # item-level net-vote fallback without including them in external-score
        # coverage or ROC-AUC.
        raw = pd.to_numeric(ext["score"], errors="coerce")
        external_scores_compressed = ext[["item_id"]].copy()
        external_scores_compressed["score"] = np.sign(raw) * np.log1p(np.abs(raw))
    elif needs_external_scores:
        log.warning(
            "BL4/BL5 skipped: external scores missing at %s",
            external_scores_path,
        )

    summary: Dict[str, Dict[str, Dict]] = {}

    for split_name, split_path in splits.items():
        all_df, train_df, val_df, test_df = load_split_data(split_path)
        split_out_dir = output_dir / split_name / "baselines"
        summary[split_name] = {}

        if not only_external:
            baseline_configs = [
                ("BL1_net_vote", dict(use_alpha=False, use_subreddit=False)),
                ("BL2_net_vote_alpha", dict(use_alpha=True, use_subreddit=False)),
                ("BL3_net_vote_alpha_subreddit", dict(use_alpha=True, use_subreddit=True)),
            ]
            for baseline_name, kwargs in baseline_configs:
                if baseline_name not in selected:
                    continue
                metric, item_scores = run_single_split_baseline(
                    split_name, all_df, train_df, val_df, test_df, baseline_name, **kwargs
                )
                if metric is None:
                    continue
                _save_split_baseline_outputs(split_out_dir, baseline_name, metric, item_scores)
                summary[split_name][baseline_name] = metric

        if external_scores_compressed is not None:
            for baseline_name, use_sub in (("BL4_reddit_score", False), ("BL5_subreddit", True)):
                if baseline_name not in selected:
                    continue
                metric, item_scores = run_single_split_external(
                    split_name,
                    external_scores_compressed,
                    train_df,
                    val_df,
                    test_df,
                    baseline_name,
                    use_subreddit=use_sub,
                )
                if metric is None:
                    continue
                _save_split_baseline_outputs(split_out_dir, baseline_name, metric, item_scores)
                summary[split_name][baseline_name] = metric

        if not summary[split_name]:
            del summary[split_name]

    output_dir.mkdir(parents=True, exist_ok=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate baselines on one common split")
    parser.add_argument(
        "--votes-dir",
        type=Path,
        default=REDDIT.splits,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REDDIT.results,
    )
    parser.add_argument(
        "--external-scores",
        type=Path,
        default=REDDIT_SCORES_PATH,
        help="Optional item_id/score table used by BL4 and BL5.",
    )
    parser.add_argument(
        "--only-external",
        action="store_true",
        help="Run only BL4/BL5.",
    )
    parser.add_argument(
        "--baseline",
        action="append",
        choices=CANONICAL_BASELINES,
        dest="baselines",
        help="Run one baseline; repeat to select several. Omit to run all five.",
    )
    args = parser.parse_args()
    evaluate_splits(
        votes_dir=args.votes_dir,
        output_dir=args.output_dir,
        external_scores_path=args.external_scores,
        only_external=args.only_external,
        baseline_names=args.baselines,
    )
