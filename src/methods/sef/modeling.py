"""Meta-model, negative predictor, weighted vote, and calibration for SEF."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from src.utils.threshold_search import macro_prf_over_thresholds

from . import runtime
from .runtime import (
    META_OOF_FOLDS,
    MIN_NEG_EXPERTS,
    MIN_SUB_SAMPLES,
    log,
)

try:
    from xgboost import XGBClassifier

    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


# 6. META-MODEL


class _XGBWrapper:
    """Implement xgbwrapper."""

    def __init__(self, clf):
        """Initialize the instance."""
        self._clf = clf
        self.classes_ = np.array([-1, 1])

    def predict(self, X):
        """Predict labels for the supplied features."""
        raw = self._clf.predict(X)
        return np.where(raw == 1, 1, -1)

    def predict_proba(self, X):
        """Predict class probabilities for the supplied features."""
        # columns: [proba_class0(=-1), proba_class1(=+1)]
        return self._clf.predict_proba(X)  # already [p0, p1]

    def get_params(self, deep=True):
        """Return estimator parameters for scikit-learn compatibility."""
        return self._clf.get_params(deep=deep)

    @property
    def feature_importances_(self):
        """Return feature weights in estimator order."""
        return self._clf.feature_importances_


def _build_candidates(y: np.ndarray) -> List[Tuple[str, Any]]:
    """Build candidates from the supplied data."""
    candidates = [
        (
            "LogReg-L2",
            LogisticRegression(
                C=0.1,
                class_weight="balanced",
                max_iter=1000,
                solver="lbfgs",
            ),
        ),
        (
            "GBT",
            GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=4,
                subsample=0.8,
                min_samples_leaf=20,
                random_state=10,
            ),
        ),
    ]
    if HAS_XGBOOST:
        pos_count = int((y == 1).sum())
        neg_count = int((y == -1).sum())
        scale_pos = neg_count / max(pos_count, 1)
        candidates.append(
            (
                "XGB",
                XGBClassifier(
                    n_estimators=200,
                    learning_rate=0.05,
                    max_depth=4,
                    subsample=0.8,
                    scale_pos_weight=scale_pos,
                    eval_metric="logloss",
                    verbosity=0,
                    random_state=10,
                    n_jobs=max(1, int(os.environ.get("OMP_NUM_THREADS", "1"))),
                ),
            )
        )
    return candidates


def train_meta_model(
    feat_df: pd.DataFrame,
    labels: Dict[str, int],
    feat_cols: List[str],
    validation_feat_df: pd.DataFrame,
    validation_labels: Dict[str, int],
    validation_fixed_predictions: Optional[Dict[str, int]] = None,
) -> Tuple[Any, StandardScaler, str]:
    """Select on external validation, then fit label-disjoint TRAIN features."""
    items = feat_df[feat_df["item_id"].isin(labels)]["item_id"].tolist()
    X = feat_df.set_index("item_id").loc[items, feat_cols].values.astype(float)
    y = np.array([labels[i] for i in items])

    val_items = validation_feat_df[validation_feat_df["item_id"].isin(validation_labels)][
        "item_id"
    ].tolist()
    if not val_items:
        raise ValueError("Meta-model selection requires a non-empty external validation set")
    X_val = validation_feat_df.set_index("item_id").loc[val_items, feat_cols].values.astype(float)
    y_val = np.array([validation_labels[i] for i in val_items])
    fixed_predictions = validation_fixed_predictions or {}

    counts = {c: (y == c).sum() for c in np.unique(y)}
    max_cnt = max(counts.values())
    sw = np.array([max_cnt / counts[yi] for yi in y])

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    X_val_s = scaler.transform(X_val)

    candidates = _build_candidates(y)
    best_name, best_clf, best_score = None, None, -1.0

    for name, clf in candidates:
        try:
            candidate = clone(clf)
            if name == "LogReg-L2":
                candidate.fit(X_s, y)
                prob_val = candidate.predict_proba(X_val_s)[:, list(candidate.classes_).index(1)]
            elif name == "XGB":
                candidate.fit(X_s, (y == 1).astype(int))
                prob_val = candidate.predict_proba(X_val_s)[:, 1]
            else:
                candidate.fit(X_s, y, sample_weight=sw)
                prob_val = candidate.predict_proba(X_val_s)[:, list(candidate.classes_).index(1)]

            candidate_thresholds = np.unique(
                np.concatenate([np.linspace(0.01, 0.99, 199), prob_val])
            )
            mean_f1 = -1.0
            for candidate_threshold in candidate_thresholds:
                pred_val = np.where(prob_val >= candidate_threshold, 1, -1)
                for idx, item_id in enumerate(val_items):
                    if item_id in fixed_predictions:
                        pred_val[idx] = fixed_predictions[item_id]
                mean_f1 = max(
                    mean_f1,
                    float(f1_score(y_val, pred_val, average="macro", zero_division=0)),
                )
            log.info("%s on external validation: macro-F1=%.4f", name, mean_f1)

            if mean_f1 > best_score:
                best_score, best_name, best_clf = mean_f1, name, candidate
        except Exception as e:
            log.warning("Training %s failed: %s", name, e)

    if best_clf is None:
        raise RuntimeError("Every SEF meta-model candidate failed")
    if best_name == "XGB":
        # The selected candidate is already fitted; expose -1/+1 classes.
        best_clf = _XGBWrapper(best_clf)

    # feature importances
    actual_clf = best_clf._clf if isinstance(best_clf, _XGBWrapper) else best_clf
    if hasattr(actual_clf, "feature_importances_"):
        top3 = sorted(
            zip(feat_cols, actual_clf.feature_importances_), key=lambda x: x[1], reverse=True
        )[:3]
    elif hasattr(actual_clf, "coef_"):
        top3 = sorted(
            zip(feat_cols, np.abs(actual_clf.coef_[0])), key=lambda x: x[1], reverse=True
        )[:3]
    else:
        top3 = []

    log.info(
        "Selected meta-model: %s (external-validation F1=%.4f); top features: %s",
        best_name,
        best_score,
        {k: round(v, 3) for k, v in top3},
    )
    return best_clf, scaler, best_name


# 6b. NEGATIVE PREDICTOR


def train_neg_predictor(
    feat_df: pd.DataFrame,
    labels: Dict[str, int],
    feat_cols: List[str],
) -> Tuple[Optional[Any], Optional[StandardScaler], Optional[List[str]]]:
    """Train neg predictor on the supplied features."""
    neg_feat_cols = [
        c
        for c in feat_cols
        if c != "neg_expert_proba"
        and any(tag in c for tag in ["neg", "wv_neg", "n_neg", "neg_expert"])
    ]
    if not neg_feat_cols:
        return None, None, None

    df = feat_df[feat_df["item_id"].isin(labels)].copy()
    df["_label"] = df["item_id"].map(labels)

    # keep only posts with at least one reliable downvoter expert
    mask = (
        df["n_neg_reliable"] >= MIN_NEG_EXPERTS
        if "n_neg_reliable" in df.columns
        else pd.Series(True, index=df.index)
    )
    df_neg = df[mask]

    if len(df_neg) < 50 or df_neg["_label"].nunique() < 2:
        log.info("Negative predictor skipped: insufficient data (%d posts)", len(df_neg))
        return None, None, None

    X = df_neg[neg_feat_cols].values.astype(float)
    y = df_neg["_label"].values.astype(int)

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    clf = LogisticRegression(
        C=0.1,
        class_weight="balanced",
        max_iter=1000,
        solver="lbfgs",
    )
    clf.fit(X_s, y)
    log.info(
        "Negative predictor trained on %d posts (%d removed, %d approved)",
        len(y),
        (y == -1).sum(),
        (y == 1).sum(),
    )
    return clf, scaler, neg_feat_cols


def add_oof_negative_probabilities(
    feat_df: pd.DataFrame,
    labels: Dict[str, int],
    feat_cols: List[str],
    n_splits: int = META_OOF_FOLDS,
) -> Tuple[
    pd.DataFrame,
    Optional[Any],
    Optional[StandardScaler],
    Optional[List[str]],
]:
    """Create stacking-safe TRAIN probabilities and fit the deployment model."""
    result = feat_df.copy()
    result["neg_expert_proba"] = 0.5
    items = result[result["item_id"].isin(labels)]["item_id"].tolist()
    y = np.array([labels[iid] for iid in items])
    class_counts = pd.Series(y).value_counts()
    folds = min(n_splits, int(class_counts.min())) if len(class_counts) >= 2 else 0

    if folds >= 2:
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=runtime.random_seed)
        item_array = np.asarray(items, dtype=object)
        for fold_idx, (fit_idx, hold_idx) in enumerate(splitter.split(item_array, y), start=1):
            fit_items = item_array[fit_idx].tolist()
            hold_items = item_array[hold_idx].tolist()
            fold_labels = {iid: labels[iid] for iid in fit_items}
            fold_train = result[result["item_id"].isin(fit_items)]
            fold_clf, fold_scaler, fold_cols = train_neg_predictor(
                fold_train, fold_labels, feat_cols
            )
            fold_prob = apply_neg_predictor(result, hold_items, fold_clf, fold_scaler, fold_cols)
            mask = result["item_id"].isin(hold_items)
            result.loc[mask, "neg_expert_proba"] = (
                result.loc[mask, "item_id"].map(fold_prob).fillna(0.5)
            )
            log.info(
                "Negative predictor OOF fold %d of %d: %d held-out items",
                fold_idx,
                folds,
                len(hold_items),
            )
    else:
        log.info("Negative predictor OOF: insufficient class support; using neutral 0.5")

    final_clf, final_scaler, final_cols = train_neg_predictor(result, labels, feat_cols)
    return result, final_clf, final_scaler, final_cols


def apply_neg_predictor(
    feat_df: pd.DataFrame,
    item_ids: List[str],
    neg_clf: Optional[Any],
    neg_scaler: Optional[StandardScaler],
    neg_feat_cols: Optional[List[str]],
) -> Dict[str, float]:
    """Apply neg predictor to the supplied data."""
    result = {iid: 0.5 for iid in item_ids}
    if neg_clf is None or neg_feat_cols is None:
        return result

    df = feat_df.set_index("item_id")
    # only predict for posts with at least one reliable neg expert
    eligible = [
        iid
        for iid in item_ids
        if iid in df.index and df.loc[iid, "n_neg_reliable"] >= MIN_NEG_EXPERTS
    ]
    if not eligible:
        return result

    X = df.loc[eligible, neg_feat_cols].values.astype(float)
    X_s = neg_scaler.transform(X)
    # proba of class -1 (remove)
    neg_class_idx = list(neg_clf.classes_).index(-1)
    probas = neg_clf.predict_proba(X_s)[:, neg_class_idx]
    for iid, p in zip(eligible, probas):
        result[iid] = float(p)
    return result


def predict_meta_model(
    feat_df: pd.DataFrame,
    item_ids: List[str],
    clf: Any,
    scaler: StandardScaler,
    feat_cols: List[str],
) -> Dict[str, Tuple[int, float]]:
    """Predict item labels with the fitted meta-model."""
    df = feat_df.set_index("item_id")
    pos_class_idx = list(clf.classes_).index(1)

    known = [iid for iid in item_ids if iid in df.index]
    result: Dict[str, Tuple[int, float]] = {iid: (1, 0.5) for iid in item_ids}
    if not known:
        return result

    X = df.loc[known, feat_cols].values.astype(float)
    X_s = scaler.transform(X)
    preds = clf.predict(X_s)
    probas = clf.predict_proba(X_s)[:, pos_class_idx]
    for iid, pred, proba in zip(known, preds, probas):
        result[iid] = (int(pred), float(proba))
    return result


def _items_with_real_expert_signal(weights_df: pd.DataFrame) -> set:
    """Return items backed by history, local evidence or a user embedding."""
    if weights_df.empty or "has_real_expert_signal" not in weights_df:
        return set()
    real_signal = weights_df.groupby("item_id")["has_real_expert_signal"].any()
    return set(real_signal[real_signal].index)


def _net_vote_predictions(
    item_ids: List[str], vote_df: pd.DataFrame
) -> Tuple[Dict[str, float], Dict[str, int]]:
    """Return raw vote means and their deterministic sign predictions."""
    raw_mean = (
        vote_df[vote_df["item_id"].isin(item_ids)].groupby("item_id")["vote"].mean().to_dict()
    )
    predictions = {
        item_id: (1 if float(raw_mean.get(item_id, 0.0)) >= 0.0 else -1) for item_id in item_ids
    }
    return raw_mean, predictions


# 8. THRESHOLD CALIBRATION


def calibrate_probability_threshold(
    sample_ids: List[str],
    vote_df: pd.DataFrame,
    positive_probabilities: Dict[str, float],
) -> float:
    """Calibrate the final meta-model probability threshold on validation."""
    labels = (
        vote_df[vote_df["item_id"].isin(sample_ids)]
        .drop_duplicates("item_id")
        .set_index("item_id")["label"]
        .to_dict()
    )
    common = [iid for iid in sample_ids if iid in labels and iid in positive_probabilities]
    if not common:
        return 0.5

    y_true = np.array([labels[iid] for iid in common])
    y_score = np.array([positive_probabilities[iid] for iid in common])

    candidates = np.unique(np.concatenate([np.linspace(0.01, 0.99, 199), y_score]))
    best_thr, best_f1 = 0.5, -1.0
    # Scores for every candidate at once; the tie-break loop below is unchanged.
    _, _, candidate_f1 = macro_prf_over_thresholds(y_true, y_score, candidates, fixed_labels=False)
    for thr, f1 in zip(candidates, candidate_f1.tolist()):
        if f1 > best_f1 or (
            np.isclose(f1, best_f1) and abs(float(thr) - 0.5) < abs(best_thr - 0.5)
        ):
            best_f1, best_thr = f1, float(thr)
    return best_thr


def calibrate_probability_thresholds_per_subreddit(
    sample_ids: List[str],
    vote_df: pd.DataFrame,
    positive_probabilities: Dict[str, float],
    min_samples: int = MIN_SUB_SAMPLES,
) -> Dict[Optional[str], float]:
    """Calibrate final probability thresholds globally and by subreddit."""
    global_thr = calibrate_probability_threshold(sample_ids, vote_df, positive_probabilities)
    thresholds: Dict[Optional[str], float] = {None: global_thr}

    post_sub = (
        vote_df[vote_df["item_id"].isin(sample_ids)]
        .drop_duplicates("item_id")
        .set_index("item_id")["community"]
        .to_dict()
    )
    sub_groups: Dict[str, List[str]] = {}
    for iid in sample_ids:
        sub = post_sub.get(iid)
        if sub:
            sub_groups.setdefault(sub, []).append(iid)

    n_specific = 0
    for sub, ids in sub_groups.items():
        if len(ids) < min_samples:
            thresholds[sub] = global_thr
        else:
            thresholds[sub] = calibrate_probability_threshold(ids, vote_df, positive_probabilities)
            n_specific += 1

    log.info("Thresholds: global=%.4f, subreddit-specific=%d", global_thr, n_specific)
    return thresholds
