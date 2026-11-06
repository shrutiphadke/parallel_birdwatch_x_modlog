"""Vectorized macro precision/recall/F1 over a whole threshold grid.

Several methods choose a decision threshold by evaluating
``where(score >= t, 1, -1)`` against binary ±1 labels for hundreds of grid
points, calling scikit-learn once per point.  Input validation inside those
calls dominated runtime.  These helpers compute every grid point at once from
exact integer counts and the same float formulas scikit-learn uses
(``F1 = 2·tp / (true + pred)``, ``zero_division=0``), so results are identical.
"""

from __future__ import annotations

import numpy as np


def _class_counts(
    y_true: np.ndarray, scores: np.ndarray, grid: np.ndarray
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Per threshold: true +1 items predicted +1, and true -1 items predicted +1.

    A NaN score is never ``>= t`` (as with ``np.where``), so it always counts
    as a -1 prediction.
    """
    counts = []
    for label in (1, -1):
        class_scores = scores[y_true == label]
        finite = np.sort(class_scores[~np.isnan(class_scores)])
        counts.append(
            (len(finite) - np.searchsorted(finite, grid, side="left"), len(class_scores))
        )
    (tp_pos, n_pos), (fp_pos, n_neg) = counts
    return tp_pos, fp_pos, n_pos, n_neg


def _divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(denominator > 0, numerator / denominator, 0.0)


def macro_prf_over_thresholds(
    y_true: np.ndarray,
    scores: np.ndarray,
    grid: np.ndarray,
    *,
    fixed_labels: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Macro precision, recall and F1 of ``where(score >= t, 1, -1)`` for each ``t``.

    ``fixed_labels=True`` matches ``labels=[1, -1]`` (both classes always
    averaged).  ``fixed_labels=False`` matches scikit-learn's default of
    averaging only labels present in ``y_true`` or in that threshold's
    predictions.
    """
    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype=float)
    grid = np.asarray(grid, dtype=float)
    tp_pos, fp_pos, n_pos, n_neg = _class_counts(y_true, scores, grid)
    per_class = []
    for tp, fp, fn in (
        (tp_pos, fp_pos, n_pos - tp_pos),  # label +1
        (n_neg - fp_pos, n_pos - tp_pos, fp_pos),  # label -1
    ):
        tp = tp.astype(float)
        predicted, actual = tp + fp, tp + fn
        per_class.append(
            (
                _divide(tp, predicted),
                _divide(tp, actual),
                _divide(2.0 * tp, 1.0 * actual + predicted),
                (actual + predicted) > 0,  # label present in y_true or y_pred
            )
        )
    results = []
    for metric in range(3):
        values = np.vstack([entry[metric] for entry in per_class])
        if fixed_labels:
            results.append(np.mean(values, axis=0))
        else:
            present = np.vstack([entry[3] for entry in per_class])
            totals = np.where(present, values, 0.0).sum(axis=0)
            counts = present.sum(axis=0)
            results.append(_divide(totals, counts.astype(float)))
    return tuple(results)


def per_class_f1_over_thresholds(
    y_true: np.ndarray, scores: np.ndarray, grid: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """F1 of label +1 and of label -1 per threshold (``average=None, labels=[1, -1]``)."""
    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype=float)
    grid = np.asarray(grid, dtype=float)
    tp_pos, fp_pos, n_pos, n_neg = _class_counts(y_true, scores, grid)
    f1 = []
    for tp, fp, fn in (
        (tp_pos, fp_pos, n_pos - tp_pos),
        (n_neg - fp_pos, n_pos - tp_pos, fp_pos),
    ):
        tp = tp.astype(float)
        f1.append(_divide(2.0 * tp, 1.0 * (tp + fn) + (tp + fp)))
    return f1[0], f1[1]


def best_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    grid: np.ndarray,
    *,
    default: float,
    fixed_labels: bool = True,
) -> tuple[float, float]:
    """First grid threshold reaching the maximal macro F1 (``max``/strict-``>`` semantics)."""
    grid = np.asarray(grid, dtype=float)
    if not len(grid):
        return default, -1.0
    _, _, f1 = macro_prf_over_thresholds(y_true, scores, grid, fixed_labels=fixed_labels)
    index = int(np.argmax(f1))
    return float(grid[index]), float(f1[index])
