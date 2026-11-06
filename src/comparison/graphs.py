"""Generate thesis tables, figures, audits, and CN-extension summaries."""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import os
from pathlib import Path
import sys
from typing import Annotated, Any, Mapping, Sequence

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str((_PROJECT_ROOT / "cache" / "matplotlib").resolve()))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import typer  # noqa: E402

from src.layout import REDDIT  # noqa: E402
from src.utils.splits import discover_splits, load_vote_partitions  # noqa: E402

log = logging.getLogger(__name__)

FIXED_SPLITS = (
    "full",
    "intersection",
    "intersection_chronological",
)
WINDOW_POPULATIONS = ("full", "intersection")
METRICS = ("macro_f1", "f1_pos", "f1_neg", "roc_auc")
METRIC_LABELS = {
    "macro_f1": "Macro F1",
    "f1_pos": "F1 approve",
    "f1_neg": "F1 remove",
    "roc_auc": "ROC AUC",
}
CN_EXTENSION_VARIANTS = {
    "cn_inductive": "CN inductive",
    "qsmf": "QSMF",
    "ma_mf": "MA-MF",
    "ma_qsmf": "MA-QSMF",
}


@dataclass(frozen=True)
class MethodSpec:
    """Canonical description of one benchmark method."""

    label: str
    relative_dir: str
    prediction_files: tuple[str, ...] = ("test_predictions.parquet", "item_scores.parquet")
    prediction_columns: tuple[str, ...] = ("y_hat", "prediction", "predicted")
    score_columns: tuple[str, ...] = ("score",)
    main_report: bool = True
    supported_splits: tuple[str, ...] | None = None


METHOD_SPECS: dict[str, MethodSpec] = {
    "Net": MethodSpec("Net", "baselines/BL1_net_vote", prediction_files=("item_scores.parquet",)),
    "WNet": MethodSpec(
        "WNet", "baselines/BL2_net_vote_alpha", prediction_files=("item_scores.parquet",)
    ),
    "SubAdj": MethodSpec(
        "SubAdj",
        "baselines/BL3_net_vote_alpha_subreddit",
        prediction_files=("item_scores.parquet",),
    ),
    "RedditScore": MethodSpec(
        "RedditScore", "baselines/BL4_reddit_score", prediction_files=("item_scores.parquet",)
    ),
    "RedditScore+Sub": MethodSpec(
        "RedditScore+Sub",
        "baselines/BL5_subreddit",
        prediction_files=("item_scores.parquet",),
    ),
    "CN": MethodSpec(
        "CN",
        "cn",
        prediction_files=("item_scores.parquet",),
        score_columns=("i_n", "score"),
    ),
    "CN inductive": MethodSpec("CN inductive", "ma_qsmf/cn_inductive"),
    "QSMF": MethodSpec("QSMF", "ma_qsmf/qsmf"),
    "MA-MF": MethodSpec("MA-MF", "ma_qsmf/ma_mf"),
    "MA-QSMF": MethodSpec("MA-QSMF", "ma_qsmf/ma_qsmf"),
    "NVSE": MethodSpec(
        "NVSE",
        "normvio-skill-extraction",
        prediction_files=("item_scores_test.parquet", "item_scores.parquet"),
        score_columns=("nvse_score", "score"),
    ),
    "VAR": MethodSpec("VAR", "var", prediction_files=("item_scores.parquet",)),
    "Bandit": MethodSpec("Bandit", "bandit"),
    "GraphPropagation": MethodSpec(
        "GraphPropagation",
        "graph_propagation",
    ),
    "RoleNSGA2": MethodSpec(
        "Role-NSGA2", "role_nsga2"
    ),
    "VirtualEnsembles": MethodSpec(
        "VirtualEnsembles",
        "virtual_ensembles",
    ),
    "BoC": MethodSpec("BoC", "boc_stacking"),
}

AUDIT_METRICS = ("macro_f1", "f1_neg")

METHOD_COLORS = {
    "CN": "#a61e2a",
    "CN inductive": "#d1495b",
    "QSMF": "#edae49",
    "MA-MF": "#00798c",
    "MA-QSMF": "#30638e",
    "SEF": "#e63946",
    "BoC": "#264653",
    "VirtualEnsembles": "#457b9d",
    "SubAdj": "#2a9d8f",
    "NVSE": "#f4a261",
    "VAR": "#e76f51",
    "Net": "#8d99ae",
    "WNet": "#6a994e",
    "RedditScore": "#7b2cbf",
    "RedditScore+Sub": "#5a189a",
    "Bandit": "#00a6a6",
    "GraphPropagation": "#3a86ff",
    "RoleNSGA2": "#ff6b6b",
}

METHOD_FAMILIES = {
    "Baseline": (
        "Net",
        "WNet",
        "SubAdj",
        "RedditScore",
        "RedditScore+Sub",
    ),
    "Factorization": (
        "CN",
        "CN inductive",
        "QSMF",
        "MA-MF",
        "MA-QSMF",
    ),
    "Observed-voter expertise": ("VAR", "NVSE", "SEF"),
    "Eligible voter base": (
        "Bandit",
        "GraphPropagation",
        "RoleNSGA2",
        "VirtualEnsembles",
        "BoC",
    ),
}

KFOLD_DATASET_LABELS = {
    "full": "Full",
    "intersection": "Intersection",
    "intersection_chronological": "Chronological",
}

WINDOW_LABELS = {
    20: "20% training",
    40: "40% training",
    60: "60% training",
    80: "80% training",
    100: "100% training",
}


def parse_result_path(metrics_path: Path, root: Path) -> tuple[str, str] | None:
    """Return the split and raw method key encoded by a result path."""
    try:
        parts = metrics_path.parent.relative_to(root).parts
    except ValueError:
        return None
    if len(parts) < 2:
        return None
    if "baselines" in parts:
        marker = parts.index("baselines")
        return "/".join(parts[:marker]), parts[-1]
    return "/".join(parts[:-1]), parts[-1]


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def _test_payload(metrics: Mapping[str, Any]) -> dict[str, Any]:
    test = metrics.get("test")
    return dict(test) if isinstance(test, Mapping) else dict(metrics)


def _split_result_dir(results_root: Path, split_name: str) -> Path:
    return results_root.joinpath(*split_name.split("/"))


def _fixed_method_dir(results_root: Path, split_name: str, method: str) -> Path | None:
    spec = METHOD_SPECS.get(method)
    if spec is None:
        return None
    return _split_result_dir(results_root, split_name) / spec.relative_dir


def resolve_method(
    results_root: Path, split_name: str, method: str
) -> tuple[Path, dict[str, Any]] | None:
    """Resolve one canonical method result."""
    if not method_supports_split(method, split_name):
        return None
    if method == "SEF":
        directory = _split_result_dir(results_root, split_name) / "expertise"
        metrics_path = directory / "metrics.json"
        if not metrics_path.exists():
            return None
        metrics = _read_json(metrics_path)
        if metrics.get("status") == "excluded":
            return None
        return directory, metrics

    directory = _fixed_method_dir(results_root, split_name, method)
    if directory is None:
        return None
    metrics_path = directory / "metrics.json"
    if not metrics_path.exists():
        return None
    return directory, _read_json(metrics_path)


def canonical_methods(include_appendix: bool = True) -> list[str]:
    """Return benchmark methods in stable report order."""
    methods = list(METHOD_SPECS)
    if not include_appendix:
        methods = [method for method in methods if METHOD_SPECS[method].main_report]
    methods.append("SEF")
    return methods


def method_supports_split(method: str, split_name: str) -> bool:
    """Return whether a method is defined for the requested evaluation split."""
    spec = METHOD_SPECS.get(method)
    return spec is None or spec.supported_splits is None or split_name in spec.supported_splits


def method_label(method: str) -> str:
    """Return the report-facing label for a canonical method key."""
    return METHOD_SPECS.get(method, MethodSpec(method, "")).label


def collect_method_split_matrix(results_root: Path, splits_root: Path) -> pd.DataFrame:
    """Inventory every discovered split x canonical-method combination."""
    rows: list[dict[str, Any]] = []
    for split_name in discover_splits(splits_root):
        for method in canonical_methods(include_appendix=True):
            supported = method_supports_split(method, split_name)
            resolved = resolve_method(results_root, split_name, method) if supported else None
            directory = resolved[0] if resolved else None
            rows.append(
                {
                    "split": split_name,
                    "method": method,
                    "label": method_label(method),
                    "supported": supported,
                    "status": (
                        "available"
                        if resolved is not None
                        else "missing"
                        if supported
                        else "not_applicable"
                    ),
                    "result_dir": directory.as_posix() if directory is not None else None,
                }
            )
    return pd.DataFrame(rows)


def _metric_row(
    split_name: str,
    method: str,
    metrics: Mapping[str, Any],
) -> dict[str, Any]:
    payload = _test_payload(metrics)
    n_items = payload.get("n_items", payload.get("n_test", payload.get("n_items_test")))
    n_test_items = payload.get("n_test_items", metrics.get("n_test_items", n_items))
    coverage = payload.get("score_coverage")
    if coverage is None:
        coverage = payload.get("coverage_test", payload.get("coverage"))
    root_coverage = metrics.get("coverage")
    if coverage is None and isinstance(root_coverage, Mapping):
        test_coverage = root_coverage.get("test")
        if isinstance(test_coverage, Mapping):
            eligible = test_coverage.get("eligible_items")
            total = test_coverage.get("total_items")
        else:
            eligible = root_coverage.get("test_eligible_items")
            total = root_coverage.get("test_total_items")
        if isinstance(eligible, (int, float)) and isinstance(total, (int, float)) and total:
            n_test_items = int(total)
            coverage = float(eligible) / float(total)
    if coverage is None and n_items is not None and n_test_items:
        coverage = float(n_items) / float(n_test_items)
    if coverage is None:
        coverage = 1.0
    fallback = payload.get("n_fallback_items", metrics.get("n_fallback_items", 0))
    auc_population = payload.get("auc_population")
    if auc_population is None:
        if payload.get("auc_observed_only"):
            auc_population = "observed_external_score_items"
        elif float(coverage) < 0.999999:
            auc_population = "prefix_eligible_test_items"
        else:
            auc_population = "all_scored_test_items"
    row: dict[str, Any] = {
        "split": split_name,
        "method": method,
        "n_items": n_items,
        "n_test_items": n_test_items,
        "score_coverage": float(coverage),
        "n_fallback_items": int(fallback or 0),
        "auc_population": auc_population,
    }
    for metric in METRICS:
        value = payload.get(metric)
        row[metric] = float(value) if isinstance(value, (int, float)) else np.nan
    return row


def collect_benchmark(
    results_root: Path,
    split_names: Sequence[str],
    include_appendix: bool = True,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Collect normalized benchmark rows and non-fatal audit findings."""
    rows: list[dict[str, Any]] = []
    findings: list[dict[str, Any]] = []
    for split_name in split_names:
        for method in canonical_methods(include_appendix=include_appendix):
            if not method_supports_split(method, split_name):
                continue
            resolved = resolve_method(results_root, split_name, method)
            if resolved is None:
                findings.append(
                    {
                        "severity": "INFO",
                        "code": "missing_metrics",
                        "split": split_name,
                        "method": method,
                        "message": "No eligible metrics.json was found.",
                    }
                )
                continue
            _, metrics = resolved
            row = _metric_row(split_name, method, metrics)
            rows.append(row)
            if row["score_coverage"] < 0.999999:
                findings.append(
                    {
                        "severity": "INFO",
                        "code": "partial_score_coverage",
                        "split": split_name,
                        "method": method,
                        "message": f"Score coverage is {row['score_coverage']:.2%}.",
                    }
                )
            if row["auc_population"] != "all_scored_test_items":
                findings.append(
                    {
                        "severity": "INFO",
                        "code": "restricted_auc_population",
                        "split": split_name,
                        "method": method,
                        "message": f"ROC-AUC population: {row['auc_population']}.",
                    }
                )
    return pd.DataFrame(rows), findings


def subset_stats(
    frame: pd.DataFrame, min_vote_thresholds: Sequence[int] = (5, 10, 20)
) -> dict[str, Any]:
    """Calculate coverage, density, and class-balance statistics."""
    if frame.empty:
        base: dict[str, Any] = {
            "n_votes": 0,
            "n_posts": 0,
            "n_users": 0,
            "n_communities": 0,
            "avg_votes_per_user": 0.0,
            "median_votes_per_user": 0.0,
            "avg_votes_per_post": 0.0,
        }
        for threshold in min_vote_thresholds:
            base[f"pct_users_ge_{threshold}_votes"] = 0.0
        return base

    votes_per_user = frame.groupby("username").size()
    votes_per_post = frame.groupby("item_id").size()
    stats: dict[str, Any] = {
        "n_votes": int(len(frame)),
        "n_posts": int(frame["item_id"].nunique()),
        "n_users": int(frame["username"].nunique()),
        "n_communities": int(frame["community"].nunique()) if "community" in frame else None,
        "avg_votes_per_user": float(votes_per_user.mean()),
        "median_votes_per_user": float(votes_per_user.median()),
        "avg_votes_per_post": float(votes_per_post.mean()),
    }
    for threshold in min_vote_thresholds:
        stats[f"pct_users_ge_{threshold}_votes"] = float((votes_per_user >= threshold).mean())
    if "community" in frame:
        user_community_counts = frame.groupby(["username", "community"]).size()
        for threshold in min_vote_thresholds:
            stats[f"pct_user_community_pairs_ge_{threshold}_votes"] = float(
                (user_community_counts >= threshold).mean()
            )
    if "label" in frame:
        labels = frame.drop_duplicates("item_id")["label"]
        stats["pct_label_approve"] = float((labels == 1).mean())
        stats["pct_label_remove"] = float((labels == -1).mean())
    return stats


def collect_split_statistics(splits_root: Path) -> pd.DataFrame:
    """Return one tidy table for every prepared split partition."""
    rows: list[dict[str, Any]] = []
    for split_name, split_dir in discover_splits(splits_root).items():
        train, validation, test = load_vote_partitions(split_dir)
        partitions = {
            "train": train,
            "val": validation,
            "test": test,
            "total": pd.concat([train, validation, test], ignore_index=True),
        }
        for partition, frame in partitions.items():
            rows.append({"split": split_name, "partition": partition, **subset_stats(frame)})
    return pd.DataFrame(rows)


def _prediction_spec(method: str) -> MethodSpec:
    if method == "SEF":
        return MethodSpec(
            "SEF",
            "expertise",
            prediction_files=("item_scores.parquet",),
            prediction_columns=("predicted",),
            score_columns=("meta_proba", "weighted_vote"),
        )
    return METHOD_SPECS[method]


def _first_existing(directory: Path, names: Sequence[str]) -> Path | None:
    for name in names:
        path = directory / name
        if path.exists():
            return path
    return None


def load_predictions(results_root: Path, split_name: str, method: str) -> pd.DataFrame:
    """Load and normalize item-level labels, decisions, and optional scores."""
    resolved = resolve_method(results_root, split_name, method)
    if resolved is None:
        raise FileNotFoundError(f"No eligible {method} result for {split_name}")
    directory, metrics = resolved
    spec = _prediction_spec(method)
    prediction_path = _first_existing(directory, spec.prediction_files)
    if prediction_path is None:
        raise FileNotFoundError(f"No prediction file for {method} in {directory}")
    frame = pd.read_parquet(prediction_path)
    required = {"item_id", "label"}
    if not required.issubset(frame.columns):
        missing = sorted(required - set(frame.columns))
        raise ValueError(f"{prediction_path} is missing columns: {missing}")
    if frame["item_id"].duplicated().any():
        raise ValueError(f"Duplicate item_id values in {prediction_path}")

    prediction_column = next(
        (column for column in spec.prediction_columns if column in frame.columns), None
    )
    if prediction_column is None and method == "NVSE":
        score_column = next(
            (column for column in spec.score_columns if column in frame.columns), None
        )
        threshold = _test_payload(metrics).get("threshold", metrics.get("threshold"))
        if score_column is not None and isinstance(threshold, (int, float)):
            frame = frame.copy()
            frame["_prediction"] = np.where(frame[score_column] >= threshold, 1, -1)
            prediction_column = "_prediction"
    if prediction_column is None:
        raise ValueError(f"No prediction column recognized in {prediction_path}")

    score_column = next((column for column in spec.score_columns if column in frame.columns), None)
    columns = ["item_id", "label", prediction_column]
    if score_column is not None:
        columns.append(score_column)
    fallback_column = next(
        (column for column in ("is_fallback", "used_fallback") if column in frame.columns),
        None,
    )
    if fallback_column is not None:
        columns.append(fallback_column)
    source_column = next(
        (column for column in ("decision_source", "score_source") if column in frame.columns),
        None,
    )
    if source_column is not None:
        columns.append(source_column)
    normalized = frame[columns].rename(columns={prediction_column: "prediction"}).copy()
    if score_column is not None:
        normalized = normalized.rename(columns={score_column: "score"})
    normalized["item_id"] = normalized["item_id"].astype(str)
    normalized["label"] = normalized["label"].astype(int)
    normalized["prediction"] = normalized["prediction"].astype(int)
    if fallback_column is not None:
        normalized["is_fallback"] = normalized[fallback_column].fillna(False).astype(bool)
        if fallback_column != "is_fallback":
            normalized = normalized.drop(columns=fallback_column)
    elif source_column is not None:
        normalized["is_fallback"] = (
            normalized[source_column]
            .astype("string")
            .str.contains("fallback", case=False, na=False)
        )
    else:
        normalized["is_fallback"] = False
    if source_column is not None:
        normalized = normalized.drop(columns=source_column)
    invalid_labels = set(normalized["label"].unique()) - {-1, 1}
    invalid_predictions = set(normalized["prediction"].unique()) - {-1, 1}
    if invalid_labels or invalid_predictions:
        raise ValueError(
            f"Invalid labels/predictions in {prediction_path}: "
            f"labels={invalid_labels}, predictions={invalid_predictions}"
        )
    return normalized.sort_values("item_id").reset_index(drop=True)


def collect_native_fallback_performance(
    results_root: Path,
    split_names: Sequence[str] = FIXED_SPLITS,
) -> pd.DataFrame:
    """Summarize native, fallback-only, and final hybrid hard-decision performance."""
    rows: list[dict[str, Any]] = []
    for split_name in split_names:
        for method in canonical_methods(include_appendix=True):
            if not method_supports_split(method, split_name):
                continue
            try:
                predictions = load_predictions(results_root, split_name, method)
            except FileNotFoundError:
                continue
            fallback = predictions["is_fallback"].astype(bool)
            if not fallback.any():
                continue
            native_frame = predictions.loc[~fallback]
            fallback_frame = predictions.loc[fallback]
            hybrid_metrics = _point_metrics(
                predictions["label"].to_numpy(dtype=int),
                predictions["prediction"].to_numpy(dtype=int),
            )

            def subset_metrics(frame: pd.DataFrame) -> dict[str, float]:
                if frame.empty:
                    return {"macro_f1": np.nan, "f1_neg": np.nan}
                return _point_metrics(
                    frame["label"].to_numpy(dtype=int),
                    frame["prediction"].to_numpy(dtype=int),
                )

            native_metrics = subset_metrics(native_frame)
            fallback_metrics = subset_metrics(fallback_frame)
            rows.append(
                {
                    "split": split_name,
                    "method": method,
                    "n_total": len(predictions),
                    "n_native": len(native_frame),
                    "native_share": len(native_frame) / len(predictions),
                    "native_macro_f1": native_metrics["macro_f1"],
                    "native_f1_remove": native_metrics["f1_neg"],
                    "n_fallback": len(fallback_frame),
                    "fallback_macro_f1": fallback_metrics["macro_f1"],
                    "fallback_f1_remove": fallback_metrics["f1_neg"],
                    "hybrid_macro_f1": hybrid_metrics["macro_f1"],
                    "hybrid_f1_remove": hybrid_metrics["f1_neg"],
                }
            )
    return pd.DataFrame(rows)


def audit_predictions(
    results_root: Path,
    splits_root: Path,
    split_names: Sequence[str],
    methods: Sequence[str],
) -> list[dict[str, Any]]:
    """Audit predictions against TEST membership, labels, and reported metrics."""
    findings: list[dict[str, Any]] = []
    split_directories = discover_splits(splits_root)
    for split_name in split_names:
        split_dir = split_directories.get(split_name)
        if split_dir is None:
            findings.append(
                {
                    "severity": "ERROR",
                    "code": "missing_test_split",
                    "split": split_name,
                    "method": None,
                    "message": "The prepared TEST partition is missing.",
                }
            )
            continue
        test_votes = pd.read_parquet(split_dir / "test_votes.parquet")
        expected = (
            test_votes[["item_id", "label"]]
            .assign(item_id=lambda frame: frame["item_id"].astype(str))
            .drop_duplicates("item_id")
            .set_index("item_id")["label"]
            .astype(int)
        )
        for method in methods:
            if not method_supports_split(method, split_name):
                continue
            try:
                predictions = load_predictions(results_root, split_name, method)
            except FileNotFoundError as exc:
                findings.append(
                    {
                        "severity": "INFO",
                        "code": "missing_predictions",
                        "split": split_name,
                        "method": method,
                        "message": str(exc),
                    }
                )
                continue
            except ValueError as exc:
                findings.append(
                    {
                        "severity": "ERROR",
                        "code": "invalid_predictions",
                        "split": split_name,
                        "method": method,
                        "message": str(exc),
                    }
                )
                continue
            prediction_ids = set(predictions["item_id"])
            expected_ids = set(expected.index)
            unknown_ids = prediction_ids - expected_ids
            missing_ids = expected_ids - prediction_ids
            if unknown_ids:
                findings.append(
                    {
                        "severity": "ERROR",
                        "code": "prediction_items_outside_test",
                        "split": split_name,
                        "method": method,
                        "message": f"Predictions contain {len(unknown_ids)} item(s) outside TEST.",
                    }
                )
            if missing_ids:
                findings.append(
                    {
                        "severity": "INFO",
                        "code": "incomplete_prediction_population",
                        "split": split_name,
                        "method": method,
                        "message": f"Predictions omit {len(missing_ids)} of {len(expected_ids)} TEST items.",
                    }
                )
            shared = predictions[predictions["item_id"].isin(expected_ids)].set_index("item_id")
            expected_labels = expected.loc[shared.index].to_numpy(dtype=int)
            if not np.array_equal(expected_labels, shared["label"].to_numpy(dtype=int)):
                findings.append(
                    {
                        "severity": "ERROR",
                        "code": "test_label_mismatch",
                        "split": split_name,
                        "method": method,
                        "message": "Prediction labels disagree with the prepared TEST partition.",
                    }
                )

            resolved = resolve_method(results_root, split_name, method)
            if resolved is not None and len(shared):
                reported = _test_payload(resolved[1])
                recomputed = _point_metrics(
                    shared["label"].to_numpy(dtype=int),
                    shared["prediction"].to_numpy(dtype=int),
                )
                for metric in AUDIT_METRICS:
                    reported_value = reported.get(metric)
                    if isinstance(reported_value, (int, float)) and not np.isclose(
                        recomputed[metric], float(reported_value), atol=1e-9, rtol=1e-7
                    ):
                        findings.append(
                            {
                                "severity": "ERROR",
                                "code": "metric_prediction_mismatch",
                                "split": split_name,
                                "method": method,
                                "message": (
                                    f"Reported {metric}={float(reported_value):.6f}, but "
                                    f"item predictions yield {recomputed[metric]:.6f}."
                                ),
                            }
                        )
            findings.append(
                {
                    "severity": "INFO",
                    "code": "prediction_population",
                    "split": split_name,
                    "method": method,
                    "message": f"Loaded {len(predictions):,} unique item predictions.",
                }
            )
    return findings


def _point_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    tp = float(np.sum((labels == 1) & (predictions == 1)))
    tn = float(np.sum((labels == -1) & (predictions == -1)))
    fp = float(np.sum((labels == -1) & (predictions == 1)))
    fn = float(np.sum((labels == 1) & (predictions == -1)))
    f1_pos = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    f1_neg = 2 * tn / (2 * tn + fp + fn) if 2 * tn + fp + fn else 0.0
    return {"macro_f1": (f1_pos + f1_neg) / 2, "f1_neg": f1_neg}


def collect_cn_extensions(
    results_root: Path, split_names: Sequence[str]
) -> pd.DataFrame:
    """Collect the Community Notes extension ablation."""
    rows: list[dict[str, Any]] = []
    for split_name in split_names:
        cn = resolve_method(results_root, split_name, "CN")
        if cn is not None:
            rows.append(_metric_row(split_name, "CN", cn[1]))
        split_dir = _split_result_dir(results_root, split_name)
        for variant, label in CN_EXTENSION_VARIANTS.items():
            metrics_path = split_dir / "ma_qsmf" / variant / "metrics.json"
            if not metrics_path.exists():
                continue
            metrics = _read_json(metrics_path)
            row = _metric_row(split_name, label, metrics)
            row["selected_lambda_y"] = metrics.get("selected_lambda_y")
            row["moderator_aligned"] = metrics.get("moderator_aligned")
            row["learn_rho"] = metrics.get("learn_rho")
            rows.append(row)
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    cn_scores = frame.loc[frame["method"] == "CN", ["split", "macro_f1", "f1_neg"]].rename(
        columns={"macro_f1": "cn_macro_f1", "f1_neg": "cn_f1_neg"}
    )
    frame = frame.merge(cn_scores, on="split", how="left")
    frame["delta_macro_f1_vs_cn"] = frame["macro_f1"] - frame["cn_macro_f1"]
    frame["delta_f1_neg_vs_cn"] = frame["f1_neg"] - frame["cn_f1_neg"]
    return frame


def _write_table(frame: pd.DataFrame, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(stem.with_suffix(".csv"), index=False)


def _save_figure(fig: plt.Figure, output_stem: Path) -> None:
    """Save one report-facing PNG figure."""
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def _method_bar_plot(
    frame: pd.DataFrame,
    metric: str,
    title: str,
    output_stem: Path,
) -> None:
    """Draw the classic coloured method bar chart used by the earlier report."""
    subset = frame.dropna(subset=[metric]).sort_values(metric, ascending=False).copy()
    if subset.empty:
        return
    labels = [method_label(method) for method in subset["method"]]
    values = subset[metric].to_numpy(dtype=float)
    colors = [METHOD_COLORS.get(method, "#8c8c8c") for method in subset["method"]]
    width = max(11.0, 0.82 * len(subset))
    fig, axis = plt.subplots(figsize=(width, 6.2))
    bars = axis.bar(labels, values, color=colors, alpha=0.92, width=0.8)
    axis.bar_label(bars, labels=[f"{value:.2f}" for value in values], padding=4, fontsize=9)
    axis.set_title(title, fontsize=15, pad=14)
    axis.set_ylabel(METRIC_LABELS[metric])
    axis.set_ylim(0, 1)
    axis.grid(axis="y", linestyle="--", alpha=0.28)
    axis.set_axisbelow(True)
    axis.tick_params(axis="x", rotation=28, labelsize=9)
    for label in axis.get_xticklabels():
        label.set_horizontalalignment("right")
    fig.tight_layout()
    _save_figure(fig, output_stem)


def _plot_benchmark(frame: pd.DataFrame, output_dir: Path) -> None:
    """Create one coloured bar chart per fixed split and metric."""
    fixed = frame[frame["split"].isin(FIXED_SPLITS)].copy()
    if fixed.empty:
        return
    split_labels = {
        "full": "Full",
        "intersection": "Intersection",
        "intersection_chronological": "Intersection chronological",
    }
    for split_name in FIXED_SPLITS:
        subset = fixed[fixed["split"] == split_name]
        for metric in METRICS:
            _method_bar_plot(
                subset,
                metric,
                f"{METRIC_LABELS[metric]} - {split_labels[split_name]}",
                output_dir / f"{metric}__split_{split_name}",
            )


def _kfold_dataset(split_name: str) -> str:
    parts = split_name.split("/")
    if len(parts) != 2 or not parts[1].startswith("fold_"):
        raise ValueError(f"Unexpected K-fold split name: {split_name}")
    return parts[0]


def _plot_kfold_metric(
    frame: pd.DataFrame, dataset: str, metric: str, output_stem: Path
) -> None:
    """Plot method means with sample-standard-deviation error bars."""
    subset = frame[frame["dataset"] == dataset]
    metric_order = subset.groupby("method", sort=False)[metric].mean().sort_values(ascending=False)
    methods = metric_order.index.tolist()
    grouped = subset.groupby("method", sort=False)[metric]
    means = grouped.mean().reindex(methods)
    deviations = grouped.std(ddof=1).reindex(methods).fillna(0.0)
    counts = grouped.count().reindex(methods).fillna(0).astype(int)
    valid = means.notna()
    methods = [method for method, keep in zip(methods, valid, strict=True) if keep]
    means, deviations, counts = means[valid], deviations[valid], counts[valid]
    if not methods:
        return

    fig, axis = plt.subplots(figsize=(max(11.0, 0.82 * len(methods)), 6.2))
    x = np.arange(len(methods))
    bars = axis.bar(
        x,
        means.to_numpy(dtype=float),
        yerr=deviations.to_numpy(dtype=float),
        capsize=4,
        color=[METHOD_COLORS.get(method, "#8c8c8c") for method in methods],
        alpha=0.92,
        width=0.8,
    )
    axis.bar_label(bars, labels=[f"{value:.2f}" for value in means], padding=6, fontsize=9)
    n_folds = int(counts.max())
    axis.set_title(
        f"{METRIC_LABELS[metric]} - K-fold {dataset}\nn = {n_folds}", fontsize=15, pad=14
    )
    axis.set_ylabel(METRIC_LABELS[metric])
    axis.set_ylim(0, 1)
    axis.set_xticks(x)
    axis.set_xticklabels(
        [method_label(method) for method in methods], rotation=28, ha="right", fontsize=9
    )
    axis.grid(axis="y", linestyle="--", alpha=0.28)
    axis.set_axisbelow(True)
    fig.tight_layout()
    _save_figure(fig, output_stem)


def _kfold_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarize fold-level metrics in conceptual method-family order."""
    grouped = frame.groupby(["dataset", "method"], sort=False)[list(METRICS)].agg(
        ["mean", "std", "count"]
    )
    grouped.columns = [f"{metric}_{stat}" for metric, stat in grouped.columns]
    summary = grouped.reset_index()

    family_lookup = {
        method: family for family, methods in METHOD_FAMILIES.items() for method in methods
    }
    family_order = {family: index for index, family in enumerate(METHOD_FAMILIES)}
    method_order = {
        method: index
        for methods in METHOD_FAMILIES.values()
        for index, method in enumerate(methods)
    }
    dataset_order = {dataset: index for index, dataset in enumerate(FIXED_SPLITS)}
    summary.insert(1, "family", summary["method"].map(family_lookup))
    summary["_dataset_order"] = summary["dataset"].map(dataset_order)
    summary["_family_order"] = summary["family"].map(family_order)
    summary["_method_order"] = summary["method"].map(method_order)
    summary = summary.sort_values(
        ["_dataset_order", "_family_order", "_method_order"], kind="stable"
    )
    return summary.drop(columns=["_dataset_order", "_family_order", "_method_order"])


def _plot_kfold_macro_f1_by_family(frame: pd.DataFrame, output_stem: Path) -> None:
    """Plot all methods by family with one mean and SD interval per protocol."""
    summary = _kfold_summary(frame)
    if summary.empty:
        return

    available = set(summary["method"])
    methods: list[str] = []
    positions: list[float] = []
    family_ranges: list[tuple[str, float, float]] = []
    cursor = 0.0
    for family, family_methods in METHOD_FAMILIES.items():
        present = [method for method in family_methods if method in available]
        if not present:
            continue
        start = cursor
        for method in present:
            methods.append(method)
            positions.append(cursor)
            cursor += 1.0
        family_ranges.append((family, start - 0.48, cursor - 0.52))
        cursor += 0.65

    fig, axis = plt.subplots(figsize=(10.8, 9.2))
    family_shades = ("#f3f4f6", "#fff7ed", "#f0fdfa", "#eff6ff")
    for (family, lower, upper), shade in zip(family_ranges, family_shades, strict=False):
        axis.axhspan(lower, upper, color=shade, zorder=0)
        axis.text(
            0.805,
            (lower + upper) / 2,
            family,
            ha="right",
            va="center",
            fontsize=8,
            color="#374151",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.5},
        )

    dataset_styles = {
        "full": ("o", "#7c3aed", -0.21),
        "intersection": ("s", "#0f766e", 0.0),
        "intersection_chronological": ("^", "#c2410c", 0.21),
    }
    position_lookup = dict(zip(methods, positions, strict=True))
    for dataset in FIXED_SPLITS:
        marker, color, offset = dataset_styles[dataset]
        subset = summary[summary["dataset"] == dataset].set_index("method")
        present = [method for method in methods if method in subset.index]
        means = subset.loc[present, "macro_f1_mean"].to_numpy(dtype=float)
        deviations = subset.loc[present, "macro_f1_std"].fillna(0.0).to_numpy(dtype=float)
        y = np.array([position_lookup[method] + offset for method in present])
        axis.errorbar(
            means,
            y,
            xerr=deviations,
            fmt=marker,
            markersize=5.2,
            capsize=2.5,
            elinewidth=1.1,
            color=color,
            linestyle="none",
            label=KFOLD_DATASET_LABELS[dataset],
            zorder=3,
        )

    axis.set_yticks(positions)
    axis.set_yticklabels([method_label(method) for method in methods], fontsize=9)
    axis.invert_yaxis()
    axis.set_xlim(0.40, 0.81)
    axis.set_xlabel("Mean macro-F1 (error bars: sample SD across five folds)")
    axis.set_title("Five-fold macro-F1 by method family", fontsize=15, pad=12)
    axis.grid(axis="x", linestyle="--", alpha=0.28)
    axis.set_axisbelow(True)
    axis.legend(loc="lower right", frameon=True, fontsize=9)
    fig.tight_layout()
    _save_figure(fig, output_stem)


def run_kfold_graphs(results_root: Path, splits_root: Path, output_dir: Path) -> int:
    """Generate K-fold figures, or skip cleanly when K-fold inputs are absent."""
    kfold_results = results_root if results_root.name == "kfold" else results_root / "kfold"
    kfold_splits = splits_root if splits_root.name == "kfold" else splits_root / "kfold"
    if not kfold_results.is_dir() or not kfold_splits.is_dir():
        log.info("K-fold inputs not found; skipping K-fold graphs.")
        return 0
    splits = discover_splits(kfold_splits)
    if not splits:
        log.info("No prepared K-fold splits found; skipping K-fold graphs.")
        return 0
    fold_metrics, _ = collect_benchmark(kfold_results, list(splits))
    if fold_metrics.empty:
        log.info("No K-fold method results found; skipping K-fold graphs.")
        return 0
    fold_metrics.insert(0, "dataset", fold_metrics["split"].map(_kfold_dataset))
    figures_dir = output_dir / "figures"
    tables_dir = output_dir / "tables"
    _write_table(fold_metrics, tables_dir / "fold_metrics")
    _write_table(_kfold_summary(fold_metrics), tables_dir / "summary_by_family")
    generated = 0
    for dataset in sorted(fold_metrics["dataset"].unique()):
        for metric in METRICS:
            output_stem = figures_dir / f"{metric}__kfold_{dataset}"
            _plot_kfold_metric(fold_metrics, dataset, metric, output_stem)
            generated += int(output_stem.with_suffix(".png").exists())
    family_stem = figures_dir / "macro_f1__kfold_by_family"
    _plot_kfold_macro_f1_by_family(fold_metrics, family_stem)
    generated += int(family_stem.with_suffix(".png").exists())
    log.info("K-fold graphs saved to %s (%d figures)", figures_dir, generated)
    return generated


def _plot_scaling(frame: pd.DataFrame, population: str, output_stem: Path) -> None:
    """Aggregate all training windows of one population into bar-chart figures."""
    prefix = f"windows/{population}/"
    subset = frame[frame["split"].str.startswith(prefix, na=False)].copy()
    if subset.empty:
        return
    subset["window"] = subset["split"].str.rsplit("/", n=1).str[-1].str[1:].astype(int)
    windows = sorted(subset["window"].unique())
    for metric in METRICS:
        fig, axes = plt.subplots(2, 3, figsize=(22, 11), sharey=True)
        for axis, window in zip(axes.flat, windows):
            window_rows = (
                subset[subset["window"] == window]
                .dropna(subset=[metric])
                .sort_values(metric, ascending=False)
            )
            labels = [method_label(method) for method in window_rows["method"]]
            values = window_rows[metric].to_numpy(dtype=float)
            colors = [METHOD_COLORS.get(method, "#8c8c8c") for method in window_rows["method"]]
            bars = axis.bar(labels, values, color=colors, alpha=0.92)
            axis.bar_label(
                bars, labels=[f"{value:.2f}" for value in values], padding=2, fontsize=7
            )
            axis.set_title(WINDOW_LABELS.get(int(window), f"{window}% training"))
            axis.set_ylim(0, 1)
            axis.grid(axis="y", linestyle="--", alpha=0.25)
            axis.set_axisbelow(True)
            axis.tick_params(axis="x", rotation=38, labelsize=7)
            for label in axis.get_xticklabels():
                label.set_horizontalalignment("right")
        for axis in axes.flat[len(windows) :]:
            axis.remove()
        axes[0, 0].set_ylabel(METRIC_LABELS[metric])
        axes[1, 0].set_ylabel(METRIC_LABELS[metric])
        fig.suptitle(f"{METRIC_LABELS[metric]} - windowed {population}", fontsize=16, y=1.01)
        fig.tight_layout()
        _save_figure(fig, output_stem.parent / f"{metric}__windows_{population}")


def _plot_cn_extensions(frame: pd.DataFrame, output_stem: Path) -> None:
    if frame.empty:
        return
    methods = ["CN", *CN_EXTENSION_VARIANTS.values()]
    available = [method for method in methods if method in set(frame["method"])]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(len(FIXED_SPLITS))
    for axis, metric in zip(axes, ("macro_f1", "f1_neg")):
        for method in available:
            indexed = frame[frame["method"] == method].set_index("split")
            values = [
                indexed.at[split, metric] if split in indexed.index else np.nan
                for split in FIXED_SPLITS
            ]
            axis.plot(x, values, marker="o", label=method)
        axis.set_title(METRIC_LABELS[metric])
        axis.set_xticks(x)
        axis.set_xticklabels(("Full", "Intersection", "Chronological"), rotation=20)
        axis.set_ylim(0, 1)
        axis.grid(linestyle="--", alpha=0.35)
    axes[0].set_ylabel("Test performance")
    axes[0].legend(fontsize=8)
    fig.suptitle("Community Notes extensions", fontsize=14)
    fig.tight_layout()
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_report(
    results_root: Path,
    splits_root: Path,
    output_dir: Path,
    sections: set[str],
    strict_audit: bool = False,
) -> None:
    """Run selected report sections and write a coherent artifact bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    discovered = discover_splits(splits_root)
    split_names = list(discovered)
    if not split_names:
        raise FileNotFoundError(f"No prepared splits found under {splits_root}")

    method_split_matrix = collect_method_split_matrix(results_root, splits_root)
    _write_table(method_split_matrix, tables_dir / "method_split_matrix")

    benchmark = pd.DataFrame()
    split_statistics = pd.DataFrame()
    cn_extensions = pd.DataFrame()
    findings: list[dict[str, Any]] = []

    needs_benchmark = bool(sections & {"audit", "benchmark", "scaling", "all"})
    if needs_benchmark:
        benchmark, benchmark_findings = collect_benchmark(results_root, split_names)
        findings.extend(benchmark_findings)
        _write_table(benchmark, tables_dir / "benchmark_all_methods")
        main_methods = set(canonical_methods(False))
        main = benchmark[benchmark["method"].isin(main_methods)].copy()
        _write_table(main, tables_dir / "benchmark_main")
        coverage_columns = [
            "split",
            "method",
            "n_items",
            "n_test_items",
            "score_coverage",
            "n_fallback_items",
            "auc_population",
        ]
        _write_table(benchmark[coverage_columns], tables_dir / "coverage_and_fallback")
        native_fallback = collect_native_fallback_performance(results_root)
        _write_table(native_fallback, tables_dir / "native_fallback_performance")
        if sections & {"benchmark", "all"}:
            _plot_benchmark(main, figures_dir)
        if sections & {"scaling", "all"}:
            for population in WINDOW_POPULATIONS:
                _plot_scaling(main, population, figures_dir / f"scaling_{population}")

    if sections & {"splits", "all"}:
        split_statistics = collect_split_statistics(splits_root)
        _write_table(split_statistics, tables_dir / "split_statistics")

    if sections & {"audit", "all"}:
        findings.extend(
            audit_predictions(
                results_root,
                splits_root,
                FIXED_SPLITS,
                canonical_methods(include_appendix=True),
            )
        )

    if sections & {"cn-extensions", "all"}:
        cn_extensions = collect_cn_extensions(results_root, FIXED_SPLITS)
        _write_table(
            cn_extensions, tables_dir / "community_notes_extension_ablation"
        )
        _plot_cn_extensions(
            cn_extensions, figures_dir / "community_notes_extension_ablation"
        )

    errors = sum(finding["severity"] == "ERROR" for finding in findings)
    log.info("Report bundle saved to %s", output_dir)
    log.info("Audit completed with %d error(s)", errors)
    if errors and strict_audit:
        raise RuntimeError(f"Report audit found {errors} invalid prediction artifact(s).")
    if errors:
        log.error(
            "Report generation completed despite %d audit error(s); "
            "pass --strict-audit to fail on them.",
            errors,
        )


def _parse_sections(values: Sequence[str]) -> set[str]:
    valid = {
        "all",
        "audit",
        "splits",
        "benchmark",
        "scaling",
        "cn-extensions",
    }
    sections = set(values)
    unknown = sections - valid
    if unknown:
        raise ValueError(f"Unknown report section(s): {sorted(unknown)}")
    return {"all"} if "all" in sections else sections


def main(
    section: Annotated[
        list[str] | None,
        typer.Option(
            "--section",
            help=(
                "Repeat to run all, audit, splits, benchmark, scaling, or cn-extensions."
            ),
        ),
    ] = None,
    results_root: Annotated[Path, typer.Option()] = REDDIT.results,
    splits_root: Annotated[Path, typer.Option()] = REDDIT.splits,
    output_dir: Annotated[Path, typer.Option()] = REDDIT.report,
    strict_audit: Annotated[bool, typer.Option()] = False,
    log_level: Annotated[str, typer.Option()] = "INFO",
    k_fold_only: Annotated[bool, typer.Option("--k-fold-only")] = False,
) -> None:
    """Run the unified comparison workflow."""
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise typer.BadParameter(f"Unknown log level: {log_level}", param_hint="--log-level")
    logging.basicConfig(level=numeric_level, format="%(message)s")
    if k_fold_only:
        run_kfold_graphs(results_root, splits_root, output_dir)
        return
    try:
        sections = _parse_sections(section or ["all"])
    except ValueError as error:
        raise typer.BadParameter(str(error), param_hint="--section") from error
    run_report(
        results_root=results_root,
        splits_root=splits_root,
        output_dir=output_dir,
        sections=sections,
        strict_audit=strict_audit,
    )
    run_kfold_graphs(results_root, splits_root, output_dir / "kfold")


if __name__ == "__main__":
    typer.run(main)
