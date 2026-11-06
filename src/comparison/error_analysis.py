"""Generate the thesis error audit for the fixed Intersection test."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.comparison.graphs import load_predictions
from src.layout import REDDIT

plt.switch_backend("Agg")

OUTPUT_DIR = REDDIT.root / "results" / "error_analysis"

log = logging.getLogger(__name__)

METHOD_FAMILIES: dict[str, tuple[str, ...]] = {
    "Baseline": ("Net", "WNet", "SubAdj", "RedditScore", "RedditScore+Sub"),
    "Factorization": ("CN", "CN inductive", "QSMF", "MA-MF", "MA-QSMF"),
    "Observed-voter expertise": ("VAR", "NVSE", "SEF"),
    "Eligible voter base": (
        "Bandit",
        "GraphPropagation",
        "RoleNSGA2",
        "VirtualEnsembles",
        "BoC",
    ),
}
METHODS = tuple(method for family in METHOD_FAMILIES.values() for method in family)
FIXED_PROTOCOLS = {
    "Full": "full",
    "Intersection": "intersection",
    "Chronological": "intersection_chronological",
}


def _prepare_output_dir(output_dir: Path) -> Path:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _load_items(splits_root: Path) -> pd.DataFrame:
    path = splits_root / "intersection" / "test_votes.parquet"
    votes = pd.read_parquet(path)
    votes["item_id"] = votes["item_id"].astype(str)
    if (votes.groupby("item_id")["label"].nunique(dropna=False) > 1).any():
        raise ValueError("Intersection labels vary within a post")
    votes["positive_vote"] = votes["vote"] == 1
    items = votes.groupby("item_id", as_index=True).agg(
        label=("label", "first"),
        n_votes=("vote", "size"),
        positive_vote_share=("positive_vote", "mean"),
    )
    items["crowd_prediction"] = np.where(items["positive_vote_share"] >= 0.5, 1, -1)
    items["label_group"] = np.where(items["label"] == -1, "Removal", "Approval")
    items["crowd_relation"] = np.where(
        items["crowd_prediction"] == items["label"], "Agree", "Disagree"
    )
    return items


def _load_errors(
    items: pd.DataFrame,
    results_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    predictions = {
        method: load_predictions(results_root, "intersection", method).set_index("item_id")
        for method in METHODS
    }
    common_items = set(items.index)
    for frame in predictions.values():
        common_items &= set(frame.index.astype(str))
    item_index = items.loc[sorted(common_items)].copy()
    if len(item_index) != 767:
        raise ValueError(f"Expected 767 common Intersection posts, found {len(item_index)}")

    errors: dict[str, pd.Series] = {}
    fallbacks: dict[str, pd.Series] = {}
    labels = item_index["label"].astype(int)
    for method, frame in predictions.items():
        frame.index = frame.index.astype(str)
        selected = frame.loc[item_index.index]
        if not np.array_equal(selected["label"].astype(int).to_numpy(), labels.to_numpy()):
            raise ValueError(f"{method} labels disagree with the Intersection test")
        decision = selected["prediction"].astype(int)
        errors[method] = decision != labels
        fallbacks[method] = selected["is_fallback"].astype(bool)
    return item_index, pd.DataFrame(errors), pd.DataFrame(fallbacks)


def _result_row(
    finding: str,
    measure: str,
    subject: str,
    *,
    protocol: str = "",
    comparison: str = "",
    group: str = "",
    count: int | None = None,
    denominator: int | None = None,
    value: float | None = None,
) -> dict[str, object]:
    return {
        "finding": finding,
        "measure": measure,
        "protocol": protocol,
        "subject": subject,
        "comparison": comparison,
        "group": group,
        "count": count,
        "denominator": denominator,
        "value": value,
    }


def _cn_crowd_alignment_results(splits_root: Path, results_root: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for protocol, split in FIXED_PROTOCOLS.items():
        split_root = splits_root / split
        test = pd.read_parquet(split_root / "test_votes.parquet")
        test["item_id"] = test["item_id"].astype(str)
        if (test.groupby("item_id")["label"].nunique(dropna=False) > 1).any():
            raise ValueError(f"{protocol} labels vary within a post")
        test["positive_vote"] = test["vote"] == 1
        posts = test.groupby("item_id", as_index=True).agg(
            label=("label", "first"),
            positive_vote_share=("positive_vote", "mean"),
        )
        posts["crowd_prediction"] = np.where(posts["positive_vote_share"] >= 0.5, 1, -1)

        cn = pd.read_parquet(results_root / split / "cn" / "item_scores.parquet").set_index(
            "item_id"
        )
        cn.index = cn.index.astype(str)
        posts = posts.loc[cn.index].copy()
        if not np.array_equal(
            posts["label"].astype(int).to_numpy(), cn["label"].astype(int).to_numpy()
        ):
            raise ValueError(f"CN labels disagree with the {protocol} test")
        cn_error = cn["prediction"].astype(int) != cn["label"].astype(int)
        crowd_agrees = posts["crowd_prediction"] == posts["label"]

        for group, mask in (
            ("Crowd agrees with moderators", crowd_agrees),
            ("Crowd disagrees with moderators", ~crowd_agrees),
        ):
            count = int(cn_error.loc[mask].sum())
            denominator = int(mask.sum())
            rows.append(
                _result_row(
                    "cn_crowd_alignment",
                    "failure_rate",
                    "CN",
                    protocol=protocol,
                    group=group,
                    count=count,
                    denominator=denominator,
                    value=count / denominator,
                )
            )

        disagreement = ~crowd_agrees
        for label, label_value in (("Removal", -1), ("Approval", 1)):
            count = int((disagreement & posts["label"].eq(label_value)).sum())
            denominator = int(disagreement.sum())
            rows.append(
                _result_row(
                    "cn_crowd_alignment",
                    "class_composition",
                    "Crowd disagrees with moderators",
                    protocol=protocol,
                    group=label,
                    count=count,
                    denominator=denominator,
                    value=count / denominator,
                )
            )

    return pd.DataFrame(rows)


def _key_results(
    items: pd.DataFrame,
    errors: pd.DataFrame,
) -> pd.DataFrame:
    n_items = len(items)
    method_count = errors.sum(axis=1)
    all_methods = method_count == len(METHODS)
    rows = [
        _result_row("scope", "posts", "Intersection", count=n_items),
        _result_row("scope", "methods", "All methods", count=len(METHODS)),
    ]
    for threshold in range(1, len(METHODS) + 1):
        indicator = method_count >= threshold
        count = int(indicator.sum())
        rows.append(
            _result_row(
                "shared_error_curve",
                "posts_failed_by_at_least",
                f"{threshold} methods",
                count=count,
                denominator=n_items,
                value=count / n_items,
            )
        )
    universal_count = int(all_methods.sum())
    for group, group_mask in (
        ("Removal", items["label_group"] == "Removal"),
        ("Crowd disagrees", items["crowd_relation"] == "Disagree"),
    ):
        count = int((all_methods & group_mask).sum())
        rows.append(
            _result_row(
                "universal_error_composition",
                "share",
                "All 18 methods wrong",
                group=group,
                count=count,
                denominator=universal_count,
                value=count / universal_count,
            )
        )

    return pd.DataFrame(rows)


def _plot(
    method_count: pd.Series,
    path: Path,
) -> None:
    thresholds = np.arange(1, len(METHODS) + 1)
    counts = np.array([(method_count >= threshold).sum() for threshold in thresholds])
    fig, axis = plt.subplots(figsize=(8.2, 4.8))
    axis.plot(thresholds, counts, color="#315b7d", marker="o", linewidth=2.2, markersize=4.5)
    axis.fill_between(thresholds, counts, color="#315b7d", alpha=0.10)
    axis.set_xticks(thresholds)
    axis.set_xlim(1, len(METHODS))
    axis.set_ylim(0, 500)
    axis.set_xlabel("Minimum number of methods wrong on the same post")
    axis.set_ylabel("Posts")
    axis.grid(axis="y", color="#d1d5db", linewidth=0.8)
    axis.spines[["top", "right"]].set_visible(False)
    for threshold in (1, 9, 18):
        count = int(counts[threshold - 1])
        axis.scatter(threshold, count, s=48, color="#b23a48", zorder=3)
        axis.annotate(
            f"{count} posts ({count / len(method_count):.1%})",
            (threshold, count),
            xytext=(0, 20 if threshold != 1 else 18),
            textcoords="offset points",
            ha="center" if threshold != 1 else "left",
            fontsize=9,
            color="#7f1d1d",
        )
    fig.tight_layout()
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def run(
    results_root: Path = REDDIT.results,
    splits_root: Path = REDDIT.splits,
    output_dir: Path = OUTPUT_DIR,
) -> None:
    output_dir = _prepare_output_dir(output_dir)
    items, errors, fallbacks = _load_errors(_load_items(splits_root), results_root)
    results = _key_results(items, errors)
    results = pd.concat(
        [results, _cn_crowd_alignment_results(splits_root, results_root)], ignore_index=True
    )
    results["count"] = results["count"].astype("Int64")
    results["denominator"] = results["denominator"].astype("Int64")
    results.to_csv(output_dir / "failure_analysis.csv", index=False)

    method_count = errors.sum(axis=1)
    all_methods_wrong = method_count == 18
    if not (
        (items.loc[all_methods_wrong, "label_group"] == "Removal").all()
        and (items.loc[all_methods_wrong, "crowd_relation"] == "Disagree").all()
    ):
        raise ValueError("Universal failures do not match the reported composition")
    _plot(method_count, output_dir / "failure_analysis.png")

    fallback_counts = {
        method: int(fallbacks[method].sum())
        for method in fallbacks
        if int(fallbacks[method].sum()) > 0
    }
    log.info(
        "Wrote two artifacts to %s; final-prediction fallbacks: %s",
        output_dir,
        fallback_counts,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=REDDIT.results)
    parser.add_argument("--splits-root", type=Path, default=REDDIT.splits)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run(args.results_root, args.splits_root, args.output_dir)
