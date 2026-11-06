"""VAR-based expert selection and multi-split evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import KFold

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.layout import REDDIT  # noqa: E402
from src.methods.expert_selection.core import (  # noqa: E402
    calculate_var_scores,
    evaluate_fold,
)
from src.utils.net_vote_fallback import (  # noqa: E402
    add_net_vote_fallback,
    calibrate_net_vote_threshold,
)
from src.utils.splits import discover_splits, load_vote_partitions  # noqa: E402

# Aggregation

N_FOLDS = 5
K_VALUES = [10]
OUTPUT_ROOT = REDDIT.results
SCALAR_METRICS = [
    "macro_f1",
    "roc_auc",
    "f1_pos",
    "f1_neg",
    "macro_precision",
    "macro_recall",
    "precision_pos",
    "recall_pos",
    "precision_neg",
    "recall_neg",
]


def _base_votes(test_df, train_scores, k):
    """Count positive and negative votes for each item."""
    test_ranked = test_df.merge(
        train_scores[["community", "username", "VAR_score"]],
        on=["community", "username"],
        how="left",
    )
    test_ranked["VAR_score"] = test_ranked["VAR_score"].fillna(0)
    top_k_users = (
        test_ranked[["community", "username", "VAR_score"]]
        .drop_duplicates()
        .sort_values("VAR_score", ascending=False)
        .groupby("community")
        .head(k)
    )
    return test_ranked.merge(top_k_users[["community", "username"]], on=["community", "username"])


def _majority(votes):
    """Return the majority-vote prediction for each item."""
    return (
        votes.groupby(["community", "item_id"])
        .agg(
            predicted=("vote", lambda x: x.value_counts().idxmax()),
            score=("vote", "mean"),
            label=("label", "first"),
        )
        .reset_index()
    )


def _weighted_majority(votes):
    """Return reliability-weighted majority predictions."""

    def wm(g):
        """Return the weighted-majority prediction for one item."""
        weights = g["VAR_score"].to_numpy(dtype=float)
        score = (
            float(np.average(g["vote"], weights=weights))
            if weights.sum() > 0
            else float(g["vote"].mean())
        )
        return pd.Series(
            {
                "predicted": 1 if score > 0 else -1,
                "score": score,
                "label": g["label"].iloc[0],
            }
        )

    return votes.groupby(["community", "item_id"]).apply(wm).reset_index()


def _top1(votes):
    """Return the vote from the highest-ranked user."""
    decisions = (
        votes.sort_values("VAR_score", ascending=False)
        .groupby(["community", "item_id"])
        .first()
        .reset_index()[["community", "item_id", "vote", "label"]]
        .rename(columns={"vote": "predicted"})
    )
    decisions["score"] = decisions["predicted"].astype(float)
    return decisions


def _weighted_threshold(votes):
    """Apply a threshold to reliability-weighted votes."""
    majority_class = int(votes["vote"].mode().iloc[0])

    def wt(g):
        """Return the weighted-threshold prediction for one item."""
        w = g["VAR_score"].values
        score = float(np.average(g["vote"], weights=w)) if w.sum() else float(g["vote"].mean())
        return pd.Series(
            {
                "predicted": int(np.sign(score or majority_class)),
                "score": score,
                "label": g["label"].iloc[0],
            }
        )

    return votes.groupby(["community", "item_id"]).apply(wt).reset_index()


def _consensus(votes):
    """Return predictions only where selected voters agree."""
    majority_class = int(votes["vote"].mode().iloc[0])

    def cons(g):
        """Return the consensus prediction for one item."""
        u = g["vote"].unique()
        return pd.Series(
            {
                "predicted": int(u[0]) if len(u) == 1 else majority_class,
                "score": float(g["vote"].mean()),
                "label": g["label"].iloc[0],
            }
        )

    return votes.groupby(["community", "item_id"]).apply(cons).reset_index()


AGGREGATIONS = {
    "majority": _majority,
    "weighted": _weighted_majority,
}


# Evaluation


def kfold_eval(decisions, n_folds=N_FOLDS):
    """Evaluate one aggregation strategy with cross-validation."""
    labeled_items = decisions[["item_id", "label"]].drop_duplicates().reset_index(drop=True)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=10)
    fold_metrics = []
    for _, test_idx in kf.split(labeled_items):
        test_ids = labeled_items.iloc[test_idx]["item_id"].values
        fold_dec = decisions[decisions["item_id"].isin(test_ids)]
        if len(fold_dec) < 2 or fold_dec["label"].nunique() < 2:
            continue
        fold_metrics.append(evaluate_fold(fold_dec))
    agg = {"n_folds": len(fold_metrics)}
    for key in SCALAR_METRICS:
        vals = [m[key] for m in fold_metrics]
        agg[key] = float(np.mean(vals))
        agg[f"{key}_std"] = float(np.std(vals))
        agg[f"{key}_per_fold"] = vals
    return agg


def _base_votes_per_item(test_df, train_scores, k):
    """Count available votes for each item."""
    test_ranked = test_df.merge(
        train_scores[["community", "username", "VAR_score"]],
        on=["community", "username"],
        how="left",
    )
    test_ranked["VAR_score"] = test_ranked["VAR_score"].fillna(0)
    # Keep the top-K voters per item.
    test_ranked["rank"] = test_ranked.groupby(["community", "item_id"])["VAR_score"].rank(
        method="first", ascending=False
    )
    return test_ranked[test_ranked["rank"] <= k].drop(columns="rank")


def run_aggregation_comparison(test_df, train_scores):
    """Run the aggregation comparison workflow."""
    rows = []

    selection_modes = {
        "per_community": _base_votes,
        "per_item": _base_votes_per_item,
    }

    for agg_name, agg_fn in AGGREGATIONS.items():
        for mode_name, base_fn in selection_modes.items():
            label = f"{agg_name}_{mode_name}"
            best_agg, best_k, best_f1 = None, None, -1
            for k in K_VALUES:
                votes = base_fn(test_df, train_scores, k)
                decisions = agg_fn(votes)
                agg = kfold_eval(decisions)
                f1, std, auc, f1n = (
                    agg["macro_f1"],
                    agg["macro_f1_std"],
                    agg["roc_auc"],
                    agg["f1_neg"],
                )
                rows.append(
                    {
                        "strategy": label,
                        "k": k,
                        "macro_f1": f1,
                        "macro_f1_std": std,
                        "roc_auc": auc,
                        "f1_neg": f1n,
                        "f1_pos": agg["f1_pos"],
                    }
                )
                if f1 > best_f1:
                    best_f1, best_k, best_agg = f1, k, agg

            out_dir = OUTPUT_ROOT / f"VAR_{label}_k{best_k}"
            out_dir.mkdir(parents=True, exist_ok=True)
            with open(out_dir / "metrics.json", "w") as f:
                json.dump(best_agg, f, indent=2)
            print(f"{label}: k={best_k} | macro-F1={best_f1:.3f}")

    return pd.DataFrame(rows)


# Summary and plots


def print_summary(results_df):
    """Print summary to the console."""
    best = (
        results_df.sort_values("macro_f1", ascending=False)
        .groupby("strategy")
        .first()
        .reset_index()
        .sort_values("macro_f1", ascending=False)[
            ["strategy", "k", "macro_f1", "macro_f1_std", "roc_auc", "f1_neg", "f1_pos"]
        ]
        .round(3)
    )
    for row in best.itertuples(index=False):
        print(
            f"{row.strategy}: k={row.k} | macro-F1={row.macro_f1:.3f} | "
            f"AUC={row.roc_auc:.3f}"
        )


def plot_comparison(results_df):
    """Plot comparison and save the figure."""
    sns.set_theme(style="white", context="talk")
    palette = {
        "majority_per_community": "#2563eb",
        "majority_per_item": "#93c5fd",
        "weighted_per_community": "#16a34a",
        "weighted_per_item": "#86efac",
    }
    fig, ax = plt.subplots(figsize=(9, 6))
    for strategy, grp in results_df.groupby("strategy"):
        grp = grp.sort_values("k")
        ax.plot(
            grp["k"],
            grp["macro_f1"],
            label=strategy,
            color=palette.get(strategy, "gray"),
            linewidth=2.5,
            marker="o",
            markersize=4,
        )
        if "macro_f1_std" in grp.columns:
            ax.fill_between(
                grp["k"],
                grp["macro_f1"] - grp["macro_f1_std"],
                grp["macro_f1"] + grp["macro_f1_std"],
                alpha=0.12,
                color=palette.get(strategy, "gray"),
            )

    ymin = results_df["macro_f1"].min()
    ymax = results_df["macro_f1"].max()
    margin = (ymax - ymin) * 0.3
    ax.set_ylim(ymin - margin, ymax + margin)

    ax.set_title("Macro F1 by aggregation strategy", fontweight="bold")
    ax.set_xlabel("K")
    ax.set_ylabel("Macro F1")
    ax.legend(frameon=False)
    sns.despine()
    plt.tight_layout()
    plt.savefig("var_aggregation_comparison.png", dpi=150, bbox_inches="tight")
    plt.savefig("var_aggregation_comparison.pdf", dpi=300, bbox_inches="tight")
    plt.show()


def show_prediction_examples(test_df, train_scores, k=5, n_examples=5):
    """Display prediction examples for inspection."""
    votes = _base_votes(test_df, train_scores, k)
    decisions = _majority(votes)

    voter_details = votes[["item_id", "username", "vote", "VAR_score"]].sort_values(
        ["item_id", "VAR_score"], ascending=[True, False]
    )

    # Voter coverage
    voters_per_item = voter_details.groupby("item_id")["username"].count()
    coverage = voters_per_item.value_counts().sort_index()
    print(f"Voter coverage (k={k})")
    print(f"  {'n_voters':>8}  {'n_items':>8}  {'pct':>7}")
    total = len(voters_per_item)
    for n_voters, n_items in coverage.items():
        print(f"  {n_voters:>8}  {n_items:>8}  {100 * n_items / total:>6.1f}%")
    print(f"  {'total':>8}  {total:>8}")
    print(
        f"mean voters/item: {voters_per_item.mean():.2f} | "
        f"median={voters_per_item.median():.0f} | max={voters_per_item.max()}"
    )

    # Prediction examples
    correct = decisions[decisions["predicted"] == decisions["label"]]
    wrong = decisions[decisions["predicted"] != decisions["label"]]

    def print_examples(subset, title, n):
        """Print examples to the console."""
        print(f"{title} (k={k})")
        for _, row in subset.sample(min(n, len(subset)), random_state=10).iterrows():
            item = row["item_id"]
            label = int(row["label"])
            pred = int(row["predicted"])
            voters = voter_details[voter_details["item_id"] == item]
            print(f"item={item}")
            print(f"label={label:+d} | prediction={pred:+d} | voters={len(voters)}")
            print(f"  {'username':<20} {'vote':>5}  {'VAR_score':>10}")
            for _, v in voters.iterrows():
                print(f"  {v['username']:<20} {int(v['vote']):>+5}  {v['VAR_score']:>10.4f}")

    print_examples(correct, "CORRECT PREDICTIONS", n_examples)
    print_examples(wrong, "WRONG PREDICTIONS", n_examples)


# Canonical benchmark: TRAIN estimates VAR, VAL selects strategy and K, and
# TEST supplies final metrics. Detail exports support user-level diagnostics.

VOTES_DIR_DEFAULT = REDDIT.splits
VAR_MULTI_SPLIT_OUTPUT = OUTPUT_ROOT
MIN_CAL_FALLBACK_ITEMS = 50


def _fallback_val_from_train(split_label: str, train_df: pd.DataFrame):
    """Create val from train when validation data is unavailable."""
    train_items = train_df["item_id"].unique()
    if len(train_items) == 0:
        return None
    rng = np.random.RandomState(10)
    sample_size = min(len(train_items), max(MIN_CAL_FALLBACK_ITEMS, int(0.2 * len(train_items))))
    sample_ids = rng.choice(train_items, size=sample_size, replace=False)
    return train_df[train_df["item_id"].isin(sample_ids)]


def _complete_with_net_vote(
    decisions: pd.DataFrame,
    all_votes: pd.DataFrame,
    threshold: float,
) -> tuple[pd.DataFrame, dict]:
    """Restore uncovered items with validation-calibrated Net Vote."""
    n_total = int(all_votes["item_id"].nunique())
    n_covered = int(decisions["item_id"].nunique()) if not decisions.empty else 0
    completed = add_net_vote_fallback(
        decisions,
        all_votes,
        threshold,
        prediction_column="predicted",
        native_score_column="score",
        native_source="var_top_k",
    )
    return completed, {
        "n_expert_covered_items": n_covered,
        "expert_coverage_rate": float(n_covered / n_total) if n_total else 0.0,
        "n_fallback_items": int(n_total - n_covered),
        "used_net_vote_fallback": bool(n_total > n_covered),
        "net_vote_fallback_threshold": float(threshold),
    }


def run_single_split_var(
    split_label: str, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame
):
    """Run the single split var workflow."""
    if test_df.empty or test_df["item_id"].nunique() == 0:
        print(f"  [{split_label}] empty test set - skipped.")
        return None, None, None

    val_for_cal = val_df
    if val_df.empty or val_df["item_id"].nunique() == 0:
        print(f"  [{split_label}] empty val split - falling back to a sample carved from TRAIN.")
        val_for_cal = _fallback_val_from_train(split_label, train_df)
        if val_for_cal is None:
            print(f"  [{split_label}] no train items either - skipped.")
            return None, None, None

    scores = calculate_var_scores(train_df)
    net_vote_threshold = calibrate_net_vote_threshold(val_for_cal)

    selection_modes = {"per_community": _base_votes, "per_item": _base_votes_per_item}

    best_cfg, best_f1_val = None, -1.0
    for agg_name, agg_fn in AGGREGATIONS.items():
        for mode_name, base_fn in selection_modes.items():
            for k in K_VALUES:
                val_votes = base_fn(val_for_cal, scores, k)
                if val_votes.empty:
                    continue
                val_decisions, _ = _complete_with_net_vote(
                    agg_fn(val_votes), val_for_cal, net_vote_threshold
                )
                if val_decisions["label"].nunique() < 2:
                    continue
                val_metrics = evaluate_fold(val_decisions)
                if val_metrics["macro_f1"] > best_f1_val:
                    best_f1_val = val_metrics["macro_f1"]
                    best_cfg = {"strategy": agg_name, "mode": mode_name, "k": k}

    if best_cfg is None:
        print(f"  [{split_label}] no valid (strategy, K) found on VAL - skipped.")
        return None, None, None

    agg_fn = AGGREGATIONS[best_cfg["strategy"]]
    base_fn = selection_modes[best_cfg["mode"]]

    test_votes = base_fn(test_df, scores, best_cfg["k"])
    test_decisions, coverage = _complete_with_net_vote(
        agg_fn(test_votes), test_df, net_vote_threshold
    )

    if test_decisions["label"].nunique() < 2:
        print(f"  [{split_label}] single class in test - skipped.")
        return None, None, None
    test_metrics = evaluate_fold(
        test_decisions
    )  # single pass over the whole TEST set, no sub-bucketing
    native_decisions = test_decisions.loc[~test_decisions["is_fallback"]]
    native_metrics = (
        evaluate_fold(native_decisions)
        if not native_decisions.empty
        else {"macro_f1": float("nan"), "f1_pos": float("nan"),
              "f1_neg": float("nan"), "roc_auc": float("nan")}
    )

    test_metrics.update(
        {
            "split": split_label,
            "strategy": best_cfg["strategy"],
            "mode": best_cfg["mode"],
            "k": best_cfg["k"],
            "val_macro_f1": best_f1_val,
            "n_train_items": int(train_df["item_id"].nunique()),
            "n_val_items": int(val_df["item_id"].nunique()),
            "n_test_items": int(test_df["item_id"].nunique()),
            "n_items": int(len(test_decisions)),
            "score_coverage": coverage["expert_coverage_rate"],
            "final_decision_coverage": 1.0,
            "native_macro_f1": native_metrics["macro_f1"],
            "native_f1_pos": native_metrics["f1_pos"],
            "native_f1_neg": native_metrics["f1_neg"],
            "native_roc_auc": native_metrics["roc_auc"],
            "auc_population": "var_top_k_items",
            **coverage,
        }
    )

    print(
        f"[{split_label}] {best_cfg['strategy']}_{best_cfg['mode']} k={best_cfg['k']} | "
        f"val-F1={best_f1_val:.3f} | test-F1={test_metrics['macro_f1']:.3f} | "
        f"AUC={test_metrics['roc_auc']:.3f} | coverage={coverage['expert_coverage_rate']:.1%}"
    )

    # Retain detail frames for export without recomputing selections.
    test_metrics["_detail_test_votes"] = test_votes
    test_metrics["_detail_test_decisions"] = test_decisions

    return best_cfg, test_metrics, scores


def evaluate_splits(
    votes_dir: Path = VOTES_DIR_DEFAULT, output_dir: Path = VAR_MULTI_SPLIT_OUTPUT
) -> dict:
    """Evaluate splits and return its metrics."""
    splits = discover_splits(votes_dir)
    if not splits:
        print(
            f"No split directories found under {votes_dir} "
            f"(expected splits/, splits_full/, splits_intersection/, "
            f"windowed_folds_full/w*, windowed_folds_intersection/w*)"
        )
        return {}

    summary = {}

    for split_name, split_path in splits.items():
        train_df, val_df, test_df = load_vote_partitions(split_path)
        best_cfg, metrics, scores = run_single_split_var(split_name, train_df, val_df, test_df)
        if metrics is None:
            continue

        # Remove detail frames before serializing scalar metrics.
        test_votes = metrics.pop("_detail_test_votes")
        test_decisions = metrics.pop("_detail_test_decisions")

        split_out_dir = output_dir / split_name / "var"
        split_out_dir.mkdir(parents=True, exist_ok=True)
        with open(split_out_dir / "metrics.json", "w") as fh:
            json.dump(metrics, fh, indent=2)

        # Selected voter scores used by user-level diagnostics.
        voter_cols = [
            c
            for c in ["community", "item_id", "username", "vote", "VAR_score"]
            if c in test_votes.columns
        ]
        test_votes[voter_cols].to_parquet(split_out_dir / "voter_scores.parquet", index=False)

        # Final decision per test item.
        item_cols = [
            c
            for c in ["community", "item_id", "predicted", "score", "label", "decision_source"]
            if c in test_decisions.columns
        ]
        test_decisions[item_cols].to_parquet(split_out_dir / "item_scores.parquet", index=False)

        # Unfiltered user-community VAR scores.
        scores.to_parquet(split_out_dir / "all_user_var_scores.parquet", index=False)

        summary[split_name] = metrics

    return summary


def main() -> None:
    """Run the canonical multi-split VAR benchmark."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--votes-dir", type=Path, default=VOTES_DIR_DEFAULT)
    parser.add_argument("--output-dir", type=Path, default=VAR_MULTI_SPLIT_OUTPUT)
    args = parser.parse_args()
    evaluate_splits(args.votes_dir, args.output_dir)


if __name__ == "__main__":
    main()
