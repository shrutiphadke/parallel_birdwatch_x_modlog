"""NormVio + Skill Extraction training and evaluation pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import KFold
from tqdm import tqdm
from transformers import BertTokenizer

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.layout import REDDIT  # noqa: E402
from src.methods.nvse.normvio import (  # noqa: E402
    BERT_TYPE,
    CATEGORIES,
    DEFAULT_VIOLATION_SCORES,
    _load_split_votes,
    get_device,
    load_votes,
    resolve_bert_source,
    score_violations,
)
from src.methods.team_formation_v2.shared_features import (  # noqa: E402
    DEFAULT_CAUSAL_FEATURES,
    attach_causal_features,
    load_causal_features,
)
from src.utils.net_vote_fallback import (  # noqa: E402
    add_net_vote_fallback,
    calibrate_net_vote_threshold,
)
from src.utils.splits import discover_splits  # noqa: E402
from src.utils.threshold_search import best_threshold  # noqa: E402

THRESHOLD_GRID = np.linspace(-1.5, 1.5, 300)
MIN_VOTES_SKILL = 5


# User skill is moderator agreement by NormVio category. Canonical runs estimate
# it on TRAIN and apply it to VAL and TEST. ``estimate_user_skill`` retains the
# legacy whole-dataset CLI path.

CAUSAL_FEATURES_PATH = DEFAULT_CAUSAL_FEATURES
CAUSAL_PROFILE_COLUMNS = [
    "prior_n_posts",
    "prior_n_comments",
    "prior_n_distinct_subreddits",
    "prior_n_replies_made",
    "tenure_days_at_vote",
    "prior_n_posts_in_sub",
    "prior_n_comments_in_sub",
    "prior_n_months_active_in_sub",
    "prior_n_interaction_partners",
    "prior_total_interactions",
    "prior_n_interaction_partners_in_sub",
    "prior_total_interactions_in_sub",
]


def _load_causal_feature_skill(
    votes: pd.DataFrame,
    causal_features: pd.DataFrame | Path,
) -> pd.Series:
    """Build a train-only user-community prior from causal per-vote snapshots."""
    causal = (
        load_causal_features(causal_features)
        if isinstance(causal_features, Path)
        else causal_features
    )
    enriched = attach_causal_features(votes, causal)
    components = []
    for column in CAUSAL_PROFILE_COLUMNS:
        values = pd.to_numeric(enriched[column], errors="coerce")
        if column == "tenure_days_at_vote":
            signal = 1.0 - np.exp(-np.log1p(values.clip(lower=0)) / 7.0)
            signal = signal.fillna(0.5)
        else:
            signal = 1.0 - np.exp(-np.log1p(values.fillna(0).clip(lower=0)) / 5.0)
        components.append(signal.to_numpy(dtype=float))
    enriched["_causal_profile"] = np.column_stack(components).mean(axis=1) - 0.5
    result = enriched.groupby(["username", "community"])["_causal_profile"].mean()
    print(
        f"  Causal feature skill computed for {len(result):,} user-community profiles "
        f"(mean={result.mean():.4f}, std={result.std():.4f})"
    )
    return result


def compute_user_skill(
    votes: pd.DataFrame,
    scores: pd.DataFrame,
    min_votes: int,
    causal_features: pd.DataFrame | Path = CAUSAL_FEATURES_PATH,
    alpha: float = 0.5,
    verbose: bool = True,
) -> pd.DataFrame:
    """Compute user skill from the supplied data."""
    scores_cat = scores[["item_id", "top_violation_category"]]
    merged = votes.merge(scores_cat, on="item_id", how="inner")
    merged = merged.dropna(subset=["label", "top_violation_category"])
    if verbose:
        print(f"Skill input: {len(merged):,} labeled votes")

    merged["agrees"] = (merged["vote"] == merged["label"]).astype(float)

    skill_feat = (
        _load_causal_feature_skill(votes, causal_features)
        if alpha < 1.0
        else pd.Series(dtype=float)
    )
    use_meta = len(skill_feat) > 0 and alpha < 1.0

    records = []
    iterator = merged.groupby(["username", "community"])
    if verbose:
        iterator = tqdm(iterator, desc="User-community skills")
    for (username, community), grp in iterator:
        row = {"username": username, "community": community, "n_votes": len(grp)}

        feat = float(skill_feat.get((username, community), np.nan)) if use_meta else np.nan

        skill_vals = []
        for cat in CATEGORIES:
            cat_grp = grp[grp["top_violation_category"] == cat]

            if len(cat_grp) >= min_votes:
                s_mod = float(cat_grp["agrees"].mean()) - 0.5
            else:
                s_mod = np.nan

            if not np.isnan(s_mod) and not np.isnan(feat):
                s = alpha * s_mod + (1 - alpha) * feat
            elif not np.isnan(s_mod):
                s = s_mod  # Use moderator agreement when metadata is unavailable.
            elif not np.isnan(feat) and alpha < 1.0:
                s = feat  # Use metadata when moderator agreement is unavailable.
            else:
                s = np.nan

            row[f"skill_{cat}"] = s
            row[f"skill_mod_{cat}"] = s_mod
            if not np.isnan(s):
                skill_vals.append(s)

        row["skill_feat"] = feat
        row["skill_overall"] = float(np.nanmean(skill_vals)) if skill_vals else np.nan
        records.append(row)

    return pd.DataFrame(records)


def estimate_user_skill(
    votes_dir: Path,
    scores_path: Path,
    out_path: Path,
    min_votes: int,
    causal_features: pd.DataFrame | Path = CAUSAL_FEATURES_PATH,
    alpha: float = 0.5,
) -> pd.DataFrame:
    """Estimate user-community skill from training votes."""
    votes = load_votes(votes_dir)
    scores = pd.read_parquet(scores_path)

    out_df = compute_user_skill(votes, scores, min_votes, causal_features, alpha, verbose=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_path, index=False)

    n_valid = out_df["skill_overall"].notna().sum()
    mean_skill = out_df["skill_overall"].mean() if n_valid else float("nan")
    print(
        f"Skill profiles: {len(out_df):,} | valid={n_valid:,} | "
        f"mean={mean_skill:.4f} | saved={out_path}"
    )
    return out_df


# Legacy shuffled K-fold benchmark


def _compute_nvse_scores(
    votes: pd.DataFrame,
    scores: pd.DataFrame,
    skills: pd.DataFrame,
) -> pd.DataFrame:
    """Compute NVSE scores from the supplied data."""
    # Merge votes with skills once per category.
    votes_scores = votes.merge(
        scores[["item_id", "top_violation_category"]], on="item_id", how="left"
    )

    records = []
    if len(skills) > 0:
        for cat in CATEGORIES:
            subset = votes_scores[votes_scores["top_violation_category"] == cat].copy()
            if subset.empty:
                continue
            skill_col = f"skill_{cat}"
            subset = subset.merge(
                skills[["username", "community", skill_col]],
                on=["username", "community"],
                how="left",
            )
            records.append(
                subset[["item_id", "vote", skill_col, "community", "label"]].rename(
                    columns={skill_col: "weight"}
                )
            )

    # Preserve uncategorized items when the skill table is empty.
    no_cat = votes_scores[votes_scores["top_violation_category"].isna()].copy()
    no_cat["weight"] = np.nan
    records.append(no_cat[["item_id", "vote", "weight", "community", "label"]])

    if len(skills) == 0:
        rest = votes_scores[votes_scores["top_violation_category"].notna()].copy()
        rest["weight"] = np.nan
        records.append(rest[["item_id", "vote", "weight", "community", "label"]])

    all_votes = pd.concat(records, ignore_index=True)

    item_scores = []
    for item_id, grp in all_votes.groupby("item_id"):
        community = grp["community"].iloc[0]
        label = grp["label"].iloc[0]
        top_cat = (
            votes_scores.loc[votes_scores["item_id"] == item_id, "top_violation_category"].iloc[0]
            if len(votes_scores[votes_scores["item_id"] == item_id])
            else None
        )

        valid = grp.dropna(subset=["weight"])
        denom = valid["weight"].abs().sum()

        if len(valid) == 0 or denom == 0:
            # Items without a skilled voter remain uncovered.
            continue
        nvse_score = float((valid["weight"] * valid["vote"]).sum() / denom)
        item_scores.append(
            {
                "item_id": item_id,
                "community": community,
                "label": label,
                "nvse_score": nvse_score,
                "top_category": top_cat,
            }
        )

    return pd.DataFrame(item_scores)


def _calibrate(item_scores: pd.DataFrame, val_ids: np.ndarray) -> float:
    """Select a decision threshold on validation scores."""
    val = item_scores[item_scores["item_id"].isin(val_ids)].dropna(subset=["label"])
    best_thr, _ = best_threshold(
        val["label"].values, val["nvse_score"].values, THRESHOLD_GRID, default=0.0
    )
    return best_thr


def _evaluate(item_scores: pd.DataFrame, test_ids: np.ndarray, threshold: float) -> Dict:
    """Calculate metrics for scored items."""
    test = item_scores[item_scores["item_id"].isin(test_ids)].dropna(subset=["label"])
    y_true = test["label"].values
    y_pred = (
        test["y_hat"].to_numpy(dtype=int)
        if "y_hat" in test
        else np.where(test["nvse_score"].values >= threshold, 1, -1)
    )
    fallback = test.get("is_fallback", pd.Series(False, index=test.index)).astype(bool)
    native = test.loc[~fallback]
    y_sc = native["nvse_score"].to_numpy(dtype=float)

    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[1, -1], average="macro", zero_division=0
    )
    p1, r1, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[1], average="binary", zero_division=0
    )
    p_, r_, f_, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[-1], average="binary", pos_label=-1, zero_division=0
    )
    try:
        auc = float(roc_auc_score((native["label"].to_numpy() == 1).astype(int), y_sc))
    except ValueError:
        auc = float("nan")

    native_macro_f1 = float("nan")
    if not native.empty:
        native_pred = np.where(native["nvse_score"].to_numpy() >= threshold, 1, -1)
        _, _, native_macro_f1, _ = precision_recall_fscore_support(
            native["label"].to_numpy(),
            native_pred,
            labels=[1, -1],
            average="macro",
            zero_division=0,
        )

    return {
        "threshold": threshold,
        "macro_f1": float(f),
        "macro_precision": float(p),
        "macro_recall": float(r),
        "f1_pos": float(f1),  # approve class (label=+1)
        "f1_neg": float(f_),  # remove  class (label=-1)
        "f1_approve": float(f1),  # alias
        "f1_remove": float(f_),  # alias
        "precision_pos": float(p1),
        "recall_pos": float(r1),
        "precision_neg": float(p_),
        "recall_neg": float(r_),
        "roc_auc": auc,
        "n_items": len(test),
        "n_native_items": int((~fallback).sum()),
        "n_fallback_items": int(fallback.sum()),
        "score_coverage": float((~fallback).mean()) if len(test) else 0.0,
        "native_macro_f1": float(native_macro_f1),
        "used_net_vote_fallback": bool(fallback.any()),
        "auc_population": "native_nvse_items",
    }


def evaluate_kfold(
    votes_dir: Path, scores_path: Path, skills_path: Path, out_dir: Path, n_folds: int
) -> Dict:
    """Evaluate kfold and return its metrics."""
    votes = load_votes(votes_dir)
    scores = pd.read_parquet(scores_path)
    skills = pd.read_parquet(skills_path)

    item_scores = _compute_nvse_scores(votes, scores, skills)
    n_total = votes["item_id"].nunique()
    n_scored = len(item_scores)

    labeled = item_scores.dropna(subset=["label"]).reset_index(drop=True)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=10)

    fold_metrics: List[Dict] = []
    fold_frames: List[pd.DataFrame] = []

    for fold_idx, (val_idx, test_idx) in enumerate(kf.split(labeled)):
        val_ids = labeled.iloc[val_idx]["item_id"].values
        test_ids = labeled.iloc[test_idx]["item_id"].values

        threshold = _calibrate(item_scores, val_ids)
        metrics = _evaluate(item_scores, test_ids, threshold)
        metrics["fold"] = fold_idx
        fold_metrics.append(metrics)

        frame = item_scores[item_scores["item_id"].isin(test_ids)].copy()
        frame["prediction"] = np.where(frame["nvse_score"] >= threshold, 1, -1)
        frame["fold"] = fold_idx
        fold_frames.append(frame)

        print(
            f"NVSE fold {fold_idx + 1}/{n_folds}: threshold={threshold:.3f} | "
            f"macro-F1={metrics['macro_f1']:.4f} | AUC={metrics['roc_auc']:.4f}"
        )

    scalar_keys = [
        k for k, v in fold_metrics[0].items() if isinstance(v, (int, float)) and k != "fold"
    ]
    agg: Dict = {"n_folds": n_folds}
    for k in scalar_keys:
        vals = [m[k] for m in fold_metrics if not np.isnan(float(m[k]))]
        agg[k] = float(np.mean(vals))
        agg[f"{k}_std"] = float(np.std(vals))
        agg[f"{k}_per_fold"] = [m[k] for m in fold_metrics]

    out_dir.mkdir(parents=True, exist_ok=True)
    pd.concat(fold_frames, ignore_index=True).to_parquet(
        out_dir / "item_scores.parquet", index=False
    )
    with open(out_dir / "metrics.json", "w") as fh:
        json.dump(agg, fh, indent=2)
    pd.DataFrame(fold_metrics).to_parquet(out_dir / "fold_details.parquet", index=False)
    print(
        f"NVSE: macro-F1={agg['macro_f1']:.4f} | AUC={agg['roc_auc']:.4f} | "
        f"coverage={n_scored / max(n_total, 1):.1%} | saved={out_dir}"
    )
    return agg


# Canonical multi-split benchmark


def evaluate_splits(
    votes_dir: Path,
    scores_path: Path,
    out_dir: Path,
    min_votes: int = MIN_VOTES_SKILL,
    causal_features_path: Path = CAUSAL_FEATURES_PATH,
    alpha: float = 0.5,
) -> Dict[str, Dict]:
    """Evaluate splits and return its metrics."""
    scores = pd.read_parquet(scores_path)
    causal = load_causal_features(causal_features_path)
    splits = discover_splits(votes_dir)

    if not splits:
        print(f"No prepared splits found under {votes_dir}")
        return {}
    summary: Dict[str, Dict] = {}

    for split_name, split_path in splits.items():
        train_votes = _load_split_votes(split_path / "train_votes.parquet")
        val_votes = _load_split_votes(split_path / "val_votes.parquet")
        test_votes = _load_split_votes(split_path / "test_votes.parquet")

        skills = compute_user_skill(train_votes, scores, min_votes, causal, alpha, verbose=False)
        n_valid_skill = int(skills["skill_overall"].notna().sum()) if len(skills) else 0

        item_scores_val = _compute_nvse_scores(val_votes, scores, skills)
        item_scores_test = _compute_nvse_scores(test_votes, scores, skills)

        if item_scores_val.empty or item_scores_test.empty:
            print(
                f"Skipped {split_name}: no scoreable validation or test items "
                f"(val={len(item_scores_val)}, test={len(item_scores_test)})"
            )
            continue

        threshold = _calibrate(item_scores_val, item_scores_val["item_id"].values)

        n_native = len(item_scores_test)
        item_scores_test = item_scores_test.copy()
        item_scores_test["y_hat"] = np.where(
            item_scores_test["nvse_score"].to_numpy() >= threshold, 1, -1
        )
        net_vote_threshold = calibrate_net_vote_threshold(val_votes)
        item_scores_test = add_net_vote_fallback(
            item_scores_test,
            test_votes,
            net_vote_threshold,
            prediction_column="y_hat",
            native_score_column="nvse_score",
            native_source="nvse",
        )
        n_fallback = len(item_scores_test) - n_native

        metrics = _evaluate(item_scores_test, item_scores_test["item_id"].values, threshold)

        n_test_items = test_votes["item_id"].nunique()
        metrics["split"] = split_name
        metrics["n_train_items"] = train_votes["item_id"].nunique()
        metrics["n_val_items"] = val_votes["item_id"].nunique()
        metrics["n_test_items"] = n_test_items
        metrics["coverage_test"] = n_native / max(n_test_items, 1)
        metrics["final_decision_coverage"] = len(item_scores_test) / max(n_test_items, 1)
        metrics["net_vote_fallback_threshold"] = net_vote_threshold
        metrics["user_profile"] = "causal train-only activity/tenure/social metadata"
        metrics["collection_time_karma_used"] = False

        print(
            f"NVSE {split_name}: threshold={threshold:.3f} | "
            f"profiles={n_valid_skill:,} | macro-F1={metrics['macro_f1']:.4f} | "
            f"AUC={metrics['roc_auc']:.4f} | coverage={metrics['coverage_test']:.1%}"
            + (f" | fallback_items={n_fallback}" if n_fallback else "")
        )

        split_out_dir = out_dir / split_name / "normvio-skill-extraction"
        split_out_dir.mkdir(parents=True, exist_ok=True)
        skills.to_parquet(split_out_dir / "user_skill.parquet", index=False)
        item_scores_val.to_parquet(split_out_dir / "item_scores_val.parquet", index=False)
        item_scores_test.to_parquet(split_out_dir / "item_scores_test.parquet", index=False)
        with open(split_out_dir / "metrics.json", "w") as fh:
            json.dump(metrics, fh, indent=2)
        summary[split_name] = metrics

    return summary


# CLI


def parse_args() -> argparse.Namespace:
    """Parse args from the supplied input."""
    p = argparse.ArgumentParser(
        description="NormVio + Skill Extraction - full pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--votes_dir", default=str(REDDIT.splits))
    p.add_argument(
        "--post_texts",
        "--items",
        default=str(REDDIT.default_inputs().post_texts),
        help="Per-item table with title/selftext or text, used for post scoring.",
    )
    p.add_argument(
        "--models",
        default=str(REDDIT.normvio_models),
        help="Dir with one sub-folder per category (finetuned_model.pt)",
    )
    p.add_argument(
        "--bert_model",
        default=BERT_TYPE,
        help="Hugging Face model id or local BERT snapshot directory.",
    )
    p.add_argument(
        "--out_dir",
        default=str(REDDIT.nvse_work),
        help="Working directory for optional legacy skill/K-fold tasks",
    )
    p.add_argument(
        "--results_dir",
        default=str(REDDIT.results),
        help="Root for canonical per-split outputs (independent of out_dir work artifacts)",
    )
    p.add_argument(
        "--violation_scores",
        default=str(DEFAULT_VIOLATION_SCORES),
        help=(
            "Shared NormVio score cache keyed by item_id. It is reused by every "
            "split and is independent of out_dir."
        ),
    )
    p.add_argument(
        "--force-rescore",
        action="store_true",
        help="Recompute and overwrite the shared NormVio score cache.",
    )
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--device", default="", help="cuda | cpu (auto-detected if empty)")
    p.add_argument(
        "--n_folds",
        type=int,
        default=5,
        help="Number of folds for the legacy shuffled K-fold benchmark (step D)",
    )
    p.add_argument(
        "--min_votes",
        type=int,
        default=MIN_VOTES_SKILL,
        help="Min votes per userxcommunityxcategory to compute skill",
    )
    p.add_argument(
        "--causal_features",
        default=str(CAUSAL_FEATURES_PATH),
        help="Per-vote causal user feature parquet",
    )
    p.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Weight for moderator-agreement skill vs metadata skill (1.0=pure mod, 0.0=pure meta)",
    )

    tasks = ["score", "skill", "kfold", "splits"]
    p.add_argument(
        "--tasks", nargs="+", default=tasks, choices=tasks, help="Tasks to run (default: all)"
    )
    p.add_argument(
        "--max_posts",
        type=int,
        default=None,
        help="Limit number of posts scored in step B (for testing)",
    )
    p.add_argument(
        "--skip-tasks",
        nargs="+",
        default=[],
        choices=tasks,
        help="Tasks to skip when existing output files can be reused",
    )
    return p.parse_args()


def main():
    """Run the command-line workflow."""
    args = parse_args()
    device = get_device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks_to_run = [task for task in args.tasks if task not in args.skip_tasks]
    print(f"NVSE: device={device} | tasks={','.join(tasks_to_run)}")

    # Intermediate artifacts
    viol_scores_path = Path(args.violation_scores)
    user_skill_path = out_dir / "user_skill.parquet"
    per_split_dir = Path(args.results_dir)

    if "score" in tasks_to_run:
        if viol_scores_path.exists() and not args.force_rescore:
            print(f"Violation scores reused: {viol_scores_path}")
        else:
            configured_bert = Path(args.bert_model)
            bert_source = (
                configured_bert
                if configured_bert.exists()
                else resolve_bert_source(args.bert_model)
            )
            print(f"BERT tokenizer source: {bert_source}")
            tokenizer = BertTokenizer.from_pretrained(str(bert_source))
            score_violations(
                Path(args.votes_dir),
                Path(args.models),
                viol_scores_path,
                tokenizer,
                device,
                args.batch_size,
                max_posts=args.max_posts,
                post_texts_path=Path(args.post_texts),
                model_source=bert_source,
            )

    downstream_tasks = {"skill", "kfold", "splits"}.intersection(tasks_to_run)
    if downstream_tasks and not viol_scores_path.exists():
        raise FileNotFoundError(
            f"Shared violation scores not found at {viol_scores_path}. "
            "Run the 'score' task once or pass --violation_scores to an existing cache."
        )

    if "skill" in tasks_to_run:
        estimate_user_skill(
            Path(args.votes_dir),
            viol_scores_path,
            user_skill_path,
            args.min_votes,
            causal_features=Path(args.causal_features),
            alpha=args.alpha,
        )

    if "kfold" in tasks_to_run:
        evaluate_kfold(
            Path(args.votes_dir), viol_scores_path, user_skill_path, out_dir, args.n_folds
        )

    if "splits" in tasks_to_run:
        evaluate_splits(
            Path(args.votes_dir),
            viol_scores_path,
            per_split_dir,
            args.min_votes,
            Path(args.causal_features),
            args.alpha,
        )


if __name__ == "__main__":
    main()
