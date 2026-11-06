"""Shared causal voter representation, graph scoring, and calibration utilities."""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import f1_score, roc_auc_score

from src.layout import REDDIT  # noqa: E402
from src.utils.net_vote_fallback import (
    add_net_vote_fallback,
    calibrate_net_vote_threshold,
)
from src.utils.threshold_search import best_threshold as grid_best_threshold

DEFAULT_CAUSAL_FEATURES = REDDIT.causal_features

CAUSAL_GLOBAL_FEATURES = [
    "prior_n_posts",
    "prior_n_comments",
    "prior_n_distinct_subreddits",
    "prior_n_replies_made",
    "tenure_days_at_vote",
]
CAUSAL_LOCAL_FEATURES = [
    "prior_n_posts_in_sub",
    "prior_n_comments_in_sub",
    "prior_n_months_active_in_sub",
]
CAUSAL_SOCIAL_FEATURES = [
    "prior_n_interaction_partners",
    "prior_total_interactions",
    "prior_n_interaction_partners_in_sub",
    "prior_total_interactions_in_sub",
]
CAUSAL_FEATURES = CAUSAL_GLOBAL_FEATURES + CAUSAL_LOCAL_FEATURES + CAUSAL_SOCIAL_FEATURES


def user_key(values: pd.Series) -> pd.Series:
    """Return stable, case-insensitive Reddit user keys."""
    return values.map(lambda value: str(value).casefold() if pd.notna(value) else "")


def unix_seconds(values: pd.Series) -> pd.Series:
    """Normalize numeric or datetime-like timestamps to Unix seconds."""
    if pd.api.types.is_datetime64_any_dtype(values.dtype):
        return values.map(lambda value: value.timestamp() if pd.notna(value) else np.nan)
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().all():
        return numeric.astype("float64")
    parsed = pd.to_datetime(values, utc=True, errors="coerce")
    return parsed.map(lambda value: value.timestamp() if pd.notna(value) else np.nan)


def load_causal_features(path: Path = DEFAULT_CAUSAL_FEATURES) -> pd.DataFrame:
    """Load and validate one causal feature row per historical vote."""
    if not path.exists():
        raise FileNotFoundError(
            f"Causal features not found at {path}. Run "
            "`python -m src.data_preparation.build_user_history_features --stage join` first."
        )
    features = pd.read_parquet(path)
    if "timestamp_semantics" not in features:
        features["timestamp_semantics"] = "unknown_legacy_artifact"
    required = {"username", "item_id", "timestamp"} | set(CAUSAL_FEATURES)
    missing = required - set(features.columns)
    if missing:
        raise ValueError(f"Causal feature file is missing columns: {sorted(missing)}")
    features = features.copy()
    features["_user_key"] = user_key(features["username"])
    features["timestamp"] = unix_seconds(features["timestamp"])
    keys = ["_user_key", "item_id", "timestamp"]
    duplicates = int(features.duplicated(keys).sum())
    if duplicates:
        raise ValueError(f"Causal feature file contains {duplicates:,} duplicate vote keys")
    for column in (
        "global_available_at",
        "subreddit_available_at",
        "interaction_state_at",
        "local_interaction_state_at",
    ):
        if column not in features:
            continue
        available = pd.to_numeric(features[column], errors="coerce").dropna()
        if not available.empty and available.median() < 100_000_000:
            raise ValueError(
                f"{column} is not in Unix seconds; regenerate the causal feature file."
            )
    features["_causal_match"] = 1
    return features.drop(columns=["username"])


def attach_causal_features(
    votes: pd.DataFrame,
    causal: pd.DataFrame,
    min_coverage: float = 0.99,
    verbose: bool = True,
) -> pd.DataFrame:
    """Attach causal features to votes using the unique historical vote key."""
    enriched = votes.copy()
    enriched["_user_key"] = user_key(enriched["username"])
    enriched["timestamp"] = unix_seconds(enriched["timestamp"])
    before = len(enriched)
    causal_columns = [
        column
        for column in causal.columns
        if column
        not in {
            "community",
            "global_available_at",
            "subreddit_available_at",
            "interaction_state_at",
            "local_interaction_state_at",
        }
    ]
    enriched = enriched.merge(
        causal[causal_columns],
        on=["_user_key", "item_id", "timestamp"],
        how="left",
        validate="many_to_one",
    )
    if len(enriched) != before:
        raise AssertionError(
            f"Causal join changed row count: before={before:,}, after={len(enriched):,}"
        )
    coverage = float(enriched["_causal_match"].notna().mean())
    if verbose:
        print(f"Causal vote join coverage: {coverage:.2%} ({before:,} rows)")
    if coverage < min_coverage:
        raise ValueError(f"Causal vote join coverage is unexpectedly low: {coverage:.2%}")
    enriched[CAUSAL_FEATURES] = enriched[CAUSAL_FEATURES].apply(
        pd.to_numeric, errors="coerce"
    )
    # A missing historical count means no observed prior activity and is a
    # genuine zero. Missing account creation is different: it must not turn an
    # unknown-tenure user into a brand-new account.
    count_features = [column for column in CAUSAL_FEATURES if column != "tenure_days_at_vote"]
    enriched[count_features] = enriched[count_features].fillna(0.0)
    enriched["tenure_available"] = enriched["tenure_days_at_vote"].notna().astype(float)
    enriched["causal_feature_available"] = enriched.pop("_causal_match").fillna(0).astype(float)
    return enriched.drop(columns=["_user_key"])


def load_split_manifest(votes_dir: Path) -> dict:
    """Load a fixed-split manifest or the parent manifest of a training window."""
    fixed_manifest = votes_dir / "split_manifest.json"
    window_manifest = votes_dir.parent / "manifest.json"
    if fixed_manifest.exists():
        return json.loads(fixed_manifest.read_text(encoding="utf-8"))
    if window_manifest.exists():
        manifest = json.loads(window_manifest.read_text(encoding="utf-8"))
        known_windows = {
            str(entry.get("folder")) for entry in manifest.get("windows", [])
        }
        if (
            manifest.get("split_type") == "nested_training_windows"
            and votes_dir.name in known_windows
        ):
            return {**manifest, "selected_window": votes_dir.name}
    raise FileNotFoundError(f"No prepared split manifest found for {votes_dir}")


def load_and_enrich_splits(
    votes_dir: Path,
    causal_path: Path = DEFAULT_CAUSAL_FEATURES,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load any prepared split and attach the shared causal representation.

    Team-formation still creates its simulator/policy prefix chronologically inside
    TRAIN.  A chronological outer split additionally supports a prospective
    interpretation; seeded-item and window splits are valid comparative benchmarks
    but do not claim that outer TRAIN precedes VAL/TEST in wall-clock time.
    """
    manifest = load_split_manifest(votes_dir)
    if manifest.get("split_unit") not in (None, "item_id"):
        raise ValueError(
            "Team-formation-v2 requires item-disjoint prepared splits "
            f"(split_unit='item_id'); received {manifest.get('split_unit')!r}"
        )
    missing = [
        name
        for name in ("train", "val", "test")
        if not (votes_dir / f"{name}_votes.parquet").exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"Incomplete prepared split at {votes_dir}; missing partitions: {missing}"
        )
    causal = load_causal_features(causal_path)
    frames = []
    for name in ("train", "val", "test"):
        votes = pd.read_parquet(votes_dir / f"{name}_votes.parquet").sort_values("timestamp")
        frames.append(attach_causal_features(votes, causal, verbose=True))
    item_sets = [set(frame["item_id"].unique()) for frame in frames]
    if any(item_sets[left] & item_sets[right] for left, right in ((0, 1), (0, 2), (1, 2))):
        raise ValueError(f"Team-formation-v2 requires item-disjoint partitions: {votes_dir}")
    return tuple(frames)


def compute_alignment_tables(train: pd.DataFrame, prior_strength: float = 5.0) -> dict:
    """Fit smoothed moderator-alignment statistics from training votes only."""
    work = train.assign(_agree=(train["vote"] == train["label"]).astype(float))
    base = float(work["_agree"].mean()) if len(work) else 0.5

    def aggregate(columns: list[str]) -> pd.DataFrame:
        table = work.groupby(columns, as_index=False).agg(
            _align_sum=("_agree", "sum"), _align_n=("_agree", "size")
        )
        table["moderator_alignment"] = (
            table["_align_sum"] + prior_strength * base
        ) / (table["_align_n"] + prior_strength)
        return table

    return {
        "base": base,
        "prior_strength": prior_strength,
        "global": aggregate(["username"]),
        "local": aggregate(["username", "community"]),
    }


def attach_alignment(
    votes: pd.DataFrame,
    tables: dict,
    leave_one_out: bool = False,
) -> pd.DataFrame:
    """Attach training-only alignment, optionally excluding the current vote."""
    out = votes.copy()
    base = float(tables["base"])
    strength = float(tables["prior_strength"])
    out = out.merge(tables["global"], on="username", how="left")
    out = out.merge(
        tables["local"], on=["username", "community"], how="left", suffixes=("_global", "_local")
    )
    agree = (out["vote"] == out["label"]).astype(float)
    for scope in ("global", "local"):
        total = out[f"_align_sum_{scope}"].fillna(0.0)
        count = out[f"_align_n_{scope}"].fillna(0.0)
        if leave_one_out:
            total = (total - agree).clip(lower=0.0)
            count = (count - 1.0).clip(lower=0.0)
        out[f"moderator_alignment_{scope}"] = (total + strength * base) / (count + strength)
        out[f"moderator_alignment_{scope}_n"] = count
    return out.drop(
        columns=[
            "_align_sum_global", "_align_n_global",
            "_align_sum_local", "_align_n_local",
        ],
        errors="ignore",
    )


def prepare_voter_representation(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Add leakage-safe moderator alignment to already causal vote features."""
    tables = compute_alignment_tables(train)
    return (
        attach_alignment(train, tables, leave_one_out=True),
        attach_alignment(val, tables, leave_one_out=False),
        attach_alignment(test, tables, leave_one_out=False),
        tables,
    )


def compute_ppr_scores(
    train_votes: pd.DataFrame,
    damping: float = 0.85,
    max_iter: int = 100,
    tol: float = 1e-10,
) -> dict[str, float]:
    """Rank users on an agreement graph with tenure similarity and soft skill seeds."""
    users = sorted(train_votes["username"].dropna().unique())
    if not users:
        return {}
    user_index = {user: idx for idx, user in enumerate(users)}
    tenure_series = train_votes.groupby("username")["tenure_days_at_vote"].median()
    known_tenure = tenure_series.dropna()
    tenure_fallback = float(known_tenure.median()) if len(known_tenure) else 0.0
    tenure = tenure_series.to_dict()
    edge_counts: defaultdict[tuple[int, int], list[int]] = defaultdict(lambda: [0, 0])
    for _, item_votes in train_votes.groupby("item_id", sort=False):
        rows = item_votes[["username", "vote"]].drop_duplicates("username").itertuples(index=False)
        for left, right in combinations(rows, 2):
            i, j = user_index[left.username], user_index[right.username]
            key = (min(i, j), max(i, j))
            edge_counts[key][0] += int(left.vote == right.vote)
            edge_counts[key][1] += 1

    edge_weights: dict[tuple[int, int], float] = {}
    for (i, j), (n_agree, n_common) in edge_counts.items():
        # Beta(1,1) smoothing avoids extreme affinity from a single co-vote.
        agreement = (n_agree + 1.0) / (n_common + 2.0)
        left_raw = float(tenure.get(users[i], tenure_fallback))
        right_raw = float(tenure.get(users[j], tenure_fallback))
        left_tenure = np.log1p(max(left_raw if np.isfinite(left_raw) else tenure_fallback, 0.0))
        right_tenure = np.log1p(
            max(right_raw if np.isfinite(right_raw) else tenure_fallback, 0.0)
        )
        tenure_similarity = np.exp(-abs(left_tenure - right_tenure) / 2.0)
        weight = agreement * np.log1p(n_common) * (0.5 + 0.5 * tenure_similarity)
        if weight > 0:
            edge_weights[i, j] = weight
            edge_weights[j, i] = weight

    if edge_weights:
        row, col, values = zip(*((i, j, value) for (i, j), value in edge_weights.items()))
        adjacency = sparse.csr_matrix((values, (row, col)), shape=(len(users), len(users)))
        degree = np.asarray(adjacency.sum(axis=1)).ravel()
        inv_degree = np.divide(1.0, degree, out=np.zeros_like(degree), where=degree > 0)
        transition = sparse.diags(inv_degree) @ adjacency
    else:
        transition = sparse.csr_matrix((len(users), len(users)))

    if "moderator_alignment_global" in train_votes:
        alignment = train_votes.groupby("username")["moderator_alignment_global"].mean()
    else:
        alignment = train_votes.assign(
            _agree=train_votes["vote"] == train_votes["label"]
        ).groupby("username")["_agree"].mean()
    seeds = np.array([max(float(alignment.get(user, 0.5)) - 0.45, 0.01) for user in users])
    personalization = seeds / seeds.sum()
    scores = personalization.copy()
    dangling = np.asarray(transition.sum(axis=1)).ravel() == 0
    for _ in range(max_iter):
        updated = damping * (transition.T @ scores)
        updated += damping * scores[dangling].sum() * personalization
        updated += (1.0 - damping) * personalization
        if np.abs(updated - scores).sum() < tol:
            scores = updated
            break
        scores = updated
    return dict(zip(users, scores.astype(float)))


def calibrate_thresholds_per_community(
    validation_scores: pd.DataFrame,
    min_items: int = 20,
    grid_size: int = 201,
) -> tuple[dict[str, float], float]:
    """Select global and community thresholds by validation macro F1."""
    if validation_scores.empty:
        return {}, 0.0

    def best_threshold(frame: pd.DataFrame) -> float:
        values = frame["score"].to_numpy(dtype=float)
        labels = frame["label"].to_numpy()
        grid = np.linspace(values.min(), values.max(), grid_size) if len(values) else np.array([0.0])
        # Vectorized equivalent of max(grid, key=f1_score(..., average="macro")).
        return grid_best_threshold(labels, values, grid, default=0.0, fixed_labels=False)[0]

    global_threshold = best_threshold(validation_scores)
    thresholds = {
        community: best_threshold(group)
        for community, group in validation_scores.groupby("community")
        if len(group) >= min_items and group["label"].nunique() > 1
    }
    return thresholds, global_threshold


def apply_thresholds(
    scores: pd.DataFrame,
    community_thresholds: dict[str, float],
    global_threshold: float,
) -> pd.DataFrame:
    """Apply community thresholds with a global fallback."""
    predictions = scores.copy()
    thresholds = predictions["community"].map(community_thresholds).fillna(global_threshold)
    predictions["y_hat"] = np.where(predictions["score"] >= thresholds, 1, -1)
    predictions["threshold"] = thresholds
    return predictions


def classification_metrics(preds: pd.DataFrame) -> dict:
    """Calculate final hard-decision metrics and native-score diagnostics."""
    y_true, y_pred = preds["label"], preds["y_hat"]
    fallback = preds.get("is_fallback", pd.Series(False, index=preds.index)).astype(bool)
    native = preds.loc[~fallback]
    metrics = {
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "f1_pos": f1_score(y_true, y_pred, pos_label=1),
        "f1_neg": f1_score(y_true, y_pred, pos_label=-1),
        "n_items": len(preds),
        "n_native_items": int((~fallback).sum()),
        "n_fallback_items": int(fallback.sum()),
        "score_coverage": float((~fallback).mean()) if len(preds) else 0.0,
        "used_net_vote_fallback": bool(fallback.any()),
        "auc_population": "native_method_items",
    }
    if not native.empty:
        metrics["native_macro_f1"] = f1_score(
            native["label"], native["y_hat"], average="macro"
        )
        metrics["native_f1_pos"] = f1_score(
            native["label"], native["y_hat"], pos_label=1
        )
        metrics["native_f1_neg"] = f1_score(
            native["label"], native["y_hat"], pos_label=-1
        )
    try:
        metrics["roc_auc"] = roc_auc_score(native["label"], native["score"])
    except ValueError:
        metrics["roc_auc"] = None
    return metrics


def complete_with_net_vote_fallback(
    predictions: pd.DataFrame,
    validation_votes: pd.DataFrame,
    evaluation_votes: pd.DataFrame,
) -> tuple[pd.DataFrame, float]:
    """Complete a team method's missing decisions with the Net Vote baseline."""
    threshold = calibrate_net_vote_threshold(validation_votes)
    completed = add_net_vote_fallback(
        predictions,
        evaluation_votes,
        threshold,
        prediction_column="y_hat",
        native_score_column="score",
    )
    return completed, threshold
