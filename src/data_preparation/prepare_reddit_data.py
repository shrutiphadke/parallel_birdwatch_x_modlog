"""Build deterministic Reddit datasets and splits without network access."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.layout import ITEM_TEXT_COLUMNS, REDDIT, load_item_scores, load_items
from src.utils.tabular import read_table, unix_seconds

INPUT_PATH = REDDIT.default_inputs().votes
OUTPUT_DIR = REDDIT.datasets_dir
SPLITS_DIR = REDDIT.splits

MIN_VOTES_PER_POST = 5
MIN_VOTES_PER_USER = 10
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
# Shared seed for every deterministic item assignment and training window.
SEED = 10

_VAR_MIN_SUB_USERS = 40
_NVSE_MIN_SKILL_VOTES = 5
_SEF_MIN_USER_VOTES = 10
_WINDOW_SIZES = [0.2, 0.4, 0.6, 0.8, 1.0]
_WINDOW_TEST_FRAC = 0.20
_WINDOW_VAL_FRAC = 0.20
_K_FOLDS = 5
_CHRONOLOGICAL_INITIAL_TRAIN_FRAC = 0.40
_CORE_COLUMNS = ["username", "community", "item_id", "timestamp", "vote", "label"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def _sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_dataset(path: Path = INPUT_PATH) -> pd.DataFrame:
    """Load, validate, clean, and deterministically deduplicate a vote table."""
    log.info("Loading dataset from %s", path)
    df = read_table(path)
    missing = set(_CORE_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Input dataset is missing columns: {sorted(missing)}")

    df["vote"] = pd.to_numeric(df["vote"], errors="coerce")
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df[df["vote"].isin([1, -1]) & df["label"].isin([1, -1])].copy()
    df["vote"] = df["vote"].astype("int8")
    df["label"] = df["label"].astype("int8")
    df["timestamp"] = unix_seconds(df["timestamp"]).astype("float64")
    df[["username", "community", "item_id"]] = df[["username", "community", "item_id"]].astype(
        "string"
    )

    before = len(df)
    df = df.dropna(subset=_CORE_COLUMNS)
    log.info("Dropped %d rows with invalid core values", before - len(df))
    inconsistent_labels = df.groupby("item_id")["label"].nunique()
    if (inconsistent_labels > 1).any():
        raise ValueError(f"{int((inconsistent_labels > 1).sum())} item(s) have conflicting labels")
    inconsistent_communities = df.groupby("item_id")["community"].nunique()
    if (inconsistent_communities > 1).any():
        raise ValueError(
            f"{int((inconsistent_communities > 1).sum())} item(s) belong to multiple communities"
        )
    before = len(df)
    df = df.drop_duplicates(subset=["username", "item_id"], keep="first")
    log.info("Dropped %d duplicate (username, item_id) pairs", before - len(df))
    df = df.reset_index(drop=True)
    log.info(
        "Clean dataset: %d votes | %d users | %d posts | %d communities",
        len(df),
        df["username"].nunique(),
        df["item_id"].nunique(),
        df["community"].nunique(),
    )
    return df


def apply_density_filter(
    df: pd.DataFrame,
    min_votes_per_post: int = MIN_VOTES_PER_POST,
    min_votes_per_user: int = MIN_VOTES_PER_USER,
) -> pd.DataFrame:
    """Iteratively prune sparse items and users until the matrix converges."""
    filtered = df.copy()
    for iteration in range(1, 101):
        previous = len(filtered)
        valid_items = filtered["item_id"].value_counts()
        filtered = filtered[
            filtered["item_id"].isin(valid_items[valid_items >= min_votes_per_post].index)
        ]
        valid_users = filtered["username"].value_counts()
        filtered = filtered[
            filtered["username"].isin(valid_users[valid_users >= min_votes_per_user].index)
        ]
        if len(filtered) == previous:
            log.info("Density filter converged after %d iteration(s)", iteration)
            break
    else:
        raise RuntimeError("Density filter did not converge within 100 iterations")

    log.info(
        "Density-filtered dataset: %d votes | %d users | %d posts | %d communities",
        len(filtered),
        filtered["username"].nunique(),
        filtered["item_id"].nunique(),
        filtered["community"].nunique(),
    )
    return filtered.reset_index(drop=True)


def _iterative_density_filter(
    data: pd.DataFrame,
    min_post_votes: int,
    min_user_votes: int,
) -> Tuple[set, set]:
    filtered = apply_density_filter(data, min_post_votes, min_user_votes)
    return set(filtered["item_id"].unique()), set(filtered["username"].unique())


def identify_method_filters(
    df: pd.DataFrame,
    external_scores_path: Optional[Path] = None,
    post_texts_path: Optional[Path] = None,
) -> Dict[str, Dict]:
    """Identify the item/user population supported by every final method."""
    all_items = set(df["item_id"].unique())
    all_users = set(df["username"].unique())
    report: Dict[str, Dict] = {}

    items_bl, users_bl = _iterative_density_filter(df, MIN_VOTES_PER_POST, MIN_VOTES_PER_USER)
    report["BL_CN"] = {
        "valid_item_ids": items_bl,
        "valid_usernames": users_bl,
        "n_items_kept": len(items_bl),
        "n_items_dropped": len(all_items) - len(items_bl),
        "n_users_kept": len(users_bl),
        "n_users_dropped": len(all_users) - len(users_bl),
        "reason": (
            f"Iterative density filter: post needs >={MIN_VOTES_PER_POST} votes, "
            f"user needs >={MIN_VOTES_PER_USER} votes."
        ),
    }

    if external_scores_path is not None and external_scores_path.exists():
        ext = load_item_scores(external_scores_path)
        items_ext = items_bl & set(ext["item_id"].dropna().unique())
        missing_ext = len(items_bl) - len(items_ext)
    else:
        items_ext = items_bl
        missing_ext = 0
        log.warning(
            "External scores not found at %s; BL4/BL5 approximated as BL/CN",
            external_scores_path,
        )
    report["BL4_BL5"] = {
        "valid_item_ids": items_ext,
        "valid_usernames": users_bl,
        "n_items_kept": len(items_ext),
        "n_items_dropped": len(all_items) - len(items_ext),
        "n_users_kept": len(users_bl),
        "n_users_dropped": len(all_users) - len(users_bl),
        "reason": (
            "BL/CN density filter plus presence in moderated_posts_scores.parquet; "
            f"{missing_ext} posts lack an external-score row."
        ),
    }

    votes_per_user_community = (
        df.groupby(["username", "community"])["item_id"].count().reset_index(name="n_votes")
    )
    users_with_skill = set(
        votes_per_user_community.loc[
            votes_per_user_community["n_votes"] >= _NVSE_MIN_SKILL_VOTES, "username"
        ].unique()
    )
    report["NVSE"] = {
        "valid_item_ids": all_items,
        "valid_usernames": users_with_skill,
        "n_items_kept": len(all_items),
        "n_items_dropped": 0,
        "n_users_kept": len(users_with_skill),
        "n_users_dropped": len(all_users) - len(users_with_skill),
        "reason": (
            f"No item filter; a user needs >={_NVSE_MIN_SKILL_VOTES} votes in at "
            "least one community to obtain a skill entry."
        ),
    }

    if post_texts_path is not None and post_texts_path.exists():
        post_texts = load_items(post_texts_path, required=ITEM_TEXT_COLUMNS)
        items_sef = set(post_texts.loc[post_texts["text"].notna(), "item_id"].unique())
    else:
        items_sef = all_items
        log.warning("Post texts not found at %s; SEF item filter not applied", post_texts_path)
    user_counts = df["username"].value_counts()
    users_sef = set(user_counts[user_counts >= _SEF_MIN_USER_VOTES].index)
    report["SEF"] = {
        "valid_item_ids": items_sef,
        "valid_usernames": users_sef,
        "n_items_kept": len(items_sef),
        "n_items_dropped": len(all_items) - len(items_sef),
        "n_users_kept": len(users_sef),
        "n_users_dropped": len(all_users) - len(users_sef),
        "reason": (
            "Post requires non-null text in post_texts.parquet; user requires "
            f">={_SEF_MIN_USER_VOTES} votes globally."
        ),
    }

    items_var_tmp, users_var_tmp = _iterative_density_filter(df, MIN_VOTES_PER_POST, 5)
    var_df = df[df["item_id"].isin(items_var_tmp) & df["username"].isin(users_var_tmp)]
    valid_communities = var_df.groupby("community")["username"].nunique()
    valid_communities = valid_communities[valid_communities >= _VAR_MIN_SUB_USERS].index
    var_df = var_df[var_df["community"].isin(valid_communities)]
    items_var = set(var_df["item_id"].unique())
    users_var = set(var_df["username"].unique())
    report["VAR"] = {
        "valid_item_ids": items_var,
        "valid_usernames": users_var,
        "n_items_kept": len(items_var),
        "n_items_dropped": len(all_items) - len(items_var),
        "n_users_kept": len(users_var),
        "n_users_dropped": len(all_users) - len(users_var),
        "reason": (
            "Iterative 5-vote item/user density filter followed by community "
            f">={_VAR_MIN_SUB_USERS} unique users."
        ),
    }
    return report


def build_intersection_dataset(
    df: pd.DataFrame,
    filter_report: Dict[str, Dict],
) -> pd.DataFrame:
    """Intersect every method population, then reapply the main density rule."""
    item_sets = [entry["valid_item_ids"] for entry in filter_report.values()]
    user_sets = [entry["valid_usernames"] for entry in filter_report.values()]
    if not item_sets or not user_sets:
        raise ValueError("At least one method filter is required")
    valid_items = set.intersection(*item_sets)
    valid_users = set.intersection(*user_sets)
    intersection = df[df["item_id"].isin(valid_items) & df["username"].isin(valid_users)].copy()
    return apply_density_filter(intersection, MIN_VOTES_PER_POST, MIN_VOTES_PER_USER)


def _validate_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("train/validation/test ratios must sum to 1")
    if min(train_ratio, val_ratio, test_ratio) <= 0:
        raise ValueError("train/validation/test ratios must all be positive")


def build_seeded_item_split(
    df: pd.DataFrame,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
    seed: int = SEED,
) -> tuple[dict[str, pd.DataFrame], dict]:
    """Build a reproducible item-level train/validation/test split."""
    _validate_ratios(train_ratio, val_ratio, test_ratio)
    item_ids = df["item_id"].drop_duplicates().tolist()
    rng = np.random.RandomState(seed)
    rng.shuffle(item_ids)
    train_end = int(len(item_ids) * train_ratio)
    val_end = train_end + int(len(item_ids) * val_ratio)
    assignments = {
        "train": set(item_ids[:train_end]),
        "val": set(item_ids[train_end:val_end]),
        "test": set(item_ids[val_end:]),
    }
    splits = {
        name: df[df["item_id"].isin(ids)].reset_index(drop=True)
        for name, ids in assignments.items()
    }
    _validate_partition(df, splits)
    manifest = _split_manifest(df, splits, "seeded_item", seed=seed)
    return splits, manifest


def _boundary_after_ties(item_times: pd.DataFrame, target: int, lower: int = 0) -> int:
    """Place a boundary after a completion-time tie near the target item count."""
    n_items = len(item_times)
    target = min(max(target, lower + 1), n_items - 1)
    timestamp = item_times.iloc[target - 1]["item_time"]
    boundary = int(item_times["item_time"].searchsorted(timestamp, side="right"))
    if boundary >= n_items:
        boundary = int(item_times["item_time"].searchsorted(timestamp, side="left"))
    if boundary <= lower or boundary >= n_items:
        raise ValueError("Cannot create non-empty chronological partitions without splitting ties")
    return boundary


def build_chronological_split(
    votes: pd.DataFrame,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
) -> tuple[dict[str, pd.DataFrame], dict]:
    """Split items by their final vote timestamp while keeping boundary ties intact."""
    _validate_ratios(train_ratio, val_ratio, test_ratio)
    votes = votes.copy()
    votes["timestamp"] = unix_seconds(votes["timestamp"])
    if votes["timestamp"].isna().any():
        raise ValueError(f"Input contains {votes['timestamp'].isna().sum():,} invalid timestamps")

    item_times = (
        votes.groupby("item_id", as_index=False)
        .agg(item_first_time=("timestamp", "min"), item_time=("timestamp", "max"))
        .sort_values(["item_time", "item_id"], kind="stable")
        .reset_index(drop=True)
    )
    if len(item_times) < 3:
        raise ValueError("At least three items are required")
    train_end = _boundary_after_ties(item_times, int(len(item_times) * train_ratio))
    val_end = _boundary_after_ties(
        item_times,
        int(len(item_times) * (train_ratio + val_ratio)),
        lower=train_end,
    )
    assignments = {
        "train": set(item_times.iloc[:train_end]["item_id"]),
        "val": set(item_times.iloc[train_end:val_end]["item_id"]),
        "test": set(item_times.iloc[val_end:]["item_id"]),
    }
    splits = {
        name: votes[votes["item_id"].isin(ids)]
        .sort_values(["timestamp", "item_id", "username"], kind="stable")
        .reset_index(drop=True)
        for name, ids in assignments.items()
    }
    _validate_partition(votes, splits)
    train_max = item_times[item_times["item_id"].isin(assignments["train"])]["item_time"].max()
    val_times = item_times[item_times["item_id"].isin(assignments["val"])]["item_time"]
    test_min = item_times[item_times["item_id"].isin(assignments["test"])]["item_time"].min()
    if not (train_max < val_times.min() and val_times.max() < test_min):
        raise AssertionError("Chronological partitions are not strictly ordered")

    manifest = _split_manifest(votes, splits, "chronological")
    manifest.update(
        {
            "item_time_definition": "maximum vote timestamp per item (case completion)",
            "tie_policy": "equal boundary timestamps stay in the earlier partition",
            "checks": {
                **manifest["checks"],
                "strict_temporal_order": True,
            },
        }
    )
    return splits, manifest


def _validate_partition(source: pd.DataFrame, splits: dict[str, pd.DataFrame]) -> None:
    item_sets = {name: set(frame["item_id"].unique()) for name, frame in splits.items()}
    if any(
        item_sets[left] & item_sets[right]
        for left, right in (("train", "val"), ("train", "test"), ("val", "test"))
    ):
        raise AssertionError("An item appears in more than one partition")
    if set().union(*item_sets.values()) != set(source["item_id"].unique()):
        raise AssertionError("Not all source items were preserved")
    if sum(len(frame) for frame in splits.values()) != len(source):
        raise AssertionError("Not all source votes were preserved")


def _split_manifest(
    source: pd.DataFrame,
    splits: dict[str, pd.DataFrame],
    split_type: str,
    seed: Optional[int] = None,
) -> dict:
    train_users = set(splits["train"]["username"])
    manifest: dict[str, Any] = {
        "split_type": split_type,
        "split_unit": "item_id",
        "seed": seed,
        "n_source_votes": len(source),
        "n_source_items": int(source["item_id"].nunique()),
        "n_source_users": int(source["username"].nunique()),
        "partitions": {},
        "checks": {
            "item_overlap": 0,
            "all_source_items_preserved": True,
            "all_source_votes_preserved": True,
        },
    }
    for name, frame in splits.items():
        users = set(frame["username"])
        manifest["partitions"][name] = {
            "n_votes": len(frame),
            "n_items": int(frame["item_id"].nunique()),
            "n_users": int(frame["username"].nunique()),
            "item_fraction": float(
                frame["item_id"].nunique() / max(source["item_id"].nunique(), 1)
            ),
            "vote_time_start": pd.to_datetime(
                frame["timestamp"].min(), unit="s", utc=True
            ).isoformat(),
            "vote_time_end": pd.to_datetime(
                frame["timestamp"].max(), unit="s", utc=True
            ).isoformat(),
            "users_seen_in_train_fraction": (
                1.0 if name == "train" else float(len(users & train_users) / max(len(users), 1))
            ),
        }
    return manifest


def _write_split(splits: dict[str, pd.DataFrame], manifest: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in splits.items():
        path = output_dir / f"{name}_votes.parquet"
        frame.to_parquet(path, index=False)
    (output_dir / "split_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


def _item_table(votes: pd.DataFrame) -> pd.DataFrame:
    """Return one deterministic row per item and reject inconsistent labels."""
    label_counts = votes.groupby("item_id")["label"].nunique(dropna=False)
    inconsistent = label_counts[label_counts != 1]
    if not inconsistent.empty:
        raise ValueError(f"{len(inconsistent):,} items have inconsistent labels")
    return (
        votes.groupby("item_id", as_index=False)
        .agg(
            label=("label", "first"),
            item_first_time=("timestamp", "min"),
            item_time=("timestamp", "max"),
        )
        .sort_values("item_id", kind="stable")
        .reset_index(drop=True)
    )


def _fold_manifest(
    population_votes: pd.DataFrame,
    splits: dict[str, pd.DataFrame],
    *,
    dataset: str,
    fold_index: int,
    n_folds: int,
    protocol: str,
    seed: int | None,
) -> dict[str, Any]:
    """Build the common manifest fields for one materialized K-fold split."""
    fold_source = pd.concat(splits.values(), ignore_index=True)
    manifest = _split_manifest(fold_source, splits, f"{protocol}_kfold", seed=seed)
    manifest.update(
        {
            "dataset": dataset,
            "source_split": dataset,
            "cv_protocol": protocol,
            "fold_index": fold_index,
            "fold_name": f"fold_{fold_index:02d}",
            "n_folds": n_folds,
            "n_population_votes": int(len(population_votes)),
            "n_population_items": int(population_votes["item_id"].nunique()),
            "n_unused_future_votes": int(len(population_votes) - len(fold_source)),
            "n_unused_future_items": int(
                population_votes["item_id"].nunique() - fold_source["item_id"].nunique()
            ),
        }
    )
    return manifest


def build_stratified_kfold_splits(
    votes: pd.DataFrame,
    dataset: str,
    n_folds: int = _K_FOLDS,
    seed: int = SEED,
    item_fold_assignment: dict[str, int] | None = None,
) -> tuple[list[tuple[dict[str, pd.DataFrame], dict]], dict[str, int]]:
    """Build item-level random folds with one rotating VAL and TEST block."""
    if n_folds < 3:
        raise ValueError("Random K-fold requires at least three folds for TRAIN/VAL/TEST")
    items = _item_table(votes)
    if item_fold_assignment is None:
        minimum_class = int(items["label"].value_counts().min())
        if minimum_class < n_folds:
            raise ValueError(
                f"The smallest label class has {minimum_class} items; cannot build {n_folds} "
                "stratified folds"
            )
        assignment: dict[str, int] = {}
        splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        for fold_index, (_, test_indices) in enumerate(splitter.split(items, items["label"])):
            assignment.update(
                {str(item_id): fold_index for item_id in items.iloc[test_indices]["item_id"]}
            )
    else:
        missing = set(items["item_id"].astype(str)).difference(item_fold_assignment)
        if missing:
            raise ValueError(f"Shared fold assignment is missing {len(missing):,} {dataset} items")
        assignment = {
            str(item_id): int(item_fold_assignment[str(item_id)]) for item_id in items["item_id"]
        }

    item_folds = items["item_id"].astype(str).map(assignment)
    outputs: list[tuple[dict[str, pd.DataFrame], dict]] = []
    for fold_index in range(n_folds):
        test_ids = set(items.loc[item_folds == fold_index, "item_id"])
        val_ids = set(items.loc[item_folds == (fold_index + 1) % n_folds, "item_id"])
        train_ids = set(items["item_id"]) - test_ids - val_ids
        splits = {
            name: votes[votes["item_id"].isin(ids)]
            .sort_values(["item_id", "timestamp", "username"], kind="stable")
            .reset_index(drop=True)
            for name, ids in (("train", train_ids), ("val", val_ids), ("test", test_ids))
        }
        _validate_partition(votes, splits)
        manifest = _fold_manifest(
            votes,
            splits,
            dataset=dataset,
            fold_index=fold_index,
            n_folds=n_folds,
            protocol="stratified_item",
            seed=seed,
        )
        manifest["validation_fold_index"] = (fold_index + 1) % n_folds
        manifest["checks"]["each_item_is_test_once"] = True
        manifest["checks"]["each_item_is_validation_once"] = True
        outputs.append((splits, manifest))
    return outputs, assignment


def build_chronological_kfold_splits(
    votes: pd.DataFrame,
    dataset: str = "intersection_chronological",
    n_folds: int = _K_FOLDS,
    initial_train_fraction: float = _CHRONOLOGICAL_INITIAL_TRAIN_FRAC,
) -> list[tuple[dict[str, pd.DataFrame], dict]]:
    """Build expanding-window folds with adjacent chronological VAL and TEST blocks."""
    if n_folds < 1:
        raise ValueError("Chronological K-fold requires at least one fold")
    if not 0.0 < initial_train_fraction < 1.0:
        raise ValueError("initial_train_fraction must be in (0, 1)")

    votes = votes.copy()
    votes["timestamp"] = unix_seconds(votes["timestamp"])
    if votes["timestamp"].isna().any():
        raise ValueError("Chronological K-fold input contains invalid timestamps")
    items = _item_table(votes).sort_values(["item_time", "item_id"], kind="stable")
    items = items.reset_index(drop=True)
    n_items = len(items)
    if n_items < n_folds + 2:
        raise ValueError(f"At least {n_folds + 2} items are required")

    # The population consists of an initial TRAIN prefix followed by K+1
    # approximately equal temporal blocks.  For fold i, block i is VAL and
    # block i+1 is TEST; every later item remains genuinely unavailable.
    remainder = n_items * (1.0 - initial_train_fraction)
    targets = [
        int(n_items * initial_train_fraction + remainder * step / (n_folds + 1))
        for step in range(n_folds + 1)
    ]
    boundaries: list[int] = []
    lower = 0
    for target in targets:
        boundary = _boundary_after_ties(items, target, lower=lower)
        boundaries.append(boundary)
        lower = boundary
    boundaries.append(n_items)

    outputs: list[tuple[dict[str, pd.DataFrame], dict]] = []
    for fold_index in range(n_folds):
        train_end, val_end, test_end = boundaries[fold_index : fold_index + 3]
        ids = {
            "train": set(items.iloc[:train_end]["item_id"]),
            "val": set(items.iloc[train_end:val_end]["item_id"]),
            "test": set(items.iloc[val_end:test_end]["item_id"]),
        }
        splits = {
            name: votes[votes["item_id"].isin(partition_ids)]
            .sort_values(["timestamp", "item_id", "username"], kind="stable")
            .reset_index(drop=True)
            for name, partition_ids in ids.items()
        }
        fold_source = votes[votes["item_id"].isin(set().union(*ids.values()))]
        _validate_partition(fold_source, splits)
        train_max = items[items["item_id"].isin(ids["train"])]["item_time"].max()
        val_times = items[items["item_id"].isin(ids["val"])]["item_time"]
        test_times = items[items["item_id"].isin(ids["test"])]["item_time"]
        if not (train_max < val_times.min() and val_times.max() < test_times.min()):
            raise AssertionError(f"Chronological fold {fold_index} is not strictly ordered")

        manifest = _fold_manifest(
            votes,
            splits,
            dataset=dataset,
            fold_index=fold_index,
            n_folds=n_folds,
            protocol="expanding_window_chronological",
            seed=None,
        )
        manifest.update(
            {
                "item_time_definition": "maximum vote timestamp per item (case completion)",
                "tie_policy": "equal boundary timestamps stay in the earlier partition",
                "initial_train_fraction_target": initial_train_fraction,
                "future_data_policy": "items after TEST are excluded from the fold",
                "checks": {
                    **manifest["checks"],
                    "strict_temporal_order": True,
                },
            }
        )
        outputs.append((splits, manifest))
    return outputs


def write_kfold_splits(
    full_votes: pd.DataFrame,
    intersection_votes: pd.DataFrame,
    output_root: Path,
    n_folds: int = _K_FOLDS,
    seed: int = SEED,
    chronological_initial_train_fraction: float = _CHRONOLOGICAL_INITIAL_TRAIN_FRAC,
) -> dict[str, Any]:
    """Materialize the three K-fold collections under one compact root."""
    output_root.mkdir(parents=True, exist_ok=True)
    random_full, shared_assignment = build_stratified_kfold_splits(
        full_votes, "full", n_folds=n_folds, seed=seed
    )
    random_intersection, _ = build_stratified_kfold_splits(
        intersection_votes,
        "intersection",
        n_folds=n_folds,
        seed=seed,
        item_fold_assignment=shared_assignment,
    )
    chronological = build_chronological_kfold_splits(
        intersection_votes,
        n_folds=n_folds,
        initial_train_fraction=chronological_initial_train_fraction,
    )
    collections = {
        "full": ("stratified_item", random_full),
        "intersection": ("stratified_item", random_intersection),
        "intersection_chronological": ("expanding_window_chronological", chronological),
    }
    populations = {
        "full": full_votes,
        "intersection": intersection_votes,
        "intersection_chronological": intersection_votes,
    }
    root_manifest: dict[str, Any] = {
        "schema_version": 1,
        "split_type": "kfold_root",
        "n_folds": n_folds,
        "seed": seed,
        "datasets": [],
    }
    for dataset, (protocol, folds) in collections.items():
        dataset_root = output_root / dataset
        test_sets = [set(splits["test"]["item_id"]) for splits, _ in folds]
        test_items = set().union(*test_sets)
        test_overlap = sum(
            len(test_sets[left] & test_sets[right])
            for left in range(len(test_sets))
            for right in range(left + 1, len(test_sets))
        )
        dataset_manifest = {
            "schema_version": 1,
            "split_type": "kfold_collection",
            "dataset": dataset,
            "protocol": protocol,
            "n_folds": n_folds,
            "seed": seed if protocol == "stratified_item" else None,
            "n_population_votes": len(populations[dataset]),
            "n_population_items": int(populations[dataset]["item_id"].nunique()),
            "n_unique_test_items": len(test_items),
            "test_population_fraction": float(
                len(test_items) / max(populations[dataset]["item_id"].nunique(), 1)
            ),
            "test_items_exhaust_population": len(test_items)
            == populations[dataset]["item_id"].nunique(),
            "checks": {
                "test_item_overlap_across_folds": test_overlap,
                "test_coverage_matches_protocol": (
                    len(test_items) == populations[dataset]["item_id"].nunique()
                    if protocol == "stratified_item"
                    else 0 < len(test_items) < populations[dataset]["item_id"].nunique()
                ),
            },
            "folds": [],
        }
        if dataset == "intersection":
            dataset_manifest["shared_item_fold_assignment"] = "full"
        if dataset == "intersection_chronological":
            dataset_manifest["initial_train_fraction_target"] = (
                chronological_initial_train_fraction
            )
        for fold_index, (splits, manifest) in enumerate(folds):
            folder = f"fold_{fold_index:02d}"
            _write_split(splits, manifest, dataset_root / folder)
            dataset_manifest["folds"].append(
                {
                    "fold": fold_index,
                    "folder": folder,
                    "n_train_items": manifest["partitions"]["train"]["n_items"],
                    "n_val_items": manifest["partitions"]["val"]["n_items"],
                    "n_test_items": manifest["partitions"]["test"]["n_items"],
                }
            )
        dataset_root.mkdir(parents=True, exist_ok=True)
        (dataset_root / "manifest.json").write_text(
            json.dumps(dataset_manifest, indent=2), encoding="utf-8"
        )
        root_manifest["datasets"].append(
            {"dataset": dataset, "folder": dataset, "protocol": protocol}
        )
    (output_root / "manifest.json").write_text(
        json.dumps(root_manifest, indent=2), encoding="utf-8"
    )
    return root_manifest


def build_windowed_splits(
    df: pd.DataFrame,
    output_dir: Path,
    window_sizes: Iterable[float] = _WINDOW_SIZES,
    test_frac: float = _WINDOW_TEST_FRAC,
    val_frac: float = _WINDOW_VAL_FRAC,
    tag: str = "full",
    seed: int = SEED,
) -> dict:
    """Write nested training windows with fixed validation and test item sets."""
    window_sizes = list(window_sizes)
    if test_frac + val_frac >= 1.0:
        raise ValueError("test_frac + val_frac must leave a non-empty training pool")
    if any(window <= 0 or window > 1 for window in window_sizes):
        raise ValueError("Window sizes must be in (0, 1]")

    item_ids = df["item_id"].drop_duplicates().tolist()
    rng = np.random.RandomState(seed)
    rng.shuffle(item_ids)
    n_test = int(len(item_ids) * test_frac)
    n_val = int(len(item_ids) * val_frac)
    test_ids = set(item_ids[:n_test])
    val_ids = set(item_ids[n_test : n_test + n_val])
    train_pool = item_ids[n_test + n_val :]
    test_votes = df[df["item_id"].isin(test_ids)].copy()
    val_votes = df[df["item_id"].isin(val_ids)].copy()

    root = output_dir / "windows" / tag
    root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "tag": tag,
        "split_type": "nested_training_windows",
        "seed": seed,
        "n_items_total": len(item_ids),
        "n_items_test": len(test_ids),
        "n_items_val": len(val_ids),
        "n_items_trainable_pool": len(train_pool),
        "test_frac": test_frac,
        "val_frac": val_frac,
        "windows": [],
    }
    previous_train_ids: set = set()
    for window in window_sizes:
        train_ids = set(train_pool[: int(len(train_pool) * window)])
        if not previous_train_ids.issubset(train_ids):
            raise AssertionError("Training windows are not nested")
        previous_train_ids = train_ids
        train_votes = df[df["item_id"].isin(train_ids)].copy()
        folder = root / f"w{int(window * 100):03d}"
        folder.mkdir(exist_ok=True)
        train_path = folder / "train_votes.parquet"
        val_path = folder / "val_votes.parquet"
        test_path = folder / "test_votes.parquet"
        train_votes.to_parquet(train_path, index=False)
        val_votes.to_parquet(val_path, index=False)
        test_votes.to_parquet(test_path, index=False)
        manifest["windows"].append(
            {
                "w": window,
                "folder": folder.name,
                "n_train_items": len(train_ids),
                "n_val_items": len(val_ids),
                "n_test_items": len(test_ids),
                "n_train_votes": len(train_votes),
                "n_val_votes": len(val_votes),
                "n_test_votes": len(test_votes),
            }
        )
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def save_filter_report(
    filter_report: Dict[str, Dict],
    density_filtered: pd.DataFrame,
    intersection: pd.DataFrame,
    output_dir: Path,
) -> dict:
    """Write the JSON-safe per-method coverage report."""
    serializable = {
        name: {
            key: value
            for key, value in entry.items()
            if key not in {"valid_item_ids", "valid_usernames"}
        }
        for name, entry in filter_report.items()
    }
    serializable["__intersection__"] = {
        "n_votes": len(intersection),
        "n_items_kept": int(intersection["item_id"].nunique()),
        "n_items_dropped": int(
            density_filtered["item_id"].nunique() - intersection["item_id"].nunique()
        ),
        "n_users_kept": int(intersection["username"].nunique()),
        "n_users_dropped": int(
            density_filtered["username"].nunique() - intersection["username"].nunique()
        ),
        "n_communities": int(intersection["community"].nunique()),
        "reason": "Intersection of all final-method populations followed by density filtering.",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "filter_report.json").write_text(
        json.dumps(serializable, indent=2), encoding="utf-8"
    )
    return serializable


def prepare_dataset(
    input_path: Path = INPUT_PATH,
    output_dir: Path = OUTPUT_DIR,
    splits_dir: Path = SPLITS_DIR,
    external_scores_path: Optional[Path] = None,
    post_texts_path: Optional[Path] = None,
    n_folds: int = _K_FOLDS,
    chronological_initial_train_fraction: float = _CHRONOLOGICAL_INITIAL_TRAIN_FRAC,
    seed: int = SEED,
) -> dict[str, Any]:
    """Run the complete offline Reddit preparation pipeline."""
    # ``None`` means the item table was not supplied: the SEF and BL4/BL5
    # population filters are then not applied (recorded in the filter report).
    cleaned = load_dataset(input_path)
    density_filtered = apply_density_filter(cleaned)
    output_dir.mkdir(parents=True, exist_ok=True)
    density_filtered.to_parquet(output_dir / "filtered_votes.parquet", index=False)

    filter_report = identify_method_filters(
        density_filtered,
        external_scores_path=external_scores_path,
        post_texts_path=post_texts_path,
    )
    intersection = build_intersection_dataset(density_filtered, filter_report)
    save_filter_report(filter_report, density_filtered, intersection, output_dir)

    full_splits, full_manifest = build_seeded_item_split(cleaned, seed=seed)
    full_manifest["population"] = "cleaned_pre_density"
    _write_split(full_splits, full_manifest, splits_dir / "full")

    intersection_splits, intersection_manifest = build_seeded_item_split(intersection, seed=seed)
    intersection_manifest["population"] = "all_method_intersection"
    _write_split(intersection_splits, intersection_manifest, splits_dir / "intersection")

    full_windows = build_windowed_splits(density_filtered, splits_dir, tag="full", seed=seed)
    intersection_windows = build_windowed_splits(
        intersection, splits_dir, tag="intersection", seed=seed
    )
    chronological_splits, chronological_manifest = build_chronological_split(intersection)
    chronological_manifest["population"] = "all_method_intersection"
    _write_split(
        chronological_splits,
        chronological_manifest,
        splits_dir / "intersection_chronological",
    )
    kfold_manifest = write_kfold_splits(
        cleaned,
        intersection,
        splits_dir / "kfold",
        n_folds=n_folds,
        seed=seed,
        chronological_initial_train_fraction=chronological_initial_train_fraction,
    )

    pipeline_manifest = {
        "pipeline": "prepare_reddit_data",
        "network_access": False,
        "seed": seed,
        "input": input_path.as_posix(),
        "input_sha256": _sha256_file(input_path),
        "optional_inputs": {
            "moderated_posts_scores": {
                "path": external_scores_path.as_posix(),
                "available": external_scores_path.exists(),
                "sha256": _sha256_file(external_scores_path),
            },
            "post_texts": {
                "path": post_texts_path.as_posix(),
                "available": post_texts_path.exists(),
                "sha256": _sha256_file(post_texts_path),
            },
        },
        "populations": {
            "cleaned": len(cleaned),
            "density_filtered": len(density_filtered),
            "intersection": len(intersection),
        },
        "outputs": {
            "filtered_votes": (output_dir / "filtered_votes.parquet").as_posix(),
            "splits_root": splits_dir.as_posix(),
            "kfold_root": (splits_dir / "kfold").as_posix(),
        },
    }
    (output_dir / "prepare_reddit_data.manifest.json").write_text(
        json.dumps(pipeline_manifest, indent=2), encoding="utf-8"
    )
    log.info("Offline Reddit preparation complete")
    return {
        "cleaned_df": cleaned,
        "filtered_df": density_filtered,
        "intersection_df": intersection,
        "filter_report": filter_report,
        "splits": {
            "full": full_splits,
            "intersection": intersection_splits,
            "intersection_chronological": chronological_splits,
        },
        "window_manifests": {
            "full": full_windows,
            "intersection": intersection_windows,
        },
        "kfold_manifest": kfold_manifest,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=INPUT_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--splits-dir", type=Path, default=SPLITS_DIR)
    parser.add_argument(
        "--external-scores", type=Path, default=REDDIT.default_inputs().item_scores
    )
    parser.add_argument("--post-texts", type=Path, default=REDDIT.default_inputs().post_texts)
    parser.add_argument("--n-folds", type=int, default=_K_FOLDS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--chronological-initial-train-fraction",
        type=float,
        default=_CHRONOLOGICAL_INITIAL_TRAIN_FRAC,
    )
    args = parser.parse_args()
    prepare_dataset(
        input_path=args.input,
        output_dir=args.output_dir,
        splits_dir=args.splits_dir,
        external_scores_path=args.external_scores,
        post_texts_path=args.post_texts,
        n_folds=args.n_folds,
        seed=args.seed,
        chronological_initial_train_fraction=args.chronological_initial_train_fraction,
    )


if __name__ == "__main__":
    main()
