"""Stack simulated consensus from two profile-based subteams."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.methods.team_formation_v2.bandit.pipeline import (  # noqa: E402
    build_candidate_profiles,
    split_train_items,
)
from src.methods.team_formation_v2.bandit.vote_simulator import (  # noqa: E402
    HierarchicalVoteSimulator,
    PostContextEncoder,
)
from src.methods.team_formation_v2.role_nsga2.pipeline import (  # noqa: E402
    case_topic_affinity,
    fit_user_topic_centroids,
)
from src.methods.team_formation_v2.shared_features import (  # noqa: E402
    DEFAULT_CAUSAL_FEATURES,
    apply_thresholds,
    calibrate_thresholds_per_community,
    classification_metrics,
    load_and_enrich_splits,
    stable_top_indices,
    with_columns,
)

DEFAULT_VOTES_DIR = "data/splits/reddit/intersection_chronological"
DEFAULT_OUT_DIR = "results/reddit/intersection_chronological/virtual_ensembles"
DEFAULT_EMBEDDING_DIR = "cache/embeddings"
TEAM_NAMES = (
    "high_tenure_low_activity",
    "domain_expert_high_moderator_agreement",
)
CONSENSUS_FEATURES = [f"consensus_{name}" for name in TEAM_NAMES]


def _stable_top(
    group: pd.DataFrame,
    usernames: list[str],
    score: np.ndarray,
    team_size: int,
    extra_columns: dict[str, np.ndarray] | None = None,
) -> pd.DataFrame:
    """Select a deterministic top-k without using current-case votes."""
    selected = stable_top_indices(score, usernames, team_size)
    trailing = {name: values[selected] for name, values in (extra_columns or {}).items()}
    trailing["profile_match"] = score[selected]
    return with_columns(group.iloc[selected], trailing=trailing)


def _team_scores(
    tenure: np.ndarray,
    local_activity: np.ndarray,
    topical_expertise: np.ndarray,
    alignment: np.ndarray,
) -> dict[str, np.ndarray]:
    """Profile-match scores for exactly the two teams in the specification."""
    return {
        "high_tenure_low_activity": np.sqrt(
            np.clip(tenure, 0, 1) * (1.0 - np.clip(local_activity, 0, 1))
        ),
        "domain_expert_high_moderator_agreement": np.sqrt(
            np.clip(topical_expertise, 0, 1) * np.clip(alignment, 0, 1)
        ),
    }


def select_virtual_teams(candidates: pd.DataFrame, team_size: int) -> dict[str, pd.DataFrame]:
    """Construct exactly the two complementary profiles in the specification."""
    scores = _team_scores(
        candidates["profile_tenure"].to_numpy(dtype=float),
        candidates["profile_local_activity"].to_numpy(dtype=float),
        candidates["topical_expertise"].to_numpy(dtype=float),
        candidates["profile_alignment_local"].to_numpy(dtype=float),
    )
    usernames = candidates["username"].astype(str).tolist()
    return {
        name: _stable_top(candidates, usernames, score, team_size)
        for name, score in scores.items()
    }


def build_case_features(
    item_id: object,
    community: str,
    label: int,
    timestamp: object,
    candidates: pd.DataFrame,
    simulator: HierarchicalVoteSimulator,
    team_size: int,
    teams: dict[str, pd.DataFrame] | None = None,
) -> tuple[dict, pd.DataFrame]:
    """Simulate the union of selected panels and summarize each team."""
    if teams is None:
        teams = select_virtual_teams(candidates, team_size)
    union_users = list(
        dict.fromkeys(
            user
            for team in teams.values()
            for user in team["username"].astype(str).tolist()
        )
    )
    simulated_scores = simulator.predict_scores(item_id, community, union_users)
    score_lookup = dict(zip(union_users, simulated_scores))
    vote_lookup = {
        username: 1 if score >= 0 else -1
        for username, score in score_lookup.items()
    }
    row = {
        "item_id": item_id,
        "community": community,
        "label": int(label),
        "timestamp": timestamp,
        "candidate_pool_size": int(len(candidates)),
        "unique_team_members": int(len(union_users)),
    }
    membership_rows = []
    for team_name, team in teams.items():
        users = team["username"].astype(str).tolist()
        binary = np.asarray([vote_lookup[user] for user in users], dtype=float)
        soft = np.asarray([score_lookup[user] for user in users], dtype=float)
        row[f"consensus_{team_name}"] = float(soft.mean())
        row[f"binary_consensus_{team_name}"] = float(binary.mean())
        row[f"size_{team_name}"] = int(len(team))
        annotated = with_columns(
            team,
            {"item_id": item_id},
            {
                "label": int(label),
                "team_name": team_name,
                "simulated_vote_score": soft,
                "simulated_vote": binary.astype(int),
                "selected_rank": np.arange(1, len(team) + 1),
            },
        )
        membership_rows.append(annotated)

    membership = pd.concat(membership_rows, ignore_index=True)
    return row, membership


def build_features(
    cases: pd.DataFrame,
    candidate_pools: dict[str, pd.DataFrame],
    simulator: HierarchicalVoteSimulator,
    centroids: dict[str, np.ndarray],
    team_size: int,
    max_candidates: int,
    split_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build label-safe item features and virtual-team memberships."""
    ordered = (
        cases.sort_values("timestamp")
        .drop_duplicates("item_id")
        [["item_id", "community", "label", "timestamp"]]
    )
    rows = []
    members = []
    # Per-community pool arrays are static; only topic affinity changes per case,
    # so teams are scored on arrays (same as attach_case_expertise followed by
    # select_virtual_teams) and frames are built for selected members only.
    prepared: dict[str, tuple] = {}
    for case in ordered.itertuples(index=False):
        pool = candidate_pools.get(str(case.community))
        if pool is None or pool.empty:
            continue
        if str(case.community) not in prepared:
            pool = pool.head(max_candidates) if max_candidates > 0 else pool
            pool = pool.reset_index(drop=True)
            prepared[str(case.community)] = (
                pool,
                pool["username"].astype(str).tolist(),
                pool["profile_tenure"].to_numpy(dtype=float),
                pool["profile_local_activity"].to_numpy(dtype=float),
                pool["profile_alignment_local"].to_numpy(dtype=float),
            )
        pool, usernames, tenure, local_activity, alignment = prepared[str(case.community)]
        affinity = case_topic_affinity(usernames, case.item_id, simulator.encoder, centroids)
        affinity_columns = {"case_topic_affinity": affinity, "topical_expertise": affinity}
        teams = {
            name: _stable_top(pool, usernames, score, team_size, affinity_columns)
            for name, score in _team_scores(
                tenure, local_activity, affinity, alignment
            ).items()
        }
        row, membership = build_case_features(
            case.item_id,
            str(case.community),
            int(case.label),
            case.timestamp,
            pool,
            simulator,
            team_size,
            teams=teams,
        )
        rows.append(row)
        members.append(membership)
    membership = pd.concat(members, ignore_index=True) if members else pd.DataFrame()
    return pd.DataFrame(rows), membership


def positive_scores(model, features: pd.DataFrame) -> np.ndarray:
    positive_index = int(np.where(model.classes_ == 1)[0][0])
    return model.predict_proba(features)[:, positive_index]


def _parse_regularization_grid(raw: str) -> list[float]:
    values = sorted({float(part.strip()) for part in raw.split(",") if part.strip()})
    if not values or any(value <= 0 for value in values):
        raise ValueError("regularization-grid must contain positive values")
    return values


def _meta_model(regularization_c: float):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=regularization_c,
            max_iter=2_000,
            class_weight="balanced",
            random_state=10,
        ),
    )


def select_regularization(
    frame: pd.DataFrame,
    feature_columns: list[str],
    regularization_grid: list[float],
    tuning_fraction: float,
) -> dict:
    """Select C on the chronologically latest part of the meta-training suffix."""
    order = np.argsort(frame["timestamp"].to_numpy(), kind="stable")
    cut = min(max(int(len(order) * (1.0 - tuning_fraction)), 2), len(order) - 2)
    fit_indices = order[:cut]
    tune_indices = order[cut:]
    labels = frame["label"].to_numpy(dtype=int)
    candidates = []
    best = None
    for regularization_c in regularization_grid:
        model = _meta_model(regularization_c)
        model.fit(frame.iloc[fit_indices][feature_columns], labels[fit_indices])
        scores = positive_scores(model, frame.iloc[tune_indices][feature_columns])
        auc = float(roc_auc_score(labels[tune_indices], scores))
        candidate = {
            "regularization_c": float(regularization_c),
            "inner_tune_auc": auc,
        }
        candidates.append(candidate)
        if best is None or auc > best["inner_tune_auc"]:
            best = candidate
    if best is None:
        raise ValueError("Could not select virtual-ensemble regularization")
    return {
        "selected": best,
        "candidates": candidates,
        "selection_metric": "ROC-AUC",
        "inner_split": {
            "protocol": "chronological_inner_holdout",
            "fit_items": int(len(fit_indices)),
            "tune_items": int(len(tune_indices)),
            "tuning_fraction": float(tuning_fraction),
            "cut_timestamp": float(frame.iloc[fit_indices[-1]]["timestamp"]),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--votes-dir", default=DEFAULT_VOTES_DIR)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--causal-features", type=Path, default=DEFAULT_CAUSAL_FEATURES)
    parser.add_argument("--embedding-dir", type=Path, default=DEFAULT_EMBEDDING_DIR)
    parser.add_argument("--team-size", type=int, default=3)
    parser.add_argument("--max-candidates", type=int, default=0)
    parser.add_argument("--min-user-history", type=int, default=2)
    parser.add_argument("--simulator-fraction", type=float, default=0.70)
    parser.add_argument("--embedding-dimensions", type=int, default=16)
    parser.add_argument("--regularization-grid", default="0.01,0.03,0.1,0.3,1.0")
    parser.add_argument("--tuning-fraction", type=float, default=0.25)
    args = parser.parse_args()
    if (
        args.team_size < 1
        or args.max_candidates < 0
        or args.min_user_history < 1
        or not 0.1 <= args.tuning_fraction <= 0.5
    ):
        raise ValueError(
            "team-size>=1, max-candidates>=0, min-user-history>=1, and "
            "tuning-fraction between 0.1 and 0.5 required"
        )
    regularization_grid = _parse_regularization_grid(args.regularization_grid)

    votes_dir = Path(args.votes_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train, val, test = load_and_enrich_splits(votes_dir, args.causal_features)
    simulator_train, meta_train, split_report = split_train_items(
        train, args.simulator_fraction
    )
    encoder = PostContextEncoder(
        args.embedding_dir, dimensions=args.embedding_dimensions
    ).fit(simulator_train)
    simulator = HierarchicalVoteSimulator(encoder).fit(simulator_train.reset_index(drop=True))
    candidate_pools, profiles = build_candidate_profiles(
        simulator_train, simulator, args.min_user_history
    )
    candidate_pools = {
        community: group.reset_index(drop=True)
        for community, group in profiles.groupby("community", sort=False)
    }
    centroids = fit_user_topic_centroids(simulator_train, encoder)

    frames = {}
    memberships = {}
    for name, cases in (("meta_train", meta_train), ("val", val), ("test", test)):
        frames[name], memberships[name] = build_features(
            cases,
            candidate_pools,
            simulator,
            centroids,
            args.team_size,
            args.max_candidates,
            name,
        )
        if frames[name].empty:
            raise ValueError(f"No eligible cases were produced for {name}")

    feature_columns = list(CONSENSUS_FEATURES)
    missing_features = set(feature_columns).difference(frames["meta_train"].columns)
    if missing_features:
        raise ValueError(f"Missing virtual-team consensuses: {sorted(missing_features)}")
    hyperparameter_search = select_regularization(
        frames["meta_train"],
        feature_columns,
        regularization_grid,
        args.tuning_fraction,
    )
    selected = hyperparameter_search["selected"]
    print(
        f"Selected meta-learner C={selected['regularization_c']:.4g} "
        f"(inner AUC={selected['inner_tune_auc']:.4f})"
    )
    model = _meta_model(selected["regularization_c"])
    model.fit(frames["meta_train"][feature_columns], frames["meta_train"]["label"])
    scored = {}
    for name, frame in frames.items():
        output = frame.copy()
        output["score"] = positive_scores(model, frame[feature_columns])
        scored[name] = output
    thresholds, global_threshold = calibrate_thresholds_per_community(scored["val"])
    predictions = {
        name: apply_thresholds(frame, thresholds, global_threshold)
        for name, frame in scored.items()
    }
    results = {
        name: classification_metrics(frame) for name, frame in predictions.items()
    }
    results["coverage"] = {
        name: {
            "eligible_items": int(len(frames[name])),
            "total_items": int(cases["item_id"].nunique()),
        }
        for name, cases in (("meta_train", meta_train), ("val", val), ("test", test))
    }
    results["simulator_fidelity_on_observed_votes"] = {
        "val": simulator.evaluate_observed(val.reset_index(drop=True)),
        "test": simulator.evaluate_observed(test.reset_index(drop=True)),
    }
    results["ensemble"] = {
        "teams": list(TEAM_NAMES),
        "team_definitions": {
            "high_tenure_low_activity": (
                "geometric conjunction of prefix-only tenure and inverse local activity"
            ),
            "domain_expert_high_moderator_agreement": (
                "geometric conjunction of case topic affinity and prefix-only local "
                "moderator agreement"
            ),
        },
        "teams_are_overlapping": True,
        "team_size": int(args.team_size),
        "meta_learner": "standardized class-balanced logistic regression",
        "regularization_c": float(selected["regularization_c"]),
        "hyperparameter_search": hyperparameter_search,
        "verdict_features": "the two continuous sub-team consensuses only",
        "n_model_features": int(len(feature_columns)),
        "selection_uses_case_votes": False,
        "case_affinity": "cosine similarity to prefix-only user content centroid",
    }
    results["protocol"] = {
        **split_report,
        "policy_suffix_use": (
            "chronological inner C tuning, then full-suffix meta-learner refit"
        ),
        "candidate_universe": "same prefix-only community pools as team_formation_v2.bandit",
        "historical_case_voters_used_for_team_selection": False,
        "votes_used_for_team_consensus": "selected users' frozen simulator outputs",
        "validation_test_updates": False,
        "max_candidates": int(args.max_candidates),
        "min_user_history": int(args.min_user_history),
        "n_candidate_profiles": int(len(profiles)),
        "n_candidate_communities": int(len(candidate_pools)),
        "simulated_estimand": True,
        "warning": (
            "Policy metrics describe a simulated team-formation world and must "
            "be reported with simulator fidelity and coverage."
        ),
    }
    (out_dir / "metrics.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    predictions["val"].to_parquet(out_dir / "val_predictions.parquet", index=False)
    predictions["test"].to_parquet(out_dir / "test_predictions.parquet", index=False)
    memberships["val"].to_parquet(out_dir / "val_team_membership.parquet", index=False)
    memberships["test"].to_parquet(out_dir / "test_team_membership.parquet", index=False)
    profiles.to_parquet(out_dir / "candidate_profiles.parquet", index=False)
    coefficient_model = model.named_steps["logisticregression"]
    coefficients = pd.DataFrame(
        {
            "feature": feature_columns,
            "coefficient": coefficient_model.coef_[0],
        }
    ).sort_values("coefficient", ascending=False)
    coefficients.to_csv(out_dir / "meta_learner_coefficients.csv", index=False)
    print(
        f"Virtual ensembles: test macro-F1={results['test']['macro_f1']:.4f} | "
        f"saved={out_dir}"
    )


if __name__ == "__main__":
    main()
