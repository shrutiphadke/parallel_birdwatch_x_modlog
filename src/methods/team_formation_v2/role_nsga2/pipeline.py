"""Form simulated panels with four-objective NSGA-II optimization."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

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
from src.methods.team_formation_v2.role_nsga2.optimizer import (  # noqa: E402
    CommunityNSGA2Optimizer,
)
from src.methods.team_formation_v2.role_utils import (  # noqa: E402
    ROLE_NAMES,
    assign_core_periphery_roles,
    fit_familiarity,
    fit_ideology,
)
from src.methods.team_formation_v2.shared_features import (  # noqa: E402
    DEFAULT_CAUSAL_FEATURES,
    apply_thresholds,
    calibrate_thresholds_per_community,
    centroid_affinities,
    classification_metrics,
    load_and_enrich_splits,
    with_columns,
)

DEFAULT_VOTES_DIR = "data/splits/reddit/intersection_chronological"
DEFAULT_OUT_DIR = "results/reddit/intersection_chronological/role_nsga2"
DEFAULT_EMBEDDING_DIR = "cache/embeddings"

def fit_user_topic_centroids(
    reference: pd.DataFrame,
    encoder: PostContextEncoder,
) -> dict[str, np.ndarray]:
    """Estimate each candidate's historical content centroid from the prefix."""
    item_vectors = {
        item_id: encoder.content_vector(item_id)[0]
        for item_id in reference["item_id"].drop_duplicates()
    }
    centroids = {}
    for username, group in reference.groupby("username", sort=False):
        vectors = np.vstack([item_vectors[item_id] for item_id in group["item_id"]])
        centroid = vectors.mean(axis=0)
        norm = float(np.linalg.norm(centroid))
        centroids[str(username)] = centroid / norm if norm > 0 else centroid
    return centroids


def case_topic_affinity(
    usernames: list[str],
    item_id: object,
    encoder: PostContextEncoder,
    centroids: dict[str, np.ndarray],
) -> np.ndarray:
    """Clipped prefix-only user/content cosine affinity for one case."""
    content, _ = encoder.content_vector(item_id)
    if float(np.linalg.norm(content)) > 0:
        affinities = centroid_affinities(usernames, centroids, content)
    else:
        affinities = np.full(len(usernames), 0.5, dtype=float)
    return np.clip(affinities, 0.0, 1.0)


def attach_case_expertise(
    candidates: pd.DataFrame,
    item_id: object,
    encoder: PostContextEncoder,
    centroids: dict[str, np.ndarray],
) -> pd.DataFrame:
    """Use prefix-only user/content affinity as topical expertise for the case."""
    affinity = case_topic_affinity(
        candidates["username"].astype(str).tolist(), item_id, encoder, centroids
    )
    return with_columns(
        candidates,
        trailing={"case_topic_affinity": affinity, "topical_expertise": affinity},
    )


def score_cases(
    cases: pd.DataFrame,
    optimizers: dict[str, CommunityNSGA2Optimizer],
    simulator: HierarchicalVoteSimulator,
    centroids: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compose one counterfactual panel for every case and simulate its votes."""
    item_rows = []
    member_rows = []
    ordered = (
        cases.sort_values("timestamp")
        .drop_duplicates("item_id")
        [["item_id", "community", "label", "timestamp"]]
    )
    for row in ordered.itertuples(index=False):
        optimizer = optimizers.get(str(row.community))
        if optimizer is None:
            item_rows.append(
                {
                    "item_id": row.item_id,
                    "community": str(row.community),
                    "label": int(row.label),
                    "timestamp": row.timestamp,
                    "score": 0.0,
                    "team_size": 0,
                    "candidate_pool_size": 0,
                    "eligible": False,
                }
            )
            continue
        candidates = optimizer.candidates
        expertise = case_topic_affinity(
            optimizer.usernames, row.item_id, simulator.encoder, centroids
        )
        seed = int.from_bytes(
            hashlib.blake2b(str(row.item_id).encode(), digest_size=4).digest(), "little"
        )
        selected, objectives, front_size, ideal_distance = optimizer.compose(
            expertise,
            seed,
        )
        # Equivalent to attach_case_expertise on the pool, restricted to the team.
        team = with_columns(
            candidates.iloc[list(selected)],
            trailing={
                "case_topic_affinity": expertise[list(selected)],
                "topical_expertise": expertise[list(selected)],
            },
        )
        simulated_scores = simulator.predict_scores(
            row.item_id, str(row.community), team["username"].astype(str).tolist()
        )
        simulated_votes = np.where(simulated_scores >= 0, 1, -1)
        item_rows.append(
            {
                "item_id": row.item_id,
                "community": str(row.community),
                "label": int(row.label),
                "timestamp": row.timestamp,
                "score": float(simulated_scores.mean()),
                "binary_vote_score": float(simulated_votes.mean()),
                "team_size": int(len(team)),
                "candidate_pool_size": int(len(candidates)),
                "eligible": True,
                "pareto_front_size": int(front_size),
                "objective_ideological_diversity": float(objectives[0]),
                "objective_topical_expertise": float(objectives[1]),
                "objective_moderator_mix": float(objectives[2]),
                "objective_familiarity": float(objectives[3]),
                "pareto_ideal_distance": float(ideal_distance),
            }
        )
        team = with_columns(
            team,
            {"item_id": row.item_id},
            {
                "label": int(row.label),
                "simulated_vote_score": simulated_scores,
                "simulated_vote": simulated_votes,
                "selected_rank": np.arange(1, len(team) + 1),
                "pareto_front_size": int(front_size),
                "pareto_ideal_distance": float(ideal_distance),
            },
        )
        member_rows.append(team)
    members = pd.concat(member_rows, ignore_index=True) if member_rows else pd.DataFrame()
    return pd.DataFrame(item_rows), members


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
    parser.add_argument("--population-size", type=int, default=32)
    parser.add_argument("--generations", type=int, default=20)
    args = parser.parse_args()
    if (
        args.team_size < 1
        or args.max_candidates < 0
        or args.min_user_history < 1
        or args.population_size < 4
        or args.generations < 1
    ):
        raise ValueError(
            "team-size>=1, max-candidates>=0, min-user-history>=1, "
            "population-size>=4, and generations>=1 are required"
        )

    votes_dir = Path(args.votes_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train, val, test = load_and_enrich_splits(votes_dir, args.causal_features)
    simulator_train, policy_suffix, split_report = split_train_items(
        train, args.simulator_fraction
    )
    encoder = PostContextEncoder(
        args.embedding_dir, dimensions=args.embedding_dimensions
    ).fit(simulator_train)
    simulator = HierarchicalVoteSimulator(encoder).fit(simulator_train.reset_index(drop=True))
    candidate_pools, profiles = build_candidate_profiles(
        simulator_train, simulator, args.min_user_history
    )
    familiarity, familiarity_scales = fit_familiarity(simulator_train)
    profiles = assign_core_periphery_roles(profiles, familiarity)
    ideology = fit_ideology(simulator_train)
    profiles["mf_viewpoint"] = profiles["username"].astype(str).map(
        {user: float(vector[0]) for user, vector in ideology.items()}
    )
    candidate_pools = {
        community: group.reset_index(drop=True)
        for community, group in profiles.groupby("community", sort=False)
    }
    centroids = fit_user_topic_centroids(simulator_train, encoder)
    optimizers = {
        community: CommunityNSGA2Optimizer(
            group.head(args.max_candidates) if args.max_candidates > 0 else group,
            ideology,
            familiarity,
            familiarity_scales,
            args.team_size,
            args.population_size,
            args.generations,
        )
        for community, group in candidate_pools.items()
    }

    val_scores, val_members = score_cases(
        val,
        optimizers,
        simulator,
        centroids,
    )
    test_scores, test_members = score_cases(
        test,
        optimizers,
        simulator,
        centroids,
    )
    val_eligible = val_scores[val_scores["eligible"]].copy()
    test_eligible = test_scores[test_scores["eligible"]].copy()
    if val_eligible.empty or test_eligible.empty:
        raise ValueError("No eligible validation/test cases for known candidate pools")
    thresholds, global_threshold = calibrate_thresholds_per_community(val_eligible)
    predictions = {
        "val": apply_thresholds(val_eligible, thresholds, global_threshold),
        "test": apply_thresholds(test_eligible, thresholds, global_threshold),
    }
    results = {name: classification_metrics(frame) for name, frame in predictions.items()}
    results["coverage"] = {
        "val_eligible_items": int(len(val_eligible)),
        "val_total_items": int(len(val_scores)),
        "test_eligible_items": int(len(test_eligible)),
        "test_total_items": int(len(test_scores)),
    }
    results["simulator_fidelity_on_observed_votes"] = {
        "val": simulator.evaluate_observed(val.reset_index(drop=True)),
        "test": simulator.evaluate_observed(test.reset_index(drop=True)),
    }
    results["composition"] = {
        "algorithm": "NSGA-II",
        "objectives": [
            "ideological_diversity",
            "case_topical_expertise",
            "moderator_proximity_mix",
            "familiarity",
        ],
        "objective_definitions": {
            "ideological_diversity": "mean pairwise distance between prefix-only CN MF user factors",
            "case_topical_expertise": "mean case-specific user/content cosine affinity",
            "moderator_proximity_mix": "coverage of both the community-relative high and middle agreement bands",
            "familiarity": "mean normalized number of prefix cases previously shared by member pairs",
        },
        "pareto_choice": (
            "ordinary four-objective nondominated sorting; final panel is the "
            "closest-to-ideal normalized compromise on the first front"
        ),
        "team_size": int(args.team_size),
        "population_size": int(args.population_size),
        "generations": int(args.generations),
        "roles": list(ROLE_NAMES),
        "roles_are_per_community": True,
        "role_assignment": "core is the maximal k-core of the prefix co-voting graph; all other observed candidates are periphery",
        "role_use": "candidate representation and composition audit, not an additional fifth objective",
        "case_affinity": "cosine similarity to prefix-only user content centroid",
        "ideology_fit": "official Community Notes one-factor MF on simulator-prefix votes",
        "familiarity_fit": "community-specific simulator-prefix shared-case counts",
    }
    results["protocol"] = {
        **split_report,
        "unused_policy_suffix_items": int(policy_suffix["item_id"].nunique()),
        "candidate_universe": "same prefix-only community pools as team_formation_v2.bandit",
        "historical_case_voters_used_for_composition": False,
        "votes_used_for_panel_decision": "selected users' frozen simulator outputs",
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
    val_members.to_parquet(out_dir / "val_team_membership.parquet", index=False)
    test_members.to_parquet(out_dir / "test_team_membership.parquet", index=False)
    profiles.to_parquet(out_dir / "candidate_roles.parquet", index=False)
    print(
        f"Role NSGA-II: test macro-F1={results['test']['macro_f1']:.4f} | "
        f"saved={out_dir}"
    )


if __name__ == "__main__":
    main()
