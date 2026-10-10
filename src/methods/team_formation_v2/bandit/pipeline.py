"""Select simulated community panels with LinUCB."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.methods.team_formation_v2.bandit.bandit import LinearUCB  # noqa: E402
from src.methods.team_formation_v2.bandit.vote_simulator import (  # noqa: E402
    HierarchicalVoteSimulator,
    PostContextEncoder,
)
from src.methods.team_formation_v2.shared_features import (  # noqa: E402
    DEFAULT_CAUSAL_FEATURES,
    apply_thresholds,
    calibrate_thresholds_per_community,
    classification_metrics,
    load_and_enrich_splits,
    with_columns,
)

DEFAULT_VOTES_DIR = "data/splits/reddit/intersection_chronological"
DEFAULT_OUT_DIR = "results/reddit/intersection_chronological/bandit"
DEFAULT_EMBEDDING_DIR = "cache/embeddings"


def split_train_items(
    train: pd.DataFrame,
    simulator_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Make a chronological simulator-prefix/policy-suffix split by case."""
    if not 0.2 <= simulator_fraction <= 0.9:
        raise ValueError("simulator-fraction must be between 0.2 and 0.9")
    order = (
        train.groupby("item_id", as_index=False)["timestamp"]
        .max()
        .sort_values(["timestamp", "item_id"], kind="stable")
    )
    if len(order) < 4:
        raise ValueError("At least four training cases are required")
    cut = min(max(int(len(order) * simulator_fraction), 2), len(order) - 2)
    prefix_ids = set(order.iloc[:cut]["item_id"])
    prefix = train[train["item_id"].isin(prefix_ids)].copy()
    suffix = train[~train["item_id"].isin(prefix_ids)].copy()
    report = {
        "protocol": "chronological_disjoint_prefix_suffix",
        "simulator_fraction": float(simulator_fraction),
        "simulator_items": int(prefix["item_id"].nunique()),
        "simulator_votes": int(len(prefix)),
        "policy_items": int(suffix["item_id"].nunique()),
        "policy_votes_ignored_except_case_labels": int(len(suffix)),
        "cut_timestamp": float(order.iloc[cut - 1]["timestamp"]),
    }
    return prefix, suffix, report


def build_candidate_profiles(
    reference: pd.DataFrame,
    simulator: HierarchicalVoteSimulator,
    min_history: int,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Create frozen user/community profiles and eligible candidate pools."""
    work = reference.copy()
    work["_agree"] = (work["vote"] == work["label"]).astype(float)
    global_base = float(work["_agree"].mean()) if len(work) else 0.5
    global_stats = work.groupby("username").agg(
        global_agree=("_agree", "sum"),
        global_n=("_agree", "size"),
    )
    local_stats = work.groupby(["username", "community"]).agg(
        local_agree=("_agree", "sum"),
        local_n=("_agree", "size"),
    )
    latest = (
        work.sort_values("timestamp")
        .groupby(["username", "community"], as_index=False)
        .tail(1)
        .copy()
    )
    latest = latest.join(global_stats, on="username").join(
        local_stats, on=["username", "community"]
    )
    strength = 5.0
    latest["alignment_global"] = (
        latest["global_agree"] + strength * global_base
    ) / (latest["global_n"] + strength)
    latest["alignment_local"] = (
        latest["local_agree"] + strength * global_base
    ) / (latest["local_n"] + strength)
    latest["simulator_n"] = latest["username"].map(simulator.user_counts).fillna(0)
    latest = latest[latest["local_n"] >= min_history].copy()

    global_activity = latest["prior_n_posts"] + latest["prior_n_comments"]
    local_activity = latest["prior_n_posts_in_sub"] + latest["prior_n_comments_in_sub"]
    latest["profile_alignment_global"] = latest["alignment_global"].clip(0, 1)
    latest["profile_alignment_local"] = latest["alignment_local"].clip(0, 1)
    latest["profile_global_history"] = np.minimum(np.log1p(latest["global_n"]) / 7.0, 1.0)
    latest["profile_local_history"] = np.minimum(np.log1p(latest["local_n"]) / 5.0, 1.0)
    latest["profile_global_activity"] = np.minimum(np.log1p(global_activity) / 10.0, 1.0)
    latest["profile_local_activity"] = np.minimum(np.log1p(local_activity) / 8.0, 1.0)
    latest["profile_domain_share"] = np.divide(
        local_activity,
        np.maximum(global_activity, 1.0),
    ).clip(0, 1)
    known_tenure = latest["tenure_days_at_vote"].notna()
    latest["profile_tenure"] = np.where(
        known_tenure,
        np.minimum(np.log1p(latest["tenure_days_at_vote"].fillna(0).clip(lower=0)) / 9.0, 1.0),
        0.5,
    )
    latest["profile_social"] = np.minimum(
        np.log1p(
            latest["prior_n_interaction_partners_in_sub"]
            + latest["prior_total_interactions_in_sub"]
        )
        / 7.0,
        1.0,
    )
    latest["profile_simulator_history"] = np.minimum(
        np.log1p(latest["simulator_n"]) / 5.0,
        1.0,
    )
    profile_columns = [column for column in latest if column.startswith("profile_")]
    keep = ["username", "community", "local_n", *profile_columns]
    profiles = latest[keep].sort_values(
        ["community", "local_n", "username"], ascending=[True, False, True]
    )
    pools = {
        community: group.reset_index(drop=True)
        for community, group in profiles.groupby("community", sort=False)
    }
    if not pools:
        raise ValueError("No eligible users remain; lower --min-user-history")
    return pools, profiles


class TeamFormationEnvironment:
    """Build contexts, select panels, simulate votes, and expose chosen rewards."""

    def __init__(
        self,
        simulator: HierarchicalVoteSimulator,
        candidate_pools: dict[str, pd.DataFrame],
        team_size: int,
        max_candidates: int,
        softmax_temperature: float,
    ):
        self.simulator = simulator
        self.candidate_pools = candidate_pools
        self.team_size = int(team_size)
        self.max_candidates = int(max_candidates)
        self.softmax_temperature = float(softmax_temperature)
        self.profile_columns = [
            column
            for column in next(iter(candidate_pools.values())).columns
            if column.startswith("profile_")
        ]
        self.community_state: defaultdict[str, dict[str, int]] = defaultdict(
            lambda: {"n": 0, "positive": 0}
        )
        # Pools are frozen, so their per-community arrays are built once.
        self._pool_cache: dict[str, tuple[pd.DataFrame, dict[str, np.ndarray], list[str]]] = {}

    @property
    def feature_dimension(self) -> int:
        # Bias + post context + community history + user profile + two
        # candidate-specific content/profile interaction blocks.
        return (
            1
            + 3 * self.simulator.encoder.content_dimensions
            + 2
            + len(self.profile_columns)
        )

    def seed_community_state(self, reference: pd.DataFrame) -> None:
        """Initialize causal community priors from the simulator prefix."""
        cases = reference.drop_duplicates("item_id")
        for community, group in cases.groupby("community", sort=False):
            self.community_state[str(community)] = {
                "n": int(len(group)),
                "positive": int((group["label"] == 1).sum()),
            }

    def _pool(self, community: str) -> pd.DataFrame:
        return self._prepared_pool(community)[0]

    def _prepared_pool(
        self, community: str
    ) -> tuple[pd.DataFrame, dict[str, np.ndarray], list[str]]:
        key = str(community)
        cached = self._pool_cache.get(key)
        if cached is None:
            pool = self._raw_pool(key)
            arrays = {
                "user": pool[self.profile_columns].to_numpy(dtype=float),
                "local_activity": pool["profile_local_activity"].to_numpy(dtype=float),
                "domain_share": pool["profile_domain_share"].to_numpy(dtype=float),
            }
            cached = (pool, arrays, pool["username"].astype(str).tolist())
            self._pool_cache[key] = cached
        return cached

    def _raw_pool(self, community: str) -> pd.DataFrame:
        pool = self.candidate_pools.get(str(community))
        if pool is None or pool.empty:
            # A community with no historical candidates cannot support genuine
            # local team formation; do not silently import unrelated users.
            return pd.DataFrame(columns=next(iter(self.candidate_pools.values())).columns)
        if self.max_candidates > 0:
            return pool.head(self.max_candidates)
        return pool

    def _features(
        self,
        item_id: object,
        community: str,
        pool: pd.DataFrame,
    ) -> np.ndarray:
        content, _ = self.simulator.encoder.content_vector(item_id)
        state = self.community_state[str(community)]
        prior = (state["positive"] + 1.0) / (state["n"] + 2.0)
        history = min(np.log1p(state["n"]) / 8.0, 1.0)
        shared = np.concatenate(([1.0], content, [prior, history]))
        prepared = self._pool_cache.get(str(community))
        if prepared is not None and prepared[0] is pool:
            arrays = prepared[1]
            user = arrays["user"]
            local_activity = arrays["local_activity"]
            domain_share = arrays["domain_share"]
        else:
            user = pool[self.profile_columns].to_numpy(dtype=float)
            local_activity = pool["profile_local_activity"].to_numpy(dtype=float)
            domain_share = pool["profile_domain_share"].to_numpy(dtype=float)
        activity_interaction = local_activity[:, None] * content[None, :]
        domain_interaction = domain_share[:, None] * content[None, :]
        return np.column_stack(
            [
                np.repeat(shared[None, :], len(pool), axis=0),
                user,
                activity_interaction,
                domain_interaction,
            ]
        )

    def form_team(
        self,
        policy: LinearUCB,
        item_id: object,
        community: str,
        label: int,
        *,
        explore: bool,
        update: bool,
    ) -> tuple[dict, pd.DataFrame]:
        pool, _, users = self._prepared_pool(community)
        if pool.empty:
            return {
                "item_id": item_id,
                "community": community,
                "label": int(label),
                "score": 0.0,
                "team_size": 0,
                "candidate_pool_size": 0,
                "eligible": False,
            }, pd.DataFrame()
        # Selection happens before any vote is generated. The simulator is
        # queried only for the chosen arms below, so unselected rewards remain
        # genuinely hidden from the policy.
        features = self._features(item_id, community, pool)
        selection_score, mean, uncertainty = policy.score(features, explore=explore)
        k = min(self.team_size, len(pool))
        # Stable ordering makes ties reproducible across machines.
        selected = np.argsort(-selection_score, kind="stable")[:k]
        selected_users = [users[index] for index in selected]
        selected_simulated_scores = self.simulator.predict_scores(
            item_id, community, selected_users
        )
        selected_votes = np.where(selected_simulated_scores >= 0, 1, -1)
        # The UCB bonus decides whom to query, but it must not also inflate a
        # selected member's aggregation weight. Trust weights therefore use
        # posterior means only, following the original softmax formulation.
        trust_logits = mean[selected] / self.softmax_temperature
        trust_logits -= trust_logits.max()
        trust_weights = np.exp(trust_logits)
        trust_weights /= trust_weights.sum()
        panel_score = float(trust_weights @ selected_simulated_scores)
        panel_prediction = 1 if panel_score >= 0 else -1
        panel_reward = 1.0 if panel_prediction == int(label) else -1.0
        item = {
            "item_id": item_id,
            "community": community,
            "label": int(label),
            "score": panel_score,
            "binary_vote_score": float(selected_votes.mean()),
            "panel_prediction_at_zero": int(panel_prediction),
            "panel_reward": float(panel_reward),
            "team_size": int(k),
            "candidate_pool_size": int(len(pool)),
            "eligible": True,
        }
        members = with_columns(
            pool.iloc[selected][["username", "community", *self.profile_columns]],
            {"item_id": item_id},
            {
                "label": int(label),
                "simulated_vote_score": selected_simulated_scores,
                "simulated_vote": selected_votes,
                "linucb_selection_score": selection_score[selected],
                "linucb_mean_reward": mean[selected],
                "linucb_uncertainty": uncertainty[selected],
                "trust_weight": trust_weights,
                "selected_rank": np.arange(1, k + 1),
                "exploration_enabled": bool(explore),
                "panel_score": panel_score,
                "panel_reward": panel_reward,
            },
        )
        if update:
            # Normalized weights give each panel one unit of update mass.
            rewards = np.full(k, panel_reward, dtype=float)
            policy.update(
                features[selected],
                rewards,
                sample_weights=trust_weights,
            )
            state = self.community_state[str(community)]
            state["n"] += 1
            state["positive"] += int(label == 1)
        return item, members


def run_cases(
    cases: pd.DataFrame,
    environment: TeamFormationEnvironment,
    policy: LinearUCB,
    *,
    explore: bool,
    update: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run one item at a time, never using the historical voter set."""
    ordered = (
        cases.sort_values("timestamp")
        .drop_duplicates("item_id")
        [["item_id", "community", "label", "timestamp"]]
    )
    item_rows = []
    member_rows = []
    for row in ordered.itertuples(index=False):
        item, members = environment.form_team(
            policy,
            row.item_id,
            str(row.community),
            int(row.label),
            explore=explore,
            update=update,
        )
        item["timestamp"] = row.timestamp
        item_rows.append(item)
        if not members.empty:
            member_rows.append(members)
    membership = pd.concat(member_rows, ignore_index=True) if member_rows else pd.DataFrame()
    return pd.DataFrame(item_rows), membership


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
    parser.add_argument("--ucb-alpha", type=float, default=0.5)
    parser.add_argument("--ridge", type=float, default=1.0)
    parser.add_argument("--softmax-temperature", type=float, default=1.0)
    args = parser.parse_args()
    if args.team_size < 1 or args.max_candidates < 0 or args.min_user_history < 1:
        raise ValueError("team-size>=1, max-candidates>=0, min-user-history>=1 required")
    if args.softmax_temperature <= 0:
        raise ValueError("softmax-temperature must be positive")

    votes_dir = Path(args.votes_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train, val, test = load_and_enrich_splits(votes_dir, args.causal_features)
    simulator_train, policy_train, split_report = split_train_items(
        train, args.simulator_fraction
    )

    encoder = PostContextEncoder(
        args.embedding_dir, dimensions=args.embedding_dimensions
    ).fit(simulator_train)
    simulator = HierarchicalVoteSimulator(encoder).fit(simulator_train.reset_index(drop=True))
    candidate_pools, candidate_profiles = build_candidate_profiles(
        simulator_train, simulator, args.min_user_history
    )
    environment = TeamFormationEnvironment(
        simulator,
        candidate_pools,
        team_size=args.team_size,
        max_candidates=args.max_candidates,
        softmax_temperature=args.softmax_temperature,
    )
    environment.seed_community_state(simulator_train)
    policy = LinearUCB(
        environment.feature_dimension,
        alpha=args.ucb_alpha,
        ridge=args.ridge,
    )

    train_scores, train_members = run_cases(
        policy_train, environment, policy, explore=True, update=True
    )
    val_scores, val_members = run_cases(val, environment, policy, explore=False, update=False)
    test_scores, test_members = run_cases(test, environment, policy, explore=False, update=False)

    val_eligible = val_scores[val_scores["eligible"]].copy()
    test_eligible = test_scores[test_scores["eligible"]].copy()
    if val_eligible.empty or test_eligible.empty:
        raise ValueError("No eligible validation/test cases for known community candidate pools")
    thresholds, global_threshold = calibrate_thresholds_per_community(val_eligible)
    predictions = {
        "policy_train": apply_thresholds(
            train_scores[train_scores["eligible"]], thresholds, global_threshold
        ),
        "val": apply_thresholds(val_eligible, thresholds, global_threshold),
        "test": apply_thresholds(test_eligible, thresholds, global_threshold),
    }
    results = {
        name: classification_metrics(frame) for name, frame in predictions.items()
    }
    results["coverage"] = {
        "val_eligible_items": int(val_eligible.shape[0]),
        "val_total_items": int(val_scores.shape[0]),
        "test_eligible_items": int(test_eligible.shape[0]),
        "test_total_items": int(test_scores.shape[0]),
    }
    results["simulator_fidelity_on_observed_votes"] = {
        "val": simulator.evaluate_observed(val.reset_index(drop=True)),
        "test": simulator.evaluate_observed(test.reset_index(drop=True)),
    }
    results["protocol"] = {
        **split_report,
        "candidate_universe": "users observed in simulator prefix within the case community",
        "historical_case_voters_used_for_selection": False,
        "votes_used_for_panel_decision": "frozen per-user simulator outputs",
        "bandit_feedback": (
            "one selected-panel reward distributed across selected arms with "
            "normalized posterior-mean softmax weights"
        ),
        "reward_unit": (
            "panel decision from the trust-weighted continuous simulated score"
        ),
        "selection_score": "posterior mean plus LinUCB uncertainty bonus",
        "aggregation_weight": "softmax of selected-arm posterior means only",
        "policy_train_exploration": True,
        "validation_test_exploration": False,
        "validation_test_updates": False,
        "team_size": int(args.team_size),
        "max_candidates": int(args.max_candidates),
        "min_user_history": int(args.min_user_history),
        "n_candidate_profiles": int(len(candidate_profiles)),
        "n_candidate_communities": int(len(candidate_pools)),
        "linucb_updates": int(policy.n_updates),
        "linucb_total_update_weight": float(policy.total_update_weight),
        "ucb_alpha": float(args.ucb_alpha),
        "ridge": float(args.ridge),
        "softmax_temperature": float(args.softmax_temperature),
        "simulated_estimand": True,
        "warning": (
            "Downstream policy metrics measure a simulated team-formation world; "
            "simulator fidelity must be reported alongside them."
        ),
    }
    (out_dir / "metrics.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    predictions["val"].to_parquet(out_dir / "val_predictions.parquet", index=False)
    predictions["test"].to_parquet(out_dir / "test_predictions.parquet", index=False)
    train_members.to_parquet(out_dir / "policy_train_team_membership.parquet", index=False)
    val_members.to_parquet(out_dir / "val_team_membership.parquet", index=False)
    test_members.to_parquet(out_dir / "test_team_membership.parquet", index=False)
    candidate_profiles.to_parquet(out_dir / "candidate_profiles.parquet", index=False)
    print(
        f"Bandit: test macro-F1={results['test']['macro_f1']:.4f} | "
        f"saved={out_dir}"
    )


if __name__ == "__main__":
    main()
