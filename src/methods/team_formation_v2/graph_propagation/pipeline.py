"""Select simulated panels with co-voting graphs and Personalized PageRank."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import roc_auc_score

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
    fit_user_topic_centroids,
)
from src.methods.team_formation_v2.shared_features import (  # noqa: E402
    DEFAULT_CAUSAL_FEATURES,
    apply_thresholds,
    calibrate_thresholds_per_community,
    centroid_affinities,
    classification_metrics,
    load_and_enrich_splits,
    stable_top_indices,
    with_columns,
)

DEFAULT_VOTES_DIR = "data/splits/reddit/intersection_chronological"
DEFAULT_OUT_DIR = "results/reddit/intersection_chronological/graph_propagation"
DEFAULT_EMBEDDING_DIR = "cache/embeddings"

ATTENTION_POLICIES = {
    "graph_only": {"attention_graph": 1.0},
    "skill_affinity": {
        "attention_graph": 1.0,
        "attention_alignment": 1.0,
        "attention_domain_affinity": 0.75,
        "attention_local_experience": 0.50,
        "attention_tenure": 0.25,
    },
    "full_social": {
        "attention_graph": 1.0,
        "attention_alignment": 1.0,
        "attention_domain_affinity": 0.75,
        "attention_local_experience": 0.50,
        "attention_tenure": 0.25,
        "attention_social_core": 0.50,
    },
    "content_social": {
        "attention_graph": 1.0,
        "attention_alignment": 1.0,
        "attention_case_affinity": 1.0,
        "attention_domain_affinity": 0.50,
        "attention_local_experience": 0.50,
        "attention_social_core": 0.50,
    },
}


class TopicConditionedPPR:
    """Prefix-only agreement graph with a case-specific restart vector."""

    def __init__(
        self,
        reference: pd.DataFrame,
        centroids: dict[str, np.ndarray],
        encoder: PostContextEncoder,
        damping: float = 0.85,
        max_iter: int = 50,
        tolerance: float = 1e-9,
    ):
        self.users = sorted(reference["username"].dropna().astype(str).unique())
        self.user_index = {user: index for index, user in enumerate(self.users)}
        self.encoder = encoder
        self.damping = float(damping)
        self.max_iter = int(max_iter)
        self.tolerance = float(tolerance)
        self._case_cache: dict[str, np.ndarray] = {}

        dimension = encoder.content_dimensions
        self.centroid_matrix = np.vstack(
            [centroids.get(user, np.zeros(dimension, dtype=float)) for user in self.users]
        )
        work = reference.assign(
            _agree=(reference["vote"] == reference["label"]).astype(float)
        )
        work["username"] = work["username"].astype(str)
        agreement = work.groupby("username")["_agree"].agg(["sum", "count"])
        agreement = agreement.reindex(self.users)
        base = float(work["_agree"].mean()) if len(work) else 0.5
        self.alignment = (
            agreement["sum"].to_numpy(dtype=float) + 5.0 * base
        ) / (agreement["count"].to_numpy(dtype=float) + 5.0)
        self.transition = self._build_transition(reference)
        # Precomputed transpose so propagation does not re-transpose per step.
        self.transition_t = self.transition.T.tocsr()
        self.dangling = np.asarray(self.transition.sum(axis=1)).ravel() == 0

    def _build_transition(self, reference: pd.DataFrame) -> sparse.csr_matrix:
        tenure = reference.groupby("username")["tenure_days_at_vote"].median()
        known_tenure = tenure.dropna()
        fallback = float(known_tenure.median()) if len(known_tenure) else 0.0
        # Every unordered pair of distinct voters on the same item, built with a
        # self-join instead of per-item Python combinations.
        voters = reference.loc[reference["item_id"].notna(), ["item_id", "username", "vote"]]
        voters = voters.assign(username=voters["username"].astype(str))
        voters = voters.drop_duplicates(["item_id", "username"])
        voters = voters.assign(
            user=voters["username"].map(self.user_index).to_numpy(dtype=np.int64)
        )[["item_id", "user", "vote"]]
        pairs = voters.merge(voters, on="item_id", suffixes=("_left", "_right"))
        pairs = pairs[pairs["user_left"] < pairs["user_right"]]
        edges = (
            pairs.assign(_agree=(pairs["vote_left"] == pairs["vote_right"]).astype(int))
            .groupby(["user_left", "user_right"], sort=False)["_agree"]
            .agg(["sum", "size"])
        )
        left = edges.index.get_level_values(0).to_numpy(dtype=np.int64)
        right = edges.index.get_level_values(1).to_numpy(dtype=np.int64)
        n_agree = edges["sum"].to_numpy(dtype=float)
        n_common = edges["size"].to_numpy(dtype=float)

        tenure.index = tenure.index.astype(str)
        user_tenure = tenure.reindex(self.users).to_numpy(dtype=float)
        user_tenure = np.where(np.isfinite(user_tenure), user_tenure, fallback)
        log_tenure = np.log1p(np.maximum(user_tenure, 0.0))
        agreement = (n_agree + 1.0) / (n_common + 2.0)
        similarity = np.exp(-np.abs(log_tenure[left] - log_tenure[right]) / 2.0)
        weight = agreement * np.log1p(n_common) * (0.5 + 0.5 * similarity)
        keep = weight > 0
        left, right, weight = left[keep], right[keep], weight[keep]
        rows = np.concatenate([left, right])
        columns = np.concatenate([right, left])
        values = np.concatenate([weight, weight])
        adjacency = sparse.csr_matrix(
            (values, (rows, columns)),
            shape=(len(self.users), len(self.users)),
        )
        degree = np.asarray(adjacency.sum(axis=1)).ravel()
        inverse = np.divide(
            1.0,
            degree,
            out=np.zeros_like(degree),
            where=degree > 0,
        )
        return sparse.diags(inverse) @ adjacency

    def _propagate(self, personalization: np.ndarray) -> np.ndarray:
        return self._propagate_many(personalization[:, None])[:, 0]

    def _propagate_many(self, personalization: np.ndarray) -> np.ndarray:
        """Run power iteration for many restart vectors (one per column).

        Each column stops exactly when its own single-vector iteration would.
        """
        result = np.array(personalization, dtype=float, copy=True)
        active = np.arange(personalization.shape[1])
        scores = result.copy()
        restart = np.asarray(personalization, dtype=float)
        for _ in range(self.max_iter):
            if not len(active):
                break
            current = scores[:, active]
            columns = restart[:, active]
            updated = self.damping * (self.transition_t @ current)
            updated += self.damping * current[self.dangling].sum(axis=0) * columns
            updated += (1.0 - self.damping) * columns
            converged = np.abs(updated - current).sum(axis=0) < self.tolerance
            scores[:, active] = updated
            result[:, active[converged]] = updated[:, converged]
            active = active[~converged]
        result[:, active] = scores[:, active]
        return result

    def global_scores(self) -> dict[str, float]:
        personalization = np.maximum(self.alignment - 0.40, 0.01)
        personalization /= personalization.sum()
        return dict(zip(self.users, self._propagate(personalization)))

    def _personalization(self, item_id: object) -> np.ndarray:
        content, _ = self.encoder.content_vector(item_id)
        norm = float(np.linalg.norm(content))
        if norm > 0:
            affinity = np.clip((self.centroid_matrix @ (content / norm) + 1.0) / 2.0, 0, 1)
        else:
            affinity = np.full(len(self.users), 0.5, dtype=float)
        personalization = np.exp(2.0 * affinity) * (0.25 + 0.75 * self.alignment)
        personalization /= personalization.sum()
        return personalization

    def precompute(self, item_ids, batch_size: int = 256) -> None:
        """Propagate many uncached cases at once; results equal the lazy path."""
        pending = list(
            dict.fromkeys(str(item) for item in item_ids if str(item) not in self._case_cache)
        )
        for start in range(0, len(pending), batch_size):
            keys = pending[start : start + batch_size]
            restart = np.column_stack([self._personalization(key) for key in keys])
            propagated = self._propagate_many(restart)
            for column, key in enumerate(keys):
                self._case_cache[key] = self._percentiles(propagated[:, column])

    @staticmethod
    def _percentiles(scores: np.ndarray) -> np.ndarray:
        order = np.argsort(scores, kind="stable")
        percentiles = np.empty(len(scores), dtype=float)
        percentiles[order] = (np.arange(len(scores), dtype=float) + 1.0) / len(scores)
        return percentiles

    def _case_percentiles(self, item_id: object) -> np.ndarray:
        key = str(item_id)
        cached = self._case_cache.get(key)
        if cached is not None:
            return cached
        percentiles = self._percentiles(self._propagate(self._personalization(item_id)))
        self._case_cache[key] = percentiles
        return percentiles

    def scores_for(self, item_id: object, usernames) -> np.ndarray:
        percentiles = self._case_percentiles(item_id)
        indices = np.fromiter(
            (self.user_index.get(str(username), -1) for username in usernames),
            dtype=np.intp,
        )
        return np.where(indices >= 0, percentiles[indices], 0.5)


def _percentile_map(values: dict[str, float]) -> dict[str, float]:
    if not values:
        return {}
    return pd.Series(values, dtype=float).rank(method="average", pct=True).to_dict()


def attach_static_attention(
    profiles: pd.DataFrame,
    ppr_scores: dict[str, float],
) -> pd.DataFrame:
    """Attach graph and frozen candidate-profile attention components."""
    out = profiles.copy()
    ppr_rank = _percentile_map(ppr_scores)
    out["attention_global_graph"] = out["username"].map(ppr_rank).fillna(0.5)
    out["attention_alignment"] = out["profile_alignment_local"]
    out["attention_domain_affinity"] = out["profile_domain_share"]
    out["attention_local_experience"] = out["profile_local_activity"]
    out["attention_tenure"] = out["profile_tenure"]
    out["attention_social_core"] = (
        0.60 * out["profile_social"] + 0.40 * out["profile_local_history"]
    ).clip(0, 1)
    return out


def form_graph_team(
    item_id: object,
    community: str,
    label: int,
    timestamp: object,
    candidates: pd.DataFrame,
    simulator: HierarchicalVoteSimulator,
    centroids: dict[str, np.ndarray],
    topic_ppr: TopicConditionedPPR,
    component_weights: dict[str, float],
    team_size: int,
) -> tuple[dict, pd.DataFrame]:
    """Select top-k graph-attention candidates, then simulate their votes."""
    # Score the whole pool on arrays; only the selected team becomes a frame.
    usernames = candidates["username"].astype(str).tolist()
    content, _ = simulator.encoder.content_vector(item_id)
    if float(np.linalg.norm(content)) > 0:
        case_affinity = np.clip(centroid_affinities(usernames, centroids, content), 0.0, 1.0)
    else:
        case_affinity = np.full(len(usernames), 0.5, dtype=float)
    computed = {
        "attention_case_affinity": case_affinity,
        "attention_graph": topic_ppr.scores_for(item_id, usernames),
    }
    logits = np.zeros(len(usernames), dtype=float)
    for component, weight in component_weights.items():
        values = computed.get(component)
        if values is None:
            values = candidates[component].to_numpy(dtype=float)
        logits += weight * values
    selected = stable_top_indices(logits, usernames, team_size)
    team = with_columns(
        candidates.iloc[selected],
        trailing={
            **{column: values[selected] for column, values in computed.items()},
            "attention_logit": logits[selected],
        },
    )
    users = [usernames[index] for index in selected]
    simulated_scores = simulator.predict_scores(item_id, community, users)
    simulated_votes = np.where(simulated_scores >= 0, 1, -1)
    logits = team["attention_logit"].to_numpy(dtype=float)
    weights = np.exp(logits - logits.max())
    weights /= weights.sum()
    item = {
        "item_id": item_id,
        "community": community,
        "label": int(label),
        "timestamp": timestamp,
        "score": float(weights @ simulated_scores),
        "binary_vote_score": float(weights @ simulated_votes),
        "team_size": int(len(team)),
        "candidate_pool_size": int(len(candidates)),
        "effective_team_size": float(1.0 / np.square(weights).sum()),
        "eligible": True,
    }
    team = with_columns(
        team,
        {"item_id": item_id},
        {
            "label": int(label),
            "simulated_vote_score": simulated_scores,
            "simulated_vote": simulated_votes,
            "attention_weight": weights,
            "selected_rank": np.arange(1, len(team) + 1),
        },
    )
    return item, team


def score_cases(
    cases: pd.DataFrame,
    candidate_pools: dict[str, pd.DataFrame],
    simulator: HierarchicalVoteSimulator,
    centroids: dict[str, np.ndarray],
    topic_ppr: TopicConditionedPPR,
    component_weights: dict[str, float],
    team_size: int,
    max_candidates: int,
    return_members: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply one fixed graph-attention policy to a chronological split."""
    ordered = (
        cases.sort_values("timestamp")
        .drop_duplicates("item_id")
        [["item_id", "community", "label", "timestamp"]]
    )
    item_rows = []
    member_rows = []
    topic_ppr.precompute(
        row.item_id
        for row in ordered.itertuples(index=False)
        if str(row.community) in candidate_pools
    )
    prepared: dict[str, pd.DataFrame] = {}
    for row in ordered.itertuples(index=False):
        pool = candidate_pools.get(str(row.community))
        if pool is None or pool.empty:
            item_rows.append(
                {
                    "item_id": row.item_id,
                    "community": str(row.community),
                    "label": int(row.label),
                    "timestamp": row.timestamp,
                    "score": 0.0,
                    "soft_score": 0.0,
                    "team_size": 0,
                    "candidate_pool_size": 0,
                    "effective_team_size": 0.0,
                    "eligible": False,
                }
            )
            continue
        candidates = prepared.get(str(row.community))
        if candidates is None:
            candidates = pool.head(max_candidates) if max_candidates > 0 else pool
            candidates = prepared[str(row.community)] = candidates.reset_index(drop=True)
        item, members = form_graph_team(
            row.item_id,
            str(row.community),
            int(row.label),
            row.timestamp,
            candidates,
            simulator,
            centroids,
            topic_ppr,
            component_weights,
            team_size,
        )
        item_rows.append(item)
        if return_members:
            member_rows.append(members)
    memberships = (
        pd.concat(member_rows, ignore_index=True) if member_rows else pd.DataFrame()
    )
    return pd.DataFrame(item_rows), memberships


def select_attention_policy(
    policy_train: pd.DataFrame,
    candidate_pools: dict[str, pd.DataFrame],
    simulator: HierarchicalVoteSimulator,
    centroids: dict[str, np.ndarray],
    topic_ppr: TopicConditionedPPR,
    team_size: int,
    max_candidates: int,
) -> tuple[dict, pd.DataFrame]:
    """Select a fixed attention family using only the TRAIN policy suffix."""
    candidate_results = []
    best = None
    best_scores = None
    for name, weights in ATTENTION_POLICIES.items():
        scores, _ = score_cases(
            policy_train,
            candidate_pools,
            simulator,
            centroids,
            topic_ppr,
            weights,
            team_size,
            max_candidates,
        )
        eligible = scores[scores["eligible"]]
        try:
            auc = float(roc_auc_score(eligible["label"], eligible["score"]))
        except ValueError:
            auc = float("-inf")
        result = {"name": name, "weights": weights, "policy_train_auc": auc}
        candidate_results.append(result)
        if best is None or auc > best["policy_train_auc"]:
            best = result
            best_scores = eligible
    if best is None or best_scores is None:
        raise ValueError("Could not select a graph-attention policy")
    return {"selected": best, "candidates": candidate_results}, best_scores


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
    args = parser.parse_args()
    if args.team_size < 1 or args.max_candidates < 0 or args.min_user_history < 1:
        raise ValueError("team-size>=1, max-candidates>=0, min-user-history>=1 required")

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
    candidate_pools, profiles = build_candidate_profiles(
        simulator_train, simulator, args.min_user_history
    )
    centroids = fit_user_topic_centroids(simulator_train, encoder)
    topic_ppr = TopicConditionedPPR(simulator_train, centroids, encoder)
    ppr_scores = topic_ppr.global_scores()
    profiles = attach_static_attention(profiles, ppr_scores)
    candidate_pools = {
        community: group.reset_index(drop=True)
        for community, group in profiles.groupby("community", sort=False)
    }
    policy_search, policy_scores = select_attention_policy(
        policy_train,
        candidate_pools,
        simulator,
        centroids,
        topic_ppr,
        args.team_size,
        args.max_candidates,
    )
    selected = policy_search["selected"]
    print(
        f"Selected {selected['name']} "
        f"(policy-suffix AUC={selected['policy_train_auc']:.4f})"
    )
    val_scores, val_members = score_cases(
        val,
        candidate_pools,
        simulator,
        centroids,
        topic_ppr,
        selected["weights"],
        args.team_size,
        args.max_candidates,
        return_members=True,
    )
    test_scores, test_members = score_cases(
        test,
        candidate_pools,
        simulator,
        centroids,
        topic_ppr,
        selected["weights"],
        args.team_size,
        args.max_candidates,
        return_members=True,
    )
    val_eligible = val_scores[val_scores["eligible"]].copy()
    test_eligible = test_scores[test_scores["eligible"]].copy()
    if val_eligible.empty or test_eligible.empty:
        raise ValueError("No eligible validation/test cases for known candidate pools")
    thresholds, global_threshold = calibrate_thresholds_per_community(val_eligible)
    predictions = {
        "policy_train": apply_thresholds(policy_scores, thresholds, global_threshold),
        "val": apply_thresholds(val_eligible, thresholds, global_threshold),
        "test": apply_thresholds(test_eligible, thresholds, global_threshold),
    }
    results = {
        name: classification_metrics(frame) for name, frame in predictions.items()
    }
    results["coverage"] = {
        "policy_train_eligible_items": int(len(policy_scores)),
        "policy_train_total_items": int(policy_train["item_id"].nunique()),
        "val_eligible_items": int(len(val_eligible)),
        "val_total_items": int(val["item_id"].nunique()),
        "test_eligible_items": int(len(test_eligible)),
        "test_total_items": int(test["item_id"].nunique()),
    }
    results["simulator_fidelity_on_observed_votes"] = {
        "val": simulator.evaluate_observed(val.reset_index(drop=True)),
        "test": simulator.evaluate_observed(test.reset_index(drop=True)),
    }
    results["graph_attention"] = {
        **policy_search,
        "n_graph_users": int(len(ppr_scores)),
        "graph_fit": (
            "simulator-prefix co-votes and tenure similarity with a per-case "
            "content-affinity restart vector"
        ),
        "topic_conditioned_restart": True,
        "aggregation": "softmax attention over continuous selected-panel scores",
    }
    results["protocol"] = {
        **split_report,
        "policy_suffix_use": "attention-family selection only",
        "candidate_universe": "same prefix-only community pools as team_formation_v2.bandit",
        "historical_case_voters_used_for_selection": False,
        "votes_used_for_panel_decision": "selected users' frozen simulator outputs",
        "validation_test_updates": False,
        "team_size": int(args.team_size),
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
    profiles.to_parquet(out_dir / "candidate_graph_profiles.parquet", index=False)
    pd.DataFrame(
        {"username": list(ppr_scores), "ppr": list(ppr_scores.values())}
    ).to_parquet(out_dir / "user_graph_scores.parquet", index=False)
    print(
        f"Graph propagation: test macro-F1={results['test']['macro_f1']:.4f} | "
        f"saved={out_dir}"
    )


if __name__ == "__main__":
    main()
