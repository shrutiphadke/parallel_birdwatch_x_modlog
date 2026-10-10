"""Optimize four team-composition objectives with NSGA-II."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from src.methods.team_formation_v2.role_utils import FamiliarityKey, crowding_distance


def nondominated_sort(objectives: list[tuple[float, ...]]) -> list[list[int]]:
    """Return ordinary Pareto fronts over all four specified objectives."""
    if not objectives:
        return []
    values = np.asarray(objectives, dtype=float)
    # matrix[i, j] is True when i is >= j on every objective and > on one.
    matrix = (values[:, None, :] >= values[None, :, :]).all(axis=2) & (
        values[:, None, :] > values[None, :, :]
    ).any(axis=2)
    # Sets are filled in ascending order, exactly as the pairwise loop did, so
    # their iteration order (and therefore front order) is unchanged.
    dominates = [set(np.flatnonzero(row).tolist()) for row in matrix]
    dominated_count = matrix.sum(axis=0).tolist()
    fronts = [[index for index, count in enumerate(dominated_count) if count == 0]]
    rank = 0
    while fronts[rank]:
        following = []
        for left_index in fronts[rank]:
            for right_index in dominates[left_index]:
                dominated_count[right_index] -= 1
                if dominated_count[right_index] == 0:
                    following.append(right_index)
        rank += 1
        fronts.append(following)
    return fronts[:-1]


_PAIR_CACHE: dict[int, tuple[dict, dict[str, list]]] = {}


def _community_pairs(
    familiarity: dict[FamiliarityKey, float],
) -> dict[str, list[tuple[tuple[str, str], float]]]:
    """Group familiarity pairs by community once per familiarity table."""
    cached = _PAIR_CACHE.get(id(familiarity))
    if cached is not None and cached[0] is familiarity:
        return cached[1]
    grouped: dict[str, list[tuple[tuple[str, str], float]]] = {}
    for (community, left, right), count in familiarity.items():
        grouped.setdefault(community, []).append(((left, right), count))
    _PAIR_CACHE.clear()
    _PAIR_CACHE[id(familiarity)] = (familiarity, grouped)
    return grouped


class CommunityNSGA2Optimizer:
    """Reuse community-static pairwise objectives for every incoming case."""

    def __init__(
        self,
        candidates: pd.DataFrame,
        ideology: dict[str, np.ndarray],
        familiarity: dict[FamiliarityKey, float],
        familiarity_scales: dict[str, float],
        team_size: int,
        population_size: int,
        generations: int,
    ):
        self.candidates = candidates.reset_index(drop=True)
        self.n_candidates = len(self.candidates)
        self.k = min(int(team_size), self.n_candidates)
        self.population_size = int(population_size)
        self.generations = int(generations)
        self.community = str(self.candidates.iloc[0]["community"])
        users = self.candidates["username"].astype(str).tolist()
        self.usernames = users

        ideology_dimension = len(next(iter(ideology.values()))) if ideology else 1
        vectors = np.vstack(
            [ideology.get(user, np.zeros(ideology_dimension)) for user in users]
        )
        known = np.asarray([user in ideology for user in users], dtype=bool)
        self.ideology_matrix = np.linalg.norm(
            vectors[:, None, :] - vectors[None, :, :],
            axis=2,
        )
        self.ideology_matrix *= known[:, None] & known[None, :]

        self.familiarity_matrix = np.zeros(
            (self.n_candidates, self.n_candidates),
            dtype=float,
        )
        scale = max(float(familiarity_scales.get(self.community, 1.0)), 1.0)
        # Visit only this community's observed pairs instead of all n^2 pairs;
        # unobserved pairs keep familiarity 0.
        position = {user: index for index, user in enumerate(users)}
        for (left_user, right_user), count in _community_pairs(familiarity).get(
            self.community, ()
        ):
            left = position.get(left_user)
            right = position.get(right_user)
            if left is None or right is None or left == right:
                continue
            value = count / scale
            self.familiarity_matrix[left, right] = value
            self.familiarity_matrix[right, left] = value

        # Relative bands avoid imposing a universal agreement threshold on
        # communities with different moderator base rates.
        alignment = self.candidates["profile_alignment_local"].to_numpy(dtype=float)
        order = np.argsort(alignment, kind="stable")
        high_count = max(1, math.ceil(self.n_candidates / 3))
        mid_count = max(1, math.ceil(self.n_candidates / 3)) if self.n_candidates > 1 else 0
        self.high_alignment = np.zeros(self.n_candidates, dtype=bool)
        self.mid_alignment = np.zeros(self.n_candidates, dtype=bool)
        self.high_alignment[order[-high_count:]] = True
        mid_stop = self.n_candidates - high_count
        mid_start = max(0, mid_stop - mid_count)
        self.mid_alignment[order[mid_start:mid_stop]] = True

    def compose(
        self,
        expertise_values: np.ndarray,
        seed: int,
    ) -> tuple[tuple[int, ...], tuple[float, ...], int, float]:
        """Optimize all four objectives and return a closest-to-ideal compromise."""
        expertise = np.asarray(expertise_values, dtype=float)
        if len(expertise) != self.n_candidates:
            raise ValueError("Expertise length does not match the candidate pool")
        objective_cache: dict[tuple[int, ...], tuple[float, ...]] = {}
        upper_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

        def evaluate(indices: tuple[int, ...]) -> tuple[float, ...]:
            cached = objective_cache.get(indices)
            if cached is not None:
                return cached
            selected = np.asarray(indices, dtype=int)
            upper = upper_cache.get(len(selected))
            if upper is None:
                upper = upper_cache[len(selected)] = np.triu_indices(len(selected), k=1)
            # Same elements, in the same order, as matrix[np.ix_(s, s)][upper];
            # sum()/n is exactly what ndarray.mean() computes.
            left, right = selected[upper[0]], selected[upper[1]]
            n_pairs = len(left)
            ideology = self.ideology_matrix[left, right]
            familiar = self.familiarity_matrix[left, right]
            objectives = (
                float(ideology.sum() / n_pairs) if n_pairs else 0.0,
                float(expertise[selected].sum() / len(selected)),
                0.5
                * (
                    float(self.high_alignment[selected].any())
                    + float(self.mid_alignment[selected].any())
                ),
                float(familiar.sum() / n_pairs) if n_pairs else 0.0,
            )
            objective_cache[indices] = objectives
            return objectives

        if self.k == self.n_candidates:
            selected = tuple(range(self.n_candidates))
            return selected, evaluate(selected), 1, 0.0

        rng = np.random.RandomState(seed)
        target_size = min(self.population_size, math.comb(self.n_candidates, self.k))

        def random_individual() -> tuple[int, ...]:
            return tuple(
                sorted(rng.choice(self.n_candidates, size=self.k, replace=False).tolist())
            )

        population = list(dict.fromkeys(random_individual() for _ in range(target_size * 2)))
        while len(population) < target_size:
            population.append(random_individual())
            population = list(dict.fromkeys(population))
        population = population[:target_size]

        def score_all(items: list[tuple[int, ...]]) -> list[tuple[float, ...]]:
            return [evaluate(item) for item in items]

        for _ in range(self.generations):
            objectives = score_all(population)
            fronts = nondominated_sort(objectives)
            rank = {
                index: number
                for number, front in enumerate(fronts)
                for index in front
            }
            crowd = {
                index: value
                for front in fronts
                for index, value in crowding_distance(front, objectives).items()
            }

            def tournament() -> tuple[int, ...]:
                left, right = rng.choice(len(population), size=2, replace=False)
                left_key = (rank[left], -crowd[left])
                right_key = (rank[right], -crowd[right])
                return population[left if left_key < right_key else right]

            offspring = []
            while len(offspring) < target_size:
                left, right = tournament(), tournament()
                crossover_pool = sorted(set(left) | set(right))
                child = tuple(
                    sorted(rng.choice(crossover_pool, size=self.k, replace=False).tolist())
                )
                if rng.rand() < 0.25:
                    selected_set = set(child)
                    members = sorted(selected_set)
                    remove = int(rng.choice(members))
                    # Same draw as rng.choice(sorted(all - child)) without
                    # materializing the whole candidate range: map the index
                    # to the matching non-member in ascending order.
                    added = int(rng.randint(0, self.n_candidates - len(members)))
                    for member in members:
                        if member <= added:
                            added += 1
                    selected_set.remove(remove)
                    selected_set.add(added)
                    child = tuple(sorted(selected_set))
                offspring.append(child)

            combined = list(dict.fromkeys(population + offspring))
            combined_objectives = score_all(combined)
            next_population = []
            for front in nondominated_sort(combined_objectives):
                if len(next_population) + len(front) <= target_size:
                    next_population.extend(combined[index] for index in front)
                    continue
                distances = crowding_distance(front, combined_objectives)
                ordered = sorted(front, key=lambda index: distances[index], reverse=True)
                slots = target_size - len(next_population)
                next_population.extend(combined[index] for index in ordered[:slots])
                break
            population = next_population

        objectives = score_all(population)
        first_front = nondominated_sort(objectives)[0]
        values = np.asarray(objectives, dtype=float)
        low = values.min(axis=0)
        span = values.max(axis=0) - low
        normalized = np.divide(
            values - low,
            span,
            out=np.ones_like(values),
            where=span > 0,
        )
        ideal_distances = np.sqrt(np.square(1.0 - normalized).mean(axis=1))
        best = min(
            first_front,
            key=lambda index: (
                float(ideal_distances[index]),
                -float(normalized[index].sum()),
                population[index],
            ),
        )
        return (
            population[best],
            objectives[best],
            len(first_front),
            float(ideal_distances[best]),
        )
