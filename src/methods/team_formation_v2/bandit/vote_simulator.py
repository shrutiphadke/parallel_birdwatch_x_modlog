"""Leakage-safe content/subreddit vote simulation with per-user heads."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, roc_auc_score


class PostContextEncoder:
    """Encode ``<post content | subreddit>`` from frozen post embeddings.

    PCA and the subreddit vocabulary are fitted on the simulator-training
    prefix only. Unknown communities receive an all-zero community block.
    """

    def __init__(self, embedding_dir: Path, dimensions: int = 16):
        self.embedding_dir = Path(embedding_dir)
        self.requested_dimensions = int(dimensions)
        self.post_ids: dict[str, int] = {}
        self.embeddings: np.ndarray | None = None
        self.reducer: PCA | None = None
        self.scale: np.ndarray | None = None
        self.communities: list[str] = []
        self.community_index: dict[str, int] = {}
        # Per-item PCA projections are deterministic, so memoize them: every
        # pipeline re-encodes the same cases thousands of times.
        self._content_cache: dict[str, tuple[np.ndarray, bool]] = {}

    def fit(self, votes: pd.DataFrame) -> "PostContextEncoder":
        ids_path = self.embedding_dir / "post_ids.json"
        embeddings_path = self.embedding_dir / "post_embeddings.npy"
        if not ids_path.exists() or not embeddings_path.exists():
            raise FileNotFoundError(
                "Post embeddings are required for counterfactual vote simulation. "
                f"Expected {ids_path} and {embeddings_path}."
            )
        ids = json.loads(ids_path.read_text(encoding="utf-8"))
        self.post_ids = {str(item_id): index for index, item_id in enumerate(ids)}
        self.embeddings = np.load(embeddings_path, mmap_mode="r")
        if len(ids) != len(self.embeddings):
            raise ValueError("post_ids.json and post_embeddings.npy have different lengths")

        train_ids = [
            str(item_id)
            for item_id in votes["item_id"].drop_duplicates()
            if str(item_id) in self.post_ids
        ]
        if len(train_ids) < 2:
            raise ValueError("At least two embedded simulator-training posts are required")
        matrix = np.asarray(
            self.embeddings[[self.post_ids[item_id] for item_id in train_ids]],
            dtype=np.float32,
        )
        n_components = min(
            self.requested_dimensions,
            matrix.shape[0] - 1,
            matrix.shape[1],
        )
        self.reducer = PCA(
            n_components=n_components,
            svd_solver="randomized",
            random_state=10,
        ).fit(matrix)
        reduced = self.reducer.transform(matrix)
        self.scale = np.maximum(np.std(reduced, axis=0), 1e-6)
        self.communities = sorted(votes["community"].dropna().astype(str).unique())
        self.community_index = {
            community: index for index, community in enumerate(self.communities)
        }
        self._content_cache = {}
        return self

    @property
    def content_dimensions(self) -> int:
        return 0 if self.reducer is None else int(self.reducer.n_components_)

    @property
    def output_dimensions(self) -> int:
        return 1 + self.content_dimensions + len(self.communities)

    def content_vector(self, item_id: object) -> tuple[np.ndarray, bool]:
        if self.reducer is None or self.embeddings is None or self.scale is None:
            raise RuntimeError("PostContextEncoder.fit must be called before transform")
        key = str(item_id)
        cached = self._content_cache.get(key)
        if cached is not None:
            return cached
        index = self.post_ids.get(key)
        if index is None:
            vector, found = np.zeros(self.content_dimensions, dtype=float), False
        else:
            raw = np.asarray(self.embeddings[index], dtype=np.float32).reshape(1, -1)
            reduced = self.reducer.transform(raw)[0]
            vector, found = np.tanh(reduced / (3.0 * self.scale)).astype(float), True
        # Shared across callers, so guard against accidental in-place edits.
        vector.setflags(write=False)
        self._content_cache[key] = (vector, found)
        return vector, found

    def transform(self, item_ids: pd.Series, communities: pd.Series) -> np.ndarray:
        item_ids = list(item_ids)
        communities = list(communities)
        content_dimensions = self.content_dimensions
        rows = np.zeros((len(item_ids), self.output_dimensions), dtype=float)
        rows[:, 0] = 1.0
        for row, (item_id, community) in enumerate(zip(item_ids, communities)):
            rows[row, 1 : 1 + content_dimensions] = self.content_vector(item_id)[0]
            community_index = self.community_index.get(str(community))
            if community_index is not None:
                rows[row, 1 + content_dimensions + community_index] = 1.0
        return rows

    def one(self, item_id: object, community: object) -> np.ndarray:
        return self.transform([item_id], [community])[0]

    def coverage(self, votes: pd.DataFrame) -> float:
        item_ids = votes["item_id"].astype(str).drop_duplicates()
        return float(item_ids.isin(self.post_ids).mean()) if len(item_ids) else 0.0


class HierarchicalVoteSimulator:
    """Global ridge vote model with regularized per-user residual heads."""

    def __init__(
        self,
        encoder: PostContextEncoder,
        global_ridge: float = 5.0,
        user_ridge: float = 20.0,
        min_user_votes: int = 2,
    ):
        self.encoder = encoder
        self.global_ridge = float(global_ridge)
        self.user_ridge = float(user_ridge)
        self.min_user_votes = int(min_user_votes)
        self.global_coef: np.ndarray | None = None
        self.user_offsets: dict[str, np.ndarray] = {}
        self.user_counts: dict[str, int] = {}
        # Row ``i`` holds global_coef + offset for user i; the final row is the
        # global fallback for users without a personal head.
        self._coefficients: np.ndarray | None = None
        self._coefficient_rows: dict[str, int] = {}

    @staticmethod
    def _solve(matrix: np.ndarray, target: np.ndarray, ridge: float) -> np.ndarray:
        penalty = ridge * np.eye(matrix.shape[1], dtype=float)
        # Do not penalize the intercept as strongly as the other coefficients.
        penalty[0, 0] = max(ridge * 0.1, 1e-6)
        return np.linalg.solve(matrix.T @ matrix + penalty, matrix.T @ target)

    def fit(self, votes: pd.DataFrame) -> "HierarchicalVoteSimulator":
        design = self.encoder.transform(votes["item_id"], votes["community"])
        target = votes["vote"].to_numpy(dtype=float)
        self.global_coef = self._solve(design, target, self.global_ridge)
        global_prediction = design @ self.global_coef
        for username, indices in votes.groupby("username", sort=False).indices.items():
            idx = np.asarray(indices, dtype=int)
            self.user_counts[str(username)] = len(idx)
            if len(idx) < self.min_user_votes:
                continue
            residual = target[idx] - global_prediction[idx]
            # Stronger shrinkage for users with little personal history.
            ridge = self.user_ridge * max(self.min_user_votes / len(idx), 0.25)
            # Columns that are zero for every one of this user's votes (mostly
            # other communities' one-hot slots) decouple from the ridge system
            # and solve to exactly zero, so only the active block is solved.
            # The intercept column is always active, keeping its penalty at 0.
            user_design = design[idx]
            active = np.flatnonzero(np.any(user_design != 0.0, axis=0))
            offset = np.zeros(design.shape[1], dtype=float)
            offset[active] = self._solve(user_design[:, active], residual, ridge)
            self.user_offsets[str(username)] = offset
        users = list(self.user_offsets)
        self._coefficient_rows = {user: row for row, user in enumerate(users)}
        self._coefficients = np.vstack(
            [self.global_coef + self.user_offsets[user] for user in users]
            + [self.global_coef]
        )
        return self

    def _coefficient_indices(self, usernames) -> np.ndarray:
        fallback = len(self._coefficient_rows)
        rows = self._coefficient_rows
        return np.fromiter(
            (rows.get(str(username), fallback) for username in usernames),
            dtype=np.intp,
        )

    def predict_scores(
        self,
        item_id: object,
        community: object,
        usernames: list[str],
    ) -> np.ndarray:
        if self.global_coef is None:
            raise RuntimeError("HierarchicalVoteSimulator.fit must be called first")
        context = self.encoder.one(item_id, community)
        scores = self._coefficients[self._coefficient_indices(usernames)] @ context
        return np.clip(scores, -1.0, 1.0)

    def evaluate_observed(self, votes: pd.DataFrame) -> dict:
        """Measure simulator fidelity only where a historical vote is observed."""
        if self._coefficients is None:
            raise RuntimeError("HierarchicalVoteSimulator.fit must be called first")
        contexts = self.encoder.transform(votes["item_id"], votes["community"])
        coefficients = self._coefficients[self._coefficient_indices(votes["username"])]
        scores = np.clip(np.einsum("ij,ij->i", contexts, coefficients), -1.0, 1.0)
        target = votes["vote"].to_numpy(dtype=int)
        prediction = np.where(scores >= 0, 1, -1)
        result = {
            "macro_f1": float(f1_score(target, prediction, average="macro")),
            "accuracy": float(np.mean(target == prediction)),
            "n_votes": int(len(votes)),
            "post_embedding_coverage": self.encoder.coverage(votes),
        }
        try:
            result["roc_auc"] = float(roc_auc_score((target == 1).astype(int), scores))
        except ValueError:
            result["roc_auc"] = None
        return result
