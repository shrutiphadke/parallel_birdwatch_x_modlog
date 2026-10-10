"""Small, auditable LinUCB implementation for panel-member selection."""

from __future__ import annotations

import numpy as np


class LinearUCB:
    """Contextual LinUCB over case-user arms with selected-arm feedback."""

    def __init__(self, dimension: int, alpha: float = 0.5, ridge: float = 1.0):
        if dimension < 1 or ridge <= 0 or alpha < 0:
            raise ValueError("dimension>=1, ridge>0, and alpha>=0 are required")
        self.dimension = int(dimension)
        self.alpha = float(alpha)
        self.ridge = float(ridge)
        self.a_inv = np.eye(self.dimension, dtype=float) / self.ridge
        self.b = np.zeros(self.dimension, dtype=float)
        self.n_updates = 0
        self.total_update_weight = 0.0

    @property
    def theta(self) -> np.ndarray:
        return self.a_inv @ self.b

    def score(
        self,
        features: np.ndarray,
        *,
        explore: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return selection score, posterior mean, and standard uncertainty."""
        matrix = np.atleast_2d(np.asarray(features, dtype=float))
        if matrix.shape[1] != self.dimension:
            raise ValueError(
                f"Expected {self.dimension} features, received {matrix.shape[1]}"
            )
        means = matrix @ self.theta
        # Row-wise x^T A^-1 x; an unoptimized 3-operand einsum is far slower.
        variances = np.sum((matrix @ self.a_inv) * matrix, axis=1)
        uncertainty = np.sqrt(np.maximum(variances, 0.0))
        bonus = self.alpha * uncertainty if explore else 0.0
        return means + bonus, means, uncertainty

    def update(
        self,
        features: np.ndarray,
        rewards: np.ndarray,
        *,
        sample_weights: np.ndarray | None = None,
    ) -> None:
        """Apply optionally weighted rank-one updates to selected arms only."""
        matrix = np.atleast_2d(np.asarray(features, dtype=float))
        targets = np.atleast_1d(np.asarray(rewards, dtype=float))
        if len(matrix) != len(targets):
            raise ValueError("One reward is required for every selected arm")
        if matrix.shape[1] != self.dimension:
            raise ValueError(
                f"Expected {self.dimension} features, received {matrix.shape[1]}"
            )
        weights = (
            np.ones(len(matrix), dtype=float)
            if sample_weights is None
            else np.atleast_1d(np.asarray(sample_weights, dtype=float))
        )
        if len(weights) != len(matrix):
            raise ValueError("One sample weight is required for every selected arm")
        if not np.all(np.isfinite(weights)) or np.any(weights < 0):
            raise ValueError("Sample weights must be finite and non-negative")

        for vector, reward, weight in zip(matrix, targets, weights):
            if weight == 0:
                continue
            # A weighted observation contributes w * x x^T to A and
            # w * r * x to b. Sherman-Morrison therefore uses sqrt(w) * x.
            weighted_vector = np.sqrt(weight) * vector
            projected = self.a_inv @ weighted_vector
            denominator = 1.0 + float(weighted_vector @ projected)
            self.a_inv -= np.outer(projected, projected) / denominator
            self.b += float(weight * reward) * vector
            self.n_updates += 1
            self.total_update_weight += float(weight)
