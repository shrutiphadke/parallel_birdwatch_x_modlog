from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import logging
import math
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.layout import REDDIT  # noqa: E402
from src.utils.threshold_search import best_threshold  # noqa: E402

DEFAULT_SPLIT_DIR = REDDIT.splits / "intersection"
DEFAULT_OUTPUT_DIR = REDDIT.results / "intersection" / "cn_official_compat"
SCORING_ROOT = REPO_ROOT / "external" / "community-notes" / "scoring" / "src"

CRH = "CRH"
CRNH = "CRNH"
NMR = "NMR"


if str(SCORING_ROOT) not in sys.path:
    sys.path.insert(0, str(SCORING_ROOT))

from scoring import constants as c
from scoring.matrix_factorization.matrix_factorization import MatrixFactorization
from scoring.matrix_factorization.normalized_loss import (
    NormalizedLossHyperparameters,
)
from scoring.matrix_factorization.pseudo_raters import PseudoRatersRunner
from scoring.matrix_factorization.wandb_utils import wandb as scorer_wandb
from scoring.quasi_clique_detection import QuasiCliqueDetection

logging.disable(logging.INFO)


QUANTILE_RANGE = np.array(
    [
        -0.92,
        -0.65,
        -0.61,
        -0.58,
        -0.55,
        -0.52,
        -0.49,
        -0.46,
        -0.43,
        -0.41,
        -0.38,
        -0.35,
        -0.33,
        -0.30,
        -0.28,
        -0.25,
        -0.22,
        -0.20,
        -0.17,
        -0.14,
        -0.12,
        -0.09,
        -0.07,
        -0.04,
        -0.01,
        0.01,
        0.04,
        0.07,
        0.09,
        0.12,
        0.14,
        0.17,
        0.20,
        0.22,
        0.25,
        0.28,
        0.30,
        0.33,
        0.35,
        0.38,
        0.41,
        0.43,
        0.46,
        0.49,
        0.52,
        0.55,
        0.58,
        0.61,
        0.65,
        0.92,
    ],
    dtype=np.float32,
)


@dataclass(frozen=True)
class GaussianParams:
    bandwidth: float = 0.1
    smoothing_weight: float = 0.4
    smoothing_value: float = 0.35
    adaptive_weight_base: int | None = 9
    prior_factor: bool = True
    prior_slope: float = 0.5
    neg_weight: float = 1.75
    min_prior: float | None = 0.2
    weight_lim: float = 1e-9
    somewhat_helpful_value: float = 0.7
    clip_lower: float = 0.05
    clip_upper: float = 0.8
    flip: bool = False
    normalize: bool = True


GAUSSIAN_CRH_PARAMS = GaussianParams()
GAUSSIAN_CRNH_PARAMS = GaussianParams(
    smoothing_weight=5.0,
    smoothing_value=0.35,
    bandwidth=0.2,
    min_prior=0.3,
    adaptive_weight_base=9,
    neg_weight=1.25,
    flip=True,
    clip_upper=0.65,
    clip_lower=0.25,
    normalize=False,
    weight_lim=5e-4,
)


@dataclass
class FitResult:
    name: str
    mf: MatrixFactorization
    note_params: pd.DataFrame
    rater_params: pd.DataFrame
    global_intercept: float
    seed: int
    loss: float


def load_split_data(
    split_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_parquet(split_dir / "train_votes.parquet")
    val = pd.read_parquet(split_dir / "val_votes.parquet")
    test = pd.read_parquet(split_dir / "test_votes.parquet")
    all_votes = pd.concat([train, val, test], ignore_index=True)
    return train, val, test, all_votes


def build_cn_inputs(
    all_votes: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[int, str]]:
    # Start at 1 because the current Gaussian empirical-prior implementation reserves 0.
    item_ids = pd.unique(all_votes["item_id"])
    str_to_int = {item_id: idx for idx, item_id in enumerate(item_ids, start=1)}
    int_to_str = {idx: item_id for item_id, idx in str_to_int.items()}

    ratings = pd.DataFrame(
        {
            c.noteIdKey: all_votes["item_id"].map(str_to_int).astype(np.int64),
            c.raterParticipantIdKey: all_votes["username"].astype(str),
            # Canonical CN binary mapping: Not Helpful=0, Helpful=1.
            c.helpfulNumKey: all_votes["vote"].eq(1).astype(np.float32),
        }
    )

    item_meta = (
        all_votes.sort_values("timestamp")
        .drop_duplicates("item_id", keep="first")[["item_id", "community", "timestamp", "label"]]
        .copy()
    )
    item_meta[c.noteIdKey] = item_meta["item_id"].map(str_to_int).astype(np.int64)
    return ratings, item_meta, int_to_str


def official_minimum_filter(
    ratings: pd.DataFrame,
    min_ratings_per_rater: int = 10,
    min_raters_per_note: int = 5,
) -> pd.DataFrame:
    """Mirror process_data.filter_ratings: note, rater, then note once more."""

    note_counts = ratings[c.noteIdKey].value_counts()
    out = ratings[ratings[c.noteIdKey].isin(note_counts[note_counts >= min_raters_per_note].index)]
    rater_counts = out[c.raterParticipantIdKey].value_counts()
    out = out[
        out[c.raterParticipantIdKey].isin(
            rater_counts[rater_counts >= min_ratings_per_rater].index
        )
    ]
    note_counts = out[c.noteIdKey].value_counts()
    out = out[out[c.noteIdKey].isin(note_counts[note_counts >= min_raters_per_note].index)]
    return out.reset_index(drop=True)


def fit_matrix_factorization(
    name: str,
    ratings: pd.DataFrame,
    seed: int,
    max_loss: float,
    expanded_consensus: bool = False,
    max_attempts: int = 5,
) -> FitResult:
    attempts: list[FitResult] = []
    for offset in range(max_attempts):
        attempt_seed = seed + offset
        kwargs: dict[str, Any] = {
            "initLearningRate": 0.2,
            "noInitLearningRate": 1.0,
            "convergence": 1e-7,
            "numFactors": 1,
            "useGlobalIntercept": True,
            "log": False,
            "userFactorLambda": 0.03,
            "noteFactorLambda": 0.03,
            "userInterceptLambda": 0.15,
            "noteInterceptLambda": 0.15,
            "globalInterceptLambda": 0.15,
            "featureCols": [c.noteIdKey, c.raterParticipantIdKey],
            "labelCol": c.helpfulNumKey,
            "seed": attempt_seed,
        }
        if expanded_consensus:
            # Current Group-14 trial configuration from official run_scoring.py.
            kwargs.update(
                {
                    "initLearningRate": 0.02,
                    "noInitLearningRate": 0.02,
                    "noteInterceptLambda": 0.03 * 30,
                    "userInterceptLambda": 0.03 * 5,
                    "globalInterceptLambda": 0.03 * 5,
                    "noteFactorLambda": 0.03 / 3,
                    "userFactorLambda": 0.03 / 4,
                    "diamondLambda": 0.03 * 25,
                    "normalizedLossHyperparameters": NormalizedLossHyperparameters(
                        globalSignNorm=True,
                        noteSignAlpha=None,
                        noteNormExp=0,
                        raterNormExp=-0.25,
                    ),
                }
            )

        mf = MatrixFactorization(**kwargs)
        note_params, rater_params, global_intercept = mf.run_mf(ratings)
        loss = float(mf.get_final_train_error() or math.inf)
        result = FitResult(
            name=name,
            mf=mf,
            note_params=note_params,
            rater_params=rater_params,
            global_intercept=float(global_intercept or 0.0),
            seed=attempt_seed,
            loss=loss,
        )
        attempts.append(result)
        print(f"[{name}] seed={attempt_seed} loss={loss:.6f}")
        if loss <= max_loss:
            return result

    best = min(attempts, key=lambda candidate: candidate.loss)
    print(
        f"[{name}] no attempt met max_loss={max_loss:.3f}; "
        f"using seed={best.seed}, loss={best.loss:.6f}"
    )
    return best


def add_item_ids(note_params: pd.DataFrame, int_to_str: dict[int, str]) -> pd.DataFrame:
    out = note_params.copy()
    out["item_id"] = out[c.noteIdKey].map(int_to_str)
    return out


def compute_rating_stats(ratings: pd.DataFrame, rater_params: pd.DataFrame) -> pd.DataFrame:
    merged = ratings.merge(
        rater_params[[c.raterParticipantIdKey, c.internalRaterFactor1Key]],
        on=c.raterParticipantIdKey,
        how="inner",
        validate="many_to_one",
    )
    merged["factor_side"] = np.where(merged[c.internalRaterFactor1Key] < 0, "neg", "pos")
    grouped = (
        merged.groupby([c.noteIdKey, "factor_side"])[c.helpfulNumKey]
        .agg(["size", "sum", "mean"])
        .unstack(fill_value=0)
    )
    grouped.columns = [f"{stat}_{side}" for stat, side in grouped.columns]
    required = [
        "size_neg",
        "size_pos",
        "sum_neg",
        "sum_pos",
        "mean_neg",
        "mean_pos",
    ]
    grouped = grouped.reindex(columns=required, fill_value=0).reset_index()
    grouped["rating_count"] = grouped["size_neg"] + grouped["size_pos"]
    grouped["net_neg"] = 2 * grouped["sum_neg"] - grouped["size_neg"]
    grouped["net_pos"] = 2 * grouped["sum_pos"] - grouped["size_pos"]
    grouped["net_min_helpful"] = grouped[["net_neg", "net_pos"]].min(axis=1).clip(lower=0)
    grouped["net_min_ratio"] = grouped["net_min_helpful"] / grouped["rating_count"]
    grouped["min_sign_count"] = grouped[["size_neg", "size_pos"]].min(axis=1)
    grouped["net_helpful_pass"] = (grouped["net_min_helpful"] >= 10) | (
        (grouped["net_min_helpful"] >= 4) & (grouped["net_min_ratio"] >= 0.05)
    )
    grouped["ratio_crnh"] = (
        (grouped["rating_count"] >= 5)
        & (grouped["min_sign_count"] >= 3)
        & (grouped["mean_neg"] <= 0.4)
        & (grouped["mean_pos"] <= 0.4)
        & (((grouped["mean_neg"] + grouped["mean_pos"]) / 2) <= 0.3)
    )
    return grouped


def compute_pseudo_bounds(fit: FitResult, ratings: pd.DataFrame) -> pd.DataFrame:
    print("[pseudo-raters] computing UCBs")
    runner = PseudoRatersRunner(
        ratings=ratings,
        noteParams=fit.note_params,
        raterParams=fit.rater_params,
        globalBias=fit.global_intercept,
        mfRanker=fit.mf,
        log=False,
        # This disables only redundant exact-equality assertions; it does not change scoring.
        checkParamsSame=False,
    )
    try:
        return runner.compute_note_parameter_confidence_bounds_with_pseudo_raters()
    finally:
        scorer_wandb.finish()


def assign_statuses(
    name: str,
    note_params: pd.DataFrame,
    rating_stats: pd.DataFrame,
    int_to_str: dict[int, str],
    *,
    crh_threshold: float,
    factor_threshold: float,
    crnh_intercept: float,
    crnh_factor_multiplier: float,
    require_net_helpful: bool,
    ucb_threshold: float | None = None,
    no_correlated_intercepts: pd.DataFrame | None = None,
    no_correlated_threshold: float | None = None,
) -> pd.DataFrame:
    out = note_params.merge(rating_stats, on=c.noteIdKey, how="left", validate="one_to_one")
    out["item_id"] = out[c.noteIdKey].map(int_to_str)
    out["status"] = NMR

    crh_mask = (
        (out["rating_count"] >= 5)
        & (out[c.internalNoteInterceptKey] >= crh_threshold)
        & (out[c.internalNoteFactor1Key].abs() < factor_threshold)
    )
    if require_net_helpful:
        crh_mask &= out["net_helpful_pass"].fillna(False)

    if no_correlated_intercepts is not None and no_correlated_threshold is not None:
        safeguard = no_correlated_intercepts[[c.noteIdKey, c.internalNoteInterceptKey]].rename(
            columns={c.internalNoteInterceptKey: "no_correlated_intercept"}
        )
        out = out.merge(safeguard, on=c.noteIdKey, how="left")
        crh_mask &= out["no_correlated_intercept"].fillna(-np.inf) >= no_correlated_threshold
    else:
        out["no_correlated_intercept"] = np.nan

    crnh_mask = (out["rating_count"] >= 5) & (
        out[c.internalNoteInterceptKey]
        <= crnh_intercept + crnh_factor_multiplier * out[c.internalNoteFactor1Key].abs()
    )
    crnh_mask |= out["ratio_crnh"].fillna(False)

    if ucb_threshold is not None and "internalNoteIntercept_max" in out.columns:
        crnh_mask |= (out["rating_count"] >= 5) & (
            out["internalNoteIntercept_max"] <= ucb_threshold
        )

    out.loc[crh_mask, "status"] = CRH
    # Current scoring applies additional CRNH rules after general CRH.
    out.loc[crnh_mask, "status"] = CRNH
    out["variant"] = name
    return out


def detect_quasi_cliques(
    ratings: pd.DataFrame, item_meta: pd.DataFrame
) -> tuple[pd.DataFrame, set[str]]:
    notes = item_meta[[c.noteIdKey, "item_id", "timestamp"]].copy()
    notes[c.tweetIdKey] = notes["item_id"].astype(str)
    notes[c.classificationKey] = c.notesSaysTweetIsMisleadingKey
    notes[c.createdAtMillisKey] = (notes["timestamp"].astype("int64") // 1_000_000).astype(
        np.int64
    )
    notes = notes[[c.noteIdKey, c.tweetIdKey, c.classificationKey, c.createdAtMillisKey]]
    detector = QuasiCliqueDetection()
    cliques = detector.get_quasi_cliques(notes, ratings)
    if cliques.empty:
        return cliques, set()
    correlated = set(cliques[c.raterParticipantIdKey].astype(str))
    return cliques, correlated


def compute_empirical_prior_df(
    ratings_with_factors: pd.DataFrame,
    bin_width: float = 0.2,
    min_neg: int = 10,
    min_pos: int = 10,
    max_neg: int = 50,
    max_pos: int = 50,
) -> pd.DataFrame:
    work = ratings_with_factors.loc[
        ratings_with_factors[c.helpfulNumKey] != 0.5,
        [c.noteIdKey, c.internalRaterFactor1Key, c.helpfulNumKey],
    ].copy()
    work["bin"] = np.floor((work[c.internalRaterFactor1Key] + 1.0) / bin_width).astype(int)
    work["bin"] = work["bin"].clip(0, int(np.floor(2.0 / bin_width)))
    sign_counts = (
        work.assign(
            neg=work[c.internalRaterFactor1Key] < 0,
            pos=work[c.internalRaterFactor1Key] > 0,
        )
        .groupby(c.noteIdKey)[["neg", "pos"]]
        .sum()
        .astype(int)
    )
    candidates = sign_counts[
        (sign_counts["neg"] >= min_neg)
        & (sign_counts["pos"] >= min_pos)
        & (sign_counts["neg"] <= max_neg)
        & (sign_counts["pos"] <= max_pos)
    ].index
    counts = (
        work[work[c.noteIdKey].isin(candidates)]
        .groupby([c.noteIdKey, "bin", c.helpfulNumKey])
        .size()
        .reset_index(name="count")
    )
    if counts.empty:
        return pd.DataFrame()
    return counts.pivot_table(
        index=c.noteIdKey,
        columns=["bin", c.helpfulNumKey],
        values="count",
        fill_value=0,
    ).astype(np.int32)


def lookup_empirical_priors(
    empirical_totals: pd.DataFrame,
    ratings_with_factors: pd.DataFrame,
    all_curves: pd.DataFrame,
    seed: int,
    strict_min: int = 7,
) -> pd.DataFrame | None:
    if empirical_totals.empty:
        return None
    query = compute_empirical_prior_df(
        ratings_with_factors,
        min_neg=3,
        max_neg=np.inf,
        min_pos=3,
        max_pos=np.inf,
    )
    if query.empty:
        return None
    query = query.reindex(columns=empirical_totals.columns, fill_value=0).astype(np.int32)
    candidate_matrix = empirical_totals.values
    candidate_ids = empirical_totals.index.to_numpy()
    query_ids = query.index.to_numpy()
    query_matrix = query.values
    required_min = np.where(query_matrix <= strict_min, query_matrix, strict_min)
    required_min = np.where(query_matrix > 0, required_min, 0)
    is_superset = np.ones((len(query_ids), len(candidate_ids)), dtype=bool)
    for feature_idx in range(candidate_matrix.shape[1]):
        is_superset &= (
            candidate_matrix[:, feature_idx][None, :] >= required_min[:, feature_idx][:, None]
        )
    candidate_positions = {note_id: idx for idx, note_id in enumerate(candidate_ids)}
    for row_idx, note_id in enumerate(query_ids):
        if note_id in candidate_positions:
            is_superset[row_idx, candidate_positions[note_id]] = False

    rng = random.Random(seed)
    sampled: list[list[int]] = []
    for row in is_superset:
        matches = candidate_ids[row].tolist()
        if len(matches) >= 30:
            sampled.append(rng.choices(matches, k=30))
        else:
            sampled.append(matches + [0] * (30 - len(matches)))
    sampled_df = pd.DataFrame(sampled, index=query_ids)

    unique_notes = np.unique(ratings_with_factors[c.noteIdKey].values)
    curve_df = pd.DataFrame(all_curves, index=unique_notes)
    curve_with_nan = pd.concat(
        [curve_df, pd.DataFrame(np.nan, columns=curve_df.columns, index=[0])]
    )
    selected = curve_with_nan.loc[sampled_df.values.ravel()]
    averaged = selected.groupby(np.arange(len(selected)) // 30).mean()
    averaged.index = query_ids
    return averaged


def _gaussian_kernel(distances: np.ndarray, params: GaussianParams) -> np.ndarray:
    norm_const = 1.0 / (params.bandwidth * np.sqrt(2 * np.pi))
    return norm_const * np.exp(-0.5 * (distances / params.bandwidth) ** 2)


def gaussian_curves(
    ratings_with_factors: pd.DataFrame,
    params: GaussianParams,
    empirical_priors: pd.DataFrame | None = None,
) -> pd.DataFrame:
    quantile_cols = [f"{value:5.3f}" for value in QUANTILE_RANGE]
    note_ids = ratings_with_factors[c.noteIdKey].to_numpy()
    helpful = ratings_with_factors[c.helpfulNumKey].to_numpy(dtype=np.float32).copy()
    helpful[helpful == 0.5] = params.somewhat_helpful_value
    if params.flip:
        helpful = 1 - helpful
    factors = ratings_with_factors[c.internalRaterFactor1Key].to_numpy(dtype=np.float32)

    unfavorable_value = 0 if not params.flip else 1
    unfavorable_summary = (
        ratings_with_factors.loc[ratings_with_factors[c.helpfulNumKey] == unfavorable_value]
        .groupby(c.noteIdKey)[c.internalRaterFactor1Key]
        .agg("mean")
        .reset_index(name="mean")
    )
    all_counts = (
        ratings_with_factors.groupby(c.noteIdKey)[c.internalRaterFactor1Key]
        .count()
        .reset_index(name="count")
    )
    unfavorable_summary = unfavorable_summary.merge(all_counts, on=c.noteIdKey)
    distances = ratings_with_factors.loc[
        ratings_with_factors[c.helpfulNumKey] == unfavorable_value,
        [c.noteIdKey, c.internalRaterFactor1Key],
    ].merge(unfavorable_summary, on=c.noteIdKey)
    distances["dist"] = np.abs(distances[c.internalRaterFactor1Key] - distances["mean"])
    distances[c.internalRaterFactor1Key] = 1 / (1 + np.exp(-200 * (distances["dist"] - 0.28 / 2)))
    unexpected = distances.groupby(c.noteIdKey)[c.internalRaterFactor1Key].sum().reset_index()
    variability = unfavorable_summary.merge(unexpected, how="left", on=c.noteIdKey).fillna(0)
    variability["pct"] = variability[c.internalRaterFactor1Key] / variability["count"]
    variability = (variability.set_index(c.noteIdKey)["pct"] * (4 if params.flip else 3) + 1).clip(
        upper=(4.0 if params.flip else 2.0)
    )

    unique_notes, note_indices = np.unique(note_ids, return_inverse=True)
    rating_counts = np.bincount(note_indices, minlength=len(unique_notes)).astype(np.float32)
    distances_to_centers = np.abs(factors[:, None] - QUANTILE_RANGE[None, :])
    weights = _gaussian_kernel(distances_to_centers, params).astype(np.float32)
    if params.normalize:
        weights = weights / weights.sum(axis=1, keepdims=True)

    total_base_weights = np.column_stack(
        [
            np.bincount(note_indices, weights=weights[:, idx], minlength=len(unique_notes))
            for idx in range(len(QUANTILE_RANGE))
        ]
    ).astype(np.float32)
    insufficient_data = (total_base_weights < params.weight_lim).any(axis=1)

    negative_mask = helpful == 0
    if negative_mask.any():
        variability = variability.reindex(unique_notes, fill_value=1.0)
        factors_by_rating = np.where(
            np.isnan(variability.reindex(note_ids).to_numpy()),
            1.0,
            variability.reindex(note_ids).to_numpy(),
        )
        weights[negative_mask] *= factors_by_rating[negative_mask, None]
    weights[negative_mask] *= params.neg_weight

    total_weights = np.column_stack(
        [
            np.bincount(note_indices, weights=weights[:, idx], minlength=len(unique_notes))
            for idx in range(len(QUANTILE_RANGE))
        ]
    ).astype(np.float32)
    weighted_values = weights * helpful[:, None]
    total_values = np.column_stack(
        [
            np.bincount(
                note_indices,
                weights=weighted_values[:, idx],
                minlength=len(unique_notes),
            )
            for idx in range(len(QUANTILE_RANGE))
        ]
    ).astype(np.float32)

    original_helpful = ratings_with_factors[c.helpfulNumKey].to_numpy(dtype=np.float32)
    if params.flip:
        original_helpful = 1 - original_helpful
    adjusted = (original_helpful - 0.5) * 2 * factors
    note_factors = (
        np.bincount(note_indices, weights=adjusted, minlength=len(unique_notes)) / rating_counts
    )

    if params.prior_factor:
        prior = (
            params.prior_slope * note_factors[:, None] * QUANTILE_RANGE[None, :]
            + params.smoothing_value
        )
        smoothing_values = np.minimum(prior, params.smoothing_value).astype(np.float32)
        if params.min_prior is not None:
            smoothing_values = np.maximum(smoothing_values, params.min_prior).astype(np.float32)
    else:
        smoothing_values = np.full(
            (len(unique_notes), len(QUANTILE_RANGE)),
            params.smoothing_value,
            dtype=np.float32,
        )

    if empirical_priors is not None and not empirical_priors.empty:
        present = np.where(np.isin(unique_notes, empirical_priors.index))[0]
        smoothing_values[present] = empirical_priors.loc[
            unique_notes[present], quantile_cols
        ].to_numpy()

    sign_counts = (
        ratings_with_factors.assign(
            neg=ratings_with_factors[c.internalRaterFactor1Key] < 0,
            pos=ratings_with_factors[c.internalRaterFactor1Key] > 0,
        )
        .groupby(c.noteIdKey)[["neg", "pos"]]
        .sum()
    )
    insufficient_ids = sign_counts[(sign_counts["neg"] < 3) | (sign_counts["pos"] < 3)].index
    smoothing_values[np.isin(unique_notes, insufficient_ids)] = 0.1

    if params.adaptive_weight_base is not None:
        smoothing_weights = (
            np.log(np.maximum(rating_counts, params.adaptive_weight_base))
            / np.log(params.adaptive_weight_base)
            * params.smoothing_weight
        ).astype(np.float32)
    else:
        smoothing_weights = np.full(len(unique_notes), params.smoothing_weight, dtype=np.float32)

    denominator = total_weights + smoothing_weights[:, None]
    numerator = total_values + smoothing_values * smoothing_weights[:, None]
    means = np.full_like(numerator, params.clip_lower, dtype=np.float32)
    means = np.divide(numerator, denominator, out=means, where=denominator != 0)
    means = np.clip(means, params.clip_lower, params.clip_upper)
    means[insufficient_data, :] = -1
    curves = pd.DataFrame(means, columns=quantile_cols, index=unique_notes)
    curves[c.internalNoteFactor1Key] = note_factors
    return curves


def gaussian_scores(
    ratings: pd.DataFrame,
    core_rater_params: pd.DataFrame,
    seed: int,
) -> tuple[pd.DataFrame, int]:
    ratings_with_factors = ratings.merge(
        core_rater_params[[c.raterParticipantIdKey, c.internalRaterFactor1Key]],
        on=c.raterParticipantIdKey,
        how="inner",
        validate="many_to_one",
    )
    crh_curves = gaussian_curves(ratings_with_factors, GAUSSIAN_CRH_PARAMS)
    empirical_totals = compute_empirical_prior_df(ratings_with_factors)
    first_crnh_curves = gaussian_curves(ratings_with_factors, GAUSSIAN_CRNH_PARAMS)
    empirical_priors = lookup_empirical_priors(
        empirical_totals,
        ratings_with_factors,
        first_crnh_curves,
        seed=seed,
    )
    crnh_curves = gaussian_curves(
        ratings_with_factors,
        GAUSSIAN_CRNH_PARAMS,
        empirical_priors=empirical_priors,
    )
    quantile_cols = [f"{value:5.3f}" for value in QUANTILE_RANGE]
    with np.errstate(invalid="ignore", divide="ignore"):
        crh_mean_log = np.mean(np.log(crh_curves[quantile_cols].to_numpy()), axis=1)
        crnh_mean_log = np.mean(np.log(crnh_curves[quantile_cols].to_numpy()), axis=1)
    # Current upstream maps a curve containing the -1 insufficient-data sentinel to
    # -inf in log space and therefore to a neutral score of zero.
    crh_mean_log = np.where(np.isnan(crh_mean_log), -np.inf, crh_mean_log)
    crnh_mean_log = np.where(np.isnan(crnh_mean_log), -np.inf, crnh_mean_log)
    crh_score = np.exp(crh_mean_log)
    crnh_score = -np.exp(crnh_mean_log)
    combined = np.where(np.abs(crh_score) < np.abs(crnh_score), crnh_score, crh_score)
    out = pd.DataFrame(
        {
            c.noteIdKey: crh_curves.index.to_numpy(),
            c.internalNoteInterceptKey: combined,
            c.internalNoteFactor1Key: crh_curves[c.internalNoteFactor1Key].to_numpy(),
            "gaussian_crh_score": crh_score,
            "gaussian_crnh_score": crnh_score,
        }
    )
    return out, len(empirical_totals)


def calibrate_binary_threshold(
    note_params: pd.DataFrame,
    val: pd.DataFrame,
) -> float:
    labels = val.drop_duplicates("item_id")[["item_id", "label"]]
    scores = labels.merge(
        note_params[["item_id", c.internalNoteInterceptKey]],
        on="item_id",
        how="left",
    )
    threshold, _ = best_threshold(
        scores["label"].to_numpy(),
        scores[c.internalNoteInterceptKey].to_numpy(dtype=float),
        np.linspace(-2.0, 2.0, 400),
        default=0.0,
    )
    return threshold


def variant_metrics(
    name: str,
    test_items: pd.DataFrame,
    status_col: str,
    score_col: str | None,
    baseline_prediction: pd.Series,
) -> dict[str, Any]:
    decided = test_items[status_col].isin([CRH, CRNH])
    y_true = test_items.loc[decided, "label"].to_numpy()
    y_pred = test_items.loc[decided, status_col].map({CRH: 1, CRNH: -1}).to_numpy()
    result: dict[str, Any] = {
        "variant": name,
        "n_test": len(test_items),
        "n_crh": int((test_items[status_col] == CRH).sum()),
        "n_crnh": int((test_items[status_col] == CRNH).sum()),
        "n_nmr": int((test_items[status_col] == NMR).sum()),
        "coverage": float(decided.mean()),
        "abstention_rate": float(1 - decided.mean()),
        "selective_accuracy": float("nan"),
        "selective_macro_f1": float("nan"),
        "selective_f1_pos": float("nan"),
        "selective_f1_neg": float("nan"),
        "auc_all_items": float("nan"),
        "agreement_with_current_on_decided": float("nan"),
    }
    if decided.any():
        result["selective_accuracy"] = float(accuracy_score(y_true, y_pred))
        _, _, f1_macro, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=[1, -1], average="macro", zero_division=0
        )
        _, _, f1_each, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=[1, -1], average=None, zero_division=0
        )
        result["selective_macro_f1"] = float(f1_macro)
        result["selective_f1_pos"] = float(f1_each[0])
        result["selective_f1_neg"] = float(f1_each[1])
        result["agreement_with_current_on_decided"] = float(
            (y_pred == baseline_prediction.loc[decided].to_numpy()).mean()
        )
    if score_col is not None and test_items[score_col].notna().all():
        try:
            result["auc_all_items"] = float(
                roc_auc_score(test_items["label"].eq(1).astype(int), test_items[score_col])
            )
        except ValueError:
            pass
    return result


def binary_metrics(
    name: str,
    labels: pd.Series,
    predictions: pd.Series,
    scores: pd.Series,
) -> dict[str, Any]:
    _, _, macro_f1, _ = precision_recall_fscore_support(
        labels, predictions, labels=[1, -1], average="macro", zero_division=0
    )
    _, _, f1_each, _ = precision_recall_fscore_support(
        labels, predictions, labels=[1, -1], average=None, zero_division=0
    )
    return {
        "variant": name,
        "n_test": len(labels),
        "n_crh": int((predictions == 1).sum()),
        "n_crnh": int((predictions == -1).sum()),
        "n_nmr": 0,
        "coverage": 1.0,
        "abstention_rate": 0.0,
        "selective_accuracy": float(accuracy_score(labels, predictions)),
        "selective_macro_f1": float(macro_f1),
        "selective_f1_pos": float(f1_each[0]),
        "selective_f1_neg": float(f1_each[1]),
        "auc_all_items": float(roc_auc_score(labels.eq(1).astype(int), scores)),
        "agreement_with_current_on_decided": float("nan"),
    }


def combine_coverage_models(
    core: pd.DataFrame,
    expanded: pd.DataFrame,
    gaussian: pd.DataFrame,
) -> pd.DataFrame:
    out = core[[c.noteIdKey, "item_id", "status"]].rename(columns={"status": "core_status"})
    out = out.merge(
        core[[c.noteIdKey, c.internalNoteInterceptKey]].rename(
            columns={c.internalNoteInterceptKey: "core_intercept"}
        ),
        on=c.noteIdKey,
    )
    out = out.merge(
        expanded[[c.noteIdKey, "status"]].rename(columns={"status": "expanded_status"}),
        on=c.noteIdKey,
        how="left",
    )
    out = out.merge(
        gaussian[[c.noteIdKey, "status", c.internalNoteInterceptKey]].rename(
            columns={
                "status": "gaussian_status",
                c.internalNoteInterceptKey: "gaussian_score",
            }
        ),
        on=c.noteIdKey,
        how="left",
    )
    out["status"] = out["core_status"]
    expanded_promote = (
        out["status"].eq(NMR) & out["expanded_status"].eq(CRH) & (out["core_intercept"] > 0.25)
    )
    out.loc[expanded_promote, "status"] = CRH
    # Current meta-scoring runs the general Gaussian coverage scorer after group models.
    gaussian_update = out["status"].eq(NMR) & out["gaussian_status"].isin([CRH, CRNH])
    out.loc[gaussian_update, "status"] = out.loc[gaussian_update, "gaussian_status"]
    return out


def run(
    seed: int,
    split_dir: Path,
    output_dir: Path,
    baseline_item_scores: Path | None = None,
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    _train, val, test, all_votes = load_split_data(split_dir.resolve())
    ratings, item_meta, int_to_str = build_cn_inputs(all_votes)
    filtered = official_minimum_filter(ratings)
    print(
        f"[data] {len(filtered)} ratings, {filtered[c.noteIdKey].nunique()} items, "
        f"{filtered[c.raterParticipantIdKey].nunique()} raters"
    )

    core_fit = fit_matrix_factorization(
        "core_01", filtered, seed=seed, max_loss=0.09, expanded_consensus=False
    )
    core_fit.note_params = add_item_ids(core_fit.note_params, int_to_str)
    rating_stats = compute_rating_stats(filtered, core_fit.rater_params)

    pseudo_note_params = compute_pseudo_bounds(core_fit, filtered)
    core_with_pseudo = core_fit.note_params.merge(
        pseudo_note_params[
            [
                c.noteIdKey,
                "internalNoteIntercept_min",
                "internalNoteIntercept_median",
                "internalNoteIntercept_max",
            ]
        ],
        on=c.noteIdKey,
        how="left",
        validate="one_to_one",
    )

    print("[quasi-clique] detecting dense correlated rater subgraphs")
    cliques, correlated_raters = detect_quasi_cliques(filtered, item_meta)
    if correlated_raters:
        uncorrelated = filtered[~filtered[c.raterParticipantIdKey].isin(correlated_raters)].copy()
        uncorrelated = official_minimum_filter(uncorrelated)
        no_corr_fit = fit_matrix_factorization(
            "core_no_correlated",
            uncorrelated,
            seed=seed,
            max_loss=0.09,
            expanded_consensus=False,
        )
        no_corr_params = no_corr_fit.note_params
    else:
        no_corr_params = core_fit.note_params
    print(f"[quasi-clique] correlated raters={len(correlated_raters)}")

    core_status = assign_statuses(
        "core_supported",
        core_with_pseudo,
        rating_stats,
        int_to_str,
        crh_threshold=0.40,
        factor_threshold=0.50,
        crnh_intercept=-0.05,
        crnh_factor_multiplier=-0.80,
        require_net_helpful=True,
        ucb_threshold=-0.04,
        no_correlated_intercepts=no_corr_params,
        no_correlated_threshold=0.37,
    )

    print("[gaussian] computing current convolution scores")
    gaussian_params, gaussian_prior_candidates = gaussian_scores(
        filtered, core_fit.rater_params, seed=seed
    )
    gaussian_status = assign_statuses(
        "gaussian_supported",
        gaussian_params,
        rating_stats,
        int_to_str,
        crh_threshold=0.51,
        factor_threshold=1.0,
        crnh_intercept=-0.50,
        crnh_factor_multiplier=0.0,
        require_net_helpful=False,
        ucb_threshold=None,
        no_correlated_intercepts=None,
        no_correlated_threshold=None,
    )

    expanded_fit = fit_matrix_factorization(
        "expanded_consensus_global",
        filtered,
        seed=seed,
        max_loss=0.16,
        expanded_consensus=True,
    )
    expanded_fit.note_params = add_item_ids(expanded_fit.note_params, int_to_str)
    expanded_stats = compute_rating_stats(filtered, expanded_fit.rater_params)
    expanded_status = assign_statuses(
        "expanded_consensus_global",
        expanded_fit.note_params,
        expanded_stats,
        int_to_str,
        crh_threshold=0.15,
        factor_threshold=0.40,
        crnh_intercept=-0.01,
        crnh_factor_multiplier=0.0,
        require_net_helpful=True,
    )

    combined = combine_coverage_models(core_status, expanded_status, gaussian_status)

    # The project's CN run for the same split sits beside this output folder.
    baseline = pd.read_parquet(
        baseline_item_scores or output_dir.parent / "cn" / "item_scores.parquet"
    )
    test_labels = test.drop_duplicates("item_id")[["item_id", "label"]].copy()
    comparison = test_labels.merge(
        baseline[["item_id", "prediction", "i_n", "f_n"]].rename(
            columns={
                "prediction": "current_prediction",
                "i_n": "current_i_n",
                "f_n": "current_f_n",
            }
        ),
        on="item_id",
        how="left",
        validate="one_to_one",
    )

    core_test = comparison[["item_id"]].merge(
        core_status[
            ["item_id", "status", c.internalNoteInterceptKey, c.internalNoteFactor1Key]
        ].rename(
            columns={
                "status": "core_status",
                c.internalNoteInterceptKey: "core_score",
                c.internalNoteFactor1Key: "core_factor",
            }
        ),
        on="item_id",
        how="left",
    )
    comparison = comparison.merge(core_test, on="item_id", how="left")
    comparison = comparison.merge(
        gaussian_status[
            ["item_id", "status", c.internalNoteInterceptKey, c.internalNoteFactor1Key]
        ].rename(
            columns={
                "status": "gaussian_status",
                c.internalNoteInterceptKey: "gaussian_score",
                c.internalNoteFactor1Key: "gaussian_factor",
            }
        ),
        on="item_id",
        how="left",
    )
    comparison = comparison.merge(
        expanded_status[
            ["item_id", "status", c.internalNoteInterceptKey, c.internalNoteFactor1Key]
        ].rename(
            columns={
                "status": "expanded_status",
                c.internalNoteInterceptKey: "expanded_score",
                c.internalNoteFactor1Key: "expanded_factor",
            }
        ),
        on="item_id",
        how="left",
    )
    comparison = comparison.merge(
        combined[["item_id", "status"]].rename(columns={"status": "combined_status"}),
        on="item_id",
        how="left",
    )

    canonical_threshold = calibrate_binary_threshold(core_fit.note_params, val)
    comparison["canonical_binary_prediction"] = np.where(
        comparison["core_score"] >= canonical_threshold, 1, -1
    )

    metrics_rows = [
        binary_metrics(
            "current_saved_binary",
            comparison["label"],
            comparison["current_prediction"],
            comparison["current_i_n"],
        ),
        binary_metrics(
            "canonical_01_val_calibrated_binary",
            comparison["label"],
            comparison["canonical_binary_prediction"],
            comparison["core_score"],
        ),
        variant_metrics(
            "core_supported",
            comparison,
            "core_status",
            "core_score",
            comparison["current_prediction"],
        ),
        variant_metrics(
            "gaussian_supported",
            comparison,
            "gaussian_status",
            "gaussian_score",
            comparison["current_prediction"],
        ),
        variant_metrics(
            "expanded_consensus_global",
            comparison,
            "expanded_status",
            "expanded_score",
            comparison["current_prediction"],
        ),
        variant_metrics(
            "core_plus_expanded_plus_gaussian",
            comparison,
            "combined_status",
            "gaussian_score",
            comparison["current_prediction"],
        ),
    ]
    metrics = pd.DataFrame(metrics_rows)

    core_before_net = (
        (core_status[c.internalNoteInterceptKey] >= 0.40)
        & (core_status[c.internalNoteFactor1Key].abs() < 0.50)
        & (core_status["rating_count"] >= 5)
    )
    summary: dict[str, Any] = {
        "seed_requested": seed,
        "n_source_ratings": len(ratings),
        "n_filtered_ratings": len(filtered),
        "n_filtered_items": int(filtered[c.noteIdKey].nunique()),
        "n_filtered_raters": int(filtered[c.raterParticipantIdKey].nunique()),
        "core_seed": core_fit.seed,
        "core_loss": core_fit.loss,
        "core_global_intercept": core_fit.global_intercept,
        "expanded_seed": expanded_fit.seed,
        "expanded_loss": expanded_fit.loss,
        "expanded_global_intercept": expanded_fit.global_intercept,
        "canonical_binary_threshold": canonical_threshold,
        "gaussian_prior_candidates": gaussian_prior_candidates,
        "n_quasi_cliques": int(cliques[c.quasiCliqueValueKey].nunique())
        if not cliques.empty
        else 0,
        "n_correlated_raters": len(correlated_raters),
        "core_crh_before_net": int(core_before_net.sum()),
        "core_crh_after_net": int((core_before_net & core_status["net_helpful_pass"]).sum()),
        "net_helpful_pass_items": int(core_status["net_helpful_pass"].sum()),
        "components_not_implemented": {
            "post_selection_similarity": "missing individual rating timestamps, note authors and peer-note structure",
            "contributor_helpfulness_filter": "missing valid-rating timing, note authors and status histories",
            "tag_filters_and_diligence": "missing rating-level reason tags",
            "population_sample_filter": "missing rating source/random-sampling indicator",
            "x_group_models": "missing X geography/language/rollout membership",
            "pcrh_and_status_stabilization": "missing rating-arrival and repeated status trajectories",
            "high_volume_7_28_day_guard": "available timestamp is a post-creation proxy, not rating time",
        },
    }

    comparison.to_parquet(output_dir / "item_scores.parquet", index=False)
    payload = {
        "split": split_dir.name,
        "seed": seed,
        "summary": summary,
        "variants": {
            row["variant"]: {
                key: None if pd.isna(value) else value
                for key, value in row.items()
                if key != "variant"
            }
            for row in metrics.to_dict("records")
        },
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("\n" + metrics.to_string(index=False))
    print(f"\nOutputs written to: {output_dir}")
    return {"metrics": metrics, "summary": summary, "comparison": comparison}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--baseline-item-scores",
        type=Path,
        default=None,
        help="Project CN item_scores.parquet to compare against "
        "(default: <output-dir>/../cn/item_scores.parquet).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        seed=args.seed,
        split_dir=args.split_dir,
        output_dir=args.output_dir,
        baseline_item_scores=args.baseline_item_scores,
    )
