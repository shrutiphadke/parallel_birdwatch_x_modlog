"""Stack simulated votes from every eligible community participant."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.layout import REDDIT  # noqa: E402
from src.methods.team_formation_v2.bandit.pipeline import (  # noqa: E402
    build_candidate_profiles,
    split_train_items,
)
from src.methods.team_formation_v2.bandit.vote_simulator import (  # noqa: E402
    HierarchicalVoteSimulator,
    PostContextEncoder,
)
from src.methods.team_formation_v2.shared_features import (  # noqa: E402
    DEFAULT_CAUSAL_FEATURES,
    apply_thresholds,
    calibrate_thresholds_per_community,
    classification_metrics,
    complete_with_net_vote_fallback,
    load_and_enrich_splits,
)

DEFAULT_VOTES_DIR = str(REDDIT.splits / "intersection_chronological")
DEFAULT_OUT_DIR = str(REDDIT.results / "intersection_chronological" / "boc_stacking")
DEFAULT_EMBEDDING_DIR = str(REDDIT.embedding_cache)


def feature_names(
    encoder: PostContextEncoder,
    prediction_components: int,
    eligibility_components: int,
) -> list[str]:
    """Return names in exactly the same order as the low-rank design matrix."""
    context = ["context_bias"]
    context.extend(
        f"context_content_{index:02d}" for index in range(encoder.content_dimensions)
    )
    context.extend(f"context_community::{name}" for name in encoder.communities)
    context.append("context_log_candidate_pool")
    predictions = [
        f"predicted_vote_latent_{index:02d}" for index in range(prediction_components)
    ]
    eligibility = [
        f"eligibility_latent_{index:02d}" for index in range(eligibility_components)
    ]
    return [*context, *predictions, *eligibility]


def fit_low_rank_designs(
    raw_designs: dict[str, sparse.csr_matrix],
    context_dimension: int,
    n_users: int,
    prediction_components: int,
    eligibility_components: int,
    fit_indices: np.ndarray | None = None,
) -> tuple[dict[str, np.ndarray], TruncatedSVD, TruncatedSVD]:
    """Fit user-block SVDs on stack-TRAIN only and transform every split."""
    training = raw_designs["stack_train"]
    fit_training = training if fit_indices is None else training[fit_indices]
    prediction_slice = slice(context_dimension, context_dimension + n_users)
    eligibility_slice = slice(context_dimension + n_users, context_dimension + 2 * n_users)
    max_components = min(fit_training.shape[0] - 1, n_users - 1)
    n_prediction = min(prediction_components, max_components)
    n_eligibility = min(eligibility_components, max_components)
    if n_prediction < 1 or n_eligibility < 1:
        raise ValueError("Low-rank BoC needs at least two training cases and user slots")
    prediction_svd = TruncatedSVD(n_components=n_prediction, random_state=10).fit(
        fit_training[:, prediction_slice]
    )
    eligibility_svd = TruncatedSVD(n_components=n_eligibility, random_state=10).fit(
        fit_training[:, eligibility_slice]
    )
    designs = {}
    for name, raw in raw_designs.items():
        designs[name] = np.column_stack(
            [
                raw[:, :context_dimension].toarray(),
                prediction_svd.transform(raw[:, prediction_slice]),
                eligibility_svd.transform(raw[:, eligibility_slice]),
            ]
        )
    return designs, prediction_svd, eligibility_svd


def _parse_grid(raw: str, cast: type) -> list:
    """Parse a comma-separated, positive, deduplicated tuning grid."""
    values = sorted({cast(part.strip()) for part in raw.split(",") if part.strip()})
    if not values or any(value <= 0 for value in values):
        raise ValueError("Tuning grids must contain positive values")
    return values


def _chronological_fit_tune_indices(
    frame: pd.DataFrame,
    tuning_fraction: float,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Reserve the chronologically latest suffix cases for inner tuning."""
    order = np.argsort(frame["timestamp"].to_numpy(), kind="stable")
    cut = min(max(int(len(order) * (1.0 - tuning_fraction)), 2), len(order) - 2)
    fit_indices = order[:cut]
    tune_indices = order[cut:]
    report = {
        "protocol": "chronological_inner_holdout",
        "fit_items": int(len(fit_indices)),
        "tune_items": int(len(tune_indices)),
        "tuning_fraction": float(tuning_fraction),
        "cut_timestamp": float(frame.iloc[fit_indices[-1]]["timestamp"]),
    }
    return fit_indices, tune_indices, report


def _stacking_model(regularization_c: float):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=regularization_c,
            class_weight="balanced",
            max_iter=2_000,
            random_state=10,
            solver="liblinear",
        ),
    )


def select_low_rank_hyperparameters(
    raw_stack_train: sparse.csr_matrix,
    stack_frame: pd.DataFrame,
    context_dimension: int,
    n_users: int,
    prediction_grid: list[int],
    eligibility_grid: list[int],
    regularization_grid: list[float],
    tuning_fraction: float,
) -> dict:
    """Select rank and regularization on a chronological inner holdout."""
    fit_indices, tune_indices, split_report = _chronological_fit_tune_indices(
        stack_frame,
        tuning_fraction,
    )
    labels = stack_frame["label"].to_numpy(dtype=int)
    candidates = []
    best = None
    for prediction_components in prediction_grid:
        for eligibility_components in eligibility_grid:
            designs, prediction_svd, eligibility_svd = fit_low_rank_designs(
                {"stack_train": raw_stack_train},
                context_dimension,
                n_users,
                prediction_components,
                eligibility_components,
                fit_indices=fit_indices,
            )
            design = designs["stack_train"]
            for regularization_c in regularization_grid:
                model = _stacking_model(regularization_c)
                model.fit(design[fit_indices], labels[fit_indices])
                scores = positive_scores(model, design[tune_indices])
                auc = float(roc_auc_score(labels[tune_indices], scores))
                candidate = {
                    "prediction_components": int(prediction_svd.n_components),
                    "eligibility_components": int(eligibility_svd.n_components),
                    "regularization_c": float(regularization_c),
                    "inner_tune_auc": auc,
                }
                candidates.append(candidate)
                if best is None or auc > best["inner_tune_auc"]:
                    best = candidate
    if best is None:
        raise ValueError("Could not select low-rank BoC hyperparameters")
    return {
        "selected": best,
        "candidates": candidates,
        "selection_metric": "ROC-AUC",
        "inner_split": split_report,
    }


def build_boc_design(
    cases: pd.DataFrame,
    candidate_pools: dict[str, pd.DataFrame],
    simulator: HierarchicalVoteSimulator,
    users: list[str],
    max_candidates: int,
    return_user_predictions: bool = False,
) -> tuple[sparse.csr_matrix, pd.DataFrame, pd.DataFrame]:
    """Build `<content|subreddit|pred_u for u in U>` without case-voter data."""
    user_index = {username: index for index, username in enumerate(users)}
    ordered = (
        cases.sort_values("timestamp")
        .drop_duplicates("item_id")
        [["item_id", "community", "label", "timestamp"]]
        .reset_index(drop=True)
    )
    context_dimension = simulator.encoder.output_dimensions + 1
    matrix_rows = []
    matrix_columns = []
    matrix_values = []
    item_rows = []
    prediction_rows = []
    for row_number, case in enumerate(ordered.itertuples(index=False)):
        community = str(case.community)
        context = simulator.encoder.one(case.item_id, community)
        pool = candidate_pools.get(community)
        if pool is None or pool.empty:
            candidate_count = 0
            eligible_users = []
            simulated_scores = np.asarray([], dtype=float)
        else:
            pool = pool.head(max_candidates) if max_candidates > 0 else pool
            eligible_users = pool["username"].astype(str).tolist()
            candidate_count = len(eligible_users)
            simulated_scores = simulator.predict_scores(
                case.item_id,
                community,
                eligible_users,
            )
        context_with_pool = np.concatenate(
            [
                np.asarray(context, dtype=np.float32),
                np.asarray(
                    [min(np.log1p(candidate_count) / 8.0, 1.0)],
                    dtype=np.float32,
                ),
            ]
        )
        context_nonzero = np.flatnonzero(context_with_pool)
        matrix_rows.append(
            np.full(len(context_nonzero), row_number, dtype=np.int32)
        )
        matrix_columns.append(context_nonzero.astype(np.int32))
        matrix_values.append(context_with_pool[context_nonzero])
        if candidate_count:
            indices = np.asarray([user_index[user] for user in eligible_users], dtype=int)
            prediction_offset = context_dimension
            eligibility_offset = context_dimension + len(users)
            matrix_rows.extend(
                [
                    np.full(candidate_count, row_number, dtype=np.int32),
                    np.full(candidate_count, row_number, dtype=np.int32),
                ]
            )
            matrix_columns.extend(
                [
                    (prediction_offset + indices).astype(np.int32),
                    (eligibility_offset + indices).astype(np.int32),
                ]
            )
            matrix_values.extend(
                [
                    np.asarray(simulated_scores, dtype=np.float32),
                    np.ones(candidate_count, dtype=np.float32),
                ]
            )
        item_rows.append(
            {
                "item_id": case.item_id,
                "community": community,
                "label": int(case.label),
                "timestamp": case.timestamp,
                "candidate_pool_size": int(candidate_count),
                "eligible": bool(candidate_count),
                "simulated_majority": (
                    float(np.where(simulated_scores >= 0, 1, -1).mean())
                    if candidate_count
                    else 0.0
                ),
                "simulated_soft_mean": (
                    float(simulated_scores.mean()) if candidate_count else 0.0
                ),
            }
        )
        if return_user_predictions and candidate_count:
            prediction_rows.append(
                pd.DataFrame(
                    {
                        "item_id": case.item_id,
                        "community": community,
                        "label": int(case.label),
                        "username": eligible_users,
                        "simulated_vote_score": simulated_scores,
                        "simulated_vote": np.where(simulated_scores >= 0, 1, -1),
                    }
                )
            )
    user_predictions = (
        pd.concat(prediction_rows, ignore_index=True)
        if prediction_rows
        else pd.DataFrame()
    )
    design = sparse.csr_matrix(
        (
            np.concatenate(matrix_values),
            (np.concatenate(matrix_rows), np.concatenate(matrix_columns)),
        ),
        shape=(len(ordered), context_dimension + 2 * len(users)),
        dtype=np.float32,
    )
    return design, pd.DataFrame(item_rows), user_predictions


def positive_scores(model, design: np.ndarray) -> np.ndarray:
    positive_index = int(np.where(model.classes_ == 1)[0][0])
    return model.predict_proba(design)[:, positive_index]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--votes-dir", default=DEFAULT_VOTES_DIR)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--causal-features", type=Path, default=DEFAULT_CAUSAL_FEATURES)
    parser.add_argument("--embedding-dir", type=Path, default=DEFAULT_EMBEDDING_DIR)
    parser.add_argument("--max-candidates", type=int, default=0)
    parser.add_argument("--min-user-history", type=int, default=2)
    parser.add_argument("--simulator-fraction", type=float, default=0.70)
    parser.add_argument("--embedding-dimensions", type=int, default=16)
    parser.add_argument("--regularization-c", type=float, default=None)
    parser.add_argument("--prediction-components", type=int, default=None)
    parser.add_argument("--eligibility-components", type=int, default=None)
    parser.add_argument("--regularization-grid", default="0.01,0.03,0.1,0.3")
    parser.add_argument("--prediction-components-grid", default="16,32,64")
    parser.add_argument("--eligibility-components-grid", default="4,8,12")
    parser.add_argument("--tuning-fraction", type=float, default=0.25)
    parser.add_argument("--save-user-predictions", action="store_true")
    args = parser.parse_args()
    if (
        args.max_candidates < 0
        or args.min_user_history < 1
        or (args.regularization_c is not None and args.regularization_c <= 0)
        or (args.prediction_components is not None and args.prediction_components < 1)
        or (args.eligibility_components is not None and args.eligibility_components < 1)
        or not 0.1 <= args.tuning_fraction <= 0.5
    ):
        raise ValueError(
            "Positive model parameters and tuning-fraction between 0.1 and 0.5 required"
        )
    prediction_grid = (
        [args.prediction_components]
        if args.prediction_components is not None
        else _parse_grid(args.prediction_components_grid, int)
    )
    eligibility_grid = (
        [args.eligibility_components]
        if args.eligibility_components is not None
        else _parse_grid(args.eligibility_components_grid, int)
    )
    regularization_grid = (
        [args.regularization_c]
        if args.regularization_c is not None
        else _parse_grid(args.regularization_grid, float)
    )

    votes_dir = Path(args.votes_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train, val, test = load_and_enrich_splits(votes_dir, args.causal_features)
    simulator_train, stack_train, split_report = split_train_items(
        train, args.simulator_fraction
    )
    encoder = PostContextEncoder(
        args.embedding_dir, dimensions=args.embedding_dimensions
    ).fit(simulator_train)
    simulator = HierarchicalVoteSimulator(encoder).fit(simulator_train.reset_index(drop=True))
    candidate_pools, profiles = build_candidate_profiles(
        simulator_train,
        simulator,
        args.min_user_history,
    )
    if args.max_candidates > 0:
        candidate_pools = {
            community: group.head(args.max_candidates).reset_index(drop=True)
            for community, group in candidate_pools.items()
        }
    users = sorted(
        {
            str(username)
            for group in candidate_pools.values()
            for username in group["username"]
        }
    )
    if not users:
        raise ValueError("The eligible candidate universe is empty")

    raw_designs = {}
    frames = {}
    user_predictions = {}
    coverage = {}
    for name, cases in (("stack_train", stack_train), ("val", val), ("test", test)):
        raw_designs[name], frames[name], user_predictions[name] = build_boc_design(
            cases,
            candidate_pools,
            simulator,
            users,
            args.max_candidates,
            return_user_predictions=(
                args.save_user_predictions and name in {"val", "test"}
            ),
        )
        eligible = frames[name]["eligible"].to_numpy(dtype=bool)
        coverage[name] = {
            "eligible_items": int(eligible.sum()),
            "total_items": int(len(eligible)),
            "mean_candidates": float(
                frames[name].loc[eligible, "candidate_pool_size"].mean()
            ),
        }
        if not eligible.any():
            raise ValueError(f"{name} contains no cases with eligible candidates")
        raw_designs[name] = raw_designs[name][eligible]
        frames[name] = frames[name].loc[eligible].reset_index(drop=True)

    context_dimension = encoder.output_dimensions + 1
    hyperparameter_search = select_low_rank_hyperparameters(
        raw_designs["stack_train"],
        frames["stack_train"],
        context_dimension,
        len(users),
        prediction_grid,
        eligibility_grid,
        regularization_grid,
        args.tuning_fraction,
    )
    selected = hyperparameter_search["selected"]
    print(
        "Selected "
        f"prediction-rank={selected['prediction_components']}, "
        f"eligibility-rank={selected['eligibility_components']}, "
        f"C={selected['regularization_c']:.4g} "
        f"(inner AUC={selected['inner_tune_auc']:.4f})"
    )
    designs, prediction_svd, eligibility_svd = fit_low_rank_designs(
        raw_designs,
        context_dimension,
        len(users),
        selected["prediction_components"],
        selected["eligibility_components"],
    )

    model = _stacking_model(selected["regularization_c"])
    model.fit(designs["stack_train"], frames["stack_train"]["label"])
    scored = {}
    for name, frame in frames.items():
        output = frame.copy()
        output["score"] = positive_scores(model, designs[name])
        scored[name] = output
    thresholds, global_threshold = calibrate_thresholds_per_community(scored["val"])
    predictions = {
        name: apply_thresholds(frame, thresholds, global_threshold)
        for name, frame in scored.items()
    }
    predictions["val"], net_vote_threshold = complete_with_net_vote_fallback(
        predictions["val"], val, val
    )
    predictions["test"], _ = complete_with_net_vote_fallback(
        predictions["test"], val, test
    )
    results = {
        name: classification_metrics(frame) for name, frame in predictions.items()
    }
    results["net_vote_fallback"] = {
        "threshold": net_vote_threshold,
        "threshold_source": "validation",
        "applies_when": "candidate_pool_empty",
    }
    results["coverage"] = coverage
    results["simulator_fidelity_on_observed_votes"] = {
        "val": simulator.evaluate_observed(val.reset_index(drop=True)),
        "test": simulator.evaluate_observed(test.reset_index(drop=True)),
    }
    names = feature_names(
        encoder,
        prediction_svd.n_components,
        eligibility_svd.n_components,
    )
    results["boc_stacking"] = {
        "interpretation": "everyone eligible is in the virtual team",
        "n_user_slots": int(len(users)),
        "n_model_features": int(len(names)),
        "raw_user_blocks": ["simulated_vote_score", "eligibility_mask"],
        "representation": "separate stack-TRAIN TruncatedSVD projections",
        "prediction_components": int(prediction_svd.n_components),
        "eligibility_components": int(eligibility_svd.n_components),
        "prediction_explained_variance": float(
            prediction_svd.explained_variance_ratio_.sum()
        ),
        "eligibility_explained_variance": float(
            eligibility_svd.explained_variance_ratio_.sum()
        ),
        "context_features": int(encoder.output_dimensions + 1),
        "classifier": "standardized class-balanced L2 logistic regression",
        "regularization_c": float(selected["regularization_c"]),
        "hyperparameter_search": hyperparameter_search,
        "current_case_historical_votes_used": False,
    }
    results["protocol"] = {
        **split_report,
        "policy_suffix_use": (
            "chronological inner hyperparameter tuning, then full-suffix "
            "stacking-classifier refit"
        ),
        "candidate_universe": "same prefix-only community pools as team_formation_v2.bandit",
        "votes_used_for_item_representation": "all eligible users' frozen simulator outputs",
        "validation_test_updates": False,
        "max_candidates": int(args.max_candidates),
        "min_user_history": int(args.min_user_history),
        "n_candidate_profiles": int(len(profiles)),
        "n_candidate_communities": int(len(candidate_pools)),
        "simulated_estimand": True,
        "warning": (
            "Policy metrics describe a simulated full-electorate world and must "
            "be reported with simulator fidelity and coverage."
        ),
    }
    (out_dir / "metrics.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    predictions["val"].to_parquet(out_dir / "val_predictions.parquet", index=False)
    predictions["test"].to_parquet(out_dir / "test_predictions.parquet", index=False)
    if args.save_user_predictions:
        user_predictions["val"].to_parquet(
            out_dir / "val_all_user_predictions.parquet", index=False
        )
        user_predictions["test"].to_parquet(
            out_dir / "test_all_user_predictions.parquet", index=False
        )
    profiles.to_parquet(out_dir / "candidate_profiles.parquet", index=False)

    logistic = model.named_steps["logisticregression"]
    scaler = model.named_steps["standardscaler"]
    coefficients = pd.DataFrame(
        {"feature": names, "standardized_coefficient": logistic.coef_[0]}
    ).sort_values("standardized_coefficient", ascending=False)
    coefficients.to_parquet(out_dir / "model_coefficients.parquet", index=False)
    effective = logistic.coef_[0] / np.maximum(scaler.scale_, 1e-12)
    prediction_latent = effective[
        context_dimension : context_dimension + prediction_svd.n_components
    ]
    eligibility_latent = effective[
        context_dimension + prediction_svd.n_components :
    ]
    user_coefficients = pd.DataFrame(
        {
            "username": users,
            "predicted_vote_coefficient": prediction_latent @ prediction_svd.components_,
            "eligibility_coefficient": eligibility_latent @ eligibility_svd.components_,
        }
    )
    user_coefficients.to_parquet(out_dir / "user_coefficients.parquet", index=False)
    loadings = pd.DataFrame({"username": users})
    for index, values in enumerate(prediction_svd.components_):
        loadings[f"predicted_vote_loading_{index:02d}"] = values
    for index, values in enumerate(eligibility_svd.components_):
        loadings[f"eligibility_loading_{index:02d}"] = values
    loadings.to_parquet(out_dir / "user_latent_loadings.parquet", index=False)
    print(
        f"BoC stacking: test macro-F1={results['test']['macro_f1']:.4f} | "
        f"saved={out_dir}"
    )


if __name__ == "__main__":
    main()
