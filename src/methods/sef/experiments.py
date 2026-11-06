"""Single-split, K-fold, multi-split, and output orchestration for SEF."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from src.methods.team_formation_v2.shared_features import attach_causal_features
from src.utils.net_vote_fallback import (
    calibrate_net_vote_threshold,
    net_vote_item_scores,
)
from src.utils.splits import discover_splits, split_dataset

from . import runtime
from .data import TemporalUserProfiles, _load_split_votes
from .evaluation import evaluate_fold, grid_search_hyperparams, predict_fold
from .features import (
    build_reference_heldout_train_features,
    compute_expert_weights,
    compute_subreddit_base_rates,
    extract_post_features,
    get_meta_feature_cols,
    precompute_vote_precision,
)
from .modeling import (
    _items_with_real_expert_signal,
    add_oof_negative_probabilities,
    apply_neg_predictor,
    calibrate_probability_thresholds_per_subreddit,
    predict_meta_model,
    train_meta_model,
)
from .runtime import (
    CAL_SAMPLE,
    LAMBDA_SMOOTH,
    N_FOLDS,
    SCALAR_METRICS,
    VAL_GRID_SAMPLE,
    _rng_for_split,
    log,
)

# 12. SINGLE-SPLIT EVALUATION  (core logic, reused by K-fold and multi-split)


def run_single_split(
    train_ids: set,
    val_ids: Optional[set],
    test_ids: set,
    vote_df: pd.DataFrame,
    post_emb: np.ndarray,
    post_id2idx: Dict[str, int],
    user_profiles: TemporalUserProfiles,
    faiss_index: faiss.Index,
    default_k: int,
    default_t: int,
    default_alpha: float,
    default_beta: float,
    min_coverage: int,
    do_grid_search: bool = True,
    split_label: str = "",
    fixed_hyperparams: Optional[Tuple[int, int, float, float]] = None,
    dataset: str | None = None,
) -> Tuple[Optional[Dict], pd.DataFrame, pd.DataFrame]:
    """Run the single split workflow."""
    train_vote_df = vote_df[vote_df["item_id"].isin(train_ids)]
    use_net_vote_fallback = True
    prec_df = precompute_vote_precision(train_vote_df, LAMBDA_SMOOTH, min_coverage)
    base_rates = compute_subreddit_base_rates(train_vote_df)
    log.info(
        "[%s] train=%d, validation=%d, test=%d; reliable users (at least %d votes): %d",
        split_label,
        len(train_ids),
        len(val_ids or []),
        len(test_ids),
        min_coverage,
        prec_df["username"].nunique(),
    )

    # hyperparameter selection: use real val if we have one, else carve from train
    if val_ids:
        val_grid_ids = sorted(val_ids)
        train_grid = train_ids
    else:
        train_list = sorted(train_ids)
        _rng_for_split(split_label, "grid-validation").shuffle(train_list)
        val_grid_ids = train_list[: min(VAL_GRID_SAMPLE, len(train_list))]
        train_grid = set(train_list[min(VAL_GRID_SAMPLE, len(train_list)) :])

    if fixed_hyperparams is not None:
        k, _legacy_t, alpha, beta = fixed_hyperparams
        t = default_t
        hyperparameter_source = "reused_existing_val_selection"
        log.info(
            "[%s] reused parameters: k=%d t=%d alpha=%.2f beta=%.2f",
            split_label,
            k,
            t,
            alpha,
            beta,
        )
    elif do_grid_search and val_grid_ids:
        grid_reference_votes = vote_df[vote_df["item_id"].isin(train_grid)]
        grid_precision = precompute_vote_precision(
            grid_reference_votes, LAMBDA_SMOOTH, min_coverage
        )
        k, _grid_t, alpha, beta = grid_search_hyperparams(
            val_grid_ids,
            train_grid,
            vote_df,
            grid_precision,
            post_emb,
            post_id2idx,
            user_profiles,
            faiss_index,
            LAMBDA_SMOOTH,
            use_net_vote_fallback=use_net_vote_fallback,
        )
        t = default_t
        hyperparameter_source = "grid_search_val"
    else:
        k, t, alpha, beta = default_k, default_t, default_alpha, default_beta
        hyperparameter_source = "cli_defaults"

    # expert weights for the TEST set (weighted using TRAIN-only precision)
    test_id_list = sorted(test_ids)
    weights_df = compute_expert_weights(
        test_id_list,
        train_ids,
        vote_df,
        prec_df,
        post_emb,
        post_id2idx,
        user_profiles,
        faiss_index,
        k,
        alpha,
        beta,
        LAMBDA_SMOOTH,
    )

    # With no external VAL (legacy k-fold), reserve val_grid_ids completely for
    # model selection and threshold calibration.  Canonical splits use their
    # real validation partition.
    meta_reference_ids = set(train_ids if val_ids else train_grid)
    train_sample = sorted(meta_reference_ids)
    _rng_for_split(split_label, "meta-model-train").shuffle(train_sample)
    # Reserve at least half of TRAIN as a label-disjoint source for reliability,
    # subreddit base rates and semantic-neighbour correctness.
    target_count = min(CAL_SAMPLE, max(len(train_sample) // 2, 1))
    train_sample = train_sample[:target_count]
    feature_reference_ids = meta_reference_ids - set(train_sample)

    train_feat_df, weights_train_sample = build_reference_heldout_train_features(
        train_sample,
        feature_reference_ids,
        vote_df,
        post_emb,
        post_id2idx,
        user_profiles,
        faiss_index,
        k,
        alpha,
        beta,
        min_coverage,
    )
    train_labels = (
        vote_df[vote_df["item_id"].isin(train_sample)]
        .drop_duplicates("item_id")
        .set_index("item_id")["label"]
        .to_dict()
    )

    raw_feat_cols = get_meta_feature_cols(train_feat_df)
    train_feat_df, neg_clf, neg_scaler, neg_feat_cols = add_oof_negative_probabilities(
        train_feat_df, train_labels, raw_feat_cols
    )
    feat_cols = get_meta_feature_cols(train_feat_df)

    # External validation features use statistics fitted on the complete
    # training reference.  They are used both for model selection and for the
    # final probability-threshold calibration; TEST remains untouched.
    cal_ids = sorted(val_ids) if val_ids else sorted(val_grid_ids)
    cal_reference_ids = set(train_ids if val_ids else train_grid)
    cal_reference_votes = vote_df[vote_df["item_id"].isin(cal_reference_ids)]
    cal_precision = precompute_vote_precision(cal_reference_votes, LAMBDA_SMOOTH, min_coverage)
    cal_base_rates = compute_subreddit_base_rates(cal_reference_votes)
    weights_cal = compute_expert_weights(
        cal_ids,
        cal_reference_ids,
        vote_df,
        cal_precision,
        post_emb,
        post_id2idx,
        user_profiles,
        faiss_index,
        k,
        alpha,
        beta,
        LAMBDA_SMOOTH,
    )
    cal_feat_df = extract_post_features(cal_ids, vote_df, weights_cal, cal_base_rates)
    neg_probas_cal = apply_neg_predictor(cal_feat_df, cal_ids, neg_clf, neg_scaler, neg_feat_cols)
    cal_feat_df["neg_expert_proba"] = cal_feat_df["item_id"].map(neg_probas_cal).fillna(0.5)
    cal_labels = (
        vote_df[vote_df["item_id"].isin(cal_ids)]
        .drop_duplicates("item_id")
        .set_index("item_id")["label"]
        .to_dict()
    )
    cal_vote_df = vote_df[vote_df["item_id"].isin(cal_ids)]
    net_vote_threshold = calibrate_net_vote_threshold(cal_vote_df)
    cal_expert_ids = _items_with_real_expert_signal(weights_cal)
    cal_net_scores = net_vote_item_scores(cal_vote_df).set_index("item_id")["net_vote_score"]
    cal_net_predictions = {
        item_id: 1 if float(cal_net_scores.get(item_id, 0.0)) >= net_vote_threshold else -1
        for item_id in cal_ids
    }
    cal_fixed_predictions = (
        {
            item_id: cal_net_predictions[item_id]
            for item_id in cal_ids
            if item_id not in cal_expert_ids
        }
        if use_net_vote_fallback
        else {}
    )

    clf, scaler, model_name = train_meta_model(
        train_feat_df,
        train_labels,
        feat_cols,
        cal_feat_df,
        cal_labels,
        validation_fixed_predictions=cal_fixed_predictions,
    )
    cal_meta = predict_meta_model(cal_feat_df, cal_ids, clf, scaler, feat_cols)
    cal_positive_probabilities = {
        iid: proba for iid, (_, proba) in cal_meta.items() if proba is not None
    }
    threshold_calibration_ids = [
        item_id for item_id in cal_ids if item_id not in cal_fixed_predictions
    ]
    thresholds = calibrate_probability_thresholds_per_subreddit(
        threshold_calibration_ids, vote_df, cal_positive_probabilities
    )

    test_feat_df = extract_post_features(test_id_list, vote_df, weights_df, base_rates)

    decisions = predict_fold(
        test_id_list,
        vote_df,
        weights_df,
        test_feat_df,
        feat_cols,
        clf,
        scaler,
        thresholds,
        t,
        neg_clf=neg_clf,
        neg_scaler=neg_scaler,
        neg_feat_cols=neg_feat_cols,
        use_net_vote_fallback=use_net_vote_fallback,
        net_vote_threshold=net_vote_threshold,
    )
    decisions["split"] = split_label
    decisions["k"] = k
    decisions["top_t"] = t
    decisions["alpha"] = alpha
    decisions["beta"] = beta

    if decisions["label"].nunique() < 2:
        log.warning("[%s] skipped: single-class test set", split_label)
        return None, decisions, weights_df

    metrics = evaluate_fold(decisions)
    temporal_covered = weights_df["has_user_embedding"].astype(bool)
    temporal_latest = weights_df.loc[temporal_covered, "semantic_profile_latest_document_utc"]
    temporal_cutoff = weights_df.loc[temporal_covered, "semantic_profile_cutoff_utc"]
    if not (temporal_latest < temporal_cutoff).all():
        raise AssertionError("SEF temporal-profile audit detected future user documents")
    closest_margin = (
        float((temporal_cutoff - temporal_latest).min()) if temporal_covered.any() else None
    )
    val_metrics = None
    if val_ids:
        val_decisions = predict_fold(
            cal_ids,
            vote_df,
            weights_cal,
            cal_feat_df,
            feat_cols,
            clf,
            scaler,
            thresholds,
            t,
            neg_clf=neg_clf,
            neg_scaler=neg_scaler,
            neg_feat_cols=neg_feat_cols,
            use_net_vote_fallback=use_net_vote_fallback,
            net_vote_threshold=net_vote_threshold,
        )
        if val_decisions["label"].nunique() >= 2:
            val_metrics = evaluate_fold(val_decisions)
            val_metrics.update(
                {
                    "n_items": len(val_decisions),
                    "fallback_pct": float(val_decisions["used_fallback"].mean()),
                }
            )
    metrics.update(
        {
            "split": split_label,
            "k": k,
            "top_t": t,
            "alpha": alpha,
            "beta": beta,
            "gamma": 1.0 - alpha - beta,
            "threshold": thresholds.get(None, 0.5),
            "n_train_items": len(train_ids),
            "n_val_items": len(val_ids) if val_ids else 0,
            "n_test": len(decisions),
            "fallback_pct": float(decisions["used_fallback"].mean()),
            "n_fallback_items": int(decisions["used_fallback"].sum()),
            "model_name": model_name,
            "hyperparameter_source": hyperparameter_source,
            "random_seed": runtime.random_seed,
            "val": val_metrics,
            "has_neg_predictor": int(neg_clf is not None),
            "train_features_cross_fitted_by_item": False,
            "train_features_reference_disjoint": True,
            "train_feature_encoding": "label_disjoint_reference_partition",
            "n_feature_reference_items": len(feature_reference_ids),
            "negative_predictor_train_proba_oof": True,
            "meta_model_selection": "external_validation",
            "decision_threshold_source": "external_validation_meta_probability",
            "top_t_is_grid_searched": False,
            "expert_weight_grid_objective": "validation_weighted_vote_proxy_macro_f1",
            "expert_weight_grid_optimizes_final_meta_model": False,
            "fallback_definition": "no_history_local_evidence_or_user_embedding",
            "used_net_vote_fallback": use_net_vote_fallback,
            "net_vote_fallback_threshold": net_vote_threshold,
            "causal_metadata_profile": "full",
            "causal_metadata_weight": runtime.causal_metadata_weight,
            "semantic_user_profiles_are_static": False,
            "semantic_profile_regime": "causal_as_of_case_time",
            "semantic_profile_timestamp_rule": ("document.timestamp < max_case_vote_timestamp"),
            "semantic_profile_cache_key": user_profiles.cache_key,
            "temporal_profile_vote_coverage": (
                float(temporal_covered.mean()) if len(temporal_covered) else 0.0
            ),
            "n_votes_with_temporal_profile": int(temporal_covered.sum()),
            "minimum_profile_cutoff_margin_seconds": closest_margin,
            "temporal_profile_audit_passed": True,
            "temporal_profile_is_causal": True,
            "prospective_interpretation": "chronological" in split_label.lower(),
        }
    )

    log.info(
        "[%s] macro-F1=%.4f | AUC=%.4f | model=%s | weights=%.2f/%.2f/%.2f",
        split_label,
        metrics["macro_f1"],
        metrics["roc_auc"],
        model_name,
        alpha,
        beta,
        1.0 - alpha - beta,
    )

    weights_df = weights_df.copy()
    weights_df["split"] = split_label
    return metrics, decisions, weights_df


# 12b. KFOLD  (legacy: shuffled folds, validation carved out of training)


def run_kfold(
    vote_df: pd.DataFrame,
    post_emb: np.ndarray,
    post_id_list: List[str],
    user_profiles: TemporalUserProfiles,
    faiss_index: faiss.Index,
    default_k: int,
    default_t: int,
    default_alpha: float,
    default_beta: float,
    min_coverage: int,
    do_grid_search: bool = True,
) -> Tuple[List[Dict], List[pd.DataFrame], List[pd.DataFrame]]:
    """Run the kfold workflow."""
    global post_ids_ordered
    post_ids_ordered = post_id_list

    post_id2idx = {pid: i for i, pid in enumerate(post_id_list)}

    labeled_items = (
        vote_df.drop_duplicates("item_id")[["item_id", "label"]]
        .dropna(subset=["label"])
        .reset_index(drop=True)
    )
    log.info("Labeled items: %d", len(labeled_items))

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=10)
    fold_metrics: List[Dict] = []
    fold_item_scores: List[pd.DataFrame] = []
    fold_weights: List[pd.DataFrame] = []

    for fold_idx, (train_val_idx, test_idx) in enumerate(kf.split(labeled_items)):
        log.info("Fold %d of %d", fold_idx + 1, N_FOLDS)

        train_ids = set(labeled_items.iloc[train_val_idx]["item_id"].tolist())
        test_ids = set(labeled_items.iloc[test_idx]["item_id"].tolist())

        metrics, decisions, weights_df = run_single_split(
            train_ids,
            None,
            test_ids,
            vote_df,
            post_emb,
            post_id2idx,
            user_profiles,
            faiss_index,
            default_k,
            default_t,
            default_alpha,
            default_beta,
            min_coverage,
            do_grid_search,
            split_label=f"fold{fold_idx}",
            dataset=None,
        )
        if metrics is None:
            continue

        metrics["fold"] = fold_idx
        decisions["fold"] = fold_idx
        weights_df["fold"] = fold_idx

        fold_metrics.append(metrics)
        fold_item_scores.append(decisions)
        fold_weights.append(weights_df)

    return fold_metrics, fold_item_scores, fold_weights


# 12c. MULTI-SPLIT BENCHMARK  (uses splits produced by prepare_data_step1)


def _load_reusable_hyperparams(
    output_dir: Path,
    split_name: str,
) -> Tuple[int, int, float, float]:
    """Load the previously VAL-selected expert parameters for one split."""
    metrics_path = output_dir / split_name / "expertise" / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(
            f"Cannot reuse hyperparameters: missing {metrics_path}. "
            "Run once with grid search or omit --reuse-hyperparams."
        )
    previous = json.loads(metrics_path.read_text(encoding="utf-8"))
    required = ("k", "top_t", "alpha", "beta")
    missing = [key for key in required if key not in previous]
    if missing:
        raise ValueError(f"Cannot reuse hyperparameters from {metrics_path}: missing {missing}")
    return (
        int(previous["k"]),
        int(previous["top_t"]),
        float(previous["alpha"]),
        float(previous["beta"]),
    )


def _evaluate_prepared_split(
    split_name: str,
    split_path: Path,
    output_dir: Path,
    post_emb: np.ndarray,
    post_id2idx: Dict[str, int],
    user_profiles: TemporalUserProfiles,
    faiss_index: faiss.Index,
    default_k: int,
    default_t: int,
    default_alpha: float,
    default_beta: float,
    min_coverage: int,
    do_grid_search: bool,
    reuse_hyperparams: bool,
) -> tuple[str, Dict | None]:
    """Evaluate one split while sharing immutable embeddings and profiles."""

    chronological = "chronological" in split_name.lower()
    train_votes = _load_split_votes(split_path / "train_votes.parquet")
    val_votes = _load_split_votes(split_path / "val_votes.parquet")
    test_votes = _load_split_votes(split_path / "test_votes.parquet")
    split_vote_df = pd.concat([train_votes, val_votes, test_votes], ignore_index=True)
    if runtime.causal_feature_table is None:
        raise RuntimeError("SEF requires the full causal metadata feature table")
    split_vote_df = attach_causal_features(split_vote_df, runtime.causal_feature_table)

    train_ids = set(train_votes["item_id"].unique())
    val_ids = set(val_votes["item_id"].unique())
    test_ids = set(test_votes["item_id"].unique())
    if not test_ids:
        log.warning("[%s] skipped: empty test set", split_name)
        return split_name, None
    fixed_hyperparams = (
        _load_reusable_hyperparams(output_dir, split_name) if reuse_hyperparams else None
    )
    metrics, decisions, weights_df = run_single_split(
        train_ids,
        val_ids,
        test_ids,
        split_vote_df,
        post_emb,
        post_id2idx,
        user_profiles,
        faiss_index,
        default_k,
        default_t,
        default_alpha,
        default_beta,
        min_coverage,
        do_grid_search,
        split_label=split_name,
        fixed_hyperparams=fixed_hyperparams,
        dataset=split_dataset(split_path, split_name),
    )
    if metrics is None:
        return split_name, None
    metrics["causal_metadata_profile"] = "full"
    metrics["causal_metadata_weight"] = runtime.causal_metadata_weight
    metrics["semantic_user_profiles_are_static"] = False
    metrics["semantic_profile_regime"] = "causal_as_of_case_time"
    metrics["chronological_split"] = chronological
    metrics["temporal_profile_is_causal"] = True
    metrics["prospective_interpretation"] = chronological

    split_out_dir = output_dir / split_name / "expertise"
    split_out_dir.mkdir(parents=True, exist_ok=True)
    decisions.to_parquet(split_out_dir / "item_scores.parquet", index=False)
    weights_df.to_parquet(split_out_dir / "weights.parquet", index=False)
    with open(split_out_dir / "metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=2)
    return split_name, metrics


def evaluate_splits(
    votes_dir: Path,
    output_dir: Path,
    post_emb: np.ndarray,
    post_id_list: List[str],
    user_profiles: TemporalUserProfiles,
    faiss_index: faiss.Index,
    default_k: int,
    default_t: int,
    default_alpha: float,
    default_beta: float,
    min_coverage: int,
    do_grid_search: bool = True,
    reuse_hyperparams: bool = False,
    n_workers: int = 1,
    split_label: str | None = None,
) -> Dict[str, Dict]:
    """Evaluate splits and return its metrics.

    ``split_label`` names a single split passed directly as ``votes_dir`` so a
    per-split process keeps the canonical name used for seeds and output paths.
    """
    global post_ids_ordered
    post_ids_ordered = post_id_list
    post_id2idx = {pid: i for i, pid in enumerate(post_id_list)}

    splits = discover_splits(votes_dir)
    if split_label is not None:
        if len(splits) != 1:
            raise ValueError("--split-label requires --votes-dir to be exactly one split")
        splits = {split_label: next(iter(splits.values()))}
    if not splits:
        log.warning(
            "No prepared splits found under %s",
            votes_dir,
        )
        return {}

    if n_workers < 1:
        raise ValueError("n_workers must be at least 1")
    items = list(splits.items())

    def evaluate(item: tuple[str, Path]) -> tuple[str, Dict | None]:
        return _evaluate_prepared_split(
            item[0],
            item[1],
            output_dir,
            post_emb,
            post_id2idx,
            user_profiles,
            faiss_index,
            default_k,
            default_t,
            default_alpha,
            default_beta,
            min_coverage,
            do_grid_search,
            reuse_hyperparams,
        )

    if n_workers == 1:
        completed = map(evaluate, items)
    else:
        executor = ThreadPoolExecutor(max_workers=min(n_workers, len(items)))
        completed = executor.map(evaluate, items)
    try:
        summary = {name: metrics for name, metrics in completed if metrics is not None}
    finally:
        if n_workers != 1:
            executor.shutdown(wait=True)

    return summary


# Output


def save_outputs(
    fold_metrics: List[Dict],
    fold_item_scores: List[pd.DataFrame],
    fold_weights: List[pd.DataFrame],
    output_dir: Path,
) -> None:
    """Write outputs to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)

    pd.concat(fold_item_scores, ignore_index=True).to_parquet(
        output_dir / "item_scores.parquet", index=False
    )
    pd.concat(fold_weights, ignore_index=True).to_parquet(
        output_dir / "weights.parquet", index=False
    )
    pd.DataFrame(
        [{k: v for k, v in m.items() if isinstance(v, (int, float, bool))} for m in fold_metrics]
    ).to_parquet(output_dir / "fold_details.parquet", index=False)

    agg: Dict = {"n_folds": len(fold_metrics)}
    for key in SCALAR_METRICS:
        vals = [m[key] for m in fold_metrics if key in m and not np.isnan(m[key])]
        agg[key] = float(np.mean(vals)) if vals else float("nan")
        agg[f"{key}_std"] = float(np.std(vals)) if vals else float("nan")
        agg[f"{key}_per_fold"] = vals

    agg["best_k_per_fold"] = [m.get("k") for m in fold_metrics]
    agg["best_t_per_fold"] = [m.get("top_t") for m in fold_metrics]
    agg["best_alpha_per_fold"] = [m.get("alpha") for m in fold_metrics]
    agg["best_beta_per_fold"] = [m.get("beta") for m in fold_metrics]
    agg["best_gamma_per_fold"] = [m.get("gamma") for m in fold_metrics]
    agg["model_per_fold"] = [m.get("model_name") for m in fold_metrics]
    agg["neg_pred_per_fold"] = [m.get("has_neg_predictor") for m in fold_metrics]
    agg["fallback_pct"] = float(np.mean([m.get("fallback_pct", 0) for m in fold_metrics]))

    # Model selection summary
    from collections import Counter

    model_counts = Counter(agg["model_per_fold"])
    agg["model_wins"] = dict(model_counts)

    with open(output_dir / "metrics.json", "w") as fh:
        json.dump(agg, fh, indent=2)

    log.info(
        "Saved to %s | macro_f1=%.4f+/-%.4f  roc_auc=%.4f+/-%.4f  f1_pos=%.4f  f1_neg=%.4f",
        output_dir,
        agg["macro_f1"],
        agg["macro_f1_std"],
        agg["roc_auc"],
        agg["roc_auc_std"],
        agg["f1_pos"],
        agg["f1_neg"],
    )
