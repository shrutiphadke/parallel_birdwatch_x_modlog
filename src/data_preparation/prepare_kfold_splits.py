"""Build K-fold artifacts by recomposing the existing canonical prepared splits."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.data_preparation.prepare_reddit_data import (
    _CHRONOLOGICAL_INITIAL_TRAIN_FRAC,
    _K_FOLDS,
    SEED,
    write_kfold_splits,
)
from src.layout import REDDIT  # noqa: E402

DEFAULT_SPLITS_ROOT = REDDIT.splits


def _load_recomposed_split(split_dir: Path) -> pd.DataFrame:
    missing = [
        name
        for name in ("train", "val", "test")
        if not (split_dir / f"{name}_votes.parquet").exists()
    ]
    if missing:
        raise FileNotFoundError(f"Incomplete canonical split at {split_dir}; missing {missing}")
    return pd.concat(
        [
            pd.read_parquet(split_dir / f"{name}_votes.parquet")
            for name in ("train", "val", "test")
        ],
        ignore_index=True,
    )


def prepare_from_existing(
    splits_root: Path = DEFAULT_SPLITS_ROOT,
    output_root: Path | None = None,
    n_folds: int = _K_FOLDS,
    seed: int = SEED,
    chronological_initial_train_fraction: float = _CHRONOLOGICAL_INITIAL_TRAIN_FRAC,
) -> dict:
    """Recompose FULL and INTERSECTION once, then materialize all three fold collections."""
    output_root = output_root or splits_root / "kfold"
    full_votes = _load_recomposed_split(splits_root / "full")
    intersection_votes = _load_recomposed_split(splits_root / "intersection")
    chronological_votes = _load_recomposed_split(splits_root / "intersection_chronological")
    if set(intersection_votes["item_id"]) != set(chronological_votes["item_id"]):
        raise ValueError(
            "intersection and intersection_chronological do not contain the same item population"
        )
    return write_kfold_splits(
        full_votes,
        intersection_votes,
        output_root,
        n_folds=n_folds,
        seed=seed,
        chronological_initial_train_fraction=chronological_initial_train_fraction,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits-root", type=Path, default=DEFAULT_SPLITS_ROOT)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--n-folds", type=int, default=_K_FOLDS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--chronological-initial-train-fraction",
        type=float,
        default=_CHRONOLOGICAL_INITIAL_TRAIN_FRAC,
    )
    args = parser.parse_args()
    manifest = prepare_from_existing(
        splits_root=args.splits_root,
        output_root=args.output_root,
        n_folds=args.n_folds,
        seed=args.seed,
        chronological_initial_train_fraction=args.chronological_initial_train_fraction,
    )
    print(
        f"K-fold splits ready: {args.output_root or args.splits_root / 'kfold'} | "
        f"datasets={len(manifest['datasets'])} | folds={manifest['n_folds']}"
    )


if __name__ == "__main__":
    main()
