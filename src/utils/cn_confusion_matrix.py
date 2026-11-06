"""Generate the fixed-split confusion matrices for the base Community Notes model."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Annotated

from src.layout import REDDIT  # noqa: E402

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str((_PROJECT_ROOT / "cache" / "matplotlib").resolve()))

import matplotlib

matplotlib.use("Agg")

from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer

log = logging.getLogger(__name__)

FIXED_SPLITS = {
    "Full": "full",
    "Intersection": "intersection",
    "Chronological": "intersection_chronological",
}
CLASS_VALUES = (-1, 1)
CLASS_LABELS = ("Remove", "Approve")
CN_COLOUR = "#a61e2a"


def load_cn_predictions(results_root: Path, split: str) -> pd.DataFrame:
    """Load and validate one decision per test post for the base CN model."""
    path = results_root / split / "cn" / "item_scores.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing CN prediction artifact: {path}")

    frame = pd.read_parquet(path)
    required = {"item_id", "label", "prediction"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    if frame["item_id"].duplicated().any():
        raise ValueError(f"{path} contains duplicate item_id values")

    frame = frame[["item_id", "label", "prediction"]].copy()
    frame[["label", "prediction"]] = frame[["label", "prediction"]].astype(int)
    invalid_labels = set(frame["label"].unique()).difference(CLASS_VALUES)
    invalid_predictions = set(frame["prediction"].unique()).difference(CLASS_VALUES)
    if invalid_labels or invalid_predictions:
        raise ValueError(
            f"Invalid values in {path}: labels={invalid_labels}, predictions={invalid_predictions}"
        )
    return frame


def confusion_counts(frame: pd.DataFrame) -> np.ndarray:
    """Return a 2-by-2 matrix with true classes on rows and predictions on columns."""
    return (
        pd.crosstab(frame["label"], frame["prediction"])
        .reindex(index=CLASS_VALUES, columns=CLASS_VALUES, fill_value=0)
        .to_numpy(dtype=int)
    )


def plot_cn_confusion_matrices(results_root: Path, output_path: Path) -> None:
    """Plot count and row-normalized confusion matrices for all fixed test splits."""
    matrices = {
        title: confusion_counts(load_cn_predictions(results_root, split))
        for title, split in FIXED_SPLITS.items()
    }
    colour_map = LinearSegmentedColormap.from_list("cn_red", ("#fff8f7", CN_COLOUR))
    figure, axes = plt.subplots(
        1,
        len(matrices),
        figsize=(13.5, 4.2),
        sharex=True,
        sharey=True,
        layout="constrained",
    )

    image = None
    for axis, (title, counts) in zip(axes, matrices.items(), strict=True):
        row_totals = counts.sum(axis=1, keepdims=True)
        shares = np.divide(
            counts,
            row_totals,
            out=np.zeros_like(counts, dtype=float),
            where=row_totals != 0,
        )
        image = axis.imshow(shares, cmap=colour_map, vmin=0.0, vmax=1.0)
        axis.set_title(f"{title}\n$n={counts.sum():,}$", fontsize=11, fontweight="bold")
        axis.set_xticks(range(2), labels=CLASS_LABELS)
        axis.set_yticks(range(2), labels=CLASS_LABELS)
        axis.set_xlabel("CN prediction")
        axis.tick_params(length=0)

        for row in range(2):
            for column in range(2):
                share = shares[row, column]
                text_colour = "white" if share >= 0.58 else "#222222"
                axis.text(
                    column,
                    row,
                    f"{counts[row, column]:,}\n({share:.1%})",
                    ha="center",
                    va="center",
                    color=text_colour,
                    fontsize=11,
                    fontweight="bold",
                )

    axes[0].set_ylabel("Moderator label")
    figure.suptitle(
        "Base Community Notes decisions by fixed test protocol",
        fontsize=14,
        fontweight="bold",
    )
    if image is None:
        raise RuntimeError("No confusion matrices were generated")
    colour_bar = figure.colorbar(image, ax=axes, shrink=0.9, pad=0.025)
    colour_bar.set_label("Share within moderator class")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    log.info("Wrote CN confusion matrix to %s", output_path)


def main(
    results_root: Annotated[Path, typer.Option()] = REDDIT.results,
    output_path: Annotated[Path, typer.Option()] = REDDIT.report
    / "figures"
    / "cn_base_confusion_matrix.png",
    log_level: Annotated[str, typer.Option()] = "INFO",
) -> None:
    """Generate the base-CN confusion-matrix figure used in Chapter 5."""
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise typer.BadParameter(f"Unknown log level: {log_level}", param_hint="--log-level")
    logging.basicConfig(level=numeric_level, format="%(message)s")
    plot_cn_confusion_matrices(results_root, output_path)


if __name__ == "__main__":
    typer.run(main)
