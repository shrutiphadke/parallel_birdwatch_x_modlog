"""Single source of truth for dataset inputs, derived artifacts, and their schemas.

Every entry point (the Typer pipeline, preparation scripts, and method runners)
resolves paths through :class:`DatasetLayout` and reads user-supplied tables
through the canonical loaders below.  Nothing else in ``src`` should spell out a
``data/...`` or ``results/...`` path or re-implement column aliasing.

User-supplied inputs (everything else is derived):

========================  ===================================================  ==========
Input                     Canonical columns                                    Needed by
========================  ===================================================  ==========
``votes``                 username, community, item_id, timestamp, vote, label  every method
``items``                 item_id, timestamp, title, selftext, [text], score    SEF, NVSE, BL4/BL5
``user_contributions/``   one ``<username>.jsonl`` per user                     NVSE, SEF, Team Formation
``user_metadata``         username, account_created_utc (optional)              causal features
========================  ===================================================  ==========

``items`` merges what used to be two separate files (``post_texts`` and
``moderated_posts_scores``).  Both legacy files are still accepted.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable

import pandas as pd

from src.utils.tabular import SUPPORTED_SUFFIXES, read_table, unix_seconds

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_NAME = re.compile(r"[A-Za-z0-9._-]+")

DEFAULT_NORMVIO_BERT = "DeepPavlov/bert-base-cased-conversational"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
NORMVIO_CATEGORIES = (
    "spam",
    "meta-rules",
    "content",
    "harassment",
    "hatespeech",
    "format",
    "off-topic",
    "trolling",
    "incivility",
)

VOTE_COLUMNS = ("username", "community", "item_id", "timestamp", "vote", "label")
ITEM_TEXT_COLUMNS = ("item_id", "text")
ITEM_NVSE_COLUMNS = ("item_id", "title", "selftext")
ITEM_SCORE_COLUMNS = ("item_id", "score")
USER_DOCUMENT_COLUMNS = ("username", "timestamp", "text")
USER_METADATA_COLUMNS = ("username", "account_created_utc")

# Accepted alternative spellings, normalized once at read time.
_COLUMN_ALIASES = {"thing_id": "item_id", "created_utc": "timestamp"}

# Conventional file stems recognized inside an ``--inputs`` directory, in
# priority order.  The legacy names keep pre-existing data folders working.
_INPUT_STEMS = {
    "votes": ("votes", "final_intersection_dataset"),
    "items": ("items",),
    "post_texts": ("post_texts",),
    "item_scores": ("item_scores", "moderated_posts_scores"),
    "user_metadata": ("user_metadata",),
    "user_documents": ("user_documents",),
}


def validate_dataset_name(dataset: str) -> str:
    if not dataset or not DATASET_NAME.fullmatch(dataset):
        raise ValueError("dataset must contain only letters, digits, '.', '_' or '-'")
    return dataset


def _find_table(directory: Path, stems: Iterable[str]) -> Path | None:
    for stem in stems:
        for suffix in sorted(SUPPORTED_SUFFIXES):
            candidate = directory / f"{stem}{suffix}"
            if candidate.is_file():
                return candidate
    return None


@dataclass(frozen=True)
class InputBundle:
    """Resolved user-supplied inputs; ``None`` means not provided."""

    votes: Path | None = None
    post_texts: Path | None = None
    item_scores: Path | None = None
    user_contributions: Path | None = None
    user_metadata: Path | None = None
    user_documents: Path | None = None

    @classmethod
    def from_directory(cls, directory: Path) -> "InputBundle":
        """Resolve conventional names inside one inputs folder.

        ``items.*`` supplies both post texts and scores; ``post_texts.*`` and
        ``item_scores.*``/``moderated_posts_scores.*`` override it individually.
        """
        directory = Path(directory)
        if not directory.is_dir():
            raise FileNotFoundError(f"Inputs directory not found: {directory}")
        items = _find_table(directory, _INPUT_STEMS["items"])
        contributions = directory / "user_contributions"
        nested = contributions / "user_contributions"
        if nested.is_dir() and not any(contributions.glob("*.jsonl")):
            contributions = nested
        return cls(
            votes=_find_table(directory, _INPUT_STEMS["votes"]),
            post_texts=_find_table(directory, _INPUT_STEMS["post_texts"]) or items,
            item_scores=_find_table(directory, _INPUT_STEMS["item_scores"]) or items,
            user_contributions=contributions if contributions.is_dir() else None,
            user_metadata=_find_table(directory, _INPUT_STEMS["user_metadata"]),
            user_documents=_find_table(directory, _INPUT_STEMS["user_documents"]),
        )


@dataclass(frozen=True)
class DatasetLayout:
    """Every conventional input and derived artifact location for one dataset.

    The original ``reddit`` dataset predates per-dataset namespacing, so its
    processed inputs, caches, and report keep their historical un-namespaced
    locations.  All other datasets are fully isolated under their own name.
    """

    dataset: str
    root: Path = PROJECT_ROOT

    def __post_init__(self) -> None:
        validate_dataset_name(self.dataset)

    @property
    def _legacy(self) -> bool:
        return self.dataset == "reddit"

    # User-supplied inputs (conventional locations) -------------------------
    @property
    def raw(self) -> Path:
        return self.root / "data" / "raw" / self.dataset

    @property
    def processed(self) -> Path:
        base = self.root / "data" / "processed"
        return base if self._legacy else base / self.dataset

    @property
    def interim(self) -> Path:
        return self.root / "data" / "interim" / self.dataset

    @property
    def auxiliary(self) -> Path:
        return self.interim / "auxiliary"

    def default_inputs(self) -> InputBundle:
        """Conventional input locations, honouring legacy Reddit file names."""
        contributions = self.raw / "user_contributions"
        if self._legacy:
            contributions = (
                self.root / "data" / "raw" / "user_contributions" / "user_contributions"
            )
        # ``items.parquet`` is the current single-file form; pre-existing
        # two-file folders keep working because an existing legacy file wins.
        items = self.items
        legacy_texts = self.auxiliary / "post_texts.parquet"
        legacy_scores = self.auxiliary / "moderated_posts_scores.parquet"
        return InputBundle(
            votes=(
                self.processed / "final_intersection_dataset.csv"
                if self._legacy
                else _find_table(self.raw, _INPUT_STEMS["votes"])
            ),
            post_texts=legacy_texts if legacy_texts.is_file() else items,
            item_scores=legacy_scores if legacy_scores.is_file() else items,
            user_contributions=contributions,
            user_metadata=self.processed / "user_metadata.csv",
            user_documents=self.user_documents,
        )

    # Derived artifacts -----------------------------------------------------
    @property
    def datasets_dir(self) -> Path:
        return self.interim / "datasets"

    @property
    def dataset_manifest(self) -> Path:
        return self.datasets_dir / "prepare_reddit_data.manifest.json"

    @property
    def filtered_votes(self) -> Path:
        return self.datasets_dir / "filtered_votes.parquet"

    @property
    def build_state(self) -> Path:
        return self.interim / "build_state"

    @property
    def author_index(self) -> Path:
        return self.build_state / "author_index.sqlite"

    @property
    def feature_root(self) -> Path:
        return self.build_state / "user_temporal_features"

    @property
    def causal_features(self) -> Path:
        return self.interim / "features" / "causal_user_vote_features.parquet"

    @property
    def user_documents(self) -> Path:
        return self.auxiliary / "user_documents.parquet"

    @property
    def items(self) -> Path:
        return self.auxiliary / "items.parquet"

    @property
    def run_manifests(self) -> Path:
        """Where ``prepare`` subcommands record their execution manifests."""
        return self.interim

    @property
    def splits(self) -> Path:
        return self.root / "data" / "splits" / self.dataset

    @property
    def cache(self) -> Path:
        base = self.root / "cache"
        return base if self._legacy else base / self.dataset

    @property
    def embedding_cache(self) -> Path:
        return self.cache / "embeddings"

    @property
    def nvse_scores(self) -> Path:
        return self.cache / "nvse" / "post_violation_scores.parquet"

    @property
    def nvse_work(self) -> Path:
        return self.cache / "nvse" / "work"

    @property
    def results(self) -> Path:
        return self.root / "results" / self.dataset

    @property
    def report(self) -> Path:
        base = self.root / "results" / "report"
        return base if self._legacy else base / self.dataset

    @property
    def normvio_models(self) -> Path:
        return self.root / "external" / "normvio" / "normvio_redditmodels"


REDDIT = DatasetLayout("reddit")


def relative_to_root(path: Path) -> Path:
    """Return ``path`` relative to the project root when it lies inside it."""
    try:
        return path.relative_to(PROJECT_ROOT)
    except ValueError:
        return path


# Canonical loaders -----------------------------------------------------------


def normalize_aliases(frame: pd.DataFrame) -> pd.DataFrame:
    """Rename legacy column spellings to their canonical names."""
    renames = {
        legacy: canonical
        for legacy, canonical in _COLUMN_ALIASES.items()
        if legacy in frame.columns and canonical not in frame.columns
    }
    return frame.rename(columns=renames) if renames else frame


def _require(frame: pd.DataFrame, required: Iterable[str], path: Path, label: str) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} {path} is missing columns: {missing}")


def _clean_text(values: pd.Series) -> pd.Series:
    text = values.astype("string").str.strip()
    return text.mask(text.isin(["", "[removed]", "[deleted]"]))


def normalize_items(frame: pd.DataFrame) -> pd.DataFrame:
    """Give every per-item table the same text columns.

    * ``text`` is derived as ``title\\n\\nselftext`` when absent (the same rule
      the Arctic Shift helper uses);
    * ``title``/``selftext`` are derived from ``text`` when absent, so a table
      with only ``text`` also feeds NormVio.
    """
    frame = normalize_aliases(frame).copy()
    has_title = "title" in frame.columns
    has_body = "selftext" in frame.columns
    if "text" not in frame.columns and (has_title or has_body):
        title = _clean_text(frame["title"]) if has_title else pd.Series(pd.NA, index=frame.index)
        body = _clean_text(frame["selftext"]) if has_body else pd.Series(pd.NA, index=frame.index)
        both = title.notna() & body.notna()
        frame["text"] = title.fillna(body)
        frame.loc[both, "text"] = title[both] + "\n\n" + body[both]
    if "text" in frame.columns:
        if not has_title:
            frame["title"] = ""
        if not has_body:
            frame["selftext"] = frame["text"]
    if "item_id" in frame.columns:
        frame["item_id"] = frame["item_id"].astype(str)
    if "timestamp" in frame.columns:
        frame["timestamp"] = unix_seconds(frame["timestamp"])
    return frame


def item_table_columns(path: Path) -> set[str]:
    """Columns a per-item table provides after canonical derivation."""
    from src.utils.tabular import table_columns

    columns = {_COLUMN_ALIASES.get(column, column) for column in table_columns(path)}
    if columns & {"title", "selftext"}:
        columns.add("text")
    if "text" in columns:
        columns |= {"title", "selftext"}
    return columns


def load_votes(path: Path, *, columns: Iterable[str] | None = None) -> pd.DataFrame:
    """Read the votes table with numeric ``vote``/``label`` and Unix-second timestamps."""
    path = Path(path)
    frame = normalize_aliases(read_table(path))
    _require(frame, VOTE_COLUMNS, path, "Votes table")
    if columns is not None:
        frame = frame[list(columns)]
    if "timestamp" in frame:
        frame["timestamp"] = unix_seconds(frame["timestamp"])
    for column in ("vote", "label"):
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def load_items(path: Path, *, required: Iterable[str] = ("item_id",)) -> pd.DataFrame:
    """Read one per-item table (texts and/or scores) in canonical form."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Item table not found: {path}")
    frame = normalize_items(read_table(path))
    _require(frame, required, path, "Item table")
    return frame


def load_item_scores(path: Path) -> pd.DataFrame:
    frame = load_items(path, required=ITEM_SCORE_COLUMNS)[list(ITEM_SCORE_COLUMNS)]
    frame["score"] = pd.to_numeric(frame["score"], errors="coerce")
    return frame


def load_user_documents(path: Path) -> pd.DataFrame:
    path = Path(path)
    frame = normalize_aliases(read_table(path))
    _require(frame, USER_DOCUMENT_COLUMNS, path, "User-document table")
    frame["timestamp"] = unix_seconds(frame["timestamp"])
    return frame


def load_user_metadata(path: Path) -> pd.DataFrame:
    path = Path(path)
    frame = read_table(path)
    _require(frame, USER_METADATA_COLUMNS, path, "User-metadata table")
    frame["account_created_utc"] = unix_seconds(frame["account_created_utc"])
    return frame
