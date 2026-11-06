"""Shared configuration and runtime state for Semantic Expert Finder."""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
import sys
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

os.environ["TOKENIZERS_PARALLELISM"] = "false"

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.layout import DEFAULT_EMBEDDING_MODEL, REDDIT  # noqa: E402

# Standalone CLI defaults; the pipeline always passes explicit paths.
SPLITS_DIR = REDDIT.splits
OUTPUT_DIR = REDDIT.results
CACHE_DIR = REDDIT.embedding_cache

# Embedding model
EMBEDDING_MODEL = DEFAULT_EMBEDDING_MODEL
EMBEDDING_DIM = 384
BATCH_SIZE = 64

# Hyperparameter defaults
K_NEIGHBORS = 5
MIN_USER_VOTES = 10
ALPHA = 0.5
BETA = 0.3
LAMBDA_SMOOTH = 2.0
TOP_T = 3
CAL_SAMPLE = 4000
MIN_SUB_SAMPLES = 50

# Number of per-rank expert slots in feature vector
RANK_N = 5

# Minimum reliable downvoter experts required to activate the neg predictor
MIN_NEG_EXPERTS = 1
NEG_RELIABILITY_THR = 0.1  # minimum reliability to count as "reliable"

# Grid search
GRID_K = [5, 10]
GRID_ALPHA = [0.2, 0.3, 0.4, 0.5, 0.6]
GRID_BETA = [0.2, 0.3, 0.4, 0.5, 0.6]
# valid combos: alpha+beta <= 1.0
VAL_GRID_SAMPLE = 3000
META_OOF_FOLDS = 5

# Multi-scale weighted votes are all meta-model inputs.  ``TOP_T`` is retained
# only for the explicit weighted-vote fallback and is not grid-searched.
TOP_T_LIST = [1, 3, 5, 20]

# Evaluation
N_FOLDS = 5
SCALAR_METRICS = [
    "macro_f1",
    "roc_auc",
    "roc_auc_system",
    "roc_auc_expert_covered",
    "f1_pos",
    "f1_neg",
    "macro_precision",
    "macro_recall",
    "precision_pos",
    "recall_pos",
    "precision_neg",
    "recall_neg",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

post_ids_ordered: List[str] = []
causal_feature_table: Optional[pd.DataFrame] = None
causal_metadata_weight = 0.20
random_seed = 10

CAUSAL_METADATA_FEATURES = [
    "prior_n_posts",
    "prior_n_comments",
    "prior_n_distinct_subreddits",
    "prior_n_replies_made",
    "tenure_days_at_vote",
    "prior_n_posts_in_sub",
    "prior_n_comments_in_sub",
    "prior_n_months_active_in_sub",
    "prior_n_interaction_partners",
    "prior_total_interactions",
    "prior_n_interaction_partners_in_sub",
    "prior_total_interactions_in_sub",
]


def _rng_for_split(split_label: str, purpose: str) -> np.random.Generator:
    """Return a stable RNG for one split and sampling purpose."""
    payload = f"{random_seed}:{split_label}:{purpose}".encode("utf-8")
    derived_seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")
    return np.random.default_rng(derived_seed)


def _causal_metadata_scores(votes: pd.DataFrame, item_ids: List[str]) -> Dict[tuple, float]:
    """Return bounded per-vote causal profile scores for semantic-weight fusion."""
    columns = CAUSAL_METADATA_FEATURES
    subset = votes[votes["item_id"].isin(item_ids)][["item_id", "username", *columns]].copy()
    subset = subset.drop_duplicates(["item_id", "username"], keep="last")
    components = []
    for column in columns:
        values = pd.to_numeric(subset[column], errors="coerce")
        if column == "tenure_days_at_vote":
            signal = 1.0 - np.exp(-np.log1p(values.clip(lower=0)) / 7.0)
            signal = signal.fillna(0.5)
        else:
            signal = 1.0 - np.exp(-np.log1p(values.fillna(0).clip(lower=0)) / 5.0)
        components.append(signal.to_numpy(dtype=float))
    subset["causal_metadata_score"] = np.column_stack(components).mean(axis=1)
    return subset.set_index(["item_id", "username"])["causal_metadata_score"].to_dict()
