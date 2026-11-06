"""Vote, text, embedding-cache, and FAISS data operations for SEF."""

from __future__ import annotations

from dataclasses import dataclass
import functools
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np
import pandas as pd
from tqdm import tqdm

from src import layout
from src.utils.splits import discover_splits
from src.utils.tabular import unix_seconds

from .runtime import (
    BATCH_SIZE,
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    log,
)

# 1. DATA LOADING


def load_votes(input_csv: Path) -> pd.DataFrame:
    """Load the configured votes table (no implicit fallbacks)."""
    if input_csv is None:
        raise ValueError("SEF requires the votes table (--input)")
    df = layout.load_votes(input_csv)
    df = df[df["vote"].isin([1, -1]) & df["label"].isin([1, -1])].copy()
    df["vote"] = df["vote"].astype(int)
    df["label"] = df["label"].astype(int)
    df = df.dropna(subset=["item_id", "username", "vote", "label"])
    df = df.drop_duplicates(subset=["username", "item_id"])
    log.info(
        "Votes: %d rows, %d posts, %d users, %d communities",
        len(df),
        df["item_id"].nunique(),
        df["username"].nunique(),
        df["community"].nunique(),
    )
    return df


def _load_split_votes(path: Path) -> pd.DataFrame:
    """Load split votes from its configured source."""
    df = pd.read_parquet(path)
    df = df.dropna(subset=["item_id", "username", "vote", "label"])
    df["vote"] = df["vote"].astype(int)
    df["label"] = df["label"].astype(int)
    if "timestamp" in df:
        df["timestamp"] = unix_seconds(df["timestamp"])
    return df


def load_post_texts(path: Path) -> pd.DataFrame:
    """Load the canonical item table rows that carry text."""
    df = layout.load_items(path, required=layout.ITEM_TEXT_COLUMNS)
    before = len(df)
    df = df[df["text"].notna()].copy()
    log.info("Post texts available: %d of %d", len(df), before)
    return df


def load_user_documents(path: Path | None) -> pd.DataFrame:
    """Load user documents; a missing table yields the neutral Signal C."""
    if path is None or not Path(path).exists():
        log.warning("User documents not found; Signal C will use the neutral value 0.5.")
        return pd.DataFrame(columns=["username", "timestamp", "text"])
    df = layout.load_user_documents(path)
    df = df[df["text"].notna()].copy()
    missing_time = int(df["timestamp"].isna().sum())
    if missing_time:
        log.warning("Dropping %d user documents without a valid timestamp", missing_time)
        df = df[df["timestamp"].notna()].copy()
    log.info("User documents: %d documents, %d users", len(df), df["username"].nunique())
    return df


# 2. EMBEDDINGS


def _get_model() -> str:
    """Return model for the supplied input."""
    return EMBEDDING_MODEL


@functools.lru_cache(maxsize=2)
def _load_encoder(model_name: str):
    """Load each encoder once per process.

    Encoding is checkpointed in 1,000-text chunks; constructing the model per
    chunk re-read the weights from disk and re-initialized the device every time.
    """
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def _encode_texts(
    texts: List[str],
    model_name: str,
    normalize: bool = True,
    batch_size: int = BATCH_SIZE,
) -> np.ndarray:
    """Encode text batches without relying on an external utility script."""
    log.info("Encoding %d texts", len(texts))
    encoder = _load_encoder(model_name)
    return encoder.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=normalize,
    ).astype(np.float32)


def _post_embedding_fingerprint(post_texts: pd.DataFrame, model_name: str) -> str:
    """Fingerprint the exact ordered item/text pairs and encoder identifier."""
    hashed_rows = pd.util.hash_pandas_object(
        post_texts[["item_id", "text"]].astype(str), index=False
    ).to_numpy()
    digest = hashlib.sha256()
    digest.update(b"sef-post-embeddings-v1")
    digest.update(model_name.encode("utf-8"))
    digest.update(hashed_rows.tobytes())
    return digest.hexdigest()


def build_post_embeddings(
    post_texts: pd.DataFrame,
    embed_dir: Path,
    model: Optional[Any] = None,
    chunk_size: int = 1000,
) -> Tuple[np.ndarray, List[str]]:
    """Build post embeddings from the supplied data."""
    emb_path = embed_dir / "post_embeddings.npy"
    ids_path = embed_dir / "post_ids.json"
    model_name = model if model is not None else _get_model()
    model_name = str(model_name)
    manifest_path = embed_dir / "post_embeddings.manifest.json"
    df = post_texts.drop_duplicates(subset=["item_id"]).reset_index(drop=True)
    fingerprint = _post_embedding_fingerprint(df, model_name)
    partial_dir = embed_dir / "post_chunks" / fingerprint[:20]

    if emb_path.exists() and ids_path.exists() and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("fingerprint") == fingerprint:
            embeddings = np.load(emb_path)
            item_ids = json.loads(ids_path.read_text(encoding="utf-8"))
            if len(embeddings) == len(item_ids) == len(df):
                log.info("Loading input-matched cached post embeddings")
                return embeddings, item_ids
        log.warning("Post embedding cache does not match the configured dataset; rebuilding")
    elif emb_path.exists() or ids_path.exists():
        log.warning(
            "Legacy/incomplete post embedding cache has no verifiable manifest; rebuilding"
        )

    embed_dir.mkdir(parents=True, exist_ok=True)
    partial_dir.mkdir(parents=True, exist_ok=True)

    all_texts = df["text"].tolist()
    all_item_ids = df["item_id"].tolist()
    n_total = len(all_texts)
    if n_total == 0:
        raise ValueError("No non-null post texts are available for embedding")
    n_chunks = (n_total + chunk_size - 1) // chunk_size

    for ci in range(n_chunks):
        lo, hi = ci * chunk_size, min((ci + 1) * chunk_size, n_total)
        chunk_path = partial_dir / f"chunk_{ci:05d}.npy"
        if chunk_path.exists():
            try:
                cached = np.load(chunk_path, mmap_mode="r")
                if cached.ndim == 2 and len(cached) == hi - lo:
                    continue
            except (OSError, ValueError):
                pass
        vecs = _encode_texts(all_texts[lo:hi], model_name, normalize=True)
        np.save(chunk_path, vecs)

    chunks = [np.load(partial_dir / f"chunk_{i:05d}.npy") for i in range(n_chunks)]
    embeddings = np.concatenate(chunks, axis=0).astype(np.float32)
    np.save(emb_path, embeddings)
    ids_path.write_text(json.dumps(all_item_ids), encoding="utf-8")
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "fingerprint": fingerprint,
                "model": model_name,
                "n_items": len(all_item_ids),
                "embedding_dimension": int(embeddings.shape[1]),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    log.info("Post embeddings: %s", embeddings.shape)
    return embeddings, all_item_ids


def build_user_embeddings(
    user_docs: pd.DataFrame,
    embed_dir: Path,
    model: Optional[Any] = None,
    chunk_size: int = 1000,
) -> Tuple[np.ndarray, List[str]]:
    """Build legacy static user embeddings; canonical SEF uses temporal profiles."""
    emb_path = embed_dir / "user_embeddings.npy"
    ids_path = embed_dir / "user_ids.json"
    partial_dir = embed_dir / "user_chunks"

    if emb_path.exists() and ids_path.exists():
        log.info("Loading cached user embeddings")
        return np.load(emb_path), json.load(open(ids_path))

    if user_docs.empty:
        log.warning("No user documents found; user embeddings will be empty.")
        embed_dir.mkdir(parents=True, exist_ok=True)
        np.save(emb_path, np.zeros((0, EMBEDDING_DIM), dtype=np.float32))
        json.dump([], open(ids_path, "w"))
        return np.zeros((0, EMBEDDING_DIM), dtype=np.float32), []

    model_name = model if model is not None else _get_model()
    embed_dir.mkdir(parents=True, exist_ok=True)
    partial_dir.mkdir(exist_ok=True)

    user_docs = user_docs.reset_index(drop=True)
    all_texts = user_docs["text"].tolist()
    n_total = len(all_texts)
    n_chunks = (n_total + chunk_size - 1) // chunk_size
    done = len(sorted(partial_dir.glob("udoc_chunk_*.npy")))

    for ci in range(done, n_chunks):
        lo, hi = ci * chunk_size, min((ci + 1) * chunk_size, n_total)
        vecs = _encode_texts(all_texts[lo:hi], model_name, normalize=False)
        np.save(partial_dir / f"udoc_chunk_{ci:05d}.npy", vecs)

    chunks = [np.load(partial_dir / f"udoc_chunk_{i:05d}.npy") for i in range(n_chunks)]
    doc_embeddings = np.concatenate(chunks, axis=0).astype(np.float32)

    user_docs["_emb_idx"] = np.arange(len(user_docs))
    usernames, user_vecs = [], []
    for uname, group in tqdm(user_docs.groupby("username"), desc="Averaging user profiles"):
        mean_vec = doc_embeddings[group["_emb_idx"].values].mean(axis=0)
        norm = np.linalg.norm(mean_vec)
        if norm > 0:
            mean_vec /= norm
        usernames.append(uname)
        user_vecs.append(mean_vec)

    user_matrix = np.stack(user_vecs, axis=0).astype(np.float32)
    np.save(emb_path, user_matrix)
    json.dump(usernames, open(ids_path, "w"))
    log.info("User embeddings: %s", user_matrix.shape)
    return user_matrix, usernames


@dataclass(frozen=True)
class TemporalUserProfiles:
    """Materialized user profiles for the case cutoffs used by the benchmark."""

    profile_embeddings: np.ndarray
    cutoffs: np.ndarray
    document_counts: np.ndarray
    latest_document_timestamps: np.ndarray
    user_offsets: dict[str, tuple[int, int]]
    cache_key: str

    @property
    def n_users_with_history(self) -> int:
        """Count users having at least one materialized pre-case profile."""
        return sum(
            bool((self.document_counts[start:end] > 0).any())
            for start, end in self.user_offsets.values()
        )

    def lookup(
        self,
        usernames: List[str],
        cutoff: float,
        post_vector: np.ndarray,
    ) -> dict[str, tuple[float, int, float]]:
        """Return similarity, document count, and latest timestamp strictly before cutoff."""
        selected: list[tuple[str, int, int, float]] = []
        for username in dict.fromkeys(usernames):
            bounds = self.user_offsets.get(username)
            if bounds is None:
                continue
            start, end = bounds
            user_cutoffs = self.cutoffs[start:end]
            relative = int(np.searchsorted(user_cutoffs, cutoff, side="left"))
            if relative >= len(user_cutoffs) or not np.isclose(
                user_cutoffs[relative], cutoff, rtol=0.0, atol=1e-6
            ):
                raise KeyError(f"No materialized temporal profile for {username} at {cutoff}")
            index = start + relative
            count = int(self.document_counts[index])
            if count <= 0:
                continue
            latest = float(self.latest_document_timestamps[index])
            if not latest < cutoff:
                raise AssertionError(
                    f"Temporal SEF profile leaked for {username}: {latest} >= {cutoff}"
                )
            selected.append((username, index, count, latest))

        if not selected:
            return {}
        vectors = np.asarray(self.profile_embeddings[[entry[1] for entry in selected]])
        similarities = np.clip(vectors @ post_vector, 0.0, 1.0)
        return {
            username: (float(similarity), count, latest)
            for (username, _, count, latest), similarity in zip(selected, similarities)
        }


def _temporal_document_fingerprint(
    documents: pd.DataFrame,
    model_name: str,
    chunk_size: int,
) -> str:
    document_columns = ["username", "timestamp", "text"]
    document_hash = pd.util.hash_pandas_object(documents[document_columns], index=False).to_numpy()
    digest = hashlib.sha256()
    digest.update(b"temporal-sef-documents-v1")
    digest.update(model_name.encode("utf-8"))
    digest.update(str(chunk_size).encode("ascii"))
    digest.update(document_hash.tobytes())
    return digest.hexdigest()


def _temporal_profile_fingerprint(document_key: str, queries: pd.DataFrame) -> str:
    query_hash = pd.util.hash_pandas_object(queries, index=False).to_numpy()
    digest = hashlib.sha256()
    digest.update(b"temporal-sef-profiles-v1")
    digest.update(document_key.encode("ascii"))
    digest.update(query_hash.tobytes())
    return digest.hexdigest()


def _profile_queries(votes: pd.DataFrame) -> pd.DataFrame:
    """Return each distinct username/case cutoff requested by SEF."""
    required = {"username", "timestamp"}
    missing = required - set(votes.columns)
    if missing:
        raise ValueError(f"Temporal profile requests are missing columns: {sorted(missing)}")
    source_columns = ["username", "timestamp"]
    if "item_id" in votes:
        source_columns.insert(1, "item_id")
    work = votes[source_columns].copy()
    work["timestamp"] = unix_seconds(work["timestamp"])
    if work["timestamp"].isna().any():
        raise ValueError("Temporal profile requests contain invalid case timestamps")
    # Match the canonical chronological protocol: an item's case time is its
    # completion time (the maximum timestamp among its observed votes).
    if "item_id" in work:
        item_cutoffs = work.groupby("item_id", as_index=False)["timestamp"].max()
        queries = (
            work[["username", "item_id"]]
            .drop_duplicates()
            .merge(item_cutoffs, on="item_id", how="left", validate="many_to_one")
        )
    else:
        queries = work
    return (
        queries[["username", "timestamp"]]
        .drop_duplicates()
        .sort_values(["username", "timestamp"], kind="stable")
        .reset_index(drop=True)
    )


def collect_temporal_profile_requests(
    base_votes: pd.DataFrame,
    votes_dir: Path,
) -> pd.DataFrame:
    """Collect every user/item timestamp population used by canonical evaluations."""
    columns = ["username", "item_id", "timestamp"]
    query_frames = [_profile_queries(base_votes[columns])]
    for split_path in discover_splits(votes_dir).values():
        partition_frames = []
        for partition in ("train", "val", "test"):
            path = split_path / f"{partition}_votes.parquet"
            if path.exists():
                partition_frames.append(pd.read_parquet(path, columns=columns))
        if partition_frames:
            split_votes = pd.concat(partition_frames, ignore_index=True)
            query_frames.append(_profile_queries(split_votes))
    requests = pd.concat(query_frames, ignore_index=True).drop_duplicates()
    log.info(
        "Temporal profile request source: %d distinct user/cutoff pairs across base data and splits",
        len(requests),
    )
    return requests


def _load_document_embedding_slice(
    chunk_dir: Path,
    start: int,
    end: int,
    chunk_size: int,
) -> np.ndarray:
    """Load a contiguous embedding slice without materializing the full corpus."""
    pieces = []
    first_chunk = start // chunk_size
    last_chunk = (end - 1) // chunk_size
    for chunk_index in range(first_chunk, last_chunk + 1):
        chunk = np.load(chunk_dir / f"chunk_{chunk_index:05d}.npy", mmap_mode="r")
        chunk_start = chunk_index * chunk_size
        lower = max(start - chunk_start, 0)
        upper = min(end - chunk_start, len(chunk))
        pieces.append(np.asarray(chunk[lower:upper], dtype=np.float32))
    return np.concatenate(pieces, axis=0)


def build_temporal_user_profiles(
    user_docs: pd.DataFrame,
    profile_requests: pd.DataFrame,
    embed_dir: Path,
    model: Optional[Any] = None,
    chunk_size: int = 1000,
) -> TemporalUserProfiles:
    """Build exact pre-case profiles only for user/cutoff pairs used by SEF."""
    user_docs = user_docs.copy()
    if "timestamp" not in user_docs and "created_utc" in user_docs:
        user_docs = user_docs.rename(columns={"created_utc": "timestamp"})
    if "item_id" not in user_docs and "thing_id" in user_docs:
        user_docs = user_docs.rename(columns={"thing_id": "item_id"})
    required = {"username", "timestamp", "text"}
    missing = required - set(user_docs.columns)
    if missing:
        raise ValueError(f"Temporal user documents are missing columns: {sorted(missing)}")

    documents = user_docs.dropna(subset=list(required)).copy()
    documents["timestamp"] = unix_seconds(documents["timestamp"])
    documents = documents.dropna(subset=["timestamp"])
    dedupe = ["username", "item_id"] if "item_id" in documents else ["username", "text"]
    documents = (
        documents.drop_duplicates(dedupe)
        .sort_values(["username", "timestamp"], kind="stable")
        .reset_index(drop=True)
    )
    queries = _profile_queries(profile_requests)
    model_name = model if model is not None else _get_model()
    document_key = _temporal_document_fingerprint(documents, str(model_name), chunk_size)[:20]
    cache_key = _temporal_profile_fingerprint(document_key, queries)[:20]
    cache_dir = embed_dir / "temporal_user_profiles" / cache_key
    profiles_path = cache_dir / "profile_embeddings.npy"
    cutoffs_path = cache_dir / "cutoffs.npy"
    counts_path = cache_dir / "document_counts.npy"
    latest_path = cache_dir / "latest_document_timestamps.npy"
    offsets_path = cache_dir / "user_offsets.json"
    manifest_path = cache_dir / "manifest.json"
    artifacts = (
        profiles_path,
        cutoffs_path,
        counts_path,
        latest_path,
        offsets_path,
        manifest_path,
    )
    if all(path.exists() for path in artifacts):
        return _load_profile_directory(cache_dir, cache_key)

    if documents.empty or queries.empty:
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(profiles_path, np.zeros((0, EMBEDDING_DIM), dtype=np.float32))
        np.save(cutoffs_path, np.zeros(0, dtype=np.float64))
        np.save(counts_path, np.zeros(0, dtype=np.int32))
        np.save(latest_path, np.zeros(0, dtype=np.float64))
        offsets_path.write_text("{}", encoding="utf-8")
        manifest_path.write_text(
            json.dumps({"cache_key": cache_key, "n_documents": 0}, indent=2),
            encoding="utf-8",
        )
        return TemporalUserProfiles(
            np.zeros((0, EMBEDDING_DIM), dtype=np.float32),
            np.zeros(0, dtype=np.float64),
            np.zeros(0, dtype=np.int32),
            np.zeros(0, dtype=np.float64),
            {},
            cache_key,
        )

    cache_dir.mkdir(parents=True, exist_ok=True)
    chunk_dir = embed_dir / "temporal_document_embeddings" / document_key / "chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    n_documents = len(documents)
    n_chunks = (n_documents + chunk_size - 1) // chunk_size
    for chunk_index in range(n_chunks):
        chunk_path = chunk_dir / f"chunk_{chunk_index:05d}.npy"
        lower = chunk_index * chunk_size
        upper = min((chunk_index + 1) * chunk_size, n_documents)
        expected_rows = upper - lower
        if chunk_path.exists():
            try:
                cached = np.load(chunk_path, mmap_mode="r")
                if cached.ndim == 2 and len(cached) == expected_rows:
                    continue
            except (OSError, ValueError):
                pass
        vectors = _encode_texts(
            documents.iloc[lower:upper]["text"].tolist(),
            str(model_name),
            normalize=False,
        )
        temporary_path = chunk_path.with_suffix(".tmp.npy")
        np.save(temporary_path, vectors)
        temporary_path.replace(chunk_path)
    embedding_dimension = int(np.load(chunk_dir / "chunk_00000.npy", mmap_mode="r").shape[1])
    profiles = np.lib.format.open_memmap(
        profiles_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(queries), embedding_dimension),
    )
    profiles[:] = 0.0
    cutoffs = queries["timestamp"].to_numpy(dtype=np.float64)
    counts = np.zeros(len(queries), dtype=np.int32)
    latest = np.full(len(queries), np.nan, dtype=np.float64)
    user_offsets: dict[str, tuple[int, int]] = {}
    document_groups = documents.groupby("username", sort=False).indices
    for username, query_indices in queries.groupby("username", sort=False).indices.items():
        query_positions = np.asarray(query_indices, dtype=int)
        query_start, query_end = int(query_positions[0]), int(query_positions[-1]) + 1
        user_offsets[str(username)] = (query_start, query_end)
        document_indices = document_groups.get(username)
        if document_indices is None:
            continue
        positions = np.asarray(document_indices, dtype=int)
        document_start, document_end = int(positions[0]), int(positions[-1]) + 1
        vectors = _load_document_embedding_slice(
            chunk_dir, document_start, document_end, chunk_size
        )
        cumulative = np.cumsum(vectors, axis=0)
        document_times = documents.iloc[document_start:document_end]["timestamp"].to_numpy(
            dtype=np.float64
        )
        user_cutoffs = cutoffs[query_start:query_end]
        user_counts = np.searchsorted(document_times, user_cutoffs, side="left")
        for relative, count in enumerate(user_counts):
            if count <= 0:
                continue
            profile_index = query_start + relative
            vector = cumulative[int(count) - 1]
            norm = np.linalg.norm(vector)
            if norm > 0:
                profiles[profile_index] = vector / norm
            counts[profile_index] = int(count)
            latest[profile_index] = float(document_times[int(count) - 1])
    profiles.flush()
    np.save(cutoffs_path, cutoffs)
    np.save(counts_path, counts)
    np.save(latest_path, latest)
    offsets_path.write_text(json.dumps(user_offsets), encoding="utf-8")
    manifest_path.write_text(
        json.dumps(
            {
                "cache_key": cache_key,
                "document_cache_key": document_key,
                "model": str(model_name),
                "n_documents": n_documents,
                "n_profile_queries": len(queries),
                "n_users": len(user_offsets),
                "n_profiles_with_history": int((counts > 0).sum()),
                "timestamp_rule": "document.timestamp < max_case_vote_timestamp",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    log.info(
        "Temporal user profiles: %d queries, %d with prior documents, %d users",
        len(queries),
        int((counts > 0).sum()),
        len(user_offsets),
    )
    return TemporalUserProfiles(
        profile_embeddings=np.load(profiles_path, mmap_mode="r"),
        cutoffs=np.load(cutoffs_path, mmap_mode="r"),
        document_counts=np.load(counts_path, mmap_mode="r"),
        latest_document_timestamps=np.load(latest_path, mmap_mode="r"),
        user_offsets=user_offsets,
        cache_key=cache_key,
    )


PROFILE_FILES = (
    "profile_embeddings.npy",
    "cutoffs.npy",
    "document_counts.npy",
    "latest_document_timestamps.npy",
    "user_offsets.json",
    "manifest.json",
)
PREPARED_CACHE_MANIFEST = "prepared_cache.json"


def _load_profile_directory(cache_dir: Path, cache_key: str) -> TemporalUserProfiles:
    """Memory-map one materialized temporal-profile cache directory."""
    offsets_payload = json.loads((cache_dir / "user_offsets.json").read_text(encoding="utf-8"))
    return TemporalUserProfiles(
        profile_embeddings=np.load(cache_dir / "profile_embeddings.npy", mmap_mode="r"),
        cutoffs=np.load(cache_dir / "cutoffs.npy", mmap_mode="r"),
        document_counts=np.load(cache_dir / "document_counts.npy", mmap_mode="r"),
        latest_document_timestamps=np.load(
            cache_dir / "latest_document_timestamps.npy", mmap_mode="r"
        ),
        user_offsets={key: tuple(value) for key, value in offsets_payload.items()},
        cache_key=cache_key,
    )


def write_prepared_cache_manifest(
    embed_dir: Path,
    *,
    profiles: TemporalUserProfiles,
    n_posts: int,
    model: str,
    input_hashes: Dict[str, Optional[str]],
    splits_roots: List[str],
) -> Path:
    """Record which post-embedding/profile cache a prepare run produced.

    Per-split SEF processes and Team Formation read this instead of re-hashing
    the (potentially large) text tables or recomputing profile requests.
    Profiles are materialized for the union of all requested splits, and
    lookups are exact per (user, cutoff), so one cache serves every split.
    """
    payload = {
        "schema_version": 1,
        "model": model,
        "n_posts": n_posts,
        "profile_cache_key": profiles.cache_key,
        "input_sha256": input_hashes,
        "splits_roots": splits_roots,
    }
    path = embed_dir / PREPARED_CACHE_MANIFEST
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)
    return path


def load_prepared_cache(
    embed_dir: Path,
    *,
    model: str,
    input_hashes: Dict[str, Optional[str]],
) -> Tuple[np.ndarray, List[str], TemporalUserProfiles]:
    """Load the post embeddings and temporal profiles built by ``prepare sef-cache``."""
    path = embed_dir / PREPARED_CACHE_MANIFEST
    if not path.is_file():
        raise FileNotFoundError(
            f"No prepared SEF cache at {path}; run `python -m src.pipeline prepare sef-cache`"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    stale = [
        name
        for name, digest in input_hashes.items()
        if payload.get("input_sha256", {}).get(name) != digest
    ]
    if payload.get("model") != model or stale:
        raise ValueError(
            f"Prepared SEF cache {path} was built from different inputs "
            f"({', '.join(stale) or 'model'}); rerun `prepare sef-cache --force`"
        )
    embeddings = np.load(embed_dir / "post_embeddings.npy", mmap_mode="r")
    item_ids = json.loads((embed_dir / "post_ids.json").read_text(encoding="utf-8"))
    if len(embeddings) != len(item_ids) or len(item_ids) != payload.get("n_posts"):
        raise ValueError(f"Prepared post embeddings under {embed_dir} are incomplete")
    cache_key = payload["profile_cache_key"]
    cache_dir = embed_dir / "temporal_user_profiles" / cache_key
    if not all((cache_dir / name).exists() for name in PROFILE_FILES):
        raise FileNotFoundError(f"Prepared temporal profile cache is incomplete: {cache_dir}")
    return (
        np.asarray(embeddings, dtype=np.float32),
        item_ids,
        _load_profile_directory(cache_dir, cache_key),
    )


# 3. FAISS INDEX


def build_faiss_index(embeddings: np.ndarray, *, announce: bool = False) -> faiss.Index:
    """Build faiss index from the supplied data."""
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    if announce:
        log.info("FAISS index ready: %d vectors", index.ntotal)
    return index
