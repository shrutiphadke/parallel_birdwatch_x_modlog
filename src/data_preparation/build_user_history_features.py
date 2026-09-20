"""Build causal user-history features from local Reddit JSONL data.

Stages index authors, build monthly history snapshots, and join strictly prior
states to votes. Each stage is resumable; ``--rebuild-index`` forces reindexing.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import sqlite3
from typing import Dict, Iterable, Iterator, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.utils.tabular import read_table, unix_seconds

try:
    from tqdm import tqdm
except ImportError:

    class _NoProgress:
        def __init__(self, iterable=None, **_kwargs):
            self.iterable = iterable

        def __iter__(self):
            return iter(self.iterable or [])

        def update(self, *_args, **_kwargs):
            pass

        def set_postfix_str(self, *_args, **_kwargs):
            pass

        def close(self):
            pass

    def tqdm(iterable=None, **kwargs):
        return _NoProgress(iterable, **kwargs)


try:
    import orjson

    def _loads(value):
        return orjson.loads(value)
except ImportError:

    def _loads(value):
        return json.loads(value)


DEFAULT_INPUT_DIR = Path("data/raw/user_contributions/user_contributions")
DEFAULT_AUTHOR_INDEX = Path("data/interim/reddit/build_state/author_index.sqlite")
DEFAULT_FEATURE_ROOT = Path("data/interim/reddit/build_state/user_temporal_features")
DEFAULT_VOTES = Path("data/processed/final_intersection_dataset.csv")
DEFAULT_METADATA = Path("data/processed/user_metadata.csv")
DEFAULT_POST_TEXTS = Path("data/interim/reddit/auxiliary/post_texts.parquet")
DEFAULT_OUTPUT = Path("data/interim/reddit/features/causal_user_vote_features.parquet")

FIELD_ID = "id"
FIELD_NAME = "name"
FIELD_BODY = "body"
FIELD_TITLE = "title"
FIELD_PARENT = "parent_id"
FIELD_SUBREDDIT = "subreddit"
FIELD_CREATED_UTC = "created_utc"
FIELD_SCORE = "score"

INDEX_BATCH_SIZE = 50_000
USERS_PER_PART = 200
SQLITE_IN_CHUNK = 900
KEY_COLUMNS = ["username", "item_id", "timestamp"]

GLOBAL_SCHEMA = pa.schema(
    [
        ("username", pa.string()),
        ("year_month", pa.string()),
        ("cum_n_posts", pa.int64()),
        ("cum_n_comments", pa.int64()),
        ("cum_n_contributions", pa.int64()),
        ("cum_karma_sum", pa.float64()),
        ("cum_n_distinct_subreddits", pa.int64()),
        ("cum_n_replies_made", pa.int64()),
    ]
)
SUBREDDIT_SCHEMA = pa.schema(
    [
        ("username", pa.string()),
        ("subreddit", pa.string()),
        ("year_month", pa.string()),
        ("cum_n_posts_in_sub", pa.int64()),
        ("cum_n_comments_in_sub", pa.int64()),
        ("cum_karma_sum_in_sub", pa.float64()),
        ("cum_n_months_active_in_sub", pa.int64()),
    ]
)
REPLY_SCHEMA = pa.schema(
    [
        ("from_user", pa.string()),
        ("to_user", pa.string()),
        ("item_id", pa.string()),
        ("parent_id", pa.string()),
        ("subreddit", pa.string()),
        ("timestamp", pa.float64()),
    ]
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def iter_user_files(input_dir: Path) -> Iterator[Tuple[str, Path]]:
    """Yield sampled usernames and JSONL paths in stable order."""
    for path in sorted(input_dir.glob("*.jsonl")):
        yield path.stem, path


def infer_fullname(record: dict) -> Optional[str]:
    """Return a canonical t1/t3 fullname for an indexable contribution."""
    name = record.get(FIELD_NAME)
    if isinstance(name, str) and (name.startswith("t1_") or name.startswith("t3_")):
        return name
    bare_id = record.get(FIELD_ID)
    if not bare_id:
        return None
    bare_id = str(bare_id)
    if bare_id.startswith("t1_") or bare_id.startswith("t3_"):
        return bare_id
    if FIELD_BODY in record and FIELD_PARENT in record:
        return f"t1_{bare_id}"
    if FIELD_TITLE in record:
        return f"t3_{bare_id}"
    return None


def peek_schema(input_dir: Path, n_files: int = 2, n_lines: int = 3) -> None:
    """Print a bounded JSONL schema preview without modifying the index."""
    print("Schema preview")
    for file_index, (username, path) in enumerate(iter_user_files(input_dir)):
        if file_index >= n_files:
            break
        print(f"File: {path.name} | author={username}")
        with path.open("r", encoding="utf-8") as handle:
            for line_index, line in enumerate(handle):
                if line_index >= n_lines:
                    break
                try:
                    record = _loads(line)
                    print(
                        f"  Line {line_index}: keys={sorted(record)}; "
                        f"fullname={infer_fullname(record)}"
                    )
                except Exception as exc:
                    print(f"  Line {line_index}: invalid JSON ({exc})")


def validate_author_index(output_db: Path, input_dir: Optional[Path] = None) -> dict:
    """Validate SQLite integrity, table schema, content, and completion checkpoint."""
    report = {
        "path": output_db.as_posix(),
        "exists": output_db.is_file(),
        "size_bytes": output_db.stat().st_size if output_db.is_file() else 0,
        "integrity": None,
        "schema_valid": False,
        "has_rows": False,
        "n_input_users": None,
        "n_completed_users": 0,
        "complete": False,
        "valid": False,
    }
    if not output_db.is_file():
        return report

    try:
        connection = sqlite3.connect(f"file:{output_db.as_posix()}?mode=ro", uri=True)
        report["integrity"] = connection.execute("PRAGMA quick_check").fetchone()[0]
        columns = connection.execute("PRAGMA table_info(author_index)").fetchall()
        expected = [("thing_id", "TEXT", 0, 1), ("author", "TEXT", 1, 0)]
        observed = [(row[1], row[2].upper(), row[3], row[5]) for row in columns]
        report["schema_valid"] = observed == expected
        if report["schema_valid"]:
            report["has_rows"] = (
                connection.execute("SELECT 1 FROM author_index LIMIT 1").fetchone() is not None
            )
        connection.close()
    except sqlite3.DatabaseError as exc:
        report["error"] = str(exc)
        return report

    done_path = output_db.with_suffix(".done_users.txt")
    if done_path.exists():
        completed = {line for line in done_path.read_text(encoding="utf-8").splitlines() if line}
        report["n_completed_users"] = len(completed)
    if input_dir is not None and input_dir.is_dir():
        report["n_input_users"] = sum(1 for _ in input_dir.glob("*.jsonl"))
        report["complete"] = report["n_completed_users"] == report["n_input_users"]
    else:
        report["complete"] = report["n_completed_users"] > 0
    report["valid"] = bool(
        report["integrity"] == "ok"
        and report["schema_valid"]
        and report["has_rows"]
        and report["complete"]
    )
    return report


def build_author_index(input_dir: Path, output_db: Path, rebuild: bool = False) -> dict:
    """Build/resume the author index, reusing a complete valid database by default."""
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    all_files = list(iter_user_files(input_dir))
    if not all_files:
        raise FileNotFoundError(f"No JSONL files found in {input_dir}")

    before = validate_author_index(output_db, input_dir)
    if before["valid"] and not rebuild:
        print(f"Valid author index already complete; reusing {output_db}")
        manifest = {"stage": "index", "action": "reused", **before}
        _write_json(output_db.with_suffix(".manifest.json"), manifest)
        return manifest
    if output_db.exists() and not before["schema_valid"] and not rebuild:
        raise RuntimeError(
            f"Existing author index failed schema validation: {output_db}. "
            "Inspect it or pass --rebuild-index explicitly."
        )

    output_db.parent.mkdir(parents=True, exist_ok=True)
    done_path = output_db.with_suffix(".done_users.txt")
    if rebuild:
        for path in (output_db, Path(f"{output_db}-wal"), Path(f"{output_db}-shm"), done_path):
            path.unlink(missing_ok=True)

    done_users = (
        set(done_path.read_text(encoding="utf-8").splitlines()) if done_path.exists() else set()
    )
    todo = [(username, path) for username, path in all_files if username not in done_users]
    print(f"Index users: {len(all_files)} total | {len(todo)} pending")

    connection = sqlite3.connect(output_db)
    connection.execute("PRAGMA journal_mode = WAL")
    connection.execute("PRAGMA synchronous = OFF")
    connection.execute(
        "CREATE TABLE IF NOT EXISTS author_index (thing_id TEXT PRIMARY KEY, author TEXT NOT NULL)"
    )
    connection.commit()
    inserted = 0
    skipped = 0
    batch: list[tuple[str, str]] = []
    with done_path.open("a", encoding="utf-8") as done_handle:
        for username, path in tqdm(todo, desc="Building author index"):
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    try:
                        fullname = infer_fullname(_loads(line))
                    except Exception:
                        fullname = None
                    if fullname is None:
                        skipped += 1
                        continue
                    batch.append((fullname, username))
                    if len(batch) >= INDEX_BATCH_SIZE:
                        connection.executemany(
                            "INSERT OR REPLACE INTO author_index (thing_id, author) VALUES (?, ?)",
                            batch,
                        )
                        connection.commit()
                        inserted += len(batch)
                        batch.clear()
            if batch:
                connection.executemany(
                    "INSERT OR REPLACE INTO author_index (thing_id, author) VALUES (?, ?)",
                    batch,
                )
                connection.commit()
                inserted += len(batch)
                batch.clear()
            done_handle.write(username + "\n")
            done_handle.flush()
    connection.execute("PRAGMA synchronous = NORMAL")
    connection.close()

    after = validate_author_index(output_db, input_dir)
    if not after["valid"]:
        raise RuntimeError(f"Author index failed post-build validation: {after}")
    manifest = {
        "stage": "index",
        "action": "rebuilt" if rebuild else "resumed_or_built",
        "rows_written_this_run": inserted,
        "rows_skipped_this_run": skipped,
        **after,
    }
    _write_json(output_db.with_suffix(".manifest.json"), manifest)
    return manifest


def infer_type_and_fullname(record: dict) -> Tuple[Optional[str], Optional[str]]:
    """Return contribution type and canonical fullname."""
    fullname = infer_fullname(record)
    if fullname is None:
        return None, None
    if fullname.startswith("t3_"):
        return "post", fullname
    if fullname.startswith("t1_"):
        return "comment", fullname
    return None, None


def year_month(timestamp: float) -> str:
    dt = datetime.fromtimestamp(float(timestamp), tz=timezone.utc)
    return f"{dt.year:04d}-{dt.month:02d}"


def load_and_sort_user_contributions(path: Path) -> List[dict]:
    """Load valid contributions for one user and sort them chronologically."""
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                record = _loads(line)
            except Exception:
                continue
            contribution_type, fullname = infer_type_and_fullname(record)
            if contribution_type is None or fullname is None:
                continue
            try:
                timestamp = float(record.get(FIELD_CREATED_UTC))
                year_month(timestamp)
            except (TypeError, ValueError, OverflowError, OSError):
                continue
            if not math.isfinite(timestamp):
                continue
            try:
                score = float(record.get(FIELD_SCORE, 0) or 0)
            except (TypeError, ValueError):
                score = 0.0
            if not math.isfinite(score):
                score = 0.0
            rows.append(
                {
                    "type": contribution_type,
                    "fullname": fullname,
                    "subreddit": record.get(FIELD_SUBREDDIT) or "__unknown__",
                    "timestamp": timestamp,
                    "score": score,
                    "parent_id": record.get(FIELD_PARENT),
                }
            )
    rows.sort(key=lambda row: row["timestamp"])
    return rows


def batch_lookup_authors(
    parent_ids: Set[str],
    author_index: sqlite3.Connection,
    chunk_size: int = SQLITE_IN_CHUNK,
) -> Dict[str, str]:
    """Resolve parent authors in bounded SQLite IN queries."""
    result: Dict[str, str] = {}
    ids = list(parent_ids)
    for offset in range(0, len(ids), chunk_size):
        chunk = ids[offset : offset + chunk_size]
        placeholders = ",".join("?" for _ in chunk)
        rows = author_index.execute(
            f"SELECT thing_id, author FROM author_index WHERE thing_id IN ({placeholders})",
            chunk,
        )
        result.update(rows)
    return result


class PartitionedParquetWriter:
    """Buffer rows and write independently readable Parquet parts."""

    def __init__(self, output_dir: Path, schema: pa.Schema, start_part: int = 0):
        self.output_dir = output_dir
        self.schema = schema
        self.part_index = start_part
        self.buffer: List[dict] = []
        output_dir.mkdir(parents=True, exist_ok=True)

    def add_rows(self, rows: List[dict]) -> None:
        self.buffer.extend(rows)

    def flush(self) -> None:
        if not self.buffer:
            return
        table = pa.Table.from_pylist(self.buffer, schema=self.schema)
        pq.write_table(table, self.output_dir / f"part_{self.part_index:06d}.parquet")
        self.part_index += 1
        self.buffer.clear()


def _next_part_index(output_dir: Path) -> int:
    parts = sorted(output_dir.glob("part_*.parquet")) if output_dir.exists() else []
    return int(parts[-1].stem.rsplit("_", 1)[1]) + 1 if parts else 0


def process_one_user(
    username: str,
    contributions: List[dict],
    author_lookup: Dict[str, str],
) -> Tuple[List[dict], List[dict], List[dict]]:
    """Replay one user's contributions into monthly cumulative states and edges."""
    global_rows: List[dict] = []
    subreddit_rows: List[dict] = []
    reply_rows: List[dict] = []
    n_posts = n_comments = n_replies = 0
    karma_sum = 0.0
    subreddits_seen: Set[str] = set()
    sub_state: Dict[str, Dict[str, float]] = defaultdict(
        lambda: {"n_posts": 0, "n_comments": 0, "karma_sum": 0.0, "n_months": 0}
    )
    current_month: Optional[str] = None
    touched_this_month: Set[str] = set()

    def flush_month(month: Optional[str]) -> None:
        if month is None:
            return
        global_rows.append(
            {
                "username": username,
                "year_month": month,
                "cum_n_posts": n_posts,
                "cum_n_comments": n_comments,
                "cum_n_contributions": n_posts + n_comments,
                "cum_karma_sum": karma_sum,
                "cum_n_distinct_subreddits": len(subreddits_seen),
                "cum_n_replies_made": n_replies,
            }
        )
        for subreddit in touched_this_month:
            state = sub_state[subreddit]
            subreddit_rows.append(
                {
                    "username": username,
                    "subreddit": subreddit,
                    "year_month": month,
                    "cum_n_posts_in_sub": int(state["n_posts"]),
                    "cum_n_comments_in_sub": int(state["n_comments"]),
                    "cum_karma_sum_in_sub": state["karma_sum"],
                    "cum_n_months_active_in_sub": int(state["n_months"]),
                }
            )

    for contribution in contributions:
        month = year_month(contribution["timestamp"])
        if current_month is None:
            current_month = month
        elif month != current_month:
            flush_month(current_month)
            touched_this_month = set()
            current_month = month

        subreddit = contribution["subreddit"]
        score = float(contribution["score"] or 0)
        if contribution["type"] == "post":
            n_posts += 1
            sub_state[subreddit]["n_posts"] += 1
        else:
            n_comments += 1
            sub_state[subreddit]["n_comments"] += 1
        karma_sum += score
        sub_state[subreddit]["karma_sum"] += score
        if subreddit not in touched_this_month:
            touched_this_month.add(subreddit)
            sub_state[subreddit]["n_months"] += 1
        subreddits_seen.add(subreddit)

        parent_id = contribution["parent_id"]
        if contribution["type"] == "comment" and parent_id:
            other_user = author_lookup.get(parent_id)
            if other_user is not None and other_user != username:
                n_replies += 1
                reply_rows.append(
                    {
                        "from_user": username,
                        "to_user": other_user,
                        "item_id": contribution["fullname"],
                        "parent_id": parent_id,
                        "subreddit": subreddit,
                        "timestamp": contribution["timestamp"],
                    }
                )
    flush_month(current_month)
    return global_rows, subreddit_rows, reply_rows


_WORKER_AUTHOR_INDEX_CONNECTIONS: Dict[str, sqlite3.Connection] = {}


def _worker_author_index_connection(author_index_path: Path) -> sqlite3.Connection:
    """Return one read-only connection per (worker process, index), reused across users."""
    key = author_index_path.as_posix()
    connection = _WORKER_AUTHOR_INDEX_CONNECTIONS.get(key)
    if connection is None:
        connection = sqlite3.connect(f"file:{key}?mode=ro", uri=True)
        _WORKER_AUTHOR_INDEX_CONNECTIONS[key] = connection
    return connection


def _replay_one_user_file(
    path: Path, author_index_path: Path
) -> Tuple[str, List[dict], List[dict], List[dict]]:
    """Load, resolve parent authors, and replay one user's file (parallel worker entry point)."""
    username = path.stem
    contributions = load_and_sort_user_contributions(path)
    parent_ids = {
        row["parent_id"]
        for row in contributions
        if row["type"] == "comment" and row["parent_id"]
    }
    lookup = (
        batch_lookup_authors(parent_ids, _worker_author_index_connection(author_index_path))
        if parent_ids
        else {}
    )
    global_rows, subreddit_rows, reply_rows = process_one_user(username, contributions, lookup)
    return username, global_rows, subreddit_rows, reply_rows


def _snapshot_output_report(feature_root: Path) -> dict:
    outputs = {}
    for name in ("user_global_snapshots", "user_subreddit_snapshots", "reply_graph"):
        parts = sorted((feature_root / name).glob("part_*.parquet"))
        outputs[name] = {
            "n_parts": len(parts),
            "size_bytes": sum(path.stat().st_size for path in parts),
        }
    return outputs


def build_snapshots(
    input_dir: Path,
    author_index_path: Path,
    feature_root: Path,
    rebuild: bool = False,
    index_report: Optional[dict] = None,
) -> dict:
    """Build/resume monthly snapshots and reply graph from local JSONL files."""
    if index_report is None:
        index_report = validate_author_index(author_index_path, input_dir)
    if not index_report["valid"]:
        raise RuntimeError(f"Author index is not complete and valid: {index_report}")

    all_files = sorted(input_dir.glob("*.jsonl"))
    if not all_files:
        raise FileNotFoundError(f"No JSONL files found in {input_dir}")
    feature_root.mkdir(parents=True, exist_ok=True)
    global_dir = feature_root / "user_global_snapshots"
    subreddit_dir = feature_root / "user_subreddit_snapshots"
    reply_dir = feature_root / "reply_graph"
    progress_path = feature_root / "_progress_users_done.txt"

    if rebuild:
        for path in (global_dir, subreddit_dir, reply_dir):
            if path.exists():
                shutil.rmtree(path)
        progress_path.unlink(missing_ok=True)
    elif progress_path.exists() and not all(
        path.is_dir() for path in (global_dir, subreddit_dir, reply_dir)
    ):
        raise RuntimeError(
            f"Incomplete snapshot resume state in {feature_root}; use --rebuild-snapshots"
        )
    elif not progress_path.exists() and any(
        next(path.glob("part_*.parquet"), None) is not None
        for path in (global_dir, subreddit_dir, reply_dir)
        if path.is_dir()
    ):
        raise RuntimeError(
            f"Snapshot parts exist without a checkpoint in {feature_root}; use --rebuild-snapshots"
        )

    done_users = (
        set(progress_path.read_text(encoding="utf-8").splitlines())
        if progress_path.exists()
        else set()
    )
    todo = [path for path in all_files if path.stem not in done_users]
    print(f"Snapshot users: {len(all_files)} total | {len(todo)} pending")

    global_writer = PartitionedParquetWriter(
        global_dir, GLOBAL_SCHEMA, _next_part_index(global_dir)
    )
    subreddit_writer = PartitionedParquetWriter(
        subreddit_dir, SUBREDDIT_SCHEMA, _next_part_index(subreddit_dir)
    )
    reply_writer = PartitionedParquetWriter(reply_dir, REPLY_SCHEMA, _next_part_index(reply_dir))
    pending_users: List[str] = []
    n_workers = min(os.cpu_count() or 1, len(todo))
    with progress_path.open("a", encoding="utf-8") as progress_handle:
        completed_count = 0

        def flush_part() -> None:
            global_writer.flush()
            subreddit_writer.flush()
            reply_writer.flush()
            if pending_users:
                progress_handle.write("\n".join(pending_users) + "\n")
                progress_handle.flush()
                pending_users.clear()

        def consume(
            username: str,
            global_rows: List[dict],
            subreddit_rows: List[dict],
            reply_rows: List[dict],
        ) -> None:
            nonlocal completed_count
            global_writer.add_rows(global_rows)
            subreddit_writer.add_rows(subreddit_rows)
            reply_writer.add_rows(reply_rows)
            pending_users.append(username)
            completed_count += 1
            if completed_count % USERS_PER_PART == 0:
                flush_part()

        # Each user's contribution file is replayed independently (only reading a
        # shared read-only author index), so this parallelizes cleanly across
        # processes; results are consumed as they complete, in any order, since
        # snapshot rows carry no cross-user ordering dependency.
        if n_workers > 1:
            ctx = mp.get_context("spawn")
            with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as pool:
                futures = [
                    pool.submit(_replay_one_user_file, path, author_index_path) for path in todo
                ]
                for future in tqdm(
                    as_completed(futures), total=len(futures), desc="Replaying contributions"
                ):
                    consume(*future.result())
        else:
            for path in tqdm(todo, desc="Replaying contributions"):
                consume(*_replay_one_user_file(path, author_index_path))
        flush_part()

    completed = set(progress_path.read_text(encoding="utf-8").splitlines())
    if len(completed) != len(all_files):
        raise RuntimeError(
            f"Snapshot checkpoint covers {len(completed)} of {len(all_files)} users"
        )
    report = {
        "stage": "snapshots",
        "action": "rebuilt" if rebuild else ("reused" if not todo else "resumed_or_built"),
        "input_dir": input_dir.as_posix(),
        "author_index": author_index_path.as_posix(),
        "n_input_users": len(all_files),
        "n_completed_users": len(completed),
        "outputs": _snapshot_output_report(feature_root),
    }
    _write_json(feature_root / "snapshots.manifest.json", report)
    return report


def audit_timestamp_semantics(
    votes: pd.DataFrame,
    post_texts_path: Path,
    override: str = "auto",
) -> dict:
    """Classify the vote timestamp field and persist the evidence."""
    timestamp = unix_seconds(votes["timestamp"])
    per_item_counts = (
        votes.assign(_timestamp=timestamp).groupby("item_id")["_timestamp"].nunique(dropna=True)
    )
    report = {
        "requested_semantics": override,
        "share_items_with_one_timestamp": float((per_item_counts <= 1).mean()),
        "post_creation_match_rate": None,
        "n_rows_compared_to_post_creation": 0,
    }
    if post_texts_path.exists():
        posts = read_table(post_texts_path)
        if "timestamp" not in posts and "created_utc" in posts:
            posts = posts.rename(columns={"created_utc": "timestamp"})
        missing = {"item_id", "timestamp"} - set(posts.columns)
        if missing:
            raise ValueError(
                f"{post_texts_path} is missing timestamp-audit columns: {sorted(missing)}"
            )
        posts = posts[["item_id", "timestamp"]]
        posts = posts.drop_duplicates("item_id", keep="last").copy()
        left = votes[["item_id"]].copy()
        left["item_id"] = left["item_id"].astype(str)
        left["_timestamp"] = timestamp
        posts["item_id"] = posts["item_id"].astype(str)
        posts["timestamp"] = unix_seconds(posts["timestamp"])
        compared = left.merge(posts, on="item_id", how="inner", validate="many_to_one")
        compared = compared.dropna(subset=["_timestamp", "timestamp"])
        report["n_rows_compared_to_post_creation"] = int(len(compared))
        if len(compared):
            report["post_creation_match_rate"] = float(
                ((compared["_timestamp"] - compared["timestamp"]).abs() <= 1.0).mean()
            )
    if override != "auto":
        semantics = override
    elif (
        report["post_creation_match_rate"] is not None
        and report["post_creation_match_rate"] >= 0.90
    ):
        semantics = "post_creation_proxy"
    elif report["share_items_with_one_timestamp"] >= 0.95:
        semantics = "likely_item_level_proxy_unknown_origin"
    else:
        semantics = "unknown_or_mixed"
    report["resolved_semantics"] = semantics
    return report


def _read_parquet_parts(
    path: Path, columns: Optional[list[str]] = None, allow_empty: bool = False
) -> pd.DataFrame:
    schema_path = (
        next(path.glob("*.parquet"), None) if path.is_dir() else (path if path.exists() else None)
    )
    if schema_path is None:
        if allow_empty:
            # e.g. a reply graph with zero edges when contribution histories hold no
            # comments (posts only) — a valid, empty result, not a failed build stage.
            return pd.DataFrame(columns=columns or [])
        if not path.exists():
            raise FileNotFoundError(path)
        raise ValueError(f"No Parquet parts found under {path}")
    if columns is None:
        return pd.read_parquet(path)
    available = set(pq.ParquetFile(schema_path).schema_arrow.names)
    aliases = {"timestamp": "created_utc", "item_id": "thing_id"}
    projected = [
        column if column in available else aliases.get(column, column) for column in columns
    ]
    frame = pd.read_parquet(path, columns=projected)
    return frame.rename(
        columns={alias: canonical for canonical, alias in aliases.items() if alias in frame}
    )


def _user_key(values: pd.Series) -> pd.Series:
    return values.map(lambda value: str(value).casefold() if pd.notna(value) else "").astype(
        object
    )


def _subreddit_key(values: pd.Series) -> pd.Series:
    def normalize(value) -> str:
        if pd.isna(value):
            return ""
        text = str(value).strip()
        return (text[2:] if text.casefold().startswith("r/") else text).casefold()

    return values.map(normalize).astype(object)


def _month_available_at(values: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(values.astype("string") + "-01", utc=True, errors="coerce")
    available = parsed + pd.offsets.MonthBegin(1)
    return available.map(lambda value: value.timestamp() if pd.notna(value) else np.nan)


def _asof_join(
    left: pd.DataFrame,
    right: pd.DataFrame,
    by: list[str],
    right_time: str,
    feature_columns: Iterable[str],
) -> pd.DataFrame:
    """Perform a strictly-prior as-of join and restore original vote order."""
    feature_columns = list(feature_columns)
    left = left.copy()
    right = right.copy()
    left["timestamp"] = unix_seconds(left["timestamp"])
    right[right_time] = unix_seconds(right[right_time])
    for column in by:
        left[column] = left[column].astype(object)
        right[column] = right[column].astype(object)
    right = right[by + [right_time] + feature_columns].dropna(subset=by + [right_time])
    right = right.sort_values([right_time] + by).drop_duplicates(by + [right_time], keep="last")
    ordered = left.sort_values(["timestamp"] + by)
    joined = pd.merge_asof(
        ordered,
        right,
        left_on="timestamp",
        right_on=right_time,
        by=by,
        direction="backward",
        allow_exact_matches=False,
    )
    return joined.sort_values("_vote_row").reset_index(drop=True)


def _build_interaction_events(
    reply_graph: pd.DataFrame,
    by_subreddit: bool = False,
) -> pd.DataFrame:
    """Create cumulative undirected interaction states from reply events."""
    edges = reply_graph.dropna(subset=["from_user", "to_user", "timestamp"]).copy()
    edges["timestamp"] = unix_seconds(edges["timestamp"])
    edges = edges.dropna(subset=["timestamp"])
    base_columns = ["from_user", "to_user", "timestamp"]
    if by_subreddit:
        base_columns.append("subreddit")
    outgoing = edges[base_columns].rename(columns={"from_user": "_user_key", "to_user": "partner"})
    incoming_columns = ["to_user", "from_user", "timestamp"]
    if by_subreddit:
        incoming_columns.append("subreddit")
    incoming = edges[incoming_columns].rename(
        columns={"to_user": "_user_key", "from_user": "partner"}
    )
    events = pd.concat([outgoing, incoming], ignore_index=True)
    events["_user_key"] = _user_key(events["_user_key"])
    events["partner"] = _user_key(events["partner"])
    if by_subreddit:
        events["_subreddit_key"] = _subreddit_key(events["subreddit"])
    events = events[events["_user_key"] != events["partner"]]
    group_columns = ["_user_key", "_subreddit_key"] if by_subreddit else ["_user_key"]
    events = events.sort_values(group_columns + ["timestamp"])

    rows: list[dict] = []
    group_key = group_columns if by_subreddit else "_user_key"
    grouped = events.groupby(group_key, sort=False)
    for key, group in tqdm(grouped, total=grouped.ngroups, desc="Interaction states"):
        if by_subreddit:
            username, subreddit = key
        else:
            username, subreddit = key, None
        partners: set[str] = set()
        total = 0
        for timestamp, at_time in group.groupby("timestamp", sort=True):
            total += len(at_time)
            partners.update(at_time["partner"].dropna().tolist())
            row = {
                "_user_key": username,
                "interaction_state_at": float(timestamp),
                "prior_n_interaction_partners": len(partners),
                "prior_total_interactions": total,
            }
            if by_subreddit:
                row["_subreddit_key"] = subreddit
                row["local_interaction_state_at"] = row.pop("interaction_state_at")
                row["prior_n_interaction_partners_in_sub"] = row.pop(
                    "prior_n_interaction_partners"
                )
                row["prior_total_interactions_in_sub"] = row.pop("prior_total_interactions")
            rows.append(row)
    return pd.DataFrame(rows)


def build_features(
    votes_path: Path,
    feature_root: Path,
    metadata_path: Path,
    include_collection_score: bool = False,
    post_texts_path: Path = DEFAULT_POST_TEXTS,
    timestamp_semantics: str = "auto",
) -> tuple[pd.DataFrame, dict]:
    """Build causal per-vote features and their timestamp/anti-leakage audit."""
    progress = tqdm(total=7, desc="Building causal features", unit="step")
    try:
        progress.set_postfix_str("loading votes")
        votes = read_table(votes_path)
        required = {"username", "item_id", "timestamp", "community"}
        missing = required - set(votes.columns)
        if missing:
            raise ValueError(f"Votes are missing required columns: {sorted(missing)}")
        timestamp_audit = audit_timestamp_semantics(
            votes, post_texts_path, override=timestamp_semantics
        )
        work = votes[["username", "item_id", "timestamp", "community"]].copy()
        work["timestamp"] = unix_seconds(work["timestamp"])
        if work["timestamp"].isna().any():
            raise ValueError(
                f"Votes contain {work['timestamp'].isna().sum():,} invalid timestamps"
            )
        work["_vote_row"] = np.arange(len(work), dtype=np.int64)
        work["_user_key"] = _user_key(work["username"])
        work["_subreddit_key"] = _subreddit_key(work["community"])
        work["timestamp_semantics"] = timestamp_audit["resolved_semantics"]
        progress.update()

        progress.set_postfix_str("global snapshots")
        global_snapshots = _read_parquet_parts(feature_root / "user_global_snapshots")
        global_snapshots["_user_key"] = _user_key(global_snapshots["username"])
        global_snapshots["global_available_at"] = _month_available_at(
            global_snapshots["year_month"]
        )
        global_features = [
            "cum_n_posts",
            "cum_n_comments",
            "cum_n_contributions",
            "cum_n_distinct_subreddits",
            "cum_n_replies_made",
        ]
        if include_collection_score:
            global_features.append("cum_karma_sum")
        global_snapshots = global_snapshots.rename(
            columns={column: f"prior_{column.removeprefix('cum_')}" for column in global_features}
        )
        renamed_global = [f"prior_{column.removeprefix('cum_')}" for column in global_features]
        work = _asof_join(
            work, global_snapshots, ["_user_key"], "global_available_at", renamed_global
        )
        progress.update()

        progress.set_postfix_str("subreddit snapshots")
        subreddit_snapshots = _read_parquet_parts(feature_root / "user_subreddit_snapshots")
        subreddit_snapshots["_user_key"] = _user_key(subreddit_snapshots["username"])
        subreddit_snapshots["_subreddit_key"] = _subreddit_key(subreddit_snapshots["subreddit"])
        subreddit_snapshots["subreddit_available_at"] = _month_available_at(
            subreddit_snapshots["year_month"]
        )
        subreddit_features = [
            "cum_n_posts_in_sub",
            "cum_n_comments_in_sub",
            "cum_n_months_active_in_sub",
        ]
        if include_collection_score:
            subreddit_features.append("cum_karma_sum_in_sub")
        subreddit_snapshots = subreddit_snapshots.rename(
            columns={
                column: f"prior_{column.removeprefix('cum_')}" for column in subreddit_features
            }
        )
        renamed_subreddit = [
            f"prior_{column.removeprefix('cum_')}" for column in subreddit_features
        ]
        work = _asof_join(
            work,
            subreddit_snapshots,
            ["_user_key", "_subreddit_key"],
            "subreddit_available_at",
            renamed_subreddit,
        )
        progress.update()

        progress.set_postfix_str("reply graph")
        reply_graph = _read_parquet_parts(
            feature_root / "reply_graph",
            columns=["from_user", "to_user", "subreddit", "timestamp"],
            allow_empty=True,
        )
        interaction_events = _build_interaction_events(reply_graph)
        if interaction_events.empty:
            work["interaction_state_at"] = np.nan
            work["prior_n_interaction_partners"] = np.nan
            work["prior_total_interactions"] = np.nan
        else:
            work = _asof_join(
                work,
                interaction_events,
                ["_user_key"],
                "interaction_state_at",
                ["prior_n_interaction_partners", "prior_total_interactions"],
            )
        local_events = _build_interaction_events(reply_graph, by_subreddit=True)
        if local_events.empty:
            work["local_interaction_state_at"] = np.nan
            work["prior_n_interaction_partners_in_sub"] = np.nan
            work["prior_total_interactions_in_sub"] = np.nan
        else:
            work = _asof_join(
                work,
                local_events,
                ["_user_key", "_subreddit_key"],
                "local_interaction_state_at",
                [
                    "prior_n_interaction_partners_in_sub",
                    "prior_total_interactions_in_sub",
                ],
            )
        progress.update()

        progress.set_postfix_str("account tenure")
        if metadata_path.exists():
            metadata = read_table(metadata_path)
            metadata["_user_key"] = _user_key(metadata["username"])
            metadata["account_created_utc"] = unix_seconds(
                metadata["account_created_utc"]
            )
            metadata = metadata.drop_duplicates("_user_key", keep="last")
            work = work.merge(
                metadata[["_user_key", "account_created_utc"]],
                on="_user_key",
                how="left",
            )
            work["tenure_days_at_vote"] = (
                work["timestamp"] - work["account_created_utc"]
            ) / 86400.0
            work.loc[work["tenure_days_at_vote"] < 0, "tenure_days_at_vote"] = np.nan
        else:
            work["account_created_utc"] = np.nan
            work["tenure_days_at_vote"] = np.nan
        progress.update()

        causal_columns = (
            renamed_global
            + renamed_subreddit
            + [
                "prior_n_interaction_partners",
                "prior_total_interactions",
                "prior_n_interaction_partners_in_sub",
                "prior_total_interactions_in_sub",
            ]
        )
        work[causal_columns] = work[causal_columns].fillna(0)
        progress.update()
        violations = {
            "global_snapshot_not_prior": int(
                (
                    work["global_available_at"].notna()
                    & (work["global_available_at"] >= work["timestamp"])
                ).sum()
            ),
            "subreddit_snapshot_not_prior": int(
                (
                    work["subreddit_available_at"].notna()
                    & (work["subreddit_available_at"] >= work["timestamp"])
                ).sum()
            ),
            "interaction_not_prior": int(
                (
                    work["interaction_state_at"].notna()
                    & (work["interaction_state_at"] >= work["timestamp"])
                ).sum()
            ),
            "local_interaction_not_prior": int(
                (
                    work["local_interaction_state_at"].notna()
                    & (work["local_interaction_state_at"] >= work["timestamp"])
                ).sum()
            ),
        }
        if any(violations.values()):
            raise AssertionError(f"Anti-leakage checks failed: {violations}")
        report = {
            "n_votes": len(work),
            "n_users": int(work["_user_key"].nunique()),
            "coverage": {
                "global_snapshot": float(work["global_available_at"].notna().mean()),
                "subreddit_snapshot": float(work["subreddit_available_at"].notna().mean()),
                "interaction_state": float(work["interaction_state_at"].notna().mean()),
                "local_interaction_state": float(
                    work["local_interaction_state_at"].notna().mean()
                ),
                "account_created": float(work["account_created_utc"].notna().mean()),
            },
            "anti_leakage_violations": violations,
            "include_collection_score": include_collection_score,
            "timestamp_audit": timestamp_audit,
        }
        progress.update()
        # Preserve the canonical Parquet column order used by downstream methods.
        output_columns = (
            KEY_COLUMNS
            + ["community"]
            + causal_columns
            + [
                "tenure_days_at_vote",
                "global_available_at",
                "subreddit_available_at",
                "interaction_state_at",
                "local_interaction_state_at",
                "timestamp_semantics",
            ]
        )
        return work[output_columns], report
    finally:
        progress.close()


def join_features(
    votes_path: Path,
    feature_root: Path,
    metadata_path: Path,
    output_path: Path,
    post_texts_path: Path = DEFAULT_POST_TEXTS,
    include_collection_score: bool = False,
    timestamp_semantics: str = "auto",
    sample_size: int = 25,
) -> dict:
    """Run and persist the causal join stage and all QA artefacts."""
    features, qa = build_features(
        votes_path,
        feature_root,
        metadata_path,
        include_collection_score,
        post_texts_path,
        timestamp_semantics,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(output_path, index=False)
    qa_path = output_path.with_suffix(".qa.json")
    sample_path = output_path.with_suffix(".qa_sample.csv")
    _write_json(qa_path, qa)
    features.sample(min(sample_size, len(features)), random_state=10).to_csv(
        sample_path, index=False
    )
    manifest = {
        "stage": "join",
        "votes": votes_path.as_posix(),
        "feature_root": feature_root.as_posix(),
        "metadata": metadata_path.as_posix(),
        "post_texts": post_texts_path.as_posix(),
        "output": output_path.as_posix(),
        "output_size_bytes": output_path.stat().st_size,
        "qa": qa,
    }
    _write_json(output_path.with_suffix(".manifest.json"), manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["index", "snapshots", "join", "all"], default="all")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--author-index", type=Path, default=DEFAULT_AUTHOR_INDEX)
    parser.add_argument("--feature-root", type=Path, default=DEFAULT_FEATURE_ROOT)
    parser.add_argument("--votes", type=Path, default=DEFAULT_VOTES)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--post-texts", type=Path, default=DEFAULT_POST_TEXTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--peek", action="store_true")
    parser.add_argument(
        "--rebuild-index",
        action="store_true",
        help="Delete and rebuild the SQLite index.",
    )
    parser.add_argument(
        "--rebuild-snapshots",
        action="store_true",
        help="Explicitly delete and rebuild snapshot Parquet parts.",
    )
    parser.add_argument("--include-collection-score", action="store_true")
    parser.add_argument("--sample-size", type=int, default=25)
    parser.add_argument(
        "--timestamp-semantics",
        choices=["auto", "vote_time", "post_creation_proxy", "unknown_or_mixed"],
        default="auto",
    )
    args = parser.parse_args()

    if args.peek:
        peek_schema(args.input_dir)
        return
    index_report = None
    if args.stage in {"index", "all"}:
        index_report = build_author_index(
            args.input_dir, args.author_index, rebuild=args.rebuild_index
        )
    if args.stage in {"snapshots", "all"}:
        build_snapshots(
            args.input_dir,
            args.author_index,
            args.feature_root,
            rebuild=args.rebuild_snapshots,
            index_report=index_report,
        )
    if args.stage in {"join", "all"}:
        join_features(
            args.votes,
            args.feature_root,
            args.metadata,
            args.output,
            post_texts_path=args.post_texts,
            include_collection_score=args.include_collection_score,
            timestamp_semantics=args.timestamp_semantics,
            sample_size=args.sample_size,
        )


if __name__ == "__main__":
    main()
