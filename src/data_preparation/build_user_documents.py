"""Build the bounded semantic user-document table from contribution JSONLs."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime, timedelta
import hashlib
import heapq
from html import unescape
import json
import logging
import math
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from src.layout import REDDIT  # noqa: E402
from src.utils.tabular import read_table, unix_seconds

DEFAULT_INPUT_DIR = REDDIT.default_inputs().user_contributions
DEFAULT_USERS_FROM = REDDIT.default_inputs().votes
DEFAULT_OUTPUT = REDDIT.user_documents
DEFAULT_N_USER_DOCS = 150
WRITE_BATCH_ROWS = 50_000

SCHEMA = pa.schema(
    [
        ("username", pa.string()),
        ("source", pa.string()),
        ("item_id", pa.string()),
        ("subreddit", pa.string()),
        ("timestamp", pa.float64()),
        ("text", pa.string()),
        ("score", pa.float64()),
    ]
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

try:
    import orjson

    def _loads(value: bytes) -> dict[str, Any]:
        return orjson.loads(value)

except ImportError:

    def _loads(value: bytes) -> dict[str, Any]:
        return json.loads(value)


def _safe_text(value: Any) -> str | None:
    if value is None:
        return None
    text = unescape(str(value).strip())
    return None if text.casefold() in {"", "[removed]", "[deleted]"} else text


def _as_document(record: dict[str, Any], username: str) -> dict[str, Any] | None:
    name = record.get("name")
    if (isinstance(name, str) and name.startswith("t1_")) or "body" in record:
        source = "comment"
        text = _safe_text(record.get("body"))
    elif (isinstance(name, str) and name.startswith("t3_")) or "title" in record:
        source = "post"
        title = _safe_text(record.get("title"))
        body = _safe_text(record.get("selftext"))
        text = f"{title}\n\n{body}" if title and body else title or body
    else:
        return None
    if text is None:
        return None

    identifier = name if isinstance(name, str) and name.startswith(("t1_", "t3_")) else record.get("id")
    if identifier is None:
        return None
    identifier = str(identifier)
    item_id = (
        identifier
        if identifier.startswith(("t1_", "t3_"))
        else f"t{1 if source == 'comment' else 3}_{identifier}"
    )

    try:
        timestamp_value = float(record.get("created_utc"))
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(timestamp_value):
        return None

    try:
        score = float(record.get("score"))
        if not math.isfinite(score):
            score = None
    except (TypeError, ValueError, OverflowError):
        score = None

    subreddit = record.get("subreddit")
    return {
        "username": username,
        "source": source,
        "item_id": item_id,
        "subreddit": str(subreddit) if subreddit is not None else None,
        "timestamp": timestamp_value,
        "text": text,
        "score": score,
    }


def _parse_boundary(value: str | None, *, end: bool) -> int | None:
    if value is None:
        return None
    parsed = datetime.fromisoformat(value)
    parsed = parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)
    if end and len(value) == 10:
        parsed += timedelta(days=1)
    return int(parsed.timestamp())


def _users_and_window(
    users_from: Path | None,
) -> tuple[dict[str, str] | None, str | None, str | None]:
    if users_from is None:
        return None, None, None
    frame = read_table(users_from)
    if "username" not in frame:
        raise ValueError(f"{users_from} has no username column")
    usernames = frame["username"].dropna().astype(str).drop_duplicates()
    users = {username.casefold(): username for username in usernames}
    if "timestamp" not in frame:
        return users, None, None
    valid = unix_seconds(frame["timestamp"]).dropna()
    if valid.empty:
        return users, None, None
    after = datetime.fromtimestamp(float(valid.min()), tz=UTC).date().isoformat()
    before = datetime.fromtimestamp(float(valid.max()), tz=UTC).date().isoformat()
    return users, after, before


def _select_documents(
    path: Path,
    username: str,
    limits: dict[str, int],
    after_timestamp: int | None,
    before_timestamp: int | None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    heaps: dict[str, list[tuple[int, int, dict[str, Any]]]] = {"post": [], "comment": []}
    seen: set[tuple[str, str]] = set()
    counts = {"lines": 0, "invalid": 0, "usable_posts": 0, "usable_comments": 0}
    sequence = 0

    with path.open("rb") as handle:
        for line in handle:
            counts["lines"] += 1
            try:
                record = _loads(line)
            except (TypeError, UnicodeError, ValueError):
                counts["invalid"] += 1
                continue
            if not isinstance(record, dict):
                counts["invalid"] += 1
                continue
            document = _as_document(record, username)
            if document is None:
                counts["invalid"] += 1
                continue
            timestamp = document["timestamp"]
            if after_timestamp is not None and timestamp < after_timestamp:
                continue
            if before_timestamp is not None and timestamp >= before_timestamp:
                continue
            source = document["source"]
            key = (source, document["item_id"])
            if key in seen:
                continue
            seen.add(key)
            counts[f"usable_{source}s"] += 1
            sequence += 1
            entry = (timestamp, sequence, document)
            heap = heaps[source]
            if len(heap) < limits[source]:
                heapq.heappush(heap, entry)
            elif entry[:2] > heap[0][:2]:
                heapq.heapreplace(heap, entry)

    selected = [entry[2] for heap in heaps.values() for entry in heap]
    selected.sort(key=lambda row: (row["timestamp"], row["source"], row["item_id"]))
    return selected, counts


def _write_rows(writer: pq.ParquetWriter, rows: list[dict[str, Any]]) -> None:
    if rows:
        writer.write_table(pa.Table.from_pylist(rows, schema=SCHEMA))


def _input_fingerprint(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        stat = path.stat()
        digest.update(path.name.casefold().encode("utf-8"))
        digest.update(str(stat.st_size).encode("ascii"))
        digest.update(str(stat.st_mtime_ns).encode("ascii"))
    return digest.hexdigest()


def build_user_documents(
    input_dir: Path,
    output: Path,
    *,
    users_from: Path | None = None,
    n_user_docs: int = DEFAULT_N_USER_DOCS,
    after: str | None = None,
    before: str | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Convert JSONLs to a bounded Parquet table without network access."""
    if n_user_docs < 2:
        raise ValueError("n_user_docs must be at least 2")
    files = sorted(input_dir.glob("*.jsonl"))
    if not files:
        raise ValueError(f"No *.jsonl files found in {input_dir}")

    users, inferred_after, inferred_before = _users_and_window(users_from)
    after = after or inferred_after
    before = before or inferred_before
    after_timestamp = _parse_boundary(after, end=False)
    before_timestamp = _parse_boundary(before, end=True)
    selected_files = [
        path for path in files if users is None or path.stem.casefold() in users
    ]
    if not selected_files:
        raise ValueError("No contribution JSONL matches a requested username")
    if output.exists() and not force:
        raise FileExistsError(f"{output} already exists; pass --force to replace it")

    matched_keys = {path.stem.casefold() for path in selected_files}
    missing_users = (
        sorted(username for key, username in users.items() if key not in matched_keys)
        if users is not None
        else []
    )
    n_posts = n_user_docs // 2
    limits = {"post": n_posts, "comment": n_user_docs - n_posts}
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    summary_path = output.with_name(f"{output.stem}_local_summary.parquet")
    summary_temporary = summary_path.with_suffix(summary_path.suffix + ".tmp")
    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    totals = {"lines": 0, "invalid": 0, "usable_posts": 0, "usable_comments": 0}
    selected_total = 0

    try:
        with pq.ParquetWriter(temporary, SCHEMA, compression="zstd") as writer:
            for path in tqdm(selected_files, desc="Local user documents"):
                username = users[path.stem.casefold()] if users is not None else path.stem
                selected, counts = _select_documents(
                    path, username, limits, after_timestamp, before_timestamp
                )
                for key in totals:
                    totals[key] += counts[key]
                selected_posts = sum(row["source"] == "post" for row in selected)
                selected_comments = len(selected) - selected_posts
                selected_total += len(selected)
                rows.extend(selected)
                summaries.append(
                    {
                        "username": username,
                        **counts,
                        "selected_posts": selected_posts,
                        "selected_comments": selected_comments,
                        "selected_documents": len(selected),
                    }
                )
                if len(rows) >= WRITE_BATCH_ROWS:
                    _write_rows(writer, rows)
                    rows.clear()
            _write_rows(writer, rows)
        pd.DataFrame(summaries).to_parquet(summary_temporary, index=False)
        temporary.replace(output)
        summary_temporary.replace(summary_path)
    finally:
        temporary.unlink(missing_ok=True)
        summary_temporary.unlink(missing_ok=True)

    report = {
        "pipeline": "build_user_documents",
        "source": input_dir.as_posix(),
        "output": output.as_posix(),
        "summary": summary_path.as_posix(),
        "users_from": users_from.as_posix() if users_from is not None else None,
        "n_jsonl_files": len(selected_files),
        "n_requested_users": len(users) if users is not None else None,
        "n_users_without_jsonl": len(missing_users),
        "users_without_jsonl_examples": missing_users[:20],
        "n_documents": selected_total,
        "n_user_docs": n_user_docs,
        "per_type_limits": limits,
        "after": after,
        "before": before,
        "totals": totals,
        "input_fingerprint": _input_fingerprint(selected_files),
    }
    output.with_suffix(".manifest.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--users-from", type=Path, default=DEFAULT_USERS_FROM)
    parser.add_argument("--n-user-docs", type=int, default=DEFAULT_N_USER_DOCS)
    parser.add_argument("--after", default=None)
    parser.add_argument("--before", default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_user_documents(
        args.input_dir,
        args.output,
        users_from=args.users_from,
        n_user_docs=args.n_user_docs,
        after=args.after,
        before=args.before,
        force=args.force,
    )
    log.info("%s", report)


if __name__ == "__main__":
    main()
