"""Fetch resumable Reddit items (texts + scores) and user documents from Arctic Shift."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

from src.layout import REDDIT
from src.utils.tabular import read_table, unix_seconds

INPUT_CSV = REDDIT.default_inputs().votes
OUTPUT_DIR = REDDIT.auxiliary
# Former section names, all served by the single ``items`` pass now.
LEGACY_SECTIONS = {"post-texts": "items", "reddit-scores": "items", "moderated-items": "items"}
ARCTIC_SHIFT_BASE = "https://arctic-shift.photon-reddit.com"

API_TIMEOUT_SEC = 30
API_RETRIES = 3
API_BACKOFF_SEC = 2.0
API_SLEEP_SEC = 0.35
POST_CHUNK_SIZE = 25
SCORE_CHUNK_SIZE = 20
N_USER_DOCS = 150

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def _chunk(values: List[Any], size: int) -> Iterable[List[Any]]:
    for offset in range(0, len(values), size):
        yield values[offset : offset + size]


def _normalize_payload(payload: Any) -> List[Dict]:
    """Normalize common Arctic Shift response envelopes to records."""
    if payload is None:
        return []
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("data", "results", "items", "children"):
            if key not in payload:
                continue
            value = payload[key]
            if isinstance(value, dict):
                return [value]
            if isinstance(value, list):
                normalized = []
                for item in value:
                    if isinstance(item, dict) and isinstance(item.get("data"), dict):
                        normalized.append(item["data"])
                    elif isinstance(item, dict):
                        normalized.append(item)
                return normalized
        return [payload]
    return []


class ArcticShiftClient:
    """Shared Arctic Shift session with retry, backoff, normalization, and pacing."""

    def __init__(
        self,
        base_url: str = ARCTIC_SHIFT_BASE,
        timeout: int = API_TIMEOUT_SEC,
        retries: int = API_RETRIES,
        backoff: float = API_BACKOFF_SEC,
        sleep_seconds: float = API_SLEEP_SEC,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.retries = retries
        self.backoff = backoff
        self.sleep_seconds = sleep_seconds
        self.session = requests.Session()
        # Shared across sections in one `--section all` run so post texts and
        # Reddit scores never fetch the same item twice.
        self._post_cache: Dict[str, Dict] = {}
        self._missing_post_ids: set[str] = set()

    def get(self, path: str, params: Optional[Dict[str, Any]] = None) -> List[Dict]:
        """GET and normalize one endpoint, returning [] after terminal failure."""
        url = f"{self.base_url}{path}"
        clean_params = {key: value for key, value in (params or {}).items() if value is not None}
        last_error: Optional[Exception] = None
        try:
            for attempt in range(1, self.retries + 1):
                try:
                    response = self.session.get(url, params=clean_params, timeout=self.timeout)
                    response.raise_for_status()
                    return _normalize_payload(response.json())
                except requests.exceptions.HTTPError as exc:
                    last_error = exc
                    if exc.response is not None and 400 <= exc.response.status_code < 500:
                        log.warning("HTTP %s for %s", exc.response.status_code, url)
                        return []
                except (requests.RequestException, ValueError) as exc:
                    last_error = exc
                if attempt < self.retries:
                    time.sleep(self.backoff * attempt)
            log.warning("All retries failed for %s (%s)", url, last_error)
            return []
        finally:
            if self.sleep_seconds:
                time.sleep(self.sleep_seconds)

    def fetch_posts_by_ids(self, item_ids: List[str]) -> List[Dict]:
        """Fetch a post chunk, accepting both bare and t3-prefixed endpoint forms."""
        requested = [_strip_prefix(item_id) for item_id in item_ids]
        unresolved = [
            item_id
            for item_id in requested
            if item_id not in self._post_cache and item_id not in self._missing_post_ids
        ]
        for use_prefix in (False, True):
            if not unresolved:
                break
            values = [f"t3_{item_id}" if use_prefix else item_id for item_id in unresolved]
            items = self.get("/api/posts/ids", params={"ids": ",".join(values)})
            for item in items:
                canonical = _canonical_post_id(item)
                if canonical is not None:
                    self._post_cache[_strip_prefix(canonical)] = item
            unresolved = [item_id for item_id in unresolved if item_id not in self._post_cache]
        self._missing_post_ids.update(unresolved)
        return [self._post_cache[item_id] for item_id in requested if item_id in self._post_cache]


def _safe_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return None if text.casefold() in {"", "[removed]", "[deleted]"} else text


def _numeric(value: Any) -> float:
    numeric = pd.to_numeric(value, errors="coerce")
    return float(numeric) if pd.notna(numeric) else np.nan


def _combine_post_text(item: Dict) -> Optional[str]:
    title = _safe_text(item.get("title"))
    body = _safe_text(item.get("selftext"))
    return f"{title}\n\n{body}" if title and body else title or body


def _strip_prefix(item_id: str) -> str:
    text = str(item_id)
    return text.split("_", 1)[1] if text.startswith(("t1_", "t3_")) else text


def _canonical_post_id(item: Dict, fallback: Optional[str] = None) -> Optional[str]:
    name = item.get("name")
    if isinstance(name, str) and name.startswith("t3_"):
        return name
    raw_id = item.get("id") or fallback
    if raw_id is None:
        return None
    raw_id = str(raw_id)
    return raw_id if raw_id.startswith("t3_") else f"t3_{_strip_prefix(raw_id)}"


def _load_ids(input_csv: Path, column: str) -> List[str]:
    if not input_csv.exists():
        raise FileNotFoundError(input_csv)
    values = read_table(input_csv, columns=[column])[column]
    return values.dropna().astype(str).drop_duplicates().tolist()


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    """Replace one checkpoint only after a complete temporary Parquet write."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def _merge_checkpoint(
    path: Path,
    rows: List[Dict],
    deduplicate_on: List[str],
) -> pd.DataFrame:
    """Merge buffered rows with an existing checkpoint idempotently."""
    new = _normalize_auxiliary_columns(pd.DataFrame(rows))
    if path.exists():
        existing = _normalize_auxiliary_columns(pd.read_parquet(path))
        new = pd.concat([existing, new], ignore_index=True) if not new.empty else existing
    if not new.empty:
        # Preserve rows from minimal compatible tables when optional identity
        # columns are absent; only rows with a complete key can be deduplicated.
        for column in deduplicate_on:
            if column not in new:
                new[column] = pd.NA
        complete_key = new[deduplicate_on].notna().all(axis=1)
        new = pd.concat(
            [
                new.loc[~complete_key],
                new.loc[complete_key].drop_duplicates(subset=deduplicate_on, keep="last"),
            ],
            ignore_index=True,
        )
        _atomic_parquet(new, path)
    return new


def _normalize_auxiliary_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize legacy checkpoint column names before resuming a fetch."""
    frame = frame.copy()
    for legacy, canonical in (("thing_id", "item_id"), ("created_utc", "timestamp")):
        if legacy not in frame:
            continue
        if canonical in frame:
            frame[canonical] = frame[canonical].combine_first(frame[legacy])
            frame = frame.drop(columns=legacy)
        else:
            frame = frame.rename(columns={legacy: canonical})
    if {"source", "item_id"}.issubset(frame.columns):
        identifiers = frame["item_id"].astype("string")
        bare = identifiers.notna() & ~identifiers.str.startswith(("t1_", "t3_"), na=False)
        prefixes = frame["source"].map({"comment": "t1_", "post": "t3_"}).fillna("")
        frame.loc[bare, "item_id"] = prefixes[bare] + identifiers[bare]
    return frame


def _infer_time_window(input_csv: Path) -> Tuple[Optional[str], Optional[str]]:
    timestamps = read_table(input_csv, columns=["timestamp"])["timestamp"]
    numeric = unix_seconds(timestamps)
    parsed = pd.to_datetime(numeric, unit="s", utc=True, errors="coerce").dropna()
    if parsed.empty:
        return None, None
    return parsed.min().date().isoformat(), parsed.max().date().isoformat()


def _collect_user_documents(
    client: ArcticShiftClient,
    username: str,
    n_posts: int,
    n_comments: int,
    after: Optional[str],
    before: Optional[str],
) -> Tuple[List[Dict], Dict]:
    posts = client.get(
        "/api/posts/search",
        params={
            "author": username,
            "after": after,
            "before": before,
            "limit": n_posts,
            "sort": "desc",
        },
    )
    comments = client.get(
        "/api/comments/search",
        params={
            "author": username,
            "after": after,
            "before": before,
            "limit": n_comments,
            "sort": "desc",
        },
    )
    documents: List[Dict] = []
    for item in posts:
        text = _combine_post_text(item)
        if text is not None:
            documents.append(
                {
                    "username": username,
                    "source": "post",
                    "item_id": _canonical_post_id(item),
                    "subreddit": item.get("subreddit"),
                    "timestamp": _numeric(item.get("created_utc")),
                    "text": text,
                    "score": _numeric(item.get("score")),
                }
            )
    for item in comments:
        text = _safe_text(item.get("body"))
        if text is not None:
            documents.append(
                {
                    "username": username,
                    "source": "comment",
                    "item_id": (
                        item.get("name")
                        if str(item.get("name", "")).startswith("t1_")
                        else f"t1_{_strip_prefix(item['id'])}"
                        if item.get("id")
                        else None
                    ),
                    "subreddit": item.get("subreddit"),
                    "timestamp": _numeric(item.get("created_utc")),
                    "text": text,
                    "score": _numeric(item.get("score")),
                }
            )
    return documents, {
        "username": username,
        "n_posts_fetched": len(posts),
        "n_comments_fetched": len(comments),
        "n_docs_with_text": len(documents),
        "posts_truncated": len(posts) >= n_posts,
        "comments_truncated": len(comments) >= n_comments,
    }


def fetch_user_documents(
    client: ArcticShiftClient,
    input_csv: Path,
    output_dir: Path,
    documents_path: Optional[Path] = None,
    n_user_docs: int = N_USER_DOCS,
    after: Optional[str] = None,
    before: Optional[str] = None,
    checkpoint_users: int = 500,
    missing_only: bool = False,
) -> dict:
    """Fetch user documents, optionally topping up a locally built table."""
    documents_path = documents_path or output_dir / "user_documents.parquet"
    summary_path = documents_path.with_name("user_history_summary.parquet")
    usernames = _load_ids(input_csv, "username")
    if missing_only:
        if documents_path.exists():
            existing = _normalize_auxiliary_columns(pd.read_parquet(documents_path))
            existing = existing.dropna(subset=["username", "timestamp", "text"])
            counts = existing["username"].astype(str).str.casefold().value_counts()
            todo = [
                username
                for username in usernames
                if counts.get(username.casefold(), 0) < n_user_docs
            ]
        else:
            todo = usernames
    else:
        done = (
            set(pd.read_parquet(summary_path, columns=["username"])["username"].dropna())
            if summary_path.exists()
            else set()
        )
        todo = [username for username in usernames if username not in done]
    if after is None and before is None:
        after, before = _infer_time_window(input_csv)
    n_posts = n_user_docs // 2
    n_comments = n_user_docs - n_posts
    document_rows: List[Dict] = []
    summary_rows: List[Dict] = []
    for user_index, username in enumerate(tqdm(todo, desc="User documents")):
        documents, summary = _collect_user_documents(
            client, username, n_posts, n_comments, after, before
        )
        document_rows.extend(documents)
        summary_rows.append(summary)
        if (user_index + 1) % checkpoint_users == 0:
            _merge_checkpoint(documents_path, document_rows, ["username", "source", "item_id"])
            _merge_checkpoint(summary_path, summary_rows, ["username"])
            document_rows.clear()
            summary_rows.clear()
    documents = _merge_checkpoint(documents_path, document_rows, ["username", "source", "item_id"])
    summaries = _merge_checkpoint(summary_path, summary_rows, ["username"])
    return {
        "section": "user-documents",
        "outputs": [documents_path.as_posix(), summary_path.as_posix()],
        "n_requested_users": len(usernames),
        "n_users": len(summaries),
        "n_documents": len(documents),
        "missing_only": missing_only,
        "n_users_requested_from_arctic_shift": len(todo),
        "after": after,
        "before": before,
    }


def _item_row(item: Dict, fallback_id: Optional[str] = None) -> Dict:
    """One canonical per-item row: texts and Reddit score from a single response."""
    title = _safe_text(item.get("title"))
    body = _safe_text(item.get("selftext"))
    return {
        "item_id": _canonical_post_id(item, fallback_id),
        "author": item.get("author"),
        "subreddit": item.get("subreddit"),
        "timestamp": _numeric(item.get("created_utc")),
        "title": title,
        "selftext": body,
        "text": _combine_post_text(item),
        "score": _numeric(item.get("score")),
        "upvote_ratio": _numeric(item.get("upvote_ratio")),
        "num_comments": _numeric(item.get("num_comments")),
        "url": item.get("url"),
        "domain": item.get("domain"),
        "is_self": item.get("is_self"),
        "removed_by_category": item.get("removed_by_category"),
        "removal_reason": item.get("removal_reason"),
        "banned_by": item.get("banned_by"),
        "locked": item.get("locked", False),
        "over_18": item.get("over_18", False),
        "link_flair_text": item.get("link_flair_text"),
        "author_flair_text": item.get("author_flair_text"),
        "permalink": item.get("permalink"),
    }


def fetch_items(
    client: ArcticShiftClient,
    votes_path: Path,
    output_dir: Path,
    checkpoint_chunks: int = 200,
) -> dict:
    """Fetch every voted item once into ``items.parquet`` (texts + score).

    This replaces the former separate post-text and score passes, which
    requested every post from Arctic Shift twice.  Items Arctic Shift does not
    return still get a row (with null fields) so resumed runs skip them.
    """
    output_path = output_dir / "items.parquet"
    all_ids = _load_ids(votes_path, "item_id")
    done = (
        set(pd.read_parquet(output_path, columns=["item_id"])["item_id"].dropna())
        if output_path.exists()
        else set()
    )
    todo = [item_id for item_id in all_ids if item_id not in done]
    rows: List[Dict] = []
    chunks = list(_chunk(todo, POST_CHUNK_SIZE))
    for chunk_index, item_chunk in enumerate(tqdm(chunks, desc="Items")):
        items = client.fetch_posts_by_ids(item_chunk)
        by_bare_id = {_strip_prefix(_canonical_post_id(item) or ""): item for item in items}
        for item_id in item_chunk:
            rows.append(_item_row(by_bare_id.get(_strip_prefix(item_id)) or {}, item_id))
        if rows and (chunk_index + 1) % checkpoint_chunks == 0:
            _merge_checkpoint(output_path, rows, ["item_id"])
            rows.clear()
    frame = _merge_checkpoint(output_path, rows, ["item_id"])
    return {
        "section": "items",
        "output": output_path.as_posix(),
        "n_requested": len(all_ids),
        "n_rows": len(frame),
        "n_with_text": int(frame["text"].notna().sum()) if not frame.empty else 0,
        "n_with_score": int(frame["score"].notna().sum()) if not frame.empty else 0,
    }


def _write_manifest(
    output_dir: Path, section: str, reports: List[dict], client: ArcticShiftClient
) -> None:
    manifest = {
        "pipeline": "fetch_reddit_auxiliary_data",
        "section": section,
        "source": client.base_url,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "retry_policy": {
            "timeout_seconds": client.timeout,
            "retries": client.retries,
            "backoff_seconds": client.backoff,
            "request_sleep_seconds": client.sleep_seconds,
        },
        "reports": reports,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "fetch_reddit_auxiliary_data.manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--section",
        choices=["items", "user-documents", "all", *LEGACY_SECTIONS],
        default="items",
    )
    parser.add_argument("--input-csv", type=Path, default=INPUT_CSV)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--user-documents-output",
        type=Path,
        default=None,
        help="Optional user-document target path instead of <output-dir>/user_documents.parquet.",
    )
    parser.add_argument("--n-user-docs", type=int, default=N_USER_DOCS)
    parser.add_argument("--after", default=None)
    parser.add_argument("--before", default=None)
    parser.add_argument("--base-url", default=ARCTIC_SHIFT_BASE)
    parser.add_argument("--request-sleep", type=float, default=API_SLEEP_SEC)
    parser.add_argument(
        "--missing-user-documents-only",
        action="store_true",
        help="For user-documents, fetch only users with fewer than --n-user-docs locally.",
    )
    args = parser.parse_args()

    client = ArcticShiftClient(base_url=args.base_url, sleep_seconds=args.request_sleep)
    reports = []
    section = LEGACY_SECTIONS.get(args.section, args.section)
    if section in {"items", "all"}:
        reports.append(fetch_items(client, args.input_csv, args.output_dir))
    if section in {"user-documents", "all"}:
        reports.append(
            fetch_user_documents(
                client,
                args.input_csv,
                args.output_dir,
                documents_path=args.user_documents_output,
                n_user_docs=args.n_user_docs,
                after=args.after,
                before=args.before,
                missing_only=args.missing_user_documents_only,
            )
        )
    _write_manifest(args.output_dir, args.section, reports, client)
    for report in reports:
        log.info("%s", report)


if __name__ == "__main__":
    main()
