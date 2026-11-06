"""Encode post texts and causal user profiles once for SEF and Team Formation.

This is the only part of SEF that needs an accelerator.  Running it as its own
preparation stage means:

* Team Formation (which reads the same post embeddings) can never start before
  they exist;
* SEF evaluation can run as one CPU process per split, all memory-mapping the
  same read-only arrays, instead of threads sharing one interpreter.

Profiles are materialized for the union of the base votes and every split under
each ``--splits-root`` (main and K-fold), so one cache serves every split.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.layout import DEFAULT_EMBEDDING_MODEL, REDDIT  # noqa: E402
from src.methods.sef.data import (  # noqa: E402
    PREPARED_CACHE_MANIFEST,
    build_post_embeddings,
    build_temporal_user_profiles,
    collect_temporal_profile_requests,
    load_post_texts,
    load_user_documents,
    load_votes,
    write_prepared_cache_manifest,
)
from src.methods.sef.runtime import log  # noqa: E402
from src.methods.semantic_expert_finder import semantic_input_hashes  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=REDDIT.default_inputs().votes)
    parser.add_argument("--post-texts", type=Path, default=REDDIT.default_inputs().post_texts)
    parser.add_argument("--user-documents", type=Path, default=REDDIT.user_documents)
    parser.add_argument(
        "--splits-root",
        type=Path,
        action="append",
        help="Split collection whose profile requests are materialized; repeatable.",
    )
    parser.add_argument("--embedding-dir", type=Path, default=REDDIT.embedding_cache)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument(
        "--posts-only",
        action="store_true",
        help="Encode post texts only (enough for Team Formation without SEF).",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    splits_roots = [root for root in (args.splits_root or [REDDIT.splits]) if root.exists()]
    hashes = semantic_input_hashes(args.post_texts, args.user_documents)
    manifest_path = args.embedding_dir / PREPARED_CACHE_MANIFEST
    if manifest_path.is_file() and not args.force:
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        same_inputs = previous.get("input_sha256") == hashes
        same_roots = set(previous.get("splits_roots", [])) >= {r.as_posix() for r in splits_roots}
        if same_inputs and same_roots and previous.get("model") == args.embedding_model:
            log.info("Prepared semantic cache is current: %s", manifest_path)
            return

    post_texts = load_post_texts(args.post_texts)
    post_embeddings, post_ids = build_post_embeddings(
        post_texts, args.embedding_dir, args.embedding_model
    )
    if args.posts_only:
        log.info("Post embeddings ready (%d); user profiles skipped", len(post_ids))
        return

    votes = load_votes(args.input)
    requests = pd.concat(
        [collect_temporal_profile_requests(votes, root) for root in splits_roots],
        ignore_index=True,
    ).drop_duplicates()
    profiles = build_temporal_user_profiles(
        load_user_documents(args.user_documents),
        requests,
        args.embedding_dir,
        args.embedding_model,
    )
    path = write_prepared_cache_manifest(
        args.embedding_dir,
        profiles=profiles,
        n_posts=len(post_ids),
        model=args.embedding_model,
        input_hashes=hashes,
        splits_roots=[root.as_posix() for root in splits_roots],
    )
    log.info(
        "Semantic cache ready: %d posts, %d profile queries -> %s",
        len(post_embeddings),
        len(profiles.cutoffs),
        path,
    )


if __name__ == "__main__":
    main()
