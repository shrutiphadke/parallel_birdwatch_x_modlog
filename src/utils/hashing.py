"""Content hashing shared by manifests and cache validation."""

from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_file(path: Path | None) -> str | None:
    """Return the file's SHA-256, or ``None`` when it does not exist."""
    if path is None:
        return None
    path = Path(path)
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
