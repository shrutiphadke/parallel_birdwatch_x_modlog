"""Run every active counterfactual team-formation method on prepared splits."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.methods.team_formation_v2.shared_features import load_split_manifest
from src.utils.splits import discover_splits

CHRONOLOGICAL_SPLIT = "intersection_chronological"
DEFAULT_VOTES_DIR = "data/splits/reddit"
DEFAULT_OUT_ROOT = "results/reddit"
DEFAULT_CAUSAL_FEATURES = "data/interim/reddit/features/causal_user_vote_features.parquet"
DEFAULT_EMBEDDING_DIR = "cache/embeddings"
DEFAULT_TEAM_SIZE = 3

METHODS = {
    "bandit": "src.methods.team_formation_v2.bandit.pipeline",
    "graph_propagation": "src.methods.team_formation_v2.graph_propagation.pipeline",
    "virtual_ensembles": "src.methods.team_formation_v2.virtual_ensembles.pipeline",
    "boc_stacking": "src.methods.team_formation_v2.boc_stacking.pipeline",
    "role_nsga2": "src.methods.team_formation_v2.role_nsga2.pipeline",
}
PANEL_METHODS = frozenset(METHODS) - {"boc_stacking"}


def write_summary(path: Path, summary: dict[str, str]) -> None:
    """Persist progress so a long multi-split run can be inspected or resumed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def _configured_team_size(payload: dict[str, object]) -> int | None:
    for section in ("protocol", "ensemble", "composition"):
        values = payload.get(section)
        if isinstance(values, dict) and isinstance(values.get("team_size"), int):
            return int(values["team_size"])
    return None


def has_active_metrics(path: Path, method: str) -> bool:
    """Recognize only outputs produced under the simulated-team protocol."""
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    protocol = payload.get("protocol") if isinstance(payload, dict) else None
    if not isinstance(protocol, dict) or protocol.get("simulated_estimand") is not True:
        return False
    return method not in PANEL_METHODS or _configured_team_size(payload) == DEFAULT_TEAM_SIZE


def resolve_splits(votes_path: Path) -> dict[str, Path]:
    """Resolve one direct prepared split or every canonical split below a root."""
    splits = discover_splits(votes_path)
    direct = all(
        (votes_path / f"{part}_votes.parquet").exists() for part in ("train", "val", "test")
    )
    if (
        direct
        and votes_path.parent.name in {"full", "intersection"}
        and votes_path.parent.parent.name == "windows"
    ):
        splits = {f"windows/{votes_path.parent.name}/{votes_path.name}": votes_path}
    if not splits:
        raise FileNotFoundError(f"No complete prepared split found under {votes_path}")
    incomplete = {
        name: [
            part
            for part in ("train", "val", "test")
            if not (path / f"{part}_votes.parquet").exists()
        ]
        for name, path in splits.items()
    }
    incomplete = {name: parts for name, parts in incomplete.items() if parts}
    if incomplete:
        raise FileNotFoundError(f"Incomplete prepared splits: {incomplete}")
    return splits


def resolve_chronological_split(votes_path: Path) -> Path:
    """Backward-compatible resolver used by the chronological K-sensitivity runner."""
    splits = resolve_splits(votes_path)
    if CHRONOLOGICAL_SPLIT in splits:
        return splits[CHRONOLOGICAL_SPLIT]
    if len(splits) == 1:
        name, path = next(iter(splits.items()))
        if name == CHRONOLOGICAL_SPLIT:
            return path
    raise FileNotFoundError(f"{CHRONOLOGICAL_SPLIT} was not found under {votes_path}")


def annotate_outer_protocol(metrics_path: Path, split_name: str, split_path: Path) -> None:
    """Record whether an output has a prospective or seeded-item outer protocol."""
    if not metrics_path.exists():
        return
    manifest = load_split_manifest(split_path)
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    split_type = manifest.get("split_type", "unknown")
    cv_protocol = manifest.get("cv_protocol")
    checks = manifest.get("checks", {})
    chronological = (
        split_type == "chronological" or cv_protocol == "expanding_window_chronological"
    )
    prospective = chronological and checks.get("strict_temporal_order") is True
    payload["outer_split_protocol"] = {
        "split": split_name,
        "split_type": split_type,
        "cv_protocol": cv_protocol,
        "strict_temporal_order": bool(prospective),
        "prospective_interpretation": bool(prospective),
        "note": (
            "Outer TRAIN strictly precedes VAL/TEST."
            if prospective
            else "Comparative seeded-item/window benchmark; the internal TRAIN prefix is "
            "chronological, but outer TRAIN need not precede VAL/TEST."
        ),
    }
    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_command(
    method: str,
    split_path: Path,
    out_dir: Path,
    causal_features: str,
    embedding_dir: str,
) -> list[str]:
    """Build one module command with the shared counterfactual protocol inputs."""
    command = [
        Path(sys.executable).as_posix(),
        "-B",
        "-m",
        METHODS[method],
        "--votes-dir",
        split_path.as_posix(),
        "--out-dir",
        out_dir.as_posix(),
        "--causal-features",
        Path(causal_features).as_posix(),
        "--embedding-dir",
        Path(embedding_dir).as_posix(),
    ]
    if method in PANEL_METHODS:
        command.extend(("--team-size", str(DEFAULT_TEAM_SIZE)))
    return command


_BLAS_THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def child_environment(jobs: int) -> dict[str, str] | None:
    """Share cores between parallel runs instead of oversubscribing BLAS."""
    if jobs <= 1:
        return None
    threads = str(max(1, (os.cpu_count() or 1) // jobs))
    environment = dict(os.environ)
    for variable in _BLAS_THREAD_VARIABLES:
        environment.setdefault(variable, threads)
    return environment


def run_command(command: list[str], log_path: Path | None, environment) -> int:
    """Run one method; parallel runs log to a file instead of the console."""
    if log_path is None:
        return subprocess.run(command, cwd=_PROJECT_ROOT, check=False).returncode
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        return subprocess.run(
            command,
            cwd=_PROJECT_ROOT,
            check=False,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=environment,
        ).returncode


def main() -> None:
    """Run every active method on all discovered prepared splits."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--votes-dir", default=DEFAULT_VOTES_DIR)
    parser.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    parser.add_argument("--causal-features", default=DEFAULT_CAUSAL_FEATURES)
    parser.add_argument("--embedding-dir", default=DEFAULT_EMBEDDING_DIR)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=tuple(METHODS),
        default=list(METHODS),
        help="Active methods to run.",
    )
    parser.add_argument(
        "--exclude-method",
        action="append",
        choices=tuple(METHODS),
        default=[],
        help="Method to omit. Repeat to omit more than one.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the split-by-method execution matrix without starting subprocesses.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=int(os.environ.get("TEAM_FORMATION_JOBS", "1")),
        help=(
            "Split/method runs to execute in parallel (default 1, or "
            "$TEAM_FORMATION_JOBS). With more than one, each run writes run_all.log "
            "in its output directory."
        ),
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip a method when its active metrics.json already exists for that split.",
    )
    args = parser.parse_args()

    if args.jobs < 1:
        parser.error("--jobs must be at least 1")
    excluded = set(args.exclude_method)
    selected_methods = [method for method in args.methods if method not in excluded]
    if not selected_methods:
        parser.error("method selection is empty after applying --exclude-method")

    splits = resolve_splits(Path(args.votes_dir))
    out_root = Path(args.out_root)
    print(f"Prepared splits: {len(splits)}")

    summary_filename = "team_formation_plan.json" if args.dry_run else "team_formation.json"
    summary_path = out_root / "summaries" / summary_filename
    summary: dict[str, str] = {}
    pending: list[tuple[str, list[str], Path, str, Path]] = []
    plan: list[str] = []
    for split_name, split_path in splits.items():
        for method in selected_methods:
            out_dir = out_root.joinpath(*split_name.split("/"), method)
            run_name = f"{split_name}/{method}"
            plan.append(run_name)
            command = build_command(
                method,
                split_path,
                out_dir,
                args.causal_features,
                args.embedding_dir,
            )
            if args.skip_existing and has_active_metrics(out_dir / "metrics.json", method):
                summary[run_name] = "SKIPPED (active metrics.json exists)"
                print(f"Running {run_name}")
                print(summary[run_name])
                write_summary(summary_path, summary)
                continue
            if args.dry_run:
                print(f"Running {run_name}")
                print(subprocess.list2cmdline(command))
                summary[run_name] = "DRY RUN"
                write_summary(summary_path, summary)
                continue
            pending.append((run_name, command, out_dir, split_name, split_path))

    def finish(run: tuple, returncode: int) -> None:
        run_name, _, out_dir, split_name, split_path = run
        summary[run_name] = "OK" if returncode == 0 else f"FAILED (exit {returncode})"
        if returncode == 0:
            annotate_outer_protocol(out_dir / "metrics.json", split_name, split_path)
        elif args.jobs > 1:
            print(f"{run_name}: {summary[run_name]} | log={out_dir / 'run_all.log'}")
        write_summary(summary_path, summary)

    if args.jobs == 1:
        for run in pending:
            print(f"Running {run[0]}")
            finish(run, run_command(run[1], None, None))
    else:
        environment = child_environment(args.jobs)
        print(f"Running {len(pending)} method runs with {args.jobs} parallel jobs")
        with ThreadPoolExecutor(max_workers=args.jobs) as executor:
            futures = {
                executor.submit(
                    run_command, run[1], run[2] / "run_all.log", environment
                ): run
                for run in pending
            }
            for future in as_completed(futures):
                run = futures[future]
                finish(run, future.result())
                print(f"Finished {run[0]}: {summary[run[0]]}")
        # Report in plan order, not completion order.
        reordered = {name: summary[name] for name in plan if name in summary}
        summary.clear()
        summary.update(reordered)

    print("Summary")
    for run_name, status in summary.items():
        print(f"{run_name}: {status}")

    write_summary(summary_path, summary)
    failures = {name: status for name, status in summary.items() if status.startswith("FAILED")}
    if failures:
        print(f"Failed runs: {len(failures)} | details={summary_path}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
