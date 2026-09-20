"""Reproducible Typer interface for the complete Reddit analysis pipeline."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys
from typing import Annotated, Sequence

import typer

from src.reproducibility import (
    ReproductionConfig,
    load_reproduction_config,
    nested_exclusions,
    resolve_method_selection,
    validate_reproduction_config,
)
from src.utils.splits import discover_splits

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = Path("data/processed/final_intersection_dataset.csv")
DEFAULT_SPLITS = Path("data/splits/reddit")
DEFAULT_RESULTS = Path("results/reddit")
DEFAULT_REPORT = Path("results/report")
KFOLD_DIRNAME = "kfold"
DEFAULT_CAUSAL_FEATURES = Path("data/interim/reddit/features/causal_user_vote_features.parquet")
DEFAULT_NVSE_SCORES = Path("cache/nvse/post_violation_scores.parquet")
DEFAULT_NVSE_WORK = Path("cache/nvse/work")
DEFAULT_POST_TEXTS = Path("data/interim/reddit/auxiliary/post_texts.parquet")
DEFAULT_USER_DOCUMENTS = Path("data/interim/reddit/auxiliary/user_documents.parquet")
DEFAULT_EXTERNAL_SCORES = Path("data/interim/reddit/auxiliary/moderated_posts_scores.parquet")
DEFAULT_NORMVIO_MODELS = Path("external/normvio/normvio_redditmodels")
DEFAULT_NORMVIO_BERT = "DeepPavlov/bert-base-cased-conversational"
DEFAULT_EMBEDDING_CACHE = Path("cache/embeddings")
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

METHOD_MODULES = {
    "baselines": "src.methods.baselines",
    "community-notes": "src.methods.community_notes",
    "var": "src.methods.expert_selection.var",
    "nvse": "src.methods.normvio_skill_extraction",
    "sef": "src.methods.semantic_expert_finder",
    "ma-qsmf": "src.methods.ma_qsmf",
    "team-formation": "src.methods.team_formation_v2.run_all",
}
TEAM_FORMATION_METHODS = (
    "bandit",
    "graph_propagation",
    "virtual_ensembles",
    "boc_stacking",
    "role_nsga2",
)
NESTED_METHODS = {"team-formation": TEAM_FORMATION_METHODS}

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Reproduce data preparation, model evaluation, and thesis figures.",
)
prepare_app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="One-time, versioned data and cache preparation.",
)
run_app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Repeatable model evaluation and reporting.",
)
app.add_typer(prepare_app, name="prepare")
app.add_typer(run_app, name="run")


def _python_module(module: str, *arguments: object) -> list[str]:
    serialized = [
        value.as_posix() if isinstance(value, Path) else str(value) for value in arguments
    ]
    return [Path(sys.executable).as_posix(), "-B", "-m", module, *serialized]


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _git_state() -> dict[str, object]:
    base = ["git", "-c", f"safe.directory={PROJECT_ROOT.as_posix()}"]
    commit = subprocess.run(
        [*base, "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    status = subprocess.run(
        [*base, "status", "--short"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "commit": commit.stdout.strip() if commit.returncode == 0 else None,
        "dirty": bool(status.stdout.strip()) if status.returncode == 0 else None,
    }


def _input_fingerprints(splits_root: Path, extra_inputs: Sequence[Path] = ()) -> dict[str, str]:
    paths = [
        PROJECT_ROOT / "requirements.txt",
        PROJECT_ROOT / "pyproject.toml",
        *extra_inputs,
    ]
    root = splits_root if splits_root.is_absolute() else PROJECT_ROOT / splits_root
    if root.exists():
        paths.extend(sorted(root.rglob("*manifest.json")))
        paths.extend(sorted(root.rglob("*_votes.parquet")))
    return {_relative(path): digest for path in paths if (digest := _sha256(path)) is not None}


def _source_tree_sha256() -> str:
    """Fingerprint source contents too, including an uncommitted working tree."""
    digest = hashlib.sha256()
    for path in sorted((PROJECT_ROOT / "src").rglob("*.py")):
        digest.update(_relative(path).encode("utf-8"))
        file_digest = _sha256(path)
        if file_digest is not None:
            digest.update(file_digest.encode("ascii"))
    return digest.hexdigest()


def _display_command(command: Sequence[str]) -> str:
    return subprocess.list2cmdline(list(command)) if os.name == "nt" else shlex.join(command)


def _write_run_manifest(
    manifest_path: Path,
    category: str,
    steps: list[dict[str, object]],
    splits_root: Path,
    extra_inputs: Sequence[Path] = (),
) -> None:
    destination = manifest_path if manifest_path.is_absolute() else PROJECT_ROOT / manifest_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "category": category,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": PROJECT_ROOT.as_posix(),
        "python": sys.version,
        "platform": platform.platform(),
        "git": _git_state(),
        "source_tree_sha256": _source_tree_sha256(),
        "input_fingerprints": _input_fingerprints(splits_root, extra_inputs),
        "steps": steps,
    }
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    typer.echo(f"Manifest: {_relative(destination)}")


def _execute(
    commands: Sequence[tuple[str, list[str]]],
    *,
    category: str,
    manifest_path: Path,
    splits_root: Path,
    dry_run: bool,
    continue_on_error: bool = False,
    extra_inputs: Sequence[Path] = (),
) -> None:
    records: list[dict[str, object]] = []
    failure = False
    for name, command in commands:
        typer.echo(f"\n[{name}] {_display_command(command)}")
        started = datetime.now(timezone.utc)
        return_code: int | None = None
        if not dry_run:
            return_code = subprocess.run(command, cwd=PROJECT_ROOT, check=False).returncode
        ended = datetime.now(timezone.utc)
        records.append(
            {
                "name": name,
                "command": command,
                "dry_run": dry_run,
                "started_at_utc": started.isoformat(),
                "ended_at_utc": ended.isoformat(),
                "duration_seconds": (ended - started).total_seconds(),
                "return_code": return_code,
                "status": ("dry-run" if dry_run else "ok" if return_code == 0 else "failed"),
            }
        )
        if return_code not in (None, 0):
            failure = True
            if not continue_on_error:
                break
    _write_run_manifest(
        manifest_path,
        category,
        records,
        splits_root,
        extra_inputs=extra_inputs,
    )
    if failure:
        raise typer.Exit(code=1)


def _prepared(path: Path, force: bool) -> bool:
    target = path if path.is_absolute() else PROJECT_ROOT / path
    if target.exists() and not force:
        typer.echo(f"Already prepared: {_relative(target)} (use --force to rebuild)")
        return True
    return False


def _scoped_root(root: Path, scope: str) -> Path:
    """Append one evaluation scope exactly once."""
    return root if root.name == scope else root / scope


def _load_config(path: Path) -> ReproductionConfig:
    try:
        return load_reproduction_config(path, PROJECT_ROOT, METHOD_MODULES, NESTED_METHODS)
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise typer.BadParameter(str(exc), param_hint="--config") from exc


def _direct_config(
    input_path: Path,
    dataset: str,
    methods: Sequence[str],
    *,
    post_texts: Path | None,
    user_documents: Path | None,
    user_contributions: Path | None,
    user_metadata: Path | None,
    external_scores: Path | None,
    download_user_documents: bool,
    n_user_documents: int,
    kfold: bool,
    force: bool,
    graphs: bool,
    device: str,
    excluded_methods: Sequence[str] = (),
    cache_dir: Path | None = None,
    normvio_models: Path | None = None,
    normvio_bert: str | None = None,
) -> ReproductionConfig:
    """Build the conventional layout used by the simple ``--input`` interface."""
    if not dataset or any(
        character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
        for character in dataset
    ):
        raise typer.BadParameter(
            "dataset must contain only letters, digits, '.', '_' or '-'",
            param_hint="--dataset",
        )
    try:
        selected, normalized_exclusions = resolve_method_selection(
            methods,
            excluded_methods,
            valid_methods=tuple(METHOD_MODULES),
            nested_methods=NESTED_METHODS,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="--exclude-method") from exc
    if cache_dir is not None:
        # Route embeddings and precompute (interim features, NVSE work, embedding
        # cache) onto a dedicated cache root, e.g. a larger/faster disk, instead of
        # the project directory.
        cache_root = cache_dir if cache_dir.is_absolute() else PROJECT_ROOT / cache_dir
        interim = cache_root / "interim" / dataset
        cache = cache_root / "cache" / dataset
    else:
        interim = PROJECT_ROOT / "data" / "interim" / dataset
        cache = PROJECT_ROOT / "cache" / dataset
        if dataset == "reddit":
            cache = PROJECT_ROOT / "cache"
    processed = PROJECT_ROOT / "data" / "processed" / dataset
    if dataset == "reddit":
        processed = PROJECT_ROOT / "data" / "processed"
    default_user_contributions = PROJECT_ROOT / "data" / "raw" / dataset / "user_contributions"
    report = PROJECT_ROOT / "results" / "report" / dataset
    if dataset == "reddit":
        default_user_contributions = (
            PROJECT_ROOT / "data" / "raw" / "user_contributions" / "user_contributions"
        )
        report = PROJECT_ROOT / "results" / "report"

    def supplied(path: Path | None, default: Path) -> Path:
        if path is None:
            return default
        return path if path.is_absolute() else PROJECT_ROOT / path

    auxiliary = interim / "auxiliary"
    return ReproductionConfig(
        source=None,
        dataset=dataset,
        votes=input_path if input_path.is_absolute() else PROJECT_ROOT / input_path,
        post_texts=supplied(post_texts, auxiliary / "post_texts.parquet"),
        user_documents=supplied(user_documents, auxiliary / "user_documents.parquet"),
        user_contributions=supplied(user_contributions, default_user_contributions),
        user_metadata=supplied(user_metadata, processed / "user_metadata.csv"),
        external_scores=supplied(external_scores, auxiliary / "moderated_posts_scores.parquet"),
        normvio_models=supplied(
            normvio_models, PROJECT_ROOT / "external" / "normvio" / "normvio_redditmodels"
        ),
        normvio_bert=normvio_bert or DEFAULT_NORMVIO_BERT,
        embedding_model=DEFAULT_EMBEDDING_MODEL,
        interim=interim,
        splits=PROJECT_ROOT / "data" / "splits" / dataset,
        results=PROJECT_ROOT / "results" / dataset,
        report=report,
        cache=cache,
        causal_features=interim / "features" / "causal_user_vote_features.parquet",
        nvse_scores=cache / "nvse" / "post_violation_scores.parquet",
        embedding_cache=cache / "embeddings",
        causal_features_are_precomputed=False,
        nvse_scores_are_precomputed=False,
        methods=selected,
        excluded_methods=normalized_exclusions,
        prepare_dataset=True,
        prepare_user_features=True,
        prepare_nvse_scores=True,
        download_user_documents=download_user_documents,
        n_user_documents=n_user_documents,
        graphs=graphs,
        kfold=kfold,
        force=force,
        continue_on_error=False,
        device=device,
        seed=10,
        reuse_sef_hyperparameters=False,
        skip_existing_team_methods=False,
        graph_sections=("all",),
    )


def _preflight(config: ReproductionConfig) -> None:
    errors, warnings = validate_reproduction_config(config, PROJECT_ROOT)
    for message in warnings:
        typer.echo(f"WARNING: {message}", err=True)
    if errors:
        for message in errors:
            typer.echo(f"ERROR: {message}", err=True)
        raise typer.Exit(code=2)
    typer.echo(
        f"Preflight OK: dataset={config.dataset} | methods={','.join(config.methods)} | "
        f"splits={_relative(config.splits)}"
        + (f" | excluded={','.join(config.excluded_methods)}" if config.excluded_methods else "")
    )


def _optional_path(path: Path | None, fallback: Path) -> Path:
    return path if path is not None else fallback


def _local_model_files(model: str) -> list[Path]:
    """Resolve a local path/cached Hugging Face snapshot for provenance hashing."""
    candidate = Path(model)
    candidate = candidate if candidate.is_absolute() else PROJECT_ROOT / candidate
    if not candidate.is_dir():
        try:
            from huggingface_hub import snapshot_download

            candidate = Path(snapshot_download(model, local_files_only=True))
        except Exception:
            return []
    return sorted(path for path in candidate.rglob("*") if path.is_file())


def _reproduction_commands(config: ReproductionConfig) -> list[tuple[str, list[str]]]:
    """Build the complete preparation/evaluation/report plan for one dataset."""
    commands: list[tuple[str, list[str]]] = []
    datasets_dir = config.interim / "datasets"
    dataset_marker = datasets_dir / "prepare_reddit_data.manifest.json"
    dataset_scheduled = config.prepare_dataset and (
        config.force or not dataset_marker.exists() or not bool(discover_splits(config.splits))
    )
    if dataset_scheduled:
        command = _python_module(
            "src.data_preparation.prepare_reddit_data",
            "--input",
            config.votes,
            "--output-dir",
            datasets_dir,
            "--splits-dir",
            config.splits,
            "--seed",
            config.seed,
        )
        if config.external_scores is not None:
            command.extend(("--external-scores", config.external_scores.as_posix()))
        if config.post_texts is not None:
            command.extend(("--post-texts", config.post_texts.as_posix()))
        commands.append(("prepare-dataset", command))

    needs_user_documents = "sef" in config.methods
    user_documents = _optional_path(
        config.user_documents, config.interim / "auxiliary" / "user_documents.parquet"
    )
    user_documents_scheduled = needs_user_documents and not user_documents.is_file()
    if (
        user_documents_scheduled
        and config.user_contributions is not None
        and config.user_contributions.is_dir()
    ):
        commands.append(
            (
                "prepare-user-documents-local",
                _python_module(
                    "src.data_preparation.build_user_documents",
                    "--input-dir",
                    config.user_contributions,
                    "--users-from",
                    config.votes,
                    "--output",
                    user_documents,
                    "--n-user-docs",
                    config.n_user_documents,
                ),
            )
        )

    if needs_user_documents and config.download_user_documents:
        commands.append(
            (
                "download-missing-user-documents",
                _python_module(
                    "src.data_preparation.fetch_reddit_auxiliary_data",
                    "--section",
                    "user-documents",
                    "--input-csv",
                    config.votes,
                    "--output-dir",
                    user_documents.parent,
                    "--user-documents-output",
                    user_documents,
                    "--n-user-docs",
                    config.n_user_documents,
                    "--missing-user-documents-only",
                ),
            )
        )

    causal_needed = bool({"nvse", "sef", "team-formation"} & set(config.methods))
    if (
        not config.causal_features_are_precomputed
        and config.prepare_user_features
        and causal_needed
        and (config.force or not config.causal_features.exists())
    ):
        build_state = config.interim / "build_state"
        command = _python_module(
            "src.data_preparation.build_user_history_features",
            "--stage",
            "all",
            "--input-dir",
            config.user_contributions,
            "--author-index",
            build_state / "author_index.sqlite",
            "--feature-root",
            build_state / "user_temporal_features",
            "--votes",
            config.votes,
            "--metadata",
            _optional_path(config.user_metadata, config.interim / "missing_user_metadata"),
            "--post-texts",
            _optional_path(config.post_texts, config.interim / "missing_post_texts"),
            "--output",
            config.causal_features,
        )
        if config.force:
            command.extend(("--rebuild-index", "--rebuild-snapshots"))
        commands.append(("prepare-causal-user-features", command))

    if (
        "nvse" in config.methods
        and not config.nvse_scores_are_precomputed
        and config.prepare_nvse_scores
        and (config.force or not config.nvse_scores.exists())
    ):
        command = _python_module(
            "src.methods.normvio_skill_extraction",
            "--votes_dir",
            config.splits,
            "--post_texts",
            config.post_texts,
            "--docs",
            _optional_path(config.user_documents, config.interim / "missing_user_documents"),
            "--models",
            config.normvio_models,
            "--bert_model",
            config.normvio_bert,
            "--violation_scores",
            config.nvse_scores,
            "--out_dir",
            config.cache / "nvse" / "work",
            "--tasks",
            "score",
            "--device",
            config.device,
        )
        if config.force:
            command.append("--force-rescore")
        commands.append(("prepare-nvse-scores", command))

    kfold_root = _scoped_root(config.splits, KFOLD_DIRNAME)
    if (
        config.kfold
        and not dataset_scheduled
        and (config.force or not (kfold_root / "manifest.json").exists())
    ):
        commands.append(
            (
                "prepare-kfold",
                _python_module(
                    "src.data_preparation.prepare_kfold_splits",
                    "--splits-root",
                    config.splits,
                    "--output-root",
                    kfold_root,
                    "--seed",
                    config.seed,
                ),
            )
        )

    post_texts = _optional_path(config.post_texts, config.interim / "missing_post_texts")
    user_documents = _optional_path(
        config.user_documents, config.interim / "auxiliary" / "user_documents.parquet"
    )
    external_scores = _optional_path(
        config.external_scores, config.interim / "missing_external_scores"
    )
    nvse_work = config.cache / "nvse" / "work"
    commands.extend(
        _method_commands(
            config.methods,
            config.splits,
            config.results,
            config.votes,
            config.causal_features,
            config.nvse_scores,
            nvse_work,
            post_texts,
            user_documents,
            external_scores,
            config.embedding_cache,
            config.embedding_model,
            config.reuse_sef_hyperparameters,
            config.skip_existing_team_methods,
            config.excluded_methods,
        )
    )

    if config.kfold:
        kfold_results = _scoped_root(config.results, KFOLD_DIRNAME)
        commands.extend(
            _method_commands(
                config.methods,
                kfold_root,
                kfold_results,
                config.votes,
                config.causal_features,
                config.nvse_scores,
                nvse_work,
                post_texts,
                user_documents,
                external_scores,
                config.embedding_cache,
                config.embedding_model,
                config.reuse_sef_hyperparameters,
                config.skip_existing_team_methods,
                config.excluded_methods,
            )
        )
        commands.append(
            (
                "kfold-graphs",
                _python_module(
                    "src.comparison.graphs",
                    "--k-fold-only",
                    "--splits-root",
                    kfold_root,
                    "--results-root",
                    kfold_results,
                    "--output-dir",
                    config.report / "kfold",
                ),
            )
        )

    if config.graphs:
        graph_args: list[object] = [
            "--results-root",
            config.results,
            "--splits-root",
            config.splits,
            "--output-dir",
            config.report,
        ]
        for section in config.graph_sections:
            graph_args.extend(("--section", section))
        commands.append(("graphs", _python_module("src.comparison.graphs", *graph_args)))
    return commands


@app.command("validate")
def validate_config(
    config_path: Annotated[Path, typer.Option("--config", "-c")],
) -> None:
    """Validate one dataset config, every schema, and external dependency."""
    _preflight(_load_config(config_path))


@app.command("reproduce")
def reproduce(
    input_path: Annotated[
        Path | None,
        typer.Option(
            "--input",
            help="Votes table (CSV, Parquet, JSON, or JSONL). Required without --config.",
        ),
    ] = None,
    dataset: Annotated[
        str,
        typer.Option("--dataset", help="Name used to isolate prepared data and outputs."),
    ] = "reddit",
    method: Annotated[
        list[str] | None,
        typer.Option(
            "--method",
            "-m",
            help="Method family to run. Repeat for multiple methods; omitted means all.",
        ),
    ] = None,
    exclude_method: Annotated[
        list[str] | None,
        typer.Option(
            "--exclude-method",
            "-x",
            help="Method name to omit, such as bandit or sef; repeat for several.",
        ),
    ] = None,
    post_texts: Annotated[
        Path | None,
        typer.Option(help="Local post-text table; otherwise use the conventional path."),
    ] = None,
    user_documents: Annotated[
        Path | None,
        typer.Option(help="Local historical user-document table for SEF."),
    ] = None,
    user_contributions: Annotated[
        Path | None,
        typer.Option(help="Directory containing one contribution JSONL per user."),
    ] = None,
    user_metadata: Annotated[
        Path | None,
        typer.Option(help="Local account metadata table used for tenure features."),
    ] = None,
    external_scores: Annotated[
        Path | None,
        typer.Option(help="Local per-item Reddit/Arctic Shift score table."),
    ] = None,
    download_user_documents: Annotated[
        bool,
        typer.Option(
            "--download-user-documents",
            help="Top up missing local user documents from Arctic Shift.",
        ),
    ] = False,
    n_user_documents: Annotated[
        int,
        typer.Option(
            "--n-user-documents",
            min=2,
            help="Maximum local documents per user for semantic embeddings.",
        ),
    ] = 150,
    kfold: Annotated[
        bool,
        typer.Option("--k-fold", help="Also run the prepared five-fold evaluation."),
    ] = False,
    force: Annotated[
        bool,
        typer.Option("--force", help="Rebuild prepared splits and caches."),
    ] = False,
    graphs: Annotated[
        bool,
        typer.Option("--graphs/--no-graphs", help="Enable or skip final figures."),
    ] = True,
    device: Annotated[
        str,
        typer.Option("--device", help="Execution device: cpu, cuda, or cuda:N."),
    ] = "cpu",
    cache_dir: Annotated[
        Path | None,
        typer.Option(
            "--cache-dir",
            help=(
                "Root directory for embeddings and precompute cache (interim features, "
                "NVSE work, embedding cache), e.g. a large/fast scratch disk. Defaults to "
                "the project directory."
            ),
        ),
    ] = None,
    normvio_models: Annotated[
        Path | None,
        typer.Option(
            "--normvio-models",
            help=(
                "Directory with one sub-folder per NormVio category (each holding "
                "finetuned_model.pt). Defaults to external/normvio/normvio_redditmodels."
            ),
        ),
    ] = None,
    normvio_bert: Annotated[
        str | None,
        typer.Option(
            "--normvio-bert",
            help="NormVio BERT encoder: a Hugging Face model id or local snapshot directory.",
        ),
    ] = None,
    config_path: Annotated[
        Path | None,
        typer.Option("--config", "-c", help="Optional advanced config for non-standard paths."),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Validate and print the plan without executing it."),
    ] = False,
) -> None:
    """Prepare, evaluate, and render one dataset in a single launch."""
    if config_path is not None and input_path is not None:
        raise typer.BadParameter("use either --input or --config, not both")
    if config_path is not None and (
        cache_dir is not None or normvio_models is not None or normvio_bert is not None
    ):
        raise typer.BadParameter(
            "--cache-dir/--normvio-models/--normvio-bert only apply to --input; "
            "set them under inputs/outputs in --config"
        )
    if config_path is not None:
        config = _load_config(config_path)
        if exclude_method:
            try:
                selected, normalized_exclusions = resolve_method_selection(
                    config.methods,
                    [*config.excluded_methods, *exclude_method],
                    valid_methods=tuple(METHOD_MODULES),
                    nested_methods=NESTED_METHODS,
                )
            except ValueError as exc:
                raise typer.BadParameter(str(exc), param_hint="--exclude-method") from exc
            config = replace(
                config,
                methods=selected,
                excluded_methods=normalized_exclusions,
            )
    elif input_path is not None:
        config = _direct_config(
            input_path,
            dataset,
            method or ["all"],
            excluded_methods=exclude_method or [],
            post_texts=post_texts,
            user_documents=user_documents,
            user_contributions=user_contributions,
            user_metadata=user_metadata,
            external_scores=external_scores,
            download_user_documents=download_user_documents,
            n_user_documents=n_user_documents,
            kfold=kfold,
            force=force,
            graphs=graphs,
            device=device,
            cache_dir=cache_dir,
            normvio_models=normvio_models,
            normvio_bert=normvio_bert,
        )
    else:
        raise typer.BadParameter("provide the votes table with --input", param_hint="--input")
    _preflight(config)
    commands = _reproduction_commands(config)
    if not commands:
        typer.echo("Nothing to run: every configured stage is disabled or already prepared.")
        return
    extra_inputs = [
        path
        for path in (
            config.source,
            config.votes,
            config.post_texts,
            config.user_documents,
            config.user_metadata,
            config.external_scores,
            config.causal_features,
            config.nvse_scores,
        )
        if path is not None and path.is_file()
    ]
    command_names = {name for name, _ in commands}
    if "prepare-nvse-scores" in command_names and config.normvio_models is not None:
        extra_inputs.extend(
            config.normvio_models / category / "finetuned_model.pt"
            for category in (
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
        )
        extra_inputs.extend(_local_model_files(config.normvio_bert))
    if "sef" in config.methods:
        extra_inputs.extend(_local_model_files(config.embedding_model))
    _execute(
        commands,
        category="reproduce",
        manifest_path=config.results / "reproducibility" / "latest_run.json",
        splits_root=config.splits,
        dry_run=dry_run,
        continue_on_error=config.continue_on_error,
        extra_inputs=extra_inputs,
    )


@prepare_app.command("auxiliary")
def prepare_auxiliary(
    section: Annotated[
        str,
        typer.Option(
            help="post-texts, user-documents, reddit-scores, moderated-items, or all."
        ),
    ] = "all",
    input_csv: Annotated[
        Path,
        typer.Option(help="Table supplying item_id, username, and timestamp."),
    ] = DEFAULT_INPUT,
    votes: Annotated[
        Path,
        typer.Option(help="Vote table used for Reddit score acquisition."),
    ] = DEFAULT_INPUT,
    output_dir: Annotated[
        Path,
        typer.Option(help="Destination for the fetched Parquet tables and manifest."),
    ] = Path("data/interim/reddit/auxiliary"),
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Print the acquisition command without running it."),
    ] = False,
) -> None:
    """Acquire optional Arctic Shift inputs once (network access required)."""
    allowed = {"post-texts", "user-documents", "reddit-scores", "moderated-items", "all"}
    if section not in allowed:
        raise typer.BadParameter(f"section must be one of {sorted(allowed)}")
    command = _python_module(
        "src.data_preparation.fetch_reddit_auxiliary_data",
        "--section",
        section,
        "--input-csv",
        input_csv,
        "--votes",
        votes,
        "--output-dir",
        output_dir,
    )
    _execute(
        [("auxiliary", command)],
        category="prepare",
        manifest_path=Path("data/interim/reddit/prepare_auxiliary_run.json"),
        splits_root=DEFAULT_SPLITS,
        dry_run=dry_run,
    )


@prepare_app.command("user-documents")
def prepare_user_documents(
    input_dir: Annotated[
        Path,
        typer.Option(help="Directory containing one contribution JSONL per user."),
    ] = Path("data/raw/user_contributions/user_contributions"),
    users_from: Annotated[
        Path,
        typer.Option(help="Votes table selecting users and the relevant time window."),
    ] = DEFAULT_INPUT,
    output: Annotated[
        Path,
        typer.Option(help="Destination user_documents.parquet."),
    ] = DEFAULT_USER_DOCUMENTS,
    n_user_docs: Annotated[
        int,
        typer.Option(min=2, help="Maximum documents per user, split between posts/comments."),
    ] = 150,
    download: Annotated[
        bool,
        typer.Option(
            "--download",
            help="Top up users below the local document limit with Arctic Shift.",
        ),
    ] = False,
    force: Annotated[bool, typer.Option("--force")] = False,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
) -> None:
    """Build user documents locally; optionally top up gaps from Arctic Shift."""
    commands: list[tuple[str, list[str]]] = []
    target = output if output.is_absolute() else PROJECT_ROOT / output
    if force or not target.is_file():
        command = _python_module(
            "src.data_preparation.build_user_documents",
            "--input-dir",
            input_dir,
            "--users-from",
            users_from,
            "--output",
            output,
            "--n-user-docs",
            n_user_docs,
        )
        if force:
            command.append("--force")
        commands.append(("user-documents-local", command))
    else:
        typer.echo(f"Already prepared: {_relative(target)}")
    if download:
        commands.append(
            (
                "user-documents-arctic-shift-top-up",
                _python_module(
                    "src.data_preparation.fetch_reddit_auxiliary_data",
                    "--section",
                    "user-documents",
                    "--input-csv",
                    users_from,
                    "--output-dir",
                    output.parent,
                    "--user-documents-output",
                    output,
                    "--n-user-docs",
                    n_user_docs,
                    "--missing-user-documents-only",
                ),
            )
        )
    if not commands:
        return
    _execute(
        commands,
        category="prepare",
        manifest_path=output.parent / "prepare_user_documents_run.json",
        splits_root=DEFAULT_SPLITS,
        dry_run=dry_run,
    )


@prepare_app.command("dataset")
def prepare_dataset(
    input_path: Annotated[Path, typer.Option("--input")] = DEFAULT_INPUT,
    output_dir: Annotated[Path, typer.Option()] = Path("data/interim/reddit/datasets"),
    splits_dir: Annotated[Path, typer.Option()] = DEFAULT_SPLITS,
    external_scores: Annotated[Path | None, typer.Option()] = None,
    post_texts: Annotated[Path | None, typer.Option()] = None,
    seed: Annotated[int, typer.Option()] = 10,
    force: Annotated[bool, typer.Option("--force")] = False,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
) -> None:
    """Create the cleaned dataset and all canonical splits once."""
    marker = output_dir / "prepare_reddit_data.manifest.json"
    if _prepared(marker, force) and not dry_run:
        return
    command = _python_module(
        "src.data_preparation.prepare_reddit_data",
        "--input",
        input_path,
        "--output-dir",
        output_dir,
        "--splits-dir",
        splits_dir,
        "--seed",
        seed,
    )
    if external_scores is not None:
        command.extend(("--external-scores", external_scores.as_posix()))
    if post_texts is not None:
        command.extend(("--post-texts", post_texts.as_posix()))
    _execute(
        [("dataset-and-splits", command)],
        category="prepare",
        manifest_path=Path("data/interim/reddit/prepare_dataset_run.json"),
        splits_root=splits_dir,
        dry_run=dry_run,
    )


@prepare_app.command("kfold")
def prepare_kfold(
    splits_root: Annotated[Path, typer.Option()] = DEFAULT_SPLITS,
    output_root: Annotated[
        Path | None,
        typer.Option(help="Defaults to <splits-root>/kfold."),
    ] = None,
    folds: Annotated[int, typer.Option(min=3)] = 5,
    seed: Annotated[int, typer.Option()] = 10,
    chronological_initial_train_fraction: Annotated[
        float,
        typer.Option(min=0.05, max=0.90),
    ] = 0.40,
    force: Annotated[bool, typer.Option("--force")] = False,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
) -> None:
    """Build random and walk-forward folds from the three canonical splits."""
    destination = output_root or _scoped_root(splits_root, KFOLD_DIRNAME)
    marker = destination / "manifest.json"
    if _prepared(marker, force) and not dry_run:
        return
    command = _python_module(
        "src.data_preparation.prepare_kfold_splits",
        "--splits-root",
        splits_root,
        "--output-root",
        destination,
        "--n-folds",
        folds,
        "--seed",
        seed,
        "--chronological-initial-train-fraction",
        chronological_initial_train_fraction,
    )
    _execute(
        [("kfold-splits", command)],
        category="prepare",
        manifest_path=Path("data/interim/reddit/prepare_kfold_run.json"),
        splits_root=destination,
        dry_run=dry_run,
    )


@prepare_app.command("user-features")
def prepare_user_features(
    input_dir: Annotated[Path, typer.Option()] = Path(
        "data/raw/user_contributions/user_contributions"
    ),
    votes: Annotated[Path, typer.Option()] = DEFAULT_INPUT,
    metadata: Annotated[Path, typer.Option()] = Path("data/processed/user_metadata.csv"),
    post_texts: Annotated[Path, typer.Option()] = Path(
        "data/interim/reddit/auxiliary/post_texts.parquet"
    ),
    output: Annotated[Path, typer.Option()] = DEFAULT_CAUSAL_FEATURES,
    author_index: Annotated[Path, typer.Option()] = Path(
        "data/interim/reddit/build_state/author_index.sqlite"
    ),
    feature_root: Annotated[Path, typer.Option()] = Path(
        "data/interim/reddit/build_state/user_temporal_features"
    ),
    force: Annotated[bool, typer.Option("--force")] = False,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
) -> None:
    """Build causal, per-vote user-history features once."""
    if _prepared(output, force) and not dry_run:
        return
    command = _python_module(
        "src.data_preparation.build_user_history_features",
        "--stage",
        "all",
        "--input-dir",
        input_dir,
        "--votes",
        votes,
        "--metadata",
        metadata,
        "--post-texts",
        post_texts,
        "--author-index",
        author_index,
        "--feature-root",
        feature_root,
        "--output",
        output,
    )
    if force:
        command.extend(("--rebuild-index", "--rebuild-snapshots"))
    _execute(
        [("causal-user-features", command)],
        category="prepare",
        manifest_path=Path("data/interim/reddit/prepare_user_features_run.json"),
        splits_root=DEFAULT_SPLITS,
        dry_run=dry_run,
    )


@prepare_app.command("nvse-cache")
def prepare_nvse_cache(
    votes_dir: Annotated[Path, typer.Option()] = DEFAULT_SPLITS,
    output: Annotated[Path, typer.Option()] = DEFAULT_NVSE_SCORES,
    post_texts: Annotated[Path, typer.Option()] = DEFAULT_POST_TEXTS,
    user_documents: Annotated[Path, typer.Option()] = DEFAULT_USER_DOCUMENTS,
    models: Annotated[Path, typer.Option()] = DEFAULT_NORMVIO_MODELS,
    bert_model: Annotated[str, typer.Option()] = DEFAULT_NORMVIO_BERT,
    work_dir: Annotated[Path, typer.Option()] = DEFAULT_NVSE_WORK,
    device: Annotated[str, typer.Option(help="cpu, cuda, or empty for auto-detect.")] = "",
    force: Annotated[bool, typer.Option("--force")] = False,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
) -> None:
    """Compute the split-independent NormVio post-score cache once."""
    if _prepared(output, force) and not dry_run:
        return
    command = _python_module(
        "src.methods.normvio_skill_extraction",
        "--votes_dir",
        votes_dir,
        "--violation_scores",
        output,
        "--post_texts",
        post_texts,
        "--docs",
        user_documents,
        "--models",
        models,
        "--bert_model",
        bert_model,
        "--out_dir",
        work_dir,
        "--tasks",
        "score",
        "--device",
        device,
    )
    if force:
        command.append("--force-rescore")
    _execute(
        [("nvse-post-scores", command)],
        category="prepare",
        manifest_path=Path("data/interim/reddit/prepare_nvse_cache_run.json"),
        splits_root=votes_dir,
        dry_run=dry_run,
    )


def _method_commands(
    selected: Sequence[str],
    splits_root: Path,
    results_root: Path,
    input_path: Path,
    causal_features: Path,
    nvse_scores: Path,
    nvse_work: Path,
    post_texts: Path,
    user_documents: Path,
    external_scores: Path,
    embedding_cache: Path,
    embedding_model: str,
    reuse_sef_hyperparameters: bool,
    skip_existing_team_methods: bool,
    excluded_methods: Sequence[str] = (),
) -> list[tuple[str, list[str]]]:
    unknown = sorted(set(selected).difference(METHOD_MODULES))
    if unknown:
        raise typer.BadParameter(
            f"unknown method(s): {unknown}; choose from {list(METHOD_MODULES)}"
        )
    commands: list[tuple[str, list[str]]] = []
    for method in selected:
        module = METHOD_MODULES[method]
        if method == "nvse":
            args: list[object] = [
                "--votes_dir",
                splits_root,
                "--violation_scores",
                nvse_scores,
                "--causal_features",
                causal_features,
                "--results_dir",
                results_root,
                "--out_dir",
                nvse_work,
                "--tasks",
                "splits",
            ]
        elif method == "sef":
            args = [
                "--votes-dir",
                splits_root,
                "--output-dir",
                results_root,
                "--causal-features",
                causal_features,
                "--input-csv",
                input_path,
                "--post-texts",
                post_texts,
                "--user-documents",
                user_documents,
                "--embedding-dir",
                embedding_cache,
                "--embedding-model",
                embedding_model,
                "--tasks",
                "splits",
            ]
            if reuse_sef_hyperparameters:
                args.append("--reuse-hyperparams")
        elif method == "team-formation":
            args = [
                "--votes-dir",
                splits_root,
                "--out-root",
                results_root,
                "--causal-features",
                causal_features,
                "--embedding-dir",
                embedding_cache,
            ]
            if skip_existing_team_methods:
                args.append("--skip-existing")
            for excluded in nested_exclusions(
                "team-formation", excluded_methods, NESTED_METHODS
            ):
                args.extend(("--exclude-method", excluded))
        elif method == "baselines":
            args = [
                "--votes-dir",
                splits_root,
                "--output-dir",
                results_root,
                "--external-scores",
                external_scores,
            ]
        else:
            output_flag = "--output-dir"
            args = ["--votes-dir", splits_root, output_flag, results_root]
        commands.append((method, _python_module(module, *args)))
    return commands


@run_app.command("methods")
def run_methods(
    method: Annotated[
        list[str] | None,
        typer.Option(
            "--method",
            "-m",
            help="Repeat for a subset; omit to run every canonical method family.",
        ),
    ] = None,
    exclude_method: Annotated[
        list[str] | None,
        typer.Option(
            "--exclude-method",
            "-x",
            help="Method name to omit, such as bandit or sef; repeat for several.",
        ),
    ] = None,
    splits_root: Annotated[Path, typer.Option()] = DEFAULT_SPLITS,
    results_root: Annotated[Path, typer.Option()] = DEFAULT_RESULTS,
    input_path: Annotated[Path, typer.Option("--input")] = DEFAULT_INPUT,
    causal_features: Annotated[Path, typer.Option()] = DEFAULT_CAUSAL_FEATURES,
    nvse_scores: Annotated[Path, typer.Option()] = DEFAULT_NVSE_SCORES,
    nvse_work: Annotated[Path, typer.Option()] = DEFAULT_NVSE_WORK,
    post_texts: Annotated[Path, typer.Option()] = DEFAULT_POST_TEXTS,
    user_documents: Annotated[Path, typer.Option()] = DEFAULT_USER_DOCUMENTS,
    external_scores: Annotated[Path, typer.Option()] = DEFAULT_EXTERNAL_SCORES,
    embedding_cache: Annotated[Path, typer.Option()] = DEFAULT_EMBEDDING_CACHE,
    embedding_model: Annotated[str, typer.Option()] = DEFAULT_EMBEDDING_MODEL,
    k_fold: Annotated[
        bool,
        typer.Option(
            "--k-fold",
            help="Run only the prepared folds under <splits-root>/kfold and write under "
            "<results-root>/kfold.",
        ),
    ] = False,
    reuse_sef_hyperparameters: Annotated[
        bool,
        typer.Option(help="Reuse each split's previously VAL-selected SEF configuration."),
    ] = False,
    skip_existing_team_methods: Annotated[
        bool,
        typer.Option(help="Reuse valid Team Formation v2 outputs."),
    ] = False,
    continue_on_error: Annotated[bool, typer.Option()] = False,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
    manifest: Annotated[Path | None, typer.Option()] = None,
) -> None:
    """Fit/evaluate selected methods on every discovered canonical split."""
    requested = method or ["all"]
    try:
        selected, normalized_exclusions = resolve_method_selection(
            requested,
            exclude_method or [],
            valid_methods=tuple(METHOD_MODULES),
            nested_methods=NESTED_METHODS,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="--exclude-method") from exc
    evaluation_splits_root = _scoped_root(splits_root, KFOLD_DIRNAME) if k_fold else splits_root
    evaluation_results_root = _scoped_root(results_root, KFOLD_DIRNAME) if k_fold else results_root
    if k_fold and not (evaluation_splits_root / "manifest.json").exists() and not dry_run:
        raise typer.BadParameter(
            f"K-fold splits are not prepared at {evaluation_splits_root}; run "
            "`python -m src.pipeline prepare kfold` first.",
            param_hint="--k-fold",
        )
    commands = _method_commands(
        selected,
        evaluation_splits_root,
        evaluation_results_root,
        input_path,
        causal_features,
        nvse_scores,
        nvse_work,
        post_texts,
        user_documents,
        external_scores,
        embedding_cache,
        embedding_model,
        reuse_sef_hyperparameters,
        skip_existing_team_methods,
        normalized_exclusions,
    )
    if k_fold:
        commands.append(
            (
                "kfold-graphs",
                _python_module(
                    "src.comparison.graphs",
                    "--k-fold-only",
                    "--splits-root",
                    evaluation_splits_root,
                    "--results-root",
                    evaluation_results_root,
                    "--output-dir",
                    DEFAULT_REPORT / "kfold",
                ),
            )
        )
    manifest_path = manifest or Path(
        "results/reproducibility/"
        + ("latest_kfold_methods_run.json" if k_fold else "latest_methods_run.json")
    )
    _execute(
        commands,
        category="methods",
        manifest_path=manifest_path,
        splits_root=evaluation_splits_root,
        dry_run=dry_run,
        continue_on_error=continue_on_error,
    )


@run_app.command("kfold-graphs")
def run_kfold_graphs(
    splits_root: Annotated[Path, typer.Option()] = DEFAULT_SPLITS,
    results_root: Annotated[Path, typer.Option()] = DEFAULT_RESULTS,
    output_dir: Annotated[Path | None, typer.Option()] = None,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
    manifest: Annotated[Path, typer.Option()] = Path(
        "results/reproducibility/latest_kfold_graphs_run.json"
    ),
) -> None:
    """Generate method-comparison graphs with mean and SD across K-fold runs."""
    evaluation_splits_root = _scoped_root(splits_root, KFOLD_DIRNAME)
    evaluation_results_root = _scoped_root(results_root, KFOLD_DIRNAME)
    destination = output_dir or DEFAULT_REPORT / "kfold"
    command = _python_module(
        "src.comparison.graphs",
        "--k-fold-only",
        "--splits-root",
        evaluation_splits_root,
        "--results-root",
        evaluation_results_root,
        "--output-dir",
        destination,
    )
    _execute(
        [("kfold-graphs", command)],
        category="report",
        manifest_path=manifest,
        splits_root=evaluation_splits_root,
        dry_run=dry_run,
    )


@run_app.command("graphs")
def run_graphs(
    section: Annotated[
        list[str] | None,
        typer.Option(
            "--section",
            help="Repeat to select sections; omit for the complete report.",
        ),
    ] = None,
    results_root: Annotated[Path, typer.Option()] = DEFAULT_RESULTS,
    splits_root: Annotated[Path, typer.Option()] = DEFAULT_SPLITS,
    output_dir: Annotated[Path, typer.Option()] = DEFAULT_REPORT,
    strict_audit: Annotated[bool, typer.Option()] = False,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
    manifest: Annotated[Path, typer.Option()] = Path(
        "results/reproducibility/latest_graphs_run.json"
    ),
) -> None:
    """Regenerate graphs/tables from all discovered split and method outputs."""
    arguments: list[object] = [
        "--results-root",
        results_root,
        "--splits-root",
        splits_root,
        "--output-dir",
        output_dir,
    ]
    for value in section or ["all"]:
        arguments.extend(("--section", value))
    if strict_audit:
        arguments.append("--strict-audit")
    _execute(
        [("graphs", _python_module("src.comparison.graphs", *arguments))],
        category="graphs",
        manifest_path=manifest,
        splits_root=splits_root,
        dry_run=dry_run,
    )


if __name__ == "__main__":
    app()
