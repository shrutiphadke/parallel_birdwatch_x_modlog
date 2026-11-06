"""Run chronological team-size sensitivity experiments.

Retrains panel methods for each K, keeps BoC as a full-electorate reference,
and selects winners by validation macro-F1 before reporting test metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.layout import REDDIT  # noqa: E402
from src.methods.team_formation_v2.run_all import (  # noqa: E402
    CHRONOLOGICAL_SPLIT,
    DEFAULT_CAUSAL_FEATURES,
    DEFAULT_EMBEDDING_DIR,
    DEFAULT_VOTES_DIR,
    METHODS,
    resolve_chronological_split,
)

PANEL_METHODS = {
    name: module
    for name, module in METHODS.items()
    if name != "boc_stacking"
}
DEFAULT_K_VALUES = (3, 5, 7, 10)
DEFAULT_OUT_ROOT = str(REDDIT.results / CHRONOLOGICAL_SPLIT / "k_sensitivity")
DEFAULT_BOC_METRICS = str(REDDIT.results / CHRONOLOGICAL_SPLIT / "boc_stacking/metrics.json")
METRIC_NAMES = ("macro_f1", "f1_neg", "roc_auc")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def _configured_team_size(metrics: dict[str, Any]) -> int | None:
    for section in ("protocol", "ensemble", "composition"):
        payload = metrics.get(section)
        if isinstance(payload, dict) and isinstance(payload.get("team_size"), int):
            return int(payload["team_size"])
    return None


def _is_complete_result(path: Path, expected_k: int) -> bool:
    if not path.exists():
        return False
    try:
        metrics = _load_json(path)
    except (OSError, json.JSONDecodeError, ValueError):
        return False
    protocol = metrics.get("protocol")
    return (
        isinstance(protocol, dict)
        and protocol.get("simulated_estimand") is True
        and _configured_team_size(metrics) == expected_k
        and isinstance(metrics.get("val"), dict)
        and isinstance(metrics.get("test"), dict)
    )


def _command(
    method: str,
    k: int,
    votes_dir: Path,
    out_dir: Path,
    causal_features: str,
    embedding_dir: str,
) -> list[str]:
    return [
        Path(sys.executable).as_posix(),
        "-B",
        "-m",
        PANEL_METHODS[method],
        "--votes-dir",
        votes_dir.as_posix(),
        "--out-dir",
        out_dir.as_posix(),
        "--causal-features",
        Path(causal_features).as_posix(),
        "--embedding-dir",
        Path(embedding_dir).as_posix(),
        "--team-size",
        str(k),
    ]


def _metric_value(metrics: dict[str, Any], split: str, metric: str) -> float | None:
    payload = metrics.get(split)
    value = payload.get(metric) if isinstance(payload, dict) else None
    return float(value) if isinstance(value, (int, float)) and math.isfinite(value) else None


def _row(
    method: str,
    k: int | str,
    metrics_path: Path,
    source: str = "sensitivity_run",
) -> dict[str, Any]:
    metrics = _load_json(metrics_path)
    row: dict[str, Any] = {
        "method": method,
        "k": k,
        "k_applicable": isinstance(k, int),
        "source": source,
        "metrics_path": metrics_path.as_posix(),
    }
    for split in ("val", "test"):
        for metric in METRIC_NAMES:
            row[f"{split}_{metric}"] = _metric_value(metrics, split, metric)
    return row


def _selection_key(row: dict[str, Any]) -> tuple[float, float, float, int]:
    def finite(name: str) -> float:
        value = row.get(name)
        return float(value) if isinstance(value, (int, float)) else float("-inf")

    k = row.get("k")
    numeric_k = int(k) if isinstance(k, int) else 10**9
    return (
        finite("val_macro_f1"),
        finite("val_f1_neg"),
        finite("val_roc_auc"),
        -numeric_k,
    )


def _add_k5_deltas(rows: list[dict[str, Any]]) -> None:
    by_method: dict[str, dict[int, dict[str, Any]]] = {}
    for row in rows:
        if isinstance(row["k"], int):
            by_method.setdefault(row["method"], {})[int(row["k"])] = row
    for row in rows:
        baseline = by_method.get(row["method"], {}).get(5)
        for split in ("val", "test"):
            name = f"{split}_macro_f1"
            delta_name = f"delta_{name}_vs_k5"
            if baseline is None or row.get(name) is None or baseline.get(name) is None:
                row[delta_name] = None
            else:
                row[delta_name] = float(row[name]) - float(baseline[name])


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _plot(rows: list[dict[str, Any]], path: Path) -> str | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        return str(error)

    panel_rows = [row for row in rows if isinstance(row["k"], int)]
    methods = sorted({row["method"] for row in panel_rows})
    k_ticks = sorted({int(row["k"]) for row in panel_rows})
    boc = next((row for row in rows if row["method"] == "boc_stacking"), None)
    figure, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for axis, split in zip(axes, ("val", "test")):
        for method in methods:
            method_rows = sorted(
                (row for row in panel_rows if row["method"] == method),
                key=lambda row: int(row["k"]),
            )
            axis.plot(
                [int(row["k"]) for row in method_rows],
                [row[f"{split}_macro_f1"] for row in method_rows],
                marker="o",
                label=method,
            )
        if boc is not None and boc.get(f"{split}_macro_f1") is not None:
            axis.axhline(
                float(boc[f"{split}_macro_f1"]),
                color="black",
                linestyle="--",
                linewidth=1.2,
                label="boc_stacking (all)",
            )
        axis.set_title(f"{split.upper()} Macro-F1")
        axis.set_xlabel("Panel size K")
        axis.set_xticks(k_ticks)
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Macro-F1")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    figure.suptitle("Chronological team-size sensitivity")
    figure.tight_layout(rect=(0, 0.12, 1, 0.95))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)
    return None


def summarize(
    out_root: Path,
    methods: list[str],
    k_values: list[int],
    boc_metrics: Path,
) -> dict[str, Any]:
    rows = []
    missing = []
    for method in methods:
        for k in k_values:
            metrics_path = out_root / method / f"k_{k:03d}" / "metrics.json"
            if _is_complete_result(metrics_path, k):
                rows.append(_row(method, k, metrics_path))
            else:
                missing.append(
                    {
                        "method": method,
                        "k": k,
                        "status": "not_executed",
                        "metrics_path": metrics_path.as_posix(),
                    }
                )

    panel_rows = [row for row in rows if isinstance(row["k"], int)]
    winners = {
        method: max(
            (row for row in panel_rows if row["method"] == method),
            key=_selection_key,
        )
        for method in methods
        if any(row["method"] == method for row in panel_rows)
    }
    boc_row = (
        _row(
            "boc_stacking",
            "all",
            boc_metrics,
            source="primary_full_electorate_reference",
        )
        if boc_metrics.exists()
        else None
    )
    if boc_row is not None:
        rows.append(boc_row)
    _add_k5_deltas(rows)
    rows.sort(key=lambda row: (row["method"], int(row["k"]) if isinstance(row["k"], int) else 10**9))

    overall_panel = max(panel_rows, key=_selection_key) if panel_rows else None
    validation_candidates = [*panel_rows, *([boc_row] if boc_row is not None else [])]
    overall_with_boc = max(validation_candidates, key=_selection_key) if validation_candidates else None
    payload = {
        "protocol": {
            "split": CHRONOLOGICAL_SPLIT,
            "k_values": k_values,
            "observed_k_values_by_method": {
                method: sorted(
                    int(row["k"])
                    for row in panel_rows
                    if row["method"] == method and isinstance(row["k"], int)
                )
                for method in methods
            },
            "panel_methods": methods,
            "boc_k_applicable": False,
            "selection_rule": (
                "maximum validation Macro-F1; ties use validation F1-remove, "
                "validation ROC-AUC, then smaller K; TEST never selects K"
            ),
        },
        "rows": rows,
        "missing": missing,
        "winners_by_method": winners,
        "overall_panel_winner": overall_panel,
        "boc_reference": boc_row,
        "best_validation_configuration_including_boc_reference": overall_with_boc,
    }
    _write_json(out_root / "sensitivity_summary.json", payload)
    _write_csv(out_root / "sensitivity_summary.csv", rows)
    plot_error = _plot(rows, out_root / "k_sensitivity.png")
    if plot_error is not None:
        payload["plot_error"] = plot_error
        _write_json(out_root / "sensitivity_summary.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--votes-dir", default=DEFAULT_VOTES_DIR)
    parser.add_argument("--out-root", type=Path, default=Path(DEFAULT_OUT_ROOT))
    parser.add_argument("--causal-features", default=DEFAULT_CAUSAL_FEATURES)
    parser.add_argument("--embedding-dir", default=DEFAULT_EMBEDDING_DIR)
    parser.add_argument("--boc-metrics", type=Path, default=Path(DEFAULT_BOC_METRICS))
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=tuple(PANEL_METHODS),
        default=list(PANEL_METHODS),
    )
    parser.add_argument("--k-values", nargs="+", type=int, default=list(DEFAULT_K_VALUES))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    methods = list(dict.fromkeys(args.methods))
    k_values = sorted(set(args.k_values))
    if not k_values or any(k < 1 for k in k_values):
        parser.error("--k-values must contain positive integers")
    votes_dir = resolve_chronological_split(Path(args.votes_dir))
    args.out_root.mkdir(parents=True, exist_ok=True)
    run_status_path = args.out_root / "run_status.json"
    run_status: dict[str, str] = {}
    if run_status_path.exists():
        existing_status = _load_json(run_status_path)
        run_status = {
            str(name): str(status)
            for name, status in existing_status.items()
            if isinstance(name, str) and isinstance(status, str)
        }

    if not args.summarize_only:
        for method in methods:
            for k in k_values:
                run_name = f"{method}/k_{k:03d}"
                out_dir = args.out_root / method / f"k_{k:03d}"
                metrics_path = out_dir / "metrics.json"
                command = _command(
                    method,
                    k,
                    votes_dir,
                    out_dir,
                    args.causal_features,
                    args.embedding_dir,
                )
                print(f"Running {run_name}", flush=True)
                if not args.overwrite and _is_complete_result(metrics_path, k):
                    run_status[run_name] = "SKIPPED (complete metrics.json exists)"
                    print(run_status[run_name], flush=True)
                    _write_json(run_status_path, run_status)
                    continue
                if args.dry_run:
                    run_status[run_name] = "DRY RUN"
                    print(subprocess.list2cmdline(command), flush=True)
                    _write_json(run_status_path, run_status)
                    continue
                result = subprocess.run(command, cwd=_PROJECT_ROOT, check=False)
                run_status[run_name] = (
                    "OK" if result.returncode == 0 else f"FAILED (exit {result.returncode})"
                )
                _write_json(run_status_path, run_status)

    if args.dry_run:
        return
    summary = summarize(args.out_root, methods, k_values, args.boc_metrics)
    print("Validation-selected winners", flush=True)
    for method, winner in summary["winners_by_method"].items():
        print(
            f"  {method}: K={winner['k']} "
            f"VAL={winner['val_macro_f1']:.4f} TEST={winner['test_macro_f1']:.4f}",
            flush=True,
        )
    if summary["missing"]:
        print(f"Missing configurations: {len(summary['missing'])}", flush=True)
    failures = [name for name, status in run_status.items() if status.startswith("FAILED")]
    if failures:
        raise SystemExit(f"{len(failures)} sensitivity run(s) failed")


if __name__ == "__main__":
    main()
