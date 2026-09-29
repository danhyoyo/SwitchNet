from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable

import numpy as np


def comparison_config(record: dict[str, Any]) -> dict[str, Any]:
    """Return fields that must stay fixed for a scientifically valid comparison."""
    config = deepcopy(record.get("config") or {})
    if not config:
        return {"dataset": record.get("dataset"), "legacy_summary": True}
    config.get("model", {}).pop("name", None)
    config.get("train", {}).pop("seed", None)
    config.get("train", {}).pop("device", None)
    config.get("data", {}).pop("root", None)
    config.get("data", {}).pop("download", None)
    config.pop("logging", None)
    return config


def signature(config: dict[str, Any]) -> str:
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


def mean_std(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    if not len(finite):
        return float("nan"), float("nan")
    return float(finite.mean()), float(finite.std(ddof=1)) if len(finite) > 1 else 0.0


def write_bar_svg(
    path: Path,
    title: str,
    ylabel: str,
    labels: list[str],
    means: list[float],
    errors: list[float],
) -> None:
    """Write a dependency-free SVG chart suitable for headless sweep jobs."""
    width, height = 900, 500
    left, right, top, bottom = 90, 30, 60, 110
    chart_width, chart_height = width - left - right, height - top - bottom
    lower = min([0.0, *(mean - error for mean, error in zip(means, errors))])
    upper = max([0.0, *(mean + error for mean, error in zip(means, errors))])
    if upper == lower:
        upper = lower + 1.0
    padding = (upper - lower) * 0.08
    lower, upper = lower - padding, upper + padding

    def y(value: float) -> float:
        return top + (upper - value) / (upper - lower) * chart_height

    slot = chart_width / max(1, len(labels))
    bar_width = slot * 0.62
    baseline = y(0.0)
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="30" text-anchor="middle" font-family="sans-serif" '
        f'font-size="20">{html.escape(title)}</text>',
        f'<line x1="{left}" y1="{baseline:.2f}" x2="{width-right}" y2="{baseline:.2f}" '
        'stroke="#333"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#333"/>',
        f'<text x="20" y="{top + chart_height / 2}" text-anchor="middle" '
        f'transform="rotate(-90 20 {top + chart_height / 2})" font-family="sans-serif" '
        f'font-size="14">{html.escape(ylabel)}</text>',
    ]
    for index, (label, mean, error) in enumerate(zip(labels, means, errors)):
        center = left + slot * (index + 0.5)
        bar_top, bar_bottom = min(y(mean), baseline), max(y(mean), baseline)
        elements.append(
            f'<rect x="{center - bar_width / 2:.2f}" y="{bar_top:.2f}" '
            f'width="{bar_width:.2f}" height="{max(1.0, bar_bottom - bar_top):.2f}" '
            'fill="#4776b4"/>'
        )
        elements.append(
            f'<line x1="{center:.2f}" y1="{y(mean-error):.2f}" x2="{center:.2f}" '
            f'y2="{y(mean+error):.2f}" stroke="#222"/>'
        )
        elements.append(
            f'<text x="{center:.2f}" y="{height-bottom+22}" text-anchor="end" '
            f'transform="rotate(-25 {center:.2f} {height-bottom+22})" '
            f'font-family="sans-serif" font-size="12">{html.escape(label)}</text>'
        )
        elements.append(
            f'<text x="{center:.2f}" y="{bar_top-7:.2f}" text-anchor="middle" '
            f'font-family="sans-serif" font-size="11">{mean:.4g}</text>'
        )
    elements.append("</svg>")
    path.write_text("\n".join(elements), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate compatible SwitchNet runs")
    parser.add_argument("--runs", default="runs")
    parser.add_argument("--output", default="reports")
    args = parser.parse_args()

    # Re-runs of the same model/config/seed are de-duplicated; the newest result wins.
    unique: dict[tuple[str, str, str], tuple[float, dict[str, Any], dict[str, Any]]] = {}
    legacy = 0
    for path in Path(args.runs).rglob("summary.json"):
        with path.open("r", encoding="utf-8") as handle:
            record = json.load(handle)
        config = comparison_config(record)
        legacy += int(bool(config.get("legacy_summary")))
        group_id = signature(config)
        key = (group_id, str(record["model"]), str(record["seed"]))
        candidate = (path.stat().st_mtime, record, config)
        if key not in unique or candidate[0] > unique[key][0]:
            unique[key] = candidate
    if not unique:
        raise SystemExit(f"No summary.json files found under {args.runs}")
    if legacy:
        print(f"Warning: {legacy} legacy summaries lack resolved config and have limited grouping.")

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    configs: dict[str, dict[str, Any]] = {}
    for (group_id, _model, _seed), (_mtime, record, config) in unique.items():
        grouped[group_id].append(record)
        configs[group_id] = config

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    metrics: dict[str, Callable[[dict[str, Any]], float]] = {
        "accuracy": lambda row: row["test"]["accuracy"],
        "macro_f1": lambda row: row["test"]["macro_f1"],
        "training_seconds": lambda row: row["training_seconds"],
        "parameters": lambda row: row["parameters"],
        "score_selectivity": lambda row: (row.get("specialization") or {}).get(
            "mean_score_selectivity", np.nan
        ),
        "one_vs_rest_auroc": lambda row: (row.get("specialization") or {}).get(
            "mean_one_vs_rest_auroc", np.nan
        ),
        "calibration_error": lambda row: (row.get("specialization") or {}).get(
            "expected_calibration_error_15_bins", np.nan
        ),
    }
    aggregate_rows: list[dict[str, Any]] = []
    for group_id, records in sorted(grouped.items()):
        group_dir = output / group_id
        group_dir.mkdir(parents=True, exist_ok=True)
        with (group_dir / "comparison_config.json").open("w", encoding="utf-8") as handle:
            json.dump(configs[group_id], handle, indent=2, sort_keys=True)
        by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in records:
            by_model[record["model"]].append(record)
        models = sorted(by_model)
        for model in models:
            row: dict[str, Any] = {
                "experiment": group_id,
                "dataset": records[0].get("dataset"),
                "model": model,
                "runs": len(by_model[model]),
            }
            for metric, getter in metrics.items():
                mean, std = mean_std([getter(item) for item in by_model[model]])
                row[f"{metric}_mean"] = mean
                row[f"{metric}_std"] = std
            aggregate_rows.append(row)
        for metric, getter in metrics.items():
            values = [
                np.asarray([getter(item) for item in by_model[model]], dtype=float)
                for model in models
            ]
            available = [
                (model, value)
                for model, value in zip(models, values)
                if np.isfinite(value).any()
            ]
            if not available:
                continue
            plot_models = [item[0] for item in available]
            means = [float(np.nanmean(item[1])) for item in available]
            errors = [
                float(np.nanstd(item[1], ddof=1))
                if np.isfinite(item[1]).sum() > 1
                else 0.0
                for item in available
            ]
            label = metric.replace("_", " ").title()
            write_bar_svg(
                group_dir / f"{metric}.svg",
                f"SwitchNet ablation: {metric.replace('_', ' ')} ({group_id})",
                label,
                plot_models,
                means,
                errors,
            )

    if aggregate_rows:
        with (output / "aggregate.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(aggregate_rows[0]))
            writer.writeheader()
            writer.writerows(aggregate_rows)
    print(
        f"Aggregated {len(unique)} de-duplicated runs into {len(grouped)} compatible "
        f"experiment groups under {output.resolve()}"
    )


if __name__ == "__main__":
    main()
