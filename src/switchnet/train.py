from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import yaml

from .config import apply_overrides, load_config
from .data import build_dataloaders
from .engine import train_experiment, write_combined_summary
from .models import MODEL_NAMES, build_model
from .utils import resolve_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SwitchNet baselines and ablations")
    parser.add_argument("--config", required=True, help="YAML experiment configuration")
    parser.add_argument(
        "--model", default=None, choices=[*MODEL_NAMES, "all"], help="Override model.name"
    )
    parser.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--resume", default=None, help="Checkpoint path (single-model runs only)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = apply_overrides(load_config(args.config), args.set)
    model_names = list(MODEL_NAMES) if args.model == "all" else [args.model or cfg.model.name]
    if args.resume is not None and len(model_names) != 1:
        raise ValueError("--resume can only be used with one model")
    device = resolve_device(cfg.train.device)
    stamp = args.run_name or datetime.now().strftime("%Y%m%d-%H%M%S")
    group_dir = Path(cfg.logging.output_dir) / cfg.data.name / stamp
    group_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for model_name in model_names:
        set_seed(cfg.train.seed, cfg.train.deterministic)
        data = build_dataloaders(cfg.data, cfg.train.seed)
        cfg.model.name = model_name
        run_dir = group_dir / model_name / f"seed_{cfg.train.seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        with (run_dir / "config.resolved.yaml").open("w", encoding="utf-8") as handle:
            yaml.safe_dump(cfg.to_dict(), handle, sort_keys=False)
        print(f"\n=== {model_name} | {cfg.data.name} | device={device} ===")
        model = build_model(model_name, cfg.model, data.input_shape, data.num_classes)
        relative_error = getattr(model, "match_metadata", {}).get("relative_error")
        if relative_error is not None and relative_error > cfg.model.parameter_match_tolerance:
            print(
                f"Warning: parameter-match error for {model_name} is {relative_error:.2%}, above "
                f"tolerance {cfg.model.parameter_match_tolerance:.2%}."
            )
        summaries.append(train_experiment(model, data, cfg, run_dir, device, args.resume))
    write_combined_summary(group_dir / "combined_summary.csv", summaries)
    print(f"\nResults written to {group_dir.resolve()}")


if __name__ == "__main__":
    main()
