from __future__ import annotations

import argparse
import itertools
import subprocess
import sys
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a Cartesian SwitchNet sweep")
    parser.add_argument("--base", required=True, help="Base experiment YAML")
    parser.add_argument("--sweep", required=True, help="Sweep YAML")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    with Path(args.sweep).open("r", encoding="utf-8") as handle:
        spec = yaml.safe_load(handle)
    grid = spec.get("grid", {})
    models = spec.get("models", ["all"])
    keys = list(grid)
    combinations = itertools.product(*(grid[key] for key in keys))
    for index, values in enumerate(combinations, start=1):
        overrides = [f"{key}={value}" for key, value in zip(keys, values)]
        for model in models:
            run_name = "sweep_" + "__".join(
                f"{key.replace('.', '-')}_{value}" for key, value in zip(keys, values)
            )
            command = [
                sys.executable,
                "-m",
                "switchnet.train",
                "--config",
                args.base,
                "--model",
                model,
                "--run-name",
                run_name,
            ]
            for override in overrides:
                command.extend(["--set", override])
            print(f"[{index}] {' '.join(command)}")
            if not args.dry_run:
                subprocess.run(command, check=True)


if __name__ == "__main__":
    main()

