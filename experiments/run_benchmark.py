# -*- coding: utf-8 -*-
"""Unified entry point for reaction-scope validation benchmarks.

The research scripts under ``Examples/`` are still the source of the individual
analyses. This runner turns them into a cleaner paper-style workflow driven by a
single JSON config.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


TASKS = {
    "dimension_model": {
        "script": "Examples/Sampling_eval/Model_eval/Regression/compare_scope_dimensions_models.py",
        "plot_script": "Examples/Sampling_eval/Model_eval/Regression/plot_scope_dimension_results.py",
        "plot_args": {"prefix": "out_prefix", "plot_prefix": "plot_prefix", "dims": "dims"},
    },
    "repeated_ci": {
        "script": "Examples/Sampling_eval/Model_eval/Regression/repeated_scope_dimension_ci.py",
        "plot_script": "Examples/Sampling_eval/Model_eval/Regression/plot_repeated_ci_results.py",
        "plot_args": {"prefix": "out_prefix", "plot_prefix": "plot_prefix", "dims": "dims"},
    },
    "boundary_sampling": {
        "script": "Examples/Sampling_eval/Model_eval/Regression/boundary_sampling_benchmark.py",
        "plot_script": "Examples/Sampling_eval/Model_eval/Regression/plot_boundary_sampling_results.py",
        "plot_args": {"prefix": "out_prefix", "plot_prefix": "plot_prefix", "dims": "dims"},
    },
    "reference_sampling": {
        "script": "Examples/Sampling_eval/Model_eval/Regression/reference_sampling_coverage.py",
        "plot_script": "Examples/Sampling_eval/Model_eval/Regression/plot_reference_sampling_coverage.py",
        "plot_args": {"prefix": "out_prefix", "plot_prefix": "plot_prefix", "dims": "dims"},
    },
    "prospective_simulation": {
        "script": "Examples/Sampling_eval/Model_eval/Regression/prospective_boundary_simulation.py",
        "plot_script": "Examples/Sampling_eval/Model_eval/Regression/plot_prospective_boundary_simulation.py",
        "plot_args": {"prefix": "out_prefix", "plot_prefix": "plot_prefix", "dims": "dims"},
    },
}

PATH_PARAMS = {"data", "space", "space_features", "reference", "reference_features"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run reaction-scope validation tasks from a JSON config.")
    parser.add_argument("--config", required=True, help="Path to a benchmark config JSON file.")
    parser.add_argument("--only", nargs="+", help="Optional task names to run.")
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used for child benchmark scripts. Defaults to the current interpreter.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    return parser.parse_args()


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if "tasks" not in config or not isinstance(config["tasks"], list):
        raise ValueError("Config must contain a 'tasks' list.")
    return config


def resolve_path(root: Path, value: str | None, default: str | None = None) -> Path:
    raw = value if value is not None else default
    if raw is None:
        raise ValueError("Missing required path value.")
    path = Path(raw)
    return path if path.is_absolute() else root / path


def flag_name(key: str) -> str:
    return "--" + key.replace("_", "-")


def append_arg(command: list[str], key: str, value: Any) -> None:
    if value is None:
        return
    if isinstance(value, bool):
        if value:
            command.append(flag_name(key))
        return
    command.append(flag_name(key))
    if isinstance(value, list):
        command.extend(str(item) for item in value)
    else:
        command.append(str(value))


def build_main_command(root: Path, task: dict[str, Any], python_executable: str) -> tuple[list[str], Path]:
    task_type = task["type"]
    metadata = TASKS[task_type]
    params = dict(task.get("params", {}))

    output_dir = resolve_path(root, task.get("output_dir"), "results")
    output_dir.mkdir(parents=True, exist_ok=True)

    for key in PATH_PARAMS:
        if key in params:
            params[key] = str(resolve_path(root, params[key]))
    params.setdefault("out_prefix", task.get("out_prefix", task["name"]))

    command = [python_executable, str(root / metadata["script"])]
    for key, value in params.items():
        append_arg(command, key, value)
    return command, output_dir


def build_plot_command(root: Path, task: dict[str, Any], python_executable: str) -> tuple[list[str], Path] | None:
    task_type = task["type"]
    metadata = TASKS[task_type]
    if not task.get("plot", False):
        return None

    output_dir = resolve_path(root, task.get("output_dir"), "results")
    params = dict(task.get("params", {}))
    params.setdefault("out_prefix", task.get("out_prefix", task["name"]))
    plot_prefix = task.get("plot_prefix", params["out_prefix"])

    command = [python_executable, str(root / metadata["plot_script"])]
    for plot_key, source_key in metadata["plot_args"].items():
        if source_key == "out_prefix":
            value = params["out_prefix"]
        elif source_key == "plot_prefix":
            value = plot_prefix
        else:
            value = params.get(source_key)
        append_arg(command, plot_key, value)
    return command, output_dir


def run_command(command: list[str], cwd: Path, dry_run: bool) -> None:
    shown = " ".join(f'"{part}"' if " " in part else part for part in command)
    print(f"\n[run] cwd={cwd}")
    print(shown)
    if dry_run:
        return
    subprocess.run(command, cwd=cwd, check=True)


def main() -> None:
    args = parse_args()
    root = project_root()
    config = load_config(resolve_path(root, args.config))
    selected = set(args.only or [])

    print("=== Reaction-scope validation runner ===")
    print(f"Project root: {root}")
    print(f"Config: {resolve_path(root, args.config)}")
    print(f"Child Python: {args.python}")

    for task in config["tasks"]:
        name = task.get("name")
        task_type = task.get("type")
        if not name:
            raise ValueError("Each task needs a name.")
        if task_type not in TASKS:
            raise ValueError(f"Unknown task type for {name}: {task_type}")
        if selected and name not in selected:
            continue

        print(f"\n=== Task: {name} ({task_type}) ===")
        command, cwd = build_main_command(root, task, args.python)
        run_command(command, cwd, args.dry_run)

        plot_command = build_plot_command(root, task, args.python)
        if plot_command is not None:
            command, cwd = plot_command
            run_command(command, cwd, args.dry_run)

    print("\nDone.")


if __name__ == "__main__":
    main()
