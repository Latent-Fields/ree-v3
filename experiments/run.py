"""
REE-v3 Experiment Harness

Runs a single experiment suite and emits a governance-compatible run pack.

Usage:
    /opt/local/bin/python3 experiments/run.py \\
        --experiment v3_exq_001_z_separation \\
        --seed 0 \\
        --output-root /path/to/REE_assembly/evidence/experiments

Governance requirements:
  run_id ends _v3
  architecture_epoch: ree_hybrid_guardrails_v1
  Run pack format: claim_probe_{claim_id}/runs/{run_id}_v3/

V3 harness changes from V2:
  - Uses split observation dict (obs_body, obs_world) from CausalGridWorld V3
  - Agent trained with multi-rate clock (SD-006)
  - fatal_error_count tracked (governance requirement)
"""

import argparse
import importlib
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from experiments.pack_writer import (
    ExperimentPackWriter,
    deterministic_run_id,
    normalize_timestamp_utc,
    resolve_output_root,
    stable_config_hash,
)
from ree_core import __version__ as REE_VERSION

RUNNER_NAME = "ree-v3-harness"
RUNNER_VERSION = "3.0.0"


def _load_experiment_module(experiment_name: str):
    """Dynamically import an experiment module by name."""
    # Allow full module path or short name
    if "." not in experiment_name:
        module_path = f"experiments.{experiment_name}"
    else:
        module_path = experiment_name
    try:
        return importlib.import_module(module_path)
    except ImportError as e:
        raise ImportError(
            f"Could not import experiment '{experiment_name}': {e}\n"
            "Available experiments are in experiments/*.py"
        ) from e


def run_experiment(
    experiment_name: str,
    seed: int,
    output_root: Path,
    num_episodes: int = 20,
    steps_per_episode: int = 200,
    extra_kwargs: dict = None,
) -> dict:
    """
    Run a single experiment and return results.

    Args:
        experiment_name: Name of experiment module (e.g. "v3_exq_001_z_separation")
        seed:            Random seed
        output_root:     Root directory for run pack output
        num_episodes:    Episodes to run
        steps_per_episode: Steps per episode
        extra_kwargs:    Additional kwargs forwarded to experiment.run()

    Returns:
        Result dict with keys: status, metrics, summary_markdown, claim_ids,
        evidence_direction, experiment_type, fatal_error_count
    """
    extra_kwargs = extra_kwargs or {}

    mod = _load_experiment_module(experiment_name)

    # Set seeds
    torch.manual_seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass

    timestamp_utc = normalize_timestamp_utc()

    try:
        result = mod.run(
            seed=seed,
            num_episodes=num_episodes,
            steps_per_episode=steps_per_episode,
            **extra_kwargs,
        )
    except Exception as e:
        tb = traceback.format_exc()
        print(f"FATAL: experiment '{experiment_name}' raised exception:\n{tb}", file=sys.stderr)
        result = {
            "status": "FAIL",
            "metrics": {"fatal_error_count": 1.0},
            "summary_markdown": f"# FAIL\n\nFatal exception during run:\n\n```\n{tb}\n```",
            "claim_ids": [],
            "evidence_direction": "unknown",
            "experiment_type": experiment_name,
            "fatal_error_count": 1,
        }

    # Ensure fatal_error_count exists in metrics
    metrics = result.get("metrics", {})
    if "fatal_error_count" not in metrics:
        metrics["fatal_error_count"] = 0.0
    result["metrics"] = metrics

    # Build run_id
    run_id = deterministic_run_id(
        experiment_type=result.get("experiment_type", experiment_name),
        seed=seed,
        timestamp_utc=timestamp_utc,
    )

    # Write run pack
    writer = ExperimentPackWriter(
        output_root=output_root,
        repo_root=Path(__file__).resolve().parents[1],
        runner_name=RUNNER_NAME,
        runner_version=RUNNER_VERSION,
    )

    scenario = {
        "seed": seed,
        "num_episodes": num_episodes,
        "steps_per_episode": steps_per_episode,
        "ree_version": REE_VERSION,
    }

    pack = writer.write_pack(
        experiment_type=result.get("experiment_type", experiment_name),
        run_id=run_id,
        timestamp_utc=timestamp_utc,
        status=result["status"],
        metrics_values=result["metrics"],
        summary_markdown=result.get("summary_markdown", f"# {result['status']}\n"),
        scenario=scenario,
        claim_ids_tested=result.get("claim_ids", []),
        evidence_class="simulation",
        evidence_direction=result.get("evidence_direction", "unknown"),
    )

    print(f"Run pack written to: {pack.run_dir}")
    print(f"Status: {result['status']}")
    for k, v in result["metrics"].items():
        print(f"  {k}: {v}")

    result["run_id"] = run_id
    result["run_dir"] = str(pack.run_dir)
    return result


def main():
    parser = argparse.ArgumentParser(description="REE-v3 Experiment Harness")
    parser.add_argument("--experiment", required=True, help="Experiment module name")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--output-root", default=None, help="Output root directory")
    parser.add_argument("--num-episodes", type=int, default=20)
    parser.add_argument("--steps-per-episode", type=int, default=200)
    args = parser.parse_args()

    output_root = resolve_output_root(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    result = run_experiment(
        experiment_name=args.experiment,
        seed=args.seed,
        output_root=output_root,
        num_episodes=args.num_episodes,
        steps_per_episode=args.steps_per_episode,
    )

    exit_code = 0 if result["status"] == "PASS" else 1
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
