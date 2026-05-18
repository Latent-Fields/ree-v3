#!/usr/bin/env python3
"""run_regression_suite.py -- staged wrapper around ree-v3/tests/.

Usage:
    run_regression_suite.py --preflight
    run_regression_suite.py --contracts
    run_regression_suite.py --changed <subsystem>
    run_regression_suite.py --list-subsystems

Exit codes:
    0  -- all selected tests passed (or no tests matched, which is treated
          as a non-error since the whole point of --changed is to be a
          light-touch pre-commit check)
    non-zero -- pytest exit code (test failures, collection errors, etc.)

The --changed mode resolves a subsystem key (usually a ree_core/
subdirectory name) to the contract tests that exercise the wiring
touched by that subsystem. When you've edited `ree_core/residue/field.py`,
`--changed residue` runs the MECH-094 / residue-write contracts and
skips the cingulate / determinism / env-boot tests.

The map is deliberately conservative -- it errs toward running extra
tests (e.g. `latent` runs determinism + boot-matrix + agent_boot because
encoder changes ripple everywhere) rather than missing coverage. If you
find a change that broke a test `--changed` didn't run, add the mapping
here.

Unknown subsystem keys fall back to a substring search over contract
test filenames, so `--changed bg` still works even though the subsystem
map key is `heartbeat`.

This script is a convenience wrapper only. It does not replace calling
`pytest` directly; CI and the runner can invoke pytest themselves.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TESTS_ROOT = REPO_ROOT / "tests"
CONTRACTS_DIR = TESTS_ROOT / "contracts"

# Subsystem -> list of contract test stems (filename minus the "test_" prefix
# and ".py" extension). Resolved against CONTRACTS_DIR at run time. Add new
# tests here so --changed picks them up.
#
# Guiding question: "If I edit code in this subsystem, which contract tests
# could plausibly break?" Err on the side of running one or two extra tests
# rather than missing a regression.
SUBSYSTEM_MAP: dict[str, list[str]] = {
    # Top-level aggregates.
    "all": [
        "agent_boot",
        "feature_flag_boot_matrix",
        "seed_determinism",
        "bg_gating_contracts",
        "imagined_acted_isolation",
        "sd032_wiring_contracts",
    ],
    # REEAgent itself touches every path.
    "agent": [
        "agent_boot",
        "feature_flag_boot_matrix",
        "seed_determinism",
        "bg_gating_contracts",
        "imagined_acted_isolation",
        "sd032_wiring_contracts",
    ],
    # Config / utils / CLI wiring is cross-cutting; boot matrix is the
    # quickest way to catch a flag-plumbing regression.
    "utils": ["feature_flag_boot_matrix", "agent_boot"],
    "config": ["feature_flag_boot_matrix", "agent_boot"],

    # Latent encoders (z_world, z_self, z_harm, z_harm_a, z_resource).
    # Changes here break determinism and flag-gated boot first.
    "latent": [
        "agent_boot",
        "feature_flag_boot_matrix",
        "seed_determinism",
    ],

    # Predictors (E1/E2/E3, harm forward models). BG gating contracts test
    # MECH-090 committed stepping which lives in E3+BetaGate.
    "predictors": [
        "agent_boot",
        "bg_gating_contracts",
        "seed_determinism",
    ],
    "e1": ["agent_boot", "seed_determinism"],
    "e2": ["agent_boot", "seed_determinism"],
    "e3": ["agent_boot", "bg_gating_contracts", "seed_determinism"],

    # Heartbeat / BetaGate drives the MECH-090/091 committed machinery.
    "heartbeat": ["bg_gating_contracts", "agent_boot"],

    # Hippocampal proposes trajectories consumed by E3 selection.
    "hippocampal": ["agent_boot", "bg_gating_contracts"],

    # ResidueField writes are gated by MECH-094 hypothesis_tag.
    "residue": ["imagined_acted_isolation", "agent_boot"],

    # Cingulate cluster (SD-032b-e) has its own wiring contracts.
    "cingulate": ["sd032_wiring_contracts", "feature_flag_boot_matrix"],

    # Neuromodulation (serotonin) is flag-gated via SWS/REM switches.
    "neuromodulation": ["feature_flag_boot_matrix"],

    # Goal state / drive_level.
    "goal": ["agent_boot", "feature_flag_boot_matrix"],

    # Environment (CausalGridWorldV2). Env changes break determinism and
    # the observation plumbing in sense() first.
    "environment": ["agent_boot", "seed_determinism"],

    # Comparator (SD-029 / MECH-256).
    "comparator": ["agent_boot"],
}


def _resolve_paths(subsystem: str) -> list[Path]:
    """Map a subsystem key to contract test paths.

    1. Exact match against SUBSYSTEM_MAP.
    2. Otherwise substring match against contract test filenames so e.g.
       `bg` finds test_bg_gating_contracts.py, `sd032` finds
       test_sd032_wiring_contracts.py.
    """
    key = subsystem.lower().strip("/").replace("ree_core/", "").split("/")[0]

    if key in SUBSYSTEM_MAP:
        paths: list[Path] = []
        for stem in SUBSYSTEM_MAP[key]:
            p = CONTRACTS_DIR / f"test_{stem}.py"
            if p.exists():
                paths.append(p)
        return paths

    # Fallback: substring match on existing contract test filenames.
    if CONTRACTS_DIR.exists():
        return sorted(p for p in CONTRACTS_DIR.glob("test_*.py") if key in p.stem)
    return []


def _run_pytest(paths: list[Path], extra_args: list[str]) -> int:
    cmd = [sys.executable, "-m", "pytest", "-q", "--tb=short"] + extra_args
    cmd += [str(p) for p in paths]
    print(f"[regression-suite] {' '.join(cmd)}", flush=True)
    return subprocess.call(cmd, cwd=str(REPO_ROOT))


def _print_subsystems() -> None:
    print("Known subsystem keys (see --changed <KEY>):")
    print()
    width = max(len(k) for k in SUBSYSTEM_MAP) + 2
    for key in sorted(SUBSYSTEM_MAP):
        stems = SUBSYSTEM_MAP[key]
        print(f"  {key.ljust(width)}-> {', '.join(stems)}")
    print()
    print("Unknown keys fall back to substring match on test filenames,")
    print("so e.g. `--changed bg` -> test_bg_gating_contracts.py.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the V3 regression suite.")
    layer = parser.add_mutually_exclusive_group(required=True)
    layer.add_argument("--preflight", action="store_true",
                       help="Runner / queue / machine-boot preflight (~3s).")
    layer.add_argument("--contracts", action="store_true",
                       help="Contract layer: all wiring tests (~14s).")
    layer.add_argument("--changed", type=str, metavar="SUBSYSTEM",
                       help="Run contract tests relevant to a ree_core/ subsystem. "
                            "See --list-subsystems for keys.")
    layer.add_argument("--list-subsystems", action="store_true",
                       help="Print the subsystem -> test map and exit.")
    parser.add_argument("pytest_args", nargs=argparse.REMAINDER,
                        help="Additional args forwarded to pytest.")
    args = parser.parse_args()

    if args.list_subsystems:
        _print_subsystems()
        return 0

    extra = list(args.pytest_args)
    if extra and extra[0] == "--":
        extra = extra[1:]

    if args.preflight:
        preflight_dir = TESTS_ROOT / "preflight"
        if not preflight_dir.exists():
            print("[regression-suite] preflight directory missing.", flush=True)
            return 1
        return _run_pytest([preflight_dir], extra)

    if args.contracts:
        if not CONTRACTS_DIR.exists():
            print("[regression-suite] contracts directory missing.", flush=True)
            return 1
        return _run_pytest([CONTRACTS_DIR], extra)

    if args.changed:
        paths = _resolve_paths(args.changed)
        if not paths:
            print(f"[regression-suite] no tests matched subsystem '{args.changed}'.",
                  flush=True)
            print("[regression-suite] known keys:",
                  ", ".join(sorted(SUBSYSTEM_MAP)), flush=True)
            return 0
        return _run_pytest(paths, extra)

    return 0


if __name__ == "__main__":
    sys.exit(main())
