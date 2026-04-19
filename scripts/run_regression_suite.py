#!/usr/bin/env python3
"""run_regression_suite.py -- staged wrapper around ree-v3/tests/.

Usage:
    run_regression_suite.py --preflight
    run_regression_suite.py --contracts          # Phase B (not yet landed)
    run_regression_suite.py --changed <subsystem> # Phase B

Exit codes:
    0  -- all selected tests passed
    non-zero -- pytest exit code (test failures, collection errors, etc.)

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


def _run_pytest(paths: list[Path], extra_args: list[str]) -> int:
    cmd = [sys.executable, "-m", "pytest", "-q", "--tb=short"] + extra_args
    cmd += [str(p) for p in paths]
    print(f"[regression-suite] {' '.join(cmd)}", flush=True)
    return subprocess.call(cmd, cwd=str(REPO_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the V3 regression suite.")
    layer = parser.add_mutually_exclusive_group(required=True)
    layer.add_argument("--preflight", action="store_true",
                       help="Runner / queue / machine-boot preflight (~30s).")
    layer.add_argument("--contracts", action="store_true",
                       help="Contract layer (Phase B; not yet landed).")
    layer.add_argument("--changed", type=str, metavar="SUBSYSTEM",
                       help="Subsystem-scoped tests (Phase B; not yet landed).")
    parser.add_argument("pytest_args", nargs=argparse.REMAINDER,
                        help="Additional args forwarded to pytest.")
    args = parser.parse_args()

    extra = list(args.pytest_args)
    if extra and extra[0] == "--":
        extra = extra[1:]

    if args.preflight:
        paths = [TESTS_ROOT / "preflight"]
        return _run_pytest(paths, extra)

    if args.contracts:
        paths = [TESTS_ROOT / "contracts"]
        if not paths[0].exists():
            print("[regression-suite] contracts layer not yet landed (PR 2).", flush=True)
            return 0
        return _run_pytest(paths, extra)

    if args.changed:
        contracts_dir = TESTS_ROOT / "contracts"
        microprobes_dir = TESTS_ROOT / "microprobes"
        paths = []
        for d in (contracts_dir, microprobes_dir):
            if d.exists():
                paths += sorted(d.glob(f"test_{args.changed}*.py"))
        if not paths:
            print(f"[regression-suite] no tests matched subsystem '{args.changed}'.",
                  flush=True)
            return 0
        return _run_pytest(paths, extra)

    return 0


if __name__ == "__main__":
    sys.exit(main())
