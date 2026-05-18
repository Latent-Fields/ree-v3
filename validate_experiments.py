#!/usr/bin/env python3
"""
validate_experiments.py -- AST-based conformance check on experiments/.

Every script in ree-v3/experiments/v3_exq_*.py MUST end its
`if __name__ == "__main__":` block with a call to `emit_outcome(...)` from
the experiment_protocol module. This is the runner-conformance contract
that replaces the fragile stdout-regex-scraping handshake (see
experiment_protocol.py for context).

Usage:
    /opt/local/bin/python3 validate_experiments.py
    /opt/local/bin/python3 validate_experiments.py --strict      # exit 1 on any non-conforming script
    /opt/local/bin/python3 validate_experiments.py --paths a.py b.py

Default mode is REPORT: prints the non-conforming list and exits 0. The
runner / CI / pre-commit hook should invoke with --strict.

This file is ASCII-safe (cp1252 / Windows terminal compatible).
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parent
EXPERIMENTS_DIR = REPO_ROOT / "experiments"
EMIT_NAME = "emit_outcome"
PROTOCOL_MODULE = "experiment_protocol"


def _has_main_block(tree: ast.Module) -> Optional[ast.If]:
    """Return the `if __name__ == "__main__":` block, or None."""
    for node in tree.body:
        if not isinstance(node, ast.If):
            continue
        test = node.test
        # Pattern: __name__ == "__main__"  OR  "__main__" == __name__
        if isinstance(test, ast.Compare) and len(test.ops) == 1 and isinstance(test.ops[0], ast.Eq):
            left, right = test.left, test.comparators[0]
            names = []
            for n in (left, right):
                if isinstance(n, ast.Name):
                    names.append(n.id)
                elif isinstance(n, ast.Constant) and isinstance(n.value, str):
                    names.append(repr(n.value))
            if "__name__" in names and "'__main__'" in names:
                return node
    return None


def _walk_calls_for_emit(nodes: Sequence[ast.stmt]) -> bool:
    """True if any descendant call is to a name matching EMIT_NAME."""
    for stmt in nodes:
        for sub in ast.walk(stmt):
            if isinstance(sub, ast.Call):
                f = sub.func
                if isinstance(f, ast.Name) and f.id == EMIT_NAME:
                    return True
                if isinstance(f, ast.Attribute) and f.attr == EMIT_NAME:
                    return True
    return False


def _has_protocol_import(tree: ast.Module) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module == PROTOCOL_MODULE:
                return True
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == PROTOCOL_MODULE:
                    return True
    return False


def check_script(path: Path) -> Tuple[bool, str]:
    """Return (ok, reason). ok=True if script conforms or is exempt."""
    try:
        src = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        return False, f"could not read: {exc}"
    try:
        tree = ast.parse(src, filename=str(path))
    except SyntaxError as exc:
        return False, f"syntax error: {exc}"

    main = _has_main_block(tree)
    if main is None:
        # Library-style file with no entry point. Exempt.
        return True, "exempt: no __main__ block"

    if not _has_protocol_import(tree):
        return False, f"missing `from {PROTOCOL_MODULE} import {EMIT_NAME}` (or equivalent)"

    if not _walk_calls_for_emit(main.body):
        return False, f"missing `{EMIT_NAME}(...)` call inside `if __name__ == \"__main__\":` block"

    return True, "ok"


def _candidate_paths(paths: Sequence[str]) -> List[Path]:
    if paths:
        return [Path(p).resolve() for p in paths]
    return sorted(EXPERIMENTS_DIR.glob("v3_exq_*.py"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate experiment scripts conform to the runner contract.")
    parser.add_argument("--strict", action="store_true",
                        help="Exit 1 on any non-conforming script. Default mode is report-only.")
    parser.add_argument("--paths", nargs="*", default=[],
                        help="Specific scripts to check (default: all v3_exq_*.py in experiments/).")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress the per-script OK lines.")
    args = parser.parse_args()

    paths = _candidate_paths(args.paths)
    if not paths:
        print("[validate_experiments] no scripts found to check", flush=True)
        return 0

    n_ok = 0
    n_exempt = 0
    failures: List[Tuple[Path, str]] = []
    for p in paths:
        ok, reason = check_script(p)
        rel = p.relative_to(REPO_ROOT) if REPO_ROOT in p.parents or p == REPO_ROOT else p
        if ok:
            if reason.startswith("exempt"):
                n_exempt += 1
                if not args.quiet:
                    print(f"[validate_experiments] EXEMPT  {rel} ({reason})", flush=True)
            else:
                n_ok += 1
                if not args.quiet:
                    print(f"[validate_experiments] OK      {rel}", flush=True)
        else:
            failures.append((p, reason))

    print("", flush=True)
    print(f"[validate_experiments] checked {len(paths)} scripts: "
          f"{n_ok} OK, {n_exempt} exempt, {len(failures)} non-conforming", flush=True)
    if failures:
        print("", flush=True)
        print("[validate_experiments] Non-conforming scripts:", flush=True)
        for p, reason in failures:
            rel = p.relative_to(REPO_ROOT) if REPO_ROOT in p.parents or p == REPO_ROOT else p
            print(f"  - {rel}: {reason}", flush=True)
        if args.strict:
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
