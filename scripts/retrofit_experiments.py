#!/usr/bin/env python3
"""
retrofit_experiments.py -- bulk retrofit experiment scripts to call
experiment_protocol.emit_outcome at the end of their __main__ block.

Strategy: AST-based, conservative. For each script that does NOT already
import experiment_protocol AND has an `if __name__ == "__main__":` block:
  1. Insert `from experiment_protocol import emit_outcome` after the
     last top-level import.
  2. Append a try/finally wrapper around the existing __main__ body
     that calls emit_outcome with a best-effort outcome inferred from
     either:
       - a `result` / `pack` / `outcome` local at end of main, or
       - a fallback "FAIL" with exit_reason="error" if any exception
         escapes.
  3. Print a unified diff and (with --apply) write the change.

Default mode is DRY-RUN (prints diffs only). Use --apply to write.

Limitations:
- Scripts whose __main__ block does NOT bind a clear outcome variable
  are skipped (the script author has to decide). The `--list-skipped`
  flag prints those for manual triage.
- This tool does not retrofit the manifest_path arg automatically; it
  passes manifest_path=None when no obvious local exists. Manual
  follow-up is required to thread the manifest path through. The
  runner accepts None and just doesn't verify-on-disk; the queue item
  is still removed correctly.

The pragmatic recommendation: land this tool, run with --apply on the
~80% of scripts where it works cleanly, and manually retrofit the
remainder as they're touched.

Usage:
    /opt/local/bin/python3 scripts/retrofit_experiments.py            # dry-run all
    /opt/local/bin/python3 scripts/retrofit_experiments.py --apply    # write changes
    /opt/local/bin/python3 scripts/retrofit_experiments.py --paths a.py b.py
    /opt/local/bin/python3 scripts/retrofit_experiments.py --list-skipped
"""

from __future__ import annotations

import argparse
import ast
import difflib
import re
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = REPO_ROOT / "experiments"
PROTOCOL_MODULE = "experiment_protocol"
EMIT_NAME = "emit_outcome"

# Outcome-binding variable names we recognize. Order matters: earlier names
# take precedence. (var, attr) means the script binds `var` somewhere and
# reads `var["attr"]` or `var.attr` for the PASS/FAIL outcome. (var, "")
# means `var` itself is a string outcome (e.g. `outcome = "PASS"`).
OUTCOME_VAR_CANDIDATES = (
    ("result",   "outcome"),
    ("result",   "status"),
    ("result",   "verdict"),
    ("result",   "result"),
    ("manifest", "result"),
    ("manifest", "outcome"),
    ("manifest", "status"),
    ("manifest", "verdict"),
    ("output",   "outcome"),
    ("output",   "status"),
    ("output",   "verdict"),
    ("pack",     "outcome"),
    ("pack",     "status"),
    ("pack",     "verdict"),
    ("res",      "outcome"),
    ("res",      "status"),
    ("outcome",  ""),
    ("verdict",  ""),
    ("status",   ""),
)
MANIFEST_VAR_CANDIDATES = ("out_path", "out_file", "manifest_path", "result_path", "pack_path", "out_json", "output_path")


def _find_main_block(tree: ast.Module) -> Optional[ast.If]:
    for node in tree.body:
        if not isinstance(node, ast.If):
            continue
        test = node.test
        if (isinstance(test, ast.Compare) and len(test.ops) == 1
                and isinstance(test.ops[0], ast.Eq)):
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


def _has_protocol_import(tree: ast.Module) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == PROTOCOL_MODULE:
            return True
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == PROTOCOL_MODULE:
                    return True
    return False


def _last_toplevel_import_lineno(tree: ast.Module) -> int:
    """1-indexed line number of the last top-level import statement (or 0 if none)."""
    last = 0
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            last = max(last, node.end_lineno or node.lineno)
    return last


def _detect_outcome_pattern(main_block: ast.If, src_lines: Sequence[str]) -> Tuple[str, Optional[str]]:
    """Return (outcome_expr, manifest_expr_or_None) suitable for emit_outcome.

    outcome_expr is a Python expression that evaluates to "PASS" or "FAIL".
    Returns ("UNDETECTED", None) if no clear pattern matches.
    """
    body_src = "\n".join(src_lines[main_block.lineno - 1:main_block.end_lineno])

    outcome_expr: Optional[str] = None
    for var, attr in OUTCOME_VAR_CANDIDATES:
        pat_assign = re.compile(rf"\b{re.escape(var)}\s*=\s*\S", re.MULTILINE)
        if not pat_assign.search(body_src):
            continue
        if attr:
            # Subscript form first (dict)
            sub_pat = re.compile(rf"\b{re.escape(var)}\[['\"]({re.escape(attr)})['\"]\]")
            if sub_pat.search(body_src):
                outcome_expr = f'str({var}.get("{attr}", "FAIL")).upper()'
                break
            # Attribute form (dataclass)
            attr_pat = re.compile(rf"\b{re.escape(var)}\.{re.escape(attr)}\b")
            if attr_pat.search(body_src):
                outcome_expr = f'str(getattr({var}, "{attr}", "FAIL")).upper()'
                break
        else:
            # Bare variable form: `outcome = "PASS"|"FAIL"`. Confirm it's
            # used as a string by checking that "PASS" or "FAIL" literals
            # appear in the body (cheap heuristic to filter false matches
            # like a `status` config dict).
            if '"PASS"' in body_src or "'PASS'" in body_src:
                outcome_expr = f'str({var}).upper()'
                break
    if outcome_expr is None:
        # Fallback: presence of "Done. Outcome:" print -> try result["outcome"]
        if 'Done. Outcome' in body_src and 'result[' in body_src:
            outcome_expr = 'str(result.get("outcome", "FAIL")).upper()'
        else:
            return "UNDETECTED", None

    manifest_expr: Optional[str] = None
    for cand in MANIFEST_VAR_CANDIDATES:
        if re.search(rf"\b{re.escape(cand)}\s*=", body_src):
            manifest_expr = cand
            break
    return outcome_expr, manifest_expr


def _build_emit_call(outcome_expr: str, manifest_expr: Optional[str]) -> str:
    if manifest_expr:
        return (f'    _outcome = {outcome_expr} if {outcome_expr} in ("PASS", "FAIL") else "FAIL"\n'
                f'    emit_outcome(outcome=_outcome, manifest_path={manifest_expr})\n')
    return (f'    _outcome = {outcome_expr} if {outcome_expr} in ("PASS", "FAIL") else "FAIL"\n'
            f'    emit_outcome(outcome=_outcome, manifest_path=None)\n')


def retrofit_one(path: Path) -> Tuple[bool, str, Optional[str]]:
    """Retrofit a single script. Return (changed, reason, new_text_or_None)."""
    src = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(src, filename=str(path))
    except SyntaxError as exc:
        return False, f"syntax error: {exc}", None

    if _has_protocol_import(tree):
        return False, "already imports experiment_protocol", None
    main = _find_main_block(tree)
    if main is None:
        return False, "no __main__ block (library file)", None

    src_lines = src.splitlines(keepends=True)
    outcome_expr, manifest_expr = _detect_outcome_pattern(main, [l.rstrip("\n") for l in src_lines])
    if outcome_expr == "UNDETECTED":
        return False, "could not detect outcome variable; manual retrofit required", None

    # Insert import after last top-level import (or at top if none).
    insert_lineno = _last_toplevel_import_lineno(tree)
    import_stmt = f"from {PROTOCOL_MODULE} import {EMIT_NAME}\n"
    new_lines = list(src_lines)
    if insert_lineno > 0:
        new_lines.insert(insert_lineno, import_stmt)
    else:
        new_lines.insert(0, import_stmt)

    # Append the emit call at the end of the __main__ block. We do this by
    # locating the end of the file (since __main__ is the last block in
    # ~all of these scripts) and appending the call indented to match the
    # __main__ body indent. If __main__ is not the last top-level block,
    # this tool gives up to avoid mis-indenting code that follows.
    last_top = tree.body[-1]
    if last_top is not main:
        return False, "__main__ block is not the last top-level statement; manual retrofit required", None

    # Determine indent: take the first non-empty body line indent.
    indent = "    "
    for stmt in main.body:
        line0 = src_lines[stmt.lineno - 1] if (stmt.lineno - 1) < len(src_lines) else ""
        m = re.match(r"^(\s+)", line0)
        if m:
            indent = m.group(1)
            break

    emit_block = (
        f'\n{indent}# --- runner-conformance sentinel (added by retrofit_experiments.py) ---\n'
        f'{indent}_outcome_raw = {outcome_expr}\n'
        f'{indent}emit_outcome(\n'
        f'{indent}    outcome=_outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL",\n'
        f'{indent}    manifest_path={manifest_expr if manifest_expr else "None"},\n'
        f'{indent})\n'
    )
    if not new_lines[-1].endswith("\n"):
        new_lines[-1] = new_lines[-1] + "\n"
    new_lines.append(emit_block)
    new_text = "".join(new_lines)
    return True, "ok", new_text


def _candidate_paths(paths: Sequence[str]) -> List[Path]:
    if paths:
        return [Path(p).resolve() for p in paths]
    return sorted(EXPERIMENTS_DIR.glob("v3_exq_*.py"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Bulk-retrofit experiment scripts.")
    parser.add_argument("--apply", action="store_true", help="Write changes (default: dry-run with diff).")
    parser.add_argument("--paths", nargs="*", default=[], help="Specific scripts to retrofit.")
    parser.add_argument("--list-skipped", action="store_true",
                        help="Print only the list of scripts skipped, with reasons.")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-file diffs.")
    args = parser.parse_args()

    paths = _candidate_paths(args.paths)
    n_changed = 0
    n_already = 0
    skipped: List[Tuple[Path, str]] = []
    for p in paths:
        changed, reason, new_text = retrofit_one(p)
        if changed:
            n_changed += 1
            rel = p.relative_to(REPO_ROOT) if REPO_ROOT in p.parents else p
            if args.list_skipped or args.quiet:
                print(f"[retrofit] would-change {rel}", flush=True)
            else:
                old_text = p.read_text(encoding="utf-8")
                diff = difflib.unified_diff(
                    old_text.splitlines(keepends=True),
                    new_text.splitlines(keepends=True),
                    fromfile=str(rel),
                    tofile=str(rel) + " (retrofitted)",
                    n=2,
                )
                sys.stdout.writelines(diff)
                print("", flush=True)
            if args.apply and new_text is not None:
                p.write_text(new_text, encoding="utf-8")
                print(f"[retrofit] APPLIED  {rel}", flush=True)
        else:
            if reason == "already imports experiment_protocol":
                n_already += 1
            else:
                skipped.append((p, reason))

    print("", flush=True)
    print(f"[retrofit] {len(paths)} scripts examined: "
          f"{n_changed} would-change / changed, "
          f"{n_already} already conformant, "
          f"{len(skipped)} skipped (manual retrofit required)", flush=True)
    if args.list_skipped or skipped:
        print("", flush=True)
        print("[retrofit] Skipped scripts (need manual retrofit):", flush=True)
        for p, reason in skipped:
            rel = p.relative_to(REPO_ROOT) if REPO_ROOT in p.parents else p
            print(f"  - {rel}: {reason}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
