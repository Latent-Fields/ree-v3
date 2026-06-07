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

# Readiness-gate static lint (proposal_trivial_prediction_readiness_gate_2026-06-06).
# A diagnostic/baseline script whose interpretation grid self-routes to one of
# these "the substrate is the limit" labels is making a high-stakes claim that is
# only legitimate on a substrate trained/configured to the level the claim
# presupposes. Such a script must declare a readiness-kind precondition + a
# load_bearing criterion so the indexer can recompute the self-route's premise.
SUBSTRATE_VERDICT_LABELS = {"substrate_ceiling", "substrate_conditional", "does_not_support"}
SUBSTRATE_VERDICT_SUFFIXES = ("_nondiscriminative", "_unmeetable")

# Same-statistic readiness heuristic (V3-EXQ-643 GAP). A readiness precondition
# must assert the SAME statistic the load-bearing criterion routes on. The
# recurring failure is a magnitude / mean-abs readiness check standing in for a
# criterion that actually gates on a cross-candidate RANGE (spread / variance /
# diversity): a uniform offset has large mean-abs but ~0 range, so the readiness
# check passes while the criterion's precondition is unmet. These token lists
# drive a best-effort name-scan WARN; see readiness_lint() for the known limits.
MAGNITUDE_NAME_TOKENS = (
    "abs_mean", "mean_abs", "max_abs", "abs_max", "_abs", "abs_",
    "magnitude", "_norm", "norm_", "l2norm", "absmean",
)
RANGE_NAME_TOKENS = (
    "range", "spread", "diversity", "variance", "_var", "var_", "entropy", "stdev", "_std", "std_",
)


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


def _readiness_and_criterion_names(tree: ast.Module) -> Tuple[List[str], List[str]]:
    """Best-effort extraction of (readiness_precondition_names, criterion_names)
    from dict literals in the script.

    A readiness-kind precondition dict carries name + measured + threshold (and
    NOT load_bearing/passed); a criterion dict carries a name with load_bearing
    or passed. Names assembled at runtime (f-strings / concatenation) are
    invisible to this scan -- accepted limitation (same class as readiness_lint).
    """
    readiness_names: List[str] = []
    criterion_names: List[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        str_keys = {}
        for k, v in zip(node.keys, node.values):
            if isinstance(k, ast.Constant) and isinstance(k.value, str):
                str_keys[k.value] = v
        name_node = str_keys.get("name")
        if not (isinstance(name_node, ast.Constant) and isinstance(name_node.value, str)):
            continue
        name = name_node.value
        is_criterion = ("load_bearing" in str_keys) or ("passed" in str_keys)
        is_readiness = ("measured" in str_keys) and ("threshold" in str_keys)
        if is_criterion:
            criterion_names.append(name)
        elif is_readiness:
            readiness_names.append(name)
    return readiness_names, criterion_names


def _name_has(name: str, tokens: Sequence[str]) -> bool:
    n = name.lower()
    return any(t in n for t in tokens)


def readiness_lint(path: Path) -> Optional[str]:
    """WARN-only readiness-gate lint. Return a warning string, or None.

    For a `diagnostic` / `baseline` script whose interpretation grid routes to a
    SUBSTRATE_VERDICT_LABELS label (or a `*_nondiscriminative` / `*_unmeetable`
    suffix), it raises up to two WARNs:

    (1) MISSING-STRUCTURE: no readiness-kind precondition (a numeric
        `measured`+`threshold` pair) and/or no `load_bearing` criterion -- the
        trivial-prediction signature the author cannot see (V3-EXQ-642/264/620)
        and the V3-EXQ-621a aggregation-vacuity pattern, respectively.

    (2) SAME-STATISTIC MISMATCH (V3-EXQ-643 GAP): a readiness precondition is
        named like a MAGNITUDE (abs / mean_abs / max_abs / norm / magnitude)
        while a criterion name or a routed-metric string references a RANGE /
        spread / variance / diversity. A magnitude (e.g. mean-abs) can be large
        while the cross-candidate range is ~0 (a uniform offset), so a
        magnitude readiness check can PASS while a range-gated criterion's
        precondition is unmet -- the readiness `measured` must assert the SAME
        statistic the load-bearing criterion routes on.

    Implementation is the lightest viable static check: a string/AST scan over the
    script's literals + dict-literal name fields. It does NOT statically interpret
    the interpretation-grid control flow, so it has known limitations -- it can
    MISS a verdict label or a metric name assembled at runtime (f-string /
    concatenation), can MISS a magnitude readiness whose name carries no
    magnitude token, and can OVER-FIRE if a label/key/metric appears only in a
    comment/docstring or if an unrelated magnitude readiness coexists with an
    unrelated range metric. WARN-only by design (proposal Q3 warn-then-error);
    never affects the exit code. Harden to ERROR after a cycle of real
    post-convention diagnostics exists.
    """
    try:
        src = path.read_text(encoding="utf-8")
        tree = ast.parse(src, filename=str(path))
    except (OSError, UnicodeDecodeError, SyntaxError):
        return None  # check_script already reports unreadable / syntax errors

    strings = set()
    purposes = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            strings.add(node.value)
        if isinstance(node, ast.keyword) and node.arg == "experiment_purpose":
            val = node.value
            if isinstance(val, ast.Constant) and isinstance(val.value, str):
                purposes.add(val.value)
        if isinstance(node, ast.Assign):
            # Match both the lowercase keyword-style `experiment_purpose = "..."`
            # and the canonical module constant `EXPERIMENT_PURPOSE = "diagnostic"`
            # (the convention real scripts use, then pass via
            # `"experiment_purpose": EXPERIMENT_PURPOSE`).
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id.lower() == "experiment_purpose":
                    val = node.value
                    if isinstance(val, ast.Constant) and isinstance(val.value, str):
                        purposes.add(val.value)

    if not (purposes & {"diagnostic", "baseline"}):
        return None
    routes_to_verdict = any(
        s in SUBSTRATE_VERDICT_LABELS or s.endswith(SUBSTRATE_VERDICT_SUFFIXES)
        for s in strings
    )
    if not routes_to_verdict:
        return None

    issues: List[str] = []

    # WARN (1): missing readiness precondition and/or load_bearing criterion.
    has_readiness = "measured" in strings and "threshold" in strings
    has_load_bearing = "load_bearing" in strings
    if not (has_readiness and has_load_bearing):
        missing = []
        if not has_readiness:
            missing.append("no readiness-kind precondition (numeric measured+threshold)")
        if not has_load_bearing:
            missing.append("no criterion tagged load_bearing")
        issues.append("routes to a substrate-verdict label but " + " AND ".join(missing)
                      + " -- add a P0 readiness-assert")

    # WARN (2): same-statistic mismatch (V3-EXQ-643). A readiness precondition
    # named like a magnitude alongside a criterion / routed metric that
    # references a range/spread/variance/diversity. Best-effort name-scan;
    # see this function's docstring for the over/under-fire limits.
    readiness_names, criterion_names = _readiness_and_criterion_names(tree)
    magnitude_readiness = sorted({n for n in readiness_names if _name_has(n, MAGNITUDE_NAME_TOKENS)})
    if magnitude_readiness:
        range_in_criteria = sorted({n for n in criterion_names if _name_has(n, RANGE_NAME_TOKENS)})
        range_in_strings = any(_name_has(s, RANGE_NAME_TOKENS) for s in strings)
        if range_in_criteria or range_in_strings:
            where = ("criterion name(s) " + ", ".join(range_in_criteria)
                     if range_in_criteria else "a routed-metric string")
            issues.append(
                "possible same-statistic mismatch (V3-EXQ-643): readiness "
                "precondition(s) " + ", ".join(magnitude_readiness)
                + " look like a MAGNITUDE while " + where + " references a "
                "RANGE/spread/variance/diversity -- the readiness `measured` "
                "must assert the SAME statistic the load-bearing criterion "
                "routes on (assert a range/spread, not a mean-abs/norm)")

    if not issues:
        return None
    return (" ; ".join(issues)
            + " (see /queue-experiment P0 readiness-assert + "
            + "proposal_trivial_prediction_readiness_gate_2026-06-06)")


# Arm-reuse fingerprint enforcement (arm_reuse_fingerprint_plan.md; determinism
# gate closed + ratified 2026-06-07). A multi-arm (seed x arm) experiment writes
# per-arm rows under "arm_results"; each cell MUST (1) reset_all_rng(seed) at cell
# entry and (2) emit a per-cell fingerprint -- either via the low-level pair
# (reset_all_rng + compute_arm_fingerprint) or the bundled arm_cell() helper.
# Without both, the cell is order-dependent and never safely reusable.
_ARM_RESULTS_KEY = "arm_results"
_FP_EMIT_NAMES = ("compute_arm_fingerprint", "arm_cell")   # arm_cell stamps the fp
_RNG_RESET_NAMES = ("reset_all_rng", "arm_cell")           # arm_cell resets on enter
_ARM_FP_EXEMPT_MARKER = "ARM_FINGERPRINT_EXEMPT"            # opt-out constant/marker


def arm_fingerprint_lint(path: Path) -> Optional[str]:
    """Multi-arm fingerprint-emission check. Return an issue string, or None.

    A script is treated as multi-arm iff it writes the canonical manifest key
    "arm_results" (the per-(seed x arm) cell rows the indexer + reuse system key
    on). Such a script MUST discharge both per-cell obligations: a complete RNG
    reset at cell entry AND a fingerprint emission -- satisfied by either the
    low-level `reset_all_rng` + `compute_arm_fingerprint` pair or the bundled
    `arm_cell()` context manager (which does both). Missing either is the issue.

    Opt-out: a script may declare `ARM_FINGERPRINT_EXEMPT = "<reason>"` (e.g. a
    legitimately single-cell run that nonetheless writes an arm_results list, or
    a stateful design the plan marks reuse-ineligible by construction). The
    marker suppresses the check.

    Static name-scan only (same class of limitation as readiness_lint): it keys
    on plain identifier/string presence, so it can over-fire if "arm_results"
    appears only in a comment/docstring, and can miss a helper aliased under a
    different name. The remedy in both directions is cheap (add the emit, or add
    the exempt marker). Whether this blocks is decided by the caller in main():
    a hard failure when the script is named explicitly via --paths (the
    /queue-experiment authoring path), advisory otherwise (grandfathers the
    pre-2026-06-07 backlog).
    """
    try:
        src = path.read_text(encoding="utf-8")
        tree = ast.parse(src, filename=str(path))
    except (OSError, UnicodeDecodeError, SyntaxError):
        return None  # check_script already reports unreadable / syntax errors

    names: set = set()
    strings: set = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.alias):
            names.add((node.asname or node.name).split(".")[-1])
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            strings.add(node.value)

    if _ARM_FP_EXEMPT_MARKER in names or _ARM_FP_EXEMPT_MARKER in strings:
        return None
    if _ARM_RESULTS_KEY not in strings:
        return None  # not a multi-arm grid script

    has_emit = any(n in names for n in _FP_EMIT_NAMES)
    has_reset = any(n in names for n in _RNG_RESET_NAMES)
    if has_emit and has_reset:
        return None

    missing = []
    if not has_reset:
        missing.append("a per-cell reset_all_rng(seed) (or arm_cell())")
    if not has_emit:
        missing.append("a per-cell compute_arm_fingerprint(...) (or arm_cell().stamp())")
    return ("writes 'arm_results' (multi-arm) but is missing "
            + " AND ".join(missing)
            + " -- emit a per-cell arm_fingerprint via experiments/_lib/arm_fingerprint.py "
            + "(arm_cell() discharges both). Exempt with ARM_FINGERPRINT_EXEMPT = \"<reason>\". "
            + "See arm_reuse_fingerprint_plan.md + /queue-experiment.")


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

    # Arm-fingerprint enforcement is HARD only when scripts are named explicitly
    # via --paths (the /queue-experiment authoring path). In full-glob mode it is
    # advisory, so the pre-2026-06-07 multi-arm backlog surfaces without blocking
    # a full sweep. A missing fingerprint on a NEW script the author is about to
    # queue is a real error; the same gap on a historical script is a backlog item.
    arm_fp_hard = bool(args.paths)

    n_ok = 0
    n_exempt = 0
    failures: List[Tuple[Path, str]] = []
    warnings: List[Tuple[Path, str]] = []
    arm_fp_warnings: List[Tuple[Path, str]] = []
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
        warn = readiness_lint(p)
        if warn:
            warnings.append((p, warn))
        arm_fp = arm_fingerprint_lint(p)
        if arm_fp:
            if arm_fp_hard:
                failures.append((p, arm_fp))
            else:
                arm_fp_warnings.append((p, arm_fp))

    print("", flush=True)
    print(f"[validate_experiments] checked {len(paths)} scripts: "
          f"{n_ok} OK, {n_exempt} exempt, {len(failures)} non-conforming, "
          f"{len(warnings)} readiness-warning(s), "
          f"{len(arm_fp_warnings)} arm-fingerprint-backlog", flush=True)
    if arm_fp_warnings:
        # Advisory in full-glob mode only (hard failures route to `failures` when
        # --paths is explicit). This is the pre-2026-06-07 multi-arm backlog --
        # historical scripts that predate the fingerprint requirement.
        print("", flush=True)
        print("[validate_experiments] Arm-fingerprint BACKLOG (advisory; hard under --paths):", flush=True)
        for p, warn in arm_fp_warnings:
            rel = p.relative_to(REPO_ROOT) if REPO_ROOT in p.parents or p == REPO_ROOT else p
            print(f"  - {rel}: {warn}", flush=True)
    if warnings:
        # Readiness-gate WARNINGS are advisory (warn-then-error rollout); they NEVER
        # affect the exit code, including under --strict. See readiness_lint().
        print("", flush=True)
        print("[validate_experiments] Readiness-gate WARNINGS (advisory, non-blocking):", flush=True)
        for p, warn in warnings:
            rel = p.relative_to(REPO_ROOT) if REPO_ROOT in p.parents or p == REPO_ROOT else p
            print(f"  - {rel}: {warn}", flush=True)
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
