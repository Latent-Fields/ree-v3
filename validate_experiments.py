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
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parent
EXPERIMENTS_DIR = REPO_ROOT / "experiments"
EMIT_NAME = "emit_outcome"
PROTOCOL_MODULE = "experiment_protocol"

# Selectable checks for --checks. Default (None) runs all of them. A caller that
# wants only one gate -- e.g. the commit-time manifest-writer gate in
# scripts/precommit_contracts.sh -- passes `--checks manifest_writer`, which keeps
# that gate surgical: it does NOT expand the emit_outcome conformance / degeneracy /
# arm-fingerprint contracts onto the broader (non-v3_exq_) script set the gate scopes.
CHECK_NAMES = ("conformance", "readiness", "arm_fingerprint", "degeneracy", "manifest_writer",
               "anchor_reachability", "precondition_recomputability",
               "e3_diagnostics_staleness", "e3_hold_weighted_readout")

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


# Degeneracy self-report enforcement (failure_autopsy_batch9_2026-06-12 Structural
# Pattern 1; the non_degenerate net + _metrics.check_degeneracy() landed 2026-06-11).
# A script that ADJUDICATES a claim-pressing discriminative criterion -- it writes an
# `evidence_direction`, carries a non-empty `claim_ids`/`CLAIM_IDS`, or uses the
# `load_bearing` criterion convention -- but never SELF-REPORTS non-degeneracy is the
# "vacuous read on an unwritten/untrained channel" failure mode (V3-EXQ-670/671/673/
# 514m/642/666a): the PASS/FAIL it emits is a property of the test design, not the
# claim. The obligation is discharged by ANY token below: a producer-side
# _metrics.check_degeneracy() / metric_is_degenerate() call, a written
# non_degenerate / degeneracy_reason manifest field, the diagnostic
# criteria_non_degenerate adjudication, or a P0 readiness / substrate_not_ready_requeue
# non-vacuity self-route (which makes a below-floor run non_contributory rather than a
# misleading verdict). This is the gate that would have caught 670/671/673 at queue time.
_DEGEN_SELFREPORT_TOKENS = (
    "check_degeneracy", "metric_is_degenerate", "metric_groups_are_degenerate",
    "non_degenerate", "non_degenerate_per_claim", "degeneracy_reason",
    "criteria_non_degenerate", "p0_readiness_gate", "P0NotReady",
    "substrate_not_ready_requeue",
)
_DEGEN_SELFREPORT_EXEMPT_MARKER = "DEGENERACY_SELFREPORT_EXEMPT"   # opt-out constant/marker

# Manifest-writer chokepoint lint (Experimental Recording Standard sec 4): a NEW
# experiment must route its flat-manifest write through the single sanctioned writer
# experiments/pack_writer.write_flat_manifest(...) (which stamps the always-record
# core and enforces the run_id/_v3 + status identity invariants) rather than a raw
# hand-rolled json.dump(manifest, f). Discharged by any of these names appearing in
# the script; opt-out via MANIFEST_WRITER_EXEMPT.
_MANIFEST_WRITER_EXEMPT_MARKER = "MANIFEST_WRITER_EXEMPT"
_CHOKEPOINT_WRITER_NAMES = ("write_flat_manifest", "write_pack", "ExperimentPackWriter")
_RAW_JSON_DUMP_NAMES = ("dump", "dumps")
_MANIFEST_IDENTITY_TOKENS = ("run_id", "evidence_direction")


# Readiness-anchor reachability enforcement (Learning 1, failure_autopsy_SD-068-rem-
# fanout-cluster_2026-07-18 sec 2; the guard landed 2026-07-18 as
# experiments/_lib/readiness_anchor.assert_anchor_reachable).
#
# An ANCHOR-KIND readiness precondition asserts that a NAMED KNOWN-POSITIVE / known-
# degenerate CONTROL reproduces a signature above a numeric gate. It is scored by a
# hand-written predicate. If that predicate is NARROWER than the state it anchors to,
# a bit-perfect replication of the control cannot clear the gate -- the precondition is
# unmeetable by construction, reports met=false on every run forever, and mislabels an
# instrument-specification gap as a substrate or scientific verdict. Confirmed instance:
# V3-EXQ-778d's `null_zero_anchor_reproduces_778c_railed_signature` scored only the
# SATURATION rail of a TWO-rail degeneracy, so a perfect replication topped out at
# 5/8 = 0.625 against a 0.75 gate -- and because
# `criteria_non_degenerate.C1_unpaired_null_derails = (readiness_ok and anchor_ok)`,
# that one mis-specified statistic accounted for the ENTIRE degeneracy flag on the
# load-bearing criterion.
#
# The obligation is discharged by replaying a frozen reference of the control through
# THE SHIPPED predicate at setup: assert_anchor_reachable(...). Opt-out marker for an
# anchor whose reachability is true by construction (e.g. an exact-equality/structural
# reproduction check, where the predicate IS the degeneracy definition).
_ANCHOR_GUARD_NAMES = ("assert_anchor_reachable", "score_reference")
_ANCHOR_REACHABILITY_EXEMPT_MARKER = "ANCHOR_REACHABILITY_EXEMPT"

# THE SECOND CATEGORY: already-ran-and-superseded. -----------------------------------
#
# EXEMPT says "there is no defect here -- reachability holds by construction". That is
# the ONLY thing it should ever say, and it is why EXEMPT silences the lint outright.
#
# A different and equally real case has no marker at all: a script that HAS the defect,
# has ALREADY RUN, and whose repair correctly lives in a successor EXQ letter rather
# than an in-place edit. Editing such a script to add a guard would force a threshold or
# predicate change that RETROACTIVELY ALTERS WHAT ITS RECORDED EVIDENCE MEANS -- the
# manifest on disk was produced by the shipped predicate, and a repaired predicate no
# longer describes it. So the correct repair is a new letter, and the old script must
# keep its defect exactly as it ran. Worked examples: the `591b/c/d/e/f` ISEF-005 family
# (readiness_anchor.py rules 3+4, lineage blocked) and V3-EXQ-778d (superseded by 778h).
#
# ANCHOR_REACHABILITY_SUPERSEDED records that status MACHINE-READABLY. Critically it
# does *NOT* silence the lint, because the defect is REAL -- it is merely not actionable
# in place. The warning still fires and still counts; it is annotated with its successor
# so a reader can tell "unrepaired backlog" from "repaired in a successor" without
# parsing a free-text reason. Silencing here would repeat the 2026-07-19 mistake in a
# new costume: an already-ran defective anchor whose warning has gone quiet is
# indistinguishable from one that was actually fixed.
_ANCHOR_REACHABILITY_SUPERSEDED_MARKER = "ANCHOR_REACHABILITY_SUPERSEDED"
# Lineage constants the corpus already uses; a SUPERSEDED marker should agree with them.
_ANCHOR_LINEAGE_NAMES = ("SUPERSEDES", "SUPERSEDES_RUN_ID")

# LINT-SPECIMEN REGISTRY -------------------------------------------------------------
#
# Some corpus files are load-bearing for the lint's OWN contract tests: they are the
# live regression specimens that prove the gate still fires on the defect that motivated
# it. Exempting one silences the canary and breaks those tests.
#
# This is not hypothetical. On 2026-07-19, closing the SD-068 anchor warnings, an
# ANCHOR_REACHABILITY_EXEMPT was added to `v3_exq_sd068_rem_unpaired_null_diagnostic.py`
# on defensible already-ran-and-superseded grounds -- and broke
# `test_a11_fires_on_the_778d_defect` + `test_a14_warn_only_under_paths_and_strict`,
# because 778d IS the specimen. Nothing in the lint said so; it was caught only by
# running the full suite, and reverted.
#
# The dependency was always deliberate (the tests carry an explicit
# `if not _D778.exists(): return  # script retired` retirement hatch) -- it just was not
# discoverable from the SCRIPT's side. This registry makes it discoverable, and
# `anchor_specimen_lint` makes it LOUD at the moment an author reaches for a marker.
_LINT_SPECIMEN_FILES = {
    "v3_exq_sd068_rem_unpaired_null_diagnostic.py": (
        "the live regression specimen for the anchor-reachability gate itself "
        "(V3-EXQ-778d, the confirmed originating defect). "
        "tests/contracts/test_anchor_reachability_lint.py::test_a11_fires_on_the_778d_defect "
        "and ::test_a14_warn_only_under_paths_and_strict both assert this file STILL "
        "warns. It is superseded by V3-EXQ-778h "
        "(v3_exq_sd068_rem_unpaired_null_anchorfix_diagnostic.py), which is the "
        "specimen for the SILENT direction (::test_a12_silent_on_the_778h_fix)"
    ),
    "v3_exq_sd068_rem_unpaired_null_anchorfix_diagnostic.py": (
        "the live regression specimen for the anchor-reachability gate's SILENT "
        "direction (V3-EXQ-778h, the repaired successor). "
        "tests/contracts/test_anchor_reachability_lint.py::test_a12_silent_on_the_778h_fix "
        "asserts this file does NOT warn -- i.e. that its assert_anchor_reachable guard "
        "stays in place. Removing the guard would break that contract"
    ),
}
# The self-route labels that make an unmeetable anchor CONSEQUENTIAL. Note this is
# deliberately WIDER than SUBSTRATE_VERDICT_LABELS: the motivating defect (778d) does
# NOT route to any of those labels -- it routes to `substrate_not_ready_requeue`, which
# is precisely the self-route an anchor governs. Scoping this gate to
# SUBSTRATE_VERDICT_LABELS alone would exempt the very run that motivated it (verified
# against the corpus 2026-07-18: 106 of 112 anchor-kind scripts are requeue-route and
# NOT substrate-verdict-class).
_ANCHOR_CONSEQUENTIAL_ROUTES = ("substrate_not_ready_requeue", "P0NotReady")


def _anchor_kind_preconditions(tree: ast.Module) -> List[str]:
    """Names of ANCHOR-KIND readiness preconditions in the script's dict literals.

    Anchor-kind = a readiness-kind precondition dict (a `name` + numeric
    `measured`/`threshold` pair, and NOT a criterion -- no `load_bearing`/`passed`)
    that ALSO carries a `control` key naming the known-positive control it anchors to.
    The `control` key is what separates an ANCHOR ("this known-degenerate reference
    reproduces its signature") from a generic readiness gate ("the substrate is trained
    enough"); only the former can be unmeetable-by-construction in the 778d way, and
    only the former is what assert_anchor_reachable guards.

    Same static limits as _readiness_and_criterion_names: a precondition assembled at
    runtime (f-string / comprehension / helper) is invisible to this scan.
    """
    anchors: List[str] = []
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
        if ("load_bearing" in str_keys) or ("passed" in str_keys):
            continue  # a criterion, not a precondition
        if not ("measured" in str_keys and "threshold" in str_keys):
            continue  # not readiness-kind
        if "control" not in str_keys:
            continue  # a generic readiness gate, not an anchor
        anchors.append(name_node.value)
    return anchors


def _module_marker_strings(tree: ast.Module, marker: str) -> List[str]:
    """Values of module-level `<marker> = "..."` assignments, in source order.

    Returns [] when the marker is absent, and [""] when it is present but not a plain
    string literal (assigned from an f-string, a call, a name...). The caller can then
    distinguish "no marker" from "marker with an unreadable reason".
    """
    out: List[str] = []
    for node in tree.body:  # module level only -- a marker inside a function is not a declaration
        if not isinstance(node, ast.Assign):
            continue
        for tgt in node.targets:
            if isinstance(tgt, ast.Name) and tgt.id == marker:
                val = node.value
                out.append(val.value if isinstance(val, ast.Constant)
                           and isinstance(val.value, str) else "")
    return out


def anchor_supersession_lint(path: Path) -> Optional[Dict[str, Any]]:
    """Machine-readable already-ran-and-superseded status for an anchor-kind script.

    Returns None when the script makes no supersession declaration. Otherwise a dict:

        {"reason": <the marker's string>,          # "" if not a plain literal
         "lineage": {"SUPERSEDES": "V3-EXQ-778h", ...},   # cross-checked constants
         "lineage_ok": bool,                        # a successor id was actually found
         "note": <str or None>}                     # cross-check complaint, if any

    WHY THIS IS A SEPARATE FUNCTION FROM THE LINT, AND WHY IT DOES NOT SUPPRESS.
    `anchor_reachability_lint` answers "is there an unguarded anchor here" -- a property
    of the CODE. This answers "is the defect repairable in place" -- a property of the
    script's LINEAGE. They are orthogonal, and collapsing them is what produced the
    2026-07-19 mistake: a superseded script was treated as an exempt one, its warning
    went quiet, and the gate's own regression specimen was silenced. So the two are
    reported side by side and the warning is annotated, never withdrawn.

    THE CROSS-CHECK. A SUPERSEDED declaration asserts a successor exists. The corpus
    already encodes lineage in `SUPERSEDES` / `SUPERSEDES_RUN_ID` module constants, so
    the claim is checkable: if neither constant is present AND the marker's reason names
    no `V3-EXQ-*` / `*_v3` successor, the declaration is unfalsifiable prose and says so
    in `note`. That is advisory -- 778d itself carries no SUPERSEDES constant despite
    genuinely being superseded by 778h, so absence is a smell, not a proof.

    Static module-level scan only, same limitation class as the other lints.
    """
    try:
        src = path.read_text(encoding="utf-8")
        tree = ast.parse(src, filename=str(path))
    except (OSError, UnicodeDecodeError, SyntaxError):
        return None

    declared = _module_marker_strings(tree, _ANCHOR_REACHABILITY_SUPERSEDED_MARKER)
    if not declared:
        return None
    reason = declared[0]

    lineage: Dict[str, Any] = {}
    for const in _ANCHOR_LINEAGE_NAMES:
        vals = _module_marker_strings(tree, const)
        if vals and vals[0]:
            lineage[const] = vals[0]

    names_a_successor = bool(re.search(r"V3-EXQ-[0-9]+[a-z]*|v3_exq_[0-9]+[a-z]*", reason))
    lineage_ok = bool(lineage) or names_a_successor

    note: Optional[str] = None
    if not reason:
        note = (f"{_ANCHOR_REACHABILITY_SUPERSEDED_MARKER} is not a plain string literal; "
                "the successor EXQ + reason cannot be read statically. Assign a literal.")
    elif not lineage_ok:
        note = (f"{_ANCHOR_REACHABILITY_SUPERSEDED_MARKER} names no successor: its reason "
                "matches no V3-EXQ-* / v3_exq_* id and the script declares neither "
                + " nor ".join(_ANCHOR_LINEAGE_NAMES)
                + ". A supersession claim that does not identify its successor cannot be "
                "checked, and is exactly the free-text opacity this marker exists to "
                "replace. Add SUPERSEDES = \"V3-EXQ-<letter>\" (the corpus convention) "
                "or name the successor in the reason.")

    return {"reason": reason, "lineage": lineage, "lineage_ok": lineage_ok, "note": note}


def anchor_specimen_lint(path: Path) -> Optional[str]:
    """Loud warning when a marker is applied to a file the lint's own tests depend on.

    A lint specimen is a real corpus file whose CURRENT lint status is asserted by
    `tests/contracts/test_anchor_reachability_lint.py`. Marking one exempt (or removing
    its guard) silences the gate's canary and breaks those contracts. Returns a warning
    string when a marker is present on a registered specimen, else None.

    This fires on ANY marker, including ANCHOR_REACHABILITY_SUPERSEDED -- even though
    SUPERSEDED does not itself suppress the warning. The point is not "this WILL break
    the tests"; it is "you are about to annotate the gate's own specimen, and the next
    step in that reasoning is usually to silence it". The 2026-07-19 mistake was
    precisely that reasoning chain, and it was defensible right up to the point it
    broke two contracts.
    """
    if path.name not in _LINT_SPECIMEN_FILES:
        return None
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, UnicodeDecodeError, SyntaxError):
        return None

    present = [m for m in (_ANCHOR_REACHABILITY_EXEMPT_MARKER,
                           _ANCHOR_REACHABILITY_SUPERSEDED_MARKER)
               if _module_marker_strings(tree, m)]
    if not present:
        return None

    return ("carries " + " + ".join(present) + ", but THIS FILE IS A LINT SPECIMEN: "
            + _LINT_SPECIMEN_FILES[path.name] + ". "
            "Confirm the change against tests/contracts/test_anchor_reachability_lint.py "
            "before landing it -- an ANCHOR_REACHABILITY_EXEMPT here SILENCES the gate's "
            "own regression canary and WILL fail those contracts (confirmed 2026-07-19: "
            "an exemption added on defensible already-ran-and-superseded grounds broke "
            "a11 + a14 and was reverted). If the script is genuinely retired, delete it "
            "-- the tests carry an explicit `if not <path>.exists(): return` hatch for "
            "that -- and drop it from _LINT_SPECIMEN_FILES in validate_experiments.py. "
            "If it merely needs its already-ran status recorded, "
            "ANCHOR_REACHABILITY_SUPERSEDED does that WITHOUT silencing the warning.")


def anchor_reachability_lint(path: Path) -> Optional[str]:
    """Readiness-anchor reachability check. Return a warning string, or None.

    A `diagnostic` / `baseline` script that (a) declares an ANCHOR-KIND readiness
    precondition -- one naming a known-positive `control` it must reproduce -- and
    (b) self-routes on that precondition to a consequential label (a
    SUBSTRATE_VERDICT_LABELS verdict, a `*_nondiscriminative` / `*_unmeetable`
    suffix, or a `substrate_not_ready_requeue` / P0-readiness requeue) MUST assert at
    setup that its frozen reference clears the gate under THE SHIPPED predicate, via
    experiments/_lib/readiness_anchor.assert_anchor_reachable(...).

    Without that assertion nothing checks the predicate against the control it claims
    to score, and a predicate narrower than the degeneracy it anchors to yields a
    guaranteed false negative that is indistinguishable, in the manifest, from a real
    substrate limitation (V3-EXQ-778d; autopsy sec 2, Learning 1).

    TWO MARKERS, AND THEY ARE NOT INTERCHANGEABLE:

      ANCHOR_REACHABILITY_EXEMPT = "<reason>"     -- SILENCES this lint. Appropriate
        ONLY when there is no defect: the predicate IS the degeneracy definition (an
        exact-equality / structural reproduction check), so reachability holds by
        construction and a replay would be tautological.

      ANCHOR_REACHABILITY_SUPERSEDED = "<successor EXQ + reason>"  -- does NOT silence
        this lint. For a script that HAS the defect but has ALREADY RUN, where the
        repair correctly lives in a successor EXQ letter: adding a guard in place would
        force a threshold or predicate change that retroactively alters what the
        recorded evidence means. The warning is annotated, not withdrawn -- see
        `anchor_supersession_lint`. Worked examples: the 591b/c/d/e/f ISEF-005 family
        and V3-EXQ-778d (superseded by 778h).

    Reaching for EXEMPT on an already-ran script is the documented error (2026-07-19),
    not a shortcut: it makes an unrepaired defect indistinguishable from a fixed one.

    Static name/string/dict-literal scan only -- the same limitation class as
    readiness_lint / arm_fingerprint_lint / degeneracy_selfreport_lint. It can MISS an
    anchor whose precondition dict is assembled at runtime, and can OVER-FIRE when a
    `control` key documents provenance on a precondition that anchors nothing
    reproducible. WARN-ONLY by design and in BOTH modes -- unlike the arm-fingerprint /
    degeneracy / manifest-writer gates it never becomes a hard failure under --paths,
    because whether a given anchor's gate is reachable is NOT statically decidable
    (`measured` is computed from live run data), so this can only ever flag a missing
    GUARD, never an actually-unreachable gate. Full-glob mode therefore surfaces the
    pre-2026-07-18 backlog without blocking, and --paths is where an author writing a
    new anchor sees it. Harden only if the guard becomes universal.
    """
    try:
        src = path.read_text(encoding="utf-8")
        tree = ast.parse(src, filename=str(path))
    except (OSError, UnicodeDecodeError, SyntaxError):
        return None  # check_script already reports unreadable / syntax errors

    if _has_main_block(tree) is None:
        return None  # library-style helper, no entry point -- exempt

    names: set = set()
    strings: set = set()
    purposes: set = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.alias):
            names.add((node.asname or node.name).split(".")[-1])
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            strings.add(node.value)
        if isinstance(node, ast.keyword) and node.arg == "experiment_purpose":
            val = node.value
            if isinstance(val, ast.Constant) and isinstance(val.value, str):
                purposes.add(val.value)
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id.lower() == "experiment_purpose":
                    val = node.value
                    if isinstance(val, ast.Constant) and isinstance(val.value, str):
                        purposes.add(val.value)

    if _ANCHOR_REACHABILITY_EXEMPT_MARKER in names or _ANCHOR_REACHABILITY_EXEMPT_MARKER in strings:
        return None
    if not (purposes & {"diagnostic", "baseline"}):
        return None

    anchors = _anchor_kind_preconditions(tree)
    if not anchors:
        return None  # no anchor-kind precondition -- nothing to guard

    consequential = (
        any(s in SUBSTRATE_VERDICT_LABELS or s.endswith(SUBSTRATE_VERDICT_SUFFIXES)
            for s in strings)
        or any(r in strings or r in names for r in _ANCHOR_CONSEQUENTIAL_ROUTES)
    )
    if not consequential:
        return None  # the anchor gates no consequential self-route

    if any(n in names for n in _ANCHOR_GUARD_NAMES):
        return None  # guard present

    return ("declares anchor-kind readiness precondition(s) "
            + ", ".join(sorted(anchors))
            + " -- each asserting a known-positive `control` reproduces a signature "
            "above a numeric gate -- and self-routes on them to a substrate-verdict / "
            "substrate_not_ready_requeue label, but never asserts the gate is REACHABLE "
            "by that control. A hand-written predicate NARROWER than the state it "
            "anchors to is unmeetable by construction: it reports met=false on every "
            "run forever and mislabels an instrument-specification gap as a substrate "
            "verdict (V3-EXQ-778d scored one rail of a two-rail degeneracy -> max 5/8 "
            "= 0.625 against a 0.75 gate, and that alone flagged the load-bearing "
            "criterion degenerate). Add a setup-time "
            "`from experiments._lib.readiness_anchor import assert_anchor_reachable` + "
            "`assert_anchor_reachable(anchor_name=..., reference_cells=<frozen recorded "
            "control>, score_fn=<THE SHIPPED PREDICATE, not a copy>, threshold=...)`. "
            "Exempt with ANCHOR_REACHABILITY_EXEMPT = \"<reason>\" ONLY when the "
            "predicate IS the degeneracy definition (no defect, reachable by "
            "construction). If instead the script has ALREADY RUN and its repair belongs "
            "in a successor EXQ letter -- because an in-place guard would force a "
            "threshold change that retroactively alters what its recorded evidence means "
            "-- use ANCHOR_REACHABILITY_SUPERSEDED = \"<successor EXQ + reason>\", which "
            "RECORDS that status without silencing this warning (591b-f, 778d->778h). "
            "See experiments/_lib/readiness_anchor.py + "
            "failure_autopsy_SD-068-rem-fanout-cluster_2026-07-18.md sec 2 (Learning 1).")


# Precondition-recomputability static lint (V3-EXQ-726, fixed 2026-07-18 fd7ca8c7cb).
#
# A precondition's whole job is to let a manifest reader re-derive the self-route's
# premise. `build_experiment_indexes._compute_adjudication` does exactly that: it
# RECOMPUTES `met` from the numeric `measured`/`threshold` pair and does NOT trust the
# author's `met`. So a precondition is only doing its job when `met` is recomputable
# from the reported measured/threshold/direction triple. Two ways that breaks:
#
#   (a) NO `direction`. The indexer then silently defaults to a FLOOR recompute
#       (`measured >= threshold`). For a ceiling-shaped check ("stayed BELOW x"), whose
#       healthy reading is `measured << threshold`, that default false-flags
#       `precondition_unmet` -- the documented 2026-06-07 V3-EXQ-648a/649 directionality
#       bug.
#   (b) `met` COMPUTED FROM A DIFFERENT STATISTIC than `measured`. V3-EXQ-726 shipped
#       `measured = round(_median(contrast_occ), 3)` (a median-across-seeds of per-seed
#       medians) alongside `met = strong_f_ok = len(contrast_seeds_strongf) >= 2` (a
#       seed COUNT). Those two statistics coincide at exactly n=3 seeds and diverge in
#       dry-run and at every other seed count, so no reader could re-derive the route.
#       The fix re-expressed both as one statistic (a seed FRACTION), which is the shape
#       this check is steering toward.
_PRECONDITION_RECOMPUTABILITY_EXEMPT_MARKER = "PRECONDITION_RECOMPUTABILITY_EXEMPT"
# Central-tendency constructs -- the `measured` side of the 726 mismatch.
_CENTRAL_TENDENCY_CALLS = (
    "median", "_median", "nanmedian", "mean", "_mean", "nanmean", "average", "avg",
    "percentile", "nanpercentile", "quantile", "nanquantile", "fmean", "median_low",
    "median_high", "median_grouped",
)
# Cardinality constructs -- the `met` side of the 726 mismatch.
_CARDINALITY_CALLS = ("len", "sum", "count", "bincount", "count_nonzero")
# Worst-case constructs -- the `met` side of the mean-vs-all mismatch (branch (d)).
# `all`/`any` quantify over a collection; `min`/`max` reduce it to an extremum. Either
# way the resulting claim is about the WORST row, not about the collection's centre.
_QUANTIFIER_CALLS = ("all", "any", "min", "max", "amin", "amax", "nanmin", "nanmax")


def _dict_str_keys(node: ast.Dict) -> Dict[str, ast.expr]:
    """Map the string-literal keys of a dict literal to their value nodes."""
    out: Dict[str, ast.expr] = {}
    for k, v in zip(node.keys, node.values):
        if isinstance(k, ast.Constant) and isinstance(k.value, str):
            out[k.value] = v
    return out


def _is_numericish(node: ast.expr) -> bool:
    """Best-effort 'this value is a number, not a label/among-a-set marker'.

    A precondition reporting a string `measured` (e.g. a regime name) or a container
    is not making the numeric floor/ceiling claim the indexer recomputes, so it is
    outside this check entirely.
    """
    if isinstance(node, ast.Constant):
        return isinstance(node.value, (int, float)) and not isinstance(node.value, bool)
    return not isinstance(node, (ast.Dict, ast.List, ast.Tuple, ast.Set, ast.JoinedStr))


def _expr_atoms(node: ast.expr) -> Tuple[set, set]:
    """(variable names, called-function names) appearing anywhere in an expression."""
    names: set = set()
    calls: set = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name):
            names.add(sub.id)
        elif isinstance(sub, ast.Attribute):
            names.add(sub.attr)
        if isinstance(sub, ast.Call):
            fn = sub.func
            if isinstance(fn, ast.Name):
                calls.add(fn.id)
            elif isinstance(fn, ast.Attribute):
                calls.add(fn.attr)
    return names, calls


def _resolve_one_level(node: ast.expr, tree: ast.Module) -> ast.expr:
    """Resolve a bare `X` / `bool(X)` / `float(X)` `met` value to X's assigned RHS.

    ONE level only, and only for the `met` side. The `met` value is almost always a
    boolean flag computed earlier in the analysis function (`met: bool(strong_f_ok)`),
    so without this hop the check would see only the flag name and could compare
    nothing. Deliberately NOT applied to `measured`, and deliberately not transitive:
    chasing `latch_seeds_frac = len(...) / len(...)` back to its own `len` would make
    the post-fix 726 shape -- where measured and met both route through that same
    fraction -- look like a median-vs-count mismatch. Shallow is what keeps this
    conservative. Last assignment wins; a name assigned in several branches resolves
    to whichever textually appears last, which is a heuristic, not a dataflow analysis.
    """
    inner = node
    while (isinstance(inner, ast.Call) and isinstance(inner.func, ast.Name)
           and inner.func.id in ("bool", "float", "int") and len(inner.args) == 1):
        inner = inner.args[0]
    if not isinstance(inner, ast.Name):
        return node
    found: Optional[ast.expr] = None
    for sub in ast.walk(tree):
        if isinstance(sub, ast.Assign):
            for tgt in sub.targets:
                if isinstance(tgt, ast.Name) and tgt.id == inner.id:
                    found = sub.value
        elif isinstance(sub, ast.AnnAssign):
            if isinstance(sub.target, ast.Name) and sub.target.id == inner.id and sub.value:
                found = sub.value
    return found if found is not None else node


_LOW_OPS = (ast.Gt, ast.GtE)
_HIGH_OPS = (ast.Lt, ast.LtE)


def _is_two_sided(node: ast.expr) -> bool:
    """True when an expression contains a genuine TWO-SIDED numeric band check.

    Two recognised spellings, both requiring the SAME subject to be bounded on
    both sides -- which is what makes this conservative enough to be WARN-worthy:

      1. Chained:  LOW < x < HIGH   -- one ast.Compare with two ops that point
         the same way (both `<`/`<=` or both `>`/`>=`), so the middle operand is
         squeezed. A chain whose ops point OPPOSITE ways (`a < b > c`) does not
         bound anything and is ignored.
      2. Conjoined:  x > LOW and x < HIGH  -- an ast.BoolOp(And) with two
         Compare children whose ops oppose AND whose subject expression is
         textually identical (compared via ast.dump, so `r["S"] > LO and
         r["S"] < HI` matches but `a > LO and b < HI` does not).
    """
    for sub in ast.walk(node):
        if isinstance(sub, ast.Compare) and len(sub.ops) == 2:
            a, bb = sub.ops
            if (isinstance(a, _HIGH_OPS) and isinstance(bb, _HIGH_OPS)) or \
               (isinstance(a, _LOW_OPS) and isinstance(bb, _LOW_OPS)):
                return True
        if isinstance(sub, ast.BoolOp) and isinstance(sub.op, ast.And):
            cmps = [v for v in sub.values
                    if isinstance(v, ast.Compare) and len(v.ops) == 1]
            for i, c1 in enumerate(cmps):
                for c2 in cmps[i + 1:]:
                    o1, o2 = c1.ops[0], c2.ops[0]
                    opposed = ((isinstance(o1, _LOW_OPS) and isinstance(o2, _HIGH_OPS))
                               or (isinstance(o1, _HIGH_OPS) and isinstance(o2, _LOW_OPS)))
                    if opposed and ast.dump(c1.left) == ast.dump(c2.left):
                        return True
    return False


def _is_one_sided_ceiling(node: ast.expr) -> bool:
    """True when an expression contains a ONE-SIDED CEILING on a row-subscript.

    Branch (e)'s second admissible saturation shape, alongside `_is_two_sided`. A
    CEILING (`r[K] < HIGH` / `<=`) is exactly a saturation guard: it asserts the
    readout has not pinned to its upper bound. A FLOOR (`>` / `>=`) is NOT -- it
    asserts the readout is above some minimum, which says nothing about headroom --
    and must never match here. That asymmetry is the whole point of this predicate:
    branch (e) originally required `_is_two_sided`, whose stated rationale ("a
    one-sided floor is not a saturation guard") is true of a floor but was
    over-generalised to ceilings, so it missed V3-EXQ-777/777a.

    The subject must be an ast.Subscript (`r["E_norm_entropy_mean"]`), i.e. a
    PER-ROW readout rather than a scalar aggregate. Measured over the full 1142-script
    corpus 2026-07-19, dropping this requirement changes nothing (both variants fire
    on exactly the same 5 scripts), so it is free prospective conservatism rather
    than a restriction paid for today: an upper bound on a scalar (`sd < X`) is
    usually a tolerance, not a headroom guard.

    Deliberately does NOT require the bound to be a `*_SAT_*`/`*_CEIL*` constant nor
    the precondition name to contain "headroom"/"saturation". Those narrowings were
    held in reserve for a noisy fire rate that did not materialise -- the real
    narrowing work is done by branch (e)'s other three conjuncts (filtered partition
    of a bare-Name source, sibling partitions exist, band does not also cover the
    unfiltered source), which is why the widened branch adds only 2 hits corpus-wide.
    """
    for sub in ast.walk(node):
        if (isinstance(sub, ast.Compare) and len(sub.ops) == 1
                and isinstance(sub.ops[0], _HIGH_OPS)
                and isinstance(sub.left, ast.Subscript)):
            return True
    return False


def _filtered_subsets(tree: ast.Module) -> Dict[str, Tuple[str, str]]:
    """Map `X = [r for r in SRC if COND]` -> {X: (SRC, dump(COND))}, for branch (e).

    Only single-generator comprehensions with exactly one `if` over a bare Name source
    count. That narrowness is the point: this is used to recognise ARM/CONDITION
    PARTITIONS of a shared row collection (the `baseline_rows` / `t1_rows` / `p1_rows`
    idiom), not comprehensions in general. A multi-source or multi-condition
    comprehension is not a clean partition and is skipped rather than guessed at.

    Last assignment wins, matching _resolve_one_level -- a heuristic, not dataflow.
    """
    out: Dict[str, Tuple[str, str]] = {}
    for sub in ast.walk(tree):
        if not isinstance(sub, ast.Assign) or len(sub.targets) != 1:
            continue
        tgt = sub.targets[0]
        if not isinstance(tgt, ast.Name):
            continue
        comp = sub.value
        if not isinstance(comp, ast.ListComp) or len(comp.generators) != 1:
            continue
        gen = comp.generators[0]
        if len(gen.ifs) != 1 or not isinstance(gen.iter, ast.Name):
            continue
        out[tgt.id] = (gen.iter.id, ast.dump(gen.ifs[0]))
    return out


def _precondition_dicts(tree: ast.Module) -> List[Tuple[str, Dict[str, ast.expr]]]:
    """(name, string-keyed fields) for every precondition-shaped dict literal.

    Precondition-shaped = a `name` plus a numeric-ish `measured`/`threshold` pair, and
    NOT a criterion (no `load_bearing` / `passed`). Note this is deliberately WIDER
    than _anchor_kind_preconditions: it does NOT require a `control` key. Recomputability
    is owed by EVERY precondition the indexer reads, not only the anchor-kind ones --
    the motivating 726 defect is a recomputability failure whether or not the entry
    anchors to a known-positive control.
    """
    out: List[Tuple[str, Dict[str, ast.expr]]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        fields = _dict_str_keys(node)
        name_node = fields.get("name")
        if not (isinstance(name_node, ast.Constant) and isinstance(name_node.value, str)):
            continue
        if ("load_bearing" in fields) or ("passed" in fields):
            continue  # a criterion, not a precondition
        if "measured" not in fields or not _is_numericish(fields["measured"]):
            continue
        # A precondition declares EITHER a single `threshold` or the two-sided
        # INTERVAL pair `threshold_low`/`threshold_high` (indexer
        # _precondition_unmet, 2026-07-19). An interval entry carries no single
        # `threshold`, so requiring one here would silently drop exactly the
        # shape this lint most needs to see.
        has_single = "threshold" in fields and _is_numericish(fields["threshold"])
        has_interval = all(
            k in fields and _is_numericish(fields[k])
            for k in ("threshold_low", "threshold_high")
        )
        if not (has_single or has_interval):
            continue
        out.append((name_node.value, fields))
    return out


def precondition_recomputability_lint(path: Path) -> Optional[str]:
    """Precondition recomputability check. Return a warning string, or None.

    WARNs when a precondition-shaped dict literal declares a numeric `measured` +
    `threshold` but either:

      (a) ships NO `direction` key -- the indexer defaults to a FLOOR recompute, which
          silently inverts a ceiling-shaped check (the 2026-06-07 V3-EXQ-648a/649
          directionality bug); or
      (b) computes `met` from a demonstrably DIFFERENT expression than the one feeding
          `measured` -- specifically a central-tendency `measured` (median / mean /
          percentile) against a cardinality `met` (`len(...) >= N` seed-count), with no
          variable shared between them. That is the V3-EXQ-726 shape exactly; or
      (c) computes `met` from a TWO-SIDED band while declaring only a SINGLE bound, so
          the undeclared leg is absent from the manifest and the indexer recomputes
          from half the check (V3-EXQ-779b baseline_entropy_headroom); or
      (d) reports a CENTRAL-TENDENCY `measured` while `met` is a WORST-CASE claim over
          the SAME collection -- an `all()`/`any()` quantifier or a `min()`/`max()`
          extremum. Same class as (b) (mean vs worst-case are different statistics),
          but (b) only fires on central-tendency-vs-CARDINALITY, so this shape slips
          past it. V3-EXQ-779b `tonic_axis_live` is the worked case. Note the shared-
          variable test below is INVERTED for (d): sharing the collection is what
          proves both sides read the same rows, so it is required, not exempting.
      (e) checks a SATURATION GUARD -- a two-sided band OR a one-sided CEILING on a
          row readout -- against only ONE partition of a row collection while SIBLING
          partitions of that same collection exist unchecked, so the readout is
          guaranteed to have room to move on the arm that was measured and is entirely
          unguarded on the arms that carry the manipulation (V3-EXQ-779b and V3-EXQ-777
          baseline_entropy_headroom; autopsy 2026-07-19 section 7). A one-sided FLOOR is
          NOT a saturation guard and never fires -- the ceiling/floor asymmetry is the
          load-bearing distinction, see _is_one_sided_ceiling.

    The shared-variable test is what keeps (b) conservative and is why the post-fix 726
    goes silent: there `measured = round(latch_seeds_frac, 4)` and `met` resolves to
    `latch_seeds_frac >= ANCHOR_MIN_LATCH_SEEDS_FRAC`, so the two sides visibly route
    through ONE statistic even though a `len()` appears further upstream in that
    fraction's own definition.

    Opt-out: PRECONDITION_RECOMPUTABILITY_EXEMPT = "<reason>" -- appropriate when `met`
    genuinely cannot be a function of the reported triple (e.g. a structural/categorical
    admissibility check whose numeric `measured` is reported for context only).

    Static name/string/dict-literal scan only -- the same limitation class as
    readiness_lint / anchor_reachability_lint. It MISSES a precondition dict assembled
    at runtime (f-string / comprehension / helper-returned), and can OVER-FIRE when
    `met` is legitimately computed through a helper whose body this shallow one-level
    resolution cannot see. WARN-ONLY by design and in BOTH modes -- like the anchor
    lint and unlike the arm-fingerprint / degeneracy / manifest-writer gates, it never
    hardens under --paths, because `measured` is computed from live run data: this can
    only ever flag a SUSPECTED mismatch between two expressions, never prove that the
    reported triple fails to recompute. It must therefore not fail a commit. Full-glob
    mode surfaces the backlog without blocking; --paths is where an author writing a new
    precondition sees it.
    """
    try:
        src = path.read_text(encoding="utf-8")
        tree = ast.parse(src, filename=str(path))
    except (OSError, UnicodeDecodeError, SyntaxError):
        return None  # check_script already reports unreadable / syntax errors

    if _has_main_block(tree) is None:
        return None  # library-style helper, no entry point -- exempt

    if _PRECONDITION_RECOMPUTABILITY_EXEMPT_MARKER in src:
        return None

    preconds = _precondition_dicts(tree)
    if not preconds:
        return None

    no_direction: List[str] = []
    mismatched: List[str] = []
    undeclared_band: List[str] = []
    central_vs_worst: List[str] = []
    partition_scoped: List[str] = []
    subsets = _filtered_subsets(tree)
    for name, fields in preconds:
        # (e) TWO-SIDED SATURATION BAND scoped to ONE partition while SIBLING
        # partitions of the same collection go unchecked. A headroom band exists to
        # certify that the readout can still MOVE; the manipulation is the thing that
        # pushes it toward a bound, so checking only the baseline partition inspects
        # the arm LEAST likely to saturate and leaves the effect-carrying arms
        # unguarded. V3-EXQ-779b is the worked case: baseline_entropy_headroom ranged
        # over `baseline_rows = [r for r in rows if r["arm"] == "T0P0"]` while
        # `t1_rows` / `p1_rows` -- sibling partitions of the same `rows` -- were never
        # band-checked. Seed 23 passed at baseline 0.6093 with its tonic-ON arms at
        # 0.8489 / 0.8587 against E_SAT_HIGH = 0.98.
        #
        # V3-EXQ-777 is the ONE-SIDED CEILING case, and the reason the original
        # two-sided-only form was too narrow: `r["E_norm_entropy_mean"] < E_SAT_CEIL`
        # over `baseline_rows = [r for r in rows if r["arm"] == "A0B0"]`, with
        # `a1_rows` / `b1_rows` unchecked -- structurally identical to 779b but with a
        # bare ceiling instead of a band. (`_is_two_sided` correctly declines it: the
        # conjunction's two Compares have DIFFERENT subjects -- E_norm_entropy_mean vs
        # D_action_mass_std -- so it is not a band on one subject. The gap was branch
        # (e)'s two-sided REQUIREMENT, not that predicate.)
        #
        # Four conjuncts keep it narrow:
        #   1. the resolved `met` is a genuine two-sided band OR a one-sided CEILING
        #      on a row-subscript (a one-sided FLOOR is not a saturation guard and
        #      must never fire -- see _is_one_sided_ceiling),
        #   2. it ranges over a name that is a single-condition filtered subset of a
        #      bare-Name source collection,
        #   3. that same source has at least one OTHER subset with a DIFFERENT
        #      condition -- i.e. sibling partitions demonstrably exist, and
        #   4. `met` does not also reference the unfiltered source directly, which
        #      would mean the band already covers every row.
        #
        # Fire rate measured over all 1142 scripts in experiments/ before widening
        # (2026-07-19): 5 hits, all named `baseline_entropy_headroom` -- the 3
        # pre-existing two-sided (779/779a/779b) plus exactly 2 new ceilings
        # (777/777a). No false positives to narrow away, so the reserve narrowings
        # (name must contain "headroom"/"saturation"; bound must be a `*_SAT_*`
        # constant) were NOT applied. Re-measure if this branch is widened again: a
        # check that fires on judgement calls gets routed around, which is worse than
        # no check at all.
        met_node_e = fields.get("met")
        if met_node_e is not None:
            resolved_e = _resolve_one_level(met_node_e, tree)
            if _is_two_sided(resolved_e) or _is_one_sided_ceiling(resolved_e):
                e_names, _ = _expr_atoms(resolved_e)
                for sub_name in sorted(e_names & set(subsets)):
                    src, cond = subsets[sub_name]
                    if src in e_names:
                        continue  # band also covers the unfiltered collection
                    siblings = [
                        other for other, (osrc, ocond) in subsets.items()
                        if other != sub_name and osrc == src and ocond != cond
                    ]
                    if siblings:
                        partition_scoped.append(name)
                        break
        # (c) TWO-SIDED backing check declared with a SINGLE bound. The
        # direction/comparator vocabulary describes one bound, so an interval
        # check (`LOW < x < HIGH`) can only declare ONE of its two legs and the
        # other vanishes from the manifest -- the indexer then recomputes `met`
        # from half the check and silently passes a violation of the undeclared
        # leg. V3-EXQ-779b baseline_entropy_headroom is the worked case: strict
        # band 0.02 < S < 0.98 declared as direction:"upper" + threshold 0.98,
        # so a saturated-to-zero baseline (S -> 0, exactly what the check exists
        # to catch) recomputed as MET. Fix: emit threshold_low + threshold_high
        # (+ comparator_low/comparator_high for strictness).
        has_interval = "threshold_low" in fields and "threshold_high" in fields
        met_node_c = fields.get("met")
        if not has_interval and met_node_c is not None:
            if _is_two_sided(_resolve_one_level(met_node_c, tree)):
                undeclared_band.append(name)
        # `comparator` satisfies the requirement too, and at HIGHER priority than
        # `direction` in _precondition_direction (comparator ">="/">" -> lower,
        # "<="/"<" -> upper; direction is only consulted when comparator is absent
        # or unrecognised). Verified against
        # REE_assembly/evidence/experiments/scripts/build_experiment_indexes.py
        # 2026-07-18. Keying this branch on `direction` alone would false-fire on a
        # precondition authored the comparator way -- fully recomputable, no defect.
        if not ({"direction", "comparator"} & set(fields)):
            no_direction.append(name)
        met_node = fields.get("met")
        if met_node is None:
            continue
        m_names, m_calls = _expr_atoms(fields["measured"])
        t_names, t_calls = _expr_atoms(_resolve_one_level(met_node, tree))
        # (d) CENTRAL-TENDENCY `measured` against a WORST-CASE `met` over the SAME
        # collection. Same class of defect as (b) -- two different statistics -- but
        # the (b) shared-variable exemption below is exactly backwards for it: here
        # sharing the collection is what PROVES both sides read the same rows, so the
        # shared name is the anchor rather than the let-off. Must therefore be tested
        # BEFORE that `continue`. Four conjuncts keep it narrow:
        #   1. `measured` is a central-tendency reduction (mean/median/percentile),
        #   2. the resolved `met` quantifies (`all`/`any`) or takes an extremum
        #      (`min`/`max`) -- i.e. it is a claim about the WORST row,
        #   3. `measured` does NOT itself quantify/reduce to an extremum -- a
        #      `measured = min(...)` worst-cell report recomputes exactly and is the
        #      shape this steers toward, so it must never fire, and
        #   4. the two sides share a variable (the collection being reduced).
        # V3-EXQ-779b is the worked case: tonic_axis_live reports
        # `statistics.fmean([r["noise_floor_temp_lift_mean"] for r in t1_rows])` while
        # `met` is `all(r["noise_floor_temp_lift_mean"] >= FLOOR for r in t1_rows)`.
        # One out-of-band row hidden by an in-band mean recomputes MET while the
        # script's own `met` is False. Its SAMPLE-kind siblings in the same file get
        # this right via a `_worst_cell(...)` helper, so `measured` IS the worst case.
        if ((m_calls & set(_CENTRAL_TENDENCY_CALLS))
                and (t_calls & set(_QUANTIFIER_CALLS))
                and not (m_calls & set(_QUANTIFIER_CALLS))
                and (m_names & t_names)):
            central_vs_worst.append(name)
        if m_names & t_names:
            continue  # measured and met visibly route through a shared statistic
        if (m_calls & set(_CENTRAL_TENDENCY_CALLS)) and (t_calls & set(_CARDINALITY_CALLS)):
            mismatched.append(name)

    if not (no_direction or mismatched or undeclared_band or central_vs_worst
            or partition_scoped):
        return None

    parts: List[str] = []
    if partition_scoped:
        parts.append(
            "precondition(s) " + ", ".join(sorted(set(partition_scoped)))
            + " check a SATURATION GUARD (a two-sided band, or a one-sided CEILING on a "
              "row readout) against only ONE partition of the row collection while "
              "SIBLING partitions of that same collection exist unchecked. A headroom "
              "guard certifies that the readout can still MOVE -- "
              "but the MANIPULATION is what pushes it toward a bound, so scoping the guard "
              "to the baseline partition inspects the arm LEAST likely to saturate and "
              "leaves the effect-carrying arms entirely unguarded. V3-EXQ-779b "
              "baseline_entropy_headroom is the worked case: it ranged over "
              "`baseline_rows` (arm == T0P0) while `t1_rows` / `p1_rows` were never "
              "band-checked, so seed 23 reported met=True at baseline 0.6093 with its "
              "tonic-ON arms at 0.8489 / 0.8587 against E_SAT_HIGH = 0.98 -- an "
              "unguarded near-ceiling exposure that surfaced only in autopsy. V3-EXQ-777 "
              "is the same defect in one-sided form: `r[\"E_norm_entropy_mean\"] < "
              "E_SAT_CEIL` over the A0B0 partition with `a1_rows` / `b1_rows` unchecked. "
              "A one-sided FLOOR is NOT a saturation guard and does not fire. FIX: do NOT "
              "widen the precondition to all arms -- a saturating TREATMENT arm is not a "
              "substrate-readiness failure and self-routing it as one mislabels the cause "
              "(the substrate was ready; the manipulation exceeded the readout's dynamic "
              "range). Emit per-arm headroom as a NON-GATING diagnostic instead: "
              "`from experiments._lib.entropy_headroom import per_arm_headroom`, then "
              "`manifest[\"diagnostics\"][\"entropy_headroom_per_arm\"] = "
              "per_arm_headroom(rows, value_key=..., low=..., high=...)`. Emit it on PASS "
              "runs too -- a diagnostic that appears only when something already looks "
              "wrong cannot establish that anything was ever right"
        )
    if central_vs_worst:
        parts.append(
            "precondition(s) " + ", ".join(sorted(central_vs_worst))
            + " report a CENTRAL-TENDENCY `measured` (mean/median/percentile) while `met` "
              "is a WORST-CASE claim over the SAME collection (an all()/any() quantifier or "
              "a min()/max() extremum) -- two DIFFERENT statistics, so a single out-of-band "
              "row whose deviation is masked by an in-band mean recomputes as MET while the "
              "script's own `met` is False. V3-EXQ-779b tonic_axis_live is the worked case: "
              "measured = fmean over the TONIC-ON cells, met = all(cell >= FLOOR). Report "
              "the WORST CELL as `measured` instead (779b's SAMPLE-kind preconditions in the "
              "same file already do exactly this via a `_worst_cell(rows, key)` helper "
              "returning the extremum plus its offending cell id, which recomputes exactly "
              "and additionally names the culprit); or, if the collection's centre really is "
              "the quantity of interest, make `met` the same central-tendency comparison"
        )
    if undeclared_band:
        parts.append(
            "precondition(s) " + ", ".join(sorted(undeclared_band))
            + " compute `met` from a TWO-SIDED band (`LOW < x < HIGH`) but declare only a "
              "SINGLE bound -- the other leg is absent from the manifest entirely, so "
              "build_experiment_indexes recomputes `met` from HALF the check and silently "
              "passes a violation of the undeclared leg. V3-EXQ-779b "
              "baseline_entropy_headroom is the worked case: a strict 0.02 < S < 0.98 band "
              "shipped as direction:\"upper\" + threshold 0.98, so a saturated-to-zero "
              "baseline (S -> 0 -- precisely the degeneracy the check exists to catch) "
              "recomputed as MET. Emit the interval instead: \"threshold_low\": LOW, "
              "\"threshold_high\": HIGH (and \"comparator_low\": \">\" / "
              "\"comparator_high\": \"<\" for strict legs; both default to inclusive). Drop "
              "the single \"threshold\" -- the indexer's _precondition_unmet prefers the "
              "interval and the legacy key is then dead weight that can drift"
        )
    if mismatched:
        parts.append(
            "precondition(s) " + ", ".join(sorted(mismatched))
            + " report a CENTRAL-TENDENCY `measured` (median/mean/percentile) while `met` "
              "is computed from a CARDINALITY expression (a len()/sum() COUNT) sharing no "
              "variable with it -- two DIFFERENT statistics, so `met` cannot be re-derived "
              "from the reported measured/threshold/direction triple. That is the "
              "V3-EXQ-726 defect: a median-across-seeds `measured` against a `>= 2 seeds` "
              "`met` coincide at exactly n=3 seeds and diverge in dry-run and at every "
              "other seed count. Re-express BOTH sides as one statistic (726 fixed it by "
              "making both a seed FRACTION, numerically identical to the pre-registered "
              "count gate at n=3, so the gate was unchanged)"
        )
    if no_direction:
        parts.append(
            "precondition(s) " + ", ".join(sorted(no_direction))
            + " declare numeric `measured` + `threshold` but NO `direction` key -- "
              "build_experiment_indexes._compute_adjudication then defaults to a FLOOR "
              "recompute (measured >= threshold), which false-flags any ceiling-shaped "
              "check (`stayed BELOW threshold`, healthy at measured << threshold) as "
              "`precondition_unmet` (the 2026-06-07 V3-EXQ-648a/649 directionality bug). "
              "Add \"direction\": \"lower\" (floor: met when measured >= threshold) or "
              "\"upper\" (ceiling: met when measured <= threshold) -- or equivalently a "
              "\"comparator\" of \">=\"/\">\" resp. \"<=\"/\"<\", which the indexer honours "
              "at higher priority"
        )
    return ("; ".join(parts)
            + ". The indexer RECOMPUTES `met` and does not trust the author's value, so a "
              "non-recomputable precondition cannot carry the self-route's premise -- which "
              "is the entire point of the block. Exempt with "
              "PRECONDITION_RECOMPUTABILITY_EXEMPT = \"<reason>\" when `met` genuinely "
              "cannot be a function of the reported triple. See V3-EXQ-726 "
              "(ree-v3 fd7ca8c7cb) for a worked before/after.")


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


def degeneracy_selfreport_lint(path: Path) -> Optional[str]:
    """Degeneracy self-report check. Return an issue string, or None.

    A script ADJUDICATES a claim-pressing discriminative criterion iff (with a
    `__main__` entry point) it does at least one of: writes an `evidence_direction`
    (it weighs governance), carries a non-empty `claim_ids` / `CLAIM_IDS` list (it
    presses a claim), or uses the `load_bearing` criterion convention. Such a script
    MUST self-report non-degeneracy at measurement time so the "vacuous read on an
    unwritten/untrained channel" family (V3-EXQ-670/671/673/514m/642/666a) is caught
    by the indexer's scoring-exclusion net rather than by a manual failure-autopsy.
    The obligation is discharged by ANY of _DEGEN_SELFREPORT_TOKENS: a producer-side
    _metrics.check_degeneracy() / metric_is_degenerate() call, a written
    non_degenerate / degeneracy_reason manifest field, the diagnostic
    criteria_non_degenerate adjudication, or a P0 readiness / substrate_not_ready_requeue
    self-route (the non-vacuity discipline that makes a below-floor run
    non_contributory instead of a misleading verdict).

    A pure substrate-readiness smoke (`claim_ids=[]`, no evidence_direction, no
    load_bearing) presses no claim and is not gated -- correctly exempt.

    Opt-out: DEGENERACY_SELFREPORT_EXEMPT = "<reason>" for a script whose
    discriminative criterion is provably non-degenerate by construction (e.g. it
    routes on an exact-equality / structural check, not a learned-channel magnitude).

    Static name/string-scan only -- same limitation class as readiness_lint /
    arm_fingerprint_lint: it keys on plain identifier/string/list-literal presence,
    so it can over-fire if a token appears only in a comment/docstring and can miss a
    claim_ids list or marker assembled at runtime. Whether this blocks is decided in
    main(): HARD when the script is named via --paths (the /queue-experiment authoring
    path -- a new claim-pressing script without self-report is a real error),
    advisory in full-glob (grandfathers the pre-2026-06-12 backlog).
    """
    try:
        src = path.read_text(encoding="utf-8")
        tree = ast.parse(src, filename=str(path))
    except (OSError, UnicodeDecodeError, SyntaxError):
        return None  # check_script already reports unreadable / syntax errors

    if _has_main_block(tree) is None:
        return None  # library-style helper, no entry point -- exempt

    names: set = set()
    strings: set = set()
    has_nonempty_claim_ids = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.alias):
            names.add((node.asname or node.name).split(".")[-1])
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            strings.add(node.value)
        elif isinstance(node, ast.Assign):
            # CLAIM_IDS = [...] / claim_ids = [...]  with a non-empty list literal.
            for tgt in node.targets:
                if (isinstance(tgt, ast.Name)
                        and tgt.id.lower() in ("claim_ids", "claim_ids_tested")
                        and isinstance(node.value, ast.List) and node.value.elts):
                    has_nonempty_claim_ids = True
        elif isinstance(node, ast.Dict):
            # {"claim_ids": [...], ...}  with a non-empty list literal.
            for k, v in zip(node.keys, node.values):
                if (isinstance(k, ast.Constant)
                        and k.value in ("claim_ids", "claim_ids_tested")
                        and isinstance(v, ast.List) and v.elts):
                    has_nonempty_claim_ids = True

    if _DEGEN_SELFREPORT_EXEMPT_MARKER in names or _DEGEN_SELFREPORT_EXEMPT_MARKER in strings:
        return None

    adjudicates = (
        has_nonempty_claim_ids
        or ("evidence_direction" in strings)
        or ("load_bearing" in strings)
    )
    if not adjudicates:
        return None  # presses no claim / no discriminative direction -- nothing to gate

    self_reports = (any(t in names for t in _DEGEN_SELFREPORT_TOKENS)
                    or any(t in strings for t in _DEGEN_SELFREPORT_TOKENS))
    if self_reports:
        return None

    return ("adjudicates a claim-pressing discriminative criterion "
            "(evidence_direction / non-empty claim_ids / load_bearing) but never "
            "self-reports non-degeneracy -- add a measurement-time "
            "_metrics.check_degeneracy(...) (writes non_degenerate / degeneracy_reason "
            "at the manifest root) or a P0 readiness / substrate_not_ready_requeue "
            "non-vacuity self-route, so the indexer can scoring-exclude a vacuous read "
            "instead of leaving it to a manual failure-autopsy (V3-EXQ-670/671/673 "
            "family). Exempt with DEGENERACY_SELFREPORT_EXEMPT = \"<reason>\". "
            "See experiments/_metrics.check_degeneracy + /queue-experiment + "
            "failure_autopsy_batch9_2026-06-12.")


def manifest_writer_lint(path: Path) -> Optional[str]:
    """Manifest-writer chokepoint check. Return an issue string, or None.

    A script WRITES A RESULT MANIFEST iff (with a `__main__` entry point) it carries
    the manifest-identity tokens `run_id` AND `evidence_direction` as strings AND
    performs a raw `json.dump`/`json.dumps`. Such a script MUST route that write
    through the single sanctioned writer `experiments/pack_writer.write_flat_manifest`
    (or the pack path `write_pack` / `ExperimentPackWriter`), which stamps the
    Experimental Recording Standard always-record core (via stamp_recording_core) and
    enforces the run_id/_v3 + status identity invariants at emission. A hand-rolled
    `json.dump(manifest, f)` bypasses the always-core -- the exact recording-debt the
    standard closes (0% of flat manifests carried a substrate_hash pre-standard).

    Discharged when any of _CHOKEPOINT_WRITER_NAMES appears in the script (it routes
    through the sanctioned writer, whatever else it dumps). A pure telemetry/helper
    with no manifest identity, or a script with no raw dump, is not gated.

    Opt-out: MANIFEST_WRITER_EXEMPT = "<reason>" (e.g. a crash-report smoke, or a
    writer whose shape is deliberately outside the standard).

    Static name/string-scan only -- same limitation class as arm_fingerprint_lint /
    degeneracy_selfreport_lint: it keys on plain identifier/string presence, so it can
    over-fire (a manifest built + dumped via a helper the scan cannot follow) or miss
    (identity tokens assembled at runtime). Whether this blocks is decided in main():
    HARD when the script is named via --paths (the /queue-experiment authoring path --
    a NEW script hand-rolling a manifest write is a real error), advisory in full-glob
    (grandfathers the ~1028-script pre-2026-07-12 backlog).
    """
    try:
        src = path.read_text(encoding="utf-8")
        tree = ast.parse(src, filename=str(path))
    except (OSError, UnicodeDecodeError, SyntaxError):
        return None  # check_script already reports unreadable / syntax errors

    if _has_main_block(tree) is None:
        return None  # library-style helper, no entry point -- exempt

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

    if _MANIFEST_WRITER_EXEMPT_MARKER in names or _MANIFEST_WRITER_EXEMPT_MARKER in strings:
        return None
    # Routes through the sanctioned writer -> discharged, regardless of any other dump.
    if any(n in names for n in _CHOKEPOINT_WRITER_NAMES):
        return None

    writes_manifest = (
        all(t in strings for t in _MANIFEST_IDENTITY_TOKENS)
        and any(n in names for n in _RAW_JSON_DUMP_NAMES)
    )
    if not writes_manifest:
        return None  # no result-manifest write to route

    return ("writes a flat experiment manifest with a raw json.dump/json.dumps "
            "instead of routing through the sanctioned single writer "
            "experiments/pack_writer.write_flat_manifest(...) -- which stamps the "
            "Experimental Recording Standard always-core (recording_schema / "
            "substrate_hash / machine / machine_class / elapsed_seconds / config / "
            "seeds via stamp_recording_core) and enforces the run_id/_v3 + status "
            "identity invariants. Replace the raw `json.dump(manifest, f)` tail with "
            "`from experiments.pack_writer import write_flat_manifest` + "
            "`write_flat_manifest(manifest, out_dir, dry_run=..., config=..., "
            "seeds=..., script_path=Path(__file__))`. Exempt with "
            "MANIFEST_WRITER_EXEMPT = \"<reason>\". See "
            "experimental_recording_standard_2026-07-12.md sec 4 + "
            "pack_writer_single_writer_migration_plan.md.")


# All SIX are assigned ONLY inside `E3Selector.select()` -- verified by AST scan of
# `ree_core/predictors/e3_selector.py` (2026-07-19): last_raw_scores:2103,
# last_score_diagnostics:2452, last_scores:2657, last_score_decomp:2659,
# last_channel_terms:2680, last_precommit_probs:2687. There is no `__init__` default and
# no reset path, so every one of them latches identically and none is a weaker signal
# than the others. `last_raw_scores` was MISSING from this tuple until 2026-07-19 -- a
# coverage hole, not a deliberate narrowing: V3-EXQ-722 carried TWO latched reads and
# `last_raw_scores` was the second one, so the attribute the lint was blind to is one the
# defect demonstrably uses. Adding it moved the corpus count by ZERO (measured), so it
# buys future coverage at no backlog cost.
_E3_LATCHED_ATTRS = ("last_score_diagnostics", "last_score_decomp", "last_channel_terms",
                     "last_scores", "last_precommit_probs", "last_raw_scores")
_E3_STALENESS_EXEMPT_MARKER = "E3_DIAGNOSTICS_STALENESS_EXEMPT"


def _e3_latched_reads(tree: ast.Module) -> List[ast.expr]:
    """Every read of an E3 `last_*` diagnostic: `x.e3.last_scores` or getattr(x.e3, "last_scores")."""
    reads: List[ast.expr] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in _E3_LATCHED_ATTRS:
            reads.append(node)
        elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "getattr" and len(node.args) >= 2
                and isinstance(node.args[1], ast.Constant)
                and node.args[1].value in _E3_LATCHED_ATTRS):
            reads.append(node)
    return reads


def _clears_an_e3_latch(tree: ast.Module) -> bool:
    """`agent.e3.last_score_diagnostics = None` -- the clear-before-select idiom."""
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant)
                and node.value.value is None
                and any(isinstance(t, ast.Attribute) and t.attr in _E3_LATCHED_ATTRS
                        for t in node.targets)):
            return True
    return False


def _guards_e3_latch_by_identity(tree: ast.Module) -> bool:
    """`pid = id(probs); fresh = pid != prev_probs_id` -- the identity-freshness idiom.

    An alternative, equally sound discharge of the same obligation. A latched read hands
    back the SAME object on every skipped tick, while a genuine `select()` allocates a new
    tensor -- so gating the record on `id(...)` changing admits exactly the fresh
    selections, which is what clear-before-select achieves by the other route.

    It is sound in the direction that matters. The failure mode of identity comparison is
    an address collision after garbage collection, which would read a FRESH value as stale
    and DROP a row -- an under-count. It cannot manufacture the inflation this lint exists
    to catch, so a false negative here costs power, never a phantom sample size.

    Recognised: `id(<latched read>)`, or `id(v)` where `v` was assigned from a latched
    read, whose result participates in a comparison. Like exemptions (a)-(c) this is
    detected file-wide rather than per-read-site (see the lint docstring's limitation
    note) -- a driver that computes the identity check but forgets to gate the append on
    it is exempted. Acceptable at WARN level, and the shape is rare enough to be a
    deliberate act: exactly ONE script in the 2026-07-19 corpus uses it.
    """
    latched_vars = set()
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and any(a in ast.dump(node.value) for a in _E3_LATCHED_ATTRS)):
            latched_vars.add(node.targets[0].id)

    def _is_latched_id_call(n: ast.AST) -> bool:
        return (isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                and n.func.id == "id" and len(n.args) == 1
                and ((isinstance(n.args[0], ast.Name) and n.args[0].id in latched_vars)
                     or any(a in ast.dump(n.args[0]) for a in _E3_LATCHED_ATTRS)))

    # Walk INTO the assigned value: the idiom is usually guarded, e.g.
    # `pid = id(probs) if probs is not None else None` (an IfExp, not a bare Call).
    id_vars = set()
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and any(_is_latched_id_call(x) for x in ast.walk(node.value))):
            id_vars.add(node.targets[0].id)

    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare):
            continue
        for side in [node.left] + list(node.comparators):
            if _is_latched_id_call(side):
                return True
            if isinstance(side, ast.Name) and side.id in id_vars:
                return True
    return False


def e3_diagnostics_staleness_lint(path: Path) -> Optional[str]:
    """Stale-E3-diagnostics pseudo-replication check. Return an issue string, or None.

    `ree_core/predictors/e3_selector.py` populates all six of `last_score_diagnostics` /
    `last_score_decomp` / `last_channel_terms` / `last_scores` / `last_precommit_probs` /
    `last_raw_scores` ONLY inside `select()` (see `_E3_LATCHED_ATTRS` for the verified
    per-attribute assignment lines). The attributes LATCH: after a tick on which `select()` did
    not run, they still hold the PREVIOUS selection's values. A driver that reads them
    once per env step, in a loop, WITHOUT clearing them first therefore re-records one
    selection as many independent observations. Nothing raises; the run simply reports
    a sample size it does not have. Measured on the V3-EXQ-785 config: 67 genuine
    `select()` calls behind 600 recorded rows (~9.0x inflation).

    MECHANISM -- the widely-assumed cause is WRONG, and the correction is why this lint
    exists. The skip is NOT `beta_gate.is_elevated`. `ree_core/agent.py` returns the
    held/stepped action on `if not ticks["e3_tick"] and self._last_action is not None:`
    BEFORE the only `e3.select()` call site; `beta_gate.is_elevated` merely chooses
    step-vs-hold WITHIN an already-skipped tick. The real driver is the E3 CADENCE:
    `heartbeat.e3_steps_per_tick` defaults to 10. CONSEQUENCE: "commitment was
    effectively disabled for this run" does NOT exculpate a driver -- a per-env-step
    diagnostics read is ~10x pseudo-replicated regardless of commitment config. A guard
    written against the beta gate would be wrong.

    The obligation is discharged by ANY of:
      (a) a `<...>.last_* = None` clear (the reference idiom -- clear immediately before
          `select_action(...)`, then record a row ONLY if it was repopulated),
      (b) a `ticks["e3_tick"]` guard (the driver already knows about the cadence), or
      (c) a direct `e3.select(...)` call site (the driver drives selection itself, so
          every read follows a selection it just caused), or
      (d) an identity-freshness guard -- `pid = id(probs)`, record only when `pid`
          changed. Equivalent in effect to (a): a latched read returns the SAME object,
          a real selection allocates a new one. See `_guards_e3_latch_by_identity`.

    Reference implementation:
    `experiments/v3_exq_785a_mech463_arousal_exogenous_urgency_decomp.py` -- clears
    before every `agent.select_action(...)`, records only on repopulation, and counts
    the skipped ticks separately as `n_latched_ticks` telemetry (real run: 1757 genuine
    selections from 15000 ticks, yield ~0.12). Emitting that counter is the convention:
    it makes the true denominator auditable from the manifest.

    SCOPE -- fires only on a read that is INSIDE a `for`/`while` body in a script that
    also calls `select_action`, i.e. the per-env-step driver-loop shape that actually
    pseudo-replicates. A one-shot read after a known selection is correctly exempt.

    Static AST scan, so it shares the limitation class of the other name-scan lints: the
    clear/guard/select exemptions are detected file-wide rather than per-read-site, so a
    script that clears one attribute but latches another is exempted (a miss), and a
    driver that reaches selection through a helper this scan cannot see may over-fire.
    Both are acceptable at WARN level. WARN-ONLY IN BOTH MODES -- it never hardens under
    `--paths`. It flags a SUSPECTED inflated denominator, never a proven one, and the
    landed corpus carries a large pre-2026-07-19 backlog whose runs are already complete
    (a completed run's pre-registered emission is not rewritten). This gates NEW scripts.

    Opt-out: E3_DIAGNOSTICS_STALENESS_EXEMPT = "<reason>".
    """
    try:
        src = path.read_text(encoding="utf-8")
        tree = ast.parse(src, filename=str(path))
    except (OSError, UnicodeDecodeError, SyntaxError):
        return None  # check_script already reports unreadable / syntax errors

    reads = _e3_latched_reads(tree)
    if not reads:
        return None

    if _E3_STALENESS_EXEMPT_MARKER in src:
        return None

    # (a) clear-before-select, (b) cadence guard, (c) driver owns the select call,
    # (d) identity-freshness guard, (e) the shared sentinel-key helper.
    if _uses_shared_fresh_select_helper(tree):
        return None
    if _clears_an_e3_latch(tree):
        return None
    if _guards_e3_latch_by_identity(tree):
        return None
    if any(isinstance(n, ast.Constant) and n.value == "e3_tick" for n in ast.walk(tree)):
        return None
    if any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
           and n.func.attr == "select" and isinstance(n.func.value, ast.Attribute)
           and n.func.value.attr == "e3" for n in ast.walk(tree)):
        return None

    # Scope to the shape that actually pseudo-replicates: a read inside a loop, in a
    # script that drives the agent per env step.
    if "select_action" not in src:
        return None
    loop_spans = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.For, ast.While)):
            last = max((getattr(x, "lineno", node.lineno) for x in ast.walk(node)),
                       default=node.lineno)
            loop_spans.append((node.lineno, last))
    looped = sorted({r.lineno for r in reads
                     if any(lo <= r.lineno <= hi for lo, hi in loop_spans)})
    if not looped:
        return None

    attrs = sorted({(n.attr if isinstance(n, ast.Attribute) else n.args[1].value)
                    for n in reads})
    return (f"STALE E3 DIAGNOSTICS: reads {', '.join(attrs)} inside a driver loop "
            f"(line(s) {', '.join(str(n) for n in looped[:6])}) without clearing the "
            "latch first. E3 populates these ONLY inside select(), which runs on ~1 tick "
            "in heartbeat.e3_steps_per_tick (default 10) -- so a per-env-step read "
            "re-records the PREVIOUS selection as a new independent observation and the "
            "run reports a sample size it does not have (V3-EXQ-785: 600 rows behind 67 "
            "genuine selections, ~9.0x). NOTE the cause is the E3 CADENCE, not "
            "beta_gate.is_elevated -- agent.py returns early on `not ticks[\"e3_tick\"]` "
            "BEFORE select() is reached, so disabled commitment does NOT exculpate this. "
            "FIX: set `agent.e3.<attr> = None` immediately before every "
            "select_action(...), record a row ONLY if it was repopulated, and emit the "
            "skipped-tick count as `n_latched_ticks` so the true denominator is auditable. "
            "Reference: experiments/v3_exq_785a_mech463_arousal_exogenous_urgency_decomp.py. "
            "Exempt with E3_DIAGNOSTICS_STALENESS_EXEMPT = \"<reason>\".")


# ---- form 2: hold-weighted readout (V3-EXQ-699) --------------------------------------
# The SECOND form of the pseudo-replication defect, established by the V3-EXQ-699
# re-adjudication (REE_assembly `ac2fb64028`). Form 1 (above) keys on a diagnostics
# LATCH. Form 2 touches no latch at all, so form 1 is structurally blind to it, and on
# 699 the unflagged exposure was the run's PRIMARY DV while the flagged one was
# incidental. These are INDEPENDENT defects and are deliberately kept as separate gates
# with separate pins: 699's `active_frac == 1.0` is INFORMATIVE precisely because its
# diagnostics are fresh, where 708's identical 1.0 was vacuous. Conflating freshness
# with replication mis-adjudicates in both directions (autopsy 699 sec 11.2).
#
# `_last_selected_trajectory` latches exactly like the six form-1 attributes -- assigned
# only in `E3Selector.select()` (`e3_selector.py:3108`; `:3224` is a read in
# `post_action_update`). It is kept HERE rather than appended to _E3_LATCHED_ATTRS so the
# form-1 corpus pin stays a measurement of form 1.
_E3_SELECTION_LATCHED_ATTRS = ("_last_selected_trajectory",)
_E3_HOLD_WEIGHTED_EXEMPT_MARKER = "E3_HOLD_WEIGHTED_READOUT_EXEMPT"

# ---- discharge (e): the SHARED fresh-select helper ------------------------------------
# `experiments/_lib/fresh_select.py` implements the sentinel-key freshness instrument:
# it stamps a namespaced private key into agent.e3.last_score_diagnostics before every
# select_action() and detects a genuine selection by that key's ABSENCE afterwards
# (select() reassigns the dict wholesale, e3_selector.py:2452 -- pinned by
# tests/contracts/test_fresh_select_wholesale_reassign.py).
#
# WHY THIS NEEDS ITS OWN DISCHARGE. Both lints pattern-match a LITERAL
# `agent.e3.<attr> = None` clear, which the sentinel deliberately does NOT do: nulling
# `_last_selected_trajectory` changes substrate behaviour via post_action_update (the
# ARC-016 deadlock fallback, which runs on EVERY step through update_residue), so the
# clear would make the run a different experiment rather than a repaired instrument.
# Before this discharge existed, sentinel-key drivers had to declare
# E3_DIAGNOSTICS_STALENESS_EXEMPT / E3_HOLD_WEIGHTED_READOUT_EXEMPT -- a blanket opt-out
# that suppressed a GENUINE guard for the rest of the file. Recognising the shared helper
# instead keeps the gate live on everything the helper does not cover.
#
# Deliberately NARROW: it requires an actual import of the shared module AND a
# construction of its probe. A comment mentioning fresh_select, or a hand-rolled
# re-implementation of the sentinel, does NOT discharge -- the whole point of the shared
# helper is that the pattern stops being hand-copied.
_FRESH_SELECT_MODULE = "fresh_select"
_FRESH_SELECT_PROBE = "FreshSelectProbe"


def _uses_shared_fresh_select_helper(tree: ast.AST) -> bool:
    """True iff the script imports experiments/_lib/fresh_select and builds its probe."""
    imported = False
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom):
            mod = n.module or ""
            if mod == _FRESH_SELECT_MODULE or mod.endswith("." + _FRESH_SELECT_MODULE):
                if any(a.name == _FRESH_SELECT_PROBE for a in n.names):
                    imported = True
        elif isinstance(n, ast.Import):
            for a in n.names:
                if a.name.endswith("." + _FRESH_SELECT_MODULE) or a.name == _FRESH_SELECT_MODULE:
                    imported = True
    if not imported:
        return False
    # the probe must actually be constructed, not merely imported
    for n in ast.walk(tree):
        if isinstance(n, ast.Call):
            f = n.func
            if isinstance(f, ast.Name) and f.id == _FRESH_SELECT_PROBE:
                return True
            if isinstance(f, ast.Attribute) and f.attr == _FRESH_SELECT_PROBE:
                return True
    return False

# Calls that reduce a tensor/list to a scalar summary. Their presence is what separates
# "this driver STEPPED the env with the action / stored the transition for training"
# (legitimate at every env step -- the held action IS the action taken) from "this driver
# turned the action into a per-step STATISTIC" (the defect). Without this requirement the
# rule fires on every replay buffer in the corpus and is unusable.
_SCALAR_REDUCTIONS = ("argmax", "argmin", "item", "max", "min", "sum", "mean", "len",
                      "int", "float", "bool", "round", "index", "count", "tolist",
                      "nonzero", "sorted", "set", "std", "var")


def _e3_root_source(node: ast.AST) -> Optional[str]:
    """The cadence-gated ROOT this expression is, if any. Two roots, both verified:

      "select_action" -- `agent.py:5430` returns the HELD action on
                         `not ticks["e3_tick"]`, BEFORE `e3.select()` is reached.
      "candidates"    -- `agent.generate_trajectories` (`agent.py:4812`) returns CACHED
                         candidates on a non-E3 tick (MECH-057a gate).

    (`_last_selected_trajectory`, root three, is handled directly as a latch read -- see
    `_e3_selection_latch_reads` -- because like the form-1 attributes the READ alone is
    the defect, with no accumulation shape required.)
    """
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        if node.func.attr == "select_action":
            return "select_action"
        if node.func.attr == "generate_trajectories":
            return "candidates"
    return None


def _derived_taint(node: ast.AST, tainted: Dict[str, str]) -> Optional[str]:
    """Does this expression DERIVE from a cadence-gated value -- by base chain, not mention?

    The distinction is the whole precision of this lint, and getting it wrong makes the
    rule useless in a way that is not obvious from a spot check. A first cut propagated
    taint through any *mention*, which meant `obs, r, done, info = env.step(action)`
    tainted the entire driver (measured: `agent`, `cfg`, `done`, `info`, `latent` all
    marked, and the rule fired on unrelated helper lines in v3_exq_785). Two rules fix it:

      1. NO propagation through tuple unpacking. `env.step(action)` returns a genuinely
         fresh observation -- the held action really is the action taken, so the env's
         response to it is a real per-step measurement, not a replicated one.
      2. The tainted name must be the BASE of the expression (`action[0].argmax().item()`),
         or an argument to a PURE wrapper (`int(...)`, `len(...)`), or the ITERABLE of a
         comprehension (`{f(t) for t in candidates}` -- 699's `pre_e3_classes`). Passing
         it to an arbitrary function produces a new value and breaks the chain.

    Rule 2 is deliberately conservative: a chain routed through a user-defined helper
    (`_traj_first_action_class(sel_traj)`) is NOT followed, so this under-fires rather
    than over-fires. That is the same static-AST limitation class the form-1 lint
    documents, and the safe direction for a WARN that drives manual triage.
    """
    if isinstance(node, ast.Name):
        return tainted.get(node.id)
    if isinstance(node, (ast.Attribute, ast.Subscript)):
        return _derived_taint(node.value, tainted)
    if isinstance(node, ast.Starred):
        return _derived_taint(node.value, tainted)
    if isinstance(node, (ast.BinOp,)):
        return (_derived_taint(node.left, tainted)
                or _derived_taint(node.right, tainted))
    if isinstance(node, ast.UnaryOp):
        return _derived_taint(node.operand, tainted)
    if isinstance(node, ast.BoolOp):
        for v in node.values:
            t = _derived_taint(v, tainted)
            if t:
                return t
        return None
    if isinstance(node, ast.IfExp):
        return (_derived_taint(node.body, tainted)
                or _derived_taint(node.orelse, tainted))
    if isinstance(node, ast.Compare):
        for side in [node.left] + list(node.comparators):
            t = _derived_taint(side, tainted)
            if t:
                return t
        return None
    if isinstance(node, (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp)):
        for gen in node.generators:
            t = _derived_taint(gen.iter, tainted)
            if t:
                return t
        return None
    if isinstance(node, ast.Call):
        root = _e3_root_source(node)
        if root:
            return root
        if (isinstance(node.func, ast.Name) and node.func.id == "getattr"
                and len(node.args) >= 2 and isinstance(node.args[1], ast.Constant)
                and node.args[1].value in _E3_SELECTION_LATCHED_ATTRS):
            return "selected_traj"
        if isinstance(node.func, ast.Attribute):
            # method call on a tainted base: `action[0].argmax()`, `probs.item()`
            base = _derived_taint(node.func.value, tainted)
            if base:
                return base
        name = (node.func.id if isinstance(node.func, ast.Name)
                else node.func.attr if isinstance(node.func, ast.Attribute) else None)
        if name in _SCALAR_REDUCTIONS:  # pure wrapper: `int(x)`, `len(x)`, `sorted(x)`
            for a in node.args:
                t = _derived_taint(a, tainted)
                if t:
                    return t
        return None
    return None


def _contains_reduction(node: ast.AST, tainted: Dict[str, str]) -> bool:
    """Does this expression reduce a cadence-gated value to a scalar summary?"""
    for n in ast.walk(node):
        if isinstance(n, ast.Call):
            name = (n.func.attr if isinstance(n.func, ast.Attribute)
                    else n.func.id if isinstance(n.func, ast.Name) else None)
            if name in _SCALAR_REDUCTIONS and _derived_taint(n, tainted):
                return True
    return False


def _e3_cadence_gated_sources(tree: ast.Module) -> Tuple[Dict[str, str], Set[str]]:
    """Variables that only refresh on an E3 tick -> (name -> root, names holding a SCALAR).

    Fixed point over SINGLE-Name assignments only (see `_derived_taint` rule 1 for why
    tuple targets are excluded), so `action = agent.select_action(...)` then
    `cls = int(action[0].argmax().item())` marks both `action` and `cls`.

    The second return value is what makes the accumulation test work. Scalar-ness is a
    property of the VARIABLE, established where it is derived, not of the site where it is
    accumulated -- 699 reduces at `:882` and accumulates at `:899`, seventeen lines apart,
    and a rule that demanded a reduction at the accumulation site missed the run's primary
    DV entirely. `action` itself is not scalar (it is a tensor, and storing it in a replay
    buffer is correct); `committed_class` is.
    """
    tainted: Dict[str, str] = {}
    scalars: Set[str] = set()
    for _ in range(6):  # fixed point; 6 is far beyond any real chain depth
        grew = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
                    continue  # rule 1: no tuple unpacking
                target, value = node.targets[0], node.value
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                target, value = node.target, node.value
            else:
                continue
            if value is None or target.id in tainted:
                continue
            src = _derived_taint(value, tainted)
            if src:
                tainted[target.id] = src
                if (_contains_reduction(value, tainted)
                        or any(isinstance(n, ast.Name) and n.id in scalars
                               for n in ast.walk(value))):
                    scalars.add(target.id)
                grew = True
        if not grew:
            break
    return tainted, scalars


def _e3_selection_latch_reads(tree: ast.Module) -> List[ast.expr]:
    """Reads of `agent.e3._last_selected_trajectory` -- the per-selection latch, form (b).

    Assigned only inside `E3Selector.select()` (`e3_selector.py:3108`), so it latches
    exactly like the six form-1 attributes and the READ alone is the defect. 699 proved
    (a) and (b) are one defect empirically: its `selected_class_entropy_nats` equalled
    `committed_class_entropy_nats` to 6dp on all 12 arm-seeds.
    """
    reads: List[ast.expr] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in _E3_SELECTION_LATCHED_ATTRS:
            reads.append(node)
        elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "getattr" and len(node.args) >= 2
                and isinstance(node.args[1], ast.Constant)
                and node.args[1].value in _E3_SELECTION_LATCHED_ATTRS):
            reads.append(node)
    return reads


def _is_scalar_use(node: ast.AST, tainted: Dict[str, str], scalars: Set[str]) -> bool:
    """Is this expression a SCALAR SUMMARY of a cadence-gated value (vs. passing it on)?

    True when it mentions a scalar-derived variable (`committed_class`), reduces one in
    place (`int(action[0].argmax().item())`), compares one (`len(pre_e3_classes) >= 2`,
    which then gates a counter), or uses one as a dict subscript KEY (the histogram shape
    699 used -- and a dict key is necessarily scalar, so no further reduction is required).

    False for `buf.append((z, action, z1))`, which stores the action tensor itself. That
    distinction is load-bearing: a replay buffer is CORRECT at every env step -- the held
    action really is the action taken -- and a rule that fired on it would flag most of
    the corpus for a non-defect.
    """
    if any(isinstance(n, ast.Name) and n.id in scalars for n in ast.walk(node)):
        return True
    for n in ast.walk(node):
        if isinstance(n, ast.Call):
            name = (n.func.attr if isinstance(n.func, ast.Attribute)
                    else n.func.id if isinstance(n.func, ast.Name) else None)
            if name in _SCALAR_REDUCTIONS and _derived_taint(n, tainted):
                return True
        if isinstance(n, ast.Compare) and _derived_taint(n, tainted):
            return True
    return False


def _hold_weighted_accumulations(tree: ast.Module, tainted: Dict[str, str],
                                 scalars: Set[str]) -> List[Tuple[int, str]]:
    """(lineno, source) for every per-step accumulation of a cadence-gated scalar.

    Recognised shapes, all drawn from the confirmed 699 sites:
      `counts[cls] = counts.get(cls, 0) + 1`   subscript-key histogram      (:899)
      `counts[cls] += 1` / `sigs[cls][s] += 1`  augmented counter           (:920)
      `vals.append(<reduction>)` / .add / .extend / .update / .setdefault
      `total += <reduction>`                    running sum
      `if <tainted compare>: n += 1`            condition-gated counter     (:902)
    """
    hits: List[Tuple[int, str]] = []

    def _add(node: ast.AST, probe: ast.AST, key: bool = False) -> None:
        src = _derived_taint(probe, tainted)
        # A dict/Counter KEY is necessarily a scalar, so the key shape needs no further
        # reduction evidence -- this is the exact shape of 699's primary DV at :899.
        if src and (key or _is_scalar_use(probe, tainted, scalars)):
            hits.append((getattr(node, "lineno", 0), src))

    for node in ast.walk(tree):
        # append/add/extend/update/setdefault of a reduced value
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr in ("append", "add", "extend", "update", "setdefault")):
            for a in node.args:
                _add(node, a)
        # subscript-key histogram, either plain or augmented
        if isinstance(node, (ast.Assign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for t in targets:
                if isinstance(t, ast.Subscript):
                    _add(node, t.slice, key=True)
        # running sum: `total += <reduction of tainted>`
        if isinstance(node, ast.AugAssign) and node.value is not None:
            _add(node, node.value)
        # condition-gated counter: `if len(pre_e3_classes) >= 2: n_pre_ge2 += 1`
        if isinstance(node, ast.If):
            src = _derived_taint(node.test, tainted)
            if src and _is_scalar_use(node.test, tainted, scalars):
                for stmt in node.body:
                    for n in ast.walk(stmt):
                        if isinstance(n, ast.AugAssign):
                            hits.append((getattr(n, "lineno", 0), src))
    return hits


def e3_hold_weighted_readout_lint(path: Path) -> Optional[str]:
    """Hold-weighted-readout check (defect form 2). Return an issue string, or None.

    THE DEFECT. `ree_core/agent.py:5430` returns the HELD action on
    `if not ticks["e3_tick"] and self._last_action is not None:` -- BEFORE `e3.select()`
    is reached. So the value handed back by `agent.select_action(...)` is UNCHANGED across
    a whole hold. A driver that accumulates a per-step STATISTIC from that return value
    therefore weights each commitment by its HOLD DURATION. Cadence defaults to 10 steps
    (`utils/config.py:2017`) and varies 5-20 under MECH-093 arousal modulation
    (`heartbeat/clock.py:52-70`), so the weighting is neither constant nor known.

    WHY THIS IS A SEPARATE GATE FROM `e3_diagnostics_staleness_lint`. That lint keys on a
    diagnostics LATCH being re-read. This form touches NO latch, so form 1 is structurally
    blind to it. On V3-EXQ-699 form 1 fired on `:929` (`last_score_diagnostics`,
    incidental) and was silent on `:882` -- the run's PRIMARY DV, and the site that forced
    the withdrawal of the `levers_compound` finding. Keeping the gates separate also keeps
    the adjudication honest in the other direction: 699's `active_frac == 1.0` is
    INFORMATIVE because its diagnostics are genuinely fresh, where 708's identical 1.0 was
    vacuous. Freshness and replication are independent defects.

    THREE COVERED EXPOSURES, all confirmed on
    `experiments/v3_exq_699_pcomp_demotion_x_gonogo_composition.py`:
      (a) `:882`/`:899` -- `committed_class = int(action[0].argmax().item())` accumulated
          into a class histogram on every P2 env step. THE PRIMARY DV.
      (b) `:913` -- `agent.e3._last_selected_trajectory`, a per-selection latch (assigned
          only in `select()`, `e3_selector.py:3108`) read once per env step. Empirical
          confirmation that (a) and (b) are the same defect: 699's
          `selected_class_entropy_nats == committed_class_entropy_nats` to 6dp on ALL 12
          arm-seeds -- two nominally independent readouts are one number.
      (c) `:856` -- `pre_e3_classes` from `agent.generate_trajectories(...)`, which returns
          CACHED candidates on a non-E3 tick (`agent.py:4812`, MECH-057a gate).

    CONSTRUCT MISMATCH IS THE GENERAL HAZARD, not staleness. The readout's sampling unit
    (env step) must match what the mechanism acts on (selection). 699's occupancy entropy
    is a genuine measurement of one thing and an invalid measurement of the thing its
    claims are about.

    SCOPE -- fires only on an ACCUMULATION (histogram / counter / running sum / gated
    increment) of a SCALAR REDUCTION of the gated value, inside a loop. Stepping the env
    with the action, and storing the action in a replay buffer, are CORRECT at every step
    (the held action really is the action taken) and must not fire; that is what the
    `_SCALAR_REDUCTIONS` requirement buys. Discharged by the same exemptions as form 1:
    clear-before-select, a `ticks["e3_tick"]` guard, a direct `e3.select(...)` call site,
    or identity-freshness dedup.

    NOT EVERY FIRE IS CONTAMINATION -- this is the triage test the 699 and 708 autopsies
    established, and it is why this gate reports rather than blocks. An inflated n is NOT
    sufficient. A gate is SAFE when THRESHOLD-INVARIANT: a floor of literally 0.0 (">0"
    cannot be manufactured from an all-zero record, nor collapsed from a genuine
    positive), an exact-zero reading, or a fraction saturated at exactly 1.0. A gate is AT
    RISK when it is a continuous margin against a non-trivial floor. It is DISQUALIFYING
    when the statistic is a DISTRIBUTION-SHAPE measure -- entropy, variance, any
    histogram-derived quantity -- because replication reweights the distribution itself,
    which is exactly the operation such statistics are sensitive to.

    CALIBRATION, and the limit of it. A matched replay on the
    `v3_exq_663_modulatory_channel_routing` driver measured this defect's cost at
    +0.01% / +0.64% / -0.87% -- sub-1% and sign-varying (REE_assembly WORKSPACE_STATE
    2026-07-20T06:25Z, ree-v3 `5433e3ab1c`), so 662/663's estimates stand. That bounds the
    defect WHERE ARM SYMMETRY MAKES IT CANCEL and where the DV is a continuous magnitude.
    It does NOT bound it for entropy DVs, nor where arms differ in hold duration -- the
    very quantity doing the weighting. See autopsy sec 4d.

    Same static-AST limitation class as form 1: exemptions are detected file-wide rather
    than per-read-site. WARN-ONLY IN BOTH MODES -- it never hardens under `--paths`. It
    flags a SUSPECTED hold-weighted readout, never a proven one, and completed runs are
    re-adjudicated via `/failure-autopsy`, never rewritten.

    Opt-out: E3_HOLD_WEIGHTED_READOUT_EXEMPT = "<reason>".
    """
    try:
        src = path.read_text(encoding="utf-8")
        tree = ast.parse(src, filename=str(path))
    except (OSError, UnicodeDecodeError, SyntaxError):
        return None  # check_script already reports unreadable / syntax errors

    if _E3_HOLD_WEIGHTED_EXEMPT_MARKER in src:
        return None
    if "select_action" not in src:
        return None  # not driving the agent

    # Same discharges as form 1, including (e) the shared sentinel-key helper.
    # `_clears_an_e3_latch` covers the form-1 attributes; a clear of
    # `_last_selected_trajectory` counts here too.
    if _uses_shared_fresh_select_helper(tree):
        return None
    if _clears_an_e3_latch(tree):
        return None
    if _guards_e3_latch_by_identity(tree):
        return None
    if any(isinstance(n, ast.Assign) and isinstance(n.value, ast.Constant)
           and n.value.value is None
           and any(isinstance(t, ast.Attribute)
                   and t.attr in _E3_SELECTION_LATCHED_ATTRS for t in n.targets)
           for n in ast.walk(tree)):
        return None
    if any(isinstance(n, ast.Constant) and n.value == "e3_tick" for n in ast.walk(tree)):
        return None
    if any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
           and n.func.attr == "select" and isinstance(n.func.value, ast.Attribute)
           and n.func.value.attr == "e3" for n in ast.walk(tree)):
        return None

    loop_spans = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.For, ast.While)):
            last = max((getattr(x, "lineno", node.lineno) for x in ast.walk(node)),
                       default=node.lineno)
            loop_spans.append((node.lineno, last))
    if not loop_spans:
        return None

    def _in_loop(ln: int) -> bool:
        return any(lo <= ln <= hi for lo, hi in loop_spans)

    tainted, scalars = _e3_cadence_gated_sources(tree)
    hits = [(ln, s) for ln, s in _hold_weighted_accumulations(tree, tainted, scalars)
            if _in_loop(ln)]
    # Form (b) needs no accumulation shape: the latch READ inside the loop is the defect,
    # exactly as in form 1.
    hits += [(r.lineno, "selected_traj") for r in _e3_selection_latch_reads(tree)
             if _in_loop(r.lineno)]
    if not hits:
        return None

    lines = sorted({ln for ln, _ in hits})
    sources = sorted({s for _, s in hits})
    _LABEL = {"select_action": "the select_action() return value",
              "selected_traj": "agent.e3._last_selected_trajectory",
              "candidates": "the e3_tick-gated candidate list"}
    what = ", ".join(_LABEL[s] for s in sources)
    return (f"HOLD-WEIGHTED E3 READOUT: accumulates a per-step statistic from {what} "
            f"inside a driver loop (line(s) {', '.join(str(n) for n in lines[:6])}). "
            "agent.py:5430 returns the HELD action on `not ticks[\"e3_tick\"]` BEFORE "
            "e3.select() is reached, and generate_trajectories (agent.py:4812) returns "
            "CACHED candidates on the same condition -- so each commitment is weighted by "
            "its HOLD DURATION (cadence default 10, varying 5-20 under MECH-093 arousal). "
            "This touches NO diagnostics latch, so the e3_diagnostics_staleness gate is "
            "blind to it: on V3-EXQ-699 that gate fired only on an incidental read while "
            "THIS site carried the primary DV, and the `levers_compound` finding was "
            "withdrawn. TRIAGE, do not assume contamination -- an inflated n is not "
            "sufficient. SAFE if threshold-invariant (a 0.0 floor, an exact zero, a "
            "fraction saturated at 1.0); AT RISK for a continuous margin against a "
            "non-trivial floor; DISQUALIFYING for a distribution-shape statistic "
            "(entropy/variance/histogram), which replication reweights directly. The "
            "663 replay bounding this at <1% and sign-varying applies only where arm "
            "symmetry cancels it and the DV is a magnitude -- not to entropy DVs, nor "
            "where arms differ in hold duration. FIX: gate the accumulation on a fresh "
            "selection (clear-before-select, or `ticks[\"e3_tick\"]`), emit "
            "`n_fresh_select` / `n_latched` / `fresh_select_yield`, and if the "
            "hold-weighted quantity is wanted too, emit BOTH kept distinct. Reference: "
            "experiments/v3_exq_785a_mech463_arousal_exogenous_urgency_decomp.py. "
            "Exempt with E3_HOLD_WEIGHTED_READOUT_EXEMPT = \"<reason>\".")


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
    parser.add_argument("--checks", nargs="*", default=None, choices=CHECK_NAMES,
                        help="Restrict to specific checks (default: all). E.g. "
                             "`--checks manifest_writer` runs ONLY the manifest-writer "
                             "chokepoint gate -- used by the commit-time gate in "
                             "scripts/precommit_contracts.sh so it does not expand the "
                             "conformance/degeneracy/arm-fingerprint contracts to the "
                             "non-v3_exq_ scripts it also scopes.")
    args = parser.parse_args()

    selected = set(args.checks) if args.checks else set(CHECK_NAMES)

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
    # Degeneracy self-report enforcement: same hard-under-`--paths` / advisory-in-
    # full-glob policy as the arm-fingerprint gate. A NEW claim-pressing script the
    # author is queuing without a non-degeneracy self-report is a real error; the same
    # gap on a historical script is a backlog item.
    degen_hard = bool(args.paths)
    # Manifest-writer chokepoint enforcement: same hard-under-`--paths` / advisory-in-
    # full-glob policy. A NEW script the author is queuing that hand-rolls a manifest
    # write instead of routing through pack_writer.write_flat_manifest is a real error;
    # the same gap on a historical script is the pre-2026-07-12 migration backlog.
    manifest_writer_hard = bool(args.paths)

    n_ok = 0
    n_exempt = 0
    failures: List[Tuple[Path, str]] = []
    warnings: List[Tuple[Path, str]] = []
    arm_fp_warnings: List[Tuple[Path, str]] = []
    degen_warnings: List[Tuple[Path, str]] = []
    manifest_writer_warnings: List[Tuple[Path, str]] = []
    anchor_warnings: List[Tuple[Path, str]] = []
    specimen_warnings: List[Tuple[Path, str]] = []
    n_anchor_superseded = 0
    recomput_warnings: List[Tuple[Path, str]] = []
    e3_stale_warnings: List[Tuple[Path, str]] = []
    e3_hold_warnings: List[Tuple[Path, str]] = []
    for p in paths:
        rel = p.relative_to(REPO_ROOT) if REPO_ROOT in p.parents or p == REPO_ROOT else p
        if "conformance" in selected:
            ok, reason = check_script(p)
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
        if "readiness" in selected:
            warn = readiness_lint(p)
            if warn:
                warnings.append((p, warn))
        if "arm_fingerprint" in selected:
            arm_fp = arm_fingerprint_lint(p)
            if arm_fp:
                if arm_fp_hard:
                    failures.append((p, arm_fp))
                else:
                    arm_fp_warnings.append((p, arm_fp))
        if "degeneracy" in selected:
            degen = degeneracy_selfreport_lint(p)
            if degen:
                if degen_hard:
                    failures.append((p, degen))
                else:
                    degen_warnings.append((p, degen))
        if "manifest_writer" in selected:
            mw = manifest_writer_lint(p)
            if mw:
                if manifest_writer_hard:
                    failures.append((p, mw))
                else:
                    manifest_writer_warnings.append((p, mw))
        if "anchor_reachability" in selected:
            anch = anchor_reachability_lint(p)
            if anch:
                # WARN-only in BOTH modes -- see anchor_reachability_lint() for why
                # this one never hardens under --paths.
                sup = anchor_supersession_lint(p)
                if sup:
                    # ANNOTATE, never withdraw: the defect is real, just not
                    # actionable in place. Prefixing (rather than re-bucketing into a
                    # separate section) is deliberate -- it keeps the warning in the
                    # REACHABILITY WARNINGS count and section, so an already-ran
                    # defect can never become invisible by being reclassified.
                    lineage = sup["lineage"].get("SUPERSEDES") or sup["reason"] or "(unstated)"
                    anch = f"[SUPERSEDED -> {lineage}] " + anch
                    if sup["note"]:
                        anch += " SUPERSESSION NOTE: " + sup["note"]
                    n_anchor_superseded += 1
                anchor_warnings.append((p, anch))
        if "anchor_reachability" in selected:
            spec = anchor_specimen_lint(p)
            if spec:
                specimen_warnings.append((p, spec))
        if "precondition_recomputability" in selected:
            rec = precondition_recomputability_lint(p)
            if rec:
                # WARN-only in BOTH modes -- see precondition_recomputability_lint()
                # for why this one never hardens under --paths.
                recomput_warnings.append((p, rec))
        if "e3_diagnostics_staleness" in selected:
            e3s = e3_diagnostics_staleness_lint(p)
            if e3s:
                # WARN-only in BOTH modes -- see e3_diagnostics_staleness_lint() for why
                # this one never hardens under --paths.
                e3_stale_warnings.append((p, e3s))
        if "e3_hold_weighted_readout" in selected:
            e3h = e3_hold_weighted_readout_lint(p)
            if e3h:
                # WARN-only in BOTH modes -- see e3_hold_weighted_readout_lint() for why
                # this one never hardens under --paths.
                e3_hold_warnings.append((p, e3h))

    print("", flush=True)
    print(f"[validate_experiments] checked {len(paths)} scripts: "
          f"{n_ok} OK, {n_exempt} exempt, {len(failures)} non-conforming, "
          f"{len(warnings)} readiness-warning(s), "
          f"{len(arm_fp_warnings)} arm-fingerprint-backlog, "
          f"{len(degen_warnings)} degeneracy-self-report-backlog, "
          f"{len(manifest_writer_warnings)} manifest-writer-backlog, "
          f"{len(anchor_warnings)} anchor-reachability-warning(s)"
          + (f" ({n_anchor_superseded} superseded)" if n_anchor_superseded else "") + ", "
          f"{len(recomput_warnings)} precondition-recomputability-warning(s), "
          f"{len(e3_stale_warnings)} stale-e3-diagnostics-warning(s), "
          f"{len(e3_hold_warnings)} hold-weighted-readout-warning(s)", flush=True)
    if e3_hold_warnings:
        # Advisory in BOTH modes (never hardens). Defect FORM 2 -- the diagnostics-latch
        # gate below is structurally blind to it, so this is a separate section rather
        # than more entries in that one. Pre-2026-07-20 backlog: drivers authored before
        # the V3-EXQ-699 re-adjudication. Fires here are a TRIAGE LIST, not a verdict --
        # threshold-invariant gates are safe, distribution-shape statistics are not.
        print("", flush=True)
        print("[validate_experiments] HOLD-WEIGHTED-E3-READOUT WARNINGS (advisory, non-blocking):", flush=True)
        for p, warn in e3_hold_warnings:
            rel = p.relative_to(REPO_ROOT) if REPO_ROOT in p.parents or p == REPO_ROOT else p
            print(f"  - {rel}: {warn}", flush=True)
    if e3_stale_warnings:
        # Advisory in BOTH modes (never hardens -- the clear/guard/select exemptions are
        # detected file-wide, so this flags a SUSPECTED inflated denominator, never a
        # proven one). Pre-2026-07-19 backlog: drivers authored before the
        # clear-before-select requirement, whose runs are already complete.
        print("", flush=True)
        print("[validate_experiments] STALE-E3-DIAGNOSTICS WARNINGS (advisory, non-blocking):", flush=True)
        for p, warn in e3_stale_warnings:
            rel = p.relative_to(REPO_ROOT) if REPO_ROOT in p.parents or p == REPO_ROOT else p
            print(f"  - {rel}: {warn}", flush=True)
    if recomput_warnings:
        # Advisory in BOTH modes (never hardens -- `measured` is computed from live run
        # data, so this flags a SUSPECTED mismatch between two expressions, never a
        # proven non-recomputable triple). Pre-2026-07-18 backlog: preconditions
        # authored before the recomputability requirement.
        print("", flush=True)
        print("[validate_experiments] Precondition-RECOMPUTABILITY WARNINGS (advisory, non-blocking):", flush=True)
        for p, warn in recomput_warnings:
            rel = p.relative_to(REPO_ROOT) if REPO_ROOT in p.parents or p == REPO_ROOT else p
            print(f"  - {rel}: {warn}", flush=True)
    if anchor_warnings:
        # Advisory in BOTH modes (never hardens -- reachability is not statically
        # decidable, so this flags a missing GUARD, not a proven-unreachable gate).
        # Pre-2026-07-18 backlog: anchors authored before assert_anchor_reachable.
        print("", flush=True)
        print("[validate_experiments] Readiness-anchor REACHABILITY WARNINGS (advisory, non-blocking):", flush=True)
        for p, warn in anchor_warnings:
            rel = p.relative_to(REPO_ROOT) if REPO_ROOT in p.parents or p == REPO_ROOT else p
            print(f"  - {rel}: {warn}", flush=True)
    if specimen_warnings:
        # Advisory, but the loudest of the advisory sections: this one says a change
        # is about to break the gate's OWN contract tests. Printed AFTER the
        # reachability list so it is the last anchor-related thing on screen.
        print("", flush=True)
        print("[validate_experiments] *** LINT-SPECIMEN WARNING -- read before landing ***", flush=True)
        for p, warn in specimen_warnings:
            rel = p.relative_to(REPO_ROOT) if REPO_ROOT in p.parents or p == REPO_ROOT else p
            print(f"  - {rel}: {warn}", flush=True)
    if manifest_writer_warnings:
        # Advisory in full-glob mode only (hard failures route to `failures` when
        # --paths is explicit). This is the pre-2026-07-12 backlog -- the ~1028
        # scripts that hand-roll a manifest write and predate the pack_writer
        # single-writer chokepoint (experimental_recording_standard sec 4).
        print("", flush=True)
        print("[validate_experiments] Manifest-writer chokepoint BACKLOG (advisory; hard under --paths):", flush=True)
        for p, warn in manifest_writer_warnings:
            rel = p.relative_to(REPO_ROOT) if REPO_ROOT in p.parents or p == REPO_ROOT else p
            print(f"  - {rel}: {warn}", flush=True)
    if degen_warnings:
        # Advisory in full-glob mode only (hard failures route to `failures` when
        # --paths is explicit). This is the pre-2026-06-12 backlog -- claim-pressing
        # scripts that predate the non_degenerate self-report net (2026-06-11).
        print("", flush=True)
        print("[validate_experiments] Degeneracy-self-report BACKLOG (advisory; hard under --paths):", flush=True)
        for p, warn in degen_warnings:
            rel = p.relative_to(REPO_ROOT) if REPO_ROOT in p.parents or p == REPO_ROOT else p
            print(f"  - {rel}: {warn}", flush=True)
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
