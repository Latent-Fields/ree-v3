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
               "ceiling_route_anchor_floor",
               "e3_diagnostics_staleness", "e3_hold_weighted_readout",
               "action_object_selection", "spearman_guard_shape",
               "dead_z_goal_stream", "hardcoded_dry_run", "emit_outcome_dry_run",
               "write_pack_dry_run", "dry_run_unreachable_criterion",
               "config_slice_declaration", "inert_salience_dacc_bias",
               "dacc_last_bundle", "agent_seed_order", "zworld_p0_warmup")

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


# Ceiling-below-random-anchor standing lint (autopsy competence-objective cluster
# 734/737b/742a, 2026-07-22, sec 5 Learning 2, user-adjudicated confirmed).
#
# A diagnostic/baseline script that SELF-ROUTES to a ceiling-class label
# (substrate_ceiling / ree_substrate_ceiling / learner_ceiling /
# learner_or_observability_ceiling) AND declares a `random_walk` FLOOR anchor is asserting
# that some LEARNER cannot clear a floor a control can, framed as a capacity / representation
# / observation-encoding limitation. Under a genuine ceiling a learner asymptotes toward its
# random anchor from below and plateaus; it does NOT end up systematically WORSE than random
# on an oracle-achievable env. A learner below its own random_walk anchor is therefore not at
# a ceiling -- it is optimising a different objective than the scored DV (the 734
# survival-vs-forage inversion: PPO survived 175.0 steps vs the oracle's 20.4 while foraging
# 17x less). Emitting a ceiling verdict on that run mislabels objective-misspecification as a
# substrate limitation -- the exact mis-route this autopsy exists to prevent, and the
# information (the below-random score) was present in 734/737b/742a and in every predecessor,
# consumed by nothing.
#
# The obligation is discharged by wiring experiments/_lib/anchor_floor_guard.py:
# `refuse_ceiling_below_random(label, learner_scores, random_anchor, ...)` into the
# self-route computation, which downgrades a ceiling label resting on a sub-random learner to
# substrate_not_ready_requeue and records the refuting numbers in the manifest. v3_exq_734 is
# the reference consumer.
#
# WARN-ONLY IN BOTH MODES -- like anchor_reachability_lint / precondition_recomputability_lint
# it never hardens under --paths, because whether a given cell's score is below its anchor is
# NOT statically decidable (both are computed from live run data). This can therefore flag
# only a MISSING GUARD, never an actually-unsupported verdict, so a full-glob run surfaces the
# backlog without blocking. It keys on an ACTUAL emitted ceiling route (a `label`/
# `interpretation_label` assignment to a ceiling constant, or a ceiling label used as an
# interpretation_grid KEY) rather than a bare string mention, so a driver that merely
# describes a ceiling label in prose does not fire (verified 2026-08-01: of the 3 corpus
# scripts declaring both a ceiling string and a random_walk floor -- 728/728b/734 -- only 734
# emits a ceiling label, and it wires the guard). Exempt with a reason ONLY when the ceiling
# route genuinely cannot rest on a sub-random learner (e.g. a readout-side control unaffected
# by the floor, the 737/742 recorded_preconditions case).
_CEILING_ANCHOR_FLOOR_EXEMPT_MARKER = "CEILING_ANCHOR_FLOOR_EXEMPT"
_CEILING_ROUTE_LABELS = frozenset({
    "substrate_ceiling", "ree_substrate_ceiling",
    "learner_ceiling", "learner_or_observability_ceiling",
})
_CEILING_GUARD_NAMES = ("refuse_ceiling_below_random", "anchor_floor_guard",
                        "ceiling_route_refusal")


def _emits_ceiling_route(tree: ast.Module) -> List[str]:
    """Ceiling-class labels this script actually SELF-ROUTES to, in source order.

    An emitted route is one of:
      * `label = "<ceiling>"` / `interpretation_label = "<ceiling>"` (a variable assignment);
      * a dict entry `"label"`/`"interpretation_label": "<ceiling>"`;
      * a `"<ceiling>": ...` dict KEY (the interpretation_grid emittable-label convention).
    A ceiling label appearing only inside a description STRING (prose) is NOT an emitted route
    and is deliberately not matched. Static scan only -- a label assembled at runtime is
    invisible, same limitation class as the other lints.
    """
    found: List[str] = []
    _LABEL_TARGETS = {"label", "interpretation_label"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            if isinstance(node.value, ast.Constant) and node.value.value in _CEILING_ROUTE_LABELS:
                for tgt in node.targets:
                    if isinstance(tgt, ast.Name) and tgt.id in _LABEL_TARGETS:
                        found.append(node.value.value)
        elif isinstance(node, ast.Dict):
            for k, v in zip(node.keys, node.values):
                k_is_str = isinstance(k, ast.Constant) and isinstance(k.value, str)
                # "<ceiling>": ...  -- a grid KEY naming an emittable label
                if k_is_str and k.value in _CEILING_ROUTE_LABELS:
                    found.append(k.value)
                # "label": "<ceiling>"
                elif (k_is_str and k.value in _LABEL_TARGETS
                        and isinstance(v, ast.Constant) and v.value in _CEILING_ROUTE_LABELS):
                    found.append(v.value)
    # de-dupe, preserve order
    seen: set = set()
    out: List[str] = []
    for lbl in found:
        if lbl not in seen:
            seen.add(lbl)
            out.append(lbl)
    return out


def _declares_random_walk_floor(tree: ast.Module) -> bool:
    """True iff the script declares a `random_walk` FLOOR anchor via a `floor="random_walk"`
    keyword argument (the capability_eval.build_report convention). Precise on purpose: a bare
    `"random_walk"` string mention (an arm listing, a docstring) is not a floor DECLARATION."""
    for node in ast.walk(tree):
        if isinstance(node, ast.keyword) and node.arg == "floor":
            v = node.value
            if isinstance(v, ast.Constant) and v.value == "random_walk":
                return True
    return False


def ceiling_route_anchor_floor_lint(path: Path) -> Optional[str]:
    """Ceiling-below-random-anchor check. Return a warning string, or None.

    Fires when a diagnostic/baseline script self-routes to a ceiling-class label AND declares a
    random_walk floor anchor BUT never wires anchor_floor_guard.refuse_ceiling_below_random --
    so a ceiling verdict resting on a learner below its own random floor would be emitted
    rather than refused (autopsy 734/737b/742a sec 5.2). WARN-only in both modes; see the
    module comment above for why it can only ever flag a missing guard.
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

    if (_CEILING_ANCHOR_FLOOR_EXEMPT_MARKER in names
            or _CEILING_ANCHOR_FLOOR_EXEMPT_MARKER in strings):
        return None
    if not (purposes & {"diagnostic", "baseline"}):
        return None

    emitted = _emits_ceiling_route(tree)
    if not emitted:
        return None  # does not self-route to a ceiling label -- nothing to guard
    if not _declares_random_walk_floor(tree):
        return None  # no random_walk floor anchor to compare a learner against

    if any(n in names for n in _CEILING_GUARD_NAMES):
        return None  # guard present

    return ("self-routes to ceiling-class label(s) "
            + ", ".join(emitted)
            + " AND declares a random_walk floor anchor, but never wires "
            "anchor_floor_guard.refuse_ceiling_below_random. A learner scoring BELOW its own "
            "random_walk anchor is NOT at a ceiling -- under a real ceiling it asymptotes "
            "toward the anchor from below, it does not end up worse than random on an "
            "oracle-achievable env; a below-random score is the signature of optimising a "
            "different objective than the scored DV (the 734 survival-vs-forage inversion, "
            "PPO survival 175.0 vs oracle 20.4 while foraging 17x less). Emitting a ceiling "
            "verdict on that run mislabels objective-misspecification as a substrate limit. "
            "Add `from experiments._lib.anchor_floor_guard import refuse_ceiling_below_random` "
            "and, after computing the self-route label, "
            "`label, rec = refuse_ceiling_below_random(label, {<learner_arm>: <score>, ...}, "
            "<random_walk_score>, rung=..., context=...)` (v3_exq_734 is the reference "
            "consumer), which downgrades a ceiling label resting on a sub-random learner to "
            "substrate_not_ready_requeue and records the refuting numbers. Exempt with "
            "CEILING_ANCHOR_FLOOR_EXEMPT = \"<reason>\" ONLY when the ceiling route genuinely "
            "cannot rest on a sub-random learner (e.g. a readout-side control unaffected by "
            "the floor -- the 737/742 recorded_preconditions case). See "
            "failure_autopsy_competence-objective-cluster-734-737b-742a_2026-07-22.md sec 5.2.")


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


# ---- dead z_goal stream (V3-EXQ-830, confirmed 2026-07-27) ---------------------------
# `z_goal` has exactly ONE writer in the whole substrate: the explicit
# `REEAgent.update_z_goal(...)` (ree_core/agent.py). Nothing in sense() /
# generate_trajectories() / select_action() / update_residue() touches it -- the two
# GoalState mutators, `goal_state.update(...)` and `goal_state.cue_pull(...)`, are BOTH
# called only from inside update_z_goal (verified: agent.py:9268 and agent.py:9195 are
# the sole call sites). So a driver that hand-rolls its inner loop and omits the call
# runs with z_goal pinned at its zero-init for the entire run. `GoalState.is_active()`
# (goal.py) then returns False, agent.py passes `current_z_goal=None` to every
# downstream consumer, and every goal-gated branch silently no-ops: the E3 goal term
# (e3_selector.py, gated on `goal_state.is_active() and goal_weight > 0`), the MECH-293
# ghost probes, MECH-288's slow BOCPD scale (z_goal joins the rollout latent_signature),
# MECH-189 super-ordinal anchors, the SD-057 incentive bank, the MECH-295 liking ->
# approach bridge, and the frontopolar counterfactual read. There is no error, no
# warning, and no manifest field that makes any of it visible.
_DEAD_ZGOAL_EXEMPT_MARKER = "DEAD_Z_GOAL_STREAM_EXEMPT"

# Config knobs whose behaviour is UNREACHABLE without the call. Deliberately just two:
# `z_goal_enabled` is the master gate (REEAgent constructs `goal_state` only when it is
# set, so every other goal knob is inert anyway without it), and
# `benefit_terrain_live_producer` is the SD-024 benefit-attractor producer, which lives
# INSIDE update_z_goal but ahead of the goal_state guard -- so it dies with the call even
# in a config that never enables z_goal. Downstream knobs (goal_weight, z_goal_inject,
# use_mech293_ghost_probes, use_incentive_token_bank, use_super_ordinal_goal_anchors)
# are NOT triggers on their own: setting one without z_goal_enabled is plain config
# inertness, which is the inert-arm-knob gate's job, not this one.
_DEAD_ZGOAL_TRIGGER_KNOBS = ("z_goal_enabled", "benefit_terrain_live_producer")


def _sets_knob_truthy(tree: ast.Module, knobs: Sequence[str]) -> List[str]:
    """Knobs the script sets to something other than a literal False/None/0.

    AST-based on purpose: a name-scan would fire on the many scripts that only DISCUSS
    `z_goal_enabled=True` in a docstring while configuring it False (V3-EXQ-551 does
    exactly that). Only two forms count as "setting" it -- a keyword argument
    (`REEConfig(..., z_goal_enabled=True)`) and an attribute assignment
    (`config.goal.z_goal_enabled = True`). A dict-literal `"z_goal_enabled": ...` is
    NOT counted: in this corpus that shape is overwhelmingly a manifest echo of the
    config, not the config itself.
    """
    def _is_false_literal(node: ast.expr) -> bool:
        return isinstance(node, ast.Constant) and node.value in (False, None, 0)

    found: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.keyword) and node.arg in knobs:
            if not _is_false_literal(node.value):
                found.add(node.arg)
        elif isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Attribute) and tgt.attr in knobs:
                    if not _is_false_literal(node.value):
                        found.add(tgt.attr)
    return sorted(found)


def _writes_z_goal_directly(tree: ast.Module) -> bool:
    """True if THIS module writes z_goal by any route (see `_writes_z_goal` for scope).

    Four discharges, all seen in the landed corpus:
      (a) `<agent>.update_z_goal(...)`         -- the canonical call.
      (b) `<...goal...>.update(...)`           -- a script that constructs its own
          GoalState and drives it directly. Matched on the RECEIVER NAME containing
          "goal" so it catches the real spellings (`goal_state.update(...)`, and the
          V3-EXQ-085h family's `goal_state_world` / `goal_state_resource`) without
          exempting every unrelated `.update()` in the file.
      (c) `<...>.cue_pull(...)`                -- the SD-057 L6 directional nudge.
      (d) assignment to a `_z_goal` attribute  -- the V3-EXQ-104/105/108/642 idiom of
          poking the attractor directly to stage a fixed goal.
    """
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            if isinstance(node, ast.Assign):
                for tgt in node.targets:
                    if isinstance(tgt, ast.Attribute) and tgt.attr == "_z_goal":
                        return True
            continue
        fn = node.func
        if fn.attr in ("update_z_goal", "cue_pull"):
            return True
        if fn.attr == "update":
            recv = fn.value
            name = (recv.id if isinstance(recv, ast.Name)
                    else recv.attr if isinstance(recv, ast.Attribute) else "")
            if "goal" in name.lower():
                return True
    return False


# Parsed `experiments/**.py` helper modules, keyed by (path, mtime_ns, size) rather than
# by path alone so a module edited mid-session is never served stale. That is not
# hypothetical here: `_lib/goal_pipeline_tier1.py` -- the module the trigger resolver
# below follows -- was being edited by a concurrent session while this resolver was
# written, and its `z_goal_enabled=True` moved line 221 -> 254 between two reads.
# Mirrors `_BASELINE_MODULE_CACHE`'s discipline (see the config_slice cross-module
# resolver) for the same reason.
_LOCAL_MODULE_CACHE: Dict[Tuple[str, int, int], Optional[ast.Module]] = {}


# Dotted-name -> rel-path resolutions. Memoised because this is FILESYSTEM work
# (`is_file()` per candidate prefix) and, since the trigger half started resolving
# helpers, it runs for every import of every driver rather than only for the handful
# that carry an in-file knob -- tens of thousands of stat calls across the corpus.
# Keyed on the dotted name alone, which is safe because it caches only WHERE a module
# is, never its CONTENT -- freshness is the stat-keyed `_LOCAL_MODULE_CACHE`'s job.
# (One consequence, stated rather than defended against: a module CREATED mid-session
# stays negatively cached. Irrelevant for a scan, and the alternative -- re-statting a
# miss -- gives back the saving, since misses are the common case.)
_LOCAL_MODULE_PATH_CACHE: Dict[str, Optional[str]] = {}


def _resolve_local_experiment_module(raw: str) -> Optional[str]:
    """Map a dotted import to a file under `experiments/`, or None.

    Scripts run with `experiments/` on `sys.path`, so both `from
    scaffolded_sd054_onboarding import ...` and `from experiments._lib.foo import ...`
    resolve there. Longest-prefix match, so `experiments._lib.stats` finds
    `_lib/stats.py`.
    """
    if raw in _LOCAL_MODULE_PATH_CACHE:
        return _LOCAL_MODULE_PATH_CACHE[raw]
    cand = raw[len("experiments."):] if raw.startswith("experiments.") else raw
    found: Optional[str] = None
    while cand:
        rel = cand.replace(".", "/") + ".py"
        if (EXPERIMENTS_DIR / rel).is_file():
            found = rel
            break
        if "." not in cand:
            break
        cand = cand.rsplit(".", 1)[0]
    _LOCAL_MODULE_PATH_CACHE[raw] = found
    return found


def _local_module_stat_key(rel: str) -> Optional[Tuple[str, int, int]]:
    path = EXPERIMENTS_DIR / rel
    try:
        st = path.stat()
    except OSError:
        return None
    return (str(path), st.st_mtime_ns, st.st_size)


def _parse_local_experiment_module(rel: str) -> Optional[ast.Module]:
    key = _local_module_stat_key(rel)
    if key is None:
        return None
    if key not in _LOCAL_MODULE_CACHE:
        try:
            _LOCAL_MODULE_CACHE[key] = ast.parse(
                (EXPERIMENTS_DIR / rel).read_text(encoding="utf-8"), filename=rel)
        except (OSError, UnicodeDecodeError, SyntaxError, ValueError):
            _LOCAL_MODULE_CACHE[key] = None
    return _LOCAL_MODULE_CACHE[key]


def _called_names(tree: ast.Module) -> Set[str]:
    """Every name that appears in call position: `foo(...)` and `mod.foo(...)`."""
    out: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                out.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                out.add(node.func.attr)
                if isinstance(node.func.value, ast.Name):
                    out.add(node.func.value.id)
    return out


def _imported_local_helpers(tree: ast.Module) -> Dict[str, Tuple[str, Optional[str]]]:
    """`local alias -> (module rel path, original name)`, for locally-resolvable imports.

    The original name is None for a whole-module import (`import _lib.foo as m`), where
    the callable is named at the CALL site (`m.build_config(...)`) rather than at the
    import. Both spellings are collected in one walk so the discharge half
    (`_uses_a_z_goal_driving_helper`) and the trigger half (`_helper_sets_knob_truthy`)
    agree by construction about which modules count as "local".
    """
    out: Dict[str, Tuple[str, Optional[str]]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            rel = _resolve_local_experiment_module(node.module)
            if rel:
                for a in node.names:
                    out[a.asname or a.name] = (rel, a.name)
        elif isinstance(node, ast.Import):
            for a in node.names:
                rel = _resolve_local_experiment_module(a.name)
                if rel:
                    out[a.asname or a.name.split(".")[-1]] = (rel, None)
    return out


# `(stat key, knobs) -> {function name: [knob it sets truthy]}` for a local helper
# module. Bounded two ways: by the stat key (one entry per module VERSION per session)
# and by the substring pre-filter in `_local_module_knob_setters`, which returns {}
# without parsing at all for a module whose text never mentions a knob. That pre-filter
# is what keeps this from adding a parse for every locally-imported module in the
# corpus -- only ~10 files under experiments/ mention either trigger knob, so the extra
# evictions of the shared one-entry corpus-scan cache (tests/contracts/conftest.py) stay
# inside the residue that test_shared_scan_parses_each_file_once already budgets for.
_LOCAL_KNOB_SETTER_CACHE: Dict[
    Tuple[Tuple[str, int, int], Tuple[str, ...]], Dict[str, List[str]]] = {}


def _local_module_knob_setters(rel: str, knobs: Sequence[str]) -> Dict[str, List[str]]:
    """`function name -> knobs that function sets truthy`, for one local helper module.

    ONE LEVEL ONLY: each function's own body is scanned with the same `_sets_knob_truthy`
    used on the driver; a function this one calls is NOT followed. That bounds the walk
    and keeps the discharge/trigger asymmetry visible -- see `_helper_sets_knob_truthy`.
    """
    key = _local_module_stat_key(rel)
    if key is None:
        return {}
    cache_key = (key, tuple(knobs))
    if cache_key in _LOCAL_KNOB_SETTER_CACHE:
        return _LOCAL_KNOB_SETTER_CACHE[cache_key]

    out: Dict[str, List[str]] = {}
    try:
        src = (EXPERIMENTS_DIR / rel).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        _LOCAL_KNOB_SETTER_CACHE[cache_key] = out
        return out
    # Substring pre-filter. A module that never spells a knob cannot set it in any of the
    # AST forms `_sets_knob_truthy` recognises, so this is a sound superset -- and it is
    # the difference between parsing ~10 modules and parsing every local import in the
    # corpus.
    if any(k in src for k in knobs):
        sub = _parse_local_experiment_module(rel)
        if sub is not None:
            for node in ast.walk(sub):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    hit = _sets_knob_truthy(
                        ast.Module(body=list(node.body), type_ignores=[]), knobs)
                    if hit:
                        out[node.name] = hit
    _LOCAL_KNOB_SETTER_CACHE[cache_key] = out
    return out


def _helper_sets_knob_truthy(tree: ast.Module, knobs: Sequence[str]) -> List[str]:
    """Knobs a LOCAL config-builder helper sets, for a driver that imports and calls it.

    THE BLIND SPOT THIS CLOSES, and why it is not merely theoretical. `_sets_knob_truthy`
    reads only in-file keyword arguments and attribute assignments, so a driver whose
    config is assembled inside a `_lib` builder shows the scan NO knob at all and the
    trigger half never arms. Confirmed live 2026-07-28: `_lib/goal_pipeline_tier1.py`'s
    `build_config()` sets `z_goal_enabled=True` (with `goal_weight=0.5`), and
    V3-EXQ-786 / 786a / 786b call it, then hand-roll a loop around the real
    `REEAgent.select_action(candidates, ticks)` with no `update_z_goal` and no
    StepHarness. Both guards were blind at once -- the lint because the knob is in
    another file, and the runtime backstop because those three never pass
    `agent=` / `z_goal_stream_stats=` to `write_flat_manifest`. That is the V3-EXQ-830
    defect shape, carried by three landed drivers.

    Mirrors the config_slice cross-module resolver (which parses the
    `_lib/baselines/<mod>.py` a helper comes from) in structure and caching discipline.

    THREE NARROWINGS, each load-bearing:
      - ONE LEVEL. The helper's own body is scanned; what IT calls is not followed. A
        transitive walk is what made the sibling discharge useless (`_lib/
        arm_fingerprint.py` reaches `_harness` and would exempt most of the corpus), and
        the same unboundedness applies in the trigger direction.
      - CALLED, not merely imported. An unused import builds no config.
      - LOCALLY RESOLVABLE ONLY. `**kwargs` splats, preset factories reached through
        arbitrary indirection, and anything outside `experiments/` stay invisible; the
        residual blind spot is narrowed, not eliminated.

    ASYMMETRY WITH THE DISCHARGE HALF -- do not "fix" this into symmetry. The discharge
    is deliberately NOT extended to follow helpers, because the helper the 786 family
    calls for its WARMUP (`warmup_train`, which drives StepHarness and therefore does
    write z_goal) is not the loop that carries the defect: their MEASUREMENT loop is
    hand-rolled and goal-dead. Discharging on a warmup helper would exempt exactly the
    three drivers this resolver exists to catch. What keeps the cohort that DELEGATES its
    whole loop (`run_seed_arm` -> StepHarness: V3-EXQ-471a/475a/483c-e/490g-k/524a/620/
    620b/625/625b/625c/784/827/827a/828) out of the fire set is the scope gate below --
    measured 2026-07-29: 20 of 20 of them make no direct `sense`/`select_action` call at
    all, so the driver-shape requirement excludes them without any discharge rule.
    """
    imported = _imported_local_helpers(tree)
    if not imported:
        return []

    # SHORT-CIRCUIT BEFORE THE CALL WALK, and this ordering is the cost of the feature.
    # `_local_module_knob_setters` is cached and does no parse for a module whose text
    # never mentions a knob, so this pass is nearly free -- whereas collecting call sites
    # is a full `ast.walk` of the driver, and this resolver now runs for EVERY driver
    # that carries no in-file knob (i.e. nearly the whole corpus). Almost none of them
    # import a knob-bearing module, so almost none of them should pay for that walk.
    relevant = {alias: (rel, orig) for alias, (rel, orig) in imported.items()
                if _local_module_knob_setters(rel, knobs)}
    if not relevant:
        return []

    # One walk for both call spellings: a bare `f(...)`, and `alias.attr(...)` (the
    # whole-module import, where the callable is named at the call site).
    called: Set[str] = set()
    mod_attr_calls: Dict[str, Set[str]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            called.add(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            called.add(node.func.attr)
            if isinstance(node.func.value, ast.Name):
                called.add(node.func.value.id)
                mod_attr_calls.setdefault(node.func.value.id, set()).add(node.func.attr)

    found: Set[str] = set()
    for alias, (rel, orig) in relevant.items():
        setters = _local_module_knob_setters(rel, knobs)
        if orig is not None and alias in called:
            found.update(setters.get(orig, ()))
        for attr in mod_attr_calls.get(alias, ()):
            found.update(setters.get(attr, ()))
    return sorted(found)


def _uses_a_z_goal_driving_helper(tree: ast.Module) -> bool:
    """True if the script CALLS a helper it imported from a z_goal-writing local module.

    This discharge exists for one real family and is deliberately narrow. The ~27 scripts
    of the V3-EXQ-460/466/467/468/629/797/799 group drive their warmup through
    `experiments/scaffolded_sd054_onboarding.py`'s ScaffoldedSD054OnboardingScheduler,
    which calls `update_z_goal` every Stage-0/P1/P2 step. Those scripts DO seed z_goal.
    Their measurement loop then stops calling it -- and because neither `REEAgent.reset()`
    (verified: it resets ~20 subsystems and the MECH-292 ghost bank, but never
    `goal_state`) nor GoalState's decay runs without the call (decay lives INSIDE
    `GoalState.update`), z_goal FREEZES at its post-warmup value rather than returning to
    zero. `is_active()` stays True and every consumer fires, just against a stale goal.

    That frozen-goal condition is a DIFFERENT defect from the dead-zero stream this lint
    names, and folding them together would make the warning text simply false for the
    larger group ("pinned at zero-init for the whole run" is not what happens to them).
    So they are discharged here and left to be triaged on their own terms.

    TWO deliberate narrowings, both load-bearing:
      - The source module must write z_goal ITSELF (`_writes_z_goal_directly`); its own
        imports are NOT followed. Following them makes the discharge useless: `_lib/
        arm_fingerprint.py` -- imported by nearly every modern script, since the
        fingerprint is mandatory -- imports `_harness`, so a transitive walk exempts most
        of the corpus including the V3-EXQ-830 case this lint exists for. Measured: 5
        fires with this narrowing, 0 without it.
      - The imported name must be CALLED, not merely imported. An unused import is not a
        z_goal write.
    """
    imported = {alias: rel for alias, (rel, _orig) in _imported_local_helpers(tree).items()}
    if not imported:
        return False
    called = _called_names(tree)
    for alias, rel in imported.items():
        if alias not in called:
            continue
        sub = _parse_local_experiment_module(rel)
        if sub is not None and _writes_z_goal_directly(sub):
            return True
    return False


def _writes_z_goal(tree: ast.Module) -> bool:
    """True if the script writes z_goal directly, or drives a helper that does."""
    return _writes_z_goal_directly(tree) or _uses_a_z_goal_driving_helper(tree)


def dead_z_goal_stream_lint(path: Path) -> Optional[str]:
    """Silently-dead z_goal stream check. Return an issue string, or None.

    Fires when a script does BOTH of:
      (1) enables a knob that is unreachable without `agent.update_z_goal(...)`
          (see `_DEAD_ZGOAL_TRIGGER_KNOBS`), and
      (2) drives a per-step loop without ever writing z_goal by any route
          (see `_writes_z_goal` for the four accepted discharges).

    Either half alone is fine. A script that never enables z_goal loses NOTHING by
    omitting the call -- `goal_state` is None, `update_z_goal` early-returns, and the
    omission is a true no-op. That asymmetry is why the trigger is knob-gated rather
    than "any hand-rolled loop": ~500 of the corpus's scripts hand-roll a loop without
    the call and are all correct.

    HOW THIS WAS FOUND. V3-EXQ-830 reused the V3-EXQ-816 policy-decomposition harness --
    which hand-rolls its loop and omits `update_z_goal`, harmless for 816 itself, which
    never reads z_goal -- to measure MECH-288's slow BOCPD scale, which DOES read it.
    830's readiness gate refused the dry-run smoke twice on `zgoal_present_frac = 0.0`
    before the cause was traced to the missing call. Without that gate the run would have
    spent ~5 hours of cloud time and reported a wiring artefact as a finding that CLOSED
    a design question. The same defect had already been confirmed once, in the opposite
    order: V3-EXQ-626's bespoke episode loop never called it, so z_goal sat at zero-init
    across every arm of a diagnostic whose C1-C5 criteria were ALL keyed on z_goal norm.
    626 was superseded by 626a (wired the call) and 626b (forced-seed positive control),
    and its manifest is correctly `superseded` / `non_contributory`.

    PREVENTION, not just detection: `experiments/_harness.py` StepHarness makes the call
    structurally unskippable (invariant 2 -- kwargs-only, because a POSITIONAL call
    collides with `latent` and raises TypeError every tick, which is how the
    EXQ-471/475/483/483a/483b/490/490b/490c/490e/490f/524 cohort failed). A script using
    StepHarness therefore cannot carry this defect, and `_writes_z_goal` sees the
    harness's own call through the import only when the harness module is the one being
    linted -- so StepHarness users are exempted explicitly below.

    RETROFIT IS NOT FREE -- say so before recommending one. `update_z_goal` is ALSO the
    SD-024 benefit-attractor producer: it calls `ResidueField.accumulate_benefit`, ahead
    of the `goal_state` guard, whenever `residue.benefit_terrain_live_producer` is set
    and the benefit pulse clears `benefit_live_producer_threshold`. Adding the call to an
    existing script therefore populates `benefit_rbf_field`, which un-zeroes the SD-025
    curiosity bonus in `HippocampalModule._curiosity_bonus` (previously exactly 0.0 on
    every call, because `RBFLayer.compute_local_density` early-returns on an empty active
    mask). That is a BEHAVIOUR CHANGE, not a wiring fix, and it means a "just add the
    call" patch to a landed harness is not comparable to the runs that came before it.

    THE TRIGGER FOLLOWS A LOCAL CONFIG-BUILDER HELPER ONE LEVEL -- and the blind spot it
    closes was a REAL CARRIER, not a hypothetical. This docstring previously framed the
    unfollowable-helper case as "UNDER-fires rather than over-fires ... the safe
    direction for a WARN". That framing was wrong twice over, and is retracted:

      - It was not merely under-firing, it was MISSING A LIVE FAMILY. Measured
        2026-07-28: `_lib/goal_pipeline_tier1.py`'s `build_config()` sets
        `z_goal_enabled=True` with `goal_weight=0.5`, and V3-EXQ-786 / 786a / 786b call
        it and then hand-roll a loop around the real
        `REEAgent.select_action(candidates, ticks)` with no `update_z_goal` and no
        StepHarness. Three landed drivers carrying the V3-EXQ-830 shape, invisible.
      - "Safe direction" presumed the runtime backstop covered the residue. For this
        family it did NOT: all three omit `agent=` / `z_goal_stream_stats=` at
        `write_flat_manifest`, so no `z_goal_stream` block is emitted at all. BOTH
        guards were blind at once, which is the state the two-guard argument below is
        supposed to rule out.

    So `_helper_sets_knob_truthy` now resolves a knob set inside a local
    (`experiments/`, `experiments/_lib/`) helper the driver imports AND calls, one level
    deep, mirroring the config_slice cross-module resolver. The residue is genuinely
    narrower but still real, and is stated as a limit rather than as a safety property:
    a `**kwargs` splat, a preset factory reached through arbitrary indirection, a helper
    outside `experiments/`, or a knob set two levels down all remain invisible. Treat
    silence from this gate as "no carrier FOUND", never as "no carrier".

    THE RUNTIME BACKSTOP IS COMPLEMENTARY BUT NOT A SUBSTITUTE -- and the 786 family is
    the proof, since it is opted out of the backstop too (see above). `REEAgent` counts,
    per `select_action` tick, how often `goal_state` was
    present and how often `GoalState.is_active()` held (`agent.z_goal_active_frac`),
    and `experiments/_lib/z_goal_stream.py` surfaces the pooled fraction as the
    manifest's `z_goal_stream` block via `manifest_core.stamp_recording_core` /
    `pack_writer.write_flat_manifest(agent=...)` and `StepHarness.z_goal_stream_stats()`.
    Being read from the run rather than the source, it does not share this scan's
    blind spot WHEN IT IS WIRED: a config built in an unfollowable helper still reports
    its true fraction -- but it is opt-in per driver, and the 786 family is the standing
    demonstration that "opt-in" and "covered" are different things.
    The two are complementary and neither subsumes the other -- this lint
    fires at AUTHORING time, before any compute is spent, whereas the counter is only
    readable once the run exists. Note the counter is RECORD-ONLY by design: 0.0 is
    correct for a goal-OFF parity arm or the ARM_NO_BENEFIT negative control, so it is
    a field to read against the run's design, never a gate.

    WARN-ONLY IN BOTH MODES -- it never hardens under `--paths`. The landed corpus's
    carriers have all run, and a completed run's pre-registered emission is not
    rewritten; nine of the ten landed carriers are already `non_contributory` and the
    tenth (V3-EXQ-615) is arm-symmetric. This gates NEW scripts.

    Opt-out: DEAD_Z_GOAL_STREAM_EXEMPT = "<reason>" -- appropriate for a script that
    deliberately measures the zero-goal condition (a goal-OFF parity arm, or a negative
    control like V3-EXQ-626b's ARM_NO_BENEFIT).
    """
    try:
        src = path.read_text(encoding="utf-8")
        tree = ast.parse(src, filename=str(path))
    except (OSError, UnicodeDecodeError, SyntaxError):
        return None  # check_script already reports unreadable / syntax errors

    if _DEAD_ZGOAL_EXEMPT_MARKER in src:
        return None

    # Direct first -- it is the cheap check and it short-circuits nearly every file in
    # the corpus. Only when the driver shows NO knob of its own is the local
    # config-builder helper resolved (the V3-EXQ-786 blind spot).
    knobs = _sets_knob_truthy(tree, _DEAD_ZGOAL_TRIGGER_KNOBS)
    via_helper = [] if knobs else _helper_sets_knob_truthy(tree, _DEAD_ZGOAL_TRIGGER_KNOBS)
    knobs = knobs or via_helper
    if not knobs:
        return None

    # StepHarness pins the call as invariant 2 -- a user of it cannot carry the defect.
    if any(isinstance(n, ast.Name) and n.id == "StepHarness" for n in ast.walk(tree)) \
            or any(isinstance(n, ast.alias) and n.name == "StepHarness"
                   for n in ast.walk(tree)):
        return None

    if _writes_z_goal(tree):
        return None

    # Scope to a driver: an agent stepped in a loop. A pure unit probe that builds a
    # config and asserts on it has no stream to kill.
    if not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
               and n.func.attr in ("sense", "select_action")
               for n in ast.walk(tree)):
        return None
    if not any(isinstance(n, (ast.For, ast.While)) for n in ast.walk(tree)):
        return None

    where = (" (set inside a local config-builder helper this driver calls, NOT in this "
             "file -- grep the helper before concluding the knob is absent)"
             if via_helper else "")
    return (f"DEAD Z_GOAL STREAM: sets {', '.join(knobs)}{where} but never writes z_goal "
            "-- no agent.update_z_goal(...), no goal_state.update(...)/cue_pull(...), no "
            "_z_goal assignment, and no StepHarness. update_z_goal is the SOLE writer in "
            "the substrate (sense/generate_trajectories/select_action/update_residue all "
            "leave z_goal untouched), so it stays pinned at zero-init for the whole run, "
            "GoalState.is_active() returns False, and agent.py passes current_z_goal=None "
            "to every consumer: the E3 goal term, MECH-293 ghost probes, MECH-288's slow "
            "BOCPD scale, MECH-189 super-ordinal anchors, the SD-057 incentive bank, the "
            "MECH-295 liking->approach bridge and the frontopolar counterfactual read all "
            "silently no-op. Nothing raises and no manifest field shows it (V3-EXQ-830: "
            "zgoal_present_frac 0.0, caught only by a readiness gate; V3-EXQ-626: all five "
            "criteria keyed on a z_goal that never left zero). FIX: drive the loop through "
            "experiments/_harness.py StepHarness, which pins the call (kwargs-only -- a "
            "POSITIONAL call collides with `latent` and raises TypeError every tick). "
            "CAUTION on retrofitting an EXISTING script: update_z_goal is also the SD-024 "
            "benefit-attractor producer (it calls ResidueField.accumulate_benefit), so "
            "adding it populates benefit_rbf_field and un-zeroes the SD-025 curiosity "
            "bonus -- a behaviour change, not a free wiring fix, and the run is then not "
            "comparable to its predecessors. Exempt with DEAD_Z_GOAL_STREAM_EXEMPT = "
            "\"<reason>\" when the zero-goal condition is the point (a goal-OFF parity "
            "arm, or a negative control like V3-EXQ-626b's ARM_NO_BENEFIT).")


# ---- hardcoded dry_run at the sanctioned writer --------------------------------------
# A driver that accepts `--dry-run` and reduces its work under it, but still hands
# `pack_writer.write_flat_manifest` a LITERAL `False`, silently disables two things:
#
#   (1) the V3-EXQ-696 relocation. `dry_run=True` makes the writer emit
#       `_dry_<run_id>.json` instead of `<run_id>.json`, which is what keeps a smoke
#       manifest out of `build_experiment_indexes.py`'s scoring set and off
#       `pending_review.md` as an action-required FAIL. With the flag hardcoded, a
#       `--dry-run` smoke writes a real-looking 1-seed / toy-episode manifest straight
#       into REE_assembly/evidence/experiments/. (`emit_outcome(dry_run=True)` relocates
#       it afterwards and `generate_pending_review.py` excludes dry_run-flagged
#       manifests, so this is a defence-in-depth gap rather than a live fire -- but the
#       first and cheapest layer is off.)
#   (2) the `[smoke] z_goal_stream:` report. `write_flat_manifest` prints the z_goal
#       liveness block (active_frac / writer_defect) ONLY under `dry_run`, deliberately:
#       gating it there is what keeps it from scrolling past unread over a multi-hour run
#       and from firing across the contract suite. Hardcoding the flag is therefore the
#       one thing that turns the V3-EXQ-830 early-warning off entirely.
#
# REACHABILITY IS THE WHOLE DISCRIMINATOR, and getting it wrong is why a grep massively
# over-counts. 617 corpus drivers pass a literal `False`; most are CORRECT, because the
# overwhelmingly common shape is
#
#     if dry_run:
#         print("dry-run complete; manifest not written.")
#         return ...
#     ... write_flat_manifest(..., dry_run=False, ...)
#
# where the writer is simply never reached in a smoke run and the literal is honest. The
# guard is also frequently in the CALLER (`if not args.dry_run: out = write_manifest(r)`,
# with the writer inside that helper), so an intraprocedural check still over-fires --
# hence the call-graph fixpoint in `_dry_reachable_functions`. Measured on this corpus:
# 617 literal-False sites -> 453 correctly quiet -> 164 genuine.
#
# Canonical fixed shape, V3-EXQ-722:
#     write_flat_manifest(manifest, out_dir, dry_run=bool(args.dry_run), ...)
#
# ADVISORY, and deliberately biased to UNDER-fire. It is a static scan: a flag threaded
# through a dict, a partial, or a module-level constant is invisible to it, and a driver
# whose smoke path exits inside a helper the fixpoint cannot resolve reads as reachable.
# A false WARN on a historical driver costs a reader one minute; hardening this would
# block commits on ~164 landed scripts whose runs are complete and whose re-run is
# hypothetical. Exempt with HARDCODED_DRY_RUN_EXEMPT = "<reason>" when the literal is
# deliberate -- e.g. the caller has already relocated or renamed the output, so a second
# `_dry_` prefix would double-prefix it.
_HARDCODED_DRY_RUN_EXEMPT_MARKER = "HARDCODED_DRY_RUN_EXEMPT"
_DRY_TOKENS = ("dry_run", "DRY_RUN", "dryrun", "smoke")


def _mentions_dry(node: ast.AST) -> bool:
    try:
        return any(t in ast.unparse(node) for t in _DRY_TOKENS)
    except Exception:
        return False


def _block_leaves_function(body: List[ast.stmt]) -> bool:
    """Does this statement list unconditionally leave the enclosing function?"""
    for st in body:
        if isinstance(st, (ast.Return, ast.Raise)):
            return True
        if isinstance(st, ast.Expr) and isinstance(st.value, ast.Call):
            fn = st.value.func
            nm = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
            if nm in ("exit", "_exit"):
                return True
    return False


def _is_negated_test(test: ast.AST) -> bool:
    try:
        t = ast.unparse(test).strip()
    except Exception:
        return False
    return (t.startswith("not ") or t.startswith("(not ")
            or " == False" in t or " is False" in t)


def _locally_dry_guarded(node: ast.AST, parent: Dict[int, ast.AST]) -> bool:
    """True when dry_run being truthy cannot reach `node` from its function's entry.

    Two shapes, both extremely common in the corpus:
      (a) node sits in the not-dry branch -- `if not dry_run: <node>` or the `else:` of
          `if dry_run: ...`;
      (b) node is dominated by an early `if dry_run: return` at some enclosing block.
    """
    chain: List[ast.AST] = [node]
    cur = node
    while id(cur) in parent:
        cur = parent[id(cur)]
        chain.append(cur)
    chain_ids = {id(x) for x in chain}

    for anc in chain:
        if isinstance(anc, ast.If) and _mentions_dry(anc.test):
            in_body = any(id(s) in chain_ids for s in anc.body)
            in_else = any(id(s) in chain_ids for s in anc.orelse)
            neg = _is_negated_test(anc.test)
            if (in_body and neg) or (in_else and not neg):
                return True

    for anc in chain:
        body = getattr(anc, "body", None)
        if not isinstance(body, list):
            continue
        idx = None
        for i, st in enumerate(body):
            if id(st) in chain_ids:
                idx = i
                break
        if idx is None:
            continue
        for st in body[:idx]:
            if isinstance(st, ast.If) and _mentions_dry(st.test):
                neg = _is_negated_test(st.test)
                if (not neg and _block_leaves_function(st.body)):
                    return True
                if neg and st.orelse and _block_leaves_function(st.orelse):
                    return True
    return False


def _dry_reachable_functions(tree: ast.AST, parent: Dict[int, ast.AST]) -> Set[Optional[str]]:
    """Names of functions callable while dry_run is truthy (module level always is).

    Least-fixpoint over intra-module call edges. Resolution is by BARE NAME, which is
    what makes this under-fire rather than over-fire on the ambiguous cases: two helpers
    sharing a name collapse to one node, so a guard on either marks both unreachable.
    """
    funcs = {n.name for n in ast.walk(tree)
             if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}

    def owner(n: ast.AST) -> Optional[str]:
        cur = parent.get(id(n))
        while cur is not None:
            if isinstance(cur, (ast.FunctionDef, ast.AsyncFunctionDef)):
                return cur.name
            cur = parent.get(id(cur))
        return None

    edges = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Call):
            fn = n.func
            nm = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", None)
            if nm in funcs:
                edges.append((nm, n))

    reachable: Set[Optional[str]] = {None}
    changed = True
    while changed:
        changed = False
        for nm, call in edges:
            if nm in reachable:
                continue
            if owner(call) in reachable and not _locally_dry_guarded(call, parent):
                reachable.add(nm)
                changed = True
    return reachable


def hardcoded_dry_run_lint(path: Path) -> Optional[str]:
    """Hardcoded `write_flat_manifest(dry_run=False)` check. Issue string, or None.

    Fires when ALL of:
      (1) the script has a smoke path -- an argparse `--dry-run` flag or a `dry_run`
          function parameter -- AND actually gates work on it (some `if`/`IfExp` tests
          it, which is what distinguishes a real smoke mode from a flag that is only
          forwarded to `emit_outcome`);
      (2) it calls `write_flat_manifest` with a LITERAL `False` for `dry_run`; and
      (3) that call site is reachable with dry_run truthy -- interprocedurally, per
          `_dry_reachable_functions`. This is the discriminator; see the block comment
          above for why (1)+(2) alone over-count by ~4x.

    Never blocking. See the block comment for the under-fire bias and the exemption.
    """
    try:
        src = path.read_text(encoding="utf-8")
        tree = ast.parse(src)
    except (OSError, SyntaxError, UnicodeDecodeError):
        return None
    if _HARDCODED_DRY_RUN_EXEMPT_MARKER in src:
        return None

    parent: Dict[int, ast.AST] = {}
    for n in ast.walk(tree):
        for c in ast.iter_child_nodes(n):
            parent[id(c)] = n

    literal_false_sites = []
    for n in ast.walk(tree):
        if not isinstance(n, ast.Call):
            continue
        fn = n.func
        nm = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", None)
        if nm != "write_flat_manifest":
            continue
        val = next((k.value for k in n.keywords if k.arg == "dry_run"), None)
        if isinstance(val, ast.Constant) and val.value is False:
            literal_false_sites.append(n)
    if not literal_false_sites:
        return None

    has_flag = False
    for n in ast.walk(tree):
        if isinstance(n, ast.Call):
            fn = n.func
            nm = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
            if nm == "add_argument":
                for a in n.args:
                    if isinstance(a, ast.Constant) and a.value in ("--dry-run", "--dry_run"):
                        has_flag = True
    has_param = any(
        isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        and any(a.arg == "dry_run"
                for a in list(n.args.args) + list(n.args.kwonlyargs))
        for n in ast.walk(tree)
    )
    gates_work = any(isinstance(n, (ast.If, ast.IfExp)) and _mentions_dry(n.test)
                     for n in ast.walk(tree))
    if not ((has_flag or has_param) and gates_work):
        return None

    reachable = _dry_reachable_functions(tree, parent)

    def owner(n: ast.AST) -> Optional[str]:
        cur = parent.get(id(n))
        while cur is not None:
            if isinstance(cur, (ast.FunctionDef, ast.AsyncFunctionDef)):
                return cur.name
            cur = parent.get(id(cur))
        return None

    live = [n for n in literal_false_sites
            if owner(n) in reachable and not _locally_dry_guarded(n, parent)]
    if not live:
        return None

    where = ", ".join(f"line {n.lineno} (in {owner(n) or '<module>'})" for n in live)
    return (
        f"accepts --dry-run and reduces its work under it, but passes a LITERAL "
        f"dry_run=False to write_flat_manifest at {where} -- and that call IS reached "
        f"in a smoke run. Two silent consequences: (1) the V3-EXQ-696 safeguard is "
        f"off, so `--dry-run` writes a real-looking 1-seed manifest as "
        f"<run_id>.json rather than _dry_<run_id>.json, straight into the indexer's "
        f"scoring set; (2) the dry_run-gated `[smoke] z_goal_stream:` liveness report "
        f"never prints, which is the one place an author sees a dead z_goal stream "
        f"BEFORE spending multi-hour cloud compute (the V3-EXQ-830 near-miss). Fix by "
        f"threading the script's own flag: `dry_run=bool(args.dry_run)` (or "
        f"`dry_run=dry_run` inside a function that takes it) -- see V3-EXQ-722 for the "
        f"canonical shape. Exempt with {_HARDCODED_DRY_RUN_EXEMPT_MARKER} = "
        f"\"<reason>\" when the literal is deliberate because the caller already "
        f"relocated or renamed the output."
    )


# ---- unthreaded emit_outcome(dry_run=) (V3-EXQ-696, second layer) ---------------------
# Sibling of `hardcoded_dry_run` above, on the OTHER end of the same smoke-containment
# chain. That gate watches the WRITER (`write_flat_manifest`), which picks the
# `_dry_<run_id>.json` filename; this one watches the SENTINEL EMITTER
# (`experiment_protocol.emit_outcome`), whose `dry_run=True` MOVES the just-written
# manifest out of REE_assembly/evidence/experiments/ into a throwaway scratch dir
# (system tempdir / ree_dry_run_manifests/) before writing the sentinel.
#
# WRITER FACTS, read off experiment_protocol.py rather than assumed:
#   * `dry_run` does NOT skip, rename or relocate the SENTINEL. The sentinel is written
#     either way, to <signal_dir>/<queue_id>.json or <signal_dir>/_manual/<stem>.json.
#     `evidence/experiments/_runner_signals/` is GITIGNORED (0 tracked files), and
#     verify_governance_cycle.py classifies it as telemetry, so the stray `_manual/`
#     sentinel a smoke leaves behind is local clutter and nothing more. It is NOT the
#     reason this gate exists.
#   * What `dry_run=True` actually does is `_relocate_dry_run_manifest(manifest_path)`.
#     That is the ONLY consequence, and it is why the gate below requires a non-None
#     `manifest_path`: an `emit_outcome` call with nothing to relocate is harmless.
#
# DEFENCE-IN-DEPTH, LIKE THE SIBLING -- DOWNGRADED 2026-07-28 (REE_assembly cb7298c1c4).
# This block previously read THIS ONE IS A LIVE FIRE, on the grounds that
# build_experiment_indexes.py contained no `dry_run` handling anywhere while
# generate_pending_review.py did, so a smoke contaminated claim_evidence.v1.json while
# pending_review.md -- the surface humans actually watch -- stayed clean. That downstream
# gap is now closed on both sides, so the consequence is the milder one. THE GATE STAYS:
# it is the first and cheapest layer, it fires at authoring time rather than after a
# manifest is on disk, and it is the only layer that lives in this repo.
#
# THE MECHANISM WAS NOT THE `_dry_` FLAT MANIFEST. That file keeps its `dry_run` flag and
# was never on the scoring path; the indexer scores the RUN PACK at
# `<experiment_type>/runs/<run_id>/manifest.json`. sync_v3_results._is_flat_v3 minted that
# pack FROM the smoke -- converting canonical `..._v3` run_ids unconditionally, consulting
# `_is_dry_run` only on the mid-string casualty branch -- and build_runpack_docs writes an
# `experiment_pack/v1` manifest with NO dry_run field, so the scored artifact was by
# construction indistinguishable from a real run. The repair gates the converter on both
# branches and has the indexer carry the flag over from the flat sibling by run_id.
#
# Confirmed instance, MECH-245: two 1-seed (`seeds: [0]`) `--dry-run` smokes of
# V3-EXQ-825 dated 2026-07-26T15:12:07Z / 15:14:39Z were tracked in git as BOTH
# `_dry_v3_exq_825_..._v3.json` (flagged, inert) and two `weakens` / FAIL run packs
# (unflagged, scored), the latter appearing in MECH-245's `recent_entries`. They were that
# claim's ENTIRE negative evidence base: `fail_runs: 2, pass_runs: 1,
# experimental_confidence: 0.571, evidence_quadrant: plausible_unproven`, where the one
# genuine run PASSED. After the repair: `fail_runs: 0`, `experimental_confidence: 0.771`,
# quadrant `confirmed_established`. Corpus-wide, 25 dry manifests had 25 matching scored
# packs and the rebuild dropped 22 entries (the rest already carried inactive directions).
# The relocation demonstrably works when it is threaded (48 relocated `_dry_` manifests
# sit in the scratch dir); the 825 pair is simply absent from it.
#
# REACHABILITY IS AGAIN THE DISCRIMINATOR, but note the grep error here runs the OPPOSITE
# way to the sibling's. A naive `grep -L` for the threading idioms over drivers that call
# emit_outcome and take --dry-run returns 82, roughly a third of the true population,
# because a driver that threads `dry_run` into `write_flat_manifest` (or any other call)
# matches the grep while its `emit_outcome` still does not. Measured on this corpus:
# 1164 drivers -> 928 call emit_outcome -> 734 also have a REAL smoke path -> 273 reach an
# unthreaded call. So a grep UNDER-counts by ~3.3x, where the sibling's over-counted by
# ~4x. Either way the AST fixpoint is what produces a usable number.
#
# Canonical fixed shape, V3-EXQ-825:
#     emit_outcome(outcome=..., manifest_path=..., run_id=..., dry_run=result["dry_run"])
#
# ADVISORY, and deliberately biased to UNDER-fire, for the same reasons as the sibling: a
# flag threaded through a dict or resolved in a helper the bare-name fixpoint cannot
# follow is invisible, and hardening would block commits on ~273 landed drivers whose runs
# are complete. Threading the flag is PROVABLY INERT on the evidence path -- it changes
# behaviour only under `--dry-run`, which by definition produced no evidence. Exempt with
# EMIT_OUTCOME_DRY_RUN_EXEMPT = "<reason>" when the caller already relocates or deletes
# the manifest itself.
_EMIT_OUTCOME_DRY_RUN_EXEMPT_MARKER = "EMIT_OUTCOME_DRY_RUN_EXEMPT"


def _emit_manifest_arg(call: ast.Call) -> Optional[ast.AST]:
    """The `manifest_path` argument node of an emit_outcome call, or None.

    None means "nothing to relocate": the argument is absent entirely, or is a literal
    `None`. emit_outcome's signature is `emit_outcome(outcome, manifest_path=None, *, ...)`
    so position 1 is the positional spelling.
    """
    node: Optional[ast.AST] = call.args[1] if len(call.args) >= 2 else None
    for k in call.keywords:
        if k.arg == "manifest_path":
            node = k.value
    if node is None or (isinstance(node, ast.Constant) and node.value is None):
        return None
    return node


def emit_outcome_dry_run_lint(path: Path) -> Optional[str]:
    """Unthreaded `emit_outcome(dry_run=)` check. Issue string, or None.

    Fires when ALL of:
      (1) the script has a smoke path -- an argparse `--dry-run` flag or a `dry_run`
          function parameter -- AND actually gates work on it (some `if`/`IfExp` tests
          it). Identical precondition to hardcoded_dry_run_lint, for the same reason: a
          flag that gates nothing is not a smoke mode.
      (2) it calls `emit_outcome` with a non-None `manifest_path` and either NO `dry_run=`
          keyword at all or a LITERAL `False`. Both leave the V3-EXQ-696 relocation off;
          the omission is by far the commoner spelling. A call with no manifest to
          relocate is skipped -- the sentinel alone is gitignored and harmless.
      (3) that call site is reachable with dry_run truthy -- interprocedurally, per
          `_dry_reachable_functions`. This is the discriminator; see the block comment
          above for the measured 3.3x gap against a grep.

    Never blocking. See the block comment for the MECH-245 evidence, why this is
    defence-in-depth rather than a live fire since 2026-07-28, and the exemption.
    """
    try:
        src = path.read_text(encoding="utf-8")
        tree = ast.parse(src)
    except (OSError, SyntaxError, UnicodeDecodeError):
        return None
    if _EMIT_OUTCOME_DRY_RUN_EXEMPT_MARKER in src:
        return None

    parent: Dict[int, ast.AST] = {}
    for n in ast.walk(tree):
        for c in ast.iter_child_nodes(n):
            parent[id(c)] = n

    unthreaded_sites = []
    for n in ast.walk(tree):
        if not isinstance(n, ast.Call):
            continue
        fn = n.func
        nm = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", None)
        if nm != "emit_outcome":
            continue
        if _emit_manifest_arg(n) is None:
            continue
        val = next((k.value for k in n.keywords if k.arg == "dry_run"), None)
        if val is None or (isinstance(val, ast.Constant) and val.value is False):
            unthreaded_sites.append(n)
    if not unthreaded_sites:
        return None

    has_flag = False
    for n in ast.walk(tree):
        if isinstance(n, ast.Call):
            fn = n.func
            nm = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
            if nm == "add_argument":
                for a in n.args:
                    if isinstance(a, ast.Constant) and a.value in ("--dry-run", "--dry_run"):
                        has_flag = True
    has_param = any(
        isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        and any(a.arg == "dry_run"
                for a in list(n.args.args) + list(n.args.kwonlyargs))
        for n in ast.walk(tree))
    gates_work = any(isinstance(n, (ast.If, ast.IfExp)) and _mentions_dry(n.test)
                     for n in ast.walk(tree))
    if not ((has_flag or has_param) and gates_work):
        return None

    reachable = _dry_reachable_functions(tree, parent)

    def owner(n: ast.AST) -> Optional[str]:
        cur = parent.get(id(n))
        while cur is not None:
            if isinstance(cur, (ast.FunctionDef, ast.AsyncFunctionDef)):
                return cur.name
            cur = parent.get(id(cur))
        return None

    live = [n for n in unthreaded_sites
            if owner(n) in reachable and not _locally_dry_guarded(n, parent)]
    if not live:
        return None

    where = ", ".join(f"line {n.lineno} (in {owner(n) or '<module>'})" for n in live)
    return (
        f"accepts --dry-run and reduces its work under it, but calls emit_outcome with a "
        f"manifest_path and no threaded dry_run= at {where} -- and that call IS reached in "
        f"a smoke run. emit_outcome(dry_run=True) is what MOVES the smoke manifest out of "
        f"REE_assembly/evidence/experiments/ into a scratch dir; unthreaded, the manifest "
        f"stays put. The _dry_ filename prefix does NOT protect it: "
        f"build_experiment_indexes.py globs *.json with no dry_run handling at all, so the "
        f"1-seed smoke lands in claim_evidence.v1.json and weights a real claim "
        f"(confirmed: two V3-EXQ-825 smokes are MECH-245's entire negative evidence base). "
        f"generate_pending_review.py excludes dry_run manifests, which is why this stays "
        f"invisible on pending_review.md. Fix by threading the script's own flag: "
        f"`dry_run=bool(args.dry_run)` (or `dry_run=result[\"dry_run\"]` -- V3-EXQ-825 is "
        f"the canonical shape). Exempt with {_EMIT_OUTCOME_DRY_RUN_EXEMPT_MARKER} = "
        f"\"<reason>\" when the caller already relocates or deletes the manifest itself."
    )


# ---- unthreaded write_pack(dry_run=) (V3-EXQ-696, third layer) -----------------------
# THIRD sibling of `hardcoded_dry_run` and `emit_outcome_dry_run`, on the one remaining
# unwatched end of the same smoke-containment chain. Read the three together:
#
#   gate                    watches                       what its dry_run does
#   hardcoded_dry_run       write_flat_manifest           picks the `_dry_<run_id>.json` name
#   emit_outcome_dry_run    experiment_protocol.emit_outcome  RELOCATES the flat manifest out
#                                                         of evidence/experiments/
#   write_pack_dry_run      pack_writer.write_pack        marks the RUN PACK dry (THIS ONE)
#
# WHY THE PACK IS THE ONE THAT MATTERS. The flat manifest is NOT on the indexer's scoring
# path; `build_experiment_indexes._scan_runs` scores the RUN PACK at
# `<experiment_type>/runs/<run_id>/manifest.json`. That is the whole mechanism of the
# MECH-245 incident written up above: the flat smoke kept its `dry_run` flag and was inert,
# while the pack minted from it carried NO dry_run field and was by construction
# indistinguishable from a real run.
#
# WHAT CHANGED ON 2026-07-28, and why this gate exists at all. Until that date a pack could
# not self-identify: `build_runpack_docs` wrote an `experiment_pack/v1` manifest with no
# dry_run key. Two repairs landed together (REE_assembly b84252a0ba; ree-v3 0f3bedbed4 /
# d5b1613615 / b281854a04): `sync_v3_results.build_runpack_docs` now copies a truthy
# top-level `dry_run` into the pack, and `ExperimentPackWriter.write_pack` gained a
# `dry_run=False` keyword that marks the pack directly.
#
# THE GAP THIS GATE CLOSES: `write_pack`'s `dry_run` IS OPT-IN. A driver that threads its
# flag into `write_flat_manifest` (so the flat file is correctly `_dry_`-prefixed) but NOT
# into `write_pack` still emits an UNFLAGGED pack. Such a pack is excluded only by
# `_load_dry_run_run_ids`'s cross-file carry BY RUN_ID from the flat sibling -- the exact
# coupling the 2026-07-28 work set out to dissolve, and the reason that run_id arm had to be
# KEPT rather than simplified away. So the coupling survives PRECISELY for drivers of this
# shape, and this gate is what makes that population visible instead of implicit.
#
# THAT THE FALLBACK IS LOAD-BEARING IS NOT THEORETICAL. The same session found
# `_load_dry_run_run_ids` had never globbed `*/*.json` (per-experiment-subdirectory flat
# manifests), so 13 dry packs were still scored as real evidence, contributing phantom runs
# to ARC-042, MECH-070, MECH-075, MECH-104, MECH-153, MECH-155, MECH-156 and MECH-231. A
# silent fallback is exactly where that class of hole hides.
#
# WHY CONDITION (b) -- "already threads dry_run into write_flat_manifest or emit_outcome".
# It is the discriminator that makes this gate about the PACK rather than about dry-run
# awareness in general. A driver that threads nothing is already reported by one or both
# siblings; firing here as well would triple-report one undifferentiated "the author did not
# think about --dry-run". Requiring a demonstrated threading elsewhere isolates the genuinely
# half-threaded shape: an author who KNOWS about smoke containment and covered one path while
# leaving the scored one open. "Threaded" means a `dry_run=` keyword whose value is not a
# literal `False` -- a literal False is the sibling gate's business, not evidence of intent.
#
# CORPUS POPULATION IS ZERO TODAY, AND THAT IS THE HONEST STATE. Of 1167 top-level
# `v3_exq_*.py` drivers, ZERO call `write_pack` at all; the only non-test caller anywhere in
# the repo is `experiments/run.py`, which has no `dry_run` and so no smoke path. Drivers
# reach the pack indirectly, via `write_flat_manifest` plus the `sync_v3_results` converter,
# which is the path the converter-side repair already gates. The gate is therefore EMPTY,
# not vacuous: the shape it detects is real, reachable and unguarded by anything else, and it
# fires the moment a driver calls `write_pack` directly under a smoke path -- which is a
# shape the pack writer's new keyword actively invites. See the pin comment in
# tests/contracts/test_write_pack_dry_run_lint.py for what would make it fire.
#
# ADVISORY, and deliberately biased to UNDER-fire, for the same reasons as both siblings: a
# flag threaded through a dict, a partial, or a helper the bare-name fixpoint cannot follow
# is invisible, and `dry_run` passed POSITIONALLY to write_pack (19th positional, so
# implausible but legal) is not matched. Exempt with WRITE_PACK_DRY_RUN_EXEMPT = "<reason>"
# when the caller already relocates, renames or deletes the pack itself.
_WRITE_PACK_DRY_RUN_EXEMPT_MARKER = "WRITE_PACK_DRY_RUN_EXEMPT"

# The calls whose threaded `dry_run=` counts as the author having demonstrated smoke
# awareness. Deliberately the two SIBLING GATES' own watched names, so the three gates
# partition one population rather than overlapping on it.
_DRY_THREADING_WITNESS_NAMES = ("write_flat_manifest", "emit_outcome")


def _call_name(node: ast.Call) -> Optional[str]:
    """Bare callee name of a Call -- `f(...)` and `obj.f(...)` both yield `f`.

    Same resolution the sibling gates use inline, factored out because this gate needs
    it at three sites. Bare-name resolution is what keeps the family under-firing rather
    than over-firing on ambiguous cases (see `_dry_reachable_functions`).
    """
    fn = node.func
    return fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", None)


def _dry_run_kw(call: ast.Call) -> Optional[ast.AST]:
    """The `dry_run=` keyword VALUE node of a call, or None when the keyword is absent."""
    return next((k.value for k in call.keywords if k.arg == "dry_run"), None)


def _threads_dry_run_elsewhere(tree: ast.AST) -> bool:
    """True when some `write_flat_manifest`/`emit_outcome` call threads a real dry_run.

    "Real" = the keyword is present and is not a literal `False`. A literal False is the
    `hardcoded_dry_run` gate's subject and is evidence of the OPPOSITE of intent, so it
    must not satisfy this precondition.
    """
    for n in ast.walk(tree):
        if not isinstance(n, ast.Call) or _call_name(n) not in _DRY_THREADING_WITNESS_NAMES:
            continue
        val = _dry_run_kw(n)
        if val is not None and not (isinstance(val, ast.Constant) and val.value is False):
            return True
    return False


def write_pack_dry_run_lint(path: Path) -> Optional[str]:
    """Unthreaded `write_pack(dry_run=)` check. Issue string, or None.

    Fires when ALL of:
      (1) it calls `write_pack` with either NO `dry_run=` keyword at all or a LITERAL
          `False`, at a site reachable with dry_run truthy -- interprocedurally, per
          `_dry_reachable_functions`. Checked FIRST because it is by far the rarest
          condition on this corpus (0 of 1167 drivers call `write_pack` at all), so it
          early-returns the whole scan for essentially every file.
      (2) the script has a smoke path -- an argparse `--dry-run` flag or a `dry_run`
          function parameter -- AND actually gates work on it (some `if`/`IfExp` tests
          it). Identical precondition to both siblings, for the same reason: a flag that
          gates nothing is not a smoke mode.
      (3) it ALREADY threads a real `dry_run` into `write_flat_manifest` or
          `emit_outcome`. This is what makes the gate about the PACK specifically rather
          than about dry-run awareness in general -- see the block comment above.

    Never blocking, in either mode. See the block comment for why the corpus population is
    currently 0, why that is empty rather than vacuous, and the exemption.
    """
    try:
        src = path.read_text(encoding="utf-8")
        tree = ast.parse(src)
    except (OSError, SyntaxError, UnicodeDecodeError):
        return None
    if _WRITE_PACK_DRY_RUN_EXEMPT_MARKER in src:
        return None

    unthreaded_sites = []
    for n in ast.walk(tree):
        if not isinstance(n, ast.Call) or _call_name(n) != "write_pack":
            continue
        val = _dry_run_kw(n)
        if val is None or (isinstance(val, ast.Constant) and val.value is False):
            unthreaded_sites.append(n)
    if not unthreaded_sites:
        return None

    parent: Dict[int, ast.AST] = {}
    for n in ast.walk(tree):
        for c in ast.iter_child_nodes(n):
            parent[id(c)] = n

    has_flag = False
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and _call_name(n) == "add_argument":
            for a in n.args:
                if isinstance(a, ast.Constant) and a.value in ("--dry-run", "--dry_run"):
                    has_flag = True
    has_param = any(
        isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        and any(a.arg == "dry_run"
                for a in list(n.args.args) + list(n.args.kwonlyargs))
        for n in ast.walk(tree))
    gates_work = any(isinstance(n, (ast.If, ast.IfExp)) and _mentions_dry(n.test)
                     for n in ast.walk(tree))
    if not ((has_flag or has_param) and gates_work):
        return None

    if not _threads_dry_run_elsewhere(tree):
        return None

    reachable = _dry_reachable_functions(tree, parent)

    def owner(n: ast.AST) -> Optional[str]:
        cur = parent.get(id(n))
        while cur is not None:
            if isinstance(cur, (ast.FunctionDef, ast.AsyncFunctionDef)):
                return cur.name
            cur = parent.get(id(cur))
        return None

    live = [n for n in unthreaded_sites
            if owner(n) in reachable and not _locally_dry_guarded(n, parent)]
    if not live:
        return None

    where = ", ".join(f"line {n.lineno} (in {owner(n) or '<module>'})" for n in live)
    return (
        f"threads dry_run into write_flat_manifest/emit_outcome -- so it KNOWS about smoke "
        f"containment -- but calls write_pack with no threaded dry_run= at {where}, and "
        f"that call IS reached in a smoke run. The run pack, NOT the flat manifest, is what "
        f"build_experiment_indexes._scan_runs scores, so an unflagged pack from a 1-seed "
        f"smoke weights a real claim (confirmed: MECH-245's entire negative evidence base "
        f"was two such packs). Since 2026-07-28 write_pack takes `dry_run=`, but it is "
        f"OPT-IN: unthreaded, the pack is excluded only by _load_dry_run_run_ids carrying "
        f"the flag over from the flat sibling BY RUN_ID -- a silent cross-file fallback that "
        f"had itself never globbed the per-experiment subdirectories, leaving 13 dry packs "
        f"scored against ARC-042 / MECH-070 / MECH-075 / MECH-104 / MECH-153 / MECH-155 / "
        f"MECH-156 / MECH-231. Fix by threading the script's own flag: "
        f"`write_pack(..., dry_run=bool(args.dry_run))`, which makes the pack self-identify "
        f"and needs no cross-file carry. Exempt with {_WRITE_PACK_DRY_RUN_EXEMPT_MARKER} = "
        f"\"<reason>\" when the caller already relocates, renames or deletes the pack itself."
    )


# ---- criterion unreachable under --dry-run (2026-07-28 smoke-in-autopsy audit) --------
# FOURTH member of the dry-run family, and the only one that is not about WRITING an
# artifact. The three siblings above all ask "does the smoke's output get correctly marked
# as a smoke?" -- write_flat_manifest's filename, emit_outcome's relocation, write_pack's
# flag. This one asks a different question about the same smoke: "is every criterion the
# driver reports on actually EVALUABLE at the reduced episode count?"
#
# THE DEFECT. A `--dry-run` smoke reduces episodes drastically (`p1_eps = 4 if dry_run else
# P1_TRAIN_EPISODES`). When a driver gates a DETECTOR on an absolute mid-training episode
# index, the reduced path never reaches that index, so the detector's latch stays False --
# not because the policy behaved well, but because the comparison is arithmetically
# unsatisfiable. The driver then reports that False as a finding. It is a VACUOUS NEGATIVE
# that is indistinguishable, in the manifest and on stdout, from a real negative result.
#
# CANONICAL INSTANCE, `v3_exq_543i_arc062_differential_heads_falsifier.py`:
#
#     MID_TRAINING_EP = 30                                    # module-level, line 255
#     ...
#     p1_eps = 4 if dry_run else P1_TRAIN_EPISODES            # line ~1495
#     _p1_train(agent, env, p1_eps, ...)                      # line 1544
#     ...
#     def _p1_train(..., num_episodes, ...):
#         for ep in range(num_episodes):                      # line 1033
#             if (probe["applicable"] and (ep + 1) >= MID_TRAINING_EP   # line 1286
#                     and probe["mean_tv_distance"] < INERT_GATING_THRESHOLD
#                     and not inert_gating_detected):
#                 inert_gating_detected = True
#
# `ep + 1` tops out at 4 in a smoke; the gate needs 30. So `p1_inert_gating_detected`
# (emitted at line 1312) is structurally unsettable in ANY dry run: 0/36 cells detected,
# exactly one probe per cell. The SIGN WAS INVERTED -- the smoke's gated arms had
# `mean_tv_distance` about 100x BELOW the inert threshold, so every arm would have flagged
# INERT had the gate not blocked it. That vacuous `false` was read as evidence of escape,
# and a bistability finding built on it blocked a claim disposition for two months. The
# whole 543 lineage (b..l) inherited the shape from one another -- 11 carriers, which is
# the entire corpus population of this gate.
#
# THIS IS THE MECHANICAL FORM OF A MANUAL STEP. The 2026-07-28 `/failure-autopsy` Step 2a
# guard tells an adjudicating session to read the dry-run reduction block against every
# criterion by hand. That is exactly the check below, run over the corpus instead of by a
# reader who has to remember to do it. Full write-up (section 1.1):
# REE_assembly/evidence/planning/dry_run_smoke_in_autopsy_audit_2026-07-28.md
#
# WHAT SEPARATES THE DEFECT FROM THE BENIGN CASE, and it is NOT "the branch is dead". Of
# the 409 corpus drivers whose dry loop bound this scan can resolve, 13 contain an
# episode-index conjunct that cannot be satisfied at the reduced count. Two of those are
# CORRECT and must not fire:
#   * v3_exq_430 line 229 -- `ep >= WARMUP_EPISODES` gates a SLEEP CYCLE;
#   * v3_exq_165 line 449 -- `ep > 0` gates a VALUE SHUFFLE.
# Both gate a scheduled ACTION. An action that a smoke skips is a smaller smoke, which is
# the point of a smoke; nothing false is reported. The 11 that do fire all gate a REPORTED
# DETECTOR LATCH. So the discriminator is the latch, not the dead branch: a name that is
# initialised `False`, set `True` only inside the unreachable branch, and whose value
# escapes into a dict entry or an f-string -- i.e. reaches a manifest or the run log, where
# a reader takes it for a measurement. Requiring that is what keeps this from firing on
# every reduced training schedule in the corpus.
#
# The earlier draft of this gate also accepted a detector-ish NAME (`*_detected`, `*_flag`,
# ...) as an alternative to being reported. That arm was dropped: on this corpus it selects
# exactly the same 11 files, so it bought no coverage and only added a naming convention to
# game. Escaping into output is the property that makes a vacuous false HARMFUL, so it is
# the only property tested.
#
# COST, because a corpus lint that gets slower every time the corpus grows is the creep
# CLAUDE.md tracks by name. Measured over the 1167-driver corpus (Mac, read+parse included,
# which the shared scan in tests/contracts/conftest.py removes): 10.8s, against 7.6s for
# `write_pack_dry_run_lint` and 14.4s for `hardcoded_dry_run_lint`. The first draft was
# 36.9s -- a fresh `ast.walk` per precondition plus one per candidate loop inside the bound
# resolver, i.e. O(loops x tree). It is now ONE walk that fills every scan at once, with the
# `_dry_reachable_functions` fixpoint DEFERRED until an unsatisfiable conjunct is actually
# found (13 of 409 resolvable drivers), so the common path never pays for it. Keep both
# properties if this is ever edited.
#
# ADVISORY, and deliberately biased to UNDER-fire, like all three siblings. It is a static
# scan: a bound assembled arithmetically, threaded through a dict, or resolved two call
# levels deep is invisible; only `range()` loops are considered; only `>=`/`>` (and the
# mirrored `<=`/`<`) against an int literal or a module-level int constant are compared;
# and interprocedural bound resolution goes exactly ONE level, by bare name, and gives up
# entirely unless EVERY call site resolves. A false WARN costs a reader one minute; the 11
# carriers are landed drivers whose runs are complete, so hardening would block commits on
# history. Exempt with DRY_RUN_UNREACHABLE_CRITERION_EXEMPT = "<reason>" when the criterion
# is genuinely not meant to be evaluable in a smoke AND the driver does not report its
# latch as a result.
_DRY_UNREACHABLE_CRITERION_EXEMPT_MARKER = "DRY_RUN_UNREACHABLE_CRITERION_EXEMPT"


def _int_const(node: Optional[ast.AST]) -> Optional[int]:
    """The int value of an integer literal, or None. `True`/`False` are NOT ints here."""
    if isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool):
        return node.value
    return None


def _module_int_constants(tree: ast.AST) -> Dict[str, int]:
    """Module-level `NAME = <int literal>` bindings -- the thresholds gates compare against."""
    out: Dict[str, int] = {}
    for st in getattr(tree, "body", []):
        if isinstance(st, ast.Assign) and len(st.targets) == 1 and isinstance(st.targets[0], ast.Name):
            v = _int_const(st.value)
            if v is not None:
                out[st.targets[0].id] = v
    return out


def _dry_reduced_int_bindings(tree: ast.AST) -> Dict[str, int]:
    """name -> the int it takes when dry_run is truthy, for the two reduction shapes.

    `X = 4 if dry_run else BIG` (overwhelmingly the common one) and `if dry_run: X = 4`.
    Rebindings resolve to the MAX, which is the conservative direction: a larger bound
    makes a gate MORE reachable and so makes this scan fire LESS.
    """
    out: Dict[str, int] = {}

    def note(name: str, val: int) -> None:
        out[name] = max(out[name], val) if name in out else val

    for n in ast.walk(tree):
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
            v = n.value
            if isinstance(v, ast.IfExp) and _mentions_dry(v.test):
                branch = v.orelse if _is_negated_test(v.test) else v.body
                iv = _int_const(branch)
                if iv is not None:
                    note(n.targets[0].id, iv)
        if isinstance(n, ast.If) and _mentions_dry(n.test):
            dry_body = n.orelse if _is_negated_test(n.test) else n.body
            for st in dry_body:
                if isinstance(st, ast.Assign) and len(st.targets) == 1 and isinstance(st.targets[0], ast.Name):
                    iv = _int_const(st.value)
                    if iv is not None:
                        note(st.targets[0].id, iv)
    return out


def _enclosing_function(node: ast.AST, parent: Dict[int, ast.AST]):
    cur = parent.get(id(node))
    while cur is not None:
        if isinstance(cur, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return cur
        cur = parent.get(id(cur))
    return None


def _resolve_dry_bound(expr: ast.AST, calls_by_name: Dict[str, List[ast.Call]],
                       parent: Dict[int, ast.AST],
                       dry_ints: Dict[str, int]) -> Optional[int]:
    """Upper bound on `expr` under dry_run, or None when it cannot be resolved.

    Three cases, in order: an int literal; a name bound by a dry-run reduction; a PARAMETER
    of the enclosing function, resolved one level out through that function's call sites by
    bare name. The parameter case is what reaches the canonical specimen, whose loop lives
    in `_p1_train(num_episodes)` while the reduction lives in its caller. It gives up unless
    EVERY call site resolves to an int, and then takes the MAX -- so a single unresolvable
    or full-size call site silences the file rather than producing a guess.

    `calls_by_name` is prebuilt by the caller from its single tree walk. Re-deriving it here
    would make the scan O(loops x tree), which on this corpus cost ~3x the sibling gates.
    """
    iv = _int_const(expr)
    if iv is not None:
        return iv
    if not isinstance(expr, ast.Name):
        return None
    if expr.id in dry_ints:
        return dry_ints[expr.id]

    fn = _enclosing_function(expr, parent)
    if fn is None:
        return None
    params = [a.arg for a in list(fn.args.args) + list(fn.args.kwonlyargs)]
    if expr.id not in params:
        return None
    pos = params.index(expr.id)

    vals: List[int] = []
    for c in calls_by_name.get(fn.name, ()):
        arg = c.args[pos] if pos < len(c.args) else next(
            (k.value for k in c.keywords if k.arg == expr.id), None)
        if arg is None:
            return None
        v = _int_const(arg)
        if v is None and isinstance(arg, ast.Name) and arg.id in dry_ints:
            v = dry_ints[arg.id]
        if v is None:
            return None
        vals.append(v)
    return max(vals) if vals else None


def _and_conjuncts(test: ast.AST) -> List[ast.AST]:
    """Flatten an `and` chain. Only conjuncts matter: a false OR-disjunct proves nothing."""
    if isinstance(test, ast.BoolOp) and isinstance(test.op, ast.And):
        out: List[ast.AST] = []
        for v in test.values:
            out.extend(_and_conjuncts(v))
        return out
    return [test]


def dry_run_unreachable_criterion_lint(path: Path) -> Optional[str]:
    """Criterion unsatisfiable under the driver's own --dry-run. Issue string, or None.

    Fires when ALL of:
      (1) the script has a smoke path -- an argparse `--dry-run` flag or a `dry_run`
          function parameter -- AND actually gates work on it. Identical precondition to
          all three siblings, for the same reason: a flag that gates nothing is not a
          smoke mode.
      (2) some name is bound to a reduced int under dry_run, and an episode loop
          `for v in range(<that name>)` -- directly, or one call level out via a parameter
          -- takes its bound from it.
      (3) an `if` inside that loop has an AND-conjunct comparing the loop variable
          (optionally `v + k`) against an int threshold that the reduced bound can never
          satisfy.
      (4) that branch latches a detector: `flag = True` for a name also assigned `False`
          elsewhere, whose value escapes into a dict entry or an f-string. This is the
          discriminator between a vacuous REPORTED negative and a merely-skipped scheduled
          action -- see the block comment for the two corpus cases it correctly spares.
      (5) the gate is reachable with dry_run truthy, interprocedurally per
          `_dry_reachable_functions` -- the claim in the message is that the gate IS
          evaluated in a smoke and cannot be true, so a gate the smoke never reaches at
          all is a different (and quieter) shape.

    Never blocking. See the block comment for the under-fire bias and the exemption.
    """
    try:
        src = path.read_text(encoding="utf-8")
        tree = ast.parse(src)
    except (OSError, SyntaxError, UnicodeDecodeError):
        return None
    if _DRY_UNREACHABLE_CRITERION_EXEMPT_MARKER in src:
        return None

    dry_ints = _dry_reduced_int_bindings(tree)
    if not dry_ints:
        return None

    # ONE walk, reused by every scan below. The naive form -- a fresh `ast.walk(tree)` per
    # precondition plus one per candidate loop inside `_resolve_dry_bound` -- measured 36.9s
    # over the 1167-driver corpus against ~11-14s for the sibling gates, which is exactly
    # the O(corpus) creep the 2026-07-28 scan-sharing work removed. Restructured to a single
    # walk plus a deferred fixpoint (below), it lands in the siblings' band.
    parent: Dict[int, ast.AST] = {}
    calls_by_name: Dict[str, List[ast.Call]] = {}
    has_flag = has_param = gates_work = False
    # `latches` = names somewhere assigned a literal False, i.e. flags with an off state.
    # `reported` = names whose value reaches a dict entry (a manifest) or an f-string (the
    # run log). Collected from the whole VALUE SUBTREE, not just a bare `Name` value -- the
    # specimen emits `"p1_inert_gating_detected": bool(inert_gating_detected)`, and a
    # bare-Name test misses that `bool(...)` wrapper, which would make this gate vacuous
    # against the very file it was written for.
    latches: Set[str] = set()
    reported: Set[str] = set()
    loops: List[ast.For] = []

    def absorb_names(node: ast.AST) -> None:
        for s in ast.walk(node):
            if isinstance(s, ast.Name):
                reported.add(s.id)

    for n in ast.walk(tree):
        for c in ast.iter_child_nodes(n):
            parent[id(c)] = n
        if isinstance(n, ast.Call):
            nm = _call_name(n)
            if nm is not None:
                calls_by_name.setdefault(nm, []).append(n)
            if nm == "add_argument":
                for a in n.args:
                    if isinstance(a, ast.Constant) and a.value in ("--dry-run", "--dry_run"):
                        has_flag = True
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if any(a.arg == "dry_run" for a in list(n.args.args) + list(n.args.kwonlyargs)):
                has_param = True
        elif isinstance(n, ast.Assign):
            if isinstance(n.value, ast.Constant) and n.value.value is False:
                for t in n.targets:
                    if isinstance(t, ast.Name):
                        latches.add(t.id)
        elif isinstance(n, ast.Dict):
            for v in n.values:
                if v is not None:          # None is the `**spread` entry's key slot
                    absorb_names(v)
        elif isinstance(n, ast.JoinedStr):
            absorb_names(n)
        elif isinstance(n, ast.For) and isinstance(n.target, ast.Name):
            it = n.iter
            if (isinstance(it, ast.Call) and _call_name(it) == "range"
                    and 1 <= len(it.args) <= 2):
                loops.append(n)
        if isinstance(n, (ast.If, ast.IfExp)) and _mentions_dry(n.test):
            gates_work = True

    if not ((has_flag or has_param) and gates_work):
        return None
    if not (loops and latches and reported):
        return None

    mod_consts = _module_int_constants(tree)

    # The reachability fixpoint is the expensive step and is DEFERRED behind the arithmetic:
    # an unsatisfiable episode-index conjunct is rare (13 of the 409 drivers whose bound
    # resolves), so paying for the fixpoint only once one is found keeps the common path
    # cheap. Computed lazily on first use rather than up front.
    reachable_cache: List[Set[Optional[str]]] = []

    def is_reachable(n: ast.AST) -> bool:
        if not reachable_cache:
            reachable_cache.append(_dry_reachable_functions(tree, parent))
        fn = _enclosing_function(n, parent)
        return (fn.name if fn is not None else None) in reachable_cache[0]

    findings: List[Tuple[int, str, str, int, int]] = []
    for loop in loops:
        bound = _resolve_dry_bound(loop.iter.args[-1], calls_by_name, parent, dry_ints)
        if bound is None:
            continue
        var, max_index = loop.target.id, bound - 1

        def lhs_max(e: ast.AST) -> Optional[int]:
            """Largest value the compared expression can take at the reduced bound."""
            if isinstance(e, ast.Name) and e.id == var:
                return max_index
            if (isinstance(e, ast.BinOp) and isinstance(e.op, ast.Add)
                    and isinstance(e.left, ast.Name) and e.left.id == var):
                k = _int_const(e.right)
                return None if k is None else max_index + k
            return None

        def threshold(e: ast.AST) -> Optional[int]:
            v = _int_const(e)
            if v is not None:
                return v
            return mod_consts.get(e.id) if isinstance(e, ast.Name) else None

        for node in ast.walk(loop):
            if not isinstance(node, ast.If):
                continue
            for cj in _and_conjuncts(node.test):
                if not (isinstance(cj, ast.Compare) and len(cj.ops) == 1):
                    continue
                op = cj.ops[0]
                lo, thr, strict = lhs_max(cj.left), threshold(cj.comparators[0]), None
                if lo is not None and thr is not None and isinstance(op, (ast.GtE, ast.Gt)):
                    strict = isinstance(op, ast.Gt)
                else:
                    # mirrored form: `THRESHOLD <= ep + 1`
                    lo, thr = lhs_max(cj.comparators[0]), threshold(cj.left)
                    if lo is not None and thr is not None and isinstance(op, (ast.LtE, ast.Lt)):
                        strict = isinstance(op, ast.Lt)
                if strict is None:
                    continue
                if not ((strict and lo <= thr) or (not strict and lo < thr)):
                    continue
                # Reachability LAST: it is the expensive check and only an unsatisfiable
                # conjunct earns it (see `is_reachable`'s deferred fixpoint above).
                if not is_reachable(node) or _locally_dry_guarded(node, parent):
                    continue
                for st in ast.walk(node):
                    if not (isinstance(st, ast.Assign) and isinstance(st.value, ast.Constant)
                            and st.value.value is True):
                        continue
                    for tg in st.targets:
                        if (isinstance(tg, ast.Name) and tg.id in latches
                                and tg.id in reported):
                            try:
                                text = ast.unparse(cj)
                            except Exception:
                                text = f"<{var} vs {thr}>"
                            findings.append((node.lineno, text, tg.id, lo, thr))
    if not findings:
        return None

    seen: Set[Tuple[int, str]] = set()
    parts = []
    for lineno, text, flag, lo, thr in findings:
        key = (lineno, flag)
        if key in seen:
            continue
        seen.add(key)
        parts.append(f"line {lineno}: `{text}` tops out at {lo} < {thr}, latching `{flag}`")
    where = "; ".join(parts)
    return (
        f"accepts --dry-run and reduces its episode counts under it, but gates a REPORTED "
        f"detector on an absolute episode index the reduced run can never reach -- {where}. "
        f"The latch is therefore hardcoded false in any smoke, WHATEVER the policy did, and "
        f"the driver reports that false into its manifest/log where it is indistinguishable "
        f"from a measured negative. This is not hypothetical: in v3_exq_543i the sign was "
        f"INVERTED -- the smoke's gated arms sat about 100x BELOW the inert threshold, so "
        f"every arm would have flagged INERT had the gate not blocked it, and the vacuous "
        f"false was read as evidence of escape, blocking a claim disposition for two "
        f"months. Fix by scaling the gate with the run: derive the threshold from the "
        f"actual episode count (e.g. a fraction of it) rather than a module-level absolute, "
        f"or exclude the criterion from the smoke explicitly instead of letting it report a "
        f"structural false. Exempt with {_DRY_UNREACHABLE_CRITERION_EXEMPT_MARKER} = "
        f"\"<reason>\" when the criterion is genuinely not meant to be evaluable in a smoke "
        f"AND its latch is not reported as a result. Full write-up: "
        f"REE_assembly/evidence/planning/dry_run_smoke_in_autopsy_audit_2026-07-28.md "
        f"section 1.1. Do NOT retro-edit a LANDED driver whose run is complete."
    )


# ---- config_slice under-declaration (V3-EXQ-798) -------------------------------------
# arm_reuse_fingerprint_plan.md section 7b: a `config_slice` that UNDER-approximates --
# omits a parameter the cell's RECORDED READOUTS depend on -- is a false-cache-HIT bug.
# The governing asymmetry is that a false MISS only wastes compute while a false HIT
# corrupts a scientific conclusion, so the slice is meant to be OVER-inclusive.
#
# The interaction that makes this sharp: `include_driver_script_in_hash=False` is
# MANDATORY for a cross-driver-reusable mint (CLAUDE.md, "Saving a baseline for reuse"),
# and it is exactly that flag which drops the driver -- and therefore every module-level
# constant defined in it -- out of the substrate hash. So the more reusable a mint is
# made, the more load the config_slice has to carry. With the driver IN the hash a
# module-level constant is already bound by content, so only the flag-False set is gated.
_CONFIG_SLICE_EXEMPT_MARKER = "CONFIG_SLICE_DECLARATION_EXEMPT"
_FP_CALL_NAMES = ("arm_cell", "compute_arm_fingerprint")
_SLICE_PASSTHROUGH_CALLS = ("dict", "copy", "deepcopy", "OrderedDict")
# A slice built by a helper imported from here is declared in ANOTHER FILE. The resolver
# below parses that file and absorbs the helper's returned dict, so those scripts ARE
# assessed; the marker now selects the band to RESOLVE rather than the band to skip. The
# skip survives only as the fallback for a helper that cannot be resolved -- see the lint
# docstring's "cross-module slice" block.
_BASELINE_HELPER_MODULE_MARKER = "_lib.baselines"

# Parsed `experiments/_lib/baselines/*.py` modules, keyed by (path, mtime_ns, size) so a
# rewritten fixture is never served stale. Only ~5 drivers in the corpus reach this, and
# the whole baselines package is 19 files, so the cache is bounded and tiny. These trees
# are READ-ONLY here for the same reason the driver tree is (see the parent-map comment
# in the lint): nothing may mutate an AST the corpus-scan cache shares.
#
# Interaction with that corpus-scan cache (`tests/contracts/conftest.py`), which is
# correct but worth stating: it is keyed on source TEXT and holds exactly ONE entry, so a
# baseline-module parse here can never be served the wrong tree, but it does EVICT the
# driver's entry mid-file -- costing the next path lint on that driver one re-parse. This
# cache is what bounds that: after the first driver, each baseline module is served from
# here and `ast.parse` is not called at all, so the ceiling is one eviction per distinct
# baseline module per session (3 in the corpus today). Measured: corpus scan 7.90s before
# / 7.84s after, i.e. no measurable change.
_BASELINE_MODULE_CACHE: Dict[Tuple[str, int, int], Optional[ast.Module]] = {}


def _baselines_module_path(driver: Path, tail: str) -> Optional[Path]:
    """Locate `_lib/baselines/<tail>.py` for a driver importing it, or None.

    Both import spellings in the corpus land on the same file:
    `from _lib.baselines.X import ...` (sys.path-rooted at experiments/) and
    `from experiments._lib.baselines.X import ...` (repo-rooted); `tail` is whatever
    follows the marker in either, so only the search root differs.
    """
    try:
        rel = Path(*tail.split(".")).with_suffix(".py")
    except (TypeError, ValueError):
        return None
    for root in (driver.parent / "_lib" / "baselines",
                 driver.parent.parent / "experiments" / "_lib" / "baselines"):
        cand = root / rel
        if cand.is_file():
            return cand
    return None


def _parse_baseline_module(path: Path) -> Optional[ast.Module]:
    """Parse a baseline module, cached. None on any failure -- the caller then skips."""
    try:
        st = path.stat()
    except OSError:
        return None
    key = (str(path), st.st_mtime_ns, st.st_size)
    if key not in _BASELINE_MODULE_CACHE:
        try:
            _BASELINE_MODULE_CACHE[key] = ast.parse(
                path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, UnicodeDecodeError, SyntaxError):
            _BASELINE_MODULE_CACHE[key] = None
    return _BASELINE_MODULE_CACHE[key]


def _is_mapping_expr(node: ast.AST) -> bool:
    """A dict LITERAL or a `dict(...)`/`OrderedDict(...)`-style construction.

    Both spellings declare keys, and the canonical baseline modules use the second:
    `MATCHED_ENVELOPE: Dict[str, Any] = dict(settling_rounds=3, ...)`. Treating only
    `ast.Dict` as a mapping is why resolving the cross-module helper initially bought
    nothing on V3-EXQ-700c -- the splice `dict(MATCHED_ENVELOPE)` was followed to a
    name bound to an `ast.Call`, which the recursion then declined to enter.
    """
    return isinstance(node, ast.Dict) or (
        isinstance(node, ast.Call) and _call_name(node) in _SLICE_PASSTHROUGH_CALLS)


def _module_assigns_and_funcs(
        tree: ast.AST) -> Tuple[Dict[str, List[ast.AST]], Dict[str, List[ast.AST]]]:
    """`name -> [bound value expr]` and `name -> [def]`, the two maps `_absorb` walks.

    ANNOTATED assignments count. The canonical baseline modules annotate exactly the
    dicts that matter (`MATCHED_ENVELOPE: Dict[str, Any] = dict(...)`), so collecting
    only `ast.Assign` leaves the splice unresolvable and the cross-module resolution
    buys nothing. This only ever ADDS declarations, so it can only reduce fires.
    """
    assigns: Dict[str, List[ast.AST]] = {}
    funcs: Dict[str, List[ast.AST]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name):
                    assigns.setdefault(tgt.id, []).append(node.value)
        elif (isinstance(node, ast.AnnAssign) and node.value is not None
                and isinstance(node.target, ast.Name)):
            assigns.setdefault(node.target.id, []).append(node.value)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            funcs.setdefault(node.name, []).append(node)
    return assigns, funcs


def _call_name(node: ast.Call) -> Optional[str]:
    f = node.func
    if isinstance(f, ast.Name):
        return f.id
    if isinstance(f, ast.Attribute):
        return f.attr
    return None


def _is_numeric_literal(node: ast.AST) -> bool:
    """A number, or a non-empty tuple/list of numbers. bool is excluded on purpose.

    Deliberately narrow. A binning scheme (SSL_BIN_EDGES = (2, 5, 12)), a learning
    rate, a budget, a gain -- the parameters that silently change what a readout MEANS
    -- are numeric. Strings are overwhelmingly labels/paths/run-ids here, and gating
    them would drown the signal.
    """
    if isinstance(node, ast.Constant):
        return isinstance(node.value, (int, float)) and not isinstance(node.value, bool)
    if isinstance(node, (ast.Tuple, ast.List)):
        return bool(node.elts) and all(
            isinstance(e, ast.Constant) and isinstance(e.value, (int, float))
            and not isinstance(e.value, bool) for e in node.elts)
    return False


def config_slice_under_declaration_lint(path: Path) -> Optional[str]:
    """Under-declared arm-reuse `config_slice` check. Return an issue string, or None.

    Gates only the CROSS-DRIVER set: a script that makes at least one
    `arm_cell(...)` / `compute_arm_fingerprint(...)` call passing
    `include_driver_script_in_hash=False`. That flag removes the driver's own content
    from the substrate hash, so every module-level constant the cell reads becomes
    invisible to the fingerprint unless the `config_slice` names it. A consumer with a
    different value for that constant then HITs the cell and silently reads a readout
    computed under a different scheme.

    Confirmed instance (V3-EXQ-798, landed 2026-07-23): the cells record
    `learn_pe_by_ssl_bin` / `learn_ssl_bin_counts` / `learn_decay_frac`, all computed
    against the module-level `SSL_BIN_EDGES = (2, 5, 12)`, which is absent from the
    slice. The successor 798a declares it (`"binning"` / `"n_bins"` /
    `"ssl_bin_edges"`) and is the reference for the correct shape.

    Fires on a module-level UPPER_SNAKE constant bound to a NUMERIC literal (or a
    tuple/list of numerics -- the bin-edges shape) that is read from the CELL'S OWN CALL
    GRAPH and is not named in the resolved config_slice, by value-expression name or by a
    key whose name matches it. Scoping to the cell call graph is what separates a
    readout-affecting parameter from an adjudication threshold applied downstream in
    `evaluate(rows)` -- without it the check over-fires ~3x on THRESH_*/FLOOR_* criterion
    constants that no cell ever reads.

    LOCATING THE CELL BODY -- TWO FORMS. The `with arm_cell(...) as cell:` form has a
    lexical body, and that body (plus its transitive module-level callees) is the scope.
    The DIRECT `compute_arm_fingerprint(...)` form has no cell body at all: the call only
    computes a hash, and the arm is run by a sibling statement. For those, the scope is
    the NEAREST ENCLOSING LOOP body of the call (plus transitive callees), falling back to
    the enclosing function when there is no loop -- because the shape is invariably

        for arm in ARMS:                     # <- the scope
            row = _run_seed_arm(arm, seed)   # the cell body, a sibling statement
            row["arm_fingerprint"] = compute_arm_fingerprint(config_slice=..., ...)

    so the loop body is the closest static analogue of a `with` body. Both root sets are
    used together, so a script mixing the forms is scoped for each.

    WHY THE LOOP AND NOT THE ENCLOSING FUNCTION (measured 2026-07-28, do not "simplify"
    this to the enclosing function). In the direct-call corpus the enclosing function is
    `run_experiment` in 13 of 15 cases -- the whole-experiment driver, which calls
    `evaluate(rows)` too -- so enclosing-function scope degenerates into very nearly the
    whole-module scan this gate exists to avoid. Calibrated against the with-form scripts,
    where the `with` body is ground truth: enclosing-function scope reproduces the correct
    missing set in only 28 of 60, adds 271 spurious constants, and is byte-identical to a
    whole-module scan in 18 of 60. Loop scope reproduces it in 40 of 60, adds 79, and
    degenerates in 7. On the direct-call scripts the difference is the adjudication block:
    loop scope excludes MIN_SEEDS_FOR_PASS / DIVERGENT_PASS_FRACTION / ABLATION_MARGIN /
    CONVERSION_MARGIN, which enclosing-function scope pulls in. Loop scope changes the
    result of ZERO with-form scripts, so this extension is strictly additive.

    CROSS-MODULE SLICE -- RESOLVED, WITH THE SKIP KEPT AS THE FALLBACK. When a
    flag-False call's `config_slice` is built by a helper imported from
    `experiments/_lib/baselines/` (`arm_config_slice(...)` /
    `off_path_config_slice(...)`), the declaration lives in ANOTHER FILE. Until
    2026-07-28 that was a blanket SKIP, because firing blind would have been a
    near-total false positive landing on exactly the scripts following the
    canonical-baseline pattern CLAUDE.md mandates for a reusable mint -- the
    best-behaved population, not the worst. Measured on V3-EXQ-700c at the time: 21 of
    the 27 constants a fire would have named were declared in that module's
    `MATCHED_ENVELOPE`.

    The resolver now parses that module and absorbs the helper's returned dict --
    including a same-module helper it tail-calls (`off_path_config_slice` ->
    `arm_config_slice`) and any module-level dict it splices in (`dict(MATCHED_ENVELOPE)`,
    `dict(ENV_KWARGS)`), whose KEYS are what actually bind the driver's constants. All
    five resolve, so the band holds no unassessed script: 700c 55 -> 31 surviving names,
    700d 56 -> 32, 833 1, and 685 / 700c_mint verifiably clean (one declared constant,
    and no module-level numeric constants at all). Calibrated before shipping -- of
    700c's 31, ZERO appear as a key or kwarg anywhere in the baseline module, so the
    resolution is complete with respect to it rather than partial.

    The skip survives ONLY as the fallback: if the module file cannot be located, parsed,
    or the named function found with a returned value, the check returns None and says
    nothing, exactly as before.

    TWO SPELLINGS DECIDE WHETHER THIS IS REAL OR NOMINAL. Parsing the module bought
    nothing at all until both were handled, because the canonical baseline modules write
    the envelope as an ANNOTATED assignment bound to a `dict(...)` CALL:
    `MATCHED_ENVELOPE: Dict[str, Any] = dict(settling_rounds=3, ...)`. Treating a mapping
    as `ast.Dict`-only, or building the name->value map from `ast.Assign` only, each
    leaves 700c at 55 names -- the module is parsed, the helper is found, and no result
    changes. Both are fixed here (`_is_mapping_expr`, `_module_assigns_and_funcs`), and
    both generalise to the single-file path: on the with-form corpus the annotation half
    alone cleared V3-EXQ-114a / 120a / 266b as pre-existing false positives (all three
    declare in an annotated `full_config: Dict[str, Any] = {...}`) and shrank five more,
    and the mapping half removed REINFORCE_BATCH_SIZE from all 11 direct-call carriers.

    Deliberately narrow in two ways. It follows the CALL form only -- a bare imported
    name used as the slice is not resolved (and, as before, does not skip either). And
    it does not chase imports out of the baseline module: a helper whose return splices
    in something imported from a THIRD file resolves partially, which is the same
    over-fire direction the gate already lives with elsewhere and is why the gate stays
    WARN-only.

    Known limits, both directions, same class as the other static lints here:
      - UNDER-fires when the value is not a module-level literal (assembled at runtime,
        imported from a _lib module other than a resolved baselines helper, read from
        argv/env), when the cell body calls a helper through an alias/partial the
        name-based call graph cannot follow, on an unresolvable cross-module slice as
        described above, and -- for the direct-call form -- when the arm is run OUTSIDE
        the loop holding the fingerprint call.
      - OVER-fires when the constant is genuinely bound by something else already in the
        slice (a derived count beside a declared edge list), when a threshold is read
        lexically inside the scope for an early-exit/progress print, and on the
        name heuristic's near-misses (declared key `ssl_bin_edges` does not
        substring-match the constant `N_SSL_BINS`).
    The remedy is cheap in both directions: add the key to the slice, or add the marker.

    Opt-out: CONFIG_SLICE_DECLARATION_EXEMPT = "<reason>".

    WARN-only in BOTH modes -- it never hardens under --paths. The fire set is
    best-effort (see the limits above), a false HIT needs a CONSUMER to exist before it
    can corrupt anything, and the landed carriers' runs are complete, so hardening would
    block commits on history rather than on the authoring path.
    """
    try:
        src = path.read_text(encoding="utf-8")
        tree = ast.parse(src, filename=str(path))
    except (OSError, UnicodeDecodeError, SyntaxError):
        return None  # check_script already reports unreadable / syntax errors

    if _CONFIG_SLICE_EXEMPT_MARKER in src:
        return None

    def _is_cross_driver(call: ast.Call) -> bool:
        return any(kw.arg == "include_driver_script_in_hash"
                   and isinstance(kw.value, ast.Constant) and kw.value.value is False
                   for kw in call.keywords)

    # -- 1a. every cross-driver fingerprint call, in either form
    fp_calls: List[ast.Call] = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call) and _call_name(node) in _FP_CALL_NAMES
        and _is_cross_driver(node)]
    if not fp_calls:
        return None  # driver is in the hash (or no cell) -- constants already bound

    # -- 1b. cell bodies of the `with arm_cell(..., include_driver=False)` form
    cell_withs: List[Tuple[ast.AST, ast.Call]] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.With, ast.AsyncWith)):
            continue
        for item in node.items:
            call = item.context_expr
            if not isinstance(call, ast.Call) or _call_name(call) not in _FP_CALL_NAMES:
                continue
            if _is_cross_driver(call):
                cell_withs.append((node, call))
    with_ctx_ids = {id(call) for _, call in cell_withs}

    # -- 1c. parent map + module-level functions (both needed below)
    #
    # An EXTERNAL id-keyed dict, deliberately -- NOT `node.parent = ...`. This lint runs
    # against a tree shared across all the path lints by the corpus-scan parse cache
    # (`tests/contracts/conftest.py`), whose soundness rests on no consumer mutating the
    # AST it is handed; that module names parent-pointer annotation as one of the
    # specific things that would break it. The dict is rebuilt per call, which is cheap
    # next to the parse the cache already saved.
    parent: Dict[int, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parent[id(child)] = node

    funcs: Dict[str, List[ast.AST]] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            funcs.setdefault(node.name, []).append(node)

    # -- 1d. helpers imported from `_lib/baselines/` -- local alias -> (module tail,
    #        name in that module). These build the slice in ANOTHER FILE; step 2 resolves
    #        them, and only an UNRESOLVABLE one falls back to the documented skip.
    baseline_helpers: Dict[str, Tuple[str, str]] = {}
    for node in ast.walk(tree):
        if not (isinstance(node, ast.ImportFrom) and node.module
                and _BASELINE_HELPER_MODULE_MARKER in node.module):
            continue
        parts = node.module.split(_BASELINE_HELPER_MODULE_MARKER + ".", 1)
        tail = parts[1] if len(parts) == 2 else ""
        for alias in node.names:
            baseline_helpers[alias.asname or alias.name] = (tail, alias.name)

    # -- 2. resolve the declared config_slice into keys + value-expression names
    assigns = _module_assigns_and_funcs(tree)[0]

    declared_keys: Set[str] = set()
    declared_names: Set[str] = set()
    # Set when a config_slice IS built by a baselines helper we could not resolve. The
    # gate then says nothing rather than firing on a declaration it cannot see.
    unresolved_cross_module: List[str] = []

    _DRIVER_CTX = "<driver>"
    # ctx key -> (assigns, funcs) for a resolved baselines module. `seen` is keyed by
    # (ctx, name) so a helper named the same in two modules does not self-block.
    module_ctx: Dict[str, Tuple[Dict[str, List[ast.AST]], Dict[str, List[ast.AST]]]] = {
        _DRIVER_CTX: (assigns, funcs)}

    def _resolve_baseline_ctx(tail: str) -> Optional[str]:
        """Parse `_lib/baselines/<tail>.py` into a ctx key, or None if unresolvable."""
        if not tail:
            return None
        mod_path = _baselines_module_path(path, tail)
        if mod_path is None:
            return None
        key = str(mod_path)
        if key not in module_ctx:
            mod_tree = _parse_baseline_module(mod_path)
            if mod_tree is None:
                return None
            module_ctx[key] = _module_assigns_and_funcs(mod_tree)
        return key

    def _absorb(expr: Optional[ast.AST], depth: int = 0, seen: Optional[frozenset] = None,
                ctx: str = _DRIVER_CTX) -> None:
        seen = seen or frozenset()
        if expr is None or depth > 8:
            return
        ctx_assigns, ctx_funcs = module_ctx[ctx]
        if isinstance(expr, ast.Name):
            declared_names.add(expr.id)
            if (ctx, expr.id) in seen:
                return
            for val in ctx_assigns.get(expr.id, []):
                _absorb(val, depth + 1, seen | {(ctx, expr.id)}, ctx)
            return
        if isinstance(expr, ast.Call):
            callee = _call_name(expr)
            if callee in _SLICE_PASSTHROUGH_CALLS:
                for arg in expr.args:
                    _absorb(arg, depth + 1, seen, ctx)
                for kw in expr.keywords:
                    if kw.arg:
                        declared_keys.add(kw.arg)
                    _absorb(kw.value, depth + 1, seen, ctx)
                return
            # `config_slice=arm_config_slice(arm, ...)` imported from `_lib/baselines/` --
            # the canonical-baseline shape CLAUDE.md mandates for a reusable mint. The
            # declaration is in that module, so resolve it there. Only reachable from the
            # driver ctx: this does not chase imports out of a baseline module.
            # Checked AFTER the local-function branch below by the `callee not in
            # ctx_funcs` guard: a local `def` of the same name shadows the import in
            # Python, so it must shadow it here too.
            if (ctx == _DRIVER_CTX and callee in baseline_helpers
                    and callee not in ctx_funcs):
                tail, orig = baseline_helpers[callee]
                sub = _resolve_baseline_ctx(tail)
                if sub is None or orig not in module_ctx.get(sub, ({}, {}))[1]:
                    unresolved_cross_module.append(callee)
                    return
                absorbed = False
                for defn in module_ctx[sub][1][orig]:
                    for node in ast.walk(defn):
                        if isinstance(node, ast.Return) and node.value is not None:
                            absorbed = True
                            _absorb(node.value, depth + 1, seen | {(sub, orig)}, sub)
                if not absorbed:
                    unresolved_cross_module.append(callee)
                return
            # `config_slice=_arm_config_slice(arm, ...)` -- a LOCAL helper returning the
            # slice dict. Absorb what it returns. Without this the slice reads as empty
            # and the fire is dominated by constants that ARE declared, just one call
            # away: measured 2026-07-28, following local returns removes false positives
            # from 13 with-form scripts (V3-EXQ-793 41 names -> 15, 794 11 -> 2, 751
            # 3 -> 0) and is what keeps the direct-call form's new coverage honest, since
            # 10 of those 15 scripts declare their slice through exactly this shape.
            # Resolved in the CURRENT ctx, so a baseline helper's tail call to a sibling
            # in its own module (`off_path_config_slice` -> `arm_config_slice`) lands
            # here with that module's funcs, not the driver's.
            if callee in ctx_funcs and (ctx, callee) not in seen:
                for defn in ctx_funcs[callee]:
                    for sub in ast.walk(defn):
                        if isinstance(sub, ast.Return) and sub.value is not None:
                            _absorb(sub.value, depth + 1, seen | {(ctx, callee)}, ctx)
                return
        if isinstance(expr, ast.Dict):
            for key, val in zip(expr.keys, expr.values):
                if key is None:          # {**other}
                    _absorb(val, depth + 1, seen, ctx)
                    continue
                if isinstance(key, ast.Constant) and isinstance(key.value, str):
                    declared_keys.add(key.value)
                for sub in ast.walk(val):
                    if isinstance(sub, ast.Name):
                        declared_names.add(sub.id)
                        # `"matched_envelope": dict(MATCHED_ENVELOPE)` -- the KEYS of a
                        # spliced module-level dict are what bind the driver's constants
                        # (SETTLING_ROUNDS <- "settling_rounds"), so a value that merely
                        # NAMES the envelope declares nothing on its own. Same recursion
                        # the generic branch below already does; without it, resolving a
                        # cross-module helper buys almost nothing, since the canonical
                        # baseline shape is exactly one such splice per group.
                        for nested in ([] if (ctx, sub.id) in seen
                                       else ctx_assigns.get(sub.id, [])):
                            if _is_mapping_expr(nested):
                                _absorb(nested, depth + 1, seen | {(ctx, sub.id)}, ctx)
                    elif isinstance(sub, ast.Attribute):
                        declared_names.add(sub.attr)
            return
        for sub in ast.walk(expr):
            if isinstance(sub, ast.Name):
                declared_names.add(sub.id)
                for val in ctx_assigns.get(sub.id, []):
                    if isinstance(val, ast.Dict):
                        _absorb(val, depth + 1, seen, ctx)
            elif isinstance(sub, ast.Attribute):
                declared_names.add(sub.attr)

    for call in fp_calls:
        got_kw = False
        for kw in call.keywords:
            if kw.arg == "config_slice":
                _absorb(kw.value)
                got_kw = True
        if not got_kw and len(call.args) >= 2:
            _absorb(call.args[1])       # positional (seed, config_slice, ...)
    # `slice_cfg.update({...})` after the dict() copy is the dominant per-arm idiom.
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "update"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id in declared_names):
            for arg in node.args:
                _absorb(arg)

    # -- 2b. the fallback: a baselines helper we could NOT resolve leaves the slice
    #        partly invisible, which is the pre-2026-07-28 situation for the whole band.
    #        Say nothing rather than fire on a declaration we cannot see.
    if unresolved_cross_module:
        return None

    # -- 3. module-level numeric constants
    consts: Set[str] = set()
    for node in tree.body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)):
            name = node.targets[0].id
            if len(name) > 2 and name.isupper() and _is_numeric_literal(node.value):
                consts.add(name)
    if not consts:
        return None

    # -- 4. the cell's own call graph, rooted per form (see docstring):
    #       `with` -> the cell body; direct call -> its nearest enclosing loop.
    def _direct_call_scope(call: ast.Call) -> Optional[ast.AST]:
        node: Optional[ast.AST] = parent.get(id(call))
        while node is not None:
            if isinstance(node, (ast.For, ast.AsyncFor, ast.While)):
                return node
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                return node     # fallback: no loop between the call and its function
            node = parent.get(id(node))
        return None

    roots: List[ast.AST] = [with_node for with_node, _ in cell_withs]
    for call in fp_calls:
        if id(call) in with_ctx_ids:
            continue            # already covered by its own `with` body
        scope = _direct_call_scope(call)
        if scope is not None:
            roots.append(scope)

    reachable: Set[str] = set()
    frontier: List[str] = []
    for root in roots:
        for stmt in getattr(root, "body", []):
            for sub in ast.walk(stmt):
                if isinstance(sub, ast.Call) and _call_name(sub) in funcs:
                    frontier.append(_call_name(sub))
    while frontier:
        name = frontier.pop()
        if name in reachable:
            continue
        reachable.add(name)
        for defn in funcs.get(name, []):
            for sub in ast.walk(defn):
                if isinstance(sub, ast.Call):
                    callee = _call_name(sub)
                    if callee in funcs and callee not in reachable:
                        frontier.append(callee)

    scopes: List[ast.AST] = list(roots)
    for name in reachable:
        scopes.extend(funcs.get(name, []))

    read_by_cell: Set[str] = set()
    for scope in scopes:
        for sub in ast.walk(scope):
            if (isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Load)
                    and sub.id in consts):
                read_by_cell.add(sub.id)

    # -- 5. undeclared?
    missing: List[str] = []
    for name in sorted(read_by_cell):
        if name in declared_names:
            continue
        low = name.lower()
        if any(low == k.lower() or low in k.lower() or k.lower() in low
               for k in declared_keys if len(k) > 2):
            continue
        missing.append(name)
    if not missing:
        return None

    shown = ", ".join(missing[:6]) + (f", +{len(missing) - 6} more" if len(missing) > 6 else "")
    return (
        f"emits a CROSS-DRIVER-reusable arm fingerprint "
        f"(include_driver_script_in_hash=False, so the driver's own content is NOT in "
        f"the substrate hash) but its config_slice omits {len(missing)} module-level "
        f"numeric constant(s) the cell's call graph reads: {shown}. Under-approximating "
        f"the slice is a false-cache-HIT bug (arm_reuse_fingerprint_plan.md 7b): a "
        f"consumer with a different value HITs these cells and silently reads readouts "
        f"computed under a different scheme -- a false MISS only wastes compute, a false "
        f"HIT corrupts a conclusion. Confirmed instance V3-EXQ-798 (SSL_BIN_EDGES absent "
        f"while the cells record learn_pe_by_ssl_bin); 798a is the reference fix -- it "
        f"declares \"binning\" / \"n_bins\" / \"ssl_bin_edges\" in its cell_config. Fix by "
        f"adding each readout-affecting constant to the config_slice dict. Exempt with "
        f"{_CONFIG_SLICE_EXEMPT_MARKER} = \"<reason>\" when the constant is genuinely "
        f"bound by something already in the slice."
    )


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


_AO_SELECTION_EXEMPT_MARKER = "ACTION_OBJECT_SELECTION_EXEMPT"


def _ao_decoder_call(node: ast.AST) -> bool:
    """True for a call whose callee is `<...>.action_object_decoder` or the
    module's `_decode_action_objects` helper."""
    if not isinstance(node, ast.Call):
        return False
    fn = node.func
    if isinstance(fn, ast.Attribute):
        return fn.attr in ("action_object_decoder", "_decode_action_objects")
    if isinstance(fn, ast.Name):
        return fn.id in ("action_object_decoder", "_decode_action_objects")
    return False


def action_object_selection_lint(path: Path) -> Optional[str]:
    """Action-object round-trip-as-action-source check. Issue string, or None.

    Fires when a driver takes an ARGMAX over the action-object decoder -- i.e.
    recovers "the action this candidate takes" via
    `argmax(action_object_decoder(traj.get_action_object_sequence()[:, 0, :]))`.

    WHY THIS IS A DEFECT AND NOT A STYLE PREFERENCE. That round trip
    (a -> E2.action_object(a) -> decoder -> a_hat) is not invertible on this
    substrate, so the argmax is a CONSTANT: an action stream selected this way is
    invariant under every manipulation of the candidate set or its scores, and
    the experiment is an arithmetically forced no-op that still produces
    plausible-looking numbers. Confirmed 2026-07-22, independently reproduced:
    action-class-scaffold candidates constructed with 5 distinct one-hot first
    actions all re-decode to the SAME class, on an untrained module and after 40
    warmup episodes alike. NEITHER component is individually degenerate -- the
    COMPOSITION is: both are untrained, and the action-object distribution is a
    small ball far from the decoder's decision boundaries, so the argmax pins to
    the decoder's own bias-argmax class. (The embedding is NOT action-invariant:
    a linear probe recovers the action class from it at 100%. What it lacks is
    STATE dependence.) See `HippocampalModule.candidate_first_action_class` for
    the measurements.

    It is NOT repaired by `use_support_preserving_cem` or
    `use_action_class_scaffold_candidates` -- both act on `Trajectory.actions`
    and neither touches the round trip.

    FIX: select with `agent.select_action(candidates, ticks)` (E3's J(zeta),
    returns the action directly), or read the candidate's real action with
    `HippocampalModule.candidate_first_action_class(traj)`.

    Directly observed consequence: V3-EXQ-801's A2_FULL and A3_NOISE arms came
    out BIT-IDENTICAL on every recorded field under the round-trip rule, and
    separated once selection moved to the E3 path. Most plausible mechanism for
    EXQ-196's harm_advantage_mean = EXACTLY 0.0 on all three seeds at
    e2_world_r2 0.766 (ARC-018), deferred by governance as non_contributory.

    NOT flagged: passing the decoder's CONTINUOUS output into an E2 rollout (the
    sanctioned CEM proposal use -- the rollout consumes the real-valued vector,
    so candidates differ even when their argmaxes coincide), or collecting
    decoder outputs for DIAGNOSTICS. Only the argmax-for-action pattern fires.

    Static AST scan with the same limitation class as the other lints here: it
    follows one level of local aliasing (`logits = ...decoder(x)` then
    `argmax(logits)`) and will miss an argmax assembled at runtime or routed
    through a helper in another module. WARN-only in BOTH modes -- it never
    hardens under `--paths`, because a driver may legitimately argmax the
    decoder to REPORT the collapse (that is what a diagnostic probe of this very
    defect looks like), and the scan cannot tell reporting from selecting.

    Opt-out: ACTION_OBJECT_SELECTION_EXEMPT = "<reason>".
    """
    try:
        src = path.read_text(encoding="utf-8")
        tree = ast.parse(src, filename=str(path))
    except (OSError, UnicodeDecodeError, SyntaxError):
        return None  # check_script already reports unreadable / syntax errors

    if _AO_SELECTION_EXEMPT_MARKER in src:
        return None

    # One level of local aliasing: names bound directly to a decoder call.
    aliases: set = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and _ao_decoder_call(node.value):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name):
                    aliases.add(tgt.id)
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            if _ao_decoder_call(node.value) and isinstance(node.target, ast.Name):
                aliases.add(node.target.id)

    def _is_decoder_derived(node: ast.AST) -> bool:
        for sub in ast.walk(node):
            if _ao_decoder_call(sub):
                return True
            if isinstance(sub, ast.Name) and sub.id in aliases:
                return True
        return False

    hits: List[int] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        is_argmax = (
            (isinstance(fn, ast.Attribute) and fn.attr == "argmax")
            or (isinstance(fn, ast.Name) and fn.id == "argmax")
        )
        if not is_argmax:
            continue
        # torch.argmax(x) -> arg; x.argmax() -> the attribute value.
        targets: List[ast.AST] = list(node.args)
        if isinstance(fn, ast.Attribute):
            targets.append(fn.value)
        if any(_is_decoder_derived(t) for t in targets):
            hits.append(getattr(node, "lineno", 0))

    if not hits:
        return None

    where = ", ".join(f"line {ln}" for ln in sorted(set(hits))[:5])
    return (
        f"takes an ARGMAX over the action-object decoder ({where}). The round "
        "trip a -> E2.action_object(a) -> action_object_decoder is NOT "
        "invertible on this substrate: the argmax collapses to a single "
        "constant class, so an action stream selected this way is INVARIANT "
        "under every manipulation of the candidate set or its scores and the "
        "experiment is an arithmetically forced no-op (measured: 5 distinct "
        "constructed first actions all re-decode to 1 class, untrained and "
        "after 40 warmup episodes; V3-EXQ-801 arms came out bit-identical). "
        "Not repaired by use_support_preserving_cem or "
        "use_action_class_scaffold_candidates -- both act on Trajectory.actions. "
        "FIX: select via agent.select_action(candidates, ticks), or read the "
        "candidate's real action via "
        "HippocampalModule.candidate_first_action_class(traj). If this argmax "
        "only REPORTS the collapse (a diagnostic probe), exempt with "
        "ACTION_OBJECT_SELECTION_EXEMPT = \"<reason>\". See "
        "HippocampalModule.candidate_first_action_class and "
        "tests/contracts/test_action_object_roundtrip_not_an_action_source.py."
    )


# ---- Spearman guard-on-rank-variance shape (SD-081) ---------------------------------
# The copy-paste defect the canonical helper experiments/_lib/stats.spearman closes:
# 18 scripts each carried a `_spearman*` copy that computed ORDINAL ranks and then
# guarded degeneracy on the variance of the RANK vector instead of the INPUT vector:
#
#     ra = np.argsort(np.argsort(a))          # ordinal ranks, ties NOT averaged
#     if np.std(ra) == 0.0: return None       # NEVER True on a constant input
#
# Double-argsort of a constant input returns a permutation of 0..K-1 whose std is large
# (9.23 at K=32), not 0 -- so a constant vector sails past the guard and Spearman is
# computed against an arbitrary stable-sort tie-break ordering (deterministic noise;
# confirmed magnitudes up to |0.74| on genuinely constant vectors). Full autopsy:
# REE_assembly/evidence/planning/failure_autopsy_sd081-spearman-degenerate-dv_2026-07-27.
#
# The 18 defective copies were migrated to import the canonical helper, and the canonical
# helper is pinned by tests/contracts/test_spearman_input_guard.py -- that closes the
# copies that EXISTED. This lint closes the SHAPE going forward: it flags a NEW hand-rolled
# ordinal-rank correlation whose degeneracy guard is on the ranks rather than the input,
# i.e. a "19th copy". The discriminator against the 16 already-safe average-rank helpers in
# the corpus is that they AVERAGE-RANK ties (a constant input -> all-equal ranks -> genuine
# 0 rank-variance), which fixes the same class of bug structurally.
_SPEARMAN_GUARD_SHAPE_EXEMPT_MARKER = "SPEARMAN_GUARD_SHAPE_EXEMPT"
_SPEARMAN_MEAN_CALLS = ("mean", "nanmean", "fmean", "average", "median")


def _sp_is_argsort_call(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call):
        return False
    fn = node.func
    return ((isinstance(fn, ast.Attribute) and fn.attr == "argsort")
            or (isinstance(fn, ast.Name) and fn.id == "argsort"))


def _sp_has_double_argsort(scope: ast.AST) -> bool:
    """`np.argsort(np.argsort(...))` or `x.argsort().argsort()` -- ordinal ranking.

    Double-argsort produces ordinal ranks (0..K-1, ties broken by stable sort) and NEVER
    averages ties -- no safe helper in the corpus uses it, so within a rank-correlation
    context its presence is the defective signature.
    """
    for node in ast.walk(scope):
        if not _sp_is_argsort_call(node):
            continue
        for a in node.args:                       # np.argsort(np.argsort(x))
            if _sp_is_argsort_call(a):
                return True
        fn = node.func                            # x.argsort().argsort()
        if isinstance(fn, ast.Attribute) and _sp_is_argsort_call(fn.value):
            return True
    return False


def _sp_has_sorted_range_key(scope: ast.AST) -> bool:
    """`sorted(range(...), key=...)` -- the ordinal index-sort idiom.

    NOTE this is used by the SAFE average-rank helpers too (to get the sort ORDER, which
    they then average-rank). So it is only a defective signal in combination with the
    NOT-averaging discriminator below; on its own it is not.
    """
    for node in ast.walk(scope):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "sorted"):
            continue
        has_range = any(isinstance(a, ast.Call) and isinstance(a.func, ast.Name)
                        and a.func.id == "range" for a in node.args)
        has_key = any(kw.arg == "key" for kw in node.keywords)
        if has_range and has_key:
            return True
    return False


def _sp_averages_ties(scope: ast.AST) -> bool:
    """The tie-averaging idiom -- present in EVERY safe average-rank helper, absent in the
    ordinal defective ones. Two independent signals, either suffices:

      (1) an adjacent-tie equality check `v[order[j+1]] == v[order[i]]` -- an ast.Compare
          with Eq where at least one operand is a Subscript, and
      (2) a mean/median call (the averaged midrank), np.mean(...) / statistics.fmean.

    The defective shapes have neither: double-argsort assigns ordinal ranks directly, and
    the `sorted(range,key=)` ordinal copy assigns `ranks[idx] = rank_val + 1` via
    enumerate with no tie run. (A `np.std(ra) == 0.0` guard is an Eq whose operands are a
    Call and a Constant, not a Subscript -- so it does not trip signal (1).)
    """
    for node in ast.walk(scope):
        if isinstance(node, ast.Compare) and any(isinstance(op, ast.Eq) for op in node.ops):
            if any(isinstance(o, ast.Subscript) for o in [node.left] + list(node.comparators)):
                return True
        if isinstance(node, ast.Call):
            fn = node.func
            nm = (fn.attr if isinstance(fn, ast.Attribute)
                  else fn.id if isinstance(fn, ast.Name) else None)
            if nm in _SPEARMAN_MEAN_CALLS:
                return True
    return False


def _sp_guards_input(scope: ast.AST, params: set) -> bool:
    """Does the scope guard the INPUT vector's degeneracy (the correct fix)?

    Recognised: a `len(set(...))` distinct-value count (the canonical helper's exact input
    guard), OR an `np.std(P)` / `P.std()` degeneracy compare whose subject references an
    input PARAMETER P (as opposed to a locally-built rank vector). A helper that already
    guards its input has closed the SD-081 defect and must not fire -- this is the
    "rather than the INPUT vector" clause.
    """
    for node in ast.walk(scope):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "len"
                and any(isinstance(a, ast.Call) and isinstance(a.func, ast.Name)
                        and a.func.id == "set" for a in node.args)):
            return True
    for node in ast.walk(scope):
        subj = _sp_std_call_subject(node)
        if subj is None:
            continue
        if {n.id for n in ast.walk(subj) if isinstance(n, ast.Name)} & params:
            return True
    return False


def _sp_computes_rank_correlation(scope: ast.AST) -> bool:
    """The scope actually forms a rank correlation -- scopes OUT `sorted(range())` used
    merely for top-k / ordering. Signalled by a corrcoef/pearsonr call, or a manual
    Pearson denominator (a sqrt / `** 0.5` of a sum-of-squares)."""
    for node in ast.walk(scope):
        if isinstance(node, ast.Call):
            fn = node.func
            nm = (fn.attr if isinstance(fn, ast.Attribute)
                  else fn.id if isinstance(fn, ast.Name) else None)
            if nm in ("corrcoef", "sqrt", "pearsonr"):
                return True
        if (isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow)
                and isinstance(node.right, ast.Constant) and node.right.value == 0.5):
            return True
    return False


def _sp_std_call_subject(node: ast.AST) -> Optional[ast.expr]:
    """If `node` is a `std(...)` call (np.std(X) or X.std()), return its subject X."""
    if not isinstance(node, ast.Call):
        return None
    fn = node.func
    if isinstance(fn, ast.Attribute) and fn.attr == "std":
        return fn.value                                   # X.std()
    if isinstance(fn, ast.Name) and fn.id == "std":
        return node.args[0] if node.args else None        # std(X) / np.std(X)
    return None


def _sp_rank_degeneracy_guard(scope: ast.AST, params: set) -> bool:
    """The DEFECTIVE guard: a Compare against a small numeric (`< 1e-N`, `<= 0`, `== 0`)
    whose subject is a std-of-RANKS or a denominator/variance name -- NOT an input
    parameter. The subject is searched inside the operand subtree so a numeric-cast
    wrapper (`float(np.std(ra)) == 0.0`) does not hide it.
    """
    for node in ast.walk(scope):
        if not isinstance(node, ast.Compare):
            continue
        for op, comp in zip(node.ops, node.comparators):
            if not isinstance(op, (ast.Lt, ast.LtE, ast.Eq)):
                continue
            if not (isinstance(comp, ast.Constant)
                    and isinstance(comp.value, (int, float))
                    and not isinstance(comp.value, bool)
                    and abs(comp.value) < 1e-3):
                continue
            left = node.left
            for sub in ast.walk(left):                    # (a) std(rank_vector) guard
                subj = _sp_std_call_subject(sub)
                if subj is None:
                    continue
                names = {n.id for n in ast.walk(subj) if isinstance(n, ast.Name)}
                if not (names & params):
                    return True
            if isinstance(left, ast.Name):                # (b) denominator/variance name
                nm = left.id.lower()
                if (nm.startswith(("den", "dx", "dy", "vx", "vy", "sd_", "std"))
                        or nm in ("d", "denom", "denominator")):
                    return True
    return False


def spearman_guard_shape_lint(path: Path) -> Optional[str]:
    """SD-081 guard-on-rank-variance shape check. Return a warning string, or None.

    Fires on a function that (all of):
      1. builds ORDINAL ranks -- `np.argsort(np.argsort(...))` or `sorted(range(...),
         key=...)` (the two spellings the corpus used),
      2. forms a rank CORRELATION from them (a corrcoef/pearsonr call, or a manual Pearson
         `** 0.5`/sqrt denominator) -- this scopes out `sorted(range())` used for top-k,
      3. does NOT average-rank ties (the discriminator against the 16 safe average-rank
         helpers -- see `_sp_averages_ties`),
      4. does NOT guard the INPUT vector's degeneracy (`len(set(x)) < 2` / `np.std(x)==0`
         on a parameter -- the correct fix; a helper that already does this is safe), and
      5. carries the DEFECTIVE degeneracy guard: a `.std()`/`np.std`/`den < 1e-N` test on
         the RANK vector (`_sp_rank_degeneracy_guard`).

    That is exactly the shape the SD-081 autopsy names: a constant input sails past a
    rank-variance guard that can never fire, and Spearman is computed against an arbitrary
    stable-sort tie-break ordering (phantom |rho| up to 0.74). The fix is the canonical,
    input-guarded, tie-averaging helper `from experiments._lib.stats import spearman`.

    Evaluated PER FUNCTION-DEF (ast.walk recurses into nested rank helpers, so an outer
    `_spearman` sees its inner `_rank`'s ordinal construct), which keeps unrelated
    module-level equality checks from masking the shape.

    WARN-ONLY in BOTH modes -- like the action-object / e3 / recomputability lints it never
    hardens under --paths. It flags a SUSPECTED defective shape via a static name/AST scan
    (same limitation class as the other lints: it can miss a helper split across modules or
    assembled at runtime, and could in principle over-fire on a coincidental ordinal-sort +
    denominator + no-averaging combination -- measured 0 such coincidences across the 1157
    corpus scripts). Triage each fire: an ordinal rank correlation guarded on rank variance
    is the defect; an average-rank helper or an input-guarded one is not.

    Opt-out: SPEARMAN_GUARD_SHAPE_EXEMPT = "<reason>".
    """
    if "experiments/_lib/" in str(path).replace("\\", "/"):
        return None  # the canonical helper lives here; excluded by contract
    try:
        src = path.read_text(encoding="utf-8")
        tree = ast.parse(src, filename=str(path))
    except (OSError, UnicodeDecodeError, SyntaxError):
        return None  # check_script already reports unreadable / syntax errors
    if _SPEARMAN_GUARD_SHAPE_EXEMPT_MARKER in src:
        return None

    hits: List[str] = []
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        params = ({a.arg for a in fn.args.args}
                  | {a.arg for a in fn.args.posonlyargs}
                  | {a.arg for a in fn.args.kwonlyargs})
        if not (_sp_has_double_argsort(fn) or _sp_has_sorted_range_key(fn)):
            continue
        if not _sp_computes_rank_correlation(fn):
            continue
        if _sp_averages_ties(fn):
            continue  # SAFE: average-ranked ties (constant input -> equal ranks -> 0 var)
        if _sp_guards_input(fn, params):
            continue  # guards the input vector, not the ranks -- SD-081 defect closed
        if not _sp_rank_degeneracy_guard(fn, params):
            continue  # no rank-variance/denominator guard -- not the guard-on-rank shape
        hits.append(fn.name)

    if not hits:
        return None
    return ("hand-rolled Spearman/rank-correlation helper(s) "
            + ", ".join(sorted(set(hits)))
            + " compute ORDINAL ranks (np.argsort(np.argsort(...)) or "
            "sorted(range(...), key=...)) and guard degeneracy on the variance/std of the "
            "RANK vector rather than the INPUT vector -- the SD-081 defect. A constant "
            "input sails past a rank-variance guard that can NEVER fire (double-argsort of "
            "a constant returns a permutation of 0..K-1, std ~9.23 at K=32, not 0), so "
            "Spearman is computed against an arbitrary stable-sort tie-break ordering -- "
            "deterministic noise, |rho| confirmed up to 0.74 on genuinely constant "
            "vectors. FIX: delete the local helper and use the canonical, input-guarded, "
            "tie-averaging one: `from experiments._lib.stats import spearman` (returns None "
            "on a degenerate input; map None to your degenerate sentinel explicitly). "
            "Exempt with SPEARMAN_GUARD_SHAPE_EXEMPT = \"<reason>\". See "
            "tests/contracts/test_spearman_input_guard.py + "
            "REE_assembly/evidence/planning/"
            "failure_autopsy_sd081-spearman-degenerate-dv_2026-07-27.md.")


# ---- inert salience_apply_to_dacc_bias (MECH-244) ------------------------------------
# agent.py (dACC score-bias activation) does, gated by this flag:
#     dacc_score_bias = dacc_score_bias * write_gate("e3_policy")
# which reads as "the MECH-261 write-gate now modulates action selection". But the bias
# it scales comes from `DACCtoE3Adapter.forward` (ree_core/cingulate/dacc.py), and that
# function multiplies the ENTIRE bias by `self.config.dacc_weight`
# (`weight = dacc_weight * drive_gain`; `bias = weight * (...)`), with `dacc_weight` and
# every per-candidate sub-weight defaulting to 0.0. So `salience_apply_to_dacc_bias=True`
# WITHOUT a positive `dacc_weight` scales the ZERO vector by the gate -- arithmetically
# inert, no error, and any arm resting on that channel is a guaranteed null that LOOKS
# measured. Caught live authoring V3-EXQ-799 (measured dacc_bias cross-candidate range =
# 0.0 in all four arms before dacc_weight was added). Same family as the
# from_dims-swallows-unknown-kwargs hazard: a flag necessary but not sufficient.
_INERT_SALIENCE_DACC_BIAS_EXEMPT_MARKER = "INERT_SALIENCE_DACC_BIAS_EXEMPT"


def _dacc_weight_is_positive(value: ast.AST) -> bool:
    """True if a `dacc_weight=<value>` expression plausibly activates the channel.

    A numeric literal is positive iff it is non-zero (an explicit `dacc_weight=0.0`
    is the inert case the gate exists to catch). `None`/`False` are not positive.
    Anything NON-literal -- a module constant `DACC_WEIGHT`, an attribute, a call, an
    arithmetic expression -- is treated as set-and-positive: its runtime value is not
    statically known, so assuming it activates the channel is the false-negative-safe
    choice (V3-EXQ-799 uses `dacc_weight=DACC_WEIGHT`, and must NOT fire).
    """
    if isinstance(value, ast.Constant):
        if isinstance(value.value, bool):
            return False
        if isinstance(value.value, (int, float)):
            return value.value != 0
        return False  # None, str, etc.
    return True  # Name / Attribute / Call / BinOp / ... -- value unknown, assume live


def _dacc_weight_set_positively_anywhere(tree: ast.AST) -> bool:
    """File-wide escape: some site sets dacc_weight to a positive value.

    Suppresses a fire when the weight is activated separately from the flag call --
    `cfg = REEConfig.from_dims(..., salience_apply_to_dacc_bias=True); cfg.dacc_weight
    = 0.5`, or a positive `dacc_weight=` keyword on a DIFFERENT config construction.
    Covers keyword args on any call, `<x>.dacc_weight = <positive>` attribute assigns,
    and `<d>["dacc_weight"] = <positive>` subscript assigns.
    """
    for n in ast.walk(tree):
        if isinstance(n, ast.Call):
            for kw in n.keywords:
                if kw.arg == "dacc_weight" and _dacc_weight_is_positive(kw.value):
                    return True
        elif isinstance(n, ast.Assign):
            for tgt in n.targets:
                if isinstance(tgt, ast.Attribute) and tgt.attr == "dacc_weight" \
                        and _dacc_weight_is_positive(n.value):
                    return True
                if isinstance(tgt, ast.Subscript) and isinstance(tgt.slice, ast.Constant) \
                        and tgt.slice.value == "dacc_weight" \
                        and _dacc_weight_is_positive(n.value):
                    return True
    return False


def inert_salience_dacc_bias_lint(path: Path) -> Optional[str]:
    """Inert `salience_apply_to_dacc_bias=True` check. Return an issue string, or None.

    Fires when a config construction passes a LITERAL `salience_apply_to_dacc_bias=True`
    but no positive `dacc_weight` is set -- neither in that same call nor anywhere in the
    file (`_dacc_weight_set_positively_anywhere` is the escape for the weight-set-
    separately idiom). Because `DACCtoE3Adapter.forward` multiplies the whole dACC->E3
    score bias by `dacc_weight` (default 0.0), the flag is then arithmetically inert: the
    `write_gate("e3_policy")` modulation in agent.py scales the zero vector, so any arm
    whose readout depends on that channel is a guaranteed null that looks measured.

    Confirmed near-miss (V3-EXQ-799, 2026-07): authored with the flag True and no
    dacc_weight; a P0 probe measured dacc_bias cross-candidate range = 0.0 in all four
    arms, and the driver added `dacc_weight=DACC_WEIGHT` + `dacc_interaction_weight` to
    make the channel live. That fixed shape (flag True WITH a positive dacc_weight) is the
    reference and must NOT fire.

    Detection is AST-call-keyword based, so it is deliberately blind to a flag that
    reaches the config through a dict/**kwargs, a helper the scan cannot follow, or a
    docstring mention (V3-EXQ-455a lists the flag in its Arms docstring but is a gated
    NotImplementedError stub that constructs no config -- correctly not a fire; its
    inertness is a stronger form than this gate targets). Under-fires in those cases,
    which is the acceptable direction for an advisory net.

    Opt-out: INERT_SALIENCE_DACC_BIAS_EXEMPT = "<reason>" -- use when the bias is
    deliberately enabled while being DRIVEN FROM ANOTHER HEAD, so the flag is meant to be
    on without a dacc_weight.

    WARN-only in BOTH modes -- never hardens under --paths. A fire is a SUSPECTED inert
    channel (the weight could be assembled at runtime in a way the scan cannot see), and a
    landed carrier's run is complete, so hardening would block commits on history.
    """
    try:
        src = path.read_text(encoding="utf-8")
        tree = ast.parse(src, filename=str(path))
    except (OSError, SyntaxError, UnicodeDecodeError):
        return None  # check_script already reports unreadable / syntax errors

    if _INERT_SALIENCE_DACC_BIAS_EXEMPT_MARKER in src:
        return None

    flag_sites = []
    for n in ast.walk(tree):
        if not isinstance(n, ast.Call):
            continue
        for kw in n.keywords:
            if kw.arg == "salience_apply_to_dacc_bias" and isinstance(kw.value, ast.Constant) \
                    and kw.value.value is True:
                # Positive dacc_weight IN THIS SAME call clears it outright.
                same_call_positive = any(
                    k.arg == "dacc_weight" and _dacc_weight_is_positive(k.value)
                    for k in n.keywords)
                if not same_call_positive:
                    flag_sites.append(n)
    if not flag_sites:
        return None

    # File-wide escape: dacc_weight set positively elsewhere (separate assign / call).
    if _dacc_weight_set_positively_anywhere(tree):
        return None

    where = ", ".join(f"line {n.lineno}" for n in flag_sites)
    return (
        f"enables salience_apply_to_dacc_bias=True at {where} but sets no positive "
        f"dacc_weight (it defaults to 0.0). agent.py scales the dACC->E3 score bias by "
        f"write_gate(\"e3_policy\"), but DACCtoE3Adapter.forward "
        f"(ree_core/cingulate/dacc.py) multiplies that whole bias by dacc_weight -- so "
        f"the gate modulates the ZERO vector. The channel is arithmetically inert: any "
        f"arm resting on it is a guaranteed null that looks measured (the V3-EXQ-799 "
        f"near-miss, where the measured dacc_bias cross-candidate range was 0.0 in every "
        f"arm). Fix by setting dacc_weight > 0 AND at least one per-candidate sub-weight "
        f"(dacc_interaction_weight / dacc_foraging_weight / dacc_suppression_weight) in "
        f"the same config -- V3-EXQ-799 is the canonical shape. Exempt with "
        f"{_INERT_SALIENCE_DACC_BIAS_EXEMPT_MARKER} = \"<reason>\" when the bias is "
        f"deliberately driven from another head."
    )


# ---- inert zworld_p0_episodes warmup (SD-070, confirmed twice) -----------------------
# experiments/_lib/allon_training._train_all_on_agent trains the shared 724-A0/734/742
# REINFORCE recipe. Its P0a stage -- the SD-070 z_world-encoder warmup -- is OPT-IN via
# `zworld_p0_episodes: int = 0`: with the default, `run_zworld_p0` never runs,
# `split_encoder.world_encoder` is never stepped, and z_world stays a frozen random
# projection for the whole run. No error, no warning -- the agent trains and every
# downstream competence/survival/foraging metric is computed exactly as if the encoder
# had been trained, just against noise.
#
# CONFIRMED TWICE, independently, on two different drivers:
#   V3-EXQ-728 (original run) -- "3/3 seeds failed"; fixed by adding zworld_p0_episodes=60.
#   V3-EXQ-875 (MECH-471, 2026-08-03) -- all three _train_all_on_agent call sites (lines
#     270/279/288) omitted the kwarg; the driver's own non-degeneracy guard correctly
#     caught the floor-pinned acquisition and self-routed substrate_not_ready_requeue
#     after a ~20.5h wall-clock run, rather than reporting a false discrimination verdict.
#     REE_assembly/evidence/planning/failure_autopsy_V3-EXQ-875_2026-08-03.md, "Root cause".
#     The corrected re-run (V3-EXQ-875a) adds zworld_p0_episodes=60 to its acquisition call.
#
# SCOPE, and why file-wide rather than per-call-site or per-training-arm. The precise rule
# (allon_training.py's own docstring) is "the FIRST (acquisition) call per shared agent
# needs the warmup; a later call training the SAME agent on a SECOND competence, or a
# targeted-update call reusing an already-trained encoder, correctly passes
# zworld_p0_episodes=0". Distinguishing "first call for this agent" from "later call for
# the same agent" statically would need data-flow tracking of the `agent` receiver across
# calls, which this scan does not attempt. Instead: fire only when NO call site in the
# file passes zworld_p0_episodes with a plausibly nonzero value ANYWHERE -- mirroring
# `_dacc_weight_set_positively_anywhere`'s file-wide escape. This is deliberately
# UNDER-firing (a driver that only wires the kwarg into a non-first call -- an authoring
# mistake in its own right -- would incorrectly escape here), which is the same accepted
# bias as every other WARN-only gate in this family: a false WARN costs a reader a minute;
# a driver that never wires the kwarg AT ALL (both confirmed incidents) is caught either
# way.
#
# SCOPE, second axis: only drivers whose acceptance criteria plausibly READ a
# z_world-dependent metric. Not every `_train_all_on_agent` caller needs this warmup -- a
# call whose downstream readout never depends on trained z_world would correctly leave the
# encoder frozen. Restricting to files that mention a competence/survival/foraging metric
# name (capability_eval's own DV vocabulary: foraging_competence, survival_horizon,
# death_rate, or the broader "competence"/"survival"/"foraging" word stems the two
# confirmed carriers' criteria are built from) keeps this from firing on a hypothetical
# caller that trains the agent for a purpose the encoder state does not affect.
#
# ADVISORY, never blocking, like every sibling in this file: a static name/keyword scan
# cannot see a kwarg threaded through a dict, a partial, or a helper wrapper (728/728b/734
# all define their OWN `zworld_p0_episodes` parameter and forward it BY NAME -- a Name
# kwarg value is treated as "assume positive", the false-negative-safe direction, exactly
# as `_dacc_weight_is_positive` does for the analogous case), and landed carriers' runs are
# complete -- do NOT retro-edit one; the remedy for a landed run is a superseding EXQ
# letter (V3-EXQ-875 -> 875a is the canonical shape) plus adjudicating the affected result
# via /failure-autopsy.
_ZWORLD_P0_WARMUP_EXEMPT_MARKER = "ZWORLD_P0_WARMUP_EXEMPT"

_ZWORLD_COMPETENCE_METRIC_TOKENS = ("survival", "foraging", "competence")


def _zworld_kwarg_is_positive(value: ast.AST) -> bool:
    """True if a `zworld_p0_episodes=<value>` expression plausibly enables the P0a warmup.

    Same shape as `_dacc_weight_is_positive`: a numeric literal is positive iff nonzero
    (an explicit `zworld_p0_episodes=0` is exactly the inert default, spelled out).
    Anything non-literal -- a module constant, a parameter forwarded by name (the
    728/728b/734 wrapper shape), an attribute, a call -- is treated as set-and-positive,
    since its runtime value is not statically known and assuming it activates the warmup
    is the false-negative-safe choice.
    """
    if isinstance(value, ast.Constant):
        if isinstance(value.value, bool):
            return False
        if isinstance(value.value, (int, float)):
            return value.value != 0
        return False  # None, str, etc.
    return True  # Name / Attribute / Call / BinOp / ... -- value unknown, assume live


def zworld_p0_warmup_lint(path: Path) -> Optional[str]:
    """Silent SD-070 z_world-encoder warmup omission. Return an issue string, or None.

    Fires when ALL of:
      (1) the driver calls `_train_all_on_agent` (bare or `<module>._train_all_on_agent`)
          at least once;
      (2) NO call site anywhere in the file passes `zworld_p0_episodes` with a plausibly
          nonzero value (see `_zworld_kwarg_is_positive`) -- the file-wide escape mirrors
          `_dacc_weight_set_positively_anywhere`'s "set positively somewhere" rule, and is
          deliberately coarser than "the FIRST call per training arm" -- see the block
          comment above for why;
      (3) the file plausibly reports a z_world-dependent competence/survival/foraging
          metric -- a case-insensitive source-text match on "survival", "foraging", or
          "competence" (capability_eval's own DV vocabulary plus the word stems the two
          confirmed carriers' criteria are built from).

    Confirmed twice: V3-EXQ-728 ("3/3 seeds failed", fixed by zworld_p0_episodes=60) and
    V3-EXQ-875 (MECH-471, all three call sites omitted it, ~20.5h wall-clock run
    self-routed substrate_not_ready_requeue with zero usable evidence -- see
    REE_assembly/evidence/planning/failure_autopsy_V3-EXQ-875_2026-08-03.md).

    Opt-out: ZWORLD_P0_WARMUP_EXEMPT = "<reason>" -- use when the driver deliberately
    trains against a frozen random z_world projection (e.g. an ablation whose whole point
    is comparing trained vs untrained encoder), or when its criteria genuinely do not
    depend on z_world-derived competence despite the token match.

    WARN-only in BOTH modes -- never hardens under --paths, like every sibling here. A
    fire is a SUSPECTED omission (the kwarg could be threaded through a dict or a helper
    this scan cannot follow), and a landed carrier's run is complete, so hardening would
    block commits on history.
    """
    try:
        src = path.read_text(encoding="utf-8")
        tree = ast.parse(src, filename=str(path))
    except (OSError, SyntaxError, UnicodeDecodeError):
        return None

    if _ZWORLD_P0_WARMUP_EXEMPT_MARKER in src:
        return None

    call_sites = [n for n in ast.walk(tree)
                 if isinstance(n, ast.Call) and _call_name(n) == "_train_all_on_agent"]
    if not call_sites:
        return None

    for n in call_sites:
        for kw in n.keywords:
            if kw.arg == "zworld_p0_episodes" and _zworld_kwarg_is_positive(kw.value):
                return None  # escape -- set positively somewhere in the file

    lowered = src.lower()
    matched_tokens = [t for t in _ZWORLD_COMPETENCE_METRIC_TOKENS if t in lowered]
    if not matched_tokens:
        return None

    where = ", ".join(f"line {n.lineno}" for n in call_sites)
    return (
        f"calls _train_all_on_agent at {where} but never passes zworld_p0_episodes with "
        f"a nonzero value anywhere in the file (it defaults to 0). With no P0a SD-070 "
        f"z_world-encoder warmup, split_encoder.world_encoder is never stepped and "
        f"z_world stays a frozen random projection for the whole run -- no error, no "
        f"warning -- and the driver's own reported competence/survival/foraging metrics "
        f"(this file references '{'/'.join(matched_tokens)}') are computed exactly as if "
        f"the encoder had been trained, just against noise. Confirmed twice: V3-EXQ-728 "
        f"('3/3 seeds failed') and V3-EXQ-875 (MECH-471, ~20.5h wall-clock self-routed "
        f"substrate_not_ready_requeue with zero usable evidence -- "
        f"REE_assembly/evidence/planning/failure_autopsy_V3-EXQ-875_2026-08-03.md). Fix "
        f"by passing zworld_p0_episodes=<n> (60 is the validated V3-EXQ-728/875a "
        f"operating point) on at least the first (acquisition) call per shared agent, "
        f"plus a zworld_p0_env built from a FRESH env with the same seed/kwargs (reusing "
        f"train_env shifts the P0b/P1 layout sequence). Exempt with "
        f"{_ZWORLD_P0_WARMUP_EXEMPT_MARKER} = \"<reason>\" when the frozen encoder is "
        f"deliberate. Do NOT retro-edit a LANDED driver whose run is complete -- queue a "
        f"superseding EXQ letter instead (V3-EXQ-875 -> 875a is the canonical shape)."
    )


_DACC_LAST_BUNDLE_EXEMPT_MARKER = "DACC_LAST_BUNDLE_EXEMPT"


def _unparse_recv(node: "ast.AST") -> str:
    """Best-effort source text for a receiver expression, for the message only.

    Goes through the module-global `ast` deliberately: the shared corpus-scan fixture
    swaps that global for a parse cache, and every other unparse in this file does the
    same. Never raises -- a lint must not fail on an exotic receiver it only wanted to
    quote back to the reader.
    """
    try:
        return ast.unparse(node).strip()
    except Exception:
        return "<expr>"


def dacc_last_bundle_lint(path: Path) -> Optional[str]:
    """Wrong-attribute dACC bundle read. Return an issue string, or None.

    Fires on any access to an attribute literally named ``_last_bundle`` -- either
    ``<expr>._last_bundle`` or ``getattr(<expr>, "_last_bundle", ...)``. NO object in
    the substrate defines that attribute. The dACC bundle lives on the AGENT as
    ``agent._dacc_last_bundle`` (written ``ree_core/agent.py:6148``, canonical read
    ``:10340``); ``ree_core/cingulate/dacc.py`` defines no ``_last_bundle`` at all.

    Why this is a MEASUREMENT defect rather than a style nit: ``getattr`` with a default
    swallows the miss, so the read yields None on every tick and every max/mean derived
    from it is pinned to 0.0 BY CONSTRUCTION -- a structural zero that is indistinguishable
    in the manifest from a measured zero. The per-candidate ``suppression`` [K] tensor
    (``dacc.py:401``, packed at ``:430``) is the usual casualty, via
    ``dacc_max_suppression``; ``pe`` and ``mode_ev`` readouts fail the same way.

    Confirmed carrier (V3-EXQ-687, 2026-06-18): self-routed ``substrate_not_ready_requeue``
    on a failed PRE_MECH260 precondition, ``dacc_max_suppression=0.0`` "with a full FIFO".
    With the corrected path a 687a smoke run reads suppression 1.0 on both dACC arms, so
    MECH-260 may never have been inoperative. The reference implementation is
    ``v3_exq_687a_mech313_committed_authority_dissociation.py::_dacc_diag``.

    SECOND-ORDER TRAP the fix must carry (this lint does not detect it -- read the site):
    several carriers pair the wrong attribute with ``bundle.get("mode_ev") or
    bundle.get("harm_interaction")``. ``mode_ev`` is a [K] tensor, so ``or`` invokes
    ``__bool__`` and raises "Boolean value of Tensor with more than one value is
    ambiguous". In the two SD-054 stage helpers that expression sat OUTSIDE the enclosing
    try, so repairing the attribute ALONE converts a silent zero into a crash. Rewrite as
    an explicit ``if sb is None:`` fallback when fixing.

    Detection is AST-based and keys on EXACT attribute-name equality, which matters twice:
    a substring test would match the CORRECT ``_dacc_last_bundle`` (it contains
    ``_last_bundle``) and fire on every repaired site, and comments/docstrings are invisible
    to the AST, so the several drivers that merely DESCRIBE the defect in prose -- 490h,
    490i, 490j, 687a, and this file's own quoting of it -- correctly do not fire. It is
    blind to an attribute name assembled at runtime, the acceptable direction for a net.

    Opt-out: DACC_LAST_BUNDLE_EXEMPT = "<reason>" -- for a genuine ``_last_bundle`` on some
    future object that really defines one.

    WARN-only in BOTH modes -- never hardens under --paths. The 15 landed carrier drivers'
    runs are COMPLETE and are deliberately not retro-edited (a completed run's
    pre-registered emission is not rewritten), so hardening would block commits on history.
    """
    try:
        src = path.read_text(encoding="utf-8")
        tree = ast.parse(src, filename=str(path))
    except (OSError, SyntaxError, UnicodeDecodeError):
        return None  # check_script already reports unreadable / syntax errors

    if _DACC_LAST_BUNDLE_EXEMPT_MARKER in src:
        return None

    # (lineno, rendered) so the message reads in SOURCE order. ast.walk is BFS, so
    # collecting straight into a list interleaves a nested site ahead of an earlier
    # top-level one -- deterministic, but it reads as though the lint missed one.
    found: List[Tuple[int, str]] = []
    for n in ast.walk(tree):
        # getattr(<expr>, "_last_bundle", ...) -- the swallowing form.
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                and n.func.id == "getattr" and len(n.args) >= 2
                and isinstance(n.args[1], ast.Constant)
                and n.args[1].value == "_last_bundle"):
            found.append((n.lineno,
                          f"line {n.lineno}: getattr({_unparse_recv(n.args[0])}, "
                          f'"_last_bundle", ...)'))
        # <expr>._last_bundle -- the raising form. EXACT match, so the correct
        # `_dacc_last_bundle` (which contains this as a substring) never fires.
        elif isinstance(n, ast.Attribute) and n.attr == "_last_bundle":
            found.append((n.lineno,
                          f"line {n.lineno}: {_unparse_recv(n.value)}._last_bundle"))

    if not found:
        return None
    sites = [s for _, s in sorted(found, key=lambda t: t[0])]

    return (
        f"reads a `_last_bundle` attribute that NO substrate object defines, at "
        f"{'; '.join(sites)}. The dACC bundle lives on the AGENT as "
        f"`agent._dacc_last_bundle` (ree_core/agent.py:6148, canonical read :10340); "
        f"ree_core/cingulate/dacc.py defines no `_last_bundle`. With a getattr default "
        f"the miss is SILENT, so the read is None every tick and any derived "
        f"max/mean (dacc_max_suppression, dacc_pe, dacc_bias_nonzero) is pinned to 0.0 "
        f"BY CONSTRUCTION -- structurally indistinguishable in the manifest from a "
        f"measured zero. This is the V3-EXQ-687 carrier shape. Fix by reading "
        f"`getattr(agent, \"_dacc_last_bundle\", None)`; reference implementation is "
        f"v3_exq_687a_...::_dacc_diag. WHEN FIXING, also check for "
        f"`bundle.get(\"mode_ev\") or bundle.get(...)` at the same site: mode_ev is a [K] "
        f"tensor, so `or` raises on __bool__ and the attribute fix alone can turn a "
        f"silent zero into a crash -- use an explicit `if sb is None:` fallback. Exempt "
        f"with {_DACC_LAST_BUNDLE_EXEMPT_MARKER} = \"<reason>\"."
    )


_AGENT_SEED_ORDER_EXEMPT_MARKER = "AGENT_SEED_ORDER_EXEMPT"

# Recognised as a torch-RNG seed event -- see agent_construction_before_seed_lint.
_TORCH_SEED_CALL_NAMES = ("manual_seed",)   # matched via full dotted path below
_TORCH_SEED_HELPER_NAMES = ("reset_all_rng", "seeded_construct")


def _is_guard_clause(body: List[ast.stmt]) -> bool:
    """True if `body` (an `if` block's statements) unconditionally exits --
    `return`/`raise`/`continue`/`break` as its last statement. See
    `_DirectFlowWalker.visit_If` for why this matters: a guard-clause branch and
    the code that follows the `if` are MUTUALLY EXCLUSIVE execution paths (taking
    the branch means never reaching what follows, and vice versa), so merging
    their events into one flat ordering check is unsound -- confirmed real
    instance: `v3_exq_418k_sd016_context_memory_reef.py::main`'s `if
    args.dry_run: ...; return` block builds an agent with no seed call of its
    own, but the REAL run path (reached only when `args.dry_run` is False) calls
    a correctly-seeded function -- the two were never going to run together.
    """
    return bool(body) and isinstance(body[-1], (ast.Return, ast.Raise, ast.Continue, ast.Break))


class _DirectFlowWalker(ast.NodeVisitor):
    """Collects `ast.Call` nodes belonging to ONE function's own execution flow.

    Deliberately does NOT descend into nested `FunctionDef` / `AsyncFunctionDef` /
    `Lambda` bodies -- those are separate call frames, invoked (if ever) at a time
    this walk cannot order relative to the enclosing function's own statements.
    Also records `with arm_cell(...) as x:` context managers, whose `__enter__`
    calls `reset_all_rng` (see `_lib/arm_fingerprint.py::_ArmCell.__enter__`) unless
    constructed with `do_reset=False`.

    A guard-clause `if` (see `_is_guard_clause`) is ISOLATED into its own
    `_DirectFlowWalker`, appended to `self.isolated_scopes`, rather than merged
    into `self.calls` -- the caller checks each scope (main flow, plus every
    isolated branch, recursively) for the ordering violation INDEPENDENTLY, since
    they are mutually-exclusive execution paths. `if`s that do NOT unconditionally
    exit are treated as before (merged into the surrounding flow) -- real branch-
    sensitive analysis for arbitrary if/else is out of scope; the guard-clause
    shape is the one confirmed real false-positive source.
    """

    def __init__(self) -> None:
        self.calls: List[ast.Call] = []
        self.arm_cell_seed_lines: List[int] = []
        self.isolated_scopes: List["_DirectFlowWalker"] = []

    def visit_FunctionDef(self, node: ast.AST) -> None:
        pass

    def visit_AsyncFunctionDef(self, node: ast.AST) -> None:
        pass

    def visit_Lambda(self, node: ast.AST) -> None:
        pass

    def visit_If(self, node: ast.If) -> None:
        self.visit(node.test)  # the condition always evaluates -- stays in this flow
        if _is_guard_clause(node.body):
            sub = _DirectFlowWalker()
            for stmt in node.body:
                sub.visit(stmt)
            self.isolated_scopes.append(sub)
            self.isolated_scopes.extend(sub.isolated_scopes)  # flatten nested guards
            sub.isolated_scopes = []
        else:
            for stmt in node.body:
                self.visit(stmt)
        for stmt in node.orelse:
            self.visit(stmt)

    def visit_With(self, node: ast.With) -> None:
        for item in node.items:
            ctx = item.context_expr
            if isinstance(ctx, ast.Call):
                if (isinstance(ctx.func, ast.Name) and ctx.func.id == "arm_cell"
                        and not any(kw.arg == "do_reset"
                                   and isinstance(kw.value, ast.Constant)
                                   and kw.value.value is False
                                   for kw in ctx.keywords)):
                    self.arm_cell_seed_lines.append(node.lineno)
                self.calls.append(ctx)
        for stmt in node.body:
            self.visit(stmt)

    def visit_AsyncWith(self, node: ast.AsyncWith) -> None:
        self.visit_With(node)  # type: ignore[arg-type]

    def visit_Call(self, node: ast.Call) -> None:
        self.calls.append(node)
        self.generic_visit(node)


def _flatten_walker(w: "_DirectFlowWalker") -> Tuple[List[ast.Call], List[int]]:
    """All calls + arm_cell lines from `w` AND every isolated sub-scope, merged.

    Used only for the coarse one-hop ctor/seed NAME resolution ("can calling
    this function ever construct/seed, in ANY branch") -- branch-sensitivity
    only matters for the per-scope ORDERING check, done separately below.
    """
    calls = list(w.calls)
    arm_cell = list(w.arm_cell_seed_lines)
    for sub in w.isolated_scopes:
        sub_calls, sub_arm = _flatten_walker(sub)
        calls += sub_calls
        arm_cell += sub_arm
    return calls, arm_cell


def _build_parent_map(tree: ast.Module) -> Dict[int, ast.AST]:
    """Map `id(child)` -> parent node, for the whole module. Built fresh (never
    cached on the nodes themselves via `setattr` -- the shared corpus-scan parse
    cache in `tests/contracts/conftest.py` requires every consumer to be
    read-only; see that module's docstring) so `_is_discarded_agent_subscript`
    can ask "is this Call's result immediately subscripted" without needing a
    full parent-tracking traversal of its own.
    """
    parent_of: Dict[int, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parent_of[id(child)] = node
    return parent_of


def _find_agent_tuple_index(fn_node: ast.AST) -> Optional[int]:
    """If `fn_node` assigns `X = REEAgent(...)` and returns a tuple literal
    containing `Name(id=X)`, return X's position in that tuple. `None` if
    undeterminable (agent not returned via a simple tuple literal, or not
    assigned to a single plain name) -- callers must treat `None` as "cannot
    verify", never as "definitely not the agent".
    """
    agent_var: Optional[str] = None
    for node in ast.walk(fn_node):
        if (isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
                and _call_name(node.value) == "REEAgent" and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)):
            agent_var = node.targets[0].id
            break
    if agent_var is None:
        return None
    for node in ast.walk(fn_node):
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Tuple):
            for i, elt in enumerate(node.value.elts):
                if isinstance(elt, ast.Name) and elt.id == agent_var:
                    return i
    return None


def _is_discarded_agent_subscript(
    call: ast.Call, parent_of: Dict[int, ast.AST], ctor_agent_index: Dict[str, Optional[int]],
) -> bool:
    """True only when POSITIVELY CONFIRMED that `call`'s result is subscripted
    for an element OTHER than the agent, so the constructed `REEAgent` itself
    is unreachable after this statement and cannot be the object any later
    code actually uses.

    Confirmed real shape (5 of the original 18 backlog carriers, 2026-08-01
    triage): `probe_slice = _build(seed)[4]` before `with arm_cell(...)`, used
    only to harvest a static config-slice dict for the fingerprint -- the
    `REEAgent` object `_build` also constructs and returns (at a DIFFERENT
    tuple position) is immediately garbage; the REAL, scored agent is built by
    a second, correctly-seeded call inside the `arm_cell` block. Flagging the
    discarded probe as "the" agent-construction event is a false positive.

    DELIBERATELY CONSERVATIVE -- returns False (do not discard, i.e. treat as a
    real construction) whenever this cannot be positively verified: `call` is
    `REEAgent(...)` directly (never itself a tuple -- subscripting it would be
    a TypeError, so this branch never applies to it), the resolved one-hop
    ctor function's agent position could not be determined
    (`_find_agent_tuple_index` returned `None`), or the subscript's index is
    not a simple integer constant. A false NEGATIVE here (missing a genuine
    two-hop bug) is the safe failure direction for a WARN-only, evidence-
    integrity-relevant lint -- silently suppressing a real fire is worse than
    occasionally still flagging a benign one.
    """
    parent = parent_of.get(id(call))
    if not isinstance(parent, ast.Subscript):
        return False
    name = _call_name(call)
    if name == "REEAgent":
        return False
    idx = ctor_agent_index.get(name) if name else None
    if idx is None:
        return False
    sl = parent.slice
    if isinstance(sl, ast.Constant) and isinstance(sl.value, int):
        return sl.value != idx
    return False


def _call_name(node: ast.Call) -> Optional[str]:
    f = node.func
    if isinstance(f, ast.Name):
        return f.id
    if isinstance(f, ast.Attribute):
        return f.attr
    return None


def _is_direct_torch_seed_call(node: ast.Call) -> bool:
    name = _call_name(node)
    if name in _TORCH_SEED_HELPER_NAMES:
        return True
    if name not in _TORCH_SEED_CALL_NAMES:
        return False
    # require the receiver to plausibly be `torch` (or `torch.random`), not an
    # unrelated `.manual_seed()` on some other object.
    f = node.func
    if not isinstance(f, ast.Attribute):
        return False
    recv = f.value
    if isinstance(recv, ast.Name):
        return recv.id == "torch"
    if isinstance(recv, ast.Attribute):
        return recv.attr == "random" and isinstance(recv.value, ast.Name) and recv.value.id == "torch"
    return False


def agent_construction_before_seed_lint(path: Path) -> Optional[str]:
    """Agent-weight non-reproducibility: `REEAgent(...)` built before torch is seeded.

    `torch.nn.Module` weight init (Linear/Conv/etc. default initialisers) draws from
    TORCH'S OWN global RNG, not numpy's or Python's `random` -- `np.random.seed(...)` /
    `random.seed(...)` alone do NOT make agent weights reproducible, only
    `torch.manual_seed(...)` (equivalently `reset_all_rng`, which calls it) does. A
    script that constructs its agent before ever seeding torch gets an agent whose
    initial weights depend on whatever the process's global torch RNG state happens to
    be at that moment -- a function of import order and prior random draws in the same
    process, NOT of `seed`. Confirmed empirically (three back-to-back calls with
    identical kwargs returned three different boundary-event counts) and by source
    read across the whole corpus, 2026-08-01.

    Fires on a MODULE-LEVEL function that, in its own direct statement flow (not
    inside a nested def -- a separate call frame), calls BOTH:
      (a) an agent constructor -- `REEAgent(...)` directly, or a call to another
          module-level function whose OWN body directly constructs `REEAgent`
          (one hop; covers the near-universal `make_agent(env)`-wraps-`REEAgent(cfg)`
          shape), and
      (b) a torch-RNG seed call -- `torch.manual_seed(...)` / `torch.random.manual_seed
          (...)`, `reset_all_rng(...)`, `seeded_construct(...)` directly, a call to
          another module-level function whose OWN body directly does one of those (one
          hop), or `with arm_cell(seed, ...) as cell:` (its `__enter__` calls
          `reset_all_rng` unless `do_reset=False`) --
    but the FIRST seed event in that function's own flow is not STRICTLY EARLIER (by
    source line) than the FIRST agent-construction event.

    THIS IS TIER 1 ONLY -- an unambiguous, high-confidence shape, not an exhaustive
    "is this script reproducible" check. It deliberately does NOT fire on a function
    that constructs an agent with NO seed call anywhere in its own local flow: that
    case is common and often fine (many diagnostic scripts never claim seed-driven
    weight reproducibility at all, and the seeding may legitimately happen in a caller
    this static, single-function scan cannot see) and is far noisier to adjudicate --
    scoped out on purpose rather than guessed at. So a clean result here is NOT proof a
    script's agent weights ARE reproducible; it only proves this specific "looks
    seeded, order is wrong" defect is absent.

    A helper function that BOTH constructs an agent AND seeds within its OWN body (e.g.
    `run_integration_arm(seed): torch.manual_seed(seed); ...; REEAgent(cfg)`, correctly
    ordered) is excluded from the one-hop NAME resolution used to interpret ITS
    CALLERS -- a single call site to such a dual-purpose helper cannot be read as
    "agent" or "seed" from the outside, since the helper's own internal order is what
    actually matters and it is independently checked on its own account (module-level
    functions are all scanned). Without this exclusion, a call to a CORRECTLY-ordered
    helper collapses to one line classified as both events simultaneously and produces
    a spurious tie -- measured and fixed during this lint's construction (2026-08-01,
    `v3_exq_519a_sd051_conditioned_safety_store_readiness.py::run_integration_arm`).

    CONFIRMED CARRIERS (2026-08-01 audit; the shape that motivated this lint): the
    Q-081 family `v3_exq_824`/`824a`/`838` and the INV-091 family `v3_exq_827`/`827a`/
    `828`/`828a` all build one shared P0 agent template per seed via `make_agent(...)`
    BEFORE any `with arm_cell(seed, ...)` in the same `run_seed`/`_run_seed`, so every
    arm within a seed shares byte-identical (but seed-UNCONTROLLED) initial weights.
    Audited immaterial to each script's OWN reported finding (arms are still matched
    via `copy.deepcopy` of that one shared template, so the reported INTACT-vs-
    manipulated comparisons are unconfounded regardless of what those shared weights
    are) -- see `REE_assembly/evidence/planning/q081_landmark_removal_arm_design.md`
    section 8. NOT retro-fixed: all four Q-081 carriers plus 849 have landed manifests;
    a completed run's pre-registered emission is not rewritten.

    ALL 18 originally-pinned carriers triaged (2026-08-01): every one immaterial to
    its own reported finding, for one of four recurring reasons -- see
    `REE_assembly/evidence/planning/agent_seed_order_lint_backlog_triage.md` for the
    per-script verdicts. Two of those reasons turned out to be genuine PRECISION
    GAPS in this lint's Tier-1 design, and were FIXED the same day (not merely
    documented), because trusting future fires shouldn't require re-deriving this
    triage each time:
      1. Branch-unawareness -- a `--dry-run`-only guard-clause branch (`if
         args.dry_run: ...; return`) was merged into the same ordering check as
         the function's real path, even though the two are mutually exclusive.
         Fixed by isolating any guard-clause `if` (body ends in an unconditional
         `return`/`raise`/`continue`/`break`) into its own independently-checked
         scope -- see `_DirectFlowWalker.visit_If` / `_is_guard_clause`.
      2. One-hop-only resolution couldn't see a "probe, discard, real-build-two-
         hops-away" idiom: `probe_slice = _build(seed)[N]` before `arm_cell`,
         used only to harvest a static config dict, while the REAL agent is
         built two call-hops away inside the (correctly resetting) `arm_cell`
         block. Fixed not by extending hop resolution (which would just move the
         blind spot) but by recognising the discard itself: a ctor call whose
         result is immediately subscripted for a POSITIVELY CONFIRMED non-agent
         tuple position doesn't count as a construction event -- see
         `_is_discarded_agent_subscript` / `_find_agent_tuple_index`. Both are
         DELIBERATELY CONSERVATIVE: when either cannot positively verify its
         precondition, they do NOT suppress (a missed fire is the worse failure
         direction for an evidence-integrity lint).
    Fixing these cleared 7 of the 18 (418j, 418k, 785, 785a, 787, 804, 805) --
    all seven were already independently triaged immaterial before the fix, so
    this is a lint-precision improvement, not a materiality reversal. Backlog
    pin is now 11 (`tests/contracts/test_agent_construction_seed_order_lint.py`).

    PREVENTION for new scripts: `experiments/_lib/arm_fingerprint.seeded_construct
    (seed, factory)` calls `reset_all_rng(seed)` THEN `factory()`, guaranteeing correct
    order BY CONSTRUCTION rather than by discipline -- use it (or seed via
    `arm_cell`/`reset_all_rng`/`torch.manual_seed` textually before the agent
    constructor call) for any new driver.

    Opt-out: AGENT_SEED_ORDER_EXEMPT = "<reason>" -- appropriate when the flagged
    function's agent is deliberately NOT meant to be seed-reproducible (e.g. an
    intentionally-shared, order-independent template whose weights the criteria never
    depend on), or when a single-arm design makes cross-seed weight variation
    immaterial by construction. Do NOT retro-edit a LANDED driver to silence this --
    the exemption is for NEW work; a landed script's genuine backlog membership is
    part of the historical record (see CONFIRMED CARRIERS above).

    WARN-only in BOTH modes -- never hardens under --paths. Interprocedural
    reproducibility (does ANY caller seed before invoking a function with no local
    seed evidence?) is out of scope by design (see TIER 1 ONLY above), so this can
    under-fire; it does not over-fire on the one-hop dual-purpose-helper shape (see
    above). The corpus fire count is a BACKLOG SIZE, not a target -- landed carriers'
    runs are complete and are deliberately not retro-edited.
    """
    try:
        src = path.read_text(encoding="utf-8")
        tree = ast.parse(src, filename=str(path))
    except (OSError, SyntaxError, UnicodeDecodeError):
        return None  # check_script already reports unreadable / syntax errors

    if _AGENT_SEED_ORDER_EXEMPT_MARKER in src:
        return None

    module_funcs = {n.name: n for n in tree.body
                    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    if not module_funcs:
        return None

    walks = {}
    for name, fn in module_funcs.items():
        w = _DirectFlowWalker()
        for stmt in fn.body:
            w.visit(stmt)
        walks[name] = w

    # Coarse "can calling this function ever construct/seed" (branch-insensitive
    # on purpose -- see _flatten_walker).
    ctor_names, seed_names = set(), set()
    for name, w in walks.items():
        all_calls, all_arm_cell = _flatten_walker(w)
        if any(_call_name(c) == "REEAgent" for c in all_calls):
            ctor_names.add(name)
        if any(_is_direct_torch_seed_call(c) for c in all_calls) or all_arm_cell:
            seed_names.add(name)

    # Dual-purpose helpers are unresolvable from a caller's single call site --
    # see the docstring. Excluded from resolving OTHER functions' calls; still
    # independently scanned below on their own account.
    dual = ctor_names & seed_names
    resolved_ctor = ctor_names - dual
    resolved_seed = seed_names - dual

    # Which tuple position (if any) is the agent, for every one-hop ctor
    # function -- used only to positively confirm a discarded-probe subscript;
    # see _is_discarded_agent_subscript.
    ctor_agent_index = {name: _find_agent_tuple_index(module_funcs[name])
                        for name in resolved_ctor}
    parent_of = _build_parent_map(tree)

    def _scope_findings(w: "_DirectFlowWalker") -> List[Tuple[int, List[int]]]:
        """Ordering violations in `w`'s OWN direct flow, plus (recursively)
        each isolated guard-clause branch, checked INDEPENDENTLY -- see
        _DirectFlowWalker.visit_If for why isolated scopes must not be merged
        with the surrounding flow (mutually-exclusive execution paths)."""
        out: List[Tuple[int, List[int]]] = []
        agent_lines = sorted({
            c.lineno for c in w.calls
            if (_call_name(c) == "REEAgent" or _call_name(c) in resolved_ctor)
            and not _is_discarded_agent_subscript(c, parent_of, ctor_agent_index)
        })
        seed_lines = sorted({c.lineno for c in w.calls
                             if _is_direct_torch_seed_call(c) or _call_name(c) in resolved_seed}
                            | set(w.arm_cell_seed_lines))
        if agent_lines and seed_lines:
            first_agent = agent_lines[0]
            if not any(s < first_agent for s in seed_lines):
                out.append((first_agent, seed_lines))
        for sub in w.isolated_scopes:
            out.extend(_scope_findings(sub))
        return out

    findings: List[Tuple[str, int, List[int]]] = []
    for name, w in walks.items():
        for first_agent, seed_lines in _scope_findings(w):
            findings.append((name, first_agent, seed_lines))

    if not findings:
        return None

    findings.sort(key=lambda f: f[1])
    parts = [
        f"{fn}() constructs an agent at line {first_agent} with no torch-RNG seed "
        f"call earlier in its own flow (seed call(s) found at line(s) "
        f"{', '.join(str(s) for s in seeds)}, all at or after {first_agent})"
        for fn, first_agent, seeds in findings
    ]
    return (
        "agent weights are not seed-reproducible: " + "; ".join(parts) + ". "
        "torch.nn.Module weight init draws from torch's OWN global RNG, so "
        "np.random.seed/random.seed alone never covers this -- only "
        "torch.manual_seed/reset_all_rng/arm_cell(do_reset=True, default) does, and "
        "it must run BEFORE the agent is constructed. Fix with "
        "experiments/_lib/arm_fingerprint.seeded_construct(seed, factory), or move the "
        "seed call textually earlier. Exempt with "
        f"{_AGENT_SEED_ORDER_EXEMPT_MARKER} = \"<reason>\" only when this agent is "
        "deliberately not meant to be seed-reproducible."
    )


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
    ceiling_floor_warnings: List[Tuple[Path, str]] = []
    e3_stale_warnings: List[Tuple[Path, str]] = []
    e3_hold_warnings: List[Tuple[Path, str]] = []
    ao_selection_warnings: List[Tuple[Path, str]] = []
    spearman_warnings: List[Tuple[Path, str]] = []
    dead_zgoal_warnings: List[Tuple[Path, str]] = []
    hardcoded_dry_run_warnings: List[Tuple[Path, str]] = []
    emit_outcome_dry_run_warnings: List[Tuple[Path, str]] = []
    write_pack_dry_run_warnings: List[Tuple[Path, str]] = []
    dry_unreachable_criterion_warnings: List[Tuple[Path, str]] = []
    config_slice_warnings: List[Tuple[Path, str]] = []
    inert_dacc_bias_warnings: List[Tuple[Path, str]] = []
    dacc_last_bundle_warnings: List[Tuple[Path, str]] = []
    agent_seed_order_warnings: List[Tuple[Path, str]] = []
    zworld_p0_warmup_warnings: List[Tuple[Path, str]] = []
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
        if "ceiling_route_anchor_floor" in selected:
            crf = ceiling_route_anchor_floor_lint(p)
            if crf:
                # WARN-only in BOTH modes -- see ceiling_route_anchor_floor_lint() for why
                # this one never hardens under --paths (below-random is not statically
                # decidable, so it can only flag a MISSING guard).
                ceiling_floor_warnings.append((p, crf))
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
        if "action_object_selection" in selected:
            aos = action_object_selection_lint(p)
            if aos:
                # WARN-only in BOTH modes -- see action_object_selection_lint()
                # for why this one never hardens under --paths (it cannot
                # distinguish selecting-on from reporting-on the collapse).
                ao_selection_warnings.append((p, aos))
        if "dead_z_goal_stream" in selected:
            dzg = dead_z_goal_stream_lint(p)
            if dzg:
                # WARN-only in BOTH modes -- see dead_z_goal_stream_lint() for why this
                # one never hardens under --paths (a config assembled in a helper the
                # AST scan cannot follow makes it under-fire, and the landed carriers'
                # runs are complete).
                dead_zgoal_warnings.append((p, dzg))
        if "hardcoded_dry_run" in selected:
            hdr = hardcoded_dry_run_lint(p)
            if hdr:
                # WARN-only in BOTH modes -- see hardcoded_dry_run_lint() for why this
                # one never hardens under --paths (a flag threaded through a dict or a
                # partial is invisible to the static scan, and the ~164 landed carriers'
                # runs are complete, so hardening would block commits on history).
                hardcoded_dry_run_warnings.append((p, hdr))
        if "emit_outcome_dry_run" in selected:
            eod = emit_outcome_dry_run_lint(p)
            if eod:
                # WARN-only in BOTH modes -- see emit_outcome_dry_run_lint() for why this
                # one never hardens under --paths (a flag resolved in a helper the bare-name
                # fixpoint cannot follow is invisible, and the ~273 landed carriers' runs are
                # complete, so hardening would block commits on history).
                emit_outcome_dry_run_warnings.append((p, eod))
        if "write_pack_dry_run" in selected:
            wpd = write_pack_dry_run_lint(p)
            if wpd:
                # WARN-only in BOTH modes -- see write_pack_dry_run_lint() for why this one
                # never hardens under --paths (same static-scan blind spots as its two
                # siblings, and the same refusal to retro-edit landed drivers whose runs
                # are complete).
                write_pack_dry_run_warnings.append((p, wpd))
        if "dry_run_unreachable_criterion" in selected:
            duc = dry_run_unreachable_criterion_lint(p)
            if duc:
                # WARN-only in BOTH modes -- see dry_run_unreachable_criterion_lint() for
                # why this one never hardens under --paths (a bound assembled arithmetically
                # or threaded through a dict is invisible to the static scan, and the 11
                # landed carriers' runs are complete, so hardening would block commits on
                # history).
                dry_unreachable_criterion_warnings.append((p, duc))
        if "config_slice_declaration" in selected:
            csd = config_slice_under_declaration_lint(p)
            if csd:
                # WARN-only in BOTH modes -- see config_slice_under_declaration_lint()
                # for why this one never hardens under --paths (best-effort static scan
                # in both directions, a false HIT needs a CONSUMER to exist before it
                # can corrupt anything, and the landed carriers' runs are complete).
                config_slice_warnings.append((p, csd))
        if "inert_salience_dacc_bias" in selected:
            isd = inert_salience_dacc_bias_lint(p)
            if isd:
                # WARN-only in BOTH modes -- see inert_salience_dacc_bias_lint() for why
                # this one never hardens under --paths (a dacc_weight assembled at runtime
                # is invisible to the static scan, and the landed carriers' runs are
                # complete).
                inert_dacc_bias_warnings.append((p, isd))
        if "dacc_last_bundle" in selected:
            dlb = dacc_last_bundle_lint(p)
            if dlb:
                # WARN-only in BOTH modes -- see dacc_last_bundle_lint() for why this one
                # never hardens under --paths (the 15 landed carrier drivers' runs are
                # complete and are deliberately not retro-edited).
                dacc_last_bundle_warnings.append((p, dlb))
        if "spearman_guard_shape" in selected:
            sps = spearman_guard_shape_lint(p)
            if sps:
                # WARN-only in BOTH modes -- see spearman_guard_shape_lint() for why
                # this one never hardens under --paths (static shape scan; flags a
                # SUSPECTED defective helper, not a proven one).
                spearman_warnings.append((p, sps))
        if "agent_seed_order" in selected:
            aso = agent_construction_before_seed_lint(p)
            if aso:
                # WARN-only in BOTH modes -- see agent_construction_before_seed_lint()
                # for why this one never hardens under --paths (interprocedural
                # reproducibility is out of scope by design, and the landed carriers'
                # runs are complete).
                agent_seed_order_warnings.append((p, aso))
        if "zworld_p0_warmup" in selected:
            zpw = zworld_p0_warmup_lint(p)
            if zpw:
                # WARN-only in BOTH modes -- see zworld_p0_warmup_lint() for why this one
                # never hardens under --paths (a kwarg threaded through a dict or a helper
                # this scan cannot follow is invisible, and a landed carrier's run is
                # complete, so hardening would block commits on history).
                zworld_p0_warmup_warnings.append((p, zpw))

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
          f"{len(ceiling_floor_warnings)} ceiling-route-anchor-floor-warning(s), "
          f"{len(e3_stale_warnings)} stale-e3-diagnostics-warning(s), "
          f"{len(e3_hold_warnings)} hold-weighted-readout-warning(s), "
          f"{len(ao_selection_warnings)} action-object-selection-warning(s), "
          f"{len(spearman_warnings)} spearman-guard-shape-warning(s), "
          f"{len(dead_zgoal_warnings)} dead-z_goal-stream-warning(s), "
          f"{len(hardcoded_dry_run_warnings)} hardcoded-dry_run-warning(s), "
          f"{len(emit_outcome_dry_run_warnings)} emit_outcome-dry_run-warning(s), "
          f"{len(write_pack_dry_run_warnings)} write_pack-dry_run-warning(s), "
          f"{len(dry_unreachable_criterion_warnings)} dry_run-unreachable-criterion-warning(s), "
          f"{len(config_slice_warnings)} config_slice-declaration-warning(s), "
          f"{len(inert_dacc_bias_warnings)} inert-salience-dacc_bias-warning(s), "
          f"{len(dacc_last_bundle_warnings)} dacc-_last_bundle-warning(s), "
          f"{len(agent_seed_order_warnings)} agent-seed-order-warning(s), "
          f"{len(zworld_p0_warmup_warnings)} zworld_p0-warmup-warning(s)", flush=True)
    if zworld_p0_warmup_warnings:
        # Advisory in BOTH modes (never hardens). A fire here means the driver calls
        # _train_all_on_agent but never passes zworld_p0_episodes anywhere in the file, so
        # the SD-070 z_world-encoder warmup never runs and z_world stays a frozen random
        # projection for the whole run -- confirmed twice (V3-EXQ-728, V3-EXQ-875) to
        # silently produce a floor-pinned, non-degenerate-looking-until-checked result.
        # Triage each: pass zworld_p0_episodes=<n> (60 is the validated operating point) on
        # at least the first (acquisition) call per shared agent, or carry
        # ZWORLD_P0_WARMUP_EXEMPT when the frozen encoder is deliberate. Do NOT retro-edit a
        # LANDED driver whose run is complete -- queue a superseding EXQ letter instead.
        print("", flush=True)
        print("[validate_experiments] ZWORLD_P0-WARMUP WARNINGS (advisory, non-blocking):", flush=True)
        for p, warn in zworld_p0_warmup_warnings:
            rel = p.relative_to(REPO_ROOT) if REPO_ROOT in p.parents or p == REPO_ROOT else p
            print(f"  - {rel}: {warn}", flush=True)
    if agent_seed_order_warnings:
        # Advisory in BOTH modes (never hardens). A fire here means the flagged
        # function constructs REEAgent before ever seeding torch's global RNG in its
        # own flow, so that agent's initial weights depend on process-level torch RNG
        # history, not on `seed` -- see agent_construction_before_seed_lint() for the
        # full reasoning and the confirmed Q-081 / INV-091 carrier shape. Triage each:
        # confirm (as for Q-081) whether the script's own comparison is arm-matched
        # via copy.deepcopy of the one shared template (immaterial to that finding) or
        # whether cross-seed/cross-run weight reproducibility is actually load-bearing
        # (matters). Fix new work with experiments/_lib/arm_fingerprint.seeded_construct.
        # Do NOT retro-edit a LANDED driver whose run is complete.
        print("", flush=True)
        print("[validate_experiments] AGENT-SEED-ORDER WARNINGS (advisory, non-blocking):", flush=True)
        for p, warn in agent_seed_order_warnings:
            rel = p.relative_to(REPO_ROOT) if REPO_ROOT in p.parents or p == REPO_ROOT else p
            print(f"  - {rel}: {warn}", flush=True)
    if dacc_last_bundle_warnings:
        # Advisory in BOTH modes (never hardens). A fire here means the script reads a
        # `_last_bundle` attribute NO substrate object defines, so the read is None every
        # tick and any derived dACC readout is a STRUCTURAL zero that looks measured.
        # Triage each: read `getattr(agent, "_dacc_last_bundle", None)` instead
        # (v3_exq_687a_...::_dacc_diag is the canonical shape), and at the same site
        # replace any `bundle.get("mode_ev") or ...` with an explicit `if sb is None:`
        # fallback -- mode_ev is a [K] tensor and `or` raises on it. Do NOT retro-edit a
        # LANDED driver whose run is complete; record it for governance instead.
        print("", flush=True)
        print("[validate_experiments] DACC-_LAST_BUNDLE WARNINGS (advisory, non-blocking):", flush=True)
        for p, warn in dacc_last_bundle_warnings:
            rel = p.relative_to(REPO_ROOT) if REPO_ROOT in p.parents or p == REPO_ROOT else p
            print(f"  - {rel}: {warn}", flush=True)
    if inert_dacc_bias_warnings:
        # Advisory in BOTH modes (never hardens). A fire here means the script enables
        # salience_apply_to_dacc_bias=True with no positive dacc_weight, so the
        # write_gate("e3_policy") modulation in agent.py scales the ZERO vector -- the
        # dACC->E3 behavioural channel is inert and any arm resting on it is a null that
        # looks measured. Triage each: set dacc_weight > 0 plus a per-candidate sub-weight
        # (V3-EXQ-799 is the canonical shape), or carry INERT_SALIENCE_DACC_BIAS_EXEMPT
        # when the bias is deliberately driven from another head. Do NOT retro-edit a
        # LANDED driver whose run is complete.
        print("", flush=True)
        print("[validate_experiments] INERT-SALIENCE-DACC_BIAS WARNINGS (advisory, non-blocking):", flush=True)
        for p, warn in inert_dacc_bias_warnings:
            rel = p.relative_to(REPO_ROOT) if REPO_ROOT in p.parents or p == REPO_ROOT else p
            print(f"  - {rel}: {warn}", flush=True)
    if config_slice_warnings:
        # Advisory in BOTH modes (never hardens). A fire here means the script mints a
        # CROSS-DRIVER-reusable arm cell whose config_slice under-approximates what the
        # cell actually reads, so a future consumer can take a false cache HIT and read
        # readouts computed under a different scheme. Triage each: add the readout-
        # affecting constant to the slice (V3-EXQ-798a is the canonical shape), or mark
        # CONFIG_SLICE_DECLARATION_EXEMPT when it is genuinely bound by a key already
        # declared. Do NOT retro-edit a LANDED driver whose run is complete -- a
        # completed run's pre-registered emission is not rewritten; fix the successor.
        print("", flush=True)
        print("[validate_experiments] CONFIG_SLICE-DECLARATION WARNINGS (advisory, non-blocking):", flush=True)
        for p, warn in config_slice_warnings:
            rel = p.relative_to(REPO_ROOT) if REPO_ROOT in p.parents or p == REPO_ROOT else p
            print(f"  - {rel}: {warn}", flush=True)
    if dry_unreachable_criterion_warnings:
        # Advisory in BOTH modes (never hardens). A fire here means the driver's --dry-run
        # smoke reports a criterion it cannot possibly evaluate: the detector is gated on an
        # absolute mid-training episode index the reduced run never reaches, so its latch is
        # hardcoded false whatever the policy did, and that false goes into the manifest
        # looking like a measurement. Triage each: scale the gate with the actual episode
        # count, or exclude the criterion from the smoke explicitly. A driver whose latch is
        # genuinely not reported should carry DRY_RUN_UNREACHABLE_CRITERION_EXEMPT rather
        # than be left to re-fire. Do NOT retro-edit a LANDED driver whose run is complete --
        # adjudicate the affected RESULT instead (that is what the /failure-autopsy Step 2a
        # dry-run guard is for).
        print("", flush=True)
        print("[validate_experiments] DRY_RUN-UNREACHABLE-CRITERION WARNINGS (advisory, non-blocking):", flush=True)
        for p, warn in dry_unreachable_criterion_warnings:
            rel = p.relative_to(REPO_ROOT) if REPO_ROOT in p.parents or p == REPO_ROOT else p
            print(f"  - {rel}: {warn}", flush=True)
    if write_pack_dry_run_warnings:
        # Advisory in BOTH modes (never hardens). A fire here means the driver already
        # threads its flag into write_flat_manifest/emit_outcome but calls write_pack
        # unthreaded, so `--dry-run` emits an UNFLAGGED run pack -- and the pack, not the
        # flat manifest, is what build_experiment_indexes scores. Such a pack is excluded
        # only by _load_dry_run_run_ids' cross-file carry by run_id, the silent fallback
        # the 2026-07-28 work set out to dissolve. Triage each: thread the flag
        # (`dry_run=bool(args.dry_run)`), which makes the pack self-identify. A caller that
        # already relocates or deletes the pack should carry WRITE_PACK_DRY_RUN_EXEMPT
        # rather than be left to re-fire. Do NOT retro-edit a LANDED driver whose run is
        # complete.
        print("", flush=True)
        print("[validate_experiments] WRITE_PACK-DRY_RUN WARNINGS (advisory, non-blocking):", flush=True)
        for p, warn in write_pack_dry_run_warnings:
            rel = p.relative_to(REPO_ROOT) if REPO_ROOT in p.parents or p == REPO_ROOT else p
            print(f"  - {rel}: {warn}", flush=True)
    if emit_outcome_dry_run_warnings:
        # Advisory in BOTH modes (never hardens). A fire here means `--dry-run` on that
        # driver leaves its 1-seed smoke manifest sitting in evidence/experiments/, where
        # build_experiment_indexes.py scores it into claim_evidence.v1.json against a real
        # claim -- the `_dry_` prefix does not exempt it, and pending_review.md hides the
        # problem because IT does exclude dry_run manifests. Triage each: thread the
        # script's own flag (V3-EXQ-825 is the canonical shape). A caller that already
        # relocates or deletes the manifest should carry EMIT_OUTCOME_DRY_RUN_EXEMPT rather
        # than be left to re-fire. Do NOT retro-edit a LANDED driver whose run is complete.
        print("", flush=True)
        print("[validate_experiments] EMIT_OUTCOME-DRY_RUN WARNINGS (advisory, non-blocking):", flush=True)
        for p, warn in emit_outcome_dry_run_warnings:
            rel = p.relative_to(REPO_ROOT) if REPO_ROOT in p.parents or p == REPO_ROOT else p
            print(f"  - {rel}: {warn}", flush=True)
    if hardcoded_dry_run_warnings:
        # Advisory in BOTH modes (never hardens). A fire here means `--dry-run` on that
        # driver writes a real-looking manifest into the indexer's scoring set AND
        # suppresses the [smoke] z_goal_stream liveness report. Triage each: the fix is
        # to thread the script's own flag (V3-EXQ-722 is the canonical shape). A literal
        # that is deliberate -- the caller already relocated or renamed the output --
        # should carry HARDCODED_DRY_RUN_EXEMPT rather than be left to re-fire.
        print("", flush=True)
        print("[validate_experiments] HARDCODED-DRY_RUN WARNINGS (advisory, non-blocking):", flush=True)
        for p, warn in hardcoded_dry_run_warnings:
            rel = p.relative_to(REPO_ROOT) if REPO_ROOT in p.parents or p == REPO_ROOT else p
            print(f"  - {rel}: {warn}", flush=True)
    if dead_zgoal_warnings:
        # Advisory in BOTH modes (never hardens). A fire here means the script declares a
        # z_goal-dependent config but nothing ever writes z_goal, so every goal-gated
        # branch silently no-ops for the whole run. Triage each: a deliberate zero-goal
        # condition (goal-OFF parity arm, negative control) is correct and should carry
        # DEAD_Z_GOAL_STREAM_EXEMPT; anything else is a wiring defect. Do NOT reflexively
        # bolt update_z_goal onto a LANDED script -- it is also the SD-024 benefit-terrain
        # producer, so the retrofit changes behaviour (see the lint docstring).
        print("", flush=True)
        print("[validate_experiments] DEAD-Z_GOAL-STREAM WARNINGS (advisory, non-blocking):", flush=True)
        for p, warn in dead_zgoal_warnings:
            rel = p.relative_to(REPO_ROOT) if REPO_ROOT in p.parents or p == REPO_ROOT else p
            print(f"  - {rel}: {warn}", flush=True)
    if spearman_warnings:
        # Advisory in BOTH modes (never hardens). A fire here means a hand-rolled rank
        # correlation may emit a phantom |rho| on a constant input (the SD-081 defect):
        # the fix is to import experiments/_lib/stats.spearman. Triage each -- an
        # average-rank helper or an input-guarded one is safe and should not appear here.
        print("", flush=True)
        print("[validate_experiments] SPEARMAN-GUARD-SHAPE WARNINGS (advisory, non-blocking):", flush=True)
        for p, warn in spearman_warnings:
            rel = p.relative_to(REPO_ROOT) if REPO_ROOT in p.parents or p == REPO_ROOT else p
            print(f"  - {rel}: {warn}", flush=True)
    if ao_selection_warnings:
        # Advisory in BOTH modes (never hardens). A fire here means the driver's
        # action stream may be INVARIANT under its own manipulation -- i.e. the
        # arms can come out bit-identical. Triage each: selecting through the
        # round trip is a defect, merely reporting on it is not.
        print("", flush=True)
        print("[validate_experiments] ACTION-OBJECT-SELECTION WARNINGS (advisory, non-blocking):", flush=True)
        for p, warn in ao_selection_warnings:
            rel = p.relative_to(REPO_ROOT) if REPO_ROOT in p.parents or p == REPO_ROOT else p
            print(f"  - {rel}: {warn}", flush=True)
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
    if ceiling_floor_warnings:
        # Advisory in BOTH modes (never hardens -- whether a cell's score is below its own
        # random_walk anchor is computed from live run data, so this flags a MISSING guard,
        # never a proven-unsupported verdict). A fire here means the driver self-routes to a
        # ceiling-class label with a random_walk floor but never wires
        # anchor_floor_guard.refuse_ceiling_below_random, so a ceiling verdict resting on a
        # sub-random learner would be emitted rather than refused (autopsy sec 5.2). Triage
        # each: wire the guard (v3_exq_734 is the reference), or mark
        # CEILING_ANCHOR_FLOOR_EXEMPT when the ceiling route cannot rest on a sub-random
        # learner (the 737/742 recorded_preconditions case). Do NOT retro-edit a LANDED
        # driver whose run is complete -- adjudicate the affected RESULT instead.
        print("", flush=True)
        print("[validate_experiments] CEILING-ROUTE-ANCHOR-FLOOR WARNINGS (advisory, non-blocking):", flush=True)
        for p, warn in ceiling_floor_warnings:
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
