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

# Selectable checks for --checks. Default (None) runs all of them. A caller that
# wants only one gate -- e.g. the commit-time manifest-writer gate in
# scripts/precommit_contracts.sh -- passes `--checks manifest_writer`, which keeps
# that gate surgical: it does NOT expand the emit_outcome conformance / degeneracy /
# arm-fingerprint contracts onto the broader (non-v3_exq_) script set the gate scopes.
CHECK_NAMES = ("conformance", "readiness", "arm_fingerprint", "degeneracy", "manifest_writer",
               "anchor_reachability", "precondition_recomputability")

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

    Opt-out: ANCHOR_REACHABILITY_EXEMPT = "<reason>" -- appropriate when the predicate
    IS the degeneracy definition (an exact-equality / structural reproduction check),
    so reachability holds by construction and a replay would be tautological.

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
            "Exempt with ANCHOR_REACHABILITY_EXEMPT = \"<reason>\" when the predicate IS "
            "the degeneracy definition. See experiments/_lib/readiness_anchor.py + "
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
      (e) checks a TWO-SIDED SATURATION BAND against only ONE partition of a row
          collection while SIBLING partitions of that same collection exist unchecked --
          so the readout is guaranteed to have room to move on the arm that was measured
          and is entirely unguarded on the arms that carry the manipulation
          (V3-EXQ-779b baseline_entropy_headroom; autopsy 2026-07-19 section 7).

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
        # Four conjuncts keep it narrow:
        #   1. the resolved `met` is a genuine two-sided band (a one-sided floor is
        #      not a saturation guard and must never fire),
        #   2. it ranges over a name that is a single-condition filtered subset of a
        #      bare-Name source collection,
        #   3. that same source has at least one OTHER subset with a DIFFERENT
        #      condition -- i.e. sibling partitions demonstrably exist, and
        #   4. `met` does not also reference the unfiltered source directly, which
        #      would mean the band already covers every row.
        met_node_e = fields.get("met")
        if met_node_e is not None:
            resolved_e = _resolve_one_level(met_node_e, tree)
            if _is_two_sided(resolved_e):
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
            + " check a TWO-SIDED SATURATION BAND against only ONE partition of the row "
              "collection while SIBLING partitions of that same collection exist "
              "unchecked. A headroom band certifies that the readout can still MOVE -- "
              "but the MANIPULATION is what pushes it toward a bound, so scoping the band "
              "to the baseline partition inspects the arm LEAST likely to saturate and "
              "leaves the effect-carrying arms entirely unguarded. V3-EXQ-779b "
              "baseline_entropy_headroom is the worked case: it ranged over "
              "`baseline_rows` (arm == T0P0) while `t1_rows` / `p1_rows` were never "
              "band-checked, so seed 23 reported met=True at baseline 0.6093 with its "
              "tonic-ON arms at 0.8489 / 0.8587 against E_SAT_HIGH = 0.98 -- an "
              "unguarded near-ceiling exposure that surfaced only in autopsy. FIX: do NOT "
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
    recomput_warnings: List[Tuple[Path, str]] = []
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
                anchor_warnings.append((p, anch))
        if "precondition_recomputability" in selected:
            rec = precondition_recomputability_lint(p)
            if rec:
                # WARN-only in BOTH modes -- see precondition_recomputability_lint()
                # for why this one never hardens under --paths.
                recomput_warnings.append((p, rec))

    print("", flush=True)
    print(f"[validate_experiments] checked {len(paths)} scripts: "
          f"{n_ok} OK, {n_exempt} exempt, {len(failures)} non-conforming, "
          f"{len(warnings)} readiness-warning(s), "
          f"{len(arm_fp_warnings)} arm-fingerprint-backlog, "
          f"{len(degen_warnings)} degeneracy-self-report-backlog, "
          f"{len(manifest_writer_warnings)} manifest-writer-backlog, "
          f"{len(anchor_warnings)} anchor-reachability-warning(s), "
          f"{len(recomput_warnings)} precondition-recomputability-warning(s)", flush=True)
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
