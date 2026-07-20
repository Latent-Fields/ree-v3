"""
inert_arm_knob -- manifest-local lint for a declared-distinct arm pair that ran identically.

Recommended by:
  REE_assembly/evidence/planning/failure_autopsy_V3-EXQ-689d-D3_2026-07-20.md sec 7 item 4
  REE_assembly/evidence/planning/intra_run_substrate_divergence_sweep_2026-07-20.md
      sec 8(a) correction (which prefers THIS lint over the whole-glob divergence check:
      that one would have fired 42 times with 42 false positives).

WHAT IT CATCHES
---------------
V3-EXQ-689d declared four arms. `ARM_PROPOSER_CTRL` and `ARM_MATCHED_NOISE` were
bit-identical on 26 of 27 recorded per-cell fields on ALL THREE seeds -- including
`n_p1_ticks` 387/3616/224, i.e. identical trajectories -- differing only in the
`temperature` field that NAMED their intended difference (1.0 vs 2.5).
`MATCHED_ENTROPY_TEMPERATURE = 2.5` was folded into `arm_fingerprint` but never reached a
sampling step, because `candidate_summary_source='proposer'` resolves by deterministic
argmin. The knob was inert; the two arms were the same arm.

The consequence is the reason this is worth a gate. The "NOT noise-as-diversity" half of
the conjunctive C_PRIMARY tested nothing, and the run PASSED -- because "strict above BOTH
X and Y" degrades SILENTLY to "strict above X" when X == Y. A conjunctive acceptance
criterion does not announce that one of its conjuncts became vacuous; it just gets easier.

WHY THE ARM FINGERPRINT CANNOT SEE THIS. The knob enters `config_slice`, so the two cells'
`arm_fingerprint` values DIFFER (`13ef7296...` vs `a0bdf349...`) while every readout is
byte-equal. The fingerprint asserts "declared distinct"; the readouts prove "not distinct".
That contradiction IS the signal, and it is only visible by comparing recorded values --
which is exactly why this lint lives here and not in `validate_experiments.py` (an
AST lint over driver source, which cannot see what a run produced).

POSTURE: RECORD-AND-WARN AT WRITE, GATE AT ADJUDICATION
-------------------------------------------------------
Never a hard failure. By manifest-write time the compute is spent, and refusing to write
would destroy an expensive run over a defect that is sometimes survivable (the other three
689d arms were fine, and D1/D3 of that same run were adjudicated non-fatal). Same posture
both E3 lints in `validate_experiments.py` take, and the posture sec 8(a) explicitly
requires. Every entry point is wrapped so a lint bug can never crash an experiment.

WHAT IT EMITS (convention copied from `substrate_stable_across_run`)
--------------------------------------------------------------------
  arm_knobs_effective : bool -- False iff some declared-distinct arm pair ran identically.
                        Assigned directly, not via _fill: a meaningful False is the whole
                        point and `_is_empty` treats False as present.
  inert_arm_knob_detail : dict -- emitted ONLY on the False verdict, offenders only (same
                        shape discipline as `degenerate_metrics` in experiments/_metrics.py).

Deliberately NOT added to `ALWAYS_CORE_KEYS`: the pre-2026-07-20 corpus cannot carry it,
and making it core would turn every legacy manifest into a WARN.

HOW "DECLARED DISTINCT" AND "THE KNOB" ARE DERIVED
---------------------------------------------------
Neither source document settles this, and `config_slice` -- the one place that literally
records an arm's distinguishing knobs -- is hashed into the fingerprint and then DISCARDED
(arm_fingerprint.py:617-638). The lint therefore requires an AUTHORITATIVE declaration and
uses exactly one source:

  `config["arms"]` -- the arm spec list. Its non-identity keys ARE the declared knobs,
  whether or not they vary. 689d has it; 48 of the 100 multi-arm manifests in the
  2026-07-20 corpus have it.

WHY THE INFERRED PATH DOES NOT FIRE (this was measured, not assumed). The tempting
fallback -- "a cell-root key constant WITHIN every arm and varying ACROSS arms is a knob"
-- was built, run over the 648-manifest corpus, and REJECTED. It fired 5 times, of which
4 were false positives (590c, 603j, 661, and arguably 768). The mechanism is a
self-fulfilling exclusion: a READOUT that happens to be constant within each arm is
classified as a knob, gets EXCLUDED from the comparison, and the lint then reports the
remaining fields as identical -- manufacturing the very identity it claims to have found.
603j is the clean illustration: `mean_fed_safety_signal` 0.0 vs 0.89 and `n_safety_credit`
0 vs 80 are READOUTS proving the arms diverged loudly, yet the inference calls them knobs
and reports the pair as inert. The degenerate case is a single-seed run (590c), where
"constant within arm" is vacuously true of every field.

This is the sweep correction's own lesson applied to this lint: a warning that is usually
wrong gets ignored, and takes the real case with it. `_inferred_knobs` is retained for
diagnosis but is not a firing path.

The cost is an honest, stated FALSE NEGATIVE: a multi-arm manifest with no `config["arms"]`
is out of scope and reports `checked: False` with a reason, rather than guessing. On the
corpus that is 52 of 100 multi-arm manifests. Widening reach means teaching drivers to
declare `config["arms"]`, not weakening the knob source.

LIMITS (state them, do not paper over them)
--------------------------------------------
  * Needs `config["arms"]`, >= 2 arms, and >= 1 shared seed with recorded per-cell fields.
    A run that records only aggregates, or declares no arm spec, is out of reach (see the
    derivation note above -- this is a deliberate false negative, not an oversight).
  * A pair differing on a knob AND on readouts is silent even if the knob is partly inert
    -- this detects TOTAL inertness only. Partial inertness is not manifest-detectable.
  * Two arms legitimately expected to coincide (a duplicate control, a null arm) will
    fire. That is a true positive about the manifest's DECLARATION -- if two arms are
    meant to be the same arm, they should not be declared distinct. Suppress with
    `INERT_ARM_KNOB_EXPECTED_IDENTICAL` (see below) rather than by weakening the lint.
  * Cells carrying a non-null `error_note` are skipped: a crashed cell has no meaningful
    readouts and two crashed cells look identical for uninteresting reasons.

ASCII-only output (repo rule). Stdlib only.
"""

from __future__ import annotations

import json
import sys
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

# Cell-root keys that are identity/provenance, never a readout and never a knob. These
# are excluded from the comparison on BOTH sides.
IDENTITY_KEYS: Tuple[str, ...] = ("arm_id", "label", "seed", "arm_fingerprint")

# Keys naming an arm in a `config["arms"]` spec entry -- identity, not a knob.
ARM_SPEC_IDENTITY_KEYS: Tuple[str, ...] = ("arm_id", "label", "name", "id")

# A pair must share at least this many comparable fields before identity means anything.
# Two cells agreeing on one bookkeeping integer is not evidence of a shared trajectory.
MIN_COMPARED_FIELDS = 3

# Opt-out marker. A manifest may set `manifest["inert_arm_knob_expected_identical"]` to a
# list of arm-id pairs (["ARM_A", "ARM_B"]) that are KNOWN to coincide by design. Named
# rather than boolean on purpose: a blanket off-switch would silence the real case too.
EXPECTED_IDENTICAL_KEY = "inert_arm_knob_expected_identical"

WARN_PREFIX = "[INERT-ARM-KNOB-WARNING]"


def _canon(value: Any) -> str:
    """Canonical, order-insensitive rendering of a recorded value for bit-comparison.

    `json.dumps(sort_keys=True)` so a nested dict (689d's `selected_class_counts`) compares
    by content rather than by insertion order. Falls back to repr for anything unserialisable
    -- a lint must never raise on a manifest it does not understand.
    """
    try:
        return json.dumps(value, sort_keys=True, default=repr)
    except Exception:
        return repr(value)


def _cells_by_arm(arm_results: Sequence[Any]) -> Dict[str, List[Mapping[str, Any]]]:
    """Group per-cell dicts by arm_id, skipping malformed and errored cells."""
    grouped: Dict[str, List[Mapping[str, Any]]] = {}
    for cell in arm_results:
        if not isinstance(cell, Mapping):
            continue
        arm_id = cell.get("arm_id")
        if not isinstance(arm_id, str) or not arm_id:
            continue
        # A crashed cell has no meaningful readouts -- see LIMITS.
        if cell.get("error_note") not in (None, "", False):
            continue
        grouped.setdefault(arm_id, []).append(cell)
    return grouped


def _declared_knobs_from_config(config: Any) -> Set[str]:
    """Knob names from `config["arms"]` -- the authoritative source when present.

    Every non-identity key on an arm spec entry is a declared knob, whether or not it
    actually varies: an author who wrote it into the spec was naming the axis.
    """
    knobs: Set[str] = set()
    if not isinstance(config, Mapping):
        return knobs
    arms = config.get("arms")
    if not isinstance(arms, (list, tuple)):
        return knobs
    for spec in arms:
        if not isinstance(spec, Mapping):
            continue
        for key in spec:
            if isinstance(key, str) and key not in ARM_SPEC_IDENTITY_KEYS:
                knobs.add(key)
    return knobs


def _inferred_knobs(grouped: Mapping[str, List[Mapping[str, Any]]]) -> Set[str]:
    """Knob derivation with NO declaration available: constant within every arm, varying
    across arms.

    RETAINED FOR DIAGNOSIS, NOT USED TO FIRE -- see `_declared_knobs_from_config` and the
    "WHY THE INFERRED PATH DOES NOT FIRE" note in the module docstring. Measured on the
    648-manifest corpus it produced 5 fires of which 4 were false positives, because a
    READOUT that happens to be constant within each arm gets classified as a knob and then
    EXCLUDED from the comparison -- manufacturing the very identity the lint reports. The
    degenerate case is a single-seed run, where "constant within arm" is vacuously true of
    every field.
    """
    all_keys: Set[str] = set()
    for cells in grouped.values():
        for cell in cells:
            all_keys.update(k for k in cell if isinstance(k, str))
    all_keys -= set(IDENTITY_KEYS)

    knobs: Set[str] = set()
    for key in all_keys:
        per_arm_values: List[str] = []
        constant_within_every_arm = True
        for cells in grouped.values():
            rendered = {_canon(c.get(key)) for c in cells}
            if len(rendered) != 1:
                constant_within_every_arm = False
                break
            per_arm_values.append(next(iter(rendered)))
        if constant_within_every_arm and len(set(per_arm_values)) > 1:
            knobs.add(key)
    return knobs


def _expected_identical_pairs(manifest: Mapping[str, Any]) -> Set[frozenset]:
    """Author-declared pairs that are known to coincide by design."""
    pairs: Set[frozenset] = set()
    declared = manifest.get(EXPECTED_IDENTICAL_KEY)
    if not isinstance(declared, (list, tuple)):
        return pairs
    for entry in declared:
        if isinstance(entry, (list, tuple)) and len(entry) == 2:
            a, b = entry
            if isinstance(a, str) and isinstance(b, str):
                pairs.add(frozenset((a, b)))
    return pairs


def _compare_pair(cells_a: Sequence[Mapping[str, Any]],
                  cells_b: Sequence[Mapping[str, Any]],
                  knobs: Set[str]) -> Optional[Dict[str, Any]]:
    """Compare one arm pair at every shared seed. Returns a finding dict, or None.

    Fires only when the pair is (i) declared distinct -- differs on at least one knob,
    (ii) compared on at least MIN_COMPARED_FIELDS non-knob fields at each shared seed,
    and (iii) bit-identical on every one of them at EVERY shared seed. One seed diverging
    is enough to clear the pair: the knob demonstrably did something.
    """
    by_seed_a = {c.get("seed"): c for c in cells_a}
    by_seed_b = {c.get("seed"): c for c in cells_b}
    shared_seeds = [s for s in by_seed_a if s in by_seed_b]
    if not shared_seeds:
        return None

    differing_knobs: Dict[str, List[Any]] = {}
    compared_fields: Set[str] = set()
    seeds_checked: List[Any] = []

    for seed in shared_seeds:
        cell_a, cell_b = by_seed_a[seed], by_seed_b[seed]
        common = (set(cell_a) & set(cell_b)) - set(IDENTITY_KEYS)
        pair_compared = common - knobs
        if len(pair_compared) < MIN_COMPARED_FIELDS:
            continue
        for key in pair_compared:
            if _canon(cell_a.get(key)) != _canon(cell_b.get(key)):
                return None  # a real difference -- the arms diverged, nothing to report
        for key in common & knobs:
            if _canon(cell_a.get(key)) != _canon(cell_b.get(key)):
                differing_knobs[key] = [cell_a.get(key), cell_b.get(key)]
        compared_fields |= pair_compared
        seeds_checked.append(seed)

    if not seeds_checked or not differing_knobs:
        # No knob differs -> the pair was never DECLARED distinct on any recorded field,
        # so identity is unremarkable and this lint has nothing to say about it.
        return None

    return {
        "arm_ids": None,  # filled by the caller, which knows the pair
        "differing_knobs": differing_knobs,
        "seeds_compared": sorted(seeds_checked, key=lambda s: (s is None, str(s))),
        "n_fields_compared": len(compared_fields),
        "identical_fields": sorted(compared_fields),
    }


def check_inert_arm_knob(manifest: Mapping[str, Any]) -> Dict[str, Any]:
    """Manifest-local lint. Returns a report dict; NEVER raises.

    Report shape:
        {"checked": bool,            # False when the manifest is out of scope
         "arm_knobs_effective": bool,
         "findings": [ ... ],        # offenders only
         "reason": str}              # why it was not checked, when checked is False
    """
    try:
        arm_results = manifest.get("arm_results")
        if not isinstance(arm_results, (list, tuple)) or not arm_results:
            return {"checked": False, "arm_knobs_effective": True, "findings": [],
                    "reason": "no arm_results"}

        grouped = _cells_by_arm(arm_results)
        if len(grouped) < 2:
            return {"checked": False, "arm_knobs_effective": True, "findings": [],
                    "reason": "fewer than 2 usable arms"}

        knobs = _declared_knobs_from_config(manifest.get("config"))
        knob_source = "config.arms"
        if not knobs:
            # No authoritative declaration -> OUT OF SCOPE, not "infer and fire". See the
            # module docstring: inferring knobs from cell values cannot separate a
            # declaration from an outcome, and excluding a misclassified readout
            # manufactures the identity the lint would then report.
            return {"checked": False, "arm_knobs_effective": True, "findings": [],
                    "reason": "no config['arms'] declaration -- knobs not authoritative"}

        expected = _expected_identical_pairs(manifest)

        findings: List[Dict[str, Any]] = []
        arm_ids = sorted(grouped)
        for i, arm_a in enumerate(arm_ids):
            for arm_b in arm_ids[i + 1:]:
                if frozenset((arm_a, arm_b)) in expected:
                    continue
                finding = _compare_pair(grouped[arm_a], grouped[arm_b], knobs)
                if finding is not None:
                    finding["arm_ids"] = [arm_a, arm_b]
                    finding["knob_source"] = knob_source
                    findings.append(finding)

        return {"checked": True, "arm_knobs_effective": not findings,
                "findings": findings, "reason": ""}
    except Exception as exc:  # pragma: no cover - defensive; a lint must not crash a run
        return {"checked": False, "arm_knobs_effective": True, "findings": [],
                "reason": "lint error: %s" % (exc,)}


def format_warning(findings: Sequence[Mapping[str, Any]]) -> str:
    """One ASCII line per offending pair, naming the inert knob and the identical fields."""
    lines: List[str] = []
    for f in findings:
        arms = f.get("arm_ids") or ["?", "?"]
        knobs = f.get("differing_knobs") or {}
        knob_desc = ", ".join(
            "%s=%s vs %s" % (k, _canon(v[0]), _canon(v[1])) for k, v in sorted(knobs.items())
        )
        lines.append(
            "arms %s and %s are declared distinct (%s) but are bit-identical on all %d "
            "recorded per-cell fields at seeds %s. The knob is INERT: these are the same "
            "arm, and any conjunctive acceptance criterion spanning them has silently "
            "degraded to its other conjunct." % (
                arms[0], arms[1], knob_desc or "(unknown knob)",
                int(f.get("n_fields_compared", 0)),
                ", ".join(str(s) for s in f.get("seeds_compared", [])),
            )
        )
    return "; ".join(lines)


def stamp_inert_arm_knob(manifest: Dict[str, Any], *, warn: bool = True) -> Dict[str, Any]:
    """Record the verdict on `manifest` in place (and WARN to the console). Never raises.

    Called from `manifest_core.stamp_recording_core`, which every manifest reaching
    `pack_writer.write_flat_manifest` passes through.
    """
    try:
        report = check_inert_arm_knob(manifest)
        if not report["checked"]:
            return manifest
        # Direct assignment, not _fill: a meaningful False must win (_is_empty treats
        # False as present, so _fill would refuse to write it).
        manifest["arm_knobs_effective"] = report["arm_knobs_effective"]
        if not report["arm_knobs_effective"]:
            manifest["inert_arm_knob_detail"] = {"findings": report["findings"]}
            if warn:
                message = "%s %s" % (WARN_PREFIX, format_warning(report["findings"]))
                # Double print: stderr for the operator, stdout because the runner
                # captures stdout into `recent_lines` (same shape as zworld_encoder_guard).
                print(message, file=sys.stderr, flush=True)
                print(message, flush=True)
    except Exception:
        pass
    return manifest


__all__ = [
    "check_inert_arm_knob",
    "stamp_inert_arm_knob",
    "format_warning",
    "IDENTITY_KEYS",
    "MIN_COMPARED_FIELDS",
    "EXPECTED_IDENTICAL_KEY",
]
