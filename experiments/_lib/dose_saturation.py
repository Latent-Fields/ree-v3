"""
dose_saturation -- manifest-local lint for two DECLARED DOSE LEVELS that produced the
same value.

Recommended by:
  REE_assembly/evidence/planning/failure_autopsy_V3-EXQ-794_2026-07-22.md sec 6 item 2
      ("a bit-identical DV across two nominally different doses is a saturation
       fingerprint and should be a standing lint ... cheap to detect automatically
       (per_level values equal beyond float noise => refuse the dose-response
       criterion) and would have caught this without an autopsy")

WHAT IT CATCHES
---------------
V3-EXQ-794 ran a 2x2 with SD-076's `waking_confidence_inflation_asymmetry` at two dose
levels, LO=0.6 and HI=0.8. Both levels returned `overconfidence_score` =
-1.004111904519277 -- bit-identical to 15 significant figures -- and `calibration_ratio`
= 2.7564936387545953, likewise identical. `rv_final` finished at EXACTLY 0.010000 on all
four inflation arms.

That is not a null. A genuine dose-response, INCLUDING A GENUINELY NULL ONE, produces
different values at different doses with seed-level variance; two doses agreeing to the
last bit means the quantity was CLAMPED before the dose could express itself. The cause
was an absolute floor (`waking_confidence_rv_floor` = 0.01) sitting 1.8x ABOVE the
substrate's un-inflated operating point of 0.005420, so `max(floor, rv)` pinned every
inflation arm from the first tick.

The cost of not having this lint: SD-076 was recorded `does_not_support` -- a refutation
charged to a claim whose lever never moved -- and MECH-204's Phase-7 correction was left
with no drift to correct, so BOTH claims went untested while appearing to have been
tested. The autopsy withdrew the direction and revised it to `non_contributory`. This
check is ~40 lines and would have caught it at manifest-write time.

HOW THIS DIFFERS FROM `inert_arm_knob` (they are siblings, not duplicates)
--------------------------------------------------------------------------
  inert_arm_knob : declared-distinct arms that RAN IDENTICALLY. The knob never reached a
                   live code path, so the two cells are the same cell -- identical
                   trajectories, identical everything.
  dose_saturation: declared-distinct doses that RAN WITH DIFFERENT KNOBS and produced
                   IDENTICAL OUTPUT. The knob reached the code path and did move the
                   dynamics; a bound downstream then erased the difference.

The 794 arms are NOT bit-identical cell-wide (their trajectories differ, and
`ARM_BOTH_LO` differs from `ARM_INFL_LO` in the 7th decimal), so `inert_arm_knob` does
not and should not fire on them. Different failure, different lint.

POSTURE: RECORD-AND-WARN AT WRITE, GATE AT ADJUDICATION
-------------------------------------------------------
Identical to `inert_arm_knob`, and for the same reason: by manifest-write time the
compute is spent, and refusing to write would destroy an expensive run over a defect that
is sometimes survivable (794's own green arms remained scorable and C4 confirmed the 774
ceiling). Every entry point is wrapped so a lint bug can never crash an experiment. The
autopsy's "REFUSE the dose-response criterion" is honoured by emitting
`dose_levels_separable: False` for the experiment's own scoring to read -- the same
division of labour the 785 per-arm gate uses.

WHAT IT EMITS (convention copied from `inert_arm_knob` / `substrate_stable_across_run`)
----------------------------------------------------------------------------------------
  dose_levels_separable : bool -- False iff some level pair ties on a non-zero float
                          readout. Assigned directly, not via _fill: a meaningful False
                          is the whole point.
  dose_saturation_detail : dict -- emitted ONLY on the False verdict, offenders only.

Deliberately NOT added to `ALWAYS_CORE_KEYS`: the pre-2026-07-22 corpus cannot carry it,
and making it core would turn every legacy manifest into a WARN.

WHY ONLY NON-ZERO FLOATS
------------------------
  - INTEGERS are excluded. `n_seeds_overconfident` = 0 at both LO and HI is the NORMAL
    way a count reports "no effect at either dose"; a tied small integer is evidence of
    nothing. Firing on it would bury the real signal in noise.
  - STRINGS / bools are excluded: identity and provenance, not readouts.
  - EXACT 0.0/0.0 ties are recorded under `zero_ties` but do NOT flip the verdict. Zero
    is overwhelmingly a not-applicable sentinel (794's own `deltas.*.mean` is 0.0 where
    `per_seed` is empty). A clamp AT zero is real but rare, so it is surfaced for a
    reader without generating a standing false positive.
  - A tie between two non-zero floats at different doses is, by contrast, almost
    impossible to produce honestly: float arithmetic over different trajectories does not
    land on the same 15 significant figures by chance.

HOW THE DOSE IS IDENTIFIED
--------------------------
A "dose candidate" is any non-identity numeric per-level key taking a DIFFERENT value at
every level. ALL candidates are excluded from the tie comparison; `manifest["dose_key"]`,
when present, is authoritative for REPORTING and is excluded too.

Excluding every candidate rather than exactly one costs nothing, and this is the whole
reason the inference is safe: a key that varies at every level can never appear in
`tied_fields` by definition, so excluding it removes no evidence and cannot manufacture
the identity the lint reports. Contrast `inert_arm_knob`, which REJECTED its analogous
inference precisely because there the exclusion was lossy -- a misclassified readout got
excluded and the remaining fields then looked identical.

Requiring the candidate to be UNIQUE was the first implementation and was exactly
backwards: a healthy run has several varying readouts, so the lint declined when it had
nothing to complain about, and declined on a PARTIALLY clamped run where it should fire.

If there are no candidates and no declaration, the manifest is OUT OF SCOPE and nothing
fires. The resolved key and its source are recorded in the detail so a reader can audit
the choice.
"""

from __future__ import annotations

import sys
from typing import Any, Dict, List, Mapping, Sequence, Tuple

# Per-level keys that name or locate a level rather than measuring it.
IDENTITY_KEYS: Tuple[str, ...] = (
    "level", "label", "name", "id", "arm", "arm_id",
    "infl_arm", "both_arm", "arms", "seeds",
)

# Explicit dose declaration, and the opt-out. Named pairs rather than a boolean on
# purpose: a blanket off-switch would silence the real case too.
DOSE_KEY_DECLARATION = "dose_key"
EXPECTED_IDENTICAL_KEY = "dose_saturation_expected_identical"

# Relative tolerance for "equal beyond float noise". Far below any real dose effect and
# far above pure float jitter, so the verdict is not sensitive to where it is set.
REL_TOL = 1e-12

# A pair must share at least this many comparable float readouts before a tie means
# anything. One tied scalar is a coincidence; three is a clamp.
MIN_COMPARED_FIELDS = 1

WARN_PREFIX = "[DOSE-SATURATION-WARNING]"


def _is_float_readout(value: Any) -> bool:
    """True for a real-valued measurement. bool is excluded (it is an int in Python)."""
    return isinstance(value, float) and not isinstance(value, bool)


def _levels(manifest: Mapping[str, Any]) -> Dict[str, Mapping[str, Any]]:
    """`aggregates.per_level`, falling back to a top-level `per_level`."""
    for container in (manifest.get("aggregates"), manifest):
        if isinstance(container, Mapping):
            per_level = container.get("per_level")
            if isinstance(per_level, Mapping) and per_level:
                return {
                    str(k): v for k, v in per_level.items() if isinstance(v, Mapping)
                }
    return {}


def _resolve_dose_key(manifest: Mapping[str, Any],
                      levels: Mapping[str, Mapping[str, Any]]) -> Tuple[str, str, List[str]]:
    """Returns (dose_key_for_reporting, source, keys_to_exclude).

    ("", "", []) when no dose can be established at all.

    A "fully-varying numeric key" -- one taking a DIFFERENT value at every level -- is a
    dose candidate. All candidates are excluded from the tie comparison, and this costs
    nothing: a key that varies at every level can never appear in `tied_fields` by
    definition, so excluding it removes no evidence. That is why the lint does NOT
    require the candidate to be UNIQUE.

    Requiring uniqueness was the first implementation and it was exactly backwards: on a
    HEALTHY run several readouts vary across levels, so there were many candidates and
    the lint declined -- going out of scope precisely when it had nothing to complain
    about, and, worse, declining on a PARTIALLY clamped run (one readout tied, one free)
    where it should fire. Caught by test_separable_levels_do_not_fire.
    """
    names = sorted(levels)
    candidates: List[str] = []
    for key in sorted(set().union(*(set(levels[n]) for n in names))):
        if key in IDENTITY_KEYS:
            continue
        values = [levels[n].get(key) for n in names]
        if any(v is None or isinstance(v, bool) or not isinstance(v, (int, float))
               for v in values):
            continue
        if len(set(float(v) for v in values)) == len(values):
            candidates.append(key)

    declared = manifest.get(DOSE_KEY_DECLARATION)
    if isinstance(declared, str) and declared:
        # An explicit declaration is authoritative for REPORTING and is always excluded,
        # whether or not it happens to vary at every level.
        exclude = sorted(set(candidates) | {declared})
        return declared, "manifest.dose_key", exclude
    if not candidates:
        return "", "", []
    if len(candidates) == 1:
        return candidates[0], "inferred (unique fully-varying numeric key)", candidates
    return (", ".join(candidates),
            "inferred (%d fully-varying numeric keys; declare manifest['dose_key'] to "
            "disambiguate)" % len(candidates),
            candidates)


def _expected_identical_pairs(manifest: Mapping[str, Any]) -> set:
    out = set()
    declared = manifest.get(EXPECTED_IDENTICAL_KEY)
    if isinstance(declared, (list, tuple)):
        for pair in declared:
            if isinstance(pair, (list, tuple)) and len(pair) == 2:
                out.add(frozenset((str(pair[0]), str(pair[1]))))
    return out


def _compare_levels(cells_a: Mapping[str, Any],
                    cells_b: Mapping[str, Any],
                    dose_keys: Any) -> Dict[str, Any]:
    """Compare two levels' float readouts. Returns a finding dict (possibly empty).

    `dose_keys` is the exclusion set from `_resolve_dose_key` (a str is accepted for
    single-key callers).
    """
    excluded = {dose_keys} if isinstance(dose_keys, str) else set(dose_keys or ())
    tied: List[str] = []
    zero_ties: List[str] = []
    compared = 0
    for key in sorted(set(cells_a) & set(cells_b)):
        if key in excluded or key in IDENTITY_KEYS:
            continue
        va, vb = cells_a[key], cells_b[key]
        if not (_is_float_readout(va) and _is_float_readout(vb)):
            continue
        compared += 1
        if va == 0.0 and vb == 0.0:
            zero_ties.append(key)
            continue
        scale = max(abs(va), abs(vb))
        if scale > 0.0 and abs(va - vb) <= REL_TOL * scale:
            tied.append(key)
    return {
        "tied_fields": tied,
        "zero_tied_fields": zero_ties,
        "n_fields_compared": compared,
    }


def check_dose_saturation(manifest: Mapping[str, Any]) -> Dict[str, Any]:
    """Manifest-local lint. Returns a report dict; NEVER raises.

    Report shape:
        {"checked": bool,                # False when the manifest is out of scope
         "dose_levels_separable": bool,
         "findings": [ ... ],            # offenders only
         "reason": str}                  # why it was not checked, when checked is False
    """
    try:
        levels = _levels(manifest)
        if len(levels) < 2:
            return {"checked": False, "dose_levels_separable": True, "findings": [],
                    "reason": "fewer than 2 declared dose levels"}

        dose_key, dose_source, dose_exclude = _resolve_dose_key(manifest, levels)
        if not dose_key:
            return {"checked": False, "dose_levels_separable": True, "findings": [],
                    "reason": "no dose could be established -- declare manifest['dose_key']"}

        expected = _expected_identical_pairs(manifest)

        findings: List[Dict[str, Any]] = []
        names = sorted(levels)
        for i, level_a in enumerate(names):
            for level_b in names[i + 1:]:
                if frozenset((level_a, level_b)) in expected:
                    continue
                result = _compare_levels(levels[level_a], levels[level_b], dose_exclude)
                if (result["tied_fields"]
                        and result["n_fields_compared"] >= MIN_COMPARED_FIELDS):
                    result["levels"] = [level_a, level_b]
                    result["dose_key"] = dose_key
                    result["dose_source"] = dose_source
                    result["dose_values"] = [
                        levels[level_a].get(dose_key), levels[level_b].get(dose_key)
                    ]
                    findings.append(result)

        return {"checked": True, "dose_levels_separable": not findings,
                "findings": findings, "reason": ""}
    except Exception as exc:  # pragma: no cover - defensive; a lint must not crash a run
        return {"checked": False, "dose_levels_separable": True, "findings": [],
                "reason": "lint error: %s" % (exc,)}


def format_warning(findings: Sequence[Mapping[str, Any]]) -> str:
    """One ASCII line per offending level pair, naming the dose and the tied readouts."""
    lines: List[str] = []
    for f in findings:
        levels = f.get("levels") or ["?", "?"]
        doses = f.get("dose_values") or [None, None]
        dose_key = f.get("dose_key", "dose")
        # The multi-candidate case has no single value to quote; name the key(s) only.
        if doses[0] is None or doses[1] is None:
            dose_desc = "dose %s" % (dose_key,)
        else:
            dose_desc = "%s = %s vs %s" % (dose_key, doses[0], doses[1])
        lines.append(
            "dose levels %s and %s (%s) produced IDENTICAL values on %s. Two distinct "
            "doses agreeing beyond float noise is a SATURATION signature, not a null -- "
            "the variable was clamped before the dose could express itself. The "
            "dose-response criterion over these levels is UNSCORED, not refuted." % (
                levels[0], levels[1], dose_desc,
                ", ".join(f.get("tied_fields") or []),
            )
        )
    return "; ".join(lines)


def stamp_dose_saturation(manifest: Dict[str, Any], *, warn: bool = True) -> Dict[str, Any]:
    """Record the verdict on `manifest` in place (and WARN to the console). Never raises.

    Called from `manifest_core.stamp_recording_core`, which every manifest reaching
    `pack_writer.write_flat_manifest` passes through.
    """
    try:
        report = check_dose_saturation(manifest)
        if not report["checked"]:
            return manifest
        # Direct assignment, not _fill: a meaningful False must win (_is_empty treats
        # False as present, so _fill would refuse to write it).
        manifest["dose_levels_separable"] = report["dose_levels_separable"]
        if not report["dose_levels_separable"]:
            manifest["dose_saturation_detail"] = {"findings": report["findings"]}
            if warn:
                message = "%s %s" % (WARN_PREFIX, format_warning(report["findings"]))
                # Double print: stderr for the operator, stdout because the runner
                # captures stdout into `recent_lines` (same shape as inert_arm_knob).
                print(message, file=sys.stderr, flush=True)
                print(message, flush=True)
    except Exception:
        pass
    return manifest


__all__ = [
    "check_dose_saturation",
    "stamp_dose_saturation",
    "format_warning",
    "IDENTITY_KEYS",
    "REL_TOL",
    "MIN_COMPARED_FIELDS",
    "DOSE_KEY_DECLARATION",
    "EXPECTED_IDENTICAL_KEY",
]
