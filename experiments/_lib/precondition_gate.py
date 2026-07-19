"""Regime-conditioned precondition gates for multi-arm telemetry probes.

THE FAILURE MODE THIS CLOSES
----------------------------
A multi-regime probe builds a non-vacuity gate per arm, then ANDs it WHOLE-RUN:

    gate_green = all(a["gate_green"] for a in regime_analyses)   # <-- the defect
    if not gate_green:
        outcome, non_degenerate = "FAIL", False
        label = "substrate_not_ready_requeue"

One arm's STRUCTURALLY IMPOSSIBLE precondition then silently vacates another arm's
valid, well-powered result. Confirmed instance: V3-EXQ-785
(`v3_exq_785_mech463_arousal_variance_amplifier_decomp.py`). Its `harm_incumbent`
arm passed all six preconditions on 3959 committed ticks and produced a strong
result (rho -0.8303 against a pre-registered +0.6, ~40 SE). Its `entropy_incumbent`
arm failed exactly one precondition -- `n_components_with_nontrivial_share`, 1.0 vs
a 1.5 floor. The whole run was recorded `non_contributory` / "substrate not ready",
burying the clean arm's finding, and the only claim-favourable number in the run came
from the vacuous arm. See
`REE_assembly/evidence/planning/failure_autopsy_V3-EXQ-785_2026-07-19.md` sections
2a and 8.

That arm's RED was foreseeable at design time. Its own config committed
`expected_incumbent_share = 1.043`; shares sum to 1.0 by construction, so every other
component summed to -0.043 and none could clear the |0.01| floor. A gate demanding
TWO components above that floor was unsatisfiable, and the number proving it was
pre-registered alongside the gate.

THE RULE, WHICH THE 785 SCRIPT ALREADY DERIVED AND DID NOT GENERALISE
---------------------------------------------------------------------
785's P1 note states the principle exactly, for one precondition:

    P1 applies ONLY where the incumbent is a MODULATORY CHANNEL. In a primary-
    component regime the modulatory channels legitimately contribute ~0 and the
    authority gate never fires -- asserting it there would make that regime's gate
    structurally un-passable and collapse the two-regime design.

P7 was written without that conditioning and did the analogous damage in reverse.
The generalisation this module enforces:

    EVERY precondition declares the regimes it applies to, and NO arm's gate
    result may vacate another arm's.

TWO DISPOSITIONS -- DO NOT CONFLATE THEM
----------------------------------------
"Structurally unsatisfiable for this arm" has two very different resolutions, and
collapsing them launders an artifact into a clean result:

(a) NOT MEANINGFUL for the regime -- scope the PRECONDITION out; the arm is still
    scorable. 785's P1 in the `harm_incumbent` regime: the modulatory channels
    legitimately contribute ~0 there, so the authority gate is simply not the right
    question. Declare with `applies_to`.

(b) MEANINGFUL, but the arm cannot satisfy it -- scope the ARM out of scoring. The
    precondition is doing real work: it has detected that the arm is vacuous. 785's
    P7 in the `entropy_incumbent` regime: `expected_incumbent_share = 1.043` forces a
    single-component decomposition, so that arm's rho +0.5879 is an ARITHMETICALLY
    FORCED ARTIFACT. The autopsy is explicit that it must not be cited as support for
    MECH-463 -- it is the run's only claim-favourable number. Declare with
    `structural_max` / `structural_min`; `detect_structural_vacuity` finds it.

Treating (b) as (a) makes the vacuous arm pass its gate and become citable, which is
a WORSE failure than the one this module exists to fix -- it manufactures support
instead of merely burying a result. Hence: a structurally vacuous arm is excluded
from scoring, is never green, and STILL does not vacate any other arm.

USAGE
-----
    from experiments._lib.precondition_gate import (
        PreconditionSpec, evaluate_arm_gate, aggregate_arm_gates,
        assert_no_structurally_unsatisfiable_gate,
    )

    SPECS = [
        PreconditionSpec(
            name="modulatory_authority_active_frac",
            description="fraction of selection ticks where the authority gate fired",
            control="candidates that genuinely differ",
            threshold=AUTHORITY_ACTIVE_FRAC_FLOOR,
            # regime conditioning: a primary-component regime cannot fire it
            applies_to=lambda ctx: ctx["expected_incumbent"].startswith("CH:"),
            applies_note="channel-incumbent regimes only -- N/A when the incumbent "
                         "is a primary score component",
        ),
        PreconditionSpec(
            name="n_components_with_nontrivial_share",
            description="number of components holding |share| > 0.01",
            control="multi-component primary score",
            threshold=1.5,
            applies_to=lambda ctx: ctx["expected_incumbent_share"] <= 1.0,
            applies_note="a pre-registered incumbent share at or above unity forces "
                         "every other component negative -- unsatisfiable by "
                         "construction, not a substrate fact",
            # design-time proof of the above, checked before any compute is spent
            structural_max=lambda ctx: 1.0 if ctx["expected_incumbent_share"] > 1.0 else None,
        ),
    ]

    # Design-audit: refuse to start a run carrying an unsatisfiable gate.
    assert_no_structurally_unsatisfiable_gate(SPECS, ARM_CONTEXTS)

    arm_gates = [evaluate_arm_gate(ctx["id"], ctx, SPECS, measured=..., ) for ctx in ...]
    gate = aggregate_arm_gates(arm_gates)

    manifest["per_arm_gate"] = gate["per_arm_gate"]          # top level -- task 3
    manifest["non_degenerate"] = gate["non_degenerate"]
    manifest["degeneracy_reason"] = gate["degeneracy_reason"]
    manifest["interpretation"]["preconditions"] = gate["adjudication_preconditions"]

WHY `adjudication_preconditions` IS NOT SIMPLY EVERY PRECONDITION
-----------------------------------------------------------------
`REE_assembly/evidence/experiments/scripts/build_experiment_indexes.py`
`_compute_adjudication` iterates `interpretation.preconditions` FLAT and ARM-BLIND,
and returns `precondition_unmet` for the WHOLE RUN on the first unmet entry. So a
probe that flattens every arm's preconditions into that list reproduces the 785
vacating at ADJUDICATION time even after the script's own routing is fixed --
the indexer re-derives it independently, and it recomputes `met` from
measured+threshold rather than trusting the author's flag.

So when SOME arms are green and some red, the red arms' preconditions are carried in
the top-level `per_arm_gate` block (in full, auditable, naming the arm) and kept OUT
of the flat adjudication list, with `preconditions_scope_note` stating exactly which
arms were excluded and why. Nothing is hidden and nothing is silently dropped: the
red arm's criteria are simultaneously marked `load_bearing: false` and
`criteria_non_degenerate: false` by `arm_criteria_non_degenerate()`, which is the
per-criterion channel the indexer DOES honour. When ALL arms are red the run really
is vacuous, every precondition goes into the flat list, and the indexer's
whole-run `precondition_unmet` is the correct verdict.

ASCII-only in printed output (Windows cp1252 terminals).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

__all__ = [
    "PreconditionSpec",
    "StructurallyUnsatisfiableGate",
    "evaluate_arm_gate",
    "aggregate_arm_gates",
    "arm_criteria_non_degenerate",
    "assert_no_structurally_unsatisfiable_gate",
    "detect_structural_vacuity",
]


class StructurallyUnsatisfiableGate(AssertionError):
    """A precondition cannot be met by an arm no matter what the substrate does."""


@dataclass
class PreconditionSpec:
    """One precondition, plus the regimes it is meaningful for.

    `applies_to` is the whole point of this module. It receives the arm context
    dict and returns False when the precondition is not meaningful for that arm.
    A scoped-out precondition does NOT enter that arm's gate and does NOT enter
    the flat adjudication list; it is recorded under the arm's `scoped_out` for
    audit, carrying `applies_note` as the reason.

    `structural_max` / `structural_min` are OPTIONAL design-time proofs. Return
    the best value the arm could attain given its PRE-REGISTERED config (or None
    when no such bound is derivable). If that bound cannot clear the threshold,
    `assert_no_structurally_unsatisfiable_gate` refuses the run before compute is
    spent -- the check that would have caught 785's `expected_incumbent_share =
    1.043` against a >= 2-components gate for free at queue time.

    `direction` mirrors the indexer's `_precondition_direction`: "lower" (a FLOOR,
    met when measured > threshold) or "upper" (a CEILING, met when measured <
    threshold). Declared explicitly on every emitted record so the indexer's
    recompute cannot default-misread it (the 2026-06-07 648a/649 directionality
    bug).
    """

    name: str
    description: str
    control: str
    threshold: float
    direction: str = "lower"
    kind: str = "readiness"
    applies_to: Optional[Callable[[Dict[str, Any]], bool]] = None
    applies_note: str = ""
    structural_max: Optional[Callable[[Dict[str, Any]], Optional[float]]] = None
    structural_min: Optional[Callable[[Dict[str, Any]], Optional[float]]] = None

    def applies(self, arm_ctx: Dict[str, Any]) -> bool:
        """True when this precondition is meaningful for `arm_ctx`."""
        if self.applies_to is None:
            return True
        return bool(self.applies_to(arm_ctx))

    def met_for(self, measured: float) -> bool:
        """Recompute `met` the same way the indexer does."""
        if _is_ceiling(self.direction):
            return float(measured) < float(self.threshold)
        return float(measured) > float(self.threshold)


def _is_ceiling(direction: str) -> bool:
    return str(direction).strip().lower() in ("upper", "ceiling", "max", "upper_bound")


def _spec_unsatisfiable(spec: PreconditionSpec,
                        arm_ctx: Dict[str, Any]) -> Optional[str]:
    """Return a reason string when `spec` is provably unmeetable by `arm_ctx`.

    Only fires on an explicitly declared structural bound. Absent one, returns
    None -- this is a design-audit aid, not an oracle.
    """
    if _is_ceiling(spec.direction):
        if spec.structural_min is None:
            return None
        bound = spec.structural_min(arm_ctx)
        if bound is None:
            return None
        if not float(bound) < float(spec.threshold):
            return (f"best attainable (minimum) value {float(bound):.6g} cannot fall "
                    f"below the ceiling {float(spec.threshold):.6g}")
        return None
    if spec.structural_max is None:
        return None
    bound = spec.structural_max(arm_ctx)
    if bound is None:
        return None
    if not float(bound) > float(spec.threshold):
        return (f"best attainable (maximum) value {float(bound):.6g} cannot exceed "
                f"the floor {float(spec.threshold):.6g}")
    return None


def detect_structural_vacuity(specs: Sequence[PreconditionSpec],
                              arm_ctx: Dict[str, Any]) -> Optional[str]:
    """Reason string when this arm provably cannot satisfy an APPLICABLE precondition.

    Disposition (b) in the module docstring. The precondition is meaningful for the
    arm and the arm cannot meet it from its pre-registered config, so the ARM is
    vacuous -- its readouts are artifacts and must not be scored or cited.

    Returns None when no applicable precondition is provably unsatisfiable. Only
    fires on an explicitly declared `structural_max` / `structural_min`; absent one
    this is silent, so it is a design-audit aid, not an oracle.
    """
    reasons: List[str] = []
    for spec in specs:
        if not spec.applies(arm_ctx):
            continue  # (a): not meaningful here -- not evidence of vacuity
        why = _spec_unsatisfiable(spec, arm_ctx)
        if why:
            reasons.append(f"'{spec.name}': {why}")
    if not reasons:
        return None
    return ("arm cannot satisfy an applicable precondition from its pre-registered "
            "config, so its readouts are arithmetically forced artifacts, not "
            "measurements -- " + "; ".join(reasons))


def assert_no_structurally_unsatisfiable_gate(
        specs: Sequence[PreconditionSpec],
        arm_contexts: Sequence[Dict[str, Any]],
        arm_id_key: str = "id",
        acknowledged_vacuous_arms: Sequence[str] = ()) -> List[Dict[str, Any]]:
    """Refuse a run carrying an arm that cannot be scored, unless acknowledged.

    Call BEFORE the expensive phase (and in --dry-run). Two correct resolutions,
    and the guard makes the author pick one deliberately:

      (a) the precondition is NOT MEANINGFUL for the arm -> give it an `applies_to`
          excluding the arm. The arm stays scorable.
      (b) the arm genuinely CANNOT be scored -> pass its id in
          `acknowledged_vacuous_arms`, and mark it vacuous at evaluation time so it
          is excluded from scoring and never cited.

    NEVER lower the threshold: that converts a detected artifact into a citable
    result. 785's entropy arm is case (b) -- its rho +0.5879 is arithmetically
    forced and is the run's only claim-favourable number.

    Raises StructurallyUnsatisfiableGate naming the arm, the precondition and the
    arithmetic. Also raises when EVERY arm is vacuous, acknowledged or not: such a
    run can produce no scorable result and should not consume compute.

    Returns the audited (spec, arm) pairs for logging.
    """
    audited: List[Dict[str, Any]] = []
    problems: List[str] = []
    acknowledged = set(acknowledged_vacuous_arms)
    vacuous_arms: List[str] = []
    all_arms: List[str] = []

    for ctx in arm_contexts:
        arm_id = str(ctx.get(arm_id_key, "?"))
        all_arms.append(arm_id)
        arm_has_problem = False
        for spec in specs:
            if not spec.applies(ctx):
                audited.append({"arm": arm_id, "precondition": spec.name,
                                "status": "scoped_out"})
                continue
            reason = _spec_unsatisfiable(spec, ctx)
            audited.append({"arm": arm_id, "precondition": spec.name,
                            "status": "unsatisfiable" if reason else "satisfiable"})
            if reason:
                arm_has_problem = True
                if arm_id not in acknowledged:
                    problems.append(
                        f"  arm '{arm_id}' precondition '{spec.name}': {reason}")
        if arm_has_problem:
            vacuous_arms.append(arm_id)

    if problems:
        raise StructurallyUnsatisfiableGate(
            "Gate is structurally unsatisfiable for at least one arm. Left as is, "
            "this arm can never be scored, and under a whole-run AND it would vacate "
            "every other arm with it:\n"
            + "\n".join(problems)
            + "\n\nPICK ONE:\n"
              "  (a) the precondition is NOT MEANINGFUL for this arm -> give it an "
              "`applies_to` excluding the arm (the arm stays scorable);\n"
              "  (b) the arm genuinely CANNOT be scored -> pass its id in "
              "`acknowledged_vacuous_arms` and mark it vacuous at evaluation time "
              "(excluded from scoring, never cited).\n"
              "Do NOT lower the threshold -- that turns a detected artifact into a "
              "citable result. See failure_autopsy_V3-EXQ-785_2026-07-19.md "
              "sections 2a and 8.")

    if all_arms and len(vacuous_arms) == len(all_arms):
        raise StructurallyUnsatisfiableGate(
            "EVERY arm is structurally vacuous (" + ", ".join(vacuous_arms) + "). "
            "This run can produce no scorable result -- do not spend compute on it. "
            "Redesign the arms or their pre-registered config.")
    return audited


def evaluate_arm_gate(arm_id: str,
                      arm_ctx: Dict[str, Any],
                      specs: Sequence[PreconditionSpec],
                      measured: Dict[str, float],
                      met_overrides: Optional[Dict[str, bool]] = None,
                      structurally_vacuous: Optional[str] = None,
                      auto_detect_vacuity: bool = True) -> Dict[str, Any]:
    """Evaluate one arm's gate, honouring each precondition's regime conditioning.

    `measured` maps precondition name -> measured value. A spec scoped OUT of this
    arm needs no entry. A spec that applies but has no measured value is a coding
    error and raises KeyError rather than defaulting -- a silently-absent
    measurement is exactly the vacuity these gates exist to catch.

    `met_overrides` supplies `met` for a precondition whose pass condition is not a
    pure threshold comparison (e.g. 785's `incumbent_identity_as_preregistered`,
    which also requires an identity match). The measured/threshold pair is still
    emitted so the indexer can recompute the numeric leg.

    `structurally_vacuous` marks disposition (b): the arm cannot be scored and its
    readouts are artifacts. Such an arm is never green, is excluded from scoring,
    and -- crucially -- still does not vacate any other arm. When
    `auto_detect_vacuity` is on (the default) `detect_structural_vacuity` also runs,
    so an arm that provably cannot meet an applicable precondition is caught even if
    the author forgot to say so. Passing an explicit reason wins.

    Precondition `name` is namespaced `<arm_id>::<name>` so a failure is
    attributable to its arm at every downstream reader.
    """
    overrides = met_overrides or {}
    vacuity = structurally_vacuous
    if vacuity is None and auto_detect_vacuity:
        vacuity = detect_structural_vacuity(specs, arm_ctx)
    applied: List[Dict[str, Any]] = []
    scoped_out: List[Dict[str, Any]] = []

    for spec in specs:
        if not spec.applies(arm_ctx):
            scoped_out.append({
                "name": f"{arm_id}::{spec.name}",
                "precondition": spec.name,
                "arm": arm_id,
                "applies": False,
                "reason": spec.applies_note or (
                    "not meaningful for this regime (no applies_note given)"),
            })
            continue
        if spec.name not in measured:
            raise KeyError(
                f"arm '{arm_id}': precondition '{spec.name}' applies to this arm but "
                f"no measured value was supplied. Either measure it or scope it out "
                f"with applies_to.")
        value = float(measured[spec.name])
        met = bool(overrides[spec.name]) if spec.name in overrides else spec.met_for(value)
        applied.append({
            "name": f"{arm_id}::{spec.name}",
            "precondition": spec.name,
            "arm": arm_id,
            "kind": spec.kind,
            "description": spec.description,
            "control": spec.control,
            "direction": spec.direction,
            "applies": True,
            "measured": value,
            "threshold": float(spec.threshold),
            "met": met,
        })

    failed = [p["precondition"] for p in applied if not p["met"]]
    green = bool(applied) and all(p["met"] for p in applied) and vacuity is None
    return {
        "arm": arm_id,
        "gate_green": green,
        "structurally_vacuous": vacuity is not None,
        "vacuity_reason": vacuity or "",
        "preconditions": applied,
        "scoped_out": scoped_out,
        "failed_preconditions": failed,
        "n_applied": len(applied),
        "n_scoped_out": len(scoped_out),
    }


def aggregate_arm_gates(arm_gates: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate per-arm gates WITHOUT letting a red arm vacate a green one.

    The semantic fix. `non_degenerate` is `any arm green`, not `all arms green`:
    a run carrying one clean, well-powered arm is NOT vacuous, whatever another
    arm did. Only an all-red run is.

    Returns a dict with:
      per_arm_gate                -- the top-level manifest block (task 3): which
                                     arms are green/red, and which preconditions
                                     failed in which arm
      non_degenerate              -- any arm green
      degeneracy_reason           -- names the red arms and their failures, "" when
                                     all green
      adjudication_preconditions  -- the flat list for interpretation.preconditions
                                     (see the module docstring for why this is
                                     green-arms-only on a partial run)
      all_green / any_green / green_arms / red_arms
    """
    gates = list(arm_gates)
    green = [g for g in gates if g["gate_green"]]
    red = [g for g in gates if not g["gate_green"]]
    green_arms = [g["arm"] for g in green]
    red_arms = [g["arm"] for g in red]
    vacuous_arms = [g["arm"] for g in gates if g.get("structurally_vacuous")]
    all_green = bool(gates) and not red
    any_green = bool(green)

    failed_by_arm = {g["arm"]: list(g["failed_preconditions"]) for g in red}
    vacuity_by_arm = {g["arm"]: g.get("vacuity_reason", "")
                      for g in gates if g.get("structurally_vacuous")}

    if all_green:
        degeneracy_reason = ""
        scope_note = ("All arms passed their gate; every precondition is carried in "
                      "interpretation.preconditions.")
        adjudication = [p for g in gates for p in g["preconditions"]]
    elif any_green:
        parts = "; ".join(
            (f"arm '{arm}' is STRUCTURALLY VACUOUS ({vacuity_by_arm[arm]})"
             + (f" [it also failed {', '.join(fp)}]" if fp else "")
             if arm in vacuity_by_arm else
             f"arm '{arm}' failed "
             f"{', '.join(fp) if fp else '(no applied preconditions)'}")
            for arm, fp in failed_by_arm.items())
        if vacuity_by_arm:
            parts += (". A structurally vacuous arm's readouts are ARTIFACTS -- do "
                      "not cite them in either direction")
        degeneracy_reason = (
            f"PARTIAL non-vacuity: {parts}. Arm(s) {', '.join(green_arms)} passed the "
            f"gate in full and ARE scored -- a red arm does NOT vacate a green one "
            f"(failure_autopsy_V3-EXQ-785_2026-07-19.md sections 2a/8). Read the red "
            f"arm(s) as unscored, NOT as a refutation.")
        scope_note = (
            f"interpretation.preconditions carries the GREEN arm(s) only "
            f"({', '.join(green_arms)}). The RED arm(s) ({', '.join(red_arms)}) are "
            f"carried in full at top level under per_arm_gate.red, and their criteria "
            f"are marked load_bearing:false + criteria_non_degenerate:false. The "
            f"indexer's _compute_adjudication reads interpretation.preconditions FLAT "
            f"and ARM-BLIND and returns precondition_unmet for the WHOLE RUN on the "
            f"first unmet entry, so including a scored-out arm's failure here would "
            f"re-vacate the green arm at adjudication time.")
        adjudication = [p for g in green for p in g["preconditions"]]
    else:
        parts = "; ".join(
            f"arm '{arm}' failed {', '.join(fp) if fp else '(no applied preconditions)'}"
            for arm, fp in failed_by_arm.items())
        degeneracy_reason = (
            f"Non-vacuity gate RED in EVERY arm: {parts}. No arm is scored; this run "
            f"is NOT a refutation.")
        scope_note = ("No arm passed its gate, so every precondition is carried in "
                      "interpretation.preconditions and the whole run is vacuous.")
        adjudication = [p for g in gates for p in g["preconditions"]]

    return {
        "non_degenerate": any_green,
        "degeneracy_reason": degeneracy_reason,
        "all_green": all_green,
        "any_green": any_green,
        "green_arms": green_arms,
        "red_arms": red_arms,
        "adjudication_preconditions": adjudication,
        "vacuous_arms": vacuous_arms,
        "per_arm_gate": {
            "green_arms": green_arms,
            "red_arms": red_arms,
            "structurally_vacuous_arms": vacuous_arms,
            "vacuity_reason_by_arm": vacuity_by_arm,
            "all_green": all_green,
            "any_green": any_green,
            "failed_preconditions_by_arm": failed_by_arm,
            "preconditions_scope_note": scope_note,
            "green": [_arm_summary(g) for g in green],
            "red": [_arm_summary(g) for g in red],
        },
    }


def _arm_summary(gate: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "arm": gate["arm"],
        "gate_green": gate["gate_green"],
        "structurally_vacuous": bool(gate.get("structurally_vacuous")),
        "vacuity_reason": gate.get("vacuity_reason", ""),
        "failed_preconditions": list(gate["failed_preconditions"]),
        "preconditions": gate["preconditions"],
        "scoped_out": gate["scoped_out"],
    }


def arm_criteria_non_degenerate(criteria_by_arm: Dict[str, Sequence[str]],
                                aggregate: Dict[str, Any],
                                extra: Optional[Dict[str, bool]] = None
                                ) -> Dict[str, bool]:
    """Per-criterion non-degeneracy keyed by the owning arm's gate.

    `criteria_by_arm` maps arm_id -> the criterion names that arm owns. A criterion
    belonging to a RED arm is non_degenerate=False; one belonging to a GREEN arm is
    True unless `extra` says otherwise (e.g. a power check such as "at least two
    deciles populated", which can fail independently of the gate).

    This is the per-criterion channel `build_experiment_indexes.py` honours, and it
    is what makes a green arm's result visibly separable at adjudication time
    instead of needing an autopsy to recover.
    """
    extras = extra or {}
    green = set(aggregate["green_arms"])
    out: Dict[str, bool] = {}
    for arm, names in criteria_by_arm.items():
        for name in names:
            out[name] = bool(arm in green and extras.get(name, True))
    return out
