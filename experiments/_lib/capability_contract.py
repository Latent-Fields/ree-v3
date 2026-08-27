"""Per-run capability / plasticity contract for organism-level experiments.

Implements the machine-verified preflight required by **GOV-CAPCONTRACT-1**
(REE_assembly/docs/claims/claims.yaml, registered 2026-08-27 in `cbac6ceea6`):

    A run is not admissible as negative evidence for a claim unless the organism
    instantiated for that run was demonstrably able to express -- and, where the
    claim is learning- or development-dependent, to acquire -- the faculty under
    test. "It did not learn" and "it could not have learned" are different
    scientific outcomes.

An experiment DECLARES `requires_mechanisms`, `requires_capabilities` and
`requires_plasticity`; this module VERIFIES each against the organism actually
instantiated and emits one top-level manifest block carrying the evidence and an
explicit interpretation ROUTE.

WHAT THIS IS NOT
----------------
It is NOT a duplicate of GOV-PATHVALID-1, which governs FALSE POSITIVES (a PASS
obtained by injecting the state an upstream stage would have supplied is not
evidence the pathway is reachable). This is its mirror: it governs whether a FAIL
is entitled to be called negative evidence.

It is NOT a duplicate of GOV-FAILLOC-1, whose four post-hoc autopsy buckets are
REE / MECHANISM / MEASURES / ENVIRONMENT. A run in which the mechanism was
correctly instantiated, the measures were adequate and the environment was
informative, but every relevant parameter was frozen, ticks NONE of those four.
Plasticity is a fifth category, and this module is where it becomes visible --
pre-registered and machine-checked, not reconstructed at autopsy.

IT COMPOSES WITH `precondition_gate.py` AND MUST NOT ABSORB IT
--------------------------------------------------------------
`precondition_gate.py` handles REGIME-CONDITIONED, PER-ARM non-vacuity. Its own
docstring records V3-EXQ-785, where a whole-run AND of per-arm gates vacated a
clean arm's valid, well-powered result. **This module must never reintroduce that
shape**, and the reason it cannot is structural rather than a matter of care:

    every check here is a property of the ONE organism instantiated for the run,
    not of an arm.

Whether `mech341` was constructed, whether gradients were enabled, whether the
intended parameters were in an optimizer -- none of these can be true for one arm
and false for another, because there is one organism and one process. So a
whole-run verdict is the CORRECT scope here, and it is not the 785 defect wearing
a new name. The 785 defect is specifically an ARM-scoped fact promoted to run
scope.

The practical consequence, and the line not to cross: **this module writes its own
top-level manifest block (`capability_contract`) and does NOT write into
`interpretation.preconditions`**, which `precondition_gate.aggregate_arm_gates()`
owns. Two writers to that list would fight, and the indexer reads it FLAT and
ARM-BLIND (`build_experiment_indexes.py::_compute_adjudication` returns whole-run
`precondition_unmet` on the first unmet entry), which is exactly the mechanism
785 was re-vacated by at adjudication time. `as_adjudication_preconditions()`
exists for a driver that uses NO per-arm gate and wants the indexer to see these
checks, and is opt-in for that reason.

A FAILED PREFLIGHT IS NOT A `FAIL`
-----------------------------------
Recording an unmet capability contract as `outcome: "FAIL"` is the precise error
GOV-CAPCONTRACT-1 exists to prevent: it puts a mis-instantiation into the evidence
record as negative evidence against the claim. So:

  * this module NEVER raises on an unmet contract -- raising invites a
    `try/except` that routes to FAIL, which is the failure mode itself;
  * `route_for_manifest()` returns `experiment_purpose="diagnostic"` and
    `evidence_direction="non_contributory"`, the two channels
    `build_experiment_indexes.py` already honours (it sets
    `scoring_excluded="diagnostic_probe"` and `scoring_excluded="non_contributory"`
    respectively), so an unmet run is excluded from confidence/conflict scoring
    without any new indexer support;
  * `assert_not_recorded_as_fail()` is the belt-and-braces check a driver can call
    immediately before writing its manifest.

"RECORDED AS NEGATIVE EVIDENCE" HAS TWO DOORS, and only one says FAIL. The
indexer INFERS a direction when the manifest sets none -- `"supports" if
final_status == "PASS" else "weakens"` -- so a driver that applies the `outcome`
override but drops `evidence_direction` gets `weakens` inferred from a non-"PASS"
outcome. That is the prohibited recording without the word FAIL appearing
anywhere. `route_for_manifest()` returns both fields as ONE bundle for that
reason, and `assert_not_recorded_as_fail()` checks both doors.

It raises only on PROGRAMMER error -- a malformed declaration, or a caller about
to make the recording above -- via `CapabilityContractDeclarationError`.

FAIL OPEN, LOUDLY
-----------------
A check this module cannot determine reports `UNDETERMINED`. It is never silently
reported as satisfied, it never contributes to an unmet route, and it never blocks
a run. `undetermined_checks` and `has_undetermined` carry it into the manifest so
the gap is visible to the reader and to a later autopsy. This is deliberate: a
contract that blocks when its own probe is broken would take the fleet out on a
probe defect, and a contract that defaults to "satisfied" would launder exactly
the mis-instantiation it exists to detect.

THE FIVE ROUTES
---------------
Reported in `interpretation_route`, ordered by ARC-130's causal-reach ladder
(existence -> representation -> recruitment -> local operation -> competitive
authority -> committed throughput -> ecological consequence -> retention), so the
MOST UPSTREAM unmet stage is the one named:

  "capability_precondition_unmet" -- a declared mechanism or capability was not
        constructed, or was constructed but not enabled. ARC-130 stage
        "existence". Mis-instantiation: outcome (b) of the claim's
        functional_restatement.
  "mechanism_unreached"           -- constructed and enabled, but never reached on
        the production path, or its decisive readouts engaged too rarely for the
        test to be non-vacuous. ARC-130 stages "endogenous recruitment" /
        "local operation". Outcome (c).
  "authority_floor_unmet"         -- reached, but below the declared
        competitive-authority or readiness floor: non-zero influence is not
        necessarily competitive influence. ARC-130 stages "competitive authority"
        / "committed throughput". Also outcome (c).
  "nonplastic_misfire"            -- the organism was capable but could not have
        ACQUIRED anything: gradients disabled, the intended parameters absent from
        any optimizer, or the parameter-delta witness showing no update. Outcome
        (d) -- the fifth GOV-FAILLOC-1 bucket.
  "interpretable"                 -- every declared precondition met (or only
        UNDETERMINED, which does not block). Outcome (a): the ONLY route whose
        FAIL may enter the evidence record against the claim.

Every triggered route is listed in `routes_triggered`; `interpretation_route`
names the most upstream. Nothing is hidden by the precedence.

PLASTICITY VOCABULARY IS A STUB -- SEE `PLASTICITY_MODES`
---------------------------------------------------------
GOV-CAPCONTRACT-1's own notes name its precondition: "a within-life plasticity
inventory across the existing long-life drivers ... Without that inventory the
`requires_plasticity` field has no vocabulary to declare against." That inventory
(`REE_assembly/evidence/planning/within_life_plasticity_inventory_*.md`,
chip `chip-20260827-plasticity-inventory`) did NOT exist when this module was
written, and was NOT invented here. `PLASTICITY_MODES` transcribes the claim
title's own enumeration as a PROVISIONAL placeholder and validation against it is
ADVISORY, never a gate -- see that constant's comment.

USAGE
-----
    from experiments._lib.capability_contract import (
        CapabilityContract, MechanismRequirement, PlasticityRequirement,
        capture_parameter_witness, verify_capability_contract,
        route_for_manifest, assert_not_recorded_as_fail,
    )

    CONTRACT = CapabilityContract(
        experiment_id="V3-EXQ-999",
        canonical_profile="ree_v3_baseline@v0",
        declared_deviations={"latent.e3_enabled": True},
        requires_mechanisms=[
            MechanismRequirement(
                name="e3_selector",
                description="E3 candidate selection must exist and run",
                config_flag="latent.e3_enabled",
                reached_metric="e3_selection_ticks_frac",
                reached_floor=0.05,
                authority_metric="e3_committed_share",
                authority_floor=0.02,
                claim_ids=("MECH-463",),
            ),
        ],
        requires_plasticity=[
            PlasticityRequirement(mode="parameters",
                                  description="policy head must be trainable"),
        ],
    )

    # before the expensive phase
    witness_before = capture_parameter_witness(agent.policy.parameters())
    ...
    result = verify_capability_contract(
        CONTRACT, agent=agent,
        engagement={"e3_selection_ticks_frac": 0.21, "e3_committed_share": 0.08},
        optimizers=[opt],
        parameter_witness_before=witness_before,
        parameter_witness_after=capture_parameter_witness(agent.policy.parameters()),
    )
    manifest["capability_contract"] = result["manifest_block"]
    if not result["satisfied"]:
        manifest.update(route_for_manifest(result))   # never a FAIL
    assert_not_recorded_as_fail(manifest, result)

ASCII-only in printed output (repo rule -- Windows cp1252 terminals).
Stdlib only at module scope: torch and ree_core are duck-typed and probed
defensively, so this module (and its contracts) import without either.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

__all__ = [
    "MANIFEST_KEY",
    "ROUTE_INTERPRETABLE",
    "ROUTE_CAPABILITY_PRECONDITION_UNMET",
    "ROUTE_MECHANISM_UNREACHED",
    "ROUTE_NONPLASTIC_MISFIRE",
    "ROUTE_AUTHORITY_FLOOR_UNMET",
    "ROUTE_PRECEDENCE",
    "PLASTICITY_MODES",
    "STATUS_SATISFIED",
    "STATUS_UNMET",
    "STATUS_UNDETERMINED",
    "ROUTED_OUTCOME",
    "CapabilityContractDeclarationError",
    "MechanismRequirement",
    "CapabilityRequirement",
    "PlasticityRequirement",
    "CapabilityContract",
    "capture_parameter_witness",
    "compare_parameter_witness",
    "learning_mode_snapshot",
    "optimizer_membership",
    "verify_capability_contract",
    "route_for_manifest",
    "assert_not_recorded_as_fail",
    "as_adjudication_preconditions",
    "format_report",
]

MANIFEST_KEY = "capability_contract"
CONTRACT_SCHEMA = "capability_contract/v1"

# --- interpretation routes -------------------------------------------------- #

ROUTE_INTERPRETABLE = "interpretable"
ROUTE_CAPABILITY_PRECONDITION_UNMET = "capability_precondition_unmet"
ROUTE_MECHANISM_UNREACHED = "mechanism_unreached"
ROUTE_AUTHORITY_FLOOR_UNMET = "authority_floor_unmet"
ROUTE_NONPLASTIC_MISFIRE = "nonplastic_misfire"

# Ordered by ARC-130's causal-reach ladder, most upstream first. The most
# upstream triggered route is reported as `interpretation_route`; every triggered
# route is still listed in `routes_triggered`, so precedence summarises and never
# hides. Plasticity sits LAST deliberately: an organism that never had the
# mechanism at all is a more fundamental mis-instantiation than one whose
# parameters were frozen, and naming the frozen parameters first would send an
# autopsy looking for an optimizer bug in a run that had no mechanism to train.
ROUTE_PRECEDENCE = (
    ROUTE_CAPABILITY_PRECONDITION_UNMET,
    ROUTE_MECHANISM_UNREACHED,
    ROUTE_AUTHORITY_FLOOR_UNMET,
    ROUTE_NONPLASTIC_MISFIRE,
)

STATUS_SATISFIED = "satisfied"
STATUS_UNMET = "unmet"
STATUS_UNDETERMINED = "undetermined"

# The manifest `outcome` an unmet contract routes to. Deliberately NOT "PASS" or
# "FAIL": both are already read as scientific verdicts, and this run has none --
# its acceptance criteria were never evaluated, because the organism could not
# express the faculty they test. A distinct token cannot be silently misread as
# either. It is a new value in this corpus (which today holds PASS / FAIL / ERROR
# / PARTIAL* / INCONCLUSIVE) and it is inert in the indexer, which branches on
# `outcome` only for "ERROR"; the exclusion is carried by `experiment_purpose` and
# `evidence_direction`, which the indexer DOES honour -- see route_for_manifest.
#
# NOT THE SAME CHANNEL AS THE RUNNER SENTINEL. `experiment_protocol.emit_outcome()`
# takes "PASS" or "FAIL" only, and is the runner's queue-conformance signal, not a
# scientific verdict. A preflight-unmet run ran to completion and did not crash, so
# it should still emit `emit_outcome("PASS", exit_reason="skipped", ...)`; the
# scientific reading lives in the manifest this module writes, not in the sentinel.
# Emitting a sentinel "FAIL" there would put the word FAIL on a run whose criteria
# were never evaluated, which is the confusion this separation exists to avoid.
ROUTED_OUTCOME = "DIAGNOSTIC"

# evidence_direction values build_experiment_indexes.py maps to scoring_excluded.
# `superseded` is deliberately absent: it means "invalidated by a corrected
# iteration", which is a different fact from "the organism could not express the
# faculty", and using it here would misattribute the exclusion.
_EXCLUDED_DIRECTIONS = frozenset({"non_contributory", "inconclusive"})

# PROVISIONAL STUB -- TODO(chip-20260827-plasticity-inventory).
#
# GOV-CAPCONTRACT-1's notes make the within-life plasticity inventory a
# PRECONDITION FOR ITS OWN IMPLEMENTATION: "Without that inventory the
# `requires_plasticity` field has no vocabulary to declare against." That
# inventory -- REE_assembly/evidence/planning/within_life_plasticity_inventory_*.md,
# owed by chip-20260827-plasticity-inventory -- did not exist when this module
# was written and NOTHING here invents it.
#
# The tuple below is a verbatim transcription of the enumeration in the
# REGISTERED claim title ("which forms of change were permitted (parameters,
# policy/value, E1/E2 representations, memory state, residue/EMA state, offline
# updates)"), quoted so `requires_plasticity` has something to name today. It is
# NOT the inventory, and it is certainly not complete: ARC-135's four-way
# continuity decomposition (cognitive/affective/mnemonic, parameter/plasticity,
# body/homeostatic, ecological/world) is the other vocabulary this field must
# eventually compose with, and the inventory is expected to enumerate hippocampal
# buffers and sleep-dependent updates separately.
#
# VALIDATION AGAINST IT IS THEREFORE ADVISORY, NEVER A GATE: an unrecognised mode
# is accepted and surfaced in `unrecognised_plasticity_modes`, not rejected.
# Hardening a stub into a gate is how a placeholder becomes a permanent, wrong
# standard -- and it would refuse exactly the declarations the real inventory is
# expected to introduce.
PLASTICITY_MODES = (
    "parameters",
    "policy_value",
    "e1_e2_representations",
    "memory_state",
    "residue_ema_state",
    "offline_updates",
)


class CapabilityContractDeclarationError(ValueError):
    """The DECLARATION is malformed -- a programmer error, not a run outcome.

    Deliberately the only exception this module raises. An unmet contract is a
    scientific outcome and is routed, never raised: a raise invites a caller-side
    `except` that records `outcome: "FAIL"`, which is the exact error
    GOV-CAPCONTRACT-1 exists to prevent.
    """


# --- declarations ------------------------------------------------------------ #


@dataclass
class MechanismRequirement:
    """One mechanism the hypothesis presupposes, and how to verify it.

    Every probe is OPTIONAL and every probe may return None meaning UNDETERMINED.
    An absent probe is not a pass: it is simply not checked, and is reported as
    `not_declared` rather than folded into `satisfied`.

    constructed / enabled
        Duck-typed callables taking the organism (whatever the driver passes as
        `agent=`) and returning True / False / None. Use them when a boolean needs
        real logic.
    config_flag
        Convenience alternative to `enabled`: a dotted path resolved against the
        organism (attributes first, then mappings), e.g. "latent.e3_enabled".
        Resolving to a missing attribute yields UNDETERMINED, not False -- a
        renamed config field must not read as "the mechanism was off", which
        would silently manufacture a mis-instantiation verdict.
    reached_metric / reached_floor
        ARC-130 "endogenous recruitment" + "local operation": the key in the
        `engagement` dict measured during the run, and the floor its value must
        EXCEED for the test to be non-vacuous. This is the "decisive readouts
        engaged often enough" leg.
    authority_metric / authority_floor
        ARC-130 "competitive authority" + "committed throughput": non-zero
        influence is not necessarily competitive influence against the dominant
        arbitration term. Omit `authority_floor` when the claim does not turn on
        competitive authority; do not set it to 0.0 to mean "unchecked" -- a floor
        of 0.0 is a real, meetable check and will be evaluated as one.
    """

    name: str
    description: str = ""
    constructed: Optional[Callable[[Any], Optional[bool]]] = None
    enabled: Optional[Callable[[Any], Optional[bool]]] = None
    config_flag: Optional[str] = None
    reached_metric: Optional[str] = None
    reached_floor: float = 0.0
    authority_metric: Optional[str] = None
    authority_floor: Optional[float] = None
    claim_ids: Sequence[str] = ()
    note: str = ""

    def __post_init__(self) -> None:
        if not str(self.name).strip():
            raise CapabilityContractDeclarationError(
                "MechanismRequirement.name must be a non-empty string")
        if self.authority_metric is None and self.authority_floor is not None:
            raise CapabilityContractDeclarationError(
                f"mechanism '{self.name}': authority_floor is declared but "
                f"authority_metric is not -- the floor has nothing to read. "
                f"Name the metric, or drop the floor.")


@dataclass
class CapabilityRequirement:
    """A behavioural / functional capability the organism must be able to express.

    Distinct from a MECHANISM: a mechanism is a component of the organism, a
    capability is something the organism can do (ARC-131's distinction between a
    component-level PASS and a whole-agent expression). Verified either by a
    duck-typed `probe` or by a measured value against a floor -- typically a
    `capability_eval.py` metric, which is claim-agnostic by construction and so
    cannot beg the question the run is asking.
    """

    name: str
    description: str = ""
    probe: Optional[Callable[[Any], Optional[bool]]] = None
    metric: Optional[str] = None
    floor: Optional[float] = None
    direction: str = "lower"
    claim_ids: Sequence[str] = ()
    note: str = ""

    def __post_init__(self) -> None:
        if not str(self.name).strip():
            raise CapabilityContractDeclarationError(
                "CapabilityRequirement.name must be a non-empty string")
        if self.metric is not None and self.floor is None:
            raise CapabilityContractDeclarationError(
                f"capability '{self.name}': metric '{self.metric}' is declared "
                f"with no floor to compare it against.")
        if self.floor is not None and self.metric is None:
            raise CapabilityContractDeclarationError(
                f"capability '{self.name}': floor is declared but no metric names "
                f"what to measure.")


@dataclass
class PlasticityRequirement:
    """One form of within-life change the hypothesis requires to be POSSIBLE.

    `mode` should name a `PLASTICITY_MODES` entry, but an unrecognised value is
    ACCEPTED and surfaced rather than refused -- see that constant's comment: the
    real vocabulary is owed by chip-20260827-plasticity-inventory and a stub must
    not be able to refuse the declarations that inventory introduces.

    `load_bearing=False` declares a mode that is permitted but whose absence is
    not fatal to the interpretation: it is verified and reported, and it cannot
    trigger `nonplastic_misfire`. Use it to record what a run ALLOWED to change,
    as against what it DEPENDED on changing -- ARC-135's point that duration under
    a broken continuity is not developmental opportunity applies to both, but only
    the second invalidates a null.

    `requires_gradients` should be False for a mode that is not gradient-based
    (memory state, residue/EMA state), so a driver correctly running under
    `torch.no_grad()` is not routed `nonplastic_misfire` for it.
    """

    mode: str
    description: str = ""
    load_bearing: bool = True
    requires_gradients: bool = True
    requires_optimizer: bool = True
    requires_delta_witness: bool = True
    claim_ids: Sequence[str] = ()
    note: str = ""

    def __post_init__(self) -> None:
        if not str(self.mode).strip():
            raise CapabilityContractDeclarationError(
                "PlasticityRequirement.mode must be a non-empty string")


@dataclass
class CapabilityContract:
    """The full per-run declaration. Pre-registered; verified, never inferred."""

    experiment_id: str
    canonical_profile: Optional[str] = None
    canonical_profile_hash: Optional[str] = None
    declared_deviations: Mapping[str, Any] = field(default_factory=dict)
    requires_mechanisms: Sequence[MechanismRequirement] = ()
    requires_capabilities: Sequence[CapabilityRequirement] = ()
    requires_plasticity: Sequence[PlasticityRequirement] = ()
    claim_ids: Sequence[str] = ()
    note: str = ""

    def __post_init__(self) -> None:
        if not str(self.experiment_id).strip():
            raise CapabilityContractDeclarationError(
                "CapabilityContract.experiment_id must be a non-empty string")
        for group, label in ((self.requires_mechanisms, "requires_mechanisms"),
                             (self.requires_capabilities, "requires_capabilities"),
                             (self.requires_plasticity, "requires_plasticity")):
            names = [getattr(r, "name", getattr(r, "mode", "")) for r in group]
            dupes = sorted({n for n in names if names.count(n) > 1})
            if dupes:
                raise CapabilityContractDeclarationError(
                    f"{label}: duplicate entries {dupes} -- each requirement must "
                    f"be named once, or the manifest cannot attribute a verdict.")

    def unrecognised_plasticity_modes(self) -> List[str]:
        """Declared modes outside the PROVISIONAL stub vocabulary. Advisory only."""
        return sorted({p.mode for p in self.requires_plasticity
                       if p.mode not in PLASTICITY_MODES})


# --- probing helpers (all duck-typed; None always means UNDETERMINED) -------- #

_MISSING = object()


def _resolve_dotted(obj: Any, path: str) -> Any:
    """Attribute-then-mapping walk down a dotted path. `_MISSING` when absent.

    Never raises: an exploding property or a hostile __getattr__ yields _MISSING,
    which the caller turns into UNDETERMINED rather than False.
    """
    cur = obj
    for part in str(path).split("."):
        if cur is None:
            return _MISSING
        nxt = _MISSING
        try:
            nxt = getattr(cur, part)
        except Exception:
            nxt = _MISSING
        if nxt is _MISSING and isinstance(cur, Mapping):
            try:
                nxt = cur[part] if part in cur else _MISSING
            except Exception:
                nxt = _MISSING
        if nxt is _MISSING:
            # A config may hold its fields on a nested `.config` / `.cfg` object.
            for holder in ("config", "cfg"):
                try:
                    inner = getattr(cur, holder)
                except Exception:
                    continue
                try:
                    cand = getattr(inner, part, _MISSING)
                except Exception:
                    cand = _MISSING
                if cand is not _MISSING:
                    nxt = cand
                    break
        if nxt is _MISSING:
            return _MISSING
        cur = nxt
    return cur


def _call_probe(probe: Optional[Callable[[Any], Optional[bool]]],
                organism: Any) -> tuple:
    """(tri_state, detail). Any exception is UNDETERMINED-with-reason, never False.

    Fail open, loudly: a broken probe must not be able to manufacture a
    mis-instantiation verdict, and must not block the run either.
    """
    if probe is None:
        return None, "not declared"
    try:
        value = probe(organism)
    except Exception as exc:  # pragma: no cover - exercised via the tests' raiser
        return None, f"probe raised {type(exc).__name__}: {exc}"
    if value is None:
        return None, "probe returned None"
    return bool(value), ""


def _tri_from_flag(organism: Any, dotted: str) -> tuple:
    value = _resolve_dotted(organism, dotted)
    if value is _MISSING:
        return None, (f"config flag '{dotted}' not found on the organism -- "
                      f"UNDETERMINED, not False (a renamed field must not read "
                      f"as 'the mechanism was off')")
    if value is None:
        return None, f"config flag '{dotted}' is None"
    return bool(value), ""


def _metric_check(engagement: Mapping[str, Any], key: str, floor: float,
                  direction: str = "lower") -> tuple:
    """(tri_state, measured, detail). A missing metric is UNDETERMINED."""
    if key not in engagement:
        return None, None, (f"metric '{key}' was not measured this run -- "
                            f"UNDETERMINED")
    raw = engagement[key]
    try:
        measured = float(raw)
    except (TypeError, ValueError):
        return None, None, f"metric '{key}' is not numeric ({raw!r}) -- UNDETERMINED"
    if measured != measured:  # NaN
        return None, measured, f"metric '{key}' is NaN -- UNDETERMINED"
    if str(direction).strip().lower() in ("upper", "ceiling", "max", "upper_bound"):
        return bool(measured < float(floor)), measured, ""
    return bool(measured > float(floor)), measured, ""


# --- plasticity primitives --------------------------------------------------- #


def _iter_params(params: Any) -> List[Any]:
    if params is None:
        return []
    try:
        return list(params)
    except Exception:
        return []


def capture_parameter_witness(params: Any, *, max_tensors: int = 8,
                              coords_per_tensor: int = 4,
                              label: str = "") -> Dict[str, Any]:
    """Cheap parameter-delta witness -- safe to leave ON by default.

    GOV-CAPCONTRACT-1 requires "a parameter-delta witness confirms an actual
    update path", and the motivating audit found such witnesses existing ad hoc in
    exactly 4 experiment scripts. This is the shared, cheap one: it samples at
    most `max_tensors` tensors and `coords_per_tensor` DETERMINISTICALLY-CHOSEN
    scalars from each, plus each sampled tensor's float norm -- never a full-tensor
    snapshot, so the cost is O(max_tensors * coords) floats regardless of model
    size and the witness can be taken unconditionally at run start and end.

    Coordinates are chosen by even stride over the flattened tensor rather than at
    random, so the witness is reproducible across processes and machines and needs
    no RNG draw (which would perturb a seeded run's stream -- a real hazard on
    this substrate, where `torch.multinomial` already diverges across machine
    classes).

    Duck-typed: works with torch parameters, numpy arrays, or plain float
    sequences. Anything it cannot read is skipped and counted in `n_unreadable`
    rather than raising -- an unreadable witness becomes UNDETERMINED downstream,
    never a silent "no change".
    """
    tensors = _iter_params(params)
    sampled: List[Dict[str, Any]] = []
    n_unreadable = 0
    n_total = len(tensors)
    for idx, tensor in enumerate(tensors[:max_tensors]):
        entry = _sample_one_tensor(tensor, idx, coords_per_tensor)
        if entry is None:
            n_unreadable += 1
            continue
        sampled.append(entry)
    signature = hashlib.sha256(
        json.dumps(sampled, sort_keys=True, separators=(",", ":"),
                   ensure_ascii=True).encode("utf-8")).hexdigest()
    return {
        "schema": "parameter_delta_witness/v1",
        "label": str(label),
        "n_tensors_total": n_total,
        "n_tensors_sampled": len(sampled),
        "n_unreadable": n_unreadable,
        "coords_per_tensor": int(coords_per_tensor),
        "sampled": sampled,
        "signature": signature,
    }


def _sample_one_tensor(tensor: Any, position: int, coords: int) -> Optional[Dict[str, Any]]:
    """Deterministic small sample of one parameter tensor, or None if unreadable."""
    try:
        obj = tensor
        detach = getattr(obj, "detach", None)
        if callable(detach):
            obj = detach()
        flat = obj
        reshape = getattr(obj, "reshape", None)
        if callable(reshape):
            flat = reshape(-1)
        else:
            flat = list(obj) if not isinstance(obj, (int, float)) else [obj]
        n = len(flat)
        if n == 0:
            return {"position": position, "n": 0, "values": [], "norm": 0.0}
        take = max(1, min(int(coords), n))
        stride = max(1, n // take)
        values = [round(float(flat[min(i * stride, n - 1)]), 12) for i in range(take)]
        norm = 0.0
        norm_fn = getattr(flat, "norm", None)
        if callable(norm_fn):
            norm = float(norm_fn())
        else:
            norm = float(sum(float(v) * float(v) for v in flat)) ** 0.5
        return {"position": position, "n": int(n), "values": values,
                "norm": round(norm, 12)}
    except Exception:
        return None


def compare_parameter_witness(before: Optional[Mapping[str, Any]],
                              after: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Did the sampled parameters actually change? None means UNDETERMINED.

    `changed` is True / False / None. None whenever the comparison is not
    trustworthy -- a missing endpoint, a shape mismatch between the two captures,
    or zero readable tensors. It is NEVER False by default, because "no witness"
    and "witnessed no change" are exactly the two things GOV-CAPCONTRACT-1 forbids
    collapsing.
    """
    if not before or not after:
        return {"changed": None, "reason": "missing witness endpoint",
                "n_changed": None, "max_abs_delta": None}
    b_sampled = list(before.get("sampled") or [])
    a_sampled = list(after.get("sampled") or [])
    if not b_sampled or not a_sampled:
        return {"changed": None, "reason": "no readable parameters were sampled",
                "n_changed": None, "max_abs_delta": None}
    if len(b_sampled) != len(a_sampled):
        return {"changed": None,
                "reason": (f"witness shape changed between endpoints "
                           f"({len(b_sampled)} -> {len(a_sampled)} tensors) -- "
                           f"not comparable"),
                "n_changed": None, "max_abs_delta": None}
    n_changed = 0
    max_abs = 0.0
    for b_entry, a_entry in zip(b_sampled, a_sampled):
        if b_entry.get("n") != a_entry.get("n"):
            return {"changed": None,
                    "reason": (f"tensor at position {b_entry.get('position')} "
                               f"changed size between endpoints -- not comparable"),
                    "n_changed": None, "max_abs_delta": None}
        deltas = [abs(float(a) - float(b))
                  for a, b in zip(a_entry.get("values") or [],
                                  b_entry.get("values") or [])]
        deltas.append(abs(float(a_entry.get("norm", 0.0))
                          - float(b_entry.get("norm", 0.0))))
        entry_max = max(deltas) if deltas else 0.0
        if entry_max > 0.0:
            n_changed += 1
        max_abs = max(max_abs, entry_max)
    return {
        "changed": bool(n_changed > 0),
        "reason": "",
        "n_changed": n_changed,
        "n_compared": len(b_sampled),
        "max_abs_delta": max_abs,
        "signature_before": before.get("signature"),
        "signature_after": after.get("signature"),
    }


def learning_mode_snapshot(*, modules: Any = None,
                           grad_enabled: Optional[bool] = None) -> Dict[str, Any]:
    """Record train/eval mode and `torch.is_grad_enabled()` at the calling moment.

    The audit behind GOV-CAPCONTRACT-1 measured ZERO occurrences of
    `torch.is_grad_enabled()` in the whole tree, while the standard observational
    long-life drivers run under `torch.no_grad()` -- so "more observed ticks" has
    been silently equatable with "more developmental opportunity". This is the
    call that makes the distinction recordable.

    torch is imported INSIDE the function, so the module stays importable without
    it. `grad_enabled` overrides the probe (a caller that already knows, or a test
    that must be deterministic); an absent torch leaves it None = UNDETERMINED.
    """
    resolved = grad_enabled
    source = "caller"
    if resolved is None:
        source = "torch.is_grad_enabled()"
        try:
            import torch  # noqa: WPS433 - deliberate lazy import
            resolved = bool(torch.is_grad_enabled())
        except Exception:
            resolved = None
            source = "unavailable (torch not importable)"
    training: Dict[str, Any] = {}
    for module in _iter_params(modules):
        name = getattr(module, "_capability_contract_name", None) or \
            type(module).__name__
        flag = getattr(module, "training", None)
        training[str(name)] = None if flag is None else bool(flag)
    return {
        "grad_enabled": resolved,
        "grad_enabled_source": source,
        "module_training_mode": training,
        "n_modules_probed": len(training),
    }


def optimizer_membership(optimizers: Any, params: Any) -> Dict[str, Any]:
    """Are the intended trainable parameters actually inside an optimizer?

    Compares by `id()`, which is what an optimizer's param_groups hold. Returns
    `all_present=None` (UNDETERMINED) when there is nothing to compare -- no
    optimizers supplied, or no intended parameters -- rather than vacuously True.
    A vacuous True here would certify a frozen organism as plastic, which is the
    precise inversion GOV-CAPCONTRACT-1 forbids.
    """
    opts = _iter_params(optimizers)
    intended = _iter_params(params)
    if not opts:
        return {"all_present": None, "reason": "no optimizer supplied",
                "n_intended": len(intended), "n_present": None,
                "n_optimizers": 0}
    if not intended:
        return {"all_present": None,
                "reason": "no intended trainable parameters supplied",
                "n_intended": 0, "n_present": None, "n_optimizers": len(opts)}
    owned = set()
    for opt in opts:
        try:
            groups = list(getattr(opt, "param_groups", []) or [])
        except Exception:
            continue
        for group in groups:
            try:
                for p in group.get("params", []) or []:
                    owned.add(id(p))
            except Exception:
                continue
    if not owned:
        return {"all_present": None,
                "reason": ("optimizer(s) exposed no readable param_groups -- "
                           "UNDETERMINED, not False"),
                "n_intended": len(intended), "n_present": None,
                "n_optimizers": len(opts)}
    present = sum(1 for p in intended if id(p) in owned)
    return {
        "all_present": bool(present == len(intended)),
        "reason": "",
        "n_intended": len(intended),
        "n_present": present,
        "n_optimizers": len(opts),
    }


# --- verification ------------------------------------------------------------ #


def _check(name: str, kind: str, status: Optional[bool], detail: str,
           route: str, **extra: Any) -> Dict[str, Any]:
    if status is None:
        state = STATUS_UNDETERMINED
    elif status:
        state = STATUS_SATISFIED
    else:
        state = STATUS_UNMET
    record = {"name": name, "kind": kind, "status": state, "detail": detail,
              "route_if_unmet": route}
    record.update(extra)
    return record


def verify_capability_contract(
        contract: CapabilityContract,
        *,
        agent: Any = None,
        engagement: Optional[Mapping[str, Any]] = None,
        capabilities_measured: Optional[Mapping[str, Any]] = None,
        optimizers: Any = None,
        trainable_params: Any = None,
        parameter_witness_before: Optional[Mapping[str, Any]] = None,
        parameter_witness_after: Optional[Mapping[str, Any]] = None,
        learning_mode: Optional[Mapping[str, Any]] = None,
        observed_profile: Optional[str] = None,
        observed_profile_hash: Optional[str] = None,
        observed_deviations: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Verify a declared contract against the organism actually instantiated.

    Returns a dict with `satisfied`, `interpretation_route`, `routes_triggered`,
    `has_undetermined`, `checks`, and `manifest_block` (the value to store at
    `manifest["capability_contract"]`).

    NEVER raises on an unmet contract -- see the module docstring. `satisfied` is
    True when no check is UNMET; UNDETERMINED checks do not make it False, and are
    surfaced separately so a reader can see what was not verified.
    """
    engagement = dict(engagement or {})
    measured = dict(capabilities_measured or {})
    checks: List[Dict[str, Any]] = []

    # --- organism identity (ARC-130 stage 0: which REE was this?) ------------ #
    identity = _identity_block(contract, observed_profile, observed_profile_hash,
                               observed_deviations)
    checks.append(_check(
        "canonical_profile_identity", "identity",
        identity["matches"], identity["detail"],
        ROUTE_CAPABILITY_PRECONDITION_UNMET,
        declared=identity["declared"], observed=identity["observed"],
        undeclared_deviations=identity["undeclared_deviations"]))

    # --- mechanisms ---------------------------------------------------------- #
    for mech in contract.requires_mechanisms:
        checks.extend(_verify_mechanism(mech, agent, engagement))

    # --- capabilities -------------------------------------------------------- #
    for cap in contract.requires_capabilities:
        checks.extend(_verify_capability(cap, agent, measured))

    # --- plasticity ---------------------------------------------------------- #
    mode_info = dict(learning_mode or {})
    if not mode_info:
        mode_info = learning_mode_snapshot()
    delta = compare_parameter_witness(parameter_witness_before,
                                      parameter_witness_after)
    membership = optimizer_membership(optimizers, trainable_params)
    for plastic in contract.requires_plasticity:
        checks.extend(_verify_plasticity(plastic, mode_info, membership, delta))

    unmet = [c for c in checks if c["status"] == STATUS_UNMET]
    undetermined = [c for c in checks if c["status"] == STATUS_UNDETERMINED]
    routes = [r for r in ROUTE_PRECEDENCE
              if any(c["route_if_unmet"] == r for c in unmet)]
    route = routes[0] if routes else ROUTE_INTERPRETABLE

    result: Dict[str, Any] = {
        "schema": CONTRACT_SCHEMA,
        "experiment_id": contract.experiment_id,
        "satisfied": not unmet,
        "interpretation_route": route,
        "routes_triggered": routes,
        "has_undetermined": bool(undetermined),
        "checks": checks,
        "unmet_checks": [c["name"] for c in unmet],
        "undetermined_checks": [c["name"] for c in undetermined],
        "identity": identity,
        "learning_mode": mode_info,
        "optimizer_membership": membership,
        "parameter_delta": delta,
        "unrecognised_plasticity_modes": contract.unrecognised_plasticity_modes(),
        "claim_ids": list(contract.claim_ids),
    }
    result["manifest_block"] = _manifest_block(contract, result)
    return result


def _identity_block(contract: CapabilityContract,
                    observed_profile: Optional[str],
                    observed_profile_hash: Optional[str],
                    observed_deviations: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Canonical profile name/hash plus EXPLICIT deviations.

    UNDETERMINED (matches=None) when either side is absent -- which is the honest
    reading for every pre-2026-08-12 manifest and any run not built from a
    declared profile, and, today, for `ree_v3_baseline@v0` itself: GOV-CAPCONTRACT-1's
    notes record that it is a PLACEHOLDER with `"overrides": {}` that no admission
    pass has ever populated, so "which REE did this experiment test?" is currently
    unanswerable by construction. Reporting that as a MATCH would certify an
    identity nobody has established; reporting it as a MISMATCH would route every
    run today to `capability_precondition_unmet` and make the contract unusable.
    UNDETERMINED is the only reading that is neither.
    """
    declared = {"canonical_profile": contract.canonical_profile,
                "canonical_profile_hash": contract.canonical_profile_hash,
                "declared_deviations": dict(contract.declared_deviations or {})}
    observed = {"canonical_profile": observed_profile,
                "canonical_profile_hash": observed_profile_hash,
                "deviations": dict(observed_deviations or {})}

    undeclared = {}
    for key, value in (observed_deviations or {}).items():
        if key not in (contract.declared_deviations or {}):
            undeclared[key] = value

    if contract.canonical_profile is None or observed_profile is None:
        return {"matches": None, "declared": declared, "observed": observed,
                "undeclared_deviations": undeclared,
                "detail": ("canonical profile not declared on both sides -- "
                           "UNDETERMINED (honest for a hand-assembled config and "
                           "for the unpopulated ree_v3_baseline@v0 placeholder)")}
    if str(contract.canonical_profile) != str(observed_profile):
        return {"matches": False, "declared": declared, "observed": observed,
                "undeclared_deviations": undeclared,
                "detail": (f"declared profile {contract.canonical_profile!r} but "
                           f"the organism was built from {observed_profile!r}")}
    if (contract.canonical_profile_hash and observed_profile_hash
            and str(contract.canonical_profile_hash) != str(observed_profile_hash)):
        return {"matches": False, "declared": declared, "observed": observed,
                "undeclared_deviations": undeclared,
                "detail": ("canonical_profile_hash differs -- the profile's content "
                           "changed under its own name")}
    if undeclared:
        return {"matches": False, "declared": declared, "observed": observed,
                "undeclared_deviations": undeclared,
                "detail": (f"organism carries {len(undeclared)} deviation(s) the "
                           f"contract did not declare: "
                           f"{', '.join(sorted(undeclared))}")}
    return {"matches": True, "declared": declared, "observed": observed,
            "undeclared_deviations": {}, "detail": ""}


def _verify_mechanism(mech: MechanismRequirement, agent: Any,
                      engagement: Mapping[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []

    constructed, detail = _call_probe(mech.constructed, agent)
    if mech.constructed is not None:
        out.append(_check(f"{mech.name}::constructed", "mechanism", constructed,
                          detail, ROUTE_CAPABILITY_PRECONDITION_UNMET,
                          mechanism=mech.name, claim_ids=list(mech.claim_ids),
                          description=mech.description))

    if mech.enabled is not None:
        enabled, detail = _call_probe(mech.enabled, agent)
        out.append(_check(f"{mech.name}::enabled", "mechanism", enabled, detail,
                          ROUTE_CAPABILITY_PRECONDITION_UNMET,
                          mechanism=mech.name, claim_ids=list(mech.claim_ids)))
    elif mech.config_flag is not None:
        enabled, detail = _tri_from_flag(agent, mech.config_flag)
        out.append(_check(f"{mech.name}::enabled", "mechanism", enabled, detail,
                          ROUTE_CAPABILITY_PRECONDITION_UNMET,
                          mechanism=mech.name, config_flag=mech.config_flag,
                          claim_ids=list(mech.claim_ids)))

    if mech.reached_metric is not None:
        reached, value, detail = _metric_check(engagement, mech.reached_metric,
                                               mech.reached_floor)
        out.append(_check(f"{mech.name}::reached", "mechanism", reached, detail,
                          ROUTE_MECHANISM_UNREACHED, mechanism=mech.name,
                          metric=mech.reached_metric, measured=value,
                          threshold=float(mech.reached_floor), direction="lower",
                          claim_ids=list(mech.claim_ids)))

    if mech.authority_metric is not None and mech.authority_floor is not None:
        authoritative, value, detail = _metric_check(engagement,
                                                     mech.authority_metric,
                                                     mech.authority_floor)
        out.append(_check(f"{mech.name}::authority", "mechanism", authoritative,
                          detail, ROUTE_AUTHORITY_FLOOR_UNMET,
                          mechanism=mech.name, metric=mech.authority_metric,
                          measured=value, threshold=float(mech.authority_floor),
                          direction="lower", claim_ids=list(mech.claim_ids)))
    return out


def _verify_capability(cap: CapabilityRequirement, agent: Any,
                       measured: Mapping[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if cap.probe is not None:
        state, detail = _call_probe(cap.probe, agent)
        out.append(_check(f"{cap.name}::probe", "capability", state, detail,
                          ROUTE_CAPABILITY_PRECONDITION_UNMET,
                          capability=cap.name, claim_ids=list(cap.claim_ids),
                          description=cap.description))
    if cap.metric is not None and cap.floor is not None:
        state, value, detail = _metric_check(measured, cap.metric, cap.floor,
                                             cap.direction)
        out.append(_check(f"{cap.name}::floor", "capability", state, detail,
                          ROUTE_CAPABILITY_PRECONDITION_UNMET,
                          capability=cap.name, metric=cap.metric, measured=value,
                          threshold=float(cap.floor), direction=cap.direction,
                          claim_ids=list(cap.claim_ids)))
    return out


def _verify_plasticity(plastic: PlasticityRequirement,
                       mode_info: Mapping[str, Any],
                       membership: Mapping[str, Any],
                       delta: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Gradients / optimizer membership / parameter-delta witness for one mode.

    A NON-load-bearing mode is verified and reported but routed
    `interpretable`: it can never trigger `nonplastic_misfire`, because its
    absence does not invalidate the null.
    """
    route = ROUTE_NONPLASTIC_MISFIRE if plastic.load_bearing else ROUTE_INTERPRETABLE
    out: List[Dict[str, Any]] = []
    common = {"plasticity_mode": plastic.mode,
              "load_bearing": bool(plastic.load_bearing),
              "claim_ids": list(plastic.claim_ids)}

    if plastic.requires_gradients:
        grad = mode_info.get("grad_enabled")
        detail = "" if grad is True else (
            "gradients were DISABLED for this run -- the organism could not have "
            "acquired anything gradient-based, so a null here is 'could not have "
            "learned', not 'did not learn'" if grad is False else
            f"grad state undetermined ({mode_info.get('grad_enabled_source')})")
        out.append(_check(f"plasticity::{plastic.mode}::gradients_enabled",
                          "plasticity", grad, detail, route, **common))

    if plastic.requires_optimizer:
        present = membership.get("all_present")
        detail = "" if present is True else (
            f"{membership.get('n_present')} of {membership.get('n_intended')} "
            f"intended trainable parameters are in an optimizer"
            if present is False else str(membership.get("reason", "")))
        out.append(_check(f"plasticity::{plastic.mode}::optimizer_membership",
                          "plasticity", present, detail, route, **common))

    if plastic.requires_delta_witness:
        changed = delta.get("changed")
        detail = "" if changed is True else (
            "the parameter-delta witness shows NO change between run start and "
            "end -- there was no actual update path" if changed is False else
            str(delta.get("reason", "")))
        out.append(_check(f"plasticity::{plastic.mode}::parameter_delta_witness",
                          "plasticity", changed, detail, route,
                          max_abs_delta=delta.get("max_abs_delta"), **common))
    return out


def _manifest_block(contract: CapabilityContract,
                    result: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "schema": CONTRACT_SCHEMA,
        "claim": "GOV-CAPCONTRACT-1",
        "experiment_id": contract.experiment_id,
        "satisfied": result["satisfied"],
        "interpretation_route": result["interpretation_route"],
        "routes_triggered": list(result["routes_triggered"]),
        "has_undetermined": result["has_undetermined"],
        "unmet_checks": list(result["unmet_checks"]),
        "undetermined_checks": list(result["undetermined_checks"]),
        "checks": list(result["checks"]),
        "identity": result["identity"],
        "learning_mode": result["learning_mode"],
        "optimizer_membership": result["optimizer_membership"],
        "parameter_delta": result["parameter_delta"],
        "declared": {
            "requires_mechanisms": [m.name for m in contract.requires_mechanisms],
            "requires_capabilities": [c.name for c in contract.requires_capabilities],
            "requires_plasticity": [p.mode for p in contract.requires_plasticity],
        },
        "unrecognised_plasticity_modes": list(
            result["unrecognised_plasticity_modes"]),
        "plasticity_vocabulary_status": (
            "PROVISIONAL -- the within-life plasticity inventory owed by "
            "chip-20260827-plasticity-inventory does not exist yet; "
            "PLASTICITY_MODES transcribes the GOV-CAPCONTRACT-1 claim title's own "
            "enumeration and validation against it is advisory, not a gate"),
        "scope_note": (
            "WHOLE-ORGANISM, whole-run by construction: every check here is a "
            "property of the single organism instantiated for this run and cannot "
            "differ between arms. This is NOT the V3-EXQ-785 shape (an ARM-scoped "
            "fact promoted to run scope) -- see precondition_gate.py, which owns "
            "per-arm non-vacuity and interpretation.preconditions[]; this block "
            "does not write there."),
    }


# --- routing ----------------------------------------------------------------- #


def route_for_manifest(result: Mapping[str, Any]) -> Dict[str, Any]:
    """Manifest fields that route an unmet contract AWAY from the evidence record.

    Returns `{}` when the contract is satisfied -- a met contract changes nothing
    about how the run is recorded; it only licenses the recording that was already
    intended.

    When unmet, returns the two channels `build_experiment_indexes.py` already
    honours, so no indexer change is needed:
      experiment_purpose="diagnostic"     -> scoring_excluded="diagnostic_probe"
      evidence_direction="non_contributory" -> scoring_excluded="non_contributory"
    plus `outcome="DIAGNOSTIC"`, `non_degenerate=False` and a `degeneracy_reason`
    naming the route.

    `outcome` is deliberately NOT "FAIL". Both fields are set because they exclude
    at different granularities -- `experiment_purpose` run-wide,
    `evidence_direction` per tagged claim -- and a run whose organism could not
    express the faculty is excluded at BOTH.
    """
    if result.get("satisfied"):
        return {}
    route = result.get("interpretation_route", ROUTE_CAPABILITY_PRECONDITION_UNMET)
    unmet = ", ".join(result.get("unmet_checks") or []) or "(unnamed)"
    reason = (
        f"GOV-CAPCONTRACT-1: capability/plasticity preflight UNMET -- route "
        f"'{route}'. Unmet: {unmet}. The organism instantiated for this run was "
        f"not demonstrably able to express (or, where required, to acquire) the "
        f"faculty under test, so this run is NOT admissible as negative evidence "
        f"for the claim. 'It did not learn' and 'it could not have learned' are "
        f"different scientific outcomes. Read this run as UNSCORED, not as a "
        f"refutation.")
    return {
        "outcome": ROUTED_OUTCOME,
        "experiment_purpose": "diagnostic",
        "evidence_direction": "non_contributory",
        "non_degenerate": False,
        "degeneracy_reason": reason,
        "capability_contract_route": route,
    }


def assert_not_recorded_as_fail(manifest: Mapping[str, Any],
                                result: Mapping[str, Any]) -> None:
    """Refuse to write a manifest that records an unmet contract as negative evidence.

    The one place this module raises about a RUN rather than a declaration, and it
    is a guard against the CALLER, not a verdict about the organism: it fires only
    when a driver has already been told the contract is unmet and is nonetheless
    about to record it against the claim. That must not be reachable by forgetting
    to call `route_for_manifest()`.

    IT CHECKS TWO DOORS, BECAUSE "RECORDED AS NEGATIVE EVIDENCE" HAS TWO.

    (1) The obvious one: `outcome: "FAIL"`.

    (2) The quiet one, which is the reason this function is not a one-liner.
        `build_experiment_indexes.py` INFERS a direction when the manifest does
        not set one:

            if inferred_direction == "unknown" and not direction_explicitly_set:
                inferred_direction = "supports" if final_status == "PASS" else "weakens"

        `final_status` is derived from the manifest's `outcome`. So a driver that
        applies the `outcome` override but drops `evidence_direction` gets
        `weakens` inferred from a NON-"PASS" outcome -- the prohibited recording,
        arriving through a different door and without the word FAIL appearing
        anywhere. `route_for_manifest()` returns both fields precisely so they are
        applied as one bundle; this check is what makes splitting them fail loudly
        instead of silently.
    """
    if result.get("satisfied"):
        return
    unmet = ", ".join(result.get("unmet_checks") or []) or "(unnamed)"
    preamble = (
        f"run whose GOV-CAPCONTRACT-1 capability preflight was UNMET (route "
        f"'{result.get('interpretation_route')}', unmet: {unmet}). A "
        f"mis-instantiated organism's null is not negative evidence. Apply "
        f"route_for_manifest(result) to the manifest -- it sets "
        f"outcome={ROUTED_OUTCOME}, experiment_purpose=diagnostic and "
        f"evidence_direction=non_contributory together, as one bundle.")

    outcome = str(manifest.get("outcome", "")).strip().upper()
    if outcome == "FAIL":
        raise CapabilityContractDeclarationError(
            "refusing to record outcome=FAIL for a " + preamble)

    direction = str(manifest.get("evidence_direction", "")).strip().lower()
    if direction not in _EXCLUDED_DIRECTIONS:
        raise CapabilityContractDeclarationError(
            f"refusing to write evidence_direction={direction or '(unset)'!r} for "
            f"a {preamble} With outcome={outcome or '(unset)'!r} and no explicit "
            f"excluded direction, the indexer INFERS 'weakens' "
            f"(build_experiment_indexes.py: \"supports\" if final_status == "
            f"\"PASS\" else \"weakens\") -- the prohibited recording without the "
            f"word FAIL appearing anywhere. Set one of "
            f"{sorted(_EXCLUDED_DIRECTIONS)}.")


def as_adjudication_preconditions(result: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """OPT-IN flat entries for `interpretation.preconditions[]`.

    ONLY for a driver that uses NO per-arm gate. If the run also calls
    `precondition_gate.aggregate_arm_gates()`, that function OWNS
    `interpretation.preconditions` and this list must not be merged into it: two
    writers to the arm-blind list is how V3-EXQ-785 was re-vacated at adjudication
    time.

    Safe here in the whole-run case because these checks genuinely are whole-run
    (see the module docstring): an unmet one warrants exactly the whole-run
    `precondition_unmet` the indexer derives. UNDETERMINED checks are OMITTED --
    the indexer recomputes `met` from measured+threshold and cannot represent
    "not verified", so emitting one would force it to read as either a pass or a
    fail, both of which are lies.
    """
    out: List[Dict[str, Any]] = []
    for check in result.get("checks") or []:
        if check.get("status") == STATUS_UNDETERMINED:
            continue
        entry = {
            "name": f"capability_contract::{check['name']}",
            "kind": check.get("kind", "capability"),
            "description": check.get("description")
            or check.get("detail") or check["name"],
            "control": "GOV-CAPCONTRACT-1 whole-organism capability contract",
            "met": check["status"] == STATUS_SATISFIED,
        }
        if check.get("measured") is not None and check.get("threshold") is not None:
            entry["measured"] = check["measured"]
            entry["threshold"] = check["threshold"]
            entry["direction"] = check.get("direction", "lower")
        out.append(entry)
    return out


def format_report(result: Mapping[str, Any]) -> str:
    """ASCII-only human summary for a driver's stdout."""
    lines = [
        f"capability contract [{result.get('experiment_id')}] -> "
        f"{result.get('interpretation_route')}",
        f"  satisfied: {result.get('satisfied')}   "
        f"undetermined: {result.get('has_undetermined')}",
    ]
    for check in result.get("checks") or []:
        mark = {STATUS_SATISFIED: "ok ", STATUS_UNMET: "NO ",
                STATUS_UNDETERMINED: "?? "}[check["status"]]
        detail = f"  -- {check['detail']}" if check.get("detail") else ""
        lines.append(f"  [{mark}] {check['name']}{detail}")
    if result.get("unrecognised_plasticity_modes"):
        lines.append("  note: plasticity mode(s) outside the PROVISIONAL stub "
                     "vocabulary (advisory, not a gate): "
                     + ", ".join(result["unrecognised_plasticity_modes"]))
    if not result.get("satisfied"):
        lines.append("  ROUTE: this run is NOT admissible as negative evidence. "
                     "Apply route_for_manifest(); do NOT record it as FAIL.")
    return "\n".join(lines)
