"""Action-sensitivity readiness gate for world-forward / transition heads.

WHAT THIS AUTHORISES, AND WHY THAT MAKES IT DANGEROUS
------------------------------------------------------
This gate answers one question: DOES THE HEAD READ ITS ACTION? A consolidation
experiment that poses a contradiction (an inverted action-map, a "confidently
wrong" condition, a novel-target replay arm) is only interpretable if the answer
is yes. If the head is near copy-the-input, an action-map inversion is not a
contradiction at all, every contradiction-conditioned contrast degenerates, and
the run produces an instrument record rather than a result.

That happened. V3-EXQ-1073 (seeds 42/123/456, 73.1 min, 72 cells) reached
conv_rel_drop 0.997-0.999 on e2.world_forward and STILL could not pose its
condition 3 -- the inverted-rule battery was EASIER than the rule the head was
trained on, on all three seeds. The run self-routed
`confidently_wrong_condition_unposeable_on_this_head`. Full record:
REE_assembly/evidence/planning/failure_autopsy_V3-EXQ-1073_2026-09-22.md.

So this gate exists to be run BEFORE that kind of experiment, not after.

THIS IS A NEGATIVE INSTRUMENT -- READ CLAUDE.md "Negative instruments"
----------------------------------------------------------------------
The gate authorises STARTING work on the strength of a negative ("no
action-blindness found -> proceed"). Its numerator and its denominator come from
the same computation, so one bug destroys both silently and the verdict reads
"ready". Three remedies are applied here, strongest first:

1. STRUCTURAL cannot-determine CATEGORY. `ActionSensitivityVerdict.status` is a
   three-valued string, never a bool, and `cannot_determine` is a first-class
   value that propagates into `to_dict()` / --json output. A print statement
   would be a courtesy the next edit can drop; a field survives refactoring.

2. KNOWN-BASELINE CANARY. `CANARY_V3_EXQ_1073` pins that run's measured values.
   `check_canary()` must keep reproducing them. This is the ONLY one of the three
   remedies that catches a PARTIALLY broken gate -- a gate that still returns
   plausible numbers but computes them wrongly.

3. PRINTED PRE-FILTER DENOMINATOR. Every verdict carries `n_rows` and
   `n_distinct_actions`, and `format_verdict()` prints them. A ratio computed
   over 3 rows is not evidence, and the reader must be able to see that.

THE MONOSTRATEGY TRAP (the reason `cannot_determine` is not decoration)
-----------------------------------------------------------------------
REE agents are monostrategy for long stretches -- a real 20-tick rollout gave
[0,3,3,3,...]. If a battery contains ONE distinct action, permuting actions is a
no-op: the ratio comes back EXACTLY 1.0. Read as a bool that is "not > 1", i.e.
"action_blind" -- and it is nothing of the kind, it is a battery that cannot test
the question. Silently converting that into a negative verdict would be the exact
fail-closed-on-a-broken-search shape this module is written to prevent. So a
battery with fewer than 2 distinct actions returns `cannot_determine`, always.

ASCII-only output (CLAUDE.md "ASCII-Only in Python Output").
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import torch

__all__ = [
    "ActionSensitivityVerdict",
    "CANARY_V3_EXQ_1073",
    "action_shuffle_ratio",
    "battery_pair_ratio",
    "check_canary",
    "format_verdict",
    "identity_predictor_mse",
    "readiness_verdict",
    "skill_vs_identity",
]

# --------------------------------------------------------------------------- #
# Defaults. Every threshold is a keyword argument; these are the defaults the   #
# V3-EXQ-1073 autopsy's acceptance criteria imply, not hard-coded policy.       #
# --------------------------------------------------------------------------- #
MIN_ROWS = 16
MIN_DISTINCT_ACTIONS = 2
RATIO_FLOOR = 1.0       # inverted/shuffled MSE must EXCEED original MSE
SKILL_FLOOR = 0.0       # skill vs the copy-the-input predictor must exceed 0


@dataclass
class ActionSensitivityVerdict:
    """Three-valued readiness verdict. `status` is NEVER a bool.

    status:
      "ready"            -- head reads its action; a contradiction can be posed.
      "action_blind"     -- MEASURED action-invariance; do not pose a contradiction.
      "cannot_determine" -- the battery could not test the question. NOT a
                            negative result. Fix the battery, do not proceed.
    """
    status: str
    reason: str
    ratio: Optional[float] = None
    skill: Optional[float] = None
    model_mse: Optional[float] = None
    identity_mse: Optional[float] = None
    counterfactual_mse: Optional[float] = None
    n_rows: int = 0
    n_distinct_actions: int = 0
    ratio_floor: float = RATIO_FLOOR
    skill_floor: float = SKILL_FLOOR
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_ready(self) -> bool:
        """True ONLY for "ready". cannot_determine is False here -- but callers
        must branch on `status`, not on this, or they collapse the third value
        back into the bool this module exists to avoid."""
        return self.status == "ready"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _finite(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def _n_distinct_actions(actions: torch.Tensor) -> int:
    """Distinct action ROWS. Works for one-hot and for continuous actions."""
    if actions.ndim == 1:
        return int(torch.unique(actions).numel())
    return int(torch.unique(actions, dim=0).shape[0])


def identity_predictor_mse(z0: torch.Tensor, z1: torch.Tensor) -> float:
    """mean((z1 - z0)^2) -- what a predictor that simply COPIES its input scores.

    This is the reference MECH-573 requires. conv_rel_drop is normalised by the
    RANDOM-INIT battery MSE, so it cannot by itself distinguish "predicts
    transitions" from "learned to output ~no change": a head at the copy-the-input
    value can show conv_rel_drop 0.999 and have no skill whatever.

    MECH-573's non-degeneracy precondition requires this be computed on the SAME
    frozen battery as the model readout, in the same units, within the same
    no-grad evaluation. Callers must honour that; this function cannot check it.
    """
    with torch.no_grad():
        return float(((z1 - z0) ** 2).mean().item())


def skill_vs_identity(model_mse: float, identity_mse: float) -> Optional[float]:
    """1 - MSE_model / MSE_identity. None when undefined rather than 0.0.

    Returns None (not a number) when identity_mse is zero or either input is
    non-finite -- a zero denominator means the battery has no movement to predict,
    which is a cannot-determine, not a skill of zero.
    """
    if not (_finite(model_mse) and _finite(identity_mse)):
        return None
    if identity_mse == 0.0:
        return None
    return 1.0 - (float(model_mse) / float(identity_mse))


def _mse(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float(((pred - target) ** 2).mean().item())


def action_shuffle_ratio(
    head: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    z0: torch.Tensor,
    actions: torch.Tensor,
    z1: torch.Tensor,
    generator: Optional[torch.Generator] = None,
) -> Tuple[Optional[float], float, Optional[float], int]:
    """SD-031's action-shuffled control: MSE under permuted actions / MSE under true.

    SD-031's Validation Experiment P1 requires an "Identity-collapse check: r2 on
    action-shuffled control must drop substantially" (registered 2026-04-18, never
    built until SD-PP-B5). This is that check, expressed as an MSE ratio so it is
    comparable with the inverted-rule battery ratio in `battery_pair_ratio`.

    A head that reads its action predicts WORSE under permuted actions, so the
    ratio EXCEEDS 1. A copy-the-input head is unaffected and the ratio sits at ~1.

    The permutation is derangement-biased: rows whose permuted action equals their
    original action carry no signal, so they are rolled until they differ where the
    battery makes that possible.

    Returns (ratio, mse_true, mse_shuffled, n_distinct_actions). ratio is None when
    undefined.
    """
    with torch.no_grad():
        n = int(z0.shape[0])
        n_distinct = _n_distinct_actions(actions)
        mse_true = _mse(head(z0, actions), z1)
        if n < 2 or n_distinct < MIN_DISTINCT_ACTIONS:
            # Permuting is a no-op; refuse to manufacture a ratio of 1.0.
            return None, mse_true, None, n_distinct

        perm = torch.randperm(n, generator=generator, device=actions.device)
        shuffled = actions[perm]
        # Roll the rows that landed on their own action, where possible.
        for _ in range(8):
            if actions.ndim == 1:
                same = shuffled == actions
            else:
                same = (shuffled == actions).all(dim=-1)
            if not bool(same.any()):
                break
            idx = torch.nonzero(same, as_tuple=False).flatten()
            shuffled[idx] = actions[torch.roll(idx, 1)]

        mse_shuffled = _mse(head(z0, shuffled), z1)
        if not (_finite(mse_true) and _finite(mse_shuffled)):
            return None, mse_true, mse_shuffled, n_distinct
        if mse_true == 0.0:
            # A PERFECT head. Not undetermined: if permuting the action makes it
            # wrong, it is maximally action-sensitive; if it stays perfect, the
            # battery cannot separate the actions at all.
            return (math.inf if mse_shuffled > 0.0 else None), \
                mse_true, mse_shuffled, n_distinct
        return mse_shuffled / mse_true, mse_true, mse_shuffled, n_distinct


def battery_pair_ratio(
    head: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    original: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    counterfactual: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> Tuple[Optional[float], float, float]:
    """MSE on a counterfactual-rule battery / MSE on the original-rule battery.

    This is the form V3-EXQ-1073 measured (0.760 / 0.901 / 0.881 -- all BELOW 1,
    i.e. the inverted rule was EASIER). It needs the caller to have generated a
    second battery under the counterfactual action-map; `action_shuffle_ratio` is
    the cheap form that needs no environment re-roll.

    Each battery is (z0, actions, z1). Returns (ratio, mse_original, mse_cf).
    """
    with torch.no_grad():
        mse_o = _mse(head(original[0], original[1]), original[2])
        mse_c = _mse(head(counterfactual[0], counterfactual[1]), counterfactual[2])
        if not (_finite(mse_o) and _finite(mse_c)):
            return None, mse_o, mse_c
        if mse_o == 0.0:
            # See action_shuffle_ratio: a perfect head is determinable.
            return (math.inf if mse_c > 0.0 else None), mse_o, mse_c
        return mse_c / mse_o, mse_o, mse_c


def readiness_verdict(
    head: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    z0: torch.Tensor,
    actions: torch.Tensor,
    z1: torch.Tensor,
    counterfactual_battery: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    min_rows: int = MIN_ROWS,
    min_distinct_actions: int = MIN_DISTINCT_ACTIONS,
    ratio_floor: float = RATIO_FLOOR,
    skill_floor: float = SKILL_FLOOR,
    generator: Optional[torch.Generator] = None,
) -> ActionSensitivityVerdict:
    """The gate. Returns a three-valued verdict; never raises on a thin battery.

    Uses `counterfactual_battery` when supplied (the V3-EXQ-1073 form), otherwise
    the action-shuffled control (SD-031's form). Both must clear `ratio_floor`,
    AND the head must clear `skill_floor` against the copy-the-input predictor
    (MECH-573) -- a head with no skill has no residual worth contradicting.
    """
    n_rows = int(z0.shape[0]) if z0.ndim > 0 else 0
    n_distinct = _n_distinct_actions(actions) if n_rows else 0

    def cd(reason: str, **kw: Any) -> ActionSensitivityVerdict:
        return ActionSensitivityVerdict(
            status="cannot_determine", reason=reason, n_rows=n_rows,
            n_distinct_actions=n_distinct, ratio_floor=ratio_floor,
            skill_floor=skill_floor, **kw)

    if n_rows < min_rows:
        return cd("battery too small: %d rows < %d" % (n_rows, min_rows))
    if n_distinct < min_distinct_actions:
        return cd(
            "battery has %d distinct action(s) < %d -- an action-map manipulation "
            "is a no-op here, so this battery cannot test action-sensitivity "
            "(monostrategy rollout)" % (n_distinct, min_distinct_actions))

    with torch.no_grad():
        model_mse = _mse(head(z0, actions), z1)
    identity_mse = identity_predictor_mse(z0, z1)
    skill = skill_vs_identity(model_mse, identity_mse)

    if skill is None:
        return cd(
            "skill undefined: identity_predictor_mse=%r model_mse=%r (a zero or "
            "non-finite denominator is not a skill of zero)" % (identity_mse, model_mse),
            model_mse=model_mse if _finite(model_mse) else None,
            identity_mse=identity_mse if _finite(identity_mse) else None)

    if counterfactual_battery is not None:
        ratio, mse_o, mse_c = battery_pair_ratio(
            head, (z0, actions, z1), counterfactual_battery)
        form = "counterfactual_battery"
        cf_mse: Optional[float] = mse_c
    else:
        ratio, mse_o, mse_c, _ = action_shuffle_ratio(
            head, z0, actions, z1, generator=generator)
        form = "action_shuffle"
        cf_mse = mse_c

    if ratio is None or (not _finite(ratio) and ratio != math.inf):
        return cd(
            "ratio undefined via %s (original_mse=%r counterfactual_mse=%r)"
            % (form, mse_o, cf_mse),
            skill=skill, model_mse=model_mse, identity_mse=identity_mse,
            counterfactual_mse=cf_mse if _finite(cf_mse) else None)

    common = dict(
        ratio=ratio, skill=skill, model_mse=model_mse, identity_mse=identity_mse,
        counterfactual_mse=cf_mse, n_rows=n_rows, n_distinct_actions=n_distinct,
        ratio_floor=ratio_floor, skill_floor=skill_floor,
        extra={"form": form},
    )

    failures = []
    if not (ratio > ratio_floor):
        failures.append("ratio %.4f <= %.4f (an action-map change is not a "
                        "contradiction for this head)" % (ratio, ratio_floor))
    if not (skill > skill_floor):
        failures.append("skill %.4f <= %.4f vs the copy-the-input predictor "
                        "(MECH-573)" % (skill, skill_floor))

    if failures:
        return ActionSensitivityVerdict(
            status="action_blind", reason="; ".join(failures), **common)
    return ActionSensitivityVerdict(
        status="ready",
        reason="ratio %.4f > %.4f and skill %.4f > %.4f"
               % (ratio, ratio_floor, skill, skill_floor),
        **common)


# --------------------------------------------------------------------------- #
# Canary -- remedy (2). The ONLY check that catches a PARTIALLY broken gate.    #
# --------------------------------------------------------------------------- #
CANARY_V3_EXQ_1073: Dict[str, Any] = {
    "run_id": "v3_exq_1073_mech572_precision_provenance_gain_20260922T182856Z_v3",
    "source": "REE_assembly/evidence/planning/failure_autopsy_V3-EXQ-1073_2026-09-22.md",
    "seeds": [42, 123, 456],
    # Inverted-rule battery pre-sleep MSE / original-rule MSE.
    "battery_ratio": [0.760, 0.901, 0.881],
    # 1 - MSE_model / MSE_identity on the same frozen battery (MECH-573).
    "skill_vs_identity": [-0.071, 0.227, -0.007],
    # Every seed must classify action_blind: ratio <= 1 on 3/3, and skill > 0 on
    # only 1/3. The gate must NOT return "ready" for any of them.
    "expected_status": ["action_blind", "action_blind", "action_blind"],
}


def check_canary(tol: float = 1e-9) -> Dict[str, Any]:
    """Replay V3-EXQ-1073's measured values through the gate's OWN decision logic.

    A gate that has silently broken -- a flipped comparison, a reciprocal ratio, a
    skill formula inverted -- will classify these known values differently. This
    reproduces the classification only; it cannot re-measure the run.

    Returns a dict with "ok" plus per-seed detail. Never raises.
    """
    results = []
    ok = True
    z = zip(CANARY_V3_EXQ_1073["seeds"],
            CANARY_V3_EXQ_1073["battery_ratio"],
            CANARY_V3_EXQ_1073["skill_vs_identity"],
            CANARY_V3_EXQ_1073["expected_status"])
    for seed, ratio, skill, expected in z:
        blind = (not (ratio > RATIO_FLOOR)) or (not (skill > SKILL_FLOOR))
        got = "action_blind" if blind else "ready"
        match = (got == expected)
        ok = ok and match
        results.append({"seed": seed, "ratio": ratio, "skill": skill,
                        "expected": expected, "got": got, "match": match})
    return {"ok": ok, "tol": tol, "n_seeds": len(results), "seeds": results,
            "source": CANARY_V3_EXQ_1073["source"]}


def format_verdict(v: ActionSensitivityVerdict, label: str = "") -> str:
    """ASCII-only one-block summary. ALWAYS prints the denominators (remedy 3)."""
    head = "action-sensitivity gate" + (" [%s]" % label if label else "")
    lines = [
        "%s: %s" % (head, v.status.upper()),
        "  reason : %s" % v.reason,
        "  ratio  : %s (floor %.4f, form %s)" % (
            "n/a" if v.ratio is None
            else ("inf (perfect head)" if v.ratio == math.inf else "%.4f" % v.ratio),
            v.ratio_floor, v.extra.get("form", "n/a")),
        "  skill  : %s (floor %.4f)" % (
            "n/a" if v.skill is None else "%.4f" % v.skill, v.skill_floor),
        "  mse    : model=%s identity=%s counterfactual=%s" % (
            "n/a" if v.model_mse is None else "%.6g" % v.model_mse,
            "n/a" if v.identity_mse is None else "%.6g" % v.identity_mse,
            "n/a" if v.counterfactual_mse is None else "%.6g" % v.counterfactual_mse),
        "  DENOM  : n_rows=%d n_distinct_actions=%d" % (v.n_rows, v.n_distinct_actions),
    ]
    if v.status == "cannot_determine":
        lines.append("  NOTE   : cannot_determine is NOT a negative result -- "
                     "the battery could not test the question.")
    return "\n".join(lines)
