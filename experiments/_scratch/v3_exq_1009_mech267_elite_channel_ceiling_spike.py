"""V3-EXQ-1009 -- MECH-267 elite-channel ceiling: 2x2 proposer-only DIAGNOSTIC spike.

red-team (fable): see the verdict line at the end of this docstring.

WHY THIS RUNS. V3-EXQ-1005 (the properly-powered CONTENT/location test of
MECH-267) was REFUSED at /queue-experiment Step 4.5 on 2026-09-07, BLOCKING,
confirmed by two probes. At the production CEM settings the 1005 design targets,
the only channel a breadth-matched scored arm can use -- elite selection feeding
the CEM refit mean -- is capped at roughly +0.0001..+0.001 standardised units
against that design's own pre-registered 0.02 floor, while its positive control
moves the DV ~0.27 through a channel (per-mode ao_std rescaling acting on a
non-linear decoder) the scored arms shut by construction. Record:
REE_assembly/evidence/planning/exq1005_mech267_location_dv_redteam_blocking_20260907.md
(section 6 specifies this spike).

Two causes are entangled in that ceiling, and this spike separates them:

  (i) the CEM's support-preserving ao_std FLOOR (0.2, clamped after every refit),
      which re-samples every iteration about a refit mean at a fixed width; and
  (ii) E2's action-object head being a FROZEN RANDOM PROJECTION -- SD-080, already
      CONFIRMED by V3-EXQ-809 -- so its recomputed action objects barely depend on
      which candidate was chosen (~0.05 across-candidate std here), which bounds
      how far elite choice can move the refit mean.

Cause (ii) is PRODUCTION, not a bench artifact: no loss anywhere in ree_core
trains action_object_head (grep: it is referenced only at its construction site
and its forward call). So the "trained E2" cell of the record's specification
cannot be obtained from the 978/1006 warmup path -- that path would leave the head
at init and the cell would be VACUOUS. The head is instead grounded with the
objective V3-EXQ-817a built and validated for exactly this purpose (world-effect
regression onto a fixed near-isometric projection, no trainable readout), with one
substantiated departure recorded below.

DEPARTURE FROM 817a, AND WHY (measured, not assumed). 817a regresses o_t onto the
ABSOLUTE realised next world-state. On this synthetic bench that target is
dominated by the skip term (world_forward computes z_{t+1} = z_t + delta_w(z_t,a_t))
and is therefore nearly action-INDEPENDENT, so grounding onto it LOWERS the
property this spike needs: measured across-candidate action-object std fell
0.063 -> 0.029 (x0.5, seeds 0-2). Regressing onto the world-effect DELTA
(z_{t+1} - z_t = delta_w(z_t, a_t)) is the same objective applied to the quantity
that actually carries the action's contribution, and it RAISES it x2.6-3.9
(0.053-0.063 -> 0.167-0.222). The delta target is used; the absolute-target
measurement is recorded as a non-gating diagnostic because it is a real finding
about 817a's target choice on an untrained-world_forward bench.

DESIGN -- 2x2, proposer-only, no rollout, no env, no agent.

  ao_head  x  cem_floor
  ---------------------
  FROZEN    : action_object_head left at init (production; SD-080 defect)
  GROUNDED  : action_object_head regressed onto the world-effect delta (above)
  floor0.2  : support_preserving_ao_std_floor=0.2 (the live default, config.py:2495)
  floor0.0  : support_preserving_ao_std_floor=0.0, use_support_preserving_cem LEFT TRUE

The second axis is deliberately NOT the documented legacy opt-out triple. That triple
flips use_support_preserving_cem, which additionally gates stratified elite selection
(module.py:1312) AND the post-CEM synthetic-candidate injection (module.py:1460) --
neither of which reads the ao_std floor, and both of which act on the DV. Using it
would make every level difference unattributable among three simultaneous changes.
The refit clamp gates on `use_support_preserving_cem AND _std_floor > 0.0`
(module.py:2305-2309), so setting the FLOOR alone to 0.0 disables exactly the clamp
and leaves the other two behaviours identical across the axis.

DV -- and a SECOND SUBSTANTIATED DEPARTURE from the record's specification.
The record (section 6) specifies the DV as delta_dbar = dbar(ORACLE) - dbar(CTRL),
the archived probe-C statistic. That statistic was measured, by this spike's own
red-team pass, to be UNFIT to carry the ceiling gate, and the departure is recorded
here rather than made silently:

  dbar is a BETWEEN-MODE separation, and each mode's oracle direction is an
  independent random draw. An oracle displacement therefore enters dbar only to
  SECOND order (d^2/2R for a random direction, against 2d for an aligned one) and
  can as easily REDUCE the separation as raise it. Measured: delta_dbar is NEGATIVE
  on the production reference cell itself (-0.000485 at n=5, -0.000009 at n=1) and
  on GROUNDED/floor0.2 (-0.004784 at n=5). A quantity described as a CEILING or an
  upper BOUND cannot be negative. Gating on it would route a bench whose channel
  genuinely widened to "no clear" because the random directions happened not to
  separate the modes.

PRIMARY DV (gated):
    relocation_ratio = mean over modes of  ||mu_ORACLE - mu_CTRL|| / mean_spread(CTRL)
the per-mode centroid relocation the STRONGEST content-selective re-ranker achieves,
expressed in that cell's own sampling-noise units. This is what "can elite choice
relocate the proposal centroid above the sampling floor" actually asks. It is also
self-normalising against the spread collapse the floor0.0 level induces -- that level
shrinks numerator and denominator together -- which is what stops a collapsed CEM
from manufacturing an apparent clear.

delta_dbar is RETAINED as a recorded secondary on every cell, so this run stays
directly comparable with the archived probe-C numbers and with the 869/923/928
lineage. It is reported, never gated.

The centroid is computed INJECTION-FREE. _inject_support_preserving_candidates runs
after the CEM (module.py:2356) on the pool final_summary reads (:2325, :2494), and its
scaffolds are one one-hot step followed by exact zeros (:1223-1230), each dragging an
action dimension's mean by ~1/64 = 0.0156 -- above the displacement being measured. It
also fires conditionally on the pool's class count, which the ORACLE arm changes by
construction, so it can fire asymmetrically between the two arms of a pair. Injected
trajectories carry source="support_preserving_cem_injected" (:1489) and are filtered
out; the centroid is then recomputed exactly as _summarize_action_tensor does
(:748-750). Arm-symmetry of injection is additionally a readiness precondition.

TWO PRE-REGISTERED CRITERIA, BOTH ON relocation_ratio.

  C1: relocation_ratio >= CONTENT_FLOOR_ABS = 0.02 in at least one cell. The floor
      VALUE is inherited from the 1005 design and is NOT moved -- moving it to admit
      a measured number would be fitting the gate to the data. What changed is the
      statistic it is applied to, for the reason given under DV above.

  C2: that cell must also BEAT THE PRODUCTION REFERENCE cell (FROZEN/floor0.2 -- the
      1005 bench itself) on the same statistic. A cell that merely reaches the
      absolute floor without improving on production has not shown that either
      manipulation lifts the channel.

Both criteria read the SAME statistic on the SAME scale, so they cannot pull in
opposite directions -- which the first draft of this design did, and which its
red-team pass identified as close to fatal: C1 on delta_dbar was inflated by the
floor0.0 collapse while a raw-displacement C2 was reduced by it, leaving the cleared
branch nearly dead. The resolved tension is recorded per cell under
custom_information.joint_satisfiability_of_c1_and_c2 so a no-clear verdict stays
readable: "the channel is capped" and "the criteria were in tension" are different
findings and only the first is a fact about the substrate.

DECISION RULE (pre-registered, per the record's section 6):
  * a cell clearing C1 AND C2 -> the 1005 design IS runnable in that regime;
    author successor V3-EXQ-1005a on that bench.
  * a cell clearing C1 but failing C2 -> the absolute floor was cleared by the
    bench's baseline behaviour rather than by either manipulation; report, do not build.
  * NO cell clearing C1 -> MECH-267's content assertion is not measurable by
    proposal-output centroid at production CEM settings on any bench tested here;
    route to /governance to either narrow what_would_answer to the breadth channel
    or register a complicated (buildable) entry in substrate_queue.json (ao_std
    floor policy under mode conditioning, or E2 action-object action-dependence).
    This script does NOT register that build -- registering ahead of the spike's
    fact is the complicated-before-complex inversion the work-graph vocabulary
    warns against, and the record is explicit about it.

POSITIVE CONTROL ON THE INSTRUMENT ITSELF. The FROZEN/floor0.2 cell is the 1005
bench's configuration, and its delta_dbar must stay in the archived probe-C band
(|delta| <= 0.005) -- carried as a readiness precondition, so a bench that does not
behave like the one that produced the refusal cannot be read as a ceiling
measurement. Stated precisely, because it is a bound and not a point match: this
driver rebuilds the residue terrain on its own RNG stream, so it reproduces the
archived MAGNITUDE (order 1e-5..1e-3, sign varying) rather than the archived digits.
That is what the precondition asserts and all it asserts.

WHAT A NULL HERE WOULD AND WOULD NOT MEAN. It would mean: the elite-selection
channel cannot relocate the proposal centroid above the pre-registered floor on
EITHER an action-dependence-grounded bench OR a floor-free CEM -- so proposal-output
centroid is the wrong readout for MECH-267's content assertion. It would NOT mean
MECH-267 is false: the claim is about mode-conditioned proposal content, and this
spike measures only whether one instrument can see it. Nothing here is evidence
for or against the claim, which is why claim_ids is deliberately EMPTY and
experiment_purpose is diagnostic.

SUBSTRATE-PATH GATE (Step 2.5c). One open corrupting entry, mode-governance-engagement
(status implemented_pending_validation), co-lists ree_core/utils/config.py with this
driver. Its defect is the SalienceCoordinator affinity-input box clamp plus a
commitment term; this bench constructs no REEAgent, no SalienceCoordinator and no
regime-occupancy gate, and SalienceCoordinator appears in hippocampal/module.py only
inside a comment. Not reachable. Three further overlapping entries are degrading or
unset severity (SD-MECH303-THRESHOLD-SOURCING, mech203-valence-pool-admissibility,
mech142-no-valence-arousal-orthogonal-axis-substrate) and are noted, not blocking.

RELATED: V3-EXQ-1005 (refused, never queued), V3-EXQ-869/869a/923/928 (the breadth
lineage), V3-EXQ-809 (SD-080 frozen-projection CONFIRMED), V3-EXQ-817a (the grounding
objective this reuses).

red-team (fable): recorded in the queue entry note for V3-EXQ-1009.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import optim

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiment_protocol import emit_outcome
from experiments.pack_writer import write_flat_manifest
from experiments._lib.arm_fingerprint import arm_cell
from experiments._lib.readiness_anchor import assert_anchor_reachable

from ree_core.hippocampal.module import HippocampalModule
from ree_core.predictors.e2_fast import E2FastPredictor
from ree_core.residue.field import ResidueField
from ree_core.utils.config import E2Config, HippocampalConfig, ResidueConfig

EXPERIMENT_TYPE = "v3_exq_1009_mech267_elite_channel_ceiling_spike"
EXPERIMENT_PURPOSE = "diagnostic"
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"

# This spike tests an INSTRUMENT's reach, not a claim's truth. A clear and a null
# are both statements about what proposal-output centroid can see -- neither is
# evidence for or against MECH-267 -- so nothing is tagged.
CLAIM_IDS: List[str] = []

RELATED_EXQ = [
    "V3-EXQ-1005", "V3-EXQ-869", "V3-EXQ-869a", "V3-EXQ-923", "V3-EXQ-928",
    "V3-EXQ-809", "V3-EXQ-817a",
]

# --- bench dims: identical to the 1005 bench, so FROZEN/floor0.2 IS that bench ---
WORLD_DIM = 32
SELF_DIM = 16
ACTION_DIM = 4
ACTION_OBJECT_DIM = 16
NUM_CANDIDATES = 16
HORIZON = 4
NUM_CEM_ITERATIONS = 3          # the PRODUCTION setting the 1005 design targets

MODES: List[str] = [
    "internal_planning",
    "external_task",
    "internal_replay",
    "offline_consolidation",
]
MODE_PAIRS: List[Tuple[str, str]] = list(itertools.combinations(MODES, 2))  # 6 pairs

SEEDS: List[int] = [0, 1, 2, 3, 4]

# --- pre-registered thresholds (constants; never derived from this run) ---------
# C1 is inherited VERBATIM from the refused V3-EXQ-1005 design. Do not move it:
# lowering it to admit a measured value is fitting the gate to the data.
CONTENT_FLOOR_ABS = 0.02

# C2 attribution guard: a clearing cell's raw centroid displacement must exceed the
# FROZEN/floor0.2 reference cell's by at least this factor.
RAW_DISPLACEMENT_MIN_RATIO = 1.0

# Readiness precondition thresholds.
ARCHIVED_CEILING_ABS_CEIL = 0.005   # FROZEN/floor0.2 must reproduce ~+0.0001..+0.001
GROUNDING_ACS_MIN_RATIO = 2.0       # grounded cells must materially lift action-dependence
SPREAD_MIN = 1e-4                   # pooled-variance denominator must be well-conditioned

# --- grounding hyperparameters (the 817a objective, delta target) ---------------
GROUND_STEPS = 800
GROUND_LR = 1e-3
GROUND_BATCH = 256
GROUND_N_TRANSITIONS = 2048

AO_HEADS = ["FROZEN", "GROUNDED"]
CEM_FLOORS = ["floor0.2", "floor0.0"]
CELLS: List[Tuple[str, str]] = [(h, f) for h in AO_HEADS for f in CEM_FLOORS]
REFERENCE_CELL = ("FROZEN", "floor0.2")   # the 1005 bench

_MODE_OFFSET = {m: i * 7_919 for i, m in enumerate(MODES)}

# --- the SHIPPED readiness predicates ------------------------------------------
# Defined once and used in BOTH places: the setup-time reachability guard below and
# the live per-cell scoring in main(). A copy in either place would let the guard
# certify a predicate the run does not actually use.
def _pred_reproduces_ceiling(v: float) -> bool:
    return abs(float(v)) <= ARCHIVED_CEILING_ABS_CEIL


def _pred_acs_ratio(v: float) -> bool:
    return float(v) >= GROUNDING_ACS_MIN_RATIO


def _pred_spread(v: float) -> bool:
    return float(v) >= SPREAD_MIN


def _pred_elite_calls(v: float) -> bool:
    return float(v) >= float(NUM_CEM_ITERATIONS)


# Frozen positive-control literals, measured at Step 2.5a on this bench (and, for the
# first anchor, taken from the archived probe-C record itself). Each anchor's gate must
# be REACHABLE by its own control, or it is a guaranteed false negative that would
# mislabel an instrument-specification gap as a substrate verdict.
_ANCHOR_REFERENCES: Dict[str, Dict[str, Any]] = {
    "frozen_production_cell_reproduces_archived_ceiling": {
        # archived probe C (record section 3) + this bench's Step 2.5a pilot
        "cells": [0.00010, 0.00009, 0.00096, 0.00010, 0.00009],
        "score_fn": _pred_reproduces_ceiling,
        "source": "exq1005_mech267_location_dv_redteam_blocking_20260907.md sec 3 probe C + 1009 pilot seeds 0-1",
    },
    "grounded_cells_lift_action_object_action_dependence": {
        "cells": [2.64, 3.93, 3.64],
        "score_fn": _pred_acs_ratio,
        "source": "1009 Step 2.5a delta-target grounding probe, seeds 0-2",
    },
    "standardised_dv_well_conditioned": {
        "cells": [0.051120, 0.031483, 0.002425, 0.004649,
                  0.060289, 0.033861, 0.032472, 0.011112],
        "score_fn": _pred_spread,
        "source": "1009 Step 2.5a spread probe, all four cells, seeds 0-1",
    },
    "oracle_elite_override_fired_every_iteration": {
        "cells": [3, 3, 3, 3],
        "score_fn": _pred_elite_calls,
        "source": "1009 Step 2.5a Q3 probe, floor0.2 and floor0.0",
    },
}


def _assert_anchors_reachable() -> Dict[str, Any]:
    """Refuse the run at setup if any shipped readiness predicate cannot score its own
    positive control above its gate."""
    payloads: Dict[str, Any] = {}
    for name, ref in _ANCHOR_REFERENCES.items():
        payloads[name] = assert_anchor_reachable(
            anchor_name=name,
            reference_cells=ref["cells"],
            score_fn=ref["score_fn"],
            threshold=1.0,          # every reference cell must clear its own gate
            reference_source=ref["source"],
        )
    return payloads


def _cell_sampling_seed(seed: int, mode: str) -> int:
    """Per-(seed, mode) CEM sampling seed, SHARED across cells and across the
    CTRL/ORACLE contrast -- common random numbers.

    Retained at the 869/923/928/1005 value (7919) so per-mode sampling stays
    comparable with the lineage. The CTRL and ORACLE runs of a cell consume the
    identical standard-normal sequence (the oracle changes which indices are
    returned, never how many draws are taken), so their difference isolates elite
    choice rather than sampling noise.
    """
    return seed * 104_729 + _MODE_OFFSET[mode]


def _location_separation(
    mu_a: Sequence[float], sd_a: Sequence[float],
    mu_b: Sequence[float], sd_b: Sequence[float],
) -> Optional[float]:
    """Per-dimension-standardized RMS centroid separation. Copied verbatim from the
    refused V3-EXQ-1005 driver so the DV is bit-comparable with the archived
    probe-C numbers this spike must reproduce."""
    if not (len(mu_a) == len(sd_a) == len(mu_b) == len(sd_b)) or not mu_a:
        return None
    acc: List[float] = []
    for i in range(len(mu_a)):
        pooled_var = 0.5 * (float(sd_a[i]) ** 2 + float(sd_b[i]) ** 2)
        if not math.isfinite(pooled_var) or pooled_var <= 0.0:
            return None
        acc.append((float(mu_a[i]) - float(mu_b[i])) ** 2 / pooled_var)
    return math.sqrt(sum(acc) / len(acc))


def _raw_displacement(mu_a: Sequence[float], mu_b: Sequence[float]) -> Optional[float]:
    """UNSTANDARDISED centroid displacement -- the C2 attribution numerator."""
    if len(mu_a) != len(mu_b) or not mu_a:
        return None
    return math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(mu_a, mu_b)))


def _make_e2(seed: int) -> E2FastPredictor:
    torch.manual_seed(seed)
    return E2FastPredictor(
        E2Config(
            self_dim=SELF_DIM, world_dim=WORLD_DIM, action_dim=ACTION_DIM,
            action_object_dim=ACTION_OBJECT_DIM, hidden_dim=64,
        )
    )


def _isometric_projection(src_dim: int, dst_dim: int) -> torch.Tensor:
    """Fixed, seed-stable near-isometric projection (QR of a fixed random matrix),
    identical in construction to V3-EXQ-817a's _consequence_projection."""
    g = torch.Generator().manual_seed(4242)
    if src_dim < dst_dim:
        R = torch.randn(src_dim, max(dst_dim, src_dim), generator=g)[:, :dst_dim]
    else:
        R = torch.randn(src_dim, dst_dim, generator=g)
    Q, _ = torch.linalg.qr(R)
    return Q[:, :dst_dim]


def _ground_ao_head(
    e2: E2FastPredictor, seed: int, target: str = "delta",
    n_steps: int = GROUND_STEPS,
) -> Dict[str, Any]:
    """V3-EXQ-817a world-effect grounding, applied to this bench.

    target='delta'    -> regress o_t onto a projection of the world EFFECT
                         (z_{t+1} - z_t = delta_w(z_t, a_t)). The used target.
    target='absolute' -> 817a's own target (the absolute next state). Measured for
                         the record only; it is action-DILUTED here (see docstring).

    Trains action_object_head parameters ONLY, on .detach()ed inputs, with no
    trainable readout -- the objective 817a selected after rejecting three others.
    """
    g = torch.Generator().manual_seed(seed + 31_000)
    z_t = torch.randn(GROUND_N_TRANSITIONS, WORLD_DIM, generator=g)
    a_t = torch.randn(GROUND_N_TRANSITIONS, ACTION_DIM, generator=g)
    with torch.no_grad():
        z_next = e2.world_forward(z_t, a_t)
    raw = z_next if target == "absolute" else (z_next - z_t)

    std = raw.std(dim=0, keepdim=True)
    std = torch.where(std > 1e-8, std, torch.ones_like(std))
    standardized = (raw - raw.mean(dim=0, keepdim=True)) / std
    tgt = standardized @ _isometric_projection(WORLD_DIM, ACTION_OBJECT_DIM)
    t_std = tgt.std(dim=0, keepdim=True)
    t_std = torch.where(t_std > 1e-8, t_std, torch.ones_like(t_std))
    tgt = (tgt - tgt.mean(dim=0, keepdim=True)) / t_std

    opt = optim.Adam(list(e2.action_object_head.parameters()), lr=GROUND_LR)
    g2 = torch.Generator().manual_seed(seed + 41_000)
    first = last = float("nan")
    for it in range(n_steps):
        idx = torch.randint(0, GROUND_N_TRANSITIONS, (GROUND_BATCH,), generator=g2)
        o_t = e2.action_object(z_t[idx].detach(), a_t[idx].detach())
        loss = F.mse_loss(o_t, tgt[idx].detach())
        if it == 0:
            first = float(loss.item())
        opt.zero_grad()
        loss.backward()
        opt.step()
        last = float(loss.item())
    return {"grounding_target": target, "grounding_mse_first": first,
            "grounding_mse_final": last, "grounding_steps": int(n_steps)}


def _across_candidate_ao_std(e2: E2FastPredictor, seed: int) -> float:
    """How much E2's action objects vary ACROSS CANDIDATES at a FIXED z_world.

    This is the exact quantity that bounds the elite channel: candidates within one
    CEM iteration share z_world and differ only in their actions, and the refit reads
    E2's RE-COMPUTED action objects, so the elite mean cannot move further than this
    spread allows. It is the statistic the GROUNDED readiness precondition asserts --
    the same statistic the mechanism under test routes on, not a magnitude proxy.
    """
    g = torch.Generator().manual_seed(seed + 555)
    z_w = torch.randn(1, WORLD_DIM, generator=g).expand(NUM_CANDIDATES, -1)
    acts = torch.randn(NUM_CANDIDATES, ACTION_DIM, generator=g)
    with torch.no_grad():
        o = e2.action_object(z_w, acts)
    return float(o.std(dim=0).mean())


def _across_candidate_ao_std_scale_free(e2: E2FastPredictor, seed: int) -> float:
    """The same across-candidate spread, divided by the head's OVERALL output scale.

    Guards a specific way the GROUNDED cell could look ready without being ready:
    training can simply inflate the whole action-object output, which raises the raw
    across-candidate std without making the head any more discriminative between
    candidates. This ratio is invariant to that rescaling, so reporting both separates
    "the head now distinguishes candidates better" from "the head just got louder".
    Recorded on every cell; the raw ratio remains the gated one because it is the
    quantity in ao units that bounds how far the elite mean can move, but a large
    divergence between the two is a reading the manifest must not hide.
    """
    g = torch.Generator().manual_seed(seed + 555)
    z_w_fixed = torch.randn(1, WORLD_DIM, generator=g).expand(NUM_CANDIDATES, -1)
    acts = torch.randn(NUM_CANDIDATES, ACTION_DIM, generator=g)
    z_w_var = torch.randn(NUM_CANDIDATES, WORLD_DIM, generator=g)
    with torch.no_grad():
        o_fixed_state = e2.action_object(z_w_fixed, acts)
        o_var_state = e2.action_object(z_w_var, acts)
    across_cand = float(o_fixed_state.std(dim=0).mean())
    overall = float(o_var_state.std(dim=0).mean())
    return across_cand / overall if overall > 0 else 0.0


def _hippocampal_kwargs(cem_floor: str) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = dict(
        world_dim=WORLD_DIM, action_dim=ACTION_DIM,
        action_object_dim=ACTION_OBJECT_DIM, hidden_dim=32,
        horizon=HORIZON, num_candidates=NUM_CANDIDATES,
        num_cem_iterations=NUM_CEM_ITERATIONS,
        # Mode conditioning OFF in every cell: the oracle IS the content
        # manipulation here, so no mode-conditioned knob is engaged and the
        # cells differ only along the two declared axes.
        mode_conditioning_enabled=False,
        mode_value_weight={},
        mode_partitioned_cem=False,
    )
    if cem_floor == "floor0.0":
        # SINGLE-FACTOR, and this is load-bearing. The documented legacy opt-out
        # TRIPLE (use_support_preserving_cem=False + stratified_elites=False +
        # floor=0.0) would move THREE things at once: the refit clamp, stratified
        # elite selection (module.py:1312) and the post-CEM synthetic-candidate
        # injection (module.py:1460) are all gated on use_support_preserving_cem,
        # and the latter two act on the DV independently of any floor. Setting the
        # FLOOR alone to 0.0 while LEAVING THE FLAG TRUE disables exactly the clamp
        # and nothing else, because the refit gates on
        #     use_support_preserving_cem AND _std_floor > 0.0
        # (module.py:2305-2309). Stratified elites and injection then behave
        # identically in both levels of this axis, so a level difference is
        # attributable to the ao_std floor and to nothing else.
        kwargs.update(support_preserving_ao_std_floor=0.0)
    return kwargs


def _config_slice(ao_head: str, cem_floor: str) -> Dict[str, Any]:
    return {
        "bench": {
            "world_dim": WORLD_DIM, "self_dim": SELF_DIM, "action_dim": ACTION_DIM,
            "action_object_dim": ACTION_OBJECT_DIM, "hidden_dim_e2": 64,
            "num_candidates": NUM_CANDIDATES, "horizon": HORIZON,
            "num_cem_iterations": NUM_CEM_ITERATIONS,
        },
        "ao_head": ao_head,
        "cem_floor": cem_floor,
        "hippocampal_kwargs": _hippocampal_kwargs(cem_floor),
        "grounding": (
            {"target": "delta", "steps": GROUND_STEPS, "lr": GROUND_LR,
             "batch": GROUND_BATCH, "n_transitions": GROUND_N_TRANSITIONS}
            if ao_head == "GROUNDED" else None
        ),
    }


def _oracle_chooser(direction: torch.Tensor):
    """The STRONGEST content-selective re-ranker: at every CEM iteration, pick the
    elites whose decoded action mean projects furthest along `direction`.

    Structure taken from the archived red-team probe
    (ree-v3/experiments/_scratch/exq1005_probe_elite_channel.py, section B/C).
    """
    def choose(trajectories, scores_tensor, elite_indices):
        k = int(elite_indices.numel())
        proj = torch.stack([
            (t.actions.mean(dim=(0, 1)) * direction).sum() for t in trajectories
        ])
        return torch.argsort(proj, descending=True)[:k]
    return choose


def _propose(
    e2: E2FastPredictor, cem_floor: str, seed: int, mode: str,
    oracle: Optional[Any],
) -> Dict[str, Any]:
    """One proposer evaluation. Returns the final-iteration decoded-action stats."""
    torch.manual_seed(seed + 77_000)
    residue = ResidueField(
        ResidueConfig(world_dim=WORLD_DIM, hidden_dim=32, num_basis_functions=8)
    )
    hip = HippocampalModule(HippocampalConfig(**_hippocampal_kwargs(cem_floor)),
                            e2, residue)

    fired: List[int] = []
    original = hip._support_preserving_elite_indices

    def patched(trajectories, scores_tensor, elite_indices):
        idx, diag = original(trajectories=trajectories, scores_tensor=scores_tensor,
                             elite_indices=elite_indices)
        fired.append(1)
        if oracle is not None:
            return oracle(trajectories, scores_tensor, elite_indices), diag
        return idx, diag

    hip._support_preserving_elite_indices = patched

    torch.manual_seed(seed + 900_000)
    z_world = torch.randn(1, WORLD_DIM)
    z_self = torch.randn(1, SELF_DIM)
    torch.manual_seed(_cell_sampling_seed(seed, mode))
    trajectories = hip.propose_trajectories(z_world, z_self,
                                            num_candidates=NUM_CANDIDATES,
                                            operating_mode={mode: 1.0})

    diagnostics = hip.get_last_propose_diagnostics()
    stats = diagnostics["action_object_decoder_raw_output_stats"]
    itd = diagnostics.get("cem_iteration_diagnostics") or []

    # ---- injection-free centroid, and why the substrate's own stats are not used --
    # _inject_support_preserving_candidates runs AFTER the CEM (module.py:2356), on
    # the pool that `all_trajectories` (:2325) already points at, and final_summary
    # (:2494) -- the source of action_object_decoder_raw_output_stats (:2552) -- is
    # computed from it. Its synthetic scaffolds are one one-hot step followed by
    # exact zeros (:1223-1230), so each one drags a whole action dimension's mean by
    # ~1/(candidates*horizon) = 1/64 = 0.0156 -- an order of magnitude ABOVE the
    # centroid displacement this spike measures. It also fires conditionally on the
    # pool's first-action class count (:1461), which the ORACLE arm changes by
    # construction, so it can fire ASYMMETRICALLY between the two arms of a pair and
    # manufacture the very displacement being read.
    # The trajectories it splices in are tagged source="support_preserving_cem_injected"
    # (:1489), so they are filtered here and the centroid recomputed exactly as
    # _summarize_action_tensor does (:748-750: reshape to [-1, action_dim], mean and
    # population std over dim 0).
    injected_flags = [
        bool((getattr(t, "metadata", None) or {}).get("source")
             == "support_preserving_cem_injected")
        for t in trajectories
    ]
    kept = [t for t, inj in zip(trajectories, injected_flags) if not inj]
    n_injected = sum(injected_flags)
    if kept:
        acts = torch.stack([t.actions for t in kept])          # [cand, batch, H, a]
        flat = acts.detach().reshape(-1, acts.shape[-1])
        mean_by_dim = flat.mean(dim=0).tolist()
        std_by_dim = flat.std(dim=0, unbiased=False).tolist()
    else:
        mean_by_dim = list(stats["mean_by_action_dim"])
        std_by_dim = list(stats["std_by_action_dim"])

    return {
        "mean_by_action_dim": mean_by_dim,
        "std_by_action_dim": std_by_dim,
        "mean_by_action_dim_substrate_incl_injected": list(stats["mean_by_action_dim"]),
        "std_by_action_dim_substrate_incl_injected": list(stats["std_by_action_dim"]),
        "n_injected_candidates": int(n_injected),
        "n_candidates_scored": len(kept),
        "elite_fn_calls": len(fired),
        "ao_std_by_iteration": [
            {"iteration": int(d.get("iteration", i)),
             "ao_std_min": float(d.get("ao_std_min", float("nan"))),
             "ao_std_max": float(d.get("ao_std_max", float("nan")))}
            for i, d in enumerate(itd)
        ],
    }


def _run_cell(
    seed: int, ao_head: str, cem_floor: str, directions: List[torch.Tensor],
    ground_steps: int, cell_index: int, n_cells: int,
) -> Dict[str, Any]:
    """One (seed x cell) unit: CTRL and ORACLE proposer runs over all 4 modes."""
    label = f"{ao_head}/{cem_floor}"
    print(f"Seed {seed} Condition {label}", flush=True)

    total_units = len(MODES) * 2   # each mode contributes a CTRL and an ORACLE run

    with arm_cell(seed, config_slice=_config_slice(ao_head, cem_floor),
                  script_path=Path(__file__)) as cell:
        e2 = _make_e2(seed)
        acs_before = _across_candidate_ao_std(e2, seed)
        acs_sf_before = _across_candidate_ao_std_scale_free(e2, seed)
        grounding: Optional[Dict[str, Any]] = None
        grounding_absolute_probe: Optional[Dict[str, Any]] = None
        if ao_head == "GROUNDED":
            # Recorded, non-gating: 817a's own ABSOLUTE target on this bench, for the
            # docstring's departure claim. Trained on a throwaway copy so the used
            # head is unaffected.
            e2_abs = _make_e2(seed)
            info_abs = _ground_ao_head(e2_abs, seed, target="absolute",
                                       n_steps=ground_steps)
            grounding_absolute_probe = {
                **info_abs,
                "across_candidate_ao_std_after": _across_candidate_ao_std(e2_abs, seed),
            }
            del e2_abs
            grounding = _ground_ao_head(e2, seed, target="delta", n_steps=ground_steps)
        acs_after = _across_candidate_ao_std(e2, seed)
        acs_sf_after = _across_candidate_ao_std_scale_free(e2, seed)

        ctrl: Dict[str, Dict[str, Any]] = {}
        orc: Dict[str, Dict[str, Any]] = {}
        elite_calls: List[int] = []
        unit = 0
        for mode_index, mode in enumerate(MODES):
            ctrl[mode] = _propose(e2, cem_floor, seed, mode, oracle=None)
            unit += 1
            print(f"  [train] {label} seed={seed} ep {unit}/{total_units} "
                  f"mode={mode} arm=CTRL", flush=True)
            orc[mode] = _propose(e2, cem_floor, seed, mode,
                                 oracle=_oracle_chooser(directions[mode_index]))
            unit += 1
            print(f"  [train] {label} seed={seed} ep {unit}/{total_units} "
                  f"mode={mode} arm=ORACLE", flush=True)
            elite_calls.append(ctrl[mode]["elite_fn_calls"])
            elite_calls.append(orc[mode]["elite_fn_calls"])

        def dbar(cells: Dict[str, Dict[str, Any]]) -> Optional[float]:
            vals = [
                _location_separation(cells[a]["mean_by_action_dim"],
                                     cells[a]["std_by_action_dim"],
                                     cells[b]["mean_by_action_dim"],
                                     cells[b]["std_by_action_dim"])
                for a, b in MODE_PAIRS
            ]
            return statistics.fmean(vals) if all(v is not None for v in vals) else None

        dbar_ctrl = dbar(ctrl)
        dbar_oracle = dbar(orc)
        delta = (dbar_oracle - dbar_ctrl
                 if (dbar_ctrl is not None and dbar_oracle is not None) else None)

        raw_disps = [
            _raw_displacement(ctrl[m]["mean_by_action_dim"], orc[m]["mean_by_action_dim"])
            for m in MODES
        ]
        raw_disp = (statistics.fmean([r for r in raw_disps if r is not None])
                    if all(r is not None for r in raw_disps) else None)

        # ---- PRIMARY DV: per-mode centroid relocation in CTRL sampling-noise units --
        # This replaces delta_dbar as the gated statistic. delta_dbar is a difference
        # of BETWEEN-MODE separations, and each mode's oracle direction is an
        # independent random draw, so an oracle displacement enters it only to second
        # order and can lower it as easily as raise it -- measured negative on the
        # production reference cell itself, and a quantity described as a CEILING
        # cannot be negative. relocation_ratio measures the thing the ceiling question
        # actually asks: how far the STRONGEST content-selective re-ranker moves a
        # mode's own centroid, expressed in that cell's own sampling spread. It is
        # also self-normalising against the spread collapse the ao_std_floor=0.0 level
        # induces -- that level shrinks numerator and denominator together -- which is
        # what stops a collapsed CEM from manufacturing an apparent clear.
        # delta_dbar is retained as a recorded secondary for comparability with the
        # archived probe-C record.
        per_mode_reloc: List[Optional[float]] = []
        for m in MODES:
            disp = _raw_displacement(ctrl[m]["mean_by_action_dim"],
                                     orc[m]["mean_by_action_dim"])
            sd = ctrl[m]["std_by_action_dim"]
            denom = statistics.fmean(float(s) for s in sd) if sd else 0.0
            per_mode_reloc.append(disp / denom if (disp is not None and denom > 0) else None)
        relocation_ratio = (
            statistics.fmean([r for r in per_mode_reloc if r is not None])
            if all(r is not None for r in per_mode_reloc) else None
        )

        all_spreads = [s for m in MODES
                       for s in (ctrl[m]["std_by_action_dim"] + orc[m]["std_by_action_dim"])]
        min_spread = min(float(s) for s in all_spreads) if all_spreads else None
        mean_spread = statistics.fmean(float(s) for s in all_spreads) if all_spreads else None

        row: Dict[str, Any] = {
            "arm_id": label,
            "ao_head": ao_head,
            "cem_floor": cem_floor,
            "seed": seed,
            "relocation_ratio": relocation_ratio,
            "relocation_ratio_per_mode": per_mode_reloc,
            "dbar_ctrl": dbar_ctrl,
            "dbar_oracle": dbar_oracle,
            "delta_dbar": delta,
            "raw_centroid_displacement": raw_disp,
            "raw_centroid_displacement_per_mode": raw_disps,
            "n_injected_ctrl": [ctrl[m]["n_injected_candidates"] for m in MODES],
            "n_injected_oracle": [orc[m]["n_injected_candidates"] for m in MODES],
            "injection_arm_symmetric": all(
                ctrl[m]["n_injected_candidates"] == orc[m]["n_injected_candidates"]
                for m in MODES
            ),
            "across_candidate_ao_std_before": acs_before,
            "across_candidate_ao_std_after": acs_after,
            "across_candidate_ao_std_ratio": (
                acs_after / acs_before if acs_before > 0 else None
            ),
            "across_candidate_ao_std_scale_free_before": acs_sf_before,
            "across_candidate_ao_std_scale_free_after": acs_sf_after,
            "across_candidate_ao_std_scale_free_ratio": (
                acs_sf_after / acs_sf_before if acs_sf_before > 0 else None
            ),
            "min_per_dim_spread": min_spread,
            "mean_per_dim_spread": mean_spread,
            "elite_fn_calls_min": min(elite_calls) if elite_calls else 0,
            "grounding": grounding,
            "grounding_absolute_target_probe": grounding_absolute_probe,
            "ao_std_by_iteration_ctrl_first_mode": ctrl[MODES[0]]["ao_std_by_iteration"],
            "per_mode_dbar_pairs_ctrl": {
                f"{a}|{b}": _location_separation(
                    ctrl[a]["mean_by_action_dim"], ctrl[a]["std_by_action_dim"],
                    ctrl[b]["mean_by_action_dim"], ctrl[b]["std_by_action_dim"])
                for a, b in MODE_PAIRS
            },
            "per_mode_dbar_pairs_oracle": {
                f"{a}|{b}": _location_separation(
                    orc[a]["mean_by_action_dim"], orc[a]["std_by_action_dim"],
                    orc[b]["mean_by_action_dim"], orc[b]["std_by_action_dim"])
                for a, b in MODE_PAIRS
            },
        }
        cell.stamp(row)

    clears_c1 = relocation_ratio is not None and relocation_ratio >= CONTENT_FLOOR_ABS
    print(f"verdict: {'PASS' if clears_c1 else 'FAIL'}", flush=True)
    return row


def _worst_cell(rows: List[Dict[str, Any]], key: str, mode: str = "min"
                ) -> Tuple[Optional[float], Optional[str]]:
    """Worst-cell extremum plus the offending cell id -- never a mean, because the
    preconditions below are all quantifier claims over cells."""
    vals = [(r.get(key), f"{r['arm_id']}@seed{r['seed']}") for r in rows
            if r.get(key) is not None]
    if not vals:
        return None, None
    return (min(vals, key=lambda v: v[0]) if mode == "min"
            else max(vals, key=lambda v: v[0]))


def main(dry_run: bool = False) -> Dict[str, Any]:
    started_at = time.perf_counter()
    # Setup-time refusal: a gate its own positive control cannot clear is a
    # guaranteed false negative. Runs BEFORE any compute.
    anchor_reachability = _assert_anchors_reachable()
    seeds = SEEDS[:1] if dry_run else SEEDS
    ground_steps = 40 if dry_run else GROUND_STEPS

    # Per-mode oracle directions, drawn from a LOCAL stream so the global RNG is
    # untouched; shared across every cell so all cells face the identical oracle.
    gen = torch.Generator().manual_seed(4242)
    directions = [torch.randn(ACTION_DIM, generator=gen) for _ in MODES]
    directions = [d / d.norm() for d in directions]

    rows: List[Dict[str, Any]] = []
    n_cells = len(seeds) * len(CELLS)
    idx = 0
    for seed in seeds:
        for ao_head, cem_floor in CELLS:
            rows.append(_run_cell(seed, ao_head, cem_floor, directions,
                                  ground_steps, idx, n_cells))
            idx += 1

    # ---- per-cell aggregation across seeds ------------------------------------
    by_cell: Dict[str, Dict[str, Any]] = {}
    for ao_head, cem_floor in CELLS:
        label = f"{ao_head}/{cem_floor}"
        sel = [r for r in rows if r["arm_id"] == label]
        deltas = [r["delta_dbar"] for r in sel if r["delta_dbar"] is not None]
        relocs = [r["relocation_ratio"] for r in sel if r["relocation_ratio"] is not None]
        disps = [r["raw_centroid_displacement"] for r in sel
                 if r["raw_centroid_displacement"] is not None]
        by_cell[label] = {
            "arm_id": label, "ao_head": ao_head, "cem_floor": cem_floor,
            "n_seeds": len(sel),
            "relocation_ratio_per_seed": [r["relocation_ratio"] for r in sel],
            "relocation_ratio_mean": statistics.fmean(relocs) if relocs else None,
            "relocation_ratio_max": max(relocs) if relocs else None,
            "injection_arm_symmetric_all_seeds": all(
                bool(r["injection_arm_symmetric"]) for r in sel),
            "n_injected_total": sum(sum(r["n_injected_ctrl"]) + sum(r["n_injected_oracle"])
                                    for r in sel),
            "delta_dbar_per_seed": [r["delta_dbar"] for r in sel],
            "delta_dbar_mean": statistics.fmean(deltas) if deltas else None,
            "delta_dbar_max": max(deltas) if deltas else None,
            "raw_centroid_displacement_mean": statistics.fmean(disps) if disps else None,
            "dbar_ctrl_mean": statistics.fmean(
                [r["dbar_ctrl"] for r in sel if r["dbar_ctrl"] is not None]) or None,
            "mean_per_dim_spread_mean": statistics.fmean(
                [r["mean_per_dim_spread"] for r in sel
                 if r["mean_per_dim_spread"] is not None]) or None,
            "across_candidate_ao_std_after_mean": statistics.fmean(
                [r["across_candidate_ao_std_after"] for r in sel]),
        }

    ref_label = f"{REFERENCE_CELL[0]}/{REFERENCE_CELL[1]}"
    ref = by_cell[ref_label]
    ref_disp = ref["raw_centroid_displacement_mean"]
    ref_reloc = ref["relocation_ratio_mean"]

    # ---- readiness preconditions ----------------------------------------------
    ref_delta_mean = ref["delta_dbar_mean"]
    reproduces = ref_delta_mean is not None and _pred_reproduces_ceiling(ref_delta_mean)

    grounded_rows = [r for r in rows if r["ao_head"] == "GROUNDED"]
    worst_acs_ratio, worst_acs_cell = _worst_cell(
        grounded_rows, "across_candidate_ao_std_ratio", "min")
    worst_spread, worst_spread_cell = _worst_cell(rows, "min_per_dim_spread", "min")
    worst_elite_calls, worst_elite_cell = _worst_cell(rows, "elite_fn_calls_min", "min")

    asym_rows = [r for r in rows if not r["injection_arm_symmetric"]]
    n_asym = len(asym_rows)
    asym_cell = (f"{asym_rows[0]['arm_id']}@seed{asym_rows[0]['seed']}"
                 if asym_rows else None)

    preconditions: List[Dict[str, Any]] = [
        {
            "name": "frozen_production_cell_reproduces_archived_ceiling",
            "description": (
                "FROZEN/floor0.2 IS the V3-EXQ-1005 bench, so its oracle ceiling must "
                "reproduce the archived probe-C value (+0.0001..+0.001). If it does not, "
                "this spike is not measuring the instrument that produced the refusal."),
            "measured": abs(ref_delta_mean) if ref_delta_mean is not None else None,
            "threshold": ARCHIVED_CEILING_ABS_CEIL,
            "direction": "upper",
            "control": "the refused 1005 bench itself, re-instantiated unchanged",
            "met": bool(reproduces),
        },
        {
            "name": "grounded_cells_lift_action_object_action_dependence",
            "description": (
                "The GROUNDED cells exist to supply action-dependence in E2's recomputed "
                "action objects. Worst cell reported (this is a quantifier claim over "
                "cells, not a central tendency). Below floor means those cells are "
                "vacuous and cannot support a no-clear verdict."),
            "measured": worst_acs_ratio,
            "threshold": GROUNDING_ACS_MIN_RATIO,
            "direction": "lower",
            "control": "same statistic the elite channel is bounded by, measured pre/post grounding",
            "offending_cell": worst_acs_cell,
        "met": bool(worst_acs_ratio is not None and _pred_acs_ratio(worst_acs_ratio)),
        },
        {
            "name": "standardised_dv_well_conditioned",
            "description": (
                "The DV divides by the pooled per-dim sample spread. Removing the ao_std "
                "floor collapses that spread, so the smallest per-dim spread over every "
                "cell must stay above a floor for the standardised statistic to be "
                "numerically trustworthy. Worst cell reported."),
            "measured": worst_spread,
            "threshold": SPREAD_MIN,
            "direction": "lower",
            "control": "min over all cells, both arms, all modes",
            "offending_cell": worst_spread_cell,
            "met": bool(worst_spread is not None and _pred_spread(worst_spread)),
        },
        {
            "name": "oracle_elite_override_fired_every_iteration",
            "description": (
                "The oracle replaces elite choice at the _support_preserving_elite_indices "
                "call site. That site must be reached once per CEM iteration in every cell, "
                "or the oracle is silently inert and the ceiling is not an oracle ceiling."),
            "measured": worst_elite_calls,
            "threshold": float(NUM_CEM_ITERATIONS),
            "direction": "lower",
            "control": "min call count over all cells and arms",
            "offending_cell": worst_elite_cell,
        "met": bool(worst_elite_calls is not None and _pred_elite_calls(worst_elite_calls)),
        },
        {
            "name": "post_cem_injection_arm_symmetric",
            "description": (
                "The post-CEM synthetic-candidate injection (module.py:2356) fires "
                "conditionally on the final pool's first-action class count, which the "
                "ORACLE arm changes by construction. Its scaffolds are excluded from the "
                "DV by source tag, but an arm-ASYMMETRIC firing also perturbs the pool the "
                "kept candidates were selected from, so symmetry is required for the "
                "CTRL/ORACLE contrast to be clean. Worst cell reported."),
            "measured": float(n_asym),
            "threshold": 0.0,
            "direction": "upper",
            "control": "count of (seed, cell, mode) triples where CTRL and ORACLE injected different counts",
            "offending_cell": asym_cell,
            "met": bool(n_asym == 0),
        },
    ]
    all_preconditions_met = all(bool(p["met"]) for p in preconditions)

    # ---- criteria ---------------------------------------------------------------
    c1_cells = [lab for lab, c in by_cell.items()
                if c["relocation_ratio_mean"] is not None
                and c["relocation_ratio_mean"] >= CONTENT_FLOOR_ABS]
    c1_pass = len(c1_cells) > 0

    def clears_c2(label: str) -> bool:
        """The manipulation must BEAT PRODUCTION on the same statistic C1 gates.

        Measured on the same scale as C1 (no opposing-direction pathology): a cell
        that merely reproduces the production bench's own relocation has not shown
        the manipulation lifts the channel, whatever absolute number it reaches.
        The reference cell IS the production condition, so this is a within-run
        contrast and needs no external calibration."""
        if label == ref_label:
            return False
        r = by_cell[label]["relocation_ratio_mean"]
        return (r is not None and ref_reloc is not None
                and r >= RAW_DISPLACEMENT_MIN_RATIO * ref_reloc)

    c2_cells = [lab for lab in c1_cells if clears_c2(lab)]
    c2_pass = len(c2_cells) > 0
    c1_only_cells = [lab for lab in c1_cells if lab not in c2_cells]

    deltas_all = [c["relocation_ratio_mean"] for c in by_cell.values()
                  if c["relocation_ratio_mean"] is not None]
    c1_non_degenerate = (
        len(deltas_all) == len(CELLS)
        and (max(deltas_all) - min(deltas_all)) > 1e-9
    )
    # C2 can only discriminate if there is a C1-clearing cell to test it on.
    c2_non_degenerate = bool(c1_pass)

    criteria = [
        {"name": "C1_any_cell_clears_content_floor", "load_bearing": True,
         "passed": bool(c1_pass), "threshold": CONTENT_FLOOR_ABS,
         "clearing_cells": c1_cells, "statistic": "relocation_ratio",
         "note": ("floor value 0.02 inherited from the refused V3-EXQ-1005 design; applied "
                  "to relocation_ratio, which is displacement in sampling-noise units -- "
                  "see the docstring for why delta_dbar could not carry it")},
        {"name": "C2_manipulation_beats_production_reference", "load_bearing": True,
         "passed": bool(c2_pass), "clearing_cells": c2_cells,
         "reference_cell": ref_label, "reference_relocation_ratio": ref_reloc,
         "statistic": "relocation_ratio",
         "note": ("same statistic and scale as C1, so the two criteria cannot pull in "
                  "opposite directions; a cell must both clear the absolute floor and "
                  "improve on the production bench")},
    ]
    combination_rule = (
        "A cell clears the elite-channel ceiling only if it satisfies C1 AND C2, both read "
        "off relocation_ratio. C1 is the absolute floor inherited from the 1005 design; C2 "
        "requires the manipulation to beat the production reference cell on the same "
        "statistic. Both are evaluated on an injection-free centroid."
    )

    # ---- verdict ------------------------------------------------------------------
    if not all_preconditions_met:
        label = "substrate_not_ready_requeue"
        outcome = "FAIL"
        summary = (
            "Readiness preconditions not met: "
            + ", ".join(p["name"] for p in preconditions if not p["met"])
            + ". The 2x2 cannot be read as a ceiling measurement in this state."
        )
        routing = "re-queue at an adequate bench; do NOT read the cells as a ceiling verdict"
    elif c2_pass:
        label = "elite_channel_ceiling_cleared::" + "+".join(sorted(c2_cells))
        outcome = "PASS"
        summary = (
            f"Cell(s) {sorted(c2_cells)} clear the pre-registered {CONTENT_FLOOR_ABS} "
            f"oracle-elite relocation floor AND beat the {ref_label} production reference "
            f"({ref_reloc}) on the same statistic, on an injection-free centroid. The "
            f"V3-EXQ-1005 design is runnable on that bench."
        )
        routing = ("author successor V3-EXQ-1005a on the clearing bench "
                   "(see the record's section 6 decision rule)")
    elif c1_pass:
        label = "floor_cleared_but_no_lift_over_production"
        outcome = "PASS"
        summary = (
            f"Cell(s) {sorted(c1_only_cells)} clear the absolute {CONTENT_FLOOR_ABS} "
            f"relocation floor but do NOT beat the {ref_label} production reference "
            f"({ref_reloc}) on the same statistic -- so the floor is cleared by the bench's "
            f"baseline behaviour rather than by either manipulation, and neither grounding "
            f"nor removing the ao_std floor lifted the elite channel."
        )
        routing = ("report to /governance; do NOT author V3-EXQ-1005a on these cells -- the "
                   "manipulation is not what cleared the floor, so the 1005 design gains "
                   "nothing from that bench")
    else:
        label = "elite_channel_ceiling_confirmed_all_benches"
        outcome = "PASS"
        summary = (
            f"No cell clears the pre-registered {CONTENT_FLOOR_ABS} oracle-elite relocation "
            f"floor (production reference {ref_label} = {ref_reloc}). "
            f"Neither grounding E2's action-object head into genuine action-dependence nor "
            f"removing the support-preserving ao_std floor lets the STRONGEST possible "
            f"content-selective re-ranker relocate the proposal centroid above the floor. "
            f"MECH-267's content assertion is not measurable by proposal-output centroid at "
            f"production CEM settings on any bench tested here."
        )
        routing = (
            "route to /governance: either narrow MECH-267's what_would_answer to the breadth "
            "channel, or register a complicated (buildable) substrate_queue entry (ao_std floor "
            "policy under mode conditioning, or E2 action-object action-dependence). This run "
            "deliberately registers NEITHER -- that is governance's call, not the spike's."
        )

    manifest: Dict[str, Any] = {
        "run_id": f"{EXPERIMENT_TYPE}_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}_v3",
        "experiment_type": EXPERIMENT_TYPE,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "architecture_epoch": ARCHITECTURE_EPOCH,
        "claim_ids": CLAIM_IDS,
        "related_exq": RELATED_EXQ,
        "outcome": outcome,
        "timestamp_utc": time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()),
        "dry_run": bool(dry_run),
        "arm_results": rows,
        "cell_summary": by_cell,
        "criteria": criteria,
        "combination_rule": combination_rule,
        "interpretation": {
            "label": label,
            "summary": summary,
            "routing": routing,
            "preconditions": preconditions,
            "anchor_reachability": anchor_reachability,
            "criteria_non_degenerate": {
                "C1_any_cell_clears_content_floor": bool(c1_non_degenerate),
                "C2_clearing_cell_relocates_centroid_not_ruler": bool(c2_non_degenerate),
            },
            "criteria_non_degenerate_note": (
                "C2 is marked non-degenerate only when at least one cell cleared C1; with no "
                "C1 clear there is nothing for the attribution guard to discriminate, and the "
                "verdict routes on C1 alone."
            ),
            "what_a_null_does_not_mean": (
                "A no-clear result is a statement about what the proposal-output centroid "
                "readout can see, NOT evidence against MECH-267. claim_ids is empty for this "
                "reason."
            ),
        },
        "non_degenerate": bool(all_preconditions_met and c1_non_degenerate),
        "pre_registered_thresholds": {
            "CONTENT_FLOOR_ABS": CONTENT_FLOOR_ABS,
            "RAW_DISPLACEMENT_MIN_RATIO": RAW_DISPLACEMENT_MIN_RATIO,
            "ARCHIVED_CEILING_ABS_CEIL": ARCHIVED_CEILING_ABS_CEIL,
            "GROUNDING_ACS_MIN_RATIO": GROUNDING_ACS_MIN_RATIO,
            "SPREAD_MIN": SPREAD_MIN,
        },
        "custom_information": {
            "dv_headroom": {
                "name": "oracle_elite_centroid_ceiling",
                "description": (
                    "The oracle elite chooser is the STRONGEST content-selective re-ranker "
                    "available at the elite-selection call site, so delta_dbar is an upper "
                    "BOUND on what any real mode-conditioned content arm could reach on that "
                    "bench. Headroom is therefore reported as the achievable-vs-floor ratio "
                    "per cell; a cell whose bound sits below the floor cannot host the 1005 "
                    "design regardless of the manipulation chosen."),
                "floor": CONTENT_FLOOR_ABS,
                "achievable_by_cell": {
                    lab: c["relocation_ratio_mean"] for lab, c in by_cell.items()},
                "headroom_ratio_by_cell": {
                    lab: (c["relocation_ratio_mean"] / CONTENT_FLOOR_ABS
                          if c["relocation_ratio_mean"] is not None else None)
                    for lab, c in by_cell.items()},
                "secondary_delta_dbar_by_cell": {
                    lab: c["delta_dbar_mean"] for lab, c in by_cell.items()},
                "margin": 1.0,
            },
            "joint_satisfiability_of_c1_and_c2": {
                "description": (
                    "HISTORICAL NOTE, kept because it drove the design. In the first draft C1 read "
                    "delta_dbar (which the ao_std_floor=0.0 level inflates by shrinking the "
                    "denominator) while C2 read raw displacement (which the same level reduces), so "
                    "the two criteria pulled in OPPOSITE directions along that axis and the "
                    "cleared branch was close to dead. Both now read relocation_ratio, which is "
                    "self-normalising against the collapse -- that level shrinks numerator and "
                    "denominator together -- so the pair is satisfiable exactly when a bench "
                    "genuinely relocates the centroid relative to its own sampling noise, which is "
                    "the spike's question. Recorded per cell so a no-clear verdict stays auditable: "
                    "'the channel is capped' and 'the criteria were in tension' are different "
                    "findings, and only the first is a fact about the substrate."),
                "per_cell": {
                    lab: {
                        "relocation_ratio_mean": c["relocation_ratio_mean"],
                        "delta_dbar_mean": c["delta_dbar_mean"],
                        "raw_centroid_displacement_mean": c["raw_centroid_displacement_mean"],
                        "mean_per_dim_spread_mean": c["mean_per_dim_spread_mean"],
                        "clears_c1": bool(c["relocation_ratio_mean"] is not None
                                          and c["relocation_ratio_mean"] >= CONTENT_FLOOR_ABS),
                        "clears_c2_beats_production": bool(
                            lab != ref_label and c["relocation_ratio_mean"] is not None
                            and ref_reloc is not None
                            and c["relocation_ratio_mean"]
                            >= RAW_DISPLACEMENT_MIN_RATIO * ref_reloc),
                    }
                    for lab, c in by_cell.items()
                },
                "any_cell_would_clear_c2_independently": bool(any(
                    c["relocation_ratio_mean"] is not None and ref_reloc is not None
                    and c["relocation_ratio_mean"] >= RAW_DISPLACEMENT_MIN_RATIO * ref_reloc
                    for lab, c in by_cell.items() if lab != ref_label)),
                "note": (
                    "any_cell_would_clear_c2_independently=false alongside a no-clear verdict "
                    "means C2 was never satisfiable on this bench family and the verdict rests on "
                    "C1 alone; true means C2 was live and genuinely discriminated."),
            },
            "standardisation_diagnostic": {
                "description": (
                    "Recorded so a C1 clear can always be attributed. delta_dbar is "
                    "spread-standardised; these are the numerator and denominator separately."),
                "raw_centroid_displacement_by_cell": {
                    lab: c["raw_centroid_displacement_mean"] for lab, c in by_cell.items()},
                "mean_per_dim_spread_by_cell": {
                    lab: c["mean_per_dim_spread_mean"] for lab, c in by_cell.items()},
                "dbar_ctrl_by_cell": {
                    lab: c["dbar_ctrl_mean"] for lab, c in by_cell.items()},
            },
            "grounding_scale_free_check": {
                "description": (
                    "The gated readiness ratio is the RAW across-candidate action-object std, "
                    "which training can raise by simply inflating the head's whole output. The "
                    "scale-free ratio divides by the head's overall output scale and is invariant "
                    "to that. Both are reported per cell so a GROUNDED cell that only got louder "
                    "is distinguishable from one that genuinely discriminates candidates better. "
                    "Note the elite channel is bounded in ao UNITS, which is why the raw ratio is "
                    "the gated one -- but a raw lift with a flat or falling scale-free ratio must "
                    "be read as rescaling, and any clear on such a cell treated accordingly."),
                "raw_ratio_by_cell": {
                    lab: statistics.fmean(
                        [r["across_candidate_ao_std_ratio"] for r in rows
                         if r["arm_id"] == lab
                         and r["across_candidate_ao_std_ratio"] is not None] or [float("nan")])
                    for lab in by_cell},
                "scale_free_ratio_by_cell": {
                    lab: statistics.fmean(
                        [r["across_candidate_ao_std_scale_free_ratio"] for r in rows
                         if r["arm_id"] == lab
                         and r["across_candidate_ao_std_scale_free_ratio"] is not None]
                        or [float("nan")])
                    for lab in by_cell},
            },
            "grounding_target_finding": (
                "V3-EXQ-817a's ABSOLUTE next-state target is action-diluted on this bench "
                "(world_forward carries a z_t skip term), and is recorded per GROUNDED cell "
                "under grounding_absolute_target_probe alongside the world-effect DELTA target "
                "actually used. This is a finding about the objective's transferability, not "
                "a criticism of 817a, whose agent-side z_world is a learned encoder output."),
            "substrate_path_gate_adjudication": (
                "Open corrupting entry mode-governance-engagement co-lists ree_core/utils/config.py "
                "with this driver; its defect is the SalienceCoordinator affinity-input clamp, and "
                "this bench constructs no REEAgent/SalienceCoordinator/regime-occupancy gate. Not "
                "reachable. Degrading/unset overlaps noted in the queue entry."),
        },
        "seeds": seeds,
    }

    out_path = write_flat_manifest(
        manifest,
        dry_run=dry_run,
        config={
            "bench": {
                "world_dim": WORLD_DIM, "self_dim": SELF_DIM, "action_dim": ACTION_DIM,
                "action_object_dim": ACTION_OBJECT_DIM, "num_candidates": NUM_CANDIDATES,
                "horizon": HORIZON, "num_cem_iterations": NUM_CEM_ITERATIONS,
            },
            "cells": [f"{h}/{f}" for h, f in CELLS],
            "modes": MODES,
            "grounding": {"target": "delta", "steps": ground_steps, "lr": GROUND_LR,
                          "batch": GROUND_BATCH, "n_transitions": GROUND_N_TRANSITIONS},
            "thresholds": manifest["pre_registered_thresholds"],
        },
        seeds=seeds,
        script_path=Path(__file__),
        started_at=started_at,
    )
    manifest["_out_path"] = str(out_path)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="V3-EXQ-1009 elite-channel ceiling spike")
    parser.add_argument("--dry-run", action="store_true",
                        help="1 seed, short grounding; manifest relocated out of evidence/")
    args = parser.parse_args()

    result = main(dry_run=args.dry_run)
    out_path = result["_out_path"]

    print()
    print("=== V3-EXQ-1009 elite-channel ceiling spike ===")
    print(f"label:   {result['interpretation']['label']}")
    print(f"outcome: {result['outcome']}")
    print(f"summary: {result['interpretation']['summary']}")
    print(f"routing: {result['interpretation']['routing']}")
    print("--- per-cell (mean over seeds) ---")
    for lab, c in result["cell_summary"].items():
        print(f"  {lab:22s} reloc={c['relocation_ratio_mean']!s:>22s}  "
              f"delta_dbar={c['delta_dbar_mean']!s:>22s}  "
              f"raw_disp={c['raw_centroid_displacement_mean']!s:>22s}  "
              f"spread={c['mean_per_dim_spread_mean']!s:>22s}")
    print(f"manifest: {out_path}")

    _outcome_raw = str(result["outcome"]).upper()
    emit_outcome(
        outcome=_outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL",
        manifest_path=out_path,
        dry_run=args.dry_run,
    )
