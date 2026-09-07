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
The record (section 6) specifies the DV as delta_dbar = dbar(ORACLE) - dbar(CTRL) and
the gate as delta_dbar >= 0.02. BOTH ARE KEPT. What changed, and had to change, is the
SAMPLING-SEED REGIME the statistic is computed under -- and the reason is measured, not
argued.

  THE DEFECT. dbar is a BETWEEN-MODE separation. Under the lineage's per-mode seed
  convention (offset 7919, from 869/923/928/1005) the four CTRL centroids already differ
  by ~0.10-0.13 standardised units of pure sampling noise, randomly oriented, while the
  oracle's centroid displacement is ~0.001-0.003 raw. delta_dbar therefore measures the
  PROJECTION of a tiny displacement onto a large randomly-oriented baseline: random-signed,
  and measured NEGATIVE on the production reference cell itself. A quantity that is
  supposed to be a ceiling cannot be negative.

  WHAT DID NOT FIX IT. Making the oracle directions maximally separating (a regular
  simplex, all six pairwise distances equal) was tried FIRST, on the theory that aligning
  the displacements would put them in the same subspace dbar measures. It did not work,
  and it is worth stating why since the reasoning is seductive: aligning the displacements
  with EACH OTHER does nothing about their projection onto the BASELINE, which is what
  actually carries the sign. Measured at n=2, the simplex made the reference cell MORE
  negative, not less (-0.005285 against -0.000728 for independent random directions).

  WHAT DID FIX IT. Collapse the baseline. Mode conditioning is disabled in every cell of
  this spike -- the oracle IS the content manipulation here -- so giving every mode the
  SAME sampling draw makes the four CTRL runs bit-identical and dbar(CTRL) exactly 0.
  Measured: 0.000000 in all 12 pilot cells, and carried as the readiness precondition
  ctrl_arm_mode_blind_under_shared_seed. delta_dbar = dbar(ORACLE) - 0 is then a pure,
  NON-NEGATIVE, FIRST-ORDER measure of the separation elite selection alone can create --
  positive in all 12 pilot cells -- on exactly the standardised scale the 0.02 floor was
  pre-registered against. The simplex directions are RETAINED, because with the baseline
  gone they now do the job they were meant to: every mode pair is equally separated, so no
  pair is privileged in the six-pair average.

BOTH SEED REGIMES ARE RUN on every cell. "shared" carries the gate. "per_mode" is recorded
as the 1005-comparable secondary and is what the archived-probe-C reproduction precondition
reads, so the link to the record that commissioned this spike is preserved rather than
traded away. custom_information.seed_regime_comparison holds both side by side.

relocation_ratio (per-mode centroid displacement in that cell's own sampling-noise units)
is retained as a further recorded secondary. It is a useful cross-check on the same
question in a different form, but it is NOT gated: on the previous iteration it was tried
as the gated statistic and the inherited 0.02 floor turned out to be vacuous on its scale
(every cell, including the production null, cleared it), which is precisely the gate-fitting
trap this design must not fall into.

The centroid is computed INJECTION-FREE. _inject_support_preserving_candidates runs
after the CEM (module.py:2356) on the pool final_summary reads (:2325, :2494), and its
scaffolds are one one-hot step followed by exact zeros (:1223-1230), each dragging an
action dimension's mean by ~1/64 = 0.0156 -- above the displacement being measured. It
also fires conditionally on the pool's class count, which the ORACLE arm changes by
construction, so it can fire asymmetrically between the two arms of a pair. Injected
trajectories carry source="support_preserving_cem_injected" (:1489) and are filtered
out; the centroid is then recomputed exactly as _summarize_action_tensor does
(:748-750). Arm-symmetry of injection is additionally a readiness precondition.

TWO PRE-REGISTERED CRITERIA.

  C1: projected_lineage_increment >= CONTENT_FLOOR_ABS = 0.02 in at least one scorable
      cell, where

          projected_lineage_increment = sqrt(B^2 + d^2) - B

      with d the clean displacement measured under the gated (shared) seed regime and B
      that cell's own sampling-noise baseline, measured directly as dbar(CTRL) under the
      lineage regime.

  C2 (attribution guard): that cell must also show a raw, UNSTANDARDISED centroid
      displacement ||mu_ORACLE - mu_CTRL|| at least equal to the FROZEN/floor0.2
      production reference cell's.

WHY C1 IS A PROJECTION AND NOT delta_dbar ITSELF -- the third thing this iteration had to
get right, and the one its own red-team caught. The 1005 gate is an INCREMENT over a
per-mode-seeded sampling-noise baseline of B ~ 0.11-0.13. An arm adding a displacement d
that is generically ORTHOGONAL to that baseline -- and it is, because B is sampling noise,
which nothing in the manipulation is aligned with -- moves dbar by sqrt(B^2 + d^2) - B,
which for d << B is d^2/(2B), NOT by d. The shared regime measures d cleanly (baseline 0),
so gating 0.02 on d directly would apply the record's number to a differently-scaled
quantity and make the floor about 8x EASIER. That is the mirror image of the previous
iteration, where restating the DV as relocation_ratio made the same floor vacuous. Neither
is inheriting a threshold; both are quietly moving it.

So the statistic is measured where it can be measured cleanly and then projected onto the
scale the threshold belongs to, using each cell's own measured B. The exact form is used,
not the d^2/(2B) expansion. The directly-measured lineage increment is carried alongside
every cell as the noisy empirical cross-check on the orthogonality assumption; on the
pilot the two agree to the same order (GROUNDED/floor0.2: projected 0.0025 from d=0.0242
and B=0.114, measured +0.00045).

C2 is load-bearing and not redundant. The projection is standardised by the pooled sample
spread, and the ao_std_floor=0.0 level COLLAPSES that spread, so a cell can post a larger
projected increment while its centroid moved LESS in absolute terms. Measured: the
floor0.0 cells post raw displacements of 0.000239 and 0.000982 against the production
reference's 0.000967 -- C2 is what stops the collapsed-CEM route being read as a clear,
which is the unattributability that blocked V3-EXQ-1005.

Both criteria are evaluated only on SCORABLE cells (see the injection handling below), and
the pair is jointly satisfiable: a cell with a genuinely larger channel posts both a larger
projected increment and a larger raw displacement, as GROUNDED/floor0.2 does on the pilot.

DECISION RULE (pre-registered, per the record's section 6):
  * a cell clearing C1 AND C2 -> the 1005 design IS runnable in that regime;
    author successor V3-EXQ-1005a on that bench.
  * a cell clearing C1 but failing C2 -> the lift came from the collapsed CEM spread
    shrinking the standardisation denominator, not from the elite channel relocating the
    centroid; report, do not build.
  * NO cell clearing C1 -> the elite-selection channel cannot, even at its oracle
    ceiling, produce the increment the 1005 design needs; MECH-267's content assertion is
    not measurable by
    proposal-output centroid at production CEM settings on any bench tested here;
    route to /governance to either narrow what_would_answer to the breadth channel
    or register a complicated (buildable) entry in substrate_queue.json (ao_std
    floor policy under mode conditioning, or E2 action-object action-dependence).
    This script does NOT register that build -- registering ahead of the spike's
    fact is the complicated-before-complex inversion the work-graph vocabulary
    warns against, and the record is explicit about it.

POSITIVE CONTROL ON THE INSTRUMENT ITSELF. The FROZEN/floor0.2 cell is the 1005 bench's
configuration, and under the LINEAGE (per-mode) seed regime its delta_dbar must stay in the
archived probe-C band (|delta| <= 0.005) -- carried as a readiness precondition, so a bench
that does not behave like the one that produced the refusal cannot be read as a ceiling
measurement. Stated precisely, because it is a bound and not a point match: this driver
rebuilds the residue terrain on its own RNG stream, so it reproduces the archived MAGNITUDE
(order 1e-5..1e-3, sign varying) rather than the archived digits. That is what the
precondition asserts and all it asserts. It reads the LINEAGE regime deliberately: that is
the configuration probe C measured, and it is the reason both regimes are run.

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

# Seed for the fixed orthonormal frame the oracle simplex is expressed in. A
# constant, never derived from a run.
ORACLE_FRAME_SEED = 4242

# Both are run every cell. GATED_REGIME carries C1/C2; the other is the
# 1005-comparable secondary. See _cell_sampling_seed for why.
SEED_REGIMES = ["shared", "per_mode"]
GATED_REGIME = "shared"
LINEAGE_REGIME = "per_mode"
CTRL_MODE_BLIND_TOL = 1e-9   # dbar(CTRL) under the shared regime must be 0
# Minimum injection-clean seeds a cell must retain to be scorable. Contaminated
# (seed, cell) rows are EXCLUDED from that cell's aggregate rather than vacating
# the whole run -- CLAUDE.md forbids AND-ing a precondition across arms.
MIN_CLEAN_SEEDS_PER_CELL = 3

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


def _pred_ctrl_mode_blind(v: float) -> bool:
    return abs(float(v)) <= CTRL_MODE_BLIND_TOL


def _pred_clean_seeds(v: float) -> bool:
    return float(v) >= float(MIN_CLEAN_SEEDS_PER_CELL)


def _pred_simplex_uniform(v: float) -> bool:
    # max-minus-min pairwise direction distance; a regular simplex has 0 spread.
    return abs(float(v)) <= 1e-6


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
    "ctrl_arm_mode_blind_under_shared_seed": {
        "cells": [0.0, 0.0, 0.0, 0.0],
        "score_fn": _pred_ctrl_mode_blind,
        "source": "1009 iter3 shared-seed pilot: dbar_ctrl measured exactly 0.000000 in all 12 cells",
    },
    "at_least_two_scorable_cells": {
        "cells": [4, 3, 2],
        "score_fn": lambda v: float(v) >= 2.0,
        "source": "scorable-cell counts reachable at n=5; the n=3 pilot left 3 of 4 scorable",
    },
    "oracle_directions_maximally_separating": {
        "cells": [0.0],
        "score_fn": _pred_simplex_uniform,
        "source": "regular simplex: all 6 pairwise distances equal sqrt(8/3), spread 0 by construction",
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


def _cell_sampling_seed(seed: int, mode: str, seed_regime: str) -> int:
    """CEM sampling seed, SHARED across cells and across the CTRL/ORACLE contrast.

    Common random numbers throughout: the CTRL and ORACLE runs of a cell consume the
    identical standard-normal sequence (the oracle changes which indices are returned,
    never how many draws are taken), so their difference isolates elite choice rather
    than sampling noise.

    TWO REGIMES, and the difference between them is the whole third iteration.

    "per_mode" (the 869/923/928/1005 convention, offset 7919) gives each mode its own
    draw. That is right for the LINEAGE, whose arms had mode conditioning ON and needed a
    sampling-noise null to beat -- but it is fatal to an ORACLE CEILING measurement. The
    per-mode draws give the four CTRL centroids a baseline between-mode separation of
    ~0.10-0.13 standardised units, randomly oriented, while the oracle's displacement is
    ~0.001-0.003 raw. dbar then moves by the PROJECTION of a tiny displacement onto a
    large randomly-oriented baseline -- random-signed, and measured negative on the
    production reference cell itself under BOTH independent-random and maximally-
    separating oracle directions (-0.000728 and -0.005285 at n=2). No choice of oracle
    direction repairs that, because the baseline, not the directions, is what makes the
    sign a coin flip.

    "shared" gives every mode the SAME draw. Mode conditioning is disabled in every cell
    of this spike (the oracle IS the manipulation), so under a shared draw the four CTRL
    runs are BIT-IDENTICAL and dbar(CTRL) is exactly 0 -- verified, and carried as a
    readiness precondition. delta_dbar = dbar(ORACLE) - 0 is then a pure, non-negative,
    FIRST-ORDER measure of the separation elite selection alone can create, which is
    exactly the ceiling the spike is after, and it is on the same standardised scale the
    0.02 floor was pre-registered against.

    Both regimes are run. "shared" carries the gate; "per_mode" is recorded as the
    1005-comparable secondary and is what the archived-probe-C reproduction precondition
    reads, so the link to the record that commissioned this spike is not lost.
    """
    if seed_regime == "shared":
        return seed * 104_729
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


def _maximally_separating_directions() -> List[torch.Tensor]:
    """One unit direction per mode, arranged as a REGULAR SIMPLEX.

    THIS IS THE FIX THAT MAKES delta_dbar A GENUINE UPPER BOUND, and it is the whole
    reason the pre-registered 0.02 floor can be inherited honestly.

    The DV is a BETWEEN-MODE separation. If the oracle displaces mode m's centroid by
    d*u_m, the pair (a, b) separation moves by d*||u_a - u_b||. With INDEPENDENT RANDOM
    u_m -- the previous construction -- E[u_a . u_b] = 0, so the displacement is
    orthogonal to the existing separation in expectation and enters dbar only to SECOND
    order (~d^2/2R), and its sign is a coin flip: measured NEGATIVE on the production
    reference cell (-0.000485 at n=5) and on GROUNDED/floor0.2 (-0.004784). A quantity
    that is supposed to be a ceiling cannot be negative, which is exactly why the
    previous iteration could not carry the gate.

    Arranging the directions to be maximally separating makes the displacement ALIGNED
    with the quantity dbar measures, so it enters at FIRST order (d*||u_a - u_b||) and
    is non-negative in expectation for every pair simultaneously.

    Why a regular simplex specifically, over the obvious alternative of two antipodal
    pairs (+u, -u, +v, -v): with 4 modes not all 6 pairs can be antipodal, so the choice
    is which pairwise structure to maximise. Antipodal pairs give 2 pairs at ||.||=2 and
    4 pairs at sqrt(2), mean 1.609, and privileges two pairs over the other four. The
    regular simplex gives ALL SIX pairs ||u_a - u_b|| = sqrt(8/3) ~ 1.633 -- a higher
    mean AND uniform, so no mode pair is privileged and dbar (which averages the six
    pairs) is maximised without being dominated by a favoured subset.

    Construction: the standard tetrahedron vertices (+-1, +-1, +-1) with an even number
    of minus signs, normalised, then embedded in the ACTION_DIM space through a fixed
    seed-stable random orthonormal frame so that no action dimension is privileged. The
    frame is drawn from a LOCAL generator, so the global RNG stream is untouched.
    """
    tetra = torch.tensor([
        [1.0, 1.0, 1.0],
        [1.0, -1.0, -1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0],
    ]) / math.sqrt(3.0)
    if len(MODES) != 4:
        raise ValueError(
            f"the simplex construction is written for exactly 4 modes, got {len(MODES)}"
        )
    gen = torch.Generator().manual_seed(ORACLE_FRAME_SEED)
    # A random orthonormal 3-frame in R^ACTION_DIM: QR of a random ACTION_DIM x 3.
    q, _ = torch.linalg.qr(torch.randn(ACTION_DIM, 3, generator=gen))  # [action_dim, 3]
    dirs = [(q @ tetra[i]) for i in range(len(MODES))]
    return [d / d.norm() for d in dirs]


def _archived_probe_c_directions() -> List[torch.Tensor]:
    """Independent random per-mode directions -- the ARCHIVED probe-C construction.

    Used ONLY for the lineage (per-mode seed) regime, whose entire job is to reproduce
    the measurement that produced the V3-EXQ-1005 refusal. That probe drew one
    independent random direction per mode (exq1005_probe_elite_channel.py section C), so
    reproducing it requires the same draw, not this spike's simplex: swapping the oracle
    construction under the reproduction check compares two different measurements and
    the check correctly fails, which is how this was caught.

    Same generator seed and normalisation as the archived probe.
    """
    gen = torch.Generator().manual_seed(ORACLE_FRAME_SEED)
    dirs = [torch.randn(ACTION_DIM, generator=gen) for _ in MODES]
    return [d / d.norm() for d in dirs]


def _direction_separation_stats(dirs: List[torch.Tensor]) -> Dict[str, float]:
    """Pairwise geometry of the oracle directions -- the readiness evidence that the
    construction is actually maximally separating rather than silently degenerate."""
    dists = [float((dirs[i] - dirs[j]).norm())
             for i, j in itertools.combinations(range(len(dirs)), 2)]
    return {
        "mean_pairwise_distance": statistics.fmean(dists),
        "min_pairwise_distance": min(dists),
        "max_pairwise_distance": max(dists),
        "regular_simplex_reference": math.sqrt(8.0 / 3.0),
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
    oracle: Optional[Any], seed_regime: str,
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
    torch.manual_seed(_cell_sampling_seed(seed, mode, seed_regime))
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
    archived_directions: List[torch.Tensor],
    ground_steps: int, cell_index: int, n_cells: int,
) -> Dict[str, Any]:
    """One (seed x cell) unit: CTRL and ORACLE proposer runs over all 4 modes."""
    label = f"{ao_head}/{cem_floor}"
    print(f"Seed {seed} Condition {label}", flush=True)

    # each mode contributes a CTRL and an ORACLE run, in each of the two seed regimes
    total_units = len(MODES) * 2 * len(SEED_REGIMES)

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

        # BOTH seed regimes. "shared" carries the gate (pure oracle bound); "per_mode"
        # is the 1005-comparable secondary and feeds the archived-reproduction check.
        by_regime: Dict[str, Dict[str, Dict[str, Any]]] = {}
        elite_calls: List[int] = []
        unit = 0
        for regime in SEED_REGIMES:
            # The gated regime uses the maximally-separating simplex; the lineage
            # regime uses the archived probe-C construction it must reproduce.
            regime_dirs = (directions if regime == GATED_REGIME
                           else archived_directions)
            ctrl_r: Dict[str, Dict[str, Any]] = {}
            orc_r: Dict[str, Dict[str, Any]] = {}
            for mode_index, mode in enumerate(MODES):
                ctrl_r[mode] = _propose(e2, cem_floor, seed, mode, oracle=None,
                                        seed_regime=regime)
                unit += 1
                print(f"  [train] {label} seed={seed} ep {unit}/{total_units} "
                      f"regime={regime} mode={mode} arm=CTRL", flush=True)
                orc_r[mode] = _propose(e2, cem_floor, seed, mode,
                                       oracle=_oracle_chooser(regime_dirs[mode_index]),
                                       seed_regime=regime)
                unit += 1
                print(f"  [train] {label} seed={seed} ep {unit}/{total_units} "
                      f"regime={regime} mode={mode} arm=ORACLE", flush=True)
                elite_calls.append(ctrl_r[mode]["elite_fn_calls"])
                elite_calls.append(orc_r[mode]["elite_fn_calls"])
            by_regime[regime] = {"ctrl": ctrl_r, "oracle": orc_r}

        ctrl = by_regime[GATED_REGIME]["ctrl"]
        orc = by_regime[GATED_REGIME]["oracle"]

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

        # Per-regime summary, so the gated bound and the 1005-comparable secondary are
        # both auditable side by side.
        regime_stats: Dict[str, Any] = {}
        for regime, arms in by_regime.items():
            c_, o_ = arms["ctrl"], arms["oracle"]
            dc_, do_ = dbar(c_), dbar(o_)
            rds = [_raw_displacement(c_[m]["mean_by_action_dim"],
                                     o_[m]["mean_by_action_dim"]) for m in MODES]
            regime_stats[regime] = {
                "dbar_ctrl": dc_,
                "dbar_oracle": do_,
                "delta_dbar": (do_ - dc_) if (dc_ is not None and do_ is not None) else None,
                "raw_centroid_displacement": (
                    statistics.fmean([r for r in rds if r is not None])
                    if all(r is not None for r in rds) else None),
            }

        # ---- THE GATED STATISTIC: the clean displacement, projected onto the scale
        # the 0.02 floor was actually pre-registered against.
        #
        # delta_dbar under the shared regime is the raw standardised displacement d that
        # the elite channel can produce (baseline 0, so it is measured directly and
        # cleanly). But the V3-EXQ-1005 gate is an INCREMENT over a per-mode-seeded
        # sampling-noise baseline B ~ 0.11-0.13: an arm adding a displacement d that is
        # generically ORTHOGONAL to that baseline moves dbar by sqrt(B^2 + d^2) - B, not
        # by d. Gating 0.02 on d directly would apply the record's number to a
        # differently-scaled quantity and make the floor ~8x EASIER -- the mirror image of
        # the previous iteration, where a different restatement made it vacuous. Neither
        # is inheriting a threshold.
        #
        # So: measure d cleanly (shared regime), measure B directly (lineage regime's own
        # CTRL arm), and project. The exact form is used, not the d^2/2B expansion.
        # Orthogonality is generic here because B is sampling noise, which nothing in the
        # manipulation is aligned with; the directly-measured lineage increment is carried
        # alongside as the noisy empirical cross-check on exactly this assumption.
        d_shared = regime_stats[GATED_REGIME]["delta_dbar"]
        b_lineage = regime_stats[LINEAGE_REGIME]["dbar_ctrl"]
        projected = (
            math.sqrt(b_lineage ** 2 + d_shared ** 2) - b_lineage
            if (d_shared is not None and b_lineage is not None and b_lineage > 0)
            else None
        )

        row: Dict[str, Any] = {
            "arm_id": label,
            "projected_lineage_increment": projected,
            "clean_displacement_d": d_shared,
            "lineage_baseline_B": b_lineage,
            "measured_lineage_increment": regime_stats[LINEAGE_REGIME]["delta_dbar"],
            "seed_regime_stats": regime_stats,
            "gated_regime": GATED_REGIME,
            "dbar_ctrl_gated_regime_abs": (
                abs(regime_stats[GATED_REGIME]["dbar_ctrl"])
                if regime_stats[GATED_REGIME]["dbar_ctrl"] is not None else None),
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

    clears_c1 = projected is not None and projected >= CONTENT_FLOOR_ABS
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

    # Per-mode oracle directions: a MAXIMALLY SEPARATING regular simplex, shared
    # across every cell so all cells face the identical oracle. See
    # _maximally_separating_directions for why this, and not independent random draws,
    # is what makes delta_dbar a genuine first-order upper bound.
    directions = _maximally_separating_directions()
    direction_stats = _direction_separation_stats(directions)
    archived_directions = _archived_probe_c_directions()

    rows: List[Dict[str, Any]] = []
    n_cells = len(seeds) * len(CELLS)
    idx = 0
    for seed in seeds:
        for ao_head, cem_floor in CELLS:
            rows.append(_run_cell(seed, ao_head, cem_floor, directions,
                                  archived_directions, ground_steps, idx, n_cells))
            idx += 1

    # ---- per-cell aggregation across seeds ------------------------------------
    by_cell: Dict[str, Dict[str, Any]] = {}
    for ao_head, cem_floor in CELLS:
        label = f"{ao_head}/{cem_floor}"
        sel_all = [r for r in rows if r["arm_id"] == label]
        # Scoring uses only rows where injection fired symmetrically between the CTRL and
        # ORACLE arms. An asymmetric firing does not merely add synthetic candidates (those
        # are already filtered from the centroid) -- it REPLACES real ones
        # (module.py:1501 keep_n = total_budget - len(injected)), so the two arms were
        # selected from differently-composed pools and the contrast is not clean.
        sel = [r for r in sel_all if r["injection_arm_symmetric"]]
        dropped = [f"seed{r['seed']}" for r in sel_all if not r["injection_arm_symmetric"]]
        deltas = [r["delta_dbar"] for r in sel if r["delta_dbar"] is not None]
        relocs = [r["relocation_ratio"] for r in sel if r["relocation_ratio"] is not None]
        projs = [r["projected_lineage_increment"] for r in sel
                 if r["projected_lineage_increment"] is not None]
        bases = [r["lineage_baseline_B"] for r in sel if r["lineage_baseline_B"] is not None]
        disps = [r["raw_centroid_displacement"] for r in sel
                 if r["raw_centroid_displacement"] is not None]
        by_cell[label] = {
            "arm_id": label, "ao_head": ao_head, "cem_floor": cem_floor,
            "n_seeds": len(sel),
            "n_seeds_attempted": len(sel_all),
            "n_seeds_dropped_injection_asymmetric": len(dropped),
            "seeds_dropped_injection_asymmetric": dropped,
            "scorable": len(sel) >= MIN_CLEAN_SEEDS_PER_CELL,
            "projected_lineage_increment_per_seed": [
                r["projected_lineage_increment"] for r in sel],
            "projected_lineage_increment_mean": statistics.fmean(projs) if projs else None,
            "lineage_baseline_B_mean": statistics.fmean(bases) if bases else None,
            "shortfall_factor_vs_floor": (
                CONTENT_FLOOR_ABS / statistics.fmean(projs)
                if projs and statistics.fmean(projs) > 0 else None),
            "relocation_ratio_per_seed": [r["relocation_ratio"] for r in sel],
            "relocation_ratio_mean": statistics.fmean(relocs) if relocs else None,
            "relocation_ratio_max": max(relocs) if relocs else None,
            "injection_arm_symmetric_all_seeds": len(dropped) == 0,
            "delta_dbar_lineage_regime_mean": statistics.fmean(
                [r["seed_regime_stats"][LINEAGE_REGIME]["delta_dbar"] for r in sel
                 if r["seed_regime_stats"][LINEAGE_REGIME]["delta_dbar"] is not None]
                or [float("nan")]),
            "dbar_ctrl_gated_regime_max_abs": max(
                (abs(r["seed_regime_stats"][GATED_REGIME]["dbar_ctrl"]) for r in sel
                 if r["seed_regime_stats"][GATED_REGIME]["dbar_ctrl"] is not None),
                default=None),
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
                [r["across_candidate_ao_std_after"] for r in sel]) if sel else None,
        }

    ref_label = f"{REFERENCE_CELL[0]}/{REFERENCE_CELL[1]}"
    ref = by_cell[ref_label]
    ref_disp = ref["raw_centroid_displacement_mean"]
    ref_reloc = ref["relocation_ratio_mean"]

    # ---- readiness preconditions ----------------------------------------------
    ref_delta_mean = ref["delta_dbar_mean"]
    ref_lineage_delta = ref["delta_dbar_lineage_regime_mean"]
    reproduces = (ref_lineage_delta is not None
                  and not math.isnan(ref_lineage_delta)
                  and _pred_reproduces_ceiling(ref_lineage_delta))

    grounded_rows = [r for r in rows if r["ao_head"] == "GROUNDED"]
    worst_acs_ratio, worst_acs_cell = _worst_cell(
        grounded_rows, "across_candidate_ao_std_ratio", "min")
    worst_spread, worst_spread_cell = _worst_cell(rows, "min_per_dim_spread", "min")
    worst_elite_calls, worst_elite_cell = _worst_cell(rows, "elite_fn_calls_min", "min")

    worst_ctrl_blind, worst_ctrl_blind_cell = _worst_cell(
        rows, "dbar_ctrl_gated_regime_abs", "max")
    dir_spread = (direction_stats["max_pairwise_distance"]
                  - direction_stats["min_pairwise_distance"])

    n_scorable = sum(1 for c in by_cell.values() if c["scorable"])

    preconditions: List[Dict[str, Any]] = [
        {
            "name": "frozen_production_cell_reproduces_archived_ceiling",
            "description": (
                "FROZEN/floor0.2 IS the V3-EXQ-1005 bench, so its oracle ceiling must "
                "reproduce the archived probe-C value (+0.0001..+0.001). If it does not, "
                "this spike is not measuring the instrument that produced the refusal."),
            "measured": (abs(ref_lineage_delta)
                         if (ref_lineage_delta is not None
                             and not math.isnan(ref_lineage_delta)) else None),
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
            "name": "ctrl_arm_mode_blind_under_shared_seed",
            "description": (
                "Under the gated (shared) seed regime with mode conditioning disabled in "
                "every cell, the four CTRL proposer runs must be bit-identical, so "
                "dbar(CTRL) is exactly 0 and delta_dbar is a PURE oracle-induced "
                "separation rather than a difference against a randomly-oriented "
                "sampling-noise baseline. A non-zero value means something mode-dependent "
                "is leaking into the CTRL arm and the bound is not pure. Worst cell "
                "reported."),
            "measured": worst_ctrl_blind,
            "threshold": CTRL_MODE_BLIND_TOL,
            "direction": "upper",
            "control": "max |dbar(CTRL)| over all cells and seeds, gated regime",
            "offending_cell": worst_ctrl_blind_cell,
            "met": bool(worst_ctrl_blind is not None
                        and _pred_ctrl_mode_blind(worst_ctrl_blind)),
        },
        {
            "name": "oracle_directions_maximally_separating",
            "description": (
                "The oracle directions must form a regular simplex -- all six pairwise "
                "distances equal -- so no mode pair is privileged and the induced "
                "separation enters dbar at first order for every pair simultaneously. "
                "Measured as the max-minus-min pairwise distance, which is 0 for a regular "
                "simplex. A non-zero spread means the construction degenerated."),
            "measured": dir_spread,
            "threshold": 1e-6,
            "direction": "upper",
            "control": "sqrt(8/3) reference; see _maximally_separating_directions",
            "met": bool(_pred_simplex_uniform(dir_spread)),
        },
        {
            "name": "at_least_two_scorable_cells",
            "description": (
                "Injection-contaminated (seed, cell) rows are dropped per CELL, and a cell "
                "retaining fewer than MIN_CLEAN_SEEDS_PER_CELL clean seeds is marked "
                "unscorable and excluded from C1/C2 -- it does NOT fail the run. This gate "
                "asks only whether ENOUGH cells survived to discriminate at all. Making it "
                "a run-level AND over every cell's retention would let one contaminated "
                "cell vacate three clean ones, which is exactly the multi-arm gate defect "
                "CLAUDE.md forbids (and which this precondition previously committed)."),
            "measured": float(n_scorable),
            "threshold": 2.0,
            "direction": "lower",
            "control": "count of cells retaining >= MIN_CLEAN_SEEDS_PER_CELL injection-clean seeds",
            "offending_cell": (
                ",".join(lab for lab, c in by_cell.items() if not c["scorable"]) or None),
            "met": bool(n_scorable >= 2),
        },
    ]
    all_preconditions_met = all(bool(p["met"]) for p in preconditions)

    # ---- criteria ---------------------------------------------------------------
    c1_cells = [lab for lab, c in by_cell.items()
                if c["scorable"] and c["projected_lineage_increment_mean"] is not None
                and c["projected_lineage_increment_mean"] >= CONTENT_FLOOR_ABS]
    c1_pass = len(c1_cells) > 0

    def clears_c2(label: str) -> bool:
        """The manipulation must BEAT PRODUCTION on the same statistic C1 gates.

        Measured on the same scale as C1 (no opposing-direction pathology): a cell
        that merely reproduces the production bench's own relocation has not shown
        the manipulation lifts the channel, whatever absolute number it reaches.
        The reference cell IS the production condition, so this is a within-run
        contrast and needs no external calibration."""
        if label == ref_label or not by_cell[label]["scorable"]:
            return False
        d = by_cell[label]["raw_centroid_displacement_mean"]
        return (d is not None and ref_disp is not None
                and d >= RAW_DISPLACEMENT_MIN_RATIO * ref_disp)

    c2_cells = [lab for lab in c1_cells if clears_c2(lab)]
    c2_pass = len(c2_cells) > 0
    c1_only_cells = [lab for lab in c1_cells if lab not in c2_cells]

    deltas_all = [c["projected_lineage_increment_mean"] for c in by_cell.values()
                  if c["scorable"] and c["projected_lineage_increment_mean"] is not None]
    # Non-degenerate requires at least two SCORABLE cells that actually differ -- a run
    # in which injection contamination left one scorable cell cannot discriminate.
    c1_non_degenerate = (
        len(deltas_all) >= 2 and (max(deltas_all) - min(deltas_all)) > 1e-9
    )
    # C2 can only discriminate if there is a C1-clearing cell to test it on.
    c2_non_degenerate = bool(c1_pass)

    criteria = [
        {"name": "C1_any_cell_clears_content_floor", "load_bearing": True,
         "passed": bool(c1_pass), "threshold": CONTENT_FLOOR_ABS,
         "clearing_cells": c1_cells,
         "statistic": "projected_lineage_increment = sqrt(B^2 + d^2) - B",
         "unscorable_cells": [lab for lab, c in by_cell.items() if not c["scorable"]],
         "note": ("statistic AND threshold both the record's own. The clean displacement d "
                  "is measured under the shared regime and projected onto the lineage scale "
                  "the 0.02 was pre-registered against, sqrt(B^2+d^2)-B, using each cell's "
                  "own measured baseline B. Gating 0.02 on d directly would make the floor "
                  "~8x easier; see the docstring.")},
        {"name": "C2_manipulation_beats_production_reference", "load_bearing": True,
         "passed": bool(c2_pass), "clearing_cells": c2_cells,
         "reference_cell": ref_label, "reference_raw_displacement": ref_disp,
         "statistic": "raw_centroid_displacement",
         "note": ("attribution guard on the UNSTANDARDISED numerator: a cell whose "
                  "projected increment rises only because its standardisation denominator "
                  "collapsed has not relocated anything. Measured on the pilot, this is "
                  "what separates the floor0.0 cells (raw displacement 0.000239/0.000982) "
                  "from the production reference (0.000967).")},
    ]
    combination_rule = (
        "A cell clears the elite-channel ceiling only if it satisfies C1 AND C2. C1 gates the "
        "record's own 0.02 on the record's own scale, reached by measuring the elite channel's "
        "displacement cleanly (shared seed regime, zero baseline) and projecting it onto the "
        "per-mode-seeded lineage scale via sqrt(B^2+d^2)-B with each cell's measured B. C2 is "
        "an attribution guard on the unstandardised numerator, so a collapsed standardisation "
        "denominator cannot manufacture a clear. Both are evaluated on an injection-free "
        "centroid, over injection-clean seeds only, on cells that retained enough clean seeds."
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
            f"(raw displacement {ref_disp}) on an injection-free centroid. The "
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
            f"(raw displacement {ref_disp}) -- so the floor is cleared by the bench's "
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
            f"floor on the lineage scale (production reference {ref_label} = "
            f"{ref['projected_lineage_increment_mean']}; shortfall factors "
            f"{ {lab: c['shortfall_factor_vs_floor'] for lab, c in by_cell.items()} }). "
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
                    lab: c["delta_dbar_mean"] for lab, c in by_cell.items()},
                "headroom_ratio_by_cell": {
                    lab: (c["delta_dbar_mean"] / CONTENT_FLOOR_ABS
                          if c["delta_dbar_mean"] is not None else None)
                    for lab, c in by_cell.items()},
                "secondary_relocation_ratio_by_cell": {
                    lab: c["relocation_ratio_mean"] for lab, c in by_cell.items()},
                "margin": 1.0,
            },
            "red_team_dispositions_iter3": {
                "description": (
                    "Findings from the third-iteration adversarial design review (verdict "
                    "CONTESTED) that were NOT repaired in code, each with its disposition. "
                    "Recorded so a reader adjudicating this run sees the same limitations "
                    "the design did. The three that WERE repaired -- the C1 scale mismatch, "
                    "the run-level AND on cell scorability, and manifest notes describing a "
                    "superseded criterion -- are not listed here; they are gone."),
                "oracle_is_greedy_not_exhaustive": (
                    "The oracle ranks candidates in ACTION space while the refit averages in "
                    "action-object space, so it is a greedy chooser, not an argmax over all "
                    "C(16,3)=560 elite subsets; and the final pool descends from the "
                    "iteration-1 refit, so one of its three firings does not affect the "
                    "measured centroid. The measured ceiling is therefore a LOWER bound on "
                    "the true oracle ceiling. THIS IS THE LIMITATION THAT MOST CONSTRAINS A "
                    "NULL: a no-clear verdict means the channel is short at THIS oracle's "
                    "strength, and the shortfall factors (recorded per cell) are what say "
                    "whether a stronger oracle could plausibly close the gap -- a 1.7x "
                    "shortfall is not the same finding as a 41x one."),
                "injection_drop_is_dv_conditioned": (
                    "Rows dropped for arm-asymmetric injection are selected on the final "
                    "pool's first-action class diversity, which the ORACLE arm changes by "
                    "construction, so the surviving aggregate is conditioned on something "
                    "correlated with the DV. Mitigated (per-cell drop, minimum clean-seed "
                    "retention, drop counts recorded per cell) but NOT eliminated: read "
                    "n_seeds_dropped_injection_asymmetric alongside any cell's result."),
                "c2_compares_across_the_injection_boundary": (
                    "Injection fires under floor0.2 and not under floor0.0, so the two levels "
                    "score centroids over differently-composed candidate pools even after the "
                    "synthetic scaffolds are filtered out. C2's cross-level comparison "
                    "inherits that. It is recorded rather than repaired because removing it "
                    "would mean disabling support-preserving CEM, which is the three-factor "
                    "change this iteration exists to avoid."),
                "two_anchors_cannot_fail_by_construction": (
                    "ctrl_arm_mode_blind_under_shared_seed and "
                    "oracle_directions_maximally_separating are structural: their references "
                    "are exactly 0 because the property is guaranteed by construction (all "
                    "operating_mode compute-readers gate on mode_conditioning_enabled; the "
                    "simplex is uniform by definition). They are LEAK CHECKS, not powered "
                    "gates -- they earn their place by failing loudly if a future substrate "
                    "change breaks the guarantee, not by discriminating between benches."),
                "simplex_embedded_in_a_fixed_3_slice": (
                    "The tetrahedron spans a 3-D subspace of the 4-D action space, chosen by "
                    "a fixed seeded frame. 'Maximally separating' is therefore exact within "
                    "that slice, and per-dimension standardisation means it is not exactly "
                    "distance-preserving in the standardised metric the DV uses. Recorded; "
                    "the effect is second-order relative to the shortfall factors measured."),
            },
            "oracle_direction_geometry": {
                "description": (
                    "The oracle directions form a regular simplex so all six mode pairs are "
                    "equally separated. This is what lets delta_dbar be a first-order "
                    "non-negative bound -- but only in combination with the shared seed "
                    "regime: see seed_regime_comparison, where the SAME simplex directions "
                    "give a negative delta_dbar under per-mode seeds."),
                **direction_stats,
                "frame_seed": ORACLE_FRAME_SEED,
                "lineage_regime_uses_archived_probe_c_directions": True,
            },
            "seed_regime_comparison": {
                "description": (
                    "The measured justification for gating on the shared regime. Under "
                    "per-mode seeds the four CTRL centroids carry a large, randomly-oriented "
                    "baseline separation and the oracle's tiny displacement enters dbar as a "
                    "random-signed projection onto it -- delta_dbar goes NEGATIVE, and no "
                    "choice of oracle direction repairs that (measured under both "
                    "independent-random and maximally-separating directions). Under shared "
                    "seeds dbar(CTRL)=0 exactly and delta_dbar is the pure oracle-induced "
                    "separation. Both are recorded per cell; only the shared regime is gated."),
                "gated_regime": GATED_REGIME,
                "lineage_regime": LINEAGE_REGIME,
                "delta_dbar_gated_by_cell": {
                    lab: c["delta_dbar_mean"] for lab, c in by_cell.items()},
                "delta_dbar_lineage_by_cell": {
                    lab: c["delta_dbar_lineage_regime_mean"] for lab, c in by_cell.items()},
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
                        "delta_dbar_mean": c["delta_dbar_mean"],
                        "relocation_ratio_mean": c["relocation_ratio_mean"],
                        "raw_centroid_displacement_mean": c["raw_centroid_displacement_mean"],
                        "mean_per_dim_spread_mean": c["mean_per_dim_spread_mean"],
                        "clears_c1": bool(c["delta_dbar_mean"] is not None
                                          and c["delta_dbar_mean"] >= CONTENT_FLOOR_ABS),
                        "clears_c2_beats_production": bool(
                            lab != ref_label and c["raw_centroid_displacement_mean"] is not None
                            and ref_disp is not None
                            and c["raw_centroid_displacement_mean"]
                            >= RAW_DISPLACEMENT_MIN_RATIO * ref_disp),
                    }
                    for lab, c in by_cell.items()
                },
                "any_cell_would_clear_c2_independently": bool(any(
                    c["raw_centroid_displacement_mean"] is not None and ref_disp is not None
                    and c["raw_centroid_displacement_mean"]
                    >= RAW_DISPLACEMENT_MIN_RATIO * ref_disp
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
        _pi = c["projected_lineage_increment_mean"]
        _sf = c["shortfall_factor_vs_floor"]
        print(f"  {lab:22s} C1_projected={_pi if _pi is None else round(_pi, 6)!s:>10s}  "
              f"(d={c['delta_dbar_mean'] if c['delta_dbar_mean'] is None else round(c['delta_dbar_mean'], 5)!s:>8s} "
              f"B={c['lineage_baseline_B_mean'] if c['lineage_baseline_B_mean'] is None else round(c['lineage_baseline_B_mean'], 5)!s:>8s})  "
              f"shortfall={_sf if _sf is None else round(_sf, 1)!s:>7s}x  "
              f"raw_disp={c['raw_centroid_displacement_mean'] if c['raw_centroid_displacement_mean'] is None else round(c['raw_centroid_displacement_mean'], 6)!s:>9s}  "
              f"scorable={c['scorable']}")
    print(f"manifest: {out_path}")

    _outcome_raw = str(result["outcome"]).upper()
    emit_outcome(
        outcome=_outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL",
        manifest_path=out_path,
        dry_run=args.dry_run,
    )
