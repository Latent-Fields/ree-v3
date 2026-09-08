"""V3-EXQ-1013 -- SD-031 shortcut-vs-model portfolio (V3-EXQ-1001 successor; GOV-FANOUT-1).

Registry question: sd031_causal_signature_shortcut_vs_model (legs H-shortcut / H-budget /
H-testbed-degenerate). Pre-registered by the CONFIRMED autopsy
REE_assembly/evidence/planning/failure_autopsy_V3-EXQ-1001_2026-09-04.{md,json}
(targets[0].fanout_recommendation, revision_note_2026_09_04, red_team F1/F4/F5). Ratified at
the /governance Step 8 gate 2026-09-04 (governance-20260904-1347), which ALSO amended
SD-031's what_would_answer so a construction-balanced (RandomPolicy, offline-scored)
comparator-only design clears the ARC-065 diversity half of its gate. Supersedes V3-EXQ-1001.

THE QUESTION 1001 COULD NOT ANSWER. 1001 rewired CausalGridWorldV2's action->displacement map
and found that the SD-031 comparator (residual_world = z_world_obs - E2WorldForward(z_prev, a)),
readapted for 120 optimizer steps on 3600 post-shift transitions, beat its own action-shuffled
control by only +0.0117 (5/5 seeds, many-sigma, 4.3x below the pre-registered 0.05 floor). The
autopsy showed that number is UNATTRIBUTABLE: (a) no same-budget positive control says what a
120-step re-derivation SHOULD achieve (the floor had no anchor); (b) the budget was 1.82% of the
original 6600-step fit, framed as "13% of epochs"; (c) the shuffled control was ONE permutation
draw per arm, with per-seed draw noise (0.0103) the same order as the effect. Three hypotheses
are live and the open question is WHICH -- so this is a fan-out, not a re-pose.

THREE LEGS, THREE AXIS FAMILIES (never a power-bump of 1001's design):

  H1 / H-shortcut (axis: measurement).  The SAME post-shift readapt budget as 1001's
      discriminating control (B0 = 3600 transitions x 8 epochs = 120 optimizer steps at
      batch 256, identical arithmetic) given to THREE starts on the SAME data, seeds and
      held-out set: READAPT (the trained PRE_BASE weights, as 1001), SCRATCH (the shared
      INIT weights PRE_BASE was itself trained from -- carried as an object, identity
      recorded by hash), and PARTIAL (the trained PRE_BASE weights with the WHOLE action-input
      pathway -- `_residual_fwd.action_encoder` AND transition_net[0]'s action-embedding input
      columns, i.e. both the key and the value half of any memorised action->effect table --
      restored to those INIT weights; state columns, first-layer bias and output layer kept).
      DV: the TRANSFER INDEX against the same-budget scratch control,
          T_full    = gap_vs_shuffled(READAPT@B0) - gap_vs_shuffled(SCRATCH@B0)
          T_partial = gap_vs_shuffled(PARTIAL@B0) - gap_vs_shuffled(SCRATCH@B0)
      where gap_vs_shuffled = MOVE_OK AUROC of the residual minus the mean over
      N_SHUFFLE_DRAWS (5, paired across cells) action-permutation draws (F5).
      WHY TWO (red-team F3): the shift is a derangement, so READAPT must first UNLEARN four
      wrong action->effect entries -- T_full nets a positive non-action transfer against that
      maximal negative one, and a null on T_full alone is over-read as 'nothing reusable
      carried'. PARTIAL discards every weight that can hold an action->effect entry and keeps
      only the state pathway and the output layer; if action-conditioning re-derives faster
      from that than from a fresh head, reusable non-lookup structure was carried (pass-2 N1
      closed the first draft's version, which had kept the value half of the table).
      DECLARED NULL (H1 STANDS): BOTH mean(T_full) AND mean(T_partial) < max(TRANSFER_ABS_FLOOR,
      1 sd). H1 REFUTED: either clears its requirement (both load-bearing; combination = OR).
      The UNSHIFTED-env arm is kept ONLY as the asymptote row: PRE_BASE's own in-distribution
      gap_vs_shuffled (1001 measured ~0.119) is what a full re-derivation converges to; a
      readapted unshifted arm has nothing to recover and cannot refute H1 (autopsy F1).
      T at the ladder's largest cell is recorded as a SECONDARY readout (F1 note).

  H2 / H-budget (axis: curriculum).  A readapt BUDGET LADDER: LADDER_EPOCHS x
      LADDER_POOL_FRACTIONS of the post-shift readapt pool, for BOTH READAPT and SCRATCH (36
      head trainings per seed), each scored on the same held-out set. Budget is recorded in
      REALISED OPTIMIZER STEPS (autopsy learning 2: never epochs across differently-sized
      datasets). THE LOAD-BEARING AXIS IS THE FULL-POOL COLUMN ONLY (red-team F5): data fixed
      at the whole 12000-transition pool, epochs {1,2,8,30,60,140} -> 47..6580 steps (140x; the
      140-epoch cell matches PRE_BASE's own ~6600-step optimizer budget on 43% of its data),
      so steps and data count are not confounded and small-pool overfitting cannot attenuate
      the slope. DV: per-seed OLS slope of gap_vs_shuffled(READAPT, full-pool column) on
      log10(optimizer steps). The pooled 18-cell slope and the per-column / per-row
      decomposition are RECORDED (descriptive).
      DECLARED NULL (H2 REFUTED): mean(slope) < max(SLOPE_ABS_FLOOR, 1 sd) -- flat over a
      >100x step range at fixed data. H2 SUPPORTED: mean(slope) >= max(SLOPE_ABS_FLOOR, 1 sd).
      POSITIVE CONTROL (red-team F4): the SCRATCH full-pool column is the instrument's own
      learning curve under the relabelling symmetry; `dv_headroom_budget_slope` requires its
      slope to reach 2x SLOPE_ABS_FLOOR, else the H2 leg is NOT READY (a flat READAPT slope
      would then be instrument starvation, not 'no scaling'). PRE_BASE additionally records
      its per-epoch held-out AUROC/MSE curve, so where the in-distribution gap emerges on the
      step axis is measured. Secondary (recorded, not load-bearing): whether the largest cell's
      gap_vs_shuffled crosses 1001's 0.05 action floor, and the transfer curve T(steps)
      (autopsy learning 5).

  H3 / H-testbed-degenerate (axis: environment).  PRE-REGISTERED AS UNTESTABLE IN THIS RUN.
      The chip permits "an environment leg whose action->effect structure is richer than a
      4-entry lookup, OR, if none is available, a pre-registered statement that H3 is
      untestable in CausalGridWorldV2 and the run reads only H1/H2". Verified at authoring
      (2026-09-08): CausalGridWorldV2 resolves displacement as `dx, dy = self._action_map[action]`
      (causal_grid_world.py:2384) from a flat, state-independent 5-entry dict; there is no
      terrain, carried-state, or regional conditioning of the map anywhere in the env. A
      driver-side subclass that made the map region-conditioned was designed and REJECTED at
      authoring: for the FROZEN arm its declared null (unshifted-subset gap == shifted-subset
      gap) is unreachable by construction -- the frozen weights see the same input distribution
      on the unshifted subset as pre-shift, so parity is only reachable if the shifted subset
      also did not degrade, which is an action-blindness reading, not a compositionality one --
      exactly the "criterion cannot discriminate by construction" family. H3 therefore stays
      ALIVE in the registry, is NOT adjudicated here, and the manifest records what an
      H3-capable environment needs (`h3_testbed_degenerate`). The H1 x H2 verdict grid is
      what routes it: H1-stands AND H2-supported is the H3-ambiguous cell (a 4-entry table
      re-fits cheaply from either initialisation) and reads `mixed`, never `weakens`.

VERDICT GRID -> SD-031 (evidence; the ONLY claim this run exercises; claim_ids = [SD-031]):
      H1 refuted  AND H2 supported -> PASS, supports:
          causal_signature_rederivable_positive_transfer_budget_scaling
      H1 stands   AND H2 refuted   -> FAIL, weakens:
          causal_signature_shortcut_no_transfer_no_scaling  (H3 does NOT predict this cell:
          a re-fittable lookup table would SCALE; flat + no transfer is the shortcut fingerprint)
      H1 stands   AND H2 supported -> FAIL, mixed:
          refit_scales_without_transfer_testbed_open  (H3 remains live; env build is next)
      H1 refuted  AND H2 refuted   -> FAIL, mixed:
          transfer_without_budget_scaling  (report; unexpected)
      Either leg's gate RED -> that leg is UNADJUDICATED (criteria_non_degenerate False for its
          criterion, never a verdict); both RED -> substrate_not_ready_requeue, non_contributory.
      Combination rule recorded in the manifest. EXT-005 is NOT tagged: no V3 run observes an
      LLM (1001's attribution_caveat), and the chip instructs tagging it only if the run
      observes something about the LLM-side assertion, which it does not.

WHAT IS CARRIED VERBATIM FROM 1001 (so the two runs are comparable): the construction-balanced
RandomPolicy collection (uniform over the action set, so a permutation of the movement actions
holds every surface marginal fixed in expectation -- measured as `surface_statistics_preserved`),
the shift definition as a DERANGEMENT of the 4 movement actions with STAY untouched, the
ground-truth-from-coordinates exogenous-change label, the STAY/MOVE_OK/MOVE_BLOCKED
stratification, world_dim 128 / alpha_world 0.9, the P0a encoder warmup (SD-070 guard), the
PRE budget (250 blocks x 160 = 40000 transitions, 60 epochs), the POST collection size (24000),
every one of 1001's readiness preconditions, and the triple-gate bookkeeping (gap_vs_bare,
gap_vs_chance, gap_vs_shuffled all recorded per cell; gap_vs_shuffled is LOAD-BEARING here and
gap_vs_chance is REPORTED against 1001's 0.05 floor, per the autopsy's F4 instruction).

WHAT CHANGES, AND WHY EACH CHANGE IS LICENSED BY THE AUTOPSY:
  - The manipulation is n=5 not n=1 on the permutation axis (learning 8): each seed applies a
    DIFFERENT derangement of the 4 movement actions (9 exist; seed index selects one). All
    four actions change effect in every seed (`action_map_genuinely_rewired`).
  - The post-shift split is 50% readapt POOL / 50% held-out (1001: 15/85), so the ladder's
    largest cell has 12000 transitions while B0 keeps 1001's exact 3600 x 8 epochs = 120 steps.
  - N_SHUFFLE_DRAWS = 5 permutation draws per evaluated cell (F5); the draw sd and the
    standard error of the shuffled mean are recorded, and `shuffle_draw_se_resolves_floor`
    requires SE <= TRANSFER_ABS_FLOOR in the worst B0 cell so the H1 floor is resolvable.
  - Per-epoch held-out AUROC + forward-MSE recovery curves are recorded for READAPT@B0,
    READAPT@max and SCRATCH@max (learning 5) -- a recorded readout, not an endpoint inference.

PRE-REGISTERED FLOORS AND THEIR HEADROOM (dv_headroom declared; W5 campaign rule). Every floor
is denominated on the DV's OWN reachable range at the budget the criterion reads -- never on the
6600-step asymptote (red-team F1/F2 found the first draft's 0.02 floor sat ABOVE the 0.0078-0.0189
readapt gap 1001 measured at this exact budget, making `supports` unreachable):
  TRANSFER_ABS_FLOOR = 0.005 -- the paired-draw precision anchor (1001's single-draw shuffle
      noise 0.0103 / sqrt(5)); the noise-aware clause max(abs, 1 sd across seeds) dominates.
      `dv_headroom_transfer_index` certifies READAPT@B0's own gap_vs_shuffled (which bounds T,
      since SCRATCH@B0 ~ 0) clears this floor in the WORST seed, signed (floor_headroom, pass-2
      N2) -- with 1001's numbers min 0.0078 vs 0.005, met.
  SLOPE_ABS_FLOOR = 0.01 per decade of optimizer steps. `dv_headroom_budget_slope` certifies
      the SCRATCH full-pool column's own slope clears this floor in the WORST seed, signed (the
      H2 positive control; a declining column cannot certify).
  Both floors are also noise-aware (max(abs, 1 sd)).

REGIME-CONDITIONED GATES (experiments/_lib/precondition_gate; the 785 rule): H1 and H2 are the
two arms. Shared readiness preconditions (1001's set) apply to both; `scratch_control_trained`,
`readapt_does_not_regress_frozen_fit`, `scratch_control_trained`, `partial_control_trained`
(each measured on the cell's OWN entry weights, red-team F6), `scratch_entry_is_shared_init`,
`shuffle_draw_se_resolves_floor` and `dv_headroom_transfer_index` apply to H1;
`ladder_step_range_spans_100x` (full-pool column) and `dv_headroom_budget_slope` apply to H2.
A red leg never vacates the other.

DV-SYMMETRY INVARIANCE (per verdict-bearing arm). H1's DV is a difference of two
gap_vs_shuffled AUROCs (rank statistics of residual norms within the held-out MOVE_OK stratum);
symmetry group: monotone rescalings of the residual norm and permutations within a label class.
The manipulation (initial weights: PRE_BASE-trained vs fresh) is neither -- it changes the
residual as a function of (z_prev, a) and so re-ranks steps across the drift/no-drift boundary.
H2's DV is a slope of that gap over log-budget; the manipulation (optimizer steps and data
count) is not a broadcast constant, a monotone rescaling of the residual, or a permutation.

GOV-REUSE-1 (Step 2.4): decisive readouts are (i) a SCRATCH-initialised same-budget
gap_vs_shuffled and (ii) gap_vs_shuffled as a function of realised optimizer steps under an
action-map rewiring. `reanalysis_query.py query --readout auroc_shuffled_move_ok --claim SD-031`
finds exactly two manifests with the readout family (V3-EXQ-995 in-distribution only; V3-EXQ-1001
one budget, one initialisation, one shuffle draw). Neither carries a scratch arm or a budget
ladder -> NOT RECOVERABLE -> run.

SUBSTRATE READINESS (Step 2.5/2.5a): identical construction to 1001 (E2WorldForward live;
`_action_map` instance override probed live by 1001 and re-probed at this authoring). Re-derive
brake (2.5b): SD-031 has 0 ceiling hits (autopsy re_derive_brake.counts_measured); this is a NEW
number on a different design axis in any case. Substrate-path overlap (2.5c): footprint identical
to 1001's; the two open CORRUPTING entries (mode-governance-engagement, SD-082) are gated off on
this exact REEConfig.from_dims(...) construction (salience_affinity_input_cap=None,
use_lateral_pfc_analog=False); open DEGRADING entries recorded in known_substrate_limitations.

red-team (Step 4.5, DIFFERENT model, model=fable, 2026-09-08). PASS 1 verdict BLOCKING on the
first draft, nine findings, all verified against the source and dispositioned: F1 (BLOCKING,
H1 floor 0.02 above the 0.0189 readapt-gap ceiling at B0) -> floor re-anchored to paired-draw
precision 0.005 + 1 sd; F2 (headroom read the 6600-step asymptote) -> both dv_headroom entries
now read the DV's own range at the evaluated budget; F3 (unlearning cost confounds the null)
-> PARTIAL start added as a second load-bearing H1 criterion, OR-combined; F4 (H2 null had no
positive control; (stands, refuted)->weakens unattributable when SCRATCH is also flat) -> the
SCRATCH full-pool slope is the H2 leg's dv_headroom gate, PRE_BASE records its per-epoch curve;
F5 (pooled OLS confounds steps with data; small-pool overfitting attenuates it) -> load-bearing
slope is the full-pool column, ladder max raised to 140 epochs (6580 steps, 140x), pooled and
per-column/per-row slopes recorded; F6 (entry-weights eval drew a second random head) -> entry
evaluation runs on the same model object before its first step; F7 (init identity unverified)
-> shared init carried as an object, hash recorded, `scratch_entry_is_shared_init` gate; F8
(paired shuffle draws) -> recorded in `summary.shuffle_draws_note`; F9 inherits F1. PASS 2
(fresh session, same model, re-spawned once because F1/F3/F4/F5 changed the causal chain)
verdict CONTESTED: F1/F2/F4/F5/F6/F7/F9 verified CLOSED, F8 DISMISSED with citation, F3
re-opened as N1 (PARTIAL had kept transition_net[0]'s action-embedding input columns, the
value half of the table, so T_partial > 0 was predicted under H-shortcut too) -> PARTIAL now
restores those columns as well; N2 (both dv_headroom gates used max_abs over seeds, so a
negative single-seed value could certify) -> floor_headroom, signed, worst seed, margin 1.0;
N3 (docstring said SE <= floor/2, code floor) -> docstring corrected; N4 (RNG/explicit-start
interaction, the 140-epoch cell, grid directions) checked clean. No third pass (the skill
forbids iterating to CLEAR); dispositions above are the record.
"""

import copy
import math
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import argparse
import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from experiment_protocol import emit_outcome  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402
from experiments._lib.arm_fingerprint import arm_cell, seeded_construct  # noqa: E402
from experiments._lib.manifest_core import stamp_recording_core  # noqa: E402
from experiments._lib.z_goal_stream import ZGoalStreamAccumulator  # noqa: E402
from experiments._lib.zworld_p0_warmup import run_zworld_p0  # noqa: E402
from experiments._lib.capability_eval import RandomPolicy  # noqa: E402
from experiments._lib.zworld_encoder_guard import (  # noqa: E402
    latent_stack_snapshot,
    assert_world_encoder_trained,
    zworld_precondition,
)
from experiments._lib.precondition_gate import (  # noqa: E402
    PreconditionSpec,
    aggregate_arm_gates,
    arm_criteria_non_degenerate,
    evaluate_arm_gate,
)
from experiments._metrics import check_degeneracy, dv_headroom_check  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.predictors.e2_world import E2WorldForward, E2WorldConfig  # noqa: E402

EXPERIMENT_PURPOSE = "evidence"
EXPERIMENT_TYPE = "v3_exq_1013_sd031_shortcut_vs_model_portfolio"
QUEUE_ID = "V3-EXQ-1013"
SUPERSEDES = "V3-EXQ-1001"
CLAIM_IDS = ["SD-031"]
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"
REGISTRY_QUESTION = "sd031_causal_signature_shortcut_vs_model"

# ---------------------------------------------------------------- configuration
SEEDS: List[int] = [43, 45, 46, 47, 48]   # identical to 995/1001; 42 = authoring pilot
WORLD_DIM = 128
ALPHA_WORLD = 0.9

ENV_KWARGS: Dict[str, Any] = dict(
    size=8,
    num_hazards=3,
    num_resources=3,
    contamination_spread=0.0,
    env_drift_interval=2,
    env_drift_prob=1.0,
    use_proxy_fields=True,
)

# PRE-shift phase (P0a + P1): identical budget to 995/1001.
EPISODES_PER_PHASE = 250
P0A_STEPS_PER_EPISODE = 100
TRANSITIONS_PER_BLOCK = 160
N_TRANSITIONS_PRE = EPISODES_PER_PHASE * TRANSITIONS_PER_BLOCK   # 40000
HELDOUT_FRACTION_PRE = 0.30
HEAD_EPOCHS = 60
HEAD_BATCH = 256
HEAD_LR = 3e-4

# POST-shift phase: same collection size as 1001; split changed to 50/50 so the ladder's
# largest cell has a real pool while B0 keeps 1001's exact 3600-transition x 8-epoch budget.
POST_EPISODES_PER_PHASE = 150
N_TRANSITIONS_POST = POST_EPISODES_PER_PHASE * TRANSITIONS_PER_BLOCK   # 24000
READAPT_POOL_FRACTION = 0.50        # 12000 pool / 12000 held-out at full scale

# H2 budget ladder (epochs x fraction of the readapt POOL), applied to BOTH initialisations.
LADDER_EPOCHS: List[int] = [1, 2, 8, 30, 60, 140]
LADDER_POOL_FRACTIONS: List[float] = [0.05, 0.30, 1.00]
# B0 = 1001's discriminating-control budget: 0.30 x 12000 = 3600 transitions, 8 epochs,
# ceil(3600/256) = 15 steps/epoch -> 120 optimizer steps. Identical arithmetic to 1001.
B0_EPOCHS = 8
B0_POOL_FRACTION = 0.30
LADDER_MAX_EPOCHS = max(LADDER_EPOCHS)
LADDER_MAX_FRACTION = max(LADDER_POOL_FRACTIONS)
INITS: List[str] = ["READAPT", "SCRATCH"]   # the ladder; PARTIAL is trained at B0 only
B0_INITS: List[str] = ["READAPT", "SCRATCH", "PARTIAL"]
FULL_POOL_FRACTION = 1.00   # the H2 load-bearing column: data FIXED at the full pool, steps vary with epochs only

N_SHUFFLE_DRAWS = 5                  # autopsy F5: >= 5 permutation draws per evaluated cell
CURVE_CELLS = {("READAPT", B0_EPOCHS, B0_POOL_FRACTION),
               ("READAPT", LADDER_MAX_EPOCHS, LADDER_MAX_FRACTION),
               ("SCRATCH", LADDER_MAX_EPOCHS, LADDER_MAX_FRACTION)}

STAY_ACTION = 4   # CausalGridWorldV2.ACTIONS[4] == (0, 0); never touched by the derangement
# The 9 derangements of 4 movement actions (index i -> which canonical action's effect
# action i inherits). Seed index selects one, so the manipulation is n=len(SEEDS), not n=1.
DERANGEMENTS: List[Tuple[int, int, int, int]] = [
    (1, 0, 3, 2), (1, 2, 3, 0), (1, 3, 0, 2),
    (2, 0, 3, 1), (2, 3, 0, 1), (2, 3, 1, 0),
    (3, 0, 1, 2), (3, 2, 0, 1), (3, 2, 1, 0),
]

# ------------------------------------------------------- pre-registered thresholds
TRANSFER_ABS_FLOOR = 0.005     # H1: the paired-draw precision anchor (1001 single-draw noise 0.0103 / sqrt(5))
TRANSFER_SD_MULT = 1.0
SLOPE_ABS_FLOOR = 0.01         # H2: gap_vs_shuffled per decade of optimizer steps
SLOPE_SD_MULT = 1.0
HEADROOM_MARGIN = 1.0          # applied to the WORST seed's SIGNED value (floor_headroom, pass-2 N2):
                               # stricter than a 2x margin on an unsigned best seed
THRESH_ACTION_ABS_FLOOR = 0.05  # 1001's gap-vs-shuffled floor -- SECONDARY crossing readout
THRESH_CHANCE_ABS_FLOOR = 0.05  # 1001's gap-vs-chance floor -- REPORTED, not load-bearing
THRESH_PRE_BASE_ABS_FLOOR = 0.05
THRESH_PRE_BASE_SD_MULT = 1.0
READAPT_RECOVERY_ABS_FLOOR = 1e-5
READAPT_RECOVERY_FRAC = 0.05
LADDER_STEP_RANGE_FLOOR = 100.0   # realised max/min optimizer steps on the FULL-POOL column must span >= 100x

# -------------------------------------------------------- readiness preconditions
FLOOR_STAY_BARE_AUROC = 0.80
BAND_MOVE_OK_BARE_LOW = 0.40
BAND_MOVE_OK_BARE_HIGH = 0.60
FLOOR_MOVE_OK_CLASS_N = 500
FLOOR_ACTION_PATHWAY = 1e-4
FLOOR_ACTIONS_REWIRED = 4
CEIL_SURFACE_STAT_DEVIATION = 0.05

H3_TESTBED_DEGENERATE = {
    "hid": "H-testbed-degenerate",
    "status": "untestable_in_causalgridworldv2_pre_registered",
    "adjudicated_by_this_run": False,
    "reason": ("CausalGridWorldV2 resolves displacement from a flat, state-independent "
               "_action_map dict (causal_grid_world.py:2384); no terrain, carried-state or "
               "regional conditioning of the action->effect map exists, so a subset-of-structure "
               "shift cannot be instantiated. A driver-side region-conditioned subclass was "
               "designed and rejected: the frozen arm's declared null (unshifted-subset gap == "
               "shifted-subset gap) is unreachable by construction because frozen weights see an "
               "unchanged input distribution on the unshifted subset."),
    "what_would_test_it": ("An environment (substrate build, not a driver hack) whose "
                           "action->effect mapping is state-dependent or compositional (e.g. "
                           "displacement conditioned on terrain or a carried state variable), "
                           "with an API to shift a SUBSET of that structure, plus a readiness "
                           "precondition that the conditioning variable is decodable from "
                           "z_world. Route via /implement-substrate ONLY if this run's H1-stands "
                           "AND H2-supported cell fires (the H3-ambiguous cell); otherwise H3 is "
                           "moot (H1 refuted) or dominated (shortcut fingerprint)."),
}


# --------------------------------------------------------------------- utilities
def _auroc(scores: np.ndarray, labels: np.ndarray) -> Optional[float]:
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=bool)
    n1 = int(y.sum())
    n0 = int((~y).sum())
    if n1 == 0 or n0 == 0:
        return None
    order = s.argsort(kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    ranks[order] = np.arange(1, len(s) + 1, dtype=float)
    _, inv, counts = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts), dtype=float)
    np.add.at(sums, inv, ranks)
    ranks = (sums / counts)[inv]
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def _mean_sd(values: List[Optional[float]]) -> Tuple[float, float]:
    arr = np.asarray([v for v in values if v is not None and v == v], dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan")
    sd = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    return float(arr.mean()), sd


def _worst_low(values: List[Tuple[Optional[float], Any]]) -> Tuple[Optional[float], Any]:
    vals = [(v, k) for v, k in values if v is not None and v == v]
    if not vals:
        return None, None
    v, k = min(vals, key=lambda t: t[0])
    return float(v), k


def _worst_interval(values: List[Tuple[Optional[float], Any]], centre: float) -> Tuple[Optional[float], Any]:
    vals = [(v, k) for v, k in values if v is not None and v == v]
    if not vals:
        return None, None
    v, k = max(vals, key=lambda t: abs(t[0] - centre))
    return float(v), k


def _ols_slope(xs: List[float], ys: List[float]) -> Optional[float]:
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return None
    x, y = x[ok], y[ok]
    vx = float(((x - x.mean()) ** 2).sum())
    if vx <= 0.0:
        return None
    return float(((x - x.mean()) * (y - y.mean())).sum() / vx)


def _cell_id(init: str, epochs: int, frac: float) -> str:
    return "%s_e%d_f%d" % (init, epochs, int(round(frac * 100)))


def _build_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(seed=seed, **ENV_KWARGS)


def _build_agent(env: CausalGridWorldV2) -> REEAgent:
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        world_dim=WORLD_DIM,
        alpha_world=ALPHA_WORLD,
    )
    return REEAgent(cfg)


def _encode_world(split_encoder, obs_dict) -> torch.Tensor:
    body = obs_dict["body_state"].float()
    world = obs_dict["world_state"].float()
    if body.dim() == 1:
        body = body.unsqueeze(0)
    if world.dim() == 1:
        world = world.unsqueeze(0)
    with torch.no_grad():
        return split_encoder(body, world)[1]


def _shift_action_map(env: CausalGridWorldV2, derangement: Tuple[int, ...]) -> Dict[str, Any]:
    """Overwrite env._action_map so movement action move_keys[i] produces the canonical
    effect of move_keys[derangement[i]]. A derangement has no fixed point, so all four
    movement actions change effect; STAY_ACTION is never touched. Applied ONCE."""
    canonical = dict(env._action_map)
    move_keys = sorted(k for k in canonical if k != STAY_ACTION)
    assert len(move_keys) == len(derangement), "derangement arity must match movement actions"
    shifted = dict(canonical)
    for i, k in enumerate(move_keys):
        shifted[k] = canonical[move_keys[derangement[i]]]
    env._action_map = shifted
    n_rewired = sum(1 for k in move_keys if shifted[k] != canonical[k])
    assert shifted[STAY_ACTION] == (0, 0), "STAY_ACTION must never be rewired"
    return {
        "canonical_action_map": {str(k): list(v) for k, v in canonical.items()},
        "shifted_action_map": {str(k): list(v) for k, v in shifted.items()},
        "n_movement_actions": len(move_keys),
        "n_actions_rewired": int(n_rewired),
        "derangement": list(derangement),
    }


def _build_shifted_env(seed: int, derangement: Tuple[int, ...]) -> Tuple[CausalGridWorldV2, Dict[str, Any]]:
    env = _build_env(seed)
    record = _shift_action_map(env, derangement)
    return env, record


# ------------------------------------------------------------------ P0a + collection
def _run_p0a(agent: REEAgent, seed: int, episodes: int, steps: int,
             dry_run: bool) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    print("Seed %d Condition P0A_ENCODER" % seed, flush=True)
    before = latent_stack_snapshot(agent)
    warm_env = _build_env(seed)
    stats = run_zworld_p0(
        agent, warm_env, seed, episodes, steps,
        policy=RandomPolicy(seed), label="sd031portfolio", dry_run=dry_run,
    )
    guard = assert_world_encoder_trained(
        agent, before, p0=episodes, strict=not dry_run,
        context="V3-EXQ-1013 SD-031 shortcut-vs-model portfolio",
        escape_hint="P0a must train split_encoder.world_encoder; a frozen random "
                    "projection makes the comparator vacuous (MECH-353 / V3-EXQ-642).",
    )
    print("verdict: PASS", flush=True)
    return stats, guard


def _collect_transitions(agent: REEAgent, env: CausalGridWorldV2, seed: int,
                          blocks: int, per_block: int, phase_label: str) -> Dict[str, Any]:
    """(z_prev, action, z_obs) transitions under a uniform RandomPolicy on `env`. Ground truth
    from agent + hazard COORDINATES, never info["env_drift_occurred"] (995's finding)."""
    print("Seed %d Condition %s" % (seed, phase_label), flush=True)
    split_encoder = agent.latent_stack.split_encoder
    rng = random.Random(seed + (900 if phase_label == "P1_COLLECT" else 1900))
    _flat, obs = env.reset()

    z_prev_l: List[torch.Tensor] = []
    act_l: List[torch.Tensor] = []
    z_obs_l: List[torch.Tensor] = []
    cls_l: List[str] = []
    drift_l: List[bool] = []
    flag_l: List[bool] = []
    n_flag_disagree = 0
    n_boundary_dropped = 0
    action_counts = [0] * env.action_dim

    for block in range(blocks):
        got = 0
        while got < per_block:
            z_prev = _encode_world(split_encoder, obs)
            ax, ay = env.agent_x, env.agent_y
            hz_before = [tuple(h[:2]) for h in env.hazards]
            action = rng.randrange(env.action_dim)
            _flat, _harm, done, info, obs = env.step(action)
            z_obs = _encode_world(split_encoder, obs)

            agent_moved = (env.agent_x, env.agent_y) != (ax, ay)
            hazards_moved = [tuple(h[:2]) for h in env.hazards] != hz_before
            drift_flag = bool(info.get("env_drift_occurred", False))
            if drift_flag != hazards_moved:
                n_flag_disagree += 1

            if action == STAY_ACTION:
                cls = "STAY"
            elif agent_moved:
                cls = "MOVE_OK"
            else:
                cls = "MOVE_BLOCKED"

            one_hot = torch.zeros(1, env.action_dim)
            one_hot[0, action] = 1.0
            z_prev_l.append(z_prev)
            act_l.append(one_hot)
            z_obs_l.append(z_obs)
            cls_l.append(cls)
            drift_l.append(bool(hazards_moved))
            flag_l.append(drift_flag)
            action_counts[action] += 1
            got += 1

            if done:
                _flat, obs = env.reset()
                n_boundary_dropped += 1

        cur = block + 1
        if cur == 1 or cur % 25 == 0 or cur == blocks:
            print("  [train] sd031portfolio seed=%d phase=%s ep %d/%d (transition collection)"
                  % (seed, phase_label, cur, blocks), flush=True)

    return {
        "z_prev": torch.cat(z_prev_l),
        "action": torch.cat(act_l),
        "z_obs": torch.cat(z_obs_l),
        "cls": np.array(cls_l),
        "drift": np.array(drift_l, dtype=bool),
        "flag": np.array(flag_l, dtype=bool),
        "n_flag_disagree": int(n_flag_disagree),
        "n_boundary_resets": int(n_boundary_dropped),
        "action_counts": action_counts,
        "n_transitions": len(cls_l),
    }


def _class_fractions(buf: Dict[str, Any]) -> Dict[str, float]:
    out = {c: float((buf["cls"] == c).mean()) for c in ("STAY", "MOVE_OK", "MOVE_BLOCKED")}
    out["drift_fraction"] = float(buf["drift"].mean())
    return out


# ------------------------------------------------------------------- training / eval
def _action_pathway_probe(model: E2WorldForward, z_prev: torch.Tensor,
                          action_dim: int) -> float:
    n = min(256, z_prev.shape[0])
    a_i = torch.zeros(n, action_dim)
    a_i[:, 0] = 1.0
    a_j = torch.zeros(n, action_dim)
    a_j[:, min(2, action_dim - 1)] = 1.0
    with torch.no_grad():
        return float((model(z_prev[:n], a_i) - model(z_prev[:n], a_j)).norm(dim=-1).mean())


def _fresh_head(action_dim: int) -> E2WorldForward:
    return E2WorldForward(E2WorldConfig(
        use_e2_world_forward=True, z_world_dim=WORLD_DIM, action_dim=action_dim,
    ))


def _state_hash(model: torch.nn.Module) -> str:
    import hashlib
    h = hashlib.sha256()
    for k, v in model.state_dict().items():
        h.update(k.encode())
        h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()[:16]


def _partial_init(pre_model: E2WorldForward, init_model: E2WorldForward) -> E2WorldForward:
    """PARTIAL start (red-team F3, tightened by pass-2 N1): the trained PRE_BASE weights with
    the WHOLE action-input pathway -- `_residual_fwd.action_encoder` (the key half of any
    memorised action->effect table) AND transition_net[0]'s 16 action-embedding input columns
    (the value half) -- restored to the shared INIT weights; the state columns, first-layer
    bias and output layer are kept. Isolates 'reusable NON-action structure is carried' from
    'four wrong entries must be unlearned first': every weight that can hold an action->effect
    entry is discarded, so positive transfer here can only come from the state pathway."""
    m = copy.deepcopy(pre_model)
    src = init_model._residual_fwd.action_encoder.state_dict()
    m._residual_fwd.action_encoder.load_state_dict(copy.deepcopy(src))
    # Red-team pass 2 N1: the action->effect table has a KEY half (action_encoder) and a
    # VALUE half -- the 16 input columns of transition_net[0] that read the action embedding
    # (forward concatenates [z_world, action_enc], stack.py ResidualHarmForward.forward, so
    # those are columns WORLD_DIM: of the first Linear's weight). Restore BOTH halves to
    # init; keep the state columns [:WORLD_DIM], the first-layer bias and the output layer.
    with torch.no_grad():
        w_tr = m._residual_fwd.transition_net[0].weight
        w_in = init_model._residual_fwd.transition_net[0].weight
        assert w_tr.shape == w_in.shape and w_tr.shape[1] == WORLD_DIM + 16, w_tr.shape
        w_tr[:, WORLD_DIM:] = w_in[:, WORLD_DIM:].clone()
    return m


def _train_head(z_prev: torch.Tensor, action: torch.Tensor, z_obs: torch.Tensor,
                 n_train: int, action_dim: int, seed: int, epochs: int, lr: float,
                 start_from: Optional[E2WorldForward] = None,
                 epoch_eval: Optional[Callable[[E2WorldForward], Dict[str, Any]]] = None,
                 entry_eval: Optional[Callable[[E2WorldForward], Dict[str, Any]]] = None,
                 ) -> Tuple[E2WorldForward, List[Dict[str, Any]], float, int, Dict[str, Any]]:
    """Train a deepcopy of `start_from` (never None here: every start -- fresh init,
    PRE_BASE, PARTIAL -- is an explicit model object, so 'what weights did this cell start
    from' is a recorded fact, not an RNG-order inference; red-team F6/F7) on the first
    n_train rows. Returns (model, curve, wiring_at_entry, realised_optimizer_steps,
    entry_record). `entry_eval` runs on the ENTRY weights before any step (the F6 fix:
    the entry evaluation is of THIS model, not a second draw); `epoch_eval` after every
    epoch, appended to the curve."""
    assert start_from is not None, "every start must be an explicit model object"
    model = copy.deepcopy(start_from)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    gen = torch.Generator().manual_seed(int(seed))
    curve: List[Dict[str, Any]] = []
    wiring_at_entry = _action_pathway_probe(model, z_prev, action_dim)
    entry_record: Dict[str, Any] = {"entry_state_hash": _state_hash(model)}
    if entry_eval is not None:
        entry_record.update(entry_eval(model))
    steps = 0
    for epoch in range(epochs):
        perm = torch.randperm(n_train, generator=gen)
        last_loss = float("nan")
        for i in range(0, n_train, HEAD_BATCH):
            idx = perm[i:i + HEAD_BATCH]
            loss = model.compute_loss(model(z_prev[idx], action[idx]), z_obs[idx].detach())
            opt.zero_grad()
            loss.backward()
            opt.step()
            last_loss = float(loss.detach())
            steps += 1
        if epoch_eval is not None:
            row = {"epoch": epoch + 1, "optimizer_steps": steps, "train_loss": last_loss}
            row.update(epoch_eval(model))
            curve.append(row)
        elif (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            curve.append({"epoch": epoch + 1, "optimizer_steps": steps, "train_loss": last_loss})
    return model, curve, wiring_at_entry, steps, entry_record


def _move_ok_auroc_block(model: E2WorldForward, z_prev: torch.Tensor, action: torch.Tensor,
                          z_obs: torch.Tensor, cls: np.ndarray, drift: np.ndarray,
                          seed_for_shuffle: int, n_draws: int = N_SHUFFLE_DRAWS) -> Dict[str, Any]:
    """Per-stratum AUROC block (comparator / bare / shuffled-action over n_draws permutation
    draws) on a fixed evaluation set. gap_vs_shuffled uses the MEAN over draws (F5)."""
    n = z_prev.shape[0]
    gen = torch.Generator().manual_seed(int(seed_for_shuffle) + 77)
    with torch.no_grad():
        res = model.comparator_residual(z_obs, z_prev, action).norm(dim=-1).numpy()
        res_shuf_draws = []
        for _d in range(n_draws):
            perm = torch.randperm(n, generator=gen)
            res_shuf_draws.append(
                model.comparator_residual(z_obs, z_prev, action[perm]).norm(dim=-1).numpy())
        fwd_mse = float(torch.nn.functional.mse_loss(model(z_prev, action), z_obs))
    bare = (z_obs - z_prev).norm(dim=-1).numpy()

    per_stratum: Dict[str, Any] = {}
    for c in ("STAY", "MOVE_OK", "MOVE_BLOCKED"):
        sel = (cls == c)
        n1 = int(drift[sel].sum())
        n0 = int((~drift[sel]).sum())
        shuf = [_auroc(r[sel], drift[sel]) for r in res_shuf_draws]
        shuf_mean, shuf_sd = _mean_sd(shuf)
        per_stratum[c] = {
            "n_drift": n1, "n_no_drift": n0,
            "auroc_comparator": _auroc(res[sel], drift[sel]),
            "auroc_bare_delta": _auroc(bare[sel], drift[sel]),
            "auroc_shuffled_action": (shuf_mean if shuf_mean == shuf_mean else None),
            "auroc_shuffled_action_draws": shuf,
            "auroc_shuffled_action_draw_sd": shuf_sd,
            "verdict_bearing": c == "MOVE_OK",
        }
    mo = per_stratum["MOVE_OK"]
    comp = mo["auroc_comparator"]
    shuf_mean = mo["auroc_shuffled_action"]
    n_ok = sum(1 for v in mo["auroc_shuffled_action_draws"] if v is not None)
    return {
        "auroc_move_ok": comp,
        "auroc_bare_move_ok": mo["auroc_bare_delta"],
        "auroc_shuffled_move_ok": shuf_mean,
        "auroc_shuffled_move_ok_draw_sd": mo["auroc_shuffled_action_draw_sd"],
        "auroc_shuffled_move_ok_se": (mo["auroc_shuffled_action_draw_sd"] / math.sqrt(n_ok)
                                      if n_ok > 0 and mo["auroc_shuffled_action_draw_sd"] == mo["auroc_shuffled_action_draw_sd"]
                                      else None),
        "n_shuffle_draws": n_draws,
        "gap_vs_bare": (comp - mo["auroc_bare_delta"]) if (comp is not None and mo["auroc_bare_delta"] is not None) else None,
        "gap_vs_chance": (comp - 0.5) if comp is not None else None,
        "gap_vs_shuffled": (comp - shuf_mean) if (comp is not None and shuf_mean is not None) else None,
        "heldout_forward_mse": fwd_mse,
        "per_stratum": per_stratum,
    }


def _config_slice(seed: int, arm_id: str) -> Dict[str, Any]:
    return {
        "arm_id": arm_id,
        "env_kwargs": dict(ENV_KWARGS),
        "world_dim": WORLD_DIM,
        "alpha_world": ALPHA_WORLD,
        "episodes_per_phase": EPISODES_PER_PHASE,
        "post_episodes_per_phase": POST_EPISODES_PER_PHASE,
        "p0a_steps_per_episode": P0A_STEPS_PER_EPISODE,
        "n_transitions_pre": N_TRANSITIONS_PRE,
        "n_transitions_post": N_TRANSITIONS_POST,
        "readapt_pool_fraction": READAPT_POOL_FRACTION,
        "ladder_epochs": list(LADDER_EPOCHS),
        "ladder_pool_fractions": list(LADDER_POOL_FRACTIONS),
        "head_epochs": HEAD_EPOCHS,
        "head_batch": HEAD_BATCH,
        "head_lr": HEAD_LR,
        "n_shuffle_draws": N_SHUFFLE_DRAWS,
        "seed": int(seed),
    }


_INELIGIBLE = ["shared_p0a_encoder_and_transition_buffer_across_arms",
               "readapt_arms_initialized_from_pre_base_trained_weights"]


def _run_seed(seed: int, seed_index: int, dry_run: bool, zg: ZGoalStreamAccumulator,
              post_episodes: int) -> Dict[str, Any]:
    env_probe = _build_env(seed)
    action_dim = env_probe.action_dim
    agent = seeded_construct(seed, lambda: _build_agent(env_probe))

    episodes = 4 if dry_run else EPISODES_PER_PHASE
    p0a_steps = 30 if dry_run else P0A_STEPS_PER_EPISODE
    pre_blocks = 4 if dry_run else EPISODES_PER_PHASE
    per_block = 90 if dry_run else TRANSITIONS_PER_BLOCK
    post_blocks = 4 if dry_run else post_episodes
    head_epochs = 6 if dry_run else HEAD_EPOCHS

    p0a_stats, guard = _run_p0a(agent, seed, episodes, p0a_steps, dry_run)
    for p in agent.latent_stack.parameters():
        p.requires_grad_(False)

    # ---------------- PRE-shift: identical to 995/1001 (P1_COLLECT + PRE_BASE) ----------
    pre_env = _build_env(seed + 500)
    buf_pre = _collect_transitions(agent, pre_env, seed, pre_blocks, per_block, "P1_COLLECT")
    n_pre = buf_pre["n_transitions"]
    n_train_pre = int((1.0 - HELDOUT_FRACTION_PRE) * n_pre)
    arm_rows: List[Dict[str, Any]] = []

    print("Seed %d Condition PRE_BASE_HEAD" % seed, flush=True)
    held_pre = slice(n_train_pre, n_pre)

    def _eval_pre(model: E2WorldForward, n_draws: int = N_SHUFFLE_DRAWS) -> Dict[str, Any]:
        return _move_ok_auroc_block(
            model, buf_pre["z_prev"][held_pre], buf_pre["action"][held_pre],
            buf_pre["z_obs"][held_pre], buf_pre["cls"][held_pre], buf_pre["drift"][held_pre],
            seed_for_shuffle=seed, n_draws=n_draws)

    with arm_cell(seed, config_slice=_config_slice(seed, "PRE_BASE"),
                  script_path=Path(__file__), config_slice_declared=True,
                  extra_ineligible_reasons=_INELIGIBLE) as cell:
        # The ONE shared init: PRE_BASE trains from it, SCRATCH cells start from a copy of it,
        # PARTIAL restores its action pathway from it. Drawn once, right after the cell's RNG
        # reset, and carried as an OBJECT (red-team F7: identity is recorded, not inferred).
        init_model = _fresh_head(action_dim)
        init_state_hash = _state_hash(init_model)

        def _pre_epoch_eval(m: E2WorldForward) -> Dict[str, Any]:
            e = _eval_pre(m, n_draws=1)
            return {"heldout_auroc_move_ok": e["auroc_move_ok"],
                    "heldout_gap_vs_shuffled_1draw": e["gap_vs_shuffled"],
                    "heldout_forward_mse": e["heldout_forward_mse"]}
        pre_model, pre_curve, pre_wiring_init, pre_steps, pre_entry = _train_head(
            buf_pre["z_prev"], buf_pre["action"], buf_pre["z_obs"],
            n_train_pre, action_dim, seed, head_epochs, HEAD_LR, start_from=init_model,
            epoch_eval=_pre_epoch_eval)
        row_pre: Dict[str, Any] = {
            "arm_id": "PRE_BASE", "seed": int(seed), "verdict_bearing": False,
            "role": "asymptote_row_in_distribution",
            "init": "FRESH", "epochs": head_epochs, "n_train": int(n_train_pre),
            "optimizer_steps": int(pre_steps),
            "log10_optimizer_steps": float(math.log10(max(pre_steps, 1))),
            "init_state_hash": init_state_hash,
            "recovery_curve": pre_curve,   # per-epoch in-distribution learning curve (red-team F4)
            "action_pathway_wiring_at_init": pre_wiring_init,
            "action_pathway_wiring_trained": _action_pathway_probe(
                pre_model, buf_pre["z_prev"], action_dim),
        }
        row_pre.update(_eval_pre(pre_model))
        row_pre.update(pre_entry)
        cell.stamp(row_pre)
    arm_rows.append(row_pre)
    print("  [train] sd031portfolio seed=%d phase=PRE_BASE_HEAD ep %d/%d" % (seed, head_epochs, head_epochs), flush=True)
    pre_class_fracs = _class_fractions(buf_pre)

    # ---------------- POST-shift: derangement of the movement actions, fresh transitions ----
    derangement = DERANGEMENTS[seed_index % len(DERANGEMENTS)]
    post_env, shift_record = _build_shifted_env(seed + 700, derangement)
    buf_post = _collect_transitions(agent, post_env, seed, post_blocks, per_block, "P3_COLLECT")
    n_post = buf_post["n_transitions"]
    n_pool = int(READAPT_POOL_FRACTION * n_post)
    heldout_slice = slice(n_pool, n_post)
    post_class_fracs = _class_fractions(buf_post)
    surface_stat_deviation = max(
        abs(pre_class_fracs[k] - post_class_fracs[k])
        for k in ("STAY", "MOVE_OK", "MOVE_BLOCKED", "drift_fraction"))

    zh, ah, oh = (buf_post["z_prev"][heldout_slice], buf_post["action"][heldout_slice],
                  buf_post["z_obs"][heldout_slice])
    ch, dh = buf_post["cls"][heldout_slice], buf_post["drift"][heldout_slice]

    def _eval(model: E2WorldForward, shuffle_seed: int, n_draws: int = N_SHUFFLE_DRAWS) -> Dict[str, Any]:
        return _move_ok_auroc_block(model, zh, ah, oh, ch, dh, seed_for_shuffle=shuffle_seed, n_draws=n_draws)

    def _entry_eval(m: E2WorldForward) -> Dict[str, Any]:
        e = _eval(m, seed + 1)
        return {"entry_weights_heldout_forward_mse": e["heldout_forward_mse"],
                "entry_weights_gap_vs_shuffled": e["gap_vs_shuffled"],
                "entry_weights_auroc_move_ok": e["auroc_move_ok"]}

    with arm_cell(seed, config_slice=_config_slice(seed, "POST_FROZEN"),
                  script_path=Path(__file__), config_slice_declared=True,
                  extra_ineligible_reasons=_INELIGIBLE) as cell:
        row_frozen: Dict[str, Any] = {
            "arm_id": "POST_FROZEN", "seed": int(seed), "verdict_bearing": False,
            "role": "reference_row_zero_shot", "init": "PRE_BASE", "epochs": 0,
            "n_train": 0, "optimizer_steps": 0,
            "n_pool": int(n_pool), "n_heldout": int(n_post - n_pool),
            "entry_state_hash": _state_hash(pre_model),
        }
        row_frozen.update(_eval(pre_model, seed + 1))
        cell.stamp(row_frozen)
    arm_rows.append(row_frozen)

    starts: Dict[str, E2WorldForward] = {
        "READAPT": pre_model,
        "SCRATCH": init_model,
        "PARTIAL": _partial_init(pre_model, init_model),
    }

    def _train_cell(init: str, epochs: int, frac: float, want_curve: bool) -> Dict[str, Any]:
        cid = _cell_id(init, epochs, frac)
        n_train = max(HEAD_BATCH if not dry_run else 8, int(round(frac * n_pool)))
        n_train = min(n_train, n_pool)
        cell_epochs = max(1, epochs // 4) if dry_run else epochs
        is_b0 = bool(epochs == B0_EPOCHS and abs(frac - B0_POOL_FRACTION) < 1e-9)
        with arm_cell(seed, config_slice=dict(_config_slice(seed, cid),
                                              ladder_epochs_cell=epochs,
                                              ladder_fraction_cell=frac, init=init),
                      script_path=Path(__file__), config_slice_declared=True,
                      extra_ineligible_reasons=_INELIGIBLE) as cell:
            epoch_eval = None
            if want_curve:
                def epoch_eval(m: E2WorldForward) -> Dict[str, Any]:
                    e = _eval(m, seed + 1, n_draws=1)
                    return {"heldout_auroc_move_ok": e["auroc_move_ok"],
                            "heldout_gap_vs_shuffled_1draw": e["gap_vs_shuffled"],
                            "heldout_forward_mse": e["heldout_forward_mse"]}
            model, curve, wiring_entry, steps, entry = _train_head(
                buf_post["z_prev"], buf_post["action"], buf_post["z_obs"],
                n_train, action_dim, seed + 2000 + 17 * epochs + int(frac * 100),
                cell_epochs, HEAD_LR, start_from=starts[init],
                epoch_eval=epoch_eval, entry_eval=(_entry_eval if is_b0 else None))
            row: Dict[str, Any] = {
                "arm_id": cid, "seed": int(seed), "verdict_bearing": True,
                "role": "ladder_cell" if init in INITS else "b0_partial_control",
                "init": init, "epochs": int(cell_epochs),
                "ladder_epochs": int(epochs), "ladder_pool_fraction": float(frac),
                "full_pool_column": bool(abs(frac - FULL_POOL_FRACTION) < 1e-9),
                "n_train": int(n_train), "optimizer_steps": int(steps),
                "log10_optimizer_steps": float(math.log10(max(steps, 1))),
                "is_b0": is_b0,
                "n_pool": int(n_pool), "n_heldout": int(n_post - n_pool),
                "action_pathway_wiring_at_entry": wiring_entry,
                "recovery_curve": curve,
            }
            row.update(entry)
            row.update(_eval(model, seed + 1))
            cell.stamp(row)
        return row

    # ---------------- H1 + H2: the ladder (READAPT, SCRATCH), same pool, same held-out --------
    ladder_rows: Dict[str, Dict[str, Any]] = {}
    for init in INITS:
        phase = "LADDER_%s" % init
        print("Seed %d Condition %s" % (seed, phase), flush=True)
        n_cells = len(LADDER_EPOCHS) * len(LADDER_POOL_FRACTIONS)
        done_cells = 0
        for epochs in LADDER_EPOCHS:
            for frac in LADDER_POOL_FRACTIONS:
                row = _train_cell(init, epochs, frac, (init, epochs, frac) in CURVE_CELLS)
                arm_rows.append(row)
                ladder_rows[row["arm_id"]] = row
                done_cells += 1
                print("  [train] sd031portfolio seed=%d phase=%s ep %d/%d (cell %s steps=%d gap_vs_shuffled=%s)"
                      % (seed, phase, done_cells, n_cells, row["arm_id"], row["optimizer_steps"],
                         ("%.4f" % row["gap_vs_shuffled"]) if row["gap_vs_shuffled"] is not None else "None"),
                      flush=True)
        print("verdict: PASS", flush=True)

    # ---------------- H1 third start: PARTIAL at B0 only (red-team F3) ------------------------
    print("Seed %d Condition B0_PARTIAL" % seed, flush=True)
    row = _train_cell("PARTIAL", B0_EPOCHS, B0_POOL_FRACTION, False)
    arm_rows.append(row)
    ladder_rows[row["arm_id"]] = row
    print("  [train] sd031portfolio seed=%d phase=B0_PARTIAL ep 1/1 (cell %s steps=%d gap_vs_shuffled=%s)"
          % (seed, row["arm_id"], row["optimizer_steps"],
             ("%.4f" % row["gap_vs_shuffled"]) if row["gap_vs_shuffled"] is not None else "None"), flush=True)
    print("verdict: PASS", flush=True)

    zg.observe(agent)

    b0 = {init: ladder_rows[_cell_id(init, B0_EPOCHS, B0_POOL_FRACTION)] for init in B0_INITS}

    def _gap(r: Dict[str, Any]) -> Optional[float]:
        return r["gap_vs_shuffled"]

    def _diff(a: Optional[float], b: Optional[float]) -> Optional[float]:
        return (a - b) if (a is not None and b is not None) else None

    t_full = _diff(_gap(b0["READAPT"]), _gap(b0["SCRATCH"]))
    t_partial = _diff(_gap(b0["PARTIAL"]), _gap(b0["SCRATCH"]))
    max_r = ladder_rows[_cell_id("READAPT", LADDER_MAX_EPOCHS, LADDER_MAX_FRACTION)]
    max_s = ladder_rows[_cell_id("SCRATCH", LADDER_MAX_EPOCHS, LADDER_MAX_FRACTION)]
    t_max = _diff(_gap(max_r), _gap(max_s))

    def _slope_over(rows: List[Dict[str, Any]], key: str) -> Optional[float]:
        return _ols_slope([r["log10_optimizer_steps"] for r in rows],
                          [r[key] if r[key] is not None else float("nan") for r in rows])

    slopes: Dict[str, Any] = {}
    for init in INITS:
        col = [ladder_rows[_cell_id(init, e, FULL_POOL_FRACTION)] for e in LADDER_EPOCHS]
        allc = [ladder_rows[_cell_id(init, e, f)] for e in LADDER_EPOCHS for f in LADDER_POOL_FRACTIONS]
        slopes[init] = {
            # LOAD-BEARING axis (H2): full-pool column, data fixed, steps vary with epochs only
            "full_pool_slope_gap_vs_shuffled_per_decade": _slope_over(col, "gap_vs_shuffled"),
            "full_pool_slope_gap_vs_chance_per_decade": _slope_over(col, "gap_vs_chance"),
            "full_pool_slope_forward_mse_per_decade": _slope_over(col, "heldout_forward_mse"),
            "full_pool_min_steps": int(min(r["optimizer_steps"] for r in col)),
            "full_pool_max_steps": int(max(r["optimizer_steps"] for r in col)),
            # DESCRIPTIVE: pooled 18-cell slope (steps x data confounded; red-team F5) and the
            # per-column (fixed data) / per-row (fixed epochs) decomposition
            "pooled_slope_gap_vs_shuffled_per_decade": _slope_over(allc, "gap_vs_shuffled"),
            "per_column_slope_gap_vs_shuffled": {
                "f%d" % int(round(f * 100)): _slope_over([ladder_rows[_cell_id(init, e, f)] for e in LADDER_EPOCHS], "gap_vs_shuffled")
                for f in LADDER_POOL_FRACTIONS},
            "per_row_slope_gap_vs_shuffled_vs_log10_ntrain": {
                "e%d" % e: _ols_slope([math.log10(max(ladder_rows[_cell_id(init, e, f)]["n_train"], 1)) for f in LADDER_POOL_FRACTIONS],
                                      [ladder_rows[_cell_id(init, e, f)]["gap_vs_shuffled"] if ladder_rows[_cell_id(init, e, f)]["gap_vs_shuffled"] is not None else float("nan") for f in LADDER_POOL_FRACTIONS])
                for e in LADDER_EPOCHS},
            "per_column_forward_mse_slope": {
                "f%d" % int(round(f * 100)): _slope_over([ladder_rows[_cell_id(init, e, f)] for e in LADDER_EPOCHS], "heldout_forward_mse")
                for f in LADDER_POOL_FRACTIONS},
        }
    transfer_curve = [
        {"cell": _cell_id("READAPT", e, f), "optimizer_steps": ladder_rows[_cell_id("READAPT", e, f)]["optimizer_steps"],
         "n_train": ladder_rows[_cell_id("READAPT", e, f)]["n_train"],
         "transfer_index": _diff(_gap(ladder_rows[_cell_id("READAPT", e, f)]), _gap(ladder_rows[_cell_id("SCRATCH", e, f)]))}
        for e in LADDER_EPOCHS for f in LADDER_POOL_FRACTIONS]

    print("transfer_index_b0 full=%s partial=%s (uncalibrated, single-seed, informational only)"
          % (("%.4f" % t_full) if t_full is not None else "None",
             ("%.4f" % t_partial) if t_partial is not None else "None"), flush=True)

    return {
        "seed": int(seed),
        "seed_index": int(seed_index),
        "p0a": p0a_stats,
        "zworld_encoder_guard": guard,
        "shift_record": shift_record,
        "init_state_hash": init_state_hash,
        "n_transitions_pre": n_pre,
        "n_transitions_post": n_post,
        "n_pool": int(n_pool),
        "n_env_drift_flag_disagreements_pre": buf_pre["n_flag_disagree"],
        "n_env_drift_flag_disagreements_post": buf_post["n_flag_disagree"],
        "class_fractions_pre": pre_class_fracs,
        "class_fractions_post": post_class_fracs,
        "surface_stat_max_deviation": surface_stat_deviation,
        "action_counts_pre": buf_pre["action_counts"],
        "action_counts_post": buf_post["action_counts"],
        "transfer_index_b0": t_full,
        "transfer_index_b0_partial": t_partial,
        "transfer_index_max_cell": t_max,
        "ladder_slopes": slopes,
        "transfer_curve": transfer_curve,
        "arms": arm_rows,
    }


# ------------------------------------------------------------------- adjudication
def _arm(r: Dict[str, Any], arm_id: str) -> Dict[str, Any]:
    return next(a for a in r["arms"] if a["arm_id"] == arm_id)


def _shared_preconditions(seed_rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """1001's readiness set, verbatim in semantics. Returns (flat entries, measured dict)."""
    out: List[Dict[str, Any]] = []
    worst = min(seed_rows,
                key=lambda r: r["zworld_encoder_guard"].get("world_encoder_max_abs_delta", 0.0))
    entry = zworld_precondition(worst["zworld_encoder_guard"],
                                context="V3-EXQ-1013 worst seed %d" % worst["seed"])
    entry["offending_cell"] = "seed=%d" % worst["seed"]
    out.append(entry)

    pre_base = [_arm(r, "PRE_BASE") for r in seed_rows]
    frozen = [_arm(r, "POST_FROZEN") for r in seed_rows]
    b0_id = _cell_id("READAPT", B0_EPOCHS, B0_POOL_FRACTION)
    b0_r = [_arm(r, b0_id) for r in seed_rows]

    v, k = _worst_low([(a["per_stratum"]["STAY"]["auroc_bare_delta"], "seed=%d/PRE" % r["seed"])
                       for r, a in zip(seed_rows, pre_base)]
                      + [(a["per_stratum"]["STAY"]["auroc_bare_delta"], "seed=%d/POST" % r["seed"])
                         for r, a in zip(seed_rows, frozen)])
    out.append({"name": "zworld_encodes_exogenous_change", "kind": "readiness",
                "description": "bare ||z_obs - z_prev|| AUROC for hazard-moved vs not, STAY stratum, pre- AND post-shift",
                "control": "positive control -- agent cannot have caused the change",
                "measured": v, "threshold": FLOOR_STAY_BARE_AUROC, "direction": "lower",
                "offending_cell": k, "met": (v is not None and v >= FLOOR_STAY_BARE_AUROC)})

    v, k = _worst_interval(
        [(a["per_stratum"]["MOVE_OK"]["auroc_bare_delta"], "seed=%d/PRE" % r["seed"]) for r, a in zip(seed_rows, pre_base)]
        + [(a["per_stratum"]["MOVE_OK"]["auroc_bare_delta"], "seed=%d/POST" % r["seed"]) for r, a in zip(seed_rows, frozen)],
        centre=0.5)
    out.append({"name": "move_ok_confound_absent", "kind": "readiness",
                "description": "bare ||z_obs - z_prev|| AUROC in MOVE_OK near chance, pre- AND post-shift",
                "control": "negative control", "measured": v,
                "threshold_low": BAND_MOVE_OK_BARE_LOW, "threshold_high": BAND_MOVE_OK_BARE_HIGH,
                "direction": "interval", "offending_cell": k,
                "met": (v is not None and BAND_MOVE_OK_BARE_LOW <= v <= BAND_MOVE_OK_BARE_HIGH)})

    v, k = _worst_low([(float(min(a["per_stratum"]["MOVE_OK"]["n_drift"], a["per_stratum"]["MOVE_OK"]["n_no_drift"])),
                        "seed=%d/POST_heldout" % r["seed"]) for r, a in zip(seed_rows, frozen)])
    out.append({"name": "causal_class_events_sufficient", "kind": "readiness",
                "description": "min(n_drift, n_no_drift) in the POST-shift held-out MOVE_OK stratum (the scoring set)",
                "control": "the exact precondition V3-EXQ-783 failed on",
                "measured": v, "threshold": float(FLOOR_MOVE_OK_CLASS_N), "direction": "lower",
                "offending_cell": k, "met": (v is not None and v >= FLOOR_MOVE_OK_CLASS_N)})

    v, k = _worst_low([(a["action_pathway_wiring_at_init"], "seed=%d PRE_BASE-init" % r["seed"]) for r, a in zip(seed_rows, pre_base)])
    out.append({"name": "e2world_action_pathway_live", "kind": "readiness",
                "description": "mean ||f(z,a_i) - f(z,a_j)|| AT INITIALISATION (PRE_BASE; SCRATCH cells share this init)",
                "control": "wiring check, before any training", "measured": v,
                "threshold": FLOOR_ACTION_PATHWAY, "comparator": ">", "direction": "lower",
                "offending_cell": k, "met": (v is not None and v > FLOOR_ACTION_PATHWAY)})

    v, k = _worst_low([(float(r["shift_record"]["n_actions_rewired"]), "seed=%d" % r["seed"]) for r in seed_rows])
    out.append({"name": "action_map_genuinely_rewired", "kind": "readiness",
                "description": "count of the 4 movement actions whose post-shift displacement differs from canonical (a derangement has no fixed point)",
                "control": "sanity check that the derangement is not an accidental no-op, verified per seed",
                "measured": v, "threshold": float(FLOOR_ACTIONS_REWIRED), "comparator": ">=",
                "direction": "lower", "offending_cell": k,
                "met": (v is not None and v >= FLOOR_ACTIONS_REWIRED)})

    v, k = _worst_low([(-r["surface_stat_max_deviation"], "seed=%d" % r["seed"]) for r in seed_rows])
    measured_dev = -v if v is not None else None
    out.append({"name": "surface_statistics_preserved", "kind": "readiness",
                "description": "max |pre-shift frac - post-shift frac| over STAY / MOVE_OK / MOVE_BLOCKED / drift-event fractions",
                "control": "the load-bearing design claim: a permutation of interchangeable movement actions under a uniform policy holds these marginals fixed in expectation",
                "measured": measured_dev, "threshold": CEIL_SURFACE_STAT_DEVIATION,
                "direction": "upper", "offending_cell": k,
                "met": (measured_dev is not None and measured_dev <= CEIL_SURFACE_STAT_DEVIATION)})

    pre_gap_chance = [(a["auroc_move_ok"] or 0.0) - 0.5 for a in pre_base]
    pre_chance_mean, pre_chance_sd = _mean_sd(pre_gap_chance)
    pre_gap_bare = [(a["auroc_move_ok"] or 0.0) - (a["auroc_bare_move_ok"] or 0.0) for a in pre_base]
    pre_bare_mean, pre_bare_sd = _mean_sd(pre_gap_bare)
    pre_chance_req = max(THRESH_PRE_BASE_ABS_FLOOR, THRESH_PRE_BASE_SD_MULT * pre_chance_sd)
    pre_bare_req = max(THRESH_PRE_BASE_ABS_FLOOR, THRESH_PRE_BASE_SD_MULT * pre_bare_sd)
    pre_met = (pre_chance_mean >= pre_chance_req) and (pre_bare_mean >= pre_bare_req)
    out.append({"name": "pre_base_in_distribution_signature", "kind": "readiness",
                "description": "PRE_BASE comparator informative in-distribution: mean gap-vs-chance AND mean gap-vs-bare each >= max(%.3f, sd)" % THRESH_PRE_BASE_ABS_FLOOR,
                "control": "mirrors 995's C1+C2 on PRE_BASE; a failing in-distribution fit makes any post-shift reading uninterpretable",
                "measured_chance_gap": pre_chance_mean, "measured_chance_gap_requirement": pre_chance_req,
                "measured_bare_gap": pre_bare_mean, "measured_bare_gap_requirement": pre_bare_req,
                "measured": min(pre_chance_mean - pre_chance_req, pre_bare_mean - pre_bare_req),
                "threshold": 0.0, "direction": "lower", "offending_cell": "PRE_BASE across seeds",
                "met": bool(pre_met)})

    v, k = _worst_low([(float(f["heldout_forward_mse"]) - float(p["heldout_forward_mse"]), "seed=%d" % r["seed"])
                       for r, p, f in zip(seed_rows, pre_base, frozen)])
    out.append({"name": "shift_degrades_frozen_forward_fit", "kind": "readiness",
                "description": "frozen model's post-shift held-out forward MSE minus its own PRE_BASE held-out forward MSE -- positive in every seed (the manipulation reached the DV)",
                "control": "manipulation-check, not a claim criterion",
                "measured": v, "threshold": 0.0, "comparator": ">", "direction": "lower",
                "offending_cell": k, "met": (v is not None and v > 0.0)})

    # ---- H1-specific: readapt non-vacuity (1001's gate), scratch/partial non-vacuity, draw precision
    readapt_margins = []
    for r, p, f, rd in zip(seed_rows, pre_base, frozen, b0_r):
        recovery = float(f["heldout_forward_mse"]) - float(rd["heldout_forward_mse"])
        degradation = float(f["heldout_forward_mse"]) - float(p["heldout_forward_mse"])
        readapt_margins.append((recovery - max(READAPT_RECOVERY_ABS_FLOOR, READAPT_RECOVERY_FRAC * max(degradation, 0.0)),
                                "seed=%d" % r["seed"]))
    v, k = _worst_low(readapt_margins)
    out.append({"name": "readapt_does_not_regress_frozen_fit", "kind": "readiness", "leg": "H1",
                "description": "READAPT@B0 held-out forward MSE improvement over FROZEN minus max(%.2g, %.0f%% of the shift-induced degradation) -- positive in every seed" % (READAPT_RECOVERY_ABS_FLOOR, READAPT_RECOVERY_FRAC * 100),
                "control": "readapt-budget non-vacuity check (1001's gate, same budget)",
                "measured": v, "threshold": 0.0, "comparator": ">", "direction": "lower",
                "offending_cell": k, "met": (v is not None and v > 0.0)})

    b0_s = [_arm(r, _cell_id("SCRATCH", B0_EPOCHS, B0_POOL_FRACTION)) for r in seed_rows]
    b0_p = [_arm(r, _cell_id("PARTIAL", B0_EPOCHS, B0_POOL_FRACTION)) for r in seed_rows]
    for tag, cells in (("scratch", b0_s), ("partial", b0_p)):
        v, k = _worst_low([(float(s["entry_weights_heldout_forward_mse"]) - float(s["heldout_forward_mse"]), "seed=%d" % r["seed"])
                           for r, s in zip(seed_rows, cells)])
        out.append({"name": "%s_control_trained" % tag, "kind": "readiness", "leg": "H1",
                    "description": "%s@B0 held-out forward MSE at THIS cell's entry weights (measured on the same model object before its first optimizer step; red-team F6) minus after the 120-step budget -- positive in every seed (the control is not a no-op)" % tag.upper(),
                    "control": "positive control on the H1 discriminator itself",
                    "measured": v, "threshold": 0.0, "comparator": ">", "direction": "lower",
                    "offending_cell": k, "met": (v is not None and v > 0.0)})

    # entry-state identity (red-team F7): SCRATCH@B0's entry hash must equal the seed's shared init
    n_ident = sum(1 for r, s in zip(seed_rows, b0_s) if s.get("entry_state_hash") == r.get("init_state_hash"))
    out.append({"name": "scratch_entry_is_shared_init", "kind": "readiness", "leg": "H1",
                "description": "count of seeds whose SCRATCH@B0 entry_state_hash equals the seed's recorded init_state_hash (the weights PRE_BASE was trained from)",
                "control": "identity recorded, not inferred from RNG order",
                "measured": float(n_ident), "threshold": float(len(seed_rows)), "comparator": ">=", "direction": "lower",
                "offending_cell": "all seeds", "met": bool(n_ident == len(seed_rows))})

    v, k = _worst_low([(-(a["auroc_shuffled_move_ok_se"] if a["auroc_shuffled_move_ok_se"] is not None else float("inf")),
                        "seed=%d/%s@B0" % (r["seed"], a["init"])) for r in seed_rows for a in (_arm(r, _cell_id(i, B0_EPOCHS, B0_POOL_FRACTION)) for i in B0_INITS)])
    se_worst = -v if v is not None else None
    out.append({"name": "shuffle_draw_se_resolves_floor", "kind": "readiness", "leg": "H1",
                "description": "standard error of the %d-draw shuffled-action AUROC mean at B0 (worst of the three starts) must be <= TRANSFER_ABS_FLOOR so the H1 floor is resolvable; draws are PAIRED across cells within a seed (same permutations), so this is conservative for T (red-team F8)" % N_SHUFFLE_DRAWS,
                "control": "instrument precision on the DV's own control term",
                "measured": se_worst, "threshold": TRANSFER_ABS_FLOOR, "direction": "upper",
                "offending_cell": k, "met": (se_worst is not None and se_worst <= TRANSFER_ABS_FLOOR)})

    # ---- H2-specific: the FULL-POOL column must span its pre-registered budget range
    v, k = _worst_low([(float(r["ladder_slopes"]["READAPT"]["full_pool_max_steps"]) / max(1.0, float(r["ladder_slopes"]["READAPT"]["full_pool_min_steps"])),
                        "seed=%d" % r["seed"]) for r in seed_rows])
    out.append({"name": "ladder_step_range_spans_100x", "kind": "readiness", "leg": "H2",
                "description": "realised max/min optimizer steps across the READAPT FULL-POOL column (data fixed at the whole pool; steps vary with epochs only -- the load-bearing H2 axis, red-team F5)",
                "control": "structural: the budget axis must cover >= 2 decades for a slope to mean anything",
                "measured": v, "threshold": LADDER_STEP_RANGE_FLOOR, "comparator": ">=", "direction": "lower",
                "offending_cell": k, "met": (v is not None and v >= LADDER_STEP_RANGE_FLOOR)})

    # ---- dv_headroom (W5 rule), each denominated on the DV's OWN reachable range at the
    # budget the criterion reads (red-team F1/F2/F4), never on the 6600-step asymptote:
    #   H1: T = R - S with S ~ 0, so R (READAPT@B0 gap_vs_shuffled, this run's own value) bounds
    #       what T can reach; the floor must sit at most half of R away.
    #   H2: the SCRATCH full-pool ladder is the instrument's own learning curve under the same
    #       relabelling symmetry -- if it cannot produce a slope >= 2x the floor, a flat READAPT
    #       slope is instrument starvation, not 'no scaling'. This IS the H2 positive control.
    headroom: Dict[str, Dict[str, Any]] = {}
    r_b0 = [a["gap_vs_shuffled"] for a in b0_r if a["gap_vs_shuffled"] is not None]
    if r_b0:
        headroom["transfer"] = dv_headroom_check(
            "dv_headroom_transfer_index",
            dv_name="transfer index T = gap_vs_shuffled(READAPT@B0) - gap_vs_shuffled(SCRATCH@B0); T's reachable ceiling is bounded by READAPT@B0's own gap",
            criterion_threshold=TRANSFER_ABS_FLOOR, control_values=r_b0, statistic="floor_headroom",
            dv_bounds=(0.0, 1.0),   # signed: min over seeds of R, not |R| (pass-2 N2)
            margin=HEADROOM_MARGIN, measured_cells=["READAPT@B0 seed=%d" % r["seed"] for r in seed_rows],
            leg="H1")
    s_scr = [r["ladder_slopes"]["SCRATCH"]["full_pool_slope_gap_vs_shuffled_per_decade"] for r in seed_rows]
    s_scr = [v for v in s_scr if v is not None]
    if s_scr:
        headroom["slope"] = dv_headroom_check(
            "dv_headroom_budget_slope",
            dv_name="d gap_vs_shuffled / d log10(optimizer steps) on the full-pool column; achievable = the SCRATCH column's own slope (the instrument's learning curve = H2 positive control)",
            criterion_threshold=SLOPE_ABS_FLOOR, control_values=s_scr, statistic="floor_headroom",
            dv_bounds=(0.0, 1.0),   # signed: min over seeds of the SCRATCH slope; a declining column cannot certify (pass-2 N2)
            margin=HEADROOM_MARGIN, measured_cells=["SCRATCH full-pool column seed=%d" % r["seed"] for r in seed_rows],
            leg="H2")
    for h in headroom.values():
        h["met"] = bool(h["measured"] == h["measured"] and h["measured"] > h["threshold"])
        out.append(h)

    measured = {p["name"]: (float(p["measured"]) if p.get("measured") is not None and p["measured"] == p["measured"] else float("nan"))
                for p in out}
    return out, measured


LEG_SPECIFIC = {"H1": {"readapt_does_not_regress_frozen_fit", "scratch_control_trained", "partial_control_trained",
                       "scratch_entry_is_shared_init", "shuffle_draw_se_resolves_floor", "dv_headroom_transfer_index"},
                "H2": {"ladder_step_range_spans_100x", "dv_headroom_budget_slope"}}


def _leg_gates(flat: List[Dict[str, Any]], measured: Dict[str, float]) -> Dict[str, Any]:
    """Regime-conditioned gates (785 rule): each precondition applies to the legs it is
    meaningful for; `met` is taken from the flat entries (several are interval / >= / upper
    bounds the helper's strict floor cannot express), so met_overrides carries every one."""
    specs: List[PreconditionSpec] = []
    for p in flat:
        own = [leg for leg, names in LEG_SPECIFIC.items() if p["name"] in names]
        legs = own or ["H1", "H2"]
        specs.append(PreconditionSpec(
            name=p["name"], description=p.get("description", ""), control=p.get("control", ""),
            threshold=float(p.get("threshold", p.get("threshold_low", 0.0)) or 0.0),
            direction=str(p.get("direction", "lower")), kind=str(p.get("kind", "readiness")),
            applies_to=(lambda ctx, _legs=tuple(legs): ctx["leg"] in _legs),
            applies_note="specific to leg(s) %s" % ",".join(legs)))
    overrides = {p["name"]: bool(p.get("met")) for p in flat}
    gates = [evaluate_arm_gate(leg, {"leg": leg}, specs, measured, met_overrides=overrides,
                               auto_detect_vacuity=False) for leg in ("H1", "H2")]
    return aggregate_arm_gates(gates)


def _adjudicate(seed_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    flat, measured = _shared_preconditions(seed_rows)
    agg = _leg_gates(flat, measured)
    h1_green = "H1" in agg["green_arms"]
    h2_green = "H2" in agg["green_arms"]

    # ---- H1: transfer index at B0, from the FULL pre-shift weights and from the PARTIAL start
    t_vals = [r["transfer_index_b0"] for r in seed_rows]
    t_mean, t_sd = _mean_sd(t_vals)
    t_req = max(TRANSFER_ABS_FLOOR, TRANSFER_SD_MULT * t_sd)
    h1_full = bool(h1_green and t_mean == t_mean and t_mean >= t_req)
    tp_vals = [r["transfer_index_b0_partial"] for r in seed_rows]
    tp_mean, tp_sd = _mean_sd(tp_vals)
    tp_req = max(TRANSFER_ABS_FLOOR, TRANSFER_SD_MULT * tp_sd)
    h1_partial = bool(h1_green and tp_mean == tp_mean and tp_mean >= tp_req)
    h1_refuted = bool(h1_full or h1_partial)
    tm_vals = [r["transfer_index_max_cell"] for r in seed_rows]
    tm_mean, tm_sd = _mean_sd(tm_vals)
    tm_req = max(TRANSFER_ABS_FLOOR, TRANSFER_SD_MULT * tm_sd)
    t_max_positive = bool(h1_green and tm_mean == tm_mean and tm_mean >= tm_req)

    # ---- H2: full-pool-column slope (READAPT), plus the secondary readouts
    s_vals = [r["ladder_slopes"]["READAPT"]["full_pool_slope_gap_vs_shuffled_per_decade"] for r in seed_rows]
    s_mean, s_sd = _mean_sd(s_vals)
    s_req = max(SLOPE_ABS_FLOOR, SLOPE_SD_MULT * s_sd)
    h2_supported = bool(h2_green and s_mean == s_mean and s_mean >= s_req)
    s_scr = [r["ladder_slopes"]["SCRATCH"]["full_pool_slope_gap_vs_shuffled_per_decade"] for r in seed_rows]
    s_scr_mean, s_scr_sd = _mean_sd(s_scr)
    s_pool = [r["ladder_slopes"]["READAPT"]["pooled_slope_gap_vs_shuffled_per_decade"] for r in seed_rows]
    s_pool_mean, s_pool_sd = _mean_sd(s_pool)
    max_id = _cell_id("READAPT", LADDER_MAX_EPOCHS, LADDER_MAX_FRACTION)
    max_gap = [_arm(r, max_id)["gap_vs_shuffled"] for r in seed_rows]
    max_gap_mean, max_gap_sd = _mean_sd(max_gap)
    crossing_req = max(THRESH_ACTION_ABS_FLOOR, 1.0 * max_gap_sd)
    ladder_max_crosses = bool(h2_green and max_gap_mean == max_gap_mean and max_gap_mean >= crossing_req)
    max_s_id = _cell_id("SCRATCH", LADDER_MAX_EPOCHS, LADDER_MAX_FRACTION)
    max_s_gap = [_arm(r, max_s_id)["gap_vs_shuffled"] for r in seed_rows]
    max_s_gap_mean, max_s_gap_sd = _mean_sd(max_s_gap)
    b0_id = _cell_id("READAPT", B0_EPOCHS, B0_POOL_FRACTION)
    b0_chance = [_arm(r, b0_id)["gap_vs_chance"] for r in seed_rows]
    b0_chance_mean, b0_chance_sd = _mean_sd(b0_chance)
    b0_shuf = [_arm(r, b0_id)["gap_vs_shuffled"] for r in seed_rows]
    b0_shuf_mean, b0_shuf_sd = _mean_sd(b0_shuf)
    b0_scr = [_arm(r, _cell_id("SCRATCH", B0_EPOCHS, B0_POOL_FRACTION))["gap_vs_shuffled"] for r in seed_rows]
    b0_scr_mean, b0_scr_sd = _mean_sd(b0_scr)
    b0_par = [_arm(r, _cell_id("PARTIAL", B0_EPOCHS, B0_POOL_FRACTION))["gap_vs_shuffled"] for r in seed_rows]
    b0_par_mean, b0_par_sd = _mean_sd(b0_par)
    asym = [_arm(r, "PRE_BASE")["gap_vs_shuffled"] for r in seed_rows]
    asym_mean, asym_sd = _mean_sd(asym)

    criteria = [
        {"name": "H1_positive_transfer_vs_scratch_same_budget", "leg": "H1", "load_bearing": True,
         "passed": h1_full, "mean": t_mean, "sd": t_sd, "requirement": t_req,
         "statistic": "mean over seeds of T = gap_vs_shuffled(READAPT@B0) - gap_vs_shuffled(SCRATCH@B0) >= max(%.3f, %.1f*sd); B0 = 3600 transitions x 8 epochs = 120 optimizer steps (1001's exact budget); shuffled term is the mean of %d paired permutation draws" % (TRANSFER_ABS_FLOOR, TRANSFER_SD_MULT, N_SHUFFLE_DRAWS),
         "passed_means": "positive transfer from the FULL pre-shift weights (H-shortcut REFUTED)",
         "failed_means": "no net positive transfer from the full weights -- NOT by itself 'nothing reusable carried': the deranged action map must be unlearned first (red-team F3); see the PARTIAL criterion",
         "per_seed": t_vals},
        {"name": "H1_positive_transfer_partial_start_vs_scratch", "leg": "H1", "load_bearing": True,
         "passed": h1_partial, "mean": tp_mean, "sd": tp_sd, "requirement": tp_req,
         "statistic": "mean over seeds of T_partial = gap_vs_shuffled(PARTIAL@B0) - gap_vs_shuffled(SCRATCH@B0) >= max(%.3f, %.1f*sd); PARTIAL = trained PRE_BASE weights with the whole action-input pathway (_residual_fwd.action_encoder AND transition_net[0]'s action-embedding input columns) restored to the shared INIT weights; state columns, first-layer bias and output layer kept" % (TRANSFER_ABS_FLOOR, TRANSFER_SD_MULT),
         "passed_means": "reusable NON-action structure is carried: action-conditioning re-derives faster from the trained state pathway + output layer than from a fresh head, with every weight that can hold an action->effect entry discarded (H-shortcut REFUTED)",
         "failed_means": "the trained state pathway gives the re-derivation nothing (H-shortcut STANDS, declared null)",
         "per_seed": tp_vals},
        {"name": "H2_recovery_scales_with_budget", "leg": "H2", "load_bearing": True,
         "passed": h2_supported, "mean": s_mean, "sd": s_sd, "requirement": s_req,
         "statistic": "mean over seeds of the per-seed OLS slope of gap_vs_shuffled(READAPT full-pool column) on log10(realised optimizer steps) across the %d epoch levels (data FIXED at the whole readapt pool, %d transitions at full scale) >= max(%.3f, %.1f*sd)" % (len(LADDER_EPOCHS), int(READAPT_POOL_FRACTION * N_TRANSITIONS_POST), SLOPE_ABS_FLOOR, SLOPE_SD_MULT),
         "passed_means": "H-budget SUPPORTED (recovery scales with optimizer budget at fixed data)",
         "failed_means": "H-budget REFUTED (declared null: flat over a >=100x step range) -- adjudicable ONLY because the H2 gate's dv_headroom_budget_slope certified the SCRATCH column DOES produce a slope here (red-team F4)",
         "per_seed": s_vals},
        {"name": "H1_transfer_at_ladder_max_cell", "leg": "H1", "load_bearing": False,
         "passed": t_max_positive, "mean": tm_mean, "sd": tm_sd, "requirement": tm_req,
         "statistic": "T at the ladder's largest cell (%s vs %s) >= max(%.3f, sd) -- SECONDARY: transfer that only appears at a larger budget (red-team F1 note)" % (max_id, max_s_id, TRANSFER_ABS_FLOOR),
         "per_seed": tm_vals},
        {"name": "ladder_max_cell_crosses_1001_action_floor", "leg": "H2", "load_bearing": False,
         "passed": ladder_max_crosses, "mean": max_gap_mean, "sd": max_gap_sd, "requirement": crossing_req,
         "statistic": "gap_vs_shuffled(READAPT@%s) mean over seeds >= max(%.3f, sd) -- 1001's D2 action floor; SECONDARY (autopsy: 'a monotone rise crossing 0.05 refutes H1')" % (max_id, THRESH_ACTION_ABS_FLOOR),
         "per_seed": max_gap},
    ]

    non_deg = arm_criteria_non_degenerate(
        {"H1": ["H1_positive_transfer_vs_scratch_same_budget", "H1_positive_transfer_partial_start_vs_scratch", "H1_transfer_at_ladder_max_cell"],
         "H2": ["H2_recovery_scales_with_budget", "ladder_max_cell_crosses_1001_action_floor"]},
        agg,
        extra={"H1_positive_transfer_vs_scratch_same_budget": bool(sum(1 for v in t_vals if v is not None) >= 2 and t_sd > 0.0),
               "H1_positive_transfer_partial_start_vs_scratch": bool(sum(1 for v in tp_vals if v is not None) >= 2 and tp_sd > 0.0),
               "H1_transfer_at_ladder_max_cell": bool(sum(1 for v in tm_vals if v is not None) >= 2 and tm_sd > 0.0),
               "H2_recovery_scales_with_budget": bool(sum(1 for v in s_vals if v is not None) >= 2 and s_sd > 0.0),
               "ladder_max_cell_crosses_1001_action_floor": bool(sum(1 for v in max_gap if v is not None) >= 2 and max_gap_sd > 0.0)})
    for c in criteria:
        if not non_deg.get(c["name"], True):
            c["load_bearing"] = False if c["leg"] not in agg["green_arms"] else c["load_bearing"]

    if not agg["any_green"]:
        label, direction, overall = "substrate_not_ready_requeue", "non_contributory", False
        grid_cell = "no_leg_adjudicated"
    elif not (h1_green and h2_green):
        leg = "H1" if h1_green else "H2"
        label = "%s_only_adjudicated_%s" % (leg.lower(), ("h1_refuted" if h1_refuted else "h1_stands") if leg == "H1" else ("h2_supported" if h2_supported else "h2_refuted"))
        direction, overall, grid_cell = "mixed", False, "one_leg_adjudicated"
    elif h1_refuted and h2_supported:
        label, direction, overall = "causal_signature_rederivable_positive_transfer_budget_scaling", "supports", True
        grid_cell = "H1_refuted_H2_supported"
    elif (not h1_refuted) and (not h2_supported):
        label, direction, overall = "causal_signature_shortcut_no_transfer_no_scaling", "weakens", False
        grid_cell = "H1_stands_H2_refuted"
    elif (not h1_refuted) and h2_supported:
        label, direction, overall = "refit_scales_without_transfer_testbed_open", "mixed", False
        grid_cell = "H1_stands_H2_supported (H3-ambiguous cell; H-testbed-degenerate stays live)"
    else:
        label, direction, overall = "transfer_without_budget_scaling", "mixed", False
        grid_cell = "H1_refuted_H2_refuted"

    hyp = {
        "H-shortcut": {"adjudicated": h1_green, "state": ("eliminated" if h1_refuted else ("alive" if h1_green else "unadjudicated"))},
        "H-budget": {"adjudicated": h2_green, "state": ("supported" if h2_supported else ("eliminated" if h2_green else "unadjudicated"))},
        "H-testbed-degenerate": {"adjudicated": False, "state": "alive", "note": H3_TESTBED_DEGENERATE["status"]},
    }

    return {
        "outcome": "PASS" if overall else "FAIL",
        "evidence_direction": direction,
        "evidence_direction_per_claim": {"SD-031": direction},
        "criteria": criteria,
        "registry_adjudication": {"question": REGISTRY_QUESTION, "hypotheses": hyp, "grid_cell": grid_cell},
        "claim_scope": {
            "claim_id_primary": "SD-031",
            "claim_ids_secondary": [],
            "what_this_run_measures": (
                "Whether SD-031's single-pass world-stream comparator re-derives ACTION-CONDITIONED "
                "discrimination under a structural relabelling of the environment's action->effect "
                "map (a) more cheaply from its pre-shift weights than from scratch at the SAME budget "
                "(H1, transfer) and (b) increasingly with budget over a >=100x optimizer-step range "
                "(H2, scaling). Construction-balanced RandomPolicy collection, offline-scored, per the "
                "SD-031 what_would_answer amendment ratified 2026-09-04 (governance-20260904-1347). "
                "Not measured: any compositional/state-dependent causal structure (H3, pre-registered "
                "untestable here), ARC-037 routing, MECH-095's TPJ comparator, EXT-005's LLM-side assertion."),
            "attribution_caveat": (
                "The environment's physics is a 4-entry lookup table; a cheap re-fit from EITHER "
                "initialisation is possible in principle (H3). The H1 x H2 grid is what keeps that "
                "honest: H1-stands AND H2-supported reads `mixed`, never `weakens`; only flat-AND-no-"
                "transfer is the shortcut fingerprint, because a re-fittable table would scale."),
        },
        "combination_rule": (
            "overall PASS = H1 REFUTED (positive transfer from the FULL weights OR from the PARTIAL "
            "start, each noise-aware, OR-combined; red-team F3) AND H2 SUPPORTED (full-pool-column "
            "slope, adjudicable only with the SCRATCH column's slope certified by dv_headroom_budget_slope; "
            "red-team F4/F5); direction per "
            "grid cell: (refuted, supported) -> supports; (stands, refuted) -> weakens; (stands, "
            "supported) -> mixed (H3-ambiguous); (refuted, refuted) -> mixed. Each leg is gated by its "
            "own regime-conditioned preconditions (experiments/_lib/precondition_gate): a RED leg is "
            "UNADJUDICATED (its criterion criteria_non_degenerate False, load_bearing False), the "
            "other leg is still scored, and the direction is `mixed` with a *_only_adjudicated label; "
            "both RED -> substrate_not_ready_requeue / non_contributory. The crossing readout is "
            "secondary and never changes outcome."),
        "dv_symmetry_declaration": (
            "H1 DV: difference of two within-stratum (post-shift held-out MOVE_OK) rank statistics "
            "(residual-norm AUROC minus mean shuffled-action AUROC); symmetry group = monotone "
            "rescalings of the residual norm and within-class permutations; the manipulation "
            "(initial weights PRE_BASE vs fresh) re-ranks steps across the drift boundary and is "
            "not in the group. H2 DV: OLS slope of that gap on log-budget; the manipulation "
            "(optimizer steps / data count) is not a broadcast constant, rescaling or permutation."),
        "requeue_semantics": (
            "substrate_not_ready_requeue fires only when BOTH legs' gates are red; a single red leg "
            "leaves the other leg scored (785 rule). It is a SIGNAL for governance, not an automated requeue."),
        "interpretation": {
            "label": label,
            "preconditions": agg["adjudication_preconditions"],
            "criteria_non_degenerate": non_deg,
            "gate_green": bool(agg["all_green"]),
        },
        "per_arm_gate": agg["per_arm_gate"],
        "non_degenerate": bool(agg["non_degenerate"]) and any(non_deg.values()),
        "degeneracy_reason": agg["degeneracy_reason"],
        "preconditions_flat_all": flat,
        "summary": {
            "transfer_index_b0_full_mean": t_mean, "transfer_index_b0_full_sd": t_sd, "transfer_requirement_full": t_req,
            "transfer_index_b0_partial_mean": tp_mean, "transfer_index_b0_partial_sd": tp_sd, "transfer_requirement_partial": tp_req,
            "transfer_index_max_cell_mean": tm_mean, "transfer_index_max_cell_sd": tm_sd,
            "readapt_b0_gap_vs_shuffled_mean": b0_shuf_mean, "readapt_b0_gap_vs_shuffled_sd": b0_shuf_sd,
            "scratch_b0_gap_vs_shuffled_mean": b0_scr_mean, "scratch_b0_gap_vs_shuffled_sd": b0_scr_sd,
            "partial_b0_gap_vs_shuffled_mean": b0_par_mean, "partial_b0_gap_vs_shuffled_sd": b0_par_sd,
            "readapt_b0_gap_vs_chance_mean": b0_chance_mean, "readapt_b0_gap_vs_chance_sd": b0_chance_sd,
            "readapt_b0_gap_vs_chance_vs_1001_floor": (b0_chance_mean - THRESH_CHANCE_ABS_FLOOR) if b0_chance_mean == b0_chance_mean else None,
            "readapt_full_pool_slope_per_decade_mean": s_mean, "readapt_full_pool_slope_sd": s_sd, "slope_requirement": s_req,
            "scratch_full_pool_slope_per_decade_mean": s_scr_mean, "scratch_full_pool_slope_sd": s_scr_sd,
            "readapt_pooled_18cell_slope_per_decade_mean": s_pool_mean, "readapt_pooled_18cell_slope_sd": s_pool_sd,
            "ladder_max_cell_readapt_gap_vs_shuffled_mean": max_gap_mean, "ladder_max_cell_readapt_gap_vs_shuffled_sd": max_gap_sd,
            "ladder_max_cell_scratch_gap_vs_shuffled_mean": max_s_gap_mean, "ladder_max_cell_scratch_gap_vs_shuffled_sd": max_s_gap_sd,
            "pre_base_asymptote_gap_vs_shuffled_mean": asym_mean, "pre_base_asymptote_gap_vs_shuffled_sd": asym_sd,
            "n_seeds_scored": sum(1 for v in t_vals if v is not None),
            "load_bearing_gap": "gap_vs_shuffled (action-conditioned component); gap_vs_chance REPORTED only",
            "shuffle_draws_note": "the %d permutation draws are PAIRED across every cell within a seed (same generator seed), so per-cell draw sds are not independent across cells and the SE is conservative for T (red-team F8)" % N_SHUFFLE_DRAWS,
        },
        "degeneracy": check_degeneracy({
            "transfer_index_b0_full": [v for v in t_vals if v is not None],
            "transfer_index_b0_partial": [v for v in tp_vals if v is not None],
            "readapt_full_pool_slope_per_decade": [v for v in s_vals if v is not None],
        }),
    }


# --------------------------------------------------------------------------- main
def run_experiment(dry_run: bool = False) -> Dict[str, Any]:
    t0 = time.perf_counter()
    seeds = SEEDS[:2] if dry_run else SEEDS
    post_episodes = 40 if dry_run else POST_EPISODES_PER_PHASE
    zg = ZGoalStreamAccumulator()
    seed_rows = [_run_seed(s, i, dry_run, zg, post_episodes) for i, s in enumerate(seeds)]
    adj = _adjudicate(seed_rows)

    run_id = "%s_%s_v3" % (EXPERIMENT_TYPE, datetime.utcnow().strftime("%Y%m%dT%H%M%SZ"))
    manifest: Dict[str, Any] = {
        "run_id": run_id,
        "queue_id": QUEUE_ID,
        "supersedes": SUPERSEDES,
        "experiment_type": EXPERIMENT_TYPE,
        "architecture_epoch": ARCHITECTURE_EPOCH,
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "outcome": adj["outcome"],
        "evidence_direction": adj["evidence_direction"],
        "evidence_direction_per_claim": adj["evidence_direction_per_claim"],
        "timestamp_utc": datetime.utcnow().strftime("%Y%m%dT%H%M%SZ"),
        "criteria": adj["criteria"],
        "registry_adjudication": adj["registry_adjudication"],
        "h3_testbed_degenerate": H3_TESTBED_DEGENERATE,
        "claim_scope": adj["claim_scope"],
        "combination_rule": adj["combination_rule"],
        "dv_symmetry_declaration": adj["dv_symmetry_declaration"],
        "requeue_semantics": adj["requeue_semantics"],
        "interpretation": adj["interpretation"],
        "per_arm_gate": adj["per_arm_gate"],
        "non_degenerate": adj["non_degenerate"],
        "degeneracy_reason": adj["degeneracy_reason"],
        "preconditions_flat_all": adj["preconditions_flat_all"],
        "dv_headroom_entries": [p for p in adj["preconditions_flat_all"] if p.get("kind") == "dv_headroom"],
        "summary": adj["summary"],
        "per_seed_results": seed_rows,
        "arm_results": [a for r in seed_rows for a in r["arms"]],
        "pre_registered_thresholds": {
            "transfer_abs_floor": TRANSFER_ABS_FLOOR, "transfer_sd_mult": TRANSFER_SD_MULT,
            "slope_abs_floor_per_decade": SLOPE_ABS_FLOOR, "slope_sd_mult": SLOPE_SD_MULT,
            "headroom_margin": HEADROOM_MARGIN,
            "secondary_action_abs_floor_1001": THRESH_ACTION_ABS_FLOOR,
            "reported_chance_abs_floor_1001": THRESH_CHANCE_ABS_FLOOR,
            "pre_base_abs_floor": THRESH_PRE_BASE_ABS_FLOOR, "pre_base_sd_mult": THRESH_PRE_BASE_SD_MULT,
            "floor_stay_bare_auroc": FLOOR_STAY_BARE_AUROC,
            "band_move_ok_bare": [BAND_MOVE_OK_BARE_LOW, BAND_MOVE_OK_BARE_HIGH],
            "floor_move_ok_class_n": FLOOR_MOVE_OK_CLASS_N,
            "floor_action_pathway": FLOOR_ACTION_PATHWAY,
            "floor_actions_rewired": FLOOR_ACTIONS_REWIRED,
            "ceil_surface_stat_deviation": CEIL_SURFACE_STAT_DEVIATION,
            "readapt_recovery_abs_floor": READAPT_RECOVERY_ABS_FLOOR, "readapt_recovery_frac": READAPT_RECOVERY_FRAC,
            "ladder_step_range_floor": LADDER_STEP_RANGE_FLOOR,
            "b0_budget": {"epochs": B0_EPOCHS, "pool_fraction": B0_POOL_FRACTION,
                          "transitions_full_scale": int(B0_POOL_FRACTION * READAPT_POOL_FRACTION * N_TRANSITIONS_POST),
                          "optimizer_steps_full_scale": int(math.ceil(B0_POOL_FRACTION * READAPT_POOL_FRACTION * N_TRANSITIONS_POST / HEAD_BATCH) * B0_EPOCHS)},
            "ladder": {"epochs": LADDER_EPOCHS, "pool_fractions": LADDER_POOL_FRACTIONS, "inits": INITS,
                       "load_bearing_column_pool_fraction": FULL_POOL_FRACTION,
                       "full_pool_steps_full_scale": [int(math.ceil(READAPT_POOL_FRACTION * N_TRANSITIONS_POST / HEAD_BATCH) * e) for e in LADDER_EPOCHS]},
            "b0_inits": B0_INITS,
            "partial_start": "trained PRE_BASE weights with _residual_fwd.action_encoder AND transition_net[0].weight[:, WORLD_DIM:] restored to the shared init weights; state columns, first-layer bias and output layer kept",
            "n_shuffle_draws": N_SHUFFLE_DRAWS,
        },
        "label_provenance": ("exogenous-change label from hazard COORDINATES before/after each step; "
                             "info['env_drift_occurred'] recorded for audit only (over-reports, see 995)"),
        "known_substrate_limitations": [
            "SD-018 (degrading): overlaps latent/stack.py + latent/zworld_p0.py; both resource heads absent here",
            "mech357-freeze-incompatible-pressure-mechanism (degrading): overlaps causal_grid_world.py; reef_enabled False, hazard_agent_pursuit 0.0",
            "SD-MECH303-THRESHOLD-SOURCING (degrading): overlaps causal_grid_world.py and utils/config.py",
            "mode-governance-engagement (corrupting, overlaps agent.py/utils/config.py): gated off, salience_affinity_input_cap=None verified live on this construction",
            "SD-082 (corrupting, overlaps agent.py): gated off, use_lateral_pfc_analog=False verified live on this construction",
        ],
        "ethics_preflight": {
            "involves_negative_valence": False,
            "involves_suffering_like_state": False,
            "involves_self_model": False,
            "involves_inescapability_or_helplessness": False,
            "involves_offline_replay_over_harm": False,
            "involves_social_mind_or_language": False,
            "involves_human_data_or_clinical_context": False,
            "decision": "allow",
        },
        "dry_run": bool(dry_run),
    }
    manifest.update(adj["degeneracy"])
    # check_degeneracy may set non_degenerate from the load-bearing spreads; the regime gate's
    # verdict must not be overwritten by it in the PASS direction, only in the FAIL direction.
    manifest["non_degenerate"] = bool(adj["non_degenerate"]) and bool(manifest.get("non_degenerate", True))
    if not manifest["non_degenerate"] and not manifest.get("degeneracy_reason"):
        manifest["degeneracy_reason"] = adj["degeneracy_reason"]

    stamp_recording_core(
        manifest,
        config={"env_kwargs": dict(ENV_KWARGS), "world_dim": WORLD_DIM, "alpha_world": ALPHA_WORLD,
                "episodes_per_phase": EPISODES_PER_PHASE, "post_episodes_per_phase": post_episodes,
                "p0a_steps_per_episode": P0A_STEPS_PER_EPISODE,
                "n_transitions_pre": N_TRANSITIONS_PRE, "n_transitions_post": N_TRANSITIONS_POST,
                "heldout_fraction_pre": HELDOUT_FRACTION_PRE, "readapt_pool_fraction": READAPT_POOL_FRACTION,
                "ladder_epochs": LADDER_EPOCHS, "ladder_pool_fractions": LADDER_POOL_FRACTIONS,
                "b0_epochs": B0_EPOCHS, "b0_pool_fraction": B0_POOL_FRACTION,
                "head_epochs": HEAD_EPOCHS, "head_batch": HEAD_BATCH, "head_lr": HEAD_LR,
                "n_shuffle_draws": N_SHUFFLE_DRAWS, "derangements": [list(d) for d in DERANGEMENTS],
                "dry_run": bool(dry_run)},
        seeds=seeds,
        script_path=Path(__file__),
        started_at=t0,
        z_goal_stream_stats=zg.stats(),
        agent=None,
    )
    return manifest


def main() -> Tuple[Dict[str, Any], Path, bool]:
    parser = argparse.ArgumentParser(description="V3-EXQ-1013 SD-031 shortcut-vs-model portfolio")
    parser.add_argument("--dry-run", action="store_true",
                        help="tiny smoke run (2 seeds, 4 blocks, reduced epochs)")
    args = parser.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    manifest = run_experiment(dry_run=args.dry_run)
    out_path = write_flat_manifest(manifest, dry_run=args.dry_run,
                                   script_path=Path(__file__), stamp=False)

    print("Outcome: %s" % manifest["outcome"], flush=True)
    print("registry: %s" % {k: v["state"] for k, v in manifest["registry_adjudication"]["hypotheses"].items()}, flush=True)
    print("label:   %s" % manifest["interpretation"]["label"], flush=True)
    print("grid:    %s" % manifest["registry_adjudication"]["grid_cell"], flush=True)
    for c in manifest["criteria"]:
        print("  %-46s leg=%s load_bearing=%-5s passed=%-5s mean=%s req=%s"
              % (c["name"], c["leg"], c["load_bearing"], c["passed"],
                 ("%.4f" % c["mean"]) if c["mean"] == c["mean"] else "nan",
                 ("%.4f" % c["requirement"]) if c["requirement"] == c["requirement"] else "nan"), flush=True)
    for p in manifest["preconditions_flat_all"]:
        print("  precondition %-36s leg=%-4s met=%-5s measured=%s"
              % (p["name"], p.get("leg", "both"), p.get("met"), p.get("measured")), flush=True)
    print("  per_arm_gate green=%s red=%s" % (manifest["per_arm_gate"]["green_arms"], manifest["per_arm_gate"]["red_arms"]), flush=True)
    if args.dry_run:
        # positive control on the decisive readout: the ladder DV must VARY across cells
        gaps = [a["gap_vs_shuffled"] for a in manifest["arm_results"]
                if a.get("role") == "ladder_cell" and a.get("gap_vs_shuffled") is not None]
        varies = len(set(round(g, 6) for g in gaps)) > 1
        print("[smoke] ladder_dv_varies_across_cells: %s (n_cells=%d, min=%.4f, max=%.4f)"
              % (varies, len(gaps), min(gaps) if gaps else float("nan"), max(gaps) if gaps else float("nan")), flush=True)
    print("Result pack written to: %s" % out_path, flush=True)
    return manifest, out_path, bool(args.dry_run)


if __name__ == "__main__":
    _manifest, _out_path, _dry = main()
    _outcome = str(_manifest["outcome"]).upper()
    emit_outcome(
        outcome=_outcome if _outcome in ("PASS", "FAIL") else "FAIL",
        manifest_path=_out_path,
        run_id=_manifest["run_id"],
        queue_id=QUEUE_ID,
        dry_run=_dry,
    )
