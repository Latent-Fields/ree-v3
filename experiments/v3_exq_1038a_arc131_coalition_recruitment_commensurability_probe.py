"""V3-EXQ-1038a -- ARC-131 coalition recruitment under CHANNEL COMMENSURABILITY (diagnostic).

Does E3's divisive channel normalisation collapse the 600x cross-seed spread in E3 candidate
margins that made V3-EXQ-1038's recruitment rate SCALE-determined rather than
CONFLICT-determined -- and therefore, is a scale-relative coalition threshold still owed on
SD-091?

  MANIPULATION: ONE swept factor -- `E3Config.use_e3_channel_commensurability` OFF (exactly
                V3-EXQ-1038's condition) vs ON. No new substrate code: the operator already
                rescales the very scores the endogenous trigger reads.
  HELD FIXED:   the 7 seeds, the agent/env construction, the shipped default
                `endogenous_coalition_margin_threshold = 0.05` (never overridden in the main
                measurement), the 1e6 guaranteed-fire readiness control, and the diagnostic
                counter this run reads. All inherited from V3-EXQ-1038's design.
  CHANGED, and ONLY for a stated instrument reason: `TICKS_PER_EPISODE` 100 -> 250 and
                `N_EPISODES` 30 -> 20. See "THE DV HEADROOM FIX" below -- V3-EXQ-1038's
                per-episode request count was arithmetically CENSORED at 2.

EXPERIMENT_PURPOSE = "diagnostic". It reports a measurement and routes a BUILD decision on
SD-091; it is not claim-confidence-bearing. ARC-131 is carried as a read-across co-tag exactly
as V3-EXQ-1038 carried it -- this run exercises ARC-131's own illustrative case (coalition
control: a mechanism that passes isolated wiring tests can still stay dormant, or fire for the
wrong reason, once composed) and is the direct successor to the run that instantiated it.

SLEEP DRIVER: not applicable -- no sleep flag is set. Recorded as sleep_driver_pattern="none".

red-team: see the RED-TEAM RECORD at the end of this docstring and the queue entry note.

Authority: CONFIRMED autopsy failure_autopsy_V3-EXQ-1038_2026-09-14 (+ .json), user gate
2026-09-16T12:22:00Z, applied by governance-20260916 (REE_assembly db6d20ebee: substrate_queue
SD-091 amended, node_class `complex (probe-gated)` GATED ON THIS RUN). SD-091's own next
decision waits on this measurement; this design is not blocked by SD-091, it unblocks it.

=== WHAT V3-EXQ-1038 ESTABLISHED, AND THE PROBLEM IT LEFT ===

The endogenous trigger is live and fires at the shipped default on 7/7 seeds, and the 1e6
control fires on every seed. But (numbers read off that run's own manifest,
v3_exq_1038_arc131_coalition_endogenous_recruitment_rate_probe_20260914T201122Z_v3.json):

  seed              0        1        2        3        4        5        6
  median E3 margin  0.1390  11.188   0.1881   3.8228   2.2118   0.01847  1.6429
  recruit rate/ep   1.233    0.133   1.333    0.133    0.100    2.000    0.233

The per-seed median margin spans 605.7x (11.188 / 0.01847). Recruitment tracks the fraction of
margin samples falling below the FIXED 0.05 almost deterministically -- the seeds with small
margins recruit constantly, the seeds with large margins barely recruit. That is a SCALE
reading, not a CONFLICT reading, and it is why SD-091's amended entry asks whether a
scale-relative threshold is owed.

`use_e3_channel_commensurability` is the substrate's existing answer to exactly that shape: a
divisive normalisation over E3's channel terms (`e3_selector.py` ~:1509 per-channel path and
~:3060 selection-level path), applied to the scores the trigger reads. If it suffices, the
spread collapses and no new threshold machinery is owed. If it does not, the build is owed.

=== THE DV: ELIGIBILITY FRACTION, NOT REQUEST COUNT (red-team BLOCKING -> DV changed) ===

V3-EXQ-1038 reported recruitment as a per-episode REQUEST COUNT, and that count is RATE-CAPPED
by the coalition debounce, not by the agent's disposition to recruit. The mechanism:
`CoalitionController.should_dissolve` is timeout-only (`coalition_controller.py:143`,
`(current_tick - opened_tick) >= max_duration_ticks`, default 50) and the endogenous trigger
runs BEFORE `coalition.tick()` inside the same `select_action` (`agent.py`, the trigger block
followed by the comment "coalition.tick(current_tick) runs after"), so the earliest re-request
is 51 ticks after the last -- and E3 only evaluates every `e3_steps_per_tick = 10` env steps, so
requests land on a 10-tick grid.

The consequence is structural and does not go away by lengthening the episode: the maximum
ATTAINABLE count and the arithmetic ceiling both scale as `ticks // ~51`, so a seed that is
eligible on most ticks sits ON the cap at ANY window length. V3-EXQ-1038's seed 5 pinned at its
100-tick ceiling of 2 in all 30 episodes for exactly this reason. MEASURED here at 250 ticks
(seed 5, 2 episodes, this driver): comm_off counts [4, 4] against a ceiling of 5 -- i.e. still
hard against the cap, with no headroom to spare. A first draft of this design GATED on that
count and would therefore have self-routed the whole run to `substrate_not_ready_requeue` on a
MECHANISM RATE CAP misread as an instrument defect, on a fix ("requeue with a longer window")
that arithmetically cannot work.

**So the recruitment DV is the ELIGIBILITY FRACTION** -- the fraction of FRESH E3 margin samples
falling below the shipped `endogenous_coalition_margin_threshold = 0.05`. It is uncapped,
continuous, and is precisely the quantity V3-EXQ-1038's own finding was about ("recruitment
tracks the fraction of margin samples below the fixed 0.05 almost deterministically"). It also
moves under the manipulation: MEASURED on seed 5 at 250 ticks, comm_off 0.938 vs comm_on 0.328.

The per-episode request count and `frac_episodes_recruited` are still RECORDED per cell, as
context and for continuity with V3-EXQ-1038, and the manifest carries an explicit
`request_count_rate_cap` block saying why nothing routes on them. `TICKS_PER_EPISODE = 250` is
kept -- not as headroom for the count, which is unobtainable, but because ~25 E3 evaluations per
episode is what makes the eligibility fraction a well-sampled per-cell statistic.

=== THE ENABLE PATH, AND THE SILENT-SWALLOW HAZARD IT WALKS PAST ===

MEASURED at authoring time, not assumed. `use_e3_channel_commensurability` is a field of
**`E3Config`** (`ree_core/utils/config.py:1073`), NOT of `REEConfig`, and it is NOT a parameter
of `REEConfig.from_dims`. So the obvious spelling

    REEConfig.from_dims(..., use_e3_channel_commensurability=True)   # SILENTLY SWALLOWED

returns a config on which the attribute does not exist at all -- the
`reference-reeconfig-from-dims-silent-kwargs` hazard, and it would have made the ON arm a
bit-identical copy of the OFF arm with no error anywhere. Setting it on the top-level
`REEConfig` object is ALSO a no-op, because `REEAgent` constructs the selector from
`config.e3` (`ree_core/agent.py:409`) and `E3TrajectorySelector` reads
`self.config.use_e3_channel_commensurability` off that sub-config.

The only enable path that reaches the operator is `cfg.e3.use_e3_channel_commensurability`, and
`_build()` ASSERTS it landed on `agent.e3.config` rather than trusting that it did. Probed
(seed 0, 120 ticks, otherwise identical agents): OFF median margin 0.2126 / 1 request with
`e3._last_commensurability_raw == {}`; ON median margin 0.1556 / 2 requests with
`_last_commensurability_raw` carrying `f_weighted, harm_weighted, residue_weighted,
benefit_weighted, goal_weighted`. The operator engages, and it moves the statistic the trigger
reads.

=== THE WARMUP NO-OP: THE HAZARD THIS DESIGN NEARLY WALKED INTO ===

FOUND BY THE SMOKE, and now gated. `_commensurability_scale` (e3_selector.py:1362) returns
**1.0 -- an exact no-op -- while `self._chan_scale_n < config.e3_commensurability_warmup_ticks`**
(default 20). The operator still populates `_last_commensurability_raw` during that warmup, so
the "did the manipulation engage?" witness reads TRUE while every channel term is being divided
by 1.0 and the ON arm is BIT-IDENTICAL to OFF.

The first smoke of this driver hit exactly that: 2 x 60 ticks gave ~16 E3 evaluations per cell,
below the warmup of 20, and ON and OFF returned the SAME median margin to the last bit
(0.21258544921875 on seed 0) with `on_margin_spread_ratio == off_margin_spread_ratio` and
`on_cv == off_cv` -- a DV-symmetry arithmetic identity, not a measurement, which a reader could
easily have taken for "normalisation changes nothing".

Measured resolution (seed 0, fresh matched envs, `_chan_scale_n` read off the selector):

    600 ticks  -> chan_scale_n 75   OFF median 0.13548 vs ON 0.17944   DIFFER
   1500 ticks  -> chan_scale_n 188  OFF median 0.10913 vs ON 0.21338   DIFFER

So the manipulation DOES reach the DV, but only past the warmup. `E3TrajectorySelector` has no
`reset()` and `REEAgent.reset()` does not clear `_chan_scale_n`, so the counter ACCUMULATES
across the episodes of a cell: at 20 x 250 ticks (~500 E3 evaluations/cell) the warmup is
crossed inside episode 1 and ~4% of samples are pre-warmup. Two gates make this auditable
rather than assumed:

  G1b  the minimum `_chan_scale_n` over the ON cells must reach the configured warmup
  G1c  the ON and OFF median margins must DIFFER on EVERY seed -- the direct, mechanical form
       of this file's DV-symmetry declaration. If they are equal the manipulation was
       arithmetically invisible to the DV and the run measured nothing.

`DRY_RUN_TICKS_PER_EPISODE = 150` (2 episodes, ~30 E3 evaluations) is set so the SMOKE itself
crosses the warmup and exercises the normalised path, rather than silently testing the no-op.

=== WHAT FALSIFIES WHAT -- PRE-REGISTERED, BOTH DIRECTIONS DECLARED ===

Preconditions. G0-G2 are INSTRUMENT gates: any red one self-routes `substrate_not_ready_requeue`
and produces no substrate verdict. G3/G3a/G3b are PREMISE gates and are handled differently --
see "A PREMISE FAILURE IS NOT AN INSTRUMENT FAILURE" below.

  G0  every (arm x seed) 1e6 control burst fires        -- the trigger path works in both arms
  G1  `_last_commensurability_raw` is non-empty on EVERY ON cell and empty on EVERY OFF cell
      -- the manipulation engaged, and ONLY on the arm that asked for it
  G1b every ON cell's commensurability warmup completed -- the operator was not a 1.0 no-op
  G1c the ON and OFF median margins DIFFER on every seed -- the manipulation is not invisible
      to the DV by arithmetic
  G2  every cell collected enough FRESH E3 margin samples for its eligibility fraction to be a
      real statistic (the request count is rate-capped and nothing routes on it)
  G3  the OFF arm's OWN cross-seed median-margin ratio is >= 10.0 -- without spread in the
      reference arm there is nothing for normalisation to collapse and the question is vacuous

Finding criteria. These route the LABEL, never the outcome (outcome PASS means "a valid,
instrument-verified measurement was obtained", exactly as in V3-EXQ-1038):

  F1  ON cross-seed median-margin ratio  <  MARGIN_SPREAD_RATIO_BAR = 10.0
      An ABSOLUTE pre-registered constant, fixed before the run from V3-EXQ-1038's published
      605.7x. Not derived from this run's own statistics.
  F2  ON |Spearman rho| between per-seed MEDIAN E3 MARGIN (the score SCALE) and per-seed
      ELIGIBILITY FRACTION (recruitment)  <  SCALE_ASSOCIATION_COLLAPSED_BAR = 0.4
      This IS the question -- "is recruitment scale-determined?" -- measured directly, against
      an ABSOLUTE constant computed from V3-EXQ-1038's own published per-seed values, which
      give |rho| = 0.830 under the OFF condition. `G3b_off_arm_is_scale_determined` requires
      the in-run OFF arm to clear 0.6, so the test cannot collapse vacuously.

      IT REPLACES an eligibility-CV halving test against the in-run OFF arm, which the second
      red-team pass showed is not a statement about the operator at all: the OFF eligibility
      fractions are BIMODAL at the fixed 0.05 threshold, and the CV of a bimodal 7-vector is a
      function of how many seeds land in the high mode (k=1 -> 2.27, k=3 -> 1.13, k=6 -> 0.40),
      so the bar would have spanned 5.7x across admissible references and identical ON data
      could route to opposite labels. That test is RETAINED in the criteria block as
      `F2c_eligibility_cv_halved_CONTEXT_ONLY`, recorded and not routed on.

      The rho form also fixes a second defect the CV form could not: if normalisation WORKS,
      the residual cross-seed eligibility spread is whatever genuine conflict heterogeneity the
      7 layouts carry -- SD-091's DESIRED end state. A CV criterion reads that as "did not
      halve" and routes to a build; rho reads it as ~0, i.e. scale removed. The design does not
      establish that the seeds are conflict-homogeneous, and with rho it does not need to.

  F1 AND F2  -> `commensurability_collapses_scale_spread`
               ROUTING: normalisation suffices; the scale-relative threshold build on SD-091 is
               NOT owed. Mark the probe sufficient.
  neither    -> `scale_spread_persists_under_commensurability`
               ROUTING: the existing operator does NOT make recruitment scale-invariant. TWO
               readings remain open and this run does NOT separate them (red-team F2):
               (i) the coalition threshold itself needs to be scale-relative -- the SD-091 build;
               (ii) the operator's own divisor is the problem -- it normalises each channel by an
               EMA of that channel's CROSS-CANDIDATE SD (e3_selector.py:1362-1377, floor 1e-12),
               which does not normalise the SUMMED score's top-2 gap and can AMPLIFY an
               intermittent low-SD channel. Measured at authoring time, the ON-arm divisors span
               orders of magnitude (seed 0: residue 5.3e-4 vs f 0.43), and within-cell margin
               dispersion GREW under ON.
               So the routing is: /failure-autopsy adjudicates (i) vs (ii) using the
               `within_cell_margin_dispersion` and `chan_scale_ema_final` this run records --
               NOT an automatic SD-091 build.
  exactly one-> `commensurability_partially_collapses_scale_spread`
               ROUTING: back to /failure-autopsy to adjudicate which half moved and why; NOT an
               automatic build and NOT an automatic close.

A null here (spread persists) is an INFORMATIVE result that discharges SD-091's probe gate. It
is not a wasted cycle and must not be read as one.

=== A PREMISE FAILURE IS NOT AN INSTRUMENT FAILURE (red-team pass 2, finding 3) ===

If only G3 / G3a / G3b are red while G0-G2 are green, NOTHING is broken: the instrument works
and this run's comm_off arm simply did not reproduce V3-EXQ-1038's scale-determined premise. A
requeue cannot change that, and it IS evidence about SD-091 -- whose amended entry rests on that
premise. That case gets its own label, `premise_not_reproduced_off_arm_not_scale_determined`,
and routes to /failure-autopsy rather than to a requeue. The regime difference is real and named
in the routing: this run uses 20 x 250 ticks per cell against V3-EXQ-1038's 30 x 100, and E3
accumulator state is never reset within a cell.

=== THE POSITIVE CONTROL CROSSES THE WARMUP (red-team pass 2, finding 4) ===

`CONTROL_TICKS = 250`, not V3-EXQ-1038's 40. At `e3_steps_per_tick = 10` a 40-tick burst is 4
E3 evaluations, leaving `_chan_scale_n` below the warmup of 20, so the ON control would have run
entirely on UN-normalised scores -- certifying a path the main measurement never uses. Each cell
also records `e3_margin_n_nonfinite`: a non-finite normalised score would deflate eligibility
(`m < 0.05` is False for NaN) while still passing G1c (NaN != NaN), which would otherwise be
attributed to the operator "pushing margins above threshold".

=== CAVEAT THAT MUST TRAVEL WITH ANY READING OF THIS RUN ===

The commensurability operator's OWN validation is **OPEN** (V3-EXQ-1012a: the instrument is
sound, but no reference value has been established). This run therefore EXERCISES an operator
whose validation is open; it does not rely on a validated one. Concretely: if F1/F2 hold, the
correct statement is "the operator as currently implemented collapses the spread", NOT "a
validated normalisation collapses the spread". Recorded in the manifest as
`operator_validation_status` so a later reader cannot lose it.

=== MACHINE CLASS ===

E3 selection uses `torch.multinomial`, so per-seed values are not bit-identical across
darwin-arm64 / linux-x86_64. `machine_affinity: any` remains appropriate: every criterion is a
cross-seed SPREAD or a within-run PAIRED ratio, never an exact committed action sequence.

=== NO ARM-REUSE MINT ===

Deliberate. Every cell here is a FRESH, UNTRAINED agent whose whole cost is rollout, not
training, so there is no expensive trained artifact for a successor to reuse; factoring an
`_lib/baselines/` module would add a maintenance surface and save nothing. Cells are stamped
with the default driver-inclusive hash (Phase-0 emit only) and are independent -- a fresh agent
and a fresh env per cell, with a complete RNG reset at cell entry.

=== RED-TEAM RECORD (Step 4.5) ===

See the queue entry note for the verdict, the reviewing model, and the disposition of every
finding.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiment_protocol import emit_outcome  # noqa: E402
from experiments._lib.arm_fingerprint import arm_cell  # noqa: E402
from experiments._lib.z_goal_stream import ZGoalStreamAccumulator  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402

_ZG = ZGoalStreamAccumulator()

ANCHOR_REACHABILITY_EXEMPT = (
    "G0's control (margin_threshold=1e6) IS the degeneracy definition, not a hand-tuned "
    "approximation of it: it is the exact value contract test W11 "
    "(tests/contracts/test_sd091_coalition_controller_wiring.py) uses to prove the driver fires "
    "on essentially every eligible tick -- reachable by construction. G1/G2/G3 are likewise "
    "definitional rather than anchored: G1 asks whether a dict the operator itself populates is "
    "non-empty, G2 is pure arithmetic against the ceiling TICKS_PER_EPISODE // "
    "coalition_max_duration_ticks, and G3's 10.0 floor is 60x BELOW the 605.7x spread "
    "V3-EXQ-1038 published on these same 7 seeds, so the reference arm clears it with two "
    "orders of magnitude to spare. Inherited unchanged from V3-EXQ-1038's own exemption."
)

EXPERIMENT_TYPE = "v3_exq_1038a_arc131_coalition_recruitment_commensurability_probe"
QUEUE_ID = "V3-EXQ-1038a"
BACKLOG_ID = "EVB-1242"
CLAIM_IDS: List[str] = ["ARC-131"]
EXPERIMENT_PURPOSE = "diagnostic"
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"

SEEDS = [0, 1, 2, 3, 4, 5, 6]
# THREE seeds, not two: F2's Spearman statistic is undefined below n=3, so a 2-seed smoke
# would leave the criterion the run routes on entirely unexercised and self-route on the
# premise gate instead.
DRY_RUN_SEEDS = [0, 1, 2]

ARM_OFF = "comm_off"
ARM_ON = "comm_on"
ARMS = [ARM_OFF, ARM_ON]

# Main measurement, at the config-shipped DEFAULT threshold. Never overridden. (V3-EXQ-1038.)
DEFAULT_MARGIN_THRESHOLD = 0.05
N_EPISODES = 20
DRY_RUN_N_EPISODES = 2
# See "THE DV HEADROOM FIX". ceiling = TICKS_PER_EPISODE // coalition_max_duration_ticks.
TICKS_PER_EPISODE = 250
DRY_RUN_TICKS_PER_EPISODE = 150
COALITION_MAX_DURATION_TICKS = 50          # the shipped default, NOT overridden
REQUEST_COUNT_CEILING = TICKS_PER_EPISODE // COALITION_MAX_DURATION_TICKS      # = 5
DRY_RUN_REQUEST_COUNT_CEILING = DRY_RUN_TICKS_PER_EPISODE // COALITION_MAX_DURATION_TICKS

# Readiness positive control, matching contract test W11 exactly. (V3-EXQ-1038.)
CONTROL_MARGIN_THRESHOLD = 1e6
# Red-team pass 2, finding 4: at e3_steps_per_tick=10 a 40-tick burst is only 4 E3 evaluations,
# so _chan_scale_n stays below the warmup of 20 and the ON control would have run entirely on
# UN-normalised scores -- certifying a path the main measurement does not use. 250 ticks gives
# ~25 evaluations and crosses the warmup.
CONTROL_TICKS = 250

# Env / agent construction -- identical to V3-EXQ-1038 and the W9-W13 contract harness.
GRID_SIZE = 5
NUM_HAZARDS = 1
NUM_RESOURCES = 1
ACTION_DIM = 4
SELF_DIM = 16
WORLD_DIM = 16

# ---- PRE-REGISTERED BARS. Constants, fixed before the run. --------------------------------
# F1: an ABSOLUTE bar from V3-EXQ-1038's PUBLISHED 605.7x cross-seed median-margin ratio.
MARGIN_SPREAD_RATIO_BAR = 10.0
# F2: the SCALE-ASSOCIATION statistic -- |Spearman rho| between a seed's score SCALE (its median
# E3 margin) and its recruitment (its eligibility fraction). This is the claim itself: "is
# recruitment scale-determined?". ABSOLUTE pre-registered bars, computed from V3-EXQ-1038's own
# published per-seed values, which give |rho| = 0.830 under the OFF condition.
SCALE_ASSOCIATION_COLLAPSED_BAR = 0.4    # ON below this -> the association collapsed
G3B_OFF_SCALE_ASSOCIATION_FLOOR = 0.6    # OFF above this -> the reference really is scale-bound
# Kept and RECORDED as context only. Red-team pass 2, finding 1: the eligibility CV of a BIMODAL
# 7-vector is a function of HOW MANY seeds land in the high mode (k=1 -> 2.27, k=3 -> 1.13,
# k=6 -> 0.40), so a halving bar denominated on it is a statement about k, not about the
# operator. Nothing routes on it.
CV_HALVING_RATIO_CONTEXT_ONLY = 0.5
# G0: every control cell must fire.
G0_CONTROL_ALL_CELLS_FIRE_THRESHOLD = 1.0
# G1: the operator engaged on every ON cell and on no OFF cell.
G1_MANIPULATION_ENGAGED_THRESHOLD = 1.0
# G3: the reference arm must actually carry the spread this run asks about.
G3_OFF_MARGIN_RATIO_FLOOR = 10.0
# G3a: and a non-flat eligibility-fraction CV, so the reference arm is not degenerate.
G3A_OFF_CV_FLOOR = 0.1
# G2: each cell must collect at least this FRACTION of the E3 evaluations its schedule implies,
# so the per-cell eligibility fraction is a real statistic rather than a handful of samples.
G2_MIN_SAMPLE_FRACTION = 0.5
E3_STEPS_PER_TICK = 10          # ree_core/heartbeat/clock.py -- the E3 evaluation cadence
# F3: "the ON arm eliminated eligibility altogether" is a DISTINCT outcome from "the CV did not
# halve", and a CV is undefined at a zero mean. Below this the ON arm counts as eliminated.
ELIGIBILITY_ELIMINATED_CEILING = 0.01

OPERATOR_VALIDATION_STATUS = {
    "operator": "E3Config.use_e3_channel_commensurability",
    "validation": "OPEN",
    "reference": ("V3-EXQ-1012a -- instrument sound, no reference value established. This run "
                  "EXERCISES an operator whose validation is open; it does not rely on a "
                  "validated one. A finding of 'spread collapses' licenses 'the operator AS "
                  "CURRENTLY IMPLEMENTED collapses the spread', never 'a validated "
                  "normalisation collapses the spread'."),
}


# --------------------------------------------------------------------------------------
def _build(seed: int, margin_threshold: float, commensurability: bool) -> Tuple[Any, Any]:
    torch.manual_seed(seed)
    env = CausalGridWorldV2(
        seed=seed, size=GRID_SIZE, num_hazards=NUM_HAZARDS,
        num_resources=NUM_RESOURCES, use_proxy_fields=True,
    )
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=ACTION_DIM,
        self_dim=SELF_DIM,
        world_dim=WORLD_DIM,
        use_coalition_controller=True,
        use_endogenous_coalition_trigger=True,
        endogenous_coalition_margin_threshold=margin_threshold,
        endogenous_coalition_demand_type="sensory_resample",
    )
    # THE SWEPT FACTOR. It is an E3Config field and is NOT a from_dims parameter -- passing it
    # to from_dims is SILENTLY SWALLOWED, and setting it on the top-level REEConfig is a no-op
    # because REEAgent builds the selector from config.e3 (ree_core/agent.py:409). See the
    # docstring's "THE ENABLE PATH" section.
    cfg.e3.use_e3_channel_commensurability = bool(commensurability)
    torch.manual_seed(1000 + seed)
    agent = REEAgent(cfg)
    agent.reset()

    # Assert the flag REACHED the selector rather than trusting that it did: a silent swallow
    # here would make the ON arm a bit-identical copy of OFF and the whole run vacuous.
    reached = getattr(agent.e3.config, "use_e3_channel_commensurability", None)
    if bool(reached) != bool(commensurability):
        raise RuntimeError(
            "use_e3_channel_commensurability did not reach E3TrajectorySelector: asked %r, "
            "agent.e3.config reports %r. The ON arm would be identical to OFF. Refusing to run."
            % (bool(commensurability), reached))
    # The shipped duration knob must be the default -- REQUEST_COUNT_CEILING is computed from
    # it, so a drifted value would silently move the headroom gate G2.
    got_dur = int(getattr(cfg, "coalition_max_duration_ticks", -1))
    if got_dur != int(COALITION_MAX_DURATION_TICKS):
        raise RuntimeError(
            "coalition_max_duration_ticks is %r, expected the shipped default %d -- "
            "REQUEST_COUNT_CEILING and gate G2 are computed from it."
            % (got_dur, COALITION_MAX_DURATION_TICKS))
    return agent, env


def _obs(od: Dict[str, Any]) -> Tuple[Any, Any]:
    b = od["body_state"]
    w = od["world_state"]
    if b.dim() == 1:
        b = b.unsqueeze(0)
    if w.dim() == 1:
        w = w.unsqueeze(0)
    return b, w


def _comm_raw_keys(agent: Any) -> List[str]:
    """The channel terms the commensurability operator itself populated, if any.

    Non-empty is the DIRECT, machine-readable witness that the operator engaged on this cell
    (probed at authoring time: f_weighted / harm_weighted / residue_weighted / benefit_weighted
    / goal_weighted under ON; empty dict under OFF).
    """
    raw = getattr(getattr(agent, "e3", None), "_last_commensurability_raw", None)
    if isinstance(raw, dict):
        return sorted(str(k) for k in raw.keys())
    return []


def _config_slice(arm: str, seed: int, sched: Dict[str, int]) -> Dict[str, Any]:
    return {
        "arm_id": arm,
        "use_e3_channel_commensurability": bool(arm == ARM_ON),
        "endogenous_coalition_margin_threshold": float(DEFAULT_MARGIN_THRESHOLD),
        "coalition_max_duration_ticks": int(COALITION_MAX_DURATION_TICKS),
        "n_episodes": int(sched["n_episodes"]),
        "ticks_per_episode": int(sched["ticks"]),
        "control_ticks": int(CONTROL_TICKS),
        "control_margin_threshold": float(CONTROL_MARGIN_THRESHOLD),
        "grid_size": GRID_SIZE, "num_hazards": NUM_HAZARDS, "num_resources": NUM_RESOURCES,
        "action_dim": ACTION_DIM, "self_dim": SELF_DIM, "world_dim": WORLD_DIM,
        "seed": int(seed),
    }


def _run_control_burst(seed: int, commensurability: bool) -> Dict[str, Any]:
    """Positive-control burst at margin_threshold=1e6 (matches contract test W11)."""
    agent, env = _build(seed, CONTROL_MARGIN_THRESHOLD, commensurability)
    _flat, od = env.reset()
    b, w = _obs(od)
    for _ in range(CONTROL_TICKS):
        with torch.no_grad():
            action = agent.act_with_split_obs(b, w)
        _flat, _harm, done, _info, od = env.step(action)
        b, w = _obs(od)
        if done:
            _flat, od = env.reset()
            b, w = _obs(od)
    out = {
        "seed": int(seed),
        "arm_id": ARM_ON if commensurability else ARM_OFF,
        "control_request_count": int(agent._endogenous_coalition_request_count),
        "control_fired": bool(agent._endogenous_coalition_request_count > 0),
        "control_comm_raw_keys": _comm_raw_keys(agent),
    }
    _ZG.observe(agent)
    return out


def _run_cell(arm: str, seed: int, sched: Dict[str, int]) -> Dict[str, Any]:
    """One (arm x seed) main measurement at the shipped DEFAULT margin threshold."""
    commensurability = bool(arm == ARM_ON)
    print("Seed %d Condition %s" % (seed, arm), flush=True)
    n_episodes = int(sched["n_episodes"])
    ticks = int(sched["ticks"])
    row: Dict[str, Any] = {}
    with arm_cell(seed, config_slice=_config_slice(arm, seed, sched),
                  script_path=Path(__file__), config_slice_declared=True) as cell:
        agent, env = _build(seed, DEFAULT_MARGIN_THRESHOLD, commensurability)
        warm_ticks = int(getattr(agent.e3.config, "e3_commensurability_warmup_ticks", 20))
        per_episode_counts: List[int] = []
        margin_samples: List[float] = []
        comm_keys_seen: List[str] = []
        n_prewarmup_samples = 0
        n_nonfinite_margins = 0
        for ep in range(n_episodes):
            agent.reset()          # also zeroes _endogenous_coalition_request_count
            last_seen_result_id = None   # _last_e3_selection_result is cleared by reset()
            _flat, od = env.reset()
            b, w = _obs(od)
            for _ in range(ticks):
                # Read the warmup counter BEFORE the tick that may produce this sample --
                # select() increments it, so reading after undercounts the pre-warmup prefix
                # by exactly one sample (red-team pass 2, nit).
                _scale_n_before = int(getattr(agent.e3, "_chan_scale_n", 0))
                with torch.no_grad():
                    action = agent.act_with_split_obs(b, w)
                # De-duplicate by object identity so a tick that ran NO fresh E3 selection
                # (the attribute LATCHES the previous tick's result -- CLAUDE.md
                # "Sample-size integrity") is never counted as a new observation.
                result = agent._last_e3_selection_result
                if result is not None and id(result) != last_seen_result_id:
                    last_seen_result_id = id(result)
                    try:
                        scores = result.scores.detach()
                        if int(scores.numel()) >= 2:
                            sorted_scores, _ = torch.sort(scores)
                            margin_samples.append(
                                float(sorted_scores[1].item() - sorted_scores[0].item()))
                            # A sample taken before the operator's warmup completed was
                            # scored with an exact 1.0 divisor -- recorded so the
                            # pre-warmup prefix is visible rather than inferred.
                            if _scale_n_before < warm_ticks:
                                n_prewarmup_samples += 1
                            if not math.isfinite(margin_samples[-1]):
                                n_nonfinite_margins += 1
                    except (AttributeError, RuntimeError, TypeError):
                        pass
                _flat, _harm, done, _info, od = env.step(action)
                b, w = _obs(od)
                if done:
                    _flat, od = env.reset()
                    b, w = _obs(od)
            per_episode_counts.append(int(agent._endogenous_coalition_request_count))
            if not comm_keys_seen:
                comm_keys_seen = _comm_raw_keys(agent)
            if (ep + 1) % 5 == 0 or ep == 0:
                print("  [train] seed=%d arm=%s ep %d/%d" % (seed, arm, ep + 1, n_episodes),
                      flush=True)

        n_recruited = sum(1 for c in per_episode_counts if c > 0)
        sorted_margins = sorted(margin_samples)
        row.update({
            "arm_id": arm,
            "seed": int(seed),
            "commensurability_on": bool(commensurability),
            "n_episodes": n_episodes,
            "ticks_per_episode": ticks,
            "per_episode_request_counts": per_episode_counts,
            "max_per_episode_request_count": (max(per_episode_counts)
                                              if per_episode_counts else 0),
            "episodes_recruited": n_recruited,
            "frac_episodes_recruited": n_recruited / len(per_episode_counts),
            "mean_request_rate_per_episode": (sum(per_episode_counts)
                                              / len(per_episode_counts)),
            "total_requests": sum(per_episode_counts),
            "e3_margin_samples": margin_samples,
            "e3_margin_n_samples": len(margin_samples),
            "e3_margin_median": (sorted_margins[len(sorted_margins) // 2]
                                 if sorted_margins else None),
            "e3_margin_mean": (sum(margin_samples) / len(margin_samples)
                               if margin_samples else None),
            "e3_margin_min": (sorted_margins[0] if sorted_margins else None),
            "e3_margin_max": (sorted_margins[-1] if sorted_margins else None),
            # THE RECRUITMENT DV. Uncapped, unlike the per-episode request count, and exactly
            # the quantity V3-EXQ-1038 found recruitment to track.
            "eligibility_fraction": ((sum(1 for m in margin_samples
                                          if m < DEFAULT_MARGIN_THRESHOLD)
                                      / len(margin_samples)) if margin_samples else None),
            # WITHIN-cell dispersion, recorded so /failure-autopsy can separate "the threshold
            # needs to be scale-relative" from "the operator amplifies an intermittent
            # low-SD channel" on a NEITHER outcome (red-team F2).
            "within_cell_margin_dispersion": (
                (float(sorted_margins[-1]) / float(sorted_margins[len(sorted_margins) // 2]))
                if sorted_margins and float(sorted_margins[len(sorted_margins) // 2]) > 0.0
                else None),
            "commensurability_raw_keys": comm_keys_seen,
            "commensurability_engaged": bool(comm_keys_seen),
            "commensurability_warmup_ticks": warm_ticks,
            "chan_scale_n_final": int(getattr(agent.e3, "_chan_scale_n", 0)),
            "chan_scale_ema_final": {str(k): float(v) for k, v in
                                     (getattr(agent.e3, "_chan_scale_ema", {}) or {}).items()},
            "e3_margin_n_samples_prewarmup": int(n_prewarmup_samples),
            # A non-finite normalised score would deflate eligibility (m < 0.05 is False for
            # NaN) while still passing G1c (NaN != NaN) -- recorded so that cannot be
            # attributed to the operator "pushing margins above threshold" (red-team pass 2,
            # finding 4).
            "e3_margin_n_nonfinite": int(n_nonfinite_margins),
        })
        _ZG.observe(agent)
        cell.stamp(row)
    print("  -> arm=%s seed=%d frac_ep=%.3f mean_rate=%.3f median_margin=%r max_count=%d"
          % (arm, seed, row["frac_episodes_recruited"], row["mean_request_rate_per_episode"],
             row["e3_margin_median"], row["max_per_episode_request_count"]), flush=True)
    print("verdict: %s" % ("PASS" if row["e3_margin_n_samples"] > 0 else "FAIL"), flush=True)
    return row


# --------------------------------------------------------------------------------------
# STATISTICS
# --------------------------------------------------------------------------------------
def _cv(values: List[float]) -> Optional[float]:
    """Population coefficient of variation. None when undefined (n<2 or mean ~ 0)."""
    vals = [float(v) for v in values if v is not None]
    if len(vals) < 2:
        return None
    mean = sum(vals) / len(vals)
    if abs(mean) < 1e-12:
        return None
    var = sum((v - mean) ** 2 for v in vals) / len(vals)
    return math.sqrt(var) / abs(mean)


def _rank(values: List[float]) -> List[float]:
    """Average ranks, ties shared -- the standard Spearman tie correction."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _spearman(xs: List[float], ys: List[float]) -> Optional[float]:
    """Spearman rank correlation. None when undefined (n<3, or a constant input)."""
    pairs = [(float(a), float(b)) for a, b in zip(xs, ys) if a is not None and b is not None]
    if len(pairs) < 3:
        return None
    rx = _rank([p[0] for p in pairs])
    ry = _rank([p[1] for p in pairs])
    n = len(pairs)
    mx, my = sum(rx) / n, sum(ry) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    sxx = sum((a - mx) ** 2 for a in rx)
    syy = sum((b - my) ** 2 for b in ry)
    if sxx <= 0.0 or syy <= 0.0:
        return None
    return sxy / math.sqrt(sxx * syy)


def _scale_association(rows: List[Dict[str, Any]]) -> Tuple[Optional[float], Dict[str, Any]]:
    """|Spearman rho| between a seed's median E3 margin (its SCALE) and its eligibility
    fraction (its recruitment). The statistic the whole question is about, and -- unlike a CV
    of a bimodal vector -- not a function of how many seeds sit in the high mode."""
    pts = [(r["seed"], r.get("e3_margin_median"), r.get("eligibility_fraction")) for r in rows]
    usable = [(s, m, e) for s, m, e in pts if m is not None and e is not None]
    rho = _spearman([m for _s, m, _e in usable], [e for _s, _m, e in usable])
    return ((abs(rho) if rho is not None else None),
            {"rho_signed": rho, "n_seeds": len(usable),
             "per_seed": {str(s): {"median_margin": m, "eligibility_fraction": e}
                          for s, m, e in usable}})


def _margin_spread_ratio(rows: List[Dict[str, Any]]) -> Tuple[Optional[float], Dict[str, Any]]:
    """max/min of the per-seed MEDIAN E3 margin -- the 605.7x statistic V3-EXQ-1038 reported."""
    meds = [(r["seed"], r["e3_margin_median"]) for r in rows
            if r.get("e3_margin_median") is not None and float(r["e3_margin_median"]) > 0.0]
    if len(meds) < 2:
        return None, {"n_usable_seeds": len(meds), "reason": "fewer than 2 usable seeds"}
    lo_seed, lo = min(meds, key=lambda t: float(t[1]))
    hi_seed, hi = max(meds, key=lambda t: float(t[1]))
    return float(hi) / float(lo), {
        "n_usable_seeds": len(meds),
        "min_seed": int(lo_seed), "min_median_margin": float(lo),
        "max_seed": int(hi_seed), "max_median_margin": float(hi),
        "per_seed_median_margin": {str(s): (float(m) if m is not None else None)
                                   for s, m in meds},
    }


def _arm_rows(rows: List[Dict[str, Any]], arm: str) -> List[Dict[str, Any]]:
    return [r for r in rows if r.get("arm_id") == arm]


# --------------------------------------------------------------------------------------
def run_experiment(dry_run: bool = False) -> Dict[str, Any]:
    seeds = DRY_RUN_SEEDS if dry_run else SEEDS
    sched = {
        "n_episodes": DRY_RUN_N_EPISODES if dry_run else N_EPISODES,
        "ticks": DRY_RUN_TICKS_PER_EPISODE if dry_run else TICKS_PER_EPISODE,
    }
    ceiling = DRY_RUN_REQUEST_COUNT_CEILING if dry_run else REQUEST_COUNT_CEILING

    control_rows: List[Dict[str, Any]] = []
    rows: List[Dict[str, Any]] = []
    for arm in ARMS:
        for seed in seeds:
            print("Seed %d Condition %s:control" % (seed, arm), flush=True)
            control_rows.append(_run_control_burst(seed, bool(arm == ARM_ON)))
            rows.append(_run_cell(arm, seed, sched))

    # ---- G0: the trigger path works in BOTH arms ------------------------------------------
    n_control_fired = sum(1 for r in control_rows if r["control_fired"])
    control_fire_fraction = n_control_fired / len(control_rows)

    # ---- G1: the manipulation engaged on the ON arm ONLY -----------------------------------
    on_rows, off_rows = _arm_rows(rows, ARM_ON), _arm_rows(rows, ARM_OFF)
    n_expected = len(on_rows) + len(off_rows)
    n_correct = (sum(1 for r in on_rows if r["commensurability_engaged"])
                 + sum(1 for r in off_rows if not r["commensurability_engaged"]))
    engaged_fraction = (n_correct / n_expected) if n_expected else 0.0

    # ---- G1b: the operator's warmup completed on every ON cell (else it divided by 1.0) -----
    warm_required = max([int(r.get("commensurability_warmup_ticks") or 0) for r in on_rows]
                        or [0])
    on_scale_ns = [(int(r.get("chan_scale_n_final") or 0), "%s@seed%s" % (r["arm_id"], r["seed"]))
                   for r in on_rows]
    worst_scale_n, worst_scale_cell = (min(on_scale_ns) if on_scale_ns else (0, None))

    # ---- G1c: the manipulation is NOT invisible to the DV by arithmetic ---------------------
    # The mechanical form of this run's DV-symmetry declaration. Equal medians at the same seed
    # means the normalisation could not move the statistic the trigger reads, so F1/F2 would be
    # identities fixed before the run rather than measurements.
    off_by_seed = {int(r["seed"]): r.get("e3_margin_median") for r in off_rows}
    n_seeds_dv_moved, dv_identical_seeds = 0, []
    for r in on_rows:
        o = off_by_seed.get(int(r["seed"]))
        n = r.get("e3_margin_median")
        if o is not None and n is not None and float(o) != float(n):
            n_seeds_dv_moved += 1
        else:
            dv_identical_seeds.append(int(r["seed"]))
    n_paired_seeds = len(on_rows)
    dv_moved_fraction = (n_seeds_dv_moved / n_paired_seeds) if n_paired_seeds else 0.0

    # ---- G2: the DV is well sampled (worst cell, not the mean) ------------------------------
    # NOT a gate on the per-episode request count: that count is RATE-CAPPED by the 51-tick
    # coalition debounce (see the docstring), so gating on it would red the run on a mechanism
    # cap misread as an instrument defect. The DV is the eligibility fraction; this gates its
    # sample size.
    expected_samples = int(sched["n_episodes"]) * (int(sched["ticks"]) // E3_STEPS_PER_TICK)
    sample_floor = float(G2_MIN_SAMPLE_FRACTION) * float(expected_samples)
    sample_ns = [(int(r["e3_margin_n_samples"]), "%s@seed%s" % (r["arm_id"], r["seed"]))
                 for r in rows]
    worst_samples, worst_samples_cell = (min(sample_ns) if sample_ns else (0, None))
    max_counts = [(r["max_per_episode_request_count"], "%s@seed%s" % (r["arm_id"], r["seed"]))
                  for r in rows]
    worst_count, worst_count_cell = (max(max_counts) if max_counts else (0, None))

    # ---- G3 / G3a: the reference arm carries the spread the question is about ---------------
    off_ratio, off_ratio_detail = _margin_spread_ratio(off_rows)
    on_ratio, on_ratio_detail = _margin_spread_ratio(on_rows)
    # THE RECRUITMENT DV: cross-seed spread of the ELIGIBILITY FRACTION (uncapped).
    off_elig = [r["eligibility_fraction"] for r in off_rows if r["eligibility_fraction"] is not None]
    on_elig = [r["eligibility_fraction"] for r in on_rows if r["eligibility_fraction"] is not None]
    off_cv, on_cv = _cv(off_elig), _cv(on_elig)
    off_assoc, off_assoc_detail = _scale_association(off_rows)
    on_assoc, on_assoc_detail = _scale_association(on_rows)
    on_elig_mean = (sum(on_elig) / len(on_elig)) if on_elig else None
    off_elig_mean = (sum(off_elig) / len(off_elig)) if off_elig else None
    on_eligibility_eliminated = bool(on_elig_mean is not None
                                     and on_elig_mean <= ELIGIBILITY_ELIMINATED_CEILING)
    # Recorded as CONTEXT ONLY -- rate-capped, nothing routes on it.
    off_rates = [r["mean_request_rate_per_episode"] for r in off_rows]
    on_rates = [r["mean_request_rate_per_episode"] for r in on_rows]
    off_rate_cv, on_rate_cv = _cv(off_rates), _cv(on_rates)

    preconditions: List[Dict[str, Any]] = [
        {"name": "G0_endogenous_trigger_mechanism_live",
         "description": "every (arm x seed) 1e6 control burst fires at least once",
         "measured": float(control_fire_fraction),
         "threshold": float(G0_CONTROL_ALL_CELLS_FIRE_THRESHOLD),
         "direction": "lower",
         "control": ("margin_threshold=1e6 guaranteed-fire burst (%d ticks/cell), matching "
                     "contract test W11" % CONTROL_TICKS),
         "met": bool(control_fire_fraction >= G0_CONTROL_ALL_CELLS_FIRE_THRESHOLD)},
        {"name": "G1_commensurability_engaged_on_arm_only",
         "description": ("_last_commensurability_raw is non-empty on EVERY comm_on cell and "
                         "empty on EVERY comm_off cell -- the manipulation engaged, and only "
                         "where it was asked for"),
         "measured": float(engaged_fraction),
         "threshold": float(G1_MANIPULATION_ENGAGED_THRESHOLD),
         "direction": "lower",
         "control": ("the operator populates this dict itself; probed at authoring time as "
                     "5 channel keys under ON and {} under OFF"),
         "met": bool(engaged_fraction >= G1_MANIPULATION_ENGAGED_THRESHOLD)},
        {"name": "G1b_commensurability_warmup_completed",
         "description": ("every comm_on cell's _chan_scale_n reached the configured warmup -- "
                         "below it _commensurability_scale returns an exact 1.0 and the "
                         "operator is a no-op while still reporting as engaged"),
         "measured": float(worst_scale_n),
         "threshold": float(warm_required),
         "direction": "lower",
         "offending_cell": worst_scale_cell,
         "control": ("worst (minimum) _chan_scale_n over every comm_on cell; measured at "
                     "authoring time as 75 after 600 ticks and 188 after 1500, against a "
                     "configured warmup of 20"),
         "met": bool(warm_required > 0 and worst_scale_n >= warm_required)},
        {"name": "G1c_manipulation_moves_the_dv",
         "description": ("the comm_on and comm_off median E3 margins DIFFER on EVERY seed -- "
                         "the mechanical form of this run's DV-symmetry declaration. Equal "
                         "medians mean the manipulation is invisible to the DV by arithmetic "
                         "and F1/F2 are identities, not measurements"),
         "measured": float(dv_moved_fraction),
         "threshold": 1.0,
         "direction": "lower",
         "offending_cell": (("seeds with identical ON/OFF medians: %s" % dv_identical_seeds)
                            if dv_identical_seeds else None),
         "control": ("paired same-seed comparison with a fresh env and a complete RNG reset "
                     "per cell; measured at authoring time as OFF 0.13548 vs ON 0.17944 at 600 "
                     "ticks on seed 0"),
         "met": bool(n_paired_seeds > 0 and n_seeds_dv_moved == n_paired_seeds)},
        {"name": "G2_eligibility_dv_sample_sufficiency",
         "description": ("every cell collected at least %.0f%% of the FRESH E3 margin samples "
                         "its schedule implies, so each cell's eligibility fraction -- the "
                         "recruitment DV -- is a real statistic. Deliberately NOT a gate on the "
                         "per-episode request count, which is rate-capped by the 51-tick "
                         "coalition debounce and on which nothing routes."
                         % (100.0 * G2_MIN_SAMPLE_FRACTION)),
         "measured": float(worst_samples),
         "threshold": float(sample_floor),
         "direction": "lower",
         "offending_cell": worst_samples_cell,
         "control": ("worst (minimum) fresh-margin sample count over every cell, against "
                     "n_episodes x (ticks // e3_steps_per_tick) = %d expected"
                     % expected_samples),
         "met": bool(worst_samples >= sample_floor)},
        {"name": "G3_off_arm_margin_spread_discriminable",
         "description": ("the comm_off reference arm's own cross-seed median-margin ratio -- "
                         "without spread here there is nothing for normalisation to collapse "
                         "and F1 would pass vacuously"),
         "measured": (float(off_ratio) if off_ratio is not None else None),
         "threshold": float(G3_OFF_MARGIN_RATIO_FLOOR),
         "direction": "lower",
         "control": ("V3-EXQ-1038 published 605.7x on these same 7 seeds under this same "
                     "condition, so this floor is cleared with ~60x to spare"),
         "met": bool(off_ratio is not None and off_ratio >= G3_OFF_MARGIN_RATIO_FLOOR)},
        {"name": "G3b_off_arm_is_scale_determined",
         "description": ("|Spearman rho| between per-seed median E3 margin and per-seed "
                         "eligibility fraction in the comm_off arm -- the PREMISE this run "
                         "tests the repair of. Without it there is no scale-determination for "
                         "normalisation to remove and F2 would collapse vacuously"),
         "measured": (float(off_assoc) if off_assoc is not None else None),
         "threshold": float(G3B_OFF_SCALE_ASSOCIATION_FLOOR),
         "direction": "lower",
         "control": ("V3-EXQ-1038's own published per-seed margins and recruitment give "
                     "|rho| = 0.830 under this same condition, so this floor is cleared with "
                     "0.23 to spare"),
         "met": bool(off_assoc is not None
                     and off_assoc >= G3B_OFF_SCALE_ASSOCIATION_FLOOR)},
        {"name": "G3a_off_arm_eligibility_cv_non_degenerate",
         "description": ("the comm_off arm's cross-seed ELIGIBILITY-FRACTION CV -- F2 is a "
                         "HALVING test against this reference and would pass vacuously on a "
                         "flat arm"),
         "measured": (float(off_cv) if off_cv is not None else None),
         "threshold": float(G3A_OFF_CV_FLOOR),
         "direction": "lower",
         "control": ("measured at authoring time on seed 5 at 250 ticks: eligibility 0.938 "
                     "(comm_off) vs 0.328 (comm_on), so the statistic is live and moves"),
         "met": bool(off_cv is not None and off_cv >= G3A_OFF_CV_FLOOR)},
    ]

    gate_green = all(bool(p["met"]) for p in preconditions)
    failed = [p["name"] for p in preconditions if not p["met"]]

    # ---- THE FINDING. Routes the LABEL, never the outcome. ---------------------------------
    f1 = bool(on_ratio is not None and on_ratio < MARGIN_SPREAD_RATIO_BAR)
    # F2 is the SCALE-ASSOCIATION collapse, against an ABSOLUTE pre-registered bar. The
    # eligibility-CV halving test it replaces is retained as context only (see
    # CV_HALVING_RATIO_CONTEXT_ONLY).
    f2 = bool(on_assoc is not None and on_assoc < SCALE_ASSOCIATION_COLLAPSED_BAR)
    cv_halving_context = bool(on_cv is not None and off_cv is not None
                              and on_cv <= CV_HALVING_RATIO_CONTEXT_ONLY * off_cv)
    # Red-team pass 2, finding 3: a G3/G3a/G3b failure is NOT an instrument defect. G0-G2 being
    # green means the instrument works; what failed is that this run's comm_off arm did not
    # reproduce V3-EXQ-1038's scale-determined premise. A requeue cannot change that, and it IS
    # evidence about SD-091 -- whose amended entry rests on that premise. Separate label,
    # separate routing.
    PREMISE_GATES = ("G3_off_arm_margin_spread_discriminable",
                     "G3a_off_arm_eligibility_cv_non_degenerate",
                     "G3b_off_arm_is_scale_determined")
    instrument_failed = [f for f in failed if f not in PREMISE_GATES]
    premise_failed = [f for f in failed if f in PREMISE_GATES]
    if instrument_failed:
        label = "substrate_not_ready_requeue"
        outcome = "FAIL"
        routing = ("instrument not ready -- %s. NOT a substrate verdict and NOT evidence about "
                   "SD-091." % ", ".join(instrument_failed))
    elif premise_failed:
        label = "premise_not_reproduced_off_arm_not_scale_determined"
        outcome = "FAIL"
        routing = ("the INSTRUMENT is sound (G0-G2 green) but this run's comm_off arm did not "
                   "reproduce V3-EXQ-1038's scale-determined premise -- %s. A requeue cannot "
                   "change this and must NOT be attempted. It IS evidence about SD-091, whose "
                   "amended entry rests on that premise: route to /failure-autopsy to "
                   "adjudicate whether the premise is regime-specific (this run uses 20 x 250 "
                   "ticks per cell against V3-EXQ-1038's 30 x 100, and E3 accumulator state is "
                   "never reset within a cell) or did not hold in the first place."
                   % ", ".join(premise_failed))
    elif on_eligibility_eliminated:
        # RED-TEAM F3. _cv is undefined at a zero mean, so an ON arm with no eligibility at all
        # would otherwise read as "the CV did not halve" and route to a build. That is a
        # DIFFERENT fact and gets its own label.
        outcome = "PASS"
        label = "commensurability_eliminates_eligibility"
        routing = ("the ON arm's mean eligibility fraction is %r, at or below the pre-registered "
                   "ELIGIBILITY_ELIMINATED_CEILING of %.3f: normalisation did not merely "
                   "compress the spread, it pushed essentially every margin ABOVE the 0.05 "
                   "threshold, so coalition recruitment stops rather than becoming "
                   "scale-invariant. That is NOT the 'normalisation suffices' outcome and must "
                   "NOT be read as one -- route to /failure-autopsy to decide whether the "
                   "threshold, the operator, or both are mis-scaled."
                   % (on_elig_mean, ELIGIBILITY_ELIMINATED_CEILING))
    else:
        outcome = "PASS"
        if f1 and f2:
            label = "commensurability_collapses_scale_spread"
            routing = ("normalisation SUFFICES: the scale-relative coalition threshold build on "
                       "SD-091 is NOT owed; mark the probe sufficient on that entry.")
        elif not f1 and not f2:
            label = "scale_spread_persists_under_commensurability"
            routing = ("the existing operator does NOT make recruitment scale-invariant. TWO "
                       "readings remain and THIS RUN DOES NOT SEPARATE THEM (red-team F2): "
                       "(i) the coalition THRESHOLD needs to be scale-relative -- the SD-091 "
                       "build; (ii) the OPERATOR's own divisor is the problem -- it normalises "
                       "each channel by an EMA of that channel's cross-candidate SD "
                       "(e3_selector.py:1362-1377, floor 1e-12), which does not normalise the "
                       "summed score's top-2 gap and can AMPLIFY an intermittent low-SD "
                       "channel. Route to /failure-autopsy to adjudicate (i) vs (ii) using the "
                       "per-cell within_cell_margin_dispersion and chan_scale_ema_final this "
                       "run records. This is an INFORMATIVE null that discharges SD-091's probe "
                       "gate; it is NOT an automatic build authorisation.")
        else:
            label = "commensurability_partially_collapses_scale_spread"
            routing = ("exactly one of F1/F2 moved -> /failure-autopsy to adjudicate which half "
                       "and why. NOT an automatic build and NOT an automatic close.")

    print("[measurement] %s | F1 margin-ratio ON=%r (bar<%.1f) OFF=%r | F2 scale-assoc |rho| "
          "ON=%r (bar<%.2f) OFF=%r (floor %.2f) | mean eligibility ON=%r OFF=%r | "
          "eligibility-CV ON=%r OFF=%r (context only)"
          % (label, on_ratio, MARGIN_SPREAD_RATIO_BAR, off_ratio,
             on_assoc, SCALE_ASSOCIATION_COLLAPSED_BAR, off_assoc,
             G3B_OFF_SCALE_ASSOCIATION_FLOOR, on_elig_mean, off_elig_mean, on_cv, off_cv),
          flush=True)

    flat: Dict[str, Any] = {
        "n_seeds": len(seeds),
        "n_arms": len(ARMS),
        "n_episodes_per_cell": int(sched["n_episodes"]),
        "ticks_per_episode": int(sched["ticks"]),
        "request_count_ceiling": int(ceiling),
        "max_per_episode_request_count": int(worst_count),
        "control_fire_fraction": float(control_fire_fraction),
        "commensurability_engaged_fraction": float(engaged_fraction),
        "default_margin_threshold": float(DEFAULT_MARGIN_THRESHOLD),
        "margin_spread_ratio_bar": float(MARGIN_SPREAD_RATIO_BAR),
        "cv_halving_ratio_context_only": float(CV_HALVING_RATIO_CONTEXT_ONLY),
        "g3b_off_scale_association_floor": float(G3B_OFF_SCALE_ASSOCIATION_FLOOR),
        "off_margin_spread_ratio": off_ratio,
        "on_margin_spread_ratio": on_ratio,
        "off_scale_association_abs_rho": off_assoc,
        "on_scale_association_abs_rho": on_assoc,
        "scale_association_collapsed_bar": float(SCALE_ASSOCIATION_COLLAPSED_BAR),
        "off_eligibility_fraction_cv": off_cv,
        "on_eligibility_fraction_cv": on_cv,
        "off_eligibility_fraction_mean": off_elig_mean,
        "on_eligibility_fraction_mean": on_elig_mean,
        "on_eligibility_eliminated": 1 if on_eligibility_eliminated else 0,
        "off_recruitment_rate_cv_context_only": off_rate_cv,
        "on_recruitment_rate_cv_context_only": on_rate_cv,
        "worst_cell_e3_margin_n_samples": int(worst_samples),
        "max_per_episode_request_count_context_only": int(worst_count),
        "f1_margin_spread_collapses": 1 if f1 else 0,
        "f2_recruitment_cv_halves": 1 if f2 else 0,
        "gate_green": 1 if gate_green else 0,
        "on_arm_worst_chan_scale_n": int(worst_scale_n),
        "commensurability_warmup_ticks": int(warm_required),
        "dv_moved_fraction": float(dv_moved_fraction),
        "n_seeds_dv_identical": int(len(dv_identical_seeds)),
    }
    for arm, arows in ((ARM_OFF, off_rows), (ARM_ON, on_rows)):
        rates = [r["mean_request_rate_per_episode"] for r in arows]
        if rates:
            flat["%s_mean_request_rate" % arm] = float(sum(rates) / len(rates))
        meds = [r["e3_margin_median"] for r in arows if r["e3_margin_median"] is not None]
        if meds:
            flat["%s_median_e3_margin_mean" % arm] = float(sum(meds) / len(meds))
    flat = {k: v for k, v in flat.items()
            if v is not None and (not isinstance(v, float) or v == v)}

    return {
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": QUEUE_ID,
        "backlog_id": BACKLOG_ID,
        "architecture_epoch": ARCHITECTURE_EPOCH,
        "claim_ids": list(CLAIM_IDS),
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "sleep_driver_pattern": "none",
        "outcome": outcome,
        "evidence_direction": "non_contributory",
        "readout": flat,
        "arm_results": rows,
        "control_results": control_rows,
        "operator_validation_status": OPERATOR_VALIDATION_STATUS,
        "request_count_rate_cap": {
            "per_episode_request_count_is_rate_capped": True,
            "ceiling_this_run": int(ceiling),
            "worst_cell_max_count": int(worst_count),
            "worst_cell": worst_count_cell,
            "why": ("CoalitionController.should_dissolve is timeout-only "
                    "(coalition_controller.py:143, >= coalition_max_duration_ticks = %d) and the "
                    "endogenous trigger runs BEFORE coalition.tick() in the same select_action, "
                    "so the earliest re-request is 51 ticks later, on a 10-tick E3 grid. The "
                    "attainable maximum and the arithmetic ceiling therefore scale together and "
                    "lengthening the episode cannot create headroom. MEASURED here: seed 5 "
                    "comm_off reached [4, 4] against a ceiling of 5 at 250 ticks."
                    % int(COALITION_MAX_DURATION_TICKS)),
            "consequence": ("the per-episode request count and frac_episodes_recruited are "
                            "RECORDED for continuity with V3-EXQ-1038 and as context. NOTHING "
                            "routes on them. The recruitment DV is eligibility_fraction."),
        },
        "margin_spread_detail": {"comm_off": off_ratio_detail, "comm_on": on_ratio_detail},
        "scale_association_detail": {
            "comm_off": off_assoc_detail, "comm_on": on_assoc_detail,
            "note": ("|Spearman rho| between per-seed median E3 margin and per-seed eligibility "
                     "fraction. THE statistic F2 routes on: 'is recruitment scale-determined?'. "
                     "A residual eligibility spread that is UNCORRELATED with scale reads here "
                     "as rho ~ 0 -- i.e. scale removed, genuine conflict heterogeneity revealed, "
                     "which is SD-091's DESIRED end state and not a failure (red-team pass 2, "
                     "finding 2). A CV-based criterion could not make that distinction."),
            "post_hoc_for_autopsy": ("if F2 fails, recompute each ON cell's eligibility at a "
                                     "SCALE-RELATIVE threshold "
                                     "0.05 * (cell_median / pooled_ON_median) from the stored "
                                     "e3_margin_samples; if its cross-seed spread matches the "
                                     "fixed-threshold one, the residual is not scale."),
        },
        "eligibility_detail": {
            "comm_off": {"per_seed_eligibility_fraction": off_elig, "cv": off_cv,
                         "mean": off_elig_mean},
            "comm_on": {"per_seed_eligibility_fraction": on_elig, "cv": on_cv,
                        "mean": on_elig_mean},
            "note": ("the RECRUITMENT DV. Fraction of fresh E3 margin samples below the shipped "
                     "endogenous_coalition_margin_threshold. Uncapped, unlike the request "
                     "count."),
        },
        "recruitment_rate_detail_context_only": {
            "comm_off": {"per_seed_rate": off_rates, "cv": off_rate_cv},
            "comm_on": {"per_seed_rate": on_rates, "cv": on_rate_cv},
            "note": "RATE-CAPPED -- see request_count_rate_cap. Nothing routes on these.",
        },
        "within_cell_margin_dispersion_detail": {
            arm: {str(r["seed"]): r.get("within_cell_margin_dispersion")
                  for r in _arm_rows(rows, arm)} for arm in ARMS},
        "chan_scale_ema_detail": {
            arm: {str(r["seed"]): r.get("chan_scale_ema_final")
                  for r in _arm_rows(rows, arm)} for arm in ARMS},
        "interpretation": {
            "label": label,
            "routing": routing,
            "preconditions": preconditions,
            "criteria": [
                {"name": "G0_readiness_control_fires",
                 "load_bearing": True,
                 "passed": bool(control_fire_fraction >= G0_CONTROL_ALL_CELLS_FIRE_THRESHOLD),
                 "measured": float(control_fire_fraction),
                 "threshold": float(G0_CONTROL_ALL_CELLS_FIRE_THRESHOLD)},
                {"name": "F1_on_arm_margin_spread_below_bar",
                 "load_bearing": False,
                 "passed": f1,
                 "measured": on_ratio,
                 "threshold": float(MARGIN_SPREAD_RATIO_BAR),
                 "threshold_note": ("ABSOLUTE pre-registered bar, fixed from V3-EXQ-1038's "
                                    "published 605.7x. Routes the LABEL, not the outcome.")},
                {"name": "F2_on_arm_scale_association_collapsed",
                 "load_bearing": False,
                 "passed": f2,
                 "measured": on_assoc,
                 "threshold": float(SCALE_ASSOCIATION_COLLAPSED_BAR),
                 "reference_off_scale_association": off_assoc,
                 "threshold_note": ("|Spearman rho| between per-seed median E3 margin (SCALE) "
                                    "and per-seed eligibility fraction (RECRUITMENT). An "
                                    "ABSOLUTE pre-registered bar from V3-EXQ-1038's own "
                                    "|rho| = 0.830, NOT a ratio against a realised reference: "
                                    "the eligibility CV of a bimodal 7-vector is a function of "
                                    "how many seeds sit in the high mode, so a halving bar on "
                                    "it would be a statement about that split rather than "
                                    "about the operator (red-team pass 2, finding 1). Routes "
                                    "the LABEL, not the outcome.")},
                {"name": "F2c_eligibility_cv_halved_CONTEXT_ONLY",
                 "load_bearing": False,
                 "passed": cv_halving_context,
                 "measured": on_cv,
                 "threshold": ((CV_HALVING_RATIO_CONTEXT_ONLY * off_cv)
                               if off_cv is not None else None),
                 "reference_off_cv": off_cv,
                 "threshold_note": ("RECORDED, NOT ROUTED ON -- superseded by F2. Retained so a "
                                    "reader can see the statistic the first draft would have "
                                    "used and why it was withdrawn.")},
            ],
            "combination_rule": ("G0 (with the other preconditions) decides the OUTCOME: all "
                                 "green -> PASS, meaning a valid instrument-verified "
                                 "measurement was obtained. F1 and F2 decide the LABEL and the "
                                 "routing: F1 AND F2 -> commensurability_collapses_scale_spread "
                                 "(no SD-091 build owed); NEITHER -> "
                                 "scale_spread_persists_under_commensurability (build owed); "
                                 "EXACTLY ONE -> partial, route to /failure-autopsy."),
            "criteria_non_degenerate": {
                "G0_readiness_control_fires": bool(len(control_rows) > 0),
                "F1_on_arm_margin_spread_below_bar": bool(
                    off_ratio is not None and off_ratio >= G3_OFF_MARGIN_RATIO_FLOOR
                    and on_ratio is not None
                    and n_paired_seeds > 0 and n_seeds_dv_moved == n_paired_seeds),
                "F2_on_arm_scale_association_collapsed": bool(
                    off_assoc is not None and off_assoc >= G3B_OFF_SCALE_ASSOCIATION_FLOOR
                    and on_assoc is not None
                    and n_paired_seeds > 0 and n_seeds_dv_moved == n_paired_seeds),
            },
            "gate_reason": (None if gate_green
                            else "preconditions unmet: %s" % ", ".join(failed)),
            "null_reading": ("A null (F1 and F2 both false) means the existing divisive "
                             "normalisation does NOT make recruitment scale-invariant on this "
                             "harness. It does NOT by itself establish that the SD-091 "
                             "threshold build is the right repair: the operator's own divisor "
                             "is an equally live account (red-team F2; see this label's "
                             "routing). It also does not bear on whether commensurability is "
                             "correct for its OWN purpose -- that operator's validation is "
                             "separately OPEN (V3-EXQ-1012a)."),
        },
        "non_degenerate": bool(gate_green),
        "degeneracy_reason": (None if gate_green
                              else "preconditions unmet: %s" % ", ".join(failed)),
    }


def _run_self_test() -> int:
    fails = 0

    def _chk(ok: bool, msg: str) -> None:
        nonlocal fails
        if not ok:
            fails += 1
            print("[self-test] FAIL: %s" % msg, flush=True)

    # The swept factor must be REACHABLE, and the obvious spelling must be known-broken.
    env = CausalGridWorldV2(seed=0, size=GRID_SIZE, num_hazards=NUM_HAZARDS,
                            num_resources=NUM_RESOURCES, use_proxy_fields=True)
    swallowed = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim, world_obs_dim=env.world_obs_dim,
        action_dim=ACTION_DIM, self_dim=SELF_DIM, world_dim=WORLD_DIM,
        use_e3_channel_commensurability=True)
    _chk(getattr(swallowed, "use_e3_channel_commensurability", None) is not True,
         "from_dims now ACCEPTS use_e3_channel_commensurability -- the docstring's "
         "silent-swallow warning has gone stale; re-check which path this driver should use")
    for want in (False, True):
        agent, _env = _build(0, DEFAULT_MARGIN_THRESHOLD, want)
        _chk(bool(getattr(agent.e3.config, "use_e3_channel_commensurability", None)) is want,
             "the swept factor did not reach agent.e3.config for want=%r" % want)

    # The headroom fix must actually give headroom over V3-EXQ-1038's realised maximum of 2.
    _chk(REQUEST_COUNT_CEILING >= 5,
         "REQUEST_COUNT_CEILING is %d; V3-EXQ-1038 PINNED at 2, so the ceiling must give real "
         "headroom" % REQUEST_COUNT_CEILING)
    _chk(TICKS_PER_EPISODE % COALITION_MAX_DURATION_TICKS == 0,
         "TICKS_PER_EPISODE must be a whole multiple of coalition_max_duration_ticks so the "
         "ceiling arithmetic G2 gates on is exact")

    # The bars must be SATISFIABLE and DISCRIMINATING against V3-EXQ-1038's published values.
    ref_1038_medians = [0.13895416259765625, 11.18798828125, 0.18807601928710938,
                        3.82275390625, 2.2117919921875, 0.01847076416015625,
                        1.642852783203125]
    ref_ratio = max(ref_1038_medians) / min(ref_1038_medians)
    _chk(ref_ratio > MARGIN_SPREAD_RATIO_BAR,
         "NEGATIVE CONTROL: V3-EXQ-1038's own OFF-condition spread (%.1fx) must FAIL the F1 "
         "bar -- a bar the unnormalised condition already clears witnesses nothing" % ref_ratio)
    _chk(ref_ratio >= G3_OFF_MARGIN_RATIO_FLOOR,
         "POSITIVE CONTROL: V3-EXQ-1038's own spread (%.1fx) must CLEAR the G3 discriminability "
         "floor, or G3 is unmeetable by the very condition it anchors to" % ref_ratio)
    # G3a is on the ELIGIBILITY FRACTION. V3-EXQ-1038 never reported that statistic, so the
    # positive control is this driver's OWN authoring-time measurement plus the per-seed
    # eligibility implied by V3-EXQ-1038's recruitment pattern (0.856 down to 0.010), which
    # is the very 'recruitment tracks the sub-threshold fraction' finding this run extends.
    # F2's bars, checked against V3-EXQ-1038's OWN published per-seed values.
    ref_margin_1038 = [0.13895416259765625, 11.18798828125, 0.18807601928710938,
                       3.82275390625, 2.2117919921875, 0.01847076416015625,
                       1.642852783203125]
    ref_rate_1038 = [1.2333333333333334, 0.13333333333333333, 1.3333333333333333,
                     0.13333333333333333, 0.1, 2.0, 0.23333333333333334]
    ref_rho = _spearman(ref_margin_1038, ref_rate_1038)
    _chk(ref_rho is not None and abs(ref_rho) >= G3B_OFF_SCALE_ASSOCIATION_FLOOR,
         "POSITIVE CONTROL: V3-EXQ-1038's own scale association (|rho|=%r) must CLEAR the G3b "
         "floor, or G3b is unmeetable by the very condition it anchors to" % (
             abs(ref_rho) if ref_rho is not None else None))
    _chk(ref_rho is not None and abs(ref_rho) >= SCALE_ASSOCIATION_COLLAPSED_BAR,
         "NEGATIVE CONTROL: V3-EXQ-1038's own OFF condition (|rho|=%r) must FAIL F2's collapse "
         "bar -- a bar the un-normalised condition already clears witnesses nothing" % (
             abs(ref_rho) if ref_rho is not None else None))
    _chk(SCALE_ASSOCIATION_COLLAPSED_BAR < G3B_OFF_SCALE_ASSOCIATION_FLOOR,
         "the collapse bar must sit BELOW the premise floor, or the two gates contradict")
    _chk(_spearman([1.0, 2.0, 3.0], [3.0, 2.0, 1.0]) == -1.0, "_spearman sign/scale is wrong")
    _chk(_spearman([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]) is None,
         "_spearman must be undefined on a constant input")
    _chk(_rank([5.0, 1.0, 5.0]) == [2.5, 1.0, 2.5], "_rank tie correction is wrong")
    ref_elig_1038 = [0.856, 0.010, 0.900, 0.010, 0.010, 0.938, 0.050]
    ref_elig_cv = _cv(ref_elig_1038)
    _chk(ref_elig_cv is not None and ref_elig_cv >= G3A_OFF_CV_FLOOR,
         "POSITIVE CONTROL: the comm_off eligibility-fraction CV implied by V3-EXQ-1038 (%r) "
         "must clear the G3a floor, or G3a is unmeetable by the condition it anchors to"
         % ref_elig_cv)
    _chk(ELIGIBILITY_ELIMINATED_CEILING < G3A_OFF_CV_FLOOR * 0 + 0.05,
         "the eliminated-ceiling must sit BELOW the shipped margin threshold's own regime, so "
         "'eliminated' cannot be confused with 'merely reduced'")
    # The request count must NOT be gated: it is rate-capped, and seed 5 was MEASURED at the
    # cap (counts [4, 4] against a ceiling of 5 at 250 ticks). A gate on it would red the run
    # on a mechanism cap. Assert no precondition names it.
    # The marker is assembled at runtime so this assertion's own text cannot match itself.
    _banned = '"name": "G2_' + 'request_count_headroom"'
    _chk(_banned not in open(__file__, encoding="utf-8").read(),
         "REGRESSION: a precondition is gating on the RATE-CAPPED per-episode request count, "
         "which is capped by the 51-tick coalition debounce and would red the run on a "
         "mechanism cap (measured: seed 5 comm_off [4, 4] against a ceiling of 5)")

    # Helper contracts the gates rest on.
    _chk(_cv([1.0, 1.0, 1.0]) == 0.0, "_cv must return 0.0 for a flat, non-zero-mean list")
    _chk(_cv([1.0]) is None, "_cv must be undefined for n<2")
    r, d = _margin_spread_ratio([{"seed": 0, "e3_margin_median": 2.0},
                                 {"seed": 1, "e3_margin_median": 0.5}])
    _chk(r == 4.0 and d["min_seed"] == 1 and d["max_seed"] == 0,
         "_margin_spread_ratio returned %r / %r" % (r, d))
    r2, _d2 = _margin_spread_ratio([{"seed": 0, "e3_margin_median": None}])
    _chk(r2 is None, "_margin_spread_ratio must be undefined with fewer than 2 usable seeds")

    print("[self-test] %d failure(s)" % fails, flush=True)
    return fails


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        sys.exit(_run_self_test())

    t0 = time.perf_counter()
    print("=== %s ===" % EXPERIMENT_TYPE)
    print("Queue ID: %s" % QUEUE_ID)
    print("Claim: %s" % CLAIM_IDS)
    print("Mode: %s" % ("DRY-RUN" if args.dry_run else "FULL RUN"))
    print("Arms: %s (swept factor: E3Config.use_e3_channel_commensurability)" % ARMS)
    print()

    result = run_experiment(dry_run=args.dry_run)

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    result["timestamp_utc"] = ts
    result["run_timestamp"] = ts
    result["run_id"] = "%s_%s_v3" % (EXPERIMENT_TYPE, ts)

    full_config = {
        "arms": list(ARMS),
        "swept_factor": "E3Config.use_e3_channel_commensurability",
        "endogenous_coalition_margin_threshold": float(DEFAULT_MARGIN_THRESHOLD),
        "coalition_max_duration_ticks": int(COALITION_MAX_DURATION_TICKS),
        "n_episodes": int(DRY_RUN_N_EPISODES if args.dry_run else N_EPISODES),
        "ticks_per_episode": int(DRY_RUN_TICKS_PER_EPISODE if args.dry_run
                                 else TICKS_PER_EPISODE),
        "request_count_ceiling": int(DRY_RUN_REQUEST_COUNT_CEILING if args.dry_run
                                     else REQUEST_COUNT_CEILING),
        "control_margin_threshold": float(CONTROL_MARGIN_THRESHOLD),
        "control_ticks": int(CONTROL_TICKS),
        "grid_size": GRID_SIZE, "num_hazards": NUM_HAZARDS, "num_resources": NUM_RESOURCES,
        "action_dim": ACTION_DIM, "self_dim": SELF_DIM, "world_dim": WORLD_DIM,
        "margin_spread_ratio_bar": float(MARGIN_SPREAD_RATIO_BAR),
        "cv_halving_ratio_context_only": float(CV_HALVING_RATIO_CONTEXT_ONLY),
        "g3b_off_scale_association_floor": float(G3B_OFF_SCALE_ASSOCIATION_FLOOR),
        "g3_off_margin_ratio_floor": float(G3_OFF_MARGIN_RATIO_FLOOR),
        "g3a_off_cv_floor": float(G3A_OFF_CV_FLOOR),
        "predecessor": "V3-EXQ-1038",
        "dry_run": bool(args.dry_run),
    }
    out_path = write_flat_manifest(
        result, dry_run=bool(args.dry_run),
        config=full_config,
        seeds=(DRY_RUN_SEEDS if args.dry_run else SEEDS),
        script_path=Path(__file__),
        started_at=t0,
        z_goal_stream_stats=_ZG.stats(),
    )

    print("\nWrote manifest to: %s" % out_path)
    print("outcome: %s (%s)" % (result["outcome"], result["interpretation"]["label"]))

    emit_outcome(
        outcome=result["outcome"],
        manifest_path=str(out_path),
        dry_run=bool(args.dry_run),
    )
