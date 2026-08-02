"""V3-EXQ-864a: SD-076 wci_symmetric_rv_ref/rv WITHIN-EPISODE trajectory diagnostic --
where/when does the inflation_lowers_rv sign flip (found in V3-EXQ-860) happen?
CORRECTED re-run of V3-EXQ-864 -- see "DESIGN FIX" below.

SLEEP DRIVER: K=1 single-fire (SleepLoopManager, sleep_loop_episodes_K=1, fires every
episode) -- UNCHANGED from the 794a/850/853/860/864 lineage.

THE PUZZLE THIS DIAGNOSES (Finding B, failure_autopsy_V3-EXQ-860_2026-08-01.md section 2).
V3-EXQ-860 measured `inflation_lowers_rv` (= mean over seeds of
wci_symmetric_rv_ref_final - rv_final_after_training) WRONG-SIGNED in BOTH arms at
STEPS_PER_EP=1000 (LO -2.76e-4, HI -1.46e-4, vs a +1e-4 threshold) -- the inflated rv
ended up ABOVE the substrate's own live un-inflated counterfactual, opposite the
intended direction. Siblings V3-EXQ-850/853 (both at STEPS_PER_EP=200) showed small
POSITIVE-but-sub-threshold values for the SAME statistic, never negative. This sign
flip appeared only at 5x longer episodes and is still undiagnosed -- see "STATUS" below.

WHY THIS SCRIPT EXISTS (supersedes V3-EXQ-864). V3-EXQ-864
(experiments/v3_exq_864_sd076_wci_rv_trajectory_crossover_diagnostic.py, run_id
v3_exq_864_sd076_wci_rv_trajectory_crossover_diagnostic_20260801T195304Z_v3) was the
first attempt at this instrumentation. Its own confirmed autopsy
(failure_autopsy_V3-EXQ-864_2026-08-01.md/.json, 2026-08-01) code-verified a defect
that made its sweep inert: `CausalGridWorldV2`'s own episode termination
(`agent_health <= 0.0 or self.steps >= 500`, `ree_core/environment/causal_grid_world.py:
2892`) fired at ~8-11 ticks/episode in this driver's env config (hazard_harm=0.04,
proximity_harm_scale=0.12, num_hazards=3, no avoidance pretraining) -- far short of any
of the three swept STEPS_PER_EP values (200/500/1000). Because 864's inner loop did
`if result.done: break`, every cell's real per-episode exposure was ~8-11 ticks
regardless of the nominal STEPS_PER_EP configured, so `diff_final` and every other
end-of-episode-snapshot readout came out bit-identical across all three sweep points in
both arms and both seeds. The "characterized, no crossover in range" verdict 864
reported was uninformative on the question it was built to answer -- the sweep never
actually varied real exposure. V3-EXQ-864's autopsy routed the repair to
`/queue-experiment`; this script is that redesign.

DESIGN FIX (what changed, and why the alternatives were rejected). The manipulated
variable this diagnostic needs is "ticks of within-episode EMA drift before the ONE
per-episode F1/REM-recalibration correction" (see MECHANISM below) -- i.e. STEPS_PER_EP
must genuinely bound real exposure between two `agent.reset()` calls. Three repair
strategies were considered:

  (a) Train/equip the agent with enough hazard-avoidance competence to survive to the
      swept episode lengths on its own. REJECTED: this requires an additional
      pretraining phase, and additional gradient steps change the very world-model /
      E3 predictor whose EMA dynamics this diagnostic is trying to observe -- it
      substitutes "what happens to rv/wci_symmetric_rv_ref over N ticks of a
      hazard-COMPETENT policy" for the substrate's actual operating point (matching
      794a/850/853/860, all of which train from scratch with no avoidance
      pretraining). Confounds the very thing under measurement.

  (b) Use a benign/reduced-hazard env config (fewer/weaker hazards) FOR THIS
      DIAGNOSTIC specifically, since it targets within-episode EMA trajectory
      dynamics, not hazard-survival behaviour. REJECTED IN ITS LITERAL FORM: the
      per-tick `harm_signal` computed from hazard contact/proximity IS the dominant
      source of prediction error this substrate operating point generates (see
      MECHANISM: both EMAs update from `e3.post_action_update`'s per-tick prediction
      error against `_current_latent.z_world`, driven by `update_residue(harm_signal=
      ...)` every training tick, `ree_core/agent.py:8826-8859`). Turning hazards down
      or off would change the CHARACTER of the very PE stream the two EMAs are
      tracking (fewer/smaller surprising events), not just how long the episode
      survives -- i.e. it would change what is measured, the same objection as (a),
      just applied to the environment instead of the agent.

  (c) Concatenate multiple short env sub-episodes to accumulate ticks up to
      STEPS_PER_EP, rather than expecting one env episode to reach the target length.
      REJECTED IN ITS LITERAL FORM, for a reason that only appears on close reading of
      `REEAgent.reset()` (`ree_core/agent.py:2939-3040`): agent.reset() is what fires
      the once-per-episode F1/REM recalibration writeback (via
      `sleep_loop.notify_episode_end`), AND it clears the agent's own recurrent state
      (`self.e1.reset_hidden_state()`, `self._current_latent`, the latent stack, and
      ~20 other per-episode accumulators). So concatenating short env sub-episodes
      forces an impossible choice: call `agent.reset()` at every sub-episode boundary
      (defeats the whole point -- STEPS_PER_EP would then govern many small
      recalibration cycles instead of ticks-before-ONE-correction, the exact quantity
      this diagnostic exists to bracket) or DON'T call it (leaves E1's recurrent state
      and latent stack stale across an environment that just silently re-randomised
      the agent's position/hazard layout underneath it -- a "teleport" the agent's own
      memory does not know happened, injecting large artificial prediction-error
      spikes at every sub-reset that are not part of the substrate's real dynamics).
      Either way, the "fix" adds a NEW confound as bad as, or worse than, the one it
      removes.

  CHOSEN FIX -- decouple STEPS_PER_EP from environment-internal termination WITHOUT
  changing hazard density, without extra training, and without touching agent.reset()
  cadence: this driver's per-episode inner loop no longer breaks on `result.done`.
  Verified by direct source read (2026-08-02) that this is safe for exactly this
  substrate:
    - `CausalGridWorld.step()` computes `done = self.agent_health <= 0.0 or
      self.steps >= 500` FRESH on every call from plain instance attributes (line
      2892) -- it does not store or check any "already done" flag, so calling
      step() again after done=True is not calling it out of its designed contract; it
      simply keeps computing normally.
    - `harm_signal` (hazard contact/proximity harm) is computed identically whether or
      not agent_health has already floored at 0.0 -- the `max(0.0, ...)` clamp is
      applied to `agent_health`, never to `harm_signal` itself (causal_grid_world.py:
      2205-2219). So the per-tick PE-generating signal this diagnostic exists to
      observe is UNCHANGED by continuing past the nominal death point.
    - `agent_health` pinned at its floor is not a dead end: ordinary resource contact
      still restores it mid-run exactly as it would pre-"death"
      (`self.agent_health = min(1.0, self.agent_health + contact_benefit * 0.5)`,
      causal_grid_world.py:2020) -- no extra revival code needed.
    - Grepping `ree_core/agent.py` and `experiments/_harness.py` for `agent_health`
      confirms NEITHER the agent NOR StepHarness special-cases a "dead" agent -- there
      is no code path that stops training, freezes the EMA, or otherwise behaves
      differently once agent_health has floored.
    - The only effect of `self.steps` exceeding the internal 500-cap besides
      contributing to `done` is the observation feature `body[9] = min(1.0,
      self.steps / 500.0)` (causal_grid_world.py:3389), which simply saturates at 1.0
      for the remainder of a >500-tick episode -- a bounded, harmless, and separately
      documented side effect (not a source of instability).
  This realizes both of failure_autopsy_V3-EXQ-864's own suggested repair options --
  "(a) disable agent-death termination" and "(b) raise the internal step cap" -- through
  ONE mechanism (ignore `result.done` in this driver's own loop), with the SAME hazard
  dynamics, the SAME agent (no extra pretraining), and the SAME once-per-STEPS_PER_EP
  recalibration cadence the puzzle needs. `result.done` is still recorded (see
  `env_done_ticks_this_episode` / `env_first_done_tick_ep0` in each row) so the
  instrumentation ALSO makes the original defect's magnitude directly visible in this
  run's own manifest: readers can see exactly how early the environment's own
  termination would have fired, right alongside the corrected trajectory that ran past
  it.

  STATUS: this redesign investigates a DIFFERENT hypothesis space than the sweep values
  alone -- STEPS_PER_EP now genuinely bounds real per-episode exposure, so the
  crossover-bracketing question 864 was built to answer can, for the first time, be
  answered rather than begged. Whether the sign flip 860 saw at 1000 is real, and where
  (if anywhere) it occurs, remains open until this run's own results are read.

MECHANISM UNDER INSTRUMENTATION (ree_core/predictors/e3_selector.py). Every tick,
`update_running_variance` (line 626) advances BOTH the inflated path
`_running_variance` (asymmetric alpha: faster when error is improving relative to its
OWN current estimate, slower when worsening) and the counterfactual
`_wci_symmetric_rv_ref` (fixed symmetric `_ema_alpha`, same `error_var`) from the SAME
per-tick prediction error. `_apply_wci_rv_floor` (line 691) then bounds the inflated
path at `waking_confidence_rv_floor_relative_frac * _wci_symmetric_rv_ref` via a soft
floor. Both EMAs are advanced from `post_action_update` (line ~3658) on every
env-step's `select_action` -> `post_action_update` call, once per training tick --
confirmed by direct source read 2026-08-02, unchanged from 860/864's own
re-verification. The F1/REM precision-recalibration WRITEBACK (MECH-204, unrelated
broadcast mechanism) nudges `_running_variance` once per EPISODE BOUNDARY
(agent.reset(), K=1) -- this is the boundary the DESIGN FIX above deliberately leaves
untouched. HYPOTHESIS: at 5x more within-episode ticks before that once-per-episode
correction, the inflated and counterfactual EMAs may diverge in a way invisible at the
original 200-step episode length. This script does not test that hypothesis -- it
INSTRUMENTS the trajectory (now at genuinely varied exposure) so a human/autopsy can
see where/when (if at all) the diff = ref - rv crosses zero.

DESIGN: sweep STEPS_PER_EP in {200, 500, 1000} -- the three points that bracket the
observed transition (200 in the 850/853 lineage: small positive/sub-threshold; 1000 in
860: negative), UNCHANGED from 864. Log (rv, wci_symmetric_rv_ref, diff=ref-rv) at ~40
evenly-spaced ticks WITHIN EVERY episode, so both the within-episode trajectory and the
across-episode trend are visible -- now over the FULL swept tick count per episode
rather than the ~8-11 ticks 864 actually achieved. Same substrate operating point and
same two inflation doses (ARM_INFL_LO=0.6, ARM_INFL_HI=0.8) as 860/864, so a direct
comparison against 860's own end-of-training readout remains possible as a replication
sanity check (with the caveat that 860 shares the SAME uninstrumented-early-termination
env config as 864 -- see NOTE ON 860 below).

NOTE ON 860's OWN EXPOSURE. V3-EXQ-860 (`experiments/
v3_exq_860_mech204_sd076_h2_steps_per_ep_probe.py`) constructs its env with the
IDENTICAL parameters this script inherited from 864 (`num_hazards=3, hazard_harm=0.04,
proximity_harm_scale=0.12, use_proxy_fields=True`, confirmed by direct read
2026-08-02) -- so 860's own STEPS_PER_EP=1000 arm almost certainly hit the SAME early
termination this diagnostic exists to route around, meaning 860's own sign-flip finding
may itself never have tested 5x-longer real exposure. This is a genuine open question
about the 860 run's own validity, but it is OUT OF SCOPE for this diagnostic to
re-litigate -- 860 is already autopsy-confirmed and closed. Flagged here for whoever
next reads 860's result: this redesign's own trajectory data is the first run in the
794a/850/853/860/864 lineage where STEPS_PER_EP demonstrably varies real exposure
(see `final_global_tick_reached` per cell, which now scales with STEPS_PER_EP by
construction), and is worth comparing against 860's readout on that basis.

REAL COST NOW MATCHES 864's ORIGINALLY-STATED (BUT NEVER ACTUALLY PAID) BUDGET.
864's own docstring pre-registered a "~54400 train-only steps (~30% of 860's 180000)"
budget assuming STEPS_PER_EP genuinely bound exposure -- but the termination defect
meant 864 actually spent only a few thousand real ticks. This redesign pays the
originally-intended cost: N_TRAIN_EPS=8 (unchanged), STEPS_PER_EP swept in
{200,500,1000}, 2 arms, 2 seeds -> (200+500+1000) * 8 * 2 * 2 = 54400 real train-only
env ticks, still ~30% of 860's 180000 with zero eval-phase overhead (860's own elapsed
time for 180000 steps: 4200.6s -> ~0.0233 s/step: this run's 54400 steps projects to
roughly ~21 minutes of pure step time, before model init / optimizer overhead).

CHEAPER THAN 860 ON PURPOSE (queue-experiment skill Step 2.4 GOV-REUSE-1 + "small,
cheap" instruction, unchanged from 864). N_TRAIN_EPS=8 (vs 860's 30) -- this
diagnostic only needs to see a handful of F1-recalibration cycles establish the
within-episode divergence pattern, not full training-scale statistical power (the
closure-fraction comparison, which DOES need that scale, was already answered
decisively by 860's Finding A). No eval phase at all -- eval never touches the
training-phase EMA dynamics this diagnostic exists to see. N_SEEDS=2 (light
replication, not full statistical power -- this is exploratory instrumentation, not a
hypothesis test).

GOV-REUSE-1 CHECK (queue-experiment Step 2.4, re-verified 2026-08-02 for this
redesign). Decisive readout: the WITHIN-EPISODE trajectory of (rv,
wci_symmetric_rv_ref, diff) at STEPS_PER_EP values that GENUINELY vary real exposure --
not recorded by any manifest today, INCLUDING 864's own run (864's trajectory data
exists but is defective: exposure never actually varied, so it cannot answer the
decisive question -- confirmed by its own autopsy's bit-identical `diff_final_by_steps`
finding). Checked via REE_assembly/scripts/reanalysis_query.py query --readout
rv_trajectory_within_episode --claim SD-076 (2026-08-02): 6 manifests scanned
(794/794a/850-h1/850-h2/860/864), 0 carry a within-episode trajectory readout at
verified-varying exposure. Not recoverable -> run (this is a redesign/re-run, not a
reanalysis of 864's data, precisely because 864's data cannot answer the question).

SUBSTRATE READINESS (queue-experiment Step 2.5). Identical substrate surface to
860/864, re-confirmed by direct source read 2026-08-02: `E3TrajectorySelector.
_running_variance` / `._wci_symmetric_rv_ref` (e3_selector.py:302,314),
`update_running_variance` (e3_selector.py:626, called from `post_action_update` at
line ~3658 on every training tick), `agent.sleep_loop` (agent.py), the MECH-204
writeback keys (`mech204_recalibration_fired` / `mech204_running_variance_{before,
after}`, phase_manager.py), and the `use_waking_confidence_inflation` /
`waking_confidence_inflation_asymmetry` / `waking_confidence_rv_floor*` REEConfig.e3
fields (utils/config.py) all confirmed present and unchanged. `CausalGridWorld.step()`
termination condition (`agent_health<=0.0 or steps>=500`, causal_grid_world.py:2892)
and the absence of any `agent_health` gating in `agent.py` / `_harness.py` both
directly re-confirmed 2026-08-02 (see DESIGN FIX above). This script manipulates
NEITHER SD-076/MECH-204 implementation, only STEPS_PER_EP, the sampling
instrumentation, and this driver's own loop-continuation policy -- no new substrate
build needed. claims.yaml: SD-076 is `candidate` / `epistemic_category: standard`, no
additional gate beyond what 860/853/850/864 already cleared.

RE-DERIVE BRAKE (queue-experiment Step 2.5b). This is not a re-test of the H1/H2/H3
discrimination at all (no dose-response criterion, no closure-fraction readout, no
substrate verdict) -- it is pure instrument/measurement re-diagnosis of the SAME puzzle
864 was built for, corrected for 864's own confirmed instrumentation defect. 864's
autopsy category is `measurement_test_design_defect` with `action: none` (no substrate
build owed -- instrument repair, not a substrate gap), so per the INSTRUMENT carve-out
in the re-derive brake counter this does NOT count toward SD-076's brake tally. 0 prior
`substrate_ceiling` autopsies exist for SD-076 (860's own re_derive_brake block:
fired=false, prior_substrate_ceiling_autopsies=[]). Brake does not fire.

CLAIM TAGGING (queue-experiment Step 3 "claim_ids accuracy rule"). CLAIM_IDS =
["SD-076"] only -- identical reasoning to 860/853/850-h2/864: Phase 7 broadcast
(`use_rem_precision_broadcast`) is OFF in every arm, so this produces no MECH-204
evidence. F1/REM recalibration is held ON/unmanipulated (same as 860/864), so its
presence yields no attributable evidence either way. This diagnostic characterizes
SD-076's own inflation-vs-counterfactual EMA dynamics, nothing else.

EXPERIMENT_PURPOSE = "diagnostic": pure instrumentation/measurement investigation of
an undiagnosed sign-flip puzzle, explicitly excluded from confidence/conflict scoring.
Does NOT feed a PASS/FAIL hypothesis about SD-076/MECH-204 -- the "verdict" this
script computes is about DATA QUALITY (did the instrumentation capture usable,
non-degenerate, GENUINELY-VARYING-EXPOSURE trajectories across the sweep), not about a
claim.

DV-SYMMETRY INVARIANCE DECLARATION (mandatory per-arm; queue-experiment Step 3). The
decisive readouts are (a) diff_final = wci_symmetric_rv_ref_after_training -
rv_final_after_training, a scalar EMA-level snapshot with no permutation symmetry
(same declaration as 860/864, unchanged: the asymmetric EMA's settling level changes
with `inflation_asymmetry` and with the number of update ticks before the snapshot --
NOT invariant under either manipulated axis, STEPS_PER_EP or arm dose -- and now that
STEPS_PER_EP genuinely varies real ticks, this non-invariance is exercised rather than
begged), and (b) the per-tick trajectory of diff = ref - rv within an episode, which is
a raw time series, not a set-aggregate -- there is no permutation/rescaling symmetry to
be invariant under; it is read directly, not computed via any aggregate statistic that
a manipulation could leave invariant.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.optim as optim

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ree_core.utils.config import REEConfig  # noqa: E402
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from experiment_protocol import emit_outcome  # noqa: E402
from experiments._harness import StepHarness  # noqa: E402
from experiments._lib.arm_fingerprint import arm_cell  # noqa: E402
from experiments._lib.precondition_gate import (  # noqa: E402
    PreconditionSpec,
    aggregate_arm_gates,
    arm_criteria_non_degenerate,
    assert_no_structurally_unsatisfiable_gate,
    evaluate_arm_gate,
)
from experiments._lib.z_goal_stream import ZGoalStreamAccumulator  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402

QUEUE_ID = "V3-EXQ-864a"
EXPERIMENT_TYPE = "v3_exq_864a_sd076_wci_rv_trajectory_crossover_diagnostic"
CLAIM_IDS = ["SD-076"]
EXPERIMENT_PURPOSE = "diagnostic"
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"
SLEEP_DRIVER_PATTERN = (
    "K=1 single-fire (SleepLoopManager, sleep_loop_episodes_K=1, fires every episode)"
)
SUPERSEDES = "V3-EXQ-864"
FANOUT_SOURCE = "failure_autopsy_V3-EXQ-864_2026-08-01.json"
PUZZLE_SOURCE_RUN_ID = "v3_exq_860_mech204_sd076_h2_steps_per_ep_probe_20260801T142420Z_v3"
PUZZLE_SOURCE_QUEUE_ID = "V3-EXQ-860"
REDESIGN_SOURCE_RUN_ID = "v3_exq_864_sd076_wci_rv_trajectory_crossover_diagnostic_20260801T195304Z_v3"
REDESIGN_SOURCE_QUEUE_ID = "V3-EXQ-864"

# ---- Run shape. STEPS_PER_EP is swept; N_TRAIN_EPS is fixed (unconfounded across the
# sweep) and REDUCED from 860's 30 -- see module docstring "CHEAPER THAN 860". ----
N_TRAIN_EPS = 8            # enough F1 firings (8) to see the within-episode pattern
                            # settle across a few recalibration cycles; not intended to
                            # match 860's full-scale statistical power.
STEPS_PER_EP_SWEEP: Tuple[int, ...] = (200, 500, 1000)   # brackets 860's transition
N_SEEDS = 2
GRID_SIZE = 12
LR = 5e-4

# ---- Substrate operating point (held constant, identical to 794a/850/853/860/864). --
SWS_CONSOLIDATION_STEPS = 8
REM_ATTRIBUTION_STEPS = 6
PRECISION_ZERO_POINT_EMA_ALPHA = 0.1
REM_PRECISION_RECALIBRATION_STEP = 0.25
ALPHA_WORLD = 0.9
ALPHA_SELF = 0.3

# ---- The two inflation doses, identical to 860/864. ----
INFLATION_ASYMMETRY_LO = 0.6
INFLATION_ASYMMETRY_HI = 0.8
INFLATION_RV_FLOOR = 0.01
INFLATION_RV_FLOOR_RELATIVE_FRAC = 0.2
INFLATION_RV_FLOOR_MODE = "soft"
INFLATION_RV_FLOOR_SOFTNESS = 0.25

# ---- Pre-registered thresholds (NOT derived from this run's own statistics). Same
# floors as 794a/850/853/860/864 for the two readiness preconditions this run
# inherits, plus one new precondition (real_exposure_achieved) that operationalizes
# the DESIGN FIX above as a measured, falsifiable check rather than an assumption. ----
PRECISION_INIT_BASELINE = 0.5
RV_LIVE_FLOOR = 1e-6
RECALIB_MOVE_FLOOR = 1e-4
# This diagnostic's own non-degeneracy floor: the trajectory must show REAL EMA
# movement (diff varies across ticks), not a frozen/constant reading. Pre-registered,
# not derived from this run's own data -- a diff that never moves by more than this
# across an entire cell's trajectory means the instrumentation captured nothing.
TRAJECTORY_VARIATION_FLOOR = 1e-6
# NEW (864a): the fraction of the nominal STEPS_PER_EP*N_TRAIN_EPS tick budget this
# cell actually executed. Under the DESIGN FIX this is trivially 1.0 by construction
# (the inner loop never breaks), but it is measured and gated anyway -- per the
# queue-experiment skill's own dose-sweep rule ("assert the DV actually varies across
# swept values before trusting a flat sweep as evidence of no effect") and as a
# regression guard: if a future edit reintroduces an early break, this precondition
# fails loudly instead of silently reproducing V3-EXQ-864's defect.
REAL_EXPOSURE_FLOOR = 0.999

# Within-episode sampling: ~40 samples per episode, regardless of STEPS_PER_EP, so the
# three swept lengths are visually/comparably resolved rather than one being sparse.
SAMPLES_PER_EPISODE_TARGET = 40


def _sample_interval(steps_per_ep: int) -> int:
    return max(1, steps_per_ep // SAMPLES_PER_EPISODE_TARGET)


# (steps_per_ep, arm_id, asymmetry) triples -- the full sweep grid.
ARMS: Tuple[Tuple[str, float], ...] = (
    ("ARM_INFL_LO", INFLATION_ASYMMETRY_LO),
    ("ARM_INFL_HI", INFLATION_ASYMMETRY_HI),
)
CELLS: Tuple[Tuple[int, str, float], ...] = tuple(
    (steps, arm_id, asym)
    for steps in STEPS_PER_EP_SWEEP
    for (arm_id, asym) in ARMS
)

_ZG = ZGoalStreamAccumulator()


# ---------------------------------------------------------------- preconditions --
# Readiness-only (this diagnostic makes no claim-scoring verdict). Three inherited
# unmodified from the 860/853/850/864 lineage's own readiness gate, plus this script's
# own trajectory non-degeneracy check, plus the NEW real_exposure_achieved check that
# makes the DESIGN FIX falsifiable. Applies per (steps_per_ep, arm) cell, aggregated
# across its 2 seeds.
PRECONDITION_SPECS: Tuple[PreconditionSpec, ...] = (
    PreconditionSpec(
        name="rv_live",
        description="rv_final differs from precision_init by more than the floor "
                    "(Q-042/530c substrate-liveness contract). Worst seed reported.",
        control="every seed of this cell; a dead rv makes the trajectory meaningless",
        threshold=RV_LIVE_FLOOR,
        direction="lower",
    ),
    PreconditionSpec(
        name="f1_recalib_engaged",
        description="mean per-cycle |rv_after - rv_before| from the F1 WRITEBACK "
                    "recalibration exceeds the floor -- confirms the reduced "
                    "N_TRAIN_EPS=8 operating point still exercises the same "
                    "substrate mechanism 860 measured at N_TRAIN_EPS=30.",
        control="F1 recalibration is ON in every cell of this design",
        threshold=RECALIB_MOVE_FLOOR,
        direction="lower",
    ),
    PreconditionSpec(
        name="trajectory_non_degenerate",
        description="max(diff) - min(diff) across this cell's sampled within-episode "
                    "trajectory exceeds the floor -- the instrumentation captured "
                    "real EMA movement, not a frozen/constant reading. Worst seed "
                    "reported.",
        control="the sampled diff series itself; a constant reading anywhere near "
                "zero variation means nothing was actually observed",
        threshold=TRAJECTORY_VARIATION_FLOOR,
        direction="lower",
    ),
    PreconditionSpec(
        name="real_exposure_achieved",
        description="fraction of this cell's nominal STEPS_PER_EP*N_TRAIN_EPS tick "
                    "budget actually executed (final_global_tick_reached + 1) / "
                    "(steps_per_ep * n_train) -- the DESIGN FIX's own falsifiable "
                    "check that STEPS_PER_EP genuinely bounds real exposure rather "
                    "than being silently overridden by env-internal termination, "
                    "the exact defect V3-EXQ-864's autopsy diagnosed. Worst seed "
                    "reported.",
        control="every seed of this cell; this is the regression guard against "
                "reintroducing 864's early-break defect",
        threshold=REAL_EXPOSURE_FLOOR,
        direction="lower",
    ),
)


def _cell_ctx(steps: int, arm_id: str, asym: float) -> Dict[str, object]:
    return {"id": f"{arm_id}::steps{steps}", "steps_per_ep": steps,
            "arm_id": arm_id, "asymmetry": asym}


CELL_CONTEXTS = [_cell_ctx(s, a, x) for (s, a, x) in CELLS]


# ------------------------------------------------------------------ build helpers --
def _make_env(seed: int, dry_run: bool = False) -> CausalGridWorldV2:
    return CausalGridWorldV2(
        seed=seed,
        size=(8 if dry_run else GRID_SIZE),
        num_hazards=3,
        num_resources=3,
        hazard_harm=0.04,
        proximity_harm_scale=0.12,
        proximity_benefit_scale=0.10,
        use_proxy_fields=True,
        resource_respawn_on_consume=True,
    )


def _make_agent(env: CausalGridWorldV2, asym: float) -> REEAgent:
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=32,
        world_dim=32,
        alpha_world=ALPHA_WORLD,
        alpha_self=ALPHA_SELF,
        sws_enabled=True,
        sws_consolidation_steps=SWS_CONSOLIDATION_STEPS,
        rem_enabled=True,
        rem_attribution_steps=REM_ATTRIBUTION_STEPS,
        use_sleep_loop=True,
        sleep_loop_episodes_K=1,
        use_rem_precision_recalibration=True,
        precision_zero_point_ema_alpha=PRECISION_ZERO_POINT_EMA_ALPHA,
        rem_precision_recalibration_step=REM_PRECISION_RECALIBRATION_STEP,
        use_rem_precision_broadcast=False,
        rem_precision_broadcast_gain=0.0,
    )
    cfg.e3.use_waking_confidence_inflation = True
    cfg.e3.waking_confidence_inflation_asymmetry = float(asym)
    cfg.e3.waking_confidence_rv_floor = INFLATION_RV_FLOOR
    cfg.e3.waking_confidence_rv_floor_relative_frac = INFLATION_RV_FLOOR_RELATIVE_FRAC
    cfg.e3.waking_confidence_rv_floor_mode = INFLATION_RV_FLOOR_MODE
    cfg.e3.waking_confidence_rv_floor_softness = INFLATION_RV_FLOOR_SOFTNESS
    cfg.serotonin.tonic_5ht_enabled = True
    return REEAgent(cfg)


def _arm_config_slice(asym: float, steps_per_ep: int, n_train: int) -> Dict:
    return {
        "grid_size": GRID_SIZE,
        "train_steps_per_ep": steps_per_ep,
        "n_train_eps": n_train,
        "lr": LR,
        "alpha_world": ALPHA_WORLD,
        "alpha_self": ALPHA_SELF,
        "sws_enabled": True,
        "rem_enabled": True,
        "use_rem_precision_recalibration": True,
        "sws_consolidation_steps": SWS_CONSOLIDATION_STEPS,
        "rem_attribution_steps": REM_ATTRIBUTION_STEPS,
        "precision_zero_point_ema_alpha": PRECISION_ZERO_POINT_EMA_ALPHA,
        "rem_precision_recalibration_step": REM_PRECISION_RECALIBRATION_STEP,
        "sleep_loop_episodes_K": 1,
        "tonic_5ht_enabled": True,
        "use_rem_precision_broadcast": False,
        "rem_precision_broadcast_gain": 0.0,
        "use_waking_confidence_inflation": True,
        "waking_confidence_inflation_asymmetry": float(asym),
        "waking_confidence_rv_floor": INFLATION_RV_FLOOR,
        "waking_confidence_rv_floor_relative_frac": INFLATION_RV_FLOOR_RELATIVE_FRAC,
        "waking_confidence_rv_floor_mode": INFLATION_RV_FLOOR_MODE,
        "waking_confidence_rv_floor_softness": INFLATION_RV_FLOOR_SOFTNESS,
        "precision_init_baseline": PRECISION_INIT_BASELINE,
        "sample_interval": _sample_interval(steps_per_ep),
        "no_eval_phase": True,
        "trajectory_variation_floor": TRAJECTORY_VARIATION_FLOOR,
        # DESIGN FIX (V3-EXQ-864a): this driver's inner loop does NOT break on
        # env-reported `done` -- see module docstring "DESIGN FIX". Recorded here so
        # a reader of the config slice (and the substrate_hash it feeds) can see the
        # loop-continuation policy without reading the driver source.
        "env_termination_ignored_by_driver": True,
        "real_exposure_floor": REAL_EXPOSURE_FLOOR,
    }


def _read_recalib_metrics(agent: REEAgent) -> Optional[Dict[str, float]]:
    if agent.sleep_loop is None:
        return None
    state = agent.sleep_loop.state
    if state is None or not state.last_metrics:
        return None
    m = dict(state.last_metrics)
    out: Dict[str, float] = {}
    if "mech204_recalibration_fired" in m:
        out["fired"] = float(m.get("mech204_recalibration_fired", 0.0))
    if "mech204_running_variance_before" in m and "mech204_running_variance_after" in m:
        out["rv_before"] = float(m["mech204_running_variance_before"])
        out["rv_after"] = float(m["mech204_running_variance_after"])
    return out or None


def _mean(vals: List[float]) -> float:
    return float(sum(vals) / len(vals)) if vals else 0.0


# ---------------------------------------------------------------------- one cell --
def _run_cell(steps_per_ep: int, arm_id: str, asym: float, seed: int,
              n_train: int, dry_run: bool = False) -> Dict:
    interval = _sample_interval(steps_per_ep)

    with arm_cell(
        seed,
        config_slice=_arm_config_slice(asym, steps_per_ep, n_train),
        script_path=Path(__file__),
        include_driver_script_in_hash=False,  # mint-as-you-go: cross-driver reusable
    ) as cell:
        env = _make_env(seed, dry_run=dry_run)
        agent = _make_agent(env, asym)
        optimizer = optim.Adam(agent.parameters(), lr=LR)

        print(f"Seed {seed} Condition {arm_id}_steps{steps_per_ep} n_train={n_train}",
              flush=True)

        trajectory: List[Dict[str, float]] = []
        recalib_moves: List[float] = []
        recalib_fired = 0
        global_tick = 0
        # DESIGN FIX instrumentation: record how early the environment's OWN
        # termination would have fired (the V3-EXQ-864 defect), even though this
        # driver no longer acts on it. env_done_ticks_total counts every tick across
        # the whole cell where the env reported done=True; env_first_done_tick_ep0 is
        # the first such tick within episode 0 (directly comparable to 864's own
        # ~8-11-tick finding).
        env_done_ticks_total = 0
        env_first_done_tick_ep0: Optional[int] = None
        train_harness = StepHarness(agent, env, train_mode=True, seed=seed)
        for ep in range(n_train):
            agent.reset()  # fires the sleep cycle for the prior episode (K=1)
            rec = _read_recalib_metrics(agent)
            if rec is not None:
                if rec.get("fired", 0.0) > 0.0:
                    recalib_fired += 1
                if "rv_before" in rec and "rv_after" in rec:
                    recalib_moves.append(abs(rec["rv_after"] - rec["rv_before"]))
            _, obs_dict = env.reset()
            train_harness.reset()
            for t in range(steps_per_ep):
                result = train_harness.step(obs_dict)
                optimizer.zero_grad()
                loss = agent.compute_prediction_loss()
                if loss.requires_grad:
                    loss.backward()
                    optimizer.step()
                obs_dict = result.next_obs_dict
                if t % interval == 0 or t == steps_per_ep - 1:
                    rv = float(agent.e3._running_variance)
                    ref = float(agent.e3._wci_symmetric_rv_ref)
                    trajectory.append({
                        "episode_idx": ep,
                        "tick_in_episode": t,
                        "global_tick": global_tick,
                        "rv": rv,
                        "wci_symmetric_rv_ref": ref,
                        "diff": ref - rv,
                    })
                if result.done:
                    env_done_ticks_total += 1
                    if ep == 0 and env_first_done_tick_ep0 is None:
                        env_first_done_tick_ep0 = t
                # DESIGN FIX (V3-EXQ-864a): intentionally NOT breaking here. See
                # module docstring "DESIGN FIX" -- CausalGridWorld.step() computes
                # `done` fresh every call with no special-casing of post-"death"
                # ticks, and neither REEAgent nor StepHarness gates on agent_health,
                # so continuing produces the same per-tick harm_signal/PE dynamics
                # the substrate would generate mid-episode. This is what makes
                # STEPS_PER_EP genuinely bound real exposure (see
                # real_exposure_achieved below), fixing V3-EXQ-864's defect.
                global_tick += 1
            if (ep + 1) % 2 == 0 or ep + 1 == n_train:
                print(
                    f"  [train] arm={arm_id} steps_per_ep={steps_per_ep} seed={seed} "
                    f"ep {ep + 1}/{n_train} "
                    f"rv={float(agent.e3._running_variance):.6f} "
                    f"ref={float(agent.e3._wci_symmetric_rv_ref):.6f} "
                    f"diff={float(agent.e3._wci_symmetric_rv_ref) - float(agent.e3._running_variance):+.6f}",
                    flush=True,
                )

        rv_final = float(agent.e3._running_variance)
        ref_final = float(agent.e3._wci_symmetric_rv_ref)
        diff_final = ref_final - rv_final

        diff_series = [row["diff"] for row in trajectory]
        traj_variation = (max(diff_series) - min(diff_series)) if diff_series else 0.0

        # NEW (864a): the last global_tick actually recorded/stepped this cell.
        # global_tick increments unconditionally every real tick (the loop never
        # breaks), so this is exactly steps_per_ep * n_train - 1 by construction --
        # measured and gated via real_exposure_achieved below rather than assumed,
        # per the "assert the DV actually varies" dry-run rule.
        final_global_tick_reached = global_tick - 1 if global_tick > 0 else -1
        nominal_total_ticks = steps_per_ep * n_train
        real_exposure_frac = (
            (final_global_tick_reached + 1) / nominal_total_ticks
            if nominal_total_ticks > 0 else 0.0
        )

        # Find the first sampled tick (in global ordering) where the sign of diff
        # flips relative to the FIRST sampled tick's sign, if any -- a coarse,
        # descriptive crossover locator, not a statistical test.
        crossover_global_tick = None
        if diff_series:
            first_sign = diff_series[0] >= 0.0
            for row in trajectory:
                if (row["diff"] >= 0.0) != first_sign:
                    crossover_global_tick = row["global_tick"]
                    break

        _moved_from_first_sample = bool(
            diff_series and (diff_series[-1] >= 0.0) != (diff_series[0] >= 0.0))
        print(
            f"  [end]  arm={arm_id} steps_per_ep={steps_per_ep} seed={seed} "
            f"rv_final={rv_final:.6f} ref_final={ref_final:.6f} "
            f"diff_final={diff_final:+.6f} "
            f"crossover_seen={'Y' if _moved_from_first_sample else 'N'} "
            f"final_global_tick={final_global_tick_reached} "
            f"real_exposure_frac={real_exposure_frac:.4f} "
            f"env_done_ticks={env_done_ticks_total} "
            f"env_first_done_tick_ep0={env_first_done_tick_ep0}",
            flush=True,
        )
        print(f"verdict: {'PASS' if traj_variation > TRAJECTORY_VARIATION_FLOOR else 'FAIL'}",
              flush=True)

        _ZG.observe(agent)

        row = {
            "steps_per_ep": steps_per_ep,
            "arm_id": arm_id,
            "seed": seed,
            "inflation_asymmetry": float(asym),
            "n_train_eps": n_train,
            "sample_interval": interval,
            "rv_final_after_training": rv_final,
            "wci_symmetric_rv_ref_after_training": ref_final,
            "diff_final": diff_final,
            "rv_delta_from_precision_init": abs(rv_final - PRECISION_INIT_BASELINE),
            "recalib_cycles_fired": recalib_fired,
            "recalib_mean_abs_move": _mean(recalib_moves),
            "trajectory_variation": traj_variation,
            "trajectory_n_samples": len(trajectory),
            "sign_flip_seen": _moved_from_first_sample,
            "crossover_global_tick": crossover_global_tick,
            # NEW (864a) -- the DESIGN FIX's own instrumentation.
            "final_global_tick_reached": final_global_tick_reached,
            "nominal_total_ticks": nominal_total_ticks,
            "real_exposure_frac": real_exposure_frac,
            "env_done_ticks_total": env_done_ticks_total,
            "env_first_done_tick_ep0": env_first_done_tick_ep0,
            "trajectory": trajectory,
        }
        cell.stamp(row)
    return row


# ---------------------------------------------------------------------- analysis --
def _analyse(cells: List[Dict], seeds: List[int]) -> Dict:
    by_cell: Dict[Tuple[int, str], Dict[int, Dict]] = {}
    for c in cells:
        by_cell.setdefault((c["steps_per_ep"], c["arm_id"]), {})[c["seed"]] = c

    # ---- readiness gates, one arm-gate per (steps_per_ep, arm_id) grid cell ----
    arm_gates = []
    for (steps, arm_id, asym) in CELLS:
        ctx = _cell_ctx(steps, arm_id, asym)
        rows = by_cell[(steps, arm_id)]
        measured: Dict[str, float] = {
            "rv_live": min(rows[s]["rv_delta_from_precision_init"] for s in seeds),
            "f1_recalib_engaged": _mean([rows[s]["recalib_mean_abs_move"] for s in seeds]),
            "trajectory_non_degenerate": min(rows[s]["trajectory_variation"] for s in seeds),
            "real_exposure_achieved": min(rows[s]["real_exposure_frac"] for s in seeds),
        }
        gate_id = f"{arm_id}::steps{steps}"
        arm_gates.append(
            evaluate_arm_gate(gate_id, ctx, list(PRECONDITION_SPECS), measured))

    gate = aggregate_arm_gates(arm_gates)

    # ---- per-cell descriptive summary (mean diff_final over seeds, and whether ANY
    # seed in this cell saw a within-episode sign flip) ----
    per_cell: Dict[str, Dict] = {}
    for (steps, arm_id, asym) in CELLS:
        rows = by_cell[(steps, arm_id)]
        diff_final_mean = _mean([rows[s]["diff_final"] for s in seeds])
        per_cell[f"{arm_id}_steps{steps}"] = {
            "steps_per_ep": steps,
            "arm_id": arm_id,
            "asymmetry": asym,
            "diff_final_mean": diff_final_mean,
            "diff_final_sign": "positive" if diff_final_mean >= 0.0 else "negative",
            "any_seed_sign_flip": any(rows[s]["sign_flip_seen"] for s in seeds),
            "n_seeds": len(rows),
            "final_global_tick_reached_min": min(
                rows[s]["final_global_tick_reached"] for s in seeds),
            "real_exposure_frac_min": min(rows[s]["real_exposure_frac"] for s in seeds),
            "env_done_ticks_total_max": max(
                rows[s]["env_done_ticks_total"] for s in seeds),
        }

    # ---- per-arm crossover bracket across the STEPS_PER_EP sweep (descriptive) ----
    per_arm_bracket: Dict[str, Dict] = {}
    for (arm_id, asym) in ARMS:
        ordered = [per_cell[f"{arm_id}_steps{s}"] for s in STEPS_PER_EP_SWEEP]
        signs = [c["diff_final_sign"] for c in ordered]
        bracket = None
        for i in range(len(ordered) - 1):
            if signs[i] != signs[i + 1]:
                bracket = (STEPS_PER_EP_SWEEP[i], STEPS_PER_EP_SWEEP[i + 1])
                break
        per_arm_bracket[arm_id] = {
            "steps_per_ep_swept": list(STEPS_PER_EP_SWEEP),
            "diff_final_by_steps": {
                str(s): per_cell[f"{arm_id}_steps{s}"]["diff_final_mean"]
                for s in STEPS_PER_EP_SWEEP
            },
            # NEW (864a) -- proves the sweep now varies real exposure (864's own
            # defect made this identical across all three sweep points; here it
            # should scale ~linearly with steps_per_ep).
            "final_global_tick_reached_by_steps": {
                str(s): per_cell[f"{arm_id}_steps{s}"]["final_global_tick_reached_min"]
                for s in STEPS_PER_EP_SWEEP
            },
            "sign_sequence": signs,
            "crossover_bracket_steps_per_ep": list(bracket) if bracket else None,
        }

    # C0 (load-bearing, diagnostic-adjudication sense): the instrumentation captured
    # real, non-degenerate, genuinely-varying-exposure trajectories across the WHOLE
    # sweep -- i.e. this run's data is trustworthy to read at all. Independent of
    # whether a crossover was found.
    c0_data_usable = bool(gate["all_green"])
    # C1 (non-load-bearing, descriptive): at least one arm's sweep brackets a sign
    # change within {200, 500, 1000} -- i.e. this run's chosen sweep actually located
    # the crossover, rather than only bracketing it outside the tested range.
    c1_crossover_bracketed = any(
        per_arm_bracket[a]["crossover_bracket_steps_per_ep"] is not None
        for (a, _) in ARMS
    )

    criteria = [
        {"name": "C0_trajectory_data_usable", "load_bearing": True,
         "passed": c0_data_usable},
        {"name": "C1_crossover_bracketed_in_swept_range", "load_bearing": False,
         "passed": c1_crossover_bracketed,
         "per_arm_bracket": per_arm_bracket},
    ]

    # C0/C1 both read ALL SIX cells jointly (the whole sweep), so they are only
    # non-degenerate if EVERY cell's gate is green -- a criterion owned by every
    # cell is degenerate if any is red. arm_criteria_non_degenerate keys per-arm
    # (here per-cell), so intersect by hand via both_green first (mirrors 860/864's
    # own "C1/C2/C3 all read BOTH arms jointly" comment) and pass a single
    # representative gate_id -- its own green/red status is redundant once
    # raw_non_degen already encodes the full intersection, exactly as in the
    # 860/864 precedent.
    both_green = bool(gate["all_green"])
    raw_non_degen = {
        "C0_trajectory_data_usable": both_green,
        "C1_crossover_bracketed_in_swept_range": both_green,
    }
    _representative_gate_id = arm_gates[0]["arm"] if arm_gates else "?"
    criteria_non_degenerate = arm_criteria_non_degenerate(
        {_representative_gate_id: list(raw_non_degen.keys())}, gate, raw_non_degen)

    readiness_ok = bool(gate["non_degenerate"]) and both_green
    if not readiness_ok:
        label = "substrate_not_ready_requeue"
        outcome = "FAIL"
    elif c1_crossover_bracketed:
        label = "wci_rv_crossover_bracketed_in_swept_range"
        outcome = "PASS"
    else:
        label = "wci_rv_trajectory_characterized_no_crossover_in_range"
        outcome = "PASS"

    per_claim = {"SD-076": "unknown"}  # instrumentation only -- see module docstring

    return {
        "outcome": outcome,
        "label": label,
        "evidence_direction": "unknown",
        "evidence_direction_per_claim": per_claim,
        "criteria": criteria,
        "criteria_non_degenerate": criteria_non_degenerate,
        "gate": gate,
        "arm_gates": arm_gates,
        "per_cell": per_cell,
        "per_arm_bracket": per_arm_bracket,
        "readiness_ok": readiness_ok,
        "thresholds": {
            "RV_LIVE_FLOOR": RV_LIVE_FLOOR,
            "RECALIB_MOVE_FLOOR": RECALIB_MOVE_FLOOR,
            "TRAJECTORY_VARIATION_FLOOR": TRAJECTORY_VARIATION_FLOOR,
            "REAL_EXPOSURE_FLOOR": REAL_EXPOSURE_FLOOR,
            "PRECISION_INIT_BASELINE": PRECISION_INIT_BASELINE,
        },
        "reference_values": {
            "v3_exq_860_inflation_lowers_rv": {"LO": -2.76e-4, "HI": -1.46e-4},
            "v3_exq_850_853_inflation_lowers_rv_context": "small positive, sub-threshold (exact values in those manifests)",
            "v3_exq_864_diff_final_by_steps_defective": {
                "note": "864's own sweep was bit-identical across all three "
                        "STEPS_PER_EP values in both arms/seeds (e.g. ARM_INFL_LO: "
                        "-0.008332025904674908 at 200, 500, AND 1000) -- the defect "
                        "this redesign fixes. Not a valid comparison point for "
                        "diff_final; the final_global_tick_reached_by_steps block "
                        "above is the direct evidence this redesign does not share "
                        "that defect.",
            },
        },
    }


# -------------------------------------------------------------------------- main --
def run_experiment(dry_run: bool = False) -> Dict:
    t0 = time.perf_counter()
    n_train = 3 if dry_run else N_TRAIN_EPS
    n_seeds = 2 if dry_run else N_SEEDS
    steps_sweep = (20, 40) if dry_run else STEPS_PER_EP_SWEEP
    seeds = list(range(n_seeds))

    if not dry_run:
        assert N_TRAIN_EPS >= 4, (
            f"Need at least a few F1-recalibration cycles to see the within-episode "
            f"pattern settle; got N_TRAIN_EPS={N_TRAIN_EPS}"
        )
        assert list(STEPS_PER_EP_SWEEP) == sorted(STEPS_PER_EP_SWEEP), (
            "STEPS_PER_EP_SWEEP must be ascending for the crossover-bracket logic"
        )

    cells_ctx = [
        (steps, arm_id, asym)
        for steps in steps_sweep
        for (arm_id, asym) in ARMS
    ] if dry_run else list(CELLS)

    assert_no_structurally_unsatisfiable_gate(list(PRECONDITION_SPECS), [
        _cell_ctx(s, a, x) for (s, a, x) in cells_ctx
    ])

    cells: List[Dict] = []
    for (steps, arm_id, asym) in cells_ctx:
        for seed in seeds:
            cells.append(_run_cell(steps, arm_id, asym, seed, n_train, dry_run=dry_run))

    # DESIGN FIX regression guard (dry-run AND full run): confirm the sweep now
    # actually varies real exposure across at least two distinct swept values, per
    # the queue-experiment skill's dose-sweep smoke rule. This is exactly the check
    # that would have caught V3-EXQ-864's defect before a multi-hour run.
    _reached_by_steps: Dict[int, List[int]] = {}
    for c in cells:
        _reached_by_steps.setdefault(c["steps_per_ep"], []).append(
            c["final_global_tick_reached"])
    _distinct_reached = {min(v) for v in _reached_by_steps.values()}
    assert len(_distinct_reached) >= min(2, len(_reached_by_steps)), (
        f"DESIGN FIX REGRESSION: final_global_tick_reached does not vary across "
        f"swept steps_per_ep values ({_reached_by_steps}) -- this reproduces the "
        f"V3-EXQ-864 defect (env termination silently overriding the swept "
        f"parameter). Check that the inner loop in _run_cell is not breaking on "
        f"result.done."
    )

    if dry_run:
        # Dry-run uses a different (smaller) steps sweep, so the full-sweep bracket
        # analysis (keyed to STEPS_PER_EP_SWEEP) doesn't apply -- just confirm cells ran.
        adj = {
            "outcome": "PASS" if all(c["trajectory_variation"] >= 0.0 for c in cells) else "FAIL",
            "label": "dry_run_smoke",
            "criteria": [],
            "readiness_ok": True,
        }
    else:
        adj = _analyse(cells, seeds)
    adj["cells"] = cells
    adj["seeds"] = seeds
    adj["elapsed_seconds"] = time.perf_counter() - t0
    adj["t0_perf"] = t0
    adj["config_n"] = {"n_train_eps": n_train, "n_seeds": n_seeds,
                       "steps_per_ep_sweep": list(steps_sweep)}
    return adj


def main(dry_run: bool = False) -> Dict:
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{ts}_v3"
    adj = run_experiment(dry_run=dry_run)
    outcome = adj["outcome"]

    print("", flush=True)
    print(f"label={adj['label']} outcome={outcome} readiness_ok={adj.get('readiness_ok')}",
          flush=True)
    if not dry_run:
        for arm_id, bracket in adj["per_arm_bracket"].items():
            print(f"  {arm_id}: diff_final_by_steps={bracket['diff_final_by_steps']} "
                  f"final_global_tick_reached_by_steps="
                  f"{bracket['final_global_tick_reached_by_steps']} "
                  f"crossover_bracket={bracket['crossover_bracket_steps_per_ep']}",
                  flush=True)
        for c in adj["criteria"]:
            lb = " (load-bearing)" if c["load_bearing"] else ""
            print(f"  {c['name']}: {'PASS' if c['passed'] else 'FAIL'}{lb}", flush=True)

    if dry_run:
        print("DRY_RUN_COMPLETE", flush=True)
        return {"outcome": outcome, "manifest_path": None, "run_id": run_id}

    full_config = {
        "grid_size": GRID_SIZE,
        "n_train_eps": adj["config_n"]["n_train_eps"],
        "n_train_eps_v3_exq_860_baseline": 30,
        "steps_per_ep_sweep": adj["config_n"]["steps_per_ep_sweep"],
        "n_seeds": adj["config_n"]["n_seeds"],
        "lr": LR,
        "alpha_world": ALPHA_WORLD,
        "alpha_self": ALPHA_SELF,
        "sws_consolidation_steps": SWS_CONSOLIDATION_STEPS,
        "rem_attribution_steps": REM_ATTRIBUTION_STEPS,
        "precision_zero_point_ema_alpha": PRECISION_ZERO_POINT_EMA_ALPHA,
        "rem_precision_recalibration_step": REM_PRECISION_RECALIBRATION_STEP,
        "use_rem_precision_broadcast": False,
        "inflation_asymmetry_lo": INFLATION_ASYMMETRY_LO,
        "inflation_asymmetry_hi": INFLATION_ASYMMETRY_HI,
        "inflation_rv_floor": INFLATION_RV_FLOOR,
        "waking_confidence_rv_floor_relative_frac": INFLATION_RV_FLOOR_RELATIVE_FRAC,
        "waking_confidence_rv_floor_mode": INFLATION_RV_FLOOR_MODE,
        "waking_confidence_rv_floor_softness": INFLATION_RV_FLOOR_SOFTNESS,
        "sleep_loop_episodes_K": 1,
        "tonic_5ht_enabled": True,
        "samples_per_episode_target": SAMPLES_PER_EPISODE_TARGET,
        "env_termination_ignored_by_driver": True,
        "real_exposure_floor": REAL_EXPOSURE_FLOOR,
        "arms": [{"arm_id": a[0], "inflation_asymmetry": float(a[1])} for a in ARMS],
        "env": {"num_hazards": 3, "num_resources": 3, "hazard_harm": 0.04,
                "proximity_harm_scale": 0.12, "proximity_benefit_scale": 0.10,
                "use_proxy_fields": True, "resource_respawn_on_consume": True},
        "seeds": adj["seeds"],
    }

    manifest = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": QUEUE_ID,
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "architecture_epoch": ARCHITECTURE_EPOCH,
        "timestamp_utc": datetime.utcnow().strftime("%Y%m%dT%H%M%SZ"),
        "outcome": outcome,
        "result": outcome,
        "evidence_direction": adj["evidence_direction"],
        "evidence_direction_per_claim": adj["evidence_direction_per_claim"],
        "sleep_driver_pattern": SLEEP_DRIVER_PATTERN,
        "supersedes": SUPERSEDES,
        "fanout_source": FANOUT_SOURCE,
        "puzzle_source_run_id": PUZZLE_SOURCE_RUN_ID,
        "puzzle_source_queue_id": PUZZLE_SOURCE_QUEUE_ID,
        "redesign_source_run_id": REDESIGN_SOURCE_RUN_ID,
        "redesign_source_queue_id": REDESIGN_SOURCE_QUEUE_ID,
        "interpretation": {
            "label": adj["label"],
            "preconditions": adj["gate"]["adjudication_preconditions"],
            "criteria": adj["criteria"],
            "criteria_non_degenerate": adj["criteria_non_degenerate"],
        },
        "per_arm_gate": adj["gate"]["per_arm_gate"],
        "non_degenerate": adj["gate"]["non_degenerate"],
        "degeneracy_reason": adj["gate"]["degeneracy_reason"],
        "aggregates": {
            "per_cell": adj["per_cell"],
            "per_arm_bracket": adj["per_arm_bracket"],
            "readiness_ok": adj["readiness_ok"],
        },
        "thresholds": adj["thresholds"],
        "reference_values": adj["reference_values"],
        "arm_results": adj["cells"],
        "per_seed_cells": adj["cells"],
        "elapsed_seconds": adj["elapsed_seconds"],
        "notes": (
            "PURE INSTRUMENTATION/MEASUREMENT DIAGNOSTIC (experiment_purpose=diagnostic, "
            "excluded from confidence/conflict scoring), superseding V3-EXQ-864. "
            "Investigates Finding B of failure_autopsy_V3-EXQ-860_2026-08-01.md: the "
            "wrong-signed inflation_lowers_rv precondition (rv ended up ABOVE the "
            "un-inflated counterfactual at STEPS_PER_EP=1000, opposite the intended "
            "direction, vs small-positive-but-sub-threshold at STEPS_PER_EP=200 in the "
            "850/853 sibling runs). V3-EXQ-864's own confirmed autopsy "
            "(failure_autopsy_V3-EXQ-864_2026-08-01.md) found that 864's sweep never "
            "actually varied real exposure -- CausalGridWorld's own episode termination "
            "fired at ~8-11 ticks/episode, far short of any swept STEPS_PER_EP value. "
            "This redesign decouples STEPS_PER_EP from env-internal termination by no "
            "longer breaking the per-episode loop on env-reported done (see module "
            "docstring DESIGN FIX for why this is safe and why the alternative repair "
            "strategies were rejected) -- final_global_tick_reached in "
            "aggregates.per_arm_bracket now genuinely scales with STEPS_PER_EP, unlike "
            "864's bit-identical readout. Same REDUCED N_TRAIN_EPS=8 (vs 860's 30), same "
            "two inflation doses, same ~40-samples-per-episode within-episode logging as "
            "864. NOT a re-test of the H1/H2/H3 discrimination and NOT new SD-076/MECH-204 "
            "evidence -- claim_ids=[SD-076] reflects only that this characterizes SD-076's "
            "own EMA dynamics. DOES NOT BLOCK the EVB-0454 SD-076 decision: that decision "
            "proceeds on V3-EXQ-860's Finding A (the closure-fraction comparison), which "
            "this diagnostic does not touch. See aggregates.per_arm_bracket for the "
            "STEPS_PER_EP interval (if any) in which each arm's diff_final crossed sign, "
            "and arm_results[*].trajectory for the raw within-episode series."
        ),
    }

    out_dir = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments" / EXPERIMENT_TYPE
    out_path = write_flat_manifest(
        manifest,
        out_dir,
        dry_run=False,
        config=full_config,
        seeds=adj["seeds"],
        script_path=Path(__file__),
        started_at=adj["t0_perf"],
        z_goal_stream_stats=_ZG.stats(),
    )
    print(f"Result written to: {out_path}", flush=True)
    return {"outcome": outcome, "manifest_path": str(out_path), "run_id": run_id}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=EXPERIMENT_TYPE)
    parser.add_argument("--dry-run", action="store_true", help="Quick smoke test (tiny).")
    args = parser.parse_args()
    result = main(dry_run=args.dry_run)
    _outcome = str(result["outcome"]).upper()
    emit_outcome(
        outcome=_outcome if _outcome in ("PASS", "FAIL") else "FAIL",
        manifest_path=result["manifest_path"],
        run_id=result["run_id"],
        dry_run=args.dry_run,
    )
    sys.exit(0)
