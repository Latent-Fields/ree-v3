"""
V3-EXQ-934 (MECH-266 / SD-032a, GOV-FANOUT-1 leg H1): affinity_input_cap
calibration sweep for external_task mode-occupancy. DIAGNOSTIC.

SLEEP DRIVER: N/A (waking goal-pipeline onboarding scheduler; no sleep loop).

WHY THIS RUN EXISTS
-------------------
V3-EXQ-464e / 467e (2026-08-13, cluster autopsy
`failure_autopsy_mech266-464e-467e-cluster_2026-08-13`) found that
`affinity_input_cap = 2.0` produces BANG-BANG mode arbitration on the full
scaffolded curriculum: 19 of 21 arm/ratio cells sat at exactly 0.0 or 1.0
external_task occupancy, no mixed regime in which a graded exit threshold could
express. The cluster autopsy pre-registered a three-leg discrimination portfolio
(frozen-ledger question `mech266_mode_arbitration_saturation`):
  H1 (this run) -- cap MIS-CALIBRATION: some cap admits a mixed regime (axis:
     representation).
  H2 -- STRUCTURAL bang-bang: SD-032a's discrete argmax cannot grade at any cap
     (axis: constitution).
  H3 -- INSTRUMENTATION: the min()-across-the-sweep gate is ill-posed (axis:
     instrumentation; re-scores banked data at zero compute -- resolved by
     `experiments/_lib/regime_occupancy_gate.py`, which THIS run consumes).

A synthetic single-tick + stateful probe against the REAL SalienceCoordinator
class (substrate_queue.json `mode-governance-engagement`
`implementation_log.cap_calibration_probe_2026_08_13`, session
jovial-shannon-35d300) found the continuous `operating_mode['external_task']`
signal IS genuinely graded across cap ~0.25-2.0, reproduced the observed cap=2.0
saturation, and produced a genuinely mixed dynamic regime at cap ~1.0-1.5. It
recommended the next real cap-sweep experiment (i) sweep cap densely in
[0.75, 1.0, 1.25, 1.5, 1.75] and (ii) additionally instrument the CONTINUOUS
pre-argmax `operating_mode['external_task']`, not only the discrete occupancy
fraction. This run does exactly that on the full 3-seed curriculum.

NOT A NAKED 464f RE-QUEUE. The cluster autopsy's re-derive brake REFUSES a
naked V3-EXQ-464f/467f re-queue (same question, same min()-gate, same substrate)
but explicitly PERMITS this H1 leg: a DIAGNOSTIC that sweeps the cap and reads
per-arm occupancy (never `min`) through the regime-conditioned gate. The design
axis (representation / cap calibration), DV (per-cap/per-arm occupancy + the
continuous margin), and gate (regime_occupancy_gate, not min-across-the-sweep)
all differ from 464e/467e's evidence-purpose rail-contrast.

DESIGN
------
Sweep `affinity_input_cap` in CAP_SWEEP = [0.75, 1.0, 1.25, 1.5, 1.75] at EVAL
time on clones of a SINGLE trained curriculum agent per seed. This is valid and
economical: `SalienceCoordinator.tick()` reads `self.config.affinity_input_cap`
LIVE at every tick (salience_coordinator.py:455), so overriding
`coord.config.affinity_input_cap` on a clone changes the arbitration output with
no retraining -- exactly the isolation the synthetic probe used, and the same
train-once/sweep-conditions-on-clones pattern V3-EXQ-467e uses for its
hysteresis-ratio sweep. Training is held at `AFFINITY_INPUT_CAP_TRAIN = 2.0`
(464e's construction) so the trained substrate is comparable to the banked
464e/467e reference; only the eval-time clamp is swept. (Empirically confirmed
before authoring: a one-tick probe against the real SalienceCoordinator showed
`operating_mode['external_task']` = 0.0 at cap=None/dacc_pe=16 and 0.79-0.86
across the [0.75,1.75] band, and that the live cap override takes effect at tick
time.)

Cells = CAP x ARM per seed. Two rail arms, unchanged from 464e:
  ARM_SYMMETRIC        -- legacy MECH-259 (no per-mode rails). The neutral
                          arbitration baseline -- the clean place to ask "does
                          the cap ALONE admit a mixed regime". PRIMARY for the
                          H1 verdict.
  ARM_ASYM_STICKY_TASK -- sticky exit rail on external_task (MECH-266
                          over-binding manipulation). Reported per-arm as
                          context; NOT load-bearing for H1 -- the sticky rail
                          deliberately pushes occupancy toward saturation, so a
                          saturated reading there is the intended manipulation,
                          not evidence against H1.
use_closure_operator OFF (closure injects a confounding mode-switch signal;
464b/c/d/e omitted it for the same reason).

DEPENDENT VARIABLES (per seed x cap x arm cell)
  fraction_in_external_task -- discrete committed-mode occupancy (coord.current_mode
      == external_task fraction of env steps). This is the SAME statistic 464e
      measured and the one the H1 null is stated in.
  ext_margin_mean / p10 / p50 / p90 -- the CONTINUOUS pre-argmax
      operating_mode['external_task'] value, aggregated over env steps (the
      probe's / H2's recommended instrumentation). Discriminates "external_task
      drive not engaging at all" (margin ~0 -> substrate_not_ready) from "drive
      engages continuously but never wins the committed argmax" (margin high,
      occupancy saturated -> STRUCTURAL, H2-favourable).
  ext_dwell_mean -- MODE-CONDITIONED mean dwell in external_task specifically
      (fixes 464e's M3 mode-agnostic dwell defect; a dwell run is counted only
      while current_mode == external_task).
  n_switches, coord_n_ticks -- switch count + the coordinator's own tick count
      (so the continuous-margin denominator is auditable, per the E3-diagnostics
      sample-integrity rule).

READINESS / NON-VACUITY GATES (route a not-ready read to
substrate_not_ready_requeue, NEVER a false verdict):
  G-contact  -- 603n contact guard: per-seed P2 contact_rate > 0 AND
      z_goal_norm_at_contact_peak > 0.4 on >= 2/3 seeds. Below -> not ready.
  G-margin   -- the external_task drive must ENGAGE at some cap: per-seed
      max-over-cells of ext_margin_mean > MARGIN_FLOOR. If the continuous margin
      is ~0 at every cap/arm, the drive is not producing the signal (a
      substrate/wiring problem, e.g. the 464d goal_state-drop or a dead drive),
      NOT structural discreteness -- so this self-routes to
      substrate_not_ready_requeue and does NOT get read as H2.

REGIME GATE (the H3-fix primitive; replaces min-across-the-sweep):
  Per (seed, arm), build one OccupancyCell per cap value and call
  evaluate_regime_occupancy_gate(floor=OCCUPANCY_FLOOR=0.1,
  ceiling=OCCUPANCY_CEILING=0.9). regime_shape in
  {unreachable, saturated_bimodal, graded}. "graded" == at least one cap yields
  occupancy strictly in (0.1, 0.9) -- exactly the null's mixed band.

H1 VERDICT (diagnostic; the null the autopsy pre-registered):
  NULL: "No cap value yields per-arm occupancy in (0.1, 0.9) on >= 2/3 seeds."
  H1 SUPPORTED (null rejected) iff, on the SYMMETRIC arm (primary), the regime is
  graded on >= 2/3 guard-passing seeds. The asym arm is reported but not
  load-bearing. If graded -> PASS, route
  `cap_recalibration_admits_mixed_regime`, and report the winning cap band(s)
  (the caps whose occupancy fell in the mixed band). If every cap is
  saturated_bimodal (with G-margin PASSED, so the drive DOES engage) ->
  FAIL/route `saturated_all_caps_supports_structural_discreteness` (H1 not
  supported, H2-favourable -- a real finding, not not-ready).

PER-CLAIM DIRECTION: diagnostic, EXCLUDED from governance confidence/conflict
scoring (experiment_purpose="diagnostic"). Directions are diagnostic context
only: MECH-266 non_contributory (this run establishes the measurement
precondition for the over-binding test; it does not itself measure over-binding);
SD-032a supports if a graded regime is reachable (the mode register CAN grade,
recalibration viable) else weakens (favours structural discreteness). Overall
non_contributory.

claim_ids: MECH-266, SD-032a.
experiment_purpose: diagnostic
predecessor: V3-EXQ-464e / V3-EXQ-467e (successor-in-spirit; NOT a supersede).
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "experiments") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "experiments"))

from experiment_protocol import emit_outcome  # noqa: E402
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402
from scaffolded_sd054_onboarding import (  # noqa: E402
    ScaffoldedSD054OnboardingConfig,
    ScaffoldedSD054OnboardingScheduler,
    _derive_env_seed,
    _sd049_kwargs,
    _sense_with_optional_harm,
    stage_plan,
)
from experiments.pack_writer import write_flat_manifest  # noqa: E402
from experiments._lib.z_goal_stream import ZGoalStreamAccumulator  # noqa: E402
from experiments._lib.regime_occupancy_gate import (  # noqa: E402
    OccupancyCell,
    classify_regime_shape,
    evaluate_regime_occupancy_gate,
)

EXPERIMENT_TYPE = "v3_exq_934_mech266_cap_sweep_mode_occupancy"
QUEUE_ID = "V3-EXQ-934"
CLAIM_IDS: List[str] = ["MECH-266", "SD-032a"]
EXPERIMENT_PURPOSE = "diagnostic"
PREDECESSOR = "V3-EXQ-464e / V3-EXQ-467e (successor-in-spirit; NOT a supersede)"

# The two readiness anchors here (foraging_contact_guard, external_task_drive_engages)
# are ordinary UPSTREAM readiness gates -- "did the curriculum train" and "does the
# external_task drive engage at all" -- NOT the "positive control reproduces a known
# degenerate signature" pattern the V3-EXQ-778d readiness-anchor reachability guard
# targets. Their thresholds are reachable BY CONSTRUCTION: the 603n contact guard was
# cleared by 464e on this exact curriculum, and the MARGIN_FLOOR=0.05 continuous-margin
# floor was cleared ~15x by the pre-authoring one-tick probe against the real
# SalienceCoordinator (operating_mode['external_task'] = 0.76-0.86 across the swept
# band). Crucially, the 778d failure mode -- an unmeetable predicate that reports
# met=false forever and mislabels an instrument gap as a substrate verdict -- does NOT
# apply: when the margin gate PASSES while occupancy nonetheless saturates, the run
# routes `saturated_all_caps_supports_structural_discreteness` (H1 not supported /
# H2-favourable), which is a VALID finding (the drive provably engaged), not a starved
# false-falsification of a test that never ran. The margin gate is deliberately a
# DIFFERENT, weaker readout than the occupancy DV precisely so it can distinguish
# not-ready (margin ~0) from ready-but-structural (margin high, occupancy saturated).
ANCHOR_REACHABILITY_EXEMPT = (
    "Readiness anchors are ordinary upstream gates (curriculum-trained; drive-engages), "
    "not degeneracy-reproduction anchors; thresholds reachable by construction (contact "
    "guard cleared by 464e on this curriculum; 0.05 margin floor cleared ~15x by the "
    "pre-authoring one-tick probe). A margin-pass-with-saturated-occupancy read routes a "
    "VALID H2-favourable structural finding, never a starved false-falsification -- the "
    "778d unmeetable-predicate failure mode does not apply."
)

SEEDS = [42, 43, 44]
CONDITION_LABEL = "CURRICULUM_BUILT_CAP_SWEEP"

# z_goal stream liveness (Experimental Recording Standard). goal_state.is_active()
# gating is exactly the mechanism the _clone_for_arm goal_state fix (ported from
# 464e) addresses. Module-level accumulator; agents fall out of scope per seed.
_ZG = ZGoalStreamAccumulator()

MODE_NAMES = [
    "external_task",
    "internal_planning",
    "internal_replay",
    "offline_consolidation",
]
STICKY_MODE = "external_task"
STICKY_EXIT = 0.05
LOOSE_EXIT = 0.90

# The H1 cap band, from the cap_calibration_probe_2026_08_13 recommendation --
# the range where the synthetic model showed real switching dynamics. Deliberately
# NOT re-testing cap=2.0 (the saturated reference is already banked in 464e/467e)
# and NOT widening the range without cause (per the chip + probe).
CAP_SWEEP: List[float] = [0.75, 1.0, 1.25, 1.5, 1.75]
# Training-time cap: held at 464e's construction value so the trained substrate is
# comparable to the banked 464e/467e reference. Only the EVAL-time clamp is swept.
AFFINITY_INPUT_CAP_TRAIN = 2.0

# The null's mixed band: per-arm occupancy in (0.1, 0.9). The regime gate calls a
# cell "graded" iff occupancy is strictly inside (OCCUPANCY_FLOOR, OCCUPANCY_CEILING).
OCCUPANCY_FLOOR = 0.10
OCCUPANCY_CEILING = 0.90
# G-margin: the continuous external_task margin must clear this at SOME cap on a
# seed, else the drive is not engaging and the read is substrate_not_ready (never
# read as H2 structural). A conservative floor -- well below the argmax boundary.
MARGIN_FLOOR = 0.05

ARM_LABELS = ["ARM_SYMMETRIC", "ARM_ASYM_STICKY_TASK"]

WORLD_DIM = 32
HARM_A_DIM = 16
HARM_OBS_A_DIM = 7
HARM_HISTORY_LEN = 10
DRIVE_WEIGHT = 2.0

STAGE0_BUDGET = 20
STAGE0B_BUDGET = 10
P0_BUDGET = 100
HAZARD_STAGE_BUDGET = 40
P1_BUDGET = 50
P2_BUDGET = 15
MODE_EVAL_EPISODES = 15
TRAIN_STEPS = 200
P1_HOLD_FRACTION = 0.3
P0_NUM_HAZARDS = 1
P2_HFA_GUARD = 0.3
P1_REEF_SPAWN_HOLD_FRACTION = 0.4

HAZARD_STAGE_NUM_HAZARDS = 4
HAZARD_STAGE_NUM_RESOURCES = 2
HAZARD_STAGE_HFA = 0.0
HAZARD_STAGE_PROXIMITY_HARM = 0.1
HAZARD_STAGE_SPAWN_IN_REEF = True
HAZARD_STAGE_SURVIVAL_GATE_STEPS = 75
HAZARD_STAGE_STABILITY_WINDOW = 10

SEED_GAIN = 1.5
SEED_BENEFIT_THRESHOLD = 0.02
SEED_DRIVE_FLOOR = 0.9
N_RESOURCE_TYPES = 3
CUE_RECALL_GAIN = 0.2

AVOIDANCE_SCAFFOLD_FLOOR_START = 0.8
AVOIDANCE_SCAFFOLD_FLOOR_END = 0.0
AVOIDANCE_THREAT_REF = 0.35
PAG_THETA_FREEZE = 0.8
PAG_DURATION_INPUT_THRESHOLD = 0.2
HARM_PATHWAY_LR = 1e-3
STAGE0B_RETENTION_GATE = 0.75

P2_ZGOAL_GATE = 0.4
CONTACT_GATE = 0.0
MIN_FRACTION = 2.0 / 3.0


def _make_scaffold_cfg(dry_run: bool,
                       env_seed: Optional[int] = None) -> ScaffoldedSD054OnboardingConfig:
    if dry_run:
        stage0, stage0b, p0, hazard, p1, p2, steps = 2, 2, 5, 5, 5, 2, 30
    else:
        stage0, stage0b, p0, hazard, p1, p2, steps = (
            STAGE0_BUDGET, STAGE0B_BUDGET, P0_BUDGET, HAZARD_STAGE_BUDGET,
            P1_BUDGET, P2_BUDGET, TRAIN_STEPS,
        )
    cfg = ScaffoldedSD054OnboardingConfig(
        use_scaffolded_sd054_onboarding_scheduler=True,
        scaffold_stage0_enabled=True,
        scaffold_stage0_episode_budget=stage0,
        scaffold_p0_episode_budget=p0,
        scaffold_p1_episode_budget=p1,
        scaffold_p2_episode_budget=p2,
        scaffold_steps_per_episode=steps,
        scaffold_p0_num_hazards=P0_NUM_HAZARDS,
        scaffold_p1_anneal_hold_fraction=P1_HOLD_FRACTION,
        scaffold_p2_hazard_food_attraction_guard=P2_HFA_GUARD,
        scaffold_developmental_window_enabled=True,
        scaffold_stage0b_enabled=True,
        scaffold_stage0b_episode_budget=stage0b,
        scaffold_stage0b_retention_gate=STAGE0B_RETENTION_GATE,
        scaffold_contact_gated_goal_updates=True,
        scaffold_z_goal_seeding_gain=SEED_GAIN,
        scaffold_benefit_threshold=SEED_BENEFIT_THRESHOLD,
        scaffold_drive_floor=SEED_DRIVE_FLOOR,
        scaffold_auto_reconcile_gating_to_seeding=True,
        scaffold_p1_reef_spawn_hold_fraction=P1_REEF_SPAWN_HOLD_FRACTION,
        scaffold_cue_recall_bridge_enabled=True,
        scaffold_cue_n_resource_types=N_RESOURCE_TYPES,
        scaffold_stage0_bind_incentive_token=True,
        scaffold_hazard_stage_enabled=True,
        scaffold_hazard_stage_episode_budget=hazard,
        scaffold_hazard_stage_num_hazards=HAZARD_STAGE_NUM_HAZARDS,
        scaffold_hazard_stage_num_resources=HAZARD_STAGE_NUM_RESOURCES,
        scaffold_hazard_stage_hazard_food_attraction=HAZARD_STAGE_HFA,
        scaffold_hazard_stage_proximity_harm_scale=HAZARD_STAGE_PROXIMITY_HARM,
        scaffold_hazard_stage_spawn_in_reef_half=HAZARD_STAGE_SPAWN_IN_REEF,
        scaffold_hazard_stage_survival_gate_steps=HAZARD_STAGE_SURVIVAL_GATE_STEPS,
        scaffold_hazard_stage_stability_window=HAZARD_STAGE_STABILITY_WINDOW,
        scaffold_avoidance_driver_enabled=True,
        scaffold_avoidance_scaffold_floor_start=AVOIDANCE_SCAFFOLD_FLOOR_START,
        scaffold_avoidance_scaffold_floor_end=AVOIDANCE_SCAFFOLD_FLOOR_END,
        scaffold_feed_harm_stream=True,
        scaffold_train_harm_pathway=True,
        scaffold_harm_pathway_lr=HARM_PATHWAY_LR,
        scaffold_harm_pathway_in_p0=True,
        scaffold_env_seed=env_seed,
    )
    if steps < 75:
        cfg.scaffold_p1_survival_gate_steps = max(1, steps // 4)
        cfg.scaffold_hazard_stage_survival_gate_steps = max(1, steps // 4)
    return cfg


def _make_config(env) -> REEConfig:
    """603n-validated foraging substrate + SalienceCoordinator + dACC + LateralPFC +
    bistable. use_closure_operator OFF (closure would inject a confounding
    mode-switch signal). salience_affinity_input_cap set to AFFINITY_INPUT_CAP_TRAIN
    (= 2.0, 464e's value) at TRAINING time; the EVAL cap is swept per cell by
    overriding coord.config.affinity_input_cap on the clone."""
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=32,
        world_dim=WORLD_DIM,
        alpha_world=0.9,
        use_harm_stream=True,
        use_affective_harm_stream=True,
        z_harm_a_dim=HARM_A_DIM,
        harm_obs_a_dim=HARM_OBS_A_DIM,
        harm_history_len=HARM_HISTORY_LEN,
        use_e2_harm_s_forward=True,
        use_support_preserving_cem=True,
        support_preserving_stratified_elites=True,
        support_preserving_ao_std_floor=0.2,
        support_preserving_min_first_action_classes=2,
        z_goal_enabled=True,
        drive_weight=DRIVE_WEIGHT,
        use_mech295_liking_bridge=True,
        use_mech307_conjunction=True,
        use_incentive_token_bank=True,
        use_cue_recall=True,
        cue_recall_gain=CUE_RECALL_GAIN,
        e2_action_contrastive_enabled=True,
        use_pag_freeze_gate=True,
        pag_theta_freeze=PAG_THETA_FREEZE,
        pag_duration_input_threshold=PAG_DURATION_INPUT_THRESHOLD,
        use_instrumental_avoidance=True,
        avoidance_threat_ref=AVOIDANCE_THREAT_REF,
        use_dacc=True,
        use_salience_coordinator=True,
        use_lateral_pfc_analog=True,
        use_closure_operator=False,
        use_external_task_drive=True,
        external_task_drive_affinity_weight=3.0,
        external_task_drive_salience_weight=2.0,
        external_task_drive_commit_weight=1.0,
        external_task_drive_proximity_weight=1.0,
        salience_affinity_input_cap=AFFINITY_INPUT_CAP_TRAIN,
    )
    cfg.latent.use_resource_encoder = True
    cfg.heartbeat.beta_gate_bistable = True
    return cfg


def _build_dual_cue_env(scaffold_cfg: ScaffoldedSD054OnboardingConfig,
                       seed: Optional[int] = None) -> CausalGridWorldV2:
    """P2-config foraging env WITH the GAP-3 dual_cue primitive (competing goals),
    identical to 464e. `seed` default None passes through to CausalGridWorldV2's
    OS-entropy default -- bit-identical to the landed 464e/467e eval env layout, so
    the cap-sweep results stay comparable to the banked reference."""
    p2_hfa = (
        scaffold_cfg.scaffold_p2_hazard_food_attraction_guard
        if scaffold_cfg.scaffold_p2_hazard_food_attraction_guard >= 0.0
        else scaffold_cfg.scaffold_p2_hazard_food_attraction
    )
    return CausalGridWorldV2(
        seed=seed,
        size=scaffold_cfg.scaffold_env_size,
        num_hazards=scaffold_cfg.scaffold_p2_num_hazards,
        num_resources=scaffold_cfg.scaffold_p2_num_resources,
        hazard_food_attraction=p2_hfa,
        proximity_harm_scale=scaffold_cfg.scaffold_p2_proximity_harm_scale,
        limb_damage_enabled=True,
        reef_enabled=True,
        reef_bipartite_layout=True,
        reef_bipartite_axis=scaffold_cfg.scaffold_reef_bipartite_axis,
        reef_bipartite_agent_band_radius=scaffold_cfg.scaffold_reef_bipartite_agent_band_radius,
        reef_bipartite_agent_spawn_in_reef_half=False,
        dual_cue_enabled=True,
        dual_cue_min_active_ticks=10,
        dual_cue_replace_on_early_consume=False,
        dual_cue_type_tags=(1, 2),
        **_sd049_kwargs(scaffold_cfg),
    )


def _clone_for_arm(trained_agent: REEAgent, device: torch.device) -> REEAgent:
    """Clone the SAME trained weights into a fresh agent (rails + eval cap applied by
    the caller). Also clones goal_state -- the 464e fix: GoalState is a plain Python
    object, invisible to state_dict(), so a weights-only clone dropped its z_goal
    attractor and hard-gated external_task_drive engagement to 0.0 for the eval."""
    cfg = copy.deepcopy(trained_agent.config)
    agent = REEAgent(cfg).to(device)
    state = {k: v.detach().clone() for k, v in trained_agent.state_dict().items()}
    try:
        agent.load_state_dict(state)
    except RuntimeError:
        agent.load_state_dict(state, strict=False)
    agent.e3._running_variance = float(trained_agent.e3._running_variance)
    if trained_agent.goal_state is not None and agent.goal_state is not None:
        agent.goal_state.load_state_dict(trained_agent.goal_state.state_dict())
    return agent


def _apply_symmetric(coord) -> None:
    coord.config.enter_thresholds = {}
    coord.config.exit_thresholds = {}


def _apply_asymmetric_sticky_task(coord) -> None:
    coord.config.enter_thresholds = {}
    coord.config.exit_thresholds = {}
    for mode in MODE_NAMES:
        coord.set_exit_threshold(mode, STICKY_EXIT if mode == STICKY_MODE else LOOSE_EXIT)


def _apply_rails(coord, arm_label: str) -> None:
    if arm_label == "ARM_SYMMETRIC":
        _apply_symmetric(coord)
    elif arm_label == "ARM_ASYM_STICKY_TASK":
        _apply_asymmetric_sticky_task(coord)
    else:
        raise ValueError(f"unknown arm {arm_label}")


def _quantile(sorted_vals: List[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    pos = q * (len(sorted_vals) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = pos - lo
    return float(sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac)


def _eval_cap_cell(
    agent: REEAgent,
    env: CausalGridWorldV2,
    cap: float,
    arm_label: str,
    scaffold_cfg: ScaffoldedSD054OnboardingConfig,
    device: torch.device,
    n_eps: int,
    steps_per_ep: int,
) -> Dict[str, Any]:
    """Frozen-policy eval for ONE (cap, arm) cell. Rails must already be applied by
    the caller; this sets the EVAL-time cap on the coordinator config (read live by
    tick()). Instruments BOTH the discrete committed-mode occupancy AND the
    continuous pre-argmax operating_mode['external_task'] margin, plus a
    MODE-CONDITIONED dwell in external_task specifically."""
    agent.eval()
    world_dim = agent.config.latent.world_dim
    coord = agent.salience
    # EVAL-time cap override (bit-live at tick(); no retraining). This is the sweep.
    coord.config.affinity_input_cap = float(cap)
    feed_harm = scaffold_cfg.scaffold_feed_harm_stream

    coord_ticks_start = int(coord.diagnostics.get("n_ticks", 0))

    mode_step_counts = {m: 0 for m in MODE_NAMES}
    other_mode_steps = 0
    total_switches = 0
    total_steps = 0
    ext_margins: List[float] = []          # continuous operating_mode[external_task] per step
    # Mode-conditioned dwell: run-lengths measured only while current_mode ==
    # external_task (fixes 464e's M3 mode-agnostic dwell).
    ext_run_lengths: List[int] = []
    all_run_lengths: List[int] = []

    with torch.no_grad():
        for _ep in range(n_eps):
            _, obs_dict = env.reset()
            agent.reset()
            prev_mode = coord.current_mode
            current_run = 1
            ext_run = 1 if prev_mode == STICKY_MODE else 0

            for _ in range(steps_per_ep):
                obs_body = obs_dict["body_state"].to(device)
                obs_world = obs_dict["world_state"].to(device)
                latent = _sense_with_optional_harm(
                    agent, obs_body, obs_world, obs_dict, device, feed_harm
                )

                ticks = agent.clock.advance()
                e1_prior = (
                    agent._e1_tick(latent) if ticks.get("e1_tick")
                    else torch.zeros(1, world_dim, device=device)
                )
                candidates = agent.generate_trajectories(latent, e1_prior, ticks)
                action = agent.select_action(candidates, ticks)
                action_idx = int(action.argmax(dim=-1).item())

                # Continuous pre-argmax margin -- the probe's / H2's recommended
                # instrumentation. operating_mode is the last softmax vector.
                ext_margins.append(float(coord.operating_mode.get(STICKY_MODE, 0.0)))

                cur_mode = coord.current_mode
                if cur_mode in mode_step_counts:
                    mode_step_counts[cur_mode] += 1
                else:
                    other_mode_steps += 1

                if cur_mode != prev_mode:
                    all_run_lengths.append(current_run)
                    if prev_mode == STICKY_MODE and ext_run > 0:
                        ext_run_lengths.append(ext_run)
                    total_switches += 1
                    current_run = 1
                    ext_run = 1 if cur_mode == STICKY_MODE else 0
                    prev_mode = cur_mode
                else:
                    current_run += 1
                    if cur_mode == STICKY_MODE:
                        ext_run += 1

                total_steps += 1
                _, _harm, done, _info, obs_dict = env.step(action_idx)
                if done:
                    all_run_lengths.append(current_run)
                    if cur_mode == STICKY_MODE and ext_run > 0:
                        ext_run_lengths.append(ext_run)
                    current_run = 0
                    ext_run = 0
                    break

            if current_run > 0:
                all_run_lengths.append(current_run)
                if prev_mode == STICKY_MODE and ext_run > 0:
                    ext_run_lengths.append(ext_run)

    frac_task = mode_step_counts[STICKY_MODE] / total_steps if total_steps else 0.0
    mean_dwell = (
        float(sum(all_run_lengths)) / len(all_run_lengths)
        if all_run_lengths else float(steps_per_ep)
    )
    ext_dwell_mean = (
        float(sum(ext_run_lengths)) / len(ext_run_lengths)
        if ext_run_lengths else 0.0
    )
    margins_sorted = sorted(ext_margins)
    margin_mean = float(sum(ext_margins) / len(ext_margins)) if ext_margins else 0.0
    coord_ticks = int(coord.diagnostics.get("n_ticks", 0)) - coord_ticks_start

    return {
        "cap": float(cap),
        "arm": arm_label,
        "fraction_in_external_task": round(frac_task, 4),
        "ext_margin_mean": round(margin_mean, 4),
        "ext_margin_p10": round(_quantile(margins_sorted, 0.10), 4),
        "ext_margin_p50": round(_quantile(margins_sorted, 0.50), 4),
        "ext_margin_p90": round(_quantile(margins_sorted, 0.90), 4),
        "ext_margin_max": round(margins_sorted[-1], 4) if margins_sorted else 0.0,
        "n_switches": total_switches,
        "mean_dwell": round(mean_dwell, 3),
        "ext_dwell_mean": round(ext_dwell_mean, 3),
        "n_ext_runs": len(ext_run_lengths),
        "mode_step_counts": mode_step_counts,
        "other_mode_steps": other_mode_steps,
        "total_steps": total_steps,
        "coord_n_ticks": coord_ticks,
        "n_episodes": n_eps,
    }


def _regime_for_arm(cells_for_arm: List[Dict[str, Any]], arm_label: str) -> Dict[str, Any]:
    """Run the regime-conditioned occupancy gate over the cap cells of ONE arm.
    cells are OccupancyCell(label=cap, fraction=occupancy). Returns the gate dict +
    the mixed-band caps."""
    occ_cells = [
        OccupancyCell(label=f"cap={c['cap']}", fraction=float(c["fraction_in_external_task"]))
        for c in cells_for_arm
    ]
    gate = evaluate_regime_occupancy_gate(
        occ_cells, mode_label=STICKY_MODE,
        floor=OCCUPANCY_FLOOR, ceiling=OCCUPANCY_CEILING,
    )
    mixed_caps = [
        float(c["cap"]) for c in cells_for_arm
        if OCCUPANCY_FLOOR < float(c["fraction_in_external_task"]) < OCCUPANCY_CEILING
    ]
    gate["arm"] = arm_label
    gate["mixed_band_caps"] = mixed_caps
    gate["graded"] = bool(gate["regime_shape"] == "graded")
    return gate


def _aborted_seed_record(seed: int, stage: str, reason: str) -> Dict[str, Any]:
    return {
        "seed": seed, "aborted_at": stage, "abort_reason": reason,
        "guard_pass": False,
        "p2_contact_rate": 0.0, "p2_z_goal_norm_at_contact_peak": 0.0,
        "p2_num_contact_events": 0,
        "cells": [],
        "regime_by_arm": {},
        "max_margin_mean": 0.0,
        "margin_engaged": False,
        "sym_graded": False,
        "any_graded": False,
    }


def _run_seed(seed: int, dry_run: bool, total_eps: int,
              env_seed_base: Optional[int] = None) -> Dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    seed_env_base = None if env_seed_base is None else int(env_seed_base) + int(seed)
    scaffold_cfg = _make_scaffold_cfg(dry_run, env_seed=seed_env_base)
    device = torch.device("cpu")
    steps_per_ep = scaffold_cfg.scaffold_steps_per_episode
    eval_eps = 2 if dry_run else MODE_EVAL_EPISODES
    caps = CAP_SWEEP[:2] if dry_run else CAP_SWEEP

    probe_env = _build_dual_cue_env(
        scaffold_cfg, seed=_derive_env_seed(seed_env_base, stream=2, idx=0)
    )
    probe_env.reset()
    agent = REEAgent(_make_config(probe_env)).to(device)
    scheduler = ScaffoldedSD054OnboardingScheduler(scaffold_cfg)

    print(f"Seed {seed} Condition {CONDITION_LABEL}", flush=True)
    done = 0

    s0 = scheduler.run_stage0_nursery(agent, device)
    done += s0.n_episodes
    print(f"  [train] stage0_nursery seed={seed} ep {done}/{total_eps}"
          f" z_goal_peak={s0.z_goal_norm_peak:.4f} formed={s0.z_goal_formed}", flush=True)
    if s0.aborted:
        print(f"verdict: FAIL seed={seed} aborted_at=stage0 reason={s0.abort_reason}", flush=True)
        return _aborted_seed_record(seed, "stage0", s0.abort_reason)

    s0b = scheduler.run_stage0b_consolidation(agent, device, stage0_baseline_norm=s0.z_goal_norm_peak)
    done += s0b.n_episodes
    print(f"  [train] stage0b_consolidate seed={seed} ep {done}/{total_eps}"
          f" retention={s0b.retention_ratio:.3f}"
          f" gate={'pass' if s0b.retention_gate_passed else 'FAIL'}", flush=True)
    if s0b.aborted:
        print(f"verdict: FAIL seed={seed} aborted_at=stage0b reason={s0b.abort_reason}", flush=True)
        return _aborted_seed_record(seed, "stage0b", s0b.abort_reason)

    p0 = scheduler.run_p0(agent, device)
    done += p0.n_episodes
    print(f"  [train] p0_guided seed={seed} ep {done}/{total_eps}"
          f" mean_len={p0.mean_episode_length:.1f} rv={p0.final_running_variance:.5f}", flush=True)
    if p0.aborted:
        print(f"verdict: FAIL seed={seed} aborted_at=p0 reason={p0.abort_reason}", flush=True)
        return _aborted_seed_record(seed, "p0", p0.abort_reason)

    hz = scheduler.run_hazard_avoidance(agent, device)
    done += hz.n_episodes
    print(f"  [train] hazard_avoidance seed={seed} ep {done}/{total_eps}"
          f" median_last={hz.median_last_window_episode_length:.1f}"
          f" survival_gate={'pass' if hz.survival_gate_passed else 'FAIL'}", flush=True)
    if hz.aborted:
        print(f"verdict: FAIL seed={seed} aborted_at=hazard reason={hz.abort_reason}", flush=True)
        return _aborted_seed_record(seed, "hazard", hz.abort_reason)

    p1 = scheduler.run_p1(agent, device)
    done += p1.n_episodes
    print(f"  [train] p1_foraging seed={seed} ep {done}/{total_eps}"
          f" median_last={p1.median_last_window_episode_length:.1f}"
          f" survival_gate={'pass' if p1.survival_gate_passed else 'FAIL'}", flush=True)

    p2 = scheduler.run_p2(agent, device)
    done += p2.n_episodes
    print(f"  [train] p2_guard seed={seed} ep {done}/{total_eps}"
          f" contact_rate={p2.contact_rate:.4f} contact_events={p2.num_contact_events}"
          f" z_goal_at_contact={p2.z_goal_norm_at_contact_peak:.4f}", flush=True)

    guard_pass = bool(
        p2.contact_rate > CONTACT_GATE
        and p2.z_goal_norm_at_contact_peak > P2_ZGOAL_GATE
    )
    _ZG.observe(agent)  # trained curriculum agent, after all training stages

    dual_env = _build_dual_cue_env(
        scaffold_cfg, seed=_derive_env_seed(seed_env_base, stream=2, idx=1)
    )
    dual_env.reset()

    # Sweep CAP x ARM on clones of the SAME trained agent.
    cells: List[Dict[str, Any]] = []
    for cap in caps:
        for arm_label in ARM_LABELS:
            agent_cell = _clone_for_arm(agent, device)
            _apply_rails(agent_cell.salience, arm_label)
            cell = _eval_cap_cell(
                agent_cell, dual_env, cap, arm_label,
                scaffold_cfg, device, eval_eps, steps_per_ep,
            )
            cells.append(cell)
            done += eval_eps
            _ZG.observe(agent_cell)
            print(f"  [eval] seed={seed} cap={cap} {arm_label}"
                  f" occ={cell['fraction_in_external_task']}"
                  f" margin_mean={cell['ext_margin_mean']}"
                  f" ext_dwell={cell['ext_dwell_mean']}"
                  f" switches={cell['n_switches']}", flush=True)

    # Per-arm regime classification across the cap cells.
    regime_by_arm: Dict[str, Any] = {}
    for arm_label in ARM_LABELS:
        arm_cells = [c for c in cells if c["arm"] == arm_label]
        regime_by_arm[arm_label] = _regime_for_arm(arm_cells, arm_label)

    sym_graded = bool(regime_by_arm["ARM_SYMMETRIC"]["graded"])
    any_graded = any(regime_by_arm[a]["graded"] for a in ARM_LABELS)
    max_margin_mean = max((float(c["ext_margin_mean"]) for c in cells), default=0.0)
    margin_engaged = bool(max_margin_mean > MARGIN_FLOOR)

    print(f"  [regime] seed={seed}"
          f" sym_shape={regime_by_arm['ARM_SYMMETRIC']['regime_shape']}"
          f" asym_shape={regime_by_arm['ARM_ASYM_STICKY_TASK']['regime_shape']}"
          f" max_margin={max_margin_mean:.4f} margin_engaged={margin_engaged}", flush=True)
    print(f"verdict: {'PASS' if (guard_pass and margin_engaged and sym_graded) else 'FAIL'}"
          f" seed={seed} guard_pass={guard_pass} margin_engaged={margin_engaged}"
          f" sym_graded={sym_graded}"
          f" (contact_rate={p2.contact_rate:.4f} z_goal_at_contact={p2.z_goal_norm_at_contact_peak:.4f})",
          flush=True)

    return {
        "seed": seed,
        "aborted_at": None,
        "abort_reason": "",
        "guard_pass": guard_pass,
        "stage0_z_goal_norm_peak": float(s0.z_goal_norm_peak),
        "p1_survival_pass": bool(p1.survival_gate_passed),
        "hazard_stage_survival_pass": bool(hz.survival_gate_passed),
        "p2_contact_rate": float(p2.contact_rate),
        "p2_z_goal_norm_at_contact_peak": float(p2.z_goal_norm_at_contact_peak),
        "p2_num_contact_events": int(p2.num_contact_events),
        "cells": cells,
        "regime_by_arm": regime_by_arm,
        "max_margin_mean": round(max_margin_mean, 4),
        "margin_engaged": margin_engaged,
        "sym_graded": sym_graded,
        "any_graded": any_graded,
    }


def _frac(flags: List[bool]) -> float:
    return float(sum(1 for f in flags if f)) / float(len(flags)) if flags else 0.0


def run_experiment(dry_run: bool = False,
                   env_seed_base: Optional[int] = None) -> Dict[str, Any]:
    print(f"[{EXPERIMENT_TYPE}] starting (dry_run={dry_run}, "
          f"env_seed_base={env_seed_base})", flush=True)
    seeds = SEEDS[:1] if dry_run else SEEDS
    caps = CAP_SWEEP[:2] if dry_run else CAP_SWEEP
    n_cells = len(caps) * len(ARM_LABELS)
    if dry_run:
        total_eps = 2 + 2 + 5 + 5 + 5 + 2 + n_cells * 2
    else:
        total_eps = (
            STAGE0_BUDGET + STAGE0B_BUDGET + P0_BUDGET + HAZARD_STAGE_BUDGET
            + P1_BUDGET + P2_BUDGET + n_cells * MODE_EVAL_EPISODES
        )

    per_seed: List[Dict[str, Any]] = []
    for s in seeds:
        per_seed.append(_run_seed(s, dry_run, total_eps, env_seed_base=env_seed_base))

    n = len(per_seed)
    guard_flags = [r["guard_pass"] for r in per_seed]
    guard_frac = _frac(guard_flags)
    guard_passing = [r for r in per_seed if r["guard_pass"]]
    contact_non_vacuity_met = bool(guard_frac >= MIN_FRACTION)

    # G-margin readiness: the external_task drive must ENGAGE at SOME cap on
    # >= 2/3 guard-passing seeds. Below floor everywhere => drive not producing
    # the signal => substrate_not_ready (NEVER read as H2 structural).
    margin_flags = [bool(r.get("margin_engaged", False)) for r in guard_passing]
    margin_frac = _frac(margin_flags)
    margin_ready_met = bool(margin_frac >= MIN_FRACTION)

    # H1 verdict: SYMMETRIC arm graded (some cap in the mixed band) on >= 2/3
    # guard-passing seeds. asym reported but not load-bearing.
    sym_graded_flags = [bool(r.get("sym_graded", False)) for r in guard_passing]
    sym_graded_frac = _frac(sym_graded_flags)
    h1_supported = bool(sym_graded_frac >= MIN_FRACTION)

    any_graded_flags = [bool(r.get("any_graded", False)) for r in guard_passing]
    any_graded_frac = _frac(any_graded_flags)

    # Non-degeneracy / "did the cap manipulation LAND at all" (the recurring
    # dose-sweep-flatness defect the skill warns about). Two axes:
    #   occupancy_varies -- the discrete DV varies across caps for some seed/arm.
    #   margin_varies    -- the CONTINUOUS operating_mode['external_task'] varies
    #                       across caps for some seed/arm.
    # A read where occupancy is dead-flat (e.g. pinned at 1.0) but the margin DOES
    # respond to the cap is NOT an instrument clamp -- it is genuine occupancy
    # saturation of a soft signal that the cap demonstrably still moves (the
    # H2-favourable structural finding). Only when NEITHER varies is the cap
    # override suspect (an instrument concern -- the probe confirmed the override
    # lands, so this should not happen). manipulation_landed gates non-degeneracy.
    def _varies(key: str) -> bool:
        for r in guard_passing:
            for arm_label in ARM_LABELS:
                vals = [float(c[key]) for c in r.get("cells", []) if c["arm"] == arm_label]
                if len(vals) >= 2 and (max(vals) - min(vals)) > 1e-6:
                    return True
        return False

    occupancy_varies = _varies("fraction_in_external_task")
    margin_varies = _varies("ext_margin_mean")
    manipulation_landed = bool(occupancy_varies or margin_varies)

    if not contact_non_vacuity_met:
        outcome = "FAIL"
        readiness_route = "substrate_not_ready_requeue"
        route_reason = "contact_guard_unmet"
    elif not margin_ready_met:
        outcome = "FAIL"
        readiness_route = "substrate_not_ready_requeue"
        route_reason = "external_task_drive_not_engaging"
    elif h1_supported:
        outcome = "PASS"
        readiness_route = "cap_recalibration_admits_mixed_regime"
        route_reason = "graded_regime_reachable_on_symmetric_arm"
    elif not manipulation_landed:
        # Occupancy AND margin both dead-flat across the swept caps: the cap
        # override is not visibly affecting arbitration at all. The pre-authoring
        # probe confirmed the override lands, so this is an instrument concern to
        # verify, NOT a confident structural (H2) conclusion.
        outcome = "FAIL"
        readiness_route = "substrate_not_ready_requeue"
        route_reason = "cap_manipulation_inert_verify_instrument"
    else:
        # Occupancy saturated but the cap demonstrably moves the soft margin: the
        # committed argmax commits hard despite a graded soft signal -> structural
        # discreteness (H2-favourable), a real finding, not not-ready.
        outcome = "FAIL"
        readiness_route = "saturated_all_caps_supports_structural_discreteness"
        route_reason = "occupancy_saturated_margin_responds_to_cap"

    # Diagnostic: excluded from confidence scoring. Directions are context only.
    if h1_supported:
        sd032a_dir = "supports"      # the mode register CAN grade; recalibration viable
    elif margin_ready_met and contact_non_vacuity_met and manipulation_landed:
        sd032a_dir = "weakens"       # engages + cap lands, but never grades -> structural (H2)
    else:
        sd032a_dir = "non_contributory"  # not ready / cap inert; says nothing about the register
    direction_map = {
        "MECH-266": "non_contributory",  # establishes the measurement precondition, not the over-binding test
        "SD-032a": sd032a_dir,
    }
    overall_direction = "non_contributory"

    # Winning cap band: caps that produced mixed occupancy on the symmetric arm of
    # any guard-passing seed.
    winning_caps = sorted({
        c
        for r in guard_passing
        for c in r.get("regime_by_arm", {}).get("ARM_SYMMETRIC", {}).get("mixed_band_caps", [])
    })

    print(f"[{EXPERIMENT_TYPE}] contact_ready={contact_non_vacuity_met}"
          f" (guard {sum(guard_flags)}/{n}) margin_ready={margin_ready_met}"
          f" (frac={margin_frac:.3f}) sym_graded_frac={sym_graded_frac:.3f}"
          f" h1_supported={h1_supported} -> outcome={outcome} route={readiness_route}", flush=True)
    print(f"[{EXPERIMENT_TYPE}] winning_cap_band(symmetric)={winning_caps}", flush=True)
    for cid in CLAIM_IDS:
        print(f"[{EXPERIMENT_TYPE}] per_claim {cid}={direction_map[cid]}", flush=True)

    acceptance = {
        "contact_non_vacuity_met": contact_non_vacuity_met,
        "guard_fraction": guard_frac,
        "n_guard_passing_seeds": len(guard_passing),
        "margin_ready_met": margin_ready_met,
        "margin_ready_fraction": margin_frac,
        "sym_graded_fraction": sym_graded_frac,
        "any_graded_fraction": any_graded_frac,
        "h1_supported": h1_supported,
        "winning_cap_band_symmetric": winning_caps,
        "occupancy_varies_across_caps": occupancy_varies,
        "margin_varies_across_caps": margin_varies,
        "manipulation_landed": manipulation_landed,
        "route_reason": route_reason,
        "per_seed_guard_pass": guard_flags,
        "per_seed_sym_graded": [bool(r.get("sym_graded", False)) for r in per_seed],
    }

    # readiness preconditions: contact guard + drive engagement (the same statistic
    # the H1 verdict routes on -- occupancy -- requires the drive to engage first).
    preconditions = [
        {
            "name": "foraging_contact_guard",
            "kind": "readiness",
            "description": "603n G2+G3 contact guard on >= 2/3 seeds. A curriculum "
                           "that never became foraging-competent makes every occupancy "
                           "reading meaningless.",
            "control": "fraction of seeds with P2 contact_rate > 0 AND "
                       "z_goal_norm_at_contact_peak > 0.4.",
            "measured": round(guard_frac, 4),
            "threshold": MIN_FRACTION,
            "direction": "lower",
            "met": contact_non_vacuity_met,
        },
        {
            "name": "external_task_drive_engages",
            "kind": "readiness",
            "description": "the external_task drive must ENGAGE at SOME swept cap -- "
                           "per-seed max-over-cells of the CONTINUOUS "
                           "operating_mode['external_task'] margin > MARGIN_FLOOR -- on "
                           ">= 2/3 guard-passing seeds. This is the readiness form of "
                           "the SAME arbitration signal the occupancy DV routes on: if "
                           "the continuous margin is ~0 at every cap, the drive is not "
                           "producing the signal (substrate/wiring not ready, e.g. a "
                           "goal_state drop or dead drive), which must self-route to "
                           "substrate_not_ready_requeue and NOT be read as structural "
                           "discreteness (H2).",
            "control": "fraction of guard-passing seeds whose best cell's continuous "
                       "ext_margin_mean clears MARGIN_FLOOR.",
            "measured": round(margin_frac, 4),
            "threshold": MIN_FRACTION,
            "direction": "lower",
            "met": margin_ready_met,
        },
    ]

    criteria = [
        {"name": "H1_symmetric_arm_graded_regime_reachable", "load_bearing": True,
         "passed": h1_supported},
    ]
    # non-degeneracy: the H1 criterion is a meaningful test only if readiness held
    # AND the cap manipulation LANDED somewhere (occupancy OR the continuous margin
    # varied across caps). If neither moved, the cap override was inert and the H1
    # fail is an instrument artefact, not a genuine null.
    crit_non_degenerate = bool(contact_non_vacuity_met and margin_ready_met and manipulation_landed)

    return {
        "outcome": outcome,
        "evidence_direction": overall_direction,
        "evidence_direction_per_claim": direction_map,
        "acceptance": acceptance,
        "interpretation": {
            "label": readiness_route,
            "readiness_route": readiness_route,
            "route_reason": route_reason,
            "hypothesis": "GOV-FANOUT-1 leg H1: some affinity_input_cap admits a "
                          "mixed external_task occupancy regime (cap mis-calibration, "
                          "not structural bang-bang).",
            "null": "No cap value yields per-arm occupancy in (0.1, 0.9) on >= 2/3 seeds.",
            "winning_cap_band_symmetric": winning_caps,
            "preconditions": preconditions,
            "criteria": criteria,
            "criteria_non_degenerate": {
                "H1_symmetric_arm_graded_regime_reachable": crit_non_degenerate,
            },
            "regime_gate": {
                "definition": "per (seed, arm), OccupancyCell over the CAP_SWEEP; "
                              "evaluate_regime_occupancy_gate(floor=0.1, ceiling=0.9). "
                              "regime_shape in {unreachable, saturated_bimodal, graded}. "
                              "graded == some cap yields occupancy strictly in (0.1, 0.9). "
                              "Replaces the min()-across-the-sweep gate (the cluster "
                              "autopsy's M1/M2 defect; H3 fix) with the shared primitive "
                              "experiments/_lib/regime_occupancy_gate.py -- never a "
                              "min-across-the-sweep statistic.",
                "occupancy_floor": OCCUPANCY_FLOOR,
                "occupancy_ceiling": OCCUPANCY_CEILING,
                "margin_floor": MARGIN_FLOOR,
                "cap_sweep": CAP_SWEEP,
                "primary_arm": "ARM_SYMMETRIC",
                "min_fraction": MIN_FRACTION,
            },
            "contact_guard": {
                "definition": "per-seed P2 contact_rate > 0 AND z_goal_norm_at_contact_peak "
                              "> 0.4; < 2/3 seeds -> substrate_not_ready_requeue.",
                "min_fraction": MIN_FRACTION,
                "p2_zgoal_gate": P2_ZGOAL_GATE,
                "contact_gate": CONTACT_GATE,
            },
        },
        "per_seed": per_seed,
    }


def main(dry_run: bool = False,
         env_seed_base: Optional[int] = None) -> Dict[str, Any]:
    t0 = time.perf_counter()
    result = run_experiment(dry_run=dry_run, env_seed_base=env_seed_base)
    if dry_run:
        print(f"[{EXPERIMENT_TYPE}] dry-run complete; manifest not written.", flush=True)
        return {"outcome": result["outcome"], "manifest_path": None}

    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{timestamp}_v3"
    out_dir = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments" / EXPERIMENT_TYPE
    out_dir.mkdir(parents=True, exist_ok=True)

    # Full config for generous recording (stamp_recording_core folds it in).
    full_config = {
        "cap_sweep": CAP_SWEEP,
        "affinity_input_cap_train": AFFINITY_INPUT_CAP_TRAIN,
        "occupancy_floor": OCCUPANCY_FLOOR,
        "occupancy_ceiling": OCCUPANCY_CEILING,
        "margin_floor": MARGIN_FLOOR,
        "arms": ARM_LABELS,
        "sticky_exit": STICKY_EXIT,
        "loose_exit": LOOSE_EXIT,
        "mode_eval_episodes_per_cell": MODE_EVAL_EPISODES,
        "train_steps": TRAIN_STEPS,
        "seeds": SEEDS,
        "min_fraction": MIN_FRACTION,
        "p2_zgoal_gate": P2_ZGOAL_GATE,
        "contact_gate": CONTACT_GATE,
        "scaffold_curriculum": {
            "stage0_budget": STAGE0_BUDGET, "stage0b_budget": STAGE0B_BUDGET,
            "p0_budget": P0_BUDGET, "hazard_stage_budget": HAZARD_STAGE_BUDGET,
            "p1_budget": P1_BUDGET, "p2_budget": P2_BUDGET,
            "n_resource_types": N_RESOURCE_TYPES,
            "config_basis": "V3-EXQ-603n",
        },
    }

    manifest = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": QUEUE_ID,
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "env_seed_base": env_seed_base,
        "timestamp_utc": timestamp,
        "outcome": result["outcome"],
        "evidence_direction": result["evidence_direction"],
        "evidence_direction_per_claim": result["evidence_direction_per_claim"],
        "sleep_driver_pattern": "N/A (waking goal-pipeline onboarding scheduler; no sleep loop)",
        "substrate": "scaffolded_sd054_onboarding (full curriculum; 603n config) + "
                     "SalienceCoordinator (SD-032a) + mode-governance-engagement "
                     "external_task drive (use_external_task_drive=True) + GAP-3 "
                     "dual_cue competing-goal env + goal_state clone fix + "
                     "salience_affinity_input_cap (trained at 2.0, EVAL cap swept in "
                     "[0.75,1.0,1.25,1.5,1.75] on clones). use_closure_operator OFF.",
        "condition": CONDITION_LABEL,
        "predecessor": PREDECESSOR,
        "gov_fanout": "GOV-FANOUT-1 leg H1 of the pre-registered frozen-ledger "
                      "question mech266_mode_arbitration_saturation "
                      "(failure_autopsy_mech266-464e-467e-cluster_2026-08-13).",
        "method_note": "Sweeps affinity_input_cap at EVAL time on clones of a single "
                       "trained curriculum agent per seed (the cap is read live at "
                       "SalienceCoordinator.tick(), so no retraining is needed -- the "
                       "same train-once/sweep-on-clones pattern 467e uses for its ratio "
                       "sweep). Training cap held at 2.0 (464e construction) for "
                       "comparability with the banked 464e/467e reference. Instruments "
                       "both the discrete committed-mode occupancy AND the continuous "
                       "pre-argmax operating_mode['external_task'] margin (probe / H2 "
                       "recommendation), plus mode-conditioned dwell in external_task "
                       "(fixes 464e's mode-agnostic dwell). Non-vacuity via "
                       "experiments/_lib/regime_occupancy_gate.py (per-arm, "
                       "any-reachable, regime-shape-conditioned) -- NEVER "
                       "min-across-the-sweep, the M1/M2 defect the cluster autopsy "
                       "identified.",
        "pre_registered_thresholds": {
            "cap_sweep": CAP_SWEEP,
            "affinity_input_cap_train": AFFINITY_INPUT_CAP_TRAIN,
            "occupancy_floor": OCCUPANCY_FLOOR,
            "occupancy_ceiling": OCCUPANCY_CEILING,
            "margin_floor": MARGIN_FLOOR,
            "min_fraction": MIN_FRACTION,
            "p2_zgoal_gate": P2_ZGOAL_GATE,
            "contact_gate": CONTACT_GATE,
            "sticky_exit": STICKY_EXIT,
            "loose_exit": LOOSE_EXIT,
        },
        "config": full_config,
        "stage_plan": stage_plan(),
    }
    manifest.update(result)
    out_path = write_flat_manifest(
        manifest,
        out_dir,
        dry_run=False,
        config=full_config,
        seeds=SEEDS,
        script_path=Path(__file__),
        z_goal_stream_stats=_ZG.stats(),
        started_at=t0,
    )
    print(f"[{EXPERIMENT_TYPE}] manifest -> {out_path}", flush=True)
    print(f"Done. Outcome: {result['outcome']}", flush=True)
    return {"outcome": result["outcome"], "manifest_path": str(out_path)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--env-seed", type=int, default=None,
        help="Opt-in env-seed base. Omitted (the default) reproduces the landed "
             "run's OS-entropy env seeding exactly. Set it and every env this run "
             "builds is deterministically seeded. A pinned run is NOT comparable to "
             "a landed one.",
    )
    args = ap.parse_args()
    _res = main(dry_run=args.dry_run, env_seed_base=args.env_seed)
    if _res.get("manifest_path"):
        _outcome_raw = str(_res["outcome"]).upper()
        emit_outcome(
            outcome=_outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL",
            manifest_path=_res["manifest_path"],
            dry_run=bool(args.dry_run),
        )
