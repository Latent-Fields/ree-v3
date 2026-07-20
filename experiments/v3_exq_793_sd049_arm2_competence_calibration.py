"""
V3-EXQ-793 -- SD-049 Phase-2 ARM_2 FORAGING-COMPETENCE CALIBRATION (2x2 attribution probe).

DIAGNOSTIC, not a measurement. This probe does NOT score the SD-049 wanting!=liking
dissociation. Its only job is to discriminate WHY the V3-EXQ-693a PRIMARY arm (ARM_2,
3type+novelty) could not clear its two contribution gates, and to hand the 693b measurement
re-issue a curriculum+env config that does clear them.

WHY THIS EXISTS (failure_autopsy_V3-EXQ-693a_2026-06-21.json, status=confirmed,
routing=implement-substrate):
V3-EXQ-693a's WL-scoring harness port WORKED (n_scored_wl_steps 15/20 on the contributing
arms ARM_1/ARM_3; identity probe 0.78/0.52; per-axis drive ANOVA F up to 1761). The blocker
moved one link downstream, to PRIMARY-arm foraging competence: ARM_2 produced 0 contributing
runs because

    behav_contact_rate 0.0099-0.0188  <  CONSUMPTION_FLOOR 0.02        (all 3 seeds)
    hazard_stage_survival_pass = False                                 (2/3 seeds)

so the non-vacuity gate (R1/R2/R3, all keyed on ARM_2) self-routed
substrate_not_ready_requeue and the discrimination criteria (C_ID / C_GR / C_WL / C_ANOVA)
were gated out before they could score. That is the goal_pipeline:GAP-2 nav/foraging-
competence ceiling, NOT a WL-harness recurrence and NOT a falsification.

THE TWO LIVE HYPOTHESES (this is a DISCRIMINATION, so GOV-FANOUT-1 applies -- a portfolio,
not one more sequential re-posed probe):

  H_DENSITY   The shortfall is RESOURCE STARVATION, structural and env-side.
              causal_grid_world.py splits a FIXED total resource budget across
              n_resource_types (`n_to_spawn = min(self.num_resources, len(forage_pool))`,
              then distributed per type), so the 3-type ARM_2 substrate has ~3x lower
              PER-TYPE density than the 1-type ARM_0 control BY CONSTRUCTION. Under
              H_DENSITY the contact shortfall is arithmetic, and the hazard-stage survival
              failure is DOWNSTREAM of it (a starved agent forages longer/further under
              hazard pressure and dies).
  H_CURRICULUM The shortfall is a TRAINING-BUDGET ceiling, independent of density: the
              3-type substrate is a harder foraging + survival problem than the curriculum's
              P0 (100 ep) / Stage-H (40 ep) budgets buy, so competence is simply undertrained
              at measurement time.

These are NOT mutually exclusive, and the interaction is the scientifically interesting cell:
if hazard-survival is rescued by density alone, survival failure was downstream of starvation;
if it needs the curriculum amend even at restored density, it is an independent second ceiling.
A single-axis probe CANNOT tell these apart -- hence the 2x2.

THE 2x2 (all four arms on the ARM_2 substrate: sd049_on=True, n_resource_types=3):

  arm     curriculum        density-preserving spawn   role
  A00     693a-as-is        OFF                        baseline; must REPRODUCE the 693a failure
  A10     AMENDED           OFF                        H_CURRICULUM main effect
  A01     693a-as-is        ON                         H_DENSITY main effect
  A11     AMENDED           ON                         joint / the config handed to 693b

3 seeds (42/43/44) x 4 arms = 12 cells. A00 is a faithful re-run of 693a's ARM_2 cell, which
is what makes the three contrasts interpretable as deltas against a reproduced failure.

SEPARABILITY (why the axes are kept orthogonal): the curriculum levers live entirely in this
script's *_AMENDED constants and in ScaffoldedSD054OnboardingConfig episode budgets; the
density lever is the single env-side kwarg sd049_preserve_per_type_density, plumbed through
scaffolded_sd054_onboarding._sd049_kwargs and touched nowhere else. Neither axis moves the
other's knobs, so a later retest can attribute a rescue to one, the other, or their
interaction. This ALSO means the density axis enriches the Stage-H (hazard) env as well as
P0/P1/P2 -- which is deliberate and is exactly the coupling the attribution question is about.

THE CURRICULUM AMEND (levers, and what each targets):
  P0_BUDGET               100 -> 150   foraging warm-up; targets behav_contact_rate
  HAZARD_STAGE_BUDGET      40 ->  80   isolated survival practice; targets hazard-stage survival
  HAZARD_STAGE_NUM_RESOURCES 2 ->  3   so Stage-H is not ITSELF resource-starved under a
                                       3-type split (the density confound reaching into the
                                       stage whose survival gate is failing)

DELIBERATELY NOT MOVED -- these are the ACCEPTANCE CRITERIA, and relaxing them would convert
a detected shortfall into a citable result rather than measuring anything:
  CONSUMPTION_FLOOR              0.02   (the 693a target, verbatim)
  HAZARD_STAGE_SURVIVAL_GATE_STEPS 75   (the Stage-H survival gate)
  P2_ZGOAL_GATE                   0.4   (the 603n contact guard)
The amend is training BUDGET and stage COMPOSITION only. No threshold in this script is
derived from the run's own statistics.

PRE-REGISTERED DVs (both are the 693a gate quantities verbatim, so a cell that clears here
clears 693b's contribution gate by construction):
  D1 contact   behav_contact_rate > CONSUMPTION_FLOOR (0.02), counted with 693a's own rule
               (benefit > SEED_BENEFIT_THRESHOLD OR a genuine consumed-type info tag), on
               >= MIN_FRACTION (2/3) seeds.
  D2 survival  hazard_stage_survival_pass on >= MIN_FRACTION (2/3) seeds.
  D3 clears    D1 AND D2 -- the autopsy's failure_record_entry `target` verbatim.

PRE-REGISTERED CRITERIA (D3 on each arm; C_JOINT is the load-bearing one):
  C_BASE_FAILS   A00 does NOT clear          (positive control: the ceiling reproduced)
  C_CURR         A10 clears                  (H_CURRICULUM sufficient alone)
  C_DENS         A01 clears                  (H_DENSITY sufficient alone)
  C_JOINT        A11 clears                  [LOAD-BEARING] -- the config 693b can run on
Attribution readout (reported, not gated): the 2x2 table of mean behav_contact_rate and
hazard-survival fraction, plus the two main effects and the interaction, computed on the
per-cell values.

NON-VACUITY / READINESS PRECONDITIONS (an unmet precondition self-routes
substrate_not_ready_requeue -- NEVER a verdict label, and NEVER a false weakens; this retains
693a's R1/R2/R3 discipline in the form this probe can carry):
  P_DENSITY_FLAG   CausalGridWorldV2.__init__ accepts sd049_preserve_per_type_density.
                   The env-side change is owned by a SEPARATE session; if it has not landed,
                   the A01/A11 arms are unconstructible and the 2x2 is not a 2x2. Measured by
                   introspection in setup, BEFORE any compute is spent.
  P_BASE_REPRO     A00 mean behav_contact_rate <= CONSUMPTION_FLOOR (direction: upper). If the
                   baseline arm unexpectedly CLEARS, the 693a ceiling did not reproduce and
                   all three contrasts are deltas against a baseline that is not the failure
                   under investigation -- uninterpretable, so re-queue rather than report.

DV-SYMMETRY DECLARATION (per-arm, mandatory -- is each manipulation invariant under a symmetry
of its own DV?):
  D1 behav_contact_rate is a count ratio over env interactions: a SET-AGGREGATE, whose symmetry
  group is permutation of interchangeable units (seeds, episodes, steps). Neither manipulation
  is a relabeling of those units. The density lever changes the NUMBER of contactable typed
  cells spawned per episode (it moves `desired` in the spawn budget, so the aggregate's inputs
  change, not their order); the curriculum lever changes the trained policy that generates the
  contacts. Not invariant under permutation -> D1 can move for both.
  D2 hazard_stage_survival_pass thresholds a MEDIAN episode length (>= 75 steps over the last
  10 Stage-H episodes): rank/order-based, so its symmetry group is monotone rescaling. Neither
  manipulation is a monotone reparameterisation of episode length -- both change the actual
  survival duration distribution (Stage-H budget changes the policy; density changes Stage-H
  resource availability, which is why HAZARD_STAGE_NUM_RESOURCES is on the curriculum axis at
  all). Not invariant -> D2 can move for both.
  No arm here is a broadcast-scalar manipulation read through an argmax/softmax DV (the
  V3-EXQ-604c class): both levers act on the environment and the training schedule, upstream of
  action selection, not as a uniform per-candidate offset.

WHAT A NULL MEANS (declared per leg, so a null is informative rather than wasted):
  A10 null + A01 clears -> H_CURRICULUM refuted at this budget; starvation is the cause.
  A01 null + A10 clears -> H_DENSITY refuted as SUFFICIENT; the ceiling is training budget.
  Both null, A11 clears -> genuine interaction; neither lever alone is enough.
  All four null      -> the ceiling is neither of the two named causes at these magnitudes;
                        routes to /failure-autopsy for a new hypothesis, NOT to a bigger budget.

claim_ids: SD-049 (diagnostic; excluded from confidence/conflict scoring by purpose)
experiment_purpose: diagnostic
predecessor: V3-EXQ-693a (FAIL / non_contributory; ARM_2 contribution collapse)
does NOT supersede 693a -- it is the substrate hand-off the 693a autopsy routed, and it leaves
the 693b letter free for the SD-049 Phase-2 measurement re-issue.
SLEEP DRIVER: N/A (waking goal-pipeline onboarding scheduler; no sleep loop).
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
from ree_core.utils.config import REEConfig  # noqa: E402
from scaffolded_sd054_onboarding import (  # noqa: E402
    ScaffoldedSD054OnboardingConfig,
    ScaffoldedSD054OnboardingScheduler,
    _build_env,
    _sense_with_optional_harm,
)
from experiments._lib.arm_fingerprint import arm_cell  # noqa: E402
from experiments._lib.manifest_core import stamp_recording_core  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402

# Both readiness preconditions here ARE their own definitions and are reachable by
# construction, so the anchor-reachability guard has nothing to add:
#   density_preserving_spawn_kwarg_available -- a constructor-signature introspection,
#     measured 1.0/0.0 against threshold 1.0. There is no hand-written predicate that could be
#     narrower than the state it anchors to: the kwarg is present or it is not.
#   baseline_arm_reproduces_693a_ceiling -- an UPPER-bound gate at CONSUMPTION_FLOOR (0.02)
#     whose known-positive control is V3-EXQ-693a's own recorded ARM_2 contact rate, measured
#     at 0.0099 / 0.0131 / 0.0188 across seeds 42/43/44. Every one of those recorded values
#     sits below the gate, so the gate is demonstrably reachable by the control rather than
#     unmeetable-by-construction (the V3-EXQ-778d defect this guard exists to catch).
ANCHOR_REACHABILITY_EXEMPT = (
    "Both readiness preconditions are their own degeneracy definitions and are reachable by"
    " construction: one is a boolean constructor-signature introspection; the other is an"
    " upper-bound gate at CONSUMPTION_FLOOR whose positive control (693a's recorded ARM_2"
    " contact rates 0.0099-0.0188) lies below the gate on every recorded seed."
)

EXPERIMENT_TYPE = "v3_exq_793_sd049_arm2_competence_calibration"
QUEUE_ID = "V3-EXQ-793"
CLAIM_IDS: List[str] = ["SD-049"]
EXPERIMENT_PURPOSE = "diagnostic"

SEEDS = [42, 43, 44]

# --- The 2x2. (arm_id, label, curriculum_amended, density_on) ---
ARM_SPECS: List[Tuple[str, str, bool, bool]] = [
    ("A00", "base_curriculum_density_off", False, False),   # positive control: reproduce 693a
    ("A10", "amended_curriculum_density_off", True, False),
    ("A01", "base_curriculum_density_on", False, True),
    ("A11", "amended_curriculum_density_on", True, True),
]
BASELINE_ARM = "A00"
JOINT_ARM = "A11"

# The ARM_2 substrate under investigation (693a's PRIMARY arm), fixed across all four arms.
ARM2_SD049_ON = True
ARM2_N_RESOURCE_TYPES = 3

# --- Goal-pipeline / encoder dims (mirror 693a / 603n / 514t exactly) ---
WORLD_DIM = 32
HARM_A_DIM = 16
HARM_OBS_A_DIM = 7
HARM_HISTORY_LEN = 10
DRIVE_WEIGHT = 2.0

# --- Curriculum budgets: BASE = 693a verbatim ---
STAGE0_BUDGET = 20
STAGE0B_BUDGET = 10
P0_BUDGET = 100
HAZARD_STAGE_BUDGET = 40
P1_BUDGET = 50
P2_BUDGET = 15
BEHAV_EVAL_EPISODES = 15
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
HAZARD_STAGE_SURVIVAL_GATE_STEPS = 75      # ACCEPTANCE CRITERION -- not a lever
HAZARD_STAGE_STABILITY_WINDOW = 10

# --- Curriculum budgets: AMENDED (the three levers; everything else identical to BASE) ---
P0_BUDGET_AMENDED = 150                     # lever 1: foraging warm-up
HAZARD_STAGE_BUDGET_AMENDED = 80            # lever 2: isolated survival practice
HAZARD_STAGE_NUM_RESOURCES_AMENDED = 3      # lever 3: Stage-H not itself starved under a 3-way split

# --- 634c seeding calibration + SD-057 cue-recall bridge (mirror 693a / 603n / 514t) ---
SEED_GAIN = 1.5
SEED_BENEFIT_THRESHOLD = 0.02
SEED_DRIVE_FLOOR = 0.9
CUE_RECALL_GAIN = 0.2

# --- SD-058 / MECH-357 protective-scaffold anneal (mirror 693a) ---
AVOIDANCE_SCAFFOLD_FLOOR_START = 0.8
AVOIDANCE_SCAFFOLD_FLOOR_END = 0.0
AVOIDANCE_THREAT_REF = 0.35
PAG_THETA_FREEZE = 0.8
PAG_DURATION_INPUT_THRESHOLD = 0.2
HARM_PATHWAY_LR = 1e-3
STAGE0B_RETENTION_GATE = 0.75

# --- Pre-registered acceptance thresholds (NOT derived from the run; 693a verbatim) ---
P2_ZGOAL_GATE = 0.4          # 603n G3 per-seed contact guard
CONTACT_GATE = 0.0           # 603n G2 per-seed contact guard
CONSUMPTION_FLOOR = 0.02     # D1: the 693a behav-eval contact-rate floor
MIN_FRACTION = 2.0 / 3.0     # >= 2/3 seeds for any per-arm aggregate


def _consumed_type_tag_from_info(info: Dict[str, Any]) -> Optional[int]:
    """681-C4 / 514t read: the AUTHORITATIVE consumed (liking) tag lives in the INFO dict,
    cached by env.step() BEFORE the cell tag is cleared. Copied verbatim from 693a so this
    probe's contact counter is bit-comparable to the DV whose floor it is testing."""
    if not isinstance(info, dict):
        return None
    raw = info.get("sd049_consumed_type_tag_this_tick", 0)
    try:
        tag = int(raw[0] if hasattr(raw, "__len__") else raw)
    except (TypeError, ValueError):
        return None
    return tag if tag > 0 else None


def _ben_drive(obs_body: torch.Tensor) -> Tuple[float, float]:
    b = obs_body.reshape(-1)
    benefit = float(b[11].item()) if b.shape[0] > 11 else 0.0
    energy = float(b[3].item()) if b.shape[0] > 3 else 0.5
    drive = max(0.0, min(1.0, 1.0 - energy))
    return benefit, drive


def _per_axis_drive_from_obs(obs_dict: Dict[str, Any], device: torch.device) -> Optional[torch.Tensor]:
    raw = obs_dict.get("per_axis_drive", None)
    if raw is None:
        return None
    if isinstance(raw, torch.Tensor):
        return raw.to(device).float()
    try:
        return torch.tensor(np.asarray(raw, dtype=np.float32), device=device)
    except (TypeError, ValueError):
        return None


def density_flag_supported() -> bool:
    """P_DENSITY_FLAG: does the CURRENT substrate's env accept the density-preserving kwarg?

    The env-side change (sd049_preserve_per_type_density in causal_grid_world.py) is owned by
    a separate implement-substrate session. Introspecting the constructor signature lets this
    probe self-route substrate_not_ready_requeue in SETUP -- before any compute is spent --
    instead of crashing 100+ episodes into the first density-ON cell.

    Introspect CausalGridWorld, NOT CausalGridWorldV2: despite the name, CausalGridWorldV2 is
    a thin `def CausalGridWorldV2(**kwargs) -> CausalGridWorld` FACTORY FUNCTION, not a class
    (causal_grid_world.py:4675). Its signature is a bare **kwargs, so asking it about a named
    parameter answers False no matter what the substrate supports -- and, being a function,
    `.__init__` resolves to the function object's own method-wrapper rather than to any
    constructor. The kwarg the factory forwards is declared on CausalGridWorld.__init__, which
    is the real ingress and the thing to check."""
    try:
        from ree_core.environment.causal_grid_world import CausalGridWorld
        return "sd049_preserve_per_type_density" in inspect.signature(
            CausalGridWorld.__init__
        ).parameters
    except Exception:
        return False


def _make_scaffold_cfg(dry_run: bool, curriculum_amended: bool,
                       density_on: bool) -> ScaffoldedSD054OnboardingConfig:
    """693a's _make_scaffold_cfg, parametrised on the 2x2 axes. Every knob not named by one of
    the three curriculum levers (or the density flag) is identical to 693a's ARM_2 cell."""
    p0_budget = P0_BUDGET_AMENDED if curriculum_amended else P0_BUDGET
    hazard_budget = HAZARD_STAGE_BUDGET_AMENDED if curriculum_amended else HAZARD_STAGE_BUDGET
    hazard_resources = (
        HAZARD_STAGE_NUM_RESOURCES_AMENDED if curriculum_amended else HAZARD_STAGE_NUM_RESOURCES
    )
    if dry_run:
        stage0, stage0b, p0, hazard, p1, p2, steps = 2, 2, 5, 5, 5, 2, 30
    else:
        stage0, stage0b, p0, hazard, p1, p2, steps = (
            STAGE0_BUDGET, STAGE0B_BUDGET, p0_budget, hazard_budget,
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
        # SD-057 cue-recall bridge -- ARM_2 substrate, fixed across the 2x2
        scaffold_cue_recall_bridge_enabled=bool(ARM2_SD049_ON),
        scaffold_cue_n_resource_types=int(ARM2_N_RESOURCE_TYPES),
        scaffold_stage0_bind_incentive_token=True,
        # THE DENSITY AXIS -- the single env-side lever, default OFF
        scaffold_sd049_preserve_per_type_density=bool(density_on),
        # isolated Stage-H
        scaffold_hazard_stage_enabled=True,
        scaffold_hazard_stage_episode_budget=hazard,
        scaffold_hazard_stage_num_hazards=HAZARD_STAGE_NUM_HAZARDS,
        scaffold_hazard_stage_num_resources=hazard_resources,
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
    )
    if steps < 75:
        cfg.scaffold_p1_survival_gate_steps = max(1, steps // 4)
        cfg.scaffold_hazard_stage_survival_gate_steps = max(1, steps // 4)
    return cfg


def _make_config(env) -> REEConfig:
    """Mirror 693a / 514t: full SD-049 Phase-2 + SD-057 substrate."""
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
    )
    cfg.latent.use_resource_encoder = True
    return cfg


def _run_contact_eval(agent, scaffold_cfg, device: torch.device, n_eps: int) -> Dict[str, Any]:
    """Frozen-policy behavioural eval on this arm's P2 env, measuring D1.

    The contact counter, the z_goal refresh at genuine contact, and the per-axis-drive
    assignment are copied from 693a's _run_arm_eval so behav_contact_rate here is the SAME
    statistic whose floor 693a's R1 gate tested. The WL / identity-probe / ANOVA collection is
    deliberately dropped -- this probe measures competence, not the dissociation."""
    env = _build_env(scaffold_cfg, "p2")
    env.reset()

    contact_steps = 0
    total_steps = 0
    consumption_events = 0
    episode_lengths: List[int] = []
    # SD-049-PHASE-2 density-preserving spawn emits sd049_density_budget_truncated when the
    # forage pool was too small to honour the scaled budget -- i.e. per-type density was NOT
    # actually held constant and the density-ON arms are re-confounded by the very effect the
    # flag exists to remove. A truncated density arm is a manipulation that did not fire, so
    # this is counted and gated, not merely reported.
    density_truncated_steps = 0

    steps_per_ep = scaffold_cfg.scaffold_steps_per_episode
    for _ep in range(n_eps):
        _, obs_dict = env.reset()
        agent.reset()
        ep_steps = 0
        for _step in range(steps_per_ep):
            obs_body = obs_dict["body_state"].to(device)
            obs_world = obs_dict["world_state"].to(device)
            with torch.no_grad():
                latent = _sense_with_optional_harm(
                    agent, obs_body, obs_world, obs_dict, device,
                    scaffold_cfg.scaffold_feed_harm_stream,
                )
                ticks = agent.clock.advance()
                e1_prior = (
                    agent._e1_tick(latent) if ticks.get("e1_tick")
                    else torch.zeros(1, agent.config.latent.world_dim, device=device)
                )
                candidates = agent.generate_trajectories(latent, e1_prior, ticks)
                action = agent.select_action(candidates, ticks)

            action_idx = int(action.argmax(dim=-1).item())
            _, harm_signal, done, info, obs_dict = env.step(action_idx)
            total_steps += 1
            ep_steps += 1

            benefit, drive = _ben_drive(obs_dict["body_state"].to(device))
            consumed_tag = _consumed_type_tag_from_info(info)

            # Read the truncation diagnostic from whichever of info / obs_dict carries it
            # (env.step populates the SD-049 block; both dicts are checked so this does not
            # depend on which one the substrate happens to surface it through).
            for _src in (info, obs_dict):
                if isinstance(_src, dict) and "sd049_density_budget_truncated" in _src:
                    if bool(_src["sd049_density_budget_truncated"]):
                        density_truncated_steps += 1
                    break

            # D1 contact counter -- 693a's rule verbatim.
            if benefit > SEED_BENEFIT_THRESHOLD or consumed_tag is not None:
                contact_steps += 1

            if consumed_tag is not None:
                consumption_events += 1
                pad2 = _per_axis_drive_from_obs(obs_dict, device)
                if pad2 is not None:
                    agent._per_axis_drive = pad2.reshape(-1)
                with torch.no_grad():
                    try:
                        agent.update_z_goal(float(benefit), drive_level=float(drive),
                                            resource_type=consumed_tag)
                    except TypeError:
                        agent.update_z_goal(float(benefit), drive_level=float(drive))

            if done:
                break
        episode_lengths.append(ep_steps)

    behav_contact_rate = (float(contact_steps) / float(total_steps)) if total_steps > 0 else 0.0
    return {
        "behav_contact_rate": behav_contact_rate,
        "behav_contact_steps": int(contact_steps),
        "behav_total_steps": int(total_steps),
        "behav_consumption_events": int(consumption_events),
        "behav_mean_episode_length": float(np.mean(episode_lengths)) if episode_lengths else 0.0,
        "behav_episode_lengths": [int(x) for x in episode_lengths],
        "density_budget_truncated_frac": (
            float(density_truncated_steps) / float(total_steps) if total_steps > 0 else 0.0
        ),
    }


def _aborted_record(seed: int, arm_id: str, label: str, curriculum_amended: bool,
                    density_on: bool, stage: str, reason: str) -> Dict[str, Any]:
    return {
        "seed": seed, "arm": arm_id, "label": label,
        "curriculum_amended": bool(curriculum_amended), "density_on": bool(density_on),
        "aborted_at": stage, "abort_reason": reason,
        "guard_pass": False, "hazard_stage_survival_pass": False,
        "p1_survival_pass": False,
        "p2_contact_rate": 0.0, "p2_z_goal_norm_at_contact_peak": 0.0,
        "p2_num_contact_events": 0,
        "behav_contact_rate": 0.0, "behav_contact_steps": 0, "behav_total_steps": 0,
        "behav_consumption_events": 0, "behav_mean_episode_length": 0.0,
        "behav_episode_lengths": [], "density_budget_truncated_frac": 0.0,
        "d1_contact_pass": False, "d2_survival_pass": False, "d3_clears": False,
    }


def _cell_total_eps(curriculum_amended: bool, dry_run: bool) -> int:
    if dry_run:
        return 2 + 2 + 5 + 5 + 5 + 2 + 3
    p0 = P0_BUDGET_AMENDED if curriculum_amended else P0_BUDGET
    hz = HAZARD_STAGE_BUDGET_AMENDED if curriculum_amended else HAZARD_STAGE_BUDGET
    return STAGE0_BUDGET + STAGE0B_BUDGET + p0 + hz + P1_BUDGET + P2_BUDGET + BEHAV_EVAL_EPISODES


def _full_config(curriculum_amended: bool, density_on: bool, dry_run: bool) -> Dict[str, Any]:
    """The declared config slice for the arm fingerprint. Declares ONLY what the cell's
    build+train+eval path reads -- schedule, env/substrate operating config, and the two 2x2
    axes. No acceptance thresholds (they gate scoring, not computation)."""
    return {
        "sd049_on": ARM2_SD049_ON,
        "n_resource_types": ARM2_N_RESOURCE_TYPES,
        "curriculum_amended": bool(curriculum_amended),
        "density_on": bool(density_on),
        "stage0_budget": STAGE0_BUDGET,
        "stage0b_budget": STAGE0B_BUDGET,
        "p0_budget": P0_BUDGET_AMENDED if curriculum_amended else P0_BUDGET,
        "hazard_stage_budget": HAZARD_STAGE_BUDGET_AMENDED if curriculum_amended else HAZARD_STAGE_BUDGET,
        "hazard_stage_num_resources": (
            HAZARD_STAGE_NUM_RESOURCES_AMENDED if curriculum_amended else HAZARD_STAGE_NUM_RESOURCES
        ),
        "hazard_stage_num_hazards": HAZARD_STAGE_NUM_HAZARDS,
        "hazard_stage_survival_gate_steps": HAZARD_STAGE_SURVIVAL_GATE_STEPS,
        "hazard_stage_stability_window": HAZARD_STAGE_STABILITY_WINDOW,
        "p1_budget": P1_BUDGET,
        "p2_budget": P2_BUDGET,
        "behav_eval_episodes": BEHAV_EVAL_EPISODES,
        "train_steps": TRAIN_STEPS,
        "p0_num_hazards": P0_NUM_HAZARDS,
        "p1_hold_fraction": P1_HOLD_FRACTION,
        "p2_hfa_guard": P2_HFA_GUARD,
        "p1_reef_spawn_hold_fraction": P1_REEF_SPAWN_HOLD_FRACTION,
        "seed_gain": SEED_GAIN,
        "seed_benefit_threshold": SEED_BENEFIT_THRESHOLD,
        "seed_drive_floor": SEED_DRIVE_FLOOR,
        "cue_recall_gain": CUE_RECALL_GAIN,
        "world_dim": WORLD_DIM,
        "alpha_world": 0.9,
        "drive_weight": DRIVE_WEIGHT,
        "dry_run": bool(dry_run),
    }


def _run_seed_arm(seed: int, arm_id: str, label: str, curriculum_amended: bool,
                  density_on: bool, dry_run: bool) -> Dict[str, Any]:
    total_eps = _cell_total_eps(curriculum_amended, dry_run)
    cfg_slice = _full_config(curriculum_amended, density_on, dry_run)

    # Complete RNG reset at cell entry + per-cell fingerprint stamp, in one call.
    # include_driver_script_in_hash=False so a later consumer with a DIFFERENT driver (e.g. the
    # 693b measurement re-issue reusing the A11 competence-cleared cell) can match this mint.
    with arm_cell(seed, config_slice=cfg_slice, script_path=Path(__file__),
                  config_slice_declared=True,
                  include_driver_script_in_hash=False) as cell:
        torch.manual_seed(seed)
        np.random.seed(seed)
        scaffold_cfg = _make_scaffold_cfg(dry_run, curriculum_amended, density_on)
        device = torch.device("cpu")

        probe_env = _build_env(scaffold_cfg, "p2")
        probe_env.reset()
        agent = REEAgent(_make_config(probe_env)).to(device)
        scheduler = ScaffoldedSD054OnboardingScheduler(scaffold_cfg)

        # Canonical seed/condition boundary line (runner resets episodes_in_run on this).
        print(f"Seed {seed} Condition {arm_id}_{label}", flush=True)
        print(f"[{arm_id}/{label}] seed {seed} curriculum_amended={curriculum_amended}"
              f" density_on={density_on} n_types={ARM2_N_RESOURCE_TYPES}", flush=True)
        done_eps = 0

        s0 = scheduler.run_stage0_nursery(agent, device)
        done_eps += s0.n_episodes
        if s0.aborted:
            print(f"  [train] {arm_id} seed={seed} ep {done_eps}/{total_eps} aborted=stage0", flush=True)
            print(f"verdict: FAIL arm={arm_id} seed={seed} aborted_at=stage0"
                  f" reason={s0.abort_reason}", flush=True)
            rec = _aborted_record(seed, arm_id, label, curriculum_amended, density_on,
                                  "stage0", s0.abort_reason)
            cell.stamp(rec)
            return rec

        s0b = scheduler.run_stage0b_consolidation(
            agent, device, stage0_baseline_norm=s0.z_goal_norm_peak)
        done_eps += s0b.n_episodes
        if s0b.aborted:
            print(f"  [train] {arm_id} seed={seed} ep {done_eps}/{total_eps} aborted=stage0b", flush=True)
            print(f"verdict: FAIL arm={arm_id} seed={seed} aborted_at=stage0b"
                  f" reason={s0b.abort_reason}", flush=True)
            rec = _aborted_record(seed, arm_id, label, curriculum_amended, density_on,
                                  "stage0b", s0b.abort_reason)
            cell.stamp(rec)
            return rec

        p0 = scheduler.run_p0(agent, device)
        done_eps += p0.n_episodes
        print(f"  [train] p0 arm={arm_id} seed={seed} ep {done_eps}/{total_eps}", flush=True)
        if p0.aborted:
            print(f"verdict: FAIL arm={arm_id} seed={seed} aborted_at=p0"
                  f" reason={p0.abort_reason}", flush=True)
            rec = _aborted_record(seed, arm_id, label, curriculum_amended, density_on,
                                  "p0", p0.abort_reason)
            cell.stamp(rec)
            return rec

        # --- D2: the isolated Stage-H survival gate (one of the two 693a failure legs) ---
        hz = scheduler.run_hazard_avoidance(agent, device)
        done_eps += hz.n_episodes
        print(f"  [train] hazard arm={arm_id} seed={seed} ep {done_eps}/{total_eps}"
              f" survival_pass={bool(hz.survival_gate_passed)}", flush=True)
        if hz.aborted:
            print(f"verdict: FAIL arm={arm_id} seed={seed} aborted_at=hazard"
                  f" reason={hz.abort_reason}", flush=True)
            rec = _aborted_record(seed, arm_id, label, curriculum_amended, density_on,
                                  "hazard", hz.abort_reason)
            cell.stamp(rec)
            return rec

        p1 = scheduler.run_p1(agent, device)
        done_eps += p1.n_episodes

        p2 = scheduler.run_p2(agent, device)
        done_eps += p2.n_episodes
        print(f"  [train] p2_guard arm={arm_id} seed={seed} ep {done_eps}/{total_eps}"
              f" contact_rate={p2.contact_rate:.4f} events={p2.num_contact_events}"
              f" z_goal_at_contact={p2.z_goal_norm_at_contact_peak:.4f}", flush=True)

        guard_pass = bool(
            p2.contact_rate > CONTACT_GATE
            and p2.z_goal_norm_at_contact_peak > P2_ZGOAL_GATE
        )

        # --- D1: the behavioural contact rate (the other 693a failure leg) ---
        n_eval = 3 if dry_run else BEHAV_EVAL_EPISODES
        behav = _run_contact_eval(agent, scaffold_cfg, device, n_eval)
        done_eps += n_eval

        d1 = bool(behav["behav_contact_rate"] > CONSUMPTION_FLOOR)
        d2 = bool(hz.survival_gate_passed)
        d3 = bool(d1 and d2)

        print(f"  [eval] arm={arm_id} seed={seed} contact={behav['behav_contact_rate']:.4f}"
              f" (floor {CONSUMPTION_FLOOR}) d1={d1} survival={d2} clears={d3}"
              f" guard_pass={guard_pass}", flush=True)
        print(f"verdict: {'PASS' if d3 else 'FAIL'} arm={arm_id} seed={seed}"
              f" d1_contact={d1} d2_survival={d2}", flush=True)

        rec: Dict[str, Any] = {
            "seed": seed, "arm": arm_id, "label": label,
            "curriculum_amended": bool(curriculum_amended), "density_on": bool(density_on),
            "aborted_at": None, "abort_reason": "",
            "guard_pass": guard_pass,
            "stage0_z_goal_norm_peak": float(s0.z_goal_norm_peak),
            "p1_survival_pass": bool(p1.survival_gate_passed),
            "hazard_stage_survival_pass": bool(hz.survival_gate_passed),
            "p2_contact_rate": float(p2.contact_rate),
            "p2_z_goal_norm_at_contact_peak": float(p2.z_goal_norm_at_contact_peak),
            "p2_num_contact_events": int(p2.num_contact_events),
            "d1_contact_pass": d1, "d2_survival_pass": d2, "d3_clears": d3,
        }
        rec.update(behav)
        cell.stamp(rec)
    return rec


def _frac(flags: List[bool]) -> float:
    return float(sum(1 for f in flags if f)) / float(len(flags)) if flags else 0.0


def _arm_rows(per_run: List[Dict[str, Any]], arm_id: str) -> List[Dict[str, Any]]:
    return [r for r in per_run if r.get("arm") == arm_id]


def _arm_summary(per_run: List[Dict[str, Any]], arm_id: str, label: str) -> Dict[str, Any]:
    rows = _arm_rows(per_run, arm_id)
    contact_vals = [float(r.get("behav_contact_rate", 0.0)) for r in rows]
    d1_frac = _frac([bool(r.get("d1_contact_pass")) for r in rows])
    d2_frac = _frac([bool(r.get("d2_survival_pass")) for r in rows])
    # D3 "arm clears" = BOTH per-arm seed fractions meet MIN_FRACTION. Evaluated on the two
    # legs SEPARATELY (not on the per-seed conjunction) because the 693a target is stated that
    # way: "contact > 0.02 AND hazard-stage survival on >= 2/3 seeds".
    clears = bool(d1_frac >= MIN_FRACTION and d2_frac >= MIN_FRACTION)
    return {
        "arm": arm_id, "label": label,
        "n_seeds": len(rows),
        "mean_behav_contact_rate": float(np.mean(contact_vals)) if contact_vals else 0.0,
        "min_behav_contact_rate": float(np.min(contact_vals)) if contact_vals else 0.0,
        "max_behav_contact_rate": float(np.max(contact_vals)) if contact_vals else 0.0,
        "d1_contact_frac": d1_frac,
        "d2_survival_frac": d2_frac,
        "d3_clears": clears,
        "per_seed_behav_contact_rate": contact_vals,
        "per_seed_survival_pass": [bool(r.get("d2_survival_pass")) for r in rows],
        "per_seed_guard_pass": [bool(r.get("guard_pass")) for r in rows],
        # Worst (not mean) truncated fraction: a single truncated cell means this arm's
        # density manipulation did not fully fire, and a mean would hide it.
        "max_density_budget_truncated_frac": max(
            [float(r.get("density_budget_truncated_frac", 0.0)) for r in rows] or [0.0]
        ),
        "n_aborted": sum(1 for r in rows if r.get("aborted_at")),
    }


def run_experiment(dry_run: bool = False) -> Dict[str, Any]:
    t0 = time.perf_counter()
    print(f"[{EXPERIMENT_TYPE}] starting (dry_run={dry_run})", flush=True)

    # --- P_DENSITY_FLAG, measured in SETUP before any compute is spent ---
    density_supported = density_flag_supported()
    print(f"[setup] P_DENSITY_FLAG: env accepts sd049_preserve_per_type_density"
          f" = {density_supported}", flush=True)

    seeds = SEEDS[:1] if dry_run else SEEDS
    arms = ARM_SPECS

    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{timestamp}_v3"

    if not density_supported:
        # The density-ON arms are unconstructible -> the 2x2 is not a 2x2. Self-route
        # substrate_not_ready_requeue. NEVER a verdict label: nothing about the ceiling has
        # been measured, and the curriculum-only contrast alone cannot answer the attribution
        # question this probe exists for.
        print("[setup] ABORT: the env-side density-preserving spawn kwarg has not landed."
              " Self-routing substrate_not_ready_requeue (no compute spent).", flush=True)
        manifest: Dict[str, Any] = {
            "run_id": run_id,
            "experiment_type": EXPERIMENT_TYPE,
            "queue_id": QUEUE_ID,
            "architecture_epoch": "ree_hybrid_guardrails_v1",
            "claim_ids": CLAIM_IDS,
            "experiment_purpose": EXPERIMENT_PURPOSE,
            "outcome": "FAIL",
            "timestamp_utc": timestamp,
            "sleep_driver_pattern": "N/A",
            "non_degenerate": False,
            "degeneracy_reason": (
                "P_DENSITY_FLAG unmet: CausalGridWorldV2.__init__ does not accept"
                " sd049_preserve_per_type_density, so the density-ON arms (A01/A11) are"
                " unconstructible and the 2x2 attribution design collapses to a"
                " curriculum-only contrast that cannot separate H_DENSITY from"
                " H_CURRICULUM. No competence measurement was attempted."
            ),
            "interpretation": {
                "label": "substrate_not_ready_requeue",
                "preconditions": [{
                    "name": "density_preserving_spawn_kwarg_available",
                    "description": ("env-side sd049_preserve_per_type_density kwarg present on"
                                    " CausalGridWorldV2.__init__ (owned by a separate"
                                    " implement-substrate session)"),
                    "measured": 0.0, "threshold": 1.0, "direction": "lower",
                    "control": "constructor-signature introspection in setup",
                    "met": False,
                }],
                "criteria_non_degenerate": {
                    "C_BASE_FAILS": False, "C_CURR": False,
                    "C_DENS": False, "C_JOINT": False,
                },
                "criteria": [
                    {"name": "C_JOINT", "load_bearing": True, "passed": False},
                ],
            },
            "arm_results": [],
        }
        stamp_recording_core(
            manifest,
            config={"density_flag_supported": False,
                    "arms": [a[0] for a in arms], "seeds": seeds},
            seeds=seeds, script_path=Path(__file__), started_at=t0,
        )
        return manifest

    per_run: List[Dict[str, Any]] = []
    for arm_id, label, curriculum_amended, density_on in arms:
        for seed in seeds:
            per_run.append(_run_seed_arm(seed, arm_id, label, curriculum_amended,
                                         density_on, dry_run))

    arm_summaries = [
        _arm_summary(per_run, arm_id, label) for arm_id, label, _, _ in arms
    ]
    by_arm = {a["arm"]: a for a in arm_summaries}

    # --- P_BASE_REPRO: did the 693a ARM_2 ceiling reproduce on the baseline arm? ---
    # UPPER-bound precondition: the baseline arm is EXPECTED to sit at or below the floor.
    # measured > threshold means the failure did NOT reproduce -> the three contrasts are
    # deltas against a baseline that is not the failure under investigation.
    base_mean_contact = float(by_arm[BASELINE_ARM]["mean_behav_contact_rate"])
    base_reproduced = bool(not by_arm[BASELINE_ARM]["d3_clears"])

    # --- P_DENSITY_EFFECTIVE: did the density manipulation actually FIRE on the ON arms? ---
    # The env sets sd049_density_budget_truncated when the forage pool could not honour the
    # scaled budget. A truncated density arm has NOT had per-type density restored -- it is
    # re-confounded by exactly the effect under test -- so A01/A11 would be measuring the same
    # starved substrate as A00/A10 while presenting as the treatment. That is a manipulation
    # that did not fire, which is a readiness failure, NOT a finding about the ceiling.
    density_on_rows = [r for r in per_run if r.get("density_on")]
    density_trunc_frac = max(
        [float(r.get("density_budget_truncated_frac", 0.0)) for r in density_on_rows] or [0.0]
    )
    worst_trunc_cell = ""
    for r in density_on_rows:
        if float(r.get("density_budget_truncated_frac", 0.0)) == density_trunc_frac:
            worst_trunc_cell = f"{r.get('arm')}/seed{r.get('seed')}"
            break

    # --- pre-registered criteria ---
    c_base_fails = base_reproduced
    c_curr = bool(by_arm["A10"]["d3_clears"])
    c_dens = bool(by_arm["A01"]["d3_clears"])
    c_joint = bool(by_arm[JOINT_ARM]["d3_clears"])

    # --- attribution readout (reported, not gated) ---
    def _mc(a: str) -> float:
        return float(by_arm[a]["mean_behav_contact_rate"])

    def _sf(a: str) -> float:
        return float(by_arm[a]["d2_survival_frac"])

    attribution = {
        "contact_rate_2x2": {a["arm"]: a["mean_behav_contact_rate"] for a in arm_summaries},
        "survival_frac_2x2": {a["arm"]: a["d2_survival_frac"] for a in arm_summaries},
        # main effects averaged over the other axis
        "contact_curriculum_main_effect": ((_mc("A10") + _mc("A11")) - (_mc("A00") + _mc("A01"))) / 2.0,
        "contact_density_main_effect": ((_mc("A01") + _mc("A11")) - (_mc("A00") + _mc("A10"))) / 2.0,
        "contact_interaction": (_mc("A11") - _mc("A10")) - (_mc("A01") - _mc("A00")),
        "survival_curriculum_main_effect": ((_sf("A10") + _sf("A11")) - (_sf("A00") + _sf("A01"))) / 2.0,
        "survival_density_main_effect": ((_sf("A01") + _sf("A11")) - (_sf("A00") + _sf("A10"))) / 2.0,
        "survival_interaction": (_sf("A11") - _sf("A10")) - (_sf("A01") - _sf("A00")),
    }

    # The attribution question the chip asks, answered explicitly from the survival legs.
    if _sf("A01") >= MIN_FRACTION and _sf("A00") < MIN_FRACTION:
        survival_attribution = "downstream_of_resource_starvation"
    elif _sf("A10") >= MIN_FRACTION and _sf("A01") < MIN_FRACTION:
        survival_attribution = "independent_of_starvation_training_budget_bound"
    elif _sf("A11") >= MIN_FRACTION and _sf("A10") < MIN_FRACTION and _sf("A01") < MIN_FRACTION:
        survival_attribution = "interaction_neither_lever_sufficient_alone"
    elif _sf("A00") >= MIN_FRACTION:
        survival_attribution = "baseline_survival_did_not_reproduce_failure"
    else:
        survival_attribution = "unrescued_by_either_lever"

    preconditions = [
        {
            "name": "density_preserving_spawn_kwarg_available",
            "description": ("env-side sd049_preserve_per_type_density kwarg present on"
                            " CausalGridWorldV2.__init__"),
            "measured": 1.0, "threshold": 1.0, "direction": "lower",
            "control": "constructor-signature introspection in setup",
            "met": True,
        },
        {
            "name": "baseline_arm_reproduces_693a_ceiling",
            "description": ("A00 (693a curriculum, density OFF) must NOT clear the D3 gate --"
                            " an upper-bound check that the ceiling under investigation"
                            " actually reproduced, so the three contrasts are deltas against"
                            " the real failure. measured = A00 mean behav_contact_rate."),
            "measured": base_mean_contact,
            "threshold": CONSUMPTION_FLOOR,
            "direction": "upper",
            "control": ("A00 is a faithful re-run of the 693a ARM_2 cell, whose observed"
                        " contact rate was 0.0099-0.0188 across 3 seeds"),
            "met": bool(base_reproduced),
        },
        {
            "name": "density_manipulation_effective_on_on_arms",
            "description": ("the SD-049 density-preserving spawn must not have been truncated"
                            " by the forage pool on the density-ON arms, else per-type density"
                            " was never restored and A01/A11 are re-confounded by the very"
                            " effect under test. measured = WORST (max) per-cell truncated-step"
                            " fraction across all density-ON cells, so a single truncated cell"
                            " cannot be averaged away."),
            "measured": density_trunc_frac,
            "threshold": 0.0,
            "direction": "upper",
            "comparator": "<=",
            "control": ("density-ON cells on a forage pool large enough for"
                        " num_resources x n_active_types; env emits"
                        " sd049_density_budget_truncated when it is not"),
            "offending_cell": worst_trunc_cell,
            "met": bool(density_trunc_frac <= 0.0),
        },
    ]

    all_preconditions_met = all(bool(p["met"]) for p in preconditions)

    # Non-degeneracy: the D1 leg is degenerate if contact rate has no cross-arm spread at all
    # (every arm pinned), which would mean neither lever moved the DV the criteria route on.
    contact_means = [float(a["mean_behav_contact_rate"]) for a in arm_summaries]
    contact_spread = (max(contact_means) - min(contact_means)) if contact_means else 0.0
    survival_fracs = [float(a["d2_survival_frac"]) for a in arm_summaries]
    survival_spread = (max(survival_fracs) - min(survival_fracs)) if survival_fracs else 0.0
    d1_non_degenerate = bool(contact_spread > 1e-6)
    d2_non_degenerate = bool(survival_spread > 1e-6)

    if not all_preconditions_met:
        outcome = "FAIL"
        label = "substrate_not_ready_requeue"
        non_degenerate = False
        unmet = [p["name"] for p in preconditions if not p["met"]]
        reasons = []
        if "baseline_arm_reproduces_693a_ceiling" in unmet:
            reasons.append(
                "P_BASE_REPRO unmet: the A00 baseline arm CLEARED the D3 gate"
                f" (mean behav_contact_rate {base_mean_contact:.4f} > CONSUMPTION_FLOOR"
                f" {CONSUMPTION_FLOOR}), so the V3-EXQ-693a ARM_2 ceiling did not reproduce"
                " and the curriculum/density contrasts are deltas against a baseline that is"
                " not the failure under investigation. Re-queue after establishing why the"
                " baseline drifted; this is NOT evidence that either lever works."
            )
        if "density_manipulation_effective_on_on_arms" in unmet:
            reasons.append(
                "P_DENSITY_EFFECTIVE unmet: the density-preserving spawn was TRUNCATED by the"
                f" forage pool on {density_trunc_frac:.3f} of steps at worst"
                f" ({worst_trunc_cell}), so per-type density was never actually restored on the"
                " density-ON arms and A01/A11 measured the same starved substrate as A00/A10"
                " while presenting as the treatment. The density axis did not fire; any"
                " apparent null on it is an artifact, NOT evidence against H_DENSITY. Re-queue"
                " with a larger forage pool (bigger env or fewer hazards) so"
                " num_resources x n_active_types fits."
            )
        degeneracy_reason = " | ".join(reasons)
    else:
        outcome = "PASS" if c_joint else "FAIL"
        non_degenerate = bool(d1_non_degenerate or d2_non_degenerate)
        degeneracy_reason = "" if non_degenerate else (
            "Neither D1 (contact rate) nor D2 (survival fraction) showed any cross-arm spread:"
            " both levers left the routed statistics pinned, so the criteria could not"
            " discriminate."
        )
        if c_joint and c_curr and c_dens:
            label = "both_levers_sufficient_alone"
        elif c_joint and c_curr:
            label = "curriculum_sufficient_density_not_required"
        elif c_joint and c_dens:
            label = "density_sufficient_curriculum_not_required"
        elif c_joint:
            label = "interaction_required_neither_lever_sufficient_alone"
        else:
            label = "arm2_competence_ceiling_unrescued_by_either_lever"

    print(f"[{EXPERIMENT_TYPE}] C_BASE_FAILS={c_base_fails} C_CURR={c_curr}"
          f" C_DENS={c_dens} C_JOINT={c_joint} -> {label}", flush=True)
    print(f"[{EXPERIMENT_TYPE}] survival_attribution={survival_attribution}", flush=True)

    manifest: Dict[str, Any] = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": QUEUE_ID,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "outcome": outcome,
        "timestamp_utc": timestamp,
        "sleep_driver_pattern": "N/A",
        "predecessor": "V3-EXQ-693a",
        "non_degenerate": non_degenerate,
        "degeneracy_reason": degeneracy_reason,
        "criteria_results": {
            "C_BASE_FAILS": c_base_fails,
            "C_CURR": c_curr,
            "C_DENS": c_dens,
            "C_JOINT": c_joint,
        },
        "attribution": attribution,
        "survival_attribution": survival_attribution,
        "arm_summaries": arm_summaries,
        "per_seed_results": per_run,
        "arm_results": per_run,
        "interpretation": {
            "label": label,
            "preconditions": preconditions,
            "criteria_non_degenerate": {
                "C_BASE_FAILS": d1_non_degenerate or d2_non_degenerate,
                "C_CURR": d1_non_degenerate or d2_non_degenerate,
                "C_DENS": d1_non_degenerate or d2_non_degenerate,
                "C_JOINT": d1_non_degenerate or d2_non_degenerate,
            },
            "criteria": [
                {"name": "C_BASE_FAILS", "load_bearing": False, "passed": c_base_fails},
                {"name": "C_CURR", "load_bearing": False, "passed": c_curr},
                {"name": "C_DENS", "load_bearing": False, "passed": c_dens},
                {"name": "C_JOINT", "load_bearing": True, "passed": c_joint},
            ],
        },
        "thresholds": {
            "consumption_floor": CONSUMPTION_FLOOR,
            "hazard_stage_survival_gate_steps": HAZARD_STAGE_SURVIVAL_GATE_STEPS,
            "p2_zgoal_gate": P2_ZGOAL_GATE,
            "contact_gate": CONTACT_GATE,
            "min_fraction": MIN_FRACTION,
        },
        "curriculum_levers": {
            "p0_budget": {"base": P0_BUDGET, "amended": P0_BUDGET_AMENDED},
            "hazard_stage_budget": {"base": HAZARD_STAGE_BUDGET,
                                    "amended": HAZARD_STAGE_BUDGET_AMENDED},
            "hazard_stage_num_resources": {"base": HAZARD_STAGE_NUM_RESOURCES,
                                           "amended": HAZARD_STAGE_NUM_RESOURCES_AMENDED},
        },
    }

    # Multi-arm: stamp AFTER arm_results is assembled so substrate_hash HOISTS from the
    # per-cell fingerprints rather than being recomputed driver-inclusive.
    stamp_recording_core(
        manifest,
        config={
            "arms": [{"arm": a, "label": l, "curriculum_amended": c, "density_on": d}
                     for a, l, c, d in arms],
            "base_curriculum": _full_config(False, False, dry_run),
            "amended_curriculum": _full_config(True, True, dry_run),
            "density_flag_supported": True,
        },
        seeds=seeds, script_path=Path(__file__), started_at=t0,
    )
    return manifest


def main(dry_run: bool = False) -> Dict[str, Any]:
    manifest = run_experiment(dry_run=dry_run)
    out_dir = (REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments")
    # stamp=False: run_experiment already called stamp_recording_core AFTER arm_results was
    # assembled, so substrate_hash hoisted from the per-cell fingerprints. Re-stamping here
    # is no-op-safe but would recompute nothing useful.
    out_path = write_flat_manifest(manifest, out_dir, dry_run=dry_run, stamp=False)
    print(f"[{EXPERIMENT_TYPE}] wrote {out_path}", flush=True)
    return {"manifest": manifest, "out_path": out_path}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=EXPERIMENT_TYPE)
    parser.add_argument("--dry-run", action="store_true",
                        help="tiny budgets / 1 seed smoke test")
    args = parser.parse_args()

    result = main(dry_run=args.dry_run)
    _outcome_raw = str(result["manifest"]["outcome"]).upper()
    emit_outcome(
        outcome=_outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL",
        manifest_path=result["out_path"],
        dry_run=args.dry_run,
    )
