"""
V3-EXQ-793a -- SD-049 Phase-2 ARM_2 FORAGING-COMPETENCE CALIBRATION, REPOWERED (2x2, more seeds
on the lever arms + a load-bearing robustness gate on the joint arm).

REPOWERS V3-EXQ-793 (same 2x2 design, same substrate, same hypotheses). Does NOT supersede it --
793's finding stands as recorded (informative but thin). This is a targeted power increase, not a
redesign, per the confirmed autopsy failure_autopsy_V3-EXQ-793_2026-07-24.{md,json}.

WHY THIS EXISTS: 793's arm-level d3_clears criterion (majority-by-mean-contact-rate) passed clean
on A10/A01/A11, licensing the both_levers_sufficient_alone reading. But per-seed guard_pass told a
different story at n=3: A11 (both levers) failed guard_pass on 2/3 seeds ([true,false,false])
despite its arm-level PASS; A01 and A10 each failed guard_pass on 1/3 seeds. d3_clears is computed
from mean behav_contact_rate and does not require guard_pass, so the arm-level PASS did not surface
this fragility. At n=3 a 2/3 guard-failure rate on the arm the whole diagnostic hands off to 693b
is a fragility flag, not noise to wave past -- but n=3 is too thin to tell "genuinely fragile" from
"one unlucky seed" apart. This experiment repowers the three lever arms (A10/A01/A11) with more
seeds and makes guard_pass LOAD-BEARING for the joint arm specifically, so a repeat of the 2/3
guard-failure signature at higher n reads as a decisive, informative finding instead of a fragility
flag riding on n=3.

SCOPE NOTE (do not conflate with SD-049's park): SD-049 is parked candidate/v3_pending/
substrate_ceiling on a DIFFERENT and unaddressed prerequisite -- Phase-2 V3-EXQ-514 full
validation is blocked on a foraging-COMPETENT policy (~0.2% consumption on the enriched reef;
2026-06-19 triage), which neither 793 nor this repowering tests or lifts. This experiment answers
a narrower curriculum/environment-design question (do these two levers robustly clear the D3 gate
on the ARM_2 substrate) and, if decisive, is informative context for that design space -- it is
not a retest of the actual park and must not be read as one.

RE-DERIVE BRAKE (GOV-REUSE-1 / re-derive brake check, run at queue time): the blanket per-claim
autopsy count on SD-049 is 4 (514l, 538a, 693, 693a; all `substrate_ceiling` / non_contributory-on-
the-primary-arm reads). ALL FOUR are about the foraging-competence / WL-dissociation Phase-2
prerequisite -- a different, broader question than the D3-gate curriculum/density lever-sufficiency
question 793/793a test. This repowering is the second attempt at 793's OWN narrower question (a
power-only re-run of a diagnostic discriminating why a ceiling holds), which the skill's own
carve-out exempts from the brake ("a diagnostic whose purpose is to discriminate why the ceiling
holds ... none of these is the re-derive loop"). 793's own re_derive_brake block (citing only 538a,
the one prior autopsy it considered most directly relevant) reached the same release and explicitly
named this exact repowering as the legitimate follow-up. Not braked.

GOV-REUSE-1 (existing-evidence check): the decisive readout -- per-seed guard_pass rate for A10/
A01/A11 at n=6 (up from n=3) -- does not exist in any recorded manifest; 793 is the only run of
this config and it ran exactly 3 seeds. Not recoverable; proceeding to run. Arm-fingerprint reuse
of 793's own cells was considered and NOT used: 793 minted every cell (all 4 arms) reuse-eligible
(include_driver_script_in_hash=False), so in principle this script's original-seed cells could
reuse 793's. But 793 ran on ree-cloud-2 (linux-x86_64-py3.10-torch2.12.0+cpu, substrate_hash
402e3f5a23a3...), and several substrate-touching changes have landed on ree-v3/main since 793's
2026-07-21 run (SD-081 dual-system arbitration, ARC-071 policy composition, MECH-048 salience
overlays, SD-076 headroom repair) that near-certainly bust that fingerprint -- reuse would refuse
(safely) rather than hit, so wiring it up would add code complexity for no expected payoff on a
lower-priority follow-up. All cells run fresh.

THE 2x2 (unchanged from 793 -- all four arms on the ARM_2 substrate: sd049_on=True,
n_resource_types=3), UNIFORMLY repowered to n=6 seeds on every arm:

  arm     curriculum        density-preserving spawn   role
  A00     693a-as-is        OFF                        baseline positive control
  A10     AMENDED           OFF                        H_CURRICULUM main effect
  A01     693a-as-is        ON                         H_DENSITY main effect
  A11     AMENDED           ON                         joint / robustness gate

Uniform seeds (rather than repowering only A10/A01/A11 and leaving A00 at 793's original n=3) is a
deliberate simplification: a non-uniform per-arm seed count would make `seeds x conditions` (the
runner's progress-bar denominator; see Step 5's queue-entry fields) not equal the actual verdict-
line count, which the skill flags as a documented ETA/overshoot failure mode. A00's positive
control does not scientifically NEED repowering (793's 3/3 was already unambiguous), but running
it at n=6 anyway is cheap (250 eps/cell, the base-curriculum length, the cheapest arm in the
design) and strengthens that control's evidence for free rather than leaving it thinner than the
arms it anchors.

New seeds 45-47 are ADDED to 793's original 42/43/44 (not a fresh independent sample), so each
arm's n=6 result extends 793's own 3 cells rather than replacing them with a different sample.

WHAT CHANGED FROM 793 (everything else -- curriculum levers, thresholds, DV-symmetry, precondition
design, env/config plumbing -- is IDENTICAL; this is a power + criteria-design fix, not a redesign):

1. SEEDS: all four arms extend from [42, 43, 44] (n=3) to [42, 43, 44, 45, 46, 47] (n=6).

2. NEW load-bearing criterion C_JOINT_ROBUST, replacing plain C_JOINT as the outcome-determining
   gate: A11 clears D3 AND A11's guard_pass fraction across its (now 6) seeds is >= MIN_FRACTION
   (2/3) -- i.e. guard_pass is now LOAD-BEARING for the joint arm specifically, per the autopsy's
   explicit repair-pathway recommendation ("make guard_pass load-bearing rather than informational
   for the joint-levers arm"). C_JOINT (the original, non-robustness-aware d3_clears-only reading)
   is retained and reported as INFORMATIONAL ONLY -- not load-bearing -- so a reader can see both
   whether the mean-contact-rate criterion still passes AND whether it does so robustly.
   guard_pass_frac is also now reported per arm (A00/A10/A01/A11) as an informational field, so
   A01/A10's 1/3 guard-failure signature (flagged by the autopsy but not made load-bearing per the
   task's explicit scope) remains visible without gating the outcome on it.

3. A new self-route label for the specific fragility signature this repowering exists to
   discriminate: if A11 still clears on mean contact_rate but its guard_pass fraction remains
   below MIN_FRACTION at the higher seed count, that is itself the decisive finding (fragility
   confirmed, not a fluke) and gets its own label (joint_arm_clears_but_fragile_guard_pass) rather
   than being folded into the "unrescued" (c_joint false) or "sufficient" (c_joint_robust true)
   buckets.

DELIBERATELY NOT MOVED from 793 (repeated here verbatim because they are the acceptance criteria,
not levers): CONSUMPTION_FLOOR (0.02), HAZARD_STAGE_SURVIVAL_GATE_STEPS (75), P2_ZGOAL_GATE (0.4),
CONTACT_GATE (0.0), MIN_FRACTION (2/3), and all three curriculum-amend values (P0 100->150,
Stage-H budget 40->80, Stage-H resources 2->3). No threshold here is derived from this run's own
statistics, and MIN_FRACTION (the same 2/3 used for D1/D2 seed aggregation) is reused for the new
guard_pass_frac gate rather than inventing a fresh threshold.

DV-SYMMETRY DECLARATION: unchanged from 793 -- D1 (behav_contact_rate) is a set-aggregate over
seeds/episodes/steps (permutation symmetry), D2 (hazard_stage_survival_pass) is a median-length
threshold (monotone-rescaling symmetry), and the new guard_pass_frac gate is itself a set-aggregate
over seeds of a per-cell boolean (same permutation-symmetry class as D1). Neither curriculum nor
density lever is a relabeling of seeds/episodes or a monotone reparameterisation of episode length
or of the per-seed guard boolean -- both act on the environment and training schedule upstream of
these statistics, so none of the three DVs is invariant under a symmetry that would make its
manipulation arithmetically undetectable.

WHAT A NULL MEANS (extends 793's declared nulls with the new robustness axis):
  A10 null + A01 clears -> H_CURRICULUM refuted at this budget; starvation is the cause.
  A01 null + A10 clears -> H_DENSITY refuted as SUFFICIENT; the ceiling is training budget.
  Both null, A11 clears (robustly) -> genuine interaction; neither lever alone is enough, but the
                                       joint config is a robust hand-off to 693b.
  A11 clears on contact_rate but guard_pass_frac stays < MIN_FRACTION at n=6 -> the joint config is
                                       NOT a robust hand-off even though the arm-level mean passes;
                                       693b should not receive this config as-is without addressing
                                       the guard failure mode (informative, not a green light).
  All four null      -> the ceiling is neither of the two named causes at these magnitudes;
                        routes to /failure-autopsy for a new hypothesis, NOT to a bigger budget.

claim_ids: SD-049 (diagnostic; excluded from confidence/conflict scoring by purpose)
experiment_purpose: diagnostic
predecessor: V3-EXQ-793 (PASS / measurement_gap per autopsy; repowered here, NOT superseded)
does NOT supersede 793 -- 793's recorded finding (informative but thin) stands; this experiment
either confirms it decisively or surfaces the fragility as a decisive finding in its own right.
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
from experiments._lib.z_goal_stream import ZGoalStreamAccumulator  # noqa: E402

# Both readiness preconditions here ARE their own definitions and are reachable by
# construction, so the anchor-reachability guard has nothing to add (identical to 793 --
# neither precondition's design changed in this repowering):
#   density_preserving_spawn_kwarg_available -- a constructor-signature introspection,
#     measured 1.0/0.0 against threshold 1.0.
#   baseline_arm_reproduces_693a_ceiling -- an UPPER-bound gate at CONSUMPTION_FLOOR (0.02)
#     whose known-positive control is V3-EXQ-793's own recorded A00 arm, which cleared the gate
#     (0.0202 mean, upper-bound met) on all 3 seeds -- and, further upstream, 693a's own recorded
#     ARM_2 contact rate (0.0099-0.0188 across seeds 42/43/44), both below the gate.
ANCHOR_REACHABILITY_EXEMPT = (
    "Both readiness preconditions are their own degeneracy definitions and are reachable by"
    " construction: one is a boolean constructor-signature introspection; the other is an"
    " upper-bound gate at CONSUMPTION_FLOOR whose positive control (793's own A00 arm plus 693a's"
    " recorded ARM_2 contact rates 0.0099-0.0188) lies below the gate on every recorded seed."
)

EXPERIMENT_TYPE = "v3_exq_793a_sd049_arm2_competence_repower"
QUEUE_ID = "V3-EXQ-793a"
CLAIM_IDS: List[str] = ["SD-049"]
EXPERIMENT_PURPOSE = "diagnostic"

# Repowered from 793's [42, 43, 44] (n=3) to n=6, uniformly across all four arms -- see the
# module docstring for why uniform (progress-instrumentation denominator) rather than targeting
# only the lever arms.
SEEDS = [42, 43, 44, 45, 46, 47]

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

# --- Goal-pipeline / encoder dims (mirror 793 / 693a / 603n / 514t exactly) ---
WORLD_DIM = 32
HARM_A_DIM = 16
HARM_OBS_A_DIM = 7
HARM_HISTORY_LEN = 10
DRIVE_WEIGHT = 2.0

# --- Curriculum budgets: BASE = 693a / 793 verbatim ---
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

# --- Curriculum budgets: AMENDED (the three 793 levers; identical to 793) ---
P0_BUDGET_AMENDED = 150                     # lever 1: foraging warm-up
HAZARD_STAGE_BUDGET_AMENDED = 80            # lever 2: isolated survival practice
HAZARD_STAGE_NUM_RESOURCES_AMENDED = 3      # lever 3: Stage-H not itself starved under a 3-way split

# --- 634c seeding calibration + SD-057 cue-recall bridge (mirror 793 / 693a / 603n / 514t) ---
SEED_GAIN = 1.5
SEED_BENEFIT_THRESHOLD = 0.02
SEED_DRIVE_FLOOR = 0.9
CUE_RECALL_GAIN = 0.2

# --- SD-058 / MECH-357 protective-scaffold anneal (mirror 793 / 693a) ---
AVOIDANCE_SCAFFOLD_FLOOR_START = 0.8
AVOIDANCE_SCAFFOLD_FLOOR_END = 0.0
AVOIDANCE_THREAT_REF = 0.35
PAG_THETA_FREEZE = 0.8
PAG_DURATION_INPUT_THRESHOLD = 0.2
HARM_PATHWAY_LR = 1e-3
STAGE0B_RETENTION_GATE = 0.75

# --- Pre-registered acceptance thresholds (NOT derived from the run; 793 / 693a verbatim) ---
P2_ZGOAL_GATE = 0.4          # 603n G3 per-seed contact guard
CONTACT_GATE = 0.0           # 603n G2 per-seed contact guard
CONSUMPTION_FLOOR = 0.02     # D1: the 693a behav-eval contact-rate floor
MIN_FRACTION = 2.0 / 3.0     # >= 2/3 seeds for any per-arm aggregate -- ALSO the new
                             # guard_pass_frac robustness gate's threshold (reused, not invented).


def _consumed_type_tag_from_info(info: Dict[str, Any]) -> Optional[int]:
    """681-C4 / 514t read: the AUTHORITATIVE consumed (liking) tag lives in the INFO dict,
    cached by env.step() BEFORE the cell tag is cleared. Copied verbatim from 793/693a so this
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

    Identical check to 793 (the flag landed 2026-07-20 and 793 already ran against it
    successfully; re-checked here rather than assumed, since this is a fresh run against
    whatever substrate is on main at run time)."""
    try:
        from ree_core.environment.causal_grid_world import CausalGridWorld
        return "sd049_preserve_per_type_density" in inspect.signature(
            CausalGridWorld.__init__
        ).parameters
    except Exception:
        return False


def _make_scaffold_cfg(dry_run: bool, curriculum_amended: bool,
                       density_on: bool) -> ScaffoldedSD054OnboardingConfig:
    """793's _make_scaffold_cfg, unchanged. Every knob not named by one of the three curriculum
    levers (or the density flag) is identical to 793's / 693a's ARM_2 cell."""
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
    """Mirror 793 / 693a / 514t: full SD-049 Phase-2 + SD-057 substrate."""
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

    Unchanged from 793: the contact counter, the z_goal refresh at genuine contact, and the
    per-axis-drive assignment are copied so behav_contact_rate here is the SAME statistic whose
    floor 793's (and 693a's) gate tested."""
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

            for _src in (info, obs_dict):
                if isinstance(_src, dict) and "sd049_density_budget_truncated" in _src:
                    if bool(_src["sd049_density_budget_truncated"]):
                        density_truncated_steps += 1
                    break

            # D1 contact counter -- 793/693a's rule verbatim.
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
    """The declared config slice for the arm fingerprint. Identical shape to 793 -- declares ONLY
    what the cell's build+train+eval path reads. No acceptance thresholds (they gate scoring, not
    computation), and no seed-count field (the seed itself is a separate fingerprint input)."""
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


# z_goal-stream liveness, pooled across the run's per-cell agents for the
# manifest block (read at end-of-cell, so no agent is retained for provenance).
_ZG = ZGoalStreamAccumulator()


def _run_seed_arm(seed: int, arm_id: str, label: str, curriculum_amended: bool,
                  density_on: bool, dry_run: bool) -> Dict[str, Any]:
    total_eps = _cell_total_eps(curriculum_amended, dry_run)
    cfg_slice = _full_config(curriculum_amended, density_on, dry_run)

    # Complete RNG reset at cell entry + per-cell fingerprint stamp, in one call.
    # include_driver_script_in_hash=False so a later consumer with a DIFFERENT driver (e.g. the
    # 693b measurement re-issue) can match this mint, and so this script's own seeds 42/43/44
    # cells fingerprint-match 793's cells for the SAME (arm, seed) where the substrate is
    # unchanged -- i.e. this run's own A00 cells are, in principle, the same content as 793's.
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
            _ZG.observe(agent)
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
            _ZG.observe(agent)
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
            _ZG.observe(agent)
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
            _ZG.observe(agent)
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
    # z_goal liveness -- read AFTER this cell stepped; the agent is not retained.
    _ZG.observe(agent)
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
    guard_pass_flags = [bool(r.get("guard_pass")) for r in rows]
    guard_pass_frac = _frac(guard_pass_flags)
    # D3 "arm clears" = BOTH per-arm seed fractions meet MIN_FRACTION. Evaluated on the two
    # legs SEPARATELY (not on the per-seed conjunction) because the 693a/793 target is stated
    # that way: "contact > 0.02 AND hazard-stage survival on >= 2/3 seeds". guard_pass_frac is
    # NOT folded into `clears` here -- it is reported per-arm for all four arms, and is made
    # load-bearing ONLY for the joint arm via the separate C_JOINT_ROBUST criterion below, per
    # the autopsy's explicit scope (A00/A10/A01 keep their existing d3_clears-only reading).
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
        "guard_pass_frac": guard_pass_frac,
        "per_seed_behav_contact_rate": contact_vals,
        "per_seed_survival_pass": [bool(r.get("d2_survival_pass")) for r in rows],
        "per_seed_guard_pass": guard_pass_flags,
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

    arms = ARM_SPECS
    seeds = SEEDS[:1] if dry_run else SEEDS

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
            "predecessor": "V3-EXQ-793",
            "non_degenerate": False,
            "degeneracy_reason": (
                "P_DENSITY_FLAG unmet: CausalGridWorld.__init__ does not accept"
                " sd049_preserve_per_type_density, so the density-ON arms (A01/A11) are"
                " unconstructible and the 2x2 attribution design collapses to a"
                " curriculum-only contrast that cannot separate H_DENSITY from"
                " H_CURRICULUM. No competence measurement was attempted. This would be"
                " a substrate REGRESSION relative to V3-EXQ-793, which ran successfully"
                " against this same flag on 2026-07-21."
            ),
            "interpretation": {
                "label": "substrate_not_ready_requeue",
                "preconditions": [{
                    "name": "density_preserving_spawn_kwarg_available",
                    "description": ("env-side sd049_preserve_per_type_density kwarg present on"
                                    " CausalGridWorld.__init__ (present as of V3-EXQ-793,"
                                    " 2026-07-21; re-checked here since the substrate may have"
                                    " moved)"),
                    "measured": 0.0, "threshold": 1.0, "direction": "lower",
                    "control": "constructor-signature introspection in setup",
                    "met": False,
                }],
                "criteria_non_degenerate": {
                    "C_BASE_FAILS": False, "C_CURR": False,
                    "C_DENS": False, "C_JOINT": False, "C_JOINT_ROBUST": False,
                },
                "criteria": [
                    {"name": "C_JOINT_ROBUST", "load_bearing": True, "passed": False},
                ],
            },
            "arm_results": [],
        }
        stamp_recording_core(
            manifest,
            config={"density_flag_supported": False,
                    "arms": [a[0] for a in arms], "seeds": seeds},
            seeds=seeds, script_path=Path(__file__), started_at=t0,
            z_goal_stream_stats=_ZG.stats(),
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
    base_mean_contact = float(by_arm[BASELINE_ARM]["mean_behav_contact_rate"])
    base_reproduced = bool(not by_arm[BASELINE_ARM]["d3_clears"])

    # --- P_DENSITY_EFFECTIVE: did the density manipulation actually FIRE on the ON arms? ---
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
    joint_guard_pass_frac = float(by_arm[JOINT_ARM]["guard_pass_frac"])
    # NEW load-bearing criterion (793a): the joint arm must clear D3 AND clear guard_pass on
    # >= MIN_FRACTION of its (now repowered) seeds. Reuses MIN_FRACTION rather than a new
    # threshold, per the autopsy's repair-pathway recommendation.
    c_joint_robust = bool(c_joint and joint_guard_pass_frac >= MIN_FRACTION)

    # --- attribution readout (reported, not gated) ---
    def _mc(a: str) -> float:
        return float(by_arm[a]["mean_behav_contact_rate"])

    def _sf(a: str) -> float:
        return float(by_arm[a]["d2_survival_frac"])

    attribution = {
        "contact_rate_2x2": {a["arm"]: a["mean_behav_contact_rate"] for a in arm_summaries},
        "survival_frac_2x2": {a["arm"]: a["d2_survival_frac"] for a in arm_summaries},
        "guard_pass_frac_2x2": {a["arm"]: a["guard_pass_frac"] for a in arm_summaries},
        # main effects averaged over the other axis
        "contact_curriculum_main_effect": ((_mc("A10") + _mc("A11")) - (_mc("A00") + _mc("A01"))) / 2.0,
        "contact_density_main_effect": ((_mc("A01") + _mc("A11")) - (_mc("A00") + _mc("A10"))) / 2.0,
        "contact_interaction": (_mc("A11") - _mc("A10")) - (_mc("A01") - _mc("A00")),
        "survival_curriculum_main_effect": ((_sf("A10") + _sf("A11")) - (_sf("A00") + _sf("A01"))) / 2.0,
        "survival_density_main_effect": ((_sf("A01") + _sf("A11")) - (_sf("A00") + _sf("A10"))) / 2.0,
        "survival_interaction": (_sf("A11") - _sf("A10")) - (_sf("A01") - _sf("A00")),
    }

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
                            " CausalGridWorld.__init__"),
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
            "control": ("A00 is a faithful re-run of the 693a / V3-EXQ-793 ARM_2 cell, whose"
                        " observed contact rate was 0.0099-0.0202 across recorded seeds"),
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

    # Non-degeneracy: the D1/D2 legs are degenerate if their routed statistics have no cross-arm
    # spread at all (every arm pinned). guard_pass_frac spread is checked separately since it is
    # now the basis of a load-bearing criterion (C_JOINT_ROBUST) and must itself be non-degenerate
    # (e.g. not 0.0 or 1.0 identically across every arm, which would mean the new gate could
    # never discriminate).
    contact_means = [float(a["mean_behav_contact_rate"]) for a in arm_summaries]
    contact_spread = (max(contact_means) - min(contact_means)) if contact_means else 0.0
    survival_fracs = [float(a["d2_survival_frac"]) for a in arm_summaries]
    survival_spread = (max(survival_fracs) - min(survival_fracs)) if survival_fracs else 0.0
    guard_pass_fracs = [float(a["guard_pass_frac"]) for a in arm_summaries]
    guard_pass_spread = (max(guard_pass_fracs) - min(guard_pass_fracs)) if guard_pass_fracs else 0.0
    d1_non_degenerate = bool(contact_spread > 1e-6)
    d2_non_degenerate = bool(survival_spread > 1e-6)
    guard_non_degenerate = bool(guard_pass_spread > 1e-6)

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
                f" {CONSUMPTION_FLOOR}), so the V3-EXQ-693a/793 ARM_2 ceiling did not reproduce"
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
        outcome = "PASS" if c_joint_robust else "FAIL"
        non_degenerate = bool(d1_non_degenerate or d2_non_degenerate or guard_non_degenerate)
        degeneracy_reason = "" if non_degenerate else (
            "None of contact rate, survival fraction, or guard_pass fraction showed any"
            " cross-arm spread: all three levers left the routed statistics pinned, so the"
            " criteria (including the new C_JOINT_ROBUST gate) could not discriminate."
        )
        if c_joint_robust and c_curr and c_dens:
            label = "both_levers_sufficient_alone_robust"
        elif c_joint_robust and c_curr:
            label = "curriculum_sufficient_density_not_required_robust"
        elif c_joint_robust and c_dens:
            label = "density_sufficient_curriculum_not_required_robust"
        elif c_joint_robust:
            label = "interaction_required_neither_lever_sufficient_alone_robust"
        elif c_joint:
            # The exact fragility signature this repowering exists to discriminate: the arm
            # still clears on mean contact_rate, but guard_pass remains below MIN_FRACTION at
            # the repowered seed count. This is itself a decisive, informative finding -- NOT
            # folded into "unrescued" (which means c_joint is false) or a "*_robust" label
            # (which would overstate what was found).
            label = "joint_arm_clears_but_fragile_guard_pass_below_floor"
        else:
            label = "arm2_competence_ceiling_unrescued_by_either_lever"

    print(f"[{EXPERIMENT_TYPE}] C_BASE_FAILS={c_base_fails} C_CURR={c_curr}"
          f" C_DENS={c_dens} C_JOINT={c_joint} C_JOINT_ROBUST={c_joint_robust}"
          f" (guard_pass_frac[A11]={joint_guard_pass_frac:.3f}) -> {label}", flush=True)
    print(f"[{EXPERIMENT_TYPE}] survival_attribution={survival_attribution}", flush=True)

    manifest = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": QUEUE_ID,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "outcome": outcome,
        "timestamp_utc": timestamp,
        "sleep_driver_pattern": "N/A",
        "predecessor": "V3-EXQ-793",
        "non_degenerate": non_degenerate,
        "degeneracy_reason": degeneracy_reason,
        "criteria_results": {
            "C_BASE_FAILS": c_base_fails,
            "C_CURR": c_curr,
            "C_DENS": c_dens,
            "C_JOINT": c_joint,
            "C_JOINT_ROBUST": c_joint_robust,
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
                "C_JOINT_ROBUST": d1_non_degenerate or d2_non_degenerate or guard_non_degenerate,
            },
            "criteria": [
                {"name": "C_BASE_FAILS", "load_bearing": False, "passed": c_base_fails},
                {"name": "C_CURR", "load_bearing": False, "passed": c_curr},
                {"name": "C_DENS", "load_bearing": False, "passed": c_dens},
                {"name": "C_JOINT", "load_bearing": False, "passed": c_joint},
                {"name": "C_JOINT_ROBUST", "load_bearing": True, "passed": c_joint_robust},
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
        "seeds": seeds,
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
            "seeds": seeds,
        },
        seeds=seeds, script_path=Path(__file__), started_at=t0,
        z_goal_stream_stats=_ZG.stats(),
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
