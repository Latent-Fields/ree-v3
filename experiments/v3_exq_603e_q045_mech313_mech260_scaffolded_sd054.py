#!/opt/local/bin/python3
"""V3-EXQ-603e -- Q-045 MECH-313 + MECH-260 + FP-2 matched-noise dissociation
on the scaffolded_sd054_onboarding substrate (re-issue of V3-EXQ-603d on the
hook-fixed scheduler + corrected config).

Supersedes V3-EXQ-603d. EXPERIMENT_PURPOSE=diagnostic (substrate-readiness
validation of the 2026-06-02 implement-substrate AMEND, not governance
evidence).

WHY 603e (vs 603d)
------------------
The failure_autopsy_V3-EXQ-603d_2026-06-01 found the C4 z_goal=0 was a
Class-1 harness/wiring artifact: (a) ScaffoldedSD054OnboardingScheduler never
called agent.update_z_goal, so GoalState.update was never reached; (b) the
603d bespoke P2 loop also never called it. The 2026-06-02 implement-substrate
AMEND (ree-v3 deb24cc) wired update_z_goal into the scheduler P1/P2. 603e
applies THREE changes vs 603d:
  1. RESTORED BUDGET: SCAFFOLD_P0_BUDGET/SCAFFOLD_P1_BUDGET 30/30 -> 100/50
     (603d's reduced budget confounded the P1 survival gate; 603c used 100/50).
  2. CONFIG FIX: from_dims now sets z_goal_enabled=True + drive_weight=2.0.
     603d omitted z_goal_enabled, so agent.goal_state was None and
     update_z_goal early-returned even with wiring in place (confirmed
     empirically 2026-06-02; the working reference V3-EXQ-622 sets it).
  3. P2 SEEDING: the bespoke _run_p2_measurement loop now calls
     agent.update_z_goal(benefit, drive) after each env.step (mirrors the
     scheduler fix + goal_stream_stages_sd054 reference runner) so the C4
     z_goal_norm_peak metric reflects a driven goal pipeline.
Acceptance (from the 603d failure_record target): z_goal_norm_peak > 0.4 on
>= 2/3 seeds in P2 AND P1 survival gate passed on >= 2/3 seeds.

Substrate-prereq history (carried from 603d; all cleared):
  GAP-C substrate prereqs that stalled the four-iteration 603 chain:

  (1) MECH-307 default-value recalibration -- CLEARED 2026-05-15 by
      V3-EXQ-540g PASS+supports (criterion_fix with delta-criterion C1).
  (2) goal-pipeline training regime produces non-trivial z_goal in default
      config -- CLEARED 2026-06-01T05:55Z by V3-EXQ-621a substrate-readiness
      PASS on the scaffolded_sd054_onboarding substrate landed 2026-05-31.
  (3) InfantCurriculumScheduler Phase 0->1 H_pos floor recalibration --
      CLEARED 2026-05-31 via /implement-substrate (H_POS_FRAC_OF_MAX 0.70
      -> 0.20 in experiments/infant_curriculum.py).

Routing source: behavioral_diversity_isolation_plan.md GAP-C resume_condition
+ failure_autopsy_V3-EXQ-591_2026-05-27.md section 7.

Why this iteration exists
-------------------------
V3-EXQ-603 / 603a / 603b / 603c were four consecutive measurement_gaps /
substrate-unblocked-but-not-yet-cleared FAILs. 603c was the P0/P1 phased-
training fix; 8/12 cells aborted on the P1 survival gate because the env's
hazard-food-attraction corridor still dominated under a still-cold goal
pipeline. The scaffolded_sd054_onboarding substrate (landed 2026-05-31)
delivers a deliberate anneal of (hazard_food_attraction, proximity_harm_scale,
mech295_min_drive_to_fire, mech307_conjunction_z_beta_threshold) across the
P1 window so the policy survives P1 into P2 with a goal pipeline that
actually fires.

603d uses ScaffoldedSD054OnboardingScheduler.run_p0 + run_p1 in place of
603c's hand-rolled P0/P1 loops; the P2 measurement loop is bespoke because
the scheduler's P2 records substrate-readiness telemetry (z_goal_norm_peak,
approach_commit_rate, bridge_cue_fires, dacc_bias_nonzero_steps) but not
the action-class entropy that Q-045 turns on. The bespoke P2 layers entropy,
position-entropy, FIFO temporal gate, and reef_visit_fraction on top of the
scheduler's substrate-readiness diagnostics.

Arms (4-arm Q-045 grid + 1 FP-2 matched-noise dissociation arm)
---------------------------------------------------------------
ARM_0_both_off       -- control. No MECH-313, no MECH-260. baseline T=1.0
                        passed to e3.select. No noise_floor regulator.
ARM_1_mech313_only   -- use_noise_floor=True, alpha=0.5. MECH-313 LC-NE
                        tonic complement; effective T = max(1.0+0.5, 1.0)
                        = 1.5 via NoiseFloor regulator.
ARM_2_mech260_only   -- use_dacc=True with dacc_suppression_weight=0.5,
                        dacc_suppression_memory=8. MECH-260 anti-recency
                        FIFO suppresses just-executed action classes.
ARM_3_both_on        -- MECH-313 + MECH-260 jointly active.
ARM_4_matched_noise  -- FP-2 control. No MECH-313 substrate, no MECH-260.
                        e3.select called with explicit temperature=1.5
                        (magnitude-matched to ARM_1/ARM_3 NoiseFloor lift)
                        for the *full P2 measurement window*. The arm tests
                        whether MECH-313's regulator architecture produces
                        diversity beyond what an equivalent-magnitude
                        constant temperature lift delivers.

Why FP-2 matched-noise matters
------------------------------
Theory 3 (MECH-313 LC-NE tonic noise floor) under R3.a / R3.c per the
isolation plan needs to dissociate "MECH-313 substrate produces diversity"
from "any non-zero entropy at this magnitude produces diversity". A
mid-or-low-confidence ARM_3 PASS would survive on entropy alone; the
FP-2 gate forces structured-vs-noise comparison. If ARM_3 selection entropy
and downstream behavioural diversity (reef_visit_fraction, position
entropy) are within FP2_MARGIN of ARM_4 matched-noise, MECH-313 is not a
load-bearing diversity substrate at this magnitude -- it's a uniform
temperature lift with extra mechanism.

Pre-registered acceptance (partial 7-criterion gate revision per 591 autopsy)
-----------------------------------------------------------------------------
The V3-EXQ-591 7-criterion gate (infant_substrate_expansion.md section 8) is
preserved selectively per failure_autopsy_V3-EXQ-591_2026-05-27 section 7.
Dropped criteria: C3 residue_coverage_pct (trivially saturating); C5 harm/
benefit ratio + C6 post-sleep retention + C7 traj cosine (sentinel-emitting
in current substrate). Kept criteria: C1 z_goal_norm > 0.4 (substrate
engagement); C2 rolling H_pos > 0.65*ln(144) ~= 3.23 (position-coverage
discrimination); C4 action-class entropy lift (Q-045 primary signal).

Mapped onto the 5-arm Q-045 design:

  C1 (Q-045 entropy)        ARM_3 entropy > ARM_0 + ENTROPY_MARGIN
  C2 (Q-045 mutually LB)    ARM_3 entropy > max(ARM_1, ARM_2) + ENTROPY_MARGIN
  C3 (Q-045 each-alone)     ARM_1 entropy > ARM_0 AND ARM_2 entropy > ARM_0
  C4 (substrate engaged)    z_goal_norm_peak_max(ARM_3) > Z_GOAL_FLOOR (0.4)
  C5 (position coverage)    rolling_h_pos_mean(ARM_3) > H_POS_FLOOR (3.23)
  FP2 (dissociation)        ARM_3 entropy > ARM_4 entropy + FP2_MARGIN AND
                            |ARM_3 reef_visit_fraction - ARM_4 reef_visit_fraction|
                                > FP2_BEHAVIOURAL_MARGIN

PASS = C2 AND C4 AND C5 AND FP2.

NOTE: C3/C5/C6/C7 from the 591 7-criterion list are deliberately NOT
implemented; the renumbered C3 above is the 603c each-alone-beats-off
criterion, distinct from the 591 C3 residue_coverage criterion that was
dropped.

4-row interpretation grid
-------------------------
+------------------------+--------------------------------------------+
| Outcome                | Routing                                    |
+------------------------+--------------------------------------------+
| PASS                   | -> /governance MECH-313 + MECH-260         |
| (C2+C4+C5+FP2 all true)| promotion review; Q-045 supports; close    |
|                        | behavioral_diversity_isolation:GAP-C       |
|                        | Theory 3 node done.                        |
+------------------------+--------------------------------------------+
| PARTIAL_NO_FP2         | -> /governance MECH-313 candidate retains; |
| (C2+C4+C5 true,        | Q-045 mixed; raise structured-vs-noise     |
| FP2 false)             | follow-up; MECH-313 reduced to "uniform    |
|                        | temperature lift at this magnitude" --     |
|                        | candidate for collapse with ARC-064        |
|                        | softmax-temperature claim cluster.         |
+------------------------+--------------------------------------------+
| SUBSTRATE_FAILURE      | -> /failure-autopsy on V3-EXQ-603d;        |
| (C4 false OR C5 false) | substrate-readiness gates failed despite   |
|                        | the scaffolded_sd054_onboarding scheduler. |
|                        | Routes to substrate enrichment review;     |
|                        | possible scheduler-budget or env-config    |
|                        | revisit. NOT a substrate-conditional V4    |
|                        | escalation -- this is a recoverable        |
|                        | calibration miss.                          |
+------------------------+--------------------------------------------+
| FAIL_NO_DIVERSITY      | -> /governance MECH-313 weakens; MECH-260  |
| (C2 false but C4+C5    | weakens; Q-045 weakens. Q-045's "MECH-313  |
| true)                  | + MECH-260 are both load-bearing for       |
|                        | behavioural diversity" hypothesis directly |
|                        | falsified on a working substrate.          |
+------------------------+--------------------------------------------+

claim_ids: Q-045, MECH-313, MECH-260
experiment_purpose: evidence

Smoke
-----
    /opt/local/bin/python3 experiments/v3_exq_603e_q045_mech313_mech260_scaffolded_sd054.py --dry-run

SLEEP DRIVER: N/A (no sleep cycles fired; use_sleep_loop=False).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiment_protocol import emit_outcome  # noqa: E402
from experiments.scaffolded_sd054_onboarding import (  # noqa: E402
    ScaffoldedSD054OnboardingConfig,
    ScaffoldedSD054OnboardingScheduler,
    _benefit_and_drive,
)
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402


EXPERIMENT_TYPE = "v3_exq_603e_q045_mech313_mech260_scaffolded_sd054"
QUEUE_ID = "V3-EXQ-603e"
CLAIM_IDS = ["Q-045", "MECH-313", "MECH-260"]
# Substrate-readiness validation of the 2026-06-02 update_z_goal-wiring AMEND;
# not governance evidence (excluded from confidence/conflict scoring).
EXPERIMENT_PURPOSE = "diagnostic"
EXPERIMENT_PURPOSE_TAG = "diagnostic"

SEEDS = [42, 43, 44]

# P2 measurement budget (preserved from 603c: 30 ep x 500 steps with
# FIFO_WARMUP_STEPS=75 before entropy counters start).
EVAL_EPISODES = 30
STEPS_PER_EPISODE = 500
FIFO_WARMUP_STEPS = 75

# Scheduler P0 / P1 budgets. RESTORED to the 603c 100/50 budget for 603e:
# 603d's reduced 30/30 confounded the P1 survival gate (failure_autopsy_
# V3-EXQ-603d_2026-06-01: P1 survival failed 2/3 seeds at 30/30, confounded
# by both the reduced budget AND the then-inert goal pipeline). With
# update_z_goal now wired + z_goal_enabled=True, the restored budget gives
# the policy enough P1 to clear the survival gate with a live goal drive.
SCAFFOLD_P0_BUDGET = 100
SCAFFOLD_P1_BUDGET = 50
SCAFFOLD_TRAIN_STEPS = 200

# Dry-run miniature so the full pipeline (P0 -> P1 -> P2) is exercised
# without burning runtime. 1 arm x 1 seed, tiny budgets.
DRY_RUN_SEEDS = [42]
DRY_RUN_P0_BUDGET = 2
DRY_RUN_P1_BUDGET = 2
DRY_RUN_P2_EPISODES = 1
DRY_RUN_STEPS = 30

# Acceptance thresholds (pre-registered).
ENTROPY_MARGIN = 0.05               # C1 / C2 / FP2 margin on selection entropy
FP2_BEHAVIOURAL_MARGIN = 0.02       # FP2 behavioural margin on reef_visit_fraction
Z_GOAL_FLOOR = 0.4                  # C4 z_goal_norm_peak_max threshold (591 7-criterion C1)
H_POS_FLOOR = 0.65 * math.log(144)  # C5 rolling H_pos floor (591 7-criterion C2; ~3.23)
H_POS_ROLLING_WINDOW = 10           # rolling-mean window for H_pos

# Substrate knobs
DACC_SUPPRESSION_WEIGHT = 0.5
DACC_SUPPRESSION_MEMORY = 8
WORLD_DIM = 32
HARM_A_DIM = 16
HARM_HISTORY_LEN = 10
# SD-022 limb_damage_enabled=True (set by scaffolded_sd054_onboarding scheduler)
# reshapes harm_obs_a from the legacy 50-dim EMA path to a 7-dim body-damage
# vector (damage[4] + max + mean + residual_pain). Encoder input dim must
# match the env-exposed shape.
HARM_OBS_A_DIM = 7
NOISE_FLOOR_ALPHA = 0.5
MATCHED_NOISE_TEMPERATURE = 1.0 + NOISE_FLOOR_ALPHA  # ARM_4 constant T lift

# Total expected runs counter (for runner ETA + verdict line accounting)
TOTAL_RUNS = 0  # set in run_experiment from len(arms) * len(seeds)


ARMS: List[Dict[str, Any]] = [
    {
        "arm": "ARM_0_both_off",
        "use_noise_floor": False,
        "noise_floor_alpha": 0.1,
        "use_dacc": False,
        "matched_noise": False,
    },
    {
        "arm": "ARM_1_mech313_only",
        "use_noise_floor": True,
        "noise_floor_alpha": NOISE_FLOOR_ALPHA,
        "use_dacc": False,
        "matched_noise": False,
    },
    {
        "arm": "ARM_2_mech260_only",
        "use_noise_floor": False,
        "noise_floor_alpha": 0.1,
        "use_dacc": True,
        "matched_noise": False,
    },
    {
        "arm": "ARM_3_both_on",
        "use_noise_floor": True,
        "noise_floor_alpha": NOISE_FLOOR_ALPHA,
        "use_dacc": True,
        "matched_noise": False,
    },
    {
        "arm": "ARM_4_matched_noise",
        "use_noise_floor": False,
        "noise_floor_alpha": 0.1,
        "use_dacc": False,
        "matched_noise": True,
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _entropy(counts: Counter) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    h = 0.0
    for c in counts.values():
        p = c / total
        if p > 0.0:
            h -= p * math.log(p)
    return float(h)


def _make_config(env: CausalGridWorldV2, arm: Dict[str, Any]) -> REEConfig:
    return REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=32,
        world_dim=WORLD_DIM,
        alpha_world=0.9,
        use_affective_harm_stream=True,
        z_harm_a_dim=HARM_A_DIM,
        harm_obs_a_dim=HARM_OBS_A_DIM,
        harm_history_len=HARM_HISTORY_LEN,
        use_gated_policy=True,
        gated_policy_use_first_action_onehot=True,
        use_support_preserving_cem=True,
        support_preserving_stratified_elites=True,
        support_preserving_ao_std_floor=0.2,
        support_preserving_min_first_action_classes=2,
        use_noise_floor=arm["use_noise_floor"],
        noise_floor_alpha=arm["noise_floor_alpha"],
        use_dacc=arm["use_dacc"],
        dacc_weight=(1.0 if arm["use_dacc"] else 0.0),
        dacc_suppression_weight=(
            DACC_SUPPRESSION_WEIGHT if arm["use_dacc"] else 0.0
        ),
        dacc_suppression_memory=DACC_SUPPRESSION_MEMORY,
        use_structured_curiosity=False,
        # 603e config fix: 603d omitted z_goal_enabled, so agent.goal_state was
        # None and update_z_goal early-returned (the second half of the 603d
        # harness/wiring root cause). z_goal_enabled=True creates the GoalState;
        # drive_weight=2.0 is the SD-012 amplification the reference V3-EXQ-622
        # uses. Without these the scheduler/P2 update_z_goal calls are inert.
        z_goal_enabled=True,
        drive_weight=2.0,
        # Goal-pipeline substrate engaged via scaffolded scheduler P1 anneal.
        use_mech295_liking_bridge=True,
        use_mech307_conjunction=True,
    )


def _make_scaffold_cfg() -> ScaffoldedSD054OnboardingConfig:
    return ScaffoldedSD054OnboardingConfig(
        use_scaffolded_sd054_onboarding_scheduler=True,
        scaffold_p0_episode_budget=SCAFFOLD_P0_BUDGET,
        scaffold_p1_episode_budget=SCAFFOLD_P1_BUDGET,
        scaffold_p2_episode_budget=1,  # P2 owned by this script, not scheduler
        scaffold_steps_per_episode=SCAFFOLD_TRAIN_STEPS,
    )


def _target_env_kwargs(scaffold_cfg: ScaffoldedSD054OnboardingConfig) -> Dict[str, Any]:
    """P2 measurement env matches the scheduler's P2 config so the FP-2 gate
    compares ARM_3 vs ARM_4 on identical env geometry."""
    return dict(
        size=scaffold_cfg.scaffold_env_size,
        num_hazards=scaffold_cfg.scaffold_p2_num_hazards,
        num_resources=scaffold_cfg.scaffold_p2_num_resources,
        hazard_harm=0.02,
        hazard_food_attraction=scaffold_cfg.scaffold_p2_hazard_food_attraction,
        proximity_harm_scale=scaffold_cfg.scaffold_p2_proximity_harm_scale,
        limb_damage_enabled=True,
        reef_enabled=True,
        reef_bipartite_layout=True,
        reef_bipartite_axis=scaffold_cfg.scaffold_reef_bipartite_axis,
        reef_bipartite_agent_band_radius=scaffold_cfg.scaffold_reef_bipartite_agent_band_radius,
        reef_bipartite_agent_spawn_in_reef_half=False,
        harm_history_len=HARM_HISTORY_LEN,
        resource_respawn_on_consume=True,
    )


def _obs_harm_a(obs_dict: Dict[str, Any]) -> Optional[torch.Tensor]:
    ha = obs_dict.get("harm_obs_a")
    if ha is None:
        return None
    t = ha.float()
    return t.unsqueeze(0) if t.dim() == 1 else t


def _obs_harm_history(obs_dict: Dict[str, Any]) -> Optional[torch.Tensor]:
    hh = obs_dict.get("harm_history")
    if hh is None:
        return None
    t = hh.float()
    return t.unsqueeze(0) if t.dim() == 1 else t


def _select_action_with_harm(
    agent: REEAgent,
    obs_body: torch.Tensor,
    obs_world: torch.Tensor,
    obs_harm_a: Optional[torch.Tensor],
    obs_harm_history: Optional[torch.Tensor],
    temperature: float,
):
    """sense + ticks + select_action path threading obs_harm_a/history and
    passing the per-arm temperature to e3.select via select_action."""
    latent_state = agent.sense(
        obs_body,
        obs_world,
        obs_harm_a=obs_harm_a,
        obs_harm_history=obs_harm_history,
    )
    ticks = agent.clock.advance()
    if ticks.get("e1_tick", False):
        e1_prior = agent._e1_tick(latent_state)
    else:
        e1_prior = torch.zeros(1, WORLD_DIM, device=agent.device)
    candidates = agent.generate_trajectories(latent_state, e1_prior, ticks)
    action = agent.select_action(candidates, ticks, temperature=temperature)
    if ticks.get("e3_quiescent", False):
        agent._do_replay(latent_state)
    agent._step_count += 1
    return action, latent_state


def _dacc_diagnostics(agent: REEAgent) -> Dict[str, Any]:
    if agent.dacc is None:
        return {
            "dacc_forward_calls": 0,
            "dacc_history_len": 0,
            "dacc_max_suppression": 0.0,
        }
    hist_len = len(agent.dacc._action_history)
    max_sup = 0.0
    bundle = getattr(agent, "_dacc_last_bundle", None)
    if bundle is not None and "suppression" in bundle:
        sup = bundle["suppression"]
        if isinstance(sup, torch.Tensor) and sup.numel() > 0:
            max_sup = float(sup.max().item())
    return {
        "dacc_forward_calls": int(agent.dacc._n_forward_calls),
        "dacc_history_len": hist_len,
        "dacc_max_suppression": round(max_sup, 6),
    }


def _z_goal_norm(agent: REEAgent) -> float:
    gs = getattr(agent, "goal_state", None)
    if gs is None:
        return 0.0
    if hasattr(gs, "goal_norm"):
        try:
            return float(gs.goal_norm())
        except TypeError:
            return float(gs.goal_norm)
    return 0.0


# ---------------------------------------------------------------------------
# P2 measurement
# ---------------------------------------------------------------------------

def _run_p2_measurement(
    agent: REEAgent,
    env: CausalGridWorldV2,
    arm: Dict[str, Any],
    seed: int,
    episodes: int,
    steps_per_episode: int,
    fifo_warmup_steps: int,
) -> Dict[str, Any]:
    """Frozen-policy P2: action-class entropy + position H_pos + z_goal_norm
    peak + reef_visit_fraction + dACC diagnostics + bridge_cue counter."""
    agent.eval()
    action_counts: Counter = Counter()
    position_counts: Counter = Counter()
    reef_steps = 0
    total_steps = 0
    measured_steps = 0
    reef_cells = set()
    max_dacc_history = 0
    max_dacc_forward = 0
    max_dacc_suppression = 0.0
    z_goal_norm_peak = 0.0
    rolling_h_pos_window: Deque[float] = deque(maxlen=H_POS_ROLLING_WINDOW)
    h_pos_rolling_log: List[float] = []
    bridge_cue_baseline = 0
    bridge = getattr(agent, "mech295_bridge", None)
    if bridge is not None:
        bridge_cue_baseline = int(getattr(bridge, "_n_cue_fires", 0))

    # Per-arm action-selection temperature. ARM_4 MATCHED_NOISE bypasses the
    # NoiseFloor substrate (which is OFF for that arm) by passing the matched
    # constant temperature directly to e3.select. All other arms use baseline
    # T=1.0; the NoiseFloor regulator (when use_noise_floor=True) adds alpha
    # to the kwarg internally so the effective T matches.
    if arm["matched_noise"]:
        temperature = MATCHED_NOISE_TEMPERATURE
    else:
        temperature = 1.0

    with torch.no_grad():
        for ep in range(episodes):
            _, obs_dict = env.reset()
            agent.reset()
            if ep == 0:
                reef_cells = set(getattr(env, "_reef_cells", set()))

            ep_position_counts: Counter = Counter()
            for step_in_ep in range(steps_per_episode):
                obs_body = obs_dict["body_state"]
                obs_world = obs_dict["world_state"]
                if obs_body.dim() == 1:
                    obs_body = obs_body.unsqueeze(0)
                if obs_world.dim() == 1:
                    obs_world = obs_world.unsqueeze(0)

                action, _ = _select_action_with_harm(
                    agent,
                    obs_body,
                    obs_world,
                    _obs_harm_a(obs_dict),
                    _obs_harm_history(obs_dict),
                    temperature=temperature,
                )

                # Track z_goal_norm peak (C4 substrate-engagement criterion).
                zg = _z_goal_norm(agent)
                if zg > z_goal_norm_peak:
                    z_goal_norm_peak = zg

                if arm["use_dacc"] and agent.dacc is not None:
                    diag = _dacc_diagnostics(agent)
                    max_dacc_history = max(max_dacc_history, diag["dacc_history_len"])
                    max_dacc_forward = max(max_dacc_forward, diag["dacc_forward_calls"])
                    max_dacc_suppression = max(
                        max_dacc_suppression, diag["dacc_max_suppression"]
                    )

                if action is None:
                    idx = random.randint(0, env.action_dim - 1)
                    action_onehot = torch.zeros(1, env.action_dim, device=agent.device)
                    action_onehot[0, idx] = 1.0
                else:
                    action_onehot = action
                    idx = int(action.argmax(dim=-1).item())

                pos = (int(env.agent_x), int(env.agent_y))
                if step_in_ep >= fifo_warmup_steps:
                    action_counts[idx] += 1
                    position_counts[pos] += 1
                    ep_position_counts[pos] += 1
                    measured_steps += 1

                if pos in reef_cells:
                    reef_steps += 1

                _flat, _harm, done, _info, obs_dict = env.step(action_onehot)
                total_steps += 1

                # 603e P2 seeding fix: the bespoke P2 loop (unlike the scheduler's
                # run_p1, which now seeds) must drive z_goal itself for the C4
                # z_goal_norm_peak metric to reflect a live goal pipeline. Mirrors
                # the scheduler _eval_episode fix + goal_stream_stages_sd054 ref
                # runner: seed from the post-step body-state every step.
                benefit, drive = _benefit_and_drive(obs_dict["body_state"].to(agent.device))
                agent.update_z_goal(benefit_exposure=benefit, drive_level=drive)
                zg_post = _z_goal_norm(agent)
                if zg_post > z_goal_norm_peak:
                    z_goal_norm_peak = zg_post

                if done:
                    break

            ep_h_pos = _entropy(ep_position_counts)
            rolling_h_pos_window.append(ep_h_pos)
            if rolling_h_pos_window:
                rolling_mean = float(np.mean(list(rolling_h_pos_window)))
                h_pos_rolling_log.append(round(rolling_mean, 4))

            if (ep + 1) % 10 == 0 or (ep + 1) == episodes:
                print(
                    f"  [P2 eval] arm={arm['arm']} seed={seed} ep {ep + 1}/{episodes}",
                    flush=True,
                )

    bridge_cue_fires = 0
    if bridge is not None:
        bridge_cue_fires = int(getattr(bridge, "_n_cue_fires", 0)) - bridge_cue_baseline

    selection_entropy = _entropy(action_counts)
    position_entropy = _entropy(position_counts)
    reef_fraction = reef_steps / max(total_steps, 1)
    steps_per_ep_measured = measured_steps / max(episodes, 1)
    fifo_gate_ok = (
        fifo_warmup_steps >= 2 * DACC_SUPPRESSION_MEMORY
        and steps_per_ep_measured >= (steps_per_episode - fifo_warmup_steps) * 0.9
    )
    rolling_h_pos_mean = (
        float(np.mean(h_pos_rolling_log[-H_POS_ROLLING_WINDOW:]))
        if h_pos_rolling_log
        else 0.0
    )

    out: Dict[str, Any] = {
        "arm": arm["arm"],
        "seed": seed,
        "selected_action_entropy": round(selection_entropy, 6),
        "position_entropy": round(position_entropy, 6),
        "rolling_h_pos_mean": round(rolling_h_pos_mean, 6),
        "rolling_h_pos_log_tail": h_pos_rolling_log[-H_POS_ROLLING_WINDOW:],
        "z_goal_norm_peak": round(z_goal_norm_peak, 6),
        "reef_fraction": round(reef_fraction, 6),
        "bridge_cue_fires": int(bridge_cue_fires),
        "total_steps": int(total_steps),
        "measured_steps": int(measured_steps),
        "fifo_warmup_steps": int(fifo_warmup_steps),
        "steps_per_episode_measured_mean": round(steps_per_ep_measured, 2),
        "fifo_temporal_gate_ok": bool(fifo_gate_ok),
        "unique_actions": len(action_counts),
        "selection_temperature_p2": float(temperature),
        "p2_run": True,
    }
    if arm["use_dacc"]:
        out.update(
            {
                "dacc_history_len_max": max_dacc_history,
                "dacc_forward_calls_max": max_dacc_forward,
                "dacc_max_suppression": round(max_dacc_suppression, 6),
                "mech260_operative": bool(
                    max_dacc_forward > 0 and max_dacc_history > 0
                ),
            }
        )
    return out


def _empty_cell_row(
    arm: Dict[str, Any],
    seed: int,
    abort_phase: str,
    abort_reason: str,
) -> Dict[str, Any]:
    """Row stub for cells that aborted at scheduler P0 or P1."""
    row: Dict[str, Any] = {
        "arm": arm["arm"],
        "seed": seed,
        "selected_action_entropy": 0.0,
        "position_entropy": 0.0,
        "rolling_h_pos_mean": 0.0,
        "rolling_h_pos_log_tail": [],
        "z_goal_norm_peak": 0.0,
        "reef_fraction": 0.0,
        "bridge_cue_fires": 0,
        "total_steps": 0,
        "measured_steps": 0,
        "fifo_warmup_steps": int(FIFO_WARMUP_STEPS),
        "steps_per_episode_measured_mean": 0.0,
        "fifo_temporal_gate_ok": False,
        "unique_actions": 0,
        "selection_temperature_p2": 0.0,
        "p2_run": False,
        "abort_phase": abort_phase,
        "abort_reason": abort_reason,
    }
    if arm["use_dacc"]:
        row.update(
            {
                "dacc_history_len_max": 0,
                "dacc_forward_calls_max": 0,
                "dacc_max_suppression": 0.0,
                "mech260_operative": False,
            }
        )
    return row


# ---------------------------------------------------------------------------
# Per-cell pipeline (Scheduler P0 -> Scheduler P1 -> bespoke P2)
# ---------------------------------------------------------------------------

def _run_arm_seed(
    arm: Dict[str, Any],
    seed: int,
    scaffold_p0_budget: int,
    scaffold_p1_budget: int,
    train_steps_per_episode: int,
    p2_episodes: int,
    p2_steps_per_episode: int,
) -> Dict[str, Any]:
    """Scaffolded P0 -> Scaffolded P1 -> bespoke P2 pipeline for one cell."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    scaffold_cfg = _make_scaffold_cfg()
    scaffold_cfg.scaffold_p0_episode_budget = scaffold_p0_budget
    scaffold_cfg.scaffold_p1_episode_budget = scaffold_p1_budget
    scaffold_cfg.scaffold_steps_per_episode = train_steps_per_episode
    # Dry-run scaling: at train_steps_per_episode < default 200, the default
    # survival_gate (75) is structurally unreachable. Scale the gate to
    # train_steps/4 (capped at 75) so dry-run exercises P2 instead of
    # aborting at P1.
    if train_steps_per_episode < 75:
        scaffold_cfg.scaffold_p1_survival_gate_steps = max(
            1, train_steps_per_episode // 4
        )

    # Build the target-env config first so the agent's REEConfig has the right
    # body / world / action dims for the scaffolded scheduler's P2 env.
    target_env = CausalGridWorldV2(seed=seed, **_target_env_kwargs(scaffold_cfg))
    cfg = _make_config(target_env, arm)
    agent = REEAgent(cfg)
    device = torch.device("cpu")
    scheduler = ScaffoldedSD054OnboardingScheduler(scaffold_cfg)

    cell_diagnostics: Dict[str, Any] = {}

    total_episodes_per_run = scaffold_p0_budget + scaffold_p1_budget + p2_episodes

    # Scheduler P0
    p0 = scheduler.run_p0(agent, device)
    cell_diagnostics["scaffold_p0"] = {
        "n_episodes": int(p0.n_episodes),
        "mean_episode_length": round(float(p0.mean_episode_length), 2),
        "final_running_variance": round(float(p0.final_running_variance), 6),
        "aborted": bool(p0.aborted),
        "abort_reason": str(p0.abort_reason),
    }
    print(
        f"  [train] scaffold_p0 arm={arm['arm']} seed={seed}"
        f" ep {p0.n_episodes}/{total_episodes_per_run}"
        f" mean_len={p0.mean_episode_length:.1f}"
        f" rv={p0.final_running_variance:.5f}",
        flush=True,
    )
    if p0.aborted:
        print(
            f"verdict: FAIL arm={arm['arm']} seed={seed} aborted_at=scaffold_p0"
            f" reason={p0.abort_reason}",
            flush=True,
        )
        row = _empty_cell_row(arm, seed, "scaffold_p0", p0.abort_reason)
        row["cell_diagnostics"] = cell_diagnostics
        return row

    # Scheduler P1
    p1 = scheduler.run_p1(agent, device)
    cell_diagnostics["scaffold_p1"] = {
        "n_episodes": int(p1.n_episodes),
        "median_last_window_episode_length": round(
            float(p1.median_last_window_episode_length), 2
        ),
        "survival_gate_passed": bool(p1.survival_gate_passed),
        "final_hazard_food_attraction": round(
            float(p1.final_hazard_food_attraction), 4
        ),
        "final_mech295_min_drive_to_fire": round(
            float(p1.final_mech295_min_drive_to_fire), 4
        ),
        "final_mech307_conjunction_z_beta_threshold": round(
            float(p1.final_mech307_conjunction_z_beta_threshold), 4
        ),
        "aborted": bool(p1.aborted),
        "abort_reason": str(p1.abort_reason),
        "p1_survival_gate_steps": int(scaffold_cfg.scaffold_p1_survival_gate_steps),
    }
    cumulative_after_p1 = p0.n_episodes + p1.n_episodes
    print(
        f"  [train] scaffold_p1 arm={arm['arm']} seed={seed}"
        f" ep {cumulative_after_p1}/{total_episodes_per_run}"
        f" median_last={p1.median_last_window_episode_length:.1f}"
        f" survival_gate={'pass' if p1.survival_gate_passed else 'fail'}",
        flush=True,
    )
    if not p1.survival_gate_passed or p1.aborted:
        print(
            f"verdict: FAIL arm={arm['arm']} seed={seed} aborted_at=scaffold_p1"
            f" reason={p1.abort_reason or 'p1_survival_gate_failed'}",
            flush=True,
        )
        row = _empty_cell_row(arm, seed, "scaffold_p1", p1.abort_reason or "p1_survival_gate_failed")
        row["cell_diagnostics"] = cell_diagnostics
        return row

    # Bespoke P2 measurement on a fresh target env (so the episode RNG sequence
    # matches the canonical 603-lineage measurement-loop pattern).
    p2_env = CausalGridWorldV2(seed=seed, **_target_env_kwargs(scaffold_cfg))
    row = _run_p2_measurement(
        agent, p2_env, arm, seed,
        episodes=p2_episodes,
        steps_per_episode=p2_steps_per_episode,
        fifo_warmup_steps=min(FIFO_WARMUP_STEPS, max(0, p2_steps_per_episode - 1)),
    )
    row["cell_diagnostics"] = cell_diagnostics
    print(f"verdict: PASS arm={arm['arm']} seed={seed}", flush=True)
    return row


# ---------------------------------------------------------------------------
# Acceptance evaluation + per-claim direction
# ---------------------------------------------------------------------------

def _evaluate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_arm: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        by_arm.setdefault(row["arm"], []).append(row)

    def mean_metric(arm_name: str, key: str) -> float:
        cells = by_arm.get(arm_name, [])
        vals = [r[key] for r in cells if r.get("p2_run", False)]
        if not vals:
            return 0.0
        return float(sum(vals) / len(vals))

    def max_metric(arm_name: str, key: str) -> float:
        cells = by_arm.get(arm_name, [])
        vals = [r[key] for r in cells if r.get("p2_run", False)]
        if not vals:
            return 0.0
        return float(max(vals))

    e0 = mean_metric("ARM_0_both_off", "selected_action_entropy")
    e1 = mean_metric("ARM_1_mech313_only", "selected_action_entropy")
    e2 = mean_metric("ARM_2_mech260_only", "selected_action_entropy")
    e3 = mean_metric("ARM_3_both_on", "selected_action_entropy")
    e4 = mean_metric("ARM_4_matched_noise", "selected_action_entropy")

    r3_reef = mean_metric("ARM_3_both_on", "reef_fraction")
    r4_reef = mean_metric("ARM_4_matched_noise", "reef_fraction")

    z_goal_arm3_max = max_metric("ARM_3_both_on", "z_goal_norm_peak")
    h_pos_arm3_mean = mean_metric("ARM_3_both_on", "rolling_h_pos_mean")

    # Acceptance criteria (per the 4-row interpretation grid above).
    c1_q045_both_beats_off = e3 > e0 + ENTROPY_MARGIN
    c2_q045_mutually_load_bearing = e3 > max(e1, e2) + ENTROPY_MARGIN
    c3_q045_each_alone_beats_off = (e1 > e0) and (e2 > e0)
    c4_z_goal_engaged = z_goal_arm3_max > Z_GOAL_FLOOR
    c5_h_pos_coverage = h_pos_arm3_mean > H_POS_FLOOR
    fp2_dissociation = (
        (e3 > e4 + ENTROPY_MARGIN)
        and (abs(r3_reef - r4_reef) > FP2_BEHAVIOURAL_MARGIN)
    )

    mech260_rows = [
        r for r in rows
        if r["arm"] in ("ARM_2_mech260_only", "ARM_3_both_on")
        and r.get("p2_run", False)
    ]
    mech260_operative_all = bool(mech260_rows) and all(
        r.get("mech260_operative", False) for r in mech260_rows
    )

    p2_rows = [r for r in rows if r.get("p2_run", False)]
    fifo_ok_all = bool(p2_rows) and all(
        r.get("fifo_temporal_gate_ok", False) for r in p2_rows
    )

    p2_cell_count = len(p2_rows)
    aborted_cells = len(rows) - p2_cell_count

    overall_pass = bool(
        c2_q045_mutually_load_bearing
        and c4_z_goal_engaged
        and c5_h_pos_coverage
        and fp2_dissociation
        and p2_cell_count == len(rows)
    )

    return {
        "entropy_ARM_0": round(e0, 6),
        "entropy_ARM_1": round(e1, 6),
        "entropy_ARM_2": round(e2, 6),
        "entropy_ARM_3": round(e3, 6),
        "entropy_ARM_4_matched_noise": round(e4, 6),
        "reef_fraction_ARM_3": round(r3_reef, 6),
        "reef_fraction_ARM_4_matched_noise": round(r4_reef, 6),
        "z_goal_norm_peak_ARM_3_max": round(z_goal_arm3_max, 6),
        "rolling_h_pos_mean_ARM_3": round(h_pos_arm3_mean, 6),
        "c1_q045_both_beats_off": bool(c1_q045_both_beats_off),
        "c2_q045_mutually_load_bearing": bool(c2_q045_mutually_load_bearing),
        "c3_q045_each_alone_beats_off": bool(c3_q045_each_alone_beats_off),
        "c4_z_goal_engaged": bool(c4_z_goal_engaged),
        "c5_h_pos_coverage": bool(c5_h_pos_coverage),
        "fp2_dissociation": bool(fp2_dissociation),
        "z_goal_floor": Z_GOAL_FLOOR,
        "h_pos_floor": round(H_POS_FLOOR, 6),
        "entropy_margin": ENTROPY_MARGIN,
        "fp2_behavioural_margin": FP2_BEHAVIOURAL_MARGIN,
        "mech260_operative_all_seeds": bool(mech260_operative_all),
        "fifo_temporal_gate_ok_all": bool(fifo_ok_all),
        "p2_cell_count": int(p2_cell_count),
        "aborted_cells": int(aborted_cells),
        "overall_pass": overall_pass,
    }


def _interpretation_label(summary: Dict[str, Any]) -> str:
    """Map summary -> 4-row interpretation grid label."""
    if not summary["c4_z_goal_engaged"] or not summary["c5_h_pos_coverage"]:
        return "SUBSTRATE_FAILURE"
    if summary["overall_pass"]:
        return "PASS"
    if (
        summary["c2_q045_mutually_load_bearing"]
        and summary["c4_z_goal_engaged"]
        and summary["c5_h_pos_coverage"]
        and not summary["fp2_dissociation"]
    ):
        return "PARTIAL_NO_FP2"
    return "FAIL_NO_DIVERSITY"


def _evidence_direction_per_claim(summary: Dict[str, Any]) -> Dict[str, str]:
    """Per-claim direction per the 4-row interpretation grid.

    Routes non_contributory when the matrix is structurally underpowered
    (P2 cell count below half) -- mirrors 603c semantics.
    """
    total_cells = 5 * 3  # 5 arms * 3 seeds
    if summary["p2_cell_count"] < total_cells // 2:
        return {
            "Q-045": "non_contributory",
            "MECH-313": "non_contributory",
            "MECH-260": "non_contributory",
        }

    label = _interpretation_label(summary)
    e0 = summary["entropy_ARM_0"]
    e1 = summary["entropy_ARM_1"]
    e2 = summary["entropy_ARM_2"]

    if label == "PASS":
        q045 = "supports"
        m313 = "supports" if e1 > e0 + ENTROPY_MARGIN else "mixed"
        m260 = "supports" if e2 > e0 + ENTROPY_MARGIN else "mixed"
    elif label == "PARTIAL_NO_FP2":
        # Substrate works but MECH-313 not separable from matched noise.
        # Per the routing grid, MECH-313 reduced to "uniform temperature lift"
        # -- weakens MECH-313 specifically; MECH-260 still supports if e2 > e0.
        q045 = "mixed"
        m313 = "weakens"
        m260 = "supports" if e2 > e0 + ENTROPY_MARGIN else "mixed"
    elif label == "SUBSTRATE_FAILURE":
        # Substrate did not engage -- result non-contributory on all three.
        q045 = "non_contributory"
        m313 = "non_contributory"
        m260 = "non_contributory"
    else:  # FAIL_NO_DIVERSITY
        q045 = "weakens"
        m313 = "weakens"
        m260 = "weakens"

    if not summary.get("mech260_operative_all_seeds", True):
        m260 = "unknown"

    return {"Q-045": q045, "MECH-313": m313, "MECH-260": m260}


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------

def run_experiment(dry_run: bool = False) -> Dict[str, Any]:
    global TOTAL_RUNS
    seeds = DRY_RUN_SEEDS if dry_run else SEEDS
    scaffold_p0 = DRY_RUN_P0_BUDGET if dry_run else SCAFFOLD_P0_BUDGET
    scaffold_p1 = DRY_RUN_P1_BUDGET if dry_run else SCAFFOLD_P1_BUDGET
    p2_episodes = DRY_RUN_P2_EPISODES if dry_run else EVAL_EPISODES
    train_steps = DRY_RUN_STEPS if dry_run else SCAFFOLD_TRAIN_STEPS
    p2_steps = DRY_RUN_STEPS if dry_run else STEPS_PER_EPISODE

    arms_to_run = ARMS[:1] if dry_run else ARMS
    TOTAL_RUNS = len(arms_to_run) * len(seeds)

    rows: List[Dict[str, Any]] = []
    for arm in arms_to_run:
        for seed in seeds:
            print(f"Seed {seed} Condition {arm['arm']}", flush=True)
            cell = _run_arm_seed(
                arm, seed,
                scaffold_p0_budget=scaffold_p0,
                scaffold_p1_budget=scaffold_p1,
                train_steps_per_episode=train_steps,
                p2_episodes=p2_episodes,
                p2_steps_per_episode=p2_steps,
            )
            rows.append(cell)

    summary = _evaluate(rows)
    label = _interpretation_label(summary)
    outcome = "PASS" if summary["overall_pass"] else "FAIL"
    edpc = _evidence_direction_per_claim(summary)

    # Set overall evidence_direction to the most informative summary -- pick
    # Q-045 since it spans the whole matrix.
    evidence_direction = edpc["Q-045"]

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{timestamp}_v3"

    manifest: Dict[str, Any] = {
        "run_id": run_id,
        "queue_id": QUEUE_ID,
        "experiment_type": EXPERIMENT_TYPE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": timestamp,
        "outcome": outcome,
        "experiment_purpose": EXPERIMENT_PURPOSE_TAG,
        "claim_ids": CLAIM_IDS,
        "evidence_direction": evidence_direction,
        "evidence_direction_per_claim": edpc,
        "interpretation_label": label,
        "supersedes": "V3-EXQ-603d",
        "dry_run": dry_run,
        "acceptance_criteria": summary,
        "summary": summary,
        "arm_results": rows,
        "fifo_warmup_steps": int(min(FIFO_WARMUP_STEPS, max(0, p2_steps - 1))),
        "steps_per_episode_p2": int(p2_steps),
        "scaffold_p0_budget": int(scaffold_p0),
        "scaffold_p1_budget": int(scaffold_p1),
        "scaffold_steps_per_episode": int(train_steps),
        "p2_episodes": int(p2_episodes),
        "substrate_prereqs_cleared": {
            "mech307_default_value_recalibration": {
                "cleared_at": "2026-05-15",
                "by": "V3-EXQ-540g",
                "outcome": "PASS",
            },
            "goal_pipeline_z_goal_substrate_readiness": {
                "cleared_at": "2026-06-01T05:55Z",
                "by": "V3-EXQ-621a",
                "outcome": "PASS",
            },
            "infant_curriculum_h_pos_floor_recalibration": {
                "cleared_at": "2026-05-31",
                "by": "/implement-substrate session 20260531T123353Z",
                "outcome": "landed",
            },
        },
        "fixes_applied": [
            "Fix F (new in 603d): use ScaffoldedSD054OnboardingScheduler.run_p0"
            " + run_p1 in place of 603c's hand-rolled P0/P1 loops. The scheduler"
            " anneals (hazard_food_attraction, proximity_harm_scale,"
            " mech295_min_drive_to_fire, mech307_conjunction_z_beta_threshold)"
            " across P1 so the policy survives into P2 with a goal pipeline"
            " that actually fires. Closes GAP-C prereq (2).",
            "Fix G (new in 603d): 5th arm ARM_4_matched_noise (no MECH-313"
            " substrate, no MECH-260, but explicit selection temperature"
            f" {MATCHED_NOISE_TEMPERATURE} matched to ARM_1/ARM_3 NoiseFloor"
            " lift) for FP-2 dissociation. Tests whether MECH-313's regulator"
            " architecture produces diversity beyond an equivalent-magnitude"
            " constant temperature lift.",
            "Fix H (new in 603d, per 591 autopsy section 7): partial"
            " 7-criterion gate revision -- dropped C3 (residue_coverage"
            " trivially saturating) and C5/C6/C7 (sentinel-emitting in current"
            " substrate); added z_goal_norm_peak > 0.4 (591 C1) and rolling"
            " H_pos > 0.65*ln(144) ~= 3.23 (591 C2) as substrate-engagement"
            " gates alongside Q-045's selection-entropy comparison.",
            "Fix C (from 603c, retained via scheduler): P0 warmup on easy env"
            " + P1 consolidation on target env before P2 measurement.",
            "Fix D (from 603c, retained via scheduler): pre-P2 survival gate"
            " -- abort cell if median ep length over last stability_window <"
            " p1_survival_gate_steps. Scheduler bakes this in.",
            "Fix 1 (from 603a, retained): sense() threads obs_harm_a +"
            " obs_harm_history so z_harm_a is populated and dACC.record_action"
            " fires every tick (only on P2; scheduler's _train_episode does"
            " not thread harm_a directly -- acceptable since dACC operative"
            " check is on P2 measurement, not P1 training).",
            f"Fix 2 (from 603a, retained): FIFO_WARMUP_STEPS={FIFO_WARMUP_STEPS}"
            " before P2 entropy + position-entropy measurement.",
            "Fix 3 (from 603a, retained): evidence_direction_per_claim vs"
            " ARM_0 baseline + interpretation_label routed via 4-row grid.",
            f"Fix A (from 603b, retained): P2 STEPS_PER_EPISODE={STEPS_PER_EPISODE}.",
        ],
    }

    out_dir = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments"
    out_path = out_dir / f"{run_id}.json"
    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as fh:
            json.dump(manifest, fh, indent=2)
        print(f"Manifest written: {out_path}", flush=True)
    else:
        out_path = Path("/dev/null")
        print("Dry run -- manifest not written.", flush=True)

    print(f"Outcome: {outcome} interpretation_label={label}", flush=True)
    manifest["manifest_path"] = str(out_path)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=EXPERIMENT_TYPE)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    result = run_experiment(dry_run=args.dry_run)
    if args.dry_run:
        sys.exit(0)
    _outcome_raw = str(result.get("outcome", "FAIL")).upper()
    emit_outcome(
        outcome=_outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL",
        manifest_path=str(result.get("manifest_path", "/dev/null")),
    )
