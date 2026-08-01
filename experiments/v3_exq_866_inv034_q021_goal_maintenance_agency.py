#!/opt/local/bin/python3
"""
V3-EXQ-866 -- INV-034 / Q-021 Goal-Maintenance-Necessary-for-Agency
Discriminative Ablation

experiment_purpose: evidence

Claims: INV-034 (primary), Q-021 (companion open question -- this design
  directly implements Q-021's own pre-specified two-arm ablation: approach
  arm [z_goal attractor + Go drive ON] vs avoidance-only arm [harm gradient
  only, Go drive ablated], now run on the post-SD-012/MECH-306/SD-057
  substrate that finally lets z_goal seed reliably).

BURNED-ID CHECK (IGW-20260801-222 DESIGN.md section 1): three prior lineages
tried this shape and all failed at the identical prerequisite (z_goal never
seeded / seeding collapsed at the contact step), before the SD-012
drive-scaled seeding fix (V3-EXQ-582a, drive_floor) and the SD-057
object-bound incentive-salience substrate (V3-EXQ-514o PASS, goal_pipeline
GAP-2/GAP-7 closed 2026-06-15) landed:
  - v3_exq_085 / 085c / 085d / 085e (INV-034-tagged, 2026-03-23/24, FAIL)
  - v3_exq_072b_q021_behavioral_flatness (Q-021-tagged, 2026-04-02,
    manifest self-marked evidence_direction=superseded; both arms
    resource_visit_rate=0.0 -- degenerate, non-discriminative)
  - v3_exq_331_arc030_approach_avoidance_balance (ARC-030-tagged,
    2026-04-11, non_contributory -- JOINT benefit <= HARM_ONLY benefit on
    every seed; Go sub-channel making zero difference because SD-012/MECH-306
    were not yet effective, not a code crash)
None of these are re-run here. This is a fresh design on the now-ready
substrate, using the CURRENT validated knobs (drive_floor=0.9,
drive_ema_alpha=1.0 per MECH-306) rather than the pre-fix defaults that
killed all three prior attempts. Re-derive brake (Step 2.5b) re-checked live
2026-08-01: zero failure_autopsy_*.json targets tag INV-034, Q-021, or
ARC-030 -- brake does not fire.

API RE-VERIFICATION (2026-08-01, against live ree-v3/ree_core/ source, per
skill Step 3 -- this is what the IGW-222 draft flagged as unconfirmed):
  - `CausalGridWorldV2` IS a live convenience function
    (`ree_core/environment/causal_grid_world.py:4929`,
    `def CausalGridWorldV2(**kwargs) -> CausalGridWorld: kwargs.setdefault(
    "use_proxy_fields", True); return CausalGridWorld(**kwargs)`), not a
    class -- confirmed valid, no change needed.
  - `REEAgent.update_z_goal(self, benefit_exposure: float, drive_level:
    float = 1.0, resource_type=None)` (agent.py:9520) and
    `REEAgent.compute_drive_level(obs_body)` (agent.py:9931, staticmethod)
    both match the draft's call shape exactly -- confirmed.
  - **FIXED**: the draft read z_goal via
    `getattr(agent, "z_goal", None) or getattr(agent._current_latent,
    "z_goal", None)` -- BOTH paths are wrong. `REEAgent` has no `.z_goal`
    attribute and `LatentState` (ree_core/latent/stack.py:735) carries no
    `z_goal` field at all (only z_self/z_world/z_beta/z_theta/z_delta/
    z_harm). The live z_goal tensor is `agent.goal_state.z_goal`
    (`GoalState.z_goal` property, ree_core/goal.py:750, backed by
    `self._z_goal`). Uncaught, this would have silently zeroed C6 (the
    mechanistic z_goal-engagement check) for BOTH arms regardless of
    whether the goal pathway actually fired.
  - All `REEConfig.from_dims(...)` kwargs used
    (`benefit_eval_enabled/benefit_weight/z_goal_enabled/goal_weight/
    e1_goal_conditioned/drive_weight/drive_floor/drive_ema_alpha/
    benefit_threshold/use_incentive_token_bank/use_event_classifier/
    use_resource_proximity_head/resource_proximity_weight`) exist in the
    live signature (ree_core/utils/config.py) -- confirmed.
  - `CausalGridWorld.__init__` accepts every kwarg used (`seed/size/
    num_resources/num_hazards/hazard_harm/resource_benefit/
    resource_respawn_on_consume/proximity_harm_scale/
    proximity_benefit_scale/proximity_approach_threshold/use_proxy_fields`)
    -- confirmed. `env.reset() -> (flat_obs, obs_dict)`,
    `env.step(action) -> (flat_obs, harm_signal, done, info, obs_dict)`,
    `info["transition_type"]` in {"resource"/"resource_contact"/
    "benefit_approach"/"hazard_approach"/"agent_caused_hazard"/...} and
    `info["benefit_exposure"]` -- all confirmed live. `consummatory_act_enabled`
    defaults False, so "resource" (not "resource_contact") is the live
    auto-consume transition type this env config produces -- the draft's
    `ttype in ("benefit_approach", "resource")` resource-visit predicate is
    correct as written.
  - `REEAgent.sense/generate_trajectories/select_action/update_residue/
    record_transition/compute_prediction_loss/compute_e2_loss/
    compute_resource_proximity_loss/compute_benefit_eval_loss` signatures
    all confirmed against live source; no other changes needed.

HOLD-WEIGHTED E3 READOUT TRIAGE (validate_experiments.py advisory, 2026-08-01):
  `policy_entropy` (C5) accumulates the REALIZED action every eval-phase env
  step, which is hold-weighted (a commitment held over N ticks repeats one
  action N times) -- correct for the behavioural claim INV-034 actually makes
  ("the minimum-action policy that causes no harm because it does nothing" is
  about realized behaviour), but potentially confounded if the two trained
  arms differ systematically in average hold duration. C5 stays gated on the
  realized/hold-weighted statistic; a companion `policy_entropy_fresh_select`
  (gated on `ticks["e3_tick"]`, i.e. only genuinely fresh E3 selections) plus
  `n_fresh_select`/`n_held` counts are recorded per cell and per-condition so
  a reader can check whether the two readings diverge before trusting C5
  alone. Not gated on -- diagnostic-only companion field.

Three conditions, matched seeds:
  A. FULL          -- goal_weight=1.0, drive_weight=2.0, drive_floor=0.9,
                       drive_ema_alpha=1.0, z_goal_enabled=True,
                       benefit_eval_enabled=True (approach + avoidance)
  B. AVOIDANCE_ONLY -- goal_weight=0.0, drive_weight=0.0, z_goal_enabled=False,
                       benefit_eval_enabled=False, e1_goal_conditioned=False
                       (harm-avoidance / E2+E3-harm training IDENTICAL to A;
                       only the wanting/goal pathway is severed)
  C. RANDOM         -- uniform action selection, no training (behavioural
                       floor; distinguishes "B is actively quiescent" from
                       "B just can't find resources")

PRE-REGISTERED GATES (in order; G0 must pass first or the run self-routes
to non_contributory -- never a false weakens of INV-034/Q-021):
  G0 non-degeneracy: resource_visit_rate_FULL >= resource_visit_rate_RANDOM
      + 0.05 on >= 2/3 seeds (FULL itself must be genuinely goal-directed;
      this is the exact precondition Q-021 pre-registers, and the failure
      mode that killed EXQ-072b).
  C1 harm-avoidance parity (the control): harm_rate_AVOIDANCE_ONLY <=
      harm_rate_FULL * 1.15 on >= 2/3 seeds.
  C2 survival parity (MECH-180 death-vs-quiescence guard):
      episode_survival_rate_AVOIDANCE_ONLY >= episode_survival_rate_FULL - 0.1
      on >= 2/3 seeds.
  C3 quiescence/flatness: resource_visit_rate_AVOIDANCE_ONLY <=
      resource_visit_rate_RANDOM + 0.05 on >= 2/3 seeds.
  C4 approach restored by goal maintenance: resource_visit_rate_FULL >=
      resource_visit_rate_AVOIDANCE_ONLY + 0.10 on >= 2/3 seeds.
  C5 entropy signature: policy_entropy_AVOIDANCE_ONLY <=
      policy_entropy_FULL - 0.10 on >= 2/3 seeds.
  C6 mechanistic check: z_goal_norm_at_contact_FULL > 0.4 (matches the
      603n/514o G3 gate) AND z_goal_norm_at_contact_AVOIDANCE_ONLY < 0.1
      on >= 2/3 seeds.

PASS (supports INV-034 + Q-021) = G0 AND C1 AND C2 AND C3 AND C4 AND C5 AND C6.
G0 fails -> non_contributory (substrate/env non-degeneracy issue; escalate to
  the full scaffolded_sd054_onboarding curriculum rather than re-running this
  lightweight harness -- see IGW-222 DESIGN.md 5.8, "Option B").
C1/C2 fail (with G0 passing) -> mixed/non_contributory (the ablation isn't
  clean -- AVOIDANCE_ONLY is also worse at avoidance or survival, so a
  resource-rate gap doesn't isolate agency from competence).
G0+C1+C2 pass but C3/C4/C5/C6 don't show the predicted gap -> weakens (clean,
  matched, non-degenerate setup; genuine evidence against the flatness
  prediction).

Biological basis: Bariselli 2018 (PMID 29481617) D1/D2 pathways evaluate
the SAME proposals; without D1 (Go/wanting) only D2 (NoGo/avoidance)
constrains selection -> quiescent default. Barch & Dowd 2010
(PMID 20868638) "wanting" (prospective) vs "liking" (reactive); avolition
in schizophrenia = intact liking, absent wanting.

SLEEP DRIVER: N/A (no sleep loop; short two-phase-plus-eval curriculum).

ethics_preflight:
  involves_negative_valence: false        # harm stream is pre-ethical instrumentation (SENT-0)
  involves_suffering_like_state: false
  involves_self_model: false
  involves_inescapability_or_helplessness: false
  involves_offline_replay_over_harm: false
  involves_social_mind_or_language: false
  involves_human_data_or_clinical_context: false
  decision: allow
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import math
import time
import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.optim as optim

from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig
from ree_core.agent import REEAgent
from experiments.pack_writer import write_flat_manifest  # noqa: E402
from experiment_protocol import emit_outcome  # noqa: E402
from experiments._lib.arm_fingerprint import arm_cell  # noqa: E402
from experiments._lib.z_goal_stream import ZGoalStreamAccumulator  # noqa: E402

# --------------------------------------------------------------------- #
# Experiment metadata
# --------------------------------------------------------------------- #
EXPERIMENT_TYPE = "v3_exq_866_inv034_q021_goal_maintenance_agency"
QUEUE_ID = "V3-EXQ-866"
CLAIM_IDS = ["INV-034", "Q-021"]
EXPERIMENT_PURPOSE = "evidence"

SEEDS = [42, 43, 44]
CONDITIONS = ["FULL", "AVOIDANCE_ONLY", "RANDOM"]

P0_EPISODES = 100   # encoder + harm/benefit-eval warmup, no goal scoring
P1_EPISODES = 150   # full pipeline; goal seeding + goal_weight active per arm
P2_EPISODES = 50    # frozen-policy eval, no training updates
STEPS_PER_EP = 200
TOTAL_TRAIN_EPS = P0_EPISODES + P1_EPISODES + P2_EPISODES  # denominator for [train] ep N/M

GRID_SIZE = 10
NUM_RESOURCES = 4
NUM_HAZARDS = 3
HAZARD_HARM = 0.15
RESOURCE_BENEFIT = 0.5
PROXIMITY_HARM_SCALE = 0.05
PROXIMITY_BENEFIT_SCALE = 0.18  # matches EXQ-189/238 validated value
PROXIMITY_APPROACH_THRESHOLD = 0.15

# Current validated goal-pipeline knobs (MECH-306 / SD-012 / SD-057).
DRIVE_WEIGHT = 2.0
GOAL_WEIGHT = 1.0
DRIVE_FLOOR = 0.9         # MECH-306: fixes the contact-step drive collapse
DRIVE_EMA_ALPHA = 1.0     # Option 1 OFF; Option 2 (drive_floor) is the validated fix
BENEFIT_THRESHOLD = 0.1

LR = 3e-4

# Pre-registered thresholds (see module docstring gates G0/C1-C6)
MIN_SEEDS_PASS = 2  # of 3 -- ">= 2/3 seeds"
G0_MARGIN = 0.05
C1_HARM_TOLERANCE = 1.15
C2_SURVIVAL_MARGIN = 0.10
C3_MARGIN = 0.05
C4_MARGIN = 0.10
C5_ENTROPY_MARGIN = 0.10
C6_ZGOAL_ACTIVE_FLOOR = 0.4
C6_ZGOAL_INACTIVE_CEIL = 0.1

DRY_RUN_EPISODES = 3
DRY_RUN_STEPS = 20


# --------------------------------------------------------------------- #
# Factories
# --------------------------------------------------------------------- #

def _make_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(
        seed=seed,
        size=GRID_SIZE,
        num_resources=NUM_RESOURCES,
        num_hazards=NUM_HAZARDS,
        hazard_harm=HAZARD_HARM,
        resource_benefit=RESOURCE_BENEFIT,
        resource_respawn_on_consume=True,
        proximity_harm_scale=PROXIMITY_HARM_SCALE,
        proximity_benefit_scale=PROXIMITY_BENEFIT_SCALE,
        proximity_approach_threshold=PROXIMITY_APPROACH_THRESHOLD,
        use_proxy_fields=True,
    )


def _make_agent(env: CausalGridWorldV2, condition: str, seed: int) -> REEAgent:
    torch.manual_seed(seed)
    full = (condition == "FULL")
    config = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=32,
        world_dim=32,
        alpha_world=0.9,
        alpha_self=0.3,
        use_event_classifier=True,
        use_resource_proximity_head=True,   # both arms -- encoder-quality fairness
        resource_proximity_weight=0.5,
        # Go / wanting channel (FULL only)
        benefit_eval_enabled=full,
        benefit_weight=1.0 if full else 0.0,
        z_goal_enabled=full,
        goal_weight=GOAL_WEIGHT if full else 0.0,
        e1_goal_conditioned=full,
        drive_weight=DRIVE_WEIGHT if full else 0.0,
        drive_floor=DRIVE_FLOOR if full else 0.0,
        drive_ema_alpha=DRIVE_EMA_ALPHA,
        benefit_threshold=BENEFIT_THRESHOLD,
        use_incentive_token_bank=full,  # SD-057 object-bound salience, FULL only
        # NoGo / harm-avoidance channel: identical in both trained arms --
        # this is the C1/C2 control. E3 harm_eval is always active.
    )
    return REEAgent(config)


def _action_entropy(action_counts: List[int]) -> float:
    total = sum(action_counts) + 1e-8
    probs = [c / total for c in action_counts]
    return -sum(p * math.log(p + 1e-9) for p in probs if p > 0)


def _cell_config_slice(condition: str, seed: int) -> Dict:
    """Fingerprint input for arm_cell -- every knob that can affect this cell's
    trajectory. Deliberately narrow (env + agent config only, not code paths)."""
    full = (condition == "FULL")
    return {
        "condition": condition,
        "seed": seed,
        "grid_size": GRID_SIZE,
        "num_resources": NUM_RESOURCES,
        "num_hazards": NUM_HAZARDS,
        "hazard_harm": HAZARD_HARM,
        "resource_benefit": RESOURCE_BENEFIT,
        "proximity_harm_scale": PROXIMITY_HARM_SCALE,
        "proximity_benefit_scale": PROXIMITY_BENEFIT_SCALE,
        "proximity_approach_threshold": PROXIMITY_APPROACH_THRESHOLD,
        "p0_episodes": P0_EPISODES,
        "p1_episodes": P1_EPISODES,
        "p2_episodes": P2_EPISODES,
        "steps_per_ep": STEPS_PER_EP,
        "drive_weight": DRIVE_WEIGHT if full else 0.0,
        "goal_weight": GOAL_WEIGHT if full else 0.0,
        "drive_floor": DRIVE_FLOOR if full else 0.0,
        "drive_ema_alpha": DRIVE_EMA_ALPHA,
        "benefit_threshold": BENEFIT_THRESHOLD,
        "benefit_eval_enabled": full,
        "z_goal_enabled": full,
        "use_incentive_token_bank": full,
        "lr": LR,
    }


# --------------------------------------------------------------------- #
# Trained conditions (FULL / AVOIDANCE_ONLY)
# --------------------------------------------------------------------- #

def _run_trained_cell(
    seed: int, condition: str, zg: ZGoalStreamAccumulator, dry_run: bool = False
) -> Dict:
    total_p0 = DRY_RUN_EPISODES if dry_run else P0_EPISODES
    total_p1 = DRY_RUN_EPISODES if dry_run else P1_EPISODES
    total_p2 = DRY_RUN_EPISODES if dry_run else P2_EPISODES
    steps_per = DRY_RUN_STEPS if dry_run else STEPS_PER_EP
    total_eps = total_p0 + total_p1 + total_p2

    print(f"Seed {seed} Condition {condition}", flush=True)

    env = _make_env(seed)
    agent = _make_agent(env, condition, seed)
    device = agent.device
    optimizer = optim.Adam(list(agent.parameters()), lr=LR)

    p2_resource_visits = 0
    p2_harm_events = 0
    p2_episodes_survived = 0
    p2_action_counts = [0] * env.action_dim          # every realized step (hold-weighted)
    p2_action_counts_fresh = [0] * env.action_dim     # only fresh E3 selections (ticks["e3_tick"])
    p2_total_steps = 0
    p2_fresh_select_steps = 0
    p2_zgoal_norms: List[float] = []

    for ep in range(total_eps):
        _, obs_dict = env.reset()
        agent.reset()

        phase = "P0" if ep < total_p0 else ("P1" if ep < total_p0 + total_p1 else "P2")
        in_eval = (phase == "P2")

        ep_died = False

        for _step in range(steps_per):
            obs_body = obs_dict["body_state"].to(device)
            obs_world = obs_dict["world_state"].to(device)

            z_self_prev: Optional[torch.Tensor] = None
            if agent._current_latent is not None:
                z_self_prev = agent._current_latent.z_self.detach().clone()

            latent = agent.sense(obs_body, obs_world)
            ticks = agent.clock.advance()
            e1_prior = (
                agent._e1_tick(latent)
                if ticks.get("e1_tick", True)
                else torch.zeros(1, 32, device=device)
            )
            candidates = agent.generate_trajectories(latent, e1_prior, ticks)
            action = agent.select_action(candidates, ticks, temperature=(1.0 if not in_eval else 0.5))
            action_idx = int(action.argmax(dim=-1).item())

            # Goal seeding: FULL only, P1/P2 (SD-012/MECH-306 signature
            # re-verified against live REEAgent.update_z_goal 2026-08-01 --
            # see module docstring API RE-VERIFICATION section).
            if condition == "FULL" and phase != "P0":
                benefit_raw = float(obs_body.flatten()[11].item()) if obs_body.shape[-1] > 11 else 0.0
                drive_level = REEAgent.compute_drive_level(obs_body)
                agent.update_z_goal(benefit_raw, drive_level)

            flat_next, harm_signal, done, info, obs_dict_next = env.step(action_idx)
            ttype = info.get("transition_type", "none")

            harm_val = abs(float(harm_signal)) if float(harm_signal) < 0 else 0.0
            benefit_val = max(0.0, float(info.get("benefit_exposure", 0.0)))

            agent.update_residue(float(harm_signal))
            if z_self_prev is not None:
                agent.record_transition(z_self_prev, action, latent.z_self.detach())

            if in_eval:
                p2_action_counts[action_idx] += 1
                p2_total_steps += 1
                if ticks.get("e3_tick", False):
                    # Fresh-selection-only companion count (validate_experiments.py
                    # hold-weighted-e3-readout advisory, 2026-08-01): the realized
                    # per-step count above is hold-weighted (a commitment held over
                    # N ticks repeats one action N times), which is the correct
                    # statistic for "did the agent's REALIZED behaviour look
                    # quiescent" (INV-034's own language) but can differ from the
                    # underlying E3 selection distribution. Track both, gate C5 on
                    # the realized one (the behavioural claim), report the fresh
                    # one alongside for triage.
                    p2_action_counts_fresh[action_idx] += 1
                    p2_fresh_select_steps += 1
                if ttype == "benefit_approach" or ttype == "resource":
                    p2_resource_visits += 1
                if ttype in ("agent_caused_hazard", "hazard_approach"):
                    p2_harm_events += 1
                if condition == "FULL" and agent.goal_state is not None:
                    # FIXED (2026-08-01, see docstring): the live z_goal tensor
                    # lives on agent.goal_state.z_goal, not agent.z_goal or
                    # agent._current_latent.z_goal -- neither of which exists.
                    p2_zgoal_norms.append(float(agent.goal_state.z_goal.norm().item()))
                if done and harm_val > 0.0:
                    ep_died = True
            else:
                optimizer.zero_grad()
                e1_loss = agent.compute_prediction_loss()
                e2_loss = agent.compute_e2_loss()
                loss = e1_loss + e2_loss

                rfv = obs_dict.get("resource_field_view", None)
                if rfv is not None:
                    rp_t = float(rfv.max().item())
                    loss = loss + agent.compute_resource_proximity_loss(rp_t, latent)

                if condition == "FULL" and phase == "P1":
                    benefit_t = torch.tensor([[benefit_val]], dtype=torch.float32, device=device)
                    bel = agent.compute_benefit_eval_loss(benefit_t)
                    loss = loss + bel

                if loss.requires_grad:
                    loss.backward()
                    nn.utils.clip_grad_norm_(list(agent.parameters()), 1.0)
                    optimizer.step()

            obs_dict = obs_dict_next
            if done:
                break

        if in_eval:
            p2_episodes_survived += 0 if ep_died else 1

        if (ep + 1) % 50 == 0 or (ep + 1) == total_eps:
            print(
                f"  [train] seed={seed} {condition} ep {ep + 1}/{total_eps} phase={phase}",
                flush=True,
            )

    zg.observe(agent)

    resource_visit_rate = p2_resource_visits / max(1, p2_total_steps)
    harm_rate = p2_harm_events / max(1, p2_total_steps)
    policy_entropy = _action_entropy(p2_action_counts)
    policy_entropy_fresh_select = (
        _action_entropy(p2_action_counts_fresh) if p2_fresh_select_steps > 0 else 0.0
    )
    survival_rate = p2_episodes_survived / max(1, total_p2)
    zgoal_norm_mean = (
        sum(p2_zgoal_norms) / len(p2_zgoal_norms) if p2_zgoal_norms else 0.0
    )

    print(
        f"  [eval] {condition} seed={seed} resource_rate={resource_visit_rate:.4f} "
        f"harm_rate={harm_rate:.4f} entropy={policy_entropy:.4f} "
        f"survival={survival_rate:.4f} zgoal_norm={zgoal_norm_mean:.4f}",
        flush=True,
    )
    print("verdict: PASS", flush=True)  # cell ran to completion; scientific verdict is aggregate-level

    return {
        "seed": seed,
        "condition": condition,
        "resource_visit_rate": resource_visit_rate,
        "harm_rate": harm_rate,
        "policy_entropy": policy_entropy,
        "policy_entropy_fresh_select": policy_entropy_fresh_select,
        "n_fresh_select": p2_fresh_select_steps,
        "n_held": p2_total_steps - p2_fresh_select_steps,
        "survival_rate": survival_rate,
        "zgoal_norm_mean": zgoal_norm_mean,
        "n_resource_events": p2_resource_visits,
        "n_harm_events": p2_harm_events,
        "total_steps": p2_total_steps,
    }


# --------------------------------------------------------------------- #
# Random baseline (no training)
# --------------------------------------------------------------------- #

def _run_random_cell(seed: int, dry_run: bool = False) -> Dict:
    import random as _random

    total_p2 = DRY_RUN_EPISODES if dry_run else P2_EPISODES
    steps_per = DRY_RUN_STEPS if dry_run else STEPS_PER_EP

    print(f"Seed {seed} Condition RANDOM", flush=True)

    _random.seed(seed)
    env = _make_env(seed)
    action_dim = env.action_dim

    resource_visits = 0
    harm_events = 0
    episodes_survived = 0
    action_counts = [0] * action_dim
    total_steps = 0

    for _ep in range(total_p2):
        _, obs_dict = env.reset()
        died = False
        for _step in range(steps_per):
            action_idx = _random.randint(0, action_dim - 1)
            action_counts[action_idx] += 1
            _, harm_signal, done, info, obs_dict = env.step(action_idx)
            ttype = info.get("transition_type", "none")
            harm_val = abs(float(harm_signal)) if float(harm_signal) < 0 else 0.0
            if ttype == "benefit_approach" or ttype == "resource":
                resource_visits += 1
            if ttype in ("agent_caused_hazard", "hazard_approach"):
                harm_events += 1
            total_steps += 1
            if done:
                if harm_val > 0.0:
                    died = True
                break
        episodes_survived += 0 if died else 1

    resource_visit_rate = resource_visits / max(1, total_steps)
    harm_rate = harm_events / max(1, total_steps)
    policy_entropy = _action_entropy(action_counts)
    survival_rate = episodes_survived / max(1, total_p2)

    print(
        f"  [eval] RANDOM seed={seed} resource_rate={resource_visit_rate:.4f} "
        f"harm_rate={harm_rate:.4f} entropy={policy_entropy:.4f} survival={survival_rate:.4f}",
        flush=True,
    )
    print("verdict: PASS", flush=True)

    return {
        "seed": seed,
        "condition": "RANDOM",
        "resource_visit_rate": resource_visit_rate,
        "harm_rate": harm_rate,
        "policy_entropy": policy_entropy,
        # RANDOM has no E3 commitment/hold mechanism -- every step is an
        # independent draw, so the fresh-select and realized statistics
        # coincide by construction (schema parity with the trained cells).
        "policy_entropy_fresh_select": policy_entropy,
        "n_fresh_select": total_steps,
        "n_held": 0,
        "survival_rate": survival_rate,
        "zgoal_norm_mean": 0.0,
        "n_resource_events": resource_visits,
        "n_harm_events": harm_events,
        "total_steps": total_steps,
    }


# --------------------------------------------------------------------- #
# Aggregate + acceptance criteria
# --------------------------------------------------------------------- #

def _frac_seeds(pred_per_seed: List[bool]) -> float:
    return sum(1 for p in pred_per_seed if p) / max(1, len(pred_per_seed))


def run(dry_run: bool = False) -> dict:
    print(f"\n[{QUEUE_ID}] INV-034 / Q-021 Goal-Maintenance-Necessary-for-Agency", flush=True)

    zg = ZGoalStreamAccumulator()
    arm_results: List[Dict] = []
    per_seed: Dict[str, Dict[int, Dict]] = {"FULL": {}, "AVOIDANCE_ONLY": {}, "RANDOM": {}}

    for seed in SEEDS:
        for condition in CONDITIONS:
            config_slice = _cell_config_slice(condition, seed)
            with arm_cell(seed, config_slice=config_slice, script_path=Path(__file__)) as cell:
                if condition == "RANDOM":
                    row = _run_random_cell(seed, dry_run=dry_run)
                else:
                    row = _run_trained_cell(seed, condition, zg, dry_run=dry_run)
                cell.stamp(row)
            arm_results.append(row)
            per_seed[condition][seed] = row

    g0_per_seed, c1_per_seed, c2_per_seed = [], [], []
    c3_per_seed, c4_per_seed, c5_per_seed, c6_per_seed = [], [], [], []

    for seed in SEEDS:
        full = per_seed["FULL"][seed]
        avoid = per_seed["AVOIDANCE_ONLY"][seed]
        rand = per_seed["RANDOM"][seed]

        g0_per_seed.append(full["resource_visit_rate"] >= rand["resource_visit_rate"] + G0_MARGIN)
        c1_per_seed.append(avoid["harm_rate"] <= full["harm_rate"] * C1_HARM_TOLERANCE)
        c2_per_seed.append(avoid["survival_rate"] >= full["survival_rate"] - C2_SURVIVAL_MARGIN)
        c3_per_seed.append(avoid["resource_visit_rate"] <= rand["resource_visit_rate"] + C3_MARGIN)
        c4_per_seed.append(full["resource_visit_rate"] >= avoid["resource_visit_rate"] + C4_MARGIN)
        c5_per_seed.append(avoid["policy_entropy"] <= full["policy_entropy"] - C5_ENTROPY_MARGIN)
        c6_per_seed.append(
            full["zgoal_norm_mean"] > C6_ZGOAL_ACTIVE_FLOOR
            and avoid["zgoal_norm_mean"] < C6_ZGOAL_INACTIVE_CEIL
        )

    g0_frac = _frac_seeds(g0_per_seed)
    c1_frac = _frac_seeds(c1_per_seed)
    c2_frac = _frac_seeds(c2_per_seed)
    c3_frac = _frac_seeds(c3_per_seed)
    c4_frac = _frac_seeds(c4_per_seed)
    c5_frac = _frac_seeds(c5_per_seed)
    c6_frac = _frac_seeds(c6_per_seed)

    threshold = MIN_SEEDS_PASS / len(SEEDS)
    g0_pass = g0_frac >= threshold
    c1_pass = c1_frac >= threshold
    c2_pass = c2_frac >= threshold
    c3_pass = c3_frac >= threshold
    c4_pass = c4_frac >= threshold
    c5_pass = c5_frac >= threshold
    c6_pass = c6_frac >= threshold

    non_degenerate = True
    degeneracy_reason = None

    if not g0_pass:
        status = "FAIL"
        evidence_direction = "non_contributory"
        route_reason = "non_degenerate_precondition_unmet"
        non_degenerate = False
        degeneracy_reason = (
            "G0 non-degeneracy gate failed: FULL arm did not clear random-baseline "
            "resource-visit rate by the pre-registered margin on >= 2/3 seeds."
        )
        interpretation = (
            "G0 non-degeneracy gate FAILED: the FULL (approach+avoidance) arm did not "
            "clear random-baseline resource-visit rate by the pre-registered margin on "
            ">= 2/3 seeds. This is a substrate/env non-degeneracy issue (mirrors the "
            "EXQ-072b failure mode), NOT evidence against INV-034/Q-021. Per IGW-222 "
            "DESIGN.md 5.8, escalate to the full scaffolded_sd054_onboarding curriculum "
            "(Option B) rather than re-running this lightweight harness -- that "
            "curriculum is the substrate that actually cleared goal_pipeline GAP-2's "
            "contact-rate ceiling."
        )
    elif not (c1_pass and c2_pass):
        status = "FAIL"
        evidence_direction = "non_contributory"
        route_reason = "ablation_not_clean_competence_confound"
        interpretation = (
            "G0 passed but the ablation is not clean: AVOIDANCE_ONLY differs from FULL "
            "in harm-avoidance (C1) and/or survival (C2), so a resource-rate gap cannot "
            "be attributed to agency specifically vs general competence loss (C1) or a "
            "MECH-180-style death-not-quiescence confound (C2). Route to diagnosis of "
            "why severing the goal_weight/drive pathway degrades harm-avoidance/survival "
            "before re-attempting this comparison."
        )
    else:
        all_pass = c3_pass and c4_pass and c5_pass and c6_pass
        status = "PASS" if all_pass else "FAIL"
        evidence_direction = "supports" if all_pass else "weakens"
        route_reason = "clean_ablation_scored"
        if all_pass:
            interpretation = (
                "INV-034 + Q-021 SUPPORTED: with harm-avoidance and survival held "
                "comparable between arms (C1/C2), the avoidance-only architecture "
                "collapses to a quiescent/flat policy at-or-below random baseline (C3) "
                "with reduced action diversity (C5), while the goal-maintaining FULL "
                "architecture sustains systematic resource-directed approach (C4) with "
                "z_goal genuinely engaged only in FULL (C6). Goal maintenance is "
                "necessary for the agent to exercise committed goal-directed agency "
                "beyond harm-avoidance alone."
            )
        else:
            interpretation = (
                "INV-034 + Q-021 WEAKENED: the ablation is clean (harm-avoidance/"
                "survival held comparable, C1/C2 PASS) but the predicted "
                "quiescence/entropy/z_goal-engagement gap did not clear threshold "
                "on >= 2/3 seeds. Genuine evidence against the flatness prediction "
                "on this substrate configuration."
            )

    metrics: Dict[str, float] = {
        "g0_frac_seeds": g0_frac,
        "c1_frac_seeds": c1_frac,
        "c2_frac_seeds": c2_frac,
        "c3_frac_seeds": c3_frac,
        "c4_frac_seeds": c4_frac,
        "c5_frac_seeds": c5_frac,
        "c6_frac_seeds": c6_frac,
        "g0_pass": 1.0 if g0_pass else 0.0,
        "c1_pass": 1.0 if c1_pass else 0.0,
        "c2_pass": 1.0 if c2_pass else 0.0,
        "c3_pass": 1.0 if c3_pass else 0.0,
        "c4_pass": 1.0 if c4_pass else 0.0,
        "c5_pass": 1.0 if c5_pass else 0.0,
        "c6_pass": 1.0 if c6_pass else 0.0,
    }
    for cond in CONDITIONS:
        rvr = [per_seed[cond][s]["resource_visit_rate"] for s in SEEDS]
        hr = [per_seed[cond][s]["harm_rate"] for s in SEEDS]
        ent = [per_seed[cond][s]["policy_entropy"] for s in SEEDS]
        ent_fresh = [per_seed[cond][s]["policy_entropy_fresh_select"] for s in SEEDS]
        surv = [per_seed[cond][s]["survival_rate"] for s in SEEDS]
        zg_vals = [per_seed[cond][s]["zgoal_norm_mean"] for s in SEEDS]
        metrics[f"resource_visit_rate_mean_{cond}"] = sum(rvr) / len(rvr)
        metrics[f"harm_rate_mean_{cond}"] = sum(hr) / len(hr)
        metrics[f"policy_entropy_mean_{cond}"] = sum(ent) / len(ent)
        metrics[f"policy_entropy_fresh_select_mean_{cond}"] = sum(ent_fresh) / len(ent_fresh)
        metrics[f"survival_rate_mean_{cond}"] = sum(surv) / len(surv)
        metrics[f"zgoal_norm_mean_{cond}"] = sum(zg_vals) / len(zg_vals)

    evidence_direction_per_claim = {
        "INV-034": evidence_direction,
        "Q-021": evidence_direction,
    }

    summary_markdown = (
        f"# {QUEUE_ID} -- INV-034 / Q-021 Goal-Maintenance-Necessary-for-Agency\n\n"
        f"**Status:** {status}  **Evidence direction:** {evidence_direction}\n"
        f"**Route reason:** {route_reason}\n"
        f"**Claims:** INV-034, Q-021\n\n"
        f"## Gates\n\n"
        f"| Gate | Frac seeds | Pass |\n|---|---|---|\n"
        f"| G0 non-degeneracy | {g0_frac:.2f} | {g0_pass} |\n"
        f"| C1 harm parity | {c1_frac:.2f} | {c1_pass} |\n"
        f"| C2 survival parity | {c2_frac:.2f} | {c2_pass} |\n"
        f"| C3 quiescence (avoidance-only flat) | {c3_frac:.2f} | {c3_pass} |\n"
        f"| C4 approach restored (FULL > avoidance) | {c4_frac:.2f} | {c4_pass} |\n"
        f"| C5 entropy signature | {c5_frac:.2f} | {c5_pass} |\n"
        f"| C6 z_goal mechanistic check | {c6_frac:.2f} | {c6_pass} |\n\n"
        f"## Interpretation\n\n{interpretation}\n"
    )

    result = {
        "status": status,
        "outcome": status,
        "metrics": metrics,
        "arm_results": arm_results,
        "summary_markdown": summary_markdown,
        "claim_ids": CLAIM_IDS,
        "evidence_direction": evidence_direction,
        "evidence_direction_per_claim": evidence_direction_per_claim,
        "experiment_type": EXPERIMENT_TYPE,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "per_seed_results": per_seed,
        "fatal_error_count": 0,
    }
    if not non_degenerate:
        result["non_degenerate"] = False
        result["degeneracy_reason"] = degeneracy_reason

    return result, zg


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    t0 = time.perf_counter()
    result, zg_accumulator = run(dry_run=args.dry_run)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result["run_timestamp"] = ts
    result["run_id"] = f"{EXPERIMENT_TYPE}_{ts}_v3"
    result["architecture_epoch"] = "ree_hybrid_guardrails_v1"

    out_path = write_flat_manifest(
        result,
        dry_run=args.dry_run,
        config={
            "seeds": SEEDS,
            "conditions": CONDITIONS,
            "drive_floor": DRIVE_FLOOR,
            "drive_ema_alpha": DRIVE_EMA_ALPHA,
            "drive_weight": DRIVE_WEIGHT,
            "goal_weight": GOAL_WEIGHT,
            "benefit_threshold": BENEFIT_THRESHOLD,
            "p0_episodes": P0_EPISODES,
            "p1_episodes": P1_EPISODES,
            "p2_episodes": P2_EPISODES,
            "steps_per_ep": STEPS_PER_EP,
        },
        seeds=SEEDS,
        script_path=__file__,
        started_at=t0,
        z_goal_stream_stats=zg_accumulator.stats(),
    )

    print(f"\nResult written to: {out_path}", flush=True)
    print(f"Status: {result['status']}", flush=True)

    emit_outcome(
        outcome=result["status"] if result["status"] in ("PASS", "FAIL") else "FAIL",
        manifest_path=out_path,
        dry_run=args.dry_run,
    )
