#!/opt/local/bin/python3
"""V3-EXQ-538 -- SD-049 Phase 2 reef behavioural validation, sleep-on ablation.

----------------------------------------------------------------------
HOLD STATUS (2026-05-08): DO NOT REQUEUE WITHOUT CHECKING THIS LIST.
----------------------------------------------------------------------
First attempt 2026-05-08T12:54Z: SIGTERM exit -15 on ree-cloud-1 (24 min in,
no manifest landed). NOT a script bug -- cloud VM killed under it. The
sentinel-file conformance contract that landed 2026-05-08T13:35Z addresses
the silent-drop pattern, but the substrate-level hold below is the binding
gate.

Held pending (per Friday PM governance cycle 2026-05-08T17:03Z):
  [ ] Tier-1: V3-EXQ-530b (ARC-016 under StepHarness) lands and clears.
        State 2026-05-08: ran 2026-05-07T21:42Z, classified UNKNOWN (Outcome
        FAIL); needs review under StepHarness post-Q-042 contracts.
  [ ] Tier-1: V3-EXQ-514g (SD-049 wider seed sweep under StepHarness)
        authored, queued, and PASSed.
        State 2026-05-08: NOT YET QUEUED. Awaiting /queue-experiment hand-off
        per the Tier-1 retest plan.
  [ ] MECH-307 substrate validated.
        State 2026-05-08: V3-EXQ-539 commit-gating ran today and FAILed
        (substrate fires -- ARM_ON liking=2711, z_beta_exc=0.157, harm
        surprise centers=232 -- but commit-chain stays inert). MECH-307
        alone does not unblock the SD-049 cohort.
  [ ] Sleep-substrate audit complete.
        State 2026-05-08: approved as next concrete step in Friday governance;
        pre-requisite to staking SD-017 as load-bearing for SD-049 evidence
        acceptance per this experiment's own PASS reading.

When all four boxes are checked: this experiment becomes the canonical
Tier-3 follow-on for the 514f cohort (post-MECH-307 + post-sleep-audit).
Re-evaluate claim_ids and acceptance criteria at requeue time -- the
substrate state will likely have moved.

Provenance: marked discussed in review_tracker.json
2026-05-08T17:30Z under session diagnose-errors-244a-538-433f.
----------------------------------------------------------------------

Supersedes test of: V3-EXQ-514f (FAIL 2026-05-08T04:32Z, C1b classifier_converges
False, C2b probe_acc_neighborhood < 0.6 across all reef arms).

Hypothesis (from 2026-05-08 review of 514f):
  514f confirmed the reef substrate fires (peak per-axis drives evolve as
  expected; 13,500 neighborhood samples per reef arm) but the identity classifier
  diverges in P0 (loss 2.92 -> 3.97 in ARM_1; ARM_3 5-type collapses
  probe_acc_neighborhood from 0.564 -> 0.270). The default
  classifier_loss_weight=0.1 is below the cross-over for offline-free training:
  the optimizer prioritises world-prediction loss and the classifier head free-
  rides on a near-uniform readout.

  Sleep-dependent schema consolidation is the missing ingredient. Lansink et
  al. 2009 (hippocampus-leads-striatum SWR replay), Stickgold 2013 (memory
  consolidation), Walker 2017 (REM-dependent generalisation) all argue that
  identity-discrimination structure is etched offline, not online during the
  prediction-loss-dominated waking pass. V3 already has the substrate
  (SD-017 SWS+REM passes; Sleep Aggregation Cluster Phases A-E landed
  2026-04-25); EXQ-503 validated SWS writes + REM rollouts firing when the
  flags are on. But 514f, like the rest of the SD-049 / SD-016 cohort, ran
  sleep OFF. This experiment turns it on.

Design (3 arms x 3 seeds, paired to 514f ARM_2_3type as canonical config):
  ARM_0_off:           sws=False, rem=False, use_sleep_loop=False
                       Replicates 514f baseline. Control.
  ARM_1_sd017_only:    sws=True, rem=True, use_sleep_loop=False.
                       agent.run_sleep_cycle() called manually between
                       episodes. Tests basic SD-017 SWS+REM consolidation
                       without Phase A K-episode driver.
  ARM_2_phase_a:       sws=True, rem=True, use_sleep_loop=True, K=3.
                       SleepLoopManager fires sleep cycle every 3 episodes
                       via REEAgent.reset() -> notify_episode_end() path.
                       Tests full Phase A scaffolding.

  All other config matches 514f ARM_2_3type (the canonical "fail to discriminate"
  arm): 3-type SD-049, reef ON, hazard_food_attraction=0.7, 10x10 grid,
  3 hazards, 12 resources, classifier_loss_weight=0.1 unchanged so this
  experiment isolates the sleep axis.

Pre-registered acceptance criteria:
----------------------------------------------------------------------
C0:  ARM_0 runs to completion without crash.
C1a: ARM_1 cumulative_sws_writes >= 1.0 per seed mean (sleep substrate fires).
C1b: ARM_1 cumulative_rem_rollouts >= 1.0 per seed mean.
C2a: ARM_2 cumulative_sws_writes >= 1.0 per seed mean (Phase A driver fires).
C2b: ARM_2 cumulative_sws_writes_per_cycle (cumulative / number of fires)
     consistent with K=3 (i.e. fewer total fires than ARM_1, which fires every
     episode).
C3a: ARM_1 OR ARM_2 probe_acc_neighborhood_mean >= 0.6 (the load-bearing
     514f failure mode -- does sleep recover discrimination?).
C3b: ARM_1 OR ARM_2 p0_classifier_loss_last_q < p0_classifier_loss_first_q
     (classifier converges in at least one sleep arm).
C4:  At least one sleep arm beats ARM_0 on probe_acc_neighborhood by >= 0.10
     absolute (sleep effect is real, not noise).

PASS = C0 AND C1a AND C1b AND C2a AND C2b AND (C3a OR C3b) AND C4.

PASS reading: sleep-dependent schema consolidation recovers the SD-049 Phase 2
identity-discrimination failure mode. Promotes SD-017 to a load-bearing
prerequisite for SD-049 / SD-015 evidence acceptance, not just a substrate
landed for its own sake. SD-049 v3_pending may be cleared on this evidence.
Likely also unblocks SD-016 cue-context discrimination (parked pending
env-entropy precondition; sleep + reef may be the joint precondition).

PARTIAL READING (C3a XOR C3b, otherwise PASS):
  Sleep helps schema etching but the recovery is incomplete. Routes to
  V3-EXQ-538a with classifier_loss_weight=0.5 (combined sleep + signal-strength
  axis).

FAIL on C3a AND C3b:
  Sleep does not recover discrimination. The 514f failure is downstream of
  consolidation -- candidates: classifier_loss_weight too low (route to 538a),
  identity classifier head architecture inadequate, or curriculum still too
  uniform (would need SD-049 Phase 3 multi-episode environmental shift).

experiment_purpose = "evidence" (governance evidence; sleep ablation of 514f).
supersedes_diagnostic: V3-EXQ-514f (does not invalidate 514f's reef-substrate
  PASSes on C0/C1a/C2a/C2c/C2d/C3a/C3b; this experiment tests the orthogonal
  sleep axis)

Run with:
  /opt/local/bin/python3 experiments/v3_exq_538_sd049_phase2_with_sleep.py
  /opt/local/bin/python3 experiments/v3_exq_538_sd049_phase2_with_sleep.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorld  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_538_sd049_phase2_with_sleep"
CLAIM_IDS = ["SD-049", "SD-015", "SD-017", "MECH-229", "MECH-230"]
EXPERIMENT_PURPOSE = "evidence"
SUPERSEDES_DIAGNOSTIC = "V3-EXQ-514f"

SEEDS = [42, 43, 44]
P0_EPISODES = 30
P1_EPISODES = 10
EVAL_EPISODES = 15
STEPS_PER_EPISODE = 300
LR = 1e-3

CLASSIFIER_LOSS_WEIGHT = 0.1  # unchanged from 514f to isolate sleep axis

# Reef parameters from V3-EXQ-521/522 (matched to 514f ARM_2 canonical).
REEF_ENABLED = True
HAZARD_FOOD_ATTRACTION = 0.7
GRID_SIZE = 10
NUM_HAZARDS = 3
NUM_RESOURCES = 12
N_RESOURCE_TYPES = 3

# Sleep parameters.
SLEEP_LOOP_K = 3                # Phase A driver fires every K episodes.
SWS_CONSOLIDATION_STEPS = 4
SWS_SCHEMA_WEIGHT = 0.1
REM_ATTRIBUTION_STEPS = 4

# Acceptance thresholds.
C1A_SWS_WRITES_FLOOR = 1.0
C1B_REM_ROLLOUTS_FLOOR = 1.0
C2A_SWS_WRITES_FLOOR = 1.0
C3A_PROBE_NEIGHBORHOOD_FLOOR = 0.6
C4_SLEEP_LIFT_FLOOR = 0.10

ARMS_CONFIG: List[Dict] = [
    dict(
        arm="ARM_0_off",
        sws_enabled=False,
        rem_enabled=False,
        use_sleep_loop=False,
        manual_sleep_between_episodes=False,
    ),
    dict(
        arm="ARM_1_sd017_only",
        sws_enabled=True,
        rem_enabled=True,
        use_sleep_loop=False,
        manual_sleep_between_episodes=True,
    ),
    dict(
        arm="ARM_2_phase_a",
        sws_enabled=True,
        rem_enabled=True,
        use_sleep_loop=True,
        manual_sleep_between_episodes=False,
    ),
]


def make_env(seed: int) -> CausalGridWorld:
    """All arms share the same env (the 514f ARM_2_3type canonical config)."""
    return CausalGridWorld(
        size=GRID_SIZE,
        num_hazards=NUM_HAZARDS,
        num_resources=NUM_RESOURCES,
        hazard_harm=0.01,
        resource_benefit=0.18,
        use_proxy_fields=True,
        seed=seed,
        proximity_benefit_scale=0.18,
        resource_respawn_on_consume=True,
        reef_enabled=REEF_ENABLED,
        n_reef_patches=3,
        reef_patch_radius=2,
        hazard_food_attraction=HAZARD_FOOD_ATTRACTION,
        multi_resource_heterogeneity_enabled=True,
        per_axis_drive_enabled=True,
        n_resource_types=N_RESOURCE_TYPES,
    )


def make_config(env: CausalGridWorld, arm_cfg: Dict) -> REEConfig:
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        alpha_world=0.9,
        world_dim=32,
        drive_weight=2.0,
        z_goal_enabled=True,
        use_sleep_loop=bool(arm_cfg["use_sleep_loop"]),
        sleep_loop_episodes_K=SLEEP_LOOP_K,
        sleep_loop_require_passes=True,
    )
    cfg.latent.use_resource_encoder = True
    cfg.latent.z_resource_dim = 32
    cfg.latent.use_identity_classifier = True
    cfg.latent.identity_classifier_n_types = N_RESOURCE_TYPES
    cfg.goal.goal_dim = cfg.latent.world_dim
    # SD-017 surface (used by both manual-sleep-between-episodes and SleepLoopManager).
    cfg.sws_enabled = bool(arm_cfg["sws_enabled"])
    cfg.sws_consolidation_steps = SWS_CONSOLIDATION_STEPS
    cfg.sws_schema_weight = SWS_SCHEMA_WEIGHT
    cfg.rem_enabled = bool(arm_cfg["rem_enabled"])
    cfg.rem_attribution_steps = REM_ATTRIBUTION_STEPS
    return cfg


def neighborhood_dominant_type(obs_dict: Dict, type_names: Tuple[str, ...]) -> int:
    if not type_names:
        return -1
    field_maxes = []
    for name in type_names:
        key = f"resource_field_view_{name}"
        if key not in obs_dict:
            return -1
        v = obs_dict[key]
        if hasattr(v, "max"):
            field_maxes.append(float(v.max().item()))
        else:
            return -1
    if max(field_maxes) < 0.05:
        return -1
    return int(np.argmax(field_maxes))


def _maybe_run_manual_sleep(agent: REEAgent, arm_cfg: Dict) -> Dict[str, float]:
    """ARM_1 only: manually trigger SD-017 SWS+REM passes between episodes."""
    if arm_cfg["manual_sleep_between_episodes"]:
        return agent.run_sleep_cycle() or {}
    return {}


def _sum_phase_a_history(agent: REEAgent) -> Tuple[float, float, int]:
    """Sum sws/rem metrics from the SleepLoopManager cycle history.

    Returns (cumulative_sws_writes, cumulative_rem_rollouts, n_fires).
    """
    if agent.sleep_loop is None:
        return 0.0, 0.0, 0
    history = getattr(agent.sleep_loop, "_cycle_history", None) or []
    sws = sum(float(c.get("sws_n_writes", 0.0) or 0.0) for c in history)
    rem = sum(float(c.get("rem_n_rollouts", 0.0) or 0.0) for c in history)
    return sws, rem, len(history)


def _phase_a_history_len(agent: REEAgent) -> int:
    """Number of sleep cycles the SleepLoopManager has fired so far.

    Use the length of the manager's internal cycle_history to detect "did a
    cycle just fire?" without re-counting state.last_metrics across the K-1
    quiescent episodes.
    """
    if agent.sleep_loop is None:
        return 0
    history = getattr(agent.sleep_loop, "_cycle_history", None)
    return len(history) if history is not None else 0


def _capture_phase_a_metrics_if_fired(
    agent: REEAgent, history_len_before: int
) -> Dict[str, float]:
    """Return the most-recent cycle metrics ONLY if a cycle just fired."""
    if agent.sleep_loop is None:
        return {}
    history = getattr(agent.sleep_loop, "_cycle_history", None)
    if not history or len(history) <= history_len_before:
        return {}
    return dict(history[-1])


def run_p0_training(
    agent: REEAgent,
    env: CausalGridWorld,
    device: torch.device,
    n_eps: int,
    classifier_loss_weight: float,
    arm_cfg: Dict,
) -> Dict:
    opt = optim.Adam(agent.parameters(), lr=LR)
    classifier_losses: List[float] = []
    prox_losses: List[float] = []
    cumulative_sws_writes = 0.0
    cumulative_rem_rollouts = 0.0
    n_sleep_fires = 0

    for ep in range(n_eps):
        _, obs_dict = env.reset()
        agent.reset()  # triggers Phase A notify_episode_end when use_sleep_loop=True
        for step in range(STEPS_PER_EPISODE):
            obs_body = obs_dict["body_state"].to(device)
            obs_world = obs_dict["world_state"].to(device)
            latent = agent.sense(obs_body, obs_world)

            ticks = agent.clock.advance()
            e1_prior = (
                agent._e1_tick(latent) if ticks.get("e1_tick")
                else torch.zeros(1, agent.config.latent.world_dim, device=device)
            )
            candidates = agent.generate_trajectories(latent, e1_prior, ticks)
            action = agent.select_action(candidates, ticks)
            action_idx = int(action.argmax(dim=-1).item())
            _, harm_signal, done, info, obs_dict = env.step(action_idx)
            ttype = info.get("transition_type", "none")

            opt.zero_grad()
            loss = agent.compute_prediction_loss()
            if (
                agent.latent_stack.resource_encoder is not None
                and latent.resource_prox_pred_r is not None
            ):
                prox_target_val = float(info.get("resource_field_at_agent", 0.0))
                prox_target = torch.tensor(
                    [[prox_target_val]], dtype=torch.float32, device=device
                )
                res_loss = agent.compute_resource_encoder_loss(prox_target, latent)
                loss = loss + res_loss
                prox_losses.append(float(res_loss.item()))
            if getattr(agent.config.latent, "use_identity_classifier", False):
                target_type = int(info.get("sd049_consumed_type_tag_this_tick", 0))
                id_loss = agent.compute_resource_identity_loss(target_type, latent)
                loss = loss + classifier_loss_weight * id_loss
                if id_loss.item() > 0:
                    classifier_losses.append(float(id_loss.item()))

            if loss.requires_grad:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                opt.step()

            if ttype == "resource":
                drive_lvl = float(REEAgent.compute_drive_level(obs_body))
                agent.update_z_goal(float(harm_signal), drive_level=drive_lvl)

            if done:
                break

        # ARM_1 only: manually fire SD-017 sleep cycle between episodes.
        # ARM_2 fires its sleep cycle inside agent.reset() via the
        # SleepLoopManager (next episode top); we read it back at end of phase
        # via _sum_phase_a_history(agent).
        manual_metrics = _maybe_run_manual_sleep(agent, arm_cfg)
        if manual_metrics:
            sws_w = float(manual_metrics.get("sws_n_writes", 0.0) or 0.0)
            rem_r = float(manual_metrics.get("rem_n_rollouts", 0.0) or 0.0)
            if sws_w > 0.0 or rem_r > 0.0:
                n_sleep_fires += 1
            cumulative_sws_writes += sws_w
            cumulative_rem_rollouts += rem_r

    # Phase A accounting: read SleepLoopManager._cycle_history once at end
    # of phase, replacing any zero counts above.
    if arm_cfg["use_sleep_loop"]:
        sws_a, rem_a, n_fires_a = _sum_phase_a_history(agent)
        cumulative_sws_writes = sws_a
        cumulative_rem_rollouts = rem_a
        n_sleep_fires = n_fires_a

    return {
        "p0_classifier_loss_first_quarter": (
            float(np.mean(classifier_losses[: max(1, len(classifier_losses) // 4)]))
            if classifier_losses else 0.0
        ),
        "p0_classifier_loss_last_quarter": (
            float(np.mean(classifier_losses[-max(1, len(classifier_losses) // 4):]))
            if classifier_losses else 0.0
        ),
        "p0_n_classifier_updates": len(classifier_losses),
        "p0_n_prox_updates": len(prox_losses),
        "p0_cumulative_sws_writes": cumulative_sws_writes,
        "p0_cumulative_rem_rollouts": cumulative_rem_rollouts,
        "p0_n_sleep_fires": n_sleep_fires,
    }


def run_p1_training(
    agent: REEAgent,
    env: CausalGridWorld,
    device: torch.device,
    n_eps: int,
    arm_cfg: Dict,
) -> Dict:
    if (
        agent.latent_stack.resource_encoder is not None
        and agent.latent_stack.resource_encoder.identity_head is not None
    ):
        for p in agent.latent_stack.resource_encoder.identity_head.parameters():
            p.requires_grad_(False)
    opt = optim.Adam(
        [p for p in agent.parameters() if p.requires_grad], lr=LR
    )
    cumulative_sws_writes = 0.0
    cumulative_rem_rollouts = 0.0

    for ep in range(n_eps):
        _, obs_dict = env.reset()
        agent.reset()
        for step in range(STEPS_PER_EPISODE):
            obs_body = obs_dict["body_state"].to(device)
            obs_world = obs_dict["world_state"].to(device)
            latent = agent.sense(obs_body, obs_world)

            ticks = agent.clock.advance()
            e1_prior = (
                agent._e1_tick(latent) if ticks.get("e1_tick")
                else torch.zeros(1, agent.config.latent.world_dim, device=device)
            )
            candidates = agent.generate_trajectories(latent, e1_prior, ticks)
            action = agent.select_action(candidates, ticks)
            action_idx = int(action.argmax(dim=-1).item())
            _, harm_signal, done, info, obs_dict = env.step(action_idx)
            ttype = info.get("transition_type", "none")

            opt.zero_grad()
            loss = agent.compute_prediction_loss()
            if (
                agent.latent_stack.resource_encoder is not None
                and latent.resource_prox_pred_r is not None
            ):
                prox_target_val = float(info.get("resource_field_at_agent", 0.0))
                prox_target = torch.tensor(
                    [[prox_target_val]], dtype=torch.float32, device=device
                )
                res_loss = agent.compute_resource_encoder_loss(prox_target, latent)
                loss = loss + res_loss
            if loss.requires_grad:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                opt.step()

            if ttype == "resource":
                drive_lvl = float(REEAgent.compute_drive_level(obs_body))
                agent.update_z_goal(float(harm_signal), drive_level=drive_lvl)

            if done:
                break

        manual_metrics = _maybe_run_manual_sleep(agent, arm_cfg)
        if manual_metrics:
            cumulative_sws_writes += float(manual_metrics.get("sws_n_writes", 0.0) or 0.0)
            cumulative_rem_rollouts += float(manual_metrics.get("rem_n_rollouts", 0.0) or 0.0)

    # P1 has no separate Phase A accounting because cycle_history is shared
    # with P0; we attribute all Phase A writes to P0 to avoid double-counting.
    if arm_cfg["use_sleep_loop"]:
        cumulative_sws_writes = 0.0
        cumulative_rem_rollouts = 0.0

    return {
        "p1_cumulative_sws_writes": cumulative_sws_writes,
        "p1_cumulative_rem_rollouts": cumulative_rem_rollouts,
    }


def run_evaluation(
    agent: REEAgent,
    env: CausalGridWorld,
    device: torch.device,
    n_eps: int,
    type_names: Tuple[str, ...],
) -> Dict:
    z_samples_consumption: List[np.ndarray] = []
    targets_consumption: List[int] = []
    z_samples_neighborhood: List[np.ndarray] = []
    targets_neighborhood: List[int] = []
    peak_drive_per_axis = np.zeros(env.n_resource_types, dtype=np.float32)

    for ep in range(n_eps):
        _, obs_dict = env.reset()
        agent.reset()  # eval: do NOT run sleep cycles (frozen substrate)
        for step in range(STEPS_PER_EPISODE):
            obs_body = obs_dict["body_state"].to(device)
            obs_world = obs_dict["world_state"].to(device)
            with torch.no_grad():
                latent = agent.sense(obs_body, obs_world)

            ticks = agent.clock.advance()
            with torch.no_grad():
                e1_prior = (
                    agent._e1_tick(latent) if ticks.get("e1_tick")
                    else torch.zeros(1, agent.config.latent.world_dim, device=device)
                )
                candidates = agent.generate_trajectories(latent, e1_prior, ticks)
                action = agent.select_action(candidates, ticks)
            action_idx = int(action.argmax(dim=-1).item())
            _, harm_signal, done, info, obs_dict = env.step(action_idx)
            ttype = info.get("transition_type", "none")

            if env.multi_resource_heterogeneity_enabled:
                peak_drive_per_axis = np.maximum(
                    peak_drive_per_axis, env._per_axis_drive
                )

            if env.multi_resource_heterogeneity_enabled and latent.z_resource is not None:
                neighborhood_label = neighborhood_dominant_type(obs_dict, type_names)
                if neighborhood_label >= 0:
                    z_samples_neighborhood.append(
                        latent.z_resource.detach().cpu().numpy().squeeze()
                    )
                    targets_neighborhood.append(neighborhood_label)

            if ttype == "resource":
                drive_lvl = float(REEAgent.compute_drive_level(obs_body))
                agent.update_z_goal(float(harm_signal), drive_level=drive_lvl)
                target_type = int(info.get("sd049_consumed_type_tag_this_tick", 0))
                if target_type > 0 and latent.z_resource is not None:
                    z_samples_consumption.append(
                        latent.z_resource.detach().cpu().numpy().squeeze()
                    )
                    targets_consumption.append(target_type - 1)

            if done:
                break

    probe_acc_consumption = identity_recovery_probe(
        z_samples_consumption, targets_consumption
    )
    probe_acc_neighborhood = identity_recovery_probe(
        z_samples_neighborhood, targets_neighborhood
    )
    return {
        "probe_acc_consumption": probe_acc_consumption,
        "n_identity_samples_consumption": len(targets_consumption),
        "probe_acc_neighborhood": probe_acc_neighborhood,
        "n_identity_samples_neighborhood": len(targets_neighborhood),
        "peak_per_axis_drive": [float(x) for x in peak_drive_per_axis.tolist()],
    }


def identity_recovery_probe(z_samples: List[np.ndarray], targets: List[int]) -> float:
    if len(z_samples) < 10:
        return 0.0
    X = np.stack(z_samples)
    y = np.array(targets)
    n = len(X)
    n_train = max(1, int(n * 0.7))
    perm = np.random.permutation(n)
    X_train, y_train = X[perm[:n_train]], y[perm[:n_train]]
    X_eval, y_eval = X[perm[n_train:]], y[perm[n_train:]]
    if len(X_eval) == 0 or len(np.unique(y_train)) < 2:
        return 0.0
    n_classes = int(max(y_train.max(), y_eval.max())) + 1
    z_dim = X_train.shape[1]
    probe = nn.Linear(z_dim, n_classes)
    opt = optim.Adam(probe.parameters(), lr=1e-2)
    Xt = torch.tensor(X_train, dtype=torch.float32)
    yt = torch.tensor(y_train, dtype=torch.long)
    for _ in range(200):
        opt.zero_grad()
        logits = probe(Xt)
        loss = F.cross_entropy(logits, yt)
        loss.backward()
        opt.step()
    with torch.no_grad():
        Xe = torch.tensor(X_eval, dtype=torch.float32)
        pred = probe(Xe).argmax(dim=-1).numpy()
        return float((pred == y_eval).mean())


def run_seed_arm(seed: int, arm_cfg: Dict, dry_run: bool) -> Dict:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cpu")

    n_p0 = 4 if dry_run else P0_EPISODES
    n_p1 = 2 if dry_run else P1_EPISODES
    n_eval = 2 if dry_run else EVAL_EPISODES

    env = make_env(seed)
    cfg = make_config(env, arm_cfg)
    agent = REEAgent(cfg)
    type_names = tuple(env.resource_type_names) if env.multi_resource_heterogeneity_enabled else tuple()

    p0_metrics = run_p0_training(agent, env, device, n_p0, CLASSIFIER_LOSS_WEIGHT, arm_cfg)
    p1_metrics = run_p1_training(agent, env, device, n_p1, arm_cfg)
    eval_metrics = run_evaluation(agent, env, device, n_eval, type_names)

    cumulative_sws = (
        p0_metrics["p0_cumulative_sws_writes"] + p1_metrics["p1_cumulative_sws_writes"]
    )
    cumulative_rem = (
        p0_metrics["p0_cumulative_rem_rollouts"] + p1_metrics["p1_cumulative_rem_rollouts"]
    )

    return {
        "seed": seed,
        "arm": arm_cfg["arm"],
        "sws_enabled": bool(arm_cfg["sws_enabled"]),
        "rem_enabled": bool(arm_cfg["rem_enabled"]),
        "use_sleep_loop": bool(arm_cfg["use_sleep_loop"]),
        "world_obs_dim": int(env.world_obs_dim),
        "n_resource_types": int(N_RESOURCE_TYPES),
        "classifier_loss_weight": float(CLASSIFIER_LOSS_WEIGHT),
        **p0_metrics,
        **p1_metrics,
        **eval_metrics,
        "cumulative_sws_writes": cumulative_sws,
        "cumulative_rem_rollouts": cumulative_rem,
    }


def aggregate(per_cell: List[Dict]) -> Dict[str, Dict]:
    bucket: Dict[str, Dict] = {}
    for r in per_cell:
        arm = r["arm"]
        if arm not in bucket:
            bucket[arm] = {
                "arm": arm,
                "sws_enabled": r["sws_enabled"],
                "rem_enabled": r["rem_enabled"],
                "use_sleep_loop": r["use_sleep_loop"],
                "n_seeds": 0,
                "probe_acc_consumption_mean": 0.0,
                "probe_acc_neighborhood_mean": 0.0,
                "p0_classifier_loss_first_q_mean": 0.0,
                "p0_classifier_loss_last_q_mean": 0.0,
                "cumulative_sws_writes_mean": 0.0,
                "cumulative_rem_rollouts_mean": 0.0,
                "n_identity_samples_consumption_total": 0,
                "n_identity_samples_neighborhood_total": 0,
            }
        b = bucket[arm]
        b["n_seeds"] += 1
        b["probe_acc_consumption_mean"] += r["probe_acc_consumption"]
        b["probe_acc_neighborhood_mean"] += r["probe_acc_neighborhood"]
        b["p0_classifier_loss_first_q_mean"] += r["p0_classifier_loss_first_quarter"]
        b["p0_classifier_loss_last_q_mean"] += r["p0_classifier_loss_last_quarter"]
        b["cumulative_sws_writes_mean"] += r["cumulative_sws_writes"]
        b["cumulative_rem_rollouts_mean"] += r["cumulative_rem_rollouts"]
        b["n_identity_samples_consumption_total"] += r["n_identity_samples_consumption"]
        b["n_identity_samples_neighborhood_total"] += r["n_identity_samples_neighborhood"]
    for arm, b in bucket.items():
        n = max(1, b["n_seeds"])
        b["probe_acc_consumption_mean"] /= n
        b["probe_acc_neighborhood_mean"] /= n
        b["p0_classifier_loss_first_q_mean"] /= n
        b["p0_classifier_loss_last_q_mean"] /= n
        b["cumulative_sws_writes_mean"] /= n
        b["cumulative_rem_rollouts_mean"] /= n
    return bucket


def evaluate_acceptance(agg: Dict[str, Dict]) -> Dict:
    a0 = agg["ARM_0_off"]
    a1 = agg["ARM_1_sd017_only"]
    a2 = agg["ARM_2_phase_a"]

    c0 = (a0["n_seeds"] > 0)
    c1a = a1["cumulative_sws_writes_mean"] >= C1A_SWS_WRITES_FLOOR
    c1b = a1["cumulative_rem_rollouts_mean"] >= C1B_REM_ROLLOUTS_FLOOR
    c2a = a2["cumulative_sws_writes_mean"] >= C2A_SWS_WRITES_FLOOR
    # C2b: ARM_2 fires fewer total cycles than ARM_1 (Phase A K=3 driver semantics).
    c2b = a2["cumulative_sws_writes_mean"] < a1["cumulative_sws_writes_mean"]

    c3a_arm1 = a1["probe_acc_neighborhood_mean"] >= C3A_PROBE_NEIGHBORHOOD_FLOOR
    c3a_arm2 = a2["probe_acc_neighborhood_mean"] >= C3A_PROBE_NEIGHBORHOOD_FLOOR
    c3a = c3a_arm1 or c3a_arm2

    c3b_arm1 = a1["p0_classifier_loss_last_q_mean"] < a1["p0_classifier_loss_first_q_mean"]
    c3b_arm2 = a2["p0_classifier_loss_last_q_mean"] < a2["p0_classifier_loss_first_q_mean"]
    c3b = c3b_arm1 or c3b_arm2

    sleep_arm_max_probe = max(
        a1["probe_acc_neighborhood_mean"], a2["probe_acc_neighborhood_mean"]
    )
    c4 = (sleep_arm_max_probe - a0["probe_acc_neighborhood_mean"]) >= C4_SLEEP_LIFT_FLOOR

    overall = c0 and c1a and c1b and c2a and c2b and (c3a or c3b) and c4
    return {
        "C0_arm0_runs_clean": bool(c0),
        "C1a_arm1_sws_writes_fire": bool(c1a),
        "C1b_arm1_rem_rollouts_fire": bool(c1b),
        "C2a_arm2_sws_writes_fire": bool(c2a),
        "C2b_arm2_fewer_fires_than_arm1": bool(c2b),
        "C3a_arm1_or_arm2_probe_acc_floor": bool(c3a),
        "C3b_arm1_or_arm2_classifier_converges": bool(c3b),
        "C4_sleep_lift_over_arm0": bool(c4),
        "all_pass": bool(overall),
    }


def main(dry_run: bool = False) -> int:
    print(f"[{EXPERIMENT_TYPE}] starting (dry_run={dry_run})")
    print(f"[{EXPERIMENT_TYPE}] supersedes_diagnostic={SUPERSEDES_DIAGNOSTIC}")
    print(f"[{EXPERIMENT_TYPE}] reef_enabled={REEF_ENABLED} hazard_food_attraction={HAZARD_FOOD_ATTRACTION}")
    print(f"[{EXPERIMENT_TYPE}] sleep_loop_K={SLEEP_LOOP_K} sws_steps={SWS_CONSOLIDATION_STEPS} rem_steps={REM_ATTRIBUTION_STEPS}")
    seeds = (SEEDS[0],) if dry_run else SEEDS
    per_cell: List[Dict] = []
    t0 = time.time()
    for seed in seeds:
        for arm_cfg in ARMS_CONFIG:
            arm_t0 = time.time()
            r = run_seed_arm(seed, arm_cfg, dry_run)
            per_cell.append(r)
            print(
                f"  seed={seed} arm={r['arm']:<18} "
                f"probe_neigh={r['probe_acc_neighborhood']:.3f} "
                f"probe_cons={r['probe_acc_consumption']:.3f} "
                f"clf_first/last={r['p0_classifier_loss_first_quarter']:.3f}/"
                f"{r['p0_classifier_loss_last_quarter']:.3f} "
                f"sws={r['cumulative_sws_writes']:.0f} rem={r['cumulative_rem_rollouts']:.0f} "
                f"elapsed={time.time()-arm_t0:.1f}s"
            )
    agg = aggregate(per_cell)
    acceptance = evaluate_acceptance(agg)
    elapsed = time.time() - t0
    outcome = "PASS" if acceptance["all_pass"] else "FAIL"

    print(f"[{EXPERIMENT_TYPE}] aggregates:")
    for arm in ("ARM_0_off", "ARM_1_sd017_only", "ARM_2_phase_a"):
        a = agg[arm]
        print(
            f"  {arm:<18} probe_neigh={a['probe_acc_neighborhood_mean']:.3f} "
            f"probe_cons={a['probe_acc_consumption_mean']:.3f} "
            f"clf_first/last={a['p0_classifier_loss_first_q_mean']:.3f}/"
            f"{a['p0_classifier_loss_last_q_mean']:.3f} "
            f"sws_mean={a['cumulative_sws_writes_mean']:.1f} "
            f"rem_mean={a['cumulative_rem_rollouts_mean']:.1f}"
        )
    print(f"[{EXPERIMENT_TYPE}] acceptance:")
    for k, v in acceptance.items():
        print(f"  {k}: {v}")
    print(f"[{EXPERIMENT_TYPE}] outcome={outcome} elapsed={elapsed:.1f}s")
    print(f"Done. Outcome: {outcome}")

    if dry_run:
        print(f"[{EXPERIMENT_TYPE}] dry-run complete; not writing manifest.")
        return 0

    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{timestamp}_v3"
    out_dir = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments" / EXPERIMENT_TYPE
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "supersedes_diagnostic": SUPERSEDES_DIAGNOSTIC,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "started_utc": datetime.utcnow().isoformat() + "Z",
        "outcome": outcome,
        "evidence_direction": "supports" if outcome == "PASS" else "weakens",
        "evidence_direction_per_claim": {
            "SD-049": "supports" if outcome == "PASS" else (
                "supports" if (acceptance["C3a_arm1_or_arm2_probe_acc_floor"] or acceptance["C3b_arm1_or_arm2_classifier_converges"]) else "weakens"
            ),
            "SD-015": "supports" if (outcome == "PASS" or acceptance["C3a_arm1_or_arm2_probe_acc_floor"]) else "weakens",
            "SD-017": "supports" if (acceptance["C1a_arm1_sws_writes_fire"] and acceptance["C2a_arm2_sws_writes_fire"]) else "weakens",
            "MECH-229": "supports" if outcome == "PASS" else "non_contributory",
            "MECH-230": "supports" if outcome == "PASS" else "non_contributory",
        },
        "elapsed_seconds": elapsed,
        "n_seeds": len(seeds),
        "p0_episodes": P0_EPISODES,
        "p1_episodes": P1_EPISODES,
        "eval_episodes": EVAL_EPISODES,
        "steps_per_episode": STEPS_PER_EPISODE,
        "lr": LR,
        "classifier_loss_weight": CLASSIFIER_LOSS_WEIGHT,
        "reef_enabled": REEF_ENABLED,
        "hazard_food_attraction": HAZARD_FOOD_ATTRACTION,
        "sleep_loop_K": SLEEP_LOOP_K,
        "sws_consolidation_steps": SWS_CONSOLIDATION_STEPS,
        "rem_attribution_steps": REM_ATTRIBUTION_STEPS,
        "arms": list(agg.values()),
        "per_seed_per_arm": per_cell,
        "acceptance": acceptance,
        "thresholds": {
            "C1a_sws_writes_floor": C1A_SWS_WRITES_FLOOR,
            "C1b_rem_rollouts_floor": C1B_REM_ROLLOUTS_FLOOR,
            "C2a_sws_writes_floor": C2A_SWS_WRITES_FLOOR,
            "C3a_probe_neighborhood_floor": C3A_PROBE_NEIGHBORHOOD_FLOOR,
            "C4_sleep_lift_floor": C4_SLEEP_LIFT_FLOOR,
        },
        "note": (
            "514f confirmed reef substrate fires (peak per-axis drives + 13.5k "
            "neighborhood samples) but classifier diverges in P0 with weight=0.1. "
            "This experiment turns on SD-017 SWS+REM consolidation between training "
            "episodes (ARM_1 manual run_sleep_cycle, ARM_2 SleepLoopManager K=3) "
            "to test whether offline schema etching recovers identity discrimination "
            "under the same waking pipeline that 514f failed."
        ),
    }
    out_path = out_dir / f"{run_id}.json"
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Result written to: {out_path}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Smoke run, no manifest.")
    args = parser.parse_args()
    sys.exit(main(dry_run=args.dry_run))
