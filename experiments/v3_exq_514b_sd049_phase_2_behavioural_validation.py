#!/opt/local/bin/python3
"""V3-EXQ-514b -- SD-049 Phase 2 behavioural validation, SIGTERM fix.

Supersedes: V3-EXQ-514a (ERROR 2026-05-04T17:12Z, exit code -15 SIGTERM).

Root cause of EXQ-514a ERROR:
  Process killed by SIGTERM after 108 minutes on ree-cloud-1 (Hetzner CX22,
  2 shared vCPU, 14.2 steps/sec). Estimated runtime was ~231 minutes:
  660 episodes x 300 steps/ep / 14.2 steps/sec / 60 = 231 min. The cloud
  provider's process timeout fired at ~108 minutes. No code crash occurred.

Fix:
  Routed to DLAPTOP-4.local (macbook, ~33 steps/sec, ~0.15 min/ep at 300
  steps). Estimated runtime: 660 episodes x 0.15 min/ep = 99 minutes
  (queue entry uses estimated_minutes=120 for 20% buffer). No script
  changes relative to EXQ-514a -- the scientific instrumentation is correct.

Claim: SD-049, SD-015, MECH-229, MECH-230 (Phase 2 hybrid encoder validation).
Status: SD-049 Phase 2 IMPLEMENTED 2026-05-04 (commit 32b45a3 in ree-v3).
Lit-pull provenance: REE_assembly/evidence/literature/
targeted_review_sd_049_encoder_identity_expansion/verdict.md (Option C
hybrid, confidence 0.78).

Why this experiment supersedes V3-EXQ-514
------------------------------------------
V3-EXQ-514 ran to completion with 7/10 acceptance criteria PASS but
3 FAILs that, on inspection, reflect instrumentation rather than
architectural failure:

(1) **Contact-rate too sparse for the linear probe.** The agent contacted
    resources at ~0.07% rate (8-9 contacts per 11000-step run on ARM_2);
    the 3-class linear probe got only 5-6 training samples per fit, and
    the probe code's "<2 unique classes -> return 0.0" guard fired
    silently across all seeds. probe_acc=0.000 is a data-density failure,
    not a representation failure.

(2) **goal_resource_r metric saturates at ~1.0 across ALL arms.**
    Cosine(z_goal, z_resource) is measured at every step where the goal
    is active. At seeding tick z_goal := alpha * z_resource + (1-alpha) * z_goal_prev,
    so cosine starts ~1.0; between seeding events neither z_goal nor
    z_resource changes much, so cosine stays ~1.0 across thousands of
    samples. The metric is structurally degenerate at the per-step
    granularity. C2c ARM_2 - ARM_0 lift cannot fire when both arms are
    at the metric ceiling.

(3) **Classifier loss diverges (4.7 -> 10.1 last-quartile in ARM_2).**
    Joint training of trunk + classifier head + downstream task losses
    is unstable when the classifier loss has unit weight against losses
    of much larger scale (E1/E3/prox losses). Standard mitigation: weight
    the auxiliary loss down so the supervised pull is informative without
    dominating.

This script implements three fixes corresponding to the three failure
modes:

(A) **Higher contact rate.** Smaller grid (8x8), more resources (15),
    no hazards (agent dies less, finds more), longer episodes (300
    steps). Target: contact rate >= 1% so the probe gets enough samples.

(B) **Better probe instrumentation.** Two probe variants reported:
    (i) `probe_acc_consumption`: classic V3-EXQ-514 methodology --
        sampled at consumption events. Sparse but matches the supervision
        target exactly. Reported for comparability.
    (ii) `probe_acc_neighborhood`: NEW. Sampled at EVERY eval step,
         labeled by argmax over per-type field views (which type is
         dominant in the agent's 5x5 neighborhood). Dense supervision
         signal; tests whether z_resource carries identity information
         the encoder picked up from the world_obs structure. The
         load-bearing acceptance test for V3-EXQ-514a.

(C) **Classifier loss weight knob.** classifier_loss_weight=0.1 default
    (down from V3-EXQ-514 implicit 1.0). The classifier head still pulls
    on the trunk during P0 -- this is the verdict's architectural
    commitment -- but the pull is calibrated against the other losses
    so joint training stays stable.

Pre-registered acceptance criteria (revised)
--------------------------------------------
ARM_0 (off baseline -- single-type substrate, SD-049 OFF, classifier OFF):
  - C0: runs to completion without crash. (Probe is undefined for
    single-type substrate; classifier disabled. C0 is a sanity check
    that the experiment harness handles the OFF arm cleanly.)

ARM_1 (2-type homeostatic; food + water; novelty distribution = 0):
  - C1a: world_obs_dim == 250 + 3*25 == 325.
  - C1b: classifier P0 last-quartile loss < first-quartile loss OR
    < 0.5 * ln(n_types) (true convergence test, NOT divergence test).

ARM_2 (3-type default; food + water + novelty):
  - C2a: world_obs_dim == 325.
  - C2b: ARM_2 probe_acc_neighborhood > 0.6 (the load-bearing
    identity-recovery test from verdict.md, dense sampling).
  - C2c: ARM_2 n_identity_samples_consumption >= 30 (data sufficiency
    sanity; if < 30, no probe fit attempted and probe_acc reads 0.0
    artifactually).
  - C2d: ARM_2 peak per-axis drive > 0.02 (sanity matching V3-EXQ-513).

ARM_3 (5-type overshoot; food + water + novelty + shelter + social):
  - C3a: world_obs_dim == 250 + 5*25 == 375.
  - C3b: classifier did fire (P0 last-quartile loss > 0; no crash).

PASS = C0 AND C1a AND C1b AND C2a AND C2b AND C2c AND C2d AND C3a AND C3b.

PASS reading: SD-049 Phase 2 hybrid encoder validated. SD-015 promotable.
SD-049 v3_pending may be cleared (pending governance review).

FAIL on C2b but C2c PASS (sufficient samples but probe still fails):
  the encoder did NOT learn identity-discriminative directions on the
  trunk. Likely cause: classifier_loss_weight too small (signal too weak)
  OR fundamental architectural mismatch. Recommendation: V3-EXQ-514c
  with classifier_loss_weight=0.5 OR 1.0 + early stopping on classifier
  loss to defend against divergence. This is verdict.md row 2/3
  territory.

FAIL on C2b AND C3 (joint failure across substrate scales):
  the substrate-ceiling falsifier branch (verdict.md row 6) applies.
  Routes MECH-229 to substrate_conditional with V4-1 multi-agent ecology
  dependency. This is the parallel to SD-047's Woo/Spelke branch.

FAIL on C1b only:
  classifier still diverging at weight=0.1. Lower weight further (0.01)
  OR detach trunk for classifier-only loss in V3-EXQ-514c. This narrows
  the architectural choice from Option C concat-trunk-shaping toward
  the verdict's "Option 2 single-output-with-supervision" instantiation
  (head trains itself; trunk shaped only by other losses).

experiment_purpose = "evidence" (governance evidence; supersedes V3-EXQ-514a).
supersedes: V3-EXQ-514a

Run with:
  /opt/local/bin/python3 experiments/v3_exq_514b_sd049_phase_2_behavioural_validation.py
  /opt/local/bin/python3 experiments/v3_exq_514b_sd049_phase_2_behavioural_validation.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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

EXPERIMENT_TYPE = "v3_exq_514b_sd049_phase_2_behavioural_validation"
CLAIM_IDS = ["SD-049", "SD-015", "MECH-229", "MECH-230"]
EXPERIMENT_PURPOSE = "evidence"
SUPERSEDES = "V3-EXQ-514a"

SEEDS = [42, 43, 44]
P0_EPISODES = 30
P1_EPISODES = 10
EVAL_EPISODES = 15
STEPS_PER_EPISODE = 300  # up from 200 (more time to find resources)
LR = 1e-3

# Fix (C): classifier loss weight knob. Default 0.1 keeps the classifier's
# supervised pull on the trunk (verdict's Option C architectural commitment)
# while preventing divergence against the much larger E1/E3/prox losses.
CLASSIFIER_LOSS_WEIGHT = 0.1

# Acceptance thresholds (pre-registered).
C1B_LOSS_DECREASE_FRAC = 0.95   # last-q must be < first-q (allow 5% noise)
C2B_PROBE_NEIGHBORHOOD_FLOOR = 0.6
C2C_N_ID_SAMPLES_FLOOR = 30
C2D_DRIVE_PEAK_FLOOR = 0.02

# Fix (A): higher contact rate via env tuning.
ARMS_CONFIG: List[Dict] = [
    dict(
        arm="ARM_0_off",
        sd049_enabled=False,
        n_resource_types=3,
        resource_type_distribution=None,
        use_identity_classifier=False,
    ),
    dict(
        arm="ARM_1_2type",
        sd049_enabled=True,
        n_resource_types=3,
        resource_type_distribution=(1.0, 1.0, 0.0),
        use_identity_classifier=True,
    ),
    dict(
        arm="ARM_2_3type",
        sd049_enabled=True,
        n_resource_types=3,
        resource_type_distribution=None,
        use_identity_classifier=True,
    ),
    dict(
        arm="ARM_3_5type",
        sd049_enabled=True,
        n_resource_types=5,
        resource_type_distribution=None,
        use_identity_classifier=True,
        resource_type_names=("food", "water", "novelty", "shelter", "social"),
        resource_type_drive_axes=(
            "hunger", "thirst", "curiosity", "warmth", "affiliation"
        ),
        resource_type_benefit_curves=(
            "sigmoidal_saturating",
            "sharp_saturation",
            "novelty_decay",
            "sigmoidal_saturating",
            "sigmoidal_saturating",
        ),
        per_axis_drive_decay=(0.001, 0.0015, 0.0005, 0.0008, 0.0008),
    ),
]


def make_env(seed: int, arm_cfg: Dict) -> CausalGridWorld:
    """Build env per arm config with high-contact-rate tuning."""
    kwargs = dict(
        size=8,                # smaller grid (was 10)
        num_hazards=0,         # no hazards (was 2) -- max contact rate; agent doesn't die
        num_resources=15,      # more resources (was 12)
        hazard_harm=0.01,
        resource_benefit=0.18,
        use_proxy_fields=True,
        seed=seed,
        proximity_benefit_scale=0.18,
        resource_respawn_on_consume=True,  # SD-012 respawn so resources don't deplete to zero
    )
    if arm_cfg["sd049_enabled"]:
        kwargs.update(
            multi_resource_heterogeneity_enabled=True,
            per_axis_drive_enabled=True,
            n_resource_types=arm_cfg["n_resource_types"],
        )
        if arm_cfg.get("resource_type_distribution") is not None:
            kwargs["resource_type_distribution"] = arm_cfg["resource_type_distribution"]
        for k in (
            "resource_type_names",
            "resource_type_drive_axes",
            "resource_type_benefit_curves",
            "per_axis_drive_decay",
        ):
            if k in arm_cfg:
                kwargs[k] = arm_cfg[k]
    return CausalGridWorld(**kwargs)


def make_config(env: CausalGridWorld, arm_cfg: Dict) -> REEConfig:
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        alpha_world=0.9,
        world_dim=32,
        drive_weight=2.0,
        z_goal_enabled=True,
    )
    cfg.latent.use_resource_encoder = True
    cfg.latent.z_resource_dim = 32
    cfg.goal.goal_dim = cfg.latent.world_dim
    if arm_cfg["use_identity_classifier"]:
        cfg.latent.use_identity_classifier = True
        cfg.latent.identity_classifier_n_types = arm_cfg["n_resource_types"]
    return cfg


def neighborhood_dominant_type(obs_dict: Dict, type_names: Tuple[str, ...]) -> int:
    """Fix (B): label z_resource at every step by the type whose 5x5 field
    view max value is largest in the agent's neighborhood. Returns 0..n_types-1
    when at least one type-field is non-trivially non-zero, or -1 (no
    discriminable signal). When SD-049 OFF, returns -1."""
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
        # No type has meaningful neighborhood presence -- skip this sample.
        return -1
    return int(np.argmax(field_maxes))


def run_p0_training(
    agent: REEAgent,
    env: CausalGridWorld,
    device: torch.device,
    n_eps: int,
    classifier_loss_weight: float,
) -> Dict:
    """P0: joint training with classifier_loss_weight scaling on the
    identity loss term."""
    opt = optim.Adam(agent.parameters(), lr=LR)
    classifier_losses: List[float] = []
    prox_losses: List[float] = []
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
                prox_losses.append(float(res_loss.item()))
            if getattr(agent.config.latent, "use_identity_classifier", False):
                target_type = int(info.get("sd049_consumed_type_tag_this_tick", 0))
                id_loss = agent.compute_resource_identity_loss(target_type, latent)
                # Fix (C): scale by classifier_loss_weight so the supervised pull
                # is informative without dominating the trunk gradient.
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
    }


def run_p1_training(
    agent: REEAgent,
    env: CausalGridWorld,
    device: torch.device,
    n_eps: int,
) -> None:
    """P1: freeze classifier head; continue trunk training under task losses."""
    if (
        agent.latent_stack.resource_encoder is not None
        and agent.latent_stack.resource_encoder.identity_head is not None
    ):
        for p in agent.latent_stack.resource_encoder.identity_head.parameters():
            p.requires_grad_(False)
    opt = optim.Adam(
        [p for p in agent.parameters() if p.requires_grad], lr=LR
    )
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


def run_evaluation(
    agent: REEAgent,
    env: CausalGridWorld,
    device: torch.device,
    n_eps: int,
    type_names: Tuple[str, ...],
) -> Dict:
    """P2 eval: collect both probe-target variants + per-axis drive evolution.

    Fix (B): TWO probe targets reported.
      probe_acc_consumption: V3-EXQ-514 methodology -- sampled at consumption
        events. Sparse but matches supervision target exactly.
      probe_acc_neighborhood: NEW -- sampled at every step, labeled by argmax
        over per-type field views. Dense; tests whether z_resource carries
        identity information from the world_obs structure. The load-bearing
        acceptance test.
    """
    # Consumption-target probe samples (V3-EXQ-514 methodology, kept for comparison)
    z_samples_consumption: List[np.ndarray] = []
    targets_consumption: List[int] = []
    # Neighborhood-target probe samples (primary methodology)
    z_samples_neighborhood: List[np.ndarray] = []
    targets_neighborhood: List[int] = []

    peak_drive_per_axis = np.zeros(env.n_resource_types, dtype=np.float32)

    for ep in range(n_eps):
        _, obs_dict = env.reset()
        agent.reset()
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

            # Per-axis drive sanity
            if env.multi_resource_heterogeneity_enabled:
                peak_drive_per_axis = np.maximum(
                    peak_drive_per_axis, env._per_axis_drive
                )

            # Neighborhood-target: every step where SD-049 emits per-type field views.
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
                # Consumption-target: at consumption tick (matches V3-EXQ-514 methodology)
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


def identity_recovery_probe(
    z_samples: List[np.ndarray], targets: List[int]
) -> float:
    """Train a linear classifier on (z_resource, type) and return held-out
    accuracy. Uses 70/30 train/eval split."""
    if len(z_samples) < 10:
        return 0.0
    X = np.stack(z_samples)
    y = np.array(targets)
    n = len(X)
    n_train = max(1, int(n * 0.7))
    perm = np.random.permutation(n)
    X_train, y_train = X[perm[:n_train]], y[perm[:n_train]]
    X_eval, y_eval = X[perm[n_train:]], y[perm[n_train:]]
    if len(X_eval) == 0:
        return 0.0
    if len(np.unique(y_train)) < 2:
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

    n_p0 = 5 if dry_run else P0_EPISODES
    n_p1 = 2 if dry_run else P1_EPISODES
    n_eval = 3 if dry_run else EVAL_EPISODES

    env = make_env(seed, arm_cfg)
    cfg = make_config(env, arm_cfg)
    agent = REEAgent(cfg)
    world_obs_dim = env.world_obs_dim
    type_names = tuple(env.resource_type_names) if env.multi_resource_heterogeneity_enabled else tuple()

    p0_metrics = run_p0_training(agent, env, device, n_p0, CLASSIFIER_LOSS_WEIGHT)
    run_p1_training(agent, env, device, n_p1)
    eval_metrics = run_evaluation(agent, env, device, n_eval, type_names)
    return {
        "seed": seed,
        "arm": arm_cfg["arm"],
        "world_obs_dim": int(world_obs_dim),
        "n_resource_types": int(arm_cfg["n_resource_types"]),
        "use_identity_classifier": bool(arm_cfg["use_identity_classifier"]),
        "classifier_loss_weight": float(CLASSIFIER_LOSS_WEIGHT),
        **p0_metrics,
        **eval_metrics,
    }


def aggregate(per_cell: List[Dict]) -> Dict[str, Dict]:
    bucket: Dict[str, Dict] = {}
    for r in per_cell:
        arm = r["arm"]
        if arm not in bucket:
            bucket[arm] = {
                "arm": arm,
                "world_obs_dim": r["world_obs_dim"],
                "n_resource_types": r["n_resource_types"],
                "use_identity_classifier": r["use_identity_classifier"],
                "classifier_loss_weight": r["classifier_loss_weight"],
                "n_seeds": 0,
                "probe_acc_consumption_mean": 0.0,
                "probe_acc_neighborhood_mean": 0.0,
                "p0_classifier_loss_first_q_mean": 0.0,
                "p0_classifier_loss_last_q_mean": 0.0,
                "peak_per_axis_drive_max": [0.0] * r["n_resource_types"],
                "n_identity_samples_consumption_total": 0,
                "n_identity_samples_neighborhood_total": 0,
            }
        b = bucket[arm]
        b["n_seeds"] += 1
        b["probe_acc_consumption_mean"] += r["probe_acc_consumption"]
        b["probe_acc_neighborhood_mean"] += r["probe_acc_neighborhood"]
        b["p0_classifier_loss_first_q_mean"] += r["p0_classifier_loss_first_quarter"]
        b["p0_classifier_loss_last_q_mean"] += r["p0_classifier_loss_last_quarter"]
        for i in range(r["n_resource_types"]):
            b["peak_per_axis_drive_max"][i] = max(
                b["peak_per_axis_drive_max"][i], r["peak_per_axis_drive"][i]
            )
        b["n_identity_samples_consumption_total"] += r["n_identity_samples_consumption"]
        b["n_identity_samples_neighborhood_total"] += r["n_identity_samples_neighborhood"]
    for arm, b in bucket.items():
        n = max(1, b["n_seeds"])
        b["probe_acc_consumption_mean"] /= n
        b["probe_acc_neighborhood_mean"] /= n
        b["p0_classifier_loss_first_q_mean"] /= n
        b["p0_classifier_loss_last_q_mean"] /= n
    return bucket


def evaluate_acceptance(agg: Dict[str, Dict]) -> Dict:
    a0 = agg["ARM_0_off"]
    a1 = agg["ARM_1_2type"]
    a2 = agg["ARM_2_3type"]
    a3 = agg["ARM_3_5type"]

    # ARM_0 just needs to run cleanly; no architectural assertion.
    c0 = (a0["n_seeds"] > 0)
    c1a = a1["world_obs_dim"] == 250 + 3 * 25
    # C1b: convergence (last < first*0.95) OR last < 0.5*ln(n_types) (better than half-random).
    expected_random_loss = float(np.log(a1["n_resource_types"]))
    c1b = (
        a1["p0_classifier_loss_last_q_mean"] < a1["p0_classifier_loss_first_q_mean"] * C1B_LOSS_DECREASE_FRAC
        or a1["p0_classifier_loss_last_q_mean"] < 0.5 * expected_random_loss
    )
    c2a = a2["world_obs_dim"] == 250 + 3 * 25
    c2b = a2["probe_acc_neighborhood_mean"] >= C2B_PROBE_NEIGHBORHOOD_FLOOR
    c2c = a2["n_identity_samples_consumption_total"] >= C2C_N_ID_SAMPLES_FLOOR
    c2d = float(np.max(a2["peak_per_axis_drive_max"])) > C2D_DRIVE_PEAK_FLOOR
    c3a = a3["world_obs_dim"] == 250 + 5 * 25
    c3b = a3["p0_classifier_loss_last_q_mean"] > 0.0

    overall = c0 and c1a and c1b and c2a and c2b and c2c and c2d and c3a and c3b
    return {
        "C0_arm0_runs_clean": bool(c0),
        "C1a_arm1_world_obs_dim_325": bool(c1a),
        "C1b_arm1_classifier_converges": bool(c1b),
        "C2a_arm2_world_obs_dim_325": bool(c2a),
        "C2b_arm2_probe_acc_neighborhood": bool(c2b),
        "C2c_arm2_n_identity_samples_consumption": bool(c2c),
        "C2d_arm2_per_axis_drive_evolves": bool(c2d),
        "C3a_arm3_world_obs_dim_375": bool(c3a),
        "C3b_arm3_classifier_did_fire": bool(c3b),
        "all_pass": bool(overall),
    }


def main(dry_run: bool = False) -> int:
    print(f"[{EXPERIMENT_TYPE}] starting (dry_run={dry_run})")
    print(f"[{EXPERIMENT_TYPE}] supersedes {SUPERSEDES}")
    print(f"[{EXPERIMENT_TYPE}] classifier_loss_weight={CLASSIFIER_LOSS_WEIGHT}")
    seeds = (SEEDS[0],) if dry_run else SEEDS
    per_cell: List[Dict] = []
    t0 = time.time()
    for seed in seeds:
        for arm_cfg in ARMS_CONFIG:
            arm_t0 = time.time()
            r = run_seed_arm(seed, arm_cfg, dry_run)
            per_cell.append(r)
            print(
                f"  seed={seed} arm={r['arm']:<14} obs_dim={r['world_obs_dim']:3d} "
                f"probe_neigh={r['probe_acc_neighborhood']:.3f} "
                f"probe_cons={r['probe_acc_consumption']:.3f} "
                f"clf_first/last={r['p0_classifier_loss_first_quarter']:.3f}/"
                f"{r['p0_classifier_loss_last_quarter']:.3f} "
                f"n_id_cons={r['n_identity_samples_consumption']} "
                f"n_id_neigh={r['n_identity_samples_neighborhood']} "
                f"peak_drive={[round(x, 3) for x in r['peak_per_axis_drive']]} "
                f"elapsed={time.time()-arm_t0:.1f}s"
            )
    agg = aggregate(per_cell)
    acceptance = evaluate_acceptance(agg)
    elapsed = time.time() - t0
    outcome = "PASS" if acceptance["all_pass"] else "FAIL"

    print(f"[{EXPERIMENT_TYPE}] aggregates:")
    for arm in ("ARM_0_off", "ARM_1_2type", "ARM_2_3type", "ARM_3_5type"):
        a = agg[arm]
        print(
            f"  {arm:<14} obs_dim={a['world_obs_dim']:3d} "
            f"probe_neigh={a['probe_acc_neighborhood_mean']:.3f} "
            f"probe_cons={a['probe_acc_consumption_mean']:.3f} "
            f"clf_first/last={a['p0_classifier_loss_first_q_mean']:.3f}/"
            f"{a['p0_classifier_loss_last_q_mean']:.3f} "
            f"n_id_cons={a['n_identity_samples_consumption_total']} "
            f"n_id_neigh={a['n_identity_samples_neighborhood_total']} "
            f"peak_drive={[round(x, 3) for x in a['peak_per_axis_drive_max']]}"
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
        "supersedes": SUPERSEDES,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "started_utc": datetime.utcnow().isoformat() + "Z",
        "outcome": outcome,
        "evidence_direction": "supports" if outcome == "PASS" else "weakens",
        "evidence_direction_per_claim": {
            "SD-049": "supports" if outcome == "PASS" else "weakens",
            "SD-015": "supports" if (outcome == "PASS" or acceptance["C2b_arm2_probe_acc_neighborhood"]) else "weakens",
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
        "arms": list(agg.values()),
        "per_seed_per_arm": per_cell,
        "acceptance": acceptance,
        "thresholds": {
            "C1b_loss_decrease_frac": C1B_LOSS_DECREASE_FRAC,
            "C2b_probe_neighborhood_floor": C2B_PROBE_NEIGHBORHOOD_FLOOR,
            "C2c_n_id_samples_floor": C2C_N_ID_SAMPLES_FLOOR,
            "C2d_drive_peak_floor": C2D_DRIVE_PEAK_FLOOR,
        },
        "verdict_provenance": (
            "REE_assembly/evidence/literature/"
            "targeted_review_sd_049_encoder_identity_expansion/verdict.md"
        ),
        "supersedes_note": (
            "V3-EXQ-514a errored with exit code -15 (SIGTERM) on ree-cloud-1 "
            "after 108 minutes. Root cause: cloud provider process timeout. "
            "Estimated runtime was ~231 minutes (660 eps x 0.35 min/ep); "
            "timeout fired at ~108 minutes. No code error. Fix: routed to "
            "DLAPTOP-4.local (macbook, ~99 min estimated). Script unchanged."
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
