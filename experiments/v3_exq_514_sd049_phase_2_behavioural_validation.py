#!/opt/local/bin/python3
"""V3-EXQ-514 -- SD-049 Phase 2 hybrid encoder behavioural validation.

Claim: SD-049 (environment.multi_resource_heterogeneity), Phase 2 (z_resource
encoder identity expansion -- Option C hybrid per verdict.md).
Status: candidate, v3_pending. Phase 1 substrate IMPLEMENTED 2026-05-03;
Phase 2 encoder IMPLEMENTED 2026-05-04.

Why this experiment exists
--------------------------
SD-049 Phase 2 lands the hybrid identity-aware z_resource encoder per the
2026-05-04 lit-pull verdict (Option C: shared trunk + identity-classifier
head + magnitude head, biology-anchored to Ballesta-Padoa-Schioppa 2019
labeled-line OFC + Quiroga 2005 sparse readouts + Schapiro 2017 hybrid CLS
bi-pathway architecture). V3-EXQ-513 confirmed Phase 1 substrate readiness
(13/13 PASS 2026-05-03). V3-EXQ-514 is the FULL CLAIM falsifiable test --
does the trained Phase 2 encoder produce identity-aware z_resource that
satisfies the SD-049 design doc acceptance criteria?

Phased training (per verdict.md):
  P0: enable use_identity_classifier=True; backprop identity cross-entropy +
      resource_prox MSE + downstream losses through trunk. Classifier head
      provides anti-collapse pull (Levi 2021 mitigation in spirit) + identity-
      discriminability supervision.
  P1: freeze identity_head.requires_grad_(False); continue trunk training
      under task losses. Trunk embedding develops similarity structure
      beyond what classifier supervision alone provides (Schapiro 2016
      distributed substrate).
  P2: evaluate identity-recovery (linear probe on z_resource) AND
      goal_resource_r AND per-axis drive evolution.

Pre-registered acceptance criteria
----------------------------------
4-arm substrate gradient sweep:

ARM_0 (off baseline -- single-type substrate, SD-049 OFF):
  - C0: goal_resource_r >= 0.06 (replicates V3-EXQ-322a baseline; sanity).

ARM_1 (2-type homeostatic; food + water; novelty distribution = 0):
  - C1a: world_obs_dim == 250 + 3*25 == 325 (per-type field views).
  - C1b: identity-classifier head trains successfully (loss decreases over P0).

ARM_2 (3-type default; food + water + novelty):
  - C2a: world_obs_dim == 325.
  - C2b: ARM_2 goal_resource_r >= 0.5 (target lift from EXQ-085x 0.066 baseline).
  - C2c: ARM_2 - ARM_0 goal_resource_r >= 0.2 (lift over single-type substrate).
  - C2d: ARM_2 identity-recovery linear-probe accuracy > 0.6 (z_resource
    carries identity; the load-bearing signal per verdict.md).
  - C2e: per-axis drive evolves over ARM_2 episodes (sanity matching
    V3-EXQ-513 C2c).

ARM_3 (5-type overshoot; food + water + novelty + shelter + social-proxy):
  - C3a: world_obs_dim == 250 + 5*25 == 375.
  - C3b: identity-classifier with n_types=5 trains successfully (no crash).

PASS = C0 AND C1a AND C1b AND C2a AND C2b AND C2c AND C2d AND C2e AND C3a AND C3b.

PASS reading: SD-049 Phase 2 hybrid encoder validated. The verdict.md
Option C architecture produces identity-aware z_resource that satisfies
the SD-049 design doc acceptance for ARM_2 lift over baseline. SD-015
becomes promotable (lift evidence on the enriched substrate); SD-049
v3_pending may be cleared pending governance review.

FAIL on C2b (goal_resource_r lift): falsifier branch row from verdict.md
applies. The verdict's 6-row interpretation grid maps the failure to a
specific architectural conclusion -- see verdict.md for the diagnostic
flowchart. The FAIL does NOT fall back to Option A; instead it flags
specific Phase 2 implementation hazards (under-trained trunk, classifier
dominating, class collapse, etc.).

FAIL on C2d (identity-recovery): the classifier head did not learn
identity-discriminative representations from the trunk. Likely cause:
P0 phase too short; recommendation is to extend P0 episodes and retest.

FAIL on both ARM_2 AND ARM_3 with similar magnitude: the substrate-
ceiling falsifier branch (per verdict.md row 6) applies. Routes MECH-229
to substrate_conditional with V4-1 multi-agent ecology dependency. This
is the parallel to SD-047's Woo/Spelke branch.

experiment_purpose = "evidence" (governance evidence; not just diagnostic).

Run with:
  /opt/local/bin/python3 experiments/v3_exq_514_sd049_phase_2_behavioural_validation.py
  /opt/local/bin/python3 experiments/v3_exq_514_sd049_phase_2_behavioural_validation.py --dry-run
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

EXPERIMENT_TYPE = "v3_exq_514_sd049_phase_2_behavioural_validation"
CLAIM_IDS = ["SD-049", "SD-015", "MECH-229", "MECH-230"]
EXPERIMENT_PURPOSE = "evidence"

SEEDS = [42, 43, 44]
# Phased training schedule (calibrated against V3-EXQ-322a baseline ~80 train + 30 eval
# eps and dry-run timing of ~150 sec/arm at 5+2+3 eps -> full run ~90 min total).
P0_EPISODES = 30   # supervised joint training (classifier + prox + downstream)
P1_EPISODES = 10   # frozen-classifier; trunk-only continuation
EVAL_EPISODES = 15
STEPS_PER_EPISODE = 200
LR = 1e-3

# Acceptance thresholds (pre-registered from verdict.md).
C0_GOAL_R_FLOOR = 0.06           # ARM_0 baseline replication floor
C2B_GOAL_R_FLOOR = 0.5           # ARM_2 goal_resource_r target
C2C_LIFT_FLOOR = 0.2             # ARM_2 - ARM_0 lift target
C2D_PROBE_ACCURACY_FLOOR = 0.6   # identity-recovery linear-probe accuracy
C2E_DRIVE_PEAK_FLOOR = 0.02      # per-axis drive must evolve

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
    """Build env per arm config. SD-049 OFF for ARM_0; ON for ARM_1/2/3."""
    kwargs = dict(
        size=10,
        num_hazards=2,
        num_resources=12,
        hazard_harm=0.02,
        resource_benefit=0.18,
        use_proxy_fields=True,
        seed=seed,
        proximity_benefit_scale=0.18,
    )
    if arm_cfg["sd049_enabled"]:
        kwargs.update(
            multi_resource_heterogeneity_enabled=True,
            per_axis_drive_enabled=True,
            n_resource_types=arm_cfg["n_resource_types"],
        )
        if arm_cfg.get("resource_type_distribution") is not None:
            kwargs["resource_type_distribution"] = arm_cfg["resource_type_distribution"]
        # ARM_3 5-type overrides
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
    """Build agent config per arm. Encoder identity classifier ON for ARM_1/2/3."""
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


def run_p0_training(
    agent: REEAgent,
    env: CausalGridWorld,
    device: torch.device,
    n_eps: int,
) -> Dict:
    """P0: joint training. ResourceEncoder trunk + classifier head + prox head
    + downstream task losses. Classifier head supervised on
    obs_dict["resource_type_at_agent"] when SD-049 multi-resource is on."""
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
            # SD-015 prox loss
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
            # SD-049 Phase 2 identity loss (non-zero only when classifier ON
            # and agent on a resource cell with target > 0).
            if getattr(agent.config.latent, "use_identity_classifier", False):
                target_type = int(info.get("sd049_consumed_type_tag_this_tick", 0))
                id_loss = agent.compute_resource_identity_loss(target_type, latent)
                loss = loss + id_loss
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
    """P1: freeze classifier head; continue trunk training under task losses
    + prox loss only. The trunk continues to develop similarity structure
    via the prox supervision and downstream task gradients."""
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
) -> Dict:
    """P2: evaluate goal_resource_r + identity-recovery linear-probe + per-axis
    drive evolution. No gradient updates."""
    cosine_sims: List[float] = []
    z_resource_samples: List[np.ndarray] = []
    identity_targets: List[int] = []
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

            # Track peak per-axis drive (SD-049 ARM_2/3 sanity check).
            if env.multi_resource_heterogeneity_enabled:
                peak_drive_per_axis = np.maximum(
                    peak_drive_per_axis, env._per_axis_drive
                )

            # On resource contact, seed z_goal AND record (z_resource, type)
            # for the identity-recovery linear probe.
            if ttype == "resource":
                drive_lvl = float(REEAgent.compute_drive_level(obs_body))
                agent.update_z_goal(float(harm_signal), drive_level=drive_lvl)
                target_type = int(info.get("sd049_consumed_type_tag_this_tick", 0))
                # Note: target_type was just-cleared in env.step() (cell tag
                # removed on consume). Use the BEFORE-step type the env cached;
                # in our case we read AFTER step but at first contact the type
                # is preserved via the contact event. For ARM_0 (no SD-049)
                # target_type will be 0 -- we skip those samples.
                if target_type > 0 and latent.z_resource is not None:
                    z_resource_samples.append(
                        latent.z_resource.detach().cpu().numpy().squeeze()
                    )
                    identity_targets.append(target_type - 1)  # 0-indexed

            # Compute goal_resource_r at every step where goal active and
            # z_resource available (matches V3-EXQ-322a methodology).
            if agent.goal_state is not None and agent.goal_state.is_active():
                with torch.no_grad():
                    z_g = agent.goal_state.z_goal
                    if z_g.ndim == 1:
                        z_g = z_g.unsqueeze(0)
                    z_ref = latent.z_resource if latent.z_resource is not None else latent.z_world
                    if z_ref.ndim == 1:
                        z_ref = z_ref.unsqueeze(0)
                    if z_g.shape[-1] == z_ref.shape[-1]:
                        cos = float(
                            F.cosine_similarity(z_g, z_ref, dim=-1).mean().item()
                        )
                        cosine_sims.append(cos)

            if done:
                break

    goal_resource_r = float(np.mean(cosine_sims)) if cosine_sims else 0.0
    identity_probe_accuracy = identity_recovery_probe(
        z_resource_samples, identity_targets
    )
    return {
        "goal_resource_r": goal_resource_r,
        "n_cosine_samples": len(cosine_sims),
        "identity_probe_accuracy": identity_probe_accuracy,
        "n_identity_samples": len(identity_targets),
        "peak_per_axis_drive": [float(x) for x in peak_drive_per_axis.tolist()],
    }


def identity_recovery_probe(
    z_samples: List[np.ndarray], targets: List[int]
) -> float:
    """Train a linear classifier on (z_resource, type) and return held-out
    accuracy. Uses 70/30 train/eval split. Returns 0 if insufficient data."""
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
        # Only one type seen during training; can't fit a multi-class probe.
        return 0.0
    # Train a small linear classifier in PyTorch (avoid sklearn dep).
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
    """Run one (seed, arm) cell through P0 -> P1 -> P2."""
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

    p0_metrics = run_p0_training(agent, env, device, n_p0)
    run_p1_training(agent, env, device, n_p1)
    eval_metrics = run_evaluation(agent, env, device, n_eval)
    return {
        "seed": seed,
        "arm": arm_cfg["arm"],
        "world_obs_dim": int(world_obs_dim),
        "n_resource_types": int(arm_cfg["n_resource_types"]),
        "use_identity_classifier": bool(arm_cfg["use_identity_classifier"]),
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
                "n_seeds": 0,
                "goal_resource_r_mean": 0.0,
                "identity_probe_accuracy_mean": 0.0,
                "p0_classifier_loss_first_q_mean": 0.0,
                "p0_classifier_loss_last_q_mean": 0.0,
                "peak_per_axis_drive_max": [0.0] * r["n_resource_types"],
                "n_cosine_samples_total": 0,
                "n_identity_samples_total": 0,
            }
        b = bucket[arm]
        b["n_seeds"] += 1
        b["goal_resource_r_mean"] += r["goal_resource_r"]
        b["identity_probe_accuracy_mean"] += r["identity_probe_accuracy"]
        b["p0_classifier_loss_first_q_mean"] += r["p0_classifier_loss_first_quarter"]
        b["p0_classifier_loss_last_q_mean"] += r["p0_classifier_loss_last_quarter"]
        for i in range(r["n_resource_types"]):
            b["peak_per_axis_drive_max"][i] = max(
                b["peak_per_axis_drive_max"][i], r["peak_per_axis_drive"][i]
            )
        b["n_cosine_samples_total"] += r["n_cosine_samples"]
        b["n_identity_samples_total"] += r["n_identity_samples"]
    for arm, b in bucket.items():
        n = max(1, b["n_seeds"])
        b["goal_resource_r_mean"] /= n
        b["identity_probe_accuracy_mean"] /= n
        b["p0_classifier_loss_first_q_mean"] /= n
        b["p0_classifier_loss_last_q_mean"] /= n
    return bucket


def evaluate_acceptance(agg: Dict[str, Dict]) -> Dict:
    a0 = agg["ARM_0_off"]
    a1 = agg["ARM_1_2type"]
    a2 = agg["ARM_2_3type"]
    a3 = agg["ARM_3_5type"]
    c0 = a0["goal_resource_r_mean"] >= C0_GOAL_R_FLOOR
    c1a = a1["world_obs_dim"] == 250 + 3 * 25
    # Loss-decreasing as classifier-trained signature: last_q < first_q OR both < ln(n_types).
    expected_random_loss = float(np.log(a1["n_resource_types"]))
    c1b = (
        a1["p0_classifier_loss_last_q_mean"]
        < max(0.5 * expected_random_loss, a1["p0_classifier_loss_first_q_mean"] - 0.05)
    )
    c2a = a2["world_obs_dim"] == 250 + 3 * 25
    c2b = a2["goal_resource_r_mean"] >= C2B_GOAL_R_FLOOR
    c2c = (a2["goal_resource_r_mean"] - a0["goal_resource_r_mean"]) >= C2C_LIFT_FLOOR
    c2d = a2["identity_probe_accuracy_mean"] >= C2D_PROBE_ACCURACY_FLOOR
    c2e = float(np.max(a2["peak_per_axis_drive_max"])) > C2E_DRIVE_PEAK_FLOOR
    c3a = a3["world_obs_dim"] == 250 + 5 * 25
    c3b = a3["p0_classifier_loss_last_q_mean"] > 0.0  # Did fire; no crash signature.

    overall = (
        c0 and c1a and c1b and c2a and c2b and c2c and c2d and c2e and c3a and c3b
    )
    return {
        "C0_arm0_baseline_replicates": bool(c0),
        "C1a_arm1_world_obs_dim_325": bool(c1a),
        "C1b_arm1_classifier_trains": bool(c1b),
        "C2a_arm2_world_obs_dim_325": bool(c2a),
        "C2b_arm2_goal_resource_r_target": bool(c2b),
        "C2c_arm2_lift_over_arm0": bool(c2c),
        "C2d_arm2_identity_probe_accuracy": bool(c2d),
        "C2e_arm2_per_axis_drive_evolves": bool(c2e),
        "C3a_arm3_world_obs_dim_375": bool(c3a),
        "C3b_arm3_classifier_did_fire": bool(c3b),
        "all_pass": bool(overall),
    }


def main(dry_run: bool = False) -> int:
    print(f"[{EXPERIMENT_TYPE}] starting (dry_run={dry_run})")
    seeds = (SEEDS[0],) if dry_run else SEEDS
    per_cell: List[Dict] = []
    t0 = time.time()
    for seed in seeds:
        for arm_cfg in ARMS_CONFIG:
            arm_t0 = time.time()
            r = run_seed_arm(seed, arm_cfg, dry_run)
            per_cell.append(r)
            print(
                f"  seed={seed} arm={r['arm']:<12} obs_dim={r['world_obs_dim']:3d} "
                f"goal_r={r['goal_resource_r']:.3f} "
                f"probe_acc={r['identity_probe_accuracy']:.3f} "
                f"clf_loss(first/last quartile)="
                f"{r['p0_classifier_loss_first_quarter']:.3f}/"
                f"{r['p0_classifier_loss_last_quarter']:.3f} "
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
            f"  {arm:<12} obs_dim={a['world_obs_dim']:3d} "
            f"goal_r={a['goal_resource_r_mean']:.3f} "
            f"probe_acc={a['identity_probe_accuracy_mean']:.3f} "
            f"clf_loss(first/last)="
            f"{a['p0_classifier_loss_first_q_mean']:.3f}/"
            f"{a['p0_classifier_loss_last_q_mean']:.3f} "
            f"peak_drive={[round(x, 3) for x in a['peak_per_axis_drive_max']]}"
        )
    print(f"[{EXPERIMENT_TYPE}] acceptance:")
    for k, v in acceptance.items():
        print(f"  {k}: {v}")
    print(f"[{EXPERIMENT_TYPE}] outcome={outcome} elapsed={elapsed:.1f}s")

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
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "started_utc": datetime.utcnow().isoformat() + "Z",
        "outcome": outcome,
        "evidence_direction": "supports" if outcome == "PASS" else "weakens",
        "evidence_direction_per_claim": {
            "SD-049": "supports" if outcome == "PASS" else "weakens",
            "SD-015": "supports" if (outcome == "PASS" or acceptance["C2b_arm2_goal_resource_r_target"]) else "weakens",
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
        "arms": list(agg.values()),
        "per_seed_per_arm": per_cell,
        "acceptance": acceptance,
        "thresholds": {
            "C0_goal_r_floor": C0_GOAL_R_FLOOR,
            "C2b_goal_r_floor": C2B_GOAL_R_FLOOR,
            "C2c_lift_floor": C2C_LIFT_FLOOR,
            "C2d_probe_accuracy_floor": C2D_PROBE_ACCURACY_FLOOR,
            "C2e_drive_peak_floor": C2E_DRIVE_PEAK_FLOOR,
        },
        "verdict_provenance": (
            "REE_assembly/evidence/literature/"
            "targeted_review_sd_049_encoder_identity_expansion/verdict.md"
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
