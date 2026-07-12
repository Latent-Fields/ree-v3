#!/opt/local/bin/python3
"""V3-EXQ-514l -- SD-049 Phase 3 MECH-229 wanting/liking behavioral dissociation retest.

MECH-229 retest on SD-049 Phase 3 substrate (MECH-295 axis-matched consumer cascade).
Supersedes V3-EXQ-514k which FAILed with "row1b_joint_identity_probe_weak" on 2026-06-01.

MECH-229 was demoted active -> provisional on 2026-06-01 with pending_retest_after_substrate=true.
Prior PASSes (EXQ-074f, 234, 354) were on single-resource substrate (degenerate test).
V3-EXQ-514j FAILED with "row1b_joint_identity_probe_weak" on 2026-05-20.
V3-EXQ-514k FAILED with the same issue on 2026-06-01.

Phase 3 changes vs 514j:
  - use_sd049_per_axis_consumer_cascade=True on ARM_1/2/3 (MECH-295 axis-matched routing).
  - claim_ids: SD-049, SD-015, MECH-229, MECH-230 (drops MECH-307; focus on behavioral test).
  - Reef + phased P0/P1/P2 + drive_floor=0.9 + SP-CEM main-path unchanged from 514j.

Pre-registered PASS (514j reef criteria):
  C0-C3b: same as V3-EXQ-514f (probe_acc_neighborhood, classifier, obs dims).
  C4: ARM_2 goal_resource_r >= 0.5.
  C5: ARM_2 - ARM_0 goal_resource_r lift >= 0.4.
  C6: ARM_2 wanting!=liking trajectory fraction >= 0.6 (|w-l| > 0.1).

FAIL interpretation grid (manifest acceptance.interpretation_branch):
  Row 1: C2b FAIL, C2c PASS -> row1_curriculum_insufficient (reef neighborhood weak).
  Row 1b: C2b FAIL, C2c FAIL -> row1b_joint_identity_probe_weak (both probes weak).
  Row 2: C2b PASS, C4 or C5 FAIL -> row2_goal_seeding_inert.
  Row 3: C2b PASS, C6 FAIL -> row3_valence_dissoc_absent.
  Row 4: C2b FAIL, C3b FAIL -> row4_substrate_ceiling (ARM_2+ARM_3 classifier dead).
  Row 5: C2b FAIL, C4 PASS, C5 or C6 FAIL -> row5_phase2_lift_or_dissoc_inert.
  Else on FAIL -> row_unmatched (should not occur; signals grid drift).

claim_ids: SD-049, SD-015, MECH-229, MECH-230
experiment_purpose: evidence
supersedes: V3-EXQ-514k
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
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from experiment_protocol import emit_outcome  # noqa: E402
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorld  # noqa: E402
from ree_core.residue.field import VALENCE_LIKING, VALENCE_WANTING  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_514l_sd049_phase3_mech229_wanting_liking_identity"
QUEUE_ID = "V3-EXQ-514l"
CLAIM_IDS = ["SD-049", "SD-015", "MECH-229", "MECH-230"]
EXPERIMENT_PURPOSE = "evidence"
SUPERSEDES = "V3-EXQ-514k"
DRIVE_FLOOR_ACTIVE = 0.9

SEEDS = [42, 43, 44]
P0_EPISODES = 30
P1_EPISODES = 10
EVAL_EPISODES = 15
STEPS_PER_EPISODE = 300
LR = 1e-3

# Unchanged from EXQ-514b (isolates the reef intervention from signal-strength axis).
CLASSIFIER_LOSS_WEIGHT = 0.1

# Reef parameters (from V3-EXQ-521/522 PASS configuration).
REEF_ENABLED = True
HAZARD_FOOD_ATTRACTION = 0.7

# Acceptance thresholds.
C1B_LOSS_DECREASE_FRAC = 0.95
C2B_PROBE_NEIGHBORHOOD_FLOOR = 0.6
C2C_N_ID_SAMPLES_FLOOR = 30
C2D_DRIVE_PEAK_FLOOR = 0.02
C4_GOAL_RESOURCE_R_FLOOR = 0.5
C5_GOAL_RESOURCE_R_LIFT = 0.4
C6_WL_DELTA = 0.1
C6_WL_TRAJECTORY_FRAC = 0.6

# Reef adds 25 reef_field_view dims; updated from EXQ-514b constants.
BASE_OBS_DIM = 250
REEF_EXTRA_DIMS = 25
PER_TYPE_DIMS = 25

ARMS_CONFIG: List[Dict] = [
    dict(
        arm="ARM_0_off",
        sd049_enabled=False,
        n_resource_types=3,
        resource_type_distribution=None,
        use_identity_classifier=False,
        reef=False,
        mech307_active=False,
        use_sd049_per_axis_consumer_cascade=False,
    ),
    dict(
        arm="ARM_1_2type",
        sd049_enabled=True,
        n_resource_types=3,
        resource_type_distribution=(1.0, 1.0, 0.0),
        use_identity_classifier=True,
        reef=True,
        mech307_active=True,
        use_sd049_per_axis_consumer_cascade=True,
    ),
    dict(
        arm="ARM_2_3type",
        sd049_enabled=True,
        n_resource_types=3,
        resource_type_distribution=None,
        use_identity_classifier=True,
        reef=True,
        mech307_active=True,
        use_sd049_per_axis_consumer_cascade=True,
    ),
    dict(
        arm="ARM_3_5type",
        sd049_enabled=True,
        n_resource_types=5,
        resource_type_distribution=None,
        use_identity_classifier=True,
        reef=True,
        mech307_active=True,
        use_sd049_per_axis_consumer_cascade=True,
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
    """Build env per arm config.

    ARM_0 (off/control): 10x10, 0 hazards, reef OFF -- clean off baseline.
    ARM_1/2/3 (active): 10x10, 3 hazards, reef ON + food-attracted hazards.
    Larger grid (10x10 vs EXQ-514b's 8x8) gives room for reef patches.
    Fewer resources (12 vs 15) since reef cells are excluded from spawn pool.
    """
    use_reef = arm_cfg["reef"]
    kwargs = dict(
        size=10,
        num_hazards=0 if not use_reef else 3,
        num_resources=12,
        hazard_harm=0.01,
        resource_benefit=0.18,
        use_proxy_fields=True,
        seed=seed,
        proximity_benefit_scale=0.18,
        resource_respawn_on_consume=True,
    )
    if use_reef:
        kwargs.update(
            reef_enabled=True,
            n_reef_patches=3,
            reef_patch_radius=2,
            hazard_food_attraction=HAZARD_FOOD_ATTRACTION,
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
    drive_floor = DRIVE_FLOOR_ACTIVE if arm_cfg.get("mech307_active") else 0.0
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        alpha_world=0.9,
        world_dim=32,
        drive_weight=2.0,
        z_goal_enabled=True,
        drive_floor=drive_floor,
        drive_ema_alpha=1.0,
    )
    cfg.latent.use_resource_encoder = True
    cfg.latent.z_resource_dim = 32
    cfg.goal.goal_dim = cfg.latent.world_dim
    cfg.residue.valence_enabled = True
    if arm_cfg["use_identity_classifier"]:
        cfg.latent.use_identity_classifier = True
        cfg.latent.identity_classifier_n_types = arm_cfg["n_resource_types"]
    if arm_cfg.get("mech307_active"):
        cfg.use_mech307_conjunction = True
        cfg.use_mech307_consumer_conjunction_read = True
        cfg.e1.schema_wanting_enabled = True
        cfg.mech295_min_drive_to_fire = 0.01
        cfg.mech295_min_z_goal_norm_to_fire = 0.005
        cfg.mech307_conjunction_z_beta_threshold = 0.3
    if arm_cfg.get("use_sd049_per_axis_consumer_cascade"):
        cfg.use_sd049_per_axis_consumer_cascade = True
    return cfg


def _pre_select_hooks(agent: REEAgent, obs_body: torch.Tensor) -> None:
    if getattr(agent.config.e1, "schema_wanting_enabled", False):
        drive_lvl = float(REEAgent.compute_drive_level(obs_body))
        agent.update_schema_wanting(drive_level=drive_lvl)


def neighborhood_dominant_type(obs_dict: Dict, type_names: Tuple[str, ...]) -> int:
    """Label z_resource at every step by the type whose 5x5 field view max is largest.
    Returns type index 0..n_types-1 when at least one type has meaningful presence,
    or -1 (no discriminable signal or SD-049 OFF)."""
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


def run_p0_training(
    agent: REEAgent,
    env: CausalGridWorld,
    device: torch.device,
    n_eps: int,
    classifier_loss_weight: float,
) -> Dict:
    """P0: joint training with classifier_loss_weight on the identity loss term."""
    opt = optim.Adam(agent.parameters(), lr=LR)
    classifier_losses: List[float] = []
    prox_losses: List[float] = []
    for ep in range(n_eps):
        if (ep + 1) % 10 == 0:
            print(f"  [train] P0 ep {ep + 1}/{n_eps}", flush=True)
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
            _pre_select_hooks(agent, obs_body)
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
            _pre_select_hooks(agent, obs_body)
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
    """P2 eval: probe targets, goal_resource_r, valence dissociation, per-axis drive."""
    z_samples_consumption: List[np.ndarray] = []
    targets_consumption: List[int] = []
    z_samples_neighborhood: List[np.ndarray] = []
    targets_neighborhood: List[int] = []
    cosine_sims: List[float] = []
    valence_steps = 0
    wl_dissoc_steps = 0
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
                _pre_select_hooks(agent, obs_body)
                candidates = agent.generate_trajectories(latent, e1_prior, ticks)
                action = agent.select_action(candidates, ticks)
            action_idx = int(action.argmax(dim=-1).item())
            _, harm_signal, done, info, obs_dict = env.step(action_idx)
            ttype = info.get("transition_type", "none")

            if agent.goal_state is not None and agent.goal_state.is_active():
                with torch.no_grad():
                    z_g = agent.goal_state.z_goal
                    if z_g.ndim == 1:
                        z_g = z_g.unsqueeze(0)
                    z_ref = (
                        latent.z_resource
                        if latent.z_resource is not None
                        else latent.z_world
                    )
                    if z_ref.ndim == 1:
                        z_ref = z_ref.unsqueeze(0)
                    if z_g.shape[-1] == z_ref.shape[-1]:
                        cos = float(
                            F.cosine_similarity(z_g, z_ref, dim=-1).mean().item()
                        )
                        cosine_sims.append(cos)
                z_w = latent.z_world
                with torch.no_grad():
                    v = agent.residue_field.evaluate_valence(z_w)
                w_amp = float(v[0, VALENCE_WANTING].item())
                l_amp = float(v[0, VALENCE_LIKING].item())
                valence_steps += 1
                if abs(w_amp - l_amp) > C6_WL_DELTA:
                    wl_dissoc_steps += 1

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
    goal_resource_r = float(np.mean(cosine_sims)) if cosine_sims else 0.0
    wl_trajectory_frac = (
        float(wl_dissoc_steps) / float(valence_steps) if valence_steps > 0 else 0.0
    )
    return {
        "probe_acc_consumption": probe_acc_consumption,
        "n_identity_samples_consumption": len(targets_consumption),
        "probe_acc_neighborhood": probe_acc_neighborhood,
        "n_identity_samples_neighborhood": len(targets_neighborhood),
        "goal_resource_r": goal_resource_r,
        "n_cosine_samples": len(cosine_sims),
        "wanting_liking_dissoc_fraction": wl_trajectory_frac,
        "n_valence_steps": valence_steps,
        "peak_per_axis_drive": [float(x) for x in peak_drive_per_axis.tolist()],
    }


def identity_recovery_probe(
    z_samples: List[np.ndarray], targets: List[int]
) -> float:
    """Train a linear classifier on (z_resource, type) and return held-out accuracy."""
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
        "mech307_active": bool(arm_cfg.get("mech307_active", False)),
        "drive_floor": DRIVE_FLOOR_ACTIVE if arm_cfg.get("mech307_active") else 0.0,
        "reef_enabled": arm_cfg["reef"],
        "use_sd049_per_axis_consumer_cascade": bool(arm_cfg.get("use_sd049_per_axis_consumer_cascade", False)),
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
                "reef_enabled": r["reef_enabled"],
                "use_sd049_per_axis_consumer_cascade": r["use_sd049_per_axis_consumer_cascade"],
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
                "goal_resource_r_mean": 0.0,
                "wanting_liking_dissoc_fraction_mean": 0.0,
            }
        b = bucket[arm]
        b["n_seeds"] += 1
        b["probe_acc_consumption_mean"] += r["probe_acc_consumption"]
        b["probe_acc_neighborhood_mean"] += r["probe_acc_neighborhood"]
        b["goal_resource_r_mean"] += r["goal_resource_r"]
        b["wanting_liking_dissoc_fraction_mean"] += r["wanting_liking_dissoc_fraction"]
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
        b["goal_resource_r_mean"] /= n
        b["wanting_liking_dissoc_fraction_mean"] /= n
        b["p0_classifier_loss_first_q_mean"] /= n
        b["p0_classifier_loss_last_q_mean"] /= n
    return bucket


def evaluate_acceptance(agg: Dict[str, Dict]) -> Dict:
    a0 = agg["ARM_0_off"]
    a1 = agg["ARM_1_2type"]
    a2 = agg["ARM_2_3type"]
    a3 = agg["ARM_3_5type"]

    # Reef adds 25 dims: ARM_1/2 = 250 + 3*25 + 25 = 350; ARM_3 = 250 + 5*25 + 25 = 400.
    c0 = (a0["n_seeds"] > 0)
    c1a = a1["world_obs_dim"] == 350
    expected_random_loss = float(np.log(a1["n_resource_types"]))
    c1b = (
        a1["p0_classifier_loss_last_q_mean"] < a1["p0_classifier_loss_first_q_mean"] * C1B_LOSS_DECREASE_FRAC
        or a1["p0_classifier_loss_last_q_mean"] < 0.5 * expected_random_loss
    )
    c2a = a2["world_obs_dim"] == 350
    c2b = a2["probe_acc_neighborhood_mean"] >= C2B_PROBE_NEIGHBORHOOD_FLOOR
    c2c = a2["n_identity_samples_consumption_total"] >= C2C_N_ID_SAMPLES_FLOOR
    c2d = float(np.max(a2["peak_per_axis_drive_max"])) > C2D_DRIVE_PEAK_FLOOR
    c3a = a3["world_obs_dim"] == 400
    c3b = a3["p0_classifier_loss_last_q_mean"] > 0.0
    c4 = a2["goal_resource_r_mean"] >= C4_GOAL_RESOURCE_R_FLOOR
    c5 = (a2["goal_resource_r_mean"] - a0["goal_resource_r_mean"]) >= C5_GOAL_RESOURCE_R_LIFT
    c6 = a2["wanting_liking_dissoc_fraction_mean"] >= C6_WL_TRAJECTORY_FRAC

    overall = (
        c0 and c1a and c1b and c2a and c2b and c2c and c2d and c3a and c3b and c4 and c5 and c6
    )
    branch = "pass"
    if not overall:
        # Most-specific first; 514j FAIL (C2b+C2c both false) was falling through to "pass".
        if not c2b and not c2c:
            branch = "row1b_joint_identity_probe_weak"
        elif not c2b and c2c:
            branch = "row1_curriculum_insufficient"
        elif c2b and (not c4 or not c5):
            branch = "row2_goal_seeding_inert"
        elif c2b and not c6:
            branch = "row3_valence_dissoc_absent"
        elif not c2b and not c3b:
            branch = "row4_substrate_ceiling"
        elif not c2b and c4 and (not c5 or not c6):
            branch = "row5_phase2_lift_or_dissoc_inert"
        else:
            branch = "row_unmatched"
    return {
        "C0_arm0_runs_clean": bool(c0),
        "C1a_arm1_world_obs_dim_350": bool(c1a),
        "C1b_arm1_classifier_converges": bool(c1b),
        "C2a_arm2_world_obs_dim_350": bool(c2a),
        "C2b_arm2_probe_acc_neighborhood": bool(c2b),
        "C2c_arm2_n_identity_samples_consumption": bool(c2c),
        "C2d_arm2_per_axis_drive_evolves": bool(c2d),
        "C3a_arm3_world_obs_dim_400": bool(c3a),
        "C3b_arm3_classifier_did_fire": bool(c3b),
        "C4_arm2_goal_resource_r": bool(c4),
        "C5_arm2_goal_resource_r_lift": bool(c5),
        "C6_arm2_wanting_liking_dissoc": bool(c6),
        "interpretation_branch": branch,
        "all_pass": bool(overall),
    }


def main(dry_run: bool = False):
    print(f"[{EXPERIMENT_TYPE}] starting (dry_run={dry_run})")
    print(f"[{EXPERIMENT_TYPE}] supersedes {SUPERSEDES}")
    print(f"[{EXPERIMENT_TYPE}] reef_enabled={REEF_ENABLED} hazard_food_attraction={HAZARD_FOOD_ATTRACTION}")
    print(f"[{EXPERIMENT_TYPE}] classifier_loss_weight={CLASSIFIER_LOSS_WEIGHT}")
    print(f"[{EXPERIMENT_TYPE}] Phase 3: use_sd049_per_axis_consumer_cascade=True on ARM_1/2/3")
    seeds = (SEEDS[0],) if dry_run else SEEDS
    per_cell: List[Dict] = []
    t0 = time.time()
    eps_per_run = (5 + 2 + 3) if dry_run else (P0_EPISODES + P1_EPISODES + EVAL_EPISODES)
    for seed in seeds:
        for arm_cfg in ARMS_CONFIG:
            print(f"Seed {seed} Condition {arm_cfg['arm']}", flush=True)
            arm_t0 = time.time()
            r = run_seed_arm(seed, arm_cfg, dry_run)
            per_cell.append(r)
            print(
                f"  seed={seed} arm={r['arm']:<14} obs_dim={r['world_obs_dim']:3d} "
                f"mech307={r['mech307_active']} drive_floor={r['drive_floor']:.1f} "
                f"reef={r['reef_enabled']} cascade={r['use_sd049_per_axis_consumer_cascade']} "
                f"probe_neigh={r['probe_acc_neighborhood']:.3f} "
                f"goal_r={r['goal_resource_r']:.3f} wl_frac={r['wanting_liking_dissoc_fraction']:.3f} "
                f"probe_cons={r['probe_acc_consumption']:.3f} "
                f"elapsed={time.time()-arm_t0:.1f}s",
                flush=True,
            )
            if (len(per_cell) % 1) == 0 and len(per_cell) > 0:
                print(
                    f"  [train] completed unit {len(per_cell)}/{len(seeds) * len(ARMS_CONFIG)} "
                    f"ep_budget={eps_per_run}",
                    flush=True,
                )
    agg = aggregate(per_cell)
    acceptance = evaluate_acceptance(agg)
    elapsed = time.time() - t0
    outcome = "PASS" if acceptance["all_pass"] else "FAIL"

    print(f"[{EXPERIMENT_TYPE}] aggregates:")
    for arm in ("ARM_0_off", "ARM_1_2type", "ARM_2_3type", "ARM_3_5type"):
        a = agg[arm]
        print(
            f"  {arm:<14} obs_dim={a['world_obs_dim']:3d} reef={a['reef_enabled']} "
            f"cascade={a['use_sd049_per_axis_consumer_cascade']} "
            f"probe_neigh={a['probe_acc_neighborhood_mean']:.3f} "
            f"probe_cons={a['probe_acc_consumption_mean']:.3f} "
            f"clf_first/last={a['p0_classifier_loss_first_q_mean']:.3f}/"
            f"{a['p0_classifier_loss_last_q_mean']:.3f} "
            f"goal_r={a['goal_resource_r_mean']:.3f} "
            f"wl_frac={a['wanting_liking_dissoc_fraction_mean']:.3f} "
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
        "queue_id": QUEUE_ID,
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
        "sd049_phase3_substrate": {
            "use_sd049_per_axis_consumer_cascade": True,
            "use_mech307_conjunction": True,
            "drive_floor_active": DRIVE_FLOOR_ACTIVE,
            "sp_cem_main_path_default": True,
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
        "arms": list(agg.values()),
        "per_seed_per_arm": per_cell,
        "acceptance": acceptance,
        "thresholds": {
            "C1b_loss_decrease_frac": C1B_LOSS_DECREASE_FRAC,
            "C2b_probe_neighborhood_floor": C2B_PROBE_NEIGHBORHOOD_FLOOR,
            "C2c_n_id_samples_floor": C2C_N_ID_SAMPLES_FLOOR,
            "C2d_drive_peak_floor": C2D_DRIVE_PEAK_FLOOR,
            "C4_goal_resource_r_floor": C4_GOAL_RESOURCE_R_FLOOR,
            "C5_goal_resource_r_lift": C5_GOAL_RESOURCE_R_LIFT,
            "C6_wl_trajectory_frac": C6_WL_TRAJECTORY_FRAC,
        },
        "supersedes_note": (
            "MECH-229 retest on SD-049 Phase 3 (MECH-295 axis-matched consumer cascade). "
            "Supersedes V3-EXQ-514j (Phase 2, pre-consumer-cascade). "
            "Operating stack: MECH-307 conjunction + drive_floor=0.9 + main-path SP-CEM + per-axis consumer cascade. "
            f"interpretation_branch={acceptance.get('interpretation_branch', 'n/a')}"
        ),
    }
    out_path = write_flat_manifest(
        manifest,
        out_dir,
        dry_run=False,
        config=manifest.get("config"),
        seeds=SEEDS,
        script_path=Path(__file__),
    )
    print(f"Result written to: {out_path}")
    return outcome, out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Smoke run, no manifest.")
    args = parser.parse_args()
    result = main(dry_run=args.dry_run)
    if result == 0:
        sys.exit(0)
    outcome, out_path = result
    emit_outcome(outcome=outcome, manifest_path=out_path)
    sys.exit(0)
