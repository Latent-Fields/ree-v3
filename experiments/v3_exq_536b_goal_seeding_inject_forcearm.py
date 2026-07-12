#!/opt/local/bin/python3
"""V3-EXQ-536b: z_goal_inject Force-Arm Probe

Diagnostic follow-on to EXQ-536. EXQ-536 produced approach_commit_rate=0.0 in
BOTH arms (pipeline-OFF and pipeline-ON) -- which means C3 goal_lift is
undefined, not informative. The persistent z_goal attractor never activated
during eval (C2 z_goal_active_fraction=0.0).

This probe uses the MECH-188 z_goal_inject hook (cfg.goal.z_goal_inject,
goal.py:78 + agent.py:2316-2321) to force-floor the action-time z_goal norm
to 0.3 regardless of whether the persistent attractor was successfully
seeded. with_injection() applies during select_action() ONLY -- it does NOT
modify the persistent state, so this isolates the question:

  When the SCORING-TIME z_goal is non-trivially active, does the downstream
  commitment chain (E3 score_trajectory + MECH-295 cue bias + BG beta
  gate elevation) recruit approach commits near resources?

experiment_purpose: diagnostic
claim_ids:           ["ARC-030","MECH-112","SD-018","SD-012","MECH-295"]
evidence_direction:  non_contributory  (probe; not weighted as evidence)

=== DESIGN ===

Two arms x 3 seeds. Both arms inherit the EXQ-536 ARM_1 pipeline
(SD-018 + z_goal + benefit_eval + MECH-295, cfg.e3.goal_weight=1.0 fix).
Only manipulated variable is cfg.goal.z_goal_inject:

  ARM_0 (no_inject):     z_goal_inject=0.0  (legacy, replicates EXQ-536 ARM_1).
  ARM_1 (forced_inject): z_goal_inject=0.3  (action-time z_goal norm floor).

Same training (random policy, 100 eps) and eval (40 eps full E3 selection)
as EXQ-536 so results are comparable.

=== METRICS ===

Primary:
  approach_commit_rate    fraction of eval steps committed AND near resource
                          (resource_proximity >= 0.25)
  goal_lift_inject        ARM_1.approach_commit_rate / ARM_0.approach_commit_rate

Auxiliary:
  z_goal_active_fraction       persistent-state is_active fraction (should
                               still be ~0 in both arms since inject does not
                               modify persistent state -- this is the
                               diagnostic that confirms the inject lever is
                               doing what we expect: scoring works WITHOUT
                               persistent seeding).
  benefit_eval_r2              SD-018 substrate quality
  mean_dacc_score_bias         downstream commit-chain signal magnitude

=== INTERPRETATION GRID ===

  ARM_1.approach_commit > 0 AND ARM_0.approach_commit ~ 0:
      Downstream commit chain works when z_goal IS active at scoring time.
      EXQ-536 root cause is upstream seeding (z_goal never reaches the
      action-selection path). Recommend pursuing z_goal seeding fix
      (drive collapse / threshold strict-greater / encoder noise).

  Both arms ~ 0:
      Downstream commit chain inert even when z_goal is forced active.
      EXQ-536 root cause is downstream of z_goal scoring -- E3 weighting,
      BG beta gate, or MECH-295 bridge gain insufficient. Aligns with
      Q-040 wired-but-inert pattern; recommend stronger E3 goal_weight
      and/or goal_proximity gain.

  Both arms > 0 with similar lift:
      EXQ-536 ARM_1 was actually working at action-selection time but
      the metric collapse (eval ran into something else, e.g. early
      death from harm). Need to revisit the eval episode dynamics.

=== ACCEPTANCE ===

Diagnostic. PASS criterion is "the probe produced an interpretable signal":
either ARM_1 > ARM_0 by >= 1.5x (commit chain works), or both arms remain
at 0 (commit chain inert). The manifest tags whichever interpretation row
the result hits; full review is manual.

Defined as outcome={DIAGNOSTIC_COMPLETE | DRY_RUN_COMPLETE}; no PASS/FAIL
auto-tag because no single C-criterion is well-posed for a probe of this
shape.

architecture_epoch: "ree_hybrid_guardrails_v1"
"""

import json
import sys
import os
import time
import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from datetime import datetime
from pathlib import Path
from typing import List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.agent import REEAgent
from ree_core.utils.config import REEConfig
from experiments.pack_writer import write_flat_manifest  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_536b_goal_seeding_inject_forcearm"
QUEUE_ID = "V3-EXQ-536b"
CLAIM_IDS = ["ARC-030", "MECH-112", "SD-018", "SD-012", "MECH-295"]

N_TRAIN_EPS    = 100
N_EVAL_EPS     = 40
N_STEPS        = 200
N_SEEDS        = 3
GRID_SIZE      = 12
APPROACH_THRESH = 0.25
INJECT_NORM    = 0.3   # MECH-188 forced action-time z_goal norm

DRY_RUN = "--dry-run" in sys.argv
if DRY_RUN:
    N_TRAIN_EPS = 8
    N_EVAL_EPS  = 5
    N_SEEDS     = 1


# ------------------------------------------------------------------ #

def _obs_tensors(obs_dict):
    body  = obs_dict["body_state"].float().unsqueeze(0)
    world = obs_dict["world_state"].float().unsqueeze(0)
    return body, world


def _resource_proximity(env) -> float:
    if not env.resources:
        return 0.0
    ax, ay = env.agent_x, env.agent_y
    min_dist = min(abs(ax - int(r[0])) + abs(ay - int(r[1])) for r in env.resources)
    return 1.0 / (1.0 + min_dist)


def _benefit_drive(obs_dict):
    body_raw = obs_dict["body_state"]
    benefit  = float(body_raw[11].item()) if body_raw.numel() > 11 else 0.0
    energy   = float(body_raw[3].item())  if body_raw.numel() > 3  else 0.5
    drive    = max(0.0, min(1.0, 1.0 - energy))
    return benefit, drive


def _r_squared(pred, actual):
    if len(pred) < 4:
        return 0.0
    mean_a = sum(actual) / len(actual)
    ss_tot = sum((a - mean_a) ** 2 for a in actual)
    ss_res = sum((p - a) ** 2 for p, a in zip(pred, actual))
    if ss_tot < 1e-9:
        return 0.0
    return 1.0 - ss_res / ss_tot


def _make_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(
        seed=seed,
        size=GRID_SIZE,
        num_hazards=4,
        num_resources=4,
        hazard_harm=0.02,
        energy_decay=0.005,
        use_proxy_fields=True,
        resource_respawn_on_consume=True,
        reef_enabled=False,
    )


def _make_agent(env, z_goal_inject: float) -> REEAgent:
    cfg = REEConfig.from_dims(
        world_obs_dim=env.world_obs_dim,
        body_obs_dim=env.body_obs_dim,
        action_dim=env.action_dim,
        use_resource_proximity_head=True,
        drive_weight=2.0,
        z_goal_enabled=True,
        benefit_eval_enabled=True,
        benefit_weight=2.0,
        use_mech295_liking_bridge=True,
        mech295_drive_to_liking_gain=1.0,
        mech295_liking_to_approach_cue_gain=0.5,
    )
    cfg.e3.goal_weight = 1.0           # EXQ-536 bug fix
    cfg.goal.z_goal_inject = z_goal_inject  # MECH-188 force-arm
    return REEAgent(cfg)


# ------------------------------------------------------------------ #

def run_arm(z_goal_inject: float, seed: int) -> dict:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    env = _make_env(seed)
    agent = _make_agent(env, z_goal_inject)
    optimizer = optim.Adam(agent.parameters(), lr=3e-4)
    benefit_eval_optimizer = optim.Adam(
        list(agent.e3.benefit_eval_head.parameters()), lr=1e-4
    )

    benefit_pred_vals: List[float] = []
    benefit_target_vals: List[float] = []
    world_dim = agent.config.latent.world_dim

    # ----------------- training -----------------
    agent.train()
    for ep in range(N_TRAIN_EPS):
        _, obs_dict = env.reset()
        agent.reset()
        for _step in range(N_STEPS):
            body, world = _obs_tensors(obs_dict)
            latent = agent.sense(obs_body=body, obs_world=world)
            agent.clock.advance()

            action_int = random.randint(0, env.action_dim - 1)
            action_oh = torch.zeros(1, env.action_dim)
            action_oh[0, action_int] = 1.0
            agent._last_action = action_oh

            _, _harm, done, _info, obs_dict = env.step(action_oh)
            benefit, drive = _benefit_drive(obs_dict)

            pred_loss = agent.compute_prediction_loss()
            e2_loss   = agent.compute_e2_loss()
            total_loss = pred_loss + e2_loss

            resource_field = obs_dict.get("resource_field_view", None)
            if resource_field is not None:
                prox_target = float(resource_field.max().item())
            else:
                prox_target = 0.0
            prox_loss = agent.compute_resource_proximity_loss(prox_target, latent)
            total_loss = total_loss + prox_loss

            with torch.no_grad():
                z_world_det = latent.z_world.detach()
            benefit_pred_train = agent.e3.benefit_eval_head(z_world_det)
            prox_t = torch.tensor([[prox_target]], dtype=torch.float32)
            b_loss = F.mse_loss(benefit_pred_train, prox_t)
            benefit_pred_vals.append(float(benefit_pred_train.item()))
            benefit_target_vals.append(prox_target)
            if len(benefit_pred_vals) > 2000:
                benefit_pred_vals = benefit_pred_vals[-2000:]
                benefit_target_vals = benefit_target_vals[-2000:]

            agent.e3.record_benefit_sample(1)
            if benefit > 0.01:
                agent.update_z_goal(benefit, drive)

            if b_loss.requires_grad:
                benefit_eval_optimizer.zero_grad()
                b_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(agent.e3.benefit_eval_head.parameters()), 0.5
                )
                benefit_eval_optimizer.step()

            if total_loss.requires_grad:
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                optimizer.step()

            if done:
                break

    benefit_eval_r2 = (
        _r_squared(benefit_pred_vals[-1000:], benefit_target_vals[-1000:])
        if benefit_pred_vals else None
    )

    # ----------------- eval -----------------
    agent.eval()
    approach_commit_steps = 0
    total_eval_steps = 0
    z_goal_active_steps = 0
    resource_contacts = 0
    inject_observed_steps = 0   # how often select_action saw an injected z_goal

    for _ep in range(N_EVAL_EPS):
        _, obs_dict = env.reset()
        agent.reset()
        if agent.goal_state is not None:
            agent.goal_state.reset()

        for _step in range(N_STEPS):
            body, world = _obs_tensors(obs_dict)
            agent.sense(obs_body=body, obs_world=world)
            ticks = agent.clock.advance()

            e1_prior = (
                agent._e1_tick(agent._current_latent)
                if ticks["e1_tick"]
                else torch.zeros(1, world_dim, device=agent.device)
            )
            candidates = agent.generate_trajectories(
                agent._current_latent, e1_prior, ticks
            )
            _action_tensor = agent.select_action(candidates, ticks)

            is_committed = agent.e3._committed_trajectory is not None
            rp = _resource_proximity(env)
            is_near_rsc = rp >= APPROACH_THRESH

            total_eval_steps += 1
            if is_committed and is_near_rsc:
                approach_commit_steps += 1
            if agent.goal_state is not None and agent.goal_state.is_active():
                z_goal_active_steps += 1
            # Inject lever activity probe: persistent z_goal norm < inject -> the
            # injected view is doing the lift at scoring time.
            if z_goal_inject > 0.0 and agent.goal_state is not None:
                if agent.goal_state.z_goal.norm().item() < z_goal_inject:
                    inject_observed_steps += 1

            action_int = int(_action_tensor.argmax(dim=-1).item())
            action_oh = torch.zeros(1, env.action_dim)
            action_oh[0, action_int] = 1.0

            _, _harm, done, info, obs_dict = env.step(action_oh)
            ttype = info.get("transition_type", "none")
            if ttype in ("resource", "benefit_approach"):
                resource_contacts += 1

            benefit, drive = _benefit_drive(obs_dict)
            if benefit > 0.01:
                agent.update_z_goal(benefit, drive)

            if done:
                break

    return {
        "approach_commit_rate":      approach_commit_steps / max(1, total_eval_steps),
        "z_goal_active_fraction":    z_goal_active_steps / max(1, total_eval_steps),
        "inject_observed_fraction":  inject_observed_steps / max(1, total_eval_steps),
        "benefit_eval_r2":           benefit_eval_r2,
        "resource_contacts_per_ep":  resource_contacts / max(1, N_EVAL_EPS),
        "total_eval_steps":          total_eval_steps,
        "approach_commit_steps":     approach_commit_steps,
    }


# ------------------------------------------------------------------ #

def main():
    start_time = time.time()
    print("V3-EXQ-536b z_goal_inject force-arm probe", flush=True)
    print(
        f"DRY_RUN={DRY_RUN} N_TRAIN={N_TRAIN_EPS} N_EVAL={N_EVAL_EPS} "
        f"N_SEEDS={N_SEEDS} INJECT_NORM={INJECT_NORM}",
        flush=True,
    )

    arm0_rates, arm1_rates, lifts, seed_details = [], [], [], []

    for seed in range(N_SEEDS):
        print(f"\n-- Seed {seed} --", flush=True)
        r0 = run_arm(z_goal_inject=0.0,         seed=seed)
        r1 = run_arm(z_goal_inject=INJECT_NORM, seed=seed)
        acr0 = r0["approach_commit_rate"]
        acr1 = r1["approach_commit_rate"]
        lift = acr1 / max(acr0, 1e-6)
        arm0_rates.append(acr0)
        arm1_rates.append(acr1)
        lifts.append(lift)
        seed_details.append({
            "seed": seed, "arm0_no_inject": r0, "arm1_inject": r1,
            "inject_lift": float(lift),
        })
        print(
            f"  ARM_0(no_inject)  approach_commit={acr0:.4f}"
            f"  ARM_1(inject={INJECT_NORM})  approach_commit={acr1:.4f}"
            f"  lift={lift:.3f}",
            flush=True,
        )

    mean_arm0 = float(np.mean(arm0_rates))
    mean_arm1 = float(np.mean(arm1_rates))
    mean_lift = float(np.mean(lifts))

    # Interpretation tag
    if mean_arm1 > 0 and mean_arm0 < 1e-4:
        interp = "downstream_chain_works_seeding_is_blocker"
    elif mean_arm0 < 1e-4 and mean_arm1 < 1e-4:
        interp = "downstream_chain_inert_even_with_active_z_goal"
    elif mean_arm0 > 0 and mean_arm1 > 0:
        interp = "both_arms_nonzero_revisit_536_metric_dynamics"
    else:
        interp = "unexpected_pattern_review_manually"

    elapsed = time.time() - start_time
    outcome = "DIAGNOSTIC_COMPLETE" if not DRY_RUN else "DRY_RUN_COMPLETE"

    print(f"\nElapsed: {elapsed:.1f}s", flush=True)
    print(f"ARM_0 mean approach_commit: {mean_arm0:.4f}", flush=True)
    print(f"ARM_1 mean approach_commit: {mean_arm1:.4f}", flush=True)
    print(f"mean inject_lift: {mean_lift:.3f}", flush=True)
    print(f"interpretation: {interp}", flush=True)

    manifest = {
        "run_id": f"{EXPERIMENT_TYPE}_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}_v3",
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": QUEUE_ID,
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": "diagnostic",
        "evidence_direction": "non_contributory",
        "evidence_direction_note": (
            "Diagnostic z_goal_inject force-arm probe. Compares EXQ-536 ARM_1 "
            "with cfg.goal.z_goal_inject=0.0 vs cfg.goal.z_goal_inject=0.3 to "
            "isolate whether the EXQ-536 approach_commit_rate=0.0 collapse is "
            "upstream (z_goal never seeded) or downstream (commit chain inert "
            "even when z_goal is action-time active). Not weighted as evidence."
        ),
        "evidence_direction_per_claim": {
            cid: "non_contributory" for cid in CLAIM_IDS
        },
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "outcome": outcome,
        "interpretation":          interp,
        "mean_arm0_approach_commit_rate": mean_arm0,
        "mean_arm1_approach_commit_rate": mean_arm1,
        "mean_inject_lift":               mean_lift,
        "per_seed_lifts":                 lifts,
        "seed_details":                   seed_details,
        "config": {
            "n_train_eps":      N_TRAIN_EPS,
            "n_eval_eps":       N_EVAL_EPS,
            "n_steps":          N_STEPS,
            "n_seeds":          N_SEEDS,
            "grid_size":        GRID_SIZE,
            "approach_thresh":  APPROACH_THRESH,
            "inject_norm":      INJECT_NORM,
            "dry_run":          DRY_RUN,
        },
        "elapsed_seconds":  elapsed,
        "generated_utc":    datetime.utcnow().isoformat() + "Z",
    }

    if DRY_RUN:
        print("[DRY RUN] Not writing evidence.", flush=True)
        return

    evidence_dir = (
        Path(__file__).resolve().parents[2]
        / "REE_assembly" / "evidence" / "experiments"
        / EXPERIMENT_TYPE
    )
    manifest_path = write_flat_manifest(
        manifest,
        evidence_dir,
        dry_run=False,
        config=manifest.get("config"),
        seeds=None,
        script_path=Path(__file__),
    )
    print(f"Manifest written: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
