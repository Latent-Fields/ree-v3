#!/opt/local/bin/python3
"""V3-EXQ-536a: Goal Seeding Instrumentation Diagnostic

Diagnostic follow-on to EXQ-536 (FAIL: z_goal_active_fraction=0.0 across all 3 seeds
despite ARM_1 having SD-018 + z_goal + benefit_eval + MECH-295 enabled and the
cfg.e3.goal_weight=1.0 bug fixed). Per-step instrumentation isolates which of
three candidate root causes is dominant.

experiment_purpose: diagnostic
claim_ids:           ["ARC-030","MECH-112","SD-018","SD-012","MECH-295"]
evidence_direction:  non_contributory  (probe; not weighted as evidence)

=== HYPOTHESES UNDER TEST ===

H_a (drive collapse on contact):
  Resource contact triggers env.agent_energy += contact_benefit*0.5
  (causal_grid_world.py:987), refilling energy precisely when benefit_exposure
  spikes. drive_level = 1 - energy collapses, so the SD-012 multiplier
  (1 + drive_weight*drive) does not amplify and effective_benefit stays at
  benefit_threshold (0.1) -- gated `>` not `>=`.

H_b (benefit_exposure never crosses):
  EMA at alpha=0.1 + sparse contacts means benefit_exposure peaks at ~0.1
  but the inner gate `effective_benefit > 0.1` is strict-greater.

H_c (update fires but z_goal stays small):
  GoalState.update is called and the inner gate trips, but z_goal.norm()
  doesn't rise enough for is_active() to register. Bug in alpha_goal=0.05
  pull, in SD-015 z_resource path, or some upstream encoder feeding zero.

=== DESIGN ===

Single seed, full ARM_1 pipeline from EXQ-536, training-light. During eval,
log every step:
  benefit_exposure          (body_state[11])
  drive_level               (1 - body_state[3])
  effective_benefit         (benefit * z_goal_seeding_gain * (1+drive_weight*drive))
  outer_gate_fired          (benefit > 0.01, the eval-loop guard)
  inner_gate_fired          (effective_benefit > goal.benefit_threshold)
  z_goal_norm_pre           (norm BEFORE update_z_goal)
  z_goal_norm_post          (norm AFTER update_z_goal)
  is_active_pre             (is_active BEFORE update_z_goal)
  is_active_post            (is_active AFTER update_z_goal)
  contact_event             (env info transition_type in {resource,benefit_approach})
  energy                    (body_state[3], for context)

Aggregates dispatch the hypotheses:

  H_a:  mean drive_level on contact ticks. If << 0.5, drive collapse is real.
  H_b:  max effective_benefit across eval. If <= 0.1, threshold never crossed.
  H_c:  count(inner_gate_fired AND z_goal_norm_post < 1e-6).

=== ACCEPTANCE ===

Diagnostic. No PASS criterion -- the manifest reports the dispatch counts and
distributions; interpretation lives in the review. The probe is informative
iff at least one hypothesis is decisively confirmed or eliminated.

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

EXPERIMENT_TYPE = "v3_exq_536a_goal_seeding_instrumentation"
QUEUE_ID = "V3-EXQ-536a"
CLAIM_IDS = ["ARC-030", "MECH-112", "SD-018", "SD-012", "MECH-295"]

N_TRAIN_EPS  = 30
N_EVAL_EPS   = 20
N_STEPS      = 200
SEED         = 0
GRID_SIZE    = 12

DRY_RUN = "--dry-run" in sys.argv
if DRY_RUN:
    N_TRAIN_EPS = 5
    N_EVAL_EPS  = 3


# ------------------------------------------------------------------ #

def _obs_tensors(obs_dict):
    body  = obs_dict["body_state"].float().unsqueeze(0)
    world = obs_dict["world_state"].float().unsqueeze(0)
    return body, world


def _benefit_drive_energy(obs_dict):
    body_raw = obs_dict["body_state"]
    benefit  = float(body_raw[11].item()) if body_raw.numel() > 11 else 0.0
    energy   = float(body_raw[3].item())  if body_raw.numel() > 3  else 0.5
    drive    = max(0.0, min(1.0, 1.0 - energy))
    return benefit, drive, energy


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


def _make_agent(env) -> REEAgent:
    cfg = REEConfig.from_dims(
        world_obs_dim=env.world_obs_dim,
        body_obs_dim=env.body_obs_dim,
        action_dim=env.action_dim,
        use_resource_proximity_head=True,  # SD-018
        drive_weight=2.0,
        z_goal_enabled=True,
        benefit_eval_enabled=True,
        benefit_weight=2.0,
        use_mech295_liking_bridge=True,
        mech295_drive_to_liking_gain=1.0,
        mech295_liking_to_approach_cue_gain=0.5,
    )
    cfg.e3.goal_weight = 1.0  # EXQ-536 bug fix
    return REEAgent(cfg)


# ------------------------------------------------------------------ #

def main():
    start_time = time.time()
    print("V3-EXQ-536a goal seeding instrumentation", flush=True)
    print(f"DRY_RUN={DRY_RUN} N_TRAIN={N_TRAIN_EPS} N_EVAL={N_EVAL_EPS} SEED={SEED}", flush=True)

    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)

    env = _make_env(SEED)
    agent = _make_agent(env)
    optimizer = optim.Adam(agent.parameters(), lr=3e-4)
    benefit_eval_optimizer = optim.Adam(
        list(agent.e3.benefit_eval_head.parameters()), lr=1e-4
    )

    benefit_threshold = float(agent.config.goal.benefit_threshold)
    drive_weight      = float(agent.config.goal.drive_weight)
    seeding_gain      = float(agent.config.goal.z_goal_seeding_gain)

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
            benefit, drive, _ = _benefit_drive_energy(obs_dict)

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

    # ----------------- eval (instrumented) -----------------
    agent.eval()
    world_dim = agent.config.latent.world_dim

    # Per-step records
    records = []          # full per-step records for first eval episode
    ep_summaries = []     # one summary per eval episode

    # Aggregates
    total_steps           = 0
    n_outer_fired         = 0
    n_inner_fired         = 0
    n_contact_events      = 0
    n_inner_no_norm_gain  = 0   # H_c counter

    benefits_on_outer     = []
    drives_on_outer       = []
    drives_on_contact     = []
    energies_on_contact   = []
    eff_benefits          = []
    z_goal_norms          = []  # at metric-sample time (start of step)
    is_active_at_sample   = 0

    max_eff_benefit       = 0.0
    z_goal_max_norm       = 0.0

    # Track first-episode trace cap so manifest doesn't explode
    FIRST_EP_TRACE_CAP = 200

    for ep in range(N_EVAL_EPS):
        _, obs_dict = env.reset()
        agent.reset()
        if agent.goal_state is not None:
            agent.goal_state.reset()

        ep_outer = 0
        ep_inner = 0
        ep_contacts = 0
        ep_active_steps = 0
        ep_steps = 0

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

            # Sample-time metrics (matches EXQ-536 ordering)
            z_norm_pre_sample = float(agent.goal_state.z_goal.norm().item())
            is_active_pre_sample = bool(agent.goal_state.is_active())
            z_goal_norms.append(z_norm_pre_sample)
            if z_norm_pre_sample > z_goal_max_norm:
                z_goal_max_norm = z_norm_pre_sample
            if is_active_pre_sample:
                is_active_at_sample += 1
                ep_active_steps += 1
            total_steps += 1
            ep_steps += 1

            action_int = int(_action_tensor.argmax(dim=-1).item())
            action_oh = torch.zeros(1, env.action_dim)
            action_oh[0, action_int] = 1.0

            _, _harm, done, info, obs_dict = env.step(action_oh)

            benefit, drive, energy = _benefit_drive_energy(obs_dict)
            ttype = info.get("transition_type", "none")
            is_contact = ttype in ("resource", "benefit_approach")
            if is_contact:
                n_contact_events += 1
                ep_contacts += 1
                drives_on_contact.append(drive)
                energies_on_contact.append(energy)

            # Replicate the inner-gate math without firing the update yet
            effective_benefit = benefit * seeding_gain * (1.0 + drive_weight * drive)
            if effective_benefit > max_eff_benefit:
                max_eff_benefit = effective_benefit
            eff_benefits.append(effective_benefit)

            outer_fired = benefit > 0.01
            inner_fired = effective_benefit > benefit_threshold
            if outer_fired:
                n_outer_fired += 1
                ep_outer += 1
                benefits_on_outer.append(benefit)
                drives_on_outer.append(drive)
            if inner_fired:
                n_inner_fired += 1
                ep_inner += 1

            # Pre-update z_goal norm for H_c counter
            z_norm_pre_update = float(agent.goal_state.z_goal.norm().item())

            if outer_fired:
                agent.update_z_goal(benefit, drive)

            z_norm_post_update = float(agent.goal_state.z_goal.norm().item())
            is_active_post_update = bool(agent.goal_state.is_active())

            # H_c: inner gate said fire AND z_goal didn't grow
            if inner_fired and (z_norm_post_update - z_norm_pre_update) < 1e-6:
                n_inner_no_norm_gain += 1

            if ep == 0 and len(records) < FIRST_EP_TRACE_CAP:
                records.append({
                    "step": _step,
                    "benefit_exposure": benefit,
                    "drive_level": drive,
                    "energy": energy,
                    "effective_benefit": effective_benefit,
                    "outer_fired": outer_fired,
                    "inner_fired": inner_fired,
                    "z_goal_norm_pre": z_norm_pre_update,
                    "z_goal_norm_post": z_norm_post_update,
                    "is_active_post": is_active_post_update,
                    "contact_event": is_contact,
                })

            if done:
                break

        ep_summaries.append({
            "episode": ep,
            "steps": ep_steps,
            "outer_fires": ep_outer,
            "inner_fires": ep_inner,
            "contacts": ep_contacts,
            "active_steps": ep_active_steps,
        })

    elapsed = time.time() - start_time

    def _stat(arr, key):
        if not arr:
            return None
        return {
            "n": len(arr),
            "mean": float(np.mean(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "p50": float(np.percentile(arr, 50)),
            "p90": float(np.percentile(arr, 90)),
        }

    # Hypothesis dispatch summary
    h_a_drive_collapse = (
        _stat(drives_on_contact, "drive_on_contact") or {"n": 0}
    )
    h_b_threshold_never = max_eff_benefit <= benefit_threshold
    h_c_inner_no_gain_rate = (
        n_inner_no_norm_gain / n_inner_fired
        if n_inner_fired > 0 else None
    )

    summary = {
        "total_eval_steps":          total_steps,
        "n_outer_fired":             n_outer_fired,
        "n_inner_fired":             n_inner_fired,
        "n_contact_events":          n_contact_events,
        "n_inner_no_norm_gain":      n_inner_no_norm_gain,
        "is_active_at_sample_steps": is_active_at_sample,
        "is_active_at_sample_frac":  is_active_at_sample / max(1, total_steps),
        "max_effective_benefit":     max_eff_benefit,
        "max_z_goal_norm":           z_goal_max_norm,
        "benefit_threshold_used":    benefit_threshold,
        "drive_weight_used":         drive_weight,
        "seeding_gain_used":         seeding_gain,
        "stat_benefit_on_outer":     _stat(benefits_on_outer, "benefit"),
        "stat_drive_on_outer":       _stat(drives_on_outer, "drive"),
        "stat_drive_on_contact":     h_a_drive_collapse,
        "stat_energy_on_contact":    _stat(energies_on_contact, "energy"),
        "stat_effective_benefit":    _stat(eff_benefits, "eff_benefit"),
        "stat_z_goal_norm_at_sample": _stat(z_goal_norms, "z_goal_norm"),
    }
    hypotheses = {
        "H_a_drive_collapse_mean_drive_on_contact":
            h_a_drive_collapse.get("mean") if h_a_drive_collapse.get("n", 0) > 0 else None,
        "H_b_threshold_never_crossed":  bool(h_b_threshold_never),
        "H_c_inner_fired_no_norm_gain_rate": h_c_inner_no_gain_rate,
    }

    print(f"\nElapsed: {elapsed:.1f}s", flush=True)
    print(f"Total eval steps: {total_steps}", flush=True)
    print(f"Outer fires (benefit>0.01): {n_outer_fired}", flush=True)
    print(f"Inner fires (eff_ben>{benefit_threshold}): {n_inner_fired}", flush=True)
    print(f"Contacts: {n_contact_events}", flush=True)
    print(f"is_active fraction at sample: {summary['is_active_at_sample_frac']:.4f}", flush=True)
    print(f"Max effective_benefit: {max_eff_benefit:.4f}", flush=True)
    print(f"Max z_goal norm: {z_goal_max_norm:.4f}", flush=True)
    print(f"H_a drive_on_contact mean: {hypotheses['H_a_drive_collapse_mean_drive_on_contact']}", flush=True)
    print(f"H_b threshold_never_crossed: {hypotheses['H_b_threshold_never_crossed']}", flush=True)
    print(f"H_c inner_fired_no_norm_gain_rate: {hypotheses['H_c_inner_fired_no_norm_gain_rate']}", flush=True)

    outcome = "DIAGNOSTIC_COMPLETE" if not DRY_RUN else "DRY_RUN_COMPLETE"

    manifest = {
        "run_id": f"{EXPERIMENT_TYPE}_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}_v3",
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": QUEUE_ID,
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": "diagnostic",
        "evidence_direction": "non_contributory",
        "evidence_direction_note": (
            "Diagnostic instrumentation probe. Logs per-step benefit / drive / "
            "effective_benefit / z_goal.norm() during eval to dispatch the "
            "EXQ-536 z_goal_active_fraction=0.0 root cause among H_a (drive "
            "collapse on contact), H_b (threshold never crossed), H_c (update "
            "fires but z_goal does not grow). Not weighted as evidence."
        ),
        "evidence_direction_per_claim": {
            cid: "non_contributory" for cid in CLAIM_IDS
        },
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "outcome": outcome,
        "summary": summary,
        "hypotheses": hypotheses,
        "ep_summaries": ep_summaries,
        "first_episode_trace": records,
        "config": {
            "n_train_eps":  N_TRAIN_EPS,
            "n_eval_eps":   N_EVAL_EPS,
            "n_steps":      N_STEPS,
            "seed":         SEED,
            "grid_size":    GRID_SIZE,
            "dry_run":      DRY_RUN,
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
