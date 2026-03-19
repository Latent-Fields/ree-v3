"""
V3-EXQ-053 — ARC-018: Rollout Viability Mapping

Claims: ARC-018

Prerequisite: EXQ-042 PASS (terrain prior trains, hippo_quality_gap > 0).

Motivation (2026-03-19):
  V2 ARC-018 FAIL (EXQ-021): tested whether E1 prediction error guides hippocampal
  navigation. Answer: no. VIABILITY_MAPPED (E1 updating) showed zero advantage over
  VIABILITY_FIXED (E1 frozen). Root cause: E1 sensory prediction error is the wrong
  signal for navigation; E3 harm/goal error is the correct one.

  V3 reframing: hippocampus builds viability map indexed by E2 action-object coordinates,
  updated by E3 harm/goal error via terrain_prior behavioral cloning. The map lives in
  residue field geometry over z_world, not in E1 prediction error.

  EXQ-042 PASS showed terrain_prior learns E3's trajectory preferences and produces
  proposals with lower residue than random. This experiment extends that result to
  test whether terrain-guided navigation produces BETTER HARM OUTCOMES — not just
  lower residue scores, but actual harm avoidance in the environment.

Protocol:
  Phase 1 (600 eps): Train full pipeline with terrain_prior (identical to EXQ-042).
  Phase 2 (eval, 50 eps each): Two conditions on the SAME trained agent:
    TERRAIN: Full hippocampal CEM proposals (terrain_prior + residue scoring).
    RANDOM:  Random action shooting (no terrain guidance).
  Compare mean episode harm across conditions.

PASS criteria (ALL must hold):
  C1: harm_per_step_terrain < harm_per_step_random * 0.90
      (terrain-guided navigation avoids at least 10% more harm)
  C2: contact_rate_terrain < contact_rate_random
      (fewer hazard contacts with terrain guidance)
  C3: hippo_quality_gap > 0 (from training — same as EXQ-042 C2)
  C4: calibration_gap_approach > 0.03 (E3 calibrated)
  C5: world_forward_r2 > 0.05

Informational:
  viability_advantage = harm_per_step_random - harm_per_step_terrain
  Larger = terrain-guided navigation more effective at harm avoidance.
"""

import math
import sys
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
import torch.optim as optim

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig


EXPERIMENT_TYPE = "v3_exq_053_arc018_rollout_viability"
CLAIM_IDS = ["ARC-018"]

APPROACH_TTYPES = {"hazard_approach"}
CONTACT_TTYPES = {"agent_caused_hazard", "env_caused_hazard"}
N_RANDOM_COMPARE = 8
CANDIDATE_HORIZON = 5


def _mean_safe(lst: list, default: float = 0.0) -> float:
    return float(sum(lst) / len(lst)) if lst else default


def run(
    seed: int = 0,
    warmup_episodes: int = 600,
    eval_episodes: int = 50,
    steps_per_episode: int = 200,
    alpha_world: float = 0.9,
    alpha_self: float = 0.3,
    harm_scale: float = 0.02,
    proximity_scale: float = 0.05,
    lr: float = 1e-3,
    self_dim: int = 32,
    world_dim: int = 32,
    **kwargs,
) -> dict:
    torch.manual_seed(seed)
    random.seed(seed)

    env = CausalGridWorldV2(
        seed=seed, size=12, num_hazards=4, num_resources=5,
        hazard_harm=harm_scale,
        env_drift_interval=5, env_drift_prob=0.1,
        proximity_harm_scale=proximity_scale,
        proximity_benefit_scale=proximity_scale * 0.6,
        proximity_approach_threshold=0.15,
        hazard_field_decay=0.5,
    )

    config = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=self_dim,
        world_dim=world_dim,
        alpha_world=alpha_world,
        alpha_self=alpha_self,
        reafference_action_dim=env.action_dim,
    )
    agent = REEAgent(config)

    # ── Optimizers (same as EXQ-042) ──────────────────────────────────────
    wf_param_ids = set(
        id(p) for p in
        list(agent.e2.world_transition.parameters()) +
        list(agent.e2.world_action_encoder.parameters())
    )
    terrain_param_ids = set(
        id(p) for p in
        list(agent.hippocampal.terrain_prior.parameters()) +
        list(agent.hippocampal.action_object_decoder.parameters())
    )
    main_params = [
        p for p in agent.parameters()
        if id(p) not in wf_param_ids and id(p) not in terrain_param_ids
    ]
    optimizer = optim.Adam(main_params, lr=lr)
    wf_optimizer = optim.Adam(
        list(agent.e2.world_transition.parameters()) +
        list(agent.e2.world_action_encoder.parameters()),
        lr=1e-3,
    )
    terrain_optimizer = optim.Adam(
        list(agent.hippocampal.terrain_prior.parameters()) +
        list(agent.hippocampal.action_object_decoder.parameters()),
        lr=5e-4,
    )

    # ── Buffers ───────────────────────────────────────────────────────────
    harm_buf_pos: List[torch.Tensor] = []
    harm_buf_neg: List[torch.Tensor] = []
    MAX_HARM_BUF = 1000
    wf_buf: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    MAX_WF_BUF = 2000

    terrain_losses_early: List[float] = []
    terrain_losses_late: List[float] = []
    hippo_residue_scores: List[float] = []
    random_residue_scores: List[float] = []

    print(
        f"[V3-EXQ-053] Phase 1: Training {warmup_episodes} eps — full pipeline + terrain_prior\n"
        f"  CausalGridWorldV2: body={env.body_obs_dim}  world={env.world_obs_dim}\n"
        f"  alpha_world={alpha_world}  harm_scale={harm_scale}",
        flush=True,
    )

    # ── Phase 1: Training (identical to EXQ-042) ─────────────────────────
    agent.train()
    e3_tick_total = 0

    for ep in range(warmup_episodes):
        flat_obs, obs_dict = env.reset()
        agent.reset()
        z_world_prev: Optional[torch.Tensor] = None
        action_prev: Optional[torch.Tensor] = None
        z_self_prev: Optional[torch.Tensor] = None

        for step in range(steps_per_episode):
            obs_body = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]

            latent = agent.sense(obs_body, obs_world)

            if z_self_prev is not None and action_prev is not None:
                agent.record_transition(z_self_prev, action_prev, latent.z_self.detach())

            ticks = agent.clock.advance()
            e1_prior = (
                agent._e1_tick(latent) if ticks["e1_tick"]
                else torch.zeros(1, world_dim, device=agent.device)
            )

            candidates = agent.generate_trajectories(latent, e1_prior, ticks)
            theta_z = agent.theta_buffer.summary()
            z_world_curr = latent.z_world.detach()

            if ticks.get("e3_tick", False) and candidates:
                e3_tick_total += 1
                result = agent.e3.select(candidates, temperature=1.0)
                action = result.selected_action.detach()
                agent._last_action = action

                selected_ao = result.selected_trajectory.get_action_object_sequence()
                if selected_ao is not None:
                    ao_mean_pred = agent.hippocampal._get_terrain_action_object_mean(
                        theta_z, e1_prior=e1_prior.detach()
                    )
                    terrain_loss = F.mse_loss(ao_mean_pred, selected_ao.detach())
                    terrain_optimizer.zero_grad()
                    terrain_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        list(agent.hippocampal.terrain_prior.parameters()) +
                        list(agent.hippocampal.action_object_decoder.parameters()),
                        1.0,
                    )
                    terrain_optimizer.step()

                    t_loss_val = float(terrain_loss.item())
                    if ep < 100:
                        terrain_losses_early.append(t_loss_val)
                    if ep >= warmup_episodes - 100:
                        terrain_losses_late.append(t_loss_val)
            else:
                action = agent._last_action
                if action is None:
                    action_idx = random.randint(0, env.action_dim - 1)
                    action = torch.zeros(1, env.action_dim, device=agent.device)
                    action[0, action_idx] = 1.0

            flat_obs, harm_signal, done, info, obs_dict = env.step(action)
            ttype = info.get("transition_type", "none")

            is_pos = ttype in APPROACH_TTYPES | CONTACT_TTYPES
            if is_pos:
                harm_buf_pos.append(theta_z.squeeze(0))
                if len(harm_buf_pos) > MAX_HARM_BUF:
                    harm_buf_pos = harm_buf_pos[-MAX_HARM_BUF:]
            else:
                harm_buf_neg.append(theta_z.squeeze(0))
                if len(harm_buf_neg) > MAX_HARM_BUF:
                    harm_buf_neg = harm_buf_neg[-MAX_HARM_BUF:]

            if z_world_prev is not None and action_prev is not None:
                wf_buf.append((z_world_prev, action_prev, z_world_curr))
                if len(wf_buf) > MAX_WF_BUF:
                    wf_buf = wf_buf[-MAX_WF_BUF:]
                with torch.no_grad():
                    z_pred = agent.e2.world_forward(z_world_prev, action_prev)
                    agent.e3.update_running_variance(z_world_curr - z_pred)

            z_world_prev = z_world_curr
            z_self_prev = latent.z_self.detach()
            action_prev = action.detach()

            # E3.harm_eval training
            if len(harm_buf_pos) >= 8 and len(harm_buf_neg) >= 8 and step % 8 == 0:
                k = min(16, len(harm_buf_pos), len(harm_buf_neg))
                pos_idx = torch.randperm(len(harm_buf_pos))[:k].tolist()
                neg_idx = torch.randperm(len(harm_buf_neg))[:k].tolist()
                pos_z = torch.stack([harm_buf_pos[i] for i in pos_idx]).to(agent.device)
                neg_z = torch.stack([harm_buf_neg[i] for i in neg_idx]).to(agent.device)
                z_batch = torch.cat([pos_z, neg_z], dim=0)
                labels = torch.cat([torch.ones(k, 1), torch.zeros(k, 1)], dim=0).to(agent.device)
                harm_loss = F.binary_cross_entropy(agent.e3.harm_eval(z_batch), labels)
                e1_loss = agent.compute_prediction_loss()
                total_loss = harm_loss + e1_loss
                if total_loss.requires_grad:
                    optimizer.zero_grad()
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(main_params, 1.0)
                    optimizer.step()

            # E2.world_forward training
            if len(wf_buf) >= 16 and step % 4 == 0:
                k = min(32, len(wf_buf))
                idxs = torch.randperm(len(wf_buf))[:k].tolist()
                zw_t = torch.cat([wf_buf[i][0] for i in idxs]).to(agent.device)
                a_t = torch.cat([wf_buf[i][1] for i in idxs]).to(agent.device)
                zw_t1 = torch.cat([wf_buf[i][2] for i in idxs]).to(agent.device)
                wf_loss = F.mse_loss(agent.e2.world_forward(zw_t, a_t), zw_t1)
                if wf_loss.requires_grad:
                    wf_optimizer.zero_grad()
                    wf_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        list(agent.e2.world_transition.parameters()) +
                        list(agent.e2.world_action_encoder.parameters()), 1.0
                    )
                    wf_optimizer.step()

            # Hippo vs random quality (every 10 steps in last 100 eps)
            if ep >= warmup_episodes - 100 and step % 10 == 0:
                with torch.no_grad():
                    z_self = latent.z_self.detach()
                    hippo_trajs = agent.hippocampal.propose_trajectories(
                        z_world=theta_z, z_self=z_self,
                        num_candidates=N_RANDOM_COMPARE, e1_prior=e1_prior.detach(),
                    )
                    random_trajs = agent.e2.generate_candidates_random(
                        initial_z_self=z_self, initial_z_world=theta_z,
                        num_candidates=N_RANDOM_COMPARE, horizon=CANDIDATE_HORIZON,
                        compute_action_objects=False,
                    )

                    def mean_residue(trajs) -> float:
                        scores = []
                        for t in trajs:
                            ws = t.get_world_state_sequence()
                            if ws is not None and not torch.isnan(ws).any():
                                val = float(agent.residue_field.evaluate_trajectory(ws).mean().item())
                                if not math.isnan(val):
                                    scores.append(val)
                        return float(sum(scores) / len(scores)) if scores else 0.0

                    hippo_residue_scores.append(mean_residue(hippo_trajs))
                    random_residue_scores.append(mean_residue(random_trajs))

            if done:
                break

        if (ep + 1) % 100 == 0 or ep == warmup_episodes - 1:
            t_early = _mean_safe(terrain_losses_early)
            t_late = _mean_safe(terrain_losses_late)
            print(
                f"  ep {ep+1}/{warmup_episodes}  e3_ticks={e3_tick_total}  "
                f"terrain_early={t_early:.4f}  terrain_late={t_late:.4f}",
                flush=True,
            )

    # ── Phase 2: Eval — TERRAIN vs RANDOM ─────────────────────────────────
    print(f"\n[V3-EXQ-053] Phase 2: Eval {eval_episodes} eps per condition", flush=True)
    agent.eval()

    def run_eval_condition(use_terrain: bool, label: str) -> dict:
        total_harm = 0.0
        total_steps = 0
        contact_count = 0
        approach_count = 0
        harm_scores_approach: List[float] = []
        harm_scores_none: List[float] = []

        with torch.no_grad():
            for ep in range(eval_episodes):
                flat_obs, obs_dict = env.reset()
                agent.reset()

                for step in range(steps_per_episode):
                    obs_body = obs_dict["body_state"]
                    obs_world = obs_dict["world_state"]

                    latent = agent.sense(obs_body, obs_world)
                    ticks = agent.clock.advance()
                    e1_prior = (
                        agent._e1_tick(latent) if ticks["e1_tick"]
                        else torch.zeros(1, world_dim, device=agent.device)
                    )
                    theta_z = agent.theta_buffer.summary()

                    harm_score = float(agent.e3.harm_eval(theta_z).mean().item())

                    if use_terrain:
                        candidates = agent.generate_trajectories(latent, e1_prior, ticks)
                        action = agent.select_action(candidates, ticks)
                    else:
                        action_idx = random.randint(0, env.action_dim - 1)
                        action = torch.zeros(1, env.action_dim, device=agent.device)
                        action[0, action_idx] = 1.0

                    flat_obs, harm_signal, done, info, obs_dict = env.step(action)
                    ttype = info.get("transition_type", "none")

                    total_harm += abs(float(harm_signal))
                    total_steps += 1

                    if ttype in CONTACT_TTYPES:
                        contact_count += 1
                    if ttype in APPROACH_TTYPES:
                        approach_count += 1
                        harm_scores_approach.append(harm_score)
                    elif ttype not in CONTACT_TTYPES:
                        harm_scores_none.append(harm_score)

                    if done:
                        break

        harm_per_step = total_harm / max(1, total_steps)
        contact_rate = contact_count / max(1, total_steps)
        mean_approach = _mean_safe(harm_scores_approach)
        mean_none = _mean_safe(harm_scores_none)
        cal_gap = mean_approach - mean_none

        print(
            f"  [{label}] harm/step={harm_per_step:.6f}  contacts={contact_count}  "
            f"contact_rate={contact_rate:.6f}  cal_gap={cal_gap:.4f}  "
            f"n_approach={approach_count}",
            flush=True,
        )
        return {
            "harm_per_step": harm_per_step,
            "contact_rate": contact_rate,
            "contact_count": contact_count,
            "approach_count": approach_count,
            "total_steps": total_steps,
            "cal_gap_approach": cal_gap,
            "mean_harm_eval_approach": mean_approach,
            "mean_harm_eval_none": mean_none,
        }

    terrain_result = run_eval_condition(use_terrain=True, label="TERRAIN")
    random_result = run_eval_condition(use_terrain=False, label="RANDOM")

    # ── Compute metrics ───────────────────────────────────────────────────
    hippo_quality_gap = _mean_safe(random_residue_scores) - _mean_safe(hippo_residue_scores)
    terrain_loss_initial = _mean_safe(terrain_losses_early)
    terrain_loss_final = _mean_safe(terrain_losses_late)
    viability_advantage = random_result["harm_per_step"] - terrain_result["harm_per_step"]

    wf_r2 = 0.0
    if len(wf_buf) >= 32:
        with torch.no_grad():
            k = min(200, len(wf_buf))
            idxs = torch.randperm(len(wf_buf))[:k].tolist()
            zw_t = torch.cat([wf_buf[i][0] for i in idxs]).to(agent.device)
            a_t = torch.cat([wf_buf[i][1] for i in idxs]).to(agent.device)
            zw_t1 = torch.cat([wf_buf[i][2] for i in idxs]).to(agent.device)
            pred = agent.e2.world_forward(zw_t, a_t)
            ss_res = (zw_t1 - pred).pow(2).sum().item()
            ss_tot = (zw_t1 - zw_t1.mean(0, keepdim=True)).pow(2).sum().item()
            wf_r2 = max(0.0, 1.0 - ss_res / (ss_tot + 1e-8))

    # ── PASS / FAIL ───────────────────────────────────────────────────────
    c1 = terrain_result["harm_per_step"] < random_result["harm_per_step"] * 0.90
    c2 = terrain_result["contact_rate"] < random_result["contact_rate"]
    c3 = hippo_quality_gap > 0.0
    c4 = terrain_result["cal_gap_approach"] > 0.03
    c5 = wf_r2 > 0.05

    all_pass = c1 and c2 and c3 and c4 and c5
    status = "PASS" if all_pass else "FAIL"
    n_met = sum([c1, c2, c3, c4, c5])

    failure_notes = []
    if not c1:
        failure_notes.append(
            f"C1 FAIL: harm/step terrain={terrain_result['harm_per_step']:.6f} not < "
            f"random*0.90={random_result['harm_per_step']*0.90:.6f}. "
            f"Terrain navigation doesn't reduce harm enough."
        )
    if not c2:
        failure_notes.append(
            f"C2 FAIL: contact_rate terrain={terrain_result['contact_rate']:.6f} >= "
            f"random={random_result['contact_rate']:.6f}."
        )
    if not c3:
        failure_notes.append(f"C3 FAIL: hippo_quality_gap={hippo_quality_gap:.6f} <= 0")
    if not c4:
        failure_notes.append(f"C4 FAIL: cal_gap_approach={terrain_result['cal_gap_approach']:.4f} <= 0.03")
    if not c5:
        failure_notes.append(f"C5 FAIL: world_forward_r2={wf_r2:.4f} <= 0.05")

    print(f"\nV3-EXQ-053 verdict: {status}  ({n_met}/5)", flush=True)
    for note in failure_notes:
        print(f"  {note}", flush=True)
    print(
        f"  viability_advantage={viability_advantage:.6f}\n"
        f"  terrain harm/step={terrain_result['harm_per_step']:.6f}  "
        f"random harm/step={random_result['harm_per_step']:.6f}\n"
        f"  hippo_quality_gap={hippo_quality_gap:.6f}  wf_r2={wf_r2:.4f}",
        flush=True,
    )

    metrics = {
        "harm_per_step_terrain": terrain_result["harm_per_step"],
        "harm_per_step_random": random_result["harm_per_step"],
        "viability_advantage": viability_advantage,
        "contact_rate_terrain": terrain_result["contact_rate"],
        "contact_rate_random": random_result["contact_rate"],
        "hippo_quality_gap": hippo_quality_gap,
        "calibration_gap_approach": terrain_result["cal_gap_approach"],
        "world_forward_r2": wf_r2,
        "terrain_loss_initial": terrain_loss_initial,
        "terrain_loss_final": terrain_loss_final,
        "e3_tick_total": float(e3_tick_total),
        "n_approach_terrain": float(terrain_result["approach_count"]),
        "n_approach_random": float(random_result["approach_count"]),
        "crit1_pass": 1.0 if c1 else 0.0,
        "crit2_pass": 1.0 if c2 else 0.0,
        "crit3_pass": 1.0 if c3 else 0.0,
        "crit4_pass": 1.0 if c4 else 0.0,
        "crit5_pass": 1.0 if c5 else 0.0,
        "criteria_met": float(n_met),
    }

    return {
        "status": status,
        "metrics": metrics,
        "claim_ids": CLAIM_IDS,
        "evidence_direction": "supports" if all_pass else ("mixed" if n_met >= 3 else "weakens"),
        "experiment_type": EXPERIMENT_TYPE,
        "fatal_error_count": 0,
    }


if __name__ == "__main__":
    import argparse
    import json
    from datetime import datetime, timezone

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",            type=int,   default=0)
    parser.add_argument("--warmup",          type=int,   default=600)
    parser.add_argument("--eval-eps",        type=int,   default=50)
    parser.add_argument("--steps",           type=int,   default=200)
    parser.add_argument("--alpha-world",     type=float, default=0.9)
    parser.add_argument("--harm-scale",      type=float, default=0.02)
    parser.add_argument("--proximity-scale", type=float, default=0.05)
    args = parser.parse_args()

    result = run(
        seed=args.seed,
        warmup_episodes=args.warmup,
        eval_episodes=args.eval_eps,
        steps_per_episode=args.steps,
        alpha_world=args.alpha_world,
        harm_scale=args.harm_scale,
        proximity_scale=args.proximity_scale,
    )

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result["run_timestamp"] = ts
    result["claim"] = CLAIM_IDS[0]
    result["verdict"] = result["status"]
    result["run_id"] = f"{EXPERIMENT_TYPE}_{ts}_v3"
    result["architecture_epoch"] = "ree_hybrid_guardrails_v1"

    out_dir = (
        Path(__file__).resolve().parents[2]
        / "REE_assembly" / "evidence" / "experiments" / EXPERIMENT_TYPE
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{EXPERIMENT_TYPE}_{ts}.json"
    out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print(f"\nResult written to: {out_path}", flush=True)
    print(f"Status: {result['status']}", flush=True)
    for k, v in result["metrics"].items():
        if isinstance(v, float):
            print(f"  {k}: {v:.6f}", flush=True)
