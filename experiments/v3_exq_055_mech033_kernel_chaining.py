"""
V3-EXQ-055 — MECH-033: Kernel Chaining Interface

Claims: MECH-033

Prerequisite: EXQ-042 PASS (terrain prior works, hippo proposals better than random).

Motivation (2026-03-19):
  V2 MECH-033 FAIL (EXQ-022/23): E2 supplied sensory-state transition kernels (z_gamma)
  that hippocampus chained into rollouts. Result: WITH_CHAIN only 1.8% better than
  NO_CHAIN (random actions). Root cause: chaining z_gamma sensory predictions gives
  negligible signal to E3 because the primitives are wrong — sensory states are high-
  dimensional and uninformative for harm navigation.

  V3 fix: E2 now supplies ACTION-OBJECT kernels (SD-004):
    o_t = E2.action_object(z_world_t, a_t)
  These are compressed representations of world-effects, not sensory states.
  HippocampalModule chains action objects via CEM in action-object space O.

  The V3 claim: the kernel → rollout handoff is load-bearing ONLY when the kernels
  are action-consequence objects. Chaining in sensory space (z_self) should fail;
  chaining in action-object space (O) should succeed.

Protocol:
  Phase 1 (600 eps): Train full pipeline with terrain_prior (same as EXQ-042).
  Phase 2 (eval, 50 eps each): Three conditions on SAME trained agent:
    AO_CHAIN:   Full hippocampal CEM in action-object space (terrain_prior + residue)
    SELF_CHAIN: CEM in z_self space (same CEM budget, but candidates roll out in z_self)
    RANDOM:     Random uniform actions (no chaining)

PASS criteria (ALL must hold):
  C1: harm_per_step_ao < harm_per_step_random * 0.85
      (AO chaining gives >=15% harm reduction vs random)
  C2: harm_per_step_ao < harm_per_step_self
      (AO chaining beats sensory-space chaining)
  C3: contact_rate_ao < contact_rate_random
      (AO chaining reduces hazard contacts)
  C4: calibration_gap_approach > 0.03 (E3 calibrated)
  C5: world_forward_r2 > 0.05

Informational:
  kernel_chaining_advantage = harm_per_step_self - harm_per_step_ao
  If positive, action-object kernels provide genuinely better planning primitives.
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
from ree_core.predictors.e2_fast import Trajectory


EXPERIMENT_TYPE = "v3_exq_055_mech033_kernel_chaining"
CLAIM_IDS = ["MECH-033"]

APPROACH_TTYPES = {"hazard_approach"}
CONTACT_TTYPES = {"agent_caused_hazard", "env_caused_hazard"}
CANDIDATE_HORIZON = 5
N_CEM_CANDIDATES = 8
N_CEM_ITERATIONS = 3
ELITE_FRACTION = 0.3


def _mean_safe(lst: list, default: float = 0.0) -> float:
    return float(sum(lst) / len(lst)) if lst else default


def _action_to_onehot(action_idx: int, n: int, device) -> torch.Tensor:
    v = torch.zeros(1, n, device=device)
    v[0, action_idx] = 1.0
    return v


def _cem_in_self_space(
    agent: REEAgent,
    z_self: torch.Tensor,
    z_world: torch.Tensor,
    n_candidates: int,
    horizon: int,
    n_iterations: int,
    elite_frac: float,
) -> torch.Tensor:
    """
    CEM in z_self space: sample action sequences, roll out in z_self,
    score via residue field over z_world projections, return best action.

    This is the WRONG space for kernel chaining (V2 failure mode) but with
    the same compute budget as AO chaining, so the comparison is fair.
    """
    n_elite = max(1, int(n_candidates * elite_frac))
    device = z_self.device
    action_dim = agent.e2.config.action_dim

    # Initialize action distribution
    a_mean = torch.zeros(1, horizon, action_dim, device=device)
    a_std = torch.ones(1, horizon, action_dim, device=device)

    best_action = None
    best_score = float("inf")

    for _ in range(n_iterations):
        scores = []
        actions_list = []

        for _ in range(n_candidates):
            noise = torch.randn(1, horizon, action_dim, device=device)
            action_seq = a_mean + a_std * noise
            actions_list.append(action_seq)

            # Roll out in z_self + z_world
            traj = agent.e2.rollout_with_world(
                z_self, z_world, action_seq, compute_action_objects=False
            )
            ws = traj.get_world_state_sequence()
            if ws is not None and not torch.isnan(ws).any():
                score = float(agent.residue_field.evaluate_trajectory(ws).sum().item())
            else:
                score = float("inf")
            scores.append(score)

            if score < best_score:
                best_score = score
                best_action = action_seq[:, 0, :].detach()

        # Refit to elite
        indices = sorted(range(len(scores)), key=lambda i: scores[i])[:n_elite]
        elite = torch.cat([actions_list[i] for i in indices], dim=0)
        a_mean = elite.mean(dim=0, keepdim=True)
        a_std = elite.std(dim=0, keepdim=True) + 1e-6

    if best_action is None:
        best_action = torch.randn(1, action_dim, device=device)
    return best_action


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

    print(
        f"[V3-EXQ-055] Phase 1: Training {warmup_episodes} eps — full pipeline\n"
        f"  CausalGridWorldV2: body={env.body_obs_dim}  world={env.world_obs_dim}",
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
                        list(agent.hippocampal.action_object_decoder.parameters()), 1.0
                    )
                    terrain_optimizer.step()
            else:
                action = agent._last_action
                if action is None:
                    action = _action_to_onehot(
                        random.randint(0, env.action_dim - 1), env.action_dim, agent.device
                    )

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

            # E3.harm_eval + E1 training
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

            if done:
                break

        if (ep + 1) % 100 == 0:
            print(f"  ep {ep+1}/{warmup_episodes}  e3_ticks={e3_tick_total}", flush=True)

    # ── Phase 2: Eval — AO_CHAIN vs SELF_CHAIN vs RANDOM ─────────────────
    print(f"\n[V3-EXQ-055] Phase 2: Eval {eval_episodes} eps per condition", flush=True)
    agent.eval()

    def run_condition(mode: str) -> dict:
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

                    if mode == "ao_chain":
                        candidates = agent.generate_trajectories(latent, e1_prior, ticks)
                        action = agent.select_action(candidates, ticks)
                    elif mode == "self_chain":
                        z_self = latent.z_self.detach()
                        action = _cem_in_self_space(
                            agent, z_self, theta_z,
                            n_candidates=N_CEM_CANDIDATES,
                            horizon=CANDIDATE_HORIZON,
                            n_iterations=N_CEM_ITERATIONS,
                            elite_frac=ELITE_FRACTION,
                        )
                    else:  # random
                        action = _action_to_onehot(
                            random.randint(0, env.action_dim - 1),
                            env.action_dim, agent.device,
                        )

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
        cal_gap = _mean_safe(harm_scores_approach) - _mean_safe(harm_scores_none)

        print(
            f"  [{mode:12s}] harm/step={harm_per_step:.6f}  contacts={contact_count}  "
            f"contact_rate={contact_rate:.6f}  cal_gap={cal_gap:.4f}",
            flush=True,
        )
        return {
            "harm_per_step": harm_per_step,
            "contact_rate": contact_rate,
            "contact_count": contact_count,
            "cal_gap_approach": cal_gap,
        }

    ao_result = run_condition("ao_chain")
    self_result = run_condition("self_chain")
    random_result = run_condition("random")

    # ── Compute metrics ───────────────────────────────────────────────────
    kernel_chaining_advantage = self_result["harm_per_step"] - ao_result["harm_per_step"]

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
    c1 = ao_result["harm_per_step"] < random_result["harm_per_step"] * 0.85
    c2 = ao_result["harm_per_step"] < self_result["harm_per_step"]
    c3 = ao_result["contact_rate"] < random_result["contact_rate"]
    c4 = ao_result["cal_gap_approach"] > 0.03
    c5 = wf_r2 > 0.05

    all_pass = c1 and c2 and c3 and c4 and c5
    status = "PASS" if all_pass else "FAIL"
    n_met = sum([c1, c2, c3, c4, c5])

    failure_notes = []
    if not c1:
        failure_notes.append(
            f"C1 FAIL: ao harm/step={ao_result['harm_per_step']:.6f} >= "
            f"random*0.85={random_result['harm_per_step']*0.85:.6f}."
        )
    if not c2:
        failure_notes.append(
            f"C2 FAIL: ao harm/step={ao_result['harm_per_step']:.6f} >= "
            f"self harm/step={self_result['harm_per_step']:.6f}. "
            f"AO chaining not better than sensory-space chaining."
        )
    if not c3:
        failure_notes.append(
            f"C3 FAIL: ao contact_rate={ao_result['contact_rate']:.6f} >= "
            f"random={random_result['contact_rate']:.6f}."
        )
    if not c4:
        failure_notes.append(f"C4 FAIL: ao cal_gap={ao_result['cal_gap_approach']:.4f} <= 0.03")
    if not c5:
        failure_notes.append(f"C5 FAIL: wf_r2={wf_r2:.4f} <= 0.05")

    print(f"\nV3-EXQ-055 verdict: {status}  ({n_met}/5)", flush=True)
    for note in failure_notes:
        print(f"  {note}", flush=True)
    print(
        f"  kernel_chaining_advantage={kernel_chaining_advantage:.6f}\n"
        f"  ao={ao_result['harm_per_step']:.6f}  self={self_result['harm_per_step']:.6f}  "
        f"random={random_result['harm_per_step']:.6f}\n"
        f"  wf_r2={wf_r2:.4f}",
        flush=True,
    )

    metrics = {
        "harm_per_step_ao": ao_result["harm_per_step"],
        "harm_per_step_self": self_result["harm_per_step"],
        "harm_per_step_random": random_result["harm_per_step"],
        "kernel_chaining_advantage": kernel_chaining_advantage,
        "contact_rate_ao": ao_result["contact_rate"],
        "contact_rate_self": self_result["contact_rate"],
        "contact_rate_random": random_result["contact_rate"],
        "cal_gap_approach_ao": ao_result["cal_gap_approach"],
        "cal_gap_approach_self": self_result["cal_gap_approach"],
        "world_forward_r2": wf_r2,
        "e3_tick_total": float(e3_tick_total),
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
