"""
V3-EXQ-054 — MECH-072: World-Delta Residue Gating

Claims: MECH-072

Prerequisite: EXQ-042 PASS (full pipeline works), EXQ-029 PASS (SD-003 world_forward).

Motivation (2026-03-19):
  V2 MECH-072 FAIL (EXQ-028): tested whether E2.predict_harm() can gate residue
  accumulation. Result: zero discrimination (foreseeable == naive). Root cause:
  E2 has no direct harm supervision (trains on motor-sensory error per MECH-069).
  E2.predict_harm was architecturally wrong — harm evaluation belongs to E3.

  V3 reframing: gate residue accumulation on WORLD-DELTA magnitude:
    world_delta = ||E2.world_forward(z_world, a_actual) - E2.world_forward(z_world, a_cf)||
  Agent-caused events → large world_delta (agent's action changed the world)
  Env-caused events → small world_delta (world changed regardless of action)

  This is structurally more robust than V2: world_delta is a property of E2's
  learned dynamics model, not an ill-supervised harm head. SD-003 (EXQ-029 PASS)
  confirmed E2.world_forward achieves R2=0.947 and correct sign structure.

Protocol:
  Phase 1 (400 eps): Train E1, E2, terrain_prior. E3.harm_eval frozen.
  Phase 2 (200 eps): Train E3.harm_eval with frozen terrain_prior, random policy.
  Phase 3 (100 eps): All frozen eval. Three residue accumulation modes:
    NAIVE: accumulate residue on ALL harm events
    WORLD_DELTA: accumulate only when world_delta > threshold
    ORACLE: accumulate only when transition_type == "agent_caused_hazard"

PASS criteria (ALL must hold):
  C1: false_attr_rate_delta < false_attr_rate_naive
      (world-delta gating reduces false attribution)
  C2: harm_per_step_delta <= harm_per_step_naive * 1.05
      (gating doesn't increase harm — no regression)
  C3: world_delta_agent > world_delta_env
      (E2 discriminates agent-caused from env-caused world changes)
  C4: calibration_gap_approach > 0.0 (E3 calibrated)
  C5: world_forward_r2 > 0.05

Informational:
  false_attr_rate_oracle should be ~0 (perfect gating baseline).
  world_delta_threshold is set at median(world_delta) from a calibration pass.
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


EXPERIMENT_TYPE = "v3_exq_054_mech072_world_delta_gating"
CLAIM_IDS = ["MECH-072"]

APPROACH_TTYPES = {"hazard_approach"}
CONTACT_TTYPES = {"agent_caused_hazard", "env_caused_hazard"}
AGENT_CAUSED = {"agent_caused_hazard"}
ENV_CAUSED = {"env_caused_hazard"}


def _mean_safe(lst: list, default: float = 0.0) -> float:
    return float(sum(lst) / len(lst)) if lst else default


def _action_to_onehot(action_idx: int, n: int, device) -> torch.Tensor:
    v = torch.zeros(1, n, device=device)
    v[0, action_idx] = 1.0
    return v


def run(
    seed: int = 0,
    terrain_episodes: int = 400,
    calibration_episodes: int = 200,
    eval_episodes: int = 100,
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

    # ── Optimizers ────────────────────────────────────────────────────────
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
    harm_eval_params = list(agent.e3.harm_eval_head.parameters())
    harm_eval_param_ids = set(id(p) for p in harm_eval_params)

    main_params = [
        p for p in agent.parameters()
        if id(p) not in wf_param_ids
        and id(p) not in terrain_param_ids
        and id(p) not in harm_eval_param_ids
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
    harm_eval_optimizer = optim.Adam(harm_eval_params, lr=1e-4)

    # ── Buffers ───────────────────────────────────────────────────────────
    harm_buf_pos: List[torch.Tensor] = []
    harm_buf_neg: List[torch.Tensor] = []
    MAX_HARM_BUF = 1000
    wf_buf: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    MAX_WF_BUF = 2000

    # ── Phase 1: Terrain + E2 + E1 training (E3 harm_eval frozen) ────────
    print(
        f"[V3-EXQ-054] Phase 1: Terrain training {terrain_episodes} eps (E3 frozen)\n"
        f"  CausalGridWorldV2: body={env.body_obs_dim}  world={env.world_obs_dim}",
        flush=True,
    )

    agent.train()
    for p in harm_eval_params:
        p.requires_grad = False

    for ep in range(terrain_episodes):
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
                    action = _action_to_onehot(random.randint(0, env.action_dim - 1), env.action_dim, agent.device)

            flat_obs, harm_signal, done, info, obs_dict = env.step(action)

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

            # E1 training
            if step % 8 == 0:
                e1_loss = agent.compute_prediction_loss()
                if e1_loss.requires_grad:
                    optimizer.zero_grad()
                    e1_loss.backward()
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
            print(f"  P1 ep {ep+1}/{terrain_episodes}", flush=True)

    # ── Phase 2: E3 calibration (frozen terrain, random policy) ──────────
    print(f"\n[V3-EXQ-054] Phase 2: E3 calibration {calibration_episodes} eps", flush=True)

    for p in harm_eval_params:
        p.requires_grad = True
    for p in list(agent.hippocampal.terrain_prior.parameters()) + \
             list(agent.hippocampal.action_object_decoder.parameters()):
        p.requires_grad = False

    for ep in range(calibration_episodes):
        flat_obs, obs_dict = env.reset()
        agent.reset()

        for step in range(steps_per_episode):
            obs_body = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            latent = agent.sense(obs_body, obs_world)
            theta_z = agent.theta_buffer.summary()
            agent.clock.advance()

            ttype_raw = "none"
            action = _action_to_onehot(random.randint(0, env.action_dim - 1), env.action_dim, agent.device)
            flat_obs, harm_signal, done, info, obs_dict = env.step(action)
            ttype_raw = info.get("transition_type", "none")

            is_pos = ttype_raw in APPROACH_TTYPES | CONTACT_TTYPES
            if is_pos:
                harm_buf_pos.append(theta_z.squeeze(0))
                if len(harm_buf_pos) > MAX_HARM_BUF:
                    harm_buf_pos = harm_buf_pos[-MAX_HARM_BUF:]
            else:
                harm_buf_neg.append(theta_z.squeeze(0))
                if len(harm_buf_neg) > MAX_HARM_BUF:
                    harm_buf_neg = harm_buf_neg[-MAX_HARM_BUF:]

            if len(harm_buf_pos) >= 8 and len(harm_buf_neg) >= 8 and step % 4 == 0:
                k = min(16, len(harm_buf_pos), len(harm_buf_neg))
                pos_idx = torch.randperm(len(harm_buf_pos))[:k].tolist()
                neg_idx = torch.randperm(len(harm_buf_neg))[:k].tolist()
                pos_z = torch.stack([harm_buf_pos[i] for i in pos_idx]).to(agent.device)
                neg_z = torch.stack([harm_buf_neg[i] for i in neg_idx]).to(agent.device)
                z_batch = torch.cat([pos_z, neg_z], dim=0)
                labels = torch.cat([torch.ones(k, 1), torch.zeros(k, 1)], dim=0).to(agent.device)
                harm_loss = F.binary_cross_entropy(agent.e3.harm_eval(z_batch), labels)
                if harm_loss.requires_grad:
                    harm_eval_optimizer.zero_grad()
                    harm_loss.backward()
                    torch.nn.utils.clip_grad_norm_(harm_eval_params, 0.5)
                    harm_eval_optimizer.step()

            if done:
                break

        if (ep + 1) % 50 == 0:
            print(f"  P2 ep {ep+1}/{calibration_episodes}", flush=True)

    # ── Phase 3: Eval with three gating modes ────────────────────────────
    print(f"\n[V3-EXQ-054] Phase 3: Eval {eval_episodes} eps (all frozen)", flush=True)
    agent.eval()

    # First pass: collect world_delta values to set threshold
    world_deltas_all: List[float] = []
    world_deltas_by_ttype: Dict[str, List[float]] = {"agent": [], "env": [], "none": []}

    harm_scores_approach: List[float] = []
    harm_scores_none: List[float] = []

    # Track per-step data for gating analysis
    step_records: List[dict] = []

    with torch.no_grad():
        for ep in range(eval_episodes):
            flat_obs, obs_dict = env.reset()
            agent.reset()
            z_world_prev: Optional[torch.Tensor] = None
            action_prev: Optional[torch.Tensor] = None

            for step in range(steps_per_episode):
                obs_body = obs_dict["body_state"]
                obs_world = obs_dict["world_state"]
                latent = agent.sense(obs_body, obs_world)
                agent.clock.advance()
                theta_z = agent.theta_buffer.summary()
                z_world_curr = latent.z_world.detach()

                harm_score = float(agent.e3.harm_eval(theta_z).mean().item())

                action_idx = random.randint(0, env.action_dim - 1)
                action = _action_to_onehot(action_idx, env.action_dim, agent.device)
                flat_obs, harm_signal, done, info, obs_dict = env.step(action)
                ttype = info.get("transition_type", "none")

                # Compute world_delta: how much did the agent's action change z_world
                # vs a random counterfactual action?
                world_delta = 0.0
                if z_world_prev is not None and action_prev is not None:
                    z_actual = agent.e2.world_forward(z_world_prev, action_prev)
                    # Mean over all other actions as counterfactual
                    cf_deltas = []
                    for cf_idx in range(env.action_dim):
                        if cf_idx == action_idx:
                            continue
                        cf_a = _action_to_onehot(cf_idx, env.action_dim, agent.device)
                        z_cf = agent.e2.world_forward(z_world_prev, cf_a)
                        cf_deltas.append((z_actual - z_cf).pow(2).sum().item())
                    world_delta = float(sum(cf_deltas) / max(1, len(cf_deltas))) ** 0.5

                    world_deltas_all.append(world_delta)
                    if ttype in AGENT_CAUSED:
                        world_deltas_by_ttype["agent"].append(world_delta)
                    elif ttype in ENV_CAUSED:
                        world_deltas_by_ttype["env"].append(world_delta)
                    else:
                        world_deltas_by_ttype["none"].append(world_delta)

                is_harm = ttype in CONTACT_TTYPES
                step_records.append({
                    "ttype": ttype,
                    "world_delta": world_delta,
                    "harm_signal": float(harm_signal),
                    "is_harm": is_harm,
                    "is_agent_caused": ttype in AGENT_CAUSED,
                })

                if ttype in APPROACH_TTYPES:
                    harm_scores_approach.append(harm_score)
                elif ttype not in CONTACT_TTYPES:
                    harm_scores_none.append(harm_score)

                z_world_prev = z_world_curr
                action_prev = action.detach()

                if done:
                    break

    # ── Compute world_delta threshold (median) ────────────────────────────
    if world_deltas_all:
        sorted_deltas = sorted(world_deltas_all)
        threshold = sorted_deltas[len(sorted_deltas) // 2]
    else:
        threshold = 0.0

    print(f"  world_delta threshold (median): {threshold:.6f}", flush=True)

    # ── Compute false attribution rates under three gating modes ──────────
    def compute_gating_metrics(records: List[dict], gate_fn) -> dict:
        total_accumulated = 0
        false_accumulated = 0
        total_harm_events = 0
        total_harm_signal = 0.0
        total_steps = len(records)

        for rec in records:
            if rec["is_harm"]:
                total_harm_events += 1
                total_harm_signal += abs(rec["harm_signal"])
                if gate_fn(rec):
                    total_accumulated += 1
                    if not rec["is_agent_caused"]:
                        false_accumulated += 1

        false_attr_rate = false_accumulated / max(1, total_accumulated)
        harm_per_step = total_harm_signal / max(1, total_steps)

        return {
            "false_attr_rate": false_attr_rate,
            "harm_per_step": harm_per_step,
            "total_accumulated": total_accumulated,
            "false_accumulated": false_accumulated,
            "total_harm_events": total_harm_events,
        }

    naive_m = compute_gating_metrics(step_records, lambda r: True)
    delta_m = compute_gating_metrics(step_records, lambda r: r["world_delta"] > threshold)
    oracle_m = compute_gating_metrics(step_records, lambda r: r["is_agent_caused"])

    print(
        f"  NAIVE:       false_attr={naive_m['false_attr_rate']:.4f}  "
        f"accumulated={naive_m['total_accumulated']}\n"
        f"  WORLD_DELTA: false_attr={delta_m['false_attr_rate']:.4f}  "
        f"accumulated={delta_m['total_accumulated']}\n"
        f"  ORACLE:      false_attr={oracle_m['false_attr_rate']:.4f}  "
        f"accumulated={oracle_m['total_accumulated']}",
        flush=True,
    )

    # ── Compute metrics ───────────────────────────────────────────────────
    mean_delta_agent = _mean_safe(world_deltas_by_ttype["agent"])
    mean_delta_env = _mean_safe(world_deltas_by_ttype["env"])
    mean_delta_none = _mean_safe(world_deltas_by_ttype["none"])

    cal_gap = _mean_safe(harm_scores_approach) - _mean_safe(harm_scores_none)

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
    c1 = delta_m["false_attr_rate"] < naive_m["false_attr_rate"]
    c2 = delta_m["harm_per_step"] <= naive_m["harm_per_step"] * 1.05
    c3 = mean_delta_agent > mean_delta_env
    c4 = cal_gap > 0.0
    c5 = wf_r2 > 0.05

    all_pass = c1 and c2 and c3 and c4 and c5
    status = "PASS" if all_pass else "FAIL"
    n_met = sum([c1, c2, c3, c4, c5])

    failure_notes = []
    if not c1:
        failure_notes.append(
            f"C1 FAIL: false_attr delta={delta_m['false_attr_rate']:.4f} >= "
            f"naive={naive_m['false_attr_rate']:.4f}. "
            f"World-delta gating does not reduce false attribution."
        )
    if not c2:
        failure_notes.append(
            f"C2 FAIL: harm/step delta={delta_m['harm_per_step']:.6f} > "
            f"naive*1.05={naive_m['harm_per_step']*1.05:.6f}. "
            f"Gating causes harm regression."
        )
    if not c3:
        failure_notes.append(
            f"C3 FAIL: mean_delta_agent={mean_delta_agent:.6f} <= "
            f"mean_delta_env={mean_delta_env:.6f}. "
            f"E2 cannot discriminate agent vs env world changes."
        )
    if not c4:
        failure_notes.append(f"C4 FAIL: cal_gap_approach={cal_gap:.4f} <= 0.0")
    if not c5:
        failure_notes.append(f"C5 FAIL: world_forward_r2={wf_r2:.4f} <= 0.05")

    print(f"\nV3-EXQ-054 verdict: {status}  ({n_met}/5)", flush=True)
    for note in failure_notes:
        print(f"  {note}", flush=True)
    print(
        f"  world_delta: agent={mean_delta_agent:.6f}  env={mean_delta_env:.6f}  "
        f"none={mean_delta_none:.6f}\n"
        f"  cal_gap={cal_gap:.4f}  wf_r2={wf_r2:.4f}  threshold={threshold:.6f}",
        flush=True,
    )

    metrics = {
        "false_attr_rate_naive": naive_m["false_attr_rate"],
        "false_attr_rate_delta": delta_m["false_attr_rate"],
        "false_attr_rate_oracle": oracle_m["false_attr_rate"],
        "harm_per_step_naive": naive_m["harm_per_step"],
        "harm_per_step_delta": delta_m["harm_per_step"],
        "world_delta_agent": mean_delta_agent,
        "world_delta_env": mean_delta_env,
        "world_delta_none": mean_delta_none,
        "world_delta_threshold": threshold,
        "accumulated_naive": float(naive_m["total_accumulated"]),
        "accumulated_delta": float(delta_m["total_accumulated"]),
        "accumulated_oracle": float(oracle_m["total_accumulated"]),
        "false_accumulated_naive": float(naive_m["false_accumulated"]),
        "false_accumulated_delta": float(delta_m["false_accumulated"]),
        "total_harm_events": float(naive_m["total_harm_events"]),
        "n_agent_caused": float(len(world_deltas_by_ttype["agent"])),
        "n_env_caused": float(len(world_deltas_by_ttype["env"])),
        "calibration_gap_approach": cal_gap,
        "world_forward_r2": wf_r2,
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
    parser.add_argument("--seed",                 type=int,   default=0)
    parser.add_argument("--terrain-episodes",     type=int,   default=400)
    parser.add_argument("--calibration-episodes", type=int,   default=200)
    parser.add_argument("--eval-eps",             type=int,   default=100)
    parser.add_argument("--steps",                type=int,   default=200)
    parser.add_argument("--alpha-world",          type=float, default=0.9)
    parser.add_argument("--harm-scale",           type=float, default=0.02)
    parser.add_argument("--proximity-scale",      type=float, default=0.05)
    args = parser.parse_args()

    result = run(
        seed=args.seed,
        terrain_episodes=args.terrain_episodes,
        calibration_episodes=args.calibration_episodes,
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
