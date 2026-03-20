"""
V3-EXQ-058 — ARC-027 / SD-010: Harm Stream Calibration Validation

Claims: ARC-027, SD-010, MECH-071

ARC-027 diagnosis (from EXQ-027b, EXQ-044, EXQ-045, EXQ-047):
  Fusing nociceptive content into z_world is the root cause of:
  - SD-003 attribution collapse (EXQ-043: causal_sig ≈ 2.7e-5)
  - EXQ-027b: reafference correction strips harm signal (harm_pred_std 0.108→0.008)
  - EXQ-045: MECH-102 escalation reversed (harm signal arrives too late)
  - EXQ-047: z_self/z_world split insufficient (21% calibration gain, needs 3rd stream)

SD-010 implementation (2026-03-20):
  - HarmEncoder(harm_obs → z_harm) in LatentStack — active when use_harm_stream=True
  - agent.sense(obs_body, obs_world, obs_harm=harm_obs) routes through HarmEncoder
  - z_harm bypasses reafference correction by construction
  - harm_obs layout: hazard_field_view[25] + resource_field_view[25] + harm_exposure[1]

This experiment validates the harm stream produces a calibrated signal:
  - E3.harm_eval_z_harm(z_harm) should show positive calibration gap at approach
  - This signal should persist even when reafference is active on z_world
  - Compare calibration_gap_z_harm vs calibration_gap_z_world (replication of EXQ-041 baseline)

Design:
  Training: 500 episodes, full pipeline with use_harm_stream=True
  Eval: 50 episodes × 200 steps
  E3 trains harm_eval_head on z_harm (not z_world) via hazard contact labels
  E2 trains world_forward separately

PASS criteria:
  C1: cal_gap_z_harm_approach > 0         (correct sign — no inversion in harm stream)
  C2: cal_gap_z_harm_approach > 0.03      (meaningful signal above noise floor)
  C3: n_approach_eval >= 30               (sufficient approach samples)
  C4: world_forward_r2 > 0.05            (E2 functional)
  C5: cal_gap_z_harm_approach > cal_gap_z_world_approach   (harm stream BETTER than z_world baseline)
"""

import sys
import random
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
import torch.optim as optim

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig

EXPERIMENT_TYPE = "v3_exq_058_arc027_harm_stream_calibration"
CLAIM_IDS = ["ARC-027", "SD-010", "MECH-071"]

WARMUP_EPISODES = 500
EVAL_EPISODES   = 50
STEPS_PER_EP    = 200
SEED            = 0
LR_E2           = 3e-4
LR_E3           = 3e-4


def _make_agent_and_env(seed: int):
    random.seed(seed); torch.manual_seed(seed)
    env = CausalGridWorldV2(num_hazards=4, num_resources=2, size=12,
                             proximity_harm_scale=0.05, use_proxy_fields=True, seed=seed)
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        alpha_world=0.9,           # SD-008
        use_harm_stream=True,      # SD-010 — nociceptive separation
    )
    agent = REEAgent(cfg)
    return agent, env, cfg


def _train(agent: REEAgent, env: CausalGridWorldV2, n_episodes: int):
    opt_e2 = optim.Adam(list(agent.e2.parameters()), lr=LR_E2)
    opt_e3 = optim.Adam(
        list(agent.e3.harm_eval_head.parameters()) +
        list(agent.e3.harm_eval_z_harm_head.parameters()),  # train both heads
        lr=LR_E3,
    )
    wf_buf: List = []

    for ep in range(n_episodes):
        flat_obs, obs_dict = env.reset()
        agent.reset()
        z_world_prev: Optional[torch.Tensor] = None
        action_prev:  Optional[torch.Tensor] = None

        for _ in range(STEPS_PER_EP):
            obs_body  = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            obs_harm  = obs_dict.get("harm_obs")           # [51] nociceptive vector

            ticks  = agent.clock.advance()
            latent = agent.sense(obs_body, obs_world, obs_harm=obs_harm)
            e1_prior = agent._e1_tick(latent) if ticks["e1_tick"] else torch.zeros(
                1, agent.config.latent.world_dim)

            candidates = agent.generate_trajectories(latent, e1_prior, ticks)
            action = agent.select_action(candidates, ticks)

            flat_obs, harm_signal, done, info, obs_dict = env.step(action)

            # E2 world_forward training
            z_world_curr = latent.z_world.detach()
            if z_world_prev is not None and action_prev is not None:
                wf_buf.append((z_world_prev.cpu(), action_prev.cpu(), z_world_curr.cpu()))
            if len(wf_buf) >= 64:
                zw_b = torch.cat([x[0] for x in wf_buf]).to(agent.device)
                a_b  = torch.cat([x[1] for x in wf_buf]).to(agent.device)
                zt_b = torch.cat([x[2] for x in wf_buf]).to(agent.device)
                pred = agent.e2.world_forward(zw_b, a_b)
                loss_e2 = F.mse_loss(pred, zt_b)
                opt_e2.zero_grad(); loss_e2.backward(); opt_e2.step()
                wf_buf.clear()

            # E3 harm_eval training — on z_harm (SD-010) AND z_world (baseline)
            ttype = info.get("transition_type", "none")
            harm_label = torch.tensor([[1.0 if ttype in ("hazard_approach",
                                        "env_caused_hazard", "agent_caused_hazard")
                                       else 0.0]])
            z_harm = latent.z_harm    # [1, z_harm_dim] — from HarmEncoder
            z_world_now = latent.z_world

            # Train z_harm head
            if z_harm is not None:
                pred_harm_z = agent.e3.harm_eval_z_harm(z_harm)
                loss_z = F.binary_cross_entropy_with_logits(pred_harm_z, harm_label)
                opt_e3.zero_grad(); loss_z.backward(); opt_e3.step()

            # Train z_world head (for baseline comparison)
            pred_harm_w = agent.e3.harm_eval(z_world_now.detach())
            loss_w = F.binary_cross_entropy_with_logits(pred_harm_w, harm_label)
            opt_e3.zero_grad(); loss_w.backward(); opt_e3.step()

            z_world_prev = z_world_curr
            action_prev  = action.detach()

            if done:
                break


def _eval(agent: REEAgent, env: CausalGridWorldV2):
    """Evaluate calibration via z_harm and z_world streams separately."""
    by_ttype_harm:  Dict[str, List[float]] = {"none": [], "hazard_approach": [], "contact": []}
    by_ttype_world: Dict[str, List[float]] = {"none": [], "hazard_approach": [], "contact": []}
    wf_r2_vals: List[float] = []
    z_world_prev = None; action_prev = None

    for _ in range(EVAL_EPISODES):
        flat_obs, obs_dict = env.reset()
        agent.reset(); z_world_prev = None; action_prev = None

        for _ in range(STEPS_PER_EP):
            obs_body  = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            obs_harm  = obs_dict.get("harm_obs")

            ticks  = agent.clock.advance()
            latent = agent.sense(obs_body, obs_world, obs_harm=obs_harm)
            e1_prior = agent._e1_tick(latent) if ticks["e1_tick"] else torch.zeros(
                1, agent.config.latent.world_dim)
            candidates = agent.generate_trajectories(latent, e1_prior, ticks)
            action = agent.select_action(candidates, ticks)

            flat_obs, harm_signal, done, info, obs_dict = env.step(action)

            ttype = info.get("transition_type", "none")
            key = ("contact" if ttype in ("env_caused_hazard", "agent_caused_hazard")
                   else ttype if ttype in ("none", "hazard_approach") else "none")

            with torch.no_grad():
                if latent.z_harm is not None:
                    h_z = torch.sigmoid(agent.e3.harm_eval_z_harm(latent.z_harm))
                    by_ttype_harm[key].append(float(h_z.mean()))
                h_w = torch.sigmoid(agent.e3.harm_eval(latent.z_world))
                by_ttype_world[key].append(float(h_w.mean()))

                z_world_curr = latent.z_world.detach()
                if z_world_prev is not None and action_prev is not None:
                    pred = agent.e2.world_forward(z_world_prev, action_prev)
                    ss_res = (z_world_curr - pred).pow(2).sum().item()
                    ss_tot = (z_world_curr - z_world_curr.mean()).pow(2).sum().item()
                    wf_r2_vals.append(1 - ss_res / (ss_tot + 1e-8))
                z_world_prev = z_world_curr
                action_prev = action.detach()

            if done:
                break

    def mean(lst): return sum(lst) / len(lst) if lst else 0.0

    return {
        "cal_gap_z_harm_approach": mean(by_ttype_harm["hazard_approach"]) - mean(by_ttype_harm["none"]),
        "cal_gap_z_harm_contact":  mean(by_ttype_harm.get("contact", [])) - mean(by_ttype_harm["none"]),
        "cal_gap_z_world_approach": mean(by_ttype_world["hazard_approach"]) - mean(by_ttype_world["none"]),
        "mean_harm_z_harm_approach": mean(by_ttype_harm["hazard_approach"]),
        "mean_harm_z_harm_none":     mean(by_ttype_harm["none"]),
        "mean_harm_z_world_approach": mean(by_ttype_world["hazard_approach"]),
        "mean_harm_z_world_none":    mean(by_ttype_world["none"]),
        "n_approach_eval": len(by_ttype_harm["hazard_approach"]),
        "n_contact_eval":  len(by_ttype_harm.get("contact", [])),
        "n_none_eval":     len(by_ttype_harm["none"]),
        "world_forward_r2": mean(wf_r2_vals),
    }


if __name__ == "__main__":
    import json
    from datetime import datetime, timezone

    agent, env, cfg = _make_agent_and_env(SEED)
    print(f"[EXQ-056] Training {WARMUP_EPISODES} eps — harm stream enabled (use_harm_stream=True)")
    _train(agent, env, WARMUP_EPISODES)

    print("[EXQ-056] Evaluating calibration (z_harm vs z_world)...")
    m = _eval(agent, env)

    # PASS criteria
    c1 = m["cal_gap_z_harm_approach"] > 0
    c2 = m["cal_gap_z_harm_approach"] > 0.03
    c3 = m["n_approach_eval"] >= 30
    c4 = m["world_forward_r2"] > 0.05
    c5 = m["cal_gap_z_harm_approach"] > m["cal_gap_z_world_approach"]
    n_pass = sum([c1, c2, c3, c4, c5])
    status = "PASS" if n_pass >= 4 else "FAIL"

    print(f"\n[EXQ-056] {status} ({n_pass}/5 criteria)")
    print(f"  cal_gap z_harm (approach):  {m['cal_gap_z_harm_approach']:.4f}")
    print(f"  cal_gap z_world (approach): {m['cal_gap_z_world_approach']:.4f}")
    print(f"  world_forward R²: {m['world_forward_r2']:.4f}")
    print(f"  n_approach: {m['n_approach_eval']}")

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result = {
        "run_id": f"{ts}_{EXPERIMENT_TYPE}_v3",
        "experiment_type": EXPERIMENT_TYPE,
        "status": status,
        "claim_ids": CLAIM_IDS,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "evidence_direction": "supports" if status == "PASS" else "weakens",
        "metrics": m,
        "criteria": {"C1": c1, "C2": c2, "C3": c3, "C4": c4, "C5": c5},
        "hyperparams": {
            "warmup_episodes": WARMUP_EPISODES,
            "eval_episodes": EVAL_EPISODES,
            "steps_per_ep": STEPS_PER_EP,
            "alpha_world": 0.9,
            "use_harm_stream": True,
            "seed": SEED,
        },
    }
    out_dir = (
        Path(__file__).resolve().parents[2]
        / "REE_assembly" / "evidence" / "experiments" / EXPERIMENT_TYPE
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{EXPERIMENT_TYPE}_{ts}.json"
    out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"[EXQ-056] Result written to {out_path.name}")
