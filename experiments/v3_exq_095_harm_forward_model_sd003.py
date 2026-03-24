"""
V3-EXQ-095 -- SD-003 Redesign: HarmForwardModel Counterfactual Pipeline

Claims: ARC-033, SD-011, SD-003

This experiment validates the redesigned SD-003 counterfactual attribution pipeline
using HarmForwardModel (ARC-033) instead of HarmBridge.

Background:
EXQ-093/094 confirmed bridge_r2=0 in both experiments -- z_world is architecturally
orthogonal to z_harm (SD-010 design), so HarmBridge(z_world -> z_harm) has nothing
to learn. The bridge approach is infeasible.

Redesign (SD-011, ARC-033):
Instead of bridging from z_world, use a harm-stream forward model:
    z_harm_s_cf = HarmForwardModel(z_harm_s, a_cf)
    causal_sig = E3(z_harm_s_actual) - E3(z_harm_s_cf)

This is tractable because z_harm_s (sensory-discriminative harm, Adelta-pathway analog)
has predictable action-conditional structure: moving away from a hazard reduces
proximity (hazard_field_view), moving toward increases it. The forward model learns
this within the harm stream without any cross-stream bridge.

Biological grounding: Keltner et al. (2006, J Neurosci) -- predictability suppresses
sensory-discriminative pain activity (S1/S2). The brain models expected nociceptive
consequences of voluntary movement (forward model over sensory-discriminative stream),
which is exactly what HarmForwardModel implements.

Protocol: identical to EXQ-093 except:
- HarmForwardModel replaces HarmBridge
- Phase 1 trains harm_fwd on (z_harm_s_t, action_t, z_harm_s_{t+1}) transitions
- Eval uses harm_fwd counterfactuals, not z_world -> z_harm bridge
- harm_obs_a (SD-011 affective stream) is collected and encoded but not in criteria

PASS criteria (ALL 5):
  C1: harm_fwd_r2 > 0.2        -- forward model learns structure (baseline: bridge 0.0)
  C2: causal_approach > 0.001  -- causal signal non-trivial via harm-stream CF
  C3: causal_contact > causal_approach  -- MECH-102 escalation preserved
  C4: calibration_gap > 0.05   -- E3 calibrated on harm stream
  C5: harm_none < harm_approach * 0.90  -- E3 discriminates none vs approach
"""

import sys
import random
from collections import deque
from pathlib import Path
from typing import Dict, List, Deque

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.latent.stack import HarmEncoder, AffectiveHarmEncoder, HarmForwardModel
from ree_core.utils.config import REEConfig


EXPERIMENT_TYPE = "v3_exq_095_harm_forward_model_sd003"
CLAIM_IDS = ["ARC-033", "SD-011", "SD-003"]

HARM_OBS_DIM   = 51
HARM_OBS_A_DIM = 50
Z_HARM_DIM     = 32
Z_HARM_A_DIM   = 16
STRAT_BUF_SIZE  = 2000
MIN_PER_BUCKET  = 4
SAMPLES_PER_BUCKET = 8
MAX_HF_DATA = 4000


def _action_to_onehot(idx: int, n: int, device) -> torch.Tensor:
    v = torch.zeros(1, n, device=device)
    v[0, idx] = 1.0
    return v


def _ttype_bucket(ttype: str) -> str:
    if ttype in ("env_caused_hazard", "agent_caused_hazard"):
        return "contact"
    elif ttype == "hazard_approach":
        return "approach"
    return "none"


def run(
    seed: int = 0,
    phase1_episodes: int = 300,
    phase2_episodes: int = 150,
    eval_episodes: int = 100,
    steps_per_episode: int = 200,
    world_dim: int = 32,
    self_dim: int = 32,
    lr: float = 1e-3,
    alpha_world: float = 0.9,
    harm_scale: float = 0.02,
    proximity_scale: float = 0.05,
    **kwargs,
) -> dict:
    torch.manual_seed(seed)
    random.seed(seed)

    env = CausalGridWorldV2(
        seed=seed, size=12, num_hazards=6, num_resources=3,
        hazard_harm=harm_scale,
        env_drift_interval=5, env_drift_prob=0.2,
        proximity_harm_scale=proximity_scale,
        proximity_benefit_scale=proximity_scale * 0.6,
        proximity_approach_threshold=0.15,
        hazard_field_decay=0.5,
        use_proxy_fields=True,
    )

    config = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=self_dim,
        world_dim=world_dim,
        alpha_world=alpha_world,
        alpha_self=0.3,
        reafference_action_dim=env.action_dim,
        use_harm_stream=True,
        harm_obs_dim=HARM_OBS_DIM,
        z_harm_dim=Z_HARM_DIM,
    )
    agent = REEAgent(config)

    # SD-010: sensory-discriminative harm stream (existing HarmEncoder)
    harm_enc = HarmEncoder(harm_obs_dim=HARM_OBS_DIM, z_harm_dim=Z_HARM_DIM)
    # SD-011: ARC-033 -- harm forward model replacing HarmBridge
    harm_fwd = HarmForwardModel(z_harm_dim=Z_HARM_DIM, action_dim=env.action_dim)
    # SD-011: affective harm stream (collected, not in criteria for this experiment)
    harm_enc_a = AffectiveHarmEncoder(harm_obs_a_dim=HARM_OBS_A_DIM, z_harm_a_dim=Z_HARM_A_DIM)

    num_actions = env.action_dim

    print(
        f"[V3-EXQ-095] SD-003 redesign: HarmForwardModel counterfactual pipeline\n"
        f"  HarmForwardModel: (z_harm_s({Z_HARM_DIM}) + action({env.action_dim})) -> z_harm_s_next\n"
        f"  Replaces HarmBridge (bridge_r2=0 architectural; z_world perp z_harm by SD-010)\n"
        f"  Phases: terrain({phase1_episodes}) -> calib({phase2_episodes}) -> eval({eval_episodes})",
        flush=True,
    )

    # Optimizers
    std_params = [p for n, p in agent.named_parameters()
                  if "harm_eval" not in n and "world_transition" not in n
                  and "world_action_encoder" not in n]
    wf_params  = (list(agent.e2.world_transition.parameters())
                  + list(agent.e2.world_action_encoder.parameters()))

    opt_std     = optim.Adam(std_params, lr=lr)
    opt_wf      = optim.Adam(wf_params, lr=1e-3)
    opt_harm    = optim.Adam(
        list(harm_enc.parameters())
        + list(agent.e3.harm_eval_z_harm_head.parameters()),
        lr=1e-3,
    )
    # ARC-033: HarmForwardModel trained jointly with harm encoder
    opt_hf      = optim.Adam(harm_fwd.parameters(), lr=1e-3)
    opt_e3_harm = optim.Adam(agent.e3.harm_eval_z_harm_head.parameters(), lr=1e-4)

    # World-forward buffer (for E2)
    wf_data: List = []
    # Harm-forward buffer: (z_harm_s_t, action_t, z_harm_s_t1)
    hf_data: List = []

    # -- Phase 1: terrain + harm forward model training ----------------------
    print(f"\n[P1] Terrain + HarmForwardModel training ({phase1_episodes} eps)...", flush=True)
    agent.train(); harm_enc.train(); harm_fwd.train(); harm_enc_a.train()

    for ep in range(phase1_episodes):
        flat_obs, obs_dict = env.reset()
        agent.reset()
        z_world_prev = None; a_prev = None
        z_harm_s_prev = None

        for _ in range(steps_per_episode):
            obs_body  = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            latent = agent.sense(obs_body, obs_world)
            agent.clock.advance()
            z_world_curr = latent.z_world.detach()

            harm_obs = obs_dict.get("harm_obs", torch.zeros(HARM_OBS_DIM)).float()
            z_harm_s_curr = harm_enc(harm_obs.unsqueeze(0)).detach()

            action_idx = random.randint(0, num_actions - 1)
            action = _action_to_onehot(action_idx, num_actions, agent.device)
            agent._last_action = action

            flat_obs, _, done, info, obs_dict = env.step(action)

            harm_obs_next = obs_dict.get("harm_obs", torch.zeros(HARM_OBS_DIM)).float()

            # Train HarmEncoder + E3 harm head jointly on proximity label
            label  = harm_obs[12].unsqueeze(0).unsqueeze(0)
            z_harm_for_train = harm_enc(harm_obs.unsqueeze(0))
            pred   = agent.e3.harm_eval_z_harm(z_harm_for_train)
            loss_he = F.mse_loss(pred, label)
            opt_harm.zero_grad(); loss_he.backward()
            opt_harm.step()

            # ARC-033: Train HarmForwardModel: harm_fwd(z_harm_s_t, a_t) ~= z_harm_s_{t+1}
            z_harm_s_next_target = harm_enc(harm_obs_next.unsqueeze(0)).detach()
            hf_data.append((z_harm_s_curr.cpu(), action.detach().cpu(),
                            z_harm_s_next_target.cpu()))
            if len(hf_data) > MAX_HF_DATA:
                hf_data = hf_data[-MAX_HF_DATA:]

            if len(hf_data) >= 16:
                k = min(32, len(hf_data))
                idxs = torch.randperm(len(hf_data))[:k].tolist()
                zh_b  = torch.cat([hf_data[i][0] for i in idxs]).to(agent.device)
                a_b   = torch.cat([hf_data[i][1] for i in idxs]).to(agent.device)
                zh1_b = torch.cat([hf_data[i][2] for i in idxs]).to(agent.device)
                hf_loss = F.mse_loss(harm_fwd(zh_b, a_b), zh1_b)
                if hf_loss.requires_grad:
                    opt_hf.zero_grad(); hf_loss.backward(); opt_hf.step()

            # World-forward data
            if z_world_prev is not None and a_prev is not None:
                wf_data.append((z_world_prev.cpu(), a_prev.cpu(), z_world_curr.cpu()))
                if len(wf_data) > MAX_HF_DATA:
                    wf_data = wf_data[-MAX_HF_DATA:]

            if len(wf_data) >= 16:
                k = min(32, len(wf_data))
                idxs = torch.randperm(len(wf_data))[:k].tolist()
                zw_b  = torch.cat([wf_data[i][0] for i in idxs]).to(agent.device)
                a_b2  = torch.cat([wf_data[i][1] for i in idxs]).to(agent.device)
                zw1_b = torch.cat([wf_data[i][2] for i in idxs]).to(agent.device)
                wf_loss = F.mse_loss(agent.e2.world_forward(zw_b, a_b2), zw1_b)
                if wf_loss.requires_grad:
                    opt_wf.zero_grad(); wf_loss.backward(); opt_wf.step()

            e1_loss = agent.compute_prediction_loss()
            e2_loss = agent.compute_e2_loss()
            total   = e1_loss + e2_loss
            if total.requires_grad:
                opt_std.zero_grad(); total.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                opt_std.step()

            z_world_prev = z_world_curr
            a_prev = action.detach()
            z_harm_s_prev = z_harm_s_curr
            if done: break

        if (ep + 1) % 100 == 0 or ep == phase1_episodes - 1:
            print(f"  [P1] ep {ep+1}/{phase1_episodes}  hf_buf={len(hf_data)}", flush=True)

    # HarmForwardModel R2 (the key metric replacing bridge_r2)
    harm_fwd_r2 = 0.0
    if len(hf_data) >= 50:
        with torch.no_grad():
            harm_enc.eval(); harm_fwd.eval()
            flat_obs, obs_dict = env.reset()
            agent.reset()
            zh_list = []; zh_pred_list = []
            prev_zh = None; prev_a = None
            for _ in range(min(200, steps_per_episode * 5)):
                harm_obs = obs_dict.get("harm_obs", torch.zeros(HARM_OBS_DIM)).float()
                zh_curr = harm_enc(harm_obs.unsqueeze(0))
                if prev_zh is not None and prev_a is not None:
                    zh_pred = harm_fwd(prev_zh, prev_a)
                    zh_list.append(zh_curr.cpu())
                    zh_pred_list.append(zh_pred.cpu())
                prev_zh = zh_curr.detach()
                action = _action_to_onehot(
                    random.randint(0, num_actions - 1), num_actions, agent.device)
                agent._last_action = action
                prev_a = action.detach()
                latent = agent.sense(obs_dict["body_state"], obs_dict["world_state"])
                agent.clock.advance()
                flat_obs, _, done, _, obs_dict = env.step(action)
                if done: break
            if len(zh_list) >= 20:
                zh_actual = torch.cat(zh_list, dim=0)
                zh_pred   = torch.cat(zh_pred_list, dim=0)
                ss_res = ((zh_actual - zh_pred)**2).sum()
                ss_tot = ((zh_actual - zh_actual.mean(0, keepdim=True))**2).sum()
                harm_fwd_r2 = float((1.0 - ss_res / (ss_tot + 1e-8)).item())
            harm_enc.train(); harm_fwd.train()
    print(f"  HarmForwardModel R2: {harm_fwd_r2:.4f}  (baseline bridge: 0.0)", flush=True)

    # -- Phase 2: E3 calibration (stratified) --------------------------------
    print(f"\n[P2] E3 calibration ({phase2_episodes} eps)...", flush=True)
    for p in agent.parameters(): p.requires_grad_(False)
    for p in harm_enc.parameters(): p.requires_grad_(False)
    for p in harm_fwd.parameters(): p.requires_grad_(False)
    for p in agent.e3.harm_eval_z_harm_head.parameters(): p.requires_grad_(True)

    strat: Dict[str, deque] = {
        "none":     deque(maxlen=STRAT_BUF_SIZE),
        "approach": deque(maxlen=STRAT_BUF_SIZE),
        "contact":  deque(maxlen=STRAT_BUF_SIZE),
    }

    for ep in range(phase2_episodes):
        flat_obs, obs_dict = env.reset()
        agent.reset()
        for _ in range(steps_per_episode):
            with torch.no_grad():
                latent = agent.sense(obs_dict["body_state"], obs_dict["world_state"])
                agent.clock.advance()
            action = _action_to_onehot(
                random.randint(0, num_actions - 1), num_actions, agent.device)
            agent._last_action = action
            flat_obs, _, done, info, obs_dict = env.step(action)
            ttype    = info.get("transition_type", "none")
            harm_obs = obs_dict.get("harm_obs", torch.zeros(HARM_OBS_DIM)).float()
            label    = float(harm_obs[12].item())
            with torch.no_grad():
                zh = harm_enc(harm_obs.unsqueeze(0))
            strat[_ttype_bucket(ttype)].append((zh.detach(), label))

            buckets_ready = [b for b in strat if len(strat[b]) >= MIN_PER_BUCKET]
            if len(buckets_ready) >= 2:
                zh_list = []; lbl_list = []
                for bk in strat:
                    buf = strat[bk]
                    if len(buf) < MIN_PER_BUCKET: continue
                    k = min(SAMPLES_PER_BUCKET, len(buf))
                    for i in random.sample(range(len(buf)), k):
                        zh_list.append(buf[i][0]); lbl_list.append(buf[i][1])
                if len(zh_list) >= 6:
                    zh_b  = torch.cat(zh_list, dim=0).to(agent.device)
                    lbl_b = torch.tensor(lbl_list, dtype=torch.float32,
                                         device=agent.device).unsqueeze(1)
                    pred  = agent.e3.harm_eval_z_harm(zh_b)
                    loss  = F.mse_loss(pred, lbl_b)
                    if loss.requires_grad:
                        opt_e3_harm.zero_grad(); loss.backward()
                        torch.nn.utils.clip_grad_norm_(
                            agent.e3.harm_eval_z_harm_head.parameters(), 0.5)
                        opt_e3_harm.step()
            if done: break

        if (ep + 1) % 50 == 0 or ep == phase2_episodes - 1:
            buf_sz = {k: len(v) for k, v in strat.items()}
            print(f"  [P2] ep {ep+1}/{phase2_episodes}  buf={buf_sz}", flush=True)

    # -- Phase 3: attribution eval -------------------------------------------
    print(f"\n[P3] Attribution eval ({eval_episodes} eps)...", flush=True)
    agent.eval(); harm_enc.eval(); harm_fwd.eval(); harm_enc_a.eval()

    causal_by: Dict[str, List[float]] = {}
    harm_by:   Dict[str, List[float]] = {}

    for ep in range(eval_episodes):
        flat_obs, obs_dict = env.reset()
        agent.reset()
        for _ in range(steps_per_episode):
            with torch.no_grad():
                latent = agent.sense(obs_dict["body_state"], obs_dict["world_state"])
                agent.clock.advance()

                harm_obs  = obs_dict.get("harm_obs", torch.zeros(HARM_OBS_DIM)).float()
                harm_obs_a = obs_dict.get("harm_obs_a", torch.zeros(HARM_OBS_A_DIM)).float()
                z_harm_s_actual = harm_enc(harm_obs.unsqueeze(0))
                harm_actual = float(agent.e3.harm_eval_z_harm(z_harm_s_actual).item())

                # ARC-033: counterfactual via HarmForwardModel (not HarmBridge)
                cf_vals = []
                for ci in range(num_actions):
                    a_cf      = _action_to_onehot(ci, num_actions, agent.device)
                    z_harm_cf = harm_fwd(z_harm_s_actual, a_cf)
                    cf_vals.append(float(agent.e3.harm_eval_z_harm(z_harm_cf).item()))

                causal_sig = harm_actual - float(np.mean(cf_vals))

                # SD-011: encode affective stream (diagnostic only this experiment)
                z_harm_a = harm_enc_a(harm_obs_a.unsqueeze(0))

            action = _action_to_onehot(
                random.randint(0, num_actions - 1), num_actions, agent.device)
            agent._last_action = action
            flat_obs, _, done, info, obs_dict = env.step(action)
            ttype  = info.get("transition_type", "none")
            bucket = _ttype_bucket(ttype)
            causal_by.setdefault(bucket, []).append(causal_sig)
            harm_by.setdefault(bucket, []).append(harm_actual)
            if done: break

    def _mean(lst): return float(np.mean(lst)) if lst else 0.0

    causal_none     = _mean(causal_by.get("none", []))
    causal_approach = _mean(causal_by.get("approach", []))
    causal_contact  = _mean(causal_by.get("contact", []))
    harm_none       = _mean(harm_by.get("none", []))
    harm_approach   = _mean(harm_by.get("approach", []))
    cal_gap         = harm_approach - harm_none
    n_approach      = len(causal_by.get("approach", []))

    print(f"\n  --- EXQ-095 results ---", flush=True)
    print(f"  harm_fwd_r2:        {harm_fwd_r2:.4f}  (baseline bridge: 0.0)", flush=True)
    print(f"  causal_none:        {causal_none:.6f}", flush=True)
    print(f"  causal_approach:    {causal_approach:.6f}", flush=True)
    print(f"  causal_contact:     {causal_contact:.6f}", flush=True)
    print(f"  calibration_gap:    {cal_gap:.4f}", flush=True)
    print(f"  harm_none:          {harm_none:.4f}", flush=True)
    print(f"  harm_approach:      {harm_approach:.4f}", flush=True)
    print(f"  n_approach_eval:    {n_approach}", flush=True)

    c1 = harm_fwd_r2     > 0.2
    c2 = causal_approach > 0.001
    c3 = causal_contact  > causal_approach
    c4 = cal_gap         > 0.05
    c5 = (harm_approach  > 0) and (harm_none < harm_approach * 0.90)

    all_pass = c1 and c2 and c3 and c4 and c5
    status   = "PASS" if all_pass else "FAIL"
    n_met    = sum([c1, c2, c3, c4, c5])

    failure_notes = []
    if not c1: failure_notes.append(
        f"C1 FAIL: harm_fwd_r2={harm_fwd_r2:.4f} <= 0.2 (forward model not learning harm dynamics)")
    if not c2: failure_notes.append(
        f"C2 FAIL: causal_approach={causal_approach:.6f} <= 0.001")
    if not c3: failure_notes.append(
        f"C3 FAIL: causal_contact={causal_contact:.6f} <= causal_approach={causal_approach:.6f}")
    if not c4: failure_notes.append(
        f"C4 FAIL: cal_gap={cal_gap:.4f} <= 0.05")
    if not c5: failure_notes.append(
        f"C5 FAIL: harm_none={harm_none:.4f} not < harm_approach*0.90={harm_approach*0.90:.4f}")

    print(f"\nV3-EXQ-095 verdict: {status}  ({n_met}/5)", flush=True)
    for note in failure_notes:
        print(f"  {note}", flush=True)

    metrics = {
        "harm_fwd_r2":         float(harm_fwd_r2),
        "causal_none":         float(causal_none),
        "causal_approach":     float(causal_approach),
        "causal_contact":      float(causal_contact),
        "calibration_gap":     float(cal_gap),
        "harm_none":           float(harm_none),
        "harm_approach":       float(harm_approach),
        "n_approach_eval":     float(n_approach),
        "crit1_pass":          1.0 if c1 else 0.0,
        "crit2_pass":          1.0 if c2 else 0.0,
        "crit3_pass":          1.0 if c3 else 0.0,
        "crit4_pass":          1.0 if c4 else 0.0,
        "crit5_pass":          1.0 if c5 else 0.0,
        "criteria_met":        float(n_met),
        "fatal_error_count":   0.0,
    }

    failure_section = ""
    if failure_notes:
        failure_section = "\n## Failure Notes\n\n" + "\n".join(f"- {n}" for n in failure_notes)

    summary_markdown = f"""# V3-EXQ-095 -- SD-003 Redesign: HarmForwardModel Counterfactual

**Status:** {status}
**Claims:** ARC-033, SD-011, SD-003
**Supersedes:** EXQ-093/094 (both confirmed bridge_r2=0; bridge approach infeasible)
**World:** CausalGridWorldV2 (6 hazards, 3 resources, 12x12)
**Protocol:** Three-phase (terrain {phase1_episodes} -> calib {phase2_episodes} -> eval {eval_episodes})

## Redesign: HarmForwardModel replaces HarmBridge

HarmBridge (z_world -> z_harm) has bridge_r2=0 by design: SD-010 makes z_world
perp z_harm. HarmForwardModel operates entirely within the harm stream:
    z_harm_s_cf = HarmForwardModel(z_harm_s_actual, a_cf)
    causal_sig = E3(z_harm_s_actual) - E3(z_harm_s_cf)

This is tractable: z_harm_s (sensory-discriminative, Adelta-pathway analog) encodes
proximity that changes predictably with movement. ARC-033 validates this approach.

## Results

| Metric | Value | Baseline (EXQ-093/094) |
|--------|-------|------------------------|
| HarmForwardModel R2 | {harm_fwd_r2:.4f} | bridge_r2=0.0 (infeasible) |
| causal_sig_none | {causal_none:.6f} | -- |
| causal_sig_approach | {causal_approach:.6f} | 0.065 |
| causal_sig_contact | {causal_contact:.6f} | 0.116 |
| calibration_gap_approach | {cal_gap:.4f} | 0.084 |

## PASS Criteria

| Criterion | Result | Value |
|-----------|--------|-------|
| C1: harm_fwd_r2 > 0.2 (forward model learns) | {"PASS" if c1 else "FAIL"} | {harm_fwd_r2:.4f} |
| C2: causal_approach > 0.001 | {"PASS" if c2 else "FAIL"} | {causal_approach:.6f} |
| C3: causal_contact > causal_approach (MECH-102) | {"PASS" if c3 else "FAIL"} | {causal_contact:.6f} vs {causal_approach:.6f} |
| C4: cal_gap > 0.05 | {"PASS" if c4 else "FAIL"} | {cal_gap:.4f} |
| C5: harm_none < harm_approach * 0.90 | {"PASS" if c5 else "FAIL"} | {harm_none:.4f} < {harm_approach*0.90:.4f} |

Criteria met: {n_met}/5 -> **{status}**
{failure_section}
"""

    return {
        "status":             status,
        "metrics":            metrics,
        "summary_markdown":   summary_markdown,
        "claim_ids":          CLAIM_IDS,
        "evidence_direction": "supports" if all_pass else ("mixed" if n_met >= 3 else "weakens"),
        "experiment_type":    EXPERIMENT_TYPE,
        "fatal_error_count":  0,
    }


if __name__ == "__main__":
    import argparse, json
    from datetime import datetime, timezone

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",   type=int, default=0)
    parser.add_argument("--phase1", type=int, default=300)
    parser.add_argument("--phase2", type=int, default=150)
    parser.add_argument("--eval",   type=int, default=100)
    parser.add_argument("--steps",  type=int, default=200)
    args = parser.parse_args()

    result = run(
        seed=args.seed,
        phase1_episodes=args.phase1,
        phase2_episodes=args.phase2,
        eval_episodes=args.eval,
        steps_per_episode=args.steps,
    )

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result["run_timestamp"]      = ts
    result["claim"]              = CLAIM_IDS[0]
    result["verdict"]            = result["status"]
    result["run_id"]             = f"{EXPERIMENT_TYPE}_{ts}_v3"
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
        print(f"  {k}: {v}", flush=True)
