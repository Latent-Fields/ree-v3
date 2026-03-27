#!/opt/local/bin/python3
"""
V3-EXQ-095b -- ARC-033 / SD-011 HarmForwardModel Slow-Learning Extension

Claims: ARC-033, SD-011

EXQ-095a FAIL (SLOW_LEARNING diagnosis, 2026-03-27):
  - Gradients flowing (C1 PASS), loss reduced 4x (C2 PASS), causal signal present (C5 PASS).
  - R2=0.0000 for both FULL and HAZARD models (C3/C4 FAIL).
  - hf_hazard_loss showed monotonic decrease throughout 900 episodes.
  - Diagnosis: model converges toward conditional mean but 900-ep budget insufficient
    to capture action-conditional variance structure.

Fix: 2x phase1 budget (1800 episodes). All else unchanged.

PASS criteria (unchanged from EXQ-095a):
  C1: max_grad_norm_hf > 1e-5        -- gradients flowing
  C2: hf_loss_final < hf_loss_init * 0.5  -- FULL loss halved
  C3: harm_fwd_r2_hazard > 0.10     -- HAZARD model achieves predictive R2
  C4: harm_fwd_r2 > 0.20            -- FULL model achieves predictive R2
  C5: causal_approach > 0.001       -- counterfactual signal non-trivial

If C3 passes but C4 fails: HAZARD_ONLY_WORKS, use hazard-only forward model.
If C3 and C4 pass: BOTH_LEARN, proceed to ARC-033 counterfactual validation.
If C1/C2 pass but C3/C4 still fail at 1800 eps: NOT_LEARNABLE, need larger model.

Supersedes: V3-EXQ-095a
"""

import sys
import random
from collections import deque
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.latent.stack import HarmEncoder, AffectiveHarmEncoder, HarmForwardModel
from ree_core.utils.config import REEConfig


EXPERIMENT_TYPE = "v3_exq_095b_harm_fwd_slow_learning"
CLAIM_IDS = ["ARC-033", "SD-011"]

HARM_OBS_DIM      = 51
HAZARD_DIM        = 25
HARM_OBS_A_DIM    = 50
Z_HARM_DIM        = 32
Z_HARM_HAZARD_DIM = 16
Z_HARM_A_DIM      = 16
STRAT_BUF_SIZE    = 2000
MIN_PER_BUCKET    = 4
SAMPLES_PER_BUCKET = 8
MAX_HF_DATA       = 4000
LOG_INTERVAL      = 100


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
    phase1_episodes: int = 1800,
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

    harm_enc  = HarmEncoder(harm_obs_dim=HARM_OBS_DIM, z_harm_dim=Z_HARM_DIM)
    harm_fwd  = HarmForwardModel(z_harm_dim=Z_HARM_DIM, action_dim=env.action_dim)

    harm_enc_hazard = HarmEncoder(
        harm_obs_dim=HAZARD_DIM, z_harm_dim=Z_HARM_HAZARD_DIM)
    harm_fwd_hazard = HarmForwardModel(
        z_harm_dim=Z_HARM_HAZARD_DIM, action_dim=env.action_dim)

    harm_enc_a = AffectiveHarmEncoder(
        harm_obs_a_dim=HARM_OBS_A_DIM, z_harm_a_dim=Z_HARM_A_DIM)

    num_actions = env.action_dim

    print(
        f"[V3-EXQ-095b] HarmForwardModel slow-learning extension\n"
        f"  FULL model: ({HARM_OBS_DIM}d harm_obs_s -> z_harm_s[{Z_HARM_DIM}])\n"
        f"  HAZARD model: ({HAZARD_DIM}d hazard_only -> z_harm_hz[{Z_HARM_HAZARD_DIM}])\n"
        f"  Phase1: {phase1_episodes} eps (2x EXQ-095a)"
        f"  logging every {LOG_INTERVAL} steps",
        flush=True,
    )

    std_params = [p for n, p in agent.named_parameters()
                  if "harm_eval" not in n and "world_transition" not in n
                  and "world_action_encoder" not in n]
    wf_params  = (list(agent.e2.world_transition.parameters())
                  + list(agent.e2.world_action_encoder.parameters()))

    opt_std    = optim.Adam(std_params, lr=lr)
    opt_wf     = optim.Adam(wf_params, lr=1e-3)
    opt_harm   = optim.Adam(
        list(harm_enc.parameters())
        + list(agent.e3.harm_eval_z_harm_head.parameters()),
        lr=1e-3,
    )
    opt_hf         = optim.Adam(harm_fwd.parameters(), lr=1e-3)
    opt_hf_hazard  = optim.Adam(harm_fwd_hazard.parameters(), lr=1e-3)
    opt_e3_harm    = optim.Adam(
        agent.e3.harm_eval_z_harm_head.parameters(), lr=1e-4)

    wf_data: List         = []
    hf_data: List         = []
    hf_hazard_data: List  = []

    hf_loss_log: List[float]        = []
    hf_hazard_loss_log: List[float] = []
    grad_norm_log: List[float]      = []
    step_counter = 0

    print(
        f"\n[P1] Terrain + HarmForwardModel training ({phase1_episodes} eps)...",
        flush=True,
    )
    agent.train()
    harm_enc.train(); harm_fwd.train()
    harm_enc_hazard.train(); harm_fwd_hazard.train()
    harm_enc_a.train()

    for ep in range(phase1_episodes):
        _, obs_dict = env.reset()
        agent.reset()

        z_world_prev = None
        a_prev = None

        for _ in range(steps_per_episode):
            obs_body  = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            latent = agent.sense(obs_body, obs_world)
            agent.clock.advance()
            z_world_curr = latent.z_world.detach()

            harm_obs = obs_dict.get(
                "harm_obs", torch.zeros(HARM_OBS_DIM)).float()

            z_harm_s_curr = harm_enc(harm_obs.unsqueeze(0)).detach()
            z_harm_hz_curr = harm_enc_hazard(
                harm_obs[:HAZARD_DIM].unsqueeze(0)).detach()

            action_idx = random.randint(0, num_actions - 1)
            action = _action_to_onehot(action_idx, num_actions, agent.device)
            agent._last_action = action

            _, _, done, info, obs_dict = env.step(action)
            harm_obs_next = obs_dict.get(
                "harm_obs", torch.zeros(HARM_OBS_DIM)).float()

            label = harm_obs[12].unsqueeze(0).unsqueeze(0)
            z_harm_for_train = harm_enc(harm_obs.unsqueeze(0))
            pred = agent.e3.harm_eval_z_harm(z_harm_for_train)
            loss_he = F.mse_loss(pred, label)
            opt_harm.zero_grad(); loss_he.backward(); opt_harm.step()

            z_harm_s_next = harm_enc(harm_obs_next.unsqueeze(0)).detach()
            hf_data.append((z_harm_s_curr.cpu(), action.detach().cpu(),
                            z_harm_s_next.cpu()))
            if len(hf_data) > MAX_HF_DATA:
                hf_data = hf_data[-MAX_HF_DATA:]

            z_harm_hz_next = harm_enc_hazard(
                harm_obs_next[:HAZARD_DIM].unsqueeze(0)).detach()
            hf_hazard_data.append((z_harm_hz_curr.cpu(), action.detach().cpu(),
                                   z_harm_hz_next.cpu()))
            if len(hf_hazard_data) > MAX_HF_DATA:
                hf_hazard_data = hf_hazard_data[-MAX_HF_DATA:]

            hf_loss_val = 0.0
            if len(hf_data) >= 16:
                k = min(32, len(hf_data))
                idxs = torch.randperm(len(hf_data))[:k].tolist()
                zh_b  = torch.cat([hf_data[i][0] for i in idxs]).to(agent.device)
                a_b   = torch.cat([hf_data[i][1] for i in idxs]).to(agent.device)
                zh1_b = torch.cat([hf_data[i][2] for i in idxs]).to(agent.device)
                hf_loss = F.mse_loss(harm_fwd(zh_b, a_b), zh1_b)
                hf_loss_val = float(hf_loss.item())
                if hf_loss.requires_grad:
                    opt_hf.zero_grad(); hf_loss.backward(); opt_hf.step()

            hf_hz_loss_val = 0.0
            if len(hf_hazard_data) >= 16:
                k = min(32, len(hf_hazard_data))
                idxs = torch.randperm(len(hf_hazard_data))[:k].tolist()
                zh_h  = torch.cat([hf_hazard_data[i][0] for i in idxs]).to(agent.device)
                a_h   = torch.cat([hf_hazard_data[i][1] for i in idxs]).to(agent.device)
                zh1_h = torch.cat([hf_hazard_data[i][2] for i in idxs]).to(agent.device)
                hf_hz_loss = F.mse_loss(harm_fwd_hazard(zh_h, a_h), zh1_h)
                hf_hz_loss_val = float(hf_hz_loss.item())
                if hf_hz_loss.requires_grad:
                    opt_hf_hazard.zero_grad(); hf_hz_loss.backward()
                    opt_hf_hazard.step()

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

            step_counter += 1
            if step_counter % LOG_INTERVAL == 0:
                hf_loss_log.append(hf_loss_val)
                hf_hazard_loss_log.append(hf_hz_loss_val)
                gnorm = sum(
                    float(p.grad.norm().item())
                    for p in harm_fwd.parameters()
                    if p.grad is not None
                )
                grad_norm_log.append(gnorm)

            if done:
                break

        if (ep + 1) % 100 == 0 or ep == phase1_episodes - 1:
            recent_hf = hf_loss_log[-5:] if hf_loss_log else [0.0]
            recent_hz = hf_hazard_loss_log[-5:] if hf_hazard_loss_log else [0.0]
            recent_gn = grad_norm_log[-5:] if grad_norm_log else [0.0]
            print(
                f"  [P1] ep {ep+1}/{phase1_episodes}"
                f"  hf_buf={len(hf_data)}"
                f"  hf_loss~{sum(recent_hf)/max(1,len(recent_hf)):.5f}"
                f"  hz_loss~{sum(recent_hz)/max(1,len(recent_hz)):.5f}"
                f"  gnorm~{sum(recent_gn)/max(1,len(recent_gn)):.5f}",
                flush=True,
            )

    def _eval_r2(enc, fwd, use_hazard: bool) -> float:
        enc.eval(); fwd.eval()
        _, obs_dict = env.reset()
        agent.reset()
        zh_list: List[torch.Tensor] = []
        zh_pred_list: List[torch.Tensor] = []
        prev_zh = None
        prev_a  = None
        with torch.no_grad():
            for _ in range(min(400, steps_per_episode * 5)):
                harm_obs = obs_dict.get(
                    "harm_obs", torch.zeros(HARM_OBS_DIM)).float()
                inp = harm_obs[:HAZARD_DIM] if use_hazard else harm_obs
                zh_curr = enc(inp.unsqueeze(0))
                if prev_zh is not None and prev_a is not None:
                    zh_pred = fwd(prev_zh, prev_a)
                    zh_list.append(zh_curr.cpu())
                    zh_pred_list.append(zh_pred.cpu())
                prev_zh = zh_curr.detach()
                action  = _action_to_onehot(
                    random.randint(0, num_actions - 1), num_actions, agent.device)
                agent._last_action = action
                prev_a  = action.detach()
                _ = agent.sense(obs_dict["body_state"], obs_dict["world_state"])
                agent.clock.advance()
                _, _, done, _, obs_dict = env.step(action)
                if done:
                    break
        enc.train(); fwd.train()
        if len(zh_list) < 20:
            return 0.0
        zh_actual = torch.cat(zh_list, dim=0)
        zh_pred_t = torch.cat(zh_pred_list, dim=0)
        ss_res = ((zh_actual - zh_pred_t) ** 2).sum()
        ss_tot = ((zh_actual - zh_actual.mean(0, keepdim=True)) ** 2).sum()
        return float((1.0 - ss_res / (ss_tot + 1e-8)).item())

    harm_fwd_r2        = _eval_r2(harm_enc, harm_fwd, use_hazard=False)
    harm_fwd_r2_hazard = _eval_r2(harm_enc_hazard, harm_fwd_hazard, use_hazard=True)

    print(
        f"  FULL  HarmForwardModel R2: {harm_fwd_r2:.4f}",
        flush=True,
    )
    print(
        f"  HAZARD HarmForwardModel R2: {harm_fwd_r2_hazard:.4f}"
        f"  (hazard-only {HAZARD_DIM} dims)",
        flush=True,
    )

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
        _, obs_dict = env.reset()
        agent.reset()
        for _ in range(steps_per_episode):
            with torch.no_grad():
                latent = agent.sense(obs_dict["body_state"], obs_dict["world_state"])
                agent.clock.advance()
            action = _action_to_onehot(
                random.randint(0, num_actions - 1), num_actions, agent.device)
            agent._last_action = action
            _, _, done, info, obs_dict = env.step(action)
            ttype    = info.get("transition_type", "none")
            harm_obs = obs_dict.get("harm_obs", torch.zeros(HARM_OBS_DIM)).float()
            label    = float(harm_obs[12].item())
            with torch.no_grad():
                zh = harm_enc(harm_obs.unsqueeze(0))
            strat[_ttype_bucket(ttype)].append((zh.detach(), label))

            buckets_ready = [b for b in strat if len(strat[b]) >= MIN_PER_BUCKET]
            if len(buckets_ready) >= 2:
                zh_list_cal: List[torch.Tensor] = []
                lbl_list_cal: List[float] = []
                for bk in strat:
                    buf = strat[bk]
                    if len(buf) < MIN_PER_BUCKET:
                        continue
                    k = min(SAMPLES_PER_BUCKET, len(buf))
                    for i in random.sample(range(len(buf)), k):
                        zh_list_cal.append(buf[i][0])
                        lbl_list_cal.append(buf[i][1])
                if len(zh_list_cal) >= 6:
                    zh_b  = torch.cat(zh_list_cal, dim=0).to(agent.device)
                    lbl_b = torch.tensor(
                        lbl_list_cal, dtype=torch.float32,
                        device=agent.device).unsqueeze(1)
                    pred  = agent.e3.harm_eval_z_harm(zh_b)
                    loss  = F.mse_loss(pred, lbl_b)
                    if loss.requires_grad:
                        opt_e3_harm.zero_grad(); loss.backward()
                        torch.nn.utils.clip_grad_norm_(
                            agent.e3.harm_eval_z_harm_head.parameters(), 0.5)
                        opt_e3_harm.step()
            if done:
                break

        if (ep + 1) % 50 == 0 or ep == phase2_episodes - 1:
            buf_sz = {k: len(v) for k, v in strat.items()}
            print(f"  [P2] ep {ep+1}/{phase2_episodes}  buf={buf_sz}", flush=True)

    print(f"\n[P3] Attribution eval ({eval_episodes} eps)...", flush=True)
    agent.eval()
    harm_enc.eval(); harm_fwd.eval()
    harm_enc_hazard.eval(); harm_fwd_hazard.eval()

    use_hazard_for_cf = harm_fwd_r2_hazard > harm_fwd_r2

    causal_by: Dict[str, List[float]] = {}
    harm_by:   Dict[str, List[float]] = {}

    for ep in range(eval_episodes):
        _, obs_dict = env.reset()
        agent.reset()
        for _ in range(steps_per_episode):
            with torch.no_grad():
                latent = agent.sense(obs_dict["body_state"], obs_dict["world_state"])
                agent.clock.advance()

                harm_obs = obs_dict.get(
                    "harm_obs", torch.zeros(HARM_OBS_DIM)).float()
                z_harm_actual = harm_enc(harm_obs.unsqueeze(0))
                harm_actual   = float(
                    agent.e3.harm_eval_z_harm(z_harm_actual).item())

                cf_vals: List[float] = []
                for ci in range(num_actions):
                    a_cf = _action_to_onehot(ci, num_actions, agent.device)
                    if use_hazard_for_cf:
                        z_hz = harm_enc_hazard(harm_obs[:HAZARD_DIM].unsqueeze(0))
                        z_cf = harm_fwd_hazard(z_hz, a_cf)
                        cf_vals.append(float(z_cf.norm().item()))
                    else:
                        z_cf = harm_fwd(z_harm_actual, a_cf)
                        cf_vals.append(
                            float(agent.e3.harm_eval_z_harm(z_cf).item()))

                baseline = float(z_harm_actual.norm().item()) if use_hazard_for_cf \
                    else harm_actual
                causal_sig = baseline - float(np.mean(cf_vals))

            action = _action_to_onehot(
                random.randint(0, num_actions - 1), num_actions, agent.device)
            agent._last_action = action
            _, _, done, info, obs_dict = env.step(action)
            ttype  = info.get("transition_type", "none")
            bucket = _ttype_bucket(ttype)
            causal_by.setdefault(bucket, []).append(causal_sig)
            harm_by.setdefault(bucket, []).append(harm_actual)
            if done:
                break

    def _mean(lst: list) -> float:
        return float(np.mean(lst)) if lst else 0.0

    causal_approach = _mean(causal_by.get("approach", []))
    causal_contact  = _mean(causal_by.get("contact", []))
    harm_none       = _mean(harm_by.get("none", []))
    harm_approach   = _mean(harm_by.get("approach", []))
    cal_gap         = harm_approach - harm_none

    max_grad_norm = max(grad_norm_log) if grad_norm_log else 0.0
    hf_loss_init  = hf_loss_log[0]  if hf_loss_log else 0.0
    hf_loss_final = hf_loss_log[-1] if hf_loss_log else 0.0

    print(f"\n  --- EXQ-095b results ---", flush=True)
    print(f"  max_grad_norm_hf:       {max_grad_norm:.6f}", flush=True)
    print(f"  hf_loss_init:           {hf_loss_init:.6f}", flush=True)
    print(f"  hf_loss_final:          {hf_loss_final:.6f}", flush=True)
    print(f"  harm_fwd_r2 (FULL):     {harm_fwd_r2:.4f}", flush=True)
    print(f"  harm_fwd_r2 (HAZARD):   {harm_fwd_r2_hazard:.4f}", flush=True)
    print(f"  causal_approach:        {causal_approach:.6f}", flush=True)
    print(f"  calibration_gap:        {cal_gap:.4f}", flush=True)

    c1 = max_grad_norm > 1e-5
    c2 = (hf_loss_final < hf_loss_init * 0.5) if hf_loss_init > 1e-8 else False
    c3 = harm_fwd_r2_hazard > 0.10
    c4 = harm_fwd_r2 > 0.20
    c5 = causal_approach > 0.001

    all_pass = c1 and c2 and c3 and c4 and c5
    status   = "PASS" if all_pass else "FAIL"
    n_met    = sum([c1, c2, c3, c4, c5])

    failure_notes = []
    if not c1:
        failure_notes.append(
            f"C1 FAIL: max_grad_norm={max_grad_norm:.6f} -- no gradient flow")
    if not c2:
        failure_notes.append(
            f"C2 FAIL: loss not halved"
            f" (init={hf_loss_init:.5f} final={hf_loss_final:.5f})")
    if not c3:
        failure_notes.append(
            f"C3 FAIL: hazard_r2={harm_fwd_r2_hazard:.4f} -- not action-learnable")
    if not c4:
        failure_notes.append(
            f"C4 FAIL: full_r2={harm_fwd_r2:.4f} -- full harm stream not learnable")
    if not c5:
        failure_notes.append(
            f"C5 FAIL: causal_approach={causal_approach:.6f}")

    if not c1:
        diagnosis = "GRADIENT_DETACH"
    elif not c2 and not c3 and not c4:
        diagnosis = "NOT_LEARNABLE"
    elif c3 and not c4:
        diagnosis = "HAZARD_ONLY_WORKS"
    elif c2 and not c4:
        diagnosis = "SLOW_LEARNING_PERSISTENT"
    elif c4:
        diagnosis = "BOTH_LEARN"
    else:
        diagnosis = "MIXED"

    print(f"\n  Diagnosis: {diagnosis}", flush=True)
    print(f"V3-EXQ-095b verdict: {status}  ({n_met}/5)", flush=True)
    for note in failure_notes:
        print(f"  {note}", flush=True)

    loss_curve_summary = "no data"
    if len(hf_loss_log) >= 3:
        n = len(hf_loss_log)
        t = max(1, n // 3)
        early = sum(hf_loss_log[:t]) / t
        mid   = sum(hf_loss_log[t:2*t]) / max(1, t)
        late  = sum(hf_loss_log[2*t:]) / max(1, n - 2*t)
        loss_curve_summary = (
            f"early={early:.5f}  mid={mid:.5f}  late={late:.5f}")

    metrics = {
        "harm_fwd_r2":           float(harm_fwd_r2),
        "harm_fwd_r2_hazard":    float(harm_fwd_r2_hazard),
        "max_grad_norm_hf":      float(max_grad_norm),
        "hf_loss_init":          float(hf_loss_init),
        "hf_loss_final":         float(hf_loss_final),
        "causal_approach":       float(causal_approach),
        "causal_contact":        float(causal_contact),
        "calibration_gap":       float(cal_gap),
        "harm_none":             float(harm_none),
        "harm_approach":         float(harm_approach),
        "crit1_grad_nonzero":    1.0 if c1 else 0.0,
        "crit2_loss_halved":     1.0 if c2 else 0.0,
        "crit3_hazard_r2":       1.0 if c3 else 0.0,
        "crit4_full_r2":         1.0 if c4 else 0.0,
        "crit5_causal":          1.0 if c5 else 0.0,
        "criteria_met":          float(n_met),
        "phase1_episodes":       float(phase1_episodes),
        "n_hf_loss_points":      float(len(hf_loss_log)),
        "fatal_error_count":     0.0,
    }

    failure_section = ""
    if failure_notes:
        failure_section = "\n## Failure Notes\n\n" + "\n".join(
            f"- {n}" for n in failure_notes)

    summary_markdown = f"""# V3-EXQ-095b -- ARC-033 / SD-011 HarmForwardModel Slow-Learning Extension

**Status:** {status}
**Claims:** ARC-033, SD-011
**Supersedes:** V3-EXQ-095a (SLOW_LEARNING at 900 eps)
**Phase 1:** {phase1_episodes} eps (2x EXQ-095a)
**Parallel tests:** FULL ({HARM_OBS_DIM}d) vs HAZARD ({HAZARD_DIM}d) forward models

## Diagnosis: {diagnosis}

## Key Metrics

| Metric | Value | Criterion |
|--------|-------|-----------|
| Max grad norm (FULL fwd) | {max_grad_norm:.6f} | > 1e-5 (C1) |
| HF loss (early/mid/late) | {loss_curve_summary} | late < 0.5x early (C2) |
| R2 FULL | {harm_fwd_r2:.4f} | > 0.20 (C4) |
| R2 HAZARD | {harm_fwd_r2_hazard:.4f} | > 0.10 (C3) |
| causal_approach | {causal_approach:.6f} | > 0.001 (C5) |
| calibration_gap | {cal_gap:.4f} | -- |

## PASS Criteria

| Criterion | Result | Value |
|-----------|--------|-------|
| C1: gradients flowing | {"PASS" if c1 else "FAIL"} | {max_grad_norm:.6f} |
| C2: loss halved | {"PASS" if c2 else "FAIL"} | {hf_loss_final:.5f} vs {hf_loss_init:.5f} |
| C3: hazard R2 > 0.10 | {"PASS" if c3 else "FAIL"} | {harm_fwd_r2_hazard:.4f} |
| C4: full R2 > 0.20 | {"PASS" if c4 else "FAIL"} | {harm_fwd_r2:.4f} |
| C5: causal_approach > 0.001 | {"PASS" if c5 else "FAIL"} | {causal_approach:.6f} |

Criteria met: {n_met}/5 -> **{status}**

## Diagnosis Codes

- GRADIENT_DETACH: No gradients -- detach bug.
- NOT_LEARNABLE: Signal lacks action-conditional structure at this scale.
- HAZARD_ONLY_WORKS: Use hazard-only ({HAZARD_DIM}d) for ARC-033 forward model.
- SLOW_LEARNING_PERSISTENT: Still converging to mean at 1800 eps -- need larger model.
- BOTH_LEARN: Full pipeline viable -- proceed to ARC-033 counterfactual validation.
{failure_section}
"""

    return {
        "status":              status,
        "metrics":             metrics,
        "summary_markdown":    summary_markdown,
        "claim_ids":           CLAIM_IDS,
        "evidence_direction":  "supports" if all_pass else (
            "mixed" if n_met >= 3 else "weakens"),
        "experiment_type":     EXPERIMENT_TYPE,
        "fatal_error_count":   0,
        "hf_loss_log":         hf_loss_log,
        "hf_hazard_loss_log":  hf_hazard_loss_log,
        "grad_norm_log":       grad_norm_log,
        "diagnosis":           diagnosis,
        "supersedes":          "v3_exq_095a_harm_fwd_diagnosis",
    }


if __name__ == "__main__":
    import argparse
    import json
    from datetime import datetime, timezone

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",   type=int, default=0)
    parser.add_argument("--phase1", type=int, default=1800)
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
