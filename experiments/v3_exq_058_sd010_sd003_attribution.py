"""
V3-EXQ-058 — SD-010: SD-003 Attribution Recovery

Claims: SD-003, SD-010

EXQ-043/044 FAILED: causal_sig collapsed to ~0.0004 (threshold 0.001). Root cause:
harm_eval(z_world) was used for attribution. As terrain training improved, the agent
spent more time at approach states; harm_buf_pos filled with approach-heavy z_world
states; E3 learned harm≈1 for all approach states → all trajectory scores converged
→ causal_sig ≈ 0 (not a failure of attribution, but of the harm signal being hijacked
by world-model feature correlates).

SD-010 fix: z_harm from HarmEncoder(harm_obs) carries ONLY proximity/nociceptive
signals. It cannot be hijacked by layout features in z_world. E3.harm_eval_z_harm
should remain calibrated even as terrain training matures.

harm_bridge: A small linear map nn.Linear(world_dim, harm_obs_dim) trained to
approximate harm_obs from z_world. Used for counterfactual attribution:
  z_harm_cf = harm_enc(harm_bridge(E2.world_forward(z_world, a_cf)))
This bridges E2's world-forward predictions into the harm space.

Three phases (follows EXQ-044 three-phase protocol):
  Phase 1 — terrain training (400 eps, random policy):
    Train E2.world_forward + HarmEncoder + harm_bridge. Standard agent losses.
  Phase 2 — E3 calibration (200 eps, random policy, frozen terrain):
    Train harm_eval_z_harm_head only. Balanced harm_buf using z_harm labels.
  Phase 3 — attribution eval (100 eps):
    Freeze all. Measure causal_sig = mean(harm_eval_z_harm(z_harm_actual))
    - mean(harm_eval_z_harm(z_harm_cf)) per transition type.

PASS criteria (ALL must hold):
  C1: causal_sig_approach > 0.001
  C2: calibration_gap_approach > 0.05
  C3: mean_harm_eval_none < 0.2   (collapse guard: E3 not predicting harm=1 everywhere)
  C4: causal_sig_contact > causal_sig_approach  (MECH-102 escalation)
  C5: n_approach_eval >= 30
"""

import sys
import random
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.latent.stack import HarmEncoder
from ree_core.utils.config import REEConfig


EXPERIMENT_TYPE = "v3_exq_058_sd010_sd003_attribution"
CLAIM_IDS = ["SD-003", "SD-010"]

HARM_OBS_DIM = 51
Z_HARM_DIM   = 32


def _action_to_onehot(action_idx: int, num_actions: int, device) -> torch.Tensor:
    v = torch.zeros(1, num_actions, device=device)
    v[0, action_idx] = 1.0
    return v


def run(
    seed: int = 0,
    phase1_episodes: int = 400,
    phase2_episodes: int = 200,
    eval_episodes: int = 100,
    steps_per_episode: int = 300,
    self_dim: int = 32,
    world_dim: int = 32,
    lr: float = 1e-3,
    alpha_world: float = 0.9,
    alpha_self: float = 0.3,
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
        use_proxy_fields=True,  # SD-010: required for harm_obs in obs_dict
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
        use_harm_stream=True,
        harm_obs_dim=HARM_OBS_DIM,
        z_harm_dim=Z_HARM_DIM,
    )
    agent = REEAgent(config)

    # SD-010: standalone HarmEncoder
    harm_enc = HarmEncoder(harm_obs_dim=HARM_OBS_DIM, z_harm_dim=Z_HARM_DIM)

    # harm_bridge: maps z_world → harm_obs_approx (for counterfactual pipeline)
    harm_bridge = nn.Linear(world_dim, HARM_OBS_DIM)

    num_actions = env.action_dim

    print(
        f"[V3-EXQ-058] SD-010: SD-003 Attribution Recovery (EXQ-044 retest)\n"
        f"  body_obs={env.body_obs_dim}  world_obs={env.world_obs_dim}\n"
        f"  Three phases: terrain({phase1_episodes}) → calibration({phase2_episodes}) → eval({eval_episodes})\n"
        f"  CF pipeline: z_harm_cf = harm_enc(harm_bridge(E2.world_forward(z_world, a_cf)))",
        flush=True,
    )

    # Optimizers — separated by phase
    standard_params = [
        p for n, p in agent.named_parameters()
        if "harm_eval_head" not in n
        and "harm_eval_z_harm_head" not in n
        and "world_transition" not in n
        and "world_action_encoder" not in n
    ]
    world_forward_params = (
        list(agent.e2.world_transition.parameters())
        + list(agent.e2.world_action_encoder.parameters())
    )

    optimizer            = optim.Adam(standard_params, lr=lr)
    world_forward_opt    = optim.Adam(world_forward_params, lr=1e-3)
    harm_enc_opt         = optim.Adam(harm_enc.parameters(), lr=1e-3)
    harm_bridge_opt      = optim.Adam(harm_bridge.parameters(), lr=1e-3)
    harm_z_harm_opt      = optim.Adam(agent.e3.harm_eval_z_harm_head.parameters(), lr=1e-4)

    wf_data:   List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    bridge_data: List[Tuple[torch.Tensor, torch.Tensor]] = []   # (z_world, harm_obs)
    MAX_WF = 5000
    MAX_BR = 5000

    # ── Phase 1: Terrain training ─────────────────────────────────────────────
    print(f"\n[V3-EXQ-058] Phase 1: Terrain training ({phase1_episodes} eps)...", flush=True)
    agent.train()
    harm_enc.train()
    harm_bridge.train()

    p1_counts: Dict[str, int] = {}

    for ep in range(phase1_episodes):
        flat_obs, obs_dict = env.reset()
        agent.reset()
        z_world_prev = None
        a_prev = None

        for _ in range(steps_per_episode):
            obs_body  = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]

            latent = agent.sense(obs_body, obs_world)
            agent.clock.advance()
            z_world_curr = latent.z_world.detach()

            action_idx = random.randint(0, num_actions - 1)
            action = _action_to_onehot(action_idx, num_actions, agent.device)
            agent._last_action = action

            flat_obs, harm_signal, done, info, obs_dict = env.step(action)
            ttype = info.get("transition_type", "none")
            p1_counts[ttype] = p1_counts.get(ttype, 0) + 1

            # World-forward data
            if z_world_prev is not None and a_prev is not None:
                wf_data.append((z_world_prev.cpu(), a_prev.cpu(), z_world_curr.cpu()))
                if len(wf_data) > MAX_WF:
                    wf_data = wf_data[-MAX_WF:]

            # Bridge data: (z_world, harm_obs) — for harm_bridge training
            harm_obs_new = obs_dict.get("harm_obs", torch.zeros(HARM_OBS_DIM))
            bridge_data.append((z_world_curr.cpu(), harm_obs_new.cpu().float()))
            if len(bridge_data) > MAX_BR:
                bridge_data = bridge_data[-MAX_BR:]

            # Train HarmEncoder: MSE on normalized hazard proximity label.
            # harm_obs_new[12] = center of hazard_field_view = hazard_field[agent] / hazard_max ∈ [0,1].
            # Raw hazard_field_at_agent is unbounded (>1) — saturates Sigmoid head.
            hazard_label = harm_obs_new[12].unsqueeze(0).unsqueeze(0).detach().float()
            harm_obs_t   = harm_obs_new.unsqueeze(0).float()
            z_harm_new   = harm_enc(harm_obs_t)
            pred_zh      = agent.e3.harm_eval_z_harm(z_harm_new)
            loss_harm    = F.mse_loss(pred_zh, hazard_label)
            harm_enc_opt.zero_grad()
            loss_harm.backward()
            torch.nn.utils.clip_grad_norm_(harm_enc.parameters(), 0.5)
            harm_enc_opt.step()

            # Train world_forward
            if len(wf_data) >= 16:
                k = min(32, len(wf_data))
                idxs = torch.randperm(len(wf_data))[:k].tolist()
                zw_b  = torch.cat([wf_data[i][0] for i in idxs]).to(agent.device)
                a_b   = torch.cat([wf_data[i][1] for i in idxs]).to(agent.device)
                zw1_b = torch.cat([wf_data[i][2] for i in idxs]).to(agent.device)
                wf_loss = F.mse_loss(agent.e2.world_forward(zw_b, a_b), zw1_b)
                if wf_loss.requires_grad:
                    world_forward_opt.zero_grad()
                    wf_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        agent.e2.world_transition.parameters(), 0.5)
                    world_forward_opt.step()

            # Train harm_bridge: MSE(harm_bridge(z_world), harm_obs)
            if len(bridge_data) >= 16:
                k = min(32, len(bridge_data))
                idxs = torch.randperm(len(bridge_data))[:k].tolist()
                zw_br = torch.cat([bridge_data[i][0] for i in idxs]).to(agent.device)
                ho_br = torch.cat([bridge_data[i][1].unsqueeze(0) for i in idxs]).to(agent.device)
                bridge_pred = harm_bridge(zw_br)
                bridge_loss = F.mse_loss(bridge_pred, ho_br)
                harm_bridge_opt.zero_grad()
                bridge_loss.backward()
                harm_bridge_opt.step()

            # Standard losses
            e1_loss = agent.compute_prediction_loss()
            e2_loss = agent.compute_e2_loss()
            total_loss = e1_loss + e2_loss
            if total_loss.requires_grad:
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                optimizer.step()

            z_world_prev = z_world_curr
            a_prev = action.detach()

            if done:
                break

        if (ep + 1) % 100 == 0 or ep == phase1_episodes - 1:
            approach = p1_counts.get("hazard_approach", 0)
            contact  = (p1_counts.get("env_caused_hazard", 0)
                        + p1_counts.get("agent_caused_hazard", 0))
            print(
                f"  [P1] ep {ep+1}/{phase1_episodes}  approach={approach}  contact={contact}",
                flush=True,
            )

    # World-forward R2
    wf_r2 = 0.0
    if len(wf_data) >= 20:
        n = len(wf_data)
        n_tr = int(n * 0.8)
        with torch.no_grad():
            zw_all  = torch.cat([d[0] for d in wf_data]).to(agent.device)
            a_all   = torch.cat([d[1] for d in wf_data]).to(agent.device)
            zw1_all = torch.cat([d[2] for d in wf_data]).to(agent.device)
            pred_all  = agent.e2.world_forward(zw_all, a_all)
            pred_test = pred_all[n_tr:]
            tgt_test  = zw1_all[n_tr:]
            if pred_test.shape[0] > 0:
                ss_res = ((tgt_test - pred_test) ** 2).sum()
                ss_tot = ((tgt_test - tgt_test.mean(0, keepdim=True)) ** 2).sum()
                wf_r2 = float((1 - ss_res / (ss_tot + 1e-8)).item())
    print(f"  world_forward R2 (test): {wf_r2:.4f}", flush=True)

    # ── Phase 2: E3 calibration ───────────────────────────────────────────────
    print(f"\n[V3-EXQ-058] Phase 2: E3 calibration ({phase2_episodes} eps, random policy)...",
          flush=True)

    # Freeze everything except harm_eval_z_harm_head
    for param in agent.parameters():
        param.requires_grad_(False)
    for param in harm_enc.parameters():
        param.requires_grad_(False)
    for param in harm_bridge.parameters():
        param.requires_grad_(False)
    for param in agent.e3.harm_eval_z_harm_head.parameters():
        param.requires_grad_(True)

    harm_buf_pos: List[torch.Tensor] = []  # z_harm at harm events
    harm_buf_neg: List[torch.Tensor] = []  # z_harm at safe steps
    MAX_BUF = 2000

    p2_counts: Dict[str, int] = {}

    for ep in range(phase2_episodes):
        flat_obs, obs_dict = env.reset()
        agent.reset()

        for _ in range(steps_per_episode):
            obs_body  = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]

            with torch.no_grad():
                latent      = agent.sense(obs_body, obs_world)
                agent.clock.advance()
                z_world_cur = latent.z_world.detach()

            action_idx = random.randint(0, num_actions - 1)
            action = _action_to_onehot(action_idx, num_actions, agent.device)
            agent._last_action = action

            flat_obs, harm_signal, done, info, obs_dict = env.step(action)
            ttype = info.get("transition_type", "none")
            p2_counts[ttype] = p2_counts.get(ttype, 0) + 1

            harm_obs_new = obs_dict.get("harm_obs", torch.zeros(HARM_OBS_DIM))
            with torch.no_grad():
                z_harm_new = harm_enc(harm_obs_new.unsqueeze(0).float())

            # Buffer: harm = 1 for hazard events, 0 for safe
            if harm_signal < 0 or ttype in ("env_caused_hazard", "agent_caused_hazard",
                                              "hazard_approach"):
                harm_buf_pos.append(z_harm_new.detach())
                if len(harm_buf_pos) > MAX_BUF:
                    harm_buf_pos = harm_buf_pos[-MAX_BUF:]
            else:
                harm_buf_neg.append(z_harm_new.detach())
                if len(harm_buf_neg) > MAX_BUF:
                    harm_buf_neg = harm_buf_neg[-MAX_BUF:]

            # Train harm_eval_z_harm: Fix2 — include E2-predicted z_harm (via harm_bridge)
            if len(harm_buf_pos) >= 4 and len(harm_buf_neg) >= 4:
                k_pos = min(16, len(harm_buf_pos))
                k_neg = min(16, len(harm_buf_neg))
                pos_idx = torch.randperm(len(harm_buf_pos))[:k_pos].tolist()
                neg_idx = torch.randperm(len(harm_buf_neg))[:k_neg].tolist()

                zh_pos = torch.cat([harm_buf_pos[i] for i in pos_idx], dim=0)
                zh_neg = torch.cat([harm_buf_neg[i] for i in neg_idx], dim=0)

                # Fix2: also include E2-predicted counterfactual z_harm
                with torch.no_grad():
                    a_rand_pos = torch.zeros(k_pos, num_actions, device=agent.device)
                    a_rand_pos[torch.arange(k_pos), torch.randint(0, num_actions, (k_pos,))] = 1.0
                    a_rand_neg = torch.zeros(k_neg, num_actions, device=agent.device)
                    a_rand_neg[torch.arange(k_neg), torch.randint(0, num_actions, (k_neg,))] = 1.0

                    # Project pos z_world (approximate) back from z_harm via reverse lookup
                    # Use the world-forward on a dummy z_world — this is approximate but
                    # ensures harm_eval_z_harm sees z_harm from CF distribution
                    zw_pos_approx = agent.e2.world_forward(z_world_cur.expand(k_pos, -1), a_rand_pos)
                    zw_neg_approx = agent.e2.world_forward(z_world_cur.expand(k_neg, -1), a_rand_neg)
                    harm_obs_pos_cf = harm_bridge(zw_pos_approx)
                    harm_obs_neg_cf = harm_bridge(zw_neg_approx)
                    zh_pos_cf = harm_enc(harm_obs_pos_cf)
                    zh_neg_cf = harm_enc(harm_obs_neg_cf)

                zh_batch = torch.cat([zh_pos, zh_neg, zh_pos_cf, zh_neg_cf], dim=0)
                target = torch.cat([
                    torch.ones(k_pos,  1, device=agent.device),
                    torch.zeros(k_neg, 1, device=agent.device),
                    torch.ones(k_pos,  1, device=agent.device),
                    torch.zeros(k_neg, 1, device=agent.device),
                ], dim=0)

                pred = agent.e3.harm_eval_z_harm(zh_batch)
                loss = F.mse_loss(pred, target)
                if loss.requires_grad:
                    harm_z_harm_opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        agent.e3.harm_eval_z_harm_head.parameters(), 0.5)
                    harm_z_harm_opt.step()

            if done:
                break

        if (ep + 1) % 50 == 0 or ep == phase2_episodes - 1:
            approach = p2_counts.get("hazard_approach", 0)
            print(
                f"  [P2] ep {ep+1}/{phase2_episodes}  "
                f"buf_pos={len(harm_buf_pos)}  buf_neg={len(harm_buf_neg)}  "
                f"approach={approach}",
                flush=True,
            )

    # ── Phase 3: Attribution eval ─────────────────────────────────────────────
    print(f"\n[V3-EXQ-058] Phase 3: Attribution eval ({eval_episodes} eps)...", flush=True)
    agent.eval()
    harm_enc.eval()
    harm_bridge.eval()

    causal_by_ttype:  Dict[str, List[float]] = {}
    harm_by_ttype:    Dict[str, List[float]] = {}
    eval_counts:      Dict[str, int] = {}

    for ep in range(eval_episodes):
        flat_obs, obs_dict = env.reset()
        agent.reset()

        for _ in range(steps_per_episode):
            obs_body  = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]

            with torch.no_grad():
                latent    = agent.sense(obs_body, obs_world)
                agent.clock.advance()
                z_world   = latent.z_world

                # z_harm for actual action (from current obs)
                harm_obs_curr = obs_dict.get("harm_obs", torch.zeros(HARM_OBS_DIM))
                z_harm_actual = harm_enc(harm_obs_curr.unsqueeze(0).float())
                harm_actual   = float(agent.e3.harm_eval_z_harm(z_harm_actual).item())

                # Counterfactual: E2-predict z_world for each action, bridge to harm_obs
                cf_harm_vals = []
                for cf_idx in range(num_actions):
                    a_cf = _action_to_onehot(cf_idx, num_actions, agent.device)
                    z_world_cf      = agent.e2.world_forward(z_world, a_cf)
                    harm_obs_cf_approx = harm_bridge(z_world_cf)
                    z_harm_cf       = harm_enc(harm_obs_cf_approx)
                    cf_harm_vals.append(
                        float(agent.e3.harm_eval_z_harm(z_harm_cf).item())
                    )

                mean_cf_harm = float(np.mean(cf_harm_vals))
                causal_sig   = harm_actual - mean_cf_harm

            action_idx = random.randint(0, num_actions - 1)
            action = _action_to_onehot(action_idx, num_actions, agent.device)
            agent._last_action = action

            flat_obs, harm_signal, done, info, obs_dict = env.step(action)
            ttype = info.get("transition_type", "none")
            eval_counts[ttype] = eval_counts.get(ttype, 0) + 1

            causal_by_ttype.setdefault(ttype, []).append(causal_sig)
            harm_by_ttype.setdefault(ttype, []).append(harm_actual)

            if done:
                break

    # ── Metrics ──────────────────────────────────────────────────────────────
    def _mean(lst): return float(np.mean(lst)) if lst else 0.0

    none_causal     = causal_by_ttype.get("none", [])
    approach_causal = causal_by_ttype.get("hazard_approach", [])
    contact_causal  = (causal_by_ttype.get("agent_caused_hazard", [])
                       + causal_by_ttype.get("env_caused_hazard", []))

    causal_sig_none     = _mean(none_causal)
    causal_sig_approach = _mean(approach_causal)
    causal_sig_contact  = _mean(contact_causal)

    none_harm    = harm_by_ttype.get("none", [])
    mean_harm_none = _mean(none_harm)

    calibration_gap_approach = _mean(
        [h for h in harm_by_ttype.get("hazard_approach", [])]
    ) - mean_harm_none

    n_approach_eval = len(approach_causal)

    print(f"\n  --- SD-003 Attribution with SD-010 (EXQ-058) ---", flush=True)
    print(f"  world_forward R2: {wf_r2:.4f}", flush=True)
    print(f"  causal_sig by ttype:", flush=True)
    print(f"    none:            {causal_sig_none:.6f}  n={len(none_causal)}", flush=True)
    print(f"    hazard_approach: {causal_sig_approach:.6f}  n={len(approach_causal)}", flush=True)
    print(f"    contact:         {causal_sig_contact:.6f}  n={len(contact_causal)}", flush=True)
    print(f"  calibration_gap_approach: {calibration_gap_approach:.4f}", flush=True)
    print(f"  mean_harm_eval_none: {mean_harm_none:.4f}", flush=True)
    print(f"  n_approach_eval: {n_approach_eval}", flush=True)

    # ── PASS / FAIL ──────────────────────────────────────────────────────────
    c1 = causal_sig_approach  > 0.001
    c2 = calibration_gap_approach > 0.05
    c3 = mean_harm_none       < 0.2
    c4 = causal_sig_contact   > causal_sig_approach
    c5 = n_approach_eval      >= 30

    all_pass = c1 and c2 and c3 and c4 and c5
    status   = "PASS" if all_pass else "FAIL"
    n_met    = sum([c1, c2, c3, c4, c5])

    failure_notes = []
    if not c1:
        failure_notes.append(
            f"C1 FAIL: causal_sig_approach={causal_sig_approach:.6f} <= 0.001. "
            f"SD-003 attribution still collapsed with z_harm. "
            f"harm_enc may not be learning proximity signal or harm_bridge misaligned."
        )
    if not c2:
        failure_notes.append(
            f"C2 FAIL: calibration_gap_approach={calibration_gap_approach:.4f} <= 0.05. "
            f"E3.harm_eval_z_harm not calibrated for approach states."
        )
    if not c3:
        failure_notes.append(
            f"C3 FAIL: mean_harm_eval_none={mean_harm_none:.4f} >= 0.2. "
            f"E3 predicting harm everywhere — collapse still occurring."
        )
    if not c4:
        failure_notes.append(
            f"C4 FAIL: causal_sig_contact={causal_sig_contact:.6f} <= "
            f"causal_sig_approach={causal_sig_approach:.6f}. "
            f"MECH-102 escalation not confirmed."
        )
    if not c5:
        failure_notes.append(
            f"C5 FAIL: n_approach_eval={n_approach_eval} < 30. "
            f"Insufficient approach events for reliable measurement."
        )

    print(f"\nV3-EXQ-058 verdict: {status}  ({n_met}/5)", flush=True)
    for note in failure_notes:
        print(f"  {note}", flush=True)

    metrics = {
        "alpha_world":               float(alpha_world),
        "world_forward_r2":          float(wf_r2),
        "causal_sig_none":           float(causal_sig_none),
        "causal_sig_approach":       float(causal_sig_approach),
        "causal_sig_contact":        float(causal_sig_contact),
        "calibration_gap_approach":  float(calibration_gap_approach),
        "mean_harm_eval_none":       float(mean_harm_none),
        "n_approach_eval":           float(n_approach_eval),
        "n_contact_eval":            float(len(contact_causal)),
        "n_none_eval":               float(len(none_causal)),
        "crit1_pass":                1.0 if c1 else 0.0,
        "crit2_pass":                1.0 if c2 else 0.0,
        "crit3_pass":                1.0 if c3 else 0.0,
        "crit4_pass":                1.0 if c4 else 0.0,
        "crit5_pass":                1.0 if c5 else 0.0,
        "criteria_met":              float(n_met),
        "fatal_error_count":         0.0,
    }

    failure_section = ""
    if failure_notes:
        failure_section = "\n## Failure Notes\n\n" + "\n".join(f"- {n}" for n in failure_notes)

    summary_markdown = f"""# V3-EXQ-058 — SD-010: SD-003 Attribution Recovery

**Status:** {status}
**Claims:** SD-003, SD-010
**World:** CausalGridWorldV2 (6 hazards, 3 resources)
**Retests:** EXQ-043/044 (causal_sig threshold 0.001)
**Protocol:** Three-phase (terrain {phase1_episodes} → calibration {phase2_episodes} → eval {eval_episodes})
**alpha_world:** {alpha_world}  |  **Seed:** {seed}

## Context

EXQ-043/044 FAILED: causal_sig ≈ 0.0004 (threshold 0.001). Root cause: harm_eval(z_world)
collapsed as terrain training matured — approach states dominated harm_buf, E3 predicted
harm≈1 for all states, causal_sig → 0.

SD-010 fix: z_harm from HarmEncoder(harm_obs) is isolated from terrain-correlated z_world
features. CF pipeline: z_harm_cf = harm_enc(harm_bridge(E2.world_forward(z_world, a_cf))).

## Results — SD-003 Attribution

| Transition type | causal_sig |
|---|---|
| none (safe locomotion) | {causal_sig_none:.6f} |
| hazard_approach        | {causal_sig_approach:.6f} |
| contact (combined)     | {causal_sig_contact:.6f} |

- **world_forward R2**: {wf_r2:.4f}
- **calibration_gap_approach**: {calibration_gap_approach:.4f}
- **mean_harm_eval_none**: {mean_harm_none:.4f}  (collapse guard: < 0.2 required)

## PASS Criteria

| Criterion | Result | Value |
|---|---|---|
| C1: causal_sig_approach > 0.001 | {"PASS" if c1 else "FAIL"} | {causal_sig_approach:.6f} |
| C2: calibration_gap_approach > 0.05 | {"PASS" if c2 else "FAIL"} | {calibration_gap_approach:.4f} |
| C3: mean_harm_eval_none < 0.2 (no collapse) | {"PASS" if c3 else "FAIL"} | {mean_harm_none:.4f} |
| C4: causal_sig_contact > causal_sig_approach (MECH-102) | {"PASS" if c4 else "FAIL"} | {causal_sig_contact:.6f} vs {causal_sig_approach:.6f} |
| C5: n_approach_eval >= 30 | {"PASS" if c5 else "FAIL"} | {n_approach_eval} |

Criteria met: {n_met}/5 → **{status}**
{failure_section}
"""

    return {
        "status":             status,
        "metrics":            metrics,
        "summary_markdown":   summary_markdown,
        "claim_ids":          CLAIM_IDS,
        "evidence_direction": (
            "supports" if all_pass
            else ("mixed" if n_met >= 3 else "weakens")
        ),
        "experiment_type":    EXPERIMENT_TYPE,
        "fatal_error_count":  0,
    }


if __name__ == "__main__":
    import argparse
    import json
    from datetime import datetime, timezone

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",            type=int,   default=0)
    parser.add_argument("--phase1",          type=int,   default=400)
    parser.add_argument("--phase2",          type=int,   default=200)
    parser.add_argument("--eval-episodes",   type=int,   default=100)
    parser.add_argument("--steps",           type=int,   default=300)
    parser.add_argument("--alpha-world",     type=float, default=0.9)
    parser.add_argument("--alpha-self",      type=float, default=0.3)
    parser.add_argument("--harm-scale",      type=float, default=0.02)
    parser.add_argument("--proximity-scale", type=float, default=0.05)
    args = parser.parse_args()

    result = run(
        seed=args.seed,
        phase1_episodes=args.phase1,
        phase2_episodes=args.phase2,
        eval_episodes=args.eval_episodes,
        steps_per_episode=args.steps,
        alpha_world=args.alpha_world,
        alpha_self=args.alpha_self,
        harm_scale=args.harm_scale,
        proximity_scale=args.proximity_scale,
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
