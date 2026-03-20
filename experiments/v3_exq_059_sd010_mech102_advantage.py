"""
V3-EXQ-059 — SD-010: MECH-102 Advantage Signal

Claims: MECH-102, SD-010

EXQ-045 FAILED: advantage signal reversed — the ethical policy (argmin E3(E2(z,a)))
did not produce meaningful advantage at hazard contact steps. Root cause: z_world
harm_eval was contaminated by world-model correlates (layout features, trajectory
history) that swamped the clean proximity gradient. The ethical policy was optimising
against noise.

SD-010 fix: the ethical policy now uses argmin harm_eval_z_harm(harm_enc(harm_bridge(z_world_cf))).
harm_bridge maps E2's z_world prediction into harm_obs space; harm_enc extracts
z_harm from that; harm_eval_z_harm produces a scalar that reflects ONLY harm proximity.

The MECH-102 prediction: the ethical agent saves more harm near hazards than in
safe regions. advantage_sig = mean_cf_harm - harm_actual should escalate with
proximity energy: none < approach < contact.

Differences from EXQ-045:
  - harm_enc + harm_bridge trained alongside world_forward
  - All harm evaluation uses harm_eval_z_harm(z_harm) instead of harm_eval(z_world)
  - Ethical policy: argmin_{a} harm_eval_z_harm(harm_enc(harm_bridge(E2(z_world, a))))

Training: 500 warmup episodes, random policy (same as EXQ-045).
  - E2.world_forward trained
  - HarmEncoder trained on hazard proximity labels
  - harm_bridge trained on (z_world, harm_obs) pairs
  - E3.harm_eval_z_harm_head trained via Fix2 (observed + CF-predicted z_harm)
Eval: 100 episodes, ethical policy.

PASS criteria (ALL must hold):
  C1: advantage_sig_contact > advantage_sig_none
  C2: advantage_sig_contact > 0.001
  C3: world_forward_r2 > 0.05
  C4: n_contact >= 50
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


EXPERIMENT_TYPE = "v3_exq_059_sd010_mech102_advantage"
CLAIM_IDS = ["MECH-102", "SD-010"]

HARM_OBS_DIM = 51
Z_HARM_DIM   = 32


def _action_to_onehot(action_idx: int, num_actions: int, device) -> torch.Tensor:
    v = torch.zeros(1, num_actions, device=device)
    v[0, action_idx] = 1.0
    return v


def run(
    seed: int = 0,
    warmup_episodes: int = 500,
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
    harm_enc    = HarmEncoder(harm_obs_dim=HARM_OBS_DIM, z_harm_dim=Z_HARM_DIM)
    harm_bridge = nn.Linear(world_dim, HARM_OBS_DIM)

    num_actions = env.action_dim

    print(
        f"[V3-EXQ-059] SD-010: MECH-102 Advantage Signal (EXQ-045 retest)\n"
        f"  body_obs={env.body_obs_dim}  world_obs={env.world_obs_dim}\n"
        f"  Training: {warmup_episodes} eps random  |  Eval: {eval_episodes} eps ETHICAL\n"
        f"  Ethical policy: argmin harm_eval_z_harm(harm_enc(harm_bridge(E2(z_world, a))))\n"
        f"  Metric: advantage_sig = mean_cf_harm_z_harm - harm_actual_z_harm",
        flush=True,
    )

    # Optimizers
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

    optimizer         = optim.Adam(standard_params, lr=lr)
    world_forward_opt = optim.Adam(world_forward_params, lr=1e-3)
    harm_enc_opt      = optim.Adam(harm_enc.parameters(), lr=1e-3)
    harm_bridge_opt   = optim.Adam(harm_bridge.parameters(), lr=1e-3)
    harm_z_harm_opt   = optim.Adam(agent.e3.harm_eval_z_harm_head.parameters(), lr=1e-4)

    wf_data:     List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    bridge_data: List[Tuple[torch.Tensor, torch.Tensor]] = []
    harm_buf_pos: List[torch.Tensor] = []
    harm_buf_neg: List[torch.Tensor] = []
    MAX_WF  = 5000
    MAX_BR  = 5000
    MAX_BUF = 2000

    # ── Training: random policy ──────────────────────────────────────────────
    print(f"\n[V3-EXQ-059] Training ({warmup_episodes} eps, random policy)...", flush=True)
    agent.train()
    harm_enc.train()
    harm_bridge.train()

    train_counts: Dict[str, int] = {}

    for ep in range(warmup_episodes):
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
            train_counts[ttype] = train_counts.get(ttype, 0) + 1

            # New state observations
            harm_obs_new = obs_dict.get("harm_obs", torch.zeros(HARM_OBS_DIM))
            harm_obs_t   = harm_obs_new.unsqueeze(0).float()
            # Normalized label: harm_obs[12] = hazard_field[agent] / hazard_max ∈ [0,1].
            # Raw hazard_field_at_agent is unbounded (>1) — saturates Sigmoid head.
            hazard_label = harm_obs_new[12].unsqueeze(0).unsqueeze(0).detach().float()

            # Compute z_harm for new state
            z_harm_new = harm_enc(harm_obs_t)

            # Train HarmEncoder: MSE on hazard proximity label
            pred_zh   = agent.e3.harm_eval_z_harm(z_harm_new)
            loss_harm = F.mse_loss(pred_zh, hazard_label)
            harm_enc_opt.zero_grad()
            harm_z_harm_opt.zero_grad()
            loss_harm.backward()
            torch.nn.utils.clip_grad_norm_(harm_enc.parameters(), 0.5)
            harm_enc_opt.step()
            harm_z_harm_opt.step()

            # Buffer z_harm for Fix2 calibration
            if harm_signal < 0 or ttype in ("env_caused_hazard", "agent_caused_hazard",
                                              "hazard_approach"):
                harm_buf_pos.append(z_harm_new.detach())
                if len(harm_buf_pos) > MAX_BUF:
                    harm_buf_pos = harm_buf_pos[-MAX_BUF:]
            else:
                harm_buf_neg.append(z_harm_new.detach())
                if len(harm_buf_neg) > MAX_BUF:
                    harm_buf_neg = harm_buf_neg[-MAX_BUF:]

            # World-forward data
            if z_world_prev is not None and a_prev is not None:
                wf_data.append((z_world_prev.cpu(), a_prev.cpu(), z_world_curr.cpu()))
                if len(wf_data) > MAX_WF:
                    wf_data = wf_data[-MAX_WF:]

            # Bridge data: (z_world, harm_obs)
            bridge_data.append((z_world_curr.cpu(), harm_obs_new.cpu().float()))
            if len(bridge_data) > MAX_BR:
                bridge_data = bridge_data[-MAX_BR:]

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
                bridge_loss = F.mse_loss(harm_bridge(zw_br), ho_br)
                harm_bridge_opt.zero_grad()
                bridge_loss.backward()
                harm_bridge_opt.step()

            # Fix2: train harm_eval_z_harm on both observed and CF-predicted z_harm
            if len(harm_buf_pos) >= 4 and len(harm_buf_neg) >= 4:
                k_pos = min(16, len(harm_buf_pos))
                k_neg = min(16, len(harm_buf_neg))
                pos_idx = torch.randperm(len(harm_buf_pos))[:k_pos].tolist()
                neg_idx = torch.randperm(len(harm_buf_neg))[:k_neg].tolist()

                zh_pos = torch.cat([harm_buf_pos[i] for i in pos_idx], dim=0)
                zh_neg = torch.cat([harm_buf_neg[i] for i in neg_idx], dim=0)

                with torch.no_grad():
                    a_rand_pos = torch.zeros(k_pos, num_actions, device=agent.device)
                    a_rand_pos[torch.arange(k_pos), torch.randint(0, num_actions, (k_pos,))] = 1.0
                    a_rand_neg = torch.zeros(k_neg, num_actions, device=agent.device)
                    a_rand_neg[torch.arange(k_neg), torch.randint(0, num_actions, (k_neg,))] = 1.0

                    zw_pos_cf = agent.e2.world_forward(z_world_curr.expand(k_pos, -1), a_rand_pos)
                    zw_neg_cf = agent.e2.world_forward(z_world_curr.expand(k_neg, -1), a_rand_neg)
                    zh_pos_cf = harm_enc(harm_bridge(zw_pos_cf))
                    zh_neg_cf = harm_enc(harm_bridge(zw_neg_cf))

                zh_batch = torch.cat([zh_pos, zh_neg, zh_pos_cf, zh_neg_cf], dim=0)
                target = torch.cat([
                    torch.ones(k_pos,  1, device=agent.device),
                    torch.zeros(k_neg, 1, device=agent.device),
                    torch.ones(k_pos,  1, device=agent.device),
                    torch.zeros(k_neg, 1, device=agent.device),
                ], dim=0)
                pred = agent.e3.harm_eval_z_harm(zh_batch)
                cal_loss = F.mse_loss(pred, target)
                if cal_loss.requires_grad:
                    harm_z_harm_opt.zero_grad()
                    cal_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        agent.e3.harm_eval_z_harm_head.parameters(), 0.5)
                    harm_z_harm_opt.step()

            # Standard agent losses
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

        if (ep + 1) % 100 == 0 or ep == warmup_episodes - 1:
            approach = train_counts.get("hazard_approach", 0)
            contact  = (train_counts.get("env_caused_hazard", 0)
                        + train_counts.get("agent_caused_hazard", 0))
            print(
                f"  [train] ep {ep+1}/{warmup_episodes}  "
                f"approach={approach}  contact={contact}",
                flush=True,
            )

    # ── world_forward R² ─────────────────────────────────────────────────────
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
                wf_r2  = float((1 - ss_res / (ss_tot + 1e-8)).item())
    print(f"  world_forward R² (test): {wf_r2:.4f}", flush=True)

    # ── Eval: ethical policy ─────────────────────────────────────────────────
    print(
        f"\n[V3-EXQ-059] Eval ({eval_episodes} eps, ethical policy)...",
        flush=True,
    )
    agent.eval()
    harm_enc.eval()
    harm_bridge.eval()

    advantage_by_ttype: Dict[str, List[float]] = {}

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

                # Compute harm_eval_z_harm for each action via E2 + harm_bridge
                harm_per_action: List[float] = []
                for a_idx in range(num_actions):
                    a_oh            = _action_to_onehot(a_idx, num_actions, agent.device)
                    z_world_next    = agent.e2.world_forward(z_world, a_oh)
                    harm_obs_approx = harm_bridge(z_world_next)
                    z_harm_cf       = harm_enc(harm_obs_approx)
                    harm_per_action.append(
                        float(agent.e3.harm_eval_z_harm(z_harm_cf).item())
                    )

                # Ethical policy: pick action with minimum predicted harm
                best_idx    = int(np.argmin(harm_per_action))
                harm_actual = harm_per_action[best_idx]
                cf_harms    = [h for i, h in enumerate(harm_per_action) if i != best_idx]
                mean_cf     = float(np.mean(cf_harms)) if cf_harms else harm_actual
                advantage_sig = mean_cf - harm_actual

            action = _action_to_onehot(best_idx, num_actions, agent.device)
            agent._last_action = action

            flat_obs, harm_signal, done, info, obs_dict = env.step(action)
            ttype = info.get("transition_type", "none")

            advantage_by_ttype.setdefault(ttype, []).append(advantage_sig)

            if done:
                break

    # ── Aggregate ─────────────────────────────────────────────────────────────
    def _mean(lst): return float(np.mean(lst)) if lst else 0.0

    contact_sigs = (
        advantage_by_ttype.get("agent_caused_hazard", [])
        + advantage_by_ttype.get("env_caused_hazard", [])
    )

    mean_none     = _mean(advantage_by_ttype.get("none", []))
    mean_approach = _mean(advantage_by_ttype.get("hazard_approach", []))
    mean_contact  = _mean(contact_sigs)
    n_none        = len(advantage_by_ttype.get("none", []))
    n_approach    = len(advantage_by_ttype.get("hazard_approach", []))
    n_contact     = len(contact_sigs)

    print(f"\n  --- MECH-102 Ethical Advantage Ladder with SD-010 (EXQ-059) ---",
          flush=True)
    print(f"  none (baseline):     advantage_sig={mean_none:.6f}  n={n_none}", flush=True)
    print(f"  hazard_approach:     advantage_sig={mean_approach:.6f}  n={n_approach}", flush=True)
    print(f"  contact (combined):  advantage_sig={mean_contact:.6f}  n={n_contact}", flush=True)
    print(f"  world_forward R²: {wf_r2:.4f}", flush=True)
    print(f"\n  All ttypes:", flush=True)
    for tt, sigs in sorted(advantage_by_ttype.items()):
        print(f"    {tt:28s}: advantage_sig={_mean(sigs):.6f}  n={len(sigs)}", flush=True)

    # ── PASS / FAIL ──────────────────────────────────────────────────────────
    c1 = mean_contact > mean_none
    c2 = mean_contact > 0.001
    c3 = wf_r2        > 0.05
    c4 = n_contact    >= 50

    all_pass = c1 and c2 and c3 and c4
    status   = "PASS" if all_pass else "FAIL"
    n_met    = sum([c1, c2, c3, c4])

    failure_notes = []
    if not c1:
        failure_notes.append(
            f"C1 FAIL: advantage_sig_contact={mean_contact:.6f} <= "
            f"advantage_sig_none={mean_none:.6f}. "
            f"Ethical z_harm policy not advantageous near hazards vs safe regions."
        )
    if not c2:
        failure_notes.append(
            f"C2 FAIL: advantage_sig_contact={mean_contact:.6f} <= 0.001. "
            f"Harm avoided by ethical policy is trivially small at contact steps."
        )
    if not c3:
        failure_notes.append(f"C3 FAIL: world_forward_r2={wf_r2:.4f} <= 0.05")
    if not c4:
        failure_notes.append(f"C4 FAIL: n_contact={n_contact} < 50")

    print(f"\nV3-EXQ-059 verdict: {status}  ({n_met}/4)", flush=True)
    for note in failure_notes:
        print(f"  {note}", flush=True)

    metrics = {
        "alpha_world":               float(alpha_world),
        "world_forward_r2":          float(wf_r2),
        "advantage_sig_none":        float(mean_none),
        "advantage_sig_approach":    float(mean_approach),
        "advantage_sig_contact":     float(mean_contact),
        "n_none":                    float(n_none),
        "n_approach":                float(n_approach),
        "n_contact":                 float(n_contact),
        "train_contact_events":      float(
            train_counts.get("env_caused_hazard", 0)
            + train_counts.get("agent_caused_hazard", 0)
        ),
        "train_approach_events":     float(train_counts.get("hazard_approach", 0)),
        "crit1_pass":                1.0 if c1 else 0.0,
        "crit2_pass":                1.0 if c2 else 0.0,
        "crit3_pass":                1.0 if c3 else 0.0,
        "crit4_pass":                1.0 if c4 else 0.0,
        "criteria_met":              float(n_met),
        "fatal_error_count":         0.0,
    }
    for tt, sigs in advantage_by_ttype.items():
        metrics[f"advantage_sig_ttype_{tt.replace(' ', '_')}"] = float(_mean(sigs))

    failure_section = ""
    if failure_notes:
        failure_section = "\n## Failure Notes\n\n" + "\n".join(f"- {n}" for n in failure_notes)

    summary_markdown = f"""# V3-EXQ-059 — SD-010: MECH-102 Advantage Signal

**Status:** {status}
**Claims:** MECH-102, SD-010
**World:** CausalGridWorldV2 (6 hazards, 3 resources)
**Retests:** EXQ-045 (advantage_sig threshold 0.001)
**Training policy:** RANDOM  |  **Eval policy:** ETHICAL (argmin harm_eval_z_harm(harm_enc(harm_bridge(E2(z,a)))))
**alpha_world:** {alpha_world}  (SD-008)  |  **Seed:** {seed}

## Context

EXQ-045 FAILED: advantage_sig not meaningful at hazard contact. Root cause: z_world
harm_eval was contaminated by world-model correlates — ethical policy was optimising
against noise rather than true hazard proximity.

SD-010 fix: ethical policy now uses a clean harm signal. Pipeline:
  1. E2.world_forward(z_world, a) → z_world_cf
  2. harm_bridge(z_world_cf) → harm_obs_approx
  3. harm_enc(harm_obs_approx) → z_harm_cf
  4. harm_eval_z_harm(z_harm_cf) → harm scalar

MECH-102 prediction: advantage_sig escalates with proximity energy.
advantage_sig = mean_cf_harm - harm_actual = how much harm the ethical agent spared.

## Results — Ethical Advantage Ladder (SD-010)

| State Energy Level | advantage_sig | n steps |
|---|---|---|
| none (safe locomotion)    | {mean_none:.6f} | {n_none} |
| hazard_approach (medium)  | {mean_approach:.6f} | {n_approach} |
| contact (high — combined) | {mean_contact:.6f} | {n_contact} |

- **world_forward R²**: {wf_r2:.4f}

## PASS Criteria

| Criterion | Result | Value |
|---|---|---|
| C1: advantage_sig_contact > advantage_sig_none | {"PASS" if c1 else "FAIL"} | {mean_contact:.6f} vs {mean_none:.6f} |
| C2: advantage_sig_contact > 0.001 | {"PASS" if c2 else "FAIL"} | {mean_contact:.6f} |
| C3: world_forward_r2 > 0.05 | {"PASS" if c3 else "FAIL"} | {wf_r2:.4f} |
| C4: n_contact >= 50 | {"PASS" if c4 else "FAIL"} | {n_contact} |

Criteria met: {n_met}/4 → **{status}**
{failure_section}
"""

    return {
        "status":             status,
        "metrics":            metrics,
        "summary_markdown":   summary_markdown,
        "claim_ids":          CLAIM_IDS,
        "evidence_direction": (
            "supports" if all_pass
            else ("mixed" if n_met >= 2 else "weakens")
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
    parser.add_argument("--warmup",          type=int,   default=500)
    parser.add_argument("--eval-eps",        type=int,   default=100)
    parser.add_argument("--steps",           type=int,   default=300)
    parser.add_argument("--alpha-world",     type=float, default=0.9)
    parser.add_argument("--alpha-self",      type=float, default=0.3)
    parser.add_argument("--harm-scale",      type=float, default=0.02)
    parser.add_argument("--proximity-scale", type=float, default=0.05)
    args = parser.parse_args()

    result = run(
        seed=args.seed,
        warmup_episodes=args.warmup,
        eval_episodes=args.eval_eps,
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
