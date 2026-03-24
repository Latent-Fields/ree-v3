"""
V3-EXQ-093 -- HarmBridge Validation with E3 Head Fix

Claims: SD-010, SD-003

Fix for EXQ-087 (FAIL: bridge_r2=0.0).

Root cause: in EXQ-087, agent.e3.harm_eval_z_harm_head was never trained during
Phase 1 or Phase 2. Phase 1 computed loss_he = MSE(harm_eval_z_harm(z_harm), label)
and called opt_harm.step(), but opt_harm only contained harm_enc.parameters() --
the head accumulated unzeroed gradients that were never applied. Phase 2 included
the head in its optimizer but HarmEncoder was already frozen with a random head,
so Phase 2 was training a head on top of a useless z_harm latent.

Fix: include agent.e3.harm_eval_z_harm_head.parameters() in opt_harm in Phase 1.
This trains the head jointly with HarmEncoder from the start, giving HarmEncoder
a meaningful gradient signal through the harm prediction loss.

Protocol identical to EXQ-087: three phases (terrain training, E3 calibration,
attribution eval). Discriminative pair matches EXQ-087 to allow direct comparison.

PASS criteria (ALL):
  C1: causal_sig_approach > 0.001
  C2: calibration_gap_approach > 0.05
  C3: mean_harm_none < mean_harm_approach * 0.80
  C4: causal_sig_contact > causal_sig_approach  (MECH-102 escalation)
  C5: harm_bridge_alignment_r2 > 0.5
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
from ree_core.latent.stack import HarmEncoder, HarmBridge
from ree_core.utils.config import REEConfig


EXPERIMENT_TYPE = "v3_exq_093_harm_bridge_e3_fix"
CLAIM_IDS = ["SD-010", "SD-003"]

HARM_OBS_DIM = 51
Z_HARM_DIM   = 32
STRAT_BUF_SIZE  = 2000
MIN_PER_BUCKET  = 4
SAMPLES_PER_BUCKET = 8


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

    harm_enc    = HarmEncoder(harm_obs_dim=HARM_OBS_DIM, z_harm_dim=Z_HARM_DIM)
    harm_bridge = HarmBridge(world_dim=world_dim, z_harm_dim=Z_HARM_DIM)

    num_actions = env.action_dim

    print(
        f"[V3-EXQ-093] HarmBridge validation fix (SD-010 / SD-003)\n"
        f"  Fix: E3 harm_eval head now trained jointly with HarmEncoder in Phase 1\n"
        f"  HarmBridge: z_world({world_dim}) -> z_harm({Z_HARM_DIM})\n"
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
    # FIX: include harm_eval_z_harm_head so it trains jointly with HarmEncoder
    opt_harm    = optim.Adam(
        list(harm_enc.parameters())
        + list(agent.e3.harm_eval_z_harm_head.parameters()),
        lr=1e-3,
    )
    opt_bridge  = optim.Adam(harm_bridge.parameters(), lr=1e-3)
    opt_e3_harm = optim.Adam(agent.e3.harm_eval_z_harm_head.parameters(), lr=1e-4)

    wf_data: List = []
    MAX_WF = 4000

    # -- Phase 1: terrain training -------------------------------------------
    print(f"\n[P1] Terrain training ({phase1_episodes} eps)...", flush=True)
    agent.train(); harm_enc.train(); harm_bridge.train()

    for ep in range(phase1_episodes):
        flat_obs, obs_dict = env.reset()
        agent.reset()
        z_world_prev = None; a_prev = None

        for _ in range(steps_per_episode):
            obs_body  = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            latent = agent.sense(obs_body, obs_world)
            agent.clock.advance()
            z_world_curr = latent.z_world.detach()

            action_idx = random.randint(0, num_actions - 1)
            action = _action_to_onehot(action_idx, num_actions, agent.device)
            agent._last_action = action

            flat_obs, _, done, info, obs_dict = env.step(action)

            harm_obs = obs_dict.get("harm_obs", torch.zeros(HARM_OBS_DIM)).float()

            # Train HarmEncoder + E3 harm head jointly on proximity label
            label  = harm_obs[12].unsqueeze(0).unsqueeze(0)
            z_harm = harm_enc(harm_obs.unsqueeze(0))
            pred   = agent.e3.harm_eval_z_harm(z_harm)
            loss_he = F.mse_loss(pred, label)
            opt_harm.zero_grad(); loss_he.backward()
            opt_harm.step()

            # Train HarmBridge: HarmBridge(z_world) ~= HarmEncoder(harm_obs)
            z_harm_target = harm_enc(harm_obs.unsqueeze(0)).detach()
            z_harm_pred   = harm_bridge(z_world_curr)
            loss_hb = F.mse_loss(z_harm_pred, z_harm_target)
            opt_bridge.zero_grad(); loss_hb.backward()
            opt_bridge.step()

            # World-forward data
            if z_world_prev is not None and a_prev is not None:
                wf_data.append((z_world_prev.cpu(), a_prev.cpu(), z_world_curr.cpu()))
                if len(wf_data) > MAX_WF:
                    wf_data = wf_data[-MAX_WF:]

            if len(wf_data) >= 16:
                k = min(32, len(wf_data))
                idxs = torch.randperm(len(wf_data))[:k].tolist()
                zw_b  = torch.cat([wf_data[i][0] for i in idxs]).to(agent.device)
                a_b   = torch.cat([wf_data[i][1] for i in idxs]).to(agent.device)
                zw1_b = torch.cat([wf_data[i][2] for i in idxs]).to(agent.device)
                wf_loss = F.mse_loss(agent.e2.world_forward(zw_b, a_b), zw1_b)
                if wf_loss.requires_grad:
                    opt_wf.zero_grad(); wf_loss.backward(); opt_wf.step()

            e1_loss = agent.compute_prediction_loss()
            e2_loss = agent.compute_e2_loss()
            total   = e1_loss + e2_loss
            if total.requires_grad:
                opt_std.zero_grad(); total.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                opt_std.step()

            z_world_prev = z_world_curr; a_prev = action.detach()
            if done: break

        if (ep + 1) % 100 == 0 or ep == phase1_episodes - 1:
            print(f"  [P1] ep {ep+1}/{phase1_episodes}", flush=True)

    # HarmBridge alignment R2
    bridge_r2 = 0.0
    if len(wf_data) >= 50:
        with torch.no_grad():
            flat_obs, obs_dict = env.reset()
            agent.reset()
            zw_list = []; zh_target_list = []
            for _ in range(min(200, steps_per_episode * 5)):
                latent = agent.sense(obs_dict["body_state"], obs_dict["world_state"])
                agent.clock.advance()
                zw_list.append(latent.z_world)
                harm_obs = obs_dict.get("harm_obs", torch.zeros(HARM_OBS_DIM)).float()
                zh_target_list.append(harm_enc(harm_obs.unsqueeze(0)))
                action = _action_to_onehot(random.randint(0, num_actions-1), num_actions, agent.device)
                agent._last_action = action
                flat_obs, _, done, _, obs_dict = env.step(action)
                if done: break
            if len(zw_list) >= 20:
                zw_t  = torch.cat(zw_list, dim=0)
                zh_t  = torch.cat(zh_target_list, dim=0)
                pred  = harm_bridge(zw_t)
                ss_res = ((zh_t - pred)**2).sum()
                ss_tot = ((zh_t - zh_t.mean(0, keepdim=True))**2).sum()
                bridge_r2 = float((1 - ss_res / (ss_tot + 1e-8)).item())
    print(f"  HarmBridge alignment R2: {bridge_r2:.4f}", flush=True)

    # -- Phase 2: E3 calibration (stratified) --------------------------------
    print(f"\n[P2] E3 calibration ({phase2_episodes} eps)...", flush=True)
    for p in agent.parameters(): p.requires_grad_(False)
    for p in harm_enc.parameters(): p.requires_grad_(False)
    for p in harm_bridge.parameters(): p.requires_grad_(False)
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
            action = _action_to_onehot(random.randint(0, num_actions-1), num_actions, agent.device)
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
    agent.eval(); harm_enc.eval(); harm_bridge.eval()

    causal_by: Dict[str, List[float]] = {}
    harm_by:   Dict[str, List[float]] = {}

    for ep in range(eval_episodes):
        flat_obs, obs_dict = env.reset()
        agent.reset()
        for _ in range(steps_per_episode):
            with torch.no_grad():
                latent = agent.sense(obs_dict["body_state"], obs_dict["world_state"])
                agent.clock.advance()
                z_world   = latent.z_world
                harm_obs  = obs_dict.get("harm_obs", torch.zeros(HARM_OBS_DIM)).float()
                z_harm_actual = harm_enc(harm_obs.unsqueeze(0))
                harm_actual   = float(agent.e3.harm_eval_z_harm(z_harm_actual).item())

                cf_vals = []
                for ci in range(num_actions):
                    a_cf      = _action_to_onehot(ci, num_actions, agent.device)
                    z_w_cf    = agent.e2.world_forward(z_world, a_cf)
                    z_harm_cf = harm_bridge(z_w_cf)
                    cf_vals.append(float(agent.e3.harm_eval_z_harm(z_harm_cf).item()))

                causal_sig = harm_actual - float(np.mean(cf_vals))

            action = _action_to_onehot(random.randint(0, num_actions-1), num_actions, agent.device)
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

    print(f"\n  --- EXQ-093 results ---", flush=True)
    print(f"  bridge_r2:          {bridge_r2:.4f}", flush=True)
    print(f"  causal_none:        {causal_none:.6f}", flush=True)
    print(f"  causal_approach:    {causal_approach:.6f}", flush=True)
    print(f"  causal_contact:     {causal_contact:.6f}", flush=True)
    print(f"  calibration_gap:    {cal_gap:.4f}", flush=True)
    print(f"  harm_none:          {harm_none:.4f}", flush=True)
    print(f"  harm_approach:      {harm_approach:.4f}", flush=True)
    print(f"  n_approach_eval:    {n_approach}", flush=True)

    c1 = causal_approach > 0.001
    c2 = cal_gap         > 0.05
    c3 = (harm_approach  > 0) and (harm_none < harm_approach * 0.80)
    c4 = causal_contact  > causal_approach
    c5 = bridge_r2       > 0.5

    all_pass = c1 and c2 and c3 and c4 and c5
    status   = "PASS" if all_pass else "FAIL"
    n_met    = sum([c1, c2, c3, c4, c5])

    failure_notes = []
    if not c1: failure_notes.append(
        f"C1 FAIL: causal_approach={causal_approach:.6f} <= 0.001")
    if not c2: failure_notes.append(
        f"C2 FAIL: cal_gap={cal_gap:.4f} <= 0.05")
    if not c3: failure_notes.append(
        f"C3 FAIL: harm_none={harm_none:.4f} not < harm_approach*0.80={harm_approach*0.80:.4f}")
    if not c4: failure_notes.append(
        f"C4 FAIL: causal_contact={causal_contact:.6f} <= causal_approach={causal_approach:.6f}")
    if not c5: failure_notes.append(
        f"C5 FAIL: bridge_r2={bridge_r2:.4f} <= 0.5")

    print(f"\nV3-EXQ-093 verdict: {status}  ({n_met}/5)", flush=True)
    for note in failure_notes:
        print(f"  {note}", flush=True)

    metrics = {
        "bridge_r2":           float(bridge_r2),
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

    summary_markdown = f"""# V3-EXQ-093 -- HarmBridge Counterfactual Validation (E3 Head Fix)

**Status:** {status}
**Claims:** SD-010, SD-003
**Fixes:** EXQ-087 (E3 harm_eval head never trained)
**World:** CausalGridWorldV2 (6 hazards, 3 resources, 12x12)
**Protocol:** Three-phase (terrain {phase1_episodes} -> calib {phase2_episodes} -> eval {eval_episodes})

## Root Cause Fixed (EXQ-087)

In EXQ-087, agent.e3.harm_eval_z_harm_head was excluded from opt_harm, so
loss_he = MSE(harm_eval_z_harm(z_harm), label) computed gradients through the head
but never applied them. Phase 2 then re-trained the head from scratch on top of a
HarmEncoder that had optimised toward a random head output.

Fix: opt_harm now includes both harm_enc.parameters() and
agent.e3.harm_eval_z_harm_head.parameters(). The head and encoder train jointly
from the start of Phase 1.

## Results

| Metric | Value |
|--------|-------|
| HarmBridge alignment R2 | {bridge_r2:.4f} |
| causal_sig_none | {causal_none:.6f} |
| causal_sig_approach | {causal_approach:.6f} |
| causal_sig_contact | {causal_contact:.6f} |
| calibration_gap_approach | {cal_gap:.4f} |

## PASS Criteria

| Criterion | Result | Value |
|-----------|--------|-------|
| C1: causal_approach > 0.001 | {"PASS" if c1 else "FAIL"} | {causal_approach:.6f} |
| C2: cal_gap > 0.05 | {"PASS" if c2 else "FAIL"} | {cal_gap:.4f} |
| C3: harm_none < harm_approach * 0.80 | {"PASS" if c3 else "FAIL"} | {harm_none:.4f} < {harm_approach*0.80:.4f} |
| C4: causal_contact > causal_approach | {"PASS" if c4 else "FAIL"} | {causal_contact:.6f} vs {causal_approach:.6f} |
| C5: bridge_r2 > 0.5 | {"PASS" if c5 else "FAIL"} | {bridge_r2:.4f} |

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
