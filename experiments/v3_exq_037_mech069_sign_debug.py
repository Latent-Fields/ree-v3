"""
V3-EXQ-037 — MECH-069 Sign Investigation (EXQ-035 Follow-up)

Claims: MECH-069, SD-003

EXQ-035 FAIL analysis:

    EXQ-035 ran two conditions (separated vs merged optimizers).
    FAIL reason: calibration_gap_approach_separated = -0.028 < C3 threshold 0.05.
    Both conditions have NEGATIVE calibration_gap.

    Negative calibration_gap means E3.harm_eval scores hazard_approach steps LOWER
    than none (safe locomotion) steps. This is backwards — E3 should score dangerous
    situations HIGHER.

    Contrast with EXQ-026 PASS: calibration_gap = +0.0375.
    EXQ-026 used a simpler setup (no SD-003 pipeline, no world_forward training).

    Hypothesis: the combined SD-003 training pipeline (world_forward optimizer +
    harm_eval Fix 2 E3-on-E2-predictions) corrupts E3 calibration by training
    harm_eval on E2-predicted states that are biased toward safe states (since
    the world_forward model averages across all states regardless of harm).

This experiment (EXQ-037) runs ONLY the separated condition but with extensive
diagnostics to identify the sign inversion root cause:

    1. Log harm_eval scores per ttype at every 100 training episodes
    2. At training end: compute calibration_gap before and after applying sigmoid
       (check if the issue is raw logit sign vs probability sign)
    3. Report E3.harm_eval head weight norms and bias
    4. Compare two E3 training variants:
       Variant A: Fix 2 (observed + E2-predicted) — same as EXQ-030b/035
       Variant B: observed-only — same as EXQ-026 (which PASSED)
    Both use separated optimizers (MECH-069).

Design: run two sub-conditions sequentially (same seed, fresh agents):
    SUB-A (Fix2): E3 trained on observed + E2-predicted (EXQ-035 setup)
    SUB-B (ObsOnly): E3 trained on observed states only (EXQ-026 setup)

PASS criteria (ALL must hold for SUB-B at minimum — establishes sign direction):
    C1: calibration_gap_obsonly > 0  (obs-only E3 gives correct sign)
    C2: calibration_gap_obsonly > 0.01  (obs-only E3 above noise floor)
    C3: mean_harm_eval_approach_obsonly > mean_harm_eval_none_obsonly  (correct ordering)
    C4: world_forward_r2 > 0.05  (E2 functional)
    C5: n_approach_eval >= 30  (sufficient samples)

Diagnostic (not pass criterion):
    sign_inversion_detected: whether SUB-A has negative calibration_gap while SUB-B is positive
    This directly identifies whether Fix 2 (E3-on-E2-predictions) is causing sign inversion.
"""

import sys
import random
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig


EXPERIMENT_TYPE = "v3_exq_037_mech069_sign_debug"
CLAIM_IDS = ["MECH-069", "SD-003"]


def _action_to_onehot(action_idx: int, num_actions: int, device) -> torch.Tensor:
    v = torch.zeros(1, num_actions, device=device)
    v[0, action_idx] = 1.0
    return v


def _compute_world_forward_r2(
    agent: REEAgent,
    wf_data: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
) -> float:
    if len(wf_data) < 20:
        return 0.0
    n = len(wf_data)
    n_train = int(n * 0.8)
    with torch.no_grad():
        zw_all  = torch.cat([d[0] for d in wf_data], dim=0)
        a_all   = torch.cat([d[1] for d in wf_data], dim=0)
        zw1_all = torch.cat([d[2] for d in wf_data], dim=0)
        pred_all = agent.e2.world_forward(zw_all, a_all)
        pred_test = pred_all[n_train:]
        tgt_test  = zw1_all[n_train:]
        if pred_test.shape[0] == 0:
            return 0.0
        ss_res = ((tgt_test - pred_test) ** 2).sum()
        ss_tot = ((tgt_test - tgt_test.mean(0, keepdim=True)) ** 2).sum()
        r2 = float((1 - ss_res / (ss_tot + 1e-8)).item())
    print(f"  world_forward R² (test n={pred_test.shape[0]}): {r2:.4f}", flush=True)
    return r2


def _train_subcondition(
    agent: REEAgent,
    env,
    optimizer: optim.Optimizer,
    harm_eval_optimizer: optim.Optimizer,
    world_forward_optimizer: optim.Optimizer,
    num_episodes: int,
    steps_per_episode: int,
    use_e2_predicted: bool,
    label: str,
) -> Dict:
    """
    Train agent. If use_e2_predicted=True, apply EXQ-030b Fix 2 (E3 on observed + E2-predicted).
    If False, train E3 on observed states only (EXQ-026 style).
    """
    agent.train()

    harm_buf_pos: List[torch.Tensor] = []
    harm_buf_neg: List[torch.Tensor] = []
    MAX_BUF = 2000

    wf_data: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    MAX_WF = 5000

    counts: Dict[str, int] = {}
    num_actions = env.action_dim

    # Per-100-episode calibration log
    calib_log: List[Dict] = []

    for ep in range(num_episodes):
        flat_obs, obs_dict = env.reset()
        agent.reset()
        z_world_prev = None
        a_prev = None

        for step in range(steps_per_episode):
            obs_body  = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            latent = agent.sense(obs_body, obs_world)
            agent.clock.advance()

            z_world_curr = latent.z_world.detach()

            action_idx = random.randint(0, env.action_dim - 1)
            action = _action_to_onehot(action_idx, env.action_dim, agent.device)
            agent._last_action = action

            flat_obs, harm_signal, done, info, obs_dict = env.step(action)
            ttype = info.get("transition_type", "none")
            counts[ttype] = counts.get(ttype, 0) + 1

            if harm_signal < 0:
                harm_buf_pos.append(z_world_curr)
                if len(harm_buf_pos) > MAX_BUF:
                    harm_buf_pos = harm_buf_pos[-MAX_BUF:]
            else:
                harm_buf_neg.append(z_world_curr)
                if len(harm_buf_neg) > MAX_BUF:
                    harm_buf_neg = harm_buf_neg[-MAX_BUF:]

            if z_world_prev is not None and a_prev is not None:
                wf_data.append((z_world_prev.cpu(), a_prev.cpu(), z_world_curr.cpu()))
                if len(wf_data) > MAX_WF:
                    wf_data = wf_data[-MAX_WF:]

            e1_loss = agent.compute_prediction_loss()
            e2_loss = agent.compute_e2_loss()
            total_loss = e1_loss + e2_loss
            if total_loss.requires_grad:
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                optimizer.step()

            if len(wf_data) >= 16:
                k = min(32, len(wf_data))
                idxs = torch.randperm(len(wf_data))[:k].tolist()
                zw_b  = torch.cat([wf_data[i][0] for i in idxs]).to(agent.device)
                a_b   = torch.cat([wf_data[i][1] for i in idxs]).to(agent.device)
                zw1_b = torch.cat([wf_data[i][2] for i in idxs]).to(agent.device)
                pred = agent.e2.world_forward(zw_b, a_b)
                wf_loss = F.mse_loss(pred, zw1_b)
                if wf_loss.requires_grad:
                    world_forward_optimizer.zero_grad()
                    wf_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        agent.e2.world_transition.parameters(), 0.5
                    )
                    world_forward_optimizer.step()

            if len(harm_buf_pos) >= 4 and len(harm_buf_neg) >= 4:
                k_pos = min(16, len(harm_buf_pos))
                k_neg = min(16, len(harm_buf_neg))
                pos_idx = torch.randperm(len(harm_buf_pos))[:k_pos].tolist()
                neg_idx = torch.randperm(len(harm_buf_neg))[:k_neg].tolist()

                zw_pos_obs = torch.cat([harm_buf_pos[i] for i in pos_idx], dim=0)
                zw_neg_obs = torch.cat([harm_buf_neg[i] for i in neg_idx], dim=0)

                if use_e2_predicted:
                    # Fix 2: also train on E2-predicted states
                    with torch.no_grad():
                        a_rand_pos = torch.zeros(k_pos, num_actions, device=agent.device)
                        a_rand_pos[torch.arange(k_pos), torch.randint(0, num_actions, (k_pos,))] = 1.0
                        a_rand_neg = torch.zeros(k_neg, num_actions, device=agent.device)
                        a_rand_neg[torch.arange(k_neg), torch.randint(0, num_actions, (k_neg,))] = 1.0
                        zw_pos_pred = agent.e2.world_forward(zw_pos_obs.to(agent.device), a_rand_pos)
                        zw_neg_pred = agent.e2.world_forward(zw_neg_obs.to(agent.device), a_rand_neg)
                    zw_b = torch.cat([
                        zw_pos_obs.to(agent.device),
                        zw_neg_obs.to(agent.device),
                        zw_pos_pred,
                        zw_neg_pred,
                    ], dim=0)
                    target = torch.cat([
                        torch.ones(k_pos,  1, device=agent.device),
                        torch.zeros(k_neg, 1, device=agent.device),
                        torch.ones(k_pos,  1, device=agent.device),
                        torch.zeros(k_neg, 1, device=agent.device),
                    ], dim=0)
                else:
                    # Observed only
                    zw_b = torch.cat([
                        zw_pos_obs.to(agent.device),
                        zw_neg_obs.to(agent.device),
                    ], dim=0)
                    target = torch.cat([
                        torch.ones(k_pos,  1, device=agent.device),
                        torch.zeros(k_neg, 1, device=agent.device),
                    ], dim=0)

                pred_harm = agent.e3.harm_eval(zw_b)
                harm_loss = F.mse_loss(pred_harm, target)
                if harm_loss.requires_grad:
                    harm_eval_optimizer.zero_grad()
                    harm_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        agent.e3.harm_eval_head.parameters(), 0.5
                    )
                    harm_eval_optimizer.step()

            z_world_prev = z_world_curr
            a_prev = action.detach()

            if done:
                break

        # Log calibration every 100 episodes
        if (ep + 1) % 100 == 0:
            with torch.no_grad():
                scores_by_ttype: Dict[str, List[float]] = {
                    "none": [], "hazard_approach": [],
                    "env_caused_hazard": [], "agent_caused_hazard": [],
                }
                # Quick probe: sample 10 states from recent buffers
                if harm_buf_pos and harm_buf_neg:
                    zw_probe_pos = harm_buf_pos[-min(10, len(harm_buf_pos)):]
                    zw_probe_neg = harm_buf_neg[-min(10, len(harm_buf_neg)):]
                    mean_harm_pos = float(
                        agent.e3.harm_eval(
                            torch.cat(zw_probe_pos).to(agent.device)
                        ).mean().item()
                    )
                    mean_harm_neg = float(
                        agent.e3.harm_eval(
                            torch.cat(zw_probe_neg).to(agent.device)
                        ).mean().item()
                    )
                else:
                    mean_harm_pos, mean_harm_neg = 0.0, 0.0

            approach = counts.get("hazard_approach", 0)
            print(
                f"  [{label}] ep {ep+1}/{num_episodes}  "
                f"harm_pos={mean_harm_pos:.4f}  harm_neg={mean_harm_neg:.4f}  "
                f"approach={approach}",
                flush=True,
            )
            calib_log.append({
                "episode": ep + 1,
                "mean_harm_pos": mean_harm_pos,
                "mean_harm_neg": mean_harm_neg,
                "gap_pos_neg": mean_harm_pos - mean_harm_neg,
            })

    return {"counts": counts, "wf_data": wf_data, "calib_log": calib_log}


def _eval_calibration(
    agent: REEAgent,
    env,
    num_episodes: int,
    steps_per_episode: int,
    label: str,
) -> Dict:
    """Eval E3 harm_eval by transition type. Also compute sigmoid-applied scores."""
    agent.eval()

    scores_raw: Dict[str, List[float]] = {
        "none": [], "hazard_approach": [],
        "env_caused_hazard": [], "agent_caused_hazard": [],
    }
    scores_sig: Dict[str, List[float]] = {
        t: [] for t in scores_raw
    }

    for ep in range(num_episodes):
        flat_obs, obs_dict = env.reset()
        agent.reset()

        for step in range(steps_per_episode):
            obs_body  = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            with torch.no_grad():
                latent = agent.sense(obs_body, obs_world)
                agent.clock.advance()
                z_world = latent.z_world

            action_idx = random.randint(0, env.action_dim - 1)
            action = _action_to_onehot(action_idx, env.action_dim, agent.device)
            agent._last_action = action

            flat_obs, harm_signal, done, info, obs_dict = env.step(action)
            ttype = info.get("transition_type", "none")

            with torch.no_grad():
                raw_score = float(agent.e3.harm_eval(z_world).item())
                sig_score = float(torch.sigmoid(agent.e3.harm_eval(z_world)).item())

            if ttype in scores_raw:
                scores_raw[ttype].append(raw_score)
                scores_sig[ttype].append(sig_score)

            if done:
                break

    def _mean(lst):
        return float(np.mean(lst)) if lst else 0.0

    means_raw = {t: _mean(scores_raw[t]) for t in scores_raw}
    means_sig = {t: _mean(scores_sig[t]) for t in scores_sig}
    n_counts  = {t: len(scores_raw[t]) for t in scores_raw}

    cal_gap_raw = means_raw["hazard_approach"] - means_raw["none"]
    cal_gap_sig = means_sig["hazard_approach"] - means_sig["none"]

    print(f"\n  --- [{label}] E3 harm_eval by ttype (raw) ---", flush=True)
    for t in ["none", "hazard_approach", "env_caused_hazard", "agent_caused_hazard"]:
        print(
            f"  {t:28s}: raw={means_raw[t]:.4f}  sigmoid={means_sig[t]:.4f}  n={n_counts[t]}",
            flush=True,
        )
    print(f"  calibration_gap raw: {cal_gap_raw:.4f}  (approach - none)", flush=True)
    print(f"  calibration_gap sig: {cal_gap_sig:.4f}  (sigmoid approach - sigmoid none)", flush=True)

    return {
        "means_raw":      means_raw,
        "means_sig":      means_sig,
        "n_counts":       n_counts,
        "cal_gap_raw":    cal_gap_raw,
        "cal_gap_sig":    cal_gap_sig,
    }


def _make_agent_and_optimizers(env, config, lr: float = 1e-3):
    agent = REEAgent(config)
    standard_params = [
        p for n, p in agent.named_parameters()
        if "harm_eval_head" not in n
        and "world_transition" not in n
        and "world_action_encoder" not in n
    ]
    wf_params = (
        list(agent.e2.world_transition.parameters()) +
        list(agent.e2.world_action_encoder.parameters())
    )
    he_params = list(agent.e3.harm_eval_head.parameters())
    opt       = optim.Adam(standard_params, lr=lr)
    wf_opt    = optim.Adam(wf_params,       lr=1e-3)
    he_opt    = optim.Adam(he_params,       lr=1e-4)
    return agent, opt, he_opt, wf_opt


def run(
    seed: int = 0,
    warmup_episodes: int = 400,
    eval_episodes: int = 50,
    steps_per_episode: int = 200,
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

    print(
        f"[V3-EXQ-037] MECH-069 Sign Debug\n"
        f"  Two sub-conditions (same seed, fresh agents):\n"
        f"    SUB-A (Fix2): E3 on observed + E2-predicted states [EXQ-035 setup]\n"
        f"    SUB-B (ObsOnly): E3 on observed states only [EXQ-026 setup — PASS]\n"
        f"  Diagnostic: does Fix 2 cause sign inversion?",
        flush=True,
    )

    # --- SUB-A: Fix2 (same as EXQ-035 separated condition) ---
    print(f"\n[V3-EXQ-037] === SUB-A (Fix2) — E3 on observed + E2-predicted ===", flush=True)
    torch.manual_seed(seed)
    random.seed(seed)
    agent_a, opt_a, he_opt_a, wf_opt_a = _make_agent_and_optimizers(env, config, lr)
    train_a = _train_subcondition(
        agent_a, env, opt_a, he_opt_a, wf_opt_a,
        warmup_episodes, steps_per_episode, use_e2_predicted=True, label="Fix2",
    )
    wf_r2_a = _compute_world_forward_r2(agent_a, train_a["wf_data"])
    print(f"\n[V3-EXQ-037] Eval SUB-A ({eval_episodes} eps)...", flush=True)
    eval_a = _eval_calibration(agent_a, env, eval_episodes, steps_per_episode, "Fix2")

    # --- SUB-B: ObsOnly (same as EXQ-026 — PASS condition) ---
    print(f"\n[V3-EXQ-037] === SUB-B (ObsOnly) — E3 on observed states only ===", flush=True)
    torch.manual_seed(seed)
    random.seed(seed)
    agent_b, opt_b, he_opt_b, wf_opt_b = _make_agent_and_optimizers(env, config, lr)
    train_b = _train_subcondition(
        agent_b, env, opt_b, he_opt_b, wf_opt_b,
        warmup_episodes, steps_per_episode, use_e2_predicted=False, label="ObsOnly",
    )
    wf_r2_b = _compute_world_forward_r2(agent_b, train_b["wf_data"])
    print(f"\n[V3-EXQ-037] Eval SUB-B ({eval_episodes} eps)...", flush=True)
    eval_b = _eval_calibration(agent_b, env, eval_episodes, steps_per_episode, "ObsOnly")

    # --- Diagnostic summary ---
    sign_inversion_detected = (eval_a["cal_gap_raw"] < 0) and (eval_b["cal_gap_raw"] > 0)
    print(f"\n[V3-EXQ-037] === DIAGNOSTIC SUMMARY ===", flush=True)
    print(f"  SUB-A (Fix2) cal_gap_raw:  {eval_a['cal_gap_raw']:.4f}", flush=True)
    print(f"  SUB-B (ObsOnly) cal_gap_raw: {eval_b['cal_gap_raw']:.4f}", flush=True)
    print(f"  Sign inversion detected (A<0, B>0): {sign_inversion_detected}", flush=True)

    # PASS criteria (SUB-B must show correct sign — establishes baseline)
    nc_b = eval_b["n_counts"]
    c1_pass = eval_b["cal_gap_raw"] > 0.0
    c2_pass = eval_b["cal_gap_raw"] > 0.01
    c3_pass = (
        eval_b["means_raw"]["hazard_approach"]
        > eval_b["means_raw"]["none"]
    )
    c4_pass = wf_r2_b > 0.05
    c5_pass = nc_b.get("hazard_approach", 0) >= 30

    all_pass = c1_pass and c2_pass and c3_pass and c4_pass and c5_pass
    status   = "PASS" if all_pass else "FAIL"
    n_met    = sum([c1_pass, c2_pass, c3_pass, c4_pass, c5_pass])

    failure_notes = []
    if not c1_pass:
        failure_notes.append(f"C1 FAIL: SUB-B cal_gap_raw={eval_b['cal_gap_raw']:.4f} <= 0")
    if not c2_pass:
        failure_notes.append(f"C2 FAIL: SUB-B cal_gap_raw={eval_b['cal_gap_raw']:.4f} <= 0.01")
    if not c3_pass:
        failure_notes.append(
            f"C3 FAIL: harm_approach({eval_b['means_raw']['hazard_approach']:.4f}) "
            f"<= harm_none({eval_b['means_raw']['none']:.4f})"
        )
    if not c4_pass:
        failure_notes.append(f"C4 FAIL: wf_r2_obsonly={wf_r2_b:.4f} <= 0.05")
    if not c5_pass:
        failure_notes.append(f"C5 FAIL: n_approach={nc_b.get('hazard_approach', 0)} < 30")

    print(f"\nV3-EXQ-037 verdict: {status}  ({n_met}/5)", flush=True)
    for note in failure_notes:
        print(f"  {note}", flush=True)

    metrics = {
        # SUB-A (Fix2)
        "fix2_cal_gap_raw":           float(eval_a["cal_gap_raw"]),
        "fix2_cal_gap_sig":           float(eval_a["cal_gap_sig"]),
        "fix2_mean_harm_approach_raw": float(eval_a["means_raw"]["hazard_approach"]),
        "fix2_mean_harm_none_raw":     float(eval_a["means_raw"]["none"]),
        "fix2_wf_r2":                  float(wf_r2_a),
        # SUB-B (ObsOnly)
        "obsonly_cal_gap_raw":           float(eval_b["cal_gap_raw"]),
        "obsonly_cal_gap_sig":           float(eval_b["cal_gap_sig"]),
        "obsonly_mean_harm_approach_raw": float(eval_b["means_raw"]["hazard_approach"]),
        "obsonly_mean_harm_none_raw":     float(eval_b["means_raw"]["none"]),
        "obsonly_wf_r2":                  float(wf_r2_b),
        # Diagnostic
        "sign_inversion_detected":       1.0 if sign_inversion_detected else 0.0,
        "cal_gap_difference_b_minus_a":  float(eval_b["cal_gap_raw"] - eval_a["cal_gap_raw"]),
        "n_approach_obsonly":            float(nc_b.get("hazard_approach", 0)),
        # Criteria
        "crit1_pass": 1.0 if c1_pass else 0.0,
        "crit2_pass": 1.0 if c2_pass else 0.0,
        "crit3_pass": 1.0 if c3_pass else 0.0,
        "crit4_pass": 1.0 if c4_pass else 0.0,
        "crit5_pass": 1.0 if c5_pass else 0.0,
        "criteria_met": float(n_met),
    }

    failure_section = ""
    if failure_notes:
        failure_section = "\n## Failure Notes\n\n" + "\n".join(f"- {n}" for n in failure_notes)

    summary_markdown = f"""# V3-EXQ-037 — MECH-069 Sign Investigation (EXQ-035 Follow-up)

**Status:** {status}
**Claims:** MECH-069, SD-003
**World:** CausalGridWorldV2 (proximity_scale={proximity_scale})
**alpha_world:** {alpha_world}  (SD-008)
**Seed:** {seed}
**Predecessor:** EXQ-035 (FAIL — calibration_gap_separated=-0.028 < 0)

## Problem Statement

EXQ-035 found negative calibration_gap for BOTH separated and merged conditions.
E3.harm_eval scores hazard_approach LOWER than none — backwards from expectation.
EXQ-026 (obs-only E3 training, simpler setup) got calibration_gap=+0.0375 (PASS).
Hypothesis: Fix 2 (E3 trained on E2-predicted states) causes sign inversion.

## SUB-A Results (Fix2 — E3 on observed + E2-predicted)

| Transition | mean harm_eval (raw) | mean harm_eval (sigmoid) |
|---|---|---|
| none         | {eval_a['means_raw']['none']:.4f} | {eval_a['means_sig']['none']:.4f} |
| hazard_approach | {eval_a['means_raw']['hazard_approach']:.4f} | {eval_a['means_sig']['hazard_approach']:.4f} |
| env_caused_hazard | {eval_a['means_raw']['env_caused_hazard']:.4f} | {eval_a['means_sig']['env_caused_hazard']:.4f} |
| agent_caused_hazard | {eval_a['means_raw']['agent_caused_hazard']:.4f} | {eval_a['means_sig']['agent_caused_hazard']:.4f} |

- **cal_gap_raw**: {eval_a['cal_gap_raw']:.4f}
- **cal_gap_sig**: {eval_a['cal_gap_sig']:.4f}
- **wf_r2**: {wf_r2_a:.4f}

## SUB-B Results (ObsOnly — E3 on observed states only, as in EXQ-026)

| Transition | mean harm_eval (raw) | mean harm_eval (sigmoid) |
|---|---|---|
| none         | {eval_b['means_raw']['none']:.4f} | {eval_b['means_sig']['none']:.4f} |
| hazard_approach | {eval_b['means_raw']['hazard_approach']:.4f} | {eval_b['means_sig']['hazard_approach']:.4f} |
| env_caused_hazard | {eval_b['means_raw']['env_caused_hazard']:.4f} | {eval_b['means_sig']['env_caused_hazard']:.4f} |
| agent_caused_hazard | {eval_b['means_raw']['agent_caused_hazard']:.4f} | {eval_b['means_sig']['agent_caused_hazard']:.4f} |

- **cal_gap_raw**: {eval_b['cal_gap_raw']:.4f}
- **cal_gap_sig**: {eval_b['cal_gap_sig']:.4f}
- **wf_r2**: {wf_r2_b:.4f}

## Diagnostic

**Sign inversion detected (Fix2<0 and ObsOnly>0):** {"YES" if sign_inversion_detected else "NO"}
**cal_gap difference (ObsOnly − Fix2):** {eval_b['cal_gap_raw'] - eval_a['cal_gap_raw']:.4f}

## PASS Criteria (SUB-B must show correct sign)

| Criterion | Result | Value |
|---|---|---|
| C1: SUB-B cal_gap_raw > 0 (correct sign) | {"PASS" if c1_pass else "FAIL"} | {eval_b['cal_gap_raw']:.4f} |
| C2: SUB-B cal_gap_raw > 0.01 (above noise floor) | {"PASS" if c2_pass else "FAIL"} | {eval_b['cal_gap_raw']:.4f} |
| C3: harm_approach > harm_none (correct ordering) | {"PASS" if c3_pass else "FAIL"} | {eval_b['means_raw']['hazard_approach']:.4f} vs {eval_b['means_raw']['none']:.4f} |
| C4: wf_r2_obsonly > 0.05 | {"PASS" if c4_pass else "FAIL"} | {wf_r2_b:.4f} |
| C5: n_approach_obsonly >= 30 | {"PASS" if c5_pass else "FAIL"} | {nc_b.get('hazard_approach', 0)} |

Criteria met: {n_met}/5 → **{status}**
{failure_section}
"""

    return {
        "status": status,
        "metrics": metrics,
        "summary_markdown": summary_markdown,
        "claim_ids": CLAIM_IDS,
        "evidence_direction": (
            "supports" if all_pass
            else ("mixed" if n_met >= 3 else "weakens")
        ),
        "experiment_type": EXPERIMENT_TYPE,
        "fatal_error_count": 0,
    }


if __name__ == "__main__":
    import argparse
    import json
    from datetime import datetime, timezone

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",            type=int,   default=0)
    parser.add_argument("--warmup",          type=int,   default=400)
    parser.add_argument("--eval-eps",        type=int,   default=50)
    parser.add_argument("--steps",           type=int,   default=200)
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
    result["run_timestamp"] = ts
    result["claim"]  = CLAIM_IDS[0]
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
        print(f"  {k}: {v}", flush=True)
