"""
V3-EXQ-035 — MECH-069 Optimizer Separation vs. Merge Ablation

Claims: MECH-069, SD-003

MECH-069 (`learning.three_incommensurable_error_signals`) asserts that sensory prediction
error (E1), motor-sensory error (E2_self), and harm/goal error (E3) are incommensurable.
Merging them into a single optimizer degrades each capacity because gradient interference
prevents proper specialization.

Two conditions run sequentially, each with a fresh agent from the same seed:

  SEPARATED: Three optimizers (as validated in EXQ-030b)
    - standard_optimizer:       all params except world_transition, world_action_encoder,
                                harm_eval_head (lr=1e-3)
    - world_forward_optimizer:  e2.world_transition + e2.world_action_encoder (lr=1e-3)
    - harm_eval_optimizer:      e3.harm_eval_head (lr=1e-4)
    Each optimizer has its own zero_grad/backward/step cycle.

  MERGED: Single Adam optimizer over ALL parameters (lr=1e-3)
    - merged_loss = e1_loss + e2_loss + wf_loss + harm_eval_loss
    - Single backward pass, single step.

PASS criteria (ALL five must hold):
  C1: calibration_gap_approach_separated > calibration_gap_approach_merged
      (separation improves calibration)
  C2: attribution_gap_separated > attribution_gap_merged
      (separation improves attribution)
  C3: calibration_gap_approach_separated > 0.05
      (separated system achieves minimum viable calibration)
  C4: wf_r2_separated > wf_r2_merged  OR  wf_r2_separated > 0.10
      (separation helps world-forward learning OR separated already good)
  C5: n_approach_eval_separated >= 50
      (sufficient approach events observed during separated eval)

Architecture basis: MECH-069 (incommensurable error signals), SD-003 (counterfactual
self-attribution), EXQ-030b (validated separated pipeline baseline).
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


EXPERIMENT_TYPE = "v3_exq_035_mech069_optimizer_merge"
CLAIM_IDS = ["MECH-069", "SD-003"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _action_to_onehot(action_idx: int, num_actions: int, device) -> torch.Tensor:
    v = torch.zeros(1, num_actions, device=device)
    v[0, action_idx] = 1.0
    return v


def _compute_world_forward_r2(
    agent: REEAgent,
    wf_data: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
) -> float:
    """R² of E2.world_forward on held-out (20 %) transitions."""
    if len(wf_data) < 20:
        return 0.0
    n = len(wf_data)
    n_train = int(n * 0.8)
    with torch.no_grad():
        zw_all  = torch.cat([d[0] for d in wf_data], dim=0)
        a_all   = torch.cat([d[1] for d in wf_data], dim=0)
        zw1_all = torch.cat([d[2] for d in wf_data], dim=0)
        pred_all  = agent.e2.world_forward(zw_all.to(agent.device), a_all.to(agent.device))
        pred_test = pred_all[n_train:].cpu()
        tgt_test  = zw1_all[n_train:]
        if pred_test.shape[0] == 0:
            return 0.0
        ss_res = ((tgt_test - pred_test) ** 2).sum()
        ss_tot = ((tgt_test - tgt_test.mean(0, keepdim=True)) ** 2).sum()
        r2 = float((1.0 - ss_res / (ss_tot + 1e-8)).item())
    print(f"  world_forward R² (test n={pred_test.shape[0]}): {r2:.4f}", flush=True)
    return r2


# ---------------------------------------------------------------------------
# Harm-eval loss computation (shared logic, both conditions)
# ---------------------------------------------------------------------------

def _compute_harm_eval_loss(
    agent: REEAgent,
    harm_buf_pos: List[torch.Tensor],
    harm_buf_neg: List[torch.Tensor],
    num_actions: int,
) -> torch.Tensor:
    """
    Computes harm_eval loss using observed + E2-predicted states (Fix 2 from EXQ-030b).
    Returns a scalar tensor with gradients.  Returns 0-grad tensor if buffers too small.
    """
    if len(harm_buf_pos) < 4 or len(harm_buf_neg) < 4:
        return torch.tensor(0.0, device=agent.device)

    k_pos = min(16, len(harm_buf_pos))
    k_neg = min(16, len(harm_buf_neg))
    pos_idx = torch.randperm(len(harm_buf_pos))[:k_pos].tolist()
    neg_idx = torch.randperm(len(harm_buf_neg))[:k_neg].tolist()

    zw_pos_obs = torch.cat([harm_buf_pos[i] for i in pos_idx], dim=0).to(agent.device)
    zw_neg_obs = torch.cat([harm_buf_neg[i] for i in neg_idx], dim=0).to(agent.device)

    with torch.no_grad():
        a_rand_pos = torch.zeros(k_pos, num_actions, device=agent.device)
        a_rand_pos[torch.arange(k_pos), torch.randint(0, num_actions, (k_pos,))] = 1.0
        a_rand_neg = torch.zeros(k_neg, num_actions, device=agent.device)
        a_rand_neg[torch.arange(k_neg), torch.randint(0, num_actions, (k_neg,))] = 1.0
        zw_pos_pred = agent.e2.world_forward(zw_pos_obs, a_rand_pos)
        zw_neg_pred = agent.e2.world_forward(zw_neg_obs, a_rand_neg)

    zw_b = torch.cat([zw_pos_obs, zw_neg_obs, zw_pos_pred, zw_neg_pred], dim=0)
    target = torch.cat([
        torch.ones(k_pos,  1, device=agent.device),
        torch.zeros(k_neg, 1, device=agent.device),
        torch.ones(k_pos,  1, device=agent.device),
        torch.zeros(k_neg, 1, device=agent.device),
    ], dim=0)

    pred_harm = agent.e3.harm_eval(zw_b)
    return F.mse_loss(pred_harm, target)


# ---------------------------------------------------------------------------
# World-forward loss computation (shared logic, both conditions)
# ---------------------------------------------------------------------------

def _compute_wf_loss(
    agent: REEAgent,
    wf_data: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
) -> torch.Tensor:
    """
    Samples a mini-batch from wf_data and returns the world-forward MSE loss tensor.
    Returns 0-grad tensor if buffer too small.
    """
    if len(wf_data) < 16:
        return torch.tensor(0.0, device=agent.device)

    k = min(32, len(wf_data))
    idxs = torch.randperm(len(wf_data))[:k].tolist()
    zw_b  = torch.cat([wf_data[i][0] for i in idxs]).to(agent.device)
    a_b   = torch.cat([wf_data[i][1] for i in idxs]).to(agent.device)
    zw1_b = torch.cat([wf_data[i][2] for i in idxs]).to(agent.device)
    pred = agent.e2.world_forward(zw_b, a_b)
    return F.mse_loss(pred, zw1_b)


# ---------------------------------------------------------------------------
# SEPARATED training
# ---------------------------------------------------------------------------

def _train_separated(
    agent: REEAgent,
    env,
    standard_optimizer: optim.Optimizer,
    world_forward_optimizer: optim.Optimizer,
    harm_eval_optimizer: optim.Optimizer,
    num_episodes: int,
    steps_per_episode: int,
) -> Dict:
    """
    Three separate optimizers — one backward pass per loss family per step.
    Replicates EXQ-030b structure exactly.
    """
    agent.train()

    harm_buf_pos: List[torch.Tensor] = []
    harm_buf_neg: List[torch.Tensor] = []
    MAX_BUF = 2000

    wf_data: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    MAX_WF = 5000

    counts: Dict[str, int] = {}
    num_actions = env.action_dim

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

            # Harm buffer (positive = any harm, including approach)
            if harm_signal < 0:
                harm_buf_pos.append(z_world_curr)
                if len(harm_buf_pos) > MAX_BUF:
                    harm_buf_pos = harm_buf_pos[-MAX_BUF:]
            else:
                harm_buf_neg.append(z_world_curr)
                if len(harm_buf_neg) > MAX_BUF:
                    harm_buf_neg = harm_buf_neg[-MAX_BUF:]

            # World-forward buffer
            if z_world_prev is not None and a_prev is not None:
                wf_data.append((z_world_prev.cpu(), a_prev.cpu(), z_world_curr.cpu()))
                if len(wf_data) > MAX_WF:
                    wf_data = wf_data[-MAX_WF:]

            # --- Optimizer 1: Standard (E1 + E2_self losses) ---
            e1_loss = agent.compute_prediction_loss()
            e2_loss = agent.compute_e2_loss()
            total_loss = e1_loss + e2_loss
            if total_loss.requires_grad:
                standard_optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                standard_optimizer.step()

            # --- Optimizer 2: World-forward (E2.world_transition) ---
            wf_loss = _compute_wf_loss(agent, wf_data)
            if wf_loss.requires_grad:
                world_forward_optimizer.zero_grad()
                wf_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(agent.e2.world_transition.parameters()) +
                    list(agent.e2.world_action_encoder.parameters()),
                    0.5,
                )
                world_forward_optimizer.step()

            # --- Optimizer 3: Harm-eval (E3.harm_eval_head) ---
            harm_eval_loss = _compute_harm_eval_loss(
                agent, harm_buf_pos, harm_buf_neg, num_actions
            )
            if harm_eval_loss.requires_grad:
                harm_eval_optimizer.zero_grad()
                harm_eval_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    agent.e3.harm_eval_head.parameters(), 0.5
                )
                harm_eval_optimizer.step()

            z_world_prev = z_world_curr
            a_prev = action.detach()

            if done:
                break

        if (ep + 1) % 50 == 0 or ep == num_episodes - 1:
            approach = counts.get("hazard_approach", 0)
            contact  = counts.get("env_caused_hazard", 0) + counts.get("agent_caused_hazard", 0)
            print(
                f"  [separated train] ep {ep+1}/{num_episodes}  "
                f"approach={approach}  contact={contact}  wf_buf={len(wf_data)}",
                flush=True,
            )

    return {"counts": counts, "wf_data": wf_data}


# ---------------------------------------------------------------------------
# MERGED training
# ---------------------------------------------------------------------------

def _train_merged(
    agent: REEAgent,
    env,
    merged_optimizer: optim.Optimizer,
    num_episodes: int,
    steps_per_episode: int,
) -> Dict:
    """
    Single optimizer — all four loss components summed into one backward pass.
    Ablation condition for MECH-069.
    """
    agent.train()

    harm_buf_pos: List[torch.Tensor] = []
    harm_buf_neg: List[torch.Tensor] = []
    MAX_BUF = 2000

    wf_data: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    MAX_WF = 5000

    counts: Dict[str, int] = {}
    num_actions = env.action_dim

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

            # Harm buffer
            if harm_signal < 0:
                harm_buf_pos.append(z_world_curr)
                if len(harm_buf_pos) > MAX_BUF:
                    harm_buf_pos = harm_buf_pos[-MAX_BUF:]
            else:
                harm_buf_neg.append(z_world_curr)
                if len(harm_buf_neg) > MAX_BUF:
                    harm_buf_neg = harm_buf_neg[-MAX_BUF:]

            # World-forward buffer
            if z_world_prev is not None and a_prev is not None:
                wf_data.append((z_world_prev.cpu(), a_prev.cpu(), z_world_curr.cpu()))
                if len(wf_data) > MAX_WF:
                    wf_data = wf_data[-MAX_WF:]

            # Collect all loss components
            e1_loss = agent.compute_prediction_loss()
            e2_loss = agent.compute_e2_loss()
            wf_loss = _compute_wf_loss(agent, wf_data)
            harm_eval_loss = _compute_harm_eval_loss(
                agent, harm_buf_pos, harm_buf_neg, num_actions
            )

            # Single combined backward pass (gradient interference — the ablation)
            merged_loss = e1_loss + e2_loss + wf_loss + harm_eval_loss
            if merged_loss.requires_grad:
                merged_optimizer.zero_grad()
                merged_loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                merged_optimizer.step()

            z_world_prev = z_world_curr
            a_prev = action.detach()

            if done:
                break

        if (ep + 1) % 50 == 0 or ep == num_episodes - 1:
            approach = counts.get("hazard_approach", 0)
            contact  = counts.get("env_caused_hazard", 0) + counts.get("agent_caused_hazard", 0)
            print(
                f"  [merged train] ep {ep+1}/{num_episodes}  "
                f"approach={approach}  contact={contact}  wf_buf={len(wf_data)}",
                flush=True,
            )

    return {"counts": counts, "wf_data": wf_data}


# ---------------------------------------------------------------------------
# Eval (shared between both conditions)
# ---------------------------------------------------------------------------

def _eval_attribution(
    agent: REEAgent,
    env,
    num_episodes: int,
    steps_per_episode: int,
    condition_label: str,
) -> Dict:
    """
    Full SD-003 attribution eval (identical to EXQ-030b).

    Per step:
      z_world_actual  = E2.world_forward(z_world_t, a_actual)
      z_world_cf      = E2.world_forward(z_world_t, a_cf)  for each a_cf != a_actual
      causal_sig      = mean_cf [ E3(z_world_actual) - E3(z_world_cf) ]
      calibration_gap_approach = mean E3.harm_eval at hazard_approach steps
                                 minus mean E3.harm_eval at env_caused_hazard steps
    """
    agent.eval()

    ttypes = ["none", "hazard_approach", "env_caused_hazard", "agent_caused_hazard"]
    causal_sigs:   Dict[str, List[float]] = {t: [] for t in ttypes}
    harm_evals:    Dict[str, List[float]] = {t: [] for t in ttypes}

    num_actions = env.action_dim

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

            action_idx = random.randint(0, num_actions - 1)
            action = _action_to_onehot(action_idx, num_actions, agent.device)
            agent._last_action = action

            flat_obs, harm_signal, done, info, obs_dict = env.step(action)
            ttype = info.get("transition_type", "none")

            with torch.no_grad():
                z_world_actual = agent.e2.world_forward(z_world, action)
                harm_actual    = agent.e3.harm_eval(z_world_actual)

                sigs = []
                for cf_idx in range(num_actions):
                    if cf_idx == action_idx:
                        continue
                    a_cf    = _action_to_onehot(cf_idx, num_actions, agent.device)
                    z_cf    = agent.e2.world_forward(z_world, a_cf)
                    harm_cf = agent.e3.harm_eval(z_cf)
                    sigs.append(float((harm_actual - harm_cf).item()))

                mean_sig = float(np.mean(sigs)) if sigs else 0.0
                # Direct harm_eval on current observed z_world (for calibration gap)
                harm_obs = float(agent.e3.harm_eval(z_world).item())

            if ttype in causal_sigs:
                causal_sigs[ttype].append(mean_sig)
                harm_evals[ttype].append(harm_obs)

            if done:
                break

    def _mean(lst: List[float]) -> float:
        return float(np.mean(lst)) if lst else 0.0

    mean_sigs = {t: _mean(causal_sigs[t])  for t in ttypes}
    mean_harm = {t: _mean(harm_evals[t])   for t in ttypes}
    n_counts  = {t: len(causal_sigs[t])    for t in ttypes}

    attribution_gap        = mean_sigs["hazard_approach"] - mean_sigs["env_caused_hazard"]
    calibration_gap_approach = mean_harm["hazard_approach"] - mean_harm["env_caused_hazard"]

    print(f"\n  --- SD-003 Attribution Eval [{condition_label}] ---", flush=True)
    for t in ttypes:
        print(
            f"  {t:28s}: causal_sig={mean_sigs[t]:.6f}  harm_eval={mean_harm[t]:.4f}  n={n_counts[t]}",
            flush=True,
        )
    print(f"  attribution_gap (approach-env):         {attribution_gap:.6f}", flush=True)
    print(f"  calibration_gap_approach (approach-env): {calibration_gap_approach:.6f}", flush=True)

    return {
        "mean_sigs":               mean_sigs,
        "mean_harm":               mean_harm,
        "n_counts":                n_counts,
        "attribution_gap":         attribution_gap,
        "calibration_gap_approach": calibration_gap_approach,
    }


# ---------------------------------------------------------------------------
# Agent + env factory
# ---------------------------------------------------------------------------

def _make_env_and_agent(
    seed: int,
    self_dim: int,
    world_dim: int,
    alpha_world: float,
    alpha_self: float,
    harm_scale: float,
    proximity_scale: float,
):
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
    return env, agent


# ---------------------------------------------------------------------------
# Main run
# ---------------------------------------------------------------------------

def run(
    seed: int = 0,
    warmup_episodes: int = 500,
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

    print(
        f"[V3-EXQ-035] MECH-069 Optimizer Separation vs Merge Ablation\n"
        f"  seed={seed}  warmup={warmup_episodes}  eval={eval_episodes}  steps={steps_per_episode}\n"
        f"  alpha_world={alpha_world}  harm_scale={harm_scale}  proximity_scale={proximity_scale}",
        flush=True,
    )

    # ------------------------------------------------------------------
    # CONDITION 1: SEPARATED
    # ------------------------------------------------------------------
    print("\n" + "=" * 60, flush=True)
    print("CONDITION: SEPARATED (three optimizers)", flush=True)
    print("=" * 60, flush=True)

    torch.manual_seed(seed)
    random.seed(seed)
    env_sep, agent_sep = _make_env_and_agent(
        seed, self_dim, world_dim, alpha_world, alpha_self, harm_scale, proximity_scale
    )
    print(
        f"  body_obs={env_sep.body_obs_dim}  world_obs={env_sep.world_obs_dim}  "
        f"action_dim={env_sep.action_dim}",
        flush=True,
    )

    standard_params = [
        p for n, p in agent_sep.named_parameters()
        if "harm_eval_head" not in n
        and "world_transition" not in n
        and "world_action_encoder" not in n
    ]
    world_forward_params = (
        list(agent_sep.e2.world_transition.parameters()) +
        list(agent_sep.e2.world_action_encoder.parameters())
    )
    harm_eval_params = list(agent_sep.e3.harm_eval_head.parameters())

    standard_optimizer       = optim.Adam(standard_params,      lr=lr)
    world_forward_optimizer  = optim.Adam(world_forward_params, lr=1e-3)
    harm_eval_optimizer      = optim.Adam(harm_eval_params,     lr=1e-4)

    print(f"  Training {warmup_episodes} episodes (separated)...", flush=True)
    train_sep = _train_separated(
        agent_sep, env_sep,
        standard_optimizer, world_forward_optimizer, harm_eval_optimizer,
        warmup_episodes, steps_per_episode,
    )

    wf_r2_sep = _compute_world_forward_r2(agent_sep, train_sep["wf_data"])

    print(f"\n  Evaluating {eval_episodes} episodes (separated)...", flush=True)
    eval_sep = _eval_attribution(agent_sep, env_sep, eval_episodes, steps_per_episode, "SEPARATED")

    # ------------------------------------------------------------------
    # CONDITION 2: MERGED
    # ------------------------------------------------------------------
    print("\n" + "=" * 60, flush=True)
    print("CONDITION: MERGED (single optimizer)", flush=True)
    print("=" * 60, flush=True)

    torch.manual_seed(seed)
    random.seed(seed)
    env_mrg, agent_mrg = _make_env_and_agent(
        seed, self_dim, world_dim, alpha_world, alpha_self, harm_scale, proximity_scale
    )

    merged_optimizer = optim.Adam(agent_mrg.parameters(), lr=1e-3)

    print(f"  Training {warmup_episodes} episodes (merged)...", flush=True)
    train_mrg = _train_merged(
        agent_mrg, env_mrg, merged_optimizer,
        warmup_episodes, steps_per_episode,
    )

    wf_r2_mrg = _compute_world_forward_r2(agent_mrg, train_mrg["wf_data"])

    print(f"\n  Evaluating {eval_episodes} episodes (merged)...", flush=True)
    eval_mrg = _eval_attribution(agent_mrg, env_mrg, eval_episodes, steps_per_episode, "MERGED")

    # ------------------------------------------------------------------
    # PASS criteria
    # ------------------------------------------------------------------
    cal_sep = eval_sep["calibration_gap_approach"]
    cal_mrg = eval_mrg["calibration_gap_approach"]
    att_sep = eval_sep["attribution_gap"]
    att_mrg = eval_mrg["attribution_gap"]
    n_app_sep = eval_sep["n_counts"]["hazard_approach"]

    c1_pass = cal_sep > cal_mrg
    c2_pass = att_sep > att_mrg
    c3_pass = cal_sep > 0.05
    c4_pass = (wf_r2_sep > wf_r2_mrg) or (wf_r2_sep > 0.10)
    c5_pass = n_app_sep >= 50

    all_pass = c1_pass and c2_pass and c3_pass and c4_pass and c5_pass
    n_met    = sum([c1_pass, c2_pass, c3_pass, c4_pass, c5_pass])
    status   = "PASS" if all_pass else "FAIL"

    failure_notes = []
    if not c1_pass:
        failure_notes.append(
            f"C1 FAIL: calibration_gap_separated({cal_sep:.6f}) <= calibration_gap_merged({cal_mrg:.6f})"
        )
    if not c2_pass:
        failure_notes.append(
            f"C2 FAIL: attribution_gap_separated({att_sep:.6f}) <= attribution_gap_merged({att_mrg:.6f})"
        )
    if not c3_pass:
        failure_notes.append(
            f"C3 FAIL: calibration_gap_separated={cal_sep:.6f} <= 0.05"
        )
    if not c4_pass:
        failure_notes.append(
            f"C4 FAIL: wf_r2_separated={wf_r2_sep:.4f} not > wf_r2_merged={wf_r2_mrg:.4f} AND not > 0.10"
        )
    if not c5_pass:
        failure_notes.append(f"C5 FAIL: n_approach_eval_separated={n_app_sep} < 50")

    print(f"\nV3-EXQ-035 verdict: {status}  ({n_met}/5)", flush=True)
    if failure_notes:
        for note in failure_notes:
            print(f"  {note}", flush=True)

    # ------------------------------------------------------------------
    # Metrics dict
    # ------------------------------------------------------------------
    ms_sep = eval_sep["mean_sigs"]
    ms_mrg = eval_mrg["mean_sigs"]
    mh_sep = eval_sep["mean_harm"]
    mh_mrg = eval_mrg["mean_harm"]
    nc_sep = eval_sep["n_counts"]
    nc_mrg = eval_mrg["n_counts"]
    tc_sep = train_sep["counts"]
    tc_mrg = train_mrg["counts"]

    metrics = {
        # Primary comparison metrics
        "calibration_gap_approach_separated": float(cal_sep),
        "calibration_gap_approach_merged":    float(cal_mrg),
        "attribution_gap_separated":          float(att_sep),
        "attribution_gap_merged":             float(att_mrg),
        "wf_r2_separated":                    float(wf_r2_sep),
        "wf_r2_merged":                       float(wf_r2_mrg),
        # Per-ttype causal_sig (separated)
        "causal_sig_none_separated":          float(ms_sep["none"]),
        "causal_sig_approach_separated":      float(ms_sep["hazard_approach"]),
        "causal_sig_env_hazard_separated":    float(ms_sep["env_caused_hazard"]),
        "causal_sig_agent_hazard_separated":  float(ms_sep["agent_caused_hazard"]),
        # Per-ttype causal_sig (merged)
        "causal_sig_none_merged":             float(ms_mrg["none"]),
        "causal_sig_approach_merged":         float(ms_mrg["hazard_approach"]),
        "causal_sig_env_hazard_merged":       float(ms_mrg["env_caused_hazard"]),
        "causal_sig_agent_hazard_merged":     float(ms_mrg["agent_caused_hazard"]),
        # Per-ttype harm_eval (separated)
        "harm_eval_none_separated":           float(mh_sep["none"]),
        "harm_eval_approach_separated":       float(mh_sep["hazard_approach"]),
        "harm_eval_env_hazard_separated":     float(mh_sep["env_caused_hazard"]),
        "harm_eval_agent_hazard_separated":   float(mh_sep["agent_caused_hazard"]),
        # Per-ttype harm_eval (merged)
        "harm_eval_none_merged":              float(mh_mrg["none"]),
        "harm_eval_approach_merged":          float(mh_mrg["hazard_approach"]),
        "harm_eval_env_hazard_merged":        float(mh_mrg["env_caused_hazard"]),
        "harm_eval_agent_hazard_merged":      float(mh_mrg["agent_caused_hazard"]),
        # Event counts (eval)
        "n_approach_eval_separated":          float(nc_sep["hazard_approach"]),
        "n_approach_eval_merged":             float(nc_mrg["hazard_approach"]),
        "n_env_hazard_eval_separated":        float(nc_sep["env_caused_hazard"]),
        "n_env_hazard_eval_merged":           float(nc_mrg["env_caused_hazard"]),
        "n_none_eval_separated":              float(nc_sep["none"]),
        "n_none_eval_merged":                 float(nc_mrg["none"]),
        # Training event counts
        "train_approach_events_separated":    float(tc_sep.get("hazard_approach", 0)),
        "train_approach_events_merged":       float(tc_mrg.get("hazard_approach", 0)),
        # Pass/fail per criterion
        "crit1_pass": 1.0 if c1_pass else 0.0,
        "crit2_pass": 1.0 if c2_pass else 0.0,
        "crit3_pass": 1.0 if c3_pass else 0.0,
        "crit4_pass": 1.0 if c4_pass else 0.0,
        "crit5_pass": 1.0 if c5_pass else 0.0,
        "criteria_met": float(n_met),
        # Config
        "alpha_world": float(alpha_world),
        "proximity_scale": float(proximity_scale),
        "warmup_episodes": float(warmup_episodes),
        "seed": float(seed),
    }

    # ------------------------------------------------------------------
    # Summary markdown
    # ------------------------------------------------------------------
    failure_section = ""
    if failure_notes:
        failure_section = "\n## Failure Notes\n\n" + "\n".join(f"- {n}" for n in failure_notes)

    def _p(v: bool) -> str:
        return "PASS" if v else "FAIL"

    summary_markdown = f"""# V3-EXQ-035 — MECH-069 Optimizer Separation vs. Merge Ablation

**Status:** {status}  ({n_met}/5 criteria)
**Claims:** MECH-069, SD-003
**World:** CausalGridWorldV2 (size=12, num_hazards=4, num_resources=5)
**alpha_world:** {alpha_world}  (SD-008)
**Seed:** {seed}
**Warmup:** {warmup_episodes} episodes  |  **Eval:** {eval_episodes} episodes

## Hypothesis

MECH-069 (`learning.three_incommensurable_error_signals`): sensory prediction error (E1),
motor-sensory error (E2_self), and harm/goal error (E3) are incommensurable. A single merged
optimizer allows gradient interference that prevents each loss from specialising correctly,
degrading calibration, attribution, and world-forward prediction relative to three separated
optimizers.

## Conditions

- **SEPARATED**: standard_optimizer (all params except wf/harm; lr=1e-3) + world_forward_optimizer
  (e2.world_transition + e2.world_action_encoder; lr=1e-3) + harm_eval_optimizer
  (e3.harm_eval_head; lr=1e-4). Independent backward passes.
- **MERGED**: Single Adam(lr=1e-3) over all parameters. merged_loss = e1 + e2 + wf + harm_eval,
  single backward pass.

Both conditions use identical CausalGridWorldV2 configuration, identical random seed reset,
identical E3-on-E2-predictions training fix (from EXQ-030b).

## Results

### Key Comparison

| Metric | Separated | Merged | Delta |
|---|---|---|---|
| calibration_gap_approach | {cal_sep:.6f} | {cal_mrg:.6f} | {cal_sep - cal_mrg:+.6f} |
| attribution_gap | {att_sep:.6f} | {att_mrg:.6f} | {att_sep - att_mrg:+.6f} |
| world_forward_r2 | {wf_r2_sep:.4f} | {wf_r2_mrg:.4f} | {wf_r2_sep - wf_r2_mrg:+.4f} |

### Causal Signature by Transition Type

| Transition | Separated | Merged |
|---|---|---|
| none (locomotion) | {ms_sep['none']:.6f} | {ms_mrg['none']:.6f} |
| hazard_approach | {ms_sep['hazard_approach']:.6f} | {ms_mrg['hazard_approach']:.6f} |
| env_caused_hazard | {ms_sep['env_caused_hazard']:.6f} | {ms_mrg['env_caused_hazard']:.6f} |
| agent_caused_hazard | {ms_sep['agent_caused_hazard']:.6f} | {ms_mrg['agent_caused_hazard']:.6f} |

### Harm-Eval Score by Transition Type

| Transition | Separated | Merged |
|---|---|---|
| none (locomotion) | {mh_sep['none']:.4f} | {mh_mrg['none']:.4f} |
| hazard_approach | {mh_sep['hazard_approach']:.4f} | {mh_mrg['hazard_approach']:.4f} |
| env_caused_hazard | {mh_sep['env_caused_hazard']:.4f} | {mh_mrg['env_caused_hazard']:.4f} |
| agent_caused_hazard | {mh_sep['agent_caused_hazard']:.4f} | {mh_mrg['agent_caused_hazard']:.4f} |

### Eval Counts (approach events)

| Condition | n_approach | n_env_hazard | n_none |
|---|---|---|---|
| Separated | {nc_sep['hazard_approach']} | {nc_sep['env_caused_hazard']} | {nc_sep['none']} |
| Merged | {nc_mrg['hazard_approach']} | {nc_mrg['env_caused_hazard']} | {nc_mrg['none']} |

## PASS Criteria

| Criterion | Result | Value |
|---|---|---|
| C1: calibration_gap_approach_separated > calibration_gap_approach_merged | {_p(c1_pass)} | {cal_sep:.6f} vs {cal_mrg:.6f} |
| C2: attribution_gap_separated > attribution_gap_merged | {_p(c2_pass)} | {att_sep:.6f} vs {att_mrg:.6f} |
| C3: calibration_gap_approach_separated > 0.05 | {_p(c3_pass)} | {cal_sep:.6f} |
| C4: wf_r2_separated > wf_r2_merged OR > 0.10 | {_p(c4_pass)} | {wf_r2_sep:.4f} vs {wf_r2_mrg:.4f} |
| C5: n_approach_eval_separated >= 50 | {_p(c5_pass)} | {n_app_sep} |

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


# ---------------------------------------------------------------------------
# Entry point (explorer-launch pattern)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import json
    from datetime import datetime, timezone

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",            type=int,   default=0)
    parser.add_argument("--warmup",          type=int,   default=500)
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
    result["claim"]          = CLAIM_IDS[0]
    result["verdict"]        = result["status"]
    result["run_id"]         = f"{EXPERIMENT_TYPE}_{ts}_v3"
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
