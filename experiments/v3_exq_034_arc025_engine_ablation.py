"""
V3-EXQ-034 — ARC-025 Three-Engine Irreducibility Ablation

Claims: ARC-025, SD-003, MECH-071

ARC-025 (architecture.three_engine_irreducibility) asserts that the three-engine
architecture (E1 + E2 + E3) is irreducible: removing any single engine collapses
a distinct capacity that the full system passes.

Three conditions (same env, same seed, same architecture — only ablations differ):

  full       — Normal pipeline. E2.world_forward trained. E3.harm_eval trained.
               Measures calibration_gap_approach + attribution_gap.

  e3_ablated — E3.harm_eval replaced by constant 0.5 at eval; harm_eval training
               skipped entirely. Tests that E3 is load-bearing for calibration.
               PREDICTION: calibration_gap_approach ≈ 0 (E3 outputs constant 0.5).

  e2_ablated — E2.world_forward replaced by identity (returns z_world unchanged,
               ignoring action) at eval. E2 trains normally. Tests that E2 is
               load-bearing for action-conditional attribution.
               PREDICTION: attribution_gap ≈ 0 (all actions predict same z_world).

Set A — calibration metrics (EXQ-029 pattern):
  harm_buf_pos: hazard_approach + contact harm events
  calibration_gap_approach = mean(harm_eval at hazard_approach) - mean(harm_eval at none)
  calibration_gap_contact  = mean(harm_eval at env/agent_caused_hazard) - mean(harm_eval at none)
  For e3_ablated: harm_eval always 0.5, so both gaps ≈ 0.

Set B — attribution metrics (EXQ-030b pattern):
  For each step: causal_sig = mean_cf [E3(E2(z_world, a_actual)) - E3(E2(z_world, a_cf))]
  attribution_gap = mean_causal_sig(hazard_approach) - mean_causal_sig(env_caused_hazard)
  For e2_ablated: world_forward = identity, so all actions predict same z_world → sig ≈ 0.

PASS criteria (ALL five must hold):
  C1: calibration_gap_approach_full > calibration_gap_approach_e3_ablated + 0.05
      (E3 is load-bearing for calibration)
  C2: attribution_gap_full > attribution_gap_e2_ablated + 0.01
      (E2 is load-bearing for attribution)
  C3: calibration_gap_approach_full > 0.10
      (full system works on calibration)
  C4: attribution_gap_full > 0.0
      (full attribution direction correct)
  C5: n_approach_eval >= 50
      (sufficient approach steps in eval)

Adapted from EXQ-029 (calibration) and EXQ-030b (attribution, Fix 2: E3 on E2-predicted
states).  Each condition gets a fresh agent and identical env seed — training and eval are
run sequentially per condition.
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


EXPERIMENT_TYPE = "v3_exq_034_arc025_engine_ablation"
CLAIM_IDS = ["ARC-025", "SD-003", "MECH-071"]

# ------------------------------------------------------------------
# Shared helpers
# ------------------------------------------------------------------

def _action_to_onehot(action_idx: int, num_actions: int, device) -> torch.Tensor:
    v = torch.zeros(1, num_actions, device=device)
    v[0, action_idx] = 1.0
    return v


def _compute_world_forward_r2(
    agent: REEAgent,
    wf_data: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
) -> float:
    """R² of E2.world_forward on held-out transitions."""
    if len(wf_data) < 20:
        return 0.0
    n = len(wf_data)
    n_train = int(n * 0.8)
    with torch.no_grad():
        zw_all  = torch.cat([d[0] for d in wf_data], dim=0)
        a_all   = torch.cat([d[1] for d in wf_data], dim=0)
        zw1_all = torch.cat([d[2] for d in wf_data], dim=0)
        pred_all  = agent.e2.world_forward(zw_all.to(agent.device), a_all.to(agent.device))
        pred_test = pred_all[n_train:]
        tgt_test  = zw1_all[n_train:].to(agent.device)
        if pred_test.shape[0] == 0:
            return 0.0
        ss_res = ((tgt_test - pred_test) ** 2).sum()
        ss_tot = ((tgt_test - tgt_test.mean(0, keepdim=True)) ** 2).sum()
        r2 = float((1 - ss_res / (ss_tot + 1e-8)).item())
    print(f"  world_forward R² (test n={pred_test.shape[0]}): {r2:.4f}", flush=True)
    return r2


# ------------------------------------------------------------------
# Training — identical for all three conditions.
# For e3_ablated: pass train_harm_eval=False → skip harm_eval optimizer steps.
# For e2_ablated: train_harm_eval=True (normal), world_forward trained normally.
# ------------------------------------------------------------------

def _train(
    agent: REEAgent,
    env,
    optimizer: optim.Optimizer,
    harm_eval_optimizer: optim.Optimizer,
    world_forward_optimizer: optim.Optimizer,
    num_episodes: int,
    steps_per_episode: int,
    train_harm_eval: bool = True,
    condition_label: str = "full",
) -> Dict:
    """
    Training loop for one condition.

    train_harm_eval=False → skip harm_eval_optimizer entirely (e3_ablated condition).
    E2.world_forward training is always active (world_forward_optimizer always used).

    E3 harm_eval is trained on both observed z_world states AND E2-predicted z_world states
    (EXQ-030b Fix 2) so that E3 can evaluate the E2-output distribution at eval time.
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

            action_idx = random.randint(0, num_actions - 1)
            action = _action_to_onehot(action_idx, num_actions, agent.device)
            agent._last_action = action

            flat_obs, harm_signal, done, info, obs_dict = env.step(action)
            ttype = info.get("transition_type", "none")
            counts[ttype] = counts.get(ttype, 0) + 1

            # Harm eval training buffer — pos includes approach AND contact events
            if harm_signal < 0:
                harm_buf_pos.append(z_world_curr)
                if len(harm_buf_pos) > MAX_BUF:
                    harm_buf_pos = harm_buf_pos[-MAX_BUF:]
            else:
                harm_buf_neg.append(z_world_curr)
                if len(harm_buf_neg) > MAX_BUF:
                    harm_buf_neg = harm_buf_neg[-MAX_BUF:]

            # World-forward training buffer: all transitions with a previous state
            if z_world_prev is not None and a_prev is not None:
                wf_data.append((z_world_prev.cpu(), a_prev.cpu(), z_world_curr.cpu()))
                if len(wf_data) > MAX_WF:
                    wf_data = wf_data[-MAX_WF:]

            # Standard E1 + E2_self losses (uses main optimizer)
            e1_loss = agent.compute_prediction_loss()
            e2_loss = agent.compute_e2_loss()
            total_loss = e1_loss + e2_loss
            if total_loss.requires_grad:
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                optimizer.step()

            # E2.world_forward training (separate optimizer — MECH-069)
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

            # E3 harm_eval training (skipped for e3_ablated condition)
            # Fix 2 (from EXQ-030b): train on observed + E2-predicted states
            if train_harm_eval and len(harm_buf_pos) >= 4 and len(harm_buf_neg) >= 4:
                k_pos = min(16, len(harm_buf_pos))
                k_neg = min(16, len(harm_buf_neg))
                pos_idx = torch.randperm(len(harm_buf_pos))[:k_pos].tolist()
                neg_idx = torch.randperm(len(harm_buf_neg))[:k_neg].tolist()

                # Observed states
                zw_pos_obs = torch.cat([harm_buf_pos[i] for i in pos_idx], dim=0)
                zw_neg_obs = torch.cat([harm_buf_neg[i] for i in neg_idx], dim=0)

                # E2-predicted states (Fix 2 core: trains E3 on E2-output distribution)
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

        if (ep + 1) % 100 == 0 or ep == num_episodes - 1:
            approach = counts.get("hazard_approach", 0)
            contact  = counts.get("env_caused_hazard", 0) + counts.get("agent_caused_hazard", 0)
            print(
                f"  [{condition_label}|train] ep {ep+1}/{num_episodes}  "
                f"approach={approach}  contact={contact}  wf_buf={len(wf_data)}",
                flush=True,
            )

    return {"counts": counts, "wf_data": wf_data}


# ------------------------------------------------------------------
# Eval — Set A (calibration) + Set B (attribution) in one pass.
# Ablations applied only at eval time via use_e3_ablation / use_e2_ablation flags.
# ------------------------------------------------------------------

def _eval_condition(
    agent: REEAgent,
    env,
    num_episodes: int,
    steps_per_episode: int,
    use_e3_ablation: bool = False,
    use_e2_ablation: bool = False,
    condition_label: str = "full",
) -> Dict:
    """
    Combined calibration + attribution eval.

    use_e3_ablation: replace agent.e3.harm_eval(z) with constant 0.5.
    use_e2_ablation: replace agent.e2.world_forward(z, a) with z.detach() (identity).

    Set A — calibration:
        harm_eval score at each step bucketed by transition_type.
        calibration_gap_approach = mean(approach) - mean(none)
        calibration_gap_contact  = mean(env/agent contact) - mean(none)

    Set B — attribution:
        For each step: compute causal_sig = mean_cf [E3(E2(z, a_actual)) - E3(E2(z, a_cf))]
        attribution_gap = mean_causal_sig(approach) - mean_causal_sig(env_caused_hazard)
    """
    agent.eval()

    ttypes_tracked = ["none", "hazard_approach", "env_caused_hazard", "agent_caused_hazard"]
    num_actions = env.action_dim

    # Set A — calibration buffers
    calibration_scores: Dict[str, List[float]] = {t: [] for t in ttypes_tracked}
    all_harm_scores: List[float] = []

    # Set B — attribution buffers
    causal_sigs: Dict[str, List[float]] = {t: [] for t in ttypes_tracked}

    def _harm_eval_fn(z: torch.Tensor) -> torch.Tensor:
        """E3 harm_eval, or constant 0.5 if e3_ablated."""
        if use_e3_ablation:
            return torch.full((z.shape[0], 1), 0.5, device=z.device)
        return agent.e3.harm_eval(z)

    def _world_forward_fn(z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """E2 world_forward, or identity if e2_ablated."""
        if use_e2_ablation:
            # Identity: ignore action, return z unchanged (detached so no grad)
            return z.detach()
        return agent.e2.world_forward(z, a)

    for ep in range(num_episodes):
        flat_obs, obs_dict = env.reset()
        agent.reset()

        for step in range(steps_per_episode):
            obs_body  = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            with torch.no_grad():
                latent = agent.sense(obs_body, obs_world)
                agent.clock.advance()
                z_world = latent.z_world  # perspective-corrected

            action_idx = random.randint(0, num_actions - 1)
            action = _action_to_onehot(action_idx, num_actions, agent.device)
            agent._last_action = action

            flat_obs, harm_signal, done, info, obs_dict = env.step(action)
            ttype = info.get("transition_type", "none")

            with torch.no_grad():
                # --- Set A: calibration score at current state ---
                harm_score = float(_harm_eval_fn(z_world).item())
                all_harm_scores.append(harm_score)
                if ttype in calibration_scores:
                    calibration_scores[ttype].append(harm_score)

                # --- Set B: attribution (SD-003 counterfactual pipeline) ---
                # Actual predicted next world state
                z_world_actual = _world_forward_fn(z_world, action)
                harm_actual    = _harm_eval_fn(z_world_actual)

                sigs: List[float] = []
                for cf_idx in range(num_actions):
                    if cf_idx == action_idx:
                        continue
                    a_cf    = _action_to_onehot(cf_idx, num_actions, agent.device)
                    z_cf    = _world_forward_fn(z_world, a_cf)
                    harm_cf = _harm_eval_fn(z_cf)
                    sig = float((harm_actual - harm_cf).item())
                    sigs.append(sig)

                mean_sig = float(np.mean(sigs)) if sigs else 0.0
                if ttype in causal_sigs:
                    causal_sigs[ttype].append(mean_sig)

            if done:
                break

    def _mean(lst: List[float]) -> float:
        return float(np.mean(lst)) if lst else 0.0

    cal_means = {t: _mean(calibration_scores[t]) for t in ttypes_tracked}
    sig_means  = {t: _mean(causal_sigs[t])        for t in ttypes_tracked}
    n_counts   = {t: len(calibration_scores[t])   for t in ttypes_tracked}

    harm_pred_std = float(torch.tensor(all_harm_scores).std().item()) if len(all_harm_scores) > 1 else 0.0

    calibration_gap_approach = cal_means["hazard_approach"] - cal_means["none"]
    calibration_gap_contact  = (
        (cal_means["env_caused_hazard"] + cal_means["agent_caused_hazard"]) / 2.0
        - cal_means["none"]
    )
    attribution_gap = sig_means["hazard_approach"] - sig_means["env_caused_hazard"]

    # Print summary
    ablation_tag = ""
    if use_e3_ablation:
        ablation_tag = " [E3_ABLATED]"
    elif use_e2_ablation:
        ablation_tag = " [E2_ABLATED]"

    print(
        f"\n  --- {condition_label}{ablation_tag} Eval ---",
        flush=True,
    )
    print(
        f"  Set A (calibration):",
        flush=True,
    )
    for t in ttypes_tracked:
        print(
            f"    {t:28s}: harm_eval={cal_means[t]:.4f}  n={n_counts[t]}",
            flush=True,
        )
    print(
        f"    calibration_gap_approach: {calibration_gap_approach:.4f}",
        flush=True,
    )
    print(
        f"    calibration_gap_contact:  {calibration_gap_contact:.4f}",
        flush=True,
    )
    print(f"    harm_pred_std: {harm_pred_std:.4f}", flush=True)

    print(f"  Set B (attribution):", flush=True)
    for t in ttypes_tracked:
        print(
            f"    {t:28s}: causal_sig={sig_means[t]:.6f}  n={len(causal_sigs[t])}",
            flush=True,
        )
    print(f"    attribution_gap (approach - env): {attribution_gap:.6f}", flush=True)

    return {
        "cal_means":               cal_means,
        "sig_means":               sig_means,
        "n_counts":                n_counts,
        "calibration_gap_approach": calibration_gap_approach,
        "calibration_gap_contact":  calibration_gap_contact,
        "attribution_gap":          attribution_gap,
        "harm_pred_std":            harm_pred_std,
    }


# ------------------------------------------------------------------
# Build agent + optimizers — reused for each condition
# ------------------------------------------------------------------

def _build_agent_and_optimizers(
    env,
    self_dim: int,
    world_dim: int,
    alpha_world: float,
    alpha_self: float,
    lr: float,
    seed: int,
) -> Tuple[REEAgent, optim.Optimizer, optim.Optimizer, optim.Optimizer]:
    """Fresh agent with three separate optimizers (MECH-069)."""
    torch.manual_seed(seed)
    random.seed(seed)

    config = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=self_dim,
        world_dim=world_dim,
        alpha_world=alpha_world,
        alpha_self=alpha_self,
        reafference_action_dim=env.action_dim,  # SD-007 enabled
    )
    agent = REEAgent(config)

    # Three separate optimizers matching MECH-069
    standard_params = [
        p for n, p in agent.named_parameters()
        if "harm_eval_head" not in n
        and "world_transition" not in n
        and "world_action_encoder" not in n
    ]
    world_forward_params = (
        list(agent.e2.world_transition.parameters()) +
        list(agent.e2.world_action_encoder.parameters())
    )
    harm_eval_params = list(agent.e3.harm_eval_head.parameters())

    optimizer               = optim.Adam(standard_params,      lr=lr)
    world_forward_optimizer = optim.Adam(world_forward_params, lr=1e-3)
    harm_eval_optimizer     = optim.Adam(harm_eval_params,     lr=1e-4)

    return agent, optimizer, world_forward_optimizer, harm_eval_optimizer


# ------------------------------------------------------------------
# Main run() — three conditions sequentially
# ------------------------------------------------------------------

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
    """
    Run all three ablation conditions and compare.
    Each condition gets a fresh agent + identical env seed.
    """
    # Single shared env — re-seeded for each condition via env constructor
    def _make_env():
        return CausalGridWorldV2(
            seed=seed, size=12, num_hazards=4, num_resources=5,
            hazard_harm=harm_scale,
            env_drift_interval=5, env_drift_prob=0.1,
            proximity_harm_scale=proximity_scale,
            proximity_benefit_scale=proximity_scale * 0.6,
            proximity_approach_threshold=0.15,
            hazard_field_decay=0.5,
        )

    print(
        f"[V3-EXQ-034] ARC-025 Three-Engine Irreducibility Ablation\n"
        f"  seed={seed}  warmup={warmup_episodes}  eval_eps={eval_episodes}\n"
        f"  alpha_world={alpha_world}  harm_scale={harm_scale}  proximity_scale={proximity_scale}\n"
        f"  Conditions: full | e3_ablated | e2_ablated",
        flush=True,
    )

    condition_results: Dict[str, Dict] = {}
    condition_wf_r2:   Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Condition 1: full
    # ------------------------------------------------------------------
    print("\n" + "=" * 60, flush=True)
    print("[V3-EXQ-034] CONDITION: full", flush=True)
    print("=" * 60, flush=True)

    env_full = _make_env()
    agent_full, opt_full, wf_opt_full, he_opt_full = _build_agent_and_optimizers(
        env_full, self_dim, world_dim, alpha_world, alpha_self, lr, seed
    )
    print(
        f"  [full] body_obs={env_full.body_obs_dim}  world_obs={env_full.world_obs_dim}  "
        f"actions={env_full.action_dim}",
        flush=True,
    )
    train_full = _train(
        agent_full, env_full, opt_full, he_opt_full, wf_opt_full,
        warmup_episodes, steps_per_episode,
        train_harm_eval=True,
        condition_label="full",
    )
    condition_wf_r2["full"] = _compute_world_forward_r2(agent_full, train_full["wf_data"])

    print(f"[V3-EXQ-034] Eval: full ({eval_episodes} eps)...", flush=True)
    result_full = _eval_condition(
        agent_full, env_full, eval_episodes, steps_per_episode,
        use_e3_ablation=False, use_e2_ablation=False,
        condition_label="full",
    )
    condition_results["full"] = result_full

    # ------------------------------------------------------------------
    # Condition 2: e3_ablated
    # E3.harm_eval training skipped (train_harm_eval=False).
    # At eval: harm_eval replaced by constant 0.5.
    # Prediction: calibration_gap_approach ≈ 0.
    # ------------------------------------------------------------------
    print("\n" + "=" * 60, flush=True)
    print("[V3-EXQ-034] CONDITION: e3_ablated", flush=True)
    print("=" * 60, flush=True)

    env_e3 = _make_env()
    agent_e3, opt_e3, wf_opt_e3, he_opt_e3 = _build_agent_and_optimizers(
        env_e3, self_dim, world_dim, alpha_world, alpha_self, lr, seed
    )
    train_e3 = _train(
        agent_e3, env_e3, opt_e3, he_opt_e3, wf_opt_e3,
        warmup_episodes, steps_per_episode,
        train_harm_eval=False,    # E3 harm_eval training skipped
        condition_label="e3_ablated",
    )
    condition_wf_r2["e3_ablated"] = _compute_world_forward_r2(agent_e3, train_e3["wf_data"])

    print(f"[V3-EXQ-034] Eval: e3_ablated ({eval_episodes} eps)...", flush=True)
    result_e3 = _eval_condition(
        agent_e3, env_e3, eval_episodes, steps_per_episode,
        use_e3_ablation=True, use_e2_ablation=False,
        condition_label="e3_ablated",
    )
    condition_results["e3_ablated"] = result_e3

    # ------------------------------------------------------------------
    # Condition 3: e2_ablated
    # E2.world_forward trains normally. At eval: world_forward replaced by identity.
    # Prediction: all actions produce the same z_world → causal_sig ≈ 0 → attribution_gap ≈ 0.
    # ------------------------------------------------------------------
    print("\n" + "=" * 60, flush=True)
    print("[V3-EXQ-034] CONDITION: e2_ablated", flush=True)
    print("=" * 60, flush=True)

    env_e2 = _make_env()
    agent_e2, opt_e2, wf_opt_e2, he_opt_e2 = _build_agent_and_optimizers(
        env_e2, self_dim, world_dim, alpha_world, alpha_self, lr, seed
    )
    train_e2 = _train(
        agent_e2, env_e2, opt_e2, he_opt_e2, wf_opt_e2,
        warmup_episodes, steps_per_episode,
        train_harm_eval=True,     # E3 trained normally
        condition_label="e2_ablated",
    )
    condition_wf_r2["e2_ablated"] = _compute_world_forward_r2(agent_e2, train_e2["wf_data"])

    print(f"[V3-EXQ-034] Eval: e2_ablated ({eval_episodes} eps)...", flush=True)
    result_e2 = _eval_condition(
        agent_e2, env_e2, eval_episodes, steps_per_episode,
        use_e3_ablation=False, use_e2_ablation=True,  # identity world_forward at eval
        condition_label="e2_ablated",
    )
    condition_results["e2_ablated"] = result_e2

    # ------------------------------------------------------------------
    # Aggregate metrics and apply PASS criteria
    # ------------------------------------------------------------------
    rf  = condition_results["full"]
    re3 = condition_results["e3_ablated"]
    re2 = condition_results["e2_ablated"]

    cal_gap_approach_full    = rf["calibration_gap_approach"]
    cal_gap_approach_e3_abl  = re3["calibration_gap_approach"]
    attr_gap_full            = rf["attribution_gap"]
    attr_gap_e2_abl          = re2["attribution_gap"]
    n_approach_eval          = rf["n_counts"].get("hazard_approach", 0)

    # PASS criteria
    c1_pass = cal_gap_approach_full > cal_gap_approach_e3_abl + 0.05
    c2_pass = attr_gap_full > attr_gap_e2_abl + 0.01
    c3_pass = cal_gap_approach_full > 0.10
    c4_pass = attr_gap_full > 0.0
    c5_pass = n_approach_eval >= 50

    all_pass = c1_pass and c2_pass and c3_pass and c4_pass and c5_pass
    status   = "PASS" if all_pass else "FAIL"
    n_met    = sum([c1_pass, c2_pass, c3_pass, c4_pass, c5_pass])

    failure_notes: List[str] = []
    if not c1_pass:
        failure_notes.append(
            f"C1 FAIL: cal_gap_approach_full({cal_gap_approach_full:.4f}) <= "
            f"cal_gap_approach_e3_abl({cal_gap_approach_e3_abl:.4f}) + 0.05 "
            f"[margin={(cal_gap_approach_full - cal_gap_approach_e3_abl - 0.05):.4f}]"
        )
    if not c2_pass:
        failure_notes.append(
            f"C2 FAIL: attr_gap_full({attr_gap_full:.6f}) <= "
            f"attr_gap_e2_abl({attr_gap_e2_abl:.6f}) + 0.01 "
            f"[margin={(attr_gap_full - attr_gap_e2_abl - 0.01):.6f}]"
        )
    if not c3_pass:
        failure_notes.append(
            f"C3 FAIL: cal_gap_approach_full={cal_gap_approach_full:.4f} <= 0.10"
        )
    if not c4_pass:
        failure_notes.append(
            f"C4 FAIL: attr_gap_full={attr_gap_full:.6f} <= 0"
        )
    if not c5_pass:
        failure_notes.append(
            f"C5 FAIL: n_approach_eval={n_approach_eval} < 50"
        )

    print(f"\n{'=' * 60}", flush=True)
    print(f"V3-EXQ-034 verdict: {status}  ({n_met}/5)", flush=True)
    for note in failure_notes:
        print(f"  {note}", flush=True)
    print(f"{'=' * 60}", flush=True)

    # ------------------------------------------------------------------
    # Build flat metrics dict
    # ------------------------------------------------------------------
    tc_full = train_full["counts"]
    tc_e3   = train_e3["counts"]
    tc_e2   = train_e2["counts"]

    metrics = {
        # Config
        "alpha_world":    float(alpha_world),
        "proximity_scale": float(proximity_scale),

        # world_forward R² — e3_ablated trains E2 too, so wf_r2 should match full
        "wf_r2_full":       float(condition_wf_r2["full"]),
        "wf_r2_e3_ablated": float(condition_wf_r2["e3_ablated"]),
        "wf_r2_e2_ablated": float(condition_wf_r2["e2_ablated"]),

        # Set A — calibration gaps
        "calibration_gap_approach_full":       float(cal_gap_approach_full),
        "calibration_gap_approach_e3_ablated": float(cal_gap_approach_e3_abl),
        "calibration_gap_approach_e2_ablated": float(re2["calibration_gap_approach"]),
        "calibration_gap_contact_full":        float(rf["calibration_gap_contact"]),
        "calibration_gap_contact_e3_ablated":  float(re3["calibration_gap_contact"]),
        "calibration_gap_contact_e2_ablated":  float(re2["calibration_gap_contact"]),

        # Set B — attribution gaps
        "attribution_gap_full":       float(attr_gap_full),
        "attribution_gap_e3_ablated": float(re3["attribution_gap"]),
        "attribution_gap_e2_ablated": float(attr_gap_e2_abl),

        # Causal sig by type (full condition)
        "causal_sig_none_full":         float(rf["sig_means"]["none"]),
        "causal_sig_approach_full":     float(rf["sig_means"]["hazard_approach"]),
        "causal_sig_env_hazard_full":   float(rf["sig_means"]["env_caused_hazard"]),

        # Calibration mean scores (full condition)
        "mean_harm_eval_none_full":         float(rf["cal_means"]["none"]),
        "mean_harm_eval_approach_full":     float(rf["cal_means"]["hazard_approach"]),
        "mean_harm_eval_env_hazard_full":   float(rf["cal_means"]["env_caused_hazard"]),

        # Calibration mean scores (e3_ablated — expected ~0.5 everywhere)
        "mean_harm_eval_none_e3_ablated":     float(re3["cal_means"]["none"]),
        "mean_harm_eval_approach_e3_ablated": float(re3["cal_means"]["hazard_approach"]),

        # harm_pred_std (full condition — should be > 0 if E3 active)
        "harm_pred_std_full":       float(rf["harm_pred_std"]),
        "harm_pred_std_e3_ablated": float(re3["harm_pred_std"]),
        "harm_pred_std_e2_ablated": float(re2["harm_pred_std"]),

        # Eval counts (from full condition)
        "n_approach_eval":      float(n_approach_eval),
        "n_env_hazard_eval":    float(rf["n_counts"].get("env_caused_hazard", 0)),
        "n_agent_hazard_eval":  float(rf["n_counts"].get("agent_caused_hazard", 0)),
        "n_none_eval":          float(rf["n_counts"].get("none", 0)),

        # Training event counts
        "train_approach_events_full":    float(tc_full.get("hazard_approach", 0)),
        "train_approach_events_e3_abl":  float(tc_e3.get("hazard_approach", 0)),
        "train_approach_events_e2_abl":  float(tc_e2.get("hazard_approach", 0)),

        # Criteria
        "crit1_pass": 1.0 if c1_pass else 0.0,
        "crit2_pass": 1.0 if c2_pass else 0.0,
        "crit3_pass": 1.0 if c3_pass else 0.0,
        "crit4_pass": 1.0 if c4_pass else 0.0,
        "crit5_pass": 1.0 if c5_pass else 0.0,
        "criteria_met": float(n_met),
    }

    # ------------------------------------------------------------------
    # Summary markdown
    # ------------------------------------------------------------------
    failure_section = ""
    if failure_notes:
        failure_section = "\n## Failure Notes\n\n" + "\n".join(f"- {n}" for n in failure_notes)

    # Shorthand for table readability
    rfw  = rf["cal_means"]
    re3w = re3["cal_means"]
    re2w = re2["cal_means"]
    rfs  = rf["sig_means"]
    re3s = re3["sig_means"]
    re2s = re2["sig_means"]
    rfnc = rf["n_counts"]

    summary_markdown = f"""# V3-EXQ-034 — ARC-025 Three-Engine Irreducibility Ablation

**Status:** {status}
**Claims:** ARC-025, SD-003, MECH-071
**World:** CausalGridWorldV2 (proximity_scale={proximity_scale}, harm_scale={harm_scale})
**alpha_world:** {alpha_world}  (SD-008)
**Seed:** {seed}
**Warmup:** {warmup_episodes} episodes  **Eval:** {eval_episodes} episodes

## Claim Being Tested

ARC-025 (architecture.three_engine_irreducibility): the three-engine architecture (E1+E2+E3)
is irreducible. Removing any single engine collapses a distinct capacity that the full system
holds. Two engines are tested here:

- **E3** (harm evaluator): ablating E3 should collapse calibration (harm detection).
- **E2** (world forward model): ablating E2 should collapse attribution (causal signature).

## Ablation Design

| Condition   | Training                        | Eval modification                          |
|-------------|--------------------------------|--------------------------------------------|
| full        | E2.world_forward + E3.harm_eval trained | None — normal pipeline              |
| e3_ablated  | E2 trained; E3 harm_eval SKIPPED | harm_eval → constant 0.5               |
| e2_ablated  | Both E2 + E3 trained normally  | world_forward → identity (ignores action)  |

## Set A — Calibration (harm_eval by transition type)

| Transition type        | full   | e3_ablated | e2_ablated |
|------------------------|--------|------------|------------|
| none (locomotion)      | {rfw['none']:.4f} | {re3w['none']:.4f} | {re2w['none']:.4f} |
| hazard_approach        | {rfw['hazard_approach']:.4f} | {re3w['hazard_approach']:.4f} | {re2w['hazard_approach']:.4f} |
| env_caused_hazard      | {rfw['env_caused_hazard']:.4f} | {re3w['env_caused_hazard']:.4f} | {re2w['env_caused_hazard']:.4f} |
| agent_caused_hazard    | {rfw['agent_caused_hazard']:.4f} | {re3w['agent_caused_hazard']:.4f} | {re2w['agent_caused_hazard']:.4f} |

| Metric                   | full   | e3_ablated | e2_ablated |
|--------------------------|--------|------------|------------|
| calibration_gap_approach | **{rf['calibration_gap_approach']:.4f}** | {re3['calibration_gap_approach']:.4f} | {re2['calibration_gap_approach']:.4f} |
| calibration_gap_contact  | {rf['calibration_gap_contact']:.4f} | {re3['calibration_gap_contact']:.4f} | {re2['calibration_gap_contact']:.4f} |
| harm_pred_std            | {rf['harm_pred_std']:.4f} | {re3['harm_pred_std']:.4f} | {re2['harm_pred_std']:.4f} |

## Set B — Attribution (SD-003 causal signature)

| Transition type     | full      | e3_ablated | e2_ablated |
|---------------------|-----------|------------|------------|
| none (locomotion)   | {rfs['none']:.6f} | {re3s['none']:.6f} | {re2s['none']:.6f} |
| hazard_approach     | {rfs['hazard_approach']:.6f} | {re3s['hazard_approach']:.6f} | {re2s['hazard_approach']:.6f} |
| env_caused_hazard   | {rfs['env_caused_hazard']:.6f} | {re3s['env_caused_hazard']:.6f} | {re2s['env_caused_hazard']:.6f} |
| agent_caused_hazard | {rfs['agent_caused_hazard']:.6f} | {re3s['agent_caused_hazard']:.6f} | {re2s['agent_caused_hazard']:.6f} |

| Metric          | full      | e3_ablated | e2_ablated |
|-----------------|-----------|------------|------------|
| attribution_gap | **{rf['attribution_gap']:.6f}** | {re3['attribution_gap']:.6f} | {re2['attribution_gap']:.6f} |

## World-Forward R²

| Condition   | wf_r2 |
|-------------|-------|
| full        | {condition_wf_r2['full']:.4f} |
| e3_ablated  | {condition_wf_r2['e3_ablated']:.4f} |
| e2_ablated  | {condition_wf_r2['e2_ablated']:.4f} |

(E2 trains normally in all three conditions; wf_r2 should be similar across conditions.)

## Training Counts

| Condition   | approach events | contact events |
|-------------|----------------|----------------|
| full        | {tc_full.get('hazard_approach', 0)} | {tc_full.get('env_caused_hazard', 0) + tc_full.get('agent_caused_hazard', 0)} |
| e3_ablated  | {tc_e3.get('hazard_approach', 0)} | {tc_e3.get('env_caused_hazard', 0) + tc_e3.get('agent_caused_hazard', 0)} |
| e2_ablated  | {tc_e2.get('hazard_approach', 0)} | {tc_e2.get('env_caused_hazard', 0) + tc_e2.get('agent_caused_hazard', 0)} |

## PASS Criteria

| Criterion | Result | Value |
|-----------|--------|-------|
| C1: cal_gap_approach_full > e3_abl + 0.05  (E3 load-bearing for calibration) | {"PASS" if c1_pass else "FAIL"} | {cal_gap_approach_full:.4f} vs {cal_gap_approach_e3_abl:.4f} + 0.05 |
| C2: attribution_gap_full > e2_abl + 0.01   (E2 load-bearing for attribution) | {"PASS" if c2_pass else "FAIL"} | {attr_gap_full:.6f} vs {attr_gap_e2_abl:.6f} + 0.01 |
| C3: cal_gap_approach_full > 0.10           (full system calibration works)  | {"PASS" if c3_pass else "FAIL"} | {cal_gap_approach_full:.4f} |
| C4: attribution_gap_full > 0               (full attribution direction correct) | {"PASS" if c4_pass else "FAIL"} | {attr_gap_full:.6f} |
| C5: n_approach_eval >= 50                  (sufficient approach steps)       | {"PASS" if c5_pass else "FAIL"} | {n_approach_eval} |

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


# ------------------------------------------------------------------
# Entry point — explorer-launch pattern
# ------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import json
    from datetime import datetime, timezone

    parser = argparse.ArgumentParser(description="V3-EXQ-034: ARC-025 Three-Engine Irreducibility Ablation")
    parser.add_argument("--seed",            type=int,   default=0)
    parser.add_argument("--warmup",          type=int,   default=500,
                        help="Warmup episodes per condition")
    parser.add_argument("--eval-eps",        type=int,   default=50,
                        help="Eval episodes per condition")
    parser.add_argument("--steps",           type=int,   default=200,
                        help="Steps per episode")
    parser.add_argument("--alpha-world",     type=float, default=0.9,
                        help="EMA alpha for z_world (SD-008; must be >= 0.9)")
    parser.add_argument("--alpha-self",      type=float, default=0.3)
    parser.add_argument("--harm-scale",      type=float, default=0.02,
                        help="hazard_harm for CausalGridWorldV2")
    parser.add_argument("--proximity-scale", type=float, default=0.05,
                        help="proximity_harm_scale for CausalGridWorldV2")
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
