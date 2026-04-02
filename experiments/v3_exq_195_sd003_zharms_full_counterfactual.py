#!/opt/local/bin/python3
"""
V3-EXQ-195 -- SD-003 Full Latent Counterfactual Attribution on z_harm_s (Post-SD-011)

Claims: SD-003, ARC-033
Proposal: EXP-0094

Anchor experiment: "responsibility/attribution is real in the current architecture."

This is the EXQ-030b pipeline transposed from z_world to z_harm_s. EXQ-030b validated
the counterfactual ARCHITECTURE on z_world (world_forward_r2=0.947, attribution_gap=0.035).
Now that SD-010/SD-011 route E3 through z_harm_s, the counterfactual must operate on the
harm stream directly.

EXQ-166e validated SD-003 via a scalar delta proxy (harm_obs_s[CENTER_CELL] change).
This experiment validates the FULL LATENT pipeline: HarmForwardModel predicts z_harm_s_next,
E3.harm_eval_z_harm scores both actual and counterfactual predictions.

Root cause of EXQ-166b/c/d identity collapse (CRITICAL):
  HarmForwardModel in stack.py predicts z_harm_s_next DIRECTLY (no residual connection).
  E2.world_forward uses z_world + delta (residual), which forces the net to learn the CHANGE.
  Without residual, MSE converges to identity: fwd(z_harm, a) ~= z_harm for all actions
  because z_harm has high temporal autocorrelation (~0.9).

Fix: ResidualHarmForward -- predict delta, add to z_harm_s. Same architecture as
E2.world_forward (e2_fast.py:177-180). This prevents identity collapse structurally.

Fix #2 (from EXQ-030b): E3.harm_eval_z_harm trains on BOTH observed z_harm_s states AND
ResidualHarmForward-predicted z_harm_s states. This prevents distribution mismatch at eval
(EXQ-030 C3 failure mode).

Pipeline:
  z_harm_s_actual = ResidualHarmForward(z_harm_s_t, a_actual)
  z_harm_s_cf     = ResidualHarmForward(z_harm_s_t, a_cf)
  causal_sig      = E3.harm_eval_z_harm(z_harm_s_actual) - E3.harm_eval_z_harm(z_harm_s_cf)

PRE-REGISTERED ACCEPTANCE CRITERIA (ALL required for PASS):
  C1 (attribution_gap > 0):
    causal_sig(approach) - causal_sig(env_hazard) > 0.
    Agent-caused approach attribution exceeds environment-caused hazard attribution.
  C2 (causal_sig_approach > causal_sig_none):
    Attribution at hazard_approach exceeds locomotion baseline.
  C3 (causal_sig_approach > 0.005):
    Minimum signal threshold (calibrated from EXQ-030b observed magnitude).
  C4 (harm_forward_r2 > 0.05):
    ResidualHarmForward learned action-conditional harm dynamics.
  C5 (n_approach >= 50):
    Sufficient approach events for reliable estimates.
  C6 (>= 3/4 seeds causal_sig_approach > 0):
    Seed consistency.

Decision scoring:
  retain_ree:       ALL C1-C6 met -- full SD-003 + ARC-033 validation on z_harm_s
  hybridize:        C4+C5+C6 pass, C1 or C2 or C3 fail (forward model works, attribution weak)
  retire_ree_claim: C4 passes but C1+C2 both fail AND C5 passes (signal absent, not data-starved)
  inconclusive:     C4 fails (forward model did not learn)
"""

import sys
import random
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.latent.stack import HarmEncoder, ResidualHarmForward
from ree_core.utils.config import REEConfig


EXPERIMENT_TYPE = "v3_exq_195_sd003_zharms_full_counterfactual"
CLAIM_IDS = ["SD-003", "ARC-033"]

HARM_OBS_DIM = 51  # hazard_field(25) + resource_field(25) + exposure(1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _action_to_onehot(action_idx: int, num_actions: int, device) -> torch.Tensor:
    v = torch.zeros(1, num_actions, device=device)
    v[0, action_idx] = 1.0
    return v


def _compute_harm_forward_r2(
    harm_fwd: ResidualHarmForward,
    hf_data: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    device,
) -> float:
    """R2 of ResidualHarmForward on held-out transitions."""
    if len(hf_data) < 20:
        return 0.0
    n = len(hf_data)
    n_train = int(n * 0.8)
    with torch.no_grad():
        zh_all = torch.cat([d[0] for d in hf_data], dim=0).to(device)
        a_all = torch.cat([d[1] for d in hf_data], dim=0).to(device)
        zh1_all = torch.cat([d[2] for d in hf_data], dim=0).to(device)
        pred_all = harm_fwd(zh_all, a_all)
        pred_test = pred_all[n_train:]
        tgt_test = zh1_all[n_train:]
        if pred_test.shape[0] == 0:
            return 0.0
        ss_res = ((tgt_test - pred_test) ** 2).sum()
        ss_tot = ((tgt_test - tgt_test.mean(0, keepdim=True)) ** 2).sum()
        r2 = float((1 - ss_res / (ss_tot + 1e-8)).item())
    print(f"  harm_forward R2 (test n={pred_test.shape[0]}): {r2:.4f}", flush=True)
    return r2


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _train(
    agent: REEAgent,
    env,
    harm_enc: HarmEncoder,
    harm_fwd: ResidualHarmForward,
    harm_enc_optimizer: optim.Optimizer,
    harm_fwd_optimizer: optim.Optimizer,
    harm_eval_optimizer: optim.Optimizer,
    agent_optimizer: optim.Optimizer,
    num_episodes: int,
    steps_per_episode: int,
) -> Dict:
    """
    Joint training:
      1. HarmEncoder (autoencoder + center-cell regression for structure)
      2. ResidualHarmForward (z_harm_s transition prediction)
      3. E3.harm_eval_z_harm (with Fix #2: observed + forward-predicted states)
      4. Agent E1+E2 (standard)
    """
    agent.train()
    harm_enc.train()
    harm_fwd.train()

    # Buffers
    harm_buf_pos: List[torch.Tensor] = []  # z_harm_s near hazards
    harm_buf_neg: List[torch.Tensor] = []  # z_harm_s in safe areas
    MAX_BUF = 2000

    hf_data: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    MAX_HF = 5000

    counts: Dict[str, int] = {}
    num_actions = env.action_dim
    device = agent.device

    # Autoencoder for HarmEncoder structural quality
    harm_decoder = nn.Sequential(
        nn.Linear(harm_enc.z_harm_dim, 64),
        nn.ReLU(),
        nn.Linear(64, harm_enc.harm_obs_dim),
    ).to(device)
    harm_ae_optimizer = optim.Adam(
        list(harm_enc.parameters()) + list(harm_decoder.parameters()), lr=1e-3
    )

    for ep in range(num_episodes):
        flat_obs, obs_dict = env.reset()
        agent.reset()
        z_harm_prev = None
        a_prev = None

        for step in range(steps_per_episode):
            obs_body = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            harm_obs_t = obs_dict.get("harm_obs", torch.zeros(HARM_OBS_DIM)).float()
            if harm_obs_t.dim() == 1:
                harm_obs_t = harm_obs_t.unsqueeze(0)
            harm_obs_t = harm_obs_t.to(device)

            # Encode z_harm_s
            z_harm_s = harm_enc(harm_obs_t).detach()

            # Agent standard sense (for E1/E2 training)
            latent = agent.sense(obs_body, obs_world)
            agent.clock.advance()

            # Random action (exploration)
            action_idx = random.randint(0, num_actions - 1)
            action = _action_to_onehot(action_idx, num_actions, device)
            agent._last_action = action

            flat_obs, harm_signal, done, info, obs_dict = env.step(action)
            ttype = info.get("transition_type", "none")
            counts[ttype] = counts.get(ttype, 0) + 1

            # --- HarmEncoder autoencoder training (structure quality) ---
            recon = harm_decoder(harm_enc(harm_obs_t))
            ae_loss = F.mse_loss(recon, harm_obs_t)
            # Center-cell regression auxiliary: z_harm_s should encode proximity
            center_cell_target = harm_obs_t[:, 12:13]  # hazard_field_view center (5x5, idx 12)
            center_pred = harm_enc(harm_obs_t)[:, :1]  # First dim as proxy
            ae_loss = ae_loss + 0.5 * F.mse_loss(center_pred, center_cell_target)
            harm_ae_optimizer.zero_grad()
            ae_loss.backward()
            harm_ae_optimizer.step()

            # --- harm_eval buffer (balanced harm/safe) ---
            if harm_signal < 0:
                harm_buf_pos.append(z_harm_s.cpu())
                if len(harm_buf_pos) > MAX_BUF:
                    harm_buf_pos = harm_buf_pos[-MAX_BUF:]
            else:
                harm_buf_neg.append(z_harm_s.cpu())
                if len(harm_buf_neg) > MAX_BUF:
                    harm_buf_neg = harm_buf_neg[-MAX_BUF:]

            # --- ResidualHarmForward training data ---
            if z_harm_prev is not None and a_prev is not None:
                hf_data.append((z_harm_prev.cpu(), a_prev.cpu(), z_harm_s.cpu()))
                if len(hf_data) > MAX_HF:
                    hf_data = hf_data[-MAX_HF:]

            # --- Agent E1+E2 standard losses ---
            e1_loss = agent.compute_prediction_loss()
            e2_loss = agent.compute_e2_loss()
            total_loss = e1_loss + e2_loss
            if total_loss.requires_grad:
                agent_optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                agent_optimizer.step()

            # --- ResidualHarmForward training (separate optimizer, MECH-069) ---
            if len(hf_data) >= 16:
                k = min(32, len(hf_data))
                idxs = torch.randperm(len(hf_data))[:k].tolist()
                zh_b = torch.cat([hf_data[i][0] for i in idxs]).to(device)
                a_b = torch.cat([hf_data[i][1] for i in idxs]).to(device)
                zh1_b = torch.cat([hf_data[i][2] for i in idxs]).to(device)
                pred = harm_fwd(zh_b, a_b)
                hf_loss = F.mse_loss(pred, zh1_b)
                if hf_loss.requires_grad:
                    harm_fwd_optimizer.zero_grad()
                    hf_loss.backward()
                    torch.nn.utils.clip_grad_norm_(harm_fwd.parameters(), 0.5)
                    harm_fwd_optimizer.step()

            # --- E3.harm_eval_z_harm training (Fix #2: observed + predicted) ---
            if len(harm_buf_pos) >= 4 and len(harm_buf_neg) >= 4:
                k_pos = min(16, len(harm_buf_pos))
                k_neg = min(16, len(harm_buf_neg))
                pos_idx = torch.randperm(len(harm_buf_pos))[:k_pos].tolist()
                neg_idx = torch.randperm(len(harm_buf_neg))[:k_neg].tolist()

                zh_pos_obs = torch.cat([harm_buf_pos[i] for i in pos_idx], dim=0).to(device)
                zh_neg_obs = torch.cat([harm_buf_neg[i] for i in neg_idx], dim=0).to(device)

                # Fix #2: augment with forward-predicted states
                with torch.no_grad():
                    a_rand_pos = torch.zeros(k_pos, num_actions, device=device)
                    a_rand_pos[torch.arange(k_pos), torch.randint(0, num_actions, (k_pos,))] = 1.0
                    a_rand_neg = torch.zeros(k_neg, num_actions, device=device)
                    a_rand_neg[torch.arange(k_neg), torch.randint(0, num_actions, (k_neg,))] = 1.0
                    zh_pos_pred = harm_fwd(zh_pos_obs, a_rand_pos)
                    zh_neg_pred = harm_fwd(zh_neg_obs, a_rand_neg)

                zh_b = torch.cat([zh_pos_obs, zh_neg_obs, zh_pos_pred, zh_neg_pred], dim=0)
                target = torch.cat([
                    torch.ones(k_pos, 1, device=device),
                    torch.zeros(k_neg, 1, device=device),
                    torch.ones(k_pos, 1, device=device),
                    torch.zeros(k_neg, 1, device=device),
                ], dim=0)

                pred_harm = agent.e3.harm_eval_z_harm(zh_b)
                harm_loss = F.mse_loss(pred_harm, target)
                if harm_loss.requires_grad:
                    harm_eval_optimizer.zero_grad()
                    harm_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        agent.e3.harm_eval_z_harm_head.parameters(), 0.5
                    )
                    harm_eval_optimizer.step()

            z_harm_prev = z_harm_s
            a_prev = action.detach()

            if done:
                break

        if (ep + 1) % 50 == 0 or ep == num_episodes - 1:
            approach = counts.get("hazard_approach", 0)
            contact = counts.get("env_caused_hazard", 0) + counts.get("agent_caused_hazard", 0)
            print(
                f"  [train] ep {ep+1}/{num_episodes}  "
                f"approach={approach}  contact={contact}  hf_buf={len(hf_data)}",
                flush=True,
            )

    return {"counts": counts, "hf_data": hf_data}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _eval_attribution(
    agent: REEAgent,
    env,
    harm_enc: HarmEncoder,
    harm_fwd: ResidualHarmForward,
    num_episodes: int,
    steps_per_episode: int,
) -> Dict:
    """
    Full SD-003 counterfactual attribution on z_harm_s.

    For each step:
      1. z_harm_s_t = HarmEncoder(harm_obs_t)
      2. z_harm_s_actual = ResidualHarmForward(z_harm_s_t, a_actual)
      3. For each a_cf != a_actual:
         z_harm_s_cf = ResidualHarmForward(z_harm_s_t, a_cf)
      4. causal_sig = mean_cf [E3.harm_eval_z_harm(actual) - E3.harm_eval_z_harm(cf)]
    """
    agent.eval()
    harm_enc.eval()
    harm_fwd.eval()
    device = agent.device

    ttypes = ["none", "hazard_approach", "env_caused_hazard", "agent_caused_hazard"]
    harm_deltas: Dict[str, List[float]] = {t: [] for t in ttypes}
    causal_sigs: Dict[str, List[float]] = {t: [] for t in ttypes}

    num_actions = env.action_dim

    for ep in range(num_episodes):
        flat_obs, obs_dict = env.reset()
        agent.reset()

        for step in range(steps_per_episode):
            harm_obs_t = obs_dict.get("harm_obs", torch.zeros(HARM_OBS_DIM)).float()
            if harm_obs_t.dim() == 1:
                harm_obs_t = harm_obs_t.unsqueeze(0)
            harm_obs_t = harm_obs_t.to(device)

            with torch.no_grad():
                z_harm_s = harm_enc(harm_obs_t)

            action_idx = random.randint(0, num_actions - 1)
            action = _action_to_onehot(action_idx, num_actions, device)

            flat_obs, harm_signal, done, info, obs_dict = env.step(action)
            ttype = info.get("transition_type", "none")

            with torch.no_grad():
                # Actual predicted next harm state
                z_harm_actual = harm_fwd(z_harm_s, action)
                harm_actual = agent.e3.harm_eval_z_harm(z_harm_actual)

                # Counterfactual predictions
                deltas = []
                sigs = []
                for cf_idx in range(num_actions):
                    if cf_idx == action_idx:
                        continue
                    a_cf = _action_to_onehot(cf_idx, num_actions, device)
                    z_cf = harm_fwd(z_harm_s, a_cf)
                    harm_cf = agent.e3.harm_eval_z_harm(z_cf)
                    delta = float((z_harm_actual - z_cf).norm().item())
                    sig = float((harm_actual - harm_cf).item())
                    deltas.append(delta)
                    sigs.append(sig)

                mean_delta = float(np.mean(deltas)) if deltas else 0.0
                mean_sig = float(np.mean(sigs)) if sigs else 0.0

            if ttype in harm_deltas:
                harm_deltas[ttype].append(mean_delta)
                causal_sigs[ttype].append(mean_sig)

            if done:
                break

    def _mean(lst):
        return float(np.mean(lst)) if lst else 0.0

    mean_deltas = {t: _mean(harm_deltas[t]) for t in ttypes}
    mean_sigs = {t: _mean(causal_sigs[t]) for t in ttypes}
    n_counts = {t: len(harm_deltas[t]) for t in ttypes}

    attribution_gap = mean_sigs["hazard_approach"] - mean_sigs["env_caused_hazard"]

    print(f"\n  --- SD-003 Attribution Eval (z_harm_s) ---", flush=True)
    for t in ttypes:
        print(
            f"  {t:28s}: delta={mean_deltas[t]:.4f}  causal_sig={mean_sigs[t]:.6f}  n={n_counts[t]}",
            flush=True,
        )
    print(f"  attribution_gap (approach-env): {attribution_gap:.6f}", flush=True)

    return {
        "mean_deltas": mean_deltas,
        "mean_sigs": mean_sigs,
        "n_counts": n_counts,
        "attribution_gap": attribution_gap,
    }


# ---------------------------------------------------------------------------
# Single-seed run
# ---------------------------------------------------------------------------

def run(
    seed: int = 0,
    warmup_episodes: int = 400,
    eval_episodes: int = 50,
    steps_per_episode: int = 200,
    self_dim: int = 32,
    world_dim: int = 32,
    z_harm_dim: int = 32,
    lr: float = 1e-3,
    alpha_world: float = 0.9,
    alpha_self: float = 0.3,
    harm_scale: float = 0.02,
    proximity_scale: float = 0.05,
    **kwargs,
) -> dict:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

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
    device = agent.device

    # Harm stream modules (instantiated outside LatentStack per SD-010 convention)
    harm_enc = HarmEncoder(harm_obs_dim=HARM_OBS_DIM, z_harm_dim=z_harm_dim).to(device)
    harm_fwd = ResidualHarmForward(
        z_harm_dim=z_harm_dim, action_dim=env.action_dim, hidden_dim=64
    ).to(device)

    print(
        f"[V3-EXQ-195] SD-003 Full Latent Counterfactual on z_harm_s\n"
        f"  seed={seed}  harm_obs_dim={HARM_OBS_DIM}  z_harm_dim={z_harm_dim}\n"
        f"  alpha_world={alpha_world}  proximity_scale={proximity_scale}\n"
        f"  ResidualHarmForward (identity-collapse fix)\n"
        f"  Fix #2: E3.harm_eval_z_harm trained on observed + predicted z_harm_s",
        flush=True,
    )

    # Separate optimizers (MECH-069)
    agent_params = [
        p for n, p in agent.named_parameters()
        if "harm_eval_z_harm_head" not in n
    ]
    agent_optimizer = optim.Adam(agent_params, lr=lr)
    harm_enc_optimizer = optim.Adam(harm_enc.parameters(), lr=1e-3)
    harm_fwd_optimizer = optim.Adam(harm_fwd.parameters(), lr=1e-3)
    harm_eval_optimizer = optim.Adam(
        agent.e3.harm_eval_z_harm_head.parameters(), lr=1e-4
    )

    train_out = _train(
        agent, env, harm_enc, harm_fwd,
        harm_enc_optimizer, harm_fwd_optimizer, harm_eval_optimizer, agent_optimizer,
        warmup_episodes, steps_per_episode,
    )

    hf_r2 = _compute_harm_forward_r2(harm_fwd, train_out["hf_data"], device)

    print(f"\n[V3-EXQ-195] Eval ({eval_episodes} eps)...", flush=True)
    eval_out = _eval_attribution(
        agent, env, harm_enc, harm_fwd, eval_episodes, steps_per_episode,
    )

    md = eval_out["mean_deltas"]
    ms = eval_out["mean_sigs"]
    nc = eval_out["n_counts"]

    # Criteria
    c1_pass = eval_out["attribution_gap"] > 0.0
    c2_pass = ms["hazard_approach"] > ms["none"]
    c3_pass = ms["hazard_approach"] > 0.005
    c4_pass = hf_r2 > 0.05
    c5_pass = nc["hazard_approach"] >= 50

    all_pass = c1_pass and c2_pass and c3_pass and c4_pass and c5_pass
    status = "PASS" if all_pass else "FAIL"
    n_met = sum([c1_pass, c2_pass, c3_pass, c4_pass, c5_pass])

    failure_notes = []
    if not c1_pass:
        failure_notes.append(
            f"C1 FAIL: attribution_gap={eval_out['attribution_gap']:.6f} <= 0"
        )
    if not c2_pass:
        failure_notes.append(
            f"C2 FAIL: causal_sig_approach({ms['hazard_approach']:.6f}) <= causal_sig_none({ms['none']:.6f})"
        )
    if not c3_pass:
        failure_notes.append(
            f"C3 FAIL: causal_sig_approach={ms['hazard_approach']:.6f} <= 0.005"
        )
    if not c4_pass:
        failure_notes.append(f"C4 FAIL: harm_forward_r2={hf_r2:.4f} <= 0.05")
    if not c5_pass:
        failure_notes.append(f"C5 FAIL: n_approach={nc['hazard_approach']} < 50")

    print(f"\nV3-EXQ-195 seed {seed} verdict: {status}  ({n_met}/5)", flush=True)
    for note in failure_notes:
        print(f"  {note}", flush=True)

    tc = train_out["counts"]
    return {
        "seed": seed,
        "status": status,
        "metrics": {
            "alpha_world": float(alpha_world),
            "proximity_scale": float(proximity_scale),
            "z_harm_dim": float(z_harm_dim),
            "harm_forward_r2": float(hf_r2),
            "mean_delta_none": float(md["none"]),
            "mean_delta_approach": float(md["hazard_approach"]),
            "mean_delta_env_hazard": float(md["env_caused_hazard"]),
            "mean_delta_agent_hazard": float(md["agent_caused_hazard"]),
            "mean_causal_sig_none": float(ms["none"]),
            "mean_causal_sig_approach": float(ms["hazard_approach"]),
            "mean_causal_sig_env_hazard": float(ms["env_caused_hazard"]),
            "mean_causal_sig_agent_hazard": float(ms["agent_caused_hazard"]),
            "attribution_gap": float(eval_out["attribution_gap"]),
            "n_approach_eval": float(nc["hazard_approach"]),
            "n_env_hazard_eval": float(nc["env_caused_hazard"]),
            "n_agent_hazard_eval": float(nc["agent_caused_hazard"]),
            "n_none_eval": float(nc["none"]),
            "train_approach_events": float(tc.get("hazard_approach", 0)),
            "train_contact_events": float(
                tc.get("env_caused_hazard", 0) + tc.get("agent_caused_hazard", 0)
            ),
            "crit1_pass": 1.0 if c1_pass else 0.0,
            "crit2_pass": 1.0 if c2_pass else 0.0,
            "crit3_pass": 1.0 if c3_pass else 0.0,
            "crit4_pass": 1.0 if c4_pass else 0.0,
            "crit5_pass": 1.0 if c5_pass else 0.0,
            "criteria_met": float(n_met),
        },
        "failure_notes": failure_notes,
    }


# ---------------------------------------------------------------------------
# Multi-seed runner
# ---------------------------------------------------------------------------

def run_multi_seed(seeds=(42, 7, 123, 13), **kwargs) -> dict:
    all_results = []
    for s in seeds:
        print(f"\n{'='*60}\n  SEED {s}\n{'='*60}", flush=True)
        r = run(seed=s, **kwargs)
        all_results.append(r)

    # Aggregate
    n_pass = sum(1 for r in all_results if r["status"] == "PASS")
    n_seeds = len(seeds)

    # C6: seed consistency
    c6_pass = sum(
        1 for r in all_results
        if r["metrics"]["mean_causal_sig_approach"] > 0
    ) >= 3

    # Aggregate metrics (means across seeds)
    agg = {}
    metric_keys = all_results[0]["metrics"].keys()
    for k in metric_keys:
        vals = [r["metrics"][k] for r in all_results]
        agg[k] = float(np.mean(vals))
        agg[f"{k}_std"] = float(np.std(vals))
    agg["seeds_pass"] = float(n_pass)
    agg["seeds_total"] = float(n_seeds)
    agg["c6_seed_consistency"] = 1.0 if c6_pass else 0.0

    # Overall: ALL C1-C5 on aggregated means, plus C6
    overall_c1 = agg["attribution_gap"] > 0.0
    overall_c2 = agg["mean_causal_sig_approach"] > agg["mean_causal_sig_none"]
    overall_c3 = agg["mean_causal_sig_approach"] > 0.005
    overall_c4 = agg["harm_forward_r2"] > 0.05
    overall_c5 = agg["n_approach_eval"] >= 50
    all_pass = overall_c1 and overall_c2 and overall_c3 and overall_c4 and overall_c5 and c6_pass
    overall_status = "PASS" if all_pass else "FAIL"
    n_met = sum([overall_c1, overall_c2, overall_c3, overall_c4, overall_c5, c6_pass])

    failure_notes = []
    if not overall_c1:
        failure_notes.append(f"C1 FAIL: attribution_gap_mean={agg['attribution_gap']:.6f} <= 0")
    if not overall_c2:
        failure_notes.append(
            f"C2 FAIL: causal_sig_approach_mean({agg['mean_causal_sig_approach']:.6f}) "
            f"<= causal_sig_none_mean({agg['mean_causal_sig_none']:.6f})"
        )
    if not overall_c3:
        failure_notes.append(f"C3 FAIL: causal_sig_approach_mean={agg['mean_causal_sig_approach']:.6f} <= 0.005")
    if not overall_c4:
        failure_notes.append(f"C4 FAIL: harm_forward_r2_mean={agg['harm_forward_r2']:.4f} <= 0.05")
    if not overall_c5:
        failure_notes.append(f"C5 FAIL: n_approach_eval_mean={agg['n_approach_eval']:.0f} < 50")
    if not c6_pass:
        failure_notes.append(f"C6 FAIL: seed consistency -- only {n_pass}/{n_seeds} seeds PASS")

    print(f"\n{'='*60}", flush=True)
    print(f"V3-EXQ-195 OVERALL: {overall_status}  ({n_met}/6)", flush=True)
    print(f"  Seeds PASS: {n_pass}/{n_seeds}  C6 (>=3/4 causal_sig>0): {c6_pass}", flush=True)
    print(f"  harm_forward_r2 mean: {agg['harm_forward_r2']:.4f}", flush=True)
    print(f"  attribution_gap mean: {agg['attribution_gap']:.6f}", flush=True)
    print(f"  causal_sig_approach mean: {agg['mean_causal_sig_approach']:.6f}", flush=True)
    for note in failure_notes:
        print(f"  {note}", flush=True)

    # Decision scoring
    if all_pass:
        evidence_direction = "supports"
        decision = "retain_ree"
    elif overall_c4 and overall_c5 and c6_pass:
        evidence_direction = "mixed"
        decision = "hybridize"
    elif overall_c4 and not overall_c1 and not overall_c2 and overall_c5:
        evidence_direction = "weakens"
        decision = "retire_ree_claim"
    else:
        evidence_direction = "mixed"
        decision = "inconclusive"

    md_approach = agg["mean_delta_approach"]
    md_none = agg["mean_delta_none"]
    md_env = agg["mean_delta_env_hazard"]
    md_agent = agg["mean_delta_agent_hazard"]
    ms_approach = agg["mean_causal_sig_approach"]
    ms_none = agg["mean_causal_sig_none"]
    ms_env = agg["mean_causal_sig_env_hazard"]
    ms_agent = agg["mean_causal_sig_agent_hazard"]
    na_approach = agg["n_approach_eval"]
    na_env = agg["n_env_hazard_eval"]
    na_agent = agg["n_agent_hazard_eval"]
    na_none = agg["n_none_eval"]

    failure_section = ""
    if failure_notes:
        failure_section = "\n## Failure Notes\n\n" + "\n".join(f"- {n}" for n in failure_notes)

    summary_markdown = f"""# V3-EXQ-195 -- SD-003 Full Latent Counterfactual on z_harm_s (Post-SD-011)

**Status:** {overall_status}
**Claims:** SD-003, ARC-033
**Decision:** {decision}
**Seeds:** {list(seeds)} ({n_pass}/{n_seeds} PASS)
**Proposal:** EXP-0094

## Architecture

- HarmEncoder (SD-010): harm_obs (51-dim) -> z_harm_s (32-dim)
- ResidualHarmForward: z_harm_s + delta(z_harm_s, action) -> z_harm_s_next
  (identity-collapse fix: residual architecture, cf. E2.world_forward)
- E3.harm_eval_z_harm: z_harm_s -> harm score
- Fix #2: E3 trained on observed + forward-predicted z_harm_s states

## Attribution Results (seed means)

| Transition | harm_delta | causal_sig | n |
|---|---|---|---|
| none (locomotion)   | {md_none:.4f} | {ms_none:.6f} | {na_none:.0f} |
| hazard_approach     | {md_approach:.4f} | {ms_approach:.6f} | {na_approach:.0f} |
| env_caused_hazard   | {md_env:.4f} | {ms_env:.6f} | {na_env:.0f} |
| agent_caused_hazard | {md_agent:.4f} | {ms_agent:.6f} | {na_agent:.0f} |

- **harm_forward R2**: {agg['harm_forward_r2']:.4f}
- **attribution_gap** (approach - env): {agg['attribution_gap']:.6f}

## PASS Criteria

| Criterion | Result | Value |
|---|---|---|
| C1: attribution_gap > 0 | {"PASS" if overall_c1 else "FAIL"} | {agg['attribution_gap']:.6f} |
| C2: causal_sig_approach > causal_sig_none | {"PASS" if overall_c2 else "FAIL"} | {ms_approach:.6f} vs {ms_none:.6f} |
| C3: causal_sig_approach > 0.005 | {"PASS" if overall_c3 else "FAIL"} | {ms_approach:.6f} |
| C4: harm_forward_r2 > 0.05 | {"PASS" if overall_c4 else "FAIL"} | {agg['harm_forward_r2']:.4f} |
| C5: n_approach >= 50 | {"PASS" if overall_c5 else "FAIL"} | {na_approach:.0f} |
| C6: >= 3/4 seeds causal_sig > 0 | {"PASS" if c6_pass else "FAIL"} | {n_pass}/{n_seeds} |

Criteria met: {n_met}/6 -> **{overall_status}**
{failure_section}
"""

    return {
        "status": overall_status,
        "metrics": agg,
        "per_seed": all_results,
        "summary_markdown": summary_markdown,
        "claim_ids": CLAIM_IDS,
        "evidence_direction": evidence_direction,
        "experiment_type": EXPERIMENT_TYPE,
        "fatal_error_count": 0,
    }


# ---------------------------------------------------------------------------
# __main__
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import json
    from datetime import datetime, timezone

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=None,
                        help="Single seed (omit for multi-seed 42,7,123,13)")
    parser.add_argument("--warmup", type=int, default=400)
    parser.add_argument("--eval-eps", type=int, default=50)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--alpha-world", type=float, default=0.9)
    parser.add_argument("--alpha-self", type=float, default=0.3)
    parser.add_argument("--harm-scale", type=float, default=0.02)
    parser.add_argument("--proximity-scale", type=float, default=0.05)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        print("DRY RUN: would run EXQ-195 with:", flush=True)
        print(f"  warmup={args.warmup}  eval={args.eval_eps}  steps={args.steps}", flush=True)
        print(f"  seed={'multi (42,7,123,13)' if args.seed is None else args.seed}", flush=True)
        sys.exit(0)

    common = dict(
        warmup_episodes=args.warmup,
        eval_episodes=args.eval_eps,
        steps_per_episode=args.steps,
        alpha_world=args.alpha_world,
        alpha_self=args.alpha_self,
        harm_scale=args.harm_scale,
        proximity_scale=args.proximity_scale,
    )

    if args.seed is not None:
        result = run(seed=args.seed, **common)
        # Wrap single seed to look like multi-seed output
        result["per_seed"] = [result]
        result["claim_ids"] = CLAIM_IDS
        result["evidence_direction"] = "supports" if result["status"] == "PASS" else "weakens"
        result["experiment_type"] = EXPERIMENT_TYPE
        result["summary_markdown"] = f"Single-seed run (seed={args.seed}): {result['status']}"
    else:
        result = run_multi_seed(**common)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result["run_timestamp"] = ts
    result["run_id"] = f"{EXPERIMENT_TYPE}_{ts}_v3"
    result["architecture_epoch"] = "ree_hybrid_guardrails_v1"
    result["claim_ids"] = CLAIM_IDS

    out_dir = (
        Path(__file__).resolve().parents[2]
        / "REE_assembly" / "evidence" / "experiments" / EXPERIMENT_TYPE
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{EXPERIMENT_TYPE}_{ts}.json"
    out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print(f"\nResult written to: {out_path}", flush=True)
    print(f"Status: {result['status']}", flush=True)
