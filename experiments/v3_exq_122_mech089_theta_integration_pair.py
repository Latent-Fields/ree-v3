#!/opt/local/bin/python3
"""
V3-EXQ-122 -- MECH-089: Theta-Gamma Integration Discriminative Pair (EXP-0020 / EVB-0016)

Claim: MECH-089
Proposal: EXP-0020 / EVB-0016
Dispatch mode: discriminative_pair
Min shared seeds: 2

MECH-089 asserts: "E1 updates are batched into theta-cycle summaries before reaching E3
(cross-frequency temporal packaging)."

Specifically:
  Fast loop (E1-rate) outputs are rolled up into a ThetaBuffer before E3 samples them.
  E3 receives the rolling mean of the last theta_k E1 z_world frames rather than the
  raw most-recent z_world frame. This temporal integration is functionally necessary:
  it reduces within-cycle noise so that E3's harm evaluation is based on a stable
  context estimate rather than a single noisy frame.

This experiment implements a clean discriminative pair by MATCHING E3 update rates
across conditions and varying ONLY whether the ThetaBuffer mean or the raw latest
z_world frame is delivered to E3:

  THETA_INTEGRATION_ON:
    theta_k = 2  (E3 ticks every 2 steps; receives mean of last 2 z_world frames)
    E3 samples ThetaBuffer.summary() -- rolling mean of the window
    The buffer contains genuine signal variation (non-trivially flat z_world)

  THETA_INTEGRATION_ABLATED:
    theta_k = 2  (E3 ticks every 2 steps -- SAME update rate as ON condition)
    E3 samples the raw most-recent z_world frame directly (no averaging)
    Isolation: E3 update rate is identical; only the content changes

Both conditions:
  - theta_k = 2  (shorter than EXQ-066's k=4, which was too coarse per evidence note)
  - E1 ticks every step (e1_steps_per_tick = 1) in both conditions
  - E3 ticks every 2 steps (e3_steps_per_tick = 2) in both conditions
  - Harm eval head trained with balanced buffer (harm / no-harm)
  - 2 matched seeds: [42, 123]
  - 400 warmup + 50 eval episodes x 200 steps
  - CausalGridWorldV2: size=6, 4 hazards, 3 resources
  - SD-008: alpha_world=0.9
  - SD-005 split latent: z_self != z_world (unified_latent_mode=False)

CLAIM_IDS RATIONALE:
  The only claim directly tested is MECH-089: does theta-cycle averaging of E1
  z_world output improve E3 harm evaluation AUC relative to raw same-rate sampling?
  MECH-089 directly predicts YES (temporal batching is necessary for stable E3 input).
  ARC-023 (thalamic heartbeat rates) is NOT tagged -- that tests multi-rate execution
  architecture, not the averaging benefit per se.

  Lesson from EXQ-066: theta_k=4 was too coarse and DEGRADED E3 performance
  (batched error 2.28x worse than raw). The redesign uses theta_k=2 and matches
  E3 update rates between conditions to cleanly isolate the averaging effect.
  The metric is harm prediction AUC (discriminative) rather than prediction error
  variance (EXQ-066 metric), providing a cleaner binary outcome.

PRE-REGISTERED THRESHOLDS (hardcoded, not inferred post-hoc):

  C1 (relative harm AUC advantage):
    harm_auc_ON >= harm_auc_ABLATED + THRESH_C1_AUC_DELTA
    Theta-averaging must provide meaningful AUC benefit over raw same-rate sampling.
    Threshold: THRESH_C1_AUC_DELTA = 0.05

  C2 (absolute harm AUC learning):
    harm_auc_ON >= THRESH_C2_AUC_ABS
    ON condition must achieve reliable above-chance harm prediction.
    Threshold: THRESH_C2_AUC_ABS = 0.60

  C3 (consistency across seeds):
    harm_auc_ON > harm_auc_ABLATED for BOTH seeds independently.
    Direction must replicate.

  C4 (data quality -- sufficient harm events):
    n_harm_steps_eval >= THRESH_C4_MIN_HARM per seed per condition.
    Threshold: THRESH_C4_MIN_HARM = 20

  C5 (buffer non-triviality):
    mean_within_batch_var_ON >= THRESH_C5_BATCH_VAR
    The ThetaBuffer must contain non-trivially uniform z_world (confirms
    there is genuine signal to average, not just a flat series).
    Threshold: THRESH_C5_BATCH_VAR = 1e-5  (diagnostic only, not required for PASS)

PASS criteria:
  C1 AND C2 AND C3 AND C4 -> PASS -> supports MECH-089
  C1 AND C3, NOT C2       -> mixed (relative advantage without absolute learning)
  NOT C1 AND C2 AND C4    -> FAIL -> theta averaging hypothesis refuted at theta_k=2
  NOT C4                  -> inconclusive (data quality failure)

Decision mapping:
  PASS            -> retain_ree
  C1+C3+C4        -> hybridize
  NOT C1 AND C4   -> retire_ree_claim (theta averaging unhelpful at this scale)
  NOT C4          -> inconclusive
"""

import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig


EXPERIMENT_TYPE = "v3_exq_122_mech089_theta_integration_pair"
CLAIM_IDS = ["MECH-089"]

# Pre-registered thresholds
THETA_K = 2                       # theta-to-gamma ratio (smaller than EXQ-066's k=4)
THRESH_C1_AUC_DELTA   = 0.05      # ON must beat ABLATED by >= 5pp harm AUC
THRESH_C2_AUC_ABS     = 0.60      # ON must reach >= 0.60 absolute harm AUC
THRESH_C4_MIN_HARM    = 20        # >= 20 harm steps in eval per (seed, condition)
THRESH_C5_BATCH_VAR   = 1e-5      # within-batch z_world variance (diagnostic only)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _action_to_onehot(action_idx: int, n: int, device) -> torch.Tensor:
    v = torch.zeros(1, n, device=device)
    v[0, action_idx] = 1.0
    return v


def _ensure_2d(t: torch.Tensor) -> torch.Tensor:
    return t.unsqueeze(0) if t.dim() == 1 else t


def _mean_safe(lst: List[float]) -> float:
    return float(sum(lst) / len(lst)) if lst else 0.0


def _compute_auc(scores: List[float], labels: List[float]) -> float:
    """Rank-based (Wilcoxon-Mann-Whitney) AUC. Returns 0.5 on degenerate input."""
    if not scores:
        return 0.5
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    pairs = sorted(zip(scores, labels), key=lambda x: x[0])
    auc = 0.0
    running_neg = 0.0
    for _, label in pairs:
        if label == 0.0:
            running_neg += 1.0
        else:
            auc += running_neg
    return auc / (n_pos * n_neg)


def _build_config(
    env: CausalGridWorldV2,
    self_dim: int,
    world_dim: int,
    alpha_world: float,
    alpha_self: float,
) -> REEConfig:
    config = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=self_dim,
        world_dim=world_dim,
        alpha_world=alpha_world,
        alpha_self=alpha_self,
        reafference_action_dim=0,  # SD-007 disabled -- isolate MECH-089 mechanism
    )
    # SD-005: split latent mode (z_self != z_world)
    config.latent.unified_latent_mode = False
    # Both conditions: E1 every step, E3 every THETA_K steps, E2 every step
    config.heartbeat.e1_steps_per_tick = 1
    config.heartbeat.e2_steps_per_tick = 1
    config.heartbeat.e3_steps_per_tick = THETA_K
    # ThetaBuffer size matches THETA_K (changed per condition below)
    config.heartbeat.theta_buffer_size = THETA_K
    # Disable MECH-093 beta-modulated rate to keep conditions clean
    config.heartbeat.beta_rate_min_steps = THETA_K
    config.heartbeat.beta_rate_max_steps = THETA_K
    return config


# ---------------------------------------------------------------------------
# Single cell runner
# ---------------------------------------------------------------------------

def _run_single(
    seed: int,
    theta_on: bool,
    warmup_episodes: int,
    eval_episodes: int,
    steps_per_episode: int,
    self_dim: int,
    world_dim: int,
    lr: float,
    alpha_world: float,
    alpha_self: float,
    harm_scale: float,
    proximity_harm_scale: float,
    env_drift_prob: float,
    env_drift_interval: int,
    dry_run: bool = False,
) -> Dict:
    """Run one (seed, condition) cell. Returns per-cell metrics."""
    torch.manual_seed(seed)
    random.seed(seed)

    cond = "THETA_INTEGRATION_ON" if theta_on else "THETA_INTEGRATION_ABLATED"

    env = CausalGridWorldV2(
        seed=seed,
        size=6,
        num_hazards=4,
        num_resources=3,
        hazard_harm=harm_scale,
        env_drift_interval=env_drift_interval,
        env_drift_prob=env_drift_prob,
        proximity_harm_scale=proximity_harm_scale,
        proximity_benefit_scale=proximity_harm_scale * 0.6,
        proximity_approach_threshold=0.15,
        hazard_field_decay=0.5,
        use_proxy_fields=True,
    )

    config = _build_config(env, self_dim, world_dim, alpha_world, alpha_self)
    agent = REEAgent(config)
    device = agent.device

    # Optimizers: E1 + E2 world forward + E3 harm eval
    e1_opt = optim.Adam(agent.e1.parameters(), lr=lr)

    # E2 world-forward optimizer
    e2_wf_params = (
        list(agent.e2.world_transition.parameters())
        + list(agent.e2.world_action_encoder.parameters())
    )
    e2_opt = optim.Adam(e2_wf_params, lr=lr)

    # E3 harm eval optimizer
    e3_harm_opt = optim.Adam(agent.e3.harm_eval_head.parameters(), lr=1e-4)

    # Replay buffers
    wf_buf: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    harm_buf_pos: List[torch.Tensor] = []
    harm_buf_neg: List[torch.Tensor] = []

    actual_warmup = min(3, warmup_episodes) if dry_run else warmup_episodes
    actual_eval   = min(2, eval_episodes)   if dry_run else eval_episodes

    within_batch_vars: List[float] = []   # for C5 batch non-triviality check
    current_window_z: List[torch.Tensor] = []
    train_harm_steps = 0

    # ---- TRAINING PHASE ----
    agent.train()

    for ep in range(actual_warmup):
        _, obs_dict = env.reset()
        agent.reset()
        current_window_z.clear()
        z_world_prev: Optional[torch.Tensor] = None
        action_prev: Optional[torch.Tensor] = None

        for _ in range(steps_per_episode):
            obs_body  = obs_dict["body_state"].to(device)
            obs_world = obs_dict["world_state"].to(device)

            latent = agent.sense(obs_body, obs_world)
            z_world_curr = _ensure_2d(latent.z_world.detach())
            current_window_z.append(z_world_curr.clone())

            ticks = agent.clock.advance()

            action_idx = random.randint(0, env.action_dim - 1)
            action_oh  = _action_to_onehot(action_idx, env.action_dim, device)
            agent._last_action = action_oh

            # Determine what E3 sees: theta average vs raw latest
            if ticks.get("e3_tick", False):
                if theta_on:
                    # ON: E3 receives ThetaBuffer mean (genuine theta integration)
                    e3_input = agent.theta_buffer.summary()
                else:
                    # ABLATED: E3 receives raw most-recent z_world (no averaging)
                    e3_input = z_world_curr.detach()

                # Record within-batch variance (for C5 diagnostic, ON condition)
                if theta_on and len(current_window_z) >= 2:
                    window_stack = torch.stack(current_window_z, dim=0)  # [T, 1, world_dim]
                    window_var = float(window_stack.var(dim=0).mean().item())
                    within_batch_vars.append(window_var)
                current_window_z.clear()

                # Harm eval training buffer update
                _, harm_signal, done, info, obs_dict = env.step(action_oh)
                train_harm_steps += (1 if float(harm_signal) < 0 else 0)

                if float(harm_signal) < 0:
                    harm_buf_pos.append(e3_input.detach())
                    if len(harm_buf_pos) > 1000:
                        harm_buf_pos = harm_buf_pos[-1000:]
                else:
                    harm_buf_neg.append(e3_input.detach())
                    if len(harm_buf_neg) > 1000:
                        harm_buf_neg = harm_buf_neg[-1000:]
            else:
                _, harm_signal, done, info, obs_dict = env.step(action_oh)

            # World-forward buffer (for E2 training)
            if z_world_prev is not None and action_prev is not None:
                wf_buf.append((z_world_prev.cpu(), action_prev.cpu(), z_world_curr.cpu()))
                if len(wf_buf) > 2000:
                    wf_buf = wf_buf[-2000:]

            # E1 loss
            e1_loss = agent.compute_prediction_loss()
            if e1_loss.requires_grad:
                e1_opt.zero_grad()
                e1_loss.backward()
                nn.utils.clip_grad_norm_(agent.e1.parameters(), 1.0)
                e1_opt.step()

            # E2 world-forward loss
            if len(wf_buf) >= 16:
                k_batch = min(32, len(wf_buf))
                idxs = torch.randperm(len(wf_buf))[:k_batch].tolist()
                zw_b  = torch.cat([wf_buf[i][0] for i in idxs]).to(device)
                a_b   = torch.cat([wf_buf[i][1] for i in idxs]).to(device)
                zw1_b = torch.cat([wf_buf[i][2] for i in idxs]).to(device)
                wf_loss = F.mse_loss(agent.e2.world_forward(zw_b, a_b), zw1_b)
                if wf_loss.requires_grad:
                    e2_opt.zero_grad()
                    wf_loss.backward()
                    nn.utils.clip_grad_norm_(e2_wf_params, 1.0)
                    e2_opt.step()

            # E3 harm eval loss (balanced mini-batch)
            if len(harm_buf_pos) >= 4 and len(harm_buf_neg) >= 4:
                k_p = min(16, len(harm_buf_pos))
                k_n = min(16, len(harm_buf_neg))
                pi = torch.randperm(len(harm_buf_pos))[:k_p].tolist()
                ni = torch.randperm(len(harm_buf_neg))[:k_n].tolist()
                zw_b = torch.cat(
                    [harm_buf_pos[i] for i in pi] + [harm_buf_neg[i] for i in ni], dim=0
                )
                target = torch.cat([
                    torch.ones(k_p, 1, device=device),
                    torch.zeros(k_n, 1, device=device),
                ], dim=0)
                pred = agent.e3.harm_eval(zw_b)
                harm_loss = F.mse_loss(pred, target)
                if harm_loss.requires_grad:
                    e3_harm_opt.zero_grad()
                    harm_loss.backward()
                    nn.utils.clip_grad_norm_(agent.e3.harm_eval_head.parameters(), 0.5)
                    e3_harm_opt.step()

            z_world_prev = z_world_curr.detach()
            action_prev  = action_oh.detach()

            if done:
                current_window_z.clear()
                break

        if (ep + 1) % 100 == 0 or ep == actual_warmup - 1:
            print(
                f"  [train] seed={seed} cond={cond}"
                f" ep {ep + 1}/{actual_warmup}"
                f" harm_buf_pos={len(harm_buf_pos)}"
                f" harm_buf_neg={len(harm_buf_neg)}"
                f" train_harm_steps={train_harm_steps}",
                flush=True,
            )

    # ---- EVAL PHASE ----
    agent.eval()
    harm_scores: List[float] = []   # E3 harm_eval output per step
    harm_labels: List[float] = []   # 1.0 = harm, 0.0 = no-harm
    harm_steps_eval = 0
    total_eval_steps = 0
    eval_within_batch_vars: List[float] = []
    eval_window_z: List[torch.Tensor] = []

    for _ in range(actual_eval):
        _, obs_dict = env.reset()
        agent.reset()
        eval_window_z.clear()
        z_world_prev = None
        action_prev  = None

        for _ in range(steps_per_episode):
            obs_body  = obs_dict["body_state"].to(device)
            obs_world = obs_dict["world_state"].to(device)

            with torch.no_grad():
                latent = agent.sense(obs_body, obs_world)
                z_world_curr = _ensure_2d(latent.z_world.detach())
                eval_window_z.append(z_world_curr.clone())

                ticks = agent.clock.advance()

                action_idx = random.randint(0, env.action_dim - 1)
                action_oh  = _action_to_onehot(action_idx, env.action_dim, device)
                agent._last_action = action_oh

                # E3 input: theta average vs raw
                if ticks.get("e3_tick", False):
                    if theta_on:
                        e3_input = agent.theta_buffer.summary()
                    else:
                        e3_input = z_world_curr.detach()

                    # C5: within-batch variance (ON condition only)
                    if theta_on and len(eval_window_z) >= 2:
                        window_stack = torch.stack(eval_window_z, dim=0)
                        window_var = float(window_stack.var(dim=0).mean().item())
                        eval_within_batch_vars.append(window_var)
                    eval_window_z.clear()

                    # Record harm prediction at E3 tick
                    harm_score = float(agent.e3.harm_eval(e3_input).item())
                    harm_scores.append(harm_score)
                    # Label filled after step below
                    harm_labels.append(None)  # placeholder
                else:
                    harm_score = None

            _, harm_signal, done, info, obs_dict = env.step(action_oh)
            total_eval_steps += 1
            if float(harm_signal) < 0:
                harm_steps_eval += 1

            # Fill the label for the E3-tick step
            if harm_score is not None:
                harm_labels[-1] = 1.0 if float(harm_signal) < 0 else 0.0

            z_world_prev = z_world_curr.detach()
            action_prev  = action_oh.detach()

            if done:
                eval_window_z.clear()
                break

    # Remove any placeholder None labels (in case e3_tick fired on the last step)
    valid_pairs = [(s, l) for s, l in zip(harm_scores, harm_labels) if l is not None]
    if valid_pairs:
        scores_clean, labels_clean = zip(*valid_pairs)
        scores_clean = list(scores_clean)
        labels_clean = list(labels_clean)
    else:
        scores_clean = []
        labels_clean = []

    harm_auc = _compute_auc(scores_clean, labels_clean)
    mean_batch_var = _mean_safe(within_batch_vars + eval_within_batch_vars) if theta_on else 0.0
    harm_rate = harm_steps_eval / max(1, total_eval_steps)

    print(
        f"  [eval] seed={seed} cond={cond}"
        f" harm_auc={harm_auc:.4f}"
        f" harm_steps_eval={harm_steps_eval}"
        f" total_eval_steps={total_eval_steps}"
        f" harm_rate={harm_rate:.4f}"
        f" n_e3_ticks_eval={len(scores_clean)}"
        f" mean_batch_var={mean_batch_var:.6f}",
        flush=True,
    )

    return {
        "seed": seed,
        "condition": cond,
        "theta_on": theta_on,
        "harm_auc": float(harm_auc),
        "harm_rate": float(harm_rate),
        "harm_steps_eval": int(harm_steps_eval),
        "total_eval_steps": int(total_eval_steps),
        "n_e3_ticks_eval": int(len(scores_clean)),
        "mean_within_batch_var": float(mean_batch_var),
        "train_harm_steps": int(train_harm_steps),
        "harm_buf_pos_final": int(len(harm_buf_pos)),
        "harm_buf_neg_final": int(len(harm_buf_neg)),
    }


# ---------------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------------

def run(
    seeds: Tuple = (42, 123),
    warmup_episodes: int = 400,
    eval_episodes: int = 50,
    steps_per_episode: int = 200,
    self_dim: int = 32,
    world_dim: int = 32,
    lr: float = 1e-3,
    alpha_world: float = 0.9,
    alpha_self: float = 0.3,
    harm_scale: float = 0.02,
    proximity_harm_scale: float = 0.01,
    env_drift_prob: float = 0.3,
    env_drift_interval: int = 3,
    dry_run: bool = False,
    **kwargs,
) -> dict:
    """Discriminative pair: THETA_INTEGRATION_ON vs THETA_INTEGRATION_ABLATED."""
    results_on:  List[Dict] = []
    results_off: List[Dict] = []

    for seed in seeds:
        for theta_on in [True, False]:
            cond = "THETA_INTEGRATION_ON" if theta_on else "THETA_INTEGRATION_ABLATED"
            print(
                f"\n[V3-EXQ-122] {cond} seed={seed}"
                f" theta_k={THETA_K}"
                f" warmup={warmup_episodes} eval={eval_episodes}"
                f" steps={steps_per_episode}"
                f" alpha_world={alpha_world}"
                f" drift_prob={env_drift_prob}"
                f" {'DRY_RUN' if dry_run else ''}",
                flush=True,
            )
            r = _run_single(
                seed=seed,
                theta_on=theta_on,
                warmup_episodes=warmup_episodes,
                eval_episodes=eval_episodes,
                steps_per_episode=steps_per_episode,
                self_dim=self_dim,
                world_dim=world_dim,
                lr=lr,
                alpha_world=alpha_world,
                alpha_self=alpha_self,
                harm_scale=harm_scale,
                proximity_harm_scale=proximity_harm_scale,
                env_drift_prob=env_drift_prob,
                env_drift_interval=env_drift_interval,
                dry_run=dry_run,
            )
            if theta_on:
                results_on.append(r)
            else:
                results_off.append(r)

    def _avg(results: List[Dict], key: str) -> float:
        vals = [r[key] for r in results]
        return float(sum(vals) / max(1, len(vals)))

    auc_on   = _avg(results_on,  "harm_auc")
    auc_off  = _avg(results_off, "harm_auc")
    auc_delta = auc_on - auc_off

    # Per-seed C3 check (consistency)
    per_seed_c3 = [
        ron["harm_auc"] > roff["harm_auc"]
        for ron, roff in zip(results_on, results_off)
    ]
    c3_pass = all(per_seed_c3)

    # C4: data quality -- min harm steps across all cells
    min_harm = min(r["harm_steps_eval"] for r in results_on + results_off)

    # C5: batch non-triviality (ON condition only, diagnostic)
    mean_batch_var = _avg(results_on, "mean_within_batch_var")
    c5_pass = mean_batch_var >= THRESH_C5_BATCH_VAR

    c1_pass = auc_delta >= THRESH_C1_AUC_DELTA
    c2_pass = auc_on   >= THRESH_C2_AUC_ABS
    c4_pass = min_harm >= THRESH_C4_MIN_HARM
    all_pass = c1_pass and c2_pass and c3_pass and c4_pass

    criteria_met = sum([c1_pass, c2_pass, c3_pass, c4_pass, c5_pass])
    status = "PASS" if all_pass else "FAIL"

    print(f"\n[V3-EXQ-122] Final results:", flush=True)
    print(
        f"  harm_auc_ON={auc_on:.4f}  harm_auc_ABLATED={auc_off:.4f}"
        f"  delta={auc_delta:+.4f}  (C1 thresh >={THRESH_C1_AUC_DELTA})"
        f"  C1={'PASS' if c1_pass else 'FAIL'}",
        flush=True,
    )
    print(
        f"  harm_auc_ON={auc_on:.4f}  (C2 absolute thresh >={THRESH_C2_AUC_ABS})"
        f"  C2={'PASS' if c2_pass else 'FAIL'}",
        flush=True,
    )
    print(
        f"  per_seed_ON_vs_ABLATED: {per_seed_c3}"
        f"  C3={'PASS' if c3_pass else 'FAIL'}",
        flush=True,
    )
    print(
        f"  min_harm_steps={min_harm}  (C4 thresh >={THRESH_C4_MIN_HARM})"
        f"  C4={'PASS' if c4_pass else 'FAIL'}",
        flush=True,
    )
    print(
        f"  mean_batch_var={mean_batch_var:.6f}"
        f"  (C5 thresh >={THRESH_C5_BATCH_VAR})"
        f"  C5={'PASS' if c5_pass else 'FAIL'} (diagnostic only)",
        flush=True,
    )
    print(f"  status={status}  ({criteria_met}/5 criteria met, 4 required)", flush=True)

    failure_notes: List[str] = []
    if not c1_pass:
        failure_notes.append(
            f"C1 FAIL: harm_auc_ON={auc_on:.4f} vs harm_auc_ABLATED={auc_off:.4f}"
            f" (delta={auc_delta:+.4f}, needs >={THRESH_C1_AUC_DELTA})."
            " Theta-averaging (k=2) did not improve harm AUC over raw same-rate sampling."
            " Possible causes: (1) theta_k=2 still too coarse for CausalGridWorldV2 dynamics,"
            " (2) harm_buf training signal too sparse at this hazard density,"
            " (3) E3 harm_eval learns equally well from raw or averaged z_world at world_dim=32."
        )
    if not c2_pass:
        failure_notes.append(
            f"C2 FAIL: harm_auc_ON={auc_on:.4f} (needs >={THRESH_C2_AUC_ABS})."
            " ON condition did not achieve reliable harm prediction."
            " Check harm_buf_pos_final -- insufficient positive samples."
        )
    if not c3_pass:
        failure_notes.append(
            f"C3 FAIL: per_seed direction inconsistent ({per_seed_c3})."
            " ON did not consistently beat ABLATED across seeds."
        )
    if not c4_pass:
        failure_notes.append(
            f"C4 FAIL: min_harm_steps={min_harm} < {THRESH_C4_MIN_HARM}."
            " Insufficient harm events in eval. Increase env_drift_prob or hazard density."
        )
    if not c5_pass:
        failure_notes.append(
            f"C5 (diagnostic): mean_batch_var={mean_batch_var:.6f} < {THRESH_C5_BATCH_VAR}."
            " ThetaBuffer z_world frames within each batch are near-identical."
            " The buffer may be trivially averaging a constant -- check alpha_world and env."
        )

    for note in failure_notes:
        print(f"  {note}", flush=True)

    if all_pass:
        interpretation = (
            "MECH-089 SUPPORTED: Theta-averaging (k=2) of z_world"
            f" achieves harm_auc_ON={auc_on:.4f} (>={THRESH_C2_AUC_ABS}) and outperforms"
            f" raw same-rate z_world by delta={auc_delta:+.4f} (>={THRESH_C1_AUC_DELTA})."
            " Direction consistent across all seeds (C3). Data quality confirmed (C4)."
            " Temporal batching of E1 outputs before E3 sampling provides"
            " measurable benefit for E3 harm prediction."
        )
    elif c1_pass and not c2_pass:
        interpretation = (
            "PARTIAL: C1 passes (relative delta={:.4f}) but C2 fails"
            " (auc_ON={:.4f} < {:.2f}). Theta-averaging provides relative advantage"
            " but neither condition learns reliable harm prediction."
            " Data sparsity or training budget insufficient."
        ).format(auc_delta, auc_on, THRESH_C2_AUC_ABS)
    elif c2_pass and not c1_pass:
        interpretation = (
            "PARTIAL: C2 passes (harm_auc_ON={:.4f} >= {:.2f}) but C1 fails"
            " (delta={:+.4f} < {:.2f}). E3 harm evaluation works but theta-averaging"
            " provides no measurable benefit over raw same-rate z_world sampling at theta_k=2."
            " Raw z_world may already be sufficiently stable at this env scale and world_dim."
        ).format(auc_on, THRESH_C2_AUC_ABS, auc_delta, THRESH_C1_AUC_DELTA)
    else:
        interpretation = (
            "MECH-089 NOT SUPPORTED at theta_k=2 operationalisation."
            f" harm_auc_ON={auc_on:.4f}, harm_auc_ABLATED={auc_off:.4f},"
            f" delta={auc_delta:+.4f}."
            " Theta-cycle averaging of E1 z_world output did not improve E3 harm"
            " prediction AUC over raw same-rate z_world delivery."
            " Consistent with EXQ-066 backwards result (k=4 degraded performance):"
            " the environment may change meaningfully every step, making any averaging"
            " (even at k=2) destroy temporal resolution E3 relies on."
        )

    failure_section = ""
    if failure_notes:
        failure_section = "\n## Failure Notes\n\n" + "\n".join(
            f"- {n}" for n in failure_notes
        )

    per_on_rows = "\n".join(
        f"  seed={r['seed']}:"
        f" harm_auc={r['harm_auc']:.4f}"
        f" harm_rate={r['harm_rate']:.4f}"
        f" harm_steps_eval={r['harm_steps_eval']}"
        f" n_e3_ticks={r['n_e3_ticks_eval']}"
        f" mean_batch_var={r['mean_within_batch_var']:.6f}"
        f" train_harm_steps={r['train_harm_steps']}"
        for r in results_on
    )
    per_off_rows = "\n".join(
        f"  seed={r['seed']}:"
        f" harm_auc={r['harm_auc']:.4f}"
        f" harm_rate={r['harm_rate']:.4f}"
        f" harm_steps_eval={r['harm_steps_eval']}"
        f" n_e3_ticks={r['n_e3_ticks_eval']}"
        f" train_harm_steps={r['train_harm_steps']}"
        for r in results_off
    )

    summary_markdown = (
        f"# V3-EXQ-122 -- MECH-089 Theta-Gamma Integration Discriminative Pair\n\n"
        f"**Status:** {status}\n"
        f"**Claim:** MECH-089\n"
        f"**Proposal:** EXP-0020 / EVB-0016\n"
        f"**Seeds:** {list(seeds)}\n"
        f"**theta_k:** {THETA_K}  (E3 ticks every {THETA_K} steps; same rate in both conditions)\n"
        f"**alpha_world:** {alpha_world}  (SD-008)\n"
        f"**Warmup:** {warmup_episodes} eps  **Eval:** {eval_episodes} eps\n"
        f"**env_drift_prob:** {env_drift_prob}  **env_drift_interval:** {env_drift_interval}\n\n"
        f"## Design\n\n"
        f"THETA_INTEGRATION_ON: E3 receives ThetaBuffer.summary() = mean of last {THETA_K} z_world"
        f" frames at each E3 tick.\n"
        f"THETA_INTEGRATION_ABLATED: E3 receives raw most-recent z_world frame at each E3 tick."
        f" Same E3 update rate (every {THETA_K} steps). Only the content differs.\n\n"
        f"Redesign rationale (vs EXQ-066): theta_k=4 degraded E3 performance (batched error"
        f" 2.28x worse than raw). Redesign uses theta_k=2 (shorter window) and measures"
        f" harm AUC rather than prediction error variance for a cleaner discriminative outcome.\n\n"
        f"## Pre-Registered Thresholds\n\n"
        f"C1: harm_auc_ON - harm_auc_ABLATED >= {THRESH_C1_AUC_DELTA}  (relative advantage)\n"
        f"C2: harm_auc_ON >= {THRESH_C2_AUC_ABS}  (absolute harm prediction learning)\n"
        f"C3: harm_auc_ON > harm_auc_ABLATED for ALL seeds  (consistency)\n"
        f"C4: min_harm_steps >= {THRESH_C4_MIN_HARM}  (data quality)\n"
        f"C5 (diagnostic): mean_within_batch_var >= {THRESH_C5_BATCH_VAR}  (buffer non-trivial)\n\n"
        f"## Aggregate Results\n\n"
        f"| Metric | THETA_ON | THETA_ABLATED | Delta | Pass |\n"
        f"|--------|----------|---------------|-------|------|\n"
        f"| harm_AUC (C1 delta) | {auc_on:.4f} | {auc_off:.4f}"
        f" | {auc_delta:+.4f} | {'YES' if c1_pass else 'NO'} |\n"
        f"| harm_AUC >= {THRESH_C2_AUC_ABS} (C2) | {auc_on:.4f} | -- | --"
        f" | {'YES' if c2_pass else 'NO'} |\n"
        f"| seed consistency (C3) | {per_seed_c3} | -- | --"
        f" | {'YES' if c3_pass else 'NO'} |\n"
        f"| min_harm_steps (C4) | {min_harm} | -- | --"
        f" | {'YES' if c4_pass else 'NO'} |\n"
        f"| mean_batch_var (C5 diag) | {mean_batch_var:.6f} | -- | --"
        f" | {'YES' if c5_pass else 'NO'} |\n\n"
        f"## Interpretation\n\n"
        f"{interpretation}\n\n"
        f"## Per-Seed (THETA_INTEGRATION_ON)\n\n"
        f"{per_on_rows}\n\n"
        f"## Per-Seed (THETA_INTEGRATION_ABLATED)\n\n"
        f"{per_off_rows}\n"
        f"{failure_section}\n"
    )

    return {
        "status": status,
        "metrics": {
            "harm_auc_on":                  float(auc_on),
            "harm_auc_ablated":             float(auc_off),
            "auc_delta":                    float(auc_delta),
            "min_harm_steps_eval":          float(min_harm),
            "mean_within_batch_var":        float(mean_batch_var),
            "crit1_pass":                   1.0 if c1_pass else 0.0,
            "crit2_pass":                   1.0 if c2_pass else 0.0,
            "crit3_pass":                   1.0 if c3_pass else 0.0,
            "crit4_pass":                   1.0 if c4_pass else 0.0,
            "crit5_pass":                   1.0 if c5_pass else 0.0,
            "criteria_met":                 float(criteria_met),
            "n_seeds":                      float(len(seeds)),
            "theta_k":                      float(THETA_K),
            "alpha_world":                  float(alpha_world),
            "env_drift_prob":               float(env_drift_prob),
        },
        "summary_markdown": summary_markdown,
        "claim_ids": CLAIM_IDS,
        "evidence_direction": (
            "supports" if all_pass
            else ("mixed" if (c1_pass or c2_pass) else "weakens")
        ),
        "experiment_type": EXPERIMENT_TYPE,
        "per_seed_on":  results_on,
        "per_seed_off": results_off,
    }


if __name__ == "__main__":
    import argparse
    import json
    from datetime import datetime, timezone

    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds",           type=int,   nargs="+", default=[42, 123])
    parser.add_argument("--warmup",          type=int,   default=400)
    parser.add_argument("--eval-eps",        type=int,   default=50)
    parser.add_argument("--steps",           type=int,   default=200)
    parser.add_argument("--self-dim",        type=int,   default=32)
    parser.add_argument("--world-dim",       type=int,   default=32)
    parser.add_argument("--lr",              type=float, default=1e-3)
    parser.add_argument("--alpha-world",     type=float, default=0.9)
    parser.add_argument("--alpha-self",      type=float, default=0.3)
    parser.add_argument("--harm-scale",      type=float, default=0.02)
    parser.add_argument("--proximity-scale", type=float, default=0.01)
    parser.add_argument("--drift-prob",      type=float, default=0.3)
    parser.add_argument("--drift-interval",  type=int,   default=3)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Quick dry-run: 3 warmup, 2 eval, 50 steps per cell. Writes JSON.",
    )
    args = parser.parse_args()

    result = run(
        seeds=tuple(args.seeds),
        warmup_episodes=args.warmup,
        eval_episodes=args.eval_eps,
        steps_per_episode=args.steps,
        self_dim=args.self_dim,
        world_dim=args.world_dim,
        lr=args.lr,
        alpha_world=args.alpha_world,
        alpha_self=args.alpha_self,
        harm_scale=args.harm_scale,
        proximity_harm_scale=args.proximity_scale,
        env_drift_prob=args.drift_prob,
        env_drift_interval=args.drift_interval,
        dry_run=args.dry_run,
    )

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result["run_timestamp"] = ts
    result["claim"] = CLAIM_IDS[0]
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

    print(f"\n[V3-EXQ-122] Result written to {out_path}", flush=True)
    print(f"[V3-EXQ-122] status={result['status']}", flush=True)
