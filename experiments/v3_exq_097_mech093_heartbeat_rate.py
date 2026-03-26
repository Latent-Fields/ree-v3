"""
V3-EXQ-097 -- MECH-093 z_beta Heartbeat Rate Modulation: Discriminative Pair

Claim: MECH-093 -- z_beta modulates E3 heartbeat frequency, distinct from
precision-weighting.

MECH-093 asserts: high harm salience (large |z_beta|) -> faster E3 updates
(finer temporal harm attribution); routine operation -> lower |z_beta| -> slower
E3 updates (more stable policy). This is DISTINCT from MECH-059 precision-weighting:
rate modulation governs how often E3 updates, not how much weight each update carries.

Discriminative pair:
  BETA_MOD_ON  -- clock.update_e3_rate_from_beta(z_beta) called each step.
                  E3 rate varies in [beta_rate_min_steps=5, beta_rate_max_steps=20]
                  based on |z_beta| magnitude.
  BETA_MOD_OFF -- E3 rate fixed at 10 steps (base). No z_beta modulation.

Both conditions: alpha_world=0.9 (SD-008), same seed, same env, same training budget.

Phase 1 (modulation gate, BETA_MOD_ON only):
  At each E3 tick during training, log e3_steps_per_tick and whether the current
  episode has seen any harm. C1 passes if high-harm episodes have a mean tick rate
  at least 2 steps lower than low-harm episodes.

Phase 2 (behavioral comparison, both conditions, eval):
  C2: BETA_MOD_ON harm_rate <= 0.95 * BETA_MOD_OFF harm_rate.
  C3: variance of action index in safe phases (BETA_MOD_ON) <= 0.95 * (BETA_MOD_OFF).

PASS if C1 AND (C2 OR C3).
If C1 fails: report "modulation not instantiated" -- MECH-093 weakened.
If C1 passes but C2 and C3 fail: "modulation confirmed but no behavioral benefit".
"""

import sys
import math
import random
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
import torch.optim as optim

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig


EXPERIMENT_TYPE = "v3_exq_097_mech093_heartbeat_rate"
CLAIM_IDS = ["MECH-093"]

HARM_THRESHOLD = 0.5   # harm_eval score above which agent flees (random action)
SAFE_WINDOW    = 15    # consecutive no-harm steps required to enter safe phase


def _action_to_onehot(action_idx: int, num_actions: int, device) -> torch.Tensor:
    v = torch.zeros(1, num_actions, device=device)
    v[0, action_idx] = 1.0
    return v


def _run_single(
    seed: int,
    beta_mod: bool,
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
) -> Dict:
    """
    Run one (seed, condition) cell and return Phase 1 and Phase 2 metrics.

    beta_mod=True  -> BETA_MOD_ON: update_e3_rate_from_beta() called each step.
    beta_mod=False -> BETA_MOD_OFF: E3 rate fixed at base (no z_beta modulation).
    """
    torch.manual_seed(seed)
    random.seed(seed)

    cond_label = "BETA_MOD_ON" if beta_mod else "BETA_MOD_OFF"

    env = CausalGridWorldV2(
        seed=seed,
        size=12,
        num_hazards=4,
        num_resources=5,
        hazard_harm=harm_scale,
        env_drift_interval=5,
        env_drift_prob=0.1,
        proximity_harm_scale=proximity_harm_scale,
        proximity_benefit_scale=proximity_harm_scale * 0.6,
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
    )
    config.latent.unified_latent_mode = False  # SD-005 split always enabled

    agent = REEAgent(config)

    # Separate optimizers: standard params + harm_eval head
    standard_params = [
        p for n, p in agent.named_parameters()
        if "harm_eval_head" not in n
    ]
    harm_eval_params = list(agent.e3.harm_eval_head.parameters())
    optimizer = optim.Adam(standard_params, lr=lr)
    harm_eval_optimizer = optim.Adam(harm_eval_params, lr=1e-4)

    # Harm buffer for E3 training (balanced)
    harm_buf_pos: List[torch.Tensor] = []
    harm_buf_neg: List[torch.Tensor] = []
    MAX_BUF_EACH = 2000

    # Phase 1 data (BETA_MOD_ON training only)
    p1_rates: List[int]   = []   # e3_steps_per_tick at each E3 tick
    p1_episode_had_harm: List[bool] = []  # did this episode have any harm?

    # ------------------------------------------------------------------ TRAIN
    agent.train()

    for ep in range(warmup_episodes):
        _, obs_dict = env.reset()
        agent.reset()

        episode_harm = 0.0
        cached_harm_score = None
        last_action_idx = None
        ep_p1_rates: List[int] = []  # rates logged this episode (for BETA_MOD_ON)

        for _ in range(steps_per_episode):
            obs_body  = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]

            latent = agent.sense(obs_body, obs_world)
            ticks  = agent.clock.advance()

            # E1 tick: populates experience buffers + calls update_e3_rate_from_beta
            if ticks["e1_tick"]:
                agent._e1_tick(latent)
                if not beta_mod:
                    # BETA_MOD_OFF: undo rate change -- keep E3 at base 10 steps
                    agent.clock._current_e3_steps = agent.clock._e3_base_steps

            # E3 tick: refresh cached harm score; log Phase 1 data
            if ticks["e3_tick"] or cached_harm_score is None:
                z_w = latent.z_world.detach()
                with torch.no_grad():
                    cached_harm_score = float(
                        agent.e3.harm_eval(z_w).item()
                    )
                if beta_mod:
                    ep_p1_rates.append(agent.clock.e3_steps_per_tick)

            # Policy: flee (random action) if harm score high; hold last action otherwise
            if cached_harm_score > HARM_THRESHOLD:
                action_idx = random.randint(0, env.action_dim - 1)
            else:
                action_idx = last_action_idx if last_action_idx is not None else 0
            last_action_idx = action_idx

            action = _action_to_onehot(action_idx, env.action_dim, agent.device)
            agent._last_action = action

            _, harm_signal, done, _, obs_dict = env.step(action)

            if float(harm_signal) < 0:
                episode_harm += abs(float(harm_signal))

            # Update harm replay buffer
            z_w_curr = latent.z_world.detach()
            is_harm = float(harm_signal) < 0
            if is_harm:
                harm_buf_pos.append(z_w_curr)
                if len(harm_buf_pos) > MAX_BUF_EACH:
                    harm_buf_pos = harm_buf_pos[-MAX_BUF_EACH:]
            else:
                harm_buf_neg.append(z_w_curr)
                if len(harm_buf_neg) > MAX_BUF_EACH:
                    harm_buf_neg = harm_buf_neg[-MAX_BUF_EACH:]

            # E1 + E2 loss (standard)
            e1_loss = agent.compute_prediction_loss()
            e2_loss = agent.compute_e2_loss()
            total_loss = e1_loss + e2_loss
            if total_loss.requires_grad:
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                optimizer.step()

            # E3 harm_eval balanced training
            if len(harm_buf_pos) >= 4 and len(harm_buf_neg) >= 4:
                k_pos = min(16, len(harm_buf_pos))
                k_neg = min(16, len(harm_buf_neg))
                pos_idx = torch.randperm(len(harm_buf_pos))[:k_pos].tolist()
                neg_idx = torch.randperm(len(harm_buf_neg))[:k_neg].tolist()
                zw_pos = torch.cat([harm_buf_pos[i] for i in pos_idx], dim=0)
                zw_neg = torch.cat([harm_buf_neg[i] for i in neg_idx], dim=0)
                zw_b   = torch.cat([zw_pos, zw_neg], dim=0)
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
                        agent.e3.harm_eval_head.parameters(), 0.5,
                    )
                    harm_eval_optimizer.step()

            if done:
                break

        # Record Phase 1 data after episode (BETA_MOD_ON only)
        if beta_mod and ep_p1_rates:
            ep_had_harm = episode_harm > 0
            for r in ep_p1_rates:
                p1_rates.append(r)
                p1_episode_had_harm.append(ep_had_harm)

        if (ep + 1) % 50 == 0 or ep == warmup_episodes - 1:
            print(
                f"  [train] cond={cond_label} seed={seed}"
                f" ep {ep+1}/{warmup_episodes}"
                f" ep_harm={episode_harm:.4f}"
                f" buf_pos={len(harm_buf_pos)} buf_neg={len(harm_buf_neg)}",
                flush=True,
            )

    # ------------------------------------------------------------------ EVAL
    agent.eval()

    eval_harm_events = 0
    eval_total_steps = 0
    safe_phase_actions: List[int] = []  # action indices in safe phases (for C3)

    for ep in range(eval_episodes):
        _, obs_dict = env.reset()
        agent.reset()

        cached_harm_score = None
        last_action_idx = None
        consec_safe_steps = 0

        for _ in range(steps_per_episode):
            obs_body  = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]

            with torch.no_grad():
                latent = agent.sense(obs_body, obs_world)
                ticks  = agent.clock.advance()

                if ticks["e1_tick"]:
                    agent._e1_tick(latent)
                    if not beta_mod:
                        agent.clock._current_e3_steps = agent.clock._e3_base_steps

                if ticks["e3_tick"] or cached_harm_score is None:
                    z_w = latent.z_world.detach()
                    cached_harm_score = float(agent.e3.harm_eval(z_w).item())

            # Policy
            if cached_harm_score > HARM_THRESHOLD:
                action_idx = random.randint(0, env.action_dim - 1)
            else:
                action_idx = last_action_idx if last_action_idx is not None else 0
            last_action_idx = action_idx

            action = _action_to_onehot(action_idx, env.action_dim, agent.device)
            agent._last_action = action

            _, harm_signal, done, _, obs_dict = env.step(action)

            eval_total_steps += 1
            if float(harm_signal) < 0:
                eval_harm_events += 1
                consec_safe_steps = 0
            else:
                consec_safe_steps += 1

            # C3: collect action indices during safe phases
            if consec_safe_steps >= SAFE_WINDOW:
                safe_phase_actions.append(action_idx)

            if done:
                break

    harm_rate = float(eval_harm_events) / max(1, eval_total_steps)

    # C3: variance of action indices in safe phases
    if len(safe_phase_actions) >= 2:
        action_var = float(
            sum((a - sum(safe_phase_actions) / len(safe_phase_actions)) ** 2
                for a in safe_phase_actions) / len(safe_phase_actions)
        )
    else:
        action_var = 0.0

    # Phase 1 aggregates (BETA_MOD_ON only; BETA_MOD_OFF returns NaN)
    if beta_mod and p1_rates:
        high_harm_rates = [r for r, h in zip(p1_rates, p1_episode_had_harm) if h]
        low_harm_rates  = [r for r, h in zip(p1_rates, p1_episode_had_harm) if not h]
        mean_high = float(sum(high_harm_rates) / max(1, len(high_harm_rates))) if high_harm_rates else float("nan")
        mean_low  = float(sum(low_harm_rates)  / max(1, len(low_harm_rates)))  if low_harm_rates  else float("nan")
        n_high    = len(high_harm_rates)
        n_low     = len(low_harm_rates)
    else:
        mean_high = float("nan")
        mean_low  = float("nan")
        n_high    = 0
        n_low     = 0

    print(
        f"  [eval] cond={cond_label} seed={seed}"
        f" harm_rate={harm_rate:.4f}"
        f" action_var_safe={action_var:.4f}"
        f" safe_phase_steps={len(safe_phase_actions)}"
        f" p1_mean_high={mean_high:.2f} p1_mean_low={mean_low:.2f}"
        f" (n_high={n_high} n_low={n_low})",
        flush=True,
    )

    return {
        "condition":           cond_label,
        "beta_mod":            beta_mod,
        "seed":                seed,
        "harm_rate":           harm_rate,
        "action_var_safe":     action_var,
        "n_safe_phase_steps":  len(safe_phase_actions),
        "p1_mean_high_harm":   mean_high,
        "p1_mean_low_harm":    mean_low,
        "p1_n_high_harm_ticks": n_high,
        "p1_n_low_harm_ticks":  n_low,
    }


def run(
    seed: int = 42,
    warmup_episodes: int = 200,
    eval_episodes: int = 40,
    steps_per_episode: int = 200,
    self_dim: int = 32,
    world_dim: int = 32,
    lr: float = 1e-3,
    alpha_world: float = 0.9,
    alpha_self: float = 0.3,
    harm_scale: float = 0.02,
    proximity_harm_scale: float = 0.05,
    **kwargs,
) -> dict:
    """
    Discriminative pair: BETA_MOD_ON vs BETA_MOD_OFF.
    Single seed (conflict-resolution experiment).
    """
    results = {}
    for beta_mod in [True, False]:
        label = "BETA_MOD_ON" if beta_mod else "BETA_MOD_OFF"
        print(
            f"\n[V3-EXQ-097] {label} seed={seed}"
            f" warmup={warmup_episodes} eval={eval_episodes}"
            f" alpha_world={alpha_world}",
            flush=True,
        )
        r = _run_single(
            seed=seed,
            beta_mod=beta_mod,
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
        )
        results[label] = r

    r_on  = results["BETA_MOD_ON"]
    r_off = results["BETA_MOD_OFF"]

    # --- C1: modulation gate (BETA_MOD_ON phase 1 data) ---
    p1_high = r_on["p1_mean_high_harm"]
    p1_low  = r_on["p1_mean_low_harm"]
    n_high  = r_on["p1_n_high_harm_ticks"]
    n_low   = r_on["p1_n_low_harm_ticks"]

    p1_rate_gap = float("nan")
    if n_high > 0 and n_low > 0 and not math.isnan(p1_high) and not math.isnan(p1_low):
        p1_rate_gap = p1_low - p1_high
        c1_pass = p1_rate_gap >= 2.0
    else:
        c1_pass = False
        p1_rate_gap = float("nan")

    # --- C2: harm_rate ---
    harm_rate_on  = r_on["harm_rate"]
    harm_rate_off = r_off["harm_rate"]
    if harm_rate_off > 1e-8:
        harm_rate_ratio = harm_rate_on / harm_rate_off
        c2_pass = harm_rate_on <= 0.95 * harm_rate_off
    else:
        harm_rate_ratio = 1.0
        c2_pass = False

    # --- C3: action entropy variance in safe phases ---
    var_on  = r_on["action_var_safe"]
    var_off = r_off["action_var_safe"]
    if var_off > 1e-6:
        action_var_ratio = var_on / var_off
        c3_pass = var_on <= 0.95 * var_off
    else:
        # Both near-zero: C3 inconclusive (not a failure -- stable in both)
        action_var_ratio = 1.0
        c3_pass = False
        c3_note = "inconclusive (var_OFF near 0 -- both conditions stable)"
    if var_off > 1e-6:
        c3_note = ""

    # --- PASS logic ---
    if not c1_pass:
        status = "FAIL"
        failure_note = "modulation not instantiated (C1 gate failed)"
    elif c2_pass or c3_pass:
        status = "PASS"
        failure_note = ""
    else:
        status = "FAIL"
        failure_note = "modulation confirmed but no behavioral benefit -- MECH-093 weakened"

    print(f"\n[V3-EXQ-097] Final results:", flush=True)
    print(
        f"  C1: p1_rate_gap={p1_rate_gap:.2f} (need >=2.0) -> {'PASS' if c1_pass else 'FAIL'}",
        flush=True,
    )
    print(
        f"  C2: harm_rate_ON={harm_rate_on:.4f} harm_rate_OFF={harm_rate_off:.4f}"
        f" ratio={harm_rate_ratio:.3f} -> {'PASS' if c2_pass else 'FAIL'}",
        flush=True,
    )
    print(
        f"  C3: action_var_ON={var_on:.4f} action_var_OFF={var_off:.4f}"
        f" ratio={action_var_ratio:.3f} -> {'PASS' if c3_pass else 'FAIL'}"
        + (f" [{c3_note}]" if c3_note else ""),
        flush=True,
    )
    print(f"  Status: {status}", flush=True)
    if failure_note:
        print(f"  Note: {failure_note}", flush=True)

    criteria_met = sum([c1_pass, c2_pass, c3_pass])

    metrics = {
        "p1_mean_rate_high_harm":    float(p1_high) if not math.isnan(p1_high) else -1.0,
        "p1_mean_rate_low_harm":     float(p1_low)  if not math.isnan(p1_low)  else -1.0,
        "p1_rate_gap":               float(p1_rate_gap) if not math.isnan(p1_rate_gap) else -1.0,
        "p1_n_high_harm_ticks":      float(n_high),
        "p1_n_low_harm_ticks":       float(n_low),
        "harm_rate_on":              float(harm_rate_on),
        "harm_rate_off":             float(harm_rate_off),
        "harm_rate_ratio":           float(harm_rate_ratio),
        "action_var_on":             float(var_on),
        "action_var_off":            float(var_off),
        "action_var_ratio":          float(action_var_ratio),
        "crit1_pass":                1.0 if c1_pass else 0.0,
        "crit2_pass":                1.0 if c2_pass else 0.0,
        "crit3_pass":                1.0 if c3_pass else 0.0,
        "criteria_met":              float(criteria_met),
        "alpha_world":               float(alpha_world),
        "seed":                      float(seed),
        "warmup_episodes":           float(warmup_episodes),
        "eval_episodes":             float(eval_episodes),
    }

    # Interpretation
    if status == "PASS":
        interpretation = (
            "MECH-093 SUPPORTED: z_beta modulation instantiated (C1) and behavioral"
            f" benefit confirmed. p1_rate_gap={p1_rate_gap:.2f} steps (high-harm episodes"
            f" trigger faster E3). harm_rate_ON={harm_rate_on:.4f} vs OFF={harm_rate_off:.4f}"
            f" (ratio={harm_rate_ratio:.3f}). action_var_ON={var_on:.4f} vs"
            f" OFF={var_off:.4f} (ratio={action_var_ratio:.3f})."
        )
    elif not c1_pass:
        interpretation = (
            "MECH-093 NOT SUPPORTED: modulation not instantiated. E3 rate did not"
            f" differ between high-harm and low-harm episodes"
            f" (p1_rate_gap={p1_rate_gap:.2f}, need >=2)."
            " z_beta may not encode harm salience after current training budget."
            " Options: more warmup episodes, explicit z_beta harm supervision,"
            " or larger harm_scale to increase z_beta excursion."
        )
    else:
        interpretation = (
            "MECH-093 WEAKENED: modulation instantiated (C1 gap="
            f"{p1_rate_gap:.2f}) but no behavioral benefit."
            f" harm_rate_ON={harm_rate_on:.4f} vs OFF={harm_rate_off:.4f}."
            f" action_var_ON={var_on:.4f} vs OFF={var_off:.4f}."
            " Rate modulation occurs but does not translate to measurable outcome"
            " improvement at this training scale."
        )
    if failure_note:
        interpretation += f" [{failure_note}]"

    failure_notes: List[str] = []
    if not c1_pass:
        failure_notes.append(
            f"C1 FAIL: p1_rate_gap={p1_rate_gap:.2f} < 2.0"
            f" (n_high_ticks={n_high}, n_low_ticks={n_low})"
        )
    if not c2_pass:
        failure_notes.append(
            f"C2 FAIL: harm_rate_ON={harm_rate_on:.4f} > 0.95 * harm_rate_OFF={harm_rate_off:.4f}"
        )
    if not c3_pass:
        msg = (
            f"C3 FAIL: action_var_ON={var_on:.4f} > 0.95 * action_var_OFF={var_off:.4f}"
        )
        if c3_note:
            msg += f" [{c3_note}]"
        failure_notes.append(msg)

    summary_markdown = (
        f"# V3-EXQ-097 -- MECH-093 z_beta Heartbeat Rate Modulation\n\n"
        f"**Status:** {status}\n"
        f"**Claims:** MECH-093\n"
        f"**Seed:** {seed}\n"
        f"**alpha_world:** {alpha_world}  (SD-008)\n"
        f"**Warmup:** {warmup_episodes} eps  **Eval:** {eval_episodes} eps"
        f"  **Steps/ep:** {steps_per_episode}\n\n"
        f"## Pre-Registered Thresholds\n\n"
        f"C1 (modulation gate): p1_rate_gap (low_harm_mean - high_harm_mean) >= 2.0 steps\n"
        f"C2 (reactivity):      harm_rate_ON <= 0.95 * harm_rate_OFF\n"
        f"C3 (stability):       action_var_ON <= 0.95 * action_var_OFF (safe phases)\n"
        f"PASS: C1 AND (C2 OR C3)\n\n"
        f"## Phase 1 Results (Modulation Gate, BETA_MOD_ON)\n\n"
        f"| Metric | Value |\n"
        f"|--------|-------|\n"
        f"| mean e3_steps_per_tick (high-harm eps) | {p1_high:.2f} (n={n_high}) |\n"
        f"| mean e3_steps_per_tick (low-harm eps)  | {p1_low:.2f} (n={n_low}) |\n"
        f"| rate_gap (low - high)                  | {p1_rate_gap:.2f} |\n"
        f"| C1 result | {'PASS' if c1_pass else 'FAIL'} |\n\n"
        f"## Phase 2 Results (Behavioral)\n\n"
        f"| Condition | harm_rate | action_var_safe | n_safe_steps |\n"
        f"|-----------|-----------|-----------------|---------------|\n"
        f"| BETA_MOD_ON  | {harm_rate_on:.4f}  | {var_on:.4f}  |"
        f" {r_on['n_safe_phase_steps']} |\n"
        f"| BETA_MOD_OFF | {harm_rate_off:.4f} | {var_off:.4f} |"
        f" {r_off['n_safe_phase_steps']} |\n\n"
        f"| Criterion | Result | Value |\n"
        f"|-----------|--------|-------|\n"
        f"| C1: rate_gap >= 2.0               | {'PASS' if c1_pass else 'FAIL'}"
        f" | {p1_rate_gap:.2f} |\n"
        f"| C2: harm_rate ratio <= 0.95       | {'PASS' if c2_pass else 'FAIL'}"
        f" | {harm_rate_ratio:.3f} |\n"
        f"| C3: action_var ratio <= 0.95      | {'PASS' if c3_pass else 'FAIL'}"
        f" | {action_var_ratio:.3f} |\n\n"
        f"Criteria met: {criteria_met}/3 -> **{status}**\n\n"
        f"## Interpretation\n\n"
        f"{interpretation}\n"
    )
    if failure_notes:
        summary_markdown += "\n## Failure Notes\n\n" + "\n".join(
            f"- {n}" for n in failure_notes
        ) + "\n"

    return {
        "status":            status,
        "metrics":           metrics,
        "summary_markdown":  summary_markdown,
        "claim_ids":         CLAIM_IDS,
        "evidence_direction": "supports" if status == "PASS" else "weakens",
        "experiment_type":   EXPERIMENT_TYPE,
        "fatal_error_count": 0,
    }


if __name__ == "__main__":
    import argparse
    import json
    from datetime import datetime, timezone

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",            type=int,   default=42)
    parser.add_argument("--warmup",          type=int,   default=200)
    parser.add_argument("--eval-eps",        type=int,   default=40)
    parser.add_argument("--steps",           type=int,   default=200)
    parser.add_argument("--alpha-world",     type=float, default=0.9)
    parser.add_argument("--alpha-self",      type=float, default=0.3)
    parser.add_argument("--harm-scale",      type=float, default=0.02)
    parser.add_argument("--proximity-scale", type=float, default=0.05)
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="Quick smoke test: 5 warmup, 3 eval, 50 steps -- no file written",
    )
    args = parser.parse_args()

    if args.smoke_test:
        warmup   = 5
        eval_eps = 3
        steps    = 50
        print("[V3-EXQ-097] SMOKE TEST MODE", flush=True)
    else:
        warmup   = args.warmup
        eval_eps = args.eval_eps
        steps    = args.steps

    result = run(
        seed=args.seed,
        warmup_episodes=warmup,
        eval_episodes=eval_eps,
        steps_per_episode=steps,
        alpha_world=args.alpha_world,
        alpha_self=args.alpha_self,
        harm_scale=args.harm_scale,
        proximity_harm_scale=args.proximity_scale,
    )

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result["run_timestamp"]       = ts
    result["claim"]               = CLAIM_IDS[0]
    result["verdict"]             = result["status"]
    result["run_id"]              = f"{EXPERIMENT_TYPE}_{ts}_v3"
    result["architecture_epoch"]  = "ree_hybrid_guardrails_v1"

    if args.smoke_test:
        print(f"\n[SMOKE] Status: {result['status']}", flush=True)
        print("[SMOKE] Key metrics:", flush=True)
        for k in [
            "p1_rate_gap", "p1_n_high_harm_ticks", "p1_n_low_harm_ticks",
            "harm_rate_on", "harm_rate_off", "harm_rate_ratio",
            "action_var_on", "action_var_off",
            "crit1_pass", "crit2_pass", "crit3_pass",
        ]:
            print(f"  {k}: {result['metrics'].get(k, 'N/A')}", flush=True)
        print("[SMOKE] Script ran without error. Not writing output file.", flush=True)
    else:
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
