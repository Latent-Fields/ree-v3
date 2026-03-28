#!/opt/local/bin/python3
"""
V3-EXQ-116 -- MECH-093 z_beta Heartbeat Rate Modulation: Multi-Seed Discriminative Pair

Claim: MECH-093 -- z_beta modulates E3 heartbeat frequency, distinct from
precision-weighting.

Proposal: EXP-0011 / EVB-0009
Dispatch mode: discriminative_pair
Min shared seeds: 2

Background:
  EXQ-097b (single seed=42, 500 warmup eps) PASS 2/3:
    C1 r=0.964 (n=29937 steps) -- step-level z_beta/rate correlation confirmed.
    C3 action_var_ratio=0.096 -- 10x lower action variance in safe phases (BETA_MOD_ON).
    C2 FAIL (harm_rate_ratio=1.58): state-selection artifact -- BETA_MOD_ON is in
      high-z_beta phases which are high-harm contexts by construction.
  MECH-093 status: "Promote to provisional pending multi-seed replication."
  This experiment provides that replication with seeds [42, 123].

Mechanism under test:
  MECH-093 asserts that z_beta magnitude drives E3 heartbeat rate: high |z_beta|
  (high harm salience / arousal) -> faster E3 updates (lower e3_steps_per_tick);
  low |z_beta| -> slower E3 updates. This is DISTINCT from precision-weighting
  (MECH-059), which scales the magnitude of each update, not the update frequency.

Discriminative pair:
  BETA_MOD_ON  -- clock.update_e3_rate_from_beta(z_beta) called each step.
                  E3 rate varies in [beta_rate_min_steps=5, beta_rate_max_steps=20]
                  based on |z_beta| magnitude.
  BETA_MOD_OFF -- E3 rate fixed at base (10 steps). No z_beta modulation.

Seeds: [42, 123] -- matched, both run under both conditions.

PRE-REGISTERED THRESHOLDS (hardcoded, not inferred post-hoc):
  C1 (modulation gate, BETA_MOD_ON):
    Pearson r(|z_beta|_norm, 1/e3_steps_per_tick) >= 0.15 per seed.
    PASS: both seeds meet threshold (pair_c1_pass_count == 2).
    The core claim: higher arousal drives faster E3 update rate step-by-step.

  C2 (consistency, C3 analog from EXQ-097b):
    action_var_ratio (ON/OFF) <= 0.90 for >= 1 of 2 seeds.
    At least one seed shows reduced safe-phase action variance in BETA_MOD_ON.
    C2 uses 0.90 (stricter than EXQ-097b 0.95) to demand a clearer signal.

  C3 (seed consistency):
    Pearson r direction positive (r >= 0) for BOTH seeds.
    Robustness check: even if C1 threshold not met for both seeds, the
    direction must be consistent (no inversion).

  C4 (data quality):
    n_steps_per_seed >= 1000 for both seeds (enough step-level data to
    compute Pearson r reliably).

PASS criteria:
  C1_both AND C3 AND C4 -> PASS -> supports MECH-093
  C1_both AND C3 AND NOT C4 -> FAIL (data quality gate) -> inconclusive
  C3 AND NOT C1_both -> FAIL -> mixed (directional but below threshold)
  NOT C3 -> FAIL -> weakens (direction inconsistent across seeds)

Decision mapping:
  PASS -> retain_ree (rate modulation confirmed with matched-seed replication)
  FAIL (C3, one seed inverted) -> hybridize (possible environment dependency)
  FAIL (C1 both, C3 passed) -> hold (below threshold -- more warmup needed?)
  FAIL (data quality) -> retry (increase warmup)

If this experiment PASSes:
  Which claim does that support and why?
  MECH-093: step-level |z_beta| correlates with instantaneous 1/e3_steps_per_tick
  in BETA_MOD_ON condition across BOTH matched seeds, confirming that the z_beta
  arousal signal drives the E3 heartbeat clock rate as claimed.
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


EXPERIMENT_TYPE = "v3_exq_116_mech093_heartbeat_multiseed"
CLAIM_IDS = ["MECH-093"]

# Pre-registered thresholds
THRESH_C1_R       = 0.15   # Pearson r >= 0.15 per seed
THRESH_C2_RATIO   = 0.90   # action_var_ratio (ON/OFF) <= 0.90 for >= 1 seed
THRESH_C4_NSTEPS  = 1000   # n_steps (step-level data) >= 1000 per seed

HARM_THRESHOLD = 0.5       # harm_eval score above which agent flees (random action)
SAFE_WINDOW    = 15        # consecutive no-harm steps required to enter safe phase


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _pearson_r(xs: List[float], ys: List[float]) -> float:
    """Pearson r between two equal-length lists. Returns NaN if undefined."""
    n = len(xs)
    if n < 3:
        return float("nan")
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    denom_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    denom_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    if denom_x < 1e-10 or denom_y < 1e-10:
        return float("nan")
    return num / (denom_x * denom_y)


def _action_to_onehot(action_idx: int, num_actions: int, device) -> torch.Tensor:
    v = torch.zeros(1, num_actions, device=device)
    v[0, action_idx] = 1.0
    return v


# ---------------------------------------------------------------------------
# Single (seed, condition) cell
# ---------------------------------------------------------------------------

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
    dry_run: bool,
) -> Dict:
    """
    Run one (seed, condition) cell and return Phase 1 and Phase 2 metrics.

    beta_mod=True  -> BETA_MOD_ON: update_e3_rate_from_beta() called each step.
    beta_mod=False -> BETA_MOD_OFF: E3 rate fixed at base 10 steps.
    """
    torch.manual_seed(seed)
    random.seed(seed)

    cond_label = "BETA_MOD_ON" if beta_mod else "BETA_MOD_OFF"

    env = CausalGridWorldV2(
        seed=seed,
        size=6,
        num_hazards=4,
        num_resources=4,
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
    config.latent.unified_latent_mode = False  # SD-005 split always on

    agent = REEAgent(config)

    standard_params = [
        p for n, p in agent.named_parameters()
        if "harm_eval_head" not in n
    ]
    harm_eval_params = list(agent.e3.harm_eval_head.parameters())
    optimizer = optim.Adam(standard_params, lr=lr)
    harm_eval_optimizer = optim.Adam(harm_eval_params, lr=1e-4)

    # Balanced harm replay buffer
    harm_buf_pos: List[torch.Tensor] = []
    harm_buf_neg: List[torch.Tensor] = []
    MAX_BUF_EACH = 2000

    # Phase 1 data (BETA_MOD_ON training only): step-level pairs
    p1_beta_norms: List[float] = []   # |z_beta|_norm at each step
    p1_inv_rates:  List[float] = []   # 1 / e3_steps_per_tick at each step

    # ------------------------------------------------------------------ TRAIN
    agent.train()

    for ep in range(warmup_episodes):
        _, obs_dict = env.reset()
        agent.reset()

        episode_harm = 0.0
        cached_harm_score = None
        last_action_idx = None

        for _ in range(steps_per_episode):
            obs_body  = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]

            latent = agent.sense(obs_body, obs_world)
            ticks  = agent.clock.advance()

            # E1 tick (also calls update_e3_rate_from_beta internally)
            if ticks["e1_tick"]:
                agent._e1_tick(latent)
                if not beta_mod:
                    # BETA_MOD_OFF: undo rate change, keep E3 at base 10 steps
                    agent.clock._current_e3_steps = agent.clock._e3_base_steps

            # Phase 1 logging: step-level |z_beta| vs instantaneous inverse rate
            if beta_mod:
                z_beta_norm = float(latent.z_beta.detach().norm(dim=-1).mean().item())
                inv_rate    = 1.0 / max(1, agent.clock.e3_steps_per_tick)
                p1_beta_norms.append(z_beta_norm)
                p1_inv_rates.append(inv_rate)

            # E3 tick: refresh cached harm score
            if ticks["e3_tick"] or cached_harm_score is None:
                z_w = latent.z_world.detach()
                with torch.no_grad():
                    cached_harm_score = float(agent.e3.harm_eval(z_w).item())

            # Policy: flee (random) if harm score high; hold last action otherwise
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
            is_harm  = float(harm_signal) < 0
            if is_harm:
                harm_buf_pos.append(z_w_curr)
                if len(harm_buf_pos) > MAX_BUF_EACH:
                    harm_buf_pos = harm_buf_pos[-MAX_BUF_EACH:]
            else:
                harm_buf_neg.append(z_w_curr)
                if len(harm_buf_neg) > MAX_BUF_EACH:
                    harm_buf_neg = harm_buf_neg[-MAX_BUF_EACH:]

            # E1 + E2 standard loss
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

        if (ep + 1) % 100 == 0 or ep == warmup_episodes - 1:
            n_so_far = len(p1_beta_norms)
            r_snap   = _pearson_r(p1_beta_norms, p1_inv_rates) if beta_mod else float("nan")
            r_str    = f"{r_snap:.3f}" if not math.isnan(r_snap) else "nan"
            print(
                f"  [train] cond={cond_label} seed={seed}"
                f" ep {ep+1}/{warmup_episodes}"
                f" ep_harm={episode_harm:.4f}"
                f" buf_pos={len(harm_buf_pos)} buf_neg={len(harm_buf_neg)}"
                + (f" p1_r={r_str} n={n_so_far}" if beta_mod else ""),
                flush=True,
            )

        if dry_run and ep >= 2:
            break

    # ------------------------------------------------------------------ EVAL
    agent.eval()

    eval_harm_events = 0
    eval_total_steps = 0
    safe_phase_actions: List[int] = []

    for ep in range(eval_episodes):
        _, obs_dict = env.reset()
        agent.reset()

        cached_harm_score  = None
        last_action_idx    = None
        consec_safe_steps  = 0

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

            if consec_safe_steps >= SAFE_WINDOW:
                safe_phase_actions.append(action_idx)

            if done:
                break

        if dry_run and ep >= 1:
            break

    harm_rate = float(eval_harm_events) / max(1, eval_total_steps)

    if len(safe_phase_actions) >= 2:
        mean_act = sum(safe_phase_actions) / len(safe_phase_actions)
        action_var = float(
            sum((a - mean_act) ** 2 for a in safe_phase_actions) / len(safe_phase_actions)
        )
    else:
        action_var = 0.0

    # Phase 1 aggregate
    if beta_mod and len(p1_beta_norms) >= 3:
        p1_r           = _pearson_r(p1_beta_norms, p1_inv_rates)
        n_p1_steps     = len(p1_beta_norms)
        mean_beta_norm = sum(p1_beta_norms) / n_p1_steps
        mean_inv_rate  = sum(p1_inv_rates)  / n_p1_steps
    else:
        p1_r           = float("nan")
        n_p1_steps     = len(p1_beta_norms)
        mean_beta_norm = float("nan")
        mean_inv_rate  = float("nan")

    p1_r_str = f"{p1_r:.3f}" if not math.isnan(p1_r) else "nan"
    print(
        f"  [eval] cond={cond_label} seed={seed}"
        f" harm_rate={harm_rate:.4f}"
        f" action_var_safe={action_var:.4f}"
        f" safe_steps={len(safe_phase_actions)}"
        + (f" p1_r={p1_r_str} n={n_p1_steps}" if beta_mod else ""),
        flush=True,
    )

    return {
        "condition":          cond_label,
        "beta_mod":           beta_mod,
        "seed":               seed,
        "harm_rate":          harm_rate,
        "action_var_safe":    action_var,
        "n_safe_steps":       len(safe_phase_actions),
        "p1_pearson_r":       p1_r,
        "p1_n_steps":         n_p1_steps,
        "p1_mean_beta_norm":  mean_beta_norm,
        "p1_mean_inv_rate":   mean_inv_rate,
    }


# ---------------------------------------------------------------------------
# Discriminative pair: BETA_MOD_ON vs BETA_MOD_OFF x seeds [42, 123]
# ---------------------------------------------------------------------------

def run(
    seeds: Tuple = (42, 123),
    warmup_episodes: int = 500,
    eval_episodes: int = 40,
    steps_per_episode: int = 200,
    self_dim: int = 32,
    world_dim: int = 32,
    lr: float = 1e-3,
    alpha_world: float = 0.9,
    alpha_self: float = 0.3,
    harm_scale: float = 0.02,
    proximity_harm_scale: float = 0.05,
    dry_run: bool = False,
    **kwargs,
) -> dict:
    """
    Matched-seed discriminative pair: BETA_MOD_ON vs BETA_MOD_OFF x seeds.
    C1: step-level Pearson r(|z_beta|, 1/e3_steps_per_tick) >= 0.15 for both seeds.
    C2: action_var_ratio (ON/OFF) <= 0.90 for >= 1 seed.
    C3: r direction positive for both seeds.
    C4: n_steps >= 1000 per seed (data quality).
    PASS: C1_both AND C3 AND C4.
    """
    results_on:  List[Dict] = []
    results_off: List[Dict] = []

    for seed in seeds:
        for beta_mod in [True, False]:
            label = "BETA_MOD_ON" if beta_mod else "BETA_MOD_OFF"
            print(
                f"\n[V3-EXQ-116] {label} seed={seed}"
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
                dry_run=dry_run,
            )
            if beta_mod:
                results_on.append(r)
            else:
                results_off.append(r)

    def _safe_r(r_val: float) -> str:
        return f"{r_val:.3f}" if not math.isnan(r_val) else "nan"

    # --- C1: per-seed Pearson r gate ---
    c1_per_seed = []
    for r_on in results_on:
        p1_r = r_on["p1_pearson_r"]
        n    = r_on["p1_n_steps"]
        if n >= 3 and not math.isnan(p1_r):
            c1_per_seed.append(p1_r >= THRESH_C1_R)
        else:
            c1_per_seed.append(False)
    pair_c1_pass_count = sum(c1_per_seed)
    c1_pass = pair_c1_pass_count == len(seeds)

    # --- C2: action_var_ratio gate (>= 1 seed) ---
    c2_per_seed = []
    for r_on, r_off in zip(results_on, results_off):
        var_on  = r_on["action_var_safe"]
        var_off = r_off["action_var_safe"]
        if var_off > 1e-6:
            ratio = var_on / var_off
            c2_per_seed.append(ratio <= THRESH_C2_RATIO)
        else:
            c2_per_seed.append(False)
    c2_pass = sum(c2_per_seed) >= 1

    # --- C3: direction consistency (r >= 0 for both seeds) ---
    c3_per_seed = []
    for r_on in results_on:
        p1_r = r_on["p1_pearson_r"]
        if not math.isnan(p1_r):
            c3_per_seed.append(p1_r >= 0.0)
        else:
            c3_per_seed.append(False)
    c3_pass = all(c3_per_seed)

    # --- C4: data quality (n_steps >= 1000 per seed) ---
    c4_per_seed = [r["p1_n_steps"] >= THRESH_C4_NSTEPS for r in results_on]
    c4_pass = all(c4_per_seed)

    # --- PASS logic ---
    if c1_pass and c3_pass and c4_pass:
        status = "PASS"
        failure_note = ""
    elif not c4_pass:
        status = "FAIL"
        failure_note = "data quality gate failed (insufficient step-level data)"
    elif not c3_pass:
        status = "FAIL"
        failure_note = "direction inconsistent across seeds -- C3 gate failed"
    else:
        status = "FAIL"
        failure_note = f"C1 gate: {pair_c1_pass_count}/{len(seeds)} seeds meet r>={THRESH_C1_R}"

    # Decision mapping
    if status == "PASS":
        decision = "retain_ree"
    elif not c3_pass:
        decision = "hybridize"
    else:
        decision = "hold_for_more_warmup"

    criteria_met = sum([c1_pass, c2_pass, c3_pass, c4_pass])

    print(f"\n[V3-EXQ-116] Summary:", flush=True)
    for i, (r_on, r_off) in enumerate(zip(results_on, results_off)):
        s = seeds[i] if i < len(seeds) else "?"
        print(
            f"  seed={s}"
            f" ON: p1_r={_safe_r(r_on['p1_pearson_r'])} n={r_on['p1_n_steps']}"
            f" harm_rate={r_on['harm_rate']:.4f}"
            f" action_var={r_on['action_var_safe']:.4f}"
            f"  |  OFF: harm_rate={r_off['harm_rate']:.4f}"
            f" action_var={r_off['action_var_safe']:.4f}",
            flush=True,
        )
    print(
        f"  C1 (both r>={THRESH_C1_R}): {'PASS' if c1_pass else 'FAIL'}"
        f" ({pair_c1_pass_count}/{len(seeds)} seeds)",
        flush=True,
    )
    print(
        f"  C2 (var_ratio<={THRESH_C2_RATIO} for >=1 seed): {'PASS' if c2_pass else 'FAIL'}"
        f" ({sum(c2_per_seed)}/{len(seeds)} seeds)",
        flush=True,
    )
    print(
        f"  C3 (r>=0 both seeds): {'PASS' if c3_pass else 'FAIL'}"
        f" ({sum(c3_per_seed)}/{len(seeds)} seeds)",
        flush=True,
    )
    print(
        f"  C4 (n_steps>={THRESH_C4_NSTEPS} both): {'PASS' if c4_pass else 'FAIL'}"
        f" ({sum(c4_per_seed)}/{len(seeds)} seeds)",
        flush=True,
    )
    print(f"  Criteria met: {criteria_met}/4  Status: {status}  Decision: {decision}", flush=True)
    if failure_note:
        print(f"  Note: {failure_note}", flush=True)

    # --- Failure notes ---
    failure_notes: List[str] = []
    for i, (r_on, r_off) in enumerate(zip(results_on, results_off)):
        s = seeds[i] if i < len(seeds) else "?"
        p1_r = r_on["p1_pearson_r"]
        n    = r_on["p1_n_steps"]
        if not c1_per_seed[i]:
            failure_notes.append(
                f"C1 FAIL seed={s}: p1_r={_safe_r(p1_r)} < {THRESH_C1_R} (n={n})"
            )
        if not c3_per_seed[i]:
            failure_notes.append(
                f"C3 FAIL seed={s}: p1_r={_safe_r(p1_r)} < 0 (direction inverted)"
            )
        var_on  = r_on["action_var_safe"]
        var_off = r_off["action_var_safe"]
        if not c2_per_seed[i]:
            ratio_str = f"{var_on/var_off:.3f}" if var_off > 1e-6 else "inf"
            failure_notes.append(
                f"C2 seed={s}: action_var_ratio={ratio_str}"
                f" (ON={var_on:.4f} OFF={var_off:.4f})"
            )
    for n_st, ok in zip([r["p1_n_steps"] for r in results_on], c4_per_seed):
        if not ok:
            failure_notes.append(
                f"C4 FAIL: n_steps={n_st} < {THRESH_C4_NSTEPS}"
            )

    # --- Interpretation ---
    if status == "PASS":
        interpretation = (
            f"MECH-093 SUPPORTED by multi-seed replication ({len(seeds)} seeds). "
            "Step-level z_beta/rate correlation confirmed for all seeds "
            f"(C1 r>={THRESH_C1_R}), direction consistent (C3), data sufficient (C4). "
            "z_beta encodes arousal state and modulates E3 heartbeat frequency at the "
            "step level. This is distinct from precision-weighting (MECH-059): rate "
            "modulation controls update frequency, not update magnitude."
        )
    elif not c3_pass:
        interpretation = (
            "MECH-093 WEAKENED: direction of z_beta/rate correlation is inconsistent "
            "across seeds. One or more seeds show a negative Pearson r "
            "(higher |z_beta| -> slower rate), which is opposite to the claim. "
            "Environment dependency possible -- z_beta encoding may vary by hazard density."
        )
    elif not c1_pass and c3_pass:
        interpretation = (
            "MECH-093 BELOW THRESHOLD: direction is consistently positive across seeds "
            f"(C3 PASS), but correlation magnitude does not reach r>={THRESH_C1_R} "
            f"for all seeds ({pair_c1_pass_count}/{len(seeds)} seeds pass). "
            "This suggests z_beta/rate coupling exists but is weaker than required for "
            "full support at current architecture scale. "
            "Options: more warmup (z_beta encoding matures with training), larger harm scale "
            "to increase z_beta variance, or explicit harm supervision on z_beta encoder."
        )
    else:
        interpretation = (
            "MECH-093 INCONCLUSIVE: insufficient step-level data to compute Pearson r "
            f"reliably (n_steps < {THRESH_C4_NSTEPS}). Increase warmup_episodes."
        )
    if failure_note:
        interpretation += f" [{failure_note}]"

    # --- Build per-seed rows for markdown ---
    per_seed_rows = "\n".join(
        f"| {seeds[i]} | {_safe_r(results_on[i]['p1_pearson_r'])}"
        f" | {results_on[i]['p1_n_steps']}"
        f" | {'PASS' if c1_per_seed[i] else 'FAIL'}"
        f" | {'PASS' if c3_per_seed[i] else 'FAIL'}"
        f" | {'PASS' if c2_per_seed[i] else '-'} |"
        for i in range(len(seeds))
    )

    failure_section = ""
    if failure_notes:
        failure_section = "\n## Failure Notes\n\n" + "\n".join(
            f"- {n}" for n in failure_notes
        ) + "\n"

    summary_markdown = (
        f"# V3-EXQ-116 -- MECH-093 z_beta Heartbeat Rate Modulation (Multi-Seed)\n\n"
        f"**Status:** {status}\n"
        f"**Claims:** MECH-093\n"
        f"**Proposal:** EXP-0011 / EVB-0009\n"
        f"**Decision:** {decision}\n"
        f"**Seeds:** {list(seeds)}\n"
        f"**alpha_world:** {alpha_world}  (SD-008)\n"
        f"**Warmup:** {warmup_episodes} eps  **Eval:** {eval_episodes} eps"
        f"  **Steps/ep:** {steps_per_episode}\n\n"
        f"## Context\n\n"
        f"EXQ-097b (seed=42 only) PASS 2/3: C1 r=0.964, C3 var_ratio=0.096."
        f" MECH-093 promoted to provisional pending multi-seed replication."
        f" This experiment provides that replication.\n\n"
        f"## Pre-Registered Thresholds\n\n"
        f"C1 (modulation gate): Pearson r(|z_beta|_norm, 1/e3_rate) >= {THRESH_C1_R}"
        f" for BOTH seeds (BETA_MOD_ON)\n"
        f"C2 (safe-phase stability): action_var_ratio (ON/OFF) <= {THRESH_C2_RATIO}"
        f" for >= 1 seed\n"
        f"C3 (direction consistency): r >= 0 for BOTH seeds (no inversion)\n"
        f"C4 (data quality): n_steps >= {THRESH_C4_NSTEPS} per seed\n"
        f"PASS: C1_both AND C3 AND C4\n\n"
        f"## Per-Seed Results (BETA_MOD_ON)\n\n"
        f"| Seed | Pearson r | n_steps | C1 | C3 | C2 |\n"
        f"|------|-----------|---------|----|----|----|\n"
        f"{per_seed_rows}\n\n"
        f"## Aggregate\n\n"
        f"| Criterion | Result | Detail |\n"
        f"|-----------|--------|--------|\n"
        f"| C1: r>={THRESH_C1_R} both seeds | {'PASS' if c1_pass else 'FAIL'}"
        f" | {pair_c1_pass_count}/{len(seeds)} seeds |\n"
        f"| C2: var_ratio<={THRESH_C2_RATIO} >=1 seed | {'PASS' if c2_pass else 'FAIL'}"
        f" | {sum(c2_per_seed)}/{len(seeds)} seeds |\n"
        f"| C3: direction consistent | {'PASS' if c3_pass else 'FAIL'}"
        f" | {sum(c3_per_seed)}/{len(seeds)} seeds |\n"
        f"| C4: data quality | {'PASS' if c4_pass else 'FAIL'}"
        f" | {sum(c4_per_seed)}/{len(seeds)} seeds |\n\n"
        f"Criteria met: {criteria_met}/4 -> **{status}**\n\n"
        f"## Interpretation\n\n"
        f"{interpretation}\n"
        f"{failure_section}"
    )

    # Flat metrics (stable numeric keys)
    per_seed_r_on = {
        f"seed{results_on[i]['seed']}_p1_pearson_r": (
            float(results_on[i]["p1_pearson_r"])
            if not math.isnan(results_on[i]["p1_pearson_r"])
            else -99.0
        )
        for i in range(len(results_on))
    }
    per_seed_n_steps = {
        f"seed{results_on[i]['seed']}_p1_n_steps": float(results_on[i]["p1_n_steps"])
        for i in range(len(results_on))
    }
    per_seed_var_ratio = {}
    for i, (r_on, r_off) in enumerate(zip(results_on, results_off)):
        s = seeds[i] if i < len(seeds) else i
        var_on  = r_on["action_var_safe"]
        var_off = r_off["action_var_safe"]
        per_seed_var_ratio[f"seed{s}_action_var_ratio"] = (
            float(var_on / var_off) if var_off > 1e-6 else 99.0
        )

    metrics: Dict = {
        "pair_c1_pass_count":   float(pair_c1_pass_count),
        "n_seeds":              float(len(seeds)),
        "c1_threshold":         float(THRESH_C1_R),
        "c2_threshold":         float(THRESH_C2_RATIO),
        "c4_threshold_nsteps":  float(THRESH_C4_NSTEPS),
        "c2_seeds_passing":     float(sum(c2_per_seed)),
        "c3_seeds_passing":     float(sum(c3_per_seed)),
        "criteria_met":         float(criteria_met),
        "crit1_pass":           1.0 if c1_pass else 0.0,
        "crit2_pass":           1.0 if c2_pass else 0.0,
        "crit3_pass":           1.0 if c3_pass else 0.0,
        "crit4_pass":           1.0 if c4_pass else 0.0,
        "alpha_world":          float(alpha_world),
        "warmup_episodes":      float(warmup_episodes),
        "eval_episodes":        float(eval_episodes),
        **per_seed_r_on,
        **per_seed_n_steps,
        **per_seed_var_ratio,
    }

    return {
        "status":              status,
        "metrics":             metrics,
        "summary_markdown":    summary_markdown,
        "claim_ids":           CLAIM_IDS,
        "evidence_direction":  "supports" if status == "PASS" else (
            "mixed" if c3_pass and criteria_met >= 2 else "weakens"
        ),
        "experiment_type":     EXPERIMENT_TYPE,
        "fatal_error_count":   0,
        "supersedes":          "v3_exq_097b_mech093_heartbeat_rate",
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import json
    from datetime import datetime, timezone

    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds",           type=int, nargs="+", default=[42, 123])
    parser.add_argument("--warmup",          type=int,   default=500)
    parser.add_argument("--eval-eps",        type=int,   default=40)
    parser.add_argument("--steps",           type=int,   default=200)
    parser.add_argument("--alpha-world",     type=float, default=0.9)
    parser.add_argument("--alpha-self",      type=float, default=0.3)
    parser.add_argument("--harm-scale",      type=float, default=0.02)
    parser.add_argument("--proximity-scale", type=float, default=0.05)
    parser.add_argument("--dry-run",         action="store_true",
                        help="Run 3 warmup + 2 eval episodes per cell to check for errors.")
    args = parser.parse_args()

    result = run(
        seeds=tuple(args.seeds),
        warmup_episodes=args.warmup,
        eval_episodes=args.eval_eps,
        steps_per_episode=args.steps,
        alpha_world=args.alpha_world,
        alpha_self=args.alpha_self,
        harm_scale=args.harm_scale,
        proximity_harm_scale=args.proximity_scale,
        dry_run=args.dry_run,
    )

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result["run_timestamp"]      = ts
    result["claim"]              = CLAIM_IDS[0]
    result["verdict"]            = result["status"]
    result["run_id"]             = f"{EXPERIMENT_TYPE}_{ts}_v3"
    result["architecture_epoch"] = "ree_hybrid_guardrails_v1"
    result["registered_thresholds"] = {
        "C1_pearson_r_per_seed":       THRESH_C1_R,
        "C2_action_var_ratio_leq":     THRESH_C2_RATIO,
        "C3_direction_consistent_r_geq_0": 0.0,
        "C4_n_steps_per_seed_geq":     THRESH_C4_NSTEPS,
    }
    result["seeds"]              = list(args.seeds)
    result["conditions"]         = ["BETA_MOD_ON", "BETA_MOD_OFF"]
    result["dispatch_mode"]      = "discriminative_pair"
    result["backlog_id"]         = "EVB-0009"
    result["claim_ids_tested"]   = CLAIM_IDS
    result["evidence_class"]     = "experiment"

    if args.dry_run:
        print(f"\n[dry-run] Status: {result['status']}", flush=True)
        print("[dry-run] Key metrics:", flush=True)
        for k, v in result["metrics"].items():
            print(f"  {k}: {v}", flush=True)
        print("[dry-run] Script ran without error. Not writing output file.", flush=True)
        sys.exit(0)

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
