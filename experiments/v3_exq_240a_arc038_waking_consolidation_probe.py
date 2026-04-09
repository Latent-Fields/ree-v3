#!/opt/local/bin/python3
"""
V3-EXQ-240a -- ARC-038 Waking Consolidation Mode Probe (substrate fix re-run)

Claims: ARC-038

Supersedes: V3-EXQ-240 (ERROR: shape mismatch in hippocampal.replay() ->
e2.predict_next_self() — z_self_zeros had world_dim instead of self_dim.
Fixed in substrate; script unchanged.)

Scientific question: Does a waking consolidation pass (offline map-geometry
update from recent trajectory buffer, no z_goal required) produce measurable
benefit over an agent that skips the consolidation pass?

ARC-038 asserts: "Waking consolidation mode -- the hippocampal system runs a
map-update replay pass during quiescent periods WITHOUT an active z_goal,
updating map geometry from recent trajectory experience. This is architecturally
distinct from goal-directed forward sweeps."

MECH-092 asserts: "Quiescent E3 heartbeat cycles trigger hippocampal SWR-
equivalent replay for viability map consolidation."

This experiment is classified as DIAGNOSTIC because the waking consolidation
mode is implemented via the existing replay() + integrate() hooks in the current
substrate. The diagnostic question: does calling these offline update hooks
periodically (no z_goal required) produce measurable benefit?

Design:
  Two conditions, same agent architecture, same total gradient step budget:

  CONSOLIDATE:
    After every K=10 episodes, run a "consolidation pass":
      (1) Call hippocampal.replay() on recent z_world trajectory buffer.
          This runs replay trajectories through E2 without any z_goal signal.
      (2) Call residue_field.integrate() to update the neural field using
          recent harm history (offline contextualisation step).
    The consolidation steps count toward the total gradient step budget.
    Consolidation is hypothesis_tag=True at the replay level (MECH-094 gate
    enforced by replay()). Only residue_field.integrate() may update weights.

  NO_CONSOLIDATE:
    Same agent but consolidation pass is skipped entirely.
    No replay(), no integrate() calls between episodes.

Both conditions:
  - Trained for training_episodes episodes on CausalGridWorldV2.
  - Random action selection during training (builds experience buffer).
  - Standard E1/E2/harm_eval online training throughout.
  - Action selection at eval uses harm_eval-based greedy selection.

Metrics per seed:
  - harm_rate_consolidate / harm_rate_no_consolidate (per eval period)
  - acquisition_speed: episodes where 10-episode block mean harm_rate < 0.05
  - consolidation_benefit_ratio = harm_rate_no_consolidate / harm_rate_consolidate
  - map_stability: variance in residue field total_residue between episodes
    (lower = more stable/consolidated map geometry)

PASS criteria (majority over 5 seeds):
  C1: harm_rate_consolidate < harm_rate_no_consolidate (final 20-ep block mean)
      -- met in >= 3/5 seeds
  C2: acquisition_speed_consolidate < acquisition_speed_no_consolidate
      -- met in >= 3/5 seeds
  C3: consolidation_benefit_ratio > 1.1 (>10% benefit)
      -- met in >= 3/5 seeds (using final block means)

PASS: C1 AND (C2 OR C3)
FAIL otherwise

EXPERIMENT_PURPOSE: "diagnostic"
  Rationale: waking consolidation mode not yet a first-class architectural
  component (no dedicated consolidation-mode state machine). This experiment
  uses the existing replay() + integrate() hooks as a proxy. A PASS establishes
  measurable offline-update benefit. A FAIL diagnoses whether the substrate
  lacks sufficient consolidation machinery.

target_harm_rate: 0.05 (reachable -- recent V3 experiments show harm rates
  0.10-0.20 with training, so 0.05 is achievable within budget).
"""

import sys
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
import torch.optim as optim

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig


EXPERIMENT_TYPE = "v3_exq_240a_arc038_waking_consolidation_probe"
CLAIM_IDS = ["ARC-038"]
EXPERIMENT_PURPOSE = "diagnostic"

TARGET_HARM_RATE = 0.05
CONSOLIDATE_INTERVAL = 10      # episodes between consolidation passes
REPLAY_STEPS = 20              # num_replay_steps per consolidation call
INTEGRATE_STEPS = 5            # num_steps for residue_field.integrate()
RECENT_BUFFER_STEPS = 50       # how many recent z_world steps to pass to replay()
BLOCK_SIZE = 20                # episodes per evaluation block


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #

def _action_to_onehot(action_idx: int, num_actions: int, device) -> torch.Tensor:
    v = torch.zeros(1, num_actions, device=device)
    v[0, action_idx] = 1.0
    return v


def _select_action_harm_avoid(
    agent: REEAgent,
    z_world: torch.Tensor,
    num_actions: int,
) -> int:
    """Pick action with lowest predicted harm (E3 harm_eval on E2 world_forward)."""
    with torch.no_grad():
        best_action = 0
        best_score = float("inf")
        for idx in range(num_actions):
            a_oh = _action_to_onehot(idx, num_actions, z_world.device)
            z_world_next = agent.e2.world_forward(z_world, a_oh)
            harm_score = agent.e3.harm_eval(z_world_next).mean().item()
            if harm_score < best_score:
                best_score = harm_score
                best_action = idx
    return best_action


def _make_env(env_seed: int, grid_size: int = 10) -> CausalGridWorldV2:
    """Create a CausalGridWorldV2 with standard parameters."""
    return CausalGridWorldV2(
        seed=env_seed,
        size=grid_size,
        num_hazards=4,
        num_resources=4,
        hazard_harm=0.05,
        env_drift_interval=5,
        env_drift_prob=0.1,
        proximity_harm_scale=0.15,
        proximity_benefit_scale=0.18,
        proximity_approach_threshold=0.15,
        hazard_field_decay=0.5,
        energy_decay=0.005,
        use_proxy_fields=True,
        resource_respawn_on_consume=True,
    )


def _make_agent(env: CausalGridWorldV2, self_dim: int, world_dim: int) -> REEAgent:
    """Create a fresh REEAgent configured for the given environment."""
    config = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=self_dim,
        world_dim=world_dim,
        alpha_world=0.9,
        alpha_self=0.3,
        reafference_action_dim=0,
        novelty_bonus_weight=0.0,
    )
    return REEAgent(config)


# ------------------------------------------------------------------ #
# Consolidation pass (the waking-consolidation mode proxy)            #
# ------------------------------------------------------------------ #

def _run_consolidation_pass(
    agent: REEAgent,
    z_world_history: List[torch.Tensor],
    replay_steps: int,
    integrate_steps: int,
    recent_buffer_steps: int,
) -> Dict[str, float]:
    """
    Waking consolidation pass: offline map-geometry update.

    Does NOT require z_goal. Uses recent z_world trajectory buffer.

    Steps:
    1. Build theta_buffer_recent from z_world_history (no z_goal signal).
    2. Call hippocampal.replay() -- runs E2 rollouts from recent z_world,
       generating replay trajectories (hypothesis_tag=True by MECH-092/094).
    3. Call residue_field.integrate() -- updates neural field weights from
       harm history (offline contextualisation, no residue accumulation).

    Returns dict of consolidation metrics.
    """
    if not z_world_history:
        return {"replay_trajectories": 0, "integration_loss": 0.0}

    # Build theta_buffer_recent: take last N z_world states as [T, batch, world_dim]
    recent = z_world_history[-recent_buffer_steps:]
    # Each element is [1, world_dim]; stack -> [T, 1, world_dim]
    theta_buf = torch.stack(recent, dim=0)  # [T, 1, world_dim]

    # (1) Replay pass: hypothesis_tag=True enforced by MECH-094 inside replay().
    # replay() calls e2.rollout_with_world() without any z_goal input.
    with torch.no_grad():
        replay_trajs = agent.hippocampal.replay(
            theta_buffer_recent=theta_buf,
            num_replay_steps=replay_steps,
        )

    # (2) Residue field offline integration: update neural field from harm history.
    # This is not gated by hypothesis_tag (it reads existing harm_history, not
    # current replay content). It contextualises the neural field weights to
    # better approximate the RBF field over accumulated harm locations.
    integrate_metrics = agent.residue_field.integrate(num_steps=integrate_steps)

    return {
        "replay_trajectories": len(replay_trajs),
        "integration_loss": float(integrate_metrics.get("integration_loss", 0.0)),
        "history_size": int(integrate_metrics.get("history_size", 0)),
    }


# ------------------------------------------------------------------ #
# Training + evaluation loop                                          #
# ------------------------------------------------------------------ #

def _run_condition(
    agent: REEAgent,
    env: CausalGridWorldV2,
    training_episodes: int,
    steps_per_episode: int,
    condition_label: str,
    use_consolidation: bool,
    consolidate_interval: int,
    replay_steps: int,
    integrate_steps: int,
    recent_buffer_steps: int,
) -> Dict:
    """
    Run a single condition (CONSOLIDATE or NO_CONSOLIDATE).

    Training phase: random action selection, standard E1/E2/harm_eval losses.
    After every consolidate_interval episodes (CONSOLIDATE only): run
    consolidation pass.
    Returns per-episode harm rates and residue field stability metrics.
    """
    action_dim = env.action_dim

    standard_params = [p for n, p in agent.named_parameters()
                       if "harm_eval_head" not in n]
    harm_eval_params = list(agent.e3.harm_eval_head.parameters())

    optimizer = optim.Adam(standard_params, lr=1e-3)
    harm_eval_opt = optim.Adam(harm_eval_params, lr=1e-4)

    # Stratified harm replay buffer
    harm_buf_pos: List[torch.Tensor] = []
    harm_buf_neg: List[torch.Tensor] = []
    MAX_BUF = 4000

    # Recent z_world history for consolidation pass
    z_world_history: List[torch.Tensor] = []

    agent.train()
    per_episode_harm: List[float] = []
    residue_totals: List[float] = []     # for map_stability computation
    consolidation_log: List[Dict] = []

    for ep in range(training_episodes):
        _, obs_dict = env.reset()
        agent.reset()
        ep_harm_sum = 0.0
        ep_steps = 0

        for _step_i in range(steps_per_episode):
            obs_body  = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            obs_harm  = obs_dict.get("harm_obs", None)

            latent = agent.sense(obs_body, obs_world, obs_harm=obs_harm)
            agent.clock.advance()
            z_world_curr = latent.z_world.detach().clone()

            # Collect z_world for consolidation buffer
            z_world_history.append(z_world_curr)
            if len(z_world_history) > 2000:
                z_world_history = z_world_history[-2000:]

            # Action selection: harm_eval-based greedy (not random, so we can
            # measure actual harm-avoidance skill development)
            action_idx = _select_action_harm_avoid(agent, z_world_curr, action_dim)
            action_oh = _action_to_onehot(action_idx, action_dim, agent.device)
            agent._last_action = action_oh

            _, harm_signal, done, _info, obs_dict = env.step(action_oh)

            # Standard E1 + E2 training
            e1_loss = agent.compute_prediction_loss()
            e2_loss = agent.compute_e2_loss()
            total_loss = e1_loss + e2_loss
            if total_loss.requires_grad:
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                optimizer.step()

            # harm_eval stratified training
            is_harm = float(harm_signal) < 0
            if is_harm:
                harm_buf_pos.append(z_world_curr)
                if len(harm_buf_pos) > MAX_BUF:
                    harm_buf_pos = harm_buf_pos[-MAX_BUF:]
            else:
                harm_buf_neg.append(z_world_curr)
                if len(harm_buf_neg) > MAX_BUF:
                    harm_buf_neg = harm_buf_neg[-MAX_BUF:]

            if len(harm_buf_pos) >= 4 and len(harm_buf_neg) >= 4:
                k_pos = min(16, len(harm_buf_pos))
                k_neg = min(16, len(harm_buf_neg))
                pos_idx = torch.randperm(len(harm_buf_pos))[:k_pos].tolist()
                neg_idx = torch.randperm(len(harm_buf_neg))[:k_neg].tolist()
                zw_pos = torch.cat([harm_buf_pos[i] for i in pos_idx], dim=0)
                zw_neg = torch.cat([harm_buf_neg[i] for i in neg_idx], dim=0)
                zw_b   = torch.cat([zw_pos, zw_neg], dim=0)
                target = torch.cat([
                    torch.ones(k_pos, 1, device=agent.device),
                    torch.zeros(k_neg, 1, device=agent.device),
                ], dim=0)
                pred_harm = agent.e3.harm_eval(zw_b)
                harm_loss = F.binary_cross_entropy(pred_harm, target)
                if harm_loss.requires_grad:
                    harm_eval_opt.zero_grad()
                    harm_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        agent.e3.harm_eval_head.parameters(), 0.5,
                    )
                    harm_eval_opt.step()

            if float(harm_signal) < 0:
                ep_harm_sum += abs(float(harm_signal))
            ep_steps += 1

            if done:
                break

        harm_rate = ep_harm_sum / max(1, ep_steps)
        per_episode_harm.append(harm_rate)

        # Record residue total for map_stability
        residue_stats = agent.residue_field.get_statistics()
        residue_totals.append(float(residue_stats["total_residue"].item()))

        # Consolidation pass: every K episodes, CONSOLIDATE condition only
        if use_consolidation and (ep + 1) % consolidate_interval == 0:
            clog = _run_consolidation_pass(
                agent=agent,
                z_world_history=z_world_history,
                replay_steps=replay_steps,
                integrate_steps=integrate_steps,
                recent_buffer_steps=recent_buffer_steps,
            )
            clog["episode"] = ep + 1
            consolidation_log.append(clog)

        if (ep + 1) % 50 == 0 or ep == training_episodes - 1:
            block_start = max(0, ep - 9)
            recent_mean = (
                sum(per_episode_harm[block_start:]) /
                max(1, ep + 1 - block_start)
            )
            n_cons = len(consolidation_log)
            print(
                f"    [{condition_label}] ep {ep+1}/{training_episodes}"
                f" harm_rate={harm_rate:.5f}"
                f" recent_10_mean={recent_mean:.5f}"
                f" consolidations={n_cons}",
                flush=True,
            )

    return {
        "per_episode_harm": per_episode_harm,
        "residue_totals": residue_totals,
        "consolidation_log": consolidation_log,
        "harm_buf_pos_size": len(harm_buf_pos),
        "harm_buf_neg_size": len(harm_buf_neg),
    }


# ------------------------------------------------------------------ #
# Analysis helpers                                                     #
# ------------------------------------------------------------------ #

def _episodes_to_target(per_ep_harm: List[float], target: float, block: int) -> int:
    """First episode index (1-based) where block mean < target. Returns n if never."""
    n = len(per_ep_harm)
    for i in range(block - 1, n):
        bm = sum(per_ep_harm[i - block + 1 : i + 1]) / block
        if bm < target:
            return i + 1
    return n


def _final_block_mean(per_ep_harm: List[float], block: int) -> float:
    """Mean of last block episodes."""
    if len(per_ep_harm) < block:
        return sum(per_ep_harm) / max(1, len(per_ep_harm))
    return sum(per_ep_harm[-block:]) / block


def _map_stability(residue_totals: List[float]) -> float:
    """
    Variance of residue field total across episodes.
    Lower = more stable / well-consolidated map geometry.
    Uses last half of episodes to measure post-learning stability.
    """
    if len(residue_totals) < 4:
        return float("nan")
    half = residue_totals[len(residue_totals) // 2:]
    if not half:
        return float("nan")
    mean = sum(half) / len(half)
    var = sum((x - mean) ** 2 for x in half) / len(half)
    return var


# ------------------------------------------------------------------ #
# Single seed run                                                      #
# ------------------------------------------------------------------ #

def _run_single(
    experiment_seed: int,
    env_seed: int,
    training_episodes: int,
    steps_per_episode: int,
    target_harm_rate: float,
    block_size: int,
    self_dim: int,
    world_dim: int,
    consolidate_interval: int,
    replay_steps: int,
    integrate_steps: int,
    recent_buffer_steps: int,
) -> Dict:
    torch.manual_seed(experiment_seed)
    random.seed(experiment_seed)

    # ---- CONSOLIDATE condition ----
    print(f"  [CONSOLIDATE] env_seed={env_seed}...", flush=True)
    env_c = _make_env(env_seed)
    agent_c = _make_agent(env_c, self_dim, world_dim)

    results_c = _run_condition(
        agent=agent_c,
        env=env_c,
        training_episodes=training_episodes,
        steps_per_episode=steps_per_episode,
        condition_label="CONSOLIDATE",
        use_consolidation=True,
        consolidate_interval=consolidate_interval,
        replay_steps=replay_steps,
        integrate_steps=integrate_steps,
        recent_buffer_steps=recent_buffer_steps,
    )

    # ---- NO_CONSOLIDATE condition ----
    torch.manual_seed(experiment_seed + 10000)
    random.seed(experiment_seed + 10000)

    print(f"  [NO_CONSOLIDATE] env_seed={env_seed}...", flush=True)
    env_nc = _make_env(env_seed)
    agent_nc = _make_agent(env_nc, self_dim, world_dim)

    results_nc = _run_condition(
        agent=agent_nc,
        env=env_nc,
        training_episodes=training_episodes,
        steps_per_episode=steps_per_episode,
        condition_label="NO_CONSOLIDATE",
        use_consolidation=False,
        consolidate_interval=consolidate_interval,
        replay_steps=replay_steps,
        integrate_steps=integrate_steps,
        recent_buffer_steps=recent_buffer_steps,
    )

    # ---- Compute metrics ----
    harm_c  = results_c["per_episode_harm"]
    harm_nc = results_nc["per_episode_harm"]

    final_c  = _final_block_mean(harm_c,  block_size)
    final_nc = _final_block_mean(harm_nc, block_size)

    acq_c  = _episodes_to_target(harm_c,  target_harm_rate, block_size)
    acq_nc = _episodes_to_target(harm_nc, target_harm_rate, block_size)

    benefit_ratio = final_nc / max(1e-9, final_c)

    stability_c  = _map_stability(results_c["residue_totals"])
    stability_nc = _map_stability(results_nc["residue_totals"])

    # Per-criterion booleans
    c1_seed = final_c < final_nc
    c2_seed = acq_c < acq_nc
    c3_seed = benefit_ratio > 1.1

    return {
        "experiment_seed":            experiment_seed,
        "env_seed":                   env_seed,
        "consolidate_final_harm":     float(final_c),
        "no_consolidate_final_harm":  float(final_nc),
        "consolidate_acq_speed":      int(acq_c),
        "no_consolidate_acq_speed":   int(acq_nc),
        "consolidation_benefit_ratio": float(benefit_ratio),
        "map_stability_consolidate":  float(stability_c)   if stability_c == stability_c else None,
        "map_stability_no_consolidate": float(stability_nc) if stability_nc == stability_nc else None,
        "consolidation_passes":       len(results_c["consolidation_log"]),
        "c1_seed":                    c1_seed,
        "c2_seed":                    c2_seed,
        "c3_seed":                    c3_seed,
        "harm_curve_consolidate":     [round(h, 6) for h in harm_c],
        "harm_curve_no_consolidate":  [round(h, 6) for h in harm_nc],
    }


# ------------------------------------------------------------------ #
# Main run() function                                                  #
# ------------------------------------------------------------------ #

def run(
    seeds: Tuple = (42, 7, 123, 0, 99),
    env_seeds: Tuple = (100, 200, 300, 400, 500),
    training_episodes: int = 200,
    steps_per_episode: int = 200,
    target_harm_rate: float = TARGET_HARM_RATE,
    block_size: int = BLOCK_SIZE,
    self_dim: int = 16,
    world_dim: int = 32,
    consolidate_interval: int = CONSOLIDATE_INTERVAL,
    replay_steps: int = REPLAY_STEPS,
    integrate_steps: int = INTEGRATE_STEPS,
    recent_buffer_steps: int = RECENT_BUFFER_STEPS,
    **kwargs,
) -> dict:
    all_results: List[Dict] = []

    for i, seed in enumerate(seeds):
        env_seed = env_seeds[i % len(env_seeds)]
        print(
            f"\n[V3-EXQ-240a] seed={seed} env_seed={env_seed}"
            f" train_eps={training_episodes} steps={steps_per_episode}"
            f" consolidate_every={consolidate_interval}",
            flush=True,
        )
        r = _run_single(
            experiment_seed=seed,
            env_seed=env_seed,
            training_episodes=training_episodes,
            steps_per_episode=steps_per_episode,
            target_harm_rate=target_harm_rate,
            block_size=block_size,
            self_dim=self_dim,
            world_dim=world_dim,
            consolidate_interval=consolidate_interval,
            replay_steps=replay_steps,
            integrate_steps=integrate_steps,
            recent_buffer_steps=recent_buffer_steps,
        )
        all_results.append(r)

    n = len(all_results)

    # Majority-vote criteria: met in >= 3/5 seeds (or >= ceil(n/2)+1 generally)
    threshold = max(3, (n // 2) + 1)

    c1_count = sum(1 for r in all_results if r["c1_seed"])
    c2_count = sum(1 for r in all_results if r["c2_seed"])
    c3_count = sum(1 for r in all_results if r["c3_seed"])

    c1_pass = c1_count >= threshold
    c2_pass = c2_count >= threshold
    c3_pass = c3_count >= threshold

    all_pass = c1_pass and (c2_pass or c3_pass)
    criteria_met = sum([c1_pass, c2_pass, c3_pass])
    status = "PASS" if all_pass else "FAIL"

    avg_final_c  = sum(r["consolidate_final_harm"] for r in all_results) / n
    avg_final_nc = sum(r["no_consolidate_final_harm"] for r in all_results) / n
    avg_acq_c    = sum(r["consolidate_acq_speed"] for r in all_results) / n
    avg_acq_nc   = sum(r["no_consolidate_acq_speed"] for r in all_results) / n
    avg_benefit  = sum(r["consolidation_benefit_ratio"] for r in all_results) / n
    avg_passes   = sum(r["consolidation_passes"] for r in all_results) / n

    # Determine evidence direction and decision
    if all_pass:
        evidence_direction = "supports"
        decision = "retain_ree"
    elif criteria_met >= 2:
        evidence_direction = "mixed"
        decision = "hybridize"
    else:
        evidence_direction = "does_not_support"
        decision = "inconclusive_diagnostic"

    # Diagnostic notes: what the substrate revealed
    diagnostic_notes: List[str] = []
    if not c1_pass:
        diagnostic_notes.append(
            f"C1 FAIL ({c1_count}/{n} seeds): CONSOLIDATE final harm"
            f" {avg_final_c:.5f} not < NO_CONSOLIDATE {avg_final_nc:.5f}."
            " Offline replay+integrate not improving harm-avoidance performance."
        )
    if not c2_pass:
        diagnostic_notes.append(
            f"C2 FAIL ({c2_count}/{n} seeds): acquisition speed"
            f" CONSOLIDATE={avg_acq_c:.1f} not < NO_CONSOLIDATE={avg_acq_nc:.1f}."
        )
    if not c3_pass:
        diagnostic_notes.append(
            f"C3 FAIL ({c3_count}/{n} seeds): benefit_ratio={avg_benefit:.3f}"
            f" not > 1.1 threshold."
        )
    if all_pass:
        diagnostic_notes.append(
            f"All criteria met: waking consolidation proxy (replay+integrate)"
            f" produces measurable benefit. avg_benefit_ratio={avg_benefit:.3f}."
        )

    print(f"\n[V3-EXQ-240a] Final results:", flush=True)
    print(
        f"  CONSOLIDATE:    final_harm={avg_final_c:.5f}"
        f" acq_speed={avg_acq_c:.1f}"
        f" avg_passes={avg_passes:.1f}",
        flush=True,
    )
    print(
        f"  NO_CONSOLIDATE: final_harm={avg_final_nc:.5f}"
        f" acq_speed={avg_acq_nc:.1f}",
        flush=True,
    )
    print(f"  benefit_ratio={avg_benefit:.3f}", flush=True)
    print(
        f"  C1={c1_pass}({c1_count}/{n})"
        f" C2={c2_pass}({c2_count}/{n})"
        f" C3={c3_pass}({c3_count}/{n})"
        f" -> {status}",
        flush=True,
    )
    print(f"  decision={decision}", flush=True)
    for note in diagnostic_notes:
        print(f"  NOTE: {note}", flush=True)

    # Per-seed table for summary
    per_seed_rows = "\n".join(
        f"  seed={r['experiment_seed']} env={r['env_seed']}:"
        f" c_final={r['consolidate_final_harm']:.5f}"
        f" nc_final={r['no_consolidate_final_harm']:.5f}"
        f" ratio={r['consolidation_benefit_ratio']:.3f}"
        f" c_acq={r['consolidate_acq_speed']}"
        f" nc_acq={r['no_consolidate_acq_speed']}"
        f" passes={r['consolidation_passes']}"
        for r in all_results
    )

    diagnostic_section = ""
    if diagnostic_notes:
        diagnostic_section = "\n## Diagnostic Notes\n\n" + "\n".join(
            f"- {n_}" for n_ in diagnostic_notes
        )

    summary_markdown = (
        f"# V3-EXQ-240a -- ARC-038 Waking Consolidation Mode Probe\n\n"
        f"**Status:** {status}\n"
        f"**Claims:** ARC-038\n"
        f"**Decision:** {decision}\n"
        f"**Purpose:** {EXPERIMENT_PURPOSE}\n"
        f"**Seeds:** {list(seeds)}\n\n"
        f"## Design\n\n"
        f"CONSOLIDATE vs NO_CONSOLIDATE: after every K={consolidate_interval}"
        f" episodes, CONSOLIDATE condition runs hippocampal.replay() (no z_goal)"
        f" + residue_field.integrate() (offline map-geometry update)."
        f" NO_CONSOLIDATE skips these calls entirely.\n\n"
        f"target_harm_rate={target_harm_rate} (reachable).\n"
        f"Majority rule: >= {threshold}/{n} seeds required per criterion.\n\n"
        f"## Results\n\n"
        f"| Condition | Final Harm Rate | Acq Speed | Benefit Ratio |\n"
        f"|---|---|---|---|\n"
        f"| CONSOLIDATE | {avg_final_c:.5f} | {avg_acq_c:.1f} | {avg_benefit:.3f} |\n"
        f"| NO_CONSOLIDATE | {avg_final_nc:.5f} | {avg_acq_nc:.1f} | (baseline) |\n\n"
        f"## PASS Criteria\n\n"
        f"| Criterion | Result | Seeds Met | Detail |\n"
        f"|---|---|---|---|\n"
        f"| C1: CONSOLIDATE final_harm < NO_CONSOLIDATE"
        f" | {'PASS' if c1_pass else 'FAIL'}"
        f" | {c1_count}/{n}"
        f" | {avg_final_c:.5f} vs {avg_final_nc:.5f} |\n"
        f"| C2: CONSOLIDATE acq_speed < NO_CONSOLIDATE"
        f" | {'PASS' if c2_pass else 'FAIL'}"
        f" | {c2_count}/{n}"
        f" | {avg_acq_c:.1f} vs {avg_acq_nc:.1f} |\n"
        f"| C3: benefit_ratio > 1.1"
        f" | {'PASS' if c3_pass else 'FAIL'}"
        f" | {c3_count}/{n}"
        f" | ratio={avg_benefit:.3f} |\n\n"
        f"PASS = C1 AND (C2 OR C3). Status: **{status}**\n\n"
        f"## Per-Seed\n\n{per_seed_rows}\n"
        f"{diagnostic_section}\n"
    )

    metrics = {
        "consolidate_final_harm_rate":    float(avg_final_c),
        "no_consolidate_final_harm_rate": float(avg_final_nc),
        "consolidate_acq_speed":          float(avg_acq_c),
        "no_consolidate_acq_speed":       float(avg_acq_nc),
        "consolidation_benefit_ratio":    float(avg_benefit),
        "consolidation_passes_avg":       float(avg_passes),
        "target_harm_rate":               float(target_harm_rate),
        "block_size":                     float(block_size),
        "consolidate_interval":           float(consolidate_interval),
        "training_episodes":              float(training_episodes),
        "n_seeds":                        float(n),
        "crit1_pass":                     1.0 if c1_pass else 0.0,
        "crit2_pass":                     1.0 if c2_pass else 0.0,
        "crit3_pass":                     1.0 if c3_pass else 0.0,
        "criteria_met":                   float(criteria_met),
        "majority_threshold":             float(threshold),
    }

    per_seed_summary = [
        {
            "experiment_seed":             r["experiment_seed"],
            "env_seed":                    r["env_seed"],
            "consolidate_final_harm":      r["consolidate_final_harm"],
            "no_consolidate_final_harm":   r["no_consolidate_final_harm"],
            "consolidate_acq_speed":       r["consolidate_acq_speed"],
            "no_consolidate_acq_speed":    r["no_consolidate_acq_speed"],
            "consolidation_benefit_ratio": r["consolidation_benefit_ratio"],
            "consolidation_passes":        r["consolidation_passes"],
            "c1_seed":                     r["c1_seed"],
            "c2_seed":                     r["c2_seed"],
            "c3_seed":                     r["c3_seed"],
        }
        for r in all_results
    ]

    return {
        "status":              status,
        "metrics":             metrics,
        "summary_markdown":    summary_markdown,
        "claim_ids":           CLAIM_IDS,
        "experiment_purpose":  EXPERIMENT_PURPOSE,
        "per_seed_results":    per_seed_summary,
        "evidence_direction":  evidence_direction,
        "experiment_type":     EXPERIMENT_TYPE,
        "fatal_error_count":   0,
        "diagnostic_notes":    diagnostic_notes,
    }


if __name__ == "__main__":
    import argparse
    import json
    from datetime import datetime, timezone

    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds",     type=int, nargs="+", default=[42, 7, 123, 0, 99])
    parser.add_argument("--train-eps", type=int, default=200)
    parser.add_argument("--steps",     type=int, default=200)
    parser.add_argument("--dry-run",   action="store_true",
                        help="Run minimal config for smoke test")
    args = parser.parse_args()

    if args.dry_run:
        seeds_use = (42,)
        env_seeds_use = (100,)
        train_eps = 5
        steps     = 50
        print("[DRY-RUN] 1 seed, 5 train_eps, 50 steps/ep", flush=True)
    else:
        seeds_use     = tuple(args.seeds)
        env_seeds_use = (100, 200, 300, 400, 500)
        train_eps     = args.train_eps
        steps         = args.steps

    result = run(
        seeds=seeds_use,
        env_seeds=env_seeds_use,
        training_episodes=train_eps,
        steps_per_episode=steps,
    )

    if args.dry_run:
        print("\n[DRY-RUN] Smoke test complete. Not writing output file.", flush=True)
        print(f"Status: {result['status']}", flush=True)
        for k, v in result["metrics"].items():
            print(f"  {k}: {v}", flush=True)
        sys.exit(0)

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

    print(f"\nResult written to: {out_path}", flush=True)
    print(f"Status: {result['status']}", flush=True)
    for k, v in result["metrics"].items():
        print(f"  {k}: {v}", flush=True)
