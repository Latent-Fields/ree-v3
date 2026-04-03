#!/opt/local/bin/python3
"""
V3-EXQ-192a -- MECH-075 Hippocampal-VTA Novelty Loop Probe

Claims: MECH-075
Proposal: EXP-0046 / EVB-0046
Dispatch mode: targeted_probe

MECH-075 asserts:
  "Basal ganglia perform dopaminergic gain/threshold setting on hippocampal
  attractor dynamics."

Specifically, the BG via VTA/SNc dopaminergic tone modulates the
gain/threshold of hippocampal trajectory proposals, affecting which
attractors the CEM search settles into. This is the Lisman & Grace (2005)
hippocampal-VTA loop: hippocampal novelty detection (mismatch between
expected and observed z_world) signals via VTA dopaminergic output, which
in turn modulates hippocampal exploration/exploitation balance.

In the REE architecture, this maps to:
  - Hippocampal novelty = |z_world_observed - z_world_predicted| (E1 error
    projected onto z_world only, not z_self -- mismatch in world model)
  - VTA-like modulation = scaling the CEM search noise (ao_std) by a
    dopaminergic gain factor derived from novelty. High novelty -> wider
    CEM search (more exploration of action-object space). Low novelty ->
    narrower search (exploitation of known attractors).
  - This is architecturally distinct from MECH-111 (E1 error EMA as
    trajectory scoring bonus). MECH-075 modulates the PROPOSAL DISTRIBUTION,
    not the scoring function. It changes WHICH trajectories are generated,
    not how they are evaluated.

Conditions
----------
Condition A: NOVELTY_LOOP_ON
  - After each E1 tick, compute z_world novelty = MSE(z_world_pred, z_world_actual).
  - Maintain novelty EMA (separate from MECH-111 novelty_ema).
  - Before HippocampalModule.propose_trajectories(), scale CEM noise std by
    (1.0 + novelty_gain * novelty_ema). High novelty -> wider proposals.
  - Expected: agent explores more unique grid cells, discovers hazards/resources
    faster, because hippocampal proposals adapt to world-model surprise.

Condition B: NOVELTY_LOOP_OFF
  - No novelty modulation of CEM noise. Fixed ao_std as default.
  - Baseline exploration from CEM noise alone.

Design rationale
----------------
The key distinction from MECH-111 is the locus of effect. MECH-111 modulates
trajectory EVALUATION (score_trajectory novelty bonus). MECH-075 modulates
trajectory GENERATION (CEM proposal width). If MECH-075 is correct, the
novelty loop should produce more efficient exploration because the agent
generates more diverse proposals when the world model is surprised, and
converges to narrower proposals in familiar territory.

The manipulation is implemented by monkey-patching the CEM noise in
propose_trajectories(). This is a clean probe because the only difference
between conditions is whether CEM proposal width adapts to world-model
novelty.

Pre-registered thresholds
--------------------------
C1: unique_cells_gap = cells_ON - cells_OFF >= THRESH_CELL_GAP (all seeds)
    (novelty loop increases spatial coverage)
C2: hazard_discovery_gap = hazards_found_ON - hazards_found_OFF >= THRESH_HAZARD_GAP (all seeds)
    (novelty-driven exploration finds hazards faster)
C3: harm_delta = harm_rate_ON - harm_rate_OFF <= THRESH_HARM_DELTA (all seeds)
    (novelty loop does not recklessly increase harm)
C4: novelty_signal > THRESH_NOVELTY_SIGNAL (all seeds, ON condition)
    (manipulation check: novelty signal actually varies with world surprise)

Interpretation:
  C1+C2+C3+C4 PASS: MECH-075 SUPPORTED. Hippocampal-VTA novelty loop
    modulates proposal diversity; exploration efficiency improves without
    harm increase. Dopaminergic gain on CEM attractor dynamics confirmed.
  C1 or C2 fail, C3 pass: directional support but below threshold; consider
    increasing novelty_gain or warmup episodes.
  C3 fail: novelty loop causes harm increase; loop does not properly
    modulate search toward safe novelty.
  C4 fail: novelty signal is trivial; E1 world-prediction error not
    propagating or too uniform.

Seeds: [42, 123] (matched -- same env per seed across conditions)
Env:   CausalGridWorld size=10, 5 hazards, 5 resources
       nav_bias=0.25 (moderate exploration forcing)
Warmup: 200 episodes x 200 steps
Eval:   50 episodes x 200 steps
Estimated runtime: ~90 min any machine
"""

import sys
import random
import math
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
import torch.optim as optim

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorld
from ree_core.utils.config import REEConfig


EXPERIMENT_TYPE = "v3_exq_192a_mech075_novelty_loop_probe"
CLAIM_IDS = ["MECH-075"]

# Pre-registered thresholds
# C1: unique cell coverage gap (ON - OFF) must be >= this, all seeds
THRESH_CELL_GAP = 3
# C2: hazard discovery gap (ON - OFF) must be >= this, all seeds
THRESH_HAZARD_GAP = 1
# C3: harm rate delta (ON - OFF) must be <= this, all seeds (loop is safe)
THRESH_HARM_DELTA = 0.02
# C4: novelty signal must be above this in ON condition (manipulation check)
THRESH_NOVELTY_SIGNAL = 1e-4

# Env / training configuration
BODY_OBS_DIM = 10
WORLD_OBS_DIM = 200   # CausalGridWorld size=10, use_proxy_fields=False
ACTION_DIM = 4

# MECH-075 novelty loop parameters
NOVELTY_GAIN = 2.0         # CEM noise scaling: ao_std *= (1 + NOVELTY_GAIN * novelty_ema)
NOVELTY_EMA_ALPHA = 0.1    # EMA decay for novelty signal


def _make_env(seed: int) -> CausalGridWorld:
    return CausalGridWorld(
        size=10,
        num_resources=5,
        num_hazards=5,
        use_proxy_fields=False,
        seed=seed,
    )


def _action_entropy(action_counts: List[int]) -> float:
    """Compute Shannon entropy of action distribution."""
    total = sum(action_counts) + 1e-8
    probs = [c / total for c in action_counts]
    return -sum(p * math.log(p + 1e-9) for p in probs if p > 0)


def _compute_world_novelty(
    agent: REEAgent,
    world_dim: int,
) -> float:
    """Compute z_world prediction novelty from E1.

    Returns MSE between E1's predicted z_world and actual z_world.
    This is the hippocampal mismatch signal: how surprising is the
    current world state relative to the world model's expectation?
    """
    if len(agent._world_experience_buffer) < 2:
        return 0.0

    # E1 predicts the next combined [z_self, z_world] from the previous step.
    # We extract just the z_world component of the prediction error.
    z_self_prev = agent._self_experience_buffer[-2]
    z_world_prev = agent._world_experience_buffer[-2]
    z_world_actual = agent._world_experience_buffer[-1]

    combined_prev = torch.cat([z_self_prev.squeeze(0), z_world_prev.squeeze(0)])
    combined_prev = combined_prev.unsqueeze(0)  # [1, total_dim]

    with torch.no_grad():
        # Clone LSTM hidden state tuple (h, c) to prevent corruption
        if agent.e1._hidden_state is not None:
            saved_hidden = (
                agent.e1._hidden_state[0].clone(),
                agent.e1._hidden_state[1].clone(),
            )
        else:
            saved_hidden = None
        # Single-step prediction
        pred, _ = agent.e1(combined_prev)
        agent.e1._hidden_state = saved_hidden
        # Extract z_world from first horizon step (last world_dim dims of total_dim)
        # pred: [batch, horizon, total_dim] -- take step 0, z_world component
        pred_world = pred[:, 0, -world_dim:]
        actual_world = z_world_actual.squeeze(0).unsqueeze(0)
        novelty = F.mse_loss(pred_world, actual_world).item()

    return novelty


class NoveltyLoopState:
    """Tracks the hippocampal-VTA novelty loop state."""

    def __init__(self, gain: float, ema_alpha: float):
        self.gain = gain
        self.ema_alpha = ema_alpha
        self.novelty_ema: float = 0.0
        self.novelty_history: List[float] = []

    def update(self, novelty_raw: float) -> None:
        self.novelty_ema = (
            (1 - self.ema_alpha) * self.novelty_ema
            + self.ema_alpha * novelty_raw
        )
        self.novelty_history.append(novelty_raw)

    def get_cem_noise_scale(self) -> float:
        """CEM noise multiplier: 1.0 + gain * novelty_ema.

        High novelty -> wider CEM search (more exploration).
        Low novelty -> narrower CEM search (exploit known attractors).
        """
        return 1.0 + self.gain * self.novelty_ema


def _patched_propose_trajectories(
    hippocampal_module,
    z_world: torch.Tensor,
    z_self: Optional[torch.Tensor] = None,
    num_candidates: Optional[int] = None,
    e1_prior: Optional[torch.Tensor] = None,
    action_bias: Optional[torch.Tensor] = None,
    cem_noise_scale: float = 1.0,
):
    """Propose trajectories with novelty-modulated CEM noise.

    Reimplements HippocampalModule.propose_trajectories() with a noise
    scale parameter. When cem_noise_scale > 1.0, the CEM search explores
    more broadly in action-object space. When == 1.0, behaves identically
    to the default.
    """
    hm = hippocampal_module
    config = hm.config
    n = num_candidates or config.num_candidates
    num_elite = max(1, int(n * config.elite_fraction))
    batch_size = z_world.shape[0]
    device = z_world.device

    if z_self is None:
        z_self = torch.zeros(batch_size, config.world_dim, device=device)

    # Initialise in action-object space (SD-004)
    ao_mean = hm._get_terrain_action_object_mean(z_world, e1_prior=e1_prior)
    ao_std = torch.ones_like(ao_mean)

    all_trajectories = []

    for _iteration in range(config.num_cem_iterations):
        trajectories = []
        scores = []

        for _ in range(n):
            noise = torch.randn_like(ao_mean)
            # MECH-075: scale CEM noise by novelty-derived gain
            action_objects_sample = ao_mean + (ao_std * cem_noise_scale) * noise

            actions = hm._decode_action_objects(action_objects_sample)
            traj = hm.e2.rollout_with_world(
                z_self, z_world, actions,
                compute_action_objects=True,
                action_bias=action_bias,
            )
            trajectories.append(traj)
            scores.append(hm._score_trajectory(traj))

        scores_tensor = torch.stack(scores)
        elite_indices = torch.argsort(scores_tensor)[:num_elite]

        elite_ao = []
        for i in elite_indices:
            ao_seq = trajectories[i].get_action_object_sequence()
            if ao_seq is not None:
                elite_ao.append(ao_seq)

        if elite_ao:
            elite_ao_tensor = torch.stack(elite_ao)
            ao_mean = elite_ao_tensor.mean(dim=0)
            ao_std = elite_ao_tensor.std(dim=0) + 1e-6

        all_trajectories = trajectories

    return all_trajectories


def _run_single(
    seed: int,
    novelty_loop_on: bool,
    novelty_gain: float,
    warmup_episodes: int,
    eval_episodes: int,
    steps_per_episode: int,
    self_dim: int,
    world_dim: int,
    lr: float,
    alpha_world: float,
    alpha_self: float,
    nav_bias: float,
    dry_run: bool,
) -> Dict:
    """Run one (seed, condition) cell.

    Returns per-seed metrics for the paired comparison.
    NOVELTY_LOOP_ON: CEM noise modulated by hippocampal novelty signal.
    NOVELTY_LOOP_OFF: Fixed CEM noise (baseline).
    """
    torch.manual_seed(seed)
    random.seed(seed)

    cond_label = "NOVELTY_LOOP_ON" if novelty_loop_on else "NOVELTY_LOOP_OFF"

    env = _make_env(seed)

    config = REEConfig.from_dims(
        body_obs_dim=BODY_OBS_DIM,
        world_obs_dim=WORLD_OBS_DIM,
        action_dim=ACTION_DIM,
        self_dim=self_dim,
        world_dim=world_dim,
        alpha_world=alpha_world,
        alpha_self=alpha_self,
    )
    agent = REEAgent(config)

    e1_opt = optim.Adam(agent.e1.parameters(), lr=lr)
    e2_opt = optim.Adam(agent.e2.parameters(), lr=lr * 3)
    e3_opt = optim.Adam(
        list(agent.e3.parameters()) + list(agent.latent_stack.parameters()),
        lr=lr,
    )

    # Novelty loop state (only active in ON condition, but tracked in both
    # for diagnostic comparison)
    novelty_state = NoveltyLoopState(
        gain=novelty_gain if novelty_loop_on else 0.0,
        ema_alpha=NOVELTY_EMA_ALPHA,
    )

    if dry_run:
        warmup_episodes = 3
        eval_episodes = 2

    print(
        f"\n[V3-EXQ-192a] TRAIN {cond_label} seed={seed}"
        f" warmup={warmup_episodes} eval={eval_episodes}"
        f" nav_bias={nav_bias}"
        f" novelty_gain={novelty_gain if novelty_loop_on else 0.0}",
        flush=True,
    )
    agent.train()

    for ep in range(warmup_episodes):
        _, obs_dict = env.reset()
        agent.reset()
        ep_harm = 0.0

        for _ in range(steps_per_episode):
            obs_body = torch.tensor(obs_dict["body_state"], dtype=torch.float32)
            obs_world = torch.tensor(obs_dict["world_state"], dtype=torch.float32)

            z_self_t = None
            if agent._current_latent is not None:
                z_self_t = agent._current_latent.z_self.detach().clone()

            latent = agent.sense(obs_body, obs_world)
            ticks = agent.clock.advance()
            e1_prior = agent._e1_tick(latent) if ticks["e1_tick"] else torch.zeros(
                1, world_dim, device=agent.device
            )

            # Compute hippocampal novelty (z_world mismatch)
            novelty_raw = _compute_world_novelty(agent, world_dim)
            novelty_state.update(novelty_raw)

            # Get CEM noise scale from novelty loop
            cem_noise_scale = novelty_state.get_cem_noise_scale()

            # Generate trajectories with novelty-modulated CEM
            if ticks["e3_tick"] or agent._committed_candidates is None:
                z_world_for_e3 = agent.theta_buffer.summary()
                candidates = _patched_propose_trajectories(
                    agent.hippocampal,
                    z_world=z_world_for_e3,
                    z_self=latent.z_self,
                    e1_prior=e1_prior,
                    action_bias=agent._cue_action_bias,
                    cem_noise_scale=cem_noise_scale,
                )
                agent._committed_candidates = candidates
            else:
                candidates = agent._committed_candidates

            action = agent.select_action(candidates, ticks, temperature=1.0)

            if z_self_t is not None:
                agent.record_transition(z_self_t, action, latent.z_self.detach().clone())

            # E1 prediction loss
            e1_loss = agent.compute_prediction_loss()
            if e1_loss.requires_grad:
                e1_opt.zero_grad()
                e1_loss.backward()
                e1_opt.step()

            # E2 loss
            e2_loss = agent.compute_e2_loss()
            if e2_loss.requires_grad:
                e2_opt.zero_grad()
                e2_loss.backward()
                e2_opt.step()

            # Nav bias: random action override for harm contact
            if random.random() < nav_bias:
                action = torch.randint(0, ACTION_DIM, (1, ACTION_DIM), dtype=torch.float32)

            _, reward, done, info, obs_dict = env.step(action)
            harm_signal = float(reward) if reward < 0 else 0.0
            ep_harm += abs(harm_signal)

            # E3 harm supervision
            if agent._current_latent is not None:
                z_world = agent._current_latent.z_world.detach()
                harm_target = torch.tensor(
                    [[1.0 if harm_signal < 0 else 0.0]], device=agent.device
                )
                harm_loss = F.mse_loss(agent.e3.harm_eval(z_world), harm_target)
                e3_opt.zero_grad()
                harm_loss.backward()
                e3_opt.step()

            agent.update_residue(harm_signal)
            if done:
                break

        if (ep + 1) % 50 == 0:
            print(
                f"  [train] {cond_label} seed={seed} ep {ep+1}/{warmup_episodes}"
                f" harm={ep_harm:.3f}"
                f" novelty_ema={novelty_state.novelty_ema:.6f}"
                f" cem_scale={novelty_state.get_cem_noise_scale():.3f}",
                flush=True,
            )

    # --- EVAL ---
    agent.eval()
    action_counts = [0] * ACTION_DIM
    harm_events = 0
    total_steps = 0
    visited_cells: Set[tuple] = set()
    hazards_found: Set[tuple] = set()
    resources_found: Set[tuple] = set()
    novelty_signals: List[float] = []
    cem_scales: List[float] = []

    # Track hazard/resource positions from env for discovery counting
    hazard_positions = set()
    resource_positions = set()
    for (hx, hy) in env.hazards:
        hazard_positions.add((int(hx), int(hy)))
    for (rx, ry) in env.resources:
        resource_positions.add((int(rx), int(ry)))

    for ep in range(eval_episodes):
        _, obs_dict = env.reset()
        agent.reset()
        # Re-read hazard/resource positions after reset (they may differ per reset)
        hazard_positions_ep = set()
        resource_positions_ep = set()
        for (hx, hy) in env.hazards:
            hazard_positions_ep.add((int(hx), int(hy)))
        for (rx, ry) in env.resources:
            resource_positions_ep.add((int(rx), int(ry)))

        for _ in range(steps_per_episode):
            obs_body = torch.tensor(obs_dict["body_state"], dtype=torch.float32)
            obs_world = torch.tensor(obs_dict["world_state"], dtype=torch.float32)

            with torch.no_grad():
                latent = agent.sense(obs_body, obs_world)
                ticks = agent.clock.advance()
                e1_prior = agent._e1_tick(latent) if ticks["e1_tick"] else torch.zeros(
                    1, world_dim, device=agent.device
                )

                # Compute novelty for diagnostic and CEM modulation
                novelty_raw = _compute_world_novelty(agent, world_dim)
                novelty_state.update(novelty_raw)
                novelty_signals.append(novelty_raw)
                cem_scale = novelty_state.get_cem_noise_scale()
                cem_scales.append(cem_scale)

                # Generate trajectories with novelty-modulated CEM
                if ticks["e3_tick"] or agent._committed_candidates is None:
                    z_world_for_e3 = agent.theta_buffer.summary()
                    candidates = _patched_propose_trajectories(
                        agent.hippocampal,
                        z_world=z_world_for_e3,
                        z_self=latent.z_self,
                        e1_prior=e1_prior,
                        action_bias=agent._cue_action_bias,
                        cem_noise_scale=cem_scale,
                    )
                    agent._committed_candidates = candidates
                else:
                    candidates = agent._committed_candidates

                action = agent.select_action(candidates, ticks, temperature=0.5)

            action_idx = int(action.squeeze().argmax().item())
            action_counts[action_idx] += 1

            _, reward, done, info, obs_dict = env.step(action)
            ttype = info.get("transition_type", "none")
            if ttype in ("agent_caused_hazard", "hazard_approach"):
                harm_events += 1

            # Track visited grid cells
            pos_x = int(obs_dict["body_state"][0] * 10)
            pos_y = int(obs_dict["body_state"][1] * 10)
            pos = (pos_x, pos_y)
            visited_cells.add(pos)

            # Track hazard/resource discovery (visited adjacent cells)
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    neighbor = (pos_x + dx, pos_y + dy)
                    if neighbor in hazard_positions_ep:
                        hazards_found.add(neighbor)
                    if neighbor in resource_positions_ep:
                        resources_found.add(neighbor)

            total_steps += 1
            if done:
                break

    policy_entropy = _action_entropy(action_counts)
    harm_rate = harm_events / max(1, total_steps)
    unique_cells = len(visited_cells)
    n_hazards_found = len(hazards_found)
    n_resources_found = len(resources_found)
    mean_novelty_signal = sum(novelty_signals) / max(1, len(novelty_signals))
    std_novelty_signal = (
        (sum((n - mean_novelty_signal) ** 2 for n in novelty_signals)
         / max(1, len(novelty_signals))) ** 0.5
    )
    mean_cem_scale = sum(cem_scales) / max(1, len(cem_scales))

    print(
        f"  [eval] {cond_label} seed={seed}"
        f" cells={unique_cells}"
        f" hazards_found={n_hazards_found}/{len(hazard_positions)}"
        f" resources_found={n_resources_found}/{len(resource_positions)}"
        f" harm_rate={harm_rate:.4f}"
        f" harm_events={harm_events}"
        f" entropy={policy_entropy:.4f}"
        f" mean_novelty={mean_novelty_signal:.6f}"
        f" mean_cem_scale={mean_cem_scale:.3f}",
        flush=True,
    )

    return {
        "seed": seed,
        "condition": cond_label,
        "novelty_loop_on": novelty_loop_on,
        "unique_cells": unique_cells,
        "n_hazards_found": n_hazards_found,
        "n_resources_found": n_resources_found,
        "policy_entropy": policy_entropy,
        "harm_rate": harm_rate,
        "harm_events": harm_events,
        "mean_novelty_signal": mean_novelty_signal,
        "std_novelty_signal": std_novelty_signal,
        "mean_cem_scale": mean_cem_scale,
        "total_steps": total_steps,
    }


def run(
    seeds: Tuple[int, ...] = (42, 123),
    novelty_gain: float = NOVELTY_GAIN,
    warmup_episodes: int = 200,
    eval_episodes: int = 50,
    steps_per_episode: int = 200,
    self_dim: int = 32,
    world_dim: int = 32,
    lr: float = 1e-3,
    alpha_world: float = 0.9,
    alpha_self: float = 0.3,
    nav_bias: float = 0.25,
    dry_run: bool = False,
) -> dict:
    """MECH-075 hippocampal-VTA novelty loop probe.

    Paired discriminative design: each seed runs both conditions (same env,
    same init). NOVELTY_LOOP_ON uses z_world mismatch to scale CEM proposal
    noise; NOVELTY_LOOP_OFF uses fixed CEM noise. If MECH-075 is correct,
    ON condition shows more efficient exploration (more unique cells, more
    hazards/resources discovered) without substantially worse harm avoidance.
    """
    print(
        f"\n[V3-EXQ-192a] MECH-075 Hippocampal-VTA Novelty Loop Probe"
        f" seeds={list(seeds)} novelty_gain={novelty_gain}",
        flush=True,
    )

    results_on: List[Dict] = []
    results_off: List[Dict] = []

    for seed in seeds:
        for loop_on in [True, False]:
            r = _run_single(
                seed=seed,
                novelty_loop_on=loop_on,
                novelty_gain=novelty_gain,
                warmup_episodes=warmup_episodes,
                eval_episodes=eval_episodes,
                steps_per_episode=steps_per_episode,
                self_dim=self_dim,
                world_dim=world_dim,
                lr=lr,
                alpha_world=alpha_world,
                alpha_self=alpha_self,
                nav_bias=nav_bias,
                dry_run=dry_run,
            )
            if loop_on:
                results_on.append(r)
            else:
                results_off.append(r)

    def _avg(results: List[Dict], key: str) -> float:
        vals = [r[key] for r in results]
        return float(sum(vals) / max(1, len(vals)))

    # Per-seed gaps (paired comparison)
    per_seed_cell_gap: List[int] = []
    per_seed_hazard_gap: List[int] = []
    per_seed_harm_delta: List[float] = []

    for r_on in results_on:
        matching = [r for r in results_off if r["seed"] == r_on["seed"]]
        if matching:
            r_off = matching[0]
            per_seed_cell_gap.append(r_on["unique_cells"] - r_off["unique_cells"])
            per_seed_hazard_gap.append(
                r_on["n_hazards_found"] - r_off["n_hazards_found"]
            )
            per_seed_harm_delta.append(r_on["harm_rate"] - r_off["harm_rate"])

    # Aggregate means
    mean_cells_on = _avg(results_on, "unique_cells")
    mean_cells_off = _avg(results_off, "unique_cells")
    mean_hazards_on = _avg(results_on, "n_hazards_found")
    mean_hazards_off = _avg(results_off, "n_hazards_found")
    mean_harm_on = _avg(results_on, "harm_rate")
    mean_harm_off = _avg(results_off, "harm_rate")
    mean_novelty_on = _avg(results_on, "mean_novelty_signal")
    mean_novelty_off = _avg(results_off, "mean_novelty_signal")
    mean_cem_scale_on = _avg(results_on, "mean_cem_scale")

    # Pre-registered PASS criteria
    # C1: unique cell gap >= THRESH_CELL_GAP all seeds
    c1_pass = (
        len(per_seed_cell_gap) > 0
        and all(g >= THRESH_CELL_GAP for g in per_seed_cell_gap)
    )
    # C2: hazard discovery gap >= THRESH_HAZARD_GAP all seeds
    c2_pass = (
        len(per_seed_hazard_gap) > 0
        and all(g >= THRESH_HAZARD_GAP for g in per_seed_hazard_gap)
    )
    # C3: harm delta <= THRESH_HARM_DELTA all seeds
    c3_pass = (
        len(per_seed_harm_delta) > 0
        and all(d <= THRESH_HARM_DELTA for d in per_seed_harm_delta)
    )
    # C4: novelty signal above threshold in ON condition all seeds
    c4_pass = all(
        r["mean_novelty_signal"] > THRESH_NOVELTY_SIGNAL for r in results_on
    )

    all_pass = c1_pass and c2_pass and c3_pass and c4_pass
    criteria_met = sum([c1_pass, c2_pass, c3_pass, c4_pass])
    status = "PASS" if all_pass else "FAIL"

    if all_pass:
        decision = "retain_ree"
    elif (c1_pass or c2_pass) and c3_pass:
        decision = "hybridize"
    elif not c3_pass:
        decision = "retire_ree_claim"
    else:
        decision = "inconclusive"

    print(
        f"\n[V3-EXQ-192a] Results:"
        f" cells ON={mean_cells_on:.1f} OFF={mean_cells_off:.1f}"
        f" hazards ON={mean_hazards_on:.1f} OFF={mean_hazards_off:.1f}"
        f" harm ON={mean_harm_on:.4f} OFF={mean_harm_off:.4f}"
        f" novelty ON={mean_novelty_on:.6f} OFF={mean_novelty_off:.6f}",
        flush=True,
    )
    print(
        f"  per_seed_cell_gap={per_seed_cell_gap}"
        f" per_seed_hazard_gap={per_seed_hazard_gap}"
        f" per_seed_harm_delta={[round(d, 4) for d in per_seed_harm_delta]}"
        f" mean_cem_scale_on={mean_cem_scale_on:.3f}"
        f" decision={decision}  status={status} ({criteria_met}/4)",
        flush=True,
    )

    failure_notes: List[str] = []
    if not c1_pass:
        failure_notes.append(
            f"C1 FAIL: per-seed cell_gap {per_seed_cell_gap}"
            f" < {THRESH_CELL_GAP}"
            " -- novelty loop does not increase spatial coverage"
        )
    if not c2_pass:
        failure_notes.append(
            f"C2 FAIL: per-seed hazard_gap {per_seed_hazard_gap}"
            f" < {THRESH_HAZARD_GAP}"
            " -- novelty loop does not improve hazard discovery"
        )
    if not c3_pass:
        failure_notes.append(
            f"C3 FAIL: per-seed harm_delta {[round(d, 4) for d in per_seed_harm_delta]}"
            f" > {THRESH_HARM_DELTA}"
            " -- novelty loop increases harm exposure; CEM broadening is reckless"
        )
    if not c4_pass:
        failing_seeds = [
            r["seed"] for r in results_on
            if r["mean_novelty_signal"] <= THRESH_NOVELTY_SIGNAL
        ]
        failure_notes.append(
            f"C4 FAIL: mean_novelty_signal <= {THRESH_NOVELTY_SIGNAL}"
            f" in seeds {failing_seeds}"
            " -- E1 world-prediction mismatch not producing novelty signal"
        )

    for note in failure_notes:
        print(f"  {note}", flush=True)

    if all_pass:
        interpretation = (
            f"MECH-075 SUPPORTED: Hippocampal-VTA novelty loop modulates CEM"
            f" proposal diversity via dopaminergic gain on attractor dynamics."
            f" NOVELTY_LOOP_ON: cells={mean_cells_on:.1f}"
            f" hazards={mean_hazards_on:.1f} harm={mean_harm_on:.4f}."
            f" NOVELTY_LOOP_OFF: cells={mean_cells_off:.1f}"
            f" hazards={mean_hazards_off:.1f} harm={mean_harm_off:.4f}."
            f" CEM noise scale ON={mean_cem_scale_on:.3f}."
            f" per-seed cell_gap={per_seed_cell_gap}"
            f" hazard_gap={per_seed_hazard_gap}"
            f" harm_delta={[round(d, 4) for d in per_seed_harm_delta]}."
            f" Novelty-driven CEM broadening increases exploration efficiency"
            f" without harm increase -- dopaminergic modulation of hippocampal"
            f" search width is confirmed."
        )
    elif (c1_pass or c2_pass) and c3_pass:
        interpretation = (
            f"Partial support: directional exploration increase observed (cells"
            f" or hazard discovery) without harm elevation, but below"
            f" pre-registered threshold on one criterion."
            f" Consider increasing novelty_gain or warmup. C1={c1_pass}"
            f" C2={c2_pass} C3={c3_pass} C4={c4_pass}."
        )
    elif not c3_pass:
        interpretation = (
            f"MECH-075 NOT SUPPORTED: novelty loop increases harm exposure."
            f" per-seed harm_delta={[round(d, 4) for d in per_seed_harm_delta]}."
            f" CEM broadening from novelty signal is not appropriately bounded;"
            f" dopaminergic gain pushes proposals into hazardous attractors."
        )
    else:
        interpretation = (
            f"MECH-075 NOT SUPPORTED: novelty loop does not produce measurable"
            f" exploration improvement. E1 world-prediction error may be too"
            f" uniform across z_world states, or CEM noise scaling has no effect"
            f" on trajectory diversity at this gain level."
            f" Criteria: C1={c1_pass} C2={c2_pass} C3={c3_pass} C4={c4_pass}."
        )

    per_on_rows = "\n".join(
        f"  seed={r['seed']}: cells={r['unique_cells']}"
        f" hazards={r['n_hazards_found']}"
        f" resources={r['n_resources_found']}"
        f" harm_rate={r['harm_rate']:.4f}"
        f" novelty={r['mean_novelty_signal']:.6f}"
        f" cem_scale={r['mean_cem_scale']:.3f}"
        for r in results_on
    )
    per_off_rows = "\n".join(
        f"  seed={r['seed']}: cells={r['unique_cells']}"
        f" hazards={r['n_hazards_found']}"
        f" resources={r['n_resources_found']}"
        f" harm_rate={r['harm_rate']:.4f}"
        f" novelty={r['mean_novelty_signal']:.6f}"
        for r in results_off
    )

    failure_section = ""
    if failure_notes:
        failure_section = "\n## Failure Notes\n\n" + "\n".join(
            f"- {n}" for n in failure_notes
        )

    summary_markdown = (
        f"# V3-EXQ-192a -- MECH-075 Hippocampal-VTA Novelty Loop Probe\n\n"
        f"**Status:** {status}\n"
        f"**Claims:** MECH-075\n"
        f"**Decision:** {decision}\n"
        f"**Seeds:** {list(seeds)}\n"
        f"**Conditions:** NOVELTY_LOOP_ON vs NOVELTY_LOOP_OFF\n"
        f"**Warmup:** {warmup_episodes} eps x {steps_per_episode} steps"
        f"  **Eval:** {eval_episodes} eps x {steps_per_episode} steps\n"
        f"**Env:** CausalGridWorld size=10, 5 hazards, 5 resources"
        f" nav_bias={nav_bias}\n"
        f"**novelty_gain:** {novelty_gain}\n\n"
        f"## Design\n\n"
        f"MECH-075 asserts that BG perform dopaminergic gain/threshold setting on"
        f" hippocampal attractor dynamics (Lisman & Grace 2005 novelty loop)."
        f" The experiment tests whether z_world mismatch detection (hippocampal"
        f" novelty) driving CEM proposal noise scaling (VTA-like dopaminergic"
        f" modulation) improves exploration efficiency. NOVELTY_LOOP_ON scales"
        f" CEM noise by (1 + gain * novelty_ema); NOVELTY_LOOP_OFF uses fixed"
        f" noise. The manipulation targets the PROPOSAL distribution, not"
        f" trajectory scoring (distinct from MECH-111).\n\n"
        f"## Pre-Registered Thresholds\n\n"
        f"C1: per-seed unique_cells_gap (ON-OFF) >= {THRESH_CELL_GAP} (all seeds)\n"
        f"C2: per-seed hazard_discovery_gap (ON-OFF) >= {THRESH_HAZARD_GAP} (all seeds)\n"
        f"C3: per-seed harm_delta (ON-OFF) <= {THRESH_HARM_DELTA} (all seeds)\n"
        f"C4: mean_novelty_signal_ON > {THRESH_NOVELTY_SIGNAL} (all seeds)\n\n"
        f"## Results\n\n"
        f"| Condition | cells | hazards_found | harm_rate | novelty | cem_scale |\n"
        f"|-----------|-------|---------------|-----------|---------|----------|\n"
        f"| NOVELTY_LOOP_ON  | {mean_cells_on:.1f}"
        f" | {mean_hazards_on:.1f} | {mean_harm_on:.4f}"
        f" | {mean_novelty_on:.6f} | {mean_cem_scale_on:.3f} |\n"
        f"| NOVELTY_LOOP_OFF | {mean_cells_off:.1f}"
        f" | {mean_hazards_off:.1f} | {mean_harm_off:.4f}"
        f" | {mean_novelty_off:.6f} | 1.000 |\n\n"
        f"**per-seed cell_gap: {per_seed_cell_gap}**\n"
        f"**per-seed hazard_gap: {per_seed_hazard_gap}**\n"
        f"**per-seed harm_delta: {[round(d, 4) for d in per_seed_harm_delta]}**\n\n"
        f"## PASS Criteria\n\n"
        f"| Criterion | Result | Value |\n"
        f"|---|---|---|\n"
        f"| C1: cell_gap >= {THRESH_CELL_GAP} (all seeds)"
        f" | {'PASS' if c1_pass else 'FAIL'}"
        f" | {per_seed_cell_gap} |\n"
        f"| C2: hazard_gap >= {THRESH_HAZARD_GAP} (all seeds)"
        f" | {'PASS' if c2_pass else 'FAIL'}"
        f" | {per_seed_hazard_gap} |\n"
        f"| C3: harm_delta <= {THRESH_HARM_DELTA} (all seeds)"
        f" | {'PASS' if c3_pass else 'FAIL'}"
        f" | {[round(d, 4) for d in per_seed_harm_delta]} |\n"
        f"| C4: novelty_signal > {THRESH_NOVELTY_SIGNAL} (all seeds)"
        f" | {'PASS' if c4_pass else 'FAIL'}"
        f" | {[round(r['mean_novelty_signal'], 7) for r in results_on]} |\n\n"
        f"Criteria met: {criteria_met}/4 -> **{status}**\n\n"
        f"## Interpretation\n\n"
        f"{interpretation}\n\n"
        f"## Per-Seed Detail\n\n"
        f"NOVELTY_LOOP_ON:\n{per_on_rows}\n\n"
        f"NOVELTY_LOOP_OFF:\n{per_off_rows}\n"
        f"{failure_section}\n"
    )

    metrics = {
        "mean_cells_on":             float(mean_cells_on),
        "mean_cells_off":            float(mean_cells_off),
        "mean_cell_gap":             float(mean_cells_on - mean_cells_off),
        "mean_hazards_on":           float(mean_hazards_on),
        "mean_hazards_off":          float(mean_hazards_off),
        "mean_hazard_gap":           float(mean_hazards_on - mean_hazards_off),
        "mean_harm_rate_on":         float(mean_harm_on),
        "mean_harm_rate_off":        float(mean_harm_off),
        "mean_harm_delta":           float(mean_harm_on - mean_harm_off),
        "mean_novelty_signal_on":    float(mean_novelty_on),
        "mean_novelty_signal_off":   float(mean_novelty_off),
        "mean_cem_scale_on":         float(mean_cem_scale_on),
        "per_seed_cell_gap_min":     float(min(per_seed_cell_gap)) if per_seed_cell_gap else 0.0,
        "per_seed_hazard_gap_min":   float(min(per_seed_hazard_gap)) if per_seed_hazard_gap else 0.0,
        "per_seed_harm_delta_max":   float(max(per_seed_harm_delta)) if per_seed_harm_delta else 0.0,
        "novelty_signal_min":        float(min(r["mean_novelty_signal"] for r in results_on)),
        "n_seeds":                   float(len(seeds)),
        "novelty_gain":              float(novelty_gain),
        "nav_bias":                  float(nav_bias),
        "crit1_pass":                1.0 if c1_pass else 0.0,
        "crit2_pass":                1.0 if c2_pass else 0.0,
        "crit3_pass":                1.0 if c3_pass else 0.0,
        "crit4_pass":                1.0 if c4_pass else 0.0,
        "criteria_met":              float(criteria_met),
    }

    return {
        "status": status,
        "metrics": metrics,
        "summary_markdown": summary_markdown,
        "claim_ids": CLAIM_IDS,
        "evidence_direction": (
            "supports" if all_pass
            else ("mixed" if criteria_met >= 2 else "weakens")
        ),
        "experiment_type": EXPERIMENT_TYPE,
        "fatal_error_count": 0,
    }


if __name__ == "__main__":
    import argparse
    import json
    from datetime import datetime, timezone

    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds",          type=int,   nargs="+", default=[42, 123])
    parser.add_argument("--novelty-gain",   type=float, default=NOVELTY_GAIN)
    parser.add_argument("--warmup",         type=int,   default=200)
    parser.add_argument("--eval-eps",       type=int,   default=50)
    parser.add_argument("--steps",          type=int,   default=200)
    parser.add_argument("--alpha-world",    type=float, default=0.9)
    parser.add_argument("--alpha-self",     type=float, default=0.3)
    parser.add_argument("--nav-bias",       type=float, default=0.25)
    parser.add_argument("--dry-run",        action="store_true",
                        help="3 warmup + 2 eval eps per cell to check for errors.")
    args = parser.parse_args()

    result = run(
        seeds=tuple(args.seeds),
        novelty_gain=args.novelty_gain,
        warmup_episodes=args.warmup,
        eval_episodes=args.eval_eps,
        steps_per_episode=args.steps,
        alpha_world=args.alpha_world,
        alpha_self=args.alpha_self,
        nav_bias=args.nav_bias,
        dry_run=args.dry_run,
    )

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result["run_timestamp"] = ts
    result["claim"] = CLAIM_IDS[0]
    result["verdict"] = result["status"]
    result["run_id"] = f"{EXPERIMENT_TYPE}_{ts}_v3"
    result["architecture_epoch"] = "ree_hybrid_guardrails_v1"
    result["registered_thresholds"] = {
        "C1_per_seed_cell_gap":       THRESH_CELL_GAP,
        "C2_per_seed_hazard_gap":     THRESH_HAZARD_GAP,
        "C3_per_seed_harm_delta":     THRESH_HARM_DELTA,
        "C4_novelty_signal_min":      THRESH_NOVELTY_SIGNAL,
    }
    result["seeds"] = list(args.seeds)
    result["conditions"] = ["NOVELTY_LOOP_ON", "NOVELTY_LOOP_OFF"]
    result["dispatch_mode"] = "targeted_probe"
    result["backlog_id"] = "EVB-0046"
    result["evidence_class"] = "targeted_probe"
    result["claim_ids_tested"] = CLAIM_IDS

    if args.dry_run:
        print("\n[dry-run] Skipping file output.", flush=True)
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
