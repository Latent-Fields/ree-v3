#!/opt/local/bin/python3
"""
V3-EXQ-143 -- MECH-118 Hopfield Familiarity Signal Discriminative Pair

Claims: MECH-118
Proposal: EXP-0076 / EVB-0063

MECH-118 asserts:
  "Hopfield-style pattern familiarity of z_self states (stability against stored
  self-state memories) is a distinct self-maintenance signal from D_eff coherence."

Specifically, a Hopfield familiarity loss -- pulling z_self toward its nearest
stored self-state pattern -- provides a recovery signal that D_eff homeostasis
alone cannot provide when the agent is in a coherent-but-unfamiliar self-state
(Regime 3 / MECH-119).

If MECH-118 is correct:
  HOPFIELD_ON:     combined loss (D_eff homeostasis + Hopfield familiarity pull)
                   restores stability after a novelty-structured (R3-style)
                   perturbation more rapidly than D_eff alone.
  HOPFIELD_ABLATED: D_eff homeostasis only (same D_eff weight) -- the agent
                   corrects D_eff dispersion but cannot recover from a
                   coherent-unfamiliar configuration because D_eff is blind
                   to pattern familiarity.

Perturbation design
-------------------
After warmup + memory collection (Hopfield memory filled with trained z_self
patterns), a Regime 3 vector is injected: z_self is replaced with an orthogonal
direction at the same norm as baseline z_self (low D_eff but near-zero stability).
This is the discriminative test case where D_eff homeostasis is insufficient:
the agent has a "coherent" self-model (D_eff similar to baseline) but it is in
completely unfamiliar territory (stability ~0).

The key prediction:
  HOPFIELD_ON:     stability recovers toward baseline during post-perturb eval
                   (Hopfield familiarity loss pulls z_self back to known region).
  HOPFIELD_ABLATED: stability stays low -- D_eff loss cannot distinguish
                   "coherent but unfamiliar" from normal (both have acceptable D_eff).

Secondary check: D_eff should be similarly controlled in both conditions after
perturbation (confirming the Hopfield signal is orthogonal to D_eff homeostasis,
not a substitute for it). This cross-validates that the two signals are
independent (Q-022 dissociation premise).

Conditions
----------
HOPFIELD_ON:
  - D_eff homeostatic loss (same as MECH-113): maint_weight=0.1, d_eff_target=1.5
  - Hopfield familiarity pull loss: hopfield_weight=0.05
    Loss = hopfield_weight * (1 - max_cosine_similarity(z_self, stored_patterns))
    Gradient: pulls z_self toward its nearest stored pattern after perturbation.
  - Memory: 64 slots, FIFO, filled from baseline eval; then frozen during eval.

HOPFIELD_ABLATED:
  - D_eff homeostatic loss only: maint_weight=0.1, d_eff_target=1.5
  - No Hopfield familiarity component (hopfield_weight=0.0).
  - Same architecture, same random seed, same env.

Pre-registered thresholds
--------------------------
C1: stab_recovery_ON (mean stability post-perturb) >= THRESH_STAB_RECOVERY_ON
    both seeds (ON condition recovers familiarity)
C2: stab_recovery_ABL (mean stability post-perturb) <= THRESH_STAB_RECOVERY_ABL
    both seeds (ABLATED condition does not recover familiarity)
C3: per-seed stab_gap (ON_post - ABL_post) >= THRESH_STAB_GAP both seeds
    (clear discriminative separation)
C4: d_eff_gap (abs(d_eff_post_ON - d_eff_post_ABL)) <= THRESH_D_EFF_CROSS
    both seeds (D_eff equally controlled -- Hopfield adds NEW information)
C5: n_stab_samples >= THRESH_N_STAB per condition per seed (data quality)

Interpretation:
  C1+C2+C3+C4+C5 ALL PASS: MECH-118 SUPPORTED. Hopfield familiarity is a
    distinct signal: ON recovers stability; ABLATED does not; D_eff is equally
    controlled in both conditions (confirming orthogonality). The Hopfield
    loss provides information D_eff alone cannot.
  C3 FAIL only: conditions do not separate clearly -- Hopfield weight may
    need increase or perturbation may be insufficient.
  C4 FAIL only: Hopfield also affects D_eff -- signals are not independent
    (may need weight rebalancing or architectural separation).
  C1+C2 FAIL: novelty perturbation not effective; increase noise or R3 angle.

Seeds: [42, 123] (matched -- same env per seed across conditions)
Env:   CausalGridWorldV2 size=10, 3 hazards, 4 resources, nav_bias=0.35
Warmup: 300 episodes x 200 steps
Memory collection: 30 baseline eval episodes (fills Hopfield memory)
Post-perturbation eval: 50 episodes x 200 steps
Estimated runtime: ~90 min any machine
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


EXPERIMENT_TYPE = "v3_exq_143_mech118_hopfield_familiarity_pair"
CLAIM_IDS = ["MECH-118"]

# ---------------------------------------------------------------------------
# Pre-registered thresholds
# ---------------------------------------------------------------------------
# C1: HOPFIELD_ON mean post-perturb stability must exceed this (both seeds)
THRESH_STAB_RECOVERY_ON = 0.30
# C2: HOPFIELD_ABLATED mean post-perturb stability must stay below this (both seeds)
THRESH_STAB_RECOVERY_ABL = 0.20
# C3: per-seed stab gap (ON_post - ABL_post) must exceed this (both seeds)
THRESH_STAB_GAP = 0.08
# C4: D_eff difference (|ON - ABL|) must not exceed this (both seeds)
#     -- confirms Hopfield adds familiarity signal without replacing D_eff control
THRESH_D_EFF_CROSS = 0.80
# C5: minimum stability samples per condition per seed (data quality)
THRESH_N_STAB = 50

# ---------------------------------------------------------------------------
# Model constants (fixed for both conditions)
# ---------------------------------------------------------------------------
BODY_OBS_DIM = 12
WORLD_OBS_DIM = 250  # CausalGridWorldV2 size=10
ACTION_DIM = 5

# Homeostatic D_eff parameters (same both conditions)
MAINT_WEIGHT = 0.1
D_EFF_TARGET = 1.5

# Hopfield parameters
HOPFIELD_WEIGHT = 0.05   # ON condition only
HOPFIELD_CAPACITY = 64   # LRU memory slots
NOISE_SIGMA = 0.5        # Baseline noise level during warmup

# ---------------------------------------------------------------------------
# Inline HopfieldMemory (MECH-118 design; Ramsauer 2021)
# Reused from EXQ-084d design. Inline to avoid ree_core dependency.
# ---------------------------------------------------------------------------

class HopfieldMemory:
    """
    Modern Hopfield pattern familiarity probe.

    Stores up to `capacity` z_self snapshots (FIFO).
    stability(z) = max cosine similarity to stored patterns.
    High stability -> current z_self resembles a familiar stored state.
    Low stability  -> current z_self is in unfamiliar territory.
    """

    def __init__(self, capacity: int = 64) -> None:
        self.patterns: List[torch.Tensor] = []
        self.capacity = capacity

    def store(self, z_1d: torch.Tensor) -> None:
        """Add z_1d ([self_dim]) to memory. Silently drops near-zero vectors."""
        if z_1d.norm().item() < 1e-6:
            return
        if len(self.patterns) >= self.capacity:
            self.patterns.pop(0)
        self.patterns.append(z_1d.detach().cpu().float().clone())

    def stability(self, z_1d: torch.Tensor) -> float:
        """Max cosine similarity to stored patterns. Returns 0.0 if empty."""
        if not self.patterns:
            return 0.0
        q = F.normalize(z_1d.detach().cpu().float().unsqueeze(0), dim=-1)  # [1, d]
        M = F.normalize(torch.stack(self.patterns), dim=-1)                 # [K, d]
        return float((q @ M.T).squeeze(0).max().item())

    def nearest_pattern_tensor(self, z_1d: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Return stored pattern most similar to z_1d.
        Used for Hopfield familiarity loss (gradient toward nearest pattern).
        """
        if not self.patterns:
            return None
        q = F.normalize(z_1d.detach().cpu().float().unsqueeze(0), dim=-1)
        M = F.normalize(torch.stack(self.patterns), dim=-1)
        idx = int((q @ M.T).squeeze(0).argmax().item())
        return self.patterns[idx].clone()

    def __len__(self) -> int:
        return len(self.patterns)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_d_eff(z_1d: torch.Tensor) -> float:
    """
    Participation ratio D_eff = (sum|z|)^2 / sum(z^2).
    High D_eff = diffuse; low D_eff = coherent.
    """
    z = z_1d.detach().float().squeeze()
    abs_sum = z.abs().sum()
    sq_sum = z.pow(2).sum()
    if sq_sum.item() < 1e-8:
        return float("nan")
    return float((abs_sum.pow(2) / sq_sum).item())


def _make_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(
        seed=seed,
        size=10,
        num_hazards=3,
        num_resources=4,
        hazard_harm=0.02,
        env_drift_interval=5,
        env_drift_prob=0.2,
    )


def _build_r3_vector(
    memory: HopfieldMemory,
    target_norm: float,
    seed: int,
) -> torch.Tensor:
    """
    Construct a coherent-but-unfamiliar perturbation vector (Regime 3).

    Unit vector orthogonal to mean(stored patterns), scaled to target_norm.
    Low D_eff (concentrated) but near-zero Hopfield stability (unfamiliar direction).
    Gram-Schmidt: removes mean pattern component from a random vector.
    """
    if not memory.patterns:
        raise RuntimeError("Cannot build R3 vector: memory is empty")

    torch.manual_seed(seed + 9999)
    M = torch.stack(memory.patterns)                                # [K, d]
    mean_z = F.normalize(M.mean(0).unsqueeze(0), dim=-1).squeeze(0)  # [d]

    rand = torch.randn(mean_z.shape)
    rand = rand - (rand @ mean_z) * mean_z
    norm = rand.norm()
    if norm.item() < 1e-8:
        dim = mean_z.shape[0]
        rand = torch.zeros(dim)
        rand[dim - 1] = 1.0
        rand = rand - (rand @ mean_z) * mean_z
        norm = rand.norm()

    r3_dir = rand / norm
    return r3_dir * target_norm


# ---------------------------------------------------------------------------
# Single-seed cell runner
# ---------------------------------------------------------------------------

def _run_single(
    seed: int,
    hopfield_enabled: bool,
    hopfield_weight: float,
    maint_weight: float,
    d_eff_target: float,
    warmup_episodes: int,
    memory_episodes: int,
    eval_episodes: int,
    steps_per_episode: int,
    hopfield_capacity: int,
    self_dim: int,
    world_dim: int,
    lr: float,
    alpha_world: float,
    alpha_self: float,
    nav_bias: float,
    dry_run: bool,
) -> Dict:
    """
    Run one (seed, condition) cell.

    HOPFIELD_ON:     D_eff homeostasis + Hopfield familiarity pull during eval.
    HOPFIELD_ABLATED: D_eff homeostasis only.

    Returns per-seed Hopfield stability and D_eff metrics for paired comparison.
    """
    cond_label = (
        f"HOPFIELD_ON(w={hopfield_weight},cap={hopfield_capacity})"
        if hopfield_enabled
        else "HOPFIELD_ABLATED"
    )

    torch.manual_seed(seed)
    random.seed(seed)

    env = _make_env(seed)

    config = REEConfig.from_dims(
        body_obs_dim=BODY_OBS_DIM,
        world_obs_dim=WORLD_OBS_DIM,
        action_dim=ACTION_DIM,
        self_dim=self_dim,
        world_dim=world_dim,
        alpha_world=alpha_world,
        alpha_self=alpha_self,
        self_maintenance_weight=maint_weight,
        self_maintenance_d_eff_target=d_eff_target,
    )
    agent = REEAgent(config)

    e1_opt = optim.Adam(agent.e1.parameters(), lr=lr)
    e2_opt = optim.Adam(agent.e2.parameters(), lr=lr * 3)
    e3_opt = optim.Adam(
        list(agent.e3.parameters()) + list(agent.latent_stack.parameters()),
        lr=lr,
    )

    if dry_run:
        warmup_episodes = 3
        memory_episodes = 2
        eval_episodes = 3

    print(
        f"\n[V3-EXQ-143] TRAIN {cond_label} seed={seed}"
        f" warmup={warmup_episodes} mem_eps={memory_episodes}"
        f" eval={eval_episodes} nav_bias={nav_bias}",
        flush=True,
    )

    # -----------------------------------------------------------------------
    # Phase 1: Warmup training (both conditions identical)
    # -----------------------------------------------------------------------
    agent.train()

    for ep in range(warmup_episodes):
        _, obs_dict = env.reset()
        agent.reset()
        ep_harm = 0.0

        for _ in range(steps_per_episode):
            obs_body  = torch.tensor(obs_dict["body_state"],  dtype=torch.float32)
            obs_world = torch.tensor(obs_dict["world_state"], dtype=torch.float32)

            z_self_t = None
            if agent._current_latent is not None:
                z_self_t = agent._current_latent.z_self.detach().clone()

            latent = agent.sense(obs_body, obs_world)
            ticks = agent.clock.advance()
            e1_prior = (
                agent._e1_tick(latent) if ticks["e1_tick"]
                else torch.zeros(1, world_dim, device=agent.device)
            )

            candidates = agent.generate_trajectories(latent, e1_prior, ticks)
            action = agent.select_action(candidates, ticks, temperature=1.0)

            if z_self_t is not None:
                agent.record_transition(z_self_t, action, latent.z_self.detach().clone())

            if random.random() < nav_bias:
                action = torch.zeros(1, ACTION_DIM)
                action[0, random.randint(0, ACTION_DIM - 1)] = 1.0

            _, reward, done, _, obs_dict = env.step(action)
            harm_signal = float(reward) if reward < 0 else 0.0
            ep_harm += abs(harm_signal)

            e1_loss = agent.compute_prediction_loss()
            e2_loss = agent.compute_e2_loss()
            total_e1_e2 = e1_loss + e2_loss
            if total_e1_e2.requires_grad:
                e1_opt.zero_grad()
                e2_opt.zero_grad()
                total_e1_e2.backward()
                e1_opt.step()
                e2_opt.step()

            if agent._current_latent is not None:
                z_world = agent._current_latent.z_world.detach()
                harm_target = torch.tensor(
                    [[1.0 if harm_signal < 0 else 0.0]], device=agent.device
                )
                harm_loss  = F.mse_loss(agent.e3.harm_eval(z_world), harm_target)
                maint_loss = agent.compute_self_maintenance_loss()
                total_e3 = harm_loss + maint_loss
                e3_opt.zero_grad()
                total_e3.backward()
                e3_opt.step()

            agent.update_residue(harm_signal)
            if done:
                break

        if (ep + 1) % 100 == 0:
            d_now = agent.compute_z_self_d_eff()
            d_str = f"{d_now:.4f}" if d_now is not None else "N/A"
            print(
                f"  [train] {cond_label} seed={seed} ep {ep+1}/{warmup_episodes}"
                f" harm={ep_harm:.3f} d_eff={d_str}",
                flush=True,
            )

    # -----------------------------------------------------------------------
    # Phase 2: Memory collection -- fill Hopfield memory from eval episodes
    # Memory is SHARED between conditions (same patterns; both agents trained
    # identically in Phase 1). Frozen after collection.
    # -----------------------------------------------------------------------
    print(
        f"  [mem collect] {cond_label} seed={seed} {memory_episodes} eps...",
        flush=True,
    )
    memory = HopfieldMemory(capacity=hopfield_capacity)
    agent.eval()

    d_eff_baseline: List[float] = []
    stab_baseline: List[float] = []
    baseline_norms: List[float] = []

    for _ in range(memory_episodes):
        _, obs_dict = env.reset()
        agent.reset()
        for _ in range(steps_per_episode):
            obs_body  = torch.tensor(obs_dict["body_state"],  dtype=torch.float32)
            obs_world = torch.tensor(obs_dict["world_state"], dtype=torch.float32)
            with torch.no_grad():
                latent = agent.sense(obs_body, obs_world)
                agent.clock.advance()
                agent._e1_tick(latent)

            z_self = latent.z_self.squeeze(0).detach().cpu().float()
            n = float(z_self.norm().item())
            if n < 1e-6:
                action_idx = random.randint(0, ACTION_DIM - 1)
                action = torch.zeros(1, ACTION_DIM)
                action[0, action_idx] = 1.0
                _, _, done, _, obs_dict = env.step(action)
                if done:
                    break
                continue

            memory.store(z_self)
            baseline_norms.append(n)
            d_eff_baseline.append(_compute_d_eff(z_self))
            # stability measured after storing (self-consistency check)
            stab_baseline.append(memory.stability(z_self))

            action_idx = random.randint(0, ACTION_DIM - 1)
            action = torch.zeros(1, ACTION_DIM)
            action[0, action_idx] = 1.0
            _, _, done, _, obs_dict = env.step(action)
            if done:
                break

    mean_norm_baseline = (
        sum(baseline_norms) / max(1, len(baseline_norms))
    )
    mean_d_eff_baseline = (
        sum(v for v in d_eff_baseline if v == v) / max(1, len(d_eff_baseline))
    )
    mean_stab_baseline = (
        sum(stab_baseline) / max(1, len(stab_baseline))
    )
    print(
        f"  [memory] filled {len(memory)}/{hopfield_capacity} patterns"
        f"  d_eff_base={mean_d_eff_baseline:.4f}"
        f"  stab_base={mean_stab_baseline:.4f}"
        f"  mean_norm={mean_norm_baseline:.4f}",
        flush=True,
    )

    # -----------------------------------------------------------------------
    # Build Regime 3 perturbation vector (coherent + unfamiliar)
    # -----------------------------------------------------------------------
    r3_vec = _build_r3_vector(memory, target_norm=mean_norm_baseline, seed=seed)
    r3_stab_analytic = memory.stability(r3_vec)
    r3_d_eff_analytic = _compute_d_eff(r3_vec)
    print(
        f"  [R3 vector] d_eff={r3_d_eff_analytic:.3f}"
        f"  stability={r3_stab_analytic:.4f}"
        f"  norm={r3_vec.norm().item():.4f}"
        f"  (target_norm={mean_norm_baseline:.4f})",
        flush=True,
    )

    # -----------------------------------------------------------------------
    # Phase 3: Post-perturbation evaluation
    # Inject R3 vector at first step of first eval episode; measure recovery.
    # HOPFIELD_ON: Hopfield familiarity gradient applied during TRAINING-like
    #   update within the eval loop (Hopfield loss only, no E1/E2/harm updates).
    # HOPFIELD_ABLATED: no Hopfield loss; D_eff gradient applied only at
    #   training time (eval runs without gradient updates in this phase).
    # Both run no-grad for sensing; only HOPFIELD_ON applies a Hopfield
    # gradient step on z_self via the latent_stack parameters.
    # -----------------------------------------------------------------------
    print(
        f"  [eval] {cond_label} seed={seed} {eval_episodes} eps...",
        flush=True,
    )
    stab_post: List[float] = []
    d_eff_post: List[float] = []

    # Hopfield-only optimizer (only for ON condition; covers latent_stack)
    hopfield_opt = (
        optim.Adam(agent.latent_stack.parameters(), lr=lr * 0.5)
        if hopfield_enabled else None
    )

    for ep_idx in range(eval_episodes):
        _, obs_dict = env.reset()
        agent.reset()

        # Inject R3 perturbation at first step of first eval episode
        inject_this_ep = ep_idx == 0

        for step in range(steps_per_episode):
            obs_body  = torch.tensor(obs_dict["body_state"],  dtype=torch.float32)
            obs_world = torch.tensor(obs_dict["world_state"], dtype=torch.float32)

            with torch.no_grad():
                latent = agent.sense(obs_body, obs_world)
                agent.clock.advance()
                agent._e1_tick(latent)

            # Inject R3 at first step of first episode
            if inject_this_ep and step == 0 and agent._current_latent is not None:
                with torch.no_grad():
                    agent._current_latent.z_self.copy_(
                        r3_vec.to(agent.device).unsqueeze(0)
                    )
                if not dry_run:
                    print(
                        f"  [perturb] {cond_label} seed={seed}"
                        f" R3 vector injected"
                        f"  r3_stab={r3_stab_analytic:.4f}"
                        f"  r3_d_eff={r3_d_eff_analytic:.3f}",
                        flush=True,
                    )
                inject_this_ep = False

            # Hopfield familiarity pull (ON only): one gradient step per step.
            # Re-run sense() with grad enabled so z_self has a grad_fn, then
            # compute familiarity loss and update latent_stack parameters.
            if (
                hopfield_enabled
                and hopfield_opt is not None
                and agent._current_latent is not None
            ):
                z_self_1d = agent._current_latent.z_self.squeeze(0).detach().cpu().float()
                nearest = memory.nearest_pattern_tensor(z_self_1d)
                if nearest is not None:
                    # Re-sense with grad tracking to get differentiable z_self
                    latent_grad = agent.sense(obs_body, obs_world)
                    z_self_grad = latent_grad.z_self           # [1, self_dim]
                    nearest_t   = nearest.to(agent.device).unsqueeze(0)  # [1, d]
                    z_norm   = F.normalize(z_self_grad, dim=-1)
                    n_norm   = F.normalize(nearest_t.detach(), dim=-1)
                    cos_sim  = (z_norm * n_norm).sum()
                    hopfield_loss = hopfield_weight * (1.0 - cos_sim)
                    hopfield_opt.zero_grad()
                    hopfield_loss.backward()
                    hopfield_opt.step()

            # Measure current stability + D_eff
            if agent._current_latent is not None:
                z_curr = agent._current_latent.z_self.squeeze(0).detach().cpu().float()
                n_curr = float(z_curr.norm().item())
                if n_curr > 1e-6:
                    stab_post.append(memory.stability(z_curr))
                    d_eff_post.append(_compute_d_eff(z_curr))

            action_idx = random.randint(0, ACTION_DIM - 1)
            action = torch.zeros(1, ACTION_DIM)
            action[0, action_idx] = 1.0
            _, _, done, _, obs_dict = env.step(action)
            if done:
                break

    mean_stab_post = (
        sum(stab_post) / max(1, len(stab_post))
    )
    mean_d_eff_post = (
        sum(v for v in d_eff_post if v == v) / max(1, len(d_eff_post))
    )

    print(
        f"  [post-eval] {cond_label} seed={seed}"
        f"  stab_post={mean_stab_post:.4f} (base={mean_stab_baseline:.4f})"
        f"  d_eff_post={mean_d_eff_post:.4f} (base={mean_d_eff_baseline:.4f})"
        f"  n_stab={len(stab_post)}",
        flush=True,
    )

    return {
        "seed": seed,
        "condition": cond_label,
        "hopfield_enabled": hopfield_enabled,
        "mean_stab_baseline": mean_stab_baseline,
        "mean_stab_post": mean_stab_post,
        "mean_d_eff_baseline": mean_d_eff_baseline,
        "mean_d_eff_post": mean_d_eff_post,
        "r3_stab_analytic": r3_stab_analytic,
        "r3_d_eff_analytic": r3_d_eff_analytic,
        "n_stab_post": len(stab_post),
        "n_memory_patterns": len(memory),
    }


# ---------------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------------

def run(
    seeds: Tuple[int, ...] = (42, 123),
    hopfield_weight: float = HOPFIELD_WEIGHT,
    hopfield_capacity: int = HOPFIELD_CAPACITY,
    maint_weight: float = MAINT_WEIGHT,
    d_eff_target: float = D_EFF_TARGET,
    warmup_episodes: int = 300,
    memory_episodes: int = 30,
    eval_episodes: int = 50,
    steps_per_episode: int = 200,
    self_dim: int = 32,
    world_dim: int = 32,
    lr: float = 1e-3,
    alpha_world: float = 0.9,
    alpha_self: float = 0.3,
    nav_bias: float = 0.35,
    dry_run: bool = False,
) -> dict:
    """MECH-118: HOPFIELD_ON vs HOPFIELD_ABLATED discriminative pair.

    Matched-seed design: same env per seed across conditions.
    Tests whether Hopfield familiarity is a distinct self-maintenance signal
    beyond D_eff homeostasis alone.
    """
    print(
        f"\n[V3-EXQ-143] MECH-118 Hopfield Familiarity Discriminative Pair"
        f" seeds={list(seeds)}"
        f" hopfield_weight={hopfield_weight}"
        f" hopfield_capacity={hopfield_capacity}"
        f" maint_weight={maint_weight}"
        f" d_eff_target={d_eff_target}",
        flush=True,
    )

    results_on:  List[Dict] = []
    results_abl: List[Dict] = []

    for seed in seeds:
        for hopfield_on in [True, False]:
            r = _run_single(
                seed=seed,
                hopfield_enabled=hopfield_on,
                hopfield_weight=hopfield_weight,
                maint_weight=maint_weight,
                d_eff_target=d_eff_target,
                warmup_episodes=warmup_episodes,
                memory_episodes=memory_episodes,
                eval_episodes=eval_episodes,
                steps_per_episode=steps_per_episode,
                hopfield_capacity=hopfield_capacity,
                self_dim=self_dim,
                world_dim=world_dim,
                lr=lr,
                alpha_world=alpha_world,
                alpha_self=alpha_self,
                nav_bias=nav_bias,
                dry_run=dry_run,
            )
            if hopfield_on:
                results_on.append(r)
            else:
                results_abl.append(r)

    def _avg(res: List[Dict], key: str) -> float:
        vals = [r[key] for r in res if r[key] == r[key]]  # skip nan
        return float(sum(vals) / max(1, len(vals)))

    # -----------------------------------------------------------------------
    # Pre-registered criteria
    # -----------------------------------------------------------------------

    # C1: HOPFIELD_ON stability post-perturb >= THRESH both seeds
    c1_per_seed = [
        r["mean_stab_post"] >= THRESH_STAB_RECOVERY_ON
        for r in results_on
    ]
    c1_pass = len(c1_per_seed) >= len(seeds) and all(c1_per_seed)

    # C2: HOPFIELD_ABLATED stability post-perturb <= THRESH both seeds
    c2_per_seed = [
        r["mean_stab_post"] <= THRESH_STAB_RECOVERY_ABL
        for r in results_abl
    ]
    c2_pass = len(c2_per_seed) >= len(seeds) and all(c2_per_seed)

    # C3: per-seed stab gap (ON_post - ABL_post) >= THRESH both seeds
    c3_per_seed = []
    per_seed_stab_gap: List[float] = []
    for r_on in results_on:
        matching = [r for r in results_abl if r["seed"] == r_on["seed"]]
        if matching:
            r_abl = matching[0]
            gap = r_on["mean_stab_post"] - r_abl["mean_stab_post"]
            per_seed_stab_gap.append(gap)
            c3_per_seed.append(gap >= THRESH_STAB_GAP)
    c3_pass = len(c3_per_seed) >= len(seeds) and all(c3_per_seed)

    # C4: D_eff difference between conditions <= THRESH (both seeds)
    #     Confirms Hopfield adds familiarity signal without disrupting D_eff control
    c4_per_seed = []
    per_seed_d_eff_diff: List[float] = []
    for r_on in results_on:
        matching = [r for r in results_abl if r["seed"] == r_on["seed"]]
        if matching:
            r_abl = matching[0]
            diff = abs(r_on["mean_d_eff_post"] - r_abl["mean_d_eff_post"])
            per_seed_d_eff_diff.append(diff)
            c4_per_seed.append(diff <= THRESH_D_EFF_CROSS)
    c4_pass = len(c4_per_seed) >= len(seeds) and all(c4_per_seed)

    # C5: data quality -- sufficient stability samples
    n_stab_min = min(r["n_stab_post"] for r in results_on + results_abl)
    c5_pass = n_stab_min >= THRESH_N_STAB

    all_pass = c1_pass and c2_pass and c3_pass and c4_pass and c5_pass
    criteria_met = sum([c1_pass, c2_pass, c3_pass, c4_pass, c5_pass])
    status = "PASS" if all_pass else "FAIL"

    if all_pass:
        decision = "retain_ree"
    elif c1_pass or c2_pass or c3_pass:
        decision = "hybridize"
    else:
        decision = "retire_ree_claim"

    # Summary averages
    mean_stab_post_on  = _avg(results_on,  "mean_stab_post")
    mean_stab_post_abl = _avg(results_abl, "mean_stab_post")
    mean_d_eff_post_on  = _avg(results_on,  "mean_d_eff_post")
    mean_d_eff_post_abl = _avg(results_abl, "mean_d_eff_post")
    mean_stab_base_on   = _avg(results_on,  "mean_stab_baseline")
    mean_stab_base_abl  = _avg(results_abl, "mean_stab_baseline")
    mean_r3_stab = _avg(results_on + results_abl, "r3_stab_analytic")
    mean_r3_d_eff = _avg(results_on + results_abl, "r3_d_eff_analytic")

    print(
        f"\n[V3-EXQ-143] Results:"
        f"  ON stab_base={mean_stab_base_on:.4f} stab_post={mean_stab_post_on:.4f}"
        f"  d_eff_post={mean_d_eff_post_on:.4f}",
        flush=True,
    )
    print(
        f"  ABL stab_base={mean_stab_base_abl:.4f} stab_post={mean_stab_post_abl:.4f}"
        f"  d_eff_post={mean_d_eff_post_abl:.4f}",
        flush=True,
    )
    print(
        f"  R3 perturbation: stab={mean_r3_stab:.4f} d_eff={mean_r3_d_eff:.3f}"
        f"  (low stab confirms coherent-unfamiliar perturbation)",
        flush=True,
    )
    print(
        f"  per_seed_stab_gap={[round(g, 4) for g in per_seed_stab_gap]}"
        f"  per_seed_d_eff_diff={[round(d, 4) for d in per_seed_d_eff_diff]}"
        f"  n_stab_min={n_stab_min}",
        flush=True,
    )
    print(
        f"  C1={c1_pass} C2={c2_pass} C3={c3_pass} C4={c4_pass} C5={c5_pass}"
        f"  decision={decision}  status={status} ({criteria_met}/5)",
        flush=True,
    )

    # Failure notes
    failure_notes: List[str] = []
    if not c1_pass:
        on_stabs = [round(r["mean_stab_post"], 4) for r in results_on]
        failure_notes.append(
            f"C1 FAIL: HOPFIELD_ON stab_post {on_stabs} < {THRESH_STAB_RECOVERY_ON}"
            " -- Hopfield loss does not recover familiarity after R3 perturbation;"
            " increase hopfield_weight or memory_episodes (more stored patterns)"
        )
    if not c2_pass:
        abl_stabs = [round(r["mean_stab_post"], 4) for r in results_abl]
        failure_notes.append(
            f"C2 FAIL: HOPFIELD_ABLATED stab_post {abl_stabs} > {THRESH_STAB_RECOVERY_ABL}"
            " -- ablated condition spontaneously recovers familiarity without Hopfield;"
            " R3 perturbation may not be sufficiently orthogonal to familiar subspace"
            " or D_eff homeostasis indirectly restores familiarity"
        )
    if not c3_pass:
        failure_notes.append(
            f"C3 FAIL: per-seed stab gap {[round(g, 4) for g in per_seed_stab_gap]}"
            f" < {THRESH_STAB_GAP} in some seeds;"
            " conditions do not discriminate -- increase hopfield_weight or eval episodes"
        )
    if not c4_pass:
        failure_notes.append(
            f"C4 FAIL: D_eff diff {[round(d, 4) for d in per_seed_d_eff_diff]}"
            f" > {THRESH_D_EFF_CROSS} in some seeds;"
            " Hopfield loss alters D_eff control -- signals are NOT orthogonal;"
            " reduce hopfield_weight or separate gradient paths"
        )
    if not c5_pass:
        failure_notes.append(
            f"C5 FAIL: n_stab_min={n_stab_min} < {THRESH_N_STAB}"
            " -- insufficient samples; increase eval_episodes or steps"
        )

    for note in failure_notes:
        print(f"  {note}", flush=True)

    # Interpretation
    if all_pass:
        interpretation = (
            f"MECH-118 SUPPORTED: Hopfield familiarity is a distinct self-maintenance"
            f" signal from D_eff coherence."
            f" HOPFIELD_ON recovered stability to {mean_stab_post_on:.4f} after R3"
            f" perturbation (base {mean_stab_base_on:.4f});"
            f" HOPFIELD_ABLATED stayed at {mean_stab_post_abl:.4f}."
            f" Per-seed gap {[round(g, 4) for g in per_seed_stab_gap]}."
            f" D_eff was equally controlled in both conditions"
            f" (diff {[round(d, 4) for d in per_seed_d_eff_diff]} <= {THRESH_D_EFF_CROSS}),"
            f" confirming orthogonality. The Hopfield familiarity loss provides information"
            f" that D_eff homeostasis alone cannot (Q-022 dissociation mechanistically confirmed)."
            f" MECH-119 (coherent-unfamiliar pathology) is a distinct regime from high-D_eff"
            f" dispersion. Proceed to combined gate design for ARC-031."
        )
    elif c3_pass:
        interpretation = (
            f"Partial support: stab_gap {[round(g, 4) for g in per_seed_stab_gap]}"
            f" observed but not all thresholds met."
            f" C1={c1_pass} C2={c2_pass} C3={c3_pass} C4={c4_pass} C5={c5_pass}."
            f" Directional effect is present but may need stronger Hopfield signal"
            f" or longer memory collection phase."
        )
    else:
        interpretation = (
            f"MECH-118 NOT SUPPORTED: Hopfield familiarity loss does not"
            f" produce discriminative stability recovery after R3 perturbation."
            f" ON stab_post={mean_stab_post_on:.4f};"
            f" ABL stab_post={mean_stab_post_abl:.4f}."
            f" Possible causes: R3 perturbation not sufficiently unfamiliar"
            f" (r3_stab_analytic={mean_r3_stab:.4f} should be near 0);"
            f" hopfield_weight too small to drive recovery;"
            f" or D_eff homeostasis incidentally restores familiarity (no dissociation)."
            f" Re-evaluate after Q-022 dissociation result (EXQ-084d)."
        )

    # Per-seed detail rows for summary markdown
    per_on_rows = "\n".join(
        f"  seed={r['seed']}: stab_base={r['mean_stab_baseline']:.4f}"
        f" stab_post={r['mean_stab_post']:.4f}"
        f" d_eff_base={r['mean_d_eff_baseline']:.4f}"
        f" d_eff_post={r['mean_d_eff_post']:.4f}"
        f" n_stab={r['n_stab_post']}"
        for r in results_on
    )
    per_abl_rows = "\n".join(
        f"  seed={r['seed']}: stab_base={r['mean_stab_baseline']:.4f}"
        f" stab_post={r['mean_stab_post']:.4f}"
        f" d_eff_base={r['mean_d_eff_baseline']:.4f}"
        f" d_eff_post={r['mean_d_eff_post']:.4f}"
        f" n_stab={r['n_stab_post']}"
        for r in results_abl
    )

    failure_section = ""
    if failure_notes:
        failure_section = "\n## Failure Notes\n\n" + "\n".join(
            f"- {n}" for n in failure_notes
        )

    summary_markdown = (
        f"# V3-EXQ-143 -- MECH-118 Hopfield Familiarity Discriminative Pair\n\n"
        f"**Status:** {status}\n"
        f"**Claims:** MECH-118\n"
        f"**Decision:** {decision}\n"
        f"**Seeds:** {list(seeds)}\n"
        f"**Conditions:** HOPFIELD_ON vs HOPFIELD_ABLATED\n"
        f"**Warmup:** {warmup_episodes} eps x {steps_per_episode} steps"
        f"  **Memory collection:** {memory_episodes} eps"
        f"  **Post-perturb eval:** {eval_episodes} eps\n"
        f"**Env:** CausalGridWorldV2 size=10, 3 hazards, 4 resources"
        f" nav_bias={nav_bias}\n"
        f"**Perturbation:** R3 coherent-unfamiliar vector (orthogonal to mean stored"
        f" pattern, same norm as baseline z_self)\n"
        f"**hopfield_weight:** {hopfield_weight}  "
        f"**hopfield_capacity:** {hopfield_capacity}  "
        f"**maint_weight:** {maint_weight}  "
        f"**d_eff_target:** {d_eff_target}\n\n"
        f"## Design\n\n"
        f"MECH-118 asserts Hopfield-style pattern familiarity is a distinct"
        f" self-maintenance signal from D_eff coherence. HOPFIELD_ON adds a"
        f" familiarity pull loss (toward nearest stored z_self pattern) on top of"
        f" the D_eff homeostatic loss. HOPFIELD_ABLATED uses D_eff homeostasis only."
        f" A Regime 3 (coherent-unfamiliar) perturbation is injected at eval start:"
        f" R3 vector has low D_eff (coherent, D_eff~1) but near-zero Hopfield stability"
        f" (unfamiliar direction). D_eff homeostasis cannot detect this state."
        f" HOPFIELD_ON should recover stability; HOPFIELD_ABLATED should not.\n\n"
        f"## R3 Perturbation Quality\n\n"
        f"| Metric | Value | Expectation |\n"
        f"|--------|-------|-------------|\n"
        f"| R3 stability (analytic) | {mean_r3_stab:.4f} | ~0 (unfamiliar) |\n"
        f"| R3 D_eff (analytic) | {mean_r3_d_eff:.3f} | ~1 (coherent) |\n\n"
        f"## Pre-Registered Thresholds\n\n"
        f"C1: HOPFIELD_ON stab_post >= {THRESH_STAB_RECOVERY_ON} both seeds"
        f" (familiarity recovery)\n"
        f"C2: HOPFIELD_ABLATED stab_post <= {THRESH_STAB_RECOVERY_ABL} both seeds"
        f" (no spontaneous recovery without Hopfield)\n"
        f"C3: per-seed stab_gap (ON_post - ABL_post) >= {THRESH_STAB_GAP} both seeds"
        f" (discriminative separation)\n"
        f"C4: D_eff difference |ON - ABL| <= {THRESH_D_EFF_CROSS} both seeds"
        f" (Hopfield adds familiarity signal, does NOT replace D_eff control)\n"
        f"C5: n_stab_samples >= {THRESH_N_STAB} per condition per seed (data quality)\n\n"
        f"## Results\n\n"
        f"| Condition | stab_baseline | stab_post | d_eff_baseline | d_eff_post |\n"
        f"|-----------|---------------|-----------|----------------|------------|\n"
        f"| HOPFIELD_ON      | {mean_stab_base_on:.4f}"
        f" | {mean_stab_post_on:.4f}"
        f" | -- | {mean_d_eff_post_on:.4f} |\n"
        f"| HOPFIELD_ABLATED | {mean_stab_base_abl:.4f}"
        f" | {mean_stab_post_abl:.4f}"
        f" | -- | {mean_d_eff_post_abl:.4f} |\n\n"
        f"**Per-seed stability gap (ON - ABL):"
        f" {[round(g, 4) for g in per_seed_stab_gap]}**\n"
        f"**Per-seed D_eff difference:"
        f" {[round(d, 4) for d in per_seed_d_eff_diff]}**\n\n"
        f"### HOPFIELD_ON per seed\n{per_on_rows}\n\n"
        f"### HOPFIELD_ABLATED per seed\n{per_abl_rows}\n\n"
        f"## PASS Criteria\n\n"
        f"| Criterion | Result |\n"
        f"|-----------|--------|\n"
        f"| C1: ON stab_post >= {THRESH_STAB_RECOVERY_ON} (both seeds)"
        f" | {'PASS' if c1_pass else 'FAIL'} |\n"
        f"| C2: ABL stab_post <= {THRESH_STAB_RECOVERY_ABL} (both seeds)"
        f" | {'PASS' if c2_pass else 'FAIL'} |\n"
        f"| C3: stab_gap >= {THRESH_STAB_GAP} (both seeds)"
        f" | {'PASS' if c3_pass else 'FAIL'} |\n"
        f"| C4: d_eff_diff <= {THRESH_D_EFF_CROSS} (both seeds)"
        f" | {'PASS' if c4_pass else 'FAIL'} |\n"
        f"| C5: n_stab >= {THRESH_N_STAB} (both conditions both seeds)"
        f" | {'PASS' if c5_pass else 'FAIL'} |\n\n"
        f"Criteria met: {criteria_met}/5 -> **{status}**\n\n"
        f"## Interpretation\n\n"
        f"{interpretation}\n"
        f"{failure_section}\n"
    )

    # Flat metrics for governance indexer
    metrics: Dict[str, float] = {
        "mean_stab_post_on":         float(mean_stab_post_on),
        "mean_stab_post_abl":        float(mean_stab_post_abl),
        "mean_stab_base_on":         float(mean_stab_base_on),
        "mean_stab_base_abl":        float(mean_stab_base_abl),
        "mean_d_eff_post_on":        float(mean_d_eff_post_on),
        "mean_d_eff_post_abl":       float(mean_d_eff_post_abl),
        "mean_r3_stab_analytic":     float(mean_r3_stab),
        "mean_r3_d_eff_analytic":    float(mean_r3_d_eff),
        "mean_stab_gap":             float(
            sum(per_seed_stab_gap) / max(1, len(per_seed_stab_gap))
        ),
        "mean_d_eff_diff":           float(
            sum(per_seed_d_eff_diff) / max(1, len(per_seed_d_eff_diff))
        ),
        "n_stab_min":                float(n_stab_min),
        "hopfield_weight":           float(hopfield_weight),
        "hopfield_capacity":         float(hopfield_capacity),
        "maint_weight":              float(maint_weight),
        "d_eff_target":              float(d_eff_target),
        "crit1_pass":                1.0 if c1_pass else 0.0,
        "crit2_pass":                1.0 if c2_pass else 0.0,
        "crit3_pass":                1.0 if c3_pass else 0.0,
        "crit4_pass":                1.0 if c4_pass else 0.0,
        "crit5_pass":                1.0 if c5_pass else 0.0,
        "criteria_met":              float(criteria_met),
    }

    return {
        "status": status,
        "metrics": metrics,
        "summary_markdown": summary_markdown,
        "claim_ids": CLAIM_IDS,
        "evidence_direction": (
            "supports" if all_pass
            else ("mixed" if criteria_met >= 3 else "weakens")
        ),
        "experiment_type": EXPERIMENT_TYPE,
        "fatal_error_count": 0,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import json
    from datetime import datetime, timezone

    parser = argparse.ArgumentParser(
        description="V3-EXQ-143 MECH-118 Hopfield Familiarity Discriminative Pair"
    )
    parser.add_argument("--seeds",           type=int, nargs="+", default=[42, 123])
    parser.add_argument("--hopfield-weight", type=float, default=HOPFIELD_WEIGHT)
    parser.add_argument("--hopfield-cap",    type=int,   default=HOPFIELD_CAPACITY)
    parser.add_argument("--maint-weight",    type=float, default=MAINT_WEIGHT)
    parser.add_argument("--d-eff-target",    type=float, default=D_EFF_TARGET)
    parser.add_argument("--warmup",          type=int,   default=300)
    parser.add_argument("--memory-eps",      type=int,   default=30)
    parser.add_argument("--eval-eps",        type=int,   default=50)
    parser.add_argument("--steps",           type=int,   default=200)
    parser.add_argument("--alpha-world",     type=float, default=0.9)
    parser.add_argument("--dry-run",         action="store_true")
    args = parser.parse_args()

    result = run(
        seeds=tuple(args.seeds),
        hopfield_weight=args.hopfield_weight,
        hopfield_capacity=args.hopfield_cap,
        maint_weight=args.maint_weight,
        d_eff_target=args.d_eff_target,
        warmup_episodes=args.warmup,
        memory_episodes=args.memory_eps,
        eval_episodes=args.eval_eps,
        steps_per_episode=args.steps,
        alpha_world=args.alpha_world,
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

    print(f"\nResult written to: {out_path}", flush=True)
    print(f"Status: {result['status']}", flush=True)
    for k, v in result["metrics"].items():
        print(f"  {k}: {v}", flush=True)
