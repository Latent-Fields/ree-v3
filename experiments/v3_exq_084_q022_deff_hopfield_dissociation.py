"""
V3-EXQ-084 -- Q-022 D_eff / Hopfield Stability Dissociation Test

Claims: Q-022, MECH-118, MECH-119
Proposal: EVB-0069

Q-022 asks: can D_eff (participation ratio = z_self coherence) and Hopfield
pattern familiarity dissociate in the REE architecture? If they always co-vary,
MECH-118/119 collapse into MECH-113 (one mechanism suffices). If they dissociate,
MECH-118 (pattern familiarity) and MECH-119 (coherent-unfamiliar pathology) are
genuine distinct signals requiring a combined gate.

Three regimes are applied analytically to collected z_self snapshots:
  R1 Normal:   z_self as produced by trained agent -- low D_eff, HIGH stability
  R2 Noise:    z_self + N(0, sigma^2*I) -- HIGH D_eff, low stability
  R3 Novelty:  unit vector orthogonal to mean(z_self)*||z||  -- D_eff~1, low stability

R3 is the dissociation case: concentrated (coherent) but outside the familiar subspace.
If R3 can be distinguished from R2 on D_eff AND from R1 on stability, the metrics
are independent -- D_eff and Hopfield familiarity measure different things.

HopfieldMemory: inline implementation (not in ree_core). Stores 64 z_self patterns
during memory-collection phase. Stability = max cosine similarity to stored patterns.
Reference: Ramsauer et al. 2021 (modern Hopfield networks); MECH-118 spec (LRU-64).

Phase structure:
  Phase 1 -- Warmup training (warmup_episodes episodes)
  Phase 2 -- Memory collection (memory_eps eval episodes) -> fills HopfieldMemory
  Phase 3 -- Analysis collection (analysis_eps eval episodes) -> R1/R2/R3 metrics
                                                                  (memory fixed)
  Phase 4 -- Aggregate, PASS criteria, write JSON

PASS criteria (ALL required for dissociation confirmation):
  C1: mean_d_eff_r3 < mean_d_eff_r2 - 1.0  (R3 more coherent than noise)
  C2: mean_stab_r3  < mean_stab_r1  - 0.05 (R3 less familiar than baseline)
  C3: mean_d_eff_r2 > mean_d_eff_r1  + 1.0 (noise raises D_eff above baseline)
  C4: mean_stab_r2  < mean_stab_r1  - 0.05 (noise lowers stability below baseline)

Decision outcomes:
  PASS: D_eff and Hopfield stability dissociate -> MECH-118/119 are distinct
        mechanisms; proceed to combined gate design for ARC-031.
  FAIL: Metrics co-vary -> MECH-118/119 collapse into MECH-113; no additional
        Hopfield infrastructure needed.

Architecture epoch: ree_hybrid_guardrails_v1
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import random
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
import torch.optim as optim

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig


EXPERIMENT_TYPE = "v3_exq_084_q022_deff_hopfield_dissociation"
CLAIM_IDS = ["Q-022", "MECH-118", "MECH-119"]


# ---------------------------------------------------------------------------
# Inline HopfieldMemory (MECH-118 design; Ramsauer 2021)
# ---------------------------------------------------------------------------

class HopfieldMemory:
    """
    Lightweight modern Hopfield stability probe.

    Stores up to `capacity` z_self snapshots (FIFO).
    stability(z) = max cosine similarity to all stored patterns.
    High stability -> current z_self resembles a familiar stored state.
    Low stability  -> current z_self is in unfamiliar territory.

    Does NOT require ree_core changes. Instantiated per run.
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

    def __len__(self) -> int:
        return len(self.patterns)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_d_eff(z_1d: torch.Tensor) -> float:
    """
    Participation ratio D_eff = (sum|z|)^2 / sum(z^2).

    From epistemic-mapping/Epistemic_monitor.py. Matches agent.compute_z_self_d_eff().
    High D_eff = diffuse/uncertain; low D_eff = coherent/focused.
    """
    abs_z = z_1d.abs()
    den = z_1d.pow(2).sum()
    if den.item() < 1e-8:
        return float("nan")
    return float((abs_z.sum().pow(2) / den).item())


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
    Construct a unit vector orthogonal to the mean of stored memory patterns,
    scaled to target_norm.

    This is the Regime 3 perturbation direction: concentrated (low D_eff ~ 1)
    but pointing away from familiar z_self territory (low Hopfield stability).

    Gram-Schmidt step removes the mean-pattern component from a random vector.
    """
    if not memory.patterns:
        raise RuntimeError("Cannot build R3 vector: memory is empty")

    torch.manual_seed(seed + 9999)  # fixed seed for reproducible R3 direction
    M = torch.stack(memory.patterns)                            # [K, d]
    mean_z = F.normalize(M.mean(0).unsqueeze(0), dim=-1).squeeze(0)  # [d]

    # Random vector, then project out the mean direction (Gram-Schmidt)
    rand = torch.randn(mean_z.shape)
    rand = rand - (rand @ mean_z) * mean_z
    norm = rand.norm()
    if norm.item() < 1e-8:
        # Fallback: use mean's orthogonal complement in the first unused dimension
        dim = mean_z.shape[0]
        rand = torch.zeros(dim)
        rand[dim - 1] = 1.0
        rand = rand - (rand @ mean_z) * mean_z
        norm = rand.norm()

    r3_dir = rand / norm           # unit vector orthogonal to mean
    return r3_dir * target_norm    # scaled to target_norm


# ---------------------------------------------------------------------------
# Main run
# ---------------------------------------------------------------------------

def run(
    seed: int = 42,
    warmup_episodes: int = 300,
    memory_eps: int = 50,
    analysis_eps: int = 100,
    steps_per_episode: int = 200,
    noise_sigma: float = 2.0,
    hopfield_capacity: int = 64,
    self_dim: int = 32,
    world_dim: int = 32,
    lr: float = 1e-3,
    alpha_world: float = 0.9,
    **kwargs,
) -> dict:
    """
    Q-022: D_eff / Hopfield stability dissociation test.
    Single agent, three analytical perturbation regimes applied to z_self.
    """
    print(
        f"\n[EXQ-084] Q-022 D_eff/Hopfield dissociation  seed={seed}"
        f"  warmup={warmup_episodes}  mem_eps={memory_eps}  anal_eps={analysis_eps}"
        f"  noise_sigma={noise_sigma}  hopfield_cap={hopfield_capacity}",
        flush=True,
    )

    torch.manual_seed(seed)
    random.seed(seed)

    env = _make_env(seed)

    config = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=self_dim,
        world_dim=world_dim,
        alpha_world=alpha_world,
        alpha_self=0.3,
        self_maintenance_weight=0.0,  # off -- measuring natural z_self
    )
    agent = REEAgent(config)

    e1_opt = optim.Adam(agent.e1.parameters(), lr=lr)
    e3_opt = optim.Adam(
        list(agent.e3.parameters()) + list(agent.latent_stack.parameters()),
        lr=lr,
    )
    e2_opt = optim.Adam(agent.e2.parameters(), lr=lr * 3)

    # -----------------------------------------------------------------------
    # Phase 1: Warmup training
    # -----------------------------------------------------------------------
    print(f"[EXQ-084] Phase 1: warmup ({warmup_episodes} eps)...", flush=True)
    agent.train()

    for ep in range(warmup_episodes):
        _, obs_dict = env.reset()
        agent.reset()

        for _ in range(steps_per_episode):
            obs_body  = torch.tensor(obs_dict["body_state"],  dtype=torch.float32)
            obs_world = torch.tensor(obs_dict["world_state"], dtype=torch.float32)

            z_self_prev = None
            if agent._current_latent is not None:
                z_self_prev = agent._current_latent.z_self.detach().clone()

            latent = agent.sense(obs_body, obs_world)
            ticks = agent.clock.advance()
            e1_prior = agent._e1_tick(latent) if ticks["e1_tick"] else torch.zeros(
                1, world_dim, device=agent.device
            )
            candidates = agent.generate_trajectories(latent, e1_prior, ticks)
            action = agent.select_action(candidates, ticks, temperature=1.0)

            if z_self_prev is not None:
                agent.record_transition(z_self_prev, action, latent.z_self.detach())

            _, reward, done, info, obs_dict = env.step(action)
            harm_signal = float(reward) if reward < 0 else 0.0

            e1_loss = agent.compute_prediction_loss()
            if e1_loss.requires_grad:
                e1_opt.zero_grad()
                e1_loss.backward()
                e1_opt.step()

            e2_loss = agent.compute_e2_loss()
            if e2_loss.requires_grad:
                e2_opt.zero_grad()
                e2_loss.backward()
                e2_opt.step()

            if agent._current_latent is not None:
                z_world = agent._current_latent.z_world.detach()
                harm_target = torch.tensor(
                    [[1.0 if harm_signal < 0 else 0.0]], device=agent.device
                )
                harm_loss = F.mse_loss(agent.e3.harm_eval(z_world), harm_target)
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
            d_str = f"{d_now:.3f}" if d_now is not None else "N/A"
            print(
                f"  [train] ep {ep+1}/{warmup_episodes}  d_eff={d_str}",
                flush=True,
            )

    # -----------------------------------------------------------------------
    # Phase 2: Memory collection (eval, no grad, fill HopfieldMemory)
    # -----------------------------------------------------------------------
    print(
        f"[EXQ-084] Phase 2: memory collection ({memory_eps} eps)...",
        flush=True,
    )
    memory = HopfieldMemory(capacity=hopfield_capacity)
    agent.eval()

    for _ in range(memory_eps):
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
            memory.store(z_self)

            action_idx = random.randint(0, env.action_dim - 1)
            action = torch.zeros(1, env.action_dim)
            action[0, action_idx] = 1.0
            _, _, done, _, obs_dict = env.step(action)
            if done:
                break

    print(
        f"  [memory] filled {len(memory)}/{hopfield_capacity} patterns",
        flush=True,
    )

    # -----------------------------------------------------------------------
    # Build Regime 3 vector (fixed after memory collection, before analysis)
    # -----------------------------------------------------------------------
    # Compute mean z_self norm from memory (R3 vectors will be scaled to this)
    M_tensor = torch.stack(memory.patterns)  # [K, self_dim]
    mean_norm = float(M_tensor.norm(dim=-1).mean().item())

    r3_vec = _build_r3_vector(memory, target_norm=mean_norm, seed=seed)
    # Verify R3 properties analytically
    r3_d_eff_analytic = _compute_d_eff(r3_vec)
    r3_stab_analytic  = memory.stability(r3_vec)
    print(
        f"  [R3 vector] d_eff={r3_d_eff_analytic:.3f}"
        f"  stability={r3_stab_analytic:.4f}"
        f"  norm={r3_vec.norm().item():.3f}"
        f"  mean_pattern_norm={mean_norm:.3f}",
        flush=True,
    )

    # -----------------------------------------------------------------------
    # Phase 3: Analysis collection (eval, no grad, measure R1/R2/R3 per step)
    # -----------------------------------------------------------------------
    print(
        f"[EXQ-084] Phase 3: analysis ({analysis_eps} eps)...",
        flush=True,
    )
    r1_d_eff: List[float] = []
    r1_stab:  List[float] = []
    r2_d_eff: List[float] = []
    r2_stab:  List[float] = []
    r3_d_eff: List[float] = []
    r3_stab:  List[float] = []

    for _ in range(analysis_eps):
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
            z_norm = float(z_self.norm().item())

            if z_norm < 1e-6:
                action_idx = random.randint(0, env.action_dim - 1)
                action = torch.zeros(1, env.action_dim)
                action[0, action_idx] = 1.0
                _, _, done, _, obs_dict = env.step(action)
                if done:
                    break
                continue

            # R1: raw z_self
            r1_d_eff.append(_compute_d_eff(z_self))
            r1_stab.append(memory.stability(z_self))

            # R2: Gaussian noise perturbation
            z_r2 = z_self + torch.randn_like(z_self) * noise_sigma
            r2_d_eff.append(_compute_d_eff(z_r2))
            r2_stab.append(memory.stability(z_r2))

            # R3: novel direction (fixed R3 vector scaled to z_norm)
            z_r3 = r3_vec * (z_norm / max(r3_vec.norm().item(), 1e-8))
            r3_d_eff.append(_compute_d_eff(z_r3))
            r3_stab.append(memory.stability(z_r3))

            action_idx = random.randint(0, env.action_dim - 1)
            action = torch.zeros(1, env.action_dim)
            action[0, action_idx] = 1.0
            _, _, done, _, obs_dict = env.step(action)
            if done:
                break

    def _mean(lst: List[float]) -> float:
        clean = [x for x in lst if x == x]  # drop nan
        return float(sum(clean) / max(1, len(clean)))

    mean_d_eff_r1 = _mean(r1_d_eff)
    mean_stab_r1  = _mean(r1_stab)
    mean_d_eff_r2 = _mean(r2_d_eff)
    mean_stab_r2  = _mean(r2_stab)
    mean_d_eff_r3 = _mean(r3_d_eff)
    mean_stab_r3  = _mean(r3_stab)

    print(
        f"\n[EXQ-084] Results  (n={len(r1_d_eff)} steps):",
        flush=True,
    )
    print(
        f"  R1 (normal):   d_eff={mean_d_eff_r1:.3f}  stability={mean_stab_r1:.4f}",
        flush=True,
    )
    print(
        f"  R2 (noise):    d_eff={mean_d_eff_r2:.3f}  stability={mean_stab_r2:.4f}",
        flush=True,
    )
    print(
        f"  R3 (novelty):  d_eff={mean_d_eff_r3:.3f}  stability={mean_stab_r3:.4f}",
        flush=True,
    )

    # -----------------------------------------------------------------------
    # PASS criteria
    # -----------------------------------------------------------------------
    c1_pass = mean_d_eff_r3 < mean_d_eff_r2 - 1.0   # R3 more coherent than noise
    c2_pass = mean_stab_r3  < mean_stab_r1  - 0.05   # R3 less familiar than baseline
    c3_pass = mean_d_eff_r2 > mean_d_eff_r1 + 1.0    # noise raises D_eff
    c4_pass = mean_stab_r2  < mean_stab_r1  - 0.05   # noise lowers stability

    all_pass     = c1_pass and c2_pass and c3_pass and c4_pass
    criteria_met = sum([c1_pass, c2_pass, c3_pass, c4_pass])
    status       = "PASS" if all_pass else "FAIL"

    print(f"\n  C1 (d_eff_r3 < d_eff_r2-1.0): {'PASS' if c1_pass else 'FAIL'}"
          f"  [{mean_d_eff_r3:.3f} < {mean_d_eff_r2-1.0:.3f}]", flush=True)
    print(f"  C2 (stab_r3  < stab_r1-0.05):  {'PASS' if c2_pass else 'FAIL'}"
          f"  [{mean_stab_r3:.4f} < {mean_stab_r1-0.05:.4f}]", flush=True)
    print(f"  C3 (d_eff_r2 > d_eff_r1+1.0):  {'PASS' if c3_pass else 'FAIL'}"
          f"  [{mean_d_eff_r2:.3f} > {mean_d_eff_r1+1.0:.3f}]", flush=True)
    print(f"  C4 (stab_r2  < stab_r1-0.05):  {'PASS' if c4_pass else 'FAIL'}"
          f"  [{mean_stab_r2:.4f} < {mean_stab_r1-0.05:.4f}]", flush=True)
    print(f"  Status: {status} ({criteria_met}/4)", flush=True)

    # -----------------------------------------------------------------------
    # Interpretation
    # -----------------------------------------------------------------------
    if all_pass:
        interpretation = (
            "DISSOCIATION CONFIRMED: D_eff and Hopfield stability measure distinct"
            " properties of z_self. R3 (coherent-unfamiliar) has D_eff similar to"
            " R1 baseline but stability similar to R2 noise. This validates Q-022:"
            " the architecture supports a 'coherent but unfamiliar' self-state regime."
            " MECH-118 (Hopfield familiarity signal) and MECH-119 (coherent-unfamiliar"
            " pathology) are distinct from MECH-113 (D_eff homeostasis)."
            " Proceed to combined gate design for ARC-031."
        )
    elif criteria_met >= 2:
        interpretation = (
            "PARTIAL DISSOCIATION: Some metric independence present but below threshold."
            " Either z_self dynamics are too uniform (agent undertrained), or the"
            " orthogonal direction chosen for R3 is still within the familiar subspace."
            " Consider longer warmup or stronger env_drift to create richer z_self variety."
        )
    else:
        interpretation = (
            "NO DISSOCIATION: D_eff and Hopfield stability co-vary. The architecture"
            " does not support a coherent-unfamiliar regime with current training."
            " Either z_self spans all dimensions evenly (metrics always correlated),"
            " or the EMA alpha=0.9 z_world training signal is insufficient to produce"
            " a structured z_self manifold. MECH-118/119 may collapse into MECH-113."
            " Re-evaluate after SD-010 (nociceptive separation) is implemented."
        )

    failure_notes: List[str] = []
    if not c1_pass:
        failure_notes.append(
            f"C1 FAIL: R3 d_eff={mean_d_eff_r3:.3f} not < R2 d_eff-1.0={mean_d_eff_r2-1.0:.3f}"
            " -- R3 vector may not be coherent (check r3_d_eff_analytic in metrics)"
        )
    if not c2_pass:
        failure_notes.append(
            f"C2 FAIL: R3 stability={mean_stab_r3:.4f} not < R1 stability-0.05={mean_stab_r1-0.05:.4f}"
            " -- orthogonal R3 direction is still within familiar subspace"
        )
    if not c3_pass:
        failure_notes.append(
            f"C3 FAIL: R2 d_eff={mean_d_eff_r2:.3f} not > R1 d_eff+1.0={mean_d_eff_r1+1.0:.3f}"
            " -- noise_sigma={noise_sigma} may be too small to disperse z_self"
        )
    if not c4_pass:
        failure_notes.append(
            f"C4 FAIL: R2 stability={mean_stab_r2:.4f} not < R1 stability-0.05={mean_stab_r1-0.05:.4f}"
            " -- noisy z_self still matching stored patterns (patterns are too diffuse)"
        )
    for n in failure_notes:
        print(f"  {n}", flush=True)

    failure_section = ""
    if failure_notes:
        failure_section = "\n## Failure Notes\n\n" + "\n".join(
            f"- {n}" for n in failure_notes
        )

    summary_markdown = (
        f"# V3-EXQ-084 -- Q-022 D_eff / Hopfield Stability Dissociation\n\n"
        f"**Status:** {status}\n"
        f"**Claims:** Q-022, MECH-118, MECH-119\n"
        f"**Seed:** {seed}  "
        f"**Warmup:** {warmup_episodes}  "
        f"**Memory eps:** {memory_eps}  "
        f"**Analysis eps:** {analysis_eps}\n"
        f"**noise_sigma:** {noise_sigma}  "
        f"**hopfield_capacity:** {hopfield_capacity}\n"
        f"**alpha_world:** {alpha_world}  (SD-008)\n\n"
        f"## Three Regimes\n\n"
        f"| Regime | D_eff (mean) | Stability (mean) | Description |\n"
        f"|--------|-------------|-----------------|-------------|\n"
        f"| R1 Normal   | {mean_d_eff_r1:.3f} | {mean_stab_r1:.4f} | raw z_self |\n"
        f"| R2 Noise    | {mean_d_eff_r2:.3f} | {mean_stab_r2:.4f} | z_self + N(0,{noise_sigma}^2) |\n"
        f"| R3 Novelty  | {mean_d_eff_r3:.3f} | {mean_stab_r3:.4f} | orthogonal unit vec * norm |\n\n"
        f"R3 analytic: d_eff={r3_d_eff_analytic:.3f}  stability={r3_stab_analytic:.4f}"
        f"  (measured before analysis phase)\n\n"
        f"## PASS Criteria\n\n"
        f"| Criterion | Result | Value vs Threshold |\n"
        f"|-----------|--------|--------------------|\n"
        f"| C1: d_eff_R3 < d_eff_R2 - 1.0 | {'PASS' if c1_pass else 'FAIL'}"
        f" | {mean_d_eff_r3:.3f} < {mean_d_eff_r2-1.0:.3f} |\n"
        f"| C2: stab_R3 < stab_R1 - 0.05  | {'PASS' if c2_pass else 'FAIL'}"
        f" | {mean_stab_r3:.4f} < {mean_stab_r1-0.05:.4f} |\n"
        f"| C3: d_eff_R2 > d_eff_R1 + 1.0 | {'PASS' if c3_pass else 'FAIL'}"
        f" | {mean_d_eff_r2:.3f} > {mean_d_eff_r1+1.0:.3f} |\n"
        f"| C4: stab_R2 < stab_R1 - 0.05  | {'PASS' if c4_pass else 'FAIL'}"
        f" | {mean_stab_r2:.4f} < {mean_stab_r1-0.05:.4f} |\n\n"
        f"Criteria met: {criteria_met}/4 -> **{status}**\n\n"
        f"## Interpretation\n\n"
        f"{interpretation}\n"
        f"{failure_section}\n"
    )

    metrics: Dict[str, float] = {
        "mean_d_eff_r1":          float(mean_d_eff_r1),
        "mean_stab_r1":           float(mean_stab_r1),
        "mean_d_eff_r2":          float(mean_d_eff_r2),
        "mean_stab_r2":           float(mean_stab_r2),
        "mean_d_eff_r3":          float(mean_d_eff_r3),
        "mean_stab_r3":           float(mean_stab_r3),
        "r3_d_eff_analytic":      float(r3_d_eff_analytic),
        "r3_stab_analytic":       float(r3_stab_analytic),
        "r3_mean_norm":           float(mean_norm),
        "noise_sigma":            float(noise_sigma),
        "hopfield_capacity":      float(hopfield_capacity),
        "hopfield_patterns_stored": float(len(memory)),
        "n_analysis_steps":       float(len(r1_d_eff)),
        "alpha_world":            float(alpha_world),
        "crit1_pass":             1.0 if c1_pass else 0.0,
        "crit2_pass":             1.0 if c2_pass else 0.0,
        "crit3_pass":             1.0 if c3_pass else 0.0,
        "crit4_pass":             1.0 if c4_pass else 0.0,
        "criteria_met":           float(criteria_met),
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
    parser.add_argument("--seed",          type=int,   default=42)
    parser.add_argument("--warmup",        type=int,   default=300)
    parser.add_argument("--memory-eps",    type=int,   default=50)
    parser.add_argument("--analysis-eps",  type=int,   default=100)
    parser.add_argument("--steps",         type=int,   default=200)
    parser.add_argument("--noise-sigma",   type=float, default=2.0)
    parser.add_argument("--hopfield-cap",  type=int,   default=64)
    parser.add_argument("--alpha-world",   type=float, default=0.9)
    args = parser.parse_args()

    result = run(
        seed=args.seed,
        warmup_episodes=args.warmup,
        memory_eps=args.memory_eps,
        analysis_eps=args.analysis_eps,
        steps_per_episode=args.steps,
        noise_sigma=args.noise_sigma,
        hopfield_capacity=args.hopfield_cap,
        alpha_world=args.alpha_world,
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
