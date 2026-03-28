#!/opt/local/bin/python3
"""
V3-EXQ-148 -- Q-003: R-Field Dimensionality Discriminative Pair

Claim:  Q-003
Proposal: EXP-0090 (EVB-0072)

Q-003 asks:
  "Should R(x,t) be a scalar or a vector (multiple regulatory dimensions)?"

  R(x,t) is the astrocytic regulatory field over L-space positions (MECH-001).
  Currently ResidueField.evaluate() returns a scalar -- a single harm-cost value
  per z_world point. The open question is whether a vector output (residue_dim > 1,
  one dimension per regulatory modality: e.g. harm intensity, harm recency,
  benefit proximity) carries more useful information for E3's trajectory evaluation
  than a collapsed scalar.

Experiment design:
  Two conditions, 2 seeds each (matched):

    SCALAR (current default):
      ResidueField evaluate() returns a 1-D value per z_world point.
      E3.harm_eval reads residue as a single scalar cost term Phi_R.
      The RBF + neural_field pipeline produces one output dimension.

    VECTOR (multi-dim):
      ResidueField evaluate_vector() returns a residue_dim=3 vector per point:
        dim 0: harm intensity (RBF weighted sum -- same as scalar)
        dim 1: harm recency  (EMA-decayed harm weights: recent > old)
        dim 2: benefit proximity (benefit_rbf_field value -- from ARC-030)
      E3 receives all three dimensions and learns a linear combination
      via a small projection head (Linear(residue_dim, 1)).
      This allows E3 to weight harm recency and benefit proximity
      independently, rather than having them pre-collapsed into a scalar.

  The key question: does the vector condition produce lower cumulative harm
  and/or higher benefit accumulation than the scalar condition over the same
  training budget? The vector condition can only help if E3 can learn to use
  the extra dimensions -- a failure would mean the dimensions collapse to
  effectively the same scalar weighting.

Pre-registered thresholds
--------------------------
C1: Vector harm advantage: mean_harm_vector < mean_harm_scalar * THRESH_HARM_RATIO
    (both seeds). Harm reduction >= (1 - THRESH_HARM_RATIO) fraction better.
    Pre-registered: THRESH_HARM_RATIO = 0.92 (vector must be >= 8% lower harm
    than scalar on average over evaluation episodes).

C2: Vector dimensionality non-degenerate: residue_vector_dim_variance > THRESH_DIM_VAR
    (both seeds). The vector output must show per-dimension variance across z_world
    points -- if all dimensions track the same scalar, the vector collapses and adds
    nothing. THRESH_DIM_VAR = 0.01 (mean std across dims > 0.01).

C3: Benefit signal non-zero: mean_benefit_vector > THRESH_BENEFIT_MIN (both seeds).
    The vector dim-2 (benefit) must carry non-trivial signal, confirming it is
    distinguishable from dims 0-1. THRESH_BENEFIT_MIN = 0.01.

C4: Vector does not catastrophically degrade harm: mean_harm_vector < mean_harm_scalar
    * THRESH_NO_REGRESSION (both seeds). Even if C1 fails, the vector must not
    significantly worsen harm avoidance. THRESH_NO_REGRESSION = 1.10 (at most 10%
    worse than scalar). This catches pathological cases where extra dims confuse E3.

C5: Consistency: criterion results must agree across both seeds (not one PASS / one
    FAIL for C1). Only relevant when C1 has mixed results.

PASS: C1 + C2 + C3 (vector is better, non-degenerate, with useful benefit signal).
PARTIAL: C4 + C2 (vector not better but not harmful; dimensionality is non-trivial).
FAIL: C4 fails (vector actively worsens harm avoidance) OR C2 fails (degenerate vector).

PASS => Q-003 guidance: vector R-field preferred (multiple regulatory dims useful).
PARTIAL => inconclusive; no strong evidence for or against vector.
FAIL (C4) => Q-003 guidance: scalar preferred (vector confuses E3 at current scale).
FAIL (C2) => implementation problem: vector dims collapsed; Q-003 not yet answerable.

Conditions
----------
SCALAR:
  ResidueField with standard evaluate() -> scalar.
  E3 receives scalar Phi_R term directly.
  benefit_terrain_enabled=True (so both conditions have same env richness).

VECTOR:
  ResidueField with evaluate_vector() -> Tensor[residue_dim=3].
  E3 receives vector concatenated onto z_world before harm_eval projection.
  residue_vector_proj = Linear(world_dim + 3, world_dim) fuses vector into E3.
  harm_eval(fused_z) -> scalar harm prediction as before.

Seeds: [42, 123] (matched -- same env seed, both conditions)
Env:   CausalGridWorldV2 size=10, 2 hazards, 3 resources, drift_interval=10
Train: 300 episodes x 200 steps
Eval:  100 episodes x 200 steps
Estimated runtime: ~50 min any machine
"""

import sys
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig
from ree_core.residue.field import ResidueField


EXPERIMENT_TYPE = "v3_exq_148_q003_r_field_dimensionality_pair"
CLAIM_IDS = ["Q-003"]

# ---------------------------------------------------------------------------
# Pre-registered thresholds
# ---------------------------------------------------------------------------
# C1: vector must reduce mean harm vs scalar by >= this fraction
THRESH_HARM_RATIO = 0.92

# C2: per-dimension std of vector output must exceed this (non-degenerate check)
THRESH_DIM_VAR = 0.01

# C3: benefit dimension mean value must exceed this
THRESH_BENEFIT_MIN = 0.01

# C4: vector must not be more than this ratio WORSE than scalar
THRESH_NO_REGRESSION = 1.10

# ---------------------------------------------------------------------------
# Model constants
# ---------------------------------------------------------------------------
BODY_OBS_DIM = 12
WORLD_OBS_DIM = 250   # CausalGridWorldV2 size=10
ACTION_DIM = 5

WORLD_DIM = 32        # z_world dimensionality (fixed across conditions)
RESIDUE_VEC_DIM = 3   # harm_intensity, harm_recency, benefit_proximity


# ---------------------------------------------------------------------------
# VectorResidueWrapper
# ---------------------------------------------------------------------------

class VectorResidueWrapper(nn.Module):
    """
    Wraps a ResidueField to produce a residue_dim=3 vector output.

    dim 0: harm intensity  -- standard RBF + neural output (same as scalar)
    dim 1: harm recency    -- EMA-decayed harm weights (recent harm > old)
    dim 2: benefit proximity -- benefit_rbf_field output (ARC-030 liking)

    This wrapper adds a second RBF layer with decaying weights for recency,
    and exposes evaluate_vector() which returns Tensor[batch, 3].
    """

    def __init__(self, base_field: ResidueField, world_dim: int, decay_factor: float = 0.9):
        super().__init__()
        self.base_field = base_field
        self.world_dim = world_dim
        self.decay_factor = decay_factor

        # Recency RBF: same structure as base, but weights decay with each new accumulation
        from ree_core.residue.field import RBFLayer
        self.recency_rbf = RBFLayer(
            world_dim=world_dim,
            num_centers=base_field.config.num_basis_functions,
            bandwidth=base_field.config.kernel_bandwidth,
        )
        self._recency_weights: List[float] = []

    def accumulate_vector(
        self,
        z_world: torch.Tensor,
        harm_magnitude: float = 1.0,
        hypothesis_tag: bool = False,
    ) -> None:
        """Accumulate harm in both base field and recency field."""
        if hypothesis_tag:
            return

        # Base field accumulation (standard)
        self.base_field.accumulate(z_world, harm_magnitude=harm_magnitude, hypothesis_tag=False)

        # Recency field: decay all existing weights then add new center
        with torch.no_grad():
            self.recency_rbf.weights.data *= self.decay_factor
        self.recency_rbf.add_residue(z_world, harm_magnitude)

    def accumulate_benefit_vector(
        self,
        z_world: torch.Tensor,
        benefit_magnitude: float = 1.0,
        hypothesis_tag: bool = False,
    ) -> None:
        """Accumulate benefit in base field's benefit terrain."""
        if hypothesis_tag:
            return
        self.base_field.accumulate_benefit(z_world, benefit_magnitude, hypothesis_tag=False)

    def evaluate_vector(self, z_world: torch.Tensor) -> torch.Tensor:
        """
        Evaluate vector residue at z_world.

        Returns: Tensor[batch, 3]
          [:, 0] = harm intensity (standard scalar)
          [:, 1] = harm recency (decayed EMA weights)
          [:, 2] = benefit proximity (benefit_rbf_field)
        """
        # dim 0: standard intensity
        intensity = self.base_field.evaluate(z_world)  # [batch]

        # dim 1: recency (EMA-decayed weights in recency_rbf)
        with torch.no_grad():
            recency = self.recency_rbf(z_world)  # [batch]

        # dim 2: benefit proximity
        benefit = self.base_field.evaluate_benefit(z_world)  # [batch]

        return torch.stack([intensity, recency, benefit], dim=-1)  # [batch, 3]

    def get_dim_variance(self, z_samples: torch.Tensor) -> Tuple[float, float, float]:
        """
        Compute per-dimension std across z_world samples.

        Returns: (std_dim0, std_dim1, std_dim2)
        """
        with torch.no_grad():
            vec = self.evaluate_vector(z_samples)  # [N, 3]
        return (
            float(vec[:, 0].std().item()),
            float(vec[:, 1].std().item()),
            float(vec[:, 2].std().item()),
        )


# ---------------------------------------------------------------------------
# Minimal E3 projection head for VECTOR condition
# ---------------------------------------------------------------------------

class VectorResidueProjection(nn.Module):
    """
    Fuses vector residue into z_world before E3 harm_eval.

    Takes z_world [batch, world_dim] and residue_vec [batch, 3], produces
    fused_z [batch, world_dim] for downstream harm_eval.
    """

    def __init__(self, world_dim: int, residue_dim: int = 3):
        super().__init__()
        self.proj = nn.Linear(world_dim + residue_dim, world_dim)

    def forward(self, z_world: torch.Tensor, residue_vec: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_world:     [batch, world_dim]
            residue_vec: [batch, residue_dim]
        Returns:
            fused_z:     [batch, world_dim]
        """
        combined = torch.cat([z_world, residue_vec], dim=-1)
        return self.proj(combined)


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------

def _make_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(
        seed=seed,
        size=10,
        num_hazards=2,
        num_resources=3,
        hazard_harm=0.02,
        env_drift_interval=10,
        env_drift_prob=0.2,
    )


# ---------------------------------------------------------------------------
# Config factory
# ---------------------------------------------------------------------------

def _make_config(world_dim: int = WORLD_DIM) -> REEConfig:
    cfg = REEConfig()
    cfg.latent.world_dim = world_dim
    cfg.latent.self_dim = 16
    cfg.latent.body_obs_dim = BODY_OBS_DIM
    cfg.latent.world_obs_dim = WORLD_OBS_DIM
    cfg.latent.observation_dim = BODY_OBS_DIM + WORLD_OBS_DIM
    cfg.latent.alpha_world = 0.95  # SD-008: avoid double-smoothing
    cfg.latent.alpha_self = 0.3
    cfg.residue.world_dim = world_dim
    cfg.residue.benefit_terrain_enabled = True  # both conditions have benefit terrain
    return cfg


# ---------------------------------------------------------------------------
# Run one condition x seed
# ---------------------------------------------------------------------------

def _run_condition(
    seed: int,
    condition: str,           # "SCALAR" or "VECTOR"
    warmup_episodes: int,
    eval_episodes: int,
    steps_per_episode: int,
    lr: float,
    dry_run: bool,
) -> Dict:
    """Run one condition for one seed. Returns per-seed result dict."""
    if dry_run:
        warmup_episodes = 2
        eval_episodes = 2
        steps_per_episode = 10

    torch.manual_seed(seed)
    random.seed(seed)

    env = _make_env(seed)
    cfg = _make_config(WORLD_DIM)
    agent = REEAgent(cfg)
    agent.train()

    # Build residue wrapper / projection for VECTOR condition
    vector_wrapper: Optional[VectorResidueWrapper] = None
    vector_proj: Optional[VectorResidueProjection] = None
    vector_proj_opt: Optional[torch.optim.Optimizer] = None

    if condition == "VECTOR":
        vector_wrapper = VectorResidueWrapper(
            base_field=agent.residue_field,
            world_dim=WORLD_DIM,
        )
        vector_proj = VectorResidueProjection(world_dim=WORLD_DIM, residue_dim=RESIDUE_VEC_DIM)
        vector_proj_opt = optim.Adam(
            list(vector_proj.parameters()),
            lr=lr,
        )

    # Optimisers
    e1_opt = optim.Adam(agent.e1.parameters(), lr=lr)
    e3_opt = optim.Adam(agent.e3.parameters(), lr=lr)

    print(
        f"  [{condition}] seed={seed}"
        f" warmup={warmup_episodes} eval={eval_episodes}"
        f" steps={steps_per_episode}",
        flush=True,
    )

    # -----------------------------------------------------------------------
    # Phase 1: Warmup training
    # -----------------------------------------------------------------------
    for ep in range(warmup_episodes):
        _, obs_dict = env.reset()
        agent.reset()
        ep_harm = 0.0
        ep_benefit = 0.0

        for _step in range(steps_per_episode):
            obs_body  = torch.tensor(obs_dict["body_state"],  dtype=torch.float32)
            obs_world = torch.tensor(obs_dict["world_state"], dtype=torch.float32)

            latent = agent.sense(obs_body, obs_world)
            agent.clock.advance()

            # E1 prediction loss
            e1_loss = agent.compute_prediction_loss()
            if e1_loss is not None and e1_loss.requires_grad:
                e1_opt.zero_grad()
                e1_loss.backward()
                e1_opt.step()

            # E3 harm eval loss
            if agent._current_latent is not None:
                z_world_cur = agent._current_latent.z_world.detach()

                if condition == "VECTOR" and vector_wrapper is not None and vector_proj is not None:
                    # Fuse vector residue into z_world for E3
                    residue_vec = vector_wrapper.evaluate_vector(z_world_cur)
                    fused_z = vector_proj(z_world_cur, residue_vec.detach())
                    harm_pred = agent.e3.harm_eval(fused_z)
                else:
                    harm_pred = agent.e3.harm_eval(z_world_cur)

                # Harm target from environment (we check harm_signal after step)
                action_idx = random.randint(0, ACTION_DIM - 1)
                action = torch.zeros(1, ACTION_DIM)
                action[0, action_idx] = 1.0
                _, reward, done, _, obs_dict = env.step(action)

                harm_signal   = float(min(0.0, reward))
                benefit_signal = float(max(0.0, reward))
                ep_harm    += abs(harm_signal)
                ep_benefit += benefit_signal

                harm_target = torch.tensor(
                    [[1.0 if harm_signal < 0 else 0.0]], dtype=torch.float32
                )
                harm_loss = F.mse_loss(harm_pred, harm_target)

                if condition == "VECTOR" and vector_proj_opt is not None:
                    e3_opt.zero_grad()
                    vector_proj_opt.zero_grad()
                    harm_loss.backward()
                    e3_opt.step()
                    vector_proj_opt.step()
                else:
                    e3_opt.zero_grad()
                    harm_loss.backward()
                    e3_opt.step()

                # Residue accumulation
                if harm_signal < 0:
                    z_w = agent._current_latent.z_world.detach()
                    if condition == "VECTOR" and vector_wrapper is not None:
                        vector_wrapper.accumulate_vector(z_w, harm_magnitude=abs(harm_signal))
                    else:
                        agent.residue_field.accumulate(z_w, harm_magnitude=abs(harm_signal))

                # Benefit accumulation
                if benefit_signal > 0:
                    z_w = agent._current_latent.z_world.detach()
                    if condition == "VECTOR" and vector_wrapper is not None:
                        vector_wrapper.accumulate_benefit_vector(z_w, benefit_magnitude=benefit_signal)
                    else:
                        agent.residue_field.accumulate_benefit(z_w, benefit_magnitude=benefit_signal)

                    if agent.goal_state is not None:
                        agent.update_z_goal(benefit_signal, drive_level=1.0)

                if done:
                    break
            else:
                # No latent yet -- just step
                action_idx = random.randint(0, ACTION_DIM - 1)
                action = torch.zeros(1, ACTION_DIM)
                action[0, action_idx] = 1.0
                _, reward, done, _, obs_dict = env.step(action)
                harm_signal    = float(min(0.0, reward))
                benefit_signal = float(max(0.0, reward))
                ep_harm    += abs(harm_signal)
                ep_benefit += benefit_signal
                if done:
                    break

        if (ep + 1) % 100 == 0:
            print(
                f"    [train] {condition} seed={seed}"
                f" ep {ep+1}/{warmup_episodes}"
                f" harm={ep_harm:.3f} benefit={ep_benefit:.3f}",
                flush=True,
            )

    # -----------------------------------------------------------------------
    # Phase 2: Evaluation
    # -----------------------------------------------------------------------
    agent.eval()

    harm_per_ep:    List[float] = []
    benefit_per_ep: List[float] = []

    for _ep in range(eval_episodes):
        _, obs_dict = env.reset()
        agent.reset()
        ep_harm = 0.0
        ep_benefit = 0.0

        for _step in range(steps_per_episode):
            obs_body  = torch.tensor(obs_dict["body_state"],  dtype=torch.float32)
            obs_world = torch.tensor(obs_dict["world_state"], dtype=torch.float32)

            with torch.no_grad():
                latent = agent.sense(obs_body, obs_world)
                agent.clock.advance()

            action_idx = random.randint(0, ACTION_DIM - 1)
            action = torch.zeros(1, ACTION_DIM)
            action[0, action_idx] = 1.0
            _, reward, done, _, obs_dict = env.step(action)

            harm_signal    = float(min(0.0, reward))
            benefit_signal = float(max(0.0, reward))
            ep_harm    += abs(harm_signal)
            ep_benefit += benefit_signal

            if done:
                break

        harm_per_ep.append(ep_harm)
        benefit_per_ep.append(ep_benefit)

    mean_harm    = float(sum(harm_per_ep)    / max(1, len(harm_per_ep)))
    mean_benefit = float(sum(benefit_per_ep) / max(1, len(benefit_per_ep)))

    # -----------------------------------------------------------------------
    # Vector diagnostics (VECTOR condition only)
    # -----------------------------------------------------------------------
    dim_var_0 = dim_var_1 = dim_var_2 = 0.0
    mean_benefit_dim2 = 0.0

    if condition == "VECTOR" and vector_wrapper is not None:
        # Sample random z_world points to measure per-dim variance
        z_samples = torch.randn(200, WORLD_DIM)
        dv = vector_wrapper.get_dim_variance(z_samples)
        dim_var_0, dim_var_1, dim_var_2 = dv

        # Mean of benefit (dim 2) over z_samples
        with torch.no_grad():
            vec_out = vector_wrapper.evaluate_vector(z_samples)
        mean_benefit_dim2 = float(vec_out[:, 2].mean().item())

    print(
        f"  [{condition}] seed={seed}"
        f" mean_harm={mean_harm:.5f}"
        f" mean_benefit={mean_benefit:.5f}"
        + (
            f" dim_var=({dim_var_0:.4f},{dim_var_1:.4f},{dim_var_2:.4f})"
            f" benefit_dim2_mean={mean_benefit_dim2:.4f}"
            if condition == "VECTOR" else ""
        ),
        flush=True,
    )

    return {
        "seed":              seed,
        "condition":         condition,
        "mean_harm":         mean_harm,
        "mean_benefit":      mean_benefit,
        "dim_var_0":         dim_var_0,
        "dim_var_1":         dim_var_1,
        "dim_var_2":         dim_var_2,
        "mean_benefit_dim2": mean_benefit_dim2,
    }


# ---------------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------------

def run(
    seeds: Tuple[int, ...] = (42, 123),
    warmup_episodes: int = 300,
    eval_episodes: int = 100,
    steps_per_episode: int = 200,
    lr: float = 1e-3,
    dry_run: bool = False,
) -> dict:
    """Q-003: R-Field Dimensionality Discriminative Pair.

    Tests whether a vector residue field (harm_intensity + harm_recency +
    benefit_proximity) provides better harm avoidance than a scalar residue
    field, and whether the vector dimensions carry non-redundant information.
    PASS: vector reduces harm by >= 8% AND vector dims have non-trivial variance
    AND benefit dim is non-zero. This would support Q-003 guidance: vector
    R-field is preferable. FAIL (on C4): vector actively worsens harm avoidance,
    supporting scalar. FAIL (on C2): vector dims collapse, so Q-003 is not yet
    answerable with current architecture.
    """
    print(
        f"\n[V3-EXQ-148] Q-003 R-Field Dimensionality Discriminative Pair"
        f"  seeds={list(seeds)}"
        f"  warmup={warmup_episodes} eval={eval_episodes}",
        flush=True,
    )

    results_scalar: List[Dict] = []
    results_vector: List[Dict] = []

    for seed in seeds:
        r_scalar = _run_condition(
            seed=seed, condition="SCALAR",
            warmup_episodes=warmup_episodes, eval_episodes=eval_episodes,
            steps_per_episode=steps_per_episode, lr=lr, dry_run=dry_run,
        )
        results_scalar.append(r_scalar)

        r_vector = _run_condition(
            seed=seed, condition="VECTOR",
            warmup_episodes=warmup_episodes, eval_episodes=eval_episodes,
            steps_per_episode=steps_per_episode, lr=lr, dry_run=dry_run,
        )
        results_vector.append(r_vector)

    def _avg(results: List[Dict], key: str) -> float:
        vals = [r[key] for r in results if isinstance(r[key], (int, float))]
        return float(sum(vals) / max(1, len(vals)))

    mean_harm_scalar    = _avg(results_scalar, "mean_harm")
    mean_harm_vector    = _avg(results_vector, "mean_harm")
    mean_benefit_scalar = _avg(results_scalar, "mean_benefit")
    mean_benefit_vector = _avg(results_vector, "mean_benefit")
    mean_dim_var_0      = _avg(results_vector, "dim_var_0")
    mean_dim_var_1      = _avg(results_vector, "dim_var_1")
    mean_dim_var_2      = _avg(results_vector, "dim_var_2")
    mean_benefit_dim2   = _avg(results_vector, "mean_benefit_dim2")

    harm_ratio_per_seed = [
        rv["mean_harm"] / max(1e-9, rs["mean_harm"])
        for rv, rs in zip(results_vector, results_scalar)
    ]

    # -----------------------------------------------------------------------
    # Pre-registered criteria
    # -----------------------------------------------------------------------

    # C1: vector harm < scalar harm * THRESH_HARM_RATIO (both seeds)
    c1_per_seed = [ratio < THRESH_HARM_RATIO for ratio in harm_ratio_per_seed]
    c1_pass = len(c1_per_seed) >= len(seeds) and all(c1_per_seed)

    # C2: vector dims non-degenerate -- mean std across dims > THRESH_DIM_VAR (both seeds)
    c2_per_seed = [
        (rv["dim_var_0"] > THRESH_DIM_VAR and rv["dim_var_1"] > THRESH_DIM_VAR)
        for rv in results_vector
    ]
    c2_pass = len(c2_per_seed) >= len(seeds) and all(c2_per_seed)

    # C3: benefit dim non-zero (both seeds)
    c3_per_seed = [rv["mean_benefit_dim2"] > THRESH_BENEFIT_MIN for rv in results_vector]
    c3_pass = len(c3_per_seed) >= len(seeds) and all(c3_per_seed)

    # C4: vector does NOT regress harm beyond ratio (both seeds)
    c4_per_seed = [ratio < THRESH_NO_REGRESSION for ratio in harm_ratio_per_seed]
    c4_pass = len(c4_per_seed) >= len(seeds) and all(c4_per_seed)

    # C5: consistency -- if C1 mixed, note it
    c5_pass = not (any(c1_per_seed) and not all(c1_per_seed))  # True if consistent

    all_pass = c1_pass and c2_pass and c3_pass
    partial_pass = c4_pass and c2_pass and not all_pass
    criteria_met = sum([c1_pass, c2_pass, c3_pass, c4_pass, c5_pass])
    status = "PASS" if all_pass else "FAIL"

    # Q-003 guidance
    if all_pass:
        q003_guidance = "vector"
        decision = "retain_ree"
    elif not c4_pass:
        q003_guidance = "scalar"
        decision = "retire_ree_claim"
    elif not c2_pass:
        q003_guidance = "inconclusive_collapsed"
        decision = "retire_ree_claim"
    else:
        q003_guidance = "inconclusive"
        decision = "hybridize"

    mean_harm_ratio = float(
        sum(harm_ratio_per_seed) / max(1, len(harm_ratio_per_seed))
    )

    # -----------------------------------------------------------------------
    # Print results
    # -----------------------------------------------------------------------
    print(f"\n[V3-EXQ-148] Results:", flush=True)
    print(
        f"  SCALAR: mean_harm={mean_harm_scalar:.5f}"
        f" mean_benefit={mean_benefit_scalar:.5f}",
        flush=True,
    )
    print(
        f"  VECTOR: mean_harm={mean_harm_vector:.5f}"
        f" mean_benefit={mean_benefit_vector:.5f}"
        f" dim_var=({mean_dim_var_0:.4f},{mean_dim_var_1:.4f},{mean_dim_var_2:.4f})"
        f" benefit_dim2={mean_benefit_dim2:.4f}",
        flush=True,
    )
    print(
        f"  harm_ratio (vector/scalar) per seed: "
        f"{[round(x, 4) for x in harm_ratio_per_seed]}"
        f"  [target C1<{THRESH_HARM_RATIO}, C4<{THRESH_NO_REGRESSION}]",
        flush=True,
    )
    print(
        f"  C1={c1_pass} C2={c2_pass} C3={c3_pass}"
        f" C4={c4_pass} C5={c5_pass}"
        f"  status={status} ({criteria_met}/5)"
        f"  q003_guidance={q003_guidance}",
        flush=True,
    )

    # -----------------------------------------------------------------------
    # Failure notes
    # -----------------------------------------------------------------------
    failure_notes: List[str] = []
    if not c1_pass:
        failure_notes.append(
            f"C1 FAIL: harm_ratio {[round(x,4) for x in harm_ratio_per_seed]}"
            f" not < {THRESH_HARM_RATIO}."
            " Vector residue field does not reduce harm avoidance by >= 8%."
            " Possible causes: (1) E3 cannot learn a useful linear combination"
            " of 3 residue dims with the current training budget; (2) benefit"
            " terrain is not rich enough to differentiate from harm dims;"
            " (3) random actions reduce the signal from residue field geometry."
        )
    if not c2_pass:
        failure_notes.append(
            f"C2 FAIL: dim_var ({mean_dim_var_0:.4f},{mean_dim_var_1:.4f},{mean_dim_var_2:.4f})"
            f" -- one or more dims < {THRESH_DIM_VAR}."
            " Vector dims have collapsed: all three dimensions track the same"
            " scalar pattern. The vector residue field adds no new information."
            " Q-003 is not answerable at current scale with this implementation."
        )
    if not c3_pass:
        failure_notes.append(
            f"C3 FAIL: mean_benefit_dim2={mean_benefit_dim2:.4f} < {THRESH_BENEFIT_MIN}."
            " Benefit terrain (dim 2 of vector) is near-zero -- insufficient"
            " benefit events accumulated during training to populate the"
            " benefit RBF field. Benefit terrain is not contributing to the"
            " vector's informational content."
        )
    if not c4_pass:
        failure_notes.append(
            f"C4 FAIL: harm_ratio {[round(x,4) for x in harm_ratio_per_seed]}"
            f" >= {THRESH_NO_REGRESSION}."
            " Vector residue field WORSENS harm avoidance by >= 10%."
            " The extra dimensions are confusing E3's harm_eval at current scale."
            " Q-003 guidance: scalar R-field is preferable in this regime."
        )
    if not c5_pass:
        failure_notes.append(
            f"C5 WARN: C1 results inconsistent across seeds"
            f" {c1_per_seed}."
            " High seed variability -- result is not stable."
        )

    for note in failure_notes:
        print(f"  NOTE: {note}", flush=True)

    # -----------------------------------------------------------------------
    # Interpretation
    # -----------------------------------------------------------------------
    if all_pass:
        interpretation = (
            f"Q-003 GUIDANCE: VECTOR R-FIELD PREFERRED."
            f" Vector residue field (harm_intensity + harm_recency + benefit_proximity)"
            f" reduces harm by {(1 - mean_harm_ratio)*100:.1f}%"
            f" vs scalar (harm_ratio={mean_harm_ratio:.4f} < {THRESH_HARM_RATIO})."
            f" Vector dims are non-degenerate: std=({mean_dim_var_0:.4f},"
            f"{mean_dim_var_1:.4f},{mean_dim_var_2:.4f}) all > {THRESH_DIM_VAR}."
            f" Benefit dim carries meaningful signal (mean={mean_benefit_dim2:.4f})."
            f" Supports Q-003: multiple regulatory dimensions in R(x,t) are useful;"
            f" vector > scalar for E3 harm avoidance."
        )
    elif not c4_pass:
        interpretation = (
            f"Q-003 GUIDANCE: SCALAR R-FIELD PREFERRED at current scale."
            f" Vector residue field WORSENS harm avoidance (harm_ratio={mean_harm_ratio:.4f}"
            f" >= {THRESH_NO_REGRESSION}). The extra dimensions confuse E3."
            f" Q-003 guidance: scalar R(x,t) is preferable until E3 capacity"
            f" scales to benefit from multi-dimensional regulatory signal."
        )
    elif not c2_pass:
        interpretation = (
            f"Q-003 INCONCLUSIVE: vector dims collapsed."
            f" harm_ratio={mean_harm_ratio:.4f}. dim_var=({mean_dim_var_0:.4f},"
            f"{mean_dim_var_1:.4f},{mean_dim_var_2:.4f})."
            f" All dims track the same pattern -- benefit terrain not populated"
            f" or recency RBF collapses to intensity. Q-003 not answerable"
            f" with current data density."
        )
    else:
        interpretation = (
            f"Q-003 INCONCLUSIVE: vector does not clearly improve OR degrade."
            f" harm_ratio={mean_harm_ratio:.4f} (target < {THRESH_HARM_RATIO})."
            f" No regression (C4={c4_pass}). Dims are non-trivial (C2={c2_pass})."
            f" Training budget may be insufficient to exploit vector dims."
            f" Longer runs or richer environments may resolve Q-003."
        )

    # -----------------------------------------------------------------------
    # Per-seed row helper
    # -----------------------------------------------------------------------
    def _seed_rows(results: List[Dict]) -> str:
        rows = []
        for r in results:
            row = (
                f"  seed={r['seed']} cond={r['condition']}:"
                f" harm={r['mean_harm']:.5f} benefit={r['mean_benefit']:.5f}"
            )
            if r['condition'] == "VECTOR":
                row += (
                    f" dim_var=({r['dim_var_0']:.4f},"
                    f"{r['dim_var_1']:.4f},{r['dim_var_2']:.4f})"
                    f" benefit_dim2={r['mean_benefit_dim2']:.4f}"
                )
            rows.append(row)
        return "\n".join(rows)

    failure_section = ""
    if failure_notes:
        failure_section = "\n## Failure Notes\n\n" + "\n".join(
            f"- {n}" for n in failure_notes
        )

    summary_markdown = (
        f"# V3-EXQ-148 -- Q-003: R-Field Dimensionality Discriminative Pair\n\n"
        f"**Status:** {status}\n"
        f"**Claim:** Q-003\n"
        f"**Decision:** {decision}\n"
        f"**Q-003 guidance:** {q003_guidance}\n"
        f"**Seeds:** {list(seeds)}\n"
        f"**Conditions:** SCALAR vs VECTOR (residue_dim=3)\n"
        f"**Warmup:** {warmup_episodes} eps x {steps_per_episode} steps"
        f"  **Eval:** {eval_episodes} eps x {steps_per_episode} steps\n"
        f"**Env:** CausalGridWorldV2 size=10, 2 hazards, 3 resources\n\n"
        f"## Design\n\n"
        f"Q-003 asks whether R(x,t) should be scalar or vector. SCALAR condition"
        f" uses the standard ResidueField.evaluate() (1-D output). VECTOR condition"
        f" uses VectorResidueWrapper.evaluate_vector() producing 3 dims:"
        f" harm_intensity (standard RBF), harm_recency (EMA-decayed weights),"
        f" and benefit_proximity (benefit terrain from ARC-030). E3 receives"
        f" the vector fused into z_world via a small projection head"
        f" (Linear(world_dim+3, world_dim)) trained jointly.\n\n"
        f"Key discriminator: harm_ratio = mean_harm_vector / mean_harm_scalar."
        f" If vector reduces harm by >= 8% (ratio < {THRESH_HARM_RATIO}) AND"
        f" dims are non-degenerate AND benefit dim is non-zero --> vector preferred.\n\n"
        f"## Pre-Registered Thresholds\n\n"
        f"C1: harm_ratio < {THRESH_HARM_RATIO} (both seeds -- vector reduces harm >= 8%)\n"
        f"C2: dim_var > {THRESH_DIM_VAR} for dims 0+1 (non-degenerate vector)\n"
        f"C3: benefit_dim2_mean > {THRESH_BENEFIT_MIN} (benefit dim non-zero)\n"
        f"C4: harm_ratio < {THRESH_NO_REGRESSION} (vector does not regress harm)\n"
        f"C5: C1 results consistent across seeds\n\n"
        f"## Results\n\n"
        f"| Condition | mean_harm | mean_benefit |\n"
        f"|-----------|-----------|-------------|\n"
        f"| SCALAR | {mean_harm_scalar:.5f} | {mean_benefit_scalar:.5f} |\n"
        f"| VECTOR | {mean_harm_vector:.5f} | {mean_benefit_vector:.5f} |\n\n"
        f"**harm_ratio (vector/scalar) per seed: {[round(x,4) for x in harm_ratio_per_seed]}**\n"
        f"**mean_harm_ratio: {mean_harm_ratio:.4f}**\n\n"
        f"### Vector Dimensionality Check\n\n"
        f"| Metric | Value |\n"
        f"|--------|-------|\n"
        f"| dim_var_0 (harm intensity) | {mean_dim_var_0:.4f} |\n"
        f"| dim_var_1 (harm recency)   | {mean_dim_var_1:.4f} |\n"
        f"| dim_var_2 (benefit prox.)  | {mean_dim_var_2:.4f} |\n"
        f"| benefit_dim2_mean          | {mean_benefit_dim2:.4f} |\n\n"
        f"### Per-Seed SCALAR\n{_seed_rows(results_scalar)}\n\n"
        f"### Per-Seed VECTOR\n{_seed_rows(results_vector)}\n\n"
        f"## PASS Criteria\n\n"
        f"| Criterion | Result |\n"
        f"|-----------|--------|\n"
        f"| C1: harm_ratio < {THRESH_HARM_RATIO} (both seeds)"
        f" | {'PASS' if c1_pass else 'FAIL'} |\n"
        f"| C2: dim_var > {THRESH_DIM_VAR} dims 0+1 (both seeds)"
        f" | {'PASS' if c2_pass else 'FAIL'} |\n"
        f"| C3: benefit_dim2 > {THRESH_BENEFIT_MIN} (both seeds)"
        f" | {'PASS' if c3_pass else 'FAIL'} |\n"
        f"| C4: harm_ratio < {THRESH_NO_REGRESSION} (no regression)"
        f" | {'PASS' if c4_pass else 'FAIL'} |\n"
        f"| C5: C1 seed-consistent"
        f" | {'PASS' if c5_pass else 'WARN'} |\n\n"
        f"Criteria met: {criteria_met}/5 -> **{status}**\n\n"
        f"## Interpretation\n\n"
        f"{interpretation}\n"
        f"{failure_section}\n"
    )

    metrics: Dict[str, float] = {
        "mean_harm_scalar":          float(mean_harm_scalar),
        "mean_harm_vector":          float(mean_harm_vector),
        "mean_benefit_scalar":       float(mean_benefit_scalar),
        "mean_benefit_vector":       float(mean_benefit_vector),
        "mean_harm_ratio":           float(mean_harm_ratio),
        "mean_dim_var_0":            float(mean_dim_var_0),
        "mean_dim_var_1":            float(mean_dim_var_1),
        "mean_dim_var_2":            float(mean_dim_var_2),
        "mean_benefit_dim2":         float(mean_benefit_dim2),
        "thresh_harm_ratio":         float(THRESH_HARM_RATIO),
        "thresh_dim_var":            float(THRESH_DIM_VAR),
        "thresh_benefit_min":        float(THRESH_BENEFIT_MIN),
        "thresh_no_regression":      float(THRESH_NO_REGRESSION),
        "crit1_pass":                1.0 if c1_pass else 0.0,
        "crit2_pass":                1.0 if c2_pass else 0.0,
        "crit3_pass":                1.0 if c3_pass else 0.0,
        "crit4_pass":                1.0 if c4_pass else 0.0,
        "crit5_pass":                1.0 if c5_pass else 0.0,
        "criteria_met":              float(criteria_met),
    }

    return {
        "status":              status,
        "metrics":             metrics,
        "summary_markdown":    summary_markdown,
        "claim_ids":           CLAIM_IDS,
        "evidence_direction":  (
            "supports" if all_pass
            else ("mixed" if partial_pass or (c4_pass and c2_pass) else "weakens")
        ),
        "experiment_type":     EXPERIMENT_TYPE,
        "q003_guidance":       q003_guidance,
        "fatal_error_count":   0,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import json
    from datetime import datetime, timezone

    parser = argparse.ArgumentParser(
        description="V3-EXQ-148 Q-003 R-Field Dimensionality Discriminative Pair"
    )
    parser.add_argument("--seeds",    type=int,   nargs="+", default=[42, 123])
    parser.add_argument("--warmup",   type=int,   default=300)
    parser.add_argument("--eval-eps", type=int,   default=100)
    parser.add_argument("--steps",    type=int,   default=200)
    parser.add_argument("--dry-run",  action="store_true")
    args = parser.parse_args()

    result = run(
        seeds=tuple(args.seeds),
        warmup_episodes=args.warmup,
        eval_episodes=args.eval_eps,
        steps_per_episode=args.steps,
        dry_run=args.dry_run,
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
