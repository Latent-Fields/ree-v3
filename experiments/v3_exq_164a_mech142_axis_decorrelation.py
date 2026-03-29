#!/opt/local/bin/python3
"""
V3-EXQ-164a -- MECH-142: Valence-arousal axis orthogonality requires active
               cholinergic-analog decorrelation maintenance during learning.

Supersedes: EXQ-164 (bug: obs_dict keys "harm_signal" and "resource_count" do not exist
in CausalGridWorldV2; both returned 0.0 every step, giving all-zero axes and metrics).
Fix: read harm_prev from harm_obs[12] (center hazard field cell) and benefit_prev from
harm_obs[37] (center resource field cell).

Claim:    MECH-142 (primary), MECH-063 (secondary)
Proposal: EXP-0117 (EVB-0087)

MECH-142 asserts:
  Valence-arousal axis orthogonality in the control plane is NOT a static
  geometric property -- it requires active cholinergic-analog maintenance
  during learning. Without explicit decorrelation, axes drift toward
  correlation under repeated co-activation patterns.

  This is complementary to EXQ-157 (Q-017), which tests whether three
  orthogonal axes produce better regime separation than two or one. EXQ-164
  uses the FULL_AXES design from EXQ-157 and tests whether an explicit
  decorrelation loss term is necessary to maintain orthogonality across
  training, and whether that maintenance preserves or improves regime
  separability.

Design:
  Two conditions x 2 seeds = 4 cells.

  Both conditions use FULL_AXES (3 control axes):
    z_tonic  = EMA(harm_signal, tau=TONIC_TAU)   -- slow ambient baseline
    z_phasic = |harm_signal - z_tonic|            -- fast interrupt
    z_affect = harm_signal - benefit_signal       -- signed valence PE
    control_vec = concat(z_tonic, z_phasic, z_affect)

  Condition A: WITH_DECORRELATION
    Adds decorrelation loss term at each training step:
      L_decorr = DECORR_WEIGHT * (
          |pearson_r(buf_tonic, buf_phasic)| +
          |pearson_r(buf_tonic, buf_affect)| +
          |pearson_r(buf_phasic, buf_affect)|
      )
    where buf_* is a rolling buffer of the last CORR_BUFFER_SIZE axis values.
    Total loss = harm_eval_loss + L_decorr.

  Condition B: NO_DECORRELATION
    Standard FULL_AXES with no decorrelation term. Identical to EXQ-157
    FULL_AXES condition.

Pre-registered thresholds
--------------------------
C1: WITH_DECORRELATION final_corr < NO_DECORRELATION final_corr - CORR_GAP_MIN
    (both seeds) -- decorrelation loss successfully maintains lower correlation.
C2: NO_DECORRELATION final_corr > NO_DECORRELATION initial_corr + DRIFT_MIN
    (both seeds) -- axes drift toward correlation without maintenance.
C3: WITH_DECORRELATION regime_sep >= NO_DECORRELATION regime_sep + REGIME_MARGIN
    (both seeds) -- decorrelation maintains or improves regime separability.
C4: WITH_DECORRELATION last_q_harm <= NO_DECORRELATION last_q_harm + THRESH_HARM_MARGIN
    (both seeds) -- decorrelation loss does not hurt policy quality.

PASS:
  C1 + C2 + C3 + C4 => MECH-142 supported: active decorrelation maintains
  orthogonality and regime separability.
PARTIAL_DRIFT_CONFIRMED:
  C1 + C2 + C4 but NOT C3 => decorrelation works but does not improve regime
  separation at this environment scale.
PARTIAL_NO_DRIFT:
  C1 + C3 + C4 but NOT C2 => decorrelation works but baseline doesn't drift =>
  environment may not co-activate axes enough to produce drift.
FAIL:
  C1 fails => decorrelation loss is ineffective (implementation problem or
  environment geometry doesn't produce axis correlation in the first place).

Checkpoints: [100, 200, 400, 800, 1200, 1600] episodes.
Axis correlation and regime separation measured over last 200 episodes at each
checkpoint window. initial_corr = checkpoint 100. final_corr = checkpoint 1600.
"""

import sys
import math
import random
import json
import uuid
import argparse
from collections import deque
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Deque

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from ree_core.environment.causal_grid_world import CausalGridWorldV2

# ---------------------------------------------------------------------------
# Experiment identity
# ---------------------------------------------------------------------------
EXPERIMENT_TYPE = "v3_exq_164a_mech142_axis_decorrelation"
CLAIM_IDS       = ["MECH-142", "MECH-063"]
RUN_ID_PREFIX   = "v3_exq_164a_mech142_axis_decorrelation"

# ---------------------------------------------------------------------------
# Protocol constants
# ---------------------------------------------------------------------------
CONDITIONS        = ["WITH_DECORRELATION", "NO_DECORRELATION"]
SEEDS             = [42, 123]
N_EPISODES        = 1600
N_EVAL_EPISODES   = 200          # last N_EVAL_EPISODES for final metrics
STEPS_PER_EPISODE = 200
CHECKPOINTS       = [100, 200, 400, 800, 1200, 1600]
CORR_WINDOW       = 200          # episodes to look back at each checkpoint

# Control-axis parameters
TONIC_TAU         = 50.0         # EMA decay constant (steps)
CORR_BUFFER_SIZE  = 100          # rolling buffer size for correlation computation
DECORR_WEIGHT     = 0.1          # weight on decorrelation loss

# Regime detection
REGIME_HARM_THRESH = 0.03        # mean harm/step: above -> AVOIDANCE
WINDOW_SIZE        = 20          # steps per regime classification window

# Model dimensions
GRID_SIZE    = 8
N_HAZARDS    = 4
N_RESOURCES  = 2
HAZARD_HARM  = 0.05
WORLD_OBS_DIM = 250
SELF_OBS_DIM  = 12
ACTION_DIM    = 5
WORLD_DIM     = 32
SELF_DIM      = 16
HARM_DIM      = 8
HIDDEN_DIM    = 32
LR            = 1e-3
ENT_BONUS     = 5e-3
COMMIT_STEP_FRAC = 0.5
NAV_BIAS      = 0.6

# Pre-registered thresholds
CORR_GAP_MIN       = 0.10   # C1: WITH lower than NO by at least this
DRIFT_MIN          = 0.10   # C2: NO_DECORRELATION final > initial by at least this
REGIME_MARGIN      = -0.05  # C3: WITH regime_sep >= NO + this (allows slight degradation)
THRESH_HARM_MARGIN = 0.03   # C4: WITH harm <= NO + this


# ---------------------------------------------------------------------------
# Models (inline, no ree_core imports beyond environment)
# ---------------------------------------------------------------------------

class WorldEncoder(nn.Module):
    def __init__(self, obs_dim: int, world_dim: int):
        super().__init__()
        self.linear = nn.Linear(obs_dim, world_dim)
        self.norm   = nn.LayerNorm(world_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.linear(x))


class SelfEncoder(nn.Module):
    def __init__(self, obs_dim: int, self_dim: int):
        super().__init__()
        self.linear = nn.Linear(obs_dim, self_dim)
        self.norm   = nn.LayerNorm(self_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.linear(x))


class ControlPolicy(nn.Module):
    """Policy: (z_world, z_self, control_vec) -> action logits."""
    def __init__(
        self,
        world_dim: int,
        self_dim: int,
        control_dim: int,
        action_dim: int,
        hidden_dim: int,
    ):
        super().__init__()
        in_dim = world_dim + self_dim + control_dim
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(
        self,
        z_world: torch.Tensor,
        z_self:  torch.Tensor,
        c:       torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([z_world, z_self, c], dim=-1)
        return self.fc2(F.relu(self.fc1(x)))


class HarmEvaluator(nn.Module):
    """E3 analog: Linear(z_world + z_self, 1) -> harm scalar."""
    def __init__(self, world_dim: int, self_dim: int):
        super().__init__()
        self.fc = nn.Linear(world_dim + self_dim, 1)

    def forward(
        self,
        z_world: torch.Tensor,
        z_self:  torch.Tensor,
    ) -> torch.Tensor:
        return self.fc(torch.cat([z_world, z_self], dim=-1))


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def _get_obs_tensor(obs_dict: dict, key: str, fallback_dim: int) -> torch.Tensor:
    """Safely extract a 1-D float tensor from obs_dict."""
    raw = obs_dict.get(key)
    if raw is None:
        return torch.zeros(fallback_dim)
    t = torch.tensor(raw, dtype=torch.float32).flatten()
    if t.shape[0] < fallback_dim:
        t = F.pad(t, (0, fallback_dim - t.shape[0]))
    elif t.shape[0] > fallback_dim:
        t = t[:fallback_dim]
    return t


def pearson_r(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Analytic Pearson r over 1-D tensors x and y.
    Returns a scalar tensor. Clamps denominator to avoid divide-by-zero.
    """
    xc = x - x.mean()
    yc = y - y.mean()
    denom = (xc.std() * yc.std()).clamp(min=1e-8)
    return (xc * yc).mean() / denom


def _classify_regime(
    harm_vals: List[float],
    resource_vals: List[float],
) -> str:
    """Classify a window into avoidance / exploitation / exploration."""
    mean_harm       = float(sum(harm_vals) / len(harm_vals)) if harm_vals else 0.0
    total_resources = sum(resource_vals)
    if mean_harm > REGIME_HARM_THRESH:
        return "avoidance"
    elif total_resources > 0:
        return "exploitation"
    else:
        return "exploration"


def _regime_separation(
    regime_harm_rates: Dict[str, List[float]],
) -> float:
    """
    Regime separation score: mean |harm_avoidance - harm_exploration|.
    Returns 0.0 if either regime bucket is empty.
    """
    avoid_rates = regime_harm_rates.get("avoidance", [])
    expl_rates  = regime_harm_rates.get("exploration", [])
    if not avoid_rates or not expl_rates:
        return 0.0
    mean_avoid = sum(avoid_rates) / len(avoid_rates)
    mean_expl  = sum(expl_rates)  / len(expl_rates)
    return abs(mean_avoid - mean_expl)


# ---------------------------------------------------------------------------
# Per-window metric snapshot helper
# ---------------------------------------------------------------------------

def _compute_snapshot(
    ep_harm:     List[float],
    ep_resource: List[float],
    buf_tonic:   Deque,
    buf_phasic:  Deque,
    buf_affect:  Deque,
    window_steps: int,
    steps_per_ep: int,
    n_ep_window:  int,
) -> Dict:
    """
    Compute axis correlation and regime separation over the last n_ep_window
    episodes of harm/resource data and the current rolling axis buffers.
    """
    # Use last n_ep_window * steps_per_ep steps from ep_harm / ep_resource
    n_steps = n_ep_window * steps_per_ep
    recent_harm     = ep_harm[-n_steps:]     if len(ep_harm)     >= n_steps else ep_harm[:]
    recent_resource = ep_resource[-n_steps:] if len(ep_resource) >= n_steps else ep_resource[:]

    # Regime separation from windowed regime classification
    regime_harm_rates: Dict[str, List[float]] = {
        "avoidance": [], "exploitation": [], "exploration": []
    }
    for w_start in range(0, len(recent_harm) - window_steps + 1, window_steps):
        w_harm = recent_harm[w_start:w_start + window_steps]
        w_res  = recent_resource[w_start:w_start + window_steps]
        regime = _classify_regime(w_harm, w_res)
        regime_harm_rates[regime].append(
            sum(w_harm) / len(w_harm)
        )

    regime_sep = _regime_separation(regime_harm_rates)

    # Axis correlation from rolling buffers
    if len(buf_tonic) >= 2 and len(buf_phasic) >= 2 and len(buf_affect) >= 2:
        t = torch.tensor(list(buf_tonic),  dtype=torch.float32)
        p = torch.tensor(list(buf_phasic), dtype=torch.float32)
        a = torch.tensor(list(buf_affect), dtype=torch.float32)
        r_tp = float(pearson_r(t, p).abs().item())
        r_ta = float(pearson_r(t, a).abs().item())
        r_pa = float(pearson_r(p, a).abs().item())
        mean_corr = (r_tp + r_ta + r_pa) / 3.0
    else:
        r_tp = r_ta = r_pa = mean_corr = 0.0

    harm_rate = sum(recent_harm) / len(recent_harm) if recent_harm else 0.0

    return {
        "regime_sep":      regime_sep,
        "mean_axis_corr":  mean_corr,
        "r_tonic_phasic":  r_tp,
        "r_tonic_affect":  r_ta,
        "r_phasic_affect": r_pa,
        "harm_rate":       harm_rate,
    }


# ---------------------------------------------------------------------------
# Core: run one condition x seed cell
# ---------------------------------------------------------------------------

def _run_cell(
    seed:         int,
    condition:    str,
    n_episodes:   int,
    dry_run:      bool,
) -> Dict:
    """
    Run one (condition, seed) cell and return a full metrics dict.

    Checkpoint list is a subset of training episodes at which we snapshot:
      - mean_axis_correlation (over rolling buffer)
      - regime_separation_score (over last CORR_WINDOW episodes)
    """
    torch.manual_seed(seed)
    random.seed(seed)

    add_decorr = (condition == "WITH_DECORRELATION")

    env = CausalGridWorldV2(
        size=GRID_SIZE,
        num_hazards=N_HAZARDS,
        num_resources=N_RESOURCES,
        hazard_harm=HAZARD_HARM,
        env_drift_interval=10,
        env_drift_prob=0.3,
        seed=seed,
    )

    world_enc = WorldEncoder(WORLD_OBS_DIM, WORLD_DIM)
    self_enc  = SelfEncoder(SELF_OBS_DIM, SELF_DIM)
    policy    = ControlPolicy(WORLD_DIM, SELF_DIM, 3, ACTION_DIM, HIDDEN_DIM)
    harm_eval = HarmEvaluator(WORLD_DIM, SELF_DIM)

    params = (
        list(world_enc.parameters())
        + list(self_enc.parameters())
        + list(policy.parameters())
        + list(harm_eval.parameters())
    )
    optimizer = optim.Adam(params, lr=LR)

    # EMA tonic state
    z_tonic_val: float = 0.0
    alpha_tonic = 1.0 / TONIC_TAU

    # Rolling axis buffers (deques of floats, length CORR_BUFFER_SIZE)
    buf_tonic:  Deque = deque(maxlen=CORR_BUFFER_SIZE)
    buf_phasic: Deque = deque(maxlen=CORR_BUFFER_SIZE)
    buf_affect: Deque = deque(maxlen=CORR_BUFFER_SIZE)

    # Step-level history (for checkpoint snapshots)
    all_harm:     List[float] = []
    all_resource: List[float] = []

    # Checkpoint results
    checkpoints_data: Dict[int, Dict] = {}

    # Dry-run override
    if dry_run:
        n_episodes = 1
        checkpoints_active = []
    else:
        checkpoints_active = CHECKPOINTS[:]

    _, obs_dict = env.reset()

    for ep in range(n_episodes):
        ep_harm_buf:     List[float] = []
        ep_resource_buf: List[float] = []

        for _step in range(STEPS_PER_EPISODE):
            # Encode observations
            obs_world = _get_obs_tensor(obs_dict, "world_state", WORLD_OBS_DIM)
            obs_self  = _get_obs_tensor(obs_dict, "body_state",  SELF_OBS_DIM)

            z_world = world_enc(obs_world)
            z_self  = self_enc(obs_self)

            # Build control axes
            # harm_obs layout: [0:25] hazard_field_view, [25:50] resource_field_view, [50] harm_exposure
            # Index 12 = center cell of 5x5 hazard field (agent position); 37 = center of resource field
            _harm_obs_t  = obs_dict.get("harm_obs")
            harm_prev    = float(_harm_obs_t[12].item()) if _harm_obs_t is not None else 0.0
            benefit_prev = float(_harm_obs_t[37].item()) if _harm_obs_t is not None else 0.0

            z_tonic_val  = z_tonic_val + alpha_tonic * (harm_prev - z_tonic_val)
            z_phasic_val = abs(harm_prev - z_tonic_val)
            z_affect_val = harm_prev - benefit_prev

            c = torch.tensor(
                [z_tonic_val, z_phasic_val, z_affect_val],
                dtype=torch.float32,
            )

            # Update rolling axis buffers
            buf_tonic.append(z_tonic_val)
            buf_phasic.append(z_phasic_val)
            buf_affect.append(z_affect_val)

            # Policy forward pass
            logits     = policy(z_world.detach(), z_self.detach(), c)
            probs      = F.softmax(logits, dim=-1)
            action_idx = torch.multinomial(probs.detach(), 1).item()
            action     = torch.zeros(ACTION_DIM)
            action[action_idx] = 1.0

            # Step environment
            _, _, done, _, obs_dict_next = env.step(action.unsqueeze(0))

            _harm_obs_n  = obs_dict_next.get("harm_obs")
            harm_signal  = float(_harm_obs_n[12].item()) if _harm_obs_n is not None else 0.0
            resource_cnt = float(_harm_obs_n[37].item()) if _harm_obs_n is not None else 0.0
            harm_actual  = torch.tensor([harm_signal], dtype=torch.float32)

            # Harm evaluator loss
            harm_pred_val  = harm_eval(z_world, z_self)
            harm_eval_loss = F.mse_loss(harm_pred_val, harm_actual)

            # Entropy bonus
            log_probs = F.log_softmax(logits, dim=-1)
            entropy   = -(probs * log_probs).sum()

            # Decorrelation loss (WITH_DECORRELATION condition only)
            l_decorr = torch.tensor(0.0)
            if add_decorr and len(buf_tonic) >= 2:
                t_buf = torch.tensor(list(buf_tonic),  dtype=torch.float32)
                p_buf = torch.tensor(list(buf_phasic), dtype=torch.float32)
                a_buf = torch.tensor(list(buf_affect), dtype=torch.float32)
                l_decorr = DECORR_WEIGHT * (
                    pearson_r(t_buf, p_buf).abs()
                    + pearson_r(t_buf, a_buf).abs()
                    + pearson_r(p_buf, a_buf).abs()
                )

            loss = harm_eval_loss + l_decorr - ENT_BONUS * entropy

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()

            ep_harm_buf.append(harm_signal)
            ep_resource_buf.append(resource_cnt)

            obs_dict = obs_dict_next
            if done:
                _, obs_dict = env.reset()

        # Accumulate step history
        all_harm.extend(ep_harm_buf)
        all_resource.extend(ep_resource_buf)

        # Check checkpoints (1-indexed episodes)
        ep_1indexed = ep + 1
        if ep_1indexed in checkpoints_active:
            snap = _compute_snapshot(
                ep_harm=all_harm,
                ep_resource=all_resource,
                buf_tonic=buf_tonic,
                buf_phasic=buf_phasic,
                buf_affect=buf_affect,
                window_steps=WINDOW_SIZE,
                steps_per_ep=STEPS_PER_EPISODE,
                n_ep_window=CORR_WINDOW,
            )
            checkpoints_data[ep_1indexed] = snap
            print(
                f"[EXQ-164a]   ep={ep_1indexed:4d}  "
                f"mean_corr={snap['mean_axis_corr']:.4f}  "
                f"regime_sep={snap['regime_sep']:.4f}  "
                f"harm={snap['harm_rate']:.4f}  "
                f"r_tp={snap['r_tonic_phasic']:.3f}  "
                f"r_ta={snap['r_tonic_affect']:.3f}  "
                f"r_pa={snap['r_phasic_affect']:.3f}"
            )

    # ---------------------------------------------------------------------------
    # Final metrics (last N_EVAL_EPISODES)
    # ---------------------------------------------------------------------------
    eval_n_steps = N_EVAL_EPISODES * STEPS_PER_EPISODE
    eval_harm     = all_harm[-eval_n_steps:]     if len(all_harm)     >= eval_n_steps else all_harm[:]
    eval_resource = all_resource[-eval_n_steps:] if len(all_resource) >= eval_n_steps else all_resource[:]

    last_q_harm = sum(eval_harm) / len(eval_harm) if eval_harm else 0.0

    final_snap = checkpoints_data.get(
        max(checkpoints_data.keys()) if checkpoints_data else 0,
        {"mean_axis_corr": 0.0, "regime_sep": 0.0},
    )

    initial_snap = checkpoints_data.get(
        CHECKPOINTS[0] if checkpoints_data else 0,
        {"mean_axis_corr": 0.0, "regime_sep": 0.0},
    )

    corr_trajectory = [
        {"episode": ep_i, "mean_axis_corr": checkpoints_data[ep_i]["mean_axis_corr"]}
        for ep_i in sorted(checkpoints_data.keys())
    ]

    return {
        "condition":             condition,
        "seed":                  seed,
        "initial_corr":          initial_snap["mean_axis_corr"],
        "final_corr":            final_snap["mean_axis_corr"],
        "axis_correlation_trajectory": corr_trajectory,
        "regime_sep":            final_snap["regime_sep"],
        "last_q_harm":           last_q_harm,
        "checkpoints":           {
            str(k): {
                "mean_axis_corr":  v["mean_axis_corr"],
                "r_tonic_phasic":  v["r_tonic_phasic"],
                "r_tonic_affect":  v["r_tonic_affect"],
                "r_phasic_affect": v["r_phasic_affect"],
                "regime_sep":      v["regime_sep"],
                "harm_rate":       v["harm_rate"],
            }
            for k, v in checkpoints_data.items()
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(dry_run: bool = False) -> None:
    run_id    = f"{RUN_ID_PREFIX}_{uuid.uuid4().hex[:8]}_v3"
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    print(f"[EXQ-164a] Starting {EXPERIMENT_TYPE}")
    print(f"[EXQ-164a] run_id    = {run_id}")
    print(f"[EXQ-164a] dry_run   = {dry_run}")
    print(f"[EXQ-164a] conditions= {CONDITIONS}")
    print(f"[EXQ-164a] seeds     = {SEEDS}")
    print(f"[EXQ-164a] n_episodes= {N_EPISODES}  (dry_run overrides to 1)")

    cells: List[Dict] = []

    for seed in SEEDS:
        for condition in CONDITIONS:
            print(
                f"[EXQ-164a] --- condition={condition}  seed={seed} ---"
            )
            result = _run_cell(
                seed=seed,
                condition=condition,
                n_episodes=N_EPISODES,
                dry_run=dry_run,
            )
            print(
                f"[EXQ-164a]   DONE  initial_corr={result['initial_corr']:.4f}  "
                f"final_corr={result['final_corr']:.4f}  "
                f"regime_sep={result['regime_sep']:.4f}  "
                f"last_q_harm={result['last_q_harm']:.4f}"
            )
            cells.append(result)

    # ---------------------------------------------------------------------------
    # Criterion evaluation
    # ---------------------------------------------------------------------------

    def _cell(condition: str, seed: int) -> Dict:
        for c in cells:
            if c["condition"] == condition and c["seed"] == seed:
                return c
        raise KeyError(f"Missing cell condition={condition} seed={seed}")

    # C1: WITH final_corr < NO final_corr - CORR_GAP_MIN (both seeds)
    c1_values = [
        _cell("NO_DECORRELATION", s)["final_corr"]
        - _cell("WITH_DECORRELATION", s)["final_corr"]
        for s in SEEDS
    ]
    c1_pass = all(v >= CORR_GAP_MIN for v in c1_values)

    # C2: NO final_corr > NO initial_corr + DRIFT_MIN (both seeds)
    c2_values = [
        _cell("NO_DECORRELATION", s)["final_corr"]
        - _cell("NO_DECORRELATION", s)["initial_corr"]
        for s in SEEDS
    ]
    c2_pass = all(v >= DRIFT_MIN for v in c2_values)

    # C3: WITH regime_sep >= NO regime_sep + REGIME_MARGIN (both seeds)
    c3_values = [
        _cell("WITH_DECORRELATION", s)["regime_sep"]
        - _cell("NO_DECORRELATION", s)["regime_sep"]
        for s in SEEDS
    ]
    c3_pass = all(v >= REGIME_MARGIN for v in c3_values)

    # C4: WITH last_q_harm <= NO last_q_harm + THRESH_HARM_MARGIN (both seeds)
    c4_values = [
        _cell("NO_DECORRELATION",   s)["last_q_harm"]
        - _cell("WITH_DECORRELATION", s)["last_q_harm"]
        for s in SEEDS
    ]
    c4_pass = all(v >= -THRESH_HARM_MARGIN for v in c4_values)

    # Outcome determination
    if c1_pass and c2_pass and c3_pass and c4_pass:
        outcome = "PASS"
    elif c1_pass and c2_pass and c4_pass and not c3_pass:
        outcome = "PARTIAL_DRIFT_CONFIRMED"
    elif c1_pass and c3_pass and c4_pass and not c2_pass:
        outcome = "PARTIAL_NO_DRIFT"
    elif not c1_pass:
        outcome = "FAIL"
    else:
        outcome = "PARTIAL_INCONCLUSIVE"

    print(f"[EXQ-164a] C1 (WITH corr < NO corr - gap):   {c1_pass}  gaps={[round(v,4) for v in c1_values]}")
    print(f"[EXQ-164a] C2 (NO corr drifts by DRIFT_MIN): {c2_pass}  drifts={[round(v,4) for v in c2_values]}")
    print(f"[EXQ-164a] C3 (WITH regime_sep >= NO + margin): {c3_pass}  diffs={[round(v,4) for v in c3_values]}")
    print(f"[EXQ-164a] C4 (WITH harm <= NO + margin):    {c4_pass}  diffs={[round(v,4) for v in c4_values]}")
    print(f"[EXQ-164a] Outcome: {outcome}")

    # ---------------------------------------------------------------------------
    # Build summary metrics
    # ---------------------------------------------------------------------------
    summary_metrics: Dict = {}
    for condition in CONDITIONS:
        for seed in SEEDS:
            cell = _cell(condition, seed)
            summary_metrics[f"{condition}_seed{seed}"] = {
                "initial_corr": round(cell["initial_corr"], 5),
                "final_corr":   round(cell["final_corr"],   5),
                "regime_sep":   round(cell["regime_sep"],   5),
                "last_q_harm":  round(cell["last_q_harm"],  5),
            }

    per_seed_results: Dict = {}
    for seed in SEEDS:
        with_cell = _cell("WITH_DECORRELATION", seed)
        no_cell   = _cell("NO_DECORRELATION",   seed)
        per_seed_results[str(seed)] = {
            "WITH_DECORRELATION": {
                "initial_corr":              round(with_cell["initial_corr"], 5),
                "final_corr":                round(with_cell["final_corr"],   5),
                "regime_sep":                round(with_cell["regime_sep"],   5),
                "last_q_harm":               round(with_cell["last_q_harm"],  5),
                "axis_correlation_trajectory": with_cell["axis_correlation_trajectory"],
                "checkpoints":               with_cell["checkpoints"],
            },
            "NO_DECORRELATION": {
                "initial_corr":              round(no_cell["initial_corr"], 5),
                "final_corr":                round(no_cell["final_corr"],   5),
                "regime_sep":                round(no_cell["regime_sep"],   5),
                "last_q_harm":               round(no_cell["last_q_harm"],  5),
                "axis_correlation_trajectory": no_cell["axis_correlation_trajectory"],
                "checkpoints":               no_cell["checkpoints"],
            },
            "C1_corr_gap":        round(
                no_cell["final_corr"] - with_cell["final_corr"], 5
            ),
            "C2_drift":           round(
                no_cell["final_corr"] - no_cell["initial_corr"], 5
            ),
            "C3_regime_sep_diff": round(
                with_cell["regime_sep"] - no_cell["regime_sep"], 5
            ),
            "C4_harm_diff":       round(
                no_cell["last_q_harm"] - with_cell["last_q_harm"], 5
            ),
        }

    # ---------------------------------------------------------------------------
    # Write output JSON (skip in dry_run)
    # ---------------------------------------------------------------------------
    output_dir = (
        Path(__file__).resolve().parents[1].parent
        / "REE_assembly" / "evidence" / "experiments"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{run_id}.json"

    pack = {
        "run_id":                run_id,
        "experiment_type":       EXPERIMENT_TYPE,
        "claim_ids_tested":      CLAIM_IDS,
        "architecture_epoch":    "ree_hybrid_guardrails_v1",
        "outcome":               outcome,
        "criteria_met": {
            "C1": c1_pass,
            "C2": c2_pass,
            "C3": c3_pass,
            "C4": c4_pass,
        },
        "criteria_values": {
            "C1_corr_gaps":         [round(v, 5) for v in c1_values],
            "C2_drift_values":      [round(v, 5) for v in c2_values],
            "C3_regime_sep_diffs":  [round(v, 5) for v in c3_values],
            "C4_harm_diffs":        [round(v, 5) for v in c4_values],
        },
        "summary_metrics":       summary_metrics,
        "per_seed_results":      per_seed_results,
        "config": {
            "n_episodes":          N_EPISODES,
            "n_eval_episodes":     N_EVAL_EPISODES,
            "steps_per_episode":   STEPS_PER_EPISODE,
            "checkpoints":         CHECKPOINTS,
            "tonic_tau":           TONIC_TAU,
            "corr_buffer_size":    CORR_BUFFER_SIZE,
            "decorr_weight":       DECORR_WEIGHT,
            "regime_harm_thresh":  REGIME_HARM_THRESH,
            "window_size":         WINDOW_SIZE,
            "grid_size":           GRID_SIZE,
            "n_hazards":           N_HAZARDS,
            "n_resources":         N_RESOURCES,
            "hazard_harm":         HAZARD_HARM,
            "lr":                  LR,
            "ent_bonus":           ENT_BONUS,
            "hidden_dim":          HIDDEN_DIM,
            "world_dim":           WORLD_DIM,
            "self_dim":            SELF_DIM,
            "seeds":               SEEDS,
            "conditions":          CONDITIONS,
        },
        "thresholds": {
            "CORR_GAP_MIN":        CORR_GAP_MIN,
            "DRIFT_MIN":           DRIFT_MIN,
            "REGIME_MARGIN":       REGIME_MARGIN,
            "THRESH_HARM_MARGIN":  THRESH_HARM_MARGIN,
        },
        "evidence_class":        "experimental",
        "evidence_direction": (
            "supports" if outcome == "PASS"
            else "mixed" if outcome.startswith("PARTIAL")
            else "weakens"
        ),
        "summary": (
            f"Outcome={outcome}. "
            f"WITH_DECORRELATION vs NO_DECORRELATION (FULL_AXES, 2 seeds). "
            f"C1 (corr gap): {c1_pass}  gaps={[round(v,4) for v in c1_values]}. "
            f"C2 (drift): {c2_pass}  drifts={[round(v,4) for v in c2_values]}. "
            f"C3 (regime sep): {c3_pass}  diffs={[round(v,4) for v in c3_values]}. "
            f"C4 (harm quality): {c4_pass}  diffs={[round(v,4) for v in c4_values]}."
        ),
        "dry_run":               dry_run,
        "timestamp_utc":         timestamp,
    }

    if not dry_run:
        with open(output_path, "w") as fh:
            json.dump(pack, fh, indent=2)
        print(f"[EXQ-164a] Written to {output_path}")
    else:
        print("[EXQ-164a] dry_run=True -- output not written")
        print(f"[EXQ-164a] Would write to {output_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="EXQ-164: MECH-142 axis decorrelation maintenance"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run 1 episode per cell, skip file output",
    )
    args = parser.parse_args()
    main(dry_run=args.dry_run)
