#!/opt/local/bin/python3
"""
V3-EXQ-163 -- MECH-141: Dual-timescale tri-loop arbitration.

Claim:    MECH-141, MECH-140
Proposal: EXP-0118

MECH-141 asserts:
  Tri-loop arbitration requires BOTH a slow proactive pathway (accumulated
  conflict, EMA tau=30 steps) AND a fast reactive hyperdirect pathway
  (current-step conflict spike). These cannot be collapsed into a single
  timescale.

Background:
  EXQ-156 (already queued) tests PRIORITY_HARM vs WEIGHTED_SUM vs WTA
  arbitration policies. EXQ-163 is complementary: it tests whether
  DUAL-TIMESCALE arbitration outperforms SINGLE-TIMESCALE (slow-only or
  fast-only) arbitration, regardless of the policy type.

  Biological basis:
  - Slow pathway: cortico-striatal indirect pathway; accumulates conflict
    over many steps (ACC sustained activity, EMA analog).
  - Fast pathway: subthalamic nucleus (STN) hyperdirect pathway; responds
    to current-step conflict spikes within a single cycle.
  Both converge on the output nuclei (GPi/SNr) to gate motor output. If
  only the slow pathway is active, the system is sluggish in novel hazard
  conditions. If only the fast pathway is active, it is reactive but cannot
  represent accumulated motivational urgency. MECH-141 predicts the dual
  pathway is necessary for full harm avoidance.

Three conditions (3 x 2 seeds = 6 cells):

Condition A: DUAL_TIMESCALE
  slow_conflict = EMA(conflict_signal, tau=30)
  fast_conflict = conflict_signal (current step)
  g_commit = sigmoid(w_slow * slow_conflict + w_fast * fast_conflict)
  w_slow=0.5, w_fast=0.5

Condition B: SLOW_ONLY
  slow_conflict = EMA(conflict_signal, tau=30)
  g_commit = sigmoid(w_slow * slow_conflict)
  w_slow=1.0

Condition C: FAST_ONLY
  fast_conflict = conflict_signal (current step)
  g_commit = sigmoid(w_fast * fast_conflict)
  w_fast=1.0

All conditions share the same three gate networks:
  g_motor = sigmoid(Linear(z_self, 1))
  g_cog   = sigmoid(Linear(z_world, 1))
  g_mot   = sigmoid(Linear([z_world, z_self], 1))

conflict_signal = |g_motor - g_cog|  (instantaneous loop tension)

Loss:
  g_commit * harm_eval_loss + (1 - g_commit) * world_pred_loss
  + GATE_REG * gate_entropy_reg

Pre-registered thresholds
--------------------------
CONFLICT_THRESH           = 0.2
GATE_REG                  = 0.001
THRESH_HARM_MARGIN        = 0.02
THRESH_SPREAD_MIN         = 0.05
THRESH_CORR_MAX           = 0.70
THRESH_CONFLICT_STEPS_MIN = 50

C1: DUAL_TIMESCALE harm_rate < best(SLOW_ONLY, FAST_ONLY) harm_rate
    - THRESH_HARM_MARGIN (both seeds)
    -- dual timescale outperforms either single-timescale pathway

C2: DUAL_TIMESCALE inter_loop_spread >= SLOW_ONLY inter_loop_spread
    (both seeds)
    -- dual timescale preserves loop independence vs slow-only

C3: DUAL_TIMESCALE timescale_gate_correlation < THRESH_CORR_MAX (both seeds)
    -- slow and fast gates carry partially independent information

C4: DUAL_TIMESCALE n_conflict_steps > THRESH_CONFLICT_STEPS_MIN (both seeds)
    -- data quality: enough conflict events

PASS: C1 + C2 + C3 + C4
  => Dual timescale is superior; MECH-141 supported.
PARTIAL_TIMESCALES_CORRELATED: C1+C2+C4 but NOT C3
  => Dual timescale performs well but slow/fast gates highly correlated --
     environment may not produce events that differentiate them.
PARTIAL_NO_ADVANTAGE: C3+C4 but NOT C1
  => Gates are independent but dual timescale doesn't improve harm avoidance
     at this scale.
FAIL: C4 fails
  => Not enough conflict events for reliable metric estimation.
"""

import sys
import math
import random
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from ree_core.environment.causal_grid_world import CausalGridWorldV2


# ---------------------------------------------------------------------------
# Experiment identifiers
# ---------------------------------------------------------------------------
EXPERIMENT_TYPE = "v3_exq_163_mech141_dual_timescale_arbitration"
CLAIM_IDS       = ["MECH-141", "MECH-140"]
RUN_ID_PREFIX   = "v3_exq_163_mech141_dual_timescale_arbitration"

# ---------------------------------------------------------------------------
# Conditions and seeds
# ---------------------------------------------------------------------------
CONDITIONS = ["DUAL_TIMESCALE", "SLOW_ONLY", "FAST_ONLY"]
SEEDS      = [42, 123]

# ---------------------------------------------------------------------------
# Protocol constants
# ---------------------------------------------------------------------------
N_EPISODES        = 400
N_EVAL_EPISODES   = 100     # last quartile
STEPS_PER_EPISODE = 200
COMMIT_STEP_FRAC  = 0.5     # fraction of episode after which commit logic active

# Timescale parameters
EMA_TAU_SLOW = 30.0
W_SLOW       = 0.5
W_FAST       = 0.5

# Regularisation
GATE_REG = 0.001

# Environment
GRID_SIZE   = 8
N_HAZARDS   = 4
N_RESOURCES = 2
HAZARD_HARM = 0.05
NAV_BIAS    = 0.6   # unused in random-action protocol; documents intent

# Optimisation
LR = 1e-3

# Model dims
HIDDEN_DIM = 32
WORLD_DIM  = 32
SELF_DIM   = 16
HARM_DIM   = 8   # reserved; harm evaluated via harm_pred (scalar head)

# Observation dims (CausalGridWorldV2 size=8)
WORLD_OBS_DIM = 250
SELF_OBS_DIM  = 12
ACTION_DIM    = 5

# ---------------------------------------------------------------------------
# Pre-registered thresholds
# ---------------------------------------------------------------------------
CONFLICT_THRESH           = 0.2
THRESH_HARM_MARGIN        = 0.02
THRESH_SPREAD_MIN         = 0.05
THRESH_CORR_MAX           = 0.70
THRESH_CONFLICT_STEPS_MIN = 50


# ---------------------------------------------------------------------------
# Models (inline, no ree_core imports beyond environment)
# ---------------------------------------------------------------------------

class WorldEncoder(nn.Module):
    """Linear(obs_dim, world_dim) + LayerNorm -> z_world."""
    def __init__(self, obs_dim: int, world_dim: int):
        super().__init__()
        self.linear = nn.Linear(obs_dim, world_dim)
        self.norm   = nn.LayerNorm(world_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.linear(x))


class SelfEncoder(nn.Module):
    """Linear(obs_dim, self_dim) + LayerNorm -> z_self."""
    def __init__(self, obs_dim: int, self_dim: int):
        super().__init__()
        self.linear = nn.Linear(obs_dim, self_dim)
        self.norm   = nn.LayerNorm(self_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.linear(x))


class LoopGate(nn.Module):
    """Single BG-like loop gate: Linear(input_dim, 1) -> sigmoid."""
    def __init__(self, input_dim: int):
        super().__init__()
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.fc(x))


class HarmPredictor(nn.Module):
    """E3 analog: Linear(z_world || z_self, 1) -> harm scalar."""
    def __init__(self, world_dim: int, self_dim: int):
        super().__init__()
        self.fc = nn.Linear(world_dim + self_dim, 1)

    def forward(self, z_world: torch.Tensor, z_self: torch.Tensor) -> torch.Tensor:
        return self.fc(torch.cat([z_world, z_self], dim=-1))


class WorldPredictor(nn.Module):
    """E1 analog: Linear(z_world || action, world_dim) -> z_world_next_pred."""
    def __init__(self, world_dim: int, action_dim: int):
        super().__init__()
        self.fc = nn.Linear(world_dim + action_dim, world_dim)

    def forward(self, z_world: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.fc(torch.cat([z_world, action], dim=-1))


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _get_obs_tensor(obs_dict: dict, key: str, fallback_dim: int) -> torch.Tensor:
    """Safely extract a flat tensor from obs_dict, zero-padding or truncating."""
    raw = obs_dict.get(key)
    if raw is None:
        return torch.zeros(fallback_dim)
    t = torch.tensor(raw, dtype=torch.float32).flatten()
    if t.shape[0] < fallback_dim:
        t = F.pad(t, (0, fallback_dim - t.shape[0]))
    elif t.shape[0] > fallback_dim:
        t = t[:fallback_dim]
    return t


def _pearson_r(xs: List[float], ys: List[float]) -> float:
    """Pearson correlation between two equal-length lists. Returns 0.0 on degenerate inputs."""
    n = len(xs)
    if n < 2:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    denom_x = math.sqrt(sum((x - mx) ** 2 for x in xs))
    denom_y = math.sqrt(sum((y - my) ** 2 for y in ys))
    if denom_x < 1e-12 or denom_y < 1e-12:
        return 0.0
    return num / (denom_x * denom_y)


# ---------------------------------------------------------------------------
# Core experiment runner — one condition x one seed
# ---------------------------------------------------------------------------

def _run_condition(
    seed: int,
    condition: str,
    total_episodes: int,
    measurement_episodes: int,
    steps_per_episode: int,
    lr: float,
    dry_run: bool,
) -> Dict:
    """
    Run one condition x seed cell.
    Returns per-cell metrics dict.
    """
    torch.manual_seed(seed)
    random.seed(seed)

    env = CausalGridWorldV2(
        size=GRID_SIZE,
        num_hazards=N_HAZARDS,
        num_resources=N_RESOURCES,
        hazard_harm=HAZARD_HARM,
        env_drift_interval=5,
        env_drift_prob=0.3,
        seed=seed,
    )

    # Build shared components
    world_enc  = WorldEncoder(WORLD_OBS_DIM, WORLD_DIM)
    self_enc   = SelfEncoder(SELF_OBS_DIM, SELF_DIM)
    harm_pred  = HarmPredictor(WORLD_DIM, SELF_DIM)
    world_pred = WorldPredictor(WORLD_DIM, ACTION_DIM)

    # Three loop gates (same for all conditions)
    gate_motor = LoopGate(SELF_DIM)                    # responds to body state
    gate_cog   = LoopGate(WORLD_DIM)                   # responds to world state
    gate_mot   = LoopGate(WORLD_DIM + SELF_DIM)        # harm-proximity motivated

    all_params = (
        list(world_enc.parameters())
        + list(self_enc.parameters())
        + list(harm_pred.parameters())
        + list(world_pred.parameters())
        + list(gate_motor.parameters())
        + list(gate_cog.parameters())
        + list(gate_mot.parameters())
    )
    optimizer = optim.Adam(all_params, lr=lr)

    measurement_start = total_episodes - measurement_episodes
    if dry_run:
        total_episodes    = 3
        measurement_start = 1

    # EMA state (reset per run)
    slow_conflict_ema = 0.0
    ema_alpha = 1.0 / EMA_TAU_SLOW  # per-step update coefficient

    # Measurement buffers
    harm_vals:          List[float] = []
    inter_loop_spreads: List[float] = []
    slow_conflict_vals: List[float] = []   # for slow_conflict_mean and correlation
    fast_conflict_vals: List[float] = []   # for fast_conflict_variance and correlation
    n_conflict_steps: int = 0

    _, obs_dict = env.reset()

    for ep in range(total_episodes):
        ep_harm  = 0.0
        ep_steps = 0
        in_measurement = ep >= measurement_start

        for step in range(steps_per_episode):
            obs_world = _get_obs_tensor(obs_dict, "world_state", WORLD_OBS_DIM)
            obs_self  = _get_obs_tensor(obs_dict, "body_state", SELF_OBS_DIM)

            z_world = world_enc(obs_world)
            z_self  = self_enc(obs_self)

            # Random action policy
            action_idx = random.randint(0, ACTION_DIM - 1)
            action = torch.zeros(ACTION_DIM)
            action[action_idx] = 1.0

            # Step environment
            _, _, done, _, obs_dict_next = env.step(action.unsqueeze(0))

            obs_world_next      = _get_obs_tensor(obs_dict_next, "world_state", WORLD_OBS_DIM)
            z_world_next_actual = world_enc(obs_world_next).detach()

            harm_signal = float(obs_dict_next.get("harm_signal", 0.0))
            harm_actual = torch.tensor([harm_signal], dtype=torch.float32)

            # Predictions
            harm_pred_val   = harm_pred(z_world, z_self)
            z_world_next_p  = world_pred(z_world, action)
            harm_eval_loss  = F.mse_loss(harm_pred_val, harm_actual)
            world_pred_loss = F.mse_loss(z_world_next_p, z_world_next_actual)

            # Compute loop gate values (detached from encoder gradients)
            g_m = gate_motor(z_self.detach())                                              # [1]
            g_c = gate_cog(z_world.detach())                                               # [1]
            g_v = gate_mot(torch.cat([z_world.detach(), z_self.detach()], dim=-1))        # [1]

            g_motor_val = float(g_m.item())
            g_cog_val   = float(g_c.item())
            g_mot_val   = float(g_v.item())

            # Instantaneous conflict signal: |g_motor - g_cog|
            fast_conflict = abs(g_motor_val - g_cog_val)

            # EMA slow conflict
            slow_conflict_ema = (1.0 - ema_alpha) * slow_conflict_ema + ema_alpha * fast_conflict
            slow_conflict = slow_conflict_ema

            # g_commit computation depends on condition
            if condition == "DUAL_TIMESCALE":
                commit_logit = W_SLOW * slow_conflict + W_FAST * fast_conflict
                # build differentiable g_commit for loss weighting
                # We use a linear combination of the gate tensors scaled by the
                # conflict weights so gradients flow back through the gates.
                g_commit = torch.sigmoid(
                    torch.tensor(W_SLOW * slow_conflict + W_FAST * fast_conflict,
                                 dtype=torch.float32)
                )

            elif condition == "SLOW_ONLY":
                g_commit = torch.sigmoid(
                    torch.tensor(slow_conflict, dtype=torch.float32)
                )

            else:  # FAST_ONLY
                g_commit = torch.sigmoid(
                    torch.tensor(fast_conflict, dtype=torch.float32)
                )

            g_commit_val = float(g_commit.item())

            # Loss: gated combination + gate entropy regularisation
            # Use g_commit as a scalar weight (no gradient through it for
            # the commit path -- commit signal is scalar from conflict, not
            # from a differentiable module; gradients flow through the gate
            # networks via the entropy regularisation and the harm/world paths)
            g_w = g_commit.detach()
            total_loss = g_w * harm_eval_loss + (1.0 - g_w) * world_pred_loss

            # Gate entropy regularisation (maximise entropy = prevent saturation)
            ent_m = -(g_m * torch.log(g_m + 1e-6) + (1.0 - g_m) * torch.log(1.0 - g_m + 1e-6))
            ent_c = -(g_c * torch.log(g_c + 1e-6) + (1.0 - g_c) * torch.log(1.0 - g_c + 1e-6))
            ent_v = -(g_v * torch.log(g_v + 1e-6) + (1.0 - g_v) * torch.log(1.0 - g_v + 1e-6))
            gate_ent_reg = -GATE_REG * (ent_m + ent_c + ent_v)  # negative = maximise entropy
            total_loss = total_loss + gate_ent_reg

            optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(all_params, 1.0)
            optimizer.step()

            ep_harm  += harm_signal
            ep_steps += 1

            # Measurement collection
            if in_measurement:
                slow_conflict_vals.append(slow_conflict)
                fast_conflict_vals.append(fast_conflict)

                if fast_conflict > CONFLICT_THRESH:
                    n_conflict_steps += 1
                    # inter_loop_spread: mean pairwise |diff| at conflict steps
                    spread = (
                        abs(g_motor_val - g_cog_val)
                        + abs(g_motor_val - g_mot_val)
                        + abs(g_cog_val - g_mot_val)
                    ) / 3.0
                    inter_loop_spreads.append(spread)

            if done:
                _, obs_dict = env.reset()
            else:
                obs_dict = obs_dict_next

        if in_measurement:
            harm_vals.append(ep_harm / max(ep_steps, 1))

        if ep % 100 == 0:
            avg_harm = ep_harm / max(ep_steps, 1)
            print(
                f"  [{condition}] seed={seed} ep={ep}/{total_episodes}"
                f" ep_harm={avg_harm:.5f}"
                f" slow_ema={slow_conflict:.4f}"
                f" fast={fast_conflict:.4f}",
                flush=True,
            )

    # Aggregate metrics
    harm_rate    = float(sum(harm_vals) / max(len(harm_vals), 1))
    inter_spread = float(sum(inter_loop_spreads) / max(len(inter_loop_spreads), 1))

    # Timescale gate correlation (DUAL_TIMESCALE only; others report 0.0)
    timescale_corr = 0.0
    if condition == "DUAL_TIMESCALE" and len(slow_conflict_vals) >= 2:
        timescale_corr = abs(_pearson_r(slow_conflict_vals, fast_conflict_vals))

    # slow_conflict_mean
    slow_conflict_mean = (
        float(sum(slow_conflict_vals) / max(len(slow_conflict_vals), 1))
    )

    # fast_conflict_variance
    if len(fast_conflict_vals) >= 2:
        fc_mean = sum(fast_conflict_vals) / len(fast_conflict_vals)
        fast_conflict_var = float(
            sum((v - fc_mean) ** 2 for v in fast_conflict_vals) / len(fast_conflict_vals)
        )
    else:
        fast_conflict_var = 0.0

    print(
        f"  [{condition}] seed={seed} harm_rate={harm_rate:.5f}"
        f" inter_spread={inter_spread:.4f}"
        f" timescale_corr={timescale_corr:.4f}"
        f" n_conflict={n_conflict_steps}"
        f" slow_mean={slow_conflict_mean:.4f}"
        f" fast_var={fast_conflict_var:.5f}",
        flush=True,
    )

    return {
        "condition":                    condition,
        "seed":                         seed,
        "harm_rate":                    harm_rate,
        "inter_loop_spread":            inter_spread,
        "timescale_gate_correlation":   timescale_corr,
        "n_conflict_steps":             n_conflict_steps,
        "slow_conflict_mean":           slow_conflict_mean,
        "fast_conflict_variance":       fast_conflict_var,
    }


# ---------------------------------------------------------------------------
# Criterion evaluation
# ---------------------------------------------------------------------------

def _evaluate_criteria(
    results_by_condition: Dict[str, List[Dict]],
) -> Tuple[Dict[str, bool], str]:
    """Evaluate pre-registered criteria. Returns (criteria_dict, outcome_string)."""
    n_s = len(SEEDS)

    def _mean(cond: str, key: str) -> float:
        return sum(r[key] for r in results_by_condition[cond]) / n_s

    # C1: DUAL harm_rate < best(SLOW_ONLY, FAST_ONLY) - THRESH_HARM_MARGIN (both seeds)
    best_single_hr_per_seed = [
        min(
            results_by_condition["SLOW_ONLY"][i]["harm_rate"],
            results_by_condition["FAST_ONLY"][i]["harm_rate"],
        )
        for i in range(n_s)
    ]
    c1 = all(
        results_by_condition["DUAL_TIMESCALE"][i]["harm_rate"]
        < best_single_hr_per_seed[i] - THRESH_HARM_MARGIN
        for i in range(n_s)
    )

    # C2: DUAL inter_loop_spread >= SLOW_ONLY inter_loop_spread (both seeds)
    c2 = all(
        results_by_condition["DUAL_TIMESCALE"][i]["inter_loop_spread"]
        >= results_by_condition["SLOW_ONLY"][i]["inter_loop_spread"]
        for i in range(n_s)
    )

    # C3: DUAL timescale_gate_correlation < THRESH_CORR_MAX (both seeds)
    c3 = all(
        results_by_condition["DUAL_TIMESCALE"][i]["timescale_gate_correlation"]
        < THRESH_CORR_MAX
        for i in range(n_s)
    )

    # C4: DUAL n_conflict_steps > THRESH_CONFLICT_STEPS_MIN (both seeds)
    c4 = all(
        results_by_condition["DUAL_TIMESCALE"][i]["n_conflict_steps"]
        > THRESH_CONFLICT_STEPS_MIN
        for i in range(n_s)
    )

    criteria = {"C1": c1, "C2": c2, "C3": c3, "C4": c4}

    # Outcome determination
    if c1 and c2 and c3 and c4:
        outcome = "PASS"
    elif c1 and c2 and c4 and not c3:
        outcome = "PARTIAL_TIMESCALES_CORRELATED"
    elif c3 and c4 and not c1:
        outcome = "PARTIAL_NO_ADVANTAGE"
    else:
        outcome = "FAIL"

    return criteria, outcome


# ---------------------------------------------------------------------------
# Main experiment orchestrator
# ---------------------------------------------------------------------------

def run_experiment(dry_run: bool = False) -> Dict:
    """Run all condition x seed cells and return the result pack."""

    # Build a stable run_id
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    run_id = f"{RUN_ID_PREFIX}_{ts}_v3"

    print(f"\nEXQ-163: {EXPERIMENT_TYPE}", flush=True)
    print(f"run_id: {run_id}", flush=True)
    print(f"dry_run: {dry_run}", flush=True)
    print(
        f"conditions={CONDITIONS}  seeds={SEEDS}"
        f"  n_episodes={N_EPISODES}  n_eval={N_EVAL_EPISODES}",
        flush=True,
    )

    results_by_condition: Dict[str, List[Dict]] = {c: [] for c in CONDITIONS}

    for condition in CONDITIONS:
        for seed in SEEDS:
            print(
                f"\n--- Condition: {condition}  Seed: {seed} ---",
                flush=True,
            )
            cell_result = _run_condition(
                seed=seed,
                condition=condition,
                total_episodes=N_EPISODES,
                measurement_episodes=N_EVAL_EPISODES,
                steps_per_episode=STEPS_PER_EPISODE,
                lr=LR,
                dry_run=dry_run,
            )
            results_by_condition[condition].append(cell_result)

    # Evaluate criteria
    criteria, outcome = _evaluate_criteria(results_by_condition)

    print(f"\nCriteria: {criteria}", flush=True)
    print(f"Outcome:  {outcome}", flush=True)

    # Summary metrics
    def _mean_metric(cond: str, key: str) -> float:
        vals = [r[key] for r in results_by_condition[cond]]
        return float(sum(vals) / max(len(vals), 1))

    summary_metrics: Dict = {}
    for cond in CONDITIONS:
        summary_metrics[cond] = {
            "harm_rate_mean":                 _mean_metric(cond, "harm_rate"),
            "inter_loop_spread_mean":         _mean_metric(cond, "inter_loop_spread"),
            "timescale_gate_correlation_mean": _mean_metric(cond, "timescale_gate_correlation"),
            "n_conflict_steps_mean":          _mean_metric(cond, "n_conflict_steps"),
            "slow_conflict_mean":             _mean_metric(cond, "slow_conflict_mean"),
            "fast_conflict_variance_mean":    _mean_metric(cond, "fast_conflict_variance"),
        }

    # Evidence direction
    if outcome == "PASS":
        evidence_direction = "supports"
    elif outcome in ("PARTIAL_TIMESCALES_CORRELATED", "PARTIAL_NO_ADVANTAGE"):
        evidence_direction = "mixed"
    else:
        evidence_direction = "inconclusive"

    pack = {
        "run_id":              run_id,
        "experiment_type":     EXPERIMENT_TYPE,
        "claim_ids_tested":    CLAIM_IDS,
        "architecture_epoch":  "ree_hybrid_guardrails_v1",
        "outcome":             outcome,
        "evidence_direction":  evidence_direction,
        "evidence_class":      "discriminative_triple",
        "criteria_met":        criteria,
        "pre_registered_thresholds": {
            "CONFLICT_THRESH":           CONFLICT_THRESH,
            "THRESH_HARM_MARGIN":        THRESH_HARM_MARGIN,
            "THRESH_SPREAD_MIN":         THRESH_SPREAD_MIN,
            "THRESH_CORR_MAX":           THRESH_CORR_MAX,
            "THRESH_CONFLICT_STEPS_MIN": THRESH_CONFLICT_STEPS_MIN,
        },
        "summary_metrics":   summary_metrics,
        "per_seed_results":  {cond: results_by_condition[cond] for cond in CONDITIONS},
        "config": {
            "n_episodes":        N_EPISODES,
            "n_eval_episodes":   N_EVAL_EPISODES,
            "steps_per_episode": STEPS_PER_EPISODE,
            "commit_step_frac":  COMMIT_STEP_FRAC,
            "ema_tau_slow":      EMA_TAU_SLOW,
            "w_slow":            W_SLOW,
            "w_fast":            W_FAST,
            "gate_reg":          GATE_REG,
            "grid_size":         GRID_SIZE,
            "n_hazards":         N_HAZARDS,
            "n_resources":       N_RESOURCES,
            "hazard_harm":       HAZARD_HARM,
            "lr":                LR,
            "hidden_dim":        HIDDEN_DIM,
            "world_dim":         WORLD_DIM,
            "self_dim":          SELF_DIM,
            "harm_dim":          HARM_DIM,
            "conflict_thresh":   CONFLICT_THRESH,
        },
        "seeds":     SEEDS,
        "scenario": (
            "Three-condition dual-timescale arbitration test:"
            " DUAL_TIMESCALE (w_slow=0.5 * EMA_tau30 + w_fast=0.5 * current_step),"
            " SLOW_ONLY (w_slow=1.0, EMA_tau30),"
            " FAST_ONLY (w_fast=1.0, current-step only)."
            " All conditions: shared g_motor/g_cog/g_mot gate nets,"
            " conflict_signal=|g_motor-g_cog|, random action policy."
            " 3 conditions x 2 seeds = 6 cells."
            " CausalGridWorldV2 size=8 num_hazards=4 num_resources=2"
            " hazard_harm=0.05 env_drift_interval=5 env_drift_prob=0.3."
            " 400 total episodes, 100 measurement episodes (last quartile)."
        ),
        "interpretation": (
            "PASS => MECH-141 supported: dual-timescale arbitration outperforms"
            " either single-timescale pathway on harm avoidance, preserves loop"
            " independence, and slow/fast signals carry partially independent info."
            " PARTIAL_TIMESCALES_CORRELATED => dual timescale performs well but"
            " slow/fast gates are highly correlated -- env may not produce events"
            " that differentiate them; scale up env drift or extend episode length."
            " PARTIAL_NO_ADVANTAGE => slow/fast are informationally distinct but"
            " dual timescale does not improve harm avoidance at this scale;"
            " MECH-141 remains open -- may need stronger hazard density or"
            " longer evaluation window."
            " FAIL => insufficient conflict events; increase N_HAZARDS or"
            " HAZARD_HARM and re-run."
        ),
        "dry_run":       dry_run,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }

    if not dry_run:
        out_dir = (
            Path(__file__).resolve().parents[2]
            / "REE_assembly"
            / "evidence"
            / "experiments"
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{run_id}.json"
        with open(out_path, "w") as f:
            json.dump(pack, f, indent=2)
        print(f"\nResult pack written to: {out_path}", flush=True)
    else:
        print("\n[dry_run] Result pack NOT written.", flush=True)
        print(json.dumps(pack, indent=2), flush=True)

    return pack


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv
    result = run_experiment(dry_run=dry_run)
    print(f"\nDone. Outcome: {result['outcome']}", flush=True)
