#!/opt/local/bin/python3
"""
V3-EXQ-161 -- Q-024: Trajectory-integral representation triple.
              Are DESCRIPTIVE, PRESCRIPTIVE, and DIAGNOSTIC test-type variants
              all needed, or does one subsume the others?

Claim:    Q-024
Proposal: EXP-0116 (EVB-0085)

Q-024 asks:
  "What is the correct formal representation for 'threshold/feedback processes
  bounded by {x...x_n} reliably reaching emergent state q' -- and are descriptive,
  prescriptive, and diagnostic variants all needed or does one subsume the others?"

  Background (from claims.yaml Q-024 notes):
    Three test-type variants exist for characterizing whether a REE agent
    reliably reaches a low-harm emergent state q:

    (1) DESCRIPTIVE -- attractor/ergodic characterization of what q the system
        tends toward. Does NOT guarantee reaching q from all initial states;
        simply reports what stable distribution the agent occupies.

    (2) PRESCRIPTIVE -- proof (or empirical proxy) that a designed system reaches
        target q. Lyapunov-like: energy function V(state) must decrease
        monotonically. In REE: cumulative harm must decrease monotonically across
        training phases (Lyapunov proxy: dV/dt < 0 equivalent).

    (3) DIAGNOSTIC -- detect when nth-order dynamics diverge from (n-1)th
        predictions. Transfer entropy from harm-history to future-harm:
        TE(harm_t-1 -> harm_t+k) > THRESH_TE_BASELINE means the system has
        detectable higher-order dynamics not captured by a memoryless (0th-order)
        model. A diagnostic pass means nth-order dynamics are non-trivial.

    Q-023 analysis (2026-03-25) clarifies: MECH-127 (counterfactual utility)
    breaks the prescriptive framework (standard potential game), so DESCRIPTIVE
    convergence does not guarantee PRESCRIPTIVE proof. And PRESCRIPTIVE proof
    does not guarantee that DIAGNOSTIC (higher-order dynamics) is trivial.

    Core discriminative hypothesis:
      If PRESCRIPTIVE and DIAGNOSTIC always co-vary, the diagnostic is redundant.
      If they dissociate (a system can pass PRESCRIPTIVE but fail DIAGNOSTIC or
      vice versa), all three variants are irreducibly needed.

  Three conditions -- each represents one test-type instantiated in a REE agent:

  Condition A: DESCRIPTIVE
    Agent trained in a stable environment with mild drift.
    Measurement: harm_rate in last quartile (attractor characterization).
    No Lyapunov constraint. No transfer-entropy measurement required.
    Discriminative prediction: system reaches low harm_rate (q_descriptive)
    but the harm trajectory may not be monotonically decreasing (oscillations
    allowed as long as final state is low).

  Condition B: PRESCRIPTIVE
    Agent trained with a Lyapunov-proxy constraint: a penalty on non-monotone
    harm increases. Specifically, a "Lyapunov loss" penalises any episode where
    harm_rate exceeds the running minimum plus a margin. This enforces
    monotonically non-increasing harm (Lyapunov energy function proxy).
    Discriminative prediction: PRESCRIPTIVE harm_rate_final <= DESCRIPTIVE
    harm_rate_final (Lyapunov constraint improves convergence quality), AND
    monotone_violation_rate < THRESH_MONO (few violations of monotone decrease).

  Condition C: DIAGNOSTIC
    Agent trained without Lyapunov constraint (same as DESCRIPTIVE).
    Additional measurement: transfer entropy TE(harm_t -> harm_t+k) for k=1,5,10.
    If TE > THRESH_TE_BASELINE, the harm dynamics have detectable higher-order
    (non-Markov) structure -- meaning a 1st-order Markov model (prescriptive
    framework) is insufficient and the diagnostic is irreducible.
    Discriminative prediction: TE_k1 > THRESH_TE_BASELINE, confirming that
    harm dynamics are NOT memoryless even when the system converges descriptively.

  Discriminative question:
    (i)  Does PRESCRIPTIVE harm_rate <= DESCRIPTIVE harm_rate?
         (If yes: the Lyapunov constraint adds value over pure attractor tracking.)
    (ii) Does DIAGNOSTIC reveal non-trivial transfer entropy even when both
         DESCRIPTIVE and PRESCRIPTIVE converge?
         (If yes: DIAGNOSTIC is irreducible -- prescriptive convergence does not
         imply Markovian harm dynamics; nth-order dynamics are real.)
    (iii) Do PRESCRIPTIVE and DIAGNOSTIC co-vary?
         (If both pass or both fail together: one may subsume the other.
          If they dissociate: all three variants are genuinely needed.)

  Scientific meaning:
    PASS (all three irreducible -- Q-024 primary hypothesis):
      => PRESCRIPTIVE harm_rate < DESCRIPTIVE harm_rate (C1).
         DIAGNOSTIC TE > THRESH_TE_BASELINE (C2, higher-order dynamics real).
         Dissociation confirmed: PRESCRIPTIVE converges (C3 mono low) while
         DIAGNOSTIC confirms nth-order dynamics persist (C4 TE > baseline).
         Q-024 answer: all three variants are needed.
    PARTIAL_PRESCRIPTIVE_DOMINATES:
      => C1 passes (PRESCRIPTIVE better), C2 fails (TE <= baseline).
         Diagnostic is redundant; prescriptive subsumes descriptive and diagnostic.
    PARTIAL_DESCRIPTIVE_SUFFICIENT:
      => C1 fails (no PRESCRIPTIVE benefit), C2 fails (TE <= baseline).
         Descriptive attractor alone is sufficient; neither constraint adds value.
    PARTIAL_DIAGNOSTIC_ONLY:
      => C1 fails (PRESCRIPTIVE no better), C2 passes (TE > baseline).
         Higher-order dynamics exist but prescriptive Lyapunov does not help;
         diagnostic is the only informative test type.
    FAIL:
      => Fewer than THRESH_MIN_HARM_EVENTS per condition per seed (data quality).

  Key metrics:
    1. harm_rate_final: mean harm rate in last MEASUREMENT_EPISODES (attractor)
    2. harm_rate_early: mean harm rate in first MEASUREMENT_EPISODES (baseline)
    3. monotone_violation_rate: fraction of episodes where harm > running_min +
       MONO_MARGIN (PRESCRIPTIVE condition only; directly tests Lyapunov proxy)
    4. te_k1: transfer entropy at lag k=1 (DIAGNOSTIC: harm_t-1 -> harm_t)
    5. te_k5: transfer entropy at lag k=5
    6. te_k10: transfer entropy at lag k=10
    7. n_harm_events: total harm events in measurement window (data quality)
    8. harm_delta: harm_rate_early - harm_rate_final (improvement)

Pre-registered thresholds
--------------------------
C1: PRESCRIPTIVE harm_rate_final <= DESCRIPTIVE harm_rate_final - THRESH_LYAP_BENEFIT
    (both seeds). Lyapunov constraint produces measurably lower final harm than
    pure attractor tracking -- prescriptive adds value over descriptive.
    THRESH_LYAP_BENEFIT = 0.02

C2: DIAGNOSTIC te_k1 > THRESH_TE_BASELINE (both seeds).
    Transfer entropy at lag-1 exceeds baseline -- harm dynamics are non-Markovian.
    This means prescriptive (1st-order convergence proof) cannot fully characterize
    the system; diagnostic is irreducible.
    THRESH_TE_BASELINE = 0.01

C3: PRESCRIPTIVE monotone_violation_rate < THRESH_MONO (both seeds).
    Lyapunov proxy is functioning: < THRESH_MONO fraction of episodes violate
    monotone decrease.
    THRESH_MONO = 0.30

C4: DIAGNOSTIC harm_rate_final <= DESCRIPTIVE harm_rate_final + THRESH_DIAG_PARITY
    (both seeds). DIAGNOSTIC (no Lyapunov constraint) converges to similar final
    harm as DESCRIPTIVE -- confirming the diagnostic measurement does not itself
    affect convergence quality.
    THRESH_DIAG_PARITY = 0.02

C5: n_harm_events >= THRESH_MIN_HARM_EVENTS per condition per seed (data quality).
    THRESH_MIN_HARM_EVENTS = 20

PASS:                              C1 + C2 + C3 + C4 + C5
PARTIAL_PRESCRIPTIVE_DOMINATES:   C1 + C3 + C5 + NOT C2
PARTIAL_DESCRIPTIVE_SUFFICIENT:   NOT C1 + NOT C2 + C5
PARTIAL_DIAGNOSTIC_ONLY:          NOT C1 + C2 + C5
FAIL:                              NOT C5

Conditions
----------
Shared architecture (each agent):
  World encoder:  Linear(world_obs_dim, world_dim) + LayerNorm -> z_world
  Self encoder:   Linear(self_obs_dim, self_dim) + LayerNorm -> z_self
  Harm evaluator: Linear(world_dim + self_dim, 1) -> harm estimate
  Policy:         Linear(world_dim + self_dim, hidden) -> ReLU -> Linear(hidden, action_dim)
  World predictor: Linear(world_dim + action_dim, world_dim)

DESCRIPTIVE:
  Standard training. Harm loss = MSE(harm_eval, harm_signal). No additional constraint.
  Measurement: harm_rate_final (where does the system end up?).

PRESCRIPTIVE:
  Same architecture. Additional Lyapunov loss:
    lyap_loss = LYAP_WEIGHT * relu(current_ep_harm - running_min_harm + MONO_MARGIN)
  where running_min_harm = min harm_rate seen so far over training.
  This penalises regressing above the best harm rate achieved (+ margin).
  Enforces a soft Lyapunov V(harm) -> must not increase.

DIAGNOSTIC:
  Same as DESCRIPTIVE (no Lyapunov constraint).
  Additional measurement: transfer entropy TE(harm_t -> harm_t+k).
  TE estimated via binned mutual information:
    TE(X->Y) = MI(Y_t; X_{t-k} | Y_{t-1}) approx MI(Y_t, X_{t-k}) - MI(Y_{t-1}, X_{t-k})
  Simplified estimator using discretized harm values (0/1 events) and empirical frequencies.

Seeds:   [42, 123]
Env:     CausalGridWorldV2 size=8, num_hazards=5, num_resources=2,
         hazard_harm=0.08, env_drift_interval=6, env_drift_prob=0.30
         (moderate harm density; drift ensures some non-Markovian dynamics)
Protocol: TOTAL_EPISODES=500
          MEASUREMENT_EPISODES=125 (last and first quartiles)
          STEPS_PER_EPISODE=200
          TE_LAG_STEPS=[1, 5, 10]
Estimated runtime:
  3 conditions x 2 seeds x 500 eps x 0.10 min/ep = 300 min Mac
  (+ 10% overhead for TE computation) => ~330 min Mac
  Queue estimate: 330 min
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
from ree_core.utils.config import REEConfig


EXPERIMENT_TYPE = "v3_exq_161_q024_trajectory_representation_triple"
CLAIM_IDS = ["Q-024"]

# ---------------------------------------------------------------------------
# Pre-registered thresholds
# ---------------------------------------------------------------------------
THRESH_LYAP_BENEFIT       = 0.02    # C1: PRESCRIPTIVE harm_rate < DESCRIPTIVE - this
THRESH_TE_BASELINE        = 0.01    # C2: DIAGNOSTIC te_k1 > this (non-Markovian dynamics)
THRESH_MONO               = 0.30    # C3: PRESCRIPTIVE monotone_violation_rate < this
THRESH_DIAG_PARITY        = 0.02    # C4: DIAGNOSTIC harm_rate within this of DESCRIPTIVE
THRESH_MIN_HARM_EVENTS    = 20      # C5: minimum harm events per condition per seed

# Lyapunov constraint parameters (PRESCRIPTIVE condition)
LYAP_WEIGHT  = 0.50    # weight on Lyapunov penalisation loss
MONO_MARGIN  = 0.005   # tolerance margin for Lyapunov violation detection

# Transfer entropy parameters (DIAGNOSTIC condition)
TE_LAG_STEPS = [1, 5, 10]   # lags for TE estimation

# Protocol constants
TOTAL_EPISODES       = 500
MEASUREMENT_EPISODES = 125   # last quartile (and first quartile for early baseline)
STEPS_PER_EPISODE    = 200
LR                   = 3e-4
ENT_BONUS            = 5e-3

SEEDS      = [42, 123]
CONDITIONS = ["DESCRIPTIVE", "PRESCRIPTIVE", "DIAGNOSTIC"]

# Model constants
WORLD_OBS_DIM = 250   # CausalGridWorldV2 size=8 world_state dim
SELF_OBS_DIM  = 12    # CausalGridWorldV2 body_state dim
ACTION_DIM    = 5
WORLD_DIM     = 32
SELF_DIM      = 16
HIDDEN_DIM    = 64


# ---------------------------------------------------------------------------
# Models
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


class WorldPredictor(nn.Module):
    """Lightweight E2 analog: (z_world, action_onehot) -> z_world_pred."""
    def __init__(self, world_dim: int, action_dim: int):
        super().__init__()
        self.fc = nn.Linear(world_dim + action_dim, world_dim)

    def forward(self, z_world: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.fc(torch.cat([z_world, action], dim=-1))


class HarmEvaluator(nn.Module):
    """E3 analog: (z_world, z_self) -> scalar harm estimate."""
    def __init__(self, world_dim: int, self_dim: int):
        super().__init__()
        self.fc = nn.Linear(world_dim + self_dim, 1)

    def forward(self, z_world: torch.Tensor, z_self: torch.Tensor) -> torch.Tensor:
        return self.fc(torch.cat([z_world, z_self], dim=-1))


class Policy(nn.Module):
    def __init__(self, world_dim: int, self_dim: int, action_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(world_dim + self_dim, HIDDEN_DIM)
        self.fc2 = nn.Linear(HIDDEN_DIM, action_dim)

    def forward(self, z_world: torch.Tensor, z_self: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.relu(self.fc1(torch.cat([z_world, z_self], dim=-1))))


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _get_obs_tensor(obs_dict: dict, key: str, fallback_dim: int) -> torch.Tensor:
    raw = obs_dict.get(key)
    if raw is None:
        return torch.zeros(fallback_dim)
    t = torch.tensor(raw, dtype=torch.float32).flatten()
    if t.shape[0] < fallback_dim:
        t = F.pad(t, (0, fallback_dim - t.shape[0]))
    elif t.shape[0] > fallback_dim:
        t = t[:fallback_dim]
    return t


def _action_one_hot(action_idx: int, action_dim: int) -> torch.Tensor:
    a = torch.zeros(action_dim)
    a[action_idx] = 1.0
    return a


def _mean(lst: List[float]) -> float:
    return float(sum(lst) / len(lst)) if lst else 0.0


def _compute_transfer_entropy(harm_series: List[float], lag: int) -> float:
    """
    Estimate TE(harm_{t-lag} -> harm_t) via binned mutual information.
    Uses binary discretization (harm_event = harm > 0).
    TE(X->Y) approx MI(Y_t; X_{t-lag}) using marginal / joint entropy estimator.
    Simplified estimator appropriate for sparse binary series.

    Returns estimated TE value (float >= 0).
    """
    n = len(harm_series)
    if n < lag + 10:
        return 0.0

    # Binarize: 1 if harm > 0, else 0
    binary = [1 if h > 0.0 else 0 for h in harm_series]

    # Build (X_{t-lag}, Y_t) pairs
    pairs: List[Tuple[int, int]] = []
    for t in range(lag, n):
        x = binary[t - lag]
        y = binary[t]
        pairs.append((x, y))

    if not pairs:
        return 0.0

    total = len(pairs)

    # Joint counts P(X_{t-lag}, Y_t)
    joint_counts: Dict[Tuple[int, int], int] = {}
    x_counts: Dict[int, int] = {}
    y_counts: Dict[int, int] = {}

    for (x, y) in pairs:
        joint_counts[(x, y)] = joint_counts.get((x, y), 0) + 1
        x_counts[x] = x_counts.get(x, 0) + 1
        y_counts[y] = y_counts.get(y, 0) + 1

    # MI(X; Y) = sum p(x,y) log(p(x,y) / (p(x)*p(y)))
    mi = 0.0
    for (x, y), cnt in joint_counts.items():
        p_xy = cnt / total
        p_x  = x_counts[x] / total
        p_y  = y_counts[y] / total
        if p_xy > 0 and p_x > 0 and p_y > 0:
            mi += p_xy * math.log(p_xy / (p_x * p_y + 1e-12) + 1e-12)

    return float(max(mi, 0.0))


# ---------------------------------------------------------------------------
# Core experiment runner
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
    torch.manual_seed(seed)
    random.seed(seed)

    env = CausalGridWorldV2(
        size=8,
        num_hazards=5,
        num_resources=2,
        hazard_harm=0.08,
        env_drift_interval=6,
        env_drift_prob=0.30,
        seed=seed,
    )

    world_enc  = WorldEncoder(WORLD_OBS_DIM, WORLD_DIM)
    self_enc   = SelfEncoder(SELF_OBS_DIM, SELF_DIM)
    world_pred = WorldPredictor(WORLD_DIM, ACTION_DIM)
    harm_eval  = HarmEvaluator(WORLD_DIM, SELF_DIM)
    policy     = Policy(WORLD_DIM, SELF_DIM, ACTION_DIM)

    all_params = (
        list(world_enc.parameters())
        + list(self_enc.parameters())
        + list(world_pred.parameters())
        + list(harm_eval.parameters())
        + list(policy.parameters())
    )
    optimizer = optim.Adam(all_params, lr=lr)

    if dry_run:
        total_episodes = 2
        measurement_episodes = 1

    measurement_start = total_episodes - measurement_episodes
    early_end         = measurement_episodes

    # Measurement buffers
    harm_vals_final:  List[float] = []
    harm_vals_early:  List[float] = []
    n_harm_events_final: int = 0

    # Full harm series for TE computation (DIAGNOSTIC condition)
    harm_series_all: List[float] = []

    # Lyapunov tracking (PRESCRIPTIVE condition)
    running_min_harm: float = float("inf")
    ep_harm_rates:    List[float] = []
    mono_violations:  int = 0

    # Per-episode harm tracking (for Lyapunov running min and violation count)
    ep_harms: List[float] = []

    z_world_prev: torch.Tensor = torch.zeros(WORLD_DIM)
    have_prev = False

    _, obs_dict = env.reset()

    for ep in range(total_episodes):
        in_final_measurement = ep >= measurement_start
        in_early_measurement = ep < early_end

        ep_harms = []

        for _step in range(steps_per_episode):
            obs_world = _get_obs_tensor(obs_dict, "world_state", WORLD_OBS_DIM)
            obs_self  = _get_obs_tensor(obs_dict, "body_state",  SELF_OBS_DIM)

            # Select action
            with torch.no_grad():
                z_world = world_enc(obs_world)
                z_self  = self_enc(obs_self)
                logits  = policy(z_world, z_self)
                probs   = F.softmax(logits, dim=-1)
                action_idx = int(torch.multinomial(probs, 1).item())

            action_tensor = _action_one_hot(action_idx, ACTION_DIM)
            _, _, done, _, obs_dict_next = env.step(action_tensor.unsqueeze(0))

            harm_signal = float(obs_dict_next.get("harm_signal", 0.0))
            ep_harms.append(harm_signal)

            # DIAGNOSTIC: collect full harm series
            if condition == "DIAGNOSTIC":
                harm_series_all.append(harm_signal)

            if in_final_measurement:
                harm_vals_final.append(harm_signal)
                if harm_signal > 0.0:
                    n_harm_events_final += 1
            if in_early_measurement:
                harm_vals_early.append(harm_signal)

            # Training step
            obs_world_next = _get_obs_tensor(obs_dict_next, "world_state", WORLD_OBS_DIM)
            obs_self_next  = _get_obs_tensor(obs_dict_next, "body_state",  SELF_OBS_DIM)

            z_world_t = world_enc(obs_world)
            z_self_t  = self_enc(obs_self)
            z_world_n = world_enc(obs_world_next).detach()

            harm_target = torch.tensor([harm_signal], dtype=torch.float32)
            harm_pred   = harm_eval(z_world_t, z_self_t)
            harm_eval_l = F.mse_loss(harm_pred, harm_target)

            if have_prev:
                z_pred       = world_pred(z_world_prev, action_tensor)
                world_pred_l = F.mse_loss(z_pred, z_world_n)
            else:
                world_pred_l = torch.tensor(0.0)

            logits_p  = policy(z_world_t.detach(), z_self_t.detach())
            probs_p   = F.softmax(logits_p, dim=-1)
            log_probs = F.log_softmax(logits_p, dim=-1)
            entropy   = -(probs_p * log_probs).sum()

            # Lyapunov penalty (PRESCRIPTIVE only)
            lyap_loss = torch.tensor(0.0)
            if condition == "PRESCRIPTIVE" and running_min_harm < float("inf"):
                ep_harm_mean = _mean(ep_harms) if ep_harms else harm_signal
                violation = max(0.0, ep_harm_mean - running_min_harm - MONO_MARGIN)
                lyap_loss = LYAP_WEIGHT * torch.tensor(violation, dtype=torch.float32)

            loss = harm_eval_l + world_pred_l + lyap_loss - ENT_BONUS * entropy

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(all_params, 1.0)
            optimizer.step()

            z_world_prev = z_world_t.detach().clone()
            have_prev = True

            obs_dict = obs_dict_next
            if done:
                _, obs_dict = env.reset()
                z_world_prev = torch.zeros(WORLD_DIM)
                have_prev = False

        # End of episode: update Lyapunov running min and count violations
        if ep_harms:
            ep_harm_rate = _mean(ep_harms)
            ep_harm_rates.append(ep_harm_rate)
            if condition == "PRESCRIPTIVE":
                if ep_harm_rate < running_min_harm:
                    running_min_harm = ep_harm_rate
                elif ep_harm_rate > running_min_harm + MONO_MARGIN and running_min_harm < float("inf"):
                    mono_violations += 1

    # ---------------------------------------------------------------------------
    # Compute metrics
    # ---------------------------------------------------------------------------
    harm_rate_final = _mean(harm_vals_final)
    harm_rate_early = _mean(harm_vals_early)
    harm_delta      = harm_rate_early - harm_rate_final

    # PRESCRIPTIVE: monotone violation rate
    monotone_violation_rate = 0.0
    if condition == "PRESCRIPTIVE" and ep_harm_rates:
        monotone_violation_rate = mono_violations / len(ep_harm_rates)

    # DIAGNOSTIC: transfer entropy at each lag
    te_k1  = 0.0
    te_k5  = 0.0
    te_k10 = 0.0
    if condition == "DIAGNOSTIC" and harm_series_all:
        te_k1  = _compute_transfer_entropy(harm_series_all, lag=1)
        te_k5  = _compute_transfer_entropy(harm_series_all, lag=5)
        te_k10 = _compute_transfer_entropy(harm_series_all, lag=10)

    return {
        "condition":                condition,
        "seed":                     seed,
        "harm_rate_final":          harm_rate_final,
        "harm_rate_early":          harm_rate_early,
        "harm_delta":               harm_delta,
        "n_harm_events_final":      n_harm_events_final,
        "n_measurement_steps":      len(harm_vals_final),
        "monotone_violation_rate":  monotone_violation_rate,
        "te_k1":                    te_k1,
        "te_k5":                    te_k5,
        "te_k10":                   te_k10,
        "n_episodes":               len(ep_harm_rates),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(dry_run: bool = False) -> None:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id    = f"{EXPERIMENT_TYPE}_{timestamp}_v3"

    print(f"[EXQ-161] Starting {EXPERIMENT_TYPE}")
    print(f"[EXQ-161] run_id = {run_id}")
    print(f"[EXQ-161] dry_run = {dry_run}")
    print(f"[EXQ-161] conditions = {CONDITIONS}, seeds = {SEEDS}")

    cells: List[Dict] = []
    for seed in SEEDS:
        for condition in CONDITIONS:
            print(f"[EXQ-161] Running condition={condition} seed={seed}")
            result = _run_condition(
                seed                 = seed,
                condition            = condition,
                total_episodes       = TOTAL_EPISODES,
                measurement_episodes = MEASUREMENT_EPISODES,
                steps_per_episode    = STEPS_PER_EPISODE,
                lr                   = LR,
                dry_run              = dry_run,
            )
            print(
                f"[EXQ-161]   harm_final={result['harm_rate_final']:.4f}  "
                f"harm_early={result['harm_rate_early']:.4f}  "
                f"delta={result['harm_delta']:.4f}  "
                f"mono_viol={result['monotone_violation_rate']:.4f}  "
                f"te_k1={result['te_k1']:.5f}  "
                f"n_harm={result['n_harm_events_final']}"
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

    # C1: PRESCRIPTIVE harm_rate_final <= DESCRIPTIVE harm_rate_final - THRESH_LYAP_BENEFIT
    #     (both seeds) -- Lyapunov constraint improves convergence quality
    c1_pass = all(
        _cell("PRESCRIPTIVE", s)["harm_rate_final"]
        <= _cell("DESCRIPTIVE", s)["harm_rate_final"] - THRESH_LYAP_BENEFIT
        for s in SEEDS
    )

    # C2: DIAGNOSTIC te_k1 > THRESH_TE_BASELINE (both seeds) -- non-Markovian dynamics
    c2_pass = all(
        _cell("DIAGNOSTIC", s)["te_k1"] > THRESH_TE_BASELINE
        for s in SEEDS
    )

    # C3: PRESCRIPTIVE monotone_violation_rate < THRESH_MONO (both seeds)
    c3_pass = all(
        _cell("PRESCRIPTIVE", s)["monotone_violation_rate"] < THRESH_MONO
        for s in SEEDS
    )

    # C4: DIAGNOSTIC harm_rate_final <= DESCRIPTIVE harm_rate_final + THRESH_DIAG_PARITY
    #     (both seeds) -- diagnostic condition converges comparably to descriptive
    c4_pass = all(
        _cell("DIAGNOSTIC", s)["harm_rate_final"]
        <= _cell("DESCRIPTIVE", s)["harm_rate_final"] + THRESH_DIAG_PARITY
        for s in SEEDS
    )

    # C5: n_harm_events_final >= THRESH_MIN_HARM_EVENTS per condition per seed
    c5_pass = all(
        _cell(cond, s)["n_harm_events_final"] >= THRESH_MIN_HARM_EVENTS
        for cond in CONDITIONS
        for s in SEEDS
    )

    # Outcome logic
    if c1_pass and c2_pass and c3_pass and c4_pass and c5_pass:
        outcome = "PASS"
    elif not c5_pass:
        outcome = "FAIL"
    elif c1_pass and c3_pass and not c2_pass and c5_pass:
        outcome = "PARTIAL_PRESCRIPTIVE_DOMINATES"
    elif not c1_pass and not c2_pass and c5_pass:
        outcome = "PARTIAL_DESCRIPTIVE_SUFFICIENT"
    elif not c1_pass and c2_pass and c5_pass:
        outcome = "PARTIAL_DIAGNOSTIC_ONLY"
    else:
        outcome = "PARTIAL_INCONCLUSIVE"

    print(f"[EXQ-161] C1 (PRESCRIPTIVE harm <= DESCRIPTIVE - {THRESH_LYAP_BENEFIT}): {c1_pass}")
    print(f"[EXQ-161] C2 (DIAGNOSTIC te_k1 > {THRESH_TE_BASELINE}): {c2_pass}")
    print(f"[EXQ-161] C3 (PRESCRIPTIVE mono_viol < {THRESH_MONO}): {c3_pass}")
    print(f"[EXQ-161] C4 (DIAGNOSTIC harm_final within {THRESH_DIAG_PARITY} of DESCRIPTIVE): {c4_pass}")
    print(f"[EXQ-161] C5 (n_harm_events >= {THRESH_MIN_HARM_EVENTS}): {c5_pass}")
    print(f"[EXQ-161] Outcome: {outcome}")

    # ---------------------------------------------------------------------------
    # Summary table and pairwise deltas
    # ---------------------------------------------------------------------------
    summary_rows = []
    for cond in CONDITIONS:
        for s in SEEDS:
            cell = _cell(cond, s)
            summary_rows.append({
                "condition":               cond,
                "seed":                    s,
                "harm_rate_final":         round(cell["harm_rate_final"], 5),
                "harm_rate_early":         round(cell["harm_rate_early"], 5),
                "harm_delta":              round(cell["harm_delta"], 5),
                "n_harm_events_final":     cell["n_harm_events_final"],
                "n_measurement_steps":     cell["n_measurement_steps"],
                "monotone_violation_rate": round(cell["monotone_violation_rate"], 5),
                "te_k1":                   round(cell["te_k1"], 6),
                "te_k5":                   round(cell["te_k5"], 6),
                "te_k10":                  round(cell["te_k10"], 6),
            })

    pairwise_deltas = []
    for s in SEEDS:
        desc  = _cell("DESCRIPTIVE",  s)
        presc = _cell("PRESCRIPTIVE", s)
        diag  = _cell("DIAGNOSTIC",   s)
        pairwise_deltas.append({
            "seed":                         s,
            "harm_final_descriptive":       round(desc["harm_rate_final"],  5),
            "harm_final_prescriptive":      round(presc["harm_rate_final"], 5),
            "harm_final_diagnostic":        round(diag["harm_rate_final"],  5),
            "prescriptive_vs_descriptive":  round(
                presc["harm_rate_final"] - desc["harm_rate_final"], 5
            ),
            "diagnostic_vs_descriptive":    round(
                diag["harm_rate_final"] - desc["harm_rate_final"], 5
            ),
            "lyap_benefit_margin":          round(
                desc["harm_rate_final"] - presc["harm_rate_final"]
                - THRESH_LYAP_BENEFIT, 5
            ),
            "monotone_violation_rate_presc": round(presc["monotone_violation_rate"], 5),
            "te_k1_diag":                   round(diag["te_k1"], 6),
            "te_k5_diag":                   round(diag["te_k5"], 6),
            "te_k10_diag":                  round(diag["te_k10"], 6),
        })

    # ---------------------------------------------------------------------------
    # Write output JSON
    # ---------------------------------------------------------------------------
    output_dir = (
        Path(__file__).resolve().parents[1].parent
        / "REE_assembly" / "evidence" / "experiments"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{run_id}.json"

    pack = {
        "run_id":             run_id,
        "experiment_type":    EXPERIMENT_TYPE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "claim_ids":          CLAIM_IDS,
        "timestamp_utc":      timestamp,
        "dry_run":            dry_run,
        "config": {
            "total_episodes":          TOTAL_EPISODES,
            "measurement_episodes":    MEASUREMENT_EPISODES,
            "steps_per_episode":       STEPS_PER_EPISODE,
            "lr":                      LR,
            "ent_bonus":               ENT_BONUS,
            "seeds":                   SEEDS,
            "conditions":              CONDITIONS,
            "lyap_weight":             LYAP_WEIGHT,
            "mono_margin":             MONO_MARGIN,
            "te_lag_steps":            TE_LAG_STEPS,
        },
        "thresholds": {
            "THRESH_LYAP_BENEFIT":    THRESH_LYAP_BENEFIT,
            "THRESH_TE_BASELINE":     THRESH_TE_BASELINE,
            "THRESH_MONO":            THRESH_MONO,
            "THRESH_DIAG_PARITY":     THRESH_DIAG_PARITY,
            "THRESH_MIN_HARM_EVENTS": THRESH_MIN_HARM_EVENTS,
        },
        "criteria": {
            "C1_prescriptive_benefit":         c1_pass,
            "C2_diagnostic_te_nonmarkovian":   c2_pass,
            "C3_prescriptive_monotone":        c3_pass,
            "C4_diagnostic_harm_parity":       c4_pass,
            "C5_harm_events_sufficient":       c5_pass,
        },
        "outcome":   outcome,
        "evidence_class":     "experimental",
        "evidence_direction": (
            "supports" if outcome == "PASS"
            else "weakens" if outcome in (
                "PARTIAL_PRESCRIPTIVE_DOMINATES",
                "PARTIAL_DESCRIPTIVE_SUFFICIENT",
            )
            else "mixed"
        ),
        "summary": (
            f"Outcome={outcome}. Q-024: Are DESCRIPTIVE, PRESCRIPTIVE, and DIAGNOSTIC "
            f"test-type variants for trajectory-integral representation all needed, or "
            f"does one subsume the others? Three conditions: DESCRIPTIVE (pure attractor "
            f"tracking), PRESCRIPTIVE (Lyapunov-proxy constraint enforcing monotone harm "
            f"decrease), DIAGNOSTIC (transfer-entropy measurement of higher-order dynamics). "
            f"C1 PRESCRIPTIVE benefit >= {THRESH_LYAP_BENEFIT}: {c1_pass}. "
            f"C2 DIAGNOSTIC te_k1 > {THRESH_TE_BASELINE} (non-Markovian): {c2_pass}. "
            f"C3 PRESCRIPTIVE mono_viol < {THRESH_MONO}: {c3_pass}. "
            f"C4 DIAGNOSTIC harm_final within {THRESH_DIAG_PARITY} of DESCRIPTIVE: {c4_pass}. "
            f"C5 harm events sufficient: {c5_pass}. "
            f"PASS => PRESCRIPTIVE adds value over DESCRIPTIVE (C1); higher-order dynamics "
            f"are real (C2); prescriptive Lyapunov is functioning (C3); diagnostic does not "
            f"disrupt convergence (C4). All three variants irreducibly needed; Q-024 "
            f"primary hypothesis supported. "
            f"PARTIAL_PRESCRIPTIVE_DOMINATES => C1+C3 pass, C2 fails: Lyapunov adds value "
            f"but diagnostic is redundant; prescriptive subsumes descriptive. "
            f"PARTIAL_DESCRIPTIVE_SUFFICIENT => neither C1 nor C2: pure attractor tracking "
            f"sufficient; no benefit to Lyapunov constraint or TE measurement. "
            f"PARTIAL_DIAGNOSTIC_ONLY => C2 but not C1: higher-order dynamics exist but "
            f"Lyapunov constraint does not help; diagnostic irreducible. "
            f"FAIL => insufficient data."
        ),
        "cells":           cells,
        "summary_table":   summary_rows,
        "pairwise_deltas": pairwise_deltas,
    }

    with open(output_path, "w") as fh:
        json.dump(pack, fh, indent=2)

    print(f"[EXQ-161] Output written to {output_path}")
    print(f"[EXQ-161] DONE -- outcome={outcome}")


if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv
    main(dry_run=dry_run)
