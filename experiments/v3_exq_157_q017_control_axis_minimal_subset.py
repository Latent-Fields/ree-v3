#!/opt/local/bin/python3
"""
V3-EXQ-157 -- Q-017: What is the minimal orthogonal control-axis subset that
              preserves observed regime separations?

Claim:    Q-017
Proposal: EXP-0108 (EVB-0081)

Q-017 asks:
  "What is the minimal orthogonal control-axis subset that preserves observed
  regime separations?"

  Background (MECH-063):
    The control plane retains orthogonal tonic/phasic axes rather than
    collapsing into one scalar (ARC-005). The axes represent functionally
    distinct precision-routing signals -- tonic baseline drive (BG direct-
    pathway analog, slow-moving), phasic surprise/interrupt (LC-NE analog,
    fast), and affective valence (habenula-ACC analog). MECH-055 asserts
    these are separable channels (hedonic tone, valence, signed PE).

    "Regime separation" is the empirically observed fact that the agent
    operates in qualitatively distinct behavioural regimes -- exploration,
    exploitation, avoidance -- each associated with a characteristic
    control-axis signature. A minimal orthogonal subset is the smallest
    set of axes that is sufficient to reproduce these regime separations
    in behaviour.

  Q-017 asks: do all three axes contribute to regime separation, or can
    one or more be removed without losing the separations?

  This experiment compares three conditions across matched seeds:

  Condition A: FULL_AXES (3-dimensional control vector)
    Tonic axis:    z_tonic   = running mean of harm_signal (EMA tau=50 steps)
                              -- slow baseline: ambient danger level
    Phasic axis:   z_phasic  = |harm_signal - z_tonic| (surprise relative to baseline)
                              -- fast event-driven interrupt
    Affective axis: z_affect = harm_signal - benefit_signal (signed valence PE)
                              -- signed error: combines harm and benefit channels
    Control vector: c = [z_tonic, z_phasic, z_affect]
    Policy: Linear(z_world + z_self + c, action_dim) + E3 harm evaluator
    Rationale: full orthogonal set as specified by MECH-063/MECH-055.
    Prediction: regime separation is maximal because each axis provides
    non-redundant information.

  Condition B: COLLAPSED_SCALAR (1-dimensional control)
    Single scalar:  c = harm_signal (raw harm signal, no decomposition)
    Policy: Linear(z_world + z_self + [c], action_dim) + E3 harm evaluator
    Rationale: null hypothesis -- all information is preserved in the raw
    harm signal and the axes are redundant.
    Prediction: regime separation is reduced because the axes are correlated
    with raw harm at the timescale of this environment.

  Condition C: MINIMAL_SUBSET (2-dimensional -- tonic + phasic, no affect)
    Tonic axis:  z_tonic  (as above)
    Phasic axis: z_phasic (as above)
    No affective axis.
    Control vector: c = [z_tonic, z_phasic]
    Rationale: benefit signals are sparse in CausalGridWorldV2; the affective
    axis (harm - benefit) may contribute noise rather than signal. If
    FULL_AXES ~ MINIMAL_SUBSET, the affective axis is dispensable at this
    scale.
    Prediction: if MECH-055 is correct, removing the affective axis
    degrades regime separation relative to FULL_AXES. If Q-017 resolves
    to "two axes are sufficient", MINIMAL_SUBSET ~ FULL_AXES.

  Discriminative question:
    (i)  Does FULL_AXES separate regimes better than COLLAPSED_SCALAR?
         (Tests whether orthogonal decomposition adds value at all.)
    (ii) Does removing the affective axis (MINIMAL_SUBSET) significantly
         degrade regime separation vs FULL_AXES?
         If NO: two orthogonal axes suffice -- Q-017 partially resolved.
         If YES: all three axes are load-bearing -- Q-017 unresolved.

  Scientific meaning:
    PASS:
      => FULL_AXES achieves best regime separation AND MINIMAL_SUBSET
         shows significant degradation relative to FULL_AXES. All three
         axes are needed. Q-017 partially resolved: the minimal set is
         {tonic, phasic, affect} (i.e., all three at V3 scale).
    PARTIAL_TWO_SUFFICIENT:
      => FULL_AXES best, but MINIMAL_SUBSET ~ FULL_AXES (regime_sep gap
         < THRESH_EQUIV). Two axes {tonic, phasic} are sufficient.
         Q-017 partially resolved: affective axis dispensable at V3 scale.
    PARTIAL_COLLAPSE_ADEQUATE:
      => COLLAPSED_SCALAR ~ FULL_AXES. Raw harm signal carries sufficient
         regime information at V3 scale; orthogonal decomposition adds no
         measurable benefit. Q-017 remains open at larger scale.
    FAIL:
      => FULL_AXES worse than COLLAPSED_SCALAR or insufficient regime
         events for reliable estimation. Implementation or data problem.

  Regime separation is measured as inter-regime distance: the mean absolute
  difference in behaviour metrics (harm_rate, action_entropy) across regime
  clusters. Regimes are inferred as rolling windows where:
    - AVOIDANCE:     harm_rate in window > REGIME_HARM_THRESH
    - EXPLOITATION:  resource_count in window > 0 and harm_rate <= REGIME_HARM_THRESH
    - EXPLORATION:   harm_rate <= REGIME_HARM_THRESH and resource_count == 0
  Regime separation score = mean |harm_rate_avoidance - harm_rate_exploration|
  + mean |action_entropy_exploitation - action_entropy_exploration|.
  (A single scalar composite that is higher when the regime clusters are
  more distinct.)

  The key metrics:
    1. regime_sep (regime separation composite score) -- primary
    2. harm_rate (harm contacts per step in last quartile) -- sanity
    3. action_entropy (per-step policy entropy in measurement window)
    4. n_avoidance_windows / n_exploitation_windows / n_exploration_windows
       -- data quality: at least THRESH_MIN_REGIME_WINDOWS of each
    5. axis_variance (per-axis variance over measurement window,
       FULL_AXES only) -- confirms axes are non-degenerate

Pre-registered thresholds
--------------------------
C1: FULL_AXES regime_sep > COLLAPSED_SCALAR regime_sep + THRESH_SEP_MARGIN (both seeds).
    (Orthogonal decomposition improves regime separation over raw harm scalar.)

C2: FULL_AXES regime_sep > THRESH_SEP_ABS_MIN (both seeds).
    (Full-axes regime separation is above floor -- enough signal to distinguish.)

C3: Each of n_avoidance_windows, n_exploitation_windows, n_exploration_windows
    >= THRESH_MIN_REGIME_WINDOWS per condition per seed.
    (Data quality gate: all three regimes were observed.)

C4: FULL_AXES axis_variance[tonic] > THRESH_AXIS_VAR_MIN AND
    FULL_AXES axis_variance[phasic] > THRESH_AXIS_VAR_MIN AND
    FULL_AXES axis_variance[affect] > THRESH_AXIS_VAR_MIN (both seeds).
    (All three axes are non-degenerate -- non-constant signals.)

C5: MINIMAL_SUBSET regime_sep direction relative to FULL_AXES is seed-consistent.
    (Used to determine whether PARTIAL_TWO_SUFFICIENT applies.)

PASS:        C1 + C2 + C3 + C4 + C5
             AND |FULL_AXES - MINIMAL_SUBSET| > THRESH_SEP_MARGIN (C5-strict).
PARTIAL_TWO_SUFFICIENT: C1 + C2 + C3 + C4
             AND |FULL_AXES - MINIMAL_SUBSET| <= THRESH_EQUIV (both seeds).
PARTIAL_COLLAPSE_ADEQUATE: NOT C1 (COLLAPSED_SCALAR ~ FULL_AXES).
FAIL:        NOT C3 OR NOT C2.

Conditions
----------
Shared architecture (all conditions):
  World encoder:  Linear(world_obs_dim, world_dim) + LayerNorm -> z_world
  Self encoder:   Linear(self_obs_dim, self_dim) + LayerNorm -> z_self
  Control MLP:    Linear(z_world + z_self + control_dim, hidden) -> ReLU
                  -> Linear(hidden, action_dim) (policy head)
  Harm evaluator: Linear(z_world + z_self, 1) (E3 analog)
  Benefit signal: resource_count from obs (binary: stepped on resource)

  Training:
    Loss = harm_eval_loss (MSE(harm_pred, harm_actual))
         + policy_entropy_bonus * ENT_BONUS (encourage exploration)
    Adam lr=LR, gradient clipping 1.0
    Policy: softmax over action logits -> action sampled from distribution

  Control vector construction:
    z_tonic = EMA(harm_signal, tau=TONIC_TAU)  -- updated after each step
    z_phasic = |harm_signal - z_tonic|          -- surprise
    z_affect = harm_signal - benefit_signal     -- signed valence PE

FULL_AXES:       control_dim=3, c = [z_tonic, z_phasic, z_affect]
COLLAPSED_SCALAR: control_dim=1, c = [harm_signal]
MINIMAL_SUBSET:  control_dim=2, c = [z_tonic, z_phasic]

Seeds:    [42, 123] (matched -- same env seed per condition)
Env:      CausalGridWorldV2 size=8, num_hazards=4, num_resources=3,
          hazard_harm=0.1, env_drift_interval=10, env_drift_prob=0.3
          (hazard density chosen to produce all three regimes reliably)
Protocol: TOTAL_EPISODES=400
          MEASUREMENT_EPISODES=100 (last 100 episodes)
          STEPS_PER_EPISODE=200
          WINDOW_SIZE=20 (steps per regime window)
Estimated runtime:
  3 conditions x 2 seeds x 400 eps x 0.10 min/ep = ~240 min Mac
  (+10% overhead) => ~264 min Mac
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


EXPERIMENT_TYPE = "v3_exq_157_q017_control_axis_minimal_subset"
CLAIM_IDS = ["Q-017"]

# ---------------------------------------------------------------------------
# Pre-registered thresholds
# ---------------------------------------------------------------------------
THRESH_SEP_MARGIN        = 0.02   # C1: min improvement of FULL_AXES over COLLAPSED
THRESH_SEP_ABS_MIN       = 0.01   # C2: min regime_sep for FULL_AXES
THRESH_MIN_REGIME_WINDOWS = 5     # C3: min windows per regime type per condition
THRESH_AXIS_VAR_MIN      = 1e-5   # C4: min per-axis variance (non-degenerate)
THRESH_EQUIV             = 0.02   # C5/PARTIAL: equivalence band for FULL vs MINIMAL

# ---------------------------------------------------------------------------
# Protocol constants
# ---------------------------------------------------------------------------
TOTAL_EPISODES          = 400
MEASUREMENT_EPISODES    = 100    # last N episodes
STEPS_PER_EPISODE       = 200
WINDOW_SIZE             = 20     # steps per regime classification window
TONIC_TAU               = 50.0   # EMA decay for tonic axis (in steps)
ENT_BONUS               = 5e-3   # entropy bonus weight
LR                      = 3e-4

SEEDS      = [42, 123]
CONDITIONS = ["FULL_AXES", "COLLAPSED_SCALAR", "MINIMAL_SUBSET"]

# Regime classification thresholds
REGIME_HARM_THRESH = 0.03    # mean harm/step in window: above -> AVOIDANCE

# ---------------------------------------------------------------------------
# Model constants
# ---------------------------------------------------------------------------
WORLD_OBS_DIM = 250   # CausalGridWorldV2 size=8 world_state dim
SELF_OBS_DIM  = 12    # CausalGridWorldV2 body_state dim
ACTION_DIM    = 5
WORLD_DIM     = 32
SELF_DIM      = 16
HIDDEN_DIM    = 64


# ---------------------------------------------------------------------------
# Control dimension per condition
# ---------------------------------------------------------------------------
_CONTROL_DIM = {
    "FULL_AXES":        3,
    "COLLAPSED_SCALAR": 1,
    "MINIMAL_SUBSET":   2,
}


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


class ControlPolicy(nn.Module):
    """
    Policy network: (z_world, z_self, control_vec) -> action logits.
    MLP: Linear(input, HIDDEN_DIM) + ReLU + Linear(HIDDEN_DIM, ACTION_DIM).
    """
    def __init__(self, world_dim: int, self_dim: int, control_dim: int, action_dim: int):
        super().__init__()
        in_dim = world_dim + self_dim + control_dim
        self.fc1   = nn.Linear(in_dim, HIDDEN_DIM)
        self.fc2   = nn.Linear(HIDDEN_DIM, action_dim)

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

    def forward(self, z_world: torch.Tensor, z_self: torch.Tensor) -> torch.Tensor:
        return self.fc(torch.cat([z_world, z_self], dim=-1))


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _get_obs_tensor(obs_dict: dict, key: str, fallback_dim: int) -> torch.Tensor:
    """Safely extract a 1-D tensor from obs_dict, zero-padding or truncating."""
    raw = obs_dict.get(key)
    if raw is None:
        return torch.zeros(fallback_dim)
    t = torch.tensor(raw, dtype=torch.float32).flatten()
    if t.shape[0] < fallback_dim:
        t = F.pad(t, (0, fallback_dim - t.shape[0]))
    elif t.shape[0] > fallback_dim:
        t = t[:fallback_dim]
    return t


def _classify_regime(harm_vals: List[float], resource_vals: List[float]) -> str:
    """Classify a window of steps into one of three regimes."""
    mean_harm = float(sum(harm_vals) / len(harm_vals)) if harm_vals else 0.0
    total_resources = sum(resource_vals)
    if mean_harm > REGIME_HARM_THRESH:
        return "avoidance"
    elif total_resources > 0:
        return "exploitation"
    else:
        return "exploration"


def _action_entropy(action_probs: List[List[float]]) -> float:
    """Mean entropy over a list of softmax probability vectors."""
    if not action_probs:
        return 0.0
    entropies = []
    for probs in action_probs:
        h = -sum(p * math.log(p + 1e-9) for p in probs)
        entropies.append(h)
    return float(sum(entropies) / len(entropies))


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
    """
    Run one condition x seed cell.
    Returns per-cell metrics dict.
    """
    torch.manual_seed(seed)
    random.seed(seed)

    control_dim = _CONTROL_DIM[condition]

    env = CausalGridWorldV2(
        size=8,
        num_hazards=4,
        num_resources=3,
        hazard_harm=0.1,
        env_drift_interval=10,
        env_drift_prob=0.3,
        seed=seed,
    )

    world_enc  = WorldEncoder(WORLD_OBS_DIM, WORLD_DIM)
    self_enc   = SelfEncoder(SELF_OBS_DIM, SELF_DIM)
    policy     = ControlPolicy(WORLD_DIM, SELF_DIM, control_dim, ACTION_DIM)
    harm_eval  = HarmEvaluator(WORLD_DIM, SELF_DIM)

    params = (
        list(world_enc.parameters())
        + list(self_enc.parameters())
        + list(policy.parameters())
        + list(harm_eval.parameters())
    )
    optimizer = optim.Adam(params, lr=lr)

    measurement_start = total_episodes - measurement_episodes

    if dry_run:
        total_episodes    = 2
        measurement_start = 0

    # Tonic EMA state
    z_tonic_val: float = 0.0
    alpha_tonic = 1.0 / TONIC_TAU   # EMA update rate

    # Measurement buffers
    harm_vals_meas:         List[float]       = []
    resource_vals_meas:     List[float]       = []
    action_probs_meas:      List[List[float]] = []

    # Axis value buffers (FULL_AXES and MINIMAL_SUBSET)
    tonic_vals:   List[float] = []
    phasic_vals:  List[float] = []
    affect_vals:  List[float] = []

    # Regime classification windows
    window_harm:     List[float] = []
    window_resource: List[float] = []

    regime_harm_rates:    Dict[str, List[float]] = {
        "avoidance": [], "exploitation": [], "exploration": []
    }
    regime_action_ent:    Dict[str, List[float]] = {
        "avoidance": [], "exploitation": [], "exploration": []
    }
    regime_probs_window:  List[List[float]] = []  # action probs in current window
    regime_counts = {"avoidance": 0, "exploitation": 0, "exploration": 0}

    _, obs_dict = env.reset()

    total_steps = 0

    for ep in range(total_episodes):
        in_measurement = ep >= measurement_start

        for step in range(steps_per_episode):
            obs_world = _get_obs_tensor(obs_dict, "world_state", WORLD_OBS_DIM)
            obs_self  = _get_obs_tensor(obs_dict, "body_state", SELF_OBS_DIM)

            z_world = world_enc(obs_world)
            z_self  = self_enc(obs_self)

            # Build control vector
            harm_prev = float(obs_dict.get("harm_signal", 0.0))
            benefit_prev = float(obs_dict.get("resource_count", 0.0))

            z_tonic_val = z_tonic_val + alpha_tonic * (harm_prev - z_tonic_val)
            z_phasic_val = abs(harm_prev - z_tonic_val)
            z_affect_val = harm_prev - benefit_prev

            if condition == "FULL_AXES":
                c = torch.tensor([z_tonic_val, z_phasic_val, z_affect_val], dtype=torch.float32)
            elif condition == "COLLAPSED_SCALAR":
                c = torch.tensor([harm_prev], dtype=torch.float32)
            else:  # MINIMAL_SUBSET
                c = torch.tensor([z_tonic_val, z_phasic_val], dtype=torch.float32)

            # Policy forward
            logits = policy(z_world.detach(), z_self.detach(), c)
            probs  = F.softmax(logits, dim=-1)
            action_idx = torch.multinomial(probs.detach(), 1).item()
            action = torch.zeros(ACTION_DIM)
            action[action_idx] = 1.0

            # Step env
            _, _, done, _, obs_dict_next = env.step(action.unsqueeze(0))

            harm_signal   = float(obs_dict_next.get("harm_signal", 0.0))
            resource_cnt  = float(obs_dict_next.get("resource_count", 0.0))
            harm_actual   = torch.tensor([harm_signal], dtype=torch.float32)

            # Harm evaluator loss
            harm_pred_val  = harm_eval(z_world, z_self)
            harm_eval_loss = F.mse_loss(harm_pred_val, harm_actual)

            # Policy entropy bonus (encourages exploration)
            log_probs = F.log_softmax(logits, dim=-1)
            entropy   = -(probs * log_probs).sum()
            loss      = harm_eval_loss - ENT_BONUS * entropy

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()

            if in_measurement:
                probs_list = probs.detach().tolist()
                harm_vals_meas.append(harm_signal)
                resource_vals_meas.append(resource_cnt)
                action_probs_meas.append(probs_list)

                if condition in ("FULL_AXES", "MINIMAL_SUBSET"):
                    tonic_vals.append(z_tonic_val)
                    phasic_vals.append(z_phasic_val)
                if condition == "FULL_AXES":
                    affect_vals.append(z_affect_val)

                # Regime window accumulation
                window_harm.append(harm_signal)
                window_resource.append(resource_cnt)
                regime_probs_window.append(probs_list)

                if len(window_harm) >= WINDOW_SIZE:
                    regime = _classify_regime(window_harm, window_resource)
                    w_harm_rate = sum(window_harm) / len(window_harm)
                    w_ent       = _action_entropy(regime_probs_window)
                    regime_harm_rates[regime].append(w_harm_rate)
                    regime_action_ent[regime].append(w_ent)
                    regime_counts[regime] += 1
                    window_harm           = []
                    window_resource       = []
                    regime_probs_window   = []

            obs_dict = obs_dict_next
            total_steps += 1
            if done:
                _, obs_dict = env.reset()

    # ---------------------------------------------------------------------------
    # Compute metrics
    # ---------------------------------------------------------------------------
    n_meas = len(harm_vals_meas)

    # Overall harm rate in measurement window
    harm_rate = float(sum(harm_vals_meas) / n_meas) if n_meas > 0 else 0.0

    # Regime separation: mean |harm_avoidance - harm_exploration|
    #                  + mean |ent_exploitation - ent_exploration|
    def _mean(lst: List[float]) -> float:
        return float(sum(lst) / len(lst)) if lst else 0.0

    harm_avoid  = _mean(regime_harm_rates["avoidance"])
    harm_expl   = _mean(regime_harm_rates["exploration"])
    ent_exploit = _mean(regime_action_ent["exploitation"])
    ent_expl    = _mean(regime_action_ent["exploration"])

    regime_sep = abs(harm_avoid - harm_expl) + abs(ent_exploit - ent_expl)

    # Axis variances (for C4)
    def _var(lst: List[float]) -> float:
        if len(lst) < 2:
            return 0.0
        m = sum(lst) / len(lst)
        return sum((v - m) ** 2 for v in lst) / len(lst)

    axis_var_tonic  = _var(tonic_vals)  if tonic_vals  else 0.0
    axis_var_phasic = _var(phasic_vals) if phasic_vals else 0.0
    axis_var_affect = _var(affect_vals) if affect_vals else 0.0

    # Action entropy over measurement window
    action_entropy = _action_entropy(action_probs_meas)

    return {
        "condition":               condition,
        "seed":                    seed,
        "regime_sep":              regime_sep,
        "harm_rate":               harm_rate,
        "action_entropy":          action_entropy,
        "n_avoidance_windows":     regime_counts["avoidance"],
        "n_exploitation_windows":  regime_counts["exploitation"],
        "n_exploration_windows":   regime_counts["exploration"],
        "harm_avoid_mean":         harm_avoid,
        "harm_expl_mean":          harm_expl,
        "ent_exploit_mean":        ent_exploit,
        "ent_expl_mean":           ent_expl,
        "axis_var_tonic":          axis_var_tonic,
        "axis_var_phasic":         axis_var_phasic,
        "axis_var_affect":         axis_var_affect,
        "n_measurement_steps":     n_meas,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(dry_run: bool = False) -> None:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id    = f"{EXPERIMENT_TYPE}_{timestamp}_v3"

    print(f"[EXQ-157] Starting {EXPERIMENT_TYPE}")
    print(f"[EXQ-157] run_id = {run_id}")
    print(f"[EXQ-157] dry_run = {dry_run}")
    print(f"[EXQ-157] conditions = {CONDITIONS}, seeds = {SEEDS}")

    cells: List[Dict] = []
    for seed in SEEDS:
        for condition in CONDITIONS:
            print(f"[EXQ-157] Running condition={condition} seed={seed}")
            result = _run_condition(
                seed               = seed,
                condition          = condition,
                total_episodes     = TOTAL_EPISODES,
                measurement_episodes = MEASUREMENT_EPISODES,
                steps_per_episode  = STEPS_PER_EPISODE,
                lr                 = LR,
                dry_run            = dry_run,
            )
            print(
                f"[EXQ-157]   regime_sep={result['regime_sep']:.4f}  "
                f"harm_rate={result['harm_rate']:.4f}  "
                f"action_entropy={result['action_entropy']:.4f}  "
                f"n_avoid={result['n_avoidance_windows']}  "
                f"n_exploit={result['n_exploitation_windows']}  "
                f"n_explore={result['n_exploration_windows']}"
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

    c1_pass = all(
        _cell("FULL_AXES", s)["regime_sep"]
        > _cell("COLLAPSED_SCALAR", s)["regime_sep"] + THRESH_SEP_MARGIN
        for s in SEEDS
    )
    c2_pass = all(
        _cell("FULL_AXES", s)["regime_sep"] > THRESH_SEP_ABS_MIN
        for s in SEEDS
    )
    c3_pass = all(
        (
            _cell(cond, s)["n_avoidance_windows"]    >= THRESH_MIN_REGIME_WINDOWS
            and _cell(cond, s)["n_exploitation_windows"] >= THRESH_MIN_REGIME_WINDOWS
            and _cell(cond, s)["n_exploration_windows"]  >= THRESH_MIN_REGIME_WINDOWS
        )
        for cond in CONDITIONS
        for s in SEEDS
    )
    c4_pass = all(
        (
            _cell("FULL_AXES", s)["axis_var_tonic"]  > THRESH_AXIS_VAR_MIN
            and _cell("FULL_AXES", s)["axis_var_phasic"] > THRESH_AXIS_VAR_MIN
            and _cell("FULL_AXES", s)["axis_var_affect"] > THRESH_AXIS_VAR_MIN
        )
        for s in SEEDS
    )
    # C5: seed-consistent direction for FULL vs MINIMAL
    full_vs_minimal_gaps = [
        _cell("FULL_AXES", s)["regime_sep"] - _cell("MINIMAL_SUBSET", s)["regime_sep"]
        for s in SEEDS
    ]
    c5_seed_consistent = all(g > 0 for g in full_vs_minimal_gaps) or all(
        g <= 0 for g in full_vs_minimal_gaps
    )
    c5_all_within_equiv = all(
        abs(g) <= THRESH_EQUIV for g in full_vs_minimal_gaps
    )
    c5_all_above_margin = all(
        g > THRESH_SEP_MARGIN for g in full_vs_minimal_gaps
    )

    if c1_pass and c2_pass and c3_pass and c4_pass and c5_all_above_margin:
        outcome = "PASS"
    elif c1_pass and c2_pass and c3_pass and c4_pass and c5_all_within_equiv:
        outcome = "PARTIAL_TWO_SUFFICIENT"
    elif not c1_pass:
        outcome = "PARTIAL_COLLAPSE_ADEQUATE"
    elif not c2_pass or not c3_pass:
        outcome = "FAIL"
    else:
        outcome = "PARTIAL_INCONCLUSIVE"

    print(f"[EXQ-157] C1 (FULL>COLLAPSED+margin): {c1_pass}")
    print(f"[EXQ-157] C2 (FULL regime_sep>floor):  {c2_pass}")
    print(f"[EXQ-157] C3 (regime window counts):   {c3_pass}")
    print(f"[EXQ-157] C4 (axis non-degenerate):    {c4_pass}")
    print(f"[EXQ-157] C5 FULL-MINIMAL gaps: {[round(g, 4) for g in full_vs_minimal_gaps]}")
    print(f"[EXQ-157] Outcome: {outcome}")

    # ---------------------------------------------------------------------------
    # Build summary table
    # ---------------------------------------------------------------------------
    summary_rows = []
    for cond in CONDITIONS:
        for s in SEEDS:
            cell = _cell(cond, s)
            summary_rows.append({
                "condition":       cond,
                "seed":            s,
                "regime_sep":      round(cell["regime_sep"], 5),
                "harm_rate":       round(cell["harm_rate"], 5),
                "action_entropy":  round(cell["action_entropy"], 4),
                "n_avoid":         cell["n_avoidance_windows"],
                "n_exploit":       cell["n_exploitation_windows"],
                "n_explore":       cell["n_exploration_windows"],
                "axis_var_tonic":  round(cell["axis_var_tonic"], 6),
                "axis_var_phasic": round(cell["axis_var_phasic"], 6),
                "axis_var_affect": round(cell["axis_var_affect"], 6),
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
            "window_size":             WINDOW_SIZE,
            "tonic_tau":               TONIC_TAU,
            "lr":                      LR,
            "ent_bonus":               ENT_BONUS,
            "seeds":                   SEEDS,
            "conditions":              CONDITIONS,
            "control_dims":            _CONTROL_DIM,
        },
        "thresholds": {
            "THRESH_SEP_MARGIN":         THRESH_SEP_MARGIN,
            "THRESH_SEP_ABS_MIN":        THRESH_SEP_ABS_MIN,
            "THRESH_MIN_REGIME_WINDOWS": THRESH_MIN_REGIME_WINDOWS,
            "THRESH_AXIS_VAR_MIN":       THRESH_AXIS_VAR_MIN,
            "THRESH_EQUIV":              THRESH_EQUIV,
            "REGIME_HARM_THRESH":        REGIME_HARM_THRESH,
        },
        "criteria": {
            "C1_full_beats_collapsed": c1_pass,
            "C2_full_regime_sep_floor": c2_pass,
            "C3_regime_window_counts": c3_pass,
            "C4_axes_non_degenerate": c4_pass,
            "C5_seed_consistent_direction": c5_seed_consistent,
            "C5_full_vs_minimal_gaps": [round(g, 5) for g in full_vs_minimal_gaps],
            "C5_all_within_equiv": c5_all_within_equiv,
            "C5_all_above_margin": c5_all_above_margin,
        },
        "outcome":         outcome,
        "evidence_class":  "experimental",
        "evidence_direction": (
            "supports"  if outcome == "PASS"
            else "mixed" if outcome in ("PARTIAL_TWO_SUFFICIENT",
                                        "PARTIAL_COLLAPSE_ADEQUATE",
                                        "PARTIAL_INCONCLUSIVE")
            else "weakens"
        ),
        "summary": (
            f"Outcome={outcome}. Conditions: FULL_AXES (3 orthogonal axes), "
            f"COLLAPSED_SCALAR (1 raw axis), MINIMAL_SUBSET (2 axes: tonic+phasic). "
            f"C1 FULL>COLLAPSED: {c1_pass}. C2 FULL floor: {c2_pass}. "
            f"C3 regime counts: {c3_pass}. C4 axes non-degenerate: {c4_pass}. "
            f"FULL vs MINIMAL regime_sep gaps: {[round(g,4) for g in full_vs_minimal_gaps]}. "
            f"PASS => all three axes load-bearing. "
            f"PARTIAL_TWO_SUFFICIENT => tonic+phasic sufficient (affect dispensable). "
            f"PARTIAL_COLLAPSE_ADEQUATE => raw harm sufficient at V3 scale. "
            f"FAIL => insufficient regime events."
        ),
        "cells":   cells,
        "summary_table": summary_rows,
    }

    with open(output_path, "w") as fh:
        json.dump(pack, fh, indent=2)

    print(f"[EXQ-157] Output written to {output_path}")
    print(f"[EXQ-157] DONE -- outcome={outcome}")


if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv
    main(dry_run=dry_run)
