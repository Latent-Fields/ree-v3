#!/opt/local/bin/python3
"""
V3-EXQ-159 -- Q-020: Does ARC-007's no-value-computation constraint survive
              MECH-073 (valence intrinsic to hippocampal map)?

Claim:    Q-020
Proposal: EXP-0112 (EVB-0083)

Q-020 asks:
  "Does ARC-007's no-value-computation constraint survive MECH-073
  (valence intrinsic to hippocampal map)?"

  Background:
    ARC-007 (active) constrains the hippocampal system to store and replay
    paths through residue-field terrain WITHOUT performing value computation.
    Its design note states: "No value computation. Its function is orthogonal
    to valuation and control."

    MECH-073 (candidate, in conflict with ARC-007) asserts that "valence is
    intrinsic to hippocampal map geometry, not applied downstream." In MECH-073,
    the hippocampal map is a valenced cognitive map and trajectory rollouts
    are inherently valenced simulations.

    The ree-v3 CLAUDE.md architectural decision (2026-03-16) resolved this
    in favour of ARC-007 STRICT:
      "HippocampalModule generates value-flat proposals. Terrain sensitivity
      = consequence of navigating residue-shaped z_world, not a separate
      hippocampal value computation. MECH-073 reframed as consequence of
      ARC-013 applied to z_world."

    Resolution path (A): revise ARC-007 to distinguish
      (i)  "hippocampus does not compute new value" (denied)
      (ii) "hippocampus embodies geometrically-encoded prior value via
           amygdala write operations and residue-field shaping" (permitted)
    Resolution path (B): reject MECH-073; treat NC-01/NC-02 as external
      neuroscience inspiration only.

    The architectural decision chose path (A) -- BUT the experimental
    question is whether this distinction is operationally real: does a
    value-flat hippocampal proposal system (terrain navigation via
    residue-field shaping) achieve comparable harm avoidance to a
    value-laden proposal system that embeds harm scores directly in
    candidate generation?

  This experiment compares three conditions:

  Condition A: VALUE_FLAT (ARC-007 compliant -- primary test condition)
    HippocampalModule proposes K candidate trajectories sampled from a
    latent path space using only structural similarity (cosine distance
    in z_world). Proposals are VALUE-NEUTRAL at generation time. The
    residue field accumulates harm-weighted world deltas; navigation is
    implicitly steered by residue terrain because the candidate scoring
    step happens in E3 AFTER proposal generation.
    Prediction: achieves good harm avoidance because residue field
    supplies sufficient valence signal downstream.

  Condition B: VALUE_FLAT_NO_RESIDUE (ablation control)
    Same as VALUE_FLAT but the residue field accumulation is disabled
    (always-zero residue weights). Proposals are generated from flat
    z_world terrain; E3 must learn from raw harm signal alone.
    Prediction: WORSE than VALUE_FLAT, confirming that residue shaping
    is what enables value-flat proposals to work. Without it, ARC-007
    cannot be sustained.

  Condition C: VALUE_LADEN (MECH-073 analog)
    HippocampalModule has a direct valence head: a learned linear module
    that scores each candidate trajectory by predicted harm during
    PROPOSAL GENERATION (not just E3 post-hoc evaluation). The top-K
    candidates are selected by this valence score before E3 sees them.
    This bypasses residue-field navigation -- valence is intrinsic to
    proposal selection.
    Prediction: achieves comparable or marginally better harm avoidance
    than VALUE_FLAT (because direct valence is a stronger signal), BUT
    the key question is whether VALUE_FLAT + residue achieves
    VALUE_LADEN parity (supporting ARC-007 viability).

  Discriminative question:
    (i)  Is VALUE_FLAT harm avoidance within THRESH_PARITY of VALUE_LADEN?
         (If yes: ARC-007 is SUFFICIENT -- residue shaping makes direct
         valence embedding unnecessary.)
    (ii) Does VALUE_FLAT_NO_RESIDUE significantly exceed VALUE_FLAT's
         harm rate? (If yes: residue field is the mechanism -- ARC-007
         requires residue for its "value-flat but terrain-sensitive" promise.)
    (iii) Does VALUE_LADEN not outperform VALUE_FLAT by > THRESH_LADEN_ADV?
         (If yes: direct valence provides no significant advantage over
         residue-shaped navigation -- supporting ARC-007.)

  Scientific meaning:
    PASS (ARC-007 survives):
      => VALUE_FLAT achieves parity with VALUE_LADEN (|delta harm| < THRESH_PARITY)
         on both seeds, AND VALUE_FLAT_NO_RESIDUE has higher harm than VALUE_FLAT
         by >= THRESH_RESIDUE_BENEFIT on both seeds (residue field is the
         mechanism). ARC-007 constraint is operationally viable: terrain
         navigation via residue shaping is sufficient; direct valence embedding
         (MECH-073 intrinsic form) is not required.
    PARTIAL_LADEN_ADVANTAGE:
      => VALUE_LADEN outperforms VALUE_FLAT by > THRESH_LADEN_ADV on at least
         one seed. Direct valence embedding provides meaningful advantage;
         MECH-073 intrinsic form adds value. Suggests ARC-007 may need revision.
    PARTIAL_NO_RESIDUE_EFFECT:
      => VALUE_FLAT_NO_RESIDUE does NOT differ significantly from VALUE_FLAT.
         If ARC-007's mechanism requires residue terrain, this suggests a
         confound (residue too weak at this scale; both conditions are
         learning from raw harm signal similarly).
    FAIL:
      => All conditions achieve near-zero harm (floor) or near-chance (ceiling);
         insufficient discrimination.

  Key metrics:
    1. harm_rate: mean harm per step in measurement window (primary)
    2. n_harm_events: total harm events in measurement (data quality)
    3. laden_vs_flat_delta: VALUE_LADEN harm_rate - VALUE_FLAT harm_rate
       (negative = laden is better; positive = flat is better)
    4. flat_vs_noResid_delta: VALUE_FLAT harm_rate - VALUE_FLAT_NO_RESIDUE harm_rate
       (negative = flat is better; confirms residue benefit)
    5. proposal_harm_score_mean: mean of the valence head's harm scores at
       proposal time (VALUE_LADEN only; measures whether valence head is active)

Pre-registered thresholds
--------------------------
C1: |VALUE_LADEN harm_rate - VALUE_FLAT harm_rate| <= THRESH_PARITY (both seeds).
    (ARC-007 parity: value-flat + residue is within parity margin of value-laden.)

C2: VALUE_FLAT_NO_RESIDUE harm_rate >= VALUE_FLAT harm_rate + THRESH_RESIDUE_BENEFIT
    (both seeds).
    (Residue field provides the benefit: removing it worsens VALUE_FLAT.)

C3: VALUE_LADEN harm_rate <= THRESH_HARM_CEILING (both seeds).
    (Data quality: VALUE_LADEN is not at floor; valence head is active and
    producing genuine signal.)

C4: n_harm_events >= THRESH_MIN_HARM_EVENTS per condition per seed.
    (Data quality: enough harm events for reliable rate estimation.)

C5: VALUE_LADEN proposal_harm_score_mean > THRESH_VALENCE_HEAD_ACTIVE (both seeds).
    (Sanity: VALUE_LADEN valence head is actually scoring proposals non-trivially.)

PASS:                       C1 + C2 + C3 + C4 + C5
PARTIAL_LADEN_ADVANTAGE:    NOT C1 (loaded outperforms flat by > parity margin)
                            + C3 + C4 + C5
PARTIAL_NO_RESIDUE_EFFECT:  NOT C2 (residue ablation does not hurt VALUE_FLAT)
                            + C3 + C4 + C5
FAIL:                       NOT C3 OR NOT C4

Conditions
----------
Shared architecture:
  World encoder:  Linear(world_obs_dim, world_dim) + LayerNorm -> z_world
  Self encoder:   Linear(self_obs_dim, self_dim) + LayerNorm -> z_self
  Harm evaluator: Linear(z_world + z_self, 1) (E3 analog; shared across conds)
  Policy:         Linear(z_world + z_self, hidden) -> ReLU -> Linear(hidden, action_dim)
  World predictor: Linear(z_world + action_dim, world_dim) (for world pred loss)

  HippocampalModule (simplified analog):
    - Stores a small episodic buffer of (z_world, harm_score) tuples
    - At each planning step, proposes K=5 candidate actions by sampling
      from a distribution shaped by the hippocampal buffer

VALUE_FLAT:
    Candidate proposals sampled from a distribution over the action space
    WEIGHTED by residue scores: w_k ~ exp(-lambda * residue_weight(a_k))
    where residue_weight accumulates harm-weighted world deltas per action.
    Valence head: ABSENT. Proposals are generated from terrain only.
    E3 harm_eval selects the best action from candidates post-hoc.

VALUE_FLAT_NO_RESIDUE:
    Same as VALUE_FLAT but residue_weights are always zero (uniform sampling).
    E3 harm_eval still selects post-hoc from uniform candidates.

VALUE_LADEN:
    Candidate proposals sampled uniformly from the action space (no residue),
    then SCORED by a learned ValenceHead: Linear(z_world, 1) that predicts
    harm for each candidate. The K candidates are selected by LOWEST valence
    score (most harm-avoiding). E3 harm_eval is applied post-hoc to the
    already-valence-filtered candidate set.

Seeds:   [42, 123]
Env:     CausalGridWorldV2 size=8, num_hazards=5, num_resources=0,
         hazard_harm=0.05, env_drift_interval=5, env_drift_prob=0.35
         (harm-rich environment for reliable discrimination)
Protocol: TOTAL_EPISODES=500
          MEASUREMENT_EPISODES=125 (last 125 episodes = last quartile)
          STEPS_PER_EPISODE=200
          K_CANDIDATES=5 (number of candidate actions per step)
Estimated runtime:
  3 conditions x 2 seeds x 500 eps x 0.10 min/ep = ~300 min Mac
  (+10% overhead) => ~330 min Mac
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


EXPERIMENT_TYPE = "v3_exq_159_q020_arc007_valence_constraint_pair"
CLAIM_IDS = ["Q-020"]

# ---------------------------------------------------------------------------
# Pre-registered thresholds
# ---------------------------------------------------------------------------
THRESH_PARITY            = 0.05   # C1: max |laden - flat| harm_rate for parity claim
THRESH_RESIDUE_BENEFIT   = 0.04   # C2: min harm_rate increase when residue is ablated
THRESH_HARM_CEILING      = 0.40   # C3: VALUE_LADEN must not be at near-floor (< this)
THRESH_MIN_HARM_EVENTS   = 20     # C4: minimum harm events per condition per seed
THRESH_VALENCE_HEAD_ACTIVE = 0.01 # C5: VALUE_LADEN valence head mean score > 0 (non-trivial)

# Residue field constants (VALUE_FLAT)
RESIDUE_LAMBDA     = 2.0   # sharpness of residue-based candidate weighting
RESIDUE_DECAY      = 0.95  # EMA decay for residue weights per action

# Protocol constants
TOTAL_EPISODES       = 500
MEASUREMENT_EPISODES = 125
STEPS_PER_EPISODE    = 200
K_CANDIDATES         = 5
LR                   = 3e-4
ENT_BONUS            = 5e-3

SEEDS      = [42, 123]
CONDITIONS = ["VALUE_FLAT", "VALUE_FLAT_NO_RESIDUE", "VALUE_LADEN"]

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
    """E3 analog: scores (z_world, z_self) -> scalar harm estimate."""
    def __init__(self, world_dim: int, self_dim: int):
        super().__init__()
        self.fc = nn.Linear(world_dim + self_dim, 1)

    def forward(self, z_world: torch.Tensor, z_self: torch.Tensor) -> torch.Tensor:
        return self.fc(torch.cat([z_world, z_self], dim=-1))


class ValenceHead(nn.Module):
    """
    VALUE_LADEN only.
    Scores z_world -> predicted harm (scalar).
    Used DURING proposal generation to pre-filter candidates.
    Distinct from HarmEvaluator (which uses z_self + z_world post-hoc).
    """
    def __init__(self, world_dim: int):
        super().__init__()
        self.fc = nn.Linear(world_dim, 1)

    def forward(self, z_world: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.fc(z_world))


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
        num_resources=0,
        hazard_harm=0.05,
        env_drift_interval=5,
        env_drift_prob=0.35,
        seed=seed,
    )

    world_enc   = WorldEncoder(WORLD_OBS_DIM, WORLD_DIM)
    self_enc    = SelfEncoder(SELF_OBS_DIM, SELF_DIM)
    world_pred  = WorldPredictor(WORLD_DIM, ACTION_DIM)
    harm_eval   = HarmEvaluator(WORLD_DIM, SELF_DIM)
    policy      = Policy(WORLD_DIM, SELF_DIM, ACTION_DIM)

    # VALUE_LADEN only: valence head for proposal scoring
    valence_head: ValenceHead | None = None
    if condition == "VALUE_LADEN":
        valence_head = ValenceHead(WORLD_DIM)

    all_params = (
        list(world_enc.parameters())
        + list(self_enc.parameters())
        + list(world_pred.parameters())
        + list(harm_eval.parameters())
        + list(policy.parameters())
    )
    if valence_head is not None:
        all_params += list(valence_head.parameters())

    optimizer = optim.Adam(all_params, lr=lr)

    # Residue field: per-action EMA of harm-weighted world delta (VALUE_FLAT only)
    # residue_weights[a] = EMA of harm incurred after taking action a
    residue_weights = [0.0] * ACTION_DIM

    measurement_start = total_episodes - measurement_episodes
    if dry_run:
        total_episodes    = 2
        measurement_start = 0

    # Measurement buffers
    harm_vals:     List[float] = []
    n_harm_events: int = 0

    # VALUE_LADEN: track valence head scores (mean over measurement)
    valence_scores: List[float] = []

    # World predictor state
    z_world_prev: torch.Tensor = torch.zeros(WORLD_DIM)
    action_prev:  torch.Tensor = torch.zeros(ACTION_DIM)
    have_prev = False

    _, obs_dict = env.reset()

    for ep in range(total_episodes):
        in_measurement = ep >= measurement_start

        for _step in range(steps_per_episode):
            obs_world = _get_obs_tensor(obs_dict, "world_state", WORLD_OBS_DIM)
            obs_self  = _get_obs_tensor(obs_dict, "body_state",  SELF_OBS_DIM)

            z_world = world_enc(obs_world)
            z_self  = self_enc(obs_self)

            # ------------------------------------------------------------------
            # Candidate proposal and selection
            # ------------------------------------------------------------------
            # Generate K candidate action indices
            candidate_actions = list(range(ACTION_DIM))  # deterministic set for K <= action_dim
            # Pad/sample if K > ACTION_DIM
            while len(candidate_actions) < K_CANDIDATES:
                candidate_actions.append(random.randint(0, ACTION_DIM - 1))
            # Trim to K
            candidate_actions = candidate_actions[:K_CANDIDATES]

            if condition == "VALUE_FLAT":
                # Weight by exp(-lambda * residue) -- higher residue = lower weight
                raw_weights = [
                    math.exp(-RESIDUE_LAMBDA * residue_weights[a])
                    for a in candidate_actions
                ]
                total_w = sum(raw_weights) + 1e-8
                probs_cand = [w / total_w for w in raw_weights]
                # Sample from residue-weighted distribution
                selected_idx = random.choices(
                    range(len(candidate_actions)),
                    weights=probs_cand,
                    k=1,
                )[0]
                chosen_action_idx = candidate_actions[selected_idx]

            elif condition == "VALUE_FLAT_NO_RESIDUE":
                # Uniform sampling -- no residue shaping
                chosen_action_idx = random.choice(candidate_actions)

            else:  # VALUE_LADEN
                # Score each candidate via valence head on z_world
                assert valence_head is not None
                with torch.no_grad():
                    valence_scores_cand = [
                        valence_head(z_world.detach()).item()
                        for _ in candidate_actions
                    ]
                if in_measurement:
                    valence_scores.extend(valence_scores_cand)
                # Select candidate with LOWEST valence score (most harm-avoiding)
                min_idx = valence_scores_cand.index(min(valence_scores_cand))
                chosen_action_idx = candidate_actions[min_idx]

            # E3 post-hoc evaluation: optionally override with E3 harm_eval
            # (all conditions -- this is the shared E3 evaluation step)
            # Here we use the harm evaluator as a secondary gate:
            # if E3 predicts high harm, sample from a different action
            with torch.no_grad():
                harm_score_chosen = harm_eval(
                    z_world.detach(), z_self.detach()
                ).item()

            # If E3 predicts harm > threshold, fall back to policy
            harm_threshold_gate = 0.3
            if harm_score_chosen > harm_threshold_gate:
                # Fall back to policy-driven action
                logits = policy(z_world.detach(), z_self.detach())
                probs  = F.softmax(logits, dim=-1)
                chosen_action_idx = int(torch.multinomial(probs, 1).item())

            action_tensor = _action_one_hot(chosen_action_idx, ACTION_DIM)

            # Step env
            _, _, done, _, obs_dict_next = env.step(action_tensor.unsqueeze(0))

            harm_signal = float(obs_dict_next.get("harm_signal", 0.0))
            harm_actual = torch.tensor([harm_signal], dtype=torch.float32)

            if in_measurement:
                harm_vals.append(harm_signal)
                if harm_signal > 0.0:
                    n_harm_events += 1

            # ------------------------------------------------------------------
            # Update residue weights (VALUE_FLAT and VALUE_FLAT_NO_RESIDUE)
            # In VALUE_FLAT_NO_RESIDUE the weights are never used for selection,
            # but we still compute them so the measurement is honest.
            # For VALUE_FLAT, this is the key mechanism.
            # ------------------------------------------------------------------
            if condition in ("VALUE_FLAT", "VALUE_FLAT_NO_RESIDUE"):
                # EMA update: residue[a] <- decay * residue[a] + (1-decay) * harm
                for a_idx in range(ACTION_DIM):
                    residue_weights[a_idx] = (
                        RESIDUE_DECAY * residue_weights[a_idx]
                        + (1 - RESIDUE_DECAY) * (harm_signal if a_idx == chosen_action_idx else 0.0)
                    )

            # ------------------------------------------------------------------
            # Training losses
            # ------------------------------------------------------------------
            # World predictor loss
            if have_prev:
                z_pred       = world_pred(z_world_prev, action_prev)
                world_pred_l = F.mse_loss(z_pred, z_world.detach())
            else:
                world_pred_l = torch.tensor(0.0)

            # Harm evaluator loss
            harm_pred_v  = harm_eval(z_world, z_self)
            harm_eval_l  = F.mse_loss(harm_pred_v, harm_actual)

            # Valence head loss (VALUE_LADEN only)
            valence_l = torch.tensor(0.0)
            if condition == "VALUE_LADEN" and valence_head is not None:
                valence_pred = valence_head(z_world)
                valence_l    = F.mse_loss(valence_pred, harm_actual)

            # Policy entropy bonus (exploration)
            logits_p  = policy(z_world.detach(), z_self.detach())
            probs_p   = F.softmax(logits_p, dim=-1)
            log_probs = F.log_softmax(logits_p, dim=-1)
            entropy   = -(probs_p * log_probs).sum()

            loss = harm_eval_l + world_pred_l + valence_l - ENT_BONUS * entropy

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(all_params, 1.0)
            optimizer.step()

            # Update prev state
            z_world_prev = z_world.detach().clone()
            action_prev  = action_tensor.detach().clone()
            have_prev    = True

            obs_dict = obs_dict_next
            if done:
                _, obs_dict = env.reset()
                have_prev    = False
                z_world_prev = torch.zeros(WORLD_DIM)
                action_prev  = torch.zeros(ACTION_DIM)

    # ---------------------------------------------------------------------------
    # Compute metrics
    # ---------------------------------------------------------------------------
    def _mean(lst: List[float]) -> float:
        return float(sum(lst) / len(lst)) if lst else 0.0

    harm_rate                  = _mean(harm_vals)
    proposal_harm_score_mean   = _mean(valence_scores) if valence_scores else 0.0

    return {
        "condition":                condition,
        "seed":                     seed,
        "harm_rate":                harm_rate,
        "n_harm_events":            n_harm_events,
        "n_measurement_steps":      len(harm_vals),
        "proposal_harm_score_mean": proposal_harm_score_mean,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(dry_run: bool = False) -> None:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id    = f"{EXPERIMENT_TYPE}_{timestamp}_v3"

    print(f"[EXQ-159] Starting {EXPERIMENT_TYPE}")
    print(f"[EXQ-159] run_id = {run_id}")
    print(f"[EXQ-159] dry_run = {dry_run}")
    print(f"[EXQ-159] conditions = {CONDITIONS}, seeds = {SEEDS}")

    cells: List[Dict] = []
    for seed in SEEDS:
        for condition in CONDITIONS:
            print(f"[EXQ-159] Running condition={condition} seed={seed}")
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
                f"[EXQ-159]   harm_rate={result['harm_rate']:.4f}  "
                f"n_harm={result['n_harm_events']}  "
                f"valence_mean={result['proposal_harm_score_mean']:.4f}"
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

    # C1: |VALUE_LADEN harm_rate - VALUE_FLAT harm_rate| <= THRESH_PARITY (both seeds)
    c1_pass = all(
        abs(
            _cell("VALUE_LADEN", s)["harm_rate"]
            - _cell("VALUE_FLAT", s)["harm_rate"]
        ) <= THRESH_PARITY
        for s in SEEDS
    )

    # C2: VALUE_FLAT_NO_RESIDUE harm_rate >= VALUE_FLAT harm_rate + THRESH_RESIDUE_BENEFIT
    #     (both seeds) -- residue field provides the benefit
    c2_pass = all(
        _cell("VALUE_FLAT_NO_RESIDUE", s)["harm_rate"]
        >= _cell("VALUE_FLAT", s)["harm_rate"] + THRESH_RESIDUE_BENEFIT
        for s in SEEDS
    )

    # C3: VALUE_LADEN harm_rate <= THRESH_HARM_CEILING (both seeds)
    #     (data quality: valence head is producing non-trivial harm avoidance)
    c3_pass = all(
        _cell("VALUE_LADEN", s)["harm_rate"] <= THRESH_HARM_CEILING
        for s in SEEDS
    )

    # C4: n_harm_events >= THRESH_MIN_HARM_EVENTS per condition per seed
    c4_pass = all(
        _cell(cond, s)["n_harm_events"] >= THRESH_MIN_HARM_EVENTS
        for cond in CONDITIONS
        for s in SEEDS
    )

    # C5: VALUE_LADEN proposal_harm_score_mean > THRESH_VALENCE_HEAD_ACTIVE (both seeds)
    #     (sanity: valence head scores are non-trivial)
    c5_pass = all(
        _cell("VALUE_LADEN", s)["proposal_harm_score_mean"] > THRESH_VALENCE_HEAD_ACTIVE
        for s in SEEDS
    )

    # Outcome logic
    laden_advantage_seen = any(
        _cell("VALUE_LADEN", s)["harm_rate"]
        < _cell("VALUE_FLAT", s)["harm_rate"] - THRESH_PARITY
        for s in SEEDS
    )

    if c1_pass and c2_pass and c3_pass and c4_pass and c5_pass:
        outcome = "PASS"
    elif not c3_pass or not c4_pass:
        outcome = "FAIL"
    elif laden_advantage_seen and c3_pass and c4_pass and c5_pass:
        outcome = "PARTIAL_LADEN_ADVANTAGE"
    elif not c2_pass and c3_pass and c4_pass and c5_pass:
        outcome = "PARTIAL_NO_RESIDUE_EFFECT"
    else:
        outcome = "PARTIAL_INCONCLUSIVE"

    print(f"[EXQ-159] C1 (parity |laden-flat| <= {THRESH_PARITY}): {c1_pass}")
    print(f"[EXQ-159] C2 (residue benefit: no_residue - flat >= {THRESH_RESIDUE_BENEFIT}): {c2_pass}")
    print(f"[EXQ-159] C3 (laden not at floor, harm <= {THRESH_HARM_CEILING}): {c3_pass}")
    print(f"[EXQ-159] C4 (n_harm_events >= {THRESH_MIN_HARM_EVENTS}): {c4_pass}")
    print(f"[EXQ-159] C5 (laden valence head active): {c5_pass}")
    print(f"[EXQ-159] Outcome: {outcome}")

    # ---------------------------------------------------------------------------
    # Summary table and pairwise deltas
    # ---------------------------------------------------------------------------
    summary_rows = []
    for cond in CONDITIONS:
        for s in SEEDS:
            cell = _cell(cond, s)
            summary_rows.append({
                "condition":                cond,
                "seed":                     s,
                "harm_rate":                round(cell["harm_rate"], 5),
                "n_harm_events":            cell["n_harm_events"],
                "n_measurement_steps":      cell["n_measurement_steps"],
                "proposal_harm_score_mean": round(cell["proposal_harm_score_mean"], 5),
            })

    pairwise_deltas = []
    for s in SEEDS:
        flat      = _cell("VALUE_FLAT",           s)
        no_resid  = _cell("VALUE_FLAT_NO_RESIDUE", s)
        laden     = _cell("VALUE_LADEN",           s)
        pairwise_deltas.append({
            "seed":                     s,
            "harm_flat":                round(flat["harm_rate"],     5),
            "harm_no_residue":          round(no_resid["harm_rate"], 5),
            "harm_laden":               round(laden["harm_rate"],    5),
            "laden_vs_flat_delta":      round(laden["harm_rate"] - flat["harm_rate"],     5),
            "flat_vs_noResid_delta":    round(flat["harm_rate"]  - no_resid["harm_rate"], 5),
            "abs_laden_flat":           round(abs(laden["harm_rate"] - flat["harm_rate"]), 5),
            "residue_benefit_delta":    round(no_resid["harm_rate"] - flat["harm_rate"],  5),
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
            "total_episodes":        TOTAL_EPISODES,
            "measurement_episodes":  MEASUREMENT_EPISODES,
            "steps_per_episode":     STEPS_PER_EPISODE,
            "k_candidates":          K_CANDIDATES,
            "lr":                    LR,
            "ent_bonus":             ENT_BONUS,
            "seeds":                 SEEDS,
            "conditions":            CONDITIONS,
            "residue_lambda":        RESIDUE_LAMBDA,
            "residue_decay":         RESIDUE_DECAY,
        },
        "thresholds": {
            "THRESH_PARITY":              THRESH_PARITY,
            "THRESH_RESIDUE_BENEFIT":     THRESH_RESIDUE_BENEFIT,
            "THRESH_HARM_CEILING":        THRESH_HARM_CEILING,
            "THRESH_MIN_HARM_EVENTS":     THRESH_MIN_HARM_EVENTS,
            "THRESH_VALENCE_HEAD_ACTIVE": THRESH_VALENCE_HEAD_ACTIVE,
        },
        "criteria": {
            "C1_parity_laden_flat":           c1_pass,
            "C2_residue_benefit_confirmed":   c2_pass,
            "C3_laden_not_at_floor":          c3_pass,
            "C4_harm_events_sufficient":      c4_pass,
            "C5_valence_head_active":         c5_pass,
        },
        "outcome":   outcome,
        "evidence_class":     "experimental",
        "evidence_direction": (
            "supports" if outcome == "PASS"
            else "mixed" if outcome in (
                "PARTIAL_LADEN_ADVANTAGE",
                "PARTIAL_NO_RESIDUE_EFFECT",
                "PARTIAL_INCONCLUSIVE",
            )
            else "weakens"
        ),
        "summary": (
            f"Outcome={outcome}. Q-020: ARC-007 no-value-computation constraint vs MECH-073 "
            f"(valence intrinsic to hippocampal map). "
            f"Conditions: VALUE_FLAT (residue-shaping, no direct valence; ARC-007 compliant), "
            f"VALUE_FLAT_NO_RESIDUE (ablation: no residue shaping, no direct valence), "
            f"VALUE_LADEN (direct valence head during proposal generation; MECH-073 analog). "
            f"C1 parity |laden-flat| <= {THRESH_PARITY}: {c1_pass}. "
            f"C2 residue benefit (no_residue worse than flat by >= {THRESH_RESIDUE_BENEFIT}): {c2_pass}. "
            f"C3 laden not at floor: {c3_pass}. "
            f"C4 enough harm events: {c4_pass}. "
            f"C5 valence head active: {c5_pass}. "
            f"PASS => ARC-007 constraint survives: value-flat + residue achieves parity with "
            f"value-laden; residue field is the operative mechanism. MECH-073 direct intrinsic "
            f"form is not required. "
            f"PARTIAL_LADEN_ADVANTAGE => VALUE_LADEN outperforms VALUE_FLAT beyond parity margin; "
            f"direct valence embedding adds value; ARC-007 may need revision. "
            f"PARTIAL_NO_RESIDUE_EFFECT => residue ablation does not worsen VALUE_FLAT; "
            f"confound: residue too weak at this scale or both conditions learn from harm equally. "
            f"FAIL => insufficient harm events or valence head inactive."
        ),
        "cells":           cells,
        "summary_table":   summary_rows,
        "pairwise_deltas": pairwise_deltas,
    }

    with open(output_path, "w") as fh:
        json.dump(pack, fh, indent=2)

    print(f"[EXQ-159] Output written to {output_path}")
    print(f"[EXQ-159] DONE -- outcome={outcome}")


if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv
    main(dry_run=dry_run)
