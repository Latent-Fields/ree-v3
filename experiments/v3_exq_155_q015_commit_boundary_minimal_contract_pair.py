#!/opt/local/bin/python3
"""
V3-EXQ-155 -- Q-015: What is the smallest commit-boundary token that still
              supports reliable multi-timescale attribution?

Claim:    Q-015
Proposal: EXP-0104 (EVB-0079)

Q-015 asks:
  "What is the smallest commit-boundary token that still supports reliable
  multi-timescale attribution?"

  Background from claims.yaml evidence_quality_note (EXQ-003, ree-v1-minimal,
  PASS, EVB-0041):
    At ree-v1-minimal scale, a WITH_BOUNDARY agent achieves lower last-Q harm
    than a BLENDED agent, and mean |pre_post_corr| = 0.044 (well below 0.7).
    This PARTIALLY answers Q-015: the minimal contract requires at minimum that
    pre-commit and post-commit signals stay uncorrelated AND routing them
    separately outperforms blending.

  This V3 experiment tests three boundary token conditions on a richer V3
  substrate (CausalGridWorldV2 with reafference-corrected z_world, SD-007):
    (A) WITH_BOUNDARY -- full pre/post commit separation (binary gate token):
        A single scalar gate token g in {0, 1} routes the loss:
          g=0 (pre-commit): error flows only to E1/E2 (prediction error path)
          g=1 (post-commit): error flows only to E3 (harm/goal evaluation path)
        This is the current REE MECH-061 design. The question is whether the
        full binary routing is necessary or whether something smaller works.
    (B) BLENDED -- no commit token (g is not present):
        All error signals flow through both prediction and harm paths simultaneously.
        Prediction loss + harm loss are combined with a fixed mixing coefficient.
        This is the null / baseline condition. MECH-061 (PASS at V1) predicts
        this will perform worse on multi-timescale attribution.
    (C) MINIMAL_BOUNDARY -- smallest possible boundary contract:
        The commit token is a soft scalar in [0, 1] (not binary) that is
        LEARNED (not externally imposed) and gates the loss split proportionally.
        Gate g is a sigmoid output of a learned gating network that takes the
        current z_self and z_world as input. This is the "minimal" boundary
        variant: the agent must discover when to split vs blend -- it is not
        told. Hypothesis: a soft learned gate can converge to near-binary
        behaviour and achieve comparable separation to WITH_BOUNDARY, revealing
        that the strict binary protocol is not necessary.

  The discriminative question:
    (i)  Can MINIMAL_BOUNDARY match WITH_BOUNDARY on attribution quality
         (|pre_post_corr| < THRESH_CORR_MAX) and policy performance
         (last_q_harm < WITH_BOUNDARY + THRESH_HARM_MARGIN)?
    (ii) Does WITH_BOUNDARY outperform BLENDED on both metrics?

  Scientific meaning:
    - If (i): the minimal sufficient boundary contract is a soft learned gate,
      not a hard binary token. REE can use a lighter implementation.
    - If NOT (i) but (ii): the strict binary boundary is necessary; a soft
      learned gate cannot discover the split reliably at V3 scale.
    - If NOT (ii): MECH-061 PASS at V1 does not replicate at V3 scale;
      architectural explanation needed (SD-007 reafference may change error
      signal properties).

  The key metrics:
    1. last_q_harm  -- mean harm in last quartile of training (policy quality)
    2. pre_post_corr -- |Pearson r| between pre-commit and post-commit error
       signals (signal separation; lower is better)
    3. gate_entropy (MINIMAL_BOUNDARY only) -- entropy of learned gate
       distribution (near 0 = converged to near-binary)

Pre-registered thresholds
--------------------------
C1: WITH_BOUNDARY last_q_harm < BLENDED last_q_harm - THRESH_HARM_MARGIN (both seeds).
    (Boundary token provides policy benefit over blending -- MECH-061 replicated at V3.)

C2: WITH_BOUNDARY mean_pre_post_corr < THRESH_CORR_MAX = 0.30 (both seeds).
    (WITH_BOUNDARY achieves signal separation -- pre/post channels stay uncorrelated.)

C3: BLENDED mean_pre_post_corr > THRESH_CORR_MIN_BLENDED = 0.25 (both seeds).
    (BLENDED shows substantial correlation -- confirms the separation is a
    boundary effect, not accidental uncorrelation from the environment.)

C4: MINIMAL_BOUNDARY last_q_harm < BLENDED last_q_harm - THRESH_HARM_MARGIN (both seeds).
    (Soft learned gate also outperforms blending -- minimal boundary is sufficient.)

C5: |MINIMAL_BOUNDARY last_q_harm - WITH_BOUNDARY last_q_harm| < THRESH_HARM_MARGIN (both seeds).
    (Minimal boundary matches full boundary within margin -- minimal contract adequate.)

C6: MINIMAL_BOUNDARY mean_pre_post_corr < THRESH_CORR_MAX (both seeds).
    (Soft gate achieves the same signal separation as full binary boundary.)

Data quality gate:
C7: n_harm_events >= THRESH_MIN_HARM_EVENTS per condition per seed.
    (Sufficient harm contacts for reliable pre_post_corr estimation.)

PASS: C1 + C2 + C3 + C4 + C5 + C6 + C7
  => Q-015 partially resolved: the minimal boundary contract is a soft learned gate
     (not necessarily a strict binary token). MECH-061 replicated at V3 scale.
     WITH_BOUNDARY and MINIMAL_BOUNDARY both outperform BLENDED.

PARTIAL_FULL_NEEDED: C1 + C2 + C3 + NOT C4 (or NOT C5 or NOT C6) + C7
  => WITH_BOUNDARY is necessary at V3 scale; soft learned gate insufficient.
     Strict binary boundary token is the minimal sufficient contract.
     Q-015 partially resolved: binary boundary required.

PARTIAL_V3_FAIL: NOT C1 + C7
  => MECH-061 does not replicate at V3 scale with reafference-corrected z_world.
     Boundary token effect may depend on V1 architecture. Architectural investigation
     needed. Evidence weakens MECH-061 applicability to V3.

FAIL: NOT C7 (insufficient harm events) or training diverged.
  => Not informative for Q-015.

Conditions
----------
WITH_BOUNDARY:
  Binary commit gate g in {0, 1}. g=0 for pre-commit steps, g=1 for post-commit.
  A simple episode-phase split is used: the first COMMIT_STEP_FRAC of each
  episode is treated as "pre-commit" (g=0), the remainder as "post-commit" (g=1).
  Pre-commit loss: world_prediction_loss only (MSE(z_world_predicted, z_world_actual)).
  Post-commit loss: harm_eval_loss only (MSE(harm_pred, harm_actual)).
  Attribution measured by: Pearson |r| between pre-commit prediction errors
  and post-commit harm errors (collected per episode over last MEASUREMENT_EPISODES).

BLENDED:
  No gate. Combined loss at every step:
    total_loss = ALPHA_BLEND * world_prediction_loss + (1-ALPHA_BLEND) * harm_eval_loss
  Same network architecture as WITH_BOUNDARY. Same measurement protocol.

MINIMAL_BOUNDARY:
  Soft gate g = sigmoid(gating_net(z_self, z_world)) in [0, 1].
  Loss: g * harm_eval_loss + (1-g) * world_prediction_loss.
  The gating network is a 2-layer MLP taking concat(z_self, z_world) as input.
  The gate is trained jointly with the rest of the network via gradient descent.
  No external scheduling or phase information is provided to the gate.
  gate_entropy = -mean(g*log(g+eps) + (1-g)*log(1-g+eps)) across episode steps.
  Low entropy = near-binary gate (agent discovered binary-like split).

Architecture (shared across conditions):
  Encoder: Linear(world_obs_dim, world_dim) + LayerNorm -> z_world
  Self-encoder: Linear(self_obs_dim, self_dim) + LayerNorm -> z_self
  E1 world predictor: Linear(world_dim + action_dim, world_dim) -> z_world_next
  E3 harm predictor:  Linear(world_dim + self_dim, 1) -> harm_pred
  Gating network (MINIMAL_BOUNDARY only): MLP(world_dim + self_dim, 32, 1) -> scalar

Seeds:    [42, 123] (matched -- same env seed per condition)
Env:      CausalGridWorldV2 size=8, num_hazards=4, num_resources=2, hazard_harm=0.05,
          env_drift_interval=5, env_drift_prob=0.3
          (moderate hazard density; sufficient pre-commit and post-commit steps)
Protocol: TOTAL_EPISODES=400
          COMMIT_STEP_FRAC=0.5  (first 50% of each episode = pre-commit)
          STEPS_PER_EPISODE=200
          MEASUREMENT_EPISODES=100 (last 100 episodes for metric collection)
Estimated runtime:
  3 conditions x 2 seeds x 400 eps x 0.10 min/ep = ~240 min Mac
  (+10% overhead) => ~264 min Mac
  (~5x slower on Daniel-PC => ~1320 min; assign to Mac)
"""

import sys
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


EXPERIMENT_TYPE = "v3_exq_155_q015_commit_boundary_minimal_contract_pair"
CLAIM_IDS = ["Q-015"]

# ---------------------------------------------------------------------------
# Pre-registered thresholds
# ---------------------------------------------------------------------------
THRESH_HARM_MARGIN          = 0.05   # C1/C4/C5: policy improvement / equivalence margin
THRESH_CORR_MAX             = 0.30   # C2/C6: maximum acceptable |pre_post_corr|
THRESH_CORR_MIN_BLENDED     = 0.25   # C3: BLENDED must show this level of correlation
THRESH_MIN_HARM_EVENTS      = 30     # C7: minimum harm events per condition per seed

# ---------------------------------------------------------------------------
# Protocol constants
# ---------------------------------------------------------------------------
TOTAL_EPISODES          = 400    # total training episodes
MEASUREMENT_EPISODES    = 100    # last N episodes used for metric measurement
STEPS_PER_EPISODE       = 200
COMMIT_STEP_FRAC        = 0.5    # fraction of each episode that is "pre-commit"
ALPHA_BLEND             = 0.5    # BLENDED mixing coefficient
LR                      = 3e-4
GATE_ENTROPY_EPS        = 1e-6

SEEDS      = [42, 123]
CONDITIONS = ["WITH_BOUNDARY", "BLENDED", "MINIMAL_BOUNDARY"]

# ---------------------------------------------------------------------------
# Model constants
# ---------------------------------------------------------------------------
WORLD_OBS_DIM = 250   # CausalGridWorldV2 size=8 obs_dict["world_state"] dim
SELF_OBS_DIM  = 12    # agent state obs_dict["body_state"] dim (CausalGridWorldV2 body_obs_dim=12)
ACTION_DIM    = 5
WORLD_DIM     = 32
SELF_DIM      = 16
HIDDEN_DIM    = 32


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


class E1WorldPredictor(nn.Module):
    """Simplified E1: Linear(z_world + action) -> z_world_next (pre-commit path)."""
    def __init__(self, world_dim: int, action_dim: int):
        super().__init__()
        self.fc = nn.Linear(world_dim + action_dim, world_dim)

    def forward(self, z_world: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.fc(torch.cat([z_world, action], dim=-1))


class E3HarmPredictor(nn.Module):
    """Simplified E3: Linear(z_world + z_self) -> harm_pred (post-commit path)."""
    def __init__(self, world_dim: int, self_dim: int):
        super().__init__()
        self.fc = nn.Linear(world_dim + self_dim, 1)

    def forward(self, z_world: torch.Tensor, z_self: torch.Tensor) -> torch.Tensor:
        return self.fc(torch.cat([z_world, z_self], dim=-1))


class GatingNetwork(nn.Module):
    """
    MINIMAL_BOUNDARY soft gate.
    Input: concat(z_world, z_self) -> scalar in [0, 1] via sigmoid.
    0 = pre-commit (route to E1 world prediction loss).
    1 = post-commit (route to E3 harm evaluation loss).
    """
    def __init__(self, world_dim: int, self_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(world_dim + self_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z_world: torch.Tensor, z_self: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(torch.cat([z_world, z_self], dim=-1)))


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _pearson_r(xs: List[float], ys: List[float]) -> float:
    """Pearson correlation. Returns 0.0 if degenerate."""
    n = len(xs)
    if n < 2:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sx  = (sum((x - mx) ** 2 for x in xs) / n) ** 0.5
    sy  = (sum((y - my) ** 2 for y in ys) / n) ** 0.5
    if sx < 1e-8 or sy < 1e-8:
        return 0.0
    return num / (n * sx * sy)


def _get_obs_tensor(obs_dict: dict, key: str, fallback_dim: int) -> torch.Tensor:
    """Safely extract a tensor from obs_dict, zero-padding if needed."""
    raw = obs_dict.get(key)
    if raw is None:
        return torch.zeros(fallback_dim)
    t = torch.tensor(raw, dtype=torch.float32)
    # Pad or truncate to fallback_dim
    if t.shape[0] < fallback_dim:
        t = F.pad(t, (0, fallback_dim - t.shape[0]))
    elif t.shape[0] > fallback_dim:
        t = t[:fallback_dim]
    return t


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

    env = CausalGridWorldV2(
        size=8,
        num_hazards=4,
        num_resources=2,
        hazard_harm=0.05,
        env_drift_interval=5,
        env_drift_prob=0.3,
        seed=seed,
    )

    # Build models
    world_enc = WorldEncoder(WORLD_OBS_DIM, WORLD_DIM)
    self_enc  = SelfEncoder(SELF_OBS_DIM, SELF_DIM)
    e1_pred   = E1WorldPredictor(WORLD_DIM, ACTION_DIM)
    e3_pred   = E3HarmPredictor(WORLD_DIM, SELF_DIM)

    params = (
        list(world_enc.parameters())
        + list(self_enc.parameters())
        + list(e1_pred.parameters())
        + list(e3_pred.parameters())
    )

    gate_net = None
    if condition == "MINIMAL_BOUNDARY":
        gate_net = GatingNetwork(WORLD_DIM, SELF_DIM, HIDDEN_DIM)
        params += list(gate_net.parameters())

    optimizer = optim.Adam(params, lr=lr)

    commit_step = int(steps_per_episode * COMMIT_STEP_FRAC)

    # Measurement buffers (last measurement_episodes)
    measurement_start = total_episodes - measurement_episodes

    last_q_harm_vals:    List[float] = []
    pre_commit_errors:   List[float] = []
    post_commit_errors:  List[float] = []
    gate_entropy_vals:   List[float] = []
    harm_event_count:    int = 0

    if dry_run:
        total_episodes = 2
        measurement_start = 0

    _, obs_dict = env.reset()

    for ep in range(total_episodes):
        ep_harm = 0.0
        ep_steps = 0

        for step in range(steps_per_episode):
            obs_world = _get_obs_tensor(obs_dict, "world_state", WORLD_OBS_DIM)
            obs_self  = _get_obs_tensor(obs_dict, "body_state", SELF_OBS_DIM)

            z_world = world_enc(obs_world)
            z_self  = self_enc(obs_self)

            # Random action
            action_idx = random.randint(0, ACTION_DIM - 1)
            action = torch.zeros(ACTION_DIM)
            action[action_idx] = 1.0

            # Step env
            _, _, done, _, obs_dict_next = env.step(action.unsqueeze(0))

            obs_world_next = _get_obs_tensor(obs_dict_next, "world_state", WORLD_OBS_DIM)
            z_world_next_actual = world_enc(obs_world_next).detach()

            harm_signal = float(obs_dict_next.get("harm_signal", 0.0))
            harm_actual = torch.tensor([harm_signal], dtype=torch.float32)
            harm_pred   = e3_pred(z_world, z_self)

            z_world_next_pred = e1_pred(z_world, action)
            world_pred_error  = F.mse_loss(z_world_next_pred, z_world_next_actual)
            harm_eval_error   = F.mse_loss(harm_pred, harm_actual)

            # Compute loss according to condition
            in_measurement = ep >= measurement_start

            if condition == "WITH_BOUNDARY":
                is_pre_commit = step < commit_step
                if is_pre_commit:
                    total_loss = world_pred_error
                    pre_commit_err = float(world_pred_error.item())
                    post_commit_err = 0.0
                else:
                    total_loss = harm_eval_error
                    pre_commit_err = 0.0
                    post_commit_err = float(harm_eval_error.item())

                if in_measurement:
                    if is_pre_commit:
                        pre_commit_errors.append(pre_commit_err)
                        post_commit_errors.append(post_commit_err)
                    else:
                        pre_commit_errors.append(pre_commit_err)
                        post_commit_errors.append(post_commit_err)

            elif condition == "BLENDED":
                total_loss = (
                    ALPHA_BLEND * world_pred_error
                    + (1.0 - ALPHA_BLEND) * harm_eval_error
                )
                if in_measurement:
                    pre_commit_errors.append(float(world_pred_error.item()))
                    post_commit_errors.append(float(harm_eval_error.item()))

            else:  # MINIMAL_BOUNDARY
                g = gate_net(z_world.detach(), z_self.detach())  # [1]
                g_val = float(g.item())
                total_loss = (
                    g * harm_eval_error + (1.0 - g) * world_pred_error
                )
                ent = -(
                    g_val * torch.log(g + GATE_ENTROPY_EPS)
                    + (1.0 - g_val) * torch.log(1.0 - g + GATE_ENTROPY_EPS)
                )
                if in_measurement:
                    pre_commit_errors.append(float(world_pred_error.item()))
                    post_commit_errors.append(float(harm_eval_error.item()))
                    gate_entropy_vals.append(float(ent.item()))

            optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()

            ep_harm  += harm_signal
            ep_steps += 1

            if harm_signal > 0.0 and in_measurement:
                harm_event_count += 1

            if done:
                _, obs_dict = env.reset()
            else:
                obs_dict = obs_dict_next

        if in_measurement:
            last_q_harm_vals.append(ep_harm / max(ep_steps, 1))

        if ep % 100 == 0:
            avg_harm = ep_harm / max(ep_steps, 1)
            print(
                f"  [{condition}] seed={seed} ep={ep}/{total_episodes}"
                f" ep_harm={avg_harm:.5f}",
                flush=True,
            )

    last_q_harm    = float(sum(last_q_harm_vals) / max(len(last_q_harm_vals), 1))
    pre_post_corr  = abs(_pearson_r(pre_commit_errors, post_commit_errors))
    mean_gate_ent  = float(sum(gate_entropy_vals) / max(len(gate_entropy_vals), 1))

    print(
        f"  [{condition}] seed={seed} last_q_harm={last_q_harm:.5f}"
        f" pre_post_corr={pre_post_corr:.4f}"
        f" harm_events={harm_event_count}",
        flush=True,
    )
    if condition == "MINIMAL_BOUNDARY":
        print(f"  [{condition}] seed={seed} mean_gate_entropy={mean_gate_ent:.5f}", flush=True)

    return {
        "condition":      condition,
        "seed":           seed,
        "last_q_harm":    last_q_harm,
        "pre_post_corr":  pre_post_corr,
        "n_harm_events":  harm_event_count,
        "mean_gate_entropy": mean_gate_ent,
    }


# ---------------------------------------------------------------------------
# Criterion evaluation
# ---------------------------------------------------------------------------

def _evaluate_criteria(
    results_by_condition: Dict[str, List[Dict]],
) -> Dict[str, bool]:
    """Evaluate pre-registered criteria across all conditions and seeds."""

    wb   = results_by_condition["WITH_BOUNDARY"]
    bl   = results_by_condition["BLENDED"]
    mb   = results_by_condition["MINIMAL_BOUNDARY"]
    n_s  = len(SEEDS)

    # C1: WITH_BOUNDARY last_q_harm < BLENDED last_q_harm - THRESH_HARM_MARGIN (both seeds)
    c1 = all(
        wb[i]["last_q_harm"] < bl[i]["last_q_harm"] - THRESH_HARM_MARGIN
        for i in range(n_s)
    )

    # C2: WITH_BOUNDARY mean_pre_post_corr < THRESH_CORR_MAX (both seeds)
    c2 = all(wb[i]["pre_post_corr"] < THRESH_CORR_MAX for i in range(n_s))

    # C3: BLENDED mean_pre_post_corr > THRESH_CORR_MIN_BLENDED (both seeds)
    c3 = all(bl[i]["pre_post_corr"] > THRESH_CORR_MIN_BLENDED for i in range(n_s))

    # C4: MINIMAL_BOUNDARY last_q_harm < BLENDED last_q_harm - THRESH_HARM_MARGIN (both seeds)
    c4 = all(
        mb[i]["last_q_harm"] < bl[i]["last_q_harm"] - THRESH_HARM_MARGIN
        for i in range(n_s)
    )

    # C5: |MINIMAL_BOUNDARY last_q_harm - WITH_BOUNDARY last_q_harm| < THRESH_HARM_MARGIN (both seeds)
    c5 = all(
        abs(mb[i]["last_q_harm"] - wb[i]["last_q_harm"]) < THRESH_HARM_MARGIN
        for i in range(n_s)
    )

    # C6: MINIMAL_BOUNDARY mean_pre_post_corr < THRESH_CORR_MAX (both seeds)
    c6 = all(mb[i]["pre_post_corr"] < THRESH_CORR_MAX for i in range(n_s))

    # C7: n_harm_events >= THRESH_MIN_HARM_EVENTS per condition per seed
    c7 = all(
        r["n_harm_events"] >= THRESH_MIN_HARM_EVENTS
        for cond_results in results_by_condition.values()
        for r in cond_results
    )

    return {
        "C1_with_boundary_beats_blended_harm":    c1,
        "C2_with_boundary_signal_separation":      c2,
        "C3_blended_correlated":                   c3,
        "C4_minimal_boundary_beats_blended_harm":  c4,
        "C5_minimal_boundary_matches_full":        c5,
        "C6_minimal_boundary_signal_separation":   c6,
        "C7_sufficient_harm_events":               c7,
    }


def _determine_outcome(criteria: Dict[str, bool]) -> str:
    c1 = criteria["C1_with_boundary_beats_blended_harm"]
    c2 = criteria["C2_with_boundary_signal_separation"]
    c3 = criteria["C3_blended_correlated"]
    c4 = criteria["C4_minimal_boundary_beats_blended_harm"]
    c5 = criteria["C5_minimal_boundary_matches_full"]
    c6 = criteria["C6_minimal_boundary_signal_separation"]
    c7 = criteria["C7_sufficient_harm_events"]

    if not c7:
        return "FAIL"

    # Full PASS: all criteria met -- minimal contract sufficient
    if c1 and c2 and c3 and c4 and c5 and c6:
        return "PASS"

    # PARTIAL_FULL_NEEDED: boundary helps but minimal gate not sufficient
    if c1 and c2 and c3 and not (c4 and c5 and c6):
        return "PARTIAL_FULL_NEEDED"

    # PARTIAL_V3_FAIL: MECH-061 does not replicate at V3 scale
    if not c1:
        return "PARTIAL_V3_FAIL"

    return "PARTIAL"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_experiment(dry_run: bool = False) -> Dict:
    """Run all conditions and compile the result pack."""
    print("=== V3-EXQ-155: Q-015 Commit Boundary Minimal Contract Pair ===", flush=True)
    print(f"Conditions: {CONDITIONS}  Seeds: {SEEDS}", flush=True)
    print("Pre-registered thresholds:", flush=True)
    print(f"  C1/C4/C5 THRESH_HARM_MARGIN         = {THRESH_HARM_MARGIN}", flush=True)
    print(f"  C2/C6    THRESH_CORR_MAX             = {THRESH_CORR_MAX}", flush=True)
    print(f"  C3       THRESH_CORR_MIN_BLENDED     = {THRESH_CORR_MIN_BLENDED}", flush=True)
    print(f"  C7       THRESH_MIN_HARM_EVENTS      = {THRESH_MIN_HARM_EVENTS}", flush=True)
    print(
        f"  TOTAL_EPISODES={TOTAL_EPISODES}"
        f"  MEASUREMENT_EPISODES={MEASUREMENT_EPISODES}",
        flush=True,
    )

    results_by_condition: Dict[str, List[Dict]] = {c: [] for c in CONDITIONS}

    for condition in CONDITIONS:
        print(f"\n=== Condition: {condition} ===", flush=True)
        for seed in SEEDS:
            result = _run_condition(
                seed=seed,
                condition=condition,
                total_episodes=TOTAL_EPISODES,
                measurement_episodes=MEASUREMENT_EPISODES,
                steps_per_episode=STEPS_PER_EPISODE,
                lr=LR,
                dry_run=dry_run,
            )
            results_by_condition[condition].append(result)

    print("\n=== Evaluating criteria ===", flush=True)
    criteria = _evaluate_criteria(results_by_condition)
    outcome  = _determine_outcome(criteria)

    for k, v in criteria.items():
        print(f"  {k}: {'PASS' if v else 'FAIL'}", flush=True)
    print(f"Overall outcome: {outcome}", flush=True)

    # Summary metrics: mean over seeds per condition
    def _mean_seeds(cond: str, key: str) -> float:
        vals = [r[key] for r in results_by_condition[cond]]
        return float(sum(vals) / max(len(vals), 1))

    summary_metrics: Dict = {}
    for cond in CONDITIONS:
        prefix = cond.lower()
        summary_metrics[f"{prefix}_last_q_harm"]       = _mean_seeds(cond, "last_q_harm")
        summary_metrics[f"{prefix}_pre_post_corr"]     = _mean_seeds(cond, "pre_post_corr")
        summary_metrics[f"{prefix}_n_harm_events"]     = _mean_seeds(cond, "n_harm_events")
        summary_metrics[f"{prefix}_mean_gate_entropy"] = _mean_seeds(cond, "mean_gate_entropy")

    # Pairwise deltas
    summary_metrics["delta_harm_with_vs_blended"]    = (
        summary_metrics["blended_last_q_harm"]
        - summary_metrics["with_boundary_last_q_harm"]
    )
    summary_metrics["delta_harm_minimal_vs_blended"] = (
        summary_metrics["blended_last_q_harm"]
        - summary_metrics["minimal_boundary_last_q_harm"]
    )
    summary_metrics["delta_harm_minimal_vs_full"]    = (
        summary_metrics["with_boundary_last_q_harm"]
        - summary_metrics["minimal_boundary_last_q_harm"]
    )

    # Evidence direction
    if outcome == "PASS":
        evidence_direction = "supports"
        guidance = "minimal_soft_gate_sufficient_binary_boundary_not_required"
    elif outcome == "PARTIAL_FULL_NEEDED":
        evidence_direction = "mixed"
        guidance = "full_binary_boundary_required_soft_gate_insufficient_at_v3_scale"
    elif outcome == "PARTIAL_V3_FAIL":
        evidence_direction = "weakens"
        guidance = "mech_061_does_not_replicate_at_v3_scale_investigation_needed"
    elif outcome == "PARTIAL":
        evidence_direction = "mixed"
        guidance = "partial_evidence_see_criteria"
    else:  # FAIL
        evidence_direction = "mixed"
        guidance = "insufficient_harm_events_or_training_divergence"

    run_id = (
        "v3_exq_155_q015_commit_boundary_minimal_contract_"
        + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + "_v3"
    )

    pack = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "claim_ids": CLAIM_IDS,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "outcome": outcome,
        "evidence_direction": evidence_direction,
        "evidence_class": "discriminative_pair",
        "guidance": guidance,
        "criteria": criteria,
        "pre_registered_thresholds": {
            "THRESH_HARM_MARGIN":      THRESH_HARM_MARGIN,
            "THRESH_CORR_MAX":         THRESH_CORR_MAX,
            "THRESH_CORR_MIN_BLENDED": THRESH_CORR_MIN_BLENDED,
            "THRESH_MIN_HARM_EVENTS":  THRESH_MIN_HARM_EVENTS,
        },
        "summary_metrics": summary_metrics,
        "protocol": {
            "total_episodes":       TOTAL_EPISODES,
            "measurement_episodes": MEASUREMENT_EPISODES,
            "steps_per_episode":    STEPS_PER_EPISODE,
            "commit_step_frac":     COMMIT_STEP_FRAC,
            "alpha_blend":          ALPHA_BLEND,
            "lr":                   LR,
        },
        "seeds": SEEDS,
        "scenario": (
            "Three-condition commit boundary minimal contract test:"
            " WITH_BOUNDARY (binary gate, episode phase split, pre=world_pred_loss only,"
            " post=harm_eval_loss only),"
            " BLENDED (no gate, alpha=0.5 mixed loss at every step),"
            " MINIMAL_BOUNDARY (learned soft gate g=sigmoid(MLP(z_world,z_self)),"
            " loss = g*harm_eval + (1-g)*world_pred, no external phase info)."
            " All conditions: random action policy, 400 total episodes,"
            " 100 measurement episodes (last quartile)."
            " Metrics: last_q_harm (policy quality), pre_post_corr (signal separation),"
            " gate_entropy (MINIMAL_BOUNDARY only -- convergence to near-binary gate)."
            " 3 conditions x 2 seeds = 6 cells."
            " CausalGridWorldV2 size=8 num_hazards=4 num_resources=2 hazard_harm=0.05"
            " env_drift_interval=5 env_drift_prob=0.3."
        ),
        "interpretation": (
            "PASS => Q-015 partially resolved: the minimal sufficient boundary contract"
            " is a soft learned gate, not a strict binary token."
            " MECH-061 replicated at V3 scale. REE can implement a lighter boundary mechanism."
            " PARTIAL_FULL_NEEDED => Full binary boundary is necessary at V3 scale."
            " Soft learned gate cannot discover the pre/post split reliably."
            " Q-015 partially resolved: strict binary token is the minimal contract."
            " PARTIAL_V3_FAIL => MECH-061 boundary benefit does not replicate at V3 scale."
            " SD-007 reafference-corrected z_world may change the error signal properties"
            " in ways that eliminate the boundary benefit seen at V1 scale."
            " FAIL => insufficient harm events or training divergence; not informative."
        ),
        "per_seed_results": {cond: results_by_condition[cond] for cond in CONDITIONS},
        "dry_run": dry_run,
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


if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv
    result = run_experiment(dry_run=dry_run)
    print(f"Done. Outcome: {result['outcome']}", flush=True)
