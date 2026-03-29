#!/opt/local/bin/python3
"""
V3-EXQ-156 -- Q-016: What arbitration policy best resolves tri-loop gate
              conflicts without coupling collapse?

Claim:    Q-016
Proposal: EXP-0106 (EVB-0080)

Q-016 asks:
  "What arbitration policy best resolves tri-loop gate conflicts without
  coupling collapse?"

  Background (MECH-062):
    E3 uses coordinated tri-loop gating (motor, cognitive-set, motivational)
    as the pre-commit eligibility layer. The three BG-like loops (ARC-021) emit
    incommensurable error signals (MECH-069). When they conflict -- one loop
    says "go", another says "stop" -- some arbitration policy must resolve the
    conflict and determine whether the commit gate opens.

  "Coupling collapse" is the failure mode where the arbitration policy allows
    signals from one loop to bleed into another, causing the three loops to
    become functionally equivalent (i.e., they no longer represent distinct
    eligibility criteria). The discriminative signature is:
      - collapsed: all three loop_gate values are near-identical regardless of
        environmental conditions (inter_loop_spread ~0)
      - intact: each loop gate responds distinctively to its own signal class
        (motor gate high during action execution; cognitive gate high during
        world prediction accuracy; motivational gate high during harm proximity)

  This experiment compares three arbitration policies under conflict conditions
  designed to generate opposing gate signals across the three loops:

  Condition A: PRIORITY_HARM
    Hierarchical: motivational gate always overrides the others when conflict
    is detected. Conflict = any two loops disagree by > CONFLICT_THRESH.
    Policy: g_commit = g_motivational if |g_motor - g_cognitive| > CONFLICT_THRESH
            else mean(g_motor, g_cognitive, g_motivational)
    Rationale: mirrors ACC/vmPFC top-down veto in affective-motivational scenarios.
    Risk: if motivational gate is always dominant, motor/cognitive loops stop
    contributing meaningful signal -> coupling collapse into single loop.

  Condition B: WEIGHTED_SUM
    Linear combination: g_commit = w_motor * g_motor + w_cog * g_cognitive
                                   + w_mot * g_motivational
    Weights fixed: w_motor=0.2, w_cog=0.3, w_mot=0.5
    (motivational loop given higher weight but does not override).
    Rationale: graded mixing preserves all three signals in the output but
    introduces cross-loop coupling. Risk: motor and cognitive loops become
    proxies for the weighted output rather than independent eligibility tests.

  Condition C: WINNER_TAKE_ALL
    Competitive inhibition: g_commit = g_loop where loop = argmax(g_motor,
    g_cognitive, g_motivational). Hard winner; other loops contribute 0.
    Rationale: closest to biological BG direct-pathway competition. Each loop
    can veto the others but the strongest signal wins absolutely.
    Risk: minority-loop information is entirely discarded.

  Discriminative question:
    (i)  Which policy achieves the lowest coupling collapse rate (measured as
         inter_loop_spread > THRESH_SPREAD, a proxy for loop independence)?
    (ii) Which policy achieves the best harm avoidance while maintaining
         loop independence (no coupling collapse)?
    (iii) Does any policy achieve both? (PASS criterion)

  Scientific meaning:
    PASS with WINNER_TAKE_ALL:
      => Competitive inhibition is the sufficient arbitration mechanism. The
         loops remain independent because hard suppression of non-winning loops
         prevents cross-loop contamination. Q-016 partially resolved: WTA is the
         optimal policy for avoiding coupling collapse.
    PASS with PRIORITY_HARM (and NOT WINNER_TAKE_ALL):
      => Hierarchical harm priority achieves harm avoidance without collapse
         because the motivational loop only activates when harm is genuinely
         proximate (low base-rate). Q-016 partially resolved: PRIORITY_HARM is
         the preferred policy when harm signals are sparse.
    PASS with WEIGHTED_SUM:
      => Surprising result: coupling collapse does not occur under weighted
         combination at current experiment scale. Replication at larger scale needed.
    FAIL_ALL_COLLAPSE:
      => All policies produce coupling collapse (inter_loop_spread -> 0). The
         tri-loop architecture itself may require additional structural separation
         (e.g., separate loss targets per loop) before arbitration policy matters.
    FAIL_DATA:
      => Insufficient conflict events for reliable metric estimation.

  The key metrics:
    1. harm_rate (harm contacts per step in last quartile) -- policy performance
    2. inter_loop_spread (mean pairwise absolute difference between loop gate
       values at conflict steps) -- loop independence; higher = more intact
    3. commit_gate_variance (variance of g_commit over time) -- gate informativeness;
       a gate stuck near 0 or 1 provides no discrimination
    4. n_conflict_steps -- count of steps where loop disagreement > CONFLICT_THRESH
    5. winner_loop_entropy (WINNER_TAKE_ALL only) -- entropy of which loop wins;
       low = one loop dominates all decisions

Pre-registered thresholds
--------------------------
C1: Best-policy harm_rate < BLENDED baseline harm_rate - THRESH_HARM_MARGIN (both seeds).
    "Best policy" = whichever policy achieves lowest harm_rate while satisfying C2.
    (The winning arbitration policy outperforms no-gate blending on harm avoidance.)

C2: Best-policy mean inter_loop_spread > THRESH_SPREAD_MIN (both seeds).
    (Winning policy does NOT produce coupling collapse -- loops remain distinct.)

C3: WEIGHTED_SUM inter_loop_spread < WINNER_TAKE_ALL inter_loop_spread (both seeds).
    (WTA preserves more inter-loop variance than weighted blending -- expected from theory.)

C4: n_conflict_steps >= THRESH_MIN_CONFLICT_STEPS per condition per seed.
    (Data quality gate: sufficient conflict events for reliable metric estimation.)

C5: Best-policy commit_gate_variance > THRESH_GATE_VAR_MIN (both seeds).
    (Gate is informative -- not stuck at a constant value.)

PASS: C1 + C2 + C3 + C4 + C5
  => Winning arbitration policy identified; loop independence maintained.
     Q-016 partially resolved: that policy is the answer to the open question.

PARTIAL_NO_WINNER: C4 + NOT C1 or NOT C2
  => No policy achieves both harm avoidance and loop independence.
     Structural separation (separate loss targets per loop) needed before
     arbitration policy can be discriminated. Q-016 remains open.

PARTIAL_COLLAPSE_ONLY: C3 + NOT C2 for all policies
  => All policies collapse. Architecture-level change needed.

FAIL: NOT C4
  => Insufficient conflict events. Increase conflict_prob or re-run.

Conditions
----------
Shared architecture (all conditions):
  Encoder: Linear(world_obs_dim, world_dim) + LayerNorm -> z_world
  Self-encoder: Linear(self_obs_dim, self_dim) + LayerNorm -> z_self
  Motor loop gate (g_motor): Linear(z_self, 1) -> sigmoid (responds to body state)
  Cognitive loop gate (g_cog): Linear(z_world, 1) -> sigmoid (responds to world state)
  Motivational loop gate (g_mot): Linear(z_world + z_self, 1) -> sigmoid (harm proximity)
  Harm predictor: Linear(z_world + z_self, 1) -> harm_pred
  World predictor: Linear(z_world + action, z_world) -> z_world_next (E1 analog)

  Conflict is operationalised as: |g_motor - g_cog| > CONFLICT_THRESH
  (motor and cognitive loops disagree; motivational loop must arbitrate)

  Training:
    Loss = harm_eval_loss (MSE(harm_pred, harm_actual)) weighted by g_commit
    + world_pred_loss (MSE(z_world_next_pred, z_world_next_actual)) * (1 - g_commit)
    + gate_entropy_regularisation * GATE_REG (small term to prevent gate saturation)

  The three gate networks are trained jointly with the rest of the model.
  No external eligibility signal is provided -- the gates must learn from
  the downstream task loss.

PRIORITY_HARM: g_commit = g_mot if conflict else mean(g_motor, g_cog, g_mot)
WEIGHTED_SUM:  g_commit = 0.2 * g_motor + 0.3 * g_cog + 0.5 * g_mot
WINNER_TAKE_ALL: g_commit = gate with maximum value among the three

Baseline (BLENDED): no gate at all -- combined loss at every step with fixed weights.
  harm_eval_loss * 0.5 + world_pred_loss * 0.5
  Used to benchmark harm_rate (C1).

Seeds:    [42, 123] (matched -- same env seed per condition)
Env:      CausalGridWorldV2 size=8, num_hazards=5, num_resources=2, hazard_harm=0.1,
          env_drift_interval=5, env_drift_prob=0.4
          (higher hazard density + drift -> more genuine loop conflicts)
Protocol: TOTAL_EPISODES=500
          MEASUREMENT_EPISODES=125 (last 125 episodes; last quartile)
          STEPS_PER_EPISODE=200
Estimated runtime:
  4 conditions x 2 seeds x 500 eps x 0.10 min/ep = ~400 min Mac
  (+10% overhead) => ~440 min Mac (~44 min Daniel-PC -- assign to Mac)
"""

import sys
import math
import random
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig


EXPERIMENT_TYPE = "v3_exq_156_q016_tri_loop_arbitration_policy_pair"
CLAIM_IDS = ["Q-016"]

# ---------------------------------------------------------------------------
# Pre-registered thresholds
# ---------------------------------------------------------------------------
THRESH_HARM_MARGIN        = 0.03   # C1: harm improvement over BLENDED baseline
THRESH_SPREAD_MIN         = 0.05   # C2: minimum inter_loop_spread for non-collapse
THRESH_GATE_VAR_MIN       = 0.01   # C5: minimum commit_gate_variance for informativeness
THRESH_MIN_CONFLICT_STEPS = 50     # C4: minimum conflict steps per condition per seed
CONFLICT_THRESH           = 0.15   # Threshold for declaring motor/cog gate conflict

# ---------------------------------------------------------------------------
# Protocol constants
# ---------------------------------------------------------------------------
TOTAL_EPISODES          = 500
MEASUREMENT_EPISODES    = 125   # last N episodes = last quartile
STEPS_PER_EPISODE       = 200
GATE_REG                = 1e-4  # gate entropy regularisation weight

# Weighted sum policy weights
W_MOTOR = 0.2
W_COG   = 0.3
W_MOT   = 0.5

LR        = 3e-4
SEEDS     = [42, 123]
CONDITIONS = ["PRIORITY_HARM", "WEIGHTED_SUM", "WINNER_TAKE_ALL", "BLENDED"]

# ---------------------------------------------------------------------------
# Model constants
# ---------------------------------------------------------------------------
WORLD_OBS_DIM = 250   # CausalGridWorldV2 size=8 world_state dim
SELF_OBS_DIM  = 12    # CausalGridWorldV2 body_state dim
ACTION_DIM    = 5
WORLD_DIM     = 32
SELF_DIM      = 16


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


class LoopGate(nn.Module):
    """
    Single BG-like loop gate: Linear(input_dim, 1) -> sigmoid.
    The three loop gates differ only in their input modality.
    """
    def __init__(self, input_dim: int):
        super().__init__()
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.fc(x))


class HarmPredictor(nn.Module):
    """E3 analog: Linear(z_world + z_self, 1) -> harm scalar."""
    def __init__(self, world_dim: int, self_dim: int):
        super().__init__()
        self.fc = nn.Linear(world_dim + self_dim, 1)

    def forward(self, z_world: torch.Tensor, z_self: torch.Tensor) -> torch.Tensor:
        return self.fc(torch.cat([z_world, z_self], dim=-1))


class WorldPredictor(nn.Module):
    """E1 analog: Linear(z_world + action, z_world) -> z_world_next."""
    def __init__(self, world_dim: int, action_dim: int):
        super().__init__()
        self.fc = nn.Linear(world_dim + action_dim, world_dim)

    def forward(self, z_world: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.fc(torch.cat([z_world, action], dim=-1))


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _get_obs_tensor(obs_dict: dict, key: str, fallback_dim: int) -> torch.Tensor:
    """Safely extract a tensor from obs_dict, zero-padding or truncating."""
    raw = obs_dict.get(key)
    if raw is None:
        return torch.zeros(fallback_dim)
    t = torch.tensor(raw, dtype=torch.float32)
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
        num_hazards=5,
        num_resources=2,
        hazard_harm=0.1,
        env_drift_interval=5,
        env_drift_prob=0.4,
        seed=seed,
    )

    # Build shared components
    world_enc   = WorldEncoder(WORLD_OBS_DIM, WORLD_DIM)
    self_enc    = SelfEncoder(SELF_OBS_DIM, SELF_DIM)
    harm_pred   = HarmPredictor(WORLD_DIM, SELF_DIM)
    world_pred  = WorldPredictor(WORLD_DIM, ACTION_DIM)

    # Three loop gates (BLENDED uses none)
    gate_motor  = None
    gate_cog    = None
    gate_mot    = None

    params = (
        list(world_enc.parameters())
        + list(self_enc.parameters())
        + list(harm_pred.parameters())
        + list(world_pred.parameters())
    )

    if condition != "BLENDED":
        gate_motor = LoopGate(SELF_DIM)          # motor: body-state input
        gate_cog   = LoopGate(WORLD_DIM)         # cognitive: world-state input
        gate_mot   = LoopGate(WORLD_DIM + SELF_DIM)  # motivational: joint
        params += (
            list(gate_motor.parameters())
            + list(gate_cog.parameters())
            + list(gate_mot.parameters())
        )

    optimizer = optim.Adam(params, lr=lr)

    measurement_start = total_episodes - measurement_episodes

    if dry_run:
        total_episodes    = 2
        measurement_start = 0

    # Measurement buffers
    harm_vals:          List[float] = []
    inter_loop_spreads: List[float] = []
    gate_commit_vals:   List[float] = []  # g_commit at each measurement step
    conflict_step_count: int = 0
    winner_loop_counts:  Dict[str, int] = {"motor": 0, "cog": 0, "mot": 0}

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

            # Step env
            _, _, done, _, obs_dict_next = env.step(action.unsqueeze(0))

            obs_world_next = _get_obs_tensor(obs_dict_next, "world_state", WORLD_OBS_DIM)
            z_world_next_actual = world_enc(obs_world_next).detach()

            harm_signal = float(obs_dict_next.get("harm_signal", 0.0))
            harm_actual = torch.tensor([harm_signal], dtype=torch.float32)

            harm_pred_val   = harm_pred(z_world, z_self)
            z_world_next_p  = world_pred(z_world, action)
            harm_eval_loss  = F.mse_loss(harm_pred_val, harm_actual)
            world_pred_loss = F.mse_loss(z_world_next_p, z_world_next_actual)

            # Compute gate values and g_commit
            if condition == "BLENDED":
                total_loss = 0.5 * harm_eval_loss + 0.5 * world_pred_loss
                g_commit_val = 0.5
                g_motor_val  = 0.5
                g_cog_val    = 0.5
                g_mot_val    = 0.5
                is_conflict  = False

            else:
                g_m = gate_motor(z_self.detach())          # [1]
                g_c = gate_cog(z_world.detach())           # [1]
                g_v = gate_mot(torch.cat([z_world.detach(), z_self.detach()], dim=-1))  # [1]

                g_motor_val = float(g_m.item())
                g_cog_val   = float(g_c.item())
                g_mot_val   = float(g_v.item())
                is_conflict = abs(g_motor_val - g_cog_val) > CONFLICT_THRESH

                if condition == "PRIORITY_HARM":
                    if is_conflict:
                        g_commit = g_v
                    else:
                        g_commit = (g_m + g_c + g_v) / 3.0
                    g_commit_val = float(g_commit.item())

                elif condition == "WEIGHTED_SUM":
                    g_commit = W_MOTOR * g_m + W_COG * g_c + W_MOT * g_v
                    g_commit_val = float(g_commit.item())

                else:  # WINNER_TAKE_ALL
                    vals = [g_motor_val, g_cog_val, g_mot_val]
                    winner_idx = int(max(range(3), key=lambda i: vals[i]))
                    gates_t = [g_m, g_c, g_v]
                    g_commit = gates_t[winner_idx].detach()  # hard selection
                    g_commit_val = float(g_commit.item())
                    loop_names = ["motor", "cog", "mot"]
                    if in_measurement:
                        winner_loop_counts[loop_names[winner_idx]] += 1

                # Loss: g_commit gates harm path; (1-g_commit) gates world path
                # Use detached g_commit for loss weighting to prevent
                # degenerate gradient that drives g_commit to 0 or 1 trivially
                g_w = g_commit.detach() if condition == "WINNER_TAKE_ALL" else g_commit
                total_loss = (
                    g_w * harm_eval_loss
                    + (1.0 - g_w) * world_pred_loss
                )

                # Gate entropy regularisation: penalise saturated gates
                ent_motor = -(g_m * torch.log(g_m + 1e-6) + (1.0 - g_m) * torch.log(1.0 - g_m + 1e-6))
                ent_cog   = -(g_c * torch.log(g_c + 1e-6) + (1.0 - g_c) * torch.log(1.0 - g_c + 1e-6))
                ent_mot   = -(g_v * torch.log(g_v + 1e-6) + (1.0 - g_v) * torch.log(1.0 - g_v + 1e-6))
                gate_ent_reg = -GATE_REG * (ent_motor + ent_cog + ent_mot)  # negative = max entropy
                total_loss = total_loss + gate_ent_reg

            optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()

            ep_harm  += harm_signal
            ep_steps += 1

            if in_measurement:
                gate_commit_vals.append(g_commit_val)
                if is_conflict:
                    conflict_step_count += 1
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
                f" ep_harm={avg_harm:.5f}",
                flush=True,
            )

    harm_rate       = float(sum(harm_vals) / max(len(harm_vals), 1))
    inter_spread    = float(sum(inter_loop_spreads) / max(len(inter_loop_spreads), 1))
    gate_var        = float(
        sum((v - sum(gate_commit_vals) / max(len(gate_commit_vals), 1)) ** 2
            for v in gate_commit_vals) / max(len(gate_commit_vals), 1)
    )

    # Winner loop entropy (WTA only)
    total_wins = sum(winner_loop_counts.values())
    winner_ent = 0.0
    if condition == "WINNER_TAKE_ALL" and total_wins > 0:
        for cnt in winner_loop_counts.values():
            p = cnt / total_wins
            if p > 0:
                winner_ent -= p * math.log(p)

    print(
        f"  [{condition}] seed={seed} harm_rate={harm_rate:.5f}"
        f" inter_spread={inter_spread:.4f}"
        f" gate_var={gate_var:.5f}"
        f" conflict_steps={conflict_step_count}",
        flush=True,
    )
    if condition == "WINNER_TAKE_ALL":
        print(
            f"  [{condition}] seed={seed} winner_loop_counts={winner_loop_counts}"
            f" winner_entropy={winner_ent:.4f}",
            flush=True,
        )

    return {
        "condition":           condition,
        "seed":                seed,
        "harm_rate":           harm_rate,
        "inter_loop_spread":   inter_spread,
        "commit_gate_variance": gate_var,
        "n_conflict_steps":    conflict_step_count,
        "winner_loop_counts":  winner_loop_counts,
        "winner_loop_entropy": winner_ent,
    }


# ---------------------------------------------------------------------------
# Criterion evaluation
# ---------------------------------------------------------------------------

def _evaluate_criteria(
    results_by_condition: Dict[str, List[Dict]],
) -> Dict[str, bool]:
    """Evaluate pre-registered criteria across all conditions and seeds."""
    n_s = len(SEEDS)

    # Identify "best policy" = non-BLENDED condition with lowest mean harm_rate
    # that satisfies C2 (inter_loop_spread > THRESH_SPREAD_MIN, both seeds).
    policy_conditions = ["PRIORITY_HARM", "WEIGHTED_SUM", "WINNER_TAKE_ALL"]

    def _mean_hr(cond: str) -> float:
        return sum(r["harm_rate"] for r in results_by_condition[cond]) / n_s

    def _spread_ok(cond: str) -> bool:
        return all(
            r["inter_loop_spread"] > THRESH_SPREAD_MIN
            for r in results_by_condition[cond]
        )

    blended_hr = _mean_hr("BLENDED")

    # Find best policy satisfying C2
    best_policy = None
    best_hr = float("inf")
    for cond in policy_conditions:
        if _spread_ok(cond):
            hr = _mean_hr(cond)
            if hr < best_hr:
                best_hr = hr
                best_policy = cond

    # C1: best policy harm_rate < BLENDED - THRESH_HARM_MARGIN (both seeds)
    if best_policy is not None:
        c1 = all(
            r["harm_rate"] < blended_hr - THRESH_HARM_MARGIN
            for r in results_by_condition[best_policy]
        )
    else:
        c1 = False

    # C2: best policy inter_loop_spread > THRESH_SPREAD_MIN (both seeds)
    c2 = best_policy is not None  # already enforced in selection above

    # C3: WEIGHTED_SUM inter_loop_spread < WINNER_TAKE_ALL inter_loop_spread (both seeds)
    c3 = all(
        results_by_condition["WEIGHTED_SUM"][i]["inter_loop_spread"]
        < results_by_condition["WINNER_TAKE_ALL"][i]["inter_loop_spread"]
        for i in range(n_s)
    )

    # C4: n_conflict_steps >= THRESH_MIN_CONFLICT_STEPS per condition per seed
    c4 = all(
        r["n_conflict_steps"] >= THRESH_MIN_CONFLICT_STEPS
        for cond in policy_conditions
        for r in results_by_condition[cond]
    )

    # C5: best policy commit_gate_variance > THRESH_GATE_VAR_MIN (both seeds)
    if best_policy is not None:
        c5 = all(
            r["commit_gate_variance"] > THRESH_GATE_VAR_MIN
            for r in results_by_condition[best_policy]
        )
    else:
        c5 = False

    return {
        "C1_best_policy_beats_blended_harm":    c1,
        "C2_best_policy_no_coupling_collapse":  c2,
        "C3_wta_more_spread_than_weighted_sum": c3,
        "C4_sufficient_conflict_steps":         c4,
        "C5_gate_informative":                  c5,
        "best_policy_identified":               best_policy,
    }


def _determine_outcome(criteria: Dict) -> str:
    c1 = criteria["C1_best_policy_beats_blended_harm"]
    c2 = criteria["C2_best_policy_no_coupling_collapse"]
    c3 = criteria["C3_wta_more_spread_than_weighted_sum"]
    c4 = criteria["C4_sufficient_conflict_steps"]
    c5 = criteria["C5_gate_informative"]

    if not c4:
        return "FAIL"

    if c1 and c2 and c3 and c5:
        return "PASS"

    # No policy avoids collapse AND outperforms blending
    if not c2 and not c1:
        return "PARTIAL_NO_WINNER"

    # All policies collapse
    if not c2:
        return "PARTIAL_COLLAPSE_ONLY"

    return "PARTIAL"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_experiment(dry_run: bool = False) -> Dict:
    """Run all conditions and compile the result pack."""
    print("=== V3-EXQ-156: Q-016 Tri-Loop Arbitration Policy Pair ===", flush=True)
    print(f"Conditions: {CONDITIONS}  Seeds: {SEEDS}", flush=True)
    print("Pre-registered thresholds:", flush=True)
    print(f"  C1 THRESH_HARM_MARGIN          = {THRESH_HARM_MARGIN}", flush=True)
    print(f"  C2 THRESH_SPREAD_MIN           = {THRESH_SPREAD_MIN}", flush=True)
    print(f"  C3 WTA spread > WEIGHTED_SUM spread (directional)", flush=True)
    print(f"  C4 THRESH_MIN_CONFLICT_STEPS   = {THRESH_MIN_CONFLICT_STEPS}", flush=True)
    print(f"  C5 THRESH_GATE_VAR_MIN         = {THRESH_GATE_VAR_MIN}", flush=True)
    print(f"  CONFLICT_THRESH                = {CONFLICT_THRESH}", flush=True)
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
    criteria  = _evaluate_criteria(results_by_condition)
    outcome   = _determine_outcome(criteria)
    best_pol  = criteria.get("best_policy_identified", None)

    for k, v in criteria.items():
        if k == "best_policy_identified":
            print(f"  {k}: {v}", flush=True)
        else:
            print(f"  {k}: {'PASS' if v else 'FAIL'}", flush=True)
    print(f"Overall outcome: {outcome}", flush=True)
    if best_pol:
        print(f"Best arbitration policy: {best_pol}", flush=True)

    # Summary metrics: mean over seeds per condition
    def _mean(cond: str, key: str) -> float:
        vals = [r[key] for r in results_by_condition[cond]]
        return float(sum(vals) / max(len(vals), 1))

    summary_metrics: Dict = {}
    for cond in CONDITIONS:
        prefix = cond.lower()
        summary_metrics[f"{prefix}_harm_rate"]            = _mean(cond, "harm_rate")
        summary_metrics[f"{prefix}_inter_loop_spread"]    = _mean(cond, "inter_loop_spread")
        summary_metrics[f"{prefix}_commit_gate_variance"] = _mean(cond, "commit_gate_variance")
        summary_metrics[f"{prefix}_n_conflict_steps"]     = _mean(cond, "n_conflict_steps")
        summary_metrics[f"{prefix}_winner_loop_entropy"]  = _mean(cond, "winner_loop_entropy")

    # Pairwise harm deltas vs BLENDED
    blended_hr = summary_metrics["blended_harm_rate"]
    for cond in ["PRIORITY_HARM", "WEIGHTED_SUM", "WINNER_TAKE_ALL"]:
        prefix = cond.lower()
        summary_metrics[f"delta_harm_{prefix}_vs_blended"] = (
            blended_hr - summary_metrics[f"{prefix}_harm_rate"]
        )

    # Evidence direction
    if outcome == "PASS":
        evidence_direction = "supports"
        guidance = f"arbitration_policy_{best_pol}_avoids_coupling_collapse_and_improves_harm_avoidance"
    elif outcome == "PARTIAL_NO_WINNER":
        evidence_direction = "mixed"
        guidance = "no_policy_avoids_collapse_and_outperforms_blending_structural_separation_needed"
    elif outcome == "PARTIAL_COLLAPSE_ONLY":
        evidence_direction = "mixed"
        guidance = "all_policies_collapse_architecture_level_change_needed_before_arbitration_matters"
    elif outcome == "PARTIAL":
        evidence_direction = "mixed"
        guidance = "partial_evidence_see_criteria_for_details"
    else:
        evidence_direction = "mixed"
        guidance = "insufficient_conflict_steps_increase_hazard_density_or_conflict_prob"

    run_id = (
        "v3_exq_156_q016_tri_loop_arbitration_policy_"
        + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + "_v3"
    )

    pack = {
        "run_id":              run_id,
        "experiment_type":     EXPERIMENT_TYPE,
        "claim_ids":           CLAIM_IDS,
        "architecture_epoch":  "ree_hybrid_guardrails_v1",
        "outcome":             outcome,
        "evidence_direction":  evidence_direction,
        "evidence_class":      "discriminative_pair",
        "guidance":            guidance,
        "best_policy":         best_pol,
        "criteria":            {
            k: v for k, v in criteria.items() if k != "best_policy_identified"
        },
        "pre_registered_thresholds": {
            "THRESH_HARM_MARGIN":        THRESH_HARM_MARGIN,
            "THRESH_SPREAD_MIN":         THRESH_SPREAD_MIN,
            "THRESH_GATE_VAR_MIN":       THRESH_GATE_VAR_MIN,
            "THRESH_MIN_CONFLICT_STEPS": THRESH_MIN_CONFLICT_STEPS,
            "CONFLICT_THRESH":           CONFLICT_THRESH,
        },
        "summary_metrics": summary_metrics,
        "protocol": {
            "total_episodes":       TOTAL_EPISODES,
            "measurement_episodes": MEASUREMENT_EPISODES,
            "steps_per_episode":    STEPS_PER_EPISODE,
            "gate_reg":             GATE_REG,
            "conflict_thresh":      CONFLICT_THRESH,
            "weighted_sum_weights": {"motor": W_MOTOR, "cog": W_COG, "mot": W_MOT},
            "lr":                   LR,
        },
        "seeds":    SEEDS,
        "scenario": (
            "Four-condition tri-loop arbitration policy test."
            " Three gated policy conditions (PRIORITY_HARM, WEIGHTED_SUM, WINNER_TAKE_ALL)"
            " plus a BLENDED baseline (no gating)."
            " Each condition has three loop gates: motor (body-state input),"
            " cognitive (world-state input), motivational (joint input)."
            " Conflict operationalised as |g_motor - g_cog| > CONFLICT_THRESH."
            " PRIORITY_HARM: motivational gate overrides at conflict steps;"
            " mean(all three) otherwise."
            " WEIGHTED_SUM: fixed weighted combination (motor=0.2, cog=0.3, mot=0.5)."
            " WINNER_TAKE_ALL: argmax gate wins; others contribute 0."
            " Loss: g_commit * harm_eval_loss + (1-g_commit) * world_pred_loss"
            " + gate entropy regularisation."
            " BLENDED: 0.5 * harm_eval_loss + 0.5 * world_pred_loss (no gates)."
            " 4 conditions x 2 seeds = 8 cells."
            " CausalGridWorldV2 size=8 num_hazards=5 num_resources=2 hazard_harm=0.1"
            " env_drift_interval=5 env_drift_prob=0.4."
            " 500 total episodes; 125 measurement episodes (last quartile)."
        ),
        "interpretation": (
            "PASS => Best arbitration policy identified (see best_policy field)."
            " That policy avoids coupling collapse (inter_loop_spread > threshold),"
            " outperforms BLENDED on harm avoidance, and has an informative commit gate."
            " WTA produces more inter-loop spread than WEIGHTED_SUM (C3), consistent"
            " with hard competitive inhibition preserving loop independence."
            " Q-016 partially resolved: the identified policy is the answer."
            " PARTIAL_NO_WINNER => All policies that avoid coupling collapse still"
            " fail to outperform BLENDED on harm avoidance, OR no policy avoids collapse."
            " Structural changes (separate loss targets per loop, or architectural BG"
            " segregation) are needed before arbitration policy can be discriminated."
            " PARTIAL_COLLAPSE_ONLY => All policies collapse (inter_loop_spread -> 0)."
            " The three loop gates converge to the same value regardless of condition."
            " Architecture-level change required. Q-016 remains open."
            " FAIL => Insufficient conflict events for reliable metric estimation."
            " Increase num_hazards or env_drift_prob."
        ),
        "per_seed_results": {cond: results_by_condition[cond] for cond in CONDITIONS},
        "dry_run":           dry_run,
        "timestamp_utc":     datetime.now(timezone.utc).isoformat(),
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
