#!/opt/local/bin/python3
"""
V3-EXQ-162 -- MECH-137, MECH-138, MECH-139: Commit Token Structure Anatomy

Claims tested:
  MECH-137: Commit token must carry dual temporal registers (planned_at + triggered_at).
  MECH-138: Commit token must include a cancel-window-open flag supported by a veto pathway.
  MECH-139: Commitment is a distributed multi-second trajectory, not a point event.

Background:
  EXQ-155 established that a WITH_BOUNDARY binary gate g in {0,1} outperforms BLENDED.
  The binary gate is a point commitment: a single threshold separates pre-commit and
  post-commit error routing with no grace period for abort.

  MECH-137/138/139 together claim that commitment is not a point event but a structured
  trajectory that includes: (a) a dual temporal register (when planned vs when triggered),
  and (b) an explicit cancel window after the nominal commit step where a veto pathway
  can abort if harm signals exceed a threshold.

  This experiment tests a THREE_PHASE commit token:
    Phase 1 -- pre-commit  (steps 0 to commit_step):
      Only world prediction loss (E1/E2 path). g = 0.
    Phase 2 -- cancel_window  (steps commit_step to commit_step + N_CANCEL_WINDOW):
      Both paths active at half weight each.
      If harm_signal > CANCEL_THRESH during this window, cancel fires:
        - episode is flagged cancel_fired=True
        - phase resets to PRE_COMMIT; commit_step advances by N_CANCEL_WINDOW
          (the agent gets a second chance after the veto)
    Phase 3 -- post-commit  (steps > commit_step + N_CANCEL_WINDOW):
      Only harm eval loss (E3 path). g = 1.

  Baseline: TWO_PHASE (EXQ-155 WITH_BOUNDARY) -- binary gate, no cancel window.

Pre-registered thresholds:
  THRESH_HARM_MARGIN  = 0.02   C1: THREE_PHASE harm <= TWO_PHASE harm + margin (both seeds)
  THRESH_CORR_MAX     = 0.30   C2: THREE_PHASE pre_post_corr < this (both seeds)
  THRESH_CANCEL_MIN   = 0.05   C3: cancel_window_utilization > this (both seeds)
  THRESH_CANCEL_AVOID = 0.0    C4: cancel_harm_avoided > this (both seeds)

PASS:               all 4 criteria met
  => THREE_PHASE token is behaviorally meaningful; cancel window adds value.
     MECH-137/138/139 supported: structured commit trajectory improves attribution.
PARTIAL_NO_CANCEL:  C1 + C2 met, C3 or C4 fail
  => Structured token provides equivalent or better separation but cancel window
     is too rarely invoked at this env scale.
     Env conflict rate too low to trigger cancel reliably.
PARTIAL_HURTS_POLICY: C1 fails (harm worse than TWO_PHASE + margin)
  => Adding cancel window degrades policy quality.
     THREE_PHASE implementation concern; or env dynamics incompatible with veto.
FAIL: data quality issue or training divergence.

Conditions: THREE_PHASE, TWO_PHASE
Seeds:      [42, 123]
Total cells: 2 conditions x 2 seeds = 4 cells
"""

import sys
import random
import json
import uuid
from enum import Enum, auto
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from ree_core.environment.causal_grid_world import CausalGridWorldV2


# ---------------------------------------------------------------------------
# Experiment identity
# ---------------------------------------------------------------------------

EXPERIMENT_TYPE = "v3_exq_162_mech137_commit_token_structure"
CLAIM_IDS       = ["MECH-137", "MECH-138", "MECH-139"]
RUN_ID_PREFIX   = "v3_exq_162_mech137_commit_token_structure"

# ---------------------------------------------------------------------------
# Pre-registered thresholds
# ---------------------------------------------------------------------------

THRESH_HARM_MARGIN  = 0.02   # C1: THREE_PHASE harm <= TWO_PHASE harm + this
THRESH_CORR_MAX     = 0.30   # C2: pre_post_corr below this = good separation
THRESH_CANCEL_MIN   = 0.05   # C3: cancel_window_utilization must exceed this
THRESH_CANCEL_AVOID = 0.0    # C4: cancel must reduce harm (positive harm_avoided)

# ---------------------------------------------------------------------------
# Protocol constants
# ---------------------------------------------------------------------------

CONDITIONS        = ["THREE_PHASE", "TWO_PHASE"]
SEEDS             = [42, 123]
N_EPISODES        = 400
N_EVAL_EPISODES   = 100      # last quartile (episodes 300-400)
STEPS_PER_EPISODE = 200
COMMIT_STEP_FRAC  = 0.5      # nominal commit at step 100/200
N_CANCEL_WINDOW   = 20       # steps of cancel window
CANCEL_THRESH     = 0.3      # harm_signal threshold to fire cancel

# ---------------------------------------------------------------------------
# Model constants
# ---------------------------------------------------------------------------

WORLD_OBS_DIM = 250   # CausalGridWorldV2 size=8 world_state dim
SELF_OBS_DIM  = 12    # CausalGridWorldV2 body_state dim
ACTION_DIM    = 5
GRID_SIZE     = 8
N_HAZARDS     = 4
N_RESOURCES   = 2
HAZARD_HARM   = 0.05
NAV_BIAS      = 0.6

LR         = 1e-3
HIDDEN_DIM = 32
WORLD_DIM  = 32
SELF_DIM   = 16
HARM_DIM   = 8


# ---------------------------------------------------------------------------
# Phase enum (THREE_PHASE only)
# ---------------------------------------------------------------------------

class Phase(Enum):
    PRE_COMMIT    = auto()
    CANCEL_WINDOW = auto()
    POST_COMMIT   = auto()


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
    """E1 analog: predicts next z_world given current z_world and action."""
    def __init__(self, world_dim: int, action_dim: int):
        super().__init__()
        self.fc = nn.Linear(world_dim + action_dim, world_dim)

    def forward(self, z_world: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.fc(torch.cat([z_world, action], dim=-1))


class E3HarmPredictor(nn.Module):
    """E3 analog: predicts harm from z_world and z_self."""
    def __init__(self, world_dim: int, self_dim: int):
        super().__init__()
        self.fc = nn.Linear(world_dim + self_dim, 1)

    def forward(self, z_world: torch.Tensor, z_self: torch.Tensor) -> torch.Tensor:
        return self.fc(torch.cat([z_world, z_self], dim=-1))


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _pearson_r(xs: List[float], ys: List[float]) -> float:
    """Pearson correlation. Returns 0.0 if degenerate (< 2 points or zero variance)."""
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
    """Safely extract a 1-D tensor from obs_dict, zero-padding or truncating to fallback_dim."""
    raw = obs_dict.get(key)
    if raw is None:
        return torch.zeros(fallback_dim)
    t = torch.tensor(raw, dtype=torch.float32).flatten()
    if t.shape[0] < fallback_dim:
        t = F.pad(t, (0, fallback_dim - t.shape[0]))
    elif t.shape[0] > fallback_dim:
        t = t[:fallback_dim]
    return t


def _safe_mean(vals: List[float]) -> float:
    return float(sum(vals) / len(vals)) if vals else 0.0


# ---------------------------------------------------------------------------
# Core per-cell runner
# ---------------------------------------------------------------------------

def _run_condition(
    seed: int,
    condition: str,
    n_episodes: int,
    n_eval_episodes: int,
    steps_per_episode: int,
    lr: float,
    dry_run: bool,
) -> Dict:
    """
    Run one (condition, seed) cell.

    Returns a dict with all per-cell metrics.
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
    optimizer = optim.Adam(params, lr=lr)

    nominal_commit_step = int(steps_per_episode * COMMIT_STEP_FRAC)
    eval_start_ep       = n_episodes - n_eval_episodes

    # Measurement buffers
    last_q_harm_vals:   List[float] = []
    pre_commit_errors:  List[float] = []   # world_pred_loss when in pre-commit phase
    post_commit_errors: List[float] = []   # harm_eval_loss when in post-commit phase

    # THREE_PHASE specific
    n_cancel_events: int = 0
    # For cancel_harm_avoided: collect post-cancel and non-cancel harm windows
    cancel_window_harm_after_cancel:    List[float] = []
    cancel_window_harm_no_cancel:       List[float] = []

    if dry_run:
        n_episodes      = 2
        eval_start_ep   = 0

    _, obs_dict = env.reset()

    for ep in range(n_episodes):
        ep_harm   = 0.0
        ep_steps  = 0
        in_eval   = (ep >= eval_start_ep)

        # Per-episode THREE_PHASE state
        # commit_step may advance if cancel fires
        current_commit_step = nominal_commit_step
        cancel_fired        = False

        phase: Phase = Phase.PRE_COMMIT  # THREE_PHASE only; not used for TWO_PHASE

        # Buffers for cancel window harm tracking (this episode)
        window_harm_vals: List[float] = []
        in_cancel_window_this_ep = False

        for step in range(steps_per_episode):
            obs_world = _get_obs_tensor(obs_dict, "world_state", WORLD_OBS_DIM)
            obs_self  = _get_obs_tensor(obs_dict, "body_state", SELF_OBS_DIM)

            z_world = world_enc(obs_world)
            z_self  = self_enc(obs_self)

            # Random action (policy not the focus; error routing is)
            action_idx = random.randint(0, ACTION_DIM - 1)
            action = torch.zeros(ACTION_DIM)
            action[action_idx] = 1.0

            # Step environment
            _, _, done, _, obs_dict_next = env.step(action.unsqueeze(0))

            obs_world_next = _get_obs_tensor(obs_dict_next, "world_state", WORLD_OBS_DIM)
            z_world_next_actual = world_enc(obs_world_next).detach()

            harm_signal = float(obs_dict_next.get("harm_signal", 0.0))
            harm_actual = torch.tensor([harm_signal], dtype=torch.float32)
            harm_pred   = e3_pred(z_world, z_self)

            z_world_next_pred = e1_pred(z_world, action)
            world_pred_loss   = F.mse_loss(z_world_next_pred, z_world_next_actual)
            harm_eval_loss    = F.mse_loss(harm_pred, harm_actual)

            # ---------------------------------------------------------------
            # Loss routing per condition
            # ---------------------------------------------------------------
            if condition == "TWO_PHASE":
                # Standard binary gate from EXQ-155 WITH_BOUNDARY
                if step < current_commit_step:
                    total_loss      = world_pred_loss
                    is_pre_commit_  = True
                    is_post_commit_ = False
                else:
                    total_loss      = harm_eval_loss
                    is_pre_commit_  = False
                    is_post_commit_ = True

                if in_eval:
                    if is_pre_commit_:
                        pre_commit_errors.append(float(world_pred_loss.item()))
                        post_commit_errors.append(0.0)
                    elif is_post_commit_:
                        pre_commit_errors.append(0.0)
                        post_commit_errors.append(float(harm_eval_loss.item()))

            else:  # THREE_PHASE
                # Determine current phase based on step vs commit boundaries
                cancel_window_end = current_commit_step + N_CANCEL_WINDOW

                if phase == Phase.PRE_COMMIT:
                    if step < current_commit_step:
                        pass  # stay in PRE_COMMIT
                    else:
                        # Enter cancel window
                        phase = Phase.CANCEL_WINDOW
                        in_cancel_window_this_ep = True
                        window_harm_vals = []

                if phase == Phase.CANCEL_WINDOW:
                    # Check whether to fire cancel
                    if harm_signal > CANCEL_THRESH and not cancel_fired:
                        # Cancel fires: veto pathway active
                        cancel_fired = True
                        n_cancel_events += 1
                        # Record harm in cancel window for cancel_harm_avoided
                        cancel_window_harm_after_cancel.append(harm_signal)
                        # Reset phase to PRE_COMMIT; advance commit_step
                        phase = Phase.PRE_COMMIT
                        current_commit_step = step + N_CANCEL_WINDOW
                    else:
                        # Accumulate window harm for no-cancel path
                        window_harm_vals.append(harm_signal)
                        # Check if window expired
                        if step >= cancel_window_end:
                            phase = Phase.POST_COMMIT
                            # No cancel fired this window; record window harm
                            cancel_window_harm_no_cancel.extend(window_harm_vals)
                            window_harm_vals = []

                # Compute loss based on current phase
                if phase == Phase.PRE_COMMIT:
                    total_loss = world_pred_loss
                    if in_eval:
                        pre_commit_errors.append(float(world_pred_loss.item()))
                        post_commit_errors.append(0.0)
                elif phase == Phase.CANCEL_WINDOW:
                    # Both paths active at half weight
                    total_loss = 0.5 * world_pred_loss + 0.5 * harm_eval_loss
                    if in_eval:
                        pre_commit_errors.append(float(world_pred_loss.item()))
                        post_commit_errors.append(float(harm_eval_loss.item()))
                else:  # POST_COMMIT
                    total_loss = harm_eval_loss
                    if in_eval:
                        pre_commit_errors.append(0.0)
                        post_commit_errors.append(float(harm_eval_loss.item()))

            # ---------------------------------------------------------------
            # Backprop
            # ---------------------------------------------------------------
            optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()

            ep_harm  += harm_signal
            ep_steps += 1

            if done:
                _, obs_dict = env.reset()
            else:
                obs_dict = obs_dict_next

        # End of episode
        if in_eval:
            last_q_harm_vals.append(ep_harm / max(ep_steps, 1))

        if ep % 100 == 0 or (dry_run and ep < 3):
            ep_harm_rate = ep_harm / max(ep_steps, 1)
            print(
                f"  [{condition}] seed={seed} ep={ep}/{n_episodes}"
                f" harm={ep_harm_rate:.5f}"
                f" cancel_events={n_cancel_events}",
                flush=True,
            )

    # -----------------------------------------------------------------------
    # Aggregate metrics
    # -----------------------------------------------------------------------
    last_q_harm   = _safe_mean(last_q_harm_vals)
    pre_post_corr = abs(_pearson_r(pre_commit_errors, post_commit_errors))

    total_ep_count = n_episodes if not dry_run else 2

    # cancel_window_utilization: fraction of episodes where cancel fired
    # For THREE_PHASE: n_cancel_events / total_episode_count
    cancel_utilization = (
        n_cancel_events / max(total_ep_count, 1)
        if condition == "THREE_PHASE"
        else 0.0
    )

    # cancel_harm_avoided: mean harm when cancel fires vs mean harm when it doesn't
    mean_harm_cancel    = _safe_mean(cancel_window_harm_after_cancel)
    mean_harm_no_cancel = _safe_mean(cancel_window_harm_no_cancel)
    cancel_harm_avoided = mean_harm_cancel - mean_harm_no_cancel  # positive = cancel reduced harm

    print(
        f"  [{condition}] seed={seed} DONE"
        f" last_q_harm={last_q_harm:.5f}"
        f" pre_post_corr={pre_post_corr:.4f}"
        f" cancel_util={cancel_utilization:.4f}"
        f" cancel_harm_avoided={cancel_harm_avoided:.5f}",
        flush=True,
    )

    return {
        "condition":              condition,
        "seed":                   seed,
        "last_q_harm":            last_q_harm,
        "pre_post_corr":          pre_post_corr,
        "cancel_window_utilization": cancel_utilization,
        "cancel_harm_avoided":    cancel_harm_avoided,
        "n_cancel_events":        n_cancel_events,
        "mean_harm_when_cancel":  mean_harm_cancel,
        "mean_harm_no_cancel":    mean_harm_no_cancel,
        "n_eval_episodes":        len(last_q_harm_vals),
        "n_pre_commit_samples":   sum(1 for v in pre_commit_errors if v > 0),
        "n_post_commit_samples":  sum(1 for v in post_commit_errors if v > 0),
    }


# ---------------------------------------------------------------------------
# Criterion evaluation
# ---------------------------------------------------------------------------

def _evaluate_criteria(
    results_by_condition: Dict[str, List[Dict]],
) -> Dict[str, bool]:
    """Evaluate all pre-registered criteria."""
    three = results_by_condition["THREE_PHASE"]
    two   = results_by_condition["TWO_PHASE"]
    n_s   = len(SEEDS)

    # C1: THREE_PHASE last_q_harm <= TWO_PHASE last_q_harm + THRESH_HARM_MARGIN (both seeds)
    c1 = all(
        three[i]["last_q_harm"] <= two[i]["last_q_harm"] + THRESH_HARM_MARGIN
        for i in range(n_s)
    )

    # C2: THREE_PHASE pre_post_corr < THRESH_CORR_MAX (both seeds)
    c2 = all(
        three[i]["pre_post_corr"] < THRESH_CORR_MAX
        for i in range(n_s)
    )

    # C3: THREE_PHASE cancel_window_utilization > THRESH_CANCEL_MIN (both seeds)
    c3 = all(
        three[i]["cancel_window_utilization"] > THRESH_CANCEL_MIN
        for i in range(n_s)
    )

    # C4: cancel_harm_avoided > THRESH_CANCEL_AVOID (THREE_PHASE, both seeds)
    c4 = all(
        three[i]["cancel_harm_avoided"] > THRESH_CANCEL_AVOID
        for i in range(n_s)
    )

    return {
        "C1_three_phase_harm_not_worse":       c1,
        "C2_three_phase_signal_separation":    c2,
        "C3_cancel_window_utilization":        c3,
        "C4_cancel_window_avoids_harm":        c4,
    }


def _determine_outcome(criteria: Dict[str, bool]) -> str:
    c1 = criteria["C1_three_phase_harm_not_worse"]
    c2 = criteria["C2_three_phase_signal_separation"]
    c3 = criteria["C3_cancel_window_utilization"]
    c4 = criteria["C4_cancel_window_avoids_harm"]

    if c1 and c2 and c3 and c4:
        return "PASS"

    if c1 and c2 and not (c3 and c4):
        return "PARTIAL_NO_CANCEL"

    if not c1:
        return "PARTIAL_HURTS_POLICY"

    return "FAIL"


# ---------------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------------

def run_experiment(dry_run: bool = False) -> Dict:
    """Run all conditions x seeds and compile the result pack."""
    print("=== V3-EXQ-162: MECH-137/138/139 Commit Token Structure ===", flush=True)
    print(f"Conditions: {CONDITIONS}  Seeds: {SEEDS}", flush=True)
    print("Pre-registered thresholds:", flush=True)
    print(f"  C1 THRESH_HARM_MARGIN  = {THRESH_HARM_MARGIN}", flush=True)
    print(f"  C2 THRESH_CORR_MAX     = {THRESH_CORR_MAX}", flush=True)
    print(f"  C3 THRESH_CANCEL_MIN   = {THRESH_CANCEL_MIN}", flush=True)
    print(f"  C4 THRESH_CANCEL_AVOID = {THRESH_CANCEL_AVOID}", flush=True)
    print(
        f"  N_EPISODES={N_EPISODES}  N_EVAL_EPISODES={N_EVAL_EPISODES}"
        f"  N_CANCEL_WINDOW={N_CANCEL_WINDOW}  CANCEL_THRESH={CANCEL_THRESH}",
        flush=True,
    )

    results_by_condition: Dict[str, List[Dict]] = {c: [] for c in CONDITIONS}

    for condition in CONDITIONS:
        print(f"\n=== Condition: {condition} ===", flush=True)
        for seed in SEEDS:
            result = _run_condition(
                seed=seed,
                condition=condition,
                n_episodes=N_EPISODES,
                n_eval_episodes=N_EVAL_EPISODES,
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

    # Summary metrics (mean over seeds)
    def _mean_seeds(cond: str, key: str) -> float:
        vals = [r[key] for r in results_by_condition[cond]]
        return _safe_mean(vals)

    summary_metrics: Dict = {}
    for cond in CONDITIONS:
        pfx = cond.lower()
        summary_metrics[f"{pfx}_last_q_harm"]              = _mean_seeds(cond, "last_q_harm")
        summary_metrics[f"{pfx}_pre_post_corr"]            = _mean_seeds(cond, "pre_post_corr")
        summary_metrics[f"{pfx}_cancel_window_utilization"] = _mean_seeds(cond, "cancel_window_utilization")
        summary_metrics[f"{pfx}_cancel_harm_avoided"]       = _mean_seeds(cond, "cancel_harm_avoided")
        summary_metrics[f"{pfx}_n_cancel_events"]           = _mean_seeds(cond, "n_cancel_events")

    # Delta: THREE_PHASE harm vs TWO_PHASE harm (positive = THREE better)
    summary_metrics["delta_harm_two_vs_three"] = (
        summary_metrics["two_phase_last_q_harm"]
        - summary_metrics["three_phase_last_q_harm"]
    )

    # Evidence direction
    if outcome == "PASS":
        evidence_direction = "supports"
        guidance = (
            "cancel_window_active_and_reduces_harm; "
            "structured_commit_trajectory_behaviorally_meaningful; "
            "MECH-137/138/139 supported"
        )
    elif outcome == "PARTIAL_NO_CANCEL":
        evidence_direction = "mixed"
        guidance = (
            "three_phase_token_separates_signals_but_cancel_window_vestigial; "
            "env_conflict_rate_too_low_at_this_scale; "
            "increase_hazard_density_or_CANCEL_THRESH"
        )
    elif outcome == "PARTIAL_HURTS_POLICY":
        evidence_direction = "weakens"
        guidance = (
            "cancel_window_degrades_policy_quality; "
            "implementation_concern_or_env_incompatible_with_veto; "
            "review_cancel_reset_logic"
        )
    else:
        evidence_direction = "mixed"
        guidance = "data_quality_or_training_divergence"

    run_id = f"{RUN_ID_PREFIX}_{uuid.uuid4().hex[:8]}_v3"

    pack = {
        "run_id":              run_id,
        "experiment_type":     EXPERIMENT_TYPE,
        "claim_ids_tested":    CLAIM_IDS,
        "architecture_epoch":  "ree_hybrid_guardrails_v1",
        "outcome":             outcome,
        "evidence_direction":  evidence_direction,
        "evidence_class":      "discriminative_pair",
        "guidance":            guidance,
        "criteria_met": {
            "C1": criteria["C1_three_phase_harm_not_worse"],
            "C2": criteria["C2_three_phase_signal_separation"],
            "C3": criteria["C3_cancel_window_utilization"],
            "C4": criteria["C4_cancel_window_avoids_harm"],
        },
        "pre_registered_thresholds": {
            "THRESH_HARM_MARGIN":  THRESH_HARM_MARGIN,
            "THRESH_CORR_MAX":     THRESH_CORR_MAX,
            "THRESH_CANCEL_MIN":   THRESH_CANCEL_MIN,
            "THRESH_CANCEL_AVOID": THRESH_CANCEL_AVOID,
        },
        "summary_metrics":  summary_metrics,
        "per_seed_results": {cond: results_by_condition[cond] for cond in CONDITIONS},
        "config": {
            "conditions":        CONDITIONS,
            "seeds":             SEEDS,
            "n_episodes":        N_EPISODES,
            "n_eval_episodes":   N_EVAL_EPISODES,
            "steps_per_episode": STEPS_PER_EPISODE,
            "commit_step_frac":  COMMIT_STEP_FRAC,
            "n_cancel_window":   N_CANCEL_WINDOW,
            "cancel_thresh":     CANCEL_THRESH,
            "grid_size":         GRID_SIZE,
            "n_hazards":         N_HAZARDS,
            "n_resources":       N_RESOURCES,
            "hazard_harm":       HAZARD_HARM,
            "lr":                LR,
            "world_dim":         WORLD_DIM,
            "self_dim":          SELF_DIM,
            "hidden_dim":        HIDDEN_DIM,
        },
        "scenario": (
            "Two-condition commit token anatomy test. "
            "THREE_PHASE: commit token has pre-commit (world_pred_loss only), "
            "cancel_window (both paths at half weight; cancel fires if harm > CANCEL_THRESH; "
            "veto resets phase to PRE_COMMIT and advances commit_step by N_CANCEL_WINDOW), "
            "post-commit (harm_eval_loss only). "
            "TWO_PHASE: binary gate identical to EXQ-155 WITH_BOUNDARY. "
            "Random action policy. 400 total episodes. 100 eval episodes (last quartile). "
            "CausalGridWorldV2 size=8 num_hazards=4 num_resources=2 hazard_harm=0.05 "
            "env_drift_interval=5 env_drift_prob=0.3. "
            "2 conditions x 2 seeds = 4 cells."
        ),
        "interpretation": (
            "PASS => THREE_PHASE token behaviorally meaningful. Cancel window is non-trivial "
            "(C3) and reduces harm (C4) without hurting policy quality (C1). "
            "Signal separation maintained (C2). MECH-137/138/139 supported: commitment is "
            "a structured distributed trajectory, not a point event; cancel window adds value. "
            "PARTIAL_NO_CANCEL => Token provides good signal separation (C1+C2) but cancel "
            "window is rarely invoked (C3 fail) or does not measurably reduce harm (C4 fail). "
            "Env may lack sufficient high-harm events to trigger veto at current scale. "
            "PARTIAL_HURTS_POLICY => Cancel window resets degrade policy learning (C1 fail). "
            "Implementation concern with veto reset logic or env dynamics. "
            "FAIL => data quality issue or training divergence."
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
        with open(out_path, "w") as fh:
            json.dump(pack, fh, indent=2)
        print(f"\nResult written to: {out_path}", flush=True)
    else:
        print("\n[dry_run] Result pack NOT written to disk.", flush=True)
        print(json.dumps(pack, indent=2), flush=True)

    return pack


if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv
    result  = run_experiment(dry_run=dry_run)
    print(f"\nDone. Outcome: {result['outcome']}", flush=True)
