#!/opt/local/bin/python3
"""
V3-EXQ-184 -- MECH-033: E2 forward-prediction kernels seed hippocampal rollouts.

Claim:    MECH-033
Why now:  Evidence conflict: EXQ-055 PASS, EXQ-124 FAIL, EXQ-171 PASS.
          Need a fresh, clean matched-seed discriminative pair to resolve.

This experiment provides a clean replication with 3 seeds (vs 2 in EXQ-171)
and matched random seeds across conditions. The design follows EXQ-171's
architecture (which fixed EXQ-124's warmup insufficiency) but adds a third
seed for stronger replication evidence.

Conditions (inline architecture, NOT full REEAgent):
  KERNEL_CHAIN: For each step, evaluate all ACTION_DIM=5 candidate actions
    by rolling out k=3 steps with the E2 forward model and scoring cumulative
    predicted harm. Select the action minimizing predicted harm total.
  NO_CHAIN (ablation): Select action uniformly at random (no E2 chaining).

Architecture (shared, trained during warmup in both conditions):
  WorldEncoder:    Linear(250, 32) + LayerNorm -> z_world (world_dim=32)
  SelfEncoder:     Linear(12,  16) + LayerNorm -> z_self  (self_dim=16)
  E2WorldForward:  Linear(32+5, 32) -> z_world_next
  HarmHead:        Linear(32+16, 1) -> harm_scalar (predicts harm from z_world+z_self)

Training (600 warmup episodes, both conditions):
  Random-action data collection. At each step:
    - E2 trained: MSE(z_world_pred, z_world_next_actual) -- world_forward loss.
    - HarmHead trained: MSE(harm_pred, harm_val) -- harm prediction loss.
  Both with Adam lr=3e-4. Single joint optimizer.

E2 convergence check (after warmup):
  Collect 1000 (z_world, action, z_world_next) transitions from random play.
  Compute R^2 = 1 - SS_res / SS_tot for E2 predictions vs actuals.
  This is C3: if R^2 < THRESH_C3_E2_QUALITY, E2 is not trained enough to
  support valid kernel chaining, and the experiment is FAIL (inconclusive,
  not a true FAIL of MECH-033).

Eval (50 episodes after training):
  KERNEL_CHAIN:
    For each step, sample all ACTION_DIM=5 candidate actions.
    For each candidate, roll out k=3 steps with E2WorldForward (no env interaction).
    Score = sum of HarmHead(z_world_t, z_self_t) over k steps.
    Select action with lowest predicted harm score.
  NO_CHAIN:
    Select action uniformly at random.

Metrics (both conditions):
  harm_rate = total harm events / total eval steps
  n_harm_contacts = count of steps where harm_val > 0 during eval

Pre-registered thresholds
--------------------------
C1: harm_rate_CHAIN < harm_rate_NO_CHAIN (all seeds).
    Chaining must reduce harm -- direction must replicate across all 3 seeds.

C2: harm_reduction_delta >= THRESH_C2_MIN_REDUCTION (averaged across seeds).
    Mean delta = mean(harm_rate_NO_CHAIN - harm_rate_CHAIN) across seeds >= threshold.
    THRESH_C2_MIN_REDUCTION = 0.01 (1 pp absolute reduction).

C3: world_forward_r2 >= THRESH_C3_E2_QUALITY (all seeds).
    E2 must be minimally trained. If this fails, outcome is FAIL (inconclusive --
    not a refutation of MECH-033, just insufficient training).
    THRESH_C3_E2_QUALITY = 0.20 (weak positive R^2).

C4: n_harm_contacts_NO_CHAIN >= THRESH_C4_MIN_CONTACTS (all seeds).
    Sufficient harm contacts in ablated condition for reliable harm rate estimation.
    THRESH_C4_MIN_CONTACTS = 5

PASS: C1 + C2 + C3 + C4 (all seeds) -- E2 chaining reduces harm, replicated.
FAIL: any of C1/C2/C3/C4 fails.
  C3 fail: inconclusive (E2 not trained enough; not a MECH-033 refutation).
  C1 or C4 fail with C3 pass: evidence against MECH-033 at this scale.
  C2 fail (C1 pass but small delta): direction correct but effect too small; mixed.

evidence_direction:
  "supports"  if PASS
  "weakens"   if C3+C4 pass but C1 fails (chaining hurts -- direction wrong)
  "mixed"     if C1+C3+C4 pass but C2 fails (direction right but below threshold)
  "mixed"     if C3 fails (E2 not trained -- inconclusive)

Decision scoring:
  "retain_ree"           if PASS
  "hybridize"            if C3+C4 pass but C2 fails (direction right, small effect)
  "retire_ree_claim"     if C3+C4 pass but C1 fails (direction wrong, E2 is trained)
  "inconclusive"         if C3 fails (E2 not trained)

Protocol:
  2 conditions x 3 seeds x 650 total episodes x 100 steps/ep.
  Estimated runtime: 2 x 3 x 650 x 0.005 min/ep = ~20 min (lightweight CPU).
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


EXPERIMENT_TYPE = "v3_exq_184_mech033_kernel_chain_pair"
CLAIM_IDS = ["MECH-033"]

# ---------------------------------------------------------------------------
# Pre-registered thresholds
# ---------------------------------------------------------------------------
THRESH_C2_MIN_REDUCTION  = 0.01   # C2: mean harm_reduction_delta >= 1 pp
THRESH_C3_E2_QUALITY     = 0.20   # C3: world_forward_r2 (E2 convergence)
THRESH_C4_MIN_CONTACTS   = 5      # C4: n_harm_contacts_NO_CHAIN data quality

# ---------------------------------------------------------------------------
# Protocol constants
# ---------------------------------------------------------------------------
WARMUP_EPISODES      = 600
EVAL_EPISODES        = 50
TOTAL_EPISODES       = WARMUP_EPISODES + EVAL_EPISODES
STEPS_PER_EPISODE    = 100
CHAIN_DEPTH          = 3      # k-step rollout depth for KERNEL_CHAIN
LR                   = 3e-4
R2_EVAL_STEPS        = 1000   # steps of random play to compute E2 R^2

SEEDS                = [42, 7, 13]
CONDITIONS           = ["KERNEL_CHAIN", "NO_CHAIN"]

# ---------------------------------------------------------------------------
# Model constants
# ---------------------------------------------------------------------------
WORLD_OBS_DIM  = 250   # CausalGridWorldV2 size=8 world_state dim
SELF_OBS_DIM   = 12    # body_state dim
ACTION_DIM     = 5
WORLD_DIM      = 32
SELF_DIM       = 16


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class WorldEncoder(nn.Module):
    def __init__(self, obs_dim: int, world_dim: int):
        super().__init__()
        self.linear = nn.Linear(obs_dim, world_dim)
        self.norm   = nn.LayerNorm(world_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(F.relu(self.linear(x)))


class SelfEncoder(nn.Module):
    def __init__(self, obs_dim: int, self_dim: int):
        super().__init__()
        self.linear = nn.Linear(obs_dim, self_dim)
        self.norm   = nn.LayerNorm(self_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(F.relu(self.linear(x)))


class E2WorldForward(nn.Module):
    """Linear E2: f(z_world, a) -> z_world_next."""
    def __init__(self, world_dim: int, action_dim: int):
        super().__init__()
        self.fc = nn.Linear(world_dim + action_dim, world_dim)

    def forward(self, z_world: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.fc(torch.cat([z_world, action], dim=-1))


class HarmHead(nn.Module):
    """Linear harm predictor: f(z_world, z_self) -> harm_scalar."""
    def __init__(self, world_dim: int, self_dim: int):
        super().__init__()
        self.fc = nn.Linear(world_dim + self_dim, 1)

    def forward(self, z_world: torch.Tensor, z_self: torch.Tensor) -> torch.Tensor:
        return self.fc(torch.cat([z_world, z_self], dim=-1))


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _get_world_obs(obs_dict: dict) -> torch.Tensor:
    raw = obs_dict.get("world_state")
    if raw is None:
        return torch.zeros(WORLD_OBS_DIM)
    t = raw.float() if isinstance(raw, torch.Tensor) else torch.tensor(raw, dtype=torch.float32)
    t = t.flatten()
    if t.shape[0] < WORLD_OBS_DIM:
        t = F.pad(t, (0, WORLD_OBS_DIM - t.shape[0]))
    elif t.shape[0] > WORLD_OBS_DIM:
        t = t[:WORLD_OBS_DIM]
    return t


def _get_self_obs(obs_dict: dict) -> torch.Tensor:
    raw = obs_dict.get("body_state")
    if raw is None:
        return torch.zeros(SELF_OBS_DIM)
    t = raw.float() if isinstance(raw, torch.Tensor) else torch.tensor(raw, dtype=torch.float32)
    t = t.flatten()
    if t.shape[0] < SELF_OBS_DIM:
        t = F.pad(t, (0, SELF_OBS_DIM - t.shape[0]))
    elif t.shape[0] > SELF_OBS_DIM:
        t = t[:SELF_OBS_DIM]
    return t


def _compute_r2(
    world_enc: WorldEncoder,
    e2_fwd: E2WorldForward,
    env: CausalGridWorldV2,
    n_steps: int,
) -> float:
    """
    Compute R^2 for E2WorldForward predictions on n_steps random transitions.
    Returns R^2 = 1 - SS_res/SS_tot; clamped to [-1, 1].
    """
    preds:   List[List[float]] = []
    actuals: List[List[float]] = []

    _, obs_dict = env.reset()

    with torch.no_grad():
        for _ in range(n_steps):
            obs_w   = _get_world_obs(obs_dict)
            z_world = world_enc(obs_w)
            a_idx   = random.randint(0, ACTION_DIM - 1)
            a_oh    = torch.zeros(ACTION_DIM)
            a_oh[a_idx] = 1.0
            z_pred  = e2_fwd(z_world, a_oh)

            _, _, done, _, obs_next = env.step(a_oh.unsqueeze(0))
            obs_w_next     = _get_world_obs(obs_next)
            z_actual       = world_enc(obs_w_next)

            preds.append(z_pred.tolist())
            actuals.append(z_actual.tolist())

            if done:
                _, obs_dict = env.reset()
            else:
                obs_dict = obs_next

    # Flatten all dimensions
    p_flat = [v for row in preds   for v in row]
    a_flat = [v for row in actuals for v in row]
    n      = len(a_flat)
    if n == 0:
        return 0.0

    mean_a  = sum(a_flat) / n
    ss_tot  = sum((a - mean_a) ** 2 for a in a_flat)
    ss_res  = sum((p - a) ** 2 for p, a in zip(p_flat, a_flat))

    if ss_tot < 1e-10:
        return 0.0
    r2 = 1.0 - ss_res / ss_tot
    return float(max(-1.0, min(1.0, r2)))


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------

def _run_condition(
    seed: int,
    condition: str,
    world_enc: WorldEncoder,
    self_enc: SelfEncoder,
    e2_fwd: E2WorldForward,
    harm_head: HarmHead,
    world_forward_r2: float,
    dry_run: bool,
) -> Dict:
    """
    Run eval phase for one condition using pre-trained models.
    Models have already been trained during warmup (shared training).
    """
    torch.manual_seed(seed + 999)   # different seed for eval randomness
    random.seed(seed + 999)

    env = CausalGridWorldV2(
        seed=seed + 1000,   # different env seed for eval
        size=8,
        num_hazards=3,
        num_resources=3,
        hazard_harm=0.02,
        env_drift_interval=5,
        env_drift_prob=0.2,
    )

    eval_eps      = EVAL_EPISODES if not dry_run else 2
    harm_events   = 0
    total_steps   = 0
    total_harm    = 0.0

    _, obs_dict = env.reset()

    for ep in range(eval_eps):
        ep_harm  = 0.0
        ep_steps = 0

        for step in range(STEPS_PER_EPISODE):
            obs_w  = _get_world_obs(obs_dict)
            obs_s  = _get_self_obs(obs_dict)

            with torch.no_grad():
                z_world = world_enc(obs_w)
                z_self  = self_enc(obs_s)

            # Action selection
            if condition == "KERNEL_CHAIN":
                best_action = 0
                best_score  = float("inf")
                for a_idx in range(ACTION_DIM):
                    a_oh = torch.zeros(ACTION_DIM)
                    a_oh[a_idx] = 1.0
                    with torch.no_grad():
                        z_w_t = z_world.clone()
                        cumulative_harm = 0.0
                        for _k in range(CHAIN_DEPTH):
                            z_w_t      = e2_fwd(z_w_t, a_oh)
                            h_pred     = harm_head(z_w_t, z_self)
                            cumulative_harm += float(h_pred.item())
                            # Use stay action for subsequent rollout steps
                            a_oh = torch.zeros(ACTION_DIM)
                            a_oh[ACTION_DIM - 1] = 1.0   # STAY
                    if cumulative_harm < best_score:
                        best_score  = cumulative_harm
                        best_action = a_idx
                action_idx = best_action
            else:
                # NO_CHAIN: random action
                action_idx = random.randint(0, ACTION_DIM - 1)

            a_oh_step = torch.zeros(ACTION_DIM)
            a_oh_step[action_idx] = 1.0

            _, harm_signal, done, _, obs_dict_next = env.step(a_oh_step.unsqueeze(0))
            harm_val = max(0.0, -harm_signal)

            ep_harm  += harm_val
            ep_steps += 1

            if harm_val > 0.0:
                harm_events += 1

            if done:
                _, obs_dict = env.reset()
            else:
                obs_dict = obs_dict_next

        total_harm  += ep_harm
        total_steps += ep_steps

    harm_rate = total_harm / max(total_steps, 1)

    print(
        "  [%s] seed=%d harm_rate=%.5f"
        " n_harm_events=%d"
        " world_forward_r2=%.4f"
        % (condition, seed, harm_rate, harm_events, world_forward_r2),
        flush=True,
    )

    return {
        "condition":         condition,
        "seed":              seed,
        "harm_rate":         harm_rate,
        "n_harm_events":     harm_events,
        "world_forward_r2":  world_forward_r2,
    }


def _train_models(
    seed: int,
    dry_run: bool,
) -> Tuple[WorldEncoder, SelfEncoder, E2WorldForward, HarmHead, float]:
    """
    Train WorldEncoder, SelfEncoder, E2WorldForward, HarmHead on warmup episodes.
    Returns trained models + E2 R^2 after warmup.
    """
    torch.manual_seed(seed)
    random.seed(seed)

    env = CausalGridWorldV2(
        seed=seed,
        size=8,
        num_hazards=3,
        num_resources=3,
        hazard_harm=0.02,
        env_drift_interval=5,
        env_drift_prob=0.2,
    )

    world_enc = WorldEncoder(WORLD_OBS_DIM, WORLD_DIM)
    self_enc  = SelfEncoder(SELF_OBS_DIM, SELF_DIM)
    e2_fwd    = E2WorldForward(WORLD_DIM, ACTION_DIM)
    harm_head = HarmHead(WORLD_DIM, SELF_DIM)

    train_params = (
        list(world_enc.parameters())
        + list(self_enc.parameters())
        + list(e2_fwd.parameters())
        + list(harm_head.parameters())
    )
    optimizer = optim.Adam(train_params, lr=LR)

    warmup_eps = WARMUP_EPISODES if not dry_run else 4

    _, obs_dict = env.reset()

    for ep in range(warmup_eps):
        for step in range(STEPS_PER_EPISODE):
            obs_w  = _get_world_obs(obs_dict)
            obs_s  = _get_self_obs(obs_dict)

            z_world = world_enc(obs_w)
            z_self  = self_enc(obs_s)

            a_idx = random.randint(0, ACTION_DIM - 1)
            a_oh  = torch.zeros(ACTION_DIM)
            a_oh[a_idx] = 1.0

            _, harm_signal, done, _, obs_dict_next = env.step(a_oh.unsqueeze(0))
            harm_val = max(0.0, -harm_signal)

            obs_w_next  = _get_world_obs(obs_dict_next)
            with torch.no_grad():
                z_world_next_actual = world_enc(obs_w_next)

            z_world_next_pred = e2_fwd(z_world, a_oh)
            e2_loss = F.mse_loss(z_world_next_pred, z_world_next_actual.detach())

            harm_actual = torch.tensor([harm_val], dtype=torch.float32)
            harm_pred   = harm_head(z_world.detach(), z_self.detach())
            harm_loss   = F.mse_loss(harm_pred, harm_actual)

            total_loss = e2_loss + harm_loss
            optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(train_params, 1.0)
            optimizer.step()

            if done:
                _, obs_dict = env.reset()
            else:
                obs_dict = obs_dict_next

        if ep % 100 == 0 or ep == warmup_eps - 1:
            print(
                "  [warmup] seed=%d ep=%d/%d"
                " e2_loss=%.5f harm_loss=%.5f"
                % (seed, ep, warmup_eps, e2_loss.item(), harm_loss.item()),
                flush=True,
            )

    # Compute E2 R^2 on fresh random transitions
    r2_steps = R2_EVAL_STEPS if not dry_run else 50
    r2 = _compute_r2(world_enc, e2_fwd, env, r2_steps)
    print(
        "  [warmup] seed=%d world_forward_r2=%.4f" % (seed, r2),
        flush=True,
    )

    return world_enc, self_enc, e2_fwd, harm_head, r2


# ---------------------------------------------------------------------------
# Criteria
# ---------------------------------------------------------------------------

def _evaluate_criteria(
    results: Dict[str, List[Dict]],
) -> Tuple[Dict, str, str, str]:
    chain     = results["KERNEL_CHAIN"]
    no_chain  = results["NO_CHAIN"]
    n_s       = len(SEEDS)

    # C1: harm_rate_CHAIN < harm_rate_NO_CHAIN (all seeds)
    c1 = all(chain[i]["harm_rate"] < no_chain[i]["harm_rate"] for i in range(n_s))

    # C2: mean(harm_rate_NO_CHAIN - harm_rate_CHAIN) >= THRESH_C2_MIN_REDUCTION
    deltas = [no_chain[i]["harm_rate"] - chain[i]["harm_rate"] for i in range(n_s)]
    mean_delta = sum(deltas) / max(len(deltas), 1)
    c2 = mean_delta >= THRESH_C2_MIN_REDUCTION

    # C3: world_forward_r2 >= THRESH_C3_E2_QUALITY (all seeds, from CHAIN results)
    c3 = all(chain[i]["world_forward_r2"] >= THRESH_C3_E2_QUALITY for i in range(n_s))

    # C4: n_harm_events_NO_CHAIN >= THRESH_C4_MIN_CONTACTS (all seeds)
    c4 = all(no_chain[i]["n_harm_events"] >= THRESH_C4_MIN_CONTACTS for i in range(n_s))

    criteria = {
        "C1_chain_harm_rate_lower_than_no_chain":  c1,
        "C2_mean_harm_reduction_above_threshold":  c2,
        "C3_e2_world_forward_r2_quality":          c3,
        "C4_no_chain_sufficient_harm_contacts":    c4,
        "mean_harm_reduction_delta":               mean_delta,
        "per_seed_deltas":                         deltas,
    }

    if c1 and c2 and c3 and c4:
        outcome            = "PASS"
        evidence_direction = "supports"
        decision           = "retain_ree"
    elif not c3:
        # E2 not trained -- inconclusive for MECH-033
        outcome            = "FAIL"
        evidence_direction = "mixed"
        decision           = "inconclusive"
    elif c3 and c4 and not c1:
        # E2 trained, sufficient data, but chaining hurts or ties
        outcome            = "FAIL"
        evidence_direction = "weakens"
        decision           = "retire_ree_claim"
    elif c1 and c3 and c4 and not c2:
        # Direction correct but effect below threshold
        outcome            = "FAIL"
        evidence_direction = "mixed"
        decision           = "hybridize"
    else:
        outcome            = "FAIL"
        evidence_direction = "mixed"
        decision           = "inconclusive"

    return criteria, outcome, evidence_direction, decision


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_experiment(dry_run: bool = False) -> Dict:
    print("=== V3-EXQ-184: MECH-033 Kernel Chain Pair ===", flush=True)
    print("Conditions: %s  Seeds: %s" % (CONDITIONS, SEEDS), flush=True)
    print("Pre-registered thresholds:", flush=True)
    print("  C2 THRESH_C2_MIN_REDUCTION = %s" % THRESH_C2_MIN_REDUCTION, flush=True)
    print("  C3 THRESH_C3_E2_QUALITY    = %s" % THRESH_C3_E2_QUALITY, flush=True)
    print("  C4 THRESH_C4_MIN_CONTACTS  = %s" % THRESH_C4_MIN_CONTACTS, flush=True)
    print(
        "  WARMUP_EPISODES=%d  EVAL_EPISODES=%d  CHAIN_DEPTH=%d"
        % (WARMUP_EPISODES, EVAL_EPISODES, CHAIN_DEPTH),
        flush=True,
    )

    results: Dict[str, List[Dict]] = {"KERNEL_CHAIN": [], "NO_CHAIN": []}

    for seed in SEEDS:
        print("\n=== Warmup training: seed=%d ===" % seed, flush=True)
        world_enc, self_enc, e2_fwd, harm_head, r2 = _train_models(
            seed=seed,
            dry_run=dry_run,
        )

        for condition in CONDITIONS:
            print(
                "\n=== Eval: condition=%s seed=%d ===" % (condition, seed),
                flush=True,
            )
            r = _run_condition(
                seed=seed,
                condition=condition,
                world_enc=world_enc,
                self_enc=self_enc,
                e2_fwd=e2_fwd,
                harm_head=harm_head,
                world_forward_r2=r2,
                dry_run=dry_run,
            )
            results[condition].append(r)

    print("\n=== Evaluating criteria ===", flush=True)
    criteria, outcome, evidence_direction, decision = _evaluate_criteria(results)
    for k, v in criteria.items():
        if isinstance(v, bool):
            print("  %s: %s" % (k, "PASS" if v else "FAIL"), flush=True)
        elif isinstance(v, list):
            print("  %s: %s" % (k, [round(d, 5) for d in v]), flush=True)
        else:
            print("  %s: %.5f" % (k, v), flush=True)
    print("Overall outcome: %s" % outcome, flush=True)
    print("Decision: %s" % decision, flush=True)

    def _mean(cond: str, key: str) -> float:
        vals = [r[key] for r in results[cond]]
        return float(sum(vals) / max(len(vals), 1))

    summary_metrics = {
        "chain_harm_rate":             _mean("KERNEL_CHAIN", "harm_rate"),
        "no_chain_harm_rate":          _mean("NO_CHAIN",     "harm_rate"),
        "chain_n_harm_events":         _mean("KERNEL_CHAIN", "n_harm_events"),
        "no_chain_n_harm_events":      _mean("NO_CHAIN",     "n_harm_events"),
        "mean_world_forward_r2":       _mean("KERNEL_CHAIN", "world_forward_r2"),
        "mean_harm_reduction_delta":   (
            _mean("NO_CHAIN", "harm_rate") - _mean("KERNEL_CHAIN", "harm_rate")
        ),
    }

    run_id = (
        "v3_exq_184_mech033_kernel_chain_pair_"
        + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + "_v3"
    )

    pack = {
        "run_id":               run_id,
        "experiment_type":      EXPERIMENT_TYPE,
        "claim_ids":            CLAIM_IDS,
        "architecture_epoch":   "ree_hybrid_guardrails_v1",
        "outcome":              outcome,
        "evidence_direction":   evidence_direction,
        "decision":             decision,
        "evidence_class":       "discriminative_pair",
        "criteria":             criteria,
        "pre_registered_thresholds": {
            "THRESH_C2_MIN_REDUCTION":  THRESH_C2_MIN_REDUCTION,
            "THRESH_C3_E2_QUALITY":     THRESH_C3_E2_QUALITY,
            "THRESH_C4_MIN_CONTACTS":   THRESH_C4_MIN_CONTACTS,
        },
        "summary_metrics": summary_metrics,
        "protocol": {
            "warmup_episodes":     WARMUP_EPISODES,
            "eval_episodes":       EVAL_EPISODES,
            "steps_per_episode":   STEPS_PER_EPISODE,
            "chain_depth":         CHAIN_DEPTH,
            "lr":                  LR,
            "r2_eval_steps":       R2_EVAL_STEPS,
        },
        "seeds": SEEDS,
        "prior_evidence": {
            "EXQ-055": "PASS (action-object chaining reduces harm 67x vs random)",
            "EXQ-124": "FAIL (warmup insufficient, E2 not trained before eval)",
            "EXQ-171": "PASS (fixed EXQ-124 root cause with 600 warmup episodes)",
        },
        "scenario": (
            "Fresh discriminative pair for MECH-033 evidence conflict resolution."
            " Two conditions: KERNEL_CHAIN vs NO_CHAIN (random baseline)."
            " KERNEL_CHAIN: at each eval step, roll out ACTION_DIM=5 candidates for"
            " k=3 steps using E2WorldForward + HarmHead; select action minimizing"
            " cumulative predicted harm."
            " NO_CHAIN: select action uniformly at random."
            " Shared warmup: 600 episodes random-action training of WorldEncoder,"
            " SelfEncoder, E2WorldForward, HarmHead (Adam lr=3e-4, joint optimizer)."
            " E2 convergence check: R^2 on 1000 fresh random transitions after warmup."
            " Eval: 50 episodes per condition per seed."
            " Architecture: WorldEncoder=Linear(250,32)+LayerNorm,"
            " SelfEncoder=Linear(12,16)+LayerNorm,"
            " E2WorldForward=Linear(37,32),"
            " HarmHead=Linear(48,1)."
            " CausalGridWorldV2 size=8 num_hazards=3 num_resources=3 hazard_harm=0.02"
            " env_drift_interval=5 env_drift_prob=0.2."
            " 2 conditions x 3 seeds, shared warmup per seed = 6 eval cells."
            " 3 seeds (42, 7, 13) for stronger replication vs EXQ-171 (2 seeds)."
        ),
        "interpretation": (
            "PASS -> MECH-033 supported at inline V3 scale: E2 forward model kernels"
            " provide a usable prediction substrate for action selection that reduces"
            " harm. Effect size (harm_reduction_delta) exceeds 1 pp threshold."
            " Replicated across 3 seeds. Resolves evidence conflict (EXQ-055 PASS,"
            " EXQ-124 FAIL, EXQ-171 PASS) in favor of support."
            " FAIL (C3 fails, E2 not trained) -> Inconclusive: 600 warmup episodes"
            " still insufficient for E2 R^2 >= 0.20. Not a MECH-033 refutation."
            " Next step: increase warmup or check encoder architecture."
            " FAIL (C3+C4 pass, C1 fails) -> Direction wrong: chaining hurts or ties."
            " Weakens MECH-033 at this architecture/scale."
            " FAIL (C1+C3+C4 pass, C2 fails) -> Direction correct but effect too small"
            " to cross 1 pp threshold. Mixed evidence."
        ),
        "per_seed_results":   {cond: results[cond] for cond in CONDITIONS},
        "dry_run":            dry_run,
        "timestamp_utc":      datetime.now(timezone.utc).isoformat(),
    }

    if not dry_run:
        out_dir = (
            Path(__file__).resolve().parents[2]
            / "REE_assembly"
            / "evidence"
            / "experiments"
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / ("%s.json" % run_id)
        with open(out_path, "w") as f:
            json.dump(pack, f, indent=2)
        print("\nResult pack written to: %s" % out_path, flush=True)
    else:
        print("\n[dry_run] Result pack NOT written.", flush=True)
        print(json.dumps(pack, indent=2), flush=True)

    return pack


if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv
    result = run_experiment(dry_run=dry_run)
    print("Done. Outcome: %s" % result["outcome"], flush=True)
