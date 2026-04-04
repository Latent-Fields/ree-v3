#!/opt/local/bin/python3
"""
V3-EXQ-231a -- MECH-106: BG Hysteresis Redesign (Valence-Modulated Threshold)

Claims: MECH-106
Supersedes: V3-EXQ-231
EXPERIMENT_PURPOSE = "evidence"

=============================================================================
WHY EXQ-231 WAS INSUFFICIENT
=============================================================================

EXQ-231 tested PERSISTENT (min-hold 20 steps after commit trigger) vs REACTIVE
(instantaneous threshold crossing). All three seeds produced:
  persist_ratio = 1.0 = reactive_ratio = 1.0
  mean_commit_steps = mean_trigger_steps  (identical in both conditions)

Root causes:
1. Variance stayed below commit_threshold throughout every episode.
   The PERSISTENT min-hold extension never fired because the trigger condition
   was satisfied continuously -- there was nothing to "persist past."
2. No valence history was ever created. The PERSISTENT condition just added an
   artificial hold timer; the REACTIVE condition lacked that timer. Neither
   condition modulated the threshold itself based on outcome valence.
3. The experiment tested commitment PERSISTENCE (which was identical by
   construction when rv never recovers), not commitment THRESHOLD MODULATION.

MECH-106 claims:
  commit_threshold_effective = commit_threshold_base * (1 + valence_bias)
  where valence_bias DECREASES after successful outcomes (D1 potentiation,
  raises effective variance threshold -> easier to commit) and INCREASES after
  harm (D2 activation, lowers effective variance threshold -> harder to commit).

Note on sign convention: MECH-106 refers to commitment in the biological sense
where "lowering the threshold" means "making the behaviour more accessible."
In our variance-based implementation, commitment fires when:
  running_variance < effective_threshold
So EASIER commitment = HIGHER effective variance threshold (rv is more likely
to be below a higher ceiling). The formula implemented here uses:
  effective_threshold = base_threshold * (1 + bias * (valence_ema - 0.5))
  valence_ema -> 1.0 (success history) -> threshold > base -> EASIER to commit
  valence_ema -> 0.0 (harm history)    -> threshold < base -> HARDER to commit
This matches the D1/D2 functional restatement in MECH-106 (Frank 2005).

=============================================================================
REDESIGN: VALENCE-MODULATED THRESHOLD + LATENCY-TO-COMMIT METRIC
=============================================================================

The experiment creates EXPLICIT positive and negative outcome histories, then
measures commitment latency -- how many steps it takes for running_variance to
drop below the effective threshold from a standardised reset value.

Latency-to-commit is sensitive to threshold shifts even when rv has converged
to a stable level: a higher threshold means rv crosses it sooner; a lower
threshold means rv takes longer (or never reaches it within the episode).

Two conditions computed from a SINGLE shared agent run per seed:
  VALENCE_BIAS: effective_threshold = base * (1 + BIAS_WEIGHT * (da - 0.5))
                where da = valence_ema updated by episode outcomes
  NO_BIAS:      effective_threshold = base (fixed, current implementation)

Running both conditions on the same agent eliminates training confounds --
the only variable is which threshold value is used when checking commitment.

Phases (per seed):
  1. TRAINING      100 eps  standard env  (8x8, 2 hazards, 3 resources)
  2. POS_HISTORY    40 eps  easy env      (8x8, 0 hazards, 4 resources)
  3. PROBE_A        25 eps  standard env  (eval mode, no weight updates)
     At each probe episode start: reset rv = RV_PROBE_INIT (0.65, above all
     effective thresholds). Track steps until rv < threshold for each condition.
  4. NEG_HISTORY    40 eps  hard env      (8x8, 5 hazards, 0 resources)
  5. PROBE_B        25 eps  standard env  (eval mode, same protocol as PROBE_A)

Primary metrics (per seed):
  da_pos, da_neg      -- valence EMA after each history phase
  da_divergence       -- da_pos - da_neg (positive = correct divergence)
  threshold_pos_vb    -- effective threshold after positive history
  threshold_neg_vb    -- effective threshold after negative history
  threshold_asymmetry -- threshold_pos_vb / threshold_neg_vb (must be > 1.0)
  latency_pos_vb      -- mean steps-to-first-commit in PROBE_A (VALENCE_BIAS)
  latency_neg_vb      -- mean steps-to-first-commit in PROBE_B (VALENCE_BIAS)
  latency_pos_nb      -- mean steps-to-first-commit in PROBE_A (NO_BIAS)
  latency_neg_nb      -- mean steps-to-first-commit in PROBE_B (NO_BIAS)
  latency_ratio_vb    -- latency_neg_vb / latency_pos_vb
  latency_ratio_nb    -- latency_neg_nb / latency_pos_nb
  asymmetry_delta     -- latency_ratio_vb - latency_ratio_nb

PASS criteria (pre-registered):
  C1: da_divergence > 0.20 in >= 4/5 seeds
      Validates that positive and negative history phases produced genuinely
      different valence signals. If C1 fails, the environment design is wrong.
  C2: threshold_asymmetry > 1.20 in >= 4/5 seeds
      Validates MECH-106 formula: effective threshold is higher after positive
      than after negative history.
  C3: latency_ratio_vb > 2.0 in >= 3/5 seeds
      VALENCE_BIAS shows asymmetric commitment: significantly slower to commit
      after harm than after success.
  C4: latency_ratio_nb in [0.5, 2.5] in >= 3/5 seeds
      Ablation control: fixed threshold produces no strong latency asymmetry
      between positive and negative history phases.
  C5: asymmetry_delta > 1.0 in >= 3/5 seeds
      VALENCE_BIAS shows more asymmetry than NO_BIAS -- the threshold modulation
      is responsible for the behavioral difference.

PASS: C1 AND C2 AND (C3 AND C4) OR (C5 AND C4)
  Primary signal: C1+C2 (mechanistic), C3 or C5 (behavioral), C4 (control clean).

Seeds: [0, 42, 100, 123, 200]
Est: ~150 min DLAPTOP-4.local (5 seeds x 230 eps x 0.10 min/ep x 1.30 overhead)
"""

import sys
import random
import json
import math
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.optim as optim
import torch.nn.functional as F

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig

# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------
EXPERIMENT_TYPE = "v3_exq_231a_mech106_bg_hysteresis_redesign"
CLAIM_IDS = ["MECH-106"]
EXPERIMENT_PURPOSE = "evidence"

# ---------------------------------------------------------------------------
# Pre-registered thresholds
# ---------------------------------------------------------------------------
MIN_DA_DIVERGENCE      = 0.20   # C1: da_pos - da_neg must exceed this
MIN_THRESHOLD_ASYMMETRY = 1.20  # C2: threshold_pos / threshold_neg > this
MIN_LATENCY_RATIO_VB   = 2.0    # C3: latency_neg_vb / latency_pos_vb > this
MAX_LATENCY_RATIO_NB   = 2.5    # C4: NO_BIAS ratio must be <= this (no strong asymmetry)
MIN_LATENCY_RATIO_NB   = 0.5    # C4: NO_BIAS ratio must be >= this
MIN_ASYMMETRY_DELTA    = 1.0    # C5: latency_ratio_vb - latency_ratio_nb > this

# ---------------------------------------------------------------------------
# Hysteresis parameters
# ---------------------------------------------------------------------------
BIAS_WEIGHT    = 1.2    # Modulation strength: threshold = base * (1 + 1.2 * (da - 0.5))
ALPHA_VALENCE  = 0.10   # Slow EMA for valence update (per episode)
RV_PROBE_INIT  = 0.65   # rv reset at start of each probe episode (above all thresholds)
OUTCOME_GAIN   = 10.0   # tanh amplification: maps small benefit/harm totals to strong signal

# Sentinel: episode where rv never crosses threshold
NEVER_COMMITS  = 201    # > STEPS_PER_EP, used as latency when no commit occurs

# ---------------------------------------------------------------------------
# Experiment dimensions
# ---------------------------------------------------------------------------
BODY_OBS_DIM  = 12
WORLD_OBS_DIM = 250
ACTION_DIM    = 5
WORLD_DIM     = 32
SELF_DIM      = 32

N_TRAIN       = 100   # training episodes
N_POS_HIST    = 40    # positive history episodes
N_NEG_HIST    = 40    # negative history episodes
N_PROBE       = 25    # probe episodes per phase
STEPS_PER_EP  = 200   # steps per episode

SEEDS = [0, 42, 100, 123, 200]


# ---------------------------------------------------------------------------
# Environment factories
# ---------------------------------------------------------------------------

def _make_env_standard(seed: int) -> CausalGridWorldV2:
    """Standard training env: 2 hazards, 3 resources, moderate drift."""
    return CausalGridWorldV2(
        seed=seed,
        size=8,
        num_hazards=2,
        num_resources=3,
        hazard_harm=0.02,
        env_drift_interval=10,
        env_drift_prob=0.2,
    )


def _make_env_easy(seed: int) -> CausalGridWorldV2:
    """Easy env: no hazards, many resources. Builds positive valence."""
    return CausalGridWorldV2(
        seed=seed,
        size=8,
        num_hazards=0,
        num_resources=4,
        hazard_harm=0.02,
        env_drift_interval=10,
        env_drift_prob=0.1,
    )


def _make_env_hard(seed: int) -> CausalGridWorldV2:
    """Hard env: many hazards, no resources. Builds negative valence."""
    return CausalGridWorldV2(
        seed=seed,
        size=8,
        num_hazards=5,
        num_resources=0,
        hazard_harm=0.05,
        env_drift_interval=5,
        env_drift_prob=0.3,
    )


def _make_config() -> REEConfig:
    return REEConfig.from_dims(
        body_obs_dim=BODY_OBS_DIM,
        world_obs_dim=WORLD_OBS_DIM,
        action_dim=ACTION_DIM,
        self_dim=SELF_DIM,
        world_dim=WORLD_DIM,
        alpha_world=0.9,
        alpha_self=0.3,
    )


# ---------------------------------------------------------------------------
# Valence EMA update
# ---------------------------------------------------------------------------

def _update_valence_ema(
    valence_ema: float,
    total_benefit: float,
    total_harm: float,
    n_steps: int,
) -> float:
    """
    Update valence EMA from episode outcomes using tanh amplification.

    ep_net = total_benefit - total_harm (net episode outcome, raw units)
    tanh(ep_net * OUTCOME_GAIN) maps small net outcomes to [-1, 1]:
      - total_benefit=0.3 (1 resource) -> tanh(3.0) ~= 0.995 -> outcome_01 ~= 1.0
      - total_harm=0.05 (1 hazard contact) -> tanh(-0.5) ~= -0.46 -> outcome_01 ~= 0.27
      - total_harm=1.0 (many hazard contacts) -> tanh(-10) ~= -1.0 -> outcome_01 ~= 0.0
      - zero net -> tanh(0) = 0 -> outcome_01 = 0.5 (neutral)

    Positive outcome -> valence_ema rises toward 1.0
    Negative outcome -> valence_ema falls toward 0.0
    """
    ep_net = total_benefit - total_harm
    ep_outcome_01 = 0.5 + 0.5 * math.tanh(ep_net * OUTCOME_GAIN)
    return (1.0 - ALPHA_VALENCE) * valence_ema + ALPHA_VALENCE * ep_outcome_01


def _effective_threshold_vb(base_threshold: float, valence_ema: float) -> float:
    """
    MECH-106 formula (variance-space):
      effective = base * (1 + BIAS_WEIGHT * (valence_ema - 0.5))

    valence_ema -> 1.0: threshold > base (easier to commit, D1 potentiation)
    valence_ema -> 0.0: threshold < base (harder to commit, D2 activation)
    """
    return base_threshold * (1.0 + BIAS_WEIGHT * (valence_ema - 0.5))


# ---------------------------------------------------------------------------
# Episode helpers
# ---------------------------------------------------------------------------

def _run_train_episode(
    agent: REEAgent,
    env: CausalGridWorldV2,
    e1_opt: optim.Adam,
    e2_opt: optim.Adam,
    e3_opt: optim.Adam,
    valence_ema: float,
    steps_per_ep: int,
) -> Tuple[float, float, float]:
    """
    Run one training episode. Returns (valence_ema_updated, total_benefit, total_harm).
    """
    _, obs_dict = env.reset()
    agent.reset()
    total_harm = 0.0
    total_benefit = 0.0

    for step in range(steps_per_ep):
        obs_body  = torch.tensor(obs_dict["body_state"],  dtype=torch.float32)
        obs_world = torch.tensor(obs_dict["world_state"], dtype=torch.float32)

        latent     = agent.sense(obs_body, obs_world)
        ticks      = agent.clock.advance()
        e1_prior   = (
            agent._e1_tick(latent) if ticks["e1_tick"]
            else torch.zeros(1, WORLD_DIM, device=agent.device)
        )
        candidates = agent.generate_trajectories(latent, e1_prior, ticks)
        action     = agent.select_action(candidates, ticks, temperature=1.0)

        _, reward, done, info, obs_dict = env.step(action)
        harm_signal = float(reward) if reward < 0 else 0.0
        if reward > 0:
            total_benefit += float(reward)
        if reward < 0:
            total_harm += abs(float(reward))

        # Update running variance (E1 prediction error)
        e1_loss = agent.compute_prediction_loss()
        if hasattr(e1_loss, "item") and e1_loss.requires_grad:
            agent.e3.update_running_variance(torch.tensor([[e1_loss.item()]]))

        # E1 + E2 weight update
        e2_loss = agent.compute_e2_loss()
        total   = e1_loss + e2_loss
        if total.requires_grad:
            e1_opt.zero_grad()
            e2_opt.zero_grad()
            total.backward()
            e1_opt.step()
            e2_opt.step()

        # E3 harm supervision
        if agent._current_latent is not None:
            z_world = agent._current_latent.z_world.detach()
            harm_t  = torch.tensor([[1.0 if harm_signal < 0 else 0.0]])
            hloss   = F.mse_loss(agent.e3.harm_eval(z_world), harm_t)
            if hloss.requires_grad:
                e3_opt.zero_grad()
                hloss.backward()
                e3_opt.step()

        agent.update_residue(harm_signal)
        if done:
            break

    ep_steps = info.get("steps", steps_per_ep)
    ep_total_benefit = info.get("total_benefit", total_benefit)
    ep_total_harm    = info.get("total_harm",    total_harm)
    valence_ema = _update_valence_ema(valence_ema, ep_total_benefit, ep_total_harm, ep_steps)
    return valence_ema, ep_total_benefit, ep_total_harm


def _run_probe_episode(
    agent: REEAgent,
    env: CausalGridWorldV2,
    threshold_vb: float,
    threshold_nb: float,
    steps_per_ep: int,
) -> Tuple[int, int]:
    """
    Run one probe episode (eval mode, no weight updates).

    At start: reset rv = RV_PROBE_INIT.
    Returns (latency_vb, latency_nb):
      latency = step index at which rv first drops below threshold.
      NEVER_COMMITS if rv stays above threshold for entire episode.
    """
    # Reset rv to standardised starting point
    agent.e3._running_variance = RV_PROBE_INIT

    _, obs_dict = env.reset()
    agent.reset()
    # Re-apply rv reset after agent.reset() (which may reinitialise rv)
    agent.e3._running_variance = RV_PROBE_INIT

    first_commit_vb: int = NEVER_COMMITS
    first_commit_nb: int = NEVER_COMMITS

    with torch.no_grad():
        for step in range(steps_per_ep):
            obs_body  = torch.tensor(obs_dict["body_state"],  dtype=torch.float32)
            obs_world = torch.tensor(obs_dict["world_state"], dtype=torch.float32)

            latent     = agent.sense(obs_body, obs_world)
            ticks      = agent.clock.advance()
            e1_prior   = (
                agent._e1_tick(latent) if ticks["e1_tick"]
                else torch.zeros(1, WORLD_DIM, device=agent.device)
            )
            candidates = agent.generate_trajectories(latent, e1_prior, ticks)
            action     = agent.select_action(candidates, ticks, temperature=1.0)

            _, reward, done, info, obs_dict = env.step(action)

            # Update rv from E1 prediction error (no grad, no optimizer step)
            e1_loss = agent.compute_prediction_loss()
            if hasattr(e1_loss, "item"):
                agent.e3.update_running_variance(
                    torch.tensor([[e1_loss.item()]])
                )

            rv = agent.e3._running_variance

            # Check threshold crossings (first time only)
            if first_commit_vb == NEVER_COMMITS and rv < threshold_vb:
                first_commit_vb = step

            if first_commit_nb == NEVER_COMMITS and rv < threshold_nb:
                first_commit_nb = step

            if done:
                break

    return first_commit_vb, first_commit_nb


# ---------------------------------------------------------------------------
# Per-seed run
# ---------------------------------------------------------------------------

def _run_seed(seed: int, n_train: int, n_pos: int, n_neg: int, n_probe: int,
              steps_per_ep: int) -> Dict:
    torch.manual_seed(seed)
    random.seed(seed)

    config = _make_config()
    agent  = REEAgent(config)
    base_threshold = agent.e3.commit_threshold

    e1_opt = optim.Adam(agent.e1.parameters(), lr=1e-3)
    e2_opt = optim.Adam(agent.e2.parameters(), lr=3e-3)
    e3_opt = optim.Adam(agent.e3.parameters(), lr=1e-3)

    valence_ema = 0.5  # initialised neutral

    print(f"  [seed={seed}] base_threshold={base_threshold:.4f}", flush=True)

    # ------------------------------------------------------------------
    # Phase 1: TRAINING (standard env)
    # ------------------------------------------------------------------
    env_standard = _make_env_standard(seed)
    agent.train()
    print(f"  [seed={seed}] TRAINING {n_train} eps ...", flush=True)
    for ep in range(n_train):
        valence_ema, _, _ = _run_train_episode(
            agent, env_standard, e1_opt, e2_opt, e3_opt, valence_ema, steps_per_ep
        )
        if (ep + 1) % 50 == 0:
            rv = agent.e3._running_variance
            print(
                f"  [seed={seed}] TRAIN ep {ep+1}/{n_train}"
                f" rv={rv:.4f} valence_ema={valence_ema:.3f}",
                flush=True,
            )

    # ------------------------------------------------------------------
    # Phase 2: POSITIVE HISTORY (easy env, 0 hazards, 4 resources)
    # ------------------------------------------------------------------
    env_easy = _make_env_easy(seed)
    agent.train()
    print(f"  [seed={seed}] POSITIVE_HISTORY {n_pos} eps ...", flush=True)
    for ep in range(n_pos):
        valence_ema, _, _ = _run_train_episode(
            agent, env_easy, e1_opt, e2_opt, e3_opt, valence_ema, steps_per_ep
        )

    da_pos = valence_ema
    threshold_pos_vb = _effective_threshold_vb(base_threshold, da_pos)
    threshold_pos_nb = base_threshold  # NO_BIAS: fixed
    print(
        f"  [seed={seed}] after POS_HIST: da_pos={da_pos:.4f}"
        f" threshold_vb={threshold_pos_vb:.4f} threshold_nb={threshold_pos_nb:.4f}",
        flush=True,
    )

    # ------------------------------------------------------------------
    # Phase 3: PROBE_A (after positive history)
    # ------------------------------------------------------------------
    env_probe = _make_env_standard(seed + 1000)  # fresh probe env
    agent.eval()
    latencies_pos_vb: List[int] = []
    latencies_pos_nb: List[int] = []
    print(f"  [seed={seed}] PROBE_A {n_probe} eps ...", flush=True)
    for ep in range(n_probe):
        lv, ln = _run_probe_episode(
            agent, env_probe, threshold_pos_vb, threshold_pos_nb, steps_per_ep
        )
        latencies_pos_vb.append(lv)
        latencies_pos_nb.append(ln)

    mean_latency_pos_vb = sum(latencies_pos_vb) / len(latencies_pos_vb)
    mean_latency_pos_nb = sum(latencies_pos_nb) / len(latencies_pos_nb)
    commits_pos_vb = sum(1 for l in latencies_pos_vb if l < NEVER_COMMITS)
    commits_pos_nb = sum(1 for l in latencies_pos_nb if l < NEVER_COMMITS)
    print(
        f"  [seed={seed}] PROBE_A done:"
        f" mean_latency_vb={mean_latency_pos_vb:.1f} ({commits_pos_vb}/{n_probe} commits)"
        f" mean_latency_nb={mean_latency_pos_nb:.1f} ({commits_pos_nb}/{n_probe} commits)",
        flush=True,
    )

    # ------------------------------------------------------------------
    # Phase 4: NEGATIVE HISTORY (hard env, 5 hazards, 0 resources)
    # ------------------------------------------------------------------
    env_hard = _make_env_hard(seed)
    agent.train()
    print(f"  [seed={seed}] NEGATIVE_HISTORY {n_neg} eps ...", flush=True)
    for ep in range(n_neg):
        valence_ema, _, _ = _run_train_episode(
            agent, env_hard, e1_opt, e2_opt, e3_opt, valence_ema, steps_per_ep
        )

    da_neg = valence_ema
    threshold_neg_vb = _effective_threshold_vb(base_threshold, da_neg)
    threshold_neg_nb = base_threshold  # NO_BIAS: fixed
    print(
        f"  [seed={seed}] after NEG_HIST: da_neg={da_neg:.4f}"
        f" threshold_vb={threshold_neg_vb:.4f} threshold_nb={threshold_neg_nb:.4f}",
        flush=True,
    )

    # ------------------------------------------------------------------
    # Phase 5: PROBE_B (after negative history)
    # ------------------------------------------------------------------
    agent.eval()
    latencies_neg_vb: List[int] = []
    latencies_neg_nb: List[int] = []
    print(f"  [seed={seed}] PROBE_B {n_probe} eps ...", flush=True)
    for ep in range(n_probe):
        lv, ln = _run_probe_episode(
            agent, env_probe, threshold_neg_vb, threshold_neg_nb, steps_per_ep
        )
        latencies_neg_vb.append(lv)
        latencies_neg_nb.append(ln)

    mean_latency_neg_vb = sum(latencies_neg_vb) / len(latencies_neg_vb)
    mean_latency_neg_nb = sum(latencies_neg_nb) / len(latencies_neg_nb)
    commits_neg_vb = sum(1 for l in latencies_neg_vb if l < NEVER_COMMITS)
    commits_neg_nb = sum(1 for l in latencies_neg_nb if l < NEVER_COMMITS)
    print(
        f"  [seed={seed}] PROBE_B done:"
        f" mean_latency_vb={mean_latency_neg_vb:.1f} ({commits_neg_vb}/{n_probe} commits)"
        f" mean_latency_nb={mean_latency_neg_nb:.1f} ({commits_neg_nb}/{n_probe} commits)",
        flush=True,
    )

    # ------------------------------------------------------------------
    # Per-seed metrics
    # ------------------------------------------------------------------
    da_divergence      = da_pos - da_neg
    threshold_asymmetry = (threshold_pos_vb / max(threshold_neg_vb, 1e-6))

    # Ratio: latency_neg / latency_pos (higher = more asymmetry = good for MECH-106)
    latency_ratio_vb = mean_latency_neg_vb / max(mean_latency_pos_vb, 1.0)
    latency_ratio_nb = mean_latency_neg_nb / max(mean_latency_pos_nb, 1.0)
    asymmetry_delta  = latency_ratio_vb - latency_ratio_nb

    c1 = da_divergence > MIN_DA_DIVERGENCE
    c2 = threshold_asymmetry > MIN_THRESHOLD_ASYMMETRY
    c3 = latency_ratio_vb > MIN_LATENCY_RATIO_VB
    c4 = MIN_LATENCY_RATIO_NB <= latency_ratio_nb <= MAX_LATENCY_RATIO_NB
    c5 = asymmetry_delta > MIN_ASYMMETRY_DELTA

    print(
        f"  [seed={seed}] RESULTS:"
        f" da_div={da_divergence:.3f} thr_asym={threshold_asymmetry:.3f}"
        f" lat_ratio_vb={latency_ratio_vb:.2f} lat_ratio_nb={latency_ratio_nb:.2f}"
        f" asym_delta={asymmetry_delta:.2f}",
        flush=True,
    )
    print(
        f"  [seed={seed}] CRITERIA:"
        f" C1={'P' if c1 else 'F'}"
        f" C2={'P' if c2 else 'F'}"
        f" C3={'P' if c3 else 'F'}"
        f" C4={'P' if c4 else 'F'}"
        f" C5={'P' if c5 else 'F'}",
        flush=True,
    )

    return {
        "seed": seed,
        "da_pos":              da_pos,
        "da_neg":              da_neg,
        "da_divergence":       da_divergence,
        "threshold_pos_vb":    threshold_pos_vb,
        "threshold_neg_vb":    threshold_neg_vb,
        "threshold_asymmetry": threshold_asymmetry,
        "mean_latency_pos_vb": mean_latency_pos_vb,
        "mean_latency_neg_vb": mean_latency_neg_vb,
        "mean_latency_pos_nb": mean_latency_pos_nb,
        "mean_latency_neg_nb": mean_latency_neg_nb,
        "latency_ratio_vb":    latency_ratio_vb,
        "latency_ratio_nb":    latency_ratio_nb,
        "asymmetry_delta":     asymmetry_delta,
        "commits_pos_vb":      commits_pos_vb,
        "commits_neg_vb":      commits_neg_vb,
        "commits_pos_nb":      commits_pos_nb,
        "commits_neg_nb":      commits_neg_nb,
        "c1": c1, "c2": c2, "c3": c3, "c4": c4, "c5": c5,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(dry_run: bool = False) -> dict:
    print(f"\n[EXQ-231a] MECH-106 BG Hysteresis Redesign (dry_run={dry_run})", flush=True)

    n_train = 5  if dry_run else N_TRAIN
    n_pos   = 5  if dry_run else N_POS_HIST
    n_neg   = 5  if dry_run else N_NEG_HIST
    n_probe = 4  if dry_run else N_PROBE
    n_steps = 20 if dry_run else STEPS_PER_EP

    seeds = SEEDS if not dry_run else SEEDS[:2]

    seed_results: List[Dict] = []
    c1_passes: List[bool] = []
    c2_passes: List[bool] = []
    c3_passes: List[bool] = []
    c4_passes: List[bool] = []
    c5_passes: List[bool] = []

    for seed in seeds:
        print(f"\n--- seed={seed} ---", flush=True)
        sr = _run_seed(seed, n_train, n_pos, n_neg, n_probe, n_steps)
        seed_results.append(sr)
        c1_passes.append(sr["c1"])
        c2_passes.append(sr["c2"])
        c3_passes.append(sr["c3"])
        c4_passes.append(sr["c4"])
        c5_passes.append(sr["c5"])

    n_seeds = len(seeds)
    # Majority thresholds
    c1_thresh = max(1, n_seeds - 1)   # 4/5 seeds (or 1/2 in dry_run)
    c2_thresh = max(1, n_seeds - 1)
    c3_thresh = (n_seeds + 1) // 2    # 3/5 seeds (or 1/2 in dry_run)
    c4_thresh = (n_seeds + 1) // 2
    c5_thresh = (n_seeds + 1) // 2

    c1_pass = sum(c1_passes) >= c1_thresh
    c2_pass = sum(c2_passes) >= c2_thresh
    c3_pass = sum(c3_passes) >= c3_thresh
    c4_pass = sum(c4_passes) >= c4_thresh
    c5_pass = sum(c5_passes) >= c5_thresh

    all_pass = c1_pass and c2_pass and (c3_pass or c5_pass) and c4_pass
    status   = "PASS" if all_pass else "FAIL"
    crit_met = sum([c1_pass, c2_pass, c3_pass, c4_pass, c5_pass])

    print(f"\n[EXQ-231a] Final summary:", flush=True)
    for sr in seed_results:
        print(
            f"  seed={sr['seed']}:"
            f" da_div={sr['da_divergence']:.3f}"
            f" thr_asym={sr['threshold_asymmetry']:.3f}"
            f" lat_ratio_vb={sr['latency_ratio_vb']:.2f}"
            f" lat_ratio_nb={sr['latency_ratio_nb']:.2f}"
            f" asym_delta={sr['asymmetry_delta']:.2f}"
            f" C1={'P' if sr['c1'] else 'F'}"
            f" C2={'P' if sr['c2'] else 'F'}"
            f" C3={'P' if sr['c3'] else 'F'}"
            f" C4={'P' if sr['c4'] else 'F'}"
            f" C5={'P' if sr['c5'] else 'F'}",
            flush=True,
        )
    print(
        f"  Status: {status} ({crit_met}/5 criteria)"
        f" C1={'P' if c1_pass else 'F'}"
        f" C2={'P' if c2_pass else 'F'}"
        f" C3={'P' if c3_pass else 'F'}"
        f" C4={'P' if c4_pass else 'F'}"
        f" C5={'P' if c5_pass else 'F'}",
        flush=True,
    )

    failure_notes: List[str] = []
    if not c1_pass:
        vals = [sr["da_divergence"] for sr in seed_results]
        failure_notes.append(
            f"C1 FAIL: da_divergences={[round(v, 3) for v in vals]}"
            f" (need > {MIN_DA_DIVERGENCE} in >= {c1_thresh}/{n_seeds})"
        )
    if not c2_pass:
        vals = [sr["threshold_asymmetry"] for sr in seed_results]
        failure_notes.append(
            f"C2 FAIL: threshold_asymmetries={[round(v, 3) for v in vals]}"
            f" (need > {MIN_THRESHOLD_ASYMMETRY} in >= {c2_thresh}/{n_seeds})"
        )
    if not c3_pass:
        vals = [sr["latency_ratio_vb"] for sr in seed_results]
        failure_notes.append(
            f"C3 FAIL: latency_ratio_vb={[round(v, 2) for v in vals]}"
            f" (need > {MIN_LATENCY_RATIO_VB} in >= {c3_thresh}/{n_seeds})"
        )
    if not c4_pass:
        vals = [sr["latency_ratio_nb"] for sr in seed_results]
        failure_notes.append(
            f"C4 FAIL: latency_ratio_nb={[round(v, 2) for v in vals]}"
            f" (need in [{MIN_LATENCY_RATIO_NB}, {MAX_LATENCY_RATIO_NB}] in >= {c4_thresh}/{n_seeds})"
        )
    if not c5_pass:
        vals = [sr["asymmetry_delta"] for sr in seed_results]
        failure_notes.append(
            f"C5 FAIL: asymmetry_delta={[round(v, 2) for v in vals]}"
            f" (need > {MIN_ASYMMETRY_DELTA} in >= {c5_thresh}/{n_seeds})"
        )

    if all_pass:
        interp = (
            "MECH-106 SUPPORTED: Valence-modulated threshold produces genuine"
            " asymmetric commitment hysteresis. Latency to first commitment is"
            " significantly shorter after positive history (higher threshold) than"
            " after negative history (lower threshold). NO_BIAS ablation shows no"
            " such asymmetry, confirming the effect is due to valence-driven"
            " threshold modulation, not training dynamics."
        )
    elif crit_met >= 3:
        interp = (
            "MECH-106 PARTIAL: Mechanistic signal present (C1/C2 likely passed)"
            " but behavioral asymmetry did not meet all criteria."
            " Check individual seed results for diagnosis."
        )
    else:
        interp = (
            "MECH-106 NOT SUPPORTED: Valence EMA did not diverge sufficiently,"
            " or threshold asymmetry did not produce detectable latency difference."
            " Check da_divergence -- if C1 fails, environment design needs revision."
        )

    # ------------------------------------------------------------------
    # Flat metrics dict
    # ------------------------------------------------------------------
    metrics: Dict = {
        "c1_pass": 1.0 if c1_pass else 0.0,
        "c2_pass": 1.0 if c2_pass else 0.0,
        "c3_pass": 1.0 if c3_pass else 0.0,
        "c4_pass": 1.0 if c4_pass else 0.0,
        "c5_pass": 1.0 if c5_pass else 0.0,
        "criteria_met": float(crit_met),
    }
    for i, sr in enumerate(seed_results):
        sfx = f"_s{i}"
        for k in [
            "da_pos", "da_neg", "da_divergence",
            "threshold_pos_vb", "threshold_neg_vb", "threshold_asymmetry",
            "mean_latency_pos_vb", "mean_latency_neg_vb",
            "mean_latency_pos_nb", "mean_latency_neg_nb",
            "latency_ratio_vb", "latency_ratio_nb", "asymmetry_delta",
        ]:
            metrics[k + sfx] = float(sr[k])

    # ------------------------------------------------------------------
    # Summary markdown
    # ------------------------------------------------------------------
    summary_markdown = (
        f"# V3-EXQ-231a -- MECH-106 BG Hysteresis Redesign\n\n"
        f"**Status:** {status}  **Criteria met:** {crit_met}/5\n"
        f"**Claims:** MECH-106  **Purpose:** evidence  **Supersedes:** V3-EXQ-231\n\n"
        f"## Why EXQ-231 was insufficient\n\n"
        f"EXQ-231 tested PERSISTENT (min-hold 20 steps) vs REACTIVE (instantaneous).\n"
        f"All seeds: persist_ratio = reactive_ratio = 1.0 (identical behavior).\n"
        f"Root cause: rv stayed below commit_threshold throughout every episode;\n"
        f"the hold extension never fired. No valence history was created.\n\n"
        f"## Redesign\n\n"
        f"VALENCE_BIAS: effective_threshold = base * (1 + {BIAS_WEIGHT} * (da - 0.5))\n"
        f"NO_BIAS: fixed threshold (current implementation / ablation)\n"
        f"Metric: latency-to-first-commitment from rv reset ({RV_PROBE_INIT}) per probe episode.\n\n"
        f"## Results\n\n"
        f"| Seed | da_div | thr_asym | lat_ratio_vb | lat_ratio_nb | asym_delta"
        f" | C1 | C2 | C3 | C4 | C5 |\n"
        f"|------|--------|----------|--------------|--------------|-----------|"
        f"----|----|----|----|----|"
    )
    for sr in seed_results:
        summary_markdown += (
            f"\n| {sr['seed']} | {sr['da_divergence']:.3f} | {sr['threshold_asymmetry']:.3f}"
            f" | {sr['latency_ratio_vb']:.2f} | {sr['latency_ratio_nb']:.2f}"
            f" | {sr['asymmetry_delta']:.2f}"
            f" | {'P' if sr['c1'] else 'F'}"
            f" | {'P' if sr['c2'] else 'F'}"
            f" | {'P' if sr['c3'] else 'F'}"
            f" | {'P' if sr['c4'] else 'F'}"
            f" | {'P' if sr['c5'] else 'F'} |"
        )
    summary_markdown += (
        f"\n\n## Interpretation\n\n{interp}\n"
    )
    if failure_notes:
        summary_markdown += "\n## Failure Notes\n\n"
        summary_markdown += "\n".join(f"- {n}" for n in failure_notes) + "\n"

    return {
        "status":             status,
        "metrics":            metrics,
        "summary_markdown":   summary_markdown,
        "claim_ids":          CLAIM_IDS,
        "evidence_direction": (
            "supports" if all_pass
            else ("mixed" if crit_met >= 3 else "weakens")
        ),
        "experiment_type":    EXPERIMENT_TYPE,
        "fatal_error_count":  0,
        "seed_results":       seed_results,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    result = run(dry_run=args.dry_run)

    ts  = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result["run_timestamp"]       = ts
    result["run_id"]              = f"{EXPERIMENT_TYPE}_{ts}_v3"
    result["architecture_epoch"]  = "ree_hybrid_guardrails_v1"

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
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}", flush=True)
        else:
            print(f"  {k}: {v}", flush=True)
