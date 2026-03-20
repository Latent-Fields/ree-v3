"""
V3-EXQ-018b — ARC-016 Relative Threshold

Claims: ARC-016 (precision.e3_derived_dynamic_precision).

Rewrite of EXQ-018 with RELATIVE commit threshold instead of absolute.

Motivation (2026-03-20):
  EXQ-018 FAIL diagnosis: The variance mechanism is confirmed working — precision
  drops 40% (718→426) under maximal perturbation. But commit_threshold=0.40 is
  100× the actual operating variance range (stable≈0.0026, perturbed≈0.0036).
  Absolute threshold calibration is fragile: any environment change shifts the
  variance operating range.

  Fix: RELATIVE threshold calibrated from training baseline.
    commit_threshold = calibration_factor × training_baseline_variance
  where training_baseline_variance = mean running_variance over the LAST 100
  training episodes, and calibration_factor = 2.0.

  Biological analogy — "sleep recalibration":
    Slow-wave sleep resets the prediction error baseline. Waking commitment
    thresholds are calibrated relative to that baseline, not against absolute
    values set at design time. An agent that wakes into a new environment will
    re-establish its baseline during the first sleep cycle and adjust its
    commitment thresholds accordingly.

  This ensures the threshold is always 2× the stable operating variance,
  regardless of the absolute scale — the agent commits when variance is within
  its normal operating range, and withholds commitment when variance exceeds
  that range by more than the calibration factor.

More extreme environment contrast than EXQ-018:
  - Stable:    num_hazards=2,  drift_interval=200, drift_prob=0.0
  - Perturbed: num_hazards=25, drift_interval=1,   drift_prob=1.0
  (maximises variance contrast between conditions)

Protocol:
  1. Training (stable env, 1000 eps): train agent + E2.world_forward.
     Record mean running_variance over LAST 100 training episodes → training_baseline_variance.
  2. Set commit_threshold = 2.0 × training_baseline_variance.
     Also update agent.e3.config.commitment_threshold to match.
  3. Eval stable (50 eps): record variance trajectory + commit decisions.
  4. Eval perturbed (50 eps, extreme drift): same metrics.
  5. Recovery eval (30 eps, stable again): verify variance recovers.

PASS criteria (ALL must hold):
  C0: training_baseline_variance > 0  (calibration succeeded)
  C1: var_diff > 0.5 × training_baseline_variance
      (relative threshold — perturbed must exceed 1.5× baseline)
  C2: mean_precision_stable > mean_precision_perturbed
      (precision drops in perturbed env)
  C3: commit_rate_stable > commit_rate_perturbed
      (main ARC-016 behavioral prediction)
  C4: n_commit_decisions_stable >= 20

Note on sleep recalibration:
  The training_baseline_variance is analogous to the prediction error baseline
  established during slow-wave sleep. The calibration_factor (2.0) is the
  "wakefulness safety margin" — the agent treats variance within 2× of sleep
  baseline as familiar territory warranting commitment, and variance beyond
  that as a signal to withhold commitment and explore.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from typing import Dict, List, Tuple
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorld
from ree_core.utils.config import REEConfig


EXPERIMENT_TYPE = "v3_exq_018b_arc016_relative_threshold"
CLAIM_IDS = ["ARC-016"]

CALIBRATION_FACTOR   = 2.0     # commit_threshold = CALIBRATION_FACTOR × baseline_variance
BASELINE_WINDOW      = 100     # last N training episodes used for baseline estimation
E3_DECISION_INTERVAL = 10      # steps between E3 candidate evaluations
NUM_CANDIDATES       = 16
CANDIDATE_HORIZON    = 5


def _action_to_onehot(action_idx: int, num_actions: int, device) -> torch.Tensor:
    v = torch.zeros(1, num_actions, device=device)
    v[0, action_idx] = 1.0
    return v


def _make_world_decoder(world_dim: int, world_obs_dim: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(world_dim, 64), nn.ReLU(), nn.Linear(64, world_obs_dim)
    )


def _generate_random_candidates(agent: REEAgent, z_self: torch.Tensor,
                                  z_world: torch.Tensor) -> list:
    """Generate lightweight random candidates for E3 evaluation."""
    return agent.e2.generate_candidates_random(
        initial_z_self=z_self,
        initial_z_world=z_world,
        num_candidates=NUM_CANDIDATES,
        horizon=CANDIDATE_HORIZON,
        compute_action_objects=True,
    )


def _run_phase(
    agent: REEAgent,
    env: CausalGridWorld,
    optimizer: optim.Optimizer,
    world_decoder: nn.Module,
    num_episodes: int,
    steps_per_episode: int,
    train: bool,
    phase_name: str,
    record_last_n_variance: int = 0,
) -> Dict:
    """
    Run one phase. Returns metrics and optionally baseline variance estimate.

    If record_last_n_variance > 0 and train=True, the mean running_variance
    over the last `record_last_n_variance` episodes is returned as
    'baseline_variance' — used for relative threshold calibration.
    """
    if train:
        agent.train()
        world_decoder.train()
    else:
        agent.eval()
        world_decoder.eval()

    running_variance_traj: List[float] = []
    precision_traj:        List[float] = []
    commit_decisions:      List[bool]  = []
    prediction_errors:     List[float] = []

    # Per-episode mean variance (for baseline estimation)
    ep_mean_variances: List[float] = []

    total_harm    = 0
    total_benefit = 0
    step_counter  = 0

    e2w_buf: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    MAX_E2W_BUF = 500

    for ep in range(num_episodes):
        flat_obs, obs_dict = env.reset()
        agent.reset()

        z_world_prev = None
        a_prev       = None
        z_self_prev  = None
        ep_variances: List[float] = []

        for step in range(steps_per_episode):
            obs_body  = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]

            latent = agent.sense(obs_body, obs_world)
            agent.clock.advance()

            z_world_curr = latent.z_world.detach()
            z_self_curr  = latent.z_self.detach()

            action_idx = random.randint(0, env.action_dim - 1)
            action = _action_to_onehot(action_idx, env.action_dim, agent.device)
            agent._last_action = action

            # Update E3 running_variance from z_world prediction error
            if z_world_prev is not None and a_prev is not None:
                with torch.no_grad():
                    z_world_predicted = agent.e2.world_forward(z_world_prev, a_prev)
                    prediction_error  = z_world_curr - z_world_predicted
                    agent.e3.update_running_variance(prediction_error)
                    pred_err_mag = float(prediction_error.pow(2).mean().item())
                    prediction_errors.append(pred_err_mag)

            # Record precision state
            running_variance_traj.append(agent.e3._running_variance)
            precision_traj.append(agent.e3.current_precision)
            ep_variances.append(agent.e3._running_variance)

            # Every E3_DECISION_INTERVAL steps: simulate E3 selection and record commit
            if step_counter % E3_DECISION_INTERVAL == 0:
                with torch.no_grad():
                    try:
                        candidates = _generate_random_candidates(
                            agent, z_self_curr, z_world_curr
                        )
                        result = agent.e3.select(candidates, temperature=1.0)
                        commit_decisions.append(result.committed)
                    except Exception:
                        pass

            flat_obs, harm_signal, done, info, obs_dict = env.step(action)
            if harm_signal < 0:
                total_harm += 1
            elif harm_signal > 0:
                total_benefit += 1

            if train:
                if z_world_prev is not None and a_prev is not None:
                    e2w_buf.append((z_world_prev, a_prev, z_world_curr))
                    if len(e2w_buf) > MAX_E2W_BUF:
                        e2w_buf = e2w_buf[-MAX_E2W_BUF:]

                e1_loss      = agent.compute_prediction_loss()
                e2_self_loss = agent.compute_e2_loss()
                obs_w = obs_world.unsqueeze(0) if obs_world.dim() == 1 else obs_world
                z_w   = agent.latent_stack.split_encoder.world_encoder(obs_w)
                recon = world_decoder(z_w)
                recon_loss = F.mse_loss(recon, obs_w)

                e2w_loss = torch.zeros(1, device=agent.device)
                if len(e2w_buf) >= 16:
                    k    = min(32, len(e2w_buf))
                    idxs = torch.randperm(len(e2w_buf))[:k].tolist()
                    zw_t  = torch.cat([e2w_buf[i][0] for i in idxs], dim=0)
                    a_t   = torch.cat([e2w_buf[i][1] for i in idxs], dim=0)
                    zw_t1 = torch.cat([e2w_buf[i][2] for i in idxs], dim=0)
                    zw_pred = agent.e2.world_forward(zw_t, a_t)
                    e2w_loss = F.mse_loss(zw_pred, zw_t1)

                total_loss = e1_loss + e2_self_loss + recon_loss + e2w_loss
                if total_loss.requires_grad:
                    optimizer.zero_grad()
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                    optimizer.step()

            z_world_prev = z_world_curr
            z_self_prev  = z_self_curr
            a_prev       = action.detach()
            step_counter += 1

            if done:
                break

        # Record per-episode mean variance for baseline estimation
        if ep_variances:
            ep_mean_variances.append(
                float(sum(ep_variances) / len(ep_variances))
            )

        if (ep + 1) % 50 == 0 or ep == num_episodes - 1:
            print(f"  [{phase_name}] ep {ep+1}/{num_episodes}  "
                  f"var={agent.e3._running_variance:.6f}  "
                  f"prec={agent.e3.current_precision:.3f}  "
                  f"commits={sum(commit_decisions)}/{len(commit_decisions)}  "
                  f"harm={total_harm}", flush=True)

    mean_var      = float(sum(running_variance_traj) / max(1, len(running_variance_traj)))
    mean_prec     = float(sum(precision_traj) / max(1, len(precision_traj)))
    commit_rate   = float(sum(commit_decisions) / max(1, len(commit_decisions)))
    mean_pred_err = float(sum(prediction_errors) / max(1, len(prediction_errors)))
    final_var     = running_variance_traj[-1] if running_variance_traj else 0.0
    final_prec    = precision_traj[-1] if precision_traj else 0.0

    # Baseline variance: mean over last BASELINE_WINDOW episodes
    baseline_variance = 0.0
    if record_last_n_variance > 0 and ep_mean_variances:
        window = ep_mean_variances[-record_last_n_variance:]
        baseline_variance = float(sum(window) / len(window))

    print(f"  [{phase_name}] Summary: mean_var={mean_var:.6f}  mean_prec={mean_prec:.3f}  "
          f"commit_rate={commit_rate:.3f}  n_decisions={len(commit_decisions)}"
          + (f"  baseline_var={baseline_variance:.6f}" if record_last_n_variance > 0 else ""),
          flush=True)

    return {
        "mean_running_variance":    mean_var,
        "mean_precision":           mean_prec,
        "commit_rate":              commit_rate,
        "n_commit_decisions":       len(commit_decisions),
        "mean_prediction_error":    mean_pred_err,
        "final_running_variance":   final_var,
        "final_precision":          final_prec,
        "total_harm":               total_harm,
        "total_benefit":            total_benefit,
        "baseline_variance":        baseline_variance,
    }


def run(
    seed: int = 0,
    train_episodes: int = 1000,
    eval_stable_episodes: int = 50,
    eval_perturbed_episodes: int = 50,
    eval_recovery_episodes: int = 30,
    steps_per_episode: int = 200,
    self_dim: int = 32,
    world_dim: int = 32,
    lr: float = 1e-3,
    harm_scale: float = 0.02,
    alpha_world: float = 0.9,
    alpha_self: float = 0.3,
    **kwargs,
) -> dict:
    torch.manual_seed(seed)
    random.seed(seed)

    # ── Environment configs ────────────────────────────────────────────────
    # More extreme contrast than EXQ-018:
    #   Stable:    2 hazards, drift_interval=200, drift_prob=0.0 (essentially static)
    #   Perturbed: 25 hazards, drift_interval=1, drift_prob=1.0 (maximally dynamic)
    env_stable = CausalGridWorld(
        seed=seed, size=12, num_hazards=2, num_resources=5,
        env_drift_interval=200, env_drift_prob=0.0,
        hazard_harm=harm_scale, contaminated_harm=harm_scale,
    )
    env_perturbed = CausalGridWorld(
        seed=seed + 100, size=12, num_hazards=25, num_resources=5,
        env_drift_interval=1, env_drift_prob=1.0,
        hazard_harm=harm_scale, contaminated_harm=harm_scale,
    )

    config = REEConfig.from_dims(
        body_obs_dim=env_stable.body_obs_dim,
        world_obs_dim=env_stable.world_obs_dim,
        action_dim=env_stable.action_dim,
        self_dim=self_dim,
        world_dim=world_dim,
        alpha_world=alpha_world,
        alpha_self=alpha_self,
    )
    agent = REEAgent(config)
    world_decoder = _make_world_decoder(world_dim, env_stable.world_obs_dim)

    params    = list(agent.parameters()) + list(world_decoder.parameters())
    optimizer = optim.Adam(params, lr=lr)

    # ── Phase 1: Training (stable env) ────────────────────────────────────
    print(f"[V3-EXQ-018b] Training: {train_episodes} eps "
          f"(stable: num_hazards=2, drift_interval=200, drift_prob=0.0)", flush=True)
    print(f"  Will record baseline variance from last {BASELINE_WINDOW} episodes.", flush=True)

    train_out = _run_phase(
        agent, env_stable, optimizer, world_decoder,
        train_episodes, steps_per_episode, train=True, phase_name="train",
        record_last_n_variance=BASELINE_WINDOW,
    )

    # ── Sleep recalibration: set relative commit threshold ─────────────────
    training_baseline_variance = train_out["baseline_variance"]
    calibrated_commit_threshold = CALIBRATION_FACTOR * training_baseline_variance

    print(f"\n[V3-EXQ-018b] Sleep recalibration:", flush=True)
    print(f"  training_baseline_variance = {training_baseline_variance:.6f}", flush=True)
    print(f"  calibration_factor         = {CALIBRATION_FACTOR}", flush=True)
    print(f"  commit_threshold (relative) = {calibrated_commit_threshold:.6f}", flush=True)
    print(f"  (Baseline is mean over last {BASELINE_WINDOW} training episodes)", flush=True)
    print(f"  (Threshold = {CALIBRATION_FACTOR}× baseline: commit when variance is within "
          f"normal operating range)", flush=True)

    # Apply calibrated threshold to agent
    agent.e3.config.commitment_threshold = calibrated_commit_threshold

    # ── Phase 2: Eval stable ───────────────────────────────────────────────
    agent.e3._running_variance = config.e3.precision_init
    print(f"\n[V3-EXQ-018b] Eval stable: {eval_stable_episodes} eps "
          f"(commit_threshold={calibrated_commit_threshold:.6f})", flush=True)
    stable_out = _run_phase(
        agent, env_stable, optimizer, world_decoder,
        eval_stable_episodes, steps_per_episode, train=False, phase_name="stable"
    )

    # ── Phase 3: Eval perturbed ────────────────────────────────────────────
    agent.e3._running_variance = config.e3.precision_init
    print(f"\n[V3-EXQ-018b] Eval perturbed: {eval_perturbed_episodes} eps "
          f"(num_hazards=25, drift_interval=1, drift_prob=1.0)", flush=True)
    perturbed_out = _run_phase(
        agent, env_perturbed, optimizer, world_decoder,
        eval_perturbed_episodes, steps_per_episode, train=False, phase_name="perturbed"
    )

    # ── Phase 4: Recovery ─────────────────────────────────────────────────
    agent.e3._running_variance = config.e3.precision_init
    print(f"\n[V3-EXQ-018b] Recovery: {eval_recovery_episodes} eps (stable again)", flush=True)
    recovery_out = _run_phase(
        agent, env_stable, optimizer, world_decoder,
        eval_recovery_episodes, steps_per_episode, train=False, phase_name="recovery"
    )

    # ── PASS / FAIL ───────────────────────────────────────────────────────
    var_diff    = perturbed_out["mean_running_variance"] - stable_out["mean_running_variance"]
    prec_diff   = stable_out["mean_precision"] - perturbed_out["mean_precision"]
    commit_diff = stable_out["commit_rate"] - perturbed_out["commit_rate"]

    # C0: calibration succeeded
    c0_pass = training_baseline_variance > 0

    # C1: RELATIVE threshold — perturbed variance exceeds 0.5× baseline above stable
    # (i.e., var_diff > 0.5 × training_baseline_variance)
    c1_threshold = 0.5 * training_baseline_variance
    c1_pass = var_diff > c1_threshold

    c2_pass = prec_diff > 0.0
    c3_pass = commit_diff > 0.0
    c4_pass = stable_out["n_commit_decisions"] >= 20

    all_pass     = c0_pass and c1_pass and c2_pass and c3_pass and c4_pass
    status       = "PASS" if all_pass else "FAIL"
    criteria_met = sum([c0_pass, c1_pass, c2_pass, c3_pass, c4_pass])

    failure_notes = []
    if not c0_pass:
        failure_notes.append(
            f"C0 FAIL: training_baseline_variance={training_baseline_variance:.6f} <= 0. "
            f"Calibration failed — no variance signal recorded during training."
        )
    if not c1_pass:
        failure_notes.append(
            f"C1 FAIL: var_diff={var_diff:.6f} <= 0.5 × baseline={c1_threshold:.6f}. "
            f"[perturbed={perturbed_out['mean_running_variance']:.6f} "
            f"stable={stable_out['mean_running_variance']:.6f}] "
            f"Perturbed environment does not produce sufficient variance elevation "
            f"relative to training baseline."
        )
    if not c2_pass:
        failure_notes.append(
            f"C2 FAIL: precision_stable={stable_out['mean_precision']:.3f} <= "
            f"precision_perturbed={perturbed_out['mean_precision']:.3f}"
        )
    if not c3_pass:
        failure_notes.append(
            f"C3 FAIL: commit_rate_stable={stable_out['commit_rate']:.3f} <= "
            f"commit_rate_perturbed={perturbed_out['commit_rate']:.3f}. "
            f"ARC-016 behavioral prediction not confirmed."
        )
    if not c4_pass:
        failure_notes.append(
            f"C4 FAIL: n_commit_decisions_stable={stable_out['n_commit_decisions']} < 20"
        )

    print(f"\nV3-EXQ-018b verdict: {status}  ({criteria_met}/5)", flush=True)
    for note in failure_notes:
        print(f"  {note}", flush=True)
    print(f"  training_baseline_variance={training_baseline_variance:.6f}  "
          f"commit_threshold={calibrated_commit_threshold:.6f}", flush=True)
    print(f"  var_diff={var_diff:.6f}  c1_threshold={c1_threshold:.6f}  "
          f"prec_diff={prec_diff:.4f}  commit_diff={commit_diff:.4f}", flush=True)

    metrics = {
        "fatal_error_count": 0.0,
        # Calibration
        "training_baseline_variance":    float(training_baseline_variance),
        "calibrated_commit_threshold":   float(calibrated_commit_threshold),
        "calibration_factor":            float(CALIBRATION_FACTOR),
        "c1_relative_threshold":         float(c1_threshold),
        # Stable phase
        "stable_mean_running_variance":  float(stable_out["mean_running_variance"]),
        "stable_mean_precision":         float(stable_out["mean_precision"]),
        "stable_commit_rate":            float(stable_out["commit_rate"]),
        "stable_n_commit_decisions":     float(stable_out["n_commit_decisions"]),
        "stable_mean_prediction_error":  float(stable_out["mean_prediction_error"]),
        # Perturbed phase
        "perturbed_mean_running_variance": float(perturbed_out["mean_running_variance"]),
        "perturbed_mean_precision":        float(perturbed_out["mean_precision"]),
        "perturbed_commit_rate":           float(perturbed_out["commit_rate"]),
        "perturbed_n_commit_decisions":    float(perturbed_out["n_commit_decisions"]),
        "perturbed_mean_prediction_error": float(perturbed_out["mean_prediction_error"]),
        # Recovery phase
        "recovery_mean_running_variance":  float(recovery_out["mean_running_variance"]),
        "recovery_commit_rate":            float(recovery_out["commit_rate"]),
        # Differences
        "running_variance_diff_perturbed_minus_stable": float(var_diff),
        "precision_diff_stable_minus_perturbed":        float(prec_diff),
        "commit_rate_diff_stable_minus_perturbed":      float(commit_diff),
        "alpha_world": float(alpha_world),
        # Criteria
        "crit0_pass": 1.0 if c0_pass else 0.0,
        "crit1_pass": 1.0 if c1_pass else 0.0,
        "crit2_pass": 1.0 if c2_pass else 0.0,
        "crit3_pass": 1.0 if c3_pass else 0.0,
        "crit4_pass": 1.0 if c4_pass else 0.0,
        "criteria_met": float(criteria_met),
    }

    failure_section = ""
    if failure_notes:
        failure_section = "\n## Failure Notes\n\n" + "\n".join(f"- {n}" for n in failure_notes)

    summary_markdown = f"""# V3-EXQ-018b — ARC-016 Relative Threshold

**Status:** {status}
**Training:** {train_episodes} eps (stable: num_hazards=2, drift_interval=200, drift_prob=0.0)
**Eval stable:** {eval_stable_episodes} eps | **Eval perturbed:** {eval_perturbed_episodes} eps (num_hazards=25, drift_interval=1, drift_prob=1.0)
**Recovery:** {eval_recovery_episodes} eps
**Seed:** {seed}

## Fix vs EXQ-018: Relative Threshold (Sleep Recalibration)

EXQ-018 FAIL: commit_threshold=0.40 was 100× the operating variance range
(stable≈0.0026, perturbed≈0.0036). Absolute threshold calibration is fragile.

Fix: **relative threshold** calibrated from training baseline:
  `commit_threshold = {CALIBRATION_FACTOR} × training_baseline_variance`

`training_baseline_variance` = mean running_variance over last {BASELINE_WINDOW} training episodes.

Biological analogy (sleep recalibration): Slow-wave sleep resets the prediction error
baseline. Waking commitment thresholds are calibrated relative to that baseline.
An agent "waking into" a stable environment establishes a reference variance level,
then commits when variance stays within 2× of that reference, and withholds
commitment when variance exceeds that margin.

## Calibration Results

| Metric | Value |
|---|---|
| training_baseline_variance | {training_baseline_variance:.6f} |
| calibration_factor | {CALIBRATION_FACTOR} |
| calibrated_commit_threshold | {calibrated_commit_threshold:.6f} |
| C1 relative threshold (0.5 × baseline) | {c1_threshold:.6f} |

## Phase Results

| Phase | mean_var | mean_precision | commit_rate | n_decisions |
|---|---|---|---|---|
| Training (stable) | {train_out["mean_running_variance"]:.6f} | {train_out["mean_precision"]:.3f} | {train_out["commit_rate"]:.3f} | {train_out["n_commit_decisions"]} |
| Eval stable | {stable_out["mean_running_variance"]:.6f} | {stable_out["mean_precision"]:.3f} | {stable_out["commit_rate"]:.3f} | {stable_out["n_commit_decisions"]} |
| Eval perturbed | {perturbed_out["mean_running_variance"]:.6f} | {perturbed_out["mean_precision"]:.3f} | {perturbed_out["commit_rate"]:.3f} | {perturbed_out["n_commit_decisions"]} |
| Recovery | {recovery_out["mean_running_variance"]:.6f} | {recovery_out["mean_precision"]:.3f} | {recovery_out["commit_rate"]:.3f} | {recovery_out["n_commit_decisions"]} |

**Key differences:**
- running_variance: perturbed - stable = {var_diff:.6f}  (c1_threshold = {c1_threshold:.6f})
- precision: stable - perturbed = {prec_diff:.4f}
- commit_rate: stable - perturbed = {commit_diff:.4f}

## PASS Criteria

| Criterion | Result | Value |
|---|---|---|
| C0: training_baseline_variance > 0 (calibration succeeded) | {"PASS" if c0_pass else "FAIL"} | {training_baseline_variance:.6f} |
| C1: var_diff > 0.5 × baseline={c1_threshold:.6f} (relative criterion) | {"PASS" if c1_pass else "FAIL"} | {var_diff:.6f} |
| C2: precision_stable > precision_perturbed | {"PASS" if c2_pass else "FAIL"} | {prec_diff:.4f} |
| C3: commit_rate_stable > commit_rate_perturbed (ARC-016) | {"PASS" if c3_pass else "FAIL"} | {commit_diff:.4f} |
| C4: n_stable_decisions >= 20 | {"PASS" if c4_pass else "FAIL"} | {stable_out["n_commit_decisions"]} |

Criteria met: {criteria_met}/5 → **{status}**

## Sleep Recalibration Note

This experiment models the biological observation that sleep establishes a "prediction
error baseline." The training phase (stable environment) corresponds to the waking
period during which the agent learns about its environment. The last {BASELINE_WINDOW}
episodes before sleep are used to compute the baseline variance — this is the agent's
best estimate of "normal" operating variance. The calibrated commit_threshold
({calibrated_commit_threshold:.6f} = {CALIBRATION_FACTOR} × {training_baseline_variance:.6f}) is then used in
the subsequent waking period (eval phases). An agent experiencing a perturbed
environment will have running_variance > threshold → withholds commitment → explores.
An agent in a familiar stable environment will have running_variance < threshold →
commits → exploits.
{failure_section}
"""

    return {
        "status":             status,
        "metrics":            metrics,
        "summary_markdown":   summary_markdown,
        "claim_ids":          CLAIM_IDS,
        "evidence_direction": "supports" if all_pass else ("mixed" if criteria_met >= 3 else "weakens"),
        "experiment_type":    EXPERIMENT_TYPE,
        "fatal_error_count":  0,
    }


if __name__ == "__main__":
    import argparse
    import json
    from datetime import datetime, timezone

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",            type=int,   default=0)
    parser.add_argument("--train-episodes",  type=int,   default=1000)
    parser.add_argument("--eval-stable",     type=int,   default=50)
    parser.add_argument("--eval-perturbed",  type=int,   default=50)
    parser.add_argument("--eval-recovery",   type=int,   default=30)
    parser.add_argument("--steps",           type=int,   default=200)
    parser.add_argument("--harm-scale",      type=float, default=0.02)
    parser.add_argument("--alpha-world",     type=float, default=0.9)
    parser.add_argument("--alpha-self",      type=float, default=0.3)
    args = parser.parse_args()

    result = run(
        seed=args.seed,
        train_episodes=args.train_episodes,
        eval_stable_episodes=args.eval_stable,
        eval_perturbed_episodes=args.eval_perturbed,
        eval_recovery_episodes=args.eval_recovery,
        steps_per_episode=args.steps,
        harm_scale=args.harm_scale,
        alpha_world=args.alpha_world,
        alpha_self=args.alpha_self,
    )

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result["run_timestamp"]      = ts
    result["claim"]              = CLAIM_IDS[0]
    result["verdict"]            = result["status"]
    result["run_id"]             = f"{EXPERIMENT_TYPE}_{ts}_v3"
    result["architecture_epoch"] = "ree_hybrid_guardrails_v1"

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
        print(f"  {k}: {v}", flush=True)
