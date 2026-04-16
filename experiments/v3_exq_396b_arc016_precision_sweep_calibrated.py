#!/opt/local/bin/python3
"""
V3-EXQ-396b -- ARC-016 Precision Sweep (EXP-0094 auto-calibration)

Supersedes: V3-EXQ-396a (which supersedes V3-EXQ-396, which supersedes V3-EXQ-038)

Root cause of EXQ-396a failure:
  Fixed threshold (commitment_threshold=0.40) >> post-training variance (~0.07).
  After 350 training episodes on a mildly-drifting env (drift_prob=0.1), variance
  converges to ~0.07 -- well below 0.40. Both stable and perturbed eval produce
  variances << 0.40, so the agent always commits. C4 always fails; C2/C3 also
  fail since var_diff is near-zero or negative.

EXQ-396b fixes:

  Fix 1 (EXP-0094 auto-calibration):
    After training, compute training_baseline_variance = mean running_variance over
    the last BASELINE_WINDOW training episodes. Set:
        calibrated_threshold = CALIBRATION_FACTOR * training_baseline_variance
    Set agent.e3.config.commitment_threshold = calibrated_threshold in-script
    (E3Config is a mutable dataclass; agent.e3.commit_threshold reads it via property).
    This makes the commit boundary 2x the stable operating variance rather than an
    absolute value calibrated for untrained agents.
    Biological analog: slow-wave sleep recalibrates the prediction-error baseline;
    waking commitment thresholds are proportional to that baseline (EXQ-018b used
    the same calibration_factor=2.0 and passed with baseline ~0.001).

  Fix 2 (truly static training env):
    EXQ-396a trained on drift_prob=0.1 -- residual hazard moves kept baseline at ~0.07.
    EXQ-396b trains on drift_interval=100, drift_prob=0.0 (truly static).
    E2 world_forward can converge to near-zero prediction error, matching EXQ-018b
    which achieved baseline ~0.001 on a static env.

  Fix 3 (calibrated C3 criterion):
    C3 now requires commit_stable > commit_perturbed + 0.05 (meaningful margin),
    not just > 0. This avoids passing on noise-level differences.

Claims: ARC-016, MECH-093

ARC-016 (precision.e3_derived_dynamic_precision): E3-derived precision tracks
environment stability, driving commitment decisions. Tested by C3 (commitment
responds to stability perturbation) and C4 (calibration sanity check).

MECH-093 (z_beta modulates E3 heartbeat frequency): precision boundary scales
with environment danger. Tested by C1/C2 (var_diff monotonic and correlated
with hazard_harm sweep). Note: MECH-093 may still fail even if ARC-016 passes --
hazard_harm affects harm magnitude but not necessarily z_world prediction error
variance. Per-claim evidence_direction handles this split.

Sweep: hazard_harm in [0.005, 0.01, 0.02, 0.05, 0.10]
  For each level:
    1. Train 400 episodes on static env (drift_prob=0.0)
    2. Compute training_baseline_variance = mean(last 100 ep variances)
    3. Set calibrated_threshold = 2.0 * training_baseline_variance
    4. Stable eval (30 eps): same static env, variance inherits from training
    5. Perturbed eval (30 eps): extreme drift (drift_prob=0.9), same starting variance
    6. Record: baseline_var, cal_threshold, var_stable, var_perturbed, var_diff,
               commit_stable, commit_perturbed, commit_diff

PASS criteria (ALL must hold):
  C1: var_diff monotonically increases with hazard_harm across >= 3 consecutive levels
      (more danger -> larger variance gap: tests MECH-093)
  C2: Pearson r(hazard_harm, var_diff) > 0.6 (positive correlation: tests MECH-093)
  C3: >= 3 of 5 levels show commit_stable > commit_perturbed + 0.05
      (commitment responds in correct direction: tests ARC-016)
  C4: >= 3 of 5 levels show calibrated_threshold > var_stable
      (calibration sanity: agent commits during stable eval)
  C5: No fatal errors across all conditions
"""

import sys
import random
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig


EXPERIMENT_TYPE = "v3_exq_396b_arc016_precision_sweep_calibrated"
CLAIM_IDS = ["ARC-016", "MECH-093"]

HAZARD_HARM_LEVELS = [0.005, 0.01, 0.02, 0.05, 0.10]
PROXIMITY_SCALE = 0.05
ALPHA_WORLD = 0.9
ALPHA_SELF = 0.3

CALIBRATION_FACTOR = 2.0  # commit_threshold = CALIBRATION_FACTOR * baseline_variance
BASELINE_WINDOW    = 100  # last N training episodes used to estimate baseline variance


def _action_to_onehot(action_idx: int, num_actions: int, device) -> torch.Tensor:
    v = torch.zeros(1, num_actions, device=device)
    v[0, action_idx] = 1.0
    return v


def _run_one_condition(
    seed: int,
    hazard_harm: float,
    train_episodes: int,
    eval_episodes: int,
    steps_per_episode: int,
    self_dim: int = 32,
    world_dim: int = 32,
    lr: float = 1e-3,
) -> Dict:
    """Train a fresh agent on a static env, auto-calibrate threshold, eval stable/perturbed."""
    torch.manual_seed(seed)
    random.seed(seed)

    # Fix 2: truly static training env (no hazard drift).
    # E2 world_forward converges to near-zero prediction error -> baseline << 0.01.
    env_train = CausalGridWorldV2(
        seed=seed, size=12, num_hazards=4, num_resources=5,
        hazard_harm=hazard_harm,
        env_drift_interval=100, env_drift_prob=0.0,
        proximity_harm_scale=PROXIMITY_SCALE,
        proximity_benefit_scale=PROXIMITY_SCALE * 0.6,
        proximity_approach_threshold=0.15,
        hazard_field_decay=0.5,
    )
    # Fix 3: extreme perturbation contrast (matches EXQ-018b design).
    env_perturbed = CausalGridWorldV2(
        seed=seed, size=12, num_hazards=4, num_resources=5,
        hazard_harm=hazard_harm,
        env_drift_interval=1, env_drift_prob=0.9,
        proximity_harm_scale=PROXIMITY_SCALE,
        proximity_benefit_scale=PROXIMITY_SCALE * 0.6,
        proximity_approach_threshold=0.15,
        hazard_field_decay=0.5,
    )

    config = REEConfig.from_dims(
        body_obs_dim=env_train.body_obs_dim,
        world_obs_dim=env_train.world_obs_dim,
        action_dim=env_train.action_dim,
        self_dim=self_dim,
        world_dim=world_dim,
        alpha_world=ALPHA_WORLD,
        alpha_self=ALPHA_SELF,
        reafference_action_dim=env_train.action_dim,
    )
    agent = REEAgent(config)

    # Separate optimizers (MECH-069: incommensurable error signals)
    standard_params = [
        p for n, p in agent.named_parameters()
        if "world_transition" not in n and "world_action_encoder" not in n
    ]
    wf_params = (
        list(agent.e2.world_transition.parameters()) +
        list(agent.e2.world_action_encoder.parameters())
    )
    optimizer    = optim.Adam(standard_params, lr=lr)
    wf_optimizer = optim.Adam(wf_params,       lr=1e-3)

    wf_data: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    MAX_WF = 5000

    # --- Training ---
    agent.train()
    episode_mean_variances: List[float] = []  # per-episode mean variance for calibration

    for ep in range(train_episodes):
        flat_obs, obs_dict = env_train.reset()
        agent.reset()
        z_world_prev = None
        a_prev = None
        ep_variances: List[float] = []

        for step in range(steps_per_episode):
            obs_body  = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            latent = agent.sense(obs_body, obs_world)
            agent.clock.advance()
            z_world_curr = latent.z_world.detach()

            action_idx = random.randint(0, env_train.action_dim - 1)
            action = _action_to_onehot(action_idx, env_train.action_dim, agent.device)
            agent._last_action = action
            flat_obs, harm_signal, done, info, obs_dict = env_train.step(action)

            if z_world_prev is not None and a_prev is not None:
                wf_data.append((z_world_prev.cpu(), a_prev.cpu(), z_world_curr.cpu()))
                if len(wf_data) > MAX_WF:
                    wf_data = wf_data[-MAX_WF:]

                # Update running_variance from step-by-step E2 prediction error.
                # Critical: variance must accumulate during training (EXQ-396a Fix 1).
                with torch.no_grad():
                    z_pred = agent.e2.world_forward(z_world_prev, a_prev)
                    rv_error = z_world_curr - z_pred.detach()
                    agent.e3.update_running_variance(rv_error)

            ep_variances.append(float(agent.e3._running_variance))

            e1_loss = agent.compute_prediction_loss()
            e2_loss = agent.compute_e2_loss()
            loss = e1_loss + e2_loss
            if loss.requires_grad:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                optimizer.step()

            if len(wf_data) >= 16:
                k = min(32, len(wf_data))
                idxs = torch.randperm(len(wf_data))[:k].tolist()
                zw_b  = torch.cat([wf_data[i][0] for i in idxs]).to(agent.device)
                a_b   = torch.cat([wf_data[i][1] for i in idxs]).to(agent.device)
                zw1_b = torch.cat([wf_data[i][2] for i in idxs]).to(agent.device)
                pred = agent.e2.world_forward(zw_b, a_b)
                wf_loss = F.mse_loss(pred, zw1_b)
                if wf_loss.requires_grad:
                    wf_optimizer.zero_grad()
                    wf_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        agent.e2.world_transition.parameters(), 0.5
                    )
                    wf_optimizer.step()

            z_world_prev = z_world_curr
            a_prev = action.detach()
            if done:
                break

        if ep_variances:
            episode_mean_variances.append(float(np.mean(ep_variances)))

    var_after_training = float(agent.e3._running_variance)

    # --- EXP-0094: Auto-calibrate commit threshold ---
    # Use mean variance over the last BASELINE_WINDOW training episodes.
    # Training on a static env -> E2 converges well -> baseline << 0.01.
    # calibrated_threshold = 2x baseline ensures stable eval commits and
    # perturbed eval (prediction errors spike) can break the threshold.
    window = episode_mean_variances[-BASELINE_WINDOW:] if len(episode_mean_variances) >= BASELINE_WINDOW \
        else episode_mean_variances
    training_baseline_variance = float(np.mean(window)) if window else var_after_training
    calibrated_threshold = CALIBRATION_FACTOR * training_baseline_variance

    # Set calibrated threshold on E3 config (mutable dataclass; read by commit_threshold property).
    agent.e3.config.commitment_threshold = calibrated_threshold

    print(
        f"  [hazard_harm={hazard_harm:.3f}]"
        f"  post-train var={var_after_training:.6f}"
        f"  baseline(last {len(window)} eps)={training_baseline_variance:.6f}"
        f"  cal_threshold={calibrated_threshold:.6f}",
        flush=True,
    )

    # --- Eval phase ---
    # Variance carries over from training (no per-episode reset).
    # Stable eval and perturbed eval both start from var_after_training (fair comparison).
    def _eval_variance(env_eval, n_eps, inherit_variance: float):
        agent.eval()
        agent.e3._running_variance = inherit_variance
        variances = []
        commit_decisions = []
        fatal_errors = 0

        for ep in range(n_eps):
            flat_obs, obs_dict = env_eval.reset()
            # Reset episode-level fields only; do NOT reset _running_variance.
            agent._step_count = 0
            agent._harm_this_episode = 0.0
            agent._committed_candidates = None
            agent._last_action = None
            agent.clock.reset()
            agent.theta_buffer.reset()
            agent.beta_gate.reset()
            agent.e1.reset_hidden_state()
            agent._current_latent = agent.latent_stack.init_state(
                batch_size=1, device=agent.device
            )
            z_world_prev_eval = None

            for step in range(steps_per_episode):
                obs_body  = obs_dict["body_state"]
                obs_world = obs_dict["world_state"]
                try:
                    with torch.no_grad():
                        latent = agent.sense(obs_body, obs_world)
                        agent.clock.advance()
                        z_world_curr = latent.z_world

                    action_idx = random.randint(0, env_eval.action_dim - 1)
                    action = _action_to_onehot(action_idx, env_eval.action_dim, agent.device)
                    agent._last_action = action
                    flat_obs, harm_signal, done, info, obs_dict = env_eval.step(action)

                    if z_world_prev_eval is not None:
                        with torch.no_grad():
                            z_pred = agent.e2.world_forward(z_world_prev_eval, action)
                            pred_error = z_world_curr - z_pred
                            agent.e3.update_running_variance(pred_error)

                    rv = float(agent.e3._running_variance)
                    variances.append(rv)

                    # Commit decision sampled every 10 steps
                    if step % 10 == 0:
                        with torch.no_grad():
                            z_self = latent.z_self
                            try:
                                candidates = agent.e2.generate_candidates_random(
                                    initial_z_self=z_self,
                                    initial_z_world=z_world_curr,
                                    num_candidates=8,
                                    horizon=3,
                                    compute_action_objects=True,
                                )
                                selection = agent.e3.select(candidates)
                                commit_decisions.append(
                                    1.0 if selection.committed else 0.0
                                )
                            except Exception:
                                fatal_errors += 1

                    z_world_prev_eval = z_world_curr

                except Exception:
                    fatal_errors += 1

                if done:
                    break

        mean_var    = float(np.mean(variances))        if variances        else 0.0
        commit_rate = float(np.mean(commit_decisions)) if commit_decisions else 0.0
        end_var     = float(agent.e3._running_variance)
        return mean_var, commit_rate, fatal_errors, end_var

    print(
        f"  [hazard_harm={hazard_harm:.3f}] Stable eval ({eval_episodes} eps)...",
        flush=True,
    )
    var_stable, commit_stable, err_stable, _ = _eval_variance(
        env_train, eval_episodes, inherit_variance=var_after_training,
    )

    print(
        f"  [hazard_harm={hazard_harm:.3f}] Perturbed eval ({eval_episodes} eps)...",
        flush=True,
    )
    var_perturbed, commit_perturbed, err_perturbed, _ = _eval_variance(
        env_perturbed, eval_episodes, inherit_variance=var_after_training,
    )

    var_diff    = var_perturbed - var_stable
    commit_diff = commit_stable - commit_perturbed

    print(
        f"  [hazard_harm={hazard_harm:.3f}]"
        f"  var_stable={var_stable:.6f}"
        f"  var_perturbed={var_perturbed:.6f}"
        f"  var_diff={var_diff:.6f}"
        f"  commit_stable={commit_stable:.3f}"
        f"  commit_perturbed={commit_perturbed:.3f}",
        flush=True,
    )

    return {
        "hazard_harm":                hazard_harm,
        "var_after_training":         var_after_training,
        "training_baseline_variance": training_baseline_variance,
        "calibrated_threshold":       calibrated_threshold,
        "var_stable":                 var_stable,
        "var_perturbed":              var_perturbed,
        "var_diff":                   var_diff,
        "commit_stable":              commit_stable,
        "commit_perturbed":           commit_perturbed,
        "commit_diff":                commit_diff,
        "fatal_errors":               err_stable + err_perturbed,
    }


def run(
    seed: int = 0,
    train_episodes: int = 400,
    eval_episodes: int = 30,
    steps_per_episode: int = 200,
    **kwargs,
) -> dict:
    print(
        f"[V3-EXQ-396b] ARC-016 Precision Sweep -- EXP-0094 auto-calibration\n"
        f"  hazard_harm levels: {HAZARD_HARM_LEVELS}\n"
        f"  train={train_episodes} eps  eval={eval_episodes} eps  steps={steps_per_episode}\n"
        f"  calibration_factor={CALIBRATION_FACTOR}x  baseline_window={BASELINE_WINDOW} eps",
        flush=True,
    )

    results_by_level: List[Dict] = []
    fatal_total = 0

    for hazard_harm in HAZARD_HARM_LEVELS:
        print(f"\n[V3-EXQ-396b] === hazard_harm={hazard_harm:.3f} ===", flush=True)
        res = _run_one_condition(
            seed=seed,
            hazard_harm=hazard_harm,
            train_episodes=train_episodes,
            eval_episodes=eval_episodes,
            steps_per_episode=steps_per_episode,
        )
        results_by_level.append(res)
        fatal_total += res["fatal_errors"]

    # --- Criterion evaluation ---
    var_diffs      = [r["var_diff"]                  for r in results_by_level]
    commit_diffs   = [r["commit_diff"]               for r in results_by_level]
    harm_levels    = [r["hazard_harm"]               for r in results_by_level]
    cal_thresholds = [r["calibrated_threshold"]      for r in results_by_level]
    var_stables    = [r["var_stable"]                for r in results_by_level]

    # C1: var_diff monotonically increases with hazard_harm (>= 3 consecutive increases)
    monotone_count = sum(
        1 for i in range(1, len(var_diffs))
        if var_diffs[i] > var_diffs[i - 1]
    )
    c1_pass = monotone_count >= 3

    # C2: Pearson r(hazard_harm, var_diff) > 0.6
    if len(harm_levels) >= 3:
        r_val = float(np.corrcoef(harm_levels, var_diffs)[0, 1])
    else:
        r_val = 0.0
    c2_pass = r_val > 0.6

    # C3: >= 3 of 5 levels show commit_stable > commit_perturbed + 0.05
    correct_direction = sum(
        1 for cd in commit_diffs if cd > 0.05
    )
    c3_pass = correct_direction >= 3

    # C4: >= 3 of 5 levels have calibrated_threshold > var_stable
    # (calibration sanity: stable variance is below threshold -> agent commits)
    calibration_ok = sum(
        1 for thr, vs in zip(cal_thresholds, var_stables)
        if thr > vs
    )
    c4_pass = calibration_ok >= 3

    # C5: no fatal errors
    c5_pass = (fatal_total == 0)

    all_pass = c1_pass and c2_pass and c3_pass and c4_pass and c5_pass
    status   = "PASS" if all_pass else "FAIL"
    n_met    = sum([c1_pass, c2_pass, c3_pass, c4_pass, c5_pass])

    failure_notes = []
    if not c1_pass:
        failure_notes.append(
            f"C1 FAIL: only {monotone_count}/4 consecutive monotone increases in var_diff"
        )
    if not c2_pass:
        failure_notes.append(
            f"C2 FAIL: Pearson r(hazard_harm, var_diff) = {r_val:.3f} <= 0.6"
        )
    if not c3_pass:
        failure_notes.append(
            f"C3 FAIL: only {correct_direction}/5 levels show"
            f" commit_stable > commit_perturbed + 0.05"
        )
    if not c4_pass:
        failure_notes.append(
            f"C4 FAIL: only {calibration_ok}/5 levels have"
            f" calibrated_threshold > var_stable"
        )
    if not c5_pass:
        failure_notes.append(f"C5 FAIL: {fatal_total} fatal errors")

    print(f"\n[V3-EXQ-396b] Summary:", flush=True)
    print(
        f"  {'harm':>6}  {'baseline':>10}  {'cal_thr':>10}"
        f"  {'var_diff':>10}  {'commit_diff':>12}",
        flush=True,
    )
    for r in results_by_level:
        print(
            f"  {r['hazard_harm']:.3f}   "
            f"{r['training_baseline_variance']:.6f}  "
            f"{r['calibrated_threshold']:.6f}  "
            f"{r['var_diff']:.6f}  "
            f"{r['commit_diff']:.4f}",
            flush=True,
        )
    print(f"  Pearson r(harm, var_diff) = {r_val:.3f}", flush=True)
    print(f"\nV3-EXQ-396b verdict: {status}  ({n_met}/5)", flush=True)
    for note in failure_notes:
        print(f"  {note}", flush=True)

    # --- Per-claim evidence direction ---
    # ARC-016: tested by C3 (commitment responds to stability) + C4 (calibration ok).
    # MECH-093: tested by C1/C2 (var_diff scales with hazard_harm).
    arc016_pass  = c3_pass and c4_pass
    mech093_pass = c1_pass and c2_pass

    arc016_dir  = "supports"      if arc016_pass  else "does_not_support"
    mech093_dir = "supports"      if mech093_pass else "does_not_support"
    # Upgrade to weakens only if the relevant sub-criteria clearly failed
    if not arc016_pass and not (c3_pass or c4_pass):
        arc016_dir = "weakens"
    if not mech093_pass and not (c1_pass or c2_pass):
        mech093_dir = "weakens"

    evidence_direction_per_claim = {
        "ARC-016":  arc016_dir,
        "MECH-093": mech093_dir,
    }
    overall_evidence_direction = (
        "supports" if all_pass
        else ("mixed" if n_met >= 3 else "weakens")
    )

    # Flatten per-level metrics
    metrics = {
        "calibration_factor":     float(CALIBRATION_FACTOR),
        "pearson_r_harm_vardiff": float(r_val),
        "monotone_increases":     float(monotone_count),
        "levels_correct_commit":  float(correct_direction),
        "levels_calibration_ok":  float(calibration_ok),
        "fatal_total":            float(fatal_total),
        "crit1_pass": 1.0 if c1_pass else 0.0,
        "crit2_pass": 1.0 if c2_pass else 0.0,
        "crit3_pass": 1.0 if c3_pass else 0.0,
        "crit4_pass": 1.0 if c4_pass else 0.0,
        "crit5_pass": 1.0 if c5_pass else 0.0,
        "criteria_met": float(n_met),
    }
    for r in results_by_level:
        key = f"harm_{str(r['hazard_harm']).replace('.', 'p')}"
        metrics[f"{key}_baseline_var"]         = float(r["training_baseline_variance"])
        metrics[f"{key}_calibrated_threshold"] = float(r["calibrated_threshold"])
        metrics[f"{key}_var_diff"]             = float(r["var_diff"])
        metrics[f"{key}_var_stable"]           = float(r["var_stable"])
        metrics[f"{key}_var_perturbed"]        = float(r["var_perturbed"])
        metrics[f"{key}_commit_stable"]        = float(r["commit_stable"])
        metrics[f"{key}_commit_perturbed"]     = float(r["commit_perturbed"])

    sweep_rows = ""
    for r in results_by_level:
        sweep_rows += (
            f"| {r['hazard_harm']:.3f}"
            f" | {r['training_baseline_variance']:.6f}"
            f" | {r['calibrated_threshold']:.6f}"
            f" | {r['var_stable']:.6f}"
            f" | {r['var_perturbed']:.6f}"
            f" | {r['var_diff']:.6f}"
            f" | {r['commit_stable']:.3f}"
            f" | {r['commit_perturbed']:.3f} |\n"
        )

    failure_section = (
        "\n## Failure Notes\n\n" + "\n".join(f"- {n}" for n in failure_notes)
        if failure_notes else ""
    )

    summary_markdown = (
        "# V3-EXQ-396b -- ARC-016 Precision Sweep (EXP-0094 auto-calibration)\n\n"
        f"**Status:** {status}\n"
        f"**Claims:** ARC-016, MECH-093\n"
        f"**World:** CausalGridWorldV2 (proximity_scale={PROXIMITY_SCALE})\n"
        f"**alpha_world:** {ALPHA_WORLD}  (SD-008)\n"
        f"**Hazard harm levels:** {HAZARD_HARM_LEVELS}\n"
        f"**Calibration factor:** {CALIBRATION_FACTOR}x (EXP-0094)\n"
        f"**Supersedes:** V3-EXQ-396a -> V3-EXQ-396 -> V3-EXQ-038\n\n"
        "## Root Cause Fixed\n\n"
        "EXQ-396a fixed training/eval variance bugs but still failed: fixed threshold (0.40)\n"
        ">> post-training variance (~0.07); agent always committed; C3/C4 always failed.\n\n"
        "EXQ-396b fixes:\n"
        "1. EXP-0094 auto-calibration: threshold = 2x mean(last 100 ep variances).\n"
        "   threshold ~= 2 * baseline, so stable eval commits and perturbed eval can break it.\n"
        "2. Truly static training env (drift_prob=0.0): baseline drops from ~0.07 to ~0.001.\n"
        "3. C3 criterion requires +0.05 margin to avoid passing on noise.\n\n"
        "Biological analog: SWS recalibrates prediction-error baseline;\n"
        "waking thresholds are 2x that baseline (EXQ-018b: same factor, PASS).\n\n"
        "## Sweep Results\n\n"
        "| hazard_harm | baseline_var | cal_threshold | var_stable"
        " | var_perturbed | var_diff | commit_stable | commit_perturbed |\n"
        "|---|---|---|---|---|---|---|---|\n"
        f"{sweep_rows}"
        f"**Pearson r(hazard_harm, var_diff):** {r_val:.3f}\n"
        f"**Monotone increases in var_diff:** {monotone_count}/4\n\n"
        "## PASS Criteria\n\n"
        "| Criterion | Result | Value |\n|---|---|---|\n"
        f"| C1: var_diff monotone (3+ consec) | {'PASS' if c1_pass else 'FAIL'}"
        f" | {monotone_count}/4 |\n"
        f"| C2: Pearson r > 0.6 | {'PASS' if c2_pass else 'FAIL'}"
        f" | {r_val:.3f} |\n"
        f"| C3: >= 3 levels commit_stable > commit_perturbed+0.05"
        f" | {'PASS' if c3_pass else 'FAIL'} | {correct_direction}/5 |\n"
        f"| C4: >= 3 levels cal_threshold > var_stable"
        f" | {'PASS' if c4_pass else 'FAIL'} | {calibration_ok}/5 |\n"
        f"| C5: no fatal errors | {'PASS' if c5_pass else 'FAIL'}"
        f" | {fatal_total} errors |\n\n"
        f"Criteria met: {n_met}/5 -> **{status}**\n"
        f"{failure_section}\n\n"
        "## Per-Claim Evidence Direction\n\n"
        f"- ARC-016 (dynamic precision drives commitment): {arc016_dir}\n"
        f"  (C3 + C4; mechanism works regardless of hazard_harm scaling)\n"
        f"- MECH-093 (precision scales with hazard danger): {mech093_dir}\n"
        f"  (C1 + C2; hazard_harm magnitude must modulate var_diff)\n"
    )

    return {
        "status":                       status,
        "metrics":                      metrics,
        "summary_markdown":             summary_markdown,
        "claim_ids":                    CLAIM_IDS,
        "evidence_direction":           overall_evidence_direction,
        "evidence_direction_per_claim": evidence_direction_per_claim,
        "experiment_type":              EXPERIMENT_TYPE,
        "fatal_error_count":            fatal_total,
    }


if __name__ == "__main__":
    import argparse
    import json
    from datetime import datetime, timezone

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",           type=int,  default=0)
    parser.add_argument("--train-episodes", type=int,  default=400)
    parser.add_argument("--eval-eps",       type=int,  default=30)
    parser.add_argument("--steps",          type=int,  default=200)
    parser.add_argument("--dry-run",        action="store_true",
                        help="Smoke test: 5 train eps, 3 eval eps, 20 steps")
    args = parser.parse_args()

    if args.dry_run:
        args.train_episodes = 5
        args.eval_eps       = 3
        args.steps          = 20
        print("[DRY RUN] Reduced to 5 train / 3 eval / 20 steps per condition.", flush=True)

    result = run(
        seed=args.seed,
        train_episodes=args.train_episodes,
        eval_episodes=args.eval_eps,
        steps_per_episode=args.steps,
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
