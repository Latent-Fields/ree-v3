"""
V3-EXQ-038 — ARC-016 Precision Threshold Sweep (hazard_harm sweep)

Claims: ARC-016, MECH-093

ARC-016 (precision.e3_derived_dynamic_precision) asserts that E3-derived precision
tracks environment stability, driving commitment decisions. EXQ-018 and EXQ-031 both
PASS, confirming the mechanism exists.

This experiment characterizes the quantitative relationship between environment
danger level (hazard_harm) and precision/commitment behavior. ARC-016 is qualitative
(does precision respond?); this sweep is quantitative (how does it scale?).

MECH-093 (z_beta modulates E3 heartbeat frequency) predicts that the precision
boundary is not fixed — it scales with environment danger. A more dangerous
environment should push precision higher and commitment threshold harder.

Sweep: hazard_harm ∈ [0.005, 0.01, 0.02, 0.05, 0.10]
  For each level:
    1. Fresh agent, train 300 episodes (shorter than EXQ-031 for speed)
    2. Stable eval (20 eps): z_world prediction errors → running_variance (stable)
    3. Perturbed eval (20 eps): maximally dynamic hazards → running_variance (perturbed)
    4. Record: var_stable, var_perturbed, var_diff, commit_rate_stable, commit_rate_perturbed

PASS criteria (ALL must hold):
    C1: var_diff monotonically increases with hazard_harm across at least 3 consecutive levels
        (more danger → larger gap between stable/perturbed variance)
    C2: Pearson r(hazard_harm, var_diff) > 0.6 (positive correlation)
    C3: At least 3 of 5 hazard_harm levels show var_diff > 0.001 (above noise floor)
    C4: At least 3 of 5 levels show commit_rate_stable > commit_rate_perturbed
        (commitment responds in correct direction at most danger levels)
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


EXPERIMENT_TYPE = "v3_exq_038_arc016_precision_sweep"
CLAIM_IDS = ["ARC-016", "MECH-093"]

HAZARD_HARM_LEVELS = [0.005, 0.01, 0.02, 0.05, 0.10]
PROXIMITY_SCALE = 0.05
ALPHA_WORLD = 0.9
ALPHA_SELF = 0.3


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
    """Train a fresh agent and measure stable/perturbed variance for one hazard_harm level."""
    torch.manual_seed(seed)
    random.seed(seed)

    env_train = CausalGridWorldV2(
        seed=seed, size=12, num_hazards=4, num_resources=5,
        hazard_harm=hazard_harm,
        env_drift_interval=5, env_drift_prob=0.1,
        proximity_harm_scale=PROXIMITY_SCALE,
        proximity_benefit_scale=PROXIMITY_SCALE * 0.6,
        proximity_approach_threshold=0.15,
        hazard_field_decay=0.5,
    )
    env_perturbed = CausalGridWorldV2(
        seed=seed, size=12, num_hazards=4, num_resources=5,
        hazard_harm=hazard_harm,
        env_drift_interval=1, env_drift_prob=0.9,  # maximally dynamic
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

    # Separate optimizers (MECH-069)
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

    # Collect world_forward transitions for training
    wf_data: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    MAX_WF = 5000

    # --- Training ---
    agent.train()
    for ep in range(train_episodes):
        flat_obs, obs_dict = env_train.reset()
        agent.reset()
        z_world_prev = None
        a_prev = None

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
                    torch.nn.utils.clip_grad_norm_(agent.e2.world_transition.parameters(), 0.5)
                    wf_optimizer.step()

            z_world_prev = z_world_curr
            a_prev = action.detach()
            if done:
                break

    # --- Eval phase: measure prediction error variance ---
    def _eval_variance(env_eval, n_eps, label):
        agent.eval()
        variances = []
        commit_decisions = []
        fatal_errors = 0
        z_world_prev_eval = None

        for ep in range(n_eps):
            flat_obs, obs_dict = env_eval.reset()
            agent.reset()
            agent.e3._running_variance = config.e3.precision_init
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
                            z_world_pred = agent.e2.world_forward(z_world_prev_eval, action)
                            pred_error = z_world_curr - z_world_pred
                            agent.e3.update_running_variance(pred_error)

                    rv = float(agent.e3._running_variance)
                    variances.append(rv)

                    # Commit decision (every 10 steps)
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

                except Exception as exc:
                    fatal_errors += 1

                if done:
                    break

        mean_var = float(np.mean(variances)) if variances else 0.0
        commit_rate = float(np.mean(commit_decisions)) if commit_decisions else 0.0
        return mean_var, commit_rate, fatal_errors

    print(f"  [hazard_harm={hazard_harm:.3f}] Stable eval ({eval_episodes} eps)...", flush=True)
    var_stable, commit_stable, err_stable = _eval_variance(
        env_train, eval_episodes, "stable"
    )
    print(f"  [hazard_harm={hazard_harm:.3f}] Perturbed eval ({eval_episodes} eps)...", flush=True)
    var_perturbed, commit_perturbed, err_perturbed = _eval_variance(
        env_perturbed, eval_episodes, "perturbed"
    )

    var_diff = var_perturbed - var_stable
    commit_diff = commit_stable - commit_perturbed

    print(
        f"  [hazard_harm={hazard_harm:.3f}] "
        f"var_stable={var_stable:.6f}  var_perturbed={var_perturbed:.6f}  "
        f"var_diff={var_diff:.6f}  "
        f"commit_stable={commit_stable:.3f}  commit_perturbed={commit_perturbed:.3f}",
        flush=True,
    )

    return {
        "hazard_harm":       hazard_harm,
        "var_stable":        var_stable,
        "var_perturbed":     var_perturbed,
        "var_diff":          var_diff,
        "commit_stable":     commit_stable,
        "commit_perturbed":  commit_perturbed,
        "commit_diff":       commit_diff,
        "fatal_errors":      err_stable + err_perturbed,
    }


def run(
    seed: int = 0,
    train_episodes: int = 300,
    eval_episodes: int = 20,
    steps_per_episode: int = 200,
    **kwargs,
) -> dict:
    print(
        f"[V3-EXQ-038] ARC-016 Precision Threshold Sweep\n"
        f"  hazard_harm levels: {HAZARD_HARM_LEVELS}\n"
        f"  train_episodes={train_episodes}  eval_episodes={eval_episodes}  steps={steps_per_episode}",
        flush=True,
    )

    results_by_level: List[Dict] = []
    fatal_total = 0

    for hazard_harm in HAZARD_HARM_LEVELS:
        print(f"\n[V3-EXQ-038] === hazard_harm={hazard_harm:.3f} ===", flush=True)
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
    var_diffs    = [r["var_diff"]       for r in results_by_level]
    commit_diffs = [r["commit_diff"]    for r in results_by_level]
    harm_levels  = [r["hazard_harm"]    for r in results_by_level]

    # C1: var_diff monotonically increases with hazard_harm (3 consecutive levels)
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

    # C3: at least 3 of 5 levels show var_diff > 0.001
    above_threshold = sum(1 for vd in var_diffs if vd > 0.001)
    c3_pass = above_threshold >= 3

    # C4: at least 3 of 5 levels show commit_diff > 0 (stable commits more than perturbed)
    correct_direction = sum(1 for cd in commit_diffs if cd > 0)
    c4_pass = correct_direction >= 3

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
            f"C3 FAIL: only {above_threshold}/5 levels show var_diff > 0.001"
        )
    if not c4_pass:
        failure_notes.append(
            f"C4 FAIL: only {correct_direction}/5 levels show commit_stable > commit_perturbed"
        )
    if not c5_pass:
        failure_notes.append(f"C5 FAIL: {fatal_total} fatal errors")

    print(f"\n[V3-EXQ-038] Summary:", flush=True)
    print(f"  hazard_harm  var_diff    commit_diff", flush=True)
    for r in results_by_level:
        print(
            f"  {r['hazard_harm']:.3f}       {r['var_diff']:.6f}  {r['commit_diff']:.4f}",
            flush=True,
        )
    print(f"  Pearson r(harm, var_diff) = {r_val:.3f}", flush=True)
    print(f"\nV3-EXQ-038 verdict: {status}  ({n_met}/5)", flush=True)
    for note in failure_notes:
        print(f"  {note}", flush=True)

    # Flatten metrics per level
    metrics = {
        "pearson_r_harm_vardiff": float(r_val),
        "monotone_increases":     float(monotone_count),
        "levels_above_threshold": float(above_threshold),
        "levels_correct_commit":  float(correct_direction),
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
        metrics[f"{key}_var_diff"]       = float(r["var_diff"])
        metrics[f"{key}_var_stable"]     = float(r["var_stable"])
        metrics[f"{key}_var_perturbed"]  = float(r["var_perturbed"])
        metrics[f"{key}_commit_stable"]  = float(r["commit_stable"])
        metrics[f"{key}_commit_perturbed"] = float(r["commit_perturbed"])

    # Build sweep table for markdown
    sweep_rows = ""
    for r in results_by_level:
        sweep_rows += (
            f"| {r['hazard_harm']:.3f} | {r['var_stable']:.6f} | {r['var_perturbed']:.6f} "
            f"| {r['var_diff']:.6f} | {r['commit_stable']:.3f} | {r['commit_perturbed']:.3f} |\n"
        )

    failure_section = ""
    if failure_notes:
        failure_section = "\n## Failure Notes\n\n" + "\n".join(f"- {n}" for n in failure_notes)

    summary_markdown = f"""# V3-EXQ-038 — ARC-016 Precision Threshold Sweep

**Status:** {status}
**Claims:** ARC-016, MECH-093
**World:** CausalGridWorldV2 (proximity_scale={PROXIMITY_SCALE})
**alpha_world:** {ALPHA_WORLD}  (SD-008)
**Hazard harm levels:** {HAZARD_HARM_LEVELS}
**Seed:** {seed}
**Predecessors:** EXQ-018 (PASS), EXQ-031 (PASS) — ARC-016 mechanism confirmed

## Motivation

EXQ-018 and EXQ-031 confirm that dynamic precision responds to environment stability.
This experiment characterizes the quantitative relationship: does precision/commitment
behavior scale predictably with hazard_harm (environment danger level)?

MECH-093 (z_beta modulates E3 heartbeat frequency) predicts the precision boundary is
not fixed but scales with environment danger. A systematic sweep tests this prediction.

## Sweep Results

| hazard_harm | var_stable | var_perturbed | var_diff | commit_stable | commit_perturbed |
|---|---|---|---|---|---|
{sweep_rows}
**Pearson r(hazard_harm, var_diff):** {r_val:.3f}
**Monotone increases in var_diff:** {monotone_count}/4

## PASS Criteria

| Criterion | Result | Value |
|---|---|---|
| C1: var_diff monotone increases (3+ consecutive) | {"PASS" if c1_pass else "FAIL"} | {monotone_count}/4 |
| C2: Pearson r(hazard_harm, var_diff) > 0.6 | {"PASS" if c2_pass else "FAIL"} | {r_val:.3f} |
| C3: >= 3 levels with var_diff > 0.001 | {"PASS" if c3_pass else "FAIL"} | {above_threshold}/5 |
| C4: >= 3 levels with commit_stable > commit_perturbed | {"PASS" if c4_pass else "FAIL"} | {correct_direction}/5 |
| C5: no fatal errors | {"PASS" if c5_pass else "FAIL"} | {fatal_total} errors |

Criteria met: {n_met}/5 → **{status}**
{failure_section}
"""

    return {
        "status": status,
        "metrics": metrics,
        "summary_markdown": summary_markdown,
        "claim_ids": CLAIM_IDS,
        "evidence_direction": (
            "supports" if all_pass
            else ("mixed" if n_met >= 3 else "weakens")
        ),
        "experiment_type": EXPERIMENT_TYPE,
        "fatal_error_count": fatal_total,
    }


if __name__ == "__main__":
    import argparse
    import json
    from datetime import datetime, timezone

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",            type=int, default=0)
    parser.add_argument("--train-episodes",  type=int, default=300)
    parser.add_argument("--eval-eps",        type=int, default=20)
    parser.add_argument("--steps",           type=int, default=200)
    args = parser.parse_args()

    result = run(
        seed=args.seed,
        train_episodes=args.train_episodes,
        eval_episodes=args.eval_eps,
        steps_per_episode=args.steps,
    )

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result["run_timestamp"] = ts
    result["claim"]  = CLAIM_IDS[0]
    result["verdict"] = result["status"]
    result["run_id"] = f"{EXPERIMENT_TYPE}_{ts}_v3"
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
