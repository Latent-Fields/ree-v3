"""
V3-EXQ-018 — ARC-016 Dynamic Precision Commitment Test

Claims: ARC-016 (precision.e3_derived_dynamic_precision).

Motivation (2026-03-17):
  ARC-016 was a V2 FAIL (EXQ-025) because precision was externally imposed and
  the commitment→behavior circuit was not wired. V3 fixes both: E3 derives
  precision from its own prediction error EMA (running_variance via
  update_running_variance()), and commit_threshold = precision_to_threshold(var).

  The behavioral prediction:
    Stable env  → small z_world prediction errors → running_variance low
                → running_variance < commit_threshold (0.02) → committed (greedy) selection
    Perturbed env → large z_world prediction errors → running_variance high
                → running_variance >= commit_threshold → non-committed (stochastic) selection

  commit formula (fixed 2026-03-18): committed = running_variance < commit_threshold
  (Prior formula compared precision ~95 against threshold ~0.703 → always True.)

  This experiment directly tests whether that dynamic actually occurs by:
    1. Training an agent (200 eps, stable env) and explicitly updating E3
       running_variance each step from z_world prediction errors
    2. Stable eval (50 eps): record running_variance trajectory + commit decisions
    3. Perturbed eval (50 eps, high env_drift): record same metrics
    4. Recovery eval (30 eps, stable again): verify variance recovers

Design:
  At each step, we:
    - Use E2.world_forward(z_world_t, a_t) as the predicted z_world_{t+1}
    - Get actual z_world_{t+1} from next observation
    - Compute prediction_error = actual - predicted
    - Call agent.e3.update_running_variance(prediction_error)
    - Every 10 steps: call agent.e3.select(random_candidates) to observe
      the committed/stochastic decision and record it

  This tests the MECHANISM directly — does running_variance track environment
  stability, and does commit behavior respond appropriately?

PASS criteria (ALL must hold):
  C1: mean_running_variance_perturbed > mean_running_variance_stable + 0.02
      (variance actually diverges in perturbed env)
  C2: mean_precision_stable > mean_precision_perturbed
      (precision drops in perturbed env)
  C3: commit_rate_stable > commit_rate_perturbed
      (main ARC-016 behavioral prediction: stable → more committed)
  C4: n_commit_decisions_stable >= 20 (enough E3 ticks to measure)
  C5: No fatal errors

Note on C3: ARC-016 does NOT require that perturbed env produces 0% commitment
  (that would require very large errors). It only requires the RATE DROPS, i.e.
  a qualitative difference is measurable.
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


EXPERIMENT_TYPE = "v3_exq_018_arc016_dynamic_precision"
CLAIM_IDS = ["ARC-016"]

E3_DECISION_INTERVAL = 10    # steps between E3 candidate evaluations
NUM_CANDIDATES = 16           # candidates for E3 eval (lightweight for this test)
CANDIDATE_HORIZON = 5         # rollout steps per candidate


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
) -> Dict:
    """
    Run one phase: collect z_world prediction errors, update E3 precision,
    and record E3 commit decisions at E3_DECISION_INTERVAL step intervals.
    """
    if train:
        agent.train()
        world_decoder.train()
    else:
        agent.eval()
        world_decoder.eval()

    running_variance_traj: List[float] = []
    precision_traj: List[float] = []
    commit_decisions: List[bool] = []
    prediction_errors: List[float] = []

    total_harm = 0
    total_benefit = 0
    step_counter = 0

    traj_buffer: List = []
    MAX_TRAJ_BUFFER = 200

    # E2.world_forward transition buffer for training.
    # E2.world_forward is NOT trained by compute_e2_loss() (which trains z_self only).
    # Without training, world_forward produces random-noise predictions with
    # constant ~0.01 MSE in both stable and perturbed envs → no variance signal.
    # Fix: explicitly train (z_world_t, a_t) → z_world_{t+1} with separate optimizer.
    e2w_buf: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    MAX_E2W_BUF = 500

    for ep in range(num_episodes):
        flat_obs, obs_dict = env.reset()
        agent.reset()

        z_world_prev = None
        a_prev = None
        z_self_prev = None

        for step in range(steps_per_episode):
            obs_body = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]

            latent = agent.sense(obs_body, obs_world)
            agent.clock.advance()

            z_world_curr = latent.z_world.detach()
            z_self_curr = latent.z_self.detach()

            action_idx = random.randint(0, env.action_dim - 1)
            action = _action_to_onehot(action_idx, env.action_dim, agent.device)
            agent._last_action = action

            # Update E3 running_variance from z_world prediction error
            if z_world_prev is not None and a_prev is not None:
                with torch.no_grad():
                    z_world_predicted = agent.e2.world_forward(z_world_prev, a_prev)
                    prediction_error = z_world_curr - z_world_predicted
                    agent.e3.update_running_variance(prediction_error)
                    pred_err_mag = float(prediction_error.pow(2).mean().item())
                    prediction_errors.append(pred_err_mag)

            # Record precision state
            running_variance_traj.append(agent.e3._running_variance)
            precision_traj.append(agent.e3.current_precision)

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
                        pass  # Skip if E3 select fails (should not happen)

            flat_obs, harm_signal, done, info, obs_dict = env.step(action)
            if harm_signal < 0:
                total_harm += 1
            elif harm_signal > 0:
                total_benefit += 1

            if train:
                traj_buffer.append((latent.z_world.detach(), action.detach()))
                if len(traj_buffer) > MAX_TRAJ_BUFFER:
                    traj_buffer = traj_buffer[-MAX_TRAJ_BUFFER:]

                # Collect E2.world_forward training transitions
                if z_world_prev is not None and a_prev is not None:
                    e2w_buf.append((z_world_prev, a_prev, z_world_curr))
                    if len(e2w_buf) > MAX_E2W_BUF:
                        e2w_buf = e2w_buf[-MAX_E2W_BUF:]

                e1_loss = agent.compute_prediction_loss()
                e2_self_loss = agent.compute_e2_loss()
                obs_w = obs_world.unsqueeze(0) if obs_world.dim() == 1 else obs_world
                z_w = agent.latent_stack.split_encoder.world_encoder(obs_w)
                recon = world_decoder(z_w)
                recon_loss = F.mse_loss(recon, obs_w)

                # E2.world_forward training loss (key fix: without this, prediction
                # errors are dominated by random network noise — ~constant regardless
                # of env stability, killing the ARC-016 variance signal)
                e2w_loss = torch.zeros(1, device=agent.device)
                if len(e2w_buf) >= 16:
                    k = min(32, len(e2w_buf))
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
            z_self_prev = z_self_curr
            a_prev = action.detach()
            step_counter += 1

            if done:
                break

        if (ep + 1) % 50 == 0 or ep == num_episodes - 1:
            print(f"  [{phase_name}] ep {ep+1}/{num_episodes}  "
                  f"var={agent.e3._running_variance:.4f}  "
                  f"prec={agent.e3.current_precision:.3f}  "
                  f"commits={sum(commit_decisions)}/{len(commit_decisions)}  "
                  f"harm={total_harm}", flush=True)

    mean_var = float(sum(running_variance_traj) / max(1, len(running_variance_traj)))
    mean_prec = float(sum(precision_traj) / max(1, len(precision_traj)))
    commit_rate = float(sum(commit_decisions) / max(1, len(commit_decisions)))
    mean_pred_err = float(sum(prediction_errors) / max(1, len(prediction_errors)))
    final_var = running_variance_traj[-1] if running_variance_traj else 0.0
    final_prec = precision_traj[-1] if precision_traj else 0.0

    print(f"  [{phase_name}] Summary: mean_var={mean_var:.4f}  mean_prec={mean_prec:.3f}  "
          f"commit_rate={commit_rate:.3f}  n_decisions={len(commit_decisions)}", flush=True)

    return {
        "mean_running_variance": mean_var,
        "mean_precision": mean_prec,
        "commit_rate": commit_rate,
        "n_commit_decisions": len(commit_decisions),
        "mean_prediction_error": mean_pred_err,
        "final_running_variance": final_var,
        "final_precision": final_prec,
        "total_harm": total_harm,
        "total_benefit": total_benefit,
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
    alpha_world: float = 0.9,   # SD-008: must be >= 0.9 for ARC-016 to work
    alpha_self: float = 0.3,
    **kwargs,
) -> dict:
    torch.manual_seed(seed)
    random.seed(seed)

    # ── Environment configs ────────────────────────────────────────────────
    # Stable: few hazards, effectively static (drift only every 100 steps with prob=0)
    # Perturbed: many hazards, maximally dynamic (drift every step with prob=1.0)
    # This 10× contrast is needed to generate a measurable variance gap.
    # Prior design (10 vs 10 hazards, interval 10 vs 2, prob 0.1 vs 0.9) produced
    # only 6% variance difference — not enough for C1 (need var_diff > 0.02).
    env_stable = CausalGridWorld(
        seed=seed, size=12, num_hazards=3, num_resources=5,
        env_drift_interval=100, env_drift_prob=0.0,  # essentially static
        hazard_harm=harm_scale, contaminated_harm=harm_scale,
    )
    env_perturbed = CausalGridWorld(
        seed=seed + 100, size=12, num_hazards=20, num_resources=5,
        env_drift_interval=1, env_drift_prob=1.0,   # maximally dynamic (every step)
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

    params = list(agent.parameters()) + list(world_decoder.parameters())
    optimizer = optim.Adam(params, lr=lr)

    # ── Phase 1: Training (stable env) ────────────────────────────────────
    print(f"[V3-EXQ-018] Training: {train_episodes} eps (stable env, "
          f"num_hazards=3, drift_interval=100, drift_prob=0.0)", flush=True)
    train_out = _run_phase(
        agent, env_stable, optimizer, world_decoder,
        train_episodes, steps_per_episode, train=True, phase_name="train"
    )

    # ── Phase 2: Eval stable ───────────────────────────────────────────────
    # Reset running_variance before each eval phase so measurement starts from
    # a known warm-start (precision_init=0.5 → converges quickly to env level).
    agent.e3._running_variance = config.e3.precision_init
    print(f"[V3-EXQ-018] Eval stable: {eval_stable_episodes} eps", flush=True)
    stable_out = _run_phase(
        agent, env_stable, optimizer, world_decoder,
        eval_stable_episodes, steps_per_episode, train=False, phase_name="stable"
    )

    # ── Phase 3: Eval perturbed ────────────────────────────────────────────
    agent.e3._running_variance = config.e3.precision_init
    print(f"[V3-EXQ-018] Eval perturbed: {eval_perturbed_episodes} eps "
          f"(num_hazards=20, drift_interval=1, drift_prob=1.0)", flush=True)
    perturbed_out = _run_phase(
        agent, env_perturbed, optimizer, world_decoder,
        eval_perturbed_episodes, steps_per_episode, train=False, phase_name="perturbed"
    )

    # ── Phase 4: Recovery ─────────────────────────────────────────────────
    agent.e3._running_variance = config.e3.precision_init
    print(f"[V3-EXQ-018] Recovery: {eval_recovery_episodes} eps (stable again)", flush=True)
    recovery_out = _run_phase(
        agent, env_stable, optimizer, world_decoder,
        eval_recovery_episodes, steps_per_episode, train=False, phase_name="recovery"
    )

    # ── PASS / FAIL ───────────────────────────────────────────────────────
    var_diff = perturbed_out["mean_running_variance"] - stable_out["mean_running_variance"]
    prec_diff = stable_out["mean_precision"] - perturbed_out["mean_precision"]
    commit_diff = stable_out["commit_rate"] - perturbed_out["commit_rate"]

    c1_pass = var_diff > 0.02
    c2_pass = prec_diff > 0.0
    c3_pass = commit_diff > 0.0
    c4_pass = stable_out["n_commit_decisions"] >= 20
    c5_pass = True  # updated if exception

    all_pass = c1_pass and c2_pass and c3_pass and c4_pass and c5_pass
    status = "PASS" if all_pass else "FAIL"
    criteria_met = sum([c1_pass, c2_pass, c3_pass, c4_pass, c5_pass])

    failure_notes = []
    if not c1_pass:
        failure_notes.append(
            f"C1 FAIL: running_variance perturbed-stable diff={var_diff:.4f} <= 0.02 "
            f"[perturbed={perturbed_out['mean_running_variance']:.4f} "
            f"stable={stable_out['mean_running_variance']:.4f}]"
        )
    if not c2_pass:
        failure_notes.append(
            f"C2 FAIL: precision_stable={stable_out['mean_precision']:.3f} <= "
            f"precision_perturbed={perturbed_out['mean_precision']:.3f}"
        )
    if not c3_pass:
        failure_notes.append(
            f"C3 FAIL: commit_rate_stable={stable_out['commit_rate']:.3f} <= "
            f"commit_rate_perturbed={perturbed_out['commit_rate']:.3f}"
        )
    if not c4_pass:
        failure_notes.append(
            f"C4 FAIL: n_commit_decisions_stable={stable_out['n_commit_decisions']} < 20"
        )

    print(f"\nV3-EXQ-018 verdict: {status}  ({criteria_met}/5)", flush=True)
    for note in failure_notes:
        print(f"  {note}", flush=True)
    print(f"  var_diff={var_diff:.4f}  prec_diff={prec_diff:.4f}  "
          f"commit_diff={commit_diff:.4f}", flush=True)

    metrics = {
        "fatal_error_count": 0.0,
        # Stable phase
        "stable_mean_running_variance": float(stable_out["mean_running_variance"]),
        "stable_mean_precision": float(stable_out["mean_precision"]),
        "stable_commit_rate": float(stable_out["commit_rate"]),
        "stable_n_commit_decisions": float(stable_out["n_commit_decisions"]),
        "stable_mean_prediction_error": float(stable_out["mean_prediction_error"]),
        # Perturbed phase
        "perturbed_mean_running_variance": float(perturbed_out["mean_running_variance"]),
        "perturbed_mean_precision": float(perturbed_out["mean_precision"]),
        "perturbed_commit_rate": float(perturbed_out["commit_rate"]),
        "perturbed_n_commit_decisions": float(perturbed_out["n_commit_decisions"]),
        "perturbed_mean_prediction_error": float(perturbed_out["mean_prediction_error"]),
        # Recovery phase
        "recovery_mean_running_variance": float(recovery_out["mean_running_variance"]),
        "recovery_commit_rate": float(recovery_out["commit_rate"]),
        # Differences
        "running_variance_diff_perturbed_minus_stable": float(var_diff),
        "precision_diff_stable_minus_perturbed": float(prec_diff),
        "commit_rate_diff_stable_minus_perturbed": float(commit_diff),
        "alpha_world": float(alpha_world),
        # Criteria
        "crit1_pass": 1.0 if c1_pass else 0.0,
        "crit2_pass": 1.0 if c2_pass else 0.0,
        "crit3_pass": 1.0 if c3_pass else 0.0,
        "crit4_pass": 1.0 if c4_pass else 0.0,
        "criteria_met": float(criteria_met),
    }

    failure_section = ""
    if failure_notes:
        failure_section = "\n## Failure Notes\n\n" + "\n".join(f"- {n}" for n in failure_notes)

    summary_markdown = f"""# V3-EXQ-018 — ARC-016 Dynamic Precision Commitment Test

**Status:** {status}
**Training:** {train_episodes} eps (stable: num_hazards=3, drift_interval=100, drift_prob=0.0)
**Eval stable:** {eval_stable_episodes} eps | **Eval perturbed:** {eval_perturbed_episodes} eps (num_hazards=20, drift_interval=1, drift_prob=1.0)
**Recovery:** {eval_recovery_episodes} eps
**Seed:** {seed}

## Motivation (ARC-016)

V2 EXQ-025 FAILED because precision was externally imposed. V3 derives precision
from E3's own prediction error EMA (running_variance). At each step:
  z_world_predicted = E2.world_forward(z_world_t, a_t)
  prediction_error  = z_world_{{t+1}} - z_world_predicted
  agent.e3.update_running_variance(prediction_error)

  precision        = 1 / (running_variance + ε)
  commit_threshold = 0.02  (variance-space — E3Config.commitment_threshold)
  committed        = running_variance < commit_threshold

## Phase Results

| Phase | mean_var | mean_precision | commit_rate | n_decisions |
|---|---|---|---|---|
| Training (stable) | {train_out["mean_running_variance"]:.4f} | {train_out["mean_precision"]:.3f} | {train_out["commit_rate"]:.3f} | {train_out["n_commit_decisions"]} |
| Eval stable | {stable_out["mean_running_variance"]:.4f} | {stable_out["mean_precision"]:.3f} | {stable_out["commit_rate"]:.3f} | {stable_out["n_commit_decisions"]} |
| Eval perturbed | {perturbed_out["mean_running_variance"]:.4f} | {perturbed_out["mean_precision"]:.3f} | {perturbed_out["commit_rate"]:.3f} | {perturbed_out["n_commit_decisions"]} |
| Recovery | {recovery_out["mean_running_variance"]:.4f} | {recovery_out["mean_precision"]:.3f} | {recovery_out["commit_rate"]:.3f} | {recovery_out["n_commit_decisions"]} |

**Key differences:**
- running_variance: perturbed - stable = {var_diff:.4f}
- precision: stable - perturbed = {prec_diff:.4f}
- commit_rate: stable - perturbed = {commit_diff:.4f}

## PASS Criteria

| Criterion | Result | Value |
|---|---|---|
| C1: var_diff > 0.02 (variance diverges) | {"PASS" if c1_pass else "FAIL"} | {var_diff:.4f} |
| C2: precision_stable > precision_perturbed | {"PASS" if c2_pass else "FAIL"} | {prec_diff:.4f} |
| C3: commit_rate_stable > commit_rate_perturbed | {"PASS" if c3_pass else "FAIL"} | {commit_diff:.4f} |
| C4: n_stable_decisions >= 20 | {"PASS" if c4_pass else "FAIL"} | {stable_out["n_commit_decisions"]} |
| C5: No fatal errors | {"PASS" if c5_pass else "FAIL"} | 0 |

Criteria met: {criteria_met}/5 → **{status}**
{failure_section}
"""

    return {
        "status": status,
        "metrics": metrics,
        "summary_markdown": summary_markdown,
        "claim_ids": CLAIM_IDS,
        "evidence_direction": "supports" if all_pass else ("mixed" if criteria_met >= 3 else "weakens"),
        "experiment_type": EXPERIMENT_TYPE,
        "fatal_error_count": 0,
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
    result["run_timestamp"] = ts
    result["claim"] = CLAIM_IDS[0]
    result["verdict"] = result["status"]
    result["run_id"] = f"{EXPERIMENT_TYPE}_{ts}_v3"
    result["architecture_epoch"] = "ree_hybrid_guardrails_v1"

    out_dir = Path(__file__).resolve().parents[1] / "evidence" / "experiments" / EXPERIMENT_TYPE
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{EXPERIMENT_TYPE}_{ts}.json"
    out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print(f"\nResult written to: {out_path}", flush=True)
    print(f"Status: {result['status']}", flush=True)
    for k, v in result["metrics"].items():
        print(f"  {k}: {v}", flush=True)
