#!/opt/local/bin/python3
"""V3-EXQ-529: MECH-098 Reafference Cancellation Selectivity

Tests whether SD-007 ReafferencePredictor makes z_world more selective for
external environmental events (SD-047 multi-source dynamics) vs agent-caused
z_world changes.

Three-arm design:
  ARM_0: no SD-047, no reafference (baseline -- no multi-source, reafference_action_dim=0)
  ARM_1: SD-047 active, no reafference (reafference_action_dim=0)
  ARM_2: SD-047 active, reafference ON (reafference_action_dim=5)

Protocol (P0 = training, P1 = structured alternating eval):
  P0: Random actions to train agent (and reafference predictor for ARM_2).
  P1: Even steps = noop (action 4); odd steps = random move (actions 0-3).
      Track z_world delta = (new_latent.z_world - prev_z_world).norm() each step.

Metrics per arm:
  noop_delta   = mean delta on even (noop) steps (pure env events)
  move_delta   = mean delta on odd (move) steps (agent-caused + env events)
  selectivity  = noop_delta / (move_delta + 1e-8)

Acceptance criteria:
  C1: ARM_2 move_delta < ARM_1 move_delta * 0.9  (reafference reduces agent-caused delta)
  C2: ARM_2 noop_delta >= ARM_1 noop_delta * 0.7  (reafference preserves env-event signal)
  C3: ARM_2 selectivity > ARM_1 selectivity
  Overall PASS = C1 AND C2 AND C3

claim_ids: ["MECH-098"]
architecture_epoch: "ree_hybrid_guardrails_v1"
run_id: ends _v3
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.optim

from ree_core.utils.config import REEConfig
from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from experiments.pack_writer import write_flat_manifest  # noqa: E402
from pathlib import Path  # noqa: E402

# ---------------------------------------------------------------------------
# Identifiers
# ---------------------------------------------------------------------------

QUEUE_ID           = "V3-EXQ-529"
EXPERIMENT_TYPE    = "v3_exq_529_mech098_reafference_selectivity"
CLAIM_IDS          = ["MECH-098"]
EXPERIMENT_PURPOSE = "evidence"

# ---------------------------------------------------------------------------
# SD-047 environment kwargs (multi-source dynamics)
# ---------------------------------------------------------------------------

SD047_KWARGS = dict(
    multi_source_dynamics_enabled=True,
    multi_source_intensity_scale=1.0,
    weather_field_enabled=True,
    transient_events_enabled=True,
    background_drift_enabled=True,
    n_drift_sources=1,
)


# ---------------------------------------------------------------------------
# Helper: build obs tensors from obs_dict
# ---------------------------------------------------------------------------

def _to_tensor(arr, device) -> torch.Tensor:
    if torch.is_tensor(arr):
        t = arr.to(device).float()
    else:
        t = torch.tensor(arr, dtype=torch.float32, device=device)
    if t.dim() == 1:
        t = t.unsqueeze(0)
    return t


def _onehot(idx: int, n: int, device) -> torch.Tensor:
    v = torch.zeros(1, n, device=device)
    v[0, idx] = 1.0
    return v


# ---------------------------------------------------------------------------
# Training phase (P0) -- calibrates reafference predictor for ARM_2
# ---------------------------------------------------------------------------

def _run_training_phase(
    agent: REEAgent,
    env: CausalGridWorldV2,
    n_episodes: int,
    rng: np.random.Generator,
    has_reafference: bool,
) -> None:
    """Train agent with random actions for n_episodes episodes."""
    device = agent.device
    optimizer = torch.optim.Adam(agent.parameters(), lr=3e-4)

    for ep in range(n_episodes):
        _, obs_dict = env.reset()
        agent.reset()

        for _step in range(200):
            obs_body  = _to_tensor(obs_dict["body_state"], device)
            obs_world = _to_tensor(obs_dict["world_state"], device)

            latent = agent.sense(obs_body, obs_world)
            agent.clock.advance()

            # Random action 0-4
            action_idx = int(rng.integers(0, env.action_dim))
            action = _onehot(action_idx, env.action_dim, device)
            agent._last_action = action

            _, _harm, done, _info, obs_dict = env.step(action_idx)

            loss = agent.compute_prediction_loss()
            if loss.requires_grad:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                optimizer.step()

            if done:
                break

        if (ep + 1) % 5 == 0 or ep == n_episodes - 1:
            print(f"  [train] ep {ep+1}/{n_episodes} reafference={has_reafference}", flush=True)


# ---------------------------------------------------------------------------
# Eval phase (P1) -- structured alternating noop / move protocol
# ---------------------------------------------------------------------------

def _run_eval_phase(
    agent: REEAgent,
    env: CausalGridWorldV2,
    n_episodes: int,
    rng: np.random.Generator,
) -> dict:
    """Alternating noop/move evaluation. Returns per-seed metric dict."""
    device = agent.device

    noop_deltas = []
    move_deltas = []

    for _ep in range(n_episodes):
        _, obs_dict = env.reset()
        agent.reset()

        prev_z_world = None

        for step in range(200):
            obs_body  = _to_tensor(obs_dict["body_state"], device)
            obs_world = _to_tensor(obs_dict["world_state"], device)

            # Store prev_z_world BEFORE calling sense
            if prev_z_world is not None:
                prev_zw = prev_z_world
            else:
                prev_zw = None

            with torch.no_grad():
                latent = agent.sense(obs_body, obs_world)
            agent.clock.advance()

            # Record delta if we have a previous latent
            if prev_zw is not None:
                delta = (latent.z_world.detach() - prev_zw).norm().item()
                if step % 2 == 0:
                    # Even step = just did a noop action on the previous step
                    noop_deltas.append(delta)
                else:
                    # Odd step = just did a move action on the previous step
                    move_deltas.append(delta)

            # Choose action: even step -> noop (4), odd step -> random move (0-3)
            if step % 2 == 0:
                action_idx = 4  # noop
            else:
                action_idx = int(rng.integers(0, 4))  # move: 0-3

            action = _onehot(action_idx, env.action_dim, device)
            agent._last_action = action

            _, _harm, done, _info, obs_dict = env.step(action_idx)

            prev_z_world = latent.z_world.detach().clone()

            if done:
                break

    noop_delta = float(np.mean(noop_deltas)) if noop_deltas else 0.0
    move_delta = float(np.mean(move_deltas)) if move_deltas else 0.0
    selectivity = noop_delta / (move_delta + 1e-8)

    return {
        "noop_delta": noop_delta,
        "move_delta": move_delta,
        "selectivity": selectivity,
        "n_noop_steps": len(noop_deltas),
        "n_move_steps": len(move_deltas),
    }


# ---------------------------------------------------------------------------
# Single arm runner
# ---------------------------------------------------------------------------

def _run_arm(
    arm_name: str,
    env_kwargs: dict,
    reafference_action_dim: int,
    n_train_eps: int,
    n_eval_eps: int,
    n_seeds: int,
    grid_size: int,
    dry_run: bool,
) -> dict:
    """Run one arm across all seeds. Returns aggregated metrics."""
    seed_results = []

    for seed in range(n_seeds):
        print(f"Seed {seed} Condition {arm_name}", flush=True)
        rng = np.random.default_rng(seed + 5290)
        torch.manual_seed(seed + 5290)

        env = CausalGridWorldV2(
            size=grid_size,
            num_hazards=4,
            num_resources=4,
            seed=seed,
            use_proxy_fields=True,
            **env_kwargs,
        )

        config = REEConfig.from_dims(
            body_obs_dim=env.body_obs_dim,
            world_obs_dim=env.world_obs_dim,
            action_dim=env.action_dim,
            reafference_action_dim=reafference_action_dim,
        )
        agent = REEAgent(config)

        # P0: Training
        has_reafference = reafference_action_dim > 0
        _run_training_phase(agent, env, n_train_eps, rng, has_reafference)

        # P1: Structured eval
        agent.eval()
        metrics = _run_eval_phase(agent, env, n_eval_eps, rng)
        metrics["seed"] = seed
        seed_results.append(metrics)

        print(f"verdict: PASS", flush=True)

        if dry_run:
            print(
                f"  {arm_name} seed={seed}: "
                f"noop_delta={metrics['noop_delta']:.4f} "
                f"move_delta={metrics['move_delta']:.4f} "
                f"selectivity={metrics['selectivity']:.4f} "
                f"(n_noop={metrics['n_noop_steps']} n_move={metrics['n_move_steps']})"
            )

    # Aggregate across seeds
    agg = {}
    for key in ["noop_delta", "move_delta", "selectivity"]:
        vals = [r[key] for r in seed_results]
        agg[f"mean_{key}"] = float(np.mean(vals))
        agg[f"std_{key}"]  = float(np.std(vals))
    agg["seed_results"] = seed_results
    return agg


# ---------------------------------------------------------------------------
# Main experiment function
# ---------------------------------------------------------------------------

def run_experiment(dry_run: bool = False) -> dict:
    n_train_eps = 3  if dry_run else 30
    n_eval_eps  = 2  if dry_run else 20
    n_seeds     = 1  if dry_run else 3
    grid_size   = 12

    # ARM_0: no SD-047, no reafference
    arm0_env_kwargs = {}
    # ARM_1: SD-047 active, no reafference
    arm1_env_kwargs = SD047_KWARGS
    # ARM_2: SD-047 active, reafference ON
    arm2_env_kwargs = SD047_KWARGS

    print(f"V3-EXQ-529 MECH-098 Reafference Selectivity")
    print(f"  dry_run={dry_run}  n_train_eps={n_train_eps}  n_eval_eps={n_eval_eps}  n_seeds={n_seeds}")
    print()

    print("Running ARM_0 (no SD-047, no reafference)...")
    arm0 = _run_arm(
        "ARM_0_baseline",
        arm0_env_kwargs,
        reafference_action_dim=0,
        n_train_eps=n_train_eps,
        n_eval_eps=n_eval_eps,
        n_seeds=n_seeds,
        grid_size=grid_size,
        dry_run=dry_run,
    )

    print("Running ARM_1 (SD-047 active, no reafference)...")
    arm1 = _run_arm(
        "ARM_1_sd047_no_reaf",
        arm1_env_kwargs,
        reafference_action_dim=0,
        n_train_eps=n_train_eps,
        n_eval_eps=n_eval_eps,
        n_seeds=n_seeds,
        grid_size=grid_size,
        dry_run=dry_run,
    )

    print("Running ARM_2 (SD-047 active, reafference ON)...")
    arm2 = _run_arm(
        "ARM_2_sd047_reaf_on",
        arm2_env_kwargs,
        reafference_action_dim=5,
        n_train_eps=n_train_eps,
        n_eval_eps=n_eval_eps,
        n_seeds=n_seeds,
        grid_size=grid_size,
        dry_run=dry_run,
    )

    # Acceptance criteria
    arm1_move = arm1["mean_move_delta"]
    arm2_move = arm2["mean_move_delta"]
    arm1_noop = arm1["mean_noop_delta"]
    arm2_noop = arm2["mean_noop_delta"]
    arm1_sel  = arm1["mean_selectivity"]
    arm2_sel  = arm2["mean_selectivity"]

    c1 = arm2_move < arm1_move * 0.9
    c2 = arm2_noop >= arm1_noop * 0.7
    c3 = arm2_sel  > arm1_sel

    overall_pass = c1 and c2 and c3

    print()
    print(f"ARM_0 noop_delta={arm0['mean_noop_delta']:.4f}  move_delta={arm0['mean_move_delta']:.4f}  selectivity={arm0['mean_selectivity']:.4f}")
    print(f"ARM_1 noop_delta={arm1_noop:.4f}  move_delta={arm1_move:.4f}  selectivity={arm1_sel:.4f}")
    print(f"ARM_2 noop_delta={arm2_noop:.4f}  move_delta={arm2_move:.4f}  selectivity={arm2_sel:.4f}")
    print(f"C1 (ARM_2 move < ARM_1 move * 0.9): {c1}  ({arm2_move:.4f} < {arm1_move * 0.9:.4f})")
    print(f"C2 (ARM_2 noop >= ARM_1 noop * 0.7): {c2}  ({arm2_noop:.4f} >= {arm1_noop * 0.7:.4f})")
    print(f"C3 (ARM_2 selectivity > ARM_1 selectivity): {c3}  ({arm2_sel:.4f} > {arm1_sel:.4f})")
    print(f"Overall: {'PASS' if overall_pass else 'FAIL'}")

    return {
        "arm0": arm0,
        "arm1": arm1,
        "arm2": arm2,
        "criteria": {"C1": c1, "C2": c2, "C3": c3},
        "criteria_thresholds": {
            "C1_arm2_move_vs_arm1_move_ratio_threshold": 0.9,
            "C2_arm2_noop_vs_arm1_noop_ratio_threshold": 0.7,
        },
        "n_train_eps": n_train_eps,
        "n_eval_eps": n_eval_eps,
        "n_seeds": n_seeds,
        "grid_size": grid_size,
        "overall_pass": overall_pass,
    }


# ---------------------------------------------------------------------------
# Write result manifest
# ---------------------------------------------------------------------------

def write_result(result: dict, run_id: str) -> None:
    output_dir = os.path.join(
        os.path.dirname(__file__), "..", "..",
        "REE_assembly", "evidence", "experiments"
    )
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{run_id}.json")

    manifest = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": QUEUE_ID,
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "evidence_direction": "supports" if result["overall_pass"] else "weakens",
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "outcome": "PASS" if result["overall_pass"] else "FAIL",
        "metrics": result,
    }

    out_path = write_flat_manifest(
        manifest,
        output_dir,
        dry_run=False,
        config=manifest.get("config"),
        seeds=None,
        script_path=Path(__file__),
    )
    print(f"Result written to {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    t0 = time.time()
    result = run_experiment(dry_run=args.dry_run)
    elapsed = time.time() - t0
    print(f"\nElapsed: {elapsed:.1f}s")

    if not args.dry_run:
        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        run_id = f"{EXPERIMENT_TYPE}_{ts}_v3"
        write_result(result, run_id)
