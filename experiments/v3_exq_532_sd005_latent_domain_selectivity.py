#!/opt/local/bin/python3
"""V3-EXQ-532: SD-005 Latent Domain Selectivity

Tests that the SD-005 z_self / z_world split actually encodes its respective
domain. Since the split is always-on in V3, this is a structural probe: after
calling sense(), z_self should covary more with body_obs changes than z_world
does, and z_world should covary more with world_obs changes than z_self does.

Measurement approach:
  At each step, compute L2 norms of:
    delta_z_self, delta_z_world  (change in latent)
    delta_body, delta_world      (change in raw observation)
  Then aggregate correlation (via np.corrcoef) of the norm series across steps.

Metrics (averaged across seeds):
  corr_z_self_body:     |corr(||delta_z_self||, ||delta_body||)|
  corr_z_world_body:    |corr(||delta_z_world||, ||delta_body||)|
  corr_z_world_world_obs: |corr(||delta_z_world||, ||delta_world||)|
  corr_z_self_world_obs:  |corr(||delta_z_self||, ||delta_world||)|
  self_body_advantage:  corr_z_self_body - corr_z_world_body
  world_obs_advantage:  corr_z_world_world_obs - corr_z_self_world_obs

Acceptance criteria:
  C1: corr_z_self_body > corr_z_world_body
  C2: corr_z_world_world_obs > corr_z_self_world_obs
  C3: self_body_advantage > 0.02

Overall PASS = C1 AND C2 AND C3

claim_ids: ["SD-005"]
experiment_purpose: evidence
"""

import os
import sys
import json
import time
import numpy as np
import torch
from datetime import datetime
import argparse
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ree_core.utils.config import REEConfig
from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from experiment_protocol import emit_outcome
from experiments.pack_writer import write_flat_manifest  # noqa: E402

QUEUE_ID = "V3-EXQ-532"
EXPERIMENT_TYPE = "v3_exq_532_sd005_latent_domain_selectivity"
CLAIM_IDS = ["SD-005"]


def _safe_corrcoef(a, b):
    """Compute |Pearson r| between two 1-D arrays. Returns 0.0 if degenerate."""
    if len(a) < 2:
        return 0.0
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    mat = np.corrcoef(a, b)
    return float(np.abs(mat[0, 1]))


def run_seed(seed, n_eps, steps_per_ep, env_kwargs, dry_run=False):
    """Run one seed. Returns dict of per-seed correlation scalars."""
    rng = np.random.default_rng(seed)
    env = CausalGridWorldV2(**env_kwargs, seed=seed)

    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
    )
    agent = REEAgent(cfg)

    # Norm series across all valid steps (skip step 0 per episode and zero-norm steps)
    z_self_norms = []
    z_world_norms = []
    body_norms = []
    world_norms = []

    for _ep in range(n_eps):
        _, obs_dict = env.reset()
        agent.reset()

        prev_body_obs = obs_dict["body_state"].numpy().flatten().astype(np.float32)
        prev_world_obs = obs_dict["world_state"].numpy().flatten().astype(np.float32)

        # Sense step 0 to get initial latent state
        obs_body_t = obs_dict["body_state"].unsqueeze(0).float()
        obs_world_t = obs_dict["world_state"].unsqueeze(0).float()
        with torch.no_grad():
            prev_latent = agent.sense(obs_body_t, obs_world_t)
        prev_z_self = prev_latent.z_self.detach().numpy().flatten()
        prev_z_world = prev_latent.z_world.detach().numpy().flatten()

        agent.clock.advance()

        for _step in range(1, steps_per_ep):
            action = int(rng.integers(0, env.action_dim))
            _, _harm, done, _info, obs_dict = env.step(action)

            body_obs = obs_dict["body_state"].numpy().flatten().astype(np.float32)
            world_obs = obs_dict["world_state"].numpy().flatten().astype(np.float32)

            obs_body_t = obs_dict["body_state"].unsqueeze(0).float()
            obs_world_t = obs_dict["world_state"].unsqueeze(0).float()
            with torch.no_grad():
                new_latent = agent.sense(obs_body_t, obs_world_t)

            z_self_np = new_latent.z_self.detach().numpy().flatten()
            z_world_np = new_latent.z_world.detach().numpy().flatten()

            delta_z_self = z_self_np - prev_z_self
            delta_z_world = z_world_np - prev_z_world
            delta_body = body_obs - prev_body_obs
            delta_world = world_obs - prev_world_obs

            n_zs = float(np.linalg.norm(delta_z_self))
            n_zw = float(np.linalg.norm(delta_z_world))
            n_b = float(np.linalg.norm(delta_body))
            n_w = float(np.linalg.norm(delta_world))

            # Skip steps where either latent norm is zero (no latent change)
            if n_zs > 0.0 or n_zw > 0.0:
                # Only include steps where at least one observation actually changed
                if n_b > 0.0 or n_w > 0.0:
                    z_self_norms.append(n_zs)
                    z_world_norms.append(n_zw)
                    body_norms.append(n_b)
                    world_norms.append(n_w)

            prev_z_self = z_self_np
            prev_z_world = z_world_np
            prev_body_obs = body_obs
            prev_world_obs = world_obs

            agent.clock.advance()

            if done:
                break

    z_self_arr = np.array(z_self_norms, dtype=np.float32)
    z_world_arr = np.array(z_world_norms, dtype=np.float32)
    body_arr = np.array(body_norms, dtype=np.float32)
    world_arr = np.array(world_norms, dtype=np.float32)

    n_samples = len(z_self_arr)

    if n_samples < 2:
        return {
            "n_samples": n_samples,
            "corr_z_self_body": 0.0,
            "corr_z_world_body": 0.0,
            "corr_z_world_world_obs": 0.0,
            "corr_z_self_world_obs": 0.0,
            "self_body_advantage": 0.0,
            "world_obs_advantage": 0.0,
        }

    corr_z_self_body = _safe_corrcoef(z_self_arr, body_arr)
    corr_z_world_body = _safe_corrcoef(z_world_arr, body_arr)
    corr_z_world_world_obs = _safe_corrcoef(z_world_arr, world_arr)
    corr_z_self_world_obs = _safe_corrcoef(z_self_arr, world_arr)

    if dry_run:
        print(
            f"  seed={seed} n_samples={n_samples} "
            f"corr_zs_body={corr_z_self_body:.4f} "
            f"corr_zw_body={corr_z_world_body:.4f} "
            f"corr_zw_world={corr_z_world_world_obs:.4f} "
            f"corr_zs_world={corr_z_self_world_obs:.4f}"
        )

    return {
        "n_samples": n_samples,
        "corr_z_self_body": corr_z_self_body,
        "corr_z_world_body": corr_z_world_body,
        "corr_z_world_world_obs": corr_z_world_world_obs,
        "corr_z_self_world_obs": corr_z_self_world_obs,
        "self_body_advantage": corr_z_self_body - corr_z_world_body,
        "world_obs_advantage": corr_z_world_world_obs - corr_z_self_world_obs,
    }


def run_experiment(dry_run=False):
    n_eps = 5 if dry_run else 30
    steps_per_ep = 50 if dry_run else 200
    n_seeds = 1 if dry_run else 3

    env_kwargs = dict(
        size=12,
        num_hazards=3,
        num_resources=5,
        use_proxy_fields=True,
    )

    rng_seeds = np.random.default_rng(532)
    seeds = [int(rng_seeds.integers(0, 100000)) for _ in range(n_seeds)]

    seed_results = []
    for i, seed in enumerate(seeds):
        print(f"Running seed {i + 1}/{n_seeds} (seed={seed})...")
        sr = run_seed(
            seed=seed,
            n_eps=n_eps,
            steps_per_ep=steps_per_ep,
            env_kwargs=env_kwargs,
            dry_run=dry_run,
        )
        seed_results.append(sr)

    def _mean(key):
        return float(np.mean([r[key] for r in seed_results]))

    corr_z_self_body = _mean("corr_z_self_body")
    corr_z_world_body = _mean("corr_z_world_body")
    corr_z_world_world_obs = _mean("corr_z_world_world_obs")
    corr_z_self_world_obs = _mean("corr_z_self_world_obs")
    self_body_advantage = _mean("self_body_advantage")
    world_obs_advantage = _mean("world_obs_advantage")
    mean_n_samples = _mean("n_samples")

    c1 = corr_z_self_body > corr_z_world_body
    c2 = corr_z_world_world_obs > corr_z_self_world_obs
    c3 = self_body_advantage > 0.02

    overall_pass = c1 and c2 and c3
    outcome = "PASS" if overall_pass else "FAIL"
    evidence_direction = "supports" if overall_pass else "weakens"

    return {
        "overall_pass": overall_pass,
        "outcome": outcome,
        "evidence_direction": evidence_direction,
        "criteria": {"C1": c1, "C2": c2, "C3": c3},
        "metrics": {
            "corr_z_self_body": corr_z_self_body,
            "corr_z_world_body": corr_z_world_body,
            "corr_z_world_world_obs": corr_z_world_world_obs,
            "corr_z_self_world_obs": corr_z_self_world_obs,
            "self_body_advantage": self_body_advantage,
            "world_obs_advantage": world_obs_advantage,
            "mean_n_samples": mean_n_samples,
        },
        "per_seed": seed_results,
        "n_eps": n_eps,
        "steps_per_ep": steps_per_ep,
        "n_seeds": n_seeds,
        "seeds": seeds,
        "env_kwargs": {k: v for k, v in env_kwargs.items()},
    }


def write_result(result, run_id):
    script_dir = Path(__file__).resolve().parents[1]
    out_dir = (
        script_dir.parent / "REE_assembly" / "evidence"
        / "experiments" / EXPERIMENT_TYPE
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"{run_id}.json"
    manifest = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": QUEUE_ID,
        "claim_ids": CLAIM_IDS,
        "evidence_direction": result["evidence_direction"],
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "outcome": result["outcome"],
        "metrics": result["metrics"],
        "criteria": result["criteria"],
        "per_seed": result["per_seed"],
        "config": {
            "n_eps": result["n_eps"],
            "steps_per_ep": result["steps_per_ep"],
            "n_seeds": result["n_seeds"],
            "seeds": result["seeds"],
            "env_kwargs": result["env_kwargs"],
        },
    }
    out_path = write_flat_manifest(
        manifest,
        out_dir,
        dry_run=False,
        config=manifest.get("config"),
        seeds=None,
        script_path=Path(__file__),
    )
    print(f"Result written to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    t0 = time.time()
    print(f"{QUEUE_ID} SD-005 Latent Domain Selectivity")
    print(f"dry_run={args.dry_run}")

    result = run_experiment(dry_run=args.dry_run)

    elapsed = time.time() - t0
    print(f"\nElapsed: {elapsed:.1f}s")
    print(f"Outcome: {result['outcome']}")
    m = result["metrics"]
    print(
        f"  corr_z_self_body={m['corr_z_self_body']:.4f}  "
        f"corr_z_world_body={m['corr_z_world_body']:.4f}  "
        f"self_body_adv={m['self_body_advantage']:.4f}  "
        f"C1={result['criteria']['C1']}"
    )
    print(
        f"  corr_z_world_world={m['corr_z_world_world_obs']:.4f}  "
        f"corr_z_self_world={m['corr_z_self_world_obs']:.4f}  "
        f"world_obs_adv={m['world_obs_advantage']:.4f}  "
        f"C2={result['criteria']['C2']}"
    )
    print(f"  C3 (adv>0.02): {result['criteria']['C3']}")
    print(f"  mean_n_samples={m['mean_n_samples']:.0f}")

    if not args.dry_run:
        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        run_id = f"{EXPERIMENT_TYPE}_{ts}_v3"
        write_result(result, run_id)

    # --- runner-conformance sentinel (added by retrofit_experiments.py) ---
    _outcome_raw = str(result.get("outcome", "FAIL")).upper()
    emit_outcome(
        outcome=_outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL",
        manifest_path=None,
    )
