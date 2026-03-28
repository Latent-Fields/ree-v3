#!/opt/local/bin/python3
"""
V3-EXQ-102 -- SD-011 harm_obs_a clip-vs-raw density diagnostic

Claims: SD-011

EXQ-101 FAIL/EMA_STILL_BROKEN (2026-03-27):
  Even after removing normalization-by-max bug, inversion persists:
  harm_obs_a mean HIGH (6 hazards)=0.2014 < LOW (2 hazards)=0.2751.
  Two hypotheses:

  H1 PLACEMENT CONFOUND: Single seed=42 places first 2 hazards at
     same positions in both HIGH and LOW conditions (seed is shared).
     Those 2 positions may be central/frequently-visited. A random walker
     encounters them more than the 4 peripheral hazards added in HIGH.
     Fix: use multiple seeds to average out placement effects.

  H2 CLIP SATURATION: The [0,1] clip in harm_obs_a EMA update destroys
     additive density information. With 6 hazards, cells near multiple
     hazards accumulate field values > 1.0 (sum formula) and all cap at 1.0.
     Local window mean does not scale with density after clipping.
     Fix: read raw (unclipped) hazard_field values directly.

  Also: EXQ-101 computed autocorr on harm_obs_a.mean() (scalar mean of 50-dim
  vector). As agent moves, the 5x5 window location shifts every step, so the
  MEAN of the vector changes rapidly even if each individual EMA dimension is
  smooth. autocorr_lag10 on the NORM is a better temporal persistence metric.

This experiment disentangles H1 from H2.

PASS criteria:
  C1: grid_mean_high > grid_mean_low * 2.5  -- field construction scales w/ n_hazards
  C2: local_clip_hazard_high > local_clip_hazard_low * 1.1  -- clipped local density
  C3: local_raw_hazard_high > local_raw_hazard_low * 1.1   -- raw (no-clip) local density
  C4: autocorr_lag10(harm_obs_a_norm) > 0.3               -- temporal persistence
  C5: no fatal errors
  PASS if C1 AND (C2 OR C3) AND C4 AND C5.

Diagnosis codes:
  PLACEMENT_CONFOUND:  C1 PASS, C2 PASS  -> multi-seed fixes inversion, clip OK
  CLIP_SATURATION:     C1 PASS, C2 FAIL, C3 PASS -> clip destroys density info
  EMA_ACORR_FAIL:      C4 FAIL -> norm autocorr too low (temporal design issue)
  FIELD_BUG:           C1 FAIL -> field construction not scaling with n_hazards
  PASS:                all criteria met

Supersedes: V3-EXQ-101 (EMA_STILL_BROKEN)
"""

import sys
import random
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig


EXPERIMENT_TYPE = "v3_exq_102_harm_obs_a_clip_diagnosis"
CLAIM_IDS       = ["SD-011"]

HARM_OBS_A_DIM = 50
HARM_OBS_DIM   = 51
Z_HARM_DIM     = 32


def _autocorr_lag(series: List[float], lag: int) -> float:
    """Pearson autocorrelation at a given lag."""
    if len(series) < lag + 2:
        return 0.0
    x = np.array(series[:-lag], dtype=np.float64)
    y = np.array(series[lag:],  dtype=np.float64)
    x -= x.mean(); y -= y.mean()
    denom = np.sqrt((x**2).sum() * (y**2).sum())
    if denom < 1e-12:
        return 0.0
    return float((x * y).sum() / denom)


def _action_to_onehot(idx: int, n: int, device) -> torch.Tensor:
    v = torch.zeros(1, n, device=device)
    v[0, idx] = 1.0
    return v


def _run_density_seed(
    seed: int,
    num_hazards: int,
    collect_steps: int,
    world_dim: int,
    self_dim: int,
    alpha_world: float,
    harm_scale: float,
    proximity_scale: float,
) -> Dict:
    """
    Collect per-step field diagnostics for one seed/condition (no encoder training).
    Returns aggregate means for grid-wide field, local clipped hazard dims,
    local raw (unclipped) hazard dims, resource dims, and norm time series.
    """
    torch.manual_seed(seed)
    random.seed(seed)

    env = CausalGridWorldV2(
        seed=seed, size=10, num_hazards=num_hazards, num_resources=3,
        hazard_harm=harm_scale,
        env_drift_interval=5, env_drift_prob=0.1,
        proximity_harm_scale=proximity_scale,
        proximity_benefit_scale=proximity_scale * 0.5,
        proximity_approach_threshold=0.15,
        hazard_field_decay=0.5,
        use_proxy_fields=True,
    )

    config = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=self_dim,
        world_dim=world_dim,
        alpha_world=alpha_world,
        alpha_self=0.3,
        reafference_action_dim=0,
        use_harm_stream=True,
        harm_obs_dim=HARM_OBS_DIM,
        z_harm_dim=Z_HARM_DIM,
    )
    agent = REEAgent(config)
    num_actions = env.action_dim

    grid_means:      List[float] = []
    local_clip_haz:  List[float] = []  # clipped hazard dims (harm_obs_a[:25])
    local_raw_haz:   List[float] = []  # raw unclipped from env.hazard_field
    resource_dims:   List[float] = []  # resource dims (harm_obs_a[25:])
    norm_series:     List[float] = []

    _, obs_dict = env.reset()
    agent.reset()

    steps = 0
    while steps < collect_steps:
        obs_body  = obs_dict["body_state"]
        obs_world = obs_dict["world_state"]
        with torch.no_grad():
            agent.sense(obs_body, obs_world)
            agent.clock.advance()

        harm_obs_a = obs_dict.get(
            "harm_obs_a", torch.zeros(HARM_OBS_A_DIM)).float()

        # Clipped EMA hazard dims (current approach): first 25
        local_clip_haz.append(float(harm_obs_a[:25].mean().item()))
        # Resource dims: second 25
        resource_dims.append(float(harm_obs_a[25:].mean().item()))
        # Norm for temporal persistence test
        norm_series.append(float(harm_obs_a.norm().item()))

        # Grid-wide raw field mean: validates field construction scales with n_hazards
        grid_means.append(float(env.hazard_field.mean()))

        # Raw local 5x5 window (no clip): reads env.hazard_field directly
        ax, ay = int(env.agent_x), int(env.agent_y)
        raw_vals = []
        for di in range(-2, 3):
            for dj in range(-2, 3):
                ni, nj = ax + di, ay + dj
                if 0 <= ni < env.size and 0 <= nj < env.size:
                    raw_vals.append(float(env.hazard_field[ni, nj]))
                else:
                    raw_vals.append(0.0)
        local_raw_haz.append(float(np.mean(raw_vals)))

        action = _action_to_onehot(
            random.randint(0, num_actions - 1), num_actions, agent.device)
        agent._last_action = action
        _, _, done, _, obs_dict = env.step(action)
        steps += 1

        if done:
            _, obs_dict = env.reset()
            agent.reset()

    return {
        "grid_mean":    float(np.mean(grid_means)),
        "local_clip":   float(np.mean(local_clip_haz)),
        "local_raw":    float(np.mean(local_raw_haz)),
        "resource":     float(np.mean(resource_dims)),
        "norm_series":  norm_series,
    }


def run(
    seeds: List[int] = None,
    collect_steps: int = 300,
    world_dim: int = 32,
    self_dim: int = 32,
    alpha_world: float = 0.9,
    harm_scale: float = 0.02,
    proximity_scale: float = 0.05,
    **kwargs,
) -> dict:
    if seeds is None:
        seeds = list(range(42, 47))  # 5 seeds: 42-46
    fatal_error_count = 0

    # ---- Phase 0: Multi-seed density comparison ----
    print("\n[V3-EXQ-102] Phase 0: Multi-seed density comparison"
          f" ({len(seeds)} seeds x {collect_steps} steps)...",
          flush=True)

    high_results: List[Dict] = []
    low_results:  List[Dict] = []

    for seed in seeds:
        print(f"  seed={seed} HIGH_HAZARD (n=6)...", flush=True)
        r_high = _run_density_seed(
            seed=seed, num_hazards=6, collect_steps=collect_steps,
            world_dim=world_dim, self_dim=self_dim, alpha_world=alpha_world,
            harm_scale=harm_scale, proximity_scale=proximity_scale,
        )
        high_results.append(r_high)
        print(
            f"    grid={r_high['grid_mean']:.4f}"
            f" clip_haz={r_high['local_clip']:.4f}"
            f" raw_haz={r_high['local_raw']:.4f}"
            f" res={r_high['resource']:.4f}",
            flush=True,
        )

        print(f"  seed={seed} LOW_HAZARD (n=2)...", flush=True)
        r_low = _run_density_seed(
            seed=seed, num_hazards=2, collect_steps=collect_steps,
            world_dim=world_dim, self_dim=self_dim, alpha_world=alpha_world,
            harm_scale=harm_scale, proximity_scale=proximity_scale,
        )
        low_results.append(r_low)
        print(
            f"    grid={r_low['grid_mean']:.4f}"
            f" clip_haz={r_low['local_clip']:.4f}"
            f" raw_haz={r_low['local_raw']:.4f}"
            f" res={r_low['resource']:.4f}",
            flush=True,
        )

    grid_high  = float(np.mean([r["grid_mean"]  for r in high_results]))
    grid_low   = float(np.mean([r["grid_mean"]  for r in low_results]))
    clip_high  = float(np.mean([r["local_clip"] for r in high_results]))
    clip_low   = float(np.mean([r["local_clip"] for r in low_results]))
    raw_high   = float(np.mean([r["local_raw"]  for r in high_results]))
    raw_low    = float(np.mean([r["local_raw"]  for r in low_results]))
    res_high   = float(np.mean([r["resource"]   for r in high_results]))
    res_low    = float(np.mean([r["resource"]   for r in low_results]))

    c1 = grid_high > grid_low * 2.5
    c2 = clip_high > clip_low * 1.1
    c3 = raw_high  > raw_low  * 1.1

    print(
        f"\n  Aggregate ({len(seeds)} seeds):",
        flush=True,
    )
    print(
        f"  Grid-wide: HIGH={grid_high:.4f} LOW={grid_low:.4f}"
        f"  ratio={grid_high/max(grid_low,1e-9):.2f}x  (threshold 2.5x)",
        flush=True,
    )
    print(
        f"  Clip haz:  HIGH={clip_high:.4f} LOW={clip_low:.4f}"
        f"  ratio={clip_high/max(clip_low,1e-9):.2f}x  (threshold 1.1x)",
        flush=True,
    )
    print(
        f"  Raw haz:   HIGH={raw_high:.4f} LOW={raw_low:.4f}"
        f"  ratio={raw_high/max(raw_low,1e-9):.2f}x  (threshold 1.1x)",
        flush=True,
    )
    print(
        f"  Resource:  HIGH={res_high:.4f} LOW={res_low:.4f}  (reference)",
        flush=True,
    )
    print(
        f"  C1 (grid scales):    {'PASS' if c1 else 'FAIL'}",
        flush=True,
    )
    print(
        f"  C2 (clipped local):  {'PASS' if c2 else 'FAIL'}",
        flush=True,
    )
    print(
        f"  C3 (raw local):      {'PASS' if c3 else 'FAIL'}",
        flush=True,
    )

    # ---- Phase 1: Temporal persistence (norm autocorrelation) ----
    print("\n[V3-EXQ-102] Phase 1: Temporal persistence"
          " (autocorr on harm_obs_a norm)...",
          flush=True)

    # Use norm_series from HIGH condition seed 42 (index 0)
    norm_series_ref = high_results[0]["norm_series"]
    ac_norm = _autocorr_lag(norm_series_ref, lag=10)
    c4 = ac_norm > 0.3

    print(
        f"  harm_obs_a norm autocorr_lag10 = {ac_norm:.3f}"
        f"  (threshold > 0.3; theory ~0.60 for tau=20 EMA if stable env)",
        flush=True,
    )
    print(f"  C4 (temporal persistence): {'PASS' if c4 else 'FAIL'}", flush=True)

    c5          = fatal_error_count == 0
    all_pass    = c1 and (c2 or c3) and c4 and c5
    n_met       = sum([c1, c2, c3, c4, c5])
    status      = "PASS" if all_pass else "FAIL"

    # Diagnosis tree
    if not c1:
        diagnosis = "FIELD_BUG"
    elif c2:
        diagnosis = "PLACEMENT_CONFOUND"
    elif c3:
        diagnosis = "CLIP_SATURATION"
    elif not c4:
        diagnosis = "EMA_ACORR_FAIL"
    else:
        diagnosis = "PASS"

    print(f"\n[V3-EXQ-102] {status} ({n_met}/5)  Diagnosis: {diagnosis}", flush=True)

    failure_notes = []
    if not c1:
        failure_notes.append(
            f"C1 FAIL: grid-wide field not scaling with n_hazards"
            f" (HIGH={grid_high:.4f} LOW={grid_low:.4f},"
            f" ratio={grid_high/max(grid_low,1e-9):.2f}x < 2.5x threshold)")
    if not c2:
        failure_notes.append(
            f"C2 FAIL: clipped local hazard dims not density-responsive"
            f" (HIGH={clip_high:.4f} LOW={clip_low:.4f},"
            f" ratio={clip_high/max(clip_low,1e-9):.2f}x < 1.1x)")
    if not c3:
        failure_notes.append(
            f"C3 FAIL: raw local hazard dims not density-responsive"
            f" (HIGH={raw_high:.4f} LOW={raw_low:.4f},"
            f" ratio={raw_high/max(raw_low,1e-9):.2f}x < 1.1x)")
    if not c4:
        failure_notes.append(
            f"C4 FAIL: harm_obs_a norm autocorr={ac_norm:.3f} < 0.3"
            " -- EMA not providing temporal persistence on agent trajectory")
    for note in failure_notes:
        print(f"  {note}", flush=True)

    # Per-seed table
    per_seed = []
    for i, seed in enumerate(seeds):
        per_seed.append({
            "seed":      seed,
            "grid_high": float(high_results[i]["grid_mean"]),
            "grid_low":  float(low_results[i]["grid_mean"]),
            "clip_high": float(high_results[i]["local_clip"]),
            "clip_low":  float(low_results[i]["local_clip"]),
            "raw_high":  float(high_results[i]["local_raw"]),
            "raw_low":   float(low_results[i]["local_raw"]),
        })

    metrics = {
        "grid_mean_high":       grid_high,
        "grid_mean_low":        grid_low,
        "local_clip_mean_high": clip_high,
        "local_clip_mean_low":  clip_low,
        "local_raw_mean_high":  raw_high,
        "local_raw_mean_low":   raw_low,
        "resource_mean_high":   res_high,
        "resource_mean_low":    res_low,
        "autocorr_norm_lag10":  ac_norm,
        "crit1_grid_scaling":   1.0 if c1 else 0.0,
        "crit2_clip_density":   1.0 if c2 else 0.0,
        "crit3_raw_density":    1.0 if c3 else 0.0,
        "crit4_autocorr":       1.0 if c4 else 0.0,
        "crit5_no_errors":      1.0 if c5 else 0.0,
        "criteria_met":         float(n_met),
        "n_seeds":              float(len(seeds)),
    }

    failure_section = ""
    if failure_notes:
        failure_section = "\n## Failure Notes\n\n" + "\n".join(
            f"- {n}" for n in failure_notes)

    per_seed_rows = "".join(
        f"| {p['seed']} | {p['grid_high']:.4f} | {p['grid_low']:.4f}"
        f" | {p['clip_high']:.4f} | {p['clip_low']:.4f}"
        f" | {p['raw_high']:.4f} | {p['raw_low']:.4f} |\n"
        for p in per_seed
    )

    summary_markdown = f"""# V3-EXQ-102 -- SD-011 harm_obs_a Clip vs Raw Density Diagnostic

**Status:** {status}
**Claims:** SD-011
**Supersedes:** V3-EXQ-101 (EMA_STILL_BROKEN after normfix)
**Seeds:** {seeds}  **Steps/seed/condition:** {collect_steps}

## Diagnosis: {diagnosis}

Hypotheses tested:
- H1 PLACEMENT_CONFOUND: seed=42 first-2-hazards placement biases single-seed test
- H2 CLIP_SATURATION: [0,1] clip destroys additive density info (cells near 2+ hazards cap)

## Phase 0: Multi-seed Density Response (N={len(seeds)} seeds x {collect_steps} steps)

| Metric | HIGH (6 hazards) | LOW (2 hazards) | Criterion |
|--------|-----------------|-----------------|-----------|
| Grid-wide field mean | {grid_high:.4f} | {grid_low:.4f} | high > low * 2.5 (C1) |
| Local CLIPPED hazard dims | {clip_high:.4f} | {clip_low:.4f} | high > low * 1.1 (C2) |
| Local RAW hazard dims | {raw_high:.4f} | {raw_low:.4f} | high > low * 1.1 (C3) |
| Resource dims | {res_high:.4f} | {res_low:.4f} | reference only |

| Criterion | Result | Interpretation |
|-----------|--------|----------------|
| C1: field scales with n_hazards | {"PASS" if c1 else "FAIL"} | field construction |
| C2: clipped local density | {"PASS" if c2 else "FAIL"} | clip approach |
| C3: raw local density | {"PASS" if c3 else "FAIL"} | raw (no-clip) approach |

## Phase 1: Temporal Persistence

| Metric | Value | Criterion |
|--------|-------|-----------|
| harm_obs_a norm autocorr_lag10 | {ac_norm:.3f} | > 0.3 (C4) |

| Criterion | Result |
|-----------|--------|
| C4: temporal persistence | {"PASS" if c4 else "FAIL"} |
| C5: no fatal errors | {"PASS" if c5 else "FAIL"} |

Criteria met: {n_met}/5 -> **{status}**
{failure_section}

## Per-Seed Results

| Seed | Grid HIGH | Grid LOW | Clip HIGH | Clip LOW | Raw HIGH | Raw LOW |
|------|-----------|----------|-----------|----------|----------|---------|
{per_seed_rows}
"""

    return {
        "status":             status,
        "metrics":            metrics,
        "per_seed":           per_seed,
        "summary_markdown":   summary_markdown,
        "claim_ids":          CLAIM_IDS,
        "evidence_direction": "supports" if all_pass else (
            "mixed" if n_met >= 3 else "weakens"),
        "experiment_type":    EXPERIMENT_TYPE,
        "fatal_error_count":  fatal_error_count,
        "diagnosis":          diagnosis,
        "supersedes":         "v3_exq_101_harm_obs_a_normfix",
    }


if __name__ == "__main__":
    import argparse
    import json
    from datetime import datetime, timezone

    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(42, 47)))
    parser.add_argument("--steps", type=int, default=300)
    args = parser.parse_args()

    result = run(seeds=args.seeds, collect_steps=args.steps)

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
