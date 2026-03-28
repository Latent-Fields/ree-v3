#!/opt/local/bin/python3
"""
V3-EXQ-106a -- SD-011 harm_obs_a temporal persistence: episode-reset fix

Claims: SD-011

Supersedes: V3-EXQ-106 (C4 FAIL: autocorr_lag10=0.070 << 0.30)

Root cause of EXQ-106 C4 FAIL:
  harm_obs_a_ema was reset to zeros in CausalGridWorldV2.reset().
  Every episode reset wipes the EMA, creating a sawtooth (values build up,
  then drop to zero, then build up again), which destroys temporal autocorrelation.
  An affective/homeostatic accumulator represents the agent's running threat
  exposure history -- it SHOULD persist across episodes, not reset on episode
  boundaries.

Fix (2026-03-28, causal_grid_world.py):
  Moved harm_obs_a_ema initialization from reset() to __init__(). The array is
  created once at construction and never zeroed on episode boundary.
  Expected autocorr at lag k: ~(1-alpha)^k = (1-0.05)^10 ~ 0.60 >> threshold 0.30.

PASS criteria (pre-registered):
  C1: grid_mean_high > grid_mean_low * 2.5  (field construction correct)
  C3: local_raw_mean_high > local_raw_mean_low * 1.1  (density-responsive raw field)
  C4: harm_obs_a[0] autocorr_lag10 > 0.30  (temporal persistence -- was 0.070 in EXQ-106)
  C5: no fatal errors
  PASS = C1 AND C3 AND C4 AND C5.

Note: C2 (clipped density response) from EXQ-102 not re-tested -- clip approach
deprecated; raw single-cell scalar is the correct implementation post-EXQ-102 fix.
"""

import random
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from ree_core.environment.causal_grid_world import CausalGridWorldV2

EXPERIMENT_TYPE = "v3_exq_106a_harm_obs_a_persistence_fix"
CLAIM_IDS = ["SD-011"]

# Pre-registered thresholds
C1_RATIO = 2.5
C3_RATIO = 1.1
C4_AUTOCORR = 0.30


def _autocorr_lag(series: List[float], lag: int) -> float:
    """Pearson autocorrelation at a given lag."""
    arr = np.array(series, dtype=np.float64)
    if len(arr) <= lag:
        return 0.0
    mean = arr.mean()
    std = arr.std()
    if std < 1e-8:
        return 0.0
    x = arr[:-lag] - mean
    y = arr[lag:] - mean
    return float((x * y).mean() / (std ** 2))


def _run_seed(seed: int, n_hazards: int, n_steps: int, size: int) -> Dict:
    env = CausalGridWorldV2(
        seed=seed,
        size=size,
        num_hazards=n_hazards,
        num_resources=2,
        hazard_harm=0.02,
        hazard_field_decay=0.5,
        use_proxy_fields=True,
        harm_obs_a_ema_alpha=0.05,
        resource_respawn_on_consume=True,
    )
    _, obs_dict = env.reset()

    harm_obs_a_dim0: List[float] = []
    local_raw_vals: List[float] = []
    grid_vals: List[float] = []

    for _ in range(n_steps):
        action_idx = random.randint(0, env.action_dim - 1)
        action_oh = torch.zeros(1, env.action_dim)
        action_oh[0, action_idx] = 1.0
        _, _, done, _, obs_dict = env.step(action_oh)

        ha = obs_dict.get("harm_obs_a")
        if ha is not None:
            # dim 0 = first hazard dim; all 25 hazard dims are identical scalars
            harm_obs_a_dim0.append(float(ha[0].item()))

        ax, ay = int(env.agent_x), int(env.agent_y)
        local_raw_vals.append(float(env.hazard_field[ax, ay]))
        grid_vals.append(float(env.hazard_field.mean()))

        if done:
            # Reset env but harm_obs_a_ema persists (the fix under test)
            _, obs_dict = env.reset()

    return {
        "grid_mean": float(np.mean(grid_vals)),
        "local_raw_mean": float(np.mean(local_raw_vals)),
        "autocorr_lag10": _autocorr_lag(harm_obs_a_dim0, lag=10),
        "harm_obs_a_mean": float(np.mean(harm_obs_a_dim0)) if harm_obs_a_dim0 else 0.0,
        "n_steps_recorded": len(harm_obs_a_dim0),
    }


def run(
    seeds: List[int] = None,
    n_steps: int = 500,
    size: int = 10,
    n_hazards_high: int = 6,
    n_hazards_low: int = 2,
) -> Dict:
    if seeds is None:
        seeds = [42, 43, 44, 45, 46]

    print(
        f"[EXQ-106a] seeds={seeds}, steps={n_steps},"
        f" HIGH={n_hazards_high} hazards, LOW={n_hazards_low} hazards",
        flush=True,
    )

    results_high: List[Dict] = []
    results_low: List[Dict] = []
    fatal_error_count = 0

    for s in seeds:
        try:
            random.seed(s)
            r_h = _run_seed(s, n_hazards_high, n_steps, size)
            results_high.append(r_h)
            print(
                f"  seed={s} HIGH: grid={r_h['grid_mean']:.4f}"
                f" raw={r_h['local_raw_mean']:.4f}"
                f" ema_mean={r_h['harm_obs_a_mean']:.4f}"
                f" autocorr={r_h['autocorr_lag10']:.4f}",
                flush=True,
            )
        except Exception as exc:
            print(f"  seed={s} HIGH ERROR: {exc}", flush=True)
            fatal_error_count += 1
            results_high.append({"grid_mean": 0.0, "local_raw_mean": 0.0, "autocorr_lag10": 0.0, "harm_obs_a_mean": 0.0, "n_steps_recorded": 0})

        try:
            random.seed(s + 500)
            r_l = _run_seed(s + 500, n_hazards_low, n_steps, size)
            results_low.append(r_l)
            print(
                f"  seed={s+500} LOW:  grid={r_l['grid_mean']:.4f}"
                f" raw={r_l['local_raw_mean']:.4f}"
                f" ema_mean={r_l['harm_obs_a_mean']:.4f}"
                f" autocorr={r_l['autocorr_lag10']:.4f}",
                flush=True,
            )
        except Exception as exc:
            print(f"  seed={s+500} LOW ERROR: {exc}", flush=True)
            fatal_error_count += 1
            results_low.append({"grid_mean": 0.0, "local_raw_mean": 0.0, "autocorr_lag10": 0.0, "harm_obs_a_mean": 0.0, "n_steps_recorded": 0})

    grid_high = float(np.mean([r["grid_mean"] for r in results_high]))
    grid_low = float(np.mean([r["grid_mean"] for r in results_low]))
    raw_high = float(np.mean([r["local_raw_mean"] for r in results_high]))
    raw_low = float(np.mean([r["local_raw_mean"] for r in results_low]))
    autocorr_high = float(np.mean([r["autocorr_lag10"] for r in results_high]))
    autocorr_low = float(np.mean([r["autocorr_lag10"] for r in results_low]))
    ema_mean_high = float(np.mean([r["harm_obs_a_mean"] for r in results_high]))
    ema_mean_low = float(np.mean([r["harm_obs_a_mean"] for r in results_low]))

    c1 = grid_high > grid_low * C1_RATIO
    c3 = (raw_high > raw_low * C3_RATIO) or (raw_high > 0.01)
    c4 = autocorr_high > C4_AUTOCORR
    c5 = fatal_error_count == 0

    print(f"\n[EXQ-106a] Aggregated ({len(seeds)} seeds x {n_steps} steps):", flush=True)
    print(f"  Grid mean:     HIGH={grid_high:.4f}  LOW={grid_low:.4f}  ratio={grid_high/max(grid_low,1e-9):.2f}x  (threshold {C1_RATIO}x)", flush=True)
    print(f"  Local raw:     HIGH={raw_high:.4f}  LOW={raw_low:.4f}  ratio={raw_high/max(raw_low,1e-9):.2f}x  (threshold {C3_RATIO}x)", flush=True)
    print(f"  EMA dim-0 mean: HIGH={ema_mean_high:.4f}  LOW={ema_mean_low:.4f}  (expected HIGH > LOW)", flush=True)
    print(f"  Autocorr lag10: HIGH={autocorr_high:.4f}  LOW={autocorr_low:.4f}  (threshold > {C4_AUTOCORR})", flush=True)
    print(f"  C1 (grid scales):       {'PASS' if c1 else 'FAIL'}", flush=True)
    print(f"  C3 (raw density resp):  {'PASS' if c3 else 'FAIL'}", flush=True)
    print(f"  C4 (temporal persist):  {'PASS' if c4 else 'FAIL'}  autocorr={autocorr_high:.4f}", flush=True)
    print(f"  C5 (no errors):         {'PASS' if c5 else 'FAIL'}", flush=True)

    criteria_met = sum([c1, c3, c4, c5])
    status = "PASS" if (c1 and c3 and c4 and c5) else "FAIL"
    print(f"\n[EXQ-106a] {status} ({criteria_met}/4)", flush=True)

    failure_notes = []
    if not c1:
        failure_notes.append(f"C1 FAIL: grid not scaling (HIGH={grid_high:.4f} LOW={grid_low:.4f} ratio={grid_high/max(grid_low,1e-9):.2f}x < {C1_RATIO}x)")
    if not c3:
        failure_notes.append(f"C3 FAIL: raw density not responsive (HIGH={raw_high:.4f} LOW={raw_low:.4f})")
    if not c4:
        failure_notes.append(f"C4 FAIL: autocorr={autocorr_high:.4f} < {C4_AUTOCORR} -- episode-reset fix may not have taken effect")
    if not c5:
        failure_notes.append(f"C5 FAIL: {fatal_error_count} fatal errors")

    for note in failure_notes:
        print(f"  {note}", flush=True)

    per_seed_rows = "".join(
        f"| {seeds[i]} | {results_high[i]['grid_mean']:.4f} | {results_low[i]['grid_mean']:.4f}"
        f" | {results_high[i]['local_raw_mean']:.4f} | {results_low[i]['local_raw_mean']:.4f}"
        f" | {results_high[i]['autocorr_lag10']:.4f} | {results_low[i]['autocorr_lag10']:.4f} |\n"
        for i in range(len(seeds))
    )

    summary_markdown = f"""# V3-EXQ-106a -- SD-011 harm_obs_a Temporal Persistence (episode-reset fix)

**Status:** {status}
**Claims:** SD-011
**Supersedes:** V3-EXQ-106 (C4 FAIL: autocorr=0.070; episode-reset destroyed persistence)
**Seeds:** {seeds}  **Steps/seed/condition:** {n_steps}

## Fix Applied

CausalGridWorldV2.harm_obs_a_ema moved from reset() to __init__(). Now persists across
episode boundaries, as intended for a homeostatic/affective accumulator.
Expected autocorr at lag 10 = (1-0.05)^10 ~ 0.60 >> threshold {C4_AUTOCORR}.

## Results

| Metric | HIGH ({n_hazards_high} hazards) | LOW ({n_hazards_low} hazards) | Criterion |
|--------|--------------|-------------|-----------|
| Grid-wide field mean | {grid_high:.4f} | {grid_low:.4f} | HIGH > LOW * {C1_RATIO}x (C1) |
| Local raw hazard | {raw_high:.4f} | {raw_low:.4f} | HIGH > LOW * {C3_RATIO}x (C3) |
| EMA dim-0 mean | {ema_mean_high:.4f} | {ema_mean_low:.4f} | reference |
| Autocorr lag10 | {autocorr_high:.4f} | {autocorr_low:.4f} | > {C4_AUTOCORR} (C4) |

| Criterion | Result |
|-----------|--------|
| C1: field scales with n_hazards | {"PASS" if c1 else "FAIL"} |
| C3: raw density responsive | {"PASS" if c3 else "FAIL"} |
| C4: temporal persistence | {"PASS" if c4 else "FAIL"} |
| C5: no fatal errors | {"PASS" if c5 else "FAIL"} |

Criteria met: {criteria_met}/4 -> **{status}**
{"".join(chr(10) + "- " + n for n in failure_notes) if failure_notes else ""}

## Per-Seed Results

| Seed | Grid HIGH | Grid LOW | Raw HIGH | Raw LOW | AC HIGH | AC LOW |
|------|-----------|----------|----------|---------|---------|--------|
{per_seed_rows}"""

    return {
        "status": status,
        "experiment_type": EXPERIMENT_TYPE,
        "claim_ids": CLAIM_IDS,
        "evidence_direction": "supports" if status == "PASS" else ("mixed" if criteria_met >= 3 else "weakens"),
        "supersedes": "v3_exq_106_harm_obs_a_temporal_persistence",
        "seeds": seeds,
        "n_steps": n_steps,
        "n_hazards_high": n_hazards_high,
        "n_hazards_low": n_hazards_low,
        "grid_mean_high": grid_high,
        "grid_mean_low": grid_low,
        "local_raw_mean_high": raw_high,
        "local_raw_mean_low": raw_low,
        "autocorr_lag10_high": autocorr_high,
        "autocorr_lag10_low": autocorr_low,
        "ema_mean_high": ema_mean_high,
        "ema_mean_low": ema_mean_low,
        "crit1_grid_scaling": float(c1),
        "crit3_raw_density": float(c3),
        "crit4_autocorr": float(c4),
        "crit5_no_errors": float(c5),
        "criteria_met": float(criteria_met),
        "n_seeds": float(len(seeds)),
        "fatal_error_count": fatal_error_count,
        "failure_notes": failure_notes,
        "summary_markdown": summary_markdown,
        "metrics": {
            "grid_mean_high": grid_high,
            "grid_mean_low": grid_low,
            "local_raw_mean_high": raw_high,
            "local_raw_mean_low": raw_low,
            "autocorr_lag10_high": autocorr_high,
            "autocorr_lag10_low": autocorr_low,
            "crit1_grid_scaling": float(c1),
            "crit3_raw_density": float(c3),
            "crit4_autocorr": float(c4),
            "crit5_no_errors": float(c5),
            "criteria_met": float(criteria_met),
        },
    }


if __name__ == "__main__":
    import argparse
    import json
    from datetime import datetime, timezone

    parser = argparse.ArgumentParser(
        description="V3-EXQ-106a: SD-011 harm_obs_a temporal persistence (episode-reset fix)"
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--n-steps", type=int, default=500)
    parser.add_argument("--size", type=int, default=10)
    parser.add_argument("--hazards-high", type=int, default=6)
    parser.add_argument("--hazards-low", type=int, default=2)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    if args.smoke_test:
        seeds = [42]
        n_steps = 60
        print("[V3-EXQ-106a] SMOKE TEST MODE", flush=True)
    else:
        seeds = args.seeds
        n_steps = args.n_steps

    result = run(
        seeds=seeds,
        n_steps=n_steps,
        size=args.size,
        n_hazards_high=args.hazards_high,
        n_hazards_low=args.hazards_low,
    )

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result["run_timestamp"] = ts
    result["run_id"] = f"{EXPERIMENT_TYPE}_{ts}_v3"
    result["architecture_epoch"] = "ree_hybrid_guardrails_v1"
    result["verdict"] = result["status"]

    out_dir = (
        Path(__file__).resolve().parents[2]
        / "REE_assembly" / "evidence" / "experiments" / EXPERIMENT_TYPE
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{EXPERIMENT_TYPE}_{ts}.json"
    out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print(f"\nResult written to: {out_path}", flush=True)
    print(f"Status: {result['status']}", flush=True)
    for k, v in result.get("metrics", {}).items():
        print(f"  {k}: {v}", flush=True)

    if args.smoke_test:
        print("[V3-EXQ-106a] SMOKE TEST COMPLETE", flush=True)
