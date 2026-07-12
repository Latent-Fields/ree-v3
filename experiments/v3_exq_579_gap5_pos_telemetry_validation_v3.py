"""
V3-EXQ-579: GAP-5 H_pos / zone_coverage telemetry env feature substrate validation.

ARM_0_off:      pos_telemetry_enabled=False, microhabitat off -- info keys present
                but inert (pos_entropy==-1.0, zone_coverage=={}). Also asserts a
                bit-identical agent/hazard/resource layout vs a default-ON env at
                the same seed across resets (telemetry has no RNG / no dynamics
                feedback, so results are unaffected ON or OFF).
ARM_1_on_stub:  default (pos_telemetry on, microhabitat off) -- single-zone-0
                stub. pos_entropy populates and stays under the ln(window)
                ceiling; zone_coverage == {0: f} with 0 < f <= 1 and monotone
                non-decreasing within an episode.
ARM_2_on_zones: pos_telemetry on + microhabitat_enabled (n_microhabitats=3) --
                zone_coverage keys subset {0,1,2,3}, each fraction in [0, 1].

A small window-cap micro-probe (pos_entropy_window=8) runs inside ARM_1 per seed:
the reported pos_entropy must match an independent recompute over only the last
<=window positions (C4).

PASS = C0 AND C1 AND C2 AND C3 AND C4 across all seeds:
  C0  ARM_0 inert sentinels every tick (pos_entropy==-1.0, zone_coverage=={},
      pos_telemetry_enabled is False) AND ARM_0 layout bit-identical to a
      default-ON env at the same seed across resets.
  C1  ARM_1 pos_entropy moves above 0 once the agent has visited >1 cell, never
      exceeds ln(pos_entropy_window) (+tol), and pos_entropy_window echo == 100.
  C2  ARM_1 zone_coverage == {0: f} with 0 < f <= 1 by episode end and
      cov[0] monotone non-decreasing within every episode.
  C3  ARM_2 zone_coverage keys subset {0,1,2,3}, every fraction in [0, 1].
  C4  ARM_1 window-cap probe: |info pos_entropy - independent recompute over
      env._pos_window| < 1e-9 every probe tick AND len(_pos_window) <= 8.

experiment_purpose: diagnostic (substrate readiness test, not a claim hypothesis
test). Unblocks DEV-NEED-001 (H_pos exploration-spread blocking gate) and
DEV-NEED-008 per REE_assembly/evidence/planning/infant_substrate_plan.md (GAP-5).
claim_ids: [] (env feature / telemetry validation only; no governance weighting).
"""

import argparse
import json
import math
import os
import random
import sys
from datetime import datetime

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from experiment_protocol import emit_outcome
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from experiments.pack_writer import write_flat_manifest  # noqa: E402
from pathlib import Path  # noqa: E402

EXPERIMENT_PURPOSE = "diagnostic"
QUEUE_ID = "V3-EXQ-579"

SEEDS = [0, 1, 2]
N_EPISODES = 100
N_STEPS = 200

WINDOW = 100
ENTROPY_CEILING = math.log(WINDOW)  # uniform-over-window upper bound (nats)
CEILING_TOL = 1e-9

PROBE_WINDOW = 8
PROBE_STEPS = 40
PROBE_TOL = 1e-9

CONDITIONS = [
    ("ARM_0_off", "off"),
    ("ARM_1_on_stub", "on_stub"),
    ("ARM_2_on_zones", "on_zones"),
]


def _layout(env):
    env.reset()
    return (
        (env.agent_x, env.agent_y),
        sorted(tuple(h) for h in env.hazards),
        sorted(tuple(r) for r in env.resources),
    )


def _recompute_entropy(window):
    """Independent nats Shannon entropy of a position window (the contract
    the env's _pos_entropy must satisfy)."""
    if not window:
        return -1.0
    counts = {}
    for cell in window:
        counts[cell] = counts.get(cell, 0) + 1
    total = float(len(window))
    h = 0.0
    for c in counts.values():
        p = c / total
        h -= p * math.log(p)
    return max(0.0, h)


def run_experiment(n_episodes, dry_run=False):
    results_by_seed = {}
    c0_by_seed = []
    c1_by_seed = []
    c2_by_seed = []
    c3_by_seed = []
    c4_by_seed = []

    for seed in SEEDS:
        results_by_seed[seed] = {}
        for cond_label, mode in CONDITIONS:
            print(f"Seed {seed} Condition {cond_label}", flush=True)

            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

            if mode == "off":
                env = CausalGridWorldV2(
                    size=12, seed=seed, num_hazards=2, num_resources=3,
                    use_proxy_fields=False, pos_telemetry_enabled=False,
                )
            elif mode == "on_stub":
                env = CausalGridWorldV2(
                    size=12, seed=seed, num_hazards=2, num_resources=3,
                    use_proxy_fields=False,
                )
            else:  # on_zones
                env = CausalGridWorldV2(
                    size=14, seed=seed, num_hazards=2, num_resources=3,
                    use_proxy_fields=False,
                    microhabitat_enabled=True, n_microhabitats=3,
                )

            inert_violations = 0
            layout_match = None
            max_pos_entropy = -1.0
            ceiling_violations = 0
            window_echo_ok = True
            cov_key_violations = 0
            cov_range_violations = 0
            cov_monotone_violations = 0
            final_stub_cov0 = -1.0
            n_zone_cov_obs = 0
            probe_max_abs_err = 0.0
            probe_len_violations = 0

            if mode == "off":
                # C0 layout bit-identical: default-ON env vs this explicit-OFF
                # env at the same seed must produce the same placement.
                env_on = CausalGridWorldV2(
                    size=12, seed=seed, num_hazards=2, num_resources=3,
                    use_proxy_fields=False,
                )
                layout_match = all(
                    _layout(env) == _layout(env_on) for _ in range(5)
                )

            if mode == "on_stub":
                # C4 window-cap micro-probe with a small window.
                pe = CausalGridWorldV2(
                    size=10, seed=seed, num_hazards=0, num_resources=0,
                    use_proxy_fields=False, pos_entropy_window=PROBE_WINDOW,
                )
                pe.reset()
                for _ in range(PROBE_STEPS):
                    _, _, pdone, pinfo, _ = pe.step(np.random.randint(0, 5))
                    indep = _recompute_entropy(pe._pos_window)
                    probe_max_abs_err = max(
                        probe_max_abs_err,
                        abs(pinfo["pos_entropy"] - indep),
                    )
                    if len(pe._pos_window) > PROBE_WINDOW:
                        probe_len_violations += 1
                    if pdone:
                        pe.reset()

            for ep in range(n_episodes):
                env.reset()
                done = False
                prev_cov0 = -1.0
                last_info = None
                for _ in range(N_STEPS):
                    if done:
                        break
                    action = np.random.randint(0, 5)
                    _, _, done, info, _ = env.step(action)
                    last_info = info

                    if mode == "off":
                        if (
                            info["pos_telemetry_enabled"] is not False
                            or info["pos_entropy"] != -1.0
                            or info["zone_coverage"] != {}
                        ):
                            inert_violations += 1
                        continue

                    pe_val = info["pos_entropy"]
                    if pe_val > max_pos_entropy:
                        max_pos_entropy = pe_val
                    if pe_val > ENTROPY_CEILING + CEILING_TOL:
                        ceiling_violations += 1
                    if info["pos_entropy_window"] != WINDOW:
                        window_echo_ok = False

                    zc = info["zone_coverage"]
                    n_zone_cov_obs += 1
                    if mode == "on_stub":
                        if set(zc.keys()) != {0}:
                            cov_key_violations += 1
                        else:
                            v = zc[0]
                            if not (0.0 <= v <= 1.0):
                                cov_range_violations += 1
                            if prev_cov0 >= 0.0 and v < prev_cov0 - 1e-12:
                                cov_monotone_violations += 1
                            prev_cov0 = v
                    else:  # on_zones
                        if not set(zc.keys()).issubset({0, 1, 2, 3}):
                            cov_key_violations += 1
                        if any(not (0.0 <= v <= 1.0) for v in zc.values()):
                            cov_range_violations += 1

                if (
                    mode == "on_stub"
                    and last_info is not None
                    and last_info["zone_coverage"]
                ):
                    final_stub_cov0 = last_info["zone_coverage"].get(0, -1.0)

                print_interval = max(1, n_episodes // 5)
                if (ep + 1) % print_interval == 0:
                    print(
                        f"  [train] seed={seed} cond={cond_label} "
                        f"ep {ep + 1}/{n_episodes}",
                        flush=True,
                    )

            results_by_seed[seed][cond_label] = {
                "mode": mode,
                "inert_violations": inert_violations,
                "layout_match": layout_match,
                "max_pos_entropy": max_pos_entropy,
                "ceiling_violations": ceiling_violations,
                "window_echo_ok": window_echo_ok,
                "cov_key_violations": cov_key_violations,
                "cov_range_violations": cov_range_violations,
                "cov_monotone_violations": cov_monotone_violations,
                "final_stub_cov0": final_stub_cov0,
                "n_zone_cov_obs": n_zone_cov_obs,
                "probe_max_abs_err": probe_max_abs_err,
                "probe_len_violations": probe_len_violations,
            }

            if mode == "off":
                c0 = inert_violations == 0 and layout_match is True
                c0_by_seed.append(c0)
                passed = c0
            elif mode == "on_stub":
                c1 = (
                    max_pos_entropy > 0.0
                    and ceiling_violations == 0
                    and window_echo_ok
                )
                c2 = (
                    cov_key_violations == 0
                    and cov_range_violations == 0
                    and cov_monotone_violations == 0
                    and 0.0 < final_stub_cov0 <= 1.0
                )
                c4 = (
                    probe_max_abs_err < PROBE_TOL
                    and probe_len_violations == 0
                )
                c1_by_seed.append(c1)
                c2_by_seed.append(c2)
                c4_by_seed.append(c4)
                passed = c1 and c2 and c4
            else:  # on_zones
                c3 = (
                    n_zone_cov_obs > 0
                    and cov_key_violations == 0
                    and cov_range_violations == 0
                )
                c3_by_seed.append(c3)
                passed = c3

            print(f"verdict: {'PASS' if passed else 'FAIL'}", flush=True)

    c0_pass = all(c0_by_seed) if c0_by_seed else False
    c1_pass = all(c1_by_seed) if c1_by_seed else False
    c2_pass = all(c2_by_seed) if c2_by_seed else False
    c3_pass = all(c3_by_seed) if c3_by_seed else False
    c4_pass = all(c4_by_seed) if c4_by_seed else False
    overall_pass = c0_pass and c1_pass and c2_pass and c3_pass and c4_pass

    return {
        "outcome": "PASS" if overall_pass else "FAIL",
        "c0_pass": c0_pass,
        "c1_pass": c1_pass,
        "c2_pass": c2_pass,
        "c3_pass": c3_pass,
        "c4_pass": c4_pass,
        "c0_by_seed": c0_by_seed,
        "c1_by_seed": c1_by_seed,
        "c2_by_seed": c2_by_seed,
        "c3_by_seed": c3_by_seed,
        "c4_by_seed": c4_by_seed,
        "results_by_seed": results_by_seed,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    dry_run = args.dry_run
    n_episodes = 5 if dry_run else N_EPISODES

    print("V3-EXQ-579 GAP-5 H_pos / zone_coverage telemetry validation", flush=True)
    print(
        f"  dry_run={dry_run} n_episodes={n_episodes} seeds={SEEDS} "
        f"window={WINDOW} probe_window={PROBE_WINDOW}",
        flush=True,
    )

    result = run_experiment(n_episodes=n_episodes, dry_run=dry_run)

    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"v3_exq_579_gap5_pos_telemetry_validation_{timestamp}_v3"

    manifest = {
        "run_id": run_id,
        "queue_id": QUEUE_ID,
        "experiment_type": "gap5_pos_telemetry_validation",
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "claim_ids": [],
        "evidence_direction": "non_contributory",
        "timestamp_utc": timestamp,
        "outcome": result["outcome"],
        "dry_run": dry_run,
        "config": {
            "seeds": SEEDS,
            "n_episodes": n_episodes,
            "n_steps": N_STEPS,
            "window": WINDOW,
            "entropy_ceiling": ENTROPY_CEILING,
            "probe_window": PROBE_WINDOW,
            "probe_steps": PROBE_STEPS,
        },
        "acceptance_checks": {
            "C0_arm0_inert_and_layout_bit_identical": result["c0_pass"],
            "C1_arm1_pos_entropy_populates_under_ceiling": result["c1_pass"],
            "C2_arm1_stub_zone_coverage_monotone": result["c2_pass"],
            "C3_arm2_gap2_zone_coverage_keys_range": result["c3_pass"],
            "C4_arm1_window_cap_recompute_match": result["c4_pass"],
        },
        "c0_by_seed": result["c0_by_seed"],
        "c1_by_seed": result["c1_by_seed"],
        "c2_by_seed": result["c2_by_seed"],
        "c3_by_seed": result["c3_by_seed"],
        "c4_by_seed": result["c4_by_seed"],
        "results_by_seed": result["results_by_seed"],
    }

    out_dir = os.path.join(
        os.path.dirname(__file__), "..", "..", "REE_assembly", "evidence", "experiments"
    )
    os.makedirs(out_dir, exist_ok=True)
    out_path = write_flat_manifest(
        manifest,
        out_dir,
        dry_run=False,
        config=manifest.get("config"),
        seeds=SEEDS,
        script_path=Path(__file__),
    )

    print(f"Manifest written: {out_path}", flush=True)
    print(f"Outcome: {result['outcome']}", flush=True)
    print(f"  C0 (ARM_0 inert + layout bit-identical): {result['c0_pass']}", flush=True)
    print(f"  C1 (ARM_1 pos_entropy under ceiling):    {result['c1_pass']}", flush=True)
    print(f"  C2 (ARM_1 stub coverage monotone):       {result['c2_pass']}", flush=True)
    print(f"  C3 (ARM_2 GAP-2 coverage keys/range):    {result['c3_pass']}", flush=True)
    print(f"  C4 (ARM_1 window-cap recompute match):   {result['c4_pass']}", flush=True)

    _outcome_raw = str(result["outcome"]).upper()
    emit_outcome(
        outcome=_outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL",
        manifest_path=out_path,
    )
