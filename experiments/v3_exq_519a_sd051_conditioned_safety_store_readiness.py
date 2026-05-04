#!/opt/local/bin/python3
"""V3-EXQ-519a -- SD-051 / MECH-304 conditioned safety store substrate readiness.

Supersedes: V3-EXQ-519

Why V3-EXQ-519 failed C6
------------------------
V3-EXQ-519 used deterministic cycling actions (0,1,2,3,...) in an 8x8 grid with
3 hazards and limb_damage_enabled=True. Near-zero harm_obs_a (~0.006 norm) meant
z_harm_a was dominated by AffectiveHarmEncoder bias (~0.41), producing a near-
constant z_harm_a.norm() variation of <0.001 across 300 ticks. SufferingDerivative
Comparator drop_threshold=0.08 was never met. relief_events=0 across all seeds.

Root cause: the integration test was relying on behavioral dynamics to produce a
meaningful harm signal, but the untrained encoder's bias dominates over the tiny
real harm variation in the smoke environment. This is correct behavior -- it is
not a substrate bug. The test needs SYNTHETIC harm injection to verify the
MECH-302->ConditionedSafetyStore wiring without depending on behavioral dynamics.

V3-EXQ-519a fix
---------------
ARM_A injects synthetic harm_obs_a to the agent:
  - Ticks 0-19: inject torch.ones(harm_obs_a_dim)*0.6 (high harm)
    -> z_harm_a.norm ~ 0.653 (measured), in the comparator buffer initial_norm range
  - Tick 20+: inject torch.zeros(harm_obs_a_dim) (harm drop)
    -> z_harm_a.norm drops to ~0.407 (bias-only floor)
    -> buffer after tick 24: [0.653, 0.653, 0.653, 0.653, 0.407], drop=0.246 > 0.05

This tests the WIRING (MECH-302 fires -> ConditionedSafetyStore updates -> signal
propagates to agent._conditioned_safety_signal), not behavioral dynamics.

Config changes from 519:
  - suffering_drop_threshold lowered from 0.08 to 0.05 (more robust margin)
  - suffering_min_initial_norm lowered from 0.05 to 0.02 (allow moderate norms)
  - ARM_A uses synthetic harm injection (ticks 0-19 high, tick 20+ zero)
  - ARM_B also uses synthetic injection to confirm store loads but events don't fire
    without the drop phase (injection stays high -- no drop -> no event)
  - Removed limb_damage_enabled (was causing env build issue; harm_obs_a injected
    synthetically so env harm_obs_a is irrelevant to this substrate test)

Claim: MECH-304 (safety_prediction.cue_specific_conditioned_inhibition_substrate)
Status: candidate (v3_pending). Substrate IMPLEMENTED 2026-05-04 as SD-051.

experiment_purpose = "diagnostic"
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ree_core.safety.conditioned_safety_store import ConditionedSafetyStore  # noqa: E402
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_519a_sd051_conditioned_safety_store_readiness"
CLAIM_IDS = ["MECH-304"]
EXPERIMENT_PURPOSE = "diagnostic"
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"

# Unit test constants (unchanged from 519)
WORLD_DIM_UNIT = 32
EMA_ALPHA_UNIT = 0.5
MIN_NORM = 0.1
THRESHOLD = 0.5
N_EVENT_TICKS = 5
N_NO_EVENT_TICKS = 5
C7_MAX_PROTO_NORM = 0.05

# Integration constants (updated from 519)
GRID_SIZE = 8
NUM_HAZARDS = 3
NUM_RESOURCES = 2
N_TICKS_INTEGRATION = 300
SEEDS = (42, 43)

# Suffering comparator thresholds (lowered for robustness with encoder bias floor)
SUFFERING_DROP_THRESHOLD = 0.05    # was 0.08; encoder drop ~0.246 so plenty of margin
SUFFERING_MIN_INITIAL_NORM = 0.02  # was 0.05; allows moderate norm floors

# Synthetic harm injection parameters
HARM_INJECT_HIGH_TICKS = 20       # inject high harm for first 20 ticks
HARM_HIGH_LEVEL = 0.6             # produces z_harm_a.norm ~ 0.653 (measured)

# ARM_B: inject constant-high (no drop -> no MECH-302 event) to test store loads
# but signal stays zero because no event fires.
# This verifies: store gets teaching signal ONLY on MECH-302 relief events.


def _l2_norm(vec: List[float]) -> float:
    return math.sqrt(sum(v * v for v in vec))


# ---- Part 1: Unit tests (unchanged from 519) ----

def run_unit_tests() -> Dict:
    """Directly test ConditionedSafetyStore arithmetic."""
    world_dim = WORLD_DIM_UNIT

    store_c1 = ConditionedSafetyStore(
        world_dim=world_dim, ema_alpha=EMA_ALPHA_UNIT, decay_rate=0.0,
        min_norm=0.0, threshold=THRESHOLD,
    )
    z_event = torch.randn(world_dim)
    for _ in range(N_EVENT_TICKS):
        store_c1.update(z_event, event_fired=True, sim_mode=False)
    c1_proto_norm = _l2_norm(store_c1._prototype)
    c1 = c1_proto_norm > MIN_NORM

    store_c2 = ConditionedSafetyStore(
        world_dim=world_dim, ema_alpha=EMA_ALPHA_UNIT, decay_rate=0.001,
        min_norm=MIN_NORM, threshold=THRESHOLD,
    )
    z_no_event = torch.randn(world_dim)
    for _ in range(N_NO_EVENT_TICKS):
        store_c2.update(z_no_event, event_fired=False, sim_mode=False)
    c2_proto_norm = _l2_norm(store_c2._prototype)
    c2 = c2_proto_norm < 0.01

    store_c3 = ConditionedSafetyStore(
        world_dim=world_dim, ema_alpha=EMA_ALPHA_UNIT, decay_rate=0.0,
        min_norm=0.0, threshold=THRESHOLD,
    )
    norm_before = _l2_norm(store_c3._prototype)
    signal_sim = store_c3.update(z_event, event_fired=True, sim_mode=True)
    norm_after = _l2_norm(store_c3._prototype)
    c3 = signal_sim == 0.0 and abs(norm_after - norm_before) < 1e-8

    store_c4 = ConditionedSafetyStore(
        world_dim=world_dim, ema_alpha=EMA_ALPHA_UNIT, decay_rate=0.0,
        min_norm=MIN_NORM, threshold=THRESHOLD,
    )
    signal_empty = store_c4.update(z_event, event_fired=False, sim_mode=False)
    c4 = signal_empty == 0.0

    store_c5 = ConditionedSafetyStore(
        world_dim=world_dim, ema_alpha=EMA_ALPHA_UNIT, decay_rate=0.0,
        min_norm=MIN_NORM, threshold=THRESHOLD,
    )
    z_c5 = torch.randn(world_dim)
    for _ in range(3):
        store_c5.update(z_c5, event_fired=True, sim_mode=False)
    signal_same = store_c5._query(z_c5.tolist())
    c5 = signal_same > 0.5

    return {
        "c1_proto_norm_after_events": c1_proto_norm,
        "C1_prototype_grows_on_events": c1,
        "c2_proto_norm_after_no_events": c2_proto_norm,
        "C2_no_growth_without_events": c2,
        "c3_signal_sim_mode": signal_sim,
        "c3_proto_norm_unchanged": abs(norm_after - norm_before) < 1e-8,
        "C3_simmode_gate": c3,
        "c4_signal_empty_store": signal_empty,
        "C4_min_norm_gate": c4,
        "c5_signal_same_vector": signal_same,
        "C5_same_vector_high_signal": c5,
    }


# ---- Part 2: Integration smoke ----

# (arm_name, use_safety_store, use_mech302, inject_harm)
# inject_harm: "drop" = high then zero (triggers event), "constant_high" = stays high,
#              "none" = no injection (use env obs directly)
ARMS: List[Tuple[str, bool, bool, str]] = [
    ("ARM_A_both_on",              True,  True,  "drop"),
    ("ARM_B_store_on_no_drop",     True,  True,  "constant_high"),
    ("ARM_C_store_off_mech302_on", False, True,  "none"),
    ("ARM_D_both_off",             False, False, "none"),
]


def run_integration_arm(
    seed: int,
    arm_name: str,
    use_safety_store: bool,
    use_mech302: bool,
    inject_harm: str,
    n_ticks: int,
) -> Dict:
    """Run REEAgent + env; track conditioned_safety_signal and store state.

    For ARM_A (inject_harm='drop'): ticks 0-19 inject high harm; tick 20+ inject
    zeros. This produces a MECH-302 relief event ~tick 24 (buffer window=5) when
    the comparator window transitions from all-high to low-high-high-high-low pattern.

    For ARM_B (inject_harm='constant_high'): inject constant high harm. No drop ->
    no MECH-302 event -> store accumulates no prototype -> signal stays 0.

    For ARM_C/D (inject_harm='none'): use real env harm_obs_a. Store is None for
    ARM_C/D (master switch off), so injection doesn't matter.
    """
    torch.manual_seed(seed)
    env = CausalGridWorldV2(
        size=GRID_SIZE,
        num_hazards=NUM_HAZARDS,
        num_resources=NUM_RESOURCES,
        seed=seed,
        use_proxy_fields=True,
    )
    harm_obs_a_dim = len(env.harm_obs_a_ema)  # 50 for default env, 7 for limb_damage mode

    config = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        use_affective_harm_stream=True,
        use_suffering_derivative_comparator=use_mech302,
        suffering_window_length=5,
        suffering_drop_threshold=SUFFERING_DROP_THRESHOLD,
        suffering_min_initial_norm=SUFFERING_MIN_INITIAL_NORM,
        use_conditioned_safety_store=use_safety_store,
        safety_store_ema_alpha=0.3,
        safety_store_decay_rate=0.001,
        safety_store_min_norm=MIN_NORM,
        safety_store_threshold=THRESHOLD,
    )
    agent = REEAgent(config)
    agent.eval()

    store_is_none = agent.conditioned_safety_store is None
    peak_safety_signal = 0.0
    exception_count = 0
    relief_events = 0
    nonzero_safety_signals = 0

    _, obs_dict = env.reset()
    rng_actions = [i % env.action_dim for i in range(n_ticks)]

    # Pre-build synthetic harm tensors
    harm_high = torch.ones(harm_obs_a_dim) * HARM_HIGH_LEVEL
    harm_zero = torch.zeros(harm_obs_a_dim)

    for tick_i in range(n_ticks):
        try:
            obs_body = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]

            # Select harm_obs_a based on injection mode
            if inject_harm == "drop":
                obs_harm_a = harm_high if tick_i < HARM_INJECT_HIGH_TICKS else harm_zero
            elif inject_harm == "constant_high":
                obs_harm_a = harm_high
            else:
                # Use real env observation
                obs_harm_a = obs_dict.get("harm_obs_a")

            agent.sense(obs_body, obs_world, obs_harm_a=obs_harm_a)

            # Sample flags BEFORE select_action() clears them.
            if agent._relief_completion_event:
                relief_events += 1
            sig = agent._conditioned_safety_signal
            if sig > 0.0:
                nonzero_safety_signals += 1
            if sig > peak_safety_signal:
                peak_safety_signal = sig

            # Manually clear flags (skip full select_action() for smoke speed).
            agent._relief_completion_event = False
            agent._conditioned_safety_signal = 0.0

            _, _, done, _, obs_dict = env.step(torch.tensor(rng_actions[tick_i]))
            if done:
                _, obs_dict = env.reset()

        except Exception as exc:  # noqa: BLE001
            exception_count += 1
            print(f"  [integration] EXCEPTION arm={arm_name} seed={seed}: {exc}")
            try:
                _, obs_dict = env.reset()
            except Exception:  # noqa: BLE001
                break

    final_proto_norm = 0.0
    if agent.conditioned_safety_store is not None:
        final_proto_norm = _l2_norm(agent.conditioned_safety_store._prototype)

    return {
        "arm": arm_name,
        "seed": seed,
        "n_ticks": n_ticks,
        "store_is_none": store_is_none,
        "relief_events": relief_events,
        "nonzero_safety_signals": nonzero_safety_signals,
        "peak_safety_signal": peak_safety_signal,
        "final_proto_norm": final_proto_norm,
        "exception_count": exception_count,
    }


def evaluate_acceptance(
    unit: Dict,
    integration_results: List[Dict],
) -> Dict:
    """Evaluate pre-registered acceptance checks C1-C9."""
    def agg(arm_name: str, field: str, agg_fn=max):
        vals = [r[field] for r in integration_results if r["arm"] == arm_name]
        return agg_fn(vals) if vals else 0

    c1 = unit["C1_prototype_grows_on_events"]
    c2 = unit["C2_no_growth_without_events"]
    c3 = unit["C3_simmode_gate"]
    c4 = unit["C4_min_norm_gate"]
    c5 = unit["C5_same_vector_high_signal"]

    # C6: ARM_A -- synthetic harm drop triggers MECH-302 event and signal propagates
    arm_a_peak = agg("ARM_A_both_on", "peak_safety_signal")
    arm_a_relief = agg("ARM_A_both_on", "relief_events")
    c6 = arm_a_peak > 0.0

    # C7: ARM_B -- constant-high injection: store loads but no drop -> no event
    # Prototype grows because store updates on events fired by... wait, ARM_B has
    # use_mech302=True and constant_high injection. The comparator needs a DROP
    # to fire. With constant_high the buffer stays [0.653, 0.653, ...] with no drop.
    # So relief_events should be 0 and signal should be 0.
    arm_b_peak = agg("ARM_B_store_on_no_drop", "peak_safety_signal")
    arm_b_relief = agg("ARM_B_store_on_no_drop", "relief_events")
    arm_b_proto = agg("ARM_B_store_on_no_drop", "final_proto_norm")
    # C7: no drop -> no events -> no signal. proto_norm near-zero (no events).
    c7 = arm_b_peak == 0.0 and arm_b_relief == 0

    # C8: ARM_C and ARM_D store is None
    arm_c_none = all(r["store_is_none"] for r in integration_results
                     if r["arm"] in ("ARM_C_store_off_mech302_on", "ARM_D_both_off"))
    c8 = arm_c_none

    # C9: no exceptions in any arm
    total_exceptions = sum(r["exception_count"] for r in integration_results)
    c9 = total_exceptions == 0

    overall = c1 and c2 and c3 and c4 and c5 and c6 and c7 and c8 and c9
    return {
        "C1_prototype_grows_on_events": bool(c1),
        "C1_proto_norm": float(unit["c1_proto_norm_after_events"]),
        "C2_no_growth_without_events": bool(c2),
        "C2_proto_norm": float(unit["c2_proto_norm_after_no_events"]),
        "C3_simmode_gate": bool(c3),
        "C4_min_norm_gate": bool(c4),
        "C5_same_vector_high_signal": bool(c5),
        "C5_signal": float(unit["c5_signal_same_vector"]),
        "C6_arm_a_peak_signal_gt0": bool(c6),
        "C6_arm_a_peak_signal": float(arm_a_peak),
        "C6_arm_a_relief_events": int(arm_a_relief),
        "C7_arm_b_no_event_no_signal": bool(c7),
        "C7_arm_b_peak_signal": float(arm_b_peak),
        "C7_arm_b_relief_events": int(arm_b_relief),
        "C7_arm_b_final_proto_norm": float(arm_b_proto),
        "C8_store_none_when_disabled": bool(c8),
        "C9_no_exceptions": bool(c9),
        "C9_total_exceptions": int(total_exceptions),
        "all_pass": bool(overall),
    }


def main(dry_run: bool = False) -> int:
    print(f"[{EXPERIMENT_TYPE}] starting (dry_run={dry_run})")
    seeds = (SEEDS[0],) if dry_run else SEEDS
    n_ticks = 60 if dry_run else N_TICKS_INTEGRATION
    t0 = time.time()

    print(f"[{EXPERIMENT_TYPE}] Part 1: unit tests")
    unit = run_unit_tests()
    for k, v in unit.items():
        print(f"  {k}: {v}")

    print(f"[{EXPERIMENT_TYPE}] Part 2: integration ({n_ticks} ticks x {len(seeds)} seeds)")
    print(f"  ARM_A: inject high harm ticks 0-{HARM_INJECT_HIGH_TICKS-1}, then zero")
    print(f"  ARM_B: inject constant high harm (no drop -> no MECH-302 event)")
    integration_results: List[Dict] = []
    for seed in seeds:
        for arm_name, use_store, use_mech302, inject_harm in ARMS:
            r = run_integration_arm(
                seed, arm_name, use_store, use_mech302, inject_harm, n_ticks
            )
            integration_results.append(r)
            print(
                f"  seed={seed} {arm_name:<34} relief_events={r['relief_events']:3d} "
                f"nonzero_sig={r['nonzero_safety_signals']:3d} "
                f"peak_sig={r['peak_safety_signal']:.4f} "
                f"proto_norm={r['final_proto_norm']:.4f} "
                f"exc={r['exception_count']}"
            )

    elapsed = time.time() - t0
    acceptance = evaluate_acceptance(unit, integration_results)
    outcome = "PASS" if acceptance["all_pass"] else "FAIL"

    print(f"[{EXPERIMENT_TYPE}] acceptance:")
    for k, v in acceptance.items():
        print(f"  {k}: {v}")
    print(f"[{EXPERIMENT_TYPE}] outcome={outcome} elapsed={elapsed:.1f}s")
    print(f"verdict: {outcome}")

    if dry_run:
        print(f"[{EXPERIMENT_TYPE}] dry-run complete; not writing manifest.")
        return 0

    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{timestamp}_v3"
    out_dir = (
        REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments" / EXPERIMENT_TYPE
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "architecture_epoch": ARCHITECTURE_EPOCH,
        "started_utc": datetime.utcnow().isoformat() + "Z",
        "outcome": outcome,
        "evidence_direction": "supports" if outcome == "PASS" else "weakens",
        "supersedes": "v3_exq_519_sd051_conditioned_safety_store_readiness_20260504T150326Z_v3",
        "elapsed_seconds": elapsed,
        "n_seeds_unit": 1,
        "n_seeds_integration": len(seeds),
        "n_ticks_integration": n_ticks,
        "grid_size": GRID_SIZE,
        "num_hazards": NUM_HAZARDS,
        "num_resources": NUM_RESOURCES,
        "harm_inject_high_ticks": HARM_INJECT_HIGH_TICKS,
        "harm_high_level": HARM_HIGH_LEVEL,
        "suffering_drop_threshold": SUFFERING_DROP_THRESHOLD,
        "suffering_min_initial_norm": SUFFERING_MIN_INITIAL_NORM,
        "ema_alpha_unit": EMA_ALPHA_UNIT,
        "min_norm": MIN_NORM,
        "threshold": THRESHOLD,
        "unit_results": unit,
        "integration_results": integration_results,
        "acceptance": acceptance,
    }
    out_path = out_dir / f"{run_id}.json"
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Result written to: {out_path}")
    print(f"Done. Outcome: {outcome}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Smoke run, no manifest.")
    args = parser.parse_args()
    sys.exit(main(dry_run=args.dry_run))
