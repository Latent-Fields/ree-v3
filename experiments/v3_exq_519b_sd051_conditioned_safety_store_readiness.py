#!/opt/local/bin/python3
"""V3-EXQ-519b -- SD-051 / MECH-304 conditioned safety store substrate readiness
(env-curriculum amend).

Supersedes V3-EXQ-519 (2026-05-04 FAIL: relief_events=0 across all integration
arms because MECH-302 substrate had no source of damage->heal trajectories in
the default env). Today (2026-05-30) the SD-022 scheduled-injection extension
landed and V3-EXQ-517c PASSed (3/3 seeds ARM_B==0, 2/3 ARM_A events fired) --
validating the MECH-302 substrate AND lifting gate (c) for V3-EXQ-519 retest.

519b mirrors EXQ-517c's env-config pattern: scheduled_limb_damage_* kwargs are
added to the CausalGridWorldV2 build in run_integration_arm() so MECH-302
relief-completion events actually fire during the integration arms. The
SD-051 conditioned-safety-store substrate, acceptance criteria (C1-C9), unit
tests, and integration-arm structure are otherwise UNCHANGED from V3-EXQ-519.

Claim: MECH-304 (safety_prediction.cue_specific_conditioned_inhibition_substrate)
Status: candidate (v3_pending). Substrate IMPLEMENTED 2026-05-04 as SD-051.

Why this experiment exists
--------------------------
SD-051 adds ConditionedSafetyStore: a non-trainable EMA prototype of z_world
at MECH-302 relief-completion event ticks. This is a SUBSTRATE READINESS
diagnostic -- it verifies the mechanics of the store, not the downstream
conditioned-inhibition behavioural consequence (tested later via discriminative
pair gated by EXQ-517c PASS).

Test structure
--------------
Part 1 (unit): Directly call ConditionedSafetyStore.update() with synthetic
event/no-event sequences and verify prototype behaviour.

Part 2 (integration): Run REEAgent + CausalGridWorldV2 (now with SD-022
scheduled_limb_damage curriculum) for N ticks per arm. Track
conditioned_safety_signal after sense(). Verify:
  - ARM_A (store ON + MECH-302 ON): prototype grows, signal fires after events
  - ARM_B (store ON, MECH-302 OFF): prototype stays near-zero (no teaching signal)
  - ARM_C (store OFF, MECH-302 ON): store is None, signal always 0
  - ARM_D (both OFF): store is None, signal always 0, bit-identical baseline

Pre-registered acceptance criteria (unchanged from V3-EXQ-519)
--------------------------------------------------------------
C1: Unit -- prototype L2 norm after 5 event ticks > MIN_NORM
C2: Unit -- prototype L2 norm after 5 no-event ticks < 0.01
C3: Unit -- update(sim_mode=True) returns 0.0 and prototype norm unchanged
C4: Unit -- update() on empty store (proto_norm < min_norm) returns 0.0
C5: Unit -- same-vector query after event returns > 0.5
C6: Integration -- ARM_A: peak conditioned_safety_signal > 0.0
C7: Integration -- ARM_B: proto_norm at end < C7_MAX_PROTO_NORM
C8: Integration -- ARM_C and ARM_D: conditioned_safety_store is None
C9: Integration -- no exceptions in any arm

PASS = all C1-C9.

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

from experiment_protocol import emit_outcome  # noqa: E402
from ree_core.safety.conditioned_safety_store import ConditionedSafetyStore  # noqa: E402
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_519b_sd051_conditioned_safety_store_readiness"
CLAIM_IDS = ["MECH-304"]
EXPERIMENT_PURPOSE = "diagnostic"
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"
SUPERSEDES = "v3_exq_519_sd051_conditioned_safety_store_readiness"

# Unit test constants
WORLD_DIM_UNIT = 32         # small world_dim for unit tests
EMA_ALPHA_UNIT = 0.5        # high alpha to see effect quickly
MIN_NORM = 0.1
THRESHOLD = 0.5
N_EVENT_TICKS = 5           # number of event ticks for C1
N_NO_EVENT_TICKS = 5        # number of no-event ticks for C2
C7_MAX_PROTO_NORM = 0.05    # ARM_B proto_norm threshold (no events -> should stay tiny)

# Integration constants
# Substrate-readiness env config (NEW for 519b): the SD-022 scheduled-injection
# curriculum is now the sole source of harm (NUM_HAZARDS=0, CONTAMINATED_HARM=0,
# DAMAGE_INCREMENT=0, FAILURE_PROB_SCALE=0). Removes unrelated death modes
# (contamination, wall-bump limb attrition) that were making the round-robin
# random-action harness die in ~13 ticks under the original 519 env -- before
# the curriculum could ever reach its first interval-50 firing. This narrows
# the diagnostic to the store/MECH-302 mechanics, which is the whole point of
# a substrate-readiness check. ARM_A still receives genuine MECH-302 events
# from the curriculum injections + heal_rate=0.002 decay trajectory.
GRID_SIZE = 8
NUM_HAZARDS = 0
NUM_RESOURCES = 2
CONTAMINATED_HARM = 0.0
DAMAGE_INCREMENT = 0.0
FAILURE_PROB_SCALE = 0.0
HEAL_RATE = 0.002
N_TICKS_INTEGRATION = 300
SEEDS = (42, 43)
HARM_EMA_ALPHA = 0.3        # elevated for faster harm dynamics in smoke

# SD-022 scheduled-injection extension curriculum (NEW for 519b; mirrors EXQ-517c).
# Identical across all integration arms so the discriminative logic isolates
# (store, MECH-302) as the only varying factors.
SCHED_INTERVAL = 50
SCHED_PROB = 0.5
SCHED_MAGNITUDE = 0.4
SCHED_LIMB_SELECTION = "random"


# ---- Part 1: Unit tests ----

def _l2_norm(vec: List[float]) -> float:
    return math.sqrt(sum(v * v for v in vec))


def run_unit_tests() -> Dict:
    """Directly test ConditionedSafetyStore arithmetic."""
    world_dim = WORLD_DIM_UNIT

    # C1: prototype grows on events
    store_c1 = ConditionedSafetyStore(
        world_dim=world_dim, ema_alpha=EMA_ALPHA_UNIT, decay_rate=0.0,
        min_norm=0.0,  # disable min_norm guard for C1
        threshold=THRESHOLD,
    )
    z_event = torch.randn(world_dim)
    for _ in range(N_EVENT_TICKS):
        store_c1.update(z_event, event_fired=True, sim_mode=False)
    c1_proto_norm = _l2_norm(store_c1._prototype)
    c1 = c1_proto_norm > MIN_NORM

    # C2: no prototype growth without events
    store_c2 = ConditionedSafetyStore(
        world_dim=world_dim, ema_alpha=EMA_ALPHA_UNIT, decay_rate=0.001,
        min_norm=MIN_NORM, threshold=THRESHOLD,
    )
    z_no_event = torch.randn(world_dim)
    for _ in range(N_NO_EVENT_TICKS):
        store_c2.update(z_no_event, event_fired=False, sim_mode=False)
    c2_proto_norm = _l2_norm(store_c2._prototype)
    c2 = c2_proto_norm < 0.01

    # C3: sim_mode gate -- no prototype advance
    store_c3 = ConditionedSafetyStore(
        world_dim=world_dim, ema_alpha=EMA_ALPHA_UNIT, decay_rate=0.0,
        min_norm=0.0, threshold=THRESHOLD,
    )
    norm_before = _l2_norm(store_c3._prototype)
    signal_sim = store_c3.update(z_event, event_fired=True, sim_mode=True)
    norm_after = _l2_norm(store_c3._prototype)
    c3 = signal_sim == 0.0 and abs(norm_after - norm_before) < 1e-8

    # C4: min_norm gate -- empty store returns 0.0
    store_c4 = ConditionedSafetyStore(
        world_dim=world_dim, ema_alpha=EMA_ALPHA_UNIT, decay_rate=0.0,
        min_norm=MIN_NORM, threshold=THRESHOLD,
    )
    signal_empty = store_c4.update(z_event, event_fired=False, sim_mode=False)
    c4 = signal_empty == 0.0

    # C5: same-vector query after event returns > 0.5
    store_c5 = ConditionedSafetyStore(
        world_dim=world_dim, ema_alpha=EMA_ALPHA_UNIT, decay_rate=0.0,
        min_norm=MIN_NORM, threshold=THRESHOLD,
    )
    z_c5 = torch.randn(world_dim)
    for _ in range(3):  # a few event ticks so prototype norm > min_norm
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

# (arm_name, use_safety_store, use_mech302)
ARMS: List[Tuple[str, bool, bool]] = [
    ("ARM_A_both_on",       True,  True),
    ("ARM_B_store_on_no_mech302", True, False),
    ("ARM_C_store_off_mech302_on", False, True),
    ("ARM_D_both_off",      False, False),
]


def run_integration_arm(
    seed: int,
    arm_name: str,
    use_safety_store: bool,
    use_mech302: bool,
    n_ticks: int,
) -> Dict:
    """Run REEAgent + env; track conditioned_safety_signal and store state."""
    torch.manual_seed(seed)
    env = CausalGridWorldV2(
        size=GRID_SIZE,
        num_hazards=NUM_HAZARDS,
        num_resources=NUM_RESOURCES,
        seed=seed,
        use_proxy_fields=True,
        harm_obs_a_ema_alpha=HARM_EMA_ALPHA,
        contaminated_harm=CONTAMINATED_HARM,
        limb_damage_enabled=True,   # required by SD-022 curriculum
        damage_increment=DAMAGE_INCREMENT,
        failure_prob_scale=FAILURE_PROB_SCALE,
        heal_rate=HEAL_RATE,
        # SD-022 scheduled-injection extension (NEW for 519b; mirrors EXQ-517c).
        # Curriculum is identical across ALL arms so the discriminative logic
        # isolates (store, MECH-302) as the only varying factors.
        scheduled_limb_damage_enabled=True,
        scheduled_limb_damage_interval=SCHED_INTERVAL,
        scheduled_limb_damage_prob=SCHED_PROB,
        scheduled_limb_damage_magnitude=SCHED_MAGNITUDE,
        scheduled_limb_damage_limb_selection=SCHED_LIMB_SELECTION,
    )
    config = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        use_affective_harm_stream=True,
        limb_damage_enabled=True,  # must match env so harm_obs_a_dim=7 is set
        use_suffering_derivative_comparator=use_mech302,
        # Comparator parameters mirror V3-EXQ-517c (validated 2026-05-30 against
        # the SD-022 curriculum heal trajectory). The original 519 parameters
        # (window=5, drop=0.08) required a drop rate >0.016/tick that exceeds
        # the curriculum heal_rate=0.002/tick, so MECH-302 could never fire.
        # 517c-calibrated (window=30, drop=0.005, min_norm=0.01) match the
        # ~0.06-over-30-ticks heal-trajectory signature the curriculum produces.
        suffering_window_length=30,
        suffering_drop_threshold=0.005,
        suffering_min_initial_norm=0.01,
        use_conditioned_safety_store=use_safety_store,
        safety_store_ema_alpha=0.3,  # elevated for faster prototype loading in smoke
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
    scheduled_injections = 0

    _, obs_dict = env.reset()
    rng_actions = [i % env.action_dim for i in range(n_ticks)]
    for tick_i in range(n_ticks):
        try:
            obs_body = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
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
            # Manually clear flags (skipping full select_action() for smoke speed).
            agent._relief_completion_event = False
            agent._conditioned_safety_signal = 0.0
            _, _, done, _info, obs_dict = env.step(torch.tensor(rng_actions[tick_i]))
            # Auxiliary diagnostic: surface curriculum activity.
            if bool(_info.get("scheduled_limb_damage_injected_this_step", False)):
                scheduled_injections += 1
            if done:
                _, obs_dict = env.reset()
        except Exception as exc:  # noqa: BLE001
            exception_count += 1
            print(f"  [integration] EXCEPTION arm={arm_name} seed={seed}: {exc}")
            try:
                _, obs_dict = env.reset()
            except Exception:  # noqa: BLE001
                break

    # Snapshot prototype norm at end (for ARM_B check).
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
        "scheduled_injections": scheduled_injections,
        "exception_count": exception_count,
    }


def evaluate_acceptance(
    unit: Dict,
    integration_results: List[Dict],
) -> Dict:
    """Evaluate pre-registered acceptance checks C1-C9."""
    # Aggregate integration results by arm.
    def agg(arm_name: str, field: str, agg_fn=max):
        vals = [r[field] for r in integration_results if r["arm"] == arm_name]
        return agg_fn(vals) if vals else 0

    # Unit checks
    c1 = unit["C1_prototype_grows_on_events"]
    c2 = unit["C2_no_growth_without_events"]
    c3 = unit["C3_simmode_gate"]
    c4 = unit["C4_min_norm_gate"]
    c5 = unit["C5_same_vector_high_signal"]

    # Integration C6: ARM_A peak signal > 0
    arm_a_peak = agg("ARM_A_both_on", "peak_safety_signal")
    c6 = arm_a_peak > 0.0

    # Integration C7: ARM_B final proto_norm stays near zero
    arm_b_proto = agg("ARM_B_store_on_no_mech302", "final_proto_norm")
    c7 = arm_b_proto < C7_MAX_PROTO_NORM

    # Integration C8: ARM_C and ARM_D store is None
    arm_c_none = all(r["store_is_none"] for r in integration_results
                     if r["arm"] in ("ARM_C_store_off_mech302_on", "ARM_D_both_off"))
    c8 = arm_c_none

    # Integration C9: no exceptions
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
        "C7_arm_b_proto_norm_below_cap": bool(c7),
        "C7_arm_b_proto_norm": float(arm_b_proto),
        "C7_max_allowed": C7_MAX_PROTO_NORM,
        "C8_store_none_when_disabled": bool(c8),
        "C9_no_exceptions": bool(c9),
        "C9_total_exceptions": int(total_exceptions),
        "all_pass": bool(overall),
    }


def main(dry_run: bool = False) -> Tuple[int, str, str]:
    print(f"[{EXPERIMENT_TYPE}] starting (dry_run={dry_run})")
    print(
        f"curriculum: interval={SCHED_INTERVAL} prob={SCHED_PROB} "
        f"magnitude={SCHED_MAGNITUDE} limb_selection={SCHED_LIMB_SELECTION}",
        flush=True,
    )
    seeds = (SEEDS[0],) if dry_run else SEEDS
    n_ticks = 60 if dry_run else N_TICKS_INTEGRATION
    t0 = time.time()

    # --- Part 1: Unit tests ---
    print(f"[{EXPERIMENT_TYPE}] Part 1: unit tests")
    unit = run_unit_tests()
    for k, v in unit.items():
        print(f"  {k}: {v}")

    # --- Part 2: Integration ---
    print(f"[{EXPERIMENT_TYPE}] Part 2: integration smoke ({n_ticks} ticks x {len(seeds)} seeds)")
    integration_results: List[Dict] = []
    total_arm_seed_units = len(seeds) * len(ARMS)
    completed = 0
    for seed in seeds:
        for arm_name, use_store, use_mech302 in ARMS:
            print(f"Seed {seed} Condition {arm_name}", flush=True)
            r = run_integration_arm(seed, arm_name, use_store, use_mech302, n_ticks)
            integration_results.append(r)
            completed += 1
            print(
                f"  [train] seed={seed} arm={arm_name:<34} ep {completed}/{total_arm_seed_units} "
                f"relief_events={r['relief_events']:3d} "
                f"nonzero_sig={r['nonzero_safety_signals']:3d} "
                f"peak_sig={r['peak_safety_signal']:.4f} "
                f"proto_norm={r['final_proto_norm']:.4f} "
                f"sched_inj={r['scheduled_injections']:3d} "
                f"exc={r['exception_count']}",
                flush=True,
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
        return 0, outcome, ""

    timestamp_utc = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{timestamp_utc}_v3"
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
        "supersedes": SUPERSEDES,
        "timestamp_utc": timestamp_utc,
        "started_utc": datetime.utcnow().isoformat() + "Z",
        "outcome": outcome,
        "evidence_direction": "supports" if outcome == "PASS" else "weakens",
        "elapsed_seconds": elapsed,
        "n_seeds_unit": 1,
        "n_seeds_integration": len(seeds),
        "n_ticks_integration": n_ticks,
        "grid_size": GRID_SIZE,
        "num_hazards": NUM_HAZARDS,
        "num_resources": NUM_RESOURCES,
        "contaminated_harm": CONTAMINATED_HARM,
        "damage_increment": DAMAGE_INCREMENT,
        "failure_prob_scale": FAILURE_PROB_SCALE,
        "heal_rate": HEAL_RATE,
        "harm_ema_alpha": HARM_EMA_ALPHA,
        "ema_alpha_unit": EMA_ALPHA_UNIT,
        "min_norm": MIN_NORM,
        "threshold": THRESHOLD,
        "scheduled_limb_damage_enabled": True,
        "scheduled_limb_damage_interval": SCHED_INTERVAL,
        "scheduled_limb_damage_prob": SCHED_PROB,
        "scheduled_limb_damage_magnitude": SCHED_MAGNITUDE,
        "scheduled_limb_damage_limb_selection": SCHED_LIMB_SELECTION,
        "unit_results": unit,
        "integration_results": integration_results,
        "acceptance": acceptance,
        "notes": (
            "V3-EXQ-519b supersedes V3-EXQ-519 (2026-05-04 FAIL: relief_events=0 "
            "across all integration arms because the default env had no source "
            "of damage->heal trajectories MECH-302 could detect). Today (2026-05-30) "
            "the SD-022 scheduled-injection extension landed and V3-EXQ-517c PASSed "
            "validating MECH-302 substrate AND lifting gate (c) for V3-EXQ-519 "
            "retest. 519b adds the scheduled_limb_damage_* env curriculum kwargs "
            "to ALL integration arms (identical across ARMs so the (store, "
            "MECH-302) factors stay the only varying axes) so MECH-302 events "
            "fire reliably in ARM_A. Acceptance criteria (C1-C9) unchanged from "
            "V3-EXQ-519. Curriculum mirrors EXQ-517c: interval=50, prob=0.5, "
            "magnitude=0.4, limb_selection=random. PASS clears MECH-304 v3_pending "
            "gate."
        ),
    }
    out_path = write_flat_manifest(
        manifest,
        out_dir,
        dry_run=False,
        config=manifest.get("config"),
        seeds=SEEDS,
        script_path=Path(__file__),
    )
    print(f"Result written to: {out_path}")
    print(f"Done. Outcome: {outcome}")
    return 0, outcome, str(out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Smoke run, no manifest.")
    args = parser.parse_args()
    _rc, _outcome, _manifest_path = main(dry_run=args.dry_run)
    if not args.dry_run:
        emit_outcome(
            outcome=_outcome if _outcome in ("PASS", "FAIL") else "FAIL",
            manifest_path=_manifest_path,
        )
    sys.exit(_rc)
