#!/opt/local/bin/python3
"""
V3-EXQ-498 -- OCD Layer 1 hypothesis: closure-threshold sweep on monostrategy.

Claims: SD-034

Purpose (diagnostic)
--------------------
The OCD three-layer architectural model registered in
REE_assembly/docs/architecture/psychiatric_failure_modes.md (2026-04-28
"OCD: A Three-Layer Architectural Failure" section) proposes that V3
monostrategy may be partly explained by a Layer 1 failure -- the
SD-034 closure-domain predicate threshold being set such that the
agent's rule-state never crosses the closure threshold, so the
current behavioural strategy is never released. The architectural
prediction: if Layer 1 contributes, then sweeping the SD-034
closure_rule_delta_threshold parameter should produce a continuous
gradient from monostrategy (tight threshold; goal never releases) to
multi-strategy (loose threshold; closure fires readily, strategy
explores).

This experiment is the cheapest diagnostic in the OCD-derived
monostrategy hypothesis set:

  Layer 1 (this EXQ): closure-threshold parameter sweep, ~30 min on Mac.
  Layer 2 (future): MECH-290 backward credit sweep ablation.
  Layer 3 (future): SD-046 multi-slot GoalState pull-forward.

The diagnostic logic:

  PASS = closure-threshold tuning ALONE attenuates baseline monostrategy.
         Layer 1 is a contributor; further SD-034 thresholding work
         is fruitful before pulling SD-045 / SD-046 forward.

  FAIL = closure-threshold is monotonically invariant on monostrategy
         across the swept range. Layer 1 is ruled OUT as the
         dominant monostrategy mechanism. The diagnostic-positive
         outcome is to escalate to Layer 2 (MECH-290 ablation) or
         Layer 3 (SD-046 pull-forward).

Architecture
------------
Base agent: SD-034 substrate-active configuration (closure operator
ENABLED, lateral PFC analogue ENABLED, dACC ENABLED). All V_s rollout
gating OFF, all abstraction substrates OFF, all sleep substrates OFF.
This isolates SD-034 closure threshold as the single manipulated
variable and excludes noise from the V_s pathway (which is itself
under separate test in EXQ-490 / EXQ-490b).

Manipulated variable: closure_rule_delta_threshold across four levels.
Held: closure_stable_ticks=3 (default), all other SD-034 sub-knobs
at default.

Environment: CausalGridWorldV2 mirroring EXQ-482 baseline-monostrategy
diagnostic, with SD-029 scheduled-external-hazard curriculum ON to
provide event-distribution variability.

Arms
----
    TIGHT       : closure_rule_delta_threshold = 0.0001
    DEFAULT     : closure_rule_delta_threshold = 0.001
    LOOSE       : closure_rule_delta_threshold = 0.005
    VERY_LOOSE  : closure_rule_delta_threshold = 0.05

Metrics
-------
    action_class_entropy: Shannon entropy over executed action-class
        histogram across the run. Primary monostrategy metric (mirrors
        EXQ-482).
    n_closure_fires: cumulative closure_operator fire count per arm
        (sanity check that the parameter is having a behavioural
        effect at the substrate level).
    action_class_counts: per-class executions (sanity).

Pass / fail rule
----------------
    PASS = at AT LEAST ONE of {LOOSE, VERY_LOOSE},
           action_class_entropy > entropy(DEFAULT) + 0.1
           in >= 2/3 seeds.

    FAIL = no entropy lift > 0.1 at any non-default threshold in the
           swept range. Layer 1 ruled out as a contributor; escalate
           to Layer 2 / Layer 3 diagnostics.

experiment_purpose=diagnostic. Does NOT directly test SD-034
correctness (the substrate is well-validated per EXQ-460 / EXQ-466);
tests whether SD-034 PARAMETER SWEEP attenuates monostrategy. PASS
informs SD-034 default-threshold tightening; FAIL informs
substrate-pull-forward priorities for SD-045 / SD-046.

See REE_assembly/docs/architecture/psychiatric_failure_modes.md
"OCD: A Three-Layer Architectural Failure" section.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig


EXPERIMENT_TYPE = "v3_exq_498_ocd_layer1_closure_threshold_sweep"
CLAIM_IDS = ["SD-034"]
EXPERIMENT_PURPOSE = "diagnostic"

SEEDS = [42, 7, 13]
EPISODES = 6
STEPS_PER_EP = 200
ENTROPY_LIFT_MIN = 0.1

ARMS: List[Dict] = [
    {"name": "TIGHT",      "closure_rule_delta_threshold": 0.0001},
    {"name": "DEFAULT",    "closure_rule_delta_threshold": 0.001},
    {"name": "LOOSE",      "closure_rule_delta_threshold": 0.005},
    {"name": "VERY_LOOSE", "closure_rule_delta_threshold": 0.05},
]

CURRICULUM_INTERVAL = 50
CURRICULUM_PROB = 0.5
CURRICULUM_ADJACENT_ONLY = True


def _make_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(
        seed=seed,
        size=10,
        num_hazards=2,
        num_resources=3,
        hazard_harm=0.04,
        proximity_harm_scale=0.12,
        proximity_benefit_scale=0.18,
        hazard_field_decay=0.5,
        energy_decay=0.005,
        use_proxy_fields=True,
        resource_respawn_on_consume=True,
        harm_history_len=10,
        scheduled_external_hazard_enabled=True,
        scheduled_external_hazard_interval=CURRICULUM_INTERVAL,
        scheduled_external_hazard_prob=CURRICULUM_PROB,
        scheduled_external_hazard_adjacent_only=CURRICULUM_ADJACENT_ONLY,
    )


def _make_agent(env: CausalGridWorldV2, closure_threshold: float) -> REEAgent:
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=32,
        world_dim=32,
        alpha_world=0.9,
        alpha_self=0.3,
        use_harm_stream=True,
        harm_obs_dim=51,
        use_affective_harm_stream=True,
        harm_obs_a_dim=50,
        harm_history_len=10,
        z_harm_a_dim=16,
        # SD-034 substrate active. Prerequisites: lateral PFC + dACC.
        use_lateral_pfc_analog=True,
        use_dacc=True,
        use_closure_operator=True,
        # Manipulated variable.
        closure_rule_delta_threshold=closure_threshold,
        # Held at defaults.
        closure_stable_ticks=3,
    )
    return REEAgent(cfg)


def _shannon_entropy(counts: Dict[int, int]) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    ent = 0.0
    for c in counts.values():
        if c <= 0:
            continue
        p = c / total
        ent -= p * math.log(p)
    return ent


def _read_closure_fires(agent: REEAgent) -> int:
    """Best-effort read of closure-operator fire count from the agent."""
    op = getattr(agent, "closure_operator", None)
    if op is None:
        return 0
    for attr in ("n_fires", "_n_fires", "fire_count", "_fire_count"):
        if hasattr(op, attr):
            try:
                return int(getattr(op, attr))
            except Exception:
                continue
    return 0


def _run_arm(seed: int, arm: Dict) -> Dict:
    torch.manual_seed(seed)
    env = _make_env(seed)
    agent = _make_agent(env, closure_threshold=arm["closure_rule_delta_threshold"])

    action_counts: Dict[int, int] = {}
    n_ticks = 0
    n_external_hazard_events = 0

    for _ep in range(EPISODES):
        obs, _info = env.reset()
        for _step in range(STEPS_PER_EP):
            action = agent.act(obs)
            a_idx = int(action[0].argmax().item())
            action_counts[a_idx] = action_counts.get(a_idx, 0) + 1
            obs, _harm, done, info, _obs_dict = env.step(a_idx)
            n_ticks += 1
            if info.get("external_hazard_injected", False):
                n_external_hazard_events += 1
            if done:
                break

    n_closure_fires = _read_closure_fires(agent)

    return {
        "arm": arm["name"],
        "closure_rule_delta_threshold": arm["closure_rule_delta_threshold"],
        "seed": seed,
        "action_class_entropy": _shannon_entropy(action_counts),
        "action_class_counts": action_counts,
        "n_actions": sum(action_counts.values()),
        "n_ticks": n_ticks,
        "n_closure_fires": n_closure_fires,
        "n_external_hazard_events": n_external_hazard_events,
    }


def _print_plan() -> None:
    print(f"{EXPERIMENT_TYPE} -- OCD Layer 1 closure-threshold sweep",
          flush=True)
    print(f"Arms: {[a['name'] for a in ARMS]} (closure_rule_delta_threshold "
          f"= {[a['closure_rule_delta_threshold'] for a in ARMS]})",
          flush=True)
    print(f"Seeds: {SEEDS}", flush=True)
    print(f"Episodes x steps_per_ep: {EPISODES} x {STEPS_PER_EP}", flush=True)
    print("Metrics: action_class_entropy (primary), n_closure_fires, "
          "action_class_counts", flush=True)
    print("PASS = at AT LEAST ONE of {LOOSE, VERY_LOOSE}, "
          f"entropy(arm) > entropy(DEFAULT) + {ENTROPY_LIFT_MIN} "
          f"in >= 2/{len(SEEDS)} seeds", flush=True)
    print(f"experiment_purpose={EXPERIMENT_PURPOSE}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=f"{EXPERIMENT_TYPE} OCD Layer 1 closure-threshold sweep"
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Print plan and exit 0; do not execute.")
    args = parser.parse_args()

    if args.dry_run:
        _print_plan()
        print("DRY RUN OK", flush=True)
        return 0

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{ts}_v3"
    out_dir = (
        Path(__file__).resolve().parents[2]
        / "REE_assembly" / "evidence" / "experiments" / EXPERIMENT_TYPE
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results: List[Dict] = []
    for seed in SEEDS:
        for arm in ARMS:
            print(f"Seed {seed} Arm {arm['name']} "
                  f"(threshold={arm['closure_rule_delta_threshold']})",
                  flush=True)
            r = _run_arm(seed=seed, arm=arm)
            print(f"  -> entropy={r['action_class_entropy']:.4f} "
                  f"closure_fires={r['n_closure_fires']} "
                  f"counts={r['action_class_counts']}", flush=True)
            all_results.append(r)

    # Pass criterion: at AT LEAST ONE of {LOOSE, VERY_LOOSE},
    # entropy(arm) > entropy(DEFAULT) + ENTROPY_LIFT_MIN in >= 2/N seeds.
    by_arm: Dict[str, Dict[int, Dict]] = {a["name"]: {} for a in ARMS}
    for r in all_results:
        by_arm[r["arm"]][r["seed"]] = r

    test_arms = ["LOOSE", "VERY_LOOSE"]
    arm_lifts: List[Dict] = []
    arm_passes: List[bool] = []
    for arm_name in test_arms:
        per_seed = []
        seeds_passing = 0
        for seed in SEEDS:
            default_ent = by_arm["DEFAULT"][seed]["action_class_entropy"]
            arm_ent = by_arm[arm_name][seed]["action_class_entropy"]
            lift = arm_ent - default_ent
            ok = lift > ENTROPY_LIFT_MIN
            per_seed.append({
                "seed": seed,
                "default_entropy": default_ent,
                "arm_entropy": arm_ent,
                "lift": lift,
                "passed": ok,
            })
            if ok:
                seeds_passing += 1
        arm_lifts.append({
            "arm": arm_name,
            "per_seed": per_seed,
            "seeds_passing": seeds_passing,
            "seeds_required": 2,
            "arm_passed": seeds_passing >= 2,
        })
        arm_passes.append(seeds_passing >= 2)

    outcome = "PASS" if any(arm_passes) else "FAIL"

    summary = {
        "gate_rule": (
            f"at AT LEAST ONE of {test_arms}, entropy(arm) > "
            f"entropy(DEFAULT) + {ENTROPY_LIFT_MIN} in >= 2/{len(SEEDS)} seeds"
        ),
        "arm_lifts": arm_lifts,
        "any_arm_passed": any(arm_passes),
        "pass": outcome == "PASS",
    }

    print(f"\nOutcome: {outcome}", flush=True)
    for arm_lift in arm_lifts:
        print(f"  Arm {arm_lift['arm']}: "
              f"seeds_passing={arm_lift['seeds_passing']}/{len(SEEDS)} "
              f"arm_passed={arm_lift['arm_passed']}", flush=True)
        for row in arm_lift["per_seed"]:
            print(f"    seed={row['seed']} "
                  f"default_ent={row['default_entropy']:.4f} "
                  f"arm_ent={row['arm_entropy']:.4f} "
                  f"lift={row['lift']:.4f} "
                  f"passed={row['passed']}", flush=True)

    per_claim = {
        cid: ("supports" if outcome == "PASS" else "non_contributory")
        for cid in CLAIM_IDS
    }

    output = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": ts,
        "outcome": outcome,
        "evidence_direction": (
            "supports" if outcome == "PASS" else "non_contributory"
        ),
        "evidence_direction_per_claim": per_claim,
        "evidence_direction_note": (
            "OCD Layer 1 closure-threshold-sweep diagnostic. PASS supports "
            "the architectural reading that monostrategy in V3 is partly "
            "explained by SD-034 closure-domain-predicate over-tightening "
            "and is parameter-tuneable; further SD-034 thresholding work "
            "is fruitful before pulling SD-045 / SD-046 forward. FAIL is "
            "non_contributory for SD-034 (the substrate is unchanged and "
            "validated by EXQ-460/466) and diagnostically positive for "
            "escalating to Layer 2 (MECH-290 ablation) or Layer 3 "
            "(SD-046 pull-forward). NOT governance evidence on SD-034 "
            "correctness."
        ),
        "pass_criteria_summary": summary,
        "per_arm_seed_results": all_results,
        "config": {
            "arms": ARMS,
            "seeds": SEEDS,
            "episodes": EPISODES,
            "steps_per_ep": STEPS_PER_EP,
            "entropy_lift_min": ENTROPY_LIFT_MIN,
            "curriculum_interval": CURRICULUM_INTERVAL,
            "curriculum_prob": CURRICULUM_PROB,
            "curriculum_adjacent_only": CURRICULUM_ADJACENT_ONLY,
        },
    }

    out_file = out_dir / f"{run_id}.json"
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Result written to: {out_file}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
