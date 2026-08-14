"""
V3-EXQ-929 -- sleep_substrate:GAP-9 within-life sleep-trigger validation

Claims: None (diagnostic substrate-readiness validation; does not weight governance)

EXPERIMENT_PURPOSE = "diagnostic"

SLEEP DRIVER: within-life-step-trigger (SleepLoopManager.notify_waking_step fires a sleep
cycle from REEAgent.update_residue once within_life_sleep_step_ceiling WAKING steps have
elapsed since the last cycle; notify_episode_end() / the K-episode cadence is NOT used --
the driver never calls agent.reset() within the single continuous life, so the ONLY path to
a sleep cycle is the new within-life trigger). This is the mechanism under validation.

WHY THIS RUN. Routing source: REE_assembly/evidence/planning/sleep_substrate_plan.md GAP-9
(registered 2026-08-12; design brief 2026-08-14 lit synthesis
targeted_review_sleep_onset_multiinput_gap9). Before the GAP-9 build, REE's sleep trigger was
BOUNDARY-only: SleepLoopManager.notify_episode_end() (the sole K-episode-cadence entry) is
reachable only across an inter-episode boundary (REEAgent.reset()), so a TRUE single-
continuous life (num_episodes=1) could never sleep -- no MECH-204 recalibration, no Phase B-E
aggregation, no GAP-5b duration scaling could ever fire within such a life, independent of
any cadence config. The GAP-9 build adds a within-life trigger behind a default-off flag
(use_within_life_sleep_trigger). This run validates it: with the flag ON in a true single
continuous life, sleep fires; with it OFF, the same life never sleeps (the GAP-9 wall,
reproduced as the negative control).

v1 SCOPE. The build wires the (a)+(b)-composed design's STEP-COUNT CEILING arm (design (a),
the anti-starvation backstop) only; the MEL/learning-demand need-crossing arm (design (b),
the primary trigger) is a planned follow-up. So this run validates that a within-life sleep
cycle FIRES at all in a continuous life, and that the fired cycles carry the ceiling arm --
NOT that a demand-sensitive trigger works (there is no demand arm yet). The arm-attribution
diagnostic (within_life_trigger_arm_ceiling) is emitted so a future need-arm run is a drop-in
comparison and a ceiling-only fire is never mistaken for a demand-sensitive one (the
V3-EXQ-718a failure mode one level up).

WHAT THIS RUN IS NOT: a claim test, a behavioural/learning experiment, or a test of sleep's
downstream EFFECT. The agent is intentionally UNTRAINED -- the DV is the STRUCTURAL
reachability of the sleep cycle from within a continuous life (does the trigger reach
_run_cycle at all), which is independent of what the agent has learned. The env is a plain
CausalGridWorld stepped with a uniform-random policy purely to accumulate genuine waking
update_residue() steps; nothing about the result depends on the policy being good.

Design (2 conditions x N seeds, single continuous life each):
  OFF  -- use_within_life_sleep_trigger=False -> within-life sleep cycles fired == 0
  ON   -- use_within_life_sleep_trigger=True  -> within-life sleep cycles fired >= 1
Both conditions: use_sleep_loop=True, sws_enabled=True, rem_enabled=True,
sleep_loop_episodes_K huge (boundary path unreachable, and the driver never resets the agent
anyway). use_mech286_sleep_onset_gate is left OFF (default) per the lit-synthesis brief: its
threat term reads a signal V3-EXQ-917 measured at chance-level place-safety discrimination and
enabling it would confound the first true-single-life sleep result.

Pre-registered acceptance (NOT derived from this run's statistics):
  C1  OFF cycles fired == 0 for every seed   (the GAP-9 wall, reproduced)
  C2  ON  cycles fired >= 1 for every seed   (sleep now fires within a continuous life)
  C3  ON  every fired cycle has within_life_trigger_arm_ceiling == 1.0 (v1 ceiling-only)
  PASS iff C1 and C2 and C3, AND the readiness preconditions hold (the life exceeded the
  ceiling so a fire was possible, and the trigger was actually wired on the ON arm).

Output:
  evidence/experiments/v3_exq_929_sleep_gap9_within_life_trigger/
    v3_exq_929_sleep_gap9_within_life_trigger_<ts>.json   (manifest)
"""

import random
from pathlib import Path
from typing import Any, Dict, List

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig
from experiment_protocol import emit_outcome
from experiments.pack_writer import write_flat_manifest
from experiments._lib.z_goal_stream import ZGoalStreamAccumulator

# z_goal is orthogonal to this run's DV (structural sleep-cycle reachability), and the
# minimal driver never calls update_z_goal -- but the run steps an agent, so record the
# stream's liveness for completeness (generous-recording standard) rather than leaving it
# unmeasured.
_ZG = ZGoalStreamAccumulator()


EXPERIMENT_TYPE    = "v3_exq_929_sleep_gap9_within_life_trigger"
EXPERIMENT_PURPOSE = "diagnostic"
CLAIM_IDS: List[str] = []
SLEEP_DRIVER_PATTERN = (
    "within-life-step-trigger (SleepLoopManager.notify_waking_step via update_residue; "
    "ceiling arm; notify_episode_end NOT used -- agent never reset within the life)"
)

# validate_experiments.py anchor-reachability advisory: this script's three readiness
# preconditions are DIRECT STRUCTURAL / CONFIG comparisons, not hand-written predicates
# scored against a known-degenerate reference sample (the V3-EXQ-778d failure mode the
# guard exists for): life_exceeds_step_ceiling compares min waking steps (LIFE_STEPS=120,
# fixed) against STEP_CEILING (25, fixed); on_arm_trigger_wired / off_arm_trigger_not_wired
# read the boolean agent.sleep_loop.within_life_trigger, which is set deterministically from
# use_within_life_sleep_trigger at agent construction. All three are reachable by
# construction (120 > 25; the ON arm sets the flag, the OFF arm leaves it default False) and
# were confirmed reachable by this script's own dry-run smoke -- there is no scored predicate
# narrower than the state it anchors to for a reachability check to guard against.
ANCHOR_REACHABILITY_EXEMPT = (
    "structural/config comparisons (min waking steps vs fixed STEP_CEILING; the "
    "agent.sleep_loop.within_life_trigger boolean set from config), not scored predicates "
    "against a reference sample; reachable by construction (LIFE_STEPS=120 > STEP_CEILING=25; "
    "ON sets the flag, OFF leaves it default False), confirmed in this script's own dry-run"
)

# ---- run geometry ----
DEFAULT_SEEDS   = [0, 1, 2]  # seed 44 excluded per CLAUDE.md reef-config instability precedent
LIFE_STEPS      = 120        # waking steps in ONE continuous life (denominator of `ep N/M`)
STEP_CEILING    = 25         # within_life_sleep_step_ceiling (ON arm) -> ~4 fires over the life
GRID_SIZE       = 8
NUM_HAZARDS     = 2
NUM_RESOURCES   = 3

# ---- pre-registered thresholds (NOT derived from this run) ----
OFF_MAX_FIRES        = 0     # C1: OFF must fire exactly zero
ON_MIN_FIRES         = 1     # C2: ON must fire at least once, every seed
CEILING_ARM_FRAC_MIN = 1.0   # C3: v1 is ceiling-only, so every ON fire is the ceiling arm


def _build_config(env, within_life: bool) -> "REEConfig":
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        use_sleep_loop=True,
        # Huge K so the boundary path is unreachable even if a reset ever happened; the
        # driver additionally never calls agent.reset() within the life.
        sleep_loop_episodes_K=10_000_000,
        use_within_life_sleep_trigger=within_life,
        within_life_sleep_step_ceiling=STEP_CEILING,
    )
    cfg.sws_enabled = True
    cfg.rem_enabled = True
    # use_mech286_sleep_onset_gate left at its default False (lit-synthesis brief).
    return cfg


def _run_continuous_life(seed: int, within_life: bool) -> Dict[str, Any]:
    """One TRUE single continuous agent life: sense + random action + env.step +
    update_residue per WAKING step, NEVER agent.reset(). Counts the sleep cycles fired
    within the life (only reachable via the within-life trigger) and the OR-arm split."""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    env = CausalGridWorldV2(
        size=GRID_SIZE, num_hazards=NUM_HAZARDS, num_resources=NUM_RESOURCES,
        max_episode_steps=LIFE_STEPS + 10, seed=seed,
    )
    _flat, obs = env.reset()
    agent = REEAgent(_build_config(env, within_life)).to(torch.device("cpu"))
    agent.eval()

    arm_label = "ON" if within_life else "OFF"
    print(f"Seed {seed} Condition {arm_label}", flush=True)

    waking_steps = 0
    for step_idx in range(LIFE_STEPS):
        obs_body = obs["body_state"]
        obs_world = obs["world_state"]
        with torch.no_grad():
            agent.sense(obs_body, obs_world)
            action = torch.zeros(1, env.action_dim)
            action[0, random.randint(0, env.action_dim - 1)] = 1.0
            agent._last_action = action
            _flat, harm_signal, done, _info, obs = env.step(action)
            agent.update_residue(float(harm_signal))
        waking_steps += 1
        if (step_idx + 1) % 20 == 0:
            print(f"  [train] gap9 seed={seed} arm={arm_label} ep {step_idx + 1}/{LIFE_STEPS} "
                  f"fires={len(agent.sleep_loop.cycle_history)}", flush=True)
        if done:
            # Env-level respawn only. The AGENT is NOT reset -> no episode boundary, so the
            # within-life trigger stays the sole path to a sleep cycle.
            _flat, obs = env.reset()

    _ZG.observe(agent)
    history = agent.sleep_loop.cycle_history
    n_fires = len(history)
    n_ceiling = sum(1 for h in history if float(h.get("within_life_trigger_arm_ceiling", 0.0)) == 1.0)
    n_need = sum(1 for h in history if float(h.get("within_life_trigger_arm_need", 0.0)) == 1.0)
    ceiling_frac = (n_ceiling / n_fires) if n_fires else 0.0

    seed_pass = (
        (n_fires >= ON_MIN_FIRES and ceiling_frac >= CEILING_ARM_FRAC_MIN) if within_life
        else (n_fires <= OFF_MAX_FIRES)
    )
    print(f"verdict: {'PASS' if seed_pass else 'FAIL'} seed={seed} arm={arm_label} "
          f"fires={n_fires} ceiling_frac={ceiling_frac:.2f}", flush=True)

    return {
        "seed": seed,
        "arm": arm_label,
        "within_life_trigger_enabled": bool(within_life),
        "trigger_wired": bool(agent.sleep_loop.within_life_trigger),
        "step_ceiling": int(agent.sleep_loop.within_life_step_ceiling),
        "waking_steps": int(waking_steps),
        "sleep_cycles_fired": int(n_fires),
        "fires_arm_ceiling": int(n_ceiling),
        "fires_arm_need": int(n_need),
        "ceiling_arm_fraction": float(ceiling_frac),
        "sws_enabled": True,
        "rem_enabled": True,
    }


def run(seeds=None, dry_run: bool = False) -> dict:
    if seeds is None:
        seeds = DEFAULT_SEEDS
    if dry_run:
        seeds = seeds[:1]
    print(f"[V3-EXQ-929] GAP-9 within-life sleep-trigger validation\n"
          f"  Seeds: {seeds}  Life: {LIFE_STEPS} waking steps  Ceiling: {STEP_CEILING}\n"
          f"  Output: REE_assembly/evidence/experiments/{EXPERIMENT_TYPE}/", flush=True)

    off_results = [_run_continuous_life(s, within_life=False) for s in seeds]
    on_results = [_run_continuous_life(s, within_life=True) for s in seeds]
    per_condition = off_results + on_results

    off_fires = [r["sleep_cycles_fired"] for r in off_results]
    on_fires = [r["sleep_cycles_fired"] for r in on_results]
    on_ceiling_fracs = [r["ceiling_arm_fraction"] for r in on_results]
    on_trigger_wired = all(r["trigger_wired"] for r in on_results)
    off_trigger_wired = any(r["trigger_wired"] for r in off_results)  # must be False
    min_waking_steps = min(r["waking_steps"] for r in per_condition)

    # --- readiness preconditions (the DV is only meaningful if these hold) ---
    life_exceeds_ceiling = bool(min_waking_steps >= STEP_CEILING)
    on_arm_wired = bool(on_trigger_wired)
    off_arm_not_wired = bool(not off_trigger_wired)
    apparatus_ready = bool(life_exceeds_ceiling and on_arm_wired and off_arm_not_wired)

    # --- pre-registered acceptance criteria ---
    c1_off_silent = bool(all(f <= OFF_MAX_FIRES for f in off_fires))
    c2_on_fires = bool(all(f >= ON_MIN_FIRES for f in on_fires))
    c3_on_ceiling_arm = bool(all(cf >= CEILING_ARM_FRAC_MIN for cf in on_ceiling_fracs))

    if not apparatus_ready:
        label = "substrate_not_ready_requeue"
        passed = False
    elif c1_off_silent and c2_on_fires and c3_on_ceiling_arm:
        label = "within_life_trigger_validated"
        passed = True
    else:
        label = "within_life_trigger_defect"
        passed = False
    outcome = "PASS" if passed else "FAIL"

    metrics: Dict[str, Any] = {
        "n_seeds": float(len(seeds)),
        "off_cycles_fired_max": float(max(off_fires)) if off_fires else 0.0,
        "off_cycles_fired_total": float(sum(off_fires)),
        "on_cycles_fired_min": float(min(on_fires)) if on_fires else 0.0,
        "on_cycles_fired_mean": float(sum(on_fires) / len(on_fires)) if on_fires else 0.0,
        "on_cycles_fired_total": float(sum(on_fires)),
        "on_ceiling_arm_fraction_min": float(min(on_ceiling_fracs)) if on_ceiling_fracs else 0.0,
        "on_need_arm_fires_total": float(sum(r["fires_arm_need"] for r in on_results)),
        "min_waking_steps": float(min_waking_steps),
        "step_ceiling": float(STEP_CEILING),
        "c1_off_silent": 1.0 if c1_off_silent else 0.0,
        "c2_on_fires": 1.0 if c2_on_fires else 0.0,
        "c3_on_ceiling_arm": 1.0 if c3_on_ceiling_arm else 0.0,
    }
    for r in per_condition:
        key = f"seed{r['seed']}_{r['arm']}_fires"
        metrics[key] = float(r["sleep_cycles_fired"])

    interpretation = {
        "label": label,
        "preconditions": [
            {"name": "life_exceeds_step_ceiling",
             "description": "the continuous life ran at least within_life_sleep_step_ceiling "
                            "waking steps, so a within-life fire was structurally possible; a "
                            "shorter life could silently make the ON arm look non-firing",
             "measured": float(min_waking_steps), "threshold": float(STEP_CEILING),
             "direction": "lower", "met": life_exceeds_ceiling,
             "control": "min waking steps across all conditions/seeds"},
            {"name": "on_arm_trigger_wired",
             "description": "the ON arm's SleepLoopManager actually has within_life_trigger set "
                            "from config -- guards against a from_dims silent-kwargs miswire "
                            "reading as a non-firing substrate rather than an unwired flag",
             "measured": 1.0 if on_arm_wired else 0.0, "threshold": 1.0,
             "direction": "lower", "met": on_arm_wired, "control": "all ON seeds"},
            {"name": "off_arm_trigger_not_wired",
             "description": "the OFF arm's within_life_trigger is False (the negative control is "
                            "genuinely the flag-off substrate, not an accidentally-on one)",
             "measured": 0.0 if off_arm_not_wired else 1.0, "threshold": 0.0,
             "direction": "upper", "met": off_arm_not_wired, "control": "all OFF seeds"},
        ],
        "criteria_non_degenerate": {
            # The DV genuinely discriminates iff ON fires while OFF does not -- if both were
            # zero (or both nonzero) the manipulation did nothing. C1 and C2 together are that
            # non-degeneracy: they are False exactly when the arms failed to separate.
            "c1_off_silent": c1_off_silent,
            "c2_on_fires": c2_on_fires,
            "c3_on_ceiling_arm": c3_on_ceiling_arm,
        },
        "criteria": [
            {"name": "c1_off_silent", "load_bearing": True, "passed": c1_off_silent},
            {"name": "c2_on_fires", "load_bearing": True, "passed": c2_on_fires},
            {"name": "c3_on_ceiling_arm", "load_bearing": True, "passed": c3_on_ceiling_arm},
        ],
        "combination_rule": "PASS = readiness AND c1_off_silent AND c2_on_fires AND c3_on_ceiling_arm",
        "dv_symmetry_note": (
            "The manipulation (use_within_life_sleep_trigger) directly gates the DV path "
            "(notify_waking_step -> _run_cycle); the cycle-fired count is NOT invariant under "
            "the flag, so this is a genuine measurement, not an arithmetic identity."
        ),
        "note": (
            "Diagnostic substrate-readiness validation for sleep_substrate:GAP-9 (v1 ceiling "
            "arm). Untrained agent by design: the DV is structural reachability of a sleep "
            "cycle from within a single continuous life, independent of learning. claim_ids=[]; "
            "does not weight governance. On PASS, GAP-9's status moves to done (ceiling arm); "
            "the MEL/need-crossing arm (design (b)) remains a planned follow-up. "
            "use_mech286_sleep_onset_gate deliberately OFF (lit-synthesis brief)."
        ),
    }

    summary_markdown = f"""# V3-EXQ-929 -- GAP-9 within-life sleep-trigger validation

**Status:** {outcome} -- label: `{label}`
**Purpose:** diagnostic substrate-readiness validation (sleep_substrate:GAP-9, v1 ceiling arm).

Before GAP-9, REE's sleep trigger was boundary-only, so a TRUE single-continuous life
(num_episodes=1) could never sleep. This run validates the within-life trigger.

- seeds: {seeds}
- life length: {LIFE_STEPS} waking steps; step ceiling: {STEP_CEILING}
- OFF cycles fired (max across seeds): {int(max(off_fires)) if off_fires else 0}  (target {OFF_MAX_FIRES})
- ON cycles fired (min across seeds): {int(min(on_fires)) if on_fires else 0}  (target >= {ON_MIN_FIRES})
- ON ceiling-arm fraction (min): {min(on_ceiling_fracs) if on_ceiling_fracs else 0.0:.2f}  (target {CEILING_ARM_FRAC_MIN})
- C1 OFF silent: {c1_off_silent} | C2 ON fires: {c2_on_fires} | C3 ceiling arm: {c3_on_ceiling_arm}

See `interpretation` for the pre-registered acceptance rule and per-condition preconditions,
and `per_condition_results` for the full per-seed x arm table.
"""

    return {
        "status": outcome,
        "outcome": outcome,
        "metrics": metrics,
        "summary_markdown": summary_markdown,
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "evidence_direction": "non_contributory",
        "experiment_type": EXPERIMENT_TYPE,
        "sleep_driver_pattern": SLEEP_DRIVER_PATTERN,
        "interpretation": interpretation,
        "per_condition_results": per_condition,
        "config": {
            "life_steps": LIFE_STEPS, "step_ceiling": STEP_CEILING, "grid_size": GRID_SIZE,
            "num_hazards": NUM_HAZARDS, "num_resources": NUM_RESOURCES,
            "sleep_loop_episodes_K": 10_000_000, "sws_enabled": True, "rem_enabled": True,
            "use_mech286_sleep_onset_gate": False,
        },
    }


if __name__ == "__main__":
    import argparse
    import time
    from datetime import datetime

    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    t0 = time.perf_counter()
    result = run(seeds=args.seeds, dry_run=args.dry_run)

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    result["run_timestamp"]      = ts
    result["timestamp_utc"]      = ts
    result["run_id"]             = f"{EXPERIMENT_TYPE}_{ts}_v3"
    result["architecture_epoch"] = "ree_hybrid_guardrails_v1"
    result["experiment_purpose"] = EXPERIMENT_PURPOSE
    result["claim_ids"]          = CLAIM_IDS

    out_dir = (Path(__file__).resolve().parents[2]
               / "REE_assembly" / "evidence" / "experiments" / EXPERIMENT_TYPE)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = write_flat_manifest(
        result,
        out_dir.parent,
        dry_run=args.dry_run,
        config=result.get("config"),
        seeds=(args.seeds if args.seeds is not None else DEFAULT_SEEDS),
        script_path=Path(__file__),
        started_at=t0,
        z_goal_stream_stats=_ZG.stats(),
    )
    print(f"\nResult written to: {out_path}", flush=True)
    print(f"Status: {result['status']}", flush=True)
    print(f"final_outcome: {result['outcome']}", flush=True)

    _outcome_raw = str(result.get("outcome", "FAIL")).upper()
    emit_outcome(outcome=_outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL",
                 manifest_path=out_path,
                 dry_run=bool(args.dry_run))
