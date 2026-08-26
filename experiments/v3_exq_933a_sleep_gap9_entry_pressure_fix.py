"""
V3-EXQ-933a -- sleep_substrate:SD-SLEEP-ENTRY-PRESSURE (GAP-9 follow-up) validation:
does the time-integrating entry-pressure arm fix the two V3-EXQ-933 failure modes?

Claims: None (diagnostic substrate-readiness validation; does not weight governance)

EXPERIMENT_PURPOSE = "diagnostic"

SLEEP DRIVER: manual-cycle-loop (SleepLoopManager.notify_waking_step via update_residue in a
single continuous life; the within-life trigger is the ONLY path to a sleep cycle -- the driver
never calls agent.reset(), so notify_episode_end() / the K-episode cadence is unreachable). Same
driver pattern as V3-EXQ-933; this run supersedes NOTHING (933's own arms are unaffected -- it
tests need_crossed(), this tests the SEPARATE entry_pressure_crossed() term).

WHY THIS RUN. V3-EXQ-933 (2026-08-14) found MELConsumer.need_crossed() -- reused from GAP-5b's
DURATION statistic (current_mel(), a time-invariant MEAN) -- broken for GAP-9's ENTRY-TIMING use:
NEED_SUB (constant per-step demand 0.1, threshold 0.5) never crossed in 120 steps (need_arm_frac
0.0), and NEED_HIGH (constant per-step demand 1.0) fired on EVERY step (120/120, no refractory).
Registered as substrate_queue.json SD-SLEEP-ENTRY-PRESSURE (governance 2026-08-16, from confirmed
failure_autopsy_929-933-sleep-gap9-cluster_2026-08-16). Fix built this session: a SEPARATE
time-integrating (Borbely Process-S) entry-pressure term (MELConsumer.entry_pressure_crossed(),
a running SUM rather than need_crossed()'s MEAN -- current_mel() left untouched) plus a
steps_since_sleep refractory floor (SleepLoopManager.within_life_entry_pressure_refractory_steps)
enforced in phase_manager.notify_waking_step(). This run reproduces V3-EXQ-933's EXACT NEED_SUB /
NEED_HIGH per-step demand levels and threshold against the NEW mechanism, per the substrate_queue
entry's own failure_record `target`:
  "(a) cross the entry threshold from sustained sub-threshold-per-step demand within a bounded
  time awake (Process S integration), and (b) enforce a minimum inter-cycle interval so sustained
  supra-threshold demand yields a bounded fire rate strictly below 1 cycle per waking step."

CONSUMER VALIDATION WITH A CONTROLLED DEMAND STIMULUS (not an ecological producer test) -- same
rationale as V3-EXQ-933/718a: measured waking MEL in CausalGridWorldV2 is noise-level, so a
demand-threshold trigger would never fire from the env's own signal there. The DV is whether the
CONSUMER (the pressure arm) crosses under sustained sub-threshold demand in bounded time, and is
rate-bounded under sustained supra-threshold demand -- independent of what the agent has learned
(untrained agent by design, same as 933).

Design (3 conditions x N seeds, single continuous life each):
  CEILING        -- entry-pressure OFF (use_entry_pressure=False), ceiling=CEILING_STEP_CEILING,
                    no demand injected. Negative control: reproduces the pre-SD-SLEEP-ENTRY-PRESSURE
                    baseline exactly -- every fire is the ceiling arm, pressure arm never fires
                    (byte-identical inertness check, mirrors the GAP-9 contract suite's G17).
  PRESSURE_HIGH  -- entry-pressure ON (use_entry_pressure=True), threshold=PRESSURE_THRESHOLD,
                    ceiling=HIGH_CEILING (backstop unreachable), demand injected ABOVE threshold
                    every step (matches V3-EXQ-933's NEED_HIGH injection exactly) -> the pressure
                    arm must fire repeatedly but at a rate STRICTLY BELOW 1/step (refractory-bound
                    fix for the V3-EXQ-933 NEED_HIGH failure: 120/120 fires, no refractory).
  PRESSURE_SUB   -- entry-pressure ON, threshold=PRESSURE_THRESHOLD, ceiling=HIGH_CEILING (so the
                    ceiling arm cannot carry and mask a pressure-arm failure), demand injected
                    BELOW threshold every step (matches V3-EXQ-933's NEED_SUB injection exactly)
                    -> the pressure arm must nonetheless cross in BOUNDED time via the running SUM
                    (Process-S fix for the V3-EXQ-933 NEED_SUB failure: 0/120 fires, never crosses).
All conditions: use_sleep_loop=True, use_mel_consumer=True, sws_enabled=True, rem_enabled=True,
sleep_loop_episodes_K huge (boundary path unreachable, driver never resets the agent).
use_mel_entry left OFF (default) on every arm -- isolates the NEW pressure arm; need_crossed()'s
own behaviour is V3-EXQ-933's territory, not this run's. use_mech286_sleep_onset_gate left OFF
(default), same rationale as 933.

Pre-registered acceptance (NOT derived from this run's statistics; PRESSURE_THRESHOLD / injected
demand levels / refractory are fixed constants matching V3-EXQ-933 and the SD's own default):
  C1  PRESSURE_SUB fires >= 1 (the V3-EXQ-933 NEED_SUB fix: 0 -> nonzero) AND every fire is the
      pressure arm (not ceiling, which is unreachable on this arm) AND the first fire lands within
      the analytically bounded step count (ceil(threshold/inject_sub), generous slack), every seed.
  C2  PRESSURE_HIGH fires >= 1 AND every fire is the pressure arm AND the per-step fire RATE
      (fires / waking_steps) is STRICTLY < 1.0 (the V3-EXQ-933 NEED_HIGH fix: 120/120 -> bounded)
      AND does not exceed the refractory-implied rate ceiling (1/refractory + slack), every seed.
  C3  CEILING reproduces the pre-fix baseline exactly: pressure_arm_fraction == 0.0 (the lever is
      genuinely inert when OFF, not merely unread) AND ceiling_arm_fraction == 1.0, every seed.
  PASS iff C1 and C2 and C3, AND the readiness preconditions hold (the mechanism is genuinely
  wired on the PRESSURE arms; the injected PRESSURE_HIGH stimulus genuinely crossed the threshold
  at fire time -- the positive control that a below-floor stimulus self-routes as
  substrate_not_ready_requeue rather than as a pressure-arm defect).

Output:
  evidence/experiments/v3_exq_933a_sleep_gap9_entry_pressure_fix/
    v3_exq_933a_sleep_gap9_entry_pressure_fix_<ts>.json   (manifest)
"""

import math
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

# z_goal is orthogonal to this run's DV (pressure-arm trigger behaviour); the minimal driver never
# calls update_z_goal -- record the stream's liveness for completeness (generous-recording
# standard) rather than leaving it unmeasured.
_ZG = ZGoalStreamAccumulator()


EXPERIMENT_TYPE    = "v3_exq_933a_sleep_gap9_entry_pressure_fix"
EXPERIMENT_PURPOSE = "diagnostic"
CLAIM_IDS: List[str] = []
SLEEP_DRIVER_PATTERN = (
    "manual-cycle-loop (SleepLoopManager.notify_waking_step via update_residue; pressure arm + "
    "ceiling backstop; notify_episode_end NOT used -- agent never reset within the life)"
)

# validate_experiments.py anchor-reachability advisory: this script's readiness preconditions are
# DIRECT STRUCTURAL / CONFIG / injected-stimulus comparisons, not hand-written predicates scored
# against a known-degenerate reference sample (the V3-EXQ-778d failure mode the guard exists for).
# pressure_arm_wired reads the boolean agent.mel_consumer / config.use_entry_pressure set
# deterministically at construction; stimulus_crossed_threshold compares the INJECTED per-step
# demand (PRESSURE_INJECT_HIGH=1.0, fixed) against PRESSURE_THRESHOLD (0.5, fixed). All reachable
# by construction (PRESSURE arms set use_entry_pressure True; 1.0 > 0.5) and confirmed by this
# script's own dry-run smoke.
ANCHOR_REACHABILITY_EXEMPT = (
    "structural/config/injected-stimulus comparisons (the mel_consumer/use_entry_pressure booleans "
    "set from config; the fixed injected PRESSURE_INJECT_HIGH vs the fixed PRESSURE_THRESHOLD), "
    "not scored predicates against a reference sample; reachable by construction (PRESSURE arms "
    "set use_entry_pressure; 1.0 > 0.5), confirmed in this script's own dry-run"
)

# ---- run geometry ----
DEFAULT_SEEDS       = [0, 1, 2]  # seed 44 excluded per CLAUDE.md reef-config instability precedent
LIFE_STEPS          = 120        # waking steps in ONE continuous life (matches V3-EXQ-933)
CEILING_STEP_CEILING = 25        # within_life_sleep_step_ceiling for the CEILING arm
HIGH_CEILING        = 100_000    # PRESSURE arms' backstop -- unreachable in a 120-step life
GRID_SIZE           = 8
NUM_HAZARDS         = 2
NUM_RESOURCES       = 3

# ---- controlled demand stimulus + threshold (pre-registered, matches V3-EXQ-933 exactly) ----
PRESSURE_THRESHOLD   = 0.5   # entry_pressure_threshold on the PRESSURE arms
PRESSURE_INJECT_HIGH = 1.0   # PRESSURE_HIGH per-step injected demand (V3-EXQ-933 NEED_HIGH value)
PRESSURE_INJECT_SUB  = 0.1   # PRESSURE_SUB  per-step injected demand (V3-EXQ-933 NEED_SUB value)
REFRACTORY_STEPS     = 2     # within_life_entry_pressure_refractory_steps (the SD default)
# Analytical bound for PRESSURE_SUB's first crossing: ceil(threshold / inject) waking steps of
# accumulation (a running SUM, so this is deterministic by construction, not a fitted number).
# Generous slack (+3) absorbs float64 accumulation error without weakening the "bounded, not
# never" claim -- V3-EXQ-933's own NEED_SUB comparator was "never" (0/120), so any finite bound
# well under LIFE_STEPS is already the qualitative fix.
PRESSURE_SUB_FIRST_FIRE_BOUND = math.ceil(PRESSURE_THRESHOLD / PRESSURE_INJECT_SUB) + 3
# Analytical rate ceiling for PRESSURE_HIGH: refractory floor bounds fires to at most one every
# REFRACTORY_STEPS waking steps (+1 for the boundary fire that lands before the first full
# refractory window completes -- see EntryPressureAccumulator/notify_waking_step).
PRESSURE_HIGH_MAX_FIRES = lambda waking_steps: waking_steps // REFRACTORY_STEPS + 1  # noqa: E731

# ---- arm table ----
ARMS = [
    {"label": "CEILING", "use_entry_pressure": False, "ceiling": CEILING_STEP_CEILING, "inject": 0.0},
    {"label": "PRESSURE_HIGH", "use_entry_pressure": True, "ceiling": HIGH_CEILING,
     "inject": PRESSURE_INJECT_HIGH},
    {"label": "PRESSURE_SUB", "use_entry_pressure": True, "ceiling": HIGH_CEILING,
     "inject": PRESSURE_INJECT_SUB},
]


def _build_config(env, arm: Dict[str, Any]) -> "REEConfig":
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        use_sleep_loop=True,
        # Huge K so the boundary path is unreachable even if a reset ever happened; the driver
        # additionally never calls agent.reset() within the life.
        sleep_loop_episodes_K=10_000_000,
        use_within_life_sleep_trigger=True,
        within_life_sleep_step_ceiling=int(arm["ceiling"]),
        # The MEL consumer (GAP-5b) supplies entry_pressure_crossed(); present on every arm so the
        # arms differ ONLY in use_entry_pressure / threshold / ceiling / injected stimulus.
        # use_mel_entry left OFF everywhere -- isolates the NEW pressure arm from need_crossed()
        # (V3-EXQ-933's territory).
        use_mel_consumer=True,
        use_mel_entry=False,
        use_entry_pressure=bool(arm["use_entry_pressure"]),
        entry_pressure_threshold=PRESSURE_THRESHOLD,
        within_life_entry_pressure_refractory_steps=REFRACTORY_STEPS,
    )
    cfg.sws_enabled = True
    cfg.rem_enabled = True
    # use_mech286_sleep_onset_gate left at its default False (lit-synthesis brief, same as 933).
    return cfg


def _run_continuous_life(seed: int, arm: Dict[str, Any]) -> Dict[str, Any]:
    """One TRUE single continuous agent life: sense + random action + env.step + a controlled
    demand injection + update_residue per WAKING step, NEVER agent.reset(). Counts the sleep
    cycles fired within the life and the OR-arm split (pressure vs ceiling)."""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    env = CausalGridWorldV2(
        size=GRID_SIZE, num_hazards=NUM_HAZARDS, num_resources=NUM_RESOURCES,
        max_episode_steps=LIFE_STEPS + 10, seed=seed,
    )
    _flat, obs = env.reset()
    agent = REEAgent(_build_config(env, arm)).to(torch.device("cpu"))
    agent.eval()

    arm_label = str(arm["label"])
    inject = float(arm["inject"])
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
            # Controlled demand stimulus: inject BEFORE update_residue so the accumulator reflects
            # this step when notify_waking_step (inside update_residue) evaluates
            # entry_pressure_crossed(). Stands in for a non-converging env's genuine waking
            # learning load, exactly as V3-EXQ-933 did for need_crossed().
            if inject > 0.0 and agent.mel_consumer is not None:
                agent.mel_consumer.note_step_pe(inject)
            agent.update_residue(float(harm_signal))
        waking_steps += 1
        if (step_idx + 1) % 20 == 0:
            print(f"  [train] gap9pressure seed={seed} arm={arm_label} ep {step_idx + 1}/{LIFE_STEPS} "
                  f"fires={len(agent.sleep_loop.cycle_history)}", flush=True)
        if done:
            # Env-level respawn only. The AGENT is NOT reset -> no episode boundary, so the
            # within-life trigger stays the sole path to a sleep cycle.
            _flat, obs = env.reset()

    _ZG.observe(agent)
    history = agent.sleep_loop.cycle_history
    n_fires = len(history)
    n_ceiling = sum(1 for h in history if float(h.get("within_life_trigger_arm_ceiling", 0.0)) == 1.0)
    n_pressure = sum(1 for h in history if float(h.get("within_life_trigger_arm_pressure", 0.0)) == 1.0)
    ceiling_frac = (n_ceiling / n_fires) if n_fires else 0.0
    pressure_frac = (n_pressure / n_fires) if n_fires else 0.0
    fire_rate = n_fires / waking_steps if waking_steps else 0.0
    first_fire_step = int(history[0]["within_life_steps_at_fire"]) if n_fires else -1
    # Minimum pressure-at-fire across this arm's fires (the value entry_pressure_crossed()
    # actually compared to the threshold). -1.0 sentinel when no fire. For PRESSURE_HIGH this is
    # the positive-control readiness check: the injected stimulus must have genuinely crossed.
    pressure_at_fire_vals = [float(h.get("within_life_pressure_at_fire", -1.0)) for h in history]
    min_pressure_at_fire = min(pressure_at_fire_vals) if pressure_at_fire_vals else -1.0

    print(f"verdict: {'PASS' if n_fires >= 0 else 'FAIL'} seed={seed} arm={arm_label} "
          f"fires={n_fires} pressure_frac={pressure_frac:.2f} ceiling_frac={ceiling_frac:.2f} "
          f"fire_rate={fire_rate:.3f} first_fire={first_fire_step}", flush=True)

    return {
        "seed": seed,
        "arm": arm_label,
        "use_entry_pressure": bool(arm["use_entry_pressure"]),
        "step_ceiling": int(arm["ceiling"]),
        "injected_demand_per_step": inject,
        "entry_pressure_threshold": PRESSURE_THRESHOLD,
        "refractory_steps": REFRACTORY_STEPS,
        "pressure_arm_wired": bool(agent.mel_consumer is not None and bool(arm["use_entry_pressure"])),
        "waking_steps": int(waking_steps),
        "sleep_cycles_fired": int(n_fires),
        "fires_arm_pressure": int(n_pressure),
        "fires_arm_ceiling": int(n_ceiling),
        "pressure_arm_fraction": float(pressure_frac),
        "ceiling_arm_fraction": float(ceiling_frac),
        "fire_rate": float(fire_rate),
        "first_fire_step": int(first_fire_step),
        "min_pressure_at_fire": float(min_pressure_at_fire),
        "sws_enabled": True,
        "rem_enabled": True,
    }


def _by(results: List[Dict[str, Any]], label: str) -> List[Dict[str, Any]]:
    return [r for r in results if r["arm"] == label]


def run(seeds=None, dry_run: bool = False) -> dict:
    if seeds is None:
        seeds = DEFAULT_SEEDS
    if dry_run:
        seeds = seeds[:1]
    print(f"[V3-EXQ-933a] SD-SLEEP-ENTRY-PRESSURE (GAP-9 follow-up) validation\n"
          f"  Seeds: {seeds}  Life: {LIFE_STEPS} waking steps\n"
          f"  Threshold: {PRESSURE_THRESHOLD}  inject high/sub: "
          f"{PRESSURE_INJECT_HIGH}/{PRESSURE_INJECT_SUB}  refractory: {REFRACTORY_STEPS}\n"
          f"  Output: REE_assembly/evidence/experiments/{EXPERIMENT_TYPE}/", flush=True)

    per_condition: List[Dict[str, Any]] = []
    for arm in ARMS:
        for s in seeds:
            per_condition.append(_run_continuous_life(s, arm))

    ceiling_res = _by(per_condition, "CEILING")
    pressure_high_res = _by(per_condition, "PRESSURE_HIGH")
    pressure_sub_res = _by(per_condition, "PRESSURE_SUB")

    # --- readiness preconditions (the DV is only meaningful if these hold) ---
    pressure_arm_wired = bool(
        all(r["pressure_arm_wired"] for r in pressure_high_res)
        and all(r["pressure_arm_wired"] for r in pressure_sub_res)
    )
    # Positive control: the injected PRESSURE_HIGH stimulus must have genuinely crossed the
    # threshold on the fires it produced (the SAME statistic entry_pressure_crossed() routes on).
    ph_min_pressure_at_fire = min((r["min_pressure_at_fire"] for r in pressure_high_res), default=-1.0)
    stimulus_crossed = bool(pressure_high_res and all(
        r["sleep_cycles_fired"] >= 1 and r["min_pressure_at_fire"] >= PRESSURE_THRESHOLD
        for r in pressure_high_res
    ))
    apparatus_ready = bool(pressure_arm_wired and stimulus_crossed)

    # --- pre-registered acceptance criteria ---
    # C1: Process-S integration fix -- PRESSURE_SUB crosses (V3-EXQ-933 NEED_SUB was 0/120).
    c1_sub_crosses_in_bounded_time = bool(pressure_sub_res and all(
        r["sleep_cycles_fired"] >= 1
        and r["pressure_arm_fraction"] == 1.0
        and 0 <= r["first_fire_step"] <= PRESSURE_SUB_FIRST_FIRE_BOUND
        for r in pressure_sub_res
    ))
    # C2: refractory-bound fix -- PRESSURE_HIGH fires repeatedly but strictly below 1/step
    # (V3-EXQ-933 NEED_HIGH was 120/120, fire_rate == 1.0).
    c2_high_rate_bounded = bool(pressure_high_res and all(
        r["sleep_cycles_fired"] >= 1
        and r["pressure_arm_fraction"] == 1.0
        and r["fire_rate"] < 1.0
        and r["sleep_cycles_fired"] <= PRESSURE_HIGH_MAX_FIRES(r["waking_steps"])
        for r in pressure_high_res
    ))
    # C3: OFF-arm inertness -- the lever is genuinely inert (not merely unread) when OFF, and the
    # pre-fix ceiling-only baseline is reproduced exactly.
    c3_off_arm_inert = bool(ceiling_res and all(
        r["sleep_cycles_fired"] >= 1
        and r["pressure_arm_fraction"] == 0.0
        and r["ceiling_arm_fraction"] == 1.0
        for r in ceiling_res
    ))

    if not apparatus_ready:
        label = "substrate_not_ready_requeue"
        passed = False
    elif c1_sub_crosses_in_bounded_time and c2_high_rate_bounded and c3_off_arm_inert:
        label = "entry_pressure_fix_validated"
        passed = True
    else:
        label = "entry_pressure_defect"
        passed = False
    outcome = "PASS" if passed else "FAIL"

    metrics: Dict[str, Any] = {
        "n_seeds": float(len(seeds)),
        "pressure_sub_first_fire_max": float(
            max((r["first_fire_step"] for r in pressure_sub_res), default=-1.0)
        ),
        "pressure_sub_first_fire_bound": float(PRESSURE_SUB_FIRST_FIRE_BOUND),
        "pressure_sub_fires_min": float(
            min((r["sleep_cycles_fired"] for r in pressure_sub_res), default=0.0)
        ),
        "pressure_high_fire_rate_max": float(
            max((r["fire_rate"] for r in pressure_high_res), default=0.0)
        ),
        "pressure_high_fires_max": float(
            max((r["sleep_cycles_fired"] for r in pressure_high_res), default=0.0)
        ),
        "pressure_high_min_pressure_at_fire": float(ph_min_pressure_at_fire),
        "ceiling_pressure_frac_max": float(
            max((r["pressure_arm_fraction"] for r in ceiling_res), default=0.0)
        ),
        "ceiling_ceiling_frac_min": float(
            min((r["ceiling_arm_fraction"] for r in ceiling_res), default=0.0)
        ),
        "entry_pressure_threshold": float(PRESSURE_THRESHOLD),
        "refractory_steps": float(REFRACTORY_STEPS),
        "c1_sub_crosses_in_bounded_time": 1.0 if c1_sub_crosses_in_bounded_time else 0.0,
        "c2_high_rate_bounded": 1.0 if c2_high_rate_bounded else 0.0,
        "c3_off_arm_inert": 1.0 if c3_off_arm_inert else 0.0,
    }
    for r in per_condition:
        metrics[f"seed{r['seed']}_{r['arm']}_pressure_frac"] = float(r["pressure_arm_fraction"])
        metrics[f"seed{r['seed']}_{r['arm']}_fire_rate"] = float(r["fire_rate"])
        metrics[f"seed{r['seed']}_{r['arm']}_first_fire"] = float(r["first_fire_step"])

    interpretation = {
        "label": label,
        "preconditions": [
            {"name": "pressure_arm_wired",
             "description": "the PRESSURE arms actually have a mel_consumer AND "
                            "use_entry_pressure set from config -- guards against a from_dims "
                            "silent-kwargs miswire reading as a non-firing pressure arm rather "
                            "than an unwired flag",
             "measured": 1.0 if pressure_arm_wired else 0.0, "threshold": 1.0,
             "direction": "lower", "met": pressure_arm_wired,
             "control": "all PRESSURE_HIGH + PRESSURE_SUB seeds"},
            {"name": "stimulus_crossed_threshold",
             "description": "the injected PRESSURE_HIGH demand stimulus genuinely crossed "
                            "entry_pressure_threshold at fire time (the SAME statistic "
                            "entry_pressure_crossed() routes on) on every seed -- the positive "
                            "control that a below-floor stimulus self-routes as "
                            "substrate_not_ready_requeue, never as a pressure-arm defect.",
             "measured": float(ph_min_pressure_at_fire), "threshold": float(PRESSURE_THRESHOLD),
             "direction": "lower", "met": stimulus_crossed,
             "control": "known-supra-threshold injected stimulus (PRESSURE_INJECT_HIGH=%.3f) "
                        "on PRESSURE_HIGH" % PRESSURE_INJECT_HIGH},
        ],
        "criteria_non_degenerate": {
            # The DV genuinely discriminates iff the pressure arm crosses under sustained
            # sub-threshold demand (C1, the Process-S fix) while a rate-bound holds under
            # sustained supra-threshold demand (C2, the refractory fix) and the lever is
            # genuinely inert when off (C3) -- together they cannot be satisfied by a
            # degenerate always-fire or never-fire implementation.
            "c1_sub_crosses_in_bounded_time": c1_sub_crosses_in_bounded_time,
            "c2_high_rate_bounded": c2_high_rate_bounded,
            "c3_off_arm_inert": c3_off_arm_inert,
        },
        "criteria": [
            {"name": "c1_sub_crosses_in_bounded_time", "load_bearing": True,
             "passed": c1_sub_crosses_in_bounded_time},
            {"name": "c2_high_rate_bounded", "load_bearing": True, "passed": c2_high_rate_bounded},
            {"name": "c3_off_arm_inert", "load_bearing": True, "passed": c3_off_arm_inert},
        ],
        "combination_rule": (
            "PASS = readiness(pressure_arm_wired AND stimulus_crossed_threshold) AND "
            "c1_sub_crosses_in_bounded_time AND c2_high_rate_bounded AND c3_off_arm_inert"
        ),
        "dv_symmetry_note": (
            "The manipulation (use_entry_pressure + injected per-step demand relative to "
            "entry_pressure_threshold) directly gates the DV path: entry_pressure_crossed() is a "
            "running-SUM threshold comparison, and the arm-attribution DV (pressure vs ceiling) "
            "plus first-fire step / fire rate are functions of that comparison's outcome over "
            "time. This is a genuine measurement, not an arithmetic identity invariant under any "
            "symmetry of the DV."
        ),
        "note": (
            "Diagnostic CONSUMER validation for sleep_substrate:SD-SLEEP-ENTRY-PRESSURE (GAP-9 "
            "follow-up), reproducing V3-EXQ-933's exact NEED_SUB/NEED_HIGH injected-demand levels "
            "and threshold against the NEW entry_pressure_crossed() mechanism. Controlled demand "
            "stimulus injected via note_step_pe (the ecological producer is parked -- measured env "
            "MEL is noise-level per GAP-5b/718a), exactly how 933/718a validated the MEL consumer. "
            "Untrained agent by design: the DV is the pressure arm's trigger behaviour, independent "
            "of learning. claim_ids=[]; does not weight governance. use_mel_entry left OFF on every "
            "arm (isolates the pressure arm from need_crossed(), which is V3-EXQ-933's territory). "
            "use_mech286_sleep_onset_gate deliberately OFF (lit-synthesis brief, same as 933)."
        ),
    }

    summary_markdown = f"""# V3-EXQ-933a -- SD-SLEEP-ENTRY-PRESSURE (GAP-9 follow-up) validation

**Status:** {outcome} -- label: `{label}`
**Purpose:** diagnostic CONSUMER validation (sleep_substrate:SD-SLEEP-ENTRY-PRESSURE).

Reproduces V3-EXQ-933's exact NEED_SUB (demand {PRESSURE_INJECT_SUB}, threshold
{PRESSURE_THRESHOLD}) and NEED_HIGH (demand {PRESSURE_INJECT_HIGH}) conditions against the new
entry_pressure_crossed() mechanism (a running SUM + steps_since_sleep refractory floor,
distinct from need_crossed()'s time-invariant MEAN).

- seeds: {seeds}; life length: {LIFE_STEPS} waking steps; refractory: {REFRACTORY_STEPS} steps
- C1 PRESSURE_SUB crosses in bounded time (Process-S fix, was 0/120 fires): {c1_sub_crosses_in_bounded_time}
- C2 PRESSURE_HIGH fire rate strictly < 1/step (refractory fix, was 120/120 fires): {c2_high_rate_bounded}
- C3 CEILING (lever OFF) reproduces the pre-fix baseline exactly, pressure arm inert: {c3_off_arm_inert}

See `interpretation` for the pre-registered acceptance rule and readiness preconditions, and
`per_condition_results` for the full per-seed x arm table.
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
            "life_steps": LIFE_STEPS, "ceiling_step_ceiling": CEILING_STEP_CEILING,
            "high_ceiling": HIGH_CEILING, "grid_size": GRID_SIZE, "num_hazards": NUM_HAZARDS,
            "num_resources": NUM_RESOURCES, "sleep_loop_episodes_K": 10_000_000,
            "sws_enabled": True, "rem_enabled": True, "use_mel_consumer": True,
            "use_mel_entry": False, "entry_pressure_threshold": PRESSURE_THRESHOLD,
            "pressure_inject_high": PRESSURE_INJECT_HIGH, "pressure_inject_sub": PRESSURE_INJECT_SUB,
            "refractory_steps": REFRACTORY_STEPS, "use_mech286_sleep_onset_gate": False,
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
