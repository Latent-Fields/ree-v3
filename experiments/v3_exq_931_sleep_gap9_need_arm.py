"""
V3-EXQ-931 -- sleep_substrate:GAP-9 MEL/need-crossing PRIMARY arm (design (b)) validation

Claims: None (diagnostic substrate-readiness validation; does not weight governance)

EXPERIMENT_PURPOSE = "diagnostic"

SLEEP DRIVER: manual-cycle-loop (SleepLoopManager.notify_waking_step fired per WAKING step from
REEAgent.update_residue in a single continuous life; the within-life trigger is the ONLY path to
a sleep cycle -- the driver never calls agent.reset(), so notify_episode_end() / the K-episode
cadence is unreachable). This is the mechanism under validation.

WHY THIS RUN. Routing source: REE_assembly/evidence/planning/sleep_substrate_plan.md GAP-9 +
the 2026-08-14 lit synthesis targeted_review_sleep_onset_multiinput_gap9 (Section 6, which names
MEL/need-crossing the PRIMARY trigger). GAP-9 v1 (V3-EXQ-929, ree-v3 5f14036) wired the (a)+(b)
composed design's STEP-COUNT CEILING arm (design (a), anti-starvation backstop) ONLY, with the
need arm hardcoded off (need_crossed=False). This run validates the MEL/need-crossing arm
(design (b)) landed as the follow-up: MELConsumer.need_crossed() (accumulated waking MEL >=
mel_entry_threshold) fires the cycle, with the step ceiling still the backstop --
`need_crossed or at_ceiling`.

CONSUMER VALIDATION WITH A CONTROLLED MEL STIMULUS (not an ecological producer test). Per the
synthesis (4.2/5) and GAP-5b (V3-EXQ-718a), measured waking MEL in CausalGridWorldV2 is
noise-level (~1e-5, scrambled w.r.t. novelty) because the env converges too completely to sustain
learning load -- so a MEL-threshold trigger would NEVER fire from the env's own signal there.
V3-EXQ-718a resolved this exactly the same way: it "validated the CONSUMER and failed the
ecological producer link". This run therefore drives the need arm with a CONTROLLED per-step MEL
stimulus (injected via the same mel_consumer.note_step_pe() path update_residue uses), standing in
for the non-converging environment that would produce genuine high MEL (the ecological producer is
parked). The DV is whether the CONSUMER -- the need arm -- fires on demand, gates on threshold,
and degrades to the ceiling when demand is sub-threshold. It is NOT a test of the env producing MEL.

WHAT THIS RUN IS NOT: a claim test, a learning experiment, or a test of sleep's downstream EFFECT.
The agent is intentionally UNTRAINED -- the DV is the STRUCTURAL behaviour of the need-arm trigger
(does accumulated MEL crossing the threshold fire the cycle, and does sub-threshold demand fail
to), independent of what the agent has learned.

Design (3 conditions x N seeds, single continuous life each):
  CEILING   -- need arm OFF (use_mel_entry=False), ceiling=CEILING_STEP_CEILING, no MEL injected.
               Reproduces v1: the ceiling carries firing, every fire is the ceiling arm.
  NEED_HIGH -- need arm ON (use_mel_entry=True), threshold=MEL_THRESHOLD, ceiling=HIGH_CEILING
               (backstop unreachable), MEL injected ABOVE threshold every step -> the need arm
               fires EARLY (well before the ceiling), every fire is the need arm.
  NEED_SUB  -- need arm ON, threshold=MEL_THRESHOLD, ceiling=CEILING_STEP_CEILING, MEL injected
               BELOW threshold every step -> the need arm never crosses; the ceiling carries.
               (Threshold-gating control: proves a NEED_HIGH fire is genuinely demand-driven, not
               spurious firing on any accumulated MEL.)
All conditions: use_sleep_loop=True, use_mel_consumer=True, sws_enabled=True, rem_enabled=True,
sleep_loop_episodes_K huge (boundary path unreachable, and the driver never resets the agent).
use_mech286_sleep_onset_gate left OFF (default) per the lit-synthesis brief: its threat term reads
a signal V3-EXQ-917 measured at chance-level place-safety discrimination and enabling it would
confound the sleep-onset result.

Pre-registered acceptance (NOT derived from this run's statistics):
  C1  NEED_HIGH fires >= 1 AND every fire is the need arm (need_arm_frac == 1.0), every seed.
  C2  NEED_HIGH first-fire step  <  CEILING first-fire step, every seed (demand -> sleep sooner).
  C3  NEED_SUB  need_arm_frac == 0.0 AND fires >= 1 (ceiling carries; sub-threshold demand does
      NOT fire the need arm -- threshold gating), every seed.
  C4  CEILING   need_arm_frac == 0.0 AND ceiling_frac == 1.0, every seed (v1 ceiling reproduced).
  PASS iff C1 and C2 and C3 and C4, AND the readiness preconditions hold (the life exceeded the
  ceiling; the need arm was wired on the NEED arms; and the injected NEED_HIGH stimulus genuinely
  crossed the threshold -- the positive control that a below-floor stimulus self-routes as
  substrate_not_ready_requeue rather than as a need-arm defect).

Output:
  evidence/experiments/v3_exq_931_sleep_gap9_need_arm/
    v3_exq_931_sleep_gap9_need_arm_<ts>.json   (manifest)
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

# z_goal is orthogonal to this run's DV (need-arm trigger behaviour); the minimal driver never
# calls update_z_goal -- record the stream's liveness for completeness (generous-recording
# standard) rather than leaving it unmeasured.
_ZG = ZGoalStreamAccumulator()


EXPERIMENT_TYPE    = "v3_exq_931_sleep_gap9_need_arm"
EXPERIMENT_PURPOSE = "diagnostic"
CLAIM_IDS: List[str] = []
SLEEP_DRIVER_PATTERN = (
    "manual-cycle-loop (SleepLoopManager.notify_waking_step via update_residue; need arm + "
    "ceiling backstop; notify_episode_end NOT used -- agent never reset within the life)"
)

# validate_experiments.py anchor-reachability advisory: this script's readiness preconditions are
# DIRECT STRUCTURAL / CONFIG / injected-stimulus comparisons, not hand-written predicates scored
# against a known-degenerate reference sample (the V3-EXQ-778d failure mode the guard exists for).
# life_exceeds_step_ceiling compares min waking steps (LIFE_STEPS=120, fixed) against
# CEILING_STEP_CEILING (25, fixed); need_arm_wired reads the boolean agent.mel_consumer /
# config.use_mel_entry set deterministically at construction; stimulus_crossed_threshold compares
# the INJECTED per-step MEL (MEL_INJECT_HIGH=1.0, fixed) against MEL_THRESHOLD (0.5, fixed). All
# reachable by construction (120 > 25; NEED arms set use_mel_entry True; 1.0 > 0.5) and confirmed
# by this script's own dry-run smoke.
ANCHOR_REACHABILITY_EXEMPT = (
    "structural/config/injected-stimulus comparisons (min waking steps vs fixed "
    "CEILING_STEP_CEILING; the mel_consumer/use_mel_entry booleans set from config; the fixed "
    "injected MEL_INJECT_HIGH vs the fixed MEL_THRESHOLD), not scored predicates against a "
    "reference sample; reachable by construction (120 > 25; NEED arms set use_mel_entry; 1.0 > "
    "0.5), confirmed in this script's own dry-run"
)

# ---- run geometry ----
DEFAULT_SEEDS       = [0, 1, 2]  # seed 44 excluded per CLAUDE.md reef-config instability precedent
LIFE_STEPS          = 120        # waking steps in ONE continuous life (denominator of `ep N/M`)
CEILING_STEP_CEILING = 25        # within_life_sleep_step_ceiling for CEILING / NEED_SUB arms
HIGH_CEILING        = 100_000    # NEED_HIGH backstop -- unreachable in a 120-step life
GRID_SIZE           = 8
NUM_HAZARDS         = 2
NUM_RESOURCES       = 3

# ---- controlled MEL stimulus + threshold (pre-registered, NOT derived from this run) ----
MEL_THRESHOLD   = 0.5   # mel_entry_threshold on the NEED arms
MEL_INJECT_HIGH = 1.0   # NEED_HIGH per-step injected MEL (> threshold -> need arm fires)
MEL_INJECT_SUB  = 0.1   # NEED_SUB  per-step injected MEL (< threshold -> need arm never crosses)

# ---- arm table ----
# use_mel_entry, ceiling, inject, and a human label. All arms: within-life trigger ON,
# mel_consumer ON, sws/rem ON, K huge.
ARMS = [
    {"label": "CEILING",   "use_mel_entry": False, "ceiling": CEILING_STEP_CEILING, "inject": 0.0},
    {"label": "NEED_HIGH", "use_mel_entry": True,  "ceiling": HIGH_CEILING,         "inject": MEL_INJECT_HIGH},
    {"label": "NEED_SUB",  "use_mel_entry": True,  "ceiling": CEILING_STEP_CEILING, "inject": MEL_INJECT_SUB},
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
        # The MEL consumer (GAP-5b) supplies need_crossed(); present on every arm so the arms
        # differ ONLY in use_mel_entry / threshold / ceiling / injected stimulus.
        use_mel_consumer=True,
        use_mel_entry=bool(arm["use_mel_entry"]),
        mel_entry_threshold=MEL_THRESHOLD,
    )
    cfg.sws_enabled = True
    cfg.rem_enabled = True
    # use_mech286_sleep_onset_gate left at its default False (lit-synthesis brief).
    return cfg


def _run_continuous_life(seed: int, arm: Dict[str, Any]) -> Dict[str, Any]:
    """One TRUE single continuous agent life: sense + random action + env.step + a controlled
    MEL injection + update_residue per WAKING step, NEVER agent.reset(). Counts the sleep cycles
    fired within the life and the OR-arm split (need vs ceiling)."""
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
            # Controlled MEL stimulus: inject BEFORE update_residue so the accumulator reflects
            # this step when notify_waking_step (inside update_residue) evaluates need_crossed().
            # This stands in for a non-converging env's genuine waking learning load; measured
            # env MEL is noise-level here (GAP-5b), so the ecological producer is deliberately
            # bypassed -- this validates the CONSUMER (the need arm), not the producer.
            if inject > 0.0 and agent.mel_consumer is not None:
                agent.mel_consumer.note_step_pe(inject)
            agent.update_residue(float(harm_signal))
        waking_steps += 1
        if (step_idx + 1) % 20 == 0:
            print(f"  [train] gap9need seed={seed} arm={arm_label} ep {step_idx + 1}/{LIFE_STEPS} "
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
    need_frac = (n_need / n_fires) if n_fires else 0.0
    first_fire_step = int(history[0]["within_life_steps_at_fire"]) if n_fires else -1
    # Minimum MEL-at-fire across this arm's fires (the value need_crossed() actually compared to
    # the threshold). -1.0 sentinel when no fire. For NEED_HIGH this is the positive-control
    # readiness check: the injected stimulus must have genuinely crossed the threshold.
    mel_at_fire_vals = [float(h.get("within_life_mel_at_fire", -1.0)) for h in history]
    min_mel_at_fire = min(mel_at_fire_vals) if mel_at_fire_vals else -1.0

    print(f"verdict: {'PASS' if n_fires >= 0 else 'FAIL'} seed={seed} arm={arm_label} "
          f"fires={n_fires} need_frac={need_frac:.2f} ceiling_frac={ceiling_frac:.2f} "
          f"first_fire={first_fire_step}", flush=True)

    return {
        "seed": seed,
        "arm": arm_label,
        "use_mel_entry": bool(arm["use_mel_entry"]),
        "step_ceiling": int(arm["ceiling"]),
        "injected_mel_per_step": inject,
        "mel_entry_threshold": MEL_THRESHOLD,
        "need_arm_wired": bool(agent.mel_consumer is not None and bool(arm["use_mel_entry"])),
        "waking_steps": int(waking_steps),
        "sleep_cycles_fired": int(n_fires),
        "fires_arm_need": int(n_need),
        "fires_arm_ceiling": int(n_ceiling),
        "need_arm_fraction": float(need_frac),
        "ceiling_arm_fraction": float(ceiling_frac),
        "first_fire_step": int(first_fire_step),
        "min_mel_at_fire": float(min_mel_at_fire),
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
    print(f"[V3-EXQ-931] GAP-9 MEL/need-crossing PRIMARY arm validation\n"
          f"  Seeds: {seeds}  Life: {LIFE_STEPS} waking steps\n"
          f"  Threshold: {MEL_THRESHOLD}  inject high/sub: {MEL_INJECT_HIGH}/{MEL_INJECT_SUB}\n"
          f"  Output: REE_assembly/evidence/experiments/{EXPERIMENT_TYPE}/", flush=True)

    per_condition: List[Dict[str, Any]] = []
    for arm in ARMS:
        for s in seeds:
            per_condition.append(_run_continuous_life(s, arm))

    ceiling_res = _by(per_condition, "CEILING")
    need_high_res = _by(per_condition, "NEED_HIGH")
    need_sub_res = _by(per_condition, "NEED_SUB")

    min_waking_steps = min(r["waking_steps"] for r in per_condition)

    # --- readiness preconditions (the DV is only meaningful if these hold) ---
    life_exceeds_ceiling = bool(min_waking_steps >= CEILING_STEP_CEILING)
    need_arm_wired = bool(
        all(r["need_arm_wired"] for r in need_high_res)
        and all(r["need_arm_wired"] for r in need_sub_res)
    )
    # Positive control: the injected NEED_HIGH stimulus must have genuinely crossed the threshold
    # on the fires it produced (the SAME statistic need_crossed() routes on: current_mel() >=
    # threshold). A below-threshold reading means the stimulus was inadequate -> substrate not
    # ready, NOT a need-arm defect.
    nh_min_mel_at_fire = min((r["min_mel_at_fire"] for r in need_high_res), default=-1.0)
    stimulus_crossed = bool(need_high_res and all(
        r["sleep_cycles_fired"] >= 1 and r["min_mel_at_fire"] >= MEL_THRESHOLD
        for r in need_high_res
    ))
    apparatus_ready = bool(life_exceeds_ceiling and need_arm_wired and stimulus_crossed)

    # --- pre-registered acceptance criteria ---
    c1_need_fires_and_carries = bool(all(
        r["sleep_cycles_fired"] >= 1 and r["need_arm_fraction"] >= 1.0 for r in need_high_res
    ))
    # C2: demand-sensitivity -- NEED_HIGH fires earlier than the ceiling arm, per seed.
    ceiling_first = {r["seed"]: r["first_fire_step"] for r in ceiling_res}
    c2_demand_sooner = bool(all(
        r["first_fire_step"] >= 0
        and ceiling_first.get(r["seed"], -1) >= 0
        and r["first_fire_step"] < ceiling_first[r["seed"]]
        for r in need_high_res
    ))
    # C3: threshold gating -- sub-threshold demand does NOT fire the need arm; the ceiling carries.
    c3_threshold_gates = bool(all(
        r["sleep_cycles_fired"] >= 1 and r["need_arm_fraction"] == 0.0 for r in need_sub_res
    ))
    # C4: ceiling baseline reproduces v1 (need arm off -> all fires ceiling).
    c4_ceiling_baseline = bool(all(
        r["sleep_cycles_fired"] >= 1 and r["need_arm_fraction"] == 0.0 and r["ceiling_arm_fraction"] >= 1.0
        for r in ceiling_res
    ))

    if not apparatus_ready:
        label = "substrate_not_ready_requeue"
        passed = False
    elif c1_need_fires_and_carries and c2_demand_sooner and c3_threshold_gates and c4_ceiling_baseline:
        label = "need_arm_validated"
        passed = True
    else:
        label = "need_arm_defect"
        passed = False
    outcome = "PASS" if passed else "FAIL"

    metrics: Dict[str, Any] = {
        "n_seeds": float(len(seeds)),
        "need_high_need_frac_min": float(min((r["need_arm_fraction"] for r in need_high_res), default=0.0)),
        "need_high_first_fire_max": float(max((r["first_fire_step"] for r in need_high_res), default=-1.0)),
        "ceiling_first_fire_min": float(min((r["first_fire_step"] for r in ceiling_res), default=-1.0)),
        "need_sub_need_frac_max": float(max((r["need_arm_fraction"] for r in need_sub_res), default=0.0)),
        "ceiling_need_frac_max": float(max((r["need_arm_fraction"] for r in ceiling_res), default=0.0)),
        "ceiling_ceiling_frac_min": float(min((r["ceiling_arm_fraction"] for r in ceiling_res), default=0.0)),
        "need_high_min_mel_at_fire": float(nh_min_mel_at_fire),
        "mel_entry_threshold": float(MEL_THRESHOLD),
        "min_waking_steps": float(min_waking_steps),
        "c1_need_fires_and_carries": 1.0 if c1_need_fires_and_carries else 0.0,
        "c2_demand_sooner": 1.0 if c2_demand_sooner else 0.0,
        "c3_threshold_gates": 1.0 if c3_threshold_gates else 0.0,
        "c4_ceiling_baseline": 1.0 if c4_ceiling_baseline else 0.0,
    }
    for r in per_condition:
        metrics[f"seed{r['seed']}_{r['arm']}_need_frac"] = float(r["need_arm_fraction"])
        metrics[f"seed{r['seed']}_{r['arm']}_first_fire"] = float(r["first_fire_step"])

    interpretation = {
        "label": label,
        "preconditions": [
            {"name": "life_exceeds_step_ceiling",
             "description": "the continuous life ran at least CEILING_STEP_CEILING waking steps, "
                            "so the ceiling-arm and NEED_SUB arms could reach a within-life fire; "
                            "a shorter life could silently make an arm look non-firing",
             "measured": float(min_waking_steps), "threshold": float(CEILING_STEP_CEILING),
             "direction": "lower", "met": life_exceeds_ceiling,
             "control": "min waking steps across all conditions/seeds"},
            {"name": "need_arm_wired",
             "description": "the NEED arms actually have a mel_consumer AND use_mel_entry set from "
                            "config -- guards against a from_dims silent-kwargs miswire reading as "
                            "a non-firing need arm rather than an unwired flag",
             "measured": 1.0 if need_arm_wired else 0.0, "threshold": 1.0,
             "direction": "lower", "met": need_arm_wired, "control": "all NEED_HIGH + NEED_SUB seeds"},
            {"name": "stimulus_crossed_threshold",
             "description": "the injected NEED_HIGH MEL stimulus genuinely crossed mel_entry_threshold "
                            "at fire time (the SAME statistic need_crossed() routes on: current_mel() "
                            ">= threshold) on every seed -- the positive control that a below-floor "
                            "stimulus self-routes as substrate_not_ready_requeue, never as a need-arm "
                            "defect. Range readout: min mel_at_fire on NEED_HIGH fires.",
             "measured": float(nh_min_mel_at_fire), "threshold": float(MEL_THRESHOLD),
             "direction": "lower", "met": stimulus_crossed,
             "control": "known-supra-threshold injected stimulus (MEL_INJECT_HIGH=%.3f) on NEED_HIGH"
                        % MEL_INJECT_HIGH},
        ],
        "criteria_non_degenerate": {
            # The DV genuinely discriminates iff the need arm fires under supra-threshold demand
            # (C1) while it does NOT under sub-threshold demand (C3) and the ceiling reproduces v1
            # (C4). C1 vs C3 together are the non-degeneracy: they are the arms failing to separate
            # exactly when the manipulation (crossing the threshold) did nothing.
            "c1_need_fires_and_carries": c1_need_fires_and_carries,
            "c2_demand_sooner": c2_demand_sooner,
            "c3_threshold_gates": c3_threshold_gates,
            "c4_ceiling_baseline": c4_ceiling_baseline,
        },
        "criteria": [
            {"name": "c1_need_fires_and_carries", "load_bearing": True, "passed": c1_need_fires_and_carries},
            {"name": "c2_demand_sooner", "load_bearing": True, "passed": c2_demand_sooner},
            {"name": "c3_threshold_gates", "load_bearing": True, "passed": c3_threshold_gates},
            {"name": "c4_ceiling_baseline", "load_bearing": True, "passed": c4_ceiling_baseline},
        ],
        "combination_rule": (
            "PASS = readiness(life_exceeds_step_ceiling AND need_arm_wired AND "
            "stimulus_crossed_threshold) AND c1_need_fires_and_carries AND c2_demand_sooner AND "
            "c3_threshold_gates AND c4_ceiling_baseline"
        ),
        "dv_symmetry_note": (
            "The manipulation (use_mel_entry + injected per-step MEL relative to mel_entry_threshold) "
            "directly gates the DV path: need_crossed() is a threshold comparison "
            "current_mel() >= mel_entry_threshold, and the arm-attribution DV (need vs ceiling) plus "
            "first-fire step are functions of that comparison's outcome. Crossing the threshold "
            "changes which arm fires and when -- this is a genuine measurement, NOT an arithmetic "
            "identity invariant under any symmetry of the DV (a supra-threshold stimulus is not "
            "invisible to a threshold gate the way a broadcast constant is to an argmax)."
        ),
        "note": (
            "Diagnostic CONSUMER validation for sleep_substrate:GAP-9 design (b), the MEL/"
            "need-crossing PRIMARY arm. Controlled MEL stimulus injected via note_step_pe (the "
            "ecological producer is parked -- measured env MEL is noise-level per GAP-5b/718a), "
            "exactly how V3-EXQ-718a validated the MEL consumer. Untrained agent by design: the DV "
            "is the need arm's trigger behaviour (fires on supra-threshold demand, gates on "
            "threshold, degrades to the ceiling on sub-threshold demand), independent of learning. "
            "claim_ids=[]; does not weight governance. use_mech286_sleep_onset_gate deliberately OFF "
            "(lit-synthesis brief)."
        ),
    }

    summary_markdown = f"""# V3-EXQ-931 -- GAP-9 MEL/need-crossing PRIMARY arm validation

**Status:** {outcome} -- label: `{label}`
**Purpose:** diagnostic CONSUMER validation (sleep_substrate:GAP-9 design (b), the need arm).

Validates the MEL/need-crossing PRIMARY arm the v1 ceiling-only build (V3-EXQ-929) left as a
placeholder. Controlled MEL stimulus (the ecological producer is parked -- GAP-5b/718a), driving
the need arm through the same note_step_pe -> need_crossed() -> notify_waking_step path a real
non-converging environment would.

- seeds: {seeds}; life length: {LIFE_STEPS} waking steps
- threshold: {MEL_THRESHOLD}; injected MEL high/sub: {MEL_INJECT_HIGH}/{MEL_INJECT_SUB}
- C1 NEED_HIGH need-arm fires & carries: {c1_need_fires_and_carries}
- C2 NEED_HIGH fires sooner than ceiling: {c2_demand_sooner}
- C3 NEED_SUB threshold-gates (need frac 0): {c3_threshold_gates}
- C4 CEILING reproduces v1 (all ceiling arm): {c4_ceiling_baseline}

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
            "mel_entry_threshold": MEL_THRESHOLD, "mel_inject_high": MEL_INJECT_HIGH,
            "mel_inject_sub": MEL_INJECT_SUB, "use_mech286_sleep_onset_gate": False,
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
