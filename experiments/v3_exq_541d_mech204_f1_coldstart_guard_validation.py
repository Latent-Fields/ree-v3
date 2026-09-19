#!/opt/local/bin/python3
"""V3-EXQ-541d -- MECH-204 F1 cold-start guard validation

!! NOT QUEUED. DO NOT QUEUE THIS SCRIPT AS IT STANDS. !!
================================================================================
RED-TEAM VERDICT: **BLOCKING** (fable, 2026-09-19, /queue-experiment Step 4.5).
Every load-bearing criterion below is pinned to its passing value by the
HARNESS, not by the manipulation, so the run would emit
`f1_coldstart_guard_validated` under essentially every outcome. The finding was
verified against source AND against measured data before being accepted:

  F1 (verified to 8e-5 on a 16-cycle traced probe, seed 42 / step 0.25).
     The two arms consume identical RNG -- the flag only adds a counter
     increment in REEAgent.sense() -- and E3's running variance is an EMA with
     alpha 0.05, so after 200 waking ticks the pre-episode rv carries weight
     0.95**200 = 3.5e-5 and the per-cycle honest precision p_j is the SAME in
     both arms. With serotonin.py:390-397 (persistent <- 0.9*persistent +
     0.1*p) the targets are then, exactly:
         OFF_k - ON_k == 0.9**k * (SENTINEL - p_1)
     Measured: predicted vs actual agree to 8e-5 at every one of 16 cycles.
     So D5's "5.59x separation" is NOT a measurement of the guard's benefit --
     it is D1 (did the cold-start cycle capture?) multiplied by a regime
     constant, (2.0 - p_1)/anchor. Re-parameterise pe_scale so E[pe^2] = 0.5
     and D5 FAILS with a perfectly working guard; use IGW-243's 0.0039 and it
     reads ~100x. The docstring claim that "D5 carries the discriminating
     information" was WRONG.
  F2 D4, the pre-registered FALSIFIER OF THE WHOLE FIX, cannot fire on this
     base. It asks whether precision-at-REM-entry drifts AWAY from the anchor;
     but on 541c's harness that precision is 1/rv of a STATIONARY synthetic
     stream (pe_scale = 0.4 + 0.3*rng.random(), a function of rng alone --
     independent of agent, episode and prior WRITEBACK). There is no realized
     precision to drift. The IGW-243 climb D4 was written from (2 -> 27 -> 49
     -> 69 -> 88) was the agent's REALIZED precision. Measured: the guard-ON
     gap to the anchor is non-monotone over the first 10 cycles.
  F3 D1 re-reads the guard's own `if` back out. At the cold-start cycle C0,
     `_waking_ticks_since_capture == 0` by driver construction, so
     `d1_cell_ok == (guard XOR captured)` is identically True. It duplicates
     contracts C1-C8 in tests/contracts/test_mech204_f1_coldstart_guard.py.
     Its "OFF captures AT THE SENTINEL" clause was never actually asserted
     (E3_SENTINEL_PRECISION is recorded but never compared).
  F4 The verdict grid therefore collapses: every readiness-met path reaches
     `f1_coldstart_guard_validated`.
  F5 Readiness P2 bands a quantity that is the driver's own constant
     (E[pe^2] = 0.31 by construction) and cannot fail.

ROOT CAUSE, stated plainly: the cold-start guard's effect on 541c's
synthetic-PE harness is a FIRST-CYCLE TRANSIENT with a closed-form decay, not
the sustained divergence the design's prediction describes. The base and the
question do not match. Choosing a different base CHANGES WHAT GETS MEASURED,
which is a user decision, not this session's -- so this script was authored,
smoke-tested, red-teamed and then DELIBERATELY NOT QUEUED.

Escalated as decision chip chip-20260919-exq541d-base-regime-blocking.
Refusal record: REE_assembly/evidence/planning/exq541d_redteam_blocking_refusal_staged_20260919.md
Governance flag raised against MECH-204.

WHAT IS STILL GOOD HERE, and why this file was kept rather than deleted: the
C0 instrumentation is correct and verified -- it is what PROVED the defect
mechanism (guard OFF anchors _persistent_zero_point at the precision_init
sentinel 1.999996 in all five step arms; guard ON declines to capture), and it
measured that this base issues exactly ONE enter_rem call per sleep cycle. A
successor on a REALIZED-PE base should reuse this instrumentation and replace
D1/D4/D5 with criteria that are not pinned by the harness.
================================================================================

SLEEP DRIVER: K=1 single-fire (SleepLoopManager, sleep_loop_episodes_K=1, fires every episode)

Validation experiment for the MECH-204 F1 cold-start guard landed 2026-09-18
(ree-v3 e1ff0927): SerotoninConfig.precision_zero_point_require_waking. When
True, a REM entry with no waking tick since the last capture does NOT touch
_persistent_zero_point. Default False -> bit-identical OFF.
Substrate record: docs/substrate/MECH-204-f1-coldstart-guard.md
Contracts: tests/contracts/test_mech204_f1_coldstart_guard.py C1-C8.

red-team (see queue entry note): verdict recorded at queue time.

THE DEFECT UNDER TEST (mechanism, measured 2026-09-19 on this exact harness)
---------------------------------------------------------------------------
541c's driver calls agent.reset() ONCE before its episode loop. With
sleep_loop_episodes_K=1 that pre-loop reset fires a full sleep cycle with ZERO
waking ticks (call it C0). At C0's REM entry, E3 has never seen a prediction
error, so current_precision is the precision_init sentinel:
    precision_init = 0.5 (a VARIANCE) -> current_precision = 1/(0.5+1e-6) = 2.0
With the guard OFF, C0 anchors _persistent_zero_point at that sentinel 2.0.
Every later cycle then blends toward the honest precision at only alpha=0.1, so
the F1 reference spends the whole run climbing out of a cold-start artefact
rather than tracking the agent's realized precision.

Measured on this harness (seed 42, recal step 0.25, 16 cycles, 200 ticks/ep):
    cycle-1 target, guard OFF = 2.1478  == 0.9*2.0 + 0.1*(1/0.287498)
    cycle-1 target, guard ON  = 3.4783  == 1/0.287498  (honest; C0 suppressed)
The OFF value reproduces V3-EXQ-541c's documented cycle-1 target of 2.148,
which is what identifies this as the same defect 541c recorded.

WAKING-TICK PRODUCER (verified against origin/main, not from the docstring)
---------------------------------------------------------------------------
note_waking_tick() is called from REEAgent.sense() (agent.py:4939, inside
`def sense(` at :4858), gated on the flag so default-off adds no call. That
comment block explicitly names v3_exq_541c as a driver it covers, and this
driver inherits 541c's _tick_wake, which calls agent.sense() every tick.
NOTE a stale comment in the substrate: serotonin.py:144 and note_waking_tick()'s
own docstring still say the producer is REEAgent.update_residue(). That is
WRONG on origin/main -- update_residue does not call it, and 541c's driver calls
neither update_residue() nor serotonin_step(). If the docstring were right this
guard would be a permanent kill switch on this base. Measured: it is not
(P1 below records _waking_ticks_since_capture at REM entry = STEPS_PER_EPISODE).

RESET ORDERING (checked, because getting it wrong measures nothing)
-------------------------------------------------------------------
SerotoninModule.reset() zeroes _waking_ticks_since_capture (serotonin.py:429)
and this driver fires its cycle via agent.reset(). REEAgent.reset() calls
sleep_loop.notify_episode_end() at agent.py:3597 and serotonin.reset() at
:3626 -- sleep FIRST, deliberately. Were it the other way round the guard would
suppress every capture.

DESIGN (pre-specified by chip-20260918-exq541d-mech204-f1-guard-validation)
---------------------------------------------------------------------------
Base: v3_exq_541c_mech204_step_size_sweep_extended_cycles.py, unchanged env,
seeds, cycle count and recal-step arm structure. precision_zero_point_require_
waking is added as a SECOND FACTOR (OFF vs ON), fully crossed: 5 steps x 2
guard levels x 3 seeds = 30 cells.

ANCHOR RE-ANCHORING (user decision 2026-09-19T22:12:50Z)
---------------------------------------------------------------------------
The chip pre-registers the guard-ON prediction as "cycle 2+ targets sit within
~2x of 1/realized-PE-variance (~255 in the IGW-20260915-243 setting)". The
ANCHOR IS A RULE -- 1/realized-PE-variance -- and the ~255 is that rule's value
in a DIFFERENT harness. 541c does not have a realized world-model PE: it drives
E3's running variance from a SYNTHETIC stream (pe_scale ~ U[0.4,0.7], so
E[pe^2] = 0.31). The user directed this run to stay on 541c's synthetic-PE base
with the criterion re-anchored by the rule the design gives. So the anchor here
is computed IN-RUN, per cell, from that cell's own PE stream:
    anchor_precision = 1 / realized_pe_variance      (measured ~3.24, not ~255)

STATED LIMITATION, PRE-REGISTERED RATHER THAN DISCOVERED AFTERWARDS
---------------------------------------------------------------------------
D2 ("within ~2x of the anchor") is NON-DISCRIMINATING in this regime and is
recorded as such (criteria_non_degenerate["D2"] = False). Reason, measured:
the discriminating power of a 2x band depends on the sentinel/anchor ratio,
which is a property of the REGIME, not of the guard. Here sentinel 2.0 vs
anchor 3.24 is a factor of 1.6 -- comfortably inside a 2x band -- whereas in
the IGW-243 setting sentinel 2.0 vs anchor 255 is a factor of 127, far outside
it. Measured over 16 cycles at step 0.25 / seed 42, BOTH arms satisfy D2 at
EVERY cycle. D2 is therefore kept in the conjunction (faithful to the
pre-registered prediction) but flagged degenerate, and the information is
carried by D5, the cross-arm anchor-gap contrast, which separates cleanly:
    mean |target - anchor|/anchor over cycles 1..8:  OFF 0.2481  ON 0.0444
    -> 5.59x separation (cycle 1 alone: OFF 0.3369 vs ON 0.0739, 4.56x)

PRE-REGISTERED CRITERIA (all four design clauses, mechanically re-anchored)
---------------------------------------------------------------------------
  D1 (load-bearing) COLD-START NO-FIRE. Guard ON: C0 does not capture, so
      _persistent_zero_point is still None after the pre-loop reset and the
      first recorded cycle's target equals the honest 1/rv at REM entry.
      Guard OFF: C0 DOES capture, at the sentinel. Required in 3/3 seeds in
      every arm. This is the design's "cycle 1 does not fire at all".
  D2  ANCHOR BAND (cycles 2+, guard ON): |log2(target/anchor)| <= 1 at every
      cycle. NON-DISCRIMINATING in this regime -- see above. Not load-bearing.
  D3  RV DIRECTION: fraction of cycles where |rv_after - realized_pe_var| <
      |rv_before - realized_pe_var| (rv moves TOWARD realized PE variance
      rather than away). Reported per arm; not load-bearing (the step=0.0 arm
      cannot move rv at all by construction).
  D4 (load-bearing) FALSIFIER OF THE WHOLE FIX. With the guard ON, does the
      target climb MONOTONICALLY AWAY from the anchor over the first 10
      cycles? If YES in >=2/3 seeds in the defensible-step arms, the
      precision_init sentinel was not the (only) cause, precision-at-REM-entry
      is not a calibration-relevant quantity, and MECH-204 Option A needs
      REDESIGN. Per the chip: that outcome DEMOTES Option A, it does not
      retune it.
  D5 (load-bearing) CROSS-ARM ANCHOR-GAP CONTRAST. Mean relative anchor gap
      over cycles 1..N_CONTRAST, guard OFF vs guard ON, same step and seed.
      Pre-registered: ON is at least D5_MIN_SEPARATION (2.0x) closer to the
      anchor than OFF, in >=2/3 seeds, in at least one defensible-step arm.
      Measured at step 0.25 / seed 42: 5.59x.
PASS = D1 AND D2 AND D4_not_falsified AND D5.

WHAT THIS RUN CANNOT SEE (recorded so a reader does not assume otherwise)
---------------------------------------------------------------------------
The chip also asks 541d to be able to see a second instance of the defect --
"a cycle issues MORE THAN ONE enter_rem() call, and the later call re-reads the
SAME precision with no intervening waking tick". Measured on this base: the
541c driver issues EXACTLY ONE enter_rem per sleep cycle (17 calls for 16
episodes = 16 cycles + C0), because ree_core has a single internal enter_rem
caller (agent.py:12701 in enter_rem_mode(), reached once per run_sleep_cycle).
The double-count instance therefore does NOT occur on this base and this run
cannot exercise it. enter_rem call counts are recorded per cycle anyway
(n_enter_rem_calls_per_cycle) so the absence is auditable rather than assumed,
and a driver that DOES multi-fire REM (the v3_exq_sd068_* family calls
enter_rem_mode directly) is where that instance should be tested.

EXPERIMENT_PURPOSE is "diagnostic", NOT "evidence" -- deliberately, and this is
the single easiest thing here to get wrong by inheriting 541c's tag. MECH-204's
what_would_answer (a) in claims.yaml states the claim is NOT ANSWERABLE until
sd_waking_confidence_inflation_headroom lands and the V3-EXQ-794a re-run
follows; that substrate entry reads status implemented / ready FALSE /
validation_experiment "V3-EXQ-794a (not yet queued)". 541d validates an
INSTRUMENT (the cold-start guard), it does not adjudicate MECH-204.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiment_protocol import emit_outcome  # noqa: E402
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402
from experiments._lib.arm_fingerprint import arm_cell  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_541d_mech204_f1_coldstart_guard_validation"
CLAIM_IDS = ["MECH-204"]
EXPERIMENT_PURPOSE = "diagnostic"
BACKLOG_ID = "EXP-0171"
PRECISION_ZERO_POINT_EMA_ALPHA = 0.1

# Both anchor-kind readiness preconditions are DIRECT COUNTER READS with a
# >=1 gate -- P1 reads SerotoninModule._waking_ticks_since_capture (incremented
# once per REEAgent.sense() call, i.e. STEPS_PER_EPISODE times per cycle) and
# P3 reads the enter_rem call count. Neither is a hand-written signature
# predicate that could be narrower than the state it anchors to, so the
# unmeetable-by-construction failure this check guards cannot arise. Measured
# directly on this exact harness (seed 42, step 0.25, 2026-09-19): P1 = 200,
# P3 = 1.
ANCHOR_REACHABILITY_EXEMPT = (
    "P1/P3 are direct counter reads gated at >=1, not scored signatures; "
    "reachability measured on this harness (P1=200, P3=1)"
)

# Seeds held identical to V3-EXQ-541c ("same seeds", per the chip's design).
SEEDS = (42, 43, 44)
RECAL_STEPS = (
    ("step_0_00", 0.0),
    ("step_0_05", 0.05),
    ("step_0_10", 0.10),
    ("step_0_25", 0.25),
    ("step_0_50", 0.50),
)
GUARD_LEVELS = (("guard_off", False), ("guard_on", True))
EPISODES_PER_RUN = 16
STEPS_PER_EPISODE = 200
SLEEP_LOOP_K = 1

# E3 cold-start sentinel: precision_init is a VARIANCE (config.py:1126).
E3_PRECISION_INIT_VARIANCE = 0.5
E3_SENTINEL_PRECISION = 1.0 / (E3_PRECISION_INIT_VARIANCE + 1e-6)  # == ~2.0

# Pre-registered thresholds.
D1_MIN_SEEDS = 3                 # cold-start no-fire must hold in 3/3 seeds
D2_ANCHOR_BAND_LOG2 = 1.0        # |log2(target/anchor)| <= 1  <=> within 2x
D4_MONOTONE_CYCLES = 10          # falsifier window (chip: "first ~10 cycles")
D4_MIN_SEEDS_FALSIFIED = 2       # >=2/3 seeds to call the fix falsified
D5_MIN_SEPARATION = 2.0          # OFF gap / ON gap must be >= this
D5_MIN_SEEDS = 2                 # in >=2/3 seeds
N_CONTRAST_CYCLES = 8            # D5 scoring window (cycles 1..8)
DEFENSIBLE_STEPS = (0.05, 0.10, 0.25)

# Readiness precondition bounds.
P1_MIN_WAKING_TICKS = 1          # floor: producer must be live at REM entry
P2_PE_VAR_LOW = 0.05             # interval: realized PE variance sane
P2_PE_VAR_HIGH = 5.0
P3_EXPECTED_C0_CALLS = 1         # floor: the cold-start cycle must exist


def _env_kwargs() -> dict:
    return dict(
        size=8,
        num_hazards=6,
        num_resources=2,
        hazard_harm=0.06,
        proximity_harm_scale=0.18,
        proximity_benefit_scale=0.10,
        env_drift_interval=5,
        env_drift_prob=0.5,
        use_proxy_fields=True,
        resource_respawn_on_consume=True,
    )


def cell_config_slice(step: float, guard: bool, episodes: int,
                      steps_per_episode: int) -> dict:
    """Everything this cell's computation reads. Declared for the fingerprint."""
    return {
        "env_kwargs": _env_kwargs(),
        "episodes_per_run": episodes,
        "steps_per_episode": steps_per_episode,
        "sleep_loop_K": SLEEP_LOOP_K,
        "precision_zero_point_ema_alpha": PRECISION_ZERO_POINT_EMA_ALPHA,
        "rem_precision_recalibration_step": step,
        "precision_zero_point_require_waking": guard,
        # Scoring constants the cell's own readouts are computed under. A
        # consumer using different values must MISS these cells rather than
        # silently reuse readouts computed under another scheme
        # (arm_reuse_fingerprint_plan.md 7b; confirmed instance V3-EXQ-798).
        "d2_anchor_band_log2": D2_ANCHOR_BAND_LOG2,
        "d4_monotone_cycles": D4_MONOTONE_CYCLES,
        "n_contrast_cycles": N_CONTRAST_CYCLES,
        "self_dim": 32,
        "world_dim": 32,
        "alpha_world": 0.9,
        "alpha_self": 0.3,
        "sws_consolidation_steps": 8,
        "sws_schema_weight": 0.1,
        "rem_attribution_steps": 6,
        "tonic_5ht_enabled": True,
    }


def _make_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(seed=seed, **_env_kwargs())


def _make_agent(env: CausalGridWorldV2, seed: int, *, step: float,
                guard: bool) -> REEAgent:
    torch.manual_seed(seed)
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=32,
        world_dim=32,
        alpha_world=0.9,
        alpha_self=0.3,
        sws_enabled=True,
        sws_consolidation_steps=8,
        sws_schema_weight=0.1,
        rem_enabled=True,
        rem_attribution_steps=6,
        use_sleep_loop=True,
        sleep_loop_episodes_K=SLEEP_LOOP_K,
        use_rem_precision_recalibration=True,
        precision_zero_point_ema_alpha=PRECISION_ZERO_POINT_EMA_ALPHA,
        rem_precision_recalibration_step=step,
    )
    cfg.serotonin.tonic_5ht_enabled = True
    # The second factor under test. Default is False (bit-identical OFF).
    cfg.serotonin.precision_zero_point_require_waking = bool(guard)
    return REEAgent(cfg)


def _one_hot_action(action_idx: int, action_dim: int) -> torch.Tensor:
    action = torch.zeros(1, action_dim)
    action[0, int(action_idx)] = 1.0
    return action


def _tick_wake(agent: REEAgent, env: CausalGridWorldV2,
               obs_dict: dict, rng: random.Random, pe_sq: list) -> dict:
    """One waking tick. Identical to 541c's, plus PE second-moment recording.

    agent.sense() is the MECH-204 waking-tick producer (agent.py:4939), so this
    is also what makes the guard's predicate satisfiable on this driver.
    """
    obs_body = obs_dict["body_state"]
    obs_world = obs_dict["world_state"]
    latent = agent.sense(
        obs_body,
        obs_world,
        obs_harm=obs_dict.get("harm_obs"),
        obs_harm_a=obs_dict.get("harm_obs_a"),
        obs_harm_history=obs_dict.get("harm_history"),
    )
    ticks = agent.clock.advance()
    if ticks.get("e1_tick", False):
        agent._e1_tick(latent)
    # Drive E3 prediction-error EMA: synthetic PE keeps _running_variance
    # moving across waking ticks so the recalibration consumer has
    # something to act on at REM entry / WRITEBACK. (541c :191-197, verbatim.)
    if hasattr(agent, "e3"):
        pe_scale = 0.4 + 0.3 * rng.random()
        synthetic_pe = torch.randn(1, 4) * pe_scale
        # Realized PE variance -- the DESIGN'S ANCHOR RULE is 1/this.
        pe_sq.append(float((synthetic_pe ** 2).mean()))
        agent.e3.update_running_variance(synthetic_pe)

    action_idx = rng.randrange(env.action_dim)
    action = _one_hot_action(action_idx, env.action_dim)
    _flat, _harm, done, _info, next_obs = env.step(action)
    if done:
        _flat, next_obs = env.reset()
    return next_obs


def _safe_ratio(num: float, den: float) -> float:
    if abs(den) < 1e-12:
        return float("inf") if abs(num) > 1e-12 else 0.0
    return num / den


def run_cell(step_label: str, step: float, guard_label: str, guard: bool,
             seed: int, episodes_per_run: int,
             steps_per_episode: int) -> dict:
    """One (recal_step x guard x seed) cell."""
    arm_label = f"{step_label}__{guard_label}"
    print(f"Seed {seed} Condition {arm_label}", flush=True)

    slice_ = cell_config_slice(step, guard, episodes_per_run, steps_per_episode)
    with arm_cell(
        seed,
        config_slice=slice_,
        script_path=Path(__file__),
        config_slice_declared=True,
        # Cross-driver reusable mint: a successor iteration citing this run can
        # match these cells. Must be identical on the consumer side.
        include_driver_script_in_hash=False,
    ) as cell:
        rng = random.Random(seed)
        env = _make_env(seed)
        agent = _make_agent(env, seed, step=step, guard=guard)
        ser = agent.serotonin

        # --- Instrument BEFORE the first reset. The pre-loop agent.reset()
        # fires the zero-waking-tick cold-start cycle C0; instrumenting after
        # it would make C0 invisible, which is the whole defect. ---
        trace = {"calls": 0, "suppressed": 0, "admitted": 0,
                 "waking_ticks_at_entry": []}
        _orig_enter_rem = ser.enter_rem

        def _traced_enter_rem(current_precision):
            trace["calls"] += 1
            before = ser._persistent_zero_point
            trace["waking_ticks_at_entry"].append(
                int(ser._waking_ticks_since_capture)
            )
            _orig_enter_rem(current_precision=current_precision)
            after = ser._persistent_zero_point
            # Suppressed == the capture did not touch the reference at all.
            if before is None and after is None:
                trace["suppressed"] += 1
            elif before is not None and after == before:
                trace["suppressed"] += 1
            else:
                trace["admitted"] += 1

        ser.enter_rem = _traced_enter_rem

        pe_sq: list = []
        _flat, obs_dict = env.reset()

        # --- C0: the cold-start cycle, fired by the pre-loop reset. ---
        agent.reset()
        agent.e1.reset_hidden_state()
        c0_calls = trace["calls"]
        c0_suppressed = trace["suppressed"]
        c0_captured = bool(ser._persistent_zero_point is not None)
        c0_persistent = (None if ser._persistent_zero_point is None
                         else float(ser._persistent_zero_point))
        st0 = agent.sleep_loop.state if agent.sleep_loop else None
        if st0 is not None and st0.last_metrics:
            st0.last_metrics = {}

        cycle_records: list = []
        for ep in range(episodes_per_run):
            if (ep + 1) % 4 == 0 or ep == 0:
                print(
                    f"  [train] {arm_label} seed={seed} "
                    f"ep {ep + 1}/{episodes_per_run}",
                    flush=True,
                )
            calls_before = trace["calls"]
            for _ in range(steps_per_episode):
                obs_dict = _tick_wake(agent, env, obs_dict, rng, pe_sq)

            rv_pre_cycle = float(agent.e3._running_variance)
            agent.reset()

            cycle_state = agent.sleep_loop.state if agent.sleep_loop else None
            if cycle_state is not None and cycle_state.last_metrics:
                metrics = dict(cycle_state.last_metrics)
                cycle_records.append({
                    "episode": ep + 1,
                    "rv_pre_cycle": rv_pre_cycle,
                    "rv_post_cycle": float(agent.e3._running_variance),
                    "n_enter_rem_calls": trace["calls"] - calls_before,
                    "mech204_recalibration_fired": float(
                        metrics.get("mech204_recalibration_fired", 0.0)),
                    "mech204_recalibration_target": float(
                        metrics.get("mech204_recalibration_target", 0.0)),
                    "mech204_running_variance_before": float(
                        metrics.get("mech204_running_variance_before",
                                    float("nan"))),
                    "mech204_running_variance_after": float(
                        metrics.get("mech204_running_variance_after",
                                    float("nan"))),
                })
                cycle_state.last_metrics = {}

        # --- The design's anchor rule, instantiated on THIS cell's own
        # realized PE stream: anchor_precision = 1 / realized_pe_variance. ---
        realized_pe_var = (sum(pe_sq) / len(pe_sq)) if pe_sq else float("nan")
        anchor = (1.0 / realized_pe_var
                  if realized_pe_var and realized_pe_var > 1e-12
                  else float("nan"))

        targets = [c["mech204_recalibration_target"] for c in cycle_records]
        rel_gaps = [
            abs(t - anchor) / anchor
            for t in targets
            if anchor == anchor and anchor > 0 and t > 0
        ]
        # D2: |log2(target/anchor)| <= 1, cycles 2+ (design says "cycle 2+").
        d2_vals = [
            abs(math.log2(t / anchor))
            for t in targets[1:]
            if anchor == anchor and anchor > 0 and t > 0
        ]
        d2_within_band = bool(d2_vals) and all(
            v <= D2_ANCHOR_BAND_LOG2 for v in d2_vals)
        # D4: does the gap to the anchor GROW every cycle over the window?
        win = [abs(t - anchor) for t in targets[:D4_MONOTONE_CYCLES]
               if anchor == anchor]
        d4_monotone_away = bool(len(win) >= 2) and all(
            win[i + 1] > win[i] for i in range(len(win) - 1))
        # D3: rv moved TOWARD the realized PE variance.
        toward = 0
        n_rv = 0
        for c in cycle_records:
            rvb = c["mech204_running_variance_before"]
            rva = c["mech204_running_variance_after"]
            if rvb != rvb or rva != rva or realized_pe_var != realized_pe_var:
                continue
            n_rv += 1
            if abs(rva - realized_pe_var) < abs(rvb - realized_pe_var):
                toward += 1
        d3_toward_frac = (toward / n_rv) if n_rv else 0.0

        mean_rel_gap_contrast = (
            sum(rel_gaps[:N_CONTRAST_CYCLES]) / len(rel_gaps[:N_CONTRAST_CYCLES])
            if rel_gaps else float("nan"))

        # D1, per cell: guard ON must NOT capture at C0; OFF must capture.
        d1_cell_ok = (not c0_captured) if guard else c0_captured
        min_waking_at_entry = (
            min(trace["waking_ticks_at_entry"][1:])
            if len(trace["waking_ticks_at_entry"]) > 1 else 0)

        row = {
            "arm": arm_label,
            "recal_step_label": step_label,
            "recal_step": step,
            "guard_label": guard_label,
            "guard": bool(guard),
            "seed": seed,
            "n_cycles": len(cycle_records),
            "realized_pe_variance": float(realized_pe_var),
            "anchor_precision": float(anchor),
            "c0_enter_rem_calls": int(c0_calls),
            "c0_suppressed": int(c0_suppressed),
            "c0_captured": bool(c0_captured),
            "c0_persistent_zero_point": c0_persistent,
            "d1_cold_start_no_fire_ok": bool(d1_cell_ok),
            "d2_within_anchor_band": bool(d2_within_band),
            "d2_max_abs_log2_ratio": (max(d2_vals) if d2_vals else float("nan")),
            "d3_rv_toward_fraction": float(d3_toward_frac),
            "d4_monotone_away_from_anchor": bool(d4_monotone_away),
            "mean_rel_anchor_gap_contrast": float(mean_rel_gap_contrast),
            "targets": targets,
            "n_enter_rem_calls_total": int(trace["calls"]),
            "n_enter_rem_suppressed": int(trace["suppressed"]),
            "n_enter_rem_admitted": int(trace["admitted"]),
            "n_enter_rem_calls_per_cycle": [
                c["n_enter_rem_calls"] for c in cycle_records],
            "min_waking_ticks_at_rem_entry": int(min_waking_at_entry),
            "cycle_records": cycle_records,
        }
        cell.stamp(row)

    print(f"verdict: {'PASS' if d1_cell_ok else 'FAIL'}", flush=True)
    return row


def _aggregate(rows: list) -> dict:
    by_arm: dict = {}
    for r in rows:
        by_arm.setdefault(r["arm"], []).append(r)

    # --- D1: cold-start no-fire, every arm, D1_MIN_SEEDS seeds ---
    d1_per_arm = {
        arm: sum(1 for r in rs if r["d1_cold_start_no_fire_ok"])
        for arm, rs in by_arm.items()
    }
    d1_pass = all(n >= D1_MIN_SEEDS for n in d1_per_arm.values())

    # --- D2: anchor band, guard-ON cells only (the design's ON prediction) ---
    on_rows = [r for r in rows if r["guard"]]
    off_rows = [r for r in rows if not r["guard"]]
    d2_on_ok = sum(1 for r in on_rows if r["d2_within_anchor_band"])
    d2_pass = bool(on_rows) and d2_on_ok == len(on_rows)
    # Degeneracy evidence: does the OFF arm satisfy it too?
    d2_off_ok = sum(1 for r in off_rows if r["d2_within_anchor_band"])
    d2_degenerate = bool(off_rows) and d2_off_ok == len(off_rows)

    # --- D3: reported (not load-bearing) ---
    d3_per_arm = {
        arm: (sum(r["d3_rv_toward_fraction"] for r in rs) / len(rs))
        for arm, rs in by_arm.items()
    }

    # --- D4 FALSIFIER: ON targets climbing monotonically AWAY from anchor ---
    d4_falsified_arms = {}
    for arm, rs in by_arm.items():
        if not rs[0]["guard"] or rs[0]["recal_step"] not in DEFENSIBLE_STEPS:
            continue
        n = sum(1 for r in rs if r["d4_monotone_away_from_anchor"])
        d4_falsified_arms[arm] = n
    d4_falsified = any(n >= D4_MIN_SEEDS_FALSIFIED
                       for n in d4_falsified_arms.values())
    d4_pass = not d4_falsified

    # --- D5: cross-arm anchor-gap contrast, matched (step, seed) ---
    d5_per_step = {}
    for step_label, step in RECAL_STEPS:
        off = {r["seed"]: r["mean_rel_anchor_gap_contrast"]
               for r in rows
               if r["recal_step_label"] == step_label and not r["guard"]}
        on = {r["seed"]: r["mean_rel_anchor_gap_contrast"]
              for r in rows
              if r["recal_step_label"] == step_label and r["guard"]}
        seps = []
        for seed in sorted(set(off) & set(on)):
            seps.append(_safe_ratio(off[seed], on[seed]))
        n_ok = sum(1 for s in seps if s >= D5_MIN_SEPARATION)
        d5_per_step[step_label] = {
            "recal_step": step,
            "per_seed_separation": seps,
            "mean_separation": (sum(s for s in seps if s != float("inf"))
                                / len([s for s in seps if s != float("inf")])
                                if [s for s in seps if s != float("inf")]
                                else float("nan")),
            "seeds_meeting_separation": n_ok,
            "per_seed_off_gap": [off[s] for s in sorted(set(off) & set(on))],
            "per_seed_on_gap": [on[s] for s in sorted(set(off) & set(on))],
        }
    d5_qualifying = [
        lbl for lbl, d in d5_per_step.items()
        if d["recal_step"] in DEFENSIBLE_STEPS
        and d["seeds_meeting_separation"] >= D5_MIN_SEEDS
    ]
    d5_pass = bool(d5_qualifying)

    overall_pass = bool(d1_pass and d2_pass and d4_pass and d5_pass)

    return {
        "d1_pass": d1_pass,
        "d1_seeds_ok_per_arm": d1_per_arm,
        "d2_pass": d2_pass,
        "d2_on_cells_ok": d2_on_ok,
        "d2_off_cells_ok": d2_off_ok,
        "d2_degenerate_both_arms_pass": d2_degenerate,
        "d3_mean_toward_fraction_per_arm": d3_per_arm,
        "d4_pass_not_falsified": d4_pass,
        "d4_falsified": d4_falsified,
        "d4_monotone_away_seeds_per_arm": d4_falsified_arms,
        "d5_pass": d5_pass,
        "d5_qualifying_steps": d5_qualifying,
        "d5_per_step": d5_per_step,
        "overall_pass": overall_pass,
    }


def _build_interpretation(rows: list, crit: dict) -> dict:
    on_rows = [r for r in rows if r["guard"]]
    # P1 -- readiness: the waking-tick PRODUCER is live at REM entry in the ON
    # arm. Same statistic D1 routes on (_waking_ticks_since_capture == 0 is the
    # guard's own predicate). If this is below floor the guard is a kill switch
    # and every other number in this run is an artefact.
    p1_measured = (min(r["min_waking_ticks_at_rem_entry"] for r in on_rows)
                   if on_rows else 0)
    worst_p1 = min(on_rows,
                   key=lambda r: r["min_waking_ticks_at_rem_entry"]) if on_rows else None
    # P2 -- realized PE variance in a sane band, so 1/rv is a usable anchor.
    # `met` is a conjunction over every cell, so the reported `measured` must be
    # the WORST CELL with respect to the BAND (the value furthest from the
    # band's geometric centre), not a mean and not a one-sided min -- otherwise
    # the indexer's recompute reads an in-band number while a sibling cell sits
    # outside it. If the furthest cell is inside the band, all of them are.
    pe_pairs = [(r["realized_pe_variance"], r) for r in rows
                if r["realized_pe_variance"] == r["realized_pe_variance"]]
    _centre = math.sqrt(P2_PE_VAR_LOW * P2_PE_VAR_HIGH)
    if pe_pairs:
        _wv, _wr = max(pe_pairs,
                       key=lambda pr: abs(math.log(max(pr[0], 1e-12) / _centre)))
        p2_measured = float(_wv)
        p2_offending = _wr["arm"] + "/seed" + str(_wr["seed"])
    else:
        p2_measured = float("nan")
        p2_offending = None
    # P3 -- the cold-start cycle C0 exists at all (else D1 tests nothing).
    p3_measured = min(r["c0_enter_rem_calls"] for r in rows) if rows else 0
    worst_p3 = min(rows, key=lambda r: r["c0_enter_rem_calls"]) if rows else None

    preconditions = [
        {
            "name": "waking_tick_producer_live_at_rem_entry",
            "description": ("guard-ON cells must reach REM entry with "
                            "_waking_ticks_since_capture > 0 (post-C0), else "
                            "the guard is a permanent kill switch rather than "
                            "a cold-start guard"),
            "control": ("guard-ON cells after the cold-start cycle; "
                        "REEAgent.sense() is the producer (agent.py:4939)"),
            "measured": float(p1_measured),
            "threshold": float(P1_MIN_WAKING_TICKS),
            "direction": "lower",
            "offending_cell": (worst_p1["arm"] + "/seed" + str(worst_p1["seed"])
                               if worst_p1 else None),
            "met": bool(p1_measured >= P1_MIN_WAKING_TICKS),
        },
        {
            "name": "realized_pe_variance_in_band",
            "description": ("the design's anchor is 1/realized-PE-variance; a "
                            "degenerate rv makes the anchor meaningless"),
            "control": "every cell's own synthetic PE stream",
            "measured": float(p2_measured),
            "threshold_low": float(P2_PE_VAR_LOW),
            "threshold_high": float(P2_PE_VAR_HIGH),
            "comparator_low": ">",
            "comparator_high": "<",
            "direction": "interval",
            "offending_cell": p2_offending,
            "met": bool(pe_pairs
                        and P2_PE_VAR_LOW < p2_measured < P2_PE_VAR_HIGH),
        },
        {
            "name": "cold_start_cycle_observed",
            "description": ("the pre-loop agent.reset() must actually fire a "
                            "zero-waking-tick REM entry (C0); without it D1 "
                            "has nothing to discriminate"),
            "control": "enter_rem calls counted from agent construction",
            "measured": float(p3_measured),
            "threshold": float(P3_EXPECTED_C0_CALLS),
            "direction": "lower",
            "offending_cell": (worst_p3["arm"] + "/seed" + str(worst_p3["seed"])
                               if worst_p3 else None),
            "met": bool(p3_measured >= P3_EXPECTED_C0_CALLS),
        },
    ]
    all_met = all(p["met"] for p in preconditions)

    # Non-degeneracy. D2 is pre-registered degenerate in this regime -- see the
    # module docstring. The others are only degenerate if they could not vary.
    targets_all = [t for r in rows for t in r["targets"]]
    varied = (len(set(round(t, 6) for t in targets_all)) > 1)
    criteria_non_degenerate = {
        "D1": bool(varied and len(set(r["c0_captured"] for r in rows)) > 1),
        # False BY PRE-REGISTRATION: the 2x band cannot separate the arms when
        # the sentinel (2.0) sits within 2x of the anchor (~3.24), as measured.
        "D2": bool(not crit["d2_degenerate_both_arms_pass"]),
        "D3": bool(varied),
        "D4": bool(varied),
        "D5": bool(varied),
    }

    if not all_met:
        label = "substrate_not_ready_requeue"
    elif crit["d4_falsified"]:
        label = "mech204_option_a_falsified_demote"
    elif crit["overall_pass"]:
        label = "f1_coldstart_guard_validated"
    else:
        label = "f1_coldstart_guard_partial"

    return {
        "label": label,
        "preconditions": preconditions,
        "criteria_non_degenerate": criteria_non_degenerate,
        "combination_rule": ("PASS = D1 AND D2 AND D4_not_falsified AND D5. "
                             "D2 is recorded but pre-registered as "
                             "NON-DISCRIMINATING in this regime "
                             "(criteria_non_degenerate.D2=false); D5 carries "
                             "the discriminating information. D3 is reported "
                             "only."),
        "criteria": [
            {"name": "D1_cold_start_no_fire", "load_bearing": True,
             "passed": bool(crit["d1_pass"]),
             "measured": float(min(crit["d1_seeds_ok_per_arm"].values())
                               if crit["d1_seeds_ok_per_arm"] else 0),
             "threshold": float(D1_MIN_SEEDS)},
            {"name": "D2_within_anchor_band", "load_bearing": False,
             "passed": bool(crit["d2_pass"]),
             "measured": float(crit["d2_on_cells_ok"]),
             "threshold": float(len([r for r in rows if r["guard"]]))},
            {"name": "D3_rv_toward_realized_pe_variance", "load_bearing": False,
             "passed": None,
             "threshold_not_applicable": ("reported only; the step=0.0 arm "
                                          "cannot move rv by construction")},
            {"name": "D4_falsifier_monotone_away", "load_bearing": True,
             "passed": bool(crit["d4_pass_not_falsified"]),
             "measured": float(max(crit["d4_monotone_away_seeds_per_arm"].values())
                               if crit["d4_monotone_away_seeds_per_arm"] else 0),
             "threshold": float(D4_MIN_SEEDS_FALSIFIED)},
            {"name": "D5_cross_arm_anchor_gap_contrast", "load_bearing": True,
             "passed": bool(crit["d5_pass"]),
             "measured": float(max(
                 (d["seeds_meeting_separation"]
                  for d in crit["d5_per_step"].values()), default=0)),
             "threshold": float(D5_MIN_SEEDS)},
        ],
    }


def _flat_scalar(rows: list, crit: dict, interp: dict) -> dict:
    """Flat dict of SCALARS the verdict turns on (booleans as 0/1, no NaN)."""
    out = {
        "d1_pass": int(bool(crit["d1_pass"])),
        "d2_pass": int(bool(crit["d2_pass"])),
        "d2_degenerate_both_arms_pass": int(bool(
            crit["d2_degenerate_both_arms_pass"])),
        "d4_falsified": int(bool(crit["d4_falsified"])),
        "d5_pass": int(bool(crit["d5_pass"])),
        "overall_pass": int(bool(crit["overall_pass"])),
        "n_cells": len(rows),
    }
    seps = [d["mean_separation"] for d in crit["d5_per_step"].values()
            if d["mean_separation"] == d["mean_separation"]]
    if seps:
        out["d5_max_mean_separation"] = float(max(seps))
    anchors = [r["anchor_precision"] for r in rows
               if r["anchor_precision"] == r["anchor_precision"]]
    if anchors:
        out["anchor_precision_mean"] = float(sum(anchors) / len(anchors))
    for p in interp["preconditions"]:
        v = p.get("measured")
        if isinstance(v, (int, float)) and not isinstance(v, bool) \
                and v == v and abs(v) != float("inf"):
            out["precondition_" + p["name"]] = float(v)
    return out


def main(dry_run: bool = False):
    seeds = (SEEDS[0],) if dry_run else SEEDS
    episodes_per_run = 2 if dry_run else EPISODES_PER_RUN
    steps_per_episode = 50 if dry_run else STEPS_PER_EPISODE

    t0 = time.perf_counter()
    rows: list = []
    for step_label, step in RECAL_STEPS:
        for guard_label, guard in GUARD_LEVELS:
            for seed in seeds:
                rows.append(run_cell(step_label, step, guard_label, guard,
                                     seed, episodes_per_run, steps_per_episode))
    elapsed = time.perf_counter() - t0

    crit = _aggregate(rows)
    interp = _build_interpretation(rows, crit)
    outcome = "PASS" if crit["overall_pass"] else "FAIL"

    print(
        f"V3-EXQ-541d MECH-204 F1 cold-start guard validation -- {outcome} "
        f"in {elapsed:.1f}s (label={interp['label']})",
        flush=True,
    )
    if dry_run:
        print("[--dry-run] smoke summary:", flush=True)
        print(json.dumps({
            "d1_pass": crit["d1_pass"],
            "d2_degenerate_both_arms_pass": crit["d2_degenerate_both_arms_pass"],
            "d4_falsified": crit["d4_falsified"],
            "preconditions": [
                {"name": p["name"], "measured": p.get("measured"),
                 "met": p["met"]} for p in interp["preconditions"]],
            "c0": [{"arm": r["arm"], "calls": r["c0_enter_rem_calls"],
                    "captured": r["c0_captured"],
                    "persistent": r["c0_persistent_zero_point"]}
                   for r in rows],
            "enter_rem_per_cycle": sorted(set(
                n for r in rows for n in r["n_enter_rem_calls_per_cycle"])),
        }, indent=1), flush=True)

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{ts}_v3"
    full_config = {
        "seeds": list(seeds),
        "recal_steps": [{"label": l, "step": s} for l, s in RECAL_STEPS],
        "guard_levels": [{"label": l, "guard": g} for l, g in GUARD_LEVELS],
        "episodes_per_run": episodes_per_run,
        "steps_per_episode": steps_per_episode,
        "sleep_loop_K": SLEEP_LOOP_K,
        "precision_zero_point_ema_alpha": PRECISION_ZERO_POINT_EMA_ALPHA,
        "env_kwargs": _env_kwargs(),
    }
    manifest = {
        "schema_version": "v1",
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": ts,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "claim_ids": CLAIM_IDS,
        "backlog_id": BACKLOG_ID,
        "outcome": outcome,
        "result": outcome,
        "evidence_direction": "non_contributory",
        "sleep_driver_pattern": "K=1 single-fire (SleepLoopManager, fires every episode)",
        "criteria": crit,
        "interpretation": interp,
        "arm_results": rows,
        "registered_thresholds": {
            "D1_MIN_SEEDS": D1_MIN_SEEDS,
            "D2_ANCHOR_BAND_LOG2": D2_ANCHOR_BAND_LOG2,
            "D4_MONOTONE_CYCLES": D4_MONOTONE_CYCLES,
            "D4_MIN_SEEDS_FALSIFIED": D4_MIN_SEEDS_FALSIFIED,
            "D5_MIN_SEPARATION": D5_MIN_SEPARATION,
            "D5_MIN_SEEDS": D5_MIN_SEEDS,
            "N_CONTRAST_CYCLES": N_CONTRAST_CYCLES,
            "DEFENSIBLE_STEPS": list(DEFENSIBLE_STEPS),
            "E3_SENTINEL_PRECISION": E3_SENTINEL_PRECISION,
        },
        "readout": _flat_scalar(rows, crit, interp),
        "diagnostics": {
            # NON-GATING per-arm view of the quantity the P2 band guards, so a
            # near-bound arm is visible on PASS runs too rather than only in
            # autopsy. The band itself stays a readiness precondition scored on
            # the worst cell; this is the sibling-partition view.
            "realized_pe_variance_per_arm": {
                arm: {
                    "values": [r["realized_pe_variance"] for r in rs],
                    "min": min(r["realized_pe_variance"] for r in rs),
                    "max": max(r["realized_pe_variance"] for r in rs),
                    "band_low": P2_PE_VAR_LOW,
                    "band_high": P2_PE_VAR_HIGH,
                }
                for arm, rs in (
                    lambda d: d
                )({a: [r for r in rows if r["arm"] == a]
                   for a in sorted(set(r["arm"] for r in rows))}).items()
            },
            "anchor_precision_per_arm": {
                a: [r["anchor_precision"] for r in rows if r["arm"] == a]
                for a in sorted(set(r["arm"] for r in rows))
            },
            "enter_rem_calls_per_cycle_observed": sorted(set(
                n for r in rows for n in r["n_enter_rem_calls_per_cycle"])),
        },
        "notes": (
            "MECH-204 F1 cold-start guard validation. Second factor "
            "precision_zero_point_require_waking (OFF/ON) crossed with 541c's "
            "recal-step arms, same env/seeds/cycle count. ANCHOR RE-ANCHORED "
            "to this harness by the rule the design gives "
            "(anchor = 1/realized-PE-variance, measured in-run ~3.24 here vs "
            "~255 in the IGW-20260915-243 realized-PE setting) per user "
            "decision 2026-09-19T22:12:50Z. D2 is PRE-REGISTERED as "
            "non-discriminating in this regime and flagged "
            "criteria_non_degenerate.D2=false: the 2x band cannot separate "
            "the arms because the precision_init sentinel (2.0) sits within "
            "2x of the anchor here, unlike IGW-243 where it is 127x away. D5 "
            "carries the discriminating information. DIAGNOSTIC, not "
            "evidence: MECH-204 is not answerable until "
            "sd_waking_confidence_inflation_headroom is validated "
            "(V3-EXQ-794a); this validates an INSTRUMENT. This base cannot "
            "exercise the second (double-enter_rem) defect instance -- the "
            "541c driver issues exactly one enter_rem per cycle; counts are "
            "recorded so the absence is auditable."
        ),
    }

    if dry_run:
        print("[--dry-run] manifest not written.", flush=True)
        return None

    out_dir = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments"
    out_path = write_flat_manifest(
        manifest,
        out_dir,
        dry_run=False,
        config=full_config,
        seeds=SEEDS,
        script_path=Path(__file__),
        started_at=t0,
    )
    print(f"Result written to: {out_path}", flush=True)
    return outcome, out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Smoke run, no manifest.")
    args = parser.parse_args()
    result = main(dry_run=args.dry_run)
    if args.dry_run or result is None:
        sys.exit(0)
    _outcome, _out_path = result
    emit_outcome(
        outcome=_outcome if _outcome in ("PASS", "FAIL") else "FAIL",
        manifest_path=_out_path,
        dry_run=args.dry_run,
    )
    sys.exit(0)
