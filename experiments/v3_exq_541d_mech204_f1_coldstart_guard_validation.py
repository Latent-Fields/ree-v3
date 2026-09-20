#!/opt/local/bin/python3
"""V3-EXQ-541d -- MECH-204 F1 cold-start guard validation on a REALISED-PE base

!! NOT QUEUED. DO NOT QUEUE THIS SCRIPT AS IT STANDS. SECOND red-team BLOCKING. !!
================================================================================
RED-TEAM VERDICT: **BLOCKING** (fable, 2026-09-20, /queue-experiment Step 4.5).
This is the SECOND blocking refusal on this item -- the first was the 541c
synthetic-PE build (see the record below). Both findings were verified against
this driver's own measured data before being accepted.

  G1  THE PRE-REGISTERED FALSIFIER IS ARITHMETICALLY UNFIRABLE -- ON ANY BASE.
      E4 asks whether |target - anchor| GROWS monotonically over ~10 cycles.
      But the F1 target starts at the sentinel (2.0) or at the first honest
      precision, both ~2-4, while anchor = 1/realized-PE-variance ~= 260. The
      target is an EMA of p_k = 1/rv_k, and rv falls as the world model learns,
      so the target RISES -- measured monotone increasing in every arm
      (OFF 2.12 -> 5.58, ON 3.16 -> 7.45), every value far BELOW the anchor.
      gap = anchor - target therefore SHRINKS by construction. For it to grow,
      rv would have to RISE across cycles, which needs realized PE > rv (~0.15,
      i.e. ~40x the measured 0.0038) for nine consecutive cycles.
      The `mech204_option_a_falsified_demote` branch is unreachable.
      THIS IS A DEFECT IN THE CHIP'S FALSIFIER TEXT, NOT IN THE BASE: the chip
      says the TARGET climbs away from 1/realized-PE-variance, but the
      IGW-20260915-243 measurement it was written from says the opposite --
      the target climbs 2.0 -> 27 -> 49 -> 69 -> 88, which is climbing TOWARD
      255. What IGW-243 recorded as moving AWAY is rv ("recalibration pushes rv
      AWAY from calibration, 0.0039 -> 0.0121"). Target and rv were conflated.
      So NO choice of base can make the falsifier-as-written fire, and the
      user's base-selection criterion ("pick the base that lets the
      pre-registered falsifier actually FIRE") cannot be satisfied as stated.

  G2  E2 DOES NOT CERTIFY WHAT IT CLAIMS, AND THIS DOCSTRING'S ORIGINAL
      RATIONALE FOR IT WAS WRONG. An earlier draft of this file argued that
      the realised base breaks the 541c closed form because "precision feeds
      selection feeds prediction error feeds precision". MEASURED: FALSE. The
      per-episode realized PE is BIT-IDENTICAL between the two guard arms at
      every episode (0.003832, 0.003786, 0.003852, ... in both), and so is the
      anchor (259.904 both). The guard changes rv; rv never reaches an action;
      the trajectories do not diverge. What actually makes p_j arm-dependent is
      the WRITEBACK's rv change surviving into the next REM entry, which
      depends on ticks-per-episode as 0.95^n. With n ~ 10 the carry-over is
      ~0.60 and E2 reads 4-10x; with n ~ 200 it is 3.5e-5 and E2 reads 1.00 and
      FAILS. So E2's verdict is a function of EPISODE LENGTH.

  G3  THE RUN IS NOT IN THE REGIME IT DECLARES. This driver specifies 30
      episodes x 200 steps = 6000 waking ticks per cell. Measured at grid 12:
      episodes END BY DEATH in 6-15 ticks (agent_health <= 0), so the real
      figure is ~300. The `min_waking_ticks_at_post_c0_rem_entry` precondition
      read 7 and was interpreted as "producer live -- MET"; it was in fact
      reporting the episode length. V3-EXQ-794, at this same nominal operating
      point, recorded rv_final ~0.0054 (rv tracking PE), which needs ~60-100
      ticks per episode -- so either 794's agents survived and this one does
      not, or the operating points differ on an axis neither script records.
      Nothing in the manifest records episode length.

WHAT WAS NONETHELESS ESTABLISHED, and it is substantive:
  - The anchor re-measured on this base is 259.9-264.3 (realized PE variance
    0.00380-0.00385). The chip's "~255 in the IGW-20260915-243 setting"
    TRANSFERS; 541c's 2.148 does not and is used nowhere here.
  - The guard works exactly as specified, in all six arms: C0 under guard OFF
    captures at target 1.999996000008 (the precision_init sentinel, exactly);
    under guard ON it does not capture at all.
  - THE UNSCORED RESULT THAT MATTERS: the WRITEBACK moves rv AWAY from the
    realized PE variance on essentially every cycle of BOTH arms (OFF 9/9,
    ON 8/9), with rv sitting 18-82x ABOVE realized PE throughout. That is
    IGW-243's finding reproduced and generalised, and it is the SUBSTANCE of
    the falsifier -- expressed on rv, the quantity IGW-243 actually measured,
    rather than on the target. Re-expressed that way the falsifier is both
    firable AND appears already SATISFIED, which would route MECH-204 Option A
    to DEMOTE. Changing the falsifier is a user decision, not this session's.

Escalated as decision chip chip-20260920-exq541d-falsifier-misspecified.
Record: REE_assembly evidence/planning/
        exq541d_redteam_blocking_refusal_staged_20260919.md (section 6).
================================================================================

SLEEP DRIVER: K=1 single-fire (SleepLoopManager, sleep_loop_episodes_K=1, fires every episode)

red-team: see queue entry note (recorded at queue time).

WHAT IS UNDER TEST
------------------
The MECH-204 F1 cold-start guard landed 2026-09-18 (ree-v3 e1ff0927):
SerotoninConfig.precision_zero_point_require_waking. When True, a REM entry
with no waking tick since the last capture does NOT touch
_persistent_zero_point. Default False -> bit-identical OFF.
Substrate record: docs/substrate/MECH-204-f1-coldstart-guard.md
Contracts: tests/contracts/test_mech204_f1_coldstart_guard.py C1-C8.

THE DEFECT. The canonical pre-waking `agent.reset()` fires a full sleep cycle
with ZERO waking ticks (call it C0). At C0's REM entry E3 has never seen a
prediction error, so current_precision is the precision_init sentinel --
precision_init is a VARIANCE (0.5, config.py:1126), so the sentinel precision
is 1/(0.5+1e-6) = 1.999996. Guard OFF anchors the F1 persistent reference
there; every later cycle then blends toward the honest precision at only
alpha=0.1, and the WRITEBACK consumer meanwhile drags rv toward 1/2.0 = 0.5,
AWAY from the agent's realized prediction-error variance.

WHY THIS DRIVER AND NOT V3-EXQ-541c's (user decision, 2026-09-19T23:52:40Z)
--------------------------------------------------------------------------
A first version of 541d was built on 541c and REFUSED at red-team (BLOCKING);
the full record is REE_assembly evidence/planning/
exq541d_redteam_blocking_refusal_staged_20260919.md. Two findings, both
verified, and both are properties of 541c's harness rather than of the guard:

  (i)  541c takes RANDOM actions (`rng.randrange`) and drives E3's running
       variance from a STATIONARY SYNTHETIC stream, so the agent's state can
       not influence the trajectory. Both guard arms therefore see an
       identical per-cycle honest precision p_j, and the cross-arm target
       difference collapses to the closed form
           OFF_k - ON_k == 0.9**k * (SENTINEL - p_1)
       which reproduced to a max absolute error of 8.07e-05 over 16 cycles.
       Any "separation" scored there restates whether C0 captured.
  (ii) With no realized precision, the chip's pre-registered FALSIFIER -- does
       the target still climb AWAY from 1/realized-PE-variance? -- has nothing
       that can drift, so it cannot fire under any outcome.

This driver moves to the REALISED-PE base the user selected: the canonical
StepHarness loop at the IGW-20260915-243 operating point (K=1 sleep, F1 recal
step 0.25), matching V3-EXQ-794's substrate operating point. Here E3's running
variance is fed by the REAL forward-model error
(`e3_selector.post_action_update`: prediction_error = actual_z_world -
predicted_world, recorded as the mean SQUARED error, i.e. a variance), and the
agent SELECTS its actions through E3 -- so precision feeds selection feeds
prediction error feeds precision. That closed loop is what breaks (i).

The alternative the user named -- the v3_exq_sd068_* multi-REM family -- was
examined and REJECTED on the user's own criterion. It never touches the F1 /
_persistent_zero_point / mech204 path at all, bypasses SleepLoopManager (so no
mech204_* WRITEBACK metrics exist), reaches REM only through an unscored
`drive_liveness_pass` wrapped in a bare `except Exception`, and its precision
target is CLAMP-PINNED (`max(1e-3, raw_target)`; V3-EXQ-778c recorded
`target_clamped` 1.0 with calibration_error pinned at the constant
998.5009992509989, "degenerate at both rails"). A clamp-pinned precision is
precisely not "realised and able to drift". Its multi-enter_rem property is
still worth testing and is recorded as owed follow-on, not folded in here.

MEASURED ON THIS BASE BEFORE QUEUEING (the user's pre-queue requirement).
Traced probe, seed 42, grid 8, 10 cycles, 150 steps/ep, recal step 0.25:
  realized PE variance   = 0.003848  ->  ANCHOR = 1/0.003848 = 259.9
  (the chip's "~255 in the IGW-20260915-243 setting" TRANSFERS; 541c's 2.148
   does NOT and is not used anywhere in this driver)
  C0, guard OFF: fires, target = 2.0 EXACTLY (the raw sentinel)
  C0, guard ON : does not fire, _persistent_zero_point stays None
  cycle-1 OFF target 2.11593 == 0.9*2.0 + 0.1*(1/0.316525), exact
  IDENTITY TEST -- actual (OFF_k - ON_k) vs the 541c closed form:
      k=1 1.00x, k=3 1.18x, k=5 1.65x, k=7 2.75x, k=9 4.17x (no drift)
      k=1 1.00x, k=3 1.28x, k=5 2.15x, k=7 3.82x, k=9 8.24x (drift LO)
    -- i.e. the two arms' rv DIVERGE from cycle 2 (0.2494 vs 0.2222) and the
    departure GROWS. On 541c this same test agreed to 8e-5 at every cycle.
    THE IDENTITY IS BROKEN ON THIS BASE. That is what E2 pre-registers.
  final rv, guard ON / guard OFF = 0.755 (no drift), 0.742 (drift LO)
    -- the guard leaves rv ~25% closer to the realized PE variance.

WHY A DRIFT SOURCE IS ARMED, AND WHY THAT IS NOT A NEW DESIGN CHOICE
--------------------------------------------------------------------
On the realised base E3.update_running_variance maintains rv as a SYMMETRIC
EMA of true prediction error, so rv ~= true prediction error BY CONSTRUCTION
(V3-EXQ-794's docstring; V3-EXQ-774 FAILed on exactly this tautology). A
precision that tracks reality by construction is realised but CANNOT DRIFT --
and the user's criterion is "realised AND able to drift". The only drift
source in the substrate is SD-076 waking confidence inflation
(E3Config.use_waking_confidence_inflation), which is also what MECH-204's own
what_would_answer (a) names as the precondition for answering the claim.
It is armed here with the SD-076 HEADROOM REPAIR, not 794's clamped floor:
  waking_confidence_rv_floor_relative_frac = 0.2   (relative, not absolute)
  waking_confidence_rv_floor_mode          = "soft" (strictly monotone)
V3-EXQ-794 was refused because its ABSOLUTE floor 0.01 sat 1.8x above the
operating point (rv 0.005420) and clamped on the first tick: rv_final was
EXACTLY 0.010000 on all four inflation arms and overconfidence_score was
bit-identical to 15 significant figures across LO and HI. E5 below exists to
detect a recurrence of that saturation rather than assume it is cured.
STATED DEPENDENCY: `sd_waking_confidence_inflation_headroom` is `implemented`
with ready FALSE and its own validation (V3-EXQ-794a) not yet queued, so this
run rides on an unvalidated repair. That is recorded, not hidden.

PRE-REGISTERED CRITERIA
-----------------------
  E1  COLD-START CONTRACT (reported, NOT load-bearing). Guard ON: C0 does not
      capture. Guard OFF: C0 captures AND its target equals the sentinel
      1.999996 within E1_SENTINEL_TOL. Deliberately not load-bearing: at C0
      `_waking_ticks_since_capture == 0` by construction, so this restates the
      guard's own predicate and duplicates contracts C1-C8. The sentinel-value
      half is the part those contracts do not assert.
  E2 (load-bearing) THE CONTRAST IS A MEASUREMENT, NOT AN IDENTITY. For each
      matched (drift, seed), compare the observed cross-arm target gap against
      the 541c closed form 0.9**k * (SENTINEL - p_1). Require the observed gap
      to DEPART from it by at least E2_MIN_DEPARTURE_RATIO at some cycle in
      the first E2_WINDOW. This is the criterion whose absence made the 541c
      design unqueueable; it certifies the base, from the run's own data.
  E3 (load-bearing) CALIBRATION DIRECTION. Guard ON must leave post-WRITEBACK
      rv strictly CLOSER to the run's own realized PE variance than guard OFF
      at matched (drift, seed): ratio <= E3_MAX_RV_RATIO in >= E3_MIN_SEEDS.
  E4 (load-bearing) FALSIFIER OF THE WHOLE FIX. With the guard ON, does
      |target - anchor| grow MONOTONICALLY over the first E4_WINDOW cycles?
      If yes in >= E4_MIN_SEEDS at any drift level, the precision_init
      sentinel was not the (only) cause, precision-at-REM-entry is not a
      calibration-relevant quantity, and MECH-204 Option A needs REDESIGN.
      Per the chip: that outcome DEMOTES Option A, it does not retune it.
  E5  DOSE NON-SATURATION (reported). LO and HI must not be bit-identical --
      the V3-EXQ-794 saturation signature.
PASS = E2 AND E3 AND E4_not_falsified.

EXPERIMENT_PURPOSE is "diagnostic", NOT "evidence". MECH-204's what_would_
answer (a) holds the claim NOT ANSWERABLE until
sd_waking_confidence_inflation_headroom is validated (V3-EXQ-794a). 541d
validates an INSTRUMENT (the cold-start guard); it does not adjudicate
MECH-204. Inheriting 541c's "evidence" tag is the single easiest error here.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.optim as optim

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiment_protocol import emit_outcome  # noqa: E402
from experiments._harness import StepHarness  # noqa: E402
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402
from experiments._lib.arm_fingerprint import arm_cell  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_541d_mech204_f1_coldstart_guard_validation"
CLAIM_IDS = ["MECH-204"]
EXPERIMENT_PURPOSE = "diagnostic"
BACKLOG_ID = "EXP-0171"

# Both anchor-kind readiness preconditions are DIRECT COUNTER READS with a >=1
# gate -- P1 reads SerotoninModule._waking_ticks_since_capture (incremented
# once per REEAgent.sense() call) and P3 reads the enter_rem call count.
# Neither is a hand-written signature predicate that could be narrower than the
# state it anchors to. Measured directly on this base (seed 42): P1 = 9 at the
# first post-C0 REM entry, P3 = 1.
ANCHOR_REACHABILITY_EXEMPT = (
    "P1/P3 are direct counter reads gated at >=1, not scored signatures; "
    "reachability measured on this base (P1=9, P3=1)"
)

# ---- Substrate operating point (IGW-20260915-243 / V3-EXQ-794; held constant) ----
GRID_SIZE = 12
STEPS_PER_EP = 200
N_TRAIN_EPS = 30          # K=1 -> 30 sleep cycles; claims.yaml floor is >= 16
LR = 5e-4
SWS_CONSOLIDATION_STEPS = 8
REM_ATTRIBUTION_STEPS = 6
ALPHA_WORLD = 0.9
ALPHA_SELF = 0.3
PRECISION_ZERO_POINT_EMA_ALPHA = 0.1
REM_PRECISION_RECALIBRATION_STEP = 0.25   # IGW-243's "F1 recal step 0.25"

# E3 cold-start sentinel: precision_init is a VARIANCE (config.py:1126).
E3_PRECISION_INIT_VARIANCE = 0.5
E3_SENTINEL_PRECISION = 1.0 / (E3_PRECISION_INIT_VARIANCE + 1e-6)  # ~1.999996

# ---- Factors ----
# Seed 44 is excluded on a reef-config env (recurring early-death instability,
# EXQ-539-540 / V3-EXQ-538a); 45 is the sanctioned substitute.
SEEDS = (42, 43, 45)
GUARD_LEVELS: Tuple[Tuple[str, bool], ...] = (("guard_off", False), ("guard_on", True))
# SD-076 asymmetry. None = master flag OFF (bit-identical symmetric path);
# deliberately NOT expressed as 0.0, because the ON path additionally applies
# the rv floor and so ON-at-0.0 is a different computation (V3-EXQ-794).
DRIFT_LEVELS: Tuple[Tuple[str, Optional[float]], ...] = (
    ("drift_off", None), ("drift_lo", 0.6), ("drift_hi", 0.8),
)
DEFENSIBLE_DRIFTS = ("drift_lo", "drift_hi")
# SD-076 headroom repair (NOT 794's clamped absolute floor).
INFLATION_RV_FLOOR_RELATIVE_FRAC = 0.2
INFLATION_RV_FLOOR_MODE = "soft"

# ---- Pre-registered thresholds (constants; NOT derived from this run) ----
E1_SENTINEL_TOL = 1e-3        # |C0 target - sentinel| under guard OFF
E2_WINDOW = 10                # cycles scored for the identity-departure test
E2_MIN_DEPARTURE_RATIO = 1.5  # observed gap / closed-form gap; measured 4.17-8.24
E2_MIN_SEEDS = 2              # in >= 2/3 seeds
E3_MAX_RV_RATIO = 0.95        # ON rv-distance / OFF rv-distance; measured ~0.75
E3_MIN_SEEDS = 2
E4_WINDOW = 10                # falsifier window ("first ~10 cycles")
E4_MIN_SEEDS = 2              # >= 2/3 seeds to call the fix falsified
E5_MIN_DOSE_SEPARATION = 1e-9 # LO vs HI bit-identical = 794's saturation signature

# Readiness precondition bounds.
P1_MIN_WAKING_TICKS = 1
P2_PE_VAR_LOW = 1e-4          # realized PE variance band -> anchor is usable
P2_PE_VAR_HIGH = 1e-1
P3_EXPECTED_C0_CALLS = 1
P4_MIN_DRIFT_EFFECT = 1e-6    # drift must actually move rv (anti-clamp, vs 794)


def _env_kwargs(dry_run: bool = False) -> dict:
    return dict(
        size=(8 if dry_run else GRID_SIZE),
        num_hazards=3,
        num_resources=3,
        hazard_harm=0.04,
        proximity_harm_scale=0.12,
        proximity_benefit_scale=0.10,
        use_proxy_fields=True,
        resource_respawn_on_consume=True,
    )


def cell_config_slice(guard: bool, asym: Optional[float], n_train: int,
                      steps: int, dry_run: bool) -> dict:
    """Everything this cell's computation reads. Declared for the fingerprint."""
    return {
        "env_kwargs": _env_kwargs(dry_run),
        "grid_size": (8 if dry_run else GRID_SIZE),
        "n_train_eps": n_train,
        "steps_per_ep": steps,
        "lr": LR,
        "sleep_loop_K": 1,
        "sws_consolidation_steps": SWS_CONSOLIDATION_STEPS,
        "rem_attribution_steps": REM_ATTRIBUTION_STEPS,
        "alpha_world": ALPHA_WORLD,
        "alpha_self": ALPHA_SELF,
        "precision_zero_point_ema_alpha": PRECISION_ZERO_POINT_EMA_ALPHA,
        "rem_precision_recalibration_step": REM_PRECISION_RECALIBRATION_STEP,
        "precision_zero_point_require_waking": bool(guard),
        "use_waking_confidence_inflation": asym is not None,
        "waking_confidence_inflation_asymmetry": (float(asym) if asym is not None else 0.0),
        "waking_confidence_rv_floor_relative_frac": (
            INFLATION_RV_FLOOR_RELATIVE_FRAC if asym is not None else 0.0),
        "waking_confidence_rv_floor_mode": (
            INFLATION_RV_FLOOR_MODE if asym is not None else "hard"),
        "self_dim": 32,
        "world_dim": 32,
        "tonic_5ht_enabled": True,
        # Scoring constants the cell's recorded readouts are computed under.
        "e2_window": E2_WINDOW,
        "e4_window": E4_WINDOW,
    }


def _make_env(seed: int, dry_run: bool = False) -> CausalGridWorldV2:
    return CausalGridWorldV2(seed=seed, **_env_kwargs(dry_run))


def _make_agent(env: CausalGridWorldV2, guard: bool,
                asym: Optional[float]) -> REEAgent:
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=32,
        world_dim=32,
        alpha_world=ALPHA_WORLD,
        alpha_self=ALPHA_SELF,
        sws_enabled=True,
        sws_consolidation_steps=SWS_CONSOLIDATION_STEPS,
        rem_enabled=True,
        rem_attribution_steps=REM_ATTRIBUTION_STEPS,
        use_sleep_loop=True,
        sleep_loop_episodes_K=1,
        use_rem_precision_recalibration=True,
        precision_zero_point_ema_alpha=PRECISION_ZERO_POINT_EMA_ALPHA,
        rem_precision_recalibration_step=REM_PRECISION_RECALIBRATION_STEP,
    )
    # Factor A: the MECH-204 F1 cold-start guard (the lever under test).
    cfg.serotonin.precision_zero_point_require_waking = bool(guard)
    # Factor B: SD-076 waking confidence inflation -- the DRIFT SOURCE, armed
    # with the headroom repair (relative + soft floor), never 794's absolute one.
    cfg.e3.use_waking_confidence_inflation = asym is not None
    cfg.e3.waking_confidence_inflation_asymmetry = (
        float(asym) if asym is not None else 0.0)
    if asym is not None:
        cfg.e3.waking_confidence_rv_floor_relative_frac = INFLATION_RV_FLOOR_RELATIVE_FRAC
        cfg.e3.waking_confidence_rv_floor_mode = INFLATION_RV_FLOOR_MODE
    # Tonic 5-HT must be on for compute_recalibration_target() to be meaningful.
    cfg.serotonin.tonic_5ht_enabled = True
    return REEAgent(cfg)


def run_cell(guard_label: str, guard: bool, drift_label: str,
             asym: Optional[float], seed: int, n_train: int, steps: int,
             dry_run: bool) -> dict:
    arm_label = f"{guard_label}__{drift_label}"
    print(f"Seed {seed} Condition {arm_label}", flush=True)

    slice_ = cell_config_slice(guard, asym, n_train, steps, dry_run)
    with arm_cell(
        seed,
        config_slice=slice_,
        script_path=Path(__file__),
        config_slice_declared=True,
        include_driver_script_in_hash=False,
        extra_ineligible_reasons=["shared_optimizer_across_episodes"],
    ) as cell:
        env = _make_env(seed, dry_run)
        agent = _make_agent(env, guard, asym)
        ser = agent.serotonin
        optimizer = optim.Adam(agent.parameters(), lr=LR)

        # --- Instrument BEFORE the first reset: the pre-loop agent.reset()
        # fires the zero-waking-tick cold-start cycle C0, and instrumenting
        # after it would make C0 invisible -- which IS the defect. ---
        trace = {"calls": 0, "suppressed": 0, "ticks_at_entry": []}
        _orig_enter_rem = ser.enter_rem

        def _traced_enter_rem(current_precision):
            trace["calls"] += 1
            before = ser._persistent_zero_point
            trace["ticks_at_entry"].append(int(ser._waking_ticks_since_capture))
            _orig_enter_rem(current_precision=current_precision)
            if ser._persistent_zero_point == before:   # covers None == None
                trace["suppressed"] += 1
        ser.enter_rem = _traced_enter_rem

        harness = StepHarness(agent, env, train_mode=True, seed=seed)
        cycles: List[dict] = []
        pe_all: List[float] = []
        c0: Optional[dict] = None

        for ep in range(n_train):
            calls_before = trace["calls"]
            agent.reset()     # fires the sleep cycle for the prior episode (K=1)
            st = agent.sleep_loop.state if agent.sleep_loop else None
            m = dict(st.last_metrics) if (st and st.last_metrics) else {}
            rec = {
                "episode": ep,
                "n_enter_rem_calls": trace["calls"] - calls_before,
                "fired": float(m.get("mech204_recalibration_fired", 0.0)),
                "target": (float(m["mech204_recalibration_target"])
                           if "mech204_recalibration_target" in m else None),
                "rv_before": (float(m["mech204_running_variance_before"])
                              if "mech204_running_variance_before" in m else None),
                "rv_after": (float(m["mech204_running_variance_after"])
                             if "mech204_running_variance_after" in m else None),
                "persistent_zero_point": (
                    None if ser._persistent_zero_point is None
                    else float(ser._persistent_zero_point)),
            }
            if ep == 0:
                c0 = rec           # the cold-start cycle
            else:
                cycles.append(rec)
            if st:
                st.last_metrics = {}

            _, obs_dict = env.reset()
            harness.reset()
            ep_pe: List[float] = []
            for _ in range(steps):
                result = harness.step(obs_dict)
                optimizer.zero_grad()
                loss = agent.compute_prediction_loss()
                if loss.requires_grad:
                    loss.backward()
                    optimizer.step()
                pe = result.residue_metrics.get("e3_prediction_error")
                if pe is not None:
                    # Already the MEAN SQUARED forward error, i.e. a variance
                    # (e3_selector.post_action_update).
                    v = float(pe.detach()) if hasattr(pe, "detach") else float(pe)
                    ep_pe.append(v)
                    pe_all.append(v)
                obs_dict = result.next_obs_dict
                if result.done:
                    break
            if ep_pe and cycles:
                cycles[-1]["ep_mean_pe_variance"] = sum(ep_pe) / len(ep_pe)
            if (ep + 1) % 5 == 0 or ep + 1 == n_train:
                print(
                    f"  [train] arm={arm_label} seed={seed} ep {ep + 1}/{n_train} "
                    f"rv={float(agent.e3._running_variance):.6f}",
                    flush=True,
                )

        # --- The design's anchor rule, instantiated on THIS cell's own
        # realized forward-model error: anchor = 1 / realized-PE-variance. ---
        realized_pe_var = (sum(pe_all) / len(pe_all)) if pe_all else float("nan")
        anchor = (1.0 / realized_pe_var
                  if realized_pe_var == realized_pe_var and realized_pe_var > 1e-12
                  else float("nan"))
        final_rv = float(agent.e3._running_variance)

        targets = [c["target"] for c in cycles if c["target"] is not None]
        # E4: does |target - anchor| GROW monotonically over the window?
        gaps = [abs(t - anchor) for t in targets[:E4_WINDOW]] if anchor == anchor else []
        e4_monotone_away = bool(len(gaps) >= 2) and all(
            gaps[i + 1] > gaps[i] for i in range(len(gaps) - 1))

        post_ticks = trace["ticks_at_entry"][1:]
        row = {
            "arm": arm_label,
            "guard_label": guard_label,
            "guard": bool(guard),
            "drift_label": drift_label,
            "inflation_asymmetry": (float(asym) if asym is not None else None),
            "seed": seed,
            "n_cycles": len(cycles),
            "realized_pe_variance": float(realized_pe_var),
            "anchor_precision": float(anchor),
            "final_rv": final_rv,
            "final_rv_distance_to_realized": abs(final_rv - realized_pe_var),
            "c0_enter_rem_calls": int(c0["n_enter_rem_calls"]) if c0 else 0,
            "c0_captured": bool(c0 and c0["persistent_zero_point"] is not None),
            "c0_target": (c0["target"] if c0 else None),
            "c0_persistent_zero_point": (c0["persistent_zero_point"] if c0 else None),
            "targets": targets,
            "e4_monotone_away_from_anchor": e4_monotone_away,
            "n_enter_rem_calls_total": int(trace["calls"]),
            "n_enter_rem_suppressed": int(trace["suppressed"]),
            "n_enter_rem_calls_per_cycle": [c["n_enter_rem_calls"] for c in cycles],
            "min_waking_ticks_at_post_c0_rem_entry": (
                min(post_ticks) if post_ticks else 0),
            "cycle_records": cycles,
        }
        cell.stamp(row)

    # E1, per cell: guard ON must NOT capture at C0; OFF must, AT the sentinel.
    if guard:
        e1_ok = not row["c0_captured"]
    else:
        e1_ok = bool(
            row["c0_captured"]
            and row["c0_target"] is not None
            and abs(row["c0_target"] - E3_SENTINEL_PRECISION) <= E1_SENTINEL_TOL)
    row["e1_cold_start_contract_ok"] = bool(e1_ok)
    print(f"verdict: {'PASS' if e1_ok else 'FAIL'}", flush=True)
    return row


def _by(rows: List[dict], **kw) -> List[dict]:
    return [r for r in rows
            if all(r.get(k) == v for k, v in kw.items())]


def _aggregate(rows: List[dict]) -> dict:
    # ---- E1 (reported) ----
    e1_ok_cells = sum(1 for r in rows if r["e1_cold_start_contract_ok"])
    e1_pass = (e1_ok_cells == len(rows))

    # ---- E2: the observed cross-arm gap must DEPART from 541c's closed form ----
    e2_per_cell = {}
    for drift_label, _ in DRIFT_LEVELS:
        for seed in SEEDS:
            off = _by(rows, drift_label=drift_label, seed=seed, guard=False)
            on = _by(rows, drift_label=drift_label, seed=seed, guard=True)
            if not off or not on:
                continue
            o_t, n_t = off[0]["targets"], on[0]["targets"]
            if not o_t or not n_t:
                continue
            p1 = n_t[0]
            ratios = []
            for k in range(1, min(len(o_t), len(n_t), E2_WINDOW) + 1):
                closed = (0.9 ** k) * (E3_SENTINEL_PRECISION - p1)
                if abs(closed) < 1e-12:
                    continue
                ratios.append(abs((o_t[k - 1] - n_t[k - 1]) / closed))
            e2_per_cell[f"{drift_label}__seed{seed}"] = {
                "max_departure_ratio": (max(ratios) if ratios else float("nan")),
                "per_cycle_departure_ratio": ratios,
                "meets": bool(ratios and max(ratios) >= E2_MIN_DEPARTURE_RATIO),
            }
    e2_seeds_meeting = {}
    for drift_label, _ in DRIFT_LEVELS:
        e2_seeds_meeting[drift_label] = sum(
            1 for s in SEEDS
            if e2_per_cell.get(f"{drift_label}__seed{s}", {}).get("meets"))
    e2_pass = any(n >= E2_MIN_SEEDS for n in e2_seeds_meeting.values())

    # ---- E3: guard ON leaves rv closer to the realized PE variance ----
    e3_per_drift = {}
    for drift_label, _ in DRIFT_LEVELS:
        ratios, n_ok = [], 0
        for seed in SEEDS:
            off = _by(rows, drift_label=drift_label, seed=seed, guard=False)
            on = _by(rows, drift_label=drift_label, seed=seed, guard=True)
            if not off or not on:
                continue
            d_off = off[0]["final_rv_distance_to_realized"]
            d_on = on[0]["final_rv_distance_to_realized"]
            if d_off <= 1e-12:
                continue
            r = d_on / d_off
            ratios.append(r)
            if r <= E3_MAX_RV_RATIO:
                n_ok += 1
        e3_per_drift[drift_label] = {
            "per_seed_rv_distance_ratio": ratios,
            "mean_ratio": (sum(ratios) / len(ratios)) if ratios else float("nan"),
            "seeds_meeting": n_ok,
        }
    e3_pass = any(d["seeds_meeting"] >= E3_MIN_SEEDS for d in e3_per_drift.values())

    # ---- E4 FALSIFIER ----
    e4_per_drift = {}
    for drift_label, _ in DRIFT_LEVELS:
        on = _by(rows, drift_label=drift_label, guard=True)
        e4_per_drift[drift_label] = sum(
            1 for r in on if r["e4_monotone_away_from_anchor"])
    e4_falsified = any(n >= E4_MIN_SEEDS for n in e4_per_drift.values())
    e4_pass = not e4_falsified

    # ---- E5: LO vs HI must not be bit-identical (794's saturation signature) ----
    e5_separations = []
    for guard_label, guard in GUARD_LEVELS:
        for seed in SEEDS:
            lo = _by(rows, drift_label="drift_lo", seed=seed, guard=guard)
            hi = _by(rows, drift_label="drift_hi", seed=seed, guard=guard)
            if lo and hi:
                e5_separations.append(abs(lo[0]["final_rv"] - hi[0]["final_rv"]))
    e5_min_sep = min(e5_separations) if e5_separations else float("nan")
    e5_pass = bool(e5_separations and e5_min_sep > E5_MIN_DOSE_SEPARATION)

    overall_pass = bool(e2_pass and e3_pass and e4_pass)
    return {
        "e1_pass": e1_pass, "e1_ok_cells": e1_ok_cells, "n_cells": len(rows),
        "e2_pass": e2_pass, "e2_per_cell": e2_per_cell,
        "e2_seeds_meeting_per_drift": e2_seeds_meeting,
        "e3_pass": e3_pass, "e3_per_drift": e3_per_drift,
        "e4_pass_not_falsified": e4_pass, "e4_falsified": e4_falsified,
        "e4_monotone_away_seeds_per_drift": e4_per_drift,
        "e5_pass_dose_non_saturated": e5_pass,
        "e5_min_lo_hi_separation": e5_min_sep,
        "overall_pass": overall_pass,
    }


def _worst(rows: List[dict], key, lo=True):
    if not rows:
        return None, None
    r = (min if lo else max)(rows, key=key)
    return key(r), f"{r['arm']}/seed{r['seed']}"


def _build_interpretation(rows: List[dict], crit: dict) -> dict:
    on_rows = [r for r in rows if r["guard"]]
    p1_measured, p1_cell = _worst(
        on_rows, lambda r: r["min_waking_ticks_at_post_c0_rem_entry"])
    # P2 is a two-sided band; `met` is a conjunction over cells, so report the
    # WORST CELL w.r.t. the band (furthest from its geometric centre) -- if that
    # one is inside, every cell is.
    pe_rows = [r for r in rows
               if r["realized_pe_variance"] == r["realized_pe_variance"]]
    centre = math.sqrt(P2_PE_VAR_LOW * P2_PE_VAR_HIGH)
    if pe_rows:
        wr = max(pe_rows, key=lambda r: abs(
            math.log(max(r["realized_pe_variance"], 1e-12) / centre)))
        p2_measured, p2_cell = wr["realized_pe_variance"], f"{wr['arm']}/seed{wr['seed']}"
    else:
        p2_measured, p2_cell = float("nan"), None
    p3_measured, p3_cell = _worst(rows, lambda r: r["c0_enter_rem_calls"])
    # P4 -- anti-794: the drift source must actually MOVE rv, not sit clamped.
    p4_deltas = []
    for guard_label, guard in GUARD_LEVELS:
        for seed in SEEDS:
            off = _by(rows, drift_label="drift_off", seed=seed, guard=guard)
            for dl in DEFENSIBLE_DRIFTS:
                dr = _by(rows, drift_label=dl, seed=seed, guard=guard)
                if off and dr:
                    p4_deltas.append(abs(dr[0]["final_rv"] - off[0]["final_rv"]))
    p4_measured = min(p4_deltas) if p4_deltas else float("nan")

    preconditions = [
        {
            "name": "waking_tick_producer_live_at_rem_entry",
            "description": ("guard-ON cells must reach every post-C0 REM entry "
                            "with _waking_ticks_since_capture > 0, else the "
                            "guard is a kill switch, not a cold-start guard"),
            "control": "guard-ON cells after C0; producer is REEAgent.sense()",
            "measured": float(p1_measured if p1_measured is not None else 0),
            "threshold": float(P1_MIN_WAKING_TICKS),
            "direction": "lower",
            "offending_cell": p1_cell,
            "met": bool((p1_measured or 0) >= P1_MIN_WAKING_TICKS),
        },
        {
            "name": "realized_pe_variance_in_band",
            "description": ("the design's anchor is 1/realized-PE-variance; a "
                            "degenerate rv makes the anchor meaningless"),
            "control": "each cell's own realized forward-model squared error",
            "measured": float(p2_measured),
            "threshold_low": float(P2_PE_VAR_LOW),
            "threshold_high": float(P2_PE_VAR_HIGH),
            "comparator_low": ">", "comparator_high": "<",
            "direction": "interval",
            "offending_cell": p2_cell,
            "met": bool(pe_rows and P2_PE_VAR_LOW < p2_measured < P2_PE_VAR_HIGH),
        },
        {
            "name": "cold_start_cycle_observed",
            "description": ("the pre-loop agent.reset() must fire a "
                            "zero-waking-tick REM entry (C0); without it E1 "
                            "has nothing to discriminate"),
            "control": "enter_rem calls counted from agent construction",
            "measured": float(p3_measured if p3_measured is not None else 0),
            "threshold": float(P3_EXPECTED_C0_CALLS),
            "direction": "lower",
            "offending_cell": p3_cell,
            "met": bool((p3_measured or 0) >= P3_EXPECTED_C0_CALLS),
        },
        {
            "name": "drift_source_not_clamped",
            "description": ("SD-076 must actually move rv relative to its "
                            "matched drift_off cell. V3-EXQ-794's absolute "
                            "floor pinned rv at exactly 0.010000 on every "
                            "inflation arm; this is the anti-recurrence check"),
            "control": "matched (guard, seed) drift_off vs drift_lo/hi cells",
            "measured": float(p4_measured),
            "threshold": float(P4_MIN_DRIFT_EFFECT),
            "direction": "lower",
            "met": bool(p4_deltas and p4_measured > P4_MIN_DRIFT_EFFECT),
        },
    ]
    all_met = all(p["met"] for p in preconditions)

    targets_all = [t for r in rows for t in r["targets"]]
    varied = len(set(round(t, 9) for t in targets_all)) > 1
    criteria_non_degenerate = {
        # E1 restates the guard's own predicate at C0 -- kept as a contract
        # report, and NOT load-bearing, for exactly that reason.
        "E1": bool(len(set(r["c0_captured"] for r in rows)) > 1),
        "E2": bool(varied and crit["e2_per_cell"]),
        "E3": bool(varied),
        "E4": bool(varied),
        "E5": bool(crit["e5_min_lo_hi_separation"] == crit["e5_min_lo_hi_separation"]),
    }

    if not all_met:
        label = "substrate_not_ready_requeue"
    elif crit["e4_falsified"]:
        label = "mech204_option_a_falsified_demote"
    elif crit["overall_pass"]:
        label = "f1_coldstart_guard_validated"
    else:
        label = "f1_coldstart_guard_partial"

    return {
        "label": label,
        "preconditions": preconditions,
        "criteria_non_degenerate": criteria_non_degenerate,
        "combination_rule": (
            "PASS = E2 AND E3 AND E4_not_falsified. E1 (cold-start contract) "
            "and E5 (dose non-saturation) are REPORTED, not gates: E1 restates "
            "the guard's own predicate at C0 and duplicates contracts C1-C8, "
            "and E5 is a saturation watchdog for the SD-076 floor."),
        "criteria": [
            {"name": "E1_cold_start_contract", "load_bearing": False,
             "passed": bool(crit["e1_pass"]),
             "measured": float(crit["e1_ok_cells"]),
             "threshold": float(crit["n_cells"])},
            {"name": "E2_contrast_departs_from_closed_form", "load_bearing": True,
             "passed": bool(crit["e2_pass"]),
             "measured": float(max(crit["e2_seeds_meeting_per_drift"].values())
                               if crit["e2_seeds_meeting_per_drift"] else 0),
             "threshold": float(E2_MIN_SEEDS)},
            {"name": "E3_guard_improves_rv_calibration", "load_bearing": True,
             "passed": bool(crit["e3_pass"]),
             "measured": float(max((d["seeds_meeting"]
                                    for d in crit["e3_per_drift"].values()), default=0)),
             "threshold": float(E3_MIN_SEEDS)},
            {"name": "E4_falsifier_monotone_away", "load_bearing": True,
             "passed": bool(crit["e4_pass_not_falsified"]),
             "measured": float(max(crit["e4_monotone_away_seeds_per_drift"].values())
                               if crit["e4_monotone_away_seeds_per_drift"] else 0),
             "threshold": float(E4_MIN_SEEDS)},
            {"name": "E5_dose_non_saturated", "load_bearing": False,
             "passed": bool(crit["e5_pass_dose_non_saturated"]),
             "measured": float(crit["e5_min_lo_hi_separation"]),
             "threshold": float(E5_MIN_DOSE_SEPARATION)},
        ],
    }


def _flat_scalar(rows: List[dict], crit: dict, interp: dict) -> dict:
    out = {
        "e1_pass": int(bool(crit["e1_pass"])),
        "e2_pass": int(bool(crit["e2_pass"])),
        "e3_pass": int(bool(crit["e3_pass"])),
        "e4_falsified": int(bool(crit["e4_falsified"])),
        "e5_pass_dose_non_saturated": int(bool(crit["e5_pass_dose_non_saturated"])),
        "overall_pass": int(bool(crit["overall_pass"])),
        "n_cells": len(rows),
    }
    deps = [d["max_departure_ratio"] for d in crit["e2_per_cell"].values()
            if d["max_departure_ratio"] == d["max_departure_ratio"]]
    if deps:
        out["e2_max_departure_ratio"] = float(max(deps))
    r3 = [d["mean_ratio"] for d in crit["e3_per_drift"].values()
          if d["mean_ratio"] == d["mean_ratio"]]
    if r3:
        out["e3_min_mean_rv_distance_ratio"] = float(min(r3))
    anchors = [r["anchor_precision"] for r in rows
               if r["anchor_precision"] == r["anchor_precision"]]
    if anchors:
        out["anchor_precision_mean"] = float(sum(anchors) / len(anchors))
    if crit["e5_min_lo_hi_separation"] == crit["e5_min_lo_hi_separation"]:
        out["e5_min_lo_hi_separation"] = float(crit["e5_min_lo_hi_separation"])
    for p in interp["preconditions"]:
        v = p.get("measured")
        if isinstance(v, (int, float)) and not isinstance(v, bool) \
                and v == v and abs(v) != float("inf"):
            out["precondition_" + p["name"]] = float(v)
    return out


def main(dry_run: bool = False):
    seeds = (SEEDS[0],) if dry_run else SEEDS
    n_train = 3 if dry_run else N_TRAIN_EPS
    steps = 25 if dry_run else STEPS_PER_EP

    t0 = time.perf_counter()
    rows: List[dict] = []
    for guard_label, guard in GUARD_LEVELS:
        for drift_label, asym in DRIFT_LEVELS:
            for seed in seeds:
                rows.append(run_cell(guard_label, guard, drift_label, asym,
                                     seed, n_train, steps, dry_run))
    elapsed = time.perf_counter() - t0

    crit = _aggregate(rows)
    interp = _build_interpretation(rows, crit)
    outcome = "PASS" if crit["overall_pass"] else "FAIL"

    print(
        f"V3-EXQ-541d MECH-204 F1 cold-start guard validation (realised PE) -- "
        f"{outcome} in {elapsed:.1f}s (label={interp['label']})",
        flush=True,
    )
    if dry_run:
        print("[--dry-run] smoke summary:", flush=True)
        print(json.dumps({
            "preconditions": [{"name": p["name"], "measured": p.get("measured"),
                               "met": p["met"]} for p in interp["preconditions"]],
            "anchor_per_arm": {r["arm"]: round(r["anchor_precision"], 2)
                               for r in rows},
            "c0": [{"arm": r["arm"], "calls": r["c0_enter_rem_calls"],
                    "captured": r["c0_captured"], "target": r["c0_target"]}
                   for r in rows],
            "enter_rem_per_cycle": sorted(set(
                n for r in rows for n in r["n_enter_rem_calls_per_cycle"])),
            "e1_pass": crit["e1_pass"],
            "e5_min_lo_hi_separation": crit["e5_min_lo_hi_separation"],
        }, indent=1, default=str), flush=True)
        print("[--dry-run] manifest not written.", flush=True)
        return None

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    full_config = {
        "seeds": list(seeds),
        "guard_levels": [{"label": l, "guard": g} for l, g in GUARD_LEVELS],
        "drift_levels": [{"label": l, "asymmetry": a} for l, a in DRIFT_LEVELS],
        "n_train_eps": n_train,
        "steps_per_ep": steps,
        "grid_size": GRID_SIZE,
        "lr": LR,
        "sleep_loop_K": 1,
        "precision_zero_point_ema_alpha": PRECISION_ZERO_POINT_EMA_ALPHA,
        "rem_precision_recalibration_step": REM_PRECISION_RECALIBRATION_STEP,
        "waking_confidence_rv_floor_relative_frac": INFLATION_RV_FLOOR_RELATIVE_FRAC,
        "waking_confidence_rv_floor_mode": INFLATION_RV_FLOOR_MODE,
        "env_kwargs": _env_kwargs(dry_run),
    }
    manifest = {
        "schema_version": "v1",
        "run_id": f"{EXPERIMENT_TYPE}_{ts}_v3",
        "experiment_type": EXPERIMENT_TYPE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": ts,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "claim_ids": CLAIM_IDS,
        "backlog_id": BACKLOG_ID,
        "outcome": outcome,
        "result": outcome,
        "evidence_direction": "non_contributory",
        "sleep_driver_pattern": (
            "K=1 single-fire (SleepLoopManager, fires every episode)"),
        "criteria": crit,
        "interpretation": interp,
        "arm_results": rows,
        "registered_thresholds": {
            "E1_SENTINEL_TOL": E1_SENTINEL_TOL,
            "E2_WINDOW": E2_WINDOW,
            "E2_MIN_DEPARTURE_RATIO": E2_MIN_DEPARTURE_RATIO,
            "E2_MIN_SEEDS": E2_MIN_SEEDS,
            "E3_MAX_RV_RATIO": E3_MAX_RV_RATIO,
            "E3_MIN_SEEDS": E3_MIN_SEEDS,
            "E4_WINDOW": E4_WINDOW,
            "E4_MIN_SEEDS": E4_MIN_SEEDS,
            "E5_MIN_DOSE_SEPARATION": E5_MIN_DOSE_SEPARATION,
            "E3_SENTINEL_PRECISION": E3_SENTINEL_PRECISION,
        },
        "readout": _flat_scalar(rows, crit, interp),
        "diagnostics": {
            "realized_pe_variance_per_arm": {
                r["arm"] + "/seed" + str(r["seed"]): r["realized_pe_variance"]
                for r in rows},
            "final_rv_per_arm": {
                r["arm"] + "/seed" + str(r["seed"]): r["final_rv"] for r in rows},
            "enter_rem_calls_per_cycle_observed": sorted(set(
                n for r in rows for n in r["n_enter_rem_calls_per_cycle"])),
        },
        "notes": (
            "MECH-204 F1 cold-start guard validation, REBUILT on a REALISED-PE "
            "base per user decision 2026-09-19T23:52:40Z after the 541c-based "
            "design was refused at red-team (BLOCKING). Base: canonical "
            "StepHarness at the IGW-20260915-243 operating point (K=1 sleep, "
            "F1 recal step 0.25), matching V3-EXQ-794's substrate point. The "
            "anchor is the design's own rule, 1/realized-PE-variance, measured "
            "IN-RUN per cell (probe: 259.9; the chip's ~255 transfers, 541c's "
            "2.148 does not and is not used). E2 pre-registers the check that "
            "the cross-arm contrast is NOT the 541c closed form "
            "0.9^k*(SENTINEL-p_1) -- probe departure 4.17x (no drift) and "
            "8.24x (drift LO) by cycle 9, vs 8e-05 agreement on 541c. SD-076 "
            "is armed as the drift source with the HEADROOM REPAIR (relative "
            "frac 0.2, soft floor), never 794's absolute 0.01 floor which "
            "clamped rv to exactly 0.010000; P4 and E5 are the anti-recurrence "
            "checks. STATED DEPENDENCY: sd_waking_confidence_inflation_headroom "
            "is implemented but ready FALSE with V3-EXQ-794a not yet queued, so "
            "this rides on an unvalidated repair. DIAGNOSTIC, not evidence: "
            "MECH-204 is not answerable until that validation lands; this "
            "validates an INSTRUMENT. The chip's second defect instance (a "
            "cycle issuing MORE THAN ONE enter_rem) does not occur on this "
            "base either -- counts are recorded so the absence is auditable -- "
            "and is owed to a driver that genuinely multi-fires REM."
        ),
    }

    out_dir = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments"
    out_path = write_flat_manifest(
        manifest, out_dir, dry_run=False, config=full_config, seeds=SEEDS,
        script_path=Path(__file__), started_at=t0, agent=None,
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
