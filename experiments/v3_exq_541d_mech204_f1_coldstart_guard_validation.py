#!/opt/local/bin/python3
"""V3-EXQ-541d -- MECH-204 Option A: does REM recalibration DE-CALIBRATE rv?

!! NOT QUEUED. DO NOT QUEUE AS IT STANDS. Red-team BLOCKING (4th pass). !!
================================================================================
The falsifier CANNOT FIRE while the drift source is unarmed, and the reason is
analytic rather than empirical, so no amount of tuning fixes it:

  With the guard ON the recalibration target is
  `serotonin._persistent_zero_point`, an EMA of `current_precision = 1/rv`
  (serotonin.py:413-419) -- A LAGGED FUNCTION OF rv ITSELF. The WRITEBACK then
  moves rv 25% toward `1/target` (e3_selector.py:1152-1155). Recalibrating rv
  toward a lagged function of rv is NEAR-IDEMPOTENT, so it cannot de-calibrate
  a converged rv. An adversarial sweep over eight rv trajectories -- stationary,
  10x and 100x monotonic decay, 5x step collapse, 5x step rise, alternating x3,
  20x single spike, sawtooth -- gives a guard-ON mean relative displacement in
  [-0.15, +0.062], never approaching the 0.25 firing bar. Measured in the smoke:
  -0.001. The `..._demote` branch is UNREACHABLE and the run could only ever
  confirm.

  ROOT CAUSE OF THIS BUILD'S DEFECT, stated plainly: this driver dropped SD-076
  (`use_waking_confidence_inflation`) to avoid depending on the unvalidated
  `sd_waking_confidence_inflation_headroom` repair. That was a mistake. SD-076
  is the ONLY substrate mechanism that makes rv diverge from realised PE while
  the guard is on, and the user's own base-selection criterion had required the
  precision be "realised AND ABLE TO DRIFT". Removing the drift source removed
  exactly the property that criterion was protecting.

WHAT THE FOUR PASSES TOGETHER ESTABLISH -- this is a RESULT, not just a defect
log. Option A's recalibration can only de-calibrate rv when its target is
contaminated or stale, i.e. (a) the precision_init cold-start sentinel, which
the landed F1 guard removes, or (b) rv far from its own lagged mean, i.e. a
non-converged rv (short episodes) or an active drift source. It is NOT
de-calibrating in principle. The 9/9 and 18-82x figures reported in GFLAG-0379
were measured in regime (b) -- 6-15-tick episodes -- and that flag's DEMOTE
recommendation is CORRECTED by GFLAG-0384 accordingly.

Escalated as decision chip chip-20260920-exq541d-driftsource-required.
Record: REE_assembly evidence/planning/
        exq541d_redteam_blocking_refusal_staged_20260919.md section 7.
Everything below is retained because it is correct and reusable: the survivable
regime (chosen by measurement), the within-cycle magnitude statistic, the
positive control, and the recorded episode-length/termination instrumentation.
The one change owed is re-arming SD-076 as the drift source.
================================================================================

SLEEP DRIVER: K=1 single-fire (SleepLoopManager, sleep_loop_episodes_K=1, fires every episode)

red-team: see queue entry note (recorded at queue time).

WHAT THIS ASKS, AND WHY IT IS NOT WHAT THE ORIGINATING CHIP ASKED
-----------------------------------------------------------------
The MECH-204 F1 cold-start guard (SerotoninConfig.precision_zero_point_require_
waking, ree-v3 e1ff0927) stops a zero-waking-tick REM entry from anchoring the
persistent precision reference on E3's precision_init sentinel. Two earlier
builds of V3-EXQ-541d were REFUSED at red-team; the full record with
measurements is REE_assembly evidence/planning/
exq541d_redteam_blocking_refusal_staged_20260919.md (sections 1-6).

The second refusal found the chip's pre-registered falsifier to be on the WRONG
VARIABLE. It asked whether the F1 TARGET climbs away from
1/realized-PE-variance. But the IGW-20260915-243 measurement the chip cites
shows the target climbing 2.0 -> 27 -> 49 -> 69 -> 88, i.e. TOWARD 255, and
what IGW-243 actually recorded moving away is rv: "recalibration pushes rv AWAY
from calibration (0.0039 -> 0.0121)". Target and rv were conflated. Since the
target starts at ~2-4 and the anchor is ~260, the gap can only shrink, so the
falsifier-as-written is unfirable on ANY base.

User decision 2026-09-20T09:29:55Z re-expressed it on rv. THE QUESTION HERE IS:

    Does the sleep WRITEBACK move E3's running variance AWAY FROM the agent's
    realized forward-model prediction-error variance -- and does the cold-start
    guard reduce that?

This is a WITHIN-CYCLE before/after comparison at the WRITEBACK, so unlike the
previous builds' criteria it is INDEPENDENT OF EPISODE LENGTH by construction.

THE SURVIVABLE REGIME, AND WHY IT IS NOT THE 794 OPERATING POINT
----------------------------------------------------------------
The second refusal also found the earlier build was not in the regime it
declared: it specified 200 steps/episode but agents died in 6-15 ticks, so rv
never converged between sleep cycles and every cross-cycle statistic was really
a function of episode length. Option B of the same user decision required a
regime measured to last ~100+ ticks, chosen by MEASUREMENT among candidates.

MEASURED (env-only survivability scan, 12 episodes x 12 seeds, 200-step cap;
random actions, then confirmed against the full driver whose BASE mean 13.1
matches the scan's 11.9):

    candidate               params  mean   median  frac>=100  terminal cause
    BASE (794 point)             0   11.9    12.5      0.000  health_depleted
    lower hazard_harm 0.01       1   11.9*   12.5      0.000  health_depleted
    P0 warmup 10 episodes        0   12.5     9.5      0.000  health_depleted
    fewer hazards (1)            1   17.5    12.0      0.000  health_depleted
    proximity_harm 0.03          1   29.8    32.5      0.000  health_depleted
    prox 0.02 + contam 0.02      2   74.1    76.5      0.000  health_depleted
    prox 0.015 + contam 0.015    2   98.8    98.0      0.417  health_depleted
 -> prox 0.01  + contam 0.01     2  149.2   148.5      1.000  health_depleted
    prox 0.005 + contam 0.005    2  200.0   200.0      1.000  STEP CAP (immortal)

    (*) byte-identical episode lengths to BASE.

SELECTION RULE, stated before the choice: among candidates clearing the bar
(>= 100 ticks in >= 2/3 of episodes), take the one that (1) changes the FEWEST
environment parameters from the IGW-243/794 operating point; (2) on a tie,
perturbs the measured realized-PE variance (this DV's own anchor) least;
(3) on a further tie, PRESERVES THE QUALITATIVE REGIME -- episodes must still
end by `health_depleted`, so harm remains a live constraint rather than being
removed. Tie-break: prefer an environment parameter over a training-schedule
change, because a schedule change alters the agent's competence and therefore
what regime is being measured.

All three candidates the decision named were ELIMINATED BY MEASUREMENT, and one
of them for an instructive reason: lowering `hazard_harm` produces BYTE-IDENTICAL
episode lengths, because contact harm is not the binding constraint. The two
actual killers are the continuous `hazard_approach` proximity drain (~0.10
health/tick at the 794 setting) and `contaminated_harm`, which defaults to 0.4
PER CONTACT and is overridden by neither 794 nor any previous 541d build. That
second killer was invisible until it was measured, which is why it was not among
the named candidates.
`prox 0.01 + contam 0.01` is the only setting that clears the bar while still
terminating by health depletion; `prox 0.005` reaches the step cap every episode,
i.e. the agent is effectively immortal and harm has been removed rather than
slowed, which rule (3) excludes. So the choice is forced, not a judgement call,
and no decision chip is owed.

ONE HONEST QUALIFICATION, from the smoke that followed. The table above is a
RANDOM-POLICY screen. Under the real driver (E3-selected actions plus training)
the agent at `prox 0.01` survives to the 200-step CAP rather than dying at ~149,
so its recorded termination is the cap, not health depletion. The distinction
rule (3) draws still holds and is still the reason to prefer this setting over
`prox 0.005`: at 0.01 a random policy STILL DIES, so survival is EARNED by
avoidance and harm remains a live constraint the agent must act against, whereas
at 0.005 it is free. What rule (3) actually discriminates is whether harm is
lethal-in-principle, and it is measured on the random screen for exactly that
reason. Episode length being pinned at the cap is a benefit here, not a problem:
it makes every criterion trivially episode-length-independent.

MEASURED ON THE CHOSEN REGIME (smoke, seed 42, 3 cycles, full 200-step cap):
  anchor = 1/realized-PE-variance = 258.78   (the chip's "~255", reproduced)
  final rv = 0.003807, and rv / realized-PE-variance = 1.0 in BOTH arms
      -- rv now CONVERGES to the realized PE variance between cycles, against
      18-82x in the pre-survivable regime. That convergence is the whole point
      of option B: the de-calibration is now measured against a converged
      baseline instead of a sawtooth artifact of 12-tick episodes.
  de-calibration away-fraction: guard OFF 1.0, guard ON 0.5
  C0: guard OFF captures at 1.999996000008; guard ON does not capture.

SD-076 waking confidence inflation is deliberately NOT armed. The previous build
armed it to give precision something to drift, which the target-based falsifier
needed; the rv-based falsifier does not, and the measured de-calibration is
present without it (9/9 cycles guard OFF, 8/9 guard ON on the pre-survivable
base). Dropping it also drops this run's dependency on
`sd_waking_confidence_inflation_headroom`, which is `implemented` with ready
FALSE and its own validation (V3-EXQ-794a) not yet queued.

PRE-REGISTERED CRITERIA
-----------------------
These were REVISED after a third red-team BLOCKING pass, whose two findings
were verified arithmetically before being accepted (the working is in the
record, section 7). Both concerned HOW the de-calibration was scored, not what
question is being asked.

  Finding A: scoring the falsifier as a COUNT of cycles that moved rv away was
  noise. In a converged regime rv ~= R and the guard-ON target ~= 1/R, so the
  sign of (|rv_after-R| - |rv_before-R|) compares two ~1e-5 quantities. The
  smoke measured an ON away-fraction of exactly 0.5 -- a coin flip. The SAME
  quantity as a MAGNITUDE separates by three orders (OFF +2.03 vs ON +0.0000,
  in units of R), so F1 is now a magnitude criterion.

  Finding B: the old F2 ("is ON less de-calibrating than OFF?") could not fail.
  rv_before is identical in both arms, the sentinel in OFF's target is the only
  difference, and it is strictly de-calibrating, so F2 was true at every cycle
  by construction -- it restated the cold-start contract F3. Verified:
  OFF relative displacement +2.029, +0.980, +0.319, +0.101, +0.009 at cycles
  1/2/5/10/19 against ON +0.0000 throughout. F2 is therefore INVERTED into a
  POSITIVE CONTROL, which is the only honest role for a quantity that cannot
  come out the other way.

  F1 (load-bearing) THE FALSIFIER, ON rv, AS A MAGNITUDE. Per guard-ON cell,
      the mean over cycles of (|rv_after - R| - |rv_before - R|) / R, where R
      is that cell's own realized PE variance. If that mean is
      >= F1_MAX_RELATIVE_DECAL in >= F1_MIN_SEEDS seeds, then removing the
      cold-start artefact does NOT stop Option A de-calibrating rv, the
      precision_init sentinel was not the (only) cause, and per the originating
      chip's pre-registration MECH-204 Option A should be DEMOTED, not retuned.
  F2 (load-bearing) POSITIVE CONTROL, on the guard-OFF arm. The OFF arm must
      itself show mean relative de-calibration >= F2_MIN_POSITIVE_CONTROL in
      >= F2_MIN_SEEDS seeds. This is a LIVENESS check, not a second test: if
      the arm that is supposed to exhibit the phenomenon does not, the statistic
      is inert and NEITHER branch of F1 may be read -- the run routes to
      substrate_not_ready_requeue rather than reporting a quiet ON arm as a
      clean result.
  F3  COLD-START CONTRACT (reported, NOT load-bearing). Guard OFF: C0 captures,
      at the sentinel 1.999996 within F3_SENTINEL_TOL. Guard ON: it does not.
      Not load-bearing because at C0 `_waking_ticks_since_capture == 0` by
      construction, so this restates the guard's own predicate and duplicates
      contracts C1-C8.
  F4  REGIME CHECK (reported). Median episode length and per-episode
      termination causes are RECORDED, and the env's own max_episode_steps is
      bound to the driver's loop bound so `done_cause` is the env's verdict
      rather than a label this driver invents. The previous build's verdict
      silently depended on episode length and never wrote it down.
PASS = F2_positive_control_live AND (NOT F1_fired).

BOTH VERDICT LABELS CARRY A `_survivable_regime` SUFFIX. This run is in a
softened harm regime (proximity 0.01 vs 0.12, contaminated 0.01 vs 0.4) and
does NOT speak to the 794 / IGW-243 operating point, where episodes last 6-15
ticks and the pre-survivable probe measured guard-ON de-calibration at 8/9
cycles. A reader must not generalise a "validated" here to that regime; the
short-episode regime is owed its own run.

EXPERIMENT_PURPOSE is "diagnostic", NOT "evidence": MECH-204's what_would_answer
(a) holds the claim not answerable until sd_waking_confidence_inflation_headroom
is validated. This measures an INSTRUMENT and feeds the disposition already
raised as GFLAG-0379.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
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

# P1/P3/P4 are DIRECT COUNTER/LENGTH READS with simple bounds, not scored
# signature predicates, so the unmeetable-by-construction failure the
# anchor-reachability check guards cannot arise. Measured on this base:
# P1 >= 1 (one note_waking_tick per sense() call), P3 = 1, P4 median ~148.
ANCHOR_REACHABILITY_EXEMPT = (
    "P1/P3/P4 are direct counter and episode-length reads with simple bounds, "
    "not scored signatures; reachability measured (P1>=1, P3=1, P4 median ~148)"
)

# ---- Substrate operating point (IGW-20260915-243 / V3-EXQ-794) ----
GRID_SIZE = 12
STEPS_PER_EP = 200           # a CAP; episodes end on health depletion
N_TRAIN_EPS = 20             # K=1 -> 19 scoring cycles; claims.yaml floor is 16
LR = 5e-4
SWS_CONSOLIDATION_STEPS = 8
REM_ATTRIBUTION_STEPS = 6
ALPHA_WORLD = 0.9
ALPHA_SELF = 0.3
PRECISION_ZERO_POINT_EMA_ALPHA = 0.1
REM_PRECISION_RECALIBRATION_STEP = 0.25

# ---- The survivable regime (option B; the two parameters that were measured) ----
PROXIMITY_HARM_SCALE = 0.01   # 794 uses 0.12; the dominant continuous killer
CONTAMINATED_HARM = 0.01      # env default 0.4 per contact; the hidden killer

E3_PRECISION_INIT_VARIANCE = 0.5
E3_SENTINEL_PRECISION = 1.0 / (E3_PRECISION_INIT_VARIANCE + 1e-6)  # ~1.999996

SEEDS = (42, 43, 45)          # 44 excluded on a reef-config env (EXQ-539-540)
GUARD_LEVELS: Tuple[Tuple[str, bool], ...] = (("guard_off", False), ("guard_on", True))

# ---- Pre-registered thresholds (constants; NOT derived from this run) ----
# F1 is scored on MAGNITUDE, not on a count of cycles. In a converged regime
# rv ~= R and the guard-ON target ~= 1/R, so a SIGN-COUNT compares two ~1e-5
# quantities and is pure noise (measured: away-fraction 0.5 on the ON arm).
# The same quantity as a magnitude separates by three orders: OFF +2.03 vs
# ON +0.0000 in units of R. Threshold sits far above the ON null and far below
# the OFF positive control.
F1_MAX_RELATIVE_DECAL = 0.25  # mean (|rv_after-R| - |rv_before-R|)/R, guard ON
F1_MIN_SEEDS = 2              # in >= 2/3 seeds
# F2 is a POSITIVE CONTROL on the guard-OFF arm, not a second test. Asking
# "is ON less de-calibrating than OFF" is true at every cycle by construction
# -- the sentinel in OFF's target is the only difference -- so it restated the
# cold-start contract rather than measuring anything. Inverted into a liveness
# check: the OFF arm MUST show the de-calibration, else the statistic is inert
# and the run measured nothing.
F2_MIN_POSITIVE_CONTROL = 0.25
F2_MIN_SEEDS = 2
F3_SENTINEL_TOL = 1e-3
P1_MIN_WAKING_TICKS = 1
P2_PE_VAR_LOW = 1e-4
P2_PE_VAR_HIGH = 1e-1
P3_EXPECTED_C0_CALLS = 1
P4_MIN_MEDIAN_EPISODE_LEN = 100   # the regime bar option B was chosen against


def _env_kwargs(dry_run: bool = False) -> dict:
    # NOTE: the grid is deliberately NOT shrunk under --dry-run; survivability
    # (and hence P4) is a property of the grid + harm settings together, and the
    # survivable regime was selected by measurement at GRID_SIZE.
    return dict(
        size=GRID_SIZE,
        num_hazards=3,
        num_resources=3,
        hazard_harm=0.04,
        proximity_harm_scale=PROXIMITY_HARM_SCALE,
        contaminated_harm=CONTAMINATED_HARM,
        proximity_benefit_scale=0.10,
        use_proxy_fields=True,
        resource_respawn_on_consume=True,
        # Bind the env's own cap to the driver's loop bound so `done_cause` is
        # the env's real verdict rather than a label this driver invents when
        # its loop runs out (the env default is 500).
        max_episode_steps=STEPS_PER_EP,
    )


def cell_config_slice(guard: bool, n_train: int, steps: int,
                      dry_run: bool) -> dict:
    return {
        "env_kwargs": _env_kwargs(dry_run),
        "grid_size": GRID_SIZE,
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
        "proximity_harm_scale": PROXIMITY_HARM_SCALE,
        "contaminated_harm": CONTAMINATED_HARM,
        "self_dim": 32,
        "world_dim": 32,
        "tonic_5ht_enabled": True,
    }


def _make_env(seed: int, dry_run: bool = False) -> CausalGridWorldV2:
    return CausalGridWorldV2(seed=seed, **_env_kwargs(dry_run))


def _make_agent(env: CausalGridWorldV2, guard: bool) -> REEAgent:
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=32, world_dim=32,
        alpha_world=ALPHA_WORLD, alpha_self=ALPHA_SELF,
        sws_enabled=True, sws_consolidation_steps=SWS_CONSOLIDATION_STEPS,
        rem_enabled=True, rem_attribution_steps=REM_ATTRIBUTION_STEPS,
        use_sleep_loop=True, sleep_loop_episodes_K=1,
        use_rem_precision_recalibration=True,
        precision_zero_point_ema_alpha=PRECISION_ZERO_POINT_EMA_ALPHA,
        rem_precision_recalibration_step=REM_PRECISION_RECALIBRATION_STEP,
    )
    cfg.serotonin.precision_zero_point_require_waking = bool(guard)
    cfg.serotonin.tonic_5ht_enabled = True
    return REEAgent(cfg)


def run_cell(guard_label: str, guard: bool, seed: int, n_train: int,
             steps: int, dry_run: bool) -> dict:
    print(f"Seed {seed} Condition {guard_label}", flush=True)
    slice_ = cell_config_slice(guard, n_train, steps, dry_run)
    with arm_cell(
        seed, config_slice=slice_, script_path=Path(__file__),
        config_slice_declared=True, include_driver_script_in_hash=False,
        extra_ineligible_reasons=["shared_optimizer_across_episodes"],
    ) as cell:
        env = _make_env(seed, dry_run)
        agent = _make_agent(env, guard)
        ser = agent.serotonin
        optimizer = optim.Adam(agent.parameters(), lr=LR)

        # Instrument BEFORE the first reset: the pre-loop agent.reset() fires
        # the zero-waking-tick cold-start cycle C0.
        trace = {"calls": 0, "suppressed": 0, "ticks_at_entry": []}
        _orig = ser.enter_rem

        def _traced(current_precision):
            trace["calls"] += 1
            before = ser._persistent_zero_point
            trace["ticks_at_entry"].append(int(ser._waking_ticks_since_capture))
            _orig(current_precision=current_precision)
            if ser._persistent_zero_point == before:
                trace["suppressed"] += 1
        ser.enter_rem = _traced

        harness = StepHarness(agent, env, train_mode=True, seed=seed)
        cycles: List[dict] = []
        ep_lengths: List[int] = []
        done_causes: List[str] = []
        pe_all: List[float] = []
        c0: Optional[dict] = None

        for ep in range(n_train):
            calls_before = trace["calls"]
            agent.reset()
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
                c0 = rec
            else:
                cycles.append(rec)
            if st:
                st.last_metrics = {}

            _, obs_dict = env.reset()
            harness.reset()
            n = 0
            cause = "cap"
            for _ in range(steps):
                result = harness.step(obs_dict)
                optimizer.zero_grad()
                loss = agent.compute_prediction_loss()
                if loss.requires_grad:
                    loss.backward()
                    optimizer.step()
                pe = result.residue_metrics.get("e3_prediction_error")
                if pe is not None:
                    v = float(pe.detach()) if hasattr(pe, "detach") else float(pe)
                    pe_all.append(v)
                obs_dict = result.next_obs_dict
                n += 1
                if result.done:
                    info = getattr(result, "info", None) or {}
                    cause = str(info.get("done_cause") or "unknown")
                    break
            ep_lengths.append(n)
            done_causes.append(cause)
            if (ep + 1) % 5 == 0 or ep + 1 == n_train:
                print(f"  [train] arm={guard_label} seed={seed} "
                      f"ep {ep + 1}/{n_train} len={n} "
                      f"rv={float(agent.e3._running_variance):.6f}", flush=True)

        realized = (sum(pe_all) / len(pe_all)) if pe_all else float("nan")
        anchor = (1.0 / realized
                  if realized == realized and realized > 1e-12 else float("nan"))

        # --- F1/F2: the WITHIN-CYCLE de-calibration statistic ---
        away = 0
        scored = 0
        rel_disp: List[float] = []
        for c in cycles:
            rb, ra = c["rv_before"], c["rv_after"]
            if rb is None or ra is None or realized != realized:
                continue
            scored += 1
            d_before = abs(rb - realized)
            d_after = abs(ra - realized)
            c["decalibration_displacement"] = d_after - d_before
            c["decalibration_relative"] = (d_after - d_before) / realized
            rel_disp.append(c["decalibration_relative"])
            if d_after > d_before:
                away += 1
        away_frac = (away / scored) if scored else float("nan")

        post_ticks = trace["ticks_at_entry"][1:]
        row = {
            "arm": guard_label,
            "guard_label": guard_label,
            "guard": bool(guard),
            "seed": seed,
            "n_cycles": len(cycles),
            "n_scored_cycles": scored,
            "realized_pe_variance": float(realized),
            "anchor_precision": float(anchor),
            "final_rv": float(agent.e3._running_variance),
            "rv_over_realized_pe": (float(agent.e3._running_variance) / realized
                                    if realized == realized and realized > 0
                                    else float("nan")),
            "decalibration_away_fraction": float(away_frac),
            "mean_relative_displacement": (
                statistics.mean(rel_disp) if rel_disp else float("nan")),
            "episode_lengths": ep_lengths,
            "median_episode_length": statistics.median(ep_lengths) if ep_lengths else 0,
            "mean_episode_length": (statistics.mean(ep_lengths) if ep_lengths else 0),
            "done_causes": done_causes,
            "c0_enter_rem_calls": int(c0["n_enter_rem_calls"]) if c0 else 0,
            "c0_captured": bool(c0 and c0["persistent_zero_point"] is not None),
            "c0_target": (c0["target"] if c0 else None),
            "n_enter_rem_calls_total": int(trace["calls"]),
            "n_enter_rem_suppressed": int(trace["suppressed"]),
            "n_enter_rem_calls_per_cycle": [c["n_enter_rem_calls"] for c in cycles],
            "min_waking_ticks_at_post_c0_rem_entry": (
                min(post_ticks) if post_ticks else 0),
            "cycle_records": cycles,
        }
        cell.stamp(row)

    if guard:
        f3_ok = not row["c0_captured"]
    else:
        f3_ok = bool(row["c0_captured"] and row["c0_target"] is not None
                     and abs(row["c0_target"] - E3_SENTINEL_PRECISION) <= F3_SENTINEL_TOL)
    row["f3_cold_start_contract_ok"] = bool(f3_ok)
    print(f"verdict: {'PASS' if f3_ok else 'FAIL'}", flush=True)
    return row


def _aggregate(rows: List[dict]) -> dict:
    on = [r for r in rows if r["guard"]]
    off = [r for r in rows if not r["guard"]]

    def _disp(r):
        return r["mean_relative_displacement"]

    # F1 -- THE FALSIFIER, on magnitude, guard-ON arm only.
    f1_seeds_fired = sum(1 for r in on
                         if _disp(r) == _disp(r) and _disp(r) >= F1_MAX_RELATIVE_DECAL)
    f1_fired = f1_seeds_fired >= F1_MIN_SEEDS

    # F2 -- POSITIVE CONTROL on the guard-OFF arm: the statistic must be able
    # to see the de-calibration it is looking for. If OFF does not show it, the
    # measurement is inert and no ON reading means anything.
    f2_seeds_live = sum(1 for r in off
                        if _disp(r) == _disp(r) and _disp(r) >= F2_MIN_POSITIVE_CONTROL)
    f2_live = f2_seeds_live >= F2_MIN_SEEDS

    f3_pass = all(r["f3_cold_start_contract_ok"] for r in rows)
    median_lens = [r["median_episode_length"] for r in rows]
    # A fired falsifier only means anything if the control proved the
    # statistic live; an un-fired falsifier likewise.
    overall_pass = bool(f2_live and not f1_fired)
    return {
        "f1_fired": f1_fired,
        "f1_seeds_fired": f1_seeds_fired,
        "f1_mean_relative_displacement_per_cell": {
            f"{r['arm']}/seed{r['seed']}": _disp(r) for r in rows},
        "f2_positive_control_live": f2_live,
        "f2_seeds_live": f2_seeds_live,
        "f3_pass": f3_pass,
        "f4_median_episode_length_per_cell": {
            f"{r['arm']}/seed{r['seed']}": r["median_episode_length"] for r in rows},
        "f4_min_median_episode_length": min(median_lens) if median_lens else 0,
        "away_fraction_per_cell": {
            f"{r['arm']}/seed{r['seed']}": r["decalibration_away_fraction"] for r in rows},
        "overall_pass": overall_pass,
    }


def _build_interpretation(rows: List[dict], crit: dict) -> dict:
    on = [r for r in rows if r["guard"]]
    p1 = min((r["min_waking_ticks_at_post_c0_rem_entry"] for r in on), default=0)
    pe_rows = [r for r in rows
               if r["realized_pe_variance"] == r["realized_pe_variance"]]
    centre = math.sqrt(P2_PE_VAR_LOW * P2_PE_VAR_HIGH)
    if pe_rows:
        wr = max(pe_rows, key=lambda r: abs(
            math.log(max(r["realized_pe_variance"], 1e-12) / centre)))
        p2, p2_cell = wr["realized_pe_variance"], f"{wr['arm']}/seed{wr['seed']}"
    else:
        p2, p2_cell = float("nan"), None
    p3 = min((r["c0_enter_rem_calls"] for r in rows), default=0)
    p4_row = min(rows, key=lambda r: r["median_episode_length"]) if rows else None
    p4 = p4_row["median_episode_length"] if p4_row else 0

    preconditions = [
        {"name": "waking_tick_producer_live_at_rem_entry",
         "description": ("guard-ON cells must reach every post-C0 REM entry with "
                         "_waking_ticks_since_capture > 0, else the guard is a "
                         "kill switch rather than a cold-start guard"),
         "control": "guard-ON cells after C0; producer is REEAgent.sense()",
         "measured": float(p1), "threshold": float(P1_MIN_WAKING_TICKS),
         "direction": "lower", "met": bool(p1 >= P1_MIN_WAKING_TICKS)},
        {"name": "realized_pe_variance_in_band",
         "description": "the DV's reference is the realized PE variance itself",
         "control": "each cell's own realized forward-model squared error",
         "measured": float(p2),
         "threshold_low": float(P2_PE_VAR_LOW), "threshold_high": float(P2_PE_VAR_HIGH),
         "comparator_low": ">", "comparator_high": "<", "direction": "interval",
         "offending_cell": p2_cell,
         "met": bool(pe_rows and P2_PE_VAR_LOW < p2 < P2_PE_VAR_HIGH)},
        {"name": "cold_start_cycle_observed",
         "description": ("the pre-loop agent.reset() must fire a zero-waking-tick "
                         "REM entry (C0), else F3 has nothing to discriminate"),
         "control": "enter_rem calls counted from agent construction",
         "measured": float(p3), "threshold": float(P3_EXPECTED_C0_CALLS),
         "direction": "lower", "met": bool(p3 >= P3_EXPECTED_C0_CALLS)},
        {"name": "survivable_regime_median_episode_length",
         "description": ("the regime option B was selected against: episodes must "
                         "last long enough that rv can converge between sleep "
                         "cycles. The previous build's verdict silently depended "
                         "on this quantity and never recorded it"),
         "control": "worst cell's median episode length, 200-step cap",
         "measured": float(p4), "threshold": float(P4_MIN_MEDIAN_EPISODE_LEN),
         "direction": "lower",
         "offending_cell": (f"{p4_row['arm']}/seed{p4_row['seed']}" if p4_row else None),
         "met": bool(p4 >= P4_MIN_MEDIAN_EPISODE_LEN)},
    ]
    all_met = all(p["met"] for p in preconditions)

    disps = [r["mean_relative_displacement"] for r in rows
             if r["mean_relative_displacement"] == r["mean_relative_displacement"]]
    # F1 is non-degenerate only if the POSITIVE CONTROL proved the statistic
    # can see the de-calibration at all. A quiet ON arm against a quiet OFF arm
    # is an inert measurement, not a clean result.
    criteria_non_degenerate = {
        "F1": bool(crit["f2_positive_control_live"]),
        "F2": bool(len(set(round(d, 9) for d in disps)) > 1),
        "F3": bool(len(set(r["c0_captured"] for r in rows)) > 1),
        "F4": True,
    }

    if not all_met:
        label = "substrate_not_ready_requeue"
    elif not crit["f2_positive_control_live"]:
        # The statistic never saw the phenomenon even in the arm that is
        # supposed to exhibit it -- nothing was measured, so neither branch of
        # the falsifier may be read.
        label = "substrate_not_ready_requeue"
    elif crit["f1_fired"]:
        label = "mech204_option_a_decalibrates_rv_demote_survivable_regime"
    elif crit["overall_pass"]:
        label = "f1_coldstart_guard_validated_survivable_regime"
    else:
        label = "f1_coldstart_guard_partial"

    return {
        "label": label,
        "preconditions": preconditions,
        "criteria_non_degenerate": criteria_non_degenerate,
        "combination_rule": (
            "PASS = F2_positive_control_live AND (NOT F1_fired). F1 is the "
            "falsifier, scored on MAGNITUDE (mean relative de-calibration) in "
            "the guard-ON arm. F2 is a POSITIVE CONTROL on the guard-OFF arm, "
            "not a second test: if OFF does not exhibit the de-calibration the "
            "statistic is inert and NEITHER branch of F1 may be read, so the run "
            "routes to substrate_not_ready_requeue. Both labels carry a "
            "_survivable_regime suffix because the run is in a softened harm "
            "regime and does not speak to the 794/IGW-243 operating point. F3 "
            "(cold-start contract) and F4 (regime check) are REPORTED, not gates."),
        "criteria": [
            {"name": "F1_writeback_decalibrates_rv_guard_on", "load_bearing": True,
             "passed": bool(not crit["f1_fired"]),
             "measured": float(crit["f1_seeds_fired"]),
             "threshold": float(F1_MIN_SEEDS)},
            {"name": "F2_positive_control_guard_off_shows_decalibration",
             "load_bearing": True,
             "passed": bool(crit["f2_positive_control_live"]),
             "measured": float(crit["f2_seeds_live"]),
             "threshold": float(F2_MIN_SEEDS)},
            {"name": "F3_cold_start_contract", "load_bearing": False,
             "passed": bool(crit["f3_pass"]),
             "measured": float(sum(1 for r in rows if r["f3_cold_start_contract_ok"])),
             "threshold": float(len(rows))},
            {"name": "F4_survivable_regime", "load_bearing": False,
             "passed": bool(crit["f4_min_median_episode_length"] >= P4_MIN_MEDIAN_EPISODE_LEN),
             "measured": float(crit["f4_min_median_episode_length"]),
             "threshold": float(P4_MIN_MEDIAN_EPISODE_LEN)},
        ],
    }


def _flat_scalar(rows: List[dict], crit: dict, interp: dict) -> dict:
    out = {
        "f1_fired": int(bool(crit["f1_fired"])),
        "f1_seeds_fired": int(crit["f1_seeds_fired"]),
        "f2_positive_control_live": int(bool(crit["f2_positive_control_live"])),
        "f3_pass": int(bool(crit["f3_pass"])),
        "overall_pass": int(bool(crit["overall_pass"])),
        "n_cells": len(rows),
        "min_median_episode_length": float(crit["f4_min_median_episode_length"]),
    }
    a = [r["decalibration_away_fraction"] for r in rows if r["guard"]
         and r["decalibration_away_fraction"] == r["decalibration_away_fraction"]]
    if a:
        out["guard_on_mean_away_fraction"] = float(sum(a) / len(a))
    b = [r["decalibration_away_fraction"] for r in rows if not r["guard"]
         and r["decalibration_away_fraction"] == r["decalibration_away_fraction"]]
    if b:
        out["guard_off_mean_away_fraction"] = float(sum(b) / len(b))
    anchors = [r["anchor_precision"] for r in rows
               if r["anchor_precision"] == r["anchor_precision"]]
    if anchors:
        out["anchor_precision_mean"] = float(sum(anchors) / len(anchors))
    rr = [r["rv_over_realized_pe"] for r in rows
          if r["rv_over_realized_pe"] == r["rv_over_realized_pe"]]
    if rr:
        out["mean_final_rv_over_realized_pe"] = float(sum(rr) / len(rr))
    for p in interp["preconditions"]:
        v = p.get("measured")
        if isinstance(v, (int, float)) and not isinstance(v, bool) \
                and v == v and abs(v) != float("inf"):
            out["precondition_" + p["name"]] = float(v)
    return out


def main(dry_run: bool = False):
    seeds = (SEEDS[0],) if dry_run else SEEDS
    n_train = 3 if dry_run else N_TRAIN_EPS
    # The step CAP and the grid are NOT reduced under --dry-run. P4 asserts a
    # median episode length >= 100, so a reduced cap would make that
    # precondition unmeetable by construction and self-route the smoke to
    # substrate_not_ready_requeue for a reason that is purely an artifact of
    # smoke scaling. Only the episode COUNT and the seed list shrink.
    steps = STEPS_PER_EP

    t0 = time.perf_counter()
    rows: List[dict] = []
    for guard_label, guard in GUARD_LEVELS:
        for seed in seeds:
            rows.append(run_cell(guard_label, guard, seed, n_train, steps, dry_run))
    elapsed = time.perf_counter() - t0

    crit = _aggregate(rows)
    interp = _build_interpretation(rows, crit)
    outcome = "PASS" if crit["overall_pass"] else "FAIL"

    print(f"V3-EXQ-541d MECH-204 rv de-calibration -- {outcome} in {elapsed:.1f}s "
          f"(label={interp['label']})", flush=True)
    if dry_run:
        print("[--dry-run] smoke summary:", flush=True)
        print(json.dumps({
            "preconditions": [{"name": p["name"], "measured": p.get("measured"),
                               "met": p["met"]} for p in interp["preconditions"]],
            "mean_relative_displacement": crit["f1_mean_relative_displacement_per_cell"],
            "away_fraction": crit["away_fraction_per_cell"],
            "median_episode_length": crit["f4_median_episode_length_per_cell"],
            "c0": [{"arm": r["arm"], "captured": r["c0_captured"],
                    "target": r["c0_target"]} for r in rows],
            "anchor": {f"{r['arm']}/seed{r['seed']}": round(r["anchor_precision"], 2)
                       for r in rows},
            "rv_over_realized_pe": {f"{r['arm']}/seed{r['seed']}":
                                    round(r["rv_over_realized_pe"], 1) for r in rows},
        }, indent=1, default=str), flush=True)
        print("[--dry-run] manifest not written.", flush=True)
        return None

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    full_config = {
        "seeds": list(seeds),
        "guard_levels": [{"label": l, "guard": g} for l, g in GUARD_LEVELS],
        "n_train_eps": n_train, "steps_per_ep": steps, "grid_size": GRID_SIZE,
        "lr": LR, "sleep_loop_K": 1,
        "precision_zero_point_ema_alpha": PRECISION_ZERO_POINT_EMA_ALPHA,
        "rem_precision_recalibration_step": REM_PRECISION_RECALIBRATION_STEP,
        "proximity_harm_scale": PROXIMITY_HARM_SCALE,
        "contaminated_harm": CONTAMINATED_HARM,
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
        "outcome": outcome, "result": outcome,
        "evidence_direction": "non_contributory",
        "sleep_driver_pattern": "K=1 single-fire (SleepLoopManager, fires every episode)",
        "criteria": crit,
        "interpretation": interp,
        "arm_results": rows,
        "registered_thresholds": {
            "F1_MAX_RELATIVE_DECAL": F1_MAX_RELATIVE_DECAL,
            "F1_MIN_SEEDS": F1_MIN_SEEDS,
            "F2_MIN_POSITIVE_CONTROL": F2_MIN_POSITIVE_CONTROL,
            "F2_MIN_SEEDS": F2_MIN_SEEDS,
            "F3_SENTINEL_TOL": F3_SENTINEL_TOL,
            "P4_MIN_MEDIAN_EPISODE_LEN": P4_MIN_MEDIAN_EPISODE_LEN,
            "E3_SENTINEL_PRECISION": E3_SENTINEL_PRECISION,
        },
        "readout": _flat_scalar(rows, crit, interp),
        "diagnostics": {
            "episode_lengths_per_cell": {
                f"{r['arm']}/seed{r['seed']}": r["episode_lengths"] for r in rows},
            "done_causes_per_cell": {
                f"{r['arm']}/seed{r['seed']}": r["done_causes"] for r in rows},
            "realized_pe_variance_per_cell": {
                f"{r['arm']}/seed{r['seed']}": r["realized_pe_variance"] for r in rows},
            "enter_rem_calls_per_cycle_observed": sorted(set(
                n for r in rows for n in r["n_enter_rem_calls_per_cycle"])),
        },
        "notes": (
            "MECH-204 Option A, falsifier RE-EXPRESSED ON rv per user decision "
            "2026-09-20T09:29:55Z after two red-team BLOCKING refusals. The "
            "chip's original falsifier was on the F1 TARGET and is unfirable on "
            "any base (target ~2-25 climbing toward an anchor ~260, so the gap "
            "can only shrink); IGW-20260915-243 measured rv, not the target, "
            "moving away from calibration. F1 here is a WITHIN-CYCLE before/after "
            "comparison at the WRITEBACK and is therefore independent of episode "
            "length -- the defect that invalidated the previous build. Survivable "
            "regime chosen BY MEASUREMENT (option B): proximity_harm_scale 0.01 + "
            "contaminated_harm 0.01, the only 2-parameter setting clearing 100 "
            "ticks while still terminating by health depletion; all three "
            "candidates the decision named were eliminated by measurement, and "
            "lowering hazard_harm gave byte-identical episode lengths because "
            "contact harm is not the binding constraint (contaminated_harm "
            "defaults to 0.4/contact and was never overridden). SD-076 is NOT "
            "armed: the rv falsifier does not need a drift source, and dropping "
            "it drops the dependency on the unvalidated "
            "sd_waking_confidence_inflation_headroom repair. Episode lengths and "
            "termination causes are RECORDED (P4/diagnostics) because the "
            "previous build's verdict depended on them silently. DIAGNOSTIC, not "
            "evidence; feeds the disposition raised as GFLAG-0379."
        ),
    }
    out_dir = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments"
    out_path = write_flat_manifest(
        manifest, out_dir, dry_run=False, config=full_config, seeds=SEEDS,
        script_path=Path(__file__), started_at=t0)
    print(f"Result written to: {out_path}", flush=True)
    return outcome, out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Smoke run, no manifest.")
    args = parser.parse_args()
    result = main(dry_run=args.dry_run)
    if args.dry_run or result is None:
        sys.exit(0)
    _outcome, _out_path = result
    emit_outcome(outcome=_outcome if _outcome in ("PASS", "FAIL") else "FAIL",
                 manifest_path=_out_path, dry_run=args.dry_run)
    sys.exit(0)
