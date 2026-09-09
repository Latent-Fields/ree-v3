"""V3-EXQ-1015: MECH-465 -- z_world WARMUP-BUDGET sweep of commit-gate running-variance
dispersion (the GFLAG-0136 decision probe; diagnostic, routes a build decision, not claim credit).

red-team (fable): CONTESTED, 6 findings, all dispositioned (none dismissed) -- F1 the plateau
test was a single doubling ratio inside the DV's 15-30% sampling SE -> C2 re-specified as the
pooled ln-gain per doubling with a moving-block bootstrap 95% CI (plateau / climbing /
indeterminate), scored window 600 -> 1200 ticks; F2 "every calibration tick is a fresh select"
was false (E3 selects 1-in-10 at the sentinel; 6 rows in the smoke's window) -> calibration
median from the every-tick rv trace; F3 a raw-median threshold committed on 0.93-1.00 of selects
at every urgency level -> thr := median / (1 + u_mid) so the grid brackets the median, and C4
restated as P2-gated; F4 the fresh-select count swung ~10x with the commit regime -> DV from
the fixed-length every-tick trace, fresh-select floor demoted to recorded; F5 C3 bar was
noise-sized -> CI-gated; F6 a single-seed clear could be labelled "exhausted" -> partial-clear
branch added.

=== THE QUESTION THIS RUN SETTLES ===
MECH-465 (candidate, substrate_ceiling) asserts arousal's effect on WHETHER commitment fires
is expressible only near the commit-gate boundary. Every attempt to CREATE that boundary regime
has failed a pre-registered P2 DISPERSION FLOOR: the gated quantity -- E3's running variance
(rv) of the e2.world_forward prediction error on z_world -- had within-seed IQR/median of
0.02-0.05 against a 0.51 bar, i.e. a per-seed point mass a threshold cannot sit inside.

The 2026-08-27 boundary-regime probe reported that shortfall as 11-19x. THAT FIGURE IS
OVERSTATED ~3-4x: all three declining probes (2026-07-20, 2026-07-21, 2026-08-27) ran on an
UNTRAINED z_world -- a frozen random projection -- because the release condition
`sd_zworld_warmup_optimizer_group` was read as pending when it had been `validated` since
2026-07-22. The confound-free spike (REE_assembly
evidence/planning/mech465_zworld_warmup_dispersion_spike_20260904.md, ac852cdab1) warmed z_world
with the validated `run_zworld_p0` recipe and found within-seed IQR/median rising MONOTONICALLY
with warmup budget on 3/3 seeds: seed 0 0.0276 -> 0.0456 -> 0.0881, seed 1 0.0230 -> 0.1705 ->
0.1899, seed 3 0.0357 -> 0.1291 -> 0.1729 for COLD / WARM60 / WARM200. P2 still fails, but by
2.7-5.8x, and the trend had NOT plateaued at WARM200. Governance (GFLAG-0136, user decision
2026-09-05) routed a warmup-budget SWEEP rather than either minting the retired gate-rescale
build or closing the route as exhausted.

So this run extends the ladder -- WARM200 (spike anchor) -> WARM400 -> WARM800 -- and adds the
full PHASED recipe (P0a SD-070 encoder warmup THEN P0b e2 forward-model contrastive warmup over
the now-meaningful z_world, the ordering `zworld_p0_warmup.py` documents as mandatory), and asks
ONE pre-registered question with two readable answers:

  POSITIVE (PASS): within-seed IQR(rv)/median(rv) reaches the P2 bar 0.51 at SOME budget or
                   recipe on a seed majority (2/3). A boundary regime with graded headroom
                   EXISTS; MECH-465's residual-DV experiment becomes designable and a substrate
                   entry for the headroom regime is owed (governance mints it).
  NULL (FAIL):     the ladder PLATEAUS below the bar -- the pooled ln-gain per budget doubling
                   over WARM200/400/800 is below ln(1.15) with 95% (bootstrap) confidence. The
                   gate-rescale route is exhausted WITH the encoder confound addressed;
                   MECH-465 stays substrate_ceiling.
  THIRD READING (FAIL, indecisive): climbing below the bar with confidence (CI lower > 0, point
                   >= ln 1.15). The "gate has no headroom" premise under
                   MECH465-COMMIT-GATE-HEADROOM was an artefact of an untrained encoder;
                   governance decides whether to extend the ladder (a lettered successor is
                   permitted -- this is a diagnostic ladder, not a claim re-test) or to restate
                   MECH-465's gap.
  FOURTH READING (FAIL, honest): the CI covers both a plateau and a climb. Neither
                   "exhausted" nor "extend" is data-supported; the run records the CI and says so
                   rather than manufacturing a label from noise (red-team F1).

=== DESIGN ===
Seeds 0, 1, 3 (the spike's). Five arms per seed, each a fresh env + agent (the spike's
V3-EXQ-785a substrate config, factored into `_lib/baselines/mech465_zworld_warmup_dispersion.py`):

  COLD       no warmup                        -- baseline; must reproduce the spike's COLD row
  WARM200    P0a 200 episodes                 -- the spike's top rung, replicated in-run
  WARM400    P0a 400 episodes
  WARM800    P0a 800 episodes
  PHASED400  P0a 400 episodes THEN P0b 100 episodes of e2 forward-model contrastive warmup
             (`allon_training._e2_contrastive_step`, the SD-056 online step every driver in the
             x734/737/808/948 family runs in its P0b). The gated quantity is e2's prediction
             error, so a trained e2 is an INDEPENDENT lever on rv, tested at matched P0a budget.

Per cell, after warmup:
  1. CALIBRATION PASS (360 ticks): commitment_threshold at the never-commit sentinel 1e-9, so
     nothing commits and rv runs free of the commitment latch. rv is fed by the harness EVERY
     tick (update_running_variance on the e2.world_forward residual), so the every-tick rv
     TRACE is the calibration sample; the median over its SECOND HALF (ticks 180-359, 180
     points) is the cell's free-running median. NOTE (red-team F2): the E3 SELECT is not
     every tick -- the heartbeat clock fires E3 every e3_steps_per_tick = 10 env steps unless a
     phase reset forces one (ree_core/heartbeat/clock.py:146-157), and at the sentinel neither
     commitment entry nor the urgency interrupt fires a reset, so only ~18 fresh-select
     diagnostic rows fall in the window. Those rows are recorded for comparison; the median is
     never taken over them.
     THRESHOLD PLACEMENT (red-team F3): thr := cal_median / (1 + U_MID), U_MID = 0.19 = the
     urgency grid's mean, so the six effective thresholds thr * (1 + u) BRACKET the median
     (0.874x .. 1.126x): at mid-grid urgency the gate sits AT the median (50% commit), lower
     levels below it, higher above. A raw-median threshold puts every level above the median
     and the gate commits on 0.93-1.00 of selects at all six levels (measured in the smoke) --
     P1 unreadable for an arithmetic reason. With the bracket, P1 readability is P2-GATED BY
     ARITHMETIC and that is the diagnostic value: commit rate is graded across levels only
     once IQR/median is comparable to the +/-13% bracket, i.e. as P2 approaches its bar.
     This is the per-arm recalibration GFLAG-0136 asked for: the spike measured all arms at
     the COLD-calibrated thresholds, under which warming shifted the rv LOCATION 1.4-3.1x
     above threshold and commitment never fired (P1 unreadable for a different reason).
     WHY THE SECOND HALF, AND A FINDING ABOUT THE EARLIER WINDOWS: rv is a symmetric EMA
     (alpha 0.05) initialised at 0.5, so its init transient is 0.5 * 0.95^t -- 4.9e-3 at
     t=90, the same order as the converged rv (3e-3..1e-2), and 5e-5 only at t=180. The
     "post-warmup tick >= 90" window used by the 08-27 probe and the 09-04 spike therefore
     spent its first ~80 ticks inside the transient, which inflates a dispersion statistic
     (identically across arms, so their BETWEEN-arm trend stands; the absolute COLD level and
     the COLD-relative ratios do not -- the transient is a larger share of a smaller rv). This
     run's scored window opens after 360 converged ticks and is transient-free, so its COLD DV
     is EXPECTED to read below the spike's; the harness-fidelity check is recorded, not gating.
  2. SCORED PASS (1200 ticks) at that threshold with exogenous urgency drawn i.i.d. per tick from
     the 785a grid [0.04 .. 0.34] (`e3.config.urgency_weight = target / ||z_harm_a||`, exact to
     float roundoff); post-warmup window tick >= 90. Fresh-select diagnostic rows (cleared
     before every select_action; latched ticks record nothing -- the 785a fix) feed the
     commit-rate / gate-margin readouts only.

THE DV is within-seed IQR(rv)/median(rv) of the EVERY-TICK rv trace over the scored window
(1110 ticks per cell, fixed). Red-team F4: the fresh-select subsample's size swings ~10x with
the commit regime (a committing cell re-selects on nearly every tick through commit -> phase
reset; a non-committing one selects 1-in-10), so a DV computed on it carries arm-dependent
noise and an n floor that certifies the regime rather than the measurement. The trace is
latch-free (the harness updates rv every tick regardless of E3's cadence) and its length is
the same in every arm. The fresh-select DV is recorded alongside as
`rv_iqr_over_med_fresh_select` for continuity with the spike's statistic (the two agreed to
within 10% in the smoke). Per-cell sampling error: rv is an alpha-0.05 EMA, so the trace is
autocorrelated (~20-tick memory); a moving-block bootstrap (block 40, 400 replicates) gives
each cell's DV a 95% CI and feeds the trend test below.
The DV is SCALE-FREE by construction: a manipulation that only rescaled rv (a location shift,
which is what the 2026-08-27 probe found the boundary regime doing) is invisible to it,
deliberately. The manipulation here (encoder / forward-model training budget) is NOT
invariant under that symmetry group -- the spike measured a 3.2-8.3x change on the same
statistic. The same statement holds for every arm.
REGIME NOTE: the calibration pass runs uncommitted (1-in-10 softmax selects) and the scored
pass runs at ~50% commit by construction, so `calibration_drift` (scored median / calibration
median) conflates the threshold with the regime shift; it is recorded, never gating.

=== PRE-REGISTERED CRITERIA ===
  C1 (LOAD-BEARING)  P2_CLEARS: max over {WARM200, WARM400, WARM800, PHASED400} of the cell
                     DV >= P2_BAR (0.51) on >= 2 of 3 seeds.                      -> PASS
  any single-seed clear (n_clear 1) -> FAIL `seed_split_partial_clear` (never "exhausted"; F6).
  C2 (label when nothing clears) PLATEAU: the POOLED ln-gain per budget doubling -- mean over
                     scorable seeds of the least-squares slope of ln DV on log2 budget across
                     WARM200/400/800 -- has a 95% moving-block-bootstrap CI whose UPPER bound
                     is below ln(PLATEAU_GAIN) = ln(1.15).                        -> FAIL, route exhausted
  C2b                CLIMBING: CI lower bound > 0 and point >= ln(1.15).           -> FAIL, extend budget
  otherwise          TREND_INDETERMINATE: the CI covers both.                      -> FAIL, honest
  C3 (secondary)     PHASED lever: |ln(DV(PHASED400)/DV(WARM400))| >= ln(1.25) AND its
                     bootstrap CI excludes 0, on >= 2 seeds -> the e2 forward-model is an
                     independent lever on rv dispersion.
  C4 (secondary)     P1 READABILITY under per-arm recalibration: commit rate at every urgency
                     level in [0.05, 0.95] (the claim's conjunct-1 band). Reported per cell;
                     P2-gated by arithmetic (see threshold placement), not a verdict.

C1 is a seed-majority verdict over seeds whose ladder cells all passed their readiness gate;
fewer than 2 such seeds -> `substrate_not_ready_requeue`, never a verdict. C2/C2b pool over
the scorable seeds.

=== READINESS GATE (per arm; regime-conditioned; a red arm never vacates a green one) ===
  p0a_world_encoder_trained    WARM*/PHASED: fraction of world_encoder tensors whose weights
                               changed > 0.5 (V3-EXQ-737a's 0/4 is the failure mode; spike 4/4)
  p0a_holdout_lift_supra_floor WARM*/PHASED: SD-070 held-out grounding lift >= 0.23 (the
                               recipe's own validated band lower edge; spike 0.546-0.554)
  p0b_e2_trained               PHASED only: e2 contrastive optimiser steps >= floor and the
                               world_forward weights moved
  rv_iqr_over_med_measurable   all arms: the load-bearing statistic itself is finite and > 0
                               on the fixed-length scored trace (same statistic the verdict
                               routes on; a zero/non-finite DV is a pinned or exploded rv)
  urgency_instrument_fidelity  all arms: max |realized - assigned| urgency <= 1e-6

Recorded, NOT gating (`interpretation.recorded_preconditions`): the COLD arm's DV within
[0.5x, 2x] of the spike's per-seed COLD value (harness fidelity; the spike ran on the Mac, the
absolute rv scale is machine-class-bound -- a cloud run re-measures rather than compares); the
calibration drift (scored median / calibration median) per cell; the fresh-select row count
(>= 40) per cell; P1 readability per cell.

=== dv_headroom ===
The DV's reachable range under the manipulation IS the question, so no headroom precondition
gates this run -- a `dv_headroom_check` on the COLD control (~0.03 vs 0.51) would self-route
every run to `substrate_not_ready_requeue` and answer nothing. The ladder measurement itself is
the headroom readout: `custom_information.dv_headroom` records the per-arm achievable DV (with
its bootstrap CI) against the floor, and C2's plateau test is the falsifier of the ceiling (a
plateau below the bar IS a measured headroom ceiling). A projected doublings-to-bar figure
with CI is recorded from the pooled trend.

=== WHAT THIS RUN DOES NOT DO ===
No claim status, confidence or evidence-direction change on MECH-465 (diagnostic; evidence
direction recorded non_contributory). It does not build the gate rescale, does not mint a
substrate entry, and does not test MECH-465's own assertion -- conjunct 3 of the claim's
what_would_answer (the residual DV) needs a headroom regime FIRST, which is what this run looks
for. The p99/p1 column is recorded for continuity with the earlier probes only; at n_post
50-120 it is a tail order statistic (spike section 4).

Known asymmetry, disclosed: the PHASED400 arm's P0b rollouts call `agent.sense` (the encoder
path the scored pass also uses, so e2 trains on the same z_world it will be scored against),
which lets residue accumulate over ~4000 random-walk ticks in that arm only. `agent.reset()` is
called symmetrically on every arm before the calibration pass; residue is not reset by design.
A PHASED-vs-WARM400 difference on C3 should be read with that channel in mind.

=== STEP 2.5c SUBSTRATE-PATH OVERLAP, dispositioned ===
Open corrupting entries co-listing modules this driver imports: mode-governance-engagement
(ree_core/agent.py, utils/config.py -- the SalienceCoordinator affinity-input clamp and the
`_et_commit` latch, inert unless `salience_affinity_input_cap` is set / `use_external_task_drive`
is on; both default-off and untouched here) and SD-082 (agent.py -- the lateral-PFC rule-bias
consumer, gated on `lateral_pfc_rule_readout_consumer` / `lateral_pfc_train_rule_bias_head`,
both default-off and untouched here). NOT REACHABLE from this config. Degrading overlaps
recorded as known limitations, not blocking: SD-018 (latent/stack.py, latent/zworld_p0.py -- the
P0a recipe runs with the directional-field leg OFF, `resource_field_weight` 0.0),
SD-MECH303-THRESHOLD-SOURCING (config/env/agent), mech357 (causal_grid_world.py). Unset-severity
overlaps mech203 (agent.py salience updaters, stack.py HarmEncoder) and mech142 (config.py) noted.
Step 2.4 GOV-REUSE-1: the decisive readout `rv_iqr_over_med` is carried by NO recorded manifest
(the 09-04 spike was scratch); Step 2.5b: MECH-465 re-derive brake hits 0.

Runs on cloud `machine_affinity: any`. Continuous dispersion statistics, not sampled discrete
actions, so not exposed to the torch.multinomial cross-machine-class divergence.

RECORDING (2026-09-09, recording-only amendment; no science change, verdict grid untouched):
this driver emits a flat scalar `readout` block alongside the nested blocks. Every quantitative
block it wrote before -- cell_summary, ladder_by_seed, trend, per_arm_gate -- is keyed by arm or
seed, and the runpack converter harvests metrics.json `values` only from a FLAT scalar dict under
one of metrics / aggregates / summary_metrics / readout, so the 20260908T202858Z pack scored with
values={}. build_experiment_indexes reads only numeric metrics.values entries, so that pack could
fire no `fail_if` stop threshold, skipped the duplicate-emission supersession fingerprint, and
carried no deltas or key-metrics columns. The nested blocks are unchanged and remain the
human-readable record; `readout` is its machine-readable projection. The already-emitted
20260908T202858Z manifest is NOT retro-fixed by this -- it only affects future emissions.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiment_protocol import emit_outcome
from experiments.pack_writer import write_flat_manifest
from experiments._lib.arm_fingerprint import arm_cell
from experiments._lib.allon_training import (
    E2_CONTRASTIVE_LR,
    _e2_contrastive_step,
)
from experiments._lib.capability_eval import RandomPolicy
from experiments._lib.precondition_gate import (
    PreconditionSpec,
    aggregate_arm_gates,
    assert_no_structurally_unsatisfiable_gate,
    evaluate_arm_gate,
)
from experiments._lib.run_id import make_run_id
from experiments._lib.z_goal_stream import ZGoalStreamAccumulator
from experiments._lib.zworld_p0_warmup import run_zworld_p0
from experiments._lib.baselines.mech465_zworld_warmup_dispersion import (
    BOOT_BLOCK_TICKS,
    BOOT_REPLICATES,
    CI_LEVEL,
    CAL_EXCLUDE_TICKS,
    CAL_SENTINEL_THRESHOLD,
    CAL_TICKS,
    GATE_MARGIN_BAND,
    P0B_BUFFER_MAXLEN,
    P0_STEPS_PER_EPISODE,
    P1_BAND,
    SCORED_TICKS,
    U_MID,
    URG,
    WARMUP_EXCLUDE_TICKS,
    arm_config_slice,
    make_agent_and_env,
    make_env,
)
from experiments._metrics import check_degeneracy

EXPERIMENT_TYPE = "v3_exq_1015_mech465_zworld_warmup_budget_dispersion_sweep"
EXPERIMENT_PURPOSE = "diagnostic"
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"
CLAIM_IDS = ["MECH-465"]
RELATED_EXQ = ["V3-EXQ-785a", "V3-EXQ-981"]

SEEDS = [0, 1, 3]

# arm_id -> (p0a_episodes, p0b_episodes)
ARMS: List[Tuple[str, int, int]] = [
    ("COLD", 0, 0),
    ("WARM200", 200, 0),
    ("WARM400", 400, 0),
    ("WARM800", 800, 0),
    ("PHASED400", 400, 100),
]
LADDER = ["WARM200", "WARM400", "WARM800"]
SWEEP_ARMS = LADDER + ["PHASED400"]
P0B_STEPS_PER_EPISODE = 40

# --- pre-registered constants ------------------------------------------------------------
P2_BAR = 0.51                 # the registered P2 dispersion floor (785a pooled figure)
PLATEAU_GAIN = 1.15           # ln-gain per budget doubling below ln(1.15) = plateau
PLATEAU_LOG_GAIN = math.log(PLATEAU_GAIN)
PHASED_LEVER_LOG_GAIN = math.log(1.25)
SEED_MAJORITY = 2
# CI_LEVEL (0.95) lives in the baseline module's readout_constants (it is in the arm slice)
N_POST_FLOOR = 40             # RECORDED floor on fresh-select rows (P1 readouts), not gating
P0A_TENSOR_FRACTION_FLOOR = 0.5
P0A_HOLDOUT_LIFT_FLOOR = 0.23
P0B_MIN_STEPS = 200
# --dry-run ONLY: the smoke runs 4-16 P0a episodes at 2 epochs, far below the recipe's
# operating point, so the real floors would gate every warm cell red and leave the verdict
# grid unexercised. These relaxations never touch a real run; the effective values are
# recorded in the manifest under pre_registered_thresholds_effective.
N_POST_FLOOR_DRY = 8
P0A_HOLDOUT_LIFT_FLOOR_DRY = -1.0
P0B_MIN_STEPS_DRY = 5
URGENCY_FIDELITY_CEILING = 1e-6
# Spike COLD readings (Mac, darwin-arm64), for the RECORDED harness-fidelity check only.
SPIKE_COLD_DV = {0: 0.0276, 1: 0.0230, 3: 0.0357}
SPIKE_FIDELITY_BAND = (0.5, 2.0)

_ZG = ZGoalStreamAccumulator()


# ------------------------------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------------------------------
def _median(xs: List[float]) -> float:
    return float(statistics.median(xs)) if xs else float("nan")


def _iqr_over_med(xs: List[float]) -> float:
    if len(xs) < 4:
        return float("nan")
    a = np.asarray(xs, dtype=float)
    med = float(np.median(a))
    if med <= 0:
        return float("nan")
    return float((np.percentile(a, 75) - np.percentile(a, 25)) / med)


def _p99_over_p1(xs: List[float]) -> float:
    if len(xs) < 4:
        return float("nan")
    a = np.asarray(xs, dtype=float)
    p1 = float(np.percentile(a, 1))
    return float(np.percentile(a, 99) / p1) if p1 > 0 else float("inf")


def _finite(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def _block_bootstrap_dv(x: List[float], block: int, n_rep: int, seed: int) -> List[float]:
    """Moving-block bootstrap replicates of IQR/median for an autocorrelated series.

    rv is an alpha-0.05 EMA, so its samples are dependent (~20-tick memory); an i.i.d.
    bootstrap would understate the DV's sampling error several-fold. Blocks of `block`
    consecutive ticks are resampled with replacement until the series length is rebuilt.
    """
    a = np.asarray(x, dtype=float)
    n = len(a)
    if n < 2 * block:
        return []
    rng = np.random.default_rng(int(seed))
    n_blocks = int(math.ceil(n / block))
    out: List[float] = []
    for _ in range(int(n_rep)):
        starts = rng.integers(0, n - block + 1, size=n_blocks)
        s = np.concatenate([a[i:i + block] for i in starts])[:n]
        med = float(np.median(s))
        if med > 0:
            out.append(float((np.percentile(s, 75) - np.percentile(s, 25)) / med))
    return out


def _ci(vals: List[float], level: float = CI_LEVEL) -> Tuple[Optional[float], Optional[float]]:
    if not vals:
        return None, None
    lo = (1.0 - level) / 2.0
    return (float(np.percentile(vals, 100 * lo)), float(np.percentile(vals, 100 * (1 - lo))))


class _RngSnapshot:
    """Restore torch / numpy / python RNG streams on exit (P0b RNG neutrality)."""

    def __enter__(self):
        self._t = torch.get_rng_state()
        self._n = np.random.get_state()
        self._p = random.getstate()
        return self

    def __exit__(self, *exc):
        torch.set_rng_state(self._t)
        np.random.set_state(self._n)
        random.setstate(self._p)
        return False


def _sense(agent, od):
    return agent.sense(
        od["body_state"].unsqueeze(0), od["world_state"].unsqueeze(0),
        obs_harm=od.get("harm_obs"), obs_harm_a=od.get("harm_obs_a"),
        obs_harm_history=od.get("harm_history"),
    )


# ------------------------------------------------------------------------------------------
# warmups
# ------------------------------------------------------------------------------------------
def _p0a(agent, seed: int, episodes: int, dry_run: bool) -> Dict[str, Any]:
    """SD-070 P0a encoder warmup with the spike's non-vacuity control (weight delta)."""
    we = agent.latent_stack.split_encoder.world_encoder
    before = {n: p.detach().clone() for n, p in we.named_parameters()}
    out: Dict[str, Any] = {"p0a_episodes": int(episodes)}
    if episodes > 0:
        wenv = make_env(seed)
        wenv.reset()
        diag = run_zworld_p0(agent, wenv, seed=seed, episodes=int(episodes),
                             steps_per_episode=P0_STEPS_PER_EPISODE,
                             policy=RandomPolicy(seed), label="mech465_sweep_p0a",
                             dry_run=dry_run)
        out.update({k: v for k, v in dict(diag).items()
                    if not isinstance(v, (list, tuple))})
        out["p0a_holdout"] = diag.get("p0a_holdout")
    else:
        out.update({"p0a_recipe": "sd070", "p0a_ran": False, "p0a_reason": "COLD arm"})
    deltas = {n: float((p.detach() - before[n]).norm().item()) for n, p in we.named_parameters()}
    out["world_encoder_weight_delta_l2"] = deltas
    out["world_encoder_n_tensors"] = len(deltas)
    out["world_encoder_tensors_changed"] = sum(1 for v in deltas.values() if v > 0)
    out["world_encoder_tensor_fraction_changed"] = (
        out["world_encoder_tensors_changed"] / max(1, len(deltas)))
    return out


def _p0b(agent, seed: int, episodes: int, steps: int) -> Dict[str, Any]:
    """P0b: SD-056 e2 forward-model contrastive warmup over the (now trained) z_world.

    Transitions come from a random-walk rollout on a dedicated env, encoded through
    `agent.sense` (the path the scored pass uses). RNG-neutral: global streams are restored on
    exit so PHASED400 and WARM400 enter the scored pass on identical draws.
    """
    out: Dict[str, Any] = {"p0b_episodes": int(episodes), "p0b_steps_per_episode": int(steps),
                           "p0b_ran": False}
    if episodes <= 0:
        return out
    # e2.world_forward is a METHOD; its parameters live in world_transition + world_action_encoder
    def _wf_params():
        for n, p in agent.e2.world_transition.named_parameters():
            yield "world_transition." + n, p
        for n, p in agent.e2.world_action_encoder.named_parameters():
            yield "world_action_encoder." + n, p
    wf_before = {n: p.detach().clone() for n, p in _wf_params()}
    with _RngSnapshot():
        wenv = make_env(seed)
        e2_opt = torch.optim.Adam(agent.e2.parameters(), lr=E2_CONTRASTIVE_LR)
        buf: Deque[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = deque(maxlen=P0B_BUFFER_MAXLEN)
        srng = random.Random(int(seed) + 7919)
        pol = RandomPolicy(int(seed) + 1)
        losses: List[float] = []
        n_steps = 0
        n_ticks = 0
        for ep in range(int(episodes)):
            _f, od = wenv.reset()
            pol.reset(wenv)
            z_prev = None
            a_prev = None
            for _s in range(int(steps)):
                with torch.no_grad():
                    lat = _sense(agent, od)
                    z = lat.z_world.detach().reshape(-1).clone()
                if z_prev is not None and a_prev is not None:
                    buf.append((z_prev, a_prev, z))
                a = pol.act(wenv, od)
                onehot = torch.zeros(wenv.action_dim, device=z.device)
                onehot[int(a) % wenv.action_dim] = 1.0
                loss = _e2_contrastive_step(agent, buf, e2_opt, srng)
                if loss is not None:
                    n_steps += 1
                    if math.isfinite(loss):
                        losses.append(float(loss))
                _f, _h, done, _i, od = wenv.step(int(a))
                n_ticks += 1
                z_prev, a_prev = z, onehot
                if done:
                    break
            cur = ep + 1
            if cur == 1 or cur % 50 == 0 or cur == int(episodes):
                print("  [train] mech465_sweep_p0b seed=%d phase=P0b ep %d/%d (e2 world_forward contrastive)"
                      % (int(seed), cur, int(episodes)), flush=True)
    deltas = {n: float((p.detach() - wf_before[n]).norm().item()) for n, p in _wf_params()}
    k = max(1, len(losses) // 10)
    out.update({
        "p0b_ran": True,
        "p0b_n_ticks": n_ticks,
        "p0b_n_e2_train_steps": n_steps,
        "p0b_loss_first_decile_mean": float(np.mean(losses[:k])) if losses else None,
        "p0b_loss_last_decile_mean": float(np.mean(losses[-k:])) if losses else None,
        "e2_world_forward_weight_delta_l2": deltas,
        "e2_world_forward_tensors_changed": sum(1 for v in deltas.values() if v > 0),
        "e2_world_forward_n_tensors": len(deltas),
    })
    agent.eval()
    return out


# ------------------------------------------------------------------------------------------
# the measurement harness (spike loop, verbatim mechanics)
# ------------------------------------------------------------------------------------------
class _Harness:
    def __init__(self, agent, env, od, rng: np.random.Generator):
        self.agent, self.env, self.od, self.rng = agent, env, od, rng
        self.zp = None
        self.ap = None
        self.rows: List[Dict[str, Any]] = []
        self.n_fresh = 0
        self.n_latched = 0
        self.rv_trace: List[float] = []   # every tick, straight off the EMA

    def tick(self, t: int, phase: str) -> None:
        agent, env, od = self.agent, self.env, self.od
        assigned = float(self.rng.choice(URG))
        with torch.no_grad():
            lat = _sense(agent, od)
            zc = lat.z_world.detach()
            if self.zp is not None and self.ap is not None:
                agent.e3.update_running_variance(
                    zc - agent.e2.world_forward(self.zp, self.ap).detach())
            sig = lat.z_harm_a
            if getattr(agent.config.latent, "use_harm_un", False) and lat.z_harm_un is not None:
                sig = lat.z_harm_un
            sn = float(sig.norm(dim=-1).mean().item()) if sig is not None else 0.0
            agent.e3.config.urgency_weight = (assigned / sn) if sn > 1e-9 else 0.0
            agent.e3.last_score_diagnostics = None
            td = agent.clock.advance()
            e1 = (agent._e1_tick(lat) if td["e1_tick"]
                  else torch.zeros(1, agent.config.latent.world_dim, device=agent.device))
            cands = agent.generate_trajectories(lat, e1, td)
            action = agent.select_action(cands, td, 1.0)
        agent._step_count += 1
        self.rv_trace.append(float(agent.e3._running_variance))
        d = agent.e3.last_score_diagnostics
        if d is None or "urgency_applied" not in d:
            self.n_latched += 1
        else:
            self.n_fresh += 1
            self.rows.append(dict(
                phase=phase, tick=t, u=assigned,
                rv=float(d["commit_variance"]),
                eff=float(d["effective_threshold"]),
                realized=float(d["urgency_applied"]),
            ))
        act_idx = int(action.argmax().item()) if torch.is_tensor(action) else int(action)
        ap = torch.zeros(1, env.action_dim, device=zc.device)
        ap[0, act_idx % env.action_dim] = 1.0
        self.ap, self.zp = ap, zc
        _o, _r, done, _i, od = env.step(act_idx % env.action_dim)
        if done:
            _o, od = env.reset()
        self.od = od


def _run_cell(arm_id: str, p0a_eps: int, p0b_eps: int, seed: int, *,
              cal_ticks: int, cal_exclude: int, scored_ticks: int, exclude: int,
              dry_run: bool) -> Dict[str, Any]:
    total_ticks = cal_ticks + scored_ticks
    print(f"Seed {seed} Condition {arm_id}", flush=True)
    t_cell = time.perf_counter()
    slice_ = arm_config_slice(p0a_eps, p0b_eps, P0B_STEPS_PER_EPISODE if p0b_eps else 0)
    with arm_cell(seed, config_slice=slice_, script_path=Path(__file__),
                  config_slice_declared=True, include_driver_script_in_hash=False) as cell:
        agent, env, od, landed_thr = make_agent_and_env(seed)
        warm = _p0a(agent, seed, p0a_eps, dry_run)
        p0b = _p0b(agent, seed, p0b_eps, P0B_STEPS_PER_EPISODE)
        agent.reset()   # symmetric on every arm; does NOT reset residue (agent invariant)
        rng = np.random.default_rng(1234 + seed)
        h = _Harness(agent, env, od, rng)

        # --- calibration pass: never-commit sentinel, free-running rv --------------------
        agent.e3.config.commitment_threshold = float(CAL_SENTINEL_THRESHOLD)
        for t in range(cal_ticks):
            h.tick(t, "cal")
            if (t + 1) % 100 == 0:
                print(f"  [train] mech465_sweep seed={seed} arm={arm_id} ep {t + 1}/{total_ticks}", flush=True)
        # calibration median from the EVERY-TICK trace (180 points), not the ~18 fresh-select
        # diagnostic rows E3's 1-in-10 cadence yields at the sentinel (red-team F2)
        cal_trace = [float(v) for v in h.rv_trace[cal_exclude:cal_ticks]]
        cal_rows = [r for r in h.rows if r["phase"] == "cal" and r["tick"] >= cal_exclude]
        cal_rv_fresh = [r["rv"] for r in cal_rows]
        cal_median = _median(cal_trace)
        # place the gate so the urgency grid brackets the median (red-team F3)
        thr = (cal_median / (1.0 + U_MID)) if (_finite(cal_median) and cal_median > 0) \
            else float(CAL_SENTINEL_THRESHOLD)
        cal_commits = sum(1 for r in cal_rows if r["rv"] < r["eff"])

        # --- scored pass at the recalibrated threshold ------------------------------------
        agent.e3.config.commitment_threshold = float(thr)
        for t in range(scored_ticks):
            h.tick(t, "scored")
            gt = cal_ticks + t + 1
            if gt % 100 == 0 or gt == total_ticks:
                print(f"  [train] mech465_sweep seed={seed} arm={arm_id} ep {gt}/{total_ticks}", flush=True)
        n_fresh_scored = sum(1 for r in h.rows if r["phase"] == "scored")
        post = [r for r in h.rows if r["phase"] == "scored" and r["tick"] >= exclude]
        rv = [r["rv"] for r in post]                       # fresh-select rows (P1 readouts)
        rv_trace_scored = [float(v) for v in h.rv_trace[cal_ticks + exclude:]]  # PRIMARY DV source
        rv_med = _median(rv_trace_scored)
        dv = _iqr_over_med(rv_trace_scored)
        dv_fresh = _iqr_over_med(rv)
        dv_boot = _block_bootstrap_dv(rv_trace_scored, BOOT_BLOCK_TICKS, BOOT_REPLICATES,
                                      seed=10_000 + 97 * seed + len(arm_id))
        dv_ci_lo, dv_ci_hi = _ci(dv_boot)
        commit_by_level = {str(u): (float(np.mean([r["rv"] < r["eff"] for r in post if r["u"] == u]))
                                    if any(r["u"] == u for r in post) else None) for u in URG}
        n_by_level = {str(u): sum(1 for r in post if r["u"] == u) for u in URG}
        levels_in_band = sum(1 for v in commit_by_level.values()
                             if v is not None and P1_BAND[0] <= v <= P1_BAND[1])
        margins = [r["rv"] / r["eff"] for r in post if r["eff"] > 0]
        margin_med = _median(margins)
        margin_frac = (float(np.mean([GATE_MARGIN_BAND[0] <= m <= GATE_MARGIN_BAND[1] for m in margins]))
                       if margins else float("nan"))
        fid = max((abs(r["realized"] - r["u"]) for r in post), default=float("nan"))
        _ZG.observe(agent)

        row: Dict[str, Any] = {
            "arm_id": arm_id, "seed": seed,
            "p0a_episodes": p0a_eps, "p0b_episodes": p0b_eps,
            "from_dims_default_commitment_threshold": landed_thr,
            "threshold_calibrated": float(thr),
            "threshold_placement": {"rule": "cal_median / (1 + U_MID)", "u_mid": U_MID,
                                    "effective_threshold_over_cal_median_by_level":
                                        {str(u): (1.0 + u) / (1.0 + U_MID) for u in URG}},
            "calibration": {
                "window_ticks": [cal_exclude, cal_ticks],
                "source": "every_tick_rv_trace",
                "n_trace": len(cal_trace), "rv_median": cal_median,
                "rv_iqr_over_med_trace": _iqr_over_med(cal_trace),
                "n_fresh_rows_in_window": len(cal_rows),
                "rv_median_fresh_rows": _median(cal_rv_fresh),
                "commits_during_calibration": cal_commits,
                "n_fresh": len([r for r in h.rows if r["phase"] == "cal"]),
                "n_latched": cal_ticks - len([r for r in h.rows if r["phase"] == "cal"]),
            },
            "n_fresh_scored": n_fresh_scored,
            "n_latched_scored": scored_ticks - n_fresh_scored,
            "fresh_select_fraction_scored": n_fresh_scored / max(1, scored_ticks),
            "n_latched_ticks": h.n_latched,
            "n_post": len(post),
            "n_trace_scored_window": len(rv_trace_scored),
            "rv_med": rv_med,
            "rv_iqr_over_med": dv,
            "rv_iqr_over_med_ci95": [dv_ci_lo, dv_ci_hi],
            "rv_iqr_over_med_boot_se": (float(np.std(dv_boot)) if dv_boot else None),
            "rv_iqr_over_med_boot": [float(v) for v in dv_boot],
            "rv_iqr_over_med_fresh_select": dv_fresh,
            "rv_med_fresh_select": _median(rv),
            "rv_p99_over_p1": _p99_over_p1(rv_trace_scored),
            "commit_rate_overall": (float(np.mean([r["rv"] < r["eff"] for r in post])) if post else None),
            "calibration_drift": (rv_med / cal_median) if (_finite(cal_median) and cal_median > 0
                                                            and _finite(rv_med)) else float("nan"),
            "commit_rate_by_level": commit_by_level,
            "n_by_level": n_by_level,
            "levels_in_p1_band": levels_in_band,
            "all_levels_in_p1_band": bool(levels_in_band == len(URG)),
            "gate_margin_median": margin_med,
            "gate_margin_frac_in_band": margin_frac,
            "max_fidelity_err": float(fid),
            "warmup": warm,
            "p0b": p0b,
            "rv_post": [float(x) for x in rv],
            "rv_trace_every_tick": [float(x) for x in h.rv_trace],
            "elapsed_seconds_cell": time.perf_counter() - t_cell,
        }
        cell.stamp(row)
    return row


# ------------------------------------------------------------------------------------------
# gate specs (regime-conditioned)
# ------------------------------------------------------------------------------------------
def _specs(dry_run: bool) -> List[PreconditionSpec]:
    warm = (lambda c: c["p0a_episodes"] > 0)
    phased = (lambda c: c["p0b_episodes"] > 0)
    return [
        PreconditionSpec(
            name="p0a_world_encoder_trained",
            description="fraction of split_encoder.world_encoder tensors whose weights moved under P0a",
            control="V3-EXQ-737a's signature (0 of 4 tensors changed) is the failure mode; spike 4/4",
            threshold=P0A_TENSOR_FRACTION_FLOOR, direction="lower",
            applies_to=warm, applies_note="COLD arm runs no P0a by design; nothing to certify",
        ),
        PreconditionSpec(
            name="p0a_holdout_lift_supra_floor",
            description="SD-070 held-out grounding lift of the P0a-trained encoder",
            control="SD-070's own validated band +0.23..+0.47; spike 0.546-0.554",
            threshold=(P0A_HOLDOUT_LIFT_FLOOR_DRY if dry_run else P0A_HOLDOUT_LIFT_FLOOR),
            direction="lower",
            applies_to=warm, applies_note="COLD arm runs no P0a by design",
        ),
        PreconditionSpec(
            name="p0b_e2_trained",
            description="number of e2 contrastive optimiser steps that actually ran in P0b",
            control="a P0b whose buffer never reached the class-diversity floor trains nothing",
            threshold=(P0B_MIN_STEPS_DRY if dry_run else P0B_MIN_STEPS), direction="lower",
            applies_to=phased, applies_note="only the PHASED arm runs P0b",
        ),
        PreconditionSpec(
            name="rv_iqr_over_med_measurable",
            description=("the load-bearing statistic itself (IQR/median of the every-tick rv trace "
                         "over the scored window) is finite and strictly positive"),
            control=("the scored window is a fixed 1110-tick trace in every arm, so n does not "
                     "swing with the commit regime (red-team F4); a zero or non-finite DV means "
                     "a pinned or exploded rv, never a null"),
            threshold=0.0, direction="lower",
        ),
        PreconditionSpec(
            name="urgency_instrument_fidelity",
            description="max |realized - assigned| exogenous urgency over scored post-warmup ticks",
            control="785a / 08-27 probes: <= 5.6e-17 when ||z_harm_a|| > 0 on every tick",
            threshold=URGENCY_FIDELITY_CEILING, direction="upper",
        ),
    ]


def _measured_for(row: Dict[str, Any]) -> Dict[str, float]:
    w = row["warmup"]
    lift = w.get("p0a_holdout_mean_lift")
    return {
        "p0a_world_encoder_trained": float(w.get("world_encoder_tensor_fraction_changed", 0.0)),
        "p0a_holdout_lift_supra_floor": float(lift) if lift is not None and _finite(lift) else -1e9,
        "p0b_e2_trained": float(row["p0b"].get("p0b_n_e2_train_steps", 0) or 0),
        "rv_iqr_over_med_measurable": (float(row["rv_iqr_over_med"])
                                       if _finite(row["rv_iqr_over_med"]) else -1.0),
        "urgency_instrument_fidelity": (float(row["max_fidelity_err"])
                                        if _finite(row["max_fidelity_err"]) else 1e9),
    }


# ------------------------------------------------------------------------------------------
# main
# ------------------------------------------------------------------------------------------
def main(dry_run: bool = False) -> Dict[str, Any]:
    started_at = time.perf_counter()
    seeds = SEEDS[:1] if dry_run else list(SEEDS)
    arms = ([("COLD", 0, 0), ("WARM200", 4, 0), ("WARM400", 8, 0), ("WARM800", 16, 0),
             ("PHASED400", 8, 3)] if dry_run else list(ARMS))
    # dry-run calibration still has to clear the EMA init transient (0.5 * 0.95^150 = 2.3e-4),
    # so the calibration pass is not shortened much; the scored pass is.
    cal_ticks = 200 if dry_run else CAL_TICKS
    cal_exclude = 150 if dry_run else CAL_EXCLUDE_TICKS
    scored_ticks = 120 if dry_run else SCORED_TICKS
    exclude = 20 if dry_run else WARMUP_EXCLUDE_TICKS
    specs = _specs(dry_run)
    # the smoke runs one seed; a majority of 1 lets it exercise the whole verdict grid
    majority = 1 if dry_run else SEED_MAJORITY

    arm_ctxs = [{"id": a, "p0a_episodes": p0a, "p0b_episodes": p0b} for a, p0a, p0b in arms]
    assert_no_structurally_unsatisfiable_gate(specs, arm_ctxs)

    rows: List[Dict[str, Any]] = []
    gates: List[Dict[str, Any]] = []
    for seed in seeds:
        for arm_id, p0a, p0b in arms:
            row = _run_cell(arm_id, p0a, p0b, seed, cal_ticks=cal_ticks, cal_exclude=cal_exclude,
                            scored_ticks=scored_ticks, exclude=exclude, dry_run=dry_run)
            ctx = {"id": arm_id, "p0a_episodes": p0a, "p0b_episodes": p0b}
            gate = evaluate_arm_gate(f"{arm_id}/s{seed}", ctx, specs, _measured_for(row))
            row["gate_green"] = bool(gate["gate_green"])
            row["failed_preconditions"] = list(gate["failed_preconditions"])
            rows.append(row)
            gates.append(gate)
            _lift = row["warmup"].get("p0a_holdout_mean_lift")
            _ci95 = row["rv_iqr_over_med_ci95"]
            _ci_s = (f"[{_ci95[0]:.4f},{_ci95[1]:.4f}]" if _ci95[0] is not None else "[nan,nan]")
            print(f"  cell {arm_id} seed={seed}: dv={row['rv_iqr_over_med']:.4f} ci95={_ci_s} "
                  f"dv_fresh={row['rv_iqr_over_med_fresh_select']:.4f} "
                  f"rv_med={row['rv_med']:.5f} thr={row['threshold_calibrated']:.5f} "
                  f"drift={row['calibration_drift']:.3f} lift={_lift} "
                  f"commit={row['commit_rate_overall']} fresh_frac={row['fresh_select_fraction_scored']:.2f} "
                  f"n_post={row['n_post']} levels_in_band={row['levels_in_p1_band']}/{len(URG)} "
                  f"gate={'GREEN' if gate['gate_green'] else 'RED:' + ','.join(gate['failed_preconditions'])}",
                  flush=True)
            print(f"verdict: {'PASS' if gate['gate_green'] else 'FAIL'}", flush=True)

    agg = aggregate_arm_gates(gates)

    # --- per-seed ladder analysis --------------------------------------------------------
    by = {(r["arm_id"], r["seed"]): r for r in rows}
    arm_ids = [a for a, _, _ in arms]

    def _dv(arm: str, seed: int) -> Optional[float]:
        r = by.get((arm, seed))
        if r is None or not r["gate_green"]:
            return None
        v = r["rv_iqr_over_med"]
        return float(v) if _finite(v) else None

    def _boot(arm: str, seed: int) -> List[float]:
        r = by.get((arm, seed))
        return list(r["rv_iqr_over_med_boot"]) if r is not None else []

    # ln-gain per budget doubling over three equally spaced rungs (log2 of 200/400/800):
    # the least-squares slope is (ln d800 - ln d200) / 2
    def _slope(d200: float, d400: float, d800: float) -> float:
        return (math.log(d800) - math.log(d200)) / 2.0

    ladder: Dict[str, Dict[str, Any]] = {}
    seed_slope_boot: Dict[int, List[float]] = {}
    for seed in seeds:
        d = {a: _dv(a, seed) for a in arm_ids}
        d200, d400, d800 = d.get("WARM200"), d.get("WARM400"), d.get("WARM800")
        dph = d.get("PHASED400")
        scorable = all(v is not None and v > 0 for v in (d200, d400, d800))
        sweep_vals = [v for a, v in d.items() if a in SWEEP_ARMS and v is not None]
        clear_arms = [a for a in SWEEP_ARMS if d.get(a) is not None and d[a] >= P2_BAR]
        slope = _slope(d200, d400, d800) if scorable else None
        slope_boot: List[float] = []
        if scorable:
            b200, b400, b800 = _boot("WARM200", seed), _boot("WARM400", seed), _boot("WARM800", seed)
            n_b = min(len(b200), len(b400), len(b800))
            slope_boot = [_slope(b200[i], b400[i], b800[i]) for i in range(n_b)
                          if b200[i] > 0 and b400[i] > 0 and b800[i] > 0]
            seed_slope_boot[seed] = slope_boot
        s_lo, s_hi = _ci(slope_boot)
        phased_log_ratio = (math.log(dph / d400) if (dph and d400 and dph > 0 and d400 > 0) else None)
        phased_boot: List[float] = []
        if phased_log_ratio is not None:
            bph, b400p = _boot("PHASED400", seed), _boot("WARM400", seed)
            n_b = min(len(bph), len(b400p))
            phased_boot = [math.log(bph[i] / b400p[i]) for i in range(n_b)
                           if bph[i] > 0 and b400p[i] > 0]
        p_lo, p_hi = _ci(phased_boot)
        ladder[str(seed)] = {
            "dv_by_arm": d,
            "dv_ci95_by_arm": {a: (by[(a, seed)]["rv_iqr_over_med_ci95"] if (a, seed) in by else None)
                               for a in arm_ids},
            "scorable_ladder": scorable,
            "clears_bar": bool(clear_arms),
            "clearing_arms": clear_arms,
            "max_sweep_dv": max(sweep_vals) if sweep_vals else None,
            "gain_400_over_200": (d400 / d200) if scorable else None,
            "gain_800_over_400": (d800 / d400) if scorable else None,
            "log_slope_per_doubling": slope,
            "log_slope_ci95": [s_lo, s_hi],
            "phased_over_warm400_log_ratio": phased_log_ratio,
            "phased_over_warm400_log_ratio_ci95": [p_lo, p_hi],
            # the lever counts only if the CI excludes zero AND the point exceeds the bar
            "phased_lever": (bool(abs(phased_log_ratio) >= PHASED_LEVER_LOG_GAIN
                                  and p_lo is not None and (p_lo > 0.0 or p_hi < 0.0))
                             if phased_log_ratio is not None else None),
        }

    scorable_seeds = [s for s in seeds if ladder[str(s)]["scorable_ladder"]]
    n_scorable = len(scorable_seeds)
    n_clear = sum(1 for s in ladder.values() if s["clears_bar"])
    n_phased_lever = sum(1 for s in ladder.values() if s["phased_lever"])
    n_phased_measured = sum(1 for s in ladder.values() if s["phased_lever"] is not None)
    max_sweep_dv = max([l["max_sweep_dv"] for l in ladder.values() if l["max_sweep_dv"] is not None]
                       or [float("nan")])

    # --- pooled trend (red-team F1): mean per-seed log-slope, CI from paired replicates ------
    pooled_slope: Optional[float] = None
    pooled_ci: Tuple[Optional[float], Optional[float]] = (None, None)
    pooled_boot: List[float] = []
    if scorable_seeds:
        pooled_slope = float(np.mean([ladder[str(s)]["log_slope_per_doubling"] for s in scorable_seeds]))
        n_b = min(len(seed_slope_boot[s]) for s in scorable_seeds)
        pooled_boot = [float(np.mean([seed_slope_boot[s][i] for s in scorable_seeds])) for i in range(n_b)]
        pooled_ci = _ci(pooled_boot)
    mean_ln_d800 = (float(np.mean([math.log(ladder[str(s)]["dv_by_arm"]["WARM800"]) for s in scorable_seeds]))
                    if scorable_seeds else None)
    doublings_to_bar: Optional[float] = None
    doublings_to_bar_ci: List[Optional[float]] = [None, None]
    if (pooled_slope is not None and pooled_slope > 0 and mean_ln_d800 is not None
            and mean_ln_d800 < math.log(P2_BAR)):
        gap = math.log(P2_BAR) - mean_ln_d800
        doublings_to_bar = gap / pooled_slope
        if pooled_ci[0] is not None and pooled_ci[0] > 0:
            doublings_to_bar_ci = [gap / pooled_ci[1], gap / pooled_ci[0]]
    plateau = bool(pooled_ci[1] is not None and pooled_ci[1] < PLATEAU_LOG_GAIN)
    climbing = bool(pooled_ci[0] is not None and pooled_ci[0] > 0.0
                    and pooled_slope is not None and pooled_slope >= PLATEAU_LOG_GAIN)
    trend = {
        "statistic": "mean over scorable seeds of the per-seed ln-gain per budget doubling "
                     "(least-squares slope of ln DV on log2 budget over WARM200/400/800)",
        "pooled_log_slope_per_doubling": pooled_slope,
        "pooled_log_slope_ci95": list(pooled_ci),
        "pooled_over_seeds": scorable_seeds,
        "plateau_bar_log_gain": PLATEAU_LOG_GAIN,
        "plateau": plateau,
        "climbing": climbing,
        "mean_ln_dv_warm800": mean_ln_d800,
        "projected_doublings_to_bar": doublings_to_bar,
        "projected_doublings_to_bar_ci95": doublings_to_bar_ci,
        "pooled_log_slope_boot": pooled_boot,
    }

    def _fmt(v: Any) -> str:
        return f"{v:.3f}" if isinstance(v, (int, float)) and _finite(v) else "nan"

    # --- verdict grid ---------------------------------------------------------------------
    clearing = {s: l["clearing_arms"] for s, l in ladder.items() if l["clearing_arms"]}
    if n_scorable < majority:
        outcome, label = "FAIL", "substrate_not_ready_requeue"
        summary = (f"Only {n_scorable} seed(s) have a fully green WARM200/400/800 ladder; a "
                   f"seed-majority verdict needs {majority}. Not a verdict on the route.")
        routing = ("re-queue at an adequate P0a (inspect failed_preconditions_by_arm); do not "
                   "read the DVs of red cells")
    elif n_clear >= majority:
        outcome, label = "PASS", "p2_clears_route_open_headroom_regime_exists"
        summary = (f"IQR(rv)/median reaches the P2 bar {P2_BAR} on {n_clear}/{len(seeds)} seeds "
                   f"(clearing arms per seed: {clearing}). A graded boundary regime exists at "
                   f"that budget/recipe.")
        routing = ("governance: mint a substrate entry for the headroom regime (budget + recipe "
                   "that cleared) and route MECH-465's residual-DV experiment (what_would_answer "
                   "conjunct 3) via /queue-experiment against it; MECH465-COMMIT-GATE-HEADROOM's "
                   "gate-rescale route stays retired")
    elif n_clear >= 1:
        outcome, label = "FAIL", "seed_split_partial_clear"
        summary = (f"The bar {P2_BAR} was cleared on {n_clear} seed(s) only ({clearing}); a seed "
                   f"majority needs {majority}. The route is NOT exhausted -- a green cell cleared it.")
        routing = ("governance: neither exhausted nor yet a regime -- read which arm cleared and on "
                   "which seed; a fourth seed at that budget/recipe is the cheapest discriminator")
    elif plateau:
        outcome, label = "FAIL", "p2_plateau_below_bar_gate_rescale_route_exhausted"
        summary = (f"No arm cleared the bar (max sweep DV {_fmt(max_sweep_dv)}) and the pooled "
                   f"ln-gain per doubling {_fmt(pooled_slope)} (95% CI [{_fmt(pooled_ci[0])}, "
                   f"{_fmt(pooled_ci[1])}]) sits below ln({PLATEAU_GAIN}) = {PLATEAU_LOG_GAIN:.3f} "
                   f"with confidence: the dispersion trend has plateaued with the encoder confound "
                   f"addressed.")
        routing = ("governance: close the gate-rescale route as exhausted on "
                   "MECH465-COMMIT-GATE-HEADROOM; MECH-465 stays candidate/substrate_ceiling; a "
                   "different gated quantity is the only remaining route and both known "
                   "alternatives are unreachable from agent.py (08-27 probe section 4)")
    elif climbing:
        outcome, label = "FAIL", "p2_unplateaued_below_bar_extend_budget"
        summary = (f"No arm cleared the bar (max sweep DV {_fmt(max_sweep_dv)}) but the pooled "
                   f"ln-gain per doubling {_fmt(pooled_slope)} (95% CI [{_fmt(pooled_ci[0])}, "
                   f"{_fmt(pooled_ci[1])}]) is positive with confidence and at or above "
                   f"ln({PLATEAU_GAIN}); projected doublings to bar {_fmt(doublings_to_bar)} "
                   f"(CI [{_fmt(doublings_to_bar_ci[0])}, {_fmt(doublings_to_bar_ci[1])}]).")
        routing = ("governance decides: extend the ladder (a lettered successor at WARM1600+ is "
                   "permitted -- diagnostic ladder, not a claim re-test) or restate MECH-465's "
                   "gap: the 'gate has no headroom' premise was measured on an untrained encoder")
    else:
        outcome, label = "FAIL", "p2_below_bar_trend_indeterminate"
        summary = (f"No arm cleared the bar (max sweep DV {_fmt(max_sweep_dv)}); the pooled "
                   f"ln-gain per doubling is {_fmt(pooled_slope)} with 95% CI [{_fmt(pooled_ci[0])}, "
                   f"{_fmt(pooled_ci[1])}], which does not separate a plateau (< "
                   f"{PLATEAU_LOG_GAIN:.3f}) from a climb (> 0 and >= {PLATEAU_LOG_GAIN:.3f}). This "
                   f"run cannot tell them apart at this n.")
        routing = ("governance: NEITHER 'route exhausted' NOR 'extend budget' is data-supported; "
                   "record the CI and do not mint or close on it. More power is a fourth seed or a "
                   "WARM1600 rung (which doubles the ladder's lever arm), not a re-run of this design.")

    # --- non-degeneracy ----------------------------------------------------------------------
    groups = []
    for seed in seeds:
        g = [by[(a, seed)]["rv_iqr_over_med"] for a in arm_ids
             if (a, seed) in by and _finite(by[(a, seed)]["rv_iqr_over_med"])]
        if g:
            groups.append(g)
    degen = check_degeneracy({"rv_iqr_over_med": {"groups": groups}})

    def _sweep_varies(seed: int) -> bool:
        vals = [by[(a, seed)]["rv_iqr_over_med"] for a in arm_ids
                if (a, seed) in by and _finite(by[(a, seed)]["rv_iqr_over_med"])]
        if len(vals) < 2:
            return False
        lo, hi = min(vals), max(vals)
        return bool(lo > 0 and (hi - lo) / lo > 0.05)

    sweep_varies_by_seed = {str(s): _sweep_varies(s) for s in seeds}
    c1_non_degenerate = bool(n_scorable >= majority
                             and sum(sweep_varies_by_seed.values()) >= majority)
    if dry_run:
        # smoke gate: a DV bit-identical across nominally different budgets is a saturation
        # fingerprint (clamp / floor absorbing the manipulation), not a null
        assert any(sweep_varies_by_seed.values()), (
            "[smoke] rv_iqr_over_med does not vary (> 5%) across swept arms -- saturation "
            f"fingerprint; per-arm values: { {a: by[(a, seeds[0])]['rv_iqr_over_med'] for a in arm_ids} }")
        print(f"[smoke] sweep DV varies across arms: {sweep_varies_by_seed}", flush=True)
        print(f"[smoke] pooled log-slope {_fmt(pooled_slope)} CI [{_fmt(pooled_ci[0])}, "
              f"{_fmt(pooled_ci[1])}] plateau={plateau} climbing={climbing}", flush=True)

    criteria = [
        {"name": "C1_p2_clears_at_some_budget", "load_bearing": True,
         "passed": bool(n_clear >= majority),
         "detail": f"seeds clearing {P2_BAR}: {n_clear}/{len(seeds)} (majority {majority})"},
        {"name": "C2_pooled_trend_plateau_below_bar", "load_bearing": False,
         "passed": bool(plateau and n_clear == 0),
         "detail": (f"pooled ln-gain per doubling {_fmt(pooled_slope)}, CI95 [{_fmt(pooled_ci[0])}, "
                    f"{_fmt(pooled_ci[1])}]; plateau iff CI upper < ln({PLATEAU_GAIN})")},
        {"name": "C2b_pooled_trend_climbing_below_bar", "load_bearing": False,
         "passed": bool(climbing and n_clear == 0),
         "detail": "climbing iff CI lower > 0 and point >= ln(1.15)"},
        {"name": "C3_phased_p0b_is_independent_lever", "load_bearing": False,
         "passed": bool(n_phased_lever >= majority),
         "detail": (f"seeds with |ln(PHASED/WARM400)| >= ln(1.25) AND bootstrap CI excluding 0: "
                    f"{n_phased_lever}/{n_phased_measured} measured")},
        {"name": "C4_p1_readable_under_recalibration", "load_bearing": False,
         "passed": bool(sum(1 for r in rows if r["arm_id"] in SWEEP_ARMS and r["gate_green"]
                            and r["all_levels_in_p1_band"]) >= majority),
         "detail": ("count of green sweep cells with all 6 urgency levels' commit rate in [0.05, 0.95]; "
                    "P2-gated by arithmetic: graded only once IQR/median is comparable to the grid's "
                    "+/-13% bracket around the median")},
    ]
    combination_rule = ("outcome = PASS iff C1. C2/C2b/C3/C4 never change the outcome: on a C1 FAIL "
                        "the label is chosen in this order -- any single-seed clear -> "
                        "seed_split_partial_clear; C2 (plateau with confidence) -> route exhausted; "
                        "C2b (climbing with confidence) -> extend budget; else trend indeterminate. "
                        "C3/C4 are recorded readouts.")

    recorded_preconditions = []
    for seed in seeds:
        r = by.get(("COLD", seed))
        ref = SPIKE_COLD_DV.get(seed)
        if r is not None and ref is not None and _finite(r["rv_iqr_over_med"]):
            v = float(r["rv_iqr_over_med"])
            recorded_preconditions.append({
                "name": f"COLD/s{seed}::cold_reproduces_spike_dispersion_band",
                "kind": "recorded",
                "measured": v, "threshold_low": ref * SPIKE_FIDELITY_BAND[0],
                "threshold_high": ref * SPIKE_FIDELITY_BAND[1], "direction": "interval",
                "met": bool(ref * SPIKE_FIDELITY_BAND[0] <= v <= ref * SPIKE_FIDELITY_BAND[1]),
                "control": (f"spike COLD row seed {seed} = {ref} (Mac darwin-arm64; the absolute rv "
                            "scale is machine-class-bound, so this is recorded, not gating)"),
            })
    for r in rows:
        if _finite(r["calibration_drift"]):
            recorded_preconditions.append({
                "name": f"{r['arm_id']}/s{r['seed']}::calibration_drift_within_2x",
                "kind": "recorded", "measured": float(r["calibration_drift"]),
                "threshold_low": 0.5, "threshold_high": 2.0, "direction": "interval",
                "met": bool(0.5 <= r["calibration_drift"] <= 2.0),
                "control": ("08-27 probe: the boundary threshold moved the realised rv median "
                            "35-165% off its calibration target on an untrained encoder"),
            })
        recorded_preconditions.append({
            "name": f"{r['arm_id']}/s{r['seed']}::n_post_fresh_selects_floor",
            "kind": "recorded", "measured": float(r["n_post"]),
            "threshold": float(N_POST_FLOOR_DRY if dry_run else N_POST_FLOOR), "direction": "lower",
            "met": bool(r["n_post"] >= (N_POST_FLOOR_DRY if dry_run else N_POST_FLOOR)),
            "control": ("fresh-select rows feed only the P1 commit-rate readouts; their count swings "
                        "~10x with the commit regime (red-team F4), so this is recorded, not gating"),
        })
        recorded_preconditions.append({
            "name": f"{r['arm_id']}/s{r['seed']}::p1_all_levels_in_band",
            "kind": "recorded", "measured": float(r["levels_in_p1_band"]),
            "threshold": float(len(URG)) - 0.5, "direction": "lower",
            "met": bool(r["all_levels_in_p1_band"]),
            "control": "MECH-465 conjunct 1 band [0.05, 0.95] at every urgency level",
        })

    cell_summary = {}
    for a in arm_ids:
        vals = [by[(a, s)]["rv_iqr_over_med"] for s in seeds if (a, s) in by]
        fin = [v for v in vals if _finite(v)]
        cell_summary[a] = {
            "rv_iqr_over_med_by_seed": {str(s): by[(a, s)]["rv_iqr_over_med"] for s in seeds if (a, s) in by},
            "rv_iqr_over_med_mean": float(np.mean(fin)) if fin else None,
            "rv_iqr_over_med_ci95_by_seed": {str(s): by[(a, s)]["rv_iqr_over_med_ci95"] for s in seeds if (a, s) in by},
            "rv_iqr_over_med_fresh_select_by_seed": {str(s): by[(a, s)]["rv_iqr_over_med_fresh_select"]
                                                     for s in seeds if (a, s) in by},
            "commit_rate_overall_by_seed": {str(s): by[(a, s)]["commit_rate_overall"] for s in seeds if (a, s) in by},
            "fresh_select_fraction_by_seed": {str(s): by[(a, s)]["fresh_select_fraction_scored"]
                                              for s in seeds if (a, s) in by},
            "rv_med_by_seed": {str(s): by[(a, s)]["rv_med"] for s in seeds if (a, s) in by},
            "levels_in_p1_band_by_seed": {str(s): by[(a, s)]["levels_in_p1_band"] for s in seeds if (a, s) in by},
            "gate_green_by_seed": {str(s): by[(a, s)]["gate_green"] for s in seeds if (a, s) in by},
            "n_post_by_seed": {str(s): by[(a, s)]["n_post"] for s in seeds if (a, s) in by},
        }

    # max DV over GREEN seeds per arm: the ladder's own headroom measurement
    achievable_by_cell: Dict[str, Optional[float]] = {}
    for a in arm_ids:
        green_vals = [by[(a, s)]["rv_iqr_over_med"] for s in seeds
                      if (a, s) in by and by[(a, s)]["gate_green"]
                      and _finite(by[(a, s)]["rv_iqr_over_med"])]
        achievable_by_cell[a] = max(green_vals) if green_vals else None

    # --- flat scalar readout: the pack's metrics.values source -------------------------------
    # The runpack converter (REE_assembly evidence/experiments/scripts/sync_v3_results.py)
    # harvests metrics.json `values` from ONE of four flat spellings -- metrics / aggregates /
    # summary_metrics / readout -- and build_experiment_indexes reads ONLY the numeric entries
    # of that block. Every quantitative block this driver emitted before now (cell_summary,
    # ladder_by_seed, trend, per_arm_gate) is a dict keyed by arm or seed, so NONE of them is
    # harvestable and the 20260908T202858Z pack scored with values={}. That is not cosmetic:
    # with no numeric values (a) no `fail_if` stop threshold can fire -- the lookup returns
    # None, the check is skipped, and final_status falls back to the manifest's self-declared
    # status, which is what claim_evidence.v1.json records; (b) the duplicate-emission
    # supersession fingerprint is skipped entirely, so a byte-identical re-emission is never
    # auto-superseded and both copies score; (c) the index carries no deltas and no key-metrics
    # columns. This block is the flat scalar projection of the quantities the verdict grid
    # actually turns on. The nested blocks above are kept unchanged -- they are the
    # human-readable record; this is the machine-readable one, and they are not redundant.
    #
    # Two encoding rules, both forced by the consumer:
    #  - booleans are emitted as 0/1 ints. `_is_number()` in build_experiment_indexes excludes
    #    bool (an int subclass) on purpose, so a raw True would be silently inert here.
    #  - non-finite and None values are DROPPED rather than emitted as nan/null. A nan would be
    #    numeric to the indexer and would pollute a delta; an absent key correctly reads as
    #    unmeasured.
    # C1 routes on the MAXIMUM sweep DV (the bar is a floor -- the question is whether ANY arm
    # reaches it), so max_sweep_dv, not a minimum, is the decisive extremum for this design.
    non_degenerate_overall = bool(agg["non_degenerate"] and degen["non_degenerate"]
                                  and c1_non_degenerate)
    _pag = agg["per_arm_gate"]

    def _flat_scalar(v: Any) -> Optional[float]:
        if isinstance(v, bool):
            return float(int(v))
        if isinstance(v, (int, float)) and _finite(v):
            return float(v)
        return None

    _readout_raw: Dict[str, Any] = {
        # C1 -- the load-bearing criterion
        "max_sweep_dv": max_sweep_dv,
        "p2_bar": P2_BAR,
        "n_seeds_clearing_bar": n_clear,
        "n_scorable_seeds": n_scorable,
        "seed_majority_required": majority,
        # C2 / C2b -- the pooled trend the plateau/climb branches route on
        "pooled_log_slope_per_doubling": pooled_slope,
        "pooled_log_slope_ci95_low": pooled_ci[0],
        "pooled_log_slope_ci95_high": pooled_ci[1],
        "plateau_bar_log_gain": PLATEAU_LOG_GAIN,
        "plateau_flag": plateau,
        "climbing_flag": climbing,
        "mean_ln_dv_warm800": mean_ln_d800,
        "projected_doublings_to_bar": doublings_to_bar,
        # C3 / C4 -- recorded readouts
        "n_seeds_phased_lever": n_phased_lever,
        "n_seeds_phased_measured": n_phased_measured,
        "n_green_sweep_cells": sum(1 for r in rows
                                   if r["arm_id"] in SWEEP_ARMS and r["gate_green"]),
        "n_green_sweep_cells_all_levels_in_band": sum(
            1 for r in rows if r["arm_id"] in SWEEP_ARMS and r["gate_green"]
            and r["all_levels_in_p1_band"]),
        # per-arm gate census (the nested per_arm_gate block, as counts)
        "n_arms_green": len(_pag.get("green_arms") or []),
        "n_arms_red": len(_pag.get("red_arms") or []),
        "n_arms_structurally_vacuous": len(_pag.get("structurally_vacuous_arms") or []),
        # criteria + degeneracy census
        "n_criteria_passed": sum(1 for c in criteria if c["passed"]),
        "n_criteria_total": len(criteria),
        "non_degenerate_flag": non_degenerate_overall,
    }
    readout = {k: v for k, v in ((k, _flat_scalar(v)) for k, v in _readout_raw.items())
               if v is not None}

    manifest: Dict[str, Any] = {
        "run_id": make_run_id(EXPERIMENT_TYPE),
        "experiment_type": EXPERIMENT_TYPE,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "architecture_epoch": ARCHITECTURE_EPOCH,
        "claim_ids": CLAIM_IDS,
        "related_exq": RELATED_EXQ,
        "evidence_direction": "non_contributory",
        "evidence_direction_note": ("diagnostic: routes a build/route decision on "
                                    "MECH465-COMMIT-GATE-HEADROOM, never claim credit for MECH-465"),
        "outcome": outcome,
        "timestamp_utc": time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()),
        "dry_run": bool(dry_run),
        "seeds": seeds,
        "arms": [{"arm_id": a, "p0a_episodes": p0a, "p0b_episodes": p0b} for a, p0a, p0b in arms],
        "arm_results": rows,
        "cell_summary": cell_summary,
        "ladder_by_seed": ladder,
        "trend": trend,
        "criteria": criteria,
        "combination_rule": combination_rule,
        "per_arm_gate": agg["per_arm_gate"],
        "readout": readout,
        "interpretation": {
            "label": label,
            "summary": summary,
            "routing": routing,
            "preconditions": agg["adjudication_preconditions"],
            "recorded_preconditions": recorded_preconditions,
            "criteria_non_degenerate": {
                "C1_p2_clears_at_some_budget": c1_non_degenerate,
                "C2_trend_plateaued_below_bar": c1_non_degenerate,
                "C3_phased_p0b_is_independent_lever": bool(n_phased_measured >= majority),
                "C4_p1_readable_under_recalibration": bool(
                    sum(1 for r in rows if r["arm_id"] in SWEEP_ARMS and r["gate_green"]) >= majority),
            },
            "sweep_dv_varies_by_seed": sweep_varies_by_seed,
            "what_a_null_does_not_mean": (
                "A plateau below the bar says the gate-rescale ROUTE is exhausted on this gated "
                "quantity with the encoder trained; it is not evidence against MECH-465's "
                "assertion, which remains untested (its DV is still saturated). A PASS says a "
                "headroom regime exists; it is not evidence FOR MECH-465 either -- the residual-DV "
                "experiment has not run."),
        },
        "non_degenerate": non_degenerate_overall,
        "degeneracy_reason": "; ".join(x for x in (agg["degeneracy_reason"], degen["degeneracy_reason"],
                                                  "" if c1_non_degenerate else
                                                  "sweep DV flat or fewer than 2 scorable seeds") if x),
        "degenerate_metrics": degen["degenerate_metrics"],
        "pre_registered_thresholds": {
            "P2_BAR": P2_BAR, "PLATEAU_GAIN": PLATEAU_GAIN, "PLATEAU_LOG_GAIN": PLATEAU_LOG_GAIN,
            "CI_LEVEL": CI_LEVEL, "BOOT_BLOCK_TICKS": BOOT_BLOCK_TICKS, "BOOT_REPLICATES": BOOT_REPLICATES,
            "U_MID": U_MID, "SEED_MAJORITY": SEED_MAJORITY,
            "PHASED_LEVER_RATIO": 1.25, "P1_BAND": list(P1_BAND), "GATE_MARGIN_BAND": list(GATE_MARGIN_BAND),
            "N_POST_FLOOR": N_POST_FLOOR, "P0A_TENSOR_FRACTION_FLOOR": P0A_TENSOR_FRACTION_FLOOR,
            "P0A_HOLDOUT_LIFT_FLOOR": P0A_HOLDOUT_LIFT_FLOOR, "P0B_MIN_STEPS": P0B_MIN_STEPS,
            "URGENCY_FIDELITY_CEILING": URGENCY_FIDELITY_CEILING,
        },
        "pre_registered_thresholds_effective": {
            "dry_run_relaxed": bool(dry_run),
            "N_POST_FLOOR": (N_POST_FLOOR_DRY if dry_run else N_POST_FLOOR),
            "P0A_HOLDOUT_LIFT_FLOOR": (P0A_HOLDOUT_LIFT_FLOOR_DRY if dry_run else P0A_HOLDOUT_LIFT_FLOOR),
            "P0B_MIN_STEPS": (P0B_MIN_STEPS_DRY if dry_run else P0B_MIN_STEPS),
        },
        "custom_information": {
            "dv_headroom": {
                "name": "rv_iqr_over_med_ladder_range",
                "dv_name": "rv_iqr_over_med",
                "floor": P2_BAR,
                "description": (
                    "The DV's reachable range under the manipulation IS this run's question, so "
                    "no headroom precondition gates it (a dv_headroom_check on the COLD control, "
                    "~0.03 vs 0.51, would self-route every run to substrate_not_ready_requeue). "
                    "The ladder is the headroom measurement: achievable_by_cell is each arm's "
                    "max DV over green seeds; C2's plateau test is the falsifier of the ceiling."),
                "achievable_by_cell": achievable_by_cell,
                "headroom_ratio_by_cell": {},
                "achievable_ci95_by_cell": {
                    a: ladder[str(s)]["dv_ci95_by_arm"].get(a)
                    for a in arm_ids for s in seeds[:1]},
                "projected_doublings_to_bar": doublings_to_bar,
                "projected_doublings_to_bar_ci95": doublings_to_bar_ci,
                "margin": 1.0,
            },
            "magnitude_correction": (
                "The 2026-08-27 'P2 fails by 11-19x' figure was measured on an untrained z_world "
                "and is overstated ~3-4x; the confound-free shortfall at WARM200 is 2.7-5.8x "
                "(mech465_zworld_warmup_dispersion_spike_20260904.md)."),
            "per_arm_recalibration": (
                "threshold := post-warmup median of the free-running rv (never-commit sentinel), "
                "per cell; the spike measured every arm at the COLD-calibrated thresholds, under "
                "which the warm arms never committed (P1 unreadable)."),
        },
        "z_goal_stream_note": "agents step through sense/select only; no update_z_goal call by design",
    }
    for a, v in list(manifest["custom_information"]["dv_headroom"]["achievable_by_cell"].items()):
        manifest["custom_information"]["dv_headroom"]["headroom_ratio_by_cell"][a] = (
            (v / P2_BAR) if (v is not None and _finite(v)) else None)

    out_path = write_flat_manifest(
        manifest,
        dry_run=dry_run,
        config={
            "seeds": seeds,
            "arms": manifest["arms"],
            "schedule": {"cal_ticks": cal_ticks, "cal_exclude_ticks": cal_exclude,
                         "scored_ticks": scored_ticks,
                         "warmup_exclude_ticks": exclude, "p0_steps_per_episode": P0_STEPS_PER_EPISODE,
                         "p0b_steps_per_episode": P0B_STEPS_PER_EPISODE, "p0b_buffer_maxlen": P0B_BUFFER_MAXLEN,
                         "cal_sentinel_threshold": CAL_SENTINEL_THRESHOLD},
            "urgency_grid": list(URG),
            "baseline_slice": arm_config_slice(0, 0, 0),
            "thresholds": manifest["pre_registered_thresholds"],
        },
        seeds=seeds,
        script_path=Path(__file__),
        started_at=started_at,
        z_goal_stream_stats=_ZG.stats(),
    )
    manifest["_out_path"] = str(out_path)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="V3-EXQ-1015 MECH-465 z_world warmup-budget dispersion sweep")
    parser.add_argument("--dry-run", action="store_true",
                        help="1 seed, tiny budgets, short passes; manifest relocated out of evidence/")
    args = parser.parse_args()

    result = main(dry_run=args.dry_run)
    out_path = result["_out_path"]

    print()
    print("=== V3-EXQ-1015 MECH-465 z_world warmup-budget dispersion sweep ===")
    print(f"label:   {result['interpretation']['label']}")
    print(f"outcome: {result['outcome']}")
    print(f"summary: {result['interpretation']['summary']}")
    print(f"routing: {result['interpretation']['routing']}")
    print("--- DV (IQR/med of rv) by arm x seed ---")
    for a, c in result["cell_summary"].items():
        vals = " ".join(f"s{s}={v:.4f}" if _finite(v) else f"s{s}=nan"
                        for s, v in c["rv_iqr_over_med_by_seed"].items())
        print(f"  {a:10s} {vals}  green={c['gate_green_by_seed']}  p1_levels={c['levels_in_p1_band_by_seed']}")
    print(f"manifest: {out_path}")

    _outcome_raw = str(result["outcome"]).upper()
    emit_outcome(
        outcome=_outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL",
        manifest_path=out_path,
        dry_run=args.dry_run,
    )
