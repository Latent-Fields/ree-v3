"""
V3-EXQ-999a -- MECH-161 (LC-NE inverted-U vigilance), INSTRUMENT-FIXED re-queue of V3-EXQ-999.

supersedes: V3-EXQ-999

=============================== 999a HEADER ===============================
WHY A LETTER. Same scientific question as V3-EXQ-999 -- is hazard-detection sensitivity an
INTERIOR-optimum function of the E3 heartbeat period P, or monotone (MECH-026's "faster is
better")? -- with the readout repaired per the CONFIRMED autopsy
failure_autopsy_V3-EXQ-999_2026-09-04 (Fable red-team CONTESTED at Step 7c, all six findings
accepted; Step 8 gate held 2026-09-04). 999 aborted at its FIRST seed on a P0 gate whose
statistic a PERFECT avoider also fails (+0.0000 vs a +0.05 floor) and whose policy-free nulls
score -0.06 to -0.44: the run measured nothing about the substrate. Registry question
mech161_vigilance_arousal_shape_vs_readout, three legs, ALL adjudicated by this run:
H-readout-invalid (measurement), H-avoidance-untrained (learning-signal), H-env-band-absent
(environment).

THE FIVE NON-NEGOTIABLES (autopsy routing_note_2026_09_04), AND HOW EACH IS MET:
  (1) DECISION-TICK SCORING. The avoidance readout is scored ONLY at E3 decision ticks (the
      ticks at which a fresh action is selected), never at held steps. 999's per-step readout on
      a walker held for P steps read GEOMETRY, not decision: the hold period IS the independent
      variable, so sensitivity(P) was monotone in P under a hazard-blind null and the DV was
      confounded with the IV for ANY agent (red-team F2). `_agent_tick` now returns whether this
      step made a fresh decision; `_run_eval_level` records every step (diagnostic) and scores
      the decision ticks (load-bearing). Readiness `null_sensitivity_flat_in_P` checks that the
      confound is gone: the hazard-blind null's decision-tick sensitivity must be flat across P.
  (2) MEASURED ORACLE AND NULL ARMS AT EVERY P DERIVE THE BAR. At each swept P, two policy-free
      arms run on the SAME env/seed with the SAME decision schedule (decide every P steps, hold
      between): CONDITIONAL_ORACLE (at a decision tick in the HIGH bin take the away-from-hazard
      argmax, otherwise a uniformly random action -- the threat-gated avoider the red team showed
      scores +0.905 unheld; an ALWAYS-avoid oracle scores exactly 0 and is NOT the reference) and
      HAZARD_BLIND_NULL (uniformly random at every decision tick). The P0 bar is DERIVED:
      agent_baseline(P_train) >= null(P_train) + BAR_FRACTION * (oracle - null)(P_train), and
      C1 is scored on the NORMALISED sensitivity s_norm(P) = (agent - null)/(oracle - null) at
      each P, so a P-dependent null cannot masquerade as a P-dependent agent.
  (3) P0 OVER ALL SEEDS. No seed-one abort. Every seed runs every level and both control arms;
      readiness is evaluated on the seed set: instrument validity (oracle - null >= ORACLE_NULL_
      SEPARATION_FLOOR at every P, worst seed), coverage_ok (SAFE and HIGH decision-tick bins
      both >= MIN_BIN_COVERAGE_TICKS in every (seed, level, arm) cell), and the derived baseline
      bar (mean over seeds). A failed readiness gate is CLASSIFIED (readiness_failure_class):
      `instrument` when the oracle/null arms themselves fail to separate or a bin is starved,
      `avoidance_untrained` when the instrument is valid but the trained agent does not clear
      the derived bar (H-avoidance-untrained SUPPORTED -- a real finding, routed to the
      learning-signal axis, not to another readout iteration), `env_band_absent` when the
      quantile SAFE bin is populated only by wall-clamped steps (H-env-band-absent).
  (4) PER-EPISODE QUANTILE BINS. At each episode reset the env's own `hazard_field` (the
      quantity `hazard_field_view` is a normalised window of) is read over all non-wall cells;
      SAFE = value <= its HAZARD_Q_LOW quantile, HIGH = value >= its HAZARD_Q_HIGH quantile, so
      both bins are populated on a 10x10 grid whatever the hazard placement (999's fixed
      SAFE < 0.15 meant Manhattan d >= 12, unreachable for a centred hazard). Readiness
      `hazard_field_replica_matches_obs` confirms the observed centre value equals the field at
      the agent's cell / max (the same replica discipline as 822e's atanh check). The
      wall-clamp fraction of the designated avoidant action in the SAFE bin is recorded.
  (5) RECORDING. Per (seed, level, arm): per-bin decision-tick n/avoidant, hit_rate,
      false_alarm_rate, sensitivity (decision-tick AND all-step, the latter diagnostic),
      coverage_ok, wall-clamp fraction, n_e3_ticks, committed_fraction, world_forward_r2 --
      stamped in every case, including a not-ready verdict (999 stamped only
      {seed, p0_not_ready}).

TRAINING IS UNCHANGED from 999 (H-avoidance-untrained is adjudicated, not pre-empted): the
warmup trains e1, the e2 world-forward model and the e3 harm_eval head and NO action policy on
harm; NAV_BIAS=0.25 pushes a quarter of training actions toward the hazard. Whether avoidance is
nonetheless expressed through E3's harm-evaluated candidate selection is exactly what the valid
readout now measures at P_train.

VERDICT LOGIC (C1 unchanged in form, on the normalised curve): the level with maximum mean
s_norm is INTERIOR and beats both extremes by >= max(MARGIN_ABS_FLOOR_NORM, 1 sd pooled) ->
PASS, supports; extreme wins or margin not met -> FAIL, weakens (monotone / indeterminate);
DV flat across levels (range < DV_MOVEMENT_FLOOR_NORM) -> non_degenerate false,
non_contributory; any readiness gate red -> substrate_not_ready_requeue / non_contributory with
the failure class above. The claim scope stays the SHAPE premise (operating-point sweep with
the MECH-093 regulator's output overwritten each tick); MECH-161's REGULATOR assertion is not
tested here (autopsy Section 5) and the manifest says so.

dv_headroom (W5 rule): `dv_headroom_interior_margin` -- the instrument's raw dynamic range
(oracle - null) at the WORST level must reach 2x MARGIN_ABS_FLOOR (the raw-scale size of the
normalised margin floor), floor_headroom, worst level, margin 2.0. Substrate readiness (2.5/2.5a):
MECH-093's rate knob live and forced per tick as in 999 (manipulation-reach fix retained); the
open DEGRADING entry dv-dynamic-range-precondition-class lists MECH-161 (this run's readiness
gates are the repair it names). Re-derive brake (2.5b): MECH-161 count 0 (autopsy); the 999
target itself counts as an instrument hit, not a ceiling.

red-team (Step 4.5, DIFFERENT model, model=fable, 2026-09-08). PASS 1 BLOCKING, 10 findings, all
verified against the source: F1 (BLOCKING: agent.update_residue at eval fired MECH-091's harm-
salience phase_reset (agent.py:10756), so the agent's decision schedule was NOT the forced P --
at slow P nearly every HIGH-bin decision tick was a harm-triggered reset tick) -> no residue
update at eval + readiness gate decision_schedule_matches_P (realised ticks / (steps/P) within
1 +/- 0.25; the commitment-boundary reset sites live inside agent.select_action, which this
harness bypasses); F2 (first agent tick at step P-1, a random held action before it) recorded;
F3 (_hazard_cells read the grid, which drops a hazard the agent walks onto) -> env.hazards; F4
(Monte Carlo: flat truth gave `weakens` 63 percent) -> `weakens` requires the MIRRORED margin
(extreme best beats the best interior level by the same noise-aware requirement), else
`inconclusive`; F5 (null flatness tautological in expectation) -> recorded, not gating; F6
(separation/headroom certify the oracle's construction) -> kept as construction checks; F7
(coverage min over the AGENT arm too) -> control-arm gates vs agent-arm gates with their own
failure classes; F8 (P0 bar scored on reset ticks) closed by F1; F9 (no substrate mechanism
makes a fresh decision worse at short P; the regulator is bypassed) acknowledged: the run
tests the SHAPE premise as an operating-point sweep; F10 (arms not paired) -> each arm rebuilds
its env from the seed per level. PASS 2 (re-spawned once, F1 changed the causal chain)
CONTESTED: F1/F3/F5/F7/F8 CLOSED, F6 dismissed with citation, F4 partial, F2/F9/F10 open-low;
N1 (env default contamination_spread 0.5 kills a STAY-holding agent by self-contamination at
step 7, coupling survival to P through the held action) -> contamination_spread 0.0 on training
and eval envs (V3-EXQ-513 precedent), episode_end_reasons recorded; N2 (a starved control bin
propagated as nan into the separation gate and classified `instrument` ahead of
`env_band_absent`) -> worst FINITE cell, env_band_absent classified first; N3 (flat-truth
false-`supports` 16-19 percent at n=3) -> 5 seeds, MARGIN_SD_MULT 1.5, >= 3 contributing seeds
per level; N4 (arms paired only while episode lengths match) recorded. No third pass.
=== 999 HEADER (inherited; superseded where the 999a header says so) ===
V3-EXQ-999 -- MECH-161 (LC-NE inverted-U vigilance): does ready-vigilance
detection performance peak at an INTERIOR heartbeat-frequency level, or does
it keep improving monotonically as MECH-093's E3 clock is sped up (the
MECH-026 "set precision/sensitivity high" alternative)?

RED-TEAM: model "fable" (2026-09-03). Verdict BLOCKING on first pass, self-
verified against source (agent.py:5866, clock.py:201-221, and this script's
own _run_eval_level/_agent_tick), FIXED, and re-verified by direct probe
(see "MANIPULATION-REACH FIX" below) -- now CLEAR on the manipulation-reach
axis. Family 2 (bin-coverage) and Family 4 (torch.multinomial non-determinism
on the uncommitted E3 path) were also raised; both are handled by this
script's existing MIN_BIN_COVERAGE_STEPS coverage gate (Step 3.5 self-review)
and by machine-affinity-pinning + reporting committed-fraction (see
"MACHINE-CLASS NOTE" below) rather than by further code changes.

Claims: MECH-161 ("Ready vigilance (MECH-026) requires an arousal regulator
that maintains an optimal sensitivity level on the LC-NE inverted-U curve,
implemented via MECH-093 heartbeat frequency modulation rather than a binary
high/low precision switch.")

EXPERIMENT_PURPOSE = "evidence" (direct falsifier test of MECH-161's SHAPE
claim: interior-optimum vs monotone-improving detection performance as a
function of E3 heartbeat rate).

SUBSTRATE READINESS (Step 2.5/2.5a, confirmed by source read + this script's
own P0 gate):
  - MECH-093 (ree_core/heartbeat/clock.py MultiRateClock.update_e3_rate_from_
    beta / _current_e3_steps) is IMPLEMENTED and WIRED: e3_steps_per_tick is a
    live, continuous knob (HeartbeatConfig.beta_rate_min_steps=5 ..
    beta_rate_max_steps=20), not a binary switch. Confirmed by direct source
    read 2026-09-03 (ree-v3 CLAUDE.md line 17 cross-referenced against the
    actual clock.py implementation).
  - MANIPULATION-REACH FIX (found by Step 4.5 red-team, self-verified against
    source, then fixed here -- read before touching _agent_tick again). The
    ORIGINAL claim in this section was WRONG: the custom _agent_tick harness
    DOES still call agent._e1_tick(latent) every step (e1_steps_per_tick=1,
    so ticks["e1_tick"] is true on virtually every step -- agent.py:373-377),
    and _e1_tick unconditionally calls
    self.clock.update_e3_rate_from_beta(latent_state.z_beta) at agent.py:5866
    with NO gating flag. update_e3_rate_from_beta overwrites
    self._current_e3_steps (clock.py:221) based on live z_beta magnitude.
    Net effect, TRACED THROUGH clock.advance()'s actual read order
    (clock.py:127-160): _run_eval_level forces _current_e3_steps=P only at
    level/episode ENTRY (before step 1's advance() call, so step 1 alone
    reads the forced P correctly) -- but step 1's _e1_tick then overwrites
    _current_e3_steps for step 2's advance() call, and every step thereafter
    for the rest of the episode reads whatever MECH-093's live z_beta dynamics
    computed, NOT the forced P. Verified directly against source (not merely
    trusted from the red-team) 2026-09-03: agent.py:5866 has no `if` guard at
    all, and _run_eval_level (then at the equivalent of the current
    _agent_tick call site) forced _current_e3_steps only once per
    level/episode, never per-step. So for 27 of the 28 planned eval steps per
    episode-that-uses-P, the swept variable P was silently inert and E3 ran
    at whatever period MECH-093 chose -- a manipulation-cannot-reach-DV defect
    that would have made every downstream hit_rate/false_alarm_rate/
    sensitivity(P) reading a measurement of MECH-093's own arousal dynamics,
    not of P.
    THE FIX: _agent_tick now takes an optional force_heartbeat: Optional[int]
    parameter; when set, it re-asserts
    agent.clock._current_e3_steps = int(force_heartbeat) immediately AFTER
    the _e1_tick() call (i.e. after MECH-093 has run and overwritten it, but
    BEFORE the next step's advance() call reads it) so the forced period
    survives from one advance() call to the next. _run_eval_level passes
    force_heartbeat=heartbeat_steps; _train_warmup passes force_heartbeat=None
    (training deliberately runs under MECH-093's live, uncontrolled dynamics
    -- a fixed rate is only the eval-time experimental manipulation, not a
    training precondition). MECH-093 (update_e3_rate_from_beta) still RUNS on
    every eval step under this fix -- nothing bypasses or disables it -- its
    output for that tick is simply overwritten before the next advance() call
    consumes it, which is the intended "operating-point sweep, trained agent
    held fixed" semantics (WORKSPACE_STATE 2026-08-22 "operating-point sweep"
    entries), now actually achieved rather than merely intended. Re-verified
    by a standalone probe (agent.clock._current_e3_steps read at every step
    of a 15-step eval-level rollout, seed 42, P=5 and P=20, matching this
    script's own env config): with force_heartbeat=None (the pre-fix
    behaviour) both P=5 and P=20 traces collapse onto the SAME
    z_beta-governed natural-dynamics trajectory by step 2
    ([5,16,14,13,12,...] vs [20,16,14,12,11,...] -- identical at step 2,
    nearly identical thereafter, small residual divergence from step 4
    onward is action-dependent z_beta feedback, not the forced P); with
    force_heartbeat=P (the fix), both traces stay pinned at the forced value
    on every one of the 15 steps ([5,5,5,...] and [20,20,20,...]).
  - MECH-026 / ARC-016 / ARC-044 (MECH-161's other depends_on) are conceptual/
    architectural claims with no separate substrate gate of their own beyond
    MECH-093's clock and the existing E3 selection + hazard-avoidance
    machinery already exercised by V3-EXQ-981 (claims.yaml: neither
    implementation_phase nor v3_pending set on any of the three).

GOV-REUSE-1 (Step 2.4): decisive readout is
  "interior-vs-extreme heartbeat-level detection sensitivity margin"
  (see DECISIVE READOUT below). claims.yaml MECH-161 carries evidence: []
  (zero prior runs). Searched REE_assembly/scripts/reanalysis_query.py for
  any manifest carrying a hazard-detection-sensitivity-by-heartbeat-rate
  readout on a compatible substrate_hash: none found (no prior script sweeps
  e3_steps_per_tick against a signal-detection DV; V3-EXQ-097/097b/116/505
  probe MECH-093's mechanics directly, not detection performance; V3-EXQ-981
  probes MECH-027's hypervigilance signature via precision/horizon/replay
  forcing, never heartbeat rate). Not recoverable -> run.

RE-DERIVE BRAKE (Step 2.5b): zero prior autopsies exist for MECH-161 (zero
runs total) -- count is 0, below the >=2 threshold. Not braked.

SUBSTRATE-PATH OVERLAP (Step 2.5c): substrate_queue.json has no open
`corrupting`-severity entry whose substrate_paths overlap
ree_core/heartbeat/clock.py, ree_core/agent.py's clock-tick path, or
ree_core/hippocampal/module.py's propose_trajectories (checked 2026-09-03;
grep for corrupting-severity entries found none touching these paths).

WHY THIS DESIGN (avoiding a tautological inverted-U). A naive design that
scores "hit rate" purely from whether a PERIODIC E3 tick lands inside a
fixed-width window after a scheduled hazard injection is degenerate: a
periodic sampler's window-coverage probability is a property of interval
arithmetic (period P vs window width W), identical for "signal" and "noise"
windows unless something ties tick TIMING to the hazard's true onset -- and
MECH-091's phase_reset (a DIFFERENT mechanism) would trivially saturate hit
rate to 1.0 at every P if invoked, confounding MECH-093 with MECH-091 and
making the manipulation reach a phase_reset-dominated DV rather than a
heartbeat-rate-dominated one. This script therefore:
  (a) NEVER calls clock.phase_reset() -- only the periodic/forced-rate path
      is exercised, isolating MECH-093 from MECH-091.
  (b) Does NOT operationalise "detection" as tick-timing-vs-injection-timing
      at all. It reuses the SAME hazard_field_view-based signal-detection
      apparatus already validated in V3-EXQ-981 (_hazard_cells /
      _avoidant_action / _hazard_bin): whether the AGENT'S OWN chosen action
      (from a genuinely TRAINED E3 selection, frozen at eval) matches the
      geometrically-correct "move away from the hazard" action, binned by
      the hazard field's live proximity value at the agent's cell (SAFE <
      0.15, AMBIGUOUS, HIGH >= 0.50 -- thresholds reused verbatim from
      V3-EXQ-981's calibration against this identical env config). This is a
      genuine behavioural readout of a trained policy, not clock-phase
      arithmetic: whether the agent takes the correct evasive action depends
      on how STALE its held action selection is (how many env steps have
      elapsed since E3 last looked), which is a real function of the
      manipulated period P.
  (c) hit_rate(P) = avoidant-action rate in the HIGH hazard bin (correct
      evasion under genuine threat). false_alarm_rate(P) = avoidant-action
      rate in the SAFE hazard bin (evasive action with no real threat
      nearby -- taking flight from nothing).
      sensitivity(P) = hit_rate(P) - false_alarm_rate(P).
      This is NOT guaranteed to be inverted-U by construction: hit_rate could
      plausibly be flat-then-saturating in P (favouring MECH-026's monotone
      story) if staleness never actually degrades correct evasion within the
      swept P range, or false_alarm_rate could stay flat at every P (no
      hypervigilance cost ever appears), either of which would produce a
      monotone or flat sensitivity(P) curve -- a genuine, falsifiable
      alternative outcome, not a guaranteed result.

DV-SYMMETRY INVARIANCE (Step 3 MANDATORY declaration). The manipulated
variable is P = agent.clock._current_e3_steps (an integer period, forced at
each eval level's entry). sensitivity(P) is computed from realised per-step
argmax action selections taken by a frozen, already-trained policy under
env/agent state that depends on P-controlled staleness of the held action.
This is NOT a uniform additive constant (P does not add a constant to E3's
candidate scores; it changes WHEN a fresh score vector is computed at all,
so it directly gates the ACTION SEQUENCE the agent executes, not merely a
score offset that argmax would cancel), NOT a monotone rescaling of an
order-based statistic (hit_rate/false_alarm_rate are behavioural EVENT RATES
over realised trajectories, not a re-ranking of a fixed candidate set), and
NOT a permutation of interchangeable units (env steps within an episode are
not interchangeable -- hazard position and agent position are causally
ordered in time). The manipulation genuinely reaches the DV.

DECISIVE READOUT: sensitivity(P) = hit_rate(P) - false_alarm_rate(P), at each
of 5 pre-registered heartbeat levels P in HEARTBEAT_LEVELS, averaged across
3 seeds. Pre-registered non-monotonicity test (see NON_MONOTONICITY test
below): the level with maximum mean sensitivity must be INTERIOR (neither
the fastest nor the slowest swept level) AND must beat BOTH extreme levels'
mean sensitivity by a margin >= max(MARGIN_ABS_FLOOR, MARGIN_SD_MULT *
pooled_cross_seed_SD). Positive control (DV moves at all): the range of mean
sensitivity across the 5 levels must exceed DV_MOVEMENT_FLOOR, else the run
self-reports non_degenerate=False rather than a false "weakens".

Routing:
  - P0 gate fails (baseline-level avoidance not trained above
    POSITIVE_CONTROL_MARGIN) -> FAIL, interpretation.label
    "substrate_not_ready_requeue", evidence_direction "non_contributory".
  - P0 gate passes but sensitivity(P) is flat across levels (positive control
    fails) -> FAIL, non_degenerate=False, evidence_direction
    "non_contributory".
  - P0 + positive control pass, interior level wins by the pre-registered
    margin -> PASS, evidence_direction "supports" (inverted-U; MECH-161).
  - P0 + positive control pass, best level is an extreme (or interior wins
    but not by the margin) -> FAIL, evidence_direction "weakens" (monotone or
    indeterminate; MECH-026's "higher is always better" story not
    contradicted).

No sleep loop is used (use_sleep_loop left at REEConfig.from_dims default,
i.e. off) -- MECH-161 makes no sleep-dependent claim, so no SLEEP DRIVER
line is needed (Step 3 sleep-driver-pattern rule N/A).

PROGRESS-INSTRUMENTATION NOTE (deliberate, documented deviation): the
decisive readout is a CROSS-SEED aggregate (mean sensitivity per heartbeat
level, averaged over all 3 seeds), so there is exactly ONE authoritative
verdict for the whole run, printed once at the very end of run_experiment
(on every path, including the P0-not-ready early-abort). The queue entry
therefore declares seeds=1, conditions=1 (matching the single verdict line),
NOT seeds=3 -- episodes_per_run is WARMUP_EPISODES (the per-seed training
loop's own denominator). Each seed's training phase still prints its own
"Seed S Condition C" boundary line and "[train] ep N/M" progress (3 boundary
resets before the single final verdict), so the runner's live progress bar
will under/over-count slightly against a literal 1-run expectation -- a
known, accepted UX imprecision, not a correctness issue (the manifest's
outcome/metrics/criteria are unaffected).
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from experiment_protocol import emit_outcome  # noqa: E402
from experiments._lib.arm_fingerprint import arm_cell, reset_all_rng  # noqa: E402
from experiments._lib.z_goal_stream import ZGoalStreamAccumulator  # noqa: E402
from experiments._metrics import (  # noqa: E402
    check_degeneracy,
    dv_headroom_check,
)
from experiments.pack_writer import write_flat_manifest  # noqa: E402
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402


EXPERIMENT_TYPE = "v3_exq_999a_mech161_vigilance_inverted_u_heartbeat"
QUEUE_ID = "V3-EXQ-999a"
SUPERSEDES = "V3-EXQ-999"
REGISTRY_QUESTION = "mech161_vigilance_arousal_shape_vs_readout"
EXPERIMENT_PURPOSE = "evidence"
CLAIM_IDS = ["MECH-161"]
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"

# Seed 44 excluded (documented reef-config early-death instability, CLAUDE.md).
SEEDS = [11, 23, 37, 45, 53]     # 999 used 3; pass-2 N3: n=3 left a 16-19 percent flat-truth false-supports rate (MC)
CONTAMINATION_SPREAD = 0.0       # pass-2 N1: the env default 0.5 kills a STAY-holding agent by self-contamination at exactly step 7,
                                 # which couples episode survival to P through the held action (lethal at P >= 8, survivable at P = 5) --
                                 # a route into s_norm(P) unrelated to hazard detection. V3-EXQ-513 precedent: pass 0.0 explicitly.
                                 # Applied to TRAINING and EVAL envs alike so the trained agent and its sweep share one regime.

# ---- Env (identical to V3-EXQ-981's calibrated config -- same hazard-field
# distribution, so the HAZARD_SAFE_MAX/HIGH_MIN thresholds below carry over
# without re-deriving them) --------------------------------------------------
ENV_SIZE = 10
NUM_HAZARDS = 1
NUM_RESOURCES = 2
HAZARD_HARM = 0.05
HAZARD_FIELD_DECAY = 0.5

# ---- Agent dims --------------------------------------------------------------
WORLD_DIM = 32
SELF_DIM = 32

# ---- Training (single warmup per seed, at the HeartbeatConfig canonical
# default rate -- the sweep below only changes EVAL-time heartbeat rate on
# this one frozen trained policy, so "different P" never confounds with
# "different training") -------------------------------------------------------
WARMUP_EPISODES = 150
STEPS_PER_EPISODE = 100
NAV_BIAS = 0.25          # mild bias toward the hazard during warmup so the
                         # agent experiences enough near-hazard states to
                         # learn a meaningful harm-avoidance signal
                         # (identical rationale/value to V3-EXQ-981).

# ---- Heartbeat sweep (Step 3 MANDATORY pre-registration) -------------------
# e3_steps_per_tick values, pre-registered BEFORE any run. Spans
# HeartbeatConfig's documented arousal envelope (beta_rate_min_steps=5 ..
# beta_rate_max_steps=20). TRAIN_HEARTBEAT_STEPS is also HEARTBEAT_LEVELS[2]
# (the middle/interior level) -- the canonical HeartbeatConfig.e3_steps_per_
# tick default (10), so training happens at a level that is also evaluated,
# with no post-hoc cherry-picking.
HEARTBEAT_LEVELS = [5, 8, 10, 15, 20]
TRAIN_HEARTBEAT_STEPS = 10
assert TRAIN_HEARTBEAT_STEPS in HEARTBEAT_LEVELS

# ---- Eval --------------------------------------------------------------------
EVAL_EPISODES_PER_LEVEL = 30     # 999 used 20; raised for decision-tick bin coverage (the readout now scores ~1 tick in P)
EVAL_STEPS_PER_EPISODE = 100

# ---- Hazard-signal bins (pre-registered; reused verbatim from V3-EXQ-981's
# calibration against this identical env config) -----------------------------
HAZARD_SAFE_MAX = 0.15
HAZARD_HIGH_MIN = 0.50
HAZARD_BIN_SAFE = "SAFE"
HAZARD_BIN_AMBIGUOUS = "AMBIGUOUS"
HAZARD_BIN_HIGH = "HIGH"
MIN_BIN_COVERAGE_STEPS = 5

# ---- Pre-registered thresholds (Step 3 MANDATORY -- fixed before any run) --
POSITIVE_CONTROL_MARGIN = 0.05   # P0 gate: baseline-level (hit - false_alarm)
                                  # must exceed this for avoidance to count as
                                  # "trained at all".
DV_MOVEMENT_FLOOR = 0.03         # positive control: range of mean
                                  # sensitivity across the 5 levels must
                                  # exceed this, else the DV is flat/degenerate.
MARGIN_ABS_FLOOR = 0.03          # 999's raw-scale floor: kept as the RAW headroom anchor (dv_headroom).
MARGIN_SD_MULT = 1.5             # pass-2 N3: 1.0 at n=3 left ~17 percent flat-truth false-supports; 1.5 at n=5 (both directions)
MIN_SEEDS_PER_LEVEL = 3          # a level with fewer contributing seeds has no usable sd -> C1 inconclusive

# ---- 999a: decision-tick readout, quantile bins, oracle/null-derived bars ---
HAZARD_Q_LOW = 0.35              # SAFE = field value <= this quantile over non-wall cells (bottom 35 percent: d >= ~6 on 10x10)
HAZARD_Q_HIGH = 0.65             # HIGH = field value >= this quantile (top 35 percent: d <= ~4). Smoke at 0.25/0.75 left the
                                 # HIGH bin under-covered for the control arms (which die on hazard contact) and the executable
                                 # SAFE bin thin (far cells sit at walls); 0.35/0.65 keeps the bins disjoint with an AMBIGUOUS band
CONTROL_EPISODES_MULT = 3        # control arms are policy-free and cheap; run 3x the agent's episodes per level for bin coverage
MIN_BIN_COVERAGE_TICKS = 10      # decision ticks per bin per (seed, level, arm) cell
ORACLE_NULL_SEPARATION_FLOOR = 0.20   # instrument validity: (oracle - null) decision-tick sensitivity, worst (seed, level)
BAR_FRACTION = 0.25              # P0: agent_baseline >= null + BAR_FRACTION * (oracle - null), at P_train, mean over seeds
NULL_FLATNESS_CEIL = 0.10        # non-negotiable (1) check: range across P of the null's mean decision-tick sensitivity
MARGIN_ABS_FLOOR_NORM = 0.05     # C1 on the NORMALISED curve s_norm = (agent - null)/(oracle - null)
DV_MOVEMENT_FLOOR_NORM = 0.05    # positive control on the normalised curve
WALL_CLAMP_CEIL = 0.50           # env leg: SAFE-bin decision ticks whose avoidant action is a wall no-op, worst cell
REPLICA_TOL = 1e-4               # observed hazard_field_view centre vs env.hazard_field[cell]/max
SCHEDULE_TOL = 0.25              # AGENT decision ticks / (steps/P) must sit within 1 +/- this (red-team F1)
HEADROOM_MARGIN = 2.0
ARMS = ["AGENT", "CONDITIONAL_ORACLE", "HAZARD_BLIND_NULL"]


def build_config(env: CausalGridWorldV2) -> REEConfig:
    """ONE config shared by every seed/level -- only agent.clock._current_
    e3_steps is forced at eval-level entry; nothing about agent structure
    changes across levels."""
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=SELF_DIM,
        world_dim=WORLD_DIM,
        reafference_action_dim=env.action_dim,
        use_event_classifier=True,
    )
    cfg.heartbeat.e3_steps_per_tick = TRAIN_HEARTBEAT_STEPS
    return cfg


def config_slice() -> Dict[str, Any]:
    """Exactly what each seed's computation reads -- no acceptance thresholds."""
    return {
        "env": "CausalGridWorldV2",
        "env_size": ENV_SIZE,
        "num_hazards": NUM_HAZARDS,
        "num_resources": NUM_RESOURCES,
        "hazard_harm": HAZARD_HARM,
        "hazard_field_decay": HAZARD_FIELD_DECAY,
        "world_dim": WORLD_DIM,
        "self_dim": SELF_DIM,
        "warmup_episodes": WARMUP_EPISODES,
        "steps_per_episode": STEPS_PER_EPISODE,
        "train_heartbeat_steps": TRAIN_HEARTBEAT_STEPS,
        "heartbeat_levels": HEARTBEAT_LEVELS,
        "eval_episodes_per_level": EVAL_EPISODES_PER_LEVEL,
        "hazard_safe_max": HAZARD_SAFE_MAX,
        "hazard_high_min": HAZARD_HIGH_MIN,
    }


def _random_onehot(action_dim: int, device) -> torch.Tensor:
    v = torch.zeros(1, action_dim, device=device)
    v[0, random.randint(0, action_dim - 1)] = 1.0
    return v


def _hazard_cells(env: CausalGridWorldV2) -> List[Tuple[int, int]]:
    # 999a (red-team F3): read the env's hazard LIST, not the grid -- stepping onto a hazard
    # erases it from env.grid until the next drift, which made `avoidant` None on exactly the
    # HIGH-bin ticks that matter and forced misses on every arm.
    hz = getattr(env, "hazards", None)
    if hz:
        return [(int(h[0]), int(h[1])) for h in hz]
    hz = np.argwhere(env.grid == env.ENTITY_TYPES["hazard"])
    return [(int(x), int(y)) for x, y in hz]


def _avoidant_action(env: CausalGridWorldV2, hazard_cells: List[Tuple[int, int]]) -> Optional[int]:
    """The single grid move maximising the dot product with the
    away-from-nearest-hazard vector. Reused verbatim from V3-EXQ-981."""
    if not hazard_cells:
        return None
    ax, ay = int(env.agent_x), int(env.agent_y)
    hx, hy = min(hazard_cells, key=lambda h: abs(h[0] - ax) + abs(h[1] - ay))
    away_dx, away_dy = ax - hx, ay - hy
    best_a, best_score = None, -1e18
    for a, (dx, dy) in env._action_map.items():
        score = dx * away_dx + dy * away_dy
        if score > best_score:
            best_score = score
            best_a = a
    return best_a


def _hazard_bin(value: float) -> str:
    if value < HAZARD_SAFE_MAX:
        return HAZARD_BIN_SAFE
    if value >= HAZARD_HIGH_MIN:
        return HAZARD_BIN_HIGH
    return HAZARD_BIN_AMBIGUOUS


class _TickState:
    def __init__(self) -> None:
        self.last_action: Optional[torch.Tensor] = None
        self.z_self_prev: Optional[torch.Tensor] = None
        self.action_prev: Optional[torch.Tensor] = None
        # MACHINE-CLASS NOTE (red-team Family 4): E3.select() samples via
        # torch.multinomial on the UNCOMMITTED path (result.committed=False),
        # which is not bit-reproducible across machine classes (darwin-arm64
        # vs linux-x86_64/torch versions -- see umbrella CLAUDE.md "Running
        # the test suite" cross-machine-class note). We do not attempt to
        # force committed=True (that would be a different, artificial
        # manipulation); instead we RECORD the committed fraction per level
        # so a cross-machine replication can check whether a result depends
        # on the uncommitted-path RNG stream rather than on the assert-
        # ability of pinning e3.select() to the committed branch only.
        self.n_e3_ticks: int = 0
        self.n_committed_ticks: int = 0


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


def _agent_tick(
    agent: REEAgent,
    env: CausalGridWorldV2,
    obs_dict: Dict[str, Any],
    state: _TickState,
    world_dim: int,
    device,
    train: bool,
    force_heartbeat: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """One environment tick. Custom harness (mirrors V3-EXQ-981's _agent_tick)
    that calls agent.clock.advance() directly rather than going through
    agent.step()'s normal path -- but note this does NOT keep
    self.clock.update_e3_rate_from_beta from running: agent._e1_tick() is
    still called below (e1_steps_per_tick=1, so this fires on virtually every
    step) and _e1_tick unconditionally calls update_e3_rate_from_beta
    (agent.py:5866, no gating flag), which overwrites
    agent.clock._current_e3_steps based on live z_beta every time it runs.
    See the module docstring "MANIPULATION-REACH FIX" section for the full
    trace and the red-team finding this fixes.

    force_heartbeat: when not None, re-asserts
    agent.clock._current_e3_steps = int(force_heartbeat) immediately AFTER
    the _e1_tick() call each step, so the forced period survives into the
    NEXT step's agent.clock.advance() call regardless of what MECH-093 just
    computed from z_beta. Pass None (the default) to let MECH-093 run
    uncontrolled (the training-time behaviour); pass the swept P value during
    eval (see _run_eval_level).

    Returns (chosen action [1,action_dim], z_world [1,world_dim] detached, fresh_decision).
    999a: `fresh_decision` is True on the steps where E3 actually selected (an e3_tick with
    candidates) -- the decision ticks the readout is scored at (non-negotiable (1)).
    """
    fresh_decision = False
    obs_body = obs_dict["body_state"]
    obs_world = obs_dict["world_state"]
    ctx = torch.no_grad() if not train else _nullcontext()
    with ctx:
        latent = agent.sense(obs_body, obs_world)
        if state.z_self_prev is not None and state.action_prev is not None:
            agent.record_transition(state.z_self_prev, state.action_prev, latent.z_self.detach())
        ticks = agent.clock.advance()
        e1_prior = (
            agent._e1_tick(latent)
            if ticks.get("e1_tick", False)
            else torch.zeros(1, world_dim, device=device)
        )
        if force_heartbeat is not None:
            # Re-assert AFTER _e1_tick (which just overwrote this via
            # MECH-093/update_e3_rate_from_beta) so the forced period is what
            # the NEXT advance() call reads. MECH-093 still ran this tick --
            # nothing is bypassed -- only its output is not allowed to
            # persist past this tick.
            agent.clock._current_e3_steps = int(force_heartbeat)
        if ticks.get("e3_tick", False):
            candidates = agent.hippocampal.propose_trajectories(
                latent.z_world, latent.z_self, e1_prior=e1_prior,
            )
            if candidates:
                result = agent.e3.select(candidates, temperature=1.0)
                state.last_action = result.selected_action.detach()
                state.n_e3_ticks += 1
                fresh_decision = True
                if bool(getattr(result, "committed", False)):
                    state.n_committed_ticks += 1
        action = state.last_action
        if action is None:
            action = _random_onehot(env.action_dim, device)
            state.last_action = action

        drive_level = REEAgent.compute_drive_level(obs_body)
        benefit_exposure = max(0.0, float(obs_dict.get("benefit_exposure", 0.0)))
        agent.update_z_goal(benefit_exposure=benefit_exposure, drive_level=drive_level)

    state.z_self_prev = latent.z_self.detach()
    state.action_prev = action.detach()
    return action, latent.z_world.detach(), fresh_decision


def _train_warmup(
    agent: REEAgent,
    env: CausalGridWorldV2,
    optimizer: optim.Optimizer,
    wf_optimizer: optim.Optimizer,
    harm_eval_optimizer: optim.Optimizer,
    num_episodes: int,
    steps_per_episode: int,
    world_dim: int,
    device,
) -> Dict[str, Any]:
    """Phased P0/P1 warmup: encoder + world-forward + harm-eval head trained
    on .detach()ed latents (no joint E3-through-encoder gradient flow).
    Structure mirrors V3-EXQ-981's _train_warmup (validated pattern)."""
    agent.train()
    state = _TickState()
    harm_buf_pos: List[torch.Tensor] = []
    harm_buf_neg: List[torch.Tensor] = []
    wf_buf: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    total_harm = 0

    for ep in range(num_episodes):
        flat_obs, obs_dict = env.reset()
        agent.reset()
        state = _TickState()
        z_world_prev: Optional[torch.Tensor] = None

        for _ in range(steps_per_episode):
            action, z_world_curr_pre, _fresh = _agent_tick(
                agent, env, obs_dict, state, world_dim=world_dim, device=device, train=True,
            )

            # nav_bias: with probability NAV_BIAS, override toward the
            # nearest hazard so training sees enough near-hazard states.
            if random.random() < NAV_BIAS:
                hz = _hazard_cells(env)
                if hz:
                    ax, ay = int(env.agent_x), int(env.agent_y)
                    hx, hy = min(hz, key=lambda h: abs(h[0] - ax) + abs(h[1] - ay))
                    dx, dy = hx - ax, hy - ay
                    best_a, best_score = None, -1e18
                    for a, (adx, ady) in env._action_map.items():
                        score = adx * dx + ady * dy
                        if score > best_score:
                            best_score = score
                            best_a = a
                    if best_a is not None:
                        action = _random_onehot(env.action_dim, device) * 0.0
                        action[0, best_a] = 1.0
                        state.last_action = action
                        state.action_prev = action.detach()

            flat_obs, harm_signal, done, info, obs_dict = env.step(action)
            agent.update_residue(
                harm_signal=float(harm_signal), world_delta=None,
                hypothesis_tag=False, owned=True,
            )

            theta_z = agent.theta_buffer.summary()
            if z_world_prev is not None:
                wf_buf.append((z_world_prev.cpu(), state.action_prev.cpu(), z_world_curr_pre.cpu()))
                if len(wf_buf) > 2000:
                    wf_buf = wf_buf[-2000:]
            if harm_signal < 0:
                total_harm += 1
                harm_buf_pos.append(theta_z.detach())
                if len(harm_buf_pos) > 1000:
                    harm_buf_pos = harm_buf_pos[-1000:]
            else:
                harm_buf_neg.append(theta_z.detach())
                if len(harm_buf_neg) > 1000:
                    harm_buf_neg = harm_buf_neg[-1000:]

            e1_loss = agent.compute_prediction_loss()
            if e1_loss.requires_grad:
                optimizer.zero_grad()
                e1_loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.e1.parameters(), 1.0)
                optimizer.step()

            if len(wf_buf) >= 16:
                k = min(32, len(wf_buf))
                idxs = torch.randperm(len(wf_buf))[:k].tolist()
                zw_b = torch.cat([wf_buf[i][0] for i in idxs]).to(device)
                a_b = torch.cat([wf_buf[i][1] for i in idxs]).to(device)
                zw1_b = torch.cat([wf_buf[i][2] for i in idxs]).to(device)
                wf_loss = F.mse_loss(agent.e2.world_forward(zw_b, a_b), zw1_b)
                if wf_loss.requires_grad:
                    wf_optimizer.zero_grad()
                    wf_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        list(agent.e2.world_transition.parameters())
                        + list(agent.e2.world_action_encoder.parameters()), 1.0,
                    )
                    wf_optimizer.step()

            if len(harm_buf_pos) >= 4 and len(harm_buf_neg) >= 4:
                k_p = min(16, len(harm_buf_pos))
                k_n = min(16, len(harm_buf_neg))
                pi = torch.randperm(len(harm_buf_pos))[:k_p].tolist()
                ni = torch.randperm(len(harm_buf_neg))[:k_n].tolist()
                zw_b = torch.cat([harm_buf_pos[i] for i in pi] + [harm_buf_neg[i] for i in ni], dim=0)
                target = torch.cat([
                    torch.ones(k_p, 1, device=device), torch.zeros(k_n, 1, device=device),
                ], dim=0)
                pred = agent.e3.harm_eval(zw_b)
                harm_loss = F.mse_loss(pred, target)
                if harm_loss.requires_grad:
                    harm_eval_optimizer.zero_grad()
                    harm_loss.backward()
                    torch.nn.utils.clip_grad_norm_(agent.e3.harm_eval_head.parameters(), 0.5)
                    harm_eval_optimizer.step()

            z_world_prev = z_world_curr_pre
            if done:
                break

        if (ep + 1) % 50 == 0 or ep == num_episodes - 1:
            print(f"  [train] ep {ep+1}/{num_episodes}  harm={total_harm}", flush=True)

    return {"total_harm": total_harm, "wf_buf": wf_buf}


def _compute_world_forward_r2(agent: REEAgent, wf_buf: List, n_test: int = 200) -> float:
    if len(wf_buf) < n_test:
        return 0.0
    idxs = list(range(len(wf_buf) - n_test, len(wf_buf)))
    with torch.no_grad():
        zw = torch.cat([wf_buf[i][0] for i in idxs])
        a = torch.cat([wf_buf[i][1] for i in idxs])
        zw1 = torch.cat([wf_buf[i][2] for i in idxs])
        pred = agent.e2.world_forward(zw, a)
        ss_res = ((zw1 - pred) ** 2).sum()
        ss_tot = ((zw1 - zw1.mean(dim=0, keepdim=True)) ** 2).sum()
    return float((1 - ss_res / (ss_tot + 1e-8)).item())


# --------------------------------------------------------------------------- 999a readout
def _field_quantiles(env: CausalGridWorldV2) -> Tuple[np.ndarray, float, float, float]:
    """Per-STEP quantile thresholds (non-negotiable (4); per-episode in the autopsy's wording, but
    hazards drift within an episode so the field is re-read every step) from the env's OWN hazard_field --
    the quantity hazard_field_view is a normalised 5x5 window of -- over every non-wall cell.
    Returns (normalised field, max, q_low, q_high)."""
    field = np.asarray(env.hazard_field, dtype=np.float64)
    fmax = float(field.max()) + 1e-6
    norm = field / fmax
    wall_id = env.ENTITY_TYPES.get("wall", None)
    mask = np.ones_like(norm, dtype=bool) if wall_id is None else (env.grid != wall_id)
    vals = norm[mask]
    return norm, fmax, float(np.quantile(vals, HAZARD_Q_LOW)), float(np.quantile(vals, HAZARD_Q_HIGH))


def _bin_q(value: float, q_low: float, q_high: float) -> str:
    if value <= q_low:
        return HAZARD_BIN_SAFE
    if value >= q_high:
        return HAZARD_BIN_HIGH
    return HAZARD_BIN_AMBIGUOUS


def _wall_clamped(env: CausalGridWorldV2, action_idx: Optional[int]) -> bool:
    """True when the designated action is a no-op: off-grid on a non-toroidal grid, or into a
    wall cell. Recorded per SAFE-bin decision tick (autopsy Section 3a)."""
    if action_idx is None:
        return True
    dx, dy = env._action_map[action_idx]
    nx, ny = int(env.agent_x) + dx, int(env.agent_y) + dy
    if getattr(env, "toroidal", False):
        nx, ny = nx % env.size, ny % env.size
    elif not (0 <= nx < env.size and 0 <= ny < env.size):
        return True
    wall_id = env.ENTITY_TYPES.get("wall", None)
    return bool(wall_id is not None and env.grid[nx, ny] == wall_id)


def _summarise_rows(decision_rows: List[Dict[str, Any]], all_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    def _bins(rows):
        bins = {b: {"n": 0, "avoidant": 0, "wall_clamped": 0, "n_executable": 0, "avoidant_executable": 0} for b in
                (HAZARD_BIN_SAFE, HAZARD_BIN_AMBIGUOUS, HAZARD_BIN_HIGH)}
        for r in rows:
            b = bins[r["hazard_bin"]]
            b["n"] += 1
            b["avoidant"] += int(r["took_avoidant"])
            b["wall_clamped"] += int(r["wall_clamped"])
            if not r["wall_clamped"]:
                b["n_executable"] += 1
                b["avoidant_executable"] += int(r["took_avoidant"])
        # The SAFE-bin rate is scored over EXECUTABLE ticks only: a tick whose designated avoidant
        # action is a wall no-op cannot express 'flight from nothing' (999 autopsy 3a: 93-100 percent
        # of its SAFE steps were exactly that). HIGH-bin ticks near a wall keep the raw rate --
        # there the avoidant action points AWAY from the hazard and is executable by construction
        # unless the hazard sits between the agent and the wall, which the executable count records.
        rates = {k: (v["avoidant"] / v["n"] if v["n"] > 0 else None) for k, v in bins.items()}
        rates[HAZARD_BIN_SAFE] = (bins[HAZARD_BIN_SAFE]["avoidant_executable"] / bins[HAZARD_BIN_SAFE]["n_executable"]
                                  if bins[HAZARD_BIN_SAFE]["n_executable"] > 0 else None)
        return bins, rates
    dbins, drates = _bins(decision_rows)
    abins, arates = _bins(all_rows)
    hit = drates[HAZARD_BIN_HIGH]
    fa = drates[HAZARD_BIN_SAFE]
    safe_n = dbins[HAZARD_BIN_SAFE]["n"]
    return {
        "n_decision_ticks": len(decision_rows),
        "n_steps": len(all_rows),
        "decision_bins": dbins, "decision_rates": drates,
        "hit_rate": hit, "false_alarm_rate": fa,
        "sensitivity": (hit - fa) if (hit is not None and fa is not None) else None,
        "coverage_ok": bool(dbins[HAZARD_BIN_SAFE]["n_executable"] >= MIN_BIN_COVERAGE_TICKS
                            and dbins[HAZARD_BIN_HIGH]["n"] >= MIN_BIN_COVERAGE_TICKS),
        "safe_executable_n": int(dbins[HAZARD_BIN_SAFE]["n_executable"]),
        "high_n": int(dbins[HAZARD_BIN_HIGH]["n"]),
        "safe_bin_wall_clamp_fraction": (dbins[HAZARD_BIN_SAFE]["wall_clamped"] / safe_n) if safe_n else None,
        # all-step (held-walker) version: 999's confounded statistic, DIAGNOSTIC only
        "allstep_bins": abins, "allstep_rates": arates,
        "allstep_sensitivity": ((arates[HAZARD_BIN_HIGH] - arates[HAZARD_BIN_SAFE])
                                if (arates[HAZARD_BIN_HIGH] is not None and arates[HAZARD_BIN_SAFE] is not None) else None),
    }


def _run_control_level(env_seed: int, heartbeat_steps: int, num_episodes: int,
                       steps_per_episode: int, policy: str, seed: int, device) -> Dict[str, Any]:
    """Policy-free control arm (non-negotiable (2)) on the SAME decision schedule as the agent:
    decide at step 0 and every P steps, hold between. CONDITIONAL_ORACLE: in the HIGH bin the
    away-from-hazard argmax, else uniform random (the threat-GATED avoider; an always-avoid
    oracle scores 0 by construction and is not the reference). HAZARD_BLIND_NULL: uniform random
    at every decision tick. Own RNG stream so the controls never perturb the agent's."""
    rng = random.Random(seed * 1000 + heartbeat_steps + (7 if policy == "CONDITIONAL_ORACLE" else 3))
    # red-team F10: each arm gets its OWN env built from the same seed, so the three arms see the
    # same board sequence (paired), rather than continuing one env's RNG across arms
    env = CausalGridWorldV2(seed=env_seed, size=ENV_SIZE, num_hazards=NUM_HAZARDS, num_resources=NUM_RESOURCES,
                            hazard_harm=HAZARD_HARM, hazard_field_decay=HAZARD_FIELD_DECAY, contamination_spread=CONTAMINATION_SPREAD)
    decision_rows: List[Dict[str, Any]] = []
    all_rows: List[Dict[str, Any]] = []
    replica_err = 0.0
    c_end_reasons: List[str] = []
    for ep in range(num_episodes):
        _flat, obs_dict = env.reset()
        held_idx = rng.randrange(env.action_dim)
        for step_idx in range(steps_per_episode):
            # hazards DRIFT within an episode, so the field (and its quantiles) is re-read
            # every step -- a reset-time read left a 0.27 replica error in the first smoke
            norm, fmax, q_lo, q_hi = _field_quantiles(env)
            hazard_cells = _hazard_cells(env)
            hv = obs_dict.get("hazard_field_view", None)
            hazard_value = float(hv[12]) if hv is not None else 0.0
            replica_err = max(replica_err, abs(hazard_value - float(norm[int(env.agent_x), int(env.agent_y)])))
            hbin = _bin_q(hazard_value, q_lo, q_hi)
            avoidant = _avoidant_action(env, hazard_cells)
            fresh = (step_idx % int(heartbeat_steps) == 0)
            if fresh:
                if policy == "CONDITIONAL_ORACLE" and hbin == HAZARD_BIN_HIGH and avoidant is not None:
                    held_idx = int(avoidant)
                else:
                    held_idx = rng.randrange(env.action_dim)
            row = {"hazard_bin": hbin, "took_avoidant": bool(avoidant is not None and held_idx == avoidant),
                   "wall_clamped": _wall_clamped(env, avoidant)}
            all_rows.append(row)
            if fresh:
                decision_rows.append(row)
            action = torch.zeros(1, env.action_dim, device=device)
            action[0, held_idx] = 1.0
            _flat, _harm, done, _info, obs_dict = env.step(action)
            if done:
                c_end_reasons.append(str(_info.get("episode_end_reason", "done")) if isinstance(_info, dict) else "done")
                break
    out = _summarise_rows(decision_rows, all_rows)
    out.update({"arm": policy, "episode_end_reasons": c_end_reasons, "heartbeat_steps": heartbeat_steps, "replica_err_max": replica_err,
                "n_e3_ticks": len(decision_rows), "committed_fraction": None})
    return out


def _run_eval_level(agent: REEAgent, env: CausalGridWorldV2, heartbeat_steps: int, num_episodes: int,
                    steps_per_episode: int, world_dim: int, device, zg: ZGoalStreamAccumulator) -> Dict[str, Any]:
    """The trained, frozen agent at a forced E3 period P (999's forced-period machinery kept
    verbatim), scored at DECISION ticks with per-episode quantile bins."""
    agent.clock._current_e3_steps = int(heartbeat_steps)
    agent.clock._e3_phase_step = 0
    agent.eval()
    decision_rows: List[Dict[str, Any]] = []
    all_rows: List[Dict[str, Any]] = []
    fatal = 0
    n_e3 = 0
    n_committed = 0
    replica_err = 0.0
    end_reasons: List[str] = []
    for ep in range(num_episodes):
        _flat, obs_dict = env.reset()
        agent.reset()
        agent.clock._current_e3_steps = int(heartbeat_steps)
        state = _TickState()
        for step_idx in range(steps_per_episode):
            norm, fmax, q_lo, q_hi = _field_quantiles(env)   # per step: hazards drift
            hazard_cells = _hazard_cells(env)
            hv = obs_dict.get("hazard_field_view", None)
            hazard_value = float(hv[12]) if hv is not None else 0.0
            replica_err = max(replica_err, abs(hazard_value - float(norm[int(env.agent_x), int(env.agent_y)])))
            hbin = _bin_q(hazard_value, q_lo, q_hi)
            avoidant = _avoidant_action(env, hazard_cells)
            try:
                action, _zw, fresh = _agent_tick(agent, env, obs_dict, state, world_dim=world_dim,
                                                 device=device, train=False, force_heartbeat=heartbeat_steps)
            except Exception:
                fatal += 1
                action = _random_onehot(env.action_dim, device)
                state.last_action = action
                fresh = False
            chosen_idx = int(action.argmax(dim=-1).item())
            row = {"hazard_bin": hbin, "took_avoidant": bool(avoidant is not None and chosen_idx == avoidant),
                   "wall_clamped": _wall_clamped(env, avoidant)}
            all_rows.append(row)
            if fresh:
                decision_rows.append(row)
            _flat, harm_signal, done, _info, obs_dict = env.step(action)
            if done:
                end_reasons.append(str(_info.get("episode_end_reason", "done")) if isinstance(_info, dict) else "done")
            # 999a (red-team F1, BLOCKING): NO agent.update_residue() at eval. In 999 that call
            # fired MECH-091's harm-salience phase_reset (agent.py:10756) on every proximity-harm
            # step, so the agent's E3 decision schedule was NOT the forced period P -- at slow P
            # nearly every HIGH-bin decision tick was a harm-triggered reset tick. The agent is
            # frozen at eval; residue accumulation is not part of the manipulation. The other
            # phase_reset sites (commitment-boundary entry/release inside agent.select_action)
            # are not on this harness's path, which calls e3.select directly. Adherence to the
            # forced schedule is MEASURED per level (`decision_schedule_matches_P`).
            if done:
                break
        n_e3 += state.n_e3_ticks
        n_committed += state.n_committed_ticks
        print(f"  [eval] heartbeat_steps={heartbeat_steps} ep {ep+1}/{num_episodes} "
              f"decision_ticks={len(decision_rows)} steps={len(all_rows)}", flush=True)
    zg.observe(agent)
    out = _summarise_rows(decision_rows, all_rows)
    expected_ticks = sum(1 for r in all_rows) / float(heartbeat_steps)
    out.update({"arm": "AGENT", "heartbeat_steps": heartbeat_steps, "fatal_errors": fatal,
                "expected_decision_ticks_at_P": float(expected_ticks),
                "episode_end_reasons": end_reasons,
                "decision_ticks_over_expected": (len(decision_rows) / expected_ticks) if expected_ticks > 0 else None,
                "replica_err_max": replica_err, "n_e3_ticks": n_e3, "n_committed_ticks": n_committed,
                "committed_fraction": (n_committed / n_e3) if n_e3 > 0 else None})
    return out


def run_seed(seed: int, dry_run: bool) -> Dict[str, Any]:
    device = torch.device("cpu")
    reset_all_rng(seed)
    print(f"\nSeed {seed} Condition vigilance_inverted_u_heartbeat_sweep_999a", flush=True)
    env = CausalGridWorldV2(seed=seed, size=ENV_SIZE, num_hazards=NUM_HAZARDS, num_resources=NUM_RESOURCES,
                            hazard_harm=HAZARD_HARM, hazard_field_decay=HAZARD_FIELD_DECAY, contamination_spread=CONTAMINATION_SPREAD)
    cfg = build_config(env)
    agent = REEAgent(cfg).to(device)
    optimizer = optim.Adam(list(agent.e1.parameters()), lr=1e-3)
    wf_optimizer = optim.Adam(list(agent.e2.world_transition.parameters()) + list(agent.e2.world_action_encoder.parameters()), lr=1e-3)
    harm_eval_optimizer = optim.Adam(list(agent.e3.harm_eval_head.parameters()), lr=1e-4)
    warmup_eps = 5 if dry_run else WARMUP_EPISODES
    warmup_steps = 15 if dry_run else STEPS_PER_EPISODE
    # dry-run eval budget is deliberately NOT toy-sized on the readout side: the instrument
    # gates (coverage, oracle/null separation, null flatness) must be exercisable in the smoke
    eval_eps = 6 if dry_run else EVAL_EPISODES_PER_LEVEL
    eval_steps = 60 if dry_run else EVAL_STEPS_PER_EPISODE
    train_out = _train_warmup(agent, env, optimizer, wf_optimizer, harm_eval_optimizer,
                              warmup_eps, warmup_steps, WORLD_DIM, device)
    world_forward_r2 = _compute_world_forward_r2(agent, train_out["wf_buf"])
    print(f"  world_forward_r2={world_forward_r2:.4f}", flush=True)
    zg = ZGoalStreamAccumulator()
    levels: Dict[int, Dict[str, Dict[str, Any]]] = {}
    for p in HEARTBEAT_LEVELS:
        env_p = CausalGridWorldV2(seed=seed, size=ENV_SIZE, num_hazards=NUM_HAZARDS, num_resources=NUM_RESOURCES,
                                  hazard_harm=HAZARD_HARM, hazard_field_decay=HAZARD_FIELD_DECAY, contamination_spread=CONTAMINATION_SPREAD)
        levels[p] = {
            "AGENT": _run_eval_level(agent, env_p, p, eval_eps, eval_steps, WORLD_DIM, device, zg),
            "CONDITIONAL_ORACLE": _run_control_level(seed, p, eval_eps * CONTROL_EPISODES_MULT, eval_steps, "CONDITIONAL_ORACLE", seed, device),
            "HAZARD_BLIND_NULL": _run_control_level(seed, p, eval_eps * CONTROL_EPISODES_MULT, eval_steps, "HAZARD_BLIND_NULL", seed, device),
        }
    return {"seed": seed, "world_forward_r2": world_forward_r2, "total_harm_train": train_out["total_harm"],
            "levels": levels, "agent": agent, "zg": zg}


def _mean_sd(vals: List[Optional[float]]) -> Tuple[float, float]:
    arr = np.asarray([v for v in vals if v is not None], dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(arr.mean()), (float(arr.std(ddof=1)) if arr.size > 1 else 0.0)


def run_experiment(seeds: List[int], dry_run: bool) -> Dict[str, Any]:
    t0 = time.perf_counter()
    arm_rows: List[Dict[str, Any]] = []
    agents: List[REEAgent] = []
    slice_ = config_slice()
    slice_.update({"hazard_q_low": HAZARD_Q_LOW, "hazard_q_high": HAZARD_Q_HIGH, "control_episodes_mult": CONTROL_EPISODES_MULT,
                   "contamination_spread": CONTAMINATION_SPREAD, "margin_sd_mult": MARGIN_SD_MULT, "min_seeds_per_level": MIN_SEEDS_PER_LEVEL,
                   "schedule_tol": SCHEDULE_TOL, "eval_episodes_per_level": EVAL_EPISODES_PER_LEVEL,
                   "min_bin_coverage_ticks": MIN_BIN_COVERAGE_TICKS, "arms": ARMS,
                   "decision_tick_scoring": True})
    for seed in seeds:
        with arm_cell(seed, config_slice=slice_, script_path=Path(__file__), config_slice_declared=True) as cell:
            result = run_seed(seed, dry_run)
            row: Dict[str, Any] = {"seed": seed, "world_forward_r2": result["world_forward_r2"],
                                   "total_harm_train": result["total_harm_train"], "levels": {}}
            for p in HEARTBEAT_LEVELS:
                row["levels"][str(p)] = {arm: result["levels"][p][arm] for arm in ARMS}
            cell.stamp(row)
        arm_rows.append(row)
        agents.append(result["agent"])

    def _val(row, p, arm, key):
        return row["levels"][str(p)][arm].get(key)

    # ---- readiness over ALL seeds (non-negotiable (3)) ----------------------------------
    replica_worst = max(_val(r, p, a, "replica_err_max") for r in arm_rows for p in HEARTBEAT_LEVELS for a in ARMS)
    CTRL = ["CONDITIONAL_ORACLE", "HAZARD_BLIND_NULL"]
    # red-team F7: coverage is classified by ARM. Control-arm starvation is the instrument /
    # environment; AGENT-arm starvation is a fact about the agent (an avoider empties its own
    # HIGH bin; a wall-hugger empties its executable SAFE bin) and is classed separately.
    safe_cells = [(_val(r, p, a, "safe_executable_n"), f"seed{r['seed']}/P{p}/{a}") for r in arm_rows for p in HEARTBEAT_LEVELS for a in CTRL]
    high_cells = [(_val(r, p, a, "high_n"), f"seed{r['seed']}/P{p}/{a}") for r in arm_rows for p in HEARTBEAT_LEVELS for a in CTRL]
    safe_worst, safe_cell = min(safe_cells, key=lambda t: t[0])
    high_worst, high_cell = min(high_cells, key=lambda t: t[0])
    agent_safe_worst, agent_safe_cell = min([(_val(r, p, "AGENT", "safe_executable_n"), f"seed{r['seed']}/P{p}") for r in arm_rows for p in HEARTBEAT_LEVELS], key=lambda t: t[0])
    agent_high_worst, agent_high_cell = min([(_val(r, p, "AGENT", "high_n"), f"seed{r['seed']}/P{p}") for r in arm_rows for p in HEARTBEAT_LEVELS], key=lambda t: t[0])
    cov_worst, cov_cell = min([(safe_worst, safe_cell), (high_worst, high_cell)], key=lambda t: t[0])
    sched_cells = [(_val(r, p, "AGENT", "decision_ticks_over_expected") or 0.0, f"seed{r['seed']}/P{p}") for r in arm_rows for p in HEARTBEAT_LEVELS]
    sched_worst_hi, sched_cell_hi = max(sched_cells, key=lambda t: t[0])
    sched_worst_lo, sched_cell_lo = min(sched_cells, key=lambda t: t[0])
    sep_cells = []
    for r in arm_rows:
        for p in HEARTBEAT_LEVELS:
            o, n = _val(r, p, "CONDITIONAL_ORACLE", "sensitivity"), _val(r, p, "HAZARD_BLIND_NULL", "sensitivity")
            sep_cells.append(((o - n) if (o is not None and n is not None) else float("nan"), f"seed{r['seed']}/P{p}"))
    # pass-2 N2: a nan here is a COVERAGE failure (a control bin starved), not a separation failure;
    # it is classified by the coverage gates, so the separation statistic is the worst FINITE cell
    _finite_seps = [t for t in sep_cells if t[0] == t[0]]
    sep_worst, sep_cell = (min(_finite_seps, key=lambda t: t[0]) if _finite_seps else (float("nan"), "no covered cell"))
    null_level_means = {p: _mean_sd([_val(r, p, "HAZARD_BLIND_NULL", "sensitivity") for r in arm_rows])[0] for p in HEARTBEAT_LEVELS}
    null_level_sds = {p: _mean_sd([_val(r, p, "HAZARD_BLIND_NULL", "sensitivity") for r in arm_rows])[1] for p in HEARTBEAT_LEVELS}
    null_range = max(null_level_means.values()) - min(null_level_means.values())
    # noise-aware: the null's mean at each level is itself a noisy estimate over seeds, so the
    # flatness ceiling is max(NULL_FLATNESS_CEIL, 2 x the pooled cross-seed sd of the null)
    _null_sd_pooled = float(np.nanmean([v for v in null_level_sds.values() if v == v])) if any(v == v for v in null_level_sds.values()) else 0.0
    null_flat_ceiling = max(NULL_FLATNESS_CEIL, 2.0 * _null_sd_pooled)
    oracle_level_means = {p: _mean_sd([_val(r, p, "CONDITIONAL_ORACLE", "sensitivity") for r in arm_rows])[0] for p in HEARTBEAT_LEVELS}
    sep_level_means = {p: oracle_level_means[p] - null_level_means[p] for p in HEARTBEAT_LEVELS}
    clamp_cells = [(_val(r, p, "AGENT", "safe_bin_wall_clamp_fraction"), f"seed{r['seed']}/P{p}") for r in arm_rows for p in HEARTBEAT_LEVELS]
    clamp_vals = [(c if c is not None else 1.0, k) for c, k in clamp_cells]
    clamp_worst, clamp_cell = max(clamp_vals, key=lambda t: t[0])
    # derived P0 bar at P_train, mean over seeds
    base_margins = []
    for r in arm_rows:
        a = _val(r, TRAIN_HEARTBEAT_STEPS, "AGENT", "sensitivity")
        o = _val(r, TRAIN_HEARTBEAT_STEPS, "CONDITIONAL_ORACLE", "sensitivity")
        n = _val(r, TRAIN_HEARTBEAT_STEPS, "HAZARD_BLIND_NULL", "sensitivity")
        base_margins.append((a - (n + BAR_FRACTION * (o - n))) if None not in (a, o, n) else None)
    base_margin_mean, base_margin_sd = _mean_sd(base_margins)

    replica_ok = bool(replica_worst <= REPLICA_TOL)
    safe_cov_ok = bool(safe_worst >= MIN_BIN_COVERAGE_TICKS)   # control arms: H-env-band-absent leg
    high_cov_ok = bool(high_worst >= MIN_BIN_COVERAGE_TICKS)   # control arms: instrument
    agent_safe_ok = bool(agent_safe_worst >= MIN_BIN_COVERAGE_TICKS)
    agent_high_ok = bool(agent_high_worst >= MIN_BIN_COVERAGE_TICKS)
    coverage_ok = bool(safe_cov_ok and high_cov_ok and agent_safe_ok and agent_high_ok)
    # red-team F1: the forced period must BE the agent's decision schedule (no phase resets):
    # realised decision ticks / (steps / P) within [1 - SCHEDULE_TOL, 1 + SCHEDULE_TOL] in every cell
    schedule_ok = bool(sched_worst_hi <= 1.0 + SCHEDULE_TOL and sched_worst_lo >= 1.0 - SCHEDULE_TOL)
    separation_ok = bool(sep_worst == sep_worst and sep_worst >= ORACLE_NULL_SEPARATION_FLOOR)
    null_flat_ok = bool(null_range == null_range and null_range <= null_flat_ceiling)
    clamp_ok = bool(clamp_worst <= WALL_CLAMP_CEIL)   # RECORDED diagnostic (not a gate): FA is scored over executable SAFE ticks
    baseline_ok = bool(base_margin_mean == base_margin_mean and base_margin_mean >= 0.0)
    hr = dv_headroom_check("dv_headroom_interior_margin",
                           dv_name="raw (oracle - null) decision-tick separation per level -- the instrument's dynamic range the normalised C1 margin is scaled by",
                           criterion_threshold=MARGIN_ABS_FLOOR, control_values=[sep_level_means[p] for p in HEARTBEAT_LEVELS],
                           statistic="floor_headroom", dv_bounds=(0.0, 2.0), margin=HEADROOM_MARGIN,
                           measured_cells=[f"P{p}" for p in HEARTBEAT_LEVELS])
    hr["met"] = bool(hr["measured"] == hr["measured"] and hr["measured"] > hr["threshold"])
    hr["offending_cell"] = "worst level"
    # red-team F5: null flatness is tautological in expectation (random decisions give hit = FA)
    # and fails by noise; it is RECORDED, not a gate. F6: the oracle/null separation and the
    # headroom entry certify the oracle's construction; kept as construction checks.
    instrument_ok = bool(replica_ok and high_cov_ok and separation_ok and schedule_ok and hr["met"])
    ready = bool(instrument_ok and safe_cov_ok and agent_high_ok and agent_safe_ok and baseline_ok)
    if ready:
        failure_class = "none"
    elif not schedule_ok:
        failure_class = "instrument"        # the manipulation did not set the decision schedule
    elif not safe_cov_ok:
        failure_class = "env_band_absent"   # the executable SAFE band is what the environment must supply (control arms starve) -- classified BEFORE instrument (pass-2 N2)
    elif not instrument_ok:
        failure_class = "instrument"
    elif not agent_high_ok:
        failure_class = "agent_starves_high_bin"   # an avoider empties its own HIGH bin: a finding about the agent, not the instrument
    elif not agent_safe_ok:
        failure_class = "agent_starves_safe_bin"   # a wall-hugger / corner-sitter: a finding about the agent
    else:
        failure_class = "avoidance_untrained"

    preconditions = [
        {"name": "hazard_field_replica_matches_obs", "kind": "readiness",
         "description": "max |hazard_field_view[12] - env.hazard_field[agent cell]/max| over every step of every arm -- the quantile bins are taken from env.hazard_field, so it must be the quantity the agent observes",
         "control": "env's own field, normalised exactly as the view is", "measured": float(replica_worst), "threshold": REPLICA_TOL, "direction": "upper", "met": replica_ok},
        {"name": "high_bin_decision_ticks_covered", "kind": "readiness",
         "description": "min over (seed, level) of the CONTROL arms' HIGH-bin DECISION ticks (non-negotiables (3)/(4)); the AGENT arm is gated separately (red-team F7)",
         "control": "quantile bins on the env's own field distribution", "measured": float(high_worst), "threshold": float(MIN_BIN_COVERAGE_TICKS), "comparator": ">=", "direction": "lower", "offending_cell": high_cell, "met": high_cov_ok},
        {"name": "safe_bin_executable_decision_ticks_covered", "kind": "readiness",
         "description": "min over (seed, level) of the CONTROL arms' SAFE-bin DECISION ticks whose designated avoidant action is EXECUTABLE (not a wall no-op) -- the H-env-band-absent leg; the false-alarm rate is scored over these ticks only; the AGENT arm is gated separately (red-team F7)",
         "control": "quantile SAFE bin, wall-clamped ticks excluded", "measured": float(safe_worst), "threshold": float(MIN_BIN_COVERAGE_TICKS), "comparator": ">=", "direction": "lower", "offending_cell": safe_cell, "met": safe_cov_ok},
        {"name": "oracle_null_separation", "kind": "readiness",
         "description": "worst (seed, level) of CONDITIONAL_ORACLE minus HAZARD_BLIND_NULL decision-tick sensitivity -- the instrument can register threat-gated avoidance (non-negotiable (2))",
         "control": "measured positive and negative control arms on the same env, seed and decision schedule", "measured": float(sep_worst), "threshold": ORACLE_NULL_SEPARATION_FLOOR, "comparator": ">=", "direction": "lower", "offending_cell": sep_cell, "met": separation_ok},
        {"name": "decision_schedule_matches_P", "kind": "readiness",
         "description": "AGENT realised decision ticks / (steps / P), every (seed, level): must sit within 1 +/- SCHEDULE_TOL -- the forced period IS the agent's schedule only if no MECH-091 phase reset fires (red-team F1, BLOCKING in the first draft)",
         "control": "the forced-period arithmetic", "measured": float(sched_worst_hi), "threshold": 1.0 + SCHEDULE_TOL, "direction": "upper",
         "offending_cell": sched_cell_hi, "met": bool(schedule_ok), "measured_low": float(sched_worst_lo), "offending_cell_low": sched_cell_lo},
        {"name": "agent_high_bin_decision_ticks_covered", "kind": "readiness",
         "description": "min over (seed, level) of the AGENT's HIGH-bin decision ticks -- an avoider that never enters HIGH is classed agent_starves_high_bin, not instrument",
         "control": "agent arm only", "measured": float(agent_high_worst), "threshold": float(MIN_BIN_COVERAGE_TICKS), "comparator": ">=", "direction": "lower", "offending_cell": agent_high_cell, "met": agent_high_ok},
        {"name": "agent_safe_bin_executable_decision_ticks_covered", "kind": "readiness",
         "description": "min over (seed, level) of the AGENT's executable SAFE-bin decision ticks -- classed agent_starves_safe_bin when only the agent starves",
         "control": "agent arm only", "measured": float(agent_safe_worst), "threshold": float(MIN_BIN_COVERAGE_TICKS), "comparator": ">=", "direction": "lower", "offending_cell": agent_safe_cell, "met": agent_safe_ok},
        {"name": "null_sensitivity_flat_in_P_RECORDED", "kind": "diagnostic",
         "description": "range across the 5 levels of the null's mean decision-tick sensitivity -- decision-tick scoring must remove the hold-period confound (999 red-team F2: the all-step null was monotone in P)",
         "control": "hazard-blind null across P; RECORDED ONLY (red-team F5: tautologically flat in expectation, fails by noise)", "measured": float(null_range), "threshold": float(null_flat_ceiling), "direction": "upper", "met": True, "raw_met": null_flat_ok, "gating": False},
        {"name": "baseline_avoidance_clears_derived_bar", "kind": "readiness",
         "description": "mean over seeds of AGENT sensitivity at P_train minus [null + BAR_FRACTION*(oracle - null)] at P_train, decision ticks -- the DERIVED positive control replacing 999's hand-picked +0.05 (H-avoidance-untrained if unmet with a valid instrument)",
         "control": "bar derived from the measured oracle/null separation at the same P", "measured": float(base_margin_mean), "measured_sd": float(base_margin_sd), "threshold": 0.0, "comparator": ">=", "direction": "lower", "met": baseline_ok},
        hr,
    ]

    # ---- normalised curve and C1 --------------------------------------------------------
    per_level_vals: Dict[int, List[float]] = {}
    per_level_raw: Dict[int, List[float]] = {}
    for p in HEARTBEAT_LEVELS:
        vals, raw = [], []
        for r in arm_rows:
            a, o, n = (_val(r, p, "AGENT", "sensitivity"), _val(r, p, "CONDITIONAL_ORACLE", "sensitivity"), _val(r, p, "HAZARD_BLIND_NULL", "sensitivity"))
            if None in (a, o, n) or (o - n) <= 0:
                continue
            vals.append((a - n) / (o - n)); raw.append(a)
        per_level_vals[p] = vals; per_level_raw[p] = raw
    per_level_mean = {p: _mean_sd(per_level_vals[p])[0] for p in HEARTBEAT_LEVELS}
    per_level_sd = {p: _mean_sd(per_level_vals[p])[1] for p in HEARTBEAT_LEVELS}
    per_level_raw_mean = {p: _mean_sd(per_level_raw[p])[0] for p in HEARTBEAT_LEVELS}
    finite = all(v == v for v in per_level_mean.values()) and all(len(per_level_vals[p]) >= MIN_SEEDS_PER_LEVEL for p in HEARTBEAT_LEVELS)
    sensitivity_range = (max(per_level_mean.values()) - min(per_level_mean.values())) if finite else float("nan")
    dv_moved = bool(finite and sensitivity_range >= DV_MOVEMENT_FLOOR_NORM)
    degeneracy = check_degeneracy({"normalised_sensitivity_range": {"values": [per_level_mean[p] for p in HEARTBEAT_LEVELS] if finite else [], "floor": None}})
    non_degenerate = bool(ready and degeneracy["non_degenerate"] and dv_moved)
    if finite:
        best_level = max(HEARTBEAT_LEVELS, key=lambda p: per_level_mean[p])
        extremes = [HEARTBEAT_LEVELS[0], HEARTBEAT_LEVELS[-1]]
        is_interior = best_level not in extremes
        pooled_sd = float(np.mean([per_level_sd[best_level]] + [per_level_sd[e] for e in extremes]))
        required_margin = max(MARGIN_ABS_FLOOR_NORM, MARGIN_SD_MULT * pooled_sd)
        margins = {str(e): per_level_mean[best_level] - per_level_mean[e] for e in extremes}
        beats_both = all(m >= required_margin for m in margins.values())
    else:
        best_level, is_interior, required_margin, margins, beats_both = None, False, float("nan"), {}, False
    c1_pass = bool(non_degenerate and is_interior and beats_both)

    # red-team F4: under a FLAT truth an extreme level is the argmax 40 percent of the time by
    # noise alone, so "extreme best" is NOT evidence against an interior optimum. `weakens`
    # now requires the MIRRORED margin: the best (extreme) level beats the best INTERIOR level
    # by >= the same noise-aware requirement. Otherwise the curve is indeterminate -> inconclusive.
    if finite and not is_interior:
        interior_levels = [p for p in HEARTBEAT_LEVELS if p not in extremes]
        best_interior = max(interior_levels, key=lambda p: per_level_mean[p])
        extreme_margin = per_level_mean[best_level] - per_level_mean[best_interior]
        weakens_clause = bool(extreme_margin >= required_margin)
    else:
        extreme_margin, weakens_clause = float("nan"), False
    if not ready:
        label = {"instrument": "substrate_not_ready_requeue", "avoidance_untrained": "avoidance_not_trained_under_warmup",
                 "env_band_absent": "safe_band_wall_clamped_env_leg", "agent_starves_high_bin": "agent_avoids_high_bin_unscorable",
                 "agent_starves_safe_bin": "agent_wall_hugging_unscorable"}.get(failure_class, "substrate_not_ready_requeue")
        direction, verdict = "non_contributory", "FAIL"
    elif not non_degenerate:
        label, direction, verdict = "normalised_sensitivity_flat_non_degenerate", "non_contributory", "FAIL"
    elif c1_pass:
        label, direction, verdict = "interior_optimum_confirmed", "supports", "PASS"
    elif weakens_clause:
        label, direction, verdict = "extreme_level_best_by_margin_monotone", "weakens", "FAIL"
    else:
        label, direction, verdict = "shape_indeterminate_no_margin_either_way", "inconclusive", "FAIL"

    registry = {
        "question": REGISTRY_QUESTION,
        "hypotheses": {
            "H-readout-invalid": {"adjudicated": True, "state": ("eliminated" if instrument_ok else "supported"),
                                  "basis": "oracle-null separation, null flatness in P, coverage and replica gates on the decision-tick readout"},
            "H-avoidance-untrained": {"adjudicated": bool(instrument_ok), "state": ("unadjudicated" if not instrument_ok else ("eliminated" if baseline_ok else "supported")),
                                      "basis": "AGENT at P_train vs the derived bar, on a validated instrument"},
            "H-env-band-absent": {"adjudicated": bool(high_cov_ok), "state": ("eliminated" if safe_cov_ok else ("supported" if high_cov_ok else "unadjudicated")),
                                  "basis": "the quantile SAFE bin supplies >= MIN_BIN_COVERAGE_TICKS EXECUTABLE decision ticks per cell (wall-clamped ticks recorded separately)"},
        },
    }

    manifest: Dict[str, Any] = {
        "run_id": f"{EXPERIMENT_TYPE}_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}_v3",
        "experiment_type": EXPERIMENT_TYPE, "architecture_epoch": ARCHITECTURE_EPOCH,
        "claim_ids": CLAIM_IDS, "claim_ids_tested": CLAIM_IDS, "experiment_purpose": EXPERIMENT_PURPOSE,
        "queue_id": QUEUE_ID, "supersedes": SUPERSEDES,
        "outcome": verdict, "evidence_class": "experimental", "evidence_direction": direction,
        "evidence_direction_per_claim": {"MECH-161": direction},
        "arm_results": arm_rows,
        "heartbeat_levels": HEARTBEAT_LEVELS, "train_heartbeat_steps": TRAIN_HEARTBEAT_STEPS,
        "readiness_failure_class": failure_class,
        "registry_adjudication": registry,
        "non_degenerate": non_degenerate,
        "degeneracy_reason": ("" if non_degenerate else (
            f"readiness {failure_class}" if not ready else
            f"normalised sensitivity range {sensitivity_range:.4f} below {DV_MOVEMENT_FLOOR_NORM}; {degeneracy['degeneracy_reason']}")),
        "metrics": {
            "per_level_mean_normalised_sensitivity": {str(p): per_level_mean[p] for p in HEARTBEAT_LEVELS},
            "per_level_sd_normalised_sensitivity": {str(p): per_level_sd[p] for p in HEARTBEAT_LEVELS},
            "per_level_mean_raw_agent_sensitivity": {str(p): per_level_raw_mean[p] for p in HEARTBEAT_LEVELS},
            "per_level_mean_null_sensitivity": {str(p): null_level_means[p] for p in HEARTBEAT_LEVELS},
            "per_level_mean_oracle_sensitivity": {str(p): oracle_level_means[p] for p in HEARTBEAT_LEVELS},
            "per_level_oracle_null_separation": {str(p): sep_level_means[p] for p in HEARTBEAT_LEVELS},
            "per_level_mean_allstep_agent_sensitivity_DIAGNOSTIC": {str(p): _mean_sd([_val(r, p, "AGENT", "allstep_sensitivity") for r in arm_rows])[0] for p in HEARTBEAT_LEVELS},
            "per_level_mean_allstep_null_sensitivity_DIAGNOSTIC": {str(p): _mean_sd([_val(r, p, "HAZARD_BLIND_NULL", "allstep_sensitivity") for r in arm_rows])[0] for p in HEARTBEAT_LEVELS},
            "best_level": best_level, "is_interior": is_interior, "required_margin_normalised": required_margin,
            "margins_vs_extremes_normalised": margins, "normalised_sensitivity_range": sensitivity_range,
            "extreme_over_best_interior_margin_normalised": extreme_margin, "weakens_clause": weakens_clause,
            "agent_decision_ticks_over_expected_per_seed_level": {str(r["seed"]): {str(p): _val(r, p, "AGENT", "decision_ticks_over_expected") for p in HEARTBEAT_LEVELS} for r in arm_rows},
            "baseline_derived_bar_margin_mean": base_margin_mean, "baseline_derived_bar_margin_per_seed": base_margins,
            "null_range_across_levels": null_range, "null_flatness_ceiling_applied": null_flat_ceiling,
            "per_level_sd_null_sensitivity": {str(p): null_level_sds[p] for p in HEARTBEAT_LEVELS},
            "oracle_null_separation_worst": sep_worst,
            "safe_bin_wall_clamp_fraction_worst": clamp_worst, "safe_bin_wall_clamp_worst_cell": clamp_cell,
            "safe_bin_wall_clamp_within_ceiling": clamp_ok,
            "safe_bin_executable_ticks_worst": safe_worst, "high_bin_ticks_worst": high_worst,
            "world_forward_r2_per_seed": {str(r["seed"]): r["world_forward_r2"] for r in arm_rows},
            "committed_fraction_per_seed_level": {str(r["seed"]): {str(p): _val(r, p, "AGENT", "committed_fraction") for p in HEARTBEAT_LEVELS} for r in arm_rows},
        },
        "interpretation": {"label": label, "preconditions": preconditions,
                           "criteria_non_degenerate": {"C1_interior_max_margin_normalised": bool(non_degenerate)},
                           "gate_green": ready},
        "criteria": [{"name": "C1_interior_max_margin_normalised", "load_bearing": True, "passed": c1_pass,
                      "description": ("On s_norm(P) = (agent - null)/(oracle - null) at decision ticks, mean over seeds: the level with "
                                      "maximum mean s_norm is INTERIOR and beats both extreme levels by >= max(MARGIN_ABS_FLOOR_NORM, "
                                      "MARGIN_SD_MULT * pooled cross-seed SD).")}],
        "combination_rule": ("PASS = every readiness gate green AND DV moved (normalised range >= floor) AND C1. A red gate routes by "
                             "readiness_failure_class (instrument -> substrate_not_ready_requeue; avoidance_untrained -> H-avoidance-"
                             "untrained supported, MECH-161 untested; env_band_absent -> H-env-band-absent supported). MECH-161 direction "
                             "is set only when ready and non-degenerate."),
        "claim_scope": {"what_this_run_measures": ("The SHAPE premise only: an operating-point sweep of the forced E3 period on a frozen trained "
                                                   "agent, with MECH-093's regulator output overwritten each tick. MECH-161's REGULATOR assertion "
                                                   "is NOT tested (autopsy Section 5); MECH-026's regulator is unbuilt.")},
        "positive_control_margin_999_legacy": POSITIVE_CONTROL_MARGIN,
        "pre_registered_thresholds": {"hazard_q_low": HAZARD_Q_LOW, "hazard_q_high": HAZARD_Q_HIGH, "min_bin_coverage_ticks": MIN_BIN_COVERAGE_TICKS,
                                      "oracle_null_separation_floor": ORACLE_NULL_SEPARATION_FLOOR, "bar_fraction": BAR_FRACTION,
                                      "null_flatness_ceil": NULL_FLATNESS_CEIL, "margin_abs_floor_norm": MARGIN_ABS_FLOOR_NORM,
                                      "dv_movement_floor_norm": DV_MOVEMENT_FLOOR_NORM, "wall_clamp_ceil": WALL_CLAMP_CEIL,
                                      "replica_tol": REPLICA_TOL, "margin_sd_mult": MARGIN_SD_MULT, "margin_abs_floor_raw": MARGIN_ABS_FLOOR},
        "ethics_preflight": {"involves_negative_valence": False, "involves_suffering_like_state": False, "involves_self_model": False,
                             "involves_inescapability_or_helplessness": False, "involves_offline_replay_over_harm": False,
                             "involves_social_mind_or_language": False, "involves_human_data_or_clinical_context": False, "decision": "allow"},
    }
    print(f"verdict: {verdict}  label={label}  class={failure_class}  best_level={best_level} is_interior={is_interior} "
          f"required_margin={required_margin:.4f} margins={margins}", flush=True)
    out_path = write_flat_manifest(manifest, dry_run=dry_run, config=slice_, seeds=seeds,
                                   script_path=Path(__file__), started_at=t0, agent=agents)
    if dry_run:
        print("[smoke] per_level normalised: " + ", ".join(f"P={p}:{per_level_mean[p]:.4f}" for p in HEARTBEAT_LEVELS), flush=True)
        print("[smoke] oracle/null per level: " + ", ".join(f"P={p}:{oracle_level_means[p]:.3f}/{null_level_means[p]:.3f}" for p in HEARTBEAT_LEVELS), flush=True)
        print(f"[smoke] replica_err={replica_worst:.2e} coverage_worst={cov_worst} ({cov_cell}) clamp_worst={clamp_worst:.3f} baseline_margin={base_margin_mean:.4f}", flush=True)
    return {"outcome": verdict, "manifest": manifest, "out_path": out_path}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--seeds", type=str, default=None)
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else (SEEDS[:2] if args.dry_run else SEEDS)
    result = run_experiment(seeds, args.dry_run)
    out_path = result["out_path"]
    print(f"\nResult written to: {out_path}", flush=True)
    print(f"Status: {result['outcome']}", flush=True)
    for p in result["manifest"]["interpretation"]["preconditions"]:
        print(f"  precondition {p['name']:<40} met={p.get('met')} measured={p.get('measured')}", flush=True)
    _reg = {k: v["state"] for k, v in result["manifest"]["registry_adjudication"]["hypotheses"].items()}
    print(f"  registry: {_reg}", flush=True)
    raw = str(result["outcome"]).upper()
    return (raw if raw in ("PASS", "FAIL") else "FAIL"), out_path, args.dry_run


if __name__ == "__main__":
    _outcome_raw, _out_path, _dry = main()
    emit_outcome(outcome=_outcome_raw, manifest_path=_out_path, queue_id=QUEUE_ID, dry_run=_dry)
