"""
V3-EXQ-981 -- MECH-027 (hypervigilance signature): control-plane channel
forcing, targeted probe, with within-run reversion check.

ID NOTE: authored as V3-EXQ-979; renumbered to V3-EXQ-981 at queue time
(2026-09-02) because V3-EXQ-979 was independently reserved by another
session (metaworker-chip-proposal-exp-0853-exq-979, TASK_CLAIMS claimed_at
2026-09-02T18:41:22Z) for an unrelated MECH-157 script -- this draft was
never pushed before that reservation landed, so the collision was
invisible to both sessions' Step 2 origin/main checks.

SLEEP DRIVER: manual-multi (force_cycle() every SLEEP_EVERY_N_EPISODES during
warmup; agent.force_sleep_cycle_at_eval_boundary() at every eval episode
boundary in all three blocks -- MECH-027 Build 2 convention, 2026-09-02).

RED-TEAM (fable, 2026-09-02): CONTESTED, 1 BLOCKING finding (F1, fixed).
See custom_information.red_team_disposition_note in the manifest output for
the full per-finding disposition (F1/F3/F6/F8 fixed; F2 verified and
dismissed with a source citation; F4/F5/F7 acknowledged, not mitigated, with
rationale).

Claims: MECH-027 ("Pathological modes reflect mis-tuned control-plane
regimes")

EXPERIMENT_PURPOSE = "evidence" (direct falsifier test of MECH-027, scoped to
one of its five named pathological labels -- see SCOPE below).

WHY THIS SCOPE (dispatch_mode: targeted_probe, per source proposal
EXP-0761/EVB-1396). A prior substrate-readiness investigation established:
  - gain/precision (ARC-016, E3TrajectorySelector.current_precision /
    _running_variance) -- IMPLEMENTED, WIRED, non-degenerate (confirmed by
    V3-EXQ-876/876a's own precision separation, 4-5 orders of magnitude).
  - prediction horizon (HippocampalConfig.horizon structural default, and the
    SD-MECH267-HORIZON-DEPTH runtime CEM elite-scoring-window mechanism,
    mode_horizon_scale + operating_mode) -- IMPLEMENTED, WIRED, non-degenerate.
  - replay/hippocampal-injection suppression (mech285 sampler,
    use_mech285_sampler / mech285_draws_per_cycle) -- IMPLEMENTED, WIRED,
    confirmed reachable (V3-EXQ-909: draws_per_cycle=50 on every seed once
    use_anchor_sets + use_mech285_sampler + use_mech272_routing are all set).
  - "learning eligibility" (needed for the MANIA signature) has NO confirmed
    substrate hook, and "hippocampal gating" (dissociation/psychosis-like
    labels) was out of scope for this investigation.
So THIS experiment tests ONLY the HYPERVIGILANCE signature, which the claim's
own text defines as needing exactly the three channels above: "elevated gain
+ shortened horizon + suppressed replay ... reproduces ... excessive
false-alarm / over-reactive responding ... AND the behavior reverts when the
channel is returned to its normal range." This is a scoped single-signature
probe (claims.yaml MECH-027 what_would_answer), not a forced 5x5 sweep across
all five pathological labels -- the other four are out of scope here.

NON-DEGENERACY PRECONDITION (claim's own text, MANDATORY): each of the three
channels must show independently measurable AND independently perturbable,
NON-DEGENERATE variance across the pathological/non-pathological range under
test. A run where any channel is flat, hardcoded, or torn down at read time
self-routes substrate_not_ready_requeue, never a verdict. MECH-025 (sharing
MECH-027's ARC-016/ARC-005 dependency) had three consecutive false-negative
instrument defects of exactly this kind (hardcoded E3 precision; missing
self-attribution channel; cached/torn-down commitment-state field; frozen
running_variance during eval) before V3-EXQ-876 finally got a fair test
(failure_autopsy_mech025-cluster-876-671b_2026-08-03). This script keeps
running_variance LIVE during eval (agent.update_residue() -> E3.post_action_
update() -> update_running_variance() every tick, exactly 876/876a's fix) and
additionally forces the channel via a per-tick multiplicative scale (see
"HOW EACH CHANNEL IS FORCED" below) so it stays perturbed AND non-degenerate
(real per-tick variance) rather than a frozen hardcoded constant.

DESIGN -- ONE agent per seed, THREE sequential eval regimes on the SAME
trained substrate (not three separately-trained agents). This is the design
the claim's own framing calls for: "mis-tuned regimes of the SAME
control-plane machinery," not a different learned substrate per condition.
  1. WARMUP (shared): standard E1/E2/E3 training under baseline channel
     settings (with periodic forced sleep cycles, mech285 replay engaged at
     the baseline draw count) -- builds ONE functional agent per seed.
  2. EVAL_BASELINE block: normal-range gain/precision, horizon, replay.
  3. EVAL_HYPERVIGILANT block (SAME agent, channels forced): elevated
     precision + shortened horizon + suppressed replay, simultaneously.
  4. EVAL_REVERSION block (SAME agent, channels reverted to baseline):
     confirms the claim's own required reversion half of the confirming
     signature -- "the behavior reverts when the channel is returned to its
     normal range."

HOW EACH CHANNEL IS FORCED (all three are RUNTIME-revertible on the live
agent -- verified against current source before writing this script, none
requires rebuilding the agent's network):

  (a) Gain/precision -- E3TrajectorySelector._running_variance (plain
      instance attribute; current_precision = 1/(running_variance+1e-6),
      ree_core/predictors/e3_selector.py:771-773). Each EVAL_HYPERVIGILANT
      tick, AFTER agent.update_residue() has driven its normal live EMA
      update (post_action_update -> update_running_variance, agent.py:10489,
      e3_selector.py:4151), the driver additionally multiplies
      _running_variance by HV_PRECISION_SCALE (<1, shrinks variance ->
      elevates precision). This keeps the channel LIVE (still moves with
      genuine per-tick prediction error) while forcing it out of its normal
      operating range -- the "directly perturb precision_init / inject a
      scaling factor" option the design brief names. EVAL_BASELINE and
      EVAL_REVERSION apply scale=1.0 (no forcing; natural ARC-016 dynamics).

      GRADED DOWNSTREAM CONSUMER (MECH-027 Build 1, 2026-09-02,
      chip-20260902-mech027-precision-replay-eval-substrate): a red-team
      review of an earlier draft of this script found channel (a) as
      described above BLOCKED -- the only pre-existing consumer of
      _running_variance was the binary ARC-016 commit gate
      (committed = commit_variance < effective_threshold), which SATURATES
      in a trained substrate (running_variance empirically ~125x below
      threshold), so once committed, further shrinking variance had NO
      further observable effect anywhere downstream: the (a) forcing above
      could move current_precision as a NUMBER while leaving E3's actual
      selection behavior untouched. FIX: this driver additionally sets
      E3Config.use_precision_scaled_commit_temperature=True (build_config()
      below), which gives current_precision/running_variance a GRADED
      consumer: the committed argmin becomes
      multinomial(softmax(-q/T_eff)) with T_eff = base_temperature +
      PRECISION_SCALED_COMMIT_ENTROPY_ALPHA * (1 - precision_margin_norm),
      where precision_margin_norm = clamp(1 - commit_variance/
      effective_threshold, 0, 1). A maximally-confident tick (precision
      forced up, precision_margin_norm -> 1) commits COLD (T_eff -> base,
      hard/rigid argmin -- the hypervigilance direction); a barely-committed
      tick commits HOTTER (softer, more exploratory). q is restricted to an
      F-eligibility envelope (PRECISION_SCALED_COMMIT_HARM_FLOOR *
      raw_score_range of the best raw score) so a hot commit-T can never
      softmax-promote a clearly-harmful candidate. This is
      READ, not additionally forced, by this driver -- per-tick
      precision_margin_norm and precision_scaled_commit_temperature_eff are
      captured from agent.e3.last_score_diagnostics at every real E3
      selection (see _agent_tick) and pooled into the new readiness checks
      below, so a run where the graded consumer never actually engaged
      (e.g. the standalone committed branch not reached, or baseline
      already precision-saturated so HV cannot push it further) self-routes
      substrate_not_ready_requeue rather than a false hypervigilance
      verdict.

  (b) Prediction horizon -- SD-MECH267-HORIZON-DEPTH
      (HippocampalModule._compute_mode_horizon_scale /
      config.hippocampal.mode_horizon_scale, module.py:1836-1860,
      1990-2016). CORRECTED FROM THE ORIGINAL BRIEF: agent.py's real
      _e3_tick() call site (agent.py:5951-6153) does NOT pass operating_mode
      to hippocampal.propose_trajectories() at all (verified by direct grep
      of the call site) -- so agent.generate_trajectories() cannot be used
      for this channel. This driver instead calls
      agent.hippocampal.propose_trajectories(..., operating_mode=...)
      DIRECTLY (see _agent_tick below), mirroring _e3_tick's essential shape
      (candidate caching between e3_ticks, theta-independent since this
      probe does not use goal/ghost-probe machinery) but with a synthetic
      operating_mode dict this driver controls per-tick. mode_horizon_scale
      scales the CEM ELITE-SELECTION SCORING WINDOW, NOT the physical
      rollout length (config.horizon / terrain_prior's output width stays a
      fixed structural network dimension, per the module's own docstring at
      1904-1910) -- so this is genuinely runtime-revertible on the live
      agent with no network rebuild, unlike HippocampalConfig.horizon
      itself.

  (c) Replay/hippocampal-injection suppression -- mech285 sampler
      (agent.sleep_loop.draws_per_cycle, ree_core/sleep/phase_manager.py).
      draws_per_cycle is a plain int on the (already-constructed)
      SleepLoopManager, not a structural dimension, so it is directly
      mutable at runtime once the sampler object exists. The three
      construction-time gates that make agent.sleep_replay_sampler
      non-None at all (use_anchor_sets, use_mech285_sampler,
      use_mech272_routing) are set on EVERY agent so the sampler is built
      once; EVAL_HYPERVIGILANT then sets
      agent.sleep_loop.draws_per_cycle = 0 (full suppression) and
      EVAL_REVERSION restores it to MECH285_BASELINE_DRAWS. This exact
      three-flag recipe (use_anchor_sets + use_mech285_sampler +
      use_mech272_routing) is the one V3-EXQ-909 empirically confirmed
      reaches a live, non-zero draws_per_cycle (measured=50.0 on every
      seed) -- see that script's module docstring for the full trace of why
      each of the three gates is independently required.

      SLEEP CYCLES ACTUALLY FIRE DURING EVAL (MECH-027 Build 2, 2026-09-02,
      same chip as Build 1 above). A prior draft of this script set
      draws_per_cycle per block but never caused a sleep cycle to fire
      during eval at all (sleep_loop_episodes_K is set to 1_000_000 so the
      automatic K-episode cadence never crosses during a 25-episode eval
      block, and every driver surveyed before Build 2 called force_cycle()
      only during warmup) -- so the (c) forcing above was STRUCTURALLY
      INERT: draws_per_cycle=0 has no observable effect if no sleep cycle
      ever consults it. FIX: this driver calls
      agent.force_sleep_cycle_at_eval_boundary() once at the end of EVERY
      eval episode, in all three blocks (see _run_eval_block) -- the
      formalized V3-EXQ-909 flush-then-force_cycle sequence, which does NOT
      call agent.reset() (residue / goal state / latent state / episode
      counters untouched) and is reachable at an eval boundary without
      corrupting the broader wake/sleep state machine. Each call's returned
      metrics dict carries mech285_n_draws -- the MEASURED number of replay
      draws actually performed that cycle -- which this driver records
      per-block (not just the configured draws_per_cycle) so the readiness
      gate below confirms real engagement (mech285_n_draws==0 under HV,
      >=1 under BASELINE/REVERSION), not merely that the knob was set.

DV -- false-alarm / over-reactive responding (claim's own phrase for the
hypervigilance signature). Operationalised as the rate of AVOIDANT action
(the single grid move whose (dx,dy) has the largest positive dot product
with the away-from-nearest-hazard vector, computed live each step from
env._action_map + env.grid's actual hazard cell(s) -- robust to any
action-index convention) taken while the CURRENT hazard signal
(hazard_field_view's centre cell, index 12 of the 5x5 proximity window,
world_state channel, ARC-024 proxy-gradient) is in the AMBIGUOUS band: a low
BUT NONZERO reading, not a genuine imminent-contact reading. This is exactly
"excessive/over-reactive responding to non-imminent signal," the claim's own
operationalisation target, as distinct from a HIGH-band reading (where
avoidant response is normatively appropriate, not a false alarm) or a SAFE
reading (near-zero; no signal to react to at all).

THRESHOLD CALIBRATION (pre-registered BEFORE the real run, not derived from
this run's own statistics -- see PRE-REGISTERED THRESHOLDS below). env is
CausalGridWorldV2(size=10, num_hazards=1), non-toroidal, hazard_field_decay
default 0.5. hazard_field_view's centre-cell value is
hazard_field[agent_pos] / hazard_field.max() = hazard_field[agent_pos] / 1.0
(the field's global max is always exactly 1.0, attained at the hazard's own
cell, since hazard_field[h] = 1/(1+0*decay) = 1.0 there) = 1/(1 + dist*0.5)
where dist is the agent's Manhattan-ish falloff distance to the nearest
hazard. On a 10x10 non-toroidal grid, this ranges from 1.0 (at the hazard)
down toward ~0.05-0.1 at the farthest corners (dist ~18-19) -- unlike the
`.claude/skills/queue-experiment/SKILL.md` "Proxy label calibration" note's
cited range [0.22, 1.0], which was measured on a SMALLER (size=6) grid whose
maximum possible distance cannot reach the low end. This driver's own
--dry-run smoke prints the empirical min/max/quantiles of the observed
centre-cell value so the three-bin thresholds below can be (and were, before
finalising this script) checked against the actual distribution rather than
assumed.

GOV-REUSE-1: checked upstream before authoring -- 0/973 manifests tag
claim_ids containing MECH-027 (no prior MECH-027 run of any kind exists to
reuse or supersede).

SLEEP: use_sleep_loop/sws_enabled/rem_enabled are on (required to build
agent.sleep_loop so draws_per_cycle exists to force/revert -- channel (c)).
During warmup, sleep cycles are driven manually via
agent.sleep_loop.force_cycle() every SLEEP_EVERY_N_EPISODES episodes.
During eval, agent.force_sleep_cycle_at_eval_boundary() is called once at
the end of every eval episode in all three blocks (see channel (c) above
and MECH-027 Build 2) -- this is a deliberate CHANGE from the earlier
"no sleep firing during eval" draft, made because draws_per_cycle has no
observable effect unless a sleep cycle actually consults it. The waking
behavioural DV (avoidant-action rate) is still measured only from waking
env steps; the forced sleep cycles run at episode boundaries, between DV
observations, not during them.

PHASED TRAINING: none. This is a direct-perturbation behavioural probe, not
a representation-learning experiment -- no NEW head is trained on z_world/
z_harm/encoder output. The warmup phase trains only the standard E1
prediction loss, E2 world-forward loss, and E3 harm-eval head (the same
always-present substrate training every prior CausalGridWorldV2 script in
this repo trains), so the P0/P1/P2 phased-training discipline (freeze
encoder, train head on .detach()ed latents) does not apply; warmup episode
count (WARMUP_EPISODES) exists solely to reach a functional world model /
harm evaluator before eval, verified via the world_forward_r2 diagnostic.

PRE-REGISTERED THRESHOLDS (all defined as module-level constants below,
fixed before the real run; the --dry-run smoke calibration check above is
what confirmed these values give non-trivial bin coverage on THIS env
config, not what derived them from a scored run's own outcome):
  HAZARD_SAFE_MAX   = 0.15   (hazard_field_view centre < this -> SAFE)
  HAZARD_HIGH_MIN   = 0.50   (>= this -> HIGH; [SAFE_MAX, HIGH_MIN) -> AMBIGUOUS)
  FALSE_ALARM_ELEVATION_MULTIPLIER = 2.0
      C1 (LOAD-BEARING): EVAL_HYPERVIGILANT's pooled (across-seed mean)
      ambiguous-band avoidant rate must be >= 2x EVAL_BASELINE's pooled
      ambiguous-band avoidant rate, AND the HV value must lie outside
      (above) BASELINE's own empirical across-seed range (non-overlapping),
      per the design brief's own suggested bar. (Corrected 2026-09-02,
      red-team F8: checked on the pooled mean only -- the per-seed print in
      run_seed() is diagnostic, not a second gate.)
  REVERSION_RECOVERY_FLOOR = 0.5
      C2 (LOAD-BEARING): at least 50% of the BASELINE->HYPERVIGILANT
      elevation must revert in the EVAL_REVERSION block. Computed from the
      POOLED rates, not averaged per-seed ratios:
      pooled_recovered_fraction = (mean_hv_rate - mean_rev_rate) /
      (mean_hv_rate - mean_base_rate), must be >= 0.5. (Corrected 2026-09-02,
      red-team F6: a per-seed recovered_fraction is an unbounded ratio that
      can sign-flip on a near-zero or negative per-seed elevation, letting
      one degenerate seed decide C2 while C1 passes cleanly on pooled means
      -- the per-seed-averaged mean_recovered_fraction is still recorded as
      a diagnostic, but is no longer load-bearing.) This is the reversion
      half of the claim's own confirming signature -- required, not
      optional.
  PRECISION_RATIO_FLOOR = 5.0 (P0 readiness, not a claim criterion): pooled
      EVAL_HYPERVIGILANT precision mean / EVAL_BASELINE precision mean.
  HORIZON_RATIO_CEIL = 0.5 (P0 readiness): structural
      effective_horizon under MODE_HV / effective_horizon under MODE_BASE.
  POSITIVE_CONTROL_MARGIN = 0.05 (P0 readiness): EVAL_BASELINE's own
      avoidant rate must be higher in the HIGH band than the SAFE band by at
      least this much -- confirms the DV/env pairing is hazard-sensitive at
      all before trusting anything built on top of it.
  MIN_BIN_COVERAGE_STEPS = 5 (P0 readiness): every (block, hazard-bin) cell
      pooled across seeds must have at least this many observed steps, or
      the bin's rate is a near-empty-sample artifact, not a measurement.

  MECH-027 Build 1/2 readiness thresholds (P0, added 2026-09-02 -- see
  "GRADED DOWNSTREAM CONSUMER" / "SLEEP CYCLES ACTUALLY FIRE DURING EVAL"
  above; these gate the two new mechanisms actually engaging, distinct from
  the raw-manipulation checks above which only confirm the forcing itself):
  PRECISION_SCALED_COMMIT_ENGAGED_FRACTION_FLOOR = 0.05 (P0 readiness):
      fraction of real E3 selections (across HV+BASELINE, all seeds) where
      last_score_diagnostics["precision_scaled_commit_active"] was True --
      confirms the standalone committed branch (the only branch this lever
      is wired at) engaged on a MEANINGFUL share of selections, not
      silently bypassed by an inactive Factor-A shortlist / loop-
      segregation path on all but a negligible few ticks. (Corrected
      2026-09-02, red-team F3: an earlier raw count>=1 floor passed on
      negligible engagement, far too rare to plausibly drive a 2x DV
      elevation.)
  PRECISION_MARGIN_HV_ELEVATION_FLOOR = 0.01 (P0 readiness): pooled mean
      precision_margin_norm under EVAL_HYPERVIGILANT minus pooled mean under
      EVAL_BASELINE, must be positive and clear this floor -- confirms the
      forced precision perturbation genuinely moved the graded-consumer
      input, not just the raw _running_variance number (guards against a
      baseline that is already saturated near precision_margin_norm~1,
      leaving no room for HV to differ).
  COMMIT_TEMPERATURE_HV_REDUCTION_FLOOR = 0.01 (P0 readiness): pooled mean
      precision_scaled_commit_temperature_eff under EVAL_BASELINE minus
      pooled mean under EVAL_HYPERVIGILANT, must be positive and clear this
      floor -- confirms HV's elevated precision genuinely produced a
      COLDER (more rigid) commit temperature than baseline, the mechanism
      MECH-027 Build 1 predicts drives the hypervigilance direction.
  SLEEP_CYCLE_FIRE_FLOOR = 1 (P0 readiness): every seed, every block, must
      show at least this many non-None
      agent.force_sleep_cycle_at_eval_boundary() returns -- confirms the
      eval-boundary interleave actually fired (use_sleep_loop reachable),
      not silently no-op'd.
  REPLAY_MEASURED_HV_CEIL = 0 (P0 readiness): max mech285_n_draws MEASURED
      (from the force_sleep_cycle_at_eval_boundary() return, not the
      configured draws_per_cycle) across EVAL_HYPERVIGILANT firings, must be
      this or below -- confirms suppression was genuinely exercised by a
      firing sleep cycle, not merely configured on a channel nothing reads.
  REPLAY_MEASURED_BASELINE_FLOOR = 1 (P0 readiness): min mech285_n_draws
      MEASURED across EVAL_BASELINE (and EVAL_REVERSION) firings, must be at
      least this -- the baseline/reversion arm's replay channel is reachable
      for real, not just nominally unsuppressed.

Overall verdict: PASS iff readiness (all P0 preconditions met) AND C1 AND C2.
FAIL with label naming which of the three failed. Any unmet P0 precondition
routes the whole run to substrate_not_ready_requeue (evidence_direction
non_contributory), never a claim-pressure verdict, per the claim's own
non-degeneracy precondition.
"""

from __future__ import annotations

import argparse
import random
import time
from datetime import datetime
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
    p0_readiness_gate,
    P0NotReady,
)
from experiments.pack_writer import write_flat_manifest  # noqa: E402
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402


EXPERIMENT_TYPE = "v3_exq_981_mech027_control_plane_pathological_modes"
QUEUE_ID = "V3-EXQ-981"
EXPERIMENT_PURPOSE = "evidence"
CLAIM_IDS = ["MECH-027"]
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"

# Seed 44 excluded (documented reef-config early-death instability, CLAUDE.md).
SEEDS = [11, 23, 37]

# ---- Env -------------------------------------------------------------------
ENV_SIZE = 10
NUM_HAZARDS = 1
NUM_RESOURCES = 2
HAZARD_HARM = 0.05
HAZARD_FIELD_DECAY = 0.5

# ---- Agent dims --------------------------------------------------------------
WORLD_DIM = 32
SELF_DIM = 32

# ---- Training ----------------------------------------------------------------
WARMUP_EPISODES = 200
STEPS_PER_EPISODE = 120
NAV_BIAS = 0.25          # mild bias toward the hazard during warmup so the
                         # agent experiences enough near-hazard states to
                         # learn a meaningful harm-avoidance signal
SLEEP_EVERY_N_EPISODES = 25   # forced sleep-cycle cadence during warmup

# ---- Eval ----------------------------------------------------------------
EVAL_EPISODES_PER_BLOCK = 25
EVAL_STEPS_PER_EPISODE = 120

# ---- Channel-forcing constants (see module docstring "HOW EACH CHANNEL IS
# FORCED") ----
HV_PRECISION_SCALE = 0.05    # multiplicative shrink of running_variance/tick
BASE_PRECISION_SCALE = 1.0
HV_HORIZON_FRAC = 0.2        # CEM elite-scoring-window fraction under MODE_HV
MODE_BASE = "MODE_BASE"
MODE_HV = "MODE_HV"
MECH285_BASELINE_DRAWS = 50
MECH285_HV_DRAWS = 0

# ---- MECH-027 Build 1: graded precision-scaled commit temperature (see
# module docstring "GRADED DOWNSTREAM CONSUMER") -- pre-registered explicitly
# rather than left to E3Config's own defaults, since these are now load-
# bearing for channel (a)'s readiness gates ----
PRECISION_SCALED_COMMIT_ENTROPY_ALPHA = 1.0
PRECISION_SCALED_COMMIT_HARM_FLOOR = 0.25

# ---- Hazard-signal bins (pre-registered; see module docstring "THRESHOLD
# CALIBRATION") ----
HAZARD_SAFE_MAX = 0.15
HAZARD_HIGH_MIN = 0.50

# ---- Pre-registered criteria thresholds (see module docstring) ----
FALSE_ALARM_ELEVATION_MULTIPLIER = 2.0
REVERSION_RECOVERY_FLOOR = 0.5
PRECISION_RATIO_FLOOR = 5.0
HORIZON_RATIO_CEIL = 0.5
POSITIVE_CONTROL_MARGIN = 0.05
MIN_BIN_COVERAGE_STEPS = 5

# ---- MECH-027 Build 1/2 readiness thresholds (see module docstring
# "MECH-027 Build 1/2 readiness thresholds") ----
# PRECISION_SCALED_COMMIT_ENGAGED_FRACTION_FLOOR replaces an earlier raw-
# count floor (red-team F3, verified 2026-09-02): a count>=1 floor passes on
# negligible engagement, which cannot plausibly drive a 2x DV elevation.
PRECISION_SCALED_COMMIT_ENGAGED_FRACTION_FLOOR = 0.05
PRECISION_MARGIN_HV_ELEVATION_FLOOR = 0.01
COMMIT_TEMPERATURE_HV_REDUCTION_FLOOR = 0.01
SLEEP_CYCLE_FIRE_FLOOR = 1
REPLAY_MEASURED_HV_CEIL = 0
REPLAY_MEASURED_BASELINE_FLOOR = 1

BLOCK_BASELINE = "EVAL_BASELINE"
BLOCK_HYPERVIGILANT = "EVAL_HYPERVIGILANT"
BLOCK_REVERSION = "EVAL_REVERSION"
BLOCKS = (BLOCK_BASELINE, BLOCK_HYPERVIGILANT, BLOCK_REVERSION)

HAZARD_BIN_SAFE = "SAFE"
HAZARD_BIN_AMBIGUOUS = "AMBIGUOUS"
HAZARD_BIN_HIGH = "HIGH"


def build_config(env: CausalGridWorldV2) -> REEConfig:
    """ONE config shared by every arm/block -- only the runtime channel
    values (precision scale, operating_mode, sleep_loop.draws_per_cycle)
    differ between blocks; nothing about the agent's structure changes.
    """
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=SELF_DIM,
        world_dim=WORLD_DIM,
        reafference_action_dim=env.action_dim,
        use_event_classifier=True,
        # Sleep/replay channel (c) -- see module docstring. Manual-cycle-loop
        # driver (matches V3-EXQ-909's SLEEP DRIVER convention): K set very
        # high so the automatic K-episode cadence never fires; every firing
        # in this script is an explicit force_cycle() call.
        use_sleep_loop=True,
        sleep_loop_episodes_K=1_000_000,
        sws_enabled=True,
        rem_enabled=True,
        use_mech285_sampler=True,
        mech285_draws_per_cycle=MECH285_BASELINE_DRAWS,
        use_mech272_routing=True,
        use_anchor_sets=True,
        # Precision channel (a) graded consumer -- MECH-027 Build 1 (see
        # module docstring "GRADED DOWNSTREAM CONSUMER"). Without this, the
        # HV_PRECISION_SCALE forcing below only moves current_precision as a
        # number; the binary ARC-016 commit gate saturates in a trained
        # substrate and channel (a) has no further observable effect on
        # selection. use_harm_variance_commit stays at its default (False,
        # world-variance commit mode) -- required for precision_margin_norm
        # to be computed at all (see e3_selector.py select()).
        use_precision_scaled_commit_temperature=True,
        precision_scaled_commit_entropy_alpha=PRECISION_SCALED_COMMIT_ENTROPY_ALPHA,
        precision_scaled_commit_harm_floor=PRECISION_SCALED_COMMIT_HARM_FLOOR,
    )
    # Horizon channel (b) -- not reachable through from_dims(); set directly
    # per SD-MECH267-HORIZON-DEPTH (ree_core/utils/config.py HippocampalConfig).
    cfg.hippocampal.mode_conditioning_enabled = True
    cfg.hippocampal.mode_horizon_scale = {MODE_BASE: 1.0, MODE_HV: HV_HORIZON_FRAC}
    return cfg


def config_slice() -> Dict[str, Any]:
    """Exactly what each cell's computation reads -- no acceptance thresholds."""
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
        "eval_episodes_per_block": EVAL_EPISODES_PER_BLOCK,
        "hv_precision_scale": HV_PRECISION_SCALE,
        "hv_horizon_frac": HV_HORIZON_FRAC,
        "mech285_baseline_draws": MECH285_BASELINE_DRAWS,
        "mech285_hv_draws": MECH285_HV_DRAWS,
        "hazard_safe_max": HAZARD_SAFE_MAX,
        "hazard_high_min": HAZARD_HIGH_MIN,
        "use_precision_scaled_commit_temperature": True,
        "precision_scaled_commit_entropy_alpha": PRECISION_SCALED_COMMIT_ENTROPY_ALPHA,
        "precision_scaled_commit_harm_floor": PRECISION_SCALED_COMMIT_HARM_FLOOR,
    }


def _random_onehot(action_dim: int, device) -> torch.Tensor:
    v = torch.zeros(1, action_dim, device=device)
    v[0, random.randint(0, action_dim - 1)] = 1.0
    return v


def _hazard_cells(env: CausalGridWorldV2) -> List[Tuple[int, int]]:
    hz = np.argwhere(env.grid == env.ENTITY_TYPES["hazard"])
    return [(int(x), int(y)) for x, y in hz]


def _avoidant_action(env: CausalGridWorldV2, hazard_cells: List[Tuple[int, int]]) -> Optional[int]:
    """The single grid move maximising the dot product with the
    away-from-nearest-hazard vector. Computed live from env._action_map (not
    cached) so it stays correct even if a future config permutes the action
    map -- not enabled here (world_rule_shift_enabled defaults False), but
    the live read costs nothing and removes the dependency."""
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
    """Multi-rate-clock bookkeeping this driver owns itself, since it calls
    hippocampal.propose_trajectories() directly (bypassing
    agent.generate_trajectories()/_e3_tick(), which never forward
    operating_mode -- see module docstring point (b))."""

    def __init__(self) -> None:
        self.last_action: Optional[torch.Tensor] = None
        self.z_self_prev: Optional[torch.Tensor] = None
        self.action_prev: Optional[torch.Tensor] = None
        self.last_precision: float = 0.0


def _agent_tick(
    agent: REEAgent,
    env: CausalGridWorldV2,
    obs_dict: Dict[str, Any],
    state: _TickState,
    operating_mode: Optional[Dict[str, float]],
    precision_scale: float,
    world_dim: int,
    device,
    train: bool,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict[str, float]]]:
    """One environment tick. Returns (chosen action [1,action_dim], z_world
    [1,world_dim] detached, precision_scaled_commit_diagnostics) -- z_world
    is returned so callers that need it (e.g. the warmup world-forward
    buffer) never have to call agent.sense() a second time for the same
    tick. precision_scaled_commit_diagnostics is a FRESH snapshot of
    {"precision_margin_norm", "precision_scaled_commit_active",
    "precision_scaled_commit_temperature_eff"} taken from
    agent.e3.last_score_diagnostics -- but ONLY on a tick where E3 actually
    ran select() this call (ticks["e3_tick"] True AND candidates non-empty);
    None otherwise. Callers must not treat a None as "zero" -- it means "no
    fresh selection this tick", the same sample-size-integrity discipline as
    state.last_precision/state.last_action (both intentionally re-read across
    skipped ticks; the diagnostics snapshot intentionally is NOT, so pooled
    precision_margin_norm/T_eff stats reflect actual E3 firings only).

    Mirrors REEAgent._e3_tick's essential shape but calls
    hippocampal.propose_trajectories() directly so operating_mode (channel
    (b), the horizon lever) can be injected -- see module docstring. Channel
    (a) (precision) is forced AFTER agent.update_residue() drives its normal
    live update, so the channel stays non-degenerate (real per-tick
    movement) rather than frozen.
    """
    obs_body = obs_dict["body_state"]
    obs_world = obs_dict["world_state"]
    ctx = torch.no_grad() if not train else _nullcontext()
    precision_diag: Optional[Dict[str, float]] = None
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
        if ticks.get("e3_tick", False):
            candidates = agent.hippocampal.propose_trajectories(
                latent.z_world,
                latent.z_self,
                e1_prior=e1_prior,
                operating_mode=operating_mode,
            )
            if candidates:
                result = agent.e3.select(candidates, temperature=1.0)
                action = result.selected_action.detach()
                state.last_action = action
                state.last_precision = float(result.precision)
                diag = agent.e3.last_score_diagnostics
                precision_diag = {
                    "precision_margin_norm": float(diag.get("precision_margin_norm", -1.0)),
                    "precision_scaled_commit_active": bool(
                        diag.get("precision_scaled_commit_active", False)
                    ),
                    "precision_scaled_commit_temperature_eff": float(
                        diag.get("precision_scaled_commit_temperature_eff", -1.0)
                    ),
                }
        action = state.last_action
        if action is None:
            action = _random_onehot(env.action_dim, device)
            state.last_action = action

        drive_level = REEAgent.compute_drive_level(obs_body)
        benefit_exposure = max(0.0, float(obs_dict.get("benefit_exposure", 0.0)))
        agent.update_z_goal(benefit_exposure=benefit_exposure, drive_level=drive_level)

    state.z_self_prev = latent.z_self.detach()
    state.action_prev = action.detach()
    return action, latent.z_world.detach(), precision_diag


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


def _apply_precision_scale(agent: REEAgent, scale: float) -> None:
    """Channel (a): force _running_variance AFTER its normal live update, so
    the channel is perturbed but never frozen. scale=1.0 is a no-op."""
    if scale == 1.0:
        return
    rv = float(agent.e3._running_variance)
    agent.e3._running_variance = max(1e-9, rv * scale)


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
    agent.train()
    state = _TickState()
    harm_buf_pos: List[torch.Tensor] = []
    harm_buf_neg: List[torch.Tensor] = []
    wf_buf: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    total_harm = 0
    sleep_fires = 0

    for ep in range(num_episodes):
        flat_obs, obs_dict = env.reset()
        agent.reset()
        state = _TickState()
        z_world_prev: Optional[torch.Tensor] = None

        for _ in range(steps_per_episode):
            action, z_world_curr_pre, _precision_diag = _agent_tick(
                agent, env, obs_dict, state,
                operating_mode={MODE_BASE: 1.0},
                precision_scale=BASE_PRECISION_SCALE,
                world_dim=world_dim, device=device, train=True,
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
            print(f"  [train] ep {ep+1}/{num_episodes}  harm={total_harm}  sleep_fires={sleep_fires}", flush=True)

        if (ep + 1) % SLEEP_EVERY_N_EPISODES == 0:
            try:
                agent._flush_exploration_episode()
            except AttributeError:
                pass
            agent.sleep_loop.force_cycle(agent)
            sleep_fires += 1

    return {"total_harm": total_harm, "wf_buf": wf_buf, "sleep_fires": sleep_fires}


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


def _run_eval_block(
    agent: REEAgent,
    env: CausalGridWorldV2,
    block: str,
    num_episodes: int,
    steps_per_episode: int,
    world_dim: int,
    device,
    zg: ZGoalStreamAccumulator,
) -> Dict[str, Any]:
    """Run one eval regime block on the (already-trained) agent. Applies the
    block's channel settings (precision scale, operating_mode, replay draw
    count) at block ENTRY -- this is what makes EVAL_REVERSION a genuine
    within-run reversion rather than a fresh agent.

    MECH-027 Build 1/2 (2026-09-02): also captures, per real E3 selection,
    the precision-scaled-commit-temperature diagnostics
    (precision_margin_norm / precision_scaled_commit_temperature_eff /
    precision_scaled_commit_active) from agent.e3.last_score_diagnostics via
    _agent_tick's third return value; and fires a real sleep cycle at every
    eval episode boundary via agent.force_sleep_cycle_at_eval_boundary(),
    recording each firing's MEASURED mech285_n_draws -- see module docstring
    "GRADED DOWNSTREAM CONSUMER" and "SLEEP CYCLES ACTUALLY FIRE DURING
    EVAL"."""
    if block == BLOCK_BASELINE:
        operating_mode = {MODE_BASE: 1.0}
        precision_scale = BASE_PRECISION_SCALE
        agent.sleep_loop.draws_per_cycle = MECH285_BASELINE_DRAWS
    elif block == BLOCK_HYPERVIGILANT:
        operating_mode = {MODE_HV: 1.0}
        precision_scale = HV_PRECISION_SCALE
        agent.sleep_loop.draws_per_cycle = MECH285_HV_DRAWS
    elif block == BLOCK_REVERSION:
        operating_mode = {MODE_BASE: 1.0}
        precision_scale = BASE_PRECISION_SCALE
        agent.sleep_loop.draws_per_cycle = MECH285_BASELINE_DRAWS
    else:
        raise ValueError(f"unknown block {block!r}")

    agent.eval()
    state = _TickState()
    step_rows: List[Dict[str, Any]] = []
    precision_samples: List[float] = []
    effective_horizon_samples: List[float] = []
    precision_margin_samples: List[float] = []
    commit_temp_eff_samples: List[float] = []
    precision_scaled_commit_engaged_count = 0
    e3_selection_count = 0
    sleep_fires = 0
    sleep_mech285_draws: List[int] = []
    fatal = 0

    for ep in range(num_episodes):
        flat_obs, obs_dict = env.reset()
        agent.reset()
        state = _TickState()

        for step_idx in range(steps_per_episode):
            hazard_cells = _hazard_cells(env)
            hv = obs_dict.get("hazard_field_view", None)
            hazard_value = float(hv[12]) if hv is not None else 0.0
            hbin = _hazard_bin(hazard_value)
            avoidant = _avoidant_action(env, hazard_cells)

            try:
                action, _zw, precision_diag = _agent_tick(
                    agent, env, obs_dict, state,
                    operating_mode=operating_mode,
                    precision_scale=precision_scale,
                    world_dim=world_dim, device=device, train=False,
                )
            except Exception:
                fatal += 1
                action = _random_onehot(env.action_dim, device)
                state.last_action = action
                precision_diag = None

            if precision_diag is not None:
                e3_selection_count += 1
                # precision_margin_norm is valid whenever the world-variance
                # commit gate ran with effective_threshold>0 -- independent
                # of whether the precision-scaled-commit BRANCH specifically
                # engaged (see module docstring). Pool it whenever it is a
                # real reading (not the -1.0 "not computed" sentinel).
                margin = precision_diag["precision_margin_norm"]
                if margin >= 0.0:
                    precision_margin_samples.append(margin)
                # precision_scaled_commit_temperature_eff, in contrast, is
                # ONLY set when the branch actually fired (envelope admitted
                # >=2 candidates); pooling it unconditionally would mix real
                # T_eff readings with the -1.0 "branch did not engage"
                # sentinel and silently bias the pooled mean (caught in
                # this script's own smoke test: an unfiltered pool put the
                # BASELINE mean below 1.0 while HV's mean, which happened to
                # engage on every tick in the tiny dry run, read ~1.0 --
                # backwards from the predicted direction).
                if precision_diag["precision_scaled_commit_active"]:
                    precision_scaled_commit_engaged_count += 1
                    commit_temp_eff_samples.append(
                        precision_diag["precision_scaled_commit_temperature_eff"]
                    )

            chosen_idx = int(action.argmax(dim=-1).item())
            took_avoidant = bool(avoidant is not None and chosen_idx == avoidant)

            step_rows.append({
                "hazard_value": hazard_value,
                "hazard_bin": hbin,
                "took_avoidant": took_avoidant,
            })

            eh = getattr(agent.hippocampal, "_last_effective_horizon", None)
            if eh is not None:
                effective_horizon_samples.append(float(eh))

            flat_obs, harm_signal, done, info, obs_dict = env.step(action)
            agent.update_residue(
                harm_signal=float(harm_signal), world_delta=None,
                hypothesis_tag=False, owned=True,
            )
            _apply_precision_scale(agent, precision_scale)
            precision_samples.append(float(agent.e3.current_precision))

            if done:
                break

        # MECH-027 Build 2 -- fire a real sleep cycle at this eval episode
        # boundary (BEFORE the next iteration's agent.reset(), which would
        # otherwise flush an already-empty exploration buffer). See module
        # docstring "SLEEP CYCLES ACTUALLY FIRE DURING EVAL". This is what
        # makes the block's draws_per_cycle setting (above) an exercised
        # lever rather than a dead parameter: mech285_n_draws in the
        # returned metrics is the MEASURED replay count for this firing.
        sleep_metrics = agent.force_sleep_cycle_at_eval_boundary()
        if sleep_metrics is not None:
            sleep_fires += 1
            sleep_mech285_draws.append(int(sleep_metrics.get("mech285_n_draws", -1)))

        print(f"  [eval] block={block} ep {ep+1}/{num_episodes} steps_logged={len(step_rows)}", flush=True)

    zg.observe(agent)

    bins: Dict[str, Dict[str, int]] = {
        HAZARD_BIN_SAFE: {"n": 0, "avoidant": 0},
        HAZARD_BIN_AMBIGUOUS: {"n": 0, "avoidant": 0},
        HAZARD_BIN_HIGH: {"n": 0, "avoidant": 0},
    }
    for row in step_rows:
        b = bins[row["hazard_bin"]]
        b["n"] += 1
        if row["took_avoidant"]:
            b["avoidant"] += 1

    rates = {
        k: (v["avoidant"] / v["n"] if v["n"] > 0 else 0.0) for k, v in bins.items()
    }

    return {
        "block": block,
        "n_steps": len(step_rows),
        "bins": bins,
        "rates": rates,
        "precision_mean": float(np.mean(precision_samples)) if precision_samples else 0.0,
        "precision_samples": precision_samples,
        "effective_horizon_mean": (
            float(np.mean(effective_horizon_samples)) if effective_horizon_samples else None
        ),
        "draws_per_cycle": int(agent.sleep_loop.draws_per_cycle),
        "fatal_errors": fatal,
        # MECH-027 Build 1 -- graded precision-scaled commit temperature.
        "precision_margin_norm_mean": (
            float(np.mean(precision_margin_samples)) if precision_margin_samples else -1.0
        ),
        "precision_margin_norm_samples": precision_margin_samples,
        "commit_temperature_eff_mean": (
            float(np.mean(commit_temp_eff_samples)) if commit_temp_eff_samples else -1.0
        ),
        "commit_temperature_eff_samples": commit_temp_eff_samples,
        "precision_scaled_commit_engaged_count": precision_scaled_commit_engaged_count,
        "e3_selection_count": e3_selection_count,
        # MECH-027 Build 2 -- eval-boundary sleep-cycle interleave.
        "sleep_fires": sleep_fires,
        "sleep_mech285_draws": sleep_mech285_draws,
        "sleep_mech285_draws_max": (
            max(sleep_mech285_draws) if sleep_mech285_draws else -1
        ),
        "sleep_mech285_draws_min": (
            min(sleep_mech285_draws) if sleep_mech285_draws else -1
        ),
    }


def run_seed(seed: int, dry_run: bool) -> Dict[str, Any]:
    device = torch.device("cpu")
    reset_all_rng(seed)

    print(f"\nSeed {seed} Condition control_plane_hypervigilance_probe", flush=True)

    env = CausalGridWorldV2(
        seed=seed, size=ENV_SIZE, num_hazards=NUM_HAZARDS, num_resources=NUM_RESOURCES,
        hazard_harm=HAZARD_HARM, hazard_field_decay=HAZARD_FIELD_DECAY,
    )
    cfg = build_config(env)
    agent = REEAgent(cfg).to(device)

    optimizer = optim.Adam(list(agent.e1.parameters()), lr=1e-3)
    wf_optimizer = optim.Adam(
        list(agent.e2.world_transition.parameters()) + list(agent.e2.world_action_encoder.parameters()),
        lr=1e-3,
    )
    harm_eval_optimizer = optim.Adam(list(agent.e3.harm_eval_head.parameters()), lr=1e-4)

    warmup_eps = 5 if dry_run else WARMUP_EPISODES
    warmup_steps = 15 if dry_run else STEPS_PER_EPISODE
    eval_eps = 3 if dry_run else EVAL_EPISODES_PER_BLOCK
    eval_steps = 15 if dry_run else EVAL_STEPS_PER_EPISODE

    train_out = _train_warmup(
        agent, env, optimizer, wf_optimizer, harm_eval_optimizer,
        warmup_eps, warmup_steps, WORLD_DIM, device,
    )
    world_forward_r2 = _compute_world_forward_r2(agent, train_out["wf_buf"])
    print(f"  world_forward_r2={world_forward_r2:.4f}  sleep_fires={train_out['sleep_fires']}"
          f"  draws_per_cycle(post-warmup)={agent.sleep_loop.draws_per_cycle}", flush=True)

    zg = ZGoalStreamAccumulator()
    block_results: Dict[str, Dict[str, Any]] = {}
    for block in BLOCKS:
        block_results[block] = _run_eval_block(
            agent, env, block, eval_eps, eval_steps, WORLD_DIM, device, zg,
        )

    base_rate = block_results[BLOCK_BASELINE]["rates"][HAZARD_BIN_AMBIGUOUS]
    hv_rate = block_results[BLOCK_HYPERVIGILANT]["rates"][HAZARD_BIN_AMBIGUOUS]
    rev_rate = block_results[BLOCK_REVERSION]["rates"][HAZARD_BIN_AMBIGUOUS]
    elevation = hv_rate - base_rate
    recovered = (hv_rate - rev_rate) / elevation if abs(elevation) > 1e-9 else 0.0

    verdict = "PASS" if (elevation > 0 and hv_rate >= FALSE_ALARM_ELEVATION_MULTIPLIER * max(base_rate, 1e-9)) else "FAIL"
    print(f"verdict: {verdict}  seed={seed}  base_ambig_rate={base_rate:.4f}"
          f"  hv_ambig_rate={hv_rate:.4f}  rev_ambig_rate={rev_rate:.4f}"
          f"  recovered_fraction={recovered:.4f}", flush=True)

    return {
        "seed": seed,
        "env_config": {
            "size": ENV_SIZE, "num_hazards": NUM_HAZARDS, "num_resources": NUM_RESOURCES,
            "hazard_harm": HAZARD_HARM, "hazard_field_decay": HAZARD_FIELD_DECAY,
        },
        "world_forward_r2": world_forward_r2,
        "sleep_fires_warmup": train_out["sleep_fires"],
        "block_results": block_results,
        "base_ambiguous_rate": base_rate,
        "hv_ambiguous_rate": hv_rate,
        "reversion_ambiguous_rate": rev_rate,
        "recovered_fraction": recovered,
        "agent": agent,
        "zg": zg,
    }


def run_experiment(seeds: List[int], dry_run: bool) -> Dict[str, Any]:
    t0 = time.perf_counter()
    per_seed: Dict[int, Dict[str, Any]] = {}
    arm_rows: List[Dict[str, Any]] = []
    slice_ = config_slice()

    for seed in seeds:
        with arm_cell(seed, config_slice=slice_, script_path=Path(__file__),
                      config_slice_declared=True) as cell:
            result = run_seed(seed, dry_run)
            row = {
                "seed": seed,
                "world_forward_r2": result["world_forward_r2"],
                "sleep_fires_warmup": result["sleep_fires_warmup"],
                "base_ambiguous_rate": result["base_ambiguous_rate"],
                "hv_ambiguous_rate": result["hv_ambiguous_rate"],
                "reversion_ambiguous_rate": result["reversion_ambiguous_rate"],
                "recovered_fraction": result["recovered_fraction"],
                "block_rates": {b: result["block_results"][b]["rates"] for b in BLOCKS},
                "block_bins": {b: result["block_results"][b]["bins"] for b in BLOCKS},
                "block_precision_mean": {b: result["block_results"][b]["precision_mean"] for b in BLOCKS},
                "block_effective_horizon_mean": {
                    b: result["block_results"][b]["effective_horizon_mean"] for b in BLOCKS
                },
                "block_draws_per_cycle": {b: result["block_results"][b]["draws_per_cycle"] for b in BLOCKS},
                "block_fatal_errors": {b: result["block_results"][b]["fatal_errors"] for b in BLOCKS},
                "block_precision_margin_norm_mean": {
                    b: result["block_results"][b]["precision_margin_norm_mean"] for b in BLOCKS
                },
                "block_commit_temperature_eff_mean": {
                    b: result["block_results"][b]["commit_temperature_eff_mean"] for b in BLOCKS
                },
                "block_precision_scaled_commit_engaged_count": {
                    b: result["block_results"][b]["precision_scaled_commit_engaged_count"] for b in BLOCKS
                },
                "block_e3_selection_count": {
                    b: result["block_results"][b]["e3_selection_count"] for b in BLOCKS
                },
                "block_sleep_fires": {b: result["block_results"][b]["sleep_fires"] for b in BLOCKS},
                "block_sleep_mech285_draws_max": {
                    b: result["block_results"][b]["sleep_mech285_draws_max"] for b in BLOCKS
                },
                "block_sleep_mech285_draws_min": {
                    b: result["block_results"][b]["sleep_mech285_draws_min"] for b in BLOCKS
                },
            }
            cell.stamp(row)
        arm_rows.append(row)
        per_seed[seed] = result

    # ---- pooled / aggregate readings ---------------------------------------
    base_rates = [r["base_ambiguous_rate"] for r in arm_rows]
    hv_rates = [r["hv_ambiguous_rate"] for r in arm_rows]
    rev_rates = [r["reversion_ambiguous_rate"] for r in arm_rows]
    recovered_fracs = [r["recovered_fraction"] for r in arm_rows]

    mean_base_rate = float(np.mean(base_rates))
    mean_hv_rate = float(np.mean(hv_rates))
    mean_rev_rate = float(np.mean(rev_rates))
    mean_recovered = float(np.mean(recovered_fracs))
    base_rate_max = float(np.max(base_rates)) if base_rates else 0.0

    # Red-team F6 (chip-20260902-mech027-precision-replay-eval-substrate,
    # verified 2026-09-02): C2's per-seed recovered_fraction is an unbounded
    # ratio (elevation_seed can be near-zero or even negative), so averaging
    # 3 of them lets one degenerate/sign-flipped seed decide C2 while C1
    # passes cleanly on pooled means. The LOAD-BEARING C2 measure below is
    # instead computed directly from the POOLED (mean_hv/mean_base/mean_rev)
    # rates -- one well-powered ratio instead of a mean of 3 noisy ones.
    # mean_recovered (per-seed-averaged) is KEPT as a diagnostic, not
    # load-bearing.
    pooled_elevation = mean_hv_rate - mean_base_rate
    pooled_recovered_fraction = (
        (mean_hv_rate - mean_rev_rate) / pooled_elevation
        if abs(pooled_elevation) > 1e-9 else 0.0
    )

    precision_base_mean = float(np.mean([r["block_precision_mean"][BLOCK_BASELINE] for r in arm_rows]))
    precision_hv_mean = float(np.mean([r["block_precision_mean"][BLOCK_HYPERVIGILANT] for r in arm_rows]))
    precision_ratio = (precision_hv_mean / precision_base_mean) if precision_base_mean > 1e-12 else 0.0

    eh_base_vals = [
        r["block_effective_horizon_mean"][BLOCK_BASELINE] for r in arm_rows
        if r["block_effective_horizon_mean"][BLOCK_BASELINE] is not None
    ]
    eh_hv_vals = [
        r["block_effective_horizon_mean"][BLOCK_HYPERVIGILANT] for r in arm_rows
        if r["block_effective_horizon_mean"][BLOCK_HYPERVIGILANT] is not None
    ]
    eh_base_mean = float(np.mean(eh_base_vals)) if eh_base_vals else None
    eh_hv_mean = float(np.mean(eh_hv_vals)) if eh_hv_vals else None
    horizon_ratio = (
        (eh_hv_mean / eh_base_mean) if (eh_base_mean and eh_hv_mean and eh_base_mean > 1e-9) else None
    )

    draws_hv_vals = [r["block_draws_per_cycle"][BLOCK_HYPERVIGILANT] for r in arm_rows]
    draws_base_vals = [r["block_draws_per_cycle"][BLOCK_BASELINE] for r in arm_rows]
    draws_rev_vals = [r["block_draws_per_cycle"][BLOCK_REVERSION] for r in arm_rows]

    # ---- MECH-027 Build 1: graded precision-scaled commit temperature -----
    precision_margin_base_mean = float(
        np.mean([r["block_precision_margin_norm_mean"][BLOCK_BASELINE] for r in arm_rows])
    )
    precision_margin_hv_mean = float(
        np.mean([r["block_precision_margin_norm_mean"][BLOCK_HYPERVIGILANT] for r in arm_rows])
    )
    precision_margin_hv_elevation = precision_margin_hv_mean - precision_margin_base_mean

    commit_temp_base_mean = float(
        np.mean([r["block_commit_temperature_eff_mean"][BLOCK_BASELINE] for r in arm_rows])
    )
    commit_temp_hv_mean = float(
        np.mean([r["block_commit_temperature_eff_mean"][BLOCK_HYPERVIGILANT] for r in arm_rows])
    )
    commit_temp_hv_reduction = commit_temp_base_mean - commit_temp_hv_mean

    precision_scaled_commit_engaged_total = sum(
        r["block_precision_scaled_commit_engaged_count"][b]
        for r in arm_rows for b in (BLOCK_BASELINE, BLOCK_HYPERVIGILANT)
    )
    e3_selection_total = sum(
        r["block_e3_selection_count"][b]
        for r in arm_rows for b in (BLOCK_BASELINE, BLOCK_HYPERVIGILANT)
    )
    # Red-team F3 (chip-20260902-mech027-precision-replay-eval-substrate,
    # verified 2026-09-02): a raw engagement COUNT (PRECISION_SCALED_
    # COMMIT_ENGAGED_FLOOR=1) is a liveness check, not a reach check -- it
    # passes even if the branch engaged on a negligible fraction of the
    # ~hundreds-to-thousands of pooled E3 selections, far too rarely to be
    # a plausible driver of a 2x DV elevation. The readiness gate below
    # uses this FRACTION instead.
    precision_scaled_commit_engaged_fraction = (
        precision_scaled_commit_engaged_total / e3_selection_total
        if e3_selection_total > 0 else 0.0
    )

    # ---- MECH-027 Build 2: eval-boundary sleep-cycle interleave -----------
    sleep_fires_vals = [r["block_sleep_fires"][b] for r in arm_rows for b in BLOCKS]
    min_sleep_fires = min(sleep_fires_vals) if sleep_fires_vals else 0

    sleep_draws_hv_max_vals = [r["block_sleep_mech285_draws_max"][BLOCK_HYPERVIGILANT] for r in arm_rows]
    sleep_draws_base_min_vals = [r["block_sleep_mech285_draws_min"][BLOCK_BASELINE] for r in arm_rows]
    sleep_draws_rev_min_vals = [r["block_sleep_mech285_draws_min"][BLOCK_REVERSION] for r in arm_rows]

    # positive control: BASELINE avoidant rate must be higher in HIGH than SAFE
    high_rates = [r["block_rates"][BLOCK_BASELINE][HAZARD_BIN_HIGH] for r in arm_rows]
    safe_rates = [r["block_rates"][BLOCK_BASELINE][HAZARD_BIN_SAFE] for r in arm_rows]
    positive_control_margin = float(np.mean(high_rates) - np.mean(safe_rates))

    # min bin coverage, pooled across seeds, per block+bin
    min_bin_coverage = min(
        r["block_bins"][b][hb]["n"]
        for r in arm_rows for b in BLOCKS for hb in (HAZARD_BIN_SAFE, HAZARD_BIN_AMBIGUOUS, HAZARD_BIN_HIGH)
    ) if arm_rows else 0

    total_fatal = sum(
        r["block_fatal_errors"][b] for r in arm_rows for b in BLOCKS
    )

    readiness_checks = [
        {
            "name": "precision_channel_non_degenerate",
            "measured": precision_ratio,
            "threshold": PRECISION_RATIO_FLOOR,
            "direction": "lower",
            "control": (
                "pooled EVAL_HYPERVIGILANT precision mean / EVAL_BASELINE precision "
                f"mean, must clear {PRECISION_RATIO_FLOOR}x -- confirms the forced "
                "precision perturbation actually elevated the channel."
            ),
        },
        {
            "name": "horizon_channel_non_degenerate",
            "measured": (horizon_ratio if horizon_ratio is not None else 1.0),
            "threshold": HORIZON_RATIO_CEIL,
            "direction": "upper",
            "control": (
                "structural effective_horizon under MODE_HV / effective_horizon "
                f"under MODE_BASE, must clear (be below) {HORIZON_RATIO_CEIL} -- "
                "confirms the SD-MECH267-HORIZON-DEPTH scoring window actually "
                "shortened under the hypervigilant regime."
            ),
        },
        {
            "name": "replay_channel_non_degenerate",
            "measured": float(max(sleep_draws_hv_max_vals) if sleep_draws_hv_max_vals else 1.0),
            "threshold": float(REPLAY_MEASURED_HV_CEIL),
            "direction": "upper",
            "control": (
                "MEASURED mech285_n_draws (from force_sleep_cycle_at_eval_boundary()'s "
                "own return, not the configured draws_per_cycle) across every "
                "EVAL_HYPERVIGILANT sleep-cycle firing, on every seed, must be "
                f"<= {REPLAY_MEASURED_HV_CEIL} -- confirms replay suppression was "
                "exercised by a REAL firing sleep cycle, not merely a configured "
                "knob nothing during eval ever reads (MECH-027 Build 2)."
            ),
        },
        {
            "name": "replay_channel_baseline_reachable",
            "measured": float(min(sleep_draws_base_min_vals) if sleep_draws_base_min_vals else 0.0),
            "threshold": float(REPLAY_MEASURED_BASELINE_FLOOR),
            "direction": "lower",
            "control": (
                "MEASURED mech285_n_draws across every EVAL_BASELINE sleep-cycle "
                f"firing, on every seed, must be >= {REPLAY_MEASURED_BASELINE_FLOOR} "
                "(the V3-EXQ-909 three-flag recipe -- use_anchor_sets + "
                "use_mech285_sampler + use_mech272_routing -- actually reached a "
                "live, non-zero draw count on a REAL firing, not just a configured "
                "value)."
            ),
        },
        {
            "name": "sleep_cycle_fires_during_eval",
            "measured": float(min_sleep_fires),
            "threshold": float(SLEEP_CYCLE_FIRE_FLOOR),
            "direction": "lower",
            "control": (
                "the smallest per-seed, per-block count of non-None "
                "agent.force_sleep_cycle_at_eval_boundary() returns -- confirms "
                "the eval-boundary sleep-cycle interleave (MECH-027 Build 2) "
                "actually fired in every block, on every seed, rather than "
                "silently no-op'ing (e.g. use_sleep_loop unreachable)."
            ),
        },
        {
            "name": "precision_scaled_commit_temperature_engaged",
            "measured": precision_scaled_commit_engaged_fraction,
            "threshold": PRECISION_SCALED_COMMIT_ENGAGED_FRACTION_FLOOR,
            "direction": "lower",
            "control": (
                "fraction (across EVAL_BASELINE + EVAL_HYPERVIGILANT, all "
                "seeds) of real E3 selections where "
                "last_score_diagnostics['precision_scaled_commit_active'] was "
                "True -- confirms the standalone committed branch (the only "
                "branch use_precision_scaled_commit_temperature is wired at) "
                "engaged on a MEANINGFUL share of selections, not just once "
                "(MECH-027 Build 1; a raw count>=1 floor is a liveness check, "
                "not a reach check -- red-team F3, "
                "chip-20260902-mech027-precision-replay-eval-substrate, "
                "verified 2026-09-02). Raw engaged/total counts are in "
                "metrics.precision_scaled_commit_engaged_total / "
                "metrics.e3_selection_total."
            ),
        },
        {
            "name": "precision_margin_norm_elevated_under_hv",
            "measured": precision_margin_hv_elevation,
            "threshold": PRECISION_MARGIN_HV_ELEVATION_FLOOR,
            "direction": "lower",
            "control": (
                "pooled mean precision_margin_norm under EVAL_HYPERVIGILANT minus "
                "pooled mean under EVAL_BASELINE -- confirms the forced precision "
                "perturbation genuinely moved the graded-consumer's input, not "
                "just the raw _running_variance number (guards against a baseline "
                "already saturated near precision_margin_norm~1, leaving HV no "
                "room to differ; MECH-027 Build 1)."
            ),
        },
        {
            "name": "commit_temperature_reduced_under_hv",
            "measured": commit_temp_hv_reduction,
            "threshold": COMMIT_TEMPERATURE_HV_REDUCTION_FLOOR,
            "direction": "lower",
            "control": (
                "pooled mean precision_scaled_commit_temperature_eff under "
                "EVAL_BASELINE minus pooled mean under EVAL_HYPERVIGILANT -- "
                "confirms HV's elevated precision genuinely produced a COLDER "
                "(more rigid) commit temperature than baseline, the mechanism "
                "MECH-027 Build 1 predicts drives the hypervigilance direction."
            ),
        },
        {
            "name": "positive_control_hazard_sensitivity",
            "measured": positive_control_margin,
            "threshold": POSITIVE_CONTROL_MARGIN,
            "direction": "lower",
            "control": (
                "EVAL_BASELINE avoidant-action rate in the HIGH hazard band minus "
                "the SAFE band -- confirms the DV/env pairing is hazard-sensitive "
                "at all before trusting anything built on top of it."
            ),
        },
        {
            "name": "hazard_bin_sample_coverage",
            "measured": float(min_bin_coverage),
            "threshold": float(MIN_BIN_COVERAGE_STEPS),
            "direction": "lower",
            "control": (
                "the smallest (block, hazard-bin) pooled step count across every "
                "seed/block/bin combination -- every bin must be genuinely "
                "populated, not a near-empty-sample artifact."
            ),
        },
        {
            "name": "no_fatal_action_selection_errors",
            "measured": float(total_fatal),
            "threshold": 0.5,
            "direction": "upper",
            "control": (
                "total count of _agent_tick exceptions across every seed/block "
                "-- each one substitutes a RANDOM action into the DV stream "
                "(_run_eval_block's except clause) -- must be 0. A partial- "
                "failure run injects near-chance avoidant-rate noise into "
                "exactly one block without tripping any other readiness gate, "
                "silently corrupting C1/C2 (red-team F1, "
                "chip-20260902-mech027-precision-replay-eval-substrate, "
                "verified 2026-09-02)."
            ),
        },
    ]

    ready = True
    preconditions: List[Dict[str, Any]] = []
    try:
        preconditions = p0_readiness_gate(readiness_checks)
    except P0NotReady as e:
        preconditions = e.preconditions
        ready = False

    degeneracy = check_degeneracy({
        "precision_samples_hv": {
            "groups": [per_seed[s]["block_results"][BLOCK_HYPERVIGILANT]["precision_samples"] for s in seeds],
        },
    })
    non_degenerate = degeneracy["non_degenerate"]

    # ---- claim criteria (load-bearing) --------------------------------------
    c1_pass = bool(
        mean_hv_rate >= FALSE_ALARM_ELEVATION_MULTIPLIER * max(mean_base_rate, 1e-9)
        and mean_hv_rate > base_rate_max
    )
    c2_pass = bool(pooled_recovered_fraction >= REVERSION_RECOVERY_FLOOR)

    all_pass = c1_pass and c2_pass
    criteria_met = sum([c1_pass, c2_pass])

    if not ready:
        status = "FAIL"
        evidence_direction = "non_contributory"
        label = "substrate_not_ready_requeue"
    elif not non_degenerate:
        status = "FAIL"
        evidence_direction = "non_contributory"
        label = "precision_channel_degenerate_vacuous_test"
    elif all_pass:
        status = "PASS"
        evidence_direction = "supports"
        label = "hypervigilance_signature_reproduced_and_reverts"
    elif c1_pass and not c2_pass:
        status = "FAIL"
        evidence_direction = "mixed"
        label = "false_alarm_elevation_confirmed_reversion_incomplete"
    else:
        status = "FAIL"
        evidence_direction = "weakens"
        label = "false_alarm_elevation_not_confirmed"

    print(f"\nV3-EXQ-981 pooled verdict: {status}  label={label}  ({criteria_met}/2)", flush=True)

    criteria = [
        {
            "name": "C1_false_alarm_elevation", "load_bearing": True, "passed": c1_pass,
            "measured": mean_hv_rate, "threshold": FALSE_ALARM_ELEVATION_MULTIPLIER * mean_base_rate,
        },
        {
            "name": "C2_reversion_recovery", "load_bearing": True, "passed": c2_pass,
            "measured": pooled_recovered_fraction, "threshold": REVERSION_RECOVERY_FLOOR,
        },
    ]
    combination_rule = (
        "overall_pass = READY (all P0 preconditions met) AND non_degenerate AND "
        "C1 (HV ambiguous-band false-alarm rate >= 2x baseline AND outside "
        "baseline's own across-seed range) AND C2 (>= 50% of the elevation "
        "reverts in the EVAL_REVERSION block, same agent, channels restored to "
        "baseline). Any P0 precondition unmet routes to substrate_not_ready_requeue."
    )

    metrics = {
        "mean_base_ambiguous_rate": mean_base_rate,
        "mean_hv_ambiguous_rate": mean_hv_rate,
        "mean_reversion_ambiguous_rate": mean_rev_rate,
        "mean_recovered_fraction": mean_recovered,
        "pooled_recovered_fraction": pooled_recovered_fraction,
        "base_ambiguous_rate_max_across_seeds": base_rate_max,
        "precision_base_mean": precision_base_mean,
        "precision_hv_mean": precision_hv_mean,
        "precision_ratio": precision_ratio,
        "effective_horizon_base_mean": eh_base_mean if eh_base_mean is not None else -1.0,
        "effective_horizon_hv_mean": eh_hv_mean if eh_hv_mean is not None else -1.0,
        "horizon_ratio": horizon_ratio if horizon_ratio is not None else -1.0,
        "draws_per_cycle_hv_max": float(max(draws_hv_vals) if draws_hv_vals else -1.0),
        "draws_per_cycle_base_min": float(min(draws_base_vals) if draws_base_vals else -1.0),
        "draws_per_cycle_reversion_min": float(min(draws_rev_vals) if draws_rev_vals else -1.0),
        "positive_control_margin": positive_control_margin,
        "min_bin_coverage_steps": float(min_bin_coverage),
        "total_fatal_errors": float(total_fatal),
        "criteria_met": float(criteria_met),
        # MECH-027 Build 1 -- graded precision-scaled commit temperature.
        "precision_margin_norm_base_mean": precision_margin_base_mean,
        "precision_margin_norm_hv_mean": precision_margin_hv_mean,
        "precision_margin_norm_hv_elevation": precision_margin_hv_elevation,
        "commit_temperature_eff_base_mean": commit_temp_base_mean,
        "commit_temperature_eff_hv_mean": commit_temp_hv_mean,
        "commit_temperature_eff_hv_reduction": commit_temp_hv_reduction,
        "precision_scaled_commit_engaged_total": float(precision_scaled_commit_engaged_total),
        "e3_selection_total": float(e3_selection_total),
        "precision_scaled_commit_engaged_fraction": precision_scaled_commit_engaged_fraction,
        # MECH-027 Build 2 -- eval-boundary sleep-cycle interleave.
        "sleep_fires_min_per_seed_block": float(min_sleep_fires),
        "sleep_mech285_draws_hv_max": float(max(sleep_draws_hv_max_vals) if sleep_draws_hv_max_vals else -1.0),
        "sleep_mech285_draws_base_min": float(min(sleep_draws_base_min_vals) if sleep_draws_base_min_vals else -1.0),
        "sleep_mech285_draws_reversion_min": float(
            min(sleep_draws_rev_min_vals) if sleep_draws_rev_min_vals else -1.0
        ),
    }
    for r in arm_rows:
        s = r["seed"]
        metrics[f"seed{s}_base_ambiguous_rate"] = r["base_ambiguous_rate"]
        metrics[f"seed{s}_hv_ambiguous_rate"] = r["hv_ambiguous_rate"]
        metrics[f"seed{s}_reversion_ambiguous_rate"] = r["reversion_ambiguous_rate"]
        metrics[f"seed{s}_recovered_fraction"] = r["recovered_fraction"]

    seed_lines = "\n".join(
        f"| {r['seed']} | {r['base_ambiguous_rate']:.4f} | {r['hv_ambiguous_rate']:.4f} |"
        f" {r['reversion_ambiguous_rate']:.4f} | {r['recovered_fraction']:.4f} |"
        f" {r['block_precision_mean'][BLOCK_BASELINE]:.4g} | {r['block_precision_mean'][BLOCK_HYPERVIGILANT]:.4g} |"
        for r in arm_rows
    )

    summary_markdown = f"""# V3-EXQ-981 -- MECH-027: control-plane hypervigilance signature probe

**Overall Status:** {status}  (label: `{label}`, {criteria_met}/2 load-bearing criteria)
**Claim:** MECH-027 -- hypervigilance is a mis-tuned regime of elevated
gain/precision + shortened prediction horizon + suppressed replay, not a
separate mechanism (scoped single-signature probe; see module docstring).
**Seeds:** {seeds}

## Per-seed ambiguous-band false-alarm rate

| Seed | Baseline | Hypervigilant | Reversion | Recovered frac | precision(base) | precision(HV) |
|---|---|---|---|---|---|---|
{seed_lines}

## Readiness (P0) preconditions

{chr(10).join(f"- {p['name']}: measured={p['measured']:.4g} threshold={p['threshold']:.4g} met={p['met']}" for p in preconditions)}

## Criteria

- C1 (LOAD-BEARING) false-alarm elevation: {"PASS" if c1_pass else "FAIL"} (mean_hv={mean_hv_rate:.4f} vs 2x mean_base={2*mean_base_rate:.4f}, base_max={base_rate_max:.4f})
- C2 (LOAD-BEARING) reversion recovery: {"PASS" if c2_pass else "FAIL"} (pooled_recovered_fraction={pooled_recovered_fraction:.4f}, per-seed-mean={mean_recovered:.4f} [diagnostic only], vs floor {REVERSION_RECOVERY_FLOOR})

{combination_rule}
"""

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    manifest: Dict[str, Any] = {
        "run_id": f"{EXPERIMENT_TYPE}_{ts}_v3",
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": QUEUE_ID,
        "architecture_epoch": ARCHITECTURE_EPOCH,
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "outcome": status,
        "timestamp_utc": ts,
        "evidence_direction": evidence_direction,
        "dry_run": dry_run,
        "metrics": metrics,
        "criteria": criteria,
        "combination_rule": combination_rule,
        "arm_results": arm_rows,
        "non_degenerate": non_degenerate,
        "degeneracy_reason": degeneracy["degeneracy_reason"],
        "interpretation": {
            "label": label,
            "preconditions": preconditions,
            "criteria_non_degenerate": {
                "C1_false_alarm_elevation": non_degenerate,
                "C2_reversion_recovery": non_degenerate,
            },
        },
        "custom_information": {
            "scope_note": (
                "Scoped single-signature probe (hypervigilance only) per "
                "MECH-027's what_would_answer and the source proposal "
                "EXP-0761/EVB-1396 dispatch_mode=targeted_probe. The other four "
                "named pathological labels (dissociation, rumination, mania, "
                "psychosis-like) are out of scope -- 'learning eligibility' has "
                "no confirmed substrate hook and 'hippocampal gating' was not "
                "investigated for this scope."
            ),
            "gov_reuse_1_note": (
                "0/973 manifests tag claim_ids containing MECH-027 as of "
                "authoring time -- no prior run to reuse or supersede."
            ),
            "channel_forcing_note": (
                "All three channels are forced/reverted on the SAME trained "
                "agent per seed (not three separately-trained agents) -- see "
                "module docstring 'HOW EACH CHANNEL IS FORCED'."
            ),
            "mech027_build_note": (
                "Channels (a) and (c) depend on two 2026-09-02 substrate builds "
                "(chip-20260902-mech027-precision-replay-eval-substrate): Build 1 "
                "(E3Config.use_precision_scaled_commit_temperature) gives "
                "current_precision/running_variance a graded downstream consumer "
                "at the committed-selection layer (previously only the saturating "
                "binary ARC-016 commit gate consumed it); Build 2 "
                "(REEAgent.force_sleep_cycle_at_eval_boundary()) makes replay "
                "suppression during eval an exercised lever by actually firing "
                "sleep cycles at eval episode boundaries (previously only warmup "
                "ever fired a sleep cycle, so draws_per_cycle=0 during eval had no "
                "observable effect). See module docstring for both."
            ),
            "red_team_disposition_note": (
                "red-team (fable), 2026-09-02, chip-20260902-mech027-precision-"
                "replay-eval-substrate: CONTESTED overall, 1 BLOCKING (F1, "
                "fixed -- see no_fatal_action_selection_errors readiness "
                "check). F3 (engagement floor too weak) fixed -- see "
                "precision_scaled_commit_temperature_engaged now a fraction "
                "gate. F6 (per-seed recovered_fraction noisy) fixed -- C2 now "
                "computed from pooled rates (pooled_recovered_fraction), see "
                "module docstring PRE-REGISTERED THRESHOLDS. F8 (docstring/"
                "code C1 mismatch) fixed -- docstring corrected to pooled- "
                "only. F2 (does channel (c) reach the DV via any path other "
                "than mech285 draws) VERIFIED AND DISMISSED: the cited "
                "mechanism (SleepLoopManager._run_cycle's "
                "offline_gradient_pass writeback to e2_harm_s) requires "
                "agent.e2_harm_s, which is built only when "
                "config.latent.use_e2_harm_s_forward is True "
                "(ree_core/agent.py:544-551); this script never sets that "
                "flag, so agent.e2_harm_s stays None and that writeback path "
                "never fires here -- confirmed by reading build_config() "
                "(no use_e2_harm_s_forward kwarg) and ree_core/agent.py's "
                "construction gate directly. F4 (fixed BASELINE->HV->"
                "REVERSION block order plus real per-episode sleep cycles in "
                "every block is an order-confound C1 cannot exclude, "
                "distinct from the manipulation) ACKNOWLEDGED, NOT MITIGATED "
                "-- this is the direct, intended consequence of this "
                "adaptation's own mandate (interleave "
                "force_sleep_cycle_at_eval_boundary() across all three "
                "blocks so replay suppression is an exercised lever, not a "
                "dead config knob); a within-eval learning drift confound is "
                "the accepted cost of that design choice. C2's reversion "
                "requirement is the only guard against it for the full-PASS "
                "cell; the C1-and-not-C2 cell's label "
                "(false_alarm_elevation_confirmed_reversion_incomplete, "
                "evidence_direction mixed) already reads as partial/"
                "unconfirmed rather than a clean claim-supporting result, so "
                "it was left unchanged. F5 (EVAL_REVERSION restores "
                "precision_scale=1.0 but not the compounded _running_"
                "variance value HV drove down) ACKNOWLEDGED, NOT MITIGATED "
                "BY DESIGN -- reversion is deliberately measured as NATURAL "
                "recovery once forcing stops (matching the claim's own "
                "phrase 'the channel is returned to its normal range', i.e. "
                "the FORCING is removed, not that internal state is "
                "snapshotted/reset), not an artificial instantaneous reset; "
                "a snapshot-restore would over-intervene and mask whether "
                "the substrate's own dynamics can recover. F7 (SAFE hazard "
                "bin may be structurally empty for a central hazard "
                "placement on this seed set) ACKNOWLEDGED, OUT OF SCOPE for "
                "this adaptation -- hazard/threshold calibration is pre-"
                "existing design from the original draft (see module "
                "docstring THRESHOLD CALIBRATION), not part of the Build 1/2 "
                "adaptation; the hazard_bin_sample_coverage P0 check already "
                "fails the run safely (substrate_not_ready_requeue) if this "
                "occurs, at the cost of burned compute -- flagged for "
                "whoever reviews the real run's dry-run bin coverage."
            ),
        },
        "ethics_preflight": {
            "involves_negative_valence": True,
            "involves_suffering_like_state": False,
            "involves_self_model": False,
            "involves_inescapability_or_helplessness": False,
            "involves_offline_replay_over_harm": True,
            "involves_social_mind_or_language": False,
            "involves_human_data_or_clinical_context": False,
            "decision": "allow",
        },
        "summary_markdown": summary_markdown,
    }

    full_config = {
        "seeds": seeds,
        **config_slice(),
        "thresholds": {
            "FALSE_ALARM_ELEVATION_MULTIPLIER": FALSE_ALARM_ELEVATION_MULTIPLIER,
            "REVERSION_RECOVERY_FLOOR": REVERSION_RECOVERY_FLOOR,
            "PRECISION_RATIO_FLOOR": PRECISION_RATIO_FLOOR,
            "HORIZON_RATIO_CEIL": HORIZON_RATIO_CEIL,
            "POSITIVE_CONTROL_MARGIN": POSITIVE_CONTROL_MARGIN,
            "MIN_BIN_COVERAGE_STEPS": MIN_BIN_COVERAGE_STEPS,
        },
    }

    agents = [per_seed[s]["agent"] for s in seeds]
    zg_stats_list = [per_seed[s]["zg"].stats() for s in seeds]
    # Merge z_goal stream stats across seeds (sum counters; max writer_defect).
    merged_zg = {"ticks_total": 0, "ticks_active": 0, "writer_calls": 0, "n_agents": 0}
    for st in zg_stats_list:
        if not st:
            continue
        for k in ("ticks_total", "ticks_active", "writer_calls", "n_agents"):
            merged_zg[k] = merged_zg.get(k, 0) + int(st.get(k, 0))

    if dry_run:
        print(
            "[smoke] hazard_field_view centre-cell calibration:"
            f" min/max across all logged eval steps -- see per-block prints above.",
            flush=True,
        )

    out_path = write_flat_manifest(
        manifest,
        dry_run=dry_run,
        config=full_config,
        seeds=seeds,
        script_path=Path(__file__),
        started_at=t0,
        agent=agents,
    )
    return {"outcome": manifest["outcome"], "manifest": manifest, "out_path": out_path}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--seeds", type=str, default=None)
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else (
        [SEEDS[0]] if args.dry_run else SEEDS
    )

    result = run_experiment(seeds, args.dry_run)
    out_path = result["out_path"]
    print(f"\nResult written to: {out_path}", flush=True)
    print(f"Status: {result['outcome']}", flush=True)
    for k, v in result["manifest"]["metrics"].items():
        print(f"  {k}: {v}", flush=True)

    raw = str(result["outcome"]).upper()
    return (raw if raw in ("PASS", "FAIL") else "FAIL"), out_path, args.dry_run


if __name__ == "__main__":
    _outcome_raw, _out_path, _dry = main()
    emit_outcome(outcome=_outcome_raw, manifest_path=_out_path, dry_run=_dry)
