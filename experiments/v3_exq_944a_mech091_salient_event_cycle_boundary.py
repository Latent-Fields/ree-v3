#!/opt/local/bin/python3
"""
V3-EXQ-944a -- MECH-091: do salient events reset E3's cycle boundary, and does
that eliminate partial-integration artefacts straddling the event?

Claims: MECH-091 (mechanism_hypothesis, candidate, implementation_phase v3)
Chip:   IGW-20260822-232 (GOV-CONFIRM-1: lit_conf 0.82, ZERO experimental evidence)
Supersedes: V3-EXQ-944 (same question, same DV; the gate aggregation was wrong)

experiment_purpose: evidence

WHY THIS RUN EXISTS -- AND WHAT 944 ACTUALLY SHOWED
---------------------------------------------------
V3-EXQ-944 ran 2026-08-22T03:52Z on ree-cloud-2 (substrate 751fc5e33d58, 15 cells,
4303s) and self-routed `substrate_not_ready_requeue` / non_contributory. Reading
its manifest, that self-route is NOT a substrate finding -- it is an artefact of
how the run AGGREGATED its own preconditions. Every load-bearing criterion passed
on EVERY seed:

  seed   ALIGNED    RATE_MATCHED   NO_RESET   C3 delta   rate dev    P3
         straddle   straddle       straddle   (>=0.50)   (<=0.35)
  ----   --------   ------------   --------   --------   --------   ----
    42     0.000        0.826        1.000      0.826      0.914     FAIL
     7     0.000        1.000        1.000      1.000      0.024     ok
    13     0.000        0.945        1.000      0.945      0.039     ok
   100     0.000        0.948        1.000      0.948      0.080     ok
   200     0.000        0.789        1.000      0.789      0.018     ok

C1/C2/C3/C4 all `passed: true`; `criteria_non_degenerate` all true. The ONLY
unmet item was precondition P3 (`rate_match_holds`), on ONE seed, computed as a
`max` OVER SEEDS -- so seed 42 vacated four clean, well-powered seeds and the
whole run was recorded as measuring nothing.

That is precisely the failure the /queue-experiment skill's MULTI-ARM /
MULTI-REGIME GATES rule forbids ("NEVER AND the gate whole-run", 2026-07-19),
derived from failure_autopsy_V3-EXQ-785_2026-07-19.md sections 2a/8: one arm's
unsatisfiable precondition silently vacating another arm's valid result. 944 was
authored before that machinery was applied to this lineage and hand-rolled its
own `min`/`max`-over-seeds aggregation.

WHY SEED 42 FAILED P3 -- A DRIVER DEFECT, NOT A SUBSTRATE GAP
--------------------------------------------------------------
Diagnosed from 944's own `arm_results` (no re-run needed):

  seed 42   mean_episode_length   resets executed / requested
  ALIGNED         22.5                 356 / 356  (100%)
  RATE_MATCHED     8.5                   4 / 236  (1.7%)
  NO_RESET          7.4                   0 / 235  (0%, by construction)

  every other seed: mean_episode_length 47.4 - 72.0, RATE_MATCHED 50% - 83%

RATE_MATCHED defers each suppressed reset to `event_step + U{K..2K}` = U{10..20}
steps later. On seed 42 the episode is ~8 steps, i.e. SHORTER THAN THE MINIMUM
DECOUPLE DELAY -- so essentially every re-issue was scheduled past the end of its
own episode and never fired. RATE_MATCHED collapsed onto NO_RESET, and the rate
match died. Two coding defects made it worse than it had to be, both fixed in
`_lib/baselines/mech091_phase_reset.py` for this run:

  FIX 1 (off-by-one).  `drain()` runs BEFORE `harness.step()`, so the largest
    `clock._global_step` any drain observes across `for _ in range(steps)` is
    `steps - 1`, never `steps`. 944 clamped to `episode_step_budget` (= steps)
    exactly, so EVERY clamped reset was scheduled one step beyond the last drain
    that could see it and could never fire -- reintroducing, inside the clamp,
    the truncation the clamp was added to remove. Now clamps to `steps - 1`.

  FIX 2 (colliding slots).  `real()` only sets the idempotent
    `_pending_phase_reset` flag, so two resets landing on the SAME step yield ONE
    extra E3 tick, not two. 944's clamp collapsed every late request onto one
    step and `drain()` fires at most one per step, so a burst of events near the
    episode tail produced a single tick while the ledger still held the rest
    pending. Each reset now gets its own step; if no free step remains inside the
    episode it is DROPPED and counted (`n_resets_dropped_unfired`), not silently
    stacked. NOTE: a `while`-drain would have been the WRONG fix -- firing N
    resets on one step still yields one tick, so it would have inflated the
    executed COUNT while leaving the RATE unchanged, making P3 look satisfied
    when it was not.

Neither fix can rescue a seed whose episodes are shorter than one E3 cycle: with
K=10 and an 8-step episode there is NO slot at all that is both inside the
episode and outside the event's own cycle. That is a real structural limit, and
this run NAMES it as a precondition (P0 below) instead of letting it resurface
downstream as an unexplained rate-match failure.

WHAT CHANGED, IN FULL (944 -> 944a)
------------------------------------
  1. PER-SEED GATE via `experiments/_lib/precondition_gate.py`. The gate unit is
     the SEED (each seed is a regime: one shared trained snapshot, three clock
     arms). `aggregate_arm_gates` makes non-degeneracy `any seed green`, not
     `all seeds green`, and emits `per_arm_gate` at manifest top level naming the
     green and red seeds. A red seed no longer vacates a green one.
  2. NEW PRECONDITION P0 `episode_admits_cycle_contrast` -- min over the seed's
     three cells of `mean_episode_length` >= 2 * e3_steps_per_tick. An episode
     shorter than two E3 cycles cannot express a cycle-BOUNDARY contrast at all,
     and cannot host a decoupled re-issue. This is the precondition seed 42
     actually violated; 944 had no such check and P3 caught the symptom instead.
  3. Criteria C1-C4 are evaluated over SCORED (green) seeds only, and
     `interpretation.preconditions` carries the green seeds only (the red seeds
     are carried in full under `per_arm_gate.red`). This is mandatory, not
     cosmetic: `build_experiment_indexes._compute_adjudication` reads that list
     FLAT and ARM-BLIND and returns `precondition_unmet` for the WHOLE RUN on the
     first unmet entry -- so including a scoped-out seed's failure there would
     re-create the 944 vacating at adjudication time even with the routing fixed.
  4. Execution accounting per cell (`n_resets_requested`,
     `n_resets_dropped_unfired`, `reset_execution_frac`) so a RATE_MATCHED cell
     degrading toward NO_RESET is readable directly off the manifest rather than
     inferred from a downstream rate deviation.
  5. One extra seed (45) for headroom, so the run still carries several scored
     seeds if one or two fail P0. Seed 44 is deliberately NOT used -- it is a
     documented per-seed instability on this env family (CLAUDE.md; EXQ-539-540,
     V3-EXQ-538a).

Seed 42 is deliberately RETAINED rather than dropped. Excluding it post hoc,
having seen it fail, would be exactly the seed-shopping pre-registration exists
to prevent; retained, it is scoped out by a pre-registered, measured precondition
and its exclusion is visible in the manifest.

RE-DERIVE BRAKE (MOVE-3): RELEASED, not bypassed. Recounted 2026-08-22 with the
run-keyed counter: MECH-091 counts 2 braking autopsies, both targets of
failure_autopsy_grandfathered-r5-batch01-mixed-findings_2026-08-08 (the two
EXQ-133 runs), both `recommended_epistemic_category: precondition_unmet` with
`recommended_substrate_queue_entry.action: "none"`. The named upstream substrate
-- substrate_queue MECH091-SALIENT-EVENT-TRIGGER-WIRING, which wired the task-
completion and commitment-boundary triggers -- is now `status: implemented`
(ree-v3 origin/main 6293b2395248, 2026-08-17), so the brake's own release clause
applies. EXQ-133 measured latent divergence as a PROXY and never touched
MultiRateClock; it is not this design.

GOV-REUSE-1: PARTIALLY recoverable -> minimal targeted re-run. 944's manifest
already carries the decisive readout (`straddle_frac` per arm per seed) on a
compatible substrate, and this design's diagnosis was derived from it WITHOUT a
re-run. What is NOT recoverable from it is a scorable verdict: its combination
rule is pre-registered as `PASS iff (P1 AND P2 AND P3) AND C1 AND C2 AND C3`,
and re-scoring it post hoc by dropping the seed that failed P3 would be exactly
the post-hoc rule change pre-registration forbids. A `substrate_not_ready_requeue`
self-route's own documented route is re-queue at adequate conditions, which is
this run. Checked: v3_exq_944_..._20260822T035234Z_v3 (substrate_hash
ad93e78bb660...), plus the two EXQ-133 runs (different proxy DV, predate
substrate_hash entirely, predate the 2026-08-17 wiring).

THE DV, AND WHY IT IS NOT THE EXQ-131 STALENESS ARTIFACT
--------------------------------------------------------
GFLAG-0037 owed one cheap puzzle-grade check before this lineage could be queued:
confirm the what_would_answer DV -- "no partial-integration artefacts straddling
a salient event" -- is observable on phase 1 and not itself absorbed by the
EXQ-131 E3-output-freeze staleness artifact. It PASSES, and 944's real run is now
the strongest evidence for that: the DV as operationalised here reads NOTHING
from `harm_eval` or any EMA term. It is a pure clock-timing readout -- the
sequence of `e3_tick` flags returned by `clock.advance()` versus the steps at
which `clock.phase_reset()` was requested. An output-freeze artifact in a
downstream evaluator cannot reach it. EXQ-131's artefact is about rate-separation
DISCRIMINABILITY (whether E3's OUTPUT distinguishes conditions); MECH-091 is
about cycle-boundary TIMING (WHEN E3 fires), and 944 measured that timing channel
cleanly across 15 cells.

Operationalisation. A "partial-integration artefact straddling a salient event"
is an E3 planning cycle whose interior CONTAINS the event: the cycle opened
before the event, so its selection was made without the event's information, yet
it governs behaviour for the post-event steps. Two readouts per cell:

  straddle_frac         fraction of salient events whose NEXT E3 tick is more
                        than 1 step away (i.e. a cycle straddles the event)
  mean_integration_lag  mean env steps from the event to the next E3 tick

THE ARMS, AND WHY RATE_MATCHED IS NOT OPTIONAL
----------------------------------------------
  ALIGNED       production: phase_reset() executes at the salient event.
  RATE_MATCHED  the request is SUPPRESSED at the event and re-issued at
                event_step + U{K..2K}, each on its own free step, clamped to the
                last drainable step of the episode: the same number of extra E3
                ticks, decoupled from the event.
  NO_RESET      phase_reset() never executes (what_would_answer's literal
                "no-reset control").

ALIGNED does not only re-align the E3 window, it also RAISES the E3 tick rate
(944 measured 0.12-0.50 vs NO_RESET's 0.013-0.097). An ALIGNED-vs-NO_RESET
contrast alone therefore cannot separate "replans more often" from "replans in
phase with the event" -- and the latter is the whole of MECH-091. RATE_MATCHED
holds the rate roughly fixed and moves only the phase. That is why P3 exists and
why it is kept at the same 0.35 tolerance rather than relaxed: relaxing a
threshold to resolve a failing gate converts a detected artifact into a citable
result, which the skill forbids. The gate is SCOPED (per seed), never lowered.

ISOLATION: one trained substrate per seed, three clock regimes. Each seed runs
ONE P0 prediction-loss warmup; the resulting weights are snapshotted and all
three arms are measured from that identical snapshot on a freshly-seeded env.
This removes the learning-trajectory confound, so the arms differ ONLY in
cycle-boundary alignment. Consequence, declared not hidden: the cells share
mutable state across arms, so every cell is stamped
`extra_ineligible_reasons=["shared_trained_snapshot_across_arms"]` and is NOT
reuse-eligible. That is the correctness guard doing its job, not a skipped mint.

DV-SYMMETRY DECLARATION (one line per arm; the 604c net)
--------------------------------------------------------
The DV, `straddle_frac`, is a function of the relative OFFSETS between the
salient-event step sequence and the E3-tick step sequence. Its symmetry group is
joint time-translation of both sequences (and, for the rate-only reading,
permutation of reset times holding their COUNT fixed).
  ALIGNED       the manipulation shifts the E3 phase counter to the event, i.e.
                it changes exactly the offset the DV reads -- not invariant.
                DECLARED WEAKNESS: for this arm the readout is close to an
                implementation identity (phase_reset() sets the next advance()
                to tick), so C2 below is deliberately NOT the finding. C3 is.
  RATE_MATCHED  invariant under the reset COUNT (it holds the count fixed by
                construction) and NOT invariant under reset TIMING, which is
                the offset the DV reads. This is the arm that carries the
                measurement, and its value is not fixed by any identity.
  NO_RESET      the manipulation removes the event-locked component of the tick
                sequence, leaving the periodic counter -- the offset
                distribution becomes uniform-ish over [1, K]; not invariant.

PRE-REGISTERED PRECONDITIONS (readiness-kind, evaluated PER SEED; a seed below
floor is SCOPED OUT of scoring, and does NOT vacate the other seeds. A run with
NO green seed self-routes substrate_not_ready_requeue, never a substrate verdict)
  P0 min-over-the-seed's-cells mean_episode_length >= 2 * e3_steps_per_tick
     -- an episode shorter than two E3 cycles cannot express a cycle-boundary
     contrast, and cannot host a decoupled re-issue (which must land >= K from
     the event). NEW in 944a; this is what seed 42 actually violated.
  P1 min-over-the-seed's-cells n_salient_events >= 20
  P2 straddle_frac(NO_RESET) >= 0.50 -- the no-reset control must actually
     EXHIBIT partial integration, else nothing is being compared
  P3 |e3_rate(RATE_MATCHED) - e3_rate(ALIGNED)| / e3_rate(ALIGNED) <= 0.35
     -- the rate match must hold, else C3 measures rate and not phase

PRE-REGISTERED ACCEPTANCE CRITERIA (evaluated over SCORED seeds only)
  C1 manipulation check (NOT load-bearing): every scored seed has
     n_resets_executed(ALIGNED) > 0 and n_resets_executed(NO_RESET) == 0
  C2 load-bearing: straddle_frac(ALIGNED) <= 0.05 on every scored seed
  C3 load-bearing: straddle_frac(RATE_MATCHED) - straddle_frac(ALIGNED) >= 0.50
     on every scored seed -- the non-tautological one; see the DV-symmetry block
  C4 corroborating (NOT load-bearing): mean_integration_lag(NO_RESET) -
     mean_integration_lag(ALIGNED) >= 3.0 steps on every scored seed

COMBINATION RULE: PASS iff at least one seed passes its gate AND C1 AND C2 AND C3
hold over the scored seeds. Plain AND. C4 is corroborating and does not gate. C1
is a manipulation check, not a finding.

  PASS         -> evidence_direction "supports": salient events measurably reset
                  E3's cycle boundary and eliminate straddling cycles, and the
                  effect survives holding the replanning RATE fixed.
  C2/C3 fail   -> "weakens". This is what_would_answer's FALSIFYING signature:
                  salient events produce no measurable change in cycle-boundary
                  timing beyond noise. Distinct from the already-ruled-out
                  "no clock exists" reading that closed both EXQ-133 runs.
  no green seed-> "non_contributory" / substrate_not_ready_requeue.

WHAT A PASS DOES AND DOES NOT ESTABLISH (read before weighting this run)
-----------------------------------------------------------------------
A PASS establishes that the salient-event triggers wired on 2026-08-17 DO reset
E3's cycle boundary on the real substrate, that the no-reset control genuinely
exhibits the straddling cycles MECH-091 predicts, and that the difference
survives holding the replanning RATE fixed. It is a STRUCTURAL/TIMING result.

It does NOT establish that the re-alignment improves behaviour. The behavioural
consequence is recorded per cell as `post_event_harm_mean` but is explicitly
NON-GATING: it was noisy and sign-inconsistent across seeds, so pre-registering
it as a criterion would have manufactured a coin flip. A run that PASSes here
should be weighted as "the mechanism does what it says", not as "the mechanism
pays off". The payoff question is separate follow-on work. C2 in particular is
close to an implementation identity (declared above); the informative content is
C3 and the recorded coverage/behavioural diagnostics, not the pass/fail bit.

A SUGGESTIVE, NON-GATING OBSERVATION carried forward from 944 for a future
reader: on seed 42, ALIGNED's mean episode length was 22.5 against NO_RESET's
7.4. That is a ~3x survival difference in the direction MECH-091 predicts, but it
is ONE seed, it is not pre-registered, and it is confounded with the very
episode-length inadequacy P0 exists to exclude. It is recorded in `diagnostics`
as a hypothesis for a successor, and must NOT be cited as evidence from this run.

KNOWN LIMITATION -- TRIGGER COVERAGE IS 2 OF 3, MEASURED AND RECORDED
---------------------------------------------------------------------
`trigger_coverage` per cell counts phase_reset() requests by MECH-091 event
class, attributed at runtime from `ree_core/agent.py`'s own source (so it
survives line drift, and `assert_trigger_wiring()` fails loudly if a substrate
edit ever REMOVES a call site). 944's real run measured, summed over 15 cells:
harm and commit_entry fired abundantly; `completion` fired ZERO times -- the
ARC-028/MECH-105 hippocampal-completion release never fires on this env
(completion signal max 0.0 against a release threshold of 0.75). So this run
exercises 2 of MECH-091's 3 named salient events. That is exactly the
under-testing the substrate entry's own severity_note anticipated and classified
`degrading`, not corrupting -- "a run simply exercises one trigger instead of
three, which is visible in the readout". It is visible here by construction:
`trigger_coverage` and `completion_signal_max` vs `completion_release_threshold`
are recorded on every cell, as a NON-GATING diagnostic. It is deliberately NOT an
`interpretation.preconditions[]` entry, because the indexer reads that list flat
and arm-blind and would flag the WHOLE run `precondition_unmet` on it, burying a
valid two-trigger result. A completion-trigger probe is separate follow-on work.

SUBSTRATE-PATH OVERLAP GATE (Step 2.5c), re-checked 2026-08-22
---------------------------------------------------------------
Two OPEN `corrupting` substrate_queue entries list `ree_core/agent.py`, which
this driver necessarily exercises. Neither defect is reachable here, and both
are held off by config rather than by assumption:
  mode-governance-engagement -- the defect is in the affinity-input clamp in
    ree_core/cingulate/salience_coordinator.py and in the per-seed existential
    gate in experiments/_lib/regime_occupancy_gate.py. This run imports neither
    and leaves use_salience_coordinator at its REEConfig default (False).
  MECH122-CONTENT-PACKAGING-SPINDLE-SELECTION -- scoped by its own author to
    agent.py::run_sws_schema_pass and mel_consumer.py::relative_novelty. This
    run has no sleep loop, so run_sws_schema_pass is never called.
mech203-valence-pool-admissibility overlaps but carries no severity -> no action.
FOUR OPEN `degrading` entries also overlap paths this driver exercises and are
recorded here per the gate rather than blocking: SD-ORIENTING-DECISION-SCALE
(agent.py::select_action), SD-E3-SCORER-COMPLETION (predictors/e3_selector.py),
SD-MECH303-THRESHOLD-SOURCING (agent.py, environment/causal_grid_world.py) and
SD-MECH267-CEM-SELECTION-FIX (hippocampal/module.py, utils/config.py). None
touches the clock-timing readout this DV is computed from (`e3_tick` flags vs
phase_reset request steps); they could in principle shift behaviour and hence
episode lengths and event counts, which is what P0/P1 measure and gate on.

SLEEP DRIVER: none (use_sleep_loop left at its REEConfig default, False).

ethics_preflight:
  involves_negative_valence: false
  involves_suffering_like_state: false
  involves_self_model: false
  involves_inescapability_or_helplessness: false
  involves_offline_replay_over_harm: false
  involves_social_mind_or_language: false
  involves_human_data_or_clinical_context: false
  decision: allow
"""

import argparse
import copy
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from experiment_protocol import emit_outcome  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402
from experiments._lib.arm_fingerprint import arm_cell  # noqa: E402
from experiments._lib.z_goal_stream import ZGoalStreamAccumulator  # noqa: E402
from experiments._metrics import check_degeneracy  # noqa: E402
from experiments._lib.precondition_gate import (  # noqa: E402
    PreconditionSpec,
    evaluate_arm_gate,
    aggregate_arm_gates,
)
from experiments._lib.baselines import mech091_phase_reset as base  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_944a_mech091_salient_event_cycle_boundary"
QUEUE_ID = "V3-EXQ-944a"
CLAIM_IDS: List[str] = ["MECH-091"]
EXPERIMENT_PURPOSE = "evidence"
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"
SUPERSEDES = "v3_exq_944_mech091_salient_event_cycle_boundary"

# Seed 44 is deliberately absent -- documented per-seed instability on this env
# family (CLAUDE.md; EXQ-539-540, V3-EXQ-538a). 45 is the sanctioned substitute.
SEEDS: List[int] = [42, 7, 13, 100, 200, 45]
ARM_NAMES = list(base.ARM_NAMES)

WARMUP_EPISODES = base.WARMUP_EPISODES
MEASURE_EPISODES = base.MEASURE_EPISODES
STEPS_PER_EPISODE = base.STEPS_PER_EPISODE

# --- Pre-registered thresholds. Defined HERE, never inferred post-hoc. -------
# P0 is expressed as a MULTIPLE of the E3 cadence K, which is itself pinned in
# the pre-registered config (heartbeat.e3_steps_per_tick = 10), so the absolute
# floor is deterministic at 20 steps and is recorded as such in the manifest.
P0_MIN_EPISODE_LEN_IN_K = 2.0
P1_MIN_SALIENT_EVENTS = 20          # floor, per cell
P2_NO_RESET_STRADDLE_FLOOR = 0.50   # floor, per seed
P3_RATE_MATCH_TOLERANCE = 0.35      # CEILING on relative e3-rate deviation
C2_ALIGNED_STRADDLE_CEILING = 0.05
C3_STRADDLE_DELTA_FLOOR = 0.50
C4_LAG_DELTA_FLOOR = 3.0

COMBINATION_RULE = (
    "PASS iff at least one seed passes its per-seed precondition gate (P0 AND P1 "
    "AND P2 AND P3) AND C1 AND C2 AND C3 hold over the SCORED (green) seeds. "
    "Plain AND. C4 is corroborating and does not gate. C1 is a manipulation "
    "check, not a finding. A red seed does NOT vacate a green one "
    "(failure_autopsy_V3-EXQ-785_2026-07-19.md sections 2a/8); a run with no "
    "green seed self-routes substrate_not_ready_requeue."
)


def _fmt(x: Optional[float], nd: int = 4) -> Optional[float]:
    return None if x is None else round(float(x), nd)


def _seed_specs(k: int) -> List[PreconditionSpec]:
    """The four per-seed preconditions. The gate unit is the SEED."""
    return [
        PreconditionSpec(
            name="episode_admits_cycle_contrast",
            description=(
                "min over this seed's three cells of mean_episode_length, against "
                "2 * e3_steps_per_tick. An episode shorter than two E3 cycles "
                "cannot express a cycle-BOUNDARY contrast and cannot host a "
                "decoupled re-issue (which must land >= K steps from the event), "
                "so RATE_MATCHED necessarily collapses toward NO_RESET. This is "
                "the precondition V3-EXQ-944 seed 42 actually violated "
                "(mean_episode_length 7.4-8.5 against K=10)."),
            control=(
                "every cell of the seed; the ALIGNED arm is the least likely to "
                "be short, so taking the MIN over arms is the conservative read"),
            threshold=float(P0_MIN_EPISODE_LEN_IN_K) * float(k),
            direction="lower",
        ),
        PreconditionSpec(
            name="salient_events_per_cell_floor",
            description=(
                "min over this seed's cells of n_salient_events -- the DV is "
                "per-event, so a cell below this measured nothing usable"),
            control=(
                "every cell of the seed; the harm trigger fires abundantly on "
                "this hazard-carrying env by construction"),
            threshold=float(P1_MIN_SALIENT_EVENTS),
            direction="lower",
        ),
        PreconditionSpec(
            name="no_reset_control_shows_partial_integration",
            description=(
                "straddle_frac(NO_RESET) for this seed. what_would_answer "
                "requires a no-reset control in which partial integration "
                "MEASURABLY OCCURS; below this floor there is nothing to compare"),
            control=(
                "NO_RESET arm -- phase_reset() never executes, so the E3 tick is "
                "the bare periodic counter and events land at arbitrary phase"),
            threshold=float(P2_NO_RESET_STRADDLE_FLOOR),
            direction="lower",
        ),
        PreconditionSpec(
            name="rate_match_holds",
            description=(
                "|e3_rate(RATE_MATCHED) - e3_rate(ALIGNED)| / e3_rate(ALIGNED) "
                "for this seed. C3 only separates PHASE from RATE while the "
                "rates are matched"),
            control=(
                "RATE_MATCHED re-issues every suppressed reset on its own free "
                "step, so its executed count matches ALIGNED's up to resets with "
                "no free slot left in the episode (counted as "
                "n_resets_dropped_unfired)"),
            threshold=float(P3_RATE_MATCH_TOLERANCE),
            direction="upper",
        ),
    ]


def run(dry_run: bool = False):
    warmup_eps = 2 if dry_run else WARMUP_EPISODES
    measure_eps = 2 if dry_run else MEASURE_EPISODES
    steps = 20 if dry_run else STEPS_PER_EPISODE
    seeds = SEEDS[:2] if dry_run else SEEDS

    sites = base.resolve_trigger_sites()
    wiring = base.assert_trigger_wiring(sites)
    print("MECH-091 trigger wiring (call sites in agent.py): %s" % wiring, flush=True)

    zg = ZGoalStreamAccumulator()
    arm_results: List[Dict[str, Any]] = []
    warmup_lengths: Dict[str, List[int]] = {}
    last_agent = None

    for seed in seeds:
        # ONE warmup per seed, shared by all three arms (isolation -- see docstring).
        env = base.build_env(seed)
        agent = base.build_agent(env)
        torch.manual_seed(seed)
        lens = base.warmup(agent, env, seed, warmup_eps, steps,
                           progress_every=max(1, warmup_eps // 4))
        warmup_lengths[str(seed)] = lens
        snapshot = copy.deepcopy(agent.state_dict())

        for arm in ARM_NAMES:
            print("Seed %d Condition %s" % (seed, arm), flush=True)
            with arm_cell(
                seed,
                config_slice=base.arm_config_slice(arm),
                script_path=Path(__file__),
                config_slice_declared=True,
                # The three arms are measured from ONE per-seed trained snapshot
                # so they differ only in cycle-boundary alignment. That is shared
                # mutable cross-cell state, so the cell is never reuse-eligible.
                extra_ineligible_reasons=["shared_trained_snapshot_across_arms"],
            ) as cell:
                env_arm = base.build_env(seed)
                agent_arm = base.build_agent(env_arm)
                agent_arm.load_state_dict(snapshot)
                row = base.measure_cell(
                    agent_arm, env_arm, arm, seed, measure_eps, steps, sites,
                    zg=zg, progress_every=max(1, measure_eps // 3),
                )
                cell.stamp(row)
            last_agent = agent_arm
            # Per-cell verdict = this cell's own adequacy (did the decisive
            # readout engage at all). The RUN outcome is the cross-arm decision
            # below; a per-cell PASS is not a claim verdict.
            cell_ok = (row["n_salient_events"] >= (1 if dry_run else P1_MIN_SALIENT_EVENTS)
                       and row["straddle_frac"] is not None)
            row["cell_adequate"] = bool(cell_ok)
            arm_results.append(row)
            print("verdict: %s" % ("PASS" if cell_ok else "FAIL"), flush=True)

    by = {(r["arm"], r["seed"]): r for r in arm_results}

    def val(arm: str, seed: int, key: str):
        r = by.get((arm, seed))
        return None if r is None else r.get(key)

    k = int(arm_results[0]["e3_steps_per_tick"]) if arm_results else 10
    specs = _seed_specs(k)
    # Dry-run uses toy episode budgets, so the P0/P1 floors are not meaningful
    # there; relax them so the smoke exercises the GATE MACHINERY rather than
    # tripping on toy scale. Thresholds are untouched on a real run.
    if dry_run:
        specs[0].threshold = 0.0
        specs[1].threshold = 0.0

    # --- PER-SEED GATE. The whole point of 944a: a red seed must not vacate a
    # green one. See failure_autopsy_V3-EXQ-785_2026-07-19.md sections 2a/8.
    seed_gates: List[Dict[str, Any]] = []
    for s in seeds:
        cells = [by[(a, s)] for a in ARM_NAMES if (a, s) in by]
        ep_lens = [c["mean_episode_length"] for c in cells]
        ev_counts = [c["n_salient_events"] for c in cells]
        a_rate = val(base.ARM_ALIGNED, s, "e3_tick_rate")
        m_rate = val(base.ARM_RATE_MATCHED, s, "e3_tick_rate")
        nr_st = val(base.ARM_NO_RESET, s, "straddle_frac")
        rate_dev = (abs(m_rate - a_rate) / a_rate
                    if (a_rate and m_rate is not None and a_rate > 0) else 1.0)
        measured = {
            "episode_admits_cycle_contrast": float(min(ep_lens)) if ep_lens else 0.0,
            "salient_events_per_cell_floor": float(min(ev_counts)) if ev_counts else 0.0,
            "no_reset_control_shows_partial_integration": (
                float(nr_st) if nr_st is not None else 0.0),
            "rate_match_holds": float(rate_dev),
        }
        gate = evaluate_arm_gate(
            arm_id="seed%d" % s,
            arm_ctx={"seed": s, "e3_steps_per_tick": k},
            specs=specs,
            measured=measured,
            # No structural bound is derivable pre-run for any of these -- they
            # are all measured quantities -- so vacuity auto-detection has
            # nothing to act on and is left at its default.
        )
        gate["seed"] = s
        seed_gates.append(gate)

    agg = aggregate_arm_gates(seed_gates)
    scored_seeds = [g["seed"] for g in seed_gates if g["gate_green"]]
    unscored_seeds = [g["seed"] for g in seed_gates if not g["gate_green"]]

    # --- Criteria, over SCORED seeds only.
    c1_seeds, c2_seeds, c3_seeds, c4_seeds = [], [], [], []
    per_seed: List[Dict[str, Any]] = []
    for s in seeds:
        a_ex = val(base.ARM_ALIGNED, s, "n_resets_executed")
        n_ex = val(base.ARM_NO_RESET, s, "n_resets_executed")
        a_st = val(base.ARM_ALIGNED, s, "straddle_frac")
        m_st = val(base.ARM_RATE_MATCHED, s, "straddle_frac")
        n_st = val(base.ARM_NO_RESET, s, "straddle_frac")
        a_lag = val(base.ARM_ALIGNED, s, "mean_integration_lag")
        n_lag = val(base.ARM_NO_RESET, s, "mean_integration_lag")
        c1 = bool(a_ex is not None and a_ex > 0 and n_ex == 0)
        c2 = bool(a_st is not None and a_st <= C2_ALIGNED_STRADDLE_CEILING)
        c3 = bool(a_st is not None and m_st is not None
                  and (m_st - a_st) >= C3_STRADDLE_DELTA_FLOOR)
        c4 = bool(a_lag is not None and n_lag is not None
                  and (n_lag - a_lag) >= C4_LAG_DELTA_FLOOR)
        if s in scored_seeds:
            c1_seeds.append(c1); c2_seeds.append(c2)
            c3_seeds.append(c3); c4_seeds.append(c4)
        per_seed.append({
            "seed": s,
            "scored": s in scored_seeds,
            "gate_failed_preconditions": next(
                (g["failed_preconditions"] for g in seed_gates if g["seed"] == s), []),
            "aligned_straddle_frac": _fmt(a_st),
            "rate_matched_straddle_frac": _fmt(m_st),
            "no_reset_straddle_frac": _fmt(n_st),
            "aligned_mean_lag": _fmt(a_lag),
            "no_reset_mean_lag": _fmt(n_lag),
            "aligned_e3_rate": _fmt(val(base.ARM_ALIGNED, s, "e3_tick_rate")),
            "rate_matched_e3_rate": _fmt(val(base.ARM_RATE_MATCHED, s, "e3_tick_rate")),
            "no_reset_e3_rate": _fmt(val(base.ARM_NO_RESET, s, "e3_tick_rate")),
            "rate_matched_reset_execution_frac": _fmt(
                val(base.ARM_RATE_MATCHED, s, "reset_execution_frac")),
            "rate_matched_resets_dropped_unfired": val(
                base.ARM_RATE_MATCHED, s, "n_resets_dropped_unfired"),
            "min_mean_episode_length_over_arms": _fmt(
                min(by[(a, s)]["mean_episode_length"] for a in ARM_NAMES if (a, s) in by)),
            "straddle_delta_rate_matched_minus_aligned": (
                None if (a_st is None or m_st is None) else _fmt(m_st - a_st)),
            "c1": c1, "c2": c2, "c3": c3, "c4": c4,
        })

    any_scored = bool(scored_seeds)
    c1_pass = bool(any_scored and all(c1_seeds))
    c2_pass = bool(any_scored and all(c2_seeds))
    c3_pass = bool(any_scored and all(c3_seeds))
    c4_pass = bool(any_scored and all(c4_seeds))
    overall_pass = bool(any_scored and c1_pass and c2_pass and c3_pass)

    scored_note = " (over scored seeds %s)" % (scored_seeds or "none")
    criteria = [
        {"name": "C1_manipulation_check", "load_bearing": False, "passed": c1_pass,
         "description": "ALIGNED executes resets and NO_RESET executes none"
                        + scored_note},
        {"name": "C2_aligned_straddle_ceiling", "load_bearing": True, "passed": c2_pass,
         "description": "straddle_frac(ALIGNED) <= %.2f%s"
                        % (C2_ALIGNED_STRADDLE_CEILING, scored_note)},
        {"name": "C3_rate_matched_straddle_delta", "load_bearing": True, "passed": c3_pass,
         "description": "straddle_frac(RATE_MATCHED) - straddle_frac(ALIGNED) >= %.2f%s "
                        "(phase, not rate)" % (C3_STRADDLE_DELTA_FLOOR, scored_note)},
        {"name": "C4_no_reset_lag_delta", "load_bearing": False, "passed": c4_pass,
         "description": "mean_lag(NO_RESET) - mean_lag(ALIGNED) >= %.1f steps%s"
                        % (C4_LAG_DELTA_FLOOR, scored_note)},
    ]

    # Degeneracy. The load-bearing criteria compare ACROSS ARMS, so the
    # collection whose spread decides degeneracy is the pooled cross-arm
    # readout: zero spread there means the manipulation moved nothing and the
    # criteria could not have discriminated. Do NOT hand check_degeneracy the
    # per-seed DELTAS instead -- a constant delta across seeds is a perfectly
    # CONSISTENT effect, and it reports as "zero spread", which would set
    # non_degenerate:false on the strongest possible result and drop it from
    # governance confidence scoring. (Caught in 944's 2026-08-22 dry run, which
    # reported exactly that on a delta of 1.0 on every seed.)
    sc = scored_seeds or seeds
    aligned_st = [v for v in (val(base.ARM_ALIGNED, s, "straddle_frac") for s in sc)
                  if v is not None]
    matched_st = [v for v in (val(base.ARM_RATE_MATCHED, s, "straddle_frac") for s in sc)
                  if v is not None]
    no_reset_st = [v for v in (val(base.ARM_NO_RESET, s, "straddle_frac") for s in sc)
                   if v is not None]
    aligned_lag = [v for v in (val(base.ARM_ALIGNED, s, "mean_integration_lag") for s in sc)
                   if v is not None]
    no_reset_lag = [v for v in (val(base.ARM_NO_RESET, s, "mean_integration_lag") for s in sc)
                    if v is not None]
    deltas = [m - a for a, m in zip(aligned_st, matched_st)]
    degeneracy = check_degeneracy({
        "straddle_frac_across_arms": {"values": aligned_st + matched_st + no_reset_st},
        "mean_integration_lag_across_arms": {"values": aligned_lag + no_reset_lag},
    })
    # The per-seed gate is authoritative for non-degeneracy: a run carrying one
    # clean, well-powered seed is NOT vacuous, whatever another seed did. Let
    # check_degeneracy veto only when it finds genuinely zero cross-arm spread.
    degeneracy["non_degenerate"] = bool(
        agg["non_degenerate"] and degeneracy.get("non_degenerate", True))
    if not agg["non_degenerate"]:
        degeneracy["degeneracy_reason"] = agg["degeneracy_reason"]
    elif agg["degeneracy_reason"]:
        degeneracy["degeneracy_reason"] = (
            (degeneracy.get("degeneracy_reason") or "") + " " + agg["degeneracy_reason"]
        ).strip()

    criteria_non_degenerate = {
        # Keyed to the gate: a criterion evaluated over ZERO scored seeds is
        # vacuous by construction, whatever its boolean says.
        "C1": bool(any_scored and sum(r["n_salient_events"] for r in arm_results) > 0),
        "C2": bool(any_scored and all(
            val(base.ARM_ALIGNED, s, "n_events_with_following_tick") or 0
            for s in scored_seeds)),
        # C3 is vacuous if the two arms are bit-identical in straddle_frac
        # (no spread across the contrast the criterion reads).
        "C3": bool(any_scored and (len(set(_fmt(d) for d in deltas)) > 1
                                   or any(d != 0 for d in deltas))),
        "C4": bool(any_scored and all(
            val(base.ARM_NO_RESET, s, "n_events_with_following_tick") or 0
            for s in scored_seeds)),
    }

    if not any_scored:
        label = "substrate_not_ready_requeue"
        evidence_direction = "non_contributory"
    elif overall_pass:
        label = "mech091_cycle_boundary_reset_confirmed"
        evidence_direction = "supports"
    elif not (c2_pass and c3_pass):
        label = "mech091_cycle_boundary_reset_not_supported"
        evidence_direction = "weakens"
    else:
        label = "mech091_cycle_boundary_reset_mixed"
        evidence_direction = "mixed"

    # Trigger coverage: recorded, NON-GATING. Deliberately not a
    # preconditions[] entry -- see the KNOWN LIMITATION block in the docstring.
    coverage_total = {c: 0 for c in base.TRIGGER_CLASSES}
    for r in arm_results:
        for c in base.TRIGGER_CLASSES:
            coverage_total[c] += r["trigger_coverage"].get(c, 0)
    comp_maxes = [r["completion_signal_max"] for r in arm_results
                  if r["completion_signal_max"] is not None]
    thresholds = [r["completion_release_threshold"] for r in arm_results
                  if r["completion_release_threshold"] is not None]

    metrics = {
        "n_cells": len(arm_results),
        "n_seeds_total": len(seeds),
        "n_seeds_scored": len(scored_seeds),
        "gate_any_green": bool(agg["any_green"]),
        "gate_all_green": bool(agg["all_green"]),
        "c1_pass": c1_pass, "c2_pass": c2_pass, "c3_pass": c3_pass, "c4_pass": c4_pass,
        "overall_pass": overall_pass,
        "mean_straddle_frac_aligned": _fmt(
            sum(aligned_st) / len(aligned_st)) if aligned_st else None,
        "mean_straddle_frac_rate_matched": _fmt(
            sum(matched_st) / len(matched_st)) if matched_st else None,
        "mean_straddle_frac_no_reset": _fmt(
            sum(no_reset_st) / len(no_reset_st)) if no_reset_st else None,
        "mean_straddle_delta": _fmt(sum(deltas) / len(deltas)) if deltas else None,
        "min_reset_execution_frac_rate_matched": _fmt(min(
            [r["reset_execution_frac"] for r in arm_results
             if r["arm"] == base.ARM_RATE_MATCHED
             and r["reset_execution_frac"] is not None] or [0.0])),
    }

    diagnostics = {
        "mech091_trigger_wiring_call_sites": wiring,
        "mech091_trigger_coverage_total": coverage_total,
        "mech091_completion_trigger_fired": coverage_total.get("completion", 0) > 0,
        "completion_signal_max_over_cells": _fmt(max(comp_maxes)) if comp_maxes else None,
        "completion_release_threshold": (thresholds[0] if thresholds else None),
        "completion_coverage_note": (
            "NON-GATING. The ARC-028/MECH-105 hippocampal-completion release did not "
            "fire at all in V3-EXQ-944's real 15-cell run on this env (completion "
            "signal max 0.0 vs release threshold 0.75), so this run is expected to "
            "exercise 2 of MECH-091's 3 named salient events. Recorded here rather "
            "than as a precondition so a valid two-trigger result is not buried "
            "under a whole-run precondition_unmet."),
        "unscored_seeds": unscored_seeds,
        "warmup_episode_lengths_by_seed": warmup_lengths,
        "supersedes_run_note": (
            "Supersedes V3-EXQ-944 (v3_exq_944_..._20260822T035234Z_v3), which "
            "measured the same DV cleanly on all 15 cells but aggregated its "
            "preconditions with a max/min OVER SEEDS, so one seed's failure "
            "vacated four clean seeds and the run self-routed "
            "substrate_not_ready_requeue with every criterion passing."),
        "non_gating_survival_observation": (
            "SUGGESTIVE ONLY, NOT EVIDENCE FROM THIS RUN. In V3-EXQ-944, seed 42's "
            "ALIGNED arm had mean_episode_length 22.5 against NO_RESET's 7.4 -- a "
            "~3x survival difference in the direction MECH-091 predicts. It is one "
            "seed, not pre-registered, and confounded with the very episode-length "
            "inadequacy P0 exists to exclude. Recorded as a hypothesis for a "
            "successor; must not be cited as support."),
        "open_degrading_substrate_entries_on_exercised_paths": [
            "SD-ORIENTING-DECISION-SCALE (agent.py::select_action)",
            "SD-E3-SCORER-COMPLETION (predictors/e3_selector.py)",
            "SD-MECH303-THRESHOLD-SOURCING (agent.py, environment/causal_grid_world.py)",
            "SD-MECH267-CEM-SELECTION-FIX (hippocampal/module.py, utils/config.py)",
        ],
    }

    result: Dict[str, Any] = {
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": QUEUE_ID,
        "claim_ids": CLAIM_IDS,
        "supersedes": SUPERSEDES,
        "status": "PASS" if overall_pass else "FAIL",
        "outcome": "PASS" if overall_pass else "FAIL",
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "evidence_direction": evidence_direction,
        "sleep_driver_pattern": "N/A (no sleep loop)",
        "combination_rule": COMBINATION_RULE,
        "metrics": metrics,
        "per_seed_results": per_seed,
        "arm_results": arm_results,
        "diagnostics": diagnostics,
        "per_arm_gate": agg["per_arm_gate"],
        "interpretation": {
            "label": label,
            # GREEN seeds only -- build_experiment_indexes._compute_adjudication
            # reads this list FLAT and ARM-BLIND and returns precondition_unmet
            # for the WHOLE RUN on the first unmet entry. The red seeds are
            # carried in full at top level under per_arm_gate.red.
            "preconditions": agg["adjudication_preconditions"],
            "preconditions_scope_note": agg["per_arm_gate"]["preconditions_scope_note"],
            "criteria_non_degenerate": criteria_non_degenerate,
            "criteria": criteria,
            "combination_rule": COMBINATION_RULE,
        },
        "fatal_error_count": 0,
    }
    result.update(degeneracy)
    return result, zg, last_agent


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    t0 = time.perf_counter()
    result, zg, last_agent = run(dry_run=args.dry_run)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result["timestamp_utc"] = ts
    result["run_id"] = "%s_%s_v3" % (EXPERIMENT_TYPE, ts)
    result["architecture_epoch"] = ARCHITECTURE_EPOCH

    full_config = {
        "arms": ARM_NAMES,
        "seeds": SEEDS,
        "env_kwargs": dict(base.ENV_KWARGS),
        "warmup_episodes": WARMUP_EPISODES,
        "measure_episodes": MEASURE_EPISODES,
        "steps_per_episode": STEPS_PER_EPISODE,
        "alpha_world": base.ALPHA_WORLD,
        "beta_gate_bistable": base.BETA_GATE_BISTABLE,
        "decouple_k_range": [base.DECOUPLE_MIN_K, base.DECOUPLE_MAX_K],
        "p0_min_episode_len_in_k": P0_MIN_EPISODE_LEN_IN_K,
        "p1_min_salient_events": P1_MIN_SALIENT_EVENTS,
        "p2_no_reset_straddle_floor": P2_NO_RESET_STRADDLE_FLOOR,
        "p3_rate_match_tolerance": P3_RATE_MATCH_TOLERANCE,
        "c2_aligned_straddle_ceiling": C2_ALIGNED_STRADDLE_CEILING,
        "c3_straddle_delta_floor": C3_STRADDLE_DELTA_FLOOR,
        "c4_lag_delta_floor": C4_LAG_DELTA_FLOOR,
        "gate_unit": "seed",
        "dry_run": bool(args.dry_run),
    }

    out_path = write_flat_manifest(
        result,
        dry_run=args.dry_run,
        config=full_config,
        seeds=SEEDS,
        script_path=__file__,
        started_at=t0,
        agent=last_agent,
        z_goal_stream_stats=zg.stats(),
    )

    print("\nResult written to: %s" % out_path, flush=True)
    print("Status: %s  label: %s" % (result["status"], result["interpretation"]["label"]),
          flush=True)
    print("Scored seeds: %s  unscored: %s"
          % (result["metrics"]["n_seeds_scored"],
             result["diagnostics"]["unscored_seeds"]), flush=True)
    print("Trigger coverage: %s" % result["diagnostics"]["mech091_trigger_coverage_total"],
          flush=True)

    if args.dry_run:
        # Smoke assertions (skill Step 3.5): the decisive readout must be
        # non-trivially engaged BEFORE committing to the full seed x arm grid.
        assert all(r["n_salient_events"] > 0 for r in result["arm_results"]), (
            "SMOKE FAIL: some cell recorded zero salient events -- the DV is "
            "per-event, so the full run would measure nothing.")
        assert any(r["straddle_frac"] is not None for r in result["arm_results"]), (
            "SMOKE FAIL: straddle_frac undefined in every cell.")
        # 944a-specific: the per-seed gate machinery must actually be wired --
        # a per_arm_gate block naming every seed, not an empty aggregate.
        assert result["per_arm_gate"]["green_arms"] or result["per_arm_gate"]["red_arms"], (
            "SMOKE FAIL: per-seed gate produced no arms -- the aggregation is "
            "not wired, which is the whole point of this successor.")
        assert len(result["interpretation"]["preconditions"]) > 0, (
            "SMOKE FAIL: no preconditions emitted for adjudication.")
        # The RATE_MATCHED arm must actually differ from NO_RESET in executed
        # resets, else the rate control has silently collapsed (the 944 seed-42
        # signature) and C3 would measure rate rather than phase.
        rm = [r for r in result["arm_results"] if r["arm"] == base.ARM_RATE_MATCHED]
        assert any((r["n_resets_executed"] or 0) > 0 for r in rm), (
            "SMOKE FAIL: RATE_MATCHED executed zero resets in every cell -- the "
            "re-issue path is dead, so the arm has collapsed onto NO_RESET.")
        print("[smoke] decisive-readout engagement OK", flush=True)
        print("[smoke] per-seed gate wired: green=%s red=%s"
              % (result["per_arm_gate"]["green_arms"],
                 result["per_arm_gate"]["red_arms"]), flush=True)

    emit_outcome(
        outcome=result["status"] if result["status"] in ("PASS", "FAIL") else "FAIL",
        manifest_path=out_path,
        dry_run=args.dry_run,
    )
