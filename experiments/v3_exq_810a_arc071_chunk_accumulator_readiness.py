"""V3-EXQ-810a -- ARC-071 / MECH-323 readiness, corrected: does the accumulator fire?

PURPOSE (diagnostic / substrate-readiness validation, NOT governance evidence).
Corrected successor to V3-EXQ-810 (FAIL, C1 form_seed_frac 0.333, label
`chunk_accumulator_silent`). 810's negative was NOT a substrate ceiling: the
accumulator was STARVED, and the starvation was an artefact of the driver and
the task, not of MECH-323. This run repairs all three starvation causes at once
and re-poses the same question with enough seeds to answer it.

SUPERSEDES V3-EXQ-810.

WHAT 810's AUTOPSY ESTABLISHED (measured, not inferred -- do not re-derive it)
-----------------------------------------------------------------------------
`record_step` is reached only on the E3 deliberation path (the non-E3 hold path
returns earlier in `select_action`), and E3 ticks every `e3_steps_per_tick = 10`
env steps. At 810's 24-step episodes that is exactly THREE symbols per episode
(measured buf = 3.00 on every seed), and `note_outcome` breaks out at
`len(actions) < size`. So:
  (i)  chunk sizes 4 and 5 were STRUCTURALLY UNREACHABLE -- `chunk_max_size = 5`,
       the MECH-323 growable ceiling and the ARC-071 growable depth were ALL
       inert at agent level, confirmed by formed lengths {2: 4, 3: 4}.
  (ii) `chunk_min_repetitions` was NEVER the binding constraint: repetition
       reached 37 against a bar of 5 and the variance gate passed too. ONLY the
       evaluative gate blocked. Lowering R_min cannot help and is not attempted.
  (iii) 810's driver ran a REDUCED agent loop that never called `update_residue`,
       and MECH-091's salient-event `clock.phase_reset()` lives INSIDE
       `REEAgent.update_residue` -- so the E3 tick was perfectly periodic, which
       is why buf was exactly 3.00 rather than fluctuating. Compounded by
       `num_hazards = 0`, since `phase_reset()` is reached only on
       `harm_signal < 0`.
  (iv) the residual blocker on the two silent seeds was BEHAVIOURAL: 43/120 and
       37/120 episodes were a SINGLE held action (vs 16/120 on the seed that
       formed), and the run yielded 17-19 distinct commitment tuples against 36.

TWO INTERVENTIONS WERE ALREADY MEASURED AND ARE NOT RE-RUN HERE
---------------------------------------------------------------
  * `use_chunk_all_position_credit=True` (ree-v3 ca2d0d1386) at 810's own
    24-step config: seed 101 formed 7->8 / crystallised 6->7, seeds 202/303
    unchanged at 0. form_seed_frac UNCHANGED at 0.333. Enabling, not sufficient.
  * 72-step episodes + that flag ON (60 eps): buf = 8.00 and formed lengths
    {2:4, 3:3, 4:6, 5:2} -- sizes 4 and 5 DO form, so the growable ceiling and
    depth stop being inert, and the forming seed improved hard (formed 8->15,
    crystallised 7->12). form_seed_frac STILL 0.333.

So neither lever closes C1, and this run does not pretend otherwise. The
UNTESTED lever is (iii)+(iv), and it is this run's PRIMARY MANIPULATION.

THE PRIMARY MANIPULATION: the full agent loop, on a hazard-bearing task
-----------------------------------------------------------------------
Every cell is driven by `experiments/_harness.StepHarness`, whose invariant 3
pins exactly one `agent.update_residue(...)` per env step, on a grid with
`num_hazards = 2`. That conjunction -- and only that conjunction -- lets
MECH-091 fire `clock.phase_reset()`, which forces an E3 tick at the next
`advance()` and de-periodises the deliberation cadence. The prediction is that
the extra, irregularly-spaced symbols broaden the action repertoire enough for
the evaluative gate to clear on more than one seed in three.

IF C1 STILL FAILS UNDER THE FULL LOOP WITH HAZARDS, THAT IS THE FINDING.
It would be a real and reportable result about the agent's behavioural
repertoire -- that the substrate composes policies only in a narrow slice of its
own seed distribution -- and it must be reported as such. No threshold in this
script may be lowered to force a pass; the pre-registered constants below are
the contract.

THE FOUR CORRECTIONS (each maps to an autopsy finding)
------------------------------------------------------
  1. FULL AGENT LOOP + HAZARDS (finding iii/iv)  -- StepHarness, num_hazards=2.
  2. 72-STEP EPISODES x 120 EPISODES (finding i)  -- the 72-step probe that
     measured buf=8.00 ran only 60 episodes, halving the outcome reports on the
     silent seeds; that confound is NOT repeated here.
  3. EIGHT SEEDS (statistical power)              -- C1 in 810 was a 3-seed
     binary vote on a stochastic event: at a true per-seed formation rate of 0.8
     it still fails ~10% of the time, so the old design could not distinguish
     "unreliable" from "broken".
  4. CREDIT-RULE ARM MATRIX (measure, don't assume) -- ARM_FULL_TRAILING vs
     ARM_FULL isolate `use_chunk_all_position_credit` at the corrected episode
     length, where its effect is no longer collapsed by a 3-symbol buffer. Note
     what this buys beyond the flag's own contribution: ARM_FULL_TRAILING is
     810's credit configuration, so its C1 fraction against 810's 0.333 measures
     what corrections 1-2 bought on their own, and ARM_FULL against it measures
     what the flag adds on top. Two nested contrasts, three arms.

INSTRUMENTATION NOTE (a correction to the chip that commissioned this run).
The claim that 810's manifest reported `chunk_acc_n_steps = 0` on rows reporting
`chunk_acc_n_formed = 7` does NOT hold: the landed manifest
(v3_exq_810_arc071_chunk_accumulator_readiness_20260723T222726Z_v3.json) reports
`chunk_acc_n_steps = 360` with `n_outcomes = 120` on exactly those rows -- i.e.
3.00 symbols per trial, which is the very number the starvation diagnosis rests
on. The zeros are the ARM_OFF rows, where 0 steps and 0 formed are both correct.
The readout was never defective, so there is nothing to repair; it is instead
PROMOTED here into a readiness precondition (`chunk_buffer_supports_size_range`)
and a per-episode series, because it is the quantity that decides whether the
size budget is reachable at all.

ARMS (3, same 8 seeds), `use_chunk_proposal_injection=False` in every arm:
  ARM_OFF            chunking off entirely -- the C3 inertness control, and the
                     lineage's reusable OFF-arm mint.
  ARM_FULL_TRAILING  MECH-323+324 with the AS-BUILT trailing-only credit rule.
                     810's exact credit configuration at the corrected episode
                     length, so the difference between its C1 fraction and 810's
                     0.333 is what episode length ALONE bought.
  ARM_FULL           MECH-323+324 with all-position credit. The primary arm; C1
                     is load-bearing on it.

THE ARM THAT IS NOT HERE, AND WHY (a compute trade, stated plainly).
A fourth arm -- ARM_FORM, MECH-323 with maintenance OFF, the registered ARM_1
dissociation in which chunks must form but never crystallise -- was designed and
then cut. Measured on the hub at this exact config, the full agent loop costs
~0.9 s/step (against 0.078 s/step for 810's reduced loop), and the chip's
non-negotiable 8 seeds x 120 episodes x 72 steps is 8640 steps per cell. So each
arm is ~17 hours and the four-arm grid is ~70 hours -- 2.3x the largest job ever
queued, with three days of exposure to a mid-run infra event. Three arms is
~50 hours.
ARM_FORM was the right one to cut, for a reason specific to it: its criterion
(C5, crystallised == 0 and selectable == 0 without maintenance) is a MECH-324
lifecycle property ALREADY pinned by module-level bench contracts, and 810's own
ARM_FORM measurement of it was near-vacuous -- 2 of its 3 seeds formed nothing
at all, so "nothing crystallised" was trivially true there rather than a
dissociation. C5 only becomes informative once formation is known to be
reliable, which is precisely what THIS run establishes. Re-adding ARM_FORM is a
one-line change to ARMS once C1 is known to close.

WHAT THIS RUN DOES *NOT* CLAIM. Behavioural latency and rollout-cost reduction
are ARC-071's headline predictions and are NOT measured here. Those need chunks
in the proposal pool (`use_chunk_proposal_injection`) per MECH-324's registered
three-arm design. Injection is OFF in every arm, so E3 never sees a chunk and
the action stream is unaffected by chunk formation. That is deliberate: a
readiness probe must not perturb the behaviour a later run would measure.
Likewise MECH-324's rapid-reacquisition falsifier (`use_chunk_dissolution_
retention`) stays OFF -- it cannot run until a chunk reliably forms,
crystallises and dissolves in a live run, which is exactly what this run tests.
The MECH-323/ARC-071 growable ceiling and depth also stay OFF (their shared
budget knob `chunk_deliberation_horizon` defaults to 0, so they would be inert
regardless); their effective/derived bounds are RECORDED so a reader can see
which bound binds.

MECH-094 (SAFETY-CRITICAL). The waking path is the only writer, so
`chunk_acc_n_replay_formed` must be 0 in every arm and
`use_chunk_replay_origin_path` is False everywhere. C4 pins it through agent.py;
the bench contracts pin the same property at module level.

DV-SYMMETRY (Step 3 mandatory declaration, per arm):
  ARM_OFF DV = chunk_acc_n_formed, structurally 0 (no operator is constructed).
    Not a measurement and not claimed as one -- it is the inertness control.
  ARM_FULL_TRAILING / ARM_FULL DV = chunk_acc_n_formed and
    chunk_lib_n_crystallised, which are COUNTS over the minted chunk set. The
    symmetry group of a count is permutation of the tallied sub-sequences plus
    any relabelling of action-class indices (a count is a symmetric function of
    its set). The manipulation is NOT invariant under it: instantiating the
    accumulator changes the CARDINALITY of the minted set from 0, and
    cardinality is preserved by no permutation of set members. Maintenance
    on/off likewise changes the cardinality of the CRYSTALLISED subset, not the
    labelling of its members. The trailing-vs-all-position manipulation changes
    WHICH sub-sequences are credited (and hence the tally cardinality), not a
    uniform offset on a shared score -- so it is not a broadcast scalar either.
  No arm's DV is derived from an argmax or any order statistic over candidates:
  with injection OFF the chunking machinery does not touch selection at all, so
  the V3-EXQ-604c broadcast-scalar hazard has no surface here.

GOV-REUSE-1 (Step 2.4). The decisive readout is `chunk_acc_n_formed` per seed
under the FULL agent loop on a hazard-bearing 72-step task. Checked: the only
recorded ARC-071 run in evidence/experiments/ is
v3_exq_810_arc071_chunk_accumulator_readiness_20260723T222726Z_v3, whose cells
are a DIFFERENT manipulation (reduced loop, num_hazards=0, 24-step episodes) on
a substrate_hash predating the credit-rule landing (ree-v3 ca2d0d1386), and
which carries no cell at all for the credit-rule contrast. Not recoverable, and
not partially recoverable -> run.

SLEEP DRIVER: not applicable -- no sleep phase is entered (the MECH-322
carve-out is OFF in every arm, and this run makes no sleep call).
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiment_protocol import emit_outcome  # noqa: E402
from experiments._lib.arm_fingerprint import arm_cell  # noqa: E402
from experiments._lib.baselines import arc071_chunking as base  # noqa: E402
from experiments._lib.precondition_gate import (  # noqa: E402
    PreconditionSpec,
    aggregate_arm_gates,
    arm_criteria_non_degenerate,
    assert_no_structurally_unsatisfiable_gate,
    evaluate_arm_gate,
)
from experiments._lib.z_goal_stream import ZGoalStreamAccumulator  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_810a_arc071_chunk_accumulator_readiness"
EXPERIMENT_PURPOSE = "diagnostic"
SUPERSEDES = "v3_exq_810_arc071_chunk_accumulator_readiness"

ANCHOR_REACHABILITY_EXEMPT = (
    "Both verdict-routing readiness predicates ARE the gates' own definitions, "
    "not hand-written proxies. MECH-323's evaluative gate is DEFINED as 'outcome "
    "mean exceeds the running baseline by a margin', structurally unsatisfiable "
    "exactly when the outcome stream has no spread, and the precondition asserts "
    "that same statistic (pstdev of the per-episode outcome). The size-range "
    "precondition asserts the accumulator's own break condition "
    "(note_outcome breaks at len(actions) < size), measured as recorded symbols "
    "per trial. Both are reachable by construction on any task whose episodes "
    "differ and whose episodes are longer than max_chunk_size * e3_steps_per_tick; "
    "there is no separate control whose reproducibility could fail independently "
    "of the asserted quantities."
)

CLAIM_IDS = ["ARC-071", "MECH-323", "MECH-324"]

# Eight seeds: correction 3. C1 in 810 was a 3-seed binary vote.
SEEDS = [101, 202, 303, 404, 505, 606, 707, 808]

ARM_OFF = "ARM_OFF"
ARM_FULL_TRAILING = "ARM_FULL_TRAILING"
ARM_FULL = "ARM_FULL"
ARMS = [ARM_OFF, ARM_FULL_TRAILING, ARM_FULL]

# The chunking-on arms. ARM_OFF never constructs the operators, so the two
# accumulator-facing preconditions are not meaningful there (disposition (a)).
CHUNKING_ON_ARMS = (ARM_FULL_TRAILING, ARM_FULL)
# The arm carrying the all-position credit rule; C1 is read on it.
CREDIT_ON_ARMS = (ARM_FULL,)

N_EPISODES = base.N_EPISODES              # 120
STEPS_PER_EPISODE = base.STEPS_PER_EPISODE  # 72

# --- Pre-registered thresholds. Defined HERE, never inferred post-hoc. -------
MIN_CHUNKS_FORMED = 1        # C1: a chunking-on arm must mint at least this many
MIN_CRYSTALLISED = 1         # C2: ARM_FULL must crystallise at least this many
SEED_PASS_FRACTION = 0.75    # 6 of 8 seeds

# C6: the size budget must be genuinely exercised. 810 could never exceed length
# 3 (buf = 3.00), so anything above it is the direct behavioural confirmation
# that correction 2 worked. Not load-bearing.
MIN_LONG_CHUNK_LENGTH = 4

# READINESS PRECONDITION 1. The evaluative gate is RELATIVE, so a flat outcome
# stream makes formation structurally impossible regardless of substrate health.
# Asserts the SAME statistic the gate routes on (spread), not a mean or a
# magnitude proxy (the V3-EXQ-643 same-statistic rule).
OUTCOME_SPREAD_FLOOR = 0.05

# READINESS PRECONDITION 2. note_outcome breaks out at len(actions) < size, so a
# buffer shorter than chunk_max_size makes the top of the size budget
# structurally unreachable -- 810's actual failure. Asserts recorded symbols per
# trial (chunk_acc_n_steps / chunk_acc_n_outcomes), the SAME quantity that break
# condition consumes. The floor is CHUNK_MAX_SIZE - 1 because the gate is a
# strict `measured > threshold`: a buffer of exactly chunk_max_size symbols is
# the minimum that makes every permitted size reachable.
SYMBOLS_PER_TRIAL_FLOOR = float(base.CHUNK_MAX_SIZE - 1)  # 4.0

# READINESS PRECONDITION 3. The primary manipulation's own precondition: harm
# events are the SOLE trigger for MECH-091's clock.phase_reset(), so with no
# negative harm_signal the intervention never fired and a C1 failure would say
# nothing about it. Measured as the fraction of env steps with harm_signal < 0.
HARM_STEP_FRACTION_FLOOR = 0.01

# Run-lifetime z_goal liveness across every cell's agent. Each cell builds a
# fresh agent inside base.run_cell, so the counters are fed from the harness's
# own per-cell block rather than by holding 32 agents alive to the end.
_ZG = ZGoalStreamAccumulator()


def _arm_flags(arm: str) -> Dict[str, Any]:
    if arm == ARM_OFF:
        return base.off_arm_flags()
    flags = base.off_arm_flags()
    flags.update({
        "use_policy_chunking": True,
        "use_chunk_maintenance": arm in (ARM_FULL_TRAILING, ARM_FULL),
        "use_chunk_all_position_credit": arm in CREDIT_ON_ARMS,
    })
    return flags


def _config_slice(arm: str) -> Dict[str, Any]:
    return base.arm_config_slice(_arm_flags(arm))


def _arm_ctx(arm: str) -> Dict[str, Any]:
    """Context the precondition specs' applies_to / structural_max read."""
    return {
        "id": arm,
        "chunking_on": arm in CHUNKING_ON_ARMS,
        "steps_per_episode": STEPS_PER_EPISODE,
    }


# --- Precondition specs (regime-conditioned; see precondition_gate.py) -------
PRECONDITIONS = [
    PreconditionSpec(
        name="episode_outcome_spread_supra_floor",
        description=("MECH-323's evaluative gate is RELATIVE to a running "
                     "baseline, so formation is structurally impossible on a "
                     "flat outcome stream. Asserts the SAME statistic the gate "
                     "routes on (population stdev of the per-episode outcome)."),
        control=("worst seed in this arm; the task is a hazard- and "
                 "resource-bearing grid whose per-episode net signal varies"),
        threshold=OUTCOME_SPREAD_FLOOR,
        direction="lower",
        kind="readiness",
        applies_to=lambda ctx: bool(ctx["chunking_on"]),
        applies_note=("ARM_OFF never constructs the accumulator, so there is no "
                      "evaluative gate whose satisfiability this could bound. "
                      "Scoped out rather than failed (disposition (a))."),
        # Unbounded above by config; no design-time proof is derivable.
        structural_max=lambda ctx: None,
    ),
    PreconditionSpec(
        name="chunk_buffer_supports_size_range",
        description=("ChunkAccumulator.note_outcome breaks out at "
                     "len(actions) < size, so a per-trial buffer shorter than "
                     "chunk_max_size makes the top of the size budget "
                     "structurally unreachable -- V3-EXQ-810's actual failure "
                     "(measured 3.00 symbols/trial at 24-step episodes). "
                     "Asserts recorded symbols per trial, the SAME quantity "
                     "that break condition consumes."),
        control=("worst seed in this arm; e3_steps_per_tick = 10 gives ~8 "
                 "periodic symbols in a 72-step episode before any MECH-091 "
                 "phase-reset extras"),
        threshold=SYMBOLS_PER_TRIAL_FLOOR,
        direction="lower",
        kind="readiness",
        applies_to=lambda ctx: bool(ctx["chunking_on"]),
        applies_note=("ARM_OFF never constructs the accumulator, so it records "
                      "no symbols and the buffer length is undefined rather "
                      "than deficient. Scoped out (disposition (a))."),
        # Best attainable: every env step an E3 tick.
        structural_max=lambda ctx: float(ctx["steps_per_episode"]),
    ),
    PreconditionSpec(
        name="salient_harm_events_fire",
        description=("MECH-091's clock.phase_reset() is reached ONLY on "
                     "harm_signal < 0 inside REEAgent.update_residue, so with "
                     "no harm events this run's PRIMARY manipulation (full "
                     "agent loop on a hazard-bearing task, de-periodising the "
                     "E3 tick) never fired and a C1 failure would carry no "
                     "information about it. Asserts the fraction of env steps "
                     "carrying negative harm_signal."),
        control=("worst seed in this arm on an 8x8 grid with num_hazards=2 and "
                 "proxy fields, where proximity to a hazard field already "
                 "yields negative harm_signal"),
        threshold=HARM_STEP_FRACTION_FLOOR,
        direction="lower",
        kind="readiness",
        # Applies to EVERY arm: it is a task property, and it is also what keeps
        # ARM_OFF's gate non-empty (an arm with zero applied preconditions can
        # never be green).
        structural_max=lambda ctx: 1.0,
    ),
]


def _worst(rows: List[Dict[str, Any]], key: str):
    """Minimum of `key` across rows, plus the offending cell id.

    Returns the EXTREMUM, not the mean, so a readiness `met` that quantifies
    over cells recomputes exactly from the reported number and names the
    culprit (the V3-EXQ-779b mean-vs-quantifier rule).
    """
    if not rows:
        return 0.0, None
    worst = min(rows, key=lambda r: r[key])
    return float(worst[key]), f"{worst['arm']}/seed{worst['seed']}"


def _run_cell(arm: str, seed: int) -> Dict[str, Any]:
    print(f"Seed {seed} Condition {arm}", flush=True)
    with arm_cell(seed,
                  config_slice=_config_slice(arm),
                  script_path=Path(__file__),
                  config_slice_declared=True,
                  # MANDATORY for a cross-driver-reusable mint: without this the
                  # driver folds into substrate_hash and a later sibling with
                  # its own driver could never match this lineage's OFF cell.
                  include_driver_script_in_hash=False) as cell:
        raw = base.run_cell(_arm_flags(arm), seed,
                            n_episodes=N_EPISODES,
                            steps_per_episode=STEPS_PER_EPISODE,
                            # Adaptive so the runner's `ep N/M` progress line is
                            # emitted at least twice per cell even under
                            # --dry-run, where N_EPISODES is tiny.
                            progress_every=max(1, min(30, N_EPISODES // 4)),
                            progress_label=f"seed={seed} arm={arm}")
        st = raw["chunking_state"]
        outcomes = raw["episode_outcomes"]
        n_outcomes = int(st.get("chunk_acc_n_outcomes", 0))
        n_symbols = int(st.get("chunk_acc_n_steps", 0))
        lengths = raw["formed_chunk_lengths"]
        row = {
            "arm": arm,
            "seed": seed,
            # --- accumulator (MECH-323)
            "chunk_acc_n_formed": int(st.get("chunk_acc_n_formed", 0)),
            "chunk_acc_n_replay_formed": int(st.get("chunk_acc_n_replay_formed", 0)),
            "chunk_acc_n_simulation_skips": int(st.get("chunk_acc_n_simulation_skips", 0)),
            "chunk_acc_n_replay_refusals": int(st.get("chunk_acc_n_replay_refusals", 0)),
            "chunk_acc_n_steps": n_symbols,
            "chunk_acc_n_outcomes": n_outcomes,
            "chunk_acc_n_tracked_sequences": int(st.get("chunk_acc_n_tracked_sequences", 0)),
            # The corrected credit-rule readouts (ree-v3 ca2d0d1386): these make
            # the trailing-vs-all-position arms separable in the manifest
            # without reading the config back.
            "chunk_acc_n_credit_events": int(st.get("chunk_acc_n_credit_events", 0)),
            "chunk_acc_all_position_credit": bool(
                st.get("chunk_acc_all_position_credit", False)),
            "chunk_acc_last_formed": list(st.get("chunk_acc_last_formed", []) or []),
            # --- size / depth budget: recorded so a reader can see WHICH bound
            # binds. Both growable flags are OFF, so effective should equal the
            # fiat max; that is the point of recording it.
            "chunk_effective_max_chunk_size": st.get("chunk_effective_max_chunk_size"),
            "chunk_n_ceiling_growths": st.get("chunk_n_ceiling_growths"),
            "chunk_effective_max_depth": st.get("chunk_effective_max_depth"),
            "chunk_depth_derived_max": st.get("chunk_depth_derived_max"),
            "chunk_depth_structural_max": st.get("chunk_depth_structural_max"),
            "chunk_n_depth_growths": st.get("chunk_n_depth_growths"),
            "chunk_n_formation_passes": st.get("chunk_n_formation_passes"),
            # --- library (MECH-324)
            "chunk_lib_size": int(st.get("chunk_lib_size", 0)),
            "chunk_lib_n_crystallised": int(st.get("chunk_lib_n_crystallised", 0)),
            "chunk_lib_n_dissolved": int(st.get("chunk_lib_n_dissolved", 0)),
            "chunk_lib_n_selectable": int(st.get("chunk_lib_n_selectable", 0)),
            "chunk_lib_n_dormant": int(st.get("chunk_lib_n_dormant", 0)),
            "chunk_lib_n_reacquisitions": int(st.get("chunk_lib_n_reacquisitions", 0)),
            "chunk_lib_n_replay_origin": int(st.get("chunk_lib_n_replay_origin", 0)),
            "chunk_lib_by_state": st.get("chunk_lib_by_state", {}),
            "chunking_instantiated": bool(raw["chunking_instantiated"]),
            # --- the two structural readouts 810 could not have produced
            "symbols_per_trial": (n_symbols / n_outcomes) if n_outcomes else 0.0,
            "formed_chunk_lengths": lengths,
            "max_formed_chunk_length": (max(int(k) for k in lengths) if lengths else 0),
            "per_episode_symbols": raw["per_episode_symbols"],
            # --- primary-manipulation readout: did MECH-091 have anything to fire on
            "n_harm_steps": raw["n_harm_steps"],
            "n_steps_total": raw["n_steps_total"],
            "harm_step_fraction": ((raw["n_harm_steps"] / raw["n_steps_total"])
                                   if raw["n_steps_total"] else 0.0),
            # --- task-side outcome stream (the evaluative gate's substrate)
            "episode_outcome_mean": (statistics.fmean(outcomes) if outcomes else 0.0),
            "episode_outcome_spread": (statistics.pstdev(outcomes)
                                       if len(outcomes) > 1 else 0.0),
            "episode_outcome_min": min(outcomes) if outcomes else 0.0,
            "episode_outcome_max": max(outcomes) if outcomes else 0.0,
            "per_episode_outcomes": outcomes,
            "n_episodes": len(outcomes),
            "z_goal_stream_stats": raw["z_goal_stream_stats"],
        }
        _ZG.observe_stats(raw["z_goal_stream_stats"])
        cell.stamp(row)
    # One verdict line per (seed x condition) cell. The runner requires the
    # count to equal seeds x conditions.
    cell_ok = (row["chunk_acc_n_formed"] == 0 if arm == ARM_OFF
               else row["chunk_acc_n_formed"] >= MIN_CHUNKS_FORMED)
    print(f"verdict: {'PASS' if cell_ok else 'FAIL'}", flush=True)
    return row


def run_experiment() -> Dict[str, Any]:
    # Design-time refusal: no arm may carry a precondition it provably cannot
    # satisfy from its pre-registered config. Cheap, and it runs before any
    # compute is spent (and again under --dry-run).
    assert_no_structurally_unsatisfiable_gate(
        PRECONDITIONS, [_arm_ctx(a) for a in ARMS])

    rows: List[Dict[str, Any]] = []
    for arm in ARMS:
        for seed in SEEDS:
            rows.append(_run_cell(arm, seed))

    by_arm = {a: [r for r in rows if r["arm"] == a] for a in ARMS}

    def frac(arm: str, pred) -> float:
        cells = by_arm[arm]
        return (sum(1 for r in cells if pred(r)) / len(cells)) if cells else 0.0

    def mean(arm: str, key: str) -> float:
        cells = by_arm[arm]
        return statistics.fmean([r[key] for r in cells]) if cells else 0.0

    # --- Per-arm readiness gate. A red arm must NOT vacate a green one
    # (failure_autopsy_V3-EXQ-785 sections 2a/8), so the gate is evaluated per
    # arm and aggregated with `any green`, never `all green`.
    arm_gates = []
    for arm in ARMS:
        cells = by_arm[arm]
        spread_w, spread_cell = _worst(cells, "episode_outcome_spread")
        symbols_w, symbols_cell = _worst(cells, "symbols_per_trial")
        harm_w, harm_cell = _worst(cells, "harm_step_fraction")
        arm_gates.append(evaluate_arm_gate(
            arm, _arm_ctx(arm), PRECONDITIONS,
            measured={
                "episode_outcome_spread_supra_floor": spread_w,
                "chunk_buffer_supports_size_range": symbols_w,
                "salient_harm_events_fire": harm_w,
            },
        ))
        # Offending-cell ids are carried alongside for the reader; the gate
        # module records measured/threshold/met itself.
        arm_gates[-1]["offending_cells"] = {
            "episode_outcome_spread_supra_floor": spread_cell,
            "chunk_buffer_supports_size_range": symbols_cell,
            "salient_harm_events_fire": harm_cell,
        }
    agg = aggregate_arm_gates(arm_gates)

    # `agg["non_degenerate"]` is `ANY arm green` -- the 785-correct aggregate, and
    # what the manifest records, because a red arm must never vacate a green one.
    # It is NOT sufficient to authorise a C1 VERDICT, though: ARM_OFF carries only
    # the task-side harm precondition, so an all-red set of chunking arms with a
    # green ARM_OFF would still read `any green` and could route a "the
    # accumulator is silent" finding off arms that were never ready. The
    # load-bearing criterion is read on the credit-ON arms, so its own arms'
    # gates -- not the aggregate -- authorise it.
    gate_by_arm = {g["arm"]: bool(g["gate_green"]) for g in arm_gates}
    c1_arms_green = all(gate_by_arm.get(a, False) for a in CREDIT_ON_ARMS)

    # --- Pre-registered criteria.
    # C1 (LOAD-BEARING): the accumulator fires at all in the primary arm.
    c1_full = frac(ARM_FULL, lambda r: r["chunk_acc_n_formed"] >= MIN_CHUNKS_FORMED)
    c1_pass = c1_full >= SEED_PASS_FRACTION

    # C2: MECH-324 crystallises in ARM_FULL.
    c2_pass = frac(ARM_FULL, lambda r: r["chunk_lib_n_crystallised"] >= MIN_CRYSTALLISED
                   ) >= SEED_PASS_FRACTION

    # C3: ARM_OFF is inert -- the operators are not constructed and nothing forms.
    c3_pass = all((not r["chunking_instantiated"]) and r["chunk_acc_n_formed"] == 0
                  for r in by_arm[ARM_OFF])

    # C4 (SAFETY, MECH-094): no chunk from replayed/imagined content, any arm.
    c4_pass = all(r["chunk_acc_n_replay_formed"] == 0
                  and r["chunk_lib_n_replay_origin"] == 0 for r in rows)

    # C6: the size budget is genuinely exercised. 810 could not exceed length 3.
    c6_max_len = max((r["max_formed_chunk_length"] for r in rows), default=0)
    c6_pass = c6_max_len >= MIN_LONG_CHUNK_LENGTH

    # C7: the credit-rule arms are separable -- all-position credit must credit
    # strictly more sub-sequences than trailing-only on the same task. A
    # plumbing assertion: equality would mean the flag is not reaching the
    # accumulator (the from_dims silent-kwarg hazard).
    trailing_credits = mean(ARM_FULL_TRAILING, "chunk_acc_n_credit_events")
    allpos_credits = mean(ARM_FULL, "chunk_acc_n_credit_events")
    c7_pass = allpos_credits > trailing_credits

    overall_pass = bool(c1_arms_green and c1_pass and c2_pass and c3_pass
                        and c4_pass and c6_pass and c7_pass)

    if not c1_arms_green:
        # The load-bearing criterion's own arms were not ready. The ONLY correct
        # route is re-queue at an adequate P0 -- never a substrate-verdict label.
        label = "substrate_not_ready_requeue"
    elif overall_pass:
        label = "chunk_accumulator_fires"
    elif not c1_pass:
        # NOT a substrate-ceiling verdict by fiat: with the gate green, the full
        # agent loop live and hazards firing, a C1 failure is a finding about the
        # agent's behavioural repertoire and routes to autopsy for adjudication.
        label = "chunk_accumulator_silent_under_full_loop"
    elif not c2_pass:
        label = "chunk_formation_without_crystallisation"
    elif not c6_pass:
        label = "chunk_size_range_unreached"
    elif not c7_pass:
        label = "chunk_credit_rule_not_plumbed"
    else:
        label = "chunk_accumulator_partial"

    metrics = {
        "c1_pass": c1_pass, "c2_pass": c2_pass, "c3_pass": c3_pass,
        "c4_pass": c4_pass, "c6_pass": c6_pass, "c7_pass": c7_pass,
        # C1 on both chunking-on arms. The TRAILING figure is not a PASS
        # criterion; it is the credit rule's contribution to C1 itself, measured
        # at the corrected episode length rather than assumed -- and it is the
        # direct comparator for V3-EXQ-810's own 0.333, since ARM_FULL_TRAILING
        # is 810's exact credit configuration at 72-step episodes.
        "c1_full_seed_frac": c1_full,
        "c1_full_trailing_seed_frac": frac(
            ARM_FULL_TRAILING, lambda r: r["chunk_acc_n_formed"] >= MIN_CHUNKS_FORMED),
        "full_n_formed_mean": mean(ARM_FULL, "chunk_acc_n_formed"),
        "full_trailing_n_formed_mean": mean(ARM_FULL_TRAILING, "chunk_acc_n_formed"),
        "full_n_crystallised_mean": mean(ARM_FULL, "chunk_lib_n_crystallised"),
        "full_trailing_n_crystallised_mean": mean(
            ARM_FULL_TRAILING, "chunk_lib_n_crystallised"),
        "full_n_tracked_mean": mean(ARM_FULL, "chunk_acc_n_tracked_sequences"),
        "full_trailing_n_credit_events_mean": trailing_credits,
        "full_n_credit_events_mean": allpos_credits,
        # The structural readouts. 810 measured symbols_per_trial = 3.00 and
        # max_formed_chunk_length = 3; these are the direct comparators.
        "symbols_per_trial_mean_chunking_on": statistics.fmean(
            [r["symbols_per_trial"] for r in rows if r["arm"] in CHUNKING_ON_ARMS]),
        "symbols_per_trial_worst": _worst(
            [r for r in rows if r["arm"] in CHUNKING_ON_ARMS], "symbols_per_trial")[0],
        "symbols_per_trial_worst_cell": _worst(
            [r for r in rows if r["arm"] in CHUNKING_ON_ARMS], "symbols_per_trial")[1],
        "max_formed_chunk_length": c6_max_len,
        "harm_step_fraction_mean": statistics.fmean(
            [r["harm_step_fraction"] for r in rows]),
        "harm_step_fraction_worst": _worst(rows, "harm_step_fraction")[0],
        "harm_step_fraction_worst_cell": _worst(rows, "harm_step_fraction")[1],
        "outcome_spread_worst": _worst(
            [r for r in rows if r["arm"] in CHUNKING_ON_ARMS],
            "episode_outcome_spread")[0],
        "outcome_spread_worst_cell": _worst(
            [r for r in rows if r["arm"] in CHUNKING_ON_ARMS],
            "episode_outcome_spread")[1],
        "n_replay_formed_total": sum(r["chunk_acc_n_replay_formed"] for r in rows),
        "n_seeds": len(SEEDS),
        # The routing basis, recorded so a reader can see WHY the label was
        # chosen without re-deriving the gate: the aggregate is `any arm green`
        # (785-correct), but the C1 verdict is authorised only by its own arms.
        "gate_any_arm_green": bool(agg["non_degenerate"]),
        "gate_c1_arms_green": bool(c1_arms_green),
        "gate_green_by_arm": gate_by_arm,
    }

    criteria_by_arm = {
        ARM_OFF: ["C3_off_arm_inert"],
        ARM_FULL_TRAILING: ["C7_credit_rule_separable"],
        ARM_FULL: ["C1_accumulator_fires", "C2_crystallisation",
                   "C6_size_range_exercised", "C4_no_replay_origin_chunks"],
    }
    criteria_non_degenerate = arm_criteria_non_degenerate(
        criteria_by_arm, agg,
        # C1 is read on BOTH credit-ON arms, so ARM_FULL's gate alone does not
        # establish its non-degeneracy; C7 is a two-arm contrast for the same
        # reason. Both need every arm they read to be green.
        extra={
            "C1_accumulator_fires": c1_arms_green,
            "C7_credit_rule_separable": (gate_by_arm.get(ARM_FULL_TRAILING, False)
                                         and gate_by_arm.get(ARM_FULL, False)),
        })
    # C3/C4 assert the ABSENCE of something that must never appear, so they are
    # meaningful whatever the readiness gate did.
    criteria_non_degenerate["C3_off_arm_inert"] = True
    criteria_non_degenerate["C4_no_replay_origin_chunks"] = True

    interpretation = {
        "label": label,
        "preconditions": agg["adjudication_preconditions"],
        "criteria": [
            {"name": "C1_accumulator_fires", "load_bearing": True, "passed": c1_pass},
            {"name": "C2_crystallisation", "load_bearing": False, "passed": c2_pass},
            {"name": "C3_off_arm_inert", "load_bearing": False, "passed": c3_pass},
            {"name": "C4_no_replay_origin_chunks", "load_bearing": False,
             "passed": c4_pass},
            {"name": "C6_size_range_exercised", "load_bearing": False, "passed": c6_pass},
            {"name": "C7_credit_rule_separable", "load_bearing": False,
             "passed": c7_pass},
        ],
        "criteria_non_degenerate": criteria_non_degenerate,
        "per_arm_gate": agg["per_arm_gate"],
        "gate_offending_cells": {g["arm"]: g["offending_cells"] for g in arm_gates},
    }

    return {
        "outcome": "PASS" if overall_pass else "FAIL",
        "metrics": metrics,
        "per_seed_rows": rows,
        "arm_results": rows,
        "interpretation": interpretation,
        "per_arm_gate": agg["per_arm_gate"],
        "non_degenerate": bool(agg["non_degenerate"]),
        "degeneracy_reason": agg["degeneracy_reason"] or None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    t0 = time.perf_counter()

    global SEEDS, N_EPISODES, STEPS_PER_EPISODE
    if args.dry_run:
        SEEDS = [101]
        N_EPISODES = 4
        STEPS_PER_EPISODE = 72

    result = run_experiment()
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    full_config = {
        "seeds": SEEDS, "arms": ARMS,
        "n_episodes": N_EPISODES,
        "steps_per_episode": STEPS_PER_EPISODE,
        "env": dict(base.ENV_KWARGS),
        "alpha_world": base.ALPHA_WORLD,
        "chunk_min_repetitions": base.CHUNK_MIN_REPETITIONS,
        "chunk_window_trials": base.CHUNK_WINDOW_TRIALS,
        "chunk_crystallisation_min": base.CHUNK_CRYSTALLISATION_MIN,
        "chunk_min_size": base.CHUNK_MIN_SIZE,
        "chunk_max_size": base.CHUNK_MAX_SIZE,
        "min_chunks_formed": MIN_CHUNKS_FORMED,
        "min_crystallised": MIN_CRYSTALLISED,
        "min_long_chunk_length": MIN_LONG_CHUNK_LENGTH,
        "seed_pass_fraction": SEED_PASS_FRACTION,
        "outcome_spread_floor": OUTCOME_SPREAD_FLOOR,
        "symbols_per_trial_floor": SYMBOLS_PER_TRIAL_FLOOR,
        "harm_step_fraction_floor": HARM_STEP_FRACTION_FLOOR,
        "arm_config_slices": {a: _config_slice(a) for a in ARMS},
    }
    manifest = {
        "run_id": f"{EXPERIMENT_TYPE}_{ts}_v3",
        "experiment_type": EXPERIMENT_TYPE,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "supersedes": SUPERSEDES,
        "claim_ids": CLAIM_IDS,
        "evidence_direction": "supports" if result["outcome"] == "PASS" else "unknown",
        "evidence_direction_per_claim": {
            "ARC-071": "supports" if result["outcome"] == "PASS" else "unknown",
            "MECH-323": "supports" if result["metrics"]["c1_pass"] else "unknown",
            "MECH-324": "supports" if result["metrics"]["c2_pass"] else "unknown",
        },
        "outcome": result["outcome"],
        "timestamp_utc": ts,
        "metrics": result["metrics"],
        "per_seed_rows": result["per_seed_rows"],
        "arm_results": result["arm_results"],
        "interpretation": result["interpretation"],
        "per_arm_gate": result["per_arm_gate"],
        "non_degenerate": result["non_degenerate"],
        "degeneracy_reason": result["degeneracy_reason"],
        "sleep_driver_pattern": "not_applicable_no_sleep_call",
    }
    out_path = write_flat_manifest(
        manifest,
        dry_run=args.dry_run,
        config=full_config,
        seeds=SEEDS,
        script_path=Path(__file__),
        started_at=t0,
        z_goal_stream_stats=_ZG.stats(),
    )
    m = result["metrics"]
    print(f"outcome: {result['outcome']}", flush=True)
    print(f"label: {result['interpretation']['label']}", flush=True)
    print(f"C1 seed_frac: FULL={m['c1_full_seed_frac']:.3f} "
          f"FULL_TRAILING={m['c1_full_trailing_seed_frac']:.3f} "
          f"(V3-EXQ-810 was 0.333)", flush=True)
    print(f"formed mean: FULL={m['full_n_formed_mean']:.2f} "
          f"FULL_TRAILING={m['full_trailing_n_formed_mean']:.2f} "
          f"| cryst FULL={m['full_n_crystallised_mean']:.2f}", flush=True)
    print(f"structural: symbols/trial={m['symbols_per_trial_mean_chunking_on']:.2f} "
          f"(worst {m['symbols_per_trial_worst']:.2f} "
          f"@ {m['symbols_per_trial_worst_cell']}) "
          f"max_chunk_len={m['max_formed_chunk_length']} "
          f"harm_step_frac={m['harm_step_fraction_mean']:.4f}", flush=True)
    print(f"credit events: TRAILING={m['full_trailing_n_credit_events_mean']:.1f} "
          f"ALL_POSITION={m['full_n_credit_events_mean']:.1f}", flush=True)
    print(f"C1={m['c1_pass']} C2={m['c2_pass']} C3={m['c3_pass']} C4={m['c4_pass']} "
          f"C6={m['c6_pass']} C7={m['c7_pass']}", flush=True)
    print(f"gate green arms: {result['per_arm_gate']['green_arms']} "
          f"red: {result['per_arm_gate']['red_arms']}", flush=True)
    print(f"wrote: {out_path}", flush=True)
    return result, out_path, args.dry_run


if __name__ == "__main__":
    _result, _out_path, _dry_run = main()
    _outcome_raw = str(_result["outcome"]).upper()
    emit_outcome(
        outcome=_outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL",
        manifest_path=str(_out_path),
        dry_run=_dry_run,
    )
