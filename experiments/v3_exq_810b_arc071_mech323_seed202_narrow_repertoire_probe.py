"""V3-EXQ-810b -- ARC-071/MECH-323 seed-202 narrow-repertoire diagnostic.

SLEEP DRIVER: not_applicable_no_sleep_call

PURPOSE (diagnostic probe, NOT governance evidence). V3-EXQ-810a's ARM_FULL
readiness result (chunk_accumulator_fires, C1 = 7/8 seeds = 0.875 against a
0.75 bar) is confirmed non-vacuous by /failure-autopsy
(failure_autopsy_V3-EXQ-810a_2026-07-30). That autopsy also flagged a residual:
seed 202 formed ZERO chunks in ARM_FULL despite its own readiness
preconditions holding at wide margin (episode_outcome_spread = 0.3488, 7.0x
the 0.05 floor; harm_step_fraction = 0.0267, 2.7x the 0.01 floor) -- 202 is in
fact the WORST cell on both, yet still comfortably clear. This is not
load-bearing against the aggregate PASS, but it is a reportable single-seed
non-formation the autopsy routed to a future targeted probe rather than noise.

THE PUZZLE THIS RUN RESOLVES. Re-reading V3-EXQ-810a's own manifest
(v3_exq_810a_arc071_chunk_accumulator_readiness_20260728T204535Z_v3), seed 202
and seed 303 have IDENTICAL chunk_acc_n_tracked_sequences (18) -- yet 202
formed 0 chunks and 303 formed 6. So the residual is NOT "202's behavioural
repertoire is smaller" in the crude sense of tracking fewer distinct
sub-sequences: the repertoire SIZE is tied. Two more specific hypotheses are
live, and this run discriminates between them:

  H1 (narrow repertoire, 810's own precedent). V3-EXQ-810's original autopsy
     found silent seeds spent 43/120 and 37/120 episodes as a SINGLE HELD
     action (16/120 on the forming seed) -- i.e. the repertoire is tied in
     COUNT but not in HOW OFTEN any one committed sequence gets exercised. If
     202 is disproportionately single-held-action relative to 303/606, this
     mechanism (same family as 810's) still explains 202's non-formation
     under 810a's corrected substrate.
  H2 (per-sequence gate failure). MECH-323's joint formation condition is
     THREE conditions on each tracked sub-sequence: repetitions >=
     min_repetitions, outcome variance < variance_low, and outcome mean >
     baseline + evaluative_margin (ChunkAccumulator.formation_candidates()).
     If 202's committed-action diversity is comparable to 303's (H1 does not
     hold), the block must be in WHICH of these three gates its 18 tracked
     sequences fail -- e.g. all 18 clear repetitions and variance but none
     clears the outcome-margin gate, meaning 202's task-outcome stream simply
     never rewards a repeated sub-sequence enough, independent of behavioural
     narrowness.

Neither hypothesis is a substrate defect: MECH-323/324 and the credit rule are
already validated (810a PASS). This run's deliverable is knowledge about WHICH
mechanism explains one seed's residual, not a substrate build --
`complex (probe-gated) / puzzle (known rules)` per the work-graph debt
vocabulary.

DESIGN. Single arm (ARM_FULL: MECH-323+324, all-position credit,
use_chunk_proposal_injection=False), at V3-EXQ-810a's EXACT config (72-step
episodes x 120 episodes, full agent loop via StepHarness, num_hazards=2), on
a 3-seed subset:
  202  the target (0 formed, worst readiness margins, tracked_sequences=18)
  303  matched-repertoire-size former (6 formed, tracked_sequences=18 --
       identical to 202; the tightest possible behavioural contrast)
  606  the strongest former (42 formed, 33 crystallised, tracked_sequences=
       465) -- the opposite extreme, for context.
Built from `experiments/_lib/baselines/arc071_chunking.py` (build_env /
build_agent / off_arm_flags -- the lineage's canonical module; see that
module's docstring, which already names this succession
"V3-EXQ-810a -> 810b -> ..."), so this cell is bit-for-bit reproducible
against 810a's own ARM_FULL cells for these seeds: same seed, same RNG reset
(arm_cell), same config_slice, same StepHarness loop. That reproducibility is
this run's own plumbing sanity check (C_repro below), not the scientific
finding.

NEW INSTRUMENTATION (diagnostic-only; reads private ChunkAccumulator state
that the substrate ITSELF already reads internally for its own diagnostics --
policy_chunking.py:1795 (growable ceiling), :1888/:1901 (maintenance
variances), :1951 (dormant-chunk revival), :1971 (replay-origin support) all
read `self.accumulator._tally` / `._episode_actions` directly. No substrate
modification; this script only READS state the accumulator already
maintains.):

  (1) Per-episode snapshot of `accumulator._episode_actions` taken
      immediately BEFORE `agent.note_chunk_outcome(...)` each episode.
      `PolicyChunking.end_episode()` (ree_core/agent.py, the per-episode
      harness boundary) clears `_episode_actions` at the START of the NEXT
      episode, so at this point in the loop the buffer still holds exactly
      the current episode's committed (E3-tick) action-class stream. Used to
      compute `single_held_action_episode_fraction` -- the SAME statistic
      810's original autopsy used (43/120, 37/120, 16/120), replicated on
      810a's corrected substrate for these 3 seeds.
  (2) Final-state per-tracked-sequence gate breakdown via `accumulator._tally`
      (key -> outcome list) and `accumulator._outcome_history` (the running
      baseline). Mirrors `ChunkAccumulator.formation_candidates()`'s own
      three-way test exactly (same statistics, same thresholds read from
      `accumulator.config`), but reports EVERY tracked sequence and which of
      the three conditions it meets, not only the ones meeting all three.
      This is a FINAL-STATE snapshot (the tally's sliding window at run end),
      not a full running history -- a sequence that transiently satisfied all
      three mid-run and later drifted out of the window will not show as
      meeting here even though it may already have been minted; that gap is
      recorded explicitly (`n_meets_all_three_gates_final_snapshot` vs
      `chunk_acc_n_formed`), not papered over.

CLAIM_IDS = ARC-071, MECH-323 (the operator this run's instrumentation reads).
MECH-324 is not tagged: crystallisation lifecycle is not this run's object.
`evidence_direction` is "non_contributory" for both, unconditionally: this run
characterises a behavioural residual on a readiness claim ALREADY PASSED by
810a: whatever H1/H2 verdict it returns, that verdict does not itself move
ARC-071/MECH-323 confidence (governance already has 810a for that). Per the
"Non-standard directions" convention (route non-evidence findings explicitly
rather than letting them default into scored directions).

MECH-094 (SAFETY, audited every run regardless of finding): the waking path is
the only writer here; `chunk_acc_n_replay_formed` and `chunk_lib_n_replay_origin`
must be 0 in every cell (`use_chunk_replay_origin_path=False`, inherited from
the lineage's off_arm_flags()).

GOV-REUSE-1 (Step 2.4). Decisive readouts: `single_held_action_episode_fraction`
per seed, and the per-tracked-sequence gate breakdown. Checked: neither the
810a manifest nor the original 810 manifest records a per-episode action
stream or a per-sequence tally breakdown (both only carry per-cell aggregate
counts -- chunk_acc_n_formed / n_tracked_sequences / n_credit_events -- never
the raw action stream or the tally contents). Not recoverable by
reprocessing an existing manifest -> run.
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
from experiments._harness import StepHarness  # noqa: E402
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

EXPERIMENT_TYPE = "v3_exq_810b_arc071_mech323_seed202_narrow_repertoire_probe"
EXPERIMENT_PURPOSE = "diagnostic"
SUPERSEDES = None  # not a correction of 810a; 810a's PASS stands unchanged.

ANCHOR_REACHABILITY_EXEMPT = (
    "The readiness preconditions reused here are the SAME statistics 810a's own "
    "gate asserts (outcome spread, symbols-per-trial, harm-step fraction), on "
    "the identical config -- reachable by the same construction argument 810a "
    "already made. This run adds no new precondition whose reachability needs "
    "separate justification."
)

CLAIM_IDS = ["ARC-071", "MECH-323"]

ARM_FULL = "ARM_FULL"
ARMS = [ARM_FULL]

# 202 = target (0 formed, worst readiness margins, tracked_sequences=18).
# 303 = matched-repertoire-size former (6 formed, tracked_sequences=18 --
#       identical to 202; the tightest possible contrast).
# 606 = strongest former (42 formed, tracked_sequences=465), for context.
TARGET_SEED = 202
COMPARISON_SEEDS = [303, 606]
SEEDS = [TARGET_SEED] + COMPARISON_SEEDS

N_EPISODES = base.N_EPISODES              # 120
STEPS_PER_EPISODE = base.STEPS_PER_EPISODE  # 72

# --- Pre-registered thresholds. Defined HERE, never inferred post-hoc. -------

# D1: informational, not load-bearing for governance (experiment_purpose is
# diagnostic and evidence_direction is non_contributory regardless of D1's
# outcome). Margin is an absolute fraction-point gap, set conservatively
# below the ~0.2-0.225 gap V3-EXQ-810's own autopsy measured between its
# silent seeds (0.358, 0.308) and its forming seed (0.133).
NARROW_REPERTOIRE_MARGIN = 0.10

# Readiness preconditions -- IDENTICAL statistics and thresholds to 810a's own
# gate (same env, same schedule, same substrate config; see
# ANCHOR_REACHABILITY_EXEMPT above for why no fresh reachability argument is
# needed).
OUTCOME_SPREAD_FLOOR = 0.05
SYMBOLS_PER_TRIAL_FLOOR = float(base.CHUNK_MAX_SIZE - 1)  # 4.0
HARM_STEP_FRACTION_FLOOR = 0.01

PRECONDITIONS = [
    PreconditionSpec(
        name="episode_outcome_spread_supra_floor",
        description=("MECH-323's evaluative gate is RELATIVE to a running "
                     "baseline, so formation is structurally impossible on a "
                     "flat outcome stream. Same statistic 810a's own gate "
                     "asserts (population stdev of the per-episode outcome)."),
        control="worst of the 3 probed seeds on this identical task config",
        threshold=OUTCOME_SPREAD_FLOOR,
        direction="lower",
        kind="readiness",
        applies_to=lambda ctx: True,
        structural_max=lambda ctx: None,
    ),
    PreconditionSpec(
        name="chunk_buffer_supports_size_range",
        description=("note_outcome breaks out at len(actions) < size, so a "
                     "per-trial buffer shorter than chunk_max_size makes the "
                     "size budget structurally unreachable. Same statistic "
                     "810a's own gate asserts (recorded symbols per trial)."),
        control="worst of the 3 probed seeds on this identical task config",
        threshold=SYMBOLS_PER_TRIAL_FLOOR,
        direction="lower",
        kind="readiness",
        applies_to=lambda ctx: True,
        structural_max=lambda ctx: float(STEPS_PER_EPISODE),
    ),
    PreconditionSpec(
        name="salient_harm_events_fire",
        description=("MECH-091's clock.phase_reset() fires only on "
                     "harm_signal < 0, and the full-loop de-periodisation is "
                     "what makes the committed-action stream informative at "
                     "all. Same statistic 810a's own gate asserts (fraction "
                     "of env steps with negative harm_signal)."),
        control="worst of the 3 probed seeds on this identical task config",
        threshold=HARM_STEP_FRACTION_FLOOR,
        direction="lower",
        kind="readiness",
        applies_to=lambda ctx: True,
        structural_max=lambda ctx: 1.0,
    ),
]


def _arm_ctx(arm: str) -> Dict[str, Any]:
    return {"id": arm}


def _arm_full_flags() -> Dict[str, Any]:
    """ARM_FULL's flags -- identical to V3-EXQ-810a's ARM_FULL cell."""
    flags = base.off_arm_flags()
    flags.update({
        "use_policy_chunking": True,
        "use_chunk_maintenance": True,
        "use_chunk_all_position_credit": True,
    })
    return flags


# --- Reproducibility comparator: 810a's own recorded ARM_FULL rows for these
# 3 seeds. Same seed + same config_slice + same RNG reset must reproduce these
# exactly -- a plumbing sanity check, not the scientific finding.
REPRO_SOURCE_RUN_ID = "v3_exq_810a_arc071_chunk_accumulator_readiness_20260728T204535Z_v3"
REPRO_EXPECTED: Dict[int, Dict[str, int]] = {
    202: {"chunk_acc_n_formed": 0, "chunk_lib_n_crystallised": 0,
          "chunk_acc_n_tracked_sequences": 18, "chunk_acc_n_credit_events": 494},
    303: {"chunk_acc_n_formed": 6, "chunk_lib_n_crystallised": 6,
          "chunk_acc_n_tracked_sequences": 18, "chunk_acc_n_credit_events": 767},
    606: {"chunk_acc_n_formed": 42, "chunk_lib_n_crystallised": 33,
          "chunk_acc_n_tracked_sequences": 465, "chunk_acc_n_credit_events": 5664},
}

_ZG = ZGoalStreamAccumulator()


def _tracked_sequence_diagnostics(accumulator: Any) -> Dict[str, Any]:
    """Final-state per-tracked-sequence gate breakdown.

    Mirrors ChunkAccumulator.formation_candidates()'s three-way joint
    condition (reps >= min_repetitions, variance < variance_low, mean >
    baseline + evaluative_margin) using the SAME statistics computed the SAME
    way (statistics.pvariance is the population variance via a two-pass mean,
    matching policy_chunking._variance exactly) -- but reports EVERY tracked
    sequence and which gate(s) block it, not only the ones passing all three.
    """
    c = accumulator.config
    history = list(accumulator._outcome_history)
    baseline = statistics.fmean(history) if history else 0.0
    per_sequence: List[Dict[str, Any]] = []
    n_meets_reps = 0
    n_meets_reps_and_var = 0
    n_meets_all_three = 0
    for key, outcomes in accumulator._tally.items():
        reps = len(outcomes)
        var = statistics.pvariance(outcomes) if reps >= 2 else 0.0
        mu = statistics.fmean(outcomes) if outcomes else 0.0
        meets_reps = reps >= c.min_repetitions
        meets_var = var < c.variance_low
        meets_margin = mu > baseline + c.evaluative_margin
        if meets_reps:
            n_meets_reps += 1
            if meets_var:
                n_meets_reps_and_var += 1
                if meets_margin:
                    n_meets_all_three += 1
        per_sequence.append({
            "length": len(key), "reps": reps, "variance": var, "mean": mu,
            "meets_reps": meets_reps, "meets_variance": meets_var,
            "meets_margin": meets_margin,
        })
    return {
        "n_tracked": len(accumulator._tally),
        "baseline_outcome_mean": baseline,
        "n_meets_reps_gate": n_meets_reps,
        "n_meets_reps_and_variance_gate": n_meets_reps_and_var,
        "n_meets_all_three_gates_final_snapshot": n_meets_all_three,
        "per_sequence": per_sequence,
    }


def _run_cell(seed: int) -> Dict[str, Any]:
    print(f"Seed {seed} Condition {ARM_FULL}", flush=True)
    flags = _arm_full_flags()
    with arm_cell(seed,
                  config_slice=base.arm_config_slice(flags),
                  script_path=Path(__file__),
                  config_slice_declared=True,
                  include_driver_script_in_hash=False) as cell:
        env = base.build_env(seed)
        agent = base.build_agent(env, flags)
        harness = StepHarness(agent, env, train_mode=False, seed=seed)
        accumulator = agent.policy_chunking.accumulator

        episode_outcomes: List[float] = []
        per_episode_symbols: List[int] = []
        episode_action_sequences: List[List[int]] = []
        n_harm_steps = 0
        n_steps_total = 0
        prev_symbols = 0

        for ep in range(N_EPISODES):
            results = harness.run_episode(STEPS_PER_EPISODE)
            ep_reward = 0.0
            for r in results:
                ep_reward += r.harm_signal
                if r.harm_signal < 0:
                    n_harm_steps += 1
            n_steps_total += len(results)

            # Snapshot BEFORE note_chunk_outcome. end_episode() clears
            # _episode_actions at the START of the NEXT episode (the harness's
            # per-episode reset), not here -- so this buffer still holds
            # exactly this episode's committed (E3-tick) action-class stream.
            episode_action_sequences.append(list(accumulator._episode_actions))

            agent.note_chunk_outcome(ep_reward)
            episode_outcomes.append(ep_reward)

            st_now = agent.get_chunking_state()
            symbols_now = int(st_now.get("chunk_acc_n_steps", 0))
            per_episode_symbols.append(symbols_now - prev_symbols)
            prev_symbols = symbols_now

            if (ep + 1) % 30 == 0:
                print(f"  [train] chunk seed={seed} arm={ARM_FULL} ep {ep+1}/{N_EPISODES} "
                      f"formed={st_now.get('chunk_acc_n_formed', 0)} "
                      f"cryst={st_now.get('chunk_lib_n_crystallised', 0)} "
                      f"symbols/ep={per_episode_symbols[-1]}", flush=True)

        st = agent.get_chunking_state()

        # --- H1: single-held-action fraction (replicates 810's own method) --
        n_nonempty = sum(1 for seq in episode_action_sequences if len(seq) > 0)
        n_single_held = sum(
            1 for seq in episode_action_sequences
            if len(seq) > 0 and len(set(seq)) == 1
        )
        single_held_action_episode_fraction = (
            n_single_held / n_nonempty if n_nonempty else 0.0
        )
        n_distinct_actions_used = len(
            {a for seq in episode_action_sequences for a in seq}
        )

        # --- H2: final-state per-tracked-sequence gate breakdown -----------
        seq_diag = _tracked_sequence_diagnostics(accumulator)

        n_symbols = int(st.get("chunk_acc_n_steps", 0))
        n_outcomes = int(st.get("chunk_acc_n_outcomes", 0))
        row = {
            "arm": ARM_FULL,
            "seed": seed,
            # --- reproduced 810a readouts (C_repro target) ---
            "chunk_acc_n_formed": int(st.get("chunk_acc_n_formed", 0)),
            "chunk_lib_n_crystallised": int(st.get("chunk_lib_n_crystallised", 0)),
            "chunk_acc_n_tracked_sequences": int(st.get("chunk_acc_n_tracked_sequences", 0)),
            "chunk_acc_n_credit_events": int(st.get("chunk_acc_n_credit_events", 0)),
            # --- MECH-094 safety audit ---
            "chunk_acc_n_replay_formed": int(st.get("chunk_acc_n_replay_formed", 0)),
            "chunk_lib_n_replay_origin": int(st.get("chunk_lib_n_replay_origin", 0)),
            # --- H1 (single-held-action) ---
            "single_held_action_episode_fraction": single_held_action_episode_fraction,
            "n_single_held_action_episodes": n_single_held,
            "n_nonempty_committed_episodes": n_nonempty,
            "n_distinct_actions_used_in_committed_stream": n_distinct_actions_used,
            # --- H2 (per-sequence gate breakdown) ---
            "tracked_sequence_diagnostics": seq_diag,
            # --- readiness / task-side context (same statistics 810a recorded) ---
            "episode_outcome_spread": (statistics.pstdev(episode_outcomes)
                                       if len(episode_outcomes) > 1 else 0.0),
            "episode_outcome_mean": (statistics.fmean(episode_outcomes)
                                     if episode_outcomes else 0.0),
            "harm_step_fraction": (n_harm_steps / n_steps_total) if n_steps_total else 0.0,
            "symbols_per_trial": (n_symbols / n_outcomes) if n_outcomes else 0.0,
            "per_episode_symbols": per_episode_symbols,
            "per_episode_outcomes": episode_outcomes,
            "n_episodes": len(episode_outcomes),
            "n_harm_steps": n_harm_steps,
            "n_steps_total": n_steps_total,
            "z_goal_stream_stats": harness.z_goal_stream_stats(),
        }
        _ZG.observe_stats(row["z_goal_stream_stats"])
        cell.stamp(row)
    print(f"verdict: {'PASS' if row['chunk_acc_n_replay_formed'] == 0 else 'FAIL'}", flush=True)
    return row


def _worst(rows: List[Dict[str, Any]], key: str):
    if not rows:
        return 0.0, None
    worst = min(rows, key=lambda r: r[key])
    return float(worst[key]), f"seed{worst['seed']}"


def run_experiment() -> Dict[str, Any]:
    assert_no_structurally_unsatisfiable_gate(PRECONDITIONS, [_arm_ctx(a) for a in ARMS])

    rows: List[Dict[str, Any]] = [_run_cell(seed) for seed in SEEDS]
    by_seed = {r["seed"]: r for r in rows}

    # --- Readiness gate (single arm; mirrors 810a's own gate for these cells) --
    spread_w, spread_cell = _worst(rows, "episode_outcome_spread")
    symbols_w, symbols_cell = _worst(rows, "symbols_per_trial")
    harm_w, harm_cell = _worst(rows, "harm_step_fraction")
    arm_gate = evaluate_arm_gate(
        ARM_FULL, _arm_ctx(ARM_FULL), PRECONDITIONS,
        measured={
            "episode_outcome_spread_supra_floor": spread_w,
            "chunk_buffer_supports_size_range": symbols_w,
            "salient_harm_events_fire": harm_w,
        },
    )
    arm_gate["offending_cells"] = {
        "episode_outcome_spread_supra_floor": spread_cell,
        "chunk_buffer_supports_size_range": symbols_cell,
        "salient_harm_events_fire": harm_cell,
    }
    agg = aggregate_arm_gates([arm_gate])
    gate_green = bool(agg["non_degenerate"])

    # --- C_repro: bit-for-bit reproducibility of 810a's ARM_FULL rows -------
    # (only checked for seeds actually run this call -- a --dry-run smoke runs
    # TARGET_SEED alone and cannot reproduce the comparison seeds' rows.)
    repro_detail: Dict[str, Any] = {}
    repro_all_match = True
    for seed, expected in REPRO_EXPECTED.items():
        if seed not in by_seed:
            continue
        got = by_seed[seed]
        diffs = {k: {"expected": v, "got": got[k]} for k, v in expected.items() if got[k] != v}
        matches = not diffs
        repro_all_match = repro_all_match and matches
        repro_detail[str(seed)] = {"matches": matches, "diffs": diffs,
                                   "source_run_id": REPRO_SOURCE_RUN_ID}

    # --- MECH-094 safety (always meaningful, whatever the gate did) --------
    safety_pass = all(
        r["chunk_acc_n_replay_formed"] == 0 and r["chunk_lib_n_replay_origin"] == 0
        for r in rows
    )

    # --- D1 (informational; does not drive outcome or governance) ----------
    # (comparison seeds may be absent under a --dry-run smoke of TARGET_SEED
    # alone -- D1 is then reported as not computable, never as a false result.)
    comparison_present = [s for s in COMPARISON_SEEDS if s in by_seed]
    if comparison_present:
        target_frac = by_seed[TARGET_SEED]["single_held_action_episode_fraction"]
        comparison_fracs = [by_seed[s]["single_held_action_episode_fraction"]
                            for s in comparison_present]
        margin_observed = target_frac - max(comparison_fracs)
        d1_narrow_repertoire_confirmed = margin_observed >= NARROW_REPERTOIRE_MARGIN
        # The comparison is non-degenerate only if the formation asymmetry this
        # run was designed to explain actually reproduced (else there is nothing
        # to explain in THIS run's own data, whatever 810a recorded).
        d1_non_degenerate = all(
            by_seed[TARGET_SEED]["chunk_acc_n_formed"] < by_seed[s]["chunk_acc_n_formed"]
            for s in comparison_present
        )
    else:
        margin_observed = None
        d1_narrow_repertoire_confirmed = False
        d1_non_degenerate = False

    overall_pass = bool(gate_green and repro_all_match and safety_pass)

    if not gate_green:
        label = "substrate_not_ready_requeue"
    elif not repro_all_match:
        label = "reproducibility_mismatch_investigate"
    elif not d1_non_degenerate:
        label = "target_asymmetry_did_not_reproduce"
    elif d1_narrow_repertoire_confirmed:
        label = "narrow_repertoire_confirmed"
    else:
        label = "narrow_repertoire_not_supported"

    metrics = {
        "c_repro_pass": repro_all_match,
        "c_safety_pass": safety_pass,
        "gate_green": gate_green,
        "single_held_action_episode_fraction_by_seed": {
            str(s): by_seed[s]["single_held_action_episode_fraction"] for s in SEEDS
        },
        "target_seed": TARGET_SEED,
        "comparison_seeds": COMPARISON_SEEDS,
        "narrow_repertoire_margin_observed": margin_observed,
        "narrow_repertoire_margin_threshold": NARROW_REPERTOIRE_MARGIN,
        "d1_narrow_repertoire_confirmed": d1_narrow_repertoire_confirmed,
        "n_tracked_sequences_by_seed": {
            str(s): by_seed[s]["chunk_acc_n_tracked_sequences"] for s in SEEDS
        },
        "n_meets_all_three_gates_final_snapshot_by_seed": {
            str(s): by_seed[s]["tracked_sequence_diagnostics"]["n_meets_all_three_gates_final_snapshot"]
            for s in SEEDS
        },
        "chunk_acc_n_formed_by_seed": {
            str(s): by_seed[s]["chunk_acc_n_formed"] for s in SEEDS
        },
    }

    criteria_non_degenerate = arm_criteria_non_degenerate(
        {ARM_FULL: ["C_repro"]}, agg,
        extra={"C_repro": True, "D1_narrow_repertoire_comparison": d1_non_degenerate},
    )
    # C_repro / safety assert an exact/absent value, meaningful whatever the
    # readiness gate did.
    criteria_non_degenerate["C_repro"] = True
    criteria_non_degenerate["C_safety_no_replay_origin"] = True

    interpretation = {
        "label": label,
        "preconditions": agg["adjudication_preconditions"],
        "criteria": [
            {"name": "C_repro", "load_bearing": True, "passed": repro_all_match},
            {"name": "C_safety_no_replay_origin", "load_bearing": True, "passed": safety_pass},
            {"name": "D1_narrow_repertoire_comparison", "load_bearing": False,
             "passed": d1_narrow_repertoire_confirmed},
        ],
        "criteria_non_degenerate": criteria_non_degenerate,
        "per_arm_gate": agg["per_arm_gate"],
        "reproducibility_detail": repro_detail,
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
        SEEDS = [TARGET_SEED]
        N_EPISODES = 4
        STEPS_PER_EPISODE = 72

    result = run_experiment()
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    full_config = {
        "seeds": SEEDS,
        "target_seed": TARGET_SEED,
        "comparison_seeds": COMPARISON_SEEDS,
        "arms": ARMS,
        "n_episodes": N_EPISODES,
        "steps_per_episode": STEPS_PER_EPISODE,
        "env": dict(base.ENV_KWARGS),
        "alpha_world": base.ALPHA_WORLD,
        "chunk_min_repetitions": base.CHUNK_MIN_REPETITIONS,
        "chunk_window_trials": base.CHUNK_WINDOW_TRIALS,
        "chunk_crystallisation_min": base.CHUNK_CRYSTALLISATION_MIN,
        "chunk_min_size": base.CHUNK_MIN_SIZE,
        "chunk_max_size": base.CHUNK_MAX_SIZE,
        "narrow_repertoire_margin": NARROW_REPERTOIRE_MARGIN,
        "outcome_spread_floor": OUTCOME_SPREAD_FLOOR,
        "symbols_per_trial_floor": SYMBOLS_PER_TRIAL_FLOOR,
        "harm_step_fraction_floor": HARM_STEP_FRACTION_FLOOR,
        "arm_config_slice": base.arm_config_slice(_arm_full_flags()),
        "repro_source_run_id": REPRO_SOURCE_RUN_ID,
    }
    manifest = {
        "run_id": f"{EXPERIMENT_TYPE}_{ts}_v3",
        "experiment_type": EXPERIMENT_TYPE,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "supersedes": SUPERSEDES,
        "claim_ids": CLAIM_IDS,
        "evidence_direction": "non_contributory",
        "evidence_direction_per_claim": {
            "ARC-071": "non_contributory",
            "MECH-323": "non_contributory",
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
    print(f"single_held_action_episode_fraction by seed: "
          f"{m['single_held_action_episode_fraction_by_seed']}", flush=True)
    print(f"chunk_acc_n_formed by seed: {m['chunk_acc_n_formed_by_seed']}", flush=True)
    _margin_obs = m['narrow_repertoire_margin_observed']
    _margin_str = f"{_margin_obs:.3f}" if _margin_obs is not None else "n/a"
    print(f"D1 narrow_repertoire_confirmed={m['d1_narrow_repertoire_confirmed']} "
          f"(margin observed {_margin_str} "
          f"vs threshold {m['narrow_repertoire_margin_threshold']:.3f})", flush=True)
    print(f"C_repro={m['c_repro_pass']} C_safety={m['c_safety_pass']} "
          f"gate_green={m['gate_green']}", flush=True)
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
