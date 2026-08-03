"""V3-EXQ-873a -- MECH-322 sleep-replay carve-out: does the AND-gate actually fire?
(supersedes V3-EXQ-873 -- corrected readiness gate, see FRACTION GATE section below.)

SLEEP DRIVER: manual-cycle-loop (single wake-sleep-wake cycle, N_CYCLES=1;
enter_sws_mode()/exit_sleep_mode() called directly once per cell, at the
SLEEP_CHECKPOINT_EPISODE boundary; no SD-017 schema-installation or REM
slot-fill passes are run -- only the MECH-322 phase-gate condition is
exercised, which is genuinely orthogonal to schema/slot-fill content).

PURPOSE (diagnostic / substrate-readiness validation, NOT a behavioural claim).
Identical scientific question to V3-EXQ-873; the ONLY change is the test-design
fix routed by confirmed failure_autopsy_V3-EXQ-873_2026-08-03 (this run supersedes
873). Zero genuine experimental evidence exists for MECH-322 (backlog EVB-0489,
claims.yaml 2026-05-11 registration). `ree-v3/ree_core/policy/policy_chunking.py`
implements the carve-out by name (`ChunkAccumulator.record_replay_sequence`,
`PolicyChunking.note_replay_sequence`, `REEAgent.note_chunk_replay_sequence`),
default-OFF (`use_chunk_replay_origin_path=False`) even when chunking itself is
on -- SAFETY-CRITICAL, since a hallucinated chunk installs a macro the agent
never executed into the atomic-commit pool (MECH-094). V3-EXQ-873 was the FIRST
run to flip that flag: in 2/3 seeds the AND-gate minted a replay_origin=True
chunk exactly as specified, and every fail-closed branch (wake-refusal,
master-switch-off, low-value) passed cleanly across all 9 cells. It read FAIL
only because of the test-design defect this iteration corrects.

FRACTION GATE (the V3-EXQ-873 fix -- decouple STOCHASTIC readiness from
DETERMINISTIC plumbing). 873's load-bearing C1 read FAIL solely because its
`replay_high_value_candidate_available` readiness precondition was aggregated
WORST-OF-N-SEED (min candidate margin across seeds via a local _worst() helper):
one seed in which no qualifying real candidate happened to emerge by the sleep
checkpoint drove the whole ARM_REPLAY_ON gate RED (label
substrate_not_ready_requeue), burying the 2/3 seeds that minted correctly. AND
873's C1 itself used SEED_PASS_FRACTION=1.0 over ALL seeds, so a seed that was
structurally unattemptable (no candidate) counted as a mint failure. That
conflates two different things:
  (i)  STOCHASTIC readiness -- does a qualifying real high-value candidate
       NATURALLY emerge by the checkpoint on this seed? This is a genuinely
       random per-seed event (MECH-323 formation dynamics + the value quantile),
       and it is CORRECT for some seeds not to produce one.
  (ii) DETERMINISTIC plumbing -- GIVEN a qualifying candidate + sleep phase +
       flag ON, does record_replay_sequence's AND-gate mint the chunk? This is
       a near-deterministic function call, and it MUST succeed on every seed
       where (i) held.
The corrected design measures each separately:
  * Readiness gate (arm-level): the FRACTION of seeds producing a qualifying
    candidate must EXCEED READINESS_SEED_FRACTION_FLOOR (measured = n_cleared /
    n_seeds; a floor, indexer-recomputable). This is the adjudicability check --
    it goes RED only when candidates rarely emerge (a real substrate/schedule
    problem), never because ONE unlucky seed missed.
  * C1 (load-bearing plumbing): among the CLEARED seeds only (those that
    produced a qualifying candidate), the fraction that minted a
    replay_origin=True chunk must be >= C1_MINT_FRACTION (1.0 -- deterministic).
    A seed that did NOT clear readiness is EXCLUDED from C1's denominator, never
    counted as a mint failure.
Seed count is raised 3 -> 8 for adequate power on the fraction statistic (the
readiness base rate was ~2/3 in 873; with 8 seeds, P(>=4 clear) is high, so the
run is robust to a couple of unlucky draws). spread/buffer/harm readiness
preconditions KEEP their worst-of-N aggregation -- those are near-deterministic
structural task-property checks that legitimately should hold on every seed
(810a confirmed them 8/8 at this env/schedule); only the candidate-availability
precondition was the stochastic-vs-deterministic conflation the autopsy named.

MECH-322 requires THREE conditions, ALL ANDed, EACH failing closed:
    (a) value_tag from prior REAL executions >= replay_value_threshold()
        (top quartile of the real-execution outcome distribution)
    (b) designated SD-017 SLEEP phase (waking DMN stays MECH-094-strict)
    (c) replay_origin=True audit flag + an ACCELERATED corroboration deadline
        (chunk_replay_corroboration_episodes waking episodes without a real
        re-execution -> retired directly to DISSOLVED)
This experiment builds a genuine real-execution history via the full agent
loop (StepHarness, ARC-071/MECH-323/324 substrate -- confirmed reliably
firing by V3-EXQ-810a, PASS, label chunk_accumulator_fires), then at a single
designated checkpoint episode tests each condition INDEPENDENTLY by attempting
the entry point under a controlled variation of exactly one condition at a
time, using the SAME real tracked sub-sequence + its genuinely measured
outcome mean wherever the docstring's "value from prior real executions"
requirement is in play:

  PROBE 1 (condition b, WAKE):   attempt with the real high-value sequence
                                  while genuinely awake (serotonin.phase=="wake")
                                  -> MUST be refused.
  PROBE 2 (conditions a+c, SWS): same sequence, now genuinely asleep
                                  (agent.enter_sws_mode(); serotonin.phase==
                                  "sws") -> MUST mint a replay_origin=True
                                  chunk.
  PROBE 3 (condition a, SWS):    a DISTINCT tracked-or-synthetic sequence,
                                  same sleep phase, but a value_tag DELIBERATELY
                                  set below replay_value_threshold() -> MUST be
                                  refused. This is a boundary probe of the
                                  API's own comparison logic, not a claim that
                                  this second sequence carries that value in
                                  reality (documented per the diagnostic
                                  descriptions convention).

A parallel ARM_STRICT_FLAG_OFF cell runs the identical three probes with
`use_chunk_replay_origin_path=False` (chunking otherwise ON) to confirm the
MASTER SWITCH alone refuses regardless of phase or value -- the shipped
default's own safety claim ("OFF by default even when chunking is on").
ARM_OFF (chunking off entirely) is the C5 inertness control, built from the
canonical `experiments/_lib/baselines/arc071_chunking.py` lineage module
(V3-EXQ-810a/810b), so its OFF-arm cell is fingerprint-identical to that
lineage's mint and reuse-eligible for any future sibling.

GOV-REUSE-1 (Step 2.4). Decisive readout: the FRACTION of candidate-bearing
seeds in which `note_chunk_replay_sequence` mints under a genuine
real-value-tagged sleep-phase attempt (chunk_acc_n_replay_formed /
chunk_lib_n_replay_origin), and whether each of the three failure variants is
correctly refused. V3-EXQ-873 (the only run that ever flipped the flag ON)
recorded these per-seed but at n=3 with the buggy worst-of-N aggregation, which
is INSUFFICIENT POWER for the fraction statistic this iteration adjudicates --
the autopsy's reanalysis of 873's own manifest (2/2 valid-precondition seeds
minted) is exactly the partial-recovery signal, but the readiness-fraction
question needs the larger n. Not recoverable at adequate power -> run. All 9
prior manifests carrying chunk_acc_n_replay_formed ran with the flag OFF (audit
convention), so none exercises the ON path; only 873 does, at n=3. Re-derive
brake (Step 2.5b): 0 prior failure_autopsy targets tag MECH-322 with a
substrate_ceiling disposition (873's autopsy is measurement_test_design_defect,
recommended_substrate_queue_entry action=none -- instrument repair, owes no
build; brake count 0). Not braked -> run.

Substrate readiness (Step 2.5 + 2.5a). `use_chunk_replay_origin_path` --
confirmed read at `ree_core/agent.py` into `PolicyChunkingConfig`, and
`chunk_replay_value_quantile` / `chunk_replay_corroboration_episodes` alongside.
Entry point confirmed wired end-to-end: `REEAgent.note_chunk_replay_sequence`
-> `PolicyChunking.note_replay_sequence` -> `ChunkAccumulator.
record_replay_sequence`, which checks the flag, then `in_sleep_phase`, then
size, then the value bar, in that order, each branch incrementing
`_n_replay_refusals` and returning None. The default `in_sleep_phase=None` path
reads the REAL substrate authority (`self.serotonin.phase`), not a caller
override -- this run deliberately uses that default path so it tests the actual
SD-017 wiring, not a bypass. EMPIRICAL confirmation is V3-EXQ-873 itself: it
minted in 2/3 seeds, so the ON path is reachable and behaves as specified.
MECH-323/324 substrate (chunk formation + crystallisation + dissolution) is
independently confirmed live by V3-EXQ-810a (PASS, 2026-07-30) at this exact
env/schedule -- reused verbatim here.

DV-SYMMETRY (Step 3 mandatory declaration). ARM_OFF's DV is
`chunk_acc_n_formed`, structurally 0 (inertness control, no operator
constructed) -- identical reasoning to V3-EXQ-810a. ARM_STRICT_FLAG_OFF's and
ARM_REPLAY_ON's load-bearing DVs (`wake_refusal_minted`, `success_minted`,
`success_minted_replay_origin`, `low_value_refusal_minted`) are each a direct
boolean readout of ONE deterministic function call
(`note_chunk_replay_sequence(sequence, value_tag, in_sleep_phase=None)`)
against ONE specific pre-selected (sequence, value_tag, phase) tuple -- there
is no candidate SET being selected among, so none of the three hazard classes
apply: it is not an argmax/softmax-sample over candidates (no broadcast-scalar
hazard), not a rank/order statistic (no monotone-rescaling hazard), and not a
set-aggregate over interchangeable units (no permutation hazard). The
readiness-fraction statistic aggregated over seeds is a COUNT of independent
per-seed emergence events, not a manipulation invariant under any symmetry of a
DV. The corroboration-deadline readout is a categorical/count state of the ONE
tracked chunk object created by PROBE 2, likewise not an aggregate. No symmetry
group renders any of these manipulations invisible to their own DV.

Ethics preflight (Step 2.6, all-false/allow -- standard V3 boilerplate):
    involves_negative_valence: false
    involves_suffering_like_state: false
    involves_self_model: false
    involves_inescapability_or_helplessness: false
    involves_offline_replay_over_harm: false
    involves_social_mind_or_language: false
    involves_human_data_or_clinical_context: false
    decision: allow
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_873a_mech322_sleep_replay_carveout_fraction_gate"
EXPERIMENT_PURPOSE = "diagnostic"

CLAIM_IDS = ["MECH-322"]

BACKLOG_ID = "EVB-0489"

# The prior iteration this supersedes (run_id / experiment_type stem).
SUPERSEDES = "v3_exq_873_mech322_sleep_replay_carveout"

# 8 seeds (873 used 3). Rationale: this iteration adjudicates on a FRACTION
# statistic (what share of seeds naturally produce a qualifying real high-value
# candidate by the sleep checkpoint), and 873's ~2/3 base rate at n=3 gave no
# power to distinguish "candidates rarely emerge" (a real substrate problem)
# from "one unlucky seed". At n=8 with READINESS_SEED_FRACTION_FLOOR=0.4 (>=4
# seeds), P(>=4 clear | p~0.67) is high, so the readiness gate is robust and the
# deterministic C1 (over cleared seeds only) is well-powered. Not a reef config,
# so the seed-44 instability guidance does not apply; distinct values chosen.
SEEDS = [101, 202, 303, 404, 505, 606, 707, 808]

ARM_OFF = "ARM_OFF"
ARM_STRICT_FLAG_OFF = "ARM_STRICT_FLAG_OFF"
ARM_REPLAY_ON = "ARM_REPLAY_ON"
ARMS = [ARM_OFF, ARM_STRICT_FLAG_OFF, ARM_REPLAY_ON]
CHUNKING_ON_ARMS = (ARM_STRICT_FLAG_OFF, ARM_REPLAY_ON)

N_EPISODES = base.N_EPISODES              # 120, same as V3-EXQ-810a
STEPS_PER_EPISODE = base.STEPS_PER_EPISODE  # 72, same as V3-EXQ-810a
# Episode index (0-based) at which the single sleep checkpoint runs, i.e.
# after this many waking episodes have completed. Chosen late enough
# (75% through) that MECH-323 formation has every opportunity to have
# produced a qualifying real candidate, matching the schedule V3-EXQ-810a
# already confirmed reliable.
SLEEP_CHECKPOINT_EPISODE = 90
# Probe-scaled DOWN from the substrate default (75) so the 30 remaining
# waking episodes after the checkpoint can plausibly observe either fate
# (accelerated dissolution or real corroboration) rather than the deadline
# simply never being reached within the run.
REPLAY_CORROBORATION_EPISODES = 15
REPLAY_VALUE_QUANTILE = 0.75  # substrate default; not itself under test here

# --- THE V3-EXQ-873 FIX: two decoupled thresholds replacing 873's single
#     SEED_PASS_FRACTION=1.0 (see the FRACTION GATE docstring section) ---
# Readiness (STOCHASTIC): fraction of an arm's seeds that must produce a
# qualifying real high-value candidate by the checkpoint for the arm's gate to
# be adjudicable. Measured = n_cleared / n_seeds; met when fraction > this
# floor. 0.4 over 8 seeds => at least 4 must clear. A floor, so the indexer's
# recompute (measured > threshold) agrees natively.
READINESS_SEED_FRACTION_FLOOR = 0.4
# Plumbing (DETERMINISTIC): among the CLEARED seeds only, the share that must
# mint a replay_origin=True chunk. 1.0 -- given a valid candidate the AND-gate
# must fire on every such seed; a cleared-but-unminted seed is a real defect.
C1_MINT_FRACTION = 1.0

# Readiness precondition thresholds shared with V3-EXQ-810a (same env/schedule).
OUTCOME_SPREAD_FLOOR = 0.05
SYMBOLS_PER_TRIAL_FLOOR = float(base.CHUNK_MAX_SIZE - 1)  # 4.0
HARM_STEP_FRACTION_FLOOR = 0.01

# A synthetic, low-repetition sequence distinct from whatever the accumulator's
# own best real candidate turns out to be, used ONLY for PROBE 3's below-bar
# value_tag boundary probe (never claimed to carry that value in reality).
_LOW_SEQ_CANDIDATES: Tuple[Tuple[int, ...], ...] = (
    (0, 1), (1, 0), (0, 2), (2, 0), (1, 2), (2, 1),
)

_ZG = ZGoalStreamAccumulator()


def _shared_config_kwargs() -> Dict[str, Any]:
    kw = dict(base.shared_config_kwargs())
    kw["chunk_replay_value_quantile"] = REPLAY_VALUE_QUANTILE
    kw["chunk_replay_corroboration_episodes"] = REPLAY_CORROBORATION_EPISODES
    # CO-REQUISITE FLAG (found during Step 3.5 code review, not in the
    # original task brief): SerotoninModule is itself a default-OFF
    # substrate (`SerotoninConfig.tonic_5ht_enabled=False`), and when
    # disabled ALL its methods -- including enter_sws()/enter_rem()/
    # exit_sleep() -- are no-ops that leave `.phase` pinned at "wake"
    # forever (ree_core/neuromodulation/serotonin.py: "When
    # tonic_5ht_enabled=False, all accessors return static defaults and
    # step/transition methods are no-ops"). REEAgent.
    # note_chunk_replay_sequence's default `in_sleep_phase=None` path reads
    # `self.serotonin.phase` -- so WITHOUT this second flag, PROBE 2 (the
    # should-succeed cell) would be refused by condition (b) on EVERY seed
    # regardless of use_chunk_replay_origin_path, silently reading as
    # "carve-out never fires" when the real defect would be "the sleep
    # substrate itself was never switched on". Confirmed by direct
    # verification (scratchpad/verify_mech322.py): enter_sws_mode() left
    # `.phase == "wake"` with this flag left at its default False.
    kw["tonic_5ht_enabled"] = True
    return kw


def _arm_flags(arm: str) -> Dict[str, Any]:
    if arm == ARM_OFF:
        return base.off_arm_flags()
    flags = base.off_arm_flags()
    flags.update({
        "use_policy_chunking": True,
        "use_chunk_maintenance": True,
        "use_chunk_replay_origin_path": arm == ARM_REPLAY_ON,
    })
    return flags


def _config_slice(arm: str) -> Dict[str, Any]:
    if arm == ARM_OFF:
        # Declares ONLY what the OFF computation reads -- identical to
        # V3-EXQ-810a's ARM_OFF, so this cell is fingerprint-reusable from
        # that mint (or reuse-eligible for any future sibling).
        return base.off_path_config_slice()
    slice_: Dict[str, Any] = {
        "env": dict(base.ENV_KWARGS),
        "schedule": {
            "n_episodes": N_EPISODES,
            "steps_per_episode": STEPS_PER_EPISODE,
            "sleep_checkpoint_episode": SLEEP_CHECKPOINT_EPISODE,
        },
        # Explicit (redundant with _shared_config_kwargs() below, which sets
        # them under the REEConfig field names) so the config_slice-
        # declaration lint's AST scan of THIS function sees the module-level
        # constants the cell's call graph actually reads, not just the
        # values one function call away.
        "replay_corroboration_episodes": REPLAY_CORROBORATION_EPISODES,
        "replay_value_quantile": REPLAY_VALUE_QUANTILE,
    }
    slice_.update(_shared_config_kwargs())
    slice_.update(_arm_flags(arm))
    return slice_


def _arm_ctx(arm: str) -> Dict[str, Any]:
    return {
        "id": arm,
        "chunking_on": arm in CHUNKING_ON_ARMS,
        "steps_per_episode": STEPS_PER_EPISODE,
        "n_seeds": len(SEEDS),
    }


PRECONDITIONS = [
    PreconditionSpec(
        name="episode_outcome_spread_supra_floor",
        description=("MECH-323's evaluative gate is RELATIVE to a running "
                     "baseline, so real chunk formation (a precursor to any "
                     "replay-origin test) is structurally impossible on a "
                     "flat outcome stream. Same statistic as V3-EXQ-810a."),
        control=("worst seed in this arm; the hazard/resource grid's "
                 "per-episode net signal varies"),
        threshold=OUTCOME_SPREAD_FLOOR,
        direction="lower",
        kind="readiness",
        applies_to=lambda ctx: bool(ctx["chunking_on"]),
        applies_note="ARM_OFF never constructs the accumulator (disposition (a)).",
        structural_max=lambda ctx: None,
    ),
    PreconditionSpec(
        name="chunk_buffer_supports_size_range",
        description=("Same buffer-reachability precondition as V3-EXQ-810a: "
                     "a per-trial buffer shorter than chunk_max_size makes "
                     "the tracked-sequence tally too shallow to hold a "
                     "genuine repeated high-value candidate."),
        control="worst seed in this arm; e3_steps_per_tick periodicity",
        threshold=SYMBOLS_PER_TRIAL_FLOOR,
        direction="lower",
        kind="readiness",
        applies_to=lambda ctx: bool(ctx["chunking_on"]),
        applies_note="ARM_OFF never constructs the accumulator (disposition (a)).",
        structural_max=lambda ctx: float(ctx["steps_per_episode"]),
    ),
    PreconditionSpec(
        name="salient_harm_events_fire",
        description=("Task-property check shared with V3-EXQ-810a: confirms "
                     "the hazard-bearing grid is genuinely exercised, so an "
                     "arm's gate is never vacuously green from zero applied "
                     "preconditions."),
        control="worst seed in this arm on the 8x8 hazard/proxy grid",
        threshold=HARM_STEP_FRACTION_FLOOR,
        direction="lower",
        kind="readiness",
        applies_to=lambda ctx: True,
        structural_max=lambda ctx: 1.0,
    ),
    PreconditionSpec(
        name="replay_high_value_candidate_seed_fraction",
        description=("V3-EXQ-873 FIX: MECH-322 condition (a)'s own readiness "
                     "precondition, aggregated as the FRACTION OF SEEDS in "
                     "which at least one tracked real sub-sequence had an "
                     "outcome mean clearing ChunkAccumulator."
                     "replay_value_threshold() by the sleep checkpoint -- NOT "
                     "873's worst-of-N candidate margin, which let one seed "
                     "with no candidate veto the whole arm. Measured = "
                     "n_cleared / n_seeds; a floor. It certifies the arm is "
                     "ADJUDICABLE (enough seeds can attempt PROBE 2); the "
                     "DETERMINISTIC per-seed mint is C1, scored over cleared "
                     "seeds only."),
        control=("fraction of this arm's seeds that produced a qualifying "
                 "candidate; goes red only when candidates rarely emerge (a "
                 "real substrate/schedule problem), never on one unlucky seed"),
        threshold=READINESS_SEED_FRACTION_FLOOR,
        direction="lower",
        kind="readiness",
        applies_to=lambda ctx: bool(ctx["chunking_on"]),
        applies_note="ARM_OFF never constructs the accumulator (disposition (a)).",
        structural_max=lambda ctx: 1.0,
    ),
]

_NO_CANDIDATE_MARGIN = -1.0e9  # sentinel: no tracked candidate at all this seed


def _worst(rows: List[Dict[str, Any]], key: str):
    if not rows:
        return 0.0, None
    worst = min(rows, key=lambda r: r[key])
    return float(worst[key]), f"{worst['arm']}/seed{worst['seed']}"


def _clear_fraction(rows: List[Dict[str, Any]]):
    """Fraction of seeds in this arm that produced a qualifying candidate.

    Returns (fraction, n_cleared, offending_cells) where offending_cells names
    the seeds that did NOT clear (so a red readiness gate is attributable),
    or None when every seed cleared.
    """
    if not rows:
        return 0.0, 0, None
    cleared = [r for r in rows if r.get("replay_high_value_candidate_available") is True]
    not_cleared = [f"{r['arm']}/seed{r['seed']}"
                   for r in rows if r.get("replay_high_value_candidate_available") is not True]
    frac = len(cleared) / len(rows)
    return frac, len(cleared), (", ".join(not_cleared) if not_cleared else None)


def _no_replay_mint(r: Dict[str, Any]) -> bool:
    """True iff none of the three probes minted a replay-origin chunk."""
    return (r.get("wake_refusal_minted") is not True
            and r.get("success_minted") is not True
            and r.get("low_value_refusal_minted") is not True)


def _run_off_cell(seed: int) -> Dict[str, Any]:
    print(f"Seed {seed} Condition {ARM_OFF}", flush=True)
    with arm_cell(seed,
                  config_slice=_config_slice(ARM_OFF),
                  script_path=Path(__file__),
                  config_slice_declared=True,
                  include_driver_script_in_hash=False) as cell:
        raw = base.run_cell(base.off_arm_flags(), seed,
                            n_episodes=N_EPISODES,
                            steps_per_episode=STEPS_PER_EPISODE,
                            progress_every=max(1, min(30, N_EPISODES // 4)),
                            progress_label=f"seed={seed} arm={ARM_OFF}")
        st = raw["chunking_state"] or {}
        outcomes = raw["episode_outcomes"]
        row = {
            "arm": ARM_OFF, "seed": seed,
            "chunk_acc_n_formed": int(st.get("chunk_acc_n_formed", 0)),
            "chunk_acc_n_replay_formed": int(st.get("chunk_acc_n_replay_formed", 0)),
            "chunk_acc_n_replay_refusals": int(st.get("chunk_acc_n_replay_refusals", 0)),
            "chunk_acc_n_steps": int(st.get("chunk_acc_n_steps", 0)),
            "chunk_acc_n_outcomes": int(st.get("chunk_acc_n_outcomes", 0)),
            "chunk_lib_n_replay_origin": int(st.get("chunk_lib_n_replay_origin", 0)),
            "chunk_lib_n_replay_deadline_dissolutions": 0,
            "chunk_lib_by_state": st.get("chunk_lib_by_state", {}),
            "chunking_instantiated": bool(raw["chunking_instantiated"]),
            "replay_high_value_candidate_available": None,
            "replay_candidate_margin": _NO_CANDIDATE_MARGIN,
            "replay_value_threshold": None,
            "high_seq_mu": None, "high_seq_len": 0, "low_seq_value_tag": None,
            "wake_refusal_minted": None, "success_minted": None,
            "success_minted_replay_origin": None, "low_value_refusal_minted": None,
            "phase_before_sleep": "wake", "phase_during_wake_refusal_attempt": "wake",
            "phase_after_sleep_exit": "wake",
            "minted_chunk_state_end_of_run": None,
            "minted_episodes_since_corroboration_end_of_run": None,
            "minted_n_dissolutions_end_of_run": None,
            "n_harm_steps": raw["n_harm_steps"], "n_steps_total": raw["n_steps_total"],
            "harm_step_fraction": ((raw["n_harm_steps"] / raw["n_steps_total"])
                                   if raw["n_steps_total"] else 0.0),
            "symbols_per_trial": 0.0,
            "episode_outcome_mean": statistics.fmean(outcomes) if outcomes else 0.0,
            "episode_outcome_spread": (statistics.pstdev(outcomes)
                                       if len(outcomes) > 1 else 0.0),
            "n_episodes": len(outcomes),
            "per_episode_outcomes": outcomes,
            "z_goal_stream_stats": raw["z_goal_stream_stats"],
        }
        _ZG.observe_stats(row["z_goal_stream_stats"])
        cell.stamp(row)
    cell_ok = row["chunk_acc_n_formed"] == 0 and not row["chunking_instantiated"]
    print(f"verdict: {'PASS' if cell_ok else 'FAIL'}", flush=True)
    return row


def _run_replay_cell(arm: str, seed: int) -> Dict[str, Any]:
    print(f"Seed {seed} Condition {arm}", flush=True)
    flags = _arm_flags(arm)
    with arm_cell(seed,
                  config_slice=_config_slice(arm),
                  script_path=Path(__file__),
                  config_slice_declared=True,
                  include_driver_script_in_hash=False) as cell:
        env = base.build_env(seed)
        agent = REEAgent(REEConfig.from_dims(
            body_obs_dim=env.body_obs_dim,
            world_obs_dim=env.world_obs_dim,
            action_dim=4,
            **_shared_config_kwargs(),
            **flags,
        ))
        harness = StepHarness(agent, env, train_mode=False, seed=seed)

        episode_outcomes: List[float] = []
        n_harm_steps = 0
        n_steps_total = 0

        # --- probe bookkeeping, filled in at the sleep checkpoint ---
        replay_high_value_candidate_available = False
        replay_candidate_margin = _NO_CANDIDATE_MARGIN
        replay_value_threshold: Optional[float] = None
        high_seq: Optional[Tuple[int, ...]] = None
        high_mu: Optional[float] = None
        low_seq: Optional[Tuple[int, ...]] = None
        low_value_tag: Optional[float] = None
        wake_refusal_minted = None
        success_minted = None
        success_minted_replay_origin = None
        low_value_refusal_minted = None
        phase_before_sleep = "wake"
        phase_after_sleep_exit = "wake"

        # Adaptive so the runner's `ep N/M` progress line is emitted at least
        # twice per cell even under --dry-run, where N_EPISODES is tiny.
        progress_every = max(1, min(30, N_EPISODES // 4))

        for ep in range(N_EPISODES):
            if ep == SLEEP_CHECKPOINT_EPISODE:
                pc = agent.policy_chunking
                replay_value_threshold = (
                    float(pc.accumulator.replay_value_threshold())
                    if pc is not None else None
                )
                # IMPORTANT: NOT formation_candidates(). A sequence qualifying
                # there almost always ALREADY holds a REAL (non-replay)
                # library slot -- PolicyChunking.note_outcome() mints+
                # registers it the instant it first qualifies, on every
                # earlier waking episode's outcome report. A replay-mint
                # attempt on that same key then fails at library.register()
                # (key collision, since a chunk already occupies it), which
                # reads identically to a MECH-322 refusal but tests nothing
                # about MECH-322. Confirmed by direct verification
                # (scratchpad/verify_mech322.py) before this fix: PROBE 2
                # returned None on every formation_candidates() pick despite
                # every AND-condition genuinely holding.
                #
                # The correct PROBE 2 candidate is a real, tracked-but-NOT-
                # YET-REGISTERED sub-sequence: it has genuine per-episode
                # outcomes credited to it (a real value_tag), but never met
                # the full joint MECH-323 formation gate (repetition AND
                # low-variance AND supra-baseline mean), so no real chunk
                # occupies its key. No public accessor exposes the raw tally
                # without that joint gate, so this reads
                # ChunkAccumulator._tally directly -- the same collection
                # PolicyChunking.note_outcome() itself reads internally to
                # build per-chunk variances; this is read-only diagnostic
                # access from the driver, not a substrate modification.
                if pc is not None:
                    unregistered = []
                    for seq, outcomes in pc.accumulator._tally.items():
                        if not outcomes or pc.library.get(seq) is not None:
                            continue
                        unregistered.append((seq, sum(outcomes) / len(outcomes)))
                    if unregistered:
                        high_seq, high_mu = max(unregistered, key=lambda c: c[1])
                        thr = (replay_value_threshold if replay_value_threshold is not None
                               else float("inf"))
                        replay_high_value_candidate_available = (
                            bool(high_mu >= thr) if thr != float("inf") else False)
                        replay_candidate_margin = (
                            float(high_mu - thr) if thr != float("inf")
                            else _NO_CANDIDATE_MARGIN)

                if high_seq is not None:
                    for cand in _LOW_SEQ_CANDIDATES:
                        if cand != high_seq:
                            low_seq = cand
                            break
                    thr = replay_value_threshold if replay_value_threshold is not None else 0.0
                    low_value_tag = float(thr) - 1.0

                    # PROBE 1 (condition b, WAKE): still genuinely awake here.
                    phase_before_sleep = str(agent.serotonin.phase)
                    wake_refusal_minted = agent.note_chunk_replay_sequence(
                        high_seq, value_tag=float(high_mu), in_sleep_phase=None
                    ) is not None

                    # Genuine SD-017 SWS entry -- real substrate transition,
                    # not an override. Schema/REM passes deliberately not run
                    # (orthogonal to the phase-gate condition under test).
                    agent.enter_sws_mode()

                    # PROBE 2 (conditions a+c, SWS): real high-value sequence,
                    # now genuinely asleep.
                    success_chunk = agent.note_chunk_replay_sequence(
                        high_seq, value_tag=float(high_mu), in_sleep_phase=None
                    )
                    success_minted = success_chunk is not None
                    success_minted_replay_origin = bool(
                        getattr(success_chunk, "replay_origin", False)
                    )

                    # PROBE 3 (condition a, SWS): distinct sequence, deliberate
                    # below-bar value_tag -- a boundary probe of the value gate
                    # in isolation, not a claim this sequence carries that value.
                    low_value_refusal_minted = agent.note_chunk_replay_sequence(
                        low_seq, value_tag=low_value_tag, in_sleep_phase=None
                    ) is not None

                    agent.exit_sleep_mode()
                    phase_after_sleep_exit = str(agent.serotonin.phase)

            results = harness.run_episode(STEPS_PER_EPISODE)
            ep_reward = 0.0
            for r in results:
                ep_reward += r.harm_signal
                if r.harm_signal < 0:
                    n_harm_steps += 1
            n_steps_total += len(results)
            agent.note_chunk_outcome(ep_reward)
            episode_outcomes.append(ep_reward)

            if (ep + 1) % progress_every == 0:
                st_now = agent.get_chunking_state()
                print(f"  [train] chunk {arm} seed={seed} ep {ep+1}/{N_EPISODES} "
                      f"formed={st_now.get('chunk_acc_n_formed', 0)} "
                      f"replay_formed={st_now.get('chunk_acc_n_replay_formed', 0)}",
                      flush=True)

        st = agent.get_chunking_state()

        minted_chunk_state = None
        minted_episodes_since_corroboration = None
        minted_n_dissolutions = None
        pc = agent.policy_chunking
        if pc is not None and high_seq is not None:
            chunk = pc.library.get(high_seq)
            if chunk is not None:
                minted_chunk_state = chunk.state.value
                minted_episodes_since_corroboration = int(chunk.episodes_since_corroboration)
                minted_n_dissolutions = int(chunk.n_dissolutions)

        n_symbols = int(st.get("chunk_acc_n_steps", 0))
        n_outcomes = int(st.get("chunk_acc_n_outcomes", 0))

        row = {
            "arm": arm, "seed": seed,
            "chunk_acc_n_formed": int(st.get("chunk_acc_n_formed", 0)),
            "chunk_acc_n_replay_formed": int(st.get("chunk_acc_n_replay_formed", 0)),
            "chunk_acc_n_replay_refusals": int(st.get("chunk_acc_n_replay_refusals", 0)),
            "chunk_acc_n_steps": n_symbols,
            "chunk_acc_n_outcomes": n_outcomes,
            "chunk_lib_n_replay_origin": int(st.get("chunk_lib_n_replay_origin", 0)),
            "chunk_lib_n_replay_deadline_dissolutions": int(
                st.get("chunk_lib_n_replay_deadline_dissolutions", 0)),
            "chunk_lib_n_crystallised": int(st.get("chunk_lib_n_crystallised", 0)),
            "chunk_lib_n_dissolved": int(st.get("chunk_lib_n_dissolved", 0)),
            "chunk_lib_by_state": st.get("chunk_lib_by_state", {}),
            "chunking_instantiated": bool(agent.policy_chunking is not None),
            "replay_high_value_candidate_available": replay_high_value_candidate_available,
            "replay_candidate_margin": replay_candidate_margin,
            "replay_value_threshold": replay_value_threshold,
            "high_seq_mu": float(high_mu) if high_mu is not None else None,
            "high_seq_len": len(high_seq) if high_seq is not None else 0,
            "low_seq_value_tag": low_value_tag,
            "wake_refusal_minted": wake_refusal_minted,
            "success_minted": success_minted,
            "success_minted_replay_origin": success_minted_replay_origin,
            "low_value_refusal_minted": low_value_refusal_minted,
            "phase_before_sleep": phase_before_sleep,
            "phase_during_wake_refusal_attempt": phase_before_sleep,
            "phase_after_sleep_exit": phase_after_sleep_exit,
            "minted_chunk_state_end_of_run": minted_chunk_state,
            "minted_episodes_since_corroboration_end_of_run": minted_episodes_since_corroboration,
            "minted_n_dissolutions_end_of_run": minted_n_dissolutions,
            "n_harm_steps": n_harm_steps, "n_steps_total": n_steps_total,
            "harm_step_fraction": (n_harm_steps / n_steps_total) if n_steps_total else 0.0,
            "symbols_per_trial": (n_symbols / n_outcomes) if n_outcomes else 0.0,
            "episode_outcome_mean": (statistics.fmean(episode_outcomes)
                                     if episode_outcomes else 0.0),
            "episode_outcome_spread": (statistics.pstdev(episode_outcomes)
                                       if len(episode_outcomes) > 1 else 0.0),
            "n_episodes": len(episode_outcomes),
            "per_episode_outcomes": episode_outcomes,
            "z_goal_stream_stats": harness.z_goal_stream_stats(),
        }
        _ZG.observe_stats(row["z_goal_stream_stats"])
        cell.stamp(row)

    if arm == ARM_REPLAY_ON:
        # Cell PASSes iff it either legitimately minted (cleared readiness), OR
        # legitimately did not attempt because no candidate emerged this seed
        # (readiness not cleared) -- the latter is a correct fail-closed, not a
        # cell defect. A cleared-but-unminted seed is the only cell FAIL.
        if replay_high_value_candidate_available:
            cell_ok = bool(row["success_minted"] and row["success_minted_replay_origin"])
        else:
            cell_ok = row["success_minted"] is not True
    else:
        cell_ok = _no_replay_mint(row)
    print(f"verdict: {'PASS' if cell_ok else 'FAIL'}", flush=True)
    return row


def _run_cell(arm: str, seed: int) -> Dict[str, Any]:
    if arm == ARM_OFF:
        return _run_off_cell(seed)
    return _run_replay_cell(arm, seed)


def run_experiment() -> Dict[str, Any]:
    assert_no_structurally_unsatisfiable_gate(
        PRECONDITIONS, [_arm_ctx(a) for a in ARMS])

    rows: List[Dict[str, Any]] = []
    for arm in ARMS:
        for seed in SEEDS:
            rows.append(_run_cell(arm, seed))

    by_arm = {a: [r for r in rows if r["arm"] == a] for a in ARMS}

    arm_gates = []
    readiness_frac_by_arm: Dict[str, float] = {}
    n_cleared_by_arm: Dict[str, int] = {}
    for arm in ARMS:
        cells = by_arm[arm]
        spread_w, spread_cell = _worst(cells, "episode_outcome_spread")
        symbols_w, symbols_cell = _worst(cells, "symbols_per_trial")
        harm_w, harm_cell = _worst(cells, "harm_step_fraction")
        cand_frac, n_cleared, no_cand_cells = _clear_fraction(cells)
        readiness_frac_by_arm[arm] = cand_frac
        n_cleared_by_arm[arm] = n_cleared
        arm_gates.append(evaluate_arm_gate(
            arm, _arm_ctx(arm), PRECONDITIONS,
            measured={
                "episode_outcome_spread_supra_floor": spread_w,
                "chunk_buffer_supports_size_range": symbols_w,
                "salient_harm_events_fire": harm_w,
                "replay_high_value_candidate_seed_fraction": cand_frac,
            },
        ))
        arm_gates[-1]["offending_cells"] = {
            "episode_outcome_spread_supra_floor": spread_cell,
            "chunk_buffer_supports_size_range": symbols_cell,
            "salient_harm_events_fire": harm_cell,
            "replay_high_value_candidate_seed_fraction": no_cand_cells,
        }
    agg = aggregate_arm_gates(arm_gates)
    gate_by_arm = {g["arm"]: bool(g["gate_green"]) for g in arm_gates}

    # C1 is authorised only when ARM_REPLAY_ON's own gate is green -- a red
    # gate elsewhere must not vacate it (785-correct aggregation).
    c1_arm_green = gate_by_arm.get(ARM_REPLAY_ON, False)

    # --- C1 (LOAD-BEARING, DETERMINISTIC PLUMBING): among the seeds that
    # CLEARED readiness (a qualifying candidate emerged), the fraction that
    # minted a replay_origin=True chunk. 873's defect was scoring this over ALL
    # seeds at SEED_PASS_FRACTION=1.0, so a no-candidate seed counted as a mint
    # failure. Here a no-candidate seed is EXCLUDED from the denominator.
    cleared_rows = [r for r in by_arm[ARM_REPLAY_ON]
                    if r["replay_high_value_candidate_available"] is True]
    c1_minted = sum(1 for r in cleared_rows
                    if r["success_minted"] and r["success_minted_replay_origin"])
    c1_pass = bool(cleared_rows) and (c1_minted / len(cleared_rows)) >= C1_MINT_FRACTION

    def _attempted(r: Dict[str, Any]) -> bool:
        return r.get("wake_refusal_minted") is not None

    replay_attempted_rows = [r for r in by_arm[ARM_REPLAY_ON] if _attempted(r)]
    strict_attempted_rows = [r for r in by_arm[ARM_STRICT_FLAG_OFF] if _attempted(r)]

    # --- C2 (SAFETY): waking attempt with a genuinely high real value_tag is
    # refused (condition b fails closed), in ARM_REPLAY_ON.
    c2_any_attempted = len(replay_attempted_rows) > 0
    c2_pass = c2_any_attempted and all(
        r["wake_refusal_minted"] is False for r in replay_attempted_rows)

    # --- C3 (SAFETY): a below-bar value_tag is refused even while asleep
    # (condition a fails closed), in ARM_REPLAY_ON.
    c3_pass = c2_any_attempted and all(
        r["low_value_refusal_minted"] is False for r in replay_attempted_rows)

    # --- C4 (SAFETY, MECH-094 GLOBAL): the master switch alone refuses every
    # attempt regardless of phase or value, in ARM_STRICT_FLAG_OFF.
    c4_any_attempted = len(strict_attempted_rows) > 0
    c4_pass = c4_any_attempted and all(
        _no_replay_mint(r) and r["chunk_acc_n_replay_formed"] == 0
        and r["chunk_lib_n_replay_origin"] == 0
        for r in strict_attempted_rows)

    # --- C5: ARM_OFF is inert (identical to V3-EXQ-810a's C3).
    c5_pass = all((not r["chunking_instantiated"]) and r["chunk_acc_n_formed"] == 0
                  for r in by_arm[ARM_OFF])

    # --- C6 (informational bookkeeping): the corroboration-deadline mechanism
    # is genuinely trackable for every chunk PROBE 2 minted (not a directional
    # claim about which of the two documented fates occurs).
    minted_rows = [r for r in by_arm[ARM_REPLAY_ON] if r["success_minted"]]
    c6_pass = all(r["minted_chunk_state_end_of_run"] is not None for r in minted_rows) \
        if minted_rows else True
    deadline_fates = {
        "dissolved": sum(1 for r in minted_rows
                         if r["minted_chunk_state_end_of_run"] == "dissolved"),
        "still_tracked": sum(1 for r in minted_rows
                             if r["minted_chunk_state_end_of_run"] not in
                             (None, "dissolved")),
    }

    # --- C7 (safety continuity): the audit counters agree with the boolean
    # readout on ARM_REPLAY_ON's success cells.
    c7_pass = all(
        r["chunk_acc_n_replay_formed"] >= 1 and r["chunk_lib_n_replay_origin"] >= 1
        for r in minted_rows
    ) if minted_rows else True

    overall_pass = bool(c1_arm_green and c1_pass and c2_pass and c3_pass
                        and c4_pass and c5_pass and c6_pass and c7_pass)

    if not c1_arm_green:
        label = "substrate_not_ready_requeue"
    elif overall_pass:
        label = "replay_carveout_fires_and_fails_closed"
    elif not c1_pass:
        label = "replay_carveout_silent_under_valid_conditions"
    elif not c2_pass:
        label = "replay_carveout_wake_refusal_defect"
    elif not c3_pass:
        label = "replay_carveout_value_gate_defect"
    elif not c4_pass:
        label = "replay_carveout_master_switch_defect_SAFETY_CRITICAL"
    elif not c5_pass:
        label = "off_arm_not_inert"
    else:
        label = "replay_carveout_partial"

    metrics = {
        "c1_pass": c1_pass, "c2_pass": c2_pass, "c3_pass": c3_pass,
        "c4_pass": c4_pass, "c5_pass": c5_pass, "c6_pass": c6_pass, "c7_pass": c7_pass,
        "gate_c1_arm_green": bool(c1_arm_green),
        "gate_any_arm_green": bool(agg["non_degenerate"]),
        "gate_green_by_arm": gate_by_arm,
        "n_seeds": len(SEEDS),
        # --- V3-EXQ-873 FIX diagnostics: the decoupled readiness/plumbing pair ---
        "readiness_seed_fraction_by_arm": readiness_frac_by_arm,
        "n_cleared_seeds_by_arm": n_cleared_by_arm,
        "readiness_seed_fraction_floor": READINESS_SEED_FRACTION_FLOOR,
        "c1_cleared_seed_count": len(cleared_rows),
        "c1_minted_cleared_count": c1_minted,
        "c1_mint_fraction": (c1_minted / len(cleared_rows)) if cleared_rows else None,
        "c1_mint_fraction_floor": C1_MINT_FRACTION,
        "n_replay_attempted_seeds": len(replay_attempted_rows),
        "n_strict_attempted_seeds": len(strict_attempted_rows),
        "corroboration_deadline_fates": deadline_fates,
        "replay_formed_total": sum(r["chunk_acc_n_replay_formed"] for r in rows),
        "replay_origin_total": sum(r["chunk_lib_n_replay_origin"] for r in rows),
        "replay_deadline_dissolutions_total": sum(
            r["chunk_lib_n_replay_deadline_dissolutions"] for r in rows),
    }

    criteria_by_arm = {
        ARM_OFF: ["C5_off_arm_inert"],
        ARM_STRICT_FLAG_OFF: ["C4_master_switch_refuses_regardless_of_phase_or_value"],
        ARM_REPLAY_ON: ["C1_carveout_fires_under_valid_conditions",
                        "C2_wake_refusal", "C3_low_value_refusal",
                        "C6_corroboration_deadline_tracked",
                        "C7_replay_counters_consistent"],
    }
    criteria_non_degenerate = arm_criteria_non_degenerate(
        criteria_by_arm, agg,
        extra={
            # C1 is non-degenerate only when the arm gate is green (>= floor of
            # seeds cleared) AND at least one cleared seed exists to score over.
            "C1_carveout_fires_under_valid_conditions": c1_arm_green and len(cleared_rows) > 0,
            "C2_wake_refusal": c2_any_attempted,
            "C3_low_value_refusal": c2_any_attempted,
            "C6_corroboration_deadline_tracked": len(minted_rows) > 0,
            "C7_replay_counters_consistent": len(minted_rows) > 0,
        })
    # Absence-type safety assertions -- meaningful whatever the readiness gate did.
    criteria_non_degenerate["C4_master_switch_refuses_regardless_of_phase_or_value"] = (
        c4_any_attempted)
    criteria_non_degenerate["C5_off_arm_inert"] = True

    interpretation = {
        "label": label,
        "preconditions": agg["adjudication_preconditions"],
        "criteria": [
            {"name": "C1_carveout_fires_under_valid_conditions", "load_bearing": True,
             "passed": c1_pass},
            {"name": "C2_wake_refusal", "load_bearing": False, "passed": c2_pass},
            {"name": "C3_low_value_refusal", "load_bearing": False, "passed": c3_pass},
            {"name": "C4_master_switch_refuses_regardless_of_phase_or_value",
             "load_bearing": False, "passed": c4_pass},
            {"name": "C5_off_arm_inert", "load_bearing": False, "passed": c5_pass},
            {"name": "C6_corroboration_deadline_tracked", "load_bearing": False,
             "passed": c6_pass},
            {"name": "C7_replay_counters_consistent", "load_bearing": False,
             "passed": c7_pass},
        ],
        "criteria_non_degenerate": criteria_non_degenerate,
        "per_arm_gate": agg["per_arm_gate"],
        "gate_offending_cells": {g["arm"]: g["offending_cells"] for g in arm_gates},
        "combination_rule": (
            "overall PASS = ARM_REPLAY_ON gate green (readiness: >= "
            f"{READINESS_SEED_FRACTION_FLOOR:.2f} of seeds produced a qualifying "
            "candidate) AND C1 (>= "
            f"{C1_MINT_FRACTION:.2f} of CLEARED seeds minted replay_origin) AND "
            "C2..C7. C1 is scored over cleared seeds only -- a no-candidate seed "
            "is excluded from C1's denominator, not counted as a mint failure "
            "(the V3-EXQ-873 test-design fix)."),
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

    global SEEDS, N_EPISODES, STEPS_PER_EPISODE, SLEEP_CHECKPOINT_EPISODE
    global REPLAY_CORROBORATION_EPISODES
    if args.dry_run:
        SEEDS = [101]
        N_EPISODES = 6
        STEPS_PER_EPISODE = 72
        SLEEP_CHECKPOINT_EPISODE = 3
        REPLAY_CORROBORATION_EPISODES = 2

    result = run_experiment()
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    full_config = {
        "seeds": SEEDS, "arms": ARMS,
        "n_episodes": N_EPISODES,
        "steps_per_episode": STEPS_PER_EPISODE,
        "sleep_checkpoint_episode": SLEEP_CHECKPOINT_EPISODE,
        "replay_corroboration_episodes": REPLAY_CORROBORATION_EPISODES,
        "replay_value_quantile": REPLAY_VALUE_QUANTILE,
        "env": dict(base.ENV_KWARGS),
        "alpha_world": base.ALPHA_WORLD,
        "chunk_min_repetitions": base.CHUNK_MIN_REPETITIONS,
        "chunk_window_trials": base.CHUNK_WINDOW_TRIALS,
        "chunk_crystallisation_min": base.CHUNK_CRYSTALLISATION_MIN,
        "chunk_min_size": base.CHUNK_MIN_SIZE,
        "chunk_max_size": base.CHUNK_MAX_SIZE,
        "readiness_seed_fraction_floor": READINESS_SEED_FRACTION_FLOOR,
        "c1_mint_fraction": C1_MINT_FRACTION,
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
        "claim_ids": CLAIM_IDS,
        "backlog_id": BACKLOG_ID,
        "supersedes": SUPERSEDES,
        "evidence_direction": "supports" if result["outcome"] == "PASS" else "unknown",
        "outcome": result["outcome"],
        "timestamp_utc": ts,
        "metrics": result["metrics"],
        "per_seed_rows": result["per_seed_rows"],
        "arm_results": result["arm_results"],
        "interpretation": result["interpretation"],
        "per_arm_gate": result["per_arm_gate"],
        "non_degenerate": result["non_degenerate"],
        "degeneracy_reason": result["degeneracy_reason"],
        "sleep_driver_pattern": (
            "manual-cycle-loop (single wake-sleep-wake cycle, N_CYCLES=1; "
            "enter_sws_mode()/exit_sleep_mode() called directly once per "
            "cell at the sleep checkpoint episode; no SD-017 schema/REM "
            "passes run)"
        ),
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
    print(f"C1={m['c1_pass']} C2={m['c2_pass']} C3={m['c3_pass']} C4={m['c4_pass']} "
          f"C5={m['c5_pass']} C6={m['c6_pass']} C7={m['c7_pass']}", flush=True)
    print(f"readiness_seed_fraction_by_arm={m['readiness_seed_fraction_by_arm']} "
          f"(floor={m['readiness_seed_fraction_floor']})", flush=True)
    print(f"c1: {m['c1_minted_cleared_count']}/{m['c1_cleared_seed_count']} cleared seeds "
          f"minted (frac={m['c1_mint_fraction']}, floor={m['c1_mint_fraction_floor']})",
          flush=True)
    print(f"replay_formed_total={m['replay_formed_total']} "
          f"replay_origin_total={m['replay_origin_total']} "
          f"deadline_dissolutions_total={m['replay_deadline_dissolutions_total']}",
          flush=True)
    print(f"corroboration_deadline_fates: {m['corroboration_deadline_fates']}", flush=True)
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
