"""V3-EXQ-892 -- MECH-322 replay-origin CORROBORATION SURVIVAL under realistic conditions.

SLEEP DRIVER: manual-cycle-loop (single wake-sleep-wake cycle, N_CYCLES=1;
enter_sws_mode()/exit_sleep_mode() called directly once per cell, at the
SLEEP_CHECKPOINT_EPISODE boundary; no SD-017 schema-installation or REM
slot-fill passes are run -- only the MECH-322 mint + its post-mint lifecycle
are exercised, orthogonal to schema/slot-fill content).

PURPOSE (diagnostic -- discriminates a post-mint lifecycle question, NOT a
behavioural claim). This is NOT a lettered continuation of V3-EXQ-873a, whose
own question ("does the AND-gate fire and fail closed?") is ANSWERED (confirmed
autopsy failure_autopsy_V3-EXQ-873a_2026-08-07: 7/8 seeds cleared readiness,
all 7 minted, all fail-closed safety checks held -> first genuine support for
MECH-322). This is a NEW EXQ NUMBER for the distinct question that 873a's
autopsy explicitly routed onward and did NOT answer:

    Does a replay-origin-tagged chunk, minted under valid conditions, ever get
    CORROBORATED by real waking execution before its dissolution deadline
    expires -- under REALISTIC (not artificially accelerated) conditions? And
    what conditions (deadline length, exposure/recurrence rate) make
    corroboration likely vs unlikely?

WHY THIS IS GENUINELY UNTESTED. In 873a all 7 minted replay-origin chunks
DISSOLVED without real-execution corroboration by the run's end. That was
EXPECTED safety-valve behaviour, not a defect -- but it was measured under a
single, ACCELERATED regime: corroboration deadline N=15 with the sleep
checkpoint at episode 90 of 120 (only 30 post-mint waking episodes), so the
deadline (ep 90+15=105) bit well before the run ended. 873a therefore recorded
"corroboration rate = 0/7 in the accelerated regime" and nothing about the
realistic regime. The substrate DEFAULT deadline is N=75
(chunk_replay_corroboration_episodes, ree_core/utils/config.py), and a chunk
minted early has ~100 waking episodes in which its sequence might recur. That
regime is present in NO recorded manifest (GOV-REUSE-1 checked below), so the
survival question is genuinely open.

THE MECHANISM UNDER OBSERVATION (ree-v3/ree_core/policy/policy_chunking.py).
A replay-origin chunk is minted by `ChunkAccumulator.record_replay_sequence`
-> `mint(..., replay_origin=True)` in state FORMING. FORMING chunks are NOT
selectable (`ChunkLibrary.selectable_chunks` returns CRYSTALLISED only), so the
agent does not preferentially execute the minted sequence -- corroboration can
happen ONLY via EMERGENT real waking re-execution:
  * `PolicyChunking.note_outcome` (per episode, via `note_chunk_outcome`) calls
    `ChunkLibrary.note_real_execution` for every library chunk `_was_executed`
    this episode. That call resets `episodes_since_corroboration = 0` and
    increments `crystallisation_counter`; at `crystallisation_min`
    (= CHUNK_CRYSTALLISATION_MIN = 2 in this lineage) the chunk CRYSTALLISES
    and becomes selectable.
  * `ChunkLibrary.note_episode_end` -> `advance_corroboration_deadlines`
    (per episode, via `agent.reset()` -> `policy_chunking.end_episode()`)
    increments `episodes_since_corroboration` and, at
    `replay_corroboration_episodes` (= N) waking episodes without a real
    execution, retires the chunk DIRECTLY to DISSOLVED (the accelerated
    deadline; `_n_replay_deadline_dissolutions`).
So a minted chunk SURVIVES iff its sequence is emergently re-executed within N
waking episodes of minting. Corroboration is on the DEFAULT `_was_executed`
predicate (trailing-match: the sequence must END the episode's recorded action
buffer -- `use_chunk_all_position_credit=False`), which is deliberately
restrictive, so recurrence -- and therefore survival -- is expected to be a
narrow corner of the exposure space. Characterising that corner is the job.

DESIGN. Three arms, all identical EXCEPT the corroboration deadline N. All arms
run the MECH-322 carve-out ON (use_chunk_replay_origin_path=True), mint one
replay-origin chunk at a SINGLE EARLY sleep checkpoint (episode 20 of 120 ->
100 post-mint waking episodes), then let the run continue and OBSERVE the
minted chunk's fate. The mint depends only on pre-checkpoint behaviour, which
is IDENTICAL across arms (N never touches the policy, only the post-mint
deadline counter), so the SAME seed mints the SAME sequence in all three arms
and the ONLY thing that differs is the episode budget within which a
corroborating re-execution must land:

  ARM_DEADLINE_15   N=15  -- the 873a ACCELERATED regime, replicated as a
                            negative-baseline control (deadline bites at ep 35).
  ARM_DEADLINE_75   N=75  -- the SUBSTRATE DEFAULT = the REALISTIC condition
                            (deadline bites at ep 95, still within the run).
  ARM_DEADLINE_200  N=200 -- deadline set BEYOND the run length (never bites),
                            isolating the deadline-INDEPENDENT corroboration
                            opportunity: this arm's corroboration count is the
                            true recurrence/exposure rate over all 100 episodes.

There is no OFF/baseline arm and no wake/low-value/master-switch safety probe:
those are V3-EXQ-873a's job and are ALREADY validated (7/8 seeds, all
fail-closed checks held). This run tests only the post-mint lifecycle. Because
all three arms run the carve-out ON with a genuine mint, ARM_FINGERPRINT is
still emitted per (arm, seed) cell (multi-arm contract) but no arm is
reuse-eligible-as-a-baseline (none is an OFF path); the cells are stamped
reuse-INELIGIBLE via `extra_ineligible_reasons` because they share no canonical
OFF module with any lineage sibling.

DV-SYMMETRY (Step 3 mandatory per-arm declaration). Each arm's load-bearing DVs
are the minted chunk's post-mint fate: `minted_survived` (state != DISSOLVED at
run end), `minted_n_corroborations_end` (crystallisation_counter), and
`minted_first_corroboration_offset` (episodes after checkpoint until the first
real re-execution). The MANIPULATION across arms is the deadline N
(chunk_replay_corroboration_episodes), an integer episode-budget threshold.
  ARM_DEADLINE_15 / ARM_DEADLINE_75 / ARM_DEADLINE_200: the arm's survival DV is
  a deadline-GATED STEP FUNCTION of N -- a chunk whose first corroborating
  re-execution lands at post-checkpoint offset E survives iff E < N (once
  DISSOLVED, retention is off so a later re-execution does NOT revive it). The
  manipulation is therefore NOT invariant under any additive, monotone, or
  permutation symmetry of the DV: N directly sets the window in which a
  corroboration event counts, so wherever a corroboration event exists in the
  [15, 200] window the manipulation moves survival by construction. The
  corroboration TIMING itself (`minted_first_corroboration_offset`) is
  N-INDEPENDENT for a given seed -- which is exactly what makes N a clean
  isolation of the deadline and not a confound. No symmetry group renders the
  N -> survival manipulation invisible to its own DV.

GOV-REUSE-1 (Step 2.4). Decisive readout: the fraction of minted replay-origin
chunks that reach `crystallisation_counter >= 1` (corroborated at least once)
before DISSOLUTION, under the DEFAULT deadline N=75 and an early checkpoint --
plus the deadline-independent recurrence count in the N=200 arm. Searched the
MECH-322 manifest corpus: only V3-EXQ-873 (n=3, accelerated N=15, checkpoint
ep 90) and V3-EXQ-873a (n=8, accelerated N=15, checkpoint ep 90) ever flipped
`use_chunk_replay_origin_path=True`; both recorded `corroboration_deadline_fates`
= all-dissolved in the ACCELERATED regime only. No recorded run carries a
realistic-N or early-checkpoint regime, so the decisive readout is neither
recorded nor derivable (the fate depends on emergent re-execution timing under
a DIFFERENT deadline, which cannot be recomputed from an accelerated-regime
manifest). Not recoverable -> run.

Re-derive brake (Step 2.5b): 0 prior failure_autopsy targets tag MECH-322 with
a substrate_ceiling disposition (873a's autopsy is `standard` / `supports`,
recommended_substrate_queue_entry action=none). Count 0 -> not braked. This is
also a NEW EXQ NUMBER testing a DISTINCT post-mint-lifecycle question, not a
lettered re-pose of 873a's answered AND-gate question -> not the re-derive loop.

Substrate readiness (Step 2.5 + 2.5a). The entire mechanism is confirmed BUILT
and VALIDATED by V3-EXQ-873a itself (7/8 seeds minted correctly). Empirically
re-confirmed: `chunk_replay_corroboration_episodes` is read into
`PolicyChunkingConfig`; `ChunkLibrary.advance_corroboration_deadlines` fires
per episode via `agent.reset()` -> `end_episode()` (ree_core/agent.py:3184-3185);
`note_real_execution` resets the deadline and increments
`crystallisation_counter` (crystallises at CHUNK_CRYSTALLISATION_MIN=2). 873a
dissolved all 7 by deadline at N=15, confirming the deadline advances per
episode in this exact StepHarness eval harness.

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

EXPERIMENT_TYPE = "v3_exq_892_mech322_replay_corroboration_survival"
EXPERIMENT_PURPOSE = "diagnostic"

CLAIM_IDS = ["MECH-322"]

BACKLOG_ID = "EVB-0489"

# 8 seeds, matching V3-EXQ-873a. The readiness base rate (a qualifying real
# high-value candidate emerging by the checkpoint) was ~7/8 in 873a; 8 seeds is
# adequate power for the survival/corroboration fraction this run adjudicates.
# Not a reef config, so the seed-44 instability guidance does not apply.
SEEDS = [101, 202, 303, 404, 505, 606, 707, 808]

# Three arms differing ONLY in the corroboration deadline N.
ARM_DEADLINE_15 = "ARM_DEADLINE_15"    # 873a accelerated regime (negative baseline)
ARM_DEADLINE_75 = "ARM_DEADLINE_75"    # substrate default == realistic condition
ARM_DEADLINE_200 = "ARM_DEADLINE_200"  # deadline beyond run length (never bites)
ARMS = [ARM_DEADLINE_15, ARM_DEADLINE_75, ARM_DEADLINE_200]
ARM_DEADLINE_N = {
    ARM_DEADLINE_15: 15,
    ARM_DEADLINE_75: 75,
    ARM_DEADLINE_200: 200,
}
# The primary / realistic arm whose survival readout answers the headline
# question; also the arm whose readiness gate authorises the load-bearing C1.
PRIMARY_ARM = ARM_DEADLINE_75

N_EPISODES = base.N_EPISODES              # 120, same as V3-EXQ-810a / 873a
STEPS_PER_EPISODE = base.STEPS_PER_EPISODE  # 72, same as V3-EXQ-810a / 873a
# EARLY checkpoint (873a used 90). At episode 20 there are 100 post-mint waking
# episodes for corroboration to occur naturally, and each arm's deadline
# (15 / 75 / 200) lands at ep 35 / 95 / 220 -- so the 15 and 75 arms both bite
# WITHIN the run (observable dissolution) while the 200 arm never bites
# (observable pure recurrence). Late enough that MECH-323 formation has built a
# real value history to draw a qualifying candidate from.
SLEEP_CHECKPOINT_EPISODE = 20
REPLAY_VALUE_QUANTILE = 0.75  # substrate default; not itself under test here

# Readiness (STOCHASTIC): fraction of an arm's seeds that must produce a
# qualifying real high-value candidate by the checkpoint for the arm's gate to
# be adjudicable. Measured = n_cleared / n_seeds; met when fraction > this
# floor. 0.4 over 8 seeds => at least 4 must clear. A floor, indexer-recomputable.
# Identical to V3-EXQ-873a's validated readiness gate.
READINESS_SEED_FRACTION_FLOOR = 0.4
# Plumbing (DETERMINISTIC): among CLEARED seeds only, the share that must mint a
# replay_origin=True chunk. 1.0 -- given a valid candidate + sleep phase + flag
# ON the AND-gate must fire on every such seed (873a confirmed 7/7). This is the
# load-bearing adjudicability gate for the survival readouts.
C1_MINT_FRACTION = 1.0

# Readiness precondition thresholds shared with V3-EXQ-810a / 873a (same env/schedule).
OUTCOME_SPREAD_FLOOR = 0.05
SYMBOLS_PER_TRIAL_FLOOR = float(base.CHUNK_MAX_SIZE - 1)  # 4.0
HARM_STEP_FRACTION_FLOOR = 0.01

_ZG = ZGoalStreamAccumulator()

_NO_CANDIDATE_MARGIN = -1.0e9  # sentinel: no tracked candidate at all this seed


def _shared_config_kwargs(deadline_n: int) -> Dict[str, Any]:
    kw = dict(base.shared_config_kwargs())
    kw["chunk_replay_value_quantile"] = REPLAY_VALUE_QUANTILE
    kw["chunk_replay_corroboration_episodes"] = int(deadline_n)
    # CO-REQUISITE FLAG (same defect V3-EXQ-873a documented): SerotoninModule is
    # itself a default-OFF substrate (SerotoninConfig.tonic_5ht_enabled=False),
    # and when disabled enter_sws()/enter_rem()/exit_sleep() are no-ops that
    # leave `.phase` pinned at "wake" forever. The default in_sleep_phase=None
    # path in note_chunk_replay_sequence reads self.serotonin.phase, so WITHOUT
    # this flag the mint would be refused by the sleep-phase condition on EVERY
    # seed, reading as "carve-out never fires" when the real cause is "the sleep
    # substrate was never switched on". Verified in 873a's own review.
    kw["tonic_5ht_enabled"] = True
    return kw


def _arm_flags() -> Dict[str, Any]:
    flags = base.off_arm_flags()
    flags.update({
        "use_policy_chunking": True,
        "use_chunk_maintenance": True,
        "use_chunk_replay_origin_path": True,
    })
    return flags


def _config_slice(arm: str) -> Dict[str, Any]:
    deadline_n = ARM_DEADLINE_N[arm]
    slice_: Dict[str, Any] = {
        "env": dict(base.ENV_KWARGS),
        "schedule": {
            "n_episodes": N_EPISODES,
            "steps_per_episode": STEPS_PER_EPISODE,
            "sleep_checkpoint_episode": SLEEP_CHECKPOINT_EPISODE,
        },
        # Explicit (redundant with _shared_config_kwargs() below, which sets it
        # under the REEConfig field name) so the config_slice-declaration lint's
        # AST scan of THIS function sees the module-level / per-arm constants the
        # cell's call graph actually reads.
        "replay_corroboration_episodes": deadline_n,
        "replay_value_quantile": REPLAY_VALUE_QUANTILE,
    }
    slice_.update(_shared_config_kwargs(deadline_n))
    slice_.update(_arm_flags())
    return slice_


def _arm_ctx(arm: str) -> Dict[str, Any]:
    return {
        "id": arm,
        "chunking_on": True,
        "steps_per_episode": STEPS_PER_EPISODE,
        "n_seeds": len(SEEDS),
    }


PRECONDITIONS = [
    PreconditionSpec(
        name="episode_outcome_spread_supra_floor",
        description=("MECH-323's evaluative gate is RELATIVE to a running "
                     "baseline, so real chunk formation (the source of the "
                     "replay-mint candidate) is structurally impossible on a "
                     "flat outcome stream. Same statistic as V3-EXQ-810a/873a."),
        control=("worst seed in this arm; the hazard/resource grid's "
                 "per-episode net signal varies"),
        threshold=OUTCOME_SPREAD_FLOOR,
        direction="lower",
        kind="readiness",
        applies_to=lambda ctx: True,
        structural_max=lambda ctx: None,
    ),
    PreconditionSpec(
        name="chunk_buffer_supports_size_range",
        description=("Same buffer-reachability precondition as V3-EXQ-810a/873a: "
                     "a per-trial buffer shorter than chunk_max_size makes the "
                     "tracked-sequence tally too shallow to hold a genuine "
                     "repeated high-value candidate."),
        control="worst seed in this arm; e3_steps_per_tick periodicity",
        threshold=SYMBOLS_PER_TRIAL_FLOOR,
        direction="lower",
        kind="readiness",
        applies_to=lambda ctx: True,
        structural_max=lambda ctx: float(ctx["steps_per_episode"]),
    ),
    PreconditionSpec(
        name="salient_harm_events_fire",
        description=("Task-property check shared with V3-EXQ-810a/873a: confirms "
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
        description=("MECH-322 condition (a)'s readiness precondition, "
                     "aggregated as the FRACTION OF SEEDS in which at least one "
                     "tracked real sub-sequence had an outcome mean clearing "
                     "ChunkAccumulator.replay_value_threshold() by the sleep "
                     "checkpoint. Measured = n_cleared / n_seeds; a floor. It "
                     "certifies the arm is ADJUDICABLE (enough seeds can attempt "
                     "the mint whose fate this run observes); the deterministic "
                     "per-seed mint is C1, scored over cleared seeds only. Same "
                     "validated gate as V3-EXQ-873a."),
        control=("fraction of this arm's seeds that produced a qualifying "
                 "candidate; goes red only when candidates rarely emerge (a "
                 "real substrate/schedule problem), never on one unlucky seed"),
        threshold=READINESS_SEED_FRACTION_FLOOR,
        direction="lower",
        kind="readiness",
        applies_to=lambda ctx: True,
        structural_max=lambda ctx: 1.0,
    ),
]


def _worst(rows: List[Dict[str, Any]], key: str):
    if not rows:
        return 0.0, None
    worst = min(rows, key=lambda r: r[key])
    return float(worst[key]), f"{worst['arm']}/seed{worst['seed']}"


def _clear_fraction(rows: List[Dict[str, Any]]):
    """Fraction of seeds in this arm that produced a qualifying candidate."""
    if not rows:
        return 0.0, 0, None
    cleared = [r for r in rows if r.get("replay_high_value_candidate_available") is True]
    not_cleared = [f"{r['arm']}/seed{r['seed']}"
                   for r in rows if r.get("replay_high_value_candidate_available") is not True]
    frac = len(cleared) / len(rows)
    return frac, len(cleared), (", ".join(not_cleared) if not_cleared else None)


def _run_cell(arm: str, seed: int) -> Dict[str, Any]:
    print(f"Seed {seed} Condition {arm}", flush=True)
    deadline_n = ARM_DEADLINE_N[arm]
    flags = _arm_flags()
    with arm_cell(seed,
                  config_slice=_config_slice(arm),
                  script_path=Path(__file__),
                  config_slice_declared=True,
                  include_driver_script_in_hash=False,
                  extra_ineligible_reasons=["no_off_baseline_arm_all_arms_carveout_on"]) as cell:
        env = base.build_env(seed)
        agent = REEAgent(REEConfig.from_dims(
            body_obs_dim=env.body_obs_dim,
            world_obs_dim=env.world_obs_dim,
            action_dim=4,
            **_shared_config_kwargs(deadline_n),
            **flags,
        ))
        harness = StepHarness(agent, env, train_mode=False, seed=seed)

        episode_outcomes: List[float] = []
        n_harm_steps = 0
        n_steps_total = 0

        # --- mint bookkeeping, filled in at the sleep checkpoint ---
        replay_high_value_candidate_available = False
        replay_candidate_margin = _NO_CANDIDATE_MARGIN
        replay_value_threshold: Optional[float] = None
        high_seq: Optional[Tuple[int, ...]] = None
        high_mu: Optional[float] = None
        success_minted: Optional[bool] = None
        success_minted_replay_origin: Optional[bool] = None
        phase_before_sleep = "wake"
        phase_after_sleep_exit = "wake"

        # --- post-mint lifecycle tracking ---
        first_corroboration_offset: Optional[int] = None  # eps after checkpoint
        n_corroborations_running = 0  # crystallisation_counter observed

        progress_every = max(1, min(30, N_EPISODES // 4))

        for ep in range(N_EPISODES):
            if ep == SLEEP_CHECKPOINT_EPISODE:
                pc = agent.policy_chunking
                replay_value_threshold = (
                    float(pc.accumulator.replay_value_threshold())
                    if pc is not None else None
                )
                # Same candidate-selection logic as V3-EXQ-873a: a real,
                # tracked-but-NOT-YET-REGISTERED sub-sequence (genuine per-episode
                # outcomes credited, but never met the full joint MECH-323
                # formation gate, so no real chunk occupies its key). Reads the
                # accumulator tally directly -- read-only diagnostic access, the
                # same collection note_outcome() reads internally.
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
                    phase_before_sleep = str(agent.serotonin.phase)
                    # Genuine SD-017 SWS entry -- real substrate transition.
                    agent.enter_sws_mode()
                    # MINT: real high-value sequence, genuinely asleep. Default
                    # in_sleep_phase=None reads the real serotonin.phase authority.
                    success_chunk = agent.note_chunk_replay_sequence(
                        high_seq, value_tag=float(high_mu), in_sleep_phase=None
                    )
                    success_minted = success_chunk is not None
                    success_minted_replay_origin = bool(
                        getattr(success_chunk, "replay_origin", False)
                    )
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

            # Post-mint: observe the minted chunk's corroboration counter after
            # this episode's note_outcome pass. crystallisation_counter counts
            # real corroborating executions (frozen once DISSOLVED, retention off),
            # so its first increment marks the first corroboration. Deadline-
            # independent in ARM_DEADLINE_200 (never dissolves within the run).
            if (high_seq is not None and success_minted
                    and ep > SLEEP_CHECKPOINT_EPISODE):
                pc = agent.policy_chunking
                if pc is not None:
                    chunk = pc.library.get(high_seq)
                    if chunk is not None:
                        cc = int(chunk.crystallisation_counter)
                        if cc > n_corroborations_running:
                            n_corroborations_running = cc
                            if first_corroboration_offset is None:
                                first_corroboration_offset = ep - SLEEP_CHECKPOINT_EPISODE

            if (ep + 1) % progress_every == 0:
                st_now = agent.get_chunking_state()
                print(f"  [train] chunk {arm} seed={seed} ep {ep+1}/{N_EPISODES} "
                      f"replay_formed={st_now.get('chunk_acc_n_replay_formed', 0)} "
                      f"corrob={n_corroborations_running}",
                      flush=True)

        st = agent.get_chunking_state()

        # --- authoritative end-of-run fate of the minted chunk ---
        minted_chunk_state = None
        minted_episodes_since_corroboration = None
        minted_n_dissolutions = None
        minted_n_corroborations = None
        minted_survived = None
        minted_crystallised = None
        pc = agent.policy_chunking
        if pc is not None and high_seq is not None and success_minted:
            chunk = pc.library.get(high_seq)
            if chunk is not None:
                minted_chunk_state = chunk.state.value
                minted_episodes_since_corroboration = int(chunk.episodes_since_corroboration)
                minted_n_dissolutions = int(chunk.n_dissolutions)
                minted_n_corroborations = int(chunk.crystallisation_counter)
                minted_survived = chunk.state.value != "dissolved"
                minted_crystallised = chunk.state.value in ("crystallised", "dissolving")

        n_symbols = int(st.get("chunk_acc_n_steps", 0))
        n_outcomes = int(st.get("chunk_acc_n_outcomes", 0))

        row = {
            "arm": arm, "seed": seed, "deadline_n": deadline_n,
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
            # readiness / mint
            "replay_high_value_candidate_available": replay_high_value_candidate_available,
            "replay_candidate_margin": replay_candidate_margin,
            "replay_value_threshold": replay_value_threshold,
            "high_seq_mu": float(high_mu) if high_mu is not None else None,
            "high_seq_len": len(high_seq) if high_seq is not None else 0,
            "success_minted": success_minted,
            "success_minted_replay_origin": success_minted_replay_origin,
            "phase_before_sleep": phase_before_sleep,
            "phase_after_sleep_exit": phase_after_sleep_exit,
            # post-mint survival readouts (the headline DVs)
            "minted_chunk_state_end_of_run": minted_chunk_state,
            "minted_survived": minted_survived,
            "minted_crystallised": minted_crystallised,
            "minted_n_corroborations_end_of_run": minted_n_corroborations,
            "minted_first_corroboration_offset": first_corroboration_offset,
            "minted_episodes_since_corroboration_end_of_run": minted_episodes_since_corroboration,
            "minted_n_dissolutions_end_of_run": minted_n_dissolutions,
            # task-property readiness stats
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

    # Cell PASSes iff it either legitimately minted (cleared readiness), OR
    # legitimately did not attempt because no candidate emerged this seed
    # (readiness not cleared) -- the latter is a correct fail-closed, not a cell
    # defect. Mirrors V3-EXQ-873a's per-cell verdict.
    if replay_high_value_candidate_available:
        cell_ok = bool(row["success_minted"] and row["success_minted_replay_origin"])
    else:
        cell_ok = row["success_minted"] is not True
    print(f"verdict: {'PASS' if cell_ok else 'FAIL'}", flush=True)
    return row


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

    # C1 is authorised only when the PRIMARY (realistic) arm's own gate is green.
    c1_arm_green = gate_by_arm.get(PRIMARY_ARM, False)

    # --- C1 (LOAD-BEARING, DETERMINISTIC PLUMBING, adjudicability gate for the
    # survival readouts): among the PRIMARY arm's seeds that CLEARED readiness,
    # the fraction that minted a replay_origin=True chunk. A no-candidate seed is
    # EXCLUDED from the denominator (873a fix). If the mint does not fire, there
    # is no chunk whose survival to observe.
    cleared_primary = [r for r in by_arm[PRIMARY_ARM]
                       if r["replay_high_value_candidate_available"] is True]
    c1_minted = sum(1 for r in cleared_primary
                    if r["success_minted"] and r["success_minted_replay_origin"])
    c1_pass = bool(cleared_primary) and (c1_minted / len(cleared_primary)) >= C1_MINT_FRACTION

    # Minted rows per arm (a mint fired -- the population whose fate we observe).
    def _minted(rows_: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [r for r in rows_ if r.get("success_minted") is True]

    minted_by_arm = {a: _minted(by_arm[a]) for a in ARMS}
    all_minted = [r for a in ARMS for r in minted_by_arm[a]]

    # --- C2 (INSTRUMENT CORRECTNESS, LOAD-BEARING): every minted chunk has a
    # trackable end-state (fate is observable) AND its audit counters are
    # internally consistent (a corroborated chunk carries >=1 replay_origin lib
    # slot; a survived chunk is not DISSOLVED). If this fails the survival
    # measurement is not readable at all.
    c2_trackable = all(r["minted_chunk_state_end_of_run"] is not None for r in all_minted) \
        if all_minted else True
    c2_consistent = all(
        (r["minted_survived"] is (r["minted_chunk_state_end_of_run"] != "dissolved"))
        and (r["chunk_lib_n_replay_origin"] >= 1)
        for r in all_minted
    ) if all_minted else True
    c2_pass = bool(c2_trackable and c2_consistent)

    # --- descriptive survival / corroboration readouts (the actual science;
    # not pass/fail gates -- a null survival result is a valid finding) ---
    def _corroborated(r: Dict[str, Any]) -> bool:
        v = r.get("minted_n_corroborations_end_of_run")
        return isinstance(v, int) and v >= 1

    def _survived(r: Dict[str, Any]) -> bool:
        return r.get("minted_survived") is True

    corroboration_rate_by_arm = {
        a: (sum(1 for r in minted_by_arm[a] if _corroborated(r)) / len(minted_by_arm[a])
            if minted_by_arm[a] else None)
        for a in ARMS
    }
    survival_rate_by_arm = {
        a: (sum(1 for r in minted_by_arm[a] if _survived(r)) / len(minted_by_arm[a])
            if minted_by_arm[a] else None)
        for a in ARMS
    }
    deadline_dissolution_count_by_arm = {
        a: sum(1 for r in minted_by_arm[a]
               if (r.get("minted_n_dissolutions_end_of_run") or 0) >= 1
               and not _corroborated(r))
        for a in ARMS
    }
    crystallised_count_by_arm = {
        a: sum(1 for r in minted_by_arm[a] if r.get("minted_crystallised") is True)
        for a in ARMS
    }
    # Deadline-INDEPENDENT exposure: recurrence counts in the never-dissolving arm.
    exposure_recurrence_n200 = sorted(
        int(r["minted_n_corroborations_end_of_run"])
        for r in minted_by_arm[ARM_DEADLINE_200]
        if isinstance(r.get("minted_n_corroborations_end_of_run"), int)
    )
    first_corroboration_offsets_n200 = sorted(
        int(r["minted_first_corroboration_offset"])
        for r in minted_by_arm[ARM_DEADLINE_200]
        if isinstance(r.get("minted_first_corroboration_offset"), int)
    )

    # Any corroboration observed anywhere (the existence readout).
    any_corroboration_observed = any(_corroborated(r) for r in all_minted)
    # Survival under the realistic default deadline.
    realistic_survival_rate = survival_rate_by_arm.get(PRIMARY_ARM)
    realistic_survived_any = bool(realistic_survival_rate) if realistic_survival_rate else False

    overall_pass = bool(c1_arm_green and c1_pass and c2_pass)

    if not c1_arm_green:
        label = "substrate_not_ready_requeue"
    elif not c1_pass:
        label = "replay_mint_silent_under_valid_conditions"
    elif not c2_pass:
        label = "replay_chunk_fate_untrackable_instrument_defect"
    elif any_corroboration_observed and realistic_survived_any:
        label = "replay_corroboration_survives_under_realistic_conditions"
    elif any_corroboration_observed:
        label = "replay_corroboration_observed_but_misses_realistic_deadline"
    else:
        label = "replay_corroboration_absent_mint_eligible_sequences_do_not_recur"

    metrics = {
        "c1_pass": c1_pass, "c2_pass": c2_pass,
        "gate_c1_arm_green": bool(c1_arm_green),
        "gate_any_arm_green": bool(agg["non_degenerate"]),
        "gate_green_by_arm": gate_by_arm,
        "primary_arm": PRIMARY_ARM,
        "n_seeds": len(SEEDS),
        "arm_deadline_n": ARM_DEADLINE_N,
        "sleep_checkpoint_episode": SLEEP_CHECKPOINT_EPISODE,
        "post_checkpoint_waking_episodes": N_EPISODES - SLEEP_CHECKPOINT_EPISODE,
        # readiness / plumbing
        "readiness_seed_fraction_by_arm": readiness_frac_by_arm,
        "n_cleared_seeds_by_arm": n_cleared_by_arm,
        "readiness_seed_fraction_floor": READINESS_SEED_FRACTION_FLOOR,
        "c1_cleared_seed_count": len(cleared_primary),
        "c1_minted_cleared_count": c1_minted,
        "c1_mint_fraction": (c1_minted / len(cleared_primary)) if cleared_primary else None,
        "c1_mint_fraction_floor": C1_MINT_FRACTION,
        "n_minted_by_arm": {a: len(minted_by_arm[a]) for a in ARMS},
        # --- the actual science: survival + corroboration ---
        "any_corroboration_observed": any_corroboration_observed,
        "corroboration_rate_by_arm": corroboration_rate_by_arm,
        "survival_rate_by_arm": survival_rate_by_arm,
        "realistic_survival_rate": realistic_survival_rate,
        "deadline_dissolution_count_by_arm": deadline_dissolution_count_by_arm,
        "crystallised_count_by_arm": crystallised_count_by_arm,
        # exposure covariate (deadline-independent, from the never-dissolving arm)
        "exposure_recurrence_counts_n200": exposure_recurrence_n200,
        "first_corroboration_offsets_n200": first_corroboration_offsets_n200,
        "replay_formed_total": sum(r["chunk_acc_n_replay_formed"] for r in rows),
        "replay_origin_total": sum(r["chunk_lib_n_replay_origin"] for r in rows),
        "replay_deadline_dissolutions_total": sum(
            r["chunk_lib_n_replay_deadline_dissolutions"] for r in rows),
    }

    criteria_by_arm = {a: ["C1_mint_fires_under_valid_conditions",
                           "C2_minted_chunk_fate_trackable"] for a in ARMS}
    criteria_non_degenerate = arm_criteria_non_degenerate(
        criteria_by_arm, agg,
        extra={
            "C1_mint_fires_under_valid_conditions": c1_arm_green and len(cleared_primary) > 0,
            "C2_minted_chunk_fate_trackable": len(all_minted) > 0,
        })

    interpretation = {
        "label": label,
        "preconditions": agg["adjudication_preconditions"],
        "criteria": [
            {"name": "C1_mint_fires_under_valid_conditions", "load_bearing": True,
             "passed": c1_pass},
            {"name": "C2_minted_chunk_fate_trackable", "load_bearing": True,
             "passed": c2_pass},
        ],
        "criteria_non_degenerate": criteria_non_degenerate,
        "per_arm_gate": agg["per_arm_gate"],
        "gate_offending_cells": {g["arm"]: g["offending_cells"] for g in arm_gates},
        "survival_readout": {
            "any_corroboration_observed": any_corroboration_observed,
            "realistic_deadline_n": ARM_DEADLINE_N[PRIMARY_ARM],
            "realistic_survival_rate": realistic_survival_rate,
            "corroboration_rate_by_arm": corroboration_rate_by_arm,
            "survival_rate_by_arm": survival_rate_by_arm,
            "exposure_recurrence_counts_n200": exposure_recurrence_n200,
        },
        "both_directions": (
            "DESCRIPTIVE (diagnostic, excluded from confidence scoring). A "
            "positive reading (corroboration observed + survival under the "
            "realistic N=75 deadline) demonstrates the replay chunk's full "
            "lifecycle -- mint -> corroborate -> survive -- and STRENGTHENS the "
            "MECH-322 carve-out characterisation. A null reading (no minted "
            "chunk corroborated before its deadline in any arm) is EQUALLY "
            "valid and does NOT weaken MECH-322: it shows the accelerated-"
            "dissolution safety valve firing exactly as designed, and localises "
            "the survival corner to the trailing-match recurrence condition. "
            "Neither direction is a claim failure; the run measures WHICH corner "
            "of the exposure space permits survival."),
        "combination_rule": (
            "overall PASS = PRIMARY arm (ARM_DEADLINE_75) gate green (readiness: "
            f">= {READINESS_SEED_FRACTION_FLOOR:.2f} of seeds produced a "
            f"qualifying candidate) AND C1 (>= {C1_MINT_FRACTION:.2f} of CLEARED "
            "seeds minted replay_origin) AND C2 (every minted chunk's fate is "
            "trackable and its audit counters consistent). PASS means the "
            "survival question was successfully MEASURED; the ANSWER (survives / "
            "does-not-survive) is in survival_readout and interpretation.label, "
            "both directions valid."),
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
        "any_corroboration_observed": any_corroboration_observed,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    t0 = time.perf_counter()

    global SEEDS, N_EPISODES, STEPS_PER_EPISODE, SLEEP_CHECKPOINT_EPISODE
    global ARM_DEADLINE_N
    if args.dry_run:
        SEEDS = [101]
        N_EPISODES = 8
        STEPS_PER_EPISODE = 72
        SLEEP_CHECKPOINT_EPISODE = 2
        # Keep the three arms distinct but small: one bites early, one bites
        # late, one never bites within the 6 post-checkpoint episodes.
        ARM_DEADLINE_N = {
            ARM_DEADLINE_15: 2,
            ARM_DEADLINE_75: 4,
            ARM_DEADLINE_200: 100,
        }

    result = run_experiment()
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    full_config = {
        "seeds": SEEDS, "arms": ARMS,
        "n_episodes": N_EPISODES,
        "steps_per_episode": STEPS_PER_EPISODE,
        "sleep_checkpoint_episode": SLEEP_CHECKPOINT_EPISODE,
        "arm_deadline_n": ARM_DEADLINE_N,
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
        "evidence_direction": (
            "supports" if (result["outcome"] == "PASS"
                           and result["any_corroboration_observed"]) else "unknown"),
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
            "enter_sws_mode()/exit_sleep_mode() called directly once per cell "
            "at the sleep checkpoint episode; no SD-017 schema/REM passes run)"
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
    print(f"C1={m['c1_pass']} C2={m['c2_pass']} "
          f"primary_gate_green={m['gate_c1_arm_green']}", flush=True)
    print(f"readiness_seed_fraction_by_arm={m['readiness_seed_fraction_by_arm']} "
          f"(floor={m['readiness_seed_fraction_floor']})", flush=True)
    print(f"n_minted_by_arm={m['n_minted_by_arm']}", flush=True)
    print(f"any_corroboration_observed={m['any_corroboration_observed']}", flush=True)
    print(f"corroboration_rate_by_arm={m['corroboration_rate_by_arm']}", flush=True)
    print(f"survival_rate_by_arm={m['survival_rate_by_arm']}", flush=True)
    print(f"deadline_dissolution_count_by_arm={m['deadline_dissolution_count_by_arm']}",
          flush=True)
    print(f"exposure_recurrence_counts_n200={m['exposure_recurrence_counts_n200']}",
          flush=True)
    print(f"first_corroboration_offsets_n200={m['first_corroboration_offsets_n200']}",
          flush=True)
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
