"""V3-EXQ-896 -- MECH-322 sleep-replay carve-out, CONFIRMATORY evidence-purpose run.

SLEEP DRIVER: manual-cycle-loop (single wake-sleep-wake cycle, N_CYCLES=1;
enter_sws_mode()/exit_sleep_mode() called directly once per cell, at
SLEEP_CHECKPOINT_EPISODE; no SD-017 schema-installation or REM slot-fill
passes are run -- only the MECH-322 mint + its post-mint lifecycle are
exercised).

GOV-CONFIRM-1 confirming run for MECH-322 (policy.composition.sleep_replay_
value_conditioned_consolidation), IGW-20260808-226. lit_conf=0.76 but
genuine_exp_count=0 in claim_evidence.v1.json: MECH-322's three prior touch-
points (V3-EXQ-873 mixed/FAIL, 873a PASS/supports, 892 PASS/supports) are ALL
EXPERIMENT_PURPOSE="diagnostic" and are therefore structurally EXCLUDED from
governance confidence/conflict scoring (queue-experiment SKILL.md: "Diagnostic
and baseline experiments are excluded ... they appear in the evidence record
as context only"). The IGW brief's "ZERO experimental evidence" is correct in
exactly that governance-scoring sense, not a stale read: three diagnostic runs
exist and are supportive, but none of them COUNTS. This run is the first
EXPERIMENT_PURPOSE="evidence" confirmer for MECH-322.

WHAT THIS TESTS (wall-independent representational/functional signature, per
the GOV-CONFIRM-1 lane -- NOT a behavioural/reward-competence proxy). The
claim's own falsifiable prediction (functional_restatement in claims.yaml) is
an OVERNIGHT-IMPROVEMENT behavioural latency effect, which is competence-wall-
bound (needs sustained multi-block task performance) and is EXPLICITLY not
what this run attempts -- per the IGW brief's own self-route instruction
("self-route substrate_not_ready_requeue if only a behavioural confirmer is
available"), this run instead confirms the two REPRESENTATION/FUNCTIONAL-
SIGNATURE pieces the mechanism itself asserts, at the evidence-scoring tier:

  (1) SAFETY: with the carve-out flag OFF (MECH-094 strict), a replayed
      sequence -- fed the identical value-tag and sleep-phase context that
      WOULD mint under the carve-out -- is refused outright. This is the
      register-scoring confirmation that MECH-094 strict gating remains
      intact when MECH-322 is not deliberately enabled; no prior run has
      measured this on a MATCHED (same-seed, same-candidate) OFF-vs-ON pair.
  (2) FUNCTION: with the carve-out flag ON, a chunk minted from a value-
      tagged, sleep-phase-gated replayed sequence achieves real-execution
      corroboration and SURVIVES to end-of-run at a rate clearing a pre-
      registered, non-vacuous floor under the REALISTIC (substrate-default)
      corroboration deadline. This is a genuine confirmatory replication of
      V3-EXQ-892's diagnostic finding (realistic_survival_rate = 0.571, n=7
      minted chunks, 8 seeds) at evidence-purpose with a pre-registered
      threshold and a fresh, larger seed set for power.

Neither (1) nor (2) has been jointly measured before: 873a measured OFF-vs-ON
MINT RATE only, at the ACCELERATED N=15 deadline (checkpoint ep 90, only 30
post-mint episodes -- deadline bites before any realistic corroboration window
exists). 892 measured realistic-deadline SURVIVAL only, with no OFF arm (its
own docstring: "There is no OFF/baseline arm ... those are V3-EXQ-873a's
job"). GOV-REUSE-1 (Step 2.4): decisive readout = OFF-vs-ON contrast under the
realistic N=75 deadline with pre-registered survival floor. Searched the
MECH-322 manifest corpus (873, 873a, 892): no manifest carries both an OFF
comparator AND a realistic-deadline post-checkpoint window. Not recoverable
from existing data -> run.

SCOPE / honesty note: this confirms the BUILT gate + post-mint lifecycle
substrate behaves as specified under a single hand-selected (highest-value)
replay candidate. It does NOT test that a full task naturally generates such
candidates end-to-end, and it does NOT test the claim's overnight-latency-
improvement behavioural prediction -- that remains a downstream, competence-
wall-bound test outside the GOV-CONFIRM-1 lane.

DESIGN. Two arms, differing ONLY in `use_chunk_replay_origin_path`:

  ARM_STRICT_FLAG_OFF  -- chunking ON (use_policy_chunking=True), carve-out
                          flag OFF. MECH-094 strict; the safety comparator.
  ARM_REPLAY_ON         -- chunking ON, carve-out flag ON. The claim under
                          test.

Both otherwise identical: same env, same schedule, same sleep checkpoint
(episode 20 of 120 -> 100 post-mint waking episodes), same realistic
corroboration deadline (N=75, the substrate default). The manipulation never
touches pre-checkpoint policy (deadline/flag only gate what happens AT and
AFTER the checkpoint), so the SAME seed reaches the SAME candidate pool in
both arms; only the outcome of the mint ATTEMPT and its downstream lifecycle
differ.

DV-SYMMETRY (Step 3 mandatory declaration). ARM_STRICT_FLAG_OFF's load-bearing
DV is `success_minted` (must be False on every attempted cell -- a refusal-
rate step function of the flag, not invariant under any symmetry: the flag is
checked FIRST in `record_replay_sequence`'s AND-chain, so it is a direct,
unconfounded gate on the outcome). ARM_REPLAY_ON's load-bearing DVs are
`success_minted_replay_origin` (mint fires) and `minted_survived` (post-mint
lifecycle fate) -- a deadline-gated step function of real-execution
recurrence timing, identical construction to V3-EXQ-892's DV-symmetry
argument (which this run inherits unchanged, same deadline N=75). No symmetry
group renders either manipulation invisible to its own DV.

Re-derive brake (Step 2.5b): 0 prior failure_autopsy targets tag MECH-322 with
a substrate_ceiling disposition (873's autopsy: measurement_test_design_defect,
action=none; 873a's: standard/supports, action=none; 892's: descriptive
closure, action=none). Count 0 -> not braked.

Substrate-path overlap gate (Step 2.5c): no OPEN `corrupting`-severity
substrate_queue.json entry declares substrate_paths touching
ree_core/policy/policy_chunking.py or the chunk_* config surface.

Substrate readiness (Step 2.5 + 2.5a). BUILT and VALIDATED three times over
(873/873a/892). Empirically re-confirmed here by direct source read:
`ChunkAccumulator.record_replay_sequence` (ree_core/policy/policy_chunking.py)
checks `use_chunk_replay_origin_path` FIRST (unconditional refusal when False,
before the value-tag or sleep-phase checks are even evaluated) -- so
ARM_STRICT_FLAG_OFF's refusal is guaranteed structural, not a coincidence of
this run's candidate values. `ChunkLibrary.advance_corroboration_deadlines`
and `note_real_execution` (crystallisation_counter, deadline reset) are the
same call sites 892 empirically confirmed fire per-episode via
`agent.reset()` -> `end_episode()`.

Ethics preflight (Step 2.6, all-false/allow -- standard V3 boilerplate,
unchanged from 873/873a/892):
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

EXPERIMENT_TYPE = "v3_exq_896_mech322_replay_confirmatory_evidence"
QUEUE_ID = "V3-EXQ-896"
EXPERIMENT_PURPOSE = "evidence"

CLAIM_IDS = ["MECH-322"]

# 12 seeds (up from 873a/892's 8) for power on the pre-registered survival-rate
# floor (C3), estimated over the ~10-11 seeds expected to mint. Fresh seeds,
# distinct from 873/873a/892's [101..808], for genuine replication rather than
# re-scoring the same samples. Not a reef config (892's own note applies
# identically here: the seed-44 instability guidance is reef-config-specific).
SEEDS = [111, 222, 333, 444, 555, 666, 777, 888, 999, 1010, 1111, 1212]

ARM_STRICT_FLAG_OFF = "ARM_STRICT_FLAG_OFF"  # MECH-094 strict; safety comparator
ARM_REPLAY_ON = "ARM_REPLAY_ON"              # MECH-322 carve-out active
ARMS = [ARM_STRICT_FLAG_OFF, ARM_REPLAY_ON]

N_EPISODES = base.N_EPISODES               # 120, matches 810a/873a/892
STEPS_PER_EPISODE = base.STEPS_PER_EPISODE  # 72, matches 810a/873a/892
# EARLY checkpoint (892's design, not 873a's ep-90): 100 post-mint waking
# episodes gives the realistic deadline (75) room to bite WITHIN the run, so
# survival is actually observable rather than truncated by run end.
SLEEP_CHECKPOINT_EPISODE = 20
REPLAY_VALUE_QUANTILE = 0.75          # substrate default; not under test here
REPLAY_CORROBORATION_EPISODES = 75    # substrate default = the realistic deadline

# Readiness (STOCHASTIC): fraction of an arm's seeds that must produce a
# qualifying real high-value candidate by the checkpoint for the arm's gate to
# be adjudicable. Identical validated floor to V3-EXQ-873a/892.
READINESS_SEED_FRACTION_FLOOR = 0.4

# --- pre-registered CONFIRMATORY thresholds (Step 3, fixed before running) ---
# C1 (SAFETY, load-bearing): among ARM_STRICT_FLAG_OFF's cleared seeds (a
# qualifying candidate existed), the mint attempt must be refused on EVERY
# seed. mint_fraction == 0.0 exactly -- the flag is checked first in the
# AND-chain, so any non-zero mint here would be a genuine MECH-094 breach.
C1_STRICT_MINT_FRACTION_CEILING = 0.0
# C2 (MECHANISM FIRES, load-bearing, replication): among ARM_REPLAY_ON's
# cleared seeds, the fraction that mint a replay_origin=True chunk. Matches
# 873a's validated 7/7 and 892's validated 7/7.
C2_REPLAY_MINT_FRACTION_FLOOR = 1.0
# C3 (HEADLINE CONFIRMATORY, load-bearing, NOVEL): among ARM_REPLAY_ON's
# minted chunks, the fraction SURVIVING to end-of-run (state != DISSOLVED)
# under the realistic N=75 deadline. Pre-registered at 0.30 -- conservative
# relative to 892's diagnostic prior (0.571, n=7) so the test retains real
# power to fail, while remaining clearly non-vacuous (well above the
# ARM_DEADLINE_15 accelerated-regime floor of 0.0 that 873/873a measured).
C3_SURVIVAL_RATE_FLOOR = 0.30

# Readiness precondition thresholds shared with V3-EXQ-810a/873a/892 (same
# env/schedule).
OUTCOME_SPREAD_FLOOR = 0.05
SYMBOLS_PER_TRIAL_FLOOR = float(base.CHUNK_MAX_SIZE - 1)  # 4.0
HARM_STEP_FRACTION_FLOOR = 0.01

_ZG = ZGoalStreamAccumulator()

_NO_CANDIDATE_MARGIN = -1.0e9  # sentinel: no tracked candidate at all this seed


def _shared_config_kwargs() -> Dict[str, Any]:
    kw = dict(base.shared_config_kwargs())
    kw["chunk_replay_value_quantile"] = REPLAY_VALUE_QUANTILE
    kw["chunk_replay_corroboration_episodes"] = REPLAY_CORROBORATION_EPISODES
    # CO-REQUISITE FLAG (same defect V3-EXQ-873a/892 documented): SerotoninModule
    # is itself default-OFF (SerotoninConfig.tonic_5ht_enabled=False), and when
    # disabled enter_sws()/exit_sleep() are no-ops leaving `.phase` pinned at
    # "wake" forever. The default in_sleep_phase=None path in
    # note_chunk_replay_sequence reads self.serotonin.phase, so without this
    # flag EVERY mint attempt (including ARM_REPLAY_ON's) would be refused by
    # the sleep-phase condition regardless of the flag under test.
    kw["tonic_5ht_enabled"] = True
    return kw


def _arm_flags(arm: str) -> Dict[str, Any]:
    flags = base.off_arm_flags()
    flags.update({
        "use_policy_chunking": True,
        "use_chunk_maintenance": True,
        "use_chunk_replay_origin_path": arm == ARM_REPLAY_ON,
    })
    return flags


def _config_slice(arm: str) -> Dict[str, Any]:
    slice_: Dict[str, Any] = {
        "env": dict(base.ENV_KWARGS),
        "schedule": {
            "n_episodes": N_EPISODES,
            "steps_per_episode": STEPS_PER_EPISODE,
            "sleep_checkpoint_episode": SLEEP_CHECKPOINT_EPISODE,
        },
        # Explicit (redundant with _shared_config_kwargs() below, which sets it
        # under the REEConfig field name) so the config_slice-declaration
        # lint's AST scan of THIS function sees the module-level constants the
        # cell's call graph actually reads.
        "replay_corroboration_episodes": REPLAY_CORROBORATION_EPISODES,
        "replay_value_quantile": REPLAY_VALUE_QUANTILE,
    }
    slice_.update(_shared_config_kwargs())
    slice_.update(_arm_flags(arm))
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
                     "flat outcome stream. Same statistic as V3-EXQ-810a/873a/892."),
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
        description=("Same buffer-reachability precondition as V3-EXQ-810a/"
                     "873a/892: a per-trial buffer shorter than chunk_max_size "
                     "makes the tracked-sequence tally too shallow to hold a "
                     "genuine repeated high-value candidate."),
        control="worst seed in this arm; e3_steps_per_tick periodicity",
        threshold=SYMBOLS_PER_TRIAL_FLOOR,
        direction="lower",
        kind="readiness",
        applies_to=lambda ctx: True,
        structural_max=lambda ctx: float(ctx["steps_per_episode"]),
    ),
    PreconditionSpec(
        name="salient_harm_events_fire",
        description=("Task-property check shared with V3-EXQ-810a/873a/892: "
                     "confirms the hazard-bearing grid is genuinely exercised, "
                     "so an arm's gate is never vacuously green from zero "
                     "applied preconditions."),
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
                     "the mint whose outcome this run scores). Same validated "
                     "gate as V3-EXQ-873a/892."),
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
    flags = _arm_flags(arm)
    with arm_cell(seed,
                  config_slice=_config_slice(arm),
                  script_path=Path(__file__),
                  config_slice_declared=True,
                  include_driver_script_in_hash=False,
                  extra_ineligible_reasons=[
                      "no_off_baseline_arm_both_arms_chunking_on"]) as cell:
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

        # --- post-mint lifecycle tracking (ARM_REPLAY_ON only; the mint never
        # fires in ARM_STRICT_FLAG_OFF so there is nothing to track there) ---
        first_corroboration_offset: Optional[int] = None
        n_corroborations_running = 0

        progress_every = max(1, min(30, N_EPISODES // 4))

        for ep in range(N_EPISODES):
            if ep == SLEEP_CHECKPOINT_EPISODE:
                pc = agent.policy_chunking
                replay_value_threshold = (
                    float(pc.accumulator.replay_value_threshold())
                    if pc is not None else None
                )
                # Same candidate-selection logic as V3-EXQ-873a/892: a real,
                # tracked-but-NOT-YET-REGISTERED sub-sequence (genuine per-
                # episode outcomes credited, but never met the full joint
                # MECH-323 formation gate, so no real chunk occupies its key).
                # Read-only diagnostic access.
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
                    # Genuine SD-017 SWS entry -- real substrate transition,
                    # identical in BOTH arms (the flag under test gates the
                    # mint call below, not sleep entry itself).
                    agent.enter_sws_mode()
                    # MINT ATTEMPT: same candidate, same value-tag, same
                    # sleep-phase context in both arms. ARM_STRICT_FLAG_OFF
                    # must be refused (use_chunk_replay_origin_path=False is
                    # checked first in the AND-chain); ARM_REPLAY_ON may mint.
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

            # Post-mint: observe the minted chunk's corroboration counter
            # (ARM_REPLAY_ON only -- success_minted is never True in
            # ARM_STRICT_FLAG_OFF, so this block is a no-op there by
            # construction, not by an explicit arm check).
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

        # --- authoritative end-of-run fate of the minted chunk (ARM_REPLAY_ON) ---
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
            # post-mint survival readouts (ARM_REPLAY_ON headline DVs)
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

    # Cell PASSes iff: (a) no candidate emerged this seed (readiness not
    # cleared -- a correct fail-closed non-attempt, not a cell defect), or
    # (b) a candidate emerged AND the arm's structural outcome held:
    # ARM_STRICT_FLAG_OFF must NOT mint; ARM_REPLAY_ON must mint replay_origin.
    if not replay_high_value_candidate_available:
        cell_ok = row["success_minted"] is not True
    elif arm == ARM_STRICT_FLAG_OFF:
        cell_ok = row["success_minted"] is not True
    else:
        cell_ok = bool(row["success_minted"] and row["success_minted_replay_origin"])
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

    # Both arms' gates must be green: each is independently adjudicable (same
    # readiness preconditions, same env/schedule); a red gate on EITHER means
    # that arm's cleared-seed population is not trustworthy for its own
    # criterion (C1 needs ARM_STRICT_FLAG_OFF green; C2/C3 need ARM_REPLAY_ON
    # green).
    strict_gate_green = gate_by_arm.get(ARM_STRICT_FLAG_OFF, False)
    replay_gate_green = gate_by_arm.get(ARM_REPLAY_ON, False)

    cleared_strict = [r for r in by_arm[ARM_STRICT_FLAG_OFF]
                      if r["replay_high_value_candidate_available"] is True]
    cleared_replay = [r for r in by_arm[ARM_REPLAY_ON]
                      if r["replay_high_value_candidate_available"] is True]

    # --- C1 (SAFETY, load-bearing, DETERMINISTIC): among ARM_STRICT_FLAG_OFF's
    # cleared seeds, the mint attempt must be refused on every one.
    strict_minted = sum(1 for r in cleared_strict if r["success_minted"])
    strict_mint_fraction = (strict_minted / len(cleared_strict)) if cleared_strict else None
    c1_pass = bool(strict_gate_green and cleared_strict
                   and strict_mint_fraction is not None
                   and strict_mint_fraction <= C1_STRICT_MINT_FRACTION_CEILING)

    # --- C2 (MECHANISM FIRES, load-bearing, DETERMINISTIC): among
    # ARM_REPLAY_ON's cleared seeds, the fraction that mint replay_origin=True.
    replay_minted = sum(1 for r in cleared_replay
                        if r["success_minted"] and r["success_minted_replay_origin"])
    replay_mint_fraction = (replay_minted / len(cleared_replay)) if cleared_replay else None
    c2_pass = bool(replay_gate_green and cleared_replay
                   and replay_mint_fraction is not None
                   and replay_mint_fraction >= C2_REPLAY_MINT_FRACTION_FLOOR)

    # Minted rows in ARM_REPLAY_ON (the population whose post-mint fate is scored).
    minted_replay = [r for r in by_arm[ARM_REPLAY_ON] if r.get("success_minted") is True]

    # --- C3 (HEADLINE CONFIRMATORY, load-bearing, pre-registered floor): the
    # fraction of minted ARM_REPLAY_ON chunks that SURVIVE to end-of-run
    # (state != DISSOLVED) under the realistic N=75 deadline.
    def _survived(r: Dict[str, Any]) -> bool:
        return r.get("minted_survived") is True

    def _corroborated(r: Dict[str, Any]) -> bool:
        v = r.get("minted_n_corroborations_end_of_run")
        return isinstance(v, int) and v >= 1

    n_survived = sum(1 for r in minted_replay if _survived(r))
    survival_rate = (n_survived / len(minted_replay)) if minted_replay else None
    corroboration_rate = (
        sum(1 for r in minted_replay if _corroborated(r)) / len(minted_replay)
        if minted_replay else None
    )
    c3_pass = bool(c2_pass and minted_replay
                   and survival_rate is not None
                   and survival_rate >= C3_SURVIVAL_RATE_FLOOR)

    overall_pass = bool(c1_pass and c2_pass and c3_pass)

    if not (strict_gate_green and replay_gate_green):
        label = "substrate_not_ready_requeue"
    elif not c1_pass:
        label = "mech094_strict_gate_breach_under_matched_conditions"
    elif not c2_pass:
        label = "replay_mint_silent_under_valid_conditions"
    elif not c3_pass:
        label = "replay_corroboration_survival_below_preregistered_floor"
    else:
        label = "mech322_confirmed_safety_and_functional_signature"

    metrics = {
        "c1_pass": c1_pass, "c2_pass": c2_pass, "c3_pass": c3_pass,
        "gate_strict_arm_green": bool(strict_gate_green),
        "gate_replay_arm_green": bool(replay_gate_green),
        "gate_any_arm_green": bool(agg["non_degenerate"]),
        "gate_green_by_arm": gate_by_arm,
        "n_seeds": len(SEEDS),
        "sleep_checkpoint_episode": SLEEP_CHECKPOINT_EPISODE,
        "post_checkpoint_waking_episodes": N_EPISODES - SLEEP_CHECKPOINT_EPISODE,
        "replay_corroboration_episodes": REPLAY_CORROBORATION_EPISODES,
        # readiness / plumbing
        "readiness_seed_fraction_by_arm": readiness_frac_by_arm,
        "n_cleared_seeds_by_arm": n_cleared_by_arm,
        "readiness_seed_fraction_floor": READINESS_SEED_FRACTION_FLOOR,
        # C1 (safety)
        "c1_strict_cleared_seed_count": len(cleared_strict),
        "c1_strict_minted_count": strict_minted,
        "c1_strict_mint_fraction": strict_mint_fraction,
        "c1_strict_mint_fraction_ceiling": C1_STRICT_MINT_FRACTION_CEILING,
        # C2 (mechanism fires)
        "c2_replay_cleared_seed_count": len(cleared_replay),
        "c2_replay_minted_count": replay_minted,
        "c2_replay_mint_fraction": replay_mint_fraction,
        "c2_replay_mint_fraction_floor": C2_REPLAY_MINT_FRACTION_FLOOR,
        # C3 (headline: functional signature)
        "c3_n_minted": len(minted_replay),
        "c3_n_survived": n_survived,
        "c3_survival_rate": survival_rate,
        "c3_survival_rate_floor": C3_SURVIVAL_RATE_FLOOR,
        "c3_corroboration_rate": corroboration_rate,
    }

    criteria_by_arm = {
        ARM_STRICT_FLAG_OFF: ["C1_strict_gate_refuses_matched_candidate"],
        ARM_REPLAY_ON: ["C2_mint_fires_under_valid_conditions",
                        "C3_survival_clears_preregistered_floor"],
    }
    criteria_non_degenerate = arm_criteria_non_degenerate(
        criteria_by_arm, agg,
        extra={
            "C1_strict_gate_refuses_matched_candidate": strict_gate_green and len(cleared_strict) > 0,
            "C2_mint_fires_under_valid_conditions": replay_gate_green and len(cleared_replay) > 0,
            "C3_survival_clears_preregistered_floor": len(minted_replay) > 0,
        })

    interpretation = {
        "label": label,
        "preconditions": agg["adjudication_preconditions"],
        "criteria": [
            {"name": "C1_strict_gate_refuses_matched_candidate", "load_bearing": True,
             "passed": c1_pass},
            {"name": "C2_mint_fires_under_valid_conditions", "load_bearing": True,
             "passed": c2_pass},
            {"name": "C3_survival_clears_preregistered_floor", "load_bearing": True,
             "passed": c3_pass},
        ],
        "criteria_non_degenerate": criteria_non_degenerate,
        "per_arm_gate": agg["per_arm_gate"],
        "gate_offending_cells": {g["arm"]: g["offending_cells"] for g in arm_gates},
        "confirmatory_readout": {
            "strict_mint_fraction": strict_mint_fraction,
            "replay_mint_fraction": replay_mint_fraction,
            "survival_rate": survival_rate,
            "survival_rate_floor": C3_SURVIVAL_RATE_FLOOR,
            "corroboration_rate": corroboration_rate,
        },
        "both_directions": (
            "CONFIRMATORY (evidence-purpose, scored by governance). C1 failing "
            "would be a genuine MECH-094 safety breach (highest-severity "
            "finding this lane can produce). C2 failing would mean the "
            "carve-out itself has regressed since 873a/892. C3 failing "
            "(survival rate below the pre-registered 0.30 floor) is a valid, "
            "non-catastrophic negative confirmatory result -- it would say the "
            "mechanism mints safely but the biology-inspired functional "
            "benefit (consolidation surviving to be usable) is weaker than "
            "892's diagnostic prior suggested, which WEAKENS but does not "
            "eliminate MECH-322 (the safety carve-out C1/C2 would still hold)."
        ),
        "combination_rule": (
            "overall PASS = both arms' readiness gates green AND C1 (strict "
            f"arm mint fraction <= {C1_STRICT_MINT_FRACTION_CEILING:.2f}) AND "
            f"C2 (replay arm mint fraction >= {C2_REPLAY_MINT_FRACTION_FLOOR:.2f}) "
            f"AND C3 (replay arm survival rate >= {C3_SURVIVAL_RATE_FLOOR:.2f}). "
            "All three are pre-registered before this run executed."
        ),
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
        "c1_pass": c1_pass, "c2_pass": c2_pass, "c3_pass": c3_pass,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    t0 = time.perf_counter()

    global SEEDS, N_EPISODES, STEPS_PER_EPISODE, SLEEP_CHECKPOINT_EPISODE
    global REPLAY_CORROBORATION_EPISODES
    if args.dry_run:
        SEEDS = [111]
        N_EPISODES = 8
        STEPS_PER_EPISODE = 72
        SLEEP_CHECKPOINT_EPISODE = 2
        REPLAY_CORROBORATION_EPISODES = 4

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
        "c1_strict_mint_fraction_ceiling": C1_STRICT_MINT_FRACTION_CEILING,
        "c2_replay_mint_fraction_floor": C2_REPLAY_MINT_FRACTION_FLOOR,
        "c3_survival_rate_floor": C3_SURVIVAL_RATE_FLOOR,
        "outcome_spread_floor": OUTCOME_SPREAD_FLOOR,
        "symbols_per_trial_floor": SYMBOLS_PER_TRIAL_FLOOR,
        "harm_step_fraction_floor": HARM_STEP_FRACTION_FLOOR,
        "arm_config_slices": {a: _config_slice(a) for a in ARMS},
    }
    manifest = {
        "run_id": f"{EXPERIMENT_TYPE}_{ts}_v3",
        "queue_id": QUEUE_ID,
        "experiment_type": EXPERIMENT_TYPE,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "claim_ids": CLAIM_IDS,
        "evidence_direction": (
            "supports" if result["outcome"] == "PASS"
            else ("weakens" if (result["c1_pass"] and result["c2_pass"]
                                and not result["c3_pass"]) else "unknown")
        ),
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
    print(f"C1(safety)={m['c1_pass']} C2(mint)={m['c2_pass']} C3(survival)={m['c3_pass']}",
          flush=True)
    print(f"readiness_seed_fraction_by_arm={m['readiness_seed_fraction_by_arm']} "
          f"(floor={m['readiness_seed_fraction_floor']})", flush=True)
    print(f"strict_mint_fraction={m['c1_strict_mint_fraction']} "
          f"(ceiling={m['c1_strict_mint_fraction_ceiling']})", flush=True)
    print(f"replay_mint_fraction={m['c2_replay_mint_fraction']} "
          f"(floor={m['c2_replay_mint_fraction_floor']})", flush=True)
    print(f"survival_rate={m['c3_survival_rate']} (floor={m['c3_survival_rate_floor']}) "
          f"n_minted={m['c3_n_minted']} n_survived={m['c3_n_survived']}", flush=True)
    print(f"corroboration_rate={m['c3_corroboration_rate']}", flush=True)
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
