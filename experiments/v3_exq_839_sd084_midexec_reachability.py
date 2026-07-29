#!/opt/local/bin/python3
"""V3-EXQ-839 -- SD-084 VALIDATION: is MECH-321's R4 MID-EXECUTION hook REACHABLE?

DIAGNOSTIC. Substrate-readiness validation for SD-084
(`REEConfig.use_persistent_committed_program_handle`, landed 2026-07-29,
ree-v3 a21f487709 / REE_assembly 72c86d5e4e). This probe does NOT weight any
claim -- `experiment_purpose=diagnostic`, `claim_ids=[]`. It BEARS ON SD-084,
MECH-321, ARC-070, ARC-071 and MECH-288.

THE QUESTION, AND WHY IT IS WORTH A RUN AT ALL
    MECH-321's R4 second phase -- mid-execution re-evaluation of the REMAINING,
    unexecuted content of an already-committed chunk trajectory -- has NEVER
    EXECUTED IN ANY EXPERIMENT. Not rarely: never, and structurally. The hook
    (`REEAgent.select_action`, ree_core/agent.py) sits behind a seven-way gate
    conjunction whose gate (4) requires `e3._committed_trajectory is not None`,
    while the LAST statement of `E3Selector.post_action_update` is an
    unconditional `self._committed_trajectory = None` and every standard driver
    calls `agent.update_residue(...)` on every step. The hook needs a trajectory
    committed on a PREVIOUS tick; every previous tick ends by destroying it.

    V3-EXQ-830 confirmed it empirically: `decomp_n_evaluated_midexec = 0` in ALL
    TEN cells (5 seeds x 2 arms) against `decomp_n_evaluated_precommit` of
    1862-2618 per cell. Decomposition demonstrably fired PRE-COMMIT; the
    mid-execution phase never occurred once. Diagnosis:
    `REE_assembly/evidence/planning/failure_autopsy_V3-EXQ-830_2026-07-29.md`
    section 3. Design: `docs/architecture/sd_084_persistent_committed_program_handle.md`.

    SD-084 adds `E3Selector._persistent_committed_trajectory`, a committed-program
    handle that SURVIVES the per-tick teardown, and points gate (4) at the UNION
    of the two handles when the flag is on. This experiment asks whether that
    makes the hook reachable in practice.

ACCEPTANCE CRITERION, TAKEN VERBATIM FROM THE AUTOPSY'S failure_record_entry
    "decomp_n_evaluated_midexec > 0 on a standard select_action -> update_residue
    driver loop, without hand-injected preconditions."
    That is C1 below, and it is the LOAD-BEARING criterion.

ARMS (paired, same seeds, one manipulation)
    ARM_HANDLE_OFF -- use_persistent_committed_program_handle=False. Must
        reproduce the V3-EXQ-830 structural zero.
    ARM_HANDLE_ON  -- ...=True. Must exceed it.
    Every other flag is identical. See
    `experiments/_lib/baselines/sd084_midexec_reachability.py`.

=====================================================================
THE DESIGN CONSTRAINT THIS SCRIPT EXISTS TO HONOUR: ATTRIBUTABILITY
=====================================================================
Natural reachability is SEED-DEPENDENT, and after gate (4) the next binding
gate is (6) `len(remaining) > 1`. The hook only fires on a tick FOLLOWING the
commit of a MULTI-ACTION program, and E3 commits a multi-action ARC-071 chunk
only when that chunk beats the CEM-optimised candidates on raw score. There is
NO chunk-selection-bias knob in the substrate to force this. Measured
2026-07-29 at this config over TEN candidate seeds (darwin-arm64, seeded env,
2 episodes x 60 steps -- full table in the baseline module): seeds 3, 47, 71 and
89 commit multi-action programs AND fire the hook when the handle is ON (14-29
mid-execution evaluations); seeds 11, 17, 23, 29, 53 and 97 commit none and fire
nothing in either arm. OFF fires ZERO on all ten, including on seed 47 which
commits 30 multi-action programs -- the teardown admits no exceptions.

So a zero DV has TWO possible causes, and they route to opposite conclusions:
    (a) SD-084 does not work            -> a substrate finding
    (b) this seed never committed a multi-action program -> no test occurred
Without separating them the experiment is UNFALSIFIABLE in exactly the way
V3-EXQ-830's midexec counter was not (830's zero was attributable because its
`decomp_n_evaluated_precommit` of 1862-2618 proved the machinery was running).

This script therefore does BOTH of the things that separation requires:
  (a) it PINS seeds verified to produce multi-action commits ON THE MACHINE
      CLASS THAT WILL RUN IT (`--seed-scan` re-measures the pool; the pinned
      set is a measurement, not a guess), AND
  (b) it instruments `multi_action_commits` PER CELL alongside the DV, and
      makes it a READINESS PRECONDITION. An ARM with zero multi-action commits
      self-routes `substrate_not_ready_requeue` -- NEVER a substrate verdict.

AND IT DEMONSTRATES (b) RATHER THAN ASSERTING IT: TWO PRE-REGISTERED SEED TIERS
    ATTRIBUTABLE (3, 47, 71, 89) -- measured to commit multi-action programs and
        to fire the hook when the handle is ON. These carry C1.
    NEGATIVE CONTROL (23, 53) -- measured to commit NONE, WITH pre-commit
        decomposition nonetheless live (146 and 202 evaluations per cell). On
        these the DV must be zero in BOTH arms, and the arms must not even
        DIVERGE: no multi-action commit means gate (6) fails, so no latch is
        released and both arms follow the same action sequence. Measured: on
        every non-attributable seed the two arms are BIT-IDENTICAL in every
        recorded field.
    So one manifest shows both halves of the attributability claim -- the handle
    makes the hook fire where there IS something to re-evaluate, and a zero is
    correctly attributable where there is NOT. The control tier is recorded as a
    pre-registered EXPECTATION (`negative_control.expectation_held`), not gated:
    a control seed that turns out to commit a multi-action program after all is a
    stale-tier bookkeeping problem, not a substrate finding.

THE GATE IS EXISTENTIAL PER ARM, AND THAT IS NOT A WEAKENING
    A per-CELL (worst-cell) readiness gate would mark an arm red whenever ANY
    seed in it failed to commit a multi-action program -- and the negative
    control tier is, by construction, exactly such a seed. It would therefore
    vacate the entire run every time, including the four attributable seeds that
    genuinely tested the hook. That is the V3-EXQ-785 "a red arm vacates a green
    one" failure one level down, at the cell. The gate quantifies the same way
    C1 does -- existentially -- and partial attributability is reported in
    `attributability{}` instead of being swallowed. See `_best_cell`.

WHY THE SEED SET MUST BE RE-MEASURED PER MACHINE CLASS, NOT INHERITED
    `torch.multinomial` returns a different category on `linux-x86_64` from
    `darwin-arm64` given a bit-identical probability tensor at the same seed,
    and `ree_core/predictors/e3_selector.py` has multinomial call sites in the
    live selection path. Which candidate wins the commit is therefore not a
    portable property, so the 2026-07-29 darwin measurements above cannot be
    assumed to hold on a cloud worker. `--seed-scan` exists for exactly this,
    and the readiness gate is the backstop when the scan is stale.

READING `agent.e3._committed_trajectory` IN THE LOOP IS NOT PSEUDO-REPLICATION
    The E3 `last_*` diagnostics LATCH -- they survive a tick on which `select()`
    did not run, so a per-step read re-records one selection as many
    observations (~9x at the default `e3_steps_per_tick=10`). `_committed_trajectory`
    is the OPPOSITE: `post_action_update` destroys it every single tick, which
    is the entire defect this experiment measures. So it is None at the start of
    every `select_action` and non-None afterwards ONLY if `e3.select()` committed
    on THIS tick. The read is fresh by construction. `n_ticks` is recorded
    alongside so the denominator is auditable rather than inferred.

HOLD-WEIGHTED-READOUT TRIAGE (validate_experiments advisory), NOT AN EXEMPTION
    The ACTION SEQUENCE is accumulated from `select_action()`'s return value on
    every env step, and `agent.py` returns the HELD action on a non-e3-tick
    before `e3.select()` is reached (cadence `e3_steps_per_tick` defaults to 10,
    varying 5-20 under MECH-093 arousal). So each selection is weighted by its
    hold duration. Triaged rather than assumed, on the gate's own rubric:

      * The PRIMARY DV is untouched. `decomp_n_evaluated_midexec` is read from
        `PolicyDecomposition.get_state()`, not from `select_action`'s return, so
        C1 -- the only load-bearing criterion -- cannot be reweighted by this.
      * The one action-derived CRITERION, C4, is THRESHOLD-INVARIANT: it gates on
        `> 0.0`, i.e. "did the arms ever differ". Hold weighting changes how many
        TICKS each selection occupies; it cannot create a difference where there
        is none, nor erase one that exists. That is the rubric's SAFE case.
      * The remaining action-derived quantities (harm, run length, divergence
        fraction) are NON-GATING recorded payload, not DVs of this run.

    The fix is applied anyway rather than waived, because the hold structure is
    itself interesting here -- a mid-execution latch release acts ON hold
    duration. Every cell emits `n_fresh_select` / `n_latched` /
    `fresh_select_yield`, and `behavioural_delta` carries BOTH the hold-weighted
    divergence and a hold-unweighted `action_divergence_frac_fresh_ref_off`
    restricted to the reference arm's fresh-selection ticks. Kept distinct, per
    the gate's "emit BOTH" instruction.

=====================================================================
THREE PARAMETER TRAPS, ALL MEASURED 2026-07-29
=====================================================================
  1. `decomposition_vs_threshold=0.99` decomposes a triggering chunk to SINGLE
     actions, leaving no remainder, so gate (6) fails on those ticks even though
     gate (4) passes. It still works at depth_cap=3 because decomposition does
     not fire on every tick -- but the margin is THIN. This is instrumented
     (`multi_action_commits`), never assumed.
  2. `decomposition_depth_cap=1` avoids trap 1 and is DEGENERATE: the substrate
     emits a UserWarning saying decomposition never fires at all. Not used.
  3. `vs_trigger` is `region_vs < threshold` -- a HIGH threshold means
     NEAR-CERTAIN decomposition, the opposite of the naive reading.

WHAT IS **NOT** CLAIMED HERE, AND WHY THE BEHAVIOURAL BLOCK IS NON-GATING
    SD-084 is NOT a pure diagnostic when ON. A reachable mid-execution hook can
    newly reach `boundary.fired`, which feeds MECH-321's R1 OR trigger, and a
    mid-execution fire RELEASES THE COMMIT LATCH, aborting the remaining macro.
    Committed action sequences therefore CHANGE when the flag is on.

    That behavioural delta is the real scientific payload of MECH-321's R4 --
    but it is NOT what this run adjudicates. The counter proves REACHABILITY;
    the behavioural delta is what R4 actually CLAIMS, and claiming it needs a
    hypothesis about direction and an env in which aborting a macro is
    supposed to help. So the delta is measured and recorded GENEROUSLY here
    (`behavioural_delta` per seed: action-sequence divergence, first divergence
    tick, committed-run length, harm) and carried by ONE NON-load-bearing
    criterion, C4, which asserts only the SD doc's own stated property -- that
    the arms diverge at all. A successor evidence experiment consumes this
    block instead of re-deriving it (Experimental Recording Standard sec 3c).

NO PHASED TRAINING IS REQUIRED, AND THAT IS NOT AN OVERSIGHT
    Phased training (P0 encoder warmup -> P1 frozen-encoder head -> P2 eval) is
    mandatory when a downstream HEAD is trained on z_world / z_harm / any
    encoder output, because a head chasing moving latent targets collapses
    (EXQ-166b/c/d, 085l, 194). This probe trains NO head: it reads counters off
    `PolicyDecomposition.get_state()` while the agent runs its ordinary online
    loop. There is no downstream target to chase.

z_goal IS DELIBERATELY INERT HERE, AND THE REASON IS TWO LAYERS DEEP
    `agent.update_z_goal(benefit_exposure=0.0, ...)` is called every tick,
    kwargs-only (a positional call collides with `latent` and raises TypeError
    every tick -- the EXQ-471/475/483/490/524 cohort bug). It is carried VERBATIM
    from the reachability contract because that is the configuration in which
    the hook has actually been observed to fire.

    The stream is inert for the OUTER of two independent reasons, and it is worth
    being exact about which, because they are different failure shapes:
      (1) `REEConfig.from_dims(z_goal_enabled=...)` DEFAULTS TO FALSE, so
          `REEAgent.goal_state` is never CONSTRUCTED at all. This is the binding
          one here. The measured smoke reading is
          `goal_state_present=False, writer_calls=96, ticks=0/0` -- i.e.
          `active_frac=unmeasured`, not `0.000`.
      (2) Even with it constructed, `GoalState.update` only pulls z_goal toward
          z_world when `benefit_exposure > benefit_threshold`, and 0.0 never
          clears it. This layer is moot while (1) holds.
    So the counters are RECORDED (absence of the block would mean "unmeasured",
    the one reading that hides a genuinely dead stream) and the correct reading
    is "goal stream off by config", NOT a writer defect.

    This lineage's question is the COMMIT LIFECYCLE, not z_goal. Enabling the
    stream would change behaviour away from the verified reference -- and via
    SD-024's benefit attractor it would also un-zero the SD-025 curiosity bonus
    -- for no gain to a reachability question.

DV-SYMMETRY INVARIANCE, DECLARED PER ARM (failure_autopsy_V3-EXQ-604c section 3)
    ARM_HANDLE_OFF and ARM_HANDLE_ON, primary DV `decomp_n_evaluated_midexec`:
    the DV is a COUNT over ticks, whose symmetry group is the permutations of
    the tick index set (a count is a symmetric function of its per-tick
    indicators). The manipulation is NOT invariant under it -- the persistent
    handle changes each tick's indicator ITSELF, by deciding whether gate (4)
    passes on that tick, rather than reordering or rescaling contributions. It
    is neither a broadcast additive constant across candidates (which would
    cancel in an argmax and in a max-minus-min range) nor a monotone rescaling
    of a ranked quantity; it is a boolean gate on a per-tick precondition.

    Both arms, secondary behavioural DVs (action divergence, committed-run
    length, harm): these derive from the argmax action sequence, whose symmetry
    group includes any uniform additive shift of candidate scores and any
    monotone rescaling. The manipulation is invariant under NEITHER: it acts by
    RELEASING THE COMMIT LATCH, which switches the agent between the held /
    stepped path and a fresh selection -- a change of BRANCH in `select_action`,
    not an offset applied to candidate scores.

    Per readiness precondition, which channel it certifies: both preconditions
    are COUNT statistics over the same commit/decomposition path that C1 routes
    on, and they certify BOTH arms (each is measured in every cell). Neither is
    a magnitude standing in for a range, so the V3-EXQ-643 mismatch does not
    arise -- the load-bearing criterion is count-gated and the readiness checks
    are count readiness.
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from experiment_protocol import emit_outcome  # noqa: E402
from experiments._lib.arm_fingerprint import arm_cell  # noqa: E402
from experiments._lib.precondition_gate import (  # noqa: E402
    PreconditionSpec,
    aggregate_arm_gates,
    arm_criteria_non_degenerate,
    assert_no_structurally_unsatisfiable_gate,
    evaluate_arm_gate,
)
from experiments._lib.z_goal_stream import ZGoalStreamAccumulator  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402
import experiments._lib.baselines.sd084_midexec_reachability as baselines  # noqa: E402

from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from ree_core.policy import ChunkedPrimitive, ChunkState  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_839_sd084_midexec_reachability"
EXPERIMENT_PURPOSE = "diagnostic"
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"

ARM_OFF = "ARM_HANDLE_OFF"
ARM_ON = "ARM_HANDLE_ON"
ARMS = (ARM_OFF, ARM_ON)

# --- Pre-registered thresholds. Constants, never derived from this run's own
# statistics. Both are strict floors of ZERO: the autopsy's acceptance target is
# literally "> 0", and the anti-vacuity precondition is literally "at least one
# multi-action program was committed". Neither is a tuned number. ---
MIDEXEC_FLOOR = 0.0
MULTI_ACTION_COMMIT_FLOOR = 0.0
PRECOMMIT_EVAL_FLOOR = 0.0

# Non-load-bearing C4: with the handle ON the SD doc states the arms take
# different action sequences. A floor of exactly zero asserts only "they differ
# at all", which is the doc's own claim -- not a tuned effect size.
BEHAVIOURAL_DIVERGENCE_FLOOR = 0.0

# A fresh REEAgent is built inside every cell, so the z_goal counters are
# accumulated rather than read off one held agent -- appending each agent to a
# list would keep every (arm x seed) agent's nets, optimiser state and replay
# buffers alive until the last cell finished. Recording the counters is always
# safe; it is ADDING an update_z_goal call to a landed script that is a
# behaviour change (SD-024 benefit attractor -> SD-025 curiosity bonus). This
# script already calls it, so there is nothing to retrofit.
_ZGOAL = ZGoalStreamAccumulator()


# ---------------------------------------------------------------------------
# Config construction
# ---------------------------------------------------------------------------
def _arm_flags(arm_id: str) -> Dict[str, Any]:
    return (baselines.off_arm_flags() if arm_id == ARM_OFF
            else baselines.on_arm_flags())


def _config_slice(arm_id: str) -> Dict[str, Any]:
    """Fingerprint config slice for one arm.

    ARM_HANDLE_OFF's slice comes from the canonical baseline module verbatim, so
    the mint this run emits matches a successor's `try_reuse_cell(...)` BY
    CONSTRUCTION. ARM_HANDLE_ON reuses the same skeleton with the ON flags, so
    the two can never collide.
    """
    if arm_id == ARM_OFF:
        return baselines.off_path_config_slice()
    slice_ = baselines.off_path_config_slice()
    slice_.update(baselines.on_arm_flags())
    return slice_


def _build(seed: int, arm_id: str) -> Tuple[CausalGridWorldV2, REEAgent, Dict[str, Any]]:
    # env_kwargs(seed) passes seed= to the env. NOT optional -- an unseeded env
    # is not reproducible even after reset_all_rng(seed), which would make the
    # paired behavioural delta measure RNG noise and make the reuse fingerprint
    # dishonest. Measured both ways; see the baseline module's ENV SEEDING block.
    env = CausalGridWorldV2(**baselines.env_kwargs(seed))
    env.reset()
    flags = _arm_flags(arm_id)
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=baselines.SELF_DIM,
        world_dim=baselines.WORLD_DIM,
        reafference_action_dim=env.action_dim,
        **flags,
    )
    agent = REEAgent(cfg)
    return env, agent, flags


def _register_chunk(agent: REEAgent) -> None:
    """Ordinary ARC-071 usage -- a precondition of the measurement, done
    identically in both arms, NOT the manipulation. See the baseline module."""
    agent.policy_chunking.library.register(
        ChunkedPrimitive(
            sequence=baselines.SEEDED_CHUNK_SEQUENCE,
            depth=baselines.SEEDED_CHUNK_DEPTH,
            state=ChunkState.CRYSTALLISED,
            selection_weight=baselines.SEEDED_CHUNK_SELECTION_WEIGHT,
        )
    )


# ---------------------------------------------------------------------------
# One cell
# ---------------------------------------------------------------------------
def _run_cell(seed: int, arm_id: str, episodes: int, steps: int,
              quiet: bool = False) -> Dict[str, Any]:
    """Drive the canonical loop for one (seed, arm) cell.

    NOTHING about the mid-execution gate is injected: no metadata is written, no
    `_committed_step_idx` is set, no `beta_gate.elevate()` is called. Every gate
    must be satisfied by the substrate's own dynamics.
    """
    env, agent, flags = _build(seed, arm_id)
    _register_chunk(agent)
    world_dim = agent.config.latent.world_dim

    n_ticks = 0
    total_commits = 0
    multi_action_commits = 0
    commit_seq_lengths: List[int] = []
    actions: List[int] = []
    harm_signals: List[float] = []
    beta_elevated_flags: List[bool] = []
    e3_tick_flags: List[bool] = []
    max_committed_step_idx = 0

    if not quiet:
        print(f"Seed {seed} Condition {arm_id}", flush=True)

    for ep in range(episodes):
        _, obs = env.reset()
        agent.reset()
        if not agent.policy_chunking.library.all_chunks():
            _register_chunk(agent)

        for _ in range(steps):
            latent = agent.sense(obs["body_state"], obs["world_state"])
            ticks = agent.clock.advance()
            e1_prior = (
                agent._e1_tick(latent)
                if ticks.get("e1_tick")
                else torch.zeros(1, world_dim, device=agent.device)
            )
            candidates = agent.generate_trajectories(latent, e1_prior, ticks)
            # Hold-weighting audit: a non-e3-tick returns the HELD action before
            # e3.select() is reached, so this flag separates fresh selections
            # from held repeats (validate_experiments hold-weighted advisory).
            e3_tick_flags.append(bool(ticks.get("e3_tick")))
            body = obs["body_state"]
            # kwargs-only -- a positional call collides with `latent`.
            agent.update_z_goal(
                benefit_exposure=0.0,
                drive_level=REEAgent.compute_drive_level(body),
            )
            action = agent.select_action(candidates, ticks)

            # --- ATTRIBUTABILITY INSTRUMENT (design constraint (b)). Read the
            # FRESH per-tick commit, BEFORE update_residue tears it down. This
            # is read-only: it observes the loop, it does not steer it. Not a
            # latched diagnostic -- see the module docstring. ---
            committed = agent.e3._committed_trajectory
            if committed is not None:
                total_commits += 1
                meta = committed.metadata or {}
                seq_len = len(meta.get("chunk_sequence", ()))
                commit_seq_lengths.append(int(seq_len))
                if seq_len > 1:
                    multi_action_commits += 1
            beta_elevated_flags.append(bool(agent.beta_gate.is_elevated))
            max_committed_step_idx = max(
                max_committed_step_idx, int(agent._committed_step_idx)
            )

            a_int = int(action.argmax(dim=-1).item())
            actions.append(a_int)
            _flat, harm, _done, _info, obs = env.step(a_int)
            harm_signals.append(float(harm))
            agent.update_residue(harm)
            n_ticks += 1

        if not quiet:
            print(
                f"  [train] rollout seed={seed} arm={arm_id} ep {ep + 1}/{episodes} "
                f"ticks={n_ticks} commits={total_commits} "
                f"multi={multi_action_commits}",
                flush=True,
            )

    # AFTER stepping -- observe() reads the counters at call time, so calling it
    # at the construction site would record zeros.
    _ZGOAL.observe(agent)

    state = agent.get_policy_decomposition_state()
    midexec = int(state.get("decomp_n_evaluated_midexec", 0))

    # Committed-run length: maximal runs of consecutive beta-elevated ticks.
    # This is the observable proxy for "a committed program is installed", and
    # it is the behavioural quantity a mid-execution latch release SHORTENS.
    runs: List[int] = []
    cur = 0
    for elevated in beta_elevated_flags:
        if elevated:
            cur += 1
        elif cur:
            runs.append(cur)
            cur = 0
    if cur:
        runs.append(cur)

    row: Dict[str, Any] = {
        "arm_id": arm_id,
        "seed": int(seed),
        "n_ticks": n_ticks,
        "episodes": episodes,
        "steps_per_episode": steps,
        # --- primary DV + its attributability instrument ---
        "decomp_n_evaluated_midexec": midexec,
        "decomp_n_evaluated_precommit": int(
            state.get("decomp_n_evaluated_precommit", 0)),
        "decomp_n_decomposed_midexec": int(
            state.get("decomp_n_decomposed_midexec", 0)),
        "decomp_n_decomposed_precommit": int(
            state.get("decomp_n_decomposed_precommit", 0)),
        "decomp_n_marked_unreliable": int(
            state.get("decomp_n_marked_unreliable", 0)),
        "decomp_n_vs_trigger": int(state.get("decomp_n_vs_trigger", 0)),
        "decomp_n_boundary_fires": int(state.get("decomp_n_boundary_fires", 0)),
        "decomp_n_boundary_fires_fast": int(
            state.get("decomp_n_boundary_fires_fast", 0)),
        "decomp_n_boundary_fires_slow": int(
            state.get("decomp_n_boundary_fires_slow", 0)),
        "decomp_n_boundary_cofire": int(state.get("decomp_n_boundary_cofire", 0)),
        "decomp_trigger_mode": state.get("decomp_trigger_mode"),
        "total_commits": total_commits,
        "multi_action_commits": multi_action_commits,
        "commit_seq_length_mean": (
            round(statistics.fmean(commit_seq_lengths), 6) if commit_seq_lengths
            else 0.0),
        "commit_seq_length_max": max(commit_seq_lengths) if commit_seq_lengths else 0,
        "max_committed_step_idx": max_committed_step_idx,
        # --- hold-weighting audit (the advisory's requested counters). The
        # denominator behind every action-derived statistic below, so a later
        # reader can see the replication factor instead of trusting len(). ---
        "n_fresh_select": sum(e3_tick_flags),
        "n_latched": len(e3_tick_flags) - sum(e3_tick_flags),
        "fresh_select_yield": round(
            sum(e3_tick_flags) / max(1, len(e3_tick_flags)), 6),
        "e3_tick_flags": e3_tick_flags,
        # --- behavioural payload (non-gating; recorded generously) ---
        "beta_elevated_frac": round(
            sum(beta_elevated_flags) / max(1, len(beta_elevated_flags)), 6),
        "committed_run_len_mean": round(statistics.fmean(runs), 6) if runs else 0.0,
        "committed_run_len_max": max(runs) if runs else 0,
        "committed_run_count": len(runs),
        "mean_harm_signal": round(statistics.fmean(harm_signals), 6)
        if harm_signals else 0.0,
        "n_harm_events": sum(1 for h in harm_signals if h < 0.0),
        "harm_event_frac": round(
            sum(1 for h in harm_signals if h < 0.0) / max(1, len(harm_signals)), 6),
        "action_sequence": actions,
        "arm_flags": dict(flags),
    }

    # Per-cell verdict: the arm's own expectation, plus attributability.
    if arm_id == ARM_ON:
        cell_pass = multi_action_commits > 0 and midexec > 0
    else:
        cell_pass = multi_action_commits > 0 and midexec == 0
    row["cell_pass"] = bool(cell_pass)
    if not quiet:
        print(f"verdict: {'PASS' if cell_pass else 'FAIL'}", flush=True)
    return row


# ---------------------------------------------------------------------------
# Preconditions
# ---------------------------------------------------------------------------
def _arm_context(arm_id: str) -> Dict[str, Any]:
    """Arm context for the precondition gate.

    The key MUST be `id`: that is `assert_no_structurally_unsatisfiable_gate`'s
    default `arm_id_key`, and passing a differently-keyed dict is silently
    useless (it degrades every arm to "?"). `arm_id` is carried alongside for
    readability at the `applies_to` call sites.
    """
    return {"id": arm_id, "arm_id": arm_id,
            "handle_on": arm_id == ARM_ON}


def _precondition_specs() -> List[PreconditionSpec]:
    """Two READINESS preconditions, both COUNT statistics, both applying to
    BOTH arms.

    Neither is scoped out anywhere: both arms must commit multi-action programs
    and both must run pre-commit decomposition. That symmetry is asserted by the
    shipped reachability contract, which makes the SAME anti-vacuity assertion
    in its OFF test as in its ON test.

    No `structural_max` / `structural_min` is declarable: both are emergent
    counts from the CEM-vs-chunk score comparison, not quantities this design
    pre-registers a value for. `assert_no_structurally_unsatisfiable_gate`
    therefore has nothing to refuse -- which is the honest state, not an
    omission.
    """
    return [
        PreconditionSpec(
            name="multi_action_commits_present",
            description=(
                "At least one cell of this ARM committed a MULTI-ACTION "
                "program. Gate (6) of the mid-execution conjunction is "
                "len(remaining) > 1, so the hook can only fire on a tick "
                "following a multi-action commit. Zero across the whole arm "
                "means NO TEST OCCURRED -- it must never be read as SD-084 "
                "failing. EXISTENTIAL, matching C1's quantifier; per-seed "
                "detail is in attributability{}, not swallowed by this gate."
            ),
            control=(
                "BEST cell of this arm (max, which recomputes the existential "
                "exactly); a crystallised 3-action ARC-071 chunk is registered "
                "in the library in both arms, and E3 commits it only when it "
                "beats the CEM-optimised candidates on raw score -- no "
                "chunk-selection-bias knob exists in the substrate"
            ),
            threshold=MULTI_ACTION_COMMIT_FLOOR,
            direction="lower",
            kind="readiness",
        ),
        PreconditionSpec(
            name="decomposition_precommit_live",
            description=(
                "Pre-commit decomposition evaluated at least once in this arm. "
                "This is the KNOWN-REACHABLE phase (V3-EXQ-830 recorded "
                "1862-2618 per cell), so a zero means the decomposition "
                "machinery is not running at all -- a config error, not a "
                "finding about SD-084."
            ),
            control=(
                "BEST cell of this arm; the pre-commit phase is the POSITIVE "
                "CONTROL because it is the one phase already measured to fire "
                "in this substrate, and it exercises the same PolicyDecomposition"
                ".evaluate() path the mid-execution phase would"
            ),
            threshold=PRECOMMIT_EVAL_FLOOR,
            direction="lower",
            kind="readiness",
        ),
    ]


def _best_cell(rows: List[Dict[str, Any]], key: str) -> Tuple[float, str]:
    """Return (best value, the cell that attains it).

    THE MAXIMUM, AND THE CHOICE IS LOAD-BEARING -- read this before "fixing" it
    to a min or a mean.

    The usual rule is to report the WORST cell, because a precondition's `met` is
    usually a worst-case claim (`all(...)`) and the indexer RECOMPUTES `met` from
    the reported numbers -- so a mean would let one out-of-band cell recompute as
    MET while the author's own `met` is False (the V3-EXQ-779b defect).

    Here `met` is an EXISTENTIAL claim, so the extremum that recomputes it
    exactly is the MAXIMUM: `any(cell.multi_action_commits > 0)` is identically
    `max(...) > 0`. It is still an extremum, never a mean, so the 779b hazard
    does not arise.

    Why existential is the RIGHT quantifier here, and not a weakening: C1 is
    itself existential (`total_midexec_on > 0` -- "does the hook EVER fire under
    a standard loop"), and a precondition must quantify the same way as the
    criterion it guards. A worst-cell gate would let ONE non-attributable seed
    -- a seed that merely never happened to commit a multi-action program --
    turn the whole arm red and vacate the seeds that DID test the hook. That is
    precisely the V3-EXQ-785 "a red arm vacates a green one" failure, arriving
    one level down at the cell rather than the arm.

    Partial attributability is therefore reported as a NON-GATING diagnostic
    (`n_attributable_cells`, `attributable_seeds` per arm) rather than swallowed
    by a gate, so a reader can see that e.g. 2 of 4 seeds carried the result.
    """
    best = max(rows, key=lambda r: r[key])
    return float(best[key]), f"{best['arm_id']}::seed{best['seed']}"


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
def _behavioural_delta(off_row: Dict[str, Any],
                       on_row: Dict[str, Any]) -> Dict[str, Any]:
    """Paired OFF-vs-ON behavioural readout for one seed.

    NON-GATING except for C4. Both arms are seeded identically and the arms are
    deterministic given the seed, so the sequences are directly comparable until
    the first latch release makes them diverge.
    """
    a_off = off_row["action_sequence"]
    a_on = on_row["action_sequence"]
    n = min(len(a_off), len(a_on))
    diffs = [i for i in range(n) if a_off[i] != a_on[i]]

    # Hold-UNWEIGHTED variant, kept DISTINCT from the hold-weighted one above
    # (validate_experiments hold-weighted advisory, "emit BOTH"). Restricted to
    # the reference arm's FRESH-SELECTION ticks, so each selection contributes
    # once instead of once per tick of its hold. The OFF arm is the reference
    # because it is the unmanipulated one -- once the arms diverge their tick
    # structures diverge too, so there is no common index set, and naming the
    # reference is the honest way to say which one indexes the comparison.
    off_fresh = off_row.get("e3_tick_flags") or []
    fresh_idx = [i for i in range(n) if i < len(off_fresh) and off_fresh[i]]
    fresh_diffs = [i for i in fresh_idx if a_off[i] != a_on[i]]

    return {
        "seed": off_row["seed"],
        "n_compared_ticks": n,
        "n_divergent_ticks": len(diffs),
        "action_divergence_frac": round(len(diffs) / max(1, n), 6),
        "first_divergence_tick": diffs[0] if diffs else None,
        "n_fresh_ref_ticks": len(fresh_idx),
        "n_divergent_fresh_ref_ticks": len(fresh_diffs),
        "action_divergence_frac_fresh_ref_off": round(
            len(fresh_diffs) / max(1, len(fresh_idx)), 6),
        "n_fresh_select_off": off_row.get("n_fresh_select"),
        "n_fresh_select_on": on_row.get("n_fresh_select"),
        "committed_run_len_mean_off": off_row["committed_run_len_mean"],
        "committed_run_len_mean_on": on_row["committed_run_len_mean"],
        "committed_run_len_mean_delta": round(
            on_row["committed_run_len_mean"] - off_row["committed_run_len_mean"], 6),
        "beta_elevated_frac_off": off_row["beta_elevated_frac"],
        "beta_elevated_frac_on": on_row["beta_elevated_frac"],
        "mean_harm_signal_off": off_row["mean_harm_signal"],
        "mean_harm_signal_on": on_row["mean_harm_signal"],
        "mean_harm_signal_delta": round(
            on_row["mean_harm_signal"] - off_row["mean_harm_signal"], 6),
        "midexec_off": off_row["decomp_n_evaluated_midexec"],
        "midexec_on": on_row["decomp_n_evaluated_midexec"],
        "decomposed_midexec_on": on_row["decomp_n_decomposed_midexec"],
        "multi_action_commits_off_on": (
            off_row["multi_action_commits"], on_row["multi_action_commits"]),
    }


def _analyse(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    off_rows = [r for r in rows if r["arm_id"] == ARM_OFF]
    on_rows = [r for r in rows if r["arm_id"] == ARM_ON]

    specs = _precondition_specs()
    arm_contexts = {a: _arm_context(a) for a in ARMS}

    arm_gates = []
    attributability: Dict[str, Any] = {}
    for arm_id, arm_rows in ((ARM_OFF, off_rows), (ARM_ON, on_rows)):
        mac, mac_cell = _best_cell(arm_rows, "multi_action_commits")
        pre, pre_cell = _best_cell(arm_rows, "decomp_n_evaluated_precommit")
        gate = evaluate_arm_gate(
            arm_id,
            arm_contexts[arm_id],
            specs,
            measured={
                "multi_action_commits_present": mac,
                "decomposition_precommit_live": pre,
            },
        )
        for p in gate["preconditions"]:
            if p["precondition"] == "multi_action_commits_present":
                p["attaining_cell"] = mac_cell
            elif p["precondition"] == "decomposition_precommit_live":
                p["attaining_cell"] = pre_cell
        arm_gates.append(gate)
        # NON-GATING partial-attributability diagnostic. The gate is
        # existential, so this is where a reader sees HOW MANY seeds actually
        # tested the hook rather than inferring it from a green light.
        attr_seeds = sorted(r["seed"] for r in arm_rows
                            if r["multi_action_commits"] > 0)
        attributability[arm_id] = {
            "n_cells": len(arm_rows),
            "n_attributable_cells": len(attr_seeds),
            "attributable_seeds": attr_seeds,
            "non_attributable_seeds": sorted(
                r["seed"] for r in arm_rows if r["multi_action_commits"] == 0),
            "multi_action_commits_per_seed": {
                str(r["seed"]): r["multi_action_commits"] for r in arm_rows},
            "midexec_per_seed": {
                str(r["seed"]): r["decomp_n_evaluated_midexec"]
                for r in arm_rows},
        }

    aggregate = aggregate_arm_gates(arm_gates)

    total_midexec_on = sum(r["decomp_n_evaluated_midexec"] for r in on_rows)
    total_midexec_off = sum(r["decomp_n_evaluated_midexec"] for r in off_rows)
    on_cells_with_midexec = [
        r for r in on_rows if r["decomp_n_evaluated_midexec"] > 0]
    on_cells_attributable = [r for r in on_rows if r["multi_action_commits"] > 0]
    on_attributable_with_midexec = [
        r for r in on_cells_attributable if r["decomp_n_evaluated_midexec"] > 0]

    seeds = sorted({r["seed"] for r in rows})
    off_by_seed = {r["seed"]: r for r in off_rows}
    on_by_seed = {r["seed"]: r for r in on_rows}
    deltas = [_behavioural_delta(off_by_seed[s], on_by_seed[s])
              for s in seeds if s in off_by_seed and s in on_by_seed]
    for d in deltas:
        d["seed_tier"] = ("attributable"
                          if d["seed"] in baselines.SEED_POOL_ATTRIBUTABLE
                          else "negative_control")
    n_seeds_diverged = sum(1 for d in deltas if d["n_divergent_ticks"] > 0)

    # C4 is computed over the DECLARED attributable tier, not a runtime-selected
    # subset: the tier is fixed in the baseline module BEFORE the run, so this is
    # pre-registration rather than post-hoc selection. Mixing the negative-control
    # seeds in would dilute the mean with seeds where the handle CANNOT act -- and
    # those seeds are more informative kept separate, as the control below.
    attr_deltas = [d for d in deltas if d["seed_tier"] == "attributable"]
    ctrl_deltas = [d for d in deltas if d["seed_tier"] == "negative_control"]
    mean_divergence = (round(statistics.fmean(
        d["action_divergence_frac"] for d in attr_deltas), 6)
        if attr_deltas else 0.0)

    # THE NEGATIVE CONTROL, stated as a pre-registered EXPECTATION rather than a
    # criterion. On a seed that commits no multi-action program the hook can
    # never fire, so no latch is ever released and the two arms should follow
    # the SAME action sequence. If a control seed diverges anyway, the handle is
    # changing behaviour through some path other than the mid-execution release
    # -- which would NOT invalidate C1 (the counter is read independently) but
    # would mean the OFF path is not as inert as the SD doc claims, and belongs
    # in the closing report. Not gated, because a control seed that turns out to
    # commit a multi-action program after all is a stale-tier problem, not a
    # substrate finding.
    negative_control = {
        "declared_seeds": list(baselines.SEED_POOL_NEGATIVE_CONTROL),
        "n_control_seeds": len(ctrl_deltas),
        "expectation": (
            "midexec == 0 in BOTH arms and action_divergence_frac == 0, because "
            "no multi-action program is committed so the hook cannot fire"),
        "midexec_on_total": sum(
            r["decomp_n_evaluated_midexec"] for r in on_rows
            if r["seed"] in baselines.SEED_POOL_NEGATIVE_CONTROL),
        "multi_action_commits_total": sum(
            r["multi_action_commits"] for r in rows
            if r["seed"] in baselines.SEED_POOL_NEGATIVE_CONTROL),
        "mean_action_divergence_frac": (round(statistics.fmean(
            d["action_divergence_frac"] for d in ctrl_deltas), 6)
            if ctrl_deltas else None),
        "n_control_seeds_diverged": sum(
            1 for d in ctrl_deltas if d["n_divergent_ticks"] > 0),
        "expectation_held": bool(
            ctrl_deltas
            and all(d["midexec_on"] == 0 for d in ctrl_deltas)
            and all(d["n_divergent_ticks"] == 0 for d in ctrl_deltas)),
        "tier_still_valid": bool(
            ctrl_deltas
            and all(d["multi_action_commits_off_on"] == (0, 0)
                    for d in ctrl_deltas)),
    }

    # --- Pre-registered criteria. Names here MUST match the keys of
    # criteria_non_degenerate exactly: a short-key/long-name mismatch is what
    # produced V3-EXQ-830's FALSE vacuous_pass flag. ---
    c1 = total_midexec_on > MIDEXEC_FLOOR
    c2 = total_midexec_off == 0
    # EXISTENTIAL per arm, matching C1's quantifier and the shipped contract's
    # own anti-vacuity assertion (`multi_commits > 0` in each of its two tests).
    # NOT `all cells` -- see _best_cell for why a universal here would let one
    # non-attributable seed vacate the seeds that did test the hook.
    c3 = all(
        any(r["multi_action_commits"] > MULTI_ACTION_COMMIT_FLOOR
            for r in arm_rows)
        for arm_rows in (off_rows, on_rows)
    )
    c4 = mean_divergence > BEHAVIOURAL_DIVERGENCE_FLOOR

    criteria = [
        {"name": "C1_MIDEXEC_REACHABLE_ON", "load_bearing": True, "passed": bool(c1),
         "measured": total_midexec_on, "threshold": MIDEXEC_FLOOR,
         "statement": ("decomp_n_evaluated_midexec > 0 in ARM_HANDLE_ON on a "
                       "standard select_action -> update_residue loop with no "
                       "hand-injected preconditions (the autopsy's "
                       "failure_record_entry target, verbatim)")},
        {"name": "C2_STRUCTURAL_ZERO_OFF", "load_bearing": False, "passed": bool(c2),
         "measured": total_midexec_off, "threshold": 0,
         "statement": ("ARM_HANDLE_OFF reproduces the V3-EXQ-830 structural "
                       "zero. This is the negative control: it is what makes "
                       "C1 mean 'the handle did it' rather than 'something "
                       "changed'.")},
        {"name": "C3_ATTRIBUTABILITY", "load_bearing": False, "passed": bool(c3),
         "measured": min(
             max((r["multi_action_commits"] for r in arm_rows), default=0)
             for arm_rows in (off_rows, on_rows)),
         "threshold": MULTI_ACTION_COMMIT_FLOOR,
         "statement": ("EACH arm committed at least one multi-action program, "
                       "so a zero DV is attributable to the handle and not to "
                       "an absent test. Existential per arm, matching C1's "
                       "quantifier; see attributability{} for how many seeds "
                       "actually carried it.")},
        {"name": "C4_BEHAVIOURAL_DELTA", "load_bearing": False, "passed": bool(c4),
         "measured": mean_divergence, "threshold": BEHAVIOURAL_DIVERGENCE_FLOOR,
         "statement": ("on the DECLARED ATTRIBUTABLE seed tier the arms take "
                       "different action sequences, as the SD-084 design doc's "
                       "'NOT a pure diagnostic' section states they must. "
                       "Computed over the pre-registered attributable tier only "
                       "(negative_control{} carries the other tier, where the "
                       "expectation is the OPPOSITE -- no divergence, because "
                       "the hook cannot fire). NON-load-bearing: this run "
                       "validates REACHABILITY, not MECH-321 R4's behavioural "
                       "claim.")},
    ]

    # --- Self-route. Four pre-registered labels; the readiness gate wins. ---
    if not aggregate["non_degenerate"]:
        label = "substrate_not_ready_requeue"
        outcome = "FAIL"
    elif not c2:
        label = "off_path_not_structurally_zero"
        outcome = "FAIL"
    elif c1:
        label = "midexec_reachable_handle_validated"
        outcome = "PASS"
    else:
        label = "midexec_still_unreachable_with_handle"
        outcome = "FAIL"

    non_degenerate = arm_criteria_non_degenerate(
        {ARM_ON: ["C1_MIDEXEC_REACHABLE_ON"],
         ARM_OFF: ["C2_STRUCTURAL_ZERO_OFF"]},
        aggregate,
    )
    # C3 and C4 are asserted ACROSS both arms, so they are degenerate unless
    # both arms are green -- `arm_criteria_non_degenerate` keys to one arm only.
    non_degenerate["C3_ATTRIBUTABILITY"] = bool(aggregate["all_green"])
    non_degenerate["C4_BEHAVIOURAL_DELTA"] = bool(aggregate["all_green"])

    return {
        "outcome": outcome,
        "interpretation_label": label,
        "criteria": criteria,
        "criteria_non_degenerate": non_degenerate,
        "preconditions": aggregate["adjudication_preconditions"],
        "per_arm_gate": aggregate["per_arm_gate"],
        "non_degenerate": bool(aggregate["non_degenerate"]),
        "degeneracy_reason": aggregate["degeneracy_reason"],
        "attributability": attributability,
        "negative_control": negative_control,
        "summary": {
            "total_midexec_on": total_midexec_on,
            "total_midexec_off": total_midexec_off,
            "n_on_cells": len(on_rows),
            "n_on_cells_with_midexec": len(on_cells_with_midexec),
            "n_on_cells_attributable": len(on_cells_attributable),
            "n_on_cells_attributable_with_midexec": len(on_attributable_with_midexec),
            "frac_on_cells_with_midexec": round(
                len(on_cells_with_midexec) / max(1, len(on_rows)), 6),
            "total_multi_action_commits": sum(
                r["multi_action_commits"] for r in rows),
            "mean_action_divergence_frac_attributable_tier": mean_divergence,
            "n_seeds_diverged": n_seeds_diverged,
            "n_seeds": len(seeds),
            "n_seeds_attributable_tier": len(attr_deltas),
            "n_seeds_negative_control_tier": len(ctrl_deltas),
            "negative_control_expectation_held": negative_control[
                "expectation_held"],
        },
        "behavioural_delta": deltas,
    }


# ---------------------------------------------------------------------------
# Seed scan
# ---------------------------------------------------------------------------
def _seed_scan(seeds: List[int], episodes: int, steps: int) -> int:
    """Re-measure which seeds commit MULTI-ACTION programs on THIS machine class.

    Screening only, and deliberately conservative in the safe direction: a seed
    that commits multi-action programs at the reduced schedule will commit more
    at the full one, so a hit is trustworthy. A miss may be a false negative at
    the shorter schedule -- which costs a seed, never a false verdict.
    """
    from experiments._lib.arm_fingerprint import machine_class

    print(f"seed-scan: machine_class={machine_class()} "
          f"episodes={episodes} steps={steps}", flush=True)
    print("seed  arm             multi_commits  total_commits  midexec  precommit",
          flush=True)
    ok: List[int] = []
    for seed in seeds:
        per_arm = {}
        for arm_id in ARMS:
            torch.manual_seed(seed)
            row = _run_cell(seed, arm_id, episodes, steps, quiet=True)
            per_arm[arm_id] = row
            print(f"{seed:<6}{arm_id:<16}{row['multi_action_commits']:<15}"
                  f"{row['total_commits']:<15}"
                  f"{row['decomp_n_evaluated_midexec']:<9}"
                  f"{row['decomp_n_evaluated_precommit']}", flush=True)
        if all(per_arm[a]["multi_action_commits"] > 0 for a in ARMS):
            ok.append(seed)
    print(f"\nATTRIBUTABLE SEEDS (multi-action commits in BOTH arms): {ok}",
          flush=True)
    print("Pin these in _lib/baselines/sd084_midexec_reachability.py SEED_POOL.",
          flush=True)
    return 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> Tuple[Optional[str], Optional[str], bool]:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--seed-scan", action="store_true",
                    help="screen a seed pool on THIS machine class and exit")
    ap.add_argument("--seeds", type=str, default="",
                    help="comma-separated seed override")
    ap.add_argument("--scan-episodes", type=int, default=4)
    args = ap.parse_args()

    seeds = ([int(s) for s in args.seeds.split(",") if s.strip()]
             if args.seeds else list(baselines.SEED_POOL))
    episodes = 2 if args.dry_run else baselines.EPISODES
    steps = 12 if args.dry_run else baselines.STEPS_PER_EPISODE

    if args.seed_scan:
        _seed_scan(seeds, args.scan_episodes, baselines.STEPS_PER_EPISODE)
        return None, None, True

    # DESIGN-TIME REFUSAL, BEFORE ANY COMPUTE IS SPENT (V3-EXQ-785). Nothing in
    # this design is structurally unsatisfiable -- both preconditions are
    # emergent counts with no pre-registered bound, so there is nothing for the
    # guard to refuse today. It is called anyway, and called HERE rather than in
    # _analyse, because its whole value is refusing BEFORE the run: a future
    # edit that introduces a pre-registered bound the gate cannot clear must
    # cost a second, not a full cloud run. Runs under --dry-run too, which is
    # what the guard's own docstring asks for.
    assert_no_structurally_unsatisfiable_gate(
        _precondition_specs(), [_arm_context(a) for a in ARMS])

    started = datetime.now(timezone.utc)
    t0 = time.perf_counter()

    rows: List[Dict[str, Any]] = []
    for arm_id in ARMS:
        for seed in seeds:
            # arm_cell resets ALL RNG on enter and stamps the fingerprint on
            # .stamp(). include_driver_script_in_hash=False on the OFF arm is
            # the MINT: it excludes this driver from substrate_hash so a
            # successor with its own driver can match the fingerprint and skip
            # re-running the control (arm_reuse_fingerprint_plan sec 9.7).
            with arm_cell(
                seed,
                config_slice=_config_slice(arm_id),
                script_path=Path(__file__),
                config_slice_declared=True,
                include_driver_script_in_hash=(arm_id == ARM_OFF),
            ) as cell:
                row = _run_cell(seed, arm_id, episodes, steps)
                cell.stamp(row)
            rows.append(row)

    result = _analyse(rows)

    run_id = (f"{EXPERIMENT_TYPE}_"
              f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_v3")
    cfg_record = {
        "arms": list(ARMS),
        "seeds": seeds,
        "episodes": episodes,
        "steps_per_episode": steps,
        # Per-seed env kwargs, so the manifest records the exact env seed each
        # cell used rather than a seed-free template.
        "env_kwargs_per_seed": {str(s): baselines.env_kwargs(s) for s in seeds},
        "env_seeded_per_cell": baselines.ENV_SEEDED_PER_CELL,
        "self_dim": baselines.SELF_DIM,
        "world_dim": baselines.WORLD_DIM,
        "seeded_chunk_sequence": list(baselines.SEEDED_CHUNK_SEQUENCE),
        "decomposition_vs_threshold": baselines.DECOMPOSITION_VS_THRESHOLD,
        "decomposition_depth_cap": baselines.DECOMPOSITION_DEPTH_CAP,
        "arm_flags": {a: _arm_flags(a) for a in ARMS},
        "thresholds": {
            "MIDEXEC_FLOOR": MIDEXEC_FLOOR,
            "MULTI_ACTION_COMMIT_FLOOR": MULTI_ACTION_COMMIT_FLOOR,
            "PRECOMMIT_EVAL_FLOOR": PRECOMMIT_EVAL_FLOOR,
            "BEHAVIOURAL_DIVERGENCE_FLOOR": BEHAVIOURAL_DIVERGENCE_FLOOR,
        },
    }

    manifest: Dict[str, Any] = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "architecture_epoch": ARCHITECTURE_EPOCH,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "outcome": result["outcome"],
        # Diagnostic: weights no claim. bears_on is the audit trail.
        "claim_ids": [],
        "bears_on": ["SD-084", "MECH-321", "ARC-070", "ARC-071", "MECH-288"],
        "evidence_direction": "non_contributory",
        "validates_substrate": "SD-084",
        "non_degenerate": result["non_degenerate"],
        "degeneracy_reason": result["degeneracy_reason"],
        "per_arm_gate": result["per_arm_gate"],
        "attributability": result["attributability"],
        "negative_control": result["negative_control"],
        "seed_tiers": {
            "attributable": list(baselines.SEED_POOL_ATTRIBUTABLE),
            "negative_control": list(baselines.SEED_POOL_NEGATIVE_CONTROL),
        },
        "interpretation": {
            "label": result["interpretation_label"],
            "preconditions": result["preconditions"],
            "criteria": result["criteria"],
            "criteria_non_degenerate": result["criteria_non_degenerate"],
            "preconditions_scope_note": result["per_arm_gate"].get(
                "preconditions_scope_note", ""),
        },
        "summary": result["summary"],
        "behavioural_delta": result["behavioural_delta"],
        "arm_results": rows,
        "per_seed_rows": rows,
        "custom_information": {
            "acceptance_criterion_source": (
                "REE_assembly/evidence/planning/"
                "failure_autopsy_V3-EXQ-830_2026-07-29.md -- "
                "recommended_substrate_queue_entry.failure_record_entry.target"),
            "sd_doc": ("REE_assembly/docs/architecture/"
                       "sd_084_persistent_committed_program_handle.md"),
            "contract": ("ree-v3/tests/contracts/"
                         "test_mech321_midexec_natural_reachability.py"),
            "predecessor_null": (
                "v3_exq_830_mech321_scale_resolved_rollout_boundary: "
                "decomp_n_evaluated_midexec = 0 in all 10 cells against "
                "decomp_n_evaluated_precommit 1862-2618"),
            "z_goal_deliberately_inert": (
                "update_z_goal is called every tick, kwargs-only, verbatim from "
                "the reachability contract -- but the stream is inert for the "
                "OUTER of two reasons: REEConfig.from_dims(z_goal_enabled=...) "
                "DEFAULTS TO FALSE, so REEAgent.goal_state is never constructed "
                "at all. Measured smoke reading: goal_state_present=False, "
                "writer_calls>0, ticks=0/0, i.e. active_frac=unmeasured and NOT "
                "0.000. The benefit-threshold layer (0.0 never clears it) is "
                "moot while that holds. Correct reading is 'goal stream off by "
                "config', not a writer defect. Enabling it would change "
                "behaviour away from the verified reference and, via SD-024's "
                "benefit attractor, un-zero the SD-025 curiosity bonus."),
            "hold_weighted_readout_triage": (
                "The action sequence is accumulated from select_action()'s "
                "return value every env step, so selections are weighted by "
                "hold duration (e3_steps_per_tick default 10). TRIAGED, not "
                "waived: the load-bearing DV decomp_n_evaluated_midexec is read "
                "from PolicyDecomposition.get_state() and is untouched; the one "
                "action-derived criterion C4 gates on > 0.0 and is therefore "
                "THRESHOLD-INVARIANT (hold weighting cannot create or erase a "
                "difference between arms); all other action-derived quantities "
                "are non-gating payload. n_fresh_select / n_latched / "
                "fresh_select_yield are emitted per cell, and "
                "behavioural_delta carries both the hold-weighted "
                "action_divergence_frac and the hold-unweighted "
                "action_divergence_frac_fresh_ref_off."),
            "behavioural_delta_is_payload_not_verdict": (
                "SD-084 ON is not a pure diagnostic: a mid-execution fire "
                "releases the commit latch and aborts the remaining macro, so "
                "action sequences change. That delta is recorded here for a "
                "successor evidence experiment to consume; this run adjudicates "
                "REACHABILITY only (C1), and C4 asserts only that the arms "
                "differ at all."),
        },
    }

    out_path = write_flat_manifest(
        manifest, None, dry_run=args.dry_run, config=cfg_record, seeds=seeds,
        script_path=Path(__file__),
        elapsed_seconds=round(time.perf_counter() - t0, 3),
        started_at=None,
        # Records the z_goal writer counters. Expected reading here is
        # active_frac=0.000 with writer_calls>0 -- correctly wired, benefit gate
        # never opened, because benefit_exposure is deliberately 0.0 (see the
        # module docstring). Absence of the block would mean "unmeasured", which
        # is the one reading that hides a genuinely dead stream.
        z_goal_stream_stats=_ZGOAL.stats(),
    )

    print(f"manifest: {out_path}", flush=True)
    s = result["summary"]
    print(
        f"outcome: {result['outcome']} label={result['interpretation_label']} "
        f"non_degenerate={result['non_degenerate']}", flush=True)
    print(
        f"  midexec ON={s['total_midexec_on']} OFF={s['total_midexec_off']} "
        f"on_cells_with_midexec={s['n_on_cells_with_midexec']}/{s['n_on_cells']} "
        f"multi_action_commits={s['total_multi_action_commits']}", flush=True)
    print(
        f"  behavioural: divergence_attributable_tier="
        f"{s['mean_action_divergence_frac_attributable_tier']} "
        f"seeds_diverged={s['n_seeds_diverged']}/{s['n_seeds']} "
        f"(attributable={s['n_seeds_attributable_tier']} "
        f"control={s['n_seeds_negative_control_tier']})", flush=True)
    nc = result["negative_control"]
    print(
        f"  negative_control: midexec_on={nc['midexec_on_total']} "
        f"multi_commits={nc['multi_action_commits_total']} "
        f"divergence={nc['mean_action_divergence_frac']} "
        f"expectation_held={nc['expectation_held']} "
        f"tier_still_valid={nc['tier_still_valid']}", flush=True)
    print(
        f"  green_arms={result['per_arm_gate']['green_arms']} "
        f"red_arms={result['per_arm_gate']['red_arms']}", flush=True)
    for c in result["criteria"]:
        print(f"  {c['name']}: passed={c['passed']} measured={c['measured']} "
              f"load_bearing={c['load_bearing']}", flush=True)
    if result["degeneracy_reason"]:
        print(f"  degeneracy_reason: {result['degeneracy_reason']}", flush=True)
    print(f"started_utc: {started.strftime('%Y%m%dT%H%M%SZ')}", flush=True)

    outcome_norm = str(result["outcome"]).upper()
    outcome_emit = outcome_norm if outcome_norm in ("PASS", "FAIL") else "FAIL"
    return outcome_emit, str(out_path), bool(args.dry_run)


if __name__ == "__main__":
    _outcome, _manifest_path, _dry_run = main()
    if _outcome is not None:
        emit_outcome(outcome=_outcome, manifest_path=_manifest_path,
                     dry_run=_dry_run)
