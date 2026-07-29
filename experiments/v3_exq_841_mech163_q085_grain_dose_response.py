#!/usr/bin/env python3
"""V3-EXQ-841 -- MECH-163 hierarchical-vs-flat falsifier + Q-085 grain dose-response.

ONE RUN, TWO QUESTIONS. This is deliberate and is the whole economy of the design
(claims.yaml Q-085 `what_would_answer`): the statistic that discriminates the two
MECH-163 architectures IS the Q-085 dose-response, so the dose arm costs ZERO
additional compute. Do NOT queue a separate Q-085 experiment.

  MECH-163 -- is post-devaluation outcome-insensitivity produced by committed-macro
      GRAIN (hierarchical, Dezfouli & Balleine 2013) or by an ARBITRATOR weighting a
      habit controller (flat)?  A_FLAT vs A_HIER_*.
  Q-085  -- is the committed-macro grain budget a CONTROLLABILITY parameter?  Does
      post-devaluation persistence rise monotonically with the REALISED mean
      committed-chunk length?  A_HIER_S2 / S3 / S5.

DV -- post-devaluation policy persistence: after a mid-episode devaluation of the
committed goal's outcome, the fraction of subsequent steps that continue the
pre-devaluation policy (operationalised as the resource-approach rate; see
`_approach_rate`). Reported RAW and under three conditionings, two of which are
pre-registered confounds that push in OPPOSITE directions (see below).

WHAT DECIDES EACH READING (pre-registered; a third outcome is explicitly allowed)
--------------------------------------------------------------------------------
  * HIERARCHICAL supported -- persistence monotone increasing in realised mean
    committed-chunk length, with A_HIER_* showing persistence and NO arbitrator
    instantiated anywhere. Grain is the cause.
  * FLAT supported -- persistence tracks arbitration weight and is invariant to
    realised chunk length once that weight is controlled for.
  * BOTH present and separable -- a third, informative outcome, pre-registered here
    rather than forced into a binary.

THE TWO OPPOSED CONFOUNDS -- both logged, both reported SEPARATELY
------------------------------------------------------------------
(1) INTERRUPTION (registered in claims.yaml Q-085). A LONGER macro meets MORE
    mid-commitment release opportunities, so it can read as LESS insensitive purely
    because it was interrupted more often. Pushes the dose-response DOWN.
    -> persistence is reported conditioned on UNINTERRUPTED macros.

(2) DEVALUATION KNOWABILITY (NEW, 2026-07-29; from the Ostlund, Winterbauer &
    Balleine 2009 lit entry, doi 10.1523/JNEUROSCI.1176-09.2009, filed `mixed` 0.72
    under evidence/literature/targeted_review_q_085/). Ostlund's sham rats were NOT
    made outcome-insensitive by chunking -- their sensitivity RELOCATED to the
    sequence boundary: they withheld the WHOLE sequence whose outcome was devalued,
    and only WITHIN-sequence adjustment was lost. Consequence for this design:
    persistence rises with realised chunk length ONLY to the extent the value change
    becomes knowable AFTER the beta gate commits. If the devaluation is fully
    knowable at chunk onset, the agent simply declines to commit, and this run
    returns a FLAT dose-response WHILE THE HYPOTHESIS IS TRUE -- a false RESOLVED-NO.
    Pushes the dose-response DOWN by a completely different route.
    -> every committed chunk records its commit tick RELATIVE to the tick at which
       the devaluation first became knowable to the agent, and persistence is
       reported conditioned on chunks committed BEFORE that tick.

Reporting only ONE of these will not disambiguate them, because they push the same
direction on the raw statistic for different reasons. Both blocks are emitted
unconditionally, on PASS runs too.

WHY "KNOWABLE" IS AN EVENT AND NOT THE DEVALUATION TICK. Setting the outcome value
does not inform the agent; CONTACTING the devalued outcome does. So the knowability
tick is the agent's FIRST post-devaluation resource contact -- the first step on
which the zero benefit is actually experienced -- not the tick the experimenter
flipped the value. Episodes with no post-devaluation contact never make it
knowable at all, and are reported as their own bucket rather than folded in.

DEVIATION FROM THE claims.yaml DESIGN, STATED PLAINLY
-----------------------------------------------------
claims.yaml names "SD-033b OFC-analog devaluation; existing driver lineage
V3-EXQ-485g/i/k/m". That route is NOT used here, on compute-feasibility grounds
that are measured rather than assumed:

  * The OFC devaluation head produces zero bias until TRAINED
    (`ofc_train_devaluation_head=False` -> last Linear zeroed), so the 485 route
    requires per-cell gradient training.
  * V3-EXQ-810a measured gradient-tracking mode in THIS env/agent at roughly
    2 s/step (its `run_cell` docstring), which is why that lineage runs
    `train_mode=False`. A per-cell trained head over any usable schedule puts a
    6-arm x 8-seed grid at ~28 h, against ~7 h for the no-grad form.
  * The chunk-formation schedule is NOT negotiable downward to buy that back:
    V3-EXQ-834 shortened it to 100 ep x 60 steps and formed ZERO chunks in all five
    arms after 7.3 h (`substrate_not_ready_requeue`), while 810a's 120 x 72 formed
    and crystallised. Gate (a) is the fragile gate; it sets the schedule.

So the devaluation here is a DIRECT OUTCOME devaluation -- `env.resource_benefit`
is driven to zero mid-episode, which is the Balleine manipulation the OFC head was
standing in for -- and the mandatory manipulation check (gate (c)) is measured on a
POLICY-INDEPENDENT probe: the benefit actually DELIVERED per post-devaluation
resource contact, which is an env-side quantity the agent's choices cannot move.
This keeps the whole run in `torch.no_grad`. The substitution is recorded in the
manifest under `devaluation_route` so no reader has to infer it.

SUPERSEDES nothing. NOT a lettered iteration of 810/834 -- different claims,
different DV, different question.

READINESS IS INHERITED, NOT RE-DERIVED
--------------------------------------
Every env / schedule / chunking constant comes from
`experiments/_lib/baselines/arc071_chunking.py`, the canonical lineage module whose
config PASSED V3-EXQ-810a (`chunk_accumulator_fires`, 2026-07-28, 8 seeds). The
claims.yaml secondary gate -- "the run should not be queued until 810's successor
clears" -- is therefore SATISFIED as of 810a. Nothing here re-tunes those numbers,
and no threshold in this script may be lowered to force a pass.

DV-SYMMETRY INVARIANCE -- mandatory per-arm declaration
------------------------------------------------------
The DV (`_approach_rate`) is a set-aggregate over steps of a distance-reduction
predicate, so it is a pure function of the realised TRAJECTORY. Its symmetry group
is therefore everything that leaves an argmax sequence fixed: a broadcast additive
constant across candidates, any monotone rescaling of candidate scores, and any
permutation of the aggregated units.

  * DEVALUATION manipulation -- NOT invariant. Driving `resource_benefit` to zero is
    not a uniform per-candidate offset and not a positive rescaling: it zeroes the
    benefit term for resource-approaching candidates specifically, which reorders
    them against non-approaching ones. And the DV's pre/post split is time-indexed,
    so it is not a permutation of the units being aggregated -- the split is exactly
    what breaks that symmetry. Holds in EVERY arm, A_OFF included.
  * DOSE manipulation (`chunk_max_size`) -- NOT invariant in A_HIER_S2/S3/S5 or
    A_FLATCHUNK_S5: it changes the realised committed length, i.e. how many
    consecutive steps execute without re-deliberation, which changes the trajectory
    itself.
  * DOSE manipulation IS invariant in A_OFF and A_FLAT -- they never construct the
    chunking operators, so `chunk_max_size` is inert there and a "dose effect" in
    those arms would be an arithmetic zero rather than a measurement. This is why
    the dose statistic is computed over DOSE_ARMS only and those two arms carry no
    dose value. Stating it because an unstated version of exactly this is the
    V3-EXQ-604c defect.

SEQUENCING TRAP (claims.yaml, do not violate): building a value-driven
mid-commitment release would DESTROY the phenomenon under test. It must come AFTER
this question is answered. Every optional release mechanism is pinned OFF below and
the pin is recorded in the manifest.

ETHICS PREFLIGHT (descriptive, non-enforced; SENT-0 -- V3 is not claimed sentient):
involves_negative_valence False (the num_hazards=2 harm stream is pre-ethical
instrumentation, unchanged from 810a), involves_suffering_like_state False (MECH-219
accumulator off), involves_self_model False, involves_inescapability False,
involves_offline_replay_over_harm False (sleep off), involves_social_mind False,
involves_human_data False. decision: allow.
"""

import argparse
import json
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
from experiments._harness import StepHarness  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_841_mech163_q085_grain_dose_response"
EXPERIMENT_PURPOSE = "evidence"

# Tag ONLY what this implementation directly tests. ARC-071 and MECH-323 are
# EXERCISED here (chunk formation, the size budget) but as PRECONDITIONS of the
# measurement, not as the hypothesis under test -- tagging them would inflate their
# evidence records with a run that cannot move them either way, which is the
# peripheral-co-tag inflation the re-derive brake's per-claim override exists to
# undo. Q-085's own note also states that registering the controllability semantics
# against MECH-323 would hide the coupling the question exists to surface.
CLAIM_IDS = ["MECH-163", "Q-085"]

SEEDS = [101, 202, 303, 404, 505, 606, 707, 808]

# --- Arms --------------------------------------------------------------------
A_OFF = "A_OFF"
A_FLAT = "A_FLAT"
A_HIER_S2 = "A_HIER_S2"
A_HIER_S3 = "A_HIER_S3"
A_HIER_S5 = "A_HIER_S5"
A_FLATCHUNK_S5 = "A_FLATCHUNK_S5"
ARMS = [A_OFF, A_FLAT, A_HIER_S2, A_HIER_S3, A_HIER_S5, A_FLATCHUNK_S5]

# The dose axis proper: chunking ON, arbitration OFF, chunk_max_size swept.
DOSE_ARMS = (A_HIER_S2, A_HIER_S3, A_HIER_S5)
DOSE_BY_ARM: Dict[str, int] = {A_HIER_S2: 2, A_HIER_S3: 3, A_HIER_S5: 5,
                               A_FLATCHUNK_S5: 5}
# Any arm that instantiates the chunking operators at all.
CHUNKING_ON_ARMS = DOSE_ARMS + (A_FLATCHUNK_S5,)
# Any arm that instantiates the MECH-477 arbitrator.
ARBITRATION_ON_ARMS = (A_FLAT, A_FLATCHUNK_S5)
# A_FLATCHUNK_S5 crosses arbitration WITH chunking at one dose level, so the dose
# axis is not confounded with the arm (synthesis 4b). One cell is enough to
# deconfound; a full cross would double the grid for no extra discrimination.

N_EPISODES_TRAIN = base.N_EPISODES          # 120 -- the 810a-proven formation schedule
STEPS_PER_EPISODE = base.STEPS_PER_EPISODE  # 72
N_EPISODES_PROBE = 30                       # devaluation probe episodes
DEVAL_STEP = STEPS_PER_EPISODE // 2         # 36: devaluation applied mid-episode

# --- Pre-registered thresholds. Defined HERE, never inferred post-hoc. -------
# Gate (a): committed arc071_chunk trajectories per seed. ZERO is a READINESS
# FAILURE that scores nothing, not a null. This is the gate V3-EXQ-810 failed and
# 810a cleared, and Jin & Costa 2010 is the reason it stays strict: sequence
# boundaries are a property of a CONSOLIDATED sequence, so an unconsolidated seed
# has no boundaries to be insensitive at and its persistence means nothing.
MIN_COMMITTED_CHUNKS = 1
# Gate (b): the REALISED mean committed length must actually differ across the dose
# conditions. Configuring chunk_max_size is not the manipulation.
MIN_REALISED_LENGTH_SPREAD = 0.5
# Gate (c): the devaluation must actually devalue, measured policy-independently.
MAX_POST_DEVAL_DELIVERED_BENEFIT = 1e-9
MIN_PRE_DEVAL_DELIVERED_BENEFIT = 0.05
# Gate (d): every input to the DV's own statistic, not merely the prominent one.
MIN_POST_DEVAL_STEPS = 200          # per cell, denominator of the persistence rate
MIN_PRE_DEVAL_APPROACH_RATE = 0.05  # a policy with no approach cannot persist in one

# Readiness inherited from 810a (same statistics, same floors).
OUTCOME_SPREAD_FLOOR = 0.05
SYMBOLS_PER_TRIAL_FLOOR = float(base.CHUNK_MAX_SIZE - 1)  # 4.0
HARM_STEP_FRACTION_FLOOR = 0.01

SEED_PASS_FRACTION = 0.75  # 6 of 8

# Q-085 RESOLVED-YES bar: Spearman rho of persistence on REALISED mean committed
# length across the dose arms, plus an absolute monotonicity margin so a
# rank-perfect but numerically trivial rise cannot carry the claim.
MONOTONIC_RHO_FLOOR = 0.6
MONOTONIC_ABS_MARGIN = 0.05

_ZG = ZGoalStreamAccumulator()


# --- Release mechanisms ------------------------------------------------------
# The five mid-commitment releases named in Q-085. FOUR are optional and are pinned
# OFF here; MECH-091 (acute z_harm urgency) is unconditional in select_action and is
# therefore the ONLY live release in this run. That is what makes the interruption
# confound tractable: an interrupted macro has exactly one possible cause.
RELEASES_PINNED_OFF = {
    "use_maintenance_release": False,            # MECH-342 degraded readiness
    "use_natural_commit_urgency_release": False,  # rung-6 natural commit + SD-033e
    "use_vs_commit_release": False,               # MECH-321 / ARC-070 V_s failure
    "use_closure_decommit": False,                # SD-034 closure de-commit
}
LIVE_RELEASE = "MECH-091 (acute z_harm urgency; unconditional in select_action)"


def _arm_flags(arm: str) -> Dict[str, Any]:
    """Switches for one arm. Built ON TOP of the lineage OFF-path flags so the
    A_OFF slice is the canonical one by construction."""
    flags = base.off_arm_flags()
    flags.update(RELEASES_PINNED_OFF)
    if arm == A_OFF:
        return flags
    if arm in CHUNKING_ON_ARMS:
        flags.update({
            "use_policy_chunking": True,
            "use_chunk_maintenance": True,
            # Injection ON: this run needs E3 to actually SEE and commit chunks.
            # (The 810a lineage kept it OFF because a readiness probe must not
            # perturb the behaviour a later run measures. This IS that later run.)
            "use_chunk_proposal_injection": True,
            "use_chunk_all_position_credit": True,
            "chunk_max_size": DOSE_BY_ARM[arm],
        })
    if arm in ARBITRATION_ON_ARMS:
        flags["use_dualsystem_arbitration"] = True
    return flags


def _config_slice(arm: str) -> Dict[str, Any]:
    """Fingerprint slice. Carries the probe schedule too, because this lineage's
    cell computes a devaluation phase the arc071_chunking mint does not -- so its
    OFF cell must NOT collide with that mint's fingerprint."""
    slice_ = base.arm_config_slice(_arm_flags(arm))
    slice_["probe"] = {
        "n_episodes_probe": N_EPISODES_PROBE,
        "deval_step": DEVAL_STEP,
        "devaluation_route": "env_resource_benefit_to_zero",
    }
    return slice_


def _arm_ctx(arm: str) -> Dict[str, Any]:
    return {
        "id": arm,
        "chunking_on": arm in CHUNKING_ON_ARMS,
        "arbitration_on": arm in ARBITRATION_ON_ARMS,
        "dose": DOSE_BY_ARM.get(arm),
    }


# --- Precondition specs (regime-conditioned) ---------------------------------
# Every precondition declares applies_to. A gate that is not meaningful for an arm
# is SCOPED OUT of it, never failed by it -- the V3-EXQ-785 rule. A_OFF instantiates
# no chunking operators, so the accumulator-facing gates are not meaningful there;
# it is retained as the devaluation-response baseline, and its DV gates DO apply.
_CHUNKING_ONLY_NOTE = (
    "A_OFF and A_FLAT never construct the chunking operators, so an "
    "accumulator-facing gate is NOT MEANINGFUL there (disposition (a)) rather "
    "than failed. Both arms remain fully scorable on the DV gates -- A_OFF is "
    "the devaluation-response baseline and A_FLAT is MECH-163's flat leg."
)

PRECONDITIONS = [
    PreconditionSpec(
        name="episode_outcome_spread_supra_floor",
        description=("MECH-323's evaluative gate is RELATIVE to a running baseline, "
                     "so a flat outcome stream makes formation structurally "
                     "impossible whatever the substrate is doing. Asserts the SAME "
                     "statistic the gate routes on (spread), not a magnitude proxy."),
        control=("Per-episode outcome pstdev over the 120-episode formation phase, "
                 "the same schedule on which V3-EXQ-810a measured 0.349."),
        threshold=OUTCOME_SPREAD_FLOOR,
        applies_to=lambda ctx: bool(ctx.get("chunking_on")),
        applies_note=_CHUNKING_ONLY_NOTE,
    ),
    PreconditionSpec(
        name="symbols_per_trial_supports_size_range",
        description=("note_outcome breaks out at len(actions) < size, so a buffer "
                     "shorter than chunk_max_size makes the top of the size budget "
                     "structurally unreachable -- V3-EXQ-810's actual failure."),
        control=("Recorded symbols per trial (chunk_acc_n_steps / n_outcomes) on "
                 "the 72-step schedule, where 810a measured 9.41 against this 4.0 "
                 "floor. At 810's 24-step schedule it was 3.00 and sizes 4-5 were "
                 "structurally unreachable."),
        threshold=SYMBOLS_PER_TRIAL_FLOOR,
        applies_to=lambda ctx: bool(ctx.get("chunking_on")),
        applies_note=_CHUNKING_ONLY_NOTE,
    ),
    PreconditionSpec(
        name="salient_harm_events_fire",
        description=("harm_signal < 0 is the SOLE trigger for MECH-091's "
                     "clock.phase_reset(), which de-periodises the E3 tick. It is "
                     "also this run's only live release, so the interruption "
                     "confound is unmeasurable without it."),
        control=("Fraction of formation-phase env steps with harm_signal < 0 on the "
                 "num_hazards=2 grid; 810a measured 0.0267 against this 0.01 floor."),
        threshold=HARM_STEP_FRACTION_FLOOR,
    ),
    # --- Gate (a) ---
    PreconditionSpec(
        name="committed_arc071_chunks_formed",
        description=("Gate (a). Committed trajectories with metadata source "
                     "arc071_chunk, counted on the PROBE agent the DV is read "
                     "from. ZERO is a readiness failure that scores nothing, NOT "
                     "a null (V3-EXQ-810; Jin & Costa 2010 -- boundaries belong "
                     "to a CONSOLIDATED sequence, so an unconsolidated seed has "
                     "no boundary to be insensitive at)."),
        control=("Segmented commitment count on the probe phase. V3-EXQ-810a "
                 "cleared formation end-to-end on this config; 810 and 834 did "
                 "not, which is why this stays strict."),
        threshold=float(MIN_COMMITTED_CHUNKS),
        applies_to=lambda ctx: bool(ctx.get("chunking_on")),
        applies_note=_CHUNKING_ONLY_NOTE,
    ),
    # --- Gate (c) ---
    PreconditionSpec(
        name="pre_deval_delivered_benefit_supra_floor",
        description=("Gate (c), positive half. Benefit DELIVERED per resource "
                     "contact BEFORE devaluation, an env-side quantity no policy "
                     "can move. Without it there is nothing to devalue."),
        control=("Mean positive harm_signal per pre-devaluation resource contact, "
                 "against env resource_benefit = 0.3."),
        threshold=MIN_PRE_DEVAL_DELIVERED_BENEFIT,
    ),
    PreconditionSpec(
        name="post_deval_delivered_benefit_at_zero",
        description=("Gate (c), negative half. Benefit delivered per resource "
                     "contact AFTER devaluation must be zero. Policy-independent, "
                     "so a null DV cannot be a devaluation failure in disguise."),
        control=("Mean positive harm_signal per post-devaluation resource contact "
                 "after resource_benefit is driven to 0.0. An upper-bound CEILING: "
                 "measured << threshold means the check PASSED."),
        threshold=MAX_POST_DEVAL_DELIVERED_BENEFIT,
        direction="upper",
    ),
    # --- Gate (d): every input to the DV's own statistic ---
    PreconditionSpec(
        name="post_deval_steps_supra_floor",
        description=("Gate (d). The persistence rate's own DENOMINATOR. The 811 "
                     "precedent: distinguish 'no data' from 'zero rate'."),
        control=("Count of post-devaluation steps per cell; 30 probe episodes x 36 "
                 "post-devaluation steps = 1080 nominal against this 200 floor, so "
                 "a shortfall means episodes were terminating early."),
        threshold=float(MIN_POST_DEVAL_STEPS),
    ),
    PreconditionSpec(
        name="pre_deval_approach_rate_supra_floor",
        description=("Gate (d). The persistence rate's own NUMERATOR baseline: a "
                     "policy that never approached the resource cannot persist in "
                     "doing so, and its ~0 persistence would be an artefact of "
                     "having nothing to persist in."),
        control=("The SAME _approach_rate statistic the DV routes on, measured on "
                 "the PRE-devaluation half of each probe episode -- a positive "
                 "control by construction, since the outcome is still valuable "
                 "there. Not a magnitude proxy (the V3-EXQ-643 rule)."),
        threshold=MIN_PRE_DEVAL_APPROACH_RATE,
    ),
]


def _worst(rows: List[Dict[str, Any]], key: str,
           mode: str = "min") -> Tuple[float, Optional[int]]:
    """Worst cell for `key` plus the offending seed.

    NEVER a mean: `met` is a worst-case quantifier over cells, so a mean `measured`
    would let one out-of-band cell recompute as MET while our own flag says False
    (the V3-EXQ-779b mean-vs-quantifier defect).
    """
    vals = [(float(r.get(key, 0.0)), int(r.get("seed", -1))) for r in rows]
    if not vals:
        return 0.0, None
    return min(vals) if mode == "min" else max(vals)


def _approach_rate(records: List[Dict[str, Any]]) -> float:
    """Fraction of steps that reduced Manhattan distance to the nearest resource.

    This is the persistence DV's primitive: continuing to pursue the (now
    devalued) outcome. Arm-neutral -- it reads the trajectory, not the mechanism
    that produced it, so it cannot advantage a chunked arm by construction.
    """
    moves = [r for r in records if r.get("d_before") is not None
             and r.get("d_after") is not None]
    if not moves:
        return 0.0
    closer = sum(1 for r in moves if r["d_after"] < r["d_before"])
    return closer / len(moves)


def _nearest_resource_distance(env: Any) -> Optional[int]:
    """Manhattan distance from the agent to the nearest resource cell."""
    import numpy as np
    try:
        rcode = env.ENTITY_TYPES["resource"]
        cells = np.argwhere(env.grid == rcode)
        if cells.size == 0:
            return None
        ax, ay = int(env.agent_x), int(env.agent_y)
        return int(np.min(np.abs(cells[:, 0] - ax) + np.abs(cells[:, 1] - ay)))
    except Exception:
        return None


def _committed_chunk_state(agent: Any) -> Tuple[Optional[Tuple[int, ...]], int]:
    """(chunk_sequence, committed_step_idx) when an arc071_chunk is committed."""
    e3 = getattr(agent, "e3", None)
    if e3 is None:
        return None, 0
    traj = getattr(e3, "_committed_trajectory", None)
    if traj is None:
        traj = getattr(e3, "_persistent_committed_trajectory", None)
    meta = getattr(traj, "metadata", None) if traj is not None else None
    if not meta or meta.get("source") != "arc071_chunk":
        return None, 0
    seq = tuple(int(a) for a in meta.get("chunk_sequence", ()))
    return (seq or None), int(getattr(agent, "_committed_step_idx", 0))


def _segment_commitments(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Segment the per-step record stream into commitment episodes.

    A NEW commitment begins when the committed sequence changes, or when
    _committed_step_idx fails to advance from the previous step (a reset). The
    REALISED length is the number of steps actually executed under one commitment
    -- never the configured chunk_max_size, which is gate (b)'s whole point.
    """
    segs: List[Dict[str, Any]] = []
    cur: Optional[Dict[str, Any]] = None
    prev_seq: Optional[Tuple[int, ...]] = None
    prev_idx = -1
    for r in records:
        seq = r.get("chunk_seq")
        idx = int(r.get("committed_step_idx", 0))
        if seq is None:
            cur = None
            prev_seq, prev_idx = None, -1
            continue
        new = (cur is None) or (seq != prev_seq) or (idx <= prev_idx)
        if new:
            cur = {"seq": seq, "nominal_length": len(seq), "realised_length": 0,
                   "commit_tick": int(r["tick"]), "steps": []}
            segs.append(cur)
        cur["realised_length"] += 1
        cur["steps"].append(r)
        prev_seq, prev_idx = seq, idx
    for s in segs:
        # Uninterrupted = the commitment ran its full nominal sequence.
        s["uninterrupted"] = bool(s["realised_length"] >= s["nominal_length"])
    return segs


def _run_probe_episode(harness: Any, agent: Any, env: Any, ep: int,
                       tick0: int) -> Dict[str, Any]:
    """One devaluation probe episode. Returns the per-step record stream.

    Devaluation is applied at DEVAL_STEP by driving env.resource_benefit to zero.
    Nothing tells the agent; it learns only by CONTACTING the devalued resource,
    which is why `knowable_tick` is the first post-devaluation contact and not the
    devaluation tick itself (the Ostlund 2009 point).
    """
    baseline_benefit = float(getattr(env, "resource_benefit", 0.0))
    records: List[Dict[str, Any]] = []
    state = {"step": 0, "knowable_tick": None,
             "pre_contacts": 0, "pre_benefit": 0.0,
             "post_contacts": 0, "post_benefit": 0.0}

    def on_step(r: Any) -> None:
        i = state["step"]
        tick = tick0 + i
        d_before = state.get("d_before")
        d_after = _nearest_resource_distance(env)
        contact = bool((r.info or {}).get("event_type") == "resource")
        post = i >= DEVAL_STEP
        if contact:
            # Delivered benefit is the POSITIVE part of the env signal: the
            # policy-independent manipulation check for gate (c).
            delivered = max(0.0, float(r.harm_signal))
            if post:
                state["post_contacts"] += 1
                state["post_benefit"] += delivered
                if state["knowable_tick"] is None:
                    state["knowable_tick"] = tick
            else:
                state["pre_contacts"] += 1
                state["pre_benefit"] += delivered
        seq, idx = _committed_chunk_state(agent)
        records.append({
            "tick": tick, "step_in_ep": i, "post_deval": post,
            "d_before": d_before, "d_after": d_after,
            "chunk_seq": seq, "committed_step_idx": idx,
            "resource_contact": contact,
        })
        state["d_before"] = d_after
        state["step"] = i + 1
        # Apply the devaluation exactly once, at the boundary.
        if state["step"] == DEVAL_STEP:
            env.resource_benefit = 0.0

    env.resource_benefit = baseline_benefit
    state["d_before"] = None
    harness.run_episode(STEPS_PER_EPISODE, on_step=on_step)
    env.resource_benefit = baseline_benefit  # restore for the next episode
    return {
        "records": records,
        "knowable_tick": state["knowable_tick"],
        "pre_contacts": state["pre_contacts"], "pre_benefit": state["pre_benefit"],
        "post_contacts": state["post_contacts"],
        "post_benefit": state["post_benefit"],
        "n_steps": state["step"],
    }


def _run_formation(harness: Any, agent: Any, n_episodes: int, *,
                   progress_every: int, label: str) -> Dict[str, Any]:
    """The formation phase, on an agent the CALLER keeps.

    Mirrors `arc071_chunking.run_cell`'s loop exactly -- one `note_chunk_outcome`
    per trial boundary, `StepHarness` driving the full agent loop so
    `update_residue` runs and MECH-091 can fire `clock.phase_reset()`. It is
    reproduced here rather than called because `base.run_cell` builds and DISCARDS
    its own agent, and this experiment's probe phase MUST run on the agent that
    actually formed the chunks -- a fresh agent would enter the probe with an empty
    chunk library, gate (a) would read ~0 in every arm, and the whole run would
    self-route `substrate_not_ready_requeue` having measured nothing. That is
    V3-EXQ-834's outcome, arrived at by a different route.

    The shared lineage module is deliberately NOT edited to return its agent:
    `_lib/**` is bound into `substrate_hash`, so changing it would churn the
    fingerprint for every arc071_chunking consumer including 810a's mint.
    """
    episode_outcomes: List[float] = []
    per_episode_symbols: List[int] = []
    n_harm_steps = 0
    n_steps_total = 0
    prev_symbols = 0
    for ep in range(n_episodes):
        results = harness.run_episode(STEPS_PER_EPISODE)
        ep_reward = 0.0
        for r in results:
            ep_reward += r.harm_signal
            if r.harm_signal < 0:
                n_harm_steps += 1
        n_steps_total += len(results)
        agent.note_chunk_outcome(ep_reward)          # trial boundary; no-op when off
        episode_outcomes.append(ep_reward)
        st_now = agent.get_chunking_state()
        symbols_now = int(st_now.get("chunk_acc_n_steps", 0))
        per_episode_symbols.append(symbols_now - prev_symbols)
        prev_symbols = symbols_now
        if (ep + 1) % progress_every == 0:
            print(f"  [train] chunk {label} ep {ep+1}/{n_episodes} "
                  f"formed={st_now.get('chunk_acc_n_formed', 0)} "
                  f"cryst={st_now.get('chunk_lib_n_crystallised', 0)} "
                  f"symbols/ep={per_episode_symbols[-1]}", flush=True)
    st = agent.get_chunking_state()
    return {
        "chunking_state": st,
        "episode_outcomes": episode_outcomes,
        "per_episode_symbols": per_episode_symbols,
        "n_harm_steps": n_harm_steps,
        "n_steps_total": n_steps_total,
        "formed_chunk_lengths": base.formed_chunk_lengths(agent),
        "chunking_instantiated": bool(agent.policy_chunking is not None),
    }


def _run_cell(arm: str, seed: int, *, n_train: int, n_probe: int) -> Dict[str, Any]:
    """One (arm, seed) cell: formation phase, then devaluation probe phase."""
    flags = _arm_flags(arm)
    # Scale the progress cadence to the schedule so a --dry-run still EMITS at
    # least one "[train] ... ep N/M" line: the skill's smoke check verifies that
    # pattern, and a fixed every-30 on a 2-episode dry-run prints nothing at all,
    # which makes the check unverifiable exactly where it is cheap to verify.
    train_every = max(1, min(30, n_train // 2))
    probe_every = max(1, min(10, n_probe // 2))
    with arm_cell(seed,
                  config_slice=_config_slice(arm),
                  config_slice_declared=True,
                  script_path=Path(__file__),
                  include_driver_script_in_hash=False) as cell:
        print(f"Seed {seed} Condition {arm}", flush=True)

        # ONE agent for BOTH phases. The probe must read the devaluation response
        # of the agent that actually built the chunk library; see _run_formation.
        # train_mode=False (no_grad) throughout: nothing here backprops, and 810a
        # measured gradient mode at ~2 s/step in this env.
        env = base.build_env(seed)
        agent = base.build_agent(env, flags)
        harness = StepHarness(agent, env, train_mode=False, seed=seed)

        # --- PHASE 1: formation. The 810a-proven schedule, verbatim. ---
        train = _run_formation(harness, agent, n_train,
                               progress_every=train_every,
                               label=f"{arm} seed={seed}")

        # --- PHASE 2: devaluation probe, SAME agent, chunk library intact.
        # agent.reset() (fired per episode by run_episode) does not touch
        # policy_chunking, which is what lets the library persist -- necessarily so,
        # since the formation phase accumulates across 120 episode resets.
        all_records: List[Dict[str, Any]] = []
        knowable_ticks: List[Optional[int]] = []
        pre_c = pre_b = post_c = post_b = 0.0
        tick = 0
        for ep in range(n_probe):
            out = _run_probe_episode(harness, agent, env, ep, tick)
            tick += out["n_steps"]
            all_records.extend(out["records"])
            knowable_ticks.append(out["knowable_tick"])
            pre_c += out["pre_contacts"]; pre_b += out["pre_benefit"]
            post_c += out["post_contacts"]; post_b += out["post_benefit"]
            if (ep + 1) % probe_every == 0:
                # DELIBERATELY NOT the "ep N/M" shape. experiment_runner's
                # RE_EP_PROGRESS is a bare r'ep\s+(\d+)/(\d+)' and it OVERWRITES
                # episodes_per_run with whatever denominator it last saw, so a
                # second phase printing "ep 10/30" would clobber the formation
                # phase's /120 and break overall_pct for the whole run.
                print(f"  [probe] {arm} seed={seed} episode {ep+1} of {n_probe} "
                      f"records={len(all_records)}", flush=True)
        _ZG.observe(agent)

        # --- DV assembly ---
        pre = [r for r in all_records if not r["post_deval"]]
        post = [r for r in all_records if r["post_deval"]]
        pre_rate = _approach_rate(pre)
        raw_persistence = _approach_rate(post)

        segs = _segment_commitments(all_records)
        realised = [s["realised_length"] for s in segs]
        uninterrupted = [s for s in segs if s["uninterrupted"]]

        # Confound (1): uninterrupted macros only.
        unint_steps = [r for s in uninterrupted for r in s["steps"]
                       if r["post_deval"]]
        persistence_uninterrupted = _approach_rate(unint_steps)

        # Confound (2, NEW): chunks committed BEFORE the devaluation became
        # knowable. Episode-scoped -- a tick is only comparable to the knowability
        # tick of its OWN episode.
        know_by_ep = {i: kt for i, kt in enumerate(knowable_ticks)}
        pre_knowable_steps: List[Dict[str, Any]] = []
        n_pre_knowable_segs = n_post_knowable_segs = n_never_knowable_segs = 0
        for s in segs:
            ep_idx = s["commit_tick"] // STEPS_PER_EPISODE
            kt = know_by_ep.get(ep_idx)
            if kt is None:
                n_never_knowable_segs += 1
                continue
            if s["commit_tick"] < kt:
                n_pre_knowable_segs += 1
                pre_knowable_steps.extend(r for r in s["steps"] if r["post_deval"])
            else:
                n_post_knowable_segs += 1
        persistence_pre_knowable = _approach_rate(pre_knowable_steps)

        row: Dict[str, Any] = {
            "arm": arm, "seed": seed, "dose_chunk_max_size": DOSE_BY_ARM.get(arm),
            # readiness (formation phase)
            "episode_outcome_spread": (
                statistics.pstdev(train["episode_outcomes"])
                if len(train["episode_outcomes"]) > 1 else 0.0),
            "symbols_per_trial": (
                float(train["chunking_state"].get("chunk_acc_n_steps", 0))
                / max(1, train["chunking_state"].get("chunk_acc_n_outcomes", 0))),
            "harm_step_fraction": (train["n_harm_steps"]
                                   / max(1, train["n_steps_total"])),
            "chunks_formed": float(train["chunking_state"].get(
                "chunk_acc_n_formed", 0)),
            "formed_chunk_lengths": train["formed_chunk_lengths"],
            "chunking_instantiated": train["chunking_instantiated"],
            # gate (a) -- on the PROBE agent, which is the one the DV is read from
            "committed_arc071_chunks": float(len(segs)),
            # gate (b)
            "realised_mean_committed_length": (
                statistics.fmean(realised) if realised else 0.0),
            "realised_committed_lengths": realised,
            # gate (c) -- policy-independent
            "pre_deval_delivered_benefit": (pre_b / pre_c) if pre_c else 0.0,
            "post_deval_delivered_benefit": (post_b / post_c) if post_c else 0.0,
            "pre_deval_contacts": pre_c, "post_deval_contacts": post_c,
            # gate (d)
            "post_deval_steps": float(len(post)),
            "pre_deval_approach_rate": pre_rate,
            # the DV, raw and under both confound conditionings
            "persistence_raw": raw_persistence,
            "persistence_uninterrupted": persistence_uninterrupted,
            "persistence_pre_knowable": persistence_pre_knowable,
            "n_segments": len(segs),
            "n_segments_uninterrupted": len(uninterrupted),
            "n_segments_pre_knowable": n_pre_knowable_segs,
            "n_segments_post_knowable": n_post_knowable_segs,
            "n_segments_never_knowable": n_never_knowable_segs,
            "n_episodes_never_knowable": sum(
                1 for kt in knowable_ticks if kt is None),
            "arbitration_instantiated": bool(
                getattr(agent, "dualsystem_arbitration", None) is not None),
            "z_goal_stream_stats": harness.z_goal_stream_stats(),
        }
        cell.stamp(row)
        verdict = "PASS" if row["committed_arc071_chunks"] > 0 or \
            arm not in CHUNKING_ON_ARMS else "FAIL"
        print(f"verdict: {verdict}", flush=True)
    return row


def _spearman(xs: List[float], ys: List[float]) -> float:
    """Spearman rho. Returns 0.0 when either side has no rank spread."""
    n = len(xs)
    if n < 3 or len(set(xs)) < 2 or len(set(ys)) < 2:
        return 0.0

    def rank(vs: List[float]) -> List[float]:
        order = sorted(range(len(vs)), key=lambda i: vs[i])
        r = [0.0] * len(vs)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and vs[order[j + 1]] == vs[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = rank(xs), rank(ys)
    mx, my = statistics.fmean(rx), statistics.fmean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = sum((a - mx) ** 2 for a in rx) ** 0.5
    dy = sum((b - my) ** 2 for b in ry) ** 0.5
    return (num / (dx * dy)) if dx > 0 and dy > 0 else 0.0


def run_experiment(*, n_train: int, n_probe: int,
                   seeds: List[int]) -> Dict[str, Any]:
    arm_ctxs = [_arm_ctx(a) for a in ARMS]
    assert_no_structurally_unsatisfiable_gate(PRECONDITIONS, arm_ctxs)

    rows: List[Dict[str, Any]] = []
    for arm in ARMS:
        for seed in seeds:
            rows.append(_run_cell(arm, seed, n_train=n_train, n_probe=n_probe))

    # Worst cell per gated readout, with the offending seed. `measured` is the
    # EXTREMUM, never a mean: the indexer recomputes `met` from it and our own
    # met is a worst-case quantifier over cells, so a mean would let one
    # out-of-band cell recompute as MET (the V3-EXQ-779b defect).
    GATED = [
        ("episode_outcome_spread_supra_floor", "episode_outcome_spread", "min"),
        ("symbols_per_trial_supports_size_range", "symbols_per_trial", "min"),
        ("salient_harm_events_fire", "harm_step_fraction", "min"),
        ("committed_arc071_chunks_formed", "committed_arc071_chunks", "min"),
        ("pre_deval_delivered_benefit_supra_floor",
         "pre_deval_delivered_benefit", "min"),
        # upper-bound ceiling -> the WORST cell is the LARGEST one.
        ("post_deval_delivered_benefit_at_zero",
         "post_deval_delivered_benefit", "max"),
        ("post_deval_steps_supra_floor", "post_deval_steps", "min"),
        ("pre_deval_approach_rate_supra_floor", "pre_deval_approach_rate", "min"),
    ]
    arm_gates: List[Dict[str, Any]] = []
    offending: Dict[str, Dict[str, Any]] = {}
    for arm in ARMS:
        cells = [r for r in rows if r["arm"] == arm]
        measured: Dict[str, float] = {}
        offending[arm] = {}
        for pname, key, mode in GATED:
            val, cell_seed = _worst(cells, key, mode)
            measured[pname] = val
            offending[arm][pname] = cell_seed
        arm_gates.append(evaluate_arm_gate(arm, _arm_ctx(arm), PRECONDITIONS,
                                           measured))
    agg = aggregate_arm_gates(arm_gates)
    green = set(agg.get("green_arms", []))

    # --- Q-085 dose-response, computed on GREEN dose arms only ---------------
    def _arm_mean(arm: str, key: str) -> float:
        cells = [r[key] for r in rows if r["arm"] == arm]
        return statistics.fmean(cells) if cells else 0.0

    dose_green = [a for a in DOSE_ARMS if a in green]
    lengths = [_arm_mean(a, "realised_mean_committed_length") for a in dose_green]
    dose_stats: Dict[str, Any] = {}
    for label, key in (("raw", "persistence_raw"),
                       ("uninterrupted", "persistence_uninterrupted"),
                       ("pre_knowable", "persistence_pre_knowable")):
        pers = [_arm_mean(a, key) for a in dose_green]
        rho = _spearman(lengths, pers)
        margin = (max(pers) - min(pers)) if pers else 0.0
        dose_stats[label] = {
            "arms": dose_green, "realised_lengths": lengths,
            "persistence": pers, "spearman_rho": rho,
            "abs_margin": margin,
            "monotonic": bool(rho >= MONOTONIC_RHO_FLOOR
                              and margin >= MONOTONIC_ABS_MARGIN),
        }

    # Gate (b): the dose axis must be LIVE -- realised lengths must actually differ.
    length_spread = (max(lengths) - min(lengths)) if len(lengths) > 1 else 0.0
    dose_axis_live = bool(len(dose_green) >= 3
                          and length_spread >= MIN_REALISED_LENGTH_SPREAD)

    # --- Criteria -----------------------------------------------------------
    # C1 is load-bearing for Q-085; C2 for MECH-163's flat leg.
    c1 = bool(dose_axis_live and dose_stats["pre_knowable"]["monotonic"])
    arb_instantiated_anywhere = any(r["arbitration_instantiated"] for r in rows)
    hier_persist = statistics.fmean(
        [_arm_mean(a, "persistence_pre_knowable") for a in dose_green]
    ) if dose_green else 0.0
    off_persist = _arm_mean(A_OFF, "persistence_pre_knowable")
    c2 = bool(A_HIER_S5 in green and hier_persist > off_persist + MONOTONIC_ABS_MARGIN)
    c3 = bool(A_FLAT in green)

    criteria = [
        {"name": "C1_dose_response_monotone_in_realised_length",
         "load_bearing": True, "passed": c1},
        {"name": "C2_hierarchical_persistence_without_arbitrator",
         "load_bearing": True, "passed": c2},
        {"name": "C3_flat_arm_scorable", "load_bearing": False, "passed": c3},
    ]
    overall_pass = c1 and c2

    if c1 and not arb_instantiated_anywhere:
        reading = "hierarchical_supported"
    elif c3 and not c1:
        reading = "flat_supported_or_grain_invariant"
    elif c1 and arb_instantiated_anywhere:
        reading = "both_effects_present_check_separability"
    else:
        reading = "inconclusive"
    if not agg.get("non_degenerate", False):
        reading = "substrate_not_ready_requeue"

    return {
        "rows": rows, "arm_gates": arm_gates, "agg": agg,
        "offending_cells": offending,
        "dose_stats": dose_stats, "dose_axis_live": dose_axis_live,
        "realised_length_spread": length_spread,
        "criteria": criteria, "overall_pass": overall_pass,
        "reading": reading,
        "arbitration_instantiated_anywhere": arb_instantiated_anywhere,
    }


def main() -> Tuple[str, Path, bool]:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    t0 = time.perf_counter()
    if args.dry_run:
        n_train, n_probe, seeds = 2, 2, [101, 202]
    else:
        n_train, n_probe, seeds = N_EPISODES_TRAIN, N_EPISODES_PROBE, SEEDS

    res = run_experiment(n_train=n_train, n_probe=n_probe, seeds=seeds)
    agg = res["agg"]
    outcome = "PASS" if res["overall_pass"] else "FAIL"

    manifest: Dict[str, Any] = {
        "run_id": (f"{EXPERIMENT_TYPE}_"
                   f"{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}_v3"),
        "experiment_type": EXPERIMENT_TYPE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "claim_ids": CLAIM_IDS,
        "outcome": outcome,
        "timestamp_utc": datetime.utcnow().strftime("%Y%m%dT%H%M%SZ"),
        "dry_run": bool(args.dry_run),
        "devaluation_route": "env_resource_benefit_to_zero",
        "devaluation_route_note": (
            "claims.yaml names the SD-033b OFC-analog head; that route needs "
            "per-cell gradient training, measured at ~2 s/step in this env by "
            "V3-EXQ-810a, which puts the grid at ~28 h. The formation schedule "
            "cannot be shortened to buy it back (V3-EXQ-834 formed zero chunks at "
            "100x60). Direct outcome devaluation keeps the run in no_grad; the "
            "gate (c) manipulation check is policy-independent either way."),
        "live_release_mechanism": LIVE_RELEASE,
        "releases_pinned_off": RELEASES_PINNED_OFF,
        "ethics_preflight": {
            "involves_negative_valence": False,
            "involves_suffering_like_state": False,
            "involves_self_model": False,
            "involves_inescapability_or_helplessness": False,
            "involves_offline_replay_over_harm": False,
            "involves_social_mind_or_language": False,
            "involves_human_data_or_clinical_context": False,
            "decision": "allow",
        },
        "dv_symmetry_declaration": {
            "dv": "_approach_rate: set-aggregate over steps of a distance-reduction "
                  "predicate; a pure function of the realised trajectory",
            "symmetry_group": ("broadcast additive constant across candidates; any "
                               "monotone rescaling of candidate scores; permutation "
                               "of the aggregated units"),
            "devaluation_not_invariant_in": ARMS,
            "dose_not_invariant_in": list(CHUNKING_ON_ARMS),
            "dose_invariant_in": [A_OFF, A_FLAT],
            "dose_invariant_note": ("chunk_max_size is INERT in the chunking-OFF "
                                    "arms, so a dose effect there would be an "
                                    "arithmetic zero, not a measurement. The dose "
                                    "statistic is computed over DOSE_ARMS only."),
        },
        "arm_results": res["rows"],
        "per_arm_gate": agg.get("per_arm_gate"),
        "green_arms": agg.get("green_arms"),
        "red_arms": agg.get("red_arms"),
        "dose_response": res["dose_stats"],
        "dose_axis_live": res["dose_axis_live"],
        "realised_length_spread": res["realised_length_spread"],
        "arbitration_instantiated_anywhere": res["arbitration_instantiated_anywhere"],
        "non_degenerate": bool(agg.get("non_degenerate", False)),
        "degeneracy_reason": agg.get("degeneracy_reason"),
        "evidence_direction": (
            "supports" if res["overall_pass"] else "non_contributory"),
        "evidence_direction_per_claim": {
            "MECH-163": ("supports" if res["reading"] == "hierarchical_supported"
                         else "mixed" if res["reading"].startswith("both")
                         else "non_contributory"),
            "Q-085": ("supports" if res["criteria"][0]["passed"]
                      else "non_contributory"),
        },
        "interpretation": {
            "label": res["reading"],
            "preconditions": agg.get("adjudication_preconditions", []),
            "preconditions_scope_note": (
                (agg.get("per_arm_gate") or {}).get("preconditions_scope_note")),
            "criteria": res["criteria"],
            # arm -> the criterion names that arm owns. A criterion belonging to a
            # RED arm reads non_degenerate=False, which is what keeps a green arm's
            # result separable at adjudication instead of needing an autopsy.
            "criteria_non_degenerate": arm_criteria_non_degenerate(
                {A_HIER_S5: ["C1_dose_response_monotone_in_realised_length",
                             "C2_hierarchical_persistence_without_arbitrator"],
                 A_FLAT: ["C3_flat_arm_scorable"]},
                res["agg"],
                # C1 is a 3-point dose statistic: it is degenerate if the axis is
                # inert however green the arm is (gate (b)).
                extra={"C1_dose_response_monotone_in_realised_length":
                       res["dose_axis_live"]}),
        },
        "offending_cells_by_arm": res["offending_cells"],
        "confound_blocks": {
            "interruption": {
                "registered_in": "claims.yaml Q-085",
                "direction": "depresses persistence at LONGER realised lengths",
                "reported_as": "persistence_uninterrupted",
            },
            "devaluation_knowability": {
                "registered_in": ("2026-07-29 Ostlund/Winterbauer/Balleine 2009 "
                                  "lit entry, targeted_review_q_085"),
                "direction": ("suppresses COMMITMENT itself when the value change "
                              "is knowable at chunk onset -> flat dose-response "
                              "while the hypothesis is TRUE (false RESOLVED-NO)"),
                "reported_as": "persistence_pre_knowable",
                "note": ("Opposite in mechanism to the interruption confound; "
                         "reporting only one cannot disambiguate them."),
            },
        },
    }

    full_config: Dict[str, Any] = {
        "env": dict(base.ENV_KWARGS),
        "schedule": {"n_episodes_train": n_train, "steps_per_episode":
                     STEPS_PER_EPISODE, "n_episodes_probe": n_probe,
                     "deval_step": DEVAL_STEP},
        "arms": ARMS,
        "arm_config_slices": {a: _config_slice(a) for a in ARMS},
        "thresholds": {
            "MIN_COMMITTED_CHUNKS": MIN_COMMITTED_CHUNKS,
            "MIN_REALISED_LENGTH_SPREAD": MIN_REALISED_LENGTH_SPREAD,
            "MAX_POST_DEVAL_DELIVERED_BENEFIT": MAX_POST_DEVAL_DELIVERED_BENEFIT,
            "MIN_PRE_DEVAL_DELIVERED_BENEFIT": MIN_PRE_DEVAL_DELIVERED_BENEFIT,
            "MIN_POST_DEVAL_STEPS": MIN_POST_DEVAL_STEPS,
            "MIN_PRE_DEVAL_APPROACH_RATE": MIN_PRE_DEVAL_APPROACH_RATE,
            "MONOTONIC_RHO_FLOOR": MONOTONIC_RHO_FLOOR,
            "MONOTONIC_ABS_MARGIN": MONOTONIC_ABS_MARGIN,
            "SEED_PASS_FRACTION": SEED_PASS_FRACTION,
        },
    }

    out_path = write_flat_manifest(
        manifest,
        dry_run=bool(args.dry_run),
        config=full_config,
        seeds=seeds,
        script_path=Path(__file__),
        started_at=t0,
        z_goal_stream_stats=_ZG.stats(),
    )

    pk = res["dose_stats"]["pre_knowable"]
    print(f"gate green arms: {agg.get('green_arms')} red: {agg.get('red_arms')}",
          flush=True)
    print(f"dose_axis_live={res['dose_axis_live']} "
          f"realised_lengths={[round(v, 3) for v in pk['realised_lengths']]}",
          flush=True)
    print(f"persistence pre_knowable={[round(v, 4) for v in pk['persistence']]} "
          f"rho={pk['spearman_rho']:.4f} margin={pk['abs_margin']:.4f}", flush=True)
    for c in res["criteria"]:
        print(f"  {c['name']}: passed={c['passed']} "
              f"load_bearing={c.get('load_bearing', False)}", flush=True)
    print(f"outcome: {outcome}  reading: {res['reading']}", flush=True)
    print(f"manifest: {out_path}", flush=True)
    return outcome, out_path, bool(args.dry_run)


if __name__ == "__main__":
    _outcome, _out_path, _dry_run = main()
    _outcome_raw = str(_outcome).upper()
    emit_outcome(
        outcome=_outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL",
        manifest_path=str(_out_path),
        dry_run=_dry_run,
    )
