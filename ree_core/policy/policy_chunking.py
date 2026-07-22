"""ARC-071 policy_composition_via_repeated_grounding -- striatal-style chunking.

WHAT THIS IS
    ARC-071 is the TRANSITION mechanism (planned -> habitual) that MECH-163
    dual_goal_directed_systems presupposes but never specifies. MECH-163 names
    the PRESENCE of a habit system and a planned system; ARC-071 is the
    machinery that pumps content from the planned system into the habit system.
    Without it the two systems in MECH-163 are static configurations -- with it
    the division of labour between them is a continuous, experience-driven
    distribution. (Lit-pull targeted_review_arc_071_composition R3, conf 0.85:
    Yin & Knowlton 2006 DMS->DLS transfer with overtraining; Smith & Graybiel
    2013 IL-disruption chunk-level causal evidence.)

    ARC-071 is NOT MECH-477. MECH-477 is the ALLOCATION mechanism (which pathway
    holds control right now; fast, uncertainty-driven, arbitration-side).
    ARC-071 is the TRANSFER mechanism (how content becomes habitual; slow,
    repetition-and-outcome-consistency driven, execution-side). MECH-163
    presupposes BOTH and specifies NEITHER; they are separate builds.

TWO OPERATORS, NOT ONE SITE (lit-pull R2, conf 0.81)
    The substrate is phase-dependent multi-substrate, not a single locus.
    Mirroring Smith & Graybiel 2013's dual-operator view, this module builds:

    ChunkAccumulator  -- MECH-323 policy.composition.chunk_accumulator_formation
                         The striatum/DLS-analog FORMATION operator. Tallies
                         (sub-sequence -> outcome) pairs over executed actions
                         and mints a chunk when repetition, outcome-consistency
                         and evaluative conditions hold jointly.

    ChunkLibrary      -- MECH-324 policy.composition.chunk_maintenance
                         The infralimbic/vmPFC-analog MAINTENANCE operator.
                         Owns the four-state lifecycle, the crystallisation
                         counter, the hysteresis dissolution gate and the
                         MECH-322 replay-origin corroboration deadline. Smith &
                         Graybiel 2013 (lit_conf 0.86): IL disruption prevents
                         habit formation, so maintenance is causally required --
                         formation alone does not produce the behavioural
                         signature. This is why the two operators carry separate
                         switches: MECH-323-only is a real, runnable arm
                         (chunks form but never crystallise), and the
                         MECH-323-only vs MECH-323+324 contrast is the
                         registered discriminative test.

FORMATION TRIGGER (MECH-323, joint AND -- lit-pull R1, conf 0.78)
    (1) repetition count >= R_min over sliding window W
    (2) outcome variance < F_low                (formation half of hysteresis)
    (3) evaluative gate: outcome mean > baseline + margin (Graybiel 2008)

    Repetition + outcome consistency is PRIMARY (the canonical Graybiel 1998
    striatal-chunking pattern). Reward-rate (Sakai 2003) and V_s-positive
    (MECH-269) are secondary modulators, not primary triggers; free-energy
    minimisation is not supported by the chunking literature and is not used.

OPTIONS STRUCTURE (lit-pull R4, conf 0.72)
    A chunk is not merely a sequence. Sutton et al. 1999's options framework
    supplies the structural requirement that R4 surfaced: a macro that is safe
    to select atomically must carry an INITIATION SET (where it may start) and a
    TERMINATION CONDITION (when it stops), not just the action list. Both are
    fields on ChunkedPrimitive.

    Adopted as STRUCTURE ONLY. REE differs from the options framework on both
    ends: discovery here is Graybiel repetition-and-consistency rather than
    bottleneck-state or value-based option discovery, and the chunk stays
    VALUE-FLAT -- value_tag is provenance metadata, never a value head, so
    ARC-007 strict (value-flat hippocampal proposals) is preserved. Downstream
    E3 evaluation supplies value at selection time, as for any other proposal.

    Recursion (chunks-of-chunks) is permitted to depth 2-3; `depth` is a field
    on the chunk and `max_depth` caps it. Chunk size is budgeted at 2-5 elements
    per level (Sakai 2003).

HYSTERESIS (lit-pull R5, conf 0.71)
    Formation and dissolution use DIFFERENT thresholds, with the formation
    threshold BELOW the dissolution threshold (F_low < F_high), and dissolution
    runs on a slower timescale than formation. Biologically this is the R5
    verdict; as engineering it is also the standard defence against threshold
    chatter on a noisy running statistic. A single shared threshold would make
    chunks flicker in and out of the pool on estimator noise alone.

    The pre-existing primitive sequence is never erased: chunks are ADDITIVE.
    A dissolved chunk leaves its sub-elements individually selectable exactly as
    before, which is what lets ARC-070 decompose a chunk back under prediction
    failure.

MECH-094 -- SAFETY-CRITICAL (lit-pull R6, conf 0.74)
    A hallucinated chunk would be catastrophic: it would install a macro the
    agent never actually executed into the pool of things it can commit to
    atomically. The DEFAULT write path is therefore MECH-094-STRICT --
    record_step() refuses outright on hypothesis_tag=True and only increments a
    simulation-skip counter. Replayed, imagined and waking-DMN sequences cannot
    mint a chunk on this path at any parameter setting.

    Biology does not gate this cleanly (Albouy 2013: hippocampal-striatal sleep
    replay drives chunking-circuit consolidation), so REE's strict
    pre-registration is MORE CONSERVATIVE than biology. Rather than relax
    MECH-094 globally, the R6 escalation was resolved 2026-05-11 by MECH-322 as
    a narrow, separately-flagged, audit-trail-bearing SECOND write path:
    record_replay_sequence(). It requires ALL THREE of --
        (a) a value-tag from prior REAL executions at or above a high-positive
            threshold (default: top quartile of the real-execution outcome
            distribution) -- mirroring Albouy 2013's reward-prediction-biased
            coupling, in which biology selectively consolidates high-reward
            replays;
        (b) designated SD-017 SLEEP phase. Waking DMN -- where the MECH-292 /
            MECH-293 ghost-goal probes operate -- stays MECH-094-strict. The
            carve-out is sleep-only;
        (c) replay_origin=True on the formed chunk plus an ACCELERATED
            DISSOLUTION deadline: uncorroborated by real waking execution within
            N episodes and the chunk is retired directly to DISSOLVED, bypassing
            the slower DISSOLVING window. Chunks formed from replay must prove
            themselves in real execution or die.

    That path is OFF by default even when chunking itself is ON
    (use_chunk_replay_origin_path), so the shipped default is strict MECH-094.

INTEGRATION
    REEAgent records each executed action class (waking, hypothesis_tag=False)
    into the accumulator, and reports the episode/segment outcome. Crystallised
    chunks are optionally spliced into the hippocampal candidate pool as single
    Trajectory objects (use_chunk_proposal_injection, default OFF), where the
    MECH-090 beta-gate commit latch executes them as one move -- which is where
    the rollout-cost and behavioural-latency drop comes from.

    Default OFF and bit-identical when OFF: REEAgent leaves the attribute None.

See REE_assembly/docs/architecture/policy_primitive_granularity.md and
evidence/literature/targeted_review_arc_071_composition/synthesis.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple


class ChunkState(str, Enum):
    """MECH-324 four-state chunk lifecycle.

    FORMING      : just minted by MECH-323. Weak selection_weight; the
                   crystallisation counter accumulates on successful REAL
                   executions only (replay never increments it).
    CRYSTALLISED : counter reached C_min. Full selection_weight. Persists until
                   outcome variance exceeds F_high, or the MECH-322
                   corroboration deadline expires.
    DISSOLVING   : variance exceeded F_high. selection_weight decays linearly
                   over T_dissolve trials. RECOVERABLE -- if variance drops back
                   below F_high the chunk returns to CRYSTALLISED.
    DISSOLVED    : removed from the proposal pool, retained in the audit trail.

    MECH-322 replay-origin chunks that miss their corroboration deadline go
    CRYSTALLISED -> DISSOLVED directly, bypassing the slower DISSOLVING window.
    """

    FORMING = "forming"
    CRYSTALLISED = "crystallised"
    DISSOLVING = "dissolving"
    DISSOLVED = "dissolved"


@dataclass
class ChunkedPrimitive:
    """A composed policy primitive -- the ARC-071 output object.

    The Sutton 1999 options structure (R4) requires initiation_set and
    termination_condition as first-class fields: a macro selected atomically
    must declare where it may start and when it stops, not merely which actions
    it contains.

    Attributes:
        sequence : the composed action classes, in execution order.
        initiation_set : context-bucket keys in which this chunk may be
            proposed. Empty set = unrestricted (the permissive default used by
            the substrate-readiness path, where context bucketing is not yet
            wired).
        termination_condition : why the chunk stops. "sequence_complete" is the
            default (run to the end of `sequence`).
        value_tag : accumulated outcome mean over the real executions that
            formed this chunk. PROVENANCE METADATA ONLY -- never a value head;
            ARC-007 strict value-flat proposals are preserved.
        replay_origin : True only for chunks minted through the MECH-322
            sleep-replay carve-out. The audit flag.
        formation_timestamp : accumulator trial index at formation.
        depth : recursion level. 1 = composed of raw actions; 2+ = composed of
            chunks (chunks-of-chunks), capped by max_depth.
        state : MECH-324 lifecycle state.
        selection_weight : proposal-pool weight in [0, 1]. 0 while DISSOLVED.
        crystallisation_counter : corroborating REAL executions since formation.
        episodes_since_corroboration : MECH-322 deadline counter; reset to 0 on
            each corroborating real execution.
        dissolving_trials : trials spent in DISSOLVING (drives the linear decay).
    """

    sequence: Tuple[int, ...]
    initiation_set: frozenset = frozenset()
    termination_condition: str = "sequence_complete"
    value_tag: float = 0.0
    replay_origin: bool = False
    formation_timestamp: int = 0
    depth: int = 1
    state: ChunkState = ChunkState.FORMING
    selection_weight: float = 0.0
    crystallisation_counter: int = 0
    episodes_since_corroboration: int = 0
    dissolving_trials: int = 0

    @property
    def key(self) -> Tuple[int, ...]:
        """Identity of the chunk = its action sequence."""
        return tuple(self.sequence)

    @property
    def is_selectable(self) -> bool:
        """True iff this chunk may be spliced into the proposal pool."""
        return (
            self.state is ChunkState.CRYSTALLISED
            or (self.state is ChunkState.DISSOLVING and self.selection_weight > 0.0)
        )

    def as_dict(self) -> dict:
        """Audit-trail snapshot (ASCII-safe, JSON-serialisable)."""
        return {
            "sequence": list(self.sequence),
            "initiation_set": sorted(self.initiation_set),
            "termination_condition": self.termination_condition,
            "value_tag": float(self.value_tag),
            "replay_origin": bool(self.replay_origin),
            "formation_timestamp": int(self.formation_timestamp),
            "depth": int(self.depth),
            "state": self.state.value,
            "selection_weight": float(self.selection_weight),
            "crystallisation_counter": int(self.crystallisation_counter),
            "episodes_since_corroboration": int(self.episodes_since_corroboration),
        }


@dataclass
class PolicyChunkingConfig:
    """Configuration for the ARC-071 chunking operators.

    Defaults are the MECH-323 / MECH-324 registered suggested defaults; the
    child-MECH validation experiments refine them.

    Attributes:
        use_policy_chunking : master switch. False = disabled (default,
            backward-compatible). REEAgent does not instantiate the operators
            when False.
        min_repetitions : R_min. Repetitions of a sub-sequence required within
            the window before it may form.
        window_trials : W. Sliding-window length over which repetitions and
            outcome variance are measured.
        variance_low : F_low. Outcome variance must be BELOW this to form.
        variance_high : F_high. Outcome variance above this starts dissolution.
            Must exceed variance_low -- that gap IS the R5 hysteresis.
        evaluative_margin : the accumulated outcome mean must exceed
            (running baseline + this) to form (Graybiel 2008 evaluative gate).
        min_chunk_size / max_chunk_size : chunk-size budget per level
            (Sakai 2003, 2-5 elements).
        max_depth : chunks-of-chunks recursion cap (R4: 2-3 levels).
        max_library_size : hard cap on retained chunks. Bounds memory; the
            lowest-value DISSOLVED chunks are evicted first.
        max_tracked_sequences : hard cap on the candidate tally table. Bounds
            the combinatorial sub-sequence enumeration.
        use_chunk_maintenance : MECH-324 sub-switch. False = chunks form but
            never crystallise (the registered ARM_1 dissociation arm).
        crystallisation_min : C_min corroborating real executions to crystallise.
        dissolve_trials : T_dissolve. Linear selection_weight decay window.
        use_chunk_replay_origin_path : MECH-322 carve-out switch.
            SAFETY-CRITICAL -- False by default EVEN WHEN chunking is on. While
            False, record_replay_sequence() is inert and no chunk can originate
            from replayed or imagined content.
        replay_value_quantile : the high-positive value-tag threshold for the
            carve-out, as a quantile of the real-execution outcome distribution.
        replay_corroboration_episodes : N. Waking episodes a replay-origin chunk
            has to earn corroboration before accelerated dissolution.
    """

    use_policy_chunking: bool = False
    min_repetitions: int = 20
    window_trials: int = 100
    variance_low: float = 0.15
    variance_high: float = 0.45
    evaluative_margin: float = 0.05
    min_chunk_size: int = 2
    max_chunk_size: int = 5
    max_depth: int = 3
    max_library_size: int = 64
    max_tracked_sequences: int = 512
    use_chunk_maintenance: bool = False
    crystallisation_min: int = 5
    dissolve_trials: int = 50
    use_chunk_replay_origin_path: bool = False
    replay_value_quantile: float = 0.75
    replay_corroboration_episodes: int = 75

    def validate(self) -> None:
        """Raise ValueError on a configuration that cannot behave as specified."""
        if self.min_repetitions < 1:
            raise ValueError("min_repetitions must be >= 1")
        if self.window_trials < self.min_repetitions:
            raise ValueError(
                "window_trials must be >= min_repetitions "
                "(a sub-sequence cannot repeat more often than the window is long)"
            )
        if not (0.0 <= self.variance_low < self.variance_high):
            raise ValueError(
                "require 0 <= variance_low < variance_high (R5 hysteresis: the "
                "formation threshold must sit BELOW the dissolution threshold)"
            )
        if self.min_chunk_size < 2:
            raise ValueError("min_chunk_size must be >= 2 (a chunk composes >= 2 elements)")
        if self.max_chunk_size < self.min_chunk_size:
            raise ValueError("max_chunk_size must be >= min_chunk_size")
        if self.max_depth < 1:
            raise ValueError("max_depth must be >= 1")
        if self.crystallisation_min < 1:
            raise ValueError("crystallisation_min must be >= 1")
        if self.dissolve_trials < 1:
            raise ValueError("dissolve_trials must be >= 1")
        if not (0.0 < self.replay_value_quantile < 1.0):
            raise ValueError("replay_value_quantile must be in (0, 1)")
        if self.replay_corroboration_episodes < 1:
            raise ValueError("replay_corroboration_episodes must be >= 1")


def _mean(values: Sequence[float]) -> float:
    """Arithmetic mean; 0.0 on empty."""
    if not values:
        return 0.0
    return float(sum(values)) / float(len(values))


def _variance(values: Sequence[float]) -> float:
    """Population variance via a two-pass mean.

    Two-pass rather than the sum-of-squares shortcut: the shortcut suffers
    catastrophic cancellation when the outcome mean is large relative to the
    spread, which is exactly the regime a consistent-outcome sub-sequence sits
    in -- and an under-estimated variance there would mint chunks that the
    formation gate should have refused.
    """
    n = len(values)
    if n < 2:
        return 0.0
    mu = _mean(values)
    return float(sum((float(v) - mu) ** 2 for v in values)) / float(n)


class ChunkAccumulator:
    """MECH-323 -- the DLS-analog chunk FORMATION operator.

    Pure-arithmetic, no learned parameters, no nn.Module inheritance. Watches
    the stream of executed action classes, tallies contiguous sub-sequences of
    length [min_chunk_size, max_chunk_size] against the outcomes that followed
    them, and mints a ChunkedPrimitive when the joint formation condition holds.

    MECH-094: record_step() REFUSES on hypothesis_tag=True. The only path that
    accepts internally-generated content is record_replay_sequence(), the
    MECH-322 carve-out, which is separately flagged and off by default.

    Bounded by construction: only contiguous sub-sequences of length 2..5 within
    the current episode buffer are tallied, and the tally table is FIFO-capped
    at max_tracked_sequences. Without those bounds the sub-sequence enumeration
    grows combinatorially with episode length.

    Diagnostics tracked:
        _n_steps_recorded        : int   (real executed steps seen)
        _n_outcomes              : int   (trials / outcome reports)
        _n_chunks_formed         : int   (MECH-094-strict path)
        _n_replay_chunks_formed  : int   (MECH-322 carve-out path)
        _n_simulation_skips      : int   (record_step refusals -- the MECH-094
                                          readout; must stay > 0 and formation
                                          from replay must stay 0)
        _n_replay_refusals       : int   (carve-out condition failures)
        _last_formed_sequence    : tuple
    """

    def __init__(self, config: Optional[PolicyChunkingConfig] = None) -> None:
        self.config = config if config is not None else PolicyChunkingConfig()
        self.config.validate()

        # Current episode's executed action classes (bounded by max_chunk_size
        # lookback plus the window; we only need recent contiguous runs).
        self._episode_actions: List[int] = []
        # sub-sequence -> outcomes observed after it, FIFO-bounded per entry.
        self._tally: Dict[Tuple[int, ...], List[float]] = {}
        # Rolling real-execution outcome distribution, for the evaluative
        # baseline and the MECH-322 value quantile.
        self._outcome_history: List[float] = []
        self._trial_index: int = 0

        self._n_steps_recorded: int = 0
        self._n_outcomes: int = 0
        self._n_chunks_formed: int = 0
        self._n_replay_chunks_formed: int = 0
        self._n_simulation_skips: int = 0
        self._n_replay_refusals: int = 0
        self._last_formed_sequence: Tuple[int, ...] = ()

    # ------------------------------------------------------------------
    # Forward path -- MECH-094-strict (real execution only)
    # ------------------------------------------------------------------
    def record_step(self, action_class: int, hypothesis_tag: bool = False) -> bool:
        """Record one EXECUTED action class. Returns True iff it was recorded.

        Args:
            action_class : the committed action class this step (the int the
                agent actually executed).
            hypothesis_tag : MECH-094 gate. True = internally generated
                (replay / simulation / waking DMN). SAFETY-CRITICAL: such a step
                is REFUSED outright -- it cannot contribute to chunk formation
                at any parameter setting. Only the simulation-skip counter
                advances.

        Returns:
            True iff the step entered the accumulator.
        """
        if hypothesis_tag:
            self._n_simulation_skips += 1
            return False

        self._episode_actions.append(int(action_class))
        self._n_steps_recorded += 1
        # Bound the buffer: only the most recent max_chunk_size actions can
        # start a new contiguous sub-sequence.
        limit = max(self.config.max_chunk_size * 4, 32)
        if len(self._episode_actions) > limit:
            del self._episode_actions[:-limit]
        return True

    def note_outcome(self, outcome_signal: float) -> None:
        """Report the outcome of the recently executed sub-sequence(s).

        Credits every contiguous sub-sequence of permitted length that ends at
        the current position with this outcome, then advances the trial index.
        Called at an episode or segment boundary by the agent.

        Args:
            outcome_signal : scalar outcome quality. Higher = better. The
                evaluative gate compares its running mean against the running
                baseline of all outcomes.
        """
        c = self.config
        outcome = float(outcome_signal)
        self._trial_index += 1
        self._n_outcomes += 1

        self._outcome_history.append(outcome)
        if len(self._outcome_history) > c.window_trials:
            del self._outcome_history[: -c.window_trials]

        actions = self._episode_actions
        for size in range(c.min_chunk_size, c.max_chunk_size + 1):
            if len(actions) < size:
                break
            key = tuple(actions[-size:])
            bucket = self._tally.get(key)
            if bucket is None:
                if len(self._tally) >= c.max_tracked_sequences:
                    # FIFO-evict the oldest tracked sequence; bounds the table.
                    oldest = next(iter(self._tally))
                    del self._tally[oldest]
                bucket = []
                self._tally[key] = bucket
            bucket.append(outcome)
            if len(bucket) > c.window_trials:
                del bucket[: -c.window_trials]

    def formation_candidates(self) -> List[Tuple[Tuple[int, ...], float, float]]:
        """Sub-sequences meeting the joint MECH-323 formation condition.

        Returns a list of (sequence, outcome_mean, outcome_variance) for every
        tracked sub-sequence satisfying ALL THREE conditions:
            (1) repetitions >= min_repetitions within the window
            (2) outcome variance < variance_low
            (3) outcome mean > running baseline + evaluative_margin
        """
        c = self.config
        baseline = _mean(self._outcome_history)
        out: List[Tuple[Tuple[int, ...], float, float]] = []
        for key, outcomes in self._tally.items():
            if len(outcomes) < c.min_repetitions:
                continue
            var = _variance(outcomes)
            if var >= c.variance_low:
                continue
            mu = _mean(outcomes)
            if mu <= baseline + c.evaluative_margin:
                continue
            out.append((key, mu, var))
        return out

    def mint(
        self,
        sequence: Tuple[int, ...],
        value_tag: float,
        depth: int = 1,
        replay_origin: bool = False,
        initiation_set: Optional[frozenset] = None,
    ) -> ChunkedPrimitive:
        """Construct a ChunkedPrimitive. Does not itself register it."""
        chunk = ChunkedPrimitive(
            sequence=tuple(sequence),
            initiation_set=initiation_set if initiation_set is not None else frozenset(),
            termination_condition="sequence_complete",
            value_tag=float(value_tag),
            replay_origin=bool(replay_origin),
            formation_timestamp=int(self._trial_index),
            depth=int(depth),
            state=ChunkState.FORMING,
            selection_weight=0.0,
        )
        if replay_origin:
            self._n_replay_chunks_formed += 1
        else:
            self._n_chunks_formed += 1
        self._last_formed_sequence = chunk.key
        return chunk

    # ------------------------------------------------------------------
    # Forward path -- MECH-322 sleep-replay carve-out (SAFETY-CRITICAL)
    # ------------------------------------------------------------------
    def replay_value_threshold(self) -> float:
        """The high-positive value bar for the MECH-322 carve-out.

        Computed as the replay_value_quantile of the REAL-execution outcome
        distribution -- so the bar is set by what the agent actually achieved
        while awake, never by replayed content.
        """
        hist = sorted(self._outcome_history)
        if not hist:
            # No real-execution history: nothing can clear the bar. Fails CLOSED.
            return float("inf")
        idx = int(self.config.replay_value_quantile * (len(hist) - 1))
        return float(hist[idx])

    def record_replay_sequence(
        self,
        sequence: Sequence[int],
        value_tag: float,
        in_sleep_phase: bool,
        hypothesis_tag: bool = True,
    ) -> Optional[ChunkedPrimitive]:
        """MECH-322 carve-out: mint a chunk from a REPLAYED sequence.

        The single sanctioned exception to MECH-094 strict gating, and the only
        method in this module that accepts hypothesis_tag=True. Every condition
        is ANDed and every one fails CLOSED.

        Args:
            sequence : the replayed action-class sequence.
            value_tag : the value carried from PRIOR REAL executions of this
                sequence. Not a value computed from the replay itself.
            in_sleep_phase : True only in a designated SD-017 sleep phase
                (SWS_ANALOG / REM_ANALOG). Waking DMN -- where MECH-292 /
                MECH-293 ghost-goal probes operate -- must pass False, and is
                thereby refused.
            hypothesis_tag : provenance of the sequence. Retained in the
                signature so the caller's provenance is explicit at the call
                site rather than implied.

        Returns:
            The minted chunk (replay_origin=True), or None if any condition
            fails.
        """
        c = self.config
        if not c.use_chunk_replay_origin_path:
            self._n_replay_refusals += 1
            return None
        if not in_sleep_phase:
            # (b) sleep-phase requirement. Waking DMN stays MECH-094-strict.
            self._n_replay_refusals += 1
            return None
        seq = tuple(int(a) for a in sequence)
        if not (c.min_chunk_size <= len(seq) <= c.max_chunk_size):
            self._n_replay_refusals += 1
            return None
        # (a) value-tag requirement, measured against REAL-execution history.
        if float(value_tag) < self.replay_value_threshold():
            self._n_replay_refusals += 1
            return None
        # (c) audit flag + accelerated dissolution deadline are carried by the
        # replay_origin field; ChunkLibrary enforces the deadline.
        return self.mint(seq, value_tag=float(value_tag), depth=1, replay_origin=True)

    # ------------------------------------------------------------------
    def end_episode(self) -> None:
        """Clear the within-episode action buffer, keeping cross-episode tallies."""
        self._episode_actions.clear()

    def reset(self) -> None:
        """Reset per-episode state and diagnostic counters.

        Note the asymmetry with end_episode(): reset() drops the accumulated
        cross-trial tallies too. Chunk formation is a SLOW, cross-episode
        process (R_min = 20 repetitions over a window of 100 trials), so an
        agent whose accumulator is reset every episode can never form anything.
        The agent calls end_episode() per episode and reset() only on a genuine
        accumulator reset.
        """
        self._episode_actions.clear()
        self._tally.clear()
        self._outcome_history.clear()
        self._trial_index = 0
        self._n_steps_recorded = 0
        self._n_outcomes = 0
        self._n_chunks_formed = 0
        self._n_replay_chunks_formed = 0
        self._n_simulation_skips = 0
        self._n_replay_refusals = 0
        self._last_formed_sequence = ()

    def get_state(self) -> dict:
        """Diagnostic snapshot for experiment manifests."""
        return {
            "chunk_acc_n_steps": self._n_steps_recorded,
            "chunk_acc_n_outcomes": self._n_outcomes,
            "chunk_acc_n_tracked_sequences": len(self._tally),
            # The substrate-readiness readout: did the accumulator fire at all.
            "chunk_acc_n_formed": self._n_chunks_formed,
            # MECH-322 audit: must stay 0 unless the carve-out is deliberately on.
            "chunk_acc_n_replay_formed": self._n_replay_chunks_formed,
            # MECH-094 audit: refusals of internally-generated steps.
            "chunk_acc_n_simulation_skips": self._n_simulation_skips,
            "chunk_acc_n_replay_refusals": self._n_replay_refusals,
            "chunk_acc_last_formed": list(self._last_formed_sequence),
        }


class ChunkLibrary:
    """MECH-324 -- the IL/vmPFC-analog chunk MAINTENANCE operator.

    Owns the four-state lifecycle, the crystallisation counter, the hysteresis
    dissolution gate and the MECH-322 corroboration deadline. Formation alone
    does not produce the behavioural signature: Smith & Graybiel 2013's IL
    disruption prevents habit formation with chunk formation otherwise intact,
    so with use_chunk_maintenance=False chunks form and stay FORMING forever --
    the registered ARM_1 arm, whose contrast against ARM_2 isolates this
    operator's contribution.

    Diagnostics tracked:
        _n_registered / _n_crystallised / _n_dissolved
        _n_replay_deadline_dissolutions : MECH-322 accelerated retirements
    """

    def __init__(self, config: Optional[PolicyChunkingConfig] = None) -> None:
        self.config = config if config is not None else PolicyChunkingConfig()
        self.config.validate()
        self._chunks: Dict[Tuple[int, ...], ChunkedPrimitive] = {}
        self._n_registered: int = 0
        self._n_crystallised: int = 0
        self._n_dissolved: int = 0
        self._n_replay_deadline_dissolutions: int = 0

    # ------------------------------------------------------------------
    def register(self, chunk: ChunkedPrimitive) -> bool:
        """Add a newly formed chunk. Returns False if already present or full."""
        if chunk.key in self._chunks:
            return False
        if len(self._chunks) >= self.config.max_library_size:
            if not self._evict_one():
                return False
        self._chunks[chunk.key] = chunk
        self._n_registered += 1
        return True

    def _evict_one(self) -> bool:
        """Evict the least valuable DISSOLVED chunk. Returns True on success.

        Only DISSOLVED chunks are evictable -- a live chunk is never silently
        dropped to make room, because that would look identical to dissolution
        in the diagnostics while having none of its meaning.
        """
        dissolved = [k for k, c in self._chunks.items() if c.state is ChunkState.DISSOLVED]
        if not dissolved:
            return False
        victim = min(dissolved, key=lambda k: self._chunks[k].value_tag)
        del self._chunks[victim]
        return True

    def get(self, sequence: Sequence[int]) -> Optional[ChunkedPrimitive]:
        """Look up a chunk by its action sequence."""
        return self._chunks.get(tuple(int(a) for a in sequence))

    def selectable_chunks(self) -> List[ChunkedPrimitive]:
        """Chunks eligible for the proposal pool, strongest weight first."""
        out = [c for c in self._chunks.values() if c.is_selectable]
        out.sort(key=lambda c: c.selection_weight, reverse=True)
        return out

    def all_chunks(self) -> List[ChunkedPrimitive]:
        """Every retained chunk, including DISSOLVED (the audit trail)."""
        return list(self._chunks.values())

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------
    def note_real_execution(
        self, sequence: Sequence[int], outcome_variance: float
    ) -> Optional[ChunkState]:
        """Credit a REAL waking execution of a chunk; advance its lifecycle.

        This is sub-mechanism (A) -- the crystallisation counter -- plus the
        hysteresis gate (B) and the corroboration reset for (C). Replayed
        executions must NOT be routed here: only real waking execution
        crystallises a chunk or clears a MECH-322 deadline.

        Args:
            sequence : the executed chunk's action sequence.
            outcome_variance : current windowed outcome variance for it.

        Returns:
            The chunk's new state, or None if the sequence is not a chunk.
        """
        chunk = self.get(sequence)
        if chunk is None:
            return None
        if not self.config.use_chunk_maintenance:
            # MECH-324 disabled: formation-only arm. Chunks stay FORMING and
            # never become selectable -- the ARM_1 dissociation.
            return chunk.state

        c = self.config
        var = float(outcome_variance)
        # (C) corroboration: any real execution clears the replay deadline.
        chunk.episodes_since_corroboration = 0

        if chunk.state is ChunkState.FORMING:
            chunk.crystallisation_counter += 1
            if chunk.crystallisation_counter >= c.crystallisation_min:
                chunk.state = ChunkState.CRYSTALLISED
                chunk.selection_weight = 1.0
                self._n_crystallised += 1
        elif chunk.state is ChunkState.CRYSTALLISED:
            chunk.crystallisation_counter += 1
            # (B) hysteresis: dissolution uses F_high, ABOVE the F_low that
            # formed it, so a chunk does not flicker on estimator noise.
            if var > c.variance_high:
                chunk.state = ChunkState.DISSOLVING
                chunk.dissolving_trials = 0
        elif chunk.state is ChunkState.DISSOLVING:
            if var <= c.variance_high:
                # Recovery: variance fell back inside the band.
                chunk.state = ChunkState.CRYSTALLISED
                chunk.selection_weight = 1.0
                chunk.dissolving_trials = 0
        return chunk.state

    def tick_maintenance(self, variances: Optional[Dict[Tuple[int, ...], float]] = None) -> None:
        """Advance dissolution timers one trial for every chunk.

        Sub-mechanism (B) slow-timescale half: DISSOLVING chunks decay their
        selection_weight linearly over T_dissolve trials and are then removed
        from the pool. Dissolution is deliberately SLOWER than formation (R5).

        Args:
            variances : optional current windowed variance per chunk sequence,
                used to start dissolution on chunks that are not being executed
                (a chunk whose outcome has gone inconsistent may simply stop
                being selected, so it would otherwise never be re-evaluated).
        """
        if not self.config.use_chunk_maintenance:
            return
        c = self.config
        for chunk in self._chunks.values():
            if chunk.state is ChunkState.CRYSTALLISED and variances is not None:
                var = variances.get(chunk.key)
                if var is not None and float(var) > c.variance_high:
                    chunk.state = ChunkState.DISSOLVING
                    chunk.dissolving_trials = 0
            if chunk.state is ChunkState.DISSOLVING:
                chunk.dissolving_trials += 1
                frac = 1.0 - (float(chunk.dissolving_trials) / float(c.dissolve_trials))
                chunk.selection_weight = max(0.0, min(1.0, frac))
                if chunk.dissolving_trials >= c.dissolve_trials:
                    chunk.state = ChunkState.DISSOLVED
                    chunk.selection_weight = 0.0
                    self._n_dissolved += 1

    def note_episode_end(self) -> int:
        """Advance the MECH-322 corroboration deadline; retire the expired.

        Sub-mechanism (C). A replay-origin chunk that has gone
        replay_corroboration_episodes waking episodes without a real execution
        is retired DIRECTLY to DISSOLVED, bypassing the slower DISSOLVING
        window. Chunks formed from replay must prove themselves in real
        execution or be removed.

        Returns:
            The number of chunks retired by the deadline this episode.
        """
        if not self.config.use_chunk_maintenance:
            return 0
        c = self.config
        retired = 0
        for chunk in self._chunks.values():
            if not chunk.replay_origin:
                continue
            if chunk.state in (ChunkState.DISSOLVED,):
                continue
            chunk.episodes_since_corroboration += 1
            if chunk.episodes_since_corroboration >= c.replay_corroboration_episodes:
                chunk.state = ChunkState.DISSOLVED
                chunk.selection_weight = 0.0
                self._n_dissolved += 1
                self._n_replay_deadline_dissolutions += 1
                retired += 1
        return retired

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Reset the library and its diagnostic counters."""
        self._chunks.clear()
        self._n_registered = 0
        self._n_crystallised = 0
        self._n_dissolved = 0
        self._n_replay_deadline_dissolutions = 0

    def get_state(self) -> dict:
        """Diagnostic snapshot for experiment manifests."""
        by_state = {s.value: 0 for s in ChunkState}
        for chunk in self._chunks.values():
            by_state[chunk.state.value] += 1
        return {
            "chunk_lib_size": len(self._chunks),
            "chunk_lib_n_registered": self._n_registered,
            # MECH-324's readout: formation alone leaves this at 0.
            "chunk_lib_n_crystallised": self._n_crystallised,
            "chunk_lib_n_dissolved": self._n_dissolved,
            "chunk_lib_n_replay_deadline_dissolutions": (
                self._n_replay_deadline_dissolutions
            ),
            "chunk_lib_n_selectable": len(self.selectable_chunks()),
            "chunk_lib_by_state": by_state,
            "chunk_lib_n_replay_origin": sum(
                1 for c in self._chunks.values() if c.replay_origin
            ),
        }


class PolicyChunking:
    """ARC-071 facade -- owns the MECH-323 accumulator and MECH-324 library.

    The single object REEAgent holds (None when the master switch is off). It
    exists so the agent has one attribute and one per-step call rather than two
    operators to keep in step, and so the formation -> registration handoff
    happens in exactly one place.
    """

    def __init__(self, config: Optional[PolicyChunkingConfig] = None) -> None:
        self.config = config if config is not None else PolicyChunkingConfig()
        self.config.validate()
        self.accumulator = ChunkAccumulator(self.config)
        self.library = ChunkLibrary(self.config)
        self._n_formation_passes: int = 0

    def record_step(self, action_class: int, hypothesis_tag: bool = False) -> bool:
        """Record one executed action class (MECH-094-strict). See ChunkAccumulator."""
        return self.accumulator.record_step(action_class, hypothesis_tag=hypothesis_tag)

    def note_outcome(self, outcome_signal: float) -> List[ChunkedPrimitive]:
        """Report an outcome, run the formation pass, register new chunks.

        Returns the chunks minted by this pass (empty list is the normal case).
        """
        self.accumulator.note_outcome(outcome_signal)
        self._n_formation_passes += 1

        formed: List[ChunkedPrimitive] = []
        for seq, mu, var in self.accumulator.formation_candidates():
            if self.library.get(seq) is not None:
                continue
            depth = self._depth_for(seq)
            if depth > self.config.max_depth:
                continue
            chunk = self.accumulator.mint(seq, value_tag=mu, depth=depth)
            if self.library.register(chunk):
                formed.append(chunk)

        # Maintain the executed chunk (crystallisation + hysteresis) and advance
        # the slow dissolution timers.
        variances = {
            key: _variance(outcomes) for key, outcomes in self.accumulator._tally.items()
        }
        for chunk in self.library.all_chunks():
            var = variances.get(chunk.key)
            if var is not None and self._was_executed(chunk):
                self.library.note_real_execution(chunk.key, var)
        self.library.tick_maintenance(variances)
        return formed

    def _was_executed(self, chunk: ChunkedPrimitive) -> bool:
        """True iff the chunk's sequence ends the current episode action buffer."""
        actions = self.accumulator._episode_actions
        n = len(chunk.sequence)
        return len(actions) >= n and tuple(actions[-n:]) == chunk.key

    def _depth_for(self, sequence: Sequence[int]) -> int:
        """Recursion depth of a candidate: 1 + the deepest chunk it contains.

        Implements the R4 chunks-of-chunks cap. A sub-sequence that already
        contains a registered chunk composes at one level above it.
        """
        deepest = 0
        for chunk in self.library.all_chunks():
            if chunk.state is ChunkState.DISSOLVED:
                continue
            seq = tuple(int(a) for a in sequence)
            n = len(chunk.sequence)
            if n <= len(seq) and any(
                seq[i : i + n] == chunk.key for i in range(len(seq) - n + 1)
            ):
                deepest = max(deepest, chunk.depth)
        return deepest + 1

    def note_replay_sequence(
        self, sequence: Sequence[int], value_tag: float, in_sleep_phase: bool
    ) -> Optional[ChunkedPrimitive]:
        """MECH-322 carve-out entry point. See ChunkAccumulator.record_replay_sequence."""
        chunk = self.accumulator.record_replay_sequence(
            sequence, value_tag=value_tag, in_sleep_phase=in_sleep_phase, hypothesis_tag=True
        )
        if chunk is not None and self.library.register(chunk):
            return chunk
        return None

    def selectable_chunks(self) -> List[ChunkedPrimitive]:
        """Crystallised chunks eligible for proposal-pool injection."""
        return self.library.selectable_chunks()

    def end_episode(self) -> None:
        """Per-episode boundary: clear the action buffer, advance MECH-322 deadlines."""
        self.accumulator.end_episode()
        self.library.note_episode_end()

    def reset(self) -> None:
        """Full reset of both operators and all diagnostic counters."""
        self.accumulator.reset()
        self.library.reset()
        self._n_formation_passes = 0

    def get_state(self) -> dict:
        """Combined diagnostic snapshot for experiment manifests."""
        state = {}
        state.update(self.accumulator.get_state())
        state.update(self.library.get_state())
        state["chunk_n_formation_passes"] = self._n_formation_passes
        return state
