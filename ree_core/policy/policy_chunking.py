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
    on the chunk and `max_depth` caps it -- also an INITIAL budget rather than a
    lifetime cap, see the growable-depth section below. Chunk size STARTS
    budgeted at 2-5 elements per level (Sakai 2003) -- see the growable-ceiling
    section below for why that is an initial budget and not a lifetime cap.

CHUNK SIZE IS A GROWABLE CEILING, NOT A CONSTANT (lit-pull 2026-07-27)
    Ramkumar, Acuna, Berniker, Grafton, Turner & Kording 2016 (Nat Commun,
    10.1038/ncomms12176) shows chunking is the OUTPUT of an efficiency/
    computation trade-off rather than a fixed capacity. Planning complexity
    grows with the horizon a movement is optimised over; chunking buys
    tractability by planning over shorter horizons, at a cost in achievable
    efficiency. The prediction is a SCHEDULE, not a constant: start with many
    short chunks because compute binds, then MERGE them as practice lowers the
    cost of more complex computation. Two macaques learning a ten-element
    reaching sequence over months do exactly this -- "as chunks become longer
    over the course of learning, movements are optimized over increasingly
    longer horizons".

    Two things a fixed 2-5 cannot express:
      (i)  a hard lifetime cap at 5 FORBIDS the most robust longitudinal
           signature in this literature, so an REE agent could not reproduce the
           observed learning trajectory even with every other trigger condition
           correct;
      (ii) two agents with different rollout budgets should settle at DIFFERENT
           chunk sizes -- so a single global maximum is the wrong SHAPE of
           parameter, not merely the wrong value.

    Bo, Borza & Seidler 2009 (J Neurophysiol, 10.1152/jn.00393.2009) constrains
    it from the other side: chunk length tracks visuospatial working-memory
    capacity and DECLINES with age. Together the two papers say the budget
    should be neither fixed across agents nor fixed within an agent's lifetime.

    use_growable_chunk_ceiling (default OFF) implements this as:

      ceiling starts at max_chunk_size (5), grows one element at a time toward
      derived_chunk_ceiling = floor(deliberation_horizon * budget_fraction),
      each step licensed by a realised marginal return >= the returns threshold.

    THE NUMBER 5 IS NOT REPLACED. Ramkumar 2016 proposes no upper bound, so it
    cannot license any other constant, and none is asserted. The fraction is
    ANCHORED (0.1667 = 5/30) so that at REE's actual rollout horizon of 30 the
    derivation returns exactly 5 -- the inherited Sakai budget, unchanged, for
    an agent with today's deliberation budget. Only an agent with a LARGER
    budget derives a larger ceiling. What changed is the parameter's shape,
    from constant to function.

    Read the anchor as a calibration, not a coincidence: the paper fixes the
    SHAPE of the relationship and nothing about its scale, so the scale is
    pinned to the one quantity that does carry warrant (Sakai's 2-5, at the
    budget the agent actually has) rather than to a number invented here.

    THE BRAKE IS THE GROWTH RULE, not a separate cap. What bounds chunk growth
    empirically is DIMINISHING RETURNS: chunk structures stay near 50-60% of the
    complexity of executing the sequence as a single unit, and the monkeys never
    collapse the whole sequence into one chunk. An accumulator that grew
    monotonically with practice would eventually do exactly that, whatever
    number it stopped at. So growth is conditioned on each merge having actually
    paid -- see ChunkAccumulator.consider_ceiling_growth -- and the plateau
    falls out of the rule instead of being a second parameter.

    DECOUPLED FROM R_min BY CONSTRUCTION. Bo et al. found the capacity-to-chunk-
    LENGTH correlation in both age groups but the capacity-to-learning-RATE
    correlation only in the young, so chunk size and formation rate are
    SEPARABLE quantities. The obvious implementation of a growable ceiling --
    grow it as repetitions accumulate -- would silently re-couple them, tying
    size to exactly the rate quantity the dissociation separates it from. Growth
    here reads REALISED MARGINAL OUTCOME GAIN and the deliberation budget, never
    the repetition tally. min_repetitions appears in the returns test only as a
    judge-ability filter (is this sequence attested enough to be measured), not
    as the thing being measured. Do not "simplify" it into a practice counter.

    TRACTABILITY IS PRESERVED, and is worth stating precisely because the
    original 2-5 bound was justified by it. The enumeration in note_outcome is
    NOT combinatorial in the ceiling: it tallies only the SUFFIX of each
    permitted length (actions[-size:]), so the per-outcome cost is O(ceiling),
    linear. The combinatorial enumeration the bound guards against is the naive
    all-sub-sequences-at-all-positions one, which this module never performed.
    The hard memory bound remains max_tracked_sequences (FIFO-capped), which is
    unchanged, and chunk_ceiling_hard_max is a further absolute backstop.

CHUNK DEPTH IS ALSO BUDGET-DERIVED (lit-pull 2026-07-27, Solway 2014)
    The R4/R3 recursion cap of 3 was the OTHER fiat constant on this trade-off,
    and Solway, Diuk, Cordova, Yee, Barto, Niv & Botvinick 2014 (PLoS Comput
    Biol, 10.1371/journal.pcbi.1003779) is the reason it could not stay one.
    That entry is filed in the corpus as a deliberate NULL: the normative
    account of what makes one action hierarchy better than another is the paper
    that WOULD carry a principled depth limit at policy grain, and it declines
    to give one. Its own analysis "assumes that hierarchies are one level deep",
    a restriction "adopted to assure computational tractability in the present
    application", with the framework generalising "without any alteration to
    deeper hierarchies".

    That is the same move ARC-071 makes and for the same stated reason, and it
    has the consequence the corpus entry draws out: a COST-DERIVED cap should
    MOVE AS COMPUTE ALLOWS rather than sitting at a fixed constant defended by
    citation. The R3 sources that supply the number 3-4 (Badre & D'Esposito
    2009's rostro-caudal hierarchy, Koechlin & Summerfield 2007's cascade) are
    anatomical grain, not policy grain, so they describe how deep a brain's
    control hierarchy runs -- not how deep THIS agent can afford to search.

    use_growable_chunk_depth (default OFF) implements this as:

      the depth ceiling starts at max_depth (3), grows one LEVEL at a time
      toward derived_chunk_max_depth = floor(chunk_deliberation_horizon *
      chunk_depth_budget_fraction), each step licensed by a realised marginal
      return, and never past what the size ceiling makes structurally
      reachable.

    THE NUMBER 3 IS NOT REPLACED, on exactly the argument used for the size
    ceiling. Solway 2014 licenses no replacement depth, so the fraction is
    ANCHORED (0.1 = 3/30) to return exactly 3 at REE's actual rollout horizon of
    30. An agent at today's deliberation budget is left precisely where it was;
    only a LARGER budget derives a deeper hierarchy. What changed is the
    parameter's shape, from constant to function.

    WHY LINEAR IN THE BUDGET AND NOT LOGARITHMIC. The intuition that depth
    should be logarithmic comes from hierarchies whose span MULTIPLIES per
    level (D levels of branching factor b span b**D primitives, so an affordable
    depth goes as log of the budget). REE's descent is not that. Each further
    level is one more sequential re-tiling pass -- HippocampalModule's
    _recursive_leaf_tiles is bounded by `iterations < depth_cap`, an ITERATION
    count -- so the cost of depth D is linear in D and the affordable depth is
    linear in the budget. The fraction absorbs the unknown per-level cost
    constant, exactly as chunk_ceiling_budget_fraction absorbs the unknown
    how-much-of-the-horizon-may-one-chunk-span constant. What the derivation
    asserts is the SHAPE; the scale is anchored, not claimed.

    THE STRUCTURAL COUPLING TO CHUNK SIZE, which is the sharp part. Depth here
    is not free of the size ceiling -- it is BOUNDED by it. _depth_for computes
    a candidate's depth as 1 + the depth of the deepest registered chunk it
    contains, and a containing sequence must be strictly LONGER than what it
    contains, so a depth-D chain needs D distinct sequence lengths drawn from
    [min_chunk_size, effective ceiling]. The deepest hierarchy this substrate
    can physically mint is therefore

        structural_max_depth = effective_max_chunk_size - min_chunk_size + 1

    which at the default 2-5 budget is 4. Verified by construction: with
    max_depth raised to 4 a four-level chain forms, and at 5 nothing further
    appears -- 5 and 4 are the same run. So a depth ceiling raised above the
    structural bound is INERT, which is precisely the defect the MECH-321
    scoping spike found in decomposition_depth_cap and which this work exists to
    avoid repeating. consider_depth_growth() therefore refuses to grow past the
    structural bound, and because that bound reads the LIVE (possibly grown)
    size ceiling, the two parameters move together mechanically and not merely
    by analogy. Note also that max_depth=3 is genuinely BINDING today: at the
    2-5 budget the substrate would mint a depth-4 chunk and the cap refuses it.

    THE BRAKE IS AGAIN THE GROWTH RULE. Growth needs a realised marginal outcome
    gain from composing AT the current depth ceiling over the best chunk one
    level shallower that it contains -- did the last NESTING actually buy
    anything. Same reasoning as for size: a depth that grew monotonically with
    practice would keep deepening until it hit whatever number it stopped at,
    which is the fixed-constant failure in a slower disguise.

    DECOUPLED FROM R_min ON THE SAME GROUNDS (Bo 2009). min_repetitions enters
    the depth returns test only as a judge-ability filter -- is this chunk's
    outcome bucket attested enough to be measured -- never as the thing being
    measured.

    NOT IMPLEMENTED -- SHRINKAGE, identically to the size ceiling. Nothing
    lowers a depth ceiling once raised; the cross-agent half is covered by the
    derivation, the within-lifetime half needs a declining-capacity signal REE
    does not have. Recorded as a gap rather than guessed at.

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

DISSOLUTION IS SUPPRESSION-WITH-RETENTION, NOT ERASURE (lit-pull 2026-07-27)
    Two independent entries in targeted_review_connectome_mech_323 converge on a
    structural correction to the DISSOLVED terminal state:

      Barnes, Kubota, Hu, Jin & Graybiel 2005 (Nature, 10.1038/nature04053) --
        sensorimotor striatum across acquisition / extinction / reacquisition.
        Task-related ensemble patterns were "successively formed, reversed and
        then re-emerged". Habits take extensive repetition to form, but
        "regaining a habit can occur quickly, with even one or a few exposures".
      Bouton, Winterbauer & Todd 2012 (Behav Processes, 10.1016/j.beproc.2012.03.004)
        -- INSTRUMENTAL extinction, i.e. at the action level ARC-071 operates on.
        Extinction "weakens behavior without erasing the original learning": it
        installs new, context-dependent learning ALONGSIDE the old. Three relapse
        effects follow -- renewal, resurgence, rapid reacquisition.

    As first built (2026-07-22), DISSOLVED was worse than erasure. The chunk was
    retained in the library dict for the audit trail, but nothing could ever
    revive it: PolicyChunking.note_outcome() skips any sequence already present
    in the library, and note_real_execution() has no DISSOLVED branch. So
    DISSOLVED was an ABSORBING TOMBSTONE that also permanently BLOCKED the
    sequence from re-forming. Measured on the contract fixture: after forcing a
    crystallised chunk to DISSOLVED, 200 further trials of the same perfectly
    consistent, above-baseline regime left it DISSOLVED with zero re-formations.
    Erasure would at least have permitted re-formation at R_min.

    use_chunk_dissolution_retention (default OFF) implements the cheapest and
    sharpest of the three relapse effects, RAPID REACQUISITION: a DISSOLVED
    chunk is DORMANT rather than dead, and re-forming it needs materially fewer
    than R_min repetitions --

        reacquisition_min_repetitions = ceil(R_min * reacquisition_repetition_factor)

    The other two gates of the MECH-323 joint formation condition (variance <
    F_low, mean > baseline + margin) are applied UNCHANGED, so only the
    repetition requirement is relaxed. That is the single-parameter form the
    Bouton entry asks for and it is directly falsifiable: dissolve a crystallised
    chunk, re-present its conditions, count repetitions-to-re-formation against
    R_min.

    Repetitions are counted SINCE DISSOLUTION (ChunkedPrimitive.
    reacquisition_repetitions), not as the raw tally-bucket length. This is not
    incidental. The accumulator's per-sequence tally is a sliding window capped
    at window_trials, and a long-lived chunk's bucket sits saturated at that cap,
    so a naive "compare the bucket length against a lowered R_min" would clear
    the reduced bar on the very first post-dissolution trial and measure nothing.

    A revived chunk returns to FORMING, not to CRYSTALLISED: rapid reacquisition
    is a claim about the FORMATION threshold, while the crystallisation counter
    (C_min) is the separate Smith & Graybiel 2013 IL sub-mechanism and is made to
    run again from zero.

    REPLAY-ORIGIN CHUNKS ARE NEVER REVIVED (fails closed). A MECH-322
    replay-origin chunk retired on its corroboration deadline died precisely
    because real waking execution never corroborated it; letting it return by a
    REDUCED threshold would be a shortcut around the MECH-094 posture the
    carve-out was built to preserve. Reacquisition would in fact be
    real-execution-driven here, but the conservative reading is the one that
    matches a SAFETY-CRITICAL flag, and the cost of refusing is nil.

    NOT IMPLEMENTED -- the other two relapse effects are design questions, not
    builds, and are recorded here so the gap stays visible:
      renewal    : requires dissolution to be gated against the chunk's existing
                   initiation_set. Formation is context-conditioned (chunks carry
                   an initiation set) while dissolution is context-BLIND, so a
                   chunk dissolved in one context is dissolved everywhere and REE
                   cannot exhibit renewal at all. Closing this needs a per-context
                   dissolution state, not a parameter.
      resurgence : requires a chunk to return because a COMPETITOR was
                   extinguished, not because its own evidence improved. That needs
                   dissolution state shared across competing library candidates;
                   registration is per-sequence and produces no such coupling.

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
from math import ceil as _ceil
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
                   TERMINAL by default. Under use_chunk_dissolution_retention it
                   is DORMANT instead: still unselectable, but re-formable at the
                   reduced reacquisition_min_repetitions bar (DISSOLVED ->
                   FORMING). See the module docstring's dissolution-is-suppression
                   section (Barnes 2005 / Bouton 2012).

    MECH-322 replay-origin chunks that miss their corroboration deadline go
    CRYSTALLISED -> DISSOLVED directly, bypassing the slower DISSOLVING window,
    and are never revivable even with retention on.
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
        n_dissolutions : how many times this chunk has reached DISSOLVED. The
            dormancy trace that survives dissolution (Barnes 2005 re-emergence).
        n_reacquisitions : how many times it has been revived out of DISSOLVED
            by the reduced-threshold path. Rapid reacquisition's readout.
        reacquisition_repetitions : executions observed SINCE the most recent
            dissolution, counted against reacquisition_min_repetitions rather
            than R_min. Zeroed on each dissolution and on each revival. Not the
            accumulator's tally length -- see the module docstring on why the
            saturated sliding window cannot serve here.
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
    n_dissolutions: int = 0
    n_reacquisitions: int = 0
    reacquisition_repetitions: int = 0

    @property
    def key(self) -> Tuple[int, ...]:
        """Identity of the chunk = its action sequence."""
        return tuple(self.sequence)

    @property
    def is_selectable(self) -> bool:
        """True iff this chunk may be spliced into the proposal pool.

        Unchanged by retention: a DORMANT chunk is SUPPRESSED, so it stays out
        of the pool until it actually re-forms and re-crystallises. Retention
        buys a cheaper route back, not a free pass back in.
        """
        return (
            self.state is ChunkState.CRYSTALLISED
            or (self.state is ChunkState.DISSOLVING and self.selection_weight > 0.0)
        )

    @property
    def is_dormant(self) -> bool:
        """True iff DISSOLVED and eligible for reduced-threshold re-formation.

        State-only, so it reads True regardless of the retention flag; the flag
        is what decides whether anything ACTS on it. Replay-origin chunks are
        excluded: they fail closed and can never be revived.
        """
        return self.state is ChunkState.DISSOLVED and not self.replay_origin

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
            "n_dissolutions": int(self.n_dissolutions),
            "n_reacquisitions": int(self.n_reacquisitions),
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
        min_chunk_size / max_chunk_size : the INITIAL chunk-size budget per
            level (Sakai 2003, 2-5 elements). Under
            use_growable_chunk_ceiling this is the STARTING value of a
            growable ceiling rather than a lifetime cap -- see the
            module docstring's growable-ceiling section.
        use_growable_chunk_ceiling : MECH-323 sub-switch. False (default) =
            max_chunk_size is a hard lifetime cap, the as-first-built
            behaviour, bit-identical. True = the effective ceiling starts at
            max_chunk_size and may grow toward derived_chunk_ceiling, one
            element at a time, each step licensed by a realised marginal
            return (Ramkumar 2016).
        chunk_deliberation_horizon : the agent's rollout/planning horizon,
            mirrored from HippocampalConfig.horizon. The ceiling is DERIVED
            from this rather than configured independently, which is what
            makes two agents with different deliberation budgets settle at
            different chunk sizes.
        chunk_ceiling_budget_fraction : the fraction of the deliberation
            horizon a single chunk may span. Default 0.1667 = 5/30 is
            ANCHORED so that at REE's actual rollout horizon (30, the
            from_dims HippocampalConfig.horizon) the derivation returns
            exactly 5 -- it recovers the inherited Sakai budget rather
            than replacing it with a new number. See derived_chunk_ceiling.
        chunk_ceiling_returns_threshold : the diminishing-returns BRAKE.
            Minimum marginal outcome gain (same 0-1 normalised scale as
            evaluative_margin) that composing at the current ceiling must
            deliver over the best sub-sequence one element shorter, before
            the ceiling may grow again. UNCALIBRATED ENGINEERING DEFAULT --
            Ramkumar 2016 establishes that a brake EXISTS and that chunk
            structures plateau around 50-60% of single-unit complexity, but
            quantifies no threshold on this scale. Exactly the status of
            F_low / F_high: the RELATION is literature-grounded, the VALUE
            is not.
        chunk_ceiling_hard_max : absolute backstop on the grown ceiling.
            A TRACTABILITY bound, not a claim about chunk size.
        max_depth : chunks-of-chunks recursion cap (R4: 2-3 levels). Under
            use_growable_chunk_depth this is the STARTING value of a growable
            depth ceiling rather than a lifetime cap -- see the module
            docstring's growable-depth section.
        use_growable_chunk_depth : ARC-071 sub-switch, the DEPTH counterpart of
            use_growable_chunk_ceiling. False (default) = max_depth is a hard
            lifetime cap, the as-first-built behaviour, bit-identical. True =
            the ceiling starts at max_depth and may grow one LEVEL at a time
            toward derived_chunk_max_depth, each step licensed by a realised
            marginal return AND by the depth being structurally reachable at
            the current size ceiling (Solway 2014).
        chunk_depth_budget_fraction : the fraction of the deliberation horizon
            a full hierarchy descent may consume. Default 0.1 = 3/30 is
            ANCHORED so that at REE's actual rollout horizon (30) the
            derivation returns exactly 3 -- it recovers the inherited R3 cap
            rather than replacing it with a new number. Shares
            chunk_deliberation_horizon with the size ceiling deliberately:
            both parameters are settings on ONE compute-versus-efficiency
            trade-off, so they must read the same budget. See
            derived_chunk_max_depth.
        chunk_depth_returns_threshold : the diminishing-returns BRAKE on depth.
            Minimum marginal outcome gain a chunk at the current depth ceiling
            must deliver over the best chunk one level shallower that it
            contains, before the ceiling may grow again. DELIBERATELY A
            SEPARATE KNOB from chunk_ceiling_returns_threshold: sharing one
            would couple depth and size through the brake, and the coupled-
            parameter experiment this substrate exists to enable could then not
            tell a shared-budget effect from a shared-threshold artefact.
            UNCALIBRATED ENGINEERING DEFAULT, same status as F_low / F_high.
        chunk_depth_hard_max : absolute backstop on the grown depth ceiling.
            A TRACTABILITY bound, not a claim about hierarchy depth -- it sits
            above the 3-4 levels the R3 anatomical sources report so that it
            never binds before the derivation or the structural bound does.
        max_library_size : hard cap on retained chunks. Bounds memory; the
            lowest-value DISSOLVED chunks are evicted first.
        max_tracked_sequences : hard cap on the candidate tally table. Bounds
            the combinatorial sub-sequence enumeration.
        use_chunk_maintenance : MECH-324 sub-switch. False = chunks form but
            never crystallise (the registered ARM_1 dissociation arm).
        crystallisation_min : C_min corroborating real executions to crystallise.
        dissolve_trials : T_dissolve. Linear selection_weight decay window.
        use_chunk_dissolution_retention : MECH-324 sub-switch under
            use_chunk_maintenance. False (default) = DISSOLVED is terminal and
            also permanently blocks the sequence from re-forming, which is the
            as-first-built behaviour. True = DISSOLVED is DORMANT: the chunk is
            retained unselectable but can re-form at the reduced
            reacquisition_min_repetitions bar (Barnes 2005 / Bouton 2012 --
            dissolution is suppression with retention, not erasure). Bit-identical
            to False-behaviour when False. Requires use_chunk_maintenance: with
            maintenance off nothing ever dissolves, so retention would be a
            silently inert flag rather than an arm -- config validation REFUSES
            that pairing rather than tolerating it.
        reacquisition_repetition_factor : the rapid-reacquisition parameter.
            Multiplies R_min to give the repetition bar a DORMANT chunk must
            clear to re-form. Must be in (0, 1] -- at 1.0 reacquisition is no
            faster than acquisition, which is the null this flag exists to test
            against. The 0.25 default is an ENGINEERING DEFAULT: both source
            entries establish that reacquisition is much faster, and neither
            quantifies how much faster, exactly as neither quantifies F_high.
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
    use_growable_chunk_ceiling: bool = False
    chunk_deliberation_horizon: int = 10
    chunk_ceiling_budget_fraction: float = 0.1667
    chunk_ceiling_returns_threshold: float = 0.10
    chunk_ceiling_hard_max: int = 12
    max_depth: int = 3
    use_growable_chunk_depth: bool = False
    chunk_depth_budget_fraction: float = 0.1
    chunk_depth_returns_threshold: float = 0.10
    chunk_depth_hard_max: int = 6
    max_library_size: int = 64
    max_tracked_sequences: int = 512
    use_chunk_maintenance: bool = False
    crystallisation_min: int = 5
    dissolve_trials: int = 50
    use_chunk_dissolution_retention: bool = False
    reacquisition_repetition_factor: float = 0.25
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
        if self.chunk_deliberation_horizon < 1:
            raise ValueError("chunk_deliberation_horizon must be >= 1")
        if not (0.0 < self.chunk_ceiling_budget_fraction <= 1.0):
            raise ValueError(
                "chunk_ceiling_budget_fraction must be in (0, 1] (a chunk cannot "
                "usefully span more than the horizon it is evaluated over)"
            )
        if self.chunk_ceiling_returns_threshold < 0.0:
            raise ValueError(
                "chunk_ceiling_returns_threshold must be >= 0 (it is the "
                "diminishing-returns BRAKE; a negative bar would license growth "
                "on a merge that made outcomes WORSE, which is precisely the "
                "monotonic-growth failure mode Ramkumar 2016 rules out)"
            )
        if self.chunk_ceiling_hard_max < self.max_chunk_size:
            raise ValueError(
                "chunk_ceiling_hard_max must be >= max_chunk_size (the ceiling "
                "starts at max_chunk_size and only ever grows)"
            )
        if self.max_depth < 1:
            raise ValueError("max_depth must be >= 1")
        if not (0.0 < self.chunk_depth_budget_fraction <= 1.0):
            raise ValueError(
                "chunk_depth_budget_fraction must be in (0, 1] (a hierarchy "
                "cannot usefully be deeper than the horizon it is evaluated "
                "over is long)"
            )
        if self.chunk_depth_returns_threshold < 0.0:
            raise ValueError(
                "chunk_depth_returns_threshold must be >= 0 (it is the "
                "diminishing-returns BRAKE on depth; a negative bar would "
                "license a further nesting level on a merge that made outcomes "
                "WORSE, the monotonic-growth failure mode Solway 2014's "
                "cost-derived framing rules out)"
            )
        if self.chunk_depth_hard_max < self.max_depth:
            raise ValueError(
                "chunk_depth_hard_max must be >= max_depth (the depth ceiling "
                "starts at max_depth and only ever grows)"
            )
        if self.crystallisation_min < 1:
            raise ValueError("crystallisation_min must be >= 1")
        if self.dissolve_trials < 1:
            raise ValueError("dissolve_trials must be >= 1")
        if not (0.0 < self.reacquisition_repetition_factor <= 1.0):
            raise ValueError(
                "reacquisition_repetition_factor must be in (0, 1] "
                "(it SCALES DOWN min_repetitions for a dormant chunk; > 1 would "
                "make reacquisition slower than acquisition, inverting the "
                "Barnes 2005 / Bouton 2012 asymmetry this implements)"
            )
        if self.use_chunk_dissolution_retention and not self.use_chunk_maintenance:
            # Loud precondition, not silent tolerance: with maintenance off no
            # chunk ever reaches DISSOLVED, so retention would read as enabled
            # in a manifest while never running -- a false null.
            raise ValueError(
                "use_chunk_dissolution_retention requires use_chunk_maintenance "
                "(with maintenance off no chunk ever dissolves, so the retention "
                "path could never fire and the flag would be silently inert)"
            )
        if not (0.0 < self.replay_value_quantile < 1.0):
            raise ValueError("replay_value_quantile must be in (0, 1)")
        if self.replay_corroboration_episodes < 1:
            raise ValueError("replay_corroboration_episodes must be >= 1")

    @property
    def reacquisition_min_repetitions(self) -> int:
        """Repetition bar for re-forming a DORMANT chunk. Floored at 1.

        ceil() rather than int(): with R_min = 20 and factor = 0.25 both give 5,
        but int() would silently collapse a small R_min to the floor (R_min = 3,
        factor = 0.25 -> int 0 -> floored to 1 = no bar at all, vs ceil 1 which
        is the same value arrived at honestly). ceil keeps the bar a genuine
        fraction of R_min at every setting.
        """
        raw = float(self.min_repetitions) * float(self.reacquisition_repetition_factor)
        return max(1, int(_ceil(raw)))

    @property
    def derived_chunk_ceiling(self) -> int:
        """Upper bound the growable ceiling may reach, DERIVED not configured.

        The Ramkumar 2016 trade-off is between planning cost -- which grows with
        the horizon a movement is optimised over -- and achievable efficiency.
        The quantity that sets how long a chunk can usefully be is therefore the
        agent's own deliberation budget: a chunk longer than the horizon it is
        evaluated over cannot be assessed as a unit, so the horizon is the
        natural bound.

            floor(chunk_deliberation_horizon * chunk_ceiling_budget_fraction)

        clamped below by max_chunk_size (the ceiling only ever grows from its
        initial value) and above by chunk_ceiling_hard_max (tractability).

        AT REE'S ACTUAL DELIBERATION BUDGET THIS RETURNS 5. The real horizon is
        30 (HippocampalConfig.horizon as set by REEConfig.from_dims -- NOT the
        HippocampalConfig dataclass default of 10, which no built agent uses),
        and 30 x 0.1667 = 5 = the inherited Sakai budget.

        That anchoring is the whole reason this is not a disguised way of
        raising the constant. Ramkumar 2016 gives the SHAPE of the relationship
        (ceiling scales with the deliberation budget) but proposes no upper
        bound, so it cannot license any particular replacement number. The scale
        therefore has to be pinned somewhere, and the only defensible anchor is
        the one number that does have warrant: the Sakai budget, at the budget
        the agent actually has. An agent at today's horizon is left exactly
        where it was; only an agent with a LARGER deliberation budget derives a
        larger ceiling. What changed is the parameter's shape, from a global
        constant to a function of the agent's compute.

        The 1e-9 is a floor guard, not a fudge: the anchor case is exactly
        integral (30 x 0.1667 = 5.001, but an exact 5/6-style fraction would
        land on 4.999...), and silently flooring the anchor to 4 would move the
        inherited budget while claiming to preserve it.
        """
        raw = float(self.chunk_deliberation_horizon) * float(
            self.chunk_ceiling_budget_fraction
        )
        return max(
            self.max_chunk_size, min(self.chunk_ceiling_hard_max, int(raw + 1e-9))
        )

    @property
    def derived_chunk_max_depth(self) -> int:
        """Deepest hierarchy the deliberation budget licenses. DERIVED, not set.

            floor(chunk_deliberation_horizon * chunk_depth_budget_fraction)

        clamped below by max_depth (the ceiling only ever grows from its initial
        value) and above by chunk_depth_hard_max (tractability).

        AT REE'S ACTUAL DELIBERATION BUDGET THIS RETURNS 3. The real horizon is
        30 (HippocampalConfig.horizon as set by REEConfig.from_dims -- NOT the
        HippocampalConfig dataclass default of 10, which no built agent uses),
        and 30 x 0.1 = 3 = the inherited R3 cap.

        Solway et al. 2014 caps its own hierarchies at one level for stated
        tractability reasons and says the framework generalises to deeper ones
        unaltered, so it establishes that REE's cap is a COST bound that should
        move with compute -- and licenses no particular replacement depth. The
        scale is therefore anchored to the one number that has warrant (the R3
        cap, at the budget the agent actually has) rather than invented here.
        Reading it any other way makes this a disguised raise of the constant.

        NOT the whole story at growth time. This is the BUDGET bound only. The
        substrate also has a STRUCTURAL bound -- a depth-D chain needs D
        distinct sequence lengths, so it cannot exceed
        effective_max_chunk_size - min_chunk_size + 1 -- and a ceiling above
        that is inert. The structural bound is applied by
        PolicyChunking.consider_depth_growth(), not here, because only the
        facade can see the accumulator's LIVE (possibly grown) size ceiling;
        keeping this property a pure function of config is what lets the
        derivation be contract-pinned without building an agent.

        The 1e-9 is the same floor guard as derived_chunk_ceiling: the anchor
        case is exactly integral and silently flooring 3.0 to 2 would move the
        inherited cap while claiming to preserve it.
        """
        raw = float(self.chunk_deliberation_horizon) * float(
            self.chunk_depth_budget_fraction
        )
        return max(self.max_depth, min(self.chunk_depth_hard_max, int(raw + 1e-9)))


def _contains_subsequence(
    haystack: Tuple[int, ...], needle: Tuple[int, ...]
) -> bool:
    """True iff `needle` appears as a CONTIGUOUS run inside `haystack`.

    The containment relation the chunk hierarchy is built on: a chunk composes
    another chunk exactly when it contains it contiguously. Shared by
    _depth_for (which assigns depth) and marginal_return_at_depth_ceiling
    (which asks whether that nesting paid), so the two can never drift.
    """
    n = len(needle)
    if n == 0 or n > len(haystack):
        return False
    return any(haystack[i : i + n] == needle for i in range(len(haystack) - n + 1))


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

        # Growable ceiling (MECH-323, Ramkumar 2016). Starts at the INITIAL
        # budget and only ever grows, one element at a time, each step licensed
        # by a realised marginal return. Read through effective_max_chunk_size,
        # never directly -- that property is what makes the flag-off path
        # bit-identical.
        self._ceiling: int = int(self.config.max_chunk_size)
        self._n_ceiling_growths: int = 0
        self._last_ceiling_gain: float = 0.0

    @property
    def effective_max_chunk_size(self) -> int:
        """The chunk-size ceiling actually in force this trial.

        With use_growable_chunk_ceiling off this is exactly config.max_chunk_size
        and the grown value is never consulted, so every enumeration bound below
        is bit-identical to the as-first-built behaviour.
        """
        if not self.config.use_growable_chunk_ceiling:
            return int(self.config.max_chunk_size)
        return int(self._ceiling)

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
        limit = max(self.effective_max_chunk_size * 4, 32)
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
        for size in range(c.min_chunk_size, self.effective_max_chunk_size + 1):
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

    # ------------------------------------------------------------------
    # Growable ceiling (Ramkumar 2016) -- growth IS the returns brake
    # ------------------------------------------------------------------
    def marginal_return_at_ceiling(self) -> Optional[float]:
        """Best realised outcome gain from composing AT the current ceiling.

        For each tracked sub-sequence whose length equals the current ceiling and
        which has been repeated enough to be judged, compare its outcome mean
        against the better of the two sub-sequences ONE ELEMENT SHORTER that it
        contains (drop-the-first and drop-the-last). That difference is the
        marginal return of the last merge -- did going from n-1 to n actually buy
        anything.

        Only the immediate n-1 predecessors are consulted, not every shorter
        sub-sequence. That is both the cheaper computation (two dict lookups per
        candidate rather than a quadratic sweep) and the more faithful one: the
        question the brake asks is whether the LAST merge paid, not whether the
        chunk beats its smallest constituent.

        Returns:
            The best marginal gain found, or None if no sequence at the ceiling
            is yet judgeable -- which is NOT a gain of zero and must not be
            treated as one. None means "no evidence either way", and the caller
            refuses to grow on it.
        """
        c = self.config
        ceiling = self.effective_max_chunk_size
        if ceiling <= c.min_chunk_size:
            # No shorter sub-sequence exists to compare against.
            return None
        best: Optional[float] = None
        for key, outcomes in self._tally.items():
            if len(key) != ceiling:
                continue
            if len(outcomes) < c.min_repetitions:
                continue
            whole = _mean(outcomes)
            sub_means = [
                _mean(bucket)
                for bucket in (self._tally.get(key[1:]), self._tally.get(key[:-1]))
                if bucket
            ]
            if not sub_means:
                continue
            gain = whole - max(sub_means)
            if best is None or gain > best:
                best = gain
        return best

    def consider_ceiling_growth(self) -> bool:
        """Grow the ceiling by one element iff the last merge paid off.

        This is the whole of the Ramkumar 2016 brake, and it is deliberately the
        GROWTH RULE rather than a separate cap bolted on afterwards. An
        accumulator that grew with accumulated practice and stopped at some
        number would still be monotonic, and would still eventually collapse the
        sequence into a single chunk once that number was reached -- which the
        efficiency/computation trade-off rules out (the macaques never do it, and
        chunk structures plateau near 50-60% of single-unit complexity). Tying
        each growth step to a realised marginal return makes the plateau an
        OUTCOME of the rule rather than a second parameter: when merging stops
        paying, growth stops on its own, at whatever size that happens to be.

        Growth is refused, and each refusal is silent-but-diagnosable via
        chunk_acc_ceiling_* in get_state(), when:
            - the flag is off                       -> never grows at any setting
            - the ceiling is already at the derived deliberation-budget bound
            - no sequence at the ceiling is judgeable yet (gain is None)
            - the best marginal gain is below chunk_ceiling_returns_threshold

        DELIBERATELY NOT IMPLEMENTED -- SHRINKAGE. Bo et al. 2009 show chunk
        length DECLINES as visuospatial working-memory capacity declines with
        age, so the fully faithful parameter would fall as well as rise. The
        cross-agent half of that is already covered, because derived_chunk_ceiling
        is a function of the agent's deliberation budget and a smaller-budget
        agent simply derives a smaller bound. The within-lifetime half is not:
        nothing here lowers a ceiling once raised. Closing it needs a declining
        capacity signal to read from, which REE does not currently have, so it is
        recorded as a gap rather than guessed at.

        Returns:
            True iff the ceiling grew this call.
        """
        c = self.config
        if not c.use_growable_chunk_ceiling:
            return False
        if self._ceiling >= c.derived_chunk_ceiling:
            return False
        gain = self.marginal_return_at_ceiling()
        if gain is None:
            return False
        self._last_ceiling_gain = float(gain)
        if gain < c.chunk_ceiling_returns_threshold:
            return False
        self._ceiling += 1
        self._n_ceiling_growths += 1
        return True

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
        if not (c.min_chunk_size <= len(seq) <= self.effective_max_chunk_size):
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
        self._ceiling = int(self.config.max_chunk_size)
        self._n_ceiling_growths = 0
        self._last_ceiling_gain = 0.0

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
            # Growable-ceiling readout (Ramkumar 2016). With the flag off the
            # effective ceiling stays pinned at max_chunk_size and growths stay
            # 0, which is what makes the OFF arm identifiable in a manifest.
            "chunk_acc_effective_ceiling": self.effective_max_chunk_size,
            "chunk_acc_ceiling_derived_max": self.config.derived_chunk_ceiling,
            "chunk_acc_n_ceiling_growths": self._n_ceiling_growths,
            "chunk_acc_last_ceiling_gain": float(self._last_ceiling_gain),
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
        _n_reacquisitions : dormant chunks revived at the reduced bar
        _n_reacquisition_refusals : revive() calls refused (flag off, not
            dormant, or replay-origin) -- refusals are counted, not dropped
    """

    def __init__(self, config: Optional[PolicyChunkingConfig] = None) -> None:
        self.config = config if config is not None else PolicyChunkingConfig()
        self.config.validate()
        self._chunks: Dict[Tuple[int, ...], ChunkedPrimitive] = {}
        self._n_registered: int = 0
        self._n_crystallised: int = 0
        self._n_dissolved: int = 0
        self._n_replay_deadline_dissolutions: int = 0
        self._n_reacquisitions: int = 0
        self._n_reacquisition_refusals: int = 0

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

        KNOWN LIMITATION under use_chunk_dissolution_retention: DISSOLVED is
        also exactly the dormant pool, so max_library_size eviction is the one
        remaining path by which a dormant trace is genuinely ERASED and its
        sequence loses reduced-threshold re-formation. Left deliberately
        unchanged -- the memory bound is not negotiable and a dormancy-aware
        eviction order is an unmeasured refinement, not a correctness fix. It
        does mean a reacquisition experiment must size max_library_size above
        the number of chunks it expects to dissolve, or it will measure the
        eviction policy instead of the mechanism.
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
        elif chunk.state is ChunkState.DISSOLVED:
            # Dissolution is SUPPRESSION, not erasure (Barnes 2005 / Bouton
            # 2012): a real execution of a DORMANT chunk accrues toward the
            # reduced re-formation bar. This does not itself revive anything --
            # PolicyChunking._attempt_reacquisition applies the two unchanged
            # MECH-323 gates before that. With retention off, DISSOLVED is
            # terminal and nothing is counted at all.
            if c.use_chunk_dissolution_retention and chunk.is_dormant:
                chunk.reacquisition_repetitions += 1
        return chunk.state

    def _mark_dissolved(self, chunk: ChunkedPrimitive) -> None:
        """Enter DISSOLVED and lay down the dormancy trace.

        One helper for BOTH dissolution sites -- the slow T_dissolve decay and
        the MECH-322 corroboration deadline -- so the trace cannot be recorded
        at one and silently forgotten at the other. Inert when retention is off:
        the two new fields are written but nothing reads them.
        """
        chunk.state = ChunkState.DISSOLVED
        chunk.selection_weight = 0.0
        chunk.n_dissolutions += 1
        chunk.reacquisition_repetitions = 0
        self._n_dissolved += 1

    def dormant_chunks(self) -> List[ChunkedPrimitive]:
        """DISSOLVED chunks eligible for reduced-threshold re-formation.

        Empty unless retention is on -- with the flag off DISSOLVED is terminal
        and calling these chunks "dormant" would misdescribe the substrate.
        """
        if not (
            self.config.use_chunk_maintenance
            and self.config.use_chunk_dissolution_retention
        ):
            return []
        return [c for c in self._chunks.values() if c.is_dormant]

    def revive(
        self, sequence: Sequence[int], value_tag: Optional[float] = None
    ) -> bool:
        """Return a DORMANT chunk to FORMING. Rapid reacquisition's write.

        FORMING, not CRYSTALLISED: rapid reacquisition is a claim about the
        FORMATION threshold. The crystallisation counter is the separate Smith &
        Graybiel 2013 IL sub-mechanism and is made to run again from zero, so a
        revived chunk still has to earn C_min real executions before it becomes
        selectable. What retention buys is a cheaper route back, not a free one.

        Every refusal path is counted rather than silently dropped, and each
        fails CLOSED:
            - retention (or maintenance) off        -> no revival at any setting
            - not DISSOLVED                         -> nothing to revive
            - replay_origin                         -> MECH-322 chunks retired
              for want of real corroboration must not return by a REDUCED bar

        Args:
            sequence : the dormant chunk's action sequence.
            value_tag : refreshed provenance value from the executions that
                earned the revival. None leaves the pre-dissolution value_tag.

        Returns:
            True iff the chunk was revived.
        """
        if not (
            self.config.use_chunk_maintenance
            and self.config.use_chunk_dissolution_retention
        ):
            self._n_reacquisition_refusals += 1
            return False
        chunk = self.get(sequence)
        if chunk is None:
            self._n_reacquisition_refusals += 1
            return False
        if not chunk.is_dormant:
            # Covers both "not DISSOLVED" and the replay-origin refusal.
            self._n_reacquisition_refusals += 1
            return False
        chunk.state = ChunkState.FORMING
        chunk.selection_weight = 0.0
        chunk.crystallisation_counter = 0
        chunk.dissolving_trials = 0
        chunk.episodes_since_corroboration = 0
        chunk.reacquisition_repetitions = 0
        chunk.n_reacquisitions += 1
        if value_tag is not None:
            chunk.value_tag = float(value_tag)
        self._n_reacquisitions += 1
        return True

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
                    self._mark_dissolved(chunk)

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
                self._mark_dissolved(chunk)
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
        self._n_reacquisitions = 0
        self._n_reacquisition_refusals = 0

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
            # Rapid-reacquisition readout (Barnes 2005 / Bouton 2012). Stays 0
            # unless use_chunk_dissolution_retention is deliberately on.
            "chunk_lib_n_reacquisitions": self._n_reacquisitions,
            "chunk_lib_n_reacquisition_refusals": self._n_reacquisition_refusals,
            "chunk_lib_n_dormant": len(self.dormant_chunks()),
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

        # Growable DEPTH ceiling (Solway 2014). Lives on the facade rather than
        # on either operator because the growth rule needs both: the library
        # supplies the chunk hierarchy and its depths, the accumulator supplies
        # the realised outcome tally and the live size ceiling the structural
        # bound is read off. Read through effective_max_depth, never directly --
        # that property is what makes the flag-off path bit-identical.
        self._depth_ceiling: int = int(self.config.max_depth)
        self._n_depth_growths: int = 0
        self._last_depth_gain: float = 0.0

    @property
    def effective_max_depth(self) -> int:
        """The chunks-of-chunks depth cap actually in force this trial.

        With use_growable_chunk_depth off this is exactly config.max_depth and
        the grown value is never consulted, so the formation gate below is
        bit-identical to the as-first-built behaviour.
        """
        if not self.config.use_growable_chunk_depth:
            return int(self.config.max_depth)
        return int(self._depth_ceiling)

    @property
    def structural_max_depth(self) -> int:
        """Deepest hierarchy this substrate can physically mint right now.

        A chunk composes another only by CONTAINING it contiguously, so each
        further level needs a strictly longer sequence, drawn from the lengths
        [min_chunk_size, effective_max_chunk_size]. The number of available
        lengths is therefore the number of available levels.

        Reads the accumulator's LIVE size ceiling, so under
        use_growable_chunk_ceiling this bound RISES as the size ceiling grows.
        That is the mechanical coupling between the two parameters: depth cannot
        outrun the size budget that has to carry it.
        """
        return max(
            1,
            int(self.accumulator.effective_max_chunk_size)
            - int(self.config.min_chunk_size)
            + 1,
        )

    def marginal_return_at_depth_ceiling(self) -> Optional[float]:
        """Best realised outcome gain from nesting AT the current depth ceiling.

        For each registered, non-DISSOLVED chunk sitting exactly at the depth
        ceiling and repeated enough to be judged, compare its realised outcome
        mean against the best mean among the chunks ONE LEVEL SHALLOWER that it
        contains. That difference is the marginal return of the last NESTING --
        did going from depth d-1 to depth d actually buy anything.

        Means come from the accumulator's live tally rather than from
        ChunkedPrimitive.value_tag: value_tag is frozen at formation, so a chunk
        whose returns have since collapsed would keep licensing growth on a
        number that stopped being true. min_repetitions is a judge-ability
        filter here and nothing more (Bo 2009 -- do not turn this into a
        practice counter).

        Returns:
            The best marginal gain found, or None if nothing at the ceiling is
            yet judgeable -- which is NOT a gain of zero and must not be treated
            as one. None means "no evidence either way".
        """
        c = self.config
        ceiling = self.effective_max_depth
        if ceiling < 2:
            # Depth 1 composes raw actions, so there is no shallower chunk to
            # have gained anything over.
            return None
        live = [
            chunk
            for chunk in self.library.all_chunks()
            if chunk.state is not ChunkState.DISSOLVED
        ]
        shallower = [chunk for chunk in live if chunk.depth == ceiling - 1]
        if not shallower:
            return None
        tally = self.accumulator._tally
        best: Optional[float] = None
        for chunk in live:
            if chunk.depth != ceiling:
                continue
            outcomes = tally.get(chunk.key)
            if not outcomes or len(outcomes) < c.min_repetitions:
                continue
            sub_means = [
                _mean(tally[sub.key])
                for sub in shallower
                if _contains_subsequence(chunk.key, sub.key)
                and len(tally.get(sub.key, ())) >= c.min_repetitions
            ]
            if not sub_means:
                continue
            gain = _mean(outcomes) - max(sub_means)
            if best is None or gain > best:
                best = gain
        return best

    def consider_depth_growth(self) -> bool:
        """Grow the depth ceiling by one level iff the last nesting paid off.

        The depth counterpart of ChunkAccumulator.consider_ceiling_growth, and
        the same argument applies: making the growth rule itself the brake is
        what stops a cost-derived cap from becoming a slower fixed constant.

        Growth is refused, and each refusal is diagnosable via chunk_depth_* in
        get_state(), when:
            - the flag is off                    -> never grows at any setting
            - the ceiling is already at the derived deliberation-budget bound
            - the ceiling is already at the STRUCTURAL bound -- growing further
              could not mint a deeper chunk at the current size ceiling, so the
              raise would be INERT. This is the refusal that keeps depth and
              size moving together, and it is the specific defect
              (a depth knob with no degree of freedom) that the MECH-321
              scoping spike found in decomposition_depth_cap.
            - no chunk at the ceiling is judgeable yet (gain is None)
            - the best marginal gain is below chunk_depth_returns_threshold

        Returns:
            True iff the depth ceiling grew this call.
        """
        c = self.config
        if not c.use_growable_chunk_depth:
            return False
        if self._depth_ceiling >= c.derived_chunk_max_depth:
            return False
        if self._depth_ceiling >= self.structural_max_depth:
            return False
        gain = self.marginal_return_at_depth_ceiling()
        if gain is None:
            return False
        self._last_depth_gain = float(gain)
        if gain < c.chunk_depth_returns_threshold:
            return False
        self._depth_ceiling += 1
        self._n_depth_growths += 1
        return True

    def record_step(self, action_class: int, hypothesis_tag: bool = False) -> bool:
        """Record one executed action class (MECH-094-strict). See ChunkAccumulator."""
        return self.accumulator.record_step(action_class, hypothesis_tag=hypothesis_tag)

    def note_outcome(self, outcome_signal: float) -> List[ChunkedPrimitive]:
        """Report an outcome, run the formation pass, register new chunks.

        Returns the chunks NEWLY AVAILABLE from this pass -- freshly minted
        ones, plus (only under use_chunk_dissolution_retention) any dormant
        chunk revived by the reduced-threshold reacquisition path. Both are
        "this sequence just became a live chunk again" from a consumer's point
        of view, which is what the caller acts on; they are told apart by
        n_reacquisitions > 0 on the returned object. Empty list is the normal
        case, and is exactly what is returned when retention is off.
        """
        self.accumulator.note_outcome(outcome_signal)
        self._n_formation_passes += 1

        formed: List[ChunkedPrimitive] = []
        for seq, mu, var in self.accumulator.formation_candidates():
            if self.library.get(seq) is not None:
                continue
            depth = self._depth_for(seq)
            if depth > self.effective_max_depth:
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
        # Rapid reacquisition runs BEFORE tick_maintenance and AFTER the
        # note_real_execution pass that feeds its counter. Ordering matters:
        # tick_maintenance only acts on CRYSTALLISED / DISSOLVING, so a chunk
        # revived to FORMING here is correctly left alone for the rest of this
        # trial rather than being re-evaluated against F_high on the strength of
        # the variance that dissolved it.
        formed.extend(self._attempt_reacquisition(variances))
        self.library.tick_maintenance(variances)
        # Growable ceiling LAST, on the fully-updated tally: the returns test
        # reads the outcome this trial just credited. A ceiling raised here
        # takes effect from the NEXT note_outcome, which is deliberate -- the
        # enumeration for this trial has already run, and re-running it at the
        # new size would credit the same outcome to a sequence that was not
        # tracked when the outcome was earned.
        self.accumulator.consider_ceiling_growth()
        # Depth ceiling after the size ceiling, on the same fully-updated state,
        # and in that order deliberately: the structural bound reads the size
        # ceiling, so evaluating depth first would judge it against a stale
        # size budget and refuse a growth the very same trial had just licensed.
        # A depth raised here likewise takes effect from the NEXT note_outcome.
        self.consider_depth_growth()
        return formed

    def _attempt_reacquisition(
        self, variances: Dict[Tuple[int, ...], float]
    ) -> List[ChunkedPrimitive]:
        """MECH-324 rapid reacquisition -- re-form dormant chunks at the low bar.

        The MECH-323 joint formation condition with ONE substitution: the
        repetition requirement is reacquisition_min_repetitions instead of
        R_min. The variance gate (< F_low) and the relative evaluative gate
        (mean > running baseline + margin) are applied UNCHANGED, so a dormant
        chunk still has to be genuinely consistent and genuinely above baseline
        again -- retention lowers the price of coming back, it does not waive
        the evidence.

        Repetitions come from ChunkedPrimitive.reacquisition_repetitions, which
        counts real executions SINCE the last dissolution. Using the
        accumulator's tally-bucket length instead would be a silent no-op: that
        bucket is a sliding window capped at window_trials and a long-lived
        chunk's is saturated, so it clears any reduced bar on the first
        post-dissolution trial.

        Inert (returns []) unless BOTH maintenance and retention are on.
        """
        c = self.config
        if not (c.use_chunk_maintenance and c.use_chunk_dissolution_retention):
            return []
        bar = c.reacquisition_min_repetitions
        baseline = _mean(self.accumulator._outcome_history)
        revived: List[ChunkedPrimitive] = []
        for chunk in self.library.dormant_chunks():
            if chunk.reacquisition_repetitions < bar:
                continue
            var = variances.get(chunk.key)
            if var is None or var >= c.variance_low:
                continue
            mu = _mean(self.accumulator._tally.get(chunk.key, ()))
            if mu <= baseline + c.evaluative_margin:
                continue
            if self.library.revive(chunk.key, value_tag=mu):
                revived.append(chunk)
        return revived

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
        seq = tuple(int(a) for a in sequence)
        for chunk in self.library.all_chunks():
            if chunk.state is ChunkState.DISSOLVED:
                continue
            if _contains_subsequence(seq, chunk.key):
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
        self._depth_ceiling = int(self.config.max_depth)
        self._n_depth_growths = 0
        self._last_depth_gain = 0.0

    def get_state(self) -> dict:
        """Combined diagnostic snapshot for experiment manifests."""
        state = {}
        state.update(self.accumulator.get_state())
        state.update(self.library.get_state())
        state["chunk_n_formation_passes"] = self._n_formation_passes
        # Growable-depth readout (Solway 2014). With the flag off the effective
        # depth stays pinned at max_depth and growths stay 0, which is what
        # makes the OFF arm identifiable in a manifest. The two bounds are
        # reported separately because WHICH ONE binds is the substantive
        # observation: structural < derived means the size ceiling, not the
        # deliberation budget, is what is holding depth back.
        state["chunk_effective_max_depth"] = self.effective_max_depth
        state["chunk_depth_derived_max"] = self.config.derived_chunk_max_depth
        state["chunk_depth_structural_max"] = self.structural_max_depth
        state["chunk_n_depth_growths"] = self._n_depth_growths
        state["chunk_last_depth_gain"] = float(self._last_depth_gain)
        return state
