"""ARC-070 policy_decomposition_via_event_segmenter -- MECH-321 substrate.

WHAT THIS IS
    ARC-070 is the DECOMPOSITION mechanism (zoom in) that is the inverse of
    ARC-071 policy_composition_via_repeated_grounding (zoom out,
    ree_core/policy/policy_chunking.py). Both are children of the ARC-069
    dynamic-regranularisation parent commitment: the unit of policy that the
    rule-apprehension (ARC-062) and diversity-generation (ARC-065) layers
    operate on is itself a dynamic representation. ARC-071 composes primitives
    into chunks on repeated grounding; ARC-070 decomposes a chunk back into
    finer primitives when it fails to predict.

    MECH-321 is ARC-070's first (and so far only) child mechanism: the
    rollout-side consumer of MECH-288 event_segmenter boundary pulses.

THE ASYMMETRY WITH ARC-071 -- do not copy its MECH-094 write-gate shape
    ARC-071 is SLOW, repetition-driven, and EXECUTION-side; its default write
    path (ChunkAccumulator.record_step) REFUSES hypothesis_tag=True content
    outright (a hallucinated chunk would install a macro the agent never
    executed into the pool it can commit to atomically -- MECH-094 strict).

    ARC-070 is FAST, V_s-driven, and SIMULATION-side. It legitimately FIRES
    under hypothesis_tag=True during rollout deliberation -- that is its
    PRIMARY phase (R4 "pre-commitment"), not an exception to be gated against.
    There is nothing to refuse: PolicyDecomposition.evaluate() never writes
    residue, never updates MECH-269 anchor sets, never touches MECH-287
    broadcast -- it is a pure query+decision function. The MECH-094
    hypothesis_tag distinction that matters here is NOT "should this be
    allowed to fire" (both phases fire); it is "may the CALLER let other,
    residue-writing MECH-094-respecting machinery react to this outcome"
    (pre-commit: no; mid-execution: yes, because the trajectory is really
    executing). See claims.yaml MECH-321 functional_restatement R4: "The
    phase distinction is enforced by MECH-094 hypothesis_tag at MECH-288's
    input_stream label, not by code internal to MECH-321." Both phases query
    input_stream="rollout" -- pre-commit because the whole chunk is still
    imagined, mid-execution because the REMAINING (unexecuted) chunk content
    is still, by definition, a prediction rather than an observation.

TRIGGER (R1, conf 0.78; lit-pull targeted_review_arc_070_decomposition)
    PRIMARY = V_s drop on the primitive's region (PE-driven; Zacks 2007 event
    segmentation theory + Schapiro 2017 hippocampal pattern-separation as a
    candidate V_s substrate) OR a MECH-288 boundary firing on the rollout
    stream for this primitive's predicted latent trajectory. Either condition
    alone is sufficient (R1 verdict: OR, not AND -- V_s is the primary/
    cleanest-substrate signal, the boundary detector is the R2 LOAD-BEARING
    shared-substrate consumer; a chunk whose region is confidently predictable
    should not need MECH-288 as a second veto to still ground, but a region
    the agent has never modelled at all may show boundary-worthy PE structure
    before per-stream V_s has had time to update).

R2 LOAD-BEARING -- SHARED SUBSTRATE (conf 0.74)
    ARC-070 is a BIDIRECTIONAL CONSUMER of MECH-288 event_segmenter, not a
    parallel detector. This module never re-implements boundary detection; it
    calls event_segmenter.boundary_on(stream="rollout", ...), which is
    structurally isolated from the observation stream (separate detector
    instances, separate outer/inner counters -- see event_segmenter.py module
    docstring) so a rollout-stream query can never perturb the observation
    stream's calibration.

DEPTH CAP (R3, conf 0.78; multi-level, Badre & D'Esposito 2009 rostro-caudal
    hierarchy, 3-4 levels; Koechlin & Summerfield 2007 cascade)
    A primitive already at or above depth_cap that still triggers is MARKED
    UNRELIABLE rather than decomposed further (downstream selection may
    exclude / abort rather than execute it blind) -- tractability floor, not
    a claim that biology stops at 3-4 levels. decompose_sequence() performs
    ONE level of tiling per call; recursive multi-level descent (evaluating
    each resulting sub-element for further decomposition) is the CALLER's
    responsibility (ree_core/hippocampal/module.py), because each further
    level needs a fresh E2 rollout of that sub-element to get its own
    predicted latent signature -- rollout machinery this module deliberately
    does not own (see "MECH-321 itself reads only the boundary signal" in the
    claims.yaml functional_restatement).

    depth_cap IS NOT A FREE PARAMETER -- it is a DERIVED MIRROR of ARC-071's
    chunk_max_depth, because the depth it tests is that hierarchy's depth (read
    off traj.metadata["chunk_depth"], which ARC-071 cannot mint above
    ChunkLibrary.max_depth). Its useful range is [2, chunk_max_depth]: above
    chunk_max_depth it is INERT, and at 1 it is DEGENERATE (MECH-321 becomes a
    pure withholding mechanism). Both are warned about at the REEAgent wiring
    site via depth_cap_config_issues() below. See the MECH-321 scoping spike
    2026-07-27 section 5a.

    The UPPER end of that range MOVES under ARC-071's use_growable_chunk_depth
    (Solway 2014). chunk_max_depth is then only the STARTING value of a ceiling
    that grows with the agent's deliberation budget, so the deepest mintable
    chunk is PolicyChunkingConfig.derived_chunk_max_depth and the useful range
    is [2, that]. depth_cap_config_issues() takes it as an optional third
    argument for exactly this reason; omitted (the default, and every shipped
    MECH-321 run) the guard is unchanged. The mirror relation itself is
    untouched -- what the cap mirrors is still ARC-071's depth budget; that
    budget simply stopped being a constant.

R5 BOTTLENECK TRIGGER MODE (trigger_mode="bottleneck"; added 2026-07-25)
    The R1 trigger above (V_s drop OR boundary) is the DEFAULT and is what
    ARM_1 of the MECH-321 discriminative validation uses. ARM_2 needs a
    DISTINCT, config-selectable trigger: "MECH-321 ON with bottleneck-state
    primary trigger (R5 alternative) -- chunks decompose only at bottleneck
    states regardless of V_s" (claims.yaml MECH-321 functional_restatement,
    ARM_2). trigger_mode selects which: "vs_boundary" (R1, default) or
    "bottleneck" (R5).

    FAITHFULNESS -- an ONLINE operationalisation of an OFFLINE mechanism.
    R5's verdict (conf 0.74) frames bottleneck detection (McGovern & Barto
    2001 subgoal discovery) as a candidate OFFLINE / CONSOLIDATION-PHASE
    mechanism: examine the ARC-071 ChunkLibrary offline for bottleneck-state
    topology and pre-decompose at bottleneck boundaries. ARM_2 instead needs
    an ONLINE, per-rollout-step primary trigger. This module operationalises
    the R5 signal as an online INCREMENTAL diverse-density accumulator -- the
    SAME statistic McGovern-Barto compute (a region's visitation frequency +
    funnel topology across paths), accumulated incrementally over the live
    rollout stream rather than in a batch offline pass. This is faithful for
    the discriminative-validation purpose because (a) it triggers on the
    frequency/funnel signal, NOT on V_s (R5's "distinct mechanism"); (b) it
    preserves the DEFINING "requires repeated traversals" property via the
    bottleneck_min_visits gate -- which is exactly what generates ARM_2's
    predicted signature (decomposition rare on one-shot tasks, appearing only
    on repeated structures; behaviour closer to ARM_0 than ARM_1 on one-shot
    environments). A trigger that fired instantly would collapse ARM_2 into
    ARM_1. It does NOT claim to be the full offline consolidation mechanism
    (ChunkLibrary topology analysis), which stays DEFERRED per R5.

    REGION KEY -- why NOT segment_id. A bottleneck must be a state that
    RECURS. MECH-288's segment_id (outer.inner) is MONOTONIC -- it only ever
    increments and is never revisited within a stream -- so a frequency
    accumulator keyed on segment_id would see every region exactly once and
    could never register a bottleneck. The key is instead a coarse, fixed
    (no learned projection) quantisation of z_world CONTENT -- the continuous-
    latent analog of McGovern-Barto's recurring tabular states -- which
    recurs when the agent re-visits the same region. boundary_on(stream=
    "rollout", ...) is still called every evaluate() (R2 shared-substrate
    consumer, and it advances the rollout stream), but its result feeds
    only the audit fields in bottleneck mode, not the decision.

    DEFAULT OFF / BIT-IDENTICAL. trigger_mode defaults to "vs_boundary"; the
    bottleneck accumulator is only allocated/updated when trigger_mode ==
    "bottleneck", so the default path is byte-for-byte the pre-2026-07-25
    evaluate(). Both call sites (HippocampalModule pre-commit sweep and the
    agent.py mid-execution beta-gate hook) flow through the same evaluate(),
    so the mode applies uniformly with no call-site change.

WHAT THIS MODULE DOES NOT DEPEND ON
    MECH-321's claims.yaml depends_on is [ARC-070, MECH-288, MECH-269,
    MECH-094] -- it does NOT list MECH-323/MECH-324 (ARC-071's formation /
    maintenance operators). decompose_sequence()'s `library` parameter is
    therefore duck-typed (Any, no import of policy_chunking) against anything
    exposing `.all_chunks()` -> objects with `.sequence` / `.depth` / `.state`
    -- it works with ree_core.policy.policy_chunking.ChunkLibrary when both
    ARC-070 and ARC-071 are enabled together (the only configuration in which
    there is a chunk in the rollout pool to decompose at all), but does not
    import it. With `library=None` (or use_policy_chunking=False), decompose
    falls back to flat single-action tiling -- still a valid, if coarse,
    decomposition.

INTEGRATION
    Default OFF and bit-identical when OFF: REEAgent leaves the attribute
    None (ree_core/agent.py) and HippocampalModule never calls
    boundary_on(stream="rollout", ...) when no decomposition_source is
    registered (ree_core/hippocampal/module.py set_decomposition_source /
    the chunk-candidate pool filtering it gates).

See REE_assembly/docs/architecture/policy_primitive_granularity.md and
evidence/literature/targeted_review_arc_070_decomposition/synthesis.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch


@dataclass
class PolicyDecompositionConfig:
    """Configuration for the ARC-070 / MECH-321 decomposition operator.

    Attributes:
        use_policy_decomposition : master switch. False = disabled (default,
            backward-compatible). REEAgent does not instantiate the operator
            when False.
        vs_decompose_threshold : V_s (region-level, mean of
            HippocampalModule.per_stream_vs -- the same aggregate
            HippocampalModule already computes for anchor-write last_vs)
            below which a primitive is considered unpredictable at its
            current grain (R1). Default 0.4 matches this codebase's existing
            V_s-unreliability convention (vs_gate_e1_threshold /
            vs_gate_e2_threshold both default 0.4).
        depth_cap : R3. Maximum primitive depth eligible for further
            decomposition; a primitive at or above this depth that still
            triggers is marked unreliable instead. Default 3 mirrors
            ARC-071's chunk_max_depth default (R3 suggests 3-4; matching
            the composition side's default keeps the inverse operations
            symmetric).
        trigger_mode : which R-verdict drives should_decompose (R1 vs R5).
            "vs_boundary" (DEFAULT) = the R1 OR trigger
            (v_s < threshold OR boundary.fired) -- the original, and the
            only mode that existed before 2026-07-25. "bottleneck" = the
            R5 alternative used by ARM_2 of the MECH-321 discriminative
            validation: decompose ONLY at bottleneck states, REGARDLESS OF
            V_s. See the "R5 bottleneck trigger mode" note in the class
            docstring for the online-operationalisation faithfulness
            argument. bottleneck_* params below apply only in this mode;
            in "vs_boundary" mode the bottleneck accumulator is never
            allocated or touched, so that mode is byte-for-byte the
            pre-2026-07-25 evaluate() path.
        bottleneck_min_visits : R5. A region must recur at least this many
            times across the run before it is eligible to be a bottleneck.
            This gate IS the "requires repeated traversals" property of
            McGovern & Barto 2001 diverse-density, and is what produces
            ARM_2's predicted signature (decomposition rare on one-shot
            tasks, appearing only on repeated structures). Default 3.
        bottleneck_min_distinct_neighbors : R5. A bottleneck must be a
            FUNNEL -- entered-from / exited-to at least this many DISTINCT
            other regions (the topological signature that distinguishes a
            genuine subgoal bottleneck from a region the agent merely
            loiters in via self-loops). Default 2.
        bottleneck_region_quant : R5. Bin width for the coarse, fixed
            (no learned projection) quantisation of z_world into a region
            key. z_world content -- NOT the monotonic MECH-288 segment_id,
            which never recurs and so could never register a bottleneck --
            is what recurs when the agent re-visits a region. Default 1.0.
        bottleneck_region_dims : R5. How many leading z_world dims enter
            the region key. Default 8 (coarse locality bucket; keeps the
            key low-dimensional so distinct regions still collide into the
            same bucket when they are genuinely nearby). Default 8.
        use_harm_aware_selection : SD-hazard-aware-policy-decomposition
            (V3-EXQ-844 autopsy successor). Master switch for the two-stage
            threat-modulated selection rule over candidate re-tilings
            (targeted_review_threat_modulated_defensive_path_selection
            SYNTHESIS.md Form B). False (default) = every leaf tile produced
            by a decomposition is added to the pool unweighted, exactly the
            pre-existing harm-blind additive recombination -- bit-identical.
        harm_bias_gain : Stage 1 (graded). Gain on the per-leaf harm-penalty
            bias fed to the caller's score_bias composition (mirrors
            InstrumentalAvoidanceGateConfig.action_bias_gain).
        harm_bias_scale : Stage 1. Clamp on the graded bias magnitude, so
            this channel cannot dominate the additive score_bias chain --
            same discipline as every other threat-response bias in this
            codebase (InstrumentalAvoidanceGate, EscapeAffordanceBridge).
        harm_threat_floor, harm_threat_ref : Stage 1/2. z_harm_a-norm linear
            ramp bounds for w(h) in [0, 1] (Cooper 2016's sigmoidal-not-
            linear finding is realised by the gain+clamp saturating the
            ramp, not by the ramp shape itself -- mirrors
            InstrumentalAvoidanceGateConfig.threat_floor/threat_ref exactly).
        harm_override_w_threshold : Stage 2 (categorical). w(h) at/above
            this restricts a withheld chunk's leaf tiles to the single
            lowest-harm-penalty one (Mobbs 2007 / Evans 2018 categorical
            regime shift), overriding the harm-blind additive default.
            Below it, all leaves are kept (Evans 2018 freeze-as-fallback).
    """

    use_policy_decomposition: bool = False
    vs_decompose_threshold: float = 0.4
    depth_cap: int = 3
    trigger_mode: str = "vs_boundary"
    bottleneck_min_visits: int = 3
    bottleneck_min_distinct_neighbors: int = 2
    bottleneck_region_quant: float = 1.0
    bottleneck_region_dims: int = 8
    use_harm_aware_selection: bool = False
    harm_bias_gain: float = 0.1
    harm_bias_scale: float = 0.1
    harm_threat_floor: float = 0.1
    harm_threat_ref: float = 0.5
    harm_override_w_threshold: float = 0.9

    _VALID_TRIGGER_MODES = ("vs_boundary", "bottleneck")

    def validate(self) -> None:
        """Raise ValueError on a configuration that cannot behave as specified."""
        if not (0.0 <= self.vs_decompose_threshold <= 1.0):
            raise ValueError("vs_decompose_threshold must be in [0, 1]")
        if self.depth_cap < 1:
            raise ValueError("depth_cap must be >= 1")
        if self.trigger_mode not in self._VALID_TRIGGER_MODES:
            raise ValueError(
                "trigger_mode must be one of %r (got %r)"
                % (list(self._VALID_TRIGGER_MODES), self.trigger_mode)
            )
        if self.bottleneck_min_visits < 1:
            raise ValueError("bottleneck_min_visits must be >= 1")
        if self.bottleneck_min_distinct_neighbors < 1:
            raise ValueError("bottleneck_min_distinct_neighbors must be >= 1")
        if self.bottleneck_region_quant <= 0.0:
            raise ValueError("bottleneck_region_quant must be > 0")
        if self.bottleneck_region_dims < 1:
            raise ValueError("bottleneck_region_dims must be >= 1")
        if self.harm_bias_gain < 0.0:
            raise ValueError("harm_bias_gain must be >= 0")
        if self.harm_bias_scale < 0.0:
            raise ValueError("harm_bias_scale must be >= 0")
        if self.harm_threat_floor < 0.0:
            raise ValueError("harm_threat_floor must be >= 0")
        if not (0.0 <= self.harm_override_w_threshold <= 1.0):
            raise ValueError("harm_override_w_threshold must be in [0, 1]")


# ----------------------------------------------------------------------
# R3 depth_cap / ARC-071 chunk_max_depth coupling guard
# (MECH-321 scoping spike 2026-07-27 section 5a,
#  REE_assembly/evidence/planning/mech321_decomposition_scale_scoping_spike_2026-07-27.md)
#
# depth_cap is NOT a free scalar. The `depth` it tests is read straight off
# traj.metadata["chunk_depth"] in HippocampalModule._apply_policy_decomposition,
# and ARC-071 cannot mint a chunk above ChunkLibrary.max_depth. So depth_cap is
# a DERIVED MIRROR of chunk_max_depth, and its useful range is
# [2, chunk_max_depth]:
#
#   depth_cap >  chunk_max_depth  -> INERT. No chunk can reach the cap, so the
#       mark-unreliable-by-cap branch in evaluate() is unreachable and
#       _recursive_leaf_tiles' `iterations < depth_cap` bound stops binding.
#       4 and 100 are the same run.
#   depth_cap == 1               -> DEGENERATE. Every chunk has depth >= 1, so
#       every triggering chunk is marked unreliable rather than re-tiled and
#       MECH-321 collapses into a pure WITHHOLDING mechanism (no decomposition
#       ever happens).
#
# Both are silent today: PolicyDecompositionConfig.validate() only checks
# depth_cap >= 1, and it cannot check the coupling because it never sees
# chunk_max_depth. The emit site is therefore REEAgent's wiring block, which
# reads both knobs (ree_core/agent.py). This function is the pure predicate so
# the condition can be contract-pinned without constructing an agent.
#
# Under ARC-071's use_growable_chunk_depth the ceiling in that range is no
# longer chunk_max_depth but derived_chunk_max_depth (the deliberation-budget
# derivation), passed in as derived_max_depth. Without it the guard would warn
# INERT about a depth_cap that the growing ceiling will in fact reach.
#
# WARN, do not raise: >= 3 shipped MECH-321 experiments and an existing
# contract already run depth_cap=4, which is inert-but-harmless. Failing them
# would change the behaviour of currently-valid configurations.
# ----------------------------------------------------------------------
def depth_cap_config_issues(
    depth_cap: int,
    chunk_max_depth: int,
    derived_max_depth: Optional[int] = None,
) -> Tuple[str, ...]:
    """Return ASCII warning messages for an inert or degenerate depth_cap.

    Empty tuple means the pairing is in the useful range. Pure and
    side-effect-free -- REEAgent turns each message into a warnings.warn().

    Args:
        depth_cap : MECH-321's decomposition_depth_cap.
        chunk_max_depth : ARC-071's INITIAL depth budget.
        derived_max_depth : the deepest chunk ARC-071 could ever mint, when
            use_growable_chunk_depth is on -- chunk_max_depth is then only the
            STARTING value of a growable ceiling, so a cap above it is not
            inert, merely not yet reachable. None (the default) means the depth
            ceiling does not grow and chunk_max_depth is the real bound; the
            guard then behaves exactly as it did before growth existed, which
            is what keeps every shipped MECH-321 configuration unchanged. A
            value at or below chunk_max_depth is ignored for the same reason.
    """
    issues: List[str] = []
    cap = int(depth_cap)
    ceiling = int(chunk_max_depth)
    label = "chunk_max_depth"
    if derived_max_depth is not None and int(derived_max_depth) > ceiling:
        ceiling = int(derived_max_depth)
        # Name the bound that actually binds, or the message would blame a knob
        # (chunk_max_depth) that is no longer the ceiling and send the reader to
        # raise something that is already growable.
        label = "derived_chunk_max_depth"

    if cap == 1:
        issues.append(
            "decomposition_depth_cap=1 is DEGENERATE: every ARC-071 chunk has "
            "depth >= 1, so every triggering chunk is marked unreliable rather "
            "than re-tiled and MECH-321 degenerates into a pure withholding "
            "mechanism (decomposition never fires). Use a value in "
            "[2, %s] (%s=%d) to get decomposition." % (label, label, ceiling)
        )
    elif cap > ceiling:
        issues.append(
            "decomposition_depth_cap=%d is INERT: it exceeds ARC-071's "
            "%s=%d, and MECH-321's depth is that hierarchy's "
            "depth, so no chunk can ever reach the cap. The "
            "mark-unreliable-by-cap branch is unreachable and the recursive "
            "leaf-tiling bound stops binding -- %d behaves identically to %d. "
            "depth_cap mirrors that bound; its useful range is "
            "[2, %s]. Raise it if deeper "
            "decomposition is what you want."
            % (cap, label, ceiling, cap, ceiling, label)
        )

    return tuple(issues)


@dataclass
class DecompositionDecision:
    """Outcome of one PolicyDecomposition.evaluate() call on one primitive.

    sub_elements is a tuple of (sequence, depth) pairs -- the ONE-LEVEL tiling
    produced by decompose_sequence() -- populated only when decomposed=True.
    depth=0 in a sub-element means an irreducible raw action (no shallower
    chunk covers that position); depth>=1 means a registered sub-chunk was
    matched.
    """

    should_decompose: bool
    decomposed: bool
    marked_unreliable: bool
    v_s: float
    boundary_fired: bool
    boundary_posterior: float
    depth: int
    hypothesis_tag: bool
    sub_elements: Tuple[Tuple[Tuple[int, ...], int], ...] = ()
    # R5 (trigger_mode="bottleneck") audit fields. In "vs_boundary" mode
    # bottleneck_fired is always False and trigger_mode is "vs_boundary".
    # In "bottleneck" mode, should_decompose == bottleneck_fired (V_s /
    # boundary are still COMPUTED and reported in v_s / boundary_fired for
    # audit -- e.g. "V_s was low yet we did not trigger" -- but they do not
    # drive the decision).
    bottleneck_fired: bool = False
    trigger_mode: str = "vs_boundary"


class PolicyDecomposition:
    """MECH-321 -- rollout-side consumer of MECH-288 boundary pulses.

    Pure decision logic, no learned parameters, no nn.Module inheritance
    (mirrors ChunkAccumulator / ChunkLibrary in policy_chunking.py). Holds no
    per-episode state that needs resetting -- only lifetime diagnostic
    counters, so (like PolicyChunking) it is NOT reset on the agent's
    per-episode reset(); see ree_core/agent.py reset() -- policy_chunking
    uses end_episode(), not reset(), for the same reason (formation /
    decomposition readiness is a property of the whole run, and the manifest
    diagnostics are meant to answer "did this fire at all across the run").

    Diagnostics tracked (get_state(), for experiment manifests):
        _n_evaluated_precommit / _n_evaluated_midexec : evaluate() call counts
            by phase (hypothesis_tag=True / False).
        _n_decomposed_precommit / _n_decomposed_midexec : successful
            decompositions by phase.
        _n_marked_unreliable : should_decompose=True but depth_cap reached or
            the primitive was already irreducible (no sub_elements).
        _n_vs_trigger / _n_boundary_fires : which R1 condition(s) fired
            (both may be true on the same call -- OR trigger).
        _n_boundary_fires_fast / _n_boundary_fires_slow / _n_boundary_cofire :
            MECH-321 scale-resolved boundary diagnostic. Which MECH-288
            SCALE produced the boundary, rather than only that one did.
            Unconditional (no flag): they simply stay 0 for a scale that
            never fires, so the default path is bit-identical. See the
            "SCALE-RESOLVED BOUNDARY DIAGNOSTIC" note below for the
            counting rule and the exact sum identity.
    """

    def __init__(self, config: Optional[PolicyDecompositionConfig] = None) -> None:
        self.config = config if config is not None else PolicyDecompositionConfig()
        self.config.validate()

        self._n_evaluated_precommit: int = 0
        self._n_evaluated_midexec: int = 0
        self._n_decomposed_precommit: int = 0
        self._n_decomposed_midexec: int = 0
        self._n_marked_unreliable: int = 0
        self._n_vs_trigger: int = 0
        self._n_boundary_fires: int = 0
        # SCALE-RESOLVED BOUNDARY DIAGNOSTIC (MECH-321 scoping spike
        # 2026-07-27 section 5b). MECH-288 ships TWO qualitatively
        # heterogeneous scales -- fast (PE-threshold z-score over
        # z_world+z_self, per-tick) and slow (BOCPD-Gaussian over z_goal,
        # hazard 1/40) -- and boundary_on() collapses them to a single
        # fired: bool. These counters recover the scale label that the
        # collapse discards, so "decomp_fired_frac == 1.0" (opaque) can
        # become "fast fires on 100% of ticks, slow on 3%" (a
        # measurement). Unconditional -- no flag -- because a scale that
        # never fires simply leaves its counter at 0.
        #
        # COUNTING RULE. All three count evaluate() CALLS (ticks), not
        # events, so they are commensurate with _n_boundary_fires:
        #   _fast   : the tick emitted >=1 non-slow event, OR a non-slow
        #             detector fired and was withheld by the cross-scale
        #             rule (boundary.suppressed_scales).
        #   _slow   : the tick emitted >=1 slow-scale event.
        #   _cofire : both of the above on the same tick.
        # "fast" means "non-slow", matching EventSegmenter.step()'s own
        # partition (slow_scale_name vs everything else) rather than the
        # literal name "fast", so a renamed or three-scale segmenter still
        # partitions exhaustively.
        #
        # WHY suppressed_scales IS PART OF _fast. A slow fire RESETS inner
        # and suppresses the same-tick fast event, so `events` can never
        # hold both scales at once and an events-only co-fire counter would
        # be a constant 0 -- unable to distinguish the spike's outcome 1
        # ("dissociable") from its outcome 3 ("two detectors, one signal"),
        # which is the whole question the probe exists to answer. Counting
        # the withheld fire makes co-fire observable AND keeps the exact
        # identity that _fast + _slow - _cofire == _n_boundary_fires (a
        # tick whose only fast fire was dropped by min_segment_length is
        # NOT counted -- it emitted nothing and fired nothing downstream).
        self._n_boundary_fires_fast: int = 0
        self._n_boundary_fires_slow: int = 0
        self._n_boundary_cofire: int = 0
        self._last_decomposed_sequence: Tuple[int, ...] = ()

        # R5 bottleneck-mode accumulator (allocated/updated only when
        # trigger_mode == "bottleneck"; empty and untouched otherwise, which
        # is what makes "vs_boundary" mode bit-identical). Keyed on the coarse
        # quantised z_world region code (see the class docstring "REGION KEY").
        self._bottleneck_visits: Dict[Tuple[int, ...], int] = {}
        self._bottleneck_neighbors: Dict[Tuple[int, ...], set] = {}
        self._bottleneck_prev_key: Optional[Tuple[int, ...]] = None
        self._n_bottleneck_fires: int = 0

        # SD-hazard-aware-policy-decomposition diagnostics. Both stay 0 when
        # use_harm_aware_selection is False (the caller never calls harm_bias
        # / select_harm_aware_leaves with a nonzero effect in that case).
        self._n_harm_override_fires: int = 0
        self._n_harm_bias_nonzero: int = 0

    # ------------------------------------------------------------------
    def evaluate(
        self,
        region_vs: float,
        latent_signature: Dict[str, Optional[torch.Tensor]],
        event_segmenter: Any,
        pe_signature: Optional[Dict[str, float]] = None,
        depth: int = 1,
        t: Optional[int] = None,
        hypothesis_tag: bool = True,
        sequence: Sequence[int] = (),
        library: Optional[Any] = None,
    ) -> DecompositionDecision:
        """Evaluate ONE primitive at ONE rollout tick (R1 trigger + R3 cap).

        Directly implements the MECH-321 functional_restatement pseudocode:

            v_s      := region_vs
            boundary := event_segmenter.boundary_on(stream="rollout",
                                                      latent=latent_signature,
                                                      pe=pe_signature, t=t)
            if v_s < vs_decompose_threshold OR boundary.fired:
                if depth < depth_cap:
                    decompose(p)  -- one-level tiling via decompose_sequence
                else:
                    mark p unreliable

        Args:
            region_vs : region-level V_s scalar (caller-supplied; this module
                does not read MECH-269 substrate directly -- see module
                docstring "what this module does not depend on").
            latent_signature : latent_dict passed straight through to
                event_segmenter.boundary_on (stream="rollout").
            event_segmenter : the MECH-288 EventSegmenter instance (or
                anything exposing a matching .boundary_on(); duck-typed, no
                import of ree_core.hippocampal.event_segmenter here to avoid
                a hard substrate coupling beyond the call itself).
            pe_signature : optional external PE dict, passed through.
            depth : the primitive's current composition depth (1 = composed
                directly of raw actions, matching ChunkedPrimitive.depth
                convention in policy_chunking.py).
            t : optional explicit rollout tick; None lets boundary_on
                auto-increment its own per-stream counter (its documented
                ergonomic for a caller with no natural tick index of its own).
            hypothesis_tag : True = pre-commitment (rollout deliberation,
                R4 first phase). False = mid-execution (the REMAINING content
                of an already-committed trajectory, R4 second phase). Purely
                a diagnostics / caller-signalling flag here -- see the module
                docstring's "asymmetry with ARC-071" note for why this method
                has no refusal branch.
            sequence : the primitive's raw action-class sequence (needed only
                to produce sub_elements on a decompose outcome; pass () if
                the caller only wants the should_decompose / marked_unreliable
                signal without materialising sub-elements).
            library : optional chunk-library-like object for
                decompose_sequence's sub-chunk tiling (duck-typed; see
                module docstring).

        Returns:
            DecompositionDecision.
        """
        if hypothesis_tag:
            self._n_evaluated_precommit += 1
        else:
            self._n_evaluated_midexec += 1

        boundary = event_segmenter.boundary_on(
            stream="rollout", latent=latent_signature, pe=pe_signature, t=t
        )
        # V_s / boundary are ALWAYS computed and counted (audit), in both
        # modes -- so the manifest can show "V_s was low yet, in bottleneck
        # mode, we did not trigger on it", the discriminative evidence ARM_2
        # is built on.
        vs_trigger = float(region_vs) < self.config.vs_decompose_threshold
        boundary_fired = bool(boundary.fired)
        if vs_trigger:
            self._n_vs_trigger += 1
        if boundary_fired:
            self._n_boundary_fires += 1
        self._accumulate_boundary_scales(boundary, event_segmenter)

        bottleneck_mode = self.config.trigger_mode == "bottleneck"
        bottleneck_fired = False
        if bottleneck_mode:
            # R5: decompose ONLY at bottleneck states, REGARDLESS OF V_s.
            bottleneck_fired = self._update_and_test_bottleneck(latent_signature)
            if bottleneck_fired:
                self._n_bottleneck_fires += 1
            should_decompose = bool(bottleneck_fired)
        else:
            # R1 (default): the original OR trigger, byte-for-byte.
            should_decompose = bool(vs_trigger or boundary_fired)

        if not should_decompose:
            return DecompositionDecision(
                should_decompose=False,
                decomposed=False,
                marked_unreliable=False,
                v_s=float(region_vs),
                boundary_fired=boundary_fired,
                boundary_posterior=float(boundary.posterior),
                depth=int(depth),
                hypothesis_tag=bool(hypothesis_tag),
                bottleneck_fired=bottleneck_fired,
                trigger_mode=self.config.trigger_mode,
            )

        seq = tuple(int(a) for a in sequence)
        sub_elements = (
            self.decompose_sequence(seq, int(depth), library) if seq else ()
        )

        if int(depth) >= self.config.depth_cap or not sub_elements:
            self._n_marked_unreliable += 1
            return DecompositionDecision(
                should_decompose=True,
                decomposed=False,
                marked_unreliable=True,
                v_s=float(region_vs),
                boundary_fired=boundary_fired,
                boundary_posterior=float(boundary.posterior),
                depth=int(depth),
                hypothesis_tag=bool(hypothesis_tag),
                bottleneck_fired=bottleneck_fired,
                trigger_mode=self.config.trigger_mode,
            )

        if hypothesis_tag:
            self._n_decomposed_precommit += 1
        else:
            self._n_decomposed_midexec += 1
        self._last_decomposed_sequence = seq
        return DecompositionDecision(
            should_decompose=True,
            decomposed=True,
            marked_unreliable=False,
            v_s=float(region_vs),
            boundary_fired=boundary_fired,
            boundary_posterior=float(boundary.posterior),
            depth=int(depth),
            hypothesis_tag=bool(hypothesis_tag),
            sub_elements=sub_elements,
            bottleneck_fired=bottleneck_fired,
            trigger_mode=self.config.trigger_mode,
        )

    # ------------------------------------------------------------------
    def _bottleneck_region_key(
        self, latent_signature: Dict[str, Optional[torch.Tensor]]
    ) -> Optional[Tuple[int, ...]]:
        """Coarse, fixed (no learned projection) quantisation of z_world
        content into a region bucket -- the recurring-region key the R5
        bottleneck accumulator is keyed on (see class docstring "REGION
        KEY" for why NOT segment_id). Returns None when z_world is absent
        (no key => this tick cannot contribute to or fire a bottleneck).
        """
        z = latent_signature.get("z_world") if latent_signature else None
        if z is None:
            return None
        try:
            flat = z.detach().reshape(-1)
            k = min(int(self.config.bottleneck_region_dims), int(flat.numel()))
            if k <= 0:
                return None
            q = float(self.config.bottleneck_region_quant)
            # round(x / q) -> integer bucket per dim; deterministic and
            # dependency-free. torch.round ties-to-even, matched across runs.
            bucket = torch.round(flat[:k] / q).to(torch.int64).tolist()
            return tuple(int(b) for b in bucket)
        except Exception:
            return None

    def _update_and_test_bottleneck(
        self, latent_signature: Dict[str, Optional[torch.Tensor]]
    ) -> bool:
        """Incrementally update the diverse-density accumulator for the
        current region and return whether that region now qualifies as a
        bottleneck. A region is a bottleneck once it has BOTH been visited
        >= bottleneck_min_visits times (the 'repeated traversals' gate --
        McGovern & Barto 2001, and the source of ARM_2's one-shot-rare
        signature) AND been entered-from / exited-to >= bottleneck_min_
        distinct_neighbors distinct other regions (the funnel topology that
        distinguishes a subgoal bottleneck from a loitered-in region).

        V_s is NOT consulted here -- that is the whole point of the R5 mode.
        """
        key = self._bottleneck_region_key(latent_signature)
        if key is None:
            return False
        self._bottleneck_visits[key] = self._bottleneck_visits.get(key, 0) + 1
        neighbors = self._bottleneck_neighbors.setdefault(key, set())
        prev = self._bottleneck_prev_key
        if prev is not None and prev != key:
            neighbors.add(prev)
            # Bidirectional: the transition also makes `key` a neighbour of
            # `prev`, so a funnel is detected symmetrically regardless of
            # traversal direction.
            self._bottleneck_neighbors.setdefault(prev, set()).add(key)
        self._bottleneck_prev_key = key
        return (
            self._bottleneck_visits[key] >= int(self.config.bottleneck_min_visits)
            and len(neighbors) >= int(self.config.bottleneck_min_distinct_neighbors)
        )

    # ------------------------------------------------------------------
    @staticmethod
    def decompose_sequence(
        sequence: Tuple[int, ...],
        depth: int,
        library: Optional[Any],
    ) -> Tuple[Tuple[Tuple[int, ...], int], ...]:
        """One-level tiling of `sequence` into finer sub-elements.

        Greedy longest-match against chunks in `library` (any object exposing
        `.all_chunks()` -> objects with `.sequence` / `.depth` / `.state` --
        duck-typed against policy_chunking.ChunkLibrary, no import
        dependency) whose depth is strictly below `depth` and whose state is
        not DISSOLVED; any position not covered by a matching sub-chunk falls
        back to a single raw action (depth 0 -- irreducible, the base case
        that stops further recursion at the caller). Returns () only when
        `len(sequence) <= 1` (nothing to split -- already atomic).

        `depth <= 1` (the common case -- most ARC-071 chunks form at depth 1,
        composed directly of raw actions) skips the library-lookup phase
        entirely (there is no shallower registered CHUNK type below depth 1)
        but still falls through to the raw-action tiling loop below: a
        depth-1 chunk's most basic decomposition IS unpacking it into its
        individual raw actions, each an irreducible depth-0 tile. Treating
        depth<=1 as a terminal no-tiles case here would make the single most
        common chunk shape (depth 1) permanently un-decomposable -- it could
        only ever be marked_unreliable and dropped, never actually
        re-segmented. This was caught by the activation smoke test
        (SMOKE TEST B): a depth-1 chunk that should_decompose landed as
        marked_unreliable=1, decomposed=0 every time.

        This mirrors the read-only lookup ChunkLibrary._depth_for() already
        performs for the OPPOSITE (composition) direction, applied here to
        materialise the tiles rather than just measure a depth.
        """
        if len(sequence) <= 1:
            return ()

        candidates: List[Tuple[Tuple[int, ...], int]] = []
        if library is not None and depth > 1:
            try:
                for chunk in library.all_chunks():
                    state = getattr(chunk, "state", None)
                    state_value = getattr(state, "value", state)
                    if state_value == "dissolved":
                        continue
                    c_depth = int(getattr(chunk, "depth", 1))
                    if not (0 < c_depth < depth):
                        continue
                    c_seq = tuple(int(a) for a in getattr(chunk, "sequence", ()))
                    if c_seq:
                        candidates.append((c_seq, c_depth))
            except Exception:
                candidates = []
        # Longest sub-chunk first: prefer a longer match over letting it
        # fragment into shorter ones / raw actions.
        candidates.sort(key=lambda kv: len(kv[0]), reverse=True)

        out: List[Tuple[Tuple[int, ...], int]] = []
        i = 0
        n = len(sequence)
        while i < n:
            matched = False
            for c_seq, c_depth in candidates:
                m = len(c_seq)
                if m <= n - i and sequence[i : i + m] == c_seq:
                    out.append((c_seq, c_depth))
                    i += m
                    matched = True
                    break
            if not matched:
                out.append(((sequence[i],), 0))
                i += 1
        return tuple(out)

    # ------------------------------------------------------------------
    def _accumulate_boundary_scales(self, boundary: Any, event_segmenter: Any) -> None:
        """Accumulate the per-scale boundary counters for ONE evaluate() tick.

        Diagnostic only -- reads the BoundaryQueryResult that evaluate()
        already obtained and touches nothing that feeds should_decompose.

        Duck-typed in both directions, because PolicyDecomposition
        deliberately does not import EventSegmenter (module docstring):
        a segmenter with no slow_scale_name attribute falls back to the
        literal "slow", and a boundary object with no .events / no
        .suppressed_scales (the pre-existing stub shape used by several
        contracts) contributes nothing rather than raising. That is what
        keeps this safe to call unconditionally.

        See the counting rule in __init__ for why suppressed fires count
        toward `fast` and why the sum identity holds.
        """
        slow_name = getattr(event_segmenter, "slow_scale_name", "slow")
        events = getattr(boundary, "events", None) or ()
        suppressed = getattr(boundary, "suppressed_scales", None) or ()

        slow_here = any(getattr(e, "scale", None) == slow_name for e in events)
        fast_here = any(getattr(e, "scale", None) != slow_name for e in events)
        # A non-slow detector fire withheld by the cross-scale rule is still
        # a fast-scale fire for co-fire purposes (see __init__).
        if any(name != slow_name for name in suppressed):
            fast_here = True

        if fast_here:
            self._n_boundary_fires_fast += 1
        if slow_here:
            self._n_boundary_fires_slow += 1
        if fast_here and slow_here:
            self._n_boundary_cofire += 1

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Reset all diagnostic counters. NOT called on agent per-episode
        reset() -- see class docstring."""
        self._n_evaluated_precommit = 0
        self._n_evaluated_midexec = 0
        self._n_decomposed_precommit = 0
        self._n_decomposed_midexec = 0
        self._n_marked_unreliable = 0
        self._n_vs_trigger = 0
        self._n_boundary_fires = 0
        self._n_boundary_fires_fast = 0
        self._n_boundary_fires_slow = 0
        self._n_boundary_cofire = 0
        self._last_decomposed_sequence = ()
        self._bottleneck_visits = {}
        self._bottleneck_neighbors = {}
        self._bottleneck_prev_key = None
        self._n_bottleneck_fires = 0
        self._n_harm_override_fires = 0
        self._n_harm_bias_nonzero = 0

    def get_state(self) -> dict:
        """Diagnostic snapshot for experiment manifests."""
        return {
            "decomp_n_evaluated_precommit": self._n_evaluated_precommit,
            "decomp_n_evaluated_midexec": self._n_evaluated_midexec,
            # The substrate-readiness readout: did decomposition fire at all,
            # split by phase (R4).
            "decomp_n_decomposed_precommit": self._n_decomposed_precommit,
            "decomp_n_decomposed_midexec": self._n_decomposed_midexec,
            "decomp_n_marked_unreliable": self._n_marked_unreliable,
            # R1 trigger-source audit (either may co-fire). In "bottleneck"
            # mode these still count V_s / boundary crossings for audit, but
            # they do NOT drive should_decompose -- decomp_n_bottleneck_fires
            # does (see decomp_trigger_mode).
            "decomp_n_vs_trigger": self._n_vs_trigger,
            "decomp_n_boundary_fires": self._n_boundary_fires,
            # MECH-321 scale-resolved boundary diagnostic (spike 5b). WHICH
            # MECH-288 scale fired, not merely that one did. Exact identity:
            #   fires_fast + fires_slow - cofire == n_boundary_fires
            # decomp_n_boundary_fires_slow stays 0 unless the slow BOCPD
            # scale has its z_goal stream on the rollout side, which is what
            # decomposition_scale_resolved_probe switches on -- so on the
            # default path these read (n_boundary_fires, 0, 0).
            "decomp_n_boundary_fires_fast": self._n_boundary_fires_fast,
            "decomp_n_boundary_fires_slow": self._n_boundary_fires_slow,
            "decomp_n_boundary_cofire": self._n_boundary_cofire,
            "decomp_last_decomposed_sequence": list(self._last_decomposed_sequence),
            # R5 bottleneck-mode audit. decomp_trigger_mode names which
            # verdict drove the decisions in this run; the *_bottleneck_*
            # fields are 0 / empty in "vs_boundary" mode (accumulator never
            # touched -- the bit-identity guarantee).
            "decomp_trigger_mode": self.config.trigger_mode,
            "decomp_n_bottleneck_fires": self._n_bottleneck_fires,
            "decomp_n_bottleneck_regions_tracked": len(self._bottleneck_visits),
            # SD-hazard-aware-policy-decomposition. use_harm_aware_selection
            # names whether the two-stage rule is active in this run;
            # n_harm_override_fires is stage 2's categorical restriction
            # (should be 0 whenever use_harm_aware_selection is False --
            # the bit-identity guarantee); n_harm_bias_nonzero is stage 1's
            # graded bias having actually fired at least once for a leaf.
            "decomp_use_harm_aware_selection": bool(self.config.use_harm_aware_selection),
            "decomp_n_harm_override_fires": self._n_harm_override_fires,
            "decomp_n_harm_bias_nonzero": self._n_harm_bias_nonzero,
        }

    # ------------------------------------------------------------------
    # SD-hazard-aware-policy-decomposition (V3-EXQ-844 autopsy successor).
    #
    # WHAT THIS ADDS. _apply_policy_decomposition (hippocampal/module.py) and
    # evaluate()/decompose_sequence() above read only z_self/z_world/z_goal
    # and perform a binary decompose/keep test per withheld chunk -- every
    # leaf tile a decomposition produces is additively recombined into the
    # candidate pool with no ranking among them (V3-EXQ-844 code-verified
    # root cause). The three methods below give the caller a harm-valence-
    # weighted RANKED SELECTION among one chunk's own candidate re-tilings,
    # per the targeted_review_threat_modulated_defensive_path_selection
    # lit-pull (9 entries; Fanselow PIC, Mobbs 2007/2020, Evans 2018, Cooper
    # 2016, Blanchard & Blanchard 1989) Form B recommendation:
    #
    #   Stage 1 (graded, always active): a per-leaf score_bias contribution
    #     -w(h) * harm_penalty(leaf), clamped -- harm_bias() below. The
    #     caller (HippocampalModule._apply_policy_decomposition) tags this
    #     onto each leaf Trajectory's metadata as "decomposition_harm_bias";
    #     REEAgent.select_action's existing additive score_bias chain (the
    #     same chain InstrumentalAvoidanceGate / EscapeAffordanceBridge /
    #     dACC compose into) gathers it in. This keeps ARC-007 value-
    #     flatness intact: PolicyDecomposition never scores a trajectory
    #     itself, it only contributes one more additive term that E3 (the
    #     sole value-supplying authority) folds in like every other
    #     threat-response bias source.
    #
    #   Stage 2 (categorical, threshold-gated): at/above harm_override_
    #     w_threshold, restrict the withheld chunk's OWN leaf tiles to the
    #     single lowest-harm-penalty one -- select_harm_aware_leaves()
    #     below. This is a pool-ADMISSION decision, the same authority
    #     _apply_policy_decomposition already exercises when it excludes a
    #     depth-capped / irreducible candidate rather than offering it
    #     blind, so it does not need (and does not use) an oversized score
    #     bias to "win" -- it simply removes the competing re-tilings for
    #     THIS chunk from the pool, mirroring Mobbs 2007 / Evans 2018's
    #     categorical vmPFC->PAG regime shift.
    #
    # harm_penalty(leaf) itself is NOT computed here -- this module stays
    # duck-typed / dependency-free (module docstring "WHAT THIS MODULE DOES
    # NOT DEPEND ON"). The caller reads it from the residue field's
    # VALENCE_HARM_DISCRIMINATIVE channel (SD-014) on the leaf's OWN
    # predicted world_states, mirroring how HippocampalModule.build_goal_
    # payload already reads VALENCE_WANTING at a z_world location for SD-039
    # -- applied here to each candidate's own rollout instead of the agent's
    # current position.
    #
    # BIT-IDENTICAL WHEN OFF. use_harm_aware_selection defaults False;
    # harm_bias() and select_harm_aware_leaves() are both unconditional
    # early-returns in that case (0.0 / all-leaves-kept respectively), and
    # the caller never even computes harm_penalty when the flag is off (see
    # HippocampalModule._apply_policy_decomposition), so this is a genuine
    # no-op path, not merely a zero-valued one.
    # ------------------------------------------------------------------

    def harm_threat_scale(self, z_harm_a_norm: float) -> float:
        """Linear ramp from 0 at harm_threat_floor to 1 at harm_threat_ref.

        Identical shape to InstrumentalAvoidanceGate.threat_scale /
        EscapeAffordanceBridge.threat_scale -- the z_harm_a-norm-to-[0,1]
        convention already shared across this codebase's PFC threat-response
        modules. Reused rather than re-derived, per biology-before-formal-
        definitions: this is an engineering primitive, not a claim.
        """
        z = float(z_harm_a_norm)
        lo = float(self.config.harm_threat_floor)
        hi = float(self.config.harm_threat_ref)
        if z <= lo:
            return 0.0
        if hi <= lo:
            return 1.0
        return float(max(0.0, min(1.0, (z - lo) / (hi - lo))))

    def harm_bias(self, harm_penalty: float, z_harm_a_norm: float) -> float:
        """Form B stage 1 (graded): w(h) * harm_penalty, clamped to
        harm_bias_scale. Positive = unfavourable (REE lower-is-better score
        convention, matching every other score_bias source in this
        codebase). Returns 0.0 when the flag is off, below harm_threat_floor,
        or harm_penalty <= 0.
        """
        if not self.config.use_harm_aware_selection:
            return 0.0
        w = self.harm_threat_scale(z_harm_a_norm)
        if w <= 0.0:
            return 0.0
        penalty = float(self.config.harm_bias_gain) * w * max(0.0, float(harm_penalty))
        biased = float(max(0.0, min(float(self.config.harm_bias_scale), penalty)))
        if biased > 0.0:
            self._n_harm_bias_nonzero += 1
        return biased

    def select_harm_aware_leaves(
        self,
        leaves_with_penalty: Sequence[Tuple[Any, float]],
        z_harm_a_norm: float,
    ) -> List[Any]:
        """Form B stage 2 (categorical override).

        leaves_with_penalty: (item, harm_penalty) pairs for ONE withheld
        chunk's own candidate re-tilings (item is caller-defined -- the
        caller passes leaf Trajectory objects; this method is duck-typed
        over item and never inspects it).

        Below harm_override_w_threshold: all items are kept, in order --
        the harm-blind additive-recombination default (Evans 2018 freeze-
        as-fallback), unchanged from pre-existing behaviour.

        At/above threshold: only the single lowest-harm-penalty item is
        kept (stable argmin -- first item wins a tie), overriding ordinary
        structural-cost scoring for this chunk's decomposition the way
        Mobbs 2007 / Evans 2018 describe a categorical regime shift.
        """
        items = [item for item, _ in leaves_with_penalty]
        if not self.config.use_harm_aware_selection or not items:
            return items
        w = self.harm_threat_scale(z_harm_a_norm)
        if w < float(self.config.harm_override_w_threshold):
            return items
        best_idx = min(
            range(len(leaves_with_penalty)),
            key=lambda i: leaves_with_penalty[i][1],
        )
        if len(items) > 1:
            self._n_harm_override_fires += 1
        return [items[best_idx]]
