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
    """

    use_policy_decomposition: bool = False
    vs_decompose_threshold: float = 0.4
    depth_cap: int = 3

    def validate(self) -> None:
        """Raise ValueError on a configuration that cannot behave as specified."""
        if not (0.0 <= self.vs_decompose_threshold <= 1.0):
            raise ValueError("vs_decompose_threshold must be in [0, 1]")
        if self.depth_cap < 1:
            raise ValueError("depth_cap must be >= 1")


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
        self._last_decomposed_sequence: Tuple[int, ...] = ()

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
        vs_trigger = float(region_vs) < self.config.vs_decompose_threshold
        boundary_fired = bool(boundary.fired)
        if vs_trigger:
            self._n_vs_trigger += 1
        if boundary_fired:
            self._n_boundary_fires += 1

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
        self._last_decomposed_sequence = ()

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
            # R1 trigger-source audit (either may co-fire).
            "decomp_n_vs_trigger": self._n_vs_trigger,
            "decomp_n_boundary_fires": self._n_boundary_fires,
            "decomp_last_decomposed_sequence": list(self._last_decomposed_sequence),
        }
