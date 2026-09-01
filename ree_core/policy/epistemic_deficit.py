"""MECH-482: epistemic_deficit_accumulator (SD-102).

Fills the `per_candidate_learning_progress` slot MECH-314c's Phase-2 per-candidate
extension reserved (ree-v3 `c0e0ce8`, 2026-08-08) but left unfilled: "there is no
live per-candidate learning-progress source... that is MECH-482, the
epistemic-deficit accumulator" (see
`ree_core/policy/structured_curiosity.py` and
REE_assembly/docs/architecture/sd_102_epistemic_deficit_accumulator.md).

WHY THIS NEEDS ITS OWN MODULE (not a StructuredCuriosity method): 314a
(novelty) and the 314b Phase-2 per-candidate path are both recomputed FRESH
every tick from the current candidate pool -- no cross-tick memory.
MECH-482's own claim text requires the opposite: "an unresolved, consequential
uncertainty accumulates ACROSS TIME STEPS even as raw novelty and
instantaneous prediction error fall to baseline" and quenches "rather than
only reducing raw uncertainty" on resolution. A per-tick-recomputed signal
cannot rise while its own instantaneous inputs fall -- it needs genuine
persistent state, keyed by WHERE in z_world space the deficit was observed (a
"target"), not by the CEM candidate index (a fresh, arbitrary K-slot identity
every tick).

TWO-PHASE SHAPE (mirrors ResidueField's RBF-center persistence, the existing
precedent for a persistent spatially-indexed accumulator in this codebase):

  UPDATE (post-hoc, once per waking tick): given the REALIZED
    (z_world_prev, action_taken, z_world_now) transition, combine three
    already-live signals into a scalar deficit_input and fold it into the
    nearest persistent target (or allocate a new one). Called from
    REEAgent._update_epistemic_deficit(), which mirrors
    REEAgent._train_e2_world_uncertainty()'s cache/cadence.

  READOUT (pre-hoc, at candidate-scoring time): given THIS tick's K live
    candidate first-step z_world summaries, look up each candidate's nearest
    persistent target (read-only) and return its current deficit as the [K]
    per_candidate_learning_progress vector StructuredCuriosity.
    compute_score_bias consumes.

CANDIDATE INPUTS (conservative subset; see the SD doc for the two inputs and
the multiplicative importance x uncertainty x resolvability x persistence
formula deliberately NOT implemented in this landing):
  1. candidate-specific predictive uncertainty -- E2WorldUncertaintyHead.
     predictive_variance(z0, a) (the SD-063 head 314b's Phase-2 path reads).
  2. persistent prediction error -- ||z_world_now - e2.world_forward(z0, a)||,
     the REALIZED point-forward error at the target (distinct from 314c's
     _lp_ema, which is an EMA of the RATE OF CHANGE of a GLOBAL PE scalar).
  3. predictive-system disagreement -- ||e2.world_forward(z0, a) -
     head.forward(z0, a)[median]||: two INDEPENDENTLY PARAMETERIZED
     predictors (SD-063's own docstring: "shares NO parameters with
     E2WorldForward or the encoder") over the same input. MECH-441's
     ModelDisagreementEnsemble was considered and rejected as this source --
     it is a separate, not-built-by-default claim gated on its own blocked
     falsifier, and using it would inject an undeclared cross-claim
     dependency MECH-482's claims.yaml depends_on does not list.

READINESS GATE (binding, per the corrected ARC-065 gate in
mech314bc_percandidate_extension_staged_2026-08-08.md section 5): the READOUT
consumer (REEAgent._curiosity_per_candidate_learning_progress) REFUSES
(returns None -> Phase-1 broadcast fallback) unless the K-candidate batch read
of head.predictive_variance(...) yields
e2_world_uncertainty_last_pvar_relative_spread > 0 this tick -- absolute
range alone (last_uncertainty_dev_range > 0) is NECESSARY BUT NOT SUFFICIENT
(an untrained head passes it with a LARGER absolute range than a trained
one). A refusal calls mark_vacuous_readout() (self-report) rather than
silently returning zeros. The UPDATE path is NOT gated on this -- it
accumulates unconditionally on every waking tick (mirrors
update_prediction_error's always-on cadence); an untrained head simply
contributes a near-uniform deficit_input, which is harmless bookkeeping.

MECH-094: update() takes simulation_mode (no-op when True, mirrors
StructuredCuriosity.update_prediction_error). No memory/replay write surface
-- this is a waking online read/accumulate, same posture as the SD-063 head
itself (see that module's own MECH-094 section).

Pure-arithmetic, no learned parameters, no nn.Module inheritance -- same
posture as StructuredCuriosity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch


@dataclass
class EpistemicDeficitConfig:
    """MECH-482 epistemic-deficit-accumulator configuration.

    All defaults reproduce the pre-SD-102 substrate exactly: with
    curiosity_learning_progress_source="broadcast" (the REEConfig default),
    REEAgent never instantiates EpistemicDeficitAccumulator at all, so these
    values are inert unless a caller explicitly opts in.

    Attributes:
        max_targets : bounded persistent-target capacity. When a deficit_input
            arrives with no existing target within match_radius and the
            accumulator is already at capacity, the LOWEST-deficit existing
            target is evicted to make room (least-consequential-first,
            mirroring "established irrelevance" fading out of the persistent
            state). Default 16.
        match_radius : L2 distance in z_world space below which a location is
            treated as "the same target" as an existing one. Default 1.0,
            matching ResidueField's default RBF bandwidth (field.py
            `bandwidth: float = 1.0`) so the two spatial accumulators are
            calibrated to the same z_world scale.
        ema_alpha : persistence/decay smoothing applied to a matched target's
            deficit on each UPDATE (matched_deficit = (1-alpha)*old +
            alpha*deficit_input). Rises while deficit_input stays high
            (persistence); decays toward zero as deficit_input drops
            (the substrate-honest proxy for "verified resolution" / "accepted
            irreducibility" quench -- there is no separate resolution-event
            channel in the current substrate to gate on). Default 0.1,
            matching curiosity_lp_ema_alpha's smoothing convention.
        uncertainty_weight : UPDATE combination weight, candidate-specific
            predictive uncertainty. Default 1.0.
        disagreement_weight : UPDATE combination weight, predictive-system
            disagreement. Default 1.0.
        persistent_pe_weight : UPDATE combination weight, persistent
            prediction error. Default 1.0.

    MULTI-TARGET READINESS (substrate_queue sd_epistemic_deficit_multitarget_
    readiness, from the ratified V3-EXQ-964 autopsy). The four fields below
    all default to the pre-readiness behaviour EXACTLY, so an existing
    opted-in config is bit-identical without them:

        match_radius_mode : "absolute" (default -- use match_radius as a
            literal L2 threshold, the pre-readiness behaviour) or "relative"
            (effective radius = match_radius_relative_frac * running_scale,
            where running_scale is the running mean of ||z - running
            centroid|| over UPDATE locations seen so far).

            WHY: match_radius's 1.0 default was justified as "matching
            ResidueField's default RBF bandwidth so the two spatial
            accumulators are calibrated to the same z_world scale". That
            reasoning conflates a SOFT RBF falloff (exp(-d^2/2b^2), which
            still grades every distance inside b) with this module's HARD
            assignment threshold (one bucket inside the radius, no grading).
            Measured on the V3-EXQ-964 config (CausalGridWorldV2, world_dim
            16, 180 ticks): the ENTIRE reachable z_world manifold has max
            pairwise L2 0.41, i.e. less than half the 1.0 default, so every
            location fell inside a single target and n_targets could not
            exceed 1 at ANY episode length. Relative mode removes the
            a-priori scale guess: it self-calibrates as the encoder trains,
            which a fixed absolute threshold in a drifting latent space
            cannot. At the measured scale, frac 0.5 yields an effective
            radius near 0.02 -- the regime where a local dwell keeps one
            target (88% of consecutive steps within) while an excursion
            allocates a new one (only 14% of all pairs within).
        match_radius_relative_frac : the fraction above. Inert unless
            match_radius_mode == "relative". Default 0.5.
        center_update : "replace" (default -- a matched target re-centers
            onto the latest observation, the pre-readiness behaviour) or
            "ema" (center = (1-beta)*center + beta*z).

            WHY: re-centering makes a matched target a RANDOM WALK. Measured
            on the same rollout, mean consecutive-step L2 was 0.0139, so a
            single target migrates 59 x 0.0139 = 0.82 of path length in one
            60-step episode -- twice the 0.41 manifold diameter. It therefore
            absorbs the whole space over time even under a correctly-scaled
            radius. This is a SECOND, independent cause of the n_targets==1
            collapse, and neither the per-episode clear nor a wider window
            addresses it. "ema" anchors a target near where its deficit was
            actually observed while still tracking slow drift.
        center_ema_beta : the beta above. Inert unless center_update ==
            "ema". Default 0.1.
        persist_targets_across_episodes : when True, reset() keeps the
            persistent targets and the running scale (diagnostic counters are
            still cleared, unchanged). Default False = the pre-readiness
            per-episode clear. This is the autopsy implementation_hint's
            "relax the per-episode clear" lever; note it is NECESSARY BUT NOT
            SUFFICIENT on its own -- V3-EXQ-964 reached n_targets==1 after 59
            updates WITHIN a single episode, so the collapse is already
            intra-episode and the two geometry knobs above are what actually
            make a second target reachable.
        readout_mode : "hard_match" (default -- a candidate reads the deficit
            of its nearest target if within the effective radius, else 0.0;
            the pre-readiness behaviour) or "rbf_weighted" (a candidate reads
            a distance-weighted sum over ALL targets,
            sum_t exp(-d_t^2 / (2*bandwidth^2)) * deficit_t / sum_t w_t, with
            bandwidth = the effective match radius).

            WHY: the hard threshold is a STEP function of candidate position,
            so it saturates in BOTH directions and there is no radius that
            reliably avoids both. Measured on the V3-EXQ-964 config: the
            per-tick cross-candidate spread of the e2.world_forward
            predictions is 0.0096, while typical target separation is ~0.05 --
            so the whole 32-candidate cloud sits inside one target's radius
            (every candidate reads the SAME deficit -> constant, the 964
            result) or outside every target's radius (every candidate reads
            0.0 -> also constant). A graded readout is continuous in candidate
            position, so it differentiates whenever the candidates differ AT
            ALL, without depending on a knife-edge radius. This also matches
            the RBF-kernel precedent this module's own docstring cites --
            ResidueField evaluates a distance-weighted sum over its persistent
            centers rather than a nearest-center lookup.
        require_differentiated_readout : when True, readout() REFUSES
            (returns None, self-reports) when the per-candidate vector it
            would return has zero range, i.e. is constant across all K
            candidates. Default False = the pre-readiness behaviour (return
            the constant).

            WHY: StructuredCuriosity applies total = total - weight * lp_vec,
            so a CONSTANT lp_vec provably cannot move an argmax -- it is a
            silent no-op that nonetheless reads downstream as "the mechanism
            fired and did nothing". This is the RUNTIME structural-
            unsatisfiability check the V3-EXQ-964 autopsy's learning #4 asks
            for: assert_no_structurally_unsatisfiable_gate is static over
            pre-registered specs and cannot see a degeneracy that emerges
            from the runtime state reached. Kept default-off because
            subtracting a constant is argmax-inert but NOT bit-identical in
            the absolute score values or in the diagnostic counters.
    """

    max_targets: int = 16
    match_radius: float = 1.0
    ema_alpha: float = 0.1
    uncertainty_weight: float = 1.0
    disagreement_weight: float = 1.0
    persistent_pe_weight: float = 1.0
    # Multi-target readiness -- all four default to pre-readiness behaviour.
    match_radius_mode: str = "absolute"
    match_radius_relative_frac: float = 0.5
    center_update: str = "replace"
    center_ema_beta: float = 0.1
    persist_targets_across_episodes: bool = False
    require_differentiated_readout: bool = False
    readout_mode: str = "hard_match"


class EpistemicDeficitAccumulator:
    """MECH-482: persistent, target-bound model-inadequacy accumulator.

    Each waking tick exposes:

        update(z_world_prev, uncertainty, disagreement, persistent_pe,
               simulation_mode=False) -> None
        readout(candidate_world_summaries) -> Optional[[K] tensor]
        mark_vacuous_readout() -> None
        get_state() -> dict
        reset() -> None

    Diagnostics tracked:
        _n_updates                 : int
        _n_readouts                : int
        _n_vacuous_readouts        : int (READOUT refused by the caller's
            readiness gate; see the module docstring)
        _last_readout_vacuous      : bool
        _last_deficit_input        : float (most recent UPDATE's combined
            scalar, pre-EMA)
        _last_matched_new_target   : bool (did the last UPDATE allocate a
            fresh target rather than matching an existing one)
        _last_n_targets_matched_at_readout : int (of the K candidates read
            out, how many matched an existing persistent target; the rest
            read 0.0 -- "no known deficit yet" for unexplored regions)
    """

    def __init__(self, config: Optional[EpistemicDeficitConfig] = None) -> None:
        self.config = config if config is not None else EpistemicDeficitConfig()
        cfg = self.config
        if cfg.max_targets < 1:
            raise ValueError(
                f"max_targets must be >= 1. Got {cfg.max_targets}."
            )
        if cfg.match_radius <= 0.0:
            raise ValueError(
                f"match_radius must be > 0. Got {cfg.match_radius}."
            )
        if not (0.0 < cfg.ema_alpha <= 1.0):
            raise ValueError(
                f"ema_alpha must be in (0, 1]. Got {cfg.ema_alpha}."
            )
        if cfg.uncertainty_weight < 0.0:
            raise ValueError(
                f"uncertainty_weight must be >= 0. Got {cfg.uncertainty_weight}."
            )
        if cfg.disagreement_weight < 0.0:
            raise ValueError(
                f"disagreement_weight must be >= 0. Got {cfg.disagreement_weight}."
            )
        if cfg.persistent_pe_weight < 0.0:
            raise ValueError(
                "persistent_pe_weight must be >= 0. Got "
                f"{cfg.persistent_pe_weight}."
            )
        if cfg.match_radius_mode not in ("absolute", "relative"):
            raise ValueError(
                "match_radius_mode must be 'absolute' or 'relative'. Got "
                f"{cfg.match_radius_mode!r}."
            )
        if cfg.match_radius_relative_frac <= 0.0:
            raise ValueError(
                "match_radius_relative_frac must be > 0. Got "
                f"{cfg.match_radius_relative_frac}."
            )
        if cfg.center_update not in ("replace", "ema"):
            raise ValueError(
                "center_update must be 'replace' or 'ema'. Got "
                f"{cfg.center_update!r}."
            )
        if not (0.0 < cfg.center_ema_beta <= 1.0):
            raise ValueError(
                f"center_ema_beta must be in (0, 1]. Got {cfg.center_ema_beta}."
            )
        if cfg.readout_mode not in ("hard_match", "rbf_weighted"):
            raise ValueError(
                "readout_mode must be 'hard_match' or 'rbf_weighted'. Got "
                f"{cfg.readout_mode!r}."
            )

        # Persistent targets: list of {"center": Tensor[world_dim], "deficit": float}.
        self._targets: List[dict] = []

        # Running z_world scale estimate, for match_radius_mode == "relative".
        # Count-based (not EMA) so it is unbiased from the very first samples
        # and needs no warmup constant: _scale_mean is the running centroid of
        # UPDATE locations, _scale_dev the running mean of ||z - centroid||.
        self._scale_n: int = 0
        self._scale_mean: Optional[torch.Tensor] = None
        self._scale_dev: float = 0.0
        self._last_effective_match_radius: float = 0.0

        # Diagnostics.
        self._n_updates: int = 0
        self._n_readouts: int = 0
        self._n_vacuous_readouts: int = 0
        self._last_readout_vacuous: bool = False
        self._last_deficit_input: float = 0.0
        self._last_matched_new_target: bool = False
        self._last_n_targets_matched_at_readout: int = 0
        self._last_n_simulation_skips: int = 0
        # Multi-target readiness diagnostics. Recorded ALWAYS (both modes) --
        # last_readout_deficit_range is the per-candidate analogue of
        # StructuredCuriosity.last_lp_dev_range and is exactly the number
        # that decides whether a READOUT can differentiate candidates at all.
        self._last_readout_deficit_range: float = 0.0
        self._last_readout_n_distinct_targets: int = 0
        self._n_undifferentiated_readouts: int = 0
        self._max_n_targets: int = 0

    # ------------------------------------------------------------------
    # UPDATE (post-hoc, realized-transition path)
    # ------------------------------------------------------------------
    def update(
        self,
        z_world_prev: torch.Tensor,
        uncertainty: float,
        disagreement: float,
        persistent_pe: float,
        simulation_mode: bool = False,
    ) -> None:
        """Fold one realized tick's deficit signal into the nearest target.

        Args:
            z_world_prev : [world_dim] (or [1, world_dim]) location the
                deficit is attributed to -- the z_world the realized action
                was taken FROM.
            uncertainty : candidate-specific predictive uncertainty at
                (z_world_prev, action_taken) -- e.g.
                E2WorldUncertaintyHead.predictive_variance(...).
            disagreement : predictive-system disagreement at the same input.
            persistent_pe : realized prediction-error magnitude at the same
                input.
            simulation_mode : MECH-094 gate. True -> no-op (mirrors
                StructuredCuriosity.update_prediction_error).
        """
        if simulation_mode:
            self.skip_simulation_tick()
            return
        cfg = self.config
        z = z_world_prev.detach().reshape(-1)

        deficit_input = (
            cfg.uncertainty_weight * float(uncertainty)
            + cfg.disagreement_weight * float(disagreement)
            + cfg.persistent_pe_weight * float(persistent_pe)
        )
        self._last_deficit_input = deficit_input
        self._n_updates += 1

        self._observe_scale(z)
        radius = self._effective_match_radius()
        self._last_effective_match_radius = radius

        idx, dist = self._nearest_target(z)
        if idx is not None and dist <= radius:
            self._last_matched_new_target = False
            old = self._targets[idx]["deficit"]
            self._targets[idx]["deficit"] = (
                (1.0 - cfg.ema_alpha) * old + cfg.ema_alpha * deficit_input
            )
            # Move the target's center toward the latest observation so a
            # slowly-drifting target (e.g. a region the agent revisits at
            # slightly different poses) does not accumulate positional error.
            # beta == 1.0 ("replace") is the pre-readiness behaviour and makes
            # the target a random walk -- see EpistemicDeficitConfig's
            # center_update docs for why that alone collapses n_targets to 1.
            beta = 1.0 if cfg.center_update == "replace" else cfg.center_ema_beta
            if beta >= 1.0:
                self._targets[idx]["center"] = z.clone()
            else:
                old_center = self._targets[idx]["center"].to(
                    dtype=z.dtype, device=z.device
                )
                self._targets[idx]["center"] = (
                    (1.0 - beta) * old_center + beta * z
                ).clone()
        else:
            self._last_matched_new_target = True
            if len(self._targets) >= cfg.max_targets:
                # Evict the lowest-deficit target (least consequential /
                # most resolved) to make room.
                evict_idx = min(
                    range(len(self._targets)),
                    key=lambda i: self._targets[i]["deficit"],
                )
                self._targets.pop(evict_idx)
            self._targets.append({"center": z.clone(), "deficit": deficit_input})
        if len(self._targets) > self._max_n_targets:
            self._max_n_targets = len(self._targets)

    # ------------------------------------------------------------------
    # Running z_world scale (match_radius_mode == "relative")
    # ------------------------------------------------------------------
    def _observe_scale(self, z: torch.Tensor) -> None:
        """Fold one UPDATE location into the running centroid + mean deviation.

        Count-based rather than EMA so the estimate is unbiased from the first
        samples and introduces no warmup constant. Maintained unconditionally
        (cheap, and it makes the measured z_world scale observable in
        get_state() even in "absolute" mode, where the 964 collapse was
        invisible precisely because nothing recorded it).
        """
        z = z.detach()
        if self._scale_mean is None:
            self._scale_n = 1
            self._scale_mean = z.clone()
            self._scale_dev = 0.0
            return
        mean = self._scale_mean.to(dtype=z.dtype, device=z.device)
        dev = float((z - mean).norm().item())
        self._scale_n += 1
        n = float(self._scale_n)
        # Running mean of the deviation, then move the centroid. Order matters
        # only for the first few samples and either is defensible; measuring
        # against the PRE-update centroid keeps the deviation an honest
        # out-of-sample distance rather than one shrunk by its own inclusion.
        self._scale_dev = self._scale_dev + (dev - self._scale_dev) / n
        self._scale_mean = mean + (z - mean) / n

    def _effective_match_radius(self) -> float:
        """The L2 threshold this UPDATE/READOUT actually uses.

        "absolute" -> config.match_radius verbatim (pre-readiness behaviour).
        "relative" -> match_radius_relative_frac * running z_world scale.
        """
        cfg = self.config
        if cfg.match_radius_mode == "absolute":
            return float(cfg.match_radius)
        return float(cfg.match_radius_relative_frac) * float(self._scale_dev)

    def _nearest_target(
        self, z: torch.Tensor
    ) -> "tuple[Optional[int], float]":
        """Index and L2 distance of the nearest existing target to z.

        Returns (None, inf) when no targets exist yet.
        """
        if len(self._targets) == 0:
            return None, float("inf")
        best_idx = None
        best_dist = float("inf")
        for i, t in enumerate(self._targets):
            center = t["center"].to(dtype=z.dtype, device=z.device)
            d = float((z - center).norm().item())
            if d < best_dist:
                best_dist = d
                best_idx = i
        return best_idx, best_dist

    # ------------------------------------------------------------------
    # READOUT (pre-hoc, candidate-scoring path)
    # ------------------------------------------------------------------
    def readout(
        self, candidate_world_summaries: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """Per-candidate persistent deficit [K], read-only against current targets.

        Candidates that match no existing target within match_radius read
        0.0 (no known deficit yet for an unexplored region -- the same
        "nothing accumulated here" semantics 314a novelty has when there are
        no active residue centers). Returns None when no targets exist at
        all yet (nothing to read out; caller should treat this the same as
        the Phase-1 broadcast fallback).
        """
        self._n_readouts += 1
        K = int(candidate_world_summaries.shape[0])
        if len(self._targets) == 0 or K == 0:
            self._last_n_targets_matched_at_readout = 0
            self._last_readout_deficit_range = 0.0
            self._last_readout_n_distinct_targets = 0
            return None

        device = candidate_world_summaries.device
        dtype = candidate_world_summaries.dtype
        centers = torch.stack(
            [t["center"].to(device=device, dtype=dtype) for t in self._targets],
            dim=0,
        )  # [N, world_dim]
        deficits = torch.tensor(
            [t["deficit"] for t in self._targets], device=device, dtype=dtype
        )  # [N]

        # [K, N] pairwise distances.
        diffs = candidate_world_summaries.unsqueeze(1) - centers.unsqueeze(0)
        dists = diffs.norm(dim=-1)
        nearest_dist, nearest_idx = dists.min(dim=-1)  # [K], [K]

        radius = self._effective_match_radius()
        self._last_effective_match_radius = radius
        matched = nearest_dist <= radius
        if self.config.readout_mode == "rbf_weighted":
            # Graded, distance-weighted read over ALL targets (the
            # ResidueField RBF precedent). Continuous in candidate position,
            # so it differentiates whenever the candidates differ at all --
            # see EpistemicDeficitConfig.readout_mode for why the hard
            # threshold saturates in both directions instead.
            bandwidth = radius if radius > 0.0 else 1.0
            w = torch.exp(-(dists ** 2) / (2.0 * bandwidth * bandwidth))  # [K, N]
            denom = w.sum(dim=-1)
            safe = denom > 0.0
            out = torch.where(
                safe,
                (w * deficits.unsqueeze(0)).sum(dim=-1) / denom.clamp(min=1e-12),
                torch.zeros(K, device=device, dtype=dtype),
            )
        else:
            out = torch.where(
                matched,
                deficits[nearest_idx],
                torch.zeros(K, device=device, dtype=dtype),
            )
        self._last_n_targets_matched_at_readout = int(matched.sum().item())
        # Differentiation diagnostics -- ALWAYS recorded, in both modes. The
        # V3-EXQ-964 autopsy's learning #1: a readiness gate for "can this
        # read change selection" must require per-candidate DIFFERENTIATION,
        # not mere non-emptiness. n_distinct counts distinct MATCHED targets;
        # deficit_range is the max-min of the vector actually returned (which
        # is what StructuredCuriosity turns into lp_contrib).
        self._last_readout_deficit_range = float(
            (out.max() - out.min()).item()
        )
        self._last_readout_n_distinct_targets = int(
            torch.unique(nearest_idx[matched]).numel()
        ) if bool(matched.any().item()) else 0
        if (
            self.config.require_differentiated_readout
            and not (self._last_readout_deficit_range > 0.0)
        ):
            # Provably argmax-inert (subtracting a constant from every
            # candidate score cannot move an argmax), so returning it would be
            # a silent no-op indistinguishable downstream from a live effect.
            # Refuse instead, and self-report: counted BOTH as vacuous (so the
            # existing vacuous_readout_rate instrument sees it) and separately
            # as undifferentiated (so the two causes stay tellable apart).
            self._n_undifferentiated_readouts += 1
            self._n_vacuous_readouts += 1
            self._last_readout_vacuous = True
            return None
        self._last_readout_vacuous = False
        return out

    def skip_simulation_tick(self) -> None:
        """MECH-094: record one simulated/imagined tick UPDATE was skipped for.

        Called by update(simulation_mode=True) directly, and available for a
        caller to call standalone (e.g. when it short-circuits before
        building the args update() needs, but still wants the tick counted)
        so no dummy tensor has to be constructed just to reach the
        simulation_mode branch.
        """
        self._last_n_simulation_skips += 1

    def mark_vacuous_readout(self) -> None:
        """Self-report: the caller refused a READOUT on the readiness gate.

        Called by REEAgent._curiosity_per_candidate_learning_progress when
        e2_world_uncertainty_last_pvar_relative_spread is not > 0 this tick,
        so the substrate does not silently look identical to "no deficit
        anywhere" -- it is instead observable in get_state() as a refused
        (not a zero) read.
        """
        self._n_vacuous_readouts += 1
        self._last_readout_vacuous = True

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Clear all persistent targets (per-episode reset).

        Mirrors StructuredCuriosity.reset()'s per-episode 314c LP-EMA clear
        ("a fresh task / environment can have a fresh learning curve") --
        MECH-482 is architecturally 314c's genuine source, so it inherits the
        same episode-scoping convention. Diagnostic counters ARE cleared too,
        matching StructuredCuriosity.reset()'s own convention (diagnostics
        are per-episode snapshots there, not lifetime totals).

        MULTI-TARGET READINESS: when persist_targets_across_episodes is True,
        the persistent targets AND the running z_world scale survive the
        episode boundary; the diagnostic counters are still cleared, so the
        per-episode-snapshot convention above is unchanged either way. See
        EpistemicDeficitConfig for why this lever is necessary but not
        sufficient on its own.
        """
        if not self.config.persist_targets_across_episodes:
            self._targets = []
            self._scale_n = 0
            self._scale_mean = None
            self._scale_dev = 0.0
        self._max_n_targets = len(self._targets)
        self._n_updates = 0
        self._n_readouts = 0
        self._n_vacuous_readouts = 0
        self._last_readout_vacuous = False
        self._last_deficit_input = 0.0
        self._last_matched_new_target = False
        self._last_n_targets_matched_at_readout = 0
        self._last_n_simulation_skips = 0
        self._last_readout_deficit_range = 0.0
        self._last_readout_n_distinct_targets = 0
        self._n_undifferentiated_readouts = 0
        self._last_effective_match_radius = 0.0

    def get_state(self) -> dict:
        """Diagnostic snapshot for experiment manifests."""
        deficits = [t["deficit"] for t in self._targets]
        return {
            "n_targets": len(self._targets),
            "mean_deficit": (sum(deficits) / len(deficits)) if deficits else 0.0,
            "max_deficit": max(deficits) if deficits else 0.0,
            "n_updates": self._n_updates,
            "n_readouts": self._n_readouts,
            "n_vacuous_readouts": self._n_vacuous_readouts,
            "last_readout_vacuous": self._last_readout_vacuous,
            "last_deficit_input": self._last_deficit_input,
            "last_matched_new_target": self._last_matched_new_target,
            "last_n_targets_matched_at_readout": self._last_n_targets_matched_at_readout,
            "last_n_simulation_skips": self._last_n_simulation_skips,
            # Multi-target readiness (recorded in BOTH modes -- the 964
            # collapse was invisible precisely because nothing recorded the
            # geometry). last_readout_deficit_range is the per-candidate
            # differentiation number: > 0 is the necessary condition for the
            # read to be able to move an argmax at all.
            "last_readout_deficit_range": self._last_readout_deficit_range,
            "last_readout_n_distinct_targets": self._last_readout_n_distinct_targets,
            "n_undifferentiated_readouts": self._n_undifferentiated_readouts,
            "max_n_targets": self._max_n_targets,
            "last_effective_match_radius": self._last_effective_match_radius,
            "zworld_scale_dev": self._scale_dev,
            "zworld_scale_n": self._scale_n,
            "readout_mode": self.config.readout_mode,
        }
