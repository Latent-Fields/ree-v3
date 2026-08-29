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
    """

    max_targets: int = 16
    match_radius: float = 1.0
    ema_alpha: float = 0.1
    uncertainty_weight: float = 1.0
    disagreement_weight: float = 1.0
    persistent_pe_weight: float = 1.0


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

        # Persistent targets: list of {"center": Tensor[world_dim], "deficit": float}.
        self._targets: List[dict] = []

        # Diagnostics.
        self._n_updates: int = 0
        self._n_readouts: int = 0
        self._n_vacuous_readouts: int = 0
        self._last_readout_vacuous: bool = False
        self._last_deficit_input: float = 0.0
        self._last_matched_new_target: bool = False
        self._last_n_targets_matched_at_readout: int = 0
        self._last_n_simulation_skips: int = 0

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

        idx, dist = self._nearest_target(z)
        if idx is not None and dist <= cfg.match_radius:
            self._last_matched_new_target = False
            old = self._targets[idx]["deficit"]
            self._targets[idx]["deficit"] = (
                (1.0 - cfg.ema_alpha) * old + cfg.ema_alpha * deficit_input
            )
            # Re-center on the latest observation so a slowly-drifting target
            # (e.g. a region the agent revisits at slightly different poses)
            # does not accumulate positional error across many updates.
            self._targets[idx]["center"] = z.clone()
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

        matched = nearest_dist <= self.config.match_radius
        out = torch.where(
            matched, deficits[nearest_idx], torch.zeros(K, device=device, dtype=dtype)
        )
        self._last_n_targets_matched_at_readout = int(matched.sum().item())
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
        """
        self._targets = []
        self._n_updates = 0
        self._n_readouts = 0
        self._n_vacuous_readouts = 0
        self._last_readout_vacuous = False
        self._last_deficit_input = 0.0
        self._last_matched_new_target = False
        self._last_n_targets_matched_at_readout = 0
        self._last_n_simulation_skips = 0

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
        }
