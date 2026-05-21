"""
HippocampalModule — V3 Implementation

V3 changes (SD-004, Q-020 ARC-007 strict):

SD-004 — Action-object space navigation:
  HippocampalModule now navigates ACTION-OBJECT space O, not raw z_world.
  E2.action_object(z_world, a) produces o_t — a compressed representation
  of the world-effect of action a from z_world. The CEM search operates
  over action-object proposals, not raw action sequences.

  This extends the effective planning horizon because:
  a) action-object space is lower-dimensional than z_world
  b) action objects are semantically grounded (encode world-effects)
  c) the hippocampal map is indexed over world-effects, not states

Q-020 — ARC-007 STRICT (decided 2026-03-16):
  HippocampalModule generates VALUE-FLAT proposals.
  Terrain sensitivity is a CONSEQUENCE of navigating a residue-shaped
  z_world, not an independent hippocampal value computation.
  No separate value head. Residue field curvature in z_world IS the terrain.
  E3 introduces all weighting via Φ_R(ζ) in J(ζ).

SD-002 (preserved from V2): E1 world-domain prior wired into terrain search.
  terrain_prior: (z_world, e1_prior, residue_val) → action_object_mean

MECH-092 (replay):
  replay() is called during quiescent E3 heartbeat cycles. Carries
  hypothesis_tag=True — replay cannot produce residue (MECH-094).
"""

from typing import Any, Dict, List, Optional, Tuple
import math
import random

import torch
import torch.nn as nn

from ree_core.utils.config import HippocampalConfig
from ree_core.predictors.e2_fast import E2FastPredictor, Trajectory
from ree_core.residue.field import ResidueField, VALENCE_WANTING
from ree_core.hippocampal.event_segmenter import (
    BoundaryEvent,
    EventSegmenter,
    Scale as EventSegmenterScale,
)
from ree_core.regulators.invalidation_trigger import (
    BroadcastEvent,
    InvalidationTrigger,
)
from ree_core.hippocampal.anchor_set import AnchorGoalPayload, AnchorSet
from ree_core.hippocampal.ghost_goal_bank import (
    GhostGoalBank,
    GhostGoalBankEntry,
    PersistenceAppraisal,
)
from ree_core.hippocampal.staleness_accumulator import StalenessAccumulator


class HippocampalModule(nn.Module):
    """
    HippocampalModule — action-object space trajectory proposal.

    V3 role:
    - Uses E2.action_object() to build proposals in action-object space O
    - CEM refinement over action-object means (not raw z_world or z_self)
    - Residue field scores terrain; proposals naturally avoid high-residue regions
    - No independent value computation (ARC-007 strict, Q-020 resolved)

    SD-002 preserved: E1 world-domain prior conditions initial terrain proposal.

    replay() method for MECH-092 SWR-equivalent consolidation.
    """

    def __init__(
        self,
        config: HippocampalConfig,
        e2: E2FastPredictor,
        residue_field: ResidueField,
    ):
        super().__init__()
        self.config = config
        self.e2 = e2
        self.residue_field = residue_field

        # SD-004 + SD-002: terrain prior maps (z_world, e1_prior, residue_val[, benefit_val])
        # → action_object_mean (in action-object space O).
        # Output: action_object_dim * horizon (flattened)
        # Input: 2*world_dim + 1 (harm only) or 2*world_dim + 2 (harm + benefit, ARC-030)
        # Benefit channel added when residue_field.benefit_terrain_enabled=True.
        # Wanting/liking distinction (MECH-117): benefit terrain = liking (hippocampal);
        # z_goal = wanting (frontal). Both are separate inputs to trajectory scoring.
        self._benefit_terrain = getattr(residue_field, "benefit_terrain_enabled", False)
        terrain_input_dim = config.world_dim * 2 + 1 + (1 if self._benefit_terrain else 0)
        self.terrain_prior = nn.Sequential(
            nn.Linear(terrain_input_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.action_object_dim * config.horizon),
        )

        # Action-object → action decoder: used to generate actual action sequences
        # from action-object proposals (needed for E2 rollout).
        # Maps o_t back to an action a_t.
        self.action_object_decoder = nn.Sequential(
            nn.Linear(config.action_object_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.action_dim),
        )

        # ARC-028 / MECH-105: last completion signal cache
        self._last_completion_signal: float = 0.0

        # MECH-290: backward trajectory credit sweep (Foster & Wilson 2006).
        # Stores the most recently committed trajectory for backward_credit_sweep().
        # Set by record_committed_trajectory() at BetaGate elevation (commit entry)
        # and cleared by reset_committed_trajectory() on episode boundaries.
        # Distinct from _exploration_buffer (MECH-165 quiescent replay source):
        # this stores what was EXECUTED, not what was considered by CEM.
        self._committed_trajectory_buffer: Optional[Trajectory] = None

        # MECH-165: exploration trajectory buffer for diverse replay
        self._exploration_buffer: List[Trajectory] = []
        self._exploration_buffer_maxlen: int = getattr(config, "exploration_buffer_len", 50)
        self._reverse_fraction: float = getattr(config, "reverse_replay_fraction", 0.3)
        self._random_fraction: float = getattr(config, "random_replay_fraction", 0.2)

        # MECH-267 diagnostics: last operating_mode passed in and the noise
        # scale that was applied. Both None when mode conditioning is disabled
        # or operating_mode is not supplied.
        self._last_operating_mode: Optional[Dict[str, float]] = None
        self._last_mode_noise_scale: Optional[float] = None

        # MECH-269 base substrate (Phase 1, 2026-04-22): per-stream
        # verisimilitude V_s scores. Populated when config.use_per_stream_vs
        # is True. Phase 1 uses identity-prediction proxy across all streams
        # (V_s = 1 - norm(Delta z_s) / scale, EMA-smoothed). Phase 2 will
        # route z_world through ReafferencePredictor (SD-007) and z_harm_s
        # through HarmForwardModel (SD-011). Phase 1 populates the dict as
        # an OBSERVABLE for downstream MECH-287 / MECH-284 wiring; the
        # signal-quality refinement is a Phase 2 concern.
        self.per_stream_vs: Dict[str, float] = {}
        self._prev_stream_values: Dict[str, torch.Tensor] = {}

        # MECH-269 Phase 2 (iii, T4, 2026-04-22): per-region per-stream V_s.
        # Keyed on (scale, segment_id) drawn from AnchorSet active_anchors().
        # Each region maintains its own prev-value cache so identity proxy
        # state evolves independently per region (C3 cross-region
        # isolation). Regions whose active anchor disappears (hysteresis
        # mark_inactive or MECH-287 broadcast reset) are pruned from both
        # dicts. Orthogonal to per_stream_vs: both can be populated
        # simultaneously when use_per_stream_vs and use_per_region_vs are
        # both True.
        self.per_region_vs: Dict[Tuple[str, str], Dict[str, float]] = {}
        self._prev_region_stream_values: Dict[
            Tuple[str, str], Dict[str, torch.Tensor]
        ] = {}

        # MECH-288 (Phase 2 of V_s invalidation runtime): hierarchical event
        # segmenter. Instantiated when config.use_event_segmenter is True.
        # BoundaryEvents emitted per tick are appended to
        # self._boundary_event_queue and consumers (MECH-287 broadcast trigger,
        # MECH-269 anchor-reset) drain via drain_boundary_events().
        self.event_segmenter: Optional[EventSegmenter] = None
        self._boundary_event_queue: List[BoundaryEvent] = []
        if getattr(config, "use_event_segmenter", False):
            seg_cfg = getattr(config, "event_segmenter", None)
            if seg_cfg is None:
                raise ValueError(
                    "HippocampalConfig.use_event_segmenter=True but "
                    "event_segmenter config is None"
                )
            scales = [
                EventSegmenterScale(
                    name=sc.name,
                    streams=tuple(sc.streams),
                    algorithm=sc.algorithm,
                    tau=sc.tau,
                    min_segment_length=sc.min_segment_length,
                    pe_threshold=sc.pe_threshold,
                    window_length=sc.pe_window_length,
                    hazard=sc.hazard,
                    posterior_threshold=sc.posterior_threshold,
                    top_k=sc.bocpd_top_k,
                    prior_var=sc.bocpd_prior_var,
                )
                for sc in seg_cfg.scales
            ]
            self.event_segmenter = EventSegmenter(
                scales=scales,
                emit_to=list(seg_cfg.emit_to),
                scale_id_format=seg_cfg.scale_id_format,
                slow_scale_name=seg_cfg.slow_scale_name,
            )

        # MECH-287 (Phase 2 iv of V_s invalidation runtime): broadcast
        # invalidation trigger. Subscribes to the MECH-288 BoundaryEvents
        # emitted during the same agent.sense() tick and re-emits them as
        # graded BroadcastEvent objects (strength = posterior * gain). The
        # broadcast queue is drained by downstream MECH-269 anchor-reset
        # (T3) / MECH-284 staleness accumulator consumers.
        #
        # Verdict-3 architectural commitment: no independent comparator
        # inside the trigger. If use_event_segmenter is False (no
        # BoundaryEvents produced), the trigger will tick silently even
        # when enabled -- this is the dissociation test (C5).
        self.invalidation_trigger: Optional[InvalidationTrigger] = None
        self._broadcast_event_queue: List[BroadcastEvent] = []
        if getattr(config, "use_invalidation_trigger", False):
            trig_cfg = getattr(config, "invalidation_trigger", None)
            if trig_cfg is None:
                raise ValueError(
                    "HippocampalConfig.use_invalidation_trigger=True but "
                    "invalidation_trigger config is None"
                )
            self.invalidation_trigger = InvalidationTrigger(trig_cfg)

        # MECH-269 Phase 2 (ii): scale-tagged anchor set. Consumes MECH-288
        # BoundaryEvents (drained via drain_boundary_events()) and installs
        # / remaps anchors with dual-trace preservation (Bouton 2004) and
        # k-consecutive hysteresis on V_s_anchor crossings. Phase 2 stand-in
        # stream_mixture = tuple(sorted(per_stream_vs.keys())) at write-time;
        # learned attribution head deferred to Phase 3.
        self.anchor_set: Optional[AnchorSet] = None
        if getattr(config, "use_anchor_sets", False):
            anchor_cfg = getattr(config, "anchor_set", None)
            if anchor_cfg is None:
                raise ValueError(
                    "HippocampalConfig.use_anchor_sets=True but "
                    "anchor_set config is None"
                )
            self.anchor_set = AnchorSet(anchor_cfg)

        # MECH-284 Phase 3: region-indexed V_s residual staleness accumulator.
        # Consumes MECH-287 BroadcastEvents and credits them across the
        # currently-active anchor set with an attribution_weight. Keyed on
        # (scale, segment_id) to match the per_region_vs readout partition.
        # When use_mech284_hysteresis is additionally True, AnchorSet
        # tick_hysteresis() is given a staleness_lookup callable that reads
        # from this accumulator instead of the internal
        # (tick - last_accessed) * staleness_rate proxy.
        self.staleness_accumulator: Optional[StalenessAccumulator] = None
        if getattr(config, "use_staleness_accumulator", False):
            sa_cfg = getattr(config, "staleness_accumulator", None)
            if sa_cfg is None:
                raise ValueError(
                    "HippocampalConfig.use_staleness_accumulator=True but "
                    "staleness_accumulator config is None"
                )
            self.staleness_accumulator = StalenessAccumulator(sa_cfg)

        # MECH-292: ranked ghost-goal bank (derived view over the SD-039
        # dual-trace anchor pool). Pure-arithmetic, non-trainable;
        # consumes payloads written by the SD-039 population layer.
        # Preconditions: anchor_set exists AND AnchorSetConfig has
        # use_sd039_anchor_payload=True (otherwise every anchor scores
        # goal_match=0.0 and the bank degenerates to empty -- raise
        # rather than silently produce a dead view).
        self.ghost_goal_bank: Optional[GhostGoalBank] = None
        if getattr(config, "use_mech292_ghost_bank", False):
            if self.anchor_set is None:
                raise ValueError(
                    "HippocampalConfig.use_mech292_ghost_bank=True but "
                    "use_anchor_sets=False; MECH-292 requires the MECH-269 "
                    "Phase 2 (ii) anchor substrate to read from."
                )
            anchor_cfg = getattr(config, "anchor_set", None)
            if anchor_cfg is None or not getattr(
                anchor_cfg, "use_sd039_anchor_payload", False
            ):
                raise ValueError(
                    "HippocampalConfig.use_mech292_ghost_bank=True but "
                    "AnchorSetConfig.use_sd039_anchor_payload=False; "
                    "MECH-292 requires SD-039 payload population to score "
                    "goal_match (otherwise every anchor scores 0.0)."
                )
            bank_cfg = getattr(config, "ghost_goal_bank_config", None)
            if bank_cfg is None:
                raise ValueError(
                    "HippocampalConfig.use_mech292_ghost_bank=True but "
                    "ghost_goal_bank_config is None"
                )
            self.ghost_goal_bank = GhostGoalBank(
                config=bank_cfg,
                anchor_set=self.anchor_set,
                staleness_accumulator=self.staleness_accumulator,
            )

        # MECH-293: waking ghost-goal probe search precondition. Consumes
        # the MECH-292 bank (rank_ghost_goals); fails loud if the bank
        # isn't wired. The MECH-292 block above transitively guarantees
        # use_anchor_sets and AnchorSetConfig.use_sd039_anchor_payload,
        # so we only need to assert the bank flag here.
        if getattr(config, "use_mech293_ghost_probes", False):
            if self.ghost_goal_bank is None:
                raise ValueError(
                    "HippocampalConfig.use_mech293_ghost_probes=True but "
                    "use_mech292_ghost_bank=False; MECH-293 requires the "
                    "MECH-292 ranked ghost-goal bank as its proposal source."
                )
        # Per-tick MECH-293 diagnostics from propose_trajectories(). Cleared
        # at the start of each propose_trajectories() call. Empty dict when
        # MECH-293 is off or the ghost branch did not fire this tick.
        self._last_propose_diagnostics: Dict[str, Any] = {}

    def drain_boundary_events(self) -> List[BoundaryEvent]:
        """Return and clear all queued boundary events from MECH-288.

        Downstream consumers (MECH-287 broadcast trigger, MECH-269 anchor-reset)
        call this once per tick to consume events that the segmenter emitted
        during agent.sense().
        """
        events = list(self._boundary_event_queue)
        self._boundary_event_queue.clear()
        return events

    def drain_broadcast_events(self) -> List[BroadcastEvent]:
        """Return and clear all queued broadcast events from MECH-287.

        Downstream consumers (MECH-269 anchor-reset T3, MECH-284 staleness
        accumulator Phase 3) call this once per tick to consume events
        that the invalidation trigger emitted during agent.sense(). No-op
        when use_invalidation_trigger is False.
        """
        events = list(self._broadcast_event_queue)
        self._broadcast_event_queue.clear()
        return events

    def reset_invalidation_trigger(self) -> None:
        """Per-episode reset of the MECH-287 invalidation trigger.

        Clears tonic history, diagnostic counters, and the broadcast
        queue. No-op when use_invalidation_trigger is False.
        """
        if self.invalidation_trigger is not None:
            self.invalidation_trigger.reset()
        self._broadcast_event_queue.clear()

    def _get_terrain_action_object_mean(
        self,
        z_world: torch.Tensor,
        e1_prior: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute terrain-informed action-object distribution mean (SD-004).

        Queries residue field at z_world and combines with E1 prior (SD-002)
        to bias the initial action-object proposal toward low-residue regions.

        Args:
            z_world:  [batch, world_dim]
            e1_prior: world-domain E1 prior [batch, world_dim]. If None, zeros.

        Returns:
            action_object_mean [batch, horizon, action_object_dim]
        """
        with torch.no_grad():
            residue_val = self.residue_field.evaluate(z_world).unsqueeze(-1)  # [batch, 1]

        if e1_prior is None:
            e1_prior = torch.zeros_like(z_world)

        if self._benefit_terrain:
            with torch.no_grad():
                benefit_val = self.residue_field.evaluate_benefit(z_world).unsqueeze(-1)
            combined = torch.cat([z_world, e1_prior, residue_val, benefit_val], dim=-1)
        else:
            combined = torch.cat([z_world, e1_prior, residue_val], dim=-1)
        mean_flat = self.terrain_prior(combined)  # [batch, action_object_dim * horizon]
        return mean_flat.view(
            z_world.shape[0], self.config.horizon, self.config.action_object_dim
        )

    def _decode_action_objects(
        self,
        action_objects: torch.Tensor,
    ) -> torch.Tensor:
        """
        Decode action-object sequence to action sequence.

        Args:
            action_objects: [batch, horizon, action_object_dim]

        Returns:
            actions [batch, horizon, action_dim]
        """
        batch, horizon, ao_dim = action_objects.shape
        flat = action_objects.reshape(batch * horizon, ao_dim)
        actions_flat = self.action_object_decoder(flat)
        return actions_flat.reshape(batch, horizon, self.config.action_dim)

    @staticmethod
    def _entropy_from_counts(counts: Dict[int, int]) -> float:
        total = sum(counts.values())
        if total <= 0:
            return 0.0
        entropy = 0.0
        for count in counts.values():
            if count <= 0:
                continue
            p = float(count) / float(total)
            entropy -= p * math.log(p)
        return float(entropy)

    @staticmethod
    def _counts_from_tensor(classes: torch.Tensor) -> Dict[int, int]:
        flat = classes.detach().reshape(-1).to(dtype=torch.long, device="cpu")
        counts: Dict[int, int] = {}
        for cls in flat.tolist():
            counts[int(cls)] = counts.get(int(cls), 0) + 1
        return dict(sorted(counts.items()))

    def _summarize_action_tensor(self, actions: torch.Tensor) -> Dict[str, Any]:
        """Summarize continuous action vectors by their downstream argmax class."""
        with torch.no_grad():
            if actions.dim() == 3:
                # [batch, horizon, action_dim] -> [candidate, batch, horizon, action_dim]
                action_batch = actions.unsqueeze(0)
            elif actions.dim() == 4:
                action_batch = actions
            else:
                raise ValueError(
                    "actions must have shape [batch,horizon,action_dim] or "
                    "[candidate,batch,horizon,action_dim]"
                )

            first_classes = action_batch[:, :, 0, :].argmax(dim=-1)
            all_classes = action_batch.argmax(dim=-1)
            first_counts = self._counts_from_tensor(first_classes)
            all_counts = self._counts_from_tensor(all_classes)

            flat_actions = action_batch.detach().reshape(-1, action_batch.shape[-1])
            raw_mean = flat_actions.mean(dim=0).detach().cpu().tolist()
            raw_std = flat_actions.std(dim=0, unbiased=False).detach().cpu().tolist()
            raw_min = flat_actions.min(dim=0).values.detach().cpu().tolist()
            raw_max = flat_actions.max(dim=0).values.detach().cpu().tolist()

        return {
            "first_action_counts": first_counts,
            "unique_first_action_classes": int(len(first_counts)),
            "first_action_entropy": self._entropy_from_counts(first_counts),
            "all_action_argmax_histogram": all_counts,
            "raw_output_stats": {
                "mean_by_action_dim": [float(v) for v in raw_mean],
                "std_by_action_dim": [float(v) for v in raw_std],
                "min_by_action_dim": [float(v) for v in raw_min],
                "max_by_action_dim": [float(v) for v in raw_max],
            },
        }

    def _summarize_trajectories(self, trajectories: List[Trajectory]) -> Dict[str, Any]:
        if not trajectories:
            return {
                "first_action_counts": {},
                "unique_first_action_classes": 0,
                "first_action_entropy": 0.0,
                "all_action_argmax_histogram": {},
                "raw_output_stats": {
                    "mean_by_action_dim": [],
                    "std_by_action_dim": [],
                    "min_by_action_dim": [],
                    "max_by_action_dim": [],
                },
            }
        actions = torch.stack([t.actions.detach() for t in trajectories], dim=0)
        return self._summarize_action_tensor(actions)

    def _build_action_class_scaffold_candidates(
        self,
        z_self: torch.Tensor,
        z_world: torch.Tensor,
        action_bias: Optional[torch.Tensor] = None,
    ) -> List[Trajectory]:
        """Diagnostic-only candidate scaffold with one first action per class."""
        batch_size = z_world.shape[0]
        device = z_world.device
        dtype = z_world.dtype
        horizon = int(self.config.horizon)
        action_dim = int(self.config.action_dim)
        scaffolds: List[Trajectory] = []

        for cls in range(action_dim):
            actions = torch.zeros(
                batch_size,
                horizon,
                action_dim,
                device=device,
                dtype=dtype,
            )
            actions[:, 0, cls] = 1.0
            traj = self.e2.rollout_with_world(
                z_self,
                z_world,
                actions,
                compute_action_objects=True,
                action_bias=action_bias,
            )
            metadata = dict(traj.metadata or {})
            metadata.update({
                "source": "action_class_scaffold",
                "first_action_class": int(cls),
            })
            traj.metadata = metadata
            scaffolds.append(traj)
        return scaffolds

    @staticmethod
    def _trajectory_first_action_class(trajectory: Trajectory) -> int:
        return int(
            trajectory.actions[:, 0, :]
            .argmax(dim=-1)
            .detach()
            .reshape(-1)[0]
            .item()
        )

    @staticmethod
    def _ao_std_stats(ao_std: torch.Tensor) -> Dict[str, float]:
        detached = ao_std.detach()
        return {
            "ao_std_mean": float(detached.mean().item()),
            "ao_std_min": float(detached.min().item()),
            "ao_std_max": float(detached.max().item()),
        }

    def _support_preserving_target_class_count(
        self,
        budget: Optional[int] = None,
    ) -> int:
        action_dim = int(self.config.action_dim)
        if action_dim <= 1:
            return 1
        configured = int(
            getattr(self.config, "support_preserving_min_first_action_classes", 2)
        )
        target = min(action_dim, max(2, configured))
        if budget is not None:
            target = min(target, max(1, int(budget)))
        return int(target)

    @staticmethod
    def _score_to_float(score: torch.Tensor) -> float:
        if isinstance(score, torch.Tensor):
            return float(score.detach().cpu().item())
        return float(score)

    def _support_preserving_elite_indices(
        self,
        trajectories: List[Trajectory],
        scores_tensor: torch.Tensor,
        elite_indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        target = self._support_preserving_target_class_count(
            budget=int(elite_indices.numel())
        )
        elite_trajectories = [
            trajectories[int(i.item() if hasattr(i, "item") else i)]
            for i in elite_indices
        ]
        before_summary = self._summarize_trajectories(elite_trajectories)
        diag: Dict[str, Any] = {
            "support_preserving_elite_active": False,
            "support_preserving_elite_target_classes": int(target),
            "support_preserving_elite_classes_before": sorted(
                int(k) for k in before_summary["first_action_counts"].keys()
            ),
            "support_preserving_elite_classes_after": sorted(
                int(k) for k in before_summary["first_action_counts"].keys()
            ),
        }

        if not bool(getattr(self.config, "use_support_preserving_cem", False)):
            return elite_indices, diag
        if target <= 1 or len(trajectories) <= 1:
            return elite_indices, diag

        use_stratified = bool(
            getattr(self.config, "support_preserving_stratified_elites", False)
        )
        per_class_quota = int(
            getattr(self.config, "support_preserving_per_class_quota", 0)
        )

        # Non-stratified, no quota: original gate (skip if already diverse enough).
        # With stratified=True or quota>0, always reselect to enforce the per-class
        # guarantee.
        if not use_stratified and per_class_quota == 0:
            if before_summary["unique_first_action_classes"] >= target:
                return elite_indices, diag

        budget = int(elite_indices.numel())
        selected: List[int] = []
        selected_set: set = set()

        if per_class_quota > 0:
            # Build per-class candidate lists, sorted by score ascending (lower = better).
            candidates_by_class: Dict[int, List[Tuple[int, float]]] = {}
            for idx, traj in enumerate(trajectories):
                cls = self._trajectory_first_action_class(traj)
                score = self._score_to_float(scores_tensor[idx])
                if cls not in candidates_by_class:
                    candidates_by_class[cls] = []
                candidates_by_class[cls].append((idx, score))
            for cls in candidates_by_class:
                candidates_by_class[cls].sort(key=lambda x: x[1])
            # Rank classes by their best (lowest) score.
            ranked_classes_q = sorted(
                candidates_by_class.items(),
                key=lambda item: (item[1][0][1], item[0]),
            )
            for cls, class_cands in ranked_classes_q:
                n_taken = 0
                for idx, _score in class_cands:
                    if len(selected) >= budget or n_taken >= per_class_quota:
                        break
                    if idx not in selected_set:
                        selected.append(idx)
                        selected_set.add(idx)
                        n_taken += 1
        else:
            # Original best-per-class logic, with or without the target gate.
            best_by_class: Dict[int, Tuple[int, float]] = {}
            for idx, traj in enumerate(trajectories):
                cls = self._trajectory_first_action_class(traj)
                score = self._score_to_float(scores_tensor[idx])
                if cls not in best_by_class or score < best_by_class[cls][1]:
                    best_by_class[cls] = (idx, score)

            if not use_stratified and len(best_by_class) < target:
                return elite_indices, diag

            selected_classes: set = set()
            ranked_classes = sorted(
                best_by_class.items(),
                key=lambda item: (item[1][1], item[0]),
            )
            # Stratified: force ALL present classes; non-stratified: up to target.
            n_classes_to_force = (
                len(best_by_class) if use_stratified else target
            )
            for cls, (idx, _score) in ranked_classes:
                if len(selected_classes) >= n_classes_to_force:
                    break
                if idx not in selected_set:
                    selected.append(idx)
                    selected_set.add(idx)
                    selected_classes.add(cls)

        # Fill remaining elite slots score-sorted from the full candidate pool.
        for raw_idx in torch.argsort(scores_tensor).detach().cpu().tolist():
            idx = int(raw_idx)
            if len(selected) >= budget:
                break
            if idx not in selected_set:
                selected.append(idx)
                selected_set.add(idx)

        if not selected:
            return elite_indices, diag

        selected = selected[:budget]
        new_indices = torch.tensor(
            selected,
            dtype=elite_indices.dtype,
            device=elite_indices.device,
        )
        after_summary = self._summarize_trajectories(
            [trajectories[i] for i in selected]
        )
        diag.update({
            "support_preserving_elite_active": bool(
                after_summary["unique_first_action_classes"]
                > before_summary["unique_first_action_classes"]
            ) or use_stratified or (per_class_quota > 0),
            "support_preserving_elite_classes_after": sorted(
                int(k) for k in after_summary["first_action_counts"].keys()
            ),
            "support_preserving_stratified_elites": use_stratified,
            "support_preserving_per_class_quota": per_class_quota,
        })
        return new_indices, diag

    def _inject_support_preserving_candidates(
        self,
        trajectories: List[Trajectory],
        z_self: torch.Tensor,
        z_world: torch.Tensor,
        action_bias: Optional[torch.Tensor],
        budget: int,
    ) -> Tuple[List[Trajectory], Dict[str, Any]]:
        target = self._support_preserving_target_class_count(budget=budget)
        pre_summary = self._summarize_trajectories(trajectories)
        diag: Dict[str, Any] = {
            "use_support_preserving_cem": bool(
                getattr(self.config, "use_support_preserving_cem", False)
            ),
            "support_preserving_min_first_action_classes": int(
                getattr(
                    self.config,
                    "support_preserving_min_first_action_classes",
                    2,
                )
            ),
            "support_preserving_target_first_action_classes": int(target),
            "support_preserving_active": False,
            "support_preserving_injected_candidates": 0,
            "support_preserving_injected_classes": [],
            "support_preserving_replaced_candidates": 0,
            "support_preserving_pre_injection_first_action_counts": pre_summary[
                "first_action_counts"
            ],
            "support_preserving_pre_injection_unique_first_action_classes": pre_summary[
                "unique_first_action_classes"
            ],
            "support_preserving_pre_injection_first_action_entropy": pre_summary[
                "first_action_entropy"
            ],
        }

        if not diag["use_support_preserving_cem"]:
            return trajectories, diag
        if target <= 1 or pre_summary["unique_first_action_classes"] >= target:
            return trajectories, diag

        present = {int(k) for k in pre_summary["first_action_counts"].keys()}
        missing_classes = [
            cls for cls in range(int(self.config.action_dim)) if cls not in present
        ]
        need = max(0, target - int(pre_summary["unique_first_action_classes"]))
        inject_classes = missing_classes[:need]
        if not inject_classes:
            return trajectories, diag

        scaffold = self._build_action_class_scaffold_candidates(
            z_self=z_self,
            z_world=z_world,
            action_bias=action_bias,
        )
        scaffold_by_class = {
            self._trajectory_first_action_class(traj): traj for traj in scaffold
        }
        injected: List[Trajectory] = []
        for cls in inject_classes:
            traj = scaffold_by_class.get(cls)
            if traj is None:
                continue
            metadata = dict(traj.metadata or {})
            metadata.update({
                "source": "support_preserving_cem_injected",
                "first_action_class": int(cls),
            })
            traj.metadata = metadata
            injected.append(traj)

        if not injected:
            return trajectories, diag

        total_budget = max(0, int(budget))
        if total_budget <= 0:
            total_budget = len(trajectories) + len(injected)
        keep_n = max(0, total_budget - len(injected))
        if keep_n >= len(trajectories):
            kept = list(trajectories)
        else:
            scored = [
                (idx, self._score_to_float(self._score_trajectory(traj)))
                for idx, traj in enumerate(trajectories)
            ]
            scored.sort(key=lambda item: (item[1], item[0]))
            keep_indices = {idx for idx, _score in scored[:keep_n]}
            kept = [
                traj for idx, traj in enumerate(trajectories)
                if idx in keep_indices
            ]
        repaired = kept + injected
        post_summary = self._summarize_trajectories(repaired)
        diag.update({
            "support_preserving_active": True,
            "support_preserving_injected_candidates": int(len(injected)),
            "support_preserving_injected_classes": [
                self._trajectory_first_action_class(traj) for traj in injected
            ],
            "support_preserving_replaced_candidates": int(
                max(0, len(trajectories) - len(kept))
            ),
            "support_preserving_post_injection_first_action_counts": post_summary[
                "first_action_counts"
            ],
            "support_preserving_post_injection_unique_first_action_classes": post_summary[
                "unique_first_action_classes"
            ],
            "support_preserving_post_injection_first_action_entropy": post_summary[
                "first_action_entropy"
            ],
        })
        return repaired, diag

    def _score_trajectory(self, trajectory: Trajectory) -> torch.Tensor:
        """
        Score a trajectory for CEM elite selection.

        ARC-007 STRICT: primary scoring is terrain-only (residue field over z_world).
        No independent harm prediction here -- E3 introduces all value weighting.

        VALENCE_WANTING extension (config.wanting_weight > 0):
        Subtracts the mean VALENCE_WANTING value along the trajectory from the
        terrain score, biasing CEM selection toward high-wanting (resource-proximal)
        regions. Subtraction because CEM minimises the score (lower = better).
        The wanting gradient is the residue field's accumulated VALENCE_WANTING
        component -- populated by SerotoninModule.update_benefit_salience() during
        waking steps when tonic_5ht_enabled=True.

        Returns a scalar score (lower = better).
        """
        if trajectory.world_states is not None:
            world_seq = trajectory.get_world_state_sequence()  # [batch, horizon, world_dim]
            if world_seq is not None:
                terrain_score = self.residue_field.evaluate_trajectory(world_seq).sum()

                if self.config.wanting_weight > 0:
                    # Reshape to [batch*horizon, world_dim] for evaluate_valence
                    batch, horizon, world_dim = world_seq.shape
                    flat = world_seq.reshape(batch * horizon, world_dim)
                    with torch.no_grad():
                        valence_flat = self.residue_field.evaluate_valence(flat)  # [B*H, 4]
                    wanting_flat = valence_flat[..., VALENCE_WANTING]  # [B*H]
                    wanting_score = wanting_flat.mean()
                    return terrain_score - self.config.wanting_weight * wanting_score

                return terrain_score

        # Fallback: residue over z_self states (pre-SD-005 wiring)
        states = trajectory.get_state_sequence()
        return self.residue_field.evaluate_trajectory(states).sum()

    def _compute_mode_noise_scale(
        self,
        operating_mode: Optional[Dict[str, float]],
    ) -> Optional[float]:
        """MECH-267: weighted CEM noise multiplier from a mode probability vector.

        Returns None when conditioning is disabled or operating_mode is None
        (caller leaves ao_std untouched). Otherwise returns
        sum_{m} operating_mode[m] * config.mode_noise_scale.get(m, 1.0).
        Modes present in operating_mode but absent from the noise-scale map
        contribute their probability with multiplier 1.0 (no shift).
        """
        if not getattr(self.config, "mode_conditioning_enabled", False):
            return None
        if operating_mode is None or len(operating_mode) == 0:
            return None
        scale_map = getattr(self.config, "mode_noise_scale", None) or {}
        scale = 0.0
        for mode_name, prob in operating_mode.items():
            scale += float(prob) * float(scale_map.get(mode_name, 1.0))
        return scale

    def propose_trajectories(
        self,
        z_world: torch.Tensor,
        z_self: Optional[torch.Tensor] = None,
        num_candidates: Optional[int] = None,
        e1_prior: Optional[torch.Tensor] = None,
        action_bias: Optional[torch.Tensor] = None,
        operating_mode: Optional[Dict[str, float]] = None,
        current_z_goal: Optional[torch.Tensor] = None,
        persistence_appraisal: Optional[PersistenceAppraisal] = None,
    ) -> List[Trajectory]:
        """
        Propose candidate trajectories via terrain-guided CEM in action-object space.

        V3 algorithm:
        1. Initialise action-object distribution mean from terrain prior
           (SD-004: in action-object space O, conditioned on E1 prior — SD-002)
        2. For each CEM iteration:
           a. Sample action-object sequences from current distribution
           b. Decode to action sequences via action_object_decoder
           c. Roll out through E2 (self + world tracks)
           d. Score via residue field terrain (ARC-007 strict — no value head)
           e. Refit distribution to elite (lowest-residue) samples
        3. Return final candidates for E3 to evaluate

        SD-016 (MECH-151): optional action_bias [batch, action_object_dim] from
        E1.extract_cue_context(). When provided, passed through to every
        E2.rollout_with_world() call so that each action-object in the CEM search
        is shifted by the cue-indexed contextual bias. When None (all existing
        callers), behaviour is unchanged.

        MECH-267: optional operating_mode dict[str, float] (probability vector
        from SD-032a SalienceCoordinator). When config.mode_conditioning_enabled
        is True AND operating_mode is supplied, the CEM proposal std is scaled
        by the per-mode weighted average from config.mode_noise_scale. When
        either condition is False, behaviour is unchanged (operating_mode is
        recorded for diagnostics but not applied).

        MECH-293: optional current_z_goal [batch, goal_dim]. When
        use_mech293_ghost_probes is True AND current_z_goal is not None AND
        the MECH-292 bank returns >=1 entry clearing goal_match_floor, a
        minority budget of CEM probes is seeded around the highest-priority
        bank entries' anchor.z_world (instead of the agent's current
        z_world). Each ghost trajectory carries hypothesis_tag=True and a
        metadata dict tagging the source anchor. mech293_replace_lowest_ranked
        controls whether ghosts replace the highest-cost value-flat
        candidates (default; total count preserved) or are appended (raises
        total). When current_z_goal is None or the bank is empty, the ghost
        branch is silent and behaviour is identical to the value-flat path.

        Args:
            z_world:        Current z_world [batch, world_dim]
            z_self:         Current z_self [batch, self_dim] (for E2 rollouts)
            num_candidates: Number of candidates per CEM iteration
            e1_prior:       E1 world-domain prior [batch, world_dim] (SD-002)
            action_bias:    [batch, action_object_dim] or None (SD-016)
            operating_mode: Dict[str, float] (mode -> prob) or None (MECH-267)
            current_z_goal: [batch, goal_dim] or None (MECH-293; consumed by
                            MECH-292 ghost-goal bank ranking)
            persistence_appraisal: optional MECH-340 global gate inputs
                            (REEAgent supplies when gate enabled)

        Returns:
            List of Trajectory objects
        """
        # Clear MECH-293 diagnostics from prior tick. Repopulated below
        # only when the ghost branch actually fires.
        self._last_propose_diagnostics = {}
        n = num_candidates or self.config.num_candidates
        num_elite = max(1, int(n * self.config.elite_fraction))
        batch_size = z_world.shape[0]
        device = z_world.device

        # Fallback z_self (zeros) when not provided
        if z_self is None:
            z_self = torch.zeros(batch_size, self.e2.config.self_dim, device=device)

        # Initialise in action-object space (SD-004)
        ao_mean = self._get_terrain_action_object_mean(z_world, e1_prior=e1_prior)
        ao_std  = torch.ones_like(ao_mean)

        # MECH-267: mode-conditioned CEM noise scale.
        self._last_operating_mode = (
            dict(operating_mode) if operating_mode is not None else None
        )
        mode_scale = self._compute_mode_noise_scale(operating_mode)
        self._last_mode_noise_scale = mode_scale
        if mode_scale is not None:
            ao_std = ao_std * mode_scale

        all_trajectories: List[Trajectory] = []

        # V3-EXQ-553 proposer-fix substrate: optional orthogonal-basis noise.
        # When use_orthogonal_cem_seeding is True, pre-generate n orthogonal
        # noise vectors per CEM iteration (QR over a stack of iid Gaussians
        # in flatten-dim = H * ao_dim) and use them in place of the legacy
        # per-candidate iid Gaussian draws. The flag is consulted once per
        # iteration; iid fallback is used when n > flatten_dim (rank-deficient
        # basis). Diagnostics counted on the per-tick propose-diagnostics dict.
        use_orthogonal = bool(
            getattr(self.config, "use_orthogonal_cem_seeding", False)
        )
        ortho_diag = {
            "use_orthogonal_cem_seeding": use_orthogonal,
            "n_orthogonal_iters": 0,
            "n_iid_fallback_candidates": 0,
        }
        cem_iteration_diagnostics: List[Dict[str, Any]] = []

        for _iteration in range(self.config.num_cem_iterations):
            trajectories: List[Trajectory] = []
            scores: List[torch.Tensor] = []
            ao_std_diag = self._ao_std_stats(ao_std)

            # Pre-generate orthogonal noise for this CEM iteration when enabled.
            ortho_noise: Optional[torch.Tensor] = None
            if use_orthogonal:
                # ao_mean shape: [batch, H, ao_dim]; flatten_dim = H * ao_dim.
                # Build [n, flatten_dim] iid Gaussian, then QR decompose so the
                # n rows form an orthonormal set (rank-deficient when
                # n > flatten_dim -> falls back to iid per-candidate).
                flatten_dim = int(ao_mean.shape[1] * ao_mean.shape[2])
                if n <= flatten_dim:
                    # Stack n iid Gaussian vectors as rows of an [n, flatten_dim]
                    # matrix; QR yields Q [n, n] and R [n, flatten_dim] with the
                    # rows of (Q.T @ R) approximating the original. We want a
                    # set of n orthogonal unit-norm vectors in flatten_dim
                    # space -- equivalent to taking the first n columns of the
                    # Q matrix from QR(A.T) where A is [n, flatten_dim].
                    raw = torch.randn(n, flatten_dim, device=device, dtype=ao_mean.dtype)
                    # QR over the transpose so we get an orthonormal basis
                    # in flatten_dim space. raw.T is [flatten_dim, n]; Q is
                    # [flatten_dim, n] with orthonormal columns.
                    q, _ = torch.linalg.qr(raw.T, mode="reduced")
                    # q.T is [n, flatten_dim] with orthonormal rows. Rescale
                    # by sqrt(flatten_dim) so each row's expected squared
                    # magnitude matches an iid standard Gaussian sample (which
                    # has expected squared norm = flatten_dim). This preserves
                    # ao_std's amplitude calibration vs. the iid path.
                    ortho_noise = q.T * float(flatten_dim) ** 0.5
                    # Shape ortho_noise into [n, batch, H, ao_dim]. Same noise
                    # pattern broadcast across batch (batch is typically 1 at
                    # this call site).
                    ortho_noise = ortho_noise.view(
                        n, ao_mean.shape[1], ao_mean.shape[2]
                    ).unsqueeze(1).expand(
                        n, ao_mean.shape[0], ao_mean.shape[1], ao_mean.shape[2]
                    ).contiguous()
                    ortho_diag["n_orthogonal_iters"] += 1
                else:
                    # Rank-deficient: fall back to iid for all candidates this
                    # iteration. Cleanly logged for diagnostics.
                    ortho_noise = None
                    ortho_diag["n_iid_fallback_candidates"] += n

            for cand_idx in range(n):
                if ortho_noise is not None:
                    noise = ortho_noise[cand_idx]
                else:
                    noise = torch.randn_like(ao_mean)
                action_objects_sample = ao_mean + ao_std * noise  # [batch, H, ao_dim]

                # Decode action objects → actions for E2 rollout
                actions = self._decode_action_objects(action_objects_sample)

                # Roll out: track both z_self and z_world for scoring.
                # SD-016: action_bias is passed through to action_object() calls.
                traj = self.e2.rollout_with_world(
                    z_self, z_world, actions,
                    compute_action_objects=True,
                    action_bias=action_bias,
                )
                trajectories.append(traj)
                scores.append(self._score_trajectory(traj))

            scores_tensor = torch.stack(scores)
            elite_indices = torch.argsort(scores_tensor)[:num_elite]
            elite_indices, support_elite_diag = (
                self._support_preserving_elite_indices(
                    trajectories=trajectories,
                    scores_tensor=scores_tensor,
                    elite_indices=elite_indices,
                )
            )
            decoded_summary = self._summarize_trajectories(trajectories)
            elite_trajectories = [
                trajectories[int(i.item() if hasattr(i, "item") else i)]
                for i in elite_indices
            ]
            elite_summary = self._summarize_trajectories(elite_trajectories)
            cem_iteration_diagnostics.append({
                "iteration": int(_iteration),
                **ao_std_diag,
                "pre_refit_first_action_counts": decoded_summary["first_action_counts"],
                "pre_refit_unique_first_action_classes": decoded_summary[
                    "unique_first_action_classes"
                ],
                "pre_refit_first_action_entropy": decoded_summary[
                    "first_action_entropy"
                ],
                "decoded_action_argmax_histogram": decoded_summary[
                    "all_action_argmax_histogram"
                ],
                "action_object_decoder_raw_output_stats": decoded_summary[
                    "raw_output_stats"
                ],
                "post_elite_refit_first_action_counts": elite_summary[
                    "first_action_counts"
                ],
                "post_elite_refit_unique_first_action_classes": elite_summary[
                    "unique_first_action_classes"
                ],
                "post_elite_refit_first_action_entropy": elite_summary[
                    "first_action_entropy"
                ],
                **support_elite_diag,
            })

            # Refit distribution to action-object sequences.
            # SD-055: when use_differentiable_cem=True, use a softmax-weighted
            # mean over ALL candidates so that ao_mean is differentiable w.r.t.
            # scores (and therefore w.r.t. cue_action_proj via E2 rollouts).
            # ARC-007 STRICT preserved: scoring still uses _score_trajectory
            # over residue terrain; no value head is added here.
            # When False (default), legacy argsort + indexed mean over elites.
            _std_floor = float(
                getattr(self.config, "support_preserving_ao_std_floor", 0.0)
            )
            if getattr(self.config, "use_differentiable_cem", False):
                # Collect (ao_seq, score) pairs for candidates that have ao seqs.
                _diff_ao, _diff_scores = [], []
                for _traj, _sc in zip(trajectories, scores):
                    _ao_seq = _traj.get_action_object_sequence()
                    if _ao_seq is not None:
                        _diff_ao.append(_ao_seq)
                        _diff_scores.append(_sc)
                if _diff_ao:
                    _T = max(
                        float(getattr(self.config, "differentiable_cem_temperature", 1.0)),
                        1e-6,
                    )
                    _all_ao = torch.stack(_diff_ao)   # [n, batch, H, ao_dim]
                    _sc_t   = torch.stack(_diff_scores)  # [n]
                    # softmax(-score/T): lower score -> higher weight (better)
                    _w = torch.softmax(-_sc_t / _T, dim=0)  # [n]
                    ao_mean = (_w.view(-1, 1, 1, 1) * _all_ao).sum(dim=0)
                    ao_std  = _all_ao.std(dim=0) + 1e-6
                    if (
                        getattr(self.config, "use_support_preserving_cem", False)
                        and _std_floor > 0.0
                    ):
                        ao_std = torch.clamp(ao_std, min=_std_floor)
                # else: keep previous distribution
            else:
                # Legacy: argsort elite selection + indexed mean.
                elite_ao = []
                for i in elite_indices:
                    idx = int(i.item() if hasattr(i, "item") else i)
                    ao_seq = trajectories[idx].get_action_object_sequence()
                    if ao_seq is not None:
                        elite_ao.append(ao_seq)
                if elite_ao:
                    elite_ao_tensor = torch.stack(elite_ao)  # [elite, batch, H, ao_dim]
                    ao_mean = elite_ao_tensor.mean(dim=0)
                    ao_std  = elite_ao_tensor.std(dim=0) + 1e-6
                    # V3-EXQ-563c: ao_std floor prevents collapse when all
                    # elites share one action class.
                    if (
                        getattr(self.config, "use_support_preserving_cem", False)
                        and _std_floor > 0.0
                    ):
                        ao_std = torch.clamp(ao_std, min=_std_floor)
                # else: keep previous distribution

            all_trajectories = trajectories

        # MECH-293: minority ghost-seeded probe budget. Reads the MECH-292
        # ranked bank (goal-match-floor filtered, no rumination) and seeds
        # CEM probes around the top-ranked anchors' z_world rather than the
        # agent's current z_world. Each ghost trajectory carries
        # hypothesis_tag=True + provenance metadata for downstream
        # diagnostics. When the bank is empty (no goal active, or no
        # anchor clears goal_match_floor), this branch is silent.
        if (
            getattr(self.config, "use_mech293_ghost_probes", False)
            and self.ghost_goal_bank is not None
        ):
            ghost_candidates = self._propose_ghost_seeded(
                current_z_goal=current_z_goal,
                n_total=n,
                z_self=z_self,
                e1_prior=e1_prior,
                action_bias=action_bias,
                persistence_appraisal=persistence_appraisal,
            )
            if ghost_candidates:
                all_trajectories = self._mix_value_flat_with_ghost(
                    value_flat=all_trajectories,
                    ghost=ghost_candidates,
                    replace_lowest=bool(
                        getattr(self.config, "mech293_replace_lowest_ranked", True)
                    ),
                )

        all_trajectories, support_preserving_diag = (
            self._inject_support_preserving_candidates(
                trajectories=all_trajectories,
                z_self=z_self,
                z_world=z_world,
                action_bias=action_bias,
                budget=n,
            )
        )

        scaffold_diag: Dict[str, Any] = {
            "use_action_class_scaffold_candidates": bool(
                getattr(self.config, "use_action_class_scaffold_candidates", False)
            ),
            "action_class_scaffold_candidates_added": 0,
            "action_class_scaffold_classes": [],
        }
        if scaffold_diag["use_action_class_scaffold_candidates"]:
            scaffold = self._build_action_class_scaffold_candidates(
                z_self=z_self,
                z_world=z_world,
                action_bias=action_bias,
            )
            if scaffold:
                scaffold_classes = [
                    int(t.actions[:, 0, :].argmax(dim=-1).flatten()[0].item())
                    for t in scaffold
                ]
                keep_n = max(0, n - len(scaffold))
                all_trajectories = list(scaffold) + list(all_trajectories[:keep_n])
                scaffold_diag.update({
                    "action_class_scaffold_candidates_added": int(len(scaffold)),
                    "action_class_scaffold_classes": scaffold_classes,
                })

        final_summary = self._summarize_trajectories(all_trajectories)
        self._last_propose_diagnostics.update({
            "candidate_first_action_counts": final_summary["first_action_counts"],
            "candidate_unique_first_action_classes": final_summary[
                "unique_first_action_classes"
            ],
            "candidate_first_action_entropy": final_summary["first_action_entropy"],
            "candidate_samples_collected": int(len(all_trajectories)),
            "decoded_action_argmax_histogram": final_summary[
                "all_action_argmax_histogram"
            ],
            "action_object_decoder_raw_output_stats": final_summary[
                "raw_output_stats"
            ],
            "cem_iteration_diagnostics": cem_iteration_diagnostics,
            **ortho_diag,
            **support_preserving_diag,
            **scaffold_diag,
        })

        return all_trajectories

    # ------------------------------------------------------------------ #
    # MECH-293: waking ghost-goal probe search                            #
    # ------------------------------------------------------------------ #

    def _propose_ghost_seeded(
        self,
        current_z_goal: Optional[torch.Tensor],
        n_total: int,
        z_self: torch.Tensor,
        e1_prior: Optional[torch.Tensor] = None,
        action_bias: Optional[torch.Tensor] = None,
        persistence_appraisal: Optional[PersistenceAppraisal] = None,
    ) -> List[Trajectory]:
        """MECH-293: generate ghost-seeded probe trajectories from the
        MECH-292 bank.

        Each probe seeds the CEM init from a top-ranked bank entry's
        anchor.z_world rather than the agent's current z_world. The
        terrain prior + a single noise draw produce the action-object
        sequence; e2.rollout_with_world rolls it forward; the resulting
        trajectory is tagged hypothesis_tag=True and stamped with
        provenance metadata.

        Single noise draw per probe (no inner CEM refit) keeps the
        ghost-probe cost <= a single value-flat sample.

        Returns [] when:
          - current_z_goal is None,
          - the bank is empty / all anchors below goal_match_floor,
          - the resolved n_ghost is zero (e.g. min/max clamps to 0).
        """
        if current_z_goal is None:
            self._last_propose_diagnostics = {
                "mech293_n_ghost_proposed": 0,
                "mech293_n_ghost_admitted": 0,
                "mech293_max_ghost_priority": 0.0,
                "mech293_mean_goal_match_at_seed": 0.0,
                "mech293_reason": "no_z_goal",
            }
            return []

        entries = self.ghost_goal_bank.rank(
            current_z_goal,
            persistence_appraisal=persistence_appraisal,
        )
        if not entries:
            self._last_propose_diagnostics = {
                "mech293_n_ghost_proposed": 0,
                "mech293_n_ghost_admitted": 0,
                "mech293_max_ghost_priority": 0.0,
                "mech293_mean_goal_match_at_seed": 0.0,
                "mech293_reason": "empty_bank",
            }
            return []

        fraction = float(getattr(self.config, "mech293_ghost_fraction", 0.2))
        min_ghost = int(getattr(self.config, "mech293_min_ghost_candidates", 1))
        max_ghost = int(getattr(self.config, "mech293_max_ghost_candidates", 8))
        # Clamp to non-negative; sentinel of 0 disables.
        min_ghost = max(0, min_ghost)
        max_ghost = max(min_ghost, max_ghost)
        # round() ties-to-even; for a fraction of 0.2 and n=16 -> 3.
        n_ghost = int(round(n_total * max(0.0, fraction)))
        n_ghost = max(min_ghost, min(n_ghost, max_ghost))
        n_ghost = min(n_ghost, len(entries))
        if n_ghost <= 0:
            self._last_propose_diagnostics = {
                "mech293_n_ghost_proposed": 0,
                "mech293_n_ghost_admitted": 0,
                "mech293_max_ghost_priority": float(entries[0].ghost_priority),
                "mech293_mean_goal_match_at_seed": 0.0,
                "mech293_reason": "n_ghost_zero",
            }
            return []

        batch_size = z_self.shape[0]
        device = z_self.device
        ghost_trajectories: List[Trajectory] = []
        seed_goal_matches: List[float] = []
        max_priority = 0.0

        for entry in entries[:n_ghost]:
            anchor = entry.anchor
            # Anchor z_world is stored as a detached clone; broadcast /
            # repeat to the current batch shape so e2 rollouts run with
            # the same batch dimension as the value-flat path.
            anchor_z = anchor.z_world.detach()
            if anchor_z.dim() == 1:
                anchor_z = anchor_z.unsqueeze(0)
            if anchor_z.shape[0] != batch_size:
                anchor_z = anchor_z.expand(batch_size, -1).contiguous()
            anchor_z = anchor_z.to(device)

            # Seed action-object distribution at the anchor's z_world (NOT
            # current). Single noise draw -- ghost probes are exploratory,
            # not optimised plans.
            ao_mean = self._get_terrain_action_object_mean(
                anchor_z, e1_prior=e1_prior
            )
            ao_std = torch.ones_like(ao_mean)
            noise = torch.randn_like(ao_mean)
            action_objects_sample = ao_mean + ao_std * noise
            actions = self._decode_action_objects(action_objects_sample)

            traj = self.e2.rollout_with_world(
                z_self,
                anchor_z,
                actions,
                compute_action_objects=True,
                action_bias=action_bias,
            )
            traj.hypothesis_tag = True
            traj.metadata = {
                "source": "mech293_ghost_probe",
                "anchor_key": anchor.key,
                "ghost_priority": float(entry.ghost_priority),
                "goal_match": float(entry.components.get("goal_match", 0.0)),
            }
            ghost_trajectories.append(traj)
            seed_goal_matches.append(float(entry.components.get("goal_match", 0.0)))
            if float(entry.ghost_priority) > max_priority:
                max_priority = float(entry.ghost_priority)

        mean_match = (
            sum(seed_goal_matches) / len(seed_goal_matches)
            if seed_goal_matches else 0.0
        )
        self._last_propose_diagnostics = {
            "mech293_n_ghost_proposed": int(n_ghost),
            "mech293_n_ghost_admitted": int(len(ghost_trajectories)),
            "mech293_max_ghost_priority": float(max_priority),
            "mech293_mean_goal_match_at_seed": float(mean_match),
            "mech293_reason": "ok",
        }
        return ghost_trajectories

    def _mix_value_flat_with_ghost(
        self,
        value_flat: List[Trajectory],
        ghost: List[Trajectory],
        replace_lowest: bool = True,
    ) -> List[Trajectory]:
        """MECH-293: combine value-flat and ghost candidate pools.

        replace_lowest=True (default): drop the highest-cost value-flat
        candidates so total count = len(value_flat). CEM is lower-is-better,
        so "highest cost" means worst score, which is what we evict.
        Preserves downstream E3 selection cost.

        replace_lowest=False: append ghost on top of the value-flat pool
        (raises total count). Diagnostic-only path; used to disambiguate
        budget effects from seed effects.

        When n_ghost >= len(value_flat) under replace_lowest=True, returns
        the ghost pool alone (degenerate; means the ghost budget swallowed
        the entire value-flat pool, which would only happen with an
        extreme fraction setting -- left observable rather than clipped
        silently).
        """
        if not ghost:
            return value_flat
        if not replace_lowest:
            return list(value_flat) + list(ghost)
        n_ghost = len(ghost)
        if n_ghost >= len(value_flat):
            return list(ghost)
        # Score value-flat trajectories; lower = better, so we evict the
        # n_ghost trajectories with the HIGHEST scores (worst).
        scored = [
            (i, float(self._score_trajectory(t).item()
                      if isinstance(self._score_trajectory(t), torch.Tensor)
                      else self._score_trajectory(t)))
            for i, t in enumerate(value_flat)
        ]
        scored.sort(key=lambda iv: iv[1])  # ascending: best first
        keep_indices = {i for i, _ in scored[: len(value_flat) - n_ghost]}
        kept = [t for i, t in enumerate(value_flat) if i in keep_indices]
        return kept + list(ghost)

    def get_last_propose_diagnostics(self) -> Dict[str, Any]:
        """Last propose_trajectories() MECH-293 diagnostics. Returns {}
        when MECH-293 is off or the ghost branch has not fired."""
        return dict(self._last_propose_diagnostics)

    def replay(
        self,
        theta_buffer_recent: torch.Tensor,
        num_replay_steps: int = 5,
        drive_state: Optional[torch.Tensor] = None,
    ) -> List[Trajectory]:
        """
        SWR-equivalent replay for viability map consolidation (MECH-092).

        Called during quiescent E3 heartbeat cycles (no salient event pending).
        All content carries hypothesis_tag=True — replay cannot produce residue
        (MECH-094 invariant).

        MECH-203 extension: when drive_state is provided (from serotonergic
        system), replay start point is selected by valence-weighted priority
        using ResidueField.get_valence_priority(). Without drive_state, falls
        back to most-recent z_world (original behavior, fully backward compat).

        Args:
            theta_buffer_recent: Recent theta-cycle buffer content
                [T, batch, world_dim]
            num_replay_steps: Number of replay trajectories to generate
            drive_state: Optional [4] drive weights for valence-weighted start
                point selection. When provided, each buffer entry is scored
                by dot(valence, drive_state) and the highest-priority entry
                is used as replay start. When None, uses most recent entry.

        Returns:
            List of Trajectory objects (all hypothesis_tag=True on caller side)
        """
        if theta_buffer_recent is None or theta_buffer_recent.shape[0] == 0:
            return []

        # Select replay start point
        if drive_state is not None and theta_buffer_recent.shape[0] > 1:
            # MECH-203: valence-weighted replay start selection
            z_world_replay = self._select_valence_weighted_start(
                theta_buffer_recent, drive_state
            )
        else:
            # Default: most recent z_world from buffer
            z_world_replay = theta_buffer_recent[-1]   # [batch, world_dim]

        batch_size = z_world_replay.shape[0]
        device = z_world_replay.device
        z_self_zeros = torch.zeros(batch_size, self.e2.config.self_dim, device=device)

        # Generate random action sequences for replay rollouts
        replay_trajectories = []
        for _ in range(num_replay_steps):
            actions = self.e2.generate_random_actions(
                batch_size, self.config.horizon, device
            )
            traj = self.e2.rollout_with_world(
                z_self_zeros, z_world_replay, actions, compute_action_objects=False
            )
            replay_trajectories.append(traj)

        return replay_trajectories

    def _select_valence_weighted_start(
        self,
        theta_buffer_recent: torch.Tensor,
        drive_state: torch.Tensor,
    ) -> torch.Tensor:
        """
        MECH-203: Select replay start point by valence priority.

        Scores each buffer entry via ResidueField.get_valence_priority()
        and returns the highest-priority entry. Falls back to most recent
        if valence scoring is unavailable.

        Args:
            theta_buffer_recent: [T, batch, world_dim]
            drive_state: [4] drive weights

        Returns:
            z_world_replay: [batch, world_dim]
        """
        if not hasattr(self.residue_field, 'get_valence_priority'):
            return theta_buffer_recent[-1]

        T = theta_buffer_recent.shape[0]
        best_idx = T - 1  # default: most recent
        best_priority = -float('inf')

        with torch.no_grad():
            for t in range(T):
                z_w = theta_buffer_recent[t]  # [batch, world_dim]
                priority = self.residue_field.get_valence_priority(
                    z_w, drive_state
                )
                # Sum across batch for comparison
                p_val = float(priority.sum().item())
                if p_val > best_priority:
                    best_priority = p_val
                    best_idx = t

        return theta_buffer_recent[best_idx]

    # ------------------------------------------------------------------ #
    # MECH-165: Reverse replay diversity scheduler                         #
    # ------------------------------------------------------------------ #

    def record_exploration_trajectory(self, trajectory: Trajectory) -> None:
        """MECH-165: record a waking exploration trajectory for replay source material.

        Detaches all tensors to avoid holding computation graphs in the buffer.
        FIFO eviction when buffer exceeds exploration_buffer_maxlen.
        """
        detached_states = [s.detach() for s in trajectory.states]
        detached_actions = trajectory.actions.detach()
        detached_world = None
        if trajectory.world_states is not None:
            detached_world = [w.detach() for w in trajectory.world_states]
        detached_ao = None
        if trajectory.action_objects is not None:
            detached_ao = [ao.detach() for ao in trajectory.action_objects]
        detached = Trajectory(
            states=detached_states,
            actions=detached_actions,
            world_states=detached_world,
            action_objects=detached_ao,
            is_reverse=False,
            memory_strength=float(getattr(trajectory, "memory_strength", 1.0)),
            arousal_tag=float(getattr(trajectory, "arousal_tag", 0.0)),
        )
        self._exploration_buffer.append(detached)
        if len(self._exploration_buffer) > self._exploration_buffer_maxlen:
            self._exploration_buffer.pop(0)  # FIFO eviction

    def reverse_replay(self, trajectory: Trajectory) -> Trajectory:
        """MECH-165: replay stored trajectory in reverse temporal order.

        Reverses the state and world_state sequences; flips action tensor along
        the time dimension. Marks the returned trajectory with is_reverse=True.
        No new E2 rollout is performed -- this is a pure temporal reversal.
        """
        reversed_states = list(reversed(trajectory.states))
        reversed_actions = trajectory.actions.flip(1)  # flip time dim
        reversed_world = (
            list(reversed(trajectory.world_states))
            if trajectory.world_states is not None
            else None
        )
        reversed_ao = (
            list(reversed(trajectory.action_objects))
            if trajectory.action_objects is not None
            else None
        )
        return Trajectory(
            states=reversed_states,
            actions=reversed_actions,
            world_states=reversed_world,
            action_objects=reversed_ao,
            is_reverse=True,
            memory_strength=float(getattr(trajectory, "memory_strength", 1.0)),
            arousal_tag=float(getattr(trajectory, "arousal_tag", 0.0)),
        )

    def get_exploration_arousal_tags(self) -> Optional[torch.Tensor]:
        """Return the BLA arousal tags stored on exploration traces, if any."""
        if len(self._exploration_buffer) == 0:
            return None
        return torch.tensor(
            [
                float(getattr(traj, "arousal_tag", 0.0))
                for traj in self._exploration_buffer
            ],
            dtype=torch.float32,
        )

    def _sample_exploration_trajectory(
        self,
        retrieval_bias: Optional[torch.Tensor] = None,
    ) -> Optional[Trajectory]:
        """Sample a stored exploration trace, optionally weighted by BLA bias."""
        if len(self._exploration_buffer) == 0:
            return None

        weights = torch.tensor(
            [
                max(1e-6, float(getattr(traj, "memory_strength", 1.0)))
                for traj in self._exploration_buffer
            ],
            dtype=torch.float32,
        )
        if retrieval_bias is not None and retrieval_bias.numel() == len(self._exploration_buffer):
            bias = retrieval_bias.detach().to(torch.float32).cpu().clamp_min(0.0)
            weights = weights * bias

        if float(weights.sum().item()) <= 0.0:
            return random.choice(self._exploration_buffer)

        idx = int(torch.multinomial(weights, num_samples=1, replacement=True).item())
        return self._exploration_buffer[idx]

    def diverse_replay(
        self,
        theta_buffer_recent: torch.Tensor,
        num_replay_steps: int = 5,
        drive_state: Optional[torch.Tensor] = None,
        mode: str = "auto",
        retrieval_bias: Optional[torch.Tensor] = None,
    ) -> List[Trajectory]:
        """MECH-165: diversity-scheduled replay.

        Modes:
          "forward"  -- existing replay() behavior (random rollout from buffer)
          "reverse"  -- pick stored traj from exploration_buffer, replay in reverse
          "random"   -- existing replay() (same as forward in current impl)
          "auto"     -- sample mode probabilistically per config fractions

        Args:
            theta_buffer_recent: [T, batch, world_dim]
            num_replay_steps: number of replay trajectories to generate
            drive_state: optional [4] drive weights (MECH-203)
            mode: replay mode selection
            retrieval_bias: optional per-trace BLA weight vector aligned to the
                exploration buffer. When provided, reverse replay samples are
                weighted by memory_strength * retrieval_bias instead of uniform.

        Returns:
            List of Trajectory objects
        """
        trajectories: List[Trajectory] = []
        for _ in range(num_replay_steps):
            step_mode = mode
            if step_mode == "auto":
                r = random.random()
                has_buffer = len(self._exploration_buffer) > 0
                if r < self._reverse_fraction and has_buffer:
                    step_mode = "reverse"
                elif r < self._reverse_fraction + self._random_fraction:
                    step_mode = "random"
                else:
                    step_mode = "forward"

            if step_mode == "reverse" and len(self._exploration_buffer) > 0:
                source = self._sample_exploration_trajectory(
                    retrieval_bias=retrieval_bias
                )
                if source is not None:
                    trajectories.append(self.reverse_replay(source))
            else:
                # forward or random: delegate to existing replay()
                step_trajs = self.replay(
                    theta_buffer_recent, num_replay_steps=1,
                    drive_state=drive_state,
                )
                trajectories.extend(step_trajs)
        return trajectories

    def compute_completion_signal(self, trajectories: List[Trajectory]) -> float:
        """
        Compute the hippocampal trajectory completion signal (ARC-028, MECH-105).

        This is the dopamine-analog value fed to BetaGate.receive_hippocampal_completion().
        It implements the subiculum->NAc->VP->VTA loop described in Lisman & Grace (2005):
        a high-quality trajectory (low residue cost) drives a dopamine-analog release that
        causes the BG beta gate to drop, allowing E3 state to propagate to action selection.

        Formula:
            best_score = min residue cost across trajectories (lower = better)
            completion_signal = sigmoid(-best_score * 0.5)
            -> maps [0, inf) residue to [0.5, 1.0) signal
            -> high completion (near 0 residue) -> signal near 1.0
            -> poor completion (high residue) -> signal near 0.5
            -> empty list -> 0.0

        The result is cached in self._last_completion_signal.

        Args:
            trajectories: List of Trajectory objects (as returned by propose_trajectories).

        Returns:
            float in [0.0, 1.0): dopamine-analog completion quality signal.
        """
        if not trajectories:
            self._last_completion_signal = 0.0
            return 0.0

        scores = [self._score_trajectory(t) for t in trajectories]
        best_score = min(s.item() if isinstance(s, torch.Tensor) else float(s) for s in scores)
        signal = float(torch.sigmoid(torch.tensor(-best_score * 0.5)).item())
        self._last_completion_signal = signal
        return signal

    # ------------------------------------------------------------------ #
    # MECH-290: backward trajectory credit sweep (2026-04-24)             #
    # ------------------------------------------------------------------ #

    def record_committed_trajectory(self, trajectory: Trajectory) -> None:
        """MECH-290: record the committed trajectory for backward credit sweep.

        Called at BetaGate elevation (commit entry) in agent.select_action().
        Detaches all tensors to avoid holding computation graphs.

        Distinct from record_exploration_trajectory() (MECH-165 quiescent
        replay source): this stores the EXECUTED committed trajectory, not a
        CEM proposal candidate.

        No-op when use_backward_credit_sweep is False (backward compat).
        """
        if not getattr(self.config, "use_backward_credit_sweep", False):
            return
        detached_states = [s.detach() for s in trajectory.states]
        detached_actions = trajectory.actions.detach()
        detached_world = None
        if trajectory.world_states is not None:
            detached_world = [w.detach() for w in trajectory.world_states]
        detached_ao = None
        if trajectory.action_objects is not None:
            detached_ao = [ao.detach() for ao in trajectory.action_objects]
        self._committed_trajectory_buffer = Trajectory(
            states=detached_states,
            actions=detached_actions,
            world_states=detached_world,
            action_objects=detached_ao,
            is_reverse=False,
            memory_strength=float(getattr(trajectory, "memory_strength", 1.0)),
            arousal_tag=float(getattr(trajectory, "arousal_tag", 0.0)),
            # MECH-293 / MECH-094: the executed trajectory IS real,
            # regardless of whether the source proposal was a ghost
            # probe. Strip the hypothesis tag and metadata so downstream
            # write-paths (backward credit sweep, residue, etc.) treat
            # this as waking content.
            hypothesis_tag=False,
            metadata=None,
        )

    def backward_credit_sweep(self, outcome_quality: float) -> dict:
        """MECH-290: backward temporal credit sweep at trajectory completion.

        Biological basis: Foster & Wilson 2006 (Nature) -- reverse replay fires
        at reward endpoint during waking, concurrent with dopamine. Credit
        propagates backward from goal to trajectory start (hippocampal
        theta-burst to SWR transition; see also ARC-028 / MECH-105).

        Called when BetaGate releases via hippocampal completion signal
        (ARC-028 / receive_hippocampal_completion). Sweeps the committed
        trajectory in reverse temporal order, updating VALENCE_WANTING at
        each z_world state proportional to:
            credit_t = outcome_quality * gamma^(T - t)
        where T = trajectory length, t = step index (0 = start, T-1 = end).
        The endpoint (T-1) receives the full outcome_quality; earlier steps
        are discounted by gamma^(steps_from_end).

        No SD-006 dependency: fires synchronously on waking path at
        BetaGate release. MECH-094: waking path, hypothesis_tag=False --
        credit is assigned from the real executed trajectory, not simulation.

        No-op when:
          - use_backward_credit_sweep is False (backward compat)
          - outcome_quality < backward_sweep_min_quality (low-quality completions
            do not deserve retroactive reward assignment)
          - _committed_trajectory_buffer is None (no committed trajectory stored)
          - trajectory has no world_states (cannot map to z_world nodes)

        Silently skips valence write if ResidueConfig.valence_enabled=False.

        Args:
            outcome_quality: float in [0, 1] -- completion signal value.
                Typically hippocampal._last_completion_signal.

        Returns:
            dict: n_steps_swept (int), mean_credit (float),
                  outcome_quality (float). Empty dict when first no-op guard.
        """
        if not getattr(self.config, "use_backward_credit_sweep", False):
            return {}
        min_quality = float(getattr(self.config, "backward_sweep_min_quality", 0.6))
        if outcome_quality < min_quality:
            return {
                "n_steps_swept": 0,
                "mean_credit": 0.0,
                "outcome_quality": outcome_quality,
            }
        if self._committed_trajectory_buffer is None:
            return {
                "n_steps_swept": 0,
                "mean_credit": 0.0,
                "outcome_quality": outcome_quality,
            }
        traj = self._committed_trajectory_buffer
        world_states = traj.world_states
        if world_states is None or len(world_states) == 0:
            return {
                "n_steps_swept": 0,
                "mean_credit": 0.0,
                "outcome_quality": outcome_quality,
            }

        gamma = float(getattr(self.config, "backward_sweep_gamma", 0.9))
        T = len(world_states)
        total_credit = 0.0
        n_steps = 0
        for t in range(T - 1, -1, -1):
            steps_from_end = T - 1 - t
            credit = outcome_quality * (gamma ** steps_from_end)
            total_credit += credit
            n_steps += 1
            z_w = world_states[t]  # [batch, world_dim]
            if hasattr(self.residue_field, "update_valence"):
                self.residue_field.update_valence(z_w, VALENCE_WANTING, credit)

        mean_credit = total_credit / n_steps if n_steps > 0 else 0.0
        return {
            "n_steps_swept": n_steps,
            "mean_credit": mean_credit,
            "outcome_quality": outcome_quality,
        }

    def reset_committed_trajectory(self) -> None:
        """Per-episode reset of the MECH-290 committed trajectory buffer.

        Called from REEAgent.reset() when use_backward_credit_sweep is True.
        Clears the stored committed trajectory so a stale trajectory from the
        previous episode cannot be swept on the first completion of the new one.
        """
        self._committed_trajectory_buffer = None

    # ------------------------------------------------------------------ #
    # MECH-269 base: per-stream verisimilitude (Phase 1, 2026-04-22)      #
    # ------------------------------------------------------------------ #

    def _stream_value(
        self,
        stream_name: str,
        latent_state,
        goal_state=None,
    ) -> Optional[torch.Tensor]:
        """Resolve a stream name to its current tensor value.

        z_harm_s maps to LatentState.z_harm (sensory-discriminative harm,
        SD-010 / SD-011 naming convention). z_goal lives on GoalState, not
        LatentState. All other streams are direct attributes on LatentState.
        Returns None when the stream is disabled in this configuration.
        """
        if stream_name == "z_harm_s":
            return getattr(latent_state, "z_harm", None)
        if stream_name == "z_goal":
            if goal_state is None:
                return None
            return getattr(goal_state, "z_goal", None)
        return getattr(latent_state, stream_name, None)

    def update_per_stream_vs(
        self,
        latent_state,
        goal_state=None,
    ) -> None:
        """MECH-269 base substrate: per-stream verisimilitude scores.

        For each registered stream, compute an identity-prediction proxy
        V_s = 1 - norm(z_curr - z_prev) / (norm(z_curr) + eps), clipped
        to [0, 1] and EMA-smoothed with config.per_stream_vs_tau. On the
        first tick of an episode (or when a stream first appears), the
        previous value is cached and the score is initialised to 1.0
        (perfect verisimilitude assumed).

        No-op when config.use_per_stream_vs is False (backward compat).
        Streams absent from latent_state / goal_state are silently skipped.

        Phase 1 contract: populates self.per_stream_vs as an OBSERVABLE.
        Downstream MECH-287 trigger and MECH-284 staleness accumulator
        (Phase 2/3) will consume these scores. Forward-predictor routing
        (ReafferencePredictor / HarmForwardModel) is deferred to Phase 2.
        """
        if not getattr(self.config, "use_per_stream_vs", False):
            return
        tau = float(getattr(self.config, "per_stream_vs_tau", 0.1))
        streams = getattr(self.config, "per_stream_vs_streams", ())
        eps = 1e-6
        for stream_name in streams:
            z_curr = self._stream_value(stream_name, latent_state, goal_state)
            if z_curr is None:
                continue
            z_curr_d = z_curr.detach()
            z_prev = self._prev_stream_values.get(stream_name)
            if z_prev is None:
                # First observation: assume perfect verisimilitude, cache.
                self.per_stream_vs[stream_name] = 1.0
                self._prev_stream_values[stream_name] = z_curr_d
                continue
            denom = float(z_curr_d.norm().item()) + eps
            err = float((z_curr_d - z_prev).norm().item()) / denom
            score = max(0.0, min(1.0, 1.0 - err))
            prev_vs = self.per_stream_vs.get(stream_name, score)
            self.per_stream_vs[stream_name] = (1.0 - tau) * prev_vs + tau * score
            self._prev_stream_values[stream_name] = z_curr_d

    def reset_per_stream_vs(self) -> None:
        """Reset per-stream V_s state (call on episode boundaries).

        Clears both the cached previous stream values and the V_s scores.
        Also clears the per-region V_s state (T4, Phase 2 iii) since
        regions become meaningless across episode boundaries.
        Subsequent updates re-initialise from the next observation.
        """
        self.per_stream_vs.clear()
        self._prev_stream_values.clear()
        self.per_region_vs.clear()
        self._prev_region_stream_values.clear()

    def compute_per_stream_staleness(self) -> Dict[str, float]:
        """Aggregate region-keyed MECH-284 staleness into per-stream scalars
        for the MECH-269b VsRolloutGate (Q-040b strong reading).

        For each stream s currently tracked in per_stream_vs:
            staleness[s] = max over active anchors a where s in a.stream_mixture
                           of staleness_accumulator.lookup_by_anchor_key(a.key)
        with 0.0 returned when no qualifying anchor exists.

        Why max-over-anchors: a stream's V_s is a global readout but the
        underlying staleness lives per (scale, segment_id) region; if any
        region the stream participates in is stale, the gate should
        treat the stream as stale. Mean would dilute under multi-region
        coverage; max captures the worst staleness the stream is
        currently exposed to.

        Returns an empty dict when MECH-269 Phase 1 is off, when the
        staleness accumulator is absent, or when the anchor set is
        absent or has no active anchors. Callers (the gate) must accept
        an empty / partial dict and fall back to the raw-V_s path for
        unrepresented streams (.get(stream, 0.0) semantics).

        Pure arithmetic (max + dict membership). MECH-094: substrate-side
        readout; replay / simulation paths must not call this method.
        """
        if not self.per_stream_vs:
            return {}
        if self.staleness_accumulator is None or self.anchor_set is None:
            return {}
        active = self.anchor_set.active_anchors()
        if not active:
            return {}
        out: Dict[str, float] = {}
        for stream in self.per_stream_vs.keys():
            best = 0.0
            for anchor in active:
                mixture = anchor.key[2]
                if stream not in mixture:
                    continue
                s_val = float(
                    self.staleness_accumulator.lookup_by_anchor_key(anchor.key)
                )
                if s_val > best:
                    best = s_val
            out[stream] = best
        return out

    def update_per_region_vs(
        self,
        latent_state,
        goal_state=None,
    ) -> None:
        """MECH-269 Phase 2 (iii, T4): per-region per-stream verisimilitude.

        For each active anchor in the AnchorSet, compute per-stream V_s
        using the same identity-prediction proxy as Phase 1, scoped to
        that region's independent prev-value cache. The region key is
        (scale, segment_id) projected from the AnchorKey; stream_mixture
        is dropped for readout simplicity (a single (scale, segment_id)
        region may be reached by multiple stream_mixture families; last
        one written wins on the readout).

        No-op when use_per_region_vs is False, when use_anchor_sets is
        False (no anchor_set to query), or when the anchor_set has no
        active anchors this tick.

        Regions whose active anchor has disappeared since the previous
        update are pruned from per_region_vs and _prev_region_stream_values.

        Phase 1 identity-proxy parity: V_s = 1 - norm(z_curr - z_prev)
        / (norm(z_curr) + eps), clipped to [0, 1], EMA-smoothed with
        config.per_stream_vs_tau. On a region's first tick (cache miss),
        V_s is seeded to 1.0 and z_curr is cached.
        """
        if not getattr(self.config, "use_per_region_vs", False):
            return
        if self.anchor_set is None:
            return
        active = self.anchor_set.active_anchors()
        if not active:
            # No active regions. Prune any stale state.
            if self.per_region_vs:
                self.per_region_vs.clear()
                self._prev_region_stream_values.clear()
            return

        tau = float(getattr(self.config, "per_stream_vs_tau", 0.1))
        streams = getattr(self.config, "per_stream_vs_streams", ())
        eps = 1e-6

        active_keys = set()
        for anchor in active:
            scale, segment_id = anchor.key[0], anchor.key[1]
            region_key = (scale, segment_id)
            active_keys.add(region_key)
            region_vs = self.per_region_vs.setdefault(region_key, {})
            region_prev = self._prev_region_stream_values.setdefault(
                region_key, {}
            )
            for stream_name in streams:
                z_curr = self._stream_value(stream_name, latent_state, goal_state)
                if z_curr is None:
                    continue
                z_curr_d = z_curr.detach()
                z_prev = region_prev.get(stream_name)
                if z_prev is None:
                    region_vs[stream_name] = 1.0
                    region_prev[stream_name] = z_curr_d
                    continue
                denom = float(z_curr_d.norm().item()) + eps
                err = float((z_curr_d - z_prev).norm().item()) / denom
                score = max(0.0, min(1.0, 1.0 - err))
                prev_vs = region_vs.get(stream_name, score)
                region_vs[stream_name] = (1.0 - tau) * prev_vs + tau * score
                region_prev[stream_name] = z_curr_d

        # Prune regions whose anchor is no longer active (hysteresis
        # mark_inactive via tick_hysteresis, FIFO cap eviction, or an
        # earlier apply_invalidation_broadcasts_to_regions call this tick).
        stale = [k for k in self.per_region_vs.keys() if k not in active_keys]
        for k in stale:
            self.per_region_vs.pop(k, None)
            self._prev_region_stream_values.pop(k, None)

    def apply_invalidation_broadcasts_to_regions(
        self,
        broadcasts: List[BroadcastEvent],
        goal_payload: Optional[AnchorGoalPayload] = None,
    ) -> List[Tuple[str, str]]:
        """MECH-287 reset path: drop per_region_vs entries for broadcast-targeted
        regions and mark the matching active anchor inactive.

        For each BroadcastEvent, the region keyed on
        (source_scale, source_segment_id_old) is the outgoing region.
        We drop its per_region_vs / _prev_region_stream_values entry and
        mark_inactive any active AnchorSet anchor on that (scale,
        segment_id_old) -- this is the T3 hysteresis-shortcut reset path
        described in the design doc (k=5 hysteresis is the passive path;
        broadcasts are the explicit-reset path).

        SD-039 population layer: when goal_payload is supplied, it is
        forwarded to AnchorSet.mark_inactive so the broadcast-driven
        invalidation refreshes the outgoing anchor's payload to the
        current motivational state at the moment of invalidation. The
        existing payload is replaced; nothing is cleared (dual-trace
        preservation per the SD-039 design).

        No-op when use_per_region_vs is False. Returns the list of
        region keys actually reset.
        """
        if not getattr(self.config, "use_per_region_vs", False):
            return []
        if not broadcasts:
            return []
        reset_keys: List[Tuple[str, str]] = []
        for bcast in broadcasts:
            region_key = (bcast.source_scale, bcast.source_segment_id_old)
            popped_vs = self.per_region_vs.pop(region_key, None)
            self._prev_region_stream_values.pop(region_key, None)
            if popped_vs is not None:
                reset_keys.append(region_key)
            # Mark_inactive any active anchor on (scale, segment_id_old),
            # independent of stream_mixture. AnchorSet.mark_inactive is
            # keyed on (scale, stream_mixture) so we scan active anchors
            # for matching (scale, segment_id).
            if self.anchor_set is not None:
                for anchor in list(self.anchor_set.active_anchors(scale=bcast.source_scale)):
                    if anchor.key[1] == bcast.source_segment_id_old:
                        self.anchor_set.mark_inactive(
                            scale=anchor.key[0],
                            stream_mixture=anchor.key[2],
                            goal_payload=goal_payload,
                        )
        return reset_keys

    def reset_event_segmenter(self) -> None:
        """Reset MECH-288 event segmenter state (call on episode boundaries).

        Clears per-scale detector state, segment counters (outer.inner back to
        0.0), and drains any pending boundary events. No-op when the segmenter
        is disabled.
        """
        if self.event_segmenter is not None:
            self.event_segmenter.reset()
        self._boundary_event_queue.clear()

    def build_goal_payload(
        self,
        latent_state,
        goal_state=None,
        residue_field: Optional[ResidueField] = None,
        bla_output=None,
        current_step: Optional[int] = None,
        simulation_mode: bool = False,
    ) -> Optional[AnchorGoalPayload]:
        """SD-039 population layer: construct an AnchorGoalPayload from the
        current waking-stream signals.

        Returns None when:
          - the anchor set substrate is disabled (no consumer),
          - AnchorSetConfig.use_sd039_anchor_payload is False (master flag OFF),
          - simulation_mode is True (MECH-094 gate -- replay / DMN paths must
            NOT populate payloads from waking signals).

        Field sourcing:
          z_goal_snapshot:    GoalState.z_goal.detach().clone() when goal_state
                              is active and z_goal exists; None otherwise.
          wanting_strength:   ResidueField.evaluate_valence(z_world) read on the
                              VALENCE_WANTING channel, taken at the current
                              z_world location. 0.0 when residue / valence is
                              disabled, no z_world is available, or no centers
                              are active.
          arousal_tag:        bla_output.arousal_tag when supplied; 0.0 otherwise.
          last_vs:            Mean of self.per_stream_vs.values() at write time
                              (Phase 2 ii proxy -- the parent (scale,
                              stream_mixture) family identity is established at
                              the per-anchor write_anchor call site by
                              consume_boundary_events; the payload is shared
                              across all anchors written in this tick so we
                              record the cross-stream mean as a representative
                              scalar). None when per_stream_vs is empty.
          staleness_at_write: max staleness from self.staleness_accumulator.snapshot()
                              when the accumulator is enabled and non-empty;
                              None otherwise. Region-keyed, but the payload is
                              shared across all anchors written this tick, so
                              the maximum across regions is the most
                              informative scalar (downstream MECH-292 ranking
                              cares about whether *any* preserved trace was
                              already heavily stale at write time).
          payload_written_step: current_step when supplied; else falls back to
                              the AnchorSet's internal monotonic _tick. Phase 2
                              stand-in for trace age.

        Intended caller: REEAgent.sense() (waking observation stream). The
        simulation_mode argument is the MECH-094 gate honoured by the caller;
        REEAgent.sense() always passes simulation_mode=False. Replay paths
        that ever begin to consume this helper must pass simulation_mode=True.
        """
        if self.anchor_set is None:
            return None
        cfg = getattr(self.anchor_set, "config", None)
        if cfg is None or not getattr(cfg, "use_sd039_anchor_payload", False):
            return None
        if simulation_mode:
            return None

        z_goal_snap = None
        if goal_state is not None:
            z_goal = getattr(goal_state, "z_goal", None)
            try:
                active = bool(goal_state.is_active())
            except Exception:
                active = z_goal is not None
            if active and z_goal is not None:
                z_goal_snap = z_goal.detach().clone()

        wanting_strength = 0.0
        z_world = getattr(latent_state, "z_world", None)
        if (
            residue_field is not None
            and z_world is not None
            and getattr(residue_field, "evaluate_valence", None) is not None
        ):
            try:
                valence = residue_field.evaluate_valence(z_world)
                if valence is not None and valence.numel() > 0:
                    wanting_strength = float(
                        valence[..., VALENCE_WANTING].mean().item()
                    )
            except Exception:
                wanting_strength = 0.0

        arousal_tag = 0.0
        if bla_output is not None:
            arousal_tag = float(getattr(bla_output, "arousal_tag", 0.0))

        last_vs: Optional[float] = None
        if self.per_stream_vs:
            vals = list(self.per_stream_vs.values())
            if vals:
                last_vs = float(sum(vals) / len(vals))

        staleness_at_write: Optional[float] = None
        if self.staleness_accumulator is not None:
            try:
                snap = self.staleness_accumulator.snapshot()
                if snap:
                    staleness_at_write = float(max(snap.values()))
            except Exception:
                staleness_at_write = None

        if current_step is None:
            current_step = int(getattr(self.anchor_set, "_tick", 0))

        return AnchorGoalPayload(
            z_goal_snapshot=z_goal_snap,
            wanting_strength=wanting_strength,
            arousal_tag=arousal_tag,
            last_vs=last_vs,
            staleness_at_write=staleness_at_write,
            payload_written_step=int(current_step),
        )

    def tick_anchor_set(
        self,
        latent_state,
        events: List[BoundaryEvent],
        goal_payload: Optional[AnchorGoalPayload] = None,
    ) -> None:
        """MECH-269 Phase 2 (ii): consume BoundaryEvents and advance hysteresis.

        Installs / remaps anchors for each BoundaryEvent using the current
        z_world snapshot and a stream_mixture derived from the live
        per_stream_vs keys. Advances per-tick hysteresis counters on all
        active anchors against the current per_stream_vs scores. No-op
        when use_anchor_sets is False.

        MECH-284 Phase 3 integration (when use_staleness_accumulator is
        True): between consume_boundary_events and tick_hysteresis,
        integrate this tick's queued MECH-287 BroadcastEvents across the
        post-consume active anchor set and apply the per-tick leak. When
        use_mech284_hysteresis is additionally True, tick_hysteresis is
        given a staleness_lookup callable reading from the accumulator
        (projection: AnchorKey -> RegionKey -> staleness). With only
        use_staleness_accumulator on, the accumulator is populated as a
        diagnostic but hysteresis stays on the internal proxy. This
        ordering (consume -> integrate -> hysteresis) means this-tick
        broadcasts affect this-tick V_s_anchor checks.

        SD-039 population layer: when goal_payload is supplied, it is
        forwarded to consume_boundary_events so newly-installed and
        dual-trace-remapped anchors carry the current motivational
        snapshot. Hysteresis-fired mark_inactive does NOT refresh the
        payload (preserves the prior payload as the "cause-of-blockage"
        trace per dual-trace semantics).

        Intended caller: REEAgent.sense(), invoked after the event segmenter
        queues BoundaryEvents and after update_per_stream_vs has populated
        per_stream_vs for the current tick.
        """
        if self.anchor_set is None:
            return
        z_world = getattr(latent_state, "z_world", None)
        mixture = tuple(sorted(self.per_stream_vs.keys()))
        if events:
            self.anchor_set.consume_boundary_events(
                events=events,
                z_world=z_world,
                stream_mixture=mixture,
                goal_payload=goal_payload,
            )

        if self.staleness_accumulator is not None:
            broadcasts = list(self._broadcast_event_queue)
            active = self.anchor_set.active_anchors()
            if broadcasts and active:
                self.staleness_accumulator.integrate(broadcasts, active)
            self.staleness_accumulator.tick_leak()

        staleness_lookup = None
        if (
            getattr(self.config, "use_mech284_hysteresis", False)
            and self.staleness_accumulator is not None
        ):
            staleness_lookup = self.staleness_accumulator.lookup_by_anchor_key
        self.anchor_set.tick_hysteresis(
            self.per_stream_vs, staleness_lookup=staleness_lookup
        )

    def reset_anchor_set(self) -> None:
        """Per-episode reset of the MECH-269 anchor set. No-op when disabled."""
        if self.anchor_set is not None:
            self.anchor_set.reset()

    def integrate_staleness(
        self,
        broadcasts: List[BroadcastEvent],
    ) -> None:
        """MECH-284 Phase 3: credit broadcasts across the active anchor set.

        Called from REEAgent.sense() after the MECH-287 trigger has
        queued BroadcastEvents for this tick and after
        update_per_stream_vs / tick_anchor_set have populated the current
        active anchor set. The accumulator:
          - credits each broadcast's strength across active anchors via
            the configured attribution_weight ("equal" or "stream_overlap")
          - applies a per-tick leak to all region entries
        No-op when use_staleness_accumulator is False. When the
        AnchorSet is disabled the integration becomes a leak-only tick.
        """
        if self.staleness_accumulator is None:
            return
        active = (
            self.anchor_set.active_anchors()
            if self.anchor_set is not None
            else []
        )
        if broadcasts and active:
            self.staleness_accumulator.integrate(broadcasts, active)
        self.staleness_accumulator.tick_leak()

    def reset_staleness_accumulator(self) -> None:
        """Per-episode reset of the MECH-284 staleness accumulator. No-op when disabled."""
        if self.staleness_accumulator is not None:
            self.staleness_accumulator.reset()

    def rank_ghost_goals(
        self,
        current_z_goal: Optional[torch.Tensor],
        persistence_appraisal: Optional[PersistenceAppraisal] = None,
    ) -> List[GhostGoalBankEntry]:
        """MECH-292: return the ranked ghost-goal bank for the current z_goal.

        Read-only derived view over the SD-039 dual-trace anchor pool.
        Returns [] when use_mech292_ghost_bank is False, when
        current_z_goal is None, or when no anchor clears the
        goal_match_floor. MECH-340: when use_persistence_efficacy_gate is
        True, anchors below the persistence license floor are excluded
        (see PersistenceAppraisal). See ree_core/hippocampal/ghost_goal_bank.py
        for the ranking formula.

        Consumers: MECH-293 (waking ghost-goal probe search) is the
        first behavioural consumer; diagnostic / replay-prioritisation
        consumers may call it directly.
        """
        if self.ghost_goal_bank is None:
            return []
        return self.ghost_goal_bank.rank(
            current_z_goal,
            persistence_appraisal=persistence_appraisal,
        )

    def reset_ghost_goal_bank(self) -> None:
        """Per-episode reset of the MECH-292 ghost-goal bank diagnostics.

        No-op when use_mech292_ghost_bank is False. The bank is otherwise
        stateless across calls -- the source-of-truth anchor pool lives
        in AnchorSet, and AnchorSet.reset() is called separately by
        reset_anchor_set() on the episode boundary.
        """
        if self.ghost_goal_bank is not None:
            self.ghost_goal_bank.reset()

    def forward(
        self,
        z_world: torch.Tensor,
        z_self: Optional[torch.Tensor] = None,
        num_candidates: Optional[int] = None,
        e1_prior: Optional[torch.Tensor] = None,
        action_bias: Optional[torch.Tensor] = None,
        operating_mode: Optional[Dict[str, float]] = None,
        current_z_goal: Optional[torch.Tensor] = None,
    ) -> List[Trajectory]:
        """Forward pass: propose trajectories from z_world."""
        return self.propose_trajectories(
            z_world, z_self=z_self, num_candidates=num_candidates,
            e1_prior=e1_prior, action_bias=action_bias,
            operating_mode=operating_mode,
            current_z_goal=current_z_goal,
        )
