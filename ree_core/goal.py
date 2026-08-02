"""
Goal state representation for REE V3.

Implements z_goal_latent: a slow-decay attractor in z_world space, updated
when benefit fires (MECH-112 wanting), maintained by E1 LSTM recurrently
(MECH-116 frontal working memory).

Claims: MECH-112, MECH-116, MECH-117, ARC-032
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional


@dataclass
class GoalConfig:
    """Configuration for z_goal persistent goal representation."""

    # Must match world_dim in LatentStackConfig
    goal_dim: int = 32

    # Slow attractor update rate when benefit fires
    # half-life ~ 1/alpha_goal reward-contact steps
    alpha_goal: float = 0.05

    # Very slow decay between benefit events
    # half-life ~ log(2)/decay_goal ~ 139 steps at 0.005
    decay_goal: float = 0.005

    # Minimum benefit_exposure to trigger z_goal update
    benefit_threshold: float = 0.1

    # Weight of goal proximity in trajectory scoring (subtracted from cost)
    goal_weight: float = 1.0

    # SD-012: drive modulation weight for z_goal update.
    # effective_benefit = benefit_exposure * (1.0 + drive_weight * drive_level)
    # drive_level = 1.0 - energy (obs_body[3]).
    # 0.0 disables drive modulation; 2.0 is the validated default for goal seeding.
    # Set to 0.0 explicitly for ablation baselines.
    drive_weight: float = 2.0

    # SD-012 sustained-drive amendment (goal_pipeline:GAP-3, Option 1).
    # EMA smoothing factor for the drive_level used in the SD-012 multiplier:
    #   trace_t = (1 - drive_ema_alpha) * trace_{t-1} + drive_ema_alpha * drive_level
    # Motivation: instantaneous drive_level collapses to ~0.005 the step a
    # resource is consumed (energy resets toward 1.0), so the multiplier
    # (1 + drive_weight * drive_level) ~ 1.0 and cancels the SD-012 benefit
    # amplification at exactly the contact events where seeding must fire
    # (EXQ-536a: H_b_threshold never crossed, mean drive on contact 0.005).
    # A slow trace keeps the multiplier elevated across the consummatory pulse
    # (Berridge/Robinson sustained anticipatory wanting).
    # drive_ema_alpha = 1.0 (default) -> trace == drive_level every step,
    #   regardless of init -> BIT-IDENTICAL to pre-amendment behaviour (OFF).
    # 0.02 ~ 35-step half-life (lit-anchored: wanting_liking synthesis 30-60
    #   step window). Lower alpha = slower / more sustained. The trace is
    #   zero-initialised, so alpha < 1.0 has a ~1/alpha-step cold-start
    #   transient that underestimates drive early in an episode (a known,
    #   accepted confound the discriminative sweep accounts for).
    drive_ema_alpha: float = 1.0

    # SD-012 sustained-drive amendment (goal_pipeline:GAP-3, Option 2).
    # Insatiability floor applied to drive_level BEFORE the EMA update:
    #   drive_level_floored = max(drive_level, drive_floor)
    # Motivation: even with Option 1 EMA, the drive stays near-zero throughout
    # the episode when the agent remains well-fed (EXQ-582: all alphas gave
    # drive_trace_at_contact ~0.0002-0.005 because drive_level was low all along,
    # not just at the consummatory step). A floor guarantees a minimum multiplier
    # contribution at every contact regardless of satiation level.
    # drive_floor = 0.0 (default) -> no floor, bit-identical to pre-amendment
    #   behaviour when combined with drive_ema_alpha=1.0.
    # drive_floor = 0.9 -> effective_benefit >= benefit_exposure * (1 + 2.0*0.9)
    #   = benefit_exposure * 2.8 (first-PASS arm for EXQ-582a given the regime's
    #   benefit_exposure ~0.03 at first contact with nociception_ema_alpha=0.1).
    # Can combine with Option 1 EMA: the floor is applied to drive_level before
    # the EMA update, so the trace stays >= drive_floor in steady state.
    drive_floor: float = 0.0

    # Whether E1 receives z_goal as conditioning input (MECH-116)
    e1_goal_conditioned: bool = True

    # Master switch -- disabled by default (ablation baseline)
    z_goal_enabled: bool = False

    # MECH-187: gain multiplier on z_goal seeding signal.
    # Applied before drive modulation: effective_benefit = benefit_exposure * z_goal_seeding_gain * (...)
    # 1.0 = no change (default, fully backward compatible).
    # < 1.0 suppresses seeding (5-HT serotonergic inhibition of incentive salience).
    # > 1.0 elevates seeding (disinhibition or pharmacological augmentation).
    # Empirical range from Korte et al. 2016: suppression ~x0.6-0.8, elevation ~x1.5-2.5.
    z_goal_seeding_gain: float = 1.0

    # MECH-186: serotonergic benefit terrain maintenance (valence_wanting floor).
    # When > 0, the z_goal norm is prevented from decaying below this value.
    # Simulates tonic serotonergic support maintaining minimum wanting tone.
    # Default None/0.0 = disabled (no floor, backward-compatible).
    # Set to 0.05 for the MECH-186 floor-maintained condition.
    valence_wanting_floor: float = 0.0

    # MECH-188: PFC top-down z_goal injection (constant floor on effective z_goal norm
    # during action selection only -- does NOT modify the persistent z_goal attractor).
    # Simulates DRN-mPFC serotonergic top-down goal persistence (Miyazaki et al. 2020):
    # when terrain-based seeding has failed (LONG_HORIZON depression attractor), an
    # external PFC signal can maintain goal representation.
    # z_goal_inject=0.0 disables (default, fully backward compatible).
    # z_goal_inject=0.3 applies a constant norm floor of 0.3 to z_goal during
    # agent.select_action() -- does not affect update() or z_goal decay.
    # Used by EXQ-253 (condition B) to test whether top-down injection suffices to
    # maintain PLANNED/HABIT behavioral gap when bottom-up terrain seeding has collapsed.
    z_goal_inject: float = 0.0

    # SD-057: object-bound incentive-salience layer (GAP-7 L2-L3-L4).
    # Master switch -- default False, bit-identical OFF (legacy single-attractor
    # seeding from z_resource/z_world). When True, GoalState owns an
    # IncentiveTokenBank: benefit at contact binds to the SD-049 resource-type
    # tag (L2 MECH-344), each type accrues a slow-decay revaluable base_value
    # token (L3 MECH-345), and z_goal is seeded FROM the most-wanted object's
    # stored embedding (L4 MECH-346; MECH-230 amend) rather than the raw
    # last-contacted latent.
    use_incentive_token_bank: bool = False

    # SD-057 L3: per-object base_value slow decay per update() call (matches
    # decay_goal cadence). 0.0 = no decay (tokens persist until revalued).
    incentive_decay: float = 0.005

    # SD-057 L3: EMA rate for revaluation of base_value toward received benefit
    # on contact (Balleine/Dickinson 1998: revaluable, not write-once).
    incentive_value_alpha: float = 0.1

    # SD-057 L3: relocated drive_weight for the at-recall wanting amplitude
    # wanting[k] = base_value[k] * (1 + incentive_drive_kappa_weight * drive_axis[k])
    # (Zhang 2009 V = r * kappa(drive)). Default mirrors GoalConfig.drive_weight.
    incentive_drive_kappa_weight: float = 2.0

    # SD-049-PHASE-2 drive-coupling amend (failure_autopsy_V3-EXQ-514r, MECH-436):
    # no-op-default scale on the effective drive->score coupling kappa. The
    # effective kappa is incentive_drive_kappa_weight * incentive_drive_kappa_scale.
    # At 1.0 (default) wanting() is byte-identical. The 514r autopsy showed the
    # in-run per-axis drive spread (~0.006) times the fixed kappa is swamped by
    # real object base_value gaps (>0.5 on seeds 45/46/47), so even an overshoot
    # drive cannot flip most_wanted at the current kappa -- scaling kappa is the
    # load-bearing half (paired with env standing differential depletion) that lets
    # a realistic drive spread compete with the real object-value landscape.
    incentive_drive_kappa_scale: float = 1.0

    # SD-057 L3: when True (default) wanting uses per-axis drive (SD-049
    # hunger/thirst/curiosity) so wanting is drive-specific / identity-matched
    # (specific PIT). When False, the scalar drive_level is applied uniformly.
    incentive_use_per_axis_drive: bool = True

    # SD-057 phase-2 L6 (MECH-347): cue-recall path master switch. When set
    # (requires use_incentive_token_bank), a perceived cue/object type can
    # retrieve its incentive token and nudge z_goal toward that object's stored
    # embedding BEFORE any benefit pulse (cue-triggered wanting; specific PIT).
    # Default False -> bit-identical (no cue path).
    use_cue_recall: bool = False

    # SD-057 phase-2 L6: z_goal cue-pull strength per cue-recall event. Separate
    # from alpha_goal (the benefit-driven seed rate) -- the cue nudge is a
    # weaker, pre-consummatory pull. effective pull = cue_recall_gain * clamped
    # wanting amplitude. Default 0.05 (matches alpha_goal magnitude).
    cue_recall_gain: float = 0.05

    # SD-057 phase-2 L6: minimum perceived-cue proximity for the AUTOMATIC
    # harness cue-perception path to fire (the explicit primitive ignores it).
    # Below this, no cue is considered perceived. Default 0.0 (any perception).
    cue_recall_min_proximity: float = 0.0

    # MECH-189: super-ordinal goal-anchor ContextMemory writes substrate
    # (infant_substrate:GAP-11 / DEV-NEED-006 / DEV-NEED-024).
    # Master switch -- default False, bit-identical OFF. When True, the AGENT
    # owns a cross-episode-persistent SuperOrdinalGoalMemory (cue-indexed:
    # key = z_world context, value = z_goal anchor). During the child phase
    # (write_enabled), a high-salience benefit contact occurring in a
    # high-contextual-complexity context writes the current z_goal as a
    # super-ordinal anchor; in adult episodes the store seeds z_goal toward the
    # retrieved childhood-formed anchor when the live z_goal is below the seed
    # floor (the "super-ordinal goals bias adult z_goal seeding" readout).
    use_super_ordinal_goal_anchors: bool = False

    # MECH-189: number of cue-indexed anchor slots (FINST-like capacity; the
    # childhood-formed super-ordinal goal hierarchy is small).
    super_ordinal_n_slots: int = 16

    # MECH-189 WRITE gate (a) high salience: minimum benefit signal (drive-
    # modulated benefit_exposure) for a contact to qualify as a "large benefit
    # spike" worth writing a super-ordinal anchor. Much higher than
    # benefit_threshold (0.1) so routine contacts do not write -- only
    # high-salience (e.g. transient-benefit-patch) contacts.
    super_ordinal_salience_threshold: float = 0.5

    # MECH-189 WRITE gate (b) high contextual complexity -- the DEV-NEED-024
    # open question ("what contextual-complexity threshold triggers a write").
    # Mode is PLUGGABLE so a future lit-pull / experiment can adjudicate:
    #   "novelty"  -- complexity = 1 - max cosine(z_world, occupied anchor keys);
    #                 self-contained, bootstraps on an empty store, and naturally
    #                 suppresses adult-routine writes (the spec's selective
    #                 neoteny). Default.
    #   "external" -- complexity is supplied by the caller (context_complexity
    #                 arg to the write site), letting an experiment drive it from
    #                 E1 cue-context entropy / prediction-error without coupling
    #                 the substrate to those optional channels.
    super_ordinal_complexity_mode: str = "novelty"

    # MECH-189 WRITE gate (b): minimum contextual-complexity value (in [0, 1])
    # required to write. Novel/empty-store contexts (novelty ~ 1.0) write;
    # contexts already covered by an anchor (low novelty) do not.
    super_ordinal_complexity_threshold: float = 0.3

    # MECH-189 WRITE: when a new context is within this cosine similarity of an
    # existing occupied anchor key, the write REINFORCES that anchor (EMA-blends
    # key+value, raises strength) instead of allocating a new slot. Above-merge
    # contexts get a fresh slot (empty, else the weakest by strength).
    super_ordinal_merge_similarity: float = 0.8

    # MECH-189 WRITE: EMA blend rate when writing/reinforcing an anchor slot
    # (key and value). Mirrors the ContextMemory write blend magnitude.
    super_ordinal_write_alpha: float = 0.3

    # MECH-189 READ (adult seeding): only seed from a super-ordinal anchor when
    # the live z_goal norm is below this floor (adult novel-context seeding;
    # matches the DEV-NEED-006 z_goal.norm() > 0.4 gate -- seed when the agent
    # has no strong episodic goal of its own yet).
    super_ordinal_seed_below_norm: float = 0.4

    # MECH-189 READ: minimum retrieval cosine match for a stored anchor to seed
    # the current context. Below this, the childhood anchor is not relevant to
    # the current adult context and does not seed.
    super_ordinal_seed_match_threshold: float = 0.3

    # MECH-189 READ: cue-pull strength applied to z_goal toward the retrieved
    # super-ordinal anchor per seeding tick (GoalState.cue_pull fraction).
    super_ordinal_seed_strength: float = 0.1

    # SD-077: common-mode-invariant (centered) super-ordinal cue key.
    # Master switch -- default False, bit-identical OFF (raw-z_world cosine, the
    # pre-SD-077 behaviour: no baseline is allocated and no baseline arithmetic
    # runs).
    #
    # Problem this solves (measured 2026-07-21 on the V3-EXQ-669b Stage-0
    # forced-feed nursery, 155 contexts): under SD-008 z_world
    # under-differentiation the untrained encoder maps every context into a
    # narrow common-mode cone -- pairwise cosine min 0.9641 / mean 0.9898, with
    # ||mean(z_world)|| / mean||z_world|| = 0.9949, i.e. essentially one shared
    # direction. The SAME contexts are richly separated in the raw world
    # observation (pairwise cosine min 0.216 / mean 0.608; 90.7% of pairs below
    # 0.8), so the diversity exists and the encoder buries it.
    #
    # Consequence on the raw-cosine store: every context lands within
    # merge_similarity of anchor 0, so all 155 contacts REINFORCE one slot and
    # anchor_count saturates at 1 -- the V3-EXQ-669b R3 readiness-gate failure
    # that self-routed substrate_not_ready_requeue and left MECH-329 / MECH-189
    # unmeasurable.
    #
    # Threshold tuning CANNOT fix this, and that is provable rather than merely
    # observed: contextual_complexity = 1 - best_cosine, and best_cosine >= 0.9641
    # everywhere in the nursery, so complexity <= 0.036 for EVERY context under
    # ANY threshold setting -- strictly below 669b's pre-registered
    # COMPLEXITY_MARGIN of 0.05. The C3 criterion is unsatisfiable by
    # construction on a raw-z_world key. Only changing the KEY SPACE helps.
    # (Note also that the failed run's docstring suggested a LOWER
    # merge_similarity; that is the wrong sign -- it moves more contexts into the
    # REINFORCE branch and makes saturation strictly worse.)
    #
    # Fix (the SD-066 pattern, which solved the identical common-mode signature
    # for the SD-051 ConditionedSafetyStore readout): maintain a slow EMA
    # baseline of presented z_world contexts and do all cue arithmetic on the
    # centered residual z_world - baseline. Measured on the same nursery, this
    # restores the full range: residual pairwise cosine spans -0.760 to 1.000
    # (97.7% of pairs below 0.8), yielding 18-66 distinct anchors instead of 1,
    # and complexity recovers the full [0, 1] interval.
    super_ordinal_cue_centering: bool = False

    # SD-077: EMA rate for the common-mode baseline, matching SD-066's validated
    # default. The baseline is lazily seeded from the first observed context (so
    # there is no zero-init cold-start transient) and is advanced only on waking
    # contexts -- simulation_mode ticks do not move it (MECH-094).
    super_ordinal_cue_baseline_alpha: float = 0.02

    # SD-093 / MECH-426 (progress_velocity_maintenance): rate-of-progress
    # (velocity) effort/persistence modulator -- the temporal derivative of
    # the on-path goal_proximity() estimate (Carver & Scheier 1990
    # second-order "velocity" control loop). Master switch -- default False,
    # bit-identical OFF: GoalState.record_progress() no-ops (history is never
    # populated) and progress_velocity_effort_modulation always returns 0.0.
    #
    # CRITICAL MODELLING CAVEAT (Carver&Scheier coasting, per the claim's own
    # notes): above-reference progress (positive velocity) produces positive
    # affect that REDUCES effort on the current goal and licenses
    # redeployment -- it must NOT be modelled as same-goal reinforcement. This
    # substrate therefore modulates E3's commit-threshold EFFORT/PERSISTENCE
    # pressure (mirroring the SD-011 urgency mechanism), never goal VALUE /
    # trajectory score (that channel stays exactly compute_goal_score() /
    # goal_proximity(), untouched by this flag).
    use_progress_velocity_effort_modulation: bool = False

    # SD-093 / MECH-426: rolling-window length (in E3-tick record_progress()
    # calls) over which the goal_proximity derivative is taken. velocity =
    # (proximity_now - proximity_{t-window}) / window. window=1 degenerates
    # to a since-last-check derivative. Minimum enforced is 2 (need at least
    # 2 samples for a derivative); values < 2 are clamped up at construction.
    progress_velocity_window: int = 5

    # SD-093 / MECH-426: gain applied to velocity before it becomes the
    # effort-modulation signal read by E3.select() (see
    # GoalState.progress_velocity_effort_modulation). 1.0 = no rescaling.
    progress_velocity_effort_gain: float = 1.0

    # SD-093 / MECH-426: saturation cap on the effort-modulation signal
    # (symmetric, applies to both the stalling/boost and
    # coasting/ease-off directions), mirroring E3Config.urgency_max's role
    # for the SD-011 urgency mechanism. Prevents a runaway velocity reading
    # from collapsing (or inflating) the effective commit threshold.
    progress_velocity_effort_max: float = 0.3

    # SD-092: cross-level subgoal credit (MECH-427 maintenance-direction /
    # MECH-428 formation-direction). Master switch -- default False,
    # bit-identical OFF. When True, GoalState additionally owns a lazily
    # allocated PARENT (superordinate) attractor `_z_goal_parent`, distinct
    # from the existing `_z_goal` (read as the child/subgoal level under this
    # scheme). A discrete subgoal-attainment event calls
    # GoalState.credit_subgoal_attainment(...) to pull the parent attractor
    # toward the attained subgoal's representation -- the cross-LEVEL
    # complement to MECH-217's within-level backward_credit_sweep /
    # spread_reverse_replay_wanting (HippocampalModule). MECH-427 (an
    # already-seeded parent gets reinforced) and MECH-428 (a near-zero parent
    # gets bootstrapped) are the SAME primitive applied to different starting
    # states of _z_goal_parent -- no separate formation-mode code path.
    # See docs/architecture/sd_092_cross_level_subgoal_credit.md.
    use_hierarchical_goal_credit: bool = False

    # SD-092: EMA pull rate applied to the parent attractor per credit event,
    # scaled by the caller-supplied `credit` magnitude (mirrors alpha_goal,
    # one hierarchy level up). effective_pull = min(1.0, parent_goal_alpha *
    # credit).
    parent_goal_alpha: float = 0.05

    # SD-092: slow per-update() decay of the parent attractor between credit
    # events (mirrors decay_goal). Applied only when use_hierarchical_goal_credit
    # is True and the parent attractor has been allocated.
    parent_goal_decay: float = 0.005

    # SD-092: minimum credit magnitude required for credit_subgoal_attainment
    # to apply a pull. 0.0 (default) = any positive credit applies (matches
    # the "no-op guard defaults to a no-op" convention used elsewhere in this
    # config; the caller decides what counts as a qualifying attainment).
    subgoal_credit_min: float = 0.0


class SuperOrdinalGoalMemory:
    """MECH-189: cross-episode-persistent, cue-indexed super-ordinal goal-anchor
    store -- the "ContextMemory writes substrate" the infant_substrate:GAP-11 /
    DEV-NEED-006 retest waits on.

    The single store missing from V3 for MECH-189: the IncentiveTokenBank
    (SD-057) is per-object and per-episode (GoalState.reset() clears it); the
    ghost-goal bank (MECH-292) is over hippocampal anchors. Neither is the
    cross-episode super-ordinal z_goal store the claim describes:

        "high-salience benefit contacts under high contextual complexity are
         written to persistent cue-indexed ContextMemory as super-ordinal goal
         anchors that bias adult z_goal seeding across novel episodes."

    Cue-indexed = each slot keys on a z_world CONTEXT and stores a z_goal ANCHOR
    (value). WRITE is gated on the MECH-189 conjunction (high salience AND high
    contextual complexity) and is permitted only while write_enabled (the child
    phase; the curriculum freezes it at child->adult transition). READ retrieves
    the best-matching anchor for the current context to seed adult z_goal.

    AGENT-OWNED and NOT reset per episode (the persistence that makes a
    super-ordinal goal hierarchy distinct from an episodic z_goal). Pure stateful
    tensor store + cosine arithmetic -- no nn.Module, no trainable parameters, no
    gradient flow. MECH-094: writes are gated on simulation_mode at the call
    site (replay/DMN must not form super-ordinal anchors).
    """

    def __init__(
        self,
        config: GoalConfig,
        context_dim: int,
        device: torch.device,
    ) -> None:
        self.config = config
        self.device = device
        self.context_dim = int(context_dim)
        self.goal_dim = int(config.goal_dim)
        n = int(config.super_ordinal_n_slots)
        self.n_slots = n
        # Cue keys (z_world context) and anchor values (z_goal).
        self._keys = torch.zeros(n, self.context_dim, device=device)
        self._values = torch.zeros(n, self.goal_dim, device=device)
        self._strength = torch.zeros(n, device=device)
        self._occupied = torch.zeros(n, dtype=torch.bool, device=device)
        # Child-phase write window. The curriculum freezes writes for adult
        # measurement via REEAgent.set_super_ordinal_write_enabled(False).
        self.write_enabled: bool = True
        # SD-077: common-mode baseline for the centered cue key. None until the
        # first waking context is observed (lazy seed -> no zero-init transient).
        # Stays None forever when centering is disabled, so the OFF path
        # allocates nothing and executes no baseline arithmetic.
        self.centering: bool = bool(
            getattr(config, "super_ordinal_cue_centering", False)
        )
        self._baseline: Optional[torch.Tensor] = None
        # Diagnostics (per-agent-lifetime; reset only on reset_anchors()).
        self._n_writes = 0
        self._n_reinforce = 0
        self._n_allocate = 0
        self._n_seeds = 0
        self._last_complexity = -1.0
        self._last_salience = -1.0
        self._last_seed_match = -1.0

    @staticmethod
    def _row(t: torch.Tensor) -> torch.Tensor:
        """Coerce a [d] / [1, d] / [b, d] tensor to a single [d] row (mean over
        a batch)."""
        t = t.detach()
        if t.dim() == 2:
            t = t.mean(dim=0)
        elif t.dim() != 1:
            t = t.reshape(-1)
        return t

    def _occupied_idx(self) -> torch.Tensor:
        return torch.nonzero(self._occupied, as_tuple=False).reshape(-1)

    def observe(self, z_world_context: torch.Tensor,
                simulation_mode: bool = False) -> None:
        """SD-077: advance the common-mode baseline with a presented context.

        No-op when centering is disabled (the OFF path must be bit-identical) or
        under simulation_mode (MECH-094: replay/DMN contexts must not shape the
        waking cue geometry). Lazily seeds the baseline from the first waking
        context, so there is no zero-init cold-start transient.
        """
        if not self.centering or simulation_mode:
            return
        key = self._row(z_world_context)
        if self._baseline is None:
            self._baseline = key.clone()
            return
        a = float(self.config.super_ordinal_cue_baseline_alpha)
        self._baseline = (1.0 - a) * self._baseline + a * key

    def _centered(self, t: torch.Tensor) -> torch.Tensor:
        """SD-077: subtract the common-mode baseline. Identity when centering is
        disabled or the baseline has not been seeded yet."""
        if not self.centering or self._baseline is None:
            return t
        return t - self._baseline

    def _best_match(self, query_key: torch.Tensor):
        """Return (best_slot_idx, best_cosine) over occupied slots, or
        (None, -1.0) when the store is empty.

        SD-077: when centering is enabled the cosine is taken on the centered
        residual of BOTH sides (query and stored keys). Keys are stored RAW and
        centered at comparison time -- deliberately, so that a drifting baseline
        moves query and stored keys together and never leaves the store
        internally inconsistent (which storing pre-centered keys would).
        """
        occ = self._occupied_idx()
        if occ.numel() == 0:
            return None, -1.0
        q = F.normalize(self._centered(query_key).unsqueeze(0), dim=-1)
        k = F.normalize(self._centered(self._keys[occ]), dim=-1)
        sims = (k @ q.t()).reshape(-1)
        j = int(torch.argmax(sims).item())
        return int(occ[j].item()), float(sims[j].item())

    def contextual_complexity(self, z_world_context: torch.Tensor) -> float:
        """MECH-189 WRITE gate (b), "novelty" mode: complexity = 1 - max cosine
        similarity of the current context to any occupied anchor key, in [0, 1].
        Empty store -> 1.0 (maximally novel; first contacts bootstrap the
        hierarchy). A context already covered by an anchor -> low complexity ->
        no write (adult-routine stability)."""
        key = self._row(z_world_context)
        _, best = self._best_match(key)
        if best < 0.0:
            return 1.0
        return float(max(0.0, min(1.0, 1.0 - best)))

    def write(
        self,
        z_world_context: torch.Tensor,
        z_goal_anchor: torch.Tensor,
        salience: float,
        context_complexity: Optional[float] = None,
        simulation_mode: bool = False,
    ) -> bool:
        """MECH-189 child-phase write. Returns True iff an anchor was written.

        Gated on: write_enabled (child phase) AND not simulation_mode (MECH-094)
        AND the conjunction salience >= super_ordinal_salience_threshold AND
        complexity >= super_ordinal_complexity_threshold. complexity is taken
        from `context_complexity` when supplied (the "external" mode hook), else
        computed via the self-contained novelty proxy.

        Reinforces the nearest anchor when the context is within
        super_ordinal_merge_similarity of it (EMA blend, raise strength); else
        allocates a fresh slot (an empty one, else the weakest by strength).
        """
        # SD-077: advance the common-mode baseline on every waking context,
        # BEFORE the write_enabled gate -- the baseline is cue geometry, not
        # anchor content, so it must keep tracking the context distribution
        # through the adult (write-frozen) phase that reads from the store.
        # No-op when centering is disabled.
        self.observe(z_world_context, simulation_mode=simulation_mode)
        if simulation_mode or not self.write_enabled:
            return False
        key = self._row(z_world_context)
        val = self._row(z_goal_anchor)
        # Gate (b): contextual complexity.
        mode = getattr(self.config, "super_ordinal_complexity_mode", "novelty")
        if mode == "external" and context_complexity is not None:
            complexity = float(context_complexity)
        else:
            complexity = self.contextual_complexity(key)
        self._last_salience = float(salience)
        self._last_complexity = float(complexity)
        best_idx, best_sim = self._best_match(key)
        a = float(self.config.super_ordinal_write_alpha)

        # Gate (a) high salience is required for ANY write (allocate OR reinforce).
        if salience < self.config.super_ordinal_salience_threshold:
            return False

        # REINFORCE: a recurring high-salience contact in an EXISTING anchor's
        # region strengthens that super-ordinal goal toward the now-matured
        # z_goal (EMA blend, raise strength). Gate (b) contextual complexity does
        # NOT apply here -- complexity governs the FORMATION of a NEW anchor, not
        # the reinforcement of one that already exists. (Without this, the anchor
        # would freeze at the tiny z_goal captured at its first contact and never
        # learn the matured childhood meta-goal.)
        if (
            best_idx is not None
            and best_sim >= self.config.super_ordinal_merge_similarity
        ):
            self._keys[best_idx] = (1.0 - a) * self._keys[best_idx] + a * key
            self._values[best_idx] = (1.0 - a) * self._values[best_idx] + a * val
            self._strength[best_idx] = self._strength[best_idx] + 1.0
            self._n_reinforce += 1
            self._n_writes += 1
            return True

        # ALLOCATE a NEW super-ordinal anchor: gate (b) high contextual complexity
        # is required (a genuinely novel/rich context warrants a new meta-goal).
        if complexity < self.config.super_ordinal_complexity_threshold:
            return False
        empty = torch.nonzero(~self._occupied, as_tuple=False).reshape(-1)
        slot = int(empty[0].item()) if empty.numel() > 0 else int(
            torch.argmin(self._strength).item()
        )
        self._keys[slot] = key
        self._values[slot] = val
        self._strength[slot] = 1.0
        self._occupied[slot] = True
        self._n_allocate += 1
        self._n_writes += 1
        return True

    def retrieve(self, z_world_query: torch.Tensor,
                 simulation_mode: bool = False):
        """MECH-189 READ. Return (z_goal_anchor [1, goal_dim], match_cosine,
        slot_idx) for the best-matching anchor, or None when the store is empty.
        Does not mutate the anchor bank.

        SD-077: the query advances the common-mode baseline (adult retrieval
        contexts are often the ONLY contexts presented once writes are frozen,
        so excluding them would freeze the baseline at its child-phase value).
        No-op when centering is disabled."""
        self.observe(z_world_query, simulation_mode=simulation_mode)
        occ = self._occupied_idx()
        if occ.numel() == 0:
            return None
        key = self._row(z_world_query)
        slot, sim = self._best_match(key)
        if slot is None:
            return None
        self._last_seed_match = float(sim)
        return (self._values[slot].unsqueeze(0).clone(), float(sim), int(slot))

    def note_seed(self) -> None:
        self._n_seeds += 1

    def n_occupied(self) -> int:
        return int(self._occupied.sum().item())

    def reset_anchors(self) -> None:
        """Full clear of the persistent store -- for a NEW developmental stage /
        fresh agent only. NOT called on per-episode reset (the whole point of a
        super-ordinal store is cross-episode persistence)."""
        self._keys.zero_()
        self._values.zero_()
        self._strength.zero_()
        self._occupied.zero_()
        # SD-077: the baseline is part of the store's cue geometry, so a full
        # clear drops it back to unseeded (the next waking context re-seeds it).
        self._baseline = None
        self._n_writes = 0
        self._n_reinforce = 0
        self._n_allocate = 0
        self._n_seeds = 0
        self._last_complexity = -1.0
        self._last_salience = -1.0
        self._last_seed_match = -1.0

    def get_state(self) -> dict:
        return {
            "super_ordinal_n_occupied": self.n_occupied(),
            "super_ordinal_n_writes": self._n_writes,
            "super_ordinal_n_reinforce": self._n_reinforce,
            "super_ordinal_n_allocate": self._n_allocate,
            "super_ordinal_n_seeds": self._n_seeds,
            "super_ordinal_max_strength": float(self._strength.max().item()),
            "super_ordinal_last_complexity": self._last_complexity,
            "super_ordinal_last_salience": self._last_salience,
            "super_ordinal_last_seed_match": self._last_seed_match,
            "super_ordinal_write_enabled": bool(self.write_enabled),
            # SD-077 diagnostics: whether the centered cue key is active, and
            # whether/how far its baseline has been seeded.
            "super_ordinal_cue_centering": bool(self.centering),
            "super_ordinal_baseline_norm": (
                float(self._baseline.norm().item())
                if self._baseline is not None else -1.0
            ),
        }

    def state_dict(self) -> dict:
        return {
            "keys": self._keys.cpu(),
            "values": self._values.cpu(),
            "strength": self._strength.cpu(),
            "occupied": self._occupied.cpu(),
            "write_enabled": self.write_enabled,
            # SD-077: None when unseeded / centering off.
            "baseline": (
                self._baseline.cpu() if self._baseline is not None else None
            ),
        }

    def load_state_dict(self, d: dict) -> None:
        self._keys = d["keys"].to(self.device)
        self._values = d["values"].to(self.device)
        self._strength = d["strength"].to(self.device)
        self._occupied = d["occupied"].to(self.device)
        # SD-077: absent from pre-SD-077 checkpoints -> stays unseeded.
        b = d.get("baseline")
        self._baseline = b.to(self.device) if b is not None else None
        self.write_enabled = bool(d.get("write_enabled", True))


class IncentiveTokenBank:
    """SD-057 (GAP-7 L2-L3): per-object incentive-salience token store.

    A stateful, NON-TRAINABLE per-resource-type bank sitting between the benefit
    pulse and z_goal. Each resource-type tag k (SD-049 1-indexed identity tag)
    accrues:
      base_value[k]: a slowly-decaying, revaluable cached incentive value
                     (Robinson/Berridge 1993 persistence; Balleine/Dickinson 1998
                     revaluable, not write-once).
      z_object[k]:   the stored z_resource identity embedding for that type
                     (the "what" the L4 goal pointer indexes).

    Wanting amplitude at recall is computed drive-revaluably:
      wanting[k] = base_value[k] * (1 + kappa_weight * drive_axis[k])
    (Zhang 2009 V = r * kappa(drive); the (1 + drive_weight * drive) multiplier
    relocated from the GoalState seeding gate onto the stored per-object value).

    No nn.Module, no trainable parameters -- pure dict state + tensor clones.
    """

    def __init__(self, config: GoalConfig, device: torch.device) -> None:
        self.config = config
        self.device = device
        # tag (int) -> base_value (float)
        self._base_value: dict = {}
        # tag (int) -> z_object tensor [1, goal_dim]
        self._z_object: dict = {}

    def is_empty(self) -> bool:
        return len(self._base_value) == 0

    def decay(self) -> None:
        """Slow decay of every token's base_value (called once per update())."""
        d = self.config.incentive_decay
        if d <= 0.0:
            return
        for k in list(self._base_value.keys()):
            self._base_value[k] *= (1.0 - d)

    def update(
        self,
        resource_type: int,
        benefit: float,
        z_object: torch.Tensor,
    ) -> None:
        """L2 bind + L3 revalue. Bind the benefit pulse to object identity
        `resource_type` and EMA-revalue that type's base_value toward the
        received benefit; refresh the stored identity embedding."""
        k = int(resource_type)
        # resource_type 0 = "no resource at agent" (SD-049 convention) -> skip.
        if k <= 0:
            return
        z = z_object.detach()
        if z.dim() == 2:
            z = z.mean(dim=0, keepdim=True)
        elif z.dim() == 1:
            z = z.unsqueeze(0)
        alpha = self.config.incentive_value_alpha
        prev = self._base_value.get(k, 0.0)
        self._base_value[k] = (1.0 - alpha) * prev + alpha * float(benefit)
        self._z_object[k] = z.clone()

    def _drive_axis_for(self, k: int, per_axis_drive, scalar_drive: float) -> float:
        """Per-axis drive for type k (SD-049 type-axis 1:1 mapping: tag k uses
        axis k-1), falling back to the scalar drive when per-axis is unavailable
        or disabled."""
        if (
            self.config.incentive_use_per_axis_drive
            and per_axis_drive is not None
        ):
            try:
                pad = per_axis_drive
                # Flatten a [1, n_axes] tensor to [n_axes]; leave 1-D / sequence as-is.
                if hasattr(pad, "dim") and pad.dim() == 2:
                    pad = pad.reshape(-1)
                axis_idx = k - 1
                if 0 <= axis_idx < len(pad):
                    return float(pad[axis_idx])
            except (TypeError, IndexError):
                pass
        return float(scalar_drive)

    def wanting(self, per_axis_drive=None, scalar_drive: float = 1.0) -> dict:
        """L3 recall: wanting[k] = base_value[k] * (1 + kappa * drive_axis[k]).

        kappa = incentive_drive_kappa_weight * incentive_drive_kappa_scale
        (SD-049-PHASE-2 drive-coupling amend; scale defaults 1.0 = bit-identical).
        """
        kappa = self.config.incentive_drive_kappa_weight * float(
            getattr(self.config, "incentive_drive_kappa_scale", 1.0)
        )
        out = {}
        for k, base in self._base_value.items():
            drive_axis = self._drive_axis_for(k, per_axis_drive, scalar_drive)
            out[k] = base * (1.0 + kappa * drive_axis)
        return out

    def most_wanted(self, per_axis_drive=None, scalar_drive: float = 1.0):
        """L4 pointer: return (k*, z_object[k*], wanting[k*]) for the
        highest-wanting object, or None when the bank is empty / has no stored
        embedding."""
        w = self.wanting(per_axis_drive=per_axis_drive, scalar_drive=scalar_drive)
        if not w:
            return None
        k_star = max(w, key=w.get)
        z = self._z_object.get(k_star)
        if z is None:
            return None
        return (k_star, z, w[k_star])

    def reset(self) -> None:
        self._base_value = {}
        self._z_object = {}

    def state_dict(self) -> dict:
        return {
            "base_value": dict(self._base_value),
            "z_object": {k: v.cpu() for k, v in self._z_object.items()},
        }

    def load_state_dict(self, d: dict) -> None:
        self._base_value = dict(d.get("base_value", {}))
        self._z_object = {
            k: v.to(self.device) for k, v in d.get("z_object", {}).items()
        }


class GoalState:
    """
    Persistent goal representation in z_world latent space.

    z_goal_latent is a slow-decay attractor: pulled toward current z_world
    when benefit fires (alpha_goal update), decaying toward zero otherwise.
    E1's LSTM counteracts decay by maintaining goal context recurrently.

    goal_proximity = 1 / (1 + MSE(z_world, z_goal)) -- bounded [0,1].
    Higher = closer to goal (the wanting signal for trajectory scoring).
    """

    def __init__(self, config: GoalConfig, device: torch.device) -> None:
        self.config = config
        self.device = device
        self._z_goal: torch.Tensor = torch.zeros(
            1, config.goal_dim, device=device
        )
        self._goal_norm_peak: float = 0.0
        # SD-012 sustained-drive EMA trace (goal_pipeline:GAP-3, Option 1).
        # Zero-init: with drive_ema_alpha=1.0 the recursion yields
        # trace == drive_level every step regardless of this value, so OFF is
        # bit-identical. With alpha < 1.0 this introduces a deliberate
        # cold-start transient (accepted per Q2).
        self._drive_trace: float = 0.0
        # SD-057 (GAP-7 L2-L3): per-object incentive-salience token bank.
        # None (and bit-identical OFF) unless use_incentive_token_bank is set.
        self.incentive_bank: Optional[IncentiveTokenBank] = (
            IncentiveTokenBank(config, device)
            if getattr(config, "use_incentive_token_bank", False)
            else None
        )
        # SD-093 / MECH-426: rolling window of goal_proximity() readings and
        # the resulting rate-of-progress (velocity) signal. Populated only by
        # record_progress(), which itself no-ops when
        # use_progress_velocity_effort_modulation is False -- the deque stays
        # empty and _progress_velocity stays 0.0 for the whole episode in
        # that (default) case, so this is bit-identical OFF.
        _pv_window = max(2, int(getattr(config, "progress_velocity_window", 5)))
        self._progress_history: Deque[float] = deque(maxlen=_pv_window)
        self._progress_velocity: float = 0.0
        # SD-092 (MECH-427/428): cross-level subgoal credit. The PARENT
        # (superordinate) attractor -- distinct from the existing `_z_goal`,
        # which is read as the child/subgoal level under this scheme. None
        # (and bit-identical OFF, no extra tensor, no extra branch taken in
        # update()) unless use_hierarchical_goal_credit is set, mirroring the
        # IncentiveTokenBank None-unless-enabled pattern above.
        self._z_goal_parent: Optional[torch.Tensor] = (
            torch.zeros(1, config.goal_dim, device=device)
            if getattr(config, "use_hierarchical_goal_credit", False)
            else None
        )
        self._parent_goal_norm_peak: float = 0.0
        self._n_subgoal_credits: int = 0

    @property
    def z_goal(self) -> torch.Tensor:
        """Current goal latent. Shape: [1, goal_dim]."""
        return self._z_goal

    def update(
        self,
        z_world_current: torch.Tensor,
        benefit_exposure: float,
        drive_level: float = 1.0,
    ) -> None:
        """
        Update z_goal from current world state and benefit signal.

        Always decays. Pulls toward z_world if drive-scaled benefit > threshold.

        Args:
            z_world_current: [batch, world_dim]
            benefit_exposure: scalar benefit this step (body_state[11])
            drive_level: homeostatic drive 0=sated, 1=depleted (SD-012).
                         EMA-smoothed into self._drive_trace (drive_ema_alpha;
                         GAP-3 Option 1), then
                         effective_benefit = benefit_exposure
                             * z_goal_seeding_gain
                             * (1 + drive_weight * drive_trace).
                         drive_ema_alpha=1.0 (default) -> trace == drive_level
                         (bit-identical to the pre-amendment instantaneous form).
                         Default 1.0 for backward compat when drive_weight=0.
        """
        # Always decay toward zero
        self._z_goal = self._z_goal * (1.0 - self.config.decay_goal)

        # MECH-186: valence_wanting floor -- prevent z_goal norm from dropping
        # below the floor value. Simulates tonic serotonergic benefit terrain
        # maintenance. Applied after decay, before any benefit-triggered update.
        # Disabled when valence_wanting_floor <= 0.0 (default).
        floor = getattr(self.config, "valence_wanting_floor", 0.0)
        if floor > 0.0:
            current_norm = self._z_goal.norm().item()
            if current_norm < floor and current_norm > 1e-9:
                # Scale up to floor norm while preserving direction
                self._z_goal = self._z_goal * (floor / current_norm)
            elif current_norm <= 1e-9 and floor > 0.0:
                # z_goal is zero vector: cannot preserve direction.
                # Floor clamp has no effect until first benefit contact seeds direction.
                pass

        # SD-012 sustained-drive amendment (goal_pipeline:GAP-3, Option 2):
        # Apply insatiability floor before the EMA update so the trace stays
        # >= drive_floor in steady state, guaranteeing a minimum multiplier
        # contribution even when the agent is well-fed (drive_level near 0).
        # drive_floor=0.0 (default) -> no-op, bit-identical to pre-amendment.
        drive_level_floored = max(drive_level, self.config.drive_floor)

        # SD-012 sustained-drive amendment (goal_pipeline:GAP-3, Option 1):
        # EMA-smooth the (floored) drive_level so the multiplier does not
        # collapse on the consummatory step. alpha=1.0 (default) -> trace ==
        # drive_level_floored every step -> bit-identical OFF at drive_floor=0.
        alpha = self.config.drive_ema_alpha
        self._drive_trace = (
            (1.0 - alpha) * self._drive_trace + alpha * drive_level_floored
        )

        # MECH-187: apply seeding gain before drive modulation
        # gain=1.0 (default) is identity -- fully backward compatible.
        # SD-012: scale benefit by the sustained drive trace
        effective_benefit = benefit_exposure * self.config.z_goal_seeding_gain * (
            1.0 + self.config.drive_weight * self._drive_trace
        )

        # Pull toward current z_world if effective benefit fires
        if effective_benefit > self.config.benefit_threshold:
            z_w = z_world_current.detach()
            if z_w.dim() == 2:
                z_w = z_w.mean(dim=0, keepdim=True)
            self._z_goal = (
                (1.0 - self.config.alpha_goal) * self._z_goal
                + self.config.alpha_goal * z_w
            )
            norm = self._z_goal.norm().item()
            if norm > self._goal_norm_peak:
                self._goal_norm_peak = norm

        # SD-092 (MECH-427/428): decay the PARENT attractor once per update()
        # tick, mirroring the unconditional decay_goal applied to _z_goal
        # above. No-op when the flag is off or the parent has not been
        # allocated -- this branch is never entered in that case, so
        # update()'s existing behaviour is untouched by construction.
        if (
            getattr(self.config, "use_hierarchical_goal_credit", False)
            and self._z_goal_parent is not None
        ):
            self._z_goal_parent = self._z_goal_parent * (
                1.0 - self.config.parent_goal_decay
            )

    def credit_subgoal_attainment(
        self, child_representation: torch.Tensor, credit: float = 1.0
    ) -> dict:
        """SD-092 (MECH-427/428): cross-level subgoal credit.

        Biological basis: Bandura & Schunk (1981) -- decomposing a distal
        goal into attainable proximal subgoals sustained motivation and
        mastery; the bare distal goal alone behaved indistinguishably from
        no goal. This is the discrete-event, cross-LEVEL complement to
        MECH-217's within-level, along-trajectory credit sweep
        (HippocampalModule.backward_credit_sweep /
        spread_reverse_replay_wanting): a subgoal-attainment event pulls the
        PARENT (superordinate) attractor toward the attained subgoal's own
        representation, exactly mirroring the alpha_goal EMA-pull already
        used to seed `_z_goal` itself -- applied one hierarchy level up and
        triggered by a discrete event rather than a continuous benefit
        signal.

        MECH-427 (maintenance: an already-seeded parent gets reinforced) and
        MECH-428 (formation: a near-zero parent gets bootstrapped) are the
        SAME call against different starting states of `_z_goal_parent` --
        there is no separate "formation mode": repeated credit calls from a
        near-zero parent ARE the MECH-428 bootstrap; credit calls against an
        already-live parent ARE the MECH-427 maintenance case.

        `child_representation` is caller-supplied (rather than implicitly
        `self.z_goal`) so this primitive stays agnostic about which
        representation counts as "the attained subgoal" -- that is an
        experiment-design decision (the env's raw waypoint z_world? the
        agent's own settled child-level z_goal at the moment of attainment?)
        left to the call site, not baked into the substrate.

        No-op (returns {}) when use_hierarchical_goal_credit is False.

        Args:
            child_representation: [1, goal_dim] or [goal_dim] (or [batch,
                goal_dim], mean-pooled) tensor representing the attained
                subgoal. Detached before use -- this is bookkeeping state,
                not a differentiable path.
            credit: scalar credit magnitude for this attainment event.
                credit <= 0 or below subgoal_credit_min applies no pull (a
                qualifying-attainment gate, not an error). >0 magnitudes
                above 1.0 are accepted (a single very-high-salience
                attainment may seed the parent in one shot -- the MECH-428
                bootstrap framing explicitly wants a fast initial rise from
                ~0); the EFFECTIVE pull fraction is clamped to 1.0 (a full
                replacement, never an overshoot past the child
                representation).

        Returns:
            dict: n_subgoal_credits (int, cumulative), parent_goal_norm
                (float, post-update), credit_applied (float, the actual pull
                fraction used -- 0.0 when the credit gate did not fire).
                Empty dict when use_hierarchical_goal_credit is False.
        """
        if not getattr(self.config, "use_hierarchical_goal_credit", False):
            return {}
        # Lazily allocate if the flag was flipped on after construction.
        if self._z_goal_parent is None:
            self._z_goal_parent = torch.zeros_like(self._z_goal)
        min_credit = float(getattr(self.config, "subgoal_credit_min", 0.0))
        if credit <= 0.0 or credit < min_credit:
            return {
                "n_subgoal_credits": self._n_subgoal_credits,
                "parent_goal_norm": self.parent_goal_norm(),
                "credit_applied": 0.0,
            }
        z = child_representation.detach()
        if z.dim() == 1:
            z = z.unsqueeze(0)
        elif z.dim() == 2 and z.shape[0] != 1:
            z = z.mean(dim=0, keepdim=True)
        a = min(1.0, float(self.config.parent_goal_alpha) * float(credit))
        self._z_goal_parent = (1.0 - a) * self._z_goal_parent + a * z
        self._n_subgoal_credits += 1
        norm = self._z_goal_parent.norm().item()
        if norm > self._parent_goal_norm_peak:
            self._parent_goal_norm_peak = norm
        return {
            "n_subgoal_credits": self._n_subgoal_credits,
            "parent_goal_norm": norm,
            "credit_applied": a,
        }

    @property
    def z_goal_parent(self) -> Optional[torch.Tensor]:
        """SD-092: current PARENT (superordinate) goal latent, or None when
        use_hierarchical_goal_credit is False / never credited. Shape when
        present: [1, goal_dim]."""
        return self._z_goal_parent

    def parent_goal_norm(self) -> float:
        """SD-092: L2 norm of the current parent attractor. 0.0 when
        use_hierarchical_goal_credit is False or the parent has not been
        allocated yet -- the metric EXP-0385/EXP-0390 both name
        (parent-goal commitment / z_goal_norm at the parent level)."""
        if self._z_goal_parent is None:
            return 0.0
        return float(self._z_goal_parent.norm().item())

    def parent_is_active(self) -> bool:
        """SD-092: True if the parent attractor has received at least one
        credit event (mirrors GoalState.is_active() for the child level)."""
        if self._z_goal_parent is None:
            return False
        return self._z_goal_parent.abs().sum().item() > 1e-6

    def goal_proximity(self, z_world: torch.Tensor) -> torch.Tensor:
        """
        Bounded wanting signal. Higher = closer to goal.
        Returns 1 / (1 + MSE_sum(z_world, z_goal)). Shape: [batch].
        """
        z_goal_exp = self._z_goal.expand_as(z_world)
        dist = F.mse_loss(z_world, z_goal_exp, reduction="none").sum(dim=-1)
        return 1.0 / (1.0 + dist)

    def goal_distance(self, z_world: torch.Tensor) -> torch.Tensor:
        """Raw MSE distance from goal. Lower = closer. Shape: [batch]."""
        z_goal_exp = self._z_goal.expand_as(z_world)
        return F.mse_loss(z_world, z_goal_exp, reduction="none").sum(dim=-1)

    def record_progress(self, z_world: torch.Tensor) -> float:
        """SD-093 / MECH-426 (progress_velocity_maintenance): record this
        tick's goal_proximity(z_world) into a rolling window and
        differentiate it into a rate-of-progress (velocity) signal -- the
        Carver & Scheier (1990) second-order "velocity" control loop, in
        which affect (here, the effort-modulation signal) is generated from
        the RATE of discrepancy reduction rather than the discrepancy
        itself.

        No-op (history left untouched, returns 0.0) when
        use_progress_velocity_effort_modulation is False -- bit-identical
        OFF. Intended call site: once per E3 tick, on the SAME z_world used
        for the tick's other one-shot (non-per-candidate) goal appraisals
        (agent.py._e3_tick's z_world_for_e3), not the per-candidate
        trajectory rollouts scored by compute_goal_score().

        Returns the freshly-computed velocity (also cached; see
        progress_velocity).
        """
        if not getattr(self.config, "use_progress_velocity_effort_modulation", False):
            return 0.0
        proximity = float(self.goal_proximity(z_world).mean().item())
        self._progress_history.append(proximity)
        if len(self._progress_history) < 2:
            self._progress_velocity = 0.0
        else:
            span = len(self._progress_history) - 1
            self._progress_velocity = (
                self._progress_history[-1] - self._progress_history[0]
            ) / span
        return self._progress_velocity

    @property
    def progress_velocity(self) -> float:
        """SD-093 / MECH-426: current rate-of-progress (temporal derivative
        of goal_proximity over the rolling window). Positive = closing the
        gap to goal (proximity increasing); negative = falling behind
        (proximity decreasing). 0.0 until record_progress() has been called
        at least twice, or whenever the master switch is off.
        """
        return self._progress_velocity

    @property
    def progress_velocity_effort_modulation(self) -> float:
        """SD-093 / MECH-426: effort/persistence modulation derived from
        progress_velocity (Carver & Scheier 1990 coasting model).

        Sign convention (consumed by E3TrajectorySelector.select(), whose
        commit rule is `committed = variance < effective_threshold`):
          POSITIVE = increased effort/persistence pressure -- progress has
            STALLED or reversed (velocity <= 0). The consumer RAISES its
            effective commit threshold (more permissive -- a given variance
            more readily counts as "confident enough"), i.e. lock in and
            push through rather than keep re-deliberating.
          NEGATIVE = decreased effort/persistence pressure -- progress is
            ABOVE the reference rate (velocity > 0), i.e. coasting. The
            consumer LOWERS its effective commit threshold (stricter -- more
            readily kicked back into deliberation), licensing redeployment.

        This is deliberately the OPPOSITE of "positive progress -> bonus":
        the claim's own notes are explicit that treating positive
        progress-affect as a same-goal-commitment bonus INVERTS the
        Carver & Scheier theory. This modulator must only ever be consumed
        as an EFFORT/PERSISTENCE signal (e.g. E3's commit threshold), never
        added into goal VALUE / trajectory score
        (compute_goal_score()/goal_proximity() are untouched by this flag).

        0.0 (no modulation) when the master switch is off.
        """
        if not getattr(self.config, "use_progress_velocity_effort_modulation", False):
            return 0.0
        gain = float(getattr(self.config, "progress_velocity_effort_gain", 1.0))
        cap = float(getattr(self.config, "progress_velocity_effort_max", 0.3))
        raw = -gain * self._progress_velocity
        return max(-cap, min(cap, raw))

    def with_injection(self, inject_norm: float) -> "GoalState":
        """
        MECH-188: Return a view of this GoalState with z_goal norm floored at inject_norm.

        Creates a lightweight wrapper that shares the same config but overrides
        _z_goal with a version that has a minimum L2 norm of inject_norm.
        Used by agent.select_action() when z_goal_inject > 0 -- applies to
        action selection only, does NOT modify the persistent attractor.

        If z_goal has no direction (norm=0), a constant unit vector is used so
        that goal_proximity still produces a non-trivial gradient for trajectory
        scoring. The first non-zero dimension (index 0) is set.

        Args:
            inject_norm: minimum L2 norm floor for the injected z_goal.

        Returns:
            A GoalState whose _z_goal has norm >= inject_norm.
        """
        injected = GoalState.__new__(GoalState)
        injected.config = self.config
        injected.device = self.device
        injected._goal_norm_peak = self._goal_norm_peak
        # SD-093 / MECH-426: propagate the rolling-window velocity state so
        # E3.select() (which receives THIS injected wrapper, not the
        # original GoalState, whenever z_goal_inject > 0) still sees the
        # real progress_velocity_effort_modulation rather than raising
        # AttributeError / silently reading a fresh zero-initialised deque.
        # Shares the SAME deque object (not a copy) -- injection is a
        # read-only scoring view for this tick only, never mutated here.
        injected._progress_history = self._progress_history
        injected._progress_velocity = self._progress_velocity
        # SD-092: carry the parent-attractor state through to the injected
        # view (a lightweight __new__ construct that bypasses __init__), so
        # parent_goal_norm()/z_goal_parent/parent_is_active() remain callable
        # on it rather than raising AttributeError. with_injection() does not
        # modify the persistent attractor either way (docstring above).
        injected._z_goal_parent = self._z_goal_parent
        injected._parent_goal_norm_peak = self._parent_goal_norm_peak
        injected._n_subgoal_credits = self._n_subgoal_credits

        current_norm = self._z_goal.norm().item()
        if current_norm >= inject_norm:
            # Already above floor: no change
            injected._z_goal = self._z_goal
        elif current_norm > 1e-9:
            # Scale up to floor norm while preserving direction
            injected._z_goal = self._z_goal * (inject_norm / current_norm)
        else:
            # z_goal is zero: use first-dimension unit vector scaled to inject_norm
            z_seed = torch.zeros_like(self._z_goal)
            z_seed[0, 0] = inject_norm
            injected._z_goal = z_seed

        return injected

    def cue_pull(self, z_object: torch.Tensor, strength: float) -> None:
        """SD-057 phase-2 L6 (MECH-347): nudge z_goal toward a stored object
        embedding WITHOUT the benefit gate and WITHOUT revaluing any token.

        This is the cue-triggered-wanting pull: perceiving a learned cue for an
        object raises wanting for that object (z_goal moves toward its identity
        embedding), which the existing E3 goal_proximity + MECH-295 approach
        bridge then translate into pre-consummatory approach -- distinct from
        the benefit-driven seed in update(), which requires a benefit pulse and
        EMA-revalues the per-object token. No decay/floor logic here; this is a
        pure directional nudge.

        Args:
            z_object: [1, goal_dim] or [goal_dim] stored object embedding.
            strength: pull fraction in [0, 1]; z_goal moves this fraction toward
                      z_object. <= 0 is a no-op.
        """
        if strength <= 0.0:
            return
        z = z_object.detach()
        if z.dim() == 1:
            z = z.unsqueeze(0)
        elif z.dim() == 2 and z.shape[0] != 1:
            z = z.mean(dim=0, keepdim=True)
        s = float(min(1.0, strength))
        self._z_goal = (1.0 - s) * self._z_goal + s * z
        norm = self._z_goal.norm().item()
        if norm > self._goal_norm_peak:
            self._goal_norm_peak = norm

    def is_active(self) -> bool:
        """True if z_goal has been updated at least once."""
        return self._z_goal.abs().sum().item() > 1e-6

    def goal_norm(self) -> float:
        """L2 norm of current z_goal."""
        return float(self._z_goal.norm().item())

    def reset(self) -> None:
        """Reset goal to zero."""
        self._z_goal = torch.zeros_like(self._z_goal)
        self._goal_norm_peak = 0.0
        # SD-012 GAP-3: the sustained-drive trace is a per-episode state
        # (the goal_pipeline Q2 zero-init cold-start is defined per episode,
        # not just per agent construction). Mirrors the _z_goal reset above
        # so eval/training loops that call reset() between episodes restart
        # the trace from the documented zero-init.
        self._drive_trace = 0.0
        # SD-057: the incentive token bank is per-episode state (per-object
        # wanting amplitudes reset alongside the z_goal attractor).
        if self.incentive_bank is not None:
            self.incentive_bank.reset()
        # SD-093 / MECH-426: the progress-velocity rolling window is
        # per-episode state (a stalled-vs-progressing read from a PRIOR
        # episode must not leak into a fresh one's effort modulation).
        self._progress_history.clear()
        self._progress_velocity = 0.0
        # SD-092: the parent (superordinate) attractor is per-episode state,
        # matching the base _z_goal reset above (this is distinct from
        # MECH-189's cross-episode-persistent SuperOrdinalGoalMemory, which
        # is a separate mechanism and is NOT reset here). No-op when the
        # flag is off / the parent was never allocated.
        if self._z_goal_parent is not None:
            self._z_goal_parent = torch.zeros_like(self._z_goal_parent)
            self._parent_goal_norm_peak = 0.0
            self._n_subgoal_credits = 0

    def state_dict(self) -> dict:
        return {
            "z_goal": self._z_goal.cpu(),
            "goal_norm_peak": self._goal_norm_peak,
            # SD-092: None when use_hierarchical_goal_credit is False / the
            # parent attractor was never allocated (matches the
            # None-unless-enabled pattern used elsewhere in this file, e.g.
            # SuperOrdinalGoalMemory.state_dict's baseline field).
            "z_goal_parent": (
                self._z_goal_parent.cpu() if self._z_goal_parent is not None else None
            ),
            "parent_goal_norm_peak": self._parent_goal_norm_peak,
            "n_subgoal_credits": self._n_subgoal_credits,
        }

    def load_state_dict(self, d: dict) -> None:
        self._z_goal = d["z_goal"].to(self.device)
        self._goal_norm_peak = float(d.get("goal_norm_peak", 0.0))
        # SD-092: absent from pre-SD-092 checkpoints -> stays unallocated
        # (None) unless the checkpoint itself has a stored parent tensor.
        zgp = d.get("z_goal_parent")
        self._z_goal_parent = zgp.to(self.device) if zgp is not None else None
        self._parent_goal_norm_peak = float(d.get("parent_goal_norm_peak", 0.0))
        self._n_subgoal_credits = int(d.get("n_subgoal_credits", 0))
