"""ARC-063 v1: policy.rule_apprehension_layer -- distributed CandidateRule field.

The non-Bayesian rule-creator that resolves arc_062_rule_apprehension:GAP-B
(MECH-309: "trainers weight rules they do not invent"). Design doc:
REE_assembly/docs/architecture/arc_063_candidate_rule_field.md.

Mint-then-weight, over a subspace-partitioned field. Separates CREATION (a
non-gradient, structural mint event) from WEIGHTING (eligibility-trace credit
that refines an existing rule's availability). The 543/598b lineage showed
that gradient descent on two scalar scoring heads has an inert equilibrium
(head_0 == head_1) and so the rule_state handed to SD-033a collapses
(598b C3 trainable_not_monomodal FAIL). ARC-063's fix is structural: distinct
rules exist because they were MINTED as distinct slots occupying distinct
pinned subspace directions, so the rule_state is differentiated by
construction -- exactly what 598b found missing.

The five faces (see design doc):
  1. CREATE   -- mint a slot on a detected recurring (context -> action-object
                 -> outcome) regularity (bottom-up ARC-064), optionally seeded
                 by the ARC-062 discriminator (top-down).
  2. REPRESENT -- a field of K slots, each a distinct pinned subspace direction
                 (Weber 2023 mixed-selectivity subspace partitioning; the
                 anti-monomodal geometry).
  3. GATE     -- tolerance-gated availability: a rule becomes AVAILABLE only
                 when its availability clears a threshold that RISES with the
                 number of competing context-matched rules (Cavanagh 2011 /
                 Frank 2006 dynamic conflict threshold). Availability != selection
                 (Frank/O'Reilly 2001).
  4. SELECT   -- cue-driven context-bound retrieval: match the current context
                 to each rule's context_tag (MECH-338).
  5. OUTPUT + CREDIT -- the available-AND-context-matched rules combine into a
                 differentiated rule_state vector for SD-033a, and an
                 eligibility trace credits the availability of rules that were
                 active when an outcome arrived (Brzosko 2015 / Kovach 2012).

This module is pure-arithmetic (no nn.Module, no trained parameters, no gradient
flow); the differentiation is structural, not learned -- matching the design's
"v1 = no new trained encoder" scope. Sibling pattern to ree_core/policy/
noise_floor.py (MECH-313), structured_curiosity.py (MECH-314), tonic_vigor.py
(MECH-320).

MECH-094: every state-advancing method accepts a simulation_mode argument and is
a no-op when True; replay / DMN paths must not mint or credit. The agent ticks
the field only on the waking select_action path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch

# Small headroom added above theta when maintenance_couple_to_theta floors a
# maintained rule's availability so it strictly CLEARS the gate (>=, not just ==).
MAINTENANCE_THETA_EPS: float = 1e-3


@dataclass
class CandidateRuleFieldConfig:
    """ARC-063 CandidateRuleField configuration.

    Attributes:
        use_candidate_rule_field: master switch (default False, backward-compatible).
        n_slots: K candidate-rule slots (the field capacity).
        rule_dim: rule_embedding dimensionality (matches SD-033a rule_dim so the
            field output drops straight into LateralPFCAnalog.rule_state).
        mint_recurrence_threshold: bottom-up creator -- a (context-bucket,
            action-object) regularity must recur this many times before a slot
            is minted for it.
        tolerance_floor: base availability threshold theta_0 (Frank 2006).
        tolerance_conflict_gain: how much theta rises per competing context-
            matched rule (the conflict-graded "hold your horses" dynamic
            threshold).
        availability_alpha: credit EMA rate on availability.
        availability_decay: slow multiplicative decay of availability per tick
            (between activations) -- "use it or lose it".
        eligibility_window: eligibility-trace decay window in ticks (recency
            credit). The per-tick multiplicative decay is (1 - 1/window).
        context_match_threshold: cosine(context, context_tag) floor for a rule
            to be retrieved as context-matched.
        seed_from_arc062: when True, the ARC-062 discriminator gating_weight
            (if supplied) nudges a freshly minted rule's initial availability
            (top-down population). Default True.
        persist_rules_across_episode_reset: when True, reset() does NOT clear
            the live rule pool (self._rules) or recurrence counters
            (self._recurrence) -- the field accumulates a differentiated rule
            pool ACROSS episodes instead of cold-starting every per-episode
            agent.reset(). Default False = bit-identical to the legacy
            per-episode wipe. Biologically faithful: PFC/BG task-set rule
            learning accumulates across experiences and is NOT reset per trial
            (Collins & Frank 2014; Mansouri rule-selective persistence). The
            per-episode wipe was the V3-EXQ-654 GAP-B falsifier blocker
            (failure_autopsy_V3-EXQ-654_2026-06-09): at ~26-tick behavioural
            episodes the recurrence-threshold mint (>=3 within one episode)
            plus the wipe meant the live pool never matured (crf_frac_active
            ~0.12 < 0.30 floor despite cumulative n_minted 131-408).
        pinned_seed: fixed seed for the deterministic pinned-distinct slot
            directions (Weber separability). Deterministic across runs.
        mature_pool_dynamics: when True, recalibrate the GATE/CREDIT/RETIRE
            dynamics so a differentiated, persistently-active pool of >=2 rules
            can form and persist (routes the gate/credit/retire knobs below).
            Default False = bit-identical to the legacy dynamics. Routed by
            failure_autopsy_V3-EXQ-654b_2026-06-11: across 654 (per-episode wipe),
            654a (crf_persist), 654b (crf_persist + 240 ep) crf_frac_active is
            pinned at ~0.13 and crf_max_pairwise_rule_dist is 0.0 every ARM_ON
            cell -- the pool churns (mint -> brief life -> retire -> re-mint) and
            never holds >=2 rules present, so the maturation-BUDGET reading is
            exhausted. The fix is the gate/credit/retire dynamics, not more
            budget. When this flag is False NONE of the mature_* knobs are
            consulted.
        mature_availability_decay: slower passive availability decay per tick
            under mature_pool_dynamics (default 0.001 vs legacy 0.005 -- 5x
            slower so rules persist longer between activations).
        mature_retire_floor: ABSOLUTE retirement floor under mature_pool_dynamics
            (default 0.05). Replaces the legacy 0.5*tolerance_floor=0.15 so a
            rule survives low availability long enough for a 2nd differentiated
            rule to co-accumulate (the primary 654b retire-churn driver).
        mature_availability_alpha_negative: asymmetric credit rate for NEGATIVE
            outcomes under mature_pool_dynamics (default 0.02 vs the symmetric
            legacy availability_alpha=0.1). Negative outcomes are frequent in a
            hazard env; a gentler negative alpha slows the availability collapse
            that empties the pool before differentiation.
        mature_tolerance_floor / mature_tolerance_conflict_gain: recalibrated
            conflict-gate pair under mature_pool_dynamics (default 0.15 / 0.25).
            Legacy 0.3 + 1.0*n_competing gives theta>=1.3 > 1.0 max availability
            whenever >=2 rules match a context, so >=2 matched rules can NEVER
            both be active (the latent deadlock). 0.15 + 0.25*n keeps theta
            reachable: theta(1)=0.40, theta(2)=0.65, theta(3)=0.90 (all < 1.0),
            so up to 4 context-matched rules can be active together.
        mature_mint_block_threshold: DECOUPLED mint-block cosine floor under
            mature_pool_dynamics (default 0.8, higher than the 0.5
            context_match_threshold used for retrieval). A new differentiated
            mint is blocked only by a *very* similar existing rule, relieving the
            secondary mint-block under low raw-z_world spread (the structural
            relief is the e2_world_forward context routing on the agent side).
        mature_mint_protection_ticks: a freshly minted rule is protected from
            retirement for this many ticks under mature_pool_dynamics (default
            30, 0 = no protection) so a 2nd differentiated rule has time to
            co-accumulate before the first is retired (directly targets the
            654b "rule drops below floor before a 2nd rule co-accumulates").
    """

    use_candidate_rule_field: bool = False
    n_slots: int = 16
    rule_dim: int = 16
    mint_recurrence_threshold: int = 3
    tolerance_floor: float = 0.3
    tolerance_conflict_gain: float = 1.0
    availability_alpha: float = 0.1
    availability_decay: float = 0.005
    eligibility_window: int = 20
    context_match_threshold: float = 0.5
    seed_from_arc062: bool = True
    persist_rules_across_episode_reset: bool = False
    pinned_seed: int = 6063
    # --- mature-pool dynamics (V3-EXQ-654b amend; all consulted only when
    # mature_pool_dynamics=True; default False -> bit-identical legacy path) ---
    mature_pool_dynamics: bool = False
    mature_availability_decay: float = 0.001
    mature_retire_floor: float = 0.05
    mature_availability_alpha_negative: float = 0.02
    mature_tolerance_floor: float = 0.15
    mature_tolerance_conflict_gain: float = 0.25
    mature_mint_block_threshold: float = 0.8
    mature_mint_protection_ticks: int = 30
    # --- crf-availability-maintenance (V3-EXQ-666 successor; all consulted only
    # when availability_maintenance=True; default False -> bit-identical legacy
    # path). The B-leaning-hybrid resolution of the differentiation<->persistence
    # tension (targeted_review_arc_063_crf_rule_cell_persistence SYNTHESIS): once
    # crf_context_from_e2_world_forward delivers a DIFFERENTIATED pool (ARM_2:
    # 10-16 distinct rules, dist 1.71), each narrowly-tuned rule matches a sparse
    # context slice, so the match-triggered-EMA availability never accumulates
    # above theta between matches and decays in the gaps (mature_availability_decay
    # 0.001/tick) -> crf_frac_active collapses to 0.016, WORSE than the
    # undifferentiated legacy 0.125. The activity-silent-synaptic literature
    # (Mongillo 2008 facilitation; Stokes 2015; Lundqvist 2018) says the
    # available-but-unselected POOL must be held silently across context-absent
    # ticks, NOT kept firing -- so the fix is (1) hold availability across silence
    # (remove the silence-driven decay; only exception/interference erodes it) and
    # (2) re-state the readiness readout on the MAINTAINED pool, not the
    # instantaneous active fraction (which is the averaged-activity artefact the
    # lit warns against).
    #   availability_maintenance: master switch.
    #   maintenance_floor: a freshly minted DIFFERENTIATED rule starts with
    #     availability at least this high (default 0.45, above the mature 2-way
    #     match theta(1)=0.40) so a maintained rule would clear threshold if its
    #     context recurred even against one competitor. Applied only at mint; the
    #     silence-hold then keeps it there. NOT re-floored every tick, so the
    #     negative-outcome credit exception/interference path can still erode a
    #     consistently-bad rule below it and retire it.
    #   maintenance_decay: optional slow long-horizon multiplicative leak per tick
    #     that REPLACES the silence-driven (mature_)availability_decay under
    #     maintenance (default 0.0 = pure hold -- the synaptic impression persists;
    #     set small to model slow forgetting at a deliberately-long horizon).
    #   engaged_sustain / engaged_sustain_rate: prescription 3 (optional,
    #     secondary) -- a short sustained-activity reverberation for the
    #     matched-and-selected (engaged) rule (Funahashi 1989 / Compte-Wang 2000),
    #     pairing with Mongillo facilitation. NOT the pool fix; default OFF.
    #   maintained_reactivation_threshold: a rule counts as
    #     maintained-and-reactivatable in the readout when availability >= this.
    #     Sentinel <=0 (default) derives it from the single-match gate floor
    #     (mature_tolerance_floor when mature, else tolerance_floor) -- i.e. the
    #     threshold the rule would clear if its context recurred and it were the
    #     sole match (the realistic reactivation case for a narrowly-tuned rule).
    availability_maintenance: bool = False
    maintenance_floor: float = 0.45
    maintenance_decay: float = 0.0
    engaged_sustain: bool = False
    engaged_sustain_rate: float = 0.1
    maintained_reactivation_threshold: float = 0.0
    # --- CRF conflict-gate calibration amend (V3-EXQ-654d successor; the
    # crf-availability-maintenance substrate_queue entry). All consulted only
    # under mature_pool_dynamics; defaults are no-op (bit-identical legacy gate).
    # Routed by failure_autopsy_V3-EXQ-654d_2026-06-16: the 666c maintenance amend
    # built + maintained a differentiated pool (crf_max_pairwise_rule_dist 1.711),
    # but activation collapsed to crf_frac_active=0.0 because the 16 minted rules'
    # context_tags are mutually within context_match_threshold (cosine >= 0.5) ->
    # 7-8 co-match per tick -> gate_and_select theta = mature_tolerance_floor(0.15)
    # + mature_tolerance_conflict_gain(0.25)*(n_matched-1) ~= 1.65 >> maintenance_floor
    # 0.45 -> EVERY matched rule gated out. 654d proved this lockout is INDEPENDENT
    # of the GAP-A selection-authority conversion (it persists on the seeds where
    # the consumed_summary spread clears the 0.05 floor) -- the CRF rule-match
    # context key and the E3 selection channel are distinct loci.
    #   mature_context_match_threshold (FAULT 1, context-key crowding): sentinel
    #     <0 (default) -> gate_and_select uses context_match_threshold (legacy).
    #     When >=0 (e.g. 0.7) the GATE match cutoff is sharpened so fewer of the
    #     clustered context_tags co-match a per-tick context -> n_matched falls to
    #     ~2-3. Decoupled from the retrieval threshold (context_match_threshold,
    #     still used by maintained_reactivatable_rules) and from mint-block
    #     (mature_mint_block_threshold, 0.8): the differentiated pool stays minted,
    #     only fewer co-fire per tick.
    #   tolerance_conflict_cap (FAULT 2a, theta growth): sentinel <0 (default) ->
    #     no cap. When >=0 (e.g. 3) theta uses min(n_competing, cap), so theta is
    #     bounded at theta_floor + theta_gain*cap (0.15 + 0.25*3 = 0.90 < 1.0) and
    #     stays reachable under match-crowding spikes.
    #   maintenance_couple_to_theta (FAULT 2b, winner-admit): default False ->
    #     legacy maintenance hold. When True (with availability_maintenance +
    #     mature_pool_dynamics) the per-tick maintenance step floors each maintained
    #     rule's availability to max(maintenance_floor, theta(_last_n_matched)+eps)
    #     (eps=MAINTENANCE_THETA_EPS) so the maintained, differentiated pool CLEARS
    #     the (capped) gate under realistic crowding rather than being suppressed
    #     wholesale -- electing the differentiated set to co-fire (the autopsy:
    #     "a maintained set of differentiated rules must be SELECTABLE, not gated
    #     out by mutual crowding"). Effective only with the cap (or a sharpened
    #     match threshold) keeping theta < 1.0; an uncapped theta>1.0 cannot be
    #     cleared even at maximal availability.
    mature_context_match_threshold: float = -1.0
    tolerance_conflict_cap: int = -1
    maintenance_couple_to_theta: bool = False


@dataclass
class CandidateRule:
    """One slot of the field.

    rule_embedding -- the "what": a pinned-distinct unit vector in rule_dim
        (the bias content handed to SD-033a). Subspace-partitioned so rules stay
        separable by construction (Weber 2023).
    context_tag -- the cue/state signature under which the rule was minted
        (the MECH-338 retrieval key; lives in context/world_dim space).
    availability -- accumulated support net of exceptions in [0, 1]; the
        tolerance-gated quantity.
    eligibility -- a decaying trace marking the rule creditable for recent
        outcomes.
    minted_step / last_active_step -- bookkeeping for credit / retirement.
    active -- set by gate_and_select each tick (available AND context-matched).
    """

    rule_embedding: torch.Tensor
    context_tag: torch.Tensor
    availability: float
    eligibility: float = 0.0
    minted_step: int = 0
    last_active_step: int = -1
    active: bool = False
    action_object_idx: int = -1


class CandidateRuleField:
    """ARC-063 v1 distributed CandidateRule field (pure-arithmetic).

    The agent calls step() once per waking tick; it credits the previous tick's
    active rules, mints on detected recurrences, gates + selects against the
    current context, and returns the differentiated rule_state vector for
    SD-033a. gate_and_select / active_rule_state / credit are also exposed for
    unit testing.
    """

    def __init__(
        self,
        context_dim: int,
        config: Optional[CandidateRuleFieldConfig] = None,
    ) -> None:
        self.config = config if config is not None else CandidateRuleFieldConfig()
        self.context_dim = int(context_dim)
        self.rule_dim = int(self.config.rule_dim)

        # Pinned-distinct slot directions (Weber separability). Deterministic
        # unit vectors in rule_dim, one per slot -- the structural guarantee
        # that distinct minted rules occupy distinct subspace directions so the
        # rule_state cannot collapse the way two shared-gradient scalar heads do.
        gen = torch.Generator()
        gen.manual_seed(int(self.config.pinned_seed))
        raw = torch.randn(self.config.n_slots, self.rule_dim, generator=gen)
        norms = raw.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        self._pinned_directions = raw / norms  # [n_slots, rule_dim], unit rows

        # Slot pool: index -> CandidateRule (only minted slots present).
        self._rules: Dict[int, CandidateRule] = {}
        # Recurrence counters keyed on (context_bucket, action_object_idx).
        self._recurrence: Dict[Tuple, int] = {}
        self._step: int = 0

        # Per-tick / cumulative diagnostics.
        self._n_minted: int = 0
        self._n_retired: int = 0
        self._n_simulation_skipped: int = 0
        self._last_n_active: int = 0
        self._last_n_matched: int = 0
        self._last_minted_this_step: int = 0
        # Ticks on which >=1 rule fired (for crf_frac_active -- the CRF-readiness
        # gate readout: crf_max_pairwise_rule_dist > floor AND frac_active >= 0.30).
        self._n_active_steps: int = 0
        # The retirement floor below which a rule's availability frees its slot.
        # mature_pool_dynamics lowers it to an absolute floor (decoupled from
        # tolerance_floor) so a rule survives long enough for a 2nd differentiated
        # rule to co-accumulate (V3-EXQ-654b retire-churn fix). Default OFF keeps
        # the legacy 0.5*tolerance_floor.
        if self.config.mature_pool_dynamics:
            self._retire_floor: float = float(self.config.mature_retire_floor)
        else:
            self._retire_floor = 0.5 * float(self.config.tolerance_floor)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _context_bucket(self, context: torch.Tensor) -> Tuple:
        """Coarse, hashable signature of a context vector for recurrence counting.

        Sign pattern of the leading min(8, context_dim) dimensions. Deliberately
        coarse: the bottom-up creator should mint on a recurring *regime*, not on
        every micro-variation of the latent.
        """
        n = min(8, context.shape[-1])
        signs = torch.sign(context[..., :n].reshape(-1)[:n])
        return tuple(int(s.item()) for s in signs)

    def _cosine(self, a: torch.Tensor, b: torch.Tensor) -> float:
        denom = (a.norm() * b.norm()).clamp_min(1e-8)
        return float((a @ b) / denom)

    def _free_slot_index(self) -> Optional[int]:
        for i in range(self.config.n_slots):
            if i not in self._rules:
                return i
        return None

    # ------------------------------------------------------------------
    # CREATE -- the rule-creator (the GAP-B blocker)
    # ------------------------------------------------------------------
    def _maybe_mint(
        self,
        context: torch.Tensor,
        action_object_idx: int,
        arc062_seed: Optional[float],
    ) -> int:
        """Mint a slot on a recurring (context-bucket, action-object) regularity.

        Returns the number of rules minted this call (0 or 1). A regularity is
        only minted when (a) it recurs >= mint_recurrence_threshold times AND
        (b) no existing rule already covers this context (context_tag cosine
        >= context_match_threshold) AND (c) a free slot exists.
        """
        bucket = self._context_bucket(context)
        key = (bucket, int(action_object_idx))
        self._recurrence[key] = self._recurrence.get(key, 0) + 1
        if self._recurrence[key] < self.config.mint_recurrence_threshold:
            return 0
        # Already covered by an existing rule? Under mature_pool_dynamics the
        # mint-block cosine floor is DECOUPLED from the retrieval threshold
        # (raised to mature_mint_block_threshold) so a new differentiated mint is
        # blocked only by a *very* similar existing rule -- relieving the
        # secondary 654b mint-block under low raw-z_world spread.
        mint_block_thresh = (
            self.config.mature_mint_block_threshold
            if self.config.mature_pool_dynamics
            else self.config.context_match_threshold
        )
        for rule in self._rules.values():
            if self._cosine(context, rule.context_tag) >= mint_block_thresh:
                return 0
        slot = self._free_slot_index()
        if slot is None:
            return 0
        embedding = self._pinned_directions[slot].to(
            dtype=context.dtype, device=context.device
        ).clone()
        # Top-down ARC-062 seed: a confident discriminator nudges the freshly
        # minted rule's initial availability above the bare floor.
        init_avail = float(self.config.tolerance_floor)
        if self.config.seed_from_arc062 and arc062_seed is not None:
            # arc062_seed is a gating_weight in [0, 1]; distance from 0.5 is the
            # discriminator's confidence that this context is a distinct regime.
            confidence = abs(float(arc062_seed) - 0.5) * 2.0
            init_avail = min(1.0, init_avail + 0.5 * confidence)
        # crf-availability-maintenance: a freshly minted differentiated rule starts
        # robustly maintained (>= maintenance_floor, above the mature 2-way-match
        # theta) so the activity-silent hold below keeps it reactivatable across
        # context-absent ticks. Applied only at mint; not re-floored every tick.
        if self.config.availability_maintenance:
            init_avail = min(1.0, max(init_avail, float(self.config.maintenance_floor)))
        self._rules[slot] = CandidateRule(
            rule_embedding=embedding,
            context_tag=context.detach().clone(),
            availability=init_avail,
            eligibility=0.0,
            minted_step=self._step,
            last_active_step=-1,
            active=False,
            action_object_idx=int(action_object_idx),
        )
        self._n_minted += 1
        return 1

    # ------------------------------------------------------------------
    # GATE + SELECT -- tolerance-gated, context-bound retrieval
    # ------------------------------------------------------------------
    def _gate_match_threshold(self) -> float:
        """The cosine cutoff gate_and_select uses to decide which rules co-match.

        CRF-gate amend (FAULT 1): under mature_pool_dynamics a non-negative
        mature_context_match_threshold SHARPENS the gate cutoff (above the legacy
        context_match_threshold) so fewer of the clustered context_tags co-match a
        per-tick context -- reducing n_matched (the 654d 7-8 -> ~2-3 fix). The
        legacy context_match_threshold is unchanged and still used for retrieval /
        mint-block / maintained_reactivatable readouts. Sentinel <0 -> legacy.
        """
        if (
            self.config.mature_pool_dynamics
            and self.config.mature_context_match_threshold >= 0.0
        ):
            return float(self.config.mature_context_match_threshold)
        return float(self.config.context_match_threshold)

    def _theta_for(self, n_matched: int) -> float:
        """Conflict-scaled availability threshold for the current match count.

        theta = theta_floor + theta_gain * n_competing, n_competing = n_matched-1.
        Legacy 0.3 + 1.0*n gives theta>=1.3 > 1.0 max availability whenever >=2
        rules match (the latent 654b deadlock); mature_pool_dynamics uses
        0.15 + 0.25*n. CRF-gate amend (FAULT 2a): a non-negative
        tolerance_conflict_cap (mature regime only) caps n_competing so theta is
        bounded at theta_floor + theta_gain*cap (e.g. 0.90 at cap=3) and stays
        reachable under match-crowding spikes. Shared by gate_and_select and the
        maintenance_couple_to_theta credit step so both agree on the gate.
        """
        if self.config.mature_pool_dynamics:
            theta_floor = self.config.mature_tolerance_floor
            theta_gain = self.config.mature_tolerance_conflict_gain
            n_competing = max(0, int(n_matched) - 1)
            if self.config.tolerance_conflict_cap >= 0:
                n_competing = min(n_competing, int(self.config.tolerance_conflict_cap))
        else:
            theta_floor = self.config.tolerance_floor
            theta_gain = self.config.tolerance_conflict_gain
            n_competing = max(0, int(n_matched) - 1)
        return float(theta_floor + theta_gain * n_competing)

    def gate_and_select(
        self, context: torch.Tensor, step: Optional[int] = None
    ) -> List[CandidateRule]:
        """Mark each rule active iff it is context-matched AND its availability
        clears the conflict-scaled tolerance threshold.

        theta = tolerance_floor + tolerance_conflict_gain * n_competing_matched
        where n_competing_matched is the count of OTHER context-matched rules.
        Availability != selection: the gate decides which rules ENTER the active
        set; E3 still selects the action among the active rules' biases.
        """
        if step is None:
            step = self._step
        match_thresh = self._gate_match_threshold()
        matched: List[CandidateRule] = [
            r
            for r in self._rules.values()
            if self._cosine(context, r.context_tag) >= match_thresh
        ]
        n_matched = len(matched)
        theta = self._theta_for(n_matched)
        for r in self._rules.values():
            r.active = False
        active: List[CandidateRule] = []
        for r in matched:
            if r.availability >= theta:
                r.active = True
                r.eligibility = 1.0
                r.last_active_step = step
                active.append(r)
        self._last_n_matched = n_matched
        self._last_n_active = len(active)
        if active:
            self._n_active_steps += 1
        return active

    def active_rule_state(
        self, active: Optional[List[CandidateRule]] = None
    ) -> torch.Tensor:
        """Combine the active rules into a differentiated [1, rule_dim] vector.

        Availability-weighted sum of the active rules' pinned-distinct
        embeddings. Differentiated by construction: distinct active rules ->
        distinct directions -> distinct rule_state. Zeros when no rule is active.
        """
        if active is None:
            active = [r for r in self._rules.values() if r.active]
        out = torch.zeros(1, self.rule_dim)
        if not active:
            return out
        ref = active[0].rule_embedding
        out = out.to(dtype=ref.dtype, device=ref.device)
        for r in active:
            out = out + r.availability * r.rule_embedding.reshape(1, -1)
        return out

    # ------------------------------------------------------------------
    # CREDIT -- eligibility-trace availability update + decay + retire
    # ------------------------------------------------------------------
    def credit(self, outcome_signal: float, step: Optional[int] = None) -> None:
        """Eligibility-weighted availability update + slow decay + retirement.

        Rules carrying eligibility are credited toward 1.0 on a non-negative
        outcome (success) and toward 0.0 on a negative outcome (exception),
        scaled by the eligibility trace (Brzosko 2015 retroactive credit;
        Kovach 2012 recency). Eligibility then decays; all availabilities decay
        slowly; rules below the retirement floor free their slot.
        """
        is_negative = float(outcome_signal) < 0.0
        target = 0.0 if is_negative else 1.0
        # Asymmetric credit under mature_pool_dynamics: negative outcomes (frequent
        # in a hazard env) use a gentler alpha so they do not collapse availability
        # below the retire floor before differentiation (654b retire-churn driver).
        if self.config.mature_pool_dynamics and is_negative:
            alpha = self.config.mature_availability_alpha_negative
        else:
            alpha = self.config.availability_alpha
        decay = (
            self.config.mature_availability_decay
            if self.config.mature_pool_dynamics
            else self.config.availability_decay
        )
        cur_step = step if step is not None else self._step
        protect = (
            self.config.mature_mint_protection_ticks
            if self.config.mature_pool_dynamics
            else 0
        )
        decay_e = max(0.0, 1.0 - 1.0 / max(1, self.config.eligibility_window))
        maint = self.config.availability_maintenance
        to_retire: List[int] = []
        for idx, r in self._rules.items():
            if r.eligibility > 1e-6:
                w = alpha * r.eligibility
                r.availability = (1.0 - w) * r.availability + w * target
            r.eligibility *= decay_e
            if maint:
                # crf-availability-maintenance (Mongillo 2008 / Stokes 2015 /
                # Lundqvist 2018): the available POOL is held activity-silently --
                # silence does NOT erode availability (the per-tick decay is
                # REMOVED), only the negative-outcome eligibility credit above (the
                # exception/interference path) drives a consistently-bad rule down
                # and retires it. Optional long-horizon leak (maintenance_decay,
                # default 0.0 = pure hold -- the synaptic impression persists).
                r.availability *= (1.0 - self.config.maintenance_decay)
                if self.config.engaged_sustain and r.eligibility > 1e-6:
                    # Prescription 3 (optional, secondary): a short sustained-
                    # activity reverberation for the engaged (recently-active) rule
                    # (Funahashi 1989 / Compte-Wang 2000), pairing with Mongillo
                    # facilitation. NOT the pool fix; default OFF.
                    r.availability = min(
                        1.0,
                        r.availability
                        + self.config.engaged_sustain_rate * r.eligibility,
                    )
                if (
                    self.config.maintenance_couple_to_theta
                    and self.config.mature_pool_dynamics
                ):
                    # CRF-gate amend (FAULT 2b, the winner-admit): couple the
                    # maintained availability to the per-tick (capped) theta so a
                    # maintained, differentiated rule CLEARS the conflict gate under
                    # realistic match-crowding instead of being suppressed wholesale
                    # (654d: maintained != active when theta(n_matched) >> the fixed
                    # maintenance_floor). Floor to max(maintenance_floor, theta+eps),
                    # clamped to [0,1]. theta uses _last_n_matched (the most recent
                    # gate count); with the cap keeping theta < 1.0 the floor is
                    # reachable, so the next gate_and_select admits the pool.
                    theta_target = self._theta_for(self._last_n_matched)
                    coupled_floor = min(
                        1.0, theta_target + MAINTENANCE_THETA_EPS
                    )
                    if r.availability < coupled_floor:
                        r.availability = coupled_floor
            else:
                r.availability *= (1.0 - decay)
            r.availability = float(min(1.0, max(0.0, r.availability)))
            # Mint-youth protection: a freshly minted rule is retirement-protected
            # for `protect` ticks so a 2nd differentiated rule has time to
            # co-accumulate before the first is retired (the direct 654b
            # "rule drops below floor before a 2nd rule co-accumulates" fix).
            if r.availability < self._retire_floor and (
                cur_step - r.minted_step
            ) >= protect:
                to_retire.append(idx)
        for idx in to_retire:
            del self._rules[idx]
            self._n_retired += 1

    # ------------------------------------------------------------------
    # One-call agent tick
    # ------------------------------------------------------------------
    def step(
        self,
        context: torch.Tensor,
        action_object_idx: int,
        outcome_signal: float = 0.0,
        arc062_seed: Optional[float] = None,
        simulation_mode: bool = False,
    ) -> torch.Tensor:
        """Waking tick: credit the previous active set, mint on recurrence, gate
        + select against the current context, and return the differentiated
        rule_state vector [1, rule_dim] for SD-033a.

        MECH-094: simulation_mode=True is a no-op (returns zeros, no state
        advance) -- replay / DMN paths must not mint or credit.
        """
        ctx = context.detach().reshape(-1)
        if simulation_mode:
            self._n_simulation_skipped += 1
            return torch.zeros(1, self.rule_dim, dtype=ctx.dtype, device=ctx.device)
        self._step += 1
        # Credit the rules that were active on the PREVIOUS tick, using the
        # outcome that has now arrived.
        self.credit(outcome_signal, step=self._step)
        # CREATE.
        self._last_minted_this_step = self._maybe_mint(
            ctx, action_object_idx, arc062_seed
        )
        # GATE + SELECT against the current context.
        active = self.gate_and_select(ctx, step=self._step)
        # OUTPUT.
        return self.active_rule_state(active)

    # ------------------------------------------------------------------
    # Lifecycle / diagnostics
    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Per-episode reset: clear the slot pool, recurrence counters, clock.

        When config.persist_rules_across_episode_reset is True the field is a
        no-op on reset -- the live rule pool (_rules), recurrence counters
        (_recurrence), and clock (_step) PERSIST across episodes so the field
        matures a differentiated pool at behavioural-runtime episode lengths
        (failure_autopsy_V3-EXQ-654_2026-06-09). Default False reproduces the
        legacy per-episode wipe bit-identically.
        """
        if self.config.persist_rules_across_episode_reset:
            return
        self._rules.clear()
        self._recurrence.clear()
        self._step = 0
        self._last_n_active = 0
        self._last_n_matched = 0
        self._last_minted_this_step = 0
        self._n_active_steps = 0

    def n_active_rules(self) -> int:
        return sum(1 for r in self._rules.values() if r.active)

    def max_pairwise_rule_distance(self) -> float:
        """Max pairwise L2 distance between minted rule embeddings (diagnostic
        for REPRESENT separability; the anti-collapse headline number)."""
        embs = [r.rule_embedding.reshape(-1) for r in self._rules.values()]
        if len(embs) < 2:
            return 0.0
        m = torch.stack(embs, dim=0)
        d = torch.cdist(m.unsqueeze(0), m.unsqueeze(0)).squeeze(0)
        return float(d.max().item())

    def maintained_reactivation_threshold(self) -> float:
        """Availability bar a rule must clear to count as maintained-and-
        reactivatable (crf-availability-maintenance readout).

        Sentinel maintained_reactivation_threshold <= 0 derives it from the
        single-match gate floor -- the threshold a rule clears when its context
        recurs and it is the sole match (the realistic reactivation case for a
        narrowly-tuned differentiated rule).
        """
        thr = float(self.config.maintained_reactivation_threshold)
        if thr > 0.0:
            return thr
        if self.config.mature_pool_dynamics:
            return float(self.config.mature_tolerance_floor)
        return float(self.config.tolerance_floor)

    def maintained_reactivatable_rules(self) -> List[CandidateRule]:
        """Minted rules whose maintained availability WOULD clear threshold if
        their context recurred -- independent of whether that context is present
        this tick (the activity-silent 'is the rule still in the store?' question;
        targeted_review_arc_063_crf_rule_cell_persistence prescription 2)."""
        thr = self.maintained_reactivation_threshold()
        return [r for r in self._rules.values() if r.availability >= thr]

    def maintained_pairwise_distance(self) -> float:
        """Max pairwise L2 distance among the MAINTAINED-reactivatable rules'
        embeddings -- the differentiation of the silently-held pool. With
        maintained count this forms the CRF-readiness gate the 666-successor
        scores: crf_maintained_pairwise_dist > floor AND
        crf_n_maintained_reactivatable >= 2 (replaces the crf_frac_active >= 0.30
        target, which is the averaged-activity artefact for a sparsely-matched
        differentiated pool)."""
        embs = [
            r.rule_embedding.reshape(-1) for r in self.maintained_reactivatable_rules()
        ]
        if len(embs) < 2:
            return 0.0
        m = torch.stack(embs, dim=0)
        d = torch.cdist(m.unsqueeze(0), m.unsqueeze(0)).squeeze(0)
        return float(d.max().item())

    def get_state(self) -> dict:
        """Diagnostic snapshot for experiment manifests."""
        _n_maintained = len(self.maintained_reactivatable_rules())
        return {
            "crf_n_slots_minted": len(self._rules),
            "crf_n_minted_total": self._n_minted,
            "crf_n_retired_total": self._n_retired,
            "crf_n_active_last": self._last_n_active,
            "crf_n_matched_last": self._last_n_matched,
            "crf_n_minted_this_step": self._last_minted_this_step,
            "crf_n_simulation_skipped": self._n_simulation_skipped,
            "crf_max_pairwise_rule_dist": self.max_pairwise_rule_distance(),
            # Fraction of ticks on which >=1 rule fired. With crf_max_pairwise_rule_dist
            # this forms the CRF-readiness gate (frac_active >= 0.30 AND
            # max_pairwise_rule_dist > floor) that must clear before a GAP-B
            # behavioural falsifier (654c successor) is scored.
            "crf_frac_active": (
                self._n_active_steps / self._step if self._step > 0 else 0.0
            ),
            "crf_n_active_steps": self._n_active_steps,
            "crf_step": self._step,
            # CRF-gate amend (654d successor) diagnostics: the (capped) conflict-gate
            # threshold at the last match count + the match cutoff actually used.
            # crf_frac_active >= 0.30 with crf_last_theta reachable (<= 1.0) is the
            # gate-firing readiness signal the 654e falsifier's C1c precondition reads.
            "crf_last_theta": self._theta_for(self._last_n_matched),
            "crf_gate_match_threshold": self._gate_match_threshold(),
            # crf-availability-maintenance readout (the maintained-pool metric that
            # REPLACES crf_frac_active as the readiness criterion per the B-leaning
            # lit verdict). crf_frac_active above is retained as the secondary
            # active-on-match efficiency readout, not the persistence criterion.
            "crf_n_maintained_reactivatable": _n_maintained,
            "crf_maintained_pairwise_dist": self.maintained_pairwise_distance(),
            "crf_frac_maintained": (
                _n_maintained / self.config.n_slots
                if self.config.n_slots > 0
                else 0.0
            ),
            "crf_maintained_reactivation_threshold": (
                self.maintained_reactivation_threshold()
            ),
        }
