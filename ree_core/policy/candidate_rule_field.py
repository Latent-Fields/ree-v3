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
        matched: List[CandidateRule] = [
            r
            for r in self._rules.values()
            if self._cosine(context, r.context_tag) >= self.config.context_match_threshold
        ]
        n_matched = len(matched)
        # Conflict-gate pair: legacy 0.3 + 1.0*n_competing gives theta>=1.3 > 1.0
        # max availability whenever >=2 rules match, so >=2 matched rules can
        # NEVER both be active (the latent 654b deadlock). mature_pool_dynamics
        # uses 0.15 + 0.25*n -> theta(1)=0.40, theta(2)=0.65, theta(3)=0.90, all
        # reachable, so a differentiated pool of >=2 matched rules can co-fire.
        if self.config.mature_pool_dynamics:
            theta_floor = self.config.mature_tolerance_floor
            theta_gain = self.config.mature_tolerance_conflict_gain
        else:
            theta_floor = self.config.tolerance_floor
            theta_gain = self.config.tolerance_conflict_gain
        for r in self._rules.values():
            r.active = False
        active: List[CandidateRule] = []
        for r in matched:
            n_competing = n_matched - 1
            theta = theta_floor + theta_gain * n_competing
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
        to_retire: List[int] = []
        for idx, r in self._rules.items():
            if r.eligibility > 1e-6:
                w = alpha * r.eligibility
                r.availability = (1.0 - w) * r.availability + w * target
            r.eligibility *= decay_e
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

    def get_state(self) -> dict:
        """Diagnostic snapshot for experiment manifests."""
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
        }
