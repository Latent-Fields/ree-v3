## SD-039 Dual-Trace Anchor Goal-Snapshot Payload -- Substrate Foundation (2026-04-26)
- SD-039 substrate-side: hippocampal.anchor_goal_snapshot_payload --
  IMPLEMENTED 2026-04-26 (substrate side only; module-level write-site
  population deferred). Module: ree_core/hippocampal/anchor_set.py
  (AnchorGoalPayload dataclass; Anchor.goal_payload field +
  Anchor.goal_match() helper; AnchorSet.write_anchor / mark_inactive /
  reset_region / consume_boundary_events accept optional goal_payload;
  AnchorSet.query_by_goal_match() helper). Resolves the substrate-side
  prerequisite for MECH-292 (ranked ghost-goal bank) and MECH-293
  (waking ghost-goal probes); both downstream consumers can now query
  preserved motivational payloads on dual-trace anchors instead of
  reasoning over staleness-only signatures.
  Architectural design (claims.yaml SD-039): "Current MECH-269 anchors
  preserve z_world and active/inactive status, while MECH-284 preserves
  a region-level staleness scalar. SD-039 adds a compact motivational
  payload to each anchor at write, remap, or invalidation time:
  z_goal_snapshot, wanting strength, arousal tag, and optional last_vs /
  staleness_at_write. The payload is preserved across mark_inactive, so
  inactive anchors remain queryable as blocked or deferred goal traces."
  Refresh-on-invalidate semantic: when a non-None goal_payload is
  supplied to mark_inactive or reset_region, the payload is written
  onto the outgoing anchor BEFORE inactivation. Existing payload is NOT
  cleared on inactivation -- inactive anchors retain motivational
  identity (the entire point of dual-trace preservation). On
  reset_region, the same payload is written to BOTH the outgoing
  inactive trace and the new active anchor (cause-of-blockage payload
  on the outgoing; new motivational state on the new active).
  AnchorGoalPayload fields:
    z_goal_snapshot: Optional[torch.Tensor] -- detached clone of z_goal
      at write time. None when no goal active.
    wanting_strength: float -- VALENCE_WANTING readout at the anchor
      location (or last cached drive*benefit proxy).
    arousal_tag: float -- BLA arousal-tag scalar at write time.
    last_vs: Optional[float] -- last V_s_anchor reading on the parent
      (scale, stream_mixture) family at write/remap/invalidate time.
    staleness_at_write: Optional[float] -- MECH-284 region staleness at
      write time.
    payload_written_step: int -- HippocampalModule tick index at write.
  Anchor.goal_match(current_z_goal) -> float: cosine similarity between
    stored z_goal_snapshot and supplied current_z_goal, clipped to
    non-negative (motivational-relevance is a non-negative signal, not
    a signed correlation). Returns 0.0 when payload is None, snapshot
    is None, current is None, or norms are degenerate.
  AnchorSet.query_by_goal_match(current_z_goal, threshold=0.0,
    scale=None, active_only=False) -> List[Tuple[Anchor, float]]:
    scans the dual-trace pool (active + inactive by default) and
    returns anchors paired with non-zero goal_match scores, sorted by
    score descending. This is the substrate hook MECH-292 will consume.
    SD-039 itself does NOT rank or implement the bank; ranking by
    ghost_priority ~ wanting * goal_match * staleness * recoverability
    lives in MECH-292's module (deferred).
    threshold=0.0 (default) excludes payload-less / norm-zero traces.
    threshold=-1.0 includes every anchor with a payload regardless of
    match (diagnostic path).
    active_only=True restricts to the active half of the dual-trace
    pool (legacy active_anchors() behaviour).
  Config: AnchorSetConfig.use_sd039_anchor_payload (bool, default False).
    Substrate-side flag. Module-level callers populate the payload from
    GoalState / VALENCE_WANTING / amygdala arousal tags when this flag
    is True; with flag OFF callers pass goal_payload=None and behaviour
    is bit-identical to pre-SD-039.
  Backward compatible: anchor.goal_payload defaults to None. Existing
    write_anchor / mark_inactive / reset_region call sites that omit
    the new goal_payload kwarg work unchanged. 164/164 contracts +
    7/7 preflight PASS with flag implicitly off (2026-04-26).
  No trainable parameters. Pure dataclass + cosine arithmetic.
  Out of scope this session (deferred follow-on):
    - Module-level write-site wiring: REEAgent / HippocampalModule
      should populate goal_payload from GoalState (z_goal_snapshot),
      ResidueField VALENCE_WANTING (wanting_strength), and amygdala
      arousal tags (arousal_tag) on anchor write/remap/invalidate.
      The substrate accepts the payload; the population layer is a
      separate session.
    - MECH-292 ranked ghost-goal bank computation.
    - MECH-293 waking ghost-goal probe budget allocation.
    - ARC-060 hybrid field+bank architectural framing.
    - Validation EXQ exercising the falsifiable test (after reward
      relocation or path blockage, inactive anchors on the formerly
      valid approach path retain non-zero goal_match with current
      z_goal while unrelated stale anchors do not).
  Biological basis (lit-pull SYNTHESIS, evidence/literature/
    targeted_review_ghost_goal_search/): Berridge 1998 + Barch 2010
    (persistent wanting / goal representations); Mattar & Daw 2018
    (utility-prioritised replay); Pfeiffer & Foster 2013 (goal-biased
    path search); Gillespie 2021 (broad / non-current trace
    reactivation); Muessig 2019 (one sequence generator across waking
    + offline modes); Berkowitz 1989 (frustration / unresolved goal
    persistence).
  MECH-094: substrate-side scope; the goal_match query and the
    goal_payload dataclass are passive. Population sites (deferred)
    will gate writes via simulation_mode / hypothesis_tag at the
    REEAgent / HippocampalModule call site (same call-site-scoping
    pattern as MECH-269 Phase 1 / 2 ii / 2 iii, MECH-288, MECH-287,
    MECH-284).
  Contract tests: tests/contracts/test_sd_039_anchor_payload.py 10/10
    PASS (S1 imports + symbol presence; S2 default backward-compat;
    S3 ON-payload-attached-on-write; S4 payload-survives-mark_inactive;
    S5 goal_match-zero-on-null-inputs; S6 goal_match-cosine-correctness;
    S7 query-returns-active-and-inactive-sorted; S8 query-empty-for-
    None-current; S9 reset_region-refreshes-payload-on-both-traces;
    S10 per-episode-reset-clears-payloads).
  Validation experiment: deferred until module-level write-site wiring
    lands. The substrate-side foundation is observable through
    contracts; behavioural validation requires the population layer.
  Design doc: REE_assembly/docs/architecture/sd_039_anchor_goal_payload.md
  See SD-039, MECH-269 (Phase 2 ii dual-trace anchor substrate being
    extended), MECH-216 (predictive wanting -- input to wanting_strength
    population), MECH-230 (z_goal latent structure -- z_goal_snapshot
    source), MECH-284 (region staleness -- staleness_at_write source),
    MECH-292 (downstream ghost-goal bank consumer), MECH-293 (downstream
    waking ghost-goal probe consumer), ARC-060 (hybrid field+bank
    architectural framing), MECH-094 (call-site scoping for population
    layer).
