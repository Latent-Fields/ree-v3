## MECH-339 C1 Composite Cue + Outshining Gate (2026-05-19)
- MECH-339: hippocampal.composite_retrieval_cue_outshining_gate -- IMPLEMENTED 2026-05-19.
  Module: ree_core/hippocampal/ghost_goal_bank.py (GhostGoalBank.rank,
  _outshine_gate, _context_salience_for_anchor).
  Constraint 1 of the retrieval-cue reframe (ghost_goal_search.md
  Section 0.2; claims.yaml MECH-339 / ARC-078). The landed MECH-292 bank
  matched the retrieval cue by z_goal cosine only and ignored the SD-039
  payload fields the match does not use. MECH-339 adds a composite cue: a
  context channel built from the stored-but-match-unused payload, combined
  by an OUTSHINING gate so a strong direct goal_match suppresses the
  context channel (Smith & Vela 2001) rather than it being summed in with
  fixed weight.
  Smallest substrate step: the context channel is sourced from
  payload.arousal_tag only -- the one SD-039 field that is both already
  stored and entirely unused by the bank. last_vs is deferred (already
  consumed by the recoverability channel; reuse double-counts) and a
  `cause` tag is deferred (not present in the implemented
  AnchorGoalPayload -- design-sketch only; an SD-039 payload extension,
  out of scope here).
  Data flow: anchor.goal_payload.arousal_tag
    -> context_salience = 1 - exp(-arousal_tag / arousal_scale)  in [0,1)
    -> gate = clip_[0,1]((outshine_pivot - goal_match) / outshine_pivot)
    -> context_term = context_weight * gate * context_salience
    -> ghost_priority += context_term (overall form stays an additive sum
       of independently gateable channels; Constraint 2 unaffected -- the
       outshining is a within-channel multiplier, not a product across
       wanting/goal_match). New components/component_sums key "context"
       present only when the master switch is on.
  Config: GhostGoalBankConfig.use_composite_cue_outshining (default False;
    set True to enable), context_weight (default 0.0), outshine_pivot
    (default 0.5), arousal_scale (default 1.0). No rank() signature
    change, no anchor_set/agent-loop wiring change, no new payload field.
  Backward compatible: with the master switch off (or context_weight 0.0)
    the bank is bit-identical to pre-MECH-339 -- no "context" key in
    components/diagnostics, priorities unchanged. Verified: inline smoke
    (4-key components + bit-identical priority off; gate 0 at/above pivot,
    monotone decreasing; context term exactly 0.0 for an anchor with a
    strong direct match, > 0 for a weak-match high-arousal anchor;
    components sum == ghost_priority) and v3_exq_496 MECH-292 validation
    --dry-run PASS with the new config fields present.
  Biological basis: encoding specificity is real but moderate and is
    outshone by a strong direct cue (Smith & Vela 2001 meta-analysis;
    Tulving & Thomson 1973); arousal_tag is the LaBar & Cabeza 2006
    per-trace arousal tag.
  Phased training required: no (pure arithmetic; no trainable params).
  MECH-094: not applicable -- read-only over payloads whose provenance was
    set at SD-039 population time (sense() passes simulation_mode=False);
    the bank has no write path.
  Validation experiment: V3-EXQ-594 queued (diagnostic, priority 10;
    experiments/v3_exq_594_mech339_composite_cue_outshining_validation.py;
    4 sub-tests = the 4 MECH-339 falsifiable predictions; smoke PASS 4/4).
  See MECH-339 (this claim), ARC-078 (parent: cue-addressed retrieval
  system), MECH-292 (the bank whose goal_match this composes), SD-039
  (supplies arousal_tag), MECH-230 (z_goal = direct-cue content).
