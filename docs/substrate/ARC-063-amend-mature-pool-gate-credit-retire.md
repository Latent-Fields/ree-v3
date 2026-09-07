## ARC-063 AMEND: mature-pool gate/credit/retire dynamics (V3-EXQ-654b GAP-B maturity) (2026-06-11)
- ARC-063 amend: policy.rule_apprehension_layer.candidate_rule_field mature-pool
  gate/credit/retire dynamics -- IMPLEMENTED 2026-06-11. Modules:
  ree_core/policy/candidate_rule_field.py (CandidateRuleFieldConfig + _maybe_mint +
  gate_and_select + credit + __init__ + reset + get_state), ree_core/agent.py
  (CRF config build site + CRF tick-site context routing), ree_core/utils/config.py
  (REEConfig fields + from_dims). Routed by the confirmed
  failure_autopsy_V3-EXQ-654b_2026-06-11 (the ARC-062 substrate_queue
  recommended_substrate_queue_entry hand-off; amend_target_impl = ARC-063
  CandidateRuleField).
  ROOT CAUSE (budget reading EXHAUSTED, not a falsification): the 2026-06-09
  cross-episode persistence amend preserves the pool ACROSS episodes but does
  nothing about WITHIN-run churn. 654 (per-episode wipe) / 654a (crf_persist) /
  654b (crf_persist + 240 ep) all leave crf_frac_active pinned ~0.12-0.14 AND
  crf_max_pairwise_rule_dist EXACTLY 0.0 on every ARM_ON cell -- the field never
  holds >=2 rules present despite 452-1014 cumulative mints (mint -> brief life ->
  retire -> re-mint). Three drivers: (1) retire-churn PRIMARY (availability_decay
  0.005 + negative-outcome credit drive availability below _retire_floor=0.5*
  tolerance_floor=0.15 before a 2nd differentiated rule co-accumulates); (2)
  mint-block SECONDARY (context_match_threshold 0.5 lets one rule cover the
  low-spread z_world space; CRF reads raw z_world so the ARC-065 GAP-A fix does
  not reach it); (3) conflict-gate deadlock LATENT (tolerance_floor 0.3 +
  tolerance_conflict_gain 1.0 -> theta>=1.3 > 1.0 max availability whenever >=2
  rules match -> >=2 matched rules can NEVER both be active).
  THE FIX (no-op-default; bit-identical OFF; two opt-in flags):
    (1) crf_mature_pool_dynamics (CandidateRuleFieldConfig.mature_pool_dynamics,
        default False) routes a bundle of recalibrated knobs consulted ONLY under
        the master flag: mature_tolerance_floor 0.15 / mature_tolerance_conflict_gain
        0.25 -> theta(1)=0.40, theta(2)=0.65, theta(3)=0.90 all < 1.0 (deadlock
        fix -- up to 4 matched rules co-fire); mature_availability_decay 0.001
        (5x slower); mature_retire_floor 0.05 (absolute, decoupled from
        tolerance_floor); mature_availability_alpha_negative 0.02 (asymmetric --
        negative outcomes erode availability gently); mature_mint_protection_ticks
        30 (a freshly minted rule is retirement-protected so a 2nd differentiated
        rule co-accumulates -- the direct "drops below floor before a 2nd
        co-accumulates" fix); mature_mint_block_threshold 0.8 (DECOUPLED from the
        0.5 retrieval threshold so a differentiated mint is blocked only by a very
        similar existing rule). All mirrored as REEConfig.crf_mature_* +
        from_dims; agent build site forwards via getattr fallback (absent flat
        attr -> bit-identical).
    (2) crf_context_from_e2_world_forward (REEConfig, default False) sources the
        CRF mint/match context from e2.world_forward(z_world, prev_action_onehot)
        (action-regime-separated, no_grad, falls back to raw z_world when
        prev-action is unset / e2 absent) instead of raw z_world -- mirrors the
        ARC-065 GAP-A re-sourcing so the mint-block does not collapse under low
        raw-z_world spread.
  CRF-readiness readout: get_state() now emits crf_frac_active (fraction of ticks
  with >=1 active rule) + crf_n_active_steps alongside crf_max_pairwise_rule_dist.
  The Step-8 readiness diagnostic asserts crf_max_pairwise_rule_dist > floor AND
  crf_frac_active >= 0.30 -- the gate that MUST clear before any GAP-B behavioural
  falsifier (654c successor) is scored.
  Backward compatible: both flags default False -> all mature_* knobs inert, CRF
  context = raw z_world -> bit-identical. 16/16 CRF contracts (11 prior + 5 new:
  C12 mature default-OFF bit-identical + frac_active readout / C13 conflict-gate
  admits >=2 matched rules [legacy theta=1.3 deadlocks, mature theta=0.40 admits
  both] / C14 the 654b inversion [legacy max_pairwise_dist=0.0 / 1 rule / READY
  False at cos-0.7 collapsed regime, mature >=2 co-present / READY True] / C15
  mint-youth protection survives a below-floor fresh rule + asymmetric negative
  credit gentler / C16 from_dims + agent wiring of both flags + e2-context path)
  + 7/7 preflight PASS; full suite 1004 passed + 1 pre-existing control_vector C4
  flake (CONFIRMED failing on a clean tree via git stash, unrelated -- CRF is None
  in every existing experiment). Activation smoke (two regimes at cos 0.7,
  frequent hazard negatives): legacy n_present=1 / max_pairwise_dist=0.0 /
  READY=False (the 654b signature) -> mature n_present=2 / max_pairwise_dist=1.49 /
  frac_active~0.99 / READY=True.
  Phased training: N/A (pure-arithmetic regulator; no learned parameters; the
  flags change only gate/credit/retire arithmetic + the context key). MECH-094:
  unchanged -- the field's per-tick state-advancing methods keep their
  simulation_mode no-op gate; the e2.world_forward context read is no_grad on the
  waking select_action path (no replay write surface).
  Evidence-staleness (Step 8.5): NOT triggered -- no-op-default flags; every
  existing experiment uses the defaults (legacy dynamics + raw z_world context),
  so no dependent claim's measured mechanism changed. KEEP all evidence.
  GOVERNANCE: MECH-309/ARC-062/ARC-063 NEITHER validated NOR weakened; stay
  candidate / substrate_ceiling / v3_pending / pending_retest_after_substrate.
  claims.yaml / governance state UNTOUCHED (substrate-only amend; the autopsy's
  failure record + amend hand-off land in substrate_queue.json via /governance).
  Validation experiment: V3-EXQ-654c-readiness CRF-readiness substrate diagnostic
  (claim_ids=[]; mature + e2-context arms vs OFF; asserts crf_max_pairwise_rule_dist
  > floor AND crf_frac_active >= 0.30 on >=2/3 seeds) queued via /queue-experiment
  as the Step-8 follow-on. The 654c GAP-B behavioural re-run is a SEPARATE later
  /queue-experiment session GATED on that readiness PASS.
  Design doc: REE_assembly/docs/architecture/arc_063_candidate_rule_field.md
  (mature-pool dynamics amend section, 2026-06-11).
  Autopsy: REE_assembly/evidence/planning/failure_autopsy_V3-EXQ-654b_2026-06-11.{md,json}.
  See ARC-063 (parent v1 + cross-episode persistence amend above), MECH-309 /
  ARC-062 (the GAP-B claims this matures the falsifier for; unweakened), SD-033a
  (rule_state consumer), ARC-065 GAP-A (the e2.world_forward context-source
  pattern this mirrors), GAP-D rule_bias_head trainability (the 654c propagation
  arm), V3-EXQ-654b (the FAIL this amend addresses), V3-EXQ-639 (PASS proving the
  field differentiates when matured), MECH-094 (call-site scoping; unchanged).
