## SD-059 / MECH-358: relief/safety escape-affordance bridge (directed escape for the MECH-357 gate) (2026-06-08)
- SD-059 (architecture) / MECH-358 (mechanism): defensive_action.escape_affordance_bridge
  -- IMPLEMENTED 2026-06-08 (substrate; v3_pending until the 4-arm validation EXQ
  PASSes). Closes the V3-EXQ-603h directed-escape gap. Routed by
  failure_autopsy_V3-EXQ-603h_2026-06-08 + thought_intake_2026-06-07_relief_safety_escape_affordance_bridge.
  SD-058/MECH-357 suppress the MECH-279 freeze but avoidance_efficacy is a GLOBAL
  SCALAR that only penalises the no-op class -- compute_action_bias by design "does
  NOT compute the escape direction". 603h (engaged-but-insufficient, readiness met):
  the gate suppressed freeze on all INTACT seeds but G_H_INTACT=0/3; seed-43 reached
  scalar efficacy 0.633 and survived WORST (11.0). The agent un-froze without
  acquiring a DIRECTED escape. Moscarello & LeDoux 2013: active avoidance needs the
  LA/BA->NAcc relief/safety action-credit half, not only suppression. REE owns relief
  (MECH-302/SD-050) + safety (MECH-303/304/SD-052/SD-051) but they were UNWIRED to
  avoidance -- this is the wiring.
  Module: ree_core/pfc/escape_affordance_bridge.py (EscapeAffordanceBridge +
  EscapeAffordanceBridgeConfig + EscapeAffordanceBridgeOutput). Pure-arithmetic
  regulator (no nn.Module, no trained params, no gradient flow); sibling to the
  SD-058/MECH-357 gate in ree_core/pfc/. Extends MECH-357's scalar avoidance_efficacy
  into a per-FIRST-ACTION-CLASS credit table (the minimal V3 rendering of
  escape_affordance[action] -- the directed escape direction in the discrete action
  space; location/policy indexing deferred, disambiguated by the validation's
  nav-competence control).
  Two independently-toggleable halves (so the 4-arm validation dissociates):
    RELIEF half (MECH-302-consistent): a directed action under threat that DROPS
      z_harm_a (delta = prev - now > relief_reward_floor) credits
      relief_affordance[action_class] (EMA toward 1) -- the d(z_harm_a)/dt<0 signal
      attributed to the specific action.
    SAFETY half (MECH-303/304-consistent): a directed action after which threat is
      absent (threat_scale <= 0) credits safety_affordance[action_class]
      (response-produced safety / conditioned inhibition).
  Approach bonus (the directed escape): under FUTURE threat (threat_scale > 0), E3
  receives a per-candidate NEGATIVE (favoured; REE lower-is-better) score-bias toward
  each candidate whose first-action class carries combined affordance credit:
  -clamp(approach_gain * threat_scale * combined_affordance[class], 0, bias_scale);
  the no-op/freeze class never gets a bonus. THREE guards: bias_scale clamp (cannot
  dominate the chain), threat-context gate (exactly zero when safe -> never swamps
  food/goal approach), per-tick leak (forgetting -> no pathological habit loop).
  DISTINCT from reflexive escape (SD-037 orexin / MECH-281 urgency -- threat/arousal
  reflexes) and from the generic relief/safety rows (MECH-302/303/304 fire on the
  CURRENT state); this is learned-efficacy-gated DIRECTED approach binding an action
  to relief/safety for future-threat use.
  Config (REEConfig + from_dims; all no-op default, bit-identical OFF):
  use_escape_affordance_bridge (False, master) + use_escape_relief_credit (True) +
  use_escape_safety_credit (True) + escape_relief_learn_rate (0.1) +
  escape_safety_learn_rate (0.1) + escape_bridge_leak_rate (0.01) +
  escape_relief_reward_floor (1e-4) + escape_threat_floor (0.1) + escape_threat_ref
  (0.5) + escape_approach_gain (0.1) + escape_bias_scale (0.1) + escape_noop_class
  (0). n_action_classes set from config.e2.action_dim at agent build.
  Agent wiring (ree_core/agent.py): build self.escape_affordance_bridge when the
  master flag is on; _eab_last_action_class cache; update() in sense() after the
  MECH-357 eligibility update (reads new_latent.z_harm_a; MECH-094 no-op under
  hypothesis_tag; bridge directedness = internal non-noop check, independent of
  whether MECH-357 is enabled); approach-bias composed after the MECH-357 action-bias
  block; action-class cache after self._last_action; reset() clears the within-episode
  trace + cache (affordance tables PERSIST across episodes, same as MECH-357 efficacy).
  Curriculum wiring (experiments/scaffolded_sd054_onboarding.py): the bridge runs
  automatically inside the existing Stage-H run_hazard_avoidance training loop once the
  agent is built with use_escape_affordance_bridge=True;
  HazardAvoidanceResult.escape_bridge_state surfaces bridge.get_state() so the
  validation manifest can gate non-vacuity (relief/safety credit actually incremented)
  before scoring G_H.
  Backward compatible: use_escape_affordance_bridge=False by default ->
  agent.escape_affordance_bridge is None; sense update + approach-bias all skipped ->
  bit-identical. Verified: default == explicit-False action stream (8-tick); 603h
  --dry-run runs unchanged (bridge OFF); scaffolded contracts 91/91; 8 new SD-059
  contracts in tests/contracts/test_sd_059_escape_affordance_bridge.py PASS. (Two
  failures in the full contract run are pre-existing/unrelated: control_vector C4
  fails at baseline; scaffolded C12 is test-ordering flakiness -- passes isolated and
  in its own file with these changes.)
  Phased training: N/A at the substrate level (pure-arithmetic regulator). BUT the
  relief detector reads z_harm_a / the world-forward, so the VALIDATION experiment
  keeps the SD-056 e2 contrastive warmup in P0 + a relief-credit non-vacuity readiness
  gate -- without a trained encoder the credit re-starves (603h n_credit 6/0 on 2/3
  seeds).
  MECH-094: both compute methods + update() no-op under simulation_mode (replay/DMN
  must not credit escape affordances or bias action selection on imagined outcomes).
  Call-site: sense() + select_action() are waking-only; no replay/memory write surface.
  Evidence-staleness: NOT triggered (no-op-default flag; no dependent claim's measured
  mechanism changed).
  Validation experiment: V3-EXQ-603i substrate-readiness diagnostic (claim_ids=[],
  queued via /queue-experiment) -- thought-intake Section 5 4-arm
  (ARM_BASE_IA_ONLY / ARM_RELIEF_BRIDGE / ARM_SAFETY_BRIDGE / ARM_RELIEF_SAFETY_BRIDGE)
  + a nav-competence positive control so a flat G_H across all bridge arms is
  attributable to a survival/navigation ceiling rather than the bridge. Non-vacuity
  gate: each enabled bridge half must increment its credit before G_H is scored.
  Acceptance: G_H >= 2/3 AND improves over ARM_BASE_IA_ONLY; secondary P1 survival
  transfer without over-avoidance/starvation. PASS unblocks goal_pipeline GAP-2 +
  ARC-060 / MECH-320 / ARC-068 / SD-054-readiness retests.
  Design doc: REE_assembly/docs/architecture/sd_059_escape_affordance_bridge.md
  Autopsy: REE_assembly/evidence/planning/failure_autopsy_V3-EXQ-603h_2026-06-08.{md,json}.
  See SD-058 / MECH-357 (the gate this extends scalar -> affordance-indexed), MECH-302
  / SD-050 (relief), MECH-303/304 / SD-052/SD-051 (safety), MECH-279 (PAG freeze),
  SD-011 (z_harm_a), SD-037 / MECH-281 (reflexive escape; distinct), SD-056
  (action-conditional divergence; relief-detector prerequisite), ARC-060 / MECH-320 /
  ARC-068 (unblocked on PASS), scaffolded_sd054_onboarding (Stage-H driver),
  goal_pipeline:GAP-2 (the survival leg this closes), V3-EXQ-603h (the FAIL this
  addresses), MECH-094 (call-site scoping).
