## consummatory_aware_reference_policies: experiments/_lib/capability_eval.py -- IMPLEMENTED (2026-07-25)
- consummatory_aware_reference_policies: experiments/_lib/capability_eval.py -- IMPLEMENTED
  2026-07-25. OraclePolicy + LocalViewGreedyPolicy select the distinct CONSUME action while
  standing on a resource cell in consummatory mode, via the shared helper
  _consummatory_consume_action(env) (gated on env.consummatory_act_enabled; reads the env's
  _on_consumable_resource "should consume now" flag, falling back to _agent_on_resource_cell()).
  WHY: with mech457_consummatory_act, arrival on a resource cell only AFFORDS consumption -- the
  benefit needs a CONSUME action. The hand-coded greedy policies otherwise return "stay" on the
  target cell and forage 0 in the consummatory env, which would leave the readiness anchors
  reading the env as unsolvable and a BC install cloned from the LocalViewGreedyPolicy
  demonstrator unable to take. This is what makes the RETENTION (BC-install) framing of leg 4
  runnable in the consummatory env (the demonstrator-free 781 framing does not need it).
  Measured: in the D3 consummatory env local_view_greedy forages ~13.5/ep, greedy_oracle ~13.0
  (vs ~17.9 / ~18.7 non-consummatory -- CONSUME costs one step per resource), random_walk ~0.33.
  So the consummatory achievable ceiling / install band is ~13, NOT the non-consummatory 20.933;
  the leg records the LIVE consummatory anchors as denominators. RandomPolicy is deliberately NOT
  made consummatory-aware (it is the floor).
  Backward compatible: default OFF (helper returns None when consummatory_act_enabled is False)
  -> byte-identical; the pre-change 5-action env has no CONSUME action. capability_eval.py is in
  experiments/_lib/**, so this busts arm fingerprints fleet-wide for FUTURE runs (expected;
  runs are byte-identical, only substrate_hash changes).
  4 contracts CG1-CG4: tests/contracts/test_consummatory_aware_policies.py (non-consummatory
  no-op + unchanged foraging; consummatory greedy policies forage >= floor; helper semantics;
  ASCII). All 4 pass locally (1.41s).
  Phased training required: no. MECH-094: not applicable.
  See REE_assembly/docs/architecture/sd_mech457_approach_extinction.md (companion build section).
