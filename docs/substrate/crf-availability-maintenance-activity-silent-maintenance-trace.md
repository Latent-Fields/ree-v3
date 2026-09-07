## crf-availability-maintenance: activity-silent maintenance trace + maintained-pool readout (V3-EXQ-666 successor; ARC-063 amend) (2026-06-11)
- crf-availability-maintenance -- IMPLEMENTED 2026-06-11. Module:
  ree_core/policy/candidate_rule_field.py (CandidateRuleFieldConfig + _maybe_mint +
  credit + new maintained_reactivation_threshold / maintained_reactivatable_rules /
  maintained_pairwise_distance + get_state), ree_core/utils/config.py (REEConfig +
  from_dims), ree_core/agent.py (CRF config build site). Routed by the confirmed
  failure_autopsy_V3-EXQ-666_2026-06-11 + the targeted lit-pull
  evidence/literature/targeted_review_arc_063_crf_rule_cell_persistence/ (SYNTHESIS
  verdict B-leaning hybrid). New substrate_queue.json entry crf-availability-maintenance.
  ROOT CAUSE (differentiation<->persistence tension, NOT a falsification): the 654b
  mature-pool amend solves DIFFERENTIATION only via crf_context_from_e2_world_forward
  (V3-EXQ-666 ARM_2: 10-16 distinct rules, crf_max_pairwise_rule_dist 1.71) but WORSENS
  PERSISTENCE -- once each rule matches only a narrow context slice, its match-triggered
  availability EMA never accumulates above theta between sparse matches and decays in the
  gaps (mature_availability_decay 0.001/tick), so crf_frac_active collapses to 0.016
  (worse than the undifferentiated legacy 0.125); 0/3 readiness cells in every arm. The
  next binding constraint is availability MAINTENANCE under sparse matching (a PFC
  sustained-activity / activity-silent analog), NOT the conflict-gate theta (654b fixed
  that). The match-triggered-EMA scheme has the SYMBOL of a rule cell but not its
  maintenance functional role.
  LIT VERDICT (the A-vs-B fork): B (activity-silent synaptic maintenance, Mongillo 2008 /
  Stokes 2015 / Lundqvist 2018) with a bounded role for A (persistent firing reserved for
  the single ENGAGED rule; Funahashi 1989 / Compte-Wang 2000, capacity-bounded to one
  attractor -- cannot scale to a 10-16-rule pool). crf_frac_active (an INSTANTANEOUS active
  fraction) is the WRONG readiness readout for a sparsely-matched pool -- redefine it to a
  maintained-pool metric (survives whichever side of the firing-vs-silent debate prevails;
  Constantinidis 2018 is the live rebuttal).
  THE FIX (no-op default; bit-identical OFF; behind one master flag):
    (1) MAINTENANCE (prescription 1, Mongillo): under crf_availability_maintenance the
        per-tick SILENCE decay in credit() is REMOVED -- a minted differentiated rule HOLDS
        its availability across context-absent ticks. The eligibility-gated negative-outcome
        credit (the exception/interference path) + retirement are UNTOUCHED, so a
        consistently-bad rule still erodes and retires; only the gaps stop being punished. A
        maintenance_floor (default 0.45, above the mature 2-way-match theta(1)=0.40) is
        applied AT MINT so a fresh differentiated rule starts robustly reactivatable; NOT
        re-floored every tick (exceptions can still cross it). maintenance_decay (default 0.0
        = pure hold) is the optional long-horizon leak knob (set the horizon from the
        measured inter-match interval, not the ~1s biological constant).
    (2) READOUT (prescription 2, CONFIRMED "changes the readout" branch): get_state() gains
        crf_n_maintained_reactivatable (minted rules with availability >= the reactivation
        threshold = the single-match gate floor, i.e. would clear theta if their context
        recurred -- independent of whether it is present this tick) + crf_maintained_pairwise_dist
        (differentiation OF the maintained subset) + crf_frac_maintained. The 666-successor
        readiness gate is re-stated on the maintained pool -- crf_maintained_pairwise_dist > 0.1
        AND crf_n_maintained_reactivatable >= 2 -- RETIRING the crf_frac_active >= 0.30 target.
        crf_frac_active is kept as the SECONDARY active-on-match efficiency readout.
    (3) (Optional, default OFF) prescription 3: engaged_sustain adds a short reverberation
        bump to the matched-and-selected rule (fork-A complement). Not the pool fix.
  Config (CandidateRuleFieldConfig + REEConfig.crf_* + from_dims, all no-op default,
  bit-identical OFF): crf_availability_maintenance (False, master), crf_maintenance_floor
  (0.45), crf_maintenance_decay (0.0), crf_engaged_sustain (False), crf_engaged_sustain_rate
  (0.1), crf_maintained_reactivation_threshold (0.0 = derive from the gate floor). Forwarded
  into the field config at the agent.py build site (getattr fallback -> absent flat attr is
  bit-identical). Designed to run WITH crf_mature_pool_dynamics + crf_context_from_e2_world_forward
  (the differentiation source); the three stay independent flags so each is ablatable.
  Backward compatible: crf_availability_maintenance=False by default -> credit() takes the
  legacy decay path, _maybe_mint does not floor init -> bit-identical. The new get_state
  keys are always emitted (cheap, behaviour-neutral). 19/19 CRF contracts (16 prior + C17
  default-OFF bit-identical + readout keys / C18 maintenance HOLDS a differentiated >=2-rule
  reactivatable pool under sparse matching where legacy erodes it out / C19 from_dims + agent
  wiring) + 7/7 preflight PASS; full contract suite 1008 passed, 0 failures (the prior
  control_vector C4 flake did not fire this run). Activation smoke (mint two distinct rules,
  then 3000 context-absent ticks): legacy mature -> both rules RETIRED (n_minted=0,
  n_maintained=0, READY=False -- the 666 churn) vs maintenance -> pool HELD (n_minted=2,
  n_maintained=2, maintained_pairwise_dist=1.49, READY=True); crf_frac_active~0.001 in BOTH
  arms (confirming frac_active is the wrong readout, the maintained-pool metric is the right
  one).
  Phased training: N/A (pure-arithmetic regulator; no learned parameters; the flag changes
  only the credit-loop silence-decay + mint-floor + readout). MECH-094: unchanged -- the
  maintenance logic lives inside the field's already simulation-gated state-advancing methods.
  Evidence-staleness (Step 8.5): NOT triggered -- no-op-default flag; every existing experiment
  uses the default (legacy decay + active-fraction readout), so no dependent claim's measured
  mechanism changed. KEEP all evidence.
  GOVERNANCE: MECH-309/ARC-062/ARC-063 NEITHER validated NOR weakened; stay candidate /
  substrate_ceiling / v3_pending / pending_retest_after_substrate. claims.yaml carries only an
  implementation_note (no flag/confidence change; substrate-only amend).
  Validation experiment: V3-EXQ-666a (claim-free CRF-readiness diagnostic, supersedes
  V3-EXQ-666) -- the 666-successor re-validation enabling crf_mature_pool_dynamics +
  crf_context_from_e2_world_forward + crf_availability_maintenance, gating on the MAINTAINED-POOL
  metric (crf_maintained_pairwise_dist > 0.1 AND crf_n_maintained_reactivatable >= 2 on >=2/3
  seeds) NOT crf_frac_active. Queued via /queue-experiment. PASS unblocks the 654c GAP-B
  behavioural re-run (a SEPARATE later session). substrate_queue.ready stays FALSE until 666a clears.
  Design doc: REE_assembly/docs/architecture/arc_063_candidate_rule_field.md
  (availability-maintenance amend section, 2026-06-11). Autopsy:
  REE_assembly/evidence/planning/failure_autopsy_V3-EXQ-666_2026-06-11.{md,json}. Lit-pull:
  REE_assembly/evidence/literature/targeted_review_arc_063_crf_rule_cell_persistence/SYNTHESIS.md.
  See ARC-063 (parent v1 + cross-episode persistence + mature-pool amends above), MECH-309 /
  ARC-062 (the GAP-B claims this matures the readiness for; unweakened), crf_context_from_e2_world_forward
  / crf_mature_pool_dynamics (the differentiation source this maintenance pairs with), SD-033a
  (rule_state consumer), V3-EXQ-666 (the FAIL this amend addresses), V3-EXQ-666a (validation),
  V3-EXQ-639 (PASS proving the field differentiates when its pool matures), MECH-094 (call-site
  scoping; unchanged).
