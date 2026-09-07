## MECH-294 AMEND: compose path reads within-cycle co-binding coherence (mode-dependent E3 bias) (2026-06-09)
- MECH-294 compose-coherence amend -- IMPLEMENTED 2026-06-09. Module:
  ree_core/latent/multi_content_theta_packet.py (ThetaPacket.currency_coherence +
  MultiContentThetaPacket.compose_e3_bias) + ree_core/agent.py (compose call site) +
  ree_core/utils/config.py (1 new no-op-default flag). Routed by the /queue-experiment
  attempt to queue the MECH-294 behavioural-evidence successor: a code trace found the
  compose path could NOT carry the joint binding into behaviour, so the behavioural
  mode-discrimination experiment was premature (it would self-route a FALSE "joint clause
  not isolated" weakening on an unmet wiring precondition).
  ROOT CAUSE (code-confirmed): compose_e3_bias computed bias = -bias_scale *
  cosine(candidate_first_action, last_packet.action_proposal). seal() sets
  action_proposal = self._win_action IDENTICALLY across joint/alternation/shuffled
  (the binding mode only changes the goal/risk/state slots), and compose_e3_bias read
  ONLY action_proposal -- so with theta_packet_compose_into_e3_bias=True the three modes
  produced BEHAVIOURALLY IDENTICAL action streams (differing only from packet-OFF).
  This is exactly the design-doc S6 C1-FAIL "wiring" case ("the downstream consumer is
  not conditioning on co-binding") and contradicts S5's intent that joint_context be the
  compose input. The 2026-04-26 governance hold demanded a substrate-side test that
  discriminates joint-packet from cross-cycle alternation; an action-only compose makes
  co-binding inert by construction, so that test was not yet buildable.
  THE FIX (parameter-free; no trained head; no phased training; bit-identical OFF):
  the per-candidate action-grounding bias is now GATED by the sealed packet's
  ThetaPacket.currency_coherence() in [0, 1] -- the fraction of the four V_s-gated
  content streams (goal, risk_sensory, risk_affective, state) whose vintage is CURRENT
  this cycle, i.e. the direct operationalisation of "the streams are bound
  co-temporally". bias = clamp(-bias_scale * coherence * cosine(cand_fa, action_proposal),
  +/-bias_scale). So the SAME proposer action yields a STRONG grounding bias under joint
  (coherence ~1.0), a WEAK one under alternation (~0.25, three streams held -- Kay-2020),
  and NONE under shuffled (0.0, every slot drawn from a different cycle). The binding
  regime now reaches E3 behaviour, which the behavioural mode-discrimination experiment
  requires. The per-candidate RANKING stays an in-space action cosine (no
  cross-semantic-space comparison -- avoids the V3-EXQ-657a coherence-metric autopsy
  pitfall); only the GATE is mode-derived.
  Config: REEConfig.theta_packet_compose_use_joint_coherence (bool, default True) +
  from_dims passthrough. True = the spec-faithful mode-dependent gating; False = recovers
  the legacy action-only cosine (gate==1.0) bit-for-bit (the validation ablation arm).
  NO-OP DEFAULT: theta_packet_compose_into_e3_bias still defaults False, so the entire
  compose block never runs for any existing experiment regardless of this flag --
  bit-identical OFF. No compose-ON experiment exists yet (V3-EXQ-657/657a were
  compose-OFF read-only-first), so nothing depended on the old formula.
  Diagnostics (get_diagnostics, for the validation manifest): mech294_last_currency_coherence,
  mech294_n_compose_calls, mech294_last_compose_coherence, mech294_last_compose_bias_absmax.
  Backward compatible: 8/8 multi_content_theta_packet contracts + 7/7 preflight PASS;
  py_compile OK; V3-EXQ-657a --dry-run unchanged (compose-OFF path). Activation smoke
  2026-06-09 (compose ON, forced-benefit regime, 3 ep x 30 steps): JOINT coherence 1.000
  (bias absmax 0.100) / ALTERNATION 0.250 (0.025) / SHUFFLED 0.000 (0.000); JOINT action
  histogram {3:59, 2:31} differs from ALT/SHUF {3:90} (binding mode changed selection);
  coh=OFF recovers gate==1.0; compose-OFF n_compose_calls=0 (bit-identical).
  Phased training: N/A (pure-arithmetic gate; no learned parameters). MECH-094: compose
  is read-only on the waking select_action path (no replay/memory write surface);
  preserved. Evidence-staleness: NOT triggered -- no-op-default flag; every existing
  experiment uses the default (compose OFF), so no dependent claim's measured mechanism
  changed. KEEP all evidence.
  HONEST LIMITATION: the per-candidate term is action-alignment GATED by coherence -- it
  is conjunctive via the gate (co-binding present), not via a learned same-cycle
  goal x risk x action readout. A richer learned joint-context reader is a future
  phased-training upgrade. The validation EXQ must guard candidate first-action diversity
  (non-vacuity: if the proposer emits a single action class, the gate has nothing to rank
  and joint==alternation behaviourally).
  GOVERNANCE: MECH-294 NEITHER promoted NOR weakened; stays candidate / v3_pending /
  implementation_phase=v3; the 2026-04-26 hold stands until the behavioural successor
  PASSes. claims.yaml carries only an implementation_note (no flag/confidence change).
  Validation experiment: V3-EXQ substrate-readiness diagnostic (claim_ids=[]; compose-ON
  mode-discrimination wiring check: does binding mode now change the committed-class
  distribution, with coh-ON vs coh-OFF ablation isolating the gating). PASS gates the
  MECH-294 behavioural-evidence successor (claim_ids=["MECH-294"]) queued as a separate
  /queue-experiment session.
  Design doc: REE_assembly/docs/architecture/mech_294_multi_content_theta_packet.md
  (compose-coherence amend section).
  See MECH-294 (parent; substrate landed 2026-06-09), MECH-089 (ThetaBuffer; composed),
  MECH-269 / MECH-269b (per-stream V_s vintaging the coherence reads), ARC-018
  (hippocampal CEM proposer; action_proposal source), Kay et al. 2020 (cross-cycle
  alternation control), V3-EXQ-657a (read-only-first readiness PASS this builds on),
  MECH-094 (call-site scoping; preserved).
