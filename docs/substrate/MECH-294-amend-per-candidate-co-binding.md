## MECH-294 AMEND: per-candidate co-binding coherence (cross-candidate-range rendering so the route-range authority + 569i top-k can carve) (2026-06-19)
- MECH-294 per-candidate-coherence amend -- IMPLEMENTED 2026-06-19 (substrate;
  MECH-294 stays candidate / substrate_ceiling / v3_pending -- this PROMOTES
  NOTHING; the 2026-04-26 governance hold stands, and the behavioural falsifier is
  a SEPARATE later /queue-experiment step). Modules:
  ree_core/latent/multi_content_theta_packet.py (ThetaPacket.action_refs /
  coherence_weights + MultiContentThetaPacket.compose_per_candidate_coherence +
  seal() per-stream action-reference binding + _action_snapshots + 3 diagnostics),
  ree_core/agent.py (compose-block branch + packet-config coherence_hold_weight
  passthrough), ree_core/utils/config.py (2 no-op-default flags + from_dims).
  Routed by the substrate-ceiling-lifted triage 2026-06-19
  (REE_assembly/evidence/planning/substrate_ceiling_lifted_triage_2026-06.md,
  verdict (b) MECH-294 STILL CEILINGED): the named substrate
  (modulatory-bias-selection-authority) is implemented but cannot carve MECH-294.
  ROOT CAUSE (V3-EXQ-661 + the route-range amend's "coherence" source, code +
  manifest confirmed): currency_coherence() is a SCALAR. compose_e3_bias produced
  bias = -scale * coherence * align where align[K] = cosine(candidate_first_action,
  the single bound action_proposal) -- a per-candidate PATTERN identical across
  binding modes (seal sets action_proposal = _win_action regardless of mode);
  coherence only scales magnitude uniformly. The route-range authority normalises
  the routed bias to unit range, so the scalar's only effect (uniform magnitude
  scaling) is ERASED -> JOINT == ALTERNATION; SHUFFLED (coherence 0) floors to 0.
  661 manifest: committed-action histograms BYTE-IDENTICAL across ARM_OFF / JOINT /
  ALT / SHUF / ALT_COH_OFF per seed (committed-dist TV ~0, max 0.0097, incl
  C3 gate-ON-vs-OFF). So the authority + 569i top-k had nothing mode-distinct to
  carve. The route-range readiness gate (V3-EXQ-663) correctly flags coherence as a
  no-range channel.
  THE FIX (no-op-default; bit-identical OFF): a PER-CANDIDATE co-binding coherence
  whose cross-candidate RANGE is mode-distinct. seal() now binds, per V_s-gated
  content stream, the action_proposal co-bound WITH that stream this cycle
  (action_refs) + a currency weight (coherence_weights), via a per-stream action
  snapshot mirror of the existing _snapshots:
    JOINT       -- all four streams current -> all refs = this-cycle action (w 1.0)
                   -> compose_per_candidate_coherence == cosine(cand, a) -> FULL
                   cross-candidate range.
    ALTERNATION -- live stream this-cycle action (w 1.0) + held streams their PRIOR
                   co-bound action at coherence_hold_weight (default 0.5) -> a
                   per-candidate PATTERN distinct from JOINT (smoke: joint-vs-alt
                   bias cosine 0.82, NOT a uniform scaling -> survives
                   range-normalisation), not merely a smaller magnitude.
    SHUFFLED    -- nothing co-bound this cycle (weights 0) -> None / zeros -> the
                   authority reads below-floor (matches currency_coherence 0.0).
  compose_per_candidate_coherence returns clamp(-bias_scale * weighted_mean_s[
  w_s * cosine(cand, ref_s)], +/-bias_scale) -- in-action-space throughout (no
  cross-semantic-space comparison; cf. the V3-EXQ-657a coherence-metric autopsy).
  The agent compose block branches on theta_packet_compose_per_candidate_coherence
  and (when ON) sets _bdc_coherence to the new per-candidate bias, so the EXISTING
  route source "coherence" (modulatory_channel_route_source) now routes a
  mode-distinct per-candidate RANGE into the modulatory accumulator the route-range
  authority rescales + the 569i top-k shortlists. currency_coherence() is NOT
  consulted or modified -- the scalar mode-discrimination (JOINT 1.0 / ALT 0.25 /
  SHUF 0.0) is preserved bit-identically.
  Config (REEConfig + from_dims + MultiContentThetaPacketConfig, all no-op default;
  bit-identical OFF): theta_packet_compose_per_candidate_coherence (False; only
  consulted when theta_packet_compose_into_e3_bias is also True -- OFF recovers the
  legacy scalar-gated action-only compose bit-for-bit) +
  theta_packet_coherence_hold_weight (0.5; held-stream prior-ref weight, 0.0 =
  pure currency-gating).
  Backward compatible: theta_packet_compose_per_candidate_coherence=False by
  default AND the packet master switch use_multi_content_theta_packet defaults
  False, so the seal-time action-ref bookkeeping runs only for packet-ON
  experiments and never changes the bound content slots / currency_coherence ->
  bit-identical. 8/8 prior multi_content_theta_packet contracts + 7/7 preflight +
  the full contract suite PASS; 7 new contracts in
  tests/contracts/test_mech294_per_candidate_coherence.py (C1 JOINT range>floor /
  C2 SHUFFLED ~0 / C3 scalar currency_coherence unregressed / C4 ALT pattern
  distinct from JOINT [the chosen design fork] / C5 monostrategy -> ~0 range
  non-vacuity / C6 default-OFF flag + legacy compose unchanged / C7 agent compose
  fires + routes to the authority via "coherence"). Smoke 2026-06-19: JOINT range
  0.1, SHUF None/0, ALT range 0.02 + joint-vs-alt pattern cosine 0.82.
  Phased training: N/A (pure-arithmetic per-stream action-cosine + weighting; no
  learned parameters). MECH-094: compose is read-only on the waking select_action
  path; the seal-time action-snapshot bookkeeping rides the existing
  simulation_mode-gated seal (no replay/memory write surface). Evidence-staleness
  (Step 8.5): NOT triggered -- no-op-default flag; every existing experiment uses
  the default (per-candidate coherence off), so no dependent claim's measured
  mechanism changed. KEEP all evidence.
  GOVERNANCE: PROMOTES NOTHING. MECH-294 stays candidate / substrate_ceiling /
  v3_pending; claims.yaml carries only an implementation_note (no flag/confidence/
  status change). The primary lit falsifier (Kay-2020 cross-cycle theta) is
  out-of-substrate for V3, so the V3 path is this per-candidate-range wiring + a
  downstream behavioural readout.
  Validation: the substrate-readiness gate is the contract suite (the carve-able
  channel now EXISTS: JOINT range>floor on a diverse candidate pool, SHUF ~0,
  scalar mode-discrimination preserved). NO behavioural EXQ queued here -- the
  MECH-294 behavioural falsifier on this channel is a SEPARATE later
  /queue-experiment step (per the task scope + the 661 implement-substrate log
  that called this MECH-294-side compose path out of scope).
  Design doc: REE_assembly/docs/architecture/mech_294_multi_content_theta_packet.md
  (per-candidate-coherence amend section). Triage:
  REE_assembly/evidence/planning/substrate_ceiling_lifted_triage_2026-06.md.
  See MECH-294 (parent; substrate 2026-06-09 + compose-coherence amend 2026-06-09
  above), modulatory-bias-selection-authority (route-range amend 2026-06-10 -- the
  "coherence" source this gives a mode-distinct per-candidate range; gain/contrast
  + margin-shortlist 2026-06-15 + top-k shortlist 2026-06-16 -- the 569i carve),
  V3-EXQ-661 (the scalar-coherence committed-dist TV ~0 FAIL this addresses),
  V3-EXQ-663 (route-range readiness gate that flagged coherence as no-range),
  MECH-089 (ThetaBuffer; state_summary source, untouched), MECH-269/269b
  (per-stream V_s vintaging that drives currency), Kay et al. 2020 (cross-cycle
  alternation control; out-of-substrate lit falsifier), MECH-094 (call-site
  scoping; preserved).
