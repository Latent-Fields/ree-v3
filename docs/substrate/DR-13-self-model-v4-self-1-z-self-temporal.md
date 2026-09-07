## DR-13 (self_model_v4:SELF-1): z_self temporal depth -- dedicated self-recurrence anchored by E1 feedback (2026-07-01)
- DR-13: latent_stack.self_recurrence -- IMPLEMENTED 2026-07-01.
  The SUBSTRATE FLOOR of self_model_v4 (generation:v4, off the V3 critical path; PROMOTES
  NOTHING in V3). User-directed build to unblock SELF-3 (DR-10) after the 2026-07-01 IGW-165
  reconcile marked SELF-3 blocked on exactly this substrate. Mechanism resolved on ARC-081
  notes 2026-06-14 (HYBRID -- both motifs committed).
  Modules: ree_core/latent/self_recurrence.py (NEW: SelfRecurrenceCell, a GRUCell over
  z_self), ree_core/latent/stack.py (LatentStack.__init__ conditional instantiation +
  encode() self_e1_anchor param + the recurrence-replaces-EMA lever + LatentState
  .self_recurrence_diag readout), ree_core/utils/config.py (LatentStackConfig 2 fields +
  REEConfig.from_dims passthrough), ree_core/agent.py (_e1_predicted_next_z_self cache in
  _e1_tick + sense() anchor passthrough + reset).
  Problem (v4_spec V4-2 DR-13): z_self = body_obs -> MLP -> fixed-alpha EMA is an
  instantaneous body snapshot; a fixed-decay EMA cannot selectively retain/gate self-state,
  so there is no stable, inspectable, lesionable SUBJECT for DR-10 (z_self-in-E3), DR-11
  (self-state goals) and the INV-064 maturational-stability gate to attach to.
  THE LEVER (no-op default; bit-identical OFF): when use_self_recurrence, encode() REPLACES
  the z_self EMA step ONLY (z_world/z_beta/z_theta/z_delta smoothing untouched) with
    h = SelfRecurrenceCell(z_self_instant, prev.z_self)            # gated recurrence, hidden = prev stateful z_self
    z_self = (1 - c) * h + c * self_e1_anchor  if anchor & c>0     # E1-feedback blend
           = h                                 otherwise           # pure recurrence
  c = self_recurrence_e1_coupling (the recorded residual tunable): 0 = pure recurrence
  (Option A, max stability-isolation) | 1 = pure E1-feedback (Option B) | 0.15 light default
  = HYBRID. The cell is perturbation-ISOLATED (only z_self flows through it -- a +5.0
  perturbation of prev.z_self leaks 0.0 into z_world; contract C5). Anchor SOURCE (v1 scope,
  user-confirmed AskUserQuestion 2026-07-01 "Proceed as planned"): the E1 PREDICTED-NEXT
  z_self, cached at _e1_tick (predictions[:,0,:] -> split_prediction[0], detached) and
  consumed on the next encode() -- side-effect-free (no extra E1 forward, no LSTM
  hidden-state mutation), the volatility_signal plumbing precedent. First tick has no cache
  -> anchor None -> pure recurrence that step. DOCUMENTED FOLLOW-ON (NOT v1): an ecological
  anchor computed inline at encode without the one-tick lag.
  Diagnostics on LatentState.self_recurrence_diag (None when OFF): active, state_departure
  (||stateful z_self - instantaneous z_self||, batch-mean -- the DR-13 non-vacuity readout),
  e1_coupling, anchor_present.
  Config (LatentStackConfig + REEConfig.from_dims, all no-op default): use_self_recurrence
  (False, master), self_recurrence_e1_coupling (0.15). Default OFF -> self_recurrence NOT
  instantiated (mirrors reafference_predictor) + verbatim legacy EMA -> bit-identical.
  Backward compatible: disabled by default; existing experiments unaffected. Full suite
  1336 passed / 4 pre-existing failures (control_vector C4 flake + 2 runner-fail-branch +
  sd016 E1-proj-dim -- all confirmed identical on the clean tree via git stash) + 11 new
  contracts in tests/contracts/test_dr13_self_recurrence.py (C1 OFF bit-identical incl.
  anchor-ignored + deterministic / C2 ON module+diag / C3 state_departure>0 / C4 anchor
  blend exact + shape-mismatch fallback / C5 self perturbation does not leak to z_world /
  C6 coupling=0 ignores anchor / agent-level: OFF never caches anchor, ON caches + runs
  end-to-end).
  Phased training: the GRUCell trains via the EXISTING E1/E2 z_self prediction losses (v1
  adds NO new loss; the anchor is an inference-time detached blend). Standard joint-collapse
  awareness applies; experiments training the recurrence should follow P0 warmup -> P1
  frozen-encoder phasing.
  MECH-094: N/A -- waking perception path (encode/sense/_e1_tick); no replay/memory write
  surface.
  Evidence-staleness (Step 8.5): NOT triggered -- no-op-default lever; every existing
  experiment uses the default (OFF), so no dependent claim's measured mechanism changed.
  KEEP all evidence.
  GOVERNANCE: PROMOTES NOTHING. ARC-081 (mechanism resolved on its notes) + MECH-215 stay
  candidate / implementation_phase=v4 -- DR-13 is the substrate half; DR-10 (SELF-3) +
  experiments remain before MECH-215 unblocks. claims.yaml gets an implementation_note only
  (no status/v3_pending change).
  Validation experiment: V4-EXQ-002 (DR-13 self-recurrence substrate-readiness falsifier)
  queued 2026-07-01 (ree-v3 main 4c24214; coordinator /queue/add applied + /queue/active
  PRESENT; smoke PASS hist/iso/anchor 1/1) -- ON vs OFF; FALSIFIER = if the stateful z_self
  does not depart from the instantaneous encode / carries no history / lesioning the
  recurrent hidden state does not change it, DR-13 buys nothing (pre-registered non-vacuity
  gate on state_departure + inert off-ramp). unblocks_claims=ARC-081/MECH-215.
  Design doc: REE_assembly/docs/architecture/dr13_self_recurrence_temporal_depth.md.
  Plan node: REE_assembly/evidence/planning/self_model_v4_plan.md (self_model_v4:SELF-1).
  See ARC-081 (self-as-object pillar, mechanism source), MECH-215 (unblocked half), SD-005
  (the split z_self this upgrades), DR-10/SELF-3 (was blocked on this build), DR-12/SELF-4
  (E2-PE->E3, already built, E-stream-native), INV-064/SELF-7 (maturational-stability gate).
