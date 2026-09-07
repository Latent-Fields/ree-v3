## MECH-294: multi-content theta-burst packet (joint {goal,action,risk,state} per-cycle binding) (2026-06-09)
- MECH-294: theta_burst.multi_content_joint_packet -- IMPLEMENTED 2026-06-09
  (substrate; v3_pending until the discriminative validation EXQ PASSes). Closes
  the /queue-experiment substrate-readiness gate that found MECH-294
  blocked_substrate on 2026-06-09: V3 implemented only single-content temporal
  averaging (MECH-089 ThetaBuffer buffers z_world/z_self and returns a
  theta-cycle-AVERAGED z_world), so any MECH-294 joint-binding experiment was
  vacuous. Per design memo REE_assembly/docs/architecture/mech_294_multi_content_theta_packet.md.
  SIBLING module, NOT a ThetaBuffer rewrite: MECH-089 is depended on by MECH-092
  replay / MECH-122 consolidation / SD-006 cross-rate E3 read, so the packet
  COMPOSES the existing ThetaBuffer for its state_summary slot and gathers the
  other three streams itself. MECH-089 untouched; MECH-294 is a strict additive
  layer.
  Module: ree_core/latent/multi_content_theta_packet.py (MultiContentThetaPacket
  + MultiContentThetaPacketConfig + ThetaPacket + ThetaPacketVintage). Pure-
  arithmetic (no nn.Module, no learned parameters, no gradient flow); sibling
  pattern to MECH-313 / MECH-320 / MECH-342. Every observed tensor is detached
  (a packet is a read-only snapshot, never a grad path).
  Four streams + concrete REE-v3 sources (memo S3): goal_latent <- GoalState.z_goal
  (when active); risk_sensory <- LatentState.z_harm (SD-010); risk_affective <-
  LatentState.z_harm_a (SD-011, kept as a DISTINCT sub-slot -- not pre-collapsed,
  preserves the SD-011 dissociation into the packet); state_summary <- the
  MECH-089 averaged z_world (theta_buffer.summary()); action_proposal <- the
  hippocampal CEM proposer lead first-action (candidates[0].actions[:,0,:]).
  Per-cycle phase-aligned binding window (memo S4.1): a theta cycle is the
  interval between two E3 heartbeat ticks. The packet opens a window, accumulates
  per-stream values during E1 ticks (observe), takes the proposer first-action at
  the E3 tick (observe_action_proposal), and SEALS one immutable ThetaPacket at
  the E3-heartbeat boundary (seal), exposed as agent.last_theta_packet for the
  proposer/E3 to read as a joint object on the next cycle. Gamma-sub-slot routing:
  type-separated named sub-slots (NOT a flat concat) so the joint-read interface
  can condition action on goal-and-risk.
  Per-stream V_s vintaging (MECH-269 / MECH-269b REUSE, memo S4.3): the packet
  applies the SAME snapshot-or-hold discipline -- refresh the per-stream snapshot
  when V_s >= snapshot_refresh_threshold (0.5), substitute the held last-verified
  snapshot when V_s < hold_threshold (0.4), 0.4-0.5 dead-band = MECH-269b Schmitt
  hysteresis. Each component records ThetaPacketVintage{is_current, age_ticks,
  v_s}; action_proposal has no V_s (its vintage is age in E3 ticks). This makes
  the packet a stream-typed object whose components may carry DIFFERENT temporal
  vintages (the MECH-294 secondary property) rather than a homogeneous current
  latent.
  Three binding regimes (the discriminative core that resolves the Kay-2020
  cross-cycle-alternation falsifier, memo S6) selected by theta_packet_binding_mode:
    "joint"       -- all four streams' current (or V_s-held) values bound
                     SIMULTANEOUSLY within one cycle (the MECH-294 hypothesis).
    "alternation" -- exactly ONE stream live per cycle (round-robin), the other
                     three HELD at their prior snapshots; only the live stream
                     refreshes (Kay-2020 one-stream-per-cycle parsimonious control).
    "shuffled"    -- each slot filled from a DIFFERENT prior cycle's value for
                     that stream (matched marginals, never co-observed;
                     independent-content control).
  All three produce structurally distinct sealed packets from the SAME input
  stream (verified C4 + smoke: last-packet structural dist joint-vs-alt 17.9,
  joint-vs-shuf 44.6) -- the substrate-side discriminability the validation
  depends on.
  Joint-read interface (memo S5): joint_context() (type-tagged concat for a
  single-context consumer), action_conditioned_on(goal, risk) (the literal "which
  action is on the table read against the same-cycle goal + risk" operation),
  risk_vector() (concat over the two SD-011 sub-slots), is_component_stale(name).
  Config (REEConfig + from_dims, all no-op default, bit-identical OFF):
    use_multi_content_theta_packet (False, master). Requires use_per_stream_vs=True
      (the packet consumes the MECH-269 per-stream V_s) -- LOUD ValueError at
      agent __init__ otherwise (same precondition pattern as MECH-269b / MECH-293).
    theta_packet_binding_mode ("joint" / "alternation" / "shuffled").
    theta_packet_snapshot_refresh_threshold (0.5), theta_packet_hold_threshold
      (0.4) -- MECH-269b reuse.
    theta_packet_compose_into_e3_bias (False) -- READ-ONLY-FIRST (memo S5): when
      False the packet is built/sealed/exposed/logged but does NOT touch E3
      selection (the validation measures it read-only). When True, joint_context
      biases E3 via a PARAMETER-FREE clamped arithmetic bias (cosine of candidate
      first-action vs the packet's co-bound action_proposal, clamped to
      +/-bias_scale) -- NO trained head, so NO phased training. Composed LAST in
      the dacc_score_bias chain so it never dominates.
    theta_packet_bias_scale (0.1).
  Agent wiring (ree_core/agent.py): self.multi_content_theta_packet built in
  __init__ when the master flag is on (else None) + self.last_theta_packet; reset()
  clears the window/snapshots/history; _e1_tick observe(...) right after
  theta_buffer.update; _e3_tick observe_action_proposal + seal -> last_theta_packet
  at the theta_buffer.summary() boundary; optional compose hook before the MECH-313
  noise-floor block. All call sites are waking (simulation_mode=False).
  MECH-094: every observe/seal method takes simulation_mode and is a no-op when
  True (a replay/DMN tick must not seal a waking packet). Call sites are waking-only
  (call-site scoping, same as MECH-269 / MECH-288 / MECH-287).
  Phased training: NONE required -- the packet is gather/route/seal arithmetic +
  dataclass plumbing over already-trained latents; no new encoder head, no gradient
  flow. (A future learned type-embedding or trained joint-read head would need
  P0/P1/P2; out of scope here.)
  Evidence-staleness (Step 8.5): NOT triggered -- no-op-default flag; every existing
  experiment uses the default (packet off), so no dependent claim's measured
  mechanism changed. KEEP all evidence.
  Backward compatible: use_multi_content_theta_packet=False by default ->
  agent.multi_content_theta_packet is None; both push sites + the seal are skipped;
  MECH-089 ThetaBuffer byte-identical. 7/7 preflight + 968 contracts PASS (8 new in
  tests/contracts/test_multi_content_theta_packet.py: C1 default-OFF no-op +
  bit-identical / C2 ON 4-slot sealed packet at the E3 boundary / C3 V_s-held
  substitution + stale vintage age / C4 joint vs alternation vs shuffled
  structurally distinct / C5 MECH-094 simulation no-op / C6 action_conditioned_on
  joint read / C7 precondition raises without use_per_stream_vs). Bit-identical OFF
  verified (default vs explicit-False action stream, seed-matched). Activation smoke
  2026-06-09 (full sense path, harm streams on): packet seals at the E3 boundary
  with risk_sensory + risk_affective + state_summary + action_proposal populated.
  GOVERNANCE: MECH-294 NEITHER promoted NOR weakened; stays candidate / v3_pending /
  implementation_phase=v3; the 2026-04-26 governance hold (needs a substrate-side
  joint-vs-alternation falsification test) stands until the C1/C2 discriminative
  validation PASSes. claims.yaml carries only an implementation_note (no flag /
  confidence change).
  Validation experiment: V3-EXQ-657 substrate-readiness diagnostic (claim_ids=[];
  4-arm matched-seed/matched-content ARM_0 OFF / ARM_1 joint / ARM_2 alternation /
  ARM_3 shuffled; G0 packet completeness + G1 vintage heterogeneity +
  NON-VACUITY precondition (shuffled must structurally differ from joint, else
  substrate_not_ready_requeue) + C1 joint!=alternation + C2 joint!=shuffled). Non-
  vacuously tests joint-within-cycle vs Kay-2020 cross-cycle alternation. PASS (per
  the memo S7.3 interpretation grid) clears the substrate-readiness gate that found
  MECH-294 blocked_substrate; only then does the MECH-294 behavioural-evidence
  successor get queued.
  Design doc: REE_assembly/docs/architecture/mech_294_multi_content_theta_packet.md
  (Status flipped design-memo -> IMPLEMENTED 2026-06-09).
  See MECH-294 (this claim), MECH-089 (parent ThetaBuffer; composed, untouched),
  MECH-269 / MECH-269b (per-stream V_s vintaging reuse), SD-010 (z_harm_s) / SD-011
  (z_harm_a) (risk sub-slots), SD-012 / SD-015 / MECH-230 (GoalState.z_goal source),
  ARC-018 (hippocampal CEM proposer; action_proposal source), SD-005 (z_world/z_self
  split; dep), MECH-090 (E3-heartbeat commit machinery; cycle boundary), Kay et al.
  2020 (cross-cycle alternation falsifier the S7 experiment settles), V3-EXQ-657
  (validation), MECH-094 (simulation-mode call-site scoping).
