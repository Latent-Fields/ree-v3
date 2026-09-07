## ARC-006 / MECH-045: token-instance object-file / entity-persistence buffer (2026-06-09)
- ARC-006 / MECH-045: entities.object_file_buffer -- IMPLEMENTED 2026-06-09 (v1
  substrate; v3_pending until V3-EXQ-658 clears at full scale). The TOKEN store of
  the ARC-080 type/token/anchor triad -- the missing third per-item store (TYPE =
  SD-057 IncentiveTokenBank goal.py; ANCHOR = SD-039/MECH-292 ghost-goal bank
  hippocampal/). A persistence-vs-ablation probe against the two LIVE stores would be
  vacuous (neither re-identifies a moved entity), which is why proposal EVB-0293 was
  marked blocked_substrate by a 2026-06-09 /queue-experiment Step-2.5 gate. Design
  memo (the spec): REE_assembly/docs/architecture/mech_045_object_file_buffer.md.
  Module: ree_core/entities/object_file_buffer.py (ObjectFileBuffer + ObjectFile +
  EntityObservation + ObjectFileBufferConfig). NON-TRAINABLE stateful regulator (no
  nn.Module, no params, no gradient flow); sibling to IncentiveTokenBank /
  BlockedAgency / EscapeAffordanceBridge. DeepSORT-style data association over
  z_world-local features, one update() per waking tick (memo Section 4.2):
    C1 token KEY    -- label-free per-entity token id by spatiotemporal continuity:
                       hard MOTION gate (obf_continuity_radius) + precision-weighted
                       appearance cost (w_motion*d_pos + w_feat*d_cos); the token
                       SURVIVES the entity moving to a new cell (NOT nearest-cell).
    C2 feature buf  -- per-token precision-weighted z_features EMA (the "file").
    C3 persistence  -- a token survives <= obf_persist_ttl unseen ticks, then dies.
    C4 attention    -- salience-gated births; tokens compete for a bounded buffer
                       (obf_max_tokens, the FINST ~4-5 capacity analogue).
    C5 precision    -- association + feature update precision-weighted (zero-precision
                       obs makes no feature move).
  type_hint records the SD-049 resource_tag as an OPTIONAL wiring hook -- NEVER the
  association key (the key is continuity, not type).
  Config (REEConfig + from_dims, all no-op default; bit-identical OFF):
    use_object_file_buffer (False, master) + obf_max_tokens (5) + obf_continuity_radius
    (2.0) + obf_w_motion (1.0) + obf_w_feat (1.0) + obf_feature_alpha (0.3) +
    obf_persist_ttl (8) + obf_min_birth_salience (0.0) + obf_use_precision_weighting (True).
  Agent wiring (ree_core/agent.py): self.object_file_buffer built when the master flag
  is on; driven on the waking stream via REEAgent.update_object_file_buffer(observations,
  simulation_mode=...) (the caller / experiment supplies the perceived entities -- the v1
  detector dependency, memo Section 4.4: SD-049 per-type resource-field views + grid
  object/hazard cells); reset() clears it per episode.
  v1 lands STANDALONE: NO action-stream consumer -> the action stream is BIT-IDENTICAL
  whether the buffer is on or off (only buffer state changes; nothing reads it yet).
  SD-057 (TYPE->token type_hint) + SD-039 (ANCHOR->token_id cross-ref) wiring is
  explicitly DEFERRED (memo Section 3). OFF the GAP-7 V3-closure path.
  MECH-094: update() is a no-op under simulation_mode (the buffer updates ONLY on the
  waking stream -- a moved-imagined-object must not rewrite the waking object-file);
  agent passes hypothesis_tag through.
  Phased training: N/A (non-trainable; the buffer consumes already-encoded z_world and
  produces token assignments; a learned association metric is the v2 upgrade path).
  Backward compatible: use_object_file_buffer=False by default -> agent.object_file_buffer
  is None and update_object_file_buffer returns {}; bit-identical. 6/6 new contracts
  (tests/contracts/test_mech_045_object_file_buffer.py: C1 default-off + bit-identical
  action stream / C2 cross-motion re-identification with same-type distractor / C3
  persistence-through-absence + eviction / C4 attention capacity cap / C5 precision
  weighting / C6 MECH-094 sim no-op) + 7/7 preflight + 973 contracts PASS (the 1
  control_vector C4 fail is the documented pre-existing baseline flake, unrelated).
  Activation smoke 2026-06-09: at world_dim=128 with two same-type instances (distinct
  appearances) + motion, the moved target re-identifies to the SAME token (reid 1.0)
  while ABLATION-OFF (no token) and ABLATION-SHUFFLE (ids permuted) do not; bit-identical
  action stream OFF==ON verified.
  Validation experiment: V3-EXQ-658 (experiments/v3_exq_658_mech045_object_file_persistence.py)
  -- the EVB-0293/EXP-0117 re-queue: 3 arms INTACT/ABLATION-OFF/ABLATION-SHUFFLE x 3 seeds
  at world_dim=128 with several same-type distractors; readiness gates G0 (token
  maintained) / G1 (feature non-degenerate d_cos>=0.1, the dim=32 guard) / G2 (entity
  moved) before C1 reid>=0.6 (PRIMARY) / C2 INTACT>SHUFFLE+0.3 AND >OFF / C3
  feature-persistence; self-routes substrate_not_ready_requeue if readiness unmet.
  claim_ids=[MECH-045] (ARC-006 bears-on, not co-tagged). substrate_queue.ready stays
  FALSE until it clears at full scale. MECH-045/ARC-006 NOT promoted/weakened (governance
  applies after the run).
  Design memo: REE_assembly/docs/architecture/mech_045_object_file_buffer.md.
  Substrate_queue: REE_assembly/evidence/planning/substrate_queue.json (mech-045-object-file-buffer).
  See ARC-006 / MECH-045 (this claim), ARC-080 (type/token/anchor umbrella + fork),
  SD-057 IncentiveTokenBank (TYPE store; proposed wiring target), SD-039 / MECH-292
  ghost-goal bank (ANCHOR store; proposed wiring target), SD-049 (per-type tags +
  proximity views -- the v1 detector source), MECH-278 (object schema; V4, bypassed),
  E2WorldForward (world_dim>=128 floor the experiment runs at), V3-EXQ-658 (validation),
  MECH-094 (waking-stream-only call-site scoping).
