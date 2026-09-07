## SD-049 Phase 2: Hybrid Identity-Aware z_resource Encoder (2026-05-04)
- SD-049 Phase 2: encoder.identity_aware_z_resource (Option C hybrid) --
  IMPLEMENTED 2026-05-04. Encoder-side Phase 2 follow-on to the SD-049
  Phase 1 substrate (env-only) landed 2026-05-03. Lands the architectural
  choice from the 2026-05-04 lit-pull verdict
  (evidence/literature/targeted_review_sd_049_encoder_identity_expansion/
  verdict.md, Option C hybrid at confidence 0.78). Biology-anchored to
  Ballesta-Padoa-Schioppa 2019 OFC labeled-line + Quiroga 2005 sparse
  readouts + Schapiro 2017 hybrid CLS bi-pathway architecture.
  Modules:
    ree_core/latent/stack.py (ResourceEncoder + LatentState)
    ree_core/utils/config.py (LatentStackConfig)
    ree_core/agent.py (compute_resource_identity_loss)
    ree_core/environment/causal_grid_world.py (info dict supervision target)
  Encoder shape (Option C hybrid):
    Shared trunk MLP encoder (Linear -> ReLU -> Linear) producing 32-dim
    z_resource (Schapiro 2017 monosynaptic-analog distributed substrate;
    Schapiro 2016 statistical-learning pathway).
    Identity-classifier head (Linear(z_resource_dim, n_resource_types))
    supervised by cross-entropy on obs_dict["sd049_consumed_type_tag_this_tick"]
    when SD-049 multi_resource_heterogeneity is on (Ballesta-Padoa-Schioppa
    labeled lines; Quiroga sparse readouts; Schapiro 2017 trisynaptic-
    analog episode pattern separation).
    Magnitude head (resource_prox_head) reusing existing SD-018 pattern
    unchanged.
  IMPORTANT design choice: z_resource OUTPUT shape unchanged at
    z_resource_dim (32). Classifier head shapes the trunk via training
    pressure (anti-collapse mitigation per Levi 2021 + identity
    discriminability supervision per the verdict). identity_logits is
    exposed as a SEPARATE LatentState field (Optional[Tensor]) for the
    cross-entropy loss; downstream consumers (GoalState seeding etc.)
    continue to read z_resource unchanged. This avoids the
    GoalState.z_goal seeding dim-mismatch that pure-concat would create
    (z_goal_dim=32; concat would grow z_resource to 32 + n_types).
  Config (LatentStackConfig):
    use_identity_classifier (bool, default False) -- master switch.
    identity_classifier_n_types (int, default 3) -- output dim of head.
    Both surfaced via direct attribute assignment on cfg.latent.* (not
    via REEConfig.from_dims kwargs -- matches existing SD-015
    use_resource_encoder pattern).
  Phased training (per verdict.md):
    P0: enable use_identity_classifier=True. Joint backprop of identity
        cross-entropy + resource_prox MSE + downstream task losses
        through the trunk. Classifier head provides anti-collapse pull
        (Levi 2021 mitigation in spirit) AND identity-discriminability
        supervision per the biology-anchored verdict.
    P1: freeze identity_head.requires_grad_(False). Continue trunk
        training under E1/E3/downstream losses. Trunk embedding develops
        similarity structure beyond what classifier supervision alone
        provides (Schapiro 2016 distributed substrate development).
    P2: evaluate identity-recovery (linear probe on z_resource) AND
        goal_resource_r AND per-axis drive evolution per V3-EXQ-514
        acceptance criteria.
  Data flow:
    ResourceEncoder.forward(world_obs) -> (z_resource [batch, 32],
      resource_prox_pred_r [batch, 1], identity_logits [batch, n_types]
      OR None when classifier disabled).
    LatentStack.encode() -> LatentState with z_resource +
      resource_prox_pred_r + identity_logits all populated when classifier
      enabled.
    agent.compute_resource_identity_loss(target_type, latent_state) ->
      cross-entropy scalar; zero when classifier disabled, target=0
      (no resource at agent), or out-of-range.
  Env supervision target plumbing (SD-049 Phase 2 env fix):
    causal_grid_world.py step() caches consumed-type tag BEFORE clearing
    the cell tag in the resource-consumption branch. Surfaced as
    info["sd049_consumed_type_tag_this_tick"] (1..n_types when consumption
    fired this tick; 0 otherwise). The supervision target for the
    identity classifier in the V3-EXQ-514 training loop. Always present
    (0 when SD-049 OFF or no consumption this tick).
  Backward compatible: use_identity_classifier=False by default;
    ResourceEncoder.identity_head=None; identity_logits=None on every
    LatentState; compute_resource_identity_loss returns 0; all existing
    SD-015 experiments (use_resource_encoder=True with classifier OFF)
    behave bit-identically. 7/7 preflight + 184/184 contracts PASS with
    classifier OFF (regression suite green, 2026-05-04).
  Activation smoke (2026-05-04):
    Default ResourceEncoder + use_identity_classifier=False -> identity_logits
    is None on LatentState; compute_resource_identity_loss returns 0.0.
    use_identity_classifier=True + n_types=3 -> identity_logits shape
    [1, 3]; cross-entropy with target=type_0 returns ~ln(3)~1.10 at random
    init; backward succeeds; target=0 (no-resource) returns 0.0 (skip).
    Env consumed-type plumbing: with multi_resource_heterogeneity_enabled=True,
    info["sd049_consumed_type_tag_this_tick"] correctly reports type_idx+1
    (1/2/3) on resource-contact ticks; 0 on non-contact ticks.
  No phased training needed for the SUBSTRATE itself (encoder change is a
  shape change with backward-compat defaults). Phased training is REQUIRED
  for V3-EXQ-514 (it is the validation methodology, not a substrate
  requirement).
  MECH-094: not applicable (waking observation stream encoder; not replay
    / simulation content). The classifier supervision target is the
    waking-stream env consumption tag.
  ML/AI engineering notes (Layer 7, per implement-substrate skill rule):
    - Class-collapse hazard (Levi et al. 2021 ICCV): mitigated by phased
      training. P0 supervised classifier head provides anti-collapse pull
      on the trunk; P1 freezes the head; P2 evaluates. No additional
      anti-collapse machinery needed at the substrate level.
    - Joint training fragility (EXQ-166b/c/d historical): the phased
      protocol decouples head learning from encoder learning across the
      P0/P1 boundary. Standard mitigation pattern.
    - z_resource shape preservation rather than concat: chosen over the
      verdict.md Option-1 concat instantiation because GoalState.z_goal
      is fixed at goal_dim=32 and would silently break under z_resource =
      concat(trunk, identity_softmax) = 32 + n_types. The "single-output-
      with-supervision" instantiation (verdict.md Option 2) is what
      most ML papers do; the classifier head is training-only in the
      sense that downstream consumers read just z_resource, but the
      identity_logits are still computed at every tick (cheap) and
      exposed for the loss term.
  Validation experiment: V3-EXQ-514 queued -- 4-arm sweep + phased
    training + identity-recovery linear probe + goal_resource_r
    measurement + per-axis drive evolution check. 10 acceptance criteria
    (C0/C1a/C1b/C2a/C2b/C2c/C2d/C2e/C3a/C3b). PASS = SD-049 Phase 2
    validated; SD-015 promotable; SD-049 v3_pending may be cleared. FAIL
    routes to the 6-row interpretation grid in verdict.md, including the
    Woo/Spelke-style substrate-ceiling falsifier branch (joint failure
    across ARM_2 AND ARM_3 routes MECH-229 to substrate_conditional with
    V4-1 multi-agent ecology dependency). estimated_minutes=90.
  Design doc: REE_assembly/docs/architecture/sd_049_multi_resource_heterogeneity.md
  Lit-pull: REE_assembly/evidence/literature/targeted_review_sd_049_encoder_identity_expansion/
  See SD-049 (parent claim, Phase 1 substrate landed 2026-05-03), SD-015
    (z_resource encoder upstream substrate this extends), MECH-229
    (wanting/liking dissociation primary test), MECH-230 (z_goal latent
    structure non-trivial), Q-030 (6-cell z_resource x z_world routing),
    SD-012 (per-axis drive extension; SD-012-emergent invariants
    pending_substrate_reconfirmation flag deferred to next governance
    cycle), MECH-094 (call-site scoping; not applicable).
  Deferred Phase 3 follow-on (substrate_queue.json SD-049-PHASE-3 entry):
    SD-032 consumer cascade migrating AIC, PCC, pACC, dACC, salience,
    override, MECH-295 from reading goal_state._last_drive_level
    (collapsed scalar) to reading obs_dict["per_axis_drive"] directly.
    Action-trigger: V3-EXQ-514 failure on SD-032-mediated mode-switching
    pathway (which is not predicted by the current encoder-driven
    failure modes per verdict.md).
