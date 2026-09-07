## SD-049: Multi-Resource Heterogeneity (Phase 1 substrate, 2026-05-03)
- SD-049 Phase 1: environment.multi_resource_heterogeneity -- IMPLEMENTED 2026-05-03
  (env-only Phase 1; encoder rebuild + SD-032 consumer cascade deferred to
  Phase 2, see Phase 2 follow-on note below).
  Module: ree_core/environment/causal_grid_world.py (CausalGridWorld /
  CausalGridWorldV2). Three additions to the env, gated by master switch
  multi_resource_heterogeneity_enabled (default False, bit-identical OFF).
  Substrate-roadmap H-priority #2; lit-anchored at sd_049 lit_conf=0.898
  (Berridge 2018 + Smith & Berridge 2007 + Kidd & Hayden 2015 + Shutts/Spelke
  2009 + Matthews/Tye 2016).
  Three additions:
    Addition 1 (multi-identity resources): num_resources cells split across
      n_resource_types qualitatively distinct types (default 3:
      food / water / novelty) per resource_type_distribution (default uniform).
      Each cell carries an identity tag stored in self._resource_type_grid
      (0 = no resource; type_idx + 1 elsewhere). Per-type resource lists in
      self._resources_by_type[type_idx]. Per-type proximity fields in
      self._resource_field_by_type[type_idx], computed in
      _compute_proximity_fields() when master is on. Per-type 5x5 field
      views appended to world_state in _get_observation_dict(); world_obs_dim
      grows by n_resource_types * 25 (250 -> 325 default 3-type ARM_2;
      250 -> 375 5-type ARM_3 overshoot). Per-type benefit profiles via
      resource_type_benefit_curves: "sigmoidal_saturating" / "sharp_saturation" /
      "novelty_decay" -- the novelty_decay curve attenuates contact benefit
      by per-cell familiarity, producing the structurally-distinct
      non-homeostatic benefit signal the wanting/liking dissociation
      requires (MECH-229 cohort).
    Addition 2 (per-axis homeostatic drive): per_axis_drive[n_axes] vector
      tracked alongside legacy agent_energy. Each per-tick depletion rate
      from per_axis_drive_decay; restoration on contact applied to the
      matching axis only (per the type-axis 1:1 mapping). When
      per_axis_drive_enabled, agent_energy collapses to
      1.0 - combiner(per_axis_drive) (combiner default "max", also
      "mean" / "sum") so all legacy SD-032 consumers (AIC / PCC / pACC /
      dACC / salience / override / MECH-295 bridge) continue to read
      obs_body[3] without modification. New observable:
      obs_dict["per_axis_drive"] for new experiments.
    Addition 3 (curriculum hook): resource_introduction_schedule:
      Dict[str, int] gates per-type spawn availability by self._global_step
      (cross-episode counter that survives reset()). Defaults to None ->
      all types available from step 0 (existing-experiment-equivalent
      behaviour even when master is on). Gate is checked at every reset()
      against current _global_step.
  Config (CausalGridWorld __init__ kwargs, env-only -- not surfaced through
  REEConfig.from_dims, matching SD-022 / SD-023 / SD-029 / SD-047 / SD-048
  precedent for env-only SDs):
    multi_resource_heterogeneity_enabled (bool, default False) -- master switch.
    n_resource_types (int, default 3).
    resource_type_names (tuple, default ("food","water","novelty")).
    resource_type_drive_axes (tuple, default ("hunger","thirst","curiosity")).
    resource_type_benefit_curves (tuple, default ("sigmoidal_saturating",
      "sharp_saturation","novelty_decay")).
    resource_type_distribution (tuple, default uniform via None).
    resource_type_benefit_amplitudes (tuple, default uniform 1.0 via None).
    per_axis_drive_enabled (bool, default False) -- per-axis vector observable
      and legacy-energy collapse.
    per_axis_drive_decay (tuple, default (0.001, 0.0015, 0.0005)).
    per_axis_drive_combiner (str, default "max"; also "mean" / "sum").
    novelty_familiarity_increment (float, default 0.2) -- per-contact
      per-cell familiarity bump.
    novelty_familiarity_recovery (float, default 0.0) -- 0 = permanent
      familiarity within episode; >0 = slow recovery toward 0.
    resource_introduction_schedule (Optional[Dict], default None).
  Defaults (truncated / padded for n_resource_types > 3) ensure the 5-type
  ARM_3 overshoot configuration runs without code changes (extra types get
  generic "type_<i>" / "axis_<i>" / "sigmoidal_saturating" entries).
  Per-resource-type bit-identical OFF: setting an entry to 0.0 in
  resource_type_distribution drops that type without code change (recovers
  ARM_1 from ARM_2 in the validation 4-arm sweep).
  Data flow:
    reset() -> distribute num_resources across active (curriculum-permitted)
      types by weighted draw -> populate _resources_by_type / _resource_type_grid
      -> _compute_proximity_fields rebuilds legacy + per-type fields ->
      world_state cat includes per-type 5x5 patches ->
      obs_dict["resource_field_view_<name>"] for each type +
      obs_dict["per_axis_drive"] + obs_dict["resource_type_at_agent"].
    step() resource consumption -> identify type from
      _resource_type_grid[new_x, new_y] -> apply type-specific benefit_curve
      with per-cell familiarity for novelty_decay -> remove cell from
      per-type list and clear tag -> apply axis restoration on matching
      axis -> increment _novelty_familiarity[cell] -> recompute proximity
      fields.
    step() post-move -> per_axis_drive[i] += per_axis_drive_decay[i]
      (gated on master + per_axis_drive_enabled) -> agent_energy =
      1 - combiner(per_axis_drive) -> _global_step += 1.
  info dict tags (always present, 0 / False when disabled):
    multi_resource_heterogeneity_enabled, sd049_n_resource_types,
    sd049_per_axis_drive_enabled, sd049_per_axis_drive_max,
    sd049_per_axis_drive_mean, sd049_n_resource_contacts_total,
    sd049_n_active_resources_by_type, sd049_global_step,
    sd049_resource_type_at_agent, sd049_preserve_per_type_density,
    sd049_density_budget_truncated.
  Backward compatible: master switch False by default; all per-type state
  initialized to empty/zero; world_obs_dim returns the legacy value;
  obs_dict surfaces no SD-049 keys; agent_energy follows legacy path.
  RNG draws gated inside `if multi_resource_heterogeneity_enabled:` so seed
  sequences for existing experiments are bit-identical when disabled.
  Bit-identical OFF guarantee verified 2026-05-03 (50-tick parity check
  across explicit-False vs default + V3-EXQ-513 C0 acceptance criterion).
  Activation smoke (2026-05-03):
    50-tick no-resource depletion test: per_axis_drive evolves linearly per
      configured rates (0.5 / 0.25 / 0.0 at 50 ticks * (0.01, 0.005, 0.0)
      decay). agent_energy collapses correctly to 1 - max_drive.
    200-tick 3-type random-policy: per-type spawn counts ~uniform across
      types (5/3/4 with 12 cells, weighted-uniform draw); contacts distribute
      across types; per-axis drive evolves with per-tick decay - per-contact
      restoration; novelty familiarity grows on contacted cells.
    Curriculum hook: water introduced at step 1000 -- spawn count [6, 0, 6]
      at episode 0; spawn count [4, 6, 2] at episode 1 after 1001 ticks
      advance _global_step.
    ARM_3 5-type overshoot: world_obs_dim 250 -> 375 (250 + 5*25); spawn
      distributes across 5 types per uniform distribution.
  Implementation choices (deviations / clarifications from SD doc):
    - Flat kwargs on CausalGridWorld.__init__ rather than nested dataclasses
      (MultiResourceHeterogeneityConfig / ResourceTypeConfig / PerAxisDriveConfig).
      Matches SD-022 / SD-023 / SD-029 / SD-047 / SD-048 precedent for env-only
      SDs. Nested dataclasses are not used elsewhere for env params.
    - Per-axis drive vector is parallel to legacy agent_energy, NOT replacing
      it. agent_energy collapses via configurable combiner so legacy SD-032
      consumers continue to read obs_body[3] unchanged. The per-axis vector
      is observable through obs_dict["per_axis_drive"] for new experiments
      and the deferred Phase 2 encoder upgrade. This is the lighter-cascade
      Phase 1 deviation from the SD doc's "drive_weight scalar must become
      a per-axis vector" -- the cascade through every SD-032 consumer is
      instead deferred to Phase 2 for clean phased validation.
    - Per-resource-type bit-identical OFF preserved: setting an entry to 0.0
      in resource_type_distribution drops the type without code change
      (per the SD doc requirement). All-zero distribution falls back to
      uniform to avoid spawn-zero pathology (use master switch to disable
      SD-049 entirely).
    - Type-axis 1:1 mapping: resource type i restores axis i. Future MECH
      may decouple via a learned mapping (registerable post-validation).
    - Encoder side NOT modified: the existing ResourceEncoder (SD-015)
      consumes the larger world_obs (325 ARM_2 / 375 ARM_3) without code
      changes -- it sees the per-type field views as additional dimensions
      in world_obs. Identity discrimination on z_resource is the Phase 2
      validation target; Phase 1 measures only substrate readiness.
  No trainable parameters. Pure env-side state + arithmetic. No phased
  training needed for Phase 1 (env-only).
  MECH-094: not applicable (env observation stream, not replay / simulation
    content). reset_to() scripted-eval bypass leaves SD-049 OFF (matches
    SD-047 / SD-048 precedent: scripted-eval comparator harnesses
    intentionally bypass enrichment substrates for clean tagging).
  Phase 2 follow-on (REGISTERED, not implemented): the validation
    experiment for SD-049 requires a z_resource encoder upgrade to
    discriminate identities (one-hot identity slot or learned embedding
    on the larger world_obs) and the SD-032 consumer cascade to read
    per_axis_drive directly rather than the collapsed obs_body[3].
    This is registered as a Phase 2 substrate task in
    REE_assembly/evidence/planning/substrate_queue.json (SD-049 entry,
    phase_2 follow-on note); the behavioural validation lands as
    V3-EXQ-514 (goal_resource_r lift + identity-recovery probe + wanting
    != liking trajectory dissociation) post-Phase 2.
  Validation experiment: V3-EXQ-513 queued (substrate readiness
    diagnostic; 4-arm sweep ARM_0/ARM_1/ARM_2/ARM_3 + curriculum check;
    13 acceptance criteria covering bit-identical OFF, per-type
    spawn / contact, per-axis drive evolution, novelty familiarity
    growth, agent_energy divergence from legacy path, ARM_3 overshoot
    obs_dim shape, curriculum gate / release. Dry-run smoke 2026-05-03:
    13/13 PASS at 30 ticks * 1 seed; full run at 200 ticks * 3 seeds
    expected ~5 min on Mac).
  Design doc: REE_assembly/docs/architecture/sd_049_multi_resource_heterogeneity.md
  Lit-pull: REE_assembly/evidence/literature/targeted_review_sd_049/
  See SD-049, SD-012 (per-axis drive extension cascade -- triggers
    pending_substrate_reconfirmation on SD-012-emergent invariants per
    invariant-types governance rule when Phase 2 lands), SD-015
    (z_resource encoder upstream substrate this enables in Phase 2),
    SD-005 (z_world routing must hold under multiple resource identities),
    MECH-229 (wanting/liking dissociation -- primary behavioural test
    post-Phase 2), MECH-230 (z_goal latent structure -- non-trivial
    multi-modal structure post-Phase 2), MECH-117 (existing
    wanting/liking trajectory dissociation -- non-degenerate evidence
    post-Phase 2), MECH-216 (schema generalisation across identity-
    distinct cues), Q-030 (6-cell z_resource x z_world routing sweep --
    well-posed post-Phase 2), ARC-030 (approach-avoidance symmetry
    across goal types), ARC-032 (theta-routing across goal identities),
    SD-047 (parallel substrate enrichment, file-coordinated, Phase 1
    landed first), SD-048 (parallel substrate enrichment, file-coordinated).
