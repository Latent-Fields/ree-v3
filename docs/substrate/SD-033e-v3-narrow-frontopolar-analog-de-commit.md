## SD-033e (V3-narrow): Frontopolar-analog de-commit lever / MECH-264 (2026-07-09)
- SD-033e (V3-narrow slice): pfc.frontopolar_analog -- IMPLEMENTED 2026-07-09.
  Module: ree_core/pfc/frontopolar_analog.py (FrontopolarAnalog, FrontopolarConfig;
  previously a V4 no-op stub). A DELIBERATE low-odds-but-owed V3-narrow slice of
  SD-033e -- ONLY function (1) MECH-264 counterfactual-value + function (4)
  disengagement-for-exploration -- landed as a commit-LATCH DE-COMMIT /
  switch-propensity lever for the DURATION face of the F-dominance conversion
  ceiling (MECH-439). The full learned MECH-264/MECH-265 heads + gateway remain
  V4 (still stubbed under the SEPARATE use_frontopolar_analog flag; single-resource
  env cannot supply the K>=2 goals MECH-265 needs). Owed residual named by
  failure_autopsy_MECH-445-cluster-715a-717_2026-07-07 ("the arming reliability
  gap needs a DISTINCT de-commit-release lever, not more F-moderation").
  Config: REEConfig.use_frontopolar_decommit (bool, default False) +
    frontopolar_gain (float, default 0.0). Requires
    use_natural_commit_urgency_release=True (the accumulator it feeds).
  Mechanism (NON-F, entry-relative -- the two mitigations against the 709/711/713
    input-reweighting wash-out):
    - MECH-264 counterfactual value = goal-proximity ADVANTAGE of the best
      UNCHOSEN candidate over the committed endpoint in z_world space:
      cfv = ||z_chosen - z_goal|| - ||z_alt - z_goal|| (parameter-free geometry,
      NO learned head, NO training phase). Sourced from goal-proximity, NOT F,
      so it is not F-monotone (does not wash like the exhausted arbitration route).
    - Release pressure = frontopolar_gain * max(0, cfv_now - cfv_at_entry) --
      ENTRY-RELATIVE derivative: fires only when the foregone alternative has
      IMPROVED relative to the commit moment (a level-based form is an
      F-derivative and washes one layer down).
  Data flow: e3.select() captures the chosen + best-unchosen z_world endpoints
    (e3._fp_chosen_world_endpoint / _fp_alt_world_endpoint; detached, unconditional,
    output-neutral) -> commit-entry hook (agent.py, alongside note_commit_entry)
    stores cfv_at_entry -> each maintenance tick the release block computes cfv_now
    + pressure and injects it via NaturalCommitUrgencyRelease.tick(
    frontopolar_pressure=...) -> fires on the SAME urgency >= release_bound ->
    beta_gate.release() + _ncl_lever_fired=True (reuses the existing rung-6 release
    block + inherits its latch-hold yield for free).
  Attribution: natural_commit_urgency.get_state()["frontopolar_release_count"] --
    count of fires with a POSITIVE frontopolar pressure at the firing tick
    (cfv_now > cfv_at_entry) -- proves a REAL switch toward an improved foregone
    option, not F noise or a flat timeout. Also frontopolar_pressure_accum /
    frontopolar_last_pressure.
  Backward compatible: use_frontopolar_decommit=False by default -> the agent does
    not instantiate FrontopolarAnalog (self.frontopolar is None) -> every read/inject
    site is a guarded no-op and NaturalCommitUrgencyRelease.tick(frontopolar_pressure
    =0.0) is bit-identical to the pre-change urgency arithmetic. Verified: NCUR +
    latch-hold contracts (17) unchanged, config/pfc/e3 contract sweep (201) unchanged,
    v3_exq_460i --dry-run (OFF path, real env) clean, and the frontopolar_gain=0
    contrast arm is byte-identical to the pure NaturalCommitUrgencyRelease lever.
  Biological basis: Boorman et al 2009 (frontopolar counterfactual value);
    Mansouri et al 2015 (frontopolar disengagement-for-exploration lesion signature,
    MECH-090-release-facilitating). Koechlin & Summerfield 2007 (BA 10 substrate).
  MECH-094: n/a on the de-commit path -- the lever reads waking selection endpoints
    and injects release pressure; it writes nothing to memory and does not run in
    simulation/replay (the NCUR tick is called only on the waking release block,
    and tick(simulation_mode=True) is a no-op by construction).
  V3-narrow scope: function (2) MECH-265 relative-importance + the gateway mode stay
    V4 stubs (compute_relative_importance / compute_disengagement_bias NOT on the
    de-commit path).
  Smoke test (2026-07-09): geometric cfv sign correct (alt-nearer>0, alt-farther<0);
    entry-relative pressure fires only on improvement, scaled by gain; NCUR
    frontopolar_pressure=0.0 bit-identical, pressure>0 fires + attributes; agent OFF
    -> frontopolar is None; agent ON -> get_state fields present; end-to-end on the
    real CausalGridWorldV2 (460i builders) the endpoint capture populates on real
    trajectories, the geometric cfv is live on real z_world endpoints, and the
    release chain fires + attributes.
  SEQUENCING GATE: validate on the ISOLATED GAP-A single-arena config only (the
    integrated all-ON agent has no competent committed foraging to de-commit FROM
    per failure_autopsy_V3-EXQ-719a_2026-07-08; the all-ON validation is gated on the
    V3-EXQ-724 competence-localization diagnostic).
  Validation experiment: V3-EXQ-<PENDING> queued (isolated-GAP-A; ON vs
    frontopolar_gain=0 flat-urgency contrast on 460h strong-F seeds; gate: median
    ncur_last_occupancy_at_release drops >=~40% on >=2/3 strong-F seeds ATTRIBUTABLE
    to frontopolar_release_count>0).
  Design doc: REE_assembly/docs/architecture/sd_033_pfc_subdivision_architecture.md
    (section SD-033e).
  See SD-033, SD-033e, MECH-264, MECH-439, MECH-090, the rung-6
    natural_commit_urgency lever, and failure_autopsy_V3-EXQ-460h /
    failure_autopsy_MECH-445-cluster-715a-717 / failure_autopsy_V3-EXQ-719a.
