## SD-049-PHASE-2 drive-coupling amend: kappa-scale + standing differential depletion (V3-EXQ-514r, MECH-436) (2026-06-17)
- SD-049-PHASE-2 drive-coupling amend -- IMPLEMENTED 2026-06-17 (substrate;
  MECH-436 stays candidate / substrate_ceiling / pending_retest_after_substrate
  until the V3-EXQ-514r-successor retest scores -- this PROMOTES NOTHING). Routed by
  the confirmed failure_autopsy_V3-EXQ-514r_2026-06-17 (overshoot disambiguator:
  drive CAN carve most_wanted at magnitude 5.0 -- flips on 2/5 guard-passing seeds,
  non-zero on all 5 -- so 514q's natural delta=0.0 is a drive-MAGNITUDE / environment
  artifact, NOT a MECH-436 falsification). The autopsy greenlit two coupled levers,
  kappa LOAD-BEARING; the 514q "DO NOT build until 514r resolves" hold is LIFTED.
  ROOT CAUSE (two structural facts the autopsy established): (1) wanting[k] =
  base_value[k] * (1 + kappa * per_axis_drive[k]) with a fixed kappa=2.0 and an in-run
  per-axis spread ~0.006 contributes ~0.012 to the multiplier -- swamped by real object
  base_value gaps that exceed 0.5 on seeds 45/46/47 (so even a magnitude-5.0 overshoot
  cannot flip them at the current kappa); (2) the P2 foraging ecology FULLY restores the
  contacted axis to 0 on consumption (causal_grid_world.py sigmoidal_saturating restore =
  cur_drive), and the WL bank is scored AROUND consumption events -- exactly where full
  restoration zeroes the just-consumed axis -- so the natural spread at scoring time is
  equalised to ~0.006.
  THE FIX (two no-op-default levers; bit-identical OFF):
    Lever (a) KAPPA SCALE (load-bearing) -- ree_core/goal.py
      (GoalConfig.incentive_drive_kappa_scale, default 1.0) + ree_core/utils/config.py
      (from_dims signature + goal_keys + local_goal_vals). IncentiveTokenBank.wanting()
      now uses effective kappa = incentive_drive_kappa_weight * incentive_drive_kappa_scale
      (getattr fallback 1.0). At 1.0 (default) wanting() is byte-identical; >1 lets a
      realistic per-axis drive spread compete with the real object base_value landscape.
    Lever (b) STANDING DIFFERENTIAL DEPLETION -- ree_core/environment/causal_grid_world.py
      (CausalGridWorldV2.per_axis_restoration_fraction, env-only kwarg, default 1.0, NOT
      surfaced through from_dims per the SD-049/SD-047/SD-048 env-only precedent). The
      curve-determined restore on contact is scaled by per_axis_restoration_fraction:
      restore = cur_drive * curve_mult * per_axis_restoration_fraction. At 1.0 (default)
      the contacted axis fully restores to 0 (bit-identical); <1.0 leaves STANDING per-axis
      drive on restored axes so a frequently-but-incompletely-restored axis carries residual
      drive at the WL scoring moment. Paired with the EXISTING divergent per_axis_drive_decay
      tuple (env-only kwarg the retest sets), this produces a persistent argmax-relevant
      per-axis drive SPREAD instead of the equalised ~0.006. (per_axis_restoration_fraction
      clamps to [0, 1].)
  Backward compatible: both default no-op. preflight 7/7 + the changed-subsystem
  contracts (goal + environment) PASS; 7 new contracts in
  tests/contracts/test_sd049_phase2_drive_coupling.py (C1 kappa_scale default bit-identical
  wanting / C2 kappa_scale>1 flips a most_wanted argmax the unscaled kappa cannot --
  the load-bearing behaviour / C3 restoration_fraction default fully restores [legacy] /
  C4 restoration_fraction<1 leaves standing drive on a real consumption step / C5 from_dims
  surfaces incentive_drive_kappa_scale onto cfg.goal / C6 restoration clamp + kappa_scale
  getattr fallback). Activation smoke 2026-06-17: env default==explicit-1.0 bit-identical
  over a scripted walk; partial restoration (0.4) raises the standing tail per-axis spread
  vs full restore.
  Phased training: N/A (no encoder head; scalar arithmetic on already-encoded latents +
  env state). MECH-094: N/A (env observation stream + waking IncentiveTokenBank arithmetic;
  no replay/memory write surface). Evidence-staleness (Step 8.5): NOT triggered -- both
  levers no-op-default; every existing experiment uses the defaults (kappa_scale 1.0,
  full restoration), so no dependent claim's measured mechanism changed. KEEP all evidence.
  GUARDRAILS HELD: MECH-229 leg (a) wanting!=liking object-bound dissociation (V3-EXQ-514o
  PASS 0.80) UNTOUCHED. No claims.yaml status flips (substrate-only); MECH-436 stays
  candidate / substrate_ceiling / pending_retest_after_substrate until the retest scores.
  Validation experiment: V3-EXQ-514s (the 514r-successor MECH-436 retest) -- re-run the
  514r overshoot + OFF/bank-disabled + recalibrated-argmax-relevance-readiness controls
  on the kappa-scaled + partial-restoration substrate. Pre-registered promotion target:
  natural drive-coupled delta mean(WL_drive - WL_nodrive) >= max(k*pstdev(delta), 0.15)
  on >=2/3 seeds converts MECH-436 substrate_ceiling -> supports. Non-vacuity preconditions
  (overshoot still flips a constructed gap, OFF/bank-disabled floor ~0, bank populated,
  recalibrated argmax-relevance) self-route substrate_not_ready_requeue, NEVER a false
  weakens. claim_ids=[MECH-436] (re-evaluated from scratch: 514r tagged MECH-229 broadly,
  but the run exercises the MECH-436 drive-coupling leg only). Queued via /queue-experiment.
  Design doc: REE_assembly/docs/architecture/sd_049_multi_resource_heterogeneity.md
  (drive-coupling amend section). Autopsy:
  REE_assembly/evidence/planning/failure_autopsy_V3-EXQ-514r_2026-06-17.{md,json}.
  Substrate_queue: REE_assembly/evidence/planning/substrate_queue.json (SD-049-PHASE-2).
  See SD-049 (parent; Phase 1 2026-05-03 + Phase 2 encoder 2026-05-04), MECH-436
  (drive-state-modulated wanting; the leg this enriches the retest for), MECH-229 leg (a)
  (untouched), SD-057 IncentiveTokenBank (the wanting() host lever (a) scales), V3-EXQ-514q
  (natural delta=0.0), V3-EXQ-514r (the overshoot disambiguator this amend is greenlit by),
  V3-EXQ-514s (validation/retest), MECH-094 (N/A).
