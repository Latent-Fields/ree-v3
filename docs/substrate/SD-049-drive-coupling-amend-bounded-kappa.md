## SD-049-PHASE-2 drive-coupling amend: BOUNDED kappa raise + deeper standing spread (V3-EXQ-514s, MECH-436) (2026-06-19)
- SD-049-PHASE-2 drive-coupling BOUNDED-RAISE amend -- IMPLEMENTED 2026-06-19 (substrate
  calibration; MECH-436 stays candidate / substrate_ceiling / pending_retest_after_substrate
  until V3-EXQ-514t scores -- this PROMOTES NOTHING). Routed by the confirmed
  failure_autopsy_V3-EXQ-514s_2026-06-18 (user-adjudicated 2026-06-18: clean self-route,
  substrate-ceiling PARTIALLY LIFTED). NO new flag, NO ree_core default change -- the two
  no-op-default levers landed 2026-06-17 (incentive_drive_kappa_scale, per_axis_restoration_fraction)
  already support the raise; this pass CALIBRATES the bounded operating point + adds the two
  invariant-guard contracts + retests.
  ROOT (514s, all five non-vacuity preconditions MET): lever (b) standing-differential-
  depletion WORKED (enriched_spread_met 1.0, mean_drive_spread_max 0.211 vs 514q's
  equalised ~0.006, per-seed 0.19-0.23) -- the env amend is done. But lever (a) kappa_scale=6.0
  is still SHORT: the WL dissociation metric is argmax-flip-gated, and on 3/5 seeds the real
  object base_value gaps exceed what kappa=6.0 x standing-spread~0.2 flips at natural magnitude
  (natural delta bimodal [-0.067, 0.176, 0.0, 0.211, 0.0]; mean 0.064 < margin 0.15; 2/5 seeds
  clear individually; overshoot mag 5.0 still flips 4/5). Residual shortfall LOCALIZED to the
  kappa lever.
  THE CALIBRATION (BOUNDED so drive does not swamp base_value; uses the autopsy's "and/or"):
    (a) RAISE incentive_drive_kappa_scale 6.0 -> 12.0 (a decisive 2x raise; effective kappa =
        incentive_drive_kappa_weight(2.0) * 12.0 = 24). BOUNDED check: at eff_kappa=24 a MODERATE
        base_value gap (1.0 vs 0.6) flips under a realistic standing spread (~0.25) while a
        CLEARLY-LARGER 10x gap (1.0 vs 0.10) does NOT -- drive carves near-ties without
        overriding a decisive base_value gap (a sated agent still wants the clearly-better
        object). Kappa-alone could not close the gap in a bounded way (the natural spread ~0.2
        is ~25x smaller than the mag-5.0 overshoot, so flipping the hardest seeds by kappa
        alone needs an UNbounded kappa) -- hence pairing with (b).
    (b) DEEPEN the standing spread: per_axis_restoration_fraction 0.3 -> 0.15 on the WL-scoring
        env only (training ecology stays 603n-canonical so survival/foraging competence is
        preserved). 0.15 leaves ~0.43 standing drive on a just-consumed axis (vs ~0.35 at 0.3),
        so the divergent-decay per-axis differences are larger at the WL scoring moment ->
        bigger argmax-relevant spread, less kappa needed per flip.
  INVARIANTS that survive the raise (the two new contracts):
    C7 OFF-floor-hard-zero: kappa multiplies ONLY the per-axis drive term, so at ZERO drive
       wanting()==base_value for ANY kappa_scale -- raising kappa cannot manufacture a
       drive-induced dissociation absent a drive signal (the substrate guarantee behind the
       experiment's bank-disabled wl_off_floor_fraction ~ 0 control). Hard-zero stays hard-zero.
    C8 bounded / wanting!=liking-from-base_value (MECH-229 leg-(a) intact): at kappa_scale=12.0 a
       clearly-larger 10x base_value gap is NOT flipped by a realistic standing spread -> drive
       does not dominate base_value. Brackets C2 (a moderate gap IS flipped).
  Backward compatible: NO ree_core default changed (kappa_scale default 1.0, restoration default
    1.0 both unchanged -> bit-identical OFF for every existing experiment). 9/9
    tests/contracts/test_sd049_phase2_drive_coupling.py (7 prior + new C7/C8) + goal/environment
    subsystem contracts (14/14) + 7/7 preflight PASS. The bounded kappa math was verified against
    the live wanting() formula (kappa_scale=12 flips a 1.0-vs-0.6 gap, holds a 1.0-vs-0.10 gap)
    and the restoration lever against the live env (0.15 -> 0.426 standing vs 0.3 -> 0.351).
  Phased training: N/A (no encoder head; scalar arithmetic on already-encoded latents + env
    state). MECH-094: N/A (env observation stream + waking IncentiveTokenBank arithmetic; no
    replay/memory write surface). Evidence-staleness (Step 8.5): NOT triggered -- no-op-default
    levers unchanged; every existing experiment uses the defaults, so no dependent claim's
    measured mechanism changed. KEEP all evidence.
  GUARDRAILS HELD: MECH-229 leg (a) wanting!=liking object-bound dissociation (V3-EXQ-514o PASS
    0.80) UNTOUCHED. No claims.yaml status flips (MECH-436 stays candidate / substrate_ceiling /
    pending_retest_after_substrate until the retest scores).
  Validation experiment: V3-EXQ-514t (the 514s-successor MECH-436 retest; supersedes V3-EXQ-514s)
    -- re-runs the 514r/514s overshoot + OFF/bank-disabled + recalibrated-argmax-relevance +
    enriched-spread controls on the kappa_scale=12.0 + restoration=0.15 substrate. Pre-registered
    promotion target unchanged: natural drive-coupled delta mean(WL_drive - WL_nodrive) >=
    max(k*pstdev(delta), 0.15) on >= 2/3 seeds -> MECH-436 substrate_ceiling -> supports. The five
    non-vacuity preconditions self-route substrate_not_ready_requeue, NEVER a false weakens; a
    preconditions-met FAIL with overshoot still flipping routes another bounded retune (514u),
    NOT a falsification. claim_ids=[MECH-436]; machine any. Queued via /queue-experiment.
  Design doc: REE_assembly/docs/architecture/sd_049_multi_resource_heterogeneity.md
    (bounded-raise amend section). Autopsy:
    REE_assembly/evidence/planning/failure_autopsy_V3-EXQ-514s_2026-06-18.{md,json}.
    Substrate_queue: REE_assembly/evidence/planning/substrate_queue.json (SD-049-PHASE-2).
  See SD-049-PHASE-2 drive-coupling amend (the 2026-06-17 lever landing this calibrates),
    MECH-436 (drive-state-modulated wanting; the leg the retest exercises), MECH-229 leg (a)
    (untouched), SD-057 IncentiveTokenBank (the wanting() host lever (a) scales), V3-EXQ-514s
    (the FAIL this amend addresses; lever b worked, kappa short), V3-EXQ-514t (validation/retest),
    MECH-094 (N/A).
