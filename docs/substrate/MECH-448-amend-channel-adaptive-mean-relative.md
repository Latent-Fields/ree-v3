## MECH-448 AMEND: channel-adaptive (mean-relative) eligibility floor (collapse ~5 per-channel hand-floor dances into one knob) (2026-06-21)
- MECH-448 channel-adaptive envelope amend -- IMPLEMENTED 2026-06-21 (substrate;
  MECH-448 stays candidate -- PROMOTES NOTHING; the readiness EXQ + the 689a-successor
  falsifier are the sequenced next steps). Makes the rank-preserving F->eligibility
  demotion envelope CHANNEL-ADAPTIVE so it auto-calibrates per channel instead of needing
  a hand-tuned global floor. Routed by the confirmed-twice no-op signature: the absolute
  share floor (f_eligibility_envelope_floor=0.30) was tuned to PASS on the GAP-A foraging
  bank (V3-EXQ-689d), but each downstream channel has a different F-merit distribution, so
  the same fixed floor mis-fires -- V3-EXQ-654h (arc_062 rule-apprehension) admitted ALL
  candidates (f_eligibility_excluded_count==0, the lever never engaged; "485i twin",
  failure_autopsy_V3-EXQ-654h REE_assembly f41fe845fd); V3-EXQ-485i->485j (OFC) needed a
  bespoke per-seed envelope-floor recalibration to engage (485j then CONFIRMED OFC
  discrimination CONVERTS under demotion -- the lever generalises off GAP-A; residual was a
  separate devaluation test-design gap re-queued as 485k, NOT the envelope; 32c2518c1d).
  Direction confirmed (MECH-448 generalises); the only gap was the per-channel floor sweep.
  THE FIX (no-op default; bit-identical OFF):
    Module: ree_core/predictors/e3_selector.py (_f_eligibility_envelope branch),
      ree_core/utils/config.py (E3Config 2 fields + from_dims passthrough).
    use_f_eligibility_adaptive_floor (E3Config, default False): replaces the fixed
      absolute share floor with a MEAN-RELATIVE one --
      floor = f_eligibility_adaptive_mean_factor * elig.mean() -- so a candidate is
      eligible iff its share of the competing merit exceeds mean_factor times the field's
      OWN mean share, rather than an absolute constant. Mean-relative is SCALE-INVARIANT
      (auto-calibrates to each channel's F-merit distribution, no per-channel hand-tuning)
      AND retains the MECH-448 CONFLICT-GRADE (a decisive F-winner pulls the mean up so
      others fall below -> narrow envelope; a near-tie sits near the mean -> wide). It is
      still a threshold on elig (monotone in merit), so the eligible set stays an F-RANK
      PREFIX (rank-preserving). For mean_factor >= 1.0 on any NON-uniform field at least
      one candidate is below the mean share, so the envelope EXCLUDES (excluded_count > 0)
      by construction -- the 654h all-admit no-op cannot recur, and the 485i/485j bespoke
      floor recalibration is no longer needed. Collapses the ~5 per-channel hand-floor
      dances (654h/485i/485j + the pending 625/445/687 successors) into ONE global knob.
    f_eligibility_adaptive_mean_factor (E3Config, default 1.0): "above-average share"
      threshold multiple. Single GLOBAL knob (NOT per-channel), sweepable.
  Preserved: the existing exact-tie / flat-F early returns (merit_sum<=1e-8 -> wide) + the
    empty-eligible all-admit fallback are unchanged; the demotion still requires a
    modulatory channel (_modulatory_accum not None). Still emits the
    f_eligibility_excluded_count>0 non-degeneracy + winner_neq_f_argmin + envelope_size +
    rank_preserving diagnostics (e3_selector.py ~1391-1423).
  Backward compatible: use_f_eligibility_adaptive_floor=False by default -> the floor reads
    the legacy fixed config value -> bit-identical; AND use_f_eligibility_demotion itself
    stays default OFF so the whole block is skipped for every existing run (double guard).
    16/16 MECH-448 contracts (8 prior + 8 new amend: adaptive config defaults no-op /
    from_dims surfaces flags / adaptive-OFF bit-identical to fixed floor / EXCLUDES across
    2 differing-scale synthetic dists where the fixed 0.30 floor no-ops on the 654h-like D1
    and collapses on D2 / rank-preserving holds / conflict-grade preserved near-tie>decisive)
    + 8/8 preflight + 48/48 E3-cluster contracts (e3 score-bias / conflict-grade / DR-12 /
    ARC-065 GAP-A / stratified-temperature) PASS.
  Phased training: N/A (pure-arithmetic selection rule; no learned parameters; no gradient
    flow). MECH-094: N/A (waking committed-selection path; no replay/memory write surface).
    Evidence-staleness (Step 8.5): NOT triggered -- no-op-default flag; every existing
    experiment uses the default (adaptive off + demotion off), so no dependent claim's
    measured mechanism changed. KEEP all evidence.
  GOVERNANCE: PROMOTES NOTHING. MECH-448 stays candidate; ARC-107 untouched; claims.yaml
    NOT modified. substrate_queue f_dominance_conversion_ceiling rung-2 implementation_log
    amended.
  Validation experiment: V3-EXQ-689e channel-adaptive envelope readiness diagnostic
    (claim_ids=[]) queued via /queue-experiment -- shows f_eligibility_excluded_count > 0
    lands in a productive range on >= 2 real channel substrates (the arc_062 rule-apprehension
    bank that no-opped in 654h + the OFC/foraging bank) with the SAME global adaptive config
    (no per-channel hand-tuning); bit-identical OFF as the negative control;
    substrate_not_ready_requeue if the adaptive floor still no-ops on any channel.
  Design doc: REE_assembly/docs/architecture/mech_448_f_eligibility_demotion.md
    (channel-adaptive amend section).
  See MECH-448 (parent; rank-preserving demotion lever landed 2026-06-20 above), ARC-107
    (architecture), MECH-439 (F-dominance root), V3-EXQ-654h (the arc_062 all-admit no-op
    this fixes), V3-EXQ-485i/485j (the OFC bespoke-recalibration this removes), V3-EXQ-689d
    (the GAP-A bank the fixed 0.30 floor was tuned to), V3-EXQ-689e (validation), MECH-094 (N/A).
