## modulatory-bias-selection-authority AMEND: CEM elite-stage authority + behavioural throughput (V3-EXQ-931, 2026-08-19)
- modulatory-bias-selection-authority amend -- IMPLEMENTED 2026-08-19. Modules:
  ree_core/hippocampal/module.py (HippocampalModule._score_trajectory decomposition +
  propose_trajectories elite-stage rescale + throughput cache),
  ree_core/predictors/e3_selector.py (shared readiness helper + E3-layer verdict),
  ree_core/agent.py ("cem_elite" route source), ree_core/utils/config.py (flags).
  Routed by the CONFIRMED, human-gated failure_autopsy_931-932-wanting-authority-cluster_2026-08-16
  (V3-EXQ-931), recommended_substrate_queue_entry.implementation_hint_addendum.
  FOURTH convergent instance of "scoring-layer signals do not reach the committed
  argmax" -- after MECH-314 curiosity bias, MECH-320 vigor penalty and MECH-341
  within-class temperature -- at a NEW call site one layer UPSTREAM of E3.select,
  inside the hippocampal CEM's own elite selection, which failure_autopsy_V3-EXQ-914-914a_2026-08-13
  explicitly flagged the implemented E3.select fix does not reach ("flagged for a
  future session if this call site recurs"). It has now recurred, with a dose-response
  and a positive control.

  TWO INDEPENDENT FAILURES, and the autopsy ranks (b) as the more important:
    (a) AUTHORITY -- wanting_authority_ratio ~= 0.0037 (the wanting term's
      cross-candidate spread is ~0.37% of the terrain term's), so the term acts as a
      near-uniform offset and cannot move the elite argmin at the documented operating
      weight wanting_weight=0.5: selection_flip_rate = 0.0 in 5/5 seeds. First flips at
      w~50; 5/5 seeds at w~500; 0.80 at w=5000.
    (b) THROUGHPUT -- at w=5000, 80.3% of genuine refits flip the elite argmin
      (43/104, 96/98) while mean_resource_proximity is BIT-IDENTICAL to ablation
      (0.6229773644254133), because REEAgent.select_action re-scores the pool with
      e3.last_scores independently of the CEM's elite pick.
    Fixing (a) alone buys flipped picks and NO behavioural change.

  Site 1 -- _score_trajectory(return_components=True) returns
    (score, {"terrain", "modulatory"}) with score == terrain + modulatory EXACTLY,
    where modulatory is the SIGNED ADDITIVE contribution of wanting + curiosity +
    mode_value (negative for a favourable term; all three are subtracted here). Sign
    convention mirrors E3's scores = scores_raw + scale_factor * modulatory_total so
    the two layers read as one mechanism. The components are carried EXPLICITLY, never
    recovered as (score - terrain): that subtraction is what produced the V3-EXQ-643
    dead gate, losing a real ~0.17 range below the float32 ULP at ~1e32 primary
    magnitude. return_components=False (every pre-existing call site) is bit-identical.
  Site 2 -- propose_trajectories elite-stage rescale, immediately before the elite
    argsort: scale = gain * terrain_spread / modulatory_spread (basis "range" | "std"),
    scores = terrain + scale * modulatory, applied iff modulatory_spread > floor. E3's
    algebra verbatim, one layer upstream. scale is a DETACHED float, so gradient still
    flows through both terms and the SD-055 differentiable-CEM path keeps its grad path
    to cue_action_proj.
  Site 3 (THROUGHPUT) -- propose_trajectories caches
    HippocampalModule.last_candidate_modulatory_bias, a [K] per-candidate contribution
    over the FINAL pool (after ghost mixing, support injection, scaffold and chunk
    splicing), index-aligned to the list E3.select receives. agent.py gains route source
    "cem_elite" which identity-routes it through the existing channel_route_bias path
    into E3's modulatory accumulator, where E3's own bounded authority rescale gives it
    reach over the committed argmin. A candidate-count mismatch falls through to None
    rather than routing a stale vector (silent misattribution of each bias to a
    DIFFERENT trajectory would still produce a plausible nonzero route_range).

  THROUGHPUT ROUTE CHOICE, stated because the autopsy left it open: it offers either
    "propagate the CEM elite pick into E3's committed selection" or "document the elite
    stage as advisory-only". Adopted: a BOUNDED form of the first, plus the second as
    the DEFAULT posture. REJECTED -- letting the CEM pick override E3's argmin:
    _score_trajectory's own contract is ARC-007 STRICT ("no independent harm prediction
    here -- E3 introduces all value weighting"), so a terrain-plus-wanting argmin
    bypassing E3 would carry authority over the committed action WITHOUT harm
    weighting. Routing instead keeps E3 the sole committed selector, keeps harm
    weighting downstream of the routed bias, and keeps the authority bounded by
    modulatory_authority_gain.
  ADVISORY-ONLY is the shipped default: with every new flag off the elite stage cannot
    reach the committed action, and is now documented as such AT the code site plus a
    cem_elite_stage_advisory_only diagnostic. Any behavioural DV read off a
    hippocampal-CEM scoring manipulation without the route returns a STRUCTURAL null
    that LOOKS like a substrate or claim finding -- the failure that invalidated
    V3-EXQ-914/914a and cost that lineage two runs. NOTE the diagnostic is a
    NECESSARY-condition read: the module sees HippocampalConfig, so False means
    "throughput is possible", not "throughput is wired" -- modulatory_channel_route_range
    is the confirming read.

  STANDING READINESS ASSERTION (half c) -- generalises the 2026-06-10 route-range
    amendment ("assert the channel's cross-candidate range EXISTS") by exactly one
    step: the range must also be COMPETITIVE with the dominant term's, not merely
    nonzero. A nonzero-range gate passes at 0.0037, which predicts the measured null
    directly. Shared helper e3_selector.authority_spread_ratio() +
    authority_ratio_is_competitive(), used at BOTH layers so the statistic has ONE
    definition and a CEM-stage reading is directly comparable with an E3-stage one.
    REPORTED, NEVER ENFORCED in the substrate: enforcing it would break every existing
    default-off configuration, and watching a sub-competitive lever run is exactly what
    produced this finding. The gate belongs at /queue-experiment time.
  SAFETY: gain=0.5 < 1.0 keeps the modulatory term subdominant to a decisive terrain
    gap. The rescale is a NORMALISATION and therefore BIDIRECTIONAL -- it raises a
    sub-competitive lever (the 931 shape) and BOUNDS an over-dominant one down to gain;
    the bounding direction is what the gain < 1.0 convention exists for.
    modulatory_spread <= floor SKIPS the rescale entirely: "scaling zero is still zero"
    (V3-EXQ-648), amplifying a flat term manufactures numerical noise, not authority.
  Config (HippocampalConfig + REEConfig + E3Config + from_dims, all default no-op /
    bit-identical OFF): use_cem_modulatory_authority (bool, False),
    cem_modulatory_authority_gain (float, 0.5),
    cem_modulatory_authority_normalize_basis (str, "range" | "std"),
    cem_modulatory_authority_min_spread_floor (float, 1e-6),
    use_cem_modulatory_throughput (bool, False),
    authority_competitive_ratio_floor (float, 0.1 -- fanned out by from_dims to all
    three homes, since each reporting layer sees only its own sub-config),
    and modulatory_channel_route_source gains the value "cem_elite".
    All six are given from_dims entries rather than left dataclass-only: a field with
    no from_dims entry falls into **kwargs and is silently dropped ([memory]
    reference-reeconfig-from-dims-silent-kwargs).
  Diagnostics: get_last_propose_diagnostics() gains
    cem_modulatory_authority_{enabled,fired_iterations,scale_factor_mean,ratio,competitive},
    cem_modulatory_throughput_{enabled,available,ratio,competitive},
    cem_elite_stage_advisory_only, authority_competitive_ratio_floor;
    e3_selector.last_score_diagnostics gains modulatory_authority_ratio +
    modulatory_authority_ratio_competitive.
  Backward compatible: verified by re-running the motivating driver itself --
    experiments/v3_exq_931_cem_wanting_weight_selection_authority.py --dry-run
    completes (exit 0) and reproduces its pre-amend shape exactly: flip_rate 0.0000 and
    seeds_with_flip 0/5 in all five arms, mean_resource_proximity 0.6078 BIT-IDENTICAL
    across w=0/0.5/50/500/5000, outcome FAIL label substrate_not_ready_requeue.
  MECH-094: N/A in the write direction -- _score_trajectory and _curiosity_bonus are
    read-only over hypothesis-space CEM candidates and write no memory; no simulation
    or replay content is produced. Phased training: N/A -- no encoder head is added and
    nothing here trains; all terms are existing read-only field evaluations recomposed
    arithmetically.
  Contracts: tests/contracts/test_cem_modulatory_authority.py (16; roughly half
    NEGATIVE CONTROLS -- levers inert by default, cache stays None not empty, flat term
    is not amplified, degenerate readiness inputs read 0.0 so the gate stays CLOSED,
    from_dims reachability, and the readiness statistic is reported-never-enforced).
  Validation experiment: NOT queued as part of this build, deliberately. Per half (c) a
    behavioural falsifier is gated on this landing AND on the spread ratio becoming
    competitive; queueing one now would reproduce V3-EXQ-931's structural null. The
    correct next step is a readiness/validation diagnostic for the build itself.
  Design doc: REE_assembly/evidence/planning/cem_elite_authority_throughput_design_2026-08-19.md
  See modulatory-bias-selection-authority (substrate_queue; AMENDED, not forked -- the
    autopsy explicitly rejects a parallel entry as fragmenting one bottleneck across two
    records), MECH-236, the entry's 18-claim unblocks_claims list,
    failure_autopsy_931-932-wanting-authority-cluster_2026-08-16,
    failure_autopsy_V3-EXQ-914-914a_2026-08-13 (whose learning "the existing lever
    already covers it" this SUPERSEDES -- the lever does not cover it, because its
    authority does not propagate to the committed action), ARC-007 STRICT (preserved).

