## modulatory-bias-selection-authority AMEND: TOP-K shortlist mode (569h conversion-ceiling, 2026-06-16)
- modulatory-bias-selection-authority TOP-K shortlist amend -- IMPLEMENTED 2026-06-16.
  Modules: ree_core/predictors/e3_selector.py (shortlist eligible-set construction +
  1 diagnostic key), ree_core/utils/config.py (2 E3Config fields + from_dims passthrough).
  Routed by the active failure_autopsy_V3-EXQ-569h_2026-06-16 (which hands off
  /implement-substrate amend modulatory-bias-selection-authority) + critical_path_synthesis
  Path 2 (shortlist-then-modulate as the real architectural change, not another gain tweak).
  behavioral_diversity_isolation:GAP-A.
  ROOT CAUSE (the 7th GAP-A amend; both prior conversion levers left ~0 behavioural
  conversion): V3-EXQ-569h FAIL/non_contributory 2026-06-16T10:11Z
  (conversion_ceiling_persists_despite_routing) -- readiness fully MET (route_range 0.31
  3/3, e2-divergence 3/3, non_degenerate, negative control clean) but the additive
  ARM_STD_G2 lever (normalize_basis=std + authority_gain=2.0) cleared C_R1B (selected-action
  entropy strict-above BOTH matched-noise and proposer) on only 1/3 seeds (need 2/3). The
  pre-existing MARGIN shortlist (the 2026-06-15 conversion amend lever (b)) ALSO failed:
  V3-EXQ-684 ARM_SHORTLIST (margin 0.25) converted 0/3, committed entropy 0.337 BELOW the
  collapsed proposer 0.549. Manifest diagnosis: modulatory_shortlist_size_mean 6.25-8.54 --
  the margin cutoff (best_raw + 0.25*raw_score_range) admitted ~7 of ~8 candidates = a
  near-WHOLE, state-STABLE eligible set, so the committed-branch argmin(_modulatory_accum)
  collapsed to the modulatory channel's GLOBAL favourite (monostrategy). The additive arms
  (which blend F) preserved MORE diversity than the margin-shortlist that let modulation
  solely decide over a near-whole set.
  THE FIX (no-op-default; bit-identical OFF): a TOP-K shortlist mode. New E3Config levers
  (mirrored on REEConfig.from_dims): modulatory_shortlist_mode "margin" (default, legacy
  bit-identical) | "top_k", and modulatory_shortlist_k (default 3). In "top_k" the eligible
  set is the k F-best candidates by PRIMARY score (k smallest raw_scores; lower-is-better,
  via torch.topk(raw_scores, k, largest=False); k clamped to [1, K]); the within-set
  selection rule is UNCHANGED (deterministic argmin of the routed _modulatory_accum when
  committed / softmax-multinomial when uncommitted). WHY top-k converts where margin
  collapsed: a SMALL fixed k gives an eligible set whose MEMBERSHIP rotates with state (the
  k F-best change as the agent moves), so argmin-of-the-routed-channel within the rotating
  small set produces committed-action diversity that reflects genuine per-candidate
  structure -- and BEATS unstructured matched-noise (the C_R1B non-vacuity bar; entropy from
  channel STRUCTURE, not sampling noise, because the within-set rule is deterministic).
  SAFETY preserved at any internal strength: only the k F-best are eligible, so a
  clearly-harmful candidate is never selectable (contract-verified). Diagnostic
  modulatory_shortlist_mode added to last_score_diagnostics (pre-seed + active site).
  Backward compatible: modulatory_shortlist_mode="margin" by default AND the entire
  shortlist block is gated on use_modulatory_shortlist_then_modulate (default False), so
  the mode is inert for every existing experiment -> bit-identical. 22/22 conversion +
  shortlist contracts PASS (18 prior + 4 new top-k in
  tests/contracts/test_e3_score_bias_candidate_support.py: default-mode-margin + shortlist-OFF
  bit-identical / restricts-to-exactly-k + safety-worst-primary-never-selected / top_k set
  SMALLER than a loose-margin set on the same pool [the 569h fix made deterministic] /
  within-top-k routed-modulatory argmin decides the winner).
  Phased training: N/A (pure-arithmetic selection rule; no learned parameters; no gradient
  flow). MECH-094: N/A (waking committed-selection path; no replay/memory write surface;
  unchanged). Evidence-staleness (Step 8.5): NOT triggered -- no-op-default levers; every
  existing experiment uses mode='margin' + shortlist OFF, so no dependent claim's measured
  mechanism changed. KEEP all evidence.
  GOVERNANCE: ARC-065 NEITHER promoted NOR weakened; stays provisional / substrate_ceiling /
  pending_retest_after_substrate; MECH-341 / ARC-062 / MECH-309 / MECH-294 untouched.
  claims.yaml NOT modified (substrate-only amend; record lands in substrate_queue.json +
  the GAP-A plan node).
  Validation experiment: V3-EXQ-569i (queued via /queue-experiment) -- a 569-lineage
  successor (3-arm matched-entropy, in-arm route-range gate + e2-divergence gate + C_R1B
  strict-above BOTH matched-noise AND proposer) arming the TOP-K shortlist config as the
  conversion constant across arms (use_modulatory_shortlist_then_modulate + mode=top_k +
  use_modulatory_channel_routing + candidate_summary_source=e2_world_forward). Pre-registers
  the conversion-ceiling off-ramp (readiness met + no C_R1B lift -> conversion_ceiling_persists,
  non_contributory, NOT an ARC-065 falsification). claim_ids=[ARC-065].
  Design doc: REE_assembly/docs/architecture/modulatory_bias_selection_authority.md
  (TOP-K shortlist amend section). Autopsy:
  REE_assembly/evidence/planning/failure_autopsy_V3-EXQ-569h_2026-06-16.* (concurrent session).
  Substrate_queue: REE_assembly/evidence/planning/substrate_queue.json
  (modulatory-bias-selection-authority entry).
  Closure node: REE_assembly/evidence/planning/behavioral_diversity_isolation_plan.md
  (behavioral_diversity_isolation:GAP-A).
  See modulatory-bias-selection-authority (parent; gap-relative 2026-06-03 + 643a float32 fix
  2026-06-06 + route-range 2026-06-10 + gain/contrast+margin-shortlist 2026-06-15 above),
  V3-EXQ-569h (the FAIL this amend addresses), V3-EXQ-684 (the margin-shortlist 0/3 +
  size 6.25-8.54 diagnosis), V3-EXQ-571 (F-dominated primary 88-89%), ARC-065 GAP-A
  (closure node), MECH-341 / ARC-062 / MECH-309 / MECH-294 (shared conversion substrate; xref),
  V3-EXQ-569i (validation/falsifier), MECH-090 (admission gate; unchanged), MECH-094 (N/A).
