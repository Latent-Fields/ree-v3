## SD Design Decisions Implemented (V3) — continued
- infant_substrate:GAP-1 / INF-ENV-001 — harm gradient env feature —
  IMPLEMENTED 2026-05-16. ree_core/environment/causal_grid_world.py.
  Env-only constructor kwargs (NOT REEConfig / from_dims): harm_gradient_enabled
  (default False), harm_gradient_outer_radius (3.0), harm_gradient_inner_radius
  (0.0), harm_gradient_scale (1.0). step(): when transition_type == "none" and
  nearest-hazard distance d in (inner, outer], apply
  -hazard_harm * (1 - d/r_outer)^2 * scale to harm_signal; transition_type
  harm_gradient. Terminal hazard contact unchanged. Info keys: harm_gradient_enabled,
  harm_gradient_reward_this_tick, harm_gradient_dist_to_nearest. Backward compatible:
  disabled by default; bit-identical OFF verified (test_harm_gradient_gap1.py 10/10).
  Not a learning module — no encoder, no phased training, MECH-094 N/A.
  Validation: V3-EXQ-576 PASS 20260516T195014Z (diagnostic, claim_ids=[]).
  Unblocks DEV-NEED-004 gate experiments (tier-1: V3-EXQ-587 GAP-10). See
  infant_substrate_plan.md, infant_substrate_expansion.md Section 5.1, ARC-013.

- commitment_closure:GAP-3 — CausalGridWorldV2 env extensions, primitives 1-3 —
  IMPLEMENTED 2026-05-17. ree_core/environment/causal_grid_world.py.
  Env-only constructor kwargs (NOT REEConfig / from_dims — same precedent as
  harm_gradient_* / microhabitat_* / transient_benefit_*). All master switches
  default no-op; bit-identical OFF verified suite-wide (full contract
  regression 434/434).
  Primitive 1 — adaptive tolerance-band completion: completion_tolerance_enabled
    (default False), _frac (0.0), _cells (-1; >=0 overrides frac), _metric
    ("chebyshev" | "manhattan"), _targets ("waypoint"; "waypoint+resource"
    RESERVED — raises ValueError, ships waypoint-only per Q-1a), _kernel
    ("hard" | "graded_exp"; credit exp(-d/lambda)), _lambda (1.0). Wraps the
    waypoint exact-match; OFF and frac=0.0 both dynamics bit-identical.
  Primitive 2 — counter-evidence injection hook (graded contingency
    degradation, NOT signed perturbation): counter_evidence_enabled (False),
    _interval (50), _prob (0.5), _degrade_step (0.2), _degrade_floor (0.0),
    _requires_persistent_rule (True). _inject_counter_evidence() cloned
    structurally from the SD-029 scheduled-injection pattern; lowers the
    committed target's outcome-validity toward the floor while the rule_state
    is persistent; committed-target reward scaled by validity; context
    (hazards/resources/drift) untouched. transition_type set by the existing
    waypoint path.
  Primitive 3 — dual simultaneously-active resource cue: dual_cue_enabled
    (False), _min_active_ticks (10), _replace_on_early_consume (False =
    invalidate-episode, Q-3b; True is diagnostic-only), _type_tags ((1,2)).
    Rides the SD-049 multi-resource path; RAISES ValueError if SD-049 not
    enabled (Q-3a fail-fast, no silent auto-enable).
  16 always-present info keys (inert sentinels when the relevant primitive is
  disabled). Backward compatible: disabled by default; existing experiments
  unaffected. Not a learning module — no encoder head, no phased training, no
  MECH-094 simulation-write surface.
  Validation: tests/contracts/test_env_extensions_gap3.py 14/14 (C1 bit-
  identical OFF + frac=0.0 dynamics-identical; C2 tolerance/graded_exp/metric;
  C3 counter-evidence persistent-only + monotone validity->floor +
  context-invariant; C4 dual-cue SD-049 fail-fast + accounting; C5 spec
  section-5 integration smoke) + full contract regression 434/434. NO
  claim-validation EXQ queued — spec section 5: Phase 3 is env infrastructure
  with no claim-validation EXQ (a spec-sanctioned deviation from the
  implement-substrate skill Step 8; concurrency also forbade queue edits).
  Spec: REE_assembly/evidence/planning/causalgridworldv2_env_extensions_spec.md
  (Status: IMPLEMENTED 2026-05-17). Closes commitment_closure:GAP-3 (unblocks
  GAP-8). Deliverable 4 (phased rule_state training curriculum) deliberately
  SEPARATE (spec section 6) — the SD-034/MECH-266/MECH-268 behavioural arms
  still need it. See commitment_closure_plan.md, claims SD-034 / MECH-266 /
  MECH-268. claims.yaml NOT modified (env infra unblocks but does not itself
  promote).

- commitment_closure:GAP-11 -- Phased rule_state training curriculum harness helper
  -- IMPLEMENTED 2026-05-17.
  File: experiments/committed_mode_curriculum.py (experiment-harness helper, NOT a
  ree_core substrate scheduler -- O-1 resolved).
  Public API: run_p0_warmup(), run_p1_consolidation(), run_p2_eval(),
  clone_trained_agent(), P0Result, P1Result, CommittedModeMetrics.
  Data flow: P0 trains E1+E2 on easy env (EXQ-321b run_training pattern) until
  running_variance < commit_threshold; P1 consolidates on target env until
  total_committed_steps per episode >= commitment_floor(100); P2 is frozen-policy
  eval measuring committed_steps / hold_rate / rule_state_norm.
  Mid-probe abort gate (default 60% of budget): fires commitment_not_elicited ->
  caller escalates as R1 substrate mis-calibration finding, not a tuning problem.
  O-2 mandatory contrast: every arm must run BOTH emergent (P0->P2) and
  forced-rv clone (clone_trained_agent + set rv=0.001 + run_p2_eval).
  O-3: at most ONE commitment_threshold relaxation step (threshold_relaxation param,
  max meaningful value 0.125); further non-convergence = substrate finding, escalate.
  Backward compatible: no ree_core changes; no experiment script changes.
  Generalises EXQ-321b run_training + clone_for_condition + EXQ-543b P0/P1 scaffolding.
  Blocks cleared: SD-034, MECH-266, MECH-268, MECH-090, SD-021 behavioural arms
  (V3-EXQ-460b/461/463b/464b/466b/467b/468b) -- see commitment_closure_plan.md.
  Pilot experiment: EXP-0157 / V3-EXQ-592 (GAP-11 pilot, 3 arms: EMERGENT/FORCED_RV/STARVED).
  New ID (not V3-EXQ-461b) because V3-EXQ-461 was a synthetic scripted PASS, not
  emergent training. Queued 2026-05-17. Supersedes V3-EXQ-461.

- INV-074 / MECH-333 / MECH-334: Phase-3 plasticity-injection crystallization
  + EWC residue write-protect -- IMPLEMENTED 2026-05-17.
  Files: ree_core/policy/gated_policy.py (GatedPolicy.crystallize() +
  expansion_parameters() + .crystallized; forward gains the post-crystallize
  expansion branch), ree_core/residue/field.py (ResidueField.
  snapshot_ewc_anchor() + ewc_penalty() + .ewc_anchored), experiments/
  infant_curriculum.py (InfantCurriculumScheduler on_phase3_entry fire-once
  hook), ree_core/utils/config.py (REEConfig + ResidueConfig + from_dims),
  ree_core/agent.py (GatedPolicyConfig passthrough).
  Config: REEConfig.crystallize_at_phase3 (default False; set True to enable).
  Subsidiary: gated_policy_crystallize_expansion_hidden (32),
  residue_ewc_lambda (0.0 = anchor captured, penalty inert). When
  crystallize_at_phase3=True, from_dims also arms ResidueConfig.ewc_enabled
  + ewc_lambda.
  Data flow: infant-curriculum Phase 2->3 transition -> scheduler fires the
  experiment's on_phase3_entry closure -> agent.gated_policy.crystallize()
  (requires_grad=False on head_0/head_1/discriminator; fresh plastic
  expansion MLP, last-Linear zero-init so output is bit-identical at the
  transition instant; forward = frozen_gated(x) + expansion(x.detach()),
  the .detach() blocking diversity gradient from the crystallized weights)
  + agent.residue_field.snapshot_ewc_anchor() (centers/weights anchor +
  established-basin Fisher proxy |anchor_w|*active_mask). The experiment's
  post-Phase-3 optimizer targets gated_policy.expansion_parameters() (plus
  dACC / MECH-313 / MECH-314a / MECH-320 diversity params) and adds
  residue_field.ewc_penalty() to its loss.
  Biological basis: Nikishin et al. 2023 NeurIPS plasticity injection
  (MECH-333 option E open-phase channel); Kirkpatrick et al. 2017 EWC
  write-protect (MECH-334 closure, faithful to "high resistance to
  overwriting established basins" -- NOT a hard freeze). Grounds INV-074
  (plasticity crystallization necessity), the V3-tractable subset of
  MECH-333/334, and ARC-075 (infant curriculum plasticity magnitude
  asymmetry, not just temporal scheduling).
  Pre-check (encoded in design doc): MECH-314b (uncertainty, reads
  e3._running_variance) and MECH-314c (learning-progress, EMA of
  |PE_t-PE_{t-K}| fed e3._running_variance) are forward-model-error-
  dependent -- 314c is the canonical Pathak 2017 ICM self-defeat case --
  and decay to ~0 before Phase-3 crystallization fires, so they cannot
  establish competitive weight on the expansion layer. MECH-313 (constant
  temperature), MECH-314a (residue-RBF novelty, Wittmann 2008 RPE-
  independent), MECH-320-primary (avg-reward-rate EWMA, Niv 2007), and
  dACC/MECH-260 (state-dependent recency) are F-robust and are the
  meaningful signals to route.
  Backward compatible: disabled by default; crystallize_at_phase3=False
  is bit-identical (forward never references the expansion; EWC penalty
  returns a 0.0 scalar). Contract regression 484/484 PASS; backward-compat
  543g dry-run reproduces the prior signature exactly (ARM_2=0.444,
  ARM_3=0.243, D2 FAIL). Not an encoder head -> no P0/P1/P2 phased
  training of a latent target; the experiment DOES swap the optimizer
  param-set at Phase 3 (gated_policy.parameters() -> expansion_parameters()
  + diversity params) -- flagged in the queue entry.
  MECH-094: GatedPolicy.forward()'s existing simulation_mode early-return
  precedes the expansion add, so replay/DMN paths never receive the
  expansion bias. Crystallization is a structural weight-state change
  (developmental closure, persists across episodes; reset() does NOT
  un-crystallize), not memory content -> hypothesis_tag N/A.
  Validation experiment: V3-EXQ-543h queued (2x2x2 use_gated_policy x
  use_dacc x crystallize_at_phase3; supersedes V3-EXQ-543g).
  Design doc: REE_assembly/docs/architecture/critical_period_crystallization.md.
  See INV-074, MECH-333, MECH-334, ARC-075, Q-052; arc_062 GAP-B.
