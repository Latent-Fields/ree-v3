## ARC-108 x ARC-110 coupling: LEARNED (dopamine-gated) CROSS-LOOP arbitration -- the named next attack on the F-dominance conversion ceiling (MECH-439) after V3-EXQ-707b (2026-07-01)
- selection.cross_loop_arbitration_plasticity -- IMPLEMENTED 2026-07-01 (substrate; ARC-108 + ARC-110
  stay candidate / substrate_conditional / implementation_phase=v3 -- PROMOTES NOTHING; the validation
  falsifier is queued SEPARATELY, new EXQ / different claim_ids, not yet scored). This is the ARC-108 x
  ARC-110 INTERSECTION the 2026-06-29 MECH-439 /claim-synthesis named (borderline child MECH-453,
  recommended DROP -- ARC-108 already depends_on ARC-110), so NO new claim is minted; the coupling is
  annotated onto ARC-108 + ARC-110. Escalation source: REE_assembly/evidence/planning/
  failure_autopsy_V3-EXQ-707b_2026-06-29.{md,json} (707b built ARC-110 loop segregation FULLY LIVE -- all
  6 non-degeneracy gates passed, limbic range 1.414 -- yet the limbic loop NEVER won: A1_LOOPS 0.838 ~
  A0_SINGLE_ARENA 0.914; the autopsy traced this to the STATIC ARITHMETIC cross-loop combine, which
  inherits F's dominance because it cannot LEARN to down-weight F). Design-of-record:
  REE_assembly/docs/architecture/learned_cross_loop_arbitration.md.
  THE BUILD (strict additive extension behind a no-op-default flag; every existing path -- the static
  spiral combine, ARC-108 w_chan, MECH-450 W_lat, MECH-451 finer -- is BYTE-IDENTICAL OFF):
    LEARNED [3,3] CROSS-LOOP MATRIX (ree_core/predictors/e3_selector.py _segregated_loop_arbitrate,
    gated on config.e3.use_learned_cross_loop_arbitration). The fixed cross-loop combine
    `final = m_a*motor_z + g_a*assoc_z + g_l*limbic_z` is replaced by a learned matrix
    W_cross = I + M_cross (M_cross a [3,3] register_buffer, loop order motor/associative/limbic =
    _LOOP_NAMES, init 0): eff = W_cross @ [motor_z; assoc_z; limbic_z]; final = m_a*eff_motor +
    g_a*eff_assoc + g_l*eff_limbic. At init M_cross==0 -> W_cross==I -> eff==the per-loop zscores -> final
    is BIT-IDENTICAL to the static combine (and OFF -> the static combine runs untouched). M_cross[i,j] is
    the learned directed influence of loop j on loop i's effective preference; M_cross[motor,limbic] is the
    learnable ascending-spiral path by which the limbic value loop learns to drive the motor commit.
    LEARNING: the SAME ARC-108 signed-RPE three-factor rule as w_chan/W_lat (one shared dopaminergic
    delta_t = R_t - V-hat_t + D1-D2 asym -- Haber's single ascending spiral), via an outer-product Hebbian
    co-activation trace coact[i,j] = post_i * pre_j where pre_j = -loop_z_j[committed] (SIGNED: >0 when
    loop j preferred the committed candidate -- directional cross-loop credit) and post_i =
    -eff_i[committed]; Delta M_cross = eta_c * delta_t * asym * coact_trace (learned_cross_loop_eta 0.01).
    Waking-only (a replay/DMN tick forms no delta_t and writes no M_cross; MECH-094). M_cross is SIGNED
    (mirrors the signed W_lat). NO autograd (register_buffer + no_grad LOCAL update). Per-episode clear of
    the coact trace + pending flag (clear_learned_channel_eligibility); learned M_cross persists.
  Config: E3Config.use_learned_cross_loop_arbitration (False, master) + learned_cross_loop_eta (0.01);
    elig_decay / asym / rpe_mode / value_baseline_beta are SHARED with the ARC-108 learned_channel_* knobs
    (one dopamine system). Threaded through REEConfig.from_dims. Requires use_loop_segregation on to act.
  Coupling with MECH-450: the per-loop settling (W_lat) shapes each loop's WITHIN-loop competition BEFORE
    normalise/arbitrate; the learned cross-loop weights arbitrate ACROSS the settled loops; both ride the
    same shared delta_t in one post_action_update (the "dopamine-into-gating + recurrent-settling, coupled"
    the BG-assembly map names).
  Safety: arbitration runs STRICTLY within the F + MECH-448/449 Go/No-Go eligible set, so a learned weight
    can reorder within-eligible candidates but can NEVER re-admit a suppressed one (inherited, weights
    irrelevant to admission).
  Backward compatible: disabled by default; existing experiments unaffected (verified: full suite green;
    byte-identical-at-init contract test asserts ON-at-init == static combine across 12 seeds).
  Biological basis: functional translation of dopamine-gated striatal plasticity operating on the
    Alexander/DeLong/Strick segregated loops via Haber's ascending dopamine spiral (ARC-106 L1-L2; NOT
    anatomical mimicry -- 4 divergence-ledger rows incl. the honest forward-linearity note; the [3,3]'s
    value is the DIRECTED credit structure it learns). Psychiatric-failure-mode column (ARC-106 EARNS):
    motor-dominant-unadaptable = avolition; runaway limbic->motor = OCD-like over-valuation; dead delta_t =
    apathy/inflexibility (loop-specific CSTC axis with a PLASTIC arbitration knob).
  Phased training required: no (reuses the already-trained valuation heads; the matrix rides the LOCAL
    three-factor update, not autograd). MECH-094: waking-only; selection-only (writes nothing to memory).
  New diagnostics: loop_learned_cross_loop_active, loop_cross_loop_w_motor_eff / _assoc_eff / _limbic_eff
    (effective column weights w_eff[j] = sum_i gain_i*W_cross[i,j] -- what the linear forward commit
    depends on; limbic learning to win == w_eff[limbic] rising toward/above w_eff[motor]),
    loop_cross_loop_limbic_ge_motor, loop_cross_loop_m_range (non-vacuity: >0 == weights moved off init),
    loop_cross_loop_limbic_to_motor (M_cross[motor,limbic]); post_action_update metrics clg_delta_t /
    clg_m_cross_range / clg_limbic_to_motor / clg_n_updates.
  Regression guard: tests/contracts/test_learned_cross_loop_arbitration.py (byte-identical-OFF/at-init;
    non-vacuity M_cross moves; MECH-094 waking gate; limbic-can-win mechanism; from_dims plumbing;
    within-eligible-set safety).
  Validation experiment: SEPARATE new-EXQ falsifier (A1_LOOPS + learned DA-gated cross-loop arbitration
    STRICT-ABOVE A1_LOOPS + static-arithmetic arbitration; different claim_ids), queued via
    /queue-experiment AFTER this build lands + tests pass (per 707b routing step 3). EXQ TBD.
  See ARC-108 / ARC-110 (annotated -- the coupling), MECH-439 (conversion-ceiling umbrella under test),
    MECH-448 / MECH-449 / ARC-107 (F-bounded eligible set; safety), MECH-450 (settling, coupled), MECH-452
    (loop-local traces), ARC-109 (D1/D2), ARC-106 (grounding), V3-EXQ-707b (escalation source),
    REE_assembly/evidence/planning/basal_ganglia_assembly_map_2026-06-22.md (the named next attack).
