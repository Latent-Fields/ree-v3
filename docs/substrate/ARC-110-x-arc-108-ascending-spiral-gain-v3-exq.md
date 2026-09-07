## ARC-110 x ARC-108: ascending-spiral gain (V3-EXQ-709/710 loop-effective-weight repair) (2026-07-03)
- ARC-110 x ARC-108: selection.cross_loop_ascending_spiral_gain -- IMPLEMENTED 2026-07-03
  (substrate; PROMOTES NOTHING -- MECH-439/ARC-108/ARC-110/MECH-450/MECH-140 stay candidate until the
  validation falsifier converts them). ree_core/predictors/e3_selector.py (_ascending_gain_matrix +
  the W_cross forward assembly + the M_cross post_action_update) + ree_core/utils/config.py.
  Repairs the loop-effective-weight ceiling BOTH the V3-EXQ-709 AND V3-EXQ-710 confirmed autopsies
  routed to as the load-bearing V3-closure build: the learned cross-loop matrix M_cross ENGAGES
  (709: 6/7 readiness gates, M_cross range 0.116, limbic routing 1.414) yet the ascending path
  M_cross[motor,limbic] peaks ~0.03 -- functionally too weak to lift a non-motor (limbic) loop to the
  motor loop's effective column weight w_eff[j]=sum_i gain_i*W_cross[i,j], so limbic_loop_can_win was
  met on only 1/4 divergent seeds (thr 2). Biology (Haber 2000): the striato-nigro-striatal spiral is
  anatomically ASYMMETRIC -- ascending (limbic -> associative -> motor) influence is the
  developmentally-strengthened, load-bearing direction. In the motor(0)/assoc(1)/limbic(2) ordering
  the forward map eff_i=sum_j W_cross[i,j]*z_j makes the ascending entries exactly the strict upper
  triangle (row<col): W_cross[0,2] (limbic->motor), W_cross[0,1] (assoc->motor), W_cross[1,2]
  (limbic->assoc). Two knobs scale ONLY those entries:
    1. forward gain (_ascending_gain_matrix in W_cross): W_cross = I + (G_fwd .* M_cross), G_fwd
       upper-tri = spiral_gain else 1.0 -- the ANATOMICAL ascending-projection strength. Raises
       w_eff[limbic]/w_eff[assoc] WITHOUT touching w_eff[motor] (motor column is diagonal + descending,
       un-amplified) -> strengthens the ascending coupling AND implicitly DE-PINS the motor(F) default.
       Keeps the map LINEAR (constant elementwise scaling of M_cross) -> w_eff-collapsibility (CLA-3)
       and bit-identical-at-init both hold (at init M_cross==0 -> gain*0==0 -> W_cross==I for any gain).
    2. plasticity maturation gain (post_action_update): the ascending entries of the M_cross
       three-factor UPDATE are scaled by plasticity_gain -- the ascending SPIRAL-MATURATION RATE
       (ascending credit accrues faster than descending). eta stays the base rate.
  Config: E3Config.use_ascending_spiral_gain (default False, master) +
  loop_segregation_ascending_spiral_gain (default 1.0, forward) +
  loop_segregation_ascending_plasticity_gain (default 1.0, maturation). Threaded through
  REEConfig.from_dims. Requires use_learned_cross_loop_arbitration (hence use_loop_segregation) on to
  act. Default False / gains 1.0 -> BIT-IDENTICAL OFF (G matrices become all-ones).
  Data flow: M_cross -> (forward) G_fwd .* M_cross -> W_cross @ [motor_z; assoc_z; limbic_z] -> final
  -> commit; (learning) three-factor delta -> G_plast .* delta (ascending entries) -> M_cross.add_.
  The w_eff / limbic_ge_motor diagnostics use the SAME gained W_cross (so limbic_loop_can_win reads
  true effective weights); clg_limbic_to_motor stays the RAW M_cross[0,2] (measures learning). New
  diagnostics: loop_ascending_spiral_gain_active / _forward. eta + P2 length remain independently
  sweepable complementary levers (already exposed).
  Safety: unchanged -- arbitration stays STRICTLY within the F+MECH-448/449 eligible set; the gain
  reorders within-eligible candidates and can NEVER re-admit a No-Go-suppressed one. F still fully owns
  the MOTOR loop; the gain only stops F from drowning the limbic value. MECH-094: the maturation gain
  rides the existing waking-only M_cross update (a simulation tick forms no delta_t, writes no M_cross);
  no new encoder / no autograd -> phased training NOT required.
  Contract tests: tests/contracts/test_ascending_spiral_gain.py (8 contracts: byte-identical OFF across
  12 seeds; byte-identical at gain==1.0; at-init identity under a large gain; asymmetry -- motor
  effective weight invariant while limbic/assoc rise; mechanism -- a small learned ascending M_cross
  flips the commit off the motor F-winner under gain + limbic_ge_motor crosses True; plasticity
  maturation scales ascending entries ~Nx while descending/diagonal stay bit-identical; from_dims
  plumbing; safety within the eligible set). Full ree-v3 suite green (1320 contracts + 8 new; the 3
  residual failures -- control_vector C4 + 2 runner-fail-branch -- are pre-existing, clean-tree
  stash-confirmed). Backward-compat: V3-EXQ-709 --dry-run reproduces the 709 FAIL signature
  bit-for-bit (preconditions_met=False, limbic_can_win=False, m_range_peak=0.0497, C1 static==learned).
  Validation experiment: V3-EXQ-711 queued (Step 8) -- a NEW-EXQ falsifier re-running the 709/710-style
  STATIC/OFF vs ascending-gain-ON arms on the GAP-A reef-bipartite foraging substrate, matched seeds,
  same non-vacuity self-route (substrate_not_ready_requeue when limbic_loop_can_win still unmet; never
  a false weakens).
  Biology + ARC-106 divergence ledger: REE_assembly/docs/architecture/learned_cross_loop_arbitration.md
  (Addendum: ascending-spiral gain). See ARC-110 / ARC-108 (owning coupling), MECH-439 (the
  F-dominance conversion ceiling), MECH-450 (settling), MECH-140 (disinhibition -- also unblocked),
  use_learned_cross_loop_arbitration (the matrix this gains), ARC-106 (grounding framework).
