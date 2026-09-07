## MECH-140 x MECH-450: disinhibitory soft-competitive settling (parameter-free) (2026-07-02)
- MECH-140 x MECH-450: selection.disinhibitory_soft_competitive_settling -- IMPLEMENTED
  2026-07-02 (substrate; MECH-140 + MECH-450 stay candidate -- this PROMOTES NOTHING, they
  stay candidate until a falsifier converts them). ree_core/predictors/e3_selector.py
  (_soft_competitive_settle) + ree_core/utils/config.py. The PARAMETER-FREE, always-graded
  complement to the LEARNED W_lat settling (use_learned_settling_step, a no-op at init): a
  few rounds of soft-competitive lateral inhibition over the F + MECH-448/449 within-eligible
  field BEFORE the commit, so the committed action emerges from a bounded recurrent SETTLING
  competition rather than a one-shot argmin (MECH-450), with losing options down-weighted
  GRADED-ly but never silenced (MECH-140 -- soft-competitive disinhibition, not
  winner-take-all). Unlike W_lat it bites IMMEDIATELY (no dopaminergic learning) so it can
  flip the committed attractor at init -- the "recurrence gives the readout an attractor-flip
  the argmin lacks" property MECH-450 asserts.
  Dynamics (COST units, lower=better): x=-field (activation); for r in R: support=softmax(x/T)
  (graded, all > 0 -> never silenced); inhib_i = gain * sum_j K_ij*support_j; x -= inhib_i;
  return -x. K is the PARAMETER-FREE class-surround kernel (1.0 within a first-action class,
  cross_class < 1 across classes, 0 diagonal -- surround inhibition between competing motor
  programs, Mink 1996; the SAME structure W_lat learns, here FIXED). The STRUCTURED kernel is
  what lets settling REORDER (a candidate crowded by same-class rivals loses to an isolated
  slightly-worse one) -- the behavioural non-vacuity the MECH-439 ceiling needs, not a
  rank-preserving sharpen.
  Config: E3Config.use_soft_competitive_settling (default False -> bit-identical OFF) +
  soft_competitive_settling_gain (default 0.0 -> EXACT no-op even when the flag is on;
  byte-identical at flag-off / at-default, mirroring noisy_selection_sigma_init=0.0 -- set a
  positive gain to activate) + soft_competitive_settling_rounds (3) / _temperature (1.0) /
  _cross_class (0.25). Threaded through REEConfig.from_dims.
  Data flow: _modulatory_accum[eligible_idx] (single arena) OR the arbitrated cross-loop
  `final` (ARC-110 segregation) -> _soft_competitive_settle -> settled field -> committed
  argmin. COMPOSES with the learned W_lat within-loop settling and the learned cross-loop
  arbitration (M_cross / use_learned_cross_loop_arbitration, V3-EXQ-709) -- it is a
  within-eligible / within-loop transform at a DIFFERENT level, orthogonal flag, default-OFF
  -> no collision with 709.
  Safety: transforms ONLY the eligible subset -> a No-Go/F-excluded candidate is never touched
  and never selectable (no global disinhibition); >= 1 survivor always; needs >= 2 eligible.
  MECH-094: waking-only (no settling on a simulation/replay tick). No learned parameters, no
  autograd (detached), so phased training NOT required. ARC-106 divergence (logged): the kernel
  is a FIXED class-surround, biology's is learned + graded-by-similarity (that is W_lat's job) --
  this is the always-on floor beneath the learned inhibition.
  Contract tests: tests/contracts/test_soft_competitive_settling.py (byte-identical OFF/gain-0
  across 12 seeds; non-vacuity flips the readout + round_delta>0; graded-not-WTA loser stays
  nonzero; safety harmful-outlier never selected; MECH-094 sim-gate; segregated-path compose).
  Full ree-v3 suite green (1312 contracts pass + 11 new; the 3 residual failures --
  control_vector C4 flake + 2 runner-fail-branch -- are pre-existing, clean-tree stash-confirmed).
  Validation experiment: <EXQ pending Step 8 -- a validation falsifier (settling ON vs OFF on the
  569i top-k + MECH-448 demotion conversion-ceiling stack) AND the PLOS-Biology ablation falsifier
  (lateral-inhibition edge intact vs ablated; pre-registered: ablation collapses task-context/
  switching while single-task performance survives -- the mouse-V1-silencing signature)>.
  Biology + ARC-106 grounding ladder, load-bearing-vs-decorative ablation test, divergence ledger,
  and the required psychiatric_failure_mode column:
  REE_assembly/docs/architecture/soft_competitive_disinhibition_settling.md. Lit basis (2026-07-02
  /lit-pull): Gallo Aquino/Rungratsameetaweemana PLOS Biology 2026 (inhibition-on-inhibition = the
  top-down context channel; ablating it collapses task-switching) + Keller et al Neuron 2020
  (VIP->SOM disinhibition necessary+sufficient for contextual modulation); Lee & Sabatini Nature
  2021 (indirect-pathway competitive disinhibition, not hard suppression) + Morita 2016 (striatal
  WTA vs cortical soft-max) for MECH-140; Wang 2002 / Rolls 2021 recurrent-attractor for MECH-450.
  See MECH-140 (soft-competitive disinhibition), MECH-450 (minimal recurrent settling),
  use_learned_settling_step (W_lat, the learned counterpart), ARC-110 / use_learned_cross_loop_
  arbitration (the loops it composes with), MECH-448 / MECH-449 (the eligible set it runs within),
  MECH-439 (the F-dominance conversion ceiling), ARC-106 (grounding framework).
