## ARC-110 x ARC-108: BOUNDED ascending-spiral gain -- target-PARITY controller (V3-EXQ-711 runaway repair) (2026-07-04)
- ARC-110 x ARC-108: selection.cross_loop_ascending_parity_controller -- IMPLEMENTED 2026-07-04
  (substrate; PROMOTES NOTHING -- MECH-439/ARC-108/ARC-110 stay candidate + pending_retest_after_substrate
  until the successor falsifier converts them). ree_core/predictors/e3_selector.py (_parity_forward_gain +
  the W_cross forward assembly + the diagnostic w_eff block + the M_cross post_action_update clamp) +
  ree_core/utils/config.py. Repairs the RUNAWAY the confirmed failure_autopsy_V3-EXQ-711_2026-07-04 found
  in the RAW-scalar ascending gain above: at raw scalar 20x-forward x 5x-plasticity the plastic ascending
  M_cross entries compounded through the positive-feedback plastic loop and RAN AWAY (M_cross range peak
  4897.8 vs the un-gained ~0.02-0.12; w_eff[limbic] peak 10-2274x w_eff[motor] across the 3 divergent
  seeds) -- a limbic-loop MONOPOLY that merely replaces the F/motor-pinning, not a fair parity win, and
  committed-class entropy FELL below baseline on 2/3 divergent seeds. The 709->711 pattern showed the raw
  scalar has NO stable parity regime (sub-threshold 709 never wins; runaway 711 monopolizes): the mechanism
  was MISSING A CONTROLLER. This replaces the unbounded multiply with actuator-saturated setpoint control:
    1. FORWARD parity-ceiling (_parity_forward_gain in the W_cross assembly): a per-step ascending gain in
       [0, parity_forward_gain] SOLVED so the limbic effective column weight w_eff[limbic] is LIFTED toward
       but HARD-CAPPED at parity_ceiling_ratio * w_eff[motor]. The motor column carries no strict-upper-tri
       entry -> w_eff[motor] is gain-invariant -> the fixed parity reference. Bounds the
       w_eff[limbic]/w_eff[motor] RATIO -> a FAIR within-eligible reorder, never a monopoly. Applied via the
       same _ascending_gain_matrix so the map stays LINEAR and bit-identical-at-init (M_cross==0 ->
       W_cross==I for any gain).
    2. MATURATION bounded loop (post_action_update): the ascending three-factor update is scaled by the
       BOUNDED parity_plasticity_gain, then the ascending (upper-tri) M_cross entries are clamped to
       [-m_cross_clamp, m_cross_clamp] -- an anti-windup clamp that stops the plastic positive-feedback loop
       from running away (the second 711 runaway source).
  Config: E3Config.use_ascending_parity_controller (default False, master switch) +
  loop_segregation_parity_forward_gain (default 1.0, lift strength) + loop_segregation_parity_ceiling_ratio
  (default 0.0 = disabled, the w_eff[limbic]<=ratio*w_eff[motor] cap) + loop_segregation_parity_plasticity_gain
  (default 1.0, bounded maturation rate) + loop_segregation_m_cross_clamp (default 0.0 = disabled, the
  ascending |M_cross| bound). Threaded through REEConfig.from_dims. Requires
  use_learned_cross_loop_arbitration (hence use_loop_segregation) on to act. TAKES PRECEDENCE over
  use_ascending_spiral_gain when both are on (the raw path is retained ONLY for 709/711 reproducibility).
  Master switch False -> BIT-IDENTICAL OFF; sub-params default inert (forward_gain 1.0 = no lift,
  ceiling_ratio 0.0 / m_cross_clamp 0.0 = disabled) so the successor's ON arm configures them explicitly.
  Data flow: M_cross -> (forward) g=_parity_forward_gain(M) -> G(g) .* M_cross -> W_cross @
  [motor_z;assoc_z;limbic_z] -> final -> commit; (learning) three-factor delta -> G_plast .* delta ->
  M_cross.add_ -> clamp ascending entries to [-clamp, clamp]. The w_eff / limbic_ge_motor diagnostics use
  the SAME parity-gained W_cross. New diagnostics: loop_ascending_parity_controller_active /
  _parity_forward_gain_applied / _parity_ceiling_ratio (and loop_ascending_spiral_gain_active now reports
  False whenever the controller is active, so a saturation guard can read which path drove selection).
  Safety: unchanged -- arbitration stays STRICTLY within the F+MECH-448/449 eligible set; the controller
  reorders within-eligible candidates and can NEVER re-admit a No-Go-suppressed one. F still fully owns the
  MOTOR loop. MECH-094: the maturation clamp rides the existing waking-only M_cross update (a simulation
  tick forms no delta_t, writes no M_cross); no new encoder / no autograd -> phased training NOT required
  by the substrate (the successor still phases P0/P1/P2 because the cross-loop M_cross LEARNS, same as 711).
  ML/AI note (Layer 7): actuator-saturated setpoint control -- output saturation = the parity ceiling,
  integrator clamp = the M_cross anti-windup clamp; the standard fix for an unbounded gain on a
  positive-feedback plastic loop that diverges. Biologically compatible: Haber's spiral is a graded, BOUNDED
  modulation held in parity by tonic-DA homeostasis + striatal lateral inhibition -- the raw scalar had the
  symbol without that bounding dependency.
  Contract tests: tests/contracts/test_ascending_parity_controller.py (9 contracts: byte-identical OFF
  across 12 seeds; at-init identity under large params; inert ON (fwd 1.0 + ceil 0.0); parity ceiling
  bounds w_eff[limbic]/w_eff[motor] where the raw scalar runs to 144x while motor stays invariant;
  maturation clamp bounds ascending M_cross where it otherwise runs to ~7.7e5; parity win not monopoly --
  limbic_ge_motor True while staying under the ceiling; from_dims plumbing; safety within the eligible set;
  precedence over the raw scalar). Both ascending suites green (18 contracts). Backward-compat: the raw
  test_ascending_spiral_gain.py 8 contracts still pass bit-for-bit (raw path untouched).
  Validation experiment: V3-EXQ-713 queued (Step 8) -- a NEW-EXQ successor falsifier (OFF vs bounded-ON on
  the GAP-A reef-bipartite substrate; a saturation-guarded limbic_loop_can_win gate requiring a parity BAND
  win + a w_eff/M_cross ceiling; C1 committed-class entropy strict-above the un-gained baseline on >=2/3
  divergent seeds; non-vacuity self-route substrate_not_ready_requeue, never a false weakens). This is a
  redesign of a DIFFERENT mechanism (the controller), NOT a raw-gain-magnitude re-letter -- the
  re-derive brake in the 711 autopsy explicitly REFUSES a same-claim raw-gain re-queue.
  Biology + ARC-106 divergence ledger: REE_assembly/docs/architecture/learned_cross_loop_arbitration.md
  (Addendum: ascending-spiral gain -> Sub-addendum: bounded parity controller). See ARC-110 / ARC-108
  (owning coupling), MECH-439 (the F-dominance conversion ceiling), use_ascending_spiral_gain (the raw path
  this bounds), use_learned_cross_loop_arbitration (the matrix this gains), ARC-106 (grounding framework).
