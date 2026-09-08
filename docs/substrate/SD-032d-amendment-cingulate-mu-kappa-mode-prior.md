## SD-032d AMENDMENT: cingulate.mu_kappa_mode_prior_overlays (MECH-048) -- IMPLEMENTED (2026-07-21)
- SD-032d AMENDMENT: cingulate.mu_kappa_mode_prior_overlays (MECH-048) --
  IMPLEMENTED 2026-07-21.
  Module: ree_core/cingulate/salience_coordinator.py (SalienceCoordinatorConfig,
    SalienceCoordinator.tick).
  Config: REEConfig.salience_use_stability_temperature (default False; set True
    to enable). Sub-knobs salience_temperature_mu_alpha (1.0),
    salience_temperature_kappa_alpha (0.0),
    salience_temperature_exponent_clip (4.0).
  Data flow: pcc.tick() -> pcc_stability [0,1] ->
    salience.update_signal("pcc_stability") -> coordinator.tick() ->
    effective_temperature = softmax_temperature *
      exp(alpha_kappa*aic_salience - alpha_mu*pcc_stability)
    -> _softmax(logits, effective_temperature) -> operating_mode
    -> mode entropy + mode_switch_trigger + write_gates.
  Backward compatible: disabled by default; when off, effective_temperature is
    softmax_temperature exactly and exp() is never evaluated, so every tick is
    bit-identical. Existing experiments unaffected. Full contract suite green
    (2019 passed) with the change in place.
  WHY. The 2026-04-19 SD-032d landing gave pcc_stability (the mu-analogue)
    authority over the switch THRESHOLD only; affinity_weights["pcc_stability"]
    was left empty. So H(operating_mode) was EXACTLY invariant under mu --
    0.167605 at pcc_stability 0.0, 1.0 AND 3.0, identical to 6 dp, measured by
    live execution on a constructed agent, while kappa (aic_salience) moved it
    freely via its own affinity weight. MECH-048 asserts the overlays shape
    BOTH mode-prior sharpness (entropy) AND switching inertia, so its entropy
    half was untestable BY CONSTRUCTION on every substrate and every seed: any
    experiment measuring mode entropy against mu would have reported an
    arithmetic zero rather than a measurement. That is the DV-symmetry vacuity
    class (failure_autopsy_V3-EXQ-604c_2026-07-20 section 3) and is why the
    2026-07-21T13:39Z audit HELD V3-EXQ-683 rather than repairing it.
  Form is the claim's own source, not an invention:
    docs/thoughts/2026-02-11_some_control_plane_maths_hypotheses.md:63 gives
    tau = tau_0 * exp(alpha_kappa*kappa - alpha_mu*mu) literally, and :104
    gives the threshold half theta = theta_0 + rho_mu*mu - rho_kappa*kappa,
    which is the pre-existing stability_scaling multiplier (unchanged). So mu
    now reaches both legs, as the claim asserts.
  WHY NOT an affinity_weights["pcc_stability"] entry (the rival mechanism):
    a per-mode logit bias makes mu a mode PREFERENCE, and
    control_plane.md#mech-048 states the overlays "are not scalar reward
    signals; they act as stability and entropy modulators". Ruled out by claim
    text, not preference. The empty dict is now commented as intentional.
  alpha_kappa defaults to 0.0 so that flipping the master switch isolates the
    mu leg (single-lever ablation) -- kappa already reaches the mode logits via
    its affinity weight, so enabling both at once confounds two paths.
  Exponent clipped at +/-4.0 because aic_salience is unbounded above
    (urgency_scaled + extra_sum); keeps the temperature factor in [0.018, 54.6].
    The multiplicative-exponential form also guarantees tau > 0, so the
    coordinator's temperature <= 0 guard is unreachable.
  tick() additionally returns effective_temperature and mode_entropy (nat
    Shannon entropy of operating_mode) so consumers read the MECH-048 DV
    rather than recomputing it inconsistently.
  MECH-094: not applicable (no memory content authored).
  Phased training: not applicable (non-trainable arithmetic, no gradient flow).
  Contracts: tests/contracts/test_mech048_stability_temperature.py (7
    contracts). All assertions are on continuous readouts -- entropy,
    temperature, threshold -- and none on a sampled discrete mode, which is not
    reproducible across machine classes.
  EXPERIMENT-DESIGN CAUTION, read before queuing anything against MECH-048:
    with mu injected DIRECTLY, "entropy falls as mu rises" is an arithmetic
    identity of the coupling. The contracts assert it as substrate wiring; it is
    NOT evidence for MECH-048. A real experiment must drive mu from upstream
    environmental state (safety / coherence / task success, via PCCAnalog) and
    read a downstream behavioural DV, or it reproduces the exact vacuity this
    build exists to remove.
  Validation experiment: NOT yet queued -- chipped as separate
    /queue-experiment work (see WORKSPACE_STATE 2026-07-21).
  STILL OPEN (separate build, deliberately not bundled): MECH-048's
    switching-pressure half stays empirically unmeasurable because the
    contested modes are never occupied -- V3-EXQ-464b/464c/467d/464d ALL FAIL
    with fraction_in_external_task = 0.0 at every seed with
    use_external_task_drive=True across a 20x hysteresis sweep; mean_dwell is
    an episode-length artefact and n_switches == n_episodes. 467d self-routed
    substrate_not_ready_requeue / external_task_mode_not_occupied.
  See MECH-048, SD-032d, SD-032a, SD-032c, MECH-259.
