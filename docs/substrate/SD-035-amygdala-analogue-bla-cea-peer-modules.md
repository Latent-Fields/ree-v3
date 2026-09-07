## SD-035: Amygdala Analogue -- BLA + CeA Peer Modules (2026-04-21)
- SD-035: amygdala.analog_bla_cea_peers -- IMPLEMENTED 2026-04-21.
  Modules:
    ree_core/amygdala/bla.py (BLAAnalog, BLAConfig, BLAOutput)
    ree_core/amygdala/cea.py (CeAAnalog, CeAConfig, CeAOutput)
  Two peer non-trainable arithmetic substrates mirroring the biological
  BLA / CeA division. Both read z_harm_a (SD-011 affective stream) and
  write to different downstream consumers. Master switch
  use_amygdala_analog gates both; per-module switches use_bla_analog and
  use_cea_analog give granular control.

  BLAAnalog (basolateral-analog, slow/content) -- MECH-074a/b/d:
    MECH-074a encoding_gain: inverted-U arousal-dependent multiplier on
      HippocampalModule write strength (Roozendaal 2011). Threshold
      on_arousal=0.4, peak=0.7, max gain=2.5, window=18000 steps,
      half-life=3600 steps. Zero below threshold; falls back to 1.0 in
      the tail of the post-encoding window.
    MECH-074b retrieval_bias: content-selective per-trace weight
      vector w_i = 1 + alpha * arousal_tag_i (NOT a scalar; LaBar &
      Cabeza 2006). Requires arousal_tags_in_context from caller
      (hippocampal retrieval side). None-passthrough when no context
      provided.
    MECH-074d remap_signal: Moita 2004 attribution-gated per-code
      remap. Fires when PE z-score > remap_pe_sigma_threshold AND
      candidate_code_contributions attribution dict is supplied.
      Output is {code_idx: 1.0} over the top remap_code_fraction of
      attribution candidates (default 33%).
  BLA outputs are cached on agent._bla_last_output. V3 hippocampal
  consumer wiring (write-gain multiplication, retrieval reweighting,
  remap handoff) is deferred -- the module emits the signals but the
  HippocampalModule does not yet read them. First-pass consumer wiring
  is gated on EXQ-B acceptance.

  CeAAnalog (central-analog, fast/scalar) -- MECH-046/074c:
    MECH-046 mode_prior: pre-softmax additive log-odds bias written
      to SalienceCoordinator.affinity_weights. Fires within 1-2 sim
      steps (~75 ms biological; Mendez-Bertolo 2016) when
      |LowFreq(z_harm_a)| > fast_route_threshold. Distinct from AIC
      urgency (SD-032c): AIC modulates mode-SWITCH threshold; CeA
      mode_prior biases mode SELECTION.
    MECH-074c fast_prime: scalar candidate-prior pulse distinct from
      mode_prior (Pessoa & Adolphs 2010 many-roads framing). Override
      window 5-10 steps; cortical_confirmation signal holds the pulse
      across the window or accelerates decay (tau=4 steps base).
    Q-036 escapability_hint: placeholder input (no-op pass-through)
      so MECH-219 escapability wiring can land without an interface
      refactor.
  CeA outputs are cached on agent._cea_last_output and injected into
  SalienceCoordinator via update_signal calls in select_action() BEFORE
  coordinator.tick() each cycle:
    update_signal("cea_mode_prior", mode_prior_float)
    update_signal("cea_fast_prime", fast_prime_float)
  Signal slots registered at agent __init__:
    affinity_weights["cea_mode_prior"] = {"external_task": 1.0}
    salience_weights["cea_fast_prime"] = 0.5

  Config: REEConfig.use_amygdala_analog (bool, default False).
  Sub-switches: use_bla_analog (bool, default True),
  use_cea_analog (bool, default True) -- only take effect when master
  is True. 14 BLA flat params (bla_encoding_gain_max, bla_encoding_gain_floor,
  bla_arousal_threshold_on, bla_arousal_peak, bla_window_steps,
  bla_window_half_life_steps, bla_retrieval_bias_alpha,
  bla_retrieval_bias_compensation, bla_retrieval_tag_at_encoding,
  bla_remap_pe_sigma_threshold, bla_remap_pe_ema_alpha,
  bla_remap_pe_std_init, bla_remap_code_fraction,
  bla_remap_requires_attribution) and 9 CeA flat params
  (cea_fast_route_threshold, cea_fast_route_input_is_lowfreq,
  cea_mode_prior_log_odds_max, cea_mode_prior_gain,
  cea_pre_softmax_additive, cea_fast_prime_amplitude,
  cea_fast_prime_decay_tau_steps, cea_fast_prime_override_window_steps,
  cea_cortical_confirmation_weight). All wired through
  REEConfig.from_dims() with synthesis-seeded defaults (see
  REE_assembly/evidence/literature/targeted_review_amygdala_analog/
  synthesis.md).

  Data flow:
    sense() -> LatentStack.encode() -> z_harm_a (SD-011, requires
      use_affective_harm_stream=True) -> bla.tick(z_harm_a, z_harm_a_pred)
      AND cea.tick(z_harm_a) -> cache outputs ->
    select_action() -> coordinator update_signal("cea_mode_prior", ...)
      AND update_signal("cea_fast_prime", ...) -> coordinator.tick() ->
      mode affinity and salience aggregate reflect the fast route.
  BLA retrieval_bias / remap_signal hippocampal wiring: DEFERRED.
  Outputs produced and cached; HippocampalModule consumer wiring lands
  when EXQ-B passes and the retrieval-bias-aware replay path is added.

  Backward compatible: use_amygdala_analog=False by default; both
  modules are None; all integration sites are no-ops. 33/33 preflight +
  contract tests PASS with flag OFF (2026-04-21). Bit-identical to
  baseline.

  Activation smoke (2026-04-21, flag ON):
    CeA mode_prior: 0.0 at rest -> 0.3 under synthetic threat
      (L1/dim=0.8; threshold 0.5; cap mode_prior_log_odds_max=0.8 *
      gain=0.5).
    CeA fast_prime: 0.0 at rest -> 0.225 under threat.
    BLA encoding_gain: 1.0 at rest -> 2.5 under synthetic arousal
      (inverted-U cap).
    BLA remap_signal: fires on synthetic PE spike when attribution
      candidates supplied (Moita 2004 gate).
    All three activation signatures confirmed; agent.sense() one-tick
    boot completes without error with amygdala ON.

  Biological basis (see synthesis.md):
    BLA encoding: McGaugh 2004, Roozendaal 2011 (arousal-dependent LTP
      modulation of hippocampal consolidation).
    BLA retrieval: LaBar & Cabeza 2006, Dolcos et al 2005, Phelps 2004
      (content-selective per-trace bias, not global arousal gain).
    BLA remap: Moita 2004, Nader 2000, Schiller 2010 (PE-spike
      remapping on violated expectation; attribution-gated).
    CeA mode_prior: LeDoux 1996 "low road", Pessoa & Adolphs 2010
      (fast subcortical route, mode SELECTION bias distinct from
      cortical mode-SWITCH threshold).
    CeA fast_prime: Mendez-Bertolo 2016 (~75 ms fast visual-amygdalar
      pulvinar route), Pessoa & Adolphs 2010 (cortical confirmation
      window).
  MECH-094: CeAAnalog.tick() accepts simulation_mode argument;
    returns zeroed output without updating state when True.
    BLAAnalog defers to caller for simulation gating (encoding-gain
    writes are gated at the HippocampalModule consumer side via
    MECH-261).
  Phased training: not applicable (non-trainable arithmetic).
  Design doc: REE_assembly/docs/architecture/sd_035_amygdala_analog.md
  Literature synthesis: REE_assembly/evidence/literature/
    targeted_review_amygdala_analog/synthesis.md
  Validation experiments: V3-EXQ-A (CeA mode-prior ablation, MECH-046)
    and V3-EXQ-B (BLA encoding + remap, MECH-074a/d) -- queued in a
    follow-up pass.
  See SD-035, MECH-046, MECH-074, MECH-074a, MECH-074b, MECH-074c,
  MECH-074d, SD-011, SD-032a, SD-032c, Q-036.

- SD-035 second pass / MECH-074d: amygdala.bla_trainable_attribution_head
  -- IMPLEMENTED 2026-08-09.
  Module: ree_core/amygdala/attribution_head.py (BLAAttributionHead,
  BLAAttributionHeadConfig). The learnable attribution head MECH-074d's
  own 2026-04-21 registration text deferred as "a deliberate second pass".
  Config: REEConfig.bla_attribution_head (default
  "contribution_threshold" = the legacy fixed non-trainable proxy, so the
  head is NOT constructed and behaviour is bit-identical to
  pre-2026-08-09; set "trainable" to enable). Twelve tuning params
  bla_attr_head_* (key_dim 16, lr 1e-3, temperature_init 0.5,
  entropy_weight 0.02, warmup_steps 200, train True, ...).
  Data flow: cat(z_self, z_world) + ContextMemory.memory (BOTH DETACHED)
  -> cosine query/key attribution a = softmax(q.k / tau) -> BLAAnalog
  candidate_code_contributions -> unchanged PE-sigma gate + Moita-2004
  top-k -> _apply_bla_context_remap. Supervised by
  MSE(W_p (sum_i a_i m_i), z_harm_a) / Var_ema(z_harm_a)
  + entropy_weight * H_norm(a), stepped in agent.sense() AFTER the gate
  consumed this tick's attribution.
  Single entry point on purpose: both implementations dispatch through
  REEAgent._get_context_memory_code_contributions, which is what the
  V3-EXQ-894/894a instrument wraps to record the attribution the gate
  actually saw.
  Backward compatible: disabled by default; existing experiments
  unaffected. Below warmup_steps the head returns None and the caller
  falls back to the legacy proxy, so a random-init head never drives real
  ContextMemory writes.
  Motivation: V3-EXQ-894 + V3-EXQ-894a (both FAIL/weakens) showed the
  fixed rule's context-selectivity does not respond to PE-threshold
  recalibration -- mass-excess falls monotonically with sigma
  (Spearman -1.0) across a 4-point sweep, refuting the over-firing /
  dilution alternative. Autopsy: competence_implementation_gap
  (failure_autopsy_V3-EXQ-894a_2026-08-08, confirmed).
  Biological basis: Moita et al 2004 -- the contextual-vs-auditory
  remapping dissociation DEVELOPS OVER TRAINING, i.e. real BLA
  attribution is a learned cue-outcome association. That is the property
  a fixed threshold rule structurally cannot have.
  Four measured design constraints, all recorded in the module docstring
  (do not "simplify" any of them):
    (a) the predictor sees ONLY the attended read, never the raw state
        (attention-shortcut failure);
    (b) W_p is LINEAR, so z_hat = sum_i a_i (W_p m_i) and a_i is
        interpretable as slot i's own harm prediction being trusted;
    (c) bias=False throughout + COSINE scoring -- with a raw scaled dot
        product over ContextMemory's 0.01-scale slot init, attention is
        EXACTLY uniform (measured: max weight 0.0625 = 1/16, unmoved
        after 259 steps), the SD-016 Part A pathology by a second route;
    (d) the MSE is divided by a running target variance -- z_harm_a is
        small enough that a raw MSE (1.9e-4) was ~100x below the entropy
        term (0.02), silently making the sparsity penalty the objective.
  Known instrument precondition, NOT a property of the head: under the
  legacy ContextMemory write path the slot bank homogenises to
  off-diagonal cosine 1.0000 within ~24 episodes (the V3-EXQ-436c
  sigmoid-midpoint payload collapse). No attribution rule can
  differentiate identical slots. Validation MUST restore a differentiated
  store per episode (the 894a harness already does, via restore_base) or
  set contextmemory_gated_content_write=True.
  MECH-094: not applicable -- the head never trains or attributes on
  simulation/replay ticks; weights are waking-stream only.
  Phased training: YES for the head, and it is enforced structurally --
  every input is detached (no gradient reaches E1 / ContextMemory / the
  latent stack) and training is skipped whenever the agent is in eval()
  mode, so a frozen-head P2 measures frozen weights.
  Design doc: REE_assembly/docs/architecture/sd_035_amygdala_analog.md
    ("Second pass: trainable attribution head").
  Validation experiment: V3-EXQ-894b (queued 2026-08-09) -- same C1/C2
    instrument as 894/894a; acceptance attribution_mass_excess > 0.05 AND
    context_jaccard_gap > 0.05 on >= 2/3 seeds at some operating point.
  See MECH-074d, SD-035, SD-016, MECH-073, ARC-033.
