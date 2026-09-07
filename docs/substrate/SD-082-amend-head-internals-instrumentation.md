## SD-082 AMEND: head-internals instrumentation (dead-ReLU / magnitude-ratio diagnostics for the still-zero V3-EXQ-822a propagation) (2026-07-27)
- SD-082 head-internals instrumentation amend -- IMPLEMENTED 2026-07-27. Routed by the
  confirmed failure_autopsy_batch-822a-826-817a-827_2026-07-26 (Section 2, "Routing --
  implement-substrate, amend SD-082"): V3-EXQ-822a re-ran V3-EXQ-822 with SD-082 correctly
  engaged (lateral_pfc_rule_readout_consumer=True, lateral_pfc_train_rule_bias_head=True)
  and a strong, non-degenerate upstream chain (on_rule_state_diff_mean 0.644), but
  readiness gate (d) propagation_non_vacuity STILL failed at exactly 0.0 for both arms --
  the SECOND consecutive identical failure of the same gate (V3-EXQ-822 failed it before
  SD-082 existed).
  ROOT CAUSE (unverified -- this IS the recording gap the amend closes): SD-082's fix
  (compute_bias, ree_core/pfc/lateral_pfc_analog.py) touches (i) centering
  candidate_world_summaries and (ii) the scaled-tanh output bound + readout_init_scale
  rescale of the rule_bias_head's LAST Linear at init. But rule_bias_head itself is
  Linear(rule_dim+world_dim, hidden_dim) -> ReLU -> Linear(hidden_dim, 1) -- the FIRST
  Linear+ReLU layer is untouched by SD-082 and always uses standard random init. Leading
  hypothesis: a dead-ReLU / magnitude-insensitivity point at that first layer absorbs the
  rule_state signal before it reaches the instrumented last layer, plausible if
  rule_state's raw magnitude is small relative to the (now-centered) world-summary
  component in the concatenated joined = cat([rule_repeated, summaries]) input -- but the
  V3-EXQ-822a manifest carried NO head-internals diagnostics to confirm or rule this out.
  THE FIX (no-op default; bit-identical OFF; ON changes zero computed bias values --
  diagnostic-only): LateralPFCConfig.capture_head_diagnostics (default False) +
  REEConfig.lateral_pfc_capture_head_diagnostics, plumbed through all three from_dims
  sites (dataclass field, signature, assignment) and agent.py's LateralPFCConfig build
  (getattr-fallback) -- same three-site pattern as rule_readout_consumer, guarding the
  documented silently-unreachable-flag hazard.
    ree_core/pfc/lateral_pfc_analog.py -- compute_bias(), when capture_head_diagnostics is
      True, unrolls rule_bias_head's Sequential layer-by-layer (Linear -> ReLU -> Linear)
      instead of a single self.rule_bias_head(joined) call -- the IDENTICAL sequence of
      ops nn.Sequential.__call__ performs internally, so bias_raw is bit-identical to the
      OFF path at every tick. From the captured hidden-layer activation it records (a)
      hidden_dead_relu_frac -- fraction of (unit, candidate) activations <= 0 after the
      first ReLU, and (b) rule_summary_magnitude_ratio -- mean(||rule_repeated||) /
      mean(||summaries||) over the (possibly SD-082-centered) concatenated input. Both
      read via get_state(); both default 0.0 and are only populated when the flag is on.
    get_state() ALSO gains four weight-norm fields (first/last Linear weight+bias norm)
      reported UNCONDITIONALLY -- these are cheap parameter reads, not per-tick
      activations, so an experiment script calling get_state() at each P0/P1/P2 phase
      boundary gets per-phase weight-norm tracking for free, with no flag needed.
    reset() (episode boundary) clears the two capture-gated diagnostic fields alongside
      the existing _last_* fields.
  Backward compatible: capture_head_diagnostics default False = zero extra compute in the
    hot path (single self.rule_bias_head(joined) call, unchanged); weight-norm fields are
    always present but were previously absent from get_state() entirely (additive, no
    existing key removed or renamed). 8 new contracts added to
    tests/contracts/test_sd082_rule_readout_consumer.py (14/14 total pass, run via
    remote_pytest.sh on the hub): flag reachability through from_dims (+absent-default),
    diagnostics-off leaves state fields at 0.0/False, diagnostics-on produces a
    BIT-IDENTICAL bias and an IDENTICAL head gradient vs diagnostics-off (the load-bearing
    no-op-instrumentation guarantee), dead-ReLU fraction and magnitude-ratio each match an
    independent manual recompute, weight-norm fields present regardless of the capture
    flag, reset() clears the new fields.
  Phased training: N/A (pure diagnostic read of existing forward-pass tensors and static
    parameter norms; no new learned parameters, no change to any training target).
    MECH-094: N/A -- read-only instrumentation on the existing waking select_action
    forward path; no memory write surface.
  GOVERNANCE: PROMOTES NOTHING, DEMOTES NOTHING. SD-082 stays
    implemented_pending_validation / ready=false. claims.yaml untouched (substrate-only
    amend; the amend record lands in the SD-082 substrate_queue.json entry).
