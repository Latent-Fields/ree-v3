## SD-057 L7 AMEND: dacc_goal_readout calibration fix (arc005_dacc_adapter_goal_proximity_training, IGW-20260801-199) (2026-08-01)
- SD-057 L7 amend: incentive.dacc_object_discriminative_readout calibration --
  IMPLEMENTED 2026-08-01. Substrate response to substrate_queue.json's
  `arc005_dacc_adapter_goal_proximity_training` entry (created from the
  failure_autopsy_V3-EXQ-848_2026-08-01 recommendation). The queue entry's own
  `implementation_hint` framed this as "dacc_adapter needs a training pass" --
  a category error: `DACCtoE3Adapter` has no `nn.Parameter` anywhere (pure
  arithmetic over `DACCConfig` float knobs); there is nothing to train. Probed
  before implementing (see below) and found two independent, non-training gaps.
  Modules:
    ree_core/cingulate/dacc.py: DACCConfig gains dacc_goal_readout_normalize
      (bool, False, no-op default).
    ree_core/agent.py: DACCConfig construction site (~line 537) forwards the new
      flag; the L7 candidate_goal_proximity computation site (select_action,
      ~line 6260) applies a per-candidate-set min-max rescale to
      _dacc_goal_prox, gated by the flag, BEFORE it reaches DACCAdaptiveControl
      / DACCtoE3Adapter (which are UNCHANGED -- they already consume
      bundle["goal_readout"] generically). A near-flat candidate set
      (max-min <= 1e-9) is left un-rescaled (no div-by-zero; a flat vector
      carries no discriminative signal regardless of rescaling).
    ree_core/utils/config.py: dacc_goal_readout_normalize added at all three
      REEConfig sites (dataclass field, from_dims kwarg, from_dims body
      assignment) alongside the existing dacc_goal_readout_weight.
  Probe finding (why 2a/normalize, not a trainable head, not an affine
  scale/bias fallback): direct instrumentation of REEAgent.select_action on
  V3-EXQ-848's own driver/config (not a re-run of the 2026-07-31 diagnosis
  session's throwaway, uncommitted probe, which did not reproduce under the
  live driver's settings) found raw goal_proximity values well inside [0,1]
  (0.3-0.8 range) with genuine, non-degenerate per-candidate-set spread
  (~0.003-0.03) -- NOT floor-pinned noise. The spread is small relative to the
  nominal [0,1] range because z_goal's operating norm (measured ~0.08 in an
  active cell; z_goal never becomes active at all in at least one probed cell,
  i.e. reward never clears benefit_threshold for that seed/content) is
  calibrated independently of, and typically much smaller than, the
  e2_world_forward candidate summaries' operating norm (~0.6-1.5, drifting
  upward within an episode) -- so goal_proximity's MSE-sum distance is
  dominated by ||z_world||^2. This is a units/calibration mismatch, confirmed
  as the SAME mechanism MECH-295's compute_approach_cue_score_bias would face
  (it calls the identical goal_state.goal_proximity(_candidate_world_summaries(...))
  pairing at agent.py ~6794/6810) -- not something specific to an "untrained"
  dacc_adapter. A companion, independent wiring gap (Fix 1, no code change):
  V3-EXQ-848's driver never set dacc_goal_readout_weight, so it stayed at its
  0.0 default and the goal-readout term contributed exactly 0.0 in that run,
  on top of the calibration issue -- any successor validation experiment MUST
  set it nonzero (precedent 0.5, V3-EXQ-637).
  Backward compatible: dacc_goal_readout_normalize=False by default (existing
  dacc_goal_readout_weight default 0.0 unchanged) -> bit-identical to today.
  Smoke tests (tensor-exact, throwaway script, not committed): (a) field
  present + defaults False end-to-end REEConfig -> DACCConfig via agent
  construction; (a2) normalize=False bias exactly matches the pre-existing
  unrescaled formula; (b) normalize=True bias exactly matches a hand-computed
  min-max-rescaled reference tensor; (b2) near-flat-vector guard produces no
  NaN/inf; (c) end-to-end REEAgent.select_action run (V3-EXQ-848's seed=1/
  content=A cell) completes without error with the adapter genuinely invoked.
  Local pytest (per this landing's explicit instruction to run on-Mac, not via
  remote_pytest.sh routing): test_flag_inertness.py,
  test_sd_057_phase2_cue_recall_consume.py, test_mech_295_liking_bridge.py,
  test_frozen_z_goal_scaffold_family.py, test_arc065_gapa_candidate_summary_source.py,
  test_inert_salience_dacc_bias_lint.py -- 78/78 PASS.
  Phased training: N/A (no learned parameters anywhere in this data path).
  MECH-094: N/A (waking action-selection control plane only, no replay/simulation
  content written to memory -- same disposition as the L7 parent entry).
  Validation experiment: NOT queued as part of this landing (out of scope for
  this session per the delegating task; a successor to V3-EXQ-848 setting both
  dacc_goal_readout_weight=0.5 and dacc_goal_readout_normalize=True is
  recommended follow-on work via /queue-experiment).
  Design doc: REE_assembly/docs/architecture/sd_057_object_bound_incentive_salience.md
    (L7 AMEND bullet, phase-2 section).
  See SD-057 phase-2 (parent entry above), MECH-348 (L7), MECH-295 (approach
    bridge -- shares the calibration mechanism), ARC-065 GAP-A (the
    _candidate_world_summaries() pattern this reuses), ARC-005 (unblocked
    claim per substrate_queue.json), V3-EXQ-848 (the run that surfaced this).
