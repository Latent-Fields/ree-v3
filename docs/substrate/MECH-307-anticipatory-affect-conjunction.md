## MECH-307 Anticipatory Affect Conjunction Architecture (2026-05-11)
- MECH-307: affect.anticipatory_conjunction_architecture -- SUBSTRATE LANDED 2026-05-11.
  Goal-pipeline GAP-1 / Phase 1 (REE_assembly/evidence/planning/goal_pipeline_plan.md).
  Four-gap substrate amendment to the SD-014 valence vector + MECH-216 schema readout
  + MECH-205 PE write site. Excitement / dread emerge as derived conjunction-states
  from the existing channel set; NOT a new primitive VALENCE channel.
  Gaps 2, 3, 4 substrate-landed 2026-05-08 in the prior MECH-307 session under flags
  use_mech307_schema_multichannel / use_mech307_predicted_location_write. This pass
  resolves Gap 1 via the 2026-05-11 user-override to Option-b (split channels) over
  the design-doc default Option-a (signed single channel). The Option-a path is
  retained behind use_mech307_signed_pe for backward compat; Option-b takes
  precedence when both flags are True.
  Modules:
    ree_core/residue/field.py -- VALENCE_DIM 4 -> 6; new constants
      VALENCE_POSITIVE_SURPRISE=4 and VALENCE_NEGATIVE_SURPRISE=5 added to
      VALENCE_COMPONENTS. valence_vecs buffer auto-resizes via VALENCE_DIM
      (RBFLayer.register_buffer). evaluate_valence return shape grows
      [batch, 4] -> [batch, 6]. Indices 4-5 stay zeroed unless
      use_mech307_split_surprise=True.
    ree_core/utils/config.py -- two new REEConfig fields:
      use_mech307_split_surprise (default False) -- Option-b master.
      use_mech307_conjunction (default False) -- convenience master flag
      that __post_init__ propagates to all three substrate-side sub-flags
      (use_mech307_split_surprise + use_mech307_schema_multichannel +
      use_mech307_predicted_location_write). Path B / consumer-side
      use_mech307_consumer_conjunction_read NOT auto-set (that is a
      downstream wiring decision).
    ree_core/agent.py -- MECH-205 PE write site (line ~3527) dispatches
      between three paths: (1) Option-b split (use_mech307_split_surprise);
      (2) Option-a signed single channel (use_mech307_signed_pe legacy);
      (3) true legacy unsigned magnitude. Option-b routes surprise to
      VALENCE_POSITIVE_SURPRISE or VALENCE_NEGATIVE_SURPRISE based on
      concurrent harm_signal sign; ALSO writes magnitude to legacy
      VALENCE_SURPRISE so MECH-205 / SD-014 consumers reading the
      magnitude slot stay bit-identical to the legacy substrate.
  Data flow (Option-b under master flag ON):
    MECH-205 fires on PE > threshold -> route by harm sign:
      harm_signal <  0 -> update_valence(z_world, VALENCE_NEGATIVE_SURPRISE, ...)
      harm_signal >= 0 -> update_valence(z_world, VALENCE_POSITIVE_SURPRISE, ...)
      Plus magnitude write to VALENCE_SURPRISE (backward compat).
    MECH-216 schema readout (under multichannel flag) -> writes anticipatory
      VALENCE_LIKING + pulses z_beta arousal (Gaps 2 + 3, unchanged from
      2026-05-08 landing).
    MECH-216 write target (under predicted-location flag) -> cached e1_prior
      rather than current z_world (Gap 4, unchanged from 2026-05-08).
  Config: REEConfig.use_mech307_conjunction (bool, default False; bit-identical
    OFF). When True, __post_init__ sets use_mech307_split_surprise =
    use_mech307_schema_multichannel = use_mech307_predicted_location_write =
    True. Sub-flags can still be set independently when finer control is
    needed. Existing gain knobs (mech307_anticipatory_liking_gain,
    mech307_z_beta_schema_gain, conjunction thresholds, conjunction_gain) are
    unchanged from the 2026-05-08 landing.
  Backward compatible: all four flags default False. With defaults:
    - residue field's valence_vecs is shape [num_centers, 6] but indices 4-5
      stay zeroed (extra buffer is ~50% overhead on a small sparse buffer);
    - evaluate_valence returns [batch, 6] but only indices 0-3 carry signal;
    - PE write site falls through to the true-legacy unsigned-magnitude path;
    - all consumers reading VALENCE_SURPRISE (MECH-205 replay-priority, SD-014
      consumers via evaluate_valence) are bit-identical.
  Regression: 309/309 contracts PASS + 7/7 preflight PASS with master OFF
    (verified 2026-05-11). Existing tests/contracts/test_mech307_conjunction_contract.py
    (12 contracts covering Gaps 1-4 under their individual flags) PASSed
    unmodified -- the Option-a Gap-1 path is preserved.
  Direct field-level smoke (2026-05-11): VALENCE_DIM=6 buffer allocation,
    update_valence to VALENCE_POSITIVE_SURPRISE / VALENCE_NEGATIVE_SURPRISE
    accumulates correctly, evaluate_valence returns [1, 6], MECH-094
    hypothesis_tag=True gate respected (write skipped).
  Biological basis: 2026-05-08 lit-pull synthesis (9 entries, lit_conf 0.77)
    on excitement-as-5th-channel; NAcc-anticipation (Knutson 2001a et seq.) +
    dopamine RPE (Bromberg-Martin 2010) + habenula dread (Bromberg-Martin 2010b)
    + Adcock 2006 preplay-priority + Berridge & Robinson 2003 wanting/liking
    dissociation. Conjunction reading is more biologically faithful than a
    new channel: biology has no VALENCE_EXCITEMENT neuron type; excitement
    is the anatomical convergence of DA RPE + hippocampal preplay + ANS
    arousal at NAcc. SD-014 6-channel amendment retained as registered
    fallback if the conjunction reading fails behavioural validation.
  MECH-094: split-channel write inherits the hypothesis_tag=False gate from
    the MECH-205 / SD-014 update_valence call site; replay / simulation
    paths cannot write to either new channel.
  Phased training: not applicable (substrate is buffer-level + write-site
    routing; no learned parameters; no gradient flow).
  Validation experiment: 4-arm discriminative-pair to be queued via
    /queue-experiment in a separate session per the user 2026-05-11 directive
    "substrate only first". Acceptance criteria per anticipatory_affect_conjunction_vs_dual_channel.md
    Validation Experiment section: all-four-gaps-fixed arm produces non-zero
    cue_fires + dacc_bias + approach_commit relative to baseline; any single-
    gap-lesioned arm collapses to baseline (the conjunction-architecture
    falsifier). Fallback (per design doc): SD-014 6-channel amendment if the
    conjunction-fix does not produce the expected derived states.
  Plan-of-record: REE_assembly/evidence/planning/goal_pipeline_plan.md
    GAP-1 status open -> done 2026-05-11; Phase 2 (GAP-2 SD-049 Phase 2
    behavioural validation under MECH-307-fixed substrate) unblocked.
  Design doc: REE_assembly/docs/architecture/anticipatory_affect_conjunction_vs_dual_channel.md
  See MECH-307 (this claim), SD-014 (4-component valence vector substrate
    being extended; 6-channel amendment retained as registered fallback),
    MECH-216 (E1 predictive wanting / schema readout; Gap 2 + 4 consumer),
    MECH-205 (surprise-gated replay write path; Gap 1 write site),
    MECH-093 (z_beta modulates E3 heartbeat rate; Gap 3 downstream consumer),
    MECH-111 (curiosity / novelty drive; likely upstream-blocked by Gap 1),
    MECH-292 (ranked ghost-goal bank; downstream conjunction consumer),
    MECH-094 (hypothesis_tag gate; preserved through call-site scoping),
    MECH-295 (drive-liking-approach bridge; Path B consumer-read target;
      out-of-scope this session per "substrate only first" directive).
